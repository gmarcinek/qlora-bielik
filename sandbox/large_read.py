"""Formal handoff of an agentic read of a large text file (notes/reads/<id>.json + .md).

The reader navigates the file itself (read / search / skip / jump / heads); every navigation step is logged here,
notes are saved with facts verified against the source, and the final synthesis becomes the handoff document.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sandbox.workspace import Session

LARGE_FILE_BYTES = 50 * 1024
NAVIGATION_OPS = {"read", "search", "skip", "jump", "head"}


def _error(message: str, status: int = 400, code: str | None = None) -> Exception:
    from sandbox.workspace import SandboxError

    return SandboxError(message, status, code)


def reads_dir(session: "Session"):
    return session.path / "notes" / "reads"


def all_reads(session: "Session") -> list[dict[str, Any]]:
    directory = reads_dir(session)
    if not directory.exists():
        return []
    return sorted(
        (json.loads(path.read_text(encoding="utf-8")) for path in directory.glob("R*.json")),
        key=lambda record: int(record["id"][1:]),
    )


def load(session: "Session", read_id: str) -> dict[str, Any]:
    path = reads_dir(session) / f"{read_id}.json"
    if not read_id.startswith("R") or not read_id[1:].isdigit() or not path.exists():
        raise _error(f"Nie ma odczytu {read_id}.", 404, "not_found")
    return json.loads(path.read_text(encoding="utf-8"))


def save(session: "Session", record: dict[str, Any]) -> None:
    directory = reads_dir(session)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{record['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / f"{record['id']}.md").write_text(render(record), encoding="utf-8")
    session.render_handoff(session.notes())


def require_reading(session: "Session", read_id: str) -> dict[str, Any]:
    session.require_open()
    record = load(session, read_id)
    if record["status"] != "reading":
        raise _error(f"Odczyt {read_id} jest już zakończony.", 409, "conflict")
    return record


def start(session: "Session", path: str, goal: str, questions: list[str] | None = None) -> dict[str, Any]:
    session.require_open()
    if not goal.strip():
        raise _error("Odczyt celowany wymaga celu (goal).", code="invalid_arguments")
    text_path, encoding, extra = session.readable(path)
    total_lines = chars = 0
    for total_lines, line in enumerate(session.iter_lines(text_path, encoding), start=1):
        chars += len(line) + 1
    read_id = f"R{len(all_reads(session)) + 1}"
    record = {
        "id": read_id,
        "source": session.display(session.existing_file(path)),
        "path": session.display(text_path),
        "goal": goal.strip(),
        "questions": [
            {"id": f"Q{index}", "text": text.strip()}
            for index, text in enumerate([item for item in questions or [] if item.strip()] or [goal], start=1)
        ],
        "file": {"bytes": text_path.stat().st_size, "total_lines": total_lines, "chars": chars, **extra},
        "status": "reading",
        "started_at": time.time(),
        "events": [],
        "notes": [],
        "facts": [],
        "interpretations": [],
        "synthesis": None,
        "telemetry": None,
    }
    save(session, record)
    return {
        "read_id": read_id,
        "path": record["path"],
        "file": record["file"],
        "questions": record["questions"],
        "handoff": f"notes/reads/{read_id}.md",
    }


def merged_ranges(record: dict[str, Any]) -> list[tuple[int, int]]:
    """Line ranges actually read (sequential reads and reconnaissance heads), merged."""
    spans = sorted(
        (int(event["start_line"]), int(event["end_line"]))
        for event in record["events"]
        if event["op"] in {"read", "head"} and event.get("end_line", 0) >= event.get("start_line", 1)
    )
    merged: list[tuple[int, int]] = []
    for first, last in spans:
        if merged and first <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], last))
        else:
            merged.append((first, last))
    return merged


def lines_read(record: dict[str, Any]) -> int:
    return sum(last - first + 1 for first, last in merged_ranges(record))


def log(session: "Session", read_id: str, event: dict[str, Any]) -> dict[str, Any]:
    record = require_reading(session, read_id)
    if event.get("op") not in NAVIGATION_OPS:
        raise _error(f"op musi być jednym z: {', '.join(sorted(NAVIGATION_OPS))}.")
    record["events"].append({"index": len(record["events"]) + 1, "at": time.time(), **event})
    save(session, record)
    total = record["file"]["total_lines"]
    return {"event": len(record["events"]), "lines_read": lines_read(record), "coverage_pct": round(100 * lines_read(record) / total, 2) if total else 100.0}


def _remap(refs: list[str], mapping: dict[str, str], known: set[str], unknown: list[str]) -> list[str]:
    result = []
    for ref in refs or []:
        target = mapping.get(ref, ref)
        if target in known:
            result.append(target)
        else:
            unknown.append(ref)
    return result


def note(
    session: "Session",
    read_id: str,
    text: str = "",
    facts: list[dict[str, Any]] | None = None,
    interpretations: list[dict[str, Any]] | None = None,
    resolved: list[dict[str, Any]] | None = None,
    open: list[dict[str, Any]] | None = None,  # noqa: A002 - name mirrors the reader contract
    sections: list[dict[str, Any]] | None = None,
    text_type: str = "",
    origin: str = "reader",
    telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Formal handoff note; facts must lie in lines that were actually read and get their quote from the source."""
    record = require_reading(session, read_id)
    total = record["file"]["total_lines"]
    covered = merged_ranges(record)
    index = len(record["notes"]) + 1
    mapping: dict[str, str] = {}
    rejected: list[dict[str, Any]] = []
    unknown: list[str] = []
    new_facts: list[str] = []
    question_ids = {item["id"] for item in record["questions"]}
    for fact in facts or []:
        start_line, end_line = int(fact.get("start_line", 0)), int(fact.get("end_line") or fact.get("start_line", 0))
        if not (1 <= start_line <= end_line <= total):
            rejected.append({"local_id": fact.get("id"), "reason": f"linie {start_line}-{end_line} poza plikiem (1-{total})"})
            continue
        if not any(first <= start_line and end_line <= last for first, last in covered):
            rejected.append({"local_id": fact.get("id"), "reason": f"linie {start_line}-{end_line} nie zostały przeczytane"})
            continue
        if not str(fact.get("text", "")).strip():
            rejected.append({"local_id": fact.get("id"), "reason": "pusty fakt"})
            continue
        fact_id = f"{read_id}.F{len(record['facts']) + 1}"
        if fact.get("id"):
            mapping[str(fact["id"])] = fact_id
        record["facts"].append(
            {
                "id": fact_id,
                "note": index,
                "text": str(fact["text"]).strip(),
                "start_line": start_line,
                "end_line": end_line,
                "quote": session.collect_lines(record["path"], start_line, end_line, numbered=False)["content"],
                "questions": [item for item in fact.get("questions") or [] if item in question_ids],
            }
        )
        new_facts.append(fact_id)
    known = {item["id"] for item in record["facts"]} | {item["id"] for item in record["interpretations"]}
    new_interpretations: list[str] = []
    for interpretation in interpretations or []:
        if not str(interpretation.get("text", "")).strip():
            continue
        interpretation_id = f"{read_id}.I{len(record['interpretations']) + 1}"
        record["interpretations"].append(
            {
                "id": interpretation_id,
                "note": index,
                "text": str(interpretation["text"]).strip(),
                "based_on": _remap(interpretation.get("based_on", []), mapping, known, unknown),
            }
        )
        known.add(interpretation_id)
        new_interpretations.append(interpretation_id)
    new_sections = [
        {
            "title": str(item.get("title", "")).strip(),
            "start_line": int(item["start_line"]),
            "end_line": int(item.get("end_line") or item["start_line"]),
        }
        for item in sections or []
        if str(item.get("title", "")).strip() and 1 <= int(item.get("start_line", 0)) <= total
    ]
    if not (str(text).strip() or new_facts or new_interpretations or new_sections):
        raise _error("Pusta notatka: podaj tekst notatki albo fakty, interpretacje lub sekcje.", code="invalid_arguments")
    record["notes"].append(
        {
            "index": index,
            "origin": origin,
            "at": time.time(),
            "text": str(text).strip(),
            "text_type": str(text_type or ""),
            "sections": new_sections,
            "facts": new_facts,
            "interpretations": new_interpretations,
            "resolved": [
                {
                    "question": item.get("question"),
                    "answer": str(item.get("answer", "")),
                    "based_on": _remap(item.get("based_on", []), mapping, known, unknown),
                }
                for item in resolved or []
            ],
            "open": [{"question": item.get("question"), "issue": str(item.get("issue", ""))} for item in open or []],
            "rejected_facts": rejected,
            "unknown_refs": unknown,
            "telemetry": telemetry or {},
        }
    )
    save(session, record)
    return {
        "note": index,
        "facts": [{"local_id": local, "id": target} for local, target in mapping.items()],
        "interpretations": new_interpretations,
        "sections": new_sections,
        "rejected_facts": rejected,
        "unknown_refs": unknown,
    }


def finish(session: "Session", read_id: str, synthesis: dict[str, Any], telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
    session.require_open()
    record = load(session, read_id)
    known = {item["id"] for item in record["facts"]} | {item["id"] for item in record["interpretations"]}
    unknown: list[str] = []
    interpretations = []
    for index, item in enumerate(synthesis.get("interpretations") or [], start=1):
        interpretations.append(
            {"id": f"{read_id}.S{index}", "text": str(item.get("text", "")), "based_on": _remap(item.get("based_on", []), {}, known, unknown)}
        )
    status = synthesis.get("status")
    record["synthesis"] = {
        "status": status if status in {"COMPLETE", "PARTIAL"} else ("PARTIAL" if synthesis.get("unresolved") else "COMPLETE"),
        "text_type": str(synthesis.get("text_type") or ""),
        "summary": str(synthesis.get("summary", "")),
        "handoff": str(synthesis.get("handoff", "")),
        "resolved": [
            {"question": item.get("question"), "answer": str(item.get("answer", "")), "based_on": _remap(item.get("based_on", []), {}, known, unknown)}
            for item in synthesis.get("resolved") or []
        ],
        "unresolved": [
            {
                "question": item.get("question"),
                "issue": str(item.get("issue", "")),
                "reason": str(item.get("reason", "")),
                "next_step": str(item.get("next_step", "")),
            }
            for item in synthesis.get("unresolved") or []
        ],
        "interpretations": interpretations,
        "contradictions": [
            {"text": str(item.get("text", "")), "based_on": _remap(item.get("based_on", []), {}, known, unknown)}
            for item in synthesis.get("contradictions") or []
        ],
        "unknown_refs": unknown,
    }
    usage = telemetry or {}
    events = record["events"]
    total_lines = record["file"]["total_lines"]
    read = lines_read(record)
    record["telemetry"] = {
        "notes": len(record["notes"]),
        "navigation_steps": len(events),
        "reads": sum(1 for event in events if event["op"] == "read"),
        "searches": sum(1 for event in events if event["op"] == "search"),
        "skips": sum(1 for event in events if event["op"] == "skip"),
        "jumps": sum(1 for event in events if event["op"] == "jump"),
        "heads": sum(1 for event in events if event["op"] == "head"),
        "lines_read": read,
        "total_lines": total_lines,
        "coverage_pct": round(100 * read / total_lines, 2) if total_lines else 100.0,
        "ranges_read": [f"L{first}–L{last}" for first, last in merged_ranges(record)],
        "chars_read": sum(int(event.get("chars") or 0) for event in events if event["op"] in {"read", "head"}),
        "facts": len(record["facts"]),
        "interpretations": len(record["interpretations"]) + len(interpretations),
        "rejected_facts": sum(len(item.get("rejected_facts") or []) for item in record["notes"]),
        "read_ms": sum(int(event.get("read_ms") or 0) for event in events),
        "llm_calls": usage.get("llm_calls"),
        "llm_ms": usage.get("llm_ms"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "duration_s": round(time.time() - record["started_at"], 1),
        "model": usage.get("model"),
    }
    record["status"] = "done"
    record["finished_at"] = time.time()
    save(session, record)
    return {"read_id": read_id, "handoff": f"notes/reads/{read_id}.md", "telemetry": record["telemetry"], "unknown_refs": unknown}


def _number(value: Any) -> str:
    return f"{value:,}".replace(",", " ") if isinstance(value, int) else "—" if value is None else str(value)


def _basis(refs: list[str]) -> str:
    return f" _(na podstawie: {', '.join(refs)})_" if refs else ""


def _question(record: dict[str, Any], question_id: Any) -> str:
    text = next((item["text"] for item in record["questions"] if item["id"] == question_id), None)
    return f"**{question_id}** ({text})" if text else "**—**"


def _event(event: dict[str, Any]) -> str:
    op = event["op"]
    if op == "read":
        detail = f"L{event['start_line']}–L{event['end_line']} · {_number(event.get('chars'))} znaków" + (" · limit kontekstu" if event.get("stopped_at_context_limit") else "")
    elif op == "head":
        detail = f"głowica {event.get('head')}: L{event['start_line']}–L{event['end_line']} · {_number(event.get('chars'))} znaków"
    elif op == "search":
        detail = f"„{event.get('query')}”{' (regex)' if event.get('regex') else ''} · trafienia {_number(event.get('total_matches'))}"
    elif op == "skip":
        detail = f"pominięto L{event.get('from_line')}–L{event.get('to_line')}"
    else:
        detail = f"skok z L{event.get('from_line')} do L{event.get('to_line')}"
    return f"{event['index']}. **{op}** {detail}"


def render(record: dict[str, Any]) -> str:
    started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record["started_at"]))
    finished = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record["finished_at"])) if record.get("finished_at") else "—"
    file = record["file"]
    synthesis = record.get("synthesis")
    out = [
        f"# Handoff odczytu {record['id']}: {record['source']}",
        "",
        "| | |",
        "|---|---|",
        f"| Cel | {record['goal']} |",
        f"| Plik czytany | `{record['path']}`" + (f" (źródło `{record['source']}`)" if record["source"] != record["path"] else "") + " |",
        f"| Rozmiar | {_number(file['bytes'])} B · {_number(file['total_lines'])} linii · {_number(file['chars'])} znaków |",
        f"| Status | {synthesis['status'] if synthesis else 'w toku'} |",
        f"| Przeczytane zakresy | {', '.join(f'L{first}–L{last}' for first, last in merged_ranges(record)) or '—'} |",
        f"| Start / koniec | {started} / {finished} |",
        "",
        "Oznaczenia: **[F]** fakt ze źródła (cytat i linie weryfikowane przez sandbox), **[I]** interpretacja czytelnika, **[S]** interpretacja z syntezy.",
        "",
        "## Pytania",
        "",
        *[f"- **{item['id']}** {item['text']}" for item in record["questions"]],
    ]
    if synthesis:
        text_type = f" · typ tekstu: {synthesis['text_type']}" if synthesis.get("text_type") else ""
        out += ["", "## Handoff", "", f"**Status:** {synthesis['status']}{text_type}", "", synthesis["summary"] or "—"]
        if synthesis.get("handoff"):
            out += ["", synthesis["handoff"]]
        out += ["", "### Rozstrzygnięte pytania", ""]
        out += [f"- {_question(record, item['question'])}: {item['answer']}{_basis(item['based_on'])}" for item in synthesis["resolved"]] or ["- —"]
        out += ["", "### Nierozstrzygnięte", ""]
        out += [
            f"- {_question(record, item['question'])}: {item['issue']}"
            + (f" — powód: {item['reason']}" if item["reason"] else "")
            + (f" — dalej: {item['next_step']}" if item["next_step"] else "")
            for item in synthesis["unresolved"]
        ] or ["- —"]
        out += ["", "### Interpretacje całościowe [S]", ""]
        out += [f"- **[{item['id']}]** {item['text']}{_basis(item['based_on'])}" for item in synthesis["interpretations"]] or ["- —"]
        out += ["", "### Sprzeczności i niespójności", ""]
        out += [f"- {item['text']}{_basis(item['based_on'])}" for item in synthesis["contradictions"]] or ["- —"]
    else:
        out += ["", "## Handoff", "", "_Odczyt w toku — synteza powstanie po finish._"]
    toc = [item for item in record["notes"] for item in item.get("sections") or []]
    if toc:
        out += ["", "## Mapa struktury (z notatek)", ""]
        seen: set[tuple[str, int]] = set()
        for item in sorted(toc, key=lambda section: section["start_line"]):
            if (item["title"], item["start_line"]) in seen:
                continue
            seen.add((item["title"], item["start_line"]))
            span = f"L{item['start_line']}" + (f"–L{item['end_line']}" if item["end_line"] != item["start_line"] else "")
            out.append(f"- {item['title']} — {span}")
    facts = {item["id"]: item for item in record["facts"]}
    interpretations = {item["id"]: item for item in record["interpretations"]}
    out += ["", "## Notatki czytelnika (formalne handoffy cząstkowe)"]
    for item in record["notes"]:
        out += ["", f"### Notatka {item['index']} · {item['origin']}" + (f" · {item['text_type']}" if item.get("text_type") else "")]
        if item["text"]:
            out += ["", item["text"]]
        if item["facts"]:
            out += ["", "**Fakty [F]**", ""]
            for fact_id in item["facts"]:
                fact = facts[fact_id]
                answers = f" _(dotyczy: {', '.join(fact['questions'])})_" if fact.get("questions") else ""
                out.append(f"- **[{fact_id}]** {fact['text']} — L{fact['start_line']}–L{fact['end_line']}{answers}")
                out += [f"  > {quoted}" for quoted in fact["quote"].split("\n")]
        if item["interpretations"]:
            out += ["", "**Interpretacje [I]**", ""]
            out += [f"- **[{ref}]** {interpretations[ref]['text']}{_basis(interpretations[ref]['based_on'])}" for ref in item["interpretations"]]
        if item["resolved"]:
            out += ["", "**Rozstrzygnięte**", ""]
            out += [f"- {_question(record, entry['question'])}: {entry['answer']}{_basis(entry['based_on'])}" for entry in item["resolved"]]
        if item["open"]:
            out += ["", "**Otwarte**", ""]
            out += [f"- {_question(record, entry['question'])}: {entry['issue']}" for entry in item["open"]]
        if item.get("rejected_facts"):
            out += ["", "**Odrzucone fakty**", ""]
            out += [f"- {entry['local_id']}: {entry['reason']}" for entry in item["rejected_facts"]]
    out += ["", "## Nawigacja", ""]
    out += [_event(event) for event in record["events"]] or ["- —"]
    telemetry = record.get("telemetry")
    if telemetry:
        out += ["", "## Telemetria", "", "| metryka | wartość |", "|---|---|"]
        labels = {
            "notes": "notatki",
            "navigation_steps": "kroki nawigacji",
            "reads": "odczyty",
            "searches": "wyszukiwania",
            "skips": "pominięcia",
            "jumps": "skoki",
            "heads": "głowice rekonesansu",
            "lines_read": "przeczytane linie",
            "total_lines": "linie pliku",
            "coverage_pct": "pokrycie %",
            "chars_read": "przeczytane znaki",
            "facts": "fakty",
            "interpretations": "interpretacje",
            "rejected_facts": "odrzucone fakty",
            "read_ms": "czas odczytu [ms]",
            "llm_calls": "wywołania LLM",
            "llm_ms": "czas LLM [ms]",
            "input_tokens": "tokeny wejściowe",
            "output_tokens": "tokeny wyjściowe",
            "duration_s": "czas całkowity [s]",
            "model": "model",
        }
        out += [f"| {label} | {_number(telemetry.get(key))} |" for key, label in labels.items()]
    return "\n".join(out) + "\n"
