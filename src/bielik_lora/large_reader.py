"""Agentic reading of large session files with a formal handoff (read_large_file_session).

The reader is a tool-calling agent with its own navigation tools (cursor reads, search, skip, jump, a one-time
multi-head reconnaissance), saves formal notes to the sandbox and ends with a synthesis that becomes the handoff.
"""

from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator

from bielik_lora.agent import CONTEXT_LIMIT_CHARS, ToolError, complete_json, run_agent, schema_errors
from bielik_lora.prompts import orchestrator_prompts, reader_prompts
from bielik_lora.sandbox_client import SandboxClient

DASH = "\u2014"
MAX_READ_LINES = 2000
MULTI_HEAD_MIN_LINES = 2000
MAX_HEADS = 16
HEAD_WORKERS = 8
READER_MAX_STEPS = 80
LANG = "polski"
# Room left in the context window for prompts, notes and answers.
PROMPT_RESERVE_CHARS = 20000
REFS = {"type": "array", "items": {"type": "string"}}
INT = {"type": "integer"}

READ_LARGE_FILE_TOOL: dict[str, Any] = {
    "name": "read_large_file_session",
    "description": (
        "Celowany odczyt du\u017cego pliku tekstowego (>50 KB, tak\u017ce dokumentu po konwersji) przez agenta-czytelnika: "
        "sam nawiguje (czytanie od kursora, wyszukiwanie, pomijanie, skoki, jednorazowy rekonesans wieloma g\u0142owicami), "
        "zapisuje formalne notatki z faktami [F] weryfikowanymi cytatem i liniami oraz interpretacjami [I], a na ko\u0144cu "
        "buduje handoff (status COMPLETE/PARTIAL, pokrycie, mapa struktury, ustalenia, czego nie zak\u0142ada\u0107) w notes/reads/<id>.md."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "goal": {"type": "string", "description": "Po co czytamy: konkretny cel odczytu (przy enumeracji napisz \u201ewypisz ka\u017cdy element osobno\u201d)."},
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Pytania do rozstrzygni\u0119cia; bez nich jedynym pytaniem jest cel.",
            },
        },
        "required": ["path", "goal"],
        "additionalProperties": False,
    },
    "returns": {
        "type": "object",
        "required": ["read_id", "handoff", "status", "source", "questions", "synthesis", "facts", "telemetry"],
        "properties": {
            "read_id": {"type": "string"},
            "handoff": {"type": "string"},
            "status": {"type": "string", "enum": ["COMPLETE", "PARTIAL"]},
            "source": {"type": "string"},
            "questions": {"type": "array", "items": {"type": "object"}},
            "synthesis": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "handoff": {"type": "string"},
                    "resolved": {"type": "array"},
                    "unresolved": {"type": "array"},
                    "interpretations": {"type": "array"},
                    "contradictions": {"type": "array"},
                },
            },
            "facts": {"type": "array", "items": {"type": "object"}},
            "telemetry": {"type": "object"},
        },
    },
}

FACTS = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["id", "text", "start_line"],
        "properties": {"id": {"type": "string"}, "text": {"type": "string"}, "start_line": INT, "end_line": INT, "questions": REFS},
    },
}
INTERPRETATIONS = {
    "type": "array",
    "items": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string"}, "based_on": REFS}},
}
RESOLVED = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["question", "answer"],
        "properties": {"question": {"type": "string"}, "answer": {"type": "string"}, "based_on": REFS},
    },
}
OPEN = {
    "type": "array",
    "items": {"type": "object", "required": ["issue"], "properties": {"question": {"type": ["string", "null"]}, "issue": {"type": "string"}}},
}
SECTIONS = {
    "type": "array",
    "items": {"type": "object", "required": ["title", "start_line"], "properties": {"title": {"type": "string"}, "start_line": INT, "end_line": INT}},
}
NOTE_FIELDS = {
    "facts": FACTS,
    "interpretations": INTERPRETATIONS,
    "resolved": RESOLVED,
    "open": OPEN,
    "sections": SECTIONS,
    "text_type": {"type": "string"},
}

HEAD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["note", "facts"],
    "properties": {"note": {"type": "string"}, **NOTE_FIELDS},
}

SYNTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["status", "summary", "handoff", "resolved", "unresolved", "interpretations", "contradictions"],
    "properties": {
        "status": {"type": "string", "enum": ["COMPLETE", "PARTIAL"]},
        "text_type": {"type": "string"},
        "summary": {"type": "string"},
        "handoff": {"type": "string"},
        "resolved": RESOLVED,
        "unresolved": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["issue"],
                "properties": {
                    "question": {"type": ["string", "null"]},
                    "issue": {"type": "string"},
                    "reason": {"type": "string"},
                    "next_step": {"type": "string"},
                },
            },
        },
        "interpretations": INTERPRETATIONS,
        "contradictions": INTERPRETATIONS,
    },
}

READER_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_lines",
        "description": "Czyta count linii od kursora (numerowane \u201eNNNNNN| \u201d) i przesuwa kursor za przeczytany zakres.",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1, "maximum": MAX_READ_LINES}},
            "required": ["count"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["start_line", "end_line", "cursor", "total_lines", "content"],
            "properties": {"start_line": INT, "end_line": INT, "cursor": INT, "total_lines": INT, "content": {"type": "string"}, "coverage_pct": {"type": "number"}},
        },
    },
    {
        "name": "search_lines",
        "description": "Przeszukuje ca\u0142y plik bez przesuwania kursora: \u0142\u0105czna liczba pasuj\u0105cych linii i ograniczone trafienia z kontekstem.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "regex": {"type": "boolean"},
                "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["total_matches", "returned", "matches"],
            "properties": {"total_matches": INT, "returned": INT, "truncated": {"type": "boolean"}, "matches": {"type": "array"}},
        },
    },
    {
        "name": "skip_lines",
        "description": "Przesuwa kursor o count linii bez czytania (pomini\u0119ty zakres zostaje odnotowany w nawigacji).",
        "parameters": {
            "type": "object",
            "properties": {"count": {"type": "integer", "minimum": 1}},
            "required": ["count"],
            "additionalProperties": False,
        },
        "returns": {"type": "object", "required": ["cursor", "total_lines"], "properties": {"cursor": INT, "total_lines": INT}},
    },
    {
        "name": "jump_to_line",
        "description": "Ustawia kursor na podanej linii (pr\u00f3bkowanie \u015brodka/ko\u0144ca, doczytywanie trafie\u0144).",
        "parameters": {
            "type": "object",
            "properties": {"line": {"type": "integer", "minimum": 1}},
            "required": ["line"],
            "additionalProperties": False,
        },
        "returns": {"type": "object", "required": ["cursor", "total_lines"], "properties": {"cursor": INT, "total_lines": INT}},
    },
    {
        "name": "run_multi_head_reconnaissance",
        "description": (
            f"JEDNORAZOWO (plik \u2265 {MULTI_HEAD_MIN_LINES} linii): dzieli ca\u0142y plik na zakresy po target_lines_per_head linii "
            f"(maks. {MAX_HEADS} g\u0142owic), ka\u017cda g\u0142owica niezale\u017cnie czyta sw\u00f3j zakres i zapisuje notatk\u0119 z faktami."
        ),
        "parameters": {
            "type": "object",
            "properties": {"target_lines_per_head": {"type": "integer", "minimum": 100, "maximum": 20000}},
            "required": ["target_lines_per_head"],
            "additionalProperties": False,
        },
        "returns": {"type": "object", "required": ["heads"], "properties": {"heads": {"type": "array"}, "coverage_pct": {"type": "number"}}},
    },
    {
        "name": "save_notes",
        "description": (
            "Zapisuje notatk\u0119 (formalny handoff cz\u0105stkowy, Markdown w note) oraz pola strukturalne. Fakty tylko z przeczytanych linii \u2014 "
            "sandbox odrzuca inne i do\u0142\u0105cza cytat ze \u017ar\u00f3d\u0142a. Id fakt\u00f3w lokalne (N1\u2026), w wyniku dostajesz trwa\u0142e (R\u2026.F\u2026)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"note": {"type": "string"}, **NOTE_FIELDS},
            "required": ["note"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["note", "facts", "rejected_facts"],
            "properties": {"note": INT, "facts": {"type": "array"}, "interpretations": REFS, "rejected_facts": {"type": "array"}, "unknown_refs": REFS},
        },
    },
    {
        "name": "finish",
        "description": "Ko\u0144czy czytanie. Wcze\u015bniej zapisz notatk\u0119 ko\u0144cow\u0105 (pokrycie, g\u0142\u00f3wne ustalenia, luki).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "returns": {"type": "object", "required": ["finished"], "properties": {"finished": {"type": "boolean"}, "notes": INT}},
    },
]


def fill(template: str, values: dict[str, Any]) -> str:
    for key, value in values.items():
        template = template.replace("{" + key + "}", str(value))
    return template


def checked_completion(
    provider: str, model: str, system: str, prompt: str, schema: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """LLM JSON answer validated against the reader contract; one corrective retry."""
    answer, usage = complete_json(provider, model, system, prompt)
    problems = schema_errors(schema, answer)
    if problems:
        retry_prompt = f"{prompt}\n\nTwoja poprzednia odpowied\u017a by\u0142a niezgodna z kontraktem: {problems}. Zwr\u00f3\u0107 poprawiony JSON."
        answer, retry = complete_json(provider, model, system, retry_prompt)
        usage = {
            **retry,
            "llm_ms": usage["llm_ms"] + retry["llm_ms"],
            "input_tokens": (usage["input_tokens"] or 0) + (retry["input_tokens"] or 0),
            "output_tokens": (usage["output_tokens"] or 0) + (retry["output_tokens"] or 0),
            "calls": 2,
        }
        problems = schema_errors(schema, answer)
        if problems:
            raise ToolError("invalid_model_output", "Reader zwr\u00f3ci\u0142 odpowied\u017a niezgodn\u0105 z kontraktem.", problems)
    return answer, usage


class Reader:
    """State of one read: cursor, context budget, LLM usage and the bridge to the sandbox record."""

    def __init__(self, sandbox: SandboxClient, session_id: str, provider: str, model: str, started: dict[str, Any], goal: str) -> None:
        self.sandbox, self.session_id, self.provider, self.model = sandbox, session_id, provider, model
        self.read_id = started["read_id"]
        self.path = started["path"]
        self.file = started["file"]
        self.total_lines = int(started["file"]["total_lines"])
        self.questions = started["questions"]
        self.goal = goal
        self.cursor = 1
        self.delivered_chars = 0
        self.heads_used = False
        self.finished = False
        self.notes = 0
        self.facts: list[dict[str, Any]] = []
        self.usage = {"llm_calls": 0, "llm_ms": 0, "input_tokens": 0, "output_tokens": 0, "model": model}
        self.prompts = reader_prompts()
        self.values = {
            "filename": self.path,
            "totalSize": self.file.get("chars", DASH),
            "totalLines": self.total_lines,
            "analysisGoal": goal,
            "questions": "\n".join(f"- {item['id']}: {item['text']}" for item in self.questions),
            "lang": LANG,
        }

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.sandbox.call(self.session_id, name, arguments)

    def log(self, event: dict[str, Any]) -> dict[str, Any]:
        return self.call("large_read_log", {"read_id": self.read_id, "event": event})

    def add_usage(self, usage: dict[str, Any]) -> None:
        self.usage["llm_calls"] += int(usage.get("calls") or 1)
        for key in ("llm_ms", "input_tokens", "output_tokens"):
            self.usage[key] += int(usage.get(key) or 0)

    def budget(self) -> int:
        left = CONTEXT_LIMIT_CHARS - PROMPT_RESERVE_CHARS - self.delivered_chars
        if left <= 0:
            raise ToolError("context_exhausted", "Wyczerpany limit kontekstu czytelnika: zapisz notatk\u0119 ko\u0144cow\u0105 (save_notes) i wywo\u0142aj finish.")
        return left

    def save(self, origin: str, answer: dict[str, Any], telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
        noted = self.call(
            "large_read_note",
            {
                "read_id": self.read_id,
                "text": answer.get("note", ""),
                "origin": origin,
                "telemetry": telemetry or {},
                **{key: answer[key] for key in NOTE_FIELDS if key in answer},
            },
        )
        mapping = {item["local_id"]: item["id"] for item in noted["facts"]}
        self.facts += [
            {"id": mapping[fact["id"]], "text": fact["text"], "start_line": fact["start_line"], "end_line": fact.get("end_line") or fact["start_line"]}
            for fact in answer.get("facts") or []
            if fact.get("id") in mapping
        ]
        self.notes += 1
        return noted

    # ---- reader tools ----------------------------------------------------------------------
    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.finished and name != "finish":
            raise ToolError("finished", "Czytanie jest zako\u0144czone (finish); odpowiedz kr\u00f3tko bez narz\u0119dzi.")
        result = getattr(self, f"tool_{name}")(**arguments)
        return self.with_progress(result) if inspect.isgenerator(result) else (result, None)

    @staticmethod
    def with_progress(steps: Generator[dict[str, Any], None, dict[str, Any]]) -> Generator[dict[str, Any], None, tuple[dict[str, Any], None]]:
        data = yield from steps
        return data, None

    def tool_read_lines(self, count: int) -> dict[str, Any]:
        if self.cursor > self.total_lines:
            raise ToolError("eof", f"Kursor jest za ko\u0144cem pliku ({self.total_lines} linii); u\u017cyj jump_to_line albo finish.")
        started = time.monotonic()
        last = min(self.total_lines, self.cursor + min(int(count), MAX_READ_LINES) - 1)
        chunk = self.call("read_lines", {"path": self.path, "start_line": self.cursor, "end_line": last, "max_chars": self.budget()})
        logged = self.log(
            {
                "op": "read",
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "chars": len(chunk["content"]),
                "read_ms": round((time.monotonic() - started) * 1000),
                "stopped_at_context_limit": bool(chunk.get("stopped_at_context_limit")),
            }
        )
        self.delivered_chars += len(chunk["content"])
        self.cursor = chunk["end_line"] + 1
        return {
            "start_line": chunk["start_line"],
            "end_line": chunk["end_line"],
            "cursor": self.cursor,
            "total_lines": self.total_lines,
            "coverage_pct": logged["coverage_pct"],
            "content": chunk["content"],
            **({"note": "Zatrzymano na limicie kontekstu."} if chunk.get("stopped_at_context_limit") else {}),
        }

    def tool_search_lines(self, query: str, regex: bool = False, context_lines: int = 2, max_matches: int = 50) -> dict[str, Any]:
        self.budget()
        started = time.monotonic()
        found = self.call(
            "search_lines",
            {"path": self.path, "query": query, "regex": regex, "context_lines": context_lines, "max_matches": max_matches},
        )
        self.log(
            {
                "op": "search",
                "query": query,
                "regex": bool(regex),
                "total_matches": found["total_matches"],
                "returned": found["returned"],
                "read_ms": round((time.monotonic() - started) * 1000),
            }
        )
        self.delivered_chars += sum(len(item["context"]) for item in found["matches"])
        return {key: found[key] for key in ("total_matches", "returned", "truncated", "matches")}

    def tool_skip_lines(self, count: int) -> dict[str, Any]:
        target = min(self.total_lines + 1, self.cursor + int(count))
        self.log({"op": "skip", "from_line": self.cursor, "to_line": target - 1})
        self.cursor = target
        return {"cursor": self.cursor, "total_lines": self.total_lines}

    def tool_jump_to_line(self, line: int) -> dict[str, Any]:
        target = max(1, min(int(line), self.total_lines))
        self.log({"op": "jump", "from_line": self.cursor, "to_line": target})
        self.cursor = target
        return {"cursor": self.cursor, "total_lines": self.total_lines}

    def tool_save_notes(self, note: str, **fields: Any) -> dict[str, Any]:
        noted = self.save("reader", {"note": note, **fields})
        return {key: noted[key] for key in ("note", "facts", "interpretations", "rejected_facts", "unknown_refs")}

    def tool_finish(self) -> dict[str, Any]:
        self.finished = True
        return {"finished": True, "notes": self.notes}

    def tool_run_multi_head_reconnaissance(self, target_lines_per_head: int) -> Generator[dict[str, Any], None, dict[str, Any]]:
        if self.heads_used:
            raise ToolError("already_used", "Rekonesans wieloma g\u0142owicami mo\u017cna uruchomi\u0107 tylko raz.")
        if self.total_lines < MULTI_HEAD_MIN_LINES:
            raise ToolError("too_small", f"Plik ma {self.total_lines} linii (< {MULTI_HEAD_MIN_LINES}); czytaj sekwencyjnie.")
        self.heads_used = True
        size = max(int(target_lines_per_head), -(-self.total_lines // MAX_HEADS))
        ranges = [(first, min(self.total_lines, first + size - 1)) for first in range(1, self.total_lines + 1, size)]
        per_head = max(1, (CONTEXT_LIMIT_CHARS - PROMPT_RESERVE_CHARS) // 2)
        yield {"type": "progress", "name": READ_LARGE_FILE_TOOL["name"], "message": f"{self.read_id}: rekonesans {len(ranges)} g\u0142owicami po ~{size} linii"}

        def run_head(item: tuple[int, tuple[int, int]]) -> tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], int]:
            number, (first, last) = item
            started = time.monotonic()
            chunk = self.call("read_lines", {"path": self.path, "start_line": first, "end_line": last, "max_chars": per_head})
            read_ms = round((time.monotonic() - started) * 1000)
            values = {**self.values, "range": f"L{chunk['start_line']}\u2013L{chunk['end_line']}"}
            answer, usage = checked_completion(
                self.provider, self.model, fill(self.prompts["head"], values), f"Zakres {values['range']}:\n\n{chunk['content']}", HEAD_SCHEMA
            )
            return number, chunk, answer, usage, read_ms

        with ThreadPoolExecutor(max_workers=HEAD_WORKERS) as pool:
            results = sorted(pool.map(run_head, enumerate(ranges, start=1)), key=lambda result: result[0])
        heads = []
        coverage = 0.0
        for number, chunk, answer, usage, read_ms in results:
            self.add_usage(usage)
            logged = self.log(
                {"op": "head", "head": number, "start_line": chunk["start_line"], "end_line": chunk["end_line"], "chars": len(chunk["content"]), "read_ms": read_ms}
            )
            coverage = logged["coverage_pct"]
            noted = self.save(f"g\u0142owica {number}", answer, usage)
            heads.append(
                {
                    "head": number,
                    "range": f"L{chunk['start_line']}\u2013L{chunk['end_line']}",
                    "note": answer.get("note", ""),
                    "facts": noted["facts"],
                    "rejected_facts": noted["rejected_facts"],
                }
            )
            self.delivered_chars += len(answer.get("note", ""))
            yield {
                "type": "progress",
                "name": READ_LARGE_FILE_TOOL["name"],
                "message": f"{self.read_id} g\u0142owica {number}: L{chunk['start_line']}\u2013L{chunk['end_line']} \u00b7 fakty {len(noted['facts'])}",
            }
        return {"heads": heads, "coverage_pct": coverage}


def progress(message: str) -> dict[str, Any]:
    return {"type": "progress", "name": READ_LARGE_FILE_TOOL["name"], "message": message}


def describe(name: str, arguments: dict[str, Any]) -> str:
    if name == "read_lines":
        return f"czyta {arguments.get('count')} linii"
    if name == "search_lines":
        return f"szuka \u201e{arguments.get('query')}\u201d"
    if name == "skip_lines":
        return f"pomija {arguments.get('count')} linii"
    if name == "jump_to_line":
        return f"skok do L{arguments.get('line')}"
    if name == "save_notes":
        return f"notatka (fakty {len(arguments.get('facts') or [])})"
    if name == "run_multi_head_reconnaissance":
        return f"rekonesans g\u0142owicami po {arguments.get('target_lines_per_head')} linii"
    return name


def synthesis_input(reader: Reader, record: dict[str, Any]) -> str:
    facts = "\n".join(f"- {item['id']} (L{item['start_line']}\u2013L{item['end_line']}): {item['text']}" for item in record["facts"]) or "- (brak)"
    interpretations = "\n".join(
        f"- {item['id']}: {item['text']} (na podstawie: {', '.join(item['based_on']) or DASH})" for item in record["interpretations"]
    ) or "- (brak)"
    sections = sorted({(item["start_line"], item["end_line"], item["title"]) for note in record["notes"] for item in note["sections"]})
    toc = "\n".join(f"- {title} \u2014 L{first}\u2013L{last}" for first, last, title in sections) or "- (brak)"
    notes = "\n\n".join(f"### Notatka {note['index']} ({note['origin']})\n{note['text']}" for note in record["notes"]) or "(brak notatek)"
    resolved = "\n".join(f"- {item['question']}: {item['answer']}" for note in record["notes"] for item in note["resolved"]) or "- (brak)"
    open_issues = "\n".join(f"- {item.get('question') or DASH}: {item['issue']}" for note in record["notes"] for item in note["open"]) or "- (brak)"
    navigation = "\n".join(
        f"- {event['op']}: "
        + (f"L{event['start_line']}\u2013L{event['end_line']}" if event["op"] in {"read", "head"} else
           f"\u201e{event.get('query')}\u201d \u2192 {event.get('total_matches')} trafie\u0144" if event["op"] == "search" else
           f"L{event.get('from_line')} \u2192 L{event.get('to_line')}")
        for event in record["events"]
    ) or "- (brak)"
    unfinished = "" if reader.finished else "\n\nUWAGA: czytelnik nie wywo\u0142a\u0142 finish (limit krok\u00f3w) \u2014 uwzgl\u0119dnij to w statusie i pokryciu."
    return (
        f"Pytania:\n{reader.values['questions']}\n\nNawigacja (kolejno):\n{navigation}\n\nMapa struktury:\n{toc}\n\n"
        f"Notatki:\n\n{notes}\n\nFakty:\n{facts}\n\nInterpretacje:\n{interpretations}\n\n"
        f"Rozstrzygni\u0119cia z notatek:\n{resolved}\n\nOtwarte kwestie z notatek:\n{open_issues}{unfinished}"
    )


def read_large_file_session(
    sandbox: SandboxClient,
    session_id: str,
    provider: str,
    model: str,
    path: str,
    goal: str,
    questions: list[str] | None = None,
) -> Generator[dict[str, Any], None, tuple[dict[str, Any], None]]:
    started = sandbox.call(session_id, "large_read_start", {"path": path, "goal": goal, "questions": questions or []})
    reader = Reader(sandbox, session_id, provider, model, started, goal)
    yield progress(
        f"{reader.read_id}: {reader.path} \u00b7 {started['file']['bytes']} B \u00b7 {reader.total_lines} linii \u00b7 "
        f"{len(reader.questions)} pyta\u0144 \u00b7 czytelnik-agent"
    )
    system = "\n\n".join(
        [fill(reader.prompts["reader"], reader.values), reader.prompts["guidance"], orchestrator_prompts()["tool_envelope"]]
    )
    kickoff = [{"role": "user", "content": f"Rozpocznij czytanie pliku {reader.path} zgodnie z celem. Kursor jest na L1."}]
    for event in run_agent(provider, model, system, kickoff, READER_TOOLS, reader, max_steps=READER_MAX_STEPS):
        if event["type"] == "usage":
            reader.add_usage(event)
        elif event["type"] == "tool_call":
            yield progress(f"{reader.read_id}: {describe(event['name'], event['arguments'])}")
        elif event["type"] == "tool_result" and not event["ok"]:
            yield progress(f"{reader.read_id}: {event['name']} \u2014 {event.get('error')}")
        elif event["type"] == "progress":
            yield event
    record = sandbox.call(session_id, "large_read_record", {"read_id": reader.read_id})
    yield progress(f"{reader.read_id}: synteza z {len(record['notes'])} notatek i {len(record['facts'])} fakt\u00f3w")
    synthesis, usage = checked_completion(
        provider, model, fill(reader.prompts["synthesis"], reader.values), synthesis_input(reader, record), SYNTHESIS_SCHEMA
    )
    reader.add_usage(usage)
    finished = sandbox.call(session_id, "large_read_finish", {"read_id": reader.read_id, "synthesis": synthesis, "telemetry": reader.usage})
    yield progress(
        f"{reader.read_id}: gotowe \u00b7 {synthesis['status']} \u00b7 pokrycie {finished['telemetry']['coverage_pct']}% \u00b7 "
        f"rozstrzygni\u0119te {len(synthesis['resolved'])}, nierozstrzygni\u0119te {len(synthesis['unresolved'])} \u00b7 {finished['handoff']}"
    )
    return {
        "read_id": reader.read_id,
        "handoff": finished["handoff"],
        "status": synthesis["status"],
        "source": reader.path,
        "questions": reader.questions,
        "synthesis": synthesis,
        "facts": reader.facts,
        "telemetry": finished["telemetry"],
    }, None
