import json

import pytest

from bielik_lora.agent import ToolError, run_tool, schema_errors
from bielik_lora.corpus_agent import CorpusAgentTools
from bielik_lora.generation import (
    ANALYST_GENERATE_TOOL,
    ANALYZE_SERIES_TOOL,
    GENERATE_EXAMPLES_TOOL,
    SAVE_EXAMPLES_TOOL,
    GeneralTools,
    analyze_series_session,
    content_of,
    generate_examples_session,
    near_duplicates,
    normalize_assignments,
    template_lines,
)
from bielik_lora.orchestration import PLAN_TOOL, plan_event, record_series, render_state, update_plan

SYSTEM = "Wyodrębnij encje i zwróć JSON."
INSTRUCTION = "Wypisz numery identyfikacyjne z tekstu."
CONTEXT = {"corpus": "test", "agent_prompt": "", "vocabulary": ""}
SOURCE_TEXT = "§ 3. Sprzedawca: NIP 777-88-99-000."


def row(index, fragment, entities, flag="positive"):
    return {
        "id": str(index),
        "split": "train",
        "flag": flag,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{INSTRUCTION}\n{fragment}"},
            {"role": "assistant", "content": json.dumps({"entities": entities, "summary": "x"}, ensure_ascii=False)},
        ],
    }


ROWS = [
    row(1, "Firma ma NIP 525-26-85-595 i działa od lat.", [{"type": "NIP", "text": "525-26-85-595"}]),
    row(2, "W umowie nie podano żadnego numeru.", [], flag="negative"),
    row(3, "REGON spółki to 011417295.", [{"type": "REGON", "text": "011417295"}]),
]


def example(text, flag="positive", rejected=None):
    entities = [{"type": "NIP", "text": text}] if flag == "positive" else []
    item = {"user": f"{INSTRUCTION}\n{text}", "assistant": json.dumps({"entities": entities, "summary": "s"}), "flag": flag}
    return {**item, **({"rejected": rejected} if rejected is not None else {})}


def drain(generator):
    events = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        return events, stop.value


class FakeStore:
    def __init__(self):
        self.saved, self.rejected, self.next_id = [], [], 100

    def save(self, drafts, batch):
        ids = []
        for draft in drafts:
            self.next_id += 1
            ids.append(str(self.next_id))
            self.saved.append({**draft, "id": ids[-1], "batch": batch})
        return {"saved": len(drafts), "batch": batch, "ids": ids}

    def reject(self, ids):
        self.rejected += ids
        return len(ids)


def general_tools(reads):
    def execute(name, arguments):
        reads.append((name, arguments))
        return {"path": arguments["path"], "content": SOURCE_TEXT}, None

    tools = [{"name": "read_lines", "description": "", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
    return GeneralTools(tools, execute)


def call(execute, tools, name, arguments):
    """Runs one tool like the agent loop does and returns its parsed envelope."""
    events, output = drain(run_tool(execute, tools, name, arguments))
    yield from events
    return json.loads(output)


def fake_runner(briefs):
    def runner(provider, model, system, messages, tools, execute, max_steps):
        brief = messages[0]["content"]
        names = {tool["name"] for tool in tools}
        briefs.append(brief)
        if "save_examples" in names:
            assert "remove_proposals" not in names and "{corpus}" not in system
            assert INSTRUCTION not in brief.split("JUŻ ISTNIEJĄ")[1], "shared instruction lines are cut from the digest"
            if "zakres: zły" in brief:
                raise RuntimeError("HTTP 500")
            if "zakres: NIP z umowy" in brief:
                source = yield from call(execute, tools, "read_lines", {"path": "converted/umowa.md"})
                assert source["data"]["content"] == SOURCE_TEXT
                first = yield from call(
                    execute,
                    tools,
                    "save_examples",
                    {
                        "examples": [
                            example("§ 3. Sprzedawca: NIP 777-88-99-000.", rejected='{"entities": [], "summary": "s"}'),
                            example("NIP 123-456-32-18 widnieje w KRS."),
                            {"user": "", "assistant": "{}", "flag": "positive", "rejected": "x"},
                        ]
                    },
                )
                reasons = [item["reason"] for item in first["data"]["rejected"]]
                assert first["data"]["saved"] == 1 and first["data"]["remaining"] == 2
                assert reasons[0].startswith("Tryb DPO: brak") and reasons[1].startswith("Puste pole user")
                yield from call(
                    execute,
                    tools,
                    "save_examples",
                    {"examples": [example("Na fakturze NIP 222-333-44-55.", rejected="{}"), example("Kontrakt: NIP 333-111-22-11.", rejected="{}")]},
                )
                done = yield from call(execute, tools, "save_examples", {"examples": [example("Nadmiarowy NIP 999.", rejected="{}")]})
                assert done["ok"] is False and done["error"]["code"] == "target_reached"
            elif "zakres: Zastąp odrzucone" in brief:
                yield from call(execute, tools, "save_examples", {"examples": [example("NIP 111-222-33-44 w stopce.", rejected="{}")]})
            elif "zakres: balans" in brief:
                yield from call(execute, tools, "save_examples", {"examples": [example("Pismo bez numerów.", "negative", rejected='{"entities": [{"type": "NIP"}]}')]})
            else:
                result = yield from call(execute, tools, "save_examples", {"examples": [example("Firma ma NIP 525-26-85-595 i działa od lat.")]})
                assert result["data"]["rejected"][0]["reason"].startswith("Duplikat")
            yield {"type": "text", "content": "Gotowe."}
            return
        assert {"remove_proposals", "regenerate_proposals", "generate_examples", "read_lines"} <= names and "save_examples" not in names
        ids = [line.split('"id": "')[1].split('"')[0] for line in brief.splitlines() if line.startswith('{"id": ')]
        assert all('"rejected"' in line for line in brief.splitlines() if line.startswith('{"id": '))
        yield from call(execute, tools, "remove_proposals", {"items": [{"id": ids[0], "reason": "duplikat"}, {"id": "obce", "reason": "x"}]})
        regenerated = yield from call(execute, tools, "regenerate_proposals", {"items": [{"id": ids[1], "instruction": "inny fragment"}]})
        assert regenerated["data"]["regenerated"] == 1 and regenerated["data"]["removed"] == 1
        extra = yield from call(execute, tools, "generate_examples", {"assignments": [{"focus": "balans negatywów", "count": 1, "flag": "negative"}]})
        assert extra["data"]["added"] == 1
        yield {"type": "text", "content": "Seria zgodna z celem."}

    return runner


def test_template_lines_are_ignored_in_similarity():
    template = template_lines(ROWS)
    assert INSTRUCTION in template
    assert content_of(ROWS[0]["messages"][1]["content"], template) == "Firma ma NIP 525-26-85-595 i działa od lat."
    pairs = near_duplicates(
        {"new": "Firma ma NIP 525-26-85-595 i działa od wielu lat."},
        {"1": "Firma ma NIP 525-26-85-595 i działa od lat.", "3": "REGON spółki to 011417295."},
        threshold=0.5,
    )
    assert [(a, b) for a, b, _ in pairs] == [("new", "1")]


def test_normalize_assignments_clamps_counts_and_budget():
    kept, dropped = normalize_assignments([{"focus": "a", "count": 99}, {"focus": "b"}, {"focus": "c"}], 40)
    assert [(item["count"], item["flag"]) for item in kept] == [(30, "mixed"), (10, "mixed")]
    assert dropped == [{"focus": "c"}]
    assert SAVE_EXAMPLES_TOOL["parameters"]["properties"]["examples"]["maxItems"] == 30
    assert ANALYST_GENERATE_TOOL["parameters"]["required"] == ["assignments"] and "batch" not in ANALYST_GENERATE_TOOL["parameters"]["properties"]
    assert GENERATE_EXAMPLES_TOOL["parameters"]["required"] == ["goal", "intent", "assignments"]


def test_generators_read_sources_and_save_dpo_pairs_then_analyst_cleans_and_regenerates():
    store, briefs, reads = FakeStore(), [], []
    tools = CorpusAgentTools(ROWS, split_ratio={"train": 80, "validation": 10, "test": 10})
    series = {
        "goal": "Więcej numerów NIP",
        "intent": "Użytkownik chce par DPO uczących nie pomijać NIP-ów w umowach.",
        "context": "Plik umowa.md przeczytany, NIP-y w § 3.",
        "training_mode": "dpo",
        "options": {"system_prompt": "none", "rejected_strategy": "pominięty NIP", "exchanges": 1},
    }
    events, (data, ui_event) = drain(
        generate_examples_session(
            "openai",
            "gpt-test",
            tools,
            {
                **series,
                "sources": [{"path": "converted/umowa.md", "start_line": 1, "end_line": 40}],
                "assignments": [{"focus": "NIP z umowy", "count": 3}, {"focus": "duplikaty", "count": 1}],
            },
            CONTEXT,
            store.save,
            general_tools(reads),
            fake_runner(briefs),
        )
    )
    assert ui_event is None and schema_errors(GENERATE_EXAMPLES_TOOL["returns"], data) == []
    brief = briefs[0]
    for section in ("INTENCJA UŻYTKOWNIKA:", "TRYB TRENINGU: DPO", "KONTEKST:", "OPCJE:", "DANE — ŹRÓDŁA DO PRZECZYTANIA", "converted/umowa.md L1–L40"):
        assert section in brief
    assert reads == [("read_lines", {"path": "converted/umowa.md"})]
    assert data["added"] == 3 and len({item["batch"] for item in store.saved}) == 1
    assert all(item["rejected"] and item["system"] == "" for item in store.saved)
    assert any("duplikaty" in problem and "0 z 1" in problem for problem in data["problems"])
    assert {event["type"] for event in events} <= {"progress", "proposals", "agent_tool"}
    generator_calls = [event for event in events if event["type"] == "agent_tool" and event["phase"] == "call"]
    assert {event["name"] for event in generator_calls} == {"read_lines", "save_examples"}
    assert all(event["agent"].startswith("generator ") and event["scope"] for event in generator_calls)
    assert any(event["detail"].startswith("3 przykładów (positive 3)") for event in generator_calls)
    assert any(event["phase"] == "result" and event["ok"] is False and event["name"] == "save_examples" for event in events if event["type"] == "agent_tool")

    store_before = len(store.saved)
    events, (review, _) = drain(
        analyze_series_session(
            "openai", "gpt-test", tools, {**series, "batch": data["batch"]}, CONTEXT, store.save, store.reject, general_tools(reads), fake_runner(briefs)
        )
    )
    assert schema_errors(ANALYZE_SERIES_TOOL["returns"], review) == []
    assert review["reviewed"] == 3 and review["removed"] == 2 and review["added"] == 2 and review["kept"] == 3
    assert store.rejected == [store.saved[0]["id"], store.saved[1]["id"]] and len(store.saved) == store_before + 2
    assert {item["batch"] for item in store.saved} == {data["batch"]}
    assert review["proportions"]["flags"] == {"positive": "2 (67%)", "negative": "1 (33%)"} and len(review["manifest"]) == 2
    # Generators called by the analyst inherit the series intent, mode and options.
    assert all("INTENCJA UŻYTKOWNIKA:" in item and "TRYB TRENINGU: DPO" in item for item in briefs[-2:])
    assert {"proposals", "proposals_changed", "progress", "agent_tool"} >= {event["type"] for event in events}
    analyst_calls = [event["name"] for event in events if event["type"] == "agent_tool" and event["agent"] == "analityk" and event["phase"] == "call"]
    assert analyst_calls == ["remove_proposals", "regenerate_proposals", "generate_examples"]
    # Generators called by the analyst are logged too.
    assert any(event["type"] == "agent_tool" and event["agent"].startswith("generator ") for event in events)


def test_failed_generator_and_missing_sandbox_are_reported():
    store = FakeStore()
    _, (data, _) = drain(
        generate_examples_session(
            "openai",
            "m",
            CorpusAgentTools(ROWS),
            {"goal": "g", "intent": "i", "assignments": [{"focus": "balans", "count": 1}, {"focus": "zły", "count": 1}]},
            CONTEXT,
            store.save,
            None,
            fake_runner([]),
        )
    )
    assert data["added"] == 1 and any("HTTP 500" in problem for problem in data["problems"])
    with pytest.raises(ToolError) as error:
        drain(generate_examples_session("openai", "m", CorpusAgentTools(ROWS), {"goal": "g", "sources": [{"path": "a"}], "assignments": [{"focus": "a"}]}, CONTEXT, store.save))
    assert error.value.code == "no_sandbox"
    with pytest.raises(ToolError):
        drain(generate_examples_session("openai", "m", CorpusAgentTools(ROWS), {"goal": "", "assignments": [{"focus": "a"}]}, CONTEXT, store.save))
    with pytest.raises(ToolError) as error:
        drain(analyze_series_session("openai", "m", CorpusAgentTools(ROWS), {"goal": "g", "batch": "proposal-x"}, CONTEXT, store.save, store.reject))
    assert error.value.code == "not_found"


def test_plan_state_is_normalized_and_rendered():
    state, data = update_plan(
        {
            "intent": "Zamienić rozdział o wyłączeniach na przykłady.",
            "status": "executing",
            "steps": [
                {"id": "1", "title": "Przeczytać OWU", "kind": "understand", "executor": "read_large_file_session", "status": "done", "result": "12 wyłączeń"},
                {"id": "2", "title": "Seria wyłączeń", "kind": "generate", "status": "in_progress"},
                {"id": "3", "title": "Analiza serii", "kind": "analyze", "status": "nonsense"},
            ],
            "findings": ["OWU ma 12 wyłączeń", " "],
            "series": [{"batch": "proposal-1", "goal": "wyłączenia", "status": "wygenerowana"}],
        }
    )
    assert data == {"saved": True, "steps": 3, "done": 1} and state["steps"][2]["status"] == "pending" and state["findings"] == ["OWU ma 12 wyłączeń"]
    rendered = render_state(state)
    assert "✓ [1] (understand · read_large_file_session) Przeczytać OWU → 12 wyłączeń" in rendered and "▶ [2]" in rendered
    assert "proposal-1 [sft] wygenerowana: wyłączenia" in rendered
    assert render_state(None).startswith("(brak")
    assert plan_event(state)["steps"][1] == {"title": "Seria wyłączeń", "status": "in_progress"}
    assert PLAN_TOOL["parameters"]["required"] == ["intent", "status", "steps"]


def test_series_are_recorded_in_case_state_by_the_system():
    generated = record_series(None, "generate", {"goal": "NIP", "intent": "przykłady z rejestru", "training_mode": "dpo"}, {"batch": "proposal-a", "added": 8})
    assert generated["intent"] == "przykłady z rejestru" and generated["series"] == [
        {"batch": "proposal-a", "goal": "NIP", "training_mode": "dpo", "status": "wygenerowana", "notes": "generatory +8"}
    ]
    planned, _ = update_plan(
        {
            "intent": "i",
            "status": "executing",
            "steps": [
                {"id": "1", "title": "Seria", "kind": "generate", "status": "in_progress"},
                {"id": "2", "title": "Analiza", "kind": "analyze", "status": "pending"},
            ],
            "series": generated["series"],
        }
    )
    analysed = record_series(planned, "analyze", {"goal": ""}, {"batch": "proposal-a", "removed": 2, "added": 1, "kept": 7})
    assert analysed["series"] == [
        {"batch": "proposal-a", "goal": "NIP", "training_mode": "dpo", "status": "przeanalizowana", "notes": "generatory +8; analiza: −2 +1 → 7"}
    ]
    assert [(step["status"], step.get("result")) for step in analysed["steps"]] == [
        ("in_progress", None),
        ("done", f"{'proposal-a'[-8:]}: analiza: −2 +1 → 7"),
    ]
    assert "proposal-a [dpo] przeanalizowana: NIP" in render_state(analysed)
