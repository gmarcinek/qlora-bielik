"""Agentic large-file reader against a real sandbox session; the LLM is replaced by a scripted reader."""

import json
import uuid

import pytest

from bielik_lora import large_reader
from bielik_lora.agent import run_tool, schema_errors
from sandbox.workspace import Workspace


def drain(generator):
    events = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        return events, stop.value


class LocalSandbox:
    def __init__(self, session):
        self.session = session

    def call(self, session_id, name, arguments):
        return self.session.call(name, arguments)


@pytest.fixture()
def session(tmp_path):
    current = Workspace(tmp_path / "sessions").session(str(uuid.uuid4()))
    current.open()
    return current


def head_answer(prompt):
    first = int(prompt.split("Zakres L", 1)[1].split("\u2013", 1)[0])
    return {
        "note": f"**Zakres pokrycia:** od L{first}.",
        "text_type": "prawny",
        "sections": [{"title": f"Blok {first}", "start_line": first}],
        "facts": [{"id": "N1", "text": f"Linia {first} wy\u0142\u0105cza sporty.", "start_line": first, "end_line": first, "questions": ["Q1"]}],
    }


SYNTHESIS = {
    "status": "PARTIAL",
    "text_type": "prawny",
    "summary": "Wy\u0142\u0105czenia [R1.F1].",
    "handoff": "**Status:** PARTIAL\n\n**Granice zakresu / czego nie zak\u0142ada\u0107:** nie uog\u00f3lnia\u0107 na aneksy.",
    "resolved": [{"question": "Q1", "answer": "Tak [R1.F1].", "based_on": ["R1.F1"]}],
    "unresolved": [{"question": "Q2", "issue": "Suma", "reason": "Brak.", "next_step": "L50\u2013L60."}],
    "interpretations": [{"text": "Sp\u00f3jne.", "based_on": ["R1.F1", "X9"]}],
    "contradictions": [],
}


def scripted_reader(steps):
    """Stands in for run_agent: executes a fixed sequence of reader tool calls through the real contracts."""

    def runner(provider, model, system, messages, tools, execute, max_steps=16):
        assert "CEL CZYTANIA: Jakie s\u0105 wy\u0142\u0105czenia?" in system and "{filename}" not in system
        for name, arguments in steps:
            yield {"type": "usage", "model": model, "llm_ms": 3, "input_tokens": 10, "output_tokens": 2}
            output = yield from run_tool(execute, tools, name, arguments)
            runner.outputs.append((name, json.loads(output)))
        yield {"type": "text", "content": "Gotowe."}

    runner.outputs = []
    return runner


def test_reader_agent_navigation_heads_notes_and_handoff(session, monkeypatch):
    (session.path / "uploads" / "owu.txt").write_text(
        "\n".join(f"\u00a7 {n}. Ubezpieczyciel nie odpowiada za sporty {n}." for n in range(1, 2201)), encoding="utf-8"
    )
    steps = [
        ("read_lines", {"count": 100}),
        ("search_lines", {"query": "sporty 2000", "context_lines": 1}),
        ("save_notes", {
            "note": "**Zakres pokrycia:** L1\u2013L100.",
            "facts": [
                {"id": "N1", "text": "Pierwsze wy\u0142\u0105czenie.", "start_line": 1, "end_line": 1, "questions": ["Q1"]},
                {"id": "N2", "text": "Nieprzeczytane.", "start_line": 1500},
            ],
            "interpretations": [{"text": "HIPOTEZA: szerokie.", "based_on": ["N1", "N2"]}],
            "sections": [{"title": "\u00a7 1", "start_line": 1, "end_line": 100}],
        }),
        ("skip_lines", {"count": 50}),
        ("jump_to_line", {"line": 2195}),
        ("read_lines", {"count": 2000}),
        ("run_multi_head_reconnaissance", {"target_lines_per_head": 1000}),
        ("run_multi_head_reconnaissance", {"target_lines_per_head": 1000}),
        ("save_notes", {"note": "**Notatka ko\u0144cowa:** ca\u0142o\u015b\u0107 pokryta g\u0142owicami."}),
        ("finish", {}),
        ("read_lines", {"count": 10}),
    ]
    runner = scripted_reader(steps)
    monkeypatch.setattr(large_reader, "run_agent", runner)

    def completion(provider, model, system, prompt, max_tokens=32000):
        answer = head_answer(prompt) if "g\u0142owic\u0105 rekonesansu" in system else SYNTHESIS
        return answer, {"model": model, "llm_ms": 5, "input_tokens": 100, "output_tokens": 20}

    monkeypatch.setattr(large_reader, "complete_json", completion)
    events, (result, _) = drain(
        large_reader.read_large_file_session(
            LocalSandbox(session), session.id, "anthropic", "claude-test", "uploads/owu.txt",
            "Jakie s\u0105 wy\u0142\u0105czenia?", ["Czy s\u0105 wy\u0142\u0105czenia?", "Jaka jest suma?"],
        )
    )
    assert runner.outputs[0][1]["data"]["end_line"] == 100
    assert runner.outputs[1][1]["data"]["total_matches"] == 1
    assert runner.outputs[2][1]["data"]["rejected_facts"][0]["reason"].endswith("nie zosta\u0142y przeczytane")
    assert runner.outputs[5][1]["data"]["start_line"] == 2195 and runner.outputs[5][1]["data"]["end_line"] == 2200
    assert runner.outputs[6][1]["ok"] and len(runner.outputs[6][1]["data"]["heads"]) == 3
    assert runner.outputs[7][1]["error"]["code"] == "already_used"
    assert runner.outputs[10][1]["error"]["code"] == "finished"

    assert schema_errors(large_reader.READ_LARGE_FILE_TOOL["returns"], result) == []
    assert result["status"] == "PARTIAL" and [item["question"] for item in result["synthesis"]["unresolved"]] == ["Q2"]
    telemetry = result["telemetry"]
    assert telemetry["coverage_pct"] == 100.0 and telemetry["heads"] == 3 and telemetry["searches"] == 1
    assert telemetry["skips"] == 1 and telemetry["jumps"] == 1 and telemetry["rejected_facts"] == 1
    assert telemetry["llm_calls"] == len(steps) + 3 + 1
    assert {event["type"] for event in events} == {"progress", "agent_tool"}
    reader_calls = [event for event in events if event["type"] == "agent_tool" and event["phase"] == "call"]
    assert len(reader_calls) == len(steps) and all(event["agent"] == "czytelnik" and event["scope"] == "R1" for event in reader_calls)
    assert any(event["phase"] == "result" and not event["ok"] for event in events if event["type"] == "agent_tool")
    progress_events = [event for event in events if event["type"] == "progress"]
    assert any("g\u0142owica 2" in event["message"] for event in progress_events) and "synteza" in progress_events[-2]["message"]

    record = json.loads((session.path / "notes" / "reads" / "R1.json").read_text(encoding="utf-8"))
    assert record["interpretations"][0]["based_on"] == ["R1.F1"]
    assert record["facts"][0]["quote"].startswith("\u00a7 1.")
    assert record["synthesis"]["unknown_refs"] == ["X9"]
    handoff = (session.path / "notes" / "reads" / "R1.md").read_text(encoding="utf-8")
    for expected in ("**Status:** PARTIAL", "czego nie zak\u0142ada\u0107", "## Mapa struktury", "- \u00a7 1 \u2014 L1\u2013L100",
                     "### Notatka 1 \u00b7 reader", "### Notatka 2 \u00b7 g\u0142owica 1", "**[R1.F1]**", "> \u00a7 1.",
                     "**search** \u201esporty 2000\u201d", "**skip** pomini\u0119to L101\u2013L150", "## Telemetria",
                     "### Nierozstrzygni\u0119te", "L50\u2013L60."):
        assert expected in handoff, expected
    assert "PARTIAL" in (session.path / "notes" / "HANDOFF.md").read_text(encoding="utf-8")


def test_search_lines_counts_all_matches_with_context(session):
    (session.path / "work" / "log.txt").write_text("\n".join(f"{n} {'ERROR' if n % 10 == 0 else 'ok'}" for n in range(1, 101)), encoding="utf-8")
    found = session.search_lines("work/log.txt", "error", context_lines=1, max_matches=3)
    assert found["total_matches"] == 10 and found["returned"] == 3 and found["truncated"]
    assert found["matches"][0]["line"] == 10 and found["matches"][0]["context"].splitlines() == ["     9| 9 ok", "    10> 10 ERROR", "    11| 11 ok"]
    assert session.search_lines("work/log.txt", r"^\d+0 ", regex=True)["total_matches"] == 10
