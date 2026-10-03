import json
import uuid

import pytest

from bielik_lora import large_reader
from bielik_lora.agent import ToolError, run_tool, schema_errors
from bielik_lora.corpus_agent import TOOLS as CORPUS_TOOLS
from bielik_lora.corpus_agent import CorpusAgentTools
from bielik_lora.generation import ANALYST_TOOLS, ANALYZE_SERIES_TOOL, GENERATE_EXAMPLES_TOOL, GENERATOR_TOOLS
from bielik_lora.orchestration import PLAN_TOOL
from bielik_lora.sandbox_client import SANDBOX_TOOLS
from sandbox.workspace import SandboxError, Workspace

CONTRACTS = {
    tool["name"]: tool
    for tool in [
        *CORPUS_TOOLS,
        *SANDBOX_TOOLS,
        large_reader.READ_LARGE_FILE_TOOL,
        GENERATE_EXAMPLES_TOOL,
        ANALYZE_SERIES_TOOL,
        PLAN_TOOL,
        *GENERATOR_TOOLS,
        *ANALYST_TOOLS,
    ]
}
READER_CONTRACTS = {tool["name"]: tool for tool in large_reader.READER_TOOLS}


def drain(generator):
    events = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        return events, stop.value


@pytest.fixture()
def session(tmp_path):
    current = Workspace(tmp_path / "sessions").session(str(uuid.uuid4()))
    current.open()
    return current


def test_every_tool_declares_input_and_output_contract():
    for tool in [*CONTRACTS.values(), *READER_CONTRACTS.values()]:
        assert tool["parameters"]["type"] == "object" and tool["parameters"].get("additionalProperties") is False
        assert tool["returns"]["type"] == "object" and tool["returns"]["required"], tool["name"]


def test_schema_errors():
    schema = CONTRACTS["read_lines"]["parameters"]
    assert schema_errors(schema, {"path": "a", "start_line": 2}) == []
    problems = schema_errors(schema, {"start_line": 0, "extra": 1, "end_line": True})
    assert any("path: wymagane" in item for item in problems)
    assert any("minimum" in item for item in problems) and any("nieznane pole" in item for item in problems)
    assert any("end_line: oczekiwano integer" in item for item in problems)


def test_run_tool_envelope_and_validation():
    tools = [CONTRACTS["read_lines"]]
    events, output = drain(run_tool(lambda name, args: ({"x": 1}, None), tools, "read_lines", {"path": "a"}))
    assert json.loads(output)["ok"] is True and json.loads(output)["data"] == {"x": 1}
    assert [event["type"] for event in events] == ["tool_call", "tool_result"]

    _, output = drain(run_tool(lambda name, args: ({}, None), tools, "read_lines", {"start_line": "1"}))
    result = json.loads(output)
    assert result["ok"] is False and result["error"]["code"] == "invalid_arguments" and result["error"]["details"]

    def failing(name, args):
        raise ToolError("use_large_reader", "za duży")

    _, output = drain(run_tool(failing, tools, "read_lines", {"path": "a"}))
    assert json.loads(output)["error"]["code"] == "use_large_reader"

    def streaming(name, args):
        yield {"type": "progress", "name": name, "message": "1/2"}
        return {"done": True}, None

    events, output = drain(run_tool(streaming, tools, "read_lines", {"path": "a"}))
    assert [event["type"] for event in events] == ["tool_call", "progress", "tool_result"]
    assert json.loads(output)["data"] == {"done": True}
    _, output = drain(run_tool(streaming, tools, "nope", {}))
    assert json.loads(output)["error"]["code"] == "unknown_tool"


def test_corpus_tools_match_output_contracts():
    rows = [
        {
            "id": "1",
            "split": "train",
            "flag": "positive",
            "messages": [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {"role": "assistant", "content": '{"entities": [{"type": "NIP", "value": "1"}]}'},
            ],
        }
    ]
    tools = CorpusAgentTools(rows)
    assert schema_errors(CONTRACTS["corpus_overview"]["returns"], tools("corpus_overview", {})[0]) == []
    assert schema_errors(CONTRACTS["list_examples"]["returns"], tools("list_examples", {"limit": 5})[0]) == []
    proposal = {"user": "inne", "assistant": '{"entities": []}', "flag": "negative"}
    assert schema_errors(CONTRACTS["propose_examples"]["returns"], tools("propose_examples", {"examples": [proposal]})[0]) == []


def test_sandbox_tools_match_output_contracts(session):
    def check(name, result):
        assert schema_errors(CONTRACTS[name]["returns"], result) == [], name

    check("write_file", session.write_file("work/a.txt", "linia 1\nlinia 2"))
    check("read_file", session.read_file("work/a.txt"))
    check("read_lines", session.read_lines("work/a.txt", 2))
    check("search_lines", session.search_lines("work/a.txt", "linia", context_lines=1))
    check("list_files", session.list_files())
    check("add_note", session.add_note("fact", "Druga linia.", source="work/a.txt", start_line=2))
    (session.path / "uploads" / "a.json").write_text('{"a": 1}', encoding="utf-8")
    check("convert_document", session.convert_document("uploads/a.json"))
    check("list_files", session.list_files(area="uploads"))
    session.save_upload("uploads/notatki.txt", ["linia\n".encode() * 3], 1000)
    check("list_attachments", {"attachments": [session.attach("uploads/notatki.txt")]})


def test_attachments_manifest_and_unique_names(session):
    first = session.save_upload("uploads/a.txt", [b"x\n"], 100, unique=True)
    second = session.save_upload("uploads/a.txt", [b"y\n"], 100, unique=True)
    assert (first["path"], second["path"]) == ("uploads/a.txt", "uploads/a (2).txt")
    text = session.attach(second["path"])
    assert text["id"] == "A1" and text["handling"] == "text" and text["lines"] == 1 and text["read_with"] == "read_file"
    session.save_upload("uploads/blob.bin", [b"\x00\x01"], 100)
    binary = session.attach("uploads/blob.bin")
    assert binary["id"] == "A2" and binary["handling"] == "binary" and binary["read_with"] == "shell/run_script"
    assert [item["id"] for item in session.info()["attachments"]] == ["A1", "A2"]
    assert {item["path"] for item in session.list_files(area="uploads")["files"]} == {"uploads/a.txt", "uploads/a (2).txt", "uploads/blob.bin"}
    with pytest.raises(SandboxError):
        session.list_files(area="nope")


def test_open_migrates_legacy_inputs_to_uploads(session):
    (session.path / "uploads" / "owu.txt").write_text("istniejący", encoding="utf-8")
    legacy = session.path / "inputs"
    legacy.mkdir()
    (legacy / "owu.txt").write_text("stary plik\n", encoding="utf-8")
    session.open()
    assert not legacy.exists()
    manifest = session.list_attachments()["attachments"]
    assert [(item["id"], item["path"]) for item in manifest] == [("A1", "uploads/owu (2).txt")]
    session.open()
    assert len(session.list_attachments()["attachments"]) == 1


def test_read_file_refuses_large_files(session):
    (session.path / "uploads" / "big.txt").write_text("x" * 100 + "\n" * 600, encoding="utf-8")
    (session.path / "uploads" / "huge.txt").write_text(("y" * 99 + "\n") * 600, encoding="utf-8")
    assert session.read_file("uploads/big.txt")["total_lines"] == 600
    with pytest.raises(SandboxError) as error:
        session.read_file("uploads/huge.txt")
    assert error.value.code == "use_large_reader" and error.value.status == 413
    assert session.read_lines("uploads/huge.txt", 10, 10)["content"].endswith("y" * 99)
