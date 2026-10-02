import json
import os
import uuid

import pytest

from sandbox.converters import convert, decode_bytes
from sandbox.workspace import SandboxError, Workspace


@pytest.fixture()
def session(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "owu.txt").write_text("Wyłączenia\nNie odpowiadamy za szkody umyślne.\n", encoding="utf-8")
    current = Workspace(tmp_path / "sessions", shared).session(str(uuid.uuid4()))
    current.open({"corpus": "test"})
    return current


def test_decode_bytes_handles_polish_legacy_encodings():
    text = "Zażółć gęślą jaźń. Ubezpieczyciel nie odpowiada za szkody wyrządzone umyślnie przez Ubezpieczonego."
    assert decode_bytes(text.encode("cp1250"))[0] == text
    assert decode_bytes(text.encode("utf-16"))[0] == text
    assert decode_bytes(b"\xef\xbb\xbfa\r\nb")[0] == "a\nb"


def test_rejects_invalid_session_id_and_path_traversal(session, tmp_path):
    with pytest.raises(SandboxError):
        Workspace(tmp_path / "sessions").session("../etc")
    with pytest.raises(SandboxError):
        session.resolve("../../outside.txt")
    with pytest.raises(SandboxError):
        session.write_file("/shared/new.txt", "x")
    assert session.resolve("/shared/owu.txt").read_text(encoding="utf-8").startswith("Wyłączenia")


def test_write_read_file_and_lines(session):
    session.write_file("work/a.txt", "\n".join(f"linia {n}" for n in range(1, 451)))
    lines = session.read_lines("work/a.txt", 10, 12)
    assert lines["content"].splitlines() == ["    10| linia 10", "    11| linia 11", "    12| linia 12"]
    assert lines["next_start_line"] == 13 and lines["total_lines"] == 450
    assert session.read_lines("work/a.txt", 1)["end_line"] == 450
    whole = session.read_file("work/a.txt")
    assert whole["content"].endswith("linia 450") and whole["next_start_line"] is None


def test_reader_continues_past_context_limit(session, monkeypatch):
    monkeypatch.setattr("sandbox.workspace.CONTEXT_LIMIT_CHARS", 100)
    (session.path / "work" / "long.txt").write_text("\n".join("x" * 30 for _ in range(10)), encoding="utf-8")
    first = session.read_file("work/long.txt")
    assert first["end_line"] == 3 and first["next_start_line"] == 4 and "read_lines" in first["note"]
    rest = session.read_lines("work/long.txt", first["next_start_line"])
    assert rest["start_line"] == 4 and rest["end_line"] == 5 and rest["stopped_at_context_limit"]
    (session.path / "work" / "one.txt").write_text("y" * 250, encoding="utf-8")
    assert session.read_file("work/one.txt")["note"].startswith("Linia 1")


def test_cp1250_file_is_read_with_detected_encoding(session):
    (session.path / "uploads" / "owu.txt").write_bytes("Zażółć gęślą jaźń\nŚwiadczenie".encode("cp1250"))
    result = session.read_lines("uploads/owu.txt")
    assert "Zażółć gęślą jaźń" in result["content"] and result["encoding"] in {"cp1250", "windows-1250"}


def test_text_conversion_json_and_jsonl(tmp_path):
    (tmp_path / "a.json").write_text('{"a": "ż"}', encoding="cp1250")
    text, details = convert(tmp_path / "a.json")
    assert json.loads(text) == {"a": "ż"} and "\n" in text
    (tmp_path / "b.jsonl").write_text('{"a": 1}\nnope\n', encoding="utf-8")
    assert "2" in convert(tmp_path / "b.jsonl")[1]["warnings"][0]


def test_handoff_notes(session):
    with pytest.raises(SandboxError):
        session.add_note("fact", "bez źródła")
    fact = session.add_note("fact", "Szkody umyślne są wyłączone.", source="/shared/owu.txt", start_line=2)
    assert fact["id"] == "F1"
    with pytest.raises(SandboxError):
        session.add_note("interpretation", "x", based_on=["F9"])
    assert session.add_note("interpretation", "Potrzebne przykłady EXCLUSION.", based_on=["F1"])["id"] == "I1"
    handoff = (session.path / "notes" / "HANDOFF.md").read_text(encoding="utf-8")
    assert "**F1**" in handoff and "> Nie odpowiadamy za szkody umyślne." in handoff and "F1)" in handoff


def test_closed_session_is_read_only(session):
    session.write_file("work/a.txt", "x")
    session.close()
    with pytest.raises(SandboxError) as error:
        session.write_file("work/b.txt", "y")
    assert error.value.status == 409
    assert session.read_file("work/a.txt")["content"] == "x"
    session.open()
    session.write_file("work/b.txt", "y")


@pytest.mark.skipif(os.name != "posix", reason="sandbox runs on Linux")
def test_shell_and_script(session):
    assert session.shell("echo ok && pwd")["stdout"].split() == ["ok", str(session.path)]
    result = session.run_script("import sys; print(sys.argv[1])", args=["x"])
    assert result["stdout"].strip() == "x" and result["script"] == "scripts/script-001.py"
    assert session.shell("sleep 5", timeout_seconds=1)["timed_out"]
