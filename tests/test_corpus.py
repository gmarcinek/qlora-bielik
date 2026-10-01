import json
from pathlib import Path

import pytest

from bielik_lora.corpus import export_jsonl, import_jsonl, initialize_database, validate_messages


def test_import_and_export_conversations(tmp_path: Path) -> None:
    database = tmp_path / "corpus.sqlite3"
    source = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Czesc"}, {"role": "assistant", "content": "Czesc"}]}) + "\n",
        encoding="utf-8",
    )
    initialize_database(database)
    assert import_jsonl(database, source, "train") == 1
    output = tmp_path / "output.jsonl"
    assert export_jsonl(database, output, "train") == 1
    assert json.loads(output.read_text(encoding="utf-8"))["messages"][-1]["role"] == "assistant"


def test_rejects_incomplete_conversation() -> None:
    with pytest.raises(ValueError, match="final message"):
        validate_messages([{"role": "user", "content": "Pytanie"}])