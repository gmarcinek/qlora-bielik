from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable


SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY,
    split TEXT NOT NULL CHECK(split IN ('train', 'validation', 'test')),
    messages_json TEXT NOT NULL,
    source TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conversations_split ON conversations(split);
"""
VALID_ROLES = {"system", "user", "assistant"}


def initialize_database(database_path: Path) -> None:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.executescript(SCHEMA)


def validate_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("every message must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError("messages require a supported role and non-empty content")
        normalized.append({"role": role, "content": content})
    if normalized[-1]["role"] != "assistant":
        raise ValueError("the final message must have the assistant role")
    return normalized


def import_jsonl(database_path: Path, jsonl_path: Path, split: str, source: str | None = None) -> int:
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    rows: list[tuple[str, str, str | None]] = []
    with jsonl_path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                messages = validate_messages(row["messages"])
            except (KeyError, json.JSONDecodeError, ValueError) as error:
                raise ValueError(f"invalid record at line {line_number}: {error}") from error
            rows.append((split, json.dumps(messages, ensure_ascii=False), source or str(jsonl_path)))
    with sqlite3.connect(database_path) as connection:
        connection.executemany(
            "INSERT INTO conversations(split, messages_json, source) VALUES (?, ?, ?)", rows
        )
    return len(rows)


def export_jsonl(database_path: Path, jsonl_path: Path, split: str) -> int:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        records: Iterable[tuple[str]] = connection.execute(
            "SELECT messages_json FROM conversations WHERE split = ? ORDER BY id", (split,)
        )
        with jsonl_path.open("w", encoding="utf-8") as output_file:
            count = 0
            for (messages_json,) in records:
                output_file.write(json.dumps({"messages": json.loads(messages_json)}, ensure_ascii=False) + "\n")
                count += 1
    return count