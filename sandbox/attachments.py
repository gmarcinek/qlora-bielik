"""Chat attachments: file on the session volume + manifest entry (attachments.json) the assistant can rely on."""

from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any

from sandbox.converters import BINARY_DOCUMENTS, TEXT_SUFFIXES
from sandbox.large_read import LARGE_FILE_BYTES

if TYPE_CHECKING:
    from sandbox.workspace import Session

# Stored as-is (no MarkItDown); everything in BINARY_DOCUMENTS is converted on upload.
PLAIN_TEXT_SUFFIXES = TEXT_SUFFIXES | {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".sql", ".sh", ".ps1", ".ini", ".toml", ".cfg", ".env.example", ".srt", ".vtt", ".tex", ".rst",
}


def manifest(session: "Session") -> list[dict[str, Any]]:
    path = session.path / "attachments.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def attach(session: "Session", path: str) -> dict[str, Any]:
    from sandbox.workspace import SandboxError

    session.require_open()
    source = session.existing_file(path)
    entries = manifest(session)
    suffix = source.suffix.lower()
    entry: dict[str, Any] = {
        "id": f"A{len(entries) + 1}",
        "name": source.name,
        "path": session.display(source),
        "bytes": source.stat().st_size,
        "extension": suffix,
        "sha256": sha256(source),
        "uploaded_at": time.time(),
    }
    readable_bytes = None
    if suffix in BINARY_DOCUMENTS:
        entry["handling"] = "markitdown"
        try:
            conversion = session.convert_document(path)
            converted = session.resolve(conversion["output"])
            readable_bytes = converted.stat().st_size
            entry.update(
                converted=conversion["output"],
                method=conversion.get("method"),
                lines=conversion.get("lines"),
                chars=conversion.get("chars"),
                warnings=conversion.get("warnings", []),
            )
        except SandboxError as error:
            entry.update(converted=None, error=str(error))
    elif suffix in PLAIN_TEXT_SUFFIXES:
        entry["handling"] = "text"
        try:
            text_path, encoding, _ = session.readable(path)
            lines = chars = 0
            for lines, line in enumerate(session.iter_lines(text_path, encoding), start=1):
                chars += len(line) + 1
            entry.update(encoding=encoding, lines=lines, chars=chars)
            readable_bytes = entry["bytes"]
        except SandboxError as error:
            entry.update(handling="binary", error=str(error))
    else:
        entry["handling"] = "binary"
    if readable_bytes is None:
        entry["read_with"] = "shell/run_script"
    else:
        entry["read_with"] = "read_large_file_session" if readable_bytes > LARGE_FILE_BYTES else "read_file"
    entries.append(entry)
    (session.path / "attachments.json").write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry
