"""File previews for the UI: text up to a size limit, spreadsheets as Markdown tables, Office documents as PDF."""

from __future__ import annotations

import csv
import hashlib
import io
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from sandbox.converters import decode_bytes

MAX_TEXT_BYTES = 300 * 1024
MAX_SHEET_ROWS = 500
MAX_SHEET_COLUMNS = 40
IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp",
}
OFFICE_TO_PDF = {".doc", ".docx", ".odt", ".rtf", ".ppt", ".pptx", ".odp"}
SPREADSHEETS = {".xlsx", ".xlsm", ".xls", ".ods"}
DELIMITED = {".csv": ",", ".tsv": "\t"}
MARKDOWN = {".md", ".markdown"}
CACHE = Path(tempfile.gettempdir()) / "preview-cache"


class PreviewError(Exception):
    pass


def cache_key(path: Path, suffix: str) -> Path:
    stat = path.stat()
    digest = hashlib.sha256(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{digest}{suffix}"


def soffice_convert(path: Path, target: str) -> Path:
    """LibreOffice conversion, cached by path, size and mtime (conversion takes seconds)."""
    cached = cache_key(path, f".{target}")
    if cached.exists():
        return cached
    if not shutil.which("soffice"):
        raise PreviewError("Brak LibreOffice (soffice) w sandboksie.")
    with tempfile.TemporaryDirectory() as temporary:
        result = subprocess.run(
            ["soffice", "--headless", f"-env:UserInstallation=file://{temporary}/profile",
             "--convert-to", target, "--outdir", temporary, str(path)],
            capture_output=True, timeout=180, check=False,
        )
        converted = Path(temporary) / f"{path.stem}.{target}"
        if result.returncode != 0 or not converted.exists():
            raise PreviewError(f"LibreOffice nie przekonwertował pliku: {result.stderr.decode(errors='replace')[-300:]}")
        shutil.move(str(converted), cached)
    return cached


def cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def markdown_table(rows: list[list[Any]]) -> tuple[str, bool]:
    rows = [row for row in rows if any(value not in (None, "") for value in row)]
    if not rows:
        return "_(pusty arkusz)_", False
    truncated = len(rows) > MAX_SHEET_ROWS + 1 or any(len(row) > MAX_SHEET_COLUMNS for row in rows)
    rows = [list(row[:MAX_SHEET_COLUMNS]) for row in rows[: MAX_SHEET_ROWS + 1]]
    width = max(len(row) for row in rows)
    header, *body = [[cell(value) for value in row] + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width, *("| " + " | ".join(row) + " |" for row in body)]
    return "\n".join(lines), truncated


def spreadsheet_markdown(path: Path) -> tuple[str, bool]:
    from openpyxl import load_workbook

    source = path if path.suffix.lower() in {".xlsx", ".xlsm"} else soffice_convert(path, "xlsx")
    workbook = load_workbook(source, read_only=True, data_only=True)
    sections, truncated = [], False
    for sheet in workbook.worksheets:
        rows = [list(row) for _, row in zip(range(MAX_SHEET_ROWS + 2), sheet.iter_rows(values_only=True))]
        table, cut = markdown_table(rows)
        truncated = truncated or cut or (sheet.max_row or 0) > MAX_SHEET_ROWS + 1
        sections.append(f"## {sheet.title}\n\n{table}")
    return "\n\n".join(sections), truncated


def delimited_markdown(path: Path, delimiter: str) -> tuple[str, bool]:
    text, _ = decode_bytes(path.read_bytes()[: 4 * MAX_TEXT_BYTES])
    try:
        delimiter = csv.Sniffer().sniff(text[:5000], delimiters=",;\t|").delimiter
    except csv.Error:
        pass
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    return markdown_table(rows)


def preview(path: Path) -> dict[str, Any]:
    """{kind: image|pdf, file, media_type} for binary previews, {kind: text|markdown, content, truncated} otherwise."""
    suffix = path.suffix.lower()
    size = path.stat().st_size
    if suffix in IMAGE_TYPES:
        return {"kind": "image", "file": path, "media_type": IMAGE_TYPES[suffix]}
    if suffix == ".pdf":
        return {"kind": "pdf", "file": path, "media_type": "application/pdf"}
    if suffix in OFFICE_TO_PDF:
        return {"kind": "pdf", "file": soffice_convert(path, "pdf"), "media_type": "application/pdf"}
    if suffix in SPREADSHEETS:
        content, truncated = spreadsheet_markdown(path)
        return {"kind": "markdown", "content": content, "truncated": truncated}
    if suffix in DELIMITED:
        content, truncated = delimited_markdown(path, DELIMITED[suffix])
        return {"kind": "markdown", "content": content, "truncated": truncated}
    head = path.read_bytes()[: MAX_TEXT_BYTES + 1]
    if b"\x00" in head[:8192]:
        return {"kind": "unsupported", "content": "Plik binarny \u2014 brak podgl\u0105du.", "truncated": False}
    if size > MAX_TEXT_BYTES:
        return {
            "kind": "unsupported",
            "content": f"Plik tekstowy ma {size // 1024} KB \u2014 podgl\u0105d do {MAX_TEXT_BYTES // 1024} KB. Pobierz plik.",
            "truncated": True,
        }
    text, _ = decode_bytes(head)
    return {"kind": "markdown" if suffix in MARKDOWN else "text", "content": text, "truncated": False}
