"""Client of the sandbox sidecar and the tool definitions exposed to the corpus agent."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, parse, request

from bielik_lora.agent import ToolError

INT = {"type": "integer"}
STR = {"type": "string"}
EXEC_RESULT = {
    "type": "object",
    "required": ["exit_code", "timed_out", "seconds", "stdout", "stderr"],
    "properties": {
        "exit_code": {"type": ["integer", "null"]},
        "timed_out": {"type": "boolean"},
        "seconds": {"type": "number"},
        "stdout": STR,
        "stderr": STR,
        "stdout_file": STR,
        "stderr_file": STR,
        "script": STR,
    },
}
READ_RESULT = {
    "type": "object",
    "required": ["path", "total_lines", "start_line", "end_line", "next_start_line", "content"],
    "properties": {
        "path": STR,
        "total_lines": INT,
        "start_line": INT,
        "end_line": INT,
        "next_start_line": {"type": ["integer", "null"]},
        "content": STR,
        "encoding": STR,
        "converted_from": STR,
        "method": {"type": ["string", "null"]},
        "stopped_at_context_limit": {"type": "boolean"},
        "note": STR,
    },
}
ATTACHMENT = {
    "type": "object",
    "required": ["id", "name", "path", "bytes", "handling", "read_with"],
    "properties": {
        "id": STR,
        "name": STR,
        "path": STR,
        "bytes": INT,
        "extension": STR,
        "sha256": STR,
        "handling": {"type": "string", "enum": ["markitdown", "text", "binary"]},
        "converted": {"type": ["string", "null"]},
        "method": {"type": ["string", "null"]},
        "lines": INT,
        "chars": INT,
        "encoding": STR,
        "warnings": {"type": "array", "items": STR},
        "error": STR,
        "read_with": STR,
        "uploaded_at": {"type": "number"},
    },
}

SANDBOX_TOOLS: list[dict[str, Any]] = [
    {
        "name": "shell",
        "description": "Polecenie bash w katalogu sesji (Linux, Python 3.11, rg, jq, pdftotext, tesseract; bez internetu).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600},
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "returns": EXEC_RESULT,
    },
    {
        "name": "run_script",
        "description": "Zapisuje skrypt w scripts/ i uruchamia go w katalogu sesji (pandas, openpyxl, pypdf, pymupdf, rapidfuzz dostępne).",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "language": {"type": "string", "enum": ["python", "bash"]},
                "name": {"type": "string", "description": "Nazwa pliku skryptu (opcjonalnie), np. extract_tables."},
                "args": {"type": "array", "items": {"type": "string"}},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
        "returns": EXEC_RESULT,
    },
    {
        "name": "read_file",
        "description": (
            "Czyta cały mały plik (do 50 KB tekstu) — dokumenty PDF/DOCX/XLSX/PPTX/obrazy są automatycznie konwertowane do Markdown. "
            "Większe pliki zwracają błąd use_large_reader: czytaj je read_large_file_session (albo read_lines dla konkretnego zakresu)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "returns": READ_RESULT,
    },
    {
        "name": "read_lines",
        "description": (
            "Czyta zakres linii (numerowane) od start_line do end_line; bez end_line — do końca pliku lub limitu kontekstu, "
            "wtedy kontynuuj od next_start_line. Służy do weryfikacji konkretnych miejsc i cytowania faktów w add_note."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "returns": READ_RESULT,
    },
    {
        "name": "search_lines",
        "description": (
            "Przeszukuje plik tekstowy (dowolnej wielkości, dokumenty po konwersji) bez czytania całości: łączna liczba pasujących "
            "linii i pierwsze trafienia z kontekstem i numerami linii. Szybkie sprawdzenie frazy, nazwy lub wzorca (regex); "
            "do analizy przekrojowej użyj read_large_file_session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "query": {"type": "string"},
                "regex": {"type": "boolean"},
                "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
                "max_matches": {"type": "integer", "minimum": 1, "maximum": 500},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path", "query"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["path", "total_lines", "total_matches", "returned", "truncated", "matches"],
            "properties": {
                "path": STR,
                "total_lines": INT,
                "total_matches": INT,
                "returned": INT,
                "truncated": {"type": "boolean"},
                "matches": {
                    "type": "array",
                    "items": {"type": "object", "required": ["line", "text", "context"], "properties": {"line": INT, "text": STR, "context": STR}},
                },
            },
        },
    },
    {
        "name": "write_file",
        "description": "Zapisuje (lub dopisuje) plik tekstowy w katalogu sesji.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "append": {"type": "boolean"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "returns": {"type": "object", "required": ["path", "bytes"], "properties": {"path": STR, "bytes": INT}},
    },
    {
        "name": "list_files",
        "description": "Lista plików sesji wg obszaru (uploads — załączniki użytkownika, work — robocze, exports — wyniki dla użytkownika, converted, notes, scripts, shared, all) albo ścieżki i wzorca glob.",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {"type": "string", "enum": ["uploads", "work", "exports", "converted", "notes", "scripts", "shared", "all"]},
                "path": {"type": "string"},
                "pattern": {"type": "string", "description": "Wzorzec glob, domyślnie **/*."},
                "limit": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["files", "truncated"],
            "properties": {
                "files": {
                    "type": "array",
                    "items": {"type": "object", "required": ["path", "bytes"], "properties": {"path": STR, "bytes": INT}},
                },
                "truncated": {"type": "boolean"},
            },
        },
    },
    {
        "name": "convert_document",
        "description": "Konwertuje dokument (PDF w różnych kodowaniach i skany przez OCR, DOC/DOCX, XLS/XLSX, PPT/PPTX, HTML, TXT/MD/JSON/JSONL) do Markdown w converted/.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "force": {"type": "boolean"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["source", "output", "chars", "lines", "method", "cached"],
            "properties": {
                "source": STR,
                "output": STR,
                "chars": INT,
                "lines": INT,
                "method": STR,
                "cached": {"type": "boolean"},
                "pages": INT,
                "ocr_pages": {"type": "array", "items": INT},
                "encoding": STR,
                "warnings": {"type": "array", "items": STR},
            },
        },
    },
    {
        "name": "add_note",
        "description": (
            "Dopisuje notatkę do notes/HANDOFF.md. kind=fact: tylko to, co wprost stoi w źródle — wymaga source i start_line "
            "(cytat zapisuje się automatycznie). kind=interpretation: Twój wniosek, wskaż based_on (id faktów)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["fact", "interpretation"]},
                "text": {"type": "string"},
                "source": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "based_on": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "text"],
            "additionalProperties": False,
        },
        "returns": {"type": "object", "required": ["id", "handoff"], "properties": {"id": STR, "handoff": STR}},
    },
    {
        "name": "list_attachments",
        "description": "Manifest załączników rozmowy (A1, A2, …): plik w uploads/, rozmiar, sposób obsługi (markitdown/text/binary), ścieżka po konwersji i zalecane narzędzie odczytu.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "returns": {
            "type": "object",
            "required": ["attachments"],
            "properties": {"attachments": {"type": "array", "items": ATTACHMENT}},
        },
    },
]
SANDBOX_TOOL_NAMES = {tool["name"] for tool in SANDBOX_TOOLS}


HTTP_CODES = {400: "invalid_arguments", 401: "unauthorized", 404: "not_found", 409: "conflict", 413: "too_large", 415: "unsupported_format", 422: "conversion_failed"}


class SandboxClient:
    def __init__(self, base_url: str | None = None, token: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("SANDBOX_URL", "")).rstrip("/")
        self.token = token or os.getenv("SANDBOX_TOKEN", "")

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def request(self, method: str, path: str, payload: Any = None, data: bytes | None = None, timeout: int = 3700) -> Any:
        body = data if data is not None else (json.dumps(payload).encode("utf-8") if payload is not None else None)
        headers = {"Authorization": f"Bearer {self.token}"}
        if data is None and payload is not None:
            headers["Content-Type"] = "application/json"
        elif data is not None:
            headers["Content-Type"] = "application/octet-stream"
        http_request = request.Request(f"{self.base_url}{path}", data=body, headers=headers, method=method)
        try:
            with request.urlopen(http_request, timeout=timeout) as response:
                return json.load(response)
        except error.HTTPError as http_error:
            raw = http_error.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw)
                message, code = str(body.get("detail", raw)), body.get("code")
            except (json.JSONDecodeError, AttributeError):
                message, code = raw, None
            raise ToolError(code or HTTP_CODES.get(http_error.code, "sandbox_error"), message) from http_error
        except error.URLError as url_error:
            raise ToolError("sandbox_unavailable", f"Sandbox niedostępny: {url_error.reason}") from url_error

    def open(self, session_id: str, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/open", meta or {}, timeout=30)

    def close(self, session_id: str) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/close", {}, timeout=30)

    def info(self, session_id: str) -> dict[str, Any]:
        return self.request("GET", f"/sessions/{session_id}", timeout=30)

    def sessions(self) -> list[dict[str, Any]]:
        return self.request("GET", "/sessions", timeout=30)

    def delete(self, session_id: str) -> dict[str, Any]:
        return self.request("DELETE", f"/sessions/{session_id}", timeout=120)

    def call(self, session_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", f"/sessions/{session_id}/tools/{name}", arguments)

    def upload(self, session_id: str, path: str, data: bytes, unique: bool = False) -> dict[str, Any]:
        query = parse.urlencode({"path": path, "unique": "true" if unique else "false"})
        return self.request("PUT", f"/sessions/{session_id}/files?{query}", data=data, timeout=300)

    def preview(self, session_id: str, path: str) -> tuple[str, bytes]:
        """(content type, body): JSON for text-like previews, the file itself for images and PDFs."""
        http_request = request.Request(
            f"{self.base_url}/sessions/{session_id}/preview?{parse.urlencode({'path': path})}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with request.urlopen(http_request, timeout=240) as response:
                return response.headers.get("Content-Type", "application/octet-stream"), response.read()
        except error.HTTPError as http_error:
            raw = http_error.read().decode("utf-8", errors="replace")
            try:
                raw = str(json.loads(raw).get("detail", raw))
            except (json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(raw) from http_error
        except error.URLError as url_error:
            raise RuntimeError(f"Sandbox niedostępny: {url_error.reason}") from url_error

    def download(self, session_id: str, path: str) -> bytes:
        http_request = request.Request(
            f"{self.base_url}/sessions/{session_id}/files?{parse.urlencode({'path': path})}",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        try:
            with request.urlopen(http_request, timeout=120) as response:
                return response.read()
        except error.HTTPError as http_error:
            raise RuntimeError(f"Sandbox HTTP {http_error.code}: {http_error.read().decode('utf-8', errors='replace')}") from http_error
        except error.URLError as url_error:
            raise RuntimeError(f"Sandbox niedostępny: {url_error.reason}") from url_error
