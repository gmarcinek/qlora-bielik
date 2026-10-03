"""Per-conversation sandbox sessions: files, shell, scripts, document reading and handoff notes."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator
from uuid import UUID

from sandbox import attachments, large_read
from sandbox.converters import BINARY_DOCUMENTS, ConversionError, convert, detect_encoding

SESSION_DIRS = ("uploads", "converted", "scripts", "notes", "work", "exports")
FILE_AREAS = {
    "uploads": "uploads",
    "work": "work",
    "exports": "exports",
    "converted": "converted",
    "notes": "notes",
    "scripts": "scripts",
    "shared": "/shared",
    "all": ".",
}
# Single ceiling for anything returned into the model context; longer content is read in parts via read_lines.
CONTEXT_LIMIT_CHARS = int(os.getenv("CONTEXT_LIMIT_CHARS", "3500000"))
ENCODING_SAMPLE_BYTES = 4 * 1024 * 1024
MAX_TIMEOUT = 3600
SCRIPT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
TOOL_NAMES = (
    "shell",
    "run_script",
    "read_file",
    "read_lines",
    "search_lines",
    "write_file",
    "list_files",
    "convert_document",
    "add_note",
    "list_attachments",
    "attach",
    "large_read_start",
    "large_read_log",
    "large_read_note",
    "large_read_record",
    "large_read_finish",
)
STATUS_CODES = {400: "invalid_arguments", 404: "not_found", 409: "conflict", 413: "too_large", 415: "unsupported_format", 422: "conversion_failed"}


class SandboxError(Exception):
    def __init__(self, message: str, status: int = 400, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code or STATUS_CODES.get(status, "sandbox_error")


class Session:
    def __init__(self, root: Path, shared: Path | None, session_id: str) -> None:
        try:
            self.id = str(UUID(session_id))
        except ValueError as error:
            raise SandboxError("Identyfikator sesji musi być UUID.") from error
        self.path = (root / self.id).resolve()
        self.shared = shared.resolve() if shared else None
        self.meta_path = self.path / "session.json"

    # ---- lifecycle -------------------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            raise SandboxError("Sesja nie istnieje.", 404)
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def open(self, meta: dict[str, Any] | None = None) -> dict[str, Any]:
        for name in SESSION_DIRS:
            (self.path / name).mkdir(parents=True, exist_ok=True)
        current = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {
            "id": self.id,
            "created_at": time.time(),
        }
        current.update({"closed_at": None, "opened_at": time.time(), **({"meta": meta} if meta else {})})
        self.meta_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        self.migrate_inputs()
        return self.info()

    def migrate_inputs(self) -> None:
        """Sessions from before the attachments manifest kept user files in inputs/; move them to uploads/ and register them."""
        legacy = self.path / "inputs"
        if not legacy.is_dir():
            return
        known = {item["path"] for item in attachments.manifest(self)}
        for source in sorted(item for item in legacy.rglob("*") if item.is_file()):
            target = self.path / "uploads" / source.name
            counter = 2
            while target.exists():
                target = target.with_name(f"{source.stem} ({counter}){source.suffix}")
                counter += 1
            source.replace(target)
            if self.display(target) not in known:
                attachments.attach(self, self.display(target))
        for directory in sorted((item for item in legacy.rglob("*") if item.is_dir()), reverse=True):
            directory.rmdir()
        legacy.rmdir()

    def close(self) -> dict[str, Any]:
        current = self.meta()
        current["closed_at"] = time.time()
        self.meta_path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.info()

    def delete(self) -> dict[str, Any]:
        if not self.path.is_dir():
            raise SandboxError("Sesja nie istnieje.", 404)
        shutil.rmtree(self.path)
        return {"id": self.id, "deleted": True}

    def info(self) -> dict[str, Any]:
        return {
            **self.meta(),
            "files": self.list_files(".", "**/*")["files"],
            "attachments": attachments.manifest(self),
        }

    def require_open(self) -> None:
        if self.meta().get("closed_at"):
            raise SandboxError("Sesja jest zamknięta.", 409)

    # ---- paths -----------------------------------------------------------------------------
    def resolve(self, path: str, write: bool = False) -> Path:
        raw = (path or ".").strip()
        if self.shared and (raw == "/shared" or raw.startswith("/shared/")):
            if write:
                raise SandboxError("/shared jest tylko do odczytu.")
            base, relative = self.shared, raw[len("/shared"):].lstrip("/")
        else:
            base = self.path
            relative = (raw[len(str(self.path)):] if raw.startswith(str(self.path)) else raw).lstrip("/")
        resolved = (base / relative).resolve()
        if resolved != base and base not in resolved.parents:
            raise SandboxError("Ścieżka wychodzi poza katalog sesji.")
        return resolved

    def display(self, path: Path) -> str:
        if self.shared and (path == self.shared or self.shared in path.parents):
            return "/shared/" + path.relative_to(self.shared).as_posix()
        return path.relative_to(self.path).as_posix()

    def existing_file(self, path: str) -> Path:
        resolved = self.resolve(path)
        if not resolved.is_file():
            raise SandboxError(f"Nie ma pliku {path}.", 404)
        return resolved

    # ---- execution -------------------------------------------------------------------------
    def environment(self) -> dict[str, str]:
        return {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(self.path),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUNBUFFERED": "1",
            "SESSION_DIR": str(self.path),
            "SHARED_DIR": str(self.shared or ""),
        }

    def execute(self, argv: list[str], timeout_seconds: int) -> dict[str, Any]:
        self.require_open()
        timeout = max(1, min(int(timeout_seconds), MAX_TIMEOUT))
        started = time.monotonic()
        process = subprocess.Popen(
            argv,
            cwd=self.path,
            env=self.environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            stdout, stderr = process.communicate()
        result: dict[str, Any] = {
            "exit_code": process.returncode,
            "timed_out": timed_out,
            "seconds": round(time.monotonic() - started, 2),
        }
        for name, data in (("stdout", stdout), ("stderr", stderr)):
            text = data.decode("utf-8", errors="replace")
            if len(text) > CONTEXT_LIMIT_CHARS:
                spill = self.path / "work" / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{process.pid}-{name}.txt"
                spill.parent.mkdir(parents=True, exist_ok=True)
                spill.write_text(text, encoding="utf-8")
                result[f"{name}_file"] = self.display(spill)
                result[f"{name}_note"] = f"Wyjście ma {len(text)} znaków; całość w {self.display(spill)} — czytaj dalej read_lines."
                text = text[:CONTEXT_LIMIT_CHARS]
            result[name] = text
        return result

    def shell(self, command: str, timeout_seconds: int = 60) -> dict[str, Any]:
        if not command.strip():
            raise SandboxError("Puste polecenie.")
        return self.execute(["bash", "-c", command], timeout_seconds)

    def run_script(
        self,
        code: str,
        language: str = "python",
        name: str | None = None,
        args: list[str] | None = None,
        timeout_seconds: int = 120,
    ) -> dict[str, Any]:
        self.require_open()
        if language not in {"python", "bash"}:
            raise SandboxError("language musi być python albo bash.")
        suffix = ".py" if language == "python" else ".sh"
        scripts = self.path / "scripts"
        if name:
            if not SCRIPT_NAME.match(name):
                raise SandboxError("Nazwa skryptu: litery, cyfry, _ . - (bez katalogów).")
            filename = name if name.endswith(suffix) else f"{name}{suffix}"
        else:
            filename = f"script-{len(list(scripts.glob('script-*'))) + 1:03d}{suffix}"
        (scripts / filename).write_text(code, encoding="utf-8")
        interpreter = "python3" if language == "python" else "bash"
        return {"script": f"scripts/{filename}", **self.execute([interpreter, f"scripts/{filename}", *(args or [])], timeout_seconds)}

    # ---- files -----------------------------------------------------------------------------
    def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        self.require_open()
        target = self.resolve(path, write=True)
        if target == self.meta_path:
            raise SandboxError("Nie można nadpisać session.json.")
        data = content.encode("utf-8")
        if len(content) > CONTEXT_LIMIT_CHARS:
            raise SandboxError(f"Treść przekracza {CONTEXT_LIMIT_CHARS} znaków; zapisuj w częściach (append).")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("ab" if append else "wb") as handle:
            handle.write(data)
        return {"path": self.display(target), "bytes": target.stat().st_size}

    def save_upload(self, path: str, chunks: Iterable[bytes], max_bytes: int, unique: bool = False) -> dict[str, Any]:
        self.require_open()
        target = self.resolve(path, write=True)
        if unique:
            stem, suffix, counter = target.stem, target.suffix, 2
            while target.exists():
                target = target.with_name(f"{stem} ({counter}){suffix}")
                counter += 1
        target.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with target.open("wb") as handle:
            for chunk in chunks:
                written += len(chunk)
                if written > max_bytes:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise SandboxError("Plik jest za duży.", 413)
                handle.write(chunk)
        return {"path": self.display(target), "bytes": written}

    def list_files(
        self, path: str = ".", pattern: str = "**/*", limit: int | None = None, area: str | None = None
    ) -> dict[str, Any]:
        if area:
            if area not in FILE_AREAS:
                raise SandboxError(f"Nieznany obszar {area}; dostępne: {', '.join(FILE_AREAS)}.")
            path = FILE_AREAS[area]
        base = self.resolve(path)
        if not base.is_dir():
            raise SandboxError(f"Nie ma katalogu {path}.", 404)
        files = []
        for item in sorted(base.glob(pattern or "*")):
            if item.is_file():
                files.append({"path": self.display(item), "bytes": item.stat().st_size})
                if limit and len(files) >= limit:
                    break
        return {"files": files, "truncated": bool(limit) and len(files) >= limit}

    # ---- documents -------------------------------------------------------------------------
    def converted_path(self, source: Path) -> Path:
        relative = Path("shared") / source.relative_to(self.shared) if self.shared and self.shared in source.parents else source.relative_to(self.path)
        return self.path / "converted" / f"{relative.as_posix()}.md"

    def convert_document(self, path: str, force: bool = False) -> dict[str, Any]:
        source = self.existing_file(path)
        if self.path / "converted" in source.parents:
            raise SandboxError("To już jest plik po konwersji.")
        target = self.converted_path(source)
        meta_path = target.with_name(target.name + ".json")
        stat = source.stat()
        if not force and target.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("source_bytes") == stat.st_size and meta.get("source_mtime") == stat.st_mtime:
                return {**meta, "cached": True}
        try:
            text, details = convert(source)
        except ConversionError as error:
            raise SandboxError(f"Konwersja nie powiodła się: {error}", 422) from error
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        meta = {
            "source": self.display(source),
            "output": self.display(target),
            "source_bytes": stat.st_size,
            "source_mtime": stat.st_mtime,
            "chars": len(text),
            "lines": text.count("\n") + 1,
            **details,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {**meta, "cached": False}

    def readable(self, path: str) -> tuple[Path, str, dict[str, Any]]:
        """File whose lines are read and its encoding; binary documents are read through their Markdown conversion."""
        source = self.existing_file(path)
        if source.suffix.lower() in BINARY_DOCUMENTS:
            conversion = self.convert_document(path)
            return self.resolve(conversion["output"]), "utf-8", {"converted_from": conversion["source"], "method": conversion.get("method")}
        with source.open("rb") as handle:
            sample = handle.read(ENCODING_SAMPLE_BYTES)
        if b"\x00" in sample[:4096] and not sample.startswith((b"\xff\xfe", b"\xfe\xff")):
            raise SandboxError("Plik binarny w nieobsługiwanym formacie; spróbuj shell (file, strings) lub skryptu.", 415)
        encoding = detect_encoding(sample)
        return source, encoding, ({"encoding": encoding} if encoding not in {"utf-8", "ascii"} else {})

    @staticmethod
    def iter_lines(path: Path, encoding: str) -> Iterator[str]:
        with path.open("r", encoding=encoding, errors="replace", newline=None) as handle:
            for line in handle:
                yield line.rstrip("\n").replace("\x00", "")

    def collect_lines(
        self, path: str, first: int, last: int | None, numbered: bool, max_chars: int | None = None
    ) -> dict[str, Any]:
        """Whole lines first..last streamed from disk, stopping only at the context limit (or a smaller max_chars)."""
        limit = min(CONTEXT_LIMIT_CHARS, int(max_chars)) if max_chars else CONTEXT_LIMIT_CHARS
        text_path, encoding, extra = self.readable(path)
        first = max(1, int(first))
        last = int(last) if last else None
        parts: list[str] = []
        used, end, total = 0, first - 1, 0
        stopped = line_truncated = False
        for number, line in enumerate(self.iter_lines(text_path, encoding), start=1):
            total = number
            if stopped or number < first or (last is not None and number > last):
                continue
            entry = f"{number:>6}| {line}" if numbered else line
            if used + len(entry) + 1 > limit:
                stopped = True
                if not parts:
                    parts.append(entry[:limit])
                    end, line_truncated = number, True
                continue
            parts.append(entry)
            used += len(entry) + 1
            end = number
        if total and first > total:
            raise SandboxError(f"Plik ma tylko {total} linii.")
        result: dict[str, Any] = {
            "path": self.display(text_path),
            **extra,
            "total_lines": total,
            "start_line": first,
            "end_line": end,
            "next_start_line": end + 1 if end < total else None,
            "content": "\n".join(parts),
        }
        if stopped:
            result["stopped_at_context_limit"] = True
        if line_truncated:
            result["note"] = f"Linia {end} przekracza {limit} znaków i została ucięta; resztę odczytaj skryptem (np. cut -c)."
        return result

    def read_file(self, path: str) -> dict[str, Any]:
        text_path, _, _ = self.readable(path)
        size = text_path.stat().st_size
        if size > large_read.LARGE_FILE_BYTES:
            raise SandboxError(
                f"Plik {self.display(text_path)} ma {size // 1024} KB (> {large_read.LARGE_FILE_BYTES // 1024} KB): "
                "użyj read_large_file_session z celem i pytaniami (formalny handoff) albo read_lines dla konkretnego zakresu.",
                413,
                "use_large_reader",
            )
        result = self.collect_lines(path, 1, None, numbered=False)
        if result["next_start_line"]:
            result["note"] = (
                f"Plik przekracza limit kontekstu {CONTEXT_LIMIT_CHARS} znaków; "
                f"czytaj dalej read_lines od linii {result['next_start_line']}."
            )
        return result

    def read_lines(
        self, path: str, start_line: int = 1, end_line: int | None = None, max_chars: int | None = None
    ) -> dict[str, Any]:
        return self.collect_lines(path, start_line, end_line, numbered=True, max_chars=max_chars)

    def search_lines(
        self,
        path: str,
        query: str,
        regex: bool = False,
        context_lines: int = 2,
        max_matches: int = 50,
        start_line: int = 1,
        end_line: int | None = None,
    ) -> dict[str, Any]:
        """Counts every matching line in the range and returns the first max_matches with surrounding context."""
        if not query.strip():
            raise SandboxError("Puste zapytanie.")
        try:
            pattern = re.compile(query if regex else re.escape(query), re.IGNORECASE)
        except re.error as error:
            raise SandboxError(f"Niepoprawny regex: {error}") from error
        text_path, encoding, extra = self.readable(path)
        context = max(0, min(int(context_lines), 20))
        limit = max(1, min(int(max_matches), 500))
        first, last = max(1, int(start_line)), int(end_line) if end_line else None
        window: list[tuple[int, str]] = []
        matches: list[dict[str, Any]] = []
        pending: list[dict[str, Any]] = []
        total = count = used = 0
        for number, line in enumerate(self.iter_lines(text_path, encoding), start=1):
            total = number
            for item in pending:
                if number <= item["line"] + context:
                    item["after"].append(f"{number:>6}| {line}")
            pending = [item for item in pending if number < item["line"] + context]
            if first <= number and (last is None or number <= last) and pattern.search(line):
                count += 1
                if len(matches) < limit and used < CONTEXT_LIMIT_CHARS:
                    item = {"line": number, "text": line, "before": [f"{n:>6}| {t}" for n, t in window], "after": []}
                    used += sum(len(entry) for entry in item["before"]) + len(line)
                    matches.append(item)
                    if context:
                        pending.append(item)
            window = [*window, (number, line)][-context:] if context else []
        return {
            "path": self.display(text_path),
            **extra,
            "total_lines": total,
            "total_matches": count,
            "returned": len(matches),
            "truncated": count > len(matches),
            "matches": [
                {"line": item["line"], "text": item["text"], "context": "\n".join([*item["before"], f"{item['line']:>6}> {item['text']}", *item["after"]])}
                for item in matches
            ],
        }

    def large_read_start(self, path: str, goal: str, questions: list[str] | None = None) -> dict[str, Any]:
        return large_read.start(self, path, goal, questions)

    def attach(self, path: str) -> dict[str, Any]:
        return attachments.attach(self, path)

    def list_attachments(self) -> dict[str, Any]:
        return {"attachments": attachments.manifest(self)}

    def large_read_log(self, read_id: str, event: dict[str, Any]) -> dict[str, Any]:
        return large_read.log(self, read_id, event)

    def large_read_record(self, read_id: str) -> dict[str, Any]:
        return large_read.load(self, read_id)

    def large_read_note(self, read_id: str, **notes: Any) -> dict[str, Any]:
        return large_read.note(self, read_id, **notes)

    def large_read_finish(
        self, read_id: str, synthesis: dict[str, Any], telemetry: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        return large_read.finish(self, read_id, synthesis, telemetry)

    # ---- handoff notes ---------------------------------------------------------------------
    def notes(self) -> list[dict[str, Any]]:
        path = self.path / "notes" / "notes.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def add_note(
        self,
        kind: str,
        text: str,
        source: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        based_on: list[str] | None = None,
    ) -> dict[str, Any]:
        self.require_open()
        if kind not in {"fact", "interpretation"}:
            raise SandboxError("kind musi być fact albo interpretation.")
        if not text.strip():
            raise SandboxError("Pusta notatka.")
        notes = self.notes()
        note: dict[str, Any] = {"kind": kind, "text": text.strip(), "created_at": time.time()}
        if kind == "fact":
            if not source or not start_line:
                raise SandboxError("Fakt wymaga źródła: source i start_line (oraz opcjonalnie end_line).")
            first, last = int(start_line), int(end_line or start_line)
            cited = self.collect_lines(source, first, last, numbered=False)
            if first > last or last > cited["total_lines"]:
                raise SandboxError(f"Zakres linii {first}-{last} poza plikiem ({cited['total_lines']} linii).")
            note.update(
                source=self.display(self.existing_file(source)),
                cited_file=cited["path"],
                start_line=first,
                end_line=last,
                quote=cited["content"],
            )
        known = {item["id"] for item in notes}
        missing = [item for item in based_on or [] if item not in known]
        if missing:
            raise SandboxError(f"Nieznane notatki w based_on: {missing}.")
        if based_on:
            note["based_on"] = based_on
        prefix = "F" if kind == "fact" else "I"
        note["id"] = f"{prefix}{sum(1 for item in notes if item['kind'] == kind) + 1}"
        with (self.path / "notes" / "notes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(note, ensure_ascii=False) + "\n")
        self.render_handoff([*notes, note])
        return {"id": note["id"], "handoff": "notes/HANDOFF.md"}

    def render_handoff(self, notes: list[dict[str, Any]]) -> None:
        lines = [f"# Handoff sesji {self.id}", "", f"_Aktualizacja: {time.strftime('%Y-%m-%d %H:%M:%S')}_", "", "## Fakty (ze źródeł)", ""]
        for note in (item for item in notes if item["kind"] == "fact"):
            location = f"`{note['cited_file']}` L{note['start_line']}–L{note['end_line']}"
            if note["source"] != note["cited_file"]:
                location = f"`{note['source']}` → {location}"
            lines += [f"- **{note['id']}** {note['text']}", f"  - źródło: {location}"]
            lines += [f"  > {quoted}" for quoted in note["quote"].split("\n")]
        lines += ["", "## Interpretacje", ""]
        for note in (item for item in notes if item["kind"] == "interpretation"):
            basis = f" _(na podstawie: {', '.join(note['based_on'])})_" if note.get("based_on") else ""
            lines.append(f"- **{note['id']}** {note['text']}{basis}")
        reads = large_read.all_reads(self)
        if reads:
            lines += ["", "## Odczyty celowe (formalne handoffy)", ""]
            for record in reads:
                synthesis = record.get("synthesis") or {}
                status = (
                    f"{synthesis.get('status', '—')}, rozstrzygnięte {len(synthesis.get('resolved', []))}, "
                    f"nierozstrzygnięte {len(synthesis.get('unresolved', []))}"
                    if record["status"] == "done"
                    else f"w toku ({len(record.get('notes', []))} notatek)"
                )
                lines.append(
                    f"- **{record['id']}** `{record['source']}` — cel: {record['goal']} — {status} — "
                    f"fakty {len(record['facts'])} — [notes/reads/{record['id']}.md](reads/{record['id']}.md)"
                )
        (self.path / "notes" / "HANDOFF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- dispatch --------------------------------------------------------------------------
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in TOOL_NAMES:
            raise SandboxError(f"Nieznane narzędzie {name}.", 404)
        self.meta()
        try:
            return getattr(self, name)(**arguments)
        except TypeError as error:
            raise SandboxError(f"Złe argumenty {name}: {error}") from error


class Workspace:
    def __init__(self, root: Path, shared: Path | None = None) -> None:
        self.root = root
        self.shared = shared if shared and shared.exists() else None
        root.mkdir(parents=True, exist_ok=True)

    def session(self, session_id: str) -> Session:
        return Session(self.root, self.shared, session_id)

    def sessions(self) -> list[dict[str, Any]]:
        """Metadata of every session folder (no file listings), for reconciling with the lab database."""
        result = []
        for path in sorted(self.root.iterdir()):
            try:
                result.append(self.session(path.name).meta())
            except (SandboxError, OSError, json.JSONDecodeError):
                continue
        return result
