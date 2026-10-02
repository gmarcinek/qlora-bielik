"""Document -> Markdown conversion with fallbacks for awkward PDFs, legacy Office files, scans and odd encodings."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".tsv", ".log", ".xml", ".yaml", ".yml", ".json", ".jsonl", ".ndjson"}
MARKITDOWN_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".epub", ".msg", ".ipynb"}
LEGACY_OFFICE = {".doc": "docx", ".rtf": "docx", ".odt": "docx", ".ppt": "pptx", ".odp": "pptx", ".ods": "xlsx"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
BINARY_DOCUMENTS = MARKITDOWN_SUFFIXES | set(LEGACY_OFFICE) | IMAGE_SUFFIXES

OCR_LANGUAGES = os.getenv("OCR_LANGUAGES", "pol+eng")
# 0 = OCR every page that lacks a usable text layer.
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "0"))
MIN_PAGE_CHARS = 40
USEFUL_CHARS = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]")
CID_GLYPHS = re.compile(r"\(cid:\d+\)")


class ConversionError(Exception):
    pass


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode text of unknown encoding (UTF-8/16 with or without BOM, cp1250, ISO-8859-2, ...)."""
    if data.startswith(b"\xef\xbb\xbf"):
        text, encoding = data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text, encoding = data.decode("utf-16", errors="replace"), "utf-16"
    else:
        try:
            text, encoding = data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            candidates: list[tuple[str, str]] = []
            try:
                from charset_normalizer import from_bytes

                best = from_bytes(data).best()
                if best is not None:
                    candidates.append((str(best), best.encoding))
            except ImportError:
                pass
            # Polish documents are usually cp1250 or ISO-8859-2; short samples fool generic detectors.
            for legacy in ("cp1250", "iso8859_2"):
                candidates.append((data.decode(legacy, errors="replace"), legacy))
            text, encoding = max(candidates, key=lambda item: useful_score(item[0]))
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", ""), encoding


def detect_encoding(sample: bytes) -> str:
    """Encoding of a file from its leading bytes (cut at a newline so UTF-8 sequences stay whole)."""
    if not sample.startswith((b"\xff\xfe", b"\xfe\xff")) and b"\n" in sample[1:]:
        sample = sample[: sample.rindex(b"\n") + 1]
    return decode_bytes(sample)[1]


def useful_score(text: str) -> int:
    cleaned = CID_GLYPHS.sub("", text)
    return len(USEFUL_CHARS.findall(cleaned)) - 10 * cleaned.count("\ufffd")


def run_tool(argv: list[str], timeout: int = 600) -> str:
    result = subprocess.run(argv, capture_output=True, timeout=timeout, check=False)
    if result.returncode != 0:
        raise ConversionError(f"{argv[0]}: {result.stderr.decode('utf-8', errors='replace').strip()[-500:]}")
    return result.stdout.decode("utf-8", errors="replace")


def markitdown_text(path: Path) -> str:
    from markitdown import MarkItDown

    return MarkItDown(enable_plugins=False).convert(str(path)).text_content or ""


def ocr_image(path: Path) -> str:
    if not shutil.which("tesseract"):
        raise ConversionError("Brak tesseract w obrazie sandboksa.")
    return run_tool(["tesseract", str(path), "stdout", "-l", OCR_LANGUAGES, "--psm", "3"])


def pdf_pymupdf(path: Path) -> tuple[str, dict[str, Any]]:
    """Per-page text; pages without a usable text layer (scans, broken ToUnicode maps) go through OCR."""
    import fitz

    document = fitz.open(str(path))
    if document.needs_pass and not document.authenticate(""):
        raise ConversionError("PDF jest zaszyfrowany hasłem.")
    pages: list[str] = []
    ocr_pages: list[int] = []
    with tempfile.TemporaryDirectory() as temporary:
        for number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True)
            stripped = text.strip()
            garbled = stripped and useful_score(stripped) < 0.5 * len(re.sub(r"\s", "", stripped))
            if (len(stripped) < MIN_PAGE_CHARS or garbled) and (not OCR_MAX_PAGES or len(ocr_pages) < OCR_MAX_PAGES) and shutil.which("tesseract"):
                image = Path(temporary) / f"page-{number}.png"
                page.get_pixmap(dpi=300).save(str(image))
                recognized = ocr_image(image)
                if useful_score(recognized) > useful_score(text):
                    text = recognized
                    ocr_pages.append(number)
            pages.append(f"<!-- strona {number} -->\n\n{text.strip()}")
    return "\n\n".join(pages), {"pages": len(pages), "ocr_pages": ocr_pages}


def pdf_candidates(path: Path) -> list[tuple[str, str, dict[str, Any]]]:
    candidates: list[tuple[str, str, dict[str, Any]]] = []
    errors: dict[str, str] = {}
    for name, extract in (
        ("pymupdf", lambda: pdf_pymupdf(path)),
        ("markitdown", lambda: (markitdown_text(path), {})),
        ("pdftotext", lambda: (run_tool(["pdftotext", "-layout", "-enc", "UTF-8", str(path), "-"]), {})),
    ):
        try:
            text, details = extract()
            candidates.append((name, text, details))
        except Exception as error:  # noqa: BLE001 - each extractor may fail on a different kind of PDF
            errors[name] = str(error)[:300]
    if not candidates:
        raise ConversionError(f"Żaden ekstraktor PDF nie zadziałał: {errors}")
    if errors:
        candidates[0][2].setdefault("errors", errors)
    return candidates


def convert_pdf(path: Path) -> tuple[str, dict[str, Any]]:
    candidates = pdf_candidates(path)
    scores = {name: useful_score(text) for name, text, _ in candidates}
    best = max(scores.values())
    # Prefer the earlier extractor (PyMuPDF keeps page markers) unless another one is clearly better.
    name, text, details = next(item for item in candidates if scores[item[0]] >= 0.97 * best)
    if details.get("ocr_pages"):
        name = f"{name}+ocr"
    return text, {"method": name, "candidate_scores": scores, **details}


def convert_legacy_office(path: Path) -> tuple[str, dict[str, Any]]:
    if not shutil.which("soffice"):
        raise ConversionError("Brak LibreOffice (soffice) do konwersji starszych formatów.")
    target = LEGACY_OFFICE[path.suffix.lower()]
    with tempfile.TemporaryDirectory() as temporary:
        run_tool(["soffice", "--headless", f"-env:UserInstallation=file://{temporary}/profile",
                  "--convert-to", target, "--outdir", temporary, str(path)])
        converted = Path(temporary) / f"{path.stem}.{target}"
        if not converted.exists():
            raise ConversionError(f"LibreOffice nie utworzył pliku {converted.name}.")
        return markitdown_text(converted), {"method": f"soffice->{target}+markitdown"}


def convert_text(path: Path) -> tuple[str, dict[str, Any]]:
    text, encoding = decode_bytes(path.read_bytes())
    details: dict[str, Any] = {"method": "decode", "encoding": encoding}
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError as error:
            details["warnings"] = [f"Niepoprawny JSON: {error.msg} (linia {error.lineno})"]
    elif suffix in {".jsonl", ".ndjson"}:
        invalid = []
        for number, line in enumerate(text.split("\n"), start=1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    invalid.append(number)
        if invalid:
            details["warnings"] = [f"Niepoprawne linie JSONL ({len(invalid)}): {invalid}"]
    return text, details


def convert(path: Path) -> tuple[str, dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        text, details = convert_pdf(path)
    elif suffix in LEGACY_OFFICE:
        text, details = convert_legacy_office(path)
    elif suffix in IMAGE_SUFFIXES:
        text, details = ocr_image(path), {"method": "tesseract"}
    elif suffix in MARKITDOWN_SUFFIXES:
        text, details = markitdown_text(path), {"method": "markitdown"}
    elif suffix in TEXT_SUFFIXES:
        text, details = convert_text(path)
    else:
        raise ConversionError(f"Nieobsługiwany format {suffix or '(brak rozszerzenia)'}.")
    text = text.replace("\r\n", "\n").replace("\x00", "")
    if useful_score(text) <= 0:
        details.setdefault("warnings", []).append("Brak czytelnego tekstu po konwersji.")
    return text, details
