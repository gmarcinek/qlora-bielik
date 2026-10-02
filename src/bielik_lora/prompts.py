"""Prompts kept as YAML in prompts/ (repo root, or PROMPTS_DIR), read on every use so edits apply without restart."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR") or Path(__file__).resolve().parents[2] / "prompts")


def load_sections(filename: str, sections: tuple[str, ...]) -> dict[str, str]:
    data = yaml.safe_load((PROMPTS_DIR / filename).read_text(encoding="utf-8"))
    missing = [key for key in sections if not isinstance(data, dict) or not str(data.get(key) or "").strip()]
    if missing:
        raise RuntimeError(f"prompts/{filename}: brak sekcji {', '.join(missing)}.")
    return {key: str(data[key]).strip() for key in sections}


def orchestrator_prompts() -> dict[str, str]:
    return load_sections("orkiestrator.yml", ("system", "corpus_context", "sandbox", "tool_envelope"))


def reader_prompts() -> dict[str, str]:
    return load_sections("reader.yml", ("reader", "guidance", "head", "synthesis"))
