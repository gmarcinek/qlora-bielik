"""Preference (DPO) pairs from plain documents: chunk the text, back-translate an instruction for each chunk,
keep the authentic chunk as the chosen answer and a model's own answer as the rejected one."""

from __future__ import annotations

import re
from typing import Any

HEADING = re.compile(
    r"^(#{1,6}\s|(rozdzia\u0142|cz\u0119\u015b\u0107|chapter|part|ksi\u0119ga|akt|scena)\b|[IVXLC]+\.\s|\d+(\.\d+)*\.?\s+\S)",
    re.IGNORECASE,
)
SENTENCE_END = re.compile(r"(?<=[.!?\u2026])\s+")
INSTRUCTION_SCHEMA = {
    "type": "object",
    "properties": {"instruction": {"type": "string"}},
    "required": ["instruction"],
    "additionalProperties": False,
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}
BACKTRANSLATION_PROMPT = (
    "Tworzysz dane treningowe dla modelu j\u0119zykowego. Dostajesz fragment tekstu. Napisz po polsku polecenie "
    "u\u017cytkownika, na kt\u00f3re ten fragment by\u0142by idealn\u0105, pe\u0142n\u0105 odpowiedzi\u0105. Polecenie ma opisywa\u0107 zadanie "
    "(temat, form\u0119, d\u0142ugo\u015b\u0107, kluczowe elementy tre\u015bci i styl), ale nie mo\u017ce cytowa\u0107 fragmentu ani powtarza\u0107 "
    "jego charakterystycznych sformu\u0142owa\u0144. Odpowiedz wy\u0142\u0105cznie obiektem JSON z polem instruction."
)


def split_long(paragraph: str, max_chars: int) -> list[str]:
    """Paragraph longer than the limit is cut at sentence ends (a single overlong sentence stays whole)."""
    parts: list[str] = []
    current = ""
    for sentence in SENTENCE_END.split(paragraph):
        if current and len(current) + 1 + len(sentence) > max_chars:
            parts.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return [*parts, *([current] if current else [])]


def chunk_text(text: str, max_chars: int = 1500, min_chars: int = 200) -> list[str]:
    """Paragraph-aware chunks up to max_chars; a heading (chapter, numbered section, Markdown #) starts a new chunk."""
    paragraphs = [block.strip() for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")) if block.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if HEADING.match(paragraph) and current:
            chunks.append(current)
            current = ""
        for part in split_long(paragraph, max_chars) if len(paragraph) > max_chars else [paragraph]:
            if current and len(current) + 2 + len(part) > max_chars:
                chunks.append(current)
                current = part
            else:
                current = f"{current}\n\n{part}" if current else part
    if current:
        chunks.append(current)
    merged: list[str] = []
    for chunk in chunks:
        # A short tail (e.g. a lone heading) is glued to its neighbour instead of becoming its own pair.
        if merged and len(chunk) < min_chars and len(merged[-1]) + 2 + len(chunk) <= max_chars * 1.5:
            merged[-1] = f"{merged[-1]}\n\n{chunk}"
        else:
            merged.append(chunk)
    if len(merged) > 1 and len(merged[0]) < min_chars:
        merged[1] = f"{merged[0]}\n\n{merged[1]}"
        merged.pop(0)
    return merged


def backtranslation_messages(chunk: str, hint: str = "") -> list[dict[str, str]]:
    guidance = f"\n\nWskaz\u00f3wki u\u017cytkownika do polecenia: {hint.strip()}" if hint.strip() else ""
    return [
        {"role": "system", "content": BACKTRANSLATION_PROMPT + guidance},
        {"role": "user", "content": f"<fragment>\n{chunk}\n</fragment>"},
    ]


def answer_messages(instruction: str, system: str = "") -> list[dict[str, str]]:
    return [
        *([{"role": "system", "content": system}] if system.strip() else []),
        {"role": "user", "content": instruction + "\n\nOdpowiedz wy\u0142\u0105cznie obiektem JSON z polem answer zawieraj\u0105cym ca\u0142\u0105 odpowied\u017a."},
    ]


def preference_messages(instruction: str, chosen: str, system: str = "") -> list[dict[str, str]]:
    return [
        *([{"role": "system", "content": system}] if system.strip() else []),
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": chosen},
    ]


def dpo_record(messages: list[dict[str, Any]], rejected: str) -> dict[str, Any]:
    """TRL conversational preference format: prompt turns, then chosen/rejected assistant turns."""
    return {
        "prompt": [message for message in messages[:-1]],
        "chosen": [{"role": "assistant", "content": messages[-1]["content"]}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }
