from __future__ import annotations

import json
from typing import Any, Callable

SEPARATORS = " \t\r\n,"


def wrap_entities_summary(content: str) -> tuple[str | None, str]:
    """Turn 'summary text {json} {json}' into {"entities": [...], "summary": "..."}.

    Returns (new_content, "") on success or (None, reason) when the answer is left untouched.
    """
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict) and "entities" in parsed:
        return None, "już w formacie entities"
    if isinstance(parsed, list):
        if not all(isinstance(item, dict) for item in parsed):
            return None, "lista JSON zawiera nie-obiekty"
        return json.dumps({"entities": parsed, "summary": ""}, ensure_ascii=False), ""

    start = text.find("{")
    if start < 0:
        return None, "brak obiektu JSON"
    summary = text[:start].strip()
    decoder = json.JSONDecoder()
    entities: list[dict[str, Any]] = []
    position = start
    while True:
        while position < len(text) and text[position] in SEPARATORS:
            position += 1
        if position >= len(text):
            break
        if text[position] != "{":
            return None, "tekst po obiekcie JSON"
        try:
            item, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return None, "niepoprawny JSON"
        entities.append(item)
    return json.dumps({"entities": entities, "summary": summary}, ensure_ascii=False), ""


def reformat_json(content: str, indent: int | None) -> tuple[str | None, str]:
    """Re-serialize an answer that is exactly one JSON value; key order and values are preserved."""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None, "odpowiedź nie jest czystym JSON"
    separators = None if indent else (",", ":")
    formatted = json.dumps(parsed, ensure_ascii=False, indent=indent, separators=separators)
    if formatted == content:
        return None, "już w tym formacie"
    return formatted, ""


TRANSFORMS: dict[str, Callable[[str], tuple[str | None, str]]] = {
    "wrap_entities_summary": wrap_entities_summary,
    "pretty_json": lambda content: reformat_json(content, 2),
    "compact_json": lambda content: reformat_json(content, None),
}


def transform_messages(messages: list[dict[str, Any]], name: str) -> tuple[list[dict[str, Any]] | None, str]:
    """Apply a transform to every assistant message; returns (new_messages, "") if anything changed."""
    transform = TRANSFORMS[name]
    changed = False
    reasons = []
    result = []
    for message in messages:
        if message.get("role") != "assistant":
            result.append(message)
            continue
        new_content, reason = transform(str(message.get("content", "")))
        if new_content is None:
            reasons.append(reason)
            result.append(message)
        else:
            changed = True
            result.append({**message, "content": new_content})
    return (result, "") if changed else (None, reasons[0] if reasons else "brak odpowiedzi asystenta")
