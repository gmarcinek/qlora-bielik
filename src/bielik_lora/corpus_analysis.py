"""Corpus analysis without a model: class/type balance and core quality checks of SFT examples."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from statistics import median
from typing import Any

from bielik_lora.evaluation import LIST_FIELDS, answer_items, parse_answer

SPLITS = ("train", "validation", "test")
FLAGS = ("positive", "negative", "unclassified")
# Task kind of an example: extraction = JSON with a list of items, classification = one label, generation = free text
# (instructions, definitions, QA, rewrites). Stored in metadata.task or detected from the answer.
TASKS = ("extraction", "classification", "generation")
LABEL_KEYS = ("label", "etykieta", "kategoria", "category", "class", "klasa")
MAX_LABEL_CHARS = 60
# Rough Polish chars-per-token for the Bielik tokenizer; lengths are estimates, not exact token counts.
CHARS_PER_TOKEN = 3.0
MIN_EXAMPLES_PER_TYPE = 10
NEGATIVE_SHARE_RANGE = (0.2, 0.6)
# Fields that must be verbatim quotes; "evidence" is excluded because corpora use it for source locations.
QUOTE_FIELDS = ("dowód", "dowod", "quote", "cytat", "text")
TYPE_TOKEN = re.compile(r"\b[A-Z][A-Z0-9_]{1,}\b")
MAX_LISTED = 200
NO_TYPE = "(bez typu)"
CHECKS = {
    "invalid_json": "Odpowiedź nie jest poprawnym obiektem JSON",
    "missing_keys": "Brak kluczy wspólnych dla korpusu",
    "flag_mismatch": "Flaga sprzeczna z odpowiedzią",
    "quote_not_in_text": "Cytat (dowód/text) nie występuje dosłownie w wiadomości użytkownika",
    "type_not_requested": "Typ elementu inny niż wskazany w poleceniu",
    "too_long": "Przykład dłuższy niż max_length treningu (szacunek)",
    "split_leakage": "Ten sam fragment źródła w train i validation/test",
    "duplicate_user": "Identyczna wiadomość użytkownika w kilku przykładach",
    "unknown_type": "Typ lub etykieta spoza słownika korpusu",
}


def normalized(text: str) -> str:
    return " ".join(text.split())


def instruction_part(user: str) -> str:
    """Polecenie to pierwszy akapit; dalej zwykle stoi fragment źródła."""
    return user.split("\n\n", 1)[0]


def source_fragment(user: str) -> str:
    parts = user.split("\n\n", 1)
    fragment = parts[1] if len(parts) == 2 else ""
    return normalized(fragment).casefold() if len(fragment) >= 30 else ""


def exchanges(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    pairs, user = [], ""
    for message in messages:
        if message.get("role") == "user":
            user = str(message.get("content", ""))
        elif message.get("role") == "assistant":
            pairs.append((user, str(message.get("content", ""))))
    return pairs


def example_types(items: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("type")) for item in items if item.get("type")})


def looks_like_json(text: str) -> bool:
    return text.strip().startswith("{")


def classification_label(text: str, parsed: dict[str, Any] | None, vocabulary: set[str] | None = None) -> str | None:
    """Label of a classification answer: a label key in JSON, or a short plain answer (a vocabulary entry if one is set)."""
    if parsed is not None:
        return next((str(parsed[key]) for key in LABEL_KEYS if isinstance(parsed.get(key), str) and parsed[key].strip()), None)
    plain = text.strip().strip(".")
    if vocabulary:
        return plain if plain in vocabulary else None
    return None


def detect_task(text: str, parsed: dict[str, Any] | None, explicit: str | None = None, vocabulary: set[str] | None = None) -> str:
    if explicit in TASKS:
        return explicit
    if parsed is not None and any(isinstance(parsed.get(field), list) for field in LIST_FIELDS):
        return "extraction"
    if classification_label(text, parsed, vocabulary):
        return "classification"
    return "generation"


def analyze(
    rows: list[dict[str, Any]],
    max_tokens: int | None = None,
    listed: int | None = MAX_LISTED,
    vocabulary: set[str] | None = None,
) -> dict[str, Any]:
    """rows: {id, split, flag, messages, task?}. Returns balance, length stats and issues with example ids.
    Extraction-specific checks (JSON keys, quotes, item types, flag vs. empty list) apply only to extraction examples."""
    parsed: list[dict[str, Any]] = []
    known_types: set[str] = set()
    key_counts: Counter[str] = Counter()
    extraction_answers = 0
    for row in rows:
        pairs = exchanges(row["messages"])
        answers = [parse_answer(answer) if looks_like_json(answer) else None for _, answer in pairs]
        last = answers[-1] if answers else None
        last_text = pairs[-1][1] if pairs else ""
        task = detect_task(last_text, last, row.get("task"), vocabulary)
        # A broken JSON answer is still meant as structured output, not free text.
        if task == "generation" and looks_like_json(last_text) and last is None and not row.get("task"):
            task = "extraction"
        items = answer_items(last) if task == "extraction" else []
        known_types.update(example_types(items))
        if task == "extraction" and last is not None:
            extraction_answers += 1
            key_counts.update(last.keys())
        parsed.append({"row": row, "pairs": pairs, "answers": answers, "items": items, "task": task, "last": last})
    common_keys = sorted(key for key, count in key_counts.items() if count >= 0.9 * extraction_answers) if extraction_answers else []

    issues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    types_table: dict[str, Counter[str]] = defaultdict(Counter)
    labels_table: dict[str, Counter[str]] = defaultdict(Counter)
    task_counts: Counter[str] = Counter()
    totals: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    lengths: list[int] = []
    with_system = 0
    exchange_counts: Counter[int] = Counter()
    fragments: dict[str, list[tuple[str, str]]] = defaultdict(list)
    users: dict[str, list[str]] = defaultdict(list)

    for entry in parsed:
        row, pairs, answers, items = entry["row"], entry["pairs"], entry["answers"], entry["items"]
        example_id, split = str(row["id"]), row["split"]
        flag = row.get("flag") if row.get("flag") in FLAGS else "unclassified"
        totals.setdefault(split, Counter())[flag] += 1
        messages = row["messages"]
        with_system += any(message.get("role") == "system" for message in messages)
        exchange_counts[len(pairs)] += 1
        tokens = round(sum(len(str(message.get("content", ""))) for message in messages) / CHARS_PER_TOKEN)
        lengths.append(tokens)
        first_user = pairs[0][0] if pairs else ""
        users[normalized(first_user)].append(example_id)
        if fragment := source_fragment(first_user):
            fragments[fragment].append((example_id, split))

        last_user = pairs[-1][0] if pairs else ""
        task = entry["task"]
        task_counts[task] += 1
        if task == "classification":
            answer_text = (pairs[-1][1] if pairs else "").strip()
            label = classification_label(answer_text, entry["last"], vocabulary) or (
                answer_text if entry["last"] is None and 0 < len(answer_text) <= MAX_LABEL_CHARS else NO_TYPE
            )
            labels_table[label][flag] += 1
            labels_table[label][split] += 1
            if vocabulary and label not in vocabulary:
                issues["unknown_type"].append({"id": example_id, "detail": f"etykieta: {label}"})
        elif task == "generation" and not (pairs and pairs[-1][1].strip()):
            issues["flag_mismatch"].append({"id": example_id, "detail": "pusta odpowied\u017a"})
        if task == "extraction":
            requested = sorted(set(TYPE_TOKEN.findall(instruction_part(last_user))) & known_types)
            types = example_types(items)
            for type_name in (types if flag == "positive" else requested or types) or [NO_TYPE]:
                types_table[type_name][flag] += 1
                types_table[type_name][split] += 1
            for index, answer in enumerate(answers):
                if answer is None:
                    issues["invalid_json"].append({"id": example_id, "detail": f"odpowied\u017a {index + 1}"})
                elif missing := [key for key in common_keys if key not in answer]:
                    issues["missing_keys"].append({"id": example_id, "detail": ", ".join(missing)})
            if flag == "positive" and answers and answers[-1] is not None and not items:
                issues["flag_mismatch"].append({"id": example_id, "detail": "pozytywny bez element\u00f3w"})
            if flag == "negative" and items:
                issues["flag_mismatch"].append({"id": example_id, "detail": f"negatywny z elementami: {', '.join(types)}"})
            user_text = normalized(" ".join(user for user, _ in pairs))
            for item in items:
                for field in QUOTE_FIELDS:
                    value = item.get(field)
                    if isinstance(value, str) and value.strip():
                        quote = normalized(value).strip("\"'\u201e\u201d\u2026. ")
                        if quote and quote not in user_text:
                            issues["quote_not_in_text"].append({"id": example_id, "detail": f"{field}: {value[:120]}"})
            if requested and items and (wrong := [name for name in types if name not in requested]):
                issues["type_not_requested"].append(
                    {"id": example_id, "detail": f"polecenie: {', '.join(requested)}; odpowied\u017a: {', '.join(wrong)}"}
                )
            if vocabulary and (outside := [name for name in types if name not in vocabulary]):
                issues["unknown_type"].append({"id": example_id, "detail": ", ".join(outside)})
        if max_tokens and tokens > max_tokens:
            issues["too_long"].append({"id": example_id, "detail": f"~{tokens} tokenów > {max_tokens}"})

    for members in fragments.values():
        splits = {split for _, split in members}
        if "train" in splits and splits & {"validation", "test"}:
            train_ids = [example_id for example_id, split in members if split == "train"]
            for example_id, split in members:
                if split != "train":
                    issues["split_leakage"].append({"id": example_id, "detail": f"{split}; w train: {len(train_ids)} (np. {train_ids[0]})"})
    for text, ids in users.items():
        if text and len(ids) > 1:
            issues["duplicate_user"] += [{"id": example_id, "detail": f"{len(ids)} kopii"} for example_id in ids]

    type_rows, warnings = balance_rows(types_table, vocabulary, negatives=True)
    label_rows, label_warnings = balance_rows(labels_table, vocabulary, negatives=False)
    warnings += [f"etykieta {item}" for item in label_warnings]
    flag_totals = Counter()
    for counts in totals.values():
        flag_totals.update(counts)
    labelled = flag_totals["positive"] + flag_totals["negative"]
    if labelled and not NEGATIVE_SHARE_RANGE[0] <= flag_totals["negative"] / labelled <= NEGATIVE_SHARE_RANGE[1]:
        warnings.insert(0, f"Ca\u0142y korpus: udzia\u0142 negatyw\u00f3w {flag_totals['negative'] / labelled:.0%}")
    if rows and not any(totals.get("validation", Counter()).values()):
        warnings.insert(0, "Ca\u0142y korpus: pusty split validation")
    ordered = sorted(lengths)
    return {
        "examples": len(rows),
        "answer_format": {"json": bool(extraction_answers), "common_keys": common_keys},
        "tasks": {task: task_counts[task] for task in TASKS},
        "splits": {split: {flag: counts[flag] for flag in FLAGS} for split, counts in totals.items()},
        "flags": {flag: flag_totals[flag] for flag in FLAGS},
        "types": type_rows,
        "labels": label_rows,
        "warnings": warnings,
        "system_prompt": {"with": with_system, "without": len(rows) - with_system},
        "exchanges": {str(count): number for count, number in sorted(exchange_counts.items())},
        "length_tokens": {
            "estimate_chars_per_token": CHARS_PER_TOKEN,
            "max_length": max_tokens,
            "median": round(median(ordered)) if ordered else 0,
            "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0,
            "max": ordered[-1] if ordered else 0,
        },
        "issues": {
            check: {"label": CHECKS[check], "count": len(found), "examples": found[:listed]}
            for check, found in ((check, issues.get(check, [])) for check in CHECKS)
        },
    }


def balance_rows(
    table: dict[str, Counter[str]], vocabulary: set[str] | None, negatives: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Per type (extraction) or label (classification) balance; negative-share rules only make sense for extraction."""
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for name, counts in sorted(table.items(), key=lambda item: -sum(item[1][flag] for flag in FLAGS)):
        total = sum(counts[flag] for flag in FLAGS)
        labelled = counts["positive"] + counts["negative"]
        negative_share = round(counts["negative"] / labelled, 3) if labelled else None
        row_warnings = []
        if vocabulary and name != NO_TYPE and name not in vocabulary:
            row_warnings.append("spoza słownika")
        if total < MIN_EXAMPLES_PER_TYPE:
            row_warnings.append(f"mało przykładów ({total} < {MIN_EXAMPLES_PER_TYPE})")
        if negatives and name != NO_TYPE:
            if not counts["negative"]:
                row_warnings.append("brak negatywów")
            elif not counts["positive"]:
                row_warnings.append("brak pozytywów")
            elif negative_share is not None and not NEGATIVE_SHARE_RANGE[0] <= negative_share <= NEGATIVE_SHARE_RANGE[1]:
                row_warnings.append(
                    f"udział negatywów {negative_share:.0%} poza {NEGATIVE_SHARE_RANGE[0]:.0%}–{NEGATIVE_SHARE_RANGE[1]:.0%}"
                )
        if counts["train"] and not counts["validation"]:
            row_warnings.append("brak w validation")
        rows.append(
            {
                "type": name,
                "total": total,
                **{flag: counts[flag] for flag in FLAGS},
                **{split: counts[split] for split in SPLITS},
                "negative_share": negative_share,
                "warnings": row_warnings,
            }
        )
        warnings += [f"{name}: {item}" for item in row_warnings]
    return rows, warnings


def example_issues(
    rows: list[dict[str, Any]], example_id: str, max_tokens: int | None = None, vocabulary: set[str] | None = None
) -> list[dict[str, str]]:
    """Issues of one example; rows must be its corpus because leakage and duplicates are cross-example checks."""
    return [
        {"check": check, "label": item["label"], "detail": entry["detail"]}
        for check, item in analyze(rows, max_tokens, listed=None, vocabulary=vocabulary)["issues"].items()
        for entry in item["examples"]
        if entry["id"] == example_id
    ]


def minimal_summary(analysis: dict[str, Any], ids_per_check: int = 10) -> dict[str, Any]:
    """Cheap default for the orchestrator: split shares and which examples are defective."""
    total = analysis["examples"] or 1
    split_counts = {split: sum(counts.values()) for split, counts in analysis["splits"].items()}
    defective = {entry["id"] for item in analysis["issues"].values() for entry in item["examples"]}
    return {
        "examples": analysis["examples"],
        "tasks": analysis["tasks"],
        "splits": {split: {"count": count, "pct": round(100 * count / total, 1)} for split, count in split_counts.items()},
        "defective": {
            "examples": len(defective),
            "by_check": {
                check: {"count": item["count"], "ids": list(dict.fromkeys(entry["id"] for entry in item["examples"]))[:ids_per_check]}
                for check, item in analysis["issues"].items()
                if item["count"]
            },
        },
        "balance_warnings": len(analysis["warnings"]),
    }


def balance_summary(analysis: dict[str, Any], examples_per_issue: int = 5) -> dict[str, Any]:
    """Compact view for the agent: balance and issue counts without long id lists."""
    return {
        "examples": analysis["examples"],
        "splits": analysis["splits"],
        "flags": analysis["flags"],
        "tasks": analysis["tasks"],
        "types": analysis["types"],
        "labels": analysis["labels"],
        "warnings": analysis["warnings"],
        "system_prompt": analysis["system_prompt"],
        "exchanges": analysis["exchanges"],
        "length_tokens": analysis["length_tokens"],
        "issues": {
            check: {"count": item["count"], "examples": [entry["id"] for entry in item["examples"][:examples_per_issue]]}
            for check, item in analysis["issues"].items()
            if item["count"]
        },
    }
