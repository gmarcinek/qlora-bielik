"""Top-level executor agents the orchestrator delegates example work to.

Tool levels:
- top: agents — the orchestrator (chat), generate_examples (team of generator agents), analyze_series (analyst agent);
- medium: corpus tools (corpus_overview, corpus_balance, list_examples, list_proposals, …) and the large-file reader;
- general: sandbox and filesystem tools, available to every agent when the conversation has a sandbox;
- private: tools of one agent only — save_examples (generator), remove_proposals / regenerate_proposals (analyst).

A generator reads sources itself (general tools), looks at the corpus (medium read tools) and saves its examples with
save_examples, 20–30 per call. Up to 5 generators run in parallel. The analyst reviews a series, removes duplicates
and unsuitable proposals and calls the generators itself (generate_examples, regenerate_proposals) to fill gaps.
The orchestrator only gets short reports: how many were added or removed, proportions and a one-line manifest."""

from __future__ import annotations

import copy
import json
import queue
import random
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Generator, Iterator
from uuid import uuid4

from bielik_lora.agent import ToolError, run_agent
from bielik_lora.corpus_agent import TOOLS as CORPUS_TOOLS
from bielik_lora.corpus_agent import (
    MAX_PROPOSALS_PER_CALL,
    CorpusAgentTools,
    draft_messages,
    entity_types,
    json_object,
    last_assistant,
    message_text,
)
from bielik_lora.corpus_analysis import TASKS, analyze, classification_label, detect_task, normalized
from bielik_lora.prompts import generator_prompts

MAX_PARALLEL = 5
MAX_ASSIGNMENTS = 10
DEFAULT_COUNT = 25
MAX_COUNT = 30
MAX_SAVE = 30
MAX_EXAMPLES_PER_RUN = 300
GENERATOR_MAX_STEPS = 20
ANALYST_MAX_STEPS = 24
DIGEST_ITEMS = 800
DIGEST_CHARS = 160
MANIFEST_CHARS = 110
REFERENCE_EXAMPLES = 3
REFERENCE_CHARS = 3000
ANALYST_TEXT_CHARS = 1500
NEAR_DUPLICATE = 0.7
COMMON_SHINGLE_DOCS = 200
TEMPLATE_LINE_SHARE = 0.2
FLAGS = ("positive", "negative", "mixed")
BATCH_PATTERN = re.compile(r"^proposal-[0-9a-f-]{36}$")

# Medium-level read tools of sub-agents; they never change the corpus.
CORPUS_READ_TOOLS = [
    copy.deepcopy(tool) for tool in CORPUS_TOOLS if tool["name"] in {"corpus_overview", "corpus_balance", "list_examples", "list_proposals", "get_examples"}
]
CORPUS_READ_NAMES = {tool["name"] for tool in CORPUS_READ_TOOLS}

EXAMPLE_ITEM = copy.deepcopy(
    next(tool for tool in CORPUS_TOOLS if tool["name"] == "propose_examples")["parameters"]["properties"]["examples"]["items"]
)
EXAMPLE_ITEM["properties"].pop("replaces")
EXAMPLE_ITEM["properties"]["rejected"] = {
    "type": "string",
    "description": "Tylko w trybie DPO (wymagane): odpowiedź odrzucona do pary preferencji — wiarygodna, ale gorsza w sposób wskazany w opcjach.",
}

TRAINING_MODES = ("sft", "dpo")
TRAINING_MODE_KNOWLEDGE = {
    "sft": (
        "SFT (supervised fine-tuning): model uczy się naśladować odpowiedź assistant dla danego polecenia. Każda odpowiedź musi "
        "być wzorcowa — dokładnie taka, jaką model ma dawać: poprawna merytorycznie, kompletna, w formacie korpusu. "
        "Błąd w odpowiedzi uczy model błędu. Pole rejected pomijasz."
    ),
    "dpo": (
        "DPO (direct preference optimization): para preferencji dla tego samego polecenia — odpowiedź wybrana (assistant, wzorcowa "
        "jak w SFT) i odrzucona (rejected). Model uczy się różnicy między nimi, więc odrzucona ma być wiarygodna i bliska wybranej, "
        "ale gorsza w JEDEN konkretny, uczący sposób (np. pominięty element, informacja spoza tekstu, zły typ lub etykieta, "
        "błędna flaga, zły format lub klucze, zbędna odmowa albo odpowiedź mimo braku podstaw) — zgodnie z opcją rejected_strategy. "
        "Nie karykaturalna, nie losowo zepsuta; różnica ma być tym, czego model ma się oduczyć."
    ),
}

OPTIONS_ITEM: dict[str, Any] = {
    "type": "object",
    "description": "Opcje generowania serii.",
    "properties": {
        "negative_share": {"type": "integer", "minimum": 0, "maximum": 100, "description": "Docelowy udział negatywów (%) w zleceniach mixed."},
        "exchanges": {"type": "integer", "minimum": 1, "maximum": 2, "description": "Liczba wymian user+assistant w przykładzie."},
        "system_prompt": {"type": "string", "enum": ["corpus", "none", "custom"], "description": "Instrukcja systemowa: korpusu, brak albo własna."},
        "custom_system": {"type": "string", "description": "Własna instrukcja systemowa (system_prompt=custom)."},
        "length": {"type": "string", "enum": ["short", "medium", "long", "mixed"], "description": "Długość poleceń i materiału."},
        "difficulty": {"type": "string", "enum": ["easy", "mixed", "hard"], "description": "Trudność przypadków (hard = graniczne, mylące)."},
        "style": {"type": "string", "description": "Styl poleceń użytkownika (np. formalny, potoczny, z literówkami, różne sformułowania)."},
        "rejected_strategy": {"type": "string", "description": "DPO: w czym odpowiedź odrzucona ma być gorsza."},
        "notes": {"type": "string", "description": "Inne opcje i ograniczenia."},
    },
    "additionalProperties": False,
}

SERIES_PROPERTIES: dict[str, Any] = {
    "goal": {"type": "string", "description": "Cel serii: czego model ma się nauczyć i jakie luki korpusu wypełnić."},
    "intent": {"type": "string", "description": "Intencja użytkownika: czego naprawdę chce (z rozmowy i planu), w jego słowach i Twoim rozumieniu."},
    "context": {"type": "string", "description": "Kontekst: ustalenia z rozmowy, wnioski z przeczytanych materiałów, decyzje, czego unikać."},
    "training_mode": {"type": "string", "enum": list(TRAINING_MODES), "description": "Tryb treningu, pod który powstają przykłady (domyślnie sft)."},
    "options": OPTIONS_ITEM,
    "guidelines": {"type": "string", "description": "Wspólne wytyczne: format, styl poleceń, długości, czego unikać."},
    "material": {"type": "string", "description": "Dane: materiał źródłowy wklejony dosłownie (gdy nie ma go w pliku)."},
}

SOURCE_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Ścieżka pliku w sandboksie rozmowy (np. converted/owu.pdf.md)."},
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
        "note": {"type": "string", "description": "Co jest w tym zakresie / czego tam szukać."},
    },
    "required": ["path"],
    "additionalProperties": False,
}

ASSIGNMENT_ITEM: dict[str, Any] = {
    "type": "object",
    "properties": {
        "focus": {"type": "string", "description": "Zakres generatora: co dokładnie ma pokryć (element, sekcja, przypadek, styl), odrębny od pozostałych zleceń."},
        "count": {"type": "integer", "minimum": 1, "maximum": MAX_COUNT, "description": f"Liczba przykładów (domyślnie {DEFAULT_COUNT})."},
        "flag": {"type": "string", "enum": list(FLAGS), "description": "positive / negative / mixed (domyślnie mixed, udział negatywów jak w korpusie)."},
        "task": {"type": "string", "enum": list(TASKS)},
        "split": {"type": "string", "enum": ["train", "validation", "test"], "description": "Tylko gdy użytkownik wskazał split."},
        "material": {"type": "string", "description": "Materiał źródłowy tylko dla tego zlecenia (dosłownie)."},
        "sources": {"type": "array", "maxItems": 20, "items": SOURCE_ITEM, "description": "Źródła tylko dla tego zlecenia (plik i zakres linii)."},
    },
    "required": ["focus"],
    "additionalProperties": False,
}

SAVE_EXAMPLES_TOOL: dict[str, Any] = {
    "name": "save_examples",
    "description": (
        f"Zapisuje przykłady od razu w zakładce Propozycje korpusu (partia tej serii). Maks. {MAX_SAVE} na wywołanie — "
        "zapisuj paczkami po 20–30, aż remaining = 0. Każdy przykład przechodzi walidację (format, flaga, słownik, duplikat); "
        "odrzucone (rejected z powodem) popraw i zapisz ponownie."
    ),
    "parameters": {
        "type": "object",
        "properties": {"examples": {"type": "array", "minItems": 1, "maxItems": MAX_SAVE, "items": EXAMPLE_ITEM}},
        "required": ["examples"],
        "additionalProperties": False,
    },
    "returns": {
        "type": "object",
        "required": ["saved", "rejected", "saved_total", "target", "remaining"],
        "properties": {
            "saved": {"type": "integer"},
            "rejected": {
                "type": "array",
                "items": {"type": "object", "required": ["index", "reason"], "properties": {"index": {"type": "integer"}, "reason": {"type": "string"}}},
            },
            "warnings": {"type": "array"},
            "saved_total": {"type": "integer"},
            "target": {"type": "integer"},
            "remaining": {"type": "integer"},
        },
    },
}

GENERATE_EXAMPLES_TOOL: dict[str, Any] = {
    "name": "generate_examples",
    "description": (
        "Zleca serię przykładów zespołowi autonomicznych agentów-generatorów "
        f"(maks. {MAX_PARALLEL} równolegle, do {MAX_COUNT} przykładów na zlecenie, maks. {MAX_ASSIGNMENTS} zleceń i {MAX_EXAMPLES_PER_RUN} przykładów na wywołanie). "
        "Każdy generator dostaje cel, wytyczne, format korpusu, przykłady wzorcowe, skrót istniejących przykładów i swoje źródła; "
        "sam czyta pliki sandboksa, przegląda korpus i zapisuje przykłady do zakładki Propozycje (jedna partia na serię). "
        "Wynik: partia, ile dodano, proporcje i skrócony manifest. "
        "Split pomiń, chyba że użytkownik wprost go wskazał — dobierany jest względem docelowych proporcji korpusu."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **SERIES_PROPERTIES,
            "sources": {"type": "array", "maxItems": 20, "items": SOURCE_ITEM, "description": "Dane: wspólne źródła w sandboksie, które generatory czytają same."},
            "batch": {"type": "string", "description": "Partia istniejącej serii do kontynuacji; pominięta = nowa seria."},
            "assignments": {"type": "array", "minItems": 1, "maxItems": MAX_ASSIGNMENTS, "items": ASSIGNMENT_ITEM},
        },
        "required": ["goal", "intent", "assignments"],
        "additionalProperties": False,
    },
    "returns": {
        "type": "object",
        "required": ["batch", "added", "proportions", "manifest"],
        "properties": {
            "batch": {"type": "string", "description": "Partia propozycji serii (do analyze_series i list_proposals)."},
            "added": {"type": "integer", "description": "Propozycje dodane w tym wywołaniu."},
            "proportions": {"type": "object", "description": "Liczności i udziały dodanych: flagi, rodzaje zadań, splity, etykiety/typy."},
            "manifest": {"type": "array", "items": {"type": "string"}, "description": "Po jednej skróconej linii na dodaną propozycję."},
            "problems": {"type": "array", "items": {"type": "string"}},
        },
    },
}

REMOVE_PROPOSALS_TOOL: dict[str, Any] = {
    "name": "remove_proposals",
    "description": "Usuwa propozycje serii (do kosza, użytkownik może je przywrócić): duplikaty, niezgodne z celem lub formatem, błędne. Tylko id z listy serii.",
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_EXAMPLES_PER_RUN,
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["id", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    "returns": {"type": "object", "required": ["removed", "unknown"], "properties": {"removed": {"type": "integer"}, "unknown": {"type": "array"}}},
}

REGENERATE_PROPOSALS_TOOL: dict[str, Any] = {
    "name": "regenerate_proposals",
    "description": (
        f"Usuwa wadliwe propozycje serii i od razu zleca generatorom ich poprawione zastępstwa (maks. {MAX_COUNT} na wywołanie; "
        "źródła i materiał z serii przechodzą automatycznie). Podaj konkretną instrukcję przy każdym id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_COUNT,
                "items": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}, "instruction": {"type": "string"}},
                    "required": ["id", "instruction"],
                    "additionalProperties": False,
                },
            },
            "sources": {"type": "array", "maxItems": 20, "items": SOURCE_ITEM, "description": "Dodatkowe źródła dla zastępstw."},
        },
        "required": ["items"],
        "additionalProperties": False,
    },
    "returns": {
        "type": "object",
        "required": ["removed", "regenerated", "unknown", "manifest"],
        "properties": {
            "removed": {"type": "integer"},
            "regenerated": {"type": "integer"},
            "unknown": {"type": "array"},
            "manifest": {"type": "array", "items": {"type": "string"}},
        },
    },
}

ANALYZE_SERIES_TOOL: dict[str, Any] = {
    "name": "analyze_series",
    "description": (
        "Zleca agentowi-analitykowi weryfikację serii (partii propozycji) względem celu. Analityk sam usuwa duplikaty i przykłady "
        "nienadające się do korpusu, zleca generatorom regenerację wadliwych i dodatkowe runy dla zbalansowania (ta sama partia), "
        "może czytać źródła w sandboksie. Wynik: ile oceniono, usunięto i dogenerowano, proporcje serii po analizie, manifest "
        "dogenerowanych i podsumowanie."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "batch": {"type": "string", "description": "Partia serii z wyniku generate_examples."},
            **SERIES_PROPERTIES,
            "sources": {"type": "array", "maxItems": 20, "items": SOURCE_ITEM, "description": "Dane: źródła serii (do weryfikacji i dla regeneracji)."},
        },
        "required": ["batch", "goal", "intent"],
        "additionalProperties": False,
    },
    "returns": {
        "type": "object",
        "required": ["batch", "reviewed", "removed", "added", "kept", "proportions", "summary"],
        "properties": {
            "batch": {"type": "string"},
            "reviewed": {"type": "integer"},
            "removed": {"type": "integer"},
            "added": {"type": "integer", "description": "Dogenerowane przez analityka (regeneracje i runy balansujące)."},
            "kept": {"type": "integer", "description": "Propozycje serii po analizie."},
            "proportions": {"type": "object", "description": "Proporcje serii po analizie."},
            "manifest": {"type": "array", "items": {"type": "string"}, "description": "Skrót dogenerowanych propozycji."},
            "summary": {"type": "string"},
            "problems": {"type": "array", "items": {"type": "string"}},
        },
    },
}

GENERATOR_PRIVATE_TOOLS = [SAVE_EXAMPLES_TOOL]
ANALYST_PRIVATE_TOOLS = [REMOVE_PROPOSALS_TOOL, REGENERATE_PROPOSALS_TOOL]
# The analyst calls the same generators, but goal, intent, context, mode, options, data and batch come from its series.
ANALYST_GENERATE_TOOL = copy.deepcopy(GENERATE_EXAMPLES_TOOL)
ANALYST_GENERATE_TOOL["description"] = (
    "Zleca generatorom dodatkowe przykłady do analizowanej serii (ta sama partia; cel, intencja, kontekst, tryb, opcje i dane "
    "serii przechodzą automatycznie — podaj tylko to, co ma być inne). Używaj dla zbalansowania i wypełnienia luk względem celu."
)
ANALYST_GENERATE_TOOL["parameters"]["required"] = ["assignments"]
ANALYST_GENERATE_TOOL["parameters"]["properties"].pop("batch")
# What each sub-agent sees besides the general (sandbox/filesystem) tools.
GENERATOR_TOOLS = [*GENERATOR_PRIVATE_TOOLS, *CORPUS_READ_TOOLS]
ANALYST_TOOLS = [*ANALYST_PRIVATE_TOOLS, ANALYST_GENERATE_TOOL, *CORPUS_READ_TOOLS]

AgentRunner = Callable[..., Iterator[dict[str, Any]]]
SaveProposals = Callable[[list[dict[str, Any]], str], dict[str, Any]]
RejectProposals = Callable[[list[str]], int]
# General-level executor: (name, arguments) -> (data, ui_event) or a generator of progress events returning it.
ToolExecutor = Callable[[str, dict[str, Any]], Any]


class GeneralTools:
    """Sandbox and filesystem tools shared by every agent of the conversation (empty without a sandbox)."""

    def __init__(self, tools: list[dict[str, Any]] | None = None, execute: ToolExecutor | None = None) -> None:
        self.tools = list(tools or []) if execute else []
        self.names = {tool["name"] for tool in self.tools}
        self.execute = execute

    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self.execute or name not in self.names:
            raise ToolError("unknown_tool", f"Nieznane narzędzie {name}.")
        return self.execute(name, arguments)

    @property
    def can_read_files(self) -> bool:
        return "read_lines" in self.names


def progress(name: str, message: str) -> dict[str, Any]:
    return {"type": "progress", "name": name, "message": message}


def tool_detail(name: str, arguments: dict[str, Any]) -> str:
    """Short description of a sub-agent tool call for the chat log (never the full payload)."""
    if name == SAVE_EXAMPLES_TOOL["name"]:
        examples = [item for item in arguments.get("examples") or [] if isinstance(item, dict)]
        flags = Counter(str(item.get("flag")) for item in examples)
        return f"{len(examples)} przykładów ({', '.join(f'{flag} {count}' for flag, count in flags.most_common())})"
    if name in {REMOVE_PROPOSALS_TOOL["name"], REGENERATE_PROPOSALS_TOOL["name"]}:
        items = arguments.get("items") or []
        reasons = Counter(clip(str(item.get("reason") or item.get("instruction") or ""), 40) for item in items if isinstance(item, dict))
        return f"{len(items)} propozycji" + (f": {'; '.join(reason for reason, _ in reasons.most_common(3))}" if reasons else "")
    if name == GENERATE_EXAMPLES_TOOL["name"]:
        assignments = [item for item in arguments.get("assignments") or [] if isinstance(item, dict)]
        return f"{len(assignments)} zleceń, {sum(int(item.get('count') or DEFAULT_COUNT) for item in assignments)} przykładów"
    return ", ".join(
        f"{key}={clip(json.dumps(value, ensure_ascii=False), 80)}"
        for key, value in arguments.items()
        if key not in {"examples", "items", "assignments", "content", "code", "script"}
    )


def agent_tool(agent: str, scope: str | None, event: dict[str, Any]) -> dict[str, Any]:
    """tool_call / tool_result of a sub-agent as a chat log event attributed to that agent."""
    base = {"type": "agent_tool", "agent": agent, "name": event["name"], **({"scope": scope} if scope else {})}
    if event["type"] == "tool_call":
        return {**base, "phase": "call", "detail": clip(tool_detail(event["name"], event.get("arguments") or {}), 300)}
    return {
        **base,
        "phase": "result",
        "ok": event["ok"],
        **({"ms": event["ms"]} if event.get("ms") is not None else {}),
        **({"error": clip(str(event["error"]), 300)} if event.get("error") else {}),
    }


def clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…"


def user_text(row: dict[str, Any]) -> str:
    return message_text(row["messages"], "user")


def template_lines(rows: list[dict[str, Any]]) -> set[str]:
    """Lines repeated across many user messages (shared instructions); they say nothing about a single example."""
    counts = Counter(line for row in rows for line in {normalized(item) for item in user_text(row).splitlines()} if line)
    threshold = max(3, TEMPLATE_LINE_SHARE * len(rows))
    return {line for line, count in counts.items() if count >= threshold}


def content_of(text: str, template: set[str]) -> str:
    lines = [normalized(line) for line in text.splitlines()]
    return " ".join(line for line in lines if line and line not in template) or normalized(text)


def task_of(row: dict[str, Any], vocabulary: set[str] | None) -> str:
    answer = last_assistant(row["messages"])
    return detect_task(answer, json_object(answer), row.get("task"), vocabulary)


def answer_digest(row: dict[str, Any], vocabulary: set[str] | None) -> str:
    answer = last_assistant(row["messages"])
    task = task_of(row, vocabulary)
    if task == "extraction":
        types = entity_types(row["messages"])
        return f"extraction: {', '.join(types) if types else 'brak elementów'}"
    if task == "classification":
        return f"classification: {classification_label(answer, json_object(answer), vocabulary)}"
    return f"generation: {clip(normalized(answer), 50)}"


def manifest_line(row: dict[str, Any], template: set[str], vocabulary: set[str] | None) -> str:
    return f"[{row['flag']} · {answer_digest(row, vocabulary)}] {clip(content_of(user_text(row), template), MANIFEST_CHARS)}"


def digest(
    rows: list[dict[str, Any]], pending: list[dict[str, Any]], template: set[str], vocabulary: set[str] | None
) -> tuple[str, int]:
    """One short line per known example: the non-template part of the prompt and what the answer contains.
    Pending proposals always fit; the corpus fills the rest with a random sample."""
    room = max(0, DIGEST_ITEMS - len(pending))
    corpus = rows if len(rows) <= room else random.sample(rows, room)
    lines = [
        f"- [{source} · {row.get('flag') or 'unclassified'} · {answer_digest(row, vocabulary)}] "
        f"{clip(content_of(user_text(row), template), DIGEST_CHARS)}"
        for source, items in (("propozycja", pending[:DIGEST_ITEMS]), ("korpus", corpus))
        for row in items
    ]
    return "\n".join(lines), len(rows) + len(pending)


def shingles(text: str) -> set[str]:
    words = re.findall(r"\w+", text.casefold())
    if len(words) < 3:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + 3]) for index in range(len(words) - 2)}


def near_duplicates(
    candidates: dict[str, str], known: dict[str, str], threshold: float = NEAR_DUPLICATE
) -> list[tuple[str, str, float]]:
    """(candidate id, other id, Jaccard of word 3-gram shingles) for pairs at or above the threshold."""
    sets = {key: shingles(text) for key, text in {**known, **candidates}.items()}
    index: dict[str, set[str]] = defaultdict(set)
    for key, items in sets.items():
        for item in items:
            index[item].add(key)
    pairs: dict[tuple[str, str], float] = {}
    for key in candidates:
        own = sets[key]
        shared: Counter[str] = Counter()
        for item in own:
            posting = index[item]
            if len(posting) <= COMMON_SHINGLE_DOCS:
                shared.update(other for other in posting if other != key)
        for other, count in shared.items():
            score = count / (len(own) + len(sets[other]) - count)
            if score >= threshold:
                pair = (key, other) if other not in candidates or key < other else (other, key)
                pairs[pair] = max(pairs.get(pair, 0.0), round(score, 2))
    return sorted(((a, b, score) for (a, b), score in pairs.items()), key=lambda item: -item[2])


def shares(counter: Counter[str]) -> dict[str, str]:
    total = sum(counter.values())
    return {key: f"{count} ({round(100 * count / total)}%)" for key, count in counter.most_common()} if total else {}


def proportions(rows: list[dict[str, Any]], vocabulary: set[str] | None) -> dict[str, dict[str, str]]:
    labels: Counter[str] = Counter()
    for row in rows:
        answer = last_assistant(row["messages"])
        if task_of(row, vocabulary) == "classification" and (label := classification_label(answer, json_object(answer), vocabulary)):
            labels[label] += 1
    types = Counter(item for row in rows for item in entity_types(row["messages"]))
    return {
        "flags": shares(Counter(row["flag"] for row in rows)),
        "tasks": shares(Counter(task_of(row, vocabulary) for row in rows)),
        "splits": shares(Counter(row["split"] for row in rows)),
        **({"labels": shares(labels)} if labels else {}),
        **({"types": shares(types)} if types else {}),
    }


def corpus_format(tools: CorpusAgentTools) -> str:
    analysis = analyze(tools.rows, tools.max_tokens, listed=0, vocabulary=tools.vocabulary)
    flags = Counter(row.get("flag") or "unclassified" for row in tools.rows)
    labelled = flags["positive"] + flags["negative"]
    system = tools.system_prompts[0][0] if tools.system_prompts else ""
    lines = [
        f"Przykładów w korpusie: {len(tools.rows)}; oczekujących propozycji: {len(tools.pending)}.",
        f"Rodzaje zadań: {json.dumps(analysis['tasks'], ensure_ascii=False)}.",
        f"Format odpowiedzi: {json.dumps(tools.profile, ensure_ascii=False)}.",
        f"Flagi: {json.dumps(dict(flags), ensure_ascii=False)}"
        + (f" (udział negatywów {round(100 * flags['negative'] / labelled)}%)." if labelled else "."),
        f"Maks. wymian user+assistant w przykładzie: {tools.max_exchanges}.",
    ]
    if analysis["labels"]:
        lines.append("Etykiety: " + ", ".join(f"{row['type']} {row['total']}" for row in analysis["labels"]) + ".")
    if analysis["types"]:
        lines.append("Typy elementów: " + ", ".join(f"{row['type']} {row['total']}" for row in analysis["types"][:60]) + ".")
    lines.append(f"Domyślna instrukcja systemowa (pole system pominięte):\n{clip(system, 4000) or '(brak — przykłady bez instrukcji systemowej)'}")
    return "\n".join(lines)


def reference_examples(tools: CorpusAgentTools, assignment: dict[str, Any]) -> str:
    flag, task = assignment.get("flag"), assignment.get("task")
    pool = [
        row
        for row in tools.rows
        if (flag not in {"positive", "negative"} or row.get("flag") == flag) and (not task or task_of(row, tools.vocabulary) == task)
    ] or tools.rows
    sample = random.sample(pool, min(REFERENCE_EXAMPLES, len(pool)))
    return "\n\n".join(
        json.dumps(
            {
                "flag": row.get("flag") or "unclassified",
                "user": clip(user_text(row), REFERENCE_CHARS),
                "assistant": clip(last_assistant(row["messages"]), REFERENCE_CHARS),
            },
            ensure_ascii=False,
        )
        for row in sample
    ) or "(korpus jest pusty — trzymaj się celu i wytycznych)"


def source_lines(sources: list[dict[str, Any]]) -> str:
    def span(item: dict[str, Any]) -> str:
        if item.get("end_line"):
            return f" L{item.get('start_line', 1)}–L{item['end_line']}"
        return f" od L{item['start_line']}" if item.get("start_line") else ""

    return "\n".join(f"- {item['path']}{span(item)}" + (f" — {item['note']}" if item.get("note") else "") for item in sources)


OPTION_LABELS = {
    "negative_share": "udział negatywów w zleceniach mixed (%)",
    "exchanges": "wymian user+assistant w przykładzie",
    "system_prompt": "instrukcja systemowa (corpus = korpusu, none = brak, custom = własna)",
    "custom_system": "własna instrukcja systemowa",
    "length": "długość poleceń i materiału",
    "difficulty": "trudność przypadków",
    "style": "styl poleceń użytkownika",
    "rejected_strategy": "DPO — w czym odpowiedź odrzucona ma być gorsza",
    "notes": "inne",
}


def training_mode(request: dict[str, Any]) -> str:
    return request.get("training_mode") if request.get("training_mode") in TRAINING_MODES else "sft"


def series_header(request: dict[str, Any], context: dict[str, str]) -> list[str]:
    """Intent, goal, training mode with what it means, context, options, guidelines and vocabulary of a series."""
    mode = training_mode(request)
    options = request.get("options") or {}
    return [
        *([f"INTENCJA UŻYTKOWNIKA:\n{request['intent']}"] if request.get("intent") else []),
        f"CEL SERII:\n{request['goal']}",
        f"TRYB TRENINGU: {mode.upper()}\n{TRAINING_MODE_KNOWLEDGE[mode]}",
        *([f"KONTEKST:\n{request['context']}"] if request.get("context") else []),
        *([f"CEL I OPIS KORPUSU (ustawienia):\n{context['agent_prompt']}"] if context.get("agent_prompt") else []),
        *(
            ["OPCJE:\n" + "\n".join(f"- {OPTION_LABELS.get(key, key)}: {value}" for key, value in options.items() if value not in (None, ""))]
            if options
            else []
        ),
        *([f"WYTYCZNE:\n{request['guidelines']}"] if request.get("guidelines") else []),
        *([f"SŁOWNIK (zamknięty):\n{context['vocabulary']}"] if context.get("vocabulary") else []),
    ]


def data_section(sources: list[dict[str, Any]], material: str, own_material: str = "") -> list[str]:
    return [
        *([f"DANE — ŹRÓDŁA DO PRZECZYTANIA (sandbox rozmowy; czytaj sam, cytuj dosłownie):\n{source_lines(sources)}"] if sources else []),
        *([f"DANE — MATERIAŁ ŹRÓDŁOWY:\n{material}"] if material else []),
        *([f"DANE — MATERIAŁ ŹRÓDŁOWY ZLECENIA:\n{own_material}"] if own_material else []),
    ]


def normalize_assignments(items: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assignments with counts clamped to the per-assignment maximum and the remaining budget."""
    kept, dropped = [], []
    for item in items:
        if budget <= 0:
            dropped.append(item)
            continue
        count = min(max(1, min(MAX_COUNT, int(item.get("count") or DEFAULT_COUNT))), budget)
        budget -= count
        kept.append(
            {
                **item,
                "count": count,
                "flag": item.get("flag") if item.get("flag") in FLAGS else "mixed",
                "task": item.get("task") if item.get("task") in TASKS else None,
                "focus": str(item.get("focus") or "").strip(),
            }
        )
    return kept, dropped


class GeneratorWorker:
    """One generator agent: reads sources and the corpus, validates and saves its examples itself (worker thread)."""

    def __init__(self, run: "GenerationRun", number: int, total: int, assignment: dict[str, Any], events: "queue.Queue") -> None:
        self.run, self.number, self.assignment, self.events = run, number, assignment, events
        self.name = f"generator {number}/{total} „{clip(assignment['focus'].splitlines()[0], 60)}”"
        self.saved = 0
        self.report = ""

    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.run.stopped.is_set():
            raise ToolError("stopped", "Seria została przerwana — zakończ pracę.")
        if name == SAVE_EXAMPLES_TOOL["name"]:
            return self.save(arguments.get("examples") or [])
        if name in CORPUS_READ_NAMES:
            with self.run.lock:
                return self.run.tools(name, arguments)
        return self.run.general(name, arguments)

    def prepare(self, example: dict[str, Any]) -> dict[str, Any]:
        item = {
            key: example[key] for key in ("system", "user", "assistant", "flag", "task", "split", "followup", "rejected") if key in example
        }
        options = self.run.request.get("options") or {}
        if self.assignment.get("task") and not item.get("task"):
            item["task"] = self.assignment["task"]
        if self.assignment.get("split") and not item.get("split"):
            item["split"] = self.assignment["split"]
        if "system" not in item and options.get("system_prompt") == "none":
            item["system"] = ""
        if "system" not in item and options.get("system_prompt") == "custom" and options.get("custom_system"):
            item["system"] = options["custom_system"]
        if options.get("exchanges") == 1:
            item.pop("followup", None)
        return item

    def save(self, examples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        target = self.assignment["count"]
        remaining = target - self.saved
        if remaining <= 0:
            raise ToolError("target_reached", f"Zapisano już {self.saved} z {target} — zakończ pracę krótkim raportem.")
        dpo = self.run.mode == "dpo"
        prepared = [self.prepare(item) for item in examples[:MAX_SAVE]]
        over = [{"index": index, "reason": "Ponad liczność zlecenia — pominięto."} for index in range(remaining, len(prepared))]
        rejected: list[dict[str, Any]] = []
        candidates: list[tuple[int, dict[str, Any]]] = []
        for index, item in enumerate(prepared[:remaining]):
            answer = str(item.pop("rejected", "") or "").strip()
            if dpo and not answer:
                rejected.append({"index": index, "reason": "Tryb DPO: brak odpowiedzi odrzuconej (rejected)."})
            elif dpo and normalized(answer) == normalized(str(item.get("assistant") or "")):
                rejected.append({"index": index, "reason": "Tryb DPO: odpowiedź odrzucona jest identyczna z wybraną."})
            else:
                candidates.append((index, {**item, **({"rejected": answer} if dpo else {})}))
        warnings: list[dict[str, Any]] = []
        saved = 0
        with self.run.lock:
            for start in range(0, len(candidates), MAX_PROPOSALS_PER_CALL):
                chunk = candidates[start : start + MAX_PROPOSALS_PER_CALL]
                result, event = self.run.tools.propose([item for _, item in chunk])
                failed = {item["index"] for item in result["rejected"]}
                rejected += [{**item, "index": chunk[item["index"]][0]} for item in result["rejected"]]
                warnings += [{**item, "index": chunk[item["index"]][0]} for item in result.get("warnings", [])]
                if event:
                    accepted = [item for position, (_, item) in enumerate(chunk) if position not in failed]
                    drafts = [
                        {**draft, **({"rejected": item["rejected"]} if dpo else {})} for draft, item in zip(event["examples"], accepted)
                    ]
                    saved += self.run.store(drafts)
        self.saved += saved
        self.events.put(
            (
                "event",
                self.number,
                progress(GENERATE_EXAMPLES_TOOL["name"], f"{self.name}: zapisano {saved} (łącznie {self.saved}/{target}), odrzucono {len(rejected)}"),
            )
        )
        data = {
            "saved": saved,
            "rejected": sorted([*rejected, *over], key=lambda item: item["index"]),
            **({"warnings": warnings} if warnings else {}),
            "saved_total": self.saved,
            "target": target,
            "remaining": target - self.saved,
        }
        return data, ({"type": "proposals", "saved": saved, "batch": self.run.batch} if saved else None)

    def work(self, system: str, brief: str, tools: list[dict[str, Any]]) -> None:
        try:
            for event in self.run.runner(
                self.run.provider, self.run.model, system, [{"role": "user", "content": brief}], tools, self, max_steps=GENERATOR_MAX_STEPS
            ):
                if self.run.stopped.is_set():
                    break
                self.events.put(("event", self.number, event))
        except Exception as error:  # noqa: BLE001 - reported as a failed generator, the series goes on
            self.events.put(("error", self.number, str(error)))
        finally:
            self.events.put(("done", self.number, None))


class GenerationRun:
    def __init__(
        self,
        provider: str,
        model: str,
        tools: CorpusAgentTools,
        request: dict[str, Any],
        context: dict[str, str],
        save: SaveProposals,
        general: GeneralTools,
        runner: AgentRunner = run_agent,
    ) -> None:
        self.provider, self.model, self.tools = provider, model, tools
        self.request, self.context = request, context
        self.save, self.general, self.runner = save, general, runner
        self.prompts = generator_prompts()
        self.batch = request.get("batch") or f"proposal-{uuid4()}"
        self.mode = training_mode(request)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.added: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.template = template_lines([*tools.rows, *tools.pending])

    def store(self, drafts: list[dict[str, Any]]) -> int:
        """Saves validated drafts to the series batch; caller holds the lock."""
        written = self.save(drafts, self.batch)
        for draft, proposal_id in zip(drafts, written["ids"]):
            row = {
                "id": str(proposal_id),
                "split": draft["split"],
                "flag": draft["flag"],
                "batch": self.batch,
                "task": draft.get("task"),
                "replaces": None,
                "rejected": draft.get("rejected"),
                "messages": draft_messages(draft),
            }
            self.tools.pending.append(row)
            self.added.append(row)
        return written["saved"]

    def brief(self, assignment: dict[str, Any], siblings: list[str], known: str, known_total: int) -> str:
        flag_rule = {
            "positive": "wszystkie positive",
            "negative": "wszystkie negative",
            "mixed": "mieszane — udział negatywów wg opcji albo jak w korpusie",
        }[assignment["flag"]]
        sources = [*(self.request.get("sources") or []), *(assignment.get("sources") or [])]
        return "\n\n".join(
            [
                *series_header(self.request, self.context),
                f"FORMAT KORPUSU:\n{corpus_format(self.tools)}",
                f"PRZYKŁADY WZORCOWE (pełne, z korpusu):\n{reference_examples(self.tools, assignment)}",
                "TWOJE ZLECENIE:\n"
                + "\n".join(
                    [
                        f"- zakres: {assignment['focus']}",
                        f"- liczność: {assignment['count']}",
                        f"- flagi: {flag_rule}",
                        *([f"- rodzaj zadania: {assignment['task']}"] if assignment.get("task") else []),
                        *([f"- split: {assignment['split']}"] if assignment.get("split") else []),
                        *(["- każdy przykład z polem rejected (para DPO)"] if self.mode == "dpo" else []),
                    ]
                ),
                *data_section(sources, self.request.get("material") or "", assignment.get("material") or ""),
                *(
                    ["ZAKRESY INNYCH GENERATORÓW (nie wchodź w nie):\n" + "\n".join(f"- {clip(item, 300)}" for item in siblings)]
                    if siblings
                    else []
                ),
                f"JUŻ ISTNIEJĄ (skrót: [źródło · flaga · zawartość odpowiedzi] treść polecenia bez wspólnej instrukcji; "
                f"pokazano {min(known_total, DIGEST_ITEMS)} z {known_total}) — nie powtarzaj ich:\n{known or '(brak)'}",
                f"Zapisz {assignment['count']} przykładów narzędziem save_examples (paczki po 20–30), potem zakończ krótkim raportem.",
            ]
        )

    def generate(self, assignments: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        name = GENERATE_EXAMPLES_TOOL["name"]
        with self.lock:
            known, known_total = digest(self.tools.rows, self.tools.pending, self.template, self.tools.vocabulary)
        system = self.prompts["generator"].replace("{corpus}", self.context["corpus"])
        tools = [*GENERATOR_TOOLS, *self.general.tools]
        events: queue.Queue = queue.Queue()
        workers = [GeneratorWorker(self, number, len(assignments), item, events) for number, item in enumerate(assignments, start=1)]
        yield progress(name, f"seria {self.batch[-8:]}: {len(workers)} generator(ów), {sum(item['count'] for item in assignments)} przykładów, do {MAX_PARALLEL} równolegle")
        executor = ThreadPoolExecutor(max_workers=MAX_PARALLEL)
        try:
            for worker in workers:
                siblings = [item["focus"] for item in assignments if item is not worker.assignment]
                executor.submit(worker.work, system, self.brief(worker.assignment, siblings, known, known_total), tools)
            running = len(workers)
            while running:
                kind, number, payload = events.get()
                worker = workers[number - 1]
                if kind == "done":
                    running -= 1
                    yield progress(
                        name,
                        f"{worker.name}: zakończył — {worker.saved}/{worker.assignment['count']}"
                        + (f" · {clip(worker.report, 200)}" if worker.report else ""),
                    )
                    if worker.saved < worker.assignment["count"]:
                        self.problems.append(f"{worker.name}: zapisano {worker.saved} z {worker.assignment['count']}")
                elif kind == "error":
                    self.problems.append(f"{worker.name}: {clip(payload, 300)}")
                    yield progress(name, f"{worker.name}: błąd — {clip(payload, 200)}")
                elif payload["type"] in {"proposals", "progress", "agent_tool"}:
                    yield payload
                elif payload["type"] in {"tool_call", "tool_result"}:
                    yield agent_tool(f"generator {worker.number}/{len(workers)}", clip(worker.assignment["focus"].splitlines()[0], 80), payload)
                elif payload["type"] == "text":
                    worker.report = payload["content"].strip()
        except GeneratorExit:
            self.stopped.set()
            raise
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def generate_examples_session(
    provider: str,
    model: str,
    tools: CorpusAgentTools,
    arguments: dict[str, Any],
    context: dict[str, str],
    save: SaveProposals,
    general: GeneralTools | None = None,
    runner: AgentRunner = run_agent,
) -> Generator[dict[str, Any], None, tuple[dict[str, Any], None]]:
    general = general or GeneralTools()
    goal = str(arguments.get("goal") or "").strip()
    if not goal:
        raise ToolError("invalid_arguments", "Podaj cel serii (goal).")
    batch = str(arguments.get("batch") or "").strip()
    if batch and not BATCH_PATTERN.match(batch):
        raise ToolError("invalid_arguments", "batch ma postać proposal-<uuid> z wyniku generate_examples.")
    assignments, dropped = normalize_assignments(arguments.get("assignments") or [], MAX_EXAMPLES_PER_RUN)
    if not assignments:
        raise ToolError("invalid_arguments", "Podaj co najmniej jedno zlecenie (assignments).")
    if not general.can_read_files and (arguments.get("sources") or any(item.get("sources") for item in assignments)):
        raise ToolError("no_sandbox", "Rozmowa nie ma sandboksa — przekaż materiał dosłownie w polu material.")
    run = GenerationRun(provider, model, tools, {**arguments, "goal": goal, "batch": batch}, context, save, general, runner)
    if dropped:
        run.problems.append(f"Pominięto {len(dropped)} zleceń ponad limit {MAX_EXAMPLES_PER_RUN} przykładów na wywołanie.")
    yield from run.generate(assignments)
    return (
        {
            "batch": run.batch,
            "added": len(run.added),
            "proportions": proportions(run.added, tools.vocabulary),
            "manifest": [manifest_line(row, run.template, tools.vocabulary) for row in run.added],
            **({"problems": run.problems} if run.problems else {}),
        },
        None,
    )


class SeriesAnalyst:
    """Analyst agent of one proposal batch: removes on its own and calls the generators to regenerate or fill gaps."""

    def __init__(
        self,
        provider: str,
        model: str,
        tools: CorpusAgentTools,
        request: dict[str, Any],
        context: dict[str, str],
        save: SaveProposals,
        reject: RejectProposals,
        general: GeneralTools,
        runner: AgentRunner = run_agent,
    ) -> None:
        self.provider, self.model, self.tools = provider, model, tools
        self.request, self.context = request, context
        self.save, self.reject, self.general, self.runner = save, reject, general, runner
        self.prompts = generator_prompts()
        self.batch = request["batch"]
        self.rows = {str(row["id"]): row for row in tools.pending if row.get("batch") == self.batch}
        self.reviewed = len(self.rows)
        self.removed = 0
        self.added: list[dict[str, Any]] = []
        self.problems: list[str] = []
        self.template = template_lines([*tools.rows, *tools.pending])

    def budget(self) -> int:
        return MAX_EXAMPLES_PER_RUN - len(self.added)

    def remove(self, ids: list[str]) -> int:
        removed = self.reject(ids) if ids else 0
        if ids:
            self.tools.reject_proposals(ids)
            for key in ids:
                self.rows.pop(key, None)
        self.removed += removed
        return removed

    def generate(self, arguments: dict[str, Any]) -> Generator[dict[str, Any], None, dict[str, Any]]:
        """Generators called by the analyst always extend this series, inherit its sources and stay within its budget."""
        assignments, dropped = normalize_assignments(arguments.get("assignments") or [], self.budget())
        if not assignments:
            raise ToolError("over_budget", f"Budżet dogenerowania w tej analizie ({MAX_EXAMPLES_PER_RUN}) jest wyczerpany.")
        request = {
            **{key: self.request.get(key) for key in ("intent", "context", "training_mode", "options", "guidelines", "material")},
            **{key: arguments[key] for key in ("goal", "intent", "context", "training_mode", "options", "guidelines", "material") if arguments.get(key)},
            "goal": arguments.get("goal") or self.request["goal"],
            "sources": [*(self.request.get("sources") or []), *(arguments.get("sources") or [])],
            "assignments": assignments,
            "batch": self.batch,
        }
        data, _ = yield from generate_examples_session(
            self.provider, self.model, self.tools, request, self.context, self.save, self.general, self.runner
        )
        added = [row for row in self.tools.pending if row.get("batch") == self.batch and str(row["id"]) not in self.rows]
        self.rows.update({str(row["id"]): row for row in added})
        self.added += added
        if dropped:
            data = {**data, "problems": [*data.get("problems", []), f"Pominięto {len(dropped)} zleceń ponad budżet analizy."]}
        return data

    def __call__(self, name: str, arguments: dict[str, Any]) -> Any:
        if name in CORPUS_READ_NAMES:
            return self.tools(name, arguments)
        if name == GENERATE_EXAMPLES_TOOL["name"]:
            return self.generate_tool(arguments)
        if name == REGENERATE_PROPOSALS_TOOL["name"]:
            return self.regenerate(arguments)
        if name == REMOVE_PROPOSALS_TOOL["name"]:
            items = arguments.get("items") or []
            known = [str(item["id"]) for item in items if str(item["id"]) in self.rows]
            removed = self.remove(known)
            event = {"type": "proposals_changed", "action": "reject_proposals", "count": removed} if removed else None
            return {"removed": removed, "unknown": [str(item["id"]) for item in items if str(item["id"]) not in self.rows and str(item["id"]) not in known]}, event
        return self.general(name, arguments)

    def generate_tool(self, arguments: dict[str, Any]) -> Generator[dict[str, Any], None, tuple[dict[str, Any], None]]:
        data = yield from self.generate(arguments)
        return data, None

    def regenerate(self, arguments: dict[str, Any]) -> Generator[dict[str, Any], None, tuple[dict[str, Any], dict[str, Any] | None]]:
        items = arguments.get("items") or []
        known = [item for item in items if str(item["id"]) in self.rows]
        unknown = [str(item["id"]) for item in items if str(item["id"]) not in self.rows]
        if not known:
            return {"removed": 0, "regenerated": 0, "unknown": unknown, "manifest": []}, None
        if len(known) > self.budget():
            raise ToolError("over_budget", f"Budżet dogenerowania pozwala jeszcze na {self.budget()} przykładów.")
        focus = "Zastąp odrzucone propozycje poprawionymi wersjami (każda pozycja = jeden nowy przykład):\n" + "\n".join(
            f"- [{self.rows[str(item['id'])]['flag']}] {item['instruction']} — oryginał: „{clip(user_text(self.rows[str(item['id'])]), 300)}”"
            for item in known
        )
        removed = self.remove([str(item["id"]) for item in known])
        yield {"type": "proposals_changed", "action": "reject_proposals", "count": removed}
        data = yield from self.generate(
            {"assignments": [{"focus": focus, "count": len(known), "flag": "mixed"}], "sources": arguments.get("sources") or []}
        )
        return {"removed": removed, "regenerated": data["added"], "unknown": unknown, "manifest": data["manifest"]}, None

    def brief(self) -> str:
        rows = list(self.rows.values())
        candidates = {key: content_of(user_text(row), self.template) for key, row in self.rows.items()}
        others = {
            str(row["id"]): content_of(user_text(row), self.template)
            for row in [*self.tools.rows, *self.tools.pending]
            if str(row["id"]) not in candidates
        }
        texts = {**others, **candidates}
        pairs = near_duplicates(candidates, others)
        return "\n\n".join(
            [
                *series_header(self.request, self.context),
                *data_section(self.request.get("sources") or [], self.request.get("material") or ""),
                f"FORMAT I STAN KORPUSU:\n{corpus_format(self.tools)}",
                f"LICZNOŚCI SERII: {json.dumps(proportions(rows, self.tools.vocabulary), ensure_ascii=False)}",
                "PODEJRZANE DUPLIKATY (podobieństwo słownych 3-gramów treści polecenia):\n"
                + (
                    "\n".join(
                        f"- {a} ~ {b if b in candidates else 'istniejący ' + b} ({score}): „{clip(texts[b], 200)}”"
                        for a, b, score in pairs[:200]
                    )
                    or "(brak)"
                ),
                f"PARTIA SERII: {self.batch} (generate_examples i regenerate_proposals dopisują do niej; budżet dogenerowania {self.budget()} przykładów)",
                "PROPOZYCJE SERII:\n"
                + "\n".join(
                    json.dumps(
                        {
                            "id": row["id"],
                            "flag": row["flag"],
                            "task": row.get("task"),
                            "split": row["split"],
                            "user": clip(user_text(row), ANALYST_TEXT_CHARS),
                            "assistant": clip(last_assistant(row["messages"]), ANALYST_TEXT_CHARS),
                            **({"rejected": clip(str(row["rejected"]), ANALYST_TEXT_CHARS)} if row.get("rejected") else {}),
                        },
                        ensure_ascii=False,
                    )
                    for row in rows
                ),
            ]
        )


def analyze_series_session(
    provider: str,
    model: str,
    tools: CorpusAgentTools,
    arguments: dict[str, Any],
    context: dict[str, str],
    save: SaveProposals,
    reject: RejectProposals,
    general: GeneralTools | None = None,
    runner: AgentRunner = run_agent,
) -> Generator[dict[str, Any], None, tuple[dict[str, Any], None]]:
    general = general or GeneralTools()
    name = ANALYZE_SERIES_TOOL["name"]
    request = {**arguments, "batch": str(arguments.get("batch") or "").strip(), "goal": str(arguments.get("goal") or "").strip()}
    if not request["goal"]:
        raise ToolError("invalid_arguments", "Podaj cel serii (goal).")
    if request.get("sources") and not general.can_read_files:
        raise ToolError("no_sandbox", "Rozmowa nie ma sandboksa — przekaż materiał dosłownie w polu material.")
    analyst = SeriesAnalyst(provider, model, tools, request, context, save, reject, general, runner)
    if not analyst.rows:
        raise ToolError("not_found", f"Brak oczekujących propozycji w partii {request['batch']}.")
    yield progress(name, f"analityk: weryfikuje {analyst.reviewed} propozycji serii {analyst.batch[-8:]}")
    summary = ""
    try:
        for event in runner(
            provider,
            model,
            analyst.prompts["analyzer"].replace("{corpus}", context["corpus"]),
            [{"role": "user", "content": analyst.brief()}],
            [*ANALYST_TOOLS, *general.tools],
            analyst,
            max_steps=ANALYST_MAX_STEPS,
        ):
            if event["type"] in {"proposals", "proposals_changed", "progress", "agent_tool"}:
                yield event
            elif event["type"] in {"tool_call", "tool_result"}:
                yield agent_tool("analityk", None, event)
            elif event["type"] == "text":
                summary = event["content"].strip()
    except (RuntimeError, OSError) as error:
        analyst.problems.append(f"analityk: {clip(str(error), 300)}")
        yield progress(name, f"analityk: błąd — {clip(str(error), 200)}")
    kept = list(analyst.rows.values())
    yield progress(
        name,
        f"analityk: usunięto {analyst.removed}, dogenerowano {len(analyst.added)}, w serii {len(kept)}" + (f" — {clip(summary, 300)}" if summary else ""),
    )
    return (
        {
            "batch": analyst.batch,
            "reviewed": analyst.reviewed,
            "removed": analyst.removed,
            "added": len(analyst.added),
            "kept": len(kept),
            "proportions": proportions(kept, tools.vocabulary),
            **({"manifest": [manifest_line(row, analyst.template, tools.vocabulary) for row in analyst.added]} if analyst.added else {}),
            "summary": summary,
            **({"problems": analyst.problems} if analyst.problems else {}),
        },
        None,
    )
