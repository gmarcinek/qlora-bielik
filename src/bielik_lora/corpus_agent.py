"""Tools that let a tool-calling LLM study a corpus and propose new training examples."""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

from bielik_lora.agent import CONTEXT_LIMIT_CHARS, ToolError
from bielik_lora.corpus_analysis import (
    LABEL_KEYS,
    MAX_LABEL_CHARS,
    TASKS,
    analyze,
    balance_summary,
    classification_label,
    detect_task,
    minimal_summary,
)
from bielik_lora.evaluation import LIST_FIELDS, answer_items, parse_answer

MAX_PROPOSALS_PER_CALL = 20
MAX_EXCHANGES = 2

TOOLS: list[dict[str, Any]] = [
    {
        "name": "corpus_balance",
        "description": (
            "Stan korpusu. Domyślnie (detail=summary) minimalna metryka: rozkład rodzajów zadań (tasks), udział % i liczność splitów train/validation/test "
            "oraz które przykłady są wadliwe (liczba i id per kontrola: JSON, klucze, sprzeczna flaga, cytat spoza tekstu, typ inny niż w poleceniu, "
            "spoza słownika, za długie, przeciek train/validation, duplikaty), plus liczba ostrzeżeń zbalansowania. "
            "detail=full: tabele typów elementów (ekstrakcja) i etykiet (klasyfikacja) z pozytywami/negatywami/splitami i ostrzeżeniami, długości, system prompty — "
            "tylko gdy planujesz generowanie pod luki albo użytkownik pyta o szczegóły. Osobno to samo dla oczekujących propozycji."
        ),
        "parameters": {
            "type": "object",
            "properties": {"detail": {"type": "string", "enum": ["summary", "full"]}},
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["detail", "corpus", "proposals"],
            "properties": {
                "detail": {"type": "string"},
                "corpus": {"type": "object", "required": ["examples"]},
                "proposals": {"type": "object", "required": ["examples"]},
            },
        },
    },
    {
        "name": "corpus_overview",
        "description": "Statystyki korpusu: liczności splitów i flag, format odpowiedzi (JSON/tekst, wspólne klucze), rozkład typów elementów (jeśli odpowiedzi zawierają listę encji), instrukcje systemowe.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        "returns": {
            "type": "object",
            "required": ["examples", "split_flag_counts", "answer_format", "entity_types", "system_prompts", "examples_without_system_prompt"],
            "properties": {
                "examples": {"type": "integer"},
                "split_flag_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
                "answer_format": {
                    "type": "object",
                    "required": ["json_share", "common_keys", "entity_list"],
                    "properties": {
                        "json_share": {"type": "number"},
                        "common_keys": {"type": "array", "items": {"type": "string"}},
                        "entity_list": {"type": "boolean"},
                    },
                },
                "entity_types": {"type": "object", "additionalProperties": {"type": "integer"}},
                "system_prompts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "count", "text"],
                        "properties": {"id": {"type": "integer"}, "count": {"type": "integer"}, "text": {"type": "string"}},
                    },
                },
                "examples_without_system_prompt": {"type": "integer"},
                "tasks": {
                    "type": "object",
                    "description": "Liczba przykładów wg rodzaju zadania (extraction / classification / generation).",
                    "additionalProperties": {"type": "integer"},
                },
                "labels": {
                    "type": "object",
                    "description": "Rozkład etykiet w przykładach klasyfikacji.",
                    "additionalProperties": {"type": "integer"},
                },
                "exchanges": {
                    "type": "object",
                    "description": "Liczba przykładów wg liczby wymian user+assistant (np. {\"1\": 845, \"2\": 45}).",
                    "additionalProperties": {"type": "integer"},
                },
            },
        },
    },
    {
        "name": "list_examples",
        "description": "Losowa próbka istniejących przykładów (pełne wiadomości) z opcjonalnym filtrowaniem.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Fragment tekstu do wyszukania (bez rozróżniania wielkości liter)."},
                "flag": {"type": "string", "enum": ["positive", "negative", "unclassified"]},
                "entity_type": {"type": "string", "description": "Tylko przykłady, których odpowiedź zawiera element tego typu (korpusy z listą encji)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["matched", "returned", "examples"],
            "properties": {
                "matched": {"type": "integer"},
                "returned": {"type": "integer"},
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "split", "flag", "system_prompt_id", "user", "assistant"],
                        "properties": {
                            "id": {"type": "string"},
                            "split": {"type": "string"},
                            "flag": {"type": "string"},
                            "system_prompt_id": {"type": ["integer", "string"]},
                            "user": {"type": "string"},
                            "assistant": {"type": "string"},
                        },
                    },
                },
            },
        },
    },
    {
        "name": "propose_examples",
        "description": (
            "Zaproponuj nowe przykłady treningowe (maks. 20 na wywołanie; domyślnie jedna wymiana user + assistant, followup tylko na wyraźną prośbę użytkownika lub zgodnie z udziałem w korpusie). "
            "Poprawne trafiają od razu do zakładki Propozycje korpusu z flagą propozycja (poza treningiem do akceptacji przez użytkownika). "
            "Gdy pominiesz system, użyta zostanie najczęstsza instrukcja systemowa korpusu; \"system\": \"\" oznacza przykład bez instrukcji systemowej."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "examples": {
                    "type": "array",
                    "maxItems": MAX_PROPOSALS_PER_CALL,
                    "items": {
                        "type": "object",
                        "properties": {
                            "system": {"type": "string", "description": "Pominięte = najczęstsza instrukcja korpusu; pusty tekst = bez instrukcji systemowej."},
                            "user": {"type": "string", "description": "Polecenie użytkownika (wraz z materiałem, jeśli korpus go zawiera)."},
                            "assistant": {"type": "string", "description": "Odpowiedź w formacie korpusu (dla korpusów JSON: JSON jako tekst)."},
                            "followup": {
                                "type": "object",
                                "description": "Opcjonalna druga wymiana (pomiń domyślnie); flaga dotyczy jej odpowiedzi.",
                                "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
                                "required": ["user", "assistant"],
                                "additionalProperties": False,
                            },
                            "flag": {"type": "string", "enum": ["positive", "negative"]},
                            "task": {
                                "type": "string",
                                "enum": list(TASKS),
                                "description": (
                                    "Rodzaj zadania: extraction (JSON z listą elementów), classification (etykieta lub JSON z kluczem label), "
                                    "generation (swobodny tekst: instrukcje, definicje, odpowiedzi na pytania, przekształcenia). "
                                    "Pominięty = wykryty z odpowiedzi."
                                ),
                            },
                            "split": {
                                "type": "string",
                                "enum": ["train", "validation", "test"],
                                "description": "Pominięte = split najbardziej brakujący względem docelowych proporcji korpusu.",
                            },
                            "replaces": {
                                "type": "string",
                                "description": (
                                    "Naprawa: id istniejącego przykładu korpusu, który ta propozycja zastąpi po akceptacji "
                                    "(oryginał trafi wtedy do kosza). Pominięty split = split oryginału."
                                ),
                            },
                        },
                        "required": ["user", "assistant", "flag"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["examples"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["accepted", "rejected"],
            "properties": {
                "accepted": {"type": "integer"},
                "saved": {"type": "integer", "description": "Zapisane w zakładce Propozycje."},
                "batch": {"type": ["string", "null"], "description": "Identyfikator partii propozycji."},
                "warnings": {
                    "type": "array",
                    "description": "Uwagi do przyjętych przykładów (nie blokują zapisu).",
                    "items": {"type": "object", "properties": {"index": {"type": "integer"}, "warning": {"type": "string"}}},
                },
                "rejected": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["index", "reason"],
                        "properties": {"index": {"type": "integer"}, "reason": {"type": "string"}},
                    },
                },
            },
        },
    },
]


PROPOSAL_EXAMPLE_PROPERTIES = {
    "system": {"type": "string", "description": "Pusty tekst = bez instrukcji systemowej."},
    "user": {"type": "string"},
    "assistant": {"type": "string"},
    "followup": {
        "type": ["object", "null"],
        "description": "Druga wymiana; null usuwa ją.",
        "properties": {"user": {"type": "string"}, "assistant": {"type": "string"}},
    },
    "flag": {"type": "string", "enum": ["positive", "negative"]},
    "task": {"type": "string", "enum": list(TASKS)},
    "split": {"type": "string", "enum": ["train", "validation", "test"]},
}

TOOLS += [
    {
        "name": "list_proposals",
        "description": (
            "Oczekujące propozycje (zakładka Propozycje): id, partia, split, proponowana flaga, system prompt i wymiany. "
            "Użyj, zanim poprawisz albo odrzucisz propozycje (list_examples ich nie pokazuje)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Fragment tekstu (bez rozróżniania wielkości liter)."},
                "batch": {"type": "string", "description": "Identyfikator partii (proposal-…)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 500},
            },
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["matched", "returned", "proposals"],
            "properties": {"matched": {"type": "integer"}, "returned": {"type": "integer"}, "proposals": {"type": "array"}},
        },
    },
    {
        "name": "update_proposals",
        "description": (
            "Poprawia oczekujące propozycje po id (maks. 20 na wywołanie). Podaj tylko zmieniane pola; reszta zostaje. "
            "Wynik przechodzi te same kontrole co propose_examples (format, flaga, duplikaty)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "maxItems": MAX_PROPOSALS_PER_CALL,
                    "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"}, **PROPOSAL_EXAMPLE_PROPERTIES},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["updates"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["updated", "rejected"],
            "properties": {"updated": {"type": "integer"}, "rejected": {"type": "array"}},
        },
    },
    {
        "name": "reject_proposals",
        "description": (
            "Usuwa (odrzuca) propozycje po id — wyłącznie oczekujące, jeszcze niezatwierdzone. Id zaakceptowanych przykładów "
            "korpusu są pomijane i wracają w polu unknown (do nich służy park_examples). Usunięte propozycje idą do kosza, można je przywrócić w UI."
        ),
        "parameters": {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 500}},
            "required": ["ids"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["rejected", "unknown"],
            "properties": {"rejected": {"type": "integer"}, "unknown": {"type": "array"}},
        },
    },
    {
        "name": "park_examples",
        "description": (
            "\"Usuwa\" zaakceptowane przykłady korpusu (train/validation/test) po id: przenosi je do przestrzeni "
            "\"bez splitu\" — poza trening, ewaluację i analizę, ale nadal widoczne dla użytkownika, który może je "
            "przywrócić albo usunąć na stałe. Dla oczekujących propozycji użyj reject_proposals."
        ),
        "parameters": {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 500}},
            "required": ["ids"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["moved", "unknown"],
            "properties": {
                "moved": {"type": "integer"},
                "parked": {"type": "integer", "description": "Faktycznie przeniesione w bazie."},
                "unknown": {"type": "array", "description": "Id spoza zaakceptowanych przykładów tego korpusu."},
            },
        },
    },
    {
        "name": "get_examples",
        "description": (
            "Pełne przykłady po id — zaakceptowane przykłady korpusu (train/validation/test) i oczekujące propozycje: status, split, "
            "flaga, instrukcja systemowa, wymiany. Użyj, gdy użytkownik lub analiza wskazuje konkretne id (np. z kontroli jakości)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 100}},
            "required": ["ids"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["examples", "unknown"],
            "properties": {
                "examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "status", "split", "flag", "user", "assistant"],
                        "properties": {
                            "id": {"type": "string"},
                            "status": {"type": "string", "description": "corpus (zaakceptowany) albo proposal (oczekująca propozycja)."},
                            "split": {"type": "string"},
                            "flag": {"type": "string"},
                            "user": {"type": "string"},
                            "assistant": {"type": "string"},
                        },
                    },
                },
                "unknown": {"type": "array"},
            },
        },
    },
    {
        "name": "update_examples",
        "description": (
            "Edytuje zaakceptowane przykłady korpusu po id (maks. 20 na wywołanie) — bezpośrednio, bez propozycji. Podaj tylko "
            "zmieniane pola (system, user, assistant, followup, flag, task, split); reszta zostaje. Wynik przechodzi te same kontrole "
            "co propozycje (format, flaga, słownik, duplikat). Poprzednia wersja jest zapisywana — użytkownik może cofnąć zmianę. "
            "Najpierw obejrzyj przykłady get_examples. Dla oczekujących propozycji użyj update_proposals."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "maxItems": MAX_PROPOSALS_PER_CALL,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            **PROPOSAL_EXAMPLE_PROPERTIES,
                            "flag": {"type": "string", "enum": ["positive", "negative", "unclassified"]},
                        },
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["updates"],
            "additionalProperties": False,
        },
        "returns": {
            "type": "object",
            "required": ["updated", "rejected"],
            "properties": {
                "updated": {"type": "integer"},
                "written": {"type": "integer", "description": "Faktycznie zapisane w bazie."},
                "revision": {"type": ["string", "null"], "description": "Zapisana poprzednia wersja (do cofnięcia przez użytkownika)."},
                "rejected": {"type": "array"},
            },
        },
    },
]


def draft_messages(draft: dict[str, Any]) -> list[dict[str, str]]:
    """Chat messages of a validated draft: optional system prompt, then 1–2 user/assistant exchanges."""
    return [
        *([{"role": "system", "content": draft["system"]}] if draft["system"] else []),
        *(
            message
            for exchange in [draft, *draft.get("turns", [])]
            for message in ({"role": "user", "content": exchange["user"]}, {"role": "assistant", "content": exchange["assistant"]})
        ),
    ]


def message_text(messages: list[dict[str, Any]], role: str) -> str:
    return next((str(m.get("content", "")) for m in messages if m.get("role") == role), "")


def last_assistant(messages: list[dict[str, Any]]) -> str:
    return next((str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "assistant"), "")


def entity_types(messages: list[dict[str, Any]]) -> list[str]:
    return sorted({str(item.get("type")) for item in answer_items(parse_answer(last_assistant(messages))) if item.get("type")})


def json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def exchanges_of(example: dict[str, Any]) -> list[dict[str, Any]]:
    followup = example.get("followup")
    # Models often send an empty followup object instead of omitting it.
    has_followup = isinstance(followup, dict) and any(str(followup.get(key) or "").strip() for key in ("user", "assistant"))
    return [example, *([followup] if has_followup else [])][:MAX_EXCHANGES]


def normalized_answer(answer: str) -> str:
    parsed = json_object(answer)
    return json.dumps(parsed, ensure_ascii=False) if parsed is not None else answer.strip()


def answer_profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Answer format learned from the corpus itself: JSON or free text, keys shared by most JSON answers."""
    parsed = [json_object(last_assistant(row["messages"])) for row in rows]
    objects = [item for item in parsed if item is not None]
    keys = Counter(key for item in objects for key in item)
    common = sorted(key for key, count in keys.items() if objects and count >= 0.9 * len(objects))
    return {
        "json_share": round(len(objects) / len(rows), 3) if rows else 0.0,
        "common_keys": common,
        "entity_list": any(key in common for key in ("entities", "exclusions")),
    }


class CorpusAgentTools:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        pending: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        split_ratio: dict[str, int] | None = None,
        vocabulary: list[dict[str, str]] | None = None,
        max_exchanges: int = MAX_EXCHANGES,
    ) -> None:
        self.split_ratio = split_ratio
        self.max_exchanges = max_exchanges
        self.vocabulary = {str(item["name"]) for item in vocabulary or []} or None
        self.split_counts = Counter(row["split"] for row in [*rows, *(pending or [])])
        self.rows = rows
        self.pending = pending or []
        self.replaced = {str(row["replaces"]) for row in self.pending if row.get("replaces")}
        self.max_tokens = max_tokens
        self.profile = answer_profile(rows)
        self.corpus_tasks = Counter(analyze(rows, max_tokens, listed=0, vocabulary=self.vocabulary)["tasks"])
        # Pending proposals do not shape the corpus profile but still count as duplicates.
        self.known_users = {message_text(row["messages"], "user").strip() for row in [*rows, *self.pending]}
        self.system_prompts = Counter(
            prompt for row in rows if (prompt := message_text(row["messages"], "system"))
        ).most_common()

    def __call__(self, name: str, arguments: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
        if name == "corpus_balance":
            return self.balance(**arguments), None
        if name == "corpus_overview":
            return self.overview(), None
        if name == "list_examples":
            return self.list_examples(**arguments), None
        if name == "propose_examples":
            return self.propose(arguments.get("examples") or [])
        if name == "list_proposals":
            return self.list_proposals(**arguments), None
        if name == "update_proposals":
            return self.update_proposals(arguments.get("updates") or [])
        if name == "reject_proposals":
            return self.reject_proposals(arguments.get("ids") or [])
        if name == "park_examples":
            return self.park_examples(arguments.get("ids") or [])
        if name == "get_examples":
            return self.get_examples(arguments.get("ids") or []), None
        if name == "update_examples":
            return self.update_examples(arguments.get("updates") or [])
        raise ToolError("unknown_tool", f"Nieznane narzędzie {name}.")

    def balance(self, detail: str = "summary") -> dict[str, Any]:
        view = balance_summary if detail == "full" else minimal_summary
        return {
            "detail": detail,
            "corpus": view(analyze(self.rows, self.max_tokens, vocabulary=self.vocabulary)),
            "proposals": view(analyze(self.pending, self.max_tokens, vocabulary=self.vocabulary)),
            "split_target": self.split_ratio,
        }

    def overview(self) -> dict[str, Any]:
        counts = Counter(f"{row['split']}/{row.get('flag') or 'unclassified'}" for row in self.rows)
        types = Counter(entity_type for row in self.rows for entity_type in entity_types(row["messages"]))
        analysis = analyze(self.rows, self.max_tokens, listed=0, vocabulary=self.vocabulary)
        return {
            "examples": len(self.rows),
            "tasks": analysis["tasks"],
            "labels": {row["type"]: row["total"] for row in analysis["labels"]},
            "split_flag_counts": dict(sorted(counts.items())),
            "answer_format": self.profile,
            "entity_types": dict(types.most_common()),
            "system_prompts": [
                {"id": index, "count": count, "text": prompt}
                for index, (prompt, count) in enumerate(self.system_prompts)
            ],
            "examples_without_system_prompt": sum(1 for row in self.rows if not message_text(row["messages"], "system")),
            "exchanges": {
                str(count): number
                for count, number in sorted(
                    Counter(sum(1 for m in row["messages"] if m.get("role") == "assistant") for row in self.rows).items()
                )
            },
        }

    def list_examples(
        self, query: str = "", flag: str | None = None, entity_type: str | None = None, limit: int = 5
    ) -> dict[str, Any]:
        needle = query.lower().strip()
        matches = [
            row
            for row in self.rows
            if (not flag or (row.get("flag") or "unclassified") == flag)
            and (not entity_type or entity_type in entity_types(row["messages"]))
            and (not needle or needle in json.dumps(row["messages"], ensure_ascii=False).lower())
        ]
        sample = random.sample(matches, min(max(1, limit), len(matches)))
        prompt_ids = {prompt: index for index, (prompt, _) in enumerate(self.system_prompts)}
        examples: list[dict[str, Any]] = []
        used = 0
        for row in sample:
            example = {
                "id": str(row["id"]),
                "split": row["split"],
                "flag": row.get("flag") or "unclassified",
                "system_prompt_id": prompt_ids.get(message_text(row["messages"], "system"), "brak"),
                "user": message_text(row["messages"], "user"),
                "assistant": last_assistant(row["messages"]),
            }
            used += len(json.dumps(example, ensure_ascii=False))
            if examples and used > CONTEXT_LIMIT_CHARS:
                break
            examples.append(example)
        return {"matched": len(matches), "returned": len(examples), "examples": examples}

    def next_split(self) -> str:
        """Split furthest below the corpus target ratio (corpus + pending + this session), train without a target."""
        if not self.split_ratio:
            return "train"
        total = sum(self.split_counts.values()) + 1
        split = max(
            ("train", "validation", "test"),
            key=lambda name: self.split_ratio.get(name, 0) / 100 * total - self.split_counts[name],
        )
        self.split_counts[split] += 1
        return split

    def propose(self, examples: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        default_system = self.system_prompts[0][0] if self.system_prompts else ""
        rows_by_id = {str(row["id"]): row for row in self.rows}
        warnings: list[dict[str, Any]] = []
        for index, example in enumerate(examples[:MAX_PROPOSALS_PER_CALL]):
            example = self.limit_exchanges(example, index, warnings)
            replaces = str(example.get("replaces") or "").strip() or None
            original = rows_by_id.get(replaces) if replaces else None
            if replaces and original is None:
                rejected.append({"index": index, "reason": f"Nie ma przykładu korpusu {replaces} do zastąpienia."})
                continue
            if replaces in self.replaced:
                rejected.append({"index": index, "reason": f"Przykład {replaces} ma już oczekującą propozycję naprawy."})
                continue
            # The fix may keep the original user message, which is not a duplicate since the original goes away.
            original_user = message_text(original["messages"], "user").strip() if original else None
            if original_user:
                self.known_users.discard(original_user)
            reason = self.validate(example)
            if reason:
                if original_user:
                    self.known_users.add(original_user)
                rejected.append({"index": index, "reason": reason})
                continue
            exchanges = [
                {"user": str(item["user"]).strip(), "assistant": normalized_answer(str(item["assistant"]))}
                for item in exchanges_of(example)
            ]
            self.known_users.add(exchanges[0]["user"])
            if replaces:
                self.replaced.add(replaces)
            if warning := self.missing_keys_warning(example):
                warnings.append({"index": index, "warning": warning})
            accepted.append(
                {
                    "system": str(example["system"]).strip() if "system" in example else default_system,
                    **exchanges[0],
                    "turns": exchanges[1:],
                    "flag": example["flag"],
                    "task": example.get("task"),
                    "split": example.get("split") or (original["split"] if original else self.next_split()),
                    "replaces": replaces,
                }
            )
        if len(examples) > MAX_PROPOSALS_PER_CALL:
            rejected.append({"index": MAX_PROPOSALS_PER_CALL, "reason": f"Maksymalnie {MAX_PROPOSALS_PER_CALL} na wywołanie."})
        result = {"accepted": len(accepted), "rejected": rejected, **({"warnings": warnings} if warnings else {})}
        return result, ({"type": "drafts", "examples": accepted} if accepted else None)

    def proposal_view(self, row: dict[str, Any]) -> dict[str, Any]:
        pairs = [
            {"user": user, "assistant": answer}
            for user, answer in zip(
                [m["content"] for m in row["messages"] if m.get("role") == "user"],
                [m["content"] for m in row["messages"] if m.get("role") == "assistant"],
            )
        ]
        return {
            "id": str(row["id"]),
            "batch": row.get("batch"),
            "split": row["split"],
            "flag": row.get("flag"),
            "system": message_text(row["messages"], "system"),
            **(pairs[0] if pairs else {"user": "", "assistant": ""}),
            "followup": pairs[1] if len(pairs) > 1 else None,
            "replaces": row.get("replaces"),
            "task": row.get("task"),
        }

    def list_proposals(self, query: str = "", batch: str | None = None, limit: int = 50) -> dict[str, Any]:
        needle = query.lower().strip()
        matches = [
            row
            for row in self.pending
            if (not batch or row.get("batch") == batch)
            and (not needle or needle in json.dumps(row["messages"], ensure_ascii=False).lower())
        ]
        proposals: list[dict[str, Any]] = []
        used = 0
        for row in matches[: max(1, limit)]:
            view = self.proposal_view(row)
            used += len(json.dumps(view, ensure_ascii=False))
            if proposals and used > CONTEXT_LIMIT_CHARS:
                break
            proposals.append(view)
        return {"matched": len(matches), "returned": len(proposals), "proposals": proposals}

    def limit_exchanges(self, example: dict[str, Any], index: int, warnings: list[dict[str, Any]]) -> dict[str, Any]:
        if self.max_exchanges > 1 or len(exchanges_of(example)) == 1:
            return example
        warnings.append({"index": index, "warning": "Pominięto followup: korpus ma ustawioną jedną wymianę (user + assistant)."})
        return {key: value for key, value in example.items() if key != "followup"}

    def update_proposals(self, updates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        by_id = {str(row["id"]): row for row in self.pending}
        changed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for index, update in enumerate(updates[:MAX_PROPOSALS_PER_CALL]):
            row = by_id.get(str(update.get("id")))
            if row is None:
                corpus_ids = {str(item["id"]) for item in self.rows}
                rejected.append(
                    {
                        "index": index,
                        "reason": (
                            f"{update.get('id')} to zaakceptowany przykład korpusu — edytuj go update_examples."
                            if str(update.get("id")) in corpus_ids
                            else f"Nie ma oczekującej propozycji {update.get('id')}."
                        ),
                    }
                )
                continue
            current = self.proposal_view(row)
            example = {**current, **{key: value for key, value in update.items() if key != "id"}}
            if not isinstance(example.get("followup"), dict):
                example.pop("followup", None)
            example = self.limit_exchanges(example, index, warnings)
            original_user = current["user"].strip()
            replaced_row = next((item for item in self.rows if str(item["id"]) == str(row.get("replaces"))), None)
            replaced_user = message_text(replaced_row["messages"], "user").strip() if replaced_row else None
            self.known_users.discard(original_user)
            if replaced_user:
                self.known_users.discard(replaced_user)
            reason = self.validate(example)
            if reason:
                self.known_users.add(original_user)
                if replaced_user:
                    self.known_users.add(replaced_user)
                rejected.append({"index": index, "reason": reason})
                continue
            exchanges = [
                {"user": str(item["user"]).strip(), "assistant": normalized_answer(str(item["assistant"]))}
                for item in exchanges_of(example)
            ]
            draft = {"system": str(example.get("system") or "").strip(), **exchanges[0], "turns": exchanges[1:]}
            self.known_users.add(exchanges[0]["user"])
            row.update(
                messages=draft_messages(draft),
                flag=example["flag"],
                split=example.get("split") or row["split"],
                task=example.get("task") or row.get("task"),
            )
            changed.append(
                {"id": str(row["id"]), "messages": row["messages"], "flag": row["flag"], "split": row["split"], "task": row["task"]}
            )
            if warning := self.missing_keys_warning(example):
                warnings.append({"index": index, "warning": warning})
        result = {"updated": len(changed), "rejected": rejected, **({"warnings": warnings} if warnings else {})}
        return result, ({"type": "proposal_updates", "updated": changed} if changed else None)

    def reject_proposals(self, ids: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        known = {str(row["id"]) for row in self.pending}
        removed = [item for item in dict.fromkeys(ids) if item in known]
        self.pending = [row for row in self.pending if str(row["id"]) not in removed]
        result = {"rejected": len(removed), "unknown": [item for item in ids if item not in known]}
        return result, ({"type": "proposal_rejections", "ids": removed} if removed else None)

    def missing_keys_warning(self, example: dict[str, Any]) -> str | None:
        """Keys most corpus answers have; a hint only — the example's own schema (e.g. its system prompt) wins."""
        answer = json_object(str(exchanges_of(example)[-1].get("assistant") or ""))
        if answer is None or detect_task("", answer, example.get("task"), self.vocabulary) != "extraction":
            return None
        missing = [key for key in self.profile["common_keys"] if key not in answer]
        return f"Brak kluczy częstych w korpusie: {', '.join(missing)} (przyjęto mimo to)." if missing else None

    def park_examples(self, ids: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        known = {str(row["id"]) for row in self.rows}
        parked = [item for item in dict.fromkeys(str(value) for value in ids) if item in known]
        for row in [row for row in self.rows if str(row["id"]) in parked]:
            self.split_counts[row["split"]] -= 1
        self.rows = [row for row in self.rows if str(row["id"]) not in parked]
        result = {"moved": len(parked), "unknown": [item for item in ids if str(item) not in known]}
        return result, ({"type": "examples_parked", "ids": parked} if parked else None)

    def get_examples(self, ids: list[str]) -> dict[str, Any]:
        rows = {str(row["id"]): ("corpus", row) for row in self.rows}
        rows.update({str(row["id"]): ("proposal", row) for row in self.pending})
        examples: list[dict[str, Any]] = []
        used = 0
        for key in dict.fromkeys(str(item) for item in ids):
            if key not in rows:
                continue
            status, row = rows[key]
            view = {**self.proposal_view(row), "status": status, "flag": row.get("flag") or "unclassified"}
            used += len(json.dumps(view, ensure_ascii=False))
            if examples and used > CONTEXT_LIMIT_CHARS:
                break
            examples.append(view)
        return {"examples": examples, "unknown": [str(item) for item in ids if str(item) not in rows]}

    def update_examples(self, updates: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Direct edits of accepted corpus examples; same checks as proposals, the caller keeps the previous version."""
        by_id = {str(row["id"]): row for row in self.rows}
        pending_ids = {str(row["id"]) for row in self.pending}
        changed: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for index, update in enumerate(updates[:MAX_PROPOSALS_PER_CALL]):
            key = str(update.get("id"))
            row = by_id.get(key)
            if row is None:
                reason = (
                    f"{key} to oczekująca propozycja — edytuj ją update_proposals."
                    if key in pending_ids
                    else f"Nie ma zaakceptowanego przykładu {key} w tym korpusie (train/validation/test)."
                )
                rejected.append({"index": index, "reason": reason})
                continue
            current = {**self.proposal_view(row), "flag": row.get("flag") or "unclassified"}
            example = {**current, **{name: value for name, value in update.items() if name != "id"}}
            if not isinstance(example.get("followup"), dict):
                example.pop("followup", None)
            example = self.limit_exchanges(example, index, warnings)
            original_user = current["user"].strip()
            self.known_users.discard(original_user)
            reason = self.validate(example)
            if reason:
                self.known_users.add(original_user)
                rejected.append({"index": index, "reason": reason})
                continue
            exchanges = [
                {"user": str(item["user"]).strip(), "assistant": normalized_answer(str(item["assistant"]))}
                for item in exchanges_of(example)
            ]
            draft = {"system": str(example.get("system") or "").strip(), **exchanges[0], "turns": exchanges[1:]}
            self.known_users.add(exchanges[0]["user"])
            split = example.get("split") or row["split"]
            if split != row["split"]:
                self.split_counts[row["split"]] -= 1
                self.split_counts[split] += 1
            row.update(messages=draft_messages(draft), flag=example["flag"], split=split, task=example.get("task") or row.get("task"))
            changed.append({"id": key, "messages": row["messages"], "flag": row["flag"], "split": split, "task": row["task"]})
            if warning := self.missing_keys_warning(example):
                warnings.append({"index": index, "warning": warning})
        result = {"updated": len(changed), "rejected": rejected, **({"warnings": warnings} if warnings else {})}
        return result, ({"type": "example_updates", "updated": changed} if changed else None)

    def validate(self, example: dict[str, Any]) -> str | None:
        exchanges = exchanges_of(example)
        if str(exchanges[0].get("user") or "").strip() in self.known_users:
            return "Duplikat istniejącego przykładu (identyczne pole user)."
        for number, exchange in enumerate(exchanges, start=1):
            where = "" if number == 1 else " (followup)"
            if not str(exchange.get("user") or "").strip():
                return f"Puste pole user{where}."
            last = number == len(exchanges)
            reason = self.validate_answer(
                str(exchange.get("assistant") or ""), example.get("flag") if last else None, example.get("task") if last else None
            )
            if reason:
                return reason + where
        return None

    def validate_answer(self, text: str, flag: str | None, task: str | None = None) -> str | None:
        """Checks depend on the task kind (given or detected): extraction needs the JSON list format, classification
        a label (from the vocabulary if set), generation only a non-empty answer."""
        if not text.strip():
            return "Pusta odpowiedź assistant."
        answer = json_object(text)
        if text.strip().startswith("{") and answer is None:
            return "Odpowiedź wygląda na JSON, ale nie jest poprawnym obiektem JSON."
        explicit = task
        task = detect_task(text, answer, task, self.vocabulary)
        if explicit is None and task != "extraction" and self.corpus_tasks[task] == 0 and self.corpus_tasks["extraction"]:
            # Single-task corpora: a stray plain-text answer is more likely a format mistake than a new task.
            return (
                f"Korpus nie ma jeszcze przykładów typu {task} (odpowiedzi to JSON z listą elementów). "
                f"Jeśli to zamierzone nowe zadanie, podaj task: \"{task}\"; inaczej odpowiedz w formacie korpusu."
            )
        if task == "classification":
            label = classification_label(text, answer, self.vocabulary)
            if not label:
                if answer is not None:
                    return f"Klasyfikacja: brak etykiety (klucz jeden z: {', '.join(LABEL_KEYS)})."
                if self.vocabulary:
                    return f"Klasyfikacja: odpowiedź musi być jedną z etykiet: {', '.join(sorted(self.vocabulary))}."
                if len(text.strip()) > MAX_LABEL_CHARS:
                    return "Klasyfikacja: odpowiedź tekstowa ma być samą etykietą (lub JSON z kluczem label)."
            elif self.vocabulary and label not in self.vocabulary:
                return f"Etykieta spoza słownika korpusu: {label}. Dozwolone: {', '.join(sorted(self.vocabulary))}."
            return None
        if task != "extraction":
            return None
        if answer is None:
            return "Ekstrakcja: odpowiedź musi być obiektem JSON z listą elementów (np. entities)."
        if not any(isinstance(answer.get(field), list) for field in LIST_FIELDS):
            return f"Ekstrakcja: lista elementów ({' / '.join(LIST_FIELDS)}) musi być tablicą."
        items = answer_items(answer)
        if any(not item.get("type") for item in items):
            return "Każdy element listy musi mieć pole type, jak w korpusie."
        if self.vocabulary and (outside := sorted({str(item["type"]) for item in items} - self.vocabulary)):
            return (
                f"Typ spoza słownika korpusu: {', '.join(outside)}. Dozwolone: {', '.join(sorted(self.vocabulary))} "
                "(podkategorię opisz w name/description, nie w type)."
            )
        if flag == "negative" and items:
            return "Ekstrakcja: przykład negatywny ma pustą listę elementów."
        if flag == "positive" and not items:
            return "Ekstrakcja: przykład pozytywny zawiera co najmniej jeden element."
        return None
