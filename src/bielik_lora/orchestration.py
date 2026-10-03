"""Orchestrator case state: its understanding of the user's intent, the execution plan (steps across understanding,
generation, analysis, review), findings, decisions, open questions and the example series in flight. Saved with the
conversation by update_plan and fed back at the start of every turn, so the plan adapts across the conversation."""

from __future__ import annotations

from typing import Any

STATUSES = ("understanding", "planning", "executing", "reviewing", "waiting_for_user", "done")
STEP_KINDS = ("understand", "plan", "generate", "analyze", "review", "ask", "report", "other")
STEP_STATUSES = ("pending", "in_progress", "done", "blocked", "skipped")
MAX_STEPS = 30
MAX_NOTES = 30
MAX_TEXT = 1000
STEP_MARKS = {"pending": "○", "in_progress": "▶", "done": "✓", "blocked": "✗", "skipped": "–"}

PLAN_TOOL: dict[str, Any] = {
    "name": "update_plan",
    "description": (
        "Zapisuje Twój stan sprawy przy rozmowie (pełny stan — zastępuje poprzedni): intencję użytkownika, status, plan kroków "
        "(topologia wykonania: co, w jakiej kolejności i czym — zrozumienie, generacja, analiza, przegląd, pytanie, raport), "
        "ustalenia, decyzje, otwarte pytania i serie przykładów (partie). Stan wraca do Ciebie na początku każdej tury. "
        "Aktualizuj go po każdym istotnym wyniku i po każdej zmianie celu przez użytkownika."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "description": "Twoje aktualne rozumienie, czego użytkownik naprawdę chce."},
            "status": {"type": "string", "enum": list(STATUSES)},
            "steps": {
                "type": "array",
                "maxItems": MAX_STEPS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string", "description": "Co ma być zrobione."},
                        "kind": {"type": "string", "enum": list(STEP_KINDS)},
                        "executor": {"type": "string", "description": "Kto/czym: np. read_large_file_session, generate_examples, analyze_series, Ty, użytkownik."},
                        "status": {"type": "string", "enum": list(STEP_STATUSES)},
                        "result": {"type": "string", "description": "Wynik lub powód blokady."},
                    },
                    "required": ["id", "title", "kind", "status"],
                    "additionalProperties": False,
                },
            },
            "findings": {"type": "array", "maxItems": MAX_NOTES, "items": {"type": "string"}, "description": "Ustalenia z materiałów, korpusu i rozmowy."},
            "decisions": {"type": "array", "maxItems": MAX_NOTES, "items": {"type": "string"}, "description": "Decyzje (Twoje i użytkownika)."},
            "open_questions": {"type": "array", "maxItems": MAX_NOTES, "items": {"type": "string"}},
            "series": {
                "type": "array",
                "maxItems": MAX_NOTES,
                "items": {
                    "type": "object",
                    "properties": {
                        "batch": {"type": "string"},
                        "goal": {"type": "string"},
                        "training_mode": {"type": "string", "enum": ["sft", "dpo"]},
                        "status": {"type": "string", "description": "np. wygenerowana, przeanalizowana, do korekty, zaakceptowana przez użytkownika."},
                        "notes": {"type": "string"},
                    },
                    "required": ["batch", "goal", "status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["intent", "status", "steps"],
        "additionalProperties": False,
    },
    "returns": {
        "type": "object",
        "required": ["saved", "steps", "done"],
        "properties": {"saved": {"type": "boolean"}, "steps": {"type": "integer"}, "done": {"type": "integer"}},
    },
}


def clip(text: Any, limit: int = MAX_TEXT) -> str:
    value = str(text or "").strip()
    return value if len(value) <= limit else f"{value[:limit]}…"


def normalize_state(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": clip(arguments.get("intent"), 2000),
        "status": arguments.get("status") if arguments.get("status") in STATUSES else "planning",
        "steps": [
            {
                "id": clip(step.get("id"), 40),
                "title": clip(step.get("title"), 300),
                "kind": step.get("kind") if step.get("kind") in STEP_KINDS else "other",
                **({"executor": clip(step["executor"], 100)} if step.get("executor") else {}),
                "status": step.get("status") if step.get("status") in STEP_STATUSES else "pending",
                **({"result": clip(step["result"], 500)} if step.get("result") else {}),
            }
            for step in (arguments.get("steps") or [])[:MAX_STEPS]
        ],
        **{
            key: [clip(item, 500) for item in (arguments.get(key) or [])[:MAX_NOTES] if str(item or "").strip()]
            for key in ("findings", "decisions", "open_questions")
        },
        "series": [
            {key: clip(item.get(key), 500) for key in ("batch", "goal", "training_mode", "status", "notes") if item.get(key)}
            for item in (arguments.get("series") or [])[-MAX_NOTES:]
        ],
    }


def render_state(state: dict[str, Any] | None) -> str:
    """Case state as the orchestrator sees it at the start of a turn."""
    if not state:
        return "(brak — rozmowa bez planu; ustal go update_plan, gdy zadanie wymaga więcej niż jednej czynności)"
    lines = [f"Intencja: {state.get('intent') or '(nieustalona)'}", f"Status: {state.get('status')}", "Plan:"]
    lines += [
        f"  {STEP_MARKS.get(step['status'], '?')} [{step['id']}] ({step['kind']}{' · ' + step['executor'] if step.get('executor') else ''}) "
        f"{step['title']}{' → ' + step['result'] if step.get('result') else ''}"
        for step in state.get("steps") or []
    ] or ["  (brak kroków)"]
    for key, label in (("findings", "Ustalenia"), ("decisions", "Decyzje"), ("open_questions", "Otwarte pytania")):
        if state.get(key):
            lines.append(f"{label}:")
            lines += [f"  - {item}" for item in state[key]]
    if state.get("series"):
        lines.append("Serie przykładów:")
        lines += [
            f"  - {item.get('batch')} [{item.get('training_mode', 'sft')}] {item.get('status')}: {item.get('goal')}"
            + (f" ({item['notes']})" if item.get("notes") else "")
            for item in state["series"]
        ]
    return "\n".join(lines)


def plan_event(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "plan",
        "intent": state["intent"],
        "status": state["status"],
        "steps": [{"title": step["title"], "status": step["status"]} for step in state["steps"]],
    }


def update_plan(arguments: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(normalized state, tool data); the caller persists the state."""
    state = normalize_state(arguments)
    return state, {"saved": True, "steps": len(state["steps"]), "done": sum(1 for step in state["steps"] if step["status"] == "done")}


def record_series(state: dict[str, Any] | None, phase: str, arguments: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """Bookkeeping done by the system, not the model: every generation and analysis lands in the case state's series."""
    current = normalize_state(state or {"intent": arguments.get("intent") or "", "status": "executing", "steps": []})
    previous = next((item for item in current["series"] if item.get("batch") == data["batch"]), {})
    if phase == "generate":
        status, note = "wygenerowana", f"generatory +{data['added']}"
    else:
        status, note = "przeanalizowana", f"analiza: −{data['removed']} +{data['added']} → {data['kept']}"
    entry = {
        "batch": data["batch"],
        "goal": arguments.get("goal") or previous.get("goal") or "",
        "training_mode": arguments.get("training_mode") or previous.get("training_mode") or "sft",
        "status": status,
        "notes": "; ".join(item for item in (previous.get("notes"), note) if item),
    }
    series = [item for item in current["series"] if item.get("batch") != data["batch"]]
    kinds = {"generate"} if phase == "generate" else {"analyze", "review"}
    step = next((item for item in current["steps"] if item["kind"] in kinds and item["status"] in {"pending", "in_progress"}), None)
    steps = [
        {**item, "status": "done", "result": item.get("result") or f"{data['batch'][-8:]}: {note}"} if item is step else item
        for item in current["steps"]
    ]
    return normalize_state({**current, "steps": steps, "series": [*series, entry][-MAX_NOTES:]})
