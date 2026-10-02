"""Provider-agnostic tool-calling loop (OpenAI Responses API and Anthropic Messages API) with tool contracts."""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from typing import Any, Callable, Generator, Iterator
from urllib import error, request

ToolOutcome = tuple[Any, dict[str, Any] | None]
# An executor returns (data, ui_event) or a generator that yields progress events and returns (data, ui_event).
ToolExecutor = Callable[[str, dict[str, Any]], Any]

# Single ceiling for anything put into the model context (tool results, chat history).
CONTEXT_LIMIT_CHARS = int(os.getenv("CONTEXT_LIMIT_CHARS", "3500000"))


class ToolError(RuntimeError):
    def __init__(self, code: str, message: str, details: list[str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or []


JSON_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def schema_errors(schema: dict[str, Any], value: Any, path: str = "$") -> list[str]:
    """Minimal JSON Schema check (type, required, properties, additionalProperties, items, enum, bounds)."""
    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        matches = any(
            isinstance(value, JSON_TYPES[item]) and not (item in {"integer", "number"} and isinstance(value, bool))
            for item in types
        )
        if not matches:
            return [f"{path}: oczekiwano {'/'.join(types)}, jest {type(value).__name__}"]
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: dozwolone {schema['enum']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: maksimum {schema['maximum']}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        errors += [f"{path}.{key}: wymagane" for key in schema.get("required", []) if key not in value]
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                errors += schema_errors(properties[key], item, f"{path}.{key}")
            elif extra is False:
                errors.append(f"{path}.{key}: nieznane pole")
            elif isinstance(extra, dict):
                errors += schema_errors(extra, item, f"{path}.{key}")
    if isinstance(value, list):
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: maksymalnie {schema['maxItems']} elementów")
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: minimalnie {schema['minItems']} elementów")
        if "items" in schema:
            for index, item in enumerate(value):
                errors += schema_errors(schema["items"], item, f"{path}[{index}]")
    return errors


def provider_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool definitions for the LLM: input schema as parameters, output contract appended to the description."""
    described = []
    for tool in tools:
        description = tool["description"]
        if tool.get("returns"):
            description += f" Wynik (data): {json.dumps(tool['returns'], ensure_ascii=False, separators=(',', ':'))}"
        described.append({"name": tool["name"], "description": description, "parameters": tool["parameters"]})
    return described


def post_json(url: str, headers: dict[str, str], payload: dict[str, Any], label: str) -> dict[str, Any]:
    http_request = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=1800) as response:
            return json.load(response)
    except error.HTTPError as http_error:
        details = http_error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"{label} zwrócił HTTP {http_error.code}: {details or http_error.reason}") from http_error


def parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def run_tool(
    execute: ToolExecutor, tools: list[dict[str, Any]], name: str, arguments: dict[str, Any]
) -> Generator[dict[str, Any], None, str]:
    """Validates arguments against the tool contract, runs the tool and wraps the outcome in the result envelope."""
    yield {"type": "tool_call", "name": name, "arguments": arguments}
    started = time.monotonic()
    extra = None
    tool = next((item for item in tools if item["name"] == name), None)
    try:
        if tool is None:
            raise ToolError("unknown_tool", f"Nie ma narzędzia {name}.")
        problems = schema_errors(tool["parameters"], arguments)
        if problems:
            raise ToolError("invalid_arguments", "Argumenty niezgodne z kontraktem narzędzia.", problems)
        outcome = execute(name, arguments)
        if inspect.isgenerator(outcome):
            outcome = yield from outcome
        data, extra = outcome
        envelope: dict[str, Any] = {"ok": True, "data": data}
    except ToolError as tool_error:
        envelope = {"ok": False, "error": {"code": tool_error.code, "message": str(tool_error), "details": tool_error.details}}
    except Exception as tool_error:  # noqa: BLE001 - errors are reported back to the model
        envelope = {"ok": False, "error": {"code": "tool_failed", "message": str(tool_error), "details": []}}
    envelope["ms"] = round((time.monotonic() - started) * 1000)
    event: dict[str, Any] = {"type": "tool_result", "name": name, "ok": envelope["ok"], "ms": envelope["ms"]}
    if not envelope["ok"]:
        event["error"] = envelope["error"]["message"]
    yield event
    if extra:
        yield extra
    return json.dumps(envelope, ensure_ascii=False)


def usage_event(model: str, started: float, usage: dict[str, Any] | None) -> dict[str, Any]:
    """Telemetry of one LLM call inside the tool loop (consumers may ignore it)."""
    usage = usage or {}
    return {
        "type": "usage",
        "model": model,
        "llm_ms": round((time.monotonic() - started) * 1000),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def run_openai(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    execute: ToolExecutor,
    max_steps: int,
) -> Iterator[dict[str, Any]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Brak OPENAI_API_KEY.")
    url = f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/responses"
    items: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in messages]
    function_tools = [{"type": "function", **tool} for tool in provider_tools(tools)]
    for _ in range(max_steps):
        started = time.monotonic()
        result = post_json(
            url,
            {"Authorization": f"Bearer {api_key}"},
            {"model": model, "instructions": system, "input": items, "tools": function_tools},
            "OpenAI",
        )
        yield usage_event(model, started, result.get("usage"))
        calls = []
        for output in result.get("output", []):
            items.append(output)
            if output.get("type") == "function_call":
                calls.append(output)
            elif output.get("type") == "message":
                text = "".join(
                    part.get("text", "") for part in output.get("content", []) if part.get("type") == "output_text"
                )
                if text.strip():
                    yield {"type": "text", "content": text}
        if not calls:
            return
        for call in calls:
            output = yield from run_tool(execute, tools, call["name"], parse_arguments(call.get("arguments")))
            items.append({"type": "function_call_output", "call_id": call["call_id"], "output": output})
    yield {"type": "text", "content": "(Przerwano: osiągnięto limit kroków agenta.)"}


def run_anthropic(
    model: str,
    system: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    execute: ToolExecutor,
    max_steps: int,
) -> Iterator[dict[str, Any]]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Brak ANTHROPIC_API_KEY.")
    conversation: list[dict[str, Any]] = [{"role": m["role"], "content": m["content"]} for m in messages]
    anthropic_tools = [
        {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
        for tool in provider_tools(tools)
    ]
    for _ in range(max_steps):
        started = time.monotonic()
        result = post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": 16000, "system": system, "messages": conversation, "tools": anthropic_tools},
            "Claude",
        )
        yield usage_event(model, started, result.get("usage"))
        content = result.get("content", [])
        conversation.append({"role": "assistant", "content": content})
        for block in content:
            if block.get("type") == "text" and block.get("text", "").strip():
                yield {"type": "text", "content": block["text"]}
        calls = [block for block in content if block.get("type") == "tool_use"]
        if not calls:
            return
        tool_results = []
        for call in calls:
            output = yield from run_tool(execute, tools, call["name"], parse_arguments(call.get("input")))
            tool_results.append({"type": "tool_result", "tool_use_id": call["id"], "content": output})
        conversation.append({"role": "user", "content": tool_results})
    yield {"type": "text", "content": "(Przerwano: osiągnięto limit kroków agenta.)"}


def run_agent(
    provider: str,
    model: str,
    system: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    execute: ToolExecutor,
    max_steps: int = 16,
) -> Iterator[dict[str, Any]]:
    runner = {"openai": run_openai, "anthropic": run_anthropic}.get(provider)
    if runner is None:
        raise RuntimeError(f"Dostawca {provider} nie obsługuje wywoływania narzędzi.")
    yield from runner(model, system, messages, tools, execute, max_steps)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ToolError("invalid_model_output", "Model nie zwrócił obiektu JSON.") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as decode_error:
            raise ToolError("invalid_model_output", f"Niepoprawny JSON od modelu: {decode_error.msg}.") from None
    if not isinstance(value, dict):
        raise ToolError("invalid_model_output", "Model nie zwrócił obiektu JSON.")
    return value


def complete_json(
    provider: str, model: str, system: str, prompt: str, max_tokens: int = 32000
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Single non-tool call that must answer with a JSON object; returns (object, call telemetry)."""
    started = time.monotonic()
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Brak ANTHROPIC_API_KEY.")
        result = post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            {"model": model, "max_tokens": max_tokens, "system": system, "messages": [{"role": "user", "content": prompt}]},
            "Claude",
        )
        text = "".join(block.get("text", "") for block in result.get("content", []) if block.get("type") == "text")
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Brak OPENAI_API_KEY.")
        # json_object mode requires the word "json" in the input itself; instructions don't count.
        openai_input = prompt if "json" in prompt.lower() else f"{prompt}\n\nOdpowiedz wyłącznie obiektem JSON."
        result = post_json(
            f"{os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')}/responses",
            {"Authorization": f"Bearer {api_key}"},
            {"model": model, "instructions": system, "input": openai_input, "text": {"format": {"type": "json_object"}}},
            "OpenAI",
        )
        text = "".join(
            part.get("text", "")
            for output in result.get("output", [])
            if output.get("type") == "message"
            for part in output.get("content", [])
            if part.get("type") == "output_text"
        )
    else:
        raise RuntimeError(f"Dostawca {provider} nie jest obsługiwany.")
    usage = result.get("usage") or {}
    return parse_json_object(text), {
        "model": model,
        "llm_ms": round((time.monotonic() - started) * 1000),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }
