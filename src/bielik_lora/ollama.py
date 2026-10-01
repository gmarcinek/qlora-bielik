from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


def generate(
    prompt: str,
    model: str | None = None,
    host: str | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """Send a single-turn prompt to a locally running Ollama instance."""
    payload: dict[str, Any] = {
        "model": model or os.getenv("OLLAMA_MODEL", "bielik"),
        "prompt": prompt,
        "stream": False,
    }
    if options:
        payload["options"] = options
    base_url = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    http_request = request.Request(
        f"{base_url}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=300) as response:
        result = json.load(response)
    return str(result["response"])


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    host: str | None = None,
    options: dict[str, Any] | None = None,
    response_format: dict[str, Any] | str | None = None,
) -> str:
    """Send a structured conversation to a locally running Ollama instance."""
    payload: dict[str, Any] = {
        "model": model or os.getenv("OLLAMA_MODEL", "bielik"),
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options
    if response_format:
        payload["format"] = response_format
    base_url = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    http_request = request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=300) as response:
        result = json.load(response)
    return str(result["message"]["content"])


def openai_chat(
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    model: str | None = None,
) -> str:
    """Send a structured request through the OpenAI Responses API."""
    url, headers, payload = openai_responses_request(
        messages,
        response_format,
        model,
        stream=False,
    )
    http_request = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=300) as response:
            result = json.load(response)
    except error.HTTPError as response_error:
        details = response_error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"OpenAI zwrócił HTTP {response_error.code}: {details or response_error.reason}"
        ) from response_error
    output_text = result.get("output_text")
    if isinstance(output_text, str):
        return output_text
    for output in result.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
    raise RuntimeError("OpenAI Responses API nie zwróciło tekstu odpowiedzi.")


def anthropic_chat(messages: list[dict[str, str]], model: str | None = None) -> str:
    """Send a request through the standard Anthropic Messages API."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Brak ANTHROPIC_API_KEY dla komercyjnej parafrazy.")
    system = "\n\n".join(
        message["content"] for message in messages if message["role"] == "system"
    )
    payload: dict[str, Any] = {
        "model": model or os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        "max_tokens": 8192,
        "messages": [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message["role"] != "system"
        ],
    }
    if system:
        payload["system"] = system
    http_request = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=300) as response:
            result = json.load(response)
    except error.HTTPError as response_error:
        details = response_error.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Claude zwrócił HTTP {response_error.code}: {details or response_error.reason}"
        ) from response_error
    return "".join(
        str(block.get("text", ""))
        for block in result.get("content", [])
        if block.get("type") == "text"
    )


def openai_chat_stream(
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    model: str | None = None,
):
    """Yield text deltas from the OpenAI Responses API SSE stream."""
    url, headers, payload = openai_responses_request(
        messages,
        response_format,
        model,
        stream=True,
    )
    http_request = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(http_request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    yield delta


def openai_responses_request(
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    model: str | None,
    stream: bool,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build a standard OpenAI Responses API request."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Brak OPENAI_API_KEY dla komercyjnej parafrazy.")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "input": [
            {
                "role": message["role"],
                "content": [{"type": "input_text", "text": message["content"]}],
            }
            for message in messages
        ],
        "text": {
            "format": {
            "type": "json_schema",
                "name": "prompt_result",
                "strict": True,
                "schema": response_format,
            }
        },
        "stream": stream,
    }
    payload["model"] = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    return url, headers, payload


def chat_stream(
    messages: list[dict[str, str]],
    model: str | None = None,
    host: str | None = None,
    options: dict[str, Any] | None = None,
):
    """Yield response fragments from Ollama's streaming chat endpoint."""
    payload: dict[str, Any] = {
        "model": model or os.getenv("OLLAMA_MODEL", "bielik"),
        "messages": messages,
        "stream": True,
    }
    if options:
        payload["options"] = options
    base_url = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    http_request = request.Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=300) as response:
        for line in response:
            event = json.loads(line)
            content = event.get("message", {}).get("content", "")
            if content:
                yield str(content)


def list_models(host: str | None = None) -> list[str]:
    """Return names of models currently installed in Ollama."""
    base_url = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    with request.urlopen(f"{base_url}/api/tags", timeout=15) as response:
        result = json.load(response)
    return [str(model["name"]) for model in result.get("models", [])]