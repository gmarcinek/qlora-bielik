from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from bielik_lora.evaluation import load_base_model, log


class AdapterServer:
    """Serves chat completions from the base model with an optional LoRA adapter."""

    def __init__(self, base_model: str, adapter: Path | None, name: str) -> None:
        self.base_model = base_model
        self.adapter = adapter
        self.name = name
        self.model: Any = None
        self.tokenizer: Any = None
        self.error: str | None = None
        self.generation_lock = threading.Lock()

    def load(self) -> None:
        try:
            model, tokenizer = load_base_model(self.base_model)
            if self.adapter is not None:
                from peft import PeftModel

                log(f"Ładowanie adaptera LoRA {self.adapter}")
                model = PeftModel.from_pretrained(model, self.adapter)
            model.eval()
            self.model, self.tokenizer = model, tokenizer
            log(f"Model {self.name} gotowy do czatu")
        except Exception as error:  # noqa: BLE001 - reported through /health
            self.error = str(error)
            log(f"BŁĄD ładowania modelu: {error}")
            raise

    def stream(self, messages: list[dict[str, str]], max_new_tokens: int):
        import torch
        from transformers import TextIteratorStreamer

        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        def generate() -> None:
            with torch.no_grad():
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                    streamer=streamer,
                )

        with self.generation_lock:
            log(f"Generowanie: prompt {inputs['input_ids'].shape[1]} tok")
            thread = threading.Thread(target=generate, daemon=True)
            thread.start()
            yield from streamer
            thread.join()


def serve(base_model: str, adapter: Path | None, name: str, port: int = 8080) -> None:
    server_state = AdapterServer(base_model, adapter, name)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def send_json(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path != "/health":
                self.send_json(404, {"error": "not found"})
                return
            self.send_json(
                200,
                {
                    "ready": server_state.model is not None,
                    "error": server_state.error,
                    "name": server_state.name,
                },
            )

        def do_POST(self) -> None:
            if self.path != "/chat":
                self.send_json(404, {"error": "not found"})
                return
            if server_state.model is None:
                self.send_json(503, {"error": server_state.error or "Model się ładuje."})
                return
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            messages = payload.get("messages") or []
            # Close-delimited NDJSON stream (HTTP/1.0 semantics of BaseHTTPRequestHandler).
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for chunk in server_state.stream(messages, int(payload.get("max_new_tokens", 2048))):
                if chunk:
                    self.wfile.write((json.dumps({"content": chunk}, ensure_ascii=False) + "\n").encode("utf-8"))
                    self.wfile.flush()

    http_server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    log(f"Serwer czatu nasłuchuje na porcie {port}, model {name}")
    server_state.load()
    threading.Event().wait()
