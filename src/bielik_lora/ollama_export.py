from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from urllib import error, request

CHATML_TEMPLATE = (
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n"
)
PARAMETERS = {"num_ctx": 32768, "num_predict": 8192, "stop": ["<|im_start|>", "<|im_end|>"], "temperature": 0}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def ollama_url(host: str | None = None) -> str:
    return (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")


def sha256_file(path: Path, chunk: int = 16 * 2**20) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    done = 0
    last_report = time.time()
    with path.open("rb") as file:
        while block := file.read(chunk):
            digest.update(block)
            done += len(block)
            if time.time() - last_report > 10:
                log(f"Suma kontrolna {done / size:.0%}")
                last_report = time.time()
    return digest.hexdigest()


def modelfile_text(gguf_name: str) -> str:
    stops = "".join(f"PARAMETER stop {stop}\n" for stop in PARAMETERS["stop"])
    return (
        f"FROM ./{gguf_name}\n"
        f'TEMPLATE "{CHATML_TEMPLATE}"\n'
        f"PARAMETER num_ctx {PARAMETERS['num_ctx']}\n"
        f"PARAMETER num_predict {PARAMETERS['num_predict']}\n"
        f"{stops}"
        f"PARAMETER temperature {PARAMETERS['temperature']}\n"
    )


def register(gguf: Path, name: str, host: str | None = None) -> None:
    """Upload a GGUF to Ollama as a blob and create a ChatML model from it."""
    base = ollama_url(host)
    (gguf.parent / f"Modelfile.{name.replace(':', '_')}").write_text(modelfile_text(gguf.name), encoding="utf-8")
    size = gguf.stat().st_size
    log(f"Liczenie SHA-256 dla {gguf.name} ({size / 2**30:.2f} GiB)")
    digest = f"sha256:{sha256_file(gguf)}"
    try:
        request.urlopen(request.Request(f"{base}/api/blobs/{digest}", method="HEAD"), timeout=30)
        log("Blob jest już w Ollamie, pomijam wysyłanie")
    except error.HTTPError as http_error:
        if http_error.code != 404:
            raise
        log(f"Wysyłanie do Ollamy ({base})")
        started = time.time()
        with gguf.open("rb") as file:
            upload = request.Request(
                f"{base}/api/blobs/{digest}",
                data=file,
                method="POST",
                headers={"Content-Length": str(size), "Content-Type": "application/octet-stream"},
            )
            request.urlopen(upload, timeout=3600).close()
        log(f"Wysłano w {time.time() - started:.0f} s")
    payload = {
        "model": name,
        "files": {gguf.name: digest},
        "template": CHATML_TEMPLATE,
        "parameters": PARAMETERS,
        "stream": False,
    }
    create = request.Request(
        f"{base}/api/create",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(create, timeout=1800) as response:
        result = json.load(response)
    if result.get("status") != "success":
        raise RuntimeError(f"Ollama create: {result}")
    log(f"Model {name} zarejestrowany w Ollamie")
