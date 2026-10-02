from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

PROGRESS_PREFIX = "BIELIK_EVAL "
LIST_FIELDS = ("entities", "exclusions")
BASE_CHECKPOINT = "base"


def parse_answer(content: str) -> dict[str, Any] | None:
    """Parse a JSON answer, tolerating text around the outermost object."""
    candidates = [content.strip()]
    start, end = content.find("{"), content.rfind("}")
    if 0 <= start < end:
        candidates.append(content[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def is_strict_json(content: str) -> bool:
    try:
        return isinstance(json.loads(content.strip()), dict)
    except json.JSONDecodeError:
        return False


def answer_items(answer: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not answer:
        return []
    return [
        item
        for field in LIST_FIELDS
        if isinstance(answer.get(field), list)
        for item in answer[field]
        if isinstance(item, dict)
    ]


def item_key(item: dict[str, Any], with_offsets: bool) -> tuple:
    key: tuple = (str(item.get("type", "")), str(item.get("text", "")))
    if with_offsets and "start" in item and "end" in item:
        key += (item.get("start"), item.get("end"))
    return key


def score_example(expected_text: str, predicted_text: str) -> dict[str, Any]:
    expected = parse_answer(expected_text)
    predicted = parse_answer(predicted_text)
    record: dict[str, Any] = {
        "json_valid": is_strict_json(predicted_text),
        "expected_empty": not answer_items(expected),
        "predicted_empty": not answer_items(predicted),
    }
    for mode, with_offsets in (("strict", True), ("relaxed", False)):
        expected_keys = Counter(item_key(item, with_offsets) for item in answer_items(expected))
        predicted_keys = Counter(item_key(item, with_offsets) for item in answer_items(predicted))
        record[mode] = {
            "tp": [list(key) for key in (expected_keys & predicted_keys).elements()],
            "fp": [list(key) for key in (predicted_keys - expected_keys).elements()],
            "fn": [list(key) for key in (expected_keys - predicted_keys).elements()],
        }
    # Same entity types in the same counts as the reference; text, offsets, order and formatting are ignored.
    expected_types = Counter(str(item.get("type", "")) for item in answer_items(expected))
    predicted_types = Counter(str(item.get("type", "")) for item in answer_items(predicted))
    record["exact_match"] = expected is not None and predicted is not None and expected_types == predicted_types
    return record


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if precision is not None and recall is not None
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    summary: dict[str, Any] = {
        "examples": count,
        "json_valid": sum(record["json_valid"] for record in records) / count if count else None,
        "exact_match": sum(record["exact_match"] for record in records) / count if count else None,
    }
    for mode in ("strict", "relaxed"):
        totals = {bucket: sum(len(record[mode][bucket]) for record in records) for bucket in ("tp", "fp", "fn")}
        summary[mode] = {**totals, **precision_recall_f1(**totals)}
    per_type: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    per_type_relaxed: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for record in records:
        for bucket in ("tp", "fp", "fn"):
            for key in record["strict"][bucket]:
                per_type[key[0]][bucket] += 1
            for key in record["relaxed"][bucket]:
                per_type_relaxed[key[0]][bucket] += 1
    summary["per_type"] = {
        entity_type: {**counts, **precision_recall_f1(**counts)}
        for entity_type, counts in sorted(per_type.items())
    }
    summary["per_type_relaxed"] = {
        entity_type: {**counts, **precision_recall_f1(**counts)}
        for entity_type, counts in sorted(per_type_relaxed.items())
    }
    negatives = [record for record in records if record["expected_empty"]]
    summary["negatives"] = {
        "examples": len(negatives),
        "correct": sum(record["predicted_empty"] for record in negatives),
    }
    return summary


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def vram() -> str:
    import torch

    if not torch.cuda.is_available():
        return "brak CUDA"
    free, total = torch.cuda.mem_get_info()
    return f"VRAM {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB"


def mapped_file_bytes() -> float:
    """Bytes of memory-mapped files resident in this process (safetensors are read via mmap)."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("RssFile:"):
                return float(line.split()[1]) * 1024
    except OSError:
        pass
    return 0.0


@contextmanager
def heartbeat(label: str, total_bytes: float = 0, every: float = 20.0) -> Iterator[None]:
    """Log elapsed time, file bytes read and VRAM periodically while a long blocking call runs."""
    stop = threading.Event()
    started = time.time()

    def beat() -> None:
        while not stop.wait(every):
            elapsed = time.time() - started
            read = mapped_file_bytes()
            details = f"{elapsed:.0f} s, {vram()}"
            if total_bytes and read:
                rate = read / elapsed
                eta = (total_bytes - read) / rate if rate else 0
                details = (
                    f"wczytano {read / 2**30:.1f}/{total_bytes / 2**30:.1f} GiB "
                    f"({read / total_bytes:.0%}), {rate / 2**20:.0f} MB/s, "
                    f"ETA {max(eta, 0) / 60:.1f} min, {details}"
                )
            log(f"{label}… {details}")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


def load_base_model(base_model: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}, {vram()}")
    else:
        log("UWAGA: CUDA niedostępna, model zostanie załadowany na CPU")
    shards = sorted(Path(base_model).glob("*.safetensors"))
    total_bytes = sum(shard.stat().st_size for shard in shards)
    size = total_bytes / 2**30
    log(f"Model bazowy: {base_model} ({len(shards)} plików, {size:.1f} GiB bf16)")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    log("Tokenizer załadowany")
    log("Ładowanie wag z kwantyzacją 4-bit NF4 (odczyt z dysku, kilka minut)")
    started = time.time()
    with heartbeat("Ładowanie wag", total_bytes):
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            max_memory={0: "14GiB", "cpu": "40GiB"},
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
        )
    placement = Counter(str(device) for device in getattr(model, "hf_device_map", {}).values())
    log(f"Wagi załadowane w {time.time() - started:.0f} s, {vram()}, rozmieszczenie modułów: {dict(placement)}")
    if any(device in {"cpu", "disk"} for device in placement):
        log("UWAGA: część modelu jest poza GPU, generowanie będzie wolne")
    return model, tokenizer


def evaluate_adapter(
    base_model: str,
    adapter: Path,
    data: Path,
    output: Path,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Generate answers with a LoRA adapter and score them against reference JSON."""
    from peft import PeftModel

    model, tokenizer = load_base_model(base_model)
    model = PeftModel.from_pretrained(model, adapter)
    return generate_and_score(model, tokenizer, adapter, data, output, max_new_tokens)


def generate_and_score(
    model: Any,
    tokenizer: Any,
    adapter: Path,
    data: Path,
    output: Path,
    max_new_tokens: int,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    import torch

    model.eval()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()
    progress = {"checkpoint": checkpoint} if checkpoint else {}
    label = checkpoint or adapter.name
    log(f"[{label}] Generowanie dla {len(rows)} przykładów z {data}")
    print(PROGRESS_PREFIX + json.dumps({**progress, "done": 0, "total": len(rows), "elapsed": 0.0}), flush=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file:
        for index, row in enumerate(rows):
            messages = row["messages"]
            last = max(position for position, message in enumerate(messages) if message["role"] == "assistant")
            expected_text = messages[last]["content"]
            inputs = tokenizer.apply_chat_template(
                messages[:last],
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            budget = min(max_new_tokens, len(tokenizer(expected_text)["input_ids"]) * 2 + 64)
            example_started = time.time()
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=budget,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )
            prompt_length = inputs["input_ids"].shape[1]
            predicted_text = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
            record = {
                "index": index,
                **score_example(expected_text, predicted_text),
                "expected": expected_text,
                "predicted": predicted_text,
            }
            records.append(record)
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            elapsed = time.time() - started
            eta = elapsed / (index + 1) * (len(rows) - index - 1)
            relaxed = record["relaxed"]
            log(
                f"[{label}] {index + 1}/{len(rows)} · prompt {prompt_length} tok · "
                f"wygenerowano {generated.shape[1] - prompt_length}/{budget} tok · "
                f"{time.time() - example_started:.1f} s · JSON {'ok' if record['json_valid'] else 'BŁĄD'} · "
                f"tol. TP {len(relaxed['tp'])} FP {len(relaxed['fp'])} FN {len(relaxed['fn'])} · "
                f"ETA {eta / 60:.1f} min"
            )
            print(
                PROGRESS_PREFIX
                + json.dumps({**progress, "done": index + 1, "total": len(rows), "elapsed": time.time() - started}),
                flush=True,
            )
    summary = {**summarize(records), "adapter": str(adapter), "data": str(data)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def generate_and_score_ollama(
    model_name: str,
    data: Path,
    output: Path,
    max_new_tokens: int,
    checkpoint: str,
) -> dict[str, Any]:
    """Same scoring and output layout as generate_and_score, generated by a merged model in Ollama."""
    from urllib import request

    from bielik_lora.ollama_export import ollama_url

    rows = [json.loads(line) for line in data.read_text(encoding="utf-8").splitlines() if line.strip()]
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.time()
    url = f"{ollama_url()}/api/chat"
    log(f"[{checkpoint}] Generowanie w Ollamie ({model_name}) dla {len(rows)} przykładów z {data}")
    print(PROGRESS_PREFIX + json.dumps({"checkpoint": checkpoint, "done": 0, "total": len(rows), "elapsed": 0.0}), flush=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as predictions_file:
        for index, row in enumerate(rows):
            messages = row["messages"]
            last = max(position for position, message in enumerate(messages) if message["role"] == "assistant")
            expected_text = messages[last]["content"]
            payload = {
                "model": model_name,
                "messages": messages[:last],
                "stream": False,
                "options": {"temperature": 0, "num_predict": max_new_tokens, "num_ctx": 32768},
            }
            example_started = time.time()
            http_request = request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(http_request, timeout=900) as response:
                result = json.load(response)
            predicted_text = str(result.get("message", {}).get("content", "")).strip()
            record = {
                "index": index,
                **score_example(expected_text, predicted_text),
                "expected": expected_text,
                "predicted": predicted_text,
            }
            records.append(record)
            predictions_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            predictions_file.flush()
            elapsed = time.time() - started
            eta = elapsed / (index + 1) * (len(rows) - index - 1)
            relaxed = record["relaxed"]
            log(
                f"[{checkpoint}] {index + 1}/{len(rows)} · prompt {result.get('prompt_eval_count', '?')} tok · "
                f"wygenerowano {result.get('eval_count', '?')} tok · {time.time() - example_started:.1f} s · "
                f"JSON {'ok' if record['json_valid'] else 'BŁĄD'} · "
                f"tol. TP {len(relaxed['tp'])} FP {len(relaxed['fp'])} FN {len(relaxed['fn'])} · "
                f"ETA {eta / 60:.1f} min"
            )
            print(
                PROGRESS_PREFIX
                + json.dumps(
                    {"checkpoint": checkpoint, "done": index + 1, "total": len(rows), "elapsed": time.time() - started}
                ),
                flush=True,
            )
    # Free VRAM for adapter checkpoints evaluated after this one.
    unload = request.Request(
        f"{ollama_url()}/api/generate",
        data=json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        request.urlopen(unload, timeout=60).close()
    except OSError:
        pass
    summary = {**summarize(records), "adapter": f"ollama:{model_name}", "data": str(data)}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def evaluate_checkpoints(
    base_model: str,
    adapter_root: Path,
    checkpoints: list[str],
    data: Path,
    output: Path,
    max_new_tokens: int = 2048,
    ollama_models: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate selected checkpoints on one loaded base model and save a comparison report."""
    from peft import PeftModel

    ollama_models = ollama_models or {}
    # Merged models run in Ollama first, so its VRAM is released before the base model loads.
    checkpoints = sorted(checkpoints, key=lambda checkpoint: checkpoint not in ollama_models)
    log(f"Ewaluacja checkpointów: {', '.join(checkpoints)}")
    base = tokenizer = None
    model = None
    results = []
    output.mkdir(parents=True, exist_ok=True)
    for checkpoint in checkpoints:
        if checkpoint in ollama_models:
            summary = generate_and_score_ollama(
                ollama_models[checkpoint], data, output / checkpoint, max_new_tokens, checkpoint
            )
            results.append({"checkpoint": checkpoint, **summary})
            log(
                f"[{checkpoint}] WYNIK: strict F1={summary['strict']['f1']} "
                f"relaxed F1={summary['relaxed']['f1']} JSON={summary['json_valid']}"
            )
            (output / "comparison.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            continue
        if base is None:
            base, tokenizer = load_base_model(base_model)
        if checkpoint == BASE_CHECKPOINT:
            log(f"[{checkpoint}] Model bazowy bez adaptera LoRA")
            with model.disable_adapter() if model is not None else nullcontext():
                summary = generate_and_score(
                    model or base, tokenizer, Path(base_model), data, output / checkpoint, max_new_tokens, checkpoint
                )
        else:
            adapter = adapter_root if checkpoint == "final" else adapter_root / checkpoint
            log(f"[{checkpoint}] Ładowanie adaptera LoRA {adapter}")
            if model is None:
                model = PeftModel.from_pretrained(base, adapter, adapter_name=checkpoint)
            else:
                model.load_adapter(adapter, adapter_name=checkpoint)
                model.set_adapter(checkpoint)
            summary = generate_and_score(
                model, tokenizer, adapter, data, output / checkpoint, max_new_tokens, checkpoint
            )
        results.append({"checkpoint": checkpoint, **summary})
        log(
            f"[{checkpoint}] WYNIK: strict F1={summary['strict']['f1']} "
            f"relaxed F1={summary['relaxed']['f1']} JSON={summary['json_valid']}"
        )
        (output / "comparison.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return results
