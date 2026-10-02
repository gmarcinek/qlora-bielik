from __future__ import annotations

import math
import re
import shutil
import subprocess
import threading
from statistics import mean
from typing import Any

from bielik_lora.evaluation import answer_items, parse_answer

DATASET_PREFIX = "BIELIK_DATASET "
LORA_PREFIX = "BIELIK_LORA "
LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")
GPU_QUERY = "utilization.gpu,temperature.gpu,power.draw,memory.used"
MAX_HISTOGRAM_BINS = 24


def percentile(sorted_values: list[int], fraction: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[min(len(sorted_values) - 1, math.ceil(fraction * len(sorted_values)) - 1)]


def length_stats(lengths: list[int], max_length: int) -> dict[str, Any]:
    ordered = sorted(lengths)
    truncated = sum(1 for length in ordered if length > max_length)
    longest = ordered[-1] if ordered else 0
    width = max(1, max_length // 8)
    bins = min(MAX_HISTOGRAM_BINS, max(1, math.ceil(max(longest, max_length) / width)))
    histogram = [0] * bins
    for length in ordered:
        histogram[min(bins - 1, length // width)] += 1
    return {
        "count": len(ordered),
        "mean": round(mean(ordered), 1) if ordered else 0,
        "p50": percentile(ordered, 0.5),
        "p90": percentile(ordered, 0.9),
        "p99": percentile(ordered, 0.99),
        "max": longest,
        "truncated": truncated,
        "truncated_pct": round(100 * truncated / len(ordered), 2) if ordered else 0,
        "bin_width": width,
        "histogram": histogram,
    }


def example_groups(messages: list[dict[str, Any]], flag: str | None) -> list[str]:
    """Groups used to break validation loss down by flag and expected entity type."""
    groups = [f"flag:{flag or 'unknown'}"]
    answer = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
    types = sorted({str(item.get("type")) for item in answer_items(parse_answer(answer)) if item.get("type")})
    groups += [f"type:{entity_type}" for entity_type in types] or ["type:(brak encji)"]
    return groups


def layer_index(parameter_name: str) -> int | None:
    match = LAYER_PATTERN.search(parameter_name)
    return int(match.group(1)) if match else None


def parse_gpu_sample(line: str) -> dict[str, float]:
    names = ("gpu_util", "gpu_temp", "gpu_power", "gpu_memory_used_mb")
    sample = {}
    for name, raw in zip(names, line.split(",")):
        try:
            sample[name] = float(raw.strip())
        except ValueError:
            continue
    return sample


class GpuSampler:
    """Polls nvidia-smi in the background so per-step values cover the whole step."""

    def __init__(self, interval: float = 3.0) -> None:
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.binary = shutil.which("nvidia-smi")
        if self.binary:
            threading.Thread(target=self.run, daemon=True).start()

    def read(self) -> dict[str, float]:
        try:
            result = subprocess.run(
                [self.binary, f"--query-gpu={GPU_QUERY}", "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return {}
        return parse_gpu_sample(result.stdout.strip().splitlines()[0]) if result.stdout.strip() else {}

    def run(self) -> None:
        while not self.stop_event.wait(self.interval):
            if sample := self.read():
                with self.lock:
                    self.samples.append(sample)

    def drain(self) -> dict[str, float]:
        with self.lock:
            samples, self.samples = self.samples, []
        if not samples:
            return {}

        def values(name: str) -> list[float]:
            return [sample[name] for sample in samples if name in sample]

        summary: dict[str, float] = {}
        for name, reduce in (("gpu_util", mean), ("gpu_temp", max), ("gpu_power", mean)):
            if found := values(name):
                summary[name] = round(reduce(found), 1)
        if memory := values("gpu_memory_used_mb"):
            summary["gpu_memory_used_gb"] = round(max(memory) / 1024, 2)
        return summary

    def stop(self) -> None:
        self.stop_event.set()
