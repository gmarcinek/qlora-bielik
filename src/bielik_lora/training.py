from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

METRIC_PREFIX = "BIELIK_METRIC "


def load_config(config_path: Path) -> dict[str, Any]:
    import yaml

    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _format_conversation(example: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def _read_flags(path: str | None, count: int) -> list[str | None]:
    if not path or not Path(path).exists():
        return [None] * count
    with Path(path).open(encoding="utf-8") as meta_file:
        flags = [json.loads(line).get("flag") for line in meta_file if line.strip()]
    return flags if len(flags) == count else [None] * count


def train(config_path: Path) -> None:
    """Fine-tune Bielik with LoRA or QLoRA from JSONL ChatML conversations."""
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    from bielik_lora.training_metrics import (
        DATASET_PREFIX,
        LORA_PREFIX,
        GpuSampler,
        example_groups,
        layer_index,
        length_stats,
    )

    class MetricsCallback(TrainerCallback):
        def __init__(self, validation: list[tuple[list[int], list[str]]]) -> None:
            self.validation = validation
            self.gpu = GpuSampler()
            self.model: Any = None
            self.layer_grads: dict[int, float] = {}
            self.lora_grad_norm: float | None = None
            self.lora_every = 1

        def emit(self, state: Any, logs: dict[str, Any]) -> None:
            metrics = {
                key: value
                for key, value in logs.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            entry = {"step": state.global_step, "max_steps": state.max_steps, "time": time.time(), **metrics}
            print(METRIC_PREFIX + json.dumps(entry), flush=True)

        def lora_parameters(self) -> list[tuple[str, Any]]:
            if self.model is None:
                return []
            return [(name, parameter) for name, parameter in self.model.named_parameters() if "lora_" in name]

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.model = kwargs.get("model")
            self.lora_every = max(1, state.max_steps // 24)
            torch.cuda.reset_peak_memory_stats()
            self.emit(state, {"num_train_epochs": args.num_train_epochs, "max_grad_norm": args.max_grad_norm})

        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            # Gradients here are already clipped and accumulated over the whole step.
            squares: dict[int, float] = {}
            total = 0.0
            for name, parameter in self.lora_parameters():
                if parameter.grad is None:
                    continue
                square = float(parameter.grad.detach().float().pow(2).sum())
                total += square
                if (layer := layer_index(name)) is not None:
                    squares[layer] = squares.get(layer, 0.0) + square
            self.layer_grads = {layer: square**0.5 for layer, square in squares.items()}
            self.lora_grad_norm = total**0.5 if squares else None

        def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
            logs = dict(logs or {})
            if "loss" in logs:
                logs.update(self.gpu.drain())
                if torch.cuda.is_available():
                    logs["vram_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
                    logs["vram_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
                    torch.cuda.reset_peak_memory_stats()
                logs.update(self.lora_norms(state))
            self.emit(state, logs)

        def lora_norms(self, state: Any) -> dict[str, float]:
            norms = {"lora_A": 0.0, "lora_B": 0.0}
            layer_b: dict[int, float] = {}
            for name, parameter in self.lora_parameters():
                square = float(parameter.detach().float().pow(2).sum())
                kind = "lora_A" if "lora_A" in name else "lora_B"
                norms[kind] += square
                if kind == "lora_B" and (layer := layer_index(name)) is not None:
                    layer_b[layer] = layer_b.get(layer, 0.0) + square
            if not layer_b:
                return {}
            if state.global_step % self.lora_every == 0 or state.global_step >= state.max_steps:
                layers = [
                    {"layer": layer, "b_norm": layer_b[layer] ** 0.5, "grad_norm": self.layer_grads.get(layer)}
                    for layer in sorted(layer_b)
                ]
                print(LORA_PREFIX + json.dumps({"step": state.global_step, "layers": layers}), flush=True)
            result = {"lora_a_norm": norms["lora_A"] ** 0.5, "lora_b_norm": norms["lora_B"] ** 0.5}
            if self.lora_grad_norm is not None:
                result["lora_grad_norm"] = self.lora_grad_norm
            return result

        def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            model = kwargs.get("model") or self.model
            if model is None or not self.validation:
                return
            was_training = model.training
            model.eval()
            device = model.get_input_embeddings().weight.device
            sums: dict[str, float] = {}
            tokens: dict[str, int] = {}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for input_ids, groups in self.validation:
                    if len(input_ids) < 2:
                        continue
                    ids = torch.tensor([input_ids], device=device)
                    loss = float(model(input_ids=ids, labels=ids).loss)
                    for group in groups:
                        sums[group] = sums.get(group, 0.0) + loss * (len(input_ids) - 1)
                        tokens[group] = tokens.get(group, 0) + len(input_ids) - 1
            if was_training:
                model.train()
            self.emit(state, {f"group_loss/{group}": sums[group] / tokens[group] for group in sums})

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.gpu.stop()

    config = load_config(config_path)
    model_config = config["model"]
    training_config = config["training"]
    qlora = training_config.get("quantization") == "4bit"
    model_kwargs: dict[str, Any] = {
        "token": os.getenv("HF_TOKEN"),
        "device_map": model_config.get("device_map", "auto"),
        "torch_dtype": torch.bfloat16,
    }
    if max_memory := model_config.get("max_memory"):
        model_kwargs["max_memory"] = max_memory
    if qlora:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=training_config.get("quantization_type", "nf4"),
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=training_config.get("double_quantization", True),
        )
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"], token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_config["name"], **model_kwargs)
    if qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_config.get("gradient_checkpointing", True)
        )
    model.config.use_cache = training_config.get("use_cache", False)
    dataset = load_dataset("json", data_files=config["data"])
    formatted = dataset.map(lambda example: _format_conversation(example, tokenizer))
    max_length = training_config["max_length"]
    token_ids = {
        split: tokenizer(formatted[split]["text"], add_special_tokens=False)["input_ids"] for split in formatted
    }
    validation: list[tuple[list[int], list[str]]] = []
    group_counts: dict[str, int] = {}
    if "validation" in formatted:
        flags = _read_flags(config.get("validation_meta"), len(formatted["validation"]))
        for ids, messages, flag in zip(token_ids["validation"], formatted["validation"]["messages"], flags):
            groups = example_groups(messages, flag)
            validation.append((ids[:max_length], groups))
            for group in groups:
                group_counts[group] = group_counts.get(group, 0) + 1
    dataset_stats = {
        "max_length": max_length,
        "splits": {split: length_stats([len(ids) for ids in token_ids[split]], max_length) for split in token_ids},
        "groups": group_counts,
    }
    print(DATASET_PREFIX + json.dumps(dataset_stats, ensure_ascii=False), flush=True)
    training_args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        per_device_eval_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        optim=training_config.get("optimizer", "paged_adamw_8bit"),
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=training_config.get("eval_steps", 25),
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 25),
        save_total_limit=training_config.get("save_total_limit"),
        bf16=True,
        max_length=training_config["max_length"],
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=training_config.get("gradient_checkpointing_kwargs"),
        use_cache=training_config.get("use_cache", False),
        torch_empty_cache_steps=training_config.get("empty_cache_steps"),
        dataset_text_field="text",
    )
    adapter_config = LoraConfig(
        r=config["lora"]["rank"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"],
        target_modules=config["lora"]["target_modules"],
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted["train"],
        eval_dataset=formatted.get("validation"),
        peft_config=adapter_config,
        processing_class=tokenizer,
        callbacks=[MetricsCallback(validation[:200])],
    )
    trainer.train()
    trainer.save_model(config["output_dir"])


def evaluate(config_path: Path) -> dict[str, float]:
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"], token=os.getenv("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"], token=os.getenv("HF_TOKEN"), device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, config["output_dir"])
    dataset = load_dataset("json", data_files={"validation": config["data"]["validation"]})
    tokenized = dataset["validation"].map(
        lambda example: tokenizer.apply_chat_template(example["messages"], truncation=True),
        remove_columns=dataset["validation"].column_names,
    )
    trainer = Trainer(model=model, args=TrainingArguments(output_dir="artifacts/evaluation"))
    return {key: float(value) for key, value in trainer.evaluate(tokenized).items()}


def merge_adapter(base_model_name: str, adapter_path: Path, output_path: Path) -> None:
    """Merge a PEFT adapter into its base model for later GGUF conversion."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, token=token, torch_dtype="auto", low_cpu_mem_usage=True
    )
    merged_model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)


def write_ollama_modelfile(gguf_path: Path, output_path: Path) -> None:
    """Write a Modelfile pointing Ollama at an already converted GGUF model."""
    output_path.write_text(
        f'FROM {gguf_path.resolve().as_posix()}\nPARAMETER temperature 0.7\n', encoding="utf-8"
    )