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

    class MetricsCallback(TrainerCallback):
        def emit(self, state: Any, logs: dict[str, Any]) -> None:
            metrics = {
                key: value
                for key, value in logs.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            entry = {"step": state.global_step, "max_steps": state.max_steps, "time": time.time(), **metrics}
            print(METRIC_PREFIX + json.dumps(entry), flush=True)

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.emit(state, {"num_train_epochs": args.num_train_epochs})

        def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
            self.emit(state, logs or {})

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
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=training_config.get("save_total_limit", 2),
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
        callbacks=[MetricsCallback()],
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