from __future__ import annotations

import argparse
from pathlib import Path

from bielik_lora.corpus import export_jsonl, import_jsonl, initialize_database
from bielik_lora.evaluation import evaluate_adapter, evaluate_checkpoints
from bielik_lora.ollama import generate
from bielik_lora.training import evaluate, merge_adapter, train, write_ollama_modelfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bielik-lab")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Generate text through local Ollama")
    serve.add_argument("prompt")
    serve.add_argument("--model")
    serve.add_argument("--host")
    corpus = commands.add_parser("corpus", help="Manage ChatML training corpora")
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)
    init = corpus_commands.add_parser("init")
    init.add_argument("--database", type=Path, default=Path("data/corpus.sqlite3"))
    import_command = corpus_commands.add_parser("import")
    import_command.add_argument("input", type=Path)
    import_command.add_argument("--split", choices=("train", "validation", "test"), required=True)
    import_command.add_argument("--database", type=Path, default=Path("data/corpus.sqlite3"))
    export_command = corpus_commands.add_parser("export")
    export_command.add_argument("output", type=Path)
    export_command.add_argument("--split", choices=("train", "validation", "test"), required=True)
    export_command.add_argument("--database", type=Path, default=Path("data/corpus.sqlite3"))
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, default=Path("configs/qlora.yaml"))
    evaluation = commands.add_parser(
        "evaluate-adapter", help="Score adapter generations against reference JSON answers"
    )
    evaluation.add_argument("--base-model", required=True)
    evaluation.add_argument("--adapter", type=Path, required=True)
    evaluation.add_argument("--data", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--max-new-tokens", type=int, default=2048)
    comparison = commands.add_parser(
        "evaluate-checkpoints", help="Compare selected adapter checkpoints against reference JSON answers"
    )
    comparison.add_argument("--base-model", required=True)
    comparison.add_argument("--adapter-root", type=Path, required=True)
    comparison.add_argument("--checkpoints", nargs="+", required=True)
    comparison.add_argument("--data", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    comparison.add_argument("--max-new-tokens", type=int, default=2048)
    comparison.add_argument(
        "--ollama-model",
        action="append",
        default=[],
        metavar="CHECKPOINT=MODEL",
        help="Evaluate this checkpoint id through a merged model served by Ollama",
    )
    register = commands.add_parser("ollama-register", help="Upload a GGUF to Ollama and create a ChatML model")
    register.add_argument("--gguf", type=Path, required=True)
    register.add_argument("--name", required=True)
    serving = commands.add_parser("serve-adapter", help="Serve chat completions from a LoRA checkpoint")
    serving.add_argument("--base-model", required=True)
    serving.add_argument("--adapter", type=Path)
    serving.add_argument("--name", required=True)
    serving.add_argument("--port", type=int, default=8080)
    adapter = commands.add_parser("adapter", help="Prepare a trained adapter for local deployment")
    adapter_commands = adapter.add_subparsers(dest="adapter_command", required=True)
    merge = adapter_commands.add_parser("merge", help="Merge a PEFT adapter into the base model")
    merge.add_argument("--base-model", required=True)
    merge.add_argument("--adapter", type=Path, required=True)
    merge.add_argument("--output", type=Path, required=True)
    modelfile = adapter_commands.add_parser("modelfile", help="Write an Ollama Modelfile for a GGUF")
    modelfile.add_argument("--gguf", type=Path, required=True)
    modelfile.add_argument("--output", type=Path, default=Path("artifacts/Modelfile"))
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.command == "serve":
        print(generate(arguments.prompt, arguments.model, arguments.host))
    elif arguments.command == "corpus" and arguments.corpus_command == "init":
        initialize_database(arguments.database)
    elif arguments.command == "corpus" and arguments.corpus_command == "import":
        print(f"Imported {import_jsonl(arguments.database, arguments.input, arguments.split)} records")
    elif arguments.command == "corpus" and arguments.corpus_command == "export":
        print(f"Exported {export_jsonl(arguments.database, arguments.output, arguments.split)} records")
    elif arguments.command == "train":
        train(arguments.config)
    elif arguments.command == "evaluate":
        print(evaluate(arguments.config))
    elif arguments.command == "evaluate-adapter":
        evaluate_adapter(
            arguments.base_model,
            arguments.adapter,
            arguments.data,
            arguments.output,
            arguments.max_new_tokens,
        )
    elif arguments.command == "evaluate-checkpoints":
        evaluate_checkpoints(
            arguments.base_model,
            arguments.adapter_root,
            arguments.checkpoints,
            arguments.data,
            arguments.output,
            arguments.max_new_tokens,
            dict(item.split("=", 1) for item in arguments.ollama_model),
        )
    elif arguments.command == "ollama-register":
        from bielik_lora.ollama_export import register

        register(arguments.gguf, arguments.name)
    elif arguments.command == "serve-adapter":
        from bielik_lora.serving import serve

        serve(arguments.base_model, arguments.adapter, arguments.name, arguments.port)
    elif arguments.command == "adapter" and arguments.adapter_command == "merge":
        merge_adapter(arguments.base_model, arguments.adapter, arguments.output)
    elif arguments.command == "adapter" and arguments.adapter_command == "modelfile":
        write_ollama_modelfile(arguments.gguf, arguments.output)


if __name__ == "__main__":
    main()