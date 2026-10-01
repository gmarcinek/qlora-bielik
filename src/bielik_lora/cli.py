from __future__ import annotations

import argparse
from pathlib import Path

from bielik_lora.corpus import export_jsonl, import_jsonl, initialize_database
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
    elif arguments.command == "adapter" and arguments.adapter_command == "merge":
        merge_adapter(arguments.base_model, arguments.adapter, arguments.output)
    elif arguments.command == "adapter" and arguments.adapter_command == "modelfile":
        write_ollama_modelfile(arguments.gguf, arguments.output)


if __name__ == "__main__":
    main()