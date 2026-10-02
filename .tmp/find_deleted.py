import json
import subprocess
from pathlib import Path

query = (
    "SELECT json_build_object('messages', e.messages, 'split', e.split)::text "
    "FROM training_examples e JOIN corpora c ON c.id = e.corpus_id "
    "WHERE c.name = 'OWU - Exclussions'"
)
raw = subprocess.run(
    ["docker", "exec", "bielik-lab-db-1", "psql", "-U", "bielik", "-d", "bielik", "-At", "-c", query],
    capture_output=True,
    check=True,
).stdout.decode("utf-8")


def key(messages):
    return json.dumps(messages, ensure_ascii=False, sort_keys=True)


db_rows = [json.loads(line) for line in raw.split("\n") if line.strip()]
db_train = {}
for row in db_rows:
    if row["split"] == "train":
        db_train.setdefault(key(row["messages"]), 0)
        db_train[key(row["messages"])] += 1

export = [json.loads(line) for line in Path("data/exports/train.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
remaining = dict(db_train)
missing = []
for record in export:
    k = key(record["messages"])
    if remaining.get(k):
        remaining[k] -= 1
    else:
        missing.append(record)

extra = sum(remaining.values())
print("db rows:", len(db_rows), "db train:", sum(db_train.values()), "export:", len(export))
print("missing from db:", len(missing), "in db but not in export:", extra)
output = Path("data/exports/recovered-train.jsonl")
output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in missing), encoding="utf-8")
print("written:", output)
for record in missing[:3]:
    user = next((m["content"] for m in record["messages"] if m["role"] == "user"), "")
    print("-", user[:120].replace("\n", " "))
