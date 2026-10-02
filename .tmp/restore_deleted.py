import json
import urllib.request
from pathlib import Path

CORPUS_ID = "27a797fe-4912-40bd-901b-7bd5de7437e7"
records = [
    json.loads(line)
    for line in Path("data/exports/recovered-train.jsonl").read_text(encoding="utf-8").split("\n")
    if line.strip()
]
payload = {
    "examples": [
        {"messages": record["messages"], "split": "train", "flag": "unclassified", "source": "przywrocone-po-usunieciu"}
        for record in records
    ]
}
request = urllib.request.Request(
    f"http://localhost:8000/api/corpora/{CORPUS_ID}/examples/import",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    print(response.status, response.read().decode("utf-8"))
