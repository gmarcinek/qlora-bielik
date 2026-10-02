import json
import sys
import urllib.request

BASE = "http://localhost:8000"
model = sys.argv[1] if len(sys.argv) > 1 else json.load(urllib.request.urlopen(f"{BASE}/api/agent/models"))["default"]
print("model:", model)
body = {
    "corpus_id": "14f0f2d6-5280-478b-bfbc-11e17438e41b",
    "model": model,
    "messages": [{"role": "user", "content": "Zaproponuj 2 nowe przykłady: 1 pozytywny NIP i 1 negatywny NIP."}],
}
request = urllib.request.Request(
    f"{BASE}/api/agent/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST"
)
with urllib.request.urlopen(request, timeout=600) as response:
    for line in response:
        event = json.loads(line)
        if event["type"] == "drafts":
            for draft in event["examples"]:
                print("DRAFT", draft["flag"], draft["split"], repr(draft["user"][:120]), draft["assistant"][:200])
        elif event["type"] == "text":
            print("TEXT", event["content"][:400])
        else:
            print(event)
