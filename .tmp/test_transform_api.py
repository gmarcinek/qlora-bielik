import json
import urllib.request

API = "http://localhost:8000"
CORPUS_ID = "27a797fe-4912-40bd-901b-7bd5de7437e7"
ENTRY = {
    "type": "BENEFIT_TABLE_ENTRY",
    "name": "abdominalOrPelvicVesselInjuryBenefit",
    "description": "Tabela przypisuje 10% sumy ubezpieczenia pozycji „Uszkodzenie dużych naczyń”.",
    "evidence": "Tabela uszkodzeń ciała wskutek nieszczęśliwego wypadku, punkt 37",
}


def call(method, path, body=None):
    request = urllib.request.Request(
        API + path,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


original = "To jest pozycja tabeli świadczeń. BENEFIT_TABLE_ENTRY\n" + json.dumps(ENTRY, ensure_ascii=False)
example = call("POST", f"/api/corpora/{CORPUS_ID}/examples", {
    "split": "train", "flag": "positive",
    "messages": [{"role": "user", "content": "TEST transformacji"}, {"role": "assistant", "content": original}],
})
ids = [example["id"]]
preview = call("POST", "/api/examples/bulk/transform", {"example_ids": ids, "transform": "wrap_entities_summary", "dry_run": True})
print("preview matched:", preview["matched"], "after:", preview["samples"][0]["after"][:90], "...")
applied = call("POST", "/api/examples/bulk/transform", {"example_ids": ids, "transform": "wrap_entities_summary", "dry_run": False})
stored = next(e for e in call("GET", f"/api/corpora/{CORPUS_ID}/examples") if e["id"] == ids[0])
print("applied:", applied["matched"], "stored keys:", list(json.loads(stored["messages"][-1]["content"])))
again = call("POST", "/api/examples/bulk/transform", {"example_ids": ids, "transform": "wrap_entities_summary", "dry_run": True})
print("second run skipped:", again["skipped"])
print("revert:", call("POST", f"/api/revisions/{applied['revision_id']}/revert"))
stored = next(e for e in call("GET", f"/api/corpora/{CORPUS_ID}/examples") if e["id"] == ids[0])
print("restored original:", stored["messages"][-1]["content"] == original)
print("cleanup:", call("DELETE", f"/api/examples/{ids[0]}")["deleted"])
