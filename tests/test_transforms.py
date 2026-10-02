import json

from bielik_lora.transforms import reformat_json, transform_messages, wrap_entities_summary

ENTRY = {
    "type": "BENEFIT_TABLE_ENTRY",
    "name": "abdominalOrPelvicVesselInjuryBenefit",
    "description": "Tabela przypisuje 10% sumy ubezpieczenia pozycji „Uszkodzenie {naczyń}”.",
    "evidence": "Tabela uszkodzeń ciała wskutek nieszczęśliwego wypadku, punkt 37",
}


def test_wraps_summary_and_single_object():
    content = "To jest pozycja tabeli świadczeń. BENEFIT_TABLE_ENTRY\n" + json.dumps(ENTRY, ensure_ascii=False)
    result, reason = wrap_entities_summary(content)
    assert reason == ""
    assert json.loads(result) == {"entities": [ENTRY], "summary": "To jest pozycja tabeli świadczeń. BENEFIT_TABLE_ENTRY"}
    assert list(json.loads(result)) == ["entities", "summary"]


def test_wraps_multiple_objects_and_bare_json():
    second = {**ENTRY, "name": "other"}
    content = "Opis\n" + json.dumps(ENTRY) + "\n" + json.dumps(second)
    assert json.loads(wrap_entities_summary(content)[0])["entities"] == [ENTRY, second]
    assert json.loads(wrap_entities_summary(json.dumps(ENTRY))[0]) == {"entities": [ENTRY], "summary": ""}


def test_skips_already_wrapped_and_unparseable():
    assert wrap_entities_summary(json.dumps({"entities": []}))[0] is None
    assert wrap_entities_summary("sam tekst bez JSON")[0] is None
    assert wrap_entities_summary('Opis {"type": "X"} i dalej tekst')[1] == "tekst po obiekcie JSON"
    assert wrap_entities_summary('Opis {"type": ')[1] == "niepoprawny JSON"


def test_transform_messages_only_touches_assistant():
    messages = [
        {"role": "system", "content": "Instrukcja {nie ruszać}"},
        {"role": "user", "content": "Pytanie"},
        {"role": "assistant", "content": "Opis " + json.dumps(ENTRY)},
    ]
    result, reason = transform_messages(messages, "wrap_entities_summary")
    assert reason == "" and result[:2] == messages[:2]
    assert json.loads(result[2]["content"])["summary"] == "Opis"
    assert transform_messages(messages[:2], "wrap_entities_summary") == (None, "brak odpowiedzi asystenta")


def test_pretty_and_compact_json_round_trip():
    compact = json.dumps({"entities": [ENTRY], "summary": "Opis"}, ensure_ascii=False, separators=(",", ":"))
    pretty, reason = reformat_json(compact, 2)
    assert reason == "" and pretty.startswith('{\n  "entities": [\n    {\n      "type"')
    assert "„Uszkodzenie" in pretty and json.loads(pretty) == json.loads(compact)
    assert reformat_json(pretty, 2) == (None, "już w tym formacie")
    assert reformat_json(pretty, None) == (compact, "")
    assert reformat_json("Opis " + compact, 2) == (None, "odpowiedź nie jest czystym JSON")
