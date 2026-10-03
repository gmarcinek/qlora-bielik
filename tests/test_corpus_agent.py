import json

from bielik_lora.agent import parse_arguments, run_tool
from bielik_lora.corpus_agent import TOOLS, CorpusAgentTools

SYSTEM = "Wyodrębnij encje i zwróć JSON."


def row(index, user, entities, flag="positive", split="train"):
    answer = json.dumps({"entities": entities, "summary": "x"}, ensure_ascii=False)
    return {
        "id": index,
        "split": split,
        "flag": flag,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
    }


ROWS = [
    row(1, "NIP: 525-26-85-595", [{"type": "NIP", "text": "525-26-85-595"}]),
    row(2, "Brak numeru.", [], flag="negative"),
]


def test_overview_and_listing():
    tools = CorpusAgentTools(ROWS)
    overview = tools.overview()
    assert overview["examples"] == 2
    assert overview["entity_types"] == {"NIP": 1}
    assert overview["system_prompts"][0]["text"] == SYSTEM
    listed = tools.list_examples(entity_type="NIP")
    assert listed["matched"] == 1 and listed["examples"][0]["system_prompt_id"] == 0


def test_propose_validates_and_emits_drafts():
    tools = CorpusAgentTools(ROWS)
    result, event = tools.propose(
        [
            {"user": "REGON: 011417295", "assistant": '{"entities": [{"type": "REGON"}], "summary": "s"}', "flag": "positive"},
            {"user": "NIP: 525-26-85-595", "assistant": '{"entities": [], "summary": "s"}', "flag": "negative"},
            {"user": "Inny tekst", "assistant": "nie json", "flag": "negative"},
            {"user": "Pusty", "assistant": '{"entities": [], "summary": "s"}', "flag": "positive"},
        ]
    )
    assert result["accepted"] == 1
    assert [item["index"] for item in result["rejected"]] == [1, 2, 3]
    draft = event["examples"][0]
    assert draft["system"] == SYSTEM and draft["split"] == "train"
    # Already proposed texts count as duplicates within the same session.
    assert tools.validate({"user": "REGON: 011417295", "assistant": "{}", "flag": "positive"}).startswith("Duplikat")


def test_propose_empty_system_means_no_system_prompt():
    tools = CorpusAgentTools(ROWS)
    _, event = tools.propose(
        [{"system": "", "user": "REGON: 999", "assistant": '{"entities": [{"type": "REGON"}], "summary": "s"}', "flag": "positive"}]
    )
    assert event["examples"][0]["system"] == ""


def test_propose_followup_exchange():
    tools = CorpusAgentTools(ROWS)
    result, event = tools.propose(
        [
            {
                "user": "Tekst A",
                "assistant": '{"entities": [{"type": "NIP"}], "summary": "s"}',
                "followup": {"user": "A teraz tylko REGON.", "assistant": '{"entities": [], "summary": "s"}'},
                "flag": "negative",
            },
            {
                "user": "Tekst B",
                "assistant": '{"entities": [], "summary": "s"}',
                "followup": {"user": "Doprecyzuj.", "assistant": '{"entities": [], "summary": "s"}'},
                "flag": "positive",
            },
        ]
    )
    assert result["accepted"] == 1 and result["rejected"][0]["reason"].endswith("(followup)")
    draft = event["examples"][0]
    assert draft["user"] == "Tekst A" and draft["turns"] == [{"user": "A teraz tylko REGON.", "assistant": '{"entities": [], "summary": "s"}'}]


def test_run_tool_reports_errors_back_to_model():
    tools = CorpusAgentTools(ROWS)
    events = []
    generator = run_tool(tools, TOOLS, "list_examples", {"unknown": 1})
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        output = stop.value
    assert json.loads(output)["error"]["code"] == "invalid_arguments"
    assert events[0]["type"] == "tool_call" and events[1]["ok"] is False
    assert parse_arguments('{"limit": 2}') == {"limit": 2} and parse_arguments("zepsute") == {}


def test_get_examples_by_id_reports_status():
    pending = [{**row(9, "Propozycja: NIP 1", [{"type": "NIP"}]), "id": "p1", "batch": "proposal-x"}]
    tools = CorpusAgentTools(ROWS, pending=pending)
    result = tools.get_examples(["1", "p1", "nope"])
    assert [(item["id"], item["status"]) for item in result["examples"]] == [("1", "corpus"), ("p1", "proposal")]
    assert result["examples"][0]["user"] == "NIP: 525-26-85-595" and result["unknown"] == ["nope"]


def test_update_examples_edits_accepted_examples_with_checks():
    tools = CorpusAgentTools([*ROWS, row(3, "REGON 123", [{"type": "REGON"}], split="test")], pending=[{**row(9, "P", []), "id": "p1"}])
    result, event = tools.update_examples(
        [
            {"id": 3, "user": "Inny fragment: REGON 999", "assistant": '{"entities": [{"type": "REGON"}], "summary": "s"}'},
            {"id": "2", "user": "NIP: 525-26-85-595"},
            {"id": "p1", "user": "x"},
            {"id": "1", "flag": "negative"},
        ]
    )
    assert result["updated"] == 1
    reasons = [item["reason"] for item in result["rejected"]]
    assert reasons[0].startswith("Duplikat") and "update_proposals" in reasons[1] and "pustą listę" in reasons[2]
    updated = event["updated"][0]
    assert updated["id"] == "3" and updated["split"] == "test" and updated["flag"] == "positive"
    assert updated["messages"][0] == {"role": "system", "content": SYSTEM} and updated["messages"][1]["content"] == "Inny fragment: REGON 999"
    # The old text is free again, the new one is taken.
    assert tools.validate({"user": "REGON 123", "assistant": '{"entities": [], "summary": "s"}', "flag": "negative"}) is None
    assert tools.validate({"user": "Inny fragment: REGON 999", "assistant": "{}", "flag": "negative"}).startswith("Duplikat")
    _, event = tools.update_proposals([{"id": "1", "user": "z"}])
    assert event is None
