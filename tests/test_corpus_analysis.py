import json

from bielik_lora.agent import schema_errors
from bielik_lora.corpus_agent import TOOLS, CorpusAgentTools
from bielik_lora.corpus_analysis import analyze

SYSTEM = {"role": "system", "content": "NER"}


def row(index, user, entities, flag, split="train", system=True):
    answer = json.dumps({"entities": entities, "summary": "s"}, ensure_ascii=False)
    messages = [*([SYSTEM] if system else []), {"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    return {"id": index, "split": split, "flag": flag, "messages": messages}


FRAGMENT = "Ubezpieczyciel nie wyp\u0142aci \u015bwiadczenia przy sportach motorowych i lotniczych."
ROWS = [
    row(1, f"Wyodr\u0119bnij EXCLUSION.\n\n{FRAGMENT}", [{"type": "EXCLUSION", "dow\u00f3d": "sportach motorowych"}], "positive"),
    row(2, f"Znajd\u017a EXCLUSION.\n\n{FRAGMENT}", [{"type": "EXCLUSION", "dow\u00f3d": "sporty wodne"}], "positive", "validation"),
    row(3, "Znajd\u017a NIP.\n\nNIP: 525-26-85-595.", [{"type": "EXCLUSION", "dow\u00f3d": "NIP"}], "positive", system=False),
    row(4, "Znajd\u017a EXCLUSION.\n\nOchrona trwa ca\u0142\u0105 dob\u0119 na ca\u0142ym \u015bwiecie bez wyj\u0105tk\u00f3w.", [], "negative"),
    row(5, "Znajd\u017a EXCLUSION.\n\nOchrona trwa ca\u0142\u0105 dob\u0119 na ca\u0142ym \u015bwiecie bez wyj\u0105tk\u00f3w.", [], "negative"),
    row(6, "Pusty pozytyw", [], "positive"),
    {"id": 7, "split": "train", "flag": "positive", "messages": [SYSTEM, {"role": "user", "content": "x"}, {"role": "assistant", "content": "{zepsuty"}]},
    row(8, "Wypisz NIP.\n\nNIP firmy: 526-10-40-567.", [{"type": "NIP", "dowód": "526-10-40-567"}], "positive", "validation"),
]


def test_balance_by_type_flag_and_split():
    analysis = analyze(ROWS, max_tokens=10)
    exclusion = next(item for item in analysis["types"] if item["type"] == "EXCLUSION")
    assert (exclusion["positive"], exclusion["negative"], exclusion["validation"]) == (3, 2, 1)
    assert "ma\u0142o przyk\u0142ad\u00f3w (5 < 10)" in exclusion["warnings"]
    assert analysis["splits"]["train"]["positive"] == 4 and analysis["flags"]["negative"] == 2
    assert analysis["system_prompt"] == {"with": 7, "without": 1}
    issues = {check: [entry["id"] for entry in item["examples"]] for check, item in analysis["issues"].items()}
    assert issues["quote_not_in_text"] == ["2"]
    assert issues["type_not_requested"] == ["3"]
    assert issues["flag_mismatch"] == ["6"]
    assert issues["invalid_json"] == ["7"]
    assert issues["split_leakage"] == ["2"]
    assert sorted(issues["duplicate_user"]) == ["4", "5"]
    # Every example except the tiny broken one (#7) exceeds 10 estimated tokens.
    assert analysis["issues"]["too_long"]["count"] == len(ROWS) - 1


def test_agent_corpus_balance_tool_contract():
    pending = [row(8, "Znajd\u017a EXCLUSION.\n\nInny fragment o nurkowaniu i paralotniarstwie.", [], "negative")]
    tools = CorpusAgentTools(ROWS, pending=pending, max_tokens=1024)
    contract = next(tool for tool in TOOLS if tool["name"] == "corpus_balance")
    summary, event = tools("corpus_balance", {})
    assert event is None and schema_errors(contract["returns"], summary) == []
    assert summary["detail"] == "summary" and "types" not in summary["corpus"]
    assert summary["corpus"]["splits"]["validation"] == {"count": 2, "pct": 25.0}
    assert summary["corpus"]["defective"]["by_check"]["quote_not_in_text"] == {"count": 1, "ids": ["2"]}
    assert sorted(summary["corpus"]["defective"]["by_check"]["duplicate_user"]["ids"]) == ["4", "5"]
    result, _ = tools("corpus_balance", {"detail": "full"})
    assert schema_errors(contract["returns"], result) == []
    assert result["proposals"]["flags"]["negative"] == 1
    assert result["corpus"]["issues"]["quote_not_in_text"] == {"count": 1, "examples": ["2"]}


def test_agent_updates_and_rejects_proposals():
    pending = [
        {**row(10, "Wy\u0142\u0105czenie: \u201esamob\u00f3jstwa\u201d.", [{"type": "EXCLUSION", "dow\u00f3d": "samob\u00f3jstwa"}], "positive"), "batch": "proposal-1"},
        {**row(11, "Inna propozycja.", [], "negative"), "batch": "proposal-1"},
    ]
    tools = CorpusAgentTools(ROWS, pending=pending)
    listed, _ = tools("list_proposals", {"batch": "proposal-1"})
    assert listed["matched"] == 2 and listed["proposals"][0]["system"] == "NER"
    user = "Artyku\u0142 9. Wy\u0142\u0105czenia\n\nUbezpieczyciel nie wyp\u0142aci \u015bwiadczenia, je\u017celi szkoda powsta\u0142a wskutek samob\u00f3jstwa."
    result, event = tools(
        "update_proposals",
        {"updates": [{"id": "10", "user": user}, {"id": "11", "flag": "positive"}, {"id": "nope"}]},
    )
    assert result["updated"] == 1 and [item["index"] for item in result["rejected"]] == [1, 2]
    assert event["type"] == "proposal_updates" and event["updated"][0]["messages"][1]["content"] == user
    assert tools("list_proposals", {"query": "Artyku\u0142 9"})[0]["matched"] == 1
    rejected, event = tools("reject_proposals", {"ids": ["11", "x"]})
    assert rejected == {"rejected": 1, "unknown": ["x"]} and event == {"type": "proposal_rejections", "ids": ["11"]}
    contracts = {tool["name"]: tool for tool in TOOLS}
    assert schema_errors(contracts["update_proposals"]["returns"], result) == []
    assert schema_errors(contracts["reject_proposals"]["returns"], rejected) == []
    assert tools.validate({"user": pending[0]["messages"][1]["content"], "assistant": "{}", "flag": "negative"}).startswith("Duplikat")


def test_next_split_follows_corpus_ratio():
    ratio = {"train": 50, "validation": 25, "test": 25}
    tools = CorpusAgentTools(ROWS, split_ratio=ratio)
    assert [tools.next_split() for _ in range(3)] == ["test", "test", "validation"]
    assert tools.balance()["split_target"] == ratio
    assert CorpusAgentTools(ROWS).next_split() == "train"


def test_fix_proposal_replaces_original_without_duplicate():
    tools = CorpusAgentTools(ROWS)
    user = f"Wyodr\u0119bnij EXCLUSION.\n\n{FRAGMENT}"
    answer = json.dumps({"entities": [{"type": "EXCLUSION", "dow\u00f3d": "sportach motorowych"}], "summary": "s"}, ensure_ascii=False)
    fix = {"user": user, "assistant": answer, "flag": "positive"}
    assert tools.propose([fix])[0]["rejected"][0]["reason"].startswith("Duplikat")
    result, event = tools.propose([{**fix, "replaces": "1"}, {**fix, "replaces": "1"}, {**fix, "replaces": "999"}])
    assert result["accepted"] == 1 and [item["index"] for item in result["rejected"]] == [1, 2]
    assert event["examples"][0]["replaces"] == "1" and event["examples"][0]["split"] == "train"


def test_fix_accepts_empty_followup_and_missing_common_keys():
    tools = CorpusAgentTools(ROWS)
    answer = json.dumps({"entities": [{"type": "EXCLUSION", "dow\u00f3d": "sportach motorowych"}]}, ensure_ascii=False)
    fix = {"user": f"Wyodr\u0119bnij EXCLUSION.\n\n{FRAGMENT}", "assistant": answer, "flag": "positive", "replaces": "1",
           "followup": {"user": "", "assistant": ""}}
    result, event = tools.propose([fix])
    assert result["accepted"] == 1 and result["rejected"] == []
    assert "summary" in result["warnings"][0]["warning"]
    assert event["examples"][0]["turns"] == []


def test_park_examples_moves_only_accepted_corpus_examples():
    tools = CorpusAgentTools(ROWS, pending=[row(99, "prop", [], "negative")])
    result, event = tools("park_examples", {"ids": ["1", "99", "nope"]})
    assert result == {"moved": 1, "unknown": ["99", "nope"]}
    assert event == {"type": "examples_parked", "ids": ["1"]}
    assert all(str(item["id"]) != "1" for item in tools.rows)
