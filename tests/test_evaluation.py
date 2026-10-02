import json

from bielik_lora.evaluation import parse_answer, score_example, summarize


def entities(*items):
    return json.dumps({"entities": [dict(zip(("type", "text", "start", "end"), item)) for item in items]})


def test_perfect_prediction_is_exact_match():
    expected = entities(("DISEASE", "grypa", 0, 5))
    record = score_example(expected, expected)
    assert record["exact_match"] and record["json_valid"]
    assert record["strict"] == {"tp": [["DISEASE", "grypa", 0, 5]], "fp": [], "fn": []}


def test_wrong_offsets_count_only_in_strict_mode():
    record = score_example(entities(("DISEASE", "grypa", 0, 5)), entities(("DISEASE", "grypa", 1, 6)))
    assert len(record["strict"]["fp"]) == 1 and len(record["strict"]["fn"]) == 1
    assert record["relaxed"]["tp"] == [["DISEASE", "grypa"]]


def test_exact_match_compares_entity_types_only():
    expected = entities(("DISEASE", "grypa", 0, 5), ("RIDER_CODE", "X1", 9, 11))
    predicted = entities(("RIDER_CODE", "X-1", 40, 42), ("DISEASE", "grypy", 20, 25))
    assert score_example(expected, predicted)["exact_match"]
    wrong_type = entities(("RIDER", "X1", 9, 11), ("DISEASE", "grypa", 0, 5))
    assert not score_example(expected, wrong_type)["exact_match"]


def test_exact_match_requires_complete_entity_set():
    expected = entities(("DISEASE", "grypa", 0, 5), ("RIDER_CODE", "X1", 9, 11))
    assert not score_example(expected, entities(("DISEASE", "grypa", 0, 5)))["exact_match"]
    extra = entities(("DISEASE", "grypa", 0, 5), ("RIDER_CODE", "X1", 9, 11), ("DISEASE", "odra", 0, 4))
    assert not score_example(expected, extra)["exact_match"]
    assert not score_example(entities(), "brak json")["exact_match"]


def test_repeated_mentions_are_matched_as_multiset():
    expected = entities(("DISEASE", "grypa", 0, 5), ("DISEASE", "grypa", 10, 15))
    predicted = entities(("DISEASE", "grypa", 0, 5))
    record = score_example(expected, predicted)
    assert len(record["strict"]["tp"]) == 1 and len(record["strict"]["fn"]) == 1


def test_text_around_json_is_tolerated_but_not_strict():
    predicted = "Odpowiedź: " + entities(("DISEASE", "grypa", 0, 5))
    record = score_example(entities(("DISEASE", "grypa", 0, 5)), predicted)
    assert not record["json_valid"] and record["exact_match"]
    assert parse_answer("brak json") is None


def test_exclusions_without_offsets_are_scored():
    expected = json.dumps({"exclusions": [{"type": "EXCLUSSION", "text": "działań wojennych"}], "scopes": {}})
    record = score_example(expected, expected)
    assert record["strict"]["tp"] == [["EXCLUSSION", "działań wojennych"]]


def test_summary_metrics_and_negatives():
    records = [
        score_example(entities(("DISEASE", "grypa", 0, 5)), entities(("DISEASE", "grypa", 0, 5))),
        score_example(entities(("DISEASE", "odra", 0, 4)), entities()),
        score_example(entities(), entities()),
        score_example(entities(), entities(("RIDER_CODE", "X1", 0, 2))),
    ]
    summary = summarize(records)
    assert summary["strict"]["tp"] == 1 and summary["strict"]["fp"] == 1 and summary["strict"]["fn"] == 1
    assert summary["strict"]["f1"] == 0.5
    assert summary["negatives"] == {"examples": 2, "correct": 1}
    assert summary["per_type"]["RIDER_CODE"]["precision"] == 0.0
    assert summary["per_type"]["DISEASE"]["recall"] == 0.5
