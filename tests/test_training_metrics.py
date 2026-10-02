import json

from bielik_lora.training_metrics import example_groups, layer_index, length_stats, parse_gpu_sample


def test_length_stats_counts_truncation_and_histogram():
    stats = length_stats([100, 200, 900, 1100, 1500], max_length=1024)
    assert stats["count"] == 5
    assert stats["truncated"] == 2
    assert stats["truncated_pct"] == 40.0
    assert stats["max"] == 1500
    assert stats["bin_width"] == 128
    assert sum(stats["histogram"]) == 5
    assert stats["histogram"][1500 // 128] == 1


def test_length_stats_empty():
    stats = length_stats([], max_length=1024)
    assert stats["count"] == 0 and stats["truncated_pct"] == 0


def test_example_groups_uses_flag_and_entity_types():
    answer = {"entities": [{"type": "B"}, {"type": "A"}, {"type": "A"}], "summary": "x"}
    messages = [{"role": "user", "content": "q"}, {"role": "assistant", "content": json.dumps(answer)}]
    assert example_groups(messages, "positive") == ["flag:positive", "type:A", "type:B"]
    empty = [{"role": "user", "content": "q"}, {"role": "assistant", "content": '{"entities": []}'}]
    assert example_groups(empty, None) == ["flag:unknown", "type:(brak encji)"]


def test_layer_index_and_gpu_parsing():
    assert layer_index("base_model.model.model.layers.12.self_attn.q_proj.lora_B.default.weight") == 12
    assert layer_index("lm_head.weight") is None
    assert parse_gpu_sample("97, 68, 262.84, 9904") == {
        "gpu_util": 97.0,
        "gpu_temp": 68.0,
        "gpu_power": 262.84,
        "gpu_memory_used_mb": 9904.0,
    }
    assert parse_gpu_sample("50, 60, [N/A], 100") == {"gpu_util": 50.0, "gpu_temp": 60.0, "gpu_memory_used_mb": 100.0}
