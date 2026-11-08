from copy import deepcopy
from pathlib import Path

import pytest

from vlmbench.config import (ConfigError, cell_id, cell_seed, load_config,
                             plan_summary, planned_cells, request_count, validate)
from vlmbench.engine import server_argv

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "a100_campaign.json"


def test_full_plan_and_matched_pairs():
    c = load_config(CONFIG)
    plan = plan_summary(c)
    assert (plan["model_count"], plan["mode_count"], plan["dataset_count"]) == (5, 2, 2)
    assert plan["cell_count"] == 2520
    assert plan["measured_requests"] == 114240
    assert plan["warmup_requests"] == 26460
    assert sum(1 for _ in planned_cells(c)) == 2520


def test_paired_cell_seed_not_model_specific():
    c = load_config(CONFIG)
    cells = list(planned_cells(c))
    def c_match(model, mode):
        return next(x for x in cells if x["model"] == model and x["mode"] == mode
                    and x["dataset"] == "memlens" and x["input_length"] == 128
                    and x["concurrency"] == 1 and x["repetition"] == 0)
    a = c_match("qwen35_08b", "base")
    b = c_match("qwen35_08b", "mtp")
    d = c_match("gemma4_e4b", "base")
    assert a["seed"] == b["seed"] == d["seed"]
    assert cell_id(a) != cell_id(b)


def test_strict_grid_rejects_shortcuts():
    c = load_config(CONFIG)
    c["performance"]["concurrencies"] = [1, 2, 4, 8, 16]
    with pytest.raises(ConfigError, match="7x6"):
        validate(c)


def test_rejects_non_a100_or_quantized_main():
    c = load_config(CONFIG)
    c["hardware"]["dtype"] = "float16"
    with pytest.raises(ConfigError, match="BF16"):
        validate(c)
    c = load_config(CONFIG)
    c["hardware"]["expected_gpu_substring"] = "L4"
    with pytest.raises(ConfigError, match="A100"):
        validate(c)


def test_gpu_job_refuses_placeholders():
    c = load_config(CONFIG)
    with pytest.raises(ConfigError, match="dry_run"):
        validate(c, execution=True)
    c["dry_run"] = False
    with pytest.raises(ConfigError, match="Pin checkpoint"):
        validate(c, execution=True)


def test_qwen_mtp_native_and_gemma_assistant():
    c = load_config(CONFIG)
    q = c["models"][0]
    g = c["models"][-1]
    q_base = server_argv(c, q, "base", "/tmp/q")
    q_mtp = server_argv(c, q, "mtp", "/tmp/q")
    g_mtp = server_argv(c, g, "mtp", "/tmp/g", "/tmp/assistant")
    assert "--speculative-config" not in q_base
    assert '"method": "mtp"' in q_mtp[-1]
    assert '"method": "mtp"' in g_mtp[-1]
    assert "/tmp/assistant" in g_mtp[-1]
    assert "--no-enable-prefix-caching" in q_mtp
    with pytest.raises(ValueError, match="assistant"):
        server_argv(c, g, "mtp", "/tmp/g")


def test_request_count_contract():
    assert [request_count(x) for x in [1, 2, 4, 8, 16, 32]] == [16, 16, 16, 32, 64, 128]
    with pytest.raises(ConfigError):
        request_count(3)
