"""ITER2 D4 S-2 (lead ruling): retained entries per layer are reported for EVERY arm, and a
matched-retention mode calibrates H2O's budget_ratio to a STARLIGHT arm's mean retained count.

Retained count is measured per DECODE step over full-attention layers (prefill cache is unbounded
and excluded). cpu_pytorch_reference_not_vllm; tiny synthetic config, no speed or quality claim.
"""
import pytest
import torch

from vlmbench.infrastructure.model_config import Qwen35ModelConfig
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

PROMPT = list(range(100, 140))  # 40 tokens, no vision tokens
N_NEW = 8


def _cfg():
    return Qwen35ModelConfig(
        model_id="Qwen/Qwen3.5-MatchedRetentionTest", model_type="qwen3_5",
        architectures=("Qwen3_5ForConditionalGeneration",),
        vocab_size=512, hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, max_position_embeddings=1024,
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
        pad_token_id=0, eos_token_id=1, full_attention_layers=(3, 7),
        recurrent_layers=(0, 1, 2, 4, 5, 6), mtp_num_speculative_tokens=2,
    )


def _run(arm, **kw):
    torch.manual_seed(1234)
    engine = RealSevenArmEngine(method_arm=arm, device="cpu", config=_cfg(), stop_at_eos=False, **kw)
    try:
        return engine.execute_request(f"mr_{arm}", prompt_token_ids=PROMPT, image_present=False,
                                      max_new_tokens=N_NEW)
    finally:
        engine.close()


@pytest.mark.parametrize("arm", ["base", "h2o_base", "starlight_s1"])
def test_every_arm_reports_retained_entries_per_layer_per_decode_step(arm):
    res = _run(arm)
    rec = res.telemetry["retained_entries_per_layer"]
    assert rec["unit"] == "per_decode_step_full_attention_layer"
    assert rec["prefill_excluded"] is True
    # One sample per full-attention layer per decode forward (N_NEW - 1 decode forwards).
    assert rec["samples"] == len(_cfg().full_attention_layers) * (N_NEW - 1)
    assert rec["mean"] is not None and rec["mean"] > 0


def test_calibration_finds_the_budget_whose_measured_retention_is_closest_to_target():
    from vlmbench.infrastructure.h2o import calibrate_h2o_ratio_to_retained

    # Synthetic monotone measure with a constant out-of-budget offset of 4 entries.
    def measure(ratio):
        return float(max(1, int(40 * ratio)) + 4)

    cal = calibrate_h2o_ratio_to_retained(measure, 29, prompt_len=40, min_budget=1)
    assert cal["budget"] == 25
    assert cal["measured_mean_retained"] == 29.0
    assert cal["budget_ratio"] == (25 + 0.5) / 40
    with pytest.raises(ValueError):
        calibrate_h2o_ratio_to_retained(measure, 29, prompt_len=0, min_budget=1)


def test_h2o_calibrated_to_starlight_mean_retention_matches_it_within_one_entry():
    from vlmbench.infrastructure.h2o import calibrate_h2o_ratio_to_retained

    star_mean = _run("starlight_s1").telemetry["retained_entries_per_layer"]["mean"]

    def measure(ratio):
        # min_budget=1 so the calibrated budget is not floored above the target.
        return _run("h2o_base", h2o_budget_ratio=ratio, h2o_min_budget=1).telemetry[
            "retained_entries_per_layer"]["mean"]

    cal = calibrate_h2o_ratio_to_retained(measure, star_mean, len(PROMPT), min_budget=1)
    assert abs(cal["measured_mean_retained"] - star_mean) <= 1.0, (star_mean, cal)
