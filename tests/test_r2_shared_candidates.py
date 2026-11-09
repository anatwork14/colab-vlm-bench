"""ITER3 R2 F-1 (lead ruling): STARLIGHT keeps ONE retained set across all full-attention layers.

Policy-invariant unit test on the tiny scaffold (no checkpoint). The candidate set is built from
layer-aggregated scores, so protection that vetoes a candidate does it on every full-attention layer
alike, and the per-layer retained counts cannot desynchronise (the cause of ITER3 F-1).
cpu_pytorch_reference_not_vllm.
"""
import numpy as np

from vlmbench.infrastructure.h2o import H2OCacheTracker, H2OConfig
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

from test_starlight_oracle import _cfg


def _two_layer_tracker_with_disagreeing_scores(length):
    t = H2OCacheTracker(config=H2OConfig(), num_layers=8, num_kv_heads=2)
    # Layer 3 would evict position 5; layer 7 would keep it (large score).
    s3 = np.array([10, 9, 8, 7, 6, 0, 5, 4, 3, 2, 1, 0], dtype=np.float32)
    s7 = np.array([1, 2, 3, 4, 5, 50, 6, 7, 8, 9, 10, 0], dtype=np.float32)
    for layer, s in ((3, s3), (7, s7)):
        t.layers[layer].scores = np.tile(s[:length], (2, 1))
    return t


def test_F1_shared_candidates_keep_every_full_layer_at_the_same_length_under_protection():
    length, budget = 12, 6
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(),
                                stop_at_eos=False, h2o_min_budget=budget, h2o_budget_ratio=0.5)
    tracker = _two_layer_tracker_with_disagreeing_scores(length)
    orig = {3: np.arange(length - 1), 7: np.arange(length - 1)}
    # Protect position 5: a veto on whichever layer has it as a candidate.
    da_spans = [("focus", 5, 6)]
    plans, next_orig, step_plans = engine._starlight_build_step_plans(
        tracker, {3: length, 7: length}, orig, length - 1, budget, da_spans, {}, ("da", "grounding"))
    retained_3 = step_plans[0].retained
    retained_7 = step_plans[1].retained
    assert step_plans[0].layer == 3 and step_plans[1].layer == 7
    assert len(retained_3) == len(retained_7), (retained_3, retained_7)
    assert retained_3 == retained_7
    assert np.array_equal(next_orig[3], next_orig[7])
    assert 5 in retained_3 and 5 in retained_7


def test_F1_shared_mask_with_one_layer_equals_the_per_layer_rule_and_has_exact_budget():
    # Guard on the refactor: the shared rule reduces to the per-layer H2O rule for a single layer.
    rng = np.random.default_rng(0)
    for length, budget in ((15, 6), (20, 9), (9, 4)):
        t = H2OCacheTracker(config=H2OConfig(), num_layers=8, num_kv_heads=2)
        t.layers[3].scores = rng.random((2, length)).astype(np.float32)
        shared = t.keep_mask_shared([3], length, budget)
        per_layer = t.keep_mask_without_protection(3, length, budget)
        assert np.array_equal(shared, per_layer), (length, budget)
        assert int(shared.sum()) == min(budget, length)


def test_F1_shared_mask_refuses_per_kv_head_selection():
    t = H2OCacheTracker(config=H2OConfig(per_kv_head_selection=True), num_layers=8, num_kv_heads=2)
    try:
        t.keep_mask_shared([3, 7], 12, 6)
    except NotImplementedError:
        return
    raise AssertionError("keep_mask_shared must refuse per_kv_head_selection, as the per-layer method does")


def test_F1_unequal_full_layer_lengths_raise_instead_of_desynchronising():
    length, budget = 12, 6
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(),
                                stop_at_eos=False, h2o_min_budget=budget, h2o_budget_ratio=0.5)
    tracker = _two_layer_tracker_with_disagreeing_scores(length)
    orig = {3: np.arange(length - 1), 7: np.arange(length - 2)}
    try:
        engine._starlight_build_step_plans(
            tracker, {3: length, 7: length - 1}, orig, length - 1, budget, [], {}, ("da", "grounding"))
    except RuntimeError as exc:
        assert "equal cache lengths" in str(exc)
        return
    raise AssertionError("unequal full-attention cache lengths must raise, not produce a plan")
