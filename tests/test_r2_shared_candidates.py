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
