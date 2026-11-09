"""ITER2 Stage 2(c): H2O per-KV-head score selection (upstream fidelity A4-02), opt-in.

Default (per_kv_head_selection=False) keeps the layer-mean selection unchanged. With the flag on,
each KV head keeps its own heavy hitters with the same budget; the recent window is shared. The
flag is refused by the shared compute_eviction_plan and by the STARLIGHT keep_mask_without_protection
(no silent mean-based result) and is NOT wired
into the engine: per-head KV compaction needs a per-head gather in the hooked attention, which is
not implemented. Tracker-level only; cpu_pytorch_reference_not_vllm.
"""
import numpy as np
import pytest

from vlmbench.infrastructure.h2o import H2OCacheTracker, H2OConfig


def _tracker(per_head: bool):
    cfg = H2OConfig(budget_ratio=0.5, recent_ratio=0.2, min_budget=1, protect_system_tokens=0,
                    protect_visual_tokens=False, per_kv_head_selection=per_head)
    t = H2OCacheTracker(config=cfg, num_layers=1, num_kv_heads=2)
    # head 0 attends to positions 2..6, head 1 to positions 7..11 (query-sum weights).
    w = np.zeros((2, 1, 12), dtype=np.float32)
    w[0, 0, 2:7] = [5, 4, 3, 2, 1]
    w[1, 0, 7:12] = [5, 4, 3, 2, 1]
    t.update_attention_scores(0, w)
    t.layers[0].budget = 5
    return t


def test_default_flag_is_off_and_keeps_the_layer_mean_selection():
    assert H2OConfig().per_kv_head_selection is False
    t = _tracker(per_head=False)
    retained, metrics = t.compute_eviction_plan(0, current_length=12)
    assert retained.shape == (12,)
    assert int(retained.sum()) == 5
    assert metrics["retained_tokens"] == 5


def test_per_head_plan_gives_each_head_its_own_heavy_hitters_with_equal_budget():
    t = _tracker(per_head=True)
    retained, metrics = t.compute_eviction_plan_per_head(0, current_length=12)
    assert retained.shape == (2, 12)
    # Same budget per head: recent window (1) + heavy hitters (4).
    assert retained.sum(axis=1).tolist() == [5, 5]
    # The shared recent window is the newest unprotected token in every head.
    assert retained[:, 11].all()
    # Each head keeps its OWN heavy hitters, so the two heads' retained sets differ.
    assert not np.array_equal(retained[0], retained[1])
    assert set(np.flatnonzero(retained[0])) - {11} <= set(range(2, 7)) | {0, 1}
    assert set(np.flatnonzero(retained[1])) - {11} <= set(range(7, 12)) | {0, 1}


def test_shared_eviction_plan_refuses_the_flag_instead_of_silently_using_the_mean():
    t = _tracker(per_head=True)
    with pytest.raises(NotImplementedError):
        t.compute_eviction_plan(0, current_length=12)


def test_per_head_plan_does_not_touch_the_layer_mask_or_budget():
    t = _tracker(per_head=True)
    before_mask = t.layers[0].evicted_mask.copy()
    t.compute_eviction_plan_per_head(0, current_length=12)
    assert np.array_equal(t.layers[0].evicted_mask, before_mask)
    assert t.layers[0].budget == 5


def test_per_head_plan_with_unset_budget_computes_it_without_storing_it_or_padding_scores():
    # ITER2 D10 N-12 (review): the docstring promises no change to the layer's budget, evicted mask
    # or scores. The budget-None path (not covered by the fixture above) must honour that too.
    cfg = H2OConfig(budget_ratio=0.5, recent_ratio=0.2, min_budget=1, protect_system_tokens=0,
                    protect_visual_tokens=False, per_kv_head_selection=True)
    t = H2OCacheTracker(config=cfg, num_layers=1, num_kv_heads=2)
    w = np.zeros((2, 1, 12), dtype=np.float32)
    w[0, 0, 2:7] = [5, 4, 3, 2, 1]
    w[1, 0, 7:12] = [5, 4, 3, 2, 1]
    t.update_attention_scores(0, w)
    assert t.layers[0].budget is None
    mask_before = t.layers[0].evicted_mask.copy()
    scores_shape_before = t.layers[0].scores.shape
    retained, metrics = t.compute_eviction_plan_per_head(0, current_length=14)
    assert metrics["budget"] == 7  # max(min_budget, int(14 * 0.5))
    assert retained.shape == (2, 14)
    assert retained.sum(axis=1).tolist() == [7, 7]
    assert t.layers[0].budget is None
    assert np.array_equal(t.layers[0].evicted_mask, mask_before)
    assert t.layers[0].scores.shape == scores_shape_before


def test_starlight_candidate_path_refuses_the_per_head_flag_instead_of_running_the_mean():
    # ITER2 D10 N-13 (review): keep_mask_without_protection is the STARLIGHT candidate selection and
    # is layer-mean based. With the per-head flag on it must refuse, not silently ignore the flag.
    t = _tracker(per_head=True)
    with pytest.raises(NotImplementedError, match="per_kv_head_selection"):
        t.keep_mask_without_protection(0, length=12, keep_budget=5)
