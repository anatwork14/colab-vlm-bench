"""ITER3 Round A H2O confounder control (cpu_pytorch_reference_not_vllm): arm h2o_shared_base.

h2o_shared_base uses the SAME shared mean-over-full-attention-layers selection as STARLIGHT, with NO internal
protection (no DA, grounding or structural signal). It separates "the shared selection" from "the protection" when
STARLIGHT is compared with per-layer H2O. The seven original arm IDs are unchanged.

Scope: tiny scaffold, no checkpoint, no speed or quality claim. Comparisons use the same prompt, budget formula,
layers, generation policy and tiny config for every arm.
"""
import hashlib

import numpy as np

from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.h2o import H2OCacheTracker, H2OConfig
from vlmbench.infrastructure.real_engine import STARLIGHT_ARM_SIGNALS, RealSevenArmEngine

from test_starlight_oracle import _cfg, _WordTok

ACCEPT_DECLARED = EvidencePolicy(accept_declared=True)  # S1: text-only keyword fixture; explicit opt-in (reported)
SHA = hashlib.sha256(b"no-image").hexdigest()
CAR = 8
ORIGINAL_ELEVEN = frozenset((
    "base", "h2o_base", "da_base", "ecrd_base", "mtp", "h2o_mtp", "da_mtp",
    "starlight_s1", "starlight_s1_no_da", "starlight_s1_no_grounding", "starlight_s1_mtp",
))
PROMPT = list(range(100, 140))  # 40 text tokens
for _pos in (5, 20, 33):
    PROMPT[_pos] = CAR  # 'car' three times: protected under the default breadth cap (3/40)
N_NEW = 4


def _pool(rid="h2s"):
    pool = EvidencePool(rid)
    pool.add_evidence(EvidenceItem(evidence_id="ev_h2", modality="text_span", content="car",
                                   source_sha256=SHA, verified=True), keywords=["car"])
    return pool


def _run(arm, *, pool=None, ratio=0.2, min_budget=16, rid="h2s"):
    engine = RealSevenArmEngine(method_arm=arm, device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=min_budget, h2o_budget_ratio=ratio, tokenizer=_WordTok(),
                                evidence_policy=ACCEPT_DECLARED)
    try:
        return engine.execute_request(request_id=rid, prompt_token_ids=PROMPT, image_present=False,
                                      max_new_tokens=N_NEW, evidence_pool=pool)
    finally:
        engine.close()


# --- registration ---------------------------------------------------------------------------------

def test_h2o_shared_base_is_registered_as_a_twelfth_arm_and_the_original_eleven_are_unchanged():
    assert RealSevenArmEngine.SUPPORTED_ARMS == ORIGINAL_ELEVEN | {"h2o_shared_base"}
    assert len(RealSevenArmEngine.SUPPORTED_ARMS) == 12
    assert STARLIGHT_ARM_SIGNALS["h2o_shared_base"] == (), "no controller signal: no internal protection"


# --- no internal protection ----------------------------------------------------------------------

def test_h2o_shared_base_protects_nothing_even_when_matching_evidence_is_present():
    res = _run("h2o_shared_base", pool=_pool())
    st = res.telemetry["starlight"]
    assert st["signals_enabled"] == []
    assert st["protected_count_by_source"] == {"da": 0, "grounding": 0, "structural": 0, "union": 0}
    assert st["grounding_protected_by_evidence"] == {}
    assert st["invariant_protected_evicted"] == 0
    assert "SHARED" in st["candidate_basis"]


def test_evidence_does_not_change_the_h2o_shared_base_selection_but_does_change_starlight():
    # Non-vacuous: STARLIGHT's grounding protects the 'car' positions, so its selection moves with the evidence.
    star_with = _run("starlight_s1", pool=_pool("h2s_a"), rid="h2s_a")
    star_without = _run("starlight_s1", pool=None, rid="h2s_a")
    assert star_with.telemetry["starlight"]["protected_count_by_source"]["grounding"] > 0
    assert star_with.telemetry["starlight"]["evicted_total"] != star_without.telemetry["starlight"]["evicted_total"]
    shared_with = _run("h2o_shared_base", pool=_pool("h2s_b"), rid="h2s_b")
    shared_without = _run("h2o_shared_base", pool=None, rid="h2s_b")
    assert shared_with.telemetry["starlight"]["evicted_total"] == shared_without.telemetry["starlight"]["evicted_total"]
    assert shared_with.tokens_generated == shared_without.tokens_generated


# --- the selection equals STARLIGHT's shared selection when STARLIGHT protects nothing -----------

def _scored_tracker(scores):
    t = H2OCacheTracker(config=H2OConfig(), num_layers=8, num_kv_heads=2)
    for layer in (3, 7):
        t.update_attention_scores(layer, np.tile(np.asarray(scores, dtype=np.float32), (4, 1, 1)))
    return t


def _plans(engine, enabled, grounding):
    scores = [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5, 0.05, 0.95, 0.15]
    length, budget = len(scores), 6
    tracker = _scored_tracker(scores)
    k_lens = {3: length, 7: length}
    orig = {3: np.arange(length - 1), 7: np.arange(length - 1)}
    return engine._starlight_build_step_plans(
        tracker, k_lens, orig, length - 1, budget, [], dict(grounding), tuple(enabled),
    )


def test_shared_selection_matches_starlight_with_zero_protection_at_every_layer():
    shared_engine = RealSevenArmEngine(method_arm="h2o_shared_base", device="cpu", config=_cfg())
    star_engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg())
    shared = _plans(shared_engine, enabled=(), grounding={})
    star_zero = _plans(star_engine, enabled=("grounding",), grounding={})  # protects nothing
    assert len(shared) == len(star_zero) > 0
    for a, b in zip(shared, star_zero):
        assert a.layer == b.layer
        assert a.retained == b.retained and a.evicted == b.evicted
        assert [d.decision for d in a.decisions] == [d.decision for d in b.decisions]


# --- comparability across the five arms the lead named -------------------------------------------

FIVE_ARMS = ("h2o_base", "h2o_shared_base", "starlight_s1_no_da", "starlight_s1_no_grounding", "starlight_s1")


def test_the_five_comparison_arms_share_prompt_budget_layers_and_generation_policy():
    runs = {arm: _run(arm, pool=_pool(f"cmp_{arm}"), rid=f"cmp_{arm}") for arm in FIVE_ARMS}
    lengths = {arm: len(r.tokens_generated) for arm, r in runs.items()}
    assert len(set(lengths.values())) == 1, lengths
    layers = {tuple(r.telemetry["full_attention_layers"]) for r in runs.values()}
    assert len(layers) == 1
    generation = {str(r.telemetry["generation"]) for r in runs.values()}
    assert len(generation) == 1
    model_ids = {r.telemetry["model_id"] for r in runs.values()}
    assert len(model_ids) == 1
    expected_budget = max(16, int(len(PROMPT) * 0.2))
    for arm in FIVE_ARMS:
        if arm == "h2o_base":
            continue  # per-layer H2O reports its own accounting; checked below
        assert runs[arm].telemetry["starlight"]["budget_total"] == expected_budget, arm
    assert runs["h2o_base"].telemetry["h2o_logical_selection_accounting"]["budget_basis"] == \
        "unprotected_prefill_tokens_fixed_per_layer"


def test_every_comparison_arm_reports_its_actual_retained_counts():
    for arm in FIVE_ARMS:
        res = _run(arm, pool=_pool(f"ret_{arm}"), rid=f"ret_{arm}")
        ret = res.telemetry["retained_entries_per_layer"]
        assert ret["mean"] is not None and ret["mean"] > 0, arm
        assert ret["samples"] > 0, arm


def test_under_a_protection_exceeding_budget_h2o_shared_base_is_feasible_and_starlight_is_not():
    # Budget 2 with three protected 'car' positions: STARLIGHT's protection exceeds the budget (INFEASIBLE, the
    # shared step keeps every entry). h2o_shared_base has no protection, so it keeps within the budget and is a
    # successful fixed-budget run. Both report what they actually retained.
    star = _run("starlight_s1", pool=_pool("inf_s"), ratio=0.0, min_budget=2, rid="inf_s")
    shared = _run("h2o_shared_base", pool=_pool("inf_h"), ratio=0.0, min_budget=2, rid="inf_h")
    assert star.telemetry["starlight"]["infeasible_steps"] > 0
    assert star.telemetry["starlight"]["fixed_budget_run"] is False
    assert shared.telemetry["starlight"]["infeasible_steps"] == 0
    assert shared.telemetry["starlight"]["fixed_budget_run"] is True
    assert shared.telemetry["starlight"]["max_retained_per_layer_step"] <= shared.telemetry["starlight"]["budget_total"]
