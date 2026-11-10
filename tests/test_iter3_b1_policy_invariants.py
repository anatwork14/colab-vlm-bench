"""ITER3 B1: POLICY-INVARIANT tests on the tiny synthetic scaffold (no checkpoint, no logits oracle).

Every test here checks a STATE or DECISION property of the STARLIGHT / H2O policy: which entries are
eligible, evicted, retained, and reported. None of them is an inference-quality claim (no tokens are
compared against a reference model, no logits, no speed). The scaffold has no RoPE, so position-identity
claims that need RoPE are in test_iter3_b1_official_oracle.py instead.

Label: cpu_pytorch_reference_not_vllm.
"""
import hashlib

import pytest

from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import STATUS_INFEASIBLE, STATUS_OK

from test_starlight_oracle import _cfg, _WordTok

SHA = hashlib.sha256(b"no-image").hexdigest()
PROMPT_30 = list(range(100, 120)) + [7, 9, 7, 8, 7] + list(range(120, 130))
PROMPT_40 = list(range(100, 140))


# ITER3 R2 C-4 RE-DERIVED: default keyword "the" -> "car" (a stop-word now protects nothing). Token 8 occurs once
# in PROMPT_30 (1/30, under the cap) and never in PROMPT_40, so PROMPT_40 runs are unchanged.
def _pool(request_id, keywords=("car",), eid="ev_1"):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality="text_span", content=" ".join(keywords),
                                   source_sha256=SHA, verified=True), keywords=list(keywords))
    return pool


def _engine(arm, budget, ratio=0.0, record=True):
    import torch
    torch.manual_seed(1234)
    return RealSevenArmEngine(method_arm=arm, device="cpu", config=_cfg(), stop_at_eos=False,
                              h2o_min_budget=budget, h2o_budget_ratio=ratio, tokenizer=_WordTok(),
                              starlight_record_decisions=record)


def _check_records(recs, budget):
    """Per-record contract (POLICY-INVARIANT). Returns the number of EVICT decisions checked."""
    evict_events = 0
    for rec in recs:
        entries = rec["entries"]
        evicted = {e["position"] for e in entries if e["decision"] == "EVICT"}
        # rec["evicted"] and rec["retained"] are COUNTS; the positions come from the entries.
        assert len(evicted) == rec["evicted"], rec["step"]
        assert rec["retained"] == rec["pre"] - len(evicted), rec["step"]
        protected = {e["position"] for e in entries
                     if e["da_protected"] or e["grounding_protected"] or e["structural_protected"]}
        assert rec["n_protected"] == len(protected), rec["step"]
        for e in entries:
            if e["decision"] == "EVICT":
                evict_events += 1
                # eligible_to_evict = h2o_candidate AND NOT protected (the Stage 1 contract).
                assert e["h2o_candidate"] is True, (rec["step"], e)
                assert e["position"] not in protected, (rec["step"], e)
        assert not (evicted & protected), rec["step"]
        if rec["status"] == STATUS_INFEASIBLE:
            # S-3 ruling: an INFEASIBLE step evicts nothing and says so.
            assert evicted == set(), rec["step"]
        else:
            assert rec["status"] == STATUS_OK, rec["status"]
            assert rec["retained"] <= budget, (rec["step"], rec["retained"], budget)
    return evict_events


def _assert_counters_clean(s):
    assert s["invariant_protected_evicted"] == 0
    assert s["invariant_evicted_not_candidate"] == 0
    assert s["cache_length_mismatch_steps"] == 0


@pytest.mark.parametrize("budget", [1, 2, 3, 5])
def test_pinv_very_small_kv_budgets_never_evict_protected_entries(budget):
    # [POLICY-INVARIANT] Tiny budgets: protected entries (DA sink/focus/question scopes) may exceed the
    # budget. The step must then be INFEASIBLE and evict nothing; it must never evict protected entries.
    result = _engine("starlight_s1", budget).execute_request(
        request_id=f"small_{budget}", prompt_token_ids=PROMPT_30, image_present=False,
        max_new_tokens=6, evidence_pool=_pool(f"small_{budget}"))
    s = result.telemetry["starlight"]
    assert s["budget_total"] == budget
    _assert_counters_clean(s)
    recs = s["decision_records"]
    assert recs
    _check_records(recs, budget)
    infeasible = [r for r in recs if r["status"] == STATUS_INFEASIBLE]
    assert s["infeasible_steps"] == len(infeasible)
    # Reported, not hidden: the summary flags a budget violation exactly when an INFEASIBLE step keeps
    # more entries than the budget (INFEASIBLE evicts nothing, so it keeps the whole cache).
    assert s["kv_budget_violated"] == any(r["retained"] > budget for r in infeasible)
    assert len(result.tokens_generated) == 6


def test_pinv_budget_zero_is_explicit_infeasible_and_evicts_nothing():
    # [POLICY-INVARIANT] Budget 0 with protected sinks: every step is INFEASIBLE, nothing is evicted.
    result = _engine("starlight_s1", 0).execute_request(
        request_id="zero", prompt_token_ids=PROMPT_30, image_present=False, max_new_tokens=3,
        evidence_pool=None)
    s = result.telemetry["starlight"]
    _assert_counters_clean(s)
    recs = s["decision_records"]
    assert recs
    _check_records(recs, budget=0)
    for r in recs:
        if r["n_protected"] > 0:
            # Protected entries cannot fit in a budget of 0: explicit INFEASIBLE, nothing evicted.
            assert r["status"] == STATUS_INFEASIBLE, r["step"]
            assert r["evicted"] == 0, r["step"]
        else:
            # Nothing protected: every H2O candidate is evicted and the feasible step keeps 0 entries.
            assert r["status"] == STATUS_OK, r["step"]
            assert r["retained"] == 0, r["step"]


def test_pinv_long_decode_keeps_budget_counters_and_record_count():
    # [POLICY-INVARIANT] 120 decode steps under real eviction pressure: no protected eviction, every
    # step recorded once per full-attention layer, budget held on every feasible step.
    budget = 16
    result = _engine("starlight_s1", budget, ratio=0.2).execute_request(
        request_id="long", prompt_token_ids=PROMPT_40, image_present=False, max_new_tokens=120,
        evidence_pool=_pool("long"))
    s = result.telemetry["starlight"]
    assert len(result.tokens_generated) == 120
    _assert_counters_clean(s)
    assert s["evicted_total"] > 0, "test would be vacuous: no eviction under pressure"
    recs = s["decision_records"]
    assert len(recs) == 2 * s["steps"], (len(recs), s["steps"])
    _check_records(recs, budget)
    assert s["max_retained_per_layer_step"] <= budget or s["infeasible_steps"] > 0


def test_pinv_interleaved_requests_do_not_leak_policy_or_evidence_state():
    # [POLICY-INVARIANT] Shared engine, requests interleaved A, B, A, B. Each repeat must reproduce its
    # first run exactly (tokens, decision records), and request B (no evidence) must never be protected
    # by A's evidence pool.
    budget = 16
    engine = _engine("starlight_s1", budget, ratio=0.2)
    pool_a = _pool("iter_a", ("car",), eid="ev_a")
    kw_a = dict(prompt_token_ids=PROMPT_40, image_present=False, max_new_tokens=8)
    kw_b = dict(prompt_token_ids=PROMPT_30, image_present=False, max_new_tokens=8, evidence_pool=None)
    a1 = engine.execute_request(request_id="iter_a", evidence_pool=pool_a, **kw_a)
    b1 = engine.execute_request(request_id="iter_b", **kw_b)
    a2 = engine.execute_request(request_id="iter_a", evidence_pool=_pool("iter_a", ("car",), eid="ev_a"), **kw_a)
    b2 = engine.execute_request(request_id="iter_b", **kw_b)
    assert a2.tokens_generated == a1.tokens_generated
    assert b2.tokens_generated == b1.tokens_generated
    assert a2.telemetry["starlight"]["decision_records"] == a1.telemetry["starlight"]["decision_records"]
    assert b2.telemetry["starlight"]["decision_records"] == b1.telemetry["starlight"]["decision_records"]
    assert b1.telemetry["starlight"]["grounding_protected_by_evidence"] == {}
    assert b2.telemetry["starlight"]["grounding_protected_by_evidence"] == {}
    for res in (a1, b1, a2, b2):
        _assert_counters_clean(res.telemetry["starlight"])


@pytest.mark.parametrize("arm", ["h2o_mtp", "da_mtp", "starlight_s1_mtp"])
def test_pinv_combined_arm_on_diagnostic_path_flags_speculation_and_keeps_its_policy(arm):
    # [POLICY-INVARIANT] An image request takes the diagnostic path. Speculation must be reported as NOT
    # executed (explicit label), and the arm's own policy must still be reported, not silently dropped.
    import torch
    torch.manual_seed(1234)
    engine = RealSevenArmEngine(method_arm=arm, device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2)
    result = engine.execute_request(request_id=f"diag_{arm}", prompt_token_ids=PROMPT_40,
                                    image_present=True, max_new_tokens=6)
    t = result.telemetry
    assert t["speculation_executed"] is False
    assert t["arm_label"] == "mtp_serving_arm_diagnostic_no_speculation"
    assert len(result.tokens_generated) == 6
    if arm == "h2o_mtp":
        assert result.physical_kv_bytes_freed and result.physical_kv_bytes_freed > 0
        assert t["h2o_logical_selection_accounting"]["cumulative_evicted_slots"] > 0
    if arm == "da_mtp":
        cost = t["da_cost_accounting"]
        assert cost["kv_residency_preserved_ratio"] == 1.0  # DA never claims a KV saving
        assert cost["physical_kv_read_saving_claimed"] is False
    if arm == "starlight_s1_mtp":
        s = t["starlight"]
        assert s["steps"] > 0 and s["arm_status"] in {"OK", "degraded_to_h2o_plus_da"}
        _assert_counters_clean(s)
