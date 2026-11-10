"""ITER3 Round A S3 (cpu_pytorch_reference_not_vllm): protected KV is never silently evicted, infeasibility is
explicit, and an infeasible configuration is flagged as NOT a successful fixed-budget run. Memory reports must not
present an infeasible run as budget-bounded (no hidden full-cache fallback).

Provisional policy A-2 (engineering hypothesis, not owner ratification): no silent eviction of protected KV;
infeasibility is reported; an infeasible configuration is not a successful fixed-budget run. Tiny scaffold only,
no checkpoint, no speed or quality claim.
"""
import hashlib

from vlmbench.infrastructure import starlight as starlight_mod
from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import (
    DECISION_EVICT,
    STATUS_INFEASIBLE,
    STATUS_OK,
    plan_starlight_layer,
    summarize_plans,
)

from test_starlight_oracle import _cfg, _WordTok

# Text-only fixtures with caller verified=True do not protect by default (S1). This S3 test must have
# grounding protection ACTIVE to create protected > budget, so it opts in explicitly (accept_declared=True).
ACCEPT_DECLARED = EvidencePolicy(accept_declared=True)
SHA = hashlib.sha256(b"no-image").hexdigest()
CAR = 8
PROMPT = list(range(100, 140))  # 40 tokens
for _pos in (5, 20, 33):  # 'car' three times: 3/40 = 0.075, under the per-keyword breadth cap
    PROMPT[_pos] = CAR


def _plan(length, budget, protected_positions, candidates):
    protected = {i: ("ev_p",) for i in protected_positions}
    return plan_starlight_layer(
        layer=3, length=length, budget=budget, h2o_candidates=list(candidates),
        signals={"da": {}, "grounding": protected, "structural": {}}, enabled_signals=("grounding",),
    )


# --- planner level: protected > budget ---------------------------------------------------------

def test_protected_above_budget_is_never_evicted_and_the_step_is_infeasible():
    protected = [0, 1, 2, 3, 4, 5]  # 6 protected, budget 4
    plan = _plan(length=10, budget=4, protected_positions=protected, candidates=[6, 7, 8, 9])
    assert plan.status == STATUS_INFEASIBLE
    assert len(plan.evicted) == 0, "an INFEASIBLE step evicts nothing, protected or unprotected"
    assert set(protected) <= set(plan.retained)
    assert not any(d.decision == DECISION_EVICT and d.internally_protected for d in plan.decisions)


def test_feasible_control_protection_within_budget_still_evicts_unprotected_candidates():
    # Positive control, so the test above is not vacuous: 2 protected within a budget of 4 is feasible.
    plan = _plan(length=10, budget=4, protected_positions=[0, 1], candidates=[2, 3, 4, 5, 6, 7, 8, 9])
    assert plan.status == STATUS_OK
    assert len(plan.evicted) > 0
    assert {0, 1} <= set(plan.retained)


# --- the run-level verdict (pure) ---------------------------------------------------------------

def test_fixed_budget_verdict_flags_an_infeasible_step_as_not_a_successful_fixed_budget_run():
    plan = _plan(length=10, budget=4, protected_positions=[0, 1, 2, 3, 4, 5], candidates=[6, 7, 8, 9])
    summary = summarize_plans([plan])
    verdict = starlight_mod.fixed_budget_verdict(summary, budget_total=4, max_retained_per_layer=len(plan.retained))
    assert verdict["fixed_budget_run"] is False
    assert verdict["status"] == starlight_mod.FIXED_BUDGET_NOT_MET
    assert verdict["infeasible_layer_steps"] == 1
    assert verdict["max_retained_over_budget"] == len(plan.retained) - 4 > 0


def test_fixed_budget_verdict_is_met_only_when_every_step_is_feasible_and_within_budget():
    plan = _plan(length=10, budget=4, protected_positions=[0, 1], candidates=[2, 3, 4, 5, 6, 7, 8, 9])
    assert plan.status == STATUS_OK
    summary = summarize_plans([plan])
    verdict = starlight_mod.fixed_budget_verdict(summary, budget_total=4, max_retained_per_layer=len(plan.retained))
    assert verdict["fixed_budget_run"] is True
    assert verdict["status"] == starlight_mod.FIXED_BUDGET_MET
    assert verdict["max_retained_over_budget"] == 0


def test_a_kv_budget_violation_alone_blocks_the_verdict_even_when_no_status_is_infeasible():
    summary = {"status_counts": {STATUS_OK: 3}, "kv_budget_violated_layers": 1}
    verdict = starlight_mod.fixed_budget_verdict(summary, budget_total=4, max_retained_per_layer=4)
    assert verdict["fixed_budget_run"] is False


def test_a_retention_above_budget_blocks_the_verdict_even_when_no_status_is_infeasible():
    summary = {"status_counts": {STATUS_OK: 3}, "kv_budget_violated_layers": 0}
    verdict = starlight_mod.fixed_budget_verdict(summary, budget_total=4, max_retained_per_layer=6)
    assert verdict["fixed_budget_run"] is False
    assert verdict["max_retained_over_budget"] == 2


# --- engine level: protected > budget on a real decode loop -------------------------------------

def _pool(rid):
    pool = EvidencePool(rid)
    pool.add_evidence(EvidenceItem(evidence_id="ev_c", modality="text_span", content="car",
                                   source_sha256=SHA, verified=True), keywords=["car"])
    return pool


def _run_over_budget(rid="s3_over"):
    # min_budget 2 with ratio 0.0 gives budget 2; three protected 'car' positions exceed it.
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=2, h2o_budget_ratio=0.0, tokenizer=_WordTok(),
                                starlight_record_decisions=True, evidence_policy=ACCEPT_DECLARED)
    try:
        return engine.execute_request(request_id=rid, prompt_token_ids=PROMPT, image_present=False,
                                      max_new_tokens=4, evidence_pool=_pool(rid))
    finally:
        engine.close()


def test_engine_run_with_protection_above_budget_is_flagged_not_a_fixed_budget_run():
    res = _run_over_budget()
    st = res.telemetry["starlight"]
    assert st["protected_count_by_source"]["grounding"] > 0, "non-vacuous: protection is active"
    assert st["infeasible_steps"] > 0
    assert st["fixed_budget_run"] is False
    assert st["fixed_budget_status"] == starlight_mod.FIXED_BUDGET_NOT_MET


def test_engine_run_never_evicts_protected_entries_even_when_protection_exceeds_budget():
    res = _run_over_budget("s3_never")
    st = res.telemetry["starlight"]
    recs = st["decision_records"]
    assert any(r["n_protected_outside_budget"] > 0 for r in recs), "non-vacuous: protected > budget occurred"
    for rec in recs:
        for e in rec["entries"]:
            if e["grounding_protected"] or e["da_protected"] or e["structural_protected"]:
                assert e["decision"] != DECISION_EVICT, e
    assert st["invariant_protected_evicted"] == 0


def test_memory_report_does_not_present_an_infeasible_run_as_budget_bounded():
    res = _run_over_budget("s3_mem")
    st = res.telemetry["starlight"]
    acct = res.telemetry["kv_memory_accounting"]
    assert acct["fixed_budget_status"] == starlight_mod.FIXED_BUDGET_NOT_MET
    assert acct["kv_retention_status"] == starlight_mod.RETENTION_AT_INFEASIBLE_STEPS
    assert acct["max_retained_over_budget_per_layer_step"] == st["max_retained_per_layer_step"] - st["budget_total"]
    assert acct["max_retained_over_budget_per_layer_step"] > 0


def test_feasible_starlight_run_is_reported_as_a_fixed_budget_run():
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="s3_ok", prompt_token_ids=list(range(100, 130)),
                                     image_present=False, max_new_tokens=4)
    finally:
        engine.close()
    st = res.telemetry["starlight"]
    acct = res.telemetry["kv_memory_accounting"]
    assert st["infeasible_steps"] == 0
    assert st["fixed_budget_run"] is True
    assert acct["fixed_budget_status"] == starlight_mod.FIXED_BUDGET_MET
    assert acct["kv_retention_status"] == starlight_mod.RETENTION_BUDGET_BOUNDED
    assert acct["max_retained_over_budget_per_layer_step"] == 0


def test_non_starlight_arm_reports_no_fixed_budget_status_rather_than_a_claim():
    engine = RealSevenArmEngine(method_arm="base", device="cpu", config=_cfg(), stop_at_eos=False)
    try:
        res = engine.execute_request(request_id="s3_base", prompt_token_ids=list(range(100, 110)),
                                     image_present=False, max_new_tokens=2)
    finally:
        engine.close()
    assert res.telemetry["kv_memory_accounting"]["fixed_budget_status"] == starlight_mod.FIXED_BUDGET_NOT_APPLICABLE
