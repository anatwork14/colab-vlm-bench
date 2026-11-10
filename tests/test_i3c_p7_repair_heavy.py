"""ITER3 closure repair (wF:p7), HEAVY (official Qwen3.5-0.8B checkpoint, CPU). Run under heavy.lock.

  F-1  image-prompt starlight_s1 (no evidence): comparison_validity = STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE
  F-10 the serving mtp text path carries the speculative_route label
  F-6  INDEPENDENT retention check: the expected per-layer retained count is read from the planner's (or the
       speculative policy's) returned slot map, not from the telemetry expression. An inflated telemetry value
       then fails here whenever the protected positions are not really retained.

cpu_pytorch_reference_not_vllm. No speed or quality claim.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "000000039769.png"
pytestmark = pytest.mark.skipif(
    not ((CKPT / "config.json").is_file() and FIXTURE.is_file()),
    reason=f"official checkpoint {CKPT} or fixture {FIXTURE} not present",
)
QUESTION = "What is shown in this picture?"
TEXT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"


@pytest.fixture(scope="module")
def image_bytes():
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def sha(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def _image_pool(request_id, sha256, box=(0.0, 0.0, 0.5, 0.5)):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id="ev_region", modality="image_region", content="top-left quadrant",
                                   source_sha256=sha256, verified=True, bounding_box=box))
    return pool


# --- F-1 ----------------------------------------------------------------------------------------------

def test_f1_full_cache_infeasible_image_run_is_labelled_not_comparable(image_bytes):
    e = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False,
                           h2o_min_budget=8, h2o_budget_ratio=0.6, evidence_policy=EvidencePolicy())
    try:
        res = e.execute_request(request_id="f1_full", image_present=True, image_bytes=image_bytes,
                                question=QUESTION, max_new_tokens=3, evidence_pool=None)
    finally:
        e.close()
    block = res.telemetry["starlight"]
    # The fixture run: nothing evicted, protection (DA FOCUS over the visual segment) above the budget.
    assert block["evicted_total"] == 0 and block["infeasible_steps"] > 0
    verdict = res.telemetry["comparison_validity"]
    assert verdict["status"] == "STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE", verdict
    assert verdict["comparable_to_matched_budget"] is False
    assert verdict["protected_at_prefill"] >= verdict["budget_total"]


# --- F-10 ---------------------------------------------------------------------------------------------

def test_f10_the_serving_mtp_text_path_carries_the_speculative_route_label():
    e = RealSevenArmEngine(method_arm="mtp", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False)
    try:
        ids = list(e.tokenizer.encode(TEXT))
        res = e.execute_request(request_id="f10_mtp", prompt_token_ids=ids, image_present=False,
                                max_new_tokens=4, evidence_pool=None)
    finally:
        e.close()
    assert res.telemetry["speculation_executed"] is True
    assert res.telemetry["speculative_route"]["status"] == "MTP_VERIFIER_TEXT"


# --- F-6: independent retention checks ---------------------------------------------------------------

@pytest.fixture(scope="module")
def no_da_engine():
    e = RealSevenArmEngine(method_arm="starlight_s1_no_da", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False,
                           h2o_min_budget=8, h2o_budget_ratio=0.6, evidence_policy=EvidencePolicy())
    yield e
    e.close()


def test_f6_visual_retention_matches_the_planner_slot_map_independently(no_da_engine, image_bytes, sha):
    # The planner returns (plans, next_orig, steps). next_orig[l] lists the ORIGINAL position of every physical slot
    # kept on layer l after the last decode step. Read from that, not from the telemetry expression.
    captured = []
    real_planner = no_da_engine._starlight_build_step_plans

    def spy(*args, **kwargs):
        out = real_planner(*args, **kwargs)
        captured.append({l: np.asarray(out[1][l]).tolist() for l in out[1]})
        return out

    no_da_engine._starlight_build_step_plans = spy
    try:
        res = no_da_engine.execute_request(request_id="f6_visual", image_present=True, image_bytes=image_bytes,
                                           question=QUESTION, max_new_tokens=3,
                                           evidence_pool=_image_pool("f6_visual", sha))
    finally:
        del no_da_engine._starlight_build_step_plans
    block = res.telemetry["starlight"]
    assert block["evicted_total"] > 0, "precondition: eviction must happen"
    assert captured, "the planner must have been called"
    final = captured[-1]
    protected = set(block["visual_protected_positions"])
    assert protected
    independent = {str(l): len(protected & set(final[l])) for l in final}
    assert block["visual_protected_retained_final_by_layer"] == independent
    assert all(count == len(protected) for count in independent.values()), independent


def test_f6_speculative_grounding_retention_matches_the_policy_slot_map_independently():
    from vlmbench.infrastructure.starlight import grounding_protection_with_report
    e = RealSevenArmEngine(method_arm="starlight_s1_mtp", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False,
                           h2o_min_budget=40, h2o_budget_ratio=0.5,
                           evidence_policy=EvidencePolicy(accept_declared=True))
    try:
        ids = list(e.tokenizer.encode(TEXT))
        pool = EvidencePool("f6_spec")
        pool.add_evidence(EvidenceItem(evidence_id="ev_paris", modality="text_span", content="Paris",
                                       source_sha256=hashlib.sha256(TEXT.encode()).hexdigest(), verified=True),
                          keywords=["Paris"])
        # The expected protected set is recomputed from the keyword policy, not read from the telemetry.
        protected, _ = grounding_protection_with_report(pool, ids, e.tokenizer.encode, "f6_spec", None,
                                                        e.evidence_policy, image_present=False,
                                                        cap_policy=e.grounding_cap_policy)
        captured = []
        real_apply = e._pol_apply

        def spy(st, plans, next_orig):
            real_apply(st, plans, next_orig)
            captured.append({l: np.asarray(next_orig[l]).tolist() for l in st.full_layers})

        e._pol_apply = spy
        try:
            res = e.execute_request(request_id="f6_spec", prompt_token_ids=ids, image_present=False,
                                    max_new_tokens=24, evidence_pool=pool)
        finally:
            del e._pol_apply
    finally:
        e.close()
    spec = res.telemetry["speculation"]
    assert spec["grounding_protected_positions"] == len(protected) > 0
    assert captured, "the policy driver must have applied at least one step"
    final = captured[-1]
    independent = {str(l): len(set(protected) & set(final[l])) for l in final}
    assert spec["grounding_protected_retained_final_by_layer"] == independent
