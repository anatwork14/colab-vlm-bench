"""ITER3 closure repair (wF:p7), light tests for the wF:p9 Round-B review findings.

  F-9  the partially covered cell count reaches telemetry (not discarded);
  F-11 a control arm is not labelled as an ablation; an ablation is labelled as one;
  F-1  the comparison_validity field: a STARLIGHT run with no eviction and protection >= budget is labelled
       STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE (DA FOCUS is not changed; owner decision).

cpu_pytorch_reference_not_vllm. No speed or quality claim.
"""
import hashlib

import pytest

from vlmbench.infrastructure import visual_mapping as vm
from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

from test_starlight_oracle import PROMPT, _cfg, _WordTok

RID = "req-f9"
IMG = b"fixture image bytes for F-9"
SHA = hashlib.sha256(IMG).hexdigest()
SPAN = (3, 3 + 64)
GEOM = vm.build_geometry(256, 256, [1, 16, 16], patch_size=16, merge_size=2,
                         min_pixels=65536, max_pixels=16777216)[0]


def _report(box, sha=SHA):
    pool = EvidencePool(RID)
    pool.add_evidence(EvidenceItem(evidence_id="v0", modality="image_region", content="r", source_sha256=sha,
                                   verified=True, bounding_box=box))
    return vm.visual_evidence_report(pool, request_id=RID, image_sha256=SHA, image_present=True,
                                     policy=EvidencePolicy(), geometry=GEOM, geometry_reason=None,
                                     span=SPAN, span_reason=None)


# --- F-9: partial cells reach telemetry -----------------------------------------------------------

def test_f9_report_items_carry_the_full_and_partial_cell_counts():
    # (0.01, 0.01, 0.5, 0.5) on the 256-px geometry: 16 cells overlap, 9 are fully inside, 7 are partial.
    item = _report((0.01, 0.01, 0.5, 0.5))["items"]["v0"]
    assert item["protects"] is True
    assert item["cells"] == {"full": 9, "partial": 7}


def test_f9_a_refused_item_reports_no_cell_counts():
    item = _report((0.01, 0.01, 0.5, 0.5), sha="0" * 64)["items"]["v0"]
    assert item["protects"] is False
    assert item["cells"] is None


# --- F-11: control arm vs ablation labels ---------------------------------------------------------

def test_f11_control_arm_without_a_grounding_signal_is_not_labelled_as_an_ablation():
    eng = RealSevenArmEngine(method_arm="h2o_shared_base", device="cpu", config=_cfg(), stop_at_eos=False,
                             h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = eng.execute_request(request_id="f11_ctrl", prompt_token_ids=PROMPT, image_present=False,
                                  max_new_tokens=2)
    finally:
        eng.close()
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "NOT_APPLICABLE_CONTROL_ARM_NO_GROUNDING_SIGNAL"
    assert block["grounding_signal_status"] == "NOT_APPLICABLE_CONTROL_ARM_NO_GROUNDING_SIGNAL"
    assert "ABLATION" not in block["visual_evidence_protection"]


def test_f11_an_ablation_without_grounding_is_labelled_as_disabled_by_ablation():
    eng = RealSevenArmEngine(method_arm="starlight_s1_no_grounding", device="cpu", config=_cfg(),
                             stop_at_eos=False, h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = eng.execute_request(request_id="f11_abl", prompt_token_ids=PROMPT, image_present=False,
                                  max_new_tokens=2)
    finally:
        eng.close()
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "DISABLED_BY_ABLATION"
    assert block["grounding_signal_status"] == "DISABLED_BY_ABLATION"


# --- F-1: comparison_validity (surfaced, not a fix of the DA scope) -------------------------------

@pytest.mark.parametrize("kwargs, status, comparable", [
    (dict(starlight_active=False, evicted_total=0, protected_at_prefill=0, budget_total=192, fixed_budget_run=True),
     "NOT_APPLICABLE_NO_STARLIGHT_CONTROLLER", False),
    # The fixture case: nothing evicted, and protection (321) is at or above the budget (192).
    (dict(starlight_active=True, evicted_total=0, protected_at_prefill=321, budget_total=192, fixed_budget_run=False),
     "STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE", False),
    # Protection exactly at the budget is also full-cache infeasible (the planner keeps every entry).
    (dict(starlight_active=True, evicted_total=0, protected_at_prefill=192, budget_total=192, fixed_budget_run=False),
     "STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE", False),
    # Nothing evicted but protection is below the budget: not the full-cache case.
    (dict(starlight_active=True, evicted_total=0, protected_at_prefill=10, budget_total=192, fixed_budget_run=True),
     "FIXED_BUDGET_RUN_MET", True),
    (dict(starlight_active=True, evicted_total=40, protected_at_prefill=50, budget_total=192, fixed_budget_run=False),
     "NOT_A_SUCCESSFUL_FIXED_BUDGET_RUN", False),
    (dict(starlight_active=True, evicted_total=40, protected_at_prefill=50, budget_total=192, fixed_budget_run=True),
     "FIXED_BUDGET_RUN_MET", True),
])
def test_f1_comparison_validity_labels_full_cache_infeasible_runs(kwargs, status, comparable):
    from vlmbench.infrastructure.starlight import comparison_validity

    verdict = comparison_validity(**kwargs)
    assert verdict["status"] == status
    assert verdict["comparable_to_matched_budget"] is comparable
    assert verdict["reason"]
