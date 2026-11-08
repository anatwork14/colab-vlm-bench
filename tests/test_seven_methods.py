import json
from pathlib import Path

import pytest

from vlmbench.methods import (METHODS, ARM_REGISTRY, UPSTREAM, MethodUnqualified,
                              blocked_method_receipt, manifest_from_path,
                              native_mode_or_block, plan, validate_method_manifest)
from vlmbench.hallucination import (AnnotationError, score_evidence_annotations,
                                   read_annotations)

BASE = Path(__file__).resolve().parents[1]


def test_all_seven_arms_with_frozen_sources():
    m = manifest_from_path(BASE / "configs/a100_qwen_7methods_pilot.json")
    p = plan(m)
    assert p["method_count"] == 7
    assert p["model_count"] == 2
    assert p["total_serving_cells"] == 168
    assert p["temperature"] == 0
    assert [x["id"] for x in p["methods"]] == list(METHODS)
    assert sum(x["implementation_status"] == "BLOCKED_EXTERNAL_ENGINE_PORT_REQUIRED"
               for x in p["methods"]) == 5
    assert all(v["commit"] == m["source_pins"][k] for k, v in UPSTREAM.items())


def test_full_arm_matrix_42_by_three():
    m = manifest_from_path(BASE / "configs/a100_qwen_7methods_full.json")
    p = plan(m)
    assert p["cells_per_model_method_dataset"] == 126
    assert p["total_serving_cells"] == 3528
    assert m["ccu"] == [1, 2, 4, 8, 16, 32]


@pytest.mark.parametrize("arm", ["h2o_base", "da_base", "ecrd_base", "h2o_mtp", "da_mtp"])
def test_unsupported_arm_cannot_run_as_stock_vllm(arm):
    with pytest.raises(MethodUnqualified, match="BLOCKED_EXTERNAL_ENGINE_PORT_REQUIRED"):
        native_mode_or_block(arm)
    receipt = blocked_method_receipt(arm, "qwen35_08b", "memlens")
    assert receipt["status"] == "BACKEND_UNQUALIFIED"
    assert receipt["latency_ms"] is None and receipt["accuracy"] is None
    assert receipt["not_a_gpu_attempt"] is True
    assert receipt["source_pin"]["commit"]


def test_only_native_routes_and_no_fake_receipt():
    assert native_mode_or_block("base") == "base"
    assert native_mode_or_block("mtp") == "mtp"
    with pytest.raises(ValueError):
        blocked_method_receipt("base", "qwen35_08b", "memlens")
    with pytest.raises(ValueError):
        native_mode_or_block("fake_h2o")


def test_method_campaign_refuses_temperature_and_changed_pins():
    m = manifest_from_path(BASE / "configs/a100_qwen_7methods_pilot.json")
    m["temperature"] = 0.5
    with pytest.raises(ValueError, match="temperature=0"):
        validate_method_manifest(m)
    m = manifest_from_path(BASE / "configs/a100_qwen_7methods_pilot.json")
    m["source_pins"]["h2o"] = "unverified"
    with pytest.raises(ValueError, match="source provenance"):
        validate_method_manifest(m)


def evidence(id, verdicts, answerability="ANSWERABLE", abstained=False):
    return {
        "id": id, "answerability": answerability, "abstained": abstained,
        "judge_provenance": {
            "judge_id": "external_judge",
            "judge_revision": "sha-a",
            "evidence_sha256": "sha-dataset",
            "rubric_version": "evidence-claim-v1"},
        "claims": [
            {"text": f"Fact #{i}", "verdict": v,
             "evidence_ids": (["page_1"] if v in ("SUPPORTED", "CONTRADICTED") else [])}
            for i, v in enumerate(verdicts)
        ],
    }


def test_hallucination_metrics_do_not_equate_wrong_answer_to_hallucination():
    xs = [
        evidence("1", ["SUPPORTED", "UNSUPPORTED"]),
        evidence("2", ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"]),
        evidence("3", [], answerability="UNANSWERABLE", abstained=True),
        evidence("4", ["UNSUPPORTED"], answerability="UNANSWERABLE", abstained=False),
    ]
    scored = score_evidence_annotations(xs, expected_ids=["1", "2", "3", "4"])
    assert scored["all_claims"] == 6
    assert scored["evaluable_claims"] == 5
    assert scored["unverifiable_claims"] == 1
    assert scored["unsupported_claim_rate"] == pytest.approx(2 / 5)
    assert scored["contradiction_rate"] == pytest.approx(1 / 5)
    assert scored["hallucinated_claim_rate"] == pytest.approx(3 / 5)
    assert scored["answer_hallucination_rate"] == pytest.approx(1)
    assert scored["false_assertion_rate"] == pytest.approx(1 / 2)
    assert scored["correct_abstention_rate"] == pytest.approx(1 / 2)
    assert scored["false_refusal_rate"] == pytest.approx(0.0)
    assert scored["answer_coverage_rate"] == pytest.approx(3 / 4)


def test_hallucination_fail_closed_missing_judge_and_coverage():
    xs = [evidence("1", ["SUPPORTED"])]
    with pytest.raises(AnnotationError, match="Incomplete"):
        score_evidence_annotations(xs, expected_ids=["1", "2"])
    partial = score_evidence_annotations(
        xs, expected_ids=["1", "2"], require_complete=False)
    assert partial["status"] == "PARTIAL_JUDGE_COVERAGE"
    assert partial["judge_coverage"] == 0.5
    xs[0]["judge_provenance"]["evidence_sha256"] = ""
    with pytest.raises(AnnotationError, match="provenance"):
        score_evidence_annotations(xs)
    with pytest.raises(AnnotationError, match="No evidence"):
        score_evidence_annotations([])


def test_hallucination_template_is_not_measured():
    rows = read_annotations(BASE / "configs/hallucination_annotation.example.jsonl")
    assert len(rows) == 1
    assert rows[0]["judge_provenance"]["judge_id"].startswith("REPLACE_")


def test_always_refusing_cannot_game_hallucination():
    xs = [
        evidence("1", [], answerability="ANSWERABLE", abstained=True),
        evidence("2", [], answerability="UNANSWERABLE", abstained=True),
    ]
    s = score_evidence_annotations(xs, expected_ids=["1", "2"])
    assert s["answer_hallucination_rate"] is None
    assert s["hallucinated_claim_rate"] is None
    assert s["false_refusal_rate"] == 1.0
    assert s["answer_coverage_rate"] == 0.0
