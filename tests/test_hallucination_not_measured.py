"""ITER2 D2-4 guard: the hallucination path must never turn "no data" into a zero.

Owned by the benchmark worker as a guard. hallucination.py itself is owned by p7 and is not
edited here. These tests pin current behaviour; a change that emits 0.0 for an undefined rate
must fail them.
"""
from __future__ import annotations

import pytest

from vlmbench import cli
from vlmbench.hallucination import AnnotationError, score_evidence_annotations

PROVENANCE = {"judge_id": "judge-x", "judge_revision": "rev-1",
              "evidence_sha256": "e" * 64, "rubric_version": "r1"}


def _answer(aid, claims, abstained=False, answerability="ANSWERABLE"):
    return {"id": aid, "judge_provenance": dict(PROVENANCE), "answerability": answerability,
            "abstained": abstained, "claims": claims}


def test_empty_annotations_raise_and_never_return_a_zero_score():
    with pytest.raises(AnnotationError, match="NOT_MEASURED"):
        score_evidence_annotations([])


def test_empty_jsonl_through_the_cli_writes_no_receipt(tmp_path):
    annotations = tmp_path / "empty.jsonl"
    annotations.write_text("", encoding="utf8")
    output = tmp_path / "scores.json"
    with pytest.raises(AnnotationError):
        cli.main(["hallucination-score", "--annotations", str(annotations), "--output", str(output)])
    assert not output.exists(), "no score file may be written for an empty annotation set"


def test_unverifiable_only_claims_give_undefined_rates_not_zero():
    out = score_evidence_annotations([_answer("a", [
        {"text": "claim", "verdict": "UNVERIFIABLE", "evidence_ids": []}])])
    assert out["evaluable_claims"] == 0
    assert out["unverifiable_claims"] == 1
    assert out["unsupported_claim_rate"] is None
    assert out["hallucinated_claim_rate"] is None
    assert out["answer_hallucination_rate"] is None


def test_partial_judge_coverage_is_labelled_partial_not_complete():
    out = score_evidence_annotations(
        [_answer("a", [{"text": "c", "verdict": "SUPPORTED", "evidence_ids": ["e1"]}])],
        expected_ids=["a", "b"], require_complete=False)
    assert out["status"] == "PARTIAL_JUDGE_COVERAGE"
    assert out["judge_coverage"] == pytest.approx(0.5)


def test_all_abstained_answers_is_a_measured_zero_coverage_not_a_missing_value():
    out = score_evidence_annotations([_answer("a", [], abstained=True)])
    assert out["answer_coverage_rate"] == 0.0
    assert out["answer_hallucination_rate"] is None  # no evaluable answer, so undefined
