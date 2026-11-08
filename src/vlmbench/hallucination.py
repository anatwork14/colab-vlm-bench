"""Validated evidence-claim annotations for hallucination scoring.

This code never invents a VLM claim judgment or substitutes EM/F1 for factual
support. Labels must come from an independent judge/human auditor with provenance.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

VERDICTS = frozenset({"SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "UNVERIFIABLE"})
ANSWERABILITY = frozenset({"ANSWERABLE", "UNANSWERABLE"})


class AnnotationError(ValueError):
    pass


def score_evidence_annotations(
    annotations: list[dict], *,
    expected_ids: list[str] | None = None,
    require_complete: bool = True,
) -> dict:
    """Claim-level support and response-level hallucination with explicit coverage.

    Rate denominator is only evidence-evaluable claims (not UNVERIFIABLE). Unknown
    is never auto-negative. Abstentions cannot be treated as factual claims.
    """
    if not annotations:
        raise AnnotationError("No evidence-claim annotations; metrics NOT_MEASURED")
    seen, claims = set(), Counter()
    total_answers = 0
    eval_answers = 0
    hallucinated_answers = 0
    unanswerable_total = 0
    false_assertions = 0
    correct_abstentions = 0
    all_details = []
    for entry in annotations:
        identifier = str(entry.get("id") or "")
        if not identifier or identifier in seen:
            raise AnnotationError("Missing or duplicate answer ID")
        seen.add(identifier)
        judge = entry.get("judge_provenance")
        if not isinstance(judge, dict) or not all(
            judge.get(k) for k in ("judge_id", "judge_revision", "evidence_sha256", "rubric_version")
        ):
            raise AnnotationError(f"Missing independent judge and evidence provenance for {identifier}")
        ans = entry.get("answerability")
        if ans not in ANSWERABILITY:
            raise AnnotationError(f"Unknown answerability for {identifier}")
        abstained = entry.get("abstained")
        if type(abstained) is not bool:
            raise AnnotationError("Abstained must be a verified boolean judge label")
        raw_claims = entry.get("claims")
        if not isinstance(raw_claims, list):
            raise AnnotationError("claims must be a list, even for abstentions")
        if abstained and raw_claims:
            raise AnnotationError("Abstentions must not carry asserted factual claims")
        local_verdicts = Counter()
        for claim in raw_claims:
            if not claim.get("text"):
                raise AnnotationError("Every factual claim must have text")
            verdict = claim.get("verdict")
            if verdict not in VERDICTS:
                raise AnnotationError(f"Invalid support verdict: {verdict}")
            evidence_ids = claim.get("evidence_ids")
            if not isinstance(evidence_ids, list) or not all(isinstance(e, str) for e in evidence_ids):
                raise AnnotationError("evidence_ids must be an explicit string list")
            if verdict in ("SUPPORTED", "CONTRADICTED") and not evidence_ids:
                raise AnnotationError("Supported/refuted evidence must cite actual evidence IDs")
            local_verdicts[verdict] += 1
            claims[verdict] += 1
        total_answers += 1
        count_evaluable = sum(local_verdicts[v] for v in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"))
        has_hallucination = bool(local_verdicts["UNSUPPORTED"] or local_verdicts["CONTRADICTED"])
        if not abstained and count_evaluable:
            eval_answers += 1
            hallucinated_answers += int(has_hallucination)
        if ans == "UNANSWERABLE":
            unanswerable_total += 1
            correct_abstentions += int(abstained)
            false_assertions += int(not abstained)
        all_details.append({
            "id": identifier, "claims": dict(local_verdicts),
            "answerability": ans, "abstained": abstained,
            "has_hallucination": has_hallucination if count_evaluable else None,
        })
    if expected_ids is not None:
        expected = set(map(str, expected_ids))
        if len(expected) != len(expected_ids) or not seen.issubset(expected):
            raise AnnotationError("Annotation IDs do not match frozen evaluated population")
        if require_complete and seen != expected:
            raise AnnotationError(f"Incomplete judge coverage: {len(seen)}/{len(expected)}")
    evaluable_claims = sum(claims[v] for v in ("SUPPORTED", "UNSUPPORTED", "CONTRADICTED"))
    return {
        "status": "EVIDENCE_JUDGE_SCORED" if (
            expected_ids is None or set(map(str, expected_ids)) == seen
        ) else "PARTIAL_JUDGE_COVERAGE",
        "annotated_answers": total_answers,
        "expected_answers": len(expected_ids) if expected_ids is not None else None,
        "judge_coverage": total_answers / len(expected_ids) if expected_ids is not None else None,
        "all_claims": sum(claims.values()),
        "evaluable_claims": evaluable_claims,
        "unverifiable_claims": claims["UNVERIFIABLE"],
        "evaluable_answer_count": eval_answers,
        "unsupported_claim_rate": (
            claims["UNSUPPORTED"] / evaluable_claims if evaluable_claims else None),
        "contradiction_rate": (
            claims["CONTRADICTED"] / evaluable_claims if evaluable_claims else None),
        "hallucinated_claim_rate": (
            (claims["UNSUPPORTED"] + claims["CONTRADICTED"]) / evaluable_claims
            if evaluable_claims else None),
        "answer_hallucination_rate": (
            hallucinated_answers / eval_answers if eval_answers else None),
        "unanswerable_questions": unanswerable_total,
        "false_assertion_rate": (
            false_assertions / unanswerable_total if unanswerable_total else None),
        "correct_abstention_rate": (
            correct_abstentions / unanswerable_total if unanswerable_total else None),
        "denominator_policy": {
            "unsupported_claim_rate": "UNSUPPORTED / (SUPPORTED + UNSUPPORTED + CONTRADICTED)",
            "hallucinated_claim_rate": "(UNSUPPORTED + CONTRADICTED) / evaluable factual claims",
            "answer_hallucination_rate": "non-abstained answers containing >=1 evaluable claim and >=1 hallucinated claim / non-abstained answers containing >=1 evaluable claim",
            "false_assertion_rate": "non-abstained on independently labeled UNANSWERABLE questions / all UNANSWERABLE",
            "unknown": "UNVERIFIABLE excluded, always reported separately",
        },
        "per_answer": all_details,
        "warning": "Provenance validation alone is not proof of judge correctness; audit sampled claims",
    }


def read_annotations(path: str | Path) -> list[dict]:
    rows = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise AnnotationError(f"Bad JSONL line {lineno}") from exc
    return rows
