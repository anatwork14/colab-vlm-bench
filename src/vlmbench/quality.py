"""Auxiliary EM/F1 only; official scores imported from independent upstream evaluator."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from .signed_records import sha256_hex


def deterministic_subset(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Stable matched subset for engineering pilot only; never official full score."""
    if count <= 0 or count >= len(rows):
        return list(rows)
    ranked = sorted(
        rows,
        key=lambda r: hashlib.sha256(
            (str(seed) + "|" + str(r["id"])).encode("utf8")
        ).hexdigest(),
    )
    ids = {str(row["id"]) for row in ranked[:count]}
    return [row for row in rows if str(row["id"]) in ids]


def normalize(s: str) -> list[str]:
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    return re.findall(r"\w+", s, flags=re.UNICODE)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = normalize(prediction), normalize(reference)
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_proxy(predictions: list[dict], examples: list[dict]) -> dict:
    """Proxy EM/F1 over the frozen example set.

    Denominator = every example that has a reference. An example with no prediction row
    (crash, circuit breaker, lost resume row) scores EM=0 and F1=0; it is never dropped.
    Duplicate prediction IDs and prediction IDs outside the frozen set are errors.
    """
    references: dict[str, str] = {}
    known: set[str] = set()
    unscored_without_reference = 0
    for example in examples:
        sid = str(example["id"])
        if sid in known:
            raise ValueError(f"Duplicate example id: {sid}")
        known.add(sid)
        if example.get("reference") is None:
            unscored_without_reference += 1
            continue
        references[sid] = str(example["reference"])

    by_id: dict[str, dict] = {}
    for pred in predictions:
        sid = str(pred.get("id"))
        if sid in by_id:
            raise ValueError(f"Duplicate prediction id: {sid}")
        if sid not in known:
            raise ValueError(f"Unknown prediction id not in frozen examples: {sid}")
        by_id[sid] = pred

    points = []
    missing_predictions = 0
    invalid_predictions = 0
    for sid, reference in references.items():
        pred = by_id.get(sid)
        if pred is None:
            missing_predictions += 1
            ok, answer = False, ""
        else:
            ok = pred.get("status") == "SUCCESS"
            if not ok:
                invalid_predictions += 1
            answer = str(pred.get("prediction") or "") if ok else ""
        points.append({
            "id": sid, "valid_prediction": ok,
            "exact_match": exact_match(answer, reference),
            "token_f1": token_f1(answer, reference),
        })
    if not points:
        return {"status": "PROXY_UNAVAILABLE", "count": 0,
                "unscored_without_reference": unscored_without_reference}
    return {
        "status": "QUALITY_PROXY_ONLY", "count": len(points),
        "exact_match": sum(x["exact_match"] for x in points) / len(points),
        "token_f1": sum(x["token_f1"] for x in points) / len(points),
        # Disjoint: a row that is present but not SUCCESS, and an example with no row.
        # failures_counted_zero = invalid_predictions_counted_zero + missing_predictions.
        "failures_counted_zero": sum(not x["valid_prediction"] for x in points),
        "invalid_predictions_counted_zero": invalid_predictions,
        "missing_predictions": missing_predictions,
        "unscored_without_reference": unscored_without_reference,
        "not_official": True, "per_sample": points,
    }


def import_official(path: str | Path, *, dataset: str, judge_revision: str,
                    sample_manifest_sha256: str, qualification: str | Path | None = None,
                    allowed_signers: str | Path | None = None, runner=subprocess.run) -> dict:
    """Import an upstream scoring result as official only with scorer provenance and a signed
    extractor-stage qualification (ITER3 C4). Otherwise the status is BLOCKED_NO_JUDGE and no payload
    is labelled official. A proxy metric is never imported as official.

    Trust: the qualifier's signature binds this file's SHA-256 to the official upstream scorer and
    the official extractor. It does not show that the upstream run was honest.
    """
    if dataset not in ("memlens", "mmlongbench_doc"):
        raise ValueError("Unsupported official scorer")
    if not judge_revision or not sample_manifest_sha256:
        raise ValueError("Official score must pin scorer and samples")
    raw = Path(path).read_bytes()
    base = {"dataset": dataset, "scorer_revision": judge_revision,
            "sample_manifest_sha256": sample_manifest_sha256, "official_source_path": str(path),
            "upstream_result_sha256": sha256_hex(raw), "official_acc": None, "official_f1": None}
    if dataset == "memlens":
        return {**base, "status": "BLOCKED_NO_JUDGE",
                "reason": "no ported scorer or pinned judge provenance for MemLens (LLM judge unqualified)"}
    from .datasets.mmlong_official_scorer import UPSTREAM_COMMIT, verify_qualification
    if judge_revision != UPSTREAM_COMMIT:
        return {**base, "status": "BLOCKED_NO_JUDGE",
                "reason": "judge_revision is not the ported upstream commit"}
    try:
        payload = json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {**base, "status": "BLOCKED_NO_JUDGE", "reason": "upstream result is not valid JSON"}
    if qualification is None:
        return {**base, "status": "BLOCKED_NO_JUDGE", "reason": "no signed extractor-stage qualification"}
    qual = verify_qualification(Path(qualification), allowed_signers=Path(allowed_signers) if allowed_signers else None,
                                sample_manifest_sha256=sample_manifest_sha256,
                                upstream_result_sha256=sha256_hex(raw), runner=runner)
    if not qual["verified"]:
        return {**base, "status": "BLOCKED_NO_JUDGE", "reason": "extractor-stage qualification did not verify",
                "qualification_reasons": qual["reasons"]}
    return {**base, "status": "OFFICIAL_IMPORTED_QUALIFIED",
            "qualification_sha256": qual["sha256"],
            "scorer_payload": payload,
            "note": "Official by signature: the qualifier vouches for this upstream result file"}
