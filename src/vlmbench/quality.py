"""Auxiliary EM/F1 only; official scores imported from independent upstream evaluator."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


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
    for sid, reference in references.items():
        pred = by_id.get(sid)
        if pred is None:
            missing_predictions += 1
            ok, answer = False, ""
        else:
            ok = pred.get("status") == "SUCCESS"
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
        "failures_counted_zero": sum(not x["valid_prediction"] for x in points),
        "missing_predictions": missing_predictions,
        "unscored_without_reference": unscored_without_reference,
        "not_official": True, "per_sample": points,
    }


def import_official(path: str | Path, *, dataset: str, judge_revision: str,
                    sample_manifest_sha256: str) -> dict:
    """Import upstream scoring receipts, don't rename an arbitrary proxy as official."""
    if dataset not in ("memlens", "mmlongbench_doc"):
        raise ValueError("Unsupported official scorer")
    if not judge_revision or not sample_manifest_sha256:
        raise ValueError("Official score must pin scorer and samples")
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    return {
        "status": "QUALITY_OFFICIAL_IMPORTED_UNVERIFIED",
        "dataset": dataset, "scorer_revision": judge_revision,
        "sample_manifest_sha256": sample_manifest_sha256,
        "official_source_path": str(path),
        "scorer_payload": payload,
        "note": "Separate auditor verifies original evaluator parity and denominators before claiming PASS",
    }
