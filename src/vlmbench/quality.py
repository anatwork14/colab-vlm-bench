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
    gt = {str(x["id"]): x.get("reference") for x in examples}
    points = []
    for pred in predictions:
        sid = str(pred.get("id"))
        if sid not in gt or gt[sid] is None:
            continue
        ok = pred.get("status") == "SUCCESS"
        answer = str(pred.get("prediction") or "") if ok else ""
        points.append({
            "id": sid, "valid_prediction": ok,
            "exact_match": exact_match(answer, str(gt[sid])),
            "token_f1": token_f1(answer, str(gt[sid])),
        })
    if not points:
        return {"status": "PROXY_UNAVAILABLE", "count": 0}
    return {
        "status": "QUALITY_PROXY_ONLY", "count": len(points),
        "exact_match": sum(x["exact_match"] for x in points) / len(points),
        "token_f1": sum(x["token_f1"] for x in points) / len(points),
        "failures_counted_zero": sum(not x["valid_prediction"] for x in points),
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
