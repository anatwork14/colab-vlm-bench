"""Matched Base-vs-MTP analysis with bootstrap descriptive uncertainty."""
from __future__ import annotations

import random
from collections import defaultdict
from math import isfinite


def percentile(samples: list[float], p: float) -> float:
    xs = sorted(samples)
    if not xs:
        raise ValueError("No samples")
    at = (len(xs) - 1) * p
    low, high = int(at), min(int(at) + 1, len(xs) - 1)
    return xs[low] + (xs[high] - xs[low]) * (at - low)


# E2E latency definitions. Receipts from before the client CCU-slot wait was added to E2E
# (ebbf60d) have no client_queue_wait fields. They measured a different quantity.
E2E_CURRENT = "INCLUDES_CLIENT_QUEUE_WAIT"
E2E_LEGACY = "LEGACY_NO_QUEUE_FIELD"


def e2e_definition(receipt: dict) -> str:
    metrics = receipt.get("metrics") or {}
    return E2E_CURRENT if "client_queue_wait_p50_ms" in metrics else E2E_LEGACY


def paired_speedups(records: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for rec in records:
        cell = rec["cell"]
        key = (cell["dataset"], cell["model"], cell["input_length"],
               cell["concurrency"], cell["repetition"])
        groups[key][cell["mode"]] = rec

    cohorts = defaultdict(list)
    exclusions = defaultdict(int)
    mixed_e2e = defaultdict(int)
    for key, pair in groups.items():
        if set(pair) != {"base", "mtp"}:
            exclusions[key[:-1]] += 1
            continue
        base, mtp = pair["base"], pair["mtp"]
        if base.get("status") != "PASS_MEASURED" or mtp.get("status") != "PASS_MEASURED":
            exclusions[key[:-1]] += 1
            continue
        if not base.get("sample_id_sha256") or base["sample_id_sha256"] != mtp.get("sample_id_sha256"):
            exclusions[key[:-1]] += 1
            continue
        # R4-B: the throughput speedup does not use E2E, so a definition mismatch does not
        # drop the pair. It is counted and each repetition records both definitions.
        base_def, mtp_def = e2e_definition(base), e2e_definition(mtp)
        if base_def != mtp_def:
            mixed_e2e[key[:-1]] += 1
        a = base.get("metrics", {}).get("output_tokens_per_s")
        b = mtp.get("metrics", {}).get("output_tokens_per_s")
        if not isinstance(a, (float, int)) or not isinstance(b, (float, int)) or a <= 0 or b <= 0:
            exclusions[key[:-1]] += 1
            continue
        cohorts[key[:-1]].append({"repetition": key[-1], "throughput_speedup": b / a,
                                   "base_tokens_per_sec": a, "mtp_tokens_per_sec": b,
                                   "e2e_definitions": [base_def, mtp_def]})
    result = []
    for cohort in sorted(set(cohorts) | set(exclusions)):
        samples = sorted(cohorts[cohort], key=lambda x: x["repetition"])
        values = [x["throughput_speedup"] for x in samples]
        summary = {
            "dataset": cohort[0], "model": cohort[1], "input_length": cohort[2],
            "concurrency": cohort[3], "paired_repetitions": len(values),
            "excluded_repetitions": exclusions[cohort],
            # Included pairs whose two receipts differ in E2E definition. Throughput is unaffected.
            "mixed_e2e_definition_pairs": mixed_e2e[cohort],
            "throughput_speedup_median": percentile(values, 0.5) if values else None,
            "status": "PAIRED_SAMPLES" if len(values) >= 3 else "INSUFFICIENT_PAIRED_REPS",
            "per_rep": samples,
            "ci95_bootstrap_descriptive": None,
        }
        if len(values) >= 3:
            rng = random.Random(42)
            bs = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(1000)]
            summary["ci95_bootstrap_descriptive"] = [
                percentile(bs, 0.025), percentile(bs, 0.975),
            ]
        result.append(summary)
    return result
