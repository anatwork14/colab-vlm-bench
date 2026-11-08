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


def paired_speedups(records: list[dict]) -> list[dict]:
    groups = defaultdict(dict)
    for rec in records:
        cell = rec["cell"]
        key = (cell["dataset"], cell["model"], cell["input_length"],
               cell["concurrency"], cell["repetition"])
        groups[key][cell["mode"]] = rec

    cohorts = defaultdict(list)
    exclusions = defaultdict(int)
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
        a = base.get("metrics", {}).get("output_tokens_per_s")
        b = mtp.get("metrics", {}).get("output_tokens_per_s")
        if not isinstance(a, (float, int)) or not isinstance(b, (float, int)) or a <= 0 or b <= 0:
            exclusions[key[:-1]] += 1
            continue
        cohorts[key[:-1]].append({"repetition": key[-1], "throughput_speedup": b / a,
                                   "base_tokens_per_sec": a, "mtp_tokens_per_sec": b})
    result = []
    for cohort in sorted(set(cohorts) | set(exclusions)):
        samples = sorted(cohorts[cohort], key=lambda x: x["repetition"])
        values = [x["throughput_speedup"] for x in samples]
        summary = {
            "dataset": cohort[0], "model": cohort[1], "input_length": cohort[2],
            "concurrency": cohort[3], "paired_repetitions": len(values),
            "excluded_repetitions": exclusions[cohort],
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
