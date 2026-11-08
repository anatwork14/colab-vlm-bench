import json
from pathlib import Path

from vlmbench.analysis import paired_speedups
from vlmbench.derivation import generate_candidates
from vlmbench.workloads import load_jsonl, exact_bin_pool


def test_pairwise_speedup_only_with_same_samples():
    records = []
    for rep in range(3):
        for mode, speed in (("base", 100.0), ("mtp", 150.0)):
            records.append({
                "cell": {"dataset": "memlens", "model": "qwen35_08b",
                         "input_length": 2048, "concurrency": 8,
                         "repetition": rep, "mode": mode},
                "status": "PASS_MEASURED", "sample_id_sha256": f"same-{rep}",
                "metrics": {"output_tokens_per_s": speed},
            })
    result = paired_speedups(records)
    assert result[0]["throughput_speedup_median"] == 1.5
    assert result[0]["paired_repetitions"] == 3
    assert result[0]["ci95_bootstrap_descriptive"] == [1.5, 1.5]
    records[-1]["sample_id_sha256"] = "mismatch"
    result = paired_speedups(records)
    assert result[0]["paired_repetitions"] == 2
    assert result[0]["status"] == "INSUFFICIENT_PAIRED_REPS"


def test_derive_seven_source_based_bins(tmp_path):
    native = tmp_path / "memlens_native.jsonl"
    original = {"id": "q01", "dataset": "memlens",
                "track": "native_quality", "native_context_complete": True,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "a " * 3000},
                    {"type": "text", "text": "Question: which one?"},
                ]}], "reference": "a"}
    native.write_text(json.dumps(original) + "\n", encoding="utf8")
    derived = tmp_path / "serving.jsonl"
    meta = generate_candidates(native, derived, "memlens")
    assert meta["derived_records"] == 7
    rows = load_jsonl(derived, dataset="memlens", track="derived_serving")
    assert all("reference" not in r for r in rows)
    assert all(r["meta"]["official_quality_allowed"] is False for r in rows)
    assert all(r["model_token_counts"] == {} for r in rows)
    assert exact_bin_pool(rows, "qwen35_08b", 2048, 0.05) == []
