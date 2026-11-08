import json

from vlmbench.artifacts import atomic_json, atomic_jsonl, sha256, sync_files, verified_resume


def test_atomic_hash_and_safe_non_destructive_export(tmp_path):
    root = tmp_path / "scratch"
    out = tmp_path / "drive"
    atomic_json(root / "receipt.json", {"status": "PASS_MEASURED", "successful_requests": 16})
    atomic_jsonl(root / "predictions.jsonl", [{"id": 1, "prediction": "text"}])
    sync_files(root, out)
    assert sha256(root / "receipt.json") == sha256(out / "receipt.json")
    sync_files(root, out)
    assert json.loads((out / "receipt.json").read_text())["status"] == "PASS_MEASURED"


def test_resume_requires_same_manifest_and_complete_success(tmp_path):
    p = tmp_path / "receipt.json"
    record = {"status": "PASS_MEASURED", "config_sha256": "same",
              "dataset_sha256": "data", "cell": {"id": "a"},
              "metrics": {"ttft_p95_ms": 4},
              "successful_requests": 16, "expected_requests": 16}
    atomic_json(p, record)
    assert verified_resume(p, {"config_sha256": "same",
                               "dataset_sha256": "data", "cell": {"id": "a"}})
    assert not verified_resume(p, {"config_sha256": "changed"})
    record["status"] = "PARTIAL_FAILURE"
    atomic_json(p, record)
    assert not verified_resume(p, {"config_sha256": "same"})
