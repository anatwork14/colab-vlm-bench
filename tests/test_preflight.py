"""CPU-only tests for 80GiB RAM/200GiB SSD Colab cost guards."""
import json
from pathlib import Path

import pytest

from vlmbench.preflight import (
    PreflightError, assess_resources, checkpoint_inventory,
    dataset_probe_memlens, dataset_probe_mmlong, serving_readiness,
)


def test_80gb_ram_200gb_disk_budget_is_advisory_and_real_free_required():
    normal = {"ram_total_gib": 80.0, "ram_available_gib": 72.0,
              "disk_total_gib": 200.0, "disk_free_gib": 160.0, "scratch": "/content"}
    result = assess_resources(normal, 9 * 1024**3)
    assert result["status"] == "CPU_PREFLIGHT_PASS"
    assert result["required_free_local_disk_gib"] >= 18.0
    exhausted = {**normal, "ram_available_gib": 7, "disk_free_gib": 1}
    bad = assess_resources(exhausted, 9 * 1024**3)
    assert bad["status"] == "CPU_PREFLIGHT_BLOCKED"
    assert len(bad["issues"]) == 2


def test_checkpoint_inventory_requires_real_weights_and_config(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    with pytest.raises(PreflightError, match="weights"):
        checkpoint_inventory(root)
    (root / "model.safetensors").write_bytes(b"dummy")
    snapshot = checkpoint_inventory(root)
    assert snapshot["weight_files"] == 1
    assert snapshot["total_bytes"] >= 5


def test_lightweight_dataset_probe_catches_wrong_pdf_path(tmp_path):
    mem = tmp_path / "dataset_32k.json"
    mem.write_text("[]")
    (tmp_path / "release_images").mkdir()
    assert dataset_probe_memlens(mem, tmp_path)["status"].startswith("SOURCE_")
    qa = tmp_path / "qa.json"
    qa.write_text(json.dumps([{"doc_id": "present.pdf", "question": "Q", "answer": "A"}]))
    docs = tmp_path / "documents"
    docs.mkdir()
    with pytest.raises(PreflightError, match="PDF files missing"):
        dataset_probe_mmlong(qa, docs)
    (docs / "present.pdf").write_bytes(b"pdf-placeholder")
    report = dataset_probe_mmlong(qa, docs)
    assert report["qa_rows"] == 1 and report["sampled_pdf_present"] == 1


def test_preflight_only_counts_verified_real_multimodal_tokens(tmp_path):
    path = tmp_path / "annotated.jsonl"
    row = {
        "dataset": "memlens", "track": "derived_serving",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "x"}]}],
    }
    with path.open("w") as f:
        for i in range(20):
            x = {**row, "id": f"sample_{i}", "derived_from": "source",
                 "model_token_counts": {"qwen35_08b": {
                     "verified_by": "vllm_effective_token_accounting",
                     "text_tokens": 2048, "effective_total_tokens": 2200,
                 }}}
            f.write(json.dumps(x) + "\n")
    cell = {"input_length": 2048, "concurrency": 1, "warmup": 1,
            "requests": 16, "repetition": 0}
    ready = serving_readiness(path, dataset="memlens", model_id="qwen35_08b",
                              cells=[cell], tolerance=0.05, max_model_len=32768)
    assert ready["status"] == "READY_CELLS_PRESENT"
    cell_big = {**cell, "concurrency": 32, "warmup": 32, "requests": 128}
    blocked = serving_readiness(path, dataset="memlens", model_id="qwen35_08b",
                                cells=[cell_big], tolerance=0.05, max_model_len=32768)
    assert blocked["status"] == "BLOCKED_NO_ELIGIBLE_CELLS"
    other = serving_readiness(path, dataset="memlens", model_id="qwen35_4b",
                              cells=[cell], tolerance=0.05, max_model_len=32768)
    assert other["status"] == "BLOCKED_NO_ELIGIBLE_CELLS"
