"""No-model-load, CPU-only safetensors dtype/shape reconciliation."""
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from vlmbench.checkpoint_payload import (
    audit_safetensors_payload, require_exact_payload_match,
)


def fixture_ckpt(tmp_path: Path, *, total_size=32, missing=False):
    weights = {
        "embedding": torch.ones(2, 3, dtype=torch.bfloat16),
        "norm": torch.ones(5, dtype=torch.float32),
    }
    filename = "model.safetensors"
    if not missing:
        save_file(weights, tmp_path / filename)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size},
        "weight_map": {k: filename for k in weights},
    }))
    return tmp_path


def test_header_bytes_reconcile_mixed_bf16_fp32(tmp_path):
    report = audit_safetensors_payload(fixture_ckpt(tmp_path))
    require_exact_payload_match(report)
    assert report["status"] == "PASS_HEADER_BYTE_RECONCILIATION"
    assert report["sum_tensor_dtype_times_shape_bytes"] == 32
    assert report["total_tensors"] == 2
    assert report["total_tensor_elements"] == 11
    assert report["dtypes"]["BF16"] == {"tensor_count": 1, "parameters": 6, "bytes": 12}
    assert report["dtypes"]["F32"] == {"tensor_count": 1, "parameters": 5, "bytes": 20}
    assert report["source_revision_verified"] is False
    assert report["full_file_integrity_verified"] is False
    assert report["shards"]["model.safetensors"]["sha256"] is None


def test_header_audit_detects_mismatched_index_size(tmp_path):
    report = audit_safetensors_payload(fixture_ckpt(tmp_path, total_size=30))
    assert report["difference_bytes"] == 2
    with pytest.raises(ValueError, match="Mismatch"):
        require_exact_payload_match(report)


def test_header_audit_detects_missing_shard(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing checkpoint shard"):
        audit_safetensors_payload(fixture_ckpt(tmp_path, missing=True))


def test_index_shard_key_mismatch_is_rejected(tmp_path):
    fixture_ckpt(tmp_path)
    idx = tmp_path / "model.safetensors.index.json"
    d = json.loads(idx.read_text())
    d["weight_map"]["unknown"] = "model.safetensors"
    idx.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Index/shard keys differ"):
        audit_safetensors_payload(tmp_path)


def test_optional_full_shard_hash(tmp_path):
    report = audit_safetensors_payload(fixture_ckpt(tmp_path), hash_shards=True)
    assert len(report["shards"]["model.safetensors"]["sha256"]) == 64
    assert report["shard_sha256_computed"] is True
    assert report["full_file_integrity_verified"] is False


def test_scalar_tensor_shape_is_valid(tmp_path):
    save_file({"scalar": torch.tensor(7.0, dtype=torch.float32)},
              tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": 4},
        "weight_map": {"scalar": "model.safetensors"},
    }))
    report = audit_safetensors_payload(tmp_path)
    require_exact_payload_match(report)
    assert report["total_tensor_elements"] == 1
    assert report["sum_tensor_dtype_times_shape_bytes"] == 4
