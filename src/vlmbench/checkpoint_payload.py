"""Read-only safetensors payload audit: reconcile HF index bytes from tensor headers.

No GPU, no model instantiation, no downloads, no weights copied. Reads exact
safetensors dtype/shape metadata and optionally computes shard SHA256.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors import safe_open

_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "F8_E4M3FN": 1, "F8_E5M2FN": 1,
    "BF16": 2, "F16": 2, "I16": 2, "U16": 2,
    "F32": 4, "I32": 4, "U32": 4,
    "F64": 8, "I64": 8, "U64": 8,
    "F4": None, "F6_E2M3": None,  # refuse packed types until audited
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_safetensors_payload(
    checkpoint_dir: str | Path, *, hash_shards: bool = False
) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    index_path = root / "model.safetensors.index.json"
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map")
    total_size = data.get("metadata", {}).get("total_size")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Safetensors index has no nonempty weight_map")
    if type(total_size) is not int or total_size <= 0:
        raise ValueError("Safetensors index total_size must be positive integer")

    names_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, shard in weight_map.items():
        if (not isinstance(key, str) or not key or not isinstance(shard, str)
                or not shard.endswith(".safetensors") or Path(shard).name != shard):
            raise ValueError("Invalid tensor key or safetensors shard path in index")
        names_by_shard[shard].add(key)

    actual_total = 0
    param_count = 0
    dtype_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tensor_count": 0, "parameters": 0, "bytes": 0}
    )
    shard_info = {}
    for shard_name, expected_keys in sorted(names_by_shard.items()):
        shard_path = root / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint shard: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as reader:
            found_keys = set(reader.keys())
            if found_keys != expected_keys:
                missing = sorted(expected_keys - found_keys)
                extras = sorted(found_keys - expected_keys)
                raise ValueError(
                    f"Index/shard keys differ for {shard_name}: "
                    f"missing={missing[:5]}, extra={extras[:5]}"
                )
            shard_bytes = 0
            for key in sorted(expected_keys):
                info = reader.get_slice(key)
                dtype = info.get_dtype()
                if dtype not in _DTYPE_BYTES or _DTYPE_BYTES[dtype] is None:
                    raise ValueError(f"Unsupported packed/unknown dtype {dtype!r} in {key}")
                dims = info.get_shape()
                # Scalar tensors have shape [] and contain exactly 1 element.
                if any(not isinstance(d, int) or d <= 0 for d in dims):
                    raise ValueError(f"Invalid tensor shape {dims} in {key}")
                n = math.prod(dims)
                nbytes = n * _DTYPE_BYTES[dtype]
                dtype_stats[dtype]["tensor_count"] += 1
                dtype_stats[dtype]["parameters"] += n
                dtype_stats[dtype]["bytes"] += nbytes
                param_count += n
                actual_total += nbytes
                shard_bytes += nbytes
        shard_info[shard_name] = {
            "tensor_count": len(expected_keys),
            "payload_bytes": shard_bytes,
            "file_size_bytes": shard_path.stat().st_size,
            "sha256": _file_sha256(shard_path) if hash_shards else None,
        }

    difference = actual_total - total_size
    return {
        "status": "PASS_HEADER_BYTE_RECONCILIATION" if difference == 0 else "FAIL_INDEX_BYTE_MISMATCH",
        "checkpoint_path": str(root),
        "index_sha256": _file_sha256(index_path),
        "index_metadata_total_size_bytes": total_size,
        "sum_tensor_dtype_times_shape_bytes": actual_total,
        "difference_bytes": difference,
        "total_tensors": len(weight_map),
        "total_tensor_elements": param_count,
        "dtypes": dict(sorted(dtype_stats.items())),
        "shards": shard_info,
        "source_revision_verified": False,
        "source_revision_note": "Index and local header checks alone do not prove HF revision/weights provenance.",
        "full_file_integrity_verified": False,
        "shard_sha256_computed": bool(hash_shards),
        "integrity_note": "Computed hashes are observations until compared against a trusted pinned publisher manifest.",
    }


def require_exact_payload_match(result: dict) -> None:
    if result["status"] != "PASS_HEADER_BYTE_RECONCILIATION":
        raise ValueError(
            f"Mismatch index.total_size - computed dtype/shape bytes: "
            f"{result['difference_bytes']} bytes"
        )


def safe_receipt_output_path(checkpoint_dir: str | Path, output_path: str | Path) -> Path:
    """Keep optional audit artifacts outside the original checkpoint tree."""
    checkpoint = Path(checkpoint_dir).resolve()
    output = Path(output_path).resolve()
    if output == checkpoint or checkpoint in output.parents:
        raise ValueError("Audit JSON output must be outside the original checkpoint source")
    return output
