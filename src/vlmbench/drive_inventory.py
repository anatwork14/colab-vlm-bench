"""Read-only, bounded inventory of the EXISTING STARLIGHT Google Drive layout.

No network, no model loading, no CPU/GPU downloads, no writes, no new
directories. This identifies legacy sources, but NEVER qualifies data integrity.
The new agent-controller 'drive_root' is a separate namespace.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LEGACY_DATASET_REL = Path("Research/Topic/Speculative Decoding/STARLIGHT")
LEGACY_MODEL_REL = Path("LLM_Weights/models/Qwen")
ACTIVE_MODELS = ("Qwen3.5-0.8B", "Qwen3.5-4B")
ACTIVE_DATASETS = ("memlens", "amber", "mmlongbench_doc")
MAX_LISTED = 32
MAX_MANIFEST_BYTES = 1024 * 1024


def _list_shallow(path: Path, max_names: int = MAX_LISTED) -> dict[str, Any]:
    if not path.is_dir():
        return {"exists": False, "path": str(path), "entries_sample": []}
    names: list[str] = []
    count_capped = False
    # Do not recurse into needle_images, the HF cache, or all PDF pages.
    iterator = iter(path.iterdir())
    for _ in range(max_names + 1):
        try:
            child = next(iterator)
        except StopIteration:
            break
        if len(names) == max_names:
            count_capped = True
            break
        names.append(child.name)
    return {
        "exists": True, "path": str(path),
        "entries_sample": sorted(names),
        "entries_truncated": count_capped,
    }


def _safe_small_json(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size > MAX_MANIFEST_BYTES:
        return None
    try:
        body = json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return body if isinstance(body, dict) else None


def _model_probe(path: Path) -> dict[str, Any]:
    info = _list_shallow(path)
    if not info["exists"]:
        return {**info, "reuse_status": "NOT_PRESENT"}
    config = path / "config.json"
    tokenizer = any((path / name).is_file() for name in (
        "tokenizer.json", "tokenizer.model", "tokenizer_config.json"))
    index = _safe_small_json(path / "model.safetensors.index.json")
    if index and isinstance(index.get("weight_map"), dict):
        names = sorted(set(map(str, index["weight_map"].values())))
    else:
        # Common single-file, sharded and alternate checkpoint conventions.
        names = [x.name for x in path.iterdir()
                 if x.is_file() and x.suffix in (".safetensors", ".bin")]
    weights = bool(names) and all(
        (path / name).is_file() and (path / name).stat().st_size > 0
        for name in names
    )
    manifest = _safe_small_json(path / ".hf_source.json")
    revision = manifest.get("revision") if manifest else None
    status = (
        "CANDIDATE_REUSE_NEEDS_HASH_PROCESSOR_AND_REVISION_AUDIT"
        if config.is_file() and tokenizer and weights
        else "INCOMPLETE_OR_UNVERIFIED_CHECKPOINT"
    )
    return {
        **info, "reuse_status": status,
        "config_present": config.is_file(),
        "tokenizer_present": tokenizer,
        "weight_files_present": weights,
        "weight_file_count": len(names),
        "weight_total_bytes": sum((path / name).stat().st_size for name in names
                                 if (path / name).is_file()),
        "pinned_revision_from_existing_manifest": revision,
        "warning": "Presence and byte counts do not prove SHA256, checkpoint loadability or A100 compatibility.",
    }


def _dataset_probe(dataset: str, path: Path) -> dict[str, Any]:
    info = _list_shallow(path)
    if not info["exists"]:
        return {**info, "reuse_status": "NOT_PRESENT"}
    status = _safe_small_json(path / "_STATUS.json")
    reported = (status or {}).get("status")
    evidence = {}
    if dataset == "memlens":
        evidence = {
            "dataset_32k_json": (path / "dataset_32k.json").is_file(),
            "needle_image_directory": (path / "release_images/needle_images").is_dir(),
            "haystack_image_directory": (path / "release_images/haystack_images").is_dir(),
        }
        ready = all(evidence.values())
    elif dataset == "amber":
        evidence = {
            "archive_directory": (path / "_archives").is_dir(),
            "official_code_directory": (path / "official_code").is_dir(),
        }
        ready = False  # Need real image/annotation asset and scorer parity checks.
    else:
        data = path / "data"
        document_dir = path / "documents"
        qa_names = ("qa.parquet", "qa.json", "qa.jsonl", "test.parquet")
        evidence = {
            "data_directory": data.is_dir(),
            "documents_directory": document_dir.is_dir(),
            "qa_filename_candidates": [n for n in qa_names if (data / n).is_file()],
            "documents_listing": _list_shallow(document_dir, 12),
        }
        ready = bool(evidence["qa_filename_candidates"]) and document_dir.is_dir()
    return {
        **info, "reuse_status": (
            "CANDIDATE_REUSE_NEEDS_CONTENT_SHA_AND_SCORER_AUDIT"
            if ready else "PARTIAL_OR_UNQUALIFIED_SOURCE"
        ),
        "source_files": evidence,
        "historical_status_from_status_json": reported,
        "warning": "Dataset staging/status is not full-file integrity or official scorer qualification.",
    }


def inspect_existing_drive(my_drive: Path = Path("/content/drive/MyDrive")) -> dict:
    """Inspect the two historical MyDrive trees without altering their contents."""
    root = Path(my_drive)
    if not root.is_dir():
        return {
            "status": "DRIVE_NOT_MOUNTED",
            "my_drive_root": str(root),
            "instructions": "Authenticate then mount Drive in an existing CPU Colab session.",
        }
    model_root = root / LEGACY_MODEL_REL
    datasets_root = root / LEGACY_DATASET_REL
    result = {
        "status": "READ_ONLY_DISCOVERY_NOT_INTEGRITY_QUALIFICATION",
        "my_drive_root": str(root),
        "legacy_model_root": str(model_root),
        "legacy_dataset_root": str(datasets_root),
        "legacy_models": {
            "qwen35_08b": _model_probe(model_root / ACTIVE_MODELS[0]),
            "qwen35_4b": _model_probe(model_root / ACTIVE_MODELS[1]),
        },
        "legacy_datasets": {
            key: _dataset_probe(key, datasets_root / key)
            for key in ACTIVE_DATASETS
        },
        "historical_audit": {
            "manual_inspection": _list_shallow(
                datasets_root / "_manual_inspection_v2", 12),
            "source_ledger_present": (datasets_root / "_LAST_RUN.json").is_file(),
        },
        "new_agent_default_root": str(root / "Colab-VLM-Bench-Agent"),
        "new_default_is_separate": True,
        "policy": (
            "Cache-first discovery only. Do not run HF prepare against the new "
            "root until existing sources have been mapped, hash-checked and "
            "approved for read-only reuse or explicitly found incomplete."
        ),
    }
    return result
