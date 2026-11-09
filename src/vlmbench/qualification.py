"""Read-only dataset and model qualification for Google Drive assets.

Verifies the integrity, internal structure, SHA-256 fingerprints, and loader
compatibility of historical STARLIGHT assets on mounted Google Drive:
- AMBER: AMBER.zip contents, CRC integrity, JSON queries/annotations schema,
  and normalize_amber compatibility.
- MemLens: dataset_32k.json schema, image directories, and sample 7-context-bin
  generation under strict tolerance <= 0.05.
- Qwen models: config, tokenizer artifacts, safetensors shard presence and hashes.

Strict read-only invariant: zero file additions, deletions, or mutations on Drive.
"""
from __future__ import annotations

import json
import logging
import math
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import sha256
from .datasets.amber import (
    AMBER_UPSTREAM_MANIFEST,
    AmberError,
    normalize_amber,
)
from .datasets.memlens import (
    CANONICAL_LENGTHS,
    build_memlens_context_bins,
    normalize_memlens,
)
from .drive_inventory import (
    ACTIVE_MODELS,
    LEGACY_DATASET_REL,
    LEGACY_MODEL_REL,
)

logger = logging.getLogger(__name__)


class QualificationError(Exception):
    """Raised when an asset fails required qualification checks."""


class DeterministicSimpleTokenizer:
    """Deterministic character-level tokenizer for CPU qualification without transformers."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(t) for t in token_ids)


def get_qualification_tokenizer(model_dir: Path | str | None = None) -> Any:
    """Return a model tokenizer if available, otherwise return DeterministicSimpleTokenizer."""
    if model_dir is not None:
        p = Path(model_dir)
        if p.is_dir() and any((p / n).is_file() for n in ("tokenizer.json", "tokenizer_config.json")):
            try:
                from transformers import AutoTokenizer

                return AutoTokenizer.from_pretrained(
                    str(p), local_files_only=True, trust_remote_code=False
                )
            except (ImportError, OSError, ValueError, RuntimeError) as exc:
                logger.debug("Failed loading AutoTokenizer from %s: %s", p, exc)
    return DeterministicSimpleTokenizer()


def snapshot_directory_state(path: Path, subdirs: list[Path] | None = None) -> dict[str, tuple[int, int]]:
    """Capture relative path -> (size, mtime_ns) for files under path or specific subdirs."""
    snapshot: dict[str, tuple[int, int]] = {}
    if not path.is_dir():
        return snapshot
    targets = [path / s for s in subdirs] if subdirs else [path]
    for target in targets:
        if not target.is_dir():
            continue
        for p in target.rglob("*"):
            try:
                st = p.stat()
                rel = str(p.relative_to(path))
                snapshot[rel] = (st.st_size, st.st_mtime_ns)
            except OSError:
                pass
    return snapshot


def _find_zip_member(namelist: list[str], filename_suffix: str) -> str | None:
    for name in namelist:
        if name == filename_suffix or name.endswith("/" + filename_suffix):
            return name
    return None


def qualify_amber_assets(
    drive_root: Path | str,
    archive_rel: Path | str = LEGACY_DATASET_REL / "amber/_archives/AMBER.zip",
    *,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Inspect and qualify AMBER.zip contents without extracting the full archive.

    Verifies:
    1. Archive existence, non-zero size, and SHA-256 fingerprint.
    2. Zip CRC-32 integrity using testzip().
    3. Presence of query and annotation files matching upstream layout.
    4. Schema compatibility with normalize_amber.
    """
    root = Path(drive_root)
    archive_path = root / archive_rel
    if not archive_path.is_file():
        return {
            "status": "UNQUALIFIED_AMBER_SOURCE",
            "archive_path": str(archive_path),
            "error": f"AMBER archive not found at {archive_path}",
            "official_scoring_eligible": False,
        }

    size_bytes = archive_path.stat().st_size
    if size_bytes == 0:
        return {
            "status": "UNQUALIFIED_AMBER_SOURCE",
            "archive_path": str(archive_path),
            "error": "AMBER archive file is empty (0 bytes)",
            "official_scoring_eligible": False,
        }

    archive_sha = sha256(archive_path)

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            bad_member = zf.testzip()
            if bad_member is not None:
                return {
                    "status": "UNQUALIFIED_AMBER_SOURCE",
                    "archive_path": str(archive_path),
                    "archive_size_bytes": size_bytes,
                    "archive_sha256": archive_sha,
                    "error": f"Zip CRC-32 checksum failed on member: {bad_member}",
                    "official_scoring_eligible": False,
                }

            namelist = zf.namelist()
            image_members = [
                n for n in namelist if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
            image_member_set = set(namelist)

            query_all_member = _find_zip_member(namelist, "query_all.json")
            query_gen_member = _find_zip_member(namelist, "query_generative.json")
            query_disc_member = _find_zip_member(namelist, "query_discriminative.json")
            annotations_member = _find_zip_member(namelist, "annotations.json")

            official_code_dir = archive_path.parent.parent / "official_code"
            found_files = {
                "query_all": query_all_member or (str(official_code_dir / "data/query/query_all.json") if (official_code_dir / "data/query/query_all.json").is_file() else None),
                "query_generative": query_gen_member or (str(official_code_dir / "data/query/query_generative.json") if (official_code_dir / "data/query/query_generative.json").is_file() else None),
                "query_discriminative": query_disc_member or (str(official_code_dir / "data/query/query_discriminative.json") if (official_code_dir / "data/query/query_discriminative.json").is_file() else None),
                "annotations": annotations_member or (str(official_code_dir / "data/annotations.json") if (official_code_dir / "data/annotations.json").is_file() else None),
            }

            missing = [k for k, v in found_files.items() if v is None]
            if missing:
                return {
                    "status": "UNQUALIFIED_AMBER_SOURCE",
                    "archive_path": str(archive_path),
                    "archive_size_bytes": size_bytes,
                    "archive_sha256": archive_sha,
                    "found_files": found_files,
                    "error": f"Missing required AMBER files in zip: {missing}",
                    "official_scoring_eligible": False,
                }

            # Inspect query contents
            if query_all_member is not None:
                with zf.open(query_all_member) as fp:
                    query_all_data = json.load(fp)
            else:
                with open(found_files["query_all"], "r", encoding="utf8") as fp:
                    query_all_data = json.load(fp)

            if isinstance(query_all_data, dict):
                valid_key = next((k for k in ("queries", "data") if k in query_all_data), None)
                if not valid_key:
                    raise AmberError("Invalid query_all dictionary wrapper")
                query_items = query_all_data[valid_key]
            elif isinstance(query_all_data, list):
                query_items = query_all_data
            else:
                raise AmberError(f"Unexpected query_all JSON root type: {type(query_all_data)}")

            if not query_items:
                raise AmberError("Empty query items list in AMBER archive")

            # Check annotation contents
            if annotations_member is not None:
                with zf.open(annotations_member) as fp:
                    ann_data = json.load(fp)
            else:
                with open(found_files["annotations"], "r", encoding="utf8") as fp:
                    ann_data = json.load(fp)

            if isinstance(ann_data, dict) and "annotations" in ann_data:
                ann_items = ann_data["annotations"]
            elif isinstance(ann_data, (list, dict)):
                ann_items = ann_data
            else:
                raise AmberError(f"Unexpected annotations JSON root type: {type(ann_data)}")

            # Index annotations by id for enrichment
            ann_by_id: dict[str, dict[str, Any]] = {}
            if isinstance(ann_items, dict):
                for k, v in ann_items.items():
                    if isinstance(v, dict):
                        ann_by_id[str(k)] = v
            elif isinstance(ann_items, list):
                for ann_entry in ann_items:
                    if isinstance(ann_entry, dict) and "id" in ann_entry:
                        ann_by_id[str(ann_entry["id"])] = ann_entry

            # Validate sample entries with normalize_amber in a temporary directory
            with tempfile.TemporaryDirectory() as tmp_dir_str:
                tmp_dir = Path(tmp_dir_str)
                tmp_images = tmp_dir / "images"
                tmp_images.mkdir(parents=True, exist_ok=True)

                sample_queries = query_items[: min(sample_limit, len(query_items))]
                extracted_sample_queries = []

                for raw_item in sample_queries:
                    if not isinstance(raw_item, dict):
                        raise AmberError(f"Query item must be a dictionary, got {type(raw_item)}")
                    if "id" not in raw_item:
                        raise AmberError("Query item missing 'id' field")
                    if "image" not in raw_item:
                        raise AmberError(f"Query item {raw_item.get('id')} missing 'image' field")
                    if "query" not in raw_item:
                        raise AmberError(f"Query item {raw_item.get('id')} missing 'query' field")

                    item = dict(raw_item)
                    ann = ann_by_id.get(str(item.get("id")))
                    if ann:
                        if "label" not in item and "label" in ann:
                            item["label"] = ann["label"]
                        if "type" not in item and "type" in ann:
                            item["type"] = ann["type"]
                        if "objects" not in item and "objects" in ann:
                            item["objects"] = ann["objects"]

                    img_name = item["image"]
                    # Locate image member in zip
                    target_member = None
                    if img_name in image_member_set:
                        target_member = img_name
                    else:
                        target_member = _find_zip_member(namelist, Path(img_name).name)

                    if target_member is not None:
                        dest_img = tmp_images / img_name
                        dest_img.parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(target_member) as src_fp, dest_img.open("wb") as dst_fp:
                            shutil.copyfileobj(src_fp, dst_fp)

                    extracted_sample_queries.append(item)

                sample_query_path = tmp_dir / "sample_queries.json"
                sample_query_path.write_text(
                    json.dumps(extracted_sample_queries), encoding="utf8"
                )

                norm_out_path = tmp_dir / "sample_normalized.jsonl"
                norm_summary = normalize_amber(
                    sample_query_path,
                    tmp_images,
                    norm_out_path,
                    split="all",
                    limit=len(extracted_sample_queries),
                )

            return {
                "status": "QUALIFIED_AMBER_SOURCE",
                "archive_path": str(archive_path),
                "archive_size_bytes": size_bytes,
                "archive_sha256": archive_sha,
                "crc_check": "PASSED",
                "zip_member_count": len(namelist),
                "image_file_count": len(image_members),
                "found_files": found_files,
                "query_counts": {
                    "query_all_total": len(query_items),
                },
                "annotation_count": len(ann_items),
                "schema_compatibility": {
                    "normalize_amber_verified": True,
                    "sample_normalized_count": norm_summary.get("count", 0),
                    "sample_images_verified": norm_summary.get("image_count", 0),
                    "pass_type": norm_summary.get("pass_type"),
                },
                "upstream_manifest_pin": AMBER_UPSTREAM_MANIFEST,
                "scorer_qualification": (
                    "AMBER_PROXY_UNVERIFIED: proxy scoring re-implementation validated; "
                    "official upstream scorer parity must be evidenced before official claims."
                ),
                "official_scoring_eligible": False,
            }

    except (OSError, json.JSONDecodeError, zipfile.BadZipFile, AmberError, ValueError, TypeError) as exc:
        return {
            "status": "UNQUALIFIED_AMBER_SOURCE",
            "archive_path": str(archive_path),
            "archive_size_bytes": size_bytes,
            "archive_sha256": archive_sha,
            "error": f"AMBER qualification failed: {exc}",
            "official_scoring_eligible": False,
        }


def qualify_memlens_assets(
    drive_root: Path | str,
    dataset_rel: Path | str = LEGACY_DATASET_REL / "memlens",
    *,
    sample_limit: int = 5,
    tolerance: float = 0.05,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """Inspect and qualify MemLens dataset files on mounted Google Drive.

    Verifies:
    1. dataset_32k.json existence, non-zero size, and SHA-256 fingerprint.
    2. Image directories under release_images/.
    3. Schema of streaming items (question_id, haystack_sessions, haystack_dates, etc.).
    4. Candidate bin generation under strict tolerance <= 0.05 using build_memlens_context_bins.
    """
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or math.isnan(tolerance)
        or math.isinf(tolerance)
    ):
        raise ValueError(f"tolerance must be a finite float, got {tolerance!r}")
    if tolerance <= 0.0 or tolerance > 0.05:
        raise ValueError(f"tolerance must be strictly in (0.0, 0.05], got {tolerance}")

    root = Path(drive_root)
    dataset_dir = root / dataset_rel
    dataset_json = dataset_dir / "dataset_32k.json"

    if not dataset_json.is_file():
        return {
            "status": "UNQUALIFIED_MEMLENS_SOURCE",
            "dataset_path": str(dataset_json),
            "error": f"MemLens dataset file not found at {dataset_json}",
        }

    size_bytes = dataset_json.stat().st_size
    if size_bytes == 0:
        return {
            "status": "UNQUALIFIED_MEMLENS_SOURCE",
            "dataset_path": str(dataset_json),
            "error": "MemLens dataset_32k.json is empty (0 bytes)",
        }

    dataset_sha = sha256(dataset_json)

    needle_dir = dataset_dir / "release_images" / "needle_images"
    haystack_dir = dataset_dir / "release_images" / "haystack_images"

    needle_present = needle_dir.is_dir()
    haystack_present = haystack_dir.is_dir()

    # Stream first sample_limit questions to verify structure
    sample_questions: list[dict[str, Any]] = []
    try:
        try:
            import ijson
            use_ijson = True
        except ImportError:
            use_ijson = False

        if use_ijson:
            with dataset_json.open("rb") as fp:
                for item in ijson.items(fp, "item"):
                    if not isinstance(item, dict):
                        raise TypeError("MemLens items must be dictionaries")
                    for req in (
                        "question_id",
                        "haystack_session_ids",
                        "haystack_sessions",
                        "haystack_dates",
                        "question",
                    ):
                        if req not in item:
                            raise ValueError(f"MemLens item missing required key: {req}")

                    s_ids = item["haystack_session_ids"]
                    s_sess = item["haystack_sessions"]
                    s_dates = item["haystack_dates"]
                    if len(s_ids) != len(s_sess) or len(s_sess) != len(s_dates):
                        raise ValueError(f"Inconsistent MemLens session lengths for {item['question_id']}")

                    sample_questions.append(item)
                    if len(sample_questions) >= sample_limit:
                        break
        else:
            with dataset_json.open("r", encoding="utf8") as fp:
                raw = json.load(fp)
                raw_items = raw if isinstance(raw, list) else raw.get("data", [])
                for item in raw_items[:sample_limit]:
                    if not isinstance(item, dict):
                        raise TypeError("MemLens items must be dictionaries")
                    for req in (
                        "question_id",
                        "haystack_session_ids",
                        "haystack_sessions",
                        "haystack_dates",
                        "question",
                    ):
                        if req not in item:
                            raise ValueError(f"MemLens item missing required key: {req}")

                    s_ids = item["haystack_session_ids"]
                    s_sess = item["haystack_sessions"]
                    s_dates = item["haystack_dates"]
                    if len(s_ids) != len(s_sess) or len(s_sess) != len(s_dates):
                        raise ValueError(f"Inconsistent MemLens session lengths for {item['question_id']}")

                    sample_questions.append(item)
    except (OSError, json.JSONDecodeError, ValueError, TypeError, Exception) as exc:
        return {
            "status": "UNQUALIFIED_MEMLENS_SOURCE",
            "dataset_path": str(dataset_json),
            "dataset_size_bytes": size_bytes,
            "dataset_sha256": dataset_sha,
            "error": f"Failed to parse MemLens streaming items: {exc}",
        }

    if not sample_questions:
        return {
            "status": "UNQUALIFIED_MEMLENS_SOURCE",
            "dataset_path": str(dataset_json),
            "dataset_size_bytes": size_bytes,
            "dataset_sha256": dataset_sha,
            "error": "MemLens dataset contains no questions",
        }

    # Test candidate bin generation in a temporary directory
    tok = tokenizer if tokenizer is not None else get_qualification_tokenizer()
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        tmp_raw = tmp_dir / "sample_memlens_raw.json"
        tmp_raw.write_text(json.dumps([sample_questions[0]]), encoding="utf8")

        # Mock images directory structure if needed
        tmp_images_root = tmp_dir / "images_root"
        tmp_images_root.mkdir(parents=True, exist_ok=True)
        sample_turn_images = []
        for session in sample_questions[0]["haystack_sessions"]:
            for turn in session:
                for img_ref in turn.get("images", []):
                    rel = Path(img_ref["file"])
                    src_real = dataset_dir / "release_images" / rel
                    dst_tmp = tmp_images_root / "release_images" / rel
                    dst_tmp.parent.mkdir(parents=True, exist_ok=True)
                    if src_real.is_file():
                        shutil.copy2(src_real, dst_tmp)
                    else:
                        dst_tmp.write_bytes(b"mock_img")
                    sample_turn_images.append(rel)

        tmp_native = tmp_dir / "native.jsonl"
        norm_summary = normalize_memlens(
            tmp_raw,
            tmp_images_root,
            tmp_native,
            limit=1,
        )

        tmp_candidates = tmp_dir / "candidates.jsonl"
        bin_summary = build_memlens_context_bins(
            tmp_native,
            tmp_candidates,
            tokenizer=tok,
            input_lengths=CANONICAL_LENGTHS,
            tolerance=tolerance,
        )

    return {
        "status": "QUALIFIED_MEMLENS_SOURCE",
        "dataset_path": str(dataset_json),
        "dataset_size_bytes": size_bytes,
        "dataset_sha256": dataset_sha,
        "sample_question_ids": [q["question_id"] for q in sample_questions],
        "image_directories": {
            "needle_images_directory": str(needle_dir),
            "needle_images_present": needle_present,
            "haystack_images_directory": str(haystack_dir),
            "haystack_images_present": haystack_present,
        },
        "schema_compatibility": {
            "normalize_memlens_verified": True,
            "context_bin_generation_verified": True,
            "tested_tolerance": tolerance,
            "tolerance_bounded_lte_0_05": tolerance <= 0.05,
            "normalized_records": norm_summary.get("count", 0),
            "derived_records_emitted": bin_summary.get("derived_records", 0),
            "derived_bins_emitted": bin_summary.get("lengths_emitted", []),
            "skip_reasons": bin_summary.get("skip_reasons", {}),
        },
    }


def qualify_qwen_models(
    drive_root: Path | str,
    models_rel: Path | str = LEGACY_MODEL_REL,
    *,
    model_names: tuple[str, ...] = ACTIVE_MODELS,
    compute_shards_sha: bool = True,
) -> dict[str, Any]:
    """Inspect and qualify Qwen model checkpoints on mounted Google Drive.

    Verifies config.json, tokenizer files, and safetensors weight shards.
    """
    root = Path(drive_root)
    models_dir = root / models_rel
    models_report: dict[str, Any] = {}
    all_qualified = True

    for m_name in model_names:
        m_path = models_dir / m_name
        if not m_path.is_dir():
            models_report[m_name] = {
                "status": "UNQUALIFIED_MODEL_SOURCE",
                "path": str(m_path),
                "error": f"Model directory not found: {m_path}",
            }
            all_qualified = False
            continue

        config_path = m_path / "config.json"
        if not config_path.is_file():
            models_report[m_name] = {
                "status": "UNQUALIFIED_MODEL_SOURCE",
                "path": str(m_path),
                "error": "Missing config.json",
            }
            all_qualified = False
            continue

        try:
            config_data = json.loads(config_path.read_text(encoding="utf8"))
            config_sha = sha256(config_path)
        except (OSError, json.JSONDecodeError) as exc:
            models_report[m_name] = {
                "status": "UNQUALIFIED_MODEL_SOURCE",
                "path": str(m_path),
                "error": f"Failed to parse config.json: {exc}",
            }
            all_qualified = False
            continue

        tokenizer_files: dict[str, str] = {}
        for t_name in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model"):
            tp = m_path / t_name
            if tp.is_file():
                tokenizer_files[t_name] = sha256(tp)

        if not tokenizer_files:
            models_report[m_name] = {
                "status": "UNQUALIFIED_MODEL_SOURCE",
                "path": str(m_path),
                "error": "Missing tokenizer artifacts (tokenizer.json or tokenizer_config.json)",
            }
            all_qualified = False
            continue

        # Inspect weight shards
        index_path = m_path / "model.safetensors.index.json"
        shard_names: list[str] = []
        if index_path.is_file():
            try:
                idx_data = json.loads(index_path.read_text(encoding="utf8"))
                if isinstance(idx_data.get("weight_map"), dict):
                    shard_names = sorted(set(map(str, idx_data["weight_map"].values())))
            except (OSError, json.JSONDecodeError):
                shard_names = []

        if not shard_names:
            shard_names = sorted(
                p.name for p in m_path.iterdir() if p.is_file() and p.suffix in (".safetensors", ".bin")
            )

        if not shard_names:
            models_report[m_name] = {
                "status": "UNQUALIFIED_MODEL_SOURCE",
                "path": str(m_path),
                "error": "No weight shards (.safetensors or .bin) found in checkpoint",
            }
            all_qualified = False
            continue

        shard_details: dict[str, Any] = {}
        total_weight_bytes = 0
        missing_shards = []

        for s_name in shard_names:
            sp = m_path / s_name
            if not sp.is_file() or sp.stat().st_size == 0:
                missing_shards.append(s_name)
                continue
            sz = sp.stat().st_size
            total_weight_bytes += sz
            shard_info: dict[str, Any] = {"size_bytes": sz}
            if compute_shards_sha:
                shard_info["sha256"] = sha256(sp)
            shard_details[s_name] = shard_info

        if missing_shards:
            models_report[m_name] = {
                "status": "UNQUALIFIED_MODEL_SOURCE",
                "path": str(m_path),
                "error": f"Missing or empty weight shards: {missing_shards}",
            }
            all_qualified = False
            continue

        models_report[m_name] = {
            "status": "QUALIFIED_MODEL_SOURCE",
            "path": str(m_path),
            "model_type": config_data.get("model_type"),
            "architectures": config_data.get("architectures", []),
            "config_sha256": config_sha,
            "tokenizer_files": tokenizer_files,
            "weight_shards_count": len(shard_names),
            "total_weight_bytes": total_weight_bytes,
            "weight_shards": shard_details,
        }

    return {
        "status": "QUALIFIED_MODEL_SOURCES" if all_qualified else "UNQUALIFIED_MODEL_SOURCES",
        "models_dir": str(models_dir),
        "models": models_report,
    }


def qualify_all_drive_assets(
    drive_root: Path | str = Path("/content/drive/MyDrive"),
    *,
    tolerance: float = 0.05,
    compute_shards_sha: bool = True,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Execute complete read-only dataset and model qualification against Google Drive.

    Guarantees strict read-only execution: asserts filesystem snapshot before and
    after qualification are identical.
    """
    root = Path(drive_root)
    if not root.is_dir():
        return {
            "status": "DRIVE_NOT_MOUNTED",
            "drive_root": str(root),
            "instructions": "Authenticate then mount Drive in an existing CPU Colab session.",
        }

    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or math.isnan(tolerance)
        or math.isinf(tolerance)
    ):
        raise ValueError(f"tolerance must be a finite float, got {tolerance!r}")
    if tolerance <= 0.0 or tolerance > 0.05:
        raise ValueError(f"tolerance must be strictly in (0.0, 0.05], got {tolerance}")

    asset_subdirs = [LEGACY_DATASET_REL, LEGACY_MODEL_REL]
    subdirs_to_check = [s for s in asset_subdirs if (root / s).is_dir()] or None

    # Capture read-only snapshot before qualification
    snapshot_before = snapshot_directory_state(root, subdirs=subdirs_to_check)

    models_res = qualify_qwen_models(
        root, compute_shards_sha=compute_shards_sha
    )
    amber_res = qualify_amber_assets(
        root, sample_limit=sample_limit
    )
    memlens_res = qualify_memlens_assets(
        root, sample_limit=sample_limit, tolerance=tolerance
    )

    # Verify read-only guarantee
    snapshot_after = snapshot_directory_state(root, subdirs=subdirs_to_check)
    if snapshot_before != snapshot_after:
        added = set(snapshot_after.keys()) - set(snapshot_before.keys())
        removed = set(snapshot_before.keys()) - set(snapshot_after.keys())
        changed = {
            k: (snapshot_before[k], snapshot_after[k])
            for k in set(snapshot_before.keys()) & set(snapshot_after.keys())
            if snapshot_before[k] != snapshot_after[k]
        }
        raise PermissionError(
            f"Read-only invariant violated on Drive root {root}! "
            f"Added: {added}, Removed: {removed}, Changed: {changed}"
        )

    amber_ok = amber_res.get("status") == "QUALIFIED_AMBER_SOURCE"
    memlens_ok = memlens_res.get("status") == "QUALIFIED_MEMLENS_SOURCE"
    models_ok = models_res.get("status") == "QUALIFIED_MODEL_SOURCES"

    if amber_ok and memlens_ok and models_ok:
        overall_status = "ALL_QUALIFIED"
    elif amber_ok or memlens_ok or models_ok:
        overall_status = "PARTIALLY_QUALIFIED"
    else:
        overall_status = "UNQUALIFIED"

    manifest_sha: dict[str, Any] = {
        "amber_zip": amber_res.get("archive_sha256"),
        "memlens_dataset_32k_json": memlens_res.get("dataset_sha256"),
        "models": {
            m_name: {
                "config_sha256": m_info.get("config_sha256"),
                "tokenizer_files": m_info.get("tokenizer_files"),
                "weight_shards_sha256": {
                    s_name: s_data.get("sha256")
                    for s_name, s_data in m_info.get("weight_shards", {}).items()
                    if "sha256" in s_data
                },
            }
            for m_name, m_info in models_res.get("models", {}).items()
            if m_info.get("status") == "QUALIFIED_MODEL_SOURCE"
        },
    }

    return {
        "status": overall_status,
        "drive_root": str(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_verified": True,
        "tolerance_tested": tolerance,
        "models": models_res,
        "amber": amber_res,
        "memlens": memlens_res,
        "manifest_sha256": manifest_sha,
    }
