"""Pinned Hugging Face model/dataset intake for agent-operated Colab sessions.

Network and file writes happen ONLY when fetch_model/fetch_dataset are invoked.
Fetch on a CPU Colab runtime into Drive; stage to SSD for A100 later.
"""
from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path

from .artifacts import atomic_json, sha256
from .preflight import checkpoint_inventory, dataset_probe_memlens, dataset_probe_mmlong

MODELS = {"qwen35_08b": "Qwen/Qwen3.5-0.8B", "qwen35_4b": "Qwen/Qwen3.5-4B"}
DATASETS = {"memlens": "xiyuRenBill/MEMLENS",
            "mmlongbench_doc": "yubo2333/MMLongBench-Doc"}
MAX_UNPACK_BYTES = 2 * 1024**3
MAX_IMAGE_FILES = 12000


def _hub():
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub on the CPU session first") from exc
    return HfApi(), snapshot_download


def _resolve_pinned(repo_id: str, *, repo_type: str, root: Path,
                    requested_revision: str | None) -> str:
    """Pin the first resolved Hub commit, forbid silent drift across sessions."""
    receipt = root / ".hf_source.json"
    prior = json.loads(receipt.read_text()) if receipt.is_file() else None
    intent = root / ".hf_download_intent.json"
    if not prior and intent.is_file():
        prior = json.loads(intent.read_text(encoding="utf8"))
    if prior and (prior.get("repo_id") != repo_id or
                  prior.get("repo_type") != repo_type):
        raise ValueError(f"Existing HF destination belongs to another repository: {root}")
    revision = requested_revision or (prior or {}).get("revision")
    api, _ = _hub()
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type,
                         revision=revision)
    resolved = info.sha
    if not resolved or len(resolved) != 40:
        raise RuntimeError(f"Hub did not resolve immutable commit for {repo_id}")
    if prior and prior.get("revision") != resolved:
        raise RuntimeError("HF revision mismatch: use a NEW source directory and campaign")
    atomic_json(root / ".hf_download_intent.json",
                {"repo_id": repo_id, "repo_type": repo_type, "revision": resolved})
    return resolved


def _record(root: Path, repo_id: str, repo_type: str,
            revision: str, inventory: dict) -> dict:
    receipt = {
        "status": "HF_PINNED_SOURCE_READY",
        "repo_id": repo_id, "repo_type": repo_type,
        "revision": revision, "inventory": inventory,
        "source": "huggingface_hub.snapshot_download",
        "note": "Backend compatibility and publisher scorer parity NOT qualified",
    }
    atomic_json(root / ".hf_source.json", receipt)
    return receipt


def _check_space(dest: Path, reserve_gib: int = 5) -> None:
    """Guard actual local filesystem; Drive FUSE quota still requires external check."""
    dest.mkdir(parents=True, exist_ok=True)
    if not str(dest.resolve()).startswith("/content/drive/"):
        # Local CPU dataset scratch can fill VM; keep explicit reserve.
        free = shutil.disk_usage(dest).free
        if free < reserve_gib * 1024**3:
            raise RuntimeError(f"Insufficient disk free before HF intake at {dest}")


def safe_extract_memlens_images(archive: Path, destination: Path) -> dict:
    """Extract the publisher's tarball with no traversal/links/devices/bombs."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    count, uncompressed = 0, 0
    with tarfile.open(archive, "r:gz") as tar:
        for entry in tar:
            name = Path(entry.name)
            if name.is_absolute() or ".." in name.parts:
                raise ValueError(f"Unsafe MemLens archive entry: {entry.name}")
            if entry.isdir():
                continue
            if not entry.isfile():
                raise ValueError("Refuse links/devices/special entries in MemLens archive")
            count += 1
            uncompressed += entry.size
            if count > MAX_IMAGE_FILES or uncompressed > MAX_UNPACK_BYTES:
                raise ValueError("MemLens archive exceeds bounded count/byte budget")
            if entry.size < 0 or entry.size > 64 * 1024**2:
                raise ValueError("MemLens single image exceeds safety budget")
            rel = name
            if rel.parts[0] in ("needle_images", "haystack_images"):
                rel = Path("release_images") / rel
            elif rel.parts[0] != "release_images":
                raise ValueError(f"Unexpected MemLens archive layout: {entry.name}")
            out = (destination / rel).resolve()
            if not out.is_relative_to(destination):
                raise ValueError("Unsafe extraction path")
            out.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(entry) as src:
                if src is None:
                    raise ValueError("Archive regular file unexpectedly unreadable")
                with out.open("wb") as f:
                    shutil.copyfileobj(src, f, length=1024 * 1024)
            if out.stat().st_size != entry.size:
                raise IOError(f"Incomplete extracted image: {out.name}")
    return {"images_extracted": count, "uncompressed_bytes": uncompressed}


def fetch_model(model_id: str, root: Path, *, revision: str | None = None) -> dict:
    if model_id not in MODELS:
        raise ValueError(f"Unapproved benchmark model: {model_id}")
    dest = Path(root) / "hf" / "models" / model_id
    _check_space(dest)
    repo_id = MODELS[model_id]
    pinned = _resolve_pinned(repo_id, repo_type="model", root=dest,
                             requested_revision=revision)
    _, download = _hub()
    download(repo_id=repo_id, repo_type="model", revision=pinned,
             local_dir=str(dest), max_workers=4)
    inventory = checkpoint_inventory(dest)
    return {**_record(dest, repo_id, "model", pinned, inventory),
            "path": str(dest), "model": model_id}


def _qa_file(root: Path) -> Path:
    matches = sorted((root / "data").rglob("*.parquet"))
    if not matches:
        matches = sorted((root / "data").rglob("*.json"))
    if not matches:
        raise FileNotFoundError("MMLongBench Hub download lacks QA parquet/json under data/")
    preferred = [x for x in matches if "train" in x.stem.lower()]
    return (preferred or matches)[0]


def fetch_dataset(dataset: str, root: Path, *,
                  revision: str | None = None) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    dest = Path(root) / "hf" / "datasets" / dataset
    _check_space(dest)
    repo_id = DATASETS[dataset]
    pinned = _resolve_pinned(repo_id, repo_type="dataset", root=dest,
                             requested_revision=revision)
    _, download = _hub()
    if dataset == "memlens":
        patterns = ["dataset_32k.json", "release_images.tar.gz", "LICENSE-DATA"]
    else:
        patterns = ["data/**", "documents/*.pdf", "README.md"]
    download(repo_id=repo_id, repo_type="dataset", revision=pinned,
             local_dir=str(dest), allow_patterns=patterns, max_workers=4)
    if dataset == "memlens":
        archive = dest / "release_images.tar.gz"
        marker = dest / ".image_unpack_complete.json"
        if not archive.is_file():
            raise FileNotFoundError("HF MemLens release_images.tar.gz missing")
        archive_sha = sha256(archive)
        if marker.is_file() and json.loads(marker.read_text()).get("archive_sha256") == archive_sha:
            unpack = {"status": "IMAGES_VERIFIED_PREVIOUS_EXTRACT"}
        else:
            unpack = safe_extract_memlens_images(archive, dest)
            atomic_json(marker, {**unpack, "archive_sha256": archive_sha})
        details = dataset_probe_memlens(dest / "dataset_32k.json", dest)
        details["image_unpack"] = unpack
        details["dataset_json_sha256"] = sha256(dest / "dataset_32k.json")
    else:
        qa = _qa_file(dest)
        details = dataset_probe_mmlong(qa, dest / "documents")
        details["qa_path"] = str(qa)
        details["qa_sha256"] = sha256(qa)
    return {**_record(dest, repo_id, "dataset", pinned, details),
            "path": str(dest), "dataset": dataset}
