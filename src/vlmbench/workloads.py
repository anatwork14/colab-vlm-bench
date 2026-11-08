"""JSONL input contracts, media-path security, token-bin verification, paired sampling."""
from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import random
from pathlib import Path


class WorkloadError(ValueError):
    pass


def load_jsonl(path: str | Path, *, dataset: str, track: str) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    results, seen = [], set()
    for line_no, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get("id"), (str, int)):
            raise WorkloadError(f"Missing sample id at line {line_no}")
        if row["dataset"] != dataset or row["track"] != track:
            raise WorkloadError(f"Incorrect dataset/track at line {line_no}")
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise WorkloadError(f"Missing chat messages at line {line_no}")
        if row["id"] in seen:
            raise WorkloadError(f"Duplicate question id at line {line_no}")
        seen.add(row["id"])
        if track == "native_quality" and not row.get("native_context_complete", False):
            raise WorkloadError("Native quality input missing complete context declaration")
        if track == "derived_serving" and row.get("derived_from") is None:
            raise WorkloadError("Derived serving input must record origin")
        results.append(row)
    if not results:
        raise WorkloadError("Empty workload")
    return results


def exact_bin_pool(rows: list[dict], model_id: str, target_length: int, tolerance: float,
                   max_effective_total: int | None = None) -> list[dict]:
    good = []
    for row in rows:
        counts = row.get("model_token_counts", {}).get(model_id)
        if not counts or counts.get("verified_by") not in (
            "model_processor", "vllm_effective_token_accounting"
        ):
            continue
        total = counts.get("effective_total_tokens")
        text_length = counts.get("text_tokens")
        if not isinstance(total, int) or total <= 0:
            continue
        if not isinstance(text_length, int) or text_length <= 0:
            continue
        if max_effective_total is not None and total >= max_effective_total:
            continue
        if abs(text_length - target_length) / target_length <= tolerance:
            good.append(row)
    return good


def sample_unique(rows: list[dict], count: int, seed: int) -> list[dict]:
    if len(rows) < count:
        raise WorkloadError(f"INSUFFICIENT_VERIFIED_UNIQUE_PROMPTS {len(rows)} < {count}")
    rng = random.Random(seed)
    # ids are stable across the paired base and MTP runs
    return rng.sample(sorted(rows, key=lambda r: str(r["id"])), count)


def sample_digest(rows: list[dict]) -> str:
    ids = [str(r["id"]) for r in rows]
    return hashlib.sha256(json.dumps(ids, ensure_ascii=False).encode()).hexdigest()


def resolve_messages(messages: list[dict]) -> list[dict]:
    """Read local real images as data URI; never silently omit images, never use gold."""
    converted = []
    for msg in messages:
        content = msg["content"]
        if not isinstance(content, list):
            converted.append({"role": msg["role"], "content": content})
            continue
        items = []
        for part in content:
            if part.get("type") == "image_path":
                path, root = Path(part["path"]).resolve(), Path(part["asset_root"]).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise WorkloadError(f"Unsafe / missing image: {path}")
                if path.is_symlink() or path.stat().st_size > 25 * 1024 * 1024:
                    raise WorkloadError("Image symlink or >25MiB blocked")
                mime = mimetypes.guess_type(str(path))[0]
                if mime not in ("image/png", "image/jpeg", "image/webp"):
                    raise WorkloadError(f"Unsupported image content type: {mime}")
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                items.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
            else:
                if part.get("type") not in ("text", "image_url"):
                    raise WorkloadError("Unsupported multimodal part")
                items.append(part)
        converted.append({"role": msg["role"], "content": items})
    return converted


def ensure_no_gold_in_request(sample: dict, serialized_request: dict) -> None:
    # This is a structural rather than semantic check: reference is stored outside messages.
    if "reference" in serialized_request or "answer" in serialized_request:
        raise WorkloadError("Evaluator-only gold leaked into request")
    if not sample.get("messages"):
        raise WorkloadError("No model input")
