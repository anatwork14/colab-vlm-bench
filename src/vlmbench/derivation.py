"""Deterministic source-derived serving bins and real model-specific token annotation.

This is an engineering workload (not an official quality subset). Source content is
not replaced by random tokens, but evidence may be omitted by the derivation.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx

from .artifacts import atomic_json, atomic_jsonl, sha256
from .config import LENGTHS
from .execution_safety import require_compute_approval
from .workloads import load_jsonl
from .bench import request_payload


def generate_candidates(native_path: Path, output: Path, dataset: str,
                        image_limit: int = 1) -> dict:
    if not 0 <= image_limit <= 8:
        raise ValueError("image_limit must be between 0 and 8")
    native = load_jsonl(native_path, dataset=dataset, track="native_quality")
    result = []
    for row in native:
        # The adapters emit the input as one multimodal user message, preserving
        # chronologically ordered session/page text and images.
        if len(row["messages"]) != 1:
            raise ValueError("Normalize native examples to one user message before deriving")
        content = row["messages"][0]["content"]
        if not isinstance(content, list) or not content or content[-1].get("type") != "text":
            raise ValueError("Last multimodal item must be final question text")
        question_part = content[-1]
        history = content[:-1]
        selected_images = [
            part for part in history if part["type"] == "image_path"
        ][-image_limit:] if image_limit else []
        for target in LENGTHS:
            # Approximate initial target from whitespace tokens. Independent
            # vLLM/processor annotation is REQUIRED before this cell is eligible.
            word_target = max(1, int(target * 0.72))
            question_words = str(question_part["text"]).split()
            remaining = max(0, word_target - len(question_words))
            historical_text = " ".join(
                str(part["text"]) for part in history if part["type"] == "text"
            ).split()
            tail = historical_text[-remaining:] if remaining > 0 else []
            parts = []
            if tail:
                parts.append({"type": "text", "text": " ".join(tail)})
            parts.extend(selected_images)
            parts.append(question_part)
            result.append({
                "id": f"{row['id']}__text{target}",
                "dataset": dataset, "track": "derived_serving",
                "derived_from": str(row["id"]),
                "messages": [{"role": "user", "content": parts}],
                "requested_text_length_bin": target,
                "derivation": "last_text_words_plus_last_k_images_v1",
                "image_count": len(selected_images),
                "model_token_counts": {},
                "meta": {"original_native_id": str(row["id"]),
                         "evidence_may_be_dropped": True,
                         "official_quality_allowed": False},
            })
    atomic_jsonl(output, result)
    return {"source_sha256": sha256(native_path), "derived_sha256": sha256(output),
            "source_records": len(native), "derived_records": len(result),
            "status": "CANDIDATES_REQUIRE_MODEL_TOKEN_ANNOTATION", "output": str(output)}


def annotate_with_backend(
    source: Path, output: Path, dataset: str,
    model_id: str, checkpoint_path: str, base_url: str, limit: int = 0,
    *, owner_approved_compute: bool = False,
) -> dict:
    """Prompt-token usage from real vLLM + model tokenizer counts of text segments.

    This may execute many **one-token GPU inference requests**, and requires
    explicit owner approval and an already-booted matching Base model. Do not
    perform automatically during a paid full benchmark.
    """
    require_compute_approval(owner_approved_compute)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to annotate model text tokens") from exc

    track = "amber_serving" if dataset == "amber" else "derived_serving"
    rows = load_jsonl(source, dataset=dataset, track=track)
    if limit < 0:
        raise ValueError("Annotation budget must be >= 0")
    source_sha = sha256(source)
    state = output.with_name(output.name + ".state.json")
    if output.exists():
        if not state.is_file():
            raise ValueError("Existing annotated file lacks source-digest state; audit/rename before restart")
        previous_state = json.loads(state.read_text(encoding="utf8"))
        if previous_state.get("source_sha256") != source_sha or previous_state.get("model") != model_id:
            raise ValueError("Annotation source/model changed; use distinct output path")
        previous = load_jsonl(output, dataset=dataset, track=track)
        if [x["id"] for x in rows] != [x["id"] for x in previous]:
            raise ValueError("Annotation resume IDs differ from original candidates")
        rows = previous
    else:
        atomic_json(state, {"source_sha256": source_sha, "model": model_id,
                            "checkpoint": checkpoint_path, "status": "PARTIAL_ANNOTATION"})
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, local_files_only=True, trust_remote_code=False
    )
    produced = []
    successes = 0
    attempted = 0
    carried = sum(x.get("model_token_counts", {}).get(model_id, {}).get("verified_by")
                  == "vllm_effective_token_accounting" for x in rows)
    url = base_url.rstrip("/") + "/v1/chat/completions"
    with httpx.Client(timeout=httpx.Timeout(360.0, connect=30.0)) as client:
        for row in rows:
            if limit and attempted >= limit:
                produced.append(row)
                continue
            previous = row.get("model_token_counts", {}).get(model_id)
            if previous and previous.get("verified_by") == "vllm_effective_token_accounting":
                produced.append(row)
                continue
            copied = dict(row)
            copied["model_token_counts"] = dict(row.get("model_token_counts", {}))
            text_parts = []
            for message in row["messages"]:
                if isinstance(message["content"], str):
                    text_parts.append(message["content"])
                else:
                    text_parts.extend(
                        str(part["text"]) for part in message["content"] if part["type"] == "text"
                    )
            text_token_count = len(tokenizer.encode(
                " ".join(text_parts), add_special_tokens=False
            ))
            try:
                attempted += 1
                request = request_payload(
                    row, model_id=model_id, max_tokens=1, streaming=False, fixed_output=False)
                response = client.post(url, json=request)
                response.raise_for_status()
                usage = response.json().get("usage", {})
                effective = usage.get("prompt_tokens")
                if not isinstance(effective, int) or effective <= 0:
                    raise ValueError("No numeric usage.prompt_tokens from backend")
                copied["model_token_counts"][model_id] = {
                    "verified_by": "vllm_effective_token_accounting",
                    "text_tokens": text_token_count,
                    "effective_total_tokens": effective,
                    "image_count": row.get("image_count", 0),
                    "text_count_method": "model_tokenizer_on_joined_text_segments",
                    "note": "Text count excludes some chat-template overhead; effective_total is actual backend usage",
                    "checkpoint_path": checkpoint_path,
                }
                successes += 1
            except (httpx.HTTPError, ValueError, OSError) as exc:
                copied.setdefault("annotation_errors", {})[model_id] = (
                    f"{type(exc).__name__}: {exc}"
                )
            produced.append(copied)
            # Incrementally persist to local/Drive staging without overwriting source.
            if len(produced) % 10 == 0:
                atomic_jsonl(output, produced + rows[len(produced):])
    atomic_jsonl(output, produced)
    verified_total = sum(x.get("model_token_counts", {}).get(model_id, {}).get("verified_by")
                         == "vllm_effective_token_accounting" for x in produced)
    state_body = {"source_sha256": source_sha, "model": model_id,
                  "checkpoint": checkpoint_path,
                  "status": "ANNOTATED" if verified_total == len(rows) else "PARTIAL_ANNOTATED",
                  "verified_total": verified_total, "total_rows": len(rows)}
    atomic_json(state, state_body)
    return {
        "status": state_body["status"],
        "source_sha256": source_sha, "output_sha256": sha256(output),
        "model": model_id, "verified_new": successes,
        "verified_before": carried, "verified_total": verified_total,
        "http_requests_this_invocation": attempted,
        "remaining_unverified": len(rows) - verified_total,
        "total_rows": len(rows), "output": str(output),
    }
