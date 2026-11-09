"""MemLens dataset adapter and 7-context-bin builder.

Normalizes raw MemLens conversation sessions to single-user multimodal format
and deterministically generates 7-context-bin serving workloads.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ..artifacts import atomic_jsonl, sha256
from ..config import LENGTHS
from ..workloads import load_jsonl

CANONICAL_LENGTHS: tuple[int, ...] = LENGTHS  # (128, 512, 1024, 2048, 4096, 8192, 16384)


def normalize_memlens(
    source: Path | str,
    images_root: Path | str,
    output: Path | str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Stream MemLens JSON question-by-question, without copying the whole file.

    Peak RAM is bounded by one 32K item and its normalized message list.
    """
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            "MemLens streaming normalization requires 'ijson'. "
            "Install with pip install 'ijson>=3.3,<4'."
        ) from exc

    source = Path(source)
    images_root = Path(images_root).resolve()
    output = Path(output)
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive or None")

    seen: set[str] = set()
    count = 0
    image_count_total = 0

    def records():
        nonlocal count, image_count_total
        with source.open("rb") as fp:
            for item in ijson.items(fp, "item"):
                if not isinstance(item, dict):
                    raise ValueError("MemLens expects a top-level JSON list of objects")
                qid = str(item["question_id"])
                if qid in seen:
                    raise ValueError(f"Duplicate MemLens question id: {qid}")
                seen.add(qid)
                session_ids = item["haystack_session_ids"]
                sessions = item["haystack_sessions"]
                dates = item["haystack_dates"]
                if len(session_ids) != len(sessions) or len(sessions) != len(dates):
                    raise ValueError(f"Inconsistent MemLens session metadata: {qid}")

                parts = []
                image_count = 0
                for sid, date, session in zip(session_ids, dates, sessions):
                    parts.append({"type": "text", "text": f"\n[Session {sid} — {date}]\n"})
                    for turn in session:
                        raw_role = str(turn["role"]).strip().lower()
                        if "assistant" in raw_role:
                            role = "assistant"
                        elif "user" in raw_role:
                            role = "user"
                        else:
                            raise ValueError(f"Unknown MemLens role {turn['role']} in {qid}")
                        parts.append({
                            "type": "text",
                            "text": f"[{role}] {turn.get('content', '')}\n",
                        })
                        for image_ref in turn.get("images", []):
                            relative = Path(image_ref["file"])
                            path = (images_root / "release_images" / relative).resolve()
                            if not path.is_relative_to(images_root) or not path.is_file():
                                raise ValueError(f"Missing or escaping MemLens image: {relative}")
                            parts.append({
                                "type": "image_path",
                                "path": str(path),
                                "asset_root": str(images_root),
                            })
                            image_count += 1

                parts.append({"type": "text", "text": f"\nQuestion: {item['question']}"})
                record = {
                    "id": qid,
                    "dataset": "memlens",
                    "track": "native_quality",
                    "messages": [{"role": "user", "content": parts}],
                    "reference": str(item["answer"]),
                    "native_context_complete": True,
                    "meta": {
                        "question_type": item["question_type"],
                        "question_date": item.get("question_date"),
                        "session_ids": session_ids,
                        "evidence_session_ids": item.get("answer_session_ids", []),
                        "image_count": image_count,
                        "adapter_provenance": (
                            "colab-vlm-bench-memlens-v2-streaming; "
                            "official_prompt_parity_pending"
                        ),
                    },
                }
                count += 1
                image_count_total += image_count
                yield record
                if limit is not None and count >= limit:
                    break

    atomic_jsonl(output, records())
    return {
        "dataset": "memlens",
        "count": count,
        "image_references": image_count_total,
        "sha256_source": sha256(source),
        "sha256_normalized": sha256(output),
        "output": str(output),
        "status": "NORMALIZED_NEEDS_OFFICIAL_PROMPT_PARITY",
        "memory_strategy": "ijson_stream_one_question_and_atomic_jsonl",
    }


def build_memlens_context_bins(
    native_path: Path | str,
    output_path: Path | str,
    *,
    input_lengths: tuple[int, ...] | list[int] = CANONICAL_LENGTHS,
    image_limit: int = 1,
    tokenizer: Any | None = None,
    tolerance: float = 0.05,
    max_per_bin: int | None = None,
) -> dict[str, Any]:
    """Generate deterministic serving candidates across the 7 canonical context bins.

    Canonical lengths: (128, 512, 1024, 2048, 4096, 8192, 16384).
    Deterministic source IDs: <source_id>__text<length>.
    Preserves genuine visual inputs and original source question IDs.
    """
    native_path = Path(native_path)
    output_path = Path(output_path)
    if not 0 <= image_limit <= 8:
        raise ValueError("image_limit must be between 0 and 8")

    canonical_targets = tuple(int(x) for x in input_lengths)
    rows = load_jsonl(native_path, dataset="memlens", track="native_quality")

    candidates: list[dict[str, Any]] = []
    bin_counts: Counter[int] = Counter()

    for row in rows:
        if len(row["messages"]) != 1:
            raise ValueError("Normalize native examples to one user message before deriving")
        content = row["messages"][0]["content"]
        if not isinstance(content, list) or not content or content[-1].get("type") != "text":
            raise ValueError("Last multimodal item must be final question text")

        question_part = content[-1]
        history_parts = content[:-1]

        # Preserve genuine visual inputs exactly from original sessions
        selected_images = [
            part for part in history_parts if part.get("type") == "image_path"
        ][-image_limit:] if image_limit else []

        history_texts = [
            str(part["text"]) for part in history_parts if part.get("type") == "text"
        ]
        history_full_text = " ".join(history_texts)
        question_text = str(question_part["text"])
        source_id = str(row["id"])

        for target in canonical_targets:
            if max_per_bin is not None and bin_counts[target] >= max_per_bin:
                continue

            # Deterministic ID formatted exactly as <source_id>__text<length>
            derived_id = f"{source_id}__text{target}"

            if tokenizer is not None:
                q_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                hist_tokens = tokenizer.encode(history_full_text, add_special_tokens=False)
                budget = target - len(q_tokens) - 1
                if budget <= 0:
                    trimmed_text = ""
                else:
                    lo, hi = max(1, budget // 2), min(len(hist_tokens), int(budget * 1.3) + 16)
                    best = None
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        excerpt = tokenizer.decode(
                            hist_tokens[-mid:], skip_special_tokens=False
                        )
                        merged = excerpt + "\n" + question_text
                        actual = len(tokenizer.encode(merged, add_special_tokens=False))
                        if best is None or abs(actual - target) < abs(best[0] - target):
                            best = (actual, excerpt)
                        if actual < target:
                            lo = mid + 1
                        elif actual > target:
                            hi = mid - 1
                        else:
                            break
                    trimmed_text = best[1] if best else ""
            else:
                # Word-based heuristic when tokenizer not present on CPU
                word_target = max(1, int(target * 0.72))
                q_words = question_text.split()
                rem = max(0, word_target - len(q_words))
                hist_words = history_full_text.split()
                tail = hist_words[-rem:] if rem > 0 else []
                trimmed_text = " ".join(tail) if tail else ""

            parts = []
            if trimmed_text:
                parts.append({"type": "text", "text": trimmed_text})
            parts.extend(selected_images)
            parts.append(question_part)

            candidate = {
                "id": derived_id,
                "dataset": "memlens",
                "track": "derived_serving",
                "derived_from": source_id,
                "messages": [{"role": "user", "content": parts}],
                "requested_text_length_bin": target,
                "derivation": "memlens_7bin_builder_v2",
                "image_count": len(selected_images),
                "model_token_counts": {},
                "meta": {
                    "original_question_id": source_id,
                    "requested_text_length_bin": target,
                    "evidence_may_be_dropped": True,
                    "official_quality_allowed": False,
                    "image_count": len(selected_images),
                },
            }
            candidates.append(candidate)
            bin_counts[target] += 1

    atomic_jsonl(output_path, candidates)
    return {
        "dataset": "memlens",
        "track": "derived_serving",
        "canonical_lengths": list(canonical_targets),
        "source_sha256": sha256(native_path),
        "derived_sha256": sha256(output_path),
        "source_records": len(rows),
        "derived_records": len(candidates),
        "counts_per_bin": {str(k): bin_counts[k] for k in canonical_targets},
        "output": str(output_path),
        "status": "CANDIDATES_REQUIRE_MODEL_TOKEN_ANNOTATION",
    }
