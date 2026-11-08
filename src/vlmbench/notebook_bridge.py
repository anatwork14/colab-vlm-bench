"""Google Colab notebook glue for honest VLM-derived serving workloads.

The notebooks are independently uploadable: they clone this repository and import
this helper. Never score these shortened/derived contexts as official dataset QA.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from vlmbench.artifacts import atomic_jsonl, sha256
from vlmbench.workloads import load_jsonl


def _digest_id(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def build_serving_candidates(
    native_path: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    model_directory: str | Path,
    input_lengths: list[int],
    max_per_bin: int = 192,
    pdf_root: str | Path | None = None,
    tolerance: float = 0.05,
) -> dict:
    """One real-source-derived text+image question per eligible question/length bin.

    Uses actual checkpoint tokenizer to fit model-tokenized *text*, with vision
    payload accounted for later using vLLM usage.prompt_tokens. The produced
    rows intentionally remain UNQUALIFIED until backend annotation succeeds.
    """
    if dataset not in ("memlens", "mmlongbench_doc"):
        raise ValueError("Unsupported dataset")
    if max_per_bin < 160:
        raise ValueError("At least 160 unique candidates per bin needed for CCU=32")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_directory), local_files_only=True, trust_remote_code=False
    )
    native = load_jsonl(native_path, dataset=dataset, track="native_quality")
    native.sort(key=lambda row: _digest_id(row["id"]))
    inputs = []
    counts = Counter()
    reasons = Counter()
    pdf_cache = {}
    for example in native:
        if all(counts[n] >= max_per_bin for n in input_lengths):
            break
        if not isinstance(example["messages"][0]["content"], list):
            reasons["not_multimodal_list"] += 1
            continue
        content = example["messages"][0]["content"]
        if not content or content[-1].get("type") != "text":
            reasons["no_final_question"] += 1
            continue
        question = str(content[-1]["text"])
        if dataset == "memlens":
            history = "\n".join(
                str(part["text"]) for part in content[:-1] if part.get("type") == "text"
            )
        else:
            if pdf_root is None:
                raise ValueError("pdf_root required for source-derived MMLong PDF text")
            doc_id = str(example.get("meta", {}).get("doc_id", ""))
            root = Path(pdf_root).resolve()
            pdf = (root / doc_id).resolve()
            if not pdf.is_relative_to(root) or not pdf.is_file():
                reasons["missing_or_unsafe_pdf"] += 1
                continue
            if doc_id not in pdf_cache:
                import fitz
                with fitz.open(pdf) as document:
                    pdf_cache[doc_id] = "\n".join(p.get_text() for p in document)
            history = pdf_cache[doc_id]
        image = next(
            (part for part in content[:-1] if part.get("type") == "image_path"), None
        )
        if image is None:
            reasons["no_real_visual_evidence"] += 1
            continue
        token_history = tokenizer.encode(history, add_special_tokens=False)
        if not token_history:
            reasons["no_text_history"] += 1
            continue
        for length in input_lengths:
            if counts[length] >= max_per_bin:
                continue
            n_question = len(tokenizer.encode(question, add_special_tokens=False))
            budget = length - n_question - 1
            if budget <= 0 or len(token_history) < budget:
                reasons[f"history_too_short_for_{length}"] += 1
                continue
            # Binary search compensates for re-encoding and join boundaries.
            lo, hi = max(1, budget // 2), min(len(token_history), int(budget * 1.3) + 16)
            best = None
            while lo <= hi:
                mid = (lo + hi) // 2
                excerpt = tokenizer.decode(
                    token_history[-mid:], skip_special_tokens=False
                )
                merged = excerpt + "\n" + question
                actual = len(tokenizer.encode(merged, add_special_tokens=False))
                if best is None or abs(actual - length) < abs(best[0] - length):
                    best = (actual, excerpt)
                if actual < length:
                    lo = mid + 1
                elif actual > length:
                    hi = mid - 1
                else:
                    break
            if best is None or abs(best[0] - length) / length > tolerance:
                reasons[f"text_bin_mismatch_{length}"] += 1
                continue
            # Keep one actual image. Derived prompts are VLM requests but NOT
            # the original full-context question; this is not official scoring.
            record = {
                "id": f"{example['id']}__text{length}", "dataset": dataset,
                "track": "derived_serving", "derived_from": str(example["id"]),
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": best[1]},
                    dict(image),
                    {"type": "text", "text": question},
                ]}],
                "requested_text_length_bin": length,
                "image_count": 1,
                "model_token_counts": {},
                "derivation": "tokenizer_exact_tail_text_one_source_image_v1",
                "meta": {"native_context_complete": False,
                         "official_quality_allowed": False,
                         "text_tokens_tokenizer_candidate": best[0],
                         "visual_tokens": "REQUIRES_VLLM_ANNOTATION"},
            }
            inputs.append(record)
            counts[length] += 1
    atomic_jsonl(output_path, inputs)
    return {
        "status": "PREPARED_NEEDS_GPU_TOKEN_ANNOTATION",
        "dataset": dataset,
        "source_sha256": sha256(native_path),
        "output_sha256": sha256(output_path),
        "total_rows": len(inputs),
        "eligible_per_bin": {str(k): counts[k] for k in input_lengths},
        "skipped_reasons": dict(reasons),
        "output_path": str(output_path),
        "note": "Use annotate-serving against matching Base endpoint before any performance cell",
    }


def write_local_campaign(
    template: str | Path, output: str | Path, *,
    model_id: str, drive_model: str, drive_results: str,
    data_paths: dict[str, dict[str, str]],
    revision: str,
    campaign_suffix: str = "notebook",
) -> dict:
    from vlmbench.config import validate, plan_summary

    config = json.loads(Path(template).read_text(encoding="utf-8"))
    if model_id not in ("qwen35_08b", "qwen35_4b"):
        raise ValueError("Only 0.8B/4B prioritized in notebooks")
    if not str(drive_model).startswith("/content/drive/"):
        raise ValueError("Model weight source must be a mounted Drive folder")
    if not str(drive_results).startswith("/content/drive/"):
        raise ValueError("Results must live on mounted Drive")
    if revision in ("", "UNPINNED", "REPLACE") or not str(revision).strip():
        raise ValueError("Checkpoint revision must be pinned from source manifest")
    config["campaign_id"] += f"-{model_id}-{campaign_suffix}"
    config["execution_scope"]["model_ids"] = [model_id]
    config["dry_run"] = False
    config["storage"]["drive_results_dir"] = str(drive_results)
    config["quality"]["max_samples"] = min(int(config["quality"].get("max_samples", 0)), 48)
    for model in config["models"]:
        if model["id"] == model_id:
            model["source_drive_path"] = str(drive_model)
            model["revision"] = str(revision)
    for entry in config["datasets"]:
        paths = data_paths[entry["id"]]
        entry["native_path"] = str(paths["native"])
        entry["derived_path"] = str(paths["derived"])
    validate(config, execution=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return plan_summary(config)
