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
    # Index only byte offsets/IDs, never 789 full 32K native prompts in RAM.
    ordered_offsets: list[tuple[str, int]] = []
    seen_ids: set[str] = set()
    with Path(native_path).open("rb") as fp:
        while True:
            offset = fp.tell()
            line = fp.readline()
            if not line:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("dataset") != dataset or sample.get("track") != "native_quality":
                raise ValueError("Unexpected dataset or track in normalized JSONL")
            sid = str(sample["id"])
            if sid in seen_ids:
                raise ValueError(f"Duplicate normalized question ID: {sid}")
            seen_ids.add(sid)
            ordered_offsets.append((_digest_id(sid), offset))
    ordered_offsets.sort()

    def ordered_native_records():
        with Path(native_path).open("rb") as fp:
            for _, offset in ordered_offsets:
                fp.seek(offset)
                yield json.loads(fp.readline())

    inputs = []
    counts = Counter()
    reasons = Counter()
    pdf_cache = {}
    for example in ordered_native_records():
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
                    text = "\n".join(p.get_text() for p in document)
                if len(pdf_cache) >= 2:
                    pdf_cache.pop(next(iter(pdf_cache)))
                pdf_cache[doc_id] = text
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



def validate_serving_candidates(
    path: str | Path, *, dataset: str, input_lengths: list[int],
) -> dict:
    """CPU-only full JSONL integrity check, before any paid vLLM startup.

    Merely finding a file on Drive does not prove that its contents survived
    interruption/sync. Validate every record, all IDs and bin memberships;
    never silently drop a malformed line, image or scientific sample.
    """
    from .workloads import WorkloadError

    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate source missing: {candidate}")
    expected = set(input_lengths)
    if not expected or any(type(n) is not int or n <= 0 for n in expected):
        raise ValueError("input_lengths must be positive integer token bins")
    counts = Counter()
    seen = set()
    with candidate.open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                raise WorkloadError(
                    f"{candidate}: blank JSONL line {lineno}; source must be regenerated")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise WorkloadError(
                    f"{candidate}: INVALID_CANDIDATE_JSONL line {lineno}, "
                    f"column {exc.colno}: {exc.msg}. Existing file is malformed "
                    "or incomplete; do not start GPU until CPU repair succeeds."
                ) from exc
            if not isinstance(item, dict):
                raise WorkloadError(f"{candidate}: JSONL line {lineno} is not an object")
            sid = item.get("id")
            if not isinstance(sid, str) or not sid or sid in seen:
                raise WorkloadError(
                    f"{candidate}: missing/duplicate sample ID at line {lineno}")
            seen.add(sid)
            if (item.get("dataset") != dataset
                    or item.get("track") != "derived_serving"
                    or not item.get("derived_from")):
                raise WorkloadError(
                    f"{candidate}: invalid dataset/track/origin at line {lineno}")
            token_bin = item.get("requested_text_length_bin")
            if type(token_bin) is not int or token_bin not in expected:
                raise WorkloadError(
                    f"{candidate}: invalid text token bin at line {lineno}")
            messages = item.get("messages")
            if not isinstance(messages, list) or len(messages) != 1:
                raise WorkloadError(
                    f"{candidate}: expected one chat message at line {lineno}")
            parts = messages[0].get("content", [])
            if (messages[0].get("role") != "user" or not isinstance(parts, list)
                    or not any(p.get("type") == "image_path" and p.get("path")
                               and p.get("asset_root") for p in parts if isinstance(p, dict))
                    or not any(p.get("type") == "text" and p.get("text")
                               for p in parts if isinstance(p, dict))):
                raise WorkloadError(
                    f"{candidate}: missing real image/text reference at line {lineno}")
            if item.get("image_count") != 1:
                raise WorkloadError(
                    f"{candidate}: unexpected image count at line {lineno}")
            counts[token_bin] += 1
    if not seen:
        raise WorkloadError(f"{candidate}: empty candidate JSONL")
    return {
        "status": "VALIDATED_SOURCE_DERIVED_CANDIDATES",
        "path": str(candidate),
        "records": len(seen),
        "per_text_bin": {str(n): counts[n] for n in sorted(expected)},
        "source_sha256": sha256(candidate),
    }


def validate_annotation_resume_source(
    candidates: str | Path, annotated: str | Path, *, model_id: str,
) -> None:
    """Fail early if saved annotation state references different source bytes."""
    from .workloads import WorkloadError

    annotated = Path(annotated)
    state = annotated.with_name(annotated.name + ".state.json")
    if not annotated.exists() and not state.exists():
        return
    if not annotated.is_file() or not state.is_file():
        raise WorkloadError(
            "Incomplete annotated dataset/state pair on Drive; manually audit "
            f"{annotated} and {state} before GPU. Neither was overwritten.")
    try:
        receipt = json.loads(state.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise WorkloadError(f"Unreadable annotation state {state}") from exc
    if (receipt.get("source_sha256") != sha256(candidates)
            or receipt.get("model") != model_id):
        raise WorkloadError(
            "Annotation resume source SHA/model mismatch. Preserve existing "
            f"{annotated}; use new campaign paths or audit before rerunning GPU.")


def prepare_serving_candidates(
    native_path: str | Path,
    output_path: str | Path,
    *,
    annotated_path: str | Path,
    dataset: str,
    model_id: str,
    model_directory: str | Path,
    input_lengths: list[int],
    max_per_bin: int,
    local_scratch_dir: str | Path,
    pdf_root: str | Path | None = None,
) -> dict:
    """Reuse only complete candidates; safely rebuild corrupt Drive JSONL on SSD.

    A corrupt source is preserved as a content-addressed .corrupt backup on
    Drive. Rebuilding is forbidden if an annotation campaign already exists;
    those receipts must never be silently invalidated.
    """
    import shutil
    from .workloads import WorkloadError

    output = Path(output_path)
    prior_error = None
    if output.is_file():
        try:
            audit = validate_serving_candidates(
                output, dataset=dataset, input_lengths=input_lengths)
        except (WorkloadError, UnicodeError, OSError) as exc:
            prior_error = str(exc)
        else:
            validate_annotation_resume_source(
                output, annotated_path, model_id=model_id)
            return {**audit, "status": "REUSED_VALIDATED_CANDIDATES"}
    if (Path(annotated_path).exists()
            or Path(str(annotated_path) + ".state.json").exists()):
        raise WorkloadError(
            "Candidate regeneration refused: annotated data/state exists; "
            "preserve and audit provenance before any replacement. "
            f"Candidates: {output}; annotations: {annotated_path}")

    # Generate on local SSD first. Never publish a partially built JSONL to
    # Drive, and never overwrite the damaged file if generation or audit fails.
    local = Path(local_scratch_dir) / (
        f"{model_id}_{dataset}_candidate_rebuild.jsonl")
    local.parent.mkdir(parents=True, exist_ok=True)
    result = build_serving_candidates(
        native_path, local, dataset=dataset,
        model_directory=model_directory, input_lengths=input_lengths,
        max_per_bin=max_per_bin, pdf_root=pdf_root)
    checked = validate_serving_candidates(
        local, dataset=dataset, input_lengths=input_lengths)
    verified_digest = checked["source_sha256"]

    output.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if output.exists():
        bad_digest = sha256(output)
        backup = output.with_name(
            output.stem + f".corrupt-{bad_digest[:12]}" + output.suffix)
        if backup.is_file():
            if sha256(backup) != bad_digest:
                raise WorkloadError(f"Corrupt candidate backup conflict: {backup}")
        else:
            shutil.copy2(output, backup)
            if sha256(backup) != bad_digest:
                raise IOError(f"Corrupt candidate backup verification failed: {backup}")
        if sha256(output) != bad_digest:
            raise WorkloadError("Candidate file changed during repair; refusing overwrite")

    pending = output.with_name(output.name + ".repair.pending")
    try:
        shutil.copy2(local, pending)
        if sha256(pending) != verified_digest:
            raise IOError("Local-to-Drive candidate staging SHA mismatch")
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
    published = validate_serving_candidates(
        output, dataset=dataset, input_lengths=input_lengths)
    if published["source_sha256"] != verified_digest:
        raise IOError("Published candidate digest differs from local validation")
    return {
        **published,
        "status": "REBUILT_CORRUPT_CANDIDATES" if prior_error
                  else "BUILT_VALIDATED_CANDIDATES",
        "prior_error": prior_error,
        "preserved_corrupt_backup": str(backup) if backup else None,
        "builder": result.get("status"),
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
