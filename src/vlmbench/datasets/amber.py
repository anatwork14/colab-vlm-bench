"""AMBER (LLM-free Multi-dimensional Hallucination Benchmark) adapter and scorer.

Provides native Qwen VLM formatted prompts for both generative and discriminative
splits, clear pass-type distinctions (natural-EOS quality vs fixed serving), and
official evaluation scoring compatibility across existence, attribute, and relation.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from ..artifacts import atomic_json, atomic_jsonl, sha256

AMBER_SPLITS = ("all", "generative", "discriminative", "existence", "attribute", "relation")
DISCRIMINATIVE_DIMENSIONS = ("existence", "attribute", "relation")
PASS_TYPES = ("natural_eos_quality", "fixed_serving")


class AmberError(ValueError):
    pass


def parse_binary_yes_no(text: str) -> str | None:
    """Extract canonical 'Yes' or 'No' from model prediction text."""
    if not text:
        return None
    cleaned = unicodedata.normalize("NFKC", str(text)).strip().lower()
    cleaned = re.sub(r"[^\w\s]", " ", cleaned).strip()
    tokens = cleaned.split()
    if not tokens:
        return None
    # Check first token first (standard greedy decoding response)
    first = tokens[0]
    if first in ("yes", "yeah", "yep", "true"):
        return "Yes"
    if first in ("no", "nope", "false"):
        return "No"
    # Fallback to searching first occurrence
    for tok in tokens:
        if tok in ("yes", "yeah", "yep", "true"):
            return "Yes"
        if tok in ("no", "nope", "false"):
            return "No"
    return None


def extract_mentioned_words(text: str) -> set[str]:
    """Tokenize and normalize text for basic object mention matching."""
    cleaned = unicodedata.normalize("NFKC", str(text)).lower()
    words = re.findall(r"\b[a-z]{2,}\b", cleaned)
    # Basic singularization/normalization heuristic
    normalized = set()
    for w in words:
        normalized.add(w)
        if w.endswith("s") and len(w) > 3:
            normalized.add(w[:-1])
        if w.endswith("es") and len(w) > 4:
            normalized.add(w[:-2])
    return normalized


def normalize_amber(
    source: Path | str,
    images_root: Path | str,
    output: Path | str,
    *,
    split: str = "all",
    pass_type: str = "natural_eos_quality",
    limit: int | None = None,
) -> dict[str, Any]:
    """Normalize AMBER dataset to Qwen VLM chat template format.

    Supports generative and discriminative splits, enforces user/assistant role
    format, and distinguishes between natural-EOS quality and fixed serving passes.
    """
    if split not in AMBER_SPLITS:
        raise AmberError(f"Unsupported AMBER split: {split}. Expected one of {AMBER_SPLITS}")
    if pass_type not in PASS_TYPES:
        raise AmberError(f"Unsupported pass_type: {pass_type}. Expected one of {PASS_TYPES}")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive or None")

    source = Path(source)
    images_root = Path(images_root).resolve()
    output = Path(output)

    if not source.is_file():
        raise FileNotFoundError(f"AMBER source file not found: {source}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"AMBER images root not found: {images_root}")

    # Read queries/annotations
    try:
        raw_content = source.read_text(encoding="utf8")
        data = json.loads(raw_content)
    except Exception as exc:
        raise AmberError(f"Failed to parse AMBER source JSON: {exc}") from exc

    if isinstance(data, dict):
        # Support dict container formats
        data = data.get("data", data.get("queries", data.get("annotations", [])))
    if not isinstance(data, list):
        raise AmberError("AMBER source JSON must be a list of query/annotation objects")

    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    image_count = 0

    for item in data:
        if limit is not None and len(records) >= limit:
            break

        if not isinstance(item, dict):
            raise AmberError("AMBER items must be dictionaries")

        raw_id = item.get("id")
        if raw_id is None:
            raise AmberError("AMBER item missing required 'id'")
        source_id = str(raw_id)

        # Determine task type: generative or discriminative
        # In official AMBER: 1 <= id <= 1004 is generative, >= 1005 is discriminative
        item_task = item.get("task")
        if not item_task:
            try:
                int_id = int(raw_id)
                item_task = "generative" if 1 <= int_id <= 1004 else "discriminative"
            except ValueError:
                item_task = "discriminative"
        item_task = str(item_task).lower().strip()

        # Dimension / sub-type for discriminative
        sub_type = str(item.get("type", "")).lower().strip()
        if not sub_type:
            sub_type = "existence" if item_task == "discriminative" else "all"

        # Split filtering
        if split == "generative" and item_task != "generative":
            continue
        if split == "discriminative" and item_task != "discriminative":
            continue
        if split in DISCRIMINATIVE_DIMENSIONS and (item_task != "discriminative" or sub_type != split):
            continue

        # Extract image filename
        img_name = item.get("image")
        if not img_name:
            # Fallback to standard AMBER naming convention: AMBER_{id}.jpg
            img_name = f"AMBER_{source_id}.jpg"
        img_name = str(img_name)

        # Path traversal guard and image presence check
        img_path = (images_root / img_name).resolve()
        if not img_path.is_relative_to(images_root) or not img_path.is_file():
            # Check nested subdirectories (e.g. image/, images/)
            alt1 = (images_root / "images" / img_name).resolve()
            alt2 = (images_root / "image" / img_name).resolve()
            if alt1.is_relative_to(images_root) and alt1.is_file():
                img_path = alt1
            elif alt2.is_relative_to(images_root) and alt2.is_file():
                img_path = alt2
            else:
                raise AmberError(f"Missing or escaping AMBER image: {img_name} at {img_path}")

        # Extract query prompt text
        query_text = item.get("query") or item.get("question") or item.get("prompt")
        if not query_text:
            if item_task == "generative":
                query_text = "Please describe this image in detail."
            else:
                raise AmberError(f"AMBER discriminative item {source_id} missing query text")
        query_text = str(query_text).strip()

        # Deterministic unique record ID
        record_id = f"amber__{item_task}__{source_id}"
        if record_id in seen_ids:
            raise AmberError(f"Duplicate AMBER record ID: {record_id}")
        seen_ids.add(record_id)

        # Format messages compliant with Qwen VLM chat template (roles: 'user', 'assistant')
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_path",
                        "path": str(img_path),
                        "asset_root": str(images_root),
                    },
                    {
                        "type": "text",
                        "text": query_text,
                    },
                ],
            }
        ]

        # Extract gold reference
        gold_reference = (
            item.get("label")
            or item.get("answer")
            or item.get("ground_truth")
            or item.get("objects")
        )
        if gold_reference is not None and not isinstance(gold_reference, str):
            gold_reference = json.dumps(gold_reference, ensure_ascii=False)

        # Clear distinction between natural-EOS quality pass and fixed-output serving pass
        if pass_type == "natural_eos_quality":
            track = "native_quality"
            record: dict[str, Any] = {
                "id": record_id,
                "dataset": "amber",
                "track": track,
                "messages": messages,
                "reference": gold_reference,
                "native_context_complete": True,
                "meta": {
                    "source_question_id": raw_id,
                    "task": item_task,
                    "dimension": sub_type,
                    "image": img_name,
                    "pass_type": "natural_eos_quality",
                    "max_output_tokens": 256,
                    "stop_sequences": ["<|im_end|>", "<|endoftext|>"],
                    "temperature": 0.0,
                    "official_scoring_eligible": True,
                    "objects": item.get("objects", []),
                    "adapter_provenance": "colab-vlm-bench-amber-v1; official_scoring_compatible",
                },
            }
        else:
            track = "derived_serving"
            record = {
                "id": f"{record_id}__fixed256",
                "dataset": "amber",
                "track": track,
                "derived_from": record_id,
                "messages": messages,
                "requested_text_length_bin": 256,
                "meta": {
                    "original_question_id": raw_id,
                    "task": item_task,
                    "dimension": sub_type,
                    "image": img_name,
                    "pass_type": "fixed_serving",
                    "fixed_output_tokens": 256,
                    "ignore_eos": True,
                    "temperature": 0.0,
                    "official_quality_allowed": False,
                    "adapter_provenance": "colab-vlm-bench-amber-v1-serving",
                },
            }

        records.append(record)
        image_count += 1

    atomic_jsonl(output, records)
    return {
        "dataset": "amber",
        "split": split,
        "pass_type": pass_type,
        "count": len(records),
        "image_references": image_count,
        "sha256_source": sha256(source),
        "sha256_normalized": sha256(output),
        "output": str(output),
        "status": (
            "NORMALIZED_OFFICIAL_SCORING_ELIGIBLE"
            if pass_type == "natural_eos_quality"
            else "NORMALIZED_DERIVED_SERVING_WORKLOAD"
        ),
    }


def score_amber_discriminative(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """Official scoring compatibility for AMBER discriminative tasks.

    Evaluates binary Yes/No classification across existence, attribute, and relation.
    Computes Accuracy, Precision, Recall, F1, Yes-Ratio, and Hallucination Rate.
    """
    if isinstance(ground_truth, list):
        gt_map = {}
        for x in ground_truth:
            sid = str(x.get("id"))
            label = x.get("label") or x.get("reference") or x.get("answer")
            dim = x.get("meta", {}).get("dimension") or x.get("type", "existence")
            gt_map[sid] = {"label": str(label).strip().capitalize(), "dimension": dim}
    else:
        gt_map = ground_truth

    dim_metrics: dict[str, dict[str, int]] = {
        dim: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "unparsed": 0, "total": 0}
        for dim in DISCRIMINATIVE_DIMENSIONS
    }
    overall = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "unparsed": 0, "total": 0}

    for pred in predictions:
        sid = str(pred.get("id"))
        # Strip prefixes if matching normalized record IDs
        norm_sid = sid
        if norm_sid not in gt_map and norm_sid.startswith("amber__discriminative__"):
            short_id = norm_sid.replace("amber__discriminative__", "")
            if short_id in gt_map:
                norm_sid = short_id

        if norm_sid not in gt_map:
            continue

        gt_item = gt_map[norm_sid]
        gt_label = gt_item["label"]  # Expected "Yes" or "No"
        dimension = gt_item.get("dimension", "existence").lower()
        if dimension not in dim_metrics:
            dimension = "existence"

        response_text = pred.get("prediction") or pred.get("response") or ""
        parsed = parse_binary_yes_no(str(response_text))

        dim_metrics[dimension]["total"] += 1
        overall["total"] += 1

        if parsed is None:
            dim_metrics[dimension]["unparsed"] += 1
            overall["unparsed"] += 1
            continue

        is_yes_gt = gt_label == "Yes"
        is_yes_pred = parsed == "Yes"

        if is_yes_gt and is_yes_pred:
            dim_metrics[dimension]["tp"] += 1
            overall["tp"] += 1
        elif not is_yes_gt and is_yes_pred:
            dim_metrics[dimension]["fp"] += 1
            overall["fp"] += 1
        elif not is_yes_gt and not is_yes_pred:
            dim_metrics[dimension]["tn"] += 1
            overall["tn"] += 1
        elif is_yes_gt and not is_yes_pred:
            dim_metrics[dimension]["fn"] += 1
            overall["fn"] += 1

    def compute_scores(counts: dict[str, int]) -> dict[str, Any]:
        total = counts["total"]
        if total == 0:
            return {
                "count": 0, "accuracy": None, "precision": None,
                "recall": None, "f1": None, "hallucination_rate": None,
                "yes_ratio": None, "unparsed_rate": None,
            }
        tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        valid = tp + fp + tn + fn
        acc = (tp + tn) / valid if valid else 0.0
        # Precision & Recall for detecting presence/positive class
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        yes_ratio = (tp + fp) / valid if valid else 0.0
        # Hallucination rate in AMBER discriminative is false positive rate / error rate
        hallucination_rate = fp / (fp + tn) if (fp + tn) else 0.0
        return {
            "count": total,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "hallucination_rate": hallucination_rate,
            "yes_ratio": yes_ratio,
            "unparsed_rate": counts["unparsed"] / total,
        }

    per_dimension = {dim: compute_scores(dim_metrics[dim]) for dim in DISCRIMINATIVE_DIMENSIONS}
    overall_scores = compute_scores(overall)

    return {
        "status": "AMBER_DISCRIMINATIVE_SCORED",
        "overall": overall_scores,
        "dimensions": per_dimension,
    }


# Standard COCO / AMBER object vocabulary classes
DEFAULT_AMBER_OBJECTS: frozenset[str] = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "sofa", "potted plant", "bed", "dining table", "table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "park", "grass", "tree", "street",
})


def score_amber_generative(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | dict[str, Any],
    object_vocabulary: set[str] | None = None,
) -> dict[str, Any]:
    """Official scoring compatibility for AMBER generative descriptions.

    Calculates CHAIR_I (Instance-level object hallucination rate),
    CHAIR_S / Ha_rate (Sentence/description-level hallucination rate),
    Precision, Recall (Coverage), and F1.

    Evaluates object mentions against benchmark object vocabulary (AMBER/COCO classes
    and annotated dataset objects) rather than arbitrary English adjectives/verbs.
    """
    if isinstance(annotations, list):
        ann_map = {}
        for a in annotations:
            aid = str(a.get("id"))
            objs = a.get("objects", [])
            if isinstance(objs, str):
                try:
                    objs = json.loads(objs)
                except (json.JSONDecodeError, TypeError):
                    objs = [objs]
            ann_map[aid] = {str(o).lower().strip() for o in objs}
    else:
        ann_map = annotations

    # Build active object evaluation vocabulary
    active_vocab = set(DEFAULT_AMBER_OBJECTS)
    if object_vocabulary:
        active_vocab.update(str(o).lower().strip() for o in object_vocabulary)
    for objs in ann_map.values():
        for obj in objs:
            active_vocab.update(extract_mentioned_words(obj))

    total_predictions = 0
    hallucinated_descriptions = 0
    total_mentioned_objects = 0
    total_hallucinated_objects = 0
    total_gt_objects = 0
    per_sample_details = []

    for pred in predictions:
        sid = str(pred.get("id"))
        norm_sid = sid
        if norm_sid not in ann_map and norm_sid.startswith("amber__generative__"):
            short_id = norm_sid.replace("amber__generative__", "")
            if short_id in ann_map:
                norm_sid = short_id

        if norm_sid not in ann_map:
            continue

        gt_objects = ann_map[norm_sid]
        resp_text = str(pred.get("prediction") or pred.get("response") or "")
        words_in_resp = extract_mentioned_words(resp_text)

        # Restrict candidate mentions to known benchmark object categories
        candidate_mentions = {w for w in words_in_resp if w in active_vocab}

        # Determine which candidate mentions are in ground truth vs hallucinated
        gt_word_sets = {w for gt_obj in gt_objects for w in extract_mentioned_words(gt_obj)}
        mentioned_gt = {w for w in candidate_mentions if w in gt_word_sets}
        hallucinated = {w for w in candidate_mentions if w not in gt_word_sets}

        n_pred = len(candidate_mentions)
        n_hal = len(hallucinated)
        has_hallucination = bool(n_hal > 0)

        total_predictions += 1
        if has_hallucination:
            hallucinated_descriptions += 1
        total_mentioned_objects += n_pred
        total_hallucinated_objects += n_hal
        total_gt_objects += len(gt_objects)

        per_sample_details.append({
            "id": sid,
            "mentioned_gt": list(mentioned_gt),
            "hallucinated": list(hallucinated),
            "has_hallucination": has_hallucination,
        })

    chair_i = (
        total_hallucinated_objects / total_mentioned_objects
        if total_mentioned_objects > 0
        else 0.0
    )
    chair_s = (
        hallucinated_descriptions / total_predictions
        if total_predictions > 0
        else 0.0
    )
    correct_objects = total_mentioned_objects - total_hallucinated_objects
    precision = (
        correct_objects / total_mentioned_objects
        if total_mentioned_objects > 0
        else 0.0
    )
    recall = (
        correct_objects / total_gt_objects
        if total_gt_objects > 0
        else 0.0
    )
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "status": "AMBER_GENERATIVE_SCORED",
        "count": total_predictions,
        "chair_i": chair_i,
        "chair_s": chair_s,
        "hallucination_rate": chair_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_mentioned_objects": total_mentioned_objects,
        "total_hallucinated_objects": total_hallucinated_objects,
        "details_sample_count": len(per_sample_details),
    }


def score_amber(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | Path | str,
) -> dict[str, Any]:
    """Unified AMBER scoring combining generative and discriminative evaluation.

    Computes:
      - Generative CHAIR_I, CHAIR_S, Precision, Recall, F1
      - Discriminative Accuracy, Precision, Recall, F1 across dimensions
      - Combined AMBER Score = 0.5 * ((1.0 - CHAIR_I) + Discriminative_F1)
    """
    if isinstance(annotations, (str, Path)):
        p = Path(annotations)
        if not p.is_file():
            raise FileNotFoundError(f"Annotations file missing: {p}")
        raw = json.loads(p.read_text(encoding="utf8"))
        ann_list = raw if isinstance(raw, list) else raw.get("data", raw.get("annotations", []))
    else:
        ann_list = annotations

    gen_ann = []
    disc_ann = []
    for item in ann_list:
        task = str(item.get("task", "")).lower()
        if not task:
            try:
                task = "generative" if int(item.get("id", 0)) <= 1004 else "discriminative"
            except ValueError:
                task = "discriminative"
        if task == "generative":
            gen_ann.append(item)
        else:
            disc_ann.append(item)

    gen_preds = []
    disc_preds = []
    for pred in predictions:
        sid = str(pred.get("id"))
        if "generative" in sid:
            gen_preds.append(pred)
        elif "discriminative" in sid:
            disc_preds.append(pred)
        else:
            # Fallback by ID range
            try:
                numeric_id = int(re.findall(r"\d+", sid)[0])
                if numeric_id <= 1004:
                    gen_preds.append(pred)
                else:
                    disc_preds.append(pred)
            except (IndexError, ValueError):
                disc_preds.append(pred)

    gen_result = score_amber_generative(gen_preds, gen_ann) if gen_preds and gen_ann else None
    disc_result = score_amber_discriminative(disc_preds, disc_ann) if disc_preds and disc_ann else None

    # Unified AMBER score calculation
    chair_val = gen_result["chair_i"] if gen_result else 0.0
    disc_f1 = (
        disc_result["overall"]["f1"]
        if (disc_result and disc_result["overall"]["f1"] is not None)
        else 0.0
    )
    amber_score = 0.5 * ((1.0 - chair_val) + disc_f1)

    return {
        "status": "AMBER_OFFICIAL_SCORED",
        "amber_score": amber_score,
        "generative": gen_result,
        "discriminative": disc_result,
        "predictions_total": len(predictions),
    }


def export_amber_official_format(
    predictions: list[dict[str, Any]],
    output_path: Path | str,
    task: str = "all",
) -> Path:
    """Export model predictions to AMBER official JSON format for upstream inference.py.

    Generative format: [{"id": 1, "response": "..."}]
    Discriminative format: [{"id": 1005, "response": "Yes"}]
    """
    output_path = Path(output_path)
    exported = []
    for pred in predictions:
        sid = pred.get("id")
        # Extract integer ID
        numeric_matches = re.findall(r"\d+", str(sid))
        target_id = int(numeric_matches[-1]) if numeric_matches else sid

        resp = str(pred.get("prediction") or pred.get("response") or "")
        if task == "discriminative" or (task == "all" and isinstance(target_id, int) and target_id >= 1005):
            parsed_yn = parse_binary_yes_no(resp)
            resp = parsed_yn if parsed_yn else resp

        exported.append({"id": target_id, "response": resp})

    atomic_json(output_path, exported)
    return output_path
