"""AMBER (LLM-free Multi-dimensional Hallucination Benchmark) adapter and scorer.

Provides Qwen VLM formatted prompts, strict upstream schema validation,
clear pass-type distinctions (natural-EOS quality vs fixed serving), and
audited proxy scoring across existence, attribute, and relation.

NOTE: Scorer is a proxy re-implementation; official upstream scorer parity
must be proven against upstream reference output before claiming official status.
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

# Upstream AMBER repository pin and provenance
AMBER_UPSTREAM_MANIFEST: dict[str, Any] = {
    "repository": "https://github.com/junyangwang0410/AMBER",
    "repo_id": "junyangwang0410/AMBER",
    "pinned_commit": "0121a5cf8a62372d8e411b58a140f7f329ee4fa9",
    "license": "Apache-2.0",
    "paper_citation": "arXiv:2311.07399",
    "expected_files": [
        "data/query/query_all.json",
        "data/query/query_generative.json",
        "data/query/query_discriminative.json",
        "data/annotation/annotations.json",
    ],
    "scorer_provenance": "junyangwang0410/AMBER/inference.py",
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


class AmberError(ValueError):
    """Base error for AMBER validation and processing failures."""


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
    normalized = set()
    for w in words:
        normalized.add(w)
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            normalized.add(w[:-1])
        if w.endswith("es") and len(w) > 4:
            normalized.add(w[:-2])
    return normalized


def extract_mentioned_objects(text: str, active_vocab: set[str]) -> set[str]:
    """Extract candidate object mentions from text matching known vocabulary.

    Supports both multi-word phrases (e.g. 'traffic light', 'fire hydrant')
    and single-word objects with singular/plural normalization.
    """
    cleaned = unicodedata.normalize("NFKC", str(text)).lower()
    single_words = extract_mentioned_words(cleaned)
    found = set()

    for obj in active_vocab:
        obj_clean = obj.lower().strip()
        if " " in obj_clean:
            # Multi-word phrase matching
            pattern = r"\b" + re.escape(obj_clean) + r"\b"
            if re.search(pattern, cleaned):
                found.add(obj_clean)
        elif obj_clean in single_words:
            found.add(obj_clean)
    return found


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

    Strict schema validation: missing required fields raise ValueError.
    Computes per-record image SHA-256 and records upstream pin manifest.
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

    try:
        raw_content = source.read_text(encoding="utf8")
        data = json.loads(raw_content)
    except Exception as exc:
        raise AmberError(f"Failed to parse AMBER source JSON: {exc}") from exc

    if isinstance(data, dict):
        valid_keys = [k for k in ("queries", "data", "annotations") if k in data]
        if not valid_keys:
            raise ValueError(
                f"AMBER source JSON dict must contain one of ('queries', 'data', 'annotations'), got keys: {list(data.keys())}"
            )
        data = data[valid_keys[0]]

    if not isinstance(data, list):
        raise TypeError(f"AMBER source data must be a list of query/annotation objects, got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError("AMBER source data must be a non-empty list of items")

    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    image_sha_cache: dict[Path, str] = {}
    image_count = 0

    for item in data:
        if limit is not None and len(records) >= limit:
            break

        if not isinstance(item, dict):
            raise TypeError("AMBER items must be dictionaries")

        # DS-02: Strict ID validation
        raw_id = item.get("id")
        if raw_id is None or raw_id == "":
            raise ValueError("AMBER item missing required 'id'")
        source_id = str(raw_id)

        # DS-02: Strict task validation
        item_task = item.get("task")
        if not item_task or not isinstance(item_task, str) or item_task.lower() not in ("generative", "discriminative"):
            raise ValueError(f"AMBER item {source_id} missing or invalid required 'task' (must be 'generative' or 'discriminative')")
        item_task = item_task.lower().strip()

        # DS-02: Strict image validation (no silent defaulting)
        img_name = item.get("image")
        if not img_name or not isinstance(img_name, str):
            raise ValueError(f"AMBER item {source_id} missing required 'image'")
        img_name = str(img_name).strip()

        # Path traversal guard and image presence check
        img_path = (images_root / img_name).resolve()
        if not img_path.is_relative_to(images_root) or not img_path.is_file():
            alt1 = (images_root / "images" / img_name).resolve()
            alt2 = (images_root / "image" / img_name).resolve()
            if alt1.is_relative_to(images_root) and alt1.is_file():
                img_path = alt1
            elif alt2.is_relative_to(images_root) and alt2.is_file():
                img_path = alt2
            else:
                raise AmberError(f"Missing or escaping AMBER image: {img_name} at {img_path}")

        # Compute per-record image SHA-256 (cached) (DS-07)
        if img_path not in image_sha_cache:
            image_sha_cache[img_path] = sha256(img_path)
        img_sha = image_sha_cache[img_path]

        # DS-02 / DS-05: Strict query validation (no fallback prompt)
        query_text = item.get("query")
        if not query_text or not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(f"AMBER item {source_id} missing required 'query'")
        query_text = query_text.strip()

        # Discriminative-specific strict validation
        if item_task == "discriminative":
            sub_type = item.get("type")
            if sub_type and isinstance(sub_type, str):
                st_lower = sub_type.lower().strip()
                if "attribute" in st_lower:
                    sub_type = "attribute"
                elif "relation" in st_lower:
                    sub_type = "relation"
                elif "existence" in st_lower:
                    sub_type = "existence"
            if not sub_type or not isinstance(sub_type, str) or sub_type.lower() not in DISCRIMINATIVE_DIMENSIONS:
                raise ValueError(
                    f"AMBER discriminative item {source_id} missing or invalid required 'type' "
                    f"(must be one of {DISCRIMINATIVE_DIMENSIONS})"
                )
            sub_type = sub_type.lower().strip()

            label = item.get("label")
            if not label or not isinstance(label, str) or label.strip() not in ("Yes", "No"):
                raise ValueError(
                    f"AMBER discriminative item {source_id} missing or invalid required 'label' (must be 'Yes' or 'No')"
                )
            gold_reference = label.strip()
        else:
            sub_type = "generative"
            # In generative queries/annotations: validate objects if provided
            objects = item.get("objects")
            if objects is not None:
                if not isinstance(objects, list) or not all(isinstance(o, str) for o in objects):
                    raise ValueError(f"AMBER generative item {source_id} 'objects' must be a list of strings")
                gold_reference = json.dumps(objects, ensure_ascii=False)
            else:
                gold_reference = None

        # Split filtering
        if split == "generative" and item_task != "generative":
            continue
        if split == "discriminative" and item_task != "discriminative":
            continue
        if split in DISCRIMINATIVE_DIMENSIONS and (item_task != "discriminative" or sub_type != split):
            continue

        record_id = f"amber__{item_task}__{source_id}"
        if record_id in seen_ids:
            raise AmberError(f"Duplicate AMBER record ID: {record_id}")
        seen_ids.add(record_id)

        # Compliant Qwen VLM message format
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_path",
                        "path": str(img_path),
                        "asset_root": str(images_root),
                        "image_sha256": img_sha,
                    },
                    {
                        "type": "text",
                        "text": query_text,
                    },
                ],
            }
        ]

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
                    "image_sha256": img_sha,
                    "pass_type": "natural_eos_quality",
                    "official_scoring_eligible": False,  # DS-01: False until official upstream parity proven
                    "objects": item.get("objects", []),
                    "adapter_provenance": "colab-vlm-bench-amber-v1; proxy_unverified",
                },
            }
        else:
            # DS-06: track='amber_serving' with no requested_text_length_bin
            track = "amber_serving"
            record = {
                "id": f"{record_id}__serving",
                "dataset": "amber",
                "track": track,
                "derived_from": record_id,
                "messages": messages,
                "meta": {
                    "original_question_id": raw_id,
                    "task": item_task,
                    "dimension": sub_type,
                    "image": img_name,
                    "image_sha256": img_sha,
                    "pass_type": "fixed_serving",
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
        "upstream_manifest": AMBER_UPSTREAM_MANIFEST,  # DS-07
        "status": (
            "AMBER_PROXY_UNVERIFIED"
            if pass_type == "natural_eos_quality"
            else "AMBER_SERVING_PREPARED"
        ),
    }


def score_amber_discriminative(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]] | dict[str, Any],
) -> dict[str, Any]:
    """Proxy scoring compatibility for AMBER discriminative tasks.

    DS-01: Status is AMBER_DISCRIMINATIVE_PROXY_UNVERIFIED.
    DS-02: Strict schema validation on ground truth.
    DS-03: Full ground truth population defines denominator; missing and unparseable
    answers count as incorrect in accuracy; empty denominators return None/NOT_MEASURED.
    """
    if isinstance(ground_truth, list):
        items_to_check = []
        for x in ground_truth:
            if not isinstance(x, dict):
                raise TypeError("Ground-truth items must be dictionaries")
            sid = x.get("id")
            if sid is None:
                raise ValueError("Ground-truth item missing required 'id'")
            items_to_check.append((str(sid), x))
    elif isinstance(ground_truth, dict):
        items_to_check = []
        for k, v in ground_truth.items():
            if not isinstance(v, dict):
                raise TypeError(f"Ground-truth item {k} value must be a dictionary")
            items_to_check.append((str(k), v))
    else:
        raise TypeError("Ground truth must be a list or dict")

    gt_map: dict[str, dict[str, str]] = {}
    for sid, x in items_to_check:
        label = x.get("label")
        if not label or not isinstance(label, str) or label.strip() not in ("Yes", "No"):
            raise ValueError(f"Ground-truth item {sid} missing or invalid required 'label' ('Yes'/'No'), got {label!r}")
        dim = x.get("type") or x.get("dimension") or (x.get("meta", {}).get("dimension") if isinstance(x.get("meta"), dict) else None)
        if not dim or not isinstance(dim, str) or dim.lower().strip() not in DISCRIMINATIVE_DIMENSIONS:
            raise ValueError(f"Ground-truth item {sid} missing or invalid required 'type' ({DISCRIMINATIVE_DIMENSIONS}), got {dim!r}")
        if sid in gt_map:
            raise ValueError(f"Duplicate ground-truth item id: {sid}")
        gt_map[sid] = {"label": label.strip().capitalize(), "dimension": dim.lower().strip()}

    total_gt = len(gt_map)
    if total_gt == 0:
        return {
            "status": "NOT_MEASURED",
            "overall": {
                "count": 0, "accuracy": None, "precision": None, "recall": None,
                "f1": None, "hallucination_rate": None, "yes_ratio": None,
                "unparseable_count": 0, "missing_count": 0,
            },
            "dimensions": {
                dim: {
                    "count": 0, "accuracy": None, "precision": None, "recall": None,
                    "f1": None, "hallucination_rate": None, "yes_ratio": None,
                    "unparseable_count": 0, "missing_count": 0,
                }
                for dim in DISCRIMINATIVE_DIMENSIONS
            },
        }

    # Index predictions
    pred_map = {}
    for pred in predictions:
        sid = str(pred.get("id"))
        norm_sid = sid
        if norm_sid.startswith("amber__discriminative__"):
            norm_sid = norm_sid.replace("amber__discriminative__", "")
        pred_map[norm_sid] = pred.get("prediction") or pred.get("response") or ""

    dim_metrics: dict[str, dict[str, int]] = {
        dim: {
            "tp": 0, "fp": 0, "tn": 0, "fn": 0,
            "unparseable": 0, "missing": 0, "total": 0,
        }
        for dim in DISCRIMINATIVE_DIMENSIONS
    }
    overall = {
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "unparseable": 0, "missing": 0, "total": total_gt,
    }

    # Score over EVERY item in the frozen ground-truth population (DS-03)
    for sid, gt_item in gt_map.items():
        dimension = gt_item["dimension"]
        gt_label = gt_item["label"]
        is_positive = (gt_label == "Yes")
        dim_metrics[dimension]["total"] += 1

        if sid not in pred_map:
            # Missing answer counts as failure in accuracy and false negative if positive
            dim_metrics[dimension]["missing"] += 1
            overall["missing"] += 1
            if is_positive:
                dim_metrics[dimension]["fn"] += 1
                overall["fn"] += 1
            continue

        raw_pred = pred_map[sid]
        parsed = parse_binary_yes_no(str(raw_pred))

        if parsed is None:
            # Unparseable answer counts as failure in accuracy and false negative if positive
            dim_metrics[dimension]["unparseable"] += 1
            overall["unparseable"] += 1
            if is_positive:
                dim_metrics[dimension]["fn"] += 1
                overall["fn"] += 1
            continue

        is_yes_pred = (parsed == "Yes")

        if is_positive and is_yes_pred:
            dim_metrics[dimension]["tp"] += 1
            overall["tp"] += 1
        elif not is_positive and is_yes_pred:
            dim_metrics[dimension]["fp"] += 1
            overall["fp"] += 1
        elif not is_positive and not is_yes_pred:
            dim_metrics[dimension]["tn"] += 1
            overall["tn"] += 1
        elif is_positive and not is_yes_pred:
            dim_metrics[dimension]["fn"] += 1
            overall["fn"] += 1

    def compute_scores(counts: dict[str, int]) -> dict[str, Any]:
        tot = counts["total"]
        if tot == 0:
            return {
                "count": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                "accuracy": None, "precision": None,
                "recall": None, "f1": None, "hallucination_rate": None,
                "yes_ratio": None, "unparseable_count": 0, "missing_count": 0,
            }
        tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        unparseable, missing = counts["unparseable"], counts["missing"]

        # Accuracy denominator includes ALL ground truth items (including missing and unparseable)
        acc = (tp + tn) / tot

        # Precision & Recall on positive class
        prec = tp / (tp + fp) if (tp + fp) > 0 else (0.0 if tp == 0 and fp > 0 else None)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (0.0 if tp == 0 and fn > 0 else None)
        f1 = (
            (2 * prec * rec / (prec + rec))
            if (prec is not None and rec is not None and (prec + rec) > 0)
            else (0.0 if (prec == 0.0 or rec == 0.0) else None)
        )

        valid_predictions = tp + fp + tn + fn
        yes_ratio = (tp + fp) / valid_predictions if valid_predictions > 0 else None

        # Hallucination rate = false positive rate on negative items
        neg_total = fp + tn
        hallucination_rate = fp / neg_total if neg_total > 0 else None

        return {
            "count": tot,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "hallucination_rate": hallucination_rate,
            "yes_ratio": yes_ratio,
            "unparseable_count": unparseable,
            "missing_count": missing,
        }

    per_dimension = {dim: compute_scores(dim_metrics[dim]) for dim in DISCRIMINATIVE_DIMENSIONS}
    overall_scores = compute_scores(overall)

    return {
        "status": "AMBER_DISCRIMINATIVE_PROXY_UNVERIFIED",  # DS-01
        "overall": overall_scores,
        "dimensions": per_dimension,
        "warning": "Proxy re-implementation; official upstream scorer parity not verified",
    }


def score_amber_generative(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | dict[str, Any],
    object_vocabulary: set[str] | None = None,
) -> dict[str, Any]:
    """Proxy scoring compatibility for AMBER generative descriptions.

    DS-01: Status is AMBER_GENERATIVE_PROXY_UNVERIFIED.
    DS-02: Strict schema validation on annotations.
    DS-03: Full population defines denominator; missing items count as failures;
    empty denominators return None/NOT_MEASURED.
    """
    if isinstance(annotations, list):
        ann_items = []
        for a in annotations:
            if not isinstance(a, dict):
                raise TypeError("Annotation items must be dictionaries")
            aid = a.get("id")
            if aid is None:
                raise ValueError("Annotation item missing required 'id'")
            ann_items.append((str(aid), a))
    elif isinstance(annotations, dict):
        ann_items = []
        for k, v in annotations.items():
            if not isinstance(v, dict):
                raise TypeError(f"Annotation item {k} value must be a dictionary")
            ann_items.append((str(k), v))
    else:
        raise TypeError("Annotations must be a list or dict")

    ann_map: dict[str, set[str]] = {}
    for aid, a in ann_items:
        objs = a.get("objects")
        if not isinstance(objs, list) or not all(isinstance(o, str) for o in objs):
            raise ValueError(f"Annotation item {aid} missing or invalid required 'objects' list")
        if aid in ann_map:
            raise ValueError(f"Duplicate annotation item id: {aid}")
        ann_map[aid] = {str(o).lower().strip() for o in objs}

    total_gt_items = len(ann_map)
    if total_gt_items == 0:
        return {
            "status": "NOT_MEASURED",
            "count": 0,
            "chair_i": None,
            "chair_s": None,
            "hallucination_rate": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "missing_count": 0,
            "coverage_rate": 0.0,
            "total_mentioned_objects": 0,
            "total_hallucinated_objects": 0,
        }

    # Index predictions
    pred_map = {}
    for pred in predictions:
        sid = str(pred.get("id"))
        norm_sid = sid
        if norm_sid.startswith("amber__generative__"):
            norm_sid = norm_sid.replace("amber__generative__", "")
        pred_map[norm_sid] = pred.get("prediction") or pred.get("response") or ""

    active_vocab = set(DEFAULT_AMBER_OBJECTS)
    if object_vocabulary:
        active_vocab.update(str(o).lower().strip() for o in object_vocabulary)
    for objs in ann_map.values():
        for obj in objs:
            active_vocab.add(str(obj).lower().strip())

    hallucinated_descriptions = 0
    total_mentioned_objects = 0
    total_hallucinated_objects = 0
    total_gt_objects = 0
    missing_count = 0
    per_sample_details = []

    # Score over EVERY item in the frozen ground-truth annotations (DS-03)
    for sid, gt_objects in ann_map.items():
        total_gt_objects += len(gt_objects)

        if sid not in pred_map:
            # Missing prediction is an execution failure; separate from hallucination count
            missing_count += 1
            continue

        resp_text = str(pred_map[sid])
        candidate_mentions = extract_mentioned_objects(resp_text, active_vocab)
        gt_objects_normalized = {str(o).lower().strip() for o in gt_objects}

        mentioned_gt = {o for o in candidate_mentions if o in gt_objects_normalized}
        hallucinated = {o for o in candidate_mentions if o not in gt_objects_normalized}

        n_pred = len(candidate_mentions)
        n_hal = len(hallucinated)
        has_hallucination = bool(n_hal > 0)

        if has_hallucination:
            hallucinated_descriptions += 1
        total_mentioned_objects += n_pred
        total_hallucinated_objects += n_hal

        per_sample_details.append({
            "id": sid,
            "mentioned_gt": list(mentioned_gt),
            "hallucinated": list(hallucinated),
            "has_hallucination": has_hallucination,
        })

    chair_i = (
        total_hallucinated_objects / total_mentioned_objects
        if total_mentioned_objects > 0
        else None
    )
    evaluated_descriptions = total_gt_items - missing_count
    chair_s = (
        hallucinated_descriptions / evaluated_descriptions
        if evaluated_descriptions > 0
        else None
    )
    coverage_rate = evaluated_descriptions / total_gt_items

    correct_objects = total_mentioned_objects - total_hallucinated_objects
    precision = (
        correct_objects / total_mentioned_objects
        if total_mentioned_objects > 0
        else None
    )
    recall = (
        correct_objects / total_gt_objects
        if total_gt_objects > 0
        else None
    )
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else (0.0 if (precision == 0.0 or recall == 0.0) else None)
    )

    return {
        "status": "AMBER_GENERATIVE_PROXY_UNVERIFIED",  # DS-01
        "count": total_gt_items,
        "evaluated_count": evaluated_descriptions,
        "chair_i": chair_i,
        "chair_s": chair_s,
        "hallucination_rate": chair_s,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing_count": missing_count,
        "coverage_rate": coverage_rate,
        "total_mentioned_objects": total_mentioned_objects,
        "total_hallucinated_objects": total_hallucinated_objects,
        "details_sample_count": len(per_sample_details),
        "warning": "Proxy re-implementation; official upstream scorer parity not verified",
    }


def score_amber(
    predictions: list[dict[str, Any]],
    annotations: list[dict[str, Any]] | Path | str,
) -> dict[str, Any]:
    """Unified AMBER scoring combining generative and discriminative evaluation.

    DS-01: Status is AMBER_PROXY_UNVERIFIED.
    DS-03: No composite score unless BOTH tasks are present and evaluable.
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
        if not isinstance(item, dict):
            raise TypeError("Annotation items must be dictionaries")
        task = item.get("task")
        if not task:
            raise ValueError(f"Annotation item {item.get('id')} missing required 'task'")
        task = str(task).lower()
        if task == "generative":
            gen_ann.append(item)
        elif task == "discriminative":
            disc_ann.append(item)
        else:
            raise ValueError(f"Unknown task {task} in annotations")

    gen_result = score_amber_generative(predictions, gen_ann) if gen_ann else None
    disc_result = score_amber_discriminative(predictions, disc_ann) if disc_ann else None

    # DS-03: No composite score unless BOTH tasks are present and evaluable
    has_valid_gen = gen_result is not None and gen_result.get("chair_i") is not None
    has_valid_disc = disc_result is not None and disc_result.get("overall", {}).get("f1") is not None

    if has_valid_gen and has_valid_disc:
        chair_val = gen_result["chair_i"]
        disc_f1 = disc_result["overall"]["f1"]
        amber_score = 0.5 * ((1.0 - chair_val) + disc_f1)
        composite_status = "COMPOSITE_SCORE_CALCULATED"
    else:
        amber_score = None  # DS-03: Return None, never misleading default
        composite_status = "COMPOSITE_SCORE_UNAVAILABLE_REQUIRES_BOTH_TASKS"

    return {
        "status": "AMBER_PROXY_UNVERIFIED",  # DS-01
        "composite_status": composite_status,
        "amber_score": amber_score,
        "generative": gen_result,
        "discriminative": disc_result,
        "predictions_total": len(predictions),
        "warning": "Proxy re-implementation; official upstream scorer parity not verified",
    }


def export_amber_official_format(
    predictions: list[dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Export raw model predictions to AMBER official JSON format for upstream inference.py.

    DS-10: Exports raw responses without local parser rewriting.
    Format: [{"id": 1, "response": "..."}]
    """
    output_path = Path(output_path)
    exported = []
    for pred in predictions:
        sid = pred.get("id")
        numeric_matches = re.findall(r"\d+", str(sid))
        target_id = int(numeric_matches[-1]) if numeric_matches else sid
        # Export raw prediction text
        resp = str(pred.get("prediction") or pred.get("response") or "")
        exported.append({"id": target_id, "response": resp})

    atomic_json(output_path, exported)
    return output_path
