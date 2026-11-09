"""Tests for AMBER dataset adapter, normalizer, and proxy scoring."""
import json
from pathlib import Path

import pytest

from vlmbench.datasets.amber import (
    AMBER_UPSTREAM_MANIFEST,
    AmberError,
    export_amber_official_format,
    normalize_amber,
    parse_binary_yes_no,
    score_amber,
    score_amber_discriminative,
    score_amber_generative,
)
from vlmbench.workloads import ensure_no_gold_in_request, load_jsonl


def _create_amber_fixtures(tmp_path: Path):
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "AMBER_1.jpg").write_bytes(b"jpeg-bytes-1")
    (img_dir / "AMBER_2.jpg").write_bytes(b"jpeg-bytes-2")
    (img_dir / "AMBER_1005.jpg").write_bytes(b"jpeg-bytes-1005")
    (img_dir / "AMBER_1006.jpg").write_bytes(b"jpeg-bytes-1006")
    (img_dir / "AMBER_1007.jpg").write_bytes(b"jpeg-bytes-1007")

    items = [
        # Generative item 1
        {
            "id": 1,
            "image": "AMBER_1.jpg",
            "task": "generative",
            "query": "Describe this image in detail.",
            "objects": ["dog", "frisbee", "park"],
        },
        # Generative item 2
        {
            "id": 2,
            "image": "AMBER_2.jpg",
            "task": "generative",
            "query": "What is in the image?",
            "objects": ["cat", "sofa"],
        },
        # Discriminative - existence
        {
            "id": 1005,
            "image": "AMBER_1005.jpg",
            "task": "discriminative",
            "type": "existence",
            "query": "Is there a dog in the image?",
            "label": "Yes",
        },
        # Discriminative - attribute
        {
            "id": 1006,
            "image": "AMBER_1006.jpg",
            "task": "discriminative",
            "type": "attribute",
            "query": "Is the car red?",
            "label": "No",
        },
        # Discriminative - relation
        {
            "id": 1007,
            "image": "AMBER_1007.jpg",
            "task": "discriminative",
            "type": "relation",
            "query": "Is the cup on the table?",
            "label": "Yes",
        },
    ]
    source_file = tmp_path / "query_all.json"
    source_file.write_text(json.dumps(items), encoding="utf8")
    return source_file, img_dir, items


def test_normalize_amber_generative_and_discriminative(tmp_path):
    source, img_dir, _ = _create_amber_fixtures(tmp_path)
    output = tmp_path / "normalized_quality.jsonl"

    result = normalize_amber(
        source, img_dir, output,
        split="all",
        pass_type="natural_eos_quality",
    )
    assert result["dataset"] == "amber"
    assert result["count"] == 5
    # DS-01: Status is AMBER_PROXY_UNVERIFIED
    assert result["status"] == "AMBER_PROXY_UNVERIFIED"
    assert len(result["sha256_source"]) == 64
    assert len(result["sha256_normalized"]) == 64
    # DS-07: Upstream manifest recorded
    assert result["upstream_manifest"]["repo_id"] == "junyangwang0410/AMBER"
    assert len(result["upstream_manifest"]["pinned_commit"]) == 40

    records = load_jsonl(output, dataset="amber", track="native_quality")
    assert len(records) == 5

    # Check Qwen chat template formatting: roles 'user', multimodal parts
    gen_rec = records[0]
    assert gen_rec["id"] == "amber__generative__1"
    assert len(gen_rec["messages"]) == 1
    assert gen_rec["messages"][0]["role"] == "user"
    parts = gen_rec["messages"][0]["content"]
    assert parts[0]["type"] == "image_path"
    assert parts[0]["path"].endswith("AMBER_1.jpg")
    # DS-07: Per-record image_sha256 present
    assert len(parts[0]["image_sha256"]) == 64
    assert len(gen_rec["meta"]["image_sha256"]) == 64
    assert parts[1]["type"] == "text"
    assert "Describe this image in detail." in parts[1]["text"]

    # DS-01: official_scoring_eligible is False
    assert gen_rec["meta"]["official_scoring_eligible"] is False
    assert gen_rec["reference"] == '["dog", "frisbee", "park"]'
    ensure_no_gold_in_request(gen_rec, {"messages": gen_rec["messages"]})

    # Check discriminative record
    disc_rec = records[2]
    assert disc_rec["id"] == "amber__discriminative__1005"
    assert disc_rec["reference"] == "Yes"
    assert disc_rec["meta"]["dimension"] == "existence"
    assert disc_rec["meta"]["official_scoring_eligible"] is False


def test_amber_split_filtering(tmp_path):
    source, img_dir, _ = _create_amber_fixtures(tmp_path)

    # Filter generative only
    out_gen = tmp_path / "gen.jsonl"
    res_gen = normalize_amber(source, img_dir, out_gen, split="generative")
    assert res_gen["count"] == 2
    records_gen = load_jsonl(out_gen, dataset="amber", track="native_quality")
    assert all("generative" in r["id"] for r in records_gen)

    # Filter discriminative only
    out_disc = tmp_path / "disc.jsonl"
    res_disc = normalize_amber(source, img_dir, out_disc, split="discriminative")
    assert res_disc["count"] == 3
    records_disc = load_jsonl(out_disc, dataset="amber", track="native_quality")
    assert all("discriminative" in r["id"] for r in records_disc)

    # Filter by specific dimension (attribute)
    out_attr = tmp_path / "attr.jsonl"
    res_attr = normalize_amber(source, img_dir, out_attr, split="attribute")
    assert res_attr["count"] == 1
    records_attr = load_jsonl(out_attr, dataset="amber", track="native_quality")
    assert records_attr[0]["id"] == "amber__discriminative__1006"
    assert records_attr[0]["meta"]["dimension"] == "attribute"


def test_amber_pass_types_distinction_and_serving_track(tmp_path):
    source, img_dir, _ = _create_amber_fixtures(tmp_path)

    # Natural-EOS quality pass
    out_qual = tmp_path / "qual.jsonl"
    res_qual = normalize_amber(source, img_dir, out_qual, pass_type="natural_eos_quality")
    assert res_qual["status"] == "AMBER_PROXY_UNVERIFIED"
    rows_qual = load_jsonl(out_qual, dataset="amber", track="native_quality")
    assert rows_qual[0]["track"] == "native_quality"
    assert rows_qual[0]["native_context_complete"] is True
    assert rows_qual[0]["meta"]["official_scoring_eligible"] is False

    # Fixed serving pass (DS-06: track='amber_serving', no requested_text_length_bin)
    out_serv = tmp_path / "serv.jsonl"
    res_serv = normalize_amber(source, img_dir, out_serv, pass_type="fixed_serving")
    assert res_serv["status"] == "AMBER_SERVING_PREPARED"
    rows_serv = load_jsonl(out_serv, dataset="amber", track="amber_serving")
    assert rows_serv[0]["track"] == "amber_serving"
    assert rows_serv[0]["derived_from"] == "amber__generative__1"
    assert "requested_text_length_bin" not in rows_serv[0]
    assert "requested_text_length_bin" not in rows_serv[0]["meta"]
    assert rows_serv[0]["meta"]["official_quality_allowed"] is False
    assert len(rows_serv[0]["meta"]["image_sha256"]) == 64


def test_amber_strict_schema_rejections(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    (img_dir / "valid.jpg").write_bytes(b"image")

    # Missing image field (DS-02: must raise ValueError, no default to AMBER_{id}.jpg)
    src_no_img = tmp_path / "no_img.json"
    src_no_img.write_text(json.dumps([{"id": 1, "task": "generative", "query": "Q"}]))
    with pytest.raises(ValueError, match="missing required 'image'"):
        normalize_amber(src_no_img, img_dir, tmp_path / "out1.jsonl")

    # Missing query field (DS-02, DS-05: must raise ValueError, no default description prompt)
    src_no_query = tmp_path / "no_query.json"
    src_no_query.write_text(json.dumps([{"id": 1, "image": "valid.jpg", "task": "generative"}]))
    with pytest.raises(ValueError, match="missing required 'query'"):
        normalize_amber(src_no_query, img_dir, tmp_path / "out2.jsonl")

    # Missing task field (DS-02)
    src_no_task = tmp_path / "no_task.json"
    src_no_task.write_text(json.dumps([{"id": 1, "image": "valid.jpg", "query": "Q"}]))
    with pytest.raises(ValueError, match="missing or invalid required 'task'"):
        normalize_amber(src_no_task, img_dir, tmp_path / "out3.jsonl")

    # Discriminative item missing type (DS-02: Reproduction 4)
    src_no_type = tmp_path / "no_type.json"
    src_no_type.write_text(json.dumps([{
        "id": 1005, "image": "valid.jpg", "task": "discriminative",
        "query": "Is there a dog?", "label": "Yes",
    }]))
    with pytest.raises(ValueError, match="missing or invalid required 'type'"):
        normalize_amber(src_no_type, img_dir, tmp_path / "out4.jsonl")

    # Discriminative item missing label or label stored under 'truth' (DS-02: Reproduction 1)
    src_truth = tmp_path / "truth_key.json"
    src_truth.write_text(json.dumps([{
        "id": 1005, "image": "valid.jpg", "task": "discriminative",
        "type": "existence", "query": "Is there a dog?", "truth": "Yes",
    }]))
    with pytest.raises(ValueError, match="missing or invalid required 'label'"):
        normalize_amber(src_truth, img_dir, tmp_path / "out5.jsonl")

    # Generative item with objects stored under 'truth' (DS-02: Reproduction 5)
    src_gen_truth = tmp_path / "gen_truth.json"
    src_gen_truth.write_text(json.dumps([{
        "id": 1, "image": "valid.jpg", "task": "generative",
        "query": "Describe", "truth": ["dog"],
    }]))
    # Note: generative query can be normalized without objects, but scoring requires objects list
    norm_out = tmp_path / "out_gen.jsonl"
    normalize_amber(src_gen_truth, img_dir, norm_out)
    with pytest.raises(ValueError, match="missing or invalid required 'objects'"):
        score_amber_generative([{"id": "1", "prediction": "A dog"}], [{"id": "1", "truth": ["dog"]}])


def test_score_amber_discriminative_denominators_and_failures():
    # DS-03: Reproduction 2 & 3
    # 3 ground truth items
    gt = [
        {"id": "1005", "label": "Yes", "type": "existence"},
        {"id": "1006", "label": "No", "type": "attribute"},
        {"id": "1007", "label": "Yes", "type": "relation"},
    ]

    # Only 1 answered correctly (1005), 1 unparseable (1006), 1 missing (1007)
    predictions = [
        {"id": "amber__discriminative__1005", "prediction": "Yes"},
        {"id": "amber__discriminative__1006", "prediction": "I don't know"},
    ]

    scores = score_amber_discriminative(predictions, gt)
    assert scores["status"] == "AMBER_DISCRIMINATIVE_PROXY_UNVERIFIED"

    # All 3 ground truth items MUST be in the denominator!
    # Correct: 1 (1005). Unparseable: 1 (1006). Missing: 1 (1007).
    # Accuracy = 1 / 3 = 0.3333... (NOT 1.0!)
    assert scores["overall"]["count"] == 3
    assert pytest.approx(scores["overall"]["accuracy"], rel=1e-3) == 1.0 / 3.0
    assert scores["overall"]["unparseable_count"] == 1
    assert scores["overall"]["missing_count"] == 1


def test_score_amber_composite_requires_both_tasks():
    # DS-03: Reproduction 7
    # Only discriminative items provided
    gt_disc_only = [
        {"id": 1005, "task": "discriminative", "label": "Yes", "type": "existence"},
    ]
    preds = [
        {"id": "amber__discriminative__1005", "prediction": "Yes"},
    ]

    report = score_amber(preds, gt_disc_only)
    assert report["status"] == "AMBER_PROXY_UNVERIFIED"
    assert report["composite_status"] == "COMPOSITE_SCORE_UNAVAILABLE_REQUIRES_BOTH_TASKS"
    # amber_score MUST be None, NEVER default to 1.0 or 0.0
    assert report["amber_score"] is None


def test_score_amber_unified_when_both_tasks_present():
    annotations = [
        {"id": 1, "task": "generative", "objects": ["dog", "grass"]},
        {"id": 1005, "task": "discriminative", "label": "Yes", "type": "existence"},
    ]
    predictions = [
        {"id": "amber__generative__1", "prediction": "A dog on green grass."},
        {"id": "amber__discriminative__1005", "prediction": "Yes."},
    ]

    report = score_amber(predictions, annotations)
    assert report["status"] == "AMBER_PROXY_UNVERIFIED"
    assert report["composite_status"] == "COMPOSITE_SCORE_CALCULATED"
    assert report["generative"]["chair_s"] == 0.0
    assert report["discriminative"]["overall"]["accuracy"] == 1.0
    assert report["amber_score"] == 1.0


def test_score_amber_multi_word_objects():
    # Multi-word objects like 'traffic light'
    ann = [
        {"id": "1", "objects": ["traffic light", "car"]},
        {"id": "2", "objects": ["dog"]},
    ]
    preds = [
        # Correctly mentions traffic light and car
        {"id": "1", "prediction": "There is a red traffic light and a white car."},
        # Mentions dog (correct) and traffic light (hallucinated)
        {"id": "2", "prediction": "A dog sitting next to a traffic light."},
    ]
    scores = score_amber_generative(preds, ann)
    assert scores["status"] == "AMBER_GENERATIVE_PROXY_UNVERIFIED"
    assert scores["count"] == 2
    # Sample 1: 0 hallucinations. Sample 2: 1 hallucination ('traffic light')
    assert scores["chair_s"] == 0.5
    assert scores["total_mentioned_objects"] == 4  # traffic light, car, dog, traffic light
    assert scores["total_hallucinated_objects"] == 1  # traffic light in sample 2


def test_export_amber_official_format_raw_responses(tmp_path):
    # DS-10: Export raw predictions without parser rewriting
    preds = [
        {"id": "amber__generative__1", "prediction": "A description text."},
        {"id": "amber__discriminative__1005", "prediction": "Definitely Yes!"},
    ]
    export_file = tmp_path / "raw_preds.json"
    export_amber_official_format(preds, export_file)
    exported = json.loads(export_file.read_text(encoding="utf8"))
    assert len(exported) == 2
    assert exported[0]["id"] == 1
    assert exported[0]["response"] == "A description text."
    assert exported[1]["id"] == 1005
    assert exported[1]["response"] == "Definitely Yes!"  # Raw, not modified


def test_amber_upstream_manifest_and_parse_helpers():
    # DS-07: Verify pinned upstream manifest
    assert AMBER_UPSTREAM_MANIFEST["repo_id"] == "junyangwang0410/AMBER"
    assert AMBER_UPSTREAM_MANIFEST["license"] == "Apache-2.0"
    assert len(AMBER_UPSTREAM_MANIFEST["pinned_commit"]) == 40

    # Test binary yes/no parsing helper
    assert parse_binary_yes_no("Yes") == "Yes"
    assert parse_binary_yes_no("no, there is none") == "No"
    assert parse_binary_yes_no("I cannot tell") is None

    # Test AmberError is a ValueError
    assert issubclass(AmberError, ValueError)


def test_amber_dict_ground_truth_strict_validation():
    # DS-02 P0: Dictionary ground truth must not bypass schema validation
    # Invalid label
    with pytest.raises(ValueError, match="missing or invalid required 'label'"):
        score_amber_discriminative(
            [{"id": "1", "prediction": "No"}],
            {"1": {"label": "INVALID", "dimension": "existence"}},
        )

    # Missing dimension/type
    with pytest.raises(ValueError, match="missing or invalid required 'type'"):
        score_amber_discriminative(
            [{"id": "1", "prediction": "Yes"}],
            {"1": {"label": "Yes"}},
        )

    # Invalid entry type (not a dict)
    with pytest.raises(TypeError, match="value must be a dictionary"):
        score_amber_discriminative(
            [{"id": "1", "prediction": "Yes"}],
            {"1": "Yes"},
        )


def test_amber_empty_or_bad_root_schema_rejected(tmp_path):
    # DS-02: normalize_amber must reject empty root {} or unsupported wrappers
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)

    # Empty root object {}
    empty_src = tmp_path / "empty_root.json"
    empty_src.write_text("{}", encoding="utf8")
    with pytest.raises(ValueError, match="dict must contain one of"):
        normalize_amber(empty_src, img_dir, tmp_path / "out1.jsonl")

    # Unsupported wrapper key
    bad_wrapper = tmp_path / "bad_wrapper.json"
    bad_wrapper.write_text(json.dumps({"unknown_wrapper": []}), encoding="utf8")
    with pytest.raises(ValueError, match="dict must contain one of"):
        normalize_amber(bad_wrapper, img_dir, tmp_path / "out2.jsonl")

    # Empty list under valid wrapper
    empty_list_src = tmp_path / "empty_list.json"
    empty_list_src.write_text(json.dumps({"queries": []}), encoding="utf8")
    with pytest.raises(ValueError, match="must be a non-empty list"):
        normalize_amber(empty_list_src, img_dir, tmp_path / "out3.jsonl")


def test_amber_missing_and_unparseable_positives_in_recall_and_composite():
    # DS-03 P1: Two positive ground-truth items
    gt = [
        {"id": "1", "label": "Yes", "type": "existence"},
        {"id": "2", "label": "Yes", "type": "existence"},
    ]

    # Case A: One answered 'Yes', one missing
    preds_missing = [{"id": "1", "prediction": "Yes"}]
    scores_a = score_amber_discriminative(preds_missing, gt)
    assert scores_a["overall"]["count"] == 2
    assert scores_a["overall"]["missing_count"] == 1
    assert scores_a["overall"]["accuracy"] == 0.5
    assert scores_a["overall"]["precision"] == 1.0
    # Recall MUST be 0.5 (1 TP, 1 FN due to missing), not 1.0!
    assert scores_a["overall"]["recall"] == 0.5
    # F1 = 2 * 1.0 * 0.5 / 1.5 = 2/3 ~ 0.6667
    assert pytest.approx(scores_a["overall"]["f1"], rel=1e-3) == 2.0 / 3.0

    # Case B: One answered 'Yes', one unparseable
    preds_unparseable = [
        {"id": "1", "prediction": "Yes"},
        {"id": "2", "prediction": "I don't know"},
    ]
    scores_b = score_amber_discriminative(preds_unparseable, gt)
    assert scores_b["overall"]["count"] == 2
    assert scores_b["overall"]["unparseable_count"] == 1
    assert scores_b["overall"]["accuracy"] == 0.5
    assert scores_b["overall"]["precision"] == 1.0
    assert scores_b["overall"]["recall"] == 0.5
    assert pytest.approx(scores_b["overall"]["f1"], rel=1e-3) == 2.0 / 3.0

    # Case C: Check composite score with a generative item
    ann_unified = [
        {"id": 1, "task": "generative", "objects": ["dog"]},
        {"id": 1005, "task": "discriminative", "label": "Yes", "type": "existence"},
        {"id": 1006, "task": "discriminative", "label": "Yes", "type": "existence"},
    ]
    preds_unified = [
        {"id": "amber__generative__1", "prediction": "A cute dog."},
        {"id": "amber__discriminative__1005", "prediction": "Yes"},
        # 1006 is missing
    ]
    report = score_amber(preds_unified, ann_unified)
    assert report["composite_status"] == "COMPOSITE_SCORE_CALCULATED"
    # CHAIR_i = 0.0 (perfect generative). Discriminative F1 = 2/3.
    # amber_score = 0.5 * ((1.0 - 0.0) + 2/3) ~ 0.8333, strictly NOT 1.0!
    assert pytest.approx(report["amber_score"], rel=1e-3) == 0.5 * (1.0 + 2.0 / 3.0)


def test_score_amber_generative_missing_does_not_count_as_hallucination():
    # DS-03: Missing prediction in generative must not be counted as hallucinated
    ann = [
        {"id": "1", "objects": ["dog"]},
        {"id": "2", "objects": ["cat"]},
    ]
    # Sample 1 answered with 0 hallucinations; Sample 2 is missing
    preds = [{"id": "1", "prediction": "A dog."}]
    scores = score_amber_generative(preds, ann)
    assert scores["count"] == 2
    assert scores["missing_count"] == 1
    assert scores["evaluated_count"] == 1
    assert scores["coverage_rate"] == 0.5
    # Evaluated sample has 0 hallucinations -> chair_s is 0.0, NOT 0.5!
    assert scores["chair_s"] == 0.0
    assert scores["hallucination_rate"] == 0.0


def test_score_amber_discriminative_mathematical_edge_cases():
    # Mathematical edge cases audit:
    # 1. Zero TP (Positives exist, predicted No)
    gt_zero_tp = [{"id": "1", "label": "Yes", "type": "existence"}]
    preds_zero_tp = [{"id": "1", "prediction": "No"}]
    res_zero_tp = score_amber_discriminative(preds_zero_tp, gt_zero_tp)
    assert res_zero_tp["overall"]["tp"] == 0
    assert res_zero_tp["overall"]["fn"] == 1
    assert res_zero_tp["overall"]["precision"] is None  # No positive predictions made
    assert res_zero_tp["overall"]["recall"] == 0.0
    assert res_zero_tp["overall"]["f1"] == 0.0  # Positives existed but none found
    assert res_zero_tp["overall"]["accuracy"] == 0.0

    # 2. Zero FP (Positives answered Yes, negatives answered No)
    gt_zero_fp = [
        {"id": "1", "label": "Yes", "type": "existence"},
        {"id": "2", "label": "No", "type": "existence"},
    ]
    preds_zero_fp = [
        {"id": "1", "prediction": "Yes"},
        {"id": "2", "prediction": "No"},
    ]
    res_zero_fp = score_amber_discriminative(preds_zero_fp, gt_zero_fp)
    assert res_zero_fp["overall"]["fp"] == 0
    assert res_zero_fp["overall"]["precision"] == 1.0
    assert res_zero_fp["overall"]["recall"] == 1.0
    assert res_zero_fp["overall"]["f1"] == 1.0
    assert res_zero_fp["overall"]["hallucination_rate"] == 0.0
    assert res_zero_fp["overall"]["accuracy"] == 1.0

    # 3. Zero TN (Negatives exist, but all predicted Yes -> all hallucinated)
    gt_zero_tn = [{"id": "1", "label": "No", "type": "existence"}]
    preds_zero_tn = [{"id": "1", "prediction": "Yes"}]
    res_zero_tn = score_amber_discriminative(preds_zero_tn, gt_zero_tn)
    assert res_zero_tn["overall"]["tn"] == 0
    assert res_zero_tn["overall"]["fp"] == 1
    assert res_zero_tn["overall"]["hallucination_rate"] == 1.0
    assert res_zero_tn["overall"]["accuracy"] == 0.0

    # 4. Zero FN (Positives exist, all predicted Yes)
    gt_zero_fn = [{"id": "1", "label": "Yes", "type": "existence"}]
    preds_zero_fn = [{"id": "1", "prediction": "Yes"}]
    res_zero_fn = score_amber_discriminative(preds_zero_fn, gt_zero_fn)
    assert res_zero_fn["overall"]["fn"] == 0
    assert res_zero_fn["overall"]["recall"] == 1.0
    assert res_zero_fn["overall"]["f1"] == 1.0

    # 5. All Missing (1 Yes, 1 No, zero predictions provided)
    gt_all_missing = [
        {"id": "1", "label": "Yes", "type": "existence"},
        {"id": "2", "label": "No", "type": "existence"},
    ]
    res_all_missing = score_amber_discriminative([], gt_all_missing)
    assert res_all_missing["overall"]["count"] == 2
    assert res_all_missing["overall"]["missing_count"] == 2
    assert res_all_missing["overall"]["accuracy"] == 0.0
    assert res_all_missing["overall"]["fn"] == 1  # Missing positive counts as FN
    assert res_all_missing["overall"]["recall"] == 0.0
    assert res_all_missing["overall"]["f1"] == 0.0

    # 6. All Unparseable (1 Yes, 1 No, non-binary text answers)
    preds_all_unparseable = [
        {"id": "1", "prediction": "Gibberish"},
        {"id": "2", "prediction": "Also gibberish"},
    ]
    res_all_unparseable = score_amber_discriminative(preds_all_unparseable, gt_all_missing)
    assert res_all_unparseable["overall"]["count"] == 2
    assert res_all_unparseable["overall"]["unparseable_count"] == 2
    assert res_all_unparseable["overall"]["accuracy"] == 0.0
    assert res_all_unparseable["overall"]["fn"] == 1  # Unparseable positive counts as FN
    assert res_all_unparseable["overall"]["recall"] == 0.0
    assert res_all_unparseable["overall"]["f1"] == 0.0


def test_score_amber_generative_all_hallucinated_edge_case():
    # Model mentions 2 objects, but both are hallucinations
    ann = [{"id": "1", "objects": ["dog"]}]
    preds = [{"id": "1", "prediction": "A red car and a bicycle."}]
    scores = score_amber_generative(preds, ann)
    assert scores["count"] == 1
    assert scores["chair_i"] == 1.0  # 2/2 objects hallucinated
    assert scores["chair_s"] == 1.0
    assert scores["precision"] == 0.0  # 0/2 correct
    assert scores["recall"] == 0.0  # 0/1 gt objects covered
    assert scores["f1"] == 0.0  # Must be 0.0, strictly not None!



