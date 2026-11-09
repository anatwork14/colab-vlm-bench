"""Tests for AMBER dataset adapter, normalizer, and official scoring."""
import json
from pathlib import Path

import pytest

from vlmbench.datasets.amber import (
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
        # Generative item
        {
            "id": 1,
            "image": "AMBER_1.jpg",
            "task": "generative",
            "type": "all",
            "query": "Describe this image in detail.",
            "objects": ["dog", "frisbee", "park"],
        },
        # Generative item 2
        {
            "id": 2,
            "image": "AMBER_2.jpg",
            "task": "generative",
            "type": "all",
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
    assert result["status"] == "NORMALIZED_OFFICIAL_SCORING_ELIGIBLE"
    assert len(result["sha256_source"]) == 64
    assert len(result["sha256_normalized"]) == 64

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
    assert parts[1]["type"] == "text"
    assert "Describe this image in detail." in parts[1]["text"]

    # Gold reference isolated from prompt messages
    assert gen_rec["reference"] == '["dog", "frisbee", "park"]'
    ensure_no_gold_in_request(gen_rec, {"messages": gen_rec["messages"]})

    # Check discriminative record
    disc_rec = records[2]
    assert disc_rec["id"] == "amber__discriminative__1005"
    assert disc_rec["reference"] == "Yes"
    assert disc_rec["meta"]["dimension"] == "existence"
    assert disc_rec["meta"]["pass_type"] == "natural_eos_quality"
    assert disc_rec["meta"]["max_output_tokens"] == 256
    assert "<|im_end|>" in disc_rec["meta"]["stop_sequences"]


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


def test_amber_pass_types_distinction(tmp_path):
    source, img_dir, _ = _create_amber_fixtures(tmp_path)

    # Natural-EOS quality pass
    out_qual = tmp_path / "qual.jsonl"
    res_qual = normalize_amber(source, img_dir, out_qual, pass_type="natural_eos_quality")
    assert res_qual["status"] == "NORMALIZED_OFFICIAL_SCORING_ELIGIBLE"
    rows_qual = load_jsonl(out_qual, dataset="amber", track="native_quality")
    assert rows_qual[0]["track"] == "native_quality"
    assert rows_qual[0]["native_context_complete"] is True
    assert rows_qual[0]["meta"]["official_scoring_eligible"] is True

    # Fixed serving pass
    out_serv = tmp_path / "serv.jsonl"
    res_serv = normalize_amber(source, img_dir, out_serv, pass_type="fixed_serving")
    assert res_serv["status"] == "NORMALIZED_DERIVED_SERVING_WORKLOAD"
    rows_serv = load_jsonl(out_serv, dataset="amber", track="derived_serving")
    assert rows_serv[0]["track"] == "derived_serving"
    assert rows_serv[0]["derived_from"] == "amber__generative__1"
    assert rows_serv[0]["meta"]["pass_type"] == "fixed_serving"
    assert rows_serv[0]["meta"]["fixed_output_tokens"] == 256
    assert rows_serv[0]["meta"]["ignore_eos"] is True
    assert rows_serv[0]["meta"]["official_quality_allowed"] is False


def test_amber_missing_image_guard(tmp_path):
    source = tmp_path / "broken.json"
    source.write_text(json.dumps([{
        "id": 999,
        "image": "nonexistent.jpg",
        "task": "discriminative",
        "query": "Is there a cat?",
        "label": "Yes",
    }]))
    with pytest.raises(AmberError, match="Missing or escaping"):
        normalize_amber(source, tmp_path, tmp_path / "out.jsonl")


def test_parse_binary_yes_no():
    assert parse_binary_yes_no("Yes") == "Yes"
    assert parse_binary_yes_no("yes, there is a cat.") == "Yes"
    assert parse_binary_yes_no("No.") == "No"
    assert parse_binary_yes_no("no, absolutely not.") == "No"
    assert parse_binary_yes_no("Definitely yes!") == "Yes"
    assert parse_binary_yes_no("I cannot tell") is None
    assert parse_binary_yes_no("") is None


def test_score_amber_discriminative_dimensions():
    ground_truth = [
        {"id": "1005", "label": "Yes", "type": "existence"},
        {"id": "1006", "label": "No", "type": "attribute"},
        {"id": "1007", "label": "Yes", "type": "relation"},
        {"id": "1008", "label": "No", "type": "existence"},
    ]
    # Predictions: 3 correct, 1 wrong (1008 predicted Yes instead of No)
    predictions = [
        {"id": "amber__discriminative__1005", "prediction": "Yes, there is a dog."},
        {"id": "amber__discriminative__1006", "prediction": "No, it is blue."},
        {"id": "amber__discriminative__1007", "prediction": "Yes, on the table."},
        {"id": "amber__discriminative__1008", "prediction": "Yes."},  # False positive -> hallucination
    ]

    scores = score_amber_discriminative(predictions, ground_truth)
    assert scores["status"] == "AMBER_DISCRIMINATIVE_SCORED"

    # Overall: 3/4 = 75% accuracy
    assert scores["overall"]["accuracy"] == 0.75
    assert scores["overall"]["count"] == 4

    # Existence: 1 TP (1005), 1 FP (1008) -> Acc = 0.5, Hallucination Rate = 1.0 (FP/(FP+TN))
    exist = scores["dimensions"]["existence"]
    assert exist["count"] == 2
    assert exist["accuracy"] == 0.5
    assert exist["hallucination_rate"] == 1.0

    # Attribute: 1 TN (1006) -> Acc = 1.0, Hallucination Rate = 0.0
    attr = scores["dimensions"]["attribute"]
    assert attr["count"] == 1
    assert attr["accuracy"] == 1.0
    assert attr["hallucination_rate"] == 0.0

    # Relation: 1 TP (1007) -> Acc = 1.0
    rel = scores["dimensions"]["relation"]
    assert rel["count"] == 1
    assert rel["accuracy"] == 1.0


def test_score_amber_generative_chair():
    annotations = [
        {"id": "1", "objects": ["dog", "frisbee", "park"]},
        {"id": "2", "objects": ["cat", "sofa"]},
    ]
    predictions = [
        # Sample 1: mentions dog, frisbee, park -> no hallucination
        {"id": "amber__generative__1", "prediction": "A happy dog catches a frisbee in a sunny park."},
        # Sample 2: mentions cat, car, bicycle -> mentions cat (correct), car & bicycle (hallucinated)
        {"id": "amber__generative__2", "prediction": "A cute cat next to a car and bicycle."},
    ]

    scores = score_amber_generative(predictions, annotations)
    assert scores["status"] == "AMBER_GENERATIVE_SCORED"
    assert scores["count"] == 2

    # Sample 1: 0 hallucinations. Sample 2: 1 hallucinated response.
    # CHAIR_S / Ha_rate = 1/2 = 0.5
    assert scores["chair_s"] == 0.5
    assert scores["hallucination_rate"] == 0.5
    assert scores["chair_i"] > 0.0
    assert scores["precision"] > 0.0
    assert scores["recall"] > 0.0


def test_score_amber_unified_and_export(tmp_path):
    annotations = [
        {"id": 1, "task": "generative", "objects": ["dog", "grass"]},
        {"id": 1005, "task": "discriminative", "label": "Yes", "type": "existence"},
    ]
    predictions = [
        {"id": "amber__generative__1", "prediction": "A dog on green grass."},
        {"id": "amber__discriminative__1005", "prediction": "Yes."},
    ]

    report = score_amber(predictions, annotations)
    assert report["status"] == "AMBER_OFFICIAL_SCORED"
    assert report["generative"]["chair_s"] == 0.0  # Perfect generation
    assert report["discriminative"]["overall"]["accuracy"] == 1.0  # Perfect discriminative
    assert report["amber_score"] == 1.0  # 0.5 * ((1.0 - 0.0) + 1.0) = 1.0

    # Test export to official AMBER format
    export_file = tmp_path / "amber_official_preds.json"
    export_amber_official_format(predictions, export_file)
    exported_data = json.loads(export_file.read_text(encoding="utf8"))
    assert len(exported_data) == 2
    assert exported_data[0]["id"] == 1
    assert exported_data[0]["response"] == "A dog on green grass."
    assert exported_data[1]["id"] == 1005
    assert exported_data[1]["response"] == "Yes"


def test_amber_validation_errors(tmp_path):
    source, img_dir, _ = _create_amber_fixtures(tmp_path)

    # Invalid split
    with pytest.raises(AmberError, match="Unsupported AMBER split"):
        normalize_amber(source, img_dir, tmp_path / "out.jsonl", split="invalid_split")

    # Invalid pass_type
    with pytest.raises(AmberError, match="Unsupported pass_type"):
        normalize_amber(source, img_dir, tmp_path / "out.jsonl", pass_type="invalid_pass")

    # Limit < 1
    with pytest.raises(ValueError, match="Limit must be positive"):
        normalize_amber(source, img_dir, tmp_path / "out.jsonl", limit=0)

    # Path traversal attack in image reference
    source_traversal = tmp_path / "traversal.json"
    source_traversal.write_text(json.dumps([{
        "id": 9999,
        "image": "../../../etc/passwd",
        "task": "discriminative",
        "query": "Attack?",
        "label": "Yes",
    }]))
    with pytest.raises(AmberError, match="Missing or escaping"):
        normalize_amber(source_traversal, img_dir, tmp_path / "out.jsonl")


def test_score_amber_empty_and_unparsed_predictions():
    # Empty predictions
    empty_disc = score_amber_discriminative([], [{"id": "1", "label": "Yes"}])
    assert empty_disc["overall"]["count"] == 0

    empty_gen = score_amber_generative([], [{"id": "1", "objects": ["dog"]}])
    assert empty_gen["count"] == 0
    assert empty_gen["chair_i"] == 0.0

    # Unparsed / non-binary predictions
    unparsed_preds = [
        {"id": "amber__discriminative__1005", "prediction": "I am not capable of answering this."},
    ]
    gt = [{"id": "1005", "label": "Yes", "type": "existence"}]
    scores = score_amber_discriminative(unparsed_preds, gt)
    assert scores["overall"]["count"] == 1
    assert scores["overall"]["unparsed_rate"] == 1.0
    assert scores["overall"]["accuracy"] == 0.0

