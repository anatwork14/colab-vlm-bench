"""Tests for Google Drive asset qualification module and CLI."""
from __future__ import annotations

import json
import math
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from vlmbench.datasets.amber import AMBER_UPSTREAM_MANIFEST
from vlmbench.qualification import (
    DeterministicSimpleTokenizer,
    QualificationError,
    get_qualification_tokenizer,
    qualify_all_drive_assets,
    qualify_amber_assets,
    qualify_memlens_assets,
    qualify_qwen_models,
    snapshot_directory_state,
)


def _create_mock_amber_zip(archive_path: Path) -> str:
    """Create a minimal valid AMBER.zip with queries, annotations, and dummy images."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    query_all = [
        {
            "id": 1,
            "image": "AMBER_1.jpg",
            "query": "Is there a dog in this image?",
            "task": "discriminative",
            "type": "existence",
        },
        {
            "id": 2,
            "image": "AMBER_2.jpg",
            "query": "Please provide a detailed description of this image.",
            "task": "generative",
        },
    ]
    query_gen = [query_all[1]]
    query_disc = [query_all[0]]
    annotations = {
        "annotations": [
            {
                "id": 1,
                "task": "discriminative",
                "type": "existence",
                "label": "Yes",
            },
            {
                "id": 2,
                "task": "generative",
                "objects": ["dog", "tree"],
            },
        ]
    }

    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("data/query/query_all.json", json.dumps(query_all))
        zf.writestr("data/query/query_generative.json", json.dumps(query_gen))
        zf.writestr("data/query/query_discriminative.json", json.dumps(query_disc))
        zf.writestr("data/annotation/annotations.json", json.dumps(annotations))
        zf.writestr("image/AMBER_1.jpg", b"\xff\xd8\xff\xe0mock_image_bytes_1\xff\xd9")
        zf.writestr("image/AMBER_2.jpg", b"\xff\xd8\xff\xe0mock_image_bytes_2\xff\xd9")

    return str(archive_path)


def _create_mock_memlens_tree(dataset_dir: Path) -> Path:
    """Create a minimal valid MemLens dataset tree."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    needle_dir = dataset_dir / "release_images" / "needle_images"
    haystack_dir = dataset_dir / "release_images" / "haystack_images"
    needle_dir.mkdir(parents=True, exist_ok=True)
    haystack_dir.mkdir(parents=True, exist_ok=True)

    img_needle = needle_dir / "test_needle.jpg"
    img_needle.write_bytes(b"\xff\xd8\xff\xe0needle\xff\xd9")

    raw_item = {
        "question_id": "ml_test_100",
        "question_type": "single_needle",
        "question": "What is the secret word in session 1?",
        "answer": "STARLIGHT",
        "haystack_session_ids": ["sess_1", "sess_2"],
        "haystack_dates": ["2026-01-01", "2026-01-02"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "Hello, here is an image with info.",
                    "images": [{"file": "needle_images/test_needle.jpg"}],
                },
                {"role": "assistant", "content": "The secret code is STARLIGHT."},
            ],
            [
                {"role": "user", "content": "Just extra conversation history text " * 50},
                {"role": "assistant", "content": "Acknowledged additional history " * 50},
            ],
        ],
    }

    dataset_json = dataset_dir / "dataset_32k.json"
    dataset_json.write_text(json.dumps([raw_item]), encoding="utf8")
    return dataset_json


def _create_mock_model_tree(model_dir: Path, name: str) -> Path:
    """Create a mock Qwen model checkpoint with config, tokenizer, and safetensors."""
    m_path = model_dir / name
    m_path.mkdir(parents=True, exist_ok=True)

    config = {
        "model_type": "qwen2",
        "architectures": ["Qwen2ForCausalLM"],
        "vocab_size": 152064,
    }
    (m_path / "config.json").write_text(json.dumps(config), encoding="utf8")
    tok_json = {
        "version": "1.0",
        "added_tokens": [],
        "model": {"type": "BPE", "vocab": {}, "merges": []},
    }
    (m_path / "tokenizer.json").write_text(json.dumps(tok_json), encoding="utf8")
    (m_path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<|im_start|>"}), encoding="utf8")
    (m_path / "model.safetensors").write_bytes(b"mock_safetensors_shard_bytes_12345")
    return m_path


def test_qualify_amber_assets_valid(tmp_path):
    zip_path = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    _create_mock_amber_zip(zip_path)

    res = qualify_amber_assets(tmp_path)
    assert res["status"] == "QUALIFIED_AMBER_SOURCE"
    assert res["crc_check"] == "PASSED"
    assert res["archive_sha256"]
    assert res["image_file_count"] == 2
    assert res["schema_compatibility"]["normalize_amber_verified"] is True
    assert res["schema_compatibility"]["sample_normalized_count"] == 2
    assert res["upstream_manifest_pin"] == AMBER_UPSTREAM_MANIFEST
    assert res["official_scoring_eligible"] is False
    assert "AMBER_PROXY_UNVERIFIED" in res["scorer_qualification"]


def test_qualify_amber_assets_corrupted_zip(tmp_path):
    zip_path = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"PK\x03\x04corrupted_zip_payload_not_real_zip")

    res = qualify_amber_assets(tmp_path)
    assert res["status"] == "UNQUALIFIED_AMBER_SOURCE"
    assert "error" in res


def test_qualify_amber_assets_missing_or_empty(tmp_path):
    # Non-existent archive
    res = qualify_amber_assets(tmp_path)
    assert res["status"] == "UNQUALIFIED_AMBER_SOURCE"
    assert "not found" in res["error"]

    # Empty 0-byte archive
    zip_path = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.write_bytes(b"")
    res_empty = qualify_amber_assets(tmp_path)
    assert res_empty["status"] == "UNQUALIFIED_AMBER_SOURCE"
    assert "empty" in res_empty["error"]


def test_qualify_amber_assets_missing_required_json(tmp_path):
    zip_path = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/query/query_all.json", "[]")

    res = qualify_amber_assets(tmp_path)
    assert res["status"] == "UNQUALIFIED_AMBER_SOURCE"
    assert "Missing required AMBER files in zip" in res["error"]


def test_qualify_memlens_assets_valid(tmp_path):
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    _create_mock_memlens_tree(memlens_dir)

    res = qualify_memlens_assets(tmp_path, tolerance=0.05)
    assert res["status"] == "QUALIFIED_MEMLENS_SOURCE"
    assert res["dataset_sha256"]
    assert res["sample_question_ids"] == ["ml_test_100"]
    assert res["image_directories"]["needle_images_present"] is True
    assert res["image_directories"]["haystack_images_present"] is True
    assert res["schema_compatibility"]["normalize_memlens_verified"] is True
    assert res["schema_compatibility"]["context_bin_generation_verified"] is True
    assert res["schema_compatibility"]["tolerance_bounded_lte_0_05"] is True


def test_qualify_memlens_assets_tolerance_boundary_enforcement(tmp_path):
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    _create_mock_memlens_tree(memlens_dir)

    # tolerance > 0.05 must raise ValueError
    with pytest.raises(ValueError, match=r"tolerance must be strictly in \(0.0, 0.05\]"):
        qualify_memlens_assets(tmp_path, tolerance=0.06)

    with pytest.raises(ValueError, match=r"tolerance must be strictly in \(0.0, 0.05\]"):
        qualify_memlens_assets(tmp_path, tolerance=0.50)

    # tolerance <= 0 must raise ValueError
    with pytest.raises(ValueError, match=r"tolerance must be strictly in \(0.0, 0.05\]"):
        qualify_memlens_assets(tmp_path, tolerance=0.0)

    # NaN / Inf must raise ValueError
    with pytest.raises(ValueError, match="tolerance must be a finite float"):
        qualify_memlens_assets(tmp_path, tolerance=math.nan)

    with pytest.raises(ValueError, match="tolerance must be a finite float"):
        qualify_memlens_assets(tmp_path, tolerance=math.inf)


def test_qualify_memlens_assets_missing_or_corrupt(tmp_path):
    res_missing = qualify_memlens_assets(tmp_path)
    assert res_missing["status"] == "UNQUALIFIED_MEMLENS_SOURCE"
    assert "not found" in res_missing["error"]

    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    memlens_dir.mkdir(parents=True, exist_ok=True)
    json_path = memlens_dir / "dataset_32k.json"
    json_path.write_text("corrupted json not valid [", encoding="utf8")

    res_corrupt = qualify_memlens_assets(tmp_path)
    assert res_corrupt["status"] == "UNQUALIFIED_MEMLENS_SOURCE"
    assert "Failed to parse" in res_corrupt["error"]


def test_qualify_qwen_models_valid(tmp_path):
    models_dir = tmp_path / "LLM_Weights/models/Qwen"
    _create_mock_model_tree(models_dir, "Qwen3.5-0.8B")
    _create_mock_model_tree(models_dir, "Qwen3.5-4B")

    res = qualify_qwen_models(tmp_path)
    assert res["status"] == "QUALIFIED_MODEL_SOURCES"
    assert "Qwen3.5-0.8B" in res["models"]
    assert "Qwen3.5-4B" in res["models"]
    assert res["models"]["Qwen3.5-0.8B"]["status"] == "QUALIFIED_MODEL_SOURCE"
    assert res["models"]["Qwen3.5-0.8B"]["config_sha256"]
    assert res["models"]["Qwen3.5-0.8B"]["weight_shards_count"] == 1


def test_qualify_qwen_models_missing_shards(tmp_path):
    models_dir = tmp_path / "LLM_Weights/models/Qwen"
    p = _create_mock_model_tree(models_dir, "Qwen3.5-0.8B")
    # Remove safetensors shard
    (p / "model.safetensors").unlink()

    res = qualify_qwen_models(tmp_path, model_names=("Qwen3.5-0.8B",))
    assert res["status"] == "UNQUALIFIED_MODEL_SOURCES"
    assert "No weight shards" in res["models"]["Qwen3.5-0.8B"]["error"]


def test_qualify_all_drive_assets_complete_and_read_only(tmp_path):
    # Setup complete mock Drive tree
    amber_zip = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    _create_mock_amber_zip(amber_zip)

    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    _create_mock_memlens_tree(memlens_dir)

    models_dir = tmp_path / "LLM_Weights/models/Qwen"
    _create_mock_model_tree(models_dir, "Qwen3.5-0.8B")
    _create_mock_model_tree(models_dir, "Qwen3.5-4B")

    before_snapshot = snapshot_directory_state(tmp_path)

    res = qualify_all_drive_assets(tmp_path, tolerance=0.05)
    assert res["status"] == "ALL_QUALIFIED"
    assert res["read_only_verified"] is True
    assert res["manifest_sha256"]["amber_zip"]
    assert res["manifest_sha256"]["memlens_dataset_32k_json"]
    assert "Qwen3.5-0.8B" in res["manifest_sha256"]["models"]
    assert "Qwen3.5-4B" in res["manifest_sha256"]["models"]

    after_snapshot = snapshot_directory_state(tmp_path)
    assert before_snapshot == after_snapshot, "Drive tree was modified during qualification"


def test_qualify_all_drive_assets_unmounted(tmp_path):
    missing_dir = tmp_path / "non_existent_drive"
    res = qualify_all_drive_assets(missing_dir)
    assert res["status"] == "DRIVE_NOT_MOUNTED"
    assert not missing_dir.exists()


def test_qualify_cli_execution(tmp_path):
    amber_zip = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    _create_mock_amber_zip(amber_zip)
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    _create_mock_memlens_tree(memlens_dir)
    models_dir = tmp_path / "LLM_Weights/models/Qwen"
    _create_mock_model_tree(models_dir, "Qwen3.5-0.8B")
    _create_mock_model_tree(models_dir, "Qwen3.5-4B")

    script_path = Path(__file__).resolve().parents[1] / "scripts/qualify_drive_assets.py"
    report_file = tmp_path.parent / "qualification_report.json"

    # Run CLI with --drive-root and --report-json
    cmd = [
        sys.executable,
        str(script_path),
        "--drive-root",
        str(tmp_path),
        "--report-json",
        str(report_file),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, f"CLI stderr: {proc.stderr}"
    assert report_file.is_file()

    report_data = json.loads(report_file.read_text(encoding="utf8"))
    assert report_data["status"] == "ALL_QUALIFIED"

    # Reject writing report-json inside drive-root
    inside_report = tmp_path / "report_inside_drive.json"
    cmd_bad = [
        sys.executable,
        str(script_path),
        "--drive-root",
        str(tmp_path),
        "--report-json",
        str(inside_report),
    ]
    proc_bad = subprocess.run(cmd_bad, capture_output=True, text=True, check=False)
    assert proc_bad.returncode == 1
    assert "Drive must remain strictly read-only" in proc_bad.stderr


def test_agent_pipeline_qualify_drive_action(tmp_path, monkeypatch):
    from vlmbench import agent_pipeline

    amber_zip = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    _create_mock_amber_zip(amber_zip)
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    _create_mock_memlens_tree(memlens_dir)
    models_dir = tmp_path / "LLM_Weights/models/Qwen"
    _create_mock_model_tree(models_dir, "Qwen3.5-0.8B")
    _create_mock_model_tree(models_dir, "Qwen3.5-4B")

    job = {
        "action": "qualify-drive",
        "drive_root": str(tmp_path),
        "tolerance": 0.05,
    }
    result = agent_pipeline.execute(job)
    assert result["status"] == "ALL_QUALIFIED"
    assert result["read_only_verified"] is True


def test_deterministic_simple_tokenizer():
    tok = DeterministicSimpleTokenizer()
    text = "Hello world! 123"
    ids = tok.encode(text)
    assert len(ids) == len(text)
    decoded = tok.decode(ids)
    assert decoded == text
    assert tok.is_synthetic is True
    assert tok.tokenizer_type == "synthetic_character_fallback"


# ---------------------------------------------------------------------------
# RCS-01 Regression Tests: Path traversal & ambiguous basename matches
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "evil_img",
    [
        "../../../etc/passwd",
        "/tmp/evil.jpg",
        "/etc/shadow",
        "..\\evil.jpg",
        "nested/../../escaped.jpg",
    ],
)
def test_qualify_amber_assets_path_traversal_rejected(tmp_path, evil_img):
    zip_path = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    query_all = [
        {
            "id": 1,
            "image": evil_img,
            "query": "Is there a dog?",
            "task": "discriminative",
            "type": "existence",
        }
    ]
    annotations = {
        "annotations": [
            {"id": 1, "task": "discriminative", "type": "existence", "label": "Yes"}
        ]
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/query/query_all.json", json.dumps(query_all))
        zf.writestr("data/query/query_generative.json", json.dumps([]))
        zf.writestr("data/query/query_discriminative.json", json.dumps(query_all))
        zf.writestr("data/annotation/annotations.json", json.dumps(annotations))
        zf.writestr("image/dummy.jpg", b"mock")

    res = qualify_amber_assets(tmp_path)
    assert res["status"] == "UNQUALIFIED_AMBER_SOURCE"
    assert "Path traversal" in res.get("error", "") or "traversal" in res.get("error", "").lower()


def test_qualify_amber_assets_ambiguous_basename_rejected(tmp_path):
    zip_path = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    query_all = [
        {
            "id": 1,
            "image": "shared_name.jpg",
            "query": "Is there a dog?",
            "task": "discriminative",
            "type": "existence",
        }
    ]
    annotations = {
        "annotations": [
            {"id": 1, "task": "discriminative", "type": "existence", "label": "Yes"}
        ]
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/query/query_all.json", json.dumps(query_all))
        zf.writestr("data/query/query_generative.json", json.dumps([]))
        zf.writestr("data/query/query_discriminative.json", json.dumps(query_all))
        zf.writestr("data/annotation/annotations.json", json.dumps(annotations))
        zf.writestr("folder_a/shared_name.jpg", b"bytes_a")
        zf.writestr("folder_b/shared_name.jpg", b"bytes_b")

    res = qualify_amber_assets(tmp_path)
    assert res["status"] == "UNQUALIFIED_AMBER_SOURCE"
    assert "Ambiguous image" in res.get("error", "")


def test_qualify_amber_assets_missing_image_in_zip_fails(tmp_path):
    zip_path = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/amber/_archives/AMBER.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    query_all = [
        {
            "id": 1,
            "image": "missing.jpg",
            "query": "Is there a dog?",
            "task": "discriminative",
            "type": "existence",
        }
    ]
    annotations = {
        "annotations": [
            {"id": 1, "task": "discriminative", "type": "existence", "label": "Yes"}
        ]
    }
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("data/query/query_all.json", json.dumps(query_all))
        zf.writestr("data/query/query_generative.json", json.dumps([]))
        zf.writestr("data/query/query_discriminative.json", json.dumps(query_all))
        zf.writestr("data/annotation/annotations.json", json.dumps(annotations))
        zf.writestr("image/other.jpg", b"other")

    res = qualify_amber_assets(tmp_path)
    assert res["status"] == "UNQUALIFIED_AMBER_SOURCE"
    assert "not found in AMBER.zip" in res.get("error", "")


# ---------------------------------------------------------------------------
# RCS-03 Regression Tests: Missing image fail (no mock bytes), tokenizer checks,
# and derived_bins_emitted reporting
# ---------------------------------------------------------------------------


def test_qualify_memlens_assets_missing_image_fails_without_fabrication(tmp_path):
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    memlens_dir.mkdir(parents=True, exist_ok=True)
    needle_dir = memlens_dir / "release_images" / "needle_images"
    haystack_dir = memlens_dir / "release_images" / "haystack_images"
    needle_dir.mkdir(parents=True, exist_ok=True)
    haystack_dir.mkdir(parents=True, exist_ok=True)

    raw_item = {
        "question_id": "ml_missing_img_1",
        "question_type": "single_needle",
        "question": "What is the answer?",
        "answer": "Answer",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2026-01-01"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "Look at this.",
                    "images": [{"file": "needle_images/strictly_missing.jpg"}],
                },
                {"role": "assistant", "content": "Done."},
            ],
        ],
    }
    (memlens_dir / "dataset_32k.json").write_text(json.dumps([raw_item]), encoding="utf8")

    res = qualify_memlens_assets(tmp_path, tolerance=0.05)
    assert res["status"] == "UNQUALIFIED_MEMLENS_SOURCE"
    assert "Referenced MemLens source image missing on Drive" in res["error"]


@pytest.mark.parametrize(
    "evil_file",
    [
        "../../../etc/passwd",
        "/etc/shadow",
        "needle_images/../../evil.jpg",
        "..\\evil.jpg",
    ],
)
def test_qualify_memlens_assets_path_traversal_rejected(tmp_path, evil_file):
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    memlens_dir.mkdir(parents=True, exist_ok=True)
    needle_dir = memlens_dir / "release_images" / "needle_images"
    haystack_dir = memlens_dir / "release_images" / "haystack_images"
    needle_dir.mkdir(parents=True, exist_ok=True)
    haystack_dir.mkdir(parents=True, exist_ok=True)

    raw_item = {
        "question_id": "ml_evil_img_1",
        "question_type": "single_needle",
        "question": "What is the answer?",
        "answer": "Answer",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2026-01-01"],
        "haystack_sessions": [
            [
                {
                    "role": "user",
                    "content": "Look at this.",
                    "images": [{"file": evil_file}],
                },
                {"role": "assistant", "content": "Done."},
            ],
        ],
    }
    (memlens_dir / "dataset_32k.json").write_text(json.dumps([raw_item]), encoding="utf8")

    res = qualify_memlens_assets(tmp_path, tolerance=0.05)
    assert res["status"] == "UNQUALIFIED_MEMLENS_SOURCE"
    assert "traversal" in res["error"].lower()


def test_get_qualification_tokenizer_require_real_fails_when_missing(tmp_path):
    empty_dir = tmp_path / "non_existent_model"

    # require_real=True must raise QualificationError when real model tokenizer unavailable
    with pytest.raises(QualificationError, match="Model directory .* does not contain required tokenizer"):
        get_qualification_tokenizer(empty_dir, require_real=True)

    with pytest.raises(QualificationError, match="no model_dir provided"):
        get_qualification_tokenizer(None, require_real=True)

    # require_real=False falls back to synthetic and labels it explicitly
    tok = get_qualification_tokenizer(empty_dir, require_real=False)
    assert tok.is_synthetic is True
    assert tok.tokenizer_type == "synthetic_character_fallback"


def test_qualify_memlens_require_real_tokenizer_fails_without_model(tmp_path):
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    _create_mock_memlens_tree(memlens_dir)

    res = qualify_memlens_assets(tmp_path, require_real_tokenizer=True)
    assert res["status"] == "UNQUALIFIED_MEMLENS_SOURCE"
    assert "Tokenizer qualification failed" in res["error"]


def test_memlens_derived_bins_emitted_reporting(tmp_path):
    memlens_dir = tmp_path / "Research/Topic/Speculative Decoding/STARLIGHT/memlens"
    _create_mock_memlens_tree(memlens_dir)

    res = qualify_memlens_assets(tmp_path, tolerance=0.05)
    assert res["status"] == "QUALIFIED_MEMLENS_SOURCE"
    schema = res["schema_compatibility"]

    # Verify derived_bins_emitted reports the actual integer bins emitted (e.g. [128, 512])
    assert isinstance(schema["derived_bins_emitted"], list)
    assert len(schema["derived_bins_emitted"]) > 0
    assert 128 in schema["derived_bins_emitted"]
    assert schema["derived_records_emitted"] > 0

    # Verify tokenizer labeling
    assert schema["tokenizer_label"] == "SYNTHETIC_PROXY"
    assert schema["tokenizer_is_synthetic"] is True
    assert schema["tokenizer_type"] == "synthetic_character_fallback"
