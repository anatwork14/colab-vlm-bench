"""CPU-only regression for damaged Google Drive JSONL reused by Colab Step 6.

No vLLM, GPU, model weights, actual datasets, or external I/O.
"""
import json
from pathlib import Path

import pytest

from vlmbench.artifacts import sha256
from vlmbench import notebook_bridge
from vlmbench.workloads import WorkloadError


def candidate(sid="source-1__text2048", dataset="memlens", length=2048):
    return {
        "id": sid, "dataset": dataset, "track": "derived_serving",
        "derived_from": "source-1",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "real source excerpt"},
            {"type": "image_path", "path": "/content/drive/a.jpg",
             "asset_root": "/content/drive"},
            {"type": "text", "text": "question"},
        ]}],
        "requested_text_length_bin": length,
        "image_count": 1,
    }


def test_corrupt_first_line_caught_before_gpu(tmp_path):
    file = tmp_path / "previously-existed.jsonl"
    file.write_text('{"id": "unterminated string\n', encoding="utf-8")
    with pytest.raises(WorkloadError, match="INVALID_CANDIDATE_JSONL line 1"):
        notebook_bridge.validate_serving_candidates(
            file, dataset="memlens", input_lengths=[2048, 16384])


def test_corrupt_middle_line_is_not_silently_dropped(tmp_path):
    file = tmp_path / "candidates.jsonl"
    file.write_text(json.dumps(candidate()) + '\n{"broken":\n',
                    encoding="utf-8")
    with pytest.raises(WorkloadError, match="INVALID_CANDIDATE_JSONL line 2"):
        notebook_bridge.validate_serving_candidates(
            file, dataset="memlens", input_lengths=[2048, 16384])


def test_corrupt_drive_candidates_rebuild_on_ssd_with_backup(tmp_path, monkeypatch):
    drive = tmp_path / "drive"
    drive.mkdir()
    broken = drive / "memlens_pilot_candidates.jsonl"
    corrupted_bytes = b'{"id": "unterminated string\n'
    broken.write_bytes(corrupted_bytes)
    expected_bad_sha = sha256(broken)
    annotated = drive / "memlens_pilot_annotated.jsonl"
    build_calls = []

    def fake_build(native_path, output_path, **kwargs):
        build_calls.append((native_path, output_path, kwargs))
        assert str(output_path).startswith(str(tmp_path / "ssd"))
        Path(output_path).write_text(
            json.dumps(candidate()) + "\n", encoding="utf-8")
        return {"status": "PREPARED_NEEDS_GPU_TOKEN_ANNOTATION"}

    monkeypatch.setattr(notebook_bridge, "build_serving_candidates", fake_build)

    def run_once():
        return notebook_bridge.prepare_serving_candidates(
            tmp_path / "native.jsonl", broken,
            annotated_path=annotated, dataset="memlens", model_id="qwen35_08b",
            model_directory=tmp_path / "model",
            input_lengths=[2048, 16384], max_per_bin=192,
            local_scratch_dir=tmp_path / "ssd")

    first = run_once()
    assert first["status"] == "REBUILT_CORRUPT_CANDIDATES"
    assert first["records"] == 1
    assert first["per_text_bin"] == {"2048": 1, "16384": 0}
    assert first["prior_error"] and "INVALID_CANDIDATE_JSONL" in first["prior_error"]
    preserved = Path(first["preserved_corrupt_backup"])
    assert preserved.read_bytes() == corrupted_bytes
    assert expected_bad_sha == sha256(preserved)
    assert first["source_sha256"] == sha256(broken)
    assert not Path(str(broken) + ".repair.pending").exists()

    second = run_once()
    assert second["status"] == "REUSED_VALIDATED_CANDIDATES"
    assert len(build_calls) == 1


def test_annotation_receipt_blocks_candidate_regeneration(tmp_path, monkeypatch):
    broken = tmp_path / "candidates.jsonl"
    broken.write_text('{"id": "unterminated string\n', encoding="utf-8")
    annotated = tmp_path / "annotated.jsonl"
    annotated.write_text("earlier science evidence\n", encoding="utf-8")
    monkeypatch.setattr(notebook_bridge, "build_serving_candidates",
                        lambda *a, **kw: pytest.fail("Must not regenerate"))
    with pytest.raises(WorkloadError, match="regeneration refused"):
        notebook_bridge.prepare_serving_candidates(
            tmp_path / "native", broken, annotated_path=annotated,
            dataset="memlens", model_id="qwen35_08b",
            model_directory=tmp_path / "model", input_lengths=[2048],
            max_per_bin=192, local_scratch_dir=tmp_path / "ssd")
    assert broken.read_text().startswith('{"id": "unterminated')


def test_annotation_source_mismatch_rejected_early(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(json.dumps(candidate()) + "\n")
    annotated = tmp_path / "annotated.jsonl"
    annotated.write_text("previous result\n")
    state = Path(str(annotated) + ".state.json")
    state.write_text(json.dumps({
        "source_sha256": "0" * 64, "model": "qwen35_08b"}))
    with pytest.raises(WorkloadError, match="source SHA/model mismatch"):
        notebook_bridge.validate_annotation_resume_source(
            candidates, annotated, model_id="qwen35_08b")


def test_valid_source_trusted_only_after_all_lines_pass(tmp_path):
    file = tmp_path / "valid.jsonl"
    file.write_text(json.dumps(candidate()) + "\n" +
                    json.dumps(candidate("source-2__text16384", length=16384)) +
                    "\n")
    audited = notebook_bridge.validate_serving_candidates(
        file, dataset="memlens", input_lengths=[2048, 16384])
    assert audited["records"] == 2
    assert audited["per_text_bin"] == {"2048": 1, "16384": 1}
