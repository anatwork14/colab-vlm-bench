"""Prevent historical GPU figures from silently being treated as current evidence."""
import json
from pathlib import Path


def test_unrequalified_published_receipt_is_superseded():
    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    active = json.loads((artifacts / "qwen35_pretrained_qualification_receipt.json").read_text())
    if active.get("qualification_verdict") == "SUPERSEDED_NO_POST_REPAIR_PRETRAINED_GPU_RECEIPT":
        assert "historical_receipt" not in active
        assert active["release_gates"]["NATIVE_SPECULATIVE_MTP_CORRECTNESS"] == "GATED"
        assert active["release_gates"]["PHYSICAL_GPU_VRAM_SAVING"] == "NOT_MEASURED"
    else:
        # Any future real-GPU result replacing the active notice must advertise
        # a new schema and explicit provenance; never accept the 2026-11-09 run.
        assert active.get("schema") == "starlight_gpu_qualification_v2"
        assert active.get("input_checkpoint_revision")
        assert active.get("post_repair_git_commit")
        assert active.get("native_mtp_correctness", {}).get("status") != "PASS_UNVERIFIED"
    legacy = json.loads((
        artifacts / "historical" /
        "qwen35_pretrained_qualification_receipt_20261109_before_scientific_repairs.json"
    ).read_text())
    assert legacy["qualification_verdict"] == "ALL_TESTS_QUALIFIED"
    assert legacy["tests"]["test_d_seven_arms"]["arms"]["mtp"]["mtp_acceptance_rate"] == 0.4


def test_four_image_tasks_cannot_report_keyword_based_accuracy():
    src = (Path(__file__).resolve().parents[1] / "scripts/run_progressive_qualification.py").read_text()
    assert "quality_score_qualified" in src
    assert "semantic_verification_pass\": None" in src
    assert 'any(w in text_lower' not in src
