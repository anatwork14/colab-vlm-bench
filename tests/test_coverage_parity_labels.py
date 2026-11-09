"""ITER2 D2b C-01/C-02: coverage and parity labels are functions of measured values.

Pure functions only. No model is loaded, and no committed receipt is read or written.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_model_coverage_and_parity.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_model_coverage_and_parity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vmc = _load()


def test_parity_status_is_measured_against_tolerance():
    assert vmc.parity_status(6.2e-06) == "PASS"
    assert vmc.parity_status(0.5) == "FAIL"
    assert vmc.parity_status(1e-4) == "FAIL", "tolerance is strict: a diff equal to 1e-4 is not a pass"
    assert vmc.parity_status(float("nan")) == "FAIL"


def test_scaffold_status_follows_measured_coverage():
    assert vmc.scaffold_scientific_status(25.82) == "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE"
    assert vmc.scaffold_scientific_status(100.0) == "FULLY_WEIGHTED_PYTORCH_REFERENCE"
    assert vmc.scaffold_scientific_status(0.0) == "UNWEIGHTED_PYTORCH_SCAFFOLD"


def test_official_coverage_status_follows_measured_coverage():
    assert vmc.official_coverage_status(100.0) == "OFFICIAL_COVERAGE_COMPLETE"
    assert vmc.official_coverage_status(99.79) == "OFFICIAL_COVERAGE_INCOMPLETE"


def test_hook_qualification_needs_full_coverage_and_passing_parity():
    qualified = "OFFICIAL_TRANSFORMERS_HOOK_QUALIFIED"
    unqualified = "OFFICIAL_TRANSFORMERS_HOOK_UNQUALIFIED"
    assert vmc.official_hook_status(100.0, "PASS") == qualified
    assert vmc.official_hook_status(100.0, "FAIL") == unqualified
    assert vmc.official_hook_status(99.0, "PASS") == unqualified


def test_unmapped_modules_are_grouped_from_measured_keys():
    keys = [
        "model.language_model.layers.13.mlp.gate_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.visual.blocks.3.attn.qkv.weight",
    ]
    assert vmc.unmapped_module_patterns(keys) == [
        "model.language_model.layers.*.mlp.gate_proj",
        "model.visual.blocks.*.attn.qkv",
    ]
    assert vmc.unmapped_module_patterns([]) == []


def test_recommendation_only_when_hook_is_qualified():
    qualified = vmc.scientific_qualification_labels(
        "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE", "OFFICIAL_TRANSFORMERS_HOOK_QUALIFIED"
    )
    assert qualified["recommendation"].startswith("Use official Transformers hooking")
    unqualified = vmc.scientific_qualification_labels(
        "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE", "OFFICIAL_TRANSFORMERS_HOOK_UNQUALIFIED"
    )
    assert "Use official" not in unqualified["recommendation"]
    assert unqualified["custom_scaffold_label"] == "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE"


def test_script_has_no_hand_written_status_literals():
    src = SCRIPT.read_text(encoding="utf8")
    assert '"scientific_status": "' not in src
    assert '"status": "PASS"' not in src and '"status": "FAIL"' not in src
    assert '"status": parity_status(max_diff)' in src
    assert "scientific_qualification_labels(" in src


def test_default_output_is_a_new_run_file_and_writes_refuse_overwrite(tmp_path):
    default = vmc.default_out_file("20261109T120000Z")
    assert default.parent == ROOT / "artifacts" / "runs"
    assert default != ROOT / "artifacts" / "model_coverage_and_parity_receipt.json"
    target = tmp_path / "receipt.json"
    vmc.write_receipt(target, {"v": 1})
    with pytest.raises(FileExistsError):
        vmc.write_receipt(target, {"v": 2})
    vmc.write_receipt(target, {"v": 3}, force=True)
    assert '"v": 3' in target.read_text(encoding="utf8")
