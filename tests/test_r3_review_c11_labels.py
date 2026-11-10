"""ITER3 round-3 independent review (wF:p9), C-11: qualification and engine labels must be derived, and exact.

Survivors of the review's mutation runs at 238c8ab (scratch copies, one change at a time):
  * the unweighted-scaffold engine label set to PASS: no test asserted the scaffold value exactly;
  * the engine ACTUAL_BACKEND_HOOK_PASS forced to PASS: test_real_engine.py asserts `in ("PASS", "GATED_...")`;
  * a release-gate status that is neither PASS nor FAIL counted as PASS: no test covered an unknown status.
These tests pin the exact values. They pass at HEAD and fail under each of those mutations.
cpu_pytorch_reference_not_vllm. The scaffold engine has no checkpoint and runs on CPU.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from vlmbench.infrastructure.real_engine import RealSevenArmEngine

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_progressive_qualification.py"
GPU = "NVIDIA GeForce GTX 1080 Ti"


def _load():
    spec = importlib.util.spec_from_file_location("run_progressive_qualification_c11", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qual = _load()


@pytest.mark.parametrize("status", ["weird", "", "pass", "SKIPPED"])
def test_a_status_that_is_not_pass_or_fail_is_never_reported_as_pass(status):
    gates = qual.compute_release_gates({"test_a_architecture": {"status": status}}, device_name=GPU)
    assert gates["POLICY_HARNESS_PASS"] == "NOT_TESTED", gates["POLICY_HARNESS_PASS"]


def test_no_recorded_outcome_yields_no_pass_gate_at_all():
    gates = qual.compute_release_gates({}, device_name=GPU)
    assert "PASS" not in gates.values(), {k: v for k, v in gates.items() if v == "PASS"}


def test_scaffold_engine_reports_the_exact_gated_labels_without_a_checkpoint():
    engine = RealSevenArmEngine(method_arm="base", device="cpu")
    res = engine.execute_request(request_id="c11_scaffold", prompt_token_ids=[100, 101, 102],
                                 max_new_tokens=2, image_present=False)
    assert res.gates["SEVEN_ARM_HOOK_COVERAGE_PASS"] == "GATED_PRETRAINED_CHECKPOINT_REQUIRED"
    assert res.gates["REAL_MODEL_FORWARD_PASS"] == "GATED_PRETRAINED_CHECKPOINT_REQUIRED"
    assert res.gates["ACTUAL_BACKEND_HOOK_PASS"] == "GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED"
