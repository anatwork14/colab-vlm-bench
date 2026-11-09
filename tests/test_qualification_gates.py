"""ITER2 D2b G-01/G-02: release gates are a function of recorded test results.

No gate may be a hand-written PASS string or carry a numeric literal. Failing or
unrun tests must surface as FAIL or NOT_TESTED. Loading the script imports torch and
transformers but does not run any model.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_progressive_qualification.py"
ARMS = ("base", "h2o_base", "da_base", "ecrd_base", "mtp", "h2o_mtp", "da_mtp")
ENGINE_GATES = {
    "ACTUAL_BACKEND_HOOK_PASS": "GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED",
    "MTP_CORRECTNESS_PASS": "GATED_NATIVE_MTP_VERIFIER_REQUIRED",
    "CCU_ISOLATION_PASS": "GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED",
}


def _load():
    spec = importlib.util.spec_from_file_location("run_progressive_qualification", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


qual = _load()


def _all_passing_tests() -> dict:
    return {
        "test_a_architecture": {
            "status": "PASS",
            "coverage_inventory": {"logits_parity": {"max_abs_logits_diff": 6.2e-06, "top1_match": True}},
        },
        "test_b_text_forward": {"status": "PASS"},
        "test_c_multimodal_forward": {"status": "PASS"},
        "test_d_seven_arms": {
            "status": "PASS",
            "arms": {arm: {"gates": dict(ENGINE_GATES)} for arm in ARMS},
        },
        "test_e_speculative_correctness": {"status": "PASS"},
        "test_f_fixed_output_serving": {"status": "PASS"},
        "test_g_ccu_concurrency": {"status": "PASS"},
    }


GPU = "NVIDIA GeForce GTX 1080 Ti"


def test_all_passing_tests_give_measured_labels():
    gates = qual.compute_release_gates(_all_passing_tests(), device_name=GPU)
    assert gates["POLICY_HARNESS_PASS"] == "PASS"
    assert gates["REAL_MODEL_FORWARD_PASS"] == "PASS"
    assert gates["REAL_MULTIMODAL_INFERENCE_PASS"] == "PASS"
    assert gates["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"
    assert gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] == "PASS"
    assert gates["CCU_REQUEST_ISOLATION_TEST_PASS"] == "PASS"
    # engine-reported gates are carried through only when every arm agrees
    assert gates["ACTUAL_BACKEND_HOOK_PASS"] == ENGINE_GATES["ACTUAL_BACKEND_HOOK_PASS"]
    assert gates["MTP_CORRECTNESS_PASS"] == ENGINE_GATES["MTP_CORRECTNESS_PASS"]
    # nothing in this script exercises vLLM or an A100, so these cannot be PASS here
    assert gates["VLLM_SCHEDULER_CONCURRENCY_PASS"] == "NOT_TESTED"
    assert gates["A100_GPU_QUALIFIED"] == "BLOCKED_PENDING_OWNER_PAID_AUTH"


def test_every_gate_value_is_a_bare_label_without_numbers_or_prose():
    gates = qual.compute_release_gates(_all_passing_tests(), device_name=GPU)
    pattern = re.compile(r"^[A-Z][A-Z0-9_]*$")
    for name, value in gates.items():
        assert pattern.match(value), f"{name} carries prose or a numeric literal: {value!r}"


def test_failed_multimodal_test_fails_only_its_dependent_gates():
    tests = _all_passing_tests()
    tests["test_c_multimodal_forward"] = {"status": "FAIL", "error": "boom"}
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["REAL_MULTIMODAL_INFERENCE_PASS"] == "FAIL"
    assert gates["REAL_MODEL_FORWARD_PASS"] == "PASS"


def test_missing_test_is_not_tested_not_pass():
    tests = _all_passing_tests()
    del tests["test_e_speculative_correctness"]
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] == "NOT_TESTED"


def test_failed_seven_arm_test_fails_engine_gates():
    tests = _all_passing_tests()
    tests["test_d_seven_arms"] = {"status": "FAIL", "error": "arm mtp broke"}
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["SEVEN_ARM_PRETRAINED_PASS"] == "FAIL"
    assert gates["ACTUAL_BACKEND_HOOK_PASS"] == "FAIL"
    assert gates["MTP_CORRECTNESS_PASS"] == "FAIL"


def test_arms_disagreeing_on_an_engine_gate_fail_it():
    tests = _all_passing_tests()
    tests["test_d_seven_arms"]["arms"]["h2o_mtp"]["gates"]["ACTUAL_BACKEND_HOOK_PASS"] = "PASS"
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["ACTUAL_BACKEND_HOOK_PASS"] == "FAIL"


def test_script_source_has_no_hand_written_gate_strings():
    src = SCRIPT.read_text(encoding="utf8")
    assert '"PASS (' not in src
    assert "max_abs_diff=0.0" not in src
    assert '"PASS_DIAGNOSTIC_ONLY"' not in src
    # the only PASS status literal is the one guarded() writes after a call completes
    assert src.count('"status": "PASS"') == 1, "status must come from guarded(), not a literal"


def test_guarded_records_failure_with_its_message_instead_of_raising():
    def boom():
        raise AssertionError("logits diverged")

    result = qual.guarded(boom)
    assert result["status"] == "FAIL"
    assert "logits diverged" in result["error"]


def test_guarded_marks_completed_test_as_pass_and_keeps_its_fields():
    assert qual.guarded(lambda: {"x": 1}) == {"status": "PASS", "x": 1}


def test_failed_gate_makes_verdict_name_the_failing_tests():
    tests = _all_passing_tests()
    tests["test_f_fixed_output_serving"] = {"status": "FAIL", "error": "256 tokens expected"}
    assert qual.verdict_for(tests) == "INCOMPLETE_OR_FAILED: test_f_fixed_output_serving"
    assert qual.verdict_for(_all_passing_tests()) == "PRETRAINED_FORWARD_AND_MTP_PROPOSAL_DIAGNOSTIC_ONLY"


# ITER2 D2b G-03/G-05/G-06: reporting uses current telemetry names; "not verified" is null.

def test_round_or_none_keeps_not_verified_as_null():
    assert qual._round_or_none(None) is None
    assert qual._round_or_none(0.123456) == 0.1235


def test_script_no_longer_uses_removed_or_fabricated_fields():
    src = SCRIPT.read_text(encoding="utf8")
    assert "logical_selection_fraction" not in src, "removed field; use h2o_logical_selection_accounting"
    assert "mtp_acceptance_rate == 0.0" not in src, "acceptance is not verified here: assert null, never zero"
    assert '"mtp_acceptance_rate": 0.0' not in src
    assert "res.mtp_acceptance_rate is None" in src


def test_reported_result_fields_exist_on_current_execution_result():
    from dataclasses import fields
    from vlmbench.infrastructure.real_engine import RealExecutionResult

    names = {f.name for f in fields(RealExecutionResult)}
    for name in (
        "per_step_evicted_slot_rate",
        "attention_read_restriction_ratio",
        "attention_mask_restriction_ratio",
        "mtp_acceptance_rate",
        "physical_kv_bytes_freed",
    ):
        assert name in names, f"{name} is not a field of RealExecutionResult"


# ITER2 D2b G-07: the default receipt never targets the supersession notice, and an
# existing receipt is never overwritten without --force.

def test_default_receipt_path_is_a_new_timestamped_run_file():
    path = qual.default_receipt_path("20261109T120000Z")
    assert path.parent == ROOT / "artifacts" / "runs"
    assert path.name == "20261109T120000Z_qwen35_pretrained_qualification.json"
    assert path != ROOT / "artifacts" / "qwen35_pretrained_qualification_receipt.json"


def test_script_has_no_default_that_points_at_the_committed_notice():
    src = SCRIPT.read_text(encoding="utf8")
    assert 'artifacts" / "qwen35_pretrained_qualification_receipt.json"' not in src


def test_write_receipt_refuses_to_overwrite_without_force(tmp_path):
    target = tmp_path / "receipt.json"
    qual.write_receipt(target, {"v": 1})
    with pytest.raises(FileExistsError):
        qual.write_receipt(target, {"v": 2})
    assert '"v": 1' in target.read_text(encoding="utf8"), "the existing receipt must be untouched"


def test_write_receipt_overwrites_only_with_force_and_creates_parents(tmp_path):
    target = tmp_path / "nested" / "receipt.json"
    qual.write_receipt(target, {"v": 1})
    qual.write_receipt(target, {"v": 2}, force=True)
    assert '"v": 2' in target.read_text(encoding="utf8")


def test_main_refuses_existing_receipt_before_running_any_test(tmp_path, monkeypatch):
    existing = tmp_path / "receipt.json"
    existing.write_text("{}", encoding="utf8")
    monkeypatch.setattr("sys.argv", ["run_progressive_qualification.py", "--receipt-path", str(existing)])
    monkeypatch.setattr(qual, "guarded", lambda *a, **k: pytest.fail("no test may run when the receipt is refused"))
    with pytest.raises(SystemExit) as exc:
        qual.main()
    assert exc.value.code != 0
    assert existing.read_text(encoding="utf8") == "{}"


# ITER2 G-1 (REVIEW_ITER2 §4): the MTP gate reads the MTP arms only. Non-MTP arms have no MTP
# correctness to report, so their values never enter the gate.

def test_non_mtp_arms_do_not_turn_a_gated_mtp_gate_into_fail():
    tests = _all_passing_tests()
    for arm in ("base", "h2o_base", "da_base", "ecrd_base"):
        tests["test_d_seven_arms"]["arms"][arm]["gates"]["MTP_CORRECTNESS_PASS"] = "NOT_APPLICABLE"
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["MTP_CORRECTNESS_PASS"] == ENGINE_GATES["MTP_CORRECTNESS_PASS"]


def test_an_mtp_arm_disagreeing_with_the_others_still_fails_the_gate():
    tests = _all_passing_tests()
    tests["test_d_seven_arms"]["arms"]["da_mtp"]["gates"]["MTP_CORRECTNESS_PASS"] = "PASS"
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["MTP_CORRECTNESS_PASS"] == "FAIL"


def test_not_applicable_is_not_a_gate_value():
    gates = qual.compute_release_gates(_all_passing_tests(), device_name=GPU)
    assert "NOT_APPLICABLE" not in gates.values()


def test_a_missing_mtp_arm_leaves_the_mtp_gate_not_tested():
    tests = _all_passing_tests()
    del tests["test_d_seven_arms"]["arms"]["da_mtp"]
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["MTP_CORRECTNESS_PASS"] == "NOT_TESTED"


def test_an_mtp_arm_reporting_not_applicable_leaves_the_mtp_gate_not_tested():
    tests = _all_passing_tests()
    tests["test_d_seven_arms"]["arms"]["h2o_mtp"]["gates"]["MTP_CORRECTNESS_PASS"] = "NOT_APPLICABLE"
    gates = qual.compute_release_gates(tests, device_name=GPU)
    assert gates["MTP_CORRECTNESS_PASS"] == "NOT_TESTED"
