"""ITER2 G-4 (REVIEW_ITER2 §4): one logits-parity tolerance, shared by both scripts.

Value: 1e-4 (strict: a difference equal to the tolerance fails). Rationale: the only measured
FP32 CPU value on record is 6.2e-06 on the Test A fixture; 1e-4 leaves about a 16x margin for
BLAS and thread-order drift between runs, while still rejecting a numerically different model.
"""
from __future__ import annotations

import re
from pathlib import Path

from vlmbench.infrastructure.parity import LOGITS_PARITY_TOLERANCE, logits_parity_passes

ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts/run_progressive_qualification.py"
VERIFY_SCRIPT = ROOT / "scripts/verify_model_coverage_and_parity.py"


def test_the_shared_tolerance_is_1e_minus_4():
    assert LOGITS_PARITY_TOLERANCE == 1e-4


def test_comparison_is_strict_and_rejects_nan():
    assert logits_parity_passes(6.2e-06) is True
    assert logits_parity_passes(0.0) is True
    assert logits_parity_passes(LOGITS_PARITY_TOLERANCE) is False
    assert logits_parity_passes(float("nan")) is False


def test_both_scripts_use_the_shared_constant_and_no_literal_tolerance():
    for path in (RUN_SCRIPT, VERIFY_SCRIPT):
        source = path.read_text(encoding="utf8")
        assert "LOGITS_PARITY_TOLERANCE" in source, path.name
        assert not re.search(r"\b1e-[45]\b", source), f"literal tolerance left in {path.name}"


def test_verify_script_tolerance_is_the_shared_one():
    import importlib.util

    spec = importlib.util.spec_from_file_location("vmc_tol", VERIFY_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PARITY_TOLERANCE == LOGITS_PARITY_TOLERANCE
    assert module.parity_status(LOGITS_PARITY_TOLERANCE) == "FAIL"
    assert module.parity_status(6.2e-06) == "PASS"
