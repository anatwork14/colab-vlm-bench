"""ITER3 B3 documentation consistency (documentation check; no model, no inference).

Checks that docs/harness/BACKEND_INTERFACE.md agrees with the code and the test files it names:
every supported arm has a matrix row, every test file it cites exists, vLLM is stated as BLOCKED,
the CPU-reference label is present, no speed figure appears, and the F-1 defect is linked to its
strict xfail test.
"""
import re
from pathlib import Path

from vlmbench.infrastructure.real_engine import RealSevenArmEngine

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "harness" / "BACKEND_INTERFACE.md"


def _text():
    return DOC.read_text(encoding="utf-8")


def test_every_supported_arm_has_a_matrix_row():
    text = _text()
    missing = [arm for arm in sorted(RealSevenArmEngine.SUPPORTED_ARMS)
               if not re.search(rf"^\| {re.escape(arm)} \|", text, re.M)]
    assert missing == [], f"arms without a matrix row: {missing}"


def test_every_test_file_the_doc_cites_exists():
    names = set(re.findall(r"test_[A-Za-z0-9_]+\.py", _text()))
    assert names, "the doc should cite test files"
    missing = sorted(n for n in names if not (ROOT / "tests" / n).is_file())
    assert missing == [], f"cited test files that do not exist: {missing}"


def test_vllm_is_stated_blocked_and_the_cpu_label_is_present():
    text = _text()
    assert "**BLOCKED.**" in text
    assert "cpu_pytorch_reference_not_vllm" in text


def test_the_doc_carries_no_speed_or_throughput_figure():
    pattern = re.compile(r"\d+(\.\d+)?\s*(tok/s|tokens/s|tokens per second|x speedup|ms per token)", re.I)
    assert pattern.search(_text()) is None


def test_F1_defect_is_documented_and_linked_to_its_strict_xfail():
    text = _text()
    assert "F-1" in text
    assert "test_F1_protected_veto_that_differs_across_layers_completes" in text
    assert "strict xfail" in text
