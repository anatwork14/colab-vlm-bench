"""ITER2 N-11 (REVIEW_ITER2B §3): the seven-arm gate has one label everywhere, and it names what it measures.

The gate is set when the official hooked checkpoint is attached across the seven arms (hook attachment and
tensor coverage). It is not an inference-correctness claim. The old name said "pretrained", which read as
correctness, so it is retired from source, scripts and tests. Historical receipts and audit notes keep their text.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = "SEVEN_ARM_" + "PRETRAINED_PASS"  # built at run time so this file does not contain the literal
NEW = "SEVEN_ARM_HOOK_COVERAGE_PASS"


def _files(*dirs):
    for d in dirs:
        yield from (ROOT / d).rglob("*.py")


def test_the_retired_seven_arm_label_is_absent_from_source_scripts_and_tests():
    offenders = []
    for path in _files("src", "scripts", "tests"):
        if path.name == Path(__file__).name:
            continue
        if OLD in path.read_text(encoding="utf8"):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], offenders


def test_the_engine_emits_the_hook_coverage_label():
    source = (ROOT / "src/vlmbench/infrastructure/real_engine.py").read_text(encoding="utf8")
    assert f'"{NEW}": seven_arm_gate' in source
