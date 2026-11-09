"""ITER2 N-10 (REVIEW_ITER2B §3): oracle checks must not disappear under `python -O`.

Asserts are stripped under -O, and `guarded` then recorded PASS for checks that never ran. The
qualification script therefore uses an explicit `require()` that raises regardless of optimisation.
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_progressive_qualification.py"


def test_the_qualification_script_contains_no_assert_statements():
    tree = ast.parse(SCRIPT.read_text(encoding="utf8"))
    asserts = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Assert)]
    assert asserts == [], f"assert statements remain at lines {asserts[:10]}"


def test_under_python_O_a_failed_oracle_is_recorded_as_fail_not_pass():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('rpq', {str(SCRIPT)!r})\n"
        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
        "def oracle():\n"
        "    m.require(1 == 2, 'deliberately false')\n"
        "    return {}\n"
        "out = m.guarded(oracle)\n"
        "print(out['status'], out.get('error', ''))\n"
        "try:\n"
        "    m.require(False, 'raises')\n"
        "    print('NO_RAISE')\n"
        "except m.OracleFailure:\n"
        "    print('RAISED')\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT / "src")}
    done = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True, text=True, env=env, timeout=300)
    lines = done.stdout.strip().splitlines()
    assert lines[0].startswith("FAIL") and "deliberately false" in lines[0], done.stdout + done.stderr
    assert lines[1] == "RAISED", done.stdout + done.stderr
