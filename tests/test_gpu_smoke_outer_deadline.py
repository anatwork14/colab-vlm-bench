"""ITER2 D8: the GPU smoke is bounded by an OUTER hard deadline, enforced by the parent process.

The child does the work. The parent kills the child's whole session at the deadline, so a hung CUDA
call cannot defeat the bound. These tests use fake children only; they touch no GPU.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pytorch_reference_gpu_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("smoke_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load()


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def test_hanging_child_is_killed_at_the_outer_deadline():
    t0 = time.monotonic()
    run = smoke.run_outer([sys.executable, "-c", "import time; time.sleep(600)"], timeout_s=2)
    assert run["outcome"] == "OUTER_DEADLINE"
    assert run["returncode"] is None
    assert time.monotonic() - t0 < 15


def test_grandchildren_of_a_hanging_child_are_killed_too(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    child = (
        "import subprocess, sys, time, pathlib\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(600)'])\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(g.pid))\n"
        "time.sleep(600)\n"
    )
    run = smoke.run_outer([sys.executable, "-c", child], timeout_s=3)
    assert run["outcome"] == "OUTER_DEADLINE"
    grandchild = int(pid_file.read_text())
    time.sleep(0.5)
    assert not _alive(grandchild)


def test_child_that_exits_before_the_deadline_reports_its_code():
    run = smoke.run_outer([sys.executable, "-c", "raise SystemExit(3)"], timeout_s=30)
    assert run == {"outcome": "EXITED", "returncode": 3, "elapsed_s": run["elapsed_s"]}


def test_main_writes_no_receipt_when_the_outer_deadline_fires(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(smoke, "WALL_LIMIT_S", 2)
    monkeypatch.setattr(smoke, "build_child_cmd", lambda result: [sys.executable, "-c", "import time; time.sleep(600)"])
    monkeypatch.setattr(sys, "argv", ["smoke", "--out-dir", str(tmp_path)])
    assert smoke.main() == 2
    assert list(tmp_path.iterdir()) == []
    assert "no receipt written" in capsys.readouterr().out


def _child_writing(status: str, result_path_holder: list):
    def build(result: Path):
        payload = {"status": status, "label": smoke.LABEL, "elapsed_wall_s": 1.0,
                   "token_agreement": "IDENTICAL_ALL_INPUTS" if status == "RUN_COMPLETED" else None}
        return [sys.executable, "-c",
                f"import pathlib; pathlib.Path({str(result)!r}).write_text({json.dumps(payload)!r})"]
    return build


def test_main_writes_no_receipt_unless_the_run_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "build_child_cmd", _child_writing("ABORTED: free GPU memory 100 MiB", []))
    monkeypatch.setattr(sys, "argv", ["smoke", "--out-dir", str(tmp_path)])
    assert smoke.main() == 1
    assert list(tmp_path.iterdir()) == []


def test_main_writes_the_receipt_when_the_run_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "build_child_cmd", _child_writing("RUN_COMPLETED", []))
    monkeypatch.setattr(sys, "argv", ["smoke", "--out-dir", str(tmp_path)])
    assert smoke.main() == 0
    written = list(tmp_path.iterdir())
    assert len(written) == 1
    assert json.loads(written[0].read_text())["status"] == "RUN_COMPLETED"
