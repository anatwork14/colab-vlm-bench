"""ITER2 D8/D9: the smoke runs are bounded by OUTER hard deadlines enforced by the parent process.

The child does the work. The parent kills the child's whole session at the deadline, so a hung CUDA or
CPU call cannot defeat the bound. A receipt is written only when a run completes. These tests use fake
children only; they touch no GPU and load no model.
"""
from __future__ import annotations

import importlib.util
import json
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
    assert run["outcome"] == "EXITED" and run["returncode"] == 3


def test_cpu_reference_has_its_own_300_second_budget(monkeypatch):
    seen = {}

    def fake_run_outer(cmd, timeout_s):
        seen["t"] = timeout_s
        return {"outcome": "OUTER_DEADLINE", "returncode": None, "elapsed_s": 0.0}

    monkeypatch.setattr(smoke, "run_outer", fake_run_outer)
    monkeypatch.setattr(sys, "argv", ["smoke", "--device", "cpu"])
    smoke.main()
    assert seen["t"] == 300


def test_no_receipt_when_the_outer_deadline_fires(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(smoke.WALL_LIMIT_S, "cuda", 2)
    monkeypatch.setattr(smoke, "build_child_cmd",
                        lambda device, result: [sys.executable, "-c", "import time; time.sleep(600)"])
    monkeypatch.setattr(sys, "argv", ["smoke", "--device", "cuda", "--out-dir", str(tmp_path)])
    assert smoke.main() == 2
    assert list(tmp_path.iterdir()) == []
    assert "no receipt written" in capsys.readouterr().out


def _child_writing(status: str):
    def build(device, result: Path):
        payload = {"status": status, "label": smoke.LABEL, "device": device, "git_sha": "a" * 40,
                   "dtype": "float16", "elapsed_wall_s": 1.0, "results": {}}
        return [sys.executable, "-c",
                f"import pathlib; pathlib.Path({str(result)!r}).write_text({json.dumps(payload)!r})"]
    return build


def test_no_receipt_unless_the_run_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "build_child_cmd", _child_writing("ABORTED: free GPU memory 100 MiB"))
    monkeypatch.setattr(sys, "argv", ["smoke", "--device", "cuda", "--out-dir", str(tmp_path)])
    assert smoke.main() == 1
    assert list(tmp_path.iterdir()) == []


def test_receipt_written_for_a_completed_run_with_a_device_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, "build_child_cmd", _child_writing("RUN_COMPLETED"))
    monkeypatch.setattr(sys, "argv", ["smoke", "--device", "cuda", "--out-dir", str(tmp_path)])
    assert smoke.main() == 0
    written = list(tmp_path.iterdir())
    assert len(written) == 1 and written[0].name.endswith("_gpu_fp16.json")
    assert json.loads(written[0].read_text())["status"] == "RUN_COMPLETED"
