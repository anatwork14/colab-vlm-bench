"""F-07: the legacy GPU helpers must not push results to git and must not install software
at runtime. These are static and refusal-path checks; nothing here contacts Colab."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "scripts/colab_controller.sh"
REMOTE_ENTRY = ROOT / "scripts/remote_entry.py"


def test_controller_contains_no_git_commit_or_push():
    text = CONTROLLER.read_text(encoding="utf8")
    assert "git push" not in text
    assert "git commit" not in text
    assert "git add" not in text


def test_controller_refuses_publish_before_any_colab_call():
    env = {k: v for k, v in os.environ.items() if k != "PATH"}
    env["PATH"] = "/usr/bin:/bin"  # no colab binary is reachable
    proc = subprocess.run(
        ["bash", str(CONTROLLER), "--session", "s", "--config", str(CONTROLLER),
         "--model", "qwen35_08b", "--mode", "base", "--dataset", "memlens",
         "--approve-paid", "--publish"],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=60,
    )
    assert proc.returncode == 2
    assert "F-07" in proc.stderr
    assert "colab" not in proc.stdout.lower()


def test_controller_records_bundle_digest_in_job_before_upload():
    text = CONTROLLER.read_text(encoding="utf8")
    assert '"bundle_sha256"' in text
    # The digest must be computed from the same zip that is uploaded.
    assert text.index("source.zip") < text.index('"bundle_sha256"')


def test_remote_entry_does_not_install_software_at_runtime():
    text = REMOTE_ENTRY.read_text(encoding="utf8")
    assert '"pip"' not in text and "pip install" not in text
    assert "PYTHONPATH" in text


def test_remote_entry_verifies_bundle_digest_before_extracting():
    text = REMOTE_ENTRY.read_text(encoding="utf8")
    assert 'job["bundle_sha256"]' in text or "job.get(\"bundle_sha256\")" in text
    assert "hashlib.sha256" in text
    assert text.index("hashlib.sha256") < text.index("extractall")
