"""ITER2 I-2 (REVIEW_ITER2B): the committed pilot inventory must come from the current script.

A stale inventory is one whose recorded script hash differs from scripts/harness_inventory.py, or whose
recorded head is not an ancestor of HEAD. Both fail here. The inventory also may not claim READY unless
its own live probe passed in that run.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "docs/harness/PILOT_INVENTORY.json"
SCRIPT = ROOT / "scripts/harness_inventory.py"


def _committed():
    return json.loads(INVENTORY.read_text(encoding="utf8"))


def test_the_committed_inventory_was_generated_by_the_current_script():
    recorded = _committed().get("script_sha256")
    current = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert recorded == current, (
        "PILOT_INVENTORY.json is stale: regenerate with scripts/harness_inventory.py "
        f"(recorded script_sha256 {recorded}, current {current})"
    )


def test_the_recorded_head_is_a_commit_in_this_history():
    head = _committed()["head"]
    assert head and head != "UNKNOWN", "inventory has no recorded head"
    if not shutil.which("git"):
        pytest.skip("git is not available")
    done = subprocess.run(["git", "merge-base", "--is-ancestor", head, "HEAD"], cwd=ROOT,
                          capture_output=True, text=True)
    assert done.returncode == 0, f"recorded head {head} is not an ancestor of HEAD"


def test_ready_is_zero_unless_the_inventorys_own_live_probe_passed():
    data = _committed()
    live_ok = bool(data["environment"]["live_probe"]["ok"])
    for name in ("pilot", "full"):
        if not live_ok:
            assert data["matrices"][name]["counts"]["READY"] == 0, name
