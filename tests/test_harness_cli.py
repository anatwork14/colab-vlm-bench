"""ITER2 N-2 (REVIEW_ITER2B §3): the inventory CLI runs the same probes as the library, or says plan-only.

The CLI must not be able to produce a report that silently lacks the owner inputs and the live probe.
These tests drive main() on the committed configs with an empty models root: no GPU job, no download.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/harness_inventory.py"


def _load():
    spec = importlib.util.spec_from_file_location("harness_inventory_cli", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load()
PILOT = ROOT / "configs/a100_qwen_priority_pilot.json"
MATRIX = ROOT / "configs/a100_qwen_7methods_full.json"


def _run(tmp_path, *extra):
    out_json, out_md = tmp_path / "inv.json", tmp_path / "inv.md"
    argv = ["--pilot-config", str(PILOT), "--matrix-manifest", str(MATRIX),
            "--models-root", str(tmp_path / "no_models"), "--authorization", str(tmp_path / "none.json"),
            "--min-free-gib", "0", "--out-json", str(out_json), "--out-md", str(out_md), *extra]
    assert inv.main(argv) == 0
    return json.loads(out_json.read_text(encoding="utf8")), out_md.read_text(encoding="utf8")


def test_plan_only_says_so_in_the_json_and_the_markdown_and_makes_no_ready(tmp_path):
    report, md = _run(tmp_path, "--plan-only")
    assert report["mode"] == "plan_only"
    assert report["environment"]["live_probe"]["ok"] is False
    assert "plan-only" in report["environment"]["live_probe"]["reasons"][0]
    assert report["matrices"]["pilot"]["counts"].get("READY", 0) == 0
    assert "**Mode:** `plan_only`" in md


def test_live_mode_is_the_default_and_the_markdown_states_the_limit(tmp_path):
    report, md = _run(tmp_path)
    assert report["mode"] == "live_probes"
    assert "**Mode:** `live_probes`" in md
    assert "not an attestation" in md
    assert report["matrices"]["full"]["counts"].get("READY", 0) == 0


def test_the_cli_with_a_signed_owner_record_runs_the_live_probe_not_a_placeholder(tmp_path):
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required")
    key, signers = tmp_path / "owner", tmp_path / "allowed_signers"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers.write_text("owner " + key.with_suffix(".pub").read_text().strip() + "\n", encoding="utf8")
    pilot = json.loads(PILOT.read_text(encoding="utf8"))
    plan = tmp_path / "plan.json"
    plan.write_bytes(PILOT.read_bytes())
    record = tmp_path / "OWNER_AUTHORIZATION.json"
    record.write_text(json.dumps({
        "campaigns": [{"campaign_id": pilot["campaign_id"],
                       "plan_sha256": inv.sha256_hex(PILOT.read_bytes())}],
        "resource": "NVIDIA A100 40GB x1", "approved_on": "2026-11-10"}), encoding="utf8")
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", inv.OWNER_NAMESPACE, str(record)],
                   check=True, capture_output=True)
    report, _ = _run(tmp_path, "--authorization", str(record), "--allowed-signers", str(signers))
    assert report["environment"]["authorization_ok"]["pilot"] is True
    live = report["environment"]["live_probe"]
    assert not any(r.startswith("not run") for r in live["reasons"])  # the probe actually executed
    assert live["ok"] is False  # and it refused this host: no vLLM distribution, GPU not the plan's resource
    assert report["matrices"]["pilot"]["counts"].get("READY", 0) == 0
