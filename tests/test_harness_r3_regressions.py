"""ITER3 round 3 regressions for the inventory's host and trust anchors.

G-PATH   host tools are run from fixed absolute locations. A PATH shim named nvidia-smi or ssh-keygen is never used.
N-R2-1   the owner-authorization anchor. Unpinned, the anchor is the operator-supplied signer file, and the report
         says so. With OWNER_KEY_FINGERPRINT pinned, only the pinned key can authorise.

Keys here are generated per test in tmp_path. No owner key is created or committed.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/harness_inventory.py"
PILOT = ROOT / "configs/a100_qwen_priority_pilot.json"
MATRIX = ROOT / "configs/a100_qwen_7methods_full.json"
CAMPAIGN = "camp-r3"


def _load():
    spec = importlib.util.spec_from_file_location("harness_inventory_r3", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _need_ssh_keygen():
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required")


def _keygen(path: Path) -> str:
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(path)], check=True)
    return path.with_suffix(".pub").read_text(encoding="utf8").strip()


def _fingerprint(pub_line: str, tmp: Path) -> str:
    key = tmp / "fp.pub"
    key.write_text(pub_line + "\n", encoding="utf8")
    out = subprocess.run(["ssh-keygen", "-lf", str(key)], capture_output=True, text=True, check=True).stdout
    return out.split()[1]


def _shim(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf8")
    path.chmod(0o755)
    return path


@pytest.fixture
def keys(tmp_path):
    _need_ssh_keygen()
    owner_key, attacker_key = tmp_path / "owner_key", tmp_path / "attacker_key"
    owner_pub, attacker_pub = _keygen(owner_key), _keygen(attacker_key)
    return {"owner_key": owner_key, "attacker_key": attacker_key,
            "owner_line": "owner " + owner_pub, "attacker_line": "owner " + attacker_pub,
            "owner_fp": _fingerprint(owner_pub, tmp_path)}


def _record(tmp_path: Path, key: Path, *, sign: bool = True) -> tuple[Path, Path]:
    plan = tmp_path / "plan.json"
    plan.write_bytes(b'{"plan": "r3"}\n')
    rec = {"campaigns": [{"campaign_id": CAMPAIGN, "plan_sha256": sha(plan.read_bytes())}],
           "resource": "NVIDIA A100-SXM4-40GB x1", "approved_on": "2026-11-10"}
    record = tmp_path / "OWNER_AUTHORIZATION.json"
    record.write_text(json.dumps(rec, indent=2), encoding="utf8")
    if sign:
        subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", inv.OWNER_NAMESPACE, str(record)],
                       check=True, capture_output=True)
    else:
        record.with_name(record.name + ".sig").write_bytes(b"not a signature\n")
    return record, plan


def _signers(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "allowed_signers"
    path.write_text("\n".join(lines) + "\n", encoding="utf8")
    return path


# --- G-PATH: fixed absolute locations, never PATH ---------------------------------------------------------

def test_G_PATH_probe_gpu_ignores_a_path_shim_nvidia_smi(tmp_path, monkeypatch):
    fixed = tmp_path / "fixed"
    _shim(fixed, "nvidia-smi", 'echo "FIXED A100, GPU-fixed, 8.0, 40960"')
    _shim(tmp_path / "shim", "nvidia-smi", 'echo "SHIM FAKE GPU, GPU-shim, 9.0, 999999"')
    monkeypatch.setattr(inv, "TRUSTED_BIN_DIRS", (str(fixed),))
    monkeypatch.setenv("PATH", f"{tmp_path / 'shim'}{os.pathsep}{os.environ.get('PATH', '')}")
    gpu = inv.probe_gpu()
    assert gpu is not None and gpu["name"] == "FIXED A100", f"PATH shim was used: {gpu}"
    assert gpu["uuid"] == "GPU-fixed"


def test_G_PATH_with_only_a_path_shim_probe_gpu_finds_nothing(tmp_path, monkeypatch):
    _shim(tmp_path / "shim", "nvidia-smi", 'echo "SHIM FAKE GPU, GPU-shim, 9.0, 999999"')
    monkeypatch.setattr(inv, "TRUSTED_BIN_DIRS", (str(tmp_path / "empty"),))
    monkeypatch.setenv("PATH", f"{tmp_path / 'shim'}{os.pathsep}{os.environ.get('PATH', '')}")
    assert inv.probe_gpu() is None


def test_G_PATH_the_trusted_directories_are_absolute():
    assert inv.TRUSTED_BIN_DIRS and all(os.path.isabs(d) for d in inv.TRUSTED_BIN_DIRS)


def test_G_PATH_a_path_shim_ssh_keygen_cannot_authorise_a_forged_record(tmp_path, monkeypatch, keys):
    # The shim always "verifies". The record is not signed by anyone (junk .sig). The check must still refuse.
    record, plan = _record(tmp_path, keys["owner_key"], sign=False)
    _shim(tmp_path / "shim", "ssh-keygen", "exit 0")
    monkeypatch.setenv("PATH", f"{tmp_path / 'shim'}{os.pathsep}{os.environ.get('PATH', '')}")
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, _signers(tmp_path, keys["owner_line"]))
    assert res["verified"] is False, res["reason"]


def test_G_PATH_positive_control_a_genuine_signature_verifies_with_a_path_shim_present(tmp_path, monkeypatch, keys):
    record, plan = _record(tmp_path, keys["owner_key"], sign=True)
    _shim(tmp_path / "shim", "ssh-keygen", "exit 1")  # a shim that always refuses must not block the trusted tool
    monkeypatch.setenv("PATH", f"{tmp_path / 'shim'}{os.pathsep}{os.environ.get('PATH', '')}")
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, _signers(tmp_path, keys["owner_line"]))
    assert res["verified"] is True, res["reason"]


# --- N-R2-1: the owner-authorization anchor ---------------------------------------------------------------

def test_N_R2_1_unpinned_anchor_is_reported_as_operator_supplied(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", None)
    record, plan = _record(tmp_path, keys["owner_key"])
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, _signers(tmp_path, keys["owner_line"]))
    assert res["verified"] is True
    assert res["anchor"] == "operator-supplied"


def test_N_R2_1_gap_documented_while_unpinned_the_operator_chosen_signer_file_is_the_anchor(tmp_path, monkeypatch, keys):
    # OPEN gap, documented in READINESS §4a. With no pin, a key the operator adds to the signer file authorises.
    # This test records that behaviour so the gap stays visible. It is not evidence the gap is closed.
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", None)
    record, plan = _record(tmp_path, keys["attacker_key"])
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, _signers(tmp_path, keys["attacker_line"]))
    assert res["verified"] is True
    assert res["anchor"] == "operator-supplied"


def test_N_R2_1_pinned_owner_key_verifies_the_owner_record(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", keys["owner_fp"])
    record, plan = _record(tmp_path, keys["owner_key"])
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, _signers(tmp_path, keys["owner_line"]))
    assert res["verified"] is True, res["reason"]
    assert res["anchor"] == f"pinned owner key {keys['owner_fp']}"


def test_N_R2_1_pinned_refuses_an_attacker_key_listed_under_the_owner_principal(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", keys["owner_fp"])
    record, plan = _record(tmp_path, keys["attacker_key"])
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, _signers(tmp_path, keys["attacker_line"]))
    assert res["verified"] is False
    assert "not in the signer file" in res["reason"]


def test_N_R2_1_pinned_refuses_an_attacker_signature_even_when_the_owner_key_is_also_listed(tmp_path, monkeypatch, keys):
    # Without the pin, the attacker's signature verifies against this same file (see the gap test above).
    signers = _signers(tmp_path, keys["owner_line"], keys["attacker_line"])
    record, plan = _record(tmp_path, keys["attacker_key"])
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", None)
    assert inv.verify_owner_authorization(record, CAMPAIGN, plan, signers)["verified"] is True
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", keys["owner_fp"])
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, signers)
    assert res["verified"] is False, res["reason"]


def test_N_R2_1_pinned_accepts_the_owner_signature_when_an_attacker_line_is_also_listed(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", keys["owner_fp"])
    record, plan = _record(tmp_path, keys["owner_key"])
    signers = _signers(tmp_path, keys["attacker_line"], "# a comment line", keys["owner_line"])
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, signers)
    assert res["verified"] is True, res["reason"]


def test_N_R2_1_pinned_with_the_owner_key_missing_from_the_signer_file_is_refused(tmp_path, monkeypatch, keys):
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", keys["owner_fp"])
    record, plan = _record(tmp_path, keys["owner_key"])
    res = inv.verify_owner_authorization(record, CAMPAIGN, plan, _signers(tmp_path, keys["attacker_line"]))
    assert res["verified"] is False
    assert "not in the signer file" in res["reason"]


def test_N_R2_1_the_report_names_the_anchor_as_operator_supplied(tmp_path, monkeypatch):
    monkeypatch.setattr(inv, "OWNER_KEY_FINGERPRINT", None)
    record = tmp_path / "OWNER_AUTHORIZATION.json"
    record.write_text("{}", encoding="utf8")
    report = inv.build_report(
        PILOT, MATRIX, models_root=tmp_path / "models", authorization=record, receipts_dir=None,
        min_free_gib=0.0, hash_weights=False, probe_gpu_fn=lambda: None, feasibility_path=None,
        allowed_signers=None, owner_record=record, receipt_index=None, live_probe=False,
    )
    text = inv.render_markdown(report)
    assert "Authorization anchor: operator-supplied" in text
    assert report["environment"]["authorization_anchor"] == "operator-supplied"
