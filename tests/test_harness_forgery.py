"""ITER2 I-1 (auditor REVIEW_ITER2 §3): READY must not be reachable from fabricated preconditions.

Each test attempts one forgery from the review. A forged precondition must leave the cell
non-READY. Positive controls show the verifiers still accept genuine inputs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/harness_inventory.py"
PILOT = ROOT / "configs/a100_qwen_priority_pilot.json"
MATRIX = ROOT / "configs/a100_qwen_7methods_full.json"


def _load():
    spec = importlib.util.spec_from_file_location("harness_inventory", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def owner_keys(tmp_path):
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required for signature tests")
    owner, other = tmp_path / "owner", tmp_path / "other"
    for key in (owner, other):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers = tmp_path / "allowed_signers"
    signers.write_text("owner " + owner.with_suffix(".pub").read_text().strip() + "\n", encoding="utf8")
    return {"owner": owner, "other": other, "signers": signers}


def _plan(tmp_path: Path) -> Path:
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"campaign_id": "camp-x", "gpu": "NVIDIA A100 40GB"}), encoding="utf8")
    return plan


def _record(tmp_path: Path, plan: Path, campaign: str = "camp-x") -> Path:
    rec = {
        "campaigns": [{"campaign_id": campaign, "plan_sha256": sha(plan.read_bytes())}],
        "resource": "NVIDIA A100 40GB x1, 40 GPU-hours",
        "approved_on": "2026-11-10",
        "approved_by": "owner",
    }
    path = tmp_path / "OWNER_AUTHORIZATION.json"
    path.write_text(json.dumps(rec, indent=2), encoding="utf8")
    return path


def _sign(key: Path, record: Path) -> None:
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", inv.OWNER_NAMESPACE, str(record)],
                   check=True, capture_output=True)


# --- authorization: a bare file, even one saying owner_approved, is not authorization ---

def test_unsigned_record_claiming_approval_is_refused(tmp_path, owner_keys):
    plan = _plan(tmp_path)
    record = _record(tmp_path, plan)  # no .sig file
    res = inv.verify_owner_authorization(record, "camp-x", plan, owner_keys["signers"])
    assert res["verified"] is False


def test_record_signed_by_a_key_that_is_not_the_owner_is_refused(tmp_path, owner_keys):
    plan = _plan(tmp_path)
    record = _record(tmp_path, plan)
    _sign(owner_keys["other"], record)
    assert inv.verify_owner_authorization(record, "camp-x", plan, owner_keys["signers"])["verified"] is False


def test_owner_signed_record_for_this_plan_is_accepted(tmp_path, owner_keys):
    plan = _plan(tmp_path)
    record = _record(tmp_path, plan)
    _sign(owner_keys["owner"], record)
    assert inv.verify_owner_authorization(record, "camp-x", plan, owner_keys["signers"])["verified"] is True


def test_owner_signed_record_is_refused_when_the_plan_changed(tmp_path, owner_keys):
    plan = _plan(tmp_path)
    record = _record(tmp_path, plan)
    _sign(owner_keys["owner"], record)
    plan.write_text('{"campaign_id": "camp-x", "gpu": "NVIDIA H100"}', encoding="utf8")
    assert inv.verify_owner_authorization(record, "camp-x", plan, owner_keys["signers"])["verified"] is False


def test_without_an_owner_signer_file_nothing_is_authorized(tmp_path, owner_keys):
    plan = _plan(tmp_path)
    record = _record(tmp_path, plan)
    _sign(owner_keys["owner"], record)
    assert inv.verify_owner_authorization(record, "camp-x", plan, None)["verified"] is False


# --- vLLM: a stub package with a version string is not an installed engine ---

def test_stub_vllm_package_is_not_verified(tmp_path, monkeypatch):
    pkg = tmp_path / "vllm"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__version__ = "0.29.0"\n', encoding="utf8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    res = inv.verify_vllm("0.29.0")
    assert res["verified"] is False
    assert res["kernels_importable"] is False


def test_vllm_without_a_configured_pin_is_not_verified():
    assert inv.verify_vllm(None)["verified"] is False


# --- weights: the manifest must be pinned and must cover every weight file ---

def _model_dir(tmp_path: Path) -> tuple[Path, bytes]:
    d = tmp_path / "Qwen3.5-0.8B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}), encoding="utf8")
    data = b"weights-bytes"
    (d / "model.safetensors").write_bytes(data)
    return d, data


def test_empty_manifest_does_not_verify_weights(tmp_path):
    d, _ = _model_dir(tmp_path)
    manifest = d / "manifest.json"
    manifest.write_text(json.dumps({"files": {}}), encoding="utf8")
    p = inv.probe_weights(d, manifest, hash_files=True, pinned_manifest_sha256=sha(manifest.read_bytes()))
    assert p["manifest_verified"] is False


def test_manifest_without_a_pin_does_not_verify_and_a_pinned_one_does(tmp_path):
    d, data = _model_dir(tmp_path)
    manifest = d / "manifest.json"
    manifest.write_text(json.dumps({"files": {"model.safetensors": {"size": len(data), "sha256": sha(data)}}}),
                        encoding="utf8")
    unpinned = inv.probe_weights(d, manifest, hash_files=True, pinned_manifest_sha256=None)
    assert unpinned["manifest_verified"] is False
    pinned = inv.probe_weights(d, manifest, hash_files=True, pinned_manifest_sha256=sha(manifest.read_bytes()))
    assert pinned["manifest_verified"] is True


def test_manifest_missing_a_weight_file_does_not_verify(tmp_path):
    d, data = _model_dir(tmp_path)
    (d / "extra.safetensors").write_bytes(b"not in manifest")
    manifest = d / "manifest.json"
    manifest.write_text(json.dumps({"files": {"model.safetensors": {"size": len(data), "sha256": sha(data)}}}),
                        encoding="utf8")
    p = inv.probe_weights(d, manifest, hash_files=True, pinned_manifest_sha256=sha(manifest.read_bytes()))
    assert p["manifest_verified"] is False


# --- revisions: a repeated-character placeholder is not a pin ---

def test_degenerate_hex_is_not_a_revision_pin():
    assert inv.is_degenerate_sha("a" * 40) is True
    assert inv.is_degenerate_sha("ab" * 20) is True
    assert inv.is_degenerate_sha("7d209f58a7d368d8b0ba629959464b44204bcba9") is False


# --- receipts: only receipts listed in the committed index, with matching hashes, count ---

def _receipt(dirpath: Path, cell: str, status: str = "PASS_MEASURED") -> bytes:
    data = json.dumps({"status": status, "cell_id": cell}).encode()
    (dirpath / f"{cell}.json").write_bytes(data)
    return data


def test_receipt_missing_from_the_index_is_ignored(tmp_path):
    rdir = tmp_path / "receipts"
    rdir.mkdir()
    _receipt(rdir, "cell-1")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"receipts": []}), encoding="utf8")
    assert inv.load_receipts(rdir, index) == {}


def test_indexed_receipt_with_a_wrong_hash_is_ignored(tmp_path):
    rdir = tmp_path / "receipts"
    rdir.mkdir()
    _receipt(rdir, "cell-1")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"receipts": [{"cell_id": "cell-1", "file": "cell-1.json", "sha256": "0" * 64}]}),
                     encoding="utf8")
    assert inv.load_receipts(rdir, index) == {}


def test_indexed_receipt_with_matching_hash_counts(tmp_path):
    rdir = tmp_path / "receipts"
    rdir.mkdir()
    data = _receipt(rdir, "cell-1")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"receipts": [{"cell_id": "cell-1", "file": "cell-1.json", "sha256": sha(data)}]}),
                     encoding="utf8")
    assert inv.load_receipts(rdir, index) == {"cell-1": True}


# --- end to end: the auditor's fabricated bundle must produce zero READY cells ---

def test_full_fabricated_bundle_yields_zero_ready(tmp_path, owner_keys, monkeypatch):
    # stub vLLM package on the import path
    stub = tmp_path / "stub" / "vllm"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text('__version__ = "0.29.0"\n', encoding="utf8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))

    # fabricated models root: config plus a manifest that is not pinned
    models = tmp_path / "models"
    model_dir = models / "Qwen" / "Qwen3.5-0.8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}),
                                           encoding="utf8")
    data = b"x"
    (model_dir / "model.safetensors").write_bytes(data)
    (model_dir / "manifest.json").write_text(json.dumps({"files": {"model.safetensors": {"size": 1, "sha256": sha(data)}}}),
                                             encoding="utf8")

    # fabricated config copies: placeholder revisions and a dataset whose checksum is self-written
    native = tmp_path / "native.jsonl"
    native.write_bytes(b"{}\n")
    pilot = json.loads(PILOT.read_text(encoding="utf8"))
    for m in pilot["models"]:
        m["revision"] = "ab" * 20
    for d in pilot["datasets"]:
        d["revision"] = "cd" * 20
        d["native_path"] = str(native)
        d["native_sha256"] = sha(native.read_bytes())
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot), encoding="utf8")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(MATRIX.read_text(encoding="utf8"), encoding="utf8")

    # forged authorization: unsigned, covering both campaigns
    forged = tmp_path / "OWNER_AUTHORIZATION.json"
    forged.write_text(json.dumps({"owner_approved_compute": True,
                                  "campaign_ids": [pilot["campaign_id"], json.loads(MATRIX.read_text())["campaign_id"]]}),
                      encoding="utf8")

    receipts = tmp_path / "receipts"
    receipts.mkdir()
    report = inv.build_report(
        pilot_path, matrix_path, models_root=models, authorization=forged, receipts_dir=receipts,
        min_free_gib=0.0, hash_weights=True,
        probe_gpu_fn=lambda: {"name": "mock A100", "compute_cap": 8.0, "memory_total_mib": 40960},
        feasibility_path=None, allowed_signers=None, pinned_manifest_sha256={}, owner_record=forged,
    )
    assert report["matrices"]["pilot"]["counts"]["READY"] == 0
    assert report["matrices"]["full"]["counts"]["READY"] == 0
