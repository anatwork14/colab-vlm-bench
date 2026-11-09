"""ITER3 round 2 (C-1, C-2): regression tests for the auditor's forgeries of the inventory.

R1  a weightless model directory (config only), or a manifest that lists no weight file, must not verify.
R2  an operator-chosen weight manifest pin, outside the owner-signed record, is not a pin.
R3  an unsigned receipt index, or one whose hash is not the signed one, counts nothing.
C-2 vLLM is checked in an isolated interpreter (-I): a PYTHONPATH stub with matching metadata is refused.

The owner record is signed with ssh-keygen (namespace vlmbench-owner-compute). Trust anchor: the owner key.
Host-produced evidence (probes, hashes, receipts) is still forgeable by a local actor; see docs/harness/READINESS.md.
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
GPU_NAME = "NVIDIA A100-SXM4-40GB"
GPU_UUID = "GPU-1111aaaa-2222-3333-4444-555566667777"


def _load():
    spec = importlib.util.spec_from_file_location("harness_inventory_r2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def owner(tmp_path):
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required")
    key = tmp_path / "owner_key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers = tmp_path / "allowed_signers"
    signers.write_text("owner " + key.with_suffix(".pub").read_text().strip() + "\n", encoding="utf8")
    return {"key": key, "signers": signers}


def _signed_record(owner, tmp_path, pilot: dict, pins: dict | None) -> Path:
    # The plan the inventory binds to is the pilot config file itself (verify_owner_authorization hashes it).
    plan = tmp_path / "pilot.json"
    rec = {"campaigns": [{"campaign_id": pilot["campaign_id"], "plan_sha256": sha(plan.read_bytes())}],
           "resource": f"{GPU_NAME} x1", "approved_on": "2026-11-10"}
    if pins is not None:
        rec["pins"] = pins
    path = tmp_path / "OWNER_AUTHORIZATION.json"
    path.write_text(json.dumps(rec, indent=2), encoding="utf8")
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(owner["key"]), "-n", inv.OWNER_NAMESPACE, str(path)],
                   check=True, capture_output=True)
    return path


def _pilot_copy(tmp_path) -> tuple[dict, Path]:
    pilot = json.loads(PILOT.read_text(encoding="utf8"))
    native = tmp_path / "native.jsonl"
    native.write_bytes(b"{}\n")
    for d in pilot["datasets"]:
        d["native_path"] = str(native)
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(pilot), encoding="utf8")
    return pilot, path


def _weightless_model(tmp_path: Path) -> tuple[Path, Path]:
    """A model directory with a config and NO safetensors shard, plus a manifest listing only the config."""
    models = tmp_path / "models"
    d = models / "Qwen" / "Qwen3.5-0.8B"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}), encoding="utf8")
    manifest = d / inv.LOCAL_WEIGHTS_MANIFEST
    manifest.write_text(json.dumps({"files": {"config.json": {
        "size": (d / "config.json").stat().st_size, "sha256": sha((d / "config.json").read_bytes())}}}),
        encoding="utf8")
    return models, manifest


def _real_model(tmp_path: Path, data: bytes = b"real-weights") -> tuple[Path, Path]:
    models = tmp_path / "models"
    d = models / "Qwen" / "Qwen3.5-0.8B"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}), encoding="utf8")
    (d / "model.safetensors").write_bytes(data)
    manifest = d / inv.LOCAL_WEIGHTS_MANIFEST
    manifest.write_text(json.dumps({"files": {
        "config.json": {"size": (d / "config.json").stat().st_size,
                        "sha256": sha((d / "config.json").read_bytes())},
        "model.safetensors": {"size": len(data), "sha256": sha(data)}}}), encoding="utf8")
    return models, manifest


def _report(tmp_path, record: Path, signers: Path, models: Path, pilot_path: Path, *, receipts=None,
            index=None) -> dict:
    receipts_dir = receipts or tmp_path / "receipts"
    receipts_dir.mkdir(exist_ok=True)
    return inv.build_report(
        pilot_path, MATRIX, models_root=models, authorization=record, receipts_dir=receipts_dir,
        min_free_gib=0.0, hash_weights=True,
        probe_gpu_fn=lambda: {"name": GPU_NAME, "uuid": GPU_UUID, "compute_cap": 8.0, "memory_total_mib": 40960},
        feasibility_path=None, allowed_signers=signers, owner_record=record,
        receipt_index=index if index is not None else tmp_path / "no_index.json",
    )


# --- R1: a weightless model directory is not weights ----------------------------------------------

def test_R1_a_manifest_listing_only_the_config_does_not_verify_a_weightless_directory(tmp_path):
    models, manifest = _weightless_model(tmp_path)
    d = models / "Qwen" / "Qwen3.5-0.8B"
    probe = inv.probe_weights(d, manifest, hash_files=True, pinned_manifest_sha256=sha(manifest.read_bytes()))
    assert probe["present"] is False, "a config alone is not weights"
    assert probe["manifest_verified"] is False


def test_R1_a_zero_byte_shard_with_a_pinned_manifest_is_not_weights(tmp_path):
    models, manifest = _real_model(tmp_path, data=b"")  # an empty model.safetensors, pinned as empty
    d = models / "Qwen" / "Qwen3.5-0.8B"
    probe = inv.probe_weights(d, manifest, hash_files=True, pinned_manifest_sha256=sha(manifest.read_bytes()))
    assert probe["manifest_verified"] is False


def test_R1_a_real_shard_with_a_pinned_manifest_is_verified_positive_control(tmp_path):
    models, manifest = _real_model(tmp_path)
    d = models / "Qwen" / "Qwen3.5-0.8B"
    probe = inv.probe_weights(d, manifest, hash_files=True, pinned_manifest_sha256=sha(manifest.read_bytes()))
    assert probe["present"] is True and probe["manifest_verified"] is True


def test_R1_end_to_end_a_weightless_model_with_a_signed_manifest_pin_is_never_ready(tmp_path, owner, monkeypatch):
    # The auditor's bundle: every other precondition passes (live probe mocked to pass, vLLM mocked to verify),
    # and the owner has signed the weightless manifest's hash. Weights must still block.
    monkeypatch.setattr(inv, "run_live_probe", lambda *a, **k: {"ok": True, "reasons": []})
    monkeypatch.setattr(inv, "verify_vllm", lambda *a, **k: {"verified": True})
    models, manifest = _weightless_model(tmp_path)
    pilot, pilot_path = _pilot_copy(tmp_path)
    pins = {"models": {"qwen35_08b": {"revision": "7d209f58a7d368d8b0ba629959464b44204bcba9",
                                      "weights_manifest_sha256": sha(manifest.read_bytes())}},
            "datasets": {d["id"]: {"revision": "9f0c1d2e3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d",
                                   "native_sha256": sha((tmp_path / "native.jsonl").read_bytes())}
                         for d in pilot["datasets"]},
            "vllm": {"version": pilot["serving"]["backend_version"], "distribution_record_sha256": "ab" * 32},
            "gpu": {"name": GPU_NAME, "uuid": GPU_UUID}}
    record = _signed_record(owner, tmp_path, pilot, pins)
    report = _report(tmp_path, record, owner["signers"], models, pilot_path)
    assert report["matrices"]["pilot"]["counts"]["READY"] == 0
    assert report["matrices"]["full"]["counts"]["READY"] == 0
    qwen08 = [c for c in report["cells"] if c["matrix"] == "pilot" and c["model"] == "qwen35_08b"]
    assert qwen08 and all(c["checks"]["weights_present"] == "FAIL" for c in qwen08)
    assert all(c["status"] != "READY" for c in qwen08)


# --- R2: operator-chosen weight pins are not pins -----------------------------------------------------

def test_R2_a_weight_manifest_pin_that_is_not_the_signed_one_is_not_a_pin(tmp_path, owner):
    models, manifest = _real_model(tmp_path)
    pilot, pilot_path = _pilot_copy(tmp_path)
    other = b'{"files": {}}'  # the owner signed some other manifest's hash
    pins = {"models": {"qwen35_08b": {"revision": "7d209f58a7d368d8b0ba629959464b44204bcba9",
                                      "weights_manifest_sha256": sha(other)}}}
    record = _signed_record(owner, tmp_path, pilot, pins)
    report = _report(tmp_path, record, owner["signers"], models, pilot_path)
    w = report["environment"]["weights"]["qwen35_08b"]
    assert w["manifest_pinned"] is False and w["manifest_verified"] is False


def test_R2_the_operator_pin_flag_is_gone_from_the_cli(tmp_path):
    with pytest.raises(SystemExit):
        inv.main(["--pin-manifest", "qwen35_08b=" + "a" * 64, "--out-json", str(tmp_path / "x.json"),
                  "--out-md", str(tmp_path / "x.md")])


# --- R3: the receipt index counts only when the owner signed its hash -------------------------------

def _receipt_fixture(tmp_path: Path, cell_id: str) -> tuple[Path, Path, bytes]:
    rdir = tmp_path / "receipts"
    rdir.mkdir(exist_ok=True)
    data = json.dumps({"status": "PASS_MEASURED", "cell_id": cell_id}).encode()
    (rdir / "r.json").write_bytes(data)
    index = tmp_path / "GPU_RECEIPT_INDEX.json"
    index.write_text(json.dumps({"receipts": [{"cell_id": cell_id, "file": "r.json", "sha256": sha(data)}]}),
                     encoding="utf8")
    return rdir, index, index.read_bytes()


def test_R3_an_unsigned_receipt_index_counts_nothing_in_the_report(tmp_path, owner):
    pilot, pilot_path = _pilot_copy(tmp_path)
    models, _ = _real_model(tmp_path)
    cell_id = inv.enumerate_pilot_cells(pilot_path)[0]["cell_id"]
    rdir, index, _ = _receipt_fixture(tmp_path, cell_id)
    record = _signed_record(owner, tmp_path, pilot, pins=None)  # no receipt_index_sha256 pinned
    report = _report(tmp_path, record, owner["signers"], models, pilot_path, receipts=rdir, index=index)
    assert report["environment"]["receipts_loaded"] == 0
    cell = next(c for c in report["cells"] if c["cell_id"] == cell_id)
    assert cell["checks"]["gpu_smoke_receipt"] == "NOT_TESTED"


def test_R3_an_index_whose_signed_hash_differs_counts_nothing(tmp_path, owner):
    pilot, pilot_path = _pilot_copy(tmp_path)
    models, _ = _real_model(tmp_path)
    cell_id = inv.enumerate_pilot_cells(pilot_path)[0]["cell_id"]
    rdir, index, _ = _receipt_fixture(tmp_path, cell_id)
    record = _signed_record(owner, tmp_path, pilot, pins={"receipt_index_sha256": "cd" * 32})
    report = _report(tmp_path, record, owner["signers"], models, pilot_path, receipts=rdir, index=index)
    assert report["environment"]["receipts_loaded"] == 0


def test_R3_positive_control_a_signed_index_hash_lets_the_receipt_count(tmp_path, owner):
    pilot, pilot_path = _pilot_copy(tmp_path)
    models, _ = _real_model(tmp_path)
    cell_id = inv.enumerate_pilot_cells(pilot_path)[0]["cell_id"]
    rdir, index, index_bytes = _receipt_fixture(tmp_path, cell_id)
    record = _signed_record(owner, tmp_path, pilot, pins={"receipt_index_sha256": sha(index_bytes)})
    report = _report(tmp_path, record, owner["signers"], models, pilot_path, receipts=rdir, index=index)
    assert report["environment"]["receipts_loaded"] == 1


# --- C-2: vLLM is checked in an isolated interpreter ---------------------------------------------------

def _stub_vllm_with_metadata(tmp_path: Path) -> tuple[Path, str]:
    """A stub `vllm` package plus a dist-info directory that a non-isolated interpreter would find."""
    stub = tmp_path / "stubpath"
    (stub / "vllm").mkdir(parents=True)
    (stub / "vllm" / "__init__.py").write_text('__version__ = "0.29.0"\n', encoding="utf8")
    (stub / "vllm" / "_C.py").write_text("", encoding="utf8")
    dist = stub / "vllm-0.29.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: vllm\nVersion: 0.29.0\n", encoding="utf8")
    (dist / "RECORD").write_text("vllm/__init__.py,,\n", encoding="utf8")
    return stub, sha((dist / "RECORD").read_bytes())


def test_C2_verify_vllm_runs_isolated_and_refuses_a_pythonpath_stub(tmp_path, monkeypatch):
    stub, record_sha = _stub_vllm_with_metadata(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(stub))
    seen = []

    def spy(args, **kwargs):
        seen.append(list(args))
        return subprocess.run(args, **kwargs)

    res = inv.verify_vllm("0.29.0", runner=spy, pinned_record_sha256=record_sha)
    assert seen and seen[0][1] == "-I", "the vLLM check must run with -I (ignore PYTHONPATH and user site)"
    assert res["verified"] is False
    assert res["installed"] is False and res["kernels_importable"] is False


def test_C2_the_live_probe_is_also_isolated():
    seen = []

    def spy(args, **kwargs):
        seen.append(list(args))
        raise OSError("probe not executed in this test")

    inv.run_live_probe(GPU_NAME, "0.29.0", {}, runner=spy, gpu_uuid=GPU_UUID, vllm_record_sha256="ab" * 32)
    assert seen and seen[0][1] == "-I"


def test_positive_control_a_signed_record_authorises_the_pilot(tmp_path, owner):
    pilot, pilot_path = _pilot_copy(tmp_path)
    models, _ = _real_model(tmp_path)
    record = _signed_record(owner, tmp_path, pilot, pins={})
    report = _report(tmp_path, record, owner["signers"], models, pilot_path)
    assert report["environment"]["authorization_ok"]["pilot"] is True
