"""ITER3 C1: the owner-signed record is the only source of pins.

Revisions, dataset checksums, the weights-manifest pin, the receipt-index hash and the GPU identity come
from an owner-signed record. A config file or a command-line flag can name the same values, but neither
authorizes them. Trust anchor: the owner's signing key, which must be kept off the benchmark host.
Host-produced evidence (probes, receipts, checksums) has no external trust anchor; see docs/harness/READINESS.md.
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
PILOT = ROOT / "configs/a100_qwen_priority_pilot.json"
MATRIX = ROOT / "configs/a100_qwen_7methods_full.json"
GPU_NAME = "NVIDIA A100-SXM4-40GB"
GPU_UUID = "GPU-1111aaaa-2222-3333-4444-555566667777"


def _load():
    spec = importlib.util.spec_from_file_location("harness_inventory_pins", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load()


@pytest.fixture
def owner(tmp_path):
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required")
    key = tmp_path / "owner_key"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers = tmp_path / "allowed_signers"
    signers.write_text("owner " + key.with_suffix(".pub").read_text().strip() + "\n", encoding="utf8")
    return {"key": key, "signers": signers}


def _sign_record(owner, tmp_path, record: dict, name="OWNER_AUTHORIZATION.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(record, indent=2), encoding="utf8")
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(owner["key"]), "-n", inv.OWNER_NAMESPACE, str(path)],
                   check=True, capture_output=True)
    return path


def _plan(tmp_path, campaign_id: str, name: str) -> Path:
    plan = tmp_path / name
    plan.write_text(json.dumps({"campaign_id": campaign_id, "gpu": GPU_NAME}), encoding="utf8")
    return plan


def _record(plan_paths: dict, campaign_ids: dict, pins: dict | None) -> dict:
    rec = {
        "campaigns": [{"campaign_id": campaign_ids[k], "plan_sha256": inv.sha256_hex(plan_paths[k].read_bytes())}
                      for k in plan_paths],
        "resource": f"{GPU_NAME} x1, 40 GPU-hours",
        "approved_on": "2026-11-10",
    }
    if pins is not None:
        rec["pins"] = pins
    return rec


def _forged_configs(tmp_path):
    """Copies of the committed configs with self-chosen revisions and a self-written dataset checksum."""
    native = tmp_path / "native.jsonl"
    native.write_bytes(b"{}\n")
    pilot = json.loads(PILOT.read_text(encoding="utf8"))
    for m in pilot["models"]:
        m["revision"] = "ab" * 20
    for d in pilot["datasets"]:
        d["revision"] = "cd" * 20
        d["native_path"] = str(native)
        d["native_sha256"] = inv.sha256_hex(native.read_bytes())
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot), encoding="utf8")
    return pilot, pilot_path, native


def _models_root(tmp_path, *, manifest: bool):
    models = tmp_path / "models"
    model_dir = models / "Qwen" / "Qwen3.5-0.8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}),
                                           encoding="utf8")
    data = b"weights"
    (model_dir / "model.safetensors").write_bytes(data)
    if manifest:
        (model_dir / inv.LOCAL_WEIGHTS_MANIFEST).write_text(
            json.dumps({"files": {"model.safetensors": {"size": len(data), "sha256": inv.sha256_hex(data)}}}),
            encoding="utf8")
    return models, model_dir


def _report(tmp_path, owner_record_path, *, models, pilot_path, signers, live=None, gpu_uuid=GPU_UUID, receipts=None):
    receipts = receipts or tmp_path / "receipts"
    receipts.mkdir(exist_ok=True)
    return inv.build_report(
        pilot_path, MATRIX,
        models_root=models, authorization=owner_record_path, receipts_dir=receipts, min_free_gib=0.0,
        hash_weights=True,
        probe_gpu_fn=lambda: {"name": GPU_NAME, "uuid": gpu_uuid, "compute_cap": 8.0, "memory_total_mib": 40960},
        feasibility_path=None, allowed_signers=signers, owner_record=owner_record_path,
    )


# --- the pins that must NOT come from config, flags or unsigned files -------------------------------

def test_config_revisions_alone_do_not_pin_a_revision(tmp_path, owner, monkeypatch):
    pilot, pilot_path, _ = _forged_configs(tmp_path)
    plan = _plan(tmp_path, pilot["campaign_id"], "plan_pilot.json")
    record = _sign_record(owner, tmp_path, _record({"pilot": plan}, {"pilot": pilot["campaign_id"]}, pins=None))
    models, _ = _models_root(tmp_path, manifest=False)
    report = _report(tmp_path, record, models=models, pilot_path=pilot_path, signers=owner["signers"])
    assert all(c["checks"]["model_revision_pinned"] == "FAIL" for c in report["cells"] if c["matrix"] == "pilot")
    assert report["matrices"]["pilot"]["counts"]["READY"] == 0


def test_a_weights_manifest_pin_outside_the_signed_record_is_not_a_pin(tmp_path, owner):
    models, model_dir = _models_root(tmp_path, manifest=True)
    manifest = model_dir / inv.LOCAL_WEIGHTS_MANIFEST
    # The local actor pins their own manifest on the command line. That is not a pin.
    probe = inv.probe_weights(model_dir, manifest, hash_files=True,
                              pinned_manifest_sha256=inv.sha256_hex(manifest.read_bytes()))
    assert probe["manifest_pinned"] is True  # the function still accepts a hash it is given...
    _, pilot_path, _ = _forged_configs(tmp_path)
    plan = _plan(tmp_path, json.loads(PILOT.read_text())["campaign_id"], "plan_pilot.json")
    record = _sign_record(owner, tmp_path, _record({"pilot": plan}, {"pilot": json.loads(PILOT.read_text())["campaign_id"]}, pins=None))
    report = _report(tmp_path, record, models=models, pilot_path=pilot_path, signers=owner["signers"])
    # ...but the report takes pins only from the signed record, so this one is not pinned.
    assert report["environment"]["weights"]["qwen35_08b"]["manifest_pinned"] is False
    assert report["environment"]["weights"]["qwen35_08b"]["manifest_verified"] is False


def test_a_dataset_checksum_written_by_the_local_actor_does_not_verify_the_dataset(tmp_path, owner):
    pilot, pilot_path, _ = _forged_configs(tmp_path)  # config holds a self-written checksum
    plan = _plan(tmp_path, pilot["campaign_id"], "plan_pilot.json")
    record = _sign_record(owner, tmp_path, _record({"pilot": plan}, {"pilot": pilot["campaign_id"]}, pins=None))
    models, _ = _models_root(tmp_path, manifest=False)
    report = _report(tmp_path, record, models=models, pilot_path=pilot_path, signers=owner["signers"])
    for name, ds in report["environment"]["datasets"].items():
        assert ds["checksum_verified"] is False, name


def test_a_receipt_index_whose_hash_is_not_the_signed_pin_counts_nothing(tmp_path, owner):
    pilot, _, _ = _forged_configs(tmp_path)
    plan = _plan(tmp_path, pilot["campaign_id"], "plan_pilot.json")
    pins = {"receipt_index_sha256": "ef" * 32}  # signed, but not the hash of the index that exists
    _sign_record(owner, tmp_path, _record({"pilot": plan}, {"pilot": pilot["campaign_id"]}, pins=pins))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    index = tmp_path / "GPU_RECEIPT_INDEX.json"
    index.write_text(json.dumps({"receipts": []}), encoding="utf8")
    loaded = inv.load_receipts(receipts, index, expected_index_sha256=pins["receipt_index_sha256"])
    assert loaded == {}


def test_a_gpu_with_the_planned_name_but_another_uuid_fails_the_identity_check(tmp_path, owner):
    pilot, pilot_path, _ = _forged_configs(tmp_path)
    plan = _plan(tmp_path, pilot["campaign_id"], "plan_pilot.json")
    pins = {"gpu": {"name": GPU_NAME, "uuid": GPU_UUID}}
    record = _sign_record(owner, tmp_path, _record({"pilot": plan}, {"pilot": pilot["campaign_id"]}, pins=pins))
    models, _ = _models_root(tmp_path, manifest=False)
    report = _report(tmp_path, record, models=models, pilot_path=pilot_path, signers=owner["signers"],
                     gpu_uuid="GPU-forged-uuid")
    cell = next(c for c in report["cells"] if c["matrix"] == "pilot")
    assert cell["checks"]["gpu_identity"] == "FAIL"
    assert cell["status"] != "READY"


# --- the explicit state ladder (ITER3 C1) ----------------------------------------------------------

SHA = "7d209f58a7d368d8b0ba629959464b44204bcba9"


def _env(**overrides) -> dict:
    env = {
        "weights": {"qwen35_08b": {"present": True, "sizes_ok": True, "hashes_ok": True,
                                   "max_position_embeddings": 262144, "manifest_present": True,
                                   "manifest_pinned": True, "manifest_verified": True}},
        "revisions": {"qwen35_08b": SHA}, "dataset_revisions": {"memlens": SHA},
        "authorization_ok": True, "vllm_installed": True, "live_probe": {"ok": True, "reasons": []},
        "gpu": {"name": GPU_NAME, "uuid": GPU_UUID, "compute_cap": 8.0, "memory_total_mib": 40960},
        "gpu_pin": {"name": GPU_NAME, "uuid": GPU_UUID},
        "serving_max_model_len": 32768,
        "datasets": {"memlens": {"present": True, "checksum_verified": True}},
        "disk_free_gib": 500.0, "min_free_gib": 20.0, "receipts": {},
    }
    env.update(overrides)
    return env


CELL = {"cell_id": "memlens__qwen35_08b__base__r0__n2048__c1", "arm": "base", "model": "qwen35_08b",
        "dataset": "memlens", "input_length": 2048, "output_tokens": 256, "concurrency": 1, "repetition": 0}


def test_the_ladder_has_the_five_named_states_in_order():
    assert inv.LADDER_STATES == ("NONE", "PLAN_AUTHORIZED", "STATIC_PREREQUISITES_VALID",
                                 "LIVE_ENVIRONMENT_VERIFIED", "GPU_SMOKE_VERIFIED", "BENCHMARK_CELL_COMPLETED")


def test_every_check_sits_on_exactly_one_rung():
    on_rungs = sorted(n for _, names in inv.LADDER_RUNGS for n in names)
    assert on_rungs == sorted(name for name, _ in inv.CHECKS)


def test_a_fully_passing_cell_reaches_gpu_smoke_verified_and_is_ready_only_there():
    st = inv.classify_cell(CELL, _env(receipts={CELL["cell_id"]: True}))
    assert st["status"] == "READY"
    assert st["ladder_state"] == "GPU_SMOKE_VERIFIED"


def test_no_environment_reaches_benchmark_cell_completed():
    # the top state is part of the vocabulary but nothing in the inventory can reach it
    for env in (_env(receipts={CELL["cell_id"]: True}), _env(), _env(live_probe={"ok": False, "reasons": []})):
        assert inv.classify_cell(CELL, env)["ladder_state"] != "BENCHMARK_CELL_COMPLETED"


def test_a_failed_live_probe_stops_the_ladder_below_live_verified():
    st = inv.classify_cell(CELL, _env(live_probe={"ok": False, "reasons": ["nonce mismatch"]},
                                      receipts={CELL["cell_id"]: True}))
    assert st["ladder_state"] == "STATIC_PREREQUISITES_VALID"


def test_missing_gpu_identity_stops_the_ladder_below_live_verified():
    st = inv.classify_cell(CELL, _env(gpu_pin={"name": GPU_NAME, "uuid": "GPU-other"}))
    assert st["checks"]["gpu_identity"] == "FAIL"
    assert st["ladder_state"] == "STATIC_PREREQUISITES_VALID"


def test_no_owner_authorization_means_no_rung_at_all():
    assert inv.classify_cell(CELL, _env(authorization_ok=False))["ladder_state"] == "NONE"


def test_a_passing_rung_above_a_failing_one_does_not_count():
    # the ladder is cumulative: a passing GPU check cannot lift a failing static check
    checks = {name: "PASS" for name, _ in inv.CHECKS}
    checks["weights_verified"] = "FAIL"
    assert inv.ladder_state(checks) == "PLAN_AUTHORIZED"


def test_a_self_reported_gpu_with_the_right_name_and_no_signed_pin_fails_identity():
    # the host says "A100" and a uuid; with no owner-signed GPU pin that is only self-report
    st = inv.classify_cell(CELL, _env(gpu_pin={}))
    assert st["checks"]["gpu_identity"] == "FAIL"
    assert st["status"] != "READY"
