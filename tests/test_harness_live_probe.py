"""ITER2 N-1 (REVIEW_ITER2B §3): READY needs a live probe that actually executed.

A signed owner plan plus a forged environment must not yield READY. The live probe runs in its own
isolated interpreter (python -I) with a fresh nonce; the parent checks the echoed nonce, the process
id, the vLLM distribution metadata, a CUDA kernel on the device, the GPU name against the plan's
resource, and the weight SHA-256 values recomputed in that process.

Honest limit, repeated in the report: a determined local actor who controls this machine can forge
local files and probe output. READY means "preconditions probed in this run", not an attestation.
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
    spec = importlib.util.spec_from_file_location("harness_inventory_live", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


inv = _load()
RESOURCE = "NVIDIA A100 40GB"
PIN = "0.29.0"
WEIGHTS = {"qwen35_08b": {"files": {"model.safetensors": {"sha256": "ab" * 32, "size": 5}}}}


def genuine(nonce: str, **overrides) -> dict:
    report = {
        "nonce": nonce, "pid": 4242, "python": "/venv/bin/python",
        "vllm": {"dist_version": PIN, "module_version": PIN, "module_file": "/venv/lib/vllm/__init__.py"},
        "nvidia_smi": {"path": "/usr/bin/nvidia-smi", "name": RESOURCE, "uuid": "GPU-1"},
        "cuda": {"device_name": RESOURCE, "kernel_sum": 256.0, "expected_sum": 256.0, "capability": [8, 0]},
        "weights": {"qwen35_08b": {"files": {"model.safetensors": {"sha256": "ab" * 32, "size": 5}}}},
    }
    report.update(overrides)
    return report


def evaluate(report, **kw):
    args = dict(nonce="n1", parent_pid=1, resource_name=RESOURCE, vllm_pin=PIN, weight_specs=WEIGHTS)
    args.update(kw)
    return inv.evaluate_probe(report, **args)


def test_a_genuine_probe_report_passes():
    out = evaluate(genuine("n1"))
    assert out["ok"] is True and out["reasons"] == []


def test_a_stub_vllm_without_distribution_metadata_is_refused():
    out = evaluate(genuine("n1", vllm={"dist_version": None, "module_version": PIN, "module_file": "/x"}))
    assert out["ok"] is False
    assert any("distribution metadata" in r for r in out["reasons"])


def test_vllm_that_fails_to_import_is_refused():
    out = evaluate(genuine("n1", vllm={"dist_version": PIN, "import_error": "ImportError: _C"}))
    assert out["ok"] is False and any("import" in r for r in out["reasons"])


def test_a_vllm_version_other_than_the_pin_is_refused():
    out = evaluate(genuine("n1", vllm={"dist_version": "0.28.0", "module_version": "0.28.0", "module_file": "/x"}))
    assert out["ok"] is False and any("pin" in r for r in out["reasons"])


def test_mocked_nvidia_smi_reporting_another_gpu_is_refused():
    out = evaluate(genuine("n1", nvidia_smi={"path": "/tmp/fake/nvidia-smi", "name": "mock A100", "uuid": "x"},
                           cuda={"device_name": "mock A100", "kernel_sum": 256.0, "expected_sum": 256.0,
                                 "capability": [8, 0]}))
    assert out["ok"] is False and any("GPU name" in r for r in out["reasons"])


def test_a_cuda_kernel_with_a_wrong_result_is_refused():
    out = evaluate(genuine("n1", cuda={"device_name": RESOURCE, "kernel_sum": 0.0, "expected_sum": 256.0,
                                       "capability": [8, 0]}))
    assert out["ok"] is False and any("kernel" in r for r in out["reasons"])


def test_a_recomputed_weight_hash_that_differs_from_the_pin_is_refused():
    out = evaluate(genuine("n1", weights={"qwen35_08b": {"files": {"model.safetensors": {"sha256": "cd" * 32,
                                                                                         "size": 5}}}}))
    assert out["ok"] is False and any("sha256" in r for r in out["reasons"])


def test_a_weight_file_missing_from_the_recomputation_is_refused():
    out = evaluate(genuine("n1", weights={"qwen35_08b": {"files": {}}}))
    assert out["ok"] is False and any("not recomputed" in r for r in out["reasons"])


def test_output_replayed_from_an_earlier_probe_is_refused():
    out = evaluate(genuine("old-nonce"), nonce="fresh-nonce")
    assert out["ok"] is False and any("nonce" in r for r in out["reasons"])


def test_output_produced_in_the_parent_process_is_refused():
    out = evaluate(genuine("n1", pid=77), parent_pid=77)
    assert out["ok"] is False and any("same process" in r for r in out["reasons"])


def test_a_plan_that_names_no_resource_cannot_pass():
    out = evaluate(genuine("n1"), resource_name=None)
    assert out["ok"] is False and any("resource" in r for r in out["reasons"])


def test_run_live_probe_refuses_a_replayed_nonce_from_the_runner():
    def replaying_runner(cmd, **kwargs):
        # argv is [python, -I, -c, CODE, nonce, spec]; reply with a stale nonce
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(genuine("stale")) + "\n", stderr="")

    out = inv.run_live_probe(RESOURCE, PIN, WEIGHTS, runner=replaying_runner)
    assert out["ok"] is False and any("nonce" in r for r in out["reasons"])


def test_run_live_probe_reports_a_probe_that_cannot_run():
    def broken(cmd, **kwargs):
        raise OSError("no interpreter")

    out = inv.run_live_probe(RESOURCE, PIN, WEIGHTS, runner=broken)
    assert out["ok"] is False and any("could not run" in r for r in out["reasons"])


def test_the_real_probe_on_this_host_cannot_pass_for_a_resource_it_is_not():
    # Executes the real isolated probe (nvidia-smi, torch CUDA, vLLM import) on this host.
    # The plan names a GPU this host does not have, so it must be refused whatever else is installed.
    out = inv.run_live_probe("NO SUCH GPU FOR THIS PLAN", PIN, {})
    assert out["ok"] is False
    assert any("GPU name" in r or "could not run" in r or "CUDA" in r for r in out["reasons"]), out["reasons"]


def test_full_bundle_with_a_signed_owner_plan_and_forged_environment_is_zero_ready(tmp_path, monkeypatch):
    # The N-1 reproducer shape without the signature shortcut: the owner record is signed (real ssh-keygen)
    # and every environment probe is forged. The live probe runs for real and must refuse.
    if not shutil.which("ssh-keygen"):
        pytest.skip("ssh-keygen is required")
    key, signers = tmp_path / "owner", tmp_path / "allowed_signers"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers.write_text("owner " + key.with_suffix(".pub").read_text().strip() + "\n", encoding="utf8")
    stub = tmp_path / "stub" / "vllm"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(f'__version__ = "{PIN}"\n', encoding="utf8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "stub"))

    models = tmp_path / "models"
    model_dir = models / "Qwen" / "Qwen3.5-0.8B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}),
                                           encoding="utf8")
    data = b"x"
    (model_dir / "model.safetensors").write_bytes(data)
    pilot = json.loads((ROOT / "configs/a100_qwen_priority_pilot.json").read_text(encoding="utf8"))
    matrix_path = ROOT / "configs/a100_qwen_7methods_full.json"
    for m in pilot["models"]:
        m["revision"] = "ab" * 20
    for d in pilot["datasets"]:
        d["revision"] = "cd" * 20
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(json.dumps(pilot), encoding="utf8")

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"campaign_id": pilot["campaign_id"], "gpu": RESOURCE}), encoding="utf8")
    record = tmp_path / "OWNER_AUTHORIZATION.json"
    record.write_text(json.dumps({
        "campaigns": [{"campaign_id": pilot["campaign_id"], "plan_sha256": inv.sha256_hex(plan.read_bytes())}],
        "resource": RESOURCE, "approved_on": "2026-11-10"}), encoding="utf8")
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", inv.OWNER_NAMESPACE, str(record)],
                   check=True, capture_output=True)

    receipts = tmp_path / "receipts"
    receipts.mkdir()
    report = inv.build_report(
        pilot_path, matrix_path, models_root=models, authorization=record, receipts_dir=receipts,
        min_free_gib=0.0, hash_weights=True,
        probe_gpu_fn=lambda: {"name": RESOURCE, "compute_cap": 8.0, "memory_total_mib": 40960},
        feasibility_path=None, allowed_signers=signers, pinned_manifest_sha256={}, owner_record=record,
    )
    assert report["matrices"]["pilot"]["counts"]["READY"] == 0
    assert report["matrices"]["full"]["counts"]["READY"] == 0
    assert report["environment"]["live_probe"]["ok"] is False
