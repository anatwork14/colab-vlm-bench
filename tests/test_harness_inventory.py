"""ITER2 D2-1: machine-checkable pilot and full-matrix execution inventory.

The status rules are tested with an injected environment, so no GPU, disk or network is
touched. One test runs the real probes and checks invariants only (not the host's state).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
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

SHA = "7d209f58a7d368d8b0ba629959464b44204bcba9"  # a realistic, non-degenerate revision (placeholders are refused)


def healthy_env(**overrides):
    """An environment where every check PASSES except the GPU smoke receipt (NOT_TESTED)."""
    env = {
        "weights": {
            "qwen35_08b": {"present": True, "sizes_ok": True, "hashes_ok": True,
                           "max_position_embeddings": 262144, "manifest_present": True, "manifest_verified": True},
            "qwen35_4b": {"present": True, "sizes_ok": True, "hashes_ok": True,
                          "max_position_embeddings": 262144, "manifest_present": True, "manifest_verified": True},
        },
        "revisions": {"qwen35_08b": SHA, "qwen35_4b": SHA},
        "dataset_revisions": {"memlens": SHA, "mmlongbench_doc": SHA},
        "authorization_ok": True,
        "vllm_installed": True,
        "live_probe": {"ok": True, "reasons": []},
        "gpu": {"name": "NVIDIA A100 40GB", "compute_cap": 8.0, "memory_total_mib": 40960},
        "serving_max_model_len": 32768,
        "datasets": {"memlens": {"present": True, "checksum_verified": True},
                     "mmlongbench_doc": {"present": True, "checksum_verified": True}},
        "disk_free_gib": 500.0,
        "min_free_gib": 20.0,
        "receipts": {},
    }
    env.update(overrides)
    return env


def _cell(model="qwen35_08b", arm="base", dataset="memlens", n=2048, ccu=1, rep=0):
    return {"cell_id": f"{dataset}__{model}__{arm}__r{rep}__n{n}__c{ccu}", "track": "t",
            "arm": arm, "model": model, "dataset": dataset, "input_length": n,
            "output_tokens": 256, "concurrency": ccu, "repetition": rep}


def _status(cell, env):
    return inv.classify_cell(cell, env)


# ---------------------------------------------------------------- enumeration from the repo's plan code

def test_pilot_enumeration_is_48_and_matches_planned_cells():
    from vlmbench.config import load_config, planned_cells, cell_id
    cells = inv.enumerate_pilot_cells(PILOT)
    assert len(cells) == 48
    expected = {cell_id(c) for c in planned_cells(load_config(PILOT))}
    assert {c["cell_id"] for c in cells} == expected


def test_matrix_enumeration_is_3528_and_matches_methods_plan_total():
    from vlmbench.methods import manifest_from_path, plan
    cells = inv.enumerate_matrix_cells(MATRIX)
    assert len(cells) == 3528
    assert plan(manifest_from_path(MATRIX))["total_serving_cells"] == len(cells)
    assert len({c["cell_id"] for c in cells}) == len(cells), "cell IDs must be unique"


def test_matrix_arm_split_is_seven_arms_by_two_models_by_two_datasets():
    cells = inv.enumerate_matrix_cells(MATRIX)
    arms = {c["arm"] for c in cells}
    assert arms == {"base", "h2o_base", "da_base", "ecrd_base", "mtp", "h2o_mtp", "da_mtp"}
    assert sum(c["arm"] in ("base", "mtp") for c in cells) == 1008


# ---------------------------------------------------------------- status rules (injected environment)

def test_ready_requires_every_precondition_including_a_gpu_smoke_receipt():
    env = healthy_env(receipts={_cell()["cell_id"]: True})
    st = _status(_cell(), env)
    assert st["status"] == "READY"
    assert st["blockers"] == []


def test_all_checks_pass_but_no_receipt_is_not_tested_never_ready():
    st = _status(_cell(), healthy_env())
    assert st["status"] == "NOT_TESTED"
    assert "gpu_smoke_receipt" in st["not_tested"]


def test_4b_weights_missing_is_the_primary_blocker():
    env = healthy_env()
    env["weights"]["qwen35_4b"] = {"present": False, "sizes_ok": False, "hashes_ok": None,
                                   "max_position_embeddings": None, "manifest_present": False}
    st = _status(_cell(model="qwen35_4b"), env)
    assert st["status"] == "BLOCKED_WEIGHTS_MISSING"
    assert "BLOCKED_WEIGHTS_MISSING" in st["blockers"]


def test_no_owner_authorization_blocks_paid_cells():
    st = _status(_cell(), healthy_env(authorization_ok=False))
    assert st["status"] == "BLOCKED_OWNER_AUTHORIZATION"


def test_1080ti_compute_capability_6_1_blocks_bf16_cells():
    env = healthy_env(vllm_installed=True,
                      gpu={"name": "NVIDIA GeForce GTX 1080 Ti", "compute_cap": 6.1,
                           "memory_total_mib": 11264})
    st = _status(_cell(), env)
    assert st["status"] == "BLOCKED_GPU_CAPABILITY"
    assert "BLOCKED_GPU_MEMORY" in st["blockers"]


def test_no_gpu_at_all_blocks_gpu_capability():
    st = _status(_cell(), healthy_env(gpu=None))
    assert st["status"] == "BLOCKED_GPU_CAPABILITY"


def test_vllm_not_importable_blocks():
    assert _status(_cell(), healthy_env(vllm_installed=False))["status"] == "BLOCKED_VLLM_NOT_INSTALLED"


def test_context_window_must_fit_input_plus_256_output_tokens():
    env = healthy_env()
    env["weights"]["qwen35_08b"]["max_position_embeddings"] = 4096
    st = _status(_cell(n=4096), env)  # 4096 + 256 > 4096
    assert st["status"] == "BLOCKED_CONTEXT_EXCEEDS_WINDOW"


def test_serving_max_model_len_also_caps_the_window():
    st = _status(_cell(n=16384), healthy_env(serving_max_model_len=8192))
    assert st["status"] == "BLOCKED_CONTEXT_EXCEEDS_WINDOW"


def test_disk_headroom_guard_blocks_at_11_gib():
    st = _status(_cell(), healthy_env(disk_free_gib=11.0, min_free_gib=20.0))
    assert st["status"] == "BLOCKED_DISK_HEADROOM"


def test_unpinned_model_revision_blocks_even_when_weights_are_present():
    env = healthy_env(revisions={"qwen35_08b": "UNPINNED", "qwen35_4b": SHA})
    assert _status(_cell(), env)["status"] == "BLOCKED_REVISION_UNPINNED"


def test_missing_dataset_blocks():
    env = healthy_env()
    env["datasets"]["memlens"] = {"present": False, "checksum_verified": False}
    assert _status(_cell(), env)["status"] == "BLOCKED_DATASET_MISSING"


def test_dataset_without_verified_checksum_blocks():
    env = healthy_env()
    env["datasets"]["memlens"] = {"present": True, "checksum_verified": False}
    assert _status(_cell(), env)["status"] == "BLOCKED_DATASET_UNVERIFIED"


def test_weights_without_a_matching_manifest_are_not_verified():
    env = healthy_env()
    env["weights"]["qwen35_08b"].update(hashes_ok=False, manifest_verified=False, manifest_pinned=True)
    assert _status(_cell(), env)["status"] == "BLOCKED_WEIGHTS_MISMATCH"


def test_unsupported_arm_is_unsupported_whatever_else_fails():
    env = healthy_env(authorization_ok=False, vllm_installed=False, gpu=None)
    st = _status(_cell(arm="h2o_base"), env)
    assert st["status"] == "UNSUPPORTED"
    assert st["blockers"][0] == "UNSUPPORTED_ARM_NO_QUALIFIED_BACKEND"


def test_no_cell_is_ready_without_zero_blockers():
    cells = [_cell(arm=a) for a in ("base", "mtp", "h2o_base")]
    env = healthy_env(receipts={c["cell_id"]: True for c in cells})
    for c in cells:
        st = _status(c, env)
        if st["status"] == "READY":
            assert st["blockers"] == []


def test_summary_counts_sum_to_cell_count():
    cells = [_cell(arm=a, model=m) for a in ("base", "h2o_base") for m in ("qwen35_08b", "qwen35_4b")]
    summary = inv.summarize([_status(c, healthy_env(authorization_ok=False)) for c in cells])
    assert sum(summary["counts"].values()) == len(cells)


# ---------------------------------------------------------------- probes (real I/O, on tmp files only)

def _weights_dir(tmp_path, payload=b"weights-bytes"):
    d = tmp_path / "Qwen3.5-0.8B"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"text_config": {"max_position_embeddings": 262144}}))
    (d / "model.safetensors").write_bytes(payload)
    manifest = {"files": {
        "config.json": {"size": (d / "config.json").stat().st_size,
                        "sha256": hashlib.sha256((d / "config.json").read_bytes()).hexdigest()},
        "model.safetensors": {"size": len(payload),
                              "sha256": hashlib.sha256(payload).hexdigest()},
    }}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_probe_weights_verifies_sizes_and_hashes(tmp_path):
    d = _weights_dir(tmp_path)
    p = inv.probe_weights(d, d / "manifest.json", hash_files=True)
    assert p["present"] and p["sizes_ok"] and p["hashes_ok"] is True
    assert p["max_position_embeddings"] == 262144


def test_probe_weights_detects_a_changed_file(tmp_path):
    d = _weights_dir(tmp_path)
    same_length = b"X" * len(b"weights-bytes")  # same size, different bytes
    (d / "model.safetensors").write_bytes(same_length)
    p = inv.probe_weights(d, d / "manifest.json", hash_files=True)
    assert p["sizes_ok"] is True and p["hashes_ok"] is False


def test_probe_weights_absent_directory_is_not_present(tmp_path):
    p = inv.probe_weights(tmp_path / "Qwen3.5-4B", None, hash_files=True)
    assert p["present"] is False and p["max_position_embeddings"] is None


def test_probe_weights_without_manifest_is_unverified(tmp_path):
    d = _weights_dir(tmp_path)
    p = inv.probe_weights(d, None, hash_files=True)
    assert p["present"] and p["manifest_present"] is False and p["hashes_ok"] is None


def test_probe_gpu_parses_nvidia_smi_csv():
    def runner(args, **kwargs):
        class Done:
            stdout = "NVIDIA GeForce GTX 1080 Ti, 6.1, 11264\n"
        return Done()
    gpu = inv.probe_gpu(runner=runner)
    assert gpu == {"name": "NVIDIA GeForce GTX 1080 Ti", "compute_cap": 6.1, "memory_total_mib": 11264}


def test_probe_gpu_without_nvidia_smi_returns_none():
    def runner(args, **kwargs):
        raise FileNotFoundError("nvidia-smi")
    assert inv.probe_gpu(runner=runner) is None


def test_inventory_invariants_hold_on_an_empty_models_root():
    report = inv.build_report(PILOT, MATRIX, models_root=ROOT / "models-not-here",
                              authorization=ROOT / "docs/harness/OWNER_AUTHORIZATION.json",
                              receipts_dir=None, min_free_gib=20.0, hash_weights=False,
                              probe_gpu_fn=lambda: None)
    for matrix in ("pilot", "full"):
        counts = report["matrices"][matrix]["counts"]
        assert sum(counts.values()) == report["matrices"][matrix]["cell_count"]
    for cell in report["cells"]:
        if cell["status"] == "READY":
            assert cell["blockers"] == [] and "gpu_smoke_receipt" not in cell["not_tested"]


def test_markdown_states_limits_and_regeneration_and_matches_the_json_counts(tmp_path):
    report = inv.build_report(PILOT, MATRIX, models_root=tmp_path / "none",
                              authorization=tmp_path / "none.json", receipts_dir=None,
                              min_free_gib=20.0, hash_weights=False, probe_gpu_fn=lambda: None)
    text = inv.render_markdown(report)
    assert "## 4. Assumptions and limits" in text
    assert "## 5. Regenerate" in text and "scripts/harness_inventory.py" in text
    assert f"| pilot ({report['matrices']['pilot']['campaign_id']}) | 48 |" in text
    assert f"| full ({report['matrices']['full']['campaign_id']}) | 3528 |" in text


# ITER2 D5-1: resolved dataset metadata is attached as metadata. It never changes a cell status.

def _report(tmp_path, feasibility_path):
    return inv.build_report(PILOT, MATRIX, models_root=tmp_path / "none", authorization=tmp_path / "no-auth.json",
                            receipts_dir=None, min_free_gib=3.0, hash_weights=False,
                            probe_gpu_fn=lambda: None, feasibility_path=feasibility_path)


def test_feasibility_metadata_is_attached_marked_unapproved_and_changes_no_status(tmp_path):
    with_meta = _report(tmp_path, inv.DEFAULT_FEASIBILITY)
    without = _report(tmp_path, None)
    feas = with_meta["dataset_feasibility"]
    assert feas["owner_approved"] is False
    assert feas["memlens"]["hf_revision_sha"] == "afa101a1907cc37db40b50d649547964387b96b7"
    assert feas["mmlongbench_doc"]["hf_revision_sha"] == "2ff6aa9237fc777b6627dc57a486e9225ac5fb86"
    assert without["dataset_feasibility"] is None
    assert [c["status"] for c in with_meta["cells"]] == [c["status"] for c in without["cells"]]
    assert all(c["status"] != "READY" for c in with_meta["cells"])


def test_a_failed_live_probe_blocks_an_otherwise_ready_cell():
    # ITER2 N-1: every static precondition passes, but no executed probe in this run: not READY.
    verdict = inv.classify_cell(_cell(), healthy_env(live_probe={"ok": False, "reasons": ["vLLM distribution metadata missing"]},
                                                     receipts={_cell()["cell_id"]: True}))
    assert verdict["status"] == "BLOCKED_LIVE_PROBE_FAILED"
    assert verdict["checks"]["live_probe"] == "FAIL"
    assert inv.classify_cell(_cell(), healthy_env(receipts={_cell()["cell_id"]: True}))["status"] == "READY"
