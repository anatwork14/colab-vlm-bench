"""F-05: resume must skip only COMPLETED cell receipts; failed or stale receipts are
flagged (default) or quarantined and re-run (explicit requeue). Valid completed receipts
are never overwritten. All GPU, server and model calls are faked; no paid action."""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlmbench import cli, resume
from vlmbench.artifacts import atomic_json
from vlmbench.config import cell_id, fingerprint, load_config, planned_cells

CONFIG = Path(__file__).resolve().parents[1] / "configs/a100_qwen_priority_pilot.json"
CODE_SHA = "a" * 40
SCOPE = {"model": "qwen35_08b", "dataset": "memlens", "mode": "base"}


def _cfg(tmp_path):
    cfg = copy.deepcopy(load_config(CONFIG))
    cfg["campaign_id"] = "resume-policy-fixture"
    cfg["dry_run"] = False
    cfg["storage"]["local_scratch_dir"] = str(tmp_path / "scratch")
    cfg["storage"]["drive_results_dir"] = str(tmp_path / "drive")
    cfg["models"][0]["source_drive_path"] = str(tmp_path / "fake-weights")
    ds = cfg["datasets"][0]
    ds["native_path"] = str(tmp_path / "native.jsonl")
    ds["derived_path"] = str(tmp_path / "annotated.jsonl")
    Path(ds["derived_path"]).write_text('{"id":"row"}\n', encoding="utf8")
    Path(ds["native_path"]).write_text("", encoding="utf8")  # presence only; loader is faked
    return cfg


def _identity(cfg, dataset_sha, code=CODE_SHA):
    return {
        "config_sha256": fingerprint(cfg),
        "dataset_sha256": dataset_sha,
        "code_sha256": code,
        "execution_pass": "FIXED_OUTPUT_SERVING_256",
        "output_contract": {"max_tokens": 256, "min_tokens": 256, "ignore_eos": True},
    }


def _scoped(cfg):
    return [c for c in planned_cells(cfg)
            if c["model"] == SCOPE["model"] and c["dataset"] == SCOPE["dataset"]
            and c["mode"] == SCOPE["mode"]]


def _pass(cell, identity, n_req=None):
    n = cell["requests"] if n_req is None else n_req
    return {**identity, "cell": cell, "status": "PASS_MEASURED",
            "sample_id_sha256": "SAMPLE", "metrics": {"ttft_p95_ms": 1.0},
            "successful_requests": n, "expected_requests": cell["requests"]}


def _error(cell, identity):
    return {**identity, "cell": cell, "status": "ERROR", "error": "boom", "metrics": {}}


def _write(path: Path, value) -> bytes:
    atomic_json(path, value)
    return path.read_bytes()


# ---------- classification (pure) ----------

def test_pass_receipt_with_matching_identity_is_completed(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _pass(cell, _identity(cfg, "D")))
    state, _ = resume.classify_cell_receipt(path, _identity(cfg, "D"), cell, "SAMPLE")
    assert state == resume.COMPLETED


def test_missing_receipt_is_missing(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    state, _ = resume.classify_cell_receipt(tmp_path / "none.json", _identity(cfg, "D"), cell, "S")
    assert state == resume.MISSING


def test_error_receipt_with_matching_identity_is_failed_not_completed(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _error(cell, _identity(cfg, "D")))
    state, reason = resume.classify_cell_receipt(path, _identity(cfg, "D"), cell, "SAMPLE")
    assert state == resume.FAILED
    assert "ERROR" in reason


def test_stale_dataset_hash_is_stale_even_if_passed(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _pass(cell, _identity(cfg, "OLD")))
    state, reason = resume.classify_cell_receipt(path, _identity(cfg, "NEW"), cell, "SAMPLE")
    assert state == resume.STALE
    assert "dataset_sha256" in reason


def test_stale_code_hash_is_stale_even_if_passed(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _pass(cell, _identity(cfg, "D", code="b" * 40)))
    state, reason = resume.classify_cell_receipt(path, _identity(cfg, "D"), cell, "SAMPLE")
    assert state == resume.STALE
    assert "code_sha256" in reason


def test_sample_set_mismatch_is_stale(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _pass(cell, _identity(cfg, "D")))
    state, reason = resume.classify_cell_receipt(path, _identity(cfg, "D"), cell, "OTHER")
    assert state == resume.STALE
    assert "sample" in reason


def test_pass_receipt_without_verifiable_sample_set_is_stale(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _pass(cell, _identity(cfg, "D")))
    state, _ = resume.classify_cell_receipt(path, _identity(cfg, "D"), cell, None)
    assert state == resume.STALE


def test_pass_receipt_under_unrecorded_code_identity_is_not_completed(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _pass(cell, _identity(cfg, "D", code="UNRECORDED")))
    state, reason = resume.classify_cell_receipt(
        path, _identity(cfg, "D", code="UNRECORDED"), cell, "SAMPLE")
    assert state == resume.STALE
    assert "UNRECORDED" in reason


def test_pass_receipt_with_short_request_count_is_invalid(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    _write(path, _pass(cell, _identity(cfg, "D"), n_req=cell["requests"] - 1))
    state, _ = resume.classify_cell_receipt(path, _identity(cfg, "D"), cell, "SAMPLE")
    assert state == resume.INVALID


def test_unreadable_receipt_is_invalid(tmp_path):
    cfg = _cfg(tmp_path)
    cell = _scoped(cfg)[0]
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf8")
    state, _ = resume.classify_cell_receipt(path, _identity(cfg, "D"), cell, "SAMPLE")
    assert state == resume.INVALID


# ---------- run_job integration with faked GPU ----------

def _fake_gpu(monkeypatch, tmp_path, cfg, calls, outcome):
    """Replace every GPU/model touchpoint. `outcome(cell)` returns 'PASS' or 'ERROR'."""
    monkeypatch.setenv("VLMBENCH_SOURCE_SHA", CODE_SHA)
    monkeypatch.setattr(cli, "load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(cli, "checkpoint_inventory", lambda *_a, **_kw: {"total_bytes": 1})
    monkeypatch.setattr(cli, "host_resources", lambda *_a, **_kw: {})
    monkeypatch.setattr(cli, "assess_resources",
                        lambda *_a, **_kw: {"status": "CPU_PREFLIGHT_PASS", "issues": []})
    monkeypatch.setattr(cli, "serving_readiness",
                        lambda *a, **kw: {"ready_cells": [1], "eligible_by_text_bin": {}})
    monkeypatch.setattr(cli, "load_jsonl", lambda *a, **kw: [])
    monkeypatch.setattr(cli, "_expected_sample_sha", lambda *a, **kw: "SAMPLE")
    monkeypatch.setattr(cli, "a100_inventory", lambda: {"name": "fake-A100"})
    monkeypatch.setattr(cli, "_installed", lambda name: cfg["serving"]["backend_version"])
    monkeypatch.setattr(cli, "staged_copy",
                        lambda *_a, **_kw: {"local_path": str(tmp_path / "stage"), "files": []})
    monkeypatch.setattr(cli, "server_argv", lambda *a, **kw: ["fake-server"])
    monkeypatch.setattr(cli, "start_server", lambda *a, **kw: object())
    monkeypatch.setattr(cli, "wait_ready", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "stop_server", lambda *a, **kw: None)
    monkeypatch.setattr(cli, "write_report", lambda *a, **kw: None)
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "fake"
    fake_torch.version = SimpleNamespace(cuda="fake")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True,
                                      is_bf16_supported=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    async def fake_cell(cfg_, cell, rows, root, drive_root, model_id, dataset_sha256):
        calls.append(cell_id(cell))
        identity = _identity(cfg_, dataset_sha256)
        receipt = (_pass(cell, identity) if outcome(cell) == "PASS"
                   else _error(cell, identity))
        atomic_json(root / "cells" / f"{cell_id(cell)}.json", receipt)
        return receipt

    monkeypatch.setattr(cli, "run_performance_cell", fake_cell)


def _args(**kw):
    base = dict(config=CONFIG, model=SCOPE["model"], mode=SCOPE["mode"],
                dataset=SCOPE["dataset"], quality=False, max_cells=6,
                owner_approved_compute=True, requeue_nonpass=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _root(cfg):
    return Path(cfg["storage"]["local_scratch_dir"]) / cfg["campaign_id"]


def _derived_sha(cfg):
    from vlmbench.artifacts import sha256
    return sha256(cfg["datasets"][0]["derived_path"])


def test_resume_after_partial_failure_reruns_only_the_failed_cell(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cells = _scoped(cfg)
    root = _root(cfg)
    ident = _identity(cfg, _derived_sha(cfg))
    done_bytes = _write(root / "cells" / f"{cell_id(cells[0])}.json", _pass(cells[0], ident))
    err_path = root / "cells" / f"{cell_id(cells[1])}.json"
    err_bytes = _write(err_path, _error(cells[1], ident))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "PASS")

    assert cli.run_job(_args(requeue_nonpass=True)) == 0

    assert calls == [cell_id(c) for c in cells[1:]], "only the failed and missing cells run"
    assert (root / "cells" / f"{cell_id(cells[0])}.json").read_bytes() == done_bytes
    quarantined = list((root / "quarantine").rglob(f"{cell_id(cells[1])}.json"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == err_bytes
    assert json.loads(err_path.read_text())["status"] == "PASS_MEASURED"


def test_stale_hash_receipt_is_quarantined_and_rerun_not_skipped(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cells = _scoped(cfg)
    root = _root(cfg)
    stale_bytes = _write(root / "cells" / f"{cell_id(cells[0])}.json",
                         _pass(cells[0], _identity(cfg, "OLD-DATASET")))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "PASS")

    assert cli.run_job(_args(requeue_nonpass=True)) == 0

    assert cell_id(cells[0]) in calls, "a stale receipt must never be skipped"
    quarantined = list((root / "quarantine").rglob(f"{cell_id(cells[0])}.json"))
    assert len(quarantined) == 1 and quarantined[0].read_bytes() == stale_bytes


def test_quarantine_also_moves_drive_copy_so_restore_cannot_resurrect_it(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cells = _scoped(cfg)
    root = _root(cfg)
    drive = Path(cfg["storage"]["drive_results_dir"]) / cfg["campaign_id"]
    ident = _identity(cfg, _derived_sha(cfg))
    _write(root / "cells" / f"{cell_id(cells[0])}.json", _error(cells[0], ident))
    _write(drive / "cells" / f"{cell_id(cells[0])}.json", _error(cells[0], ident))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "PASS")

    assert cli.run_job(_args(requeue_nonpass=True)) == 0

    moved = list((drive / "quarantine").rglob(f"{cell_id(cells[0])}.json"))
    assert len(moved) == 1
    assert json.loads(moved[0].read_text())["status"] == "ERROR"


def test_noncomplete_receipts_block_gpu_by_default_and_are_not_touched(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cells = _scoped(cfg)
    root = _root(cfg)
    err_bytes = _write(root / "cells" / f"{cell_id(cells[0])}.json",
                       _error(cells[0], _identity(cfg, _derived_sha(cfg))))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "PASS")

    assert cli.run_job(_args()) == 2
    run = json.loads((root / "run_status.json").read_text())
    assert run["status"] == "BLOCKED_NONCOMPLETE_RECEIPTS"
    assert run["receipts"][0]["state"] == resume.FAILED
    assert calls == []
    assert (root / "cells" / f"{cell_id(cells[0])}.json").read_bytes() == err_bytes
    assert not (root / "quarantine").exists()


def test_completed_cells_are_skipped_without_gpu_or_rewrite(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cells = _scoped(cfg)
    root = _root(cfg)
    ident = _identity(cfg, _derived_sha(cfg))
    bytes_by_cell = {}
    for cell in cells:
        bytes_by_cell[cell_id(cell)] = _write(root / "cells" / f"{cell_id(cell)}.json",
                                              _pass(cell, ident))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "PASS")

    assert cli.run_job(_args()) == 0
    assert calls == []
    for cell in cells:
        path = root / "cells" / f"{cell_id(cell)}.json"
        assert path.read_bytes() == bytes_by_cell[cell_id(cell)]


# ---------- F-06: exit status must reflect cell outcomes ----------

def _non_pilot(cfg):
    cfg["execution_scope"] = {**(cfg.get("execution_scope") or {}), "kind": "full_study"}
    return cfg


def _status(tmp_path_cfg):
    return json.loads((_root(tmp_path_cfg) / "run_status.json").read_text())


def test_every_attempted_cell_failing_exits_nonzero_with_all_failed_status(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "ERROR")

    rc = cli.run_job(_args())

    assert rc == 5
    run = _status(cfg)
    assert run["status"] == "ALL_ATTEMPTED_FAILED"
    assert run["counts"]["attempted"] == len(calls) == 6
    assert run["counts"]["completed"] == 0


def test_mixed_outcomes_exit_nonzero_with_partial_failure_status(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    cells = _scoped(cfg)
    fail_ids = {cell_id(c) for c in cells[1::2]}
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls,
              lambda cell: "ERROR" if cell_id(cell) in fail_ids else "PASS")

    rc = cli.run_job(_args())

    assert rc == 6
    run = _status(cfg)
    assert run["status"] == "PARTIAL_FAILURE"
    assert run["counts"] == {"planned": 6, "attempted": 6, "completed": 3, "non_completed": 3}


def test_all_cells_completed_exits_zero_with_complete_status(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "PASS")

    assert cli.run_job(_args()) == 0
    run = _status(cfg)
    assert run["status"] == "COMPLETE"
    assert run["counts"]["completed"] == 6


def test_cell_cap_reached_before_finish_exits_zero_in_progress(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "PASS")

    assert cli.run_job(_args(max_cells=2)) == 0
    run = _status(cfg)
    assert run["status"] == "IN_PROGRESS"
    assert run["counts"]["completed"] == 2 and run["counts"]["non_completed"] == 0


def test_pilot_circuit_breaker_has_its_own_nonzero_status(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)  # engineering pilot scope
    assert cfg["execution_scope"]["kind"] == "engineering_pilot"
    calls: list[str] = []
    _fake_gpu(monkeypatch, tmp_path, cfg, calls, lambda cell: "ERROR")

    assert cli.run_job(_args()) == 4
    assert _status(cfg)["status"] == "PILOT_STOPPED_ON_FAILURE"
    assert len(calls) == 1


def _quality_args():
    return _args(quality=True)


def _fake_quality(monkeypatch, receipt):
    async def fake(*_a, **_kw):
        return receipt
    monkeypatch.setattr(cli, "run_quality_predictions", fake)


def test_quality_all_predictions_failed_exits_nonzero(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    _fake_gpu(monkeypatch, tmp_path, cfg, [], lambda cell: "PASS")
    _fake_quality(monkeypatch, {"status": "QUALITY_PREDICTIONS_UNSCORED",
                                "count": 2, "expected_count": 2, "failures": 2})
    assert cli.run_job(_quality_args()) == 5
    assert _status(cfg)["status"] == "QUALITY_ALL_FAILED"


def test_quality_circuit_breaker_exits_nonzero(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    _fake_gpu(monkeypatch, tmp_path, cfg, [], lambda cell: "PASS")
    _fake_quality(monkeypatch, {"status": "QUALITY_STOPPED_AFTER_REPEATED_FAILURES",
                                "count": 2, "expected_count": 9, "failures": 2})
    assert cli.run_job(_quality_args()) == 4
    assert _status(cfg)["status"] == "QUALITY_STOPPED_AFTER_REPEATED_FAILURES"


def test_quality_partial_failures_exit_nonzero(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    _fake_gpu(monkeypatch, tmp_path, cfg, [], lambda cell: "PASS")
    _fake_quality(monkeypatch, {"status": "QUALITY_PREDICTIONS_UNSCORED",
                                "count": 4, "expected_count": 4, "failures": 1})
    assert cli.run_job(_quality_args()) == 6
    assert _status(cfg)["status"] == "QUALITY_PARTIAL_FAILURE"


def test_quality_success_exits_zero(monkeypatch, tmp_path):
    cfg = _non_pilot(_cfg(tmp_path))
    _fake_gpu(monkeypatch, tmp_path, cfg, [], lambda cell: "PASS")
    _fake_quality(monkeypatch, {"status": "QUALITY_PREDICTIONS_UNSCORED",
                                "count": 4, "expected_count": 4, "failures": 0})
    assert cli.run_job(_quality_args()) == 0
    assert _status(cfg)["status"] == "QUALITY_PREDICTIONS_UNSCORED"


def test_remote_benchmark_wrapper_fails_and_names_the_status(monkeypatch, tmp_path):
    # The remote wrapper treats a raised RuntimeError as ACTION_FAILED; the message must
    # name the run status instead of reporting GPU_JOB_FINISHED after an all-failed run.
    from vlmbench import agent_pipeline as ap

    cfg = _non_pilot(_cfg(tmp_path))
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}", encoding="utf8")
    annotated = tmp_path / "annotated.jsonl"
    annotated.write_text("", encoding="utf8")
    root = _root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "run_status.json", {"status": "ALL_ATTEMPTED_FAILED", "exit_code": 5})
    seen = {}

    def fake_run_job(args):
        seen["requeue"] = args.requeue_nonpass
        return 5

    monkeypatch.setattr(ap, "require_compute_approval", lambda *_a, **_kw: None)
    monkeypatch.setattr(ap, "paths_for", lambda job: {"annotated": annotated})
    monkeypatch.setattr(ap, "require_prepared", lambda *_a, **_kw: None)
    monkeypatch.setattr(ap, "campaign", lambda job, paths: (cfg, cfg_path))
    import vlmbench.cli as cli_mod
    monkeypatch.setattr(cli_mod, "run_job", fake_run_job)
    job = {"model": "qwen35_08b", "mode": "base", "dataset": "memlens", "phase": "pilot",
           "quality": False, "max_cells": 1, "requeue_nonpass": True,
           "owner_approved_compute": True}

    with pytest.raises(RuntimeError, match="ALL_ATTEMPTED_FAILED"):
        ap.benchmark(job)
    assert seen["requeue"] is True

    with pytest.raises(ValueError, match="requeue_nonpass must be a boolean"):
        ap.benchmark({**job, "requeue_nonpass": "yes"})
