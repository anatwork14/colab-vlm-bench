"""Exercise no-GPU paths of real CLI run-job using CPU-only fixtures."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlmbench import cli
from vlmbench.artifacts import atomic_json
from vlmbench.config import load_config, planned_cells, cell_id

CONFIG = Path(__file__).resolve().parents[1] / "configs/a100_qwen_priority_pilot.json"


def _prepare(monkeypatch, tmp_path):
    cfg = load_config(CONFIG)
    cfg["campaign_id"] = "cpu-preflight-fixture"
    cfg["dry_run"] = False
    cfg["storage"]["local_scratch_dir"] = str(tmp_path / "scratch")
    cfg["storage"]["drive_results_dir"] = str(tmp_path / "durable-drive")
    cfg["models"][0]["source_drive_path"] = str(tmp_path / "fake-weights")
    ds = cfg["datasets"][0]
    ds["native_path"] = str(tmp_path / "native.jsonl")
    ds["derived_path"] = str(tmp_path / "annotated.jsonl")
    Path(ds["derived_path"]).write_text('{"id":"row"}\n')
    monkeypatch.setattr(cli, "load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(cli, "checkpoint_inventory", lambda *_a, **_kw: {"total_bytes": 1})
    monkeypatch.setattr(cli, "host_resources", lambda *_a, **_kw: {})
    monkeypatch.setattr(cli, "assess_resources",
                        lambda *_a, **_kw: {"status": "CPU_PREFLIGHT_PASS", "issues": []})
    def forbidden(*args, **kwargs):
        raise AssertionError("No GPU model or staging may be accessed on blocked/pre-completed jobs")
    monkeypatch.setattr(cli, "a100_inventory", forbidden)
    monkeypatch.setattr(cli, "staged_copy", forbidden)
    monkeypatch.setattr(cli, "start_server", forbidden)
    args = SimpleNamespace(
        config=CONFIG, model="qwen35_08b", mode="base",
        dataset="memlens", quality=False, max_cells=6, owner_approved_compute=True,
    )
    return cfg, args


def test_zero_eligible_cells_returns_before_cuda_staging(monkeypatch, tmp_path):
    cfg, args = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "serving_readiness",
                        lambda *a, **kw: {
                            "ready_cells": [], "eligible_by_text_bin": {"2048": 0},
                        })
    assert cli.run_job(args) == 2
    run = json.loads(
        (Path(cfg["storage"]["local_scratch_dir"]) / cfg["campaign_id"]
         / "run_status.json").read_text())
    assert run["status"] == "BLOCKED_NO_ELIGIBLE_CELLS"


def test_existing_all_cell_receipts_skip_gpu_even_if_ready(monkeypatch, tmp_path):
    cfg, args = _prepare(monkeypatch, tmp_path)
    root = Path(cfg["storage"]["local_scratch_dir"]) / cfg["campaign_id"]
    for cell in planned_cells(cfg):
        if (cell["dataset"], cell["model"], cell["mode"]) == ("memlens", "qwen35_08b", "base"):
            atomic_json(root / "cells" / (cell_id(cell) + ".json"),
                        {"cell": cell, "status": "ERROR", "metrics": {}})
    monkeypatch.setattr(cli, "serving_readiness",
                        lambda *a, **kw: (_ for _ in ()).throw(
                            AssertionError("All cells recorded, readiness scan unnecessary")))
    assert cli.run_job(args) == 0
    run = json.loads((root / "run_status.json").read_text())
    assert run["status"] == "NO_PENDING_CELLS_NO_GPU_START"


def test_quality_image_limit_fails_without_gpu(monkeypatch, tmp_path):
    cfg, args = _prepare(monkeypatch, tmp_path)
    args.quality = True
    sample = {
        "id": "big", "dataset": "memlens", "track": "native_quality",
        "native_context_complete": True,
        "messages": [{"role": "user", "content": [
            *[{"type": "image_path", "path": "/fake/img.jpg", "asset_root": "/fake"}
              for _ in range(17)],
            {"type": "text", "text": "What is this?"}
        ]}],
    }
    Path(cfg["datasets"][0]["native_path"]).write_text(json.dumps(sample)+"\n")
    cfg["serving"]["max_images_per_prompt"] = 16
    assert cli.run_job(args) == 2
    run = json.loads((Path(cfg["storage"]["local_scratch_dir"]) / cfg["campaign_id"]
                     / "run_status.json").read_text())
    assert run["status"] == "BLOCKED_QUALITY_IMAGE_LIMIT"
