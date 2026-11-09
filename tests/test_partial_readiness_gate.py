"""Regression: partial annotation must not poison future eligible cells.

CPU-only test; fake GPU calls and fake vLLM startup. No model inference.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

from vlmbench import cli
from vlmbench.artifacts import atomic_json
from vlmbench.config import cell_id, load_config, planned_cells

TEMPLATE = Path(__file__).resolve().parents[1] / "configs/a100_qwen_priority_pilot.json"


def test_eligible_pending_cells_preserves_unready_bins_for_future_annotation():
    cells = [
        {"input_length": length, "concurrency": ccu, "repetition": 0}
        for length in (2048, 16384) for ccu in (1, 8, 32)
    ]
    readiness = {"ready_cells": [
        {"text_bin": 2048, "ccu": 1, "repetition": 0,
         "eligible": 24, "required": 17}
    ]}
    selected = cli.eligible_pending_cells(cells, readiness)
    assert selected == [cells[0]]
    assert cli.eligible_pending_cells(cells, {"ready_cells": []}) == []


def test_partial_readiness_runs_only_eligible_and_does_not_write_failure_receipts(
    tmp_path, monkeypatch,
):
    cfg = load_config(TEMPLATE)
    cfg["campaign_id"] = "cpu-partial-readiness-test"
    cfg["dry_run"] = False
    cfg["storage"]["local_scratch_dir"] = str(tmp_path / "scratch")
    cfg["storage"]["drive_results_dir"] = str(tmp_path / "durable-drive")
    cfg["models"][0]["source_drive_path"] = str(tmp_path / "fake-model-drive")
    cfg["datasets"][0]["native_path"] = str(tmp_path / "native.jsonl")
    cfg["datasets"][0]["derived_path"] = str(tmp_path / "annotated.jsonl")
    Path(cfg["datasets"][0]["derived_path"]).write_text('{"id":"sample"}\n')
    monkeypatch.setattr(cli, "load_config", lambda *_a, **_kw: cfg)
    monkeypatch.setattr(cli, "checkpoint_inventory", lambda *_a, **_kw: {
        "total_bytes": 5})
    monkeypatch.setattr(cli, "host_resources", lambda *_a, **_kw: {})
    monkeypatch.setattr(cli, "assess_resources", lambda *_a, **_kw: {
        "status": "CPU_PREFLIGHT_PASS", "issues": []})
    monkeypatch.setattr(cli, "serving_readiness", lambda *_a, **_kw: {
        "status": "READY_CELLS_PRESENT",
        "eligible_by_text_bin": {"2048": 24, "16384": 0},
        "ready_cells": [{"text_bin": 2048, "ccu": 1, "repetition": 0,
                         "eligible": 24, "required": 17}],
    })
    monkeypatch.setattr(cli, "a100_inventory", lambda: {
        "gpu_name": "FAKE-A100-NO-GPU-USED"})
    monkeypatch.setattr(cli, "_installed", lambda name: "0.29.0")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="fake", version=SimpleNamespace(cuda="fake"),
        cuda=SimpleNamespace(is_available=lambda: True,
                             is_bf16_supported=lambda: True)))
    monkeypatch.setattr(cli, "staged_copy", lambda *_a, **_kw: {
        "local_path": str(tmp_path / "fake-staged"), "files": []})
    monkeypatch.setattr(cli, "server_argv", lambda *_a, **_kw: ["fake-vllm"])
    monkeypatch.setattr(cli, "start_server", lambda *_a, **_kw: object())
    monkeypatch.setattr(cli, "wait_ready", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "stop_server", lambda *_a, **_kw: None)
    monkeypatch.setattr(cli, "load_jsonl", lambda *_a, **_kw: [{"id": "fake"}])
    monkeypatch.setattr(cli, "write_report", lambda *_a, **_kw: {})
    monkeypatch.setattr(cli, "sync_files", lambda *_a, **_kw: None)

    attempted = []
    async def fake_cell(_cfg, cell, _rows, root, _drive, _model, _digest):
        attempted.append(cell_id(cell))
        atomic_json(Path(root) / "cells" / (cell_id(cell) + ".json"), {
            "cell": cell, "status": "PASS_MEASURED",
            "metrics": {"output_tokens_per_s": 1.0}})
        return {"status": "PASS_MEASURED"}
    monkeypatch.setattr(cli, "run_performance_cell", fake_cell)

    args = SimpleNamespace(config=TEMPLATE, model="qwen35_08b",
                           mode="base", dataset="memlens",
                           quality=False, max_cells=6)
    assert cli.run_job(args) == 0
    root = Path(cfg["storage"]["local_scratch_dir"]) / cfg["campaign_id"]
    all_cells = [c for c in planned_cells(cfg)
                 if c["model"] == "qwen35_08b" and c["dataset"] == "memlens"
                 and c["mode"] == "base"]
    assert len(all_cells) == 6
    assert attempted == [cell_id(all_cells[0])]
    assert (root / "cells" / (cell_id(all_cells[0]) + ".json")).is_file()
    assert all(not (root / "cells" / (cell_id(c) + ".json")).exists()
               for c in all_cells[1:])
    saved = json.loads((root / "run_status.json").read_text())
    assert saved["cells_attempted_this_invocation"] == 1
