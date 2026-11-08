"""No imputation: summarize observed per-cell receipts and record unsupported cells."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .artifacts import atomic_json
from .config import cell_id, planned_cells


def aggregate(root: Path, config: dict) -> dict:
    root = Path(root)
    groups = defaultdict(list)
    statuses = Counter()
    rows = []
    for cell in planned_cells(config):
        path = root / "cells" / (cell_id(cell) + ".json")
        if path.exists():
            item = json.loads(path.read_text(encoding="utf8"))
            if item.get("cell") != cell:
                raise ValueError(f"Receipt does not match frozen cell: {path}")
            status = item.get("status", "UNKNOWN")
            measured = item.get("metrics") or {}
        else:
            status = "NOT_RUN"
            measured = {}
        statuses[status] += 1
        row = {**cell, "status": status,
               "ttft_p95_ms": measured.get("ttft_p95_ms"),
               "e2e_p95_ms": measured.get("e2e_p95_ms"),
               "tpot_p95_ms": measured.get("tpot_p95_ms"),
               "output_tokens_per_s": measured.get("output_tokens_per_s"),
               "peak_total_gpu_used_mib": measured.get("peak_total_gpu_used_mib")}
        rows.append(row)
        groups[(cell["dataset"], cell["model"], cell["mode"])].append(row)
    return {
        "campaign_id": config["campaign_id"], "status_counts": dict(statuses),
        "total_planned_cells": len(rows),
        "groups": [
            {"dataset": ds, "model": model, "mode": mode, "total_cells": len(group),
             "passed": sum(x["status"] == "PASS_MEASURED" for x in group)}
            for (ds, model, mode), group in sorted(groups.items())
        ],
        "rows": rows,
        "warning": "Only PASS_MEASURED with matching frozen manifests supports system performance claims",
    }


def render_markdown(summary: dict) -> str:
    lines = [
        f"# Benchmark runbook — {summary['campaign_id']}", "",
        "> Results are derived solely from per-cell receipts. Missing data is NOT_MEASURED.", "",
        f"Planned cells: **{summary['total_planned_cells']}**",
        f"Status counts: \`{json.dumps(summary['status_counts'], sort_keys=True)}\`", "",
        "| Dataset | Model | Mode | PASS / Planned |",
        "|---|---|---|---:|",
    ]
    for x in summary["groups"]:
        lines.append(f"| {x['dataset']} | {x['model']} | {x['mode']} | {x['passed']} / {x['total_cells']} |")
    lines += [
        "", "## Cell measurements", "",
        "| Dataset | Model | Mode | Input | CCU | Rep | Status | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Output tok/s | GPU used MiB |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary["rows"]:
        def f(key):
            val = r[key]
            return "NOT_MEASURED" if val is None else f"{val:.2f}"
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['mode']} | {r['input_length']} | "
            f"{r['concurrency']} | {r['repetition']} | {r['status']} | "
            f"{f('ttft_p95_ms')} | {f('tpot_p95_ms')} | {f('e2e_p95_ms')} | "
            f"{f('output_tokens_per_s')} | {f('peak_total_gpu_used_mib')} |"
        )
    lines += [
        "", "## Scientific limitations", "",
        "- CCU denotes offered parallel requests; GPU active concurrency may be smaller due to vLLM queueing.",
        "- Per-stream-chunk latency is not per-token latency under MTP batching.",
        "- GPU used MiB is total GPU memory, not physical KV pages.",
        "- Derived context bins are serving-only, not official dataset accuracy.",
        "- Official quality score requires upstream scorer and independent audit.",
        "- One A100 hardware SKU does not establish GPU-platform generalization.", "",
    ]
    return "\n".join(lines)


def write_report(root: Path, config: dict) -> dict:
    s = aggregate(root, config)
    atomic_json(Path(root) / "summary.json", s)
    (Path(root) / "RUNBOOK.md").write_text(render_markdown(s), encoding="utf8")
    return s
