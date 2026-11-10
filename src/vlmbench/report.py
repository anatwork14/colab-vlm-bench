"""Receipt-only reporting, native quality status and paired Base/MTP speedups."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from .analysis import E2E_LEGACY, e2e_definition, paired_speedups
from .artifacts import atomic_json
from .config import cell_id, planned_cells

METRICS = ("ttft_p95_ms", "tpot_p95_ms", "e2e_p95_ms",
           "output_tokens_per_s", "peak_total_gpu_used_mib")


def aggregate(root: Path, config: dict) -> dict:
    root = Path(root)
    grouped = defaultdict(list)
    statuses = Counter()
    rows, receipts = [], []
    for cell in planned_cells(config):
        path = root / "cells" / (cell_id(cell) + ".json")
        raw = json.loads(path.read_text(encoding="utf8")) if path.is_file() else {}
        if raw and raw.get("cell") != cell:
            raise ValueError(f"Invalid receipt cell identity: {path}")
        status = raw.get("status", "NOT_RUN")
        observed = raw.get("metrics") or {}
        statuses[status] += 1
        receipts.append({"cell": cell, "status": status,
                         "sample_id_sha256": raw.get("sample_id_sha256"),
                         "metrics": observed})
        row = {**cell, "status": status}
        row.update({metric: observed.get(metric) for metric in METRICS})
        # R3-F: a receipt written before the client queue wait was part of E2E measured a
        # different quantity. Its E2E value is withheld, so it cannot be pooled or compared.
        row["e2e_definition"] = e2e_definition(raw) if raw else None
        if raw and row["e2e_definition"] == E2E_LEGACY and row["e2e_p95_ms"] is not None:
            row["e2e_p95_ms"] = None
            row["e2e_p95_status"] = "NOT_COMPARABLE_LEGACY_DEFINITION"
        elif row["e2e_p95_ms"] is not None:
            row["e2e_p95_status"] = "COMPARABLE"
        else:
            row["e2e_p95_status"] = "NOT_MEASURED"
        rows.append(row)
        grouped[(cell["dataset"], cell["model"], cell["mode"])].append(row)
    quality = []
    for path in sorted((root / "quality").glob("*.receipt.json")):
        raw = json.loads(path.read_text(encoding="utf8"))
        quality.append({"filename": path.name, "status": raw.get("status"),
                        "count": raw.get("count"), "failures": raw.get("failures"),
                        "official_accuracy": raw.get("official_accuracy", "NOT_MEASURED")})
    pass_defs = {r["e2e_definition"] for r in rows
                 if r["status"] == "PASS_MEASURED" and r["e2e_definition"]}
    return {
        "campaign_id": config["campaign_id"], "status_counts": dict(statuses),
        "total_planned_cells": len(rows),
        "e2e_definitions_present": sorted(pass_defs),
        "e2e_comparability": ("MIXED_NOT_COMPARABLE" if len(pass_defs) > 1
                              else "SINGLE_DEFINITION" if pass_defs else "NO_E2E_DATA"),
        "groups": [
            {"dataset": ds, "model": model, "mode": mode, "total_cells": len(group),
             "passed": sum(x["status"] == "PASS_MEASURED" for x in group)}
            for (ds, model, mode), group in sorted(grouped.items())
        ],
        "paired_comparisons": paired_speedups(receipts),
        "quality_receipts": quality, "rows": rows,
        "warning": "Only matching PASS_MEASURED data support performance claims",
    }


def render_markdown(summary: dict) -> str:
    lines = [
        f"# Benchmark runbook — {summary['campaign_id']}", "",
        "> Only observed receipts support claims; missing data = NOT_MEASURED.", "",
        f"Planned cells: **{summary['total_planned_cells']}**",
        "Status counts: " + json.dumps(summary["status_counts"], sort_keys=True), "",
        "| Dataset | Model | Mode | PASS / Planned |", "|---|---|---|---:|",
    ]
    for x in summary["groups"]:
        lines.append(f"| {x['dataset']} | {x['model']} | {x['mode']} | "
                     f"{x['passed']} / {x['total_cells']} |")
    lines += [
        "", "## Serving measurements", "",
        "| Dataset | Model | Mode | Text bin | CCU | Rep | Status | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | Output tok/s | GPU used MiB |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        def metric(key):
            v = row[key]
            return "NOT_MEASURED" if v is None else f"{v:.2f}"

        def e2e_label():
            if row.get("e2e_p95_status") == "NOT_COMPARABLE_LEGACY_DEFINITION":
                return "NOT_COMPARABLE (pre-queue E2E)"
            return metric("e2e_p95_ms")
        lines.append(
            f"| {row['dataset']} | {row['model']} | {row['mode']} | "
            f"{row['input_length']} | {row['concurrency']} | "
            f"{row['repetition']} | {row['status']} | "
            f"{metric('ttft_p95_ms')} | {metric('tpot_p95_ms')} | "
            f"{e2e_label()} | {metric('output_tokens_per_s')} | "
            f"{metric('peak_total_gpu_used_mib')} |"
        )
    lines += [
        "", "## Matched Base versus MTP (throughput; E2E is not used)", "",
        "| Dataset | Model | Bin | CCU | Repetitions | Mixed-E2E-definition pairs | Median speedup | 95% bootstrap CI (descriptive) |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for x in summary["paired_comparisons"]:
        speed = x["throughput_speedup_median"]
        ci = x["ci95_bootstrap_descriptive"]
        label = f"{speed:.3f}x" if speed is not None else "NOT_MEASURED"
        ci_label = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "NOT_ESTIMATED"
        lines.append(f"| {x['dataset']} | {x['model']} | {x['input_length']} | "
                     f"{x['concurrency']} | {x['paired_repetitions']} | "
                     f"{x['mixed_e2e_definition_pairs']} | "
                     f"{label} | {ci_label} |")
    lines += [
        "", "## Native quality (external official scorer pending)", "",
        "| Job | Status | Predictions | Failures | Official accuracy |",
        "|---|---|---:|---:|---|",
    ]
    for x in summary["quality_receipts"]:
        lines.append(f"| {x['filename']} | {x['status']} | {x['count']} | "
                     f"{x['failures']} | {x['official_accuracy']} |")
    lines += [
        "", "## Caveats", "",
        "- CCU measures offered requests; vLLM may queue excess requests.",
        "- Input bin targets model-tokenized text, plus visual tokens separately.",
        "- Streaming content chunks are not token-level ITL in MTP.",
        "- Total GPU memory used is not physical KV resident bytes.",
        "- MTP acceptance metrics require backend instrumentation.",
        "- Truncated serving tasks never produce official accuracy.",
        "- Official MemLens / MMLongBench scores require independent upstream scoring.",
        "- Bootstrap CI with three repetitions is only descriptive.", "",
    ]
    if summary.get("e2e_comparability") == "MIXED_NOT_COMPARABLE":
        lines[-1:-1] = [
            "- E2E: this campaign mixes receipts with and without the client queue wait "
            "field. Legacy E2E values are withheld (NOT_COMPARABLE). Paired throughput "
            "speedups do not use E2E, so pairs with different E2E definitions are included; "
            "they are counted in the Mixed-E2E-definition column.",
        ]
    return "\n".join(lines)


def write_report(root: Path, config: dict) -> dict:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    summary = aggregate(root, config)
    atomic_json(root / "summary.json", summary)
    tmp = root / "RUNBOOK.md.pending"
    tmp.write_text(render_markdown(summary), encoding="utf8")
    tmp.replace(root / "RUNBOOK.md")
    return summary
