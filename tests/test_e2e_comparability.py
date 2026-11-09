"""R3-F (review F-02): E2E now includes the client CCU-slot wait. Receipts written before that
change have no client_queue_wait fields and measured a different quantity. They must not be
pooled with current receipts, and the report must say so."""
from __future__ import annotations

import json
from pathlib import Path

from vlmbench import report
from vlmbench.analysis import E2E_CURRENT, E2E_LEGACY, e2e_definition, paired_speedups
from vlmbench.artifacts import atomic_json
from vlmbench.config import cell_id, load_config, planned_cells

CONFIG = Path(__file__).resolve().parents[1] / "configs/a100_qwen_priority_pilot.json"


def _cell(mode="base", rep=0, ccu=1):
    return {"dataset": "memlens", "model": "qwen35_08b", "mode": mode,
            "input_length": 2048, "concurrency": ccu, "repetition": rep}


def _metrics(e2e, current, tps=100.0):
    m = {"e2e_p95_ms": e2e, "ttft_p95_ms": 10.0, "tpot_p95_ms": 5.0,
         "output_tokens_per_s": tps, "peak_total_gpu_used_mib": 1000}
    if current:
        m["client_queue_wait_p50_ms"] = 0.0
        m["client_queue_wait_p95_ms"] = 0.0
    return m


def _receipt(cell, metrics, sample="S"):
    return {"cell": cell, "status": "PASS_MEASURED", "sample_id_sha256": sample,
            "metrics": metrics}


def test_definition_is_read_from_receipt_fields():
    assert e2e_definition(_receipt(_cell(), _metrics(1.0, current=True))) == E2E_CURRENT
    assert e2e_definition(_receipt(_cell(), _metrics(1.0, current=False))) == E2E_LEGACY


def test_report_marks_legacy_e2e_not_comparable_and_keeps_current_value(tmp_path):
    config = load_config(CONFIG)
    cells = list(planned_cells(config))
    legacy, current = cells[0], cells[1]
    root = tmp_path / "campaign"
    atomic_json(root / "cells" / f"{cell_id(legacy)}.json",
                _receipt(legacy, _metrics(111.0, current=False)))
    atomic_json(root / "cells" / f"{cell_id(current)}.json",
                _receipt(current, _metrics(222.0, current=True)))

    summary = report.aggregate(root, config)
    rows = {cell_id(r): r for r in summary["rows"] if r["status"] == "PASS_MEASURED"}
    assert rows[cell_id(legacy)]["e2e_p95_ms"] is None
    assert rows[cell_id(legacy)]["e2e_p95_status"] == "NOT_COMPARABLE_LEGACY_DEFINITION"
    assert rows[cell_id(current)]["e2e_p95_ms"] == 222.0
    assert summary["e2e_comparability"] == "MIXED_NOT_COMPARABLE"

    text = report.render_markdown(summary)
    assert "NOT_COMPARABLE" in text
    assert "111" not in text


def test_paired_speedups_do_not_pool_base_and_mtp_with_different_e2e_definitions():
    records = [
        {"cell": _cell("base"), **_receipt(_cell("base"), _metrics(1, current=False), "S")},
        {"cell": _cell("mtp"), **_receipt(_cell("mtp"), _metrics(1, current=True), "S")},
    ]
    (cohort,) = paired_speedups(records)
    assert cohort["paired_repetitions"] == 0
    assert cohort["excluded_mixed_e2e_definition"] == 1
    assert cohort["throughput_speedup_median"] is None


def test_paired_speedups_pool_repetitions_with_the_same_definition():
    records = []
    for rep in range(3):
        for mode, tps in (("base", 100.0), ("mtp", 150.0)):
            c = _cell(mode, rep)
            records.append({"cell": c, **_receipt(c, _metrics(1, current=True, tps=tps), "S")})
    (cohort,) = paired_speedups(records)
    assert cohort["paired_repetitions"] == 3
    assert cohort["excluded_mixed_e2e_definition"] == 0
    assert round(cohort["throughput_speedup_median"], 6) == 1.5
