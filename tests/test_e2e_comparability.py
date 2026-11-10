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


def test_throughput_pairs_are_not_dropped_for_an_e2e_definition_mismatch():
    # R4-B (review 1k): the speedup is output tokens / wall time and does not use E2E.
    # A definition mismatch therefore does not drop the pair. It is counted and reported.
    records = [
        {"cell": _cell("base"), **_receipt(_cell("base"), _metrics(1, current=False, tps=100.0), "S")},
        {"cell": _cell("mtp"), **_receipt(_cell("mtp"), _metrics(1, current=True, tps=150.0), "S")},
    ]
    (cohort,) = paired_speedups(records)
    assert cohort["paired_repetitions"] == 1
    assert cohort["excluded_repetitions"] == 0
    assert cohort["mixed_e2e_definition_pairs"] == 1
    assert round(cohort["throughput_speedup_median"], 6) == 1.5
    assert cohort["per_rep"][0]["e2e_definitions"] == [E2E_LEGACY, E2E_CURRENT]


def test_paired_speedups_pool_repetitions_with_the_same_definition():
    records = []
    for rep in range(3):
        for mode, tps in (("base", 100.0), ("mtp", 150.0)):
            c = _cell(mode, rep)
            records.append({"cell": c, **_receipt(c, _metrics(1, current=True, tps=tps), "S")})
    (cohort,) = paired_speedups(records)
    assert cohort["paired_repetitions"] == 3
    assert cohort["mixed_e2e_definition_pairs"] == 0
    assert round(cohort["throughput_speedup_median"], 6) == 1.5


def test_report_shows_mixed_definition_pairs_and_keeps_throughput_speedup(tmp_path):
    # R4-B: the matched table counts the mixed-definition pair and still reports its speedup.
    config = load_config(CONFIG)
    cells = list(planned_cells(config))
    base = next(c for c in cells if c["mode"] == "base")
    mtp = next(c for c in cells if c["mode"] == "mtp" and
               {k: v for k, v in c.items() if k != "mode"} ==
               {k: v for k, v in base.items() if k != "mode"})
    root = tmp_path / "campaign"
    atomic_json(root / "cells" / f"{cell_id(base)}.json",
                {"cell": base, "status": "PASS_MEASURED", "sample_id_sha256": "S",
                 "metrics": _metrics(111.0, current=False, tps=100.0)})
    atomic_json(root / "cells" / f"{cell_id(mtp)}.json",
                {"cell": mtp, "status": "PASS_MEASURED", "sample_id_sha256": "S",
                 "metrics": _metrics(222.0, current=True, tps=150.0)})
    summary = report.aggregate(root, config)
    (pair,) = [x for x in summary["paired_comparisons"] if x["paired_repetitions"] == 1]
    assert pair["mixed_e2e_definition_pairs"] == 1
    assert round(pair["throughput_speedup_median"], 6) == 1.5
    text = report.render_markdown(summary)
    assert "Mixed-E2E-definition pairs" in text
    assert "pairs with different E2E definitions are included" in text
