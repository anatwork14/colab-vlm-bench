"""F-09: derived serving rows from different derivation routes must never be pooled.

Covers the label constants, the streaming label reader, and the resume identity that
makes a receipt from another route STALE. The run_job refusal paths are in
tests/test_resume_policy.py, which owns the GPU fakes."""
from __future__ import annotations

import json
from pathlib import Path

from vlmbench import derivation_labels as dl
from vlmbench.resume import IDENTITY_KEYS, STALE, classify_cell_receipt
from vlmbench.artifacts import atomic_json
from vlmbench.config import cell_id, planned_cells, load_config

SRC = Path(__file__).resolve().parents[1] / "src/vlmbench"
CONFIG = Path(__file__).resolve().parents[1] / "configs/a100_qwen_priority_pilot.json"


def test_route_labels_are_distinct_and_known():
    labels = [dl.MEMLENS_7BIN_BUILDER, dl.SERVING_TAIL_TEXT_ONE_IMAGE, dl.LAST_WORDS_LAST_K_IMAGES]
    assert len(set(labels)) == len(labels) == 3
    assert set(labels) == dl.KNOWN_ROUTE_LABELS


def test_each_route_writes_its_own_constant_not_a_literal():
    # The two MemLens routes (memlens.py builder and notebook_bridge serving candidates)
    # plus the generic derivation.py route must reference their own constant.
    memlens = (SRC / "datasets/memlens.py").read_text(encoding="utf8")
    bridge = (SRC / "notebook_bridge.py").read_text(encoding="utf8")
    generic = (SRC / "derivation.py").read_text(encoding="utf8")
    assert "derivation_labels.MEMLENS_7BIN_BUILDER" in memlens
    assert "derivation_labels.SERVING_TAIL_TEXT_ONE_IMAGE" in bridge
    assert "derivation_labels.LAST_WORDS_LAST_K_IMAGES" in generic
    for text, literal in ((memlens, dl.SERVING_TAIL_TEXT_ONE_IMAGE),
                          (bridge, dl.MEMLENS_7BIN_BUILDER),
                          (generic, dl.MEMLENS_7BIN_BUILDER)):
        assert literal not in text, "a route must not write another route's label"


def test_read_derivation_labels_counts_every_row(tmp_path):
    path = tmp_path / "derived.jsonl"
    rows = [{"id": 1, "derivation": dl.MEMLENS_7BIN_BUILDER},
            {"id": 2, "derivation": dl.SERVING_TAIL_TEXT_ONE_IMAGE},
            {"id": 3}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n", encoding="utf8")
    counts = dl.read_derivation_labels(path)
    assert counts == {dl.MEMLENS_7BIN_BUILDER: 1, dl.SERVING_TAIL_TEXT_ONE_IMAGE: 1, None: 1}


def test_identity_includes_derivation_so_other_route_receipts_are_stale(tmp_path):
    assert "derivation" in IDENTITY_KEYS
    cfg = load_config(CONFIG)
    cell = next(iter(planned_cells(cfg)))
    base = {"config_sha256": "C", "dataset_sha256": "D", "code_sha256": "X" * 40,
            "execution_pass": "FIXED_OUTPUT_SERVING_256",
            "output_contract": {"max_tokens": 256, "min_tokens": 256, "ignore_eos": True}}
    path = tmp_path / "cell.json"
    receipt = {**base, "cell": cell, "status": "PASS_MEASURED",
               "derivation": dl.SERVING_TAIL_TEXT_ONE_IMAGE,
               "sample_id_sha256": "S", "metrics": {"x": 1},
               "successful_requests": cell["requests"], "expected_requests": cell["requests"]}
    atomic_json(path, receipt)

    other = {**base, "derivation": dl.MEMLENS_7BIN_BUILDER}
    state, reason = classify_cell_receipt(path, other, cell, "S")
    assert state == STALE and "derivation" in reason

    same = {**base, "derivation": dl.SERVING_TAIL_TEXT_ONE_IMAGE}
    state, _ = classify_cell_receipt(path, same, cell, "S")
    assert state == "COMPLETED"


def test_receipt_without_a_derivation_field_is_stale(tmp_path):
    cfg = load_config(CONFIG)
    cell = next(iter(planned_cells(cfg)))
    base = {"config_sha256": "C", "dataset_sha256": "D", "code_sha256": "X" * 40,
            "execution_pass": "FIXED_OUTPUT_SERVING_256",
            "output_contract": {"max_tokens": 256, "min_tokens": 256, "ignore_eos": True}}
    path = tmp_path / "cell.json"
    atomic_json(path, {**base, "cell": cell, "status": "ERROR"})
    state, _ = classify_cell_receipt(path, {**base, "derivation": dl.MEMLENS_7BIN_BUILDER},
                                     cell, None)
    assert state == STALE
