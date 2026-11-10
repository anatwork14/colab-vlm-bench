"""ITER3 round B: arm sets and grid sizes agree across the engine, the configs, the pilots and the inventory.

ITER3 round A added the H2O confounder control ``h2o_shared_base`` (and the four STARLIGHT arms) to the engine registry.
They are NOT part of the seven-method confirmatory matrix (3,528 cells), the 168-cell seven-method pilot, or the
48-cell priority pilot. These tests read the configs and the engine directly, so the docs and the recorded inventory
cannot drift from that split without a failing test.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/harness_inventory.py"
MATRIX = ROOT / "configs/a100_qwen_7methods_full.json"
PILOT7 = ROOT / "configs/a100_qwen_7methods_pilot.json"
PILOT48 = ROOT / "configs/a100_qwen_priority_pilot.json"
INVENTORY = ROOT / "docs/harness/PILOT_INVENTORY.json"

SEVEN = frozenset(("base", "h2o_base", "da_base", "ecrd_base", "mtp", "h2o_mtp", "da_mtp"))
STARLIGHT = frozenset(("starlight_s1", "starlight_s1_no_da", "starlight_s1_no_grounding", "starlight_s1_mtp"))
CONTROL = "h2o_shared_base"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inv():
    spec = importlib.util.spec_from_file_location("harness_inventory", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _grid(cfg: dict) -> int:
    return (len(cfg["models"]) * len(cfg["methods"]) * len(cfg["datasets"])
            * len(cfg["input_lengths"]) * len(cfg["ccu"]) * cfg["repetitions"])


def test_matrix_manifest_methods_are_exactly_the_seven_original_methods():
    assert frozenset(_load(MATRIX)["methods"]) == SEVEN


def test_matrix_grid_is_3528_cells_and_the_enumeration_agrees():
    cfg = _load(MATRIX)
    assert _grid(cfg) == 3528
    assert len(_inv().enumerate_matrix_cells(MATRIX)) == 3528


def test_seven_method_pilot_is_168_cells_with_the_same_methods():
    cfg = _load(PILOT7)
    assert frozenset(cfg["methods"]) == SEVEN
    assert _grid(cfg) == 168


def test_priority_pilot_inventory_is_48_cells_with_base_and_mtp_only():
    cells = _inv().enumerate_pilot_cells(PILOT48)
    assert len(cells) == 48
    assert {c["arm"] for c in cells} == {"base", "mtp"}


def test_native_arms_are_a_subset_of_the_matrix_methods():
    from vlmbench.methods import NATIVE_AVAILABLE

    assert NATIVE_AVAILABLE <= SEVEN


def test_engine_registry_is_seven_plus_four_starlight_plus_one_control():
    from vlmbench.infrastructure.real_engine import RealSevenArmEngine

    assert RealSevenArmEngine.SUPPORTED_ARMS == SEVEN | STARLIGHT | {CONTROL}
    assert len(RealSevenArmEngine.SUPPORTED_ARMS) == 12


def test_control_and_starlight_arms_are_outside_the_matrix_pilots_and_recorded_inventory():
    inv = _inv()
    outside = STARLIGHT | {CONTROL}
    assert not outside & frozenset(_load(MATRIX)["methods"])
    assert not outside & frozenset(_load(PILOT7)["methods"])
    assert not outside & {c["arm"] for c in inv.enumerate_pilot_cells(PILOT48)}
    assert not outside & {c["arm"] for c in inv.enumerate_matrix_cells(MATRIX)}
    recorded = _load(INVENTORY)["cells"]
    assert not outside & {c["arm"] for c in recorded}
    assert {c["arm"] for c in recorded} <= SEVEN
    assert len(recorded) == 48 + 3528, "recorded inventory must hold the pilot (48) and the matrix (3528) only"
