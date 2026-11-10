"""Audit item 6: unpinned inputs are rendered as UNPINNED or BLOCKED, never as a value, and a
fabricated or placeholder SHA cannot pass as a pin. No network, no downloads."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from vlmbench import repro_manifest as rm

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "configs/a100_qwen_priority_pilot.json"
FULL = ROOT / "configs/a100_qwen_priority_full.json"
COMMITTED = ROOT / "docs/audit/PINNED_INPUTS.md"
REAL_LOOKING = "9c1d4e7a0b2f8e3d5c6a1b9f0e4d7c2a8b3e5f61"


def _rows(path):
    text, rows = rm.generate(path)
    return text, rows


@pytest.mark.parametrize("config", [PILOT, FULL])
def test_every_configured_model_revision_renders_as_unpinned(config):
    _, rows = _rows(config)
    models = [r for r in rows if r["input"].startswith("model revision")]
    assert models, "config must list models"
    for row in models:
        assert row["status"] == rm.UNPINNED
        assert row["value"] == "UNPINNED"


def test_qwen_4b_revision_is_unpinned_not_a_value():
    _, rows = _rows(PILOT)
    (row,) = [r for r in rows if "Qwen/Qwen3.5-4B" in r["input"]]
    assert row["status"] == rm.UNPINNED
    assert row["value"] == "UNPINNED"


def test_dataset_revisions_are_blocked_not_values():
    _, rows = _rows(PILOT)
    datasets = [r for r in rows if r["input"].startswith("dataset revision")]
    assert {r["input"] for r in datasets} == {
        "dataset revision: xiyuRenBill/MEMLENS",
        "dataset revision: yubo2333/MMLongBench-Doc",
    }
    for row in datasets:
        assert row["status"] == rm.BLOCKED
        assert row["value"] == "BLOCKED"


@pytest.mark.parametrize("fake", [
    "0" * 40,                 # all zeros
    "a" * 40,                 # one repeated character
    "ab" * 20,                # two characters
    "abcde" * 8,              # five-character cycle, still low entropy
])
def test_placeholder_looking_sha_is_refused(fake):
    with pytest.raises(rm.PinError, match="placeholder"):
        rm.classify_revision(fake)


@pytest.mark.parametrize("bad", ["main", "v1.0", "abc123", "9C1D4E7A0B2F8E3D5C6A1B9F0E4D7C2A8B3E5F61", None, 7])
def test_non_commit_revision_is_refused_unless_it_is_an_unpinned_token(bad):
    if bad is None:
        assert rm.classify_revision(bad) == (rm.UNPINNED, rm.UNPINNED)
        return
    with pytest.raises(rm.PinError):
        rm.classify_revision(bad)


def test_real_looking_commit_sha_is_the_only_way_to_render_a_pin():
    assert rm.classify_revision(REAL_LOOKING) == (rm.PINNED, REAL_LOOKING)


def test_fabricated_pin_in_a_config_stops_manifest_generation(tmp_path):
    cfg = __import__("json").loads(PILOT.read_text(encoding="utf8"))
    cfg["models"][0]["revision"] = "0" * 40
    path = tmp_path / "fabricated.json"
    path.write_text(__import__("json").dumps(cfg), encoding="utf8")
    with pytest.raises(rm.PinError):
        rm.generate(path)


def test_committed_pinned_inputs_match_generator_output():
    text, _ = _rows(PILOT)
    assert COMMITTED.is_file(), "run: python -m vlmbench.repro_manifest --config configs/a100_qwen_priority_pilot.json --out docs/audit/PINNED_INPUTS.md"
    assert COMMITTED.read_text(encoding="utf8") == text


def test_audit_docs_contain_no_placeholder_commit_sha():
    hits = []
    for doc in list((ROOT / "docs/audit").glob("*.md")) + list((ROOT / "runbooks").glob("*.md")):
        for sha in re.findall(r"\b[0-9a-f]{40}\b", doc.read_text(encoding="utf8")):
            if rm._looks_like_placeholder(sha):
                hits.append(f"{doc.name}: {sha}")
    assert hits == []
