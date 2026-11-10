"""ITER3 D3: staging storage check. Refuses when free space after the transfer would fall below the guard, reuses
files already present at the expected size, never deletes, copies or writes."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/storage_safety_check.py"
GiB = 2 ** 30


def _load():
    spec = importlib.util.spec_from_file_location("storage_safety_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load()


def test_refuses_when_free_space_after_the_transfer_is_below_the_guard(tmp_path):
    res = ss.evaluate(tmp_path, expected={"w.safetensors": 9 * GiB}, guard_bytes=20 * GiB, free_bytes=13 * GiB)
    assert res["decision"] == "REFUSE" and res["exit_code"] == 2
    assert res["need_bytes"] == 9 * GiB
    assert res["free_after_bytes"] == 4 * GiB


def test_passes_when_free_space_after_the_transfer_meets_the_guard(tmp_path):
    res = ss.evaluate(tmp_path, expected={"w.safetensors": 9 * GiB}, guard_bytes=20 * GiB, free_bytes=40 * GiB)
    assert res["decision"] == "PASS" and res["exit_code"] == 0


def test_a_file_already_present_at_the_expected_size_is_reused_not_transferred(tmp_path):
    (tmp_path / "a.json").write_bytes(b"x" * 10)
    res = ss.evaluate(tmp_path, expected={"a.json": 10, "b.bin": 100}, guard_bytes=0, free_bytes=1000)
    assert res["reuse"] == ["a.json"]
    assert res["need"] == ["b.bin"] and res["need_bytes"] == 100


def test_a_size_mismatch_is_transferred_again_and_nothing_is_deleted(tmp_path):
    part = tmp_path / "a.json"
    part.write_bytes(b"x" * 7)
    stray = tmp_path / "stray.txt"
    stray.write_text("keep me", encoding="utf8")
    res = ss.evaluate(tmp_path, expected={"a.json": 10}, guard_bytes=0, free_bytes=1000)
    assert res["mismatch"] == ["a.json"] and res["need"] == ["a.json"]
    assert part.read_bytes() == b"x" * 7 and stray.read_text(encoding="utf8") == "keep me"


def test_a_target_that_does_not_exist_uses_its_nearest_existing_ancestor_and_creates_nothing(tmp_path):
    seen: list[Path] = []

    def fake_free(path: Path) -> int:
        seen.append(Path(path))
        return 10 * GiB

    target = tmp_path / "new" / "deeper"
    res = ss.evaluate(target, expected={}, guard_bytes=0, free_bytes_fn=fake_free)
    assert seen == [tmp_path]
    assert res["free_bytes"] == 10 * GiB
    assert not (tmp_path / "new").exists()


def test_a_path_that_is_a_file_is_an_error(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x", encoding="utf8")
    res = ss.evaluate(f, expected={"a": 1}, guard_bytes=0, free_bytes=10)
    assert res["decision"] == "ERROR" and res["exit_code"] == 3


def test_the_default_guard_matches_the_harness_guard():
    assert ss.DEFAULT_GUARD_GIB == 20


def test_the_script_has_no_delete_copy_move_or_write_calls():
    src = SCRIPT.read_text(encoding="utf8")
    for token in ("unlink(", "rmtree", "os.remove", "os.rename", "shutil.", "write_bytes", "write_text",
                  "open(", "mkdir("):
        assert token not in src, f"the storage check must not use {token}"


def test_cli_prints_json_and_exit_codes_match_the_decision(tmp_path):
    exp = tmp_path / "expected.json"
    exp.write_text(json.dumps({"w.safetensors": 9 * GiB}), encoding="utf8")
    refuse = subprocess.run([sys.executable, str(SCRIPT), "--target", str(tmp_path), "--expected", str(exp),
                             "--guard-gib", "20", "--free-bytes", str(13 * GiB)],
                            capture_output=True, text=True, timeout=60)
    assert refuse.returncode == 2
    assert json.loads(refuse.stdout)["decision"] == "REFUSE"
    passed = subprocess.run([sys.executable, str(SCRIPT), "--target", str(tmp_path), "--expected", str(exp),
                             "--guard-gib", "20", "--free-bytes", str(40 * GiB)],
                            capture_output=True, text=True, timeout=60)
    assert passed.returncode == 0
    assert json.loads(passed.stdout)["decision"] == "PASS"
