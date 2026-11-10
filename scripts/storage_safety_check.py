#!/usr/bin/env python3
"""Storage safety check for staging (ITER3 D3).

Reads the free space of the nearest existing ancestor of a target directory, compares the expected files (name to
size, from the Drive metadata or the signed plan) with what is already there, and decides:

- files present at the expected size are REUSED (not transferred);
- files missing, or present at a different size, are NEED (transferred in full);
- PASS if free space minus the NEED bytes is at least the guard; otherwise REFUSE.

It never deletes, copies, moves, creates or writes anything. A size mismatch is reported, and the existing file is
left as it is. Exit codes: 0 PASS, 2 REFUSE, 3 ERROR.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

DEFAULT_GUARD_GIB = 20  # matches scripts/harness_inventory.py --min-free-gib
GiB = 2 ** 30
EXIT = {"PASS": 0, "REFUSE": 2, "ERROR": 3}


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _statvfs_free(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def _result(decision: str, **fields: Any) -> dict[str, Any]:
    return {"decision": decision, "exit_code": EXIT[decision], **fields}


def evaluate(target: Path | str, *, expected: dict[str, int], guard_bytes: int,
             free_bytes: int | None = None, free_bytes_fn: Callable[[Path], int] | None = None) -> dict[str, Any]:
    target = Path(target)
    base = {"target": str(target), "guard_bytes": guard_bytes}
    if target.exists() and not target.is_dir():
        return _result("ERROR", reason="target exists and is not a directory", **base)
    anchor = _nearest_existing(target)
    if not anchor.is_dir():
        return _result("ERROR", reason="no existing directory above the target", **base)
    for name in expected:
        if not name or "/" in name or "\\" in name or name in (".", ".."):
            return _result("ERROR", reason=f"unsafe expected file name {name!r}", **base)
    if free_bytes is not None:
        free, source = free_bytes, "override"
    else:
        free, source = (free_bytes_fn or _statvfs_free)(anchor), "statvfs"

    reuse: list[str] = []
    mismatch: list[str] = []
    missing: list[str] = []
    need_bytes = 0
    for name, size in sorted(expected.items()):
        path = target / name
        if path.is_file():
            if path.stat().st_size == size:
                reuse.append(name)
                continue
            mismatch.append(name)
        else:
            missing.append(name)
        need_bytes += size
    need = sorted(mismatch + missing)
    free_after = free - need_bytes
    decision = "PASS" if free_after >= guard_bytes else "REFUSE"
    return _result(decision, free_bytes=free, free_source=source, free_after_bytes=free_after,
                   need_bytes=need_bytes, need=need, reuse=reuse, mismatch=mismatch, missing=missing,
                   **base)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=Path,
                        help="JSON object mapping file name to expected size in bytes")
    parser.add_argument("--guard-gib", type=float, default=DEFAULT_GUARD_GIB)
    parser.add_argument("--free-bytes", type=int, default=None,
                        help="override the free space (for tests and dry runs); reported as free_source=override")
    args = parser.parse_args(argv)
    try:
        expected = json.loads(args.expected.read_text(encoding="utf8"))
        if not isinstance(expected, dict) or not all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in expected.values()):
            raise ValueError("expected must map file names to non-negative integer sizes")
    except (OSError, ValueError) as exc:
        result = _result("ERROR", reason=f"cannot read expected sizes: {exc}", target=str(args.target))
    else:
        result = evaluate(args.target, expected=expected, guard_bytes=int(args.guard_gib * GiB),
                          free_bytes=args.free_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
