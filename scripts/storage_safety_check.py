#!/usr/bin/env python3
"""Storage safety check for staging (ITER3 D3). SKELETON: behaviour not implemented yet."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

DEFAULT_GUARD_GIB = 20


def evaluate(target: Path | str, *, expected: dict[str, int], guard_bytes: int,
             free_bytes: int | None = None, free_bytes_fn: Callable[[Path], int] | None = None) -> dict[str, Any]:
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
