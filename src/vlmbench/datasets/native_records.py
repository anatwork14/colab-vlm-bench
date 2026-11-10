"""Native inference records from pinned upstream dataset files (ITER3 D2). SKELETON: behaviour not implemented yet."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

TRANSFORM_VERSION = "native-records/1"


class MissingReferences(ValueError):
    """Every missing reference found in one pass. Raised before any output is written."""

    def __init__(self, missing: list[dict[str, Any]]):
        self.missing = missing
        super().__init__(f"Missing references ({len(missing)})")


def transform_params_sha256(params: dict[str, Any]) -> str:
    raise NotImplementedError


def check_record(record: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


def receipt(checks: list[dict[str, Any]]) -> dict[str, Any]:
    raise NotImplementedError


def build_mmlong_records(qa_rows: list[dict[str, Any]], *, pdf_root: Path, images_out: Path, dpi: int,
                         render_pdf: Callable[[Path, Path, int], int]) -> list[dict[str, Any]]:
    raise NotImplementedError
