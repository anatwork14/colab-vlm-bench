"""MMLongBench-Doc dataset adapter (native records).

The record construction lives in `native_records.build_mmlong_records` and is tested with an injected renderer.
This module supplies the real PyMuPDF renderer and the file input and output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..artifacts import atomic_jsonl, sha256
from .native_records import (TRANSFORM_VERSION, build_mmlong_records, check_record, receipt,
                             transform_params_sha256)


def _iter_qa(source: Path):
    if source.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet MMLongBench QA requires pyarrow: pip install pyarrow") from exc
        pf = pq.ParquetFile(source)
        for batch in pf.iter_batches(batch_size=8):
            yield from batch.to_pylist()
    else:
        data = json.loads(source.read_text(encoding="utf8"))
        if isinstance(data, dict):
            data = data.get("data", data.get("questions", []))
        if not isinstance(data, list):
            raise ValueError("Expected QA list or .parquet")
        yield from data


def _render_with_pymupdf(fitz_module):
    def render(pdf_path: Path, out_dir: Path, dpi: int) -> int:
        with fitz_module.open(pdf_path) as doc:
            count = len(doc)
            for index in range(count):
                dst = out_dir / f"p{index + 1:04d}.jpg"
                if not dst.is_file():
                    pix = doc[index].get_pixmap(matrix=fitz_module.Matrix(dpi / 72, dpi / 72), alpha=False)
                    pix.save(dst)
        return count
    return render


def normalize_mmlong(
    source: Path | str,
    pdf_root: Path | str,
    images_out: Path | str,
    output: Path | str,
    limit: int | None = None,
    dpi: int = 96,
) -> dict[str, Any]:
    """Render EVERY page of each referenced PDF. Never drop evidence for official quality.

    Missing documents are all reported together and nothing is published (`MissingReferences`).
    """
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("MMLongBench PDF rendering needs PyMuPDF: pip install -e '.[datasets]'") from e

    source = Path(source)
    output = Path(output)
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive or None")
    rows = list(_iter_qa(source))
    if limit is not None:
        rows = rows[:limit]
    records = build_mmlong_records(rows, pdf_root=pdf_root, images_out=images_out, dpi=dpi,
                                   render_pdf=_render_with_pymupdf(fitz))
    checks = [check_record(r) for r in records]
    atomic_jsonl(output, records)
    source_sha = sha256(source)
    output_sha = sha256(output)
    params = {"version": TRANSFORM_VERSION, "dataset": "mmlongbench_doc", "dpi": dpi, "limit": limit,
              "renderer": "pymupdf"}
    return {
        "dataset": "mmlongbench_doc",
        "count": len(records),
        "sha256_source": source_sha,
        "sha256_normalized": output_sha,
        "output": str(output),
        "status": "NORMALIZED_NEEDS_OFFICIAL_PROMPT_PARITY",
        "transform": {"version": TRANSFORM_VERSION, "params_sha256": transform_params_sha256(params),
                      "source_sha256": source_sha, "output_sha256": output_sha},
        "gold_free_receipt": receipt(checks),
    }
