"""MMLongBench-Doc dataset adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..artifacts import atomic_jsonl, sha256


def normalize_mmlong(
    source: Path | str,
    pdf_root: Path | str,
    images_out: Path | str,
    output: Path | str,
    limit: int | None = None,
    dpi: int = 96,
) -> dict[str, Any]:
    """Render EVERY page of each referenced PDF. Never drop evidence for official quality."""
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("MMLongBench PDF rendering needs PyMuPDF: pip install -e '.[datasets]'") from e

    source = Path(source)
    pdf_root = Path(pdf_root).resolve()
    images_out = Path(images_out).resolve()
    output = Path(output)
    if not 72 <= dpi <= 200:
        raise ValueError("DPI must be 72..200")

    def iter_qa():
        if source.suffix.lower() == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet MMLongBench QA requires pyarrow: pip install pyarrow"
                ) from exc
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

    count = 0
    seen = set()
    images_out.mkdir(parents=True, exist_ok=True)

    def records():
        nonlocal count
        for item in iter_qa():
            doc_name = item["doc_id"]
            pdf = (pdf_root / doc_name).resolve()
            if not pdf.is_relative_to(pdf_root) or not pdf.is_file():
                raise ValueError(f"PDF missing or escapes designated root: {doc_name}")
            q = str(item["question"])
            uid = hashlib.sha256((str(doc_name) + "\n" + q).encode()).hexdigest()[:20]
            if uid in seen:
                raise ValueError(f"Duplicate MMLongBench question/doc: {uid}")
            seen.add(uid)
            page_parts = []
            with fitz.open(pdf) as doc:
                page_count = len(doc)
                for index in range(page_count):
                    # DPI is part of the cache key: a reused 72-dpi page must never
                    # silently satisfy a 96-dpi run (different visual evidence).
                    dst = (
                        images_out
                        / f"{hashlib.sha256(doc_name.encode()).hexdigest()[:16]}_dpi{dpi}"
                        / f"p{index+1:04d}.jpg"
                    )
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.is_file():
                        pix = doc[index].get_pixmap(
                            matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False
                        )
                        pix.save(dst)
                    page_parts.extend([
                        {"type": "text", "text": f"Document page {index + 1}/{page_count}:"},
                        {"type": "image_path", "path": str(dst), "asset_root": str(images_out)},
                    ])
            messages = [
                {
                    "role": "user",
                    "content": page_parts + [
                        {
                            "type": "text",
                            "text": f"Question: {q}\nGive a concise answer; if not answerable say so.",
                        }
                    ],
                }
            ]
            record = {
                "id": uid,
                "dataset": "mmlongbench_doc",
                "track": "native_quality",
                "messages": messages,
                "reference": item.get("answer"),
                "native_context_complete": True,
                "meta": {
                    "doc_id": doc_name,
                    "question": q,
                    "doc_type": item.get("doc_type"),
                    "evidence_pages": item.get("evidence_pages"),
                    "evidence_sources": item.get("evidence_sources"),
                    "answer_format": item.get("answer_format"),
                    "page_count": page_count,
                    "render_dpi": dpi,
                    "adapter_provenance": (
                        "colab-vlm-bench-mmlong-v1; official_prompt_parity_pending"
                    ),
                },
            }
            count += 1
            yield record
            if limit is not None and count >= limit:
                break

    atomic_jsonl(output, records())
    return {
        "dataset": "mmlongbench_doc",
        "count": count,
        "sha256_source": sha256(source),
        "sha256_normalized": sha256(output),
        "output": str(output),
        "status": "NORMALIZED_NEEDS_OFFICIAL_PROMPT_PARITY",
    }
