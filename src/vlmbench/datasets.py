"""Normalization from upstream artifacts to native full-context VLM questions.

These are auditable adapters, not claims of byte-identical official prompt construction.
Official scorer parity MUST be audited against upstream before publication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from .artifacts import atomic_jsonl, sha256



def normalize_memlens(source: Path, images_root: Path, output: Path,
                      limit: int | None = None) -> dict:
    """Stream MemLens JSON question-by-question, without copying the whole file.

    Peak RAM is bounded by one 32K item and its normalized message list, rather
    than the entire dataset, a second role-normalized copy, and all outputs.
    """
    try:
        import ijson
    except ImportError as exc:
        raise RuntimeError(
            "MemLens streaming normalization requires 'ijson'. "
            "Install with pip install 'ijson>=3.3,<4'."
        ) from exc

    source, images_root, output = Path(source), Path(images_root).resolve(), Path(output)
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive or None")
    seen: set[str] = set()
    count = 0
    image_count_total = 0

    def records():
        nonlocal count, image_count_total
        with source.open("rb") as fp:
            for item in ijson.items(fp, "item"):
                if not isinstance(item, dict):
                    raise ValueError("MemLens expects a top-level JSON list of objects")
                qid = str(item["question_id"])
                if qid in seen:
                    raise ValueError(f"Duplicate MemLens question id: {qid}")
                seen.add(qid)
                session_ids = item["haystack_session_ids"]
                sessions = item["haystack_sessions"]
                dates = item["haystack_dates"]
                if len(session_ids) != len(sessions) or len(sessions) != len(dates):
                    raise ValueError(f"Inconsistent MemLens session metadata: {qid}")
                parts = []
                image_count = 0
                for sid, date, session in zip(session_ids, dates, sessions):
                    parts.append({"type": "text", "text": f"\n[Session {sid} — {date}]\n"})
                    for turn in session:
                        raw_role = str(turn["role"]).strip().lower()
                        if "assistant" in raw_role:
                            role = "assistant"
                        elif "user" in raw_role:
                            role = "user"
                        else:
                            raise ValueError(f"Unknown MemLens role {turn['role']} in {qid}")
                        parts.append({
                            "type": "text",
                            "text": f"[{role}] {turn.get('content', '')}\n",
                        })
                        for image_ref in turn.get("images", []):
                            relative = Path(image_ref["file"])
                            path = (images_root / "release_images" / relative).resolve()
                            if not path.is_relative_to(images_root) or not path.is_file():
                                raise ValueError(f"Missing or escaping MemLens image: {relative}")
                            parts.append({
                                "type": "image_path", "path": str(path),
                                "asset_root": str(images_root),
                            })
                            image_count += 1
                parts.append({"type": "text", "text": f"\nQuestion: {item['question']}"})
                record = {
                    "id": qid, "dataset": "memlens", "track": "native_quality",
                    "messages": [{"role": "user", "content": parts}],
                    "reference": str(item["answer"]), "native_context_complete": True,
                    "meta": {
                        "question_type": item["question_type"],
                        "question_date": item.get("question_date"),
                        "session_ids": session_ids,
                        "evidence_session_ids": item.get("answer_session_ids", []),
                        "image_count": image_count,
                        "adapter_provenance":
                            "colab-vlm-bench-memlens-v2-streaming; official_prompt_parity_pending",
                    },
                }
                count += 1
                image_count_total += image_count
                yield record
                if limit is not None and count >= limit:
                    break

    # atomic_jsonl iterates lazily and writes a .pending file before a final
    # replace; no giant in-memory records array is created.
    atomic_jsonl(output, records())
    return {
        "dataset": "memlens", "count": count, "image_references": image_count_total,
        "sha256_source": sha256(source), "sha256_normalized": sha256(output),
        "output": str(output),
        "status": "NORMALIZED_NEEDS_OFFICIAL_PROMPT_PARITY",
        "memory_strategy": "ijson_stream_one_question_and_atomic_jsonl",
    }

def normalize_mmlong(source: Path, pdf_root: Path, images_out: Path, output: Path,
                     limit: int | None = None, dpi: int = 96) -> dict:
    """Render EVERY page of each referenced PDF. Never drop evidence for official quality."""
    try:
        import fitz
    except ImportError as e:
        raise RuntimeError("MMLongBench PDF rendering needs PyMuPDF: pip install -e '.[datasets]'") from e
    source, pdf_root = Path(source), Path(pdf_root).resolve()
    images_out, output = Path(images_out).resolve(), Path(output)
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
            # Tiny batches keep memory stable even if a rowgroup is large.
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
                    dst = images_out / hashlib.sha256(doc_name.encode()).hexdigest()[:16] / f"p{index+1:04d}.jpg"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.is_file():
                        pix = doc[index].get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
                        pix.save(dst)
                    page_parts.extend([
                        {"type": "text", "text": f"Document page {index + 1}/{page_count}:"},
                        {"type": "image_path", "path": str(dst), "asset_root": str(images_out)},
                    ])
            messages = [
                {"role": "user", "content": page_parts + [
                    {"type": "text", "text": f"Question: {q}\nGive a concise answer; if not answerable say so."}
                ]}
            ]
            record = {
                "id": uid, "dataset": "mmlongbench_doc", "track": "native_quality",
                "messages": messages, "reference": item.get("answer"),
                "native_context_complete": True,
                "meta": {
                    "doc_id": doc_name, "question": q, "doc_type": item.get("doc_type"),
                    "evidence_pages": item.get("evidence_pages"),
                    "evidence_sources": item.get("evidence_sources"),
                    "answer_format": item.get("answer_format"),
                    "page_count": page_count, "render_dpi": dpi,
                    "adapter_provenance": "colab-vlm-bench-mmlong-v1; official_prompt_parity_pending",
                },
            }
            count += 1
            yield record
            if limit is not None and count >= limit:
                break
    atomic_jsonl(output, records())
    return {"dataset": "mmlongbench_doc", "count": count,
            "sha256_source": sha256(source), "sha256_normalized": sha256(output),
            "output": str(output), "status": "NORMALIZED_NEEDS_OFFICIAL_PROMPT_PARITY"}
