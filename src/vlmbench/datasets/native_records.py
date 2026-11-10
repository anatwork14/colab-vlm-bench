"""Native inference records from pinned upstream dataset files (ITER3 D2).

Shared rules for the MemLens and MMLongBench-Doc adapters:

- Sample ids are the upstream identifiers (MemLens `question_id`; MMLongBench-Doc a hash of the upstream
  `doc_id` and `question`, which does not depend on row order).
- Output keeps source order. The same input gives the same output bytes.
- Every missing image or document is found in one pass and reported together (`MissingReferences`). Nothing is
  published when any reference is missing.
- A gold-free check runs on every record. Structural leaks (gold keys or gold text inside the request) fail. A gold
  token in the context is reported in a receipt, because haystack text can contain the answer legitimately.
- Native records are never derived. Derived workloads live in `derivation.py` and `memlens.build_memlens_context_bins`.

Limits (stated, not hidden):
- The prompt text is the adapter's own. It is not verified against the upstream official prompt. Official prompt
  parity stays pending (`docs/harness/DATASET_READINESS.md` §2.5, §3.6).
- The MMLongBench-Doc page count is what the renderer reports. It is not checked against the PDF independently.
- Rendered page image bytes are not hashed for the transform: they depend on the renderer library version. The PDF
  SHA-256 and the DPI are recorded instead.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

TRANSFORM_VERSION = "native-records/1"

# Keys that must never appear anywhere inside the request messages.
STRUCTURAL_KEYS = frozenset({
    "reference", "answer", "answer_format", "answer_session_ids", "answer_session_id",
    "evidence_pages", "evidence_sources", "evidence_session_ids", "meta",
})

# Constant wording that the adapters send with every row. It is not per-item, so it is not counted as context.
TEMPLATE_PATTERNS = (
    re.compile(r"Give a concise answer; if not answerable say so\."),
    re.compile(r"Document page \d+/\d+:"),
    re.compile(r"\[Session [^\]]*\]"),
)

PROMPT_MMLONG = "Question: {q}\nGive a concise answer; if not answerable say so."


class MissingReferences(ValueError):
    """Every missing reference found in one pass. Raised before any output is written."""

    def __init__(self, missing: list[dict[str, Any]]):
        self.missing = missing
        head = "; ".join(f"{m['kind']} {m['id']} {m['path']}" for m in missing[:20])
        super().__init__(f"Missing references ({len(missing)}): {head}")


def transform_params_sha256(params: dict[str, Any]) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf8")).hexdigest()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains_sequence(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


def _scan_keys(node: Any, path: str, found: list[str]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in STRUCTURAL_KEYS:
                found.append(f"{path}.{key} (forbidden key)")
            _scan_keys(value, f"{path}.{key}", found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _scan_keys(value, f"{path}[{index}]", found)


def check_record(record: dict[str, Any]) -> dict[str, Any]:
    """Gold-free check for one record. `structural` non-empty means the request leaks gold or gold keys."""
    structural: list[str] = []
    messages = record.get("messages") or []
    _scan_keys(messages, "messages", structural)
    gold = record.get("reference")
    gold_text = "" if gold is None else str(gold).strip()
    gold_tokens = _tokens(gold_text) if gold_text else []
    gold_in_context = False
    for mi, message in enumerate(messages):
        content = message.get("content")
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for pi, part in enumerate(parts):
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            text = str(part.get("text", ""))
            if gold_text and text.strip() == gold_text:
                structural.append(f"messages[{mi}].content[{pi}]: text equals the reference")
            context = text.split("Question:", 1)[0]
            for pattern in TEMPLATE_PATTERNS:
                context = pattern.sub(" ", context)
            if gold_tokens and _contains_sequence(_tokens(context), gold_tokens):
                gold_in_context = True
    return {"id": record.get("id"), "structural": structural, "gold_in_context": gold_in_context}


def receipt(checks: list[dict[str, Any]]) -> dict[str, Any]:
    structural = [str(c["id"]) for c in checks if c["structural"]]
    context = [str(c["id"]) for c in checks if c["gold_in_context"]]
    if structural:
        status = "LEAKED"
    elif context:
        status = "GOLD_TOKEN_IN_CONTEXT"
    else:
        status = "GOLD_FREE"
    return {"transform": TRANSFORM_VERSION, "records": len(checks), "status": status,
            "structural_violations": structural, "gold_token_in_context": context}


def mmlong_sample_id(doc_id: str, question: str) -> str:
    return hashlib.sha256((str(doc_id) + "\n" + str(question)).encode()).hexdigest()[:20]


def build_mmlong_records(qa_rows: list[dict[str, Any]], *, pdf_root: Path | str, images_out: Path | str,
                         dpi: int, render_pdf: Callable[[Path, Path, int], int]) -> list[dict[str, Any]]:
    """Native MMLongBench-Doc records. `render_pdf(pdf, out_dir, dpi)` writes `pNNNN.jpg` pages and returns the
    page count it reports. Every page of every referenced document is included."""
    if not 72 <= dpi <= 200:
        raise ValueError("DPI must be 72..200")
    pdf_root = Path(pdf_root).resolve()
    images_out = Path(images_out).resolve()

    # Pass 1: every reference and every sample id, before anything is rendered.
    missing: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in qa_rows:
        doc = str(row["doc_id"])
        question = str(row["question"])
        uid = mmlong_sample_id(doc, question)
        if uid in seen:
            raise ValueError(f"Duplicate MMLongBench question/doc: {uid}")
        seen.add(uid)
        pdf = (pdf_root / doc).resolve()
        if not pdf.is_relative_to(pdf_root):
            raise ValueError(f"MMLongBench document path escapes the designated root: {doc}")
        if not pdf.is_file():
            missing.append({"kind": "mmlong_pdf", "id": uid, "path": doc})
    if missing:
        raise MissingReferences(missing)

    # Pass 2: render each document once, then build the records in input order.
    rendered: dict[str, tuple[int, list[Path], str]] = {}
    records: list[dict[str, Any]] = []
    for row in qa_rows:
        doc = str(row["doc_id"])
        question = str(row["question"])
        pdf = (pdf_root / doc).resolve()
        if doc not in rendered:
            out_dir = images_out / f"{hashlib.sha256(doc.encode()).hexdigest()[:16]}_dpi{dpi}"
            out_dir.mkdir(parents=True, exist_ok=True)
            count = int(render_pdf(pdf, out_dir, dpi))
            pages = sorted(out_dir.glob("p[0-9][0-9][0-9][0-9].jpg"))
            if count < 1 or len(pages) != count:
                raise ValueError(
                    f"Renderer wrote {len(pages)} page files for {doc}, but reported {count} pages")
            rendered[doc] = (count, pages, _sha256_file(pdf))
        count, pages, pdf_sha = rendered[doc]
        page_parts: list[dict[str, Any]] = []
        for index, page in enumerate(pages, 1):
            page_parts.extend([
                {"type": "text", "text": f"Document page {index}/{count}:"},
                {"type": "image_path", "path": str(page), "asset_root": str(images_out)},
            ])
        messages = [{"role": "user", "content": page_parts + [
            {"type": "text", "text": PROMPT_MMLONG.format(q=question)}]}]
        records.append({
            "id": mmlong_sample_id(doc, question),
            "dataset": "mmlongbench_doc",
            "track": "native_quality",
            "messages": messages,
            "reference": row.get("answer"),
            "native_context_complete": True,
            "meta": {
                "doc_id": doc,
                "question": question,
                "doc_type": row.get("doc_type"),
                "evidence_pages": row.get("evidence_pages"),
                "evidence_sources": row.get("evidence_sources"),
                "answer_format": row.get("answer_format"),
                "page_count": count,
                "page_files": [p.name for p in pages],
                "render_dpi": dpi,
                "pdf_sha256": pdf_sha,
                "transform_version": TRANSFORM_VERSION,
                "adapter_provenance": "colab-vlm-bench-mmlong-v2-native-records; official_prompt_parity_pending",
            },
        })
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
