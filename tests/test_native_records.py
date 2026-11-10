"""ITER3 D2: native inference records from pinned upstream files (MemLens, MMLongBench-Doc).

Required properties (docs/harness/DATASET_READINESS.md §2.5, §3.6, §4):
  exact upstream sample ids, stable source ordering, repeatable output hashes, every image or page reference
  listed in full, every missing reference reported at once (nothing published), gold-free request check with
  a receipt, native records kept on the native track, and the scorer still BLOCKED_NO_JUDGE without a judge.

The MMLongBench-Doc renderer is injected. PyMuPDF is not installed in the test venv, and no installs are allowed.
The fake renderer writes placeholder page files; it does not produce real page images.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlmbench import quality
from vlmbench.artifacts import sha256
from vlmbench.datasets import native_records as nr
from vlmbench.datasets.memlens import normalize_memlens
from vlmbench.workloads import WorkloadError, load_jsonl


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf8")
    return path


def _mem_item(qid: str, answer: str, text: str = "filler text", images=None, sid: str = "s1") -> dict:
    return {
        "question_id": qid, "question_type": "information_extraction",
        "question": f"What about {qid}?", "answer": answer,
        "haystack_dates": ["2024-06-01"], "haystack_session_ids": [sid],
        "haystack_sessions": [[{"role": "user", "content": text, "images": images or []}]],
        "answer_session_ids": [sid],
    }


def _memlens_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    (data / "release_images" / "n").mkdir(parents=True)
    return data


# --- MemLens ----------------------------------------------------------------------------------------------

def test_memlens_ids_are_the_upstream_ids_in_source_order_and_the_output_is_repeatable(tmp_path):
    data = _memlens_data(tmp_path)
    src = _write_json(tmp_path / "src.json", [_mem_item("q_b", "X"), _mem_item("q_a", "Y")])
    out1, out2 = tmp_path / "n1.jsonl", tmp_path / "n2.jsonl"
    summary = normalize_memlens(src, data, out1)
    normalize_memlens(src, data, out2)
    ids = [json.loads(line)["id"] for line in out1.read_text(encoding="utf8").splitlines()]
    assert ids == ["q_b", "q_a"], "stable source order, exact upstream ids"
    assert sha256(out1) == sha256(out2), "same input, same output bytes"
    assert summary["transform"]["version"] == nr.TRANSFORM_VERSION
    assert summary["transform"]["source_sha256"] == sha256(src)
    assert summary["transform"]["output_sha256"] == sha256(out1)


def test_memlens_output_is_native_and_is_refused_on_the_derived_track(tmp_path):
    data = _memlens_data(tmp_path)
    src = _write_json(tmp_path / "src.json", [_mem_item("q1", "X")])
    out = tmp_path / "n.jsonl"
    normalize_memlens(src, data, out)
    rows = load_jsonl(out, dataset="memlens", track="native_quality")
    assert rows[0]["track"] == "native_quality" and rows[0]["native_context_complete"] is True
    with pytest.raises(WorkloadError):
        load_jsonl(out, dataset="memlens", track="derived_serving")


def test_memlens_reports_every_missing_image_at_once_and_publishes_nothing(tmp_path):
    data = _memlens_data(tmp_path)
    src = _write_json(tmp_path / "src.json", [
        _mem_item("q1", "X", images=[{"file": "a/missing1.jpg"}]),
        _mem_item("q2", "Y", images=[{"file": "b/missing2.jpg"}]),
    ])
    out = tmp_path / "n.jsonl"
    with pytest.raises(nr.MissingReferences) as exc:
        normalize_memlens(src, data, out)
    refs = {(m["id"], m["path"]) for m in exc.value.missing}
    assert refs == {("q1", "a/missing1.jpg"), ("q2", "b/missing2.jpg")}
    assert not out.exists() and not out.with_name(out.name + ".pending").exists()


def test_memlens_meta_lists_every_image_reference_with_its_hash(tmp_path):
    data = _memlens_data(tmp_path)
    img = data / "release_images" / "n" / "one.jpg"
    img.write_bytes(b"\xff\xd8one")
    src = _write_json(tmp_path / "src.json", [
        _mem_item("q1", "X", images=[{"file": "n/one.jpg"}, {"file": "n/one.jpg"}]),
    ])
    out = tmp_path / "n.jsonl"
    normalize_memlens(src, data, out)
    meta = json.loads(out.read_text(encoding="utf8"))["meta"]
    assert meta["image_count"] == 2
    assert meta["image_refs"] == [{"path": "n/one.jpg", "sha256": sha256(img)}] * 2


def test_memlens_receipt_marks_gold_in_haystack_as_context_and_never_as_clean(tmp_path):
    data = _memlens_data(tmp_path)
    src = _write_json(tmp_path / "src.json", [
        _mem_item("q_clean", "June", text="nothing relevant here"),
        _mem_item("q_ctx", "June", text="we met in June"),
    ])
    out = tmp_path / "n.jsonl"
    summary = normalize_memlens(src, data, out)
    receipt = summary["gold_free_receipt"]
    assert receipt["status"] == "GOLD_TOKEN_IN_CONTEXT"
    assert receipt["gold_token_in_context"] == ["q_ctx"]
    assert receipt["structural_violations"] == []
    assert receipt["records"] == 2


# --- gold-free check ---------------------------------------------------------------------------------------

def test_gold_free_check_refuses_gold_keys_inside_messages():
    rec = {"id": "q1", "dataset": "memlens", "track": "native_quality", "native_context_complete": True,
           "reference": "June",
           "messages": [{"role": "user", "content": [{"type": "text", "text": "x", "answer": "June"}]}]}
    check = nr.check_record(rec)
    assert check["structural"], "a gold key inside the request must be a structural violation"
    assert nr.receipt([check])["status"] == "LEAKED"


def test_gold_free_check_refuses_a_message_part_equal_to_the_gold():
    rec = {"id": "q1", "dataset": "memlens", "track": "native_quality", "native_context_complete": True,
           "reference": "June",
           "messages": [{"role": "user", "content": [{"type": "text", "text": "June"}]}]}
    check = nr.check_record(rec)
    assert check["structural"]
    assert nr.receipt([check])["status"] == "LEAKED"


def test_mmlong_template_wording_is_not_a_gold_leak_for_an_unanswerable_row():
    rec = {"id": "u1", "dataset": "mmlongbench_doc", "track": "native_quality", "native_context_complete": True,
           "reference": "Not answerable",
           "messages": [{"role": "user", "content": [
               {"type": "text", "text": "Document page 1/1:"},
               {"type": "image_path", "path": "/x/p0001.jpg", "asset_root": "/x"},
               {"type": "text", "text": "Question: What is the total?\nGive a concise answer; if not answerable say so."},
           ]}]}
    check = nr.check_record(rec)
    assert check["gold_in_context"] is False
    assert nr.receipt([check])["status"] == "GOLD_FREE"


def test_the_gold_inside_the_question_text_is_not_counted_as_context():
    rec = {"id": "q1", "dataset": "memlens", "track": "native_quality", "native_context_complete": True,
           "reference": "June",
           "messages": [{"role": "user", "content": [
               {"type": "text", "text": "nothing here"},
               {"type": "text", "text": "\nQuestion: When was June?"}]}]}
    check = nr.check_record(rec)
    assert check["gold_in_context"] is False


def test_transform_params_hash_is_stable_and_changes_with_the_parameters():
    base = {"dpi": 96, "version": nr.TRANSFORM_VERSION}
    assert nr.transform_params_sha256(base) == nr.transform_params_sha256(
        {"version": nr.TRANSFORM_VERSION, "dpi": 96})
    assert nr.transform_params_sha256(base) != nr.transform_params_sha256({**base, "dpi": 72})


# --- MMLongBench-Doc ---------------------------------------------------------------------------------------

def _fake_renderer(calls: list, pages_per_pdf: dict[str, int], *, write_fewer: bool = False):
    def render(pdf_path: Path, out_dir: Path, dpi: int) -> int:
        calls.append((pdf_path.name, dpi))
        count = pages_per_pdf[pdf_path.name]
        out_dir.mkdir(parents=True, exist_ok=True)
        written = count - 1 if write_fewer else count
        for index in range(1, written + 1):
            (out_dir / f"p{index:04d}.jpg").write_bytes(b"jpg-" + str(index).encode())
        return count
    return render


def _mm_row(doc: str, question: str, answer: str = "1") -> dict:
    return {"doc_id": doc, "question": question, "answer": answer, "answer_format": "Int",
            "evidence_pages": [1], "evidence_sources": ["Text"], "doc_type": "Research report"}


def _pdfs(tmp_path: Path, names: dict[str, bytes]) -> Path:
    root = tmp_path / "pdfs"
    root.mkdir()
    for name, data in names.items():
        (root / name).write_bytes(data)
    return root


def test_mmlong_reports_every_missing_document_and_renders_nothing(tmp_path):
    calls: list = []
    rows = [_mm_row("a.pdf", "Q1"), _mm_row("b.pdf", "Q2")]
    images = tmp_path / "img"
    with pytest.raises(nr.MissingReferences) as exc:
        nr.build_mmlong_records(rows, pdf_root=tmp_path / "pdfs", images_out=images, dpi=96,
                                render_pdf=_fake_renderer(calls, {}))
    assert {m["path"] for m in exc.value.missing} == {"a.pdf", "b.pdf"}
    assert calls == [], "nothing may be rendered before every reference is checked"
    assert not images.exists() or not any(images.iterdir())


def test_mmlong_record_carries_every_page_and_the_pdf_hash(tmp_path):
    pdf_bytes = b"PAGES=3"
    root = _pdfs(tmp_path, {"doc.pdf": pdf_bytes})
    calls: list = []
    records = nr.build_mmlong_records([_mm_row("doc.pdf", "Q1")], pdf_root=root, images_out=tmp_path / "img",
                                      dpi=96, render_pdf=_fake_renderer(calls, {"doc.pdf": 3}))
    rec = records[0]
    pages = [p for p in rec["messages"][0]["content"] if p["type"] == "image_path"]
    assert len(pages) == 3 == rec["meta"]["page_count"]
    assert [Path(p["path"]).name for p in pages] == ["p0001.jpg", "p0002.jpg", "p0003.jpg"]
    assert rec["meta"]["pdf_sha256"] == sha256(root / "doc.pdf")
    assert rec["meta"]["render_dpi"] == 96
    assert rec["track"] == "native_quality" and rec["native_context_complete"] is True


def test_mmlong_fails_closed_when_the_renderer_returns_fewer_pages_than_it_counts(tmp_path):
    root = _pdfs(tmp_path, {"doc.pdf": b"PAGES=3"})
    with pytest.raises(ValueError, match="page"):
        nr.build_mmlong_records([_mm_row("doc.pdf", "Q1")], pdf_root=root, images_out=tmp_path / "img",
                                dpi=96, render_pdf=_fake_renderer([], {"doc.pdf": 3}, write_fewer=True))


def test_mmlong_ids_do_not_depend_on_row_order(tmp_path):
    root = _pdfs(tmp_path, {"a.pdf": b"PAGES=1", "b.pdf": b"PAGES=1"})
    rows = [_mm_row("a.pdf", "Q1"), _mm_row("b.pdf", "Q2")]
    render = _fake_renderer([], {"a.pdf": 1, "b.pdf": 1})
    forward = nr.build_mmlong_records(rows, pdf_root=root, images_out=tmp_path / "img", dpi=96, render_pdf=render)
    backward = nr.build_mmlong_records(list(reversed(rows)), pdf_root=root, images_out=tmp_path / "img",
                                       dpi=96, render_pdf=render)
    by_id_fwd = {r["id"]: r["meta"]["question"] for r in forward}
    by_id_bwd = {r["id"]: r["meta"]["question"] for r in backward}
    assert by_id_fwd == by_id_bwd
    assert [r["id"] for r in forward] == list(reversed([r["id"] for r in backward]))


def test_mmlong_duplicate_doc_and_question_fails(tmp_path):
    root = _pdfs(tmp_path, {"a.pdf": b"PAGES=1"})
    rows = [_mm_row("a.pdf", "Q1"), _mm_row("a.pdf", "Q1")]
    with pytest.raises(ValueError, match="Duplicate"):
        nr.build_mmlong_records(rows, pdf_root=root, images_out=tmp_path / "img", dpi=96,
                                render_pdf=_fake_renderer([], {"a.pdf": 1}))


def test_mmlong_records_pass_the_gold_free_check_and_keep_the_answer_out_of_the_request(tmp_path):
    root = _pdfs(tmp_path, {"a.pdf": b"PAGES=1"})
    records = nr.build_mmlong_records([_mm_row("a.pdf", "What is the total?", answer="Not answerable")],
                                      pdf_root=root, images_out=tmp_path / "img", dpi=96,
                                      render_pdf=_fake_renderer([], {"a.pdf": 1}))
    receipt = nr.receipt([nr.check_record(r) for r in records])
    assert receipt["status"] == "GOLD_FREE", receipt
    assert "Not answerable" not in json.dumps(records[0]["messages"], ensure_ascii=False)


# --- scorer gate -------------------------------------------------------------------------------------------

def test_memlens_official_import_stays_blocked_without_a_judge(tmp_path):
    upstream = _write_json(tmp_path / "upstream.json", {"acc": 1.0})
    res = quality.import_official(upstream, dataset="memlens", judge_revision="unpinned",
                                  sample_manifest_sha256="a" * 64)
    assert res["status"] == "BLOCKED_NO_JUDGE"
    assert res["official_acc"] is None and res["official_f1"] is None
