import json
from pathlib import Path

import pytest

from vlmbench.datasets import normalize_memlens
from vlmbench.workloads import (WorkloadError, exact_bin_pool, load_jsonl,
                                resolve_messages, sample_unique, ensure_no_gold_in_request)


def test_memlens_preserves_all_sessions_and_gold_isolated(tmp_path):
    imgs = tmp_path / "data" / "release_images" / "needle_images"
    imgs.mkdir(parents=True)
    (imgs / "one.jpg").write_bytes(b"fake-image-for-path-only")
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps([{
        "question_id": "q_01", "question_type": "information_extraction",
        "question": "What did we discuss?", "answer": "June",
        "haystack_dates": ["2024-06-01"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "I was there in June",
                                "images": [{"file": "needle_images/one.jpg"}]}]],
        "answer_session_ids": ["s1"],
    }]))
    output = tmp_path / "native.jsonl"
    result = normalize_memlens(source, tmp_path / "data", output)
    assert result["count"] == 1
    records = load_jsonl(output, dataset="memlens", track="native_quality")
    r = records[0]
    assert len(r["messages"]) == 1
    assert r["reference"] == "June"
    assert "Question: What did we discuss?" in r["messages"][-1]["content"][-1]["text"]
    assert r["meta"]["image_count"] == 1
    request = {"model": "x", "messages": r["messages"]}
    ensure_no_gold_in_request(r, request)


def test_missing_image_is_not_dropped(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps([{
        "question_id": "q_01", "question_type": "answer_refusal",
        "question": "Question?", "answer": "not known",
        "haystack_dates": ["today"], "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "assistant", "content": "Text", "images":
                               [{"file": "needle_images/no.jpg"}]}]],
    }]))
    with pytest.raises(ValueError, match="Missing"):
        normalize_memlens(source, tmp_path, tmp_path / "out.jsonl")


def test_bin_requires_real_model_token_counts():
    rows = [{
        "id": "q1", "model_token_counts": {"qwen35_08b": {
            "verified_by": "vllm_effective_token_accounting",
            "text_tokens": 1022, "effective_total_tokens": 1450,
        }}
    }, {"id": "q2", "model_token_counts": {"qwen35_08b": {
        "verified_by": "approximate_word_count", "text_tokens": 1024, "effective_total_tokens": 1450,
    }}}]
    assert [r["id"] for r in exact_bin_pool(rows, "qwen35_08b", 1024, 0.05)] == ["q1"]
    assert exact_bin_pool(rows, "qwen35_2b", 1024, 0.05) == []


def test_sample_unique_never_duplicates_ids():
    rows = [{"id": str(i)} for i in range(10)]
    selected = sample_unique(rows, 5, 42)
    assert len({r["id"] for r in selected}) == 5
    assert sample_unique(rows, 5, 42) == selected
    with pytest.raises(WorkloadError, match="INSUFFICIENT"):
        sample_unique(rows, 11, 42)


def test_relative_image_escape_rejected(tmp_path):
    with pytest.raises(WorkloadError, match="Unsafe"):
        resolve_messages([{"role": "user", "content": [{
            "type": "image_path", "path": str(tmp_path / "outside.png"),
            "asset_root": str(tmp_path / "other")
        }]}])


def test_derived_missing_origin_refused(tmp_path):
    p = tmp_path / "derived.jsonl"
    p.write_text(json.dumps({"id": "q1", "dataset": "memlens", "track": "derived_serving",
                            "messages": [{"role": "user", "content": "prompt"}]}) + "\n")
    with pytest.raises(WorkloadError, match="[Dd]erived"):
        load_jsonl(p, dataset="memlens", track="derived_serving")


def test_memlens_streams_question_by_question_and_normalizes_ai_assistant(tmp_path, monkeypatch):
    images = tmp_path / "publisher" / "release_images"
    images.mkdir(parents=True)
    (images / "one.jpg").write_bytes(b"fixture")
    entries = [
        {
            "question_id": f"q_{i:04d}",
            "question_type": "information_extraction",
            "question": "What month?", "answer": "June",
            "haystack_dates": ["2026-06-01"], "haystack_session_ids": ["s1"],
            "haystack_sessions": [[
                {"role": "USER", "content": "June", "images": [{"file": "one.jpg"}]},
                {"role": "AI Assistant", "content": "Understood", "images": []},
            ]],
        }
        for i in range(150)
    ]
    source = tmp_path / "dataset_32k.json"
    source.write_text(json.dumps(entries), encoding="utf8")
    output = tmp_path / "native.jsonl"
    original_read_text = Path.read_text

    def block_full_source_reads(path, *args, **kwargs):
        if path == source:
            raise AssertionError("Never call read_text/json.load on the entire MemLens source")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", block_full_source_reads)
    result = normalize_memlens(source, tmp_path / "publisher", output)
    assert result["count"] == 150
    assert result["memory_strategy"] == "ijson_stream_one_question_and_atomic_jsonl"
    assert not output.with_name(output.name + ".pending").exists()
    rows = load_jsonl(output, dataset="memlens", track="native_quality")
    assert len(rows) == 150
    assert "[assistant] Understood" in str(rows[0]["messages"])
    assert rows[0]["meta"]["image_count"] == 1
    assert rows[-1]["id"] == "q_0149"


def test_memlens_duplicate_failure_does_not_publish_partial_output(tmp_path):
    row = {
        "question_id": "q", "question_type": "answer_refusal",
        "question": "unknown?", "answer": "unknown",
        "haystack_session_ids": ["s"], "haystack_dates": ["today"],
        "haystack_sessions": [[{"role": "User", "content": "Hello", "images": []}]],
    }
    source = tmp_path / "dataset.json"
    source.write_text(json.dumps([row, row]), encoding="utf8")
    output = tmp_path / "native.jsonl"
    with pytest.raises(ValueError, match="Duplicate MemLens"):
        normalize_memlens(source, tmp_path, output)
    assert not output.exists()
    assert not output.with_name(output.name + ".pending").exists()


def test_derived_candidates_index_only_native_offsets(tmp_path, monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys
    from vlmbench.notebook_bridge import build_serving_candidates

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(x) for x in text]
        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(x) for x in ids)

    module = ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: FakeTokenizer())
    monkeypatch.setitem(sys.modules, "transformers", module)
    row = {
        "id": "q1", "dataset": "memlens", "track": "native_quality",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "x" * 500},
            {"type": "image_path", "path": "/dummy/img.jpg", "asset_root": "/dummy"},
            {"type": "text", "text": "Question: Which?"}
        ]}],
    }
    source = tmp_path / "native.jsonl"
    source.write_text(json.dumps(row) + "\n", encoding="utf8")
    output = tmp_path / "derived.jsonl"
    report = build_serving_candidates(
        source, output, dataset="memlens", model_directory=tmp_path,
        input_lengths=[128], max_per_bin=160)
    assert report["total_rows"] == 1
    assert report["eligible_per_bin"]["128"] == 1
    derived = load_jsonl(output, dataset="memlens", track="derived_serving")
    assert derived[0]["meta"]["official_quality_allowed"] is False
    assert derived[0]["image_count"] == 1
