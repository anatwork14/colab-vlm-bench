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
            "effective_total_tokens": 1022,
        }}
    }, {"id": "q2", "model_token_counts": {"qwen35_08b": {
        "verified_by": "approximate_word_count", "effective_total_tokens": 1024,
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
