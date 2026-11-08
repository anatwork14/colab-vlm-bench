"""Prove the native long-context quality fail-fast policy without A100."""
import asyncio
import json

import httpx
import pytest

from vlmbench import bench


def test_quality_pilot_aborts_after_two_http_errors_and_exports(tmp_path, monkeypatch):
    orig = httpx.AsyncClient
    calls = []
    def fail(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "Context length exceeded"}})
    monkeypatch.setattr(
        bench.httpx, "AsyncClient",
        lambda *a, **kw: orig(*a, transport=httpx.MockTransport(fail), **kw),
    )
    rows = [
        {"id": f"sample{i}", "messages": [{"role": "user", "content": "Q"}]}
        for i in range(10)
    ]
    cfg = {
        "serving": {"port": 8000},
        "quality": {"max_output_tokens": 256},
        "execution_scope": {"kind": "engineering_pilot"},
    }
    root, drive = tmp_path / "scratch", tmp_path / "drive"
    result = asyncio.run(
        bench.run_quality_predictions(
            cfg, "qwen35_08b", "base", "memlens", rows, root, drive)
    )
    assert result["status"] == "QUALITY_STOPPED_AFTER_REPEATED_FAILURES"
    assert result["count"] == 2
    assert result["expected_count"] == 10
    assert result["failures"] == 2
    assert len(calls) == 2  # no 10-request CU waste
    assert (drive / "quality" / "memlens__qwen35_08b__base.jsonl").is_file()
    assert (drive / "quality" / "memlens__qwen35_08b__base.receipt.json").is_file()
    with pytest.raises(ValueError, match="inconsistent"):
        asyncio.run(
            bench.run_quality_predictions(
                cfg, "qwen35_08b", "base", "memlens", rows, root, drive)
        )


def test_warmup_must_be_parallel_at_offered_ccu():
    from pathlib import Path
    src = Path(bench.__file__).read_text(encoding="utf8")
    assert "warm_observations = await asyncio.gather" in src
    assert "for warmup in warmups:" not in src
