from pathlib import Path

import pytest

from vlmbench import cli
from vlmbench.bench import request_payload
from vlmbench.engine import wait_ready


@pytest.mark.parametrize("finish_reason,http_status", [
    ("stop", 200), ("length", 200), (None, 200), (None, 500),
])
def test_quality_predictions_preserve_finish_reason(tmp_path, monkeypatch,
                                                  finish_reason, http_status):
    import asyncio
    import json

    import httpx

    from vlmbench import bench

    row = {"id": "quality-one", "track": "native_quality", "max_output_tokens": 256,
           "natural_eos": True, "messages": [{"role": "user", "content": "Describe."}]}
    def respond(request):
        payload = json.loads(request.content)
        assert payload["max_tokens"] == 256
        assert "ignore_eos" not in payload and "min_tokens" not in payload
        choice = {"message": {"content": "An answer."}}
        if finish_reason is not None:
            choice["finish_reason"] = finish_reason
        return httpx.Response(http_status, json={"choices": [choice]})
    client = httpx.AsyncClient
    monkeypatch.setattr(bench.httpx, "AsyncClient", lambda **kw: client(
        transport=httpx.MockTransport(respond), **kw))
    local = tmp_path / "local"
    asyncio.run(bench.run_quality_predictions(
        {"serving": {"port": 8000}, "quality": {"max_output_tokens": 256}},
        "qwen35_08b", "base", "amber", [row], local, tmp_path / "drive"))
    prediction = json.loads((local / "quality/amber__qwen35_08b__base.jsonl").read_text())
    assert "finish_reason" in prediction
    assert prediction["finish_reason"] == finish_reason
    assert prediction["status"] == ("SUCCESS" if http_status == 200 else "FAILED")


def test_fixed_serving_and_natural_eos_quality_have_distinct_payloads():
    sample = {"messages": [{"role": "user", "content": "Describe the image."}]}
    for model in ("qwen35_08b", "qwen35_4b"):
        serving = request_payload(sample, model_id=model, max_tokens=256,
                                  streaming=True, fixed_output=True)
        quality = request_payload(sample, model_id=model, max_tokens=256,
                                  streaming=False, fixed_output=False)
        assert serving["ignore_eos"] is True
        assert serving["min_tokens"] == serving["max_tokens"] == 256
        assert quality["max_tokens"] == 256
        assert "ignore_eos" not in quality and "min_tokens" not in quality
        for payload in (serving, quality):
            assert payload["temperature"] == 0 and payload["top_p"] == 1
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.parametrize("count,status", [(256, "PASS_MEASURED"),
                                          (255, "PARTIAL_FAILURE"),
                                          (None, "PARTIAL_FAILURE")])
def test_serving_pass_checks_actual_usage_and_records_contract(tmp_path, monkeypatch, count, status):
    import asyncio
    import json

    import httpx

    from vlmbench import bench

    row = {"id": "one", "messages": [{"role": "user", "content": "Question"}]}
    monkeypatch.setattr(bench, "exact_bin_pool", lambda *a, **kw: [row])
    monkeypatch.setattr(bench, "sample_unique", lambda *a: [row])
    class Sampler:
        def __init__(self, *_):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def report(self):
            return dict.fromkeys(("peak_total_gpu_used_mib", "minimum_available_system_ram_gib",
                                  "minimum_free_local_ssd_gib", "peak_system_ram_used_gib"), 0)
    monkeypatch.setattr(bench, "GPUSampler", Sampler)
    def respond(request):
        payload = json.loads(request.content)
        assert payload["min_tokens"] == payload["max_tokens"] == 256
        assert payload["ignore_eos"] is True
        events = [{"choices": [{"delta": {"content": "answer"}}]},
                  {"choices": [], "usage": {"completion_tokens": count}}]
        text = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
        return httpx.Response(200, text=text + "data: [DONE]\n\n")
    client = httpx.AsyncClient
    monkeypatch.setattr(bench.httpx, "AsyncClient", lambda **kw: client(
        transport=httpx.MockTransport(respond), **kw))
    cfg = {"performance": {"max_bin_relative_error": .05, "request_timeout_s": 1,
                           "cell_timeout_s": 1, "output_tokens": 256},
           "serving": {"max_model_len": 32768, "port": 8000}}
    cell = {"model": "qwen35_08b", "dataset": "memlens", "mode": "base",
            "input_length": 128, "output_tokens": 256, "concurrency": 1,
            "requests": 1, "warmup": 0, "seed": 42, "repetition": 0}
    result = asyncio.run(bench.run_performance_cell(
        cfg, cell, [row], tmp_path / "local", tmp_path / "drive", "qwen35_08b", "sha"))
    assert result["status"] == status
    assert result["execution_pass"] == "FIXED_OUTPUT_SERVING_256"
    assert result["successful_requests"] == int(count == 256)


def test_run_job_readiness_guard_is_before_a100_and_weight_staging():
    # For budget safety, these are source-order tests on a narrowly scoped CLI
    # and do not require allocating a fake GPU. They catch accidental overwrites
    # of our preflight when another engine change updates this same file.
    src = Path(cli.__file__).read_text(encoding="utf8")
    relevant = src[src.index("def run_job("):src.index("\ndef main(", src.index("def run_job("))]
    inventory = relevant.index("inventory = a100_inventory()")
    stage = relevant.index("target = staged_copy(")
    assert relevant.index("checkpoint_inventory(") < inventory
    assert relevant.index("serving_readiness(") < inventory
    assert relevant.index("BLOCKED_NO_ELIGIBLE_CELLS") < inventory
    assert relevant.index("NO_PENDING_CELLS_NO_GPU_START") < inventory
    assert relevant.index("BLOCKED_QUALITY_IMAGE_LIMIT") < inventory
    assert inventory < stage < relevant.index("start_server(")


def test_run_job_quality_resume_skips_gpu():
    src = Path(cli.__file__).read_text(encoding="utf8")
    assert src.index("QUALITY_ALREADY_RECORDED_NO_GPU_START") < src.index("inventory = a100_inventory()")
    assert "expected_model=args.model" in src


def test_wait_ready_checks_correct_servable_name(tmp_path, monkeypatch):
    import json

    from vlmbench import engine
    class Process:
        returncode = None
        def poll(self):
            return None
    class MockResponse:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return None
        def read(self, *args):
            return json.dumps({"data": [{"id": "unrelated-model"}]}).encode()
    monkeypatch.setattr(engine.urllib.request, "urlopen", lambda *a, **kw: MockResponse())
    try:
        wait_ready(Process(), 8000, timeout_s=1, expected_model="qwen35_08b")
    except RuntimeError as exc:
        assert "wrong model" in str(exc)
    else:
        raise AssertionError("Must not accept stale model bound to port")
