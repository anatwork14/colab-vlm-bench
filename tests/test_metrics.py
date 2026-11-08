import asyncio
import json
from pathlib import Path

import httpx
import pytest

from vlmbench.bench import quantile, request_payload, stream_completion
from vlmbench.quality import exact_match, import_official, score_proxy, token_f1
from vlmbench.report import aggregate, render_markdown, write_report
from vlmbench.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "a100_campaign.json"


def test_quantiles_and_tpot_definition():
    assert quantile([1, 2, 3, 4, 5], 0.95) == pytest.approx(4.8)
    assert quantile([], 0.95) is None


def test_stream_usage_counts_and_no_gold():
    event1 = json.dumps({"choices": [{"delta": {"content": "Hi"}}]})
    event2 = json.dumps({"choices": [], "usage": {"completion_tokens": 2, "prompt_tokens": 7}})
    def respond(request):
        return httpx.Response(200, text=f"data: {event1}\n\ndata: {event2}\n\ndata: [DONE]\n\n")
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await stream_completion(client, "https://test/v1/chat/completions",
                                           {"stream": True}, "test-id")
    x = asyncio.run(run())
    assert x["status"] == "HTTP_OK"
    assert x["output_tokens"] == 2
    assert x["ttft_ms"] is not None
    assert x["tpot_ms"] is not None
    assert x["stream_content_chunks"] == 1


def test_fixed_output_stream_payload_not_quality_payload():
    row = {"id": "q", "reference": "secret", "messages": [{"role": "user", "content": "Question?"}]}
    perf = request_payload(row, model_id="qwen", max_tokens=256,
                           streaming=True, fixed_output=True)
    quality = request_payload(row, model_id="qwen", max_tokens=256,
                              streaming=False, fixed_output=False)
    assert perf["ignore_eos"] and perf["min_tokens"] == 256
    assert perf["stream_options"]["include_usage"]
    assert "ignore_eos" not in quality
    assert "reference" not in str(perf)


def test_proxy_never_official():
    out = score_proxy([{"id": "q", "status": "SUCCESS", "prediction": "June"}],
                      [{"id": "q", "reference": "June"}])
    assert out["status"] == "QUALITY_PROXY_ONLY"
    assert out["token_f1"] == 1
    assert out["not_official"]
    assert token_f1("red green", "red") == pytest.approx(2 / 3)
    assert exact_match("June!", "june") == 1


def test_official_import_requires_provenance(tmp_path):
    p = tmp_path / "result.json"
    p.write_text("{}")
    with pytest.raises(ValueError, match="pin"):
        import_official(p, dataset="memlens", judge_revision="", sample_manifest_sha256="")
    obj = import_official(p, dataset="memlens", judge_revision="frozen", sample_manifest_sha256="abc")
    assert obj["status"] == "QUALITY_OFFICIAL_IMPORTED_UNVERIFIED"


def test_empty_campaign_not_misreported_as_gpu(tmp_path):
    cfg = load_config(CONFIG)
    s = write_report(tmp_path, cfg)
    assert s["total_planned_cells"] == 2520
    assert s["status_counts"] == {"NOT_RUN": 2520}
    report = (tmp_path / "RUNBOOK.md").read_text()
    assert "NOT_MEASURED" in report
    assert "PASS_MEASURED" not in s["status_counts"]


def test_deterministic_pilot_native_quality_subset():
    from vlmbench.quality import deterministic_subset
    rows = [{"id": f"sample-{i}"} for i in range(100)]
    a = deterministic_subset(rows, 48, 42)
    b = deterministic_subset(rows, 48, 42)
    assert [x["id"] for x in a] == [x["id"] for x in b]
    assert len({x["id"] for x in a}) == 48
    assert deterministic_subset(rows, 0, 42) == rows
    assert [x["id"] for x in deterministic_subset(rows, 48, 43)] != [x["id"] for x in a]


def test_greedy_accuracy_payload_is_frozen():
    from vlmbench.config import ConfigError, validate
    from vlmbench.bench import request_payload
    from pathlib import Path
    c = load_config(CONFIG)
    assert c["serving"]["temperature"] == 0
    row = {"id": "a", "messages": [{"role": "user", "content": "Q"}]}
    request = request_payload(row, model_id="qwen35_08b", max_tokens=20,
                              streaming=False, fixed_output=False)
    assert request["temperature"] == 0 and request["top_p"] == 1.0
    assert request["chat_template_kwargs"]["enable_thinking"] is False
    c["serving"]["temperature"] = 0.7
    with pytest.raises(ConfigError, match="temperature=0"):
        validate(c)


def test_host_gpu_memory_are_separately_named_in_sampler():
    from vlmbench.telemetry import GPUSampler
    monitor = GPUSampler()
    monitor.samples = [
        {"used_mib": 12000, "ram_available_gib": 62.0,
         "ram_used_gib": 18.0, "disk_free_gib": 130.0},
        {"used_mib": 14000, "ram_available_gib": 58.0,
         "ram_used_gib": 22.0, "disk_free_gib": 128.0},
    ]
    report = monitor.report()
    assert report["peak_total_gpu_used_mib"] == 14000
    assert report["minimum_available_system_ram_gib"] == 58.0
    assert report["minimum_free_local_ssd_gib"] == 128.0
    assert report["peak_system_ram_used_gib"] == 22.0
    assert report["physical_kv_bytes"] == "NOT_MEASURED"
