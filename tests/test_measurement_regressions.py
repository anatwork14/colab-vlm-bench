"""Regression tests for measurement bugs found in the 2026-11-09 benchmark audit.

Each test encodes the corrected behaviour. They were written before the fixes
and failed against cfccc4b (see docs/audit/NUMERIC_INTEGRITY_LEDGER.md).
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from time import perf_counter

import httpx
import pytest

from vlmbench.bench import admitted_stream, stream_completion
from vlmbench.quality import score_proxy
from vlmbench.telemetry import parse_nvidia_smi_csv


# ---------------------------------------------------------------- quality denominator (C2 accuracy/EM/F1)

def test_missing_prediction_is_counted_as_zero_not_dropped():
    # Circuit breaker or crash left id "b" without a prediction row. It must stay
    # in the denominator with EM=0 / F1=0, never silently disappear.
    out = score_proxy(
        [{"id": "a", "status": "SUCCESS", "prediction": "June"}],
        [{"id": "a", "reference": "June"}, {"id": "b", "reference": "May"}],
    )
    assert out["count"] == 2
    assert out["exact_match"] == pytest.approx(0.5)
    assert out["token_f1"] == pytest.approx(0.5)
    assert out["missing_predictions"] == 1


def test_duplicate_prediction_id_is_rejected():
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        score_proxy(
            [{"id": "a", "status": "SUCCESS", "prediction": "June"},
             {"id": "a", "status": "SUCCESS", "prediction": "June"}],
            [{"id": "a", "reference": "June"}],
        )


def test_prediction_for_unknown_id_is_rejected():
    with pytest.raises(ValueError, match="[Uu]nknown"):
        score_proxy(
            [{"id": "a", "status": "SUCCESS", "prediction": "June"},
             {"id": "zzz", "status": "SUCCESS", "prediction": "x"}],
            [{"id": "a", "reference": "June"}],
        )


def test_example_without_reference_is_reported_not_silently_dropped():
    out = score_proxy(
        [{"id": "a", "status": "SUCCESS", "prediction": "June"}],
        [{"id": "a", "reference": "June"}, {"id": "b", "reference": None}],
    )
    assert out["count"] == 1
    assert out["unscored_without_reference"] == 1


# ---------------------------------------------------------------- GPU telemetry parse (C2 peak GPU memory)

def test_single_gpu_csv_parses():
    used, util, gpus = parse_nvidia_smi_csv("12000, 55\n")
    assert (used, util, gpus) == (12000, 55, 1)


def test_multi_gpu_csv_sums_devices_instead_of_dropping_sample():
    used, util, gpus = parse_nvidia_smi_csv("1234, 0\n5678, 3\n")
    assert used == 1234 + 5678
    assert util == 3
    assert gpus == 2


def test_malformed_csv_raises_so_sampler_can_count_failures():
    with pytest.raises(ValueError):
        parse_nvidia_smi_csv("not, a, number\n")


def test_sampler_reports_parse_failures_instead_of_silent_drop(monkeypatch):
    import subprocess

    from vlmbench import telemetry

    calls = {"n": 0}
    sampler = telemetry.GPUSampler(interval_s=0.001)

    class Done:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 2:
            sampler.stop_event.set()  # end the sampling loop deterministically
            raise subprocess.SubprocessError("stop")
        return Done("bad-output\n")

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)
    sampler._sample()
    report = sampler.report()
    assert report["samples"] == 0
    assert report["parse_failures"] >= 1


# ---------------------------------------------------------------- client-side admission latency (C2 e2e / queue)

def _sse(text_events: int = 1, completion_tokens: int = 2) -> str:
    events = [json.dumps({"choices": [{"delta": {"content": "Hi"}}]})
              for _ in range(text_events)]
    events.append(json.dumps({"choices": [],
                              "usage": {"completion_tokens": completion_tokens,
                                        "prompt_tokens": 7}}))
    return "".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n"


def test_e2e_is_measured_from_admission_including_client_queue_wait():
    """CCU=1 with two requests submitted together: the second waits behind the first.

    The second request's e2e must include that queue wait, because a user submitting
    at t0 experiences it. TTFT stays HTTP-send based, per docs/METRICS.md.
    """
    async def slow_handler(request):
        await asyncio.sleep(0.2)
        return httpx.Response(200, text=_sse())

    async def run():
        semaphore = asyncio.Semaphore(1)
        submitted = perf_counter()
        async with httpx.AsyncClient(transport=httpx.MockTransport(slow_handler)) as client:
            return await asyncio.gather(
                admitted_stream(client, "http://t/v1", {"stream": True}, "a", semaphore, submitted),
                admitted_stream(client, "http://t/v1", {"stream": True}, "b", semaphore, submitted),
            )

    first, second = asyncio.run(run())
    assert first["status"] == "HTTP_OK" and second["status"] == "HTTP_OK"
    assert first["client_queue_wait_ms"] < 100
    assert second["client_queue_wait_ms"] >= 150
    assert second["e2e_ms"] >= 350, "e2e must include client queue wait from admission"
    assert second["ttft_ms"] < second["e2e_ms"]
    assert second["ttft_ms"] < 350, "TTFT is HTTP-send based, must not include queue wait"


def test_stream_completion_is_service_timing_only():
    # stream_completion has no admission knowledge; slot wait is added by admitted_stream.
    def respond(request):
        return httpx.Response(200, text=_sse())

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await stream_completion(client, "http://t/v1", {"stream": True}, "x")

    x = asyncio.run(run())
    assert "client_queue_wait_ms" not in x
    assert x["e2e_ms"] >= x["ttft_ms"]


# ---------------------------------------------------------------- MMLongBench render cache (C1 derived images)

class _FakeMatrix:
    def __init__(self, a, d):
        self.a = a


class _FakePixmap:
    def __init__(self, payload: bytes):
        self.payload = payload

    def save(self, path):
        from pathlib import Path
        Path(path).write_bytes(self.payload)


class _FakePage:
    def __init__(self, renders: list):
        self.renders = renders

    def get_pixmap(self, matrix, alpha):
        dpi = round(matrix.a * 72)
        self.renders.append(dpi)
        return _FakePixmap(f"render-dpi-{dpi}".encode())


class _FakeDoc(list):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _install_fake_fitz(monkeypatch, renders: list):
    fake = types.ModuleType("fitz")
    fake.Matrix = _FakeMatrix
    fake.open = lambda path: _FakeDoc([_FakePage(renders)])
    monkeypatch.setitem(sys.modules, "fitz", fake)


def test_mmlong_render_cache_is_keyed_by_dpi(tmp_path, monkeypatch):
    from pathlib import Path

    from vlmbench.datasets.mmlong import normalize_mmlong

    pdf_root = tmp_path / "docs"
    pdf_root.mkdir()
    (pdf_root / "d.pdf").write_bytes(b"%PDF-fake")
    source = tmp_path / "qa.json"
    source.write_text(json.dumps([{"doc_id": "d.pdf", "question": "Q?", "answer": "A"}]))
    images = tmp_path / "images"

    renders: list = []
    _install_fake_fitz(monkeypatch, renders)
    first = normalize_mmlong(source, pdf_root, images, tmp_path / "n72.jsonl", dpi=72)
    second = normalize_mmlong(source, pdf_root, images, tmp_path / "n96.jsonl", dpi=96)

    assert first["count"] == 1 and second["count"] == 1
    assert renders == [72, 96], "a 96-dpi run must not reuse 72-dpi page images"
    row_96 = json.loads((tmp_path / "n96.jsonl").read_text().splitlines()[0])
    image_path = Path(row_96["messages"][0]["content"][1]["path"])
    assert images.resolve() in image_path.parents
    assert image_path.read_bytes() == b"render-dpi-96"
