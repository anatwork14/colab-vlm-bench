"""AMBER campaign and read-only source integration without CUDA or networking."""
import asyncio
import json
import zipfile
from pathlib import Path

import pytest

from vlmbench import agent_pipeline, bench, source_mapping
from vlmbench.artifacts import sha256
from vlmbench.config import load_config, planned_cells, validate
from vlmbench.preflight import serving_readiness

ROOT = Path(__file__).resolve().parents[1]


def amber_config():
    cfg = load_config(ROOT / "configs/a100_qwen_priority_full.json")
    cfg["datasets"][1]["id"] = "amber"
    return cfg


def test_amber_has_six_ccus_without_context_bins():
    cfg = amber_config()
    validate(cfg)
    cells = [c for c in planned_cells(cfg) if c["dataset"] == "amber"]
    assert {c["input_length"] for c in cells} == {None}
    assert {c["concurrency"] for c in cells} == {1, 2, 4, 8, 16, 32}
    assert {c["model"] for c in cells} == {"qwen35_08b", "qwen35_4b"}
    assert {c["mode"] for c in cells} == {"base", "mtp"}
    assert len(cells) == 2 * 2 * 3 * 6


@pytest.mark.parametrize("text_tokens", [12, None])
def test_amber_readiness_uses_real_effective_counts_without_text_bins(tmp_path, text_tokens):
    cfg = amber_config()
    cells = [c for c in planned_cells(cfg) if c["dataset"] == "amber" and c["model"] == "qwen35_08b"]
    rows = [{"id": str(i), "dataset": "amber", "track": "amber_serving",
             "model_token_counts": {"qwen35_08b": {
                 "verified_by": "vllm_effective_token_accounting", "text_tokens": text_tokens,
                 "effective_total_tokens": 512}}} for i in range(160)]
    path = tmp_path / "annotated.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = serving_readiness(path, dataset="amber", model_id="qwen35_08b",
                               cells=cells, tolerance=.05, max_model_len=32768)
    assert len(result["ready_cells"]) == len(cells)
    assert result["blocked_cells"] == 0


def test_historical_prepare_preserves_originals_and_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(source_mapping, "DRIVE", tmp_path)
    model = tmp_path / "original-model"
    data = tmp_path / "original-dataset"
    model.mkdir()
    (data / "images").mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_bytes(b"frozen-source")
    (data / "images/one.jpg").write_bytes(b"image")
    (data / "queries.json").write_text(json.dumps([{
        "id": 1, "task": "generative", "image": "one.jpg", "query": "Describe.",
        "objects": ["dog"]}]))
    mapping = {"model": "qwen35_08b", "dataset": "amber", "model_root": str(model),
               "dataset_root": str(data), "query_path": str(data / "queries.json"),
               "images_root": str(data / "images"), "model_revision": "a" * 40,
               "dataset_revision": "b" * 40,
               "model_files": {p.name: sha256(p) for p in model.iterdir()},
               "dataset_files": {str(p.relative_to(data)): sha256(p)
                                 for p in data.rglob("*") if p.is_file()}}
    output = tmp_path / "independent-output"
    output.mkdir()
    source_map = output / "sources.json"
    source_map.write_text(json.dumps(mapping))
    p = {"root": output, "model_root": model, "dataset_root": data,
         "native": output / "native.jsonl", "candidates": output / "serving.jsonl",
         "annotated": output / "annotated.jsonl", "manifest": output / "manifest.json",
         "source_map": source_map, "mapping": mapping, "suffix": "all"}
    originals = {str(p): sha256(p) for root in (model, data)
                 for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(agent_pipeline, "require_cpu_for_hf_intake", lambda: None)
    monkeypatch.setattr(agent_pipeline, "paths_for", lambda job: p)
    def forbidden(*args, **kwargs):
        pytest.fail("historical reuse must not fetch sources")
    monkeypatch.setattr(agent_pipeline, "fetch_model", forbidden)
    monkeypatch.setattr(agent_pipeline, "fetch_dataset", forbidden)
    job = {"model": "qwen35_08b", "dataset": "amber", "phase": "pilot", "qa_limit": 0}
    assert agent_pipeline.prepare(job)["status"] == "CPU_PREPARED_NEEDS_GPU_TOKEN_ANNOTATION"
    assert agent_pipeline.prepare(job)["status"] == "PREPARED_REUSED_NO_NEW_DOWNLOAD"
    assert json.loads(p["native"].read_text())["track"] == "native_quality"
    serving = json.loads(p["candidates"].read_text())
    assert serving["track"] == "amber_serving"
    assert "requested_text_length_bin" not in serving
    assert all(sha256(path) == digest for path, digest in originals.items())
    (model / "config.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        agent_pipeline.prepare(job)


def test_total_deadline_includes_warmup_and_preserves_cancelled_requests(tmp_path, monkeypatch):
    rows = [{"id": str(i), "messages": [{"role": "user", "content": "Q"}]} for i in range(2)]
    monkeypatch.setattr(bench, "exact_bin_pool", lambda *a, **kw: rows)
    monkeypatch.setattr(bench, "sample_unique", lambda *a: rows)
    class Sampler:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
    monkeypatch.setattr(bench, "GPUSampler", Sampler)
    async def completion(client, url, payload, request_id):
        await asyncio.sleep(.04)
        return {"id": request_id, "status": "HTTP_OK", "output_tokens": 256, "error": None}
    monkeypatch.setattr(bench, "stream_completion", completion)
    cfg = {"performance": {"output_tokens": 256, "max_bin_relative_error": .05,
                           "request_timeout_s": 1, "cell_timeout_s": .06},
           "serving": {"port": 8000, "max_model_len": 32768}}
    cell = {"model": "qwen35_08b", "dataset": "memlens", "mode": "base",
            "repetition": 0, "input_length": 128, "output_tokens": 256,
            "concurrency": 1, "requests": 1, "warmup": 1, "seed": 42}
    result = asyncio.run(bench.run_performance_cell(
        cfg, cell, rows, tmp_path / "local", tmp_path / "drive", "qwen35_08b", "sha"))
    assert result["status"] == "TIMEOUT"
    evidence = next((tmp_path / "local/requests").glob("*.jsonl")).read_text()
    receipts = [json.loads(x) for x in evidence.splitlines()]
    assert any(r["phase"] == "warmup" and r["status"] == "HTTP_OK" for r in receipts)
    assert any(r["phase"] == "measurement" and r["status"] == "CANCELLED" for r in receipts)


@pytest.mark.parametrize("unsafe", [False, True])
def test_pinned_amber_archive_stages_without_changing_original(tmp_path, monkeypatch, unsafe):
    monkeypatch.setattr(source_mapping, "DRIVE", tmp_path)
    archive = tmp_path / "original/amber/_archives/AMBER.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape" if unsafe else "data/query/query_all.json", "[]")
    digest = sha256(archive)
    outputs = tmp_path / "independent-output"
    destination = outputs / "source_cache/amber" / digest
    data = {"dataset": "amber", "archive_path": str(archive),
            "archive_sha256": digest, "dataset_root": str(destination)}
    if unsafe:
        with pytest.raises(ValueError, match="Unsafe"):
            source_mapping.stage_amber_archive(data, outputs)
        assert not destination.exists()
    else:
        source_mapping.stage_amber_archive(data, outputs)
        assert (destination / "data/query/query_all.json").read_text() == "[]"
        source_mapping.stage_amber_archive(data, outputs)
    assert sha256(archive) == digest
    assert not (archive.parent / "data").exists()


@pytest.mark.parametrize("ccu", [1, 2, 4, 8, 16, 32])
def test_warmup_and_measurement_offer_parallelism_without_overlap(tmp_path, monkeypatch, ccu):
    rows = [{"id": str(i), "messages": [{"role": "user", "content": "Q"}]}
            for i in range(2 * ccu)]
    monkeypatch.setattr(bench, "exact_bin_pool", lambda *a, **kw: rows)
    monkeypatch.setattr(bench, "sample_unique", lambda *a: rows)
    class Sampler:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def report(self):
            return dict.fromkeys(("peak_total_gpu_used_mib", "minimum_available_system_ram_gib",
                                  "minimum_free_local_ssd_gib", "peak_system_ram_used_gib"), 0)
    monkeypatch.setattr(bench, "GPUSampler", Sampler)
    active = peak = warm_completed = 0
    async def completion(client, url, payload, request_id):
        nonlocal active, peak, warm_completed
        if int(request_id) >= ccu:
            assert warm_completed == ccu
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        if int(request_id) < ccu:
            warm_completed += 1
        return {"id": request_id, "status": "HTTP_OK", "output_tokens": 256, "error": None,
                "ttft_ms": 1, "tpot_ms": 1, "e2e_ms": 1, "input_tokens_reported": 128}
    monkeypatch.setattr(bench, "stream_completion", completion)
    cfg = {"performance": {"output_tokens": 256, "max_bin_relative_error": .05,
                           "request_timeout_s": 1, "cell_timeout_s": 5},
           "serving": {"port": 8000, "max_model_len": 32768}}
    cell = {"model": "qwen35_08b", "dataset": "memlens", "mode": "base",
            "repetition": 0, "input_length": 128, "output_tokens": 256,
            "concurrency": ccu, "requests": ccu, "warmup": ccu, "seed": 42}
    result = asyncio.run(bench.run_performance_cell(
        cfg, cell, rows, tmp_path / "local", tmp_path / "drive", "qwen35_08b", "sha"))
    assert result["status"] == "PASS_MEASURED"
    assert result["successful_requests"] == ccu
    assert peak == ccu and active == 0
