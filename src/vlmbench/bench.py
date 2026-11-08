"""Real async streaming serving measurement, paired workloads, and native prediction replay.

Measured requests with missing or mismatched output token counts are NOT PASS.
"""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx

from .artifacts import atomic_json, atomic_jsonl, sha256, sync_files, verified_resume
from .config import cell_id, fingerprint
from .telemetry import GPUSampler
from .workloads import (WorkloadError, exact_bin_pool, resolve_messages,
                        sample_digest, sample_unique, ensure_no_gold_in_request)


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    pos = (len(arr) - 1) * q
    a, b = int(math.floor(pos)), int(math.ceil(pos))
    return arr[a] + (arr[b] - arr[a]) * (pos - a)


async def stream_completion(client: httpx.AsyncClient, url: str, request: dict, request_id: str) -> dict:
    start = perf_counter()
    first_token_at = None
    emitted_chunks = 0
    output_text = []
    usage = {}
    ended = None
    status = "ERROR"
    error = None
    try:
        async with client.stream("POST", url, json=request) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                if not payload:
                    continue
                obj = json.loads(payload)
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                if obj.get("error"):
                    raise ValueError(str(obj["error"]))
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = delta.get("content", "")
                    if isinstance(text, str) and text:
                        if first_token_at is None:
                            first_token_at = perf_counter()
                        emitted_chunks += 1
                        output_text.append(text)
            ended = perf_counter()
            status = "HTTP_OK"
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        ended = perf_counter()
    output_tokens = usage.get("completion_tokens")
    return {
        "id": request_id, "status": status, "error": error,
        "ttft_ms": (first_token_at - start) * 1000 if first_token_at is not None else None,
        "e2e_ms": (ended - start) * 1000,
        "tpot_ms": (
            (ended - first_token_at) * 1000 / (output_tokens - 1)
            if first_token_at is not None and isinstance(output_tokens, int) and output_tokens > 1
            else None
        ),
        "stream_content_chunks": emitted_chunks,
        "output_tokens": output_tokens,
        "input_tokens_reported": usage.get("prompt_tokens"),
        "output_text_sha256": fingerprint("".join(output_text)),
        "note": "stream chunks != individual generated tokens",
    }


def request_payload(sample: dict, *, model_id: str, max_tokens: int,
                    streaming: bool, fixed_output: bool) -> dict:
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": resolve_messages(sample["messages"]),
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": streaming,
    }
    # Qwen3.5: disable thinking identically for Base and speculative arms.
    if model_id.startswith("qwen35_"):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if fixed_output:
        payload.update({"ignore_eos": True, "min_tokens": max_tokens})
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    ensure_no_gold_in_request(sample, payload)
    return payload


async def run_performance_cell(
    cfg: dict, cell: dict, rows: list[dict], root: Path, drive_root: Path,
    model_id: str, dataset_sha256: str,
) -> dict:
    """A complete cell includes unique warmups + measured requests, with a hard deadline."""
    p, s = cfg["performance"], cfg["serving"]
    out = root / "cells" / f"{cell_id(cell)}.json"
    expected = {
        "config_sha256": fingerprint(cfg), "dataset_sha256": dataset_sha256, "cell": cell,
    }
    if verified_resume(out, expected):
        return {"status": "RESUMED_PASS_MEASURED", "cell": cell}

    # Native tasks and synthetic reference snippets must not be assigned invented token counts.
    pool = exact_bin_pool(rows, model_id, cell["input_length"], p["max_bin_relative_error"],
                          max_effective_total=s["max_model_len"] - cell["output_tokens"])
    total = cell["requests"] + cell["warmup"]
    try:
        chosen = sample_unique(pool, total, cell["seed"])
    except WorkloadError as exc:
        receipt = {**expected, "status": "INSUFFICIENT_WORKLOAD", "error": str(exc),
                   "eligible_prompts": len(pool), "required_unique_prompts": total}
        atomic_json(out, receipt)
        sync_files(root, drive_root)
        return receipt

    # Warmup samples are distinct from measured requests and never included in their metrics.
    warmups = chosen[:cell["warmup"]]
    measured = chosen[cell["warmup"]:]
    url = f"http://127.0.0.1:{s['port']}/v1/chat/completions"
    timeout = httpx.Timeout(timeout=float(p["request_timeout_s"]), connect=30.0)
    semaphore = asyncio.Semaphore(cell["concurrency"])
    failures: list[dict] = []
    start_cell = perf_counter()
    observations: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def one(sample):
                async with semaphore:
                    payload = request_payload(sample, model_id=model_id,
                                              max_tokens=cell["output_tokens"],
                                              streaming=True, fixed_output=True)
                    return await stream_completion(client, url, payload, str(sample["id"]))
            # Warm up the actual offered parallelism. Thirty-two sequential
            # 16K requests are slow, costly, and not a CCU32 warmup.
            warm_observations = await asyncio.gather(*(one(w) for w in warmups))
            for warm in warm_observations:
                if warm["status"] != "HTTP_OK" or warm["output_tokens"] != cell["output_tokens"]:
                    raise WorkloadError(f"WARMUP_INVALID: {warm['error']} / {warm['output_tokens']}")
            start_cell = perf_counter()
            with GPUSampler(0.5) as gpu:
                observations = await asyncio.wait_for(
                    asyncio.gather(*(one(sample) for sample in measured)),
                    timeout=float(p["cell_timeout_s"]),
                )
            gpu_report = gpu.report()
    except (WorkloadError, asyncio.TimeoutError, httpx.HTTPError, ValueError) as exc:
        receipt = {**expected, "status": "TIMEOUT" if isinstance(exc, asyncio.TimeoutError)
                   else "ERROR", "error": str(exc),
                   "warmup": cell["warmup"], "expected_requests": cell["requests"]}
        atomic_json(out, receipt)
        sync_files(root, drive_root)
        return receipt

    total_time = max(perf_counter() - start_cell, 1e-9)
    valid = [
        x for x in observations
        if x["status"] == "HTTP_OK" and x["output_tokens"] == cell["output_tokens"]
    ]
    failures = [x for x in observations if x not in valid]
    metric = {
        "ttft_p50_ms": quantile([x["ttft_ms"] for x in valid if x["ttft_ms"] is not None], 0.50),
        "ttft_p95_ms": quantile([x["ttft_ms"] for x in valid if x["ttft_ms"] is not None], 0.95),
        "ttft_p99_ms": quantile([x["ttft_ms"] for x in valid if x["ttft_ms"] is not None], 0.99),
        "tpot_p50_ms": quantile([x["tpot_ms"] for x in valid if x["tpot_ms"] is not None], 0.50),
        "tpot_p95_ms": quantile([x["tpot_ms"] for x in valid if x["tpot_ms"] is not None], 0.95),
        "e2e_p50_ms": quantile([x["e2e_ms"] for x in valid], 0.50),
        "e2e_p95_ms": quantile([x["e2e_ms"] for x in valid], 0.95),
        "e2e_p99_ms": quantile([x["e2e_ms"] for x in valid], 0.99),
        "actual_effective_prompt_tokens_p50": quantile(
            [x["input_tokens_reported"] for x in valid
             if isinstance(x["input_tokens_reported"], int)], 0.50),
        "actual_effective_prompt_tokens_p95": quantile(
            [x["input_tokens_reported"] for x in valid
             if isinstance(x["input_tokens_reported"], int)], 0.95),
        "output_tokens_per_s": sum(x["output_tokens"] for x in valid) / total_time,
        "requests_per_s": len(valid) / total_time,
        "cell_elapsed_s": total_time,
        "peak_total_gpu_used_mib": gpu_report["peak_total_gpu_used_mib"],
        "minimum_available_system_ram_gib": gpu_report["minimum_available_system_ram_gib"],
        "minimum_free_local_ssd_gib": gpu_report["minimum_free_local_ssd_gib"],
        "peak_system_ram_used_gib": gpu_report["peak_system_ram_used_gib"],
        "physical_kv_bytes": "NOT_MEASURED",
        "mtp_acceptance_rate": "NOT_MEASURED",
        "queue_wait_ms": "NOT_MEASURED",
        "stream_inter_token_latency": "NOT_MEASURED",
    }
    status = "PASS_MEASURED" if len(valid) == cell["requests"] and metric["ttft_p95_ms"] is not None else "PARTIAL_FAILURE"
    receipt = {
        **expected, "status": status, "method": cell["mode"], "offered_ccu": cell["concurrency"],
        "warmup": cell["warmup"], "expected_requests": cell["requests"],
        "text_token_bin_requested": cell["input_length"],
        "successful_requests": len(valid), "failed_requests": len(failures),
        "sample_id_sha256": sample_digest(measured),
        "all_sample_ids": [str(x["id"]) for x in measured],
        "metrics": metric,
        "notes": [
            "Fixed length request count checked against per-request streaming usage.completion_tokens",
            "TTFT based on first streamed content chunk; TPOT derived from stream interval/output token count",
            "nvidia-smi total MiB is not KV resident memory",
        ],
    }
    atomic_jsonl(root / "requests" / f"{cell_id(cell)}.jsonl", observations)
    atomic_json(out, receipt)
    sync_files(root, drive_root)
    return receipt


async def run_quality_predictions(cfg: dict, model_id: str, mode: str,
                                  dataset: str, rows: list[dict],
                                  root: Path, drive_root: Path) -> dict:
    """Unmodified full-context predictions. Never calculate an official score here."""
    s = cfg["serving"]
    output = root / "quality" / f"{dataset}__{model_id}__{mode}.jsonl"
    receipt_path = root / "quality" / f"{dataset}__{model_id}__{mode}.receipt.json"
    if receipt_path.exists():
        completed = json.loads(receipt_path.read_text(encoding="utf8"))
        if completed.get("count") == len(rows) and output.exists():
            return {**completed, "resume_status": "RESUMED_COMPLETE"}
        raise ValueError("Quality receipt inconsistent; preserve it and audit before retry")
    results = []
    if output.exists():
        results = [json.loads(line) for line in output.read_text(encoding="utf8").splitlines() if line.strip()]
        expected_ids = [str(r["id"]) for r in rows]
        actual_ids = [str(r["id"]) for r in results]
        if len(actual_ids) != len(set(actual_ids)) or actual_ids != expected_ids[:len(actual_ids)]:
            raise ValueError("Quality resume order or IDs conflict with frozen native dataset")
    url = f"http://127.0.0.1:{s['port']}/v1/chat/completions"
    consecutive_failures = 0
    aborted_quality = False
    async with httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=30.0)) as client:
        for row in rows[len(results):]:
            try:
                payload = request_payload(row, model_id=model_id,
                                          max_tokens=cfg["quality"]["max_output_tokens"],
                                          streaming=False, fixed_output=False)
                start = perf_counter()
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                pred = data["choices"][0]["message"].get("content", "")
                results.append({"id": row["id"], "status": "SUCCESS",
                                "prediction": pred, "usage": data.get("usage", {}),
                                "e2e_ms": (perf_counter() - start) * 1000})
                consecutive_failures = 0
            except Exception as exc:
                results.append({"id": row["id"], "status": "FAILED",
                                "error": f"{type(exc).__name__}: {exc}", "prediction": None})
                consecutive_failures += 1
            # Avoid losing all quality predictions if Colab disconnects.
            atomic_jsonl(output, results)
            sync_files(root / "quality", drive_root / "quality")
            # Protect A100 budget: two consecutive native long-context
            # inference failures stop the engineering pilot.
            if (cfg.get("execution_scope", {}).get("kind") == "engineering_pilot"
                    and consecutive_failures >= 2):
                aborted_quality = True
                print("Native quality pilot stopped after 2 consecutive failures; "
                      "inspect receipts before retrying.", flush=True)
                break
    receipt = {
        "dataset": dataset, "model": model_id, "mode": mode,
        "status": ("QUALITY_STOPPED_AFTER_REPEATED_FAILURES" if aborted_quality
                   else "QUALITY_PREDICTIONS_UNSCORED"),
        "count": len(results),
        "expected_count": len(rows),
        "circuit_breaker_after_consecutive_failures": 2 if aborted_quality else None,
        "sample_id_sha256": sample_digest(rows),
        "sample_ids": [str(row["id"]) for row in rows],
        "evaluation_scope": "FROZEN_NATIVE_SUBSET_OR_FULL_UNSCORED",
        "failures": sum(r["status"] == "FAILED" for r in results),
        "official_accuracy": "NOT_MEASURED",
        "prediction_sha256": sha256(output),
        "note": "Official scorer must independently evaluate these complete-context predictions",
    }
    atomic_json(receipt_path, receipt)
    sync_files(root, drive_root)
    return receipt
