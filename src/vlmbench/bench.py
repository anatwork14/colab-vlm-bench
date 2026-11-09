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

from .artifacts import atomic_json, atomic_jsonl, code_identity, sha256, sync_files, verified_resume
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


async def admitted_stream(client: httpx.AsyncClient, url: str, request: dict, request_id: str,
                          semaphore: asyncio.Semaphore, submitted_at: float) -> dict:
    """Wait for a CCU slot, then stream. E2E counts from submission, not from the slot.

    A user who submitted at ``submitted_at`` experiences the client-side slot wait, so
    it is added to e2e. TTFT stays HTTP-send based (docs/METRICS.md). Before this fix
    the slot wait was invisible: a CCU=1 second request reported only its service time.
    """
    async with semaphore:
        acquired_at = perf_counter()
        result = await stream_completion(client, url, request, request_id)
    queue_ms = max(0.0, (acquired_at - submitted_at) * 1000)
    result["client_queue_wait_ms"] = queue_ms
    if isinstance(result.get("e2e_ms"), (int, float)):
        result["e2e_ms"] = result["e2e_ms"] + queue_ms
    return result


async def stream_completion(client: httpx.AsyncClient, url: str, request: dict, request_id: str) -> dict:
    start = perf_counter()  # HTTP send: service-side timing origin
    first_token_at = None
    emitted_chunks = 0
    output_text = []
    usage = {}
    ended = None
    status = "ERROR"
    error = None
    terminal_seen = False
    finish_reason = None
    try:
        async with client.stream("POST", url, json=request) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    terminal_seen = True
                    break
                if not payload:
                    continue
                obj = json.loads(payload)
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                if obj.get("error"):
                    raise ValueError(str(obj["error"]))
                for choice in obj.get("choices", []):
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                        terminal_seen = True
                    delta = choice.get("delta") or {}
                    text = delta.get("content", "")
                    if isinstance(text, str) and text:
                        if first_token_at is None:
                            first_token_at = perf_counter()
                        emitted_chunks += 1
                        output_text.append(text)
            ended = perf_counter()
            if not terminal_seen:
                raise ValueError("INCOMPLETE_STREAM: no terminal DONE or finish_reason")
            status = "HTTP_OK"
    except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        ended = perf_counter()
    output_tokens = usage.get("completion_tokens")
    return {
        "id": request_id, "status": status, "error": error,
        "terminal_seen": terminal_seen, "finish_reason": finish_reason,
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


def eligible_pool(cfg: dict, cell: dict, rows: list[dict], model_id: str) -> list[dict]:
    """Rows eligible for this cell's text bin under the verified-token policy."""
    p, s = cfg["performance"], cfg["serving"]
    # Native tasks and synthetic reference snippets must not be assigned invented token counts.
    if cell["dataset"] == "amber":
        pool = []
        for row in rows:
            stat = row.get("model_token_counts", {}).get(model_id, {})
            total_tokens = stat.get("effective_total_tokens")
            if (row.get("track") == "amber_serving"
                    and stat.get("verified_by") == "vllm_effective_token_accounting"
                    and type(total_tokens) is int
                    and 0 < total_tokens < s["max_model_len"] - cell["output_tokens"]):
                pool.append(row)
        return pool
    return exact_bin_pool(rows, model_id, cell["input_length"], p["max_bin_relative_error"],
                          max_effective_total=s["max_model_len"] - cell["output_tokens"])


def select_cell_samples(pool: list[dict], cell: dict) -> tuple[list[dict], list[dict]]:
    """Deterministic (warmup, measured) prompts for one cell from its eligible pool.

    Shared by the runner and by the resume check, so a receipt's sample_id_sha256 can be
    verified against the current derived dataset. Raises WorkloadError if the pool is short.
    """
    total = cell["requests"] + cell["warmup"]
    chosen = sample_unique(pool, total, cell["seed"])
    # Warmup samples are distinct from measured requests and never included in their metrics.
    return chosen[:cell["warmup"]], chosen[cell["warmup"]:]


async def run_performance_cell(
    cfg: dict, cell: dict, rows: list[dict], root: Path, drive_root: Path,
    model_id: str, dataset_sha256: str,
) -> dict:
    """A complete cell includes unique warmups + measured requests, with a hard deadline."""
    p, s = cfg["performance"], cfg["serving"]
    if p["output_tokens"] != 256 or cell["output_tokens"] != 256:
        raise ValueError("Fixed serving pass requires exactly 256 output tokens")
    out = root / "cells" / f"{cell_id(cell)}.json"
    expected = {
        "config_sha256": fingerprint(cfg), "dataset_sha256": dataset_sha256,
        "code_sha256": code_identity(), "cell": cell,
        "execution_pass": "FIXED_OUTPUT_SERVING_256",
        "output_contract": {"max_tokens": 256, "min_tokens": 256, "ignore_eos": True},
    }
    if verified_resume(out, expected):
        return {"status": "RESUMED_PASS_MEASURED", "cell": cell}

    pool = eligible_pool(cfg, cell, rows, model_id)
    try:
        warmups, measured = select_cell_samples(pool, cell)
    except WorkloadError as exc:
        receipt = {**expected, "status": "INSUFFICIENT_WORKLOAD", "error": str(exc),
                   "eligible_prompts": len(pool),
                   "required_unique_prompts": cell["requests"] + cell["warmup"]}
        atomic_json(out, receipt)
        sync_files(root, drive_root)
        return receipt

    url = f"http://127.0.0.1:{s['port']}/v1/chat/completions"
    timeout = httpx.Timeout(timeout=float(p["request_timeout_s"]), connect=30.0)
    semaphore = asyncio.Semaphore(cell["concurrency"])
    failures: list[dict] = []
    start_cell = perf_counter()
    observations: list[dict] = []
    request_evidence: list[dict] = []
    gpu_report = {}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async def one(sample, phase):
                submitted_at = perf_counter()
                try:
                    payload = request_payload(sample, model_id=model_id,
                                              max_tokens=cell["output_tokens"],
                                              streaming=True, fixed_output=True)
                    result = await admitted_stream(client, url, payload, str(sample["id"]),
                                                   semaphore, submitted_at)
                except asyncio.CancelledError:
                    request_evidence.append({"id": str(sample["id"]), "phase": phase,
                                             "status": "CANCELLED", "output_tokens": None})
                    raise
                except (WorkloadError, OSError, ValueError, httpx.HTTPError) as exc:
                    result = {"id": str(sample["id"]), "status": "ERROR", "output_tokens": None,
                              "client_queue_wait_ms": None,
                              "error": f"{type(exc).__name__}: {exc}"}
                request_evidence.append({**result, "phase": phase})
                return result
            async def phases():
                nonlocal observations, start_cell, gpu_report
                active_tasks = [asyncio.create_task(one(w, "warmup")) for w in warmups]
                try:
                    warm_observations = await asyncio.gather(*active_tasks)
                    for warm in warm_observations:
                        if warm["status"] != "HTTP_OK" or warm["output_tokens"] != cell["output_tokens"]:
                            raise WorkloadError(f"WARMUP_INVALID: {warm['error']} / {warm['output_tokens']}")
                    start_cell = perf_counter()
                    with GPUSampler(0.5) as gpu:
                        active_tasks = [asyncio.create_task(one(sample, "measurement")) for sample in measured]
                        observations = await asyncio.gather(*active_tasks)
                    gpu_report = gpu.report()
                finally:
                    for task in active_tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*active_tasks, return_exceptions=True)
            await asyncio.wait_for(phases(), timeout=float(p["cell_timeout_s"]))
    except (WorkloadError, asyncio.TimeoutError, httpx.HTTPError, ValueError) as exc:
        receipt = {**expected, "status": "TIMEOUT" if isinstance(exc, asyncio.TimeoutError)
                   else "ERROR", "error": str(exc),
                   "warmup": cell["warmup"], "expected_requests": cell["requests"]}
        atomic_jsonl(root / "requests" / f"{cell_id(cell)}.jsonl", request_evidence)
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
        # Client-side wait for a CCU slot before the HTTP send (included in e2e).
        "client_queue_wait_p50_ms": quantile(
            [x["client_queue_wait_ms"] for x in valid if x.get("client_queue_wait_ms") is not None], 0.50),
        "client_queue_wait_p95_ms": quantile(
            [x["client_queue_wait_ms"] for x in valid if x.get("client_queue_wait_ms") is not None], 0.95),
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
        # Telemetry trust fields: a non-zero failure count means the GPU peak is not reliable.
        "gpu_samples": gpu_report.get("samples"),
        "gpu_parse_failures": gpu_report.get("parse_failures"),
        "gpu_query_failures": gpu_report.get("query_failures"),
        "gpu_count_max": gpu_report.get("gpu_count_max"),
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
            "TTFT from HTTP send to first streamed content chunk; E2E from submission (includes client CCU-slot queue wait); TPOT from stream interval/output token count",
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
                choice = data["choices"][0]
                pred = choice["message"].get("content", "")
                results.append({"id": row["id"], "status": "SUCCESS",
                                "prediction": pred, "usage": data.get("usage", {}),
                                "finish_reason": choice.get("finish_reason"),
                                "e2e_ms": (perf_counter() - start) * 1000})
                consecutive_failures = 0
            except Exception as exc:
                results.append({"id": row["id"], "status": "FAILED",
                                "error": f"{type(exc).__name__}: {exc}", "prediction": None,
                                "finish_reason": None})
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
        "execution_pass": "NATURAL_EOS_QUALITY",
        "output_contract": {"max_tokens": cfg["quality"]["max_output_tokens"],
                            "ignore_eos": False},
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
