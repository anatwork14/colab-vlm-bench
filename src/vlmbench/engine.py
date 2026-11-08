"""A100-specific backend launch commands and environment checks."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path


def a100_inventory() -> dict:
    p = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    lines = [x.strip() for x in p.stdout.splitlines() if x.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one GPU, found {len(lines)}")
    items = [x.strip() for x in lines[0].split(",")]
    if len(items) != 4:
        raise RuntimeError("Cannot parse nvidia-smi A100 inventory")
    mib = int(float(items[1]))
    if "A100" not in items[0] or not 39000 <= mib <= 45000:
        raise RuntimeError(f"Exact A100 40GB required; got {items[0]}, {mib}MiB")
    return {"gpu_name": items[0], "vram_total_mib": mib,
            "driver_version": items[2], "compute_capability": items[3]}


def server_argv(config: dict, model: dict, mode: str,
                target_path: str, assistant_path: str | None = None) -> list[str]:
    if mode not in {"base", "mtp"}:
        raise ValueError("Unknown inference mode")
    h, s = config["hardware"], config["serving"]
    args = [
        "vllm", "serve", str(target_path),
        "--served-model-name", model["id"],
        "--host", "127.0.0.1", "--port", str(s["port"]),
        "--dtype", h["dtype"],
        "--tensor-parallel-size", str(h["tensor_parallel_size"]),
        "--max-model-len", str(s["max_model_len"]),
        "--max-num-seqs", str(s["max_num_seqs"]),
        "--max-num-batched-tokens", str(s["max_num_batched_tokens"]),
        "--gpu-memory-utilization", str(h["gpu_memory_utilization"]),
        "--limit-mm-per-prompt", json.dumps({"image": s["max_images_per_prompt"]}),
        "--no-enable-prefix-caching",
    ]
    if mode == "mtp":
        speculative = {"method": "mtp", "num_speculative_tokens": model["mtp_tokens"]}
        if model["id"] == "gemma4_e4b":
            if not assistant_path:
                raise ValueError("Gemma4 MTP requires a verified matching assistant")
            speculative["model"] = assistant_path
        args += ["--speculative-config", json.dumps(speculative)]
    return args


def start_server(argv: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    logfile = log.open("a", encoding="utf8")
    try:
        p = subprocess.Popen(argv, stdout=logfile, stderr=subprocess.STDOUT,
                             start_new_session=True)
    finally:
        logfile.close()
    return p


def wait_ready(proc: subprocess.Popen, port: int, timeout_s: int = 360) -> None:
    end = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM terminated during boot with exit status {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError("vLLM readiness timeout")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
