#!/usr/bin/env python3
"""Bounded PyTorch-reference smoke on the local GTX 1080 Ti (ITER2 D5-3).

Runs the official-hooked RealSevenArmEngine for arms `base` and `h2o_base` on the Qwen3.5-0.8B
checkpoint: fp16 on CUDA (sm_61 has no native bf16), against an fp32 CPU reference for the same
inputs. Two short text prompts and the canonical image fixture, 8 greedy tokens each.

Limits, enforced in code: abort if free GPU memory is below 3 GiB before any load or request;
wall-clock limit 600 s for the whole run. The GPU is shared with another process that uses about
6 GB; this script never touches that process.

Label: pytorch_reference_gtx1080ti_smoke_not_vllm_not_performance. The receipt makes no TTFT,
throughput or vLLM claim. Token agreement is reported as measured, including divergence.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))  # must precede the vlmbench imports below

LABEL = "pytorch_reference_gtx1080ti_smoke_not_vllm_not_performance"
MIN_FREE_MIB = 3072
WALL_LIMIT_S = 600
ARMS = ("base", "h2o_base")
MAX_NEW_TOKENS = 8
CHECKPOINT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "000000039769.png"
TEXT_PROMPTS = (
    "Answer in one word: what colour is an emerald?",
    "Complete the sentence: The capital of France is",
)
IMAGE_QUESTION = "Are there cats lying on the couch in the living room?"


class SmokeAbort(RuntimeError):
    """A bound was hit. The run stops and the receipt records why."""


# --- pure helpers (unit-tested) ------------------------------------------------------

def parse_smi_csv(line: str) -> dict[str, Any]:
    name, uuid, free, total = [part.strip() for part in line.split(",")]
    return {"name": name, "uuid": uuid, "free_mib": int(free), "total_mib": int(total)}


def check_free_memory(free_mib: int, min_free: int = MIN_FREE_MIB) -> None:
    if free_mib < min_free:
        raise SmokeAbort(f"free GPU memory {free_mib} MiB is below {min_free} MiB; not loading")


def compare_tokens(a: list[int], b: list[int]) -> dict[str, Any]:
    n = min(len(a), len(b))
    prefix = 0
    while prefix < n and a[prefix] == b[prefix]:
        prefix += 1
    identical = len(a) == len(b) and prefix == n
    return {
        "identical": identical,
        "first_divergence": None if identical else prefix,
        "matching_prefix_len": prefix,
        "len_a": len(a),
        "len_b": len(b),
    }


# --- run mechanics -----------------------------------------------------------------

def gpu_snapshot() -> dict[str, Any]:
    out = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=name,uuid,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip().splitlines()[0]
    return parse_smi_csv(out)


def git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _guard(deadline: float, device: str) -> None:
    if time.monotonic() > deadline:
        raise SmokeAbort("wall-clock limit reached")
    if device.startswith("cuda"):
        check_free_memory(gpu_snapshot()["free_mib"])


def _record(name: str, prompt: str, res: Any) -> dict[str, Any]:
    tokens = [int(t) for t in res.tokens_generated]
    return {
        "input": name,
        "prompt": prompt,
        "tokens": tokens,
        "tokens_generated_count": len(tokens),
        "output_text": res.output_text,
    }


def run_arm(device: str, arm: str, dtype_override: str | None, deadline: float) -> list[dict[str, Any]]:
    import torch

    from vlmbench.infrastructure.real_engine import RealSevenArmEngine
    from vlmbench.multimodal_fixture import CANONICAL_IMAGE_SHA256

    import hashlib

    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != CANONICAL_IMAGE_SHA256:
        raise SmokeAbort("canonical image fixture hash mismatch")

    engine = RealSevenArmEngine(
        method_arm=arm, device=device, checkpoint_dir=CHECKPOINT, dtype_override=dtype_override,
    )
    results: list[dict[str, Any]] = []
    try:
        if not (engine.is_pretrained and engine.is_official_hf_hooked):
            raise SmokeAbort(f"{arm}: official hooked checkpoint did not load")
        for i, prompt in enumerate(TEXT_PROMPTS):
            _guard(deadline, device)
            messages = [{"role": "user", "content": prompt}]
            text = engine.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = engine.processor(text=[text], return_tensors="pt")["input_ids"][0].tolist()
            res = engine.execute_request(
                request_id=f"smoke_{arm}_text{i}", prompt_token_ids=ids,
                max_new_tokens=MAX_NEW_TOKENS, image_present=False,
            )
            results.append(_record(f"text{i}", prompt, res))
        _guard(deadline, device)
        res = engine.execute_request(
            request_id=f"smoke_{arm}_image", max_new_tokens=MAX_NEW_TOKENS, image_present=True,
            image_path=FIXTURE, image_bytes=FIXTURE.read_bytes(), question=IMAGE_QUESTION,
        )
        results.append(_record("image", IMAGE_QUESTION, res))
    finally:
        engine.close()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return results


def build_receipt() -> dict[str, Any]:
    """The child's work: the bounded smoke. Runs only inside the child process."""
    import torch
    import transformers

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; this smoke is GPU-specific")

    t0 = time.monotonic()
    deadline = t0 + WALL_LIMIT_S
    cap = torch.cuda.get_device_capability(0)
    receipt: dict[str, Any] = {
        "label": LABEL,
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "checkpoint": str(CHECKPOINT),
        "hardware": {**gpu_snapshot(), "compute_capability": f"{cap[0]}.{cap[1]}"},
        "gpu_dtype": "float16",
        "reference_dtype": "float32 (CPU)",
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_free_mib": MIN_FREE_MIB,
        "wall_limit_s": WALL_LIMIT_S,
        "arms": list(ARMS),
        "text_prompts": list(TEXT_PROMPTS),
        "image_question": IMAGE_QUESTION,
        "free_memory_checks": [],
        "gpu": {},
        "cpu_reference": {},
        "comparison": {},
        "status": None,
    }

    try:
        for arm in ARMS:
            snap = gpu_snapshot()
            receipt["free_memory_checks"].append({"before_arm": arm, "free_mib": snap["free_mib"]})
            check_free_memory(snap["free_mib"])
            receipt["gpu"][arm] = run_arm("cuda", arm, "fp16", deadline)
        for arm in ARMS:
            receipt["cpu_reference"][arm] = run_arm("cpu", arm, None, deadline)
        identical_all = True
        for arm in ARMS:
            per_input = []
            for g, c in zip(receipt["gpu"][arm], receipt["cpu_reference"][arm]):
                cmp = compare_tokens(g["tokens"], c["tokens"])
                identical_all &= cmp["identical"]
                per_input.append({"input": g["input"], **cmp})
            receipt["comparison"][arm] = per_input
        complete = all(
            r["tokens_generated_count"] == MAX_NEW_TOKENS
            for arm in ARMS for r in receipt["gpu"][arm] + receipt["cpu_reference"][arm]
        )
        receipt["status"] = "RUN_COMPLETED" if complete else "RUN_INCOMPLETE"
        receipt["token_agreement"] = "IDENTICAL_ALL_INPUTS" if identical_all else "DIVERGENCE_PRESENT_SEE_COMPARISON"
    except SmokeAbort as exc:
        receipt["status"] = f"ABORTED: {exc}"
    except Exception as exc:  # recorded honestly; the run exits non-zero below
        receipt["status"] = f"FAILED: {type(exc).__name__}: {exc}"

    receipt["elapsed_wall_s"] = round(time.monotonic() - t0, 1)
    return receipt


def child_main(result_path: Path) -> int:
    receipt = build_receipt()
    result_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf8")
    return 0


def build_child_cmd(result_path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--child", "--result", str(result_path)]


def run_outer(cmd: list[str], timeout_s: float) -> dict[str, Any]:
    """Run cmd in its own session; at the deadline kill the whole session, grandchildren included.

    The parent enforces the bound, so a hung CUDA call inside the child cannot defeat it.
    """
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, start_new_session=True)
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return {"outcome": "OUTER_DEADLINE", "returncode": None, "elapsed_s": round(time.monotonic() - t0, 1)}
    return {"outcome": "EXITED", "returncode": rc, "elapsed_s": round(time.monotonic() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "runs")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child_main(args.result)

    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        run = run_outer(build_child_cmd(result_path), WALL_LIMIT_S)
        if run["outcome"] == "OUTER_DEADLINE":
            print(f"ABORTED: outer hard deadline {WALL_LIMIT_S}s reached; child session killed; no receipt written",
                  flush=True)
            return 2
        if not result_path.exists():
            print(f"FAILED: child exited {run['returncode']} without a result; no receipt written", flush=True)
            return 1
        receipt = json.loads(result_path.read_text(encoding="utf8"))

    if receipt["status"] != "RUN_COMPLETED":
        print(f"status={receipt['status']}; not a completed run, so no receipt written", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = args.out_dir / f"{stamp}_pytorch_reference_gtx1080ti_smoke.json"
    with open(out, "x", encoding="utf8") as f:  # exclusive create: never replace a receipt
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    print(f"status={receipt['status']} token_agreement={receipt.get('token_agreement')} "
          f"elapsed={receipt['elapsed_wall_s']}s receipt={out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
