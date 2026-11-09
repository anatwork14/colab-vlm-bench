#!/usr/bin/env python3
"""Bounded PyTorch-reference smoke on the local GTX 1080 Ti (ITER2 D5-3, D8, D9).

Two separate runs, each under its own OUTER hard deadline enforced by this process:
  * `--device cuda`: arms base and h2o_base on Qwen3.5-0.8B, fp16, 600 s budget.
  * `--device cpu`:  the same arms, fp32, the short CPU reference, 300 s budget.
Each run uses two short text prompts and the canonical image fixture, at most 8 greedy tokens each.
`--compare GPU.json CPU.json` writes a comparison receipt only from two completed runs at one git SHA.

A receipt is written only when a run completes (status RUN_COMPLETED). A deadline, a refusal or an
incomplete run writes no receipt. Label: pytorch_reference_gtx1080ti_smoke_not_vllm_not_performance.
This makes no TTFT, throughput or vLLM claim. Token agreement is reported as measured, including divergence.
"""
from __future__ import annotations

import argparse
import hashlib
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
WALL_LIMIT_S = {"cuda": 600, "cpu": 300}
DTYPE = {"cuda": "float16", "cpu": "float32"}
DTYPE_OVERRIDE = {"cuda": "fp16", "cpu": None}
TAG = {"cuda": "gpu_fp16", "cpu": "cpu_fp32"}
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
    """A bound was hit. The run stops and its status says why."""


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


def compare_receipts(gpu: dict[str, Any], cpu: dict[str, Any]) -> dict[str, Any]:
    """Token comparison between two COMPLETED runs at one git SHA. Refuses anything else."""
    for name, receipt in (("gpu", gpu), ("cpu", cpu)):
        if receipt.get("status") != "RUN_COMPLETED":
            raise ValueError(f"{name} receipt is not RUN_COMPLETED: {receipt.get('status')!r}")
    if gpu["git_sha"] != cpu["git_sha"]:
        raise ValueError("receipts come from different git SHAs")
    identical_all = True
    comparison: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        per_input = []
        for g, c in zip(gpu["results"][arm], cpu["results"][arm], strict=True):
            if g["input"] != c["input"]:
                raise ValueError(f"input mismatch in {arm}: {g['input']} vs {c['input']}")
            cmp = compare_tokens(g["tokens"], c["tokens"])
            identical_all &= cmp["identical"]
            per_input.append({"input": g["input"], **cmp})
        comparison[arm] = per_input
    return {
        "token_agreement": "IDENTICAL_ALL_INPUTS" if identical_all else "DIVERGENCE_PRESENT_SEE_COMPARISON",
        "comparison": comparison,
        "gpu_dtype": gpu["dtype"],
        "reference_dtype": cpu["dtype"],
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
        raise SmokeAbort("wall-clock limit reached before a request")
    if device == "cuda":
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


def run_arm(device: str, arm: str, deadline: float) -> list[dict[str, Any]]:
    import torch

    from vlmbench.infrastructure.real_engine import RealSevenArmEngine
    from vlmbench.multimodal_fixture import CANONICAL_IMAGE_SHA256

    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != CANONICAL_IMAGE_SHA256:
        raise SmokeAbort("canonical image fixture hash mismatch")

    engine = RealSevenArmEngine(
        method_arm=arm, device=device, checkpoint_dir=CHECKPOINT, dtype_override=DTYPE_OVERRIDE[device],
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
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def build_receipt(device: str) -> dict[str, Any]:
    """The child's work: one device's bounded run. Runs only inside the child process."""
    import torch
    import transformers

    t0 = time.monotonic()
    deadline = t0 + WALL_LIMIT_S[device]
    receipt: dict[str, Any] = {
        "label": LABEL,
        "device": device,
        "tag": TAG[device],
        "git_sha": git_sha(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "checkpoint": str(CHECKPOINT),
        "dtype": DTYPE[device],
        "max_new_tokens": MAX_NEW_TOKENS,
        "wall_limit_s": WALL_LIMIT_S[device],
        "arms": list(ARMS),
        "text_prompts": list(TEXT_PROMPTS),
        "image_question": IMAGE_QUESTION,
        "free_memory_checks": [],
        "results": {},
        "status": None,
    }
    try:
        if device == "cuda":
            if not torch.cuda.is_available():
                raise SmokeAbort("CUDA is not available for the cuda run")
            cap = torch.cuda.get_device_capability(0)
            receipt["hardware"] = {**gpu_snapshot(), "compute_capability": f"{cap[0]}.{cap[1]}"}
        else:
            receipt["hardware"] = {"device": "host CPU", "note": "fp32 reference, no GPU"}
        for arm in ARMS:
            if device == "cuda":
                snap = gpu_snapshot()
                receipt["free_memory_checks"].append({"before_arm": arm, "free_mib": snap["free_mib"]})
                check_free_memory(snap["free_mib"])
            receipt["results"][arm] = run_arm(device, arm, deadline)
        complete = all(
            r["tokens_generated_count"] == MAX_NEW_TOKENS for arm in ARMS for r in receipt["results"][arm]
        )
        receipt["status"] = "RUN_COMPLETED" if complete else "RUN_INCOMPLETE"
    except SmokeAbort as exc:
        receipt["status"] = f"ABORTED: {exc}"
    except Exception as exc:  # recorded honestly; no receipt is written for a non-completed run
        receipt["status"] = f"FAILED: {type(exc).__name__}: {exc}"
    receipt["elapsed_wall_s"] = round(time.monotonic() - t0, 1)
    return receipt


def child_main(device: str, result_path: Path) -> int:
    receipt = build_receipt(device)
    result_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf8")
    return 0


def build_child_cmd(device: str, result_path: Path) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--child", "--device", device, "--result", str(result_path)]


def run_outer(cmd: list[str], timeout_s: float) -> dict[str, Any]:
    """Run cmd in its own session; at the deadline kill the whole session, grandchildren included.

    The parent enforces the bound, so a hung CUDA or CPU call inside the child cannot defeat it.
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


def _write_exclusive(out_dir: Path, name: str, payload: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = out_dir / f"{stamp}_pytorch_reference_gtx1080ti_smoke_{name}.json"
    with open(out, "x", encoding="utf8") as f:  # exclusive create: never replace a receipt
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "runs")
    parser.add_argument("--compare", nargs=2, metavar=("GPU_RECEIPT", "CPU_RECEIPT"))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.child:
        return child_main(args.device, args.result)

    if args.compare:
        gpu, cpu = (json.loads(Path(p).read_text(encoding="utf8")) for p in args.compare)
        try:
            out = compare_receipts(gpu, cpu)
        except ValueError as exc:
            print(f"REFUSED: {exc}; no comparison receipt written", flush=True)
            return 1
        out.update({"label": LABEL, "git_sha": gpu["git_sha"], "inputs": [args.compare[0], args.compare[1]]})
        path = _write_exclusive(args.out_dir, "compare", out)
        print(f"token_agreement={out['token_agreement']} receipt={path}", flush=True)
        return 0

    if args.device is None:
        parser.error("--device is required unless --compare is given")
    limit = WALL_LIMIT_S[args.device]
    with tempfile.TemporaryDirectory() as tmp:
        result_path = Path(tmp) / "result.json"
        run = run_outer(build_child_cmd(args.device, result_path), limit)
        if run["outcome"] == "OUTER_DEADLINE":
            print(f"ABORTED: outer hard deadline {limit}s reached ({args.device}); child session killed; "
                  f"no receipt written", flush=True)
            return 2
        if not result_path.exists():
            print(f"FAILED: child exited {run['returncode']} without a result; no receipt written", flush=True)
            return 1
        receipt = json.loads(result_path.read_text(encoding="utf8"))

    if receipt["status"] != "RUN_COMPLETED":
        print(f"status={receipt['status']}; not a completed run, so no receipt written", flush=True)
        return 1
    path = _write_exclusive(args.out_dir, TAG[args.device], receipt)
    print(f"status={receipt['status']} elapsed={receipt['elapsed_wall_s']}s receipt={path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
