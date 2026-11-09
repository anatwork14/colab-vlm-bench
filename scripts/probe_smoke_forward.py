"""Bounded diagnosis of the GPU smoke stall (ITER2 D8). A probe, not a smoke: it writes no receipt.

One mode per process. `plain` runs one 1-token fp16 forward through plain transformers with no
custom hooks. `engine` runs one 1-token greedy request through RealSevenArmEngine, arm `base`, fp16.
Each stage is timestamped. faulthandler dumps every thread's stack to stderr if the deadline is
reached, so the stall location is recorded even if the process is then killed by the outer timeout.
The GPU free-memory floor (3 GiB) is checked before any load.
"""
from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
CHECKPOINT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
T0 = time.monotonic()


def stamp(msg: str) -> None:
    print(f"[{time.monotonic() - T0:7.2f}s] {msg}", flush=True)


def gpu_gate() -> None:
    spec = importlib.util.spec_from_file_location("smoke_helpers", REPO_ROOT / "scripts/pytorch_reference_gpu_smoke.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    snap = smoke.gpu_snapshot()
    stamp(f"GPU {snap['name']} free={snap['free_mib']} MiB of {snap['total_mib']}")
    smoke.check_free_memory(snap["free_mib"])


def kernel_report() -> None:
    for mod in ("fla", "causal_conv1d"):
        stamp(f"module {mod}: {'installed' if importlib.util.find_spec(mod) else 'NOT installed'}")


def plain() -> None:
    import torch
    import transformers
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    stamp(f"torch {torch.__version__} transformers {transformers.__version__} cap {_cap()}")
    kernel_report()
    tok = AutoTokenizer.from_pretrained(CHECKPOINT)
    token_id = tok("The").input_ids[0]
    stamp("loading Qwen3_5ForConditionalGeneration fp16 on CPU")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(CHECKPOINT, dtype=torch.float16)
    model.eval()
    stamp("loaded; moving to cuda")
    model = model.to("cuda")
    torch.cuda.synchronize()
    stamp("on cuda; starting 1-token forward")
    ids = torch.tensor([[token_id]], device="cuda")
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits
    torch.cuda.synchronize()
    stamp(f"forward done; logits shape {tuple(logits.shape)} finite={bool(torch.isfinite(logits).all())} "
          f"argmax={int(logits[0, -1].argmax())}")
    stamp(f"peak GPU alloc {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")


def engine() -> None:
    import torch
    from vlmbench.infrastructure.real_engine import RealSevenArmEngine

    stamp(f"torch {torch.__version__} cap {_cap()}")
    kernel_report()
    stamp("constructing RealSevenArmEngine arm=base device=cuda dtype_override=fp16")
    eng = RealSevenArmEngine(method_arm="base", device="cuda", checkpoint_dir=CHECKPOINT, dtype_override="fp16")
    stamp(f"constructed; is_pretrained={eng.is_pretrained} is_official_hf_hooked={eng.is_official_hf_hooked}")
    stamp("starting 1-token execute_request")
    res = eng.execute_request(request_id="probe_base", prompt_token_ids=[760], max_new_tokens=1, image_present=False)
    torch.cuda.synchronize()
    stamp(f"request done; tokens={[int(t) for t in res.tokens_generated]}")
    eng.close()


def _cap() -> str:
    import torch

    return str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else "n/a (cpu run)"


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()


def sequence(arm: str = "base", device: str = "cuda") -> None:
    """The smoke's GPU requests for arm base, one stage at a time: two chat-template text prompts
    with 8 new tokens each, then the canonical image question. No CPU reference."""
    import torch
    from vlmbench.infrastructure.real_engine import RealSevenArmEngine
    from vlmbench.multimodal_fixture import CANONICAL_IMAGE_SHA256
    import hashlib

    fixture = REPO_ROOT / "tests" / "fixtures" / "000000039769.png"
    stamp(f"fixture sha256 ok={hashlib.sha256(fixture.read_bytes()).hexdigest() == CANONICAL_IMAGE_SHA256}")
    stamp(f"torch {torch.__version__} cap {_cap()}")
    dtype = "fp16" if device == "cuda" else None
    stamp(f"constructing RealSevenArmEngine arm={arm} device={device} dtype_override={dtype}")
    eng = RealSevenArmEngine(method_arm=arm, device=device, checkpoint_dir=CHECKPOINT, dtype_override=dtype)
    stamp(f"constructed; is_official_hf_hooked={eng.is_official_hf_hooked}")
    prompts = ("Answer in one word: what colour is an emerald?", "Complete the sentence: The capital of France is")
    for i, prompt in enumerate(prompts):
        text = eng.processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = eng.processor(text=[text], return_tensors="pt")["input_ids"][0].tolist()
        stamp(f"text{i}: prompt tokens={len(ids)}; starting 8-token request")
        res = eng.execute_request(request_id=f"probe_text{i}", prompt_token_ids=ids, max_new_tokens=8, image_present=False)
        _sync(device)
        stamp(f"text{i}: done tokens={[int(t) for t in res.tokens_generated]}")
    stamp("image: starting 8-token request with canonical fixture")
    res = eng.execute_request(
        request_id="probe_image", max_new_tokens=8, image_present=True, image_path=fixture,
        image_bytes=fixture.read_bytes(), question="Are there cats lying on the couch in the living room?",
    )
    _sync(device)
    stamp(f"image: done tokens={[int(t) for t in res.tokens_generated]}")
    eng.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("plain", "engine", "sequence"))
    parser.add_argument("--deadline-s", type=int, default=75)
    parser.add_argument("--arm", default="base")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args()
    faulthandler.dump_traceback_later(args.deadline_s, exit=True)
    gpu_gate()
    {"plain": plain, "engine": engine, "sequence": lambda: sequence(args.arm, args.device)}[args.mode]()
    faulthandler.cancel_dump_traceback_later()
    stamp("PROBE COMPLETED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
