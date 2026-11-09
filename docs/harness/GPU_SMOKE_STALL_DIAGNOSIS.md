# GPU smoke stall: bounded diagnosis (ITER2 D8)

Status: **BLOCKED.** The smoke has no completed run and no receipt. No TTFT, throughput or
token-agreement claim is made. Probe script: `scripts/probe_smoke_forward.py` (writes no receipt).

## What was reproduced

| Probe | Device | Result | Time |
|---|---|---|---|
| `plain`: plain transformers, `Qwen3_5ForConditionalGeneration`, fp16, 1-token forward, no hooks | cuda | completed; logits finite, argmax 2614 | 3.3 s after load; peak 1675 MiB |
| `engine`: `RealSevenArmEngine` arm `base`, fp16, 1-token greedy | cuda | completed; tokens [2614] | construction 8.4 s; request 0.2 s |
| `sequence`: smoke's GPU requests, arm `base`: 2 text prompts (8 tokens) + image (8 tokens) | cuda | completed; all 3 requests | construction 10.6 s; requests about 3 s |
| `sequence`: same, arm `h2o_base` | cuda | completed; all 3 requests | construction 6.8 s; requests about 2 s |
| `sequence`: arm `base` | **cpu fp32** | **did not complete construction within 90 s** | killed by faulthandler at 90 s |

Logs: `/tmp/claude-5011/-mnt-ssd-genos2-research-workspace/464083ed-2e0f-4614-8b26-edbc6edd10b3/scratchpad/d8/engine_probe.log`, `/tmp/claude-5011/-mnt-ssd-genos2-research-workspace/464083ed-2e0f-4614-8b26-edbc6edd10b3/scratchpad/d8/sequence_probe.log`, `/tmp/claude-5011/-mnt-ssd-genos2-research-workspace/464083ed-2e0f-4614-8b26-edbc6edd10b3/scratchpad/d8/h2o_sequence_probe.log`, `/tmp/claude-5011/-mnt-ssd-genos2-research-workspace/464083ed-2e0f-4614-8b26-edbc6edd10b3/scratchpad/d8/cpu_sequence_probe.log`.

Environment at probe time: torch 2.6.0+cu124, transformers 5.22.2, GTX 1080 Ti (compute capability 6.1), free 5090 MiB of 11264 (above the 3 GiB floor). `fla` and `causal_conv1d` are **not installed**, so transformers falls back to the pure-PyTorch reference kernels. That fallback is slow on CPU but completed on the GPU in the probes above.

## Where the CPU path stalls

The faulthandler dump at 90 s shows the stack inside `RealSevenArmEngine.__init__` → `load_pretrained_weights` → `_load_official_hf_bundle` (`src/vlmbench/infrastructure/real_engine.py:400`). That line is the **hooked** parity forward. The constructor runs the same parity prompt through the plain model (line ~388) and then through the hooked model (line ~400). Both run inside the constructor, before any request. The innermost frame is `torch_chunk_gated_delta_rule` (`modeling_qwen3_5.py:366`), the pure-PyTorch fallback, in fp32 on CPU.

## What is inferred, not reproduced

- **The original smoke stall is most consistent with the fp32 CPU reference phase.** The smoke runs both GPU arms first and then constructs the CPU reference. The original run printed nothing after weight loading and stayed at about 175% CPU, which fits CPU-bound work. The GPU phase, which the same code path runs on cuda, completed in seconds in every probe above. The original run did not dump stacks, so this is inference, not a captured trace.
- The CPU measurement is confounded by host load. During the CPU probe another session's pytest was using about 6 cores (pid 1672872, not touched). The probe did not measure how long a CPU fp32 reference would take on an unloaded host.
- The full smoke was **not** re-run end to end. The GPU phase was reproduced piecewise (each arm's requests) but not as one process.

## Fixes and what remains

- Done (`9b332bc`): the wall limit is an outer hard deadline. The parent kills the child's session at 600 s and writes no receipt on a deadline or incomplete run. A stall is now bounded, not open-ended.
- Not done: the CPU fp32 reference phase cannot finish within the bound with the current reference kernels. Options, none applied here: (a) drop the CPU reference from the smoke and report GPU-only, which is honest but gives no fp32 comparison; (b) install `fla` and `causal_conv1d` for the optimized kernels, which is an install and needs owner authorization; (c) a shorter CPU reference (one prompt, fewer tokens) with its own budget. Option (b) is also untested on Pascal, so it is not a plan yet.
- No receipt exists. A receipt is written only when a run completes, so none is written here.

## D9 update: short CPU fp32 reference (decision c), BLOCKED

- The GPU run completed: `RUN_COMPLETED` in 12.3 s, receipt `artifacts/runs/20261109T200806Z_pytorch_reference_gtx1080ti_smoke_gpu_fp16.json`. The first GPU attempt was `RUN_INCOMPLETE` only because the check required 8 tokens; the engine stops at EOS (`<|im_end|>`, id 248046), which is correct. Fixed in `960fb17`.
- The short CPU fp32 reference, separate run, 300 s outer deadline, two attempts, **both aborted at 300 s with no receipt**:
  - Attempt 1: host load about 17 (8 cores), another session's pytest using about 580% CPU.
  - Attempt 2: host load 18.2, the same competing process at about 700% CPU.
- No CPU reference result exists, so no GPU-vs-CPU token comparison is claimed. Status: BLOCKED_CPU_REFERENCE_TIMEOUT on this host under load.
- Not done: raising the budget (not authorized), killing the other session's process (not ours). The CPU run should be repeated when the host is quiet, with `--device cpu`, and the comparison step run only if both receipts complete.
