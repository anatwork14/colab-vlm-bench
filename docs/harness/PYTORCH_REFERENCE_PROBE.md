# PyTorch reference smoke option on the GTX 1080 Ti (ITER2 D2-5)

**Label: PyTorch reference (Hugging Face), not vLLM.** This is a capability probe only. The model was **not loaded** and no inference ran. It took about 3 s, well inside the 60 s limit.

Command (HEAD `1c2bbd8`, repo venv):

```
CUDA_VISIBLE_DEVICES=0 timeout 60 colab-vlm-bench/.venv/bin/python -   # inline probe, output below
```

| Field | Value |
|---|---|
| torch | `2.6.0+cu124` (CUDA build 12.4) |
| CUDA available | true |
| device | NVIDIA GeForce GTX 1080 Ti |
| compute capability | 6.1 |
| total / free VRAM | 11170 MiB / **4954 MiB** (about 6 GB used by another process, not touched) |
| `torch.cuda.is_bf16_supported()` | true (as reported by PyTorch) |
| compiled arch list | sm_50, sm_60, sm_70, sm_75, sm_80, sm_86, sm_90 |
| transformers | 5.22.2, `qwen3_5` registered in `CONFIG_MAPPING_NAMES` |

## What this proves and does not prove

- **Proves:** the local PyTorch build sees the 1080 Ti and can initialise CUDA. The installed transformers knows the `qwen3_5` architecture. About 5 GB is free.
- **Prior evidence on this GPU:** the existing receipts already record HF forward and logits-parity runs on `device: cuda` on this host (`artifacts/model_coverage_and_parity_receipt.json`, and the historical receipt with `cuda_device_name: NVIDIA GeForce GTX 1080 Ti`). These are forward checks, not serving benchmarks.
- **Does not prove:** that a 0.8B generation fits in the free VRAM, that bf16 runs at useful speed, or any throughput, latency or memory number. Compute capability 6.1 has no bf16 tensor cores. The `is_bf16_supported()` value may come from emulation, so do not assume bf16 is fast here. fp16 or fp32 are the safer choices for a smoke test.
- **Not vLLM.** vLLM is not installed, and the benchmark's vLLM path cannot run on this GPU (`BLOCKED_GPU_CAPABILITY`, `docs/harness/PILOT_INVENTORY.md`).

## Option (not run)

A short PyTorch reference smoke on the 0.8B weights is possible in principle: load in fp16, generate one short greedy answer, and check the output against the existing HF logits-parity receipt. It would take a few minutes and could use most of the 4.95 GB free. It was **not started**, because the brief forbids long jobs and VRAM is shared with another process. It would produce a PyTorch reference result, not a vLLM result, and not a benchmark measurement.
