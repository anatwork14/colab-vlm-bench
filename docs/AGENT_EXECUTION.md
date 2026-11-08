# Agent GPU execution deliverables

> Active execution scope: Qwen3.5-0.8B and Qwen3.5-4B only; follow [PRIORITY_EXECUTION.md](PRIORITY_EXECUTION.md). The original five-model work below is a **later** phase and must not block the two-model priority.


**P0 — CPU qualification.** Model/processor SHA pinned and staged in owner Drive; dataset source hashes pinned; code CI/test gate; 5×2×2×42×3 fixed grid; no gold in requests, legal dataset usage.

**P1 — A100 GPU pilot.** Exactly 1× A100 40GB, BF16, TP1. Confirm native vLLM 0.29.0 compatible wheels, all five VLM model forward passes, Gemma E4B *matching assistant* native MTP and Qwen native MTP. Verify real visual tokens and input-count calibration.

**P2 — Real data replay.** Build source-derived workload variants for seven text-context bins, model-specific processor token counts and enough unique prompts per cell. Input 128 tokens + vision payload may exceed 128 effective tokens; record two counts and mark infeasible bins. Native MemLens 32K and full-document MMLongBench must never lose evidence.

**P3 — Full serving.** Paired Base/MTP on equal prompts, 1/2/4/8/16/32 offered CCU, 3 reps, measured output 256 tokens. Save each cell and failures to Drive before moving on. Recalculate runtime from actual pilot; do not silently change GPU, precision, CCU or context.

**P4 — Quality.** Native full context, original scorer compatibility reviewed, MemLens LLM judge external if needed; unverified EM/F1 labeled PROXY. Report native context unsupported on A100 rather than truncating.

**P5 — Report and review.** All 2,520 cells have a status, exact run ID, checkpoint hashes, environmental receipt, output length, per-request event summaries, uncertainty and independent review. Publish GitHub Markdown and JSON summaries; raw receipts Drive.

## Known engineering gaps (not scientific PASS)
- Source-derived, **processor-verified** 7-bin dataset builder needs to be qualified on the exact five model processors; current normalized input adapters do not automatically produce those variants.
- Native dataset prompt rendering has not been certified identical to upstream evaluation templates.
- Official MemLens judge and MMLongBench scoring require independent upstream pipelines; existing proxy scorer is not a replacement.
- Server-side token-level ITL, queue wait, MTP acceptance rate and actual physical KV bytes are not captured by client-only runtime; remain NOT_MEASURED.
- Multi-request long-running sequence replay is a future derived workload, not equivalent to treating independent requests at CCU 32 as a persistent conversation.


## Concrete workload instructions
Run prepare-memlens / prepare-mmlong (full evidence), derive-serving for each dataset, and annotate-serving against every pinned Base checkpoint. Only after annotations have been frozen and sufficient unique prompts satisfy each text-token bin may the 42-cell Base/MTP paired benchmark begin. Never run a random-token workload under a real-dataset label.
