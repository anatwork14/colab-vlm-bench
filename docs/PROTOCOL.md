# Protocol v1 — 2026-11-08 (prospective; no GPU qualification yet)

## Research goals
1. Does native MTP improve committed output tokens/s, TTFT/TPOT/E2E and service capacity over the *same* non-speculative VLM?
2. Does the VLM retain native MemLens and MMLongBench quality? Treat an MTP model's output as **target-preserving only if native acceptance/verification correctness is tested**, not just because a flag exists.
3. What performance/capacity boundary does A100 40GB impose at offered CCU 1,2,4,8,16,32 and context bins 128..16K?

## Cohort and workload semantics
Five target checkpoints, two modes, two datasets. Freeze revision, processor and original dataset population. **Performance serving** uses manually/algorithmically derived **real-source** prompts and 7 target model-token input bins at ±5%, recorded with model-specific tokenizer/processor counts; image tokens separate from text tokens. The 128-token bin is **128 text tokens + vision payload**, not 128 total effective model tokens; log actual prompt_tokens separately. Derivation uses last text plus limited images and has NO official quality claim. Native quality uses unmodified full evidence and question; no truncated official quality.

Grid: 7 **model-tokenized text-content** lengths × 6 CCU = 42 cells/config/dataset, 3 independent repetitions, randomized/counterbalanced order. Each cell measured count=max(16,4×CCU), warmup=CCU, fixed target 256 generated tokens with natural EOS disabled for performance. Fail closed on output-token-count mismatch. Per-request schedule is same across two modes. Samples must differ across requests; record if reused due to population limits. No prefix caching in primary track. CCU refers to **offered concurrent requests**, not guaranteed active GPU sequences.

Quality: MemLens 32K full (if fits); MMLongBench full document. Preserve all images and evidence. Report official upstream scores and separate auxiliary EM/token F1; independent judge may require additional resources. For failed/OOM items, show success rate and policy of denominator. A100-only hardware result does not establish inter-GPU generalization.

## Serving metrics
For each completed request: TTFT arrival-to-first output **stream chunk**; E2E arrival-to-finish; TPOT=(finish-first-token)/(output_tokens-1) where available; stream interchunk latency is **NOT** per-token ITL for batched/MTP outputs. At each cell: p50/p95/p99, error ratio, timeout count, exact output tokens, committed tokens/s, requests/s, GPU total used MiB sampled, queue wait if instrumented. Backend physical KV bytes and SD acceptance rate must be explicitly instrumented or marked NOT_MEASURED. Report all cells including unsupported/OOM.

## Fair comparisons and inference methods
BF16, TP1, A100 40GB, pinned vLLM, max_num_seqs=32, no prefix caching, max_model_len=32768 for serving. Native MemLens 32K+vision may require higher max_model_len than the 32K serving setting; qualify and record separately. Gemma E4B assistant uses `method: mtp` with matching official assistant model; don't guess its repo ID. Begin speculative depth=1 for both families so one algorithmic choice is controlled; predeclare depth sensitivity as later ablation.

## Status semantics
PLANNED, DRY_RUN, BLOCKED_CONFIG, BLOCKED_HARDWARE, UNSUPPORTED_MODEL, UNSUPPORTED_CONTEXT, OOM, TIMEOUT, ERROR, PASS_MEASURED, QUALITY_PREDICTIONS_UNSCORED, QUALITY_OFFICIAL_SCORED, QUALITY_PROXY_ONLY. CPU validation never yields a GPU PASS.

## Time budget
Pre-pilot hypothesis: ~75–185 A100-GPU-hours including 3-rep serving and native quality (wide, **unverified estimate**). 2,520 serving cells in the full matrix; do not extrapolate exact duration from random text-only examples. Pilot measured wall times are required to recalibrate and prioritize based on quota. Colab runtime/availability/compute-unit allocation are not guaranteed.

## Sources, provenance and licensing
- MemLens: https://github.com/xrenaf/MEMLENS ; https://huggingface.co/datasets/xiyuRenBill/MEMLENS (data CC-BY-4.0; source images may vary)
- MMLongBench-Doc: https://github.com/mayubo2333/MMLongBench-Doc (research non-commercial license; official QA revised)
- vLLM MTP: https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- Colab CLI: https://github.com/googlecolab/google-colab-cli

Run cards must link exact source revisions or SHA256; cite upstream as required by dataset/model license.
