# STARLIGHT seven-arm quantitative integrity audit — 2026-11-09

**Audited source:** PR #20 at base commit `c564a720ce5e95e35c76d09bbc31100f34cd59ea`; follow-up correction PR #22.  
**Verification method:** independent source-code review, public HF checkpoint safetensors index, upstream vLLM Qwen3.5 MTP tests, original Declarative Attention design, and GitHub Actions CPU test logs. This is **not** a post-repair GPU receipt.

## Numeric evidence ledger

| Claim | Reconciled value | Integrity classification | Evidence and limitations |
|---|---:|---|---|
| Qwen3.5-0.8B layer count | 24 total: 18 Gated DeltaNet, 6 full-attention | Source-consistent | Official model config + engine; six at indices 3/7/11/15/19/23 |
| Qwen checkpoint tensor keys | 488 in pinned source index; 473 base VL + 15 MTP | Source-consistent, checkpoint proof is historical | HF `model.safetensors.index.json` and historical local safetensors receipt; do not infer per-layer correctness from only key coverage |
| Counted parameters | 852,985,920 base + 20,452,864 MTP = **873,438,784** | Arithmetic consistent with reported key-shape inventory | Requires fresh independent per-tensor dtype/shape inventory to substantiate loaded bytes |
| HF `metadata.total_size` | 1,746,882,752 **bytes** | Verified published index | 2 bytes × 873,438,784 = 1,746,877,568 bytes, **5,184 fewer bytes**. The gap needs a dtype-by-tensor reconciliation; do not declare every key BF16 merely from aggregate size |
| MTP speculative depth | `num_speculative_tokens` is a runtime/configurable depth, not MTP weight count | Verified by upstream vLLM | 15 `mtp.*` keys do **not** mean 15 speculative steps |
| Historical MTP acceptance | Receipt says 0.4 (Test D) and 0.5 (Test E); some prior summaries said 20–25% | **INVALID/RETRACTED** | Original `step % 2 == 0` forced draft acceptance; new PyTorch MTP is candidate-only with no persistent draft cache, and has no legitimate acceptance/speedup metric |
| Historical DA 0.1895 | Was targeted in source code | **INVALID as an empirical paper result** | Replaced with scripted geometry mask. Last decode-row logical suppression and causal-L×L area sparsity now reported separately; model-declared upstream `<focus magic_chunks>` and physical KV read skipping NOT implemented |
| Historical H2O 638,976 B / 811,008 B | **624 KiB / 792 KiB** of SUMMED per-step KV shape deltas | **NOT actual peak GPU VRAM saved** | The code had summed differences in `numel×element_size` before/after `index_select`; repeated steps and CUDA caching mean these numbers are not allocator-reserved, allocated, or peak-memory deltas |
| Image processor shapes | 1,200 × 1,536 pixel patch values; 300 × 1,024 merged embeddings for one COCO fixture | Source receipt only, image-specific | Do not generalize 300 visual tokens to all AMBER/MemLens images |
| Four image diagnostic tasks | Four image/ViT executions, **zero official benchmark quality scores** | **FORMAT/INFERENCE SMOKE ONLY** | Prior checker wrongly accepted attribute answer `pink` for `ground_truth_label=Yellow` or generic `color/pattern/couch` words; do not report “4/4 accuracy” |
| CI after correction | GitHub Actions unit suite, CPU-only | Confirm the exact PR HEAD log before quoting | CPU CI skips actual pretrained checkpoint and A100/native vLLM tests. A green CI is not a GPU scientific gate |
| Old 256-token throughput | Local legacy receipt, generated before MTP-cost separation | **INVALID for paired performance conclusions** | Old Base unnecessarily executed MTP head; post-fix real pretrained GPU metrics have NOT been collected |
| H₂O visual safety | Previously one global visual token mask was compacted per layer | **BUG REPAIRED IN SOURCE, CPU REGRESSION REQUIRED** | Different full-attention layers have different retained-index layouts; new per-layer masks, divergent compaction regression and rollback safety |
| Native vLLM / A100 | No current qualification receipt | BLOCKED | Never infer from PyTorch / GTX 1080 Ti or CPU CI |

## Number-level definitions required by the paper

- `mtp_proposal_match_rate` = ordinary target matches / eligible reference MTP proposals, **diagnostic only**; not speculative acceptance. The current reference head has no persistent MTP history and cannot be used for production acceptance-rate publication.
- `mtp_acceptance_rate` = accepted, **committed speculative** tokens / truly proposed tokens. Cannot be reported until a native correct verifier exists. An internal compatibility value of zero when there is **no speculative serving** is a *no-attempt sentinel*, NOT an observation that the pretrained MTP head has 0% acceptance.
- `attention_mask_restriction_ratio` = disallowed keys / total eligible keys **in the final decode query row**, when all methods use the same logical prompt. This is not bytes read or time saved. `full_causal_mask_area_restriction_ratio` is distinct: excluded cells / all causal cells of a hypothetical prefill-style matrix.
- `physical_kv_bytes_freed` is a **deprecated legacy field name** currently mapping to a sum of per-step KV tensor payload deltas, not peak/allocated/reserved GPU VRAM. Prefer `kv_memory_accounting.compaction_tensor_byte_deltas_sum`; report `peak_cuda_allocated_saving_bytes=null` and `peak_cuda_reserved_saving_bytes=null` until measured with matched runs.
- `CCU_REQUEST_ISOLATION_PASS` is a per-request correctness observation. It does not certify actual simultaneous GPU scheduler throughput, queue discipline or 16/32 concurrency feasibility.

## Remaining blocking GPU evidence

1. Re-run progressive suite against actual pinned pretrained 0.8B after all fixes; compare Base and diagnostic MTP output tokens using the SAME model checkpoint, tokenizer, visual bytes and seeds. Preserve raw receipts and exact backend/device.
2. Independently validate that the official MTP `num_speculative_tokens` runtime accepts/rejects through target verification; use vLLM Qwen3.5 official MTP e2e tests as comparators.
3. Reconcile tensor dtypes/shapes behind 5,184-byte HF index gap and inspect full 488-key source mapping. Verify pins and SHA256 for actual local checkpoint.
4. Test H₂O per-layer visual retention on the **real** Qwen hybrid model, and measure actual CUDA peak allocated/reserved with matched baseline. Report KV tensor payload separately.
5. Verify DA upstream tag FSM, segment IDs, next-step transitions, native attention metadata, device-vs-mirror KV read counts and no-DA parity. Qwen3.5 on the target GPU needs its own profile/contract validation.
6. Use AMBER official scorer to obtain accuracy/F1/hallucination metrics; canonical image fixtures cannot substitute.
7. Hold all full AMBER/MemLens and paid A100 experiments until an authorized bounded GPU qualification pass.

## Direct primary source links

- HF index: https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/model.safetensors.index.json
- HF model: https://huggingface.co/Qwen/Qwen3.5-0.8B
- vLLM Qwen3.5 native MTP tests: https://github.com/vllm-project/vllm/blob/main/tests/v1/e2e/spec_decode/mtp/qwen3_5/test_mtp.py
- Original DA design: https://github.com/itsnamgyu/declarative-attention/blob/main/docs/DESIGN.md
- DA paper: https://arxiv.org/abs/2609.02737

**Publication verdict:** IMPORTANT REVISIONS APPLIED; **NOT SEVEN-ARM PUBLISHABLE**. There is no claim of zero bugs or any post-repair A100 measurement.
