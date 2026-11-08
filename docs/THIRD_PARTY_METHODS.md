# Upstream implementations and compatibility audit (2026-11-08)

| Source | Repo SHA pinned | What upstream really provides | Qwen3.5 + A100 + vLLM status |
|---|---|---|---|
| H₂O | `ac75c2a8a9e76832b2a4139b9363373b56336bfb` | HF/FlexGen heavy-hitter/recent KV eviction | **NOT PORTED** |
| Declarative Attention | `24110b7b441ad9423f55952a2c02d23eac80ec52` | DA state machine, tag layouts, compact vLLM attention metadata, original KV preserved | **NOT QUALIFIED** for Qwen3.5 0.8B/4B and A100 |
| ECRD | `0bc209f599bfef0d28d19e9a34024464406a9295` | HF `LogitsProcessor`, evidence pool, GRIT visual decider example for Qwen2.5-VL | **NOT PORTED** for Qwen3.5/vLLM |

See full source links in `docs/SEVEN_ARM_PROTOCOL.md`.

## Porting criteria
**H₂O**: implement recurrent/hybrid layer eligibility (only actual attention KV evicted), preserve attention heavy-hitter score semantics, compare with unmodified checkpoint with 100% cache before 20/40/60% eviction. Test near-needle retrieval, prefill and every decode, real physical memory release, eviction safety under MTP verification/rollback.

**DA**: implement the full DA contract (tags global/focus/local, attention mask/read selection, prompt layout, segment addressing, valid KV group/block accounting). A100 profile needs actual CUDA attention mask kernels. Report KV **read** savings separately from KV **resident** memory. Test strict prompt parity, BOS/EOS, multimodal image token ranges, and correct declarations from model.

**ECRD**: port evidence extraction, token-level logits reweighting and optionally visual decider to target Qwen3.5; pin auxiliary model revision and include auxiliary GPU/CPU/API latency, VRAM, throughput, and token budget. Verify `temperature=0` yields greedy selection over *reweighted* logits and that it is intentionally not a lossless target-equivalent method. Track when evidence was externally extracted vs generated, false evidence and evidence-loop truncation.

**H₂O+MTP, DA+MTP**: draft-target acceptance and rollback property tests, cache/scheduler semantics, no hidden extra prefill; both must use same target checkpoint with no accidental double counting. No baseline surrogate is acceptable.

## Licenses and reproducibility
All three source repos are independently owned. Do NOT blindly copy large upstream trees or vendor mismatched framework forks. Inspect LICENSE and dependencies, pin SHAs, document patches and patch provenance, and use separate backend adapters after real GPU tests. Source availability ≠ implementation compatibility.
