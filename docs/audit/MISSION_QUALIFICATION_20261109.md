# STARLIGHT mission qualification — 2026-11-09

- Base (PR #20 head at start): `cfccc4b5b3c24621298ddf93144709cace9944d2`
- Integrated code state qualified here: `a3526c0` (this file is committed on top of it; see `git log`).
- Hardware available: CPU only for all evidence. The host GPU is a GTX 1080 Ti (sm_61, 11 GB); vLLM is not installed. No paid GPU was provisioned.
- Team (observed banners, not labels): lead wF:p5 = Sonnet 5.5; wF:p7, wF:p8, wF:p9 = **Haiku 5.5** (wF:p9 is labelled "opus" but is Haiku 5.5; no Opus review took place).
- Test evidence at `a3526c0`: `PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q` → **445 passed, 10 skipped, 0 failed** (the 10 skips are retired-notebook tests). Baseline at `cfccc4b` was 272 passed, 1 failed, 10 skipped.

Passing a lower level never implies a higher one. Every PASS below is CPU PyTorch / Hugging Face execution of Qwen3.5-0.8B only.

## Gate matrix (G0..G8 × seven arms)

| Gate | base | h2o_base | da_base | ecrd_base | mtp | h2o_mtp | da_mtp |
|---|---|---|---|---|---|---|---|
| G0 Provenance | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| G1 Structural | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| G2 Pretrained (0.8B, CPU HF) | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| G3 Multimodal (one canonical fixture) | PASS | PASS | PASS | PASS | PASS | PASS | PASS |
| G4 Method correctness | PARTIAL | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| G5 Benchmark integrity | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| G6 GPU / vLLM execution | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| G7 Performance | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |
| G8 Publication evidence | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED | BLOCKED |

Reasons (source: `docs/audit/REVIEW_INTEGRATED_R4.md` §4, `docs/audit/SOURCE_ALIGNMENT_MATRIX.md`):

- **G0:** the 0.8B weights header/byte totals and MTP inventory (15 keys) reconcile with the pinned manifest and `config.json`. Dataset revisions are unpinned (owner-supplied SHAs needed), the 4B model has no pinned config, no vLLM/transformers pins for the campaign.
- **G2:** official checkpoint loads 488/488 tensors with logits parity against the unhooked HF model. The parity reference is the same HF implementation, not an independent trusted reference.
- **G3:** a real image reaches the vision encoder and the language model on one canonical fixture; benchmark images are not exercised.
- **G4 base:** closest to correct; no per-arm blockers except unpinned processor/transformers versions (EOS stop added in round 5). PARTIAL, not PASS.
- **G4 h2o_*:** budget now fixed from prefill and enforced; protected tokens are outside the budget (a labelled variant, `h2o_visual_anchor_text_budget`, not upstream H2O); scores averaged over KV heads vs upstream per-head; `min_budget=32` unsourced. On VLM prompts total retention is ~1.0 by construction.
- **G4 da_*:** scripted, decode-only, not model-emitted declarations; LOCAL/GLOBAL scopes differ from the paper (`docs/DA_DEVIATIONS.md`). The masking itself is measured (masked-key attention weights exactly zero in the hooked forward). It is a labelled diagnostic, not paper DA.
- **G4 ecrd_base:** keyword logit bias, not ECRD; condition-dependent unsourced alpha; unsupported-penalty inert; no evidence source in the benchmark path.
- **G4 mtp, h2o_mtp, da_mtp:** the serving-style arm does not speculate (`speculation_executed=False`, acceptance `None`). A separate CPU greedy k=1 reference (`cpu_greedy_k1_reference_not_vllm_mtp`) is correct as a verifier (token-identical to greedy for accepted, rejected and mixed cases, KV + Gated-DeltaNet state rollback tested) but its draft ignores MTP KV history (A2-14). The `h2o_mtp` rollback of physical KV is unimplemented.
- **G5:** metric code fixed and tested (F-01..F-06, F-09), independent review by wF:p9 done for them; dataset provenance/revisions, licences and scorers are not resolved. Metric changes were reviewed by a Haiku-5.5 reviewer, not a human.
- **G6–G8:** no vLLM, no target GPU run, no paired measurements. No GPU/serving/performance number from this repo should be quoted.

## Numerical integrity (summary; detail in `docs/audit/NUMERIC_INTEGRITY_LEDGER.md`)

| Quantity | Status |
|---|---|
| Greedy token streams, logits parity (CPU, 0.8B) | real, CPU HF |
| MTP acceptance (CPU reference helper) | real, verifier-derived, diagnostic only; one-off CPU counts 3/12 and 8/23 on a text prompt are correctness-path counts, not performance |
| MTP acceptance (serving-style arm) | unavailable (`None`) |
| DA masked-key attention weight | real, measured in hooked forward on a scripted mask |
| DA attention/KV-read ratios | diagnostic (mask-derived), no physical traffic measured |
| H2O cumulative evicted/retained slots | real logical counts; no VRAM/physical saving measured |
| TTFT / TPOT / e2e / throughput | CPU harness values only; no GPU serving measurement exists |
| Peak GPU memory, concurrency under vLLM | unavailable |
| Dataset accuracy / EM / F1 / grounding | no benchmark run on real data in this mission |

## Open owner decisions

1. Dataset revision SHAs, 4B model pin, vLLM/transformers/driver version pins.
2. Authorization for a paid A100/L4 run (none was used).
3. Implement the upstream-style unprotected H2O variant (per-KV-head scores, sourced budget) before any accuracy-vs-KV claim; keep or replace `min_budget=32`.
4. Whether to wire the CPU MTP verifier into the serving arm, and replace the single-token MTP draft with MTP KV history.
5. Whether DA must use model-emitted declarations (paper DA) or stays a labelled scripted diagnostic; ECRD alpha and evidence source.
6. Unexplained one-off 3-failure full-suite run at `ebbf60d` (never reproduced in ≥6 later runs; traceback not captured).

## Verdict

`PRETRAINED_MODEL_VALIDATED` — scoped to Qwen3.5-0.8B, CPU Hugging Face forward (text and one canonical image), 488/488 tensors with logits parity. This does **not** mean any method arm is paper-correct, GPU-ready, or performance-qualified.
