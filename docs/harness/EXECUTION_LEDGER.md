# STARLIGHT harness — execution ledger

Single chronological record of iterations. Historical reports elsewhere are not sources of truth.
Agents (observed banners): p5 Sonnet 5.5 (lead); p7, p8, p9 Haiku 5.5. No Opus review has occurred.

## Iteration 0 (previous mission, 2026-11-09)
- cfccc4b -> 7d209f5, 82 commits, 445 passed / 10 skipped / 0 failed (CPU). Verdict PRETRAINED_MODEL_VALIDATED (0.8B CPU HF only).
- Evidence: docs/audit/MISSION_QUALIFICATION_20261109.md, docs/audit/REVIEW_INTEGRATED_R4.md.

## Iteration 1 (this directive)
- Start SHA: 7d209f5 (= PR #20 head, verified by fetch).
- Reconciliation facts (verified by lead):
  - Weights present: Qwen3.5-0.8B only. 4B absent.
  - vLLM: not installed. Local GPU GTX 1080 Ti sm_61 (vLLM unsupported), ~6 GB used by another process.
  - /mnt/ssd 100% full, 11 GB free.
  - Real pilot receipts: none known; inventory by wF:p8 (docs/harness/PILOT_INVENTORY.md) pending.
  - STARLIGHT controller: absent.
  - PR #20 body was stale/overclaiming; replaced by lead.
- Assignments: D1 p7 Stage-1 controller; D2 p8 pilot inventory + scorer audit; D3 p9 independent invariants oracle + claim reconciliation.

## Blocker register (current)
| ID | Dependency | Required for | Status |
|---|---|---|---|
| B-GPU-AUTH | Owner authorization for a specific paid GPU (A100 40GB preferred) | G6, G7, pilot cells, Stage 2 perf | BLOCKED_OWNER_AUTHORIZATION |
| B-VLLM | vLLM install + supported GPU (sm>=7.0); local 1080 Ti is sm_61 | native vLLM Base/MTP, G6 | BLOCKED |
| B-4B-WEIGHTS | Qwen3.5-4B weights/revision pin (also ~9 GB; local disk has 11 GB free) | all 4B cells | BLOCKED |
| B-DATA-PINS | Owner-approved dataset revision SHAs | G0, G5 | BLOCKED |
| B-DISK | /mnt/ssd 100% full | any large artifact | OPEN |

## Iteration 1 — outcome (integrated HEAD 1167cbc plus this ledger commit)
- Start 7d209f5 -> integrated 1167cbc. Full suite (CPU): **744 passed, 10 skipped, 0 failed** (baseline 445 at a3526c0).
- Agents (observed): p5 Sonnet 5.5; p7, p8, p9 Haiku 5.5. No Opus review has occurred.
- Delivered (CPU reference, label `cpu_pytorch_reference_not_vllm`; no speed/VRAM claims):
  - STARLIGHT Stage 1 controller (`src/vlmbench/infrastructure/starlight.py`, arms `starlight_s1`, `_no_da`, `_no_grounding`, `starlight_s1_mtp`), design in `STARLIGHT_STAGE1_DESIGN.md`. Contract: eligible_to_evict = h2o_candidate AND NOT protected; INFEASIBLE evicts nothing and flags `kv_budget_violated`.
  - Stage 2 CPU reference: serving-style `mtp` arm runs verifier-backed speculation; policy-state rollback for `h2o_mtp` and `starlight_s1_mtp` (text-only, no-evidence scope); `da_mtp` stays non-speculative. Per-KV-head H2O is tracker-only.
  - Execution inventory (`PILOT_INVENTORY.md`): pilot 48 cells, 0 READY (24 BLOCKED_WEIGHTS_MISSING for 4B, all BLOCKED on dataset missing/GPU/authorization/vLLM/revision pins); full matrix 3528, 0 READY, 2520 UNSUPPORTED (arms without a qualified backend).
  - MMLongBench-Doc official stage-3 scorer ported with provenance; stage 2 BLOCKED_NO_JUDGE. Release gates now derived from measured results (no literals). Claim reconciliation in `CLAIM_RECONCILIATION.md`.
- Independent reviews (Haiku 5.5): `REVIEW_ITER2.md`, `REVIEW_ITER2B.md`. Findings S-1..S-8, G-1..G-4, I-1, N-1..N-13 fixed or routed; closure of the final round (D9-D11 fixes) was NOT re-reviewed independently. Open: A-2/A-4/A-5/A-6 owner rulings (lead rulings recorded in commits), S-9 (DA scripted, not model-declared), HC-5 relevance labels (NOT_MEASURED).
- Real execution: GPU PyTorch reference probes on GTX 1080 Ti completed for base/h2o_base (piecewise, no receipt; fla/causal_conv1d absent; CPU fp32 reference too slow under host load). ITER2 C-7 correction: one GPU smoke receipt exists as a LOCAL, gitignored file (not in git, not indexed; see GPU_RECEIPT_INDEX.json note). The 'no receipt' wording above refers to committed receipts. No vLLM, no real benchmark cells, no Qwen3.5-4B.

## Blocker handoff (update)
| ID | Needed | Status |
|---|---|---|
| B-GPU-AUTH | Owner authorization naming a specific GPU resource (A100 40GB preferred) | BLOCKED_OWNER_AUTHORIZATION |
| B-VLLM | vLLM install on sm>=7.0 GPU (local 1080 Ti is sm_61) | BLOCKED |
| B-4B-WEIGHTS | Qwen3.5-4B weights + pin; local disks lack space (/mnt/ssd 10 GiB free vs 20 GiB guard; /mnt/hdd not writable) | BLOCKED |
| B-DATA | MemLens and MMLongBench-Doc not present locally; revision pins/download space needed | BLOCKED |
| B-JUDGE | LLM judge/extractor for MMLongBench-Doc stage 2 | BLOCKED_NO_JUDGE |
| B-DISK | Free space or writable large volume | OPEN |
Next executable action once authorized: run `scripts/harness_inventory.py` live probes on the A100 host, then pilot Base/MTP cells.

## Iteration 2 (owner directive "ITERATION 2+")
- Start SHA d8ebc16 (verified: local = origin/feat/qwen35-pretrained-validation = PR #20 head). Agents (observed): p5 Sonnet 5.5; p7, p8, p9 Haiku 5.5. No Opus review.
- Round 1: p9 closure matrix + protection-signal audit (`ADVERSARIAL_CLOSURE_ITER2.md`); p7 B1/B2/B3 tests + `BACKEND_INTERFACE.md` (found defect F-1); p8 readiness ladder, signed pins, scorer provenance, `READINESS.md`, `DATASET_READINESS.md`, `QWEN35_4B_STAGING_PLAN.md`, `GPU_RUNBOOK.md` (all unexecuted/planning).
- Round 2: F-1 fixed (shared retained set across full-attention layers; labelled deviation from per-layer H2O, STARLIGHT-vs-H2O now also measures it); C-3/C-5 labels (evidence `CALLER_DECLARED_NOT_VERIFIED`, `visual_evidence_protection: NOT_IMPLEMENTED`); C-4 per-keyword cap; C-1 R1 weightless pin, C-2 isolated vLLM check, C-9 receipt order; p9 re-review added F-1 coverage tests (7 mutants, all caught) and refused unknown modalities.
- Round 3: union cap on protected keywords (0.5, owner-ratifiable); fixed host tool paths (G-PATH); owner-key fingerprint pin mechanism present but UNSET (needs owner fingerprint).
- Lead fixes: stale fixtures (modality `text`->`text_span`), retired-label literal, seeded one random-init test.
- Tests: first full run at c8a352e had 14 failures (fixtures, not product); fixed; single rerun at ea8cfae: **890 passed, 10 skipped, 0 failed** (iteration 1: 744). Stage 2 files run alone earlier: 31 passed, 1 xfail (F-1, since fixed).
- Real execution: none. 0 GPU receipts, 0 benchmark cells, vLLM not run. Pilot inventory READY = 0.
- Open (not fixed): N-R2-1 operator-supplied signer anchor (no external trust anchor); C-3 `verified` still honoured for protection; C-5 no visual-token protection; A-2/A-4/A-5 owner rulings; S-9 DA scripted; HC-5 NOT_MEASURED; C-11 not re-checked; round-3 changes not independently re-reviewed; `engine.py` bare nvidia-smi; stopword list and caps are lead/p7 values awaiting owner ratification.
- Blockers unchanged: B-GPU-AUTH, B-VLLM, B-4B-WEIGHTS, B-DATA(-PINS), B-JUDGE, B-DISK, plus B-OWNER-KEY (owner key fingerprint). Next executable action once authorized: `GPU_RUNBOOK.md`.
