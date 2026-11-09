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
