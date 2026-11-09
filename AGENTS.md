# Agents — Colab VLM Bench

## EXISTING DRIVE CACHE — READ FIRST

Review `docs/EXISTING_DRIVE_LAYOUT.md`, then use read-only
`python scripts/colab_agent.py drive-inventory --session <existing-cpu-session>`.
STARLIGHT datasets and Qwen weights already live in two DIFFERENT Drive trees;
the current CLI default cache folder is separate. Avoid unnecessary HF
downloads, overwrite and false qualification.

## PRIMARY EXECUTION PATH — Colab CLI agents (2026-11-09)

**The two uploadable notebooks are now legacy, not active.** Read [docs/AGENT_COLAB_CLI.md](docs/AGENT_COLAB_CLI.md) and use `python scripts/colab_agent.py` on an authenticated Linux/macOS host. Agents operate named Colab CLI sessions, acquire source-pinned Qwen3.5-0.8B/4B and real MemLens/MMLongBench-Doc directly from Hugging Face during a CPU session, preserve them on mounted Drive, and submit tightly bounded Base/MTP GPU jobs only after explicit owner-paid-compute approval. Do not provision/stop Colab session silently. Never mark H2O/DA/ECRD qualified without verified backend ports.


## IMMEDIATE OWNER PRIORITY (2026-11-08)

**Do Qwen3.5-0.8B and Qwen3.5-4B only; extend 7 methods (Base, H2O+Base, DA+Base, ECRD+Base, MTP, H2O+MTP, DA+MTP).** Stop spending effort/GPU budget on Qwen3.5-2B/9B and Gemma 4 E4B for now. Those models remain registered for later.

**First read:** docs/SEVEN_ARM_PROTOCOL.md, docs/SEVEN_ARM_AGENT_TASKS.md, docs/HALLUCINATION_EVALUATION.md, docs/THIRD_PARTY_METHODS.md. Always keep `temperature=0` and `top_p=1.0`, and do not call wrong answers hallucinations without independent evidence. Source compatibility matters: H2O is not native Qwen3.5 vLLM, DA preserves original KV and its published profiles target SM100, and ECRD reweights decoding logits (not a KV safety mask). Unported arms return BACKEND_UNQUALIFIED, not fabricated Base measurements.

**Legacy first-results sequence (still valid as a subset):** docs/PRIORITY_EXECUTION.md; use configs/a100_qwen_priority_pilot.json (engineering) and configs/a100_qwen_priority_full.json (confirmatory). Never use configs/a100_campaign.json for immediate GPU execution. Stage only the two selected models and leave unpinned deferred model placeholders untouched. Do not claim full 5-model readiness or official quality from pilot/truncated questions.

Goal order: freeze seven-arm/greedy/evidence contracts → 0.8B Base smoke → 0.8B MTP smoke → 0.8B paired MemLens+MMLong pilot → 4B Base/MTP qualification and pilot → native quality subset → first runbook → full 1,008-cell priority campaign and official-quality scoring where feasible. **Publish the first truthful partial benchmark results before expanding the roster.**


**Current task**: independent A100 inference benchmarking, NOT STARLIGHT. Read README.md, docs/PROTOCOL.md, docs/COLAB.md, docs/ARCHITECTURE.md before editing.

## Ordering / ownership

1. CPU preflight: inspect git SHA/working tree; verify model IDs and precise processor/checkpoint revisions, Drive inventory, dataset licenses and host CLI. Do not overwrite existing Drive datasets.
2. Develop on branch; run unit tests and static checks. Use independent reviewer for changes to methodology, model pairing, dataset adapters and time/metric definitions.
3. Freeze campaign config with pinned revisions, actual GPU SKU, SHA256 of all inputs and split IDs, exact runtime versions, seed, quality scorer, output cap. **No unverified placeholder values in GPU campaign**.
4. Request owner approval for paid A100 GPU allocation/usage. Do not start/stop another user's runtime. First Drive mount requires user authentication. No secret in prompts or Git.
5. Pilot model base and MTP independently; verify MTP mode is actually active using backend trace and its correct version. Do not use `draft_model` instead of Gemma assistant MTP.
6. Serving benchmark: 42 cells per dataset per config, attempts at CCU 32 recorded, capacity/oom/cancel/timeout explicit; repeat three times on same GPU. Keep identical request IDs, model, prompt and seed across paired arms; don't optimize based on held-out quality test.
7. Native quality: MemLens entire context, MMLongBench full document, scorer upstream only; if A100 cannot fit, mark NOT_SUPPORTED_CONTEXT. Never truncate and still report official score.
8. After each cell, persist raw receipts and model/hardware/provenance to Drive. Aggregate and publish runbook summaries via reviewed controller. Never fabricate metrics. No arbitrary background paid jobs.

## Gates

- `CPU_TESTS_PASS` does not imply `GPU_MODEL_LOAD_PASS`
- `GPU_MODEL_LOAD_PASS` does not imply `MTP_CORRECTNESS_PASS`
- `PREDICTIONS_READY` does not imply `OFFICIAL_ACCURACY_PASS`
- Missing actual generated token counts, source hashes or backend conditions = INVALID/NOT_MEASURED.
- Store failing cells, failed start attempts and model incompatibilities. Avoid changing A100/BF16/TP1/hardware mid-campaign.

Scope allowed: this repo. No cross-repo changes unless owner explicitly asks.
