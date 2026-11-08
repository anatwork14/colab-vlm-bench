# Agents — Colab VLM Bench

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
