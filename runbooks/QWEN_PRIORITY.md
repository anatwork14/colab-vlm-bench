# Qwen priority benchmark runbook

**Status: NOT_RUN (planned only).**

Owner priority: **Qwen3.5-0.8B and Qwen3.5-4B**, Base and native MTP. A100 40GB; MemLens + MMLongBench-Doc.

| Phase | Serving cells | Measured requests planned | Native predictions planned | Status |
|---|---:|---:|---:|---|
| Pilot | 48 | 2,816 | up to 384 | NOT_RUN |
| Confirmatory | 1,008 | 45,696 | up to 7,520 | NOT_RUN |

Runbooks produced by measured jobs will live under distinct IDs `runbooks/a100-qwen08b-4b-pilot-v1` and `runbooks/a100-qwen08b-4b-confirmatory-v1`, with raw receipts held on owner Google Drive. A newly observed result must never be inferred from this planned table. Other registered models remain deferred.

[Agent run instructions](../docs/PRIORITY_EXECUTION.md) · [Pilot config](../configs/a100_qwen_priority_pilot.json) · [Confirmatory config](../configs/a100_qwen_priority_full.json)
