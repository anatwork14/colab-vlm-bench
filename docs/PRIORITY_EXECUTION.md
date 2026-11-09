# Active campaign: Qwen3.5-0.8B and Qwen3.5-4B first

**2026-11-08 extension:** Seven methods now in scope: Base, H2O+Base, DA+Base, ECRD+Base, MTP, H2O+MTP, DA+MTP. This document's two-arm steps are the first **working subset** only. See [seven-arm protocol](SEVEN_ARM_PROTOCOL.md), [agent tasks](SEVEN_ARM_AGENT_TASKS.md) and [independent hallucination metrics](HALLUCINATION_EVALUATION.md). The additional five arms require real backend ports; do not pretend their metrics exist. Greedy `temperature=0` is mandatory throughout.


**Owner decision (2026-11-08):** Stop expansion to other model families; generate real reproducible results for the two priority Qwen3.5 VLMs before beginning Qwen3.5-2B, Qwen3.5-9B or Gemma 4 E4B. These deferred checkpoints stay in the registry, not in GPU execution scope.

## Execution sequence (A100 40GB, BF16, TP=1)

| Order | Step | Allowed models/modes | Artifact / exit gate |
|---|---|---|---|
| 0 | CPU contract + Google Drive inventory, source hashes, publisher dataset license | 0.8B/4B | Model/data SHA, 26+ CPU tests, dataset manifest |
| 1 | GPU smoke | Qwen0.8B Base → MTP | Real image+long prompt, output usage, MTP backend trace |
| 2 | Source-derived multimodal serving annotation | Qwen0.8B | Exact model-tokenized text bins and effective multimodal usage |
| 3 | 48-cell pilot, start 0.8B Base/MTP then 4B Base/MTP | Both datasets; 2K/16K; CCU 1/8/32, one repetition | First measured TTFT/TPOT/E2E/tok/s/capacity; failures preserved |
| 4 | Native quality pilot | 48 preselected IDs per dataset, same for Base/MTP and models | Up to 384 unscored predictions, validated scorer or clearly labeled proxy |
| 5 | GPU-hour projection and peer review | Same two model pairs | Pilot runbook, judged correctness, full-run budget and stop/go |
| 6 | Full priority confirmatory (new campaign ID) | Both datasets, all 7×6 cells, three repetitions | 1,008 cell statuses, 45,696 measured requests if all viable |
| 7 | Native full-context quality | MemLens 32K, MMLongBench full document | Up to ~7,520 predictions, independent official scorer receipts |
| 8 | Publish and decide on remaining models | No new models until report | Comparative Base/MTP graphs, uncertainty, OOM boundaries and quality |

**Immediate stop condition:** never label MTP equal to Base unless actual vLLM MTP activation and correctness are observed. If unsupported, mark BLOCKED_MTP; Base can still produce a valid baseline but no paired speedup claim. Never silently switch to text-only mode. Never score truncated or image-dropped samples as official native accuracy.

## Frozen and separate campaign files

- [Engineering pilot](../configs/a100_qwen_priority_pilot.json): `campaign_id=a100-qwen08b-4b-pilot-v1`; `execution_scope.kind=engineering_pilot`; models 08B,4B; lengths 2048/16384; CCU 1/8/32; repetition 0. 48 serving cells. Native quality sample cap 48 per dataset and model/mode, seed 42.
- [Full confirmatory](../configs/a100_qwen_priority_full.json): `campaign_id=a100-qwen08b-4b-confirmatory-v1`; kind `priority_confirmatory`; 7 lengths, 6 CCU, 3 repetitions. 1,008 serving cells. Full native quality (no sample cap).
- [Future five-model study](../configs/a100_campaign.json): unchanged, explicitly deferred.

These configs contain dummy Drive paths and are **dry_run=true**. Fill an owner-local copy with pinned revision/Drive paths for **only two selected models**, set a unique run ID if any protocol changes, and obtain explicit owner GPU resource approval. Never commit real Drive secrets or tokens.

## Execution on an existing approved Colab A100 session

Confirm `colab status -s vlmbench-a100`, Drive mount, source checkpoint hashes, exact A100 40GB, vLLM MTP and 256-output-token policy before running.

Example host-side invocation (after owner approves and a **private** pilot config exists with dry_run=false):

~~~bash
python -m vlmbench.cli plan --config configs/a100_qwen_priority_pilot.json

bash scripts/colab_controller.sh \
  --session vlmbench-a100 \
  --config /LOCAL/APPROVED/a100_qwen_priority_pilot.local.json \
  --model qwen35_08b --mode base --dataset memlens \
  --max-cells 6
~~~

The helper never commits or pushes. Copy the report from `runbooks/<campaign_id>/` after review, then commit it by hand. `--publish` is refused (audit F-07).

Repeat in matched order for `mtp`, other dataset and `qwen35_4b`. A separate invocation with `--quality` runs a deterministic **native-context subset**; it is not a full official quality score. Before confirmatory, construct a **new** local config copied from a100_qwen_priority_full.json with its own fixed campaign ID, digests and seeds.

## Release checklist for the first useful results

- Both models have actual GPU and image-path qualification; all deferred models have NOT_RUN status, not an invented failure.
- 48 pilot cells have explicit outcome receipts (PASS/FAIL/UNSUPPORTED/OOM/INSUFFICIENT_WORKLOAD), with attempted CCU 32 documented.
- Actual paired Base vs MTP sample hashes match. Check effective prompt tokens and verified 256 generated tokens per request.
- Available TTFT/TPOT/E2E/throughput/peak VRAM are measured in correct units. Internal MTP acceptance, physical KV bytes and true per-token ITL stay NOT_MEASURED without instrumentation.
- Native subset predictions are archived with SHA/IDs and official scorer parity remains explicitly pending until checked.
- Google Drive stores raw logs and failures. GitHub runbook contains statuses, comparisons and source evidence pointers, never invented performance.

**No GPU result has been recorded at the time this plan was added.**
