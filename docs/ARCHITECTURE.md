# Architecture / boundaries

```text
configs/a100_campaign.json ──> config validation ──> immutable run card
                                     │
Drive checkpoint & source ──> stage/hash preflight (only after approval)
                                     │
                         A100 vLLM Base or MTP
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
           Serving replay                       Native quality replay
   derived real-source prompts                 full dataset prompt/image
    7 lengths × 6 CCU × 3 reps                     natural EOS
                  │                                     │
    streaming timing / GPU telemetry               predictions only
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     │
                      Drive raw receipts, JSONL
                                     │
                offline scoring / aggregate / runbook
                                     │
                   GitHub small summary (reviewed)
```

Python modules:
- `config.py`: closed schema, method and grid plan
- `datasets.py`: deterministic MemLens and MMLongBench-Doc native input normalization; no silent truncation
- `workloads.py`: normalized JSONL loading, verified length/paired sample selection
- `engine.py`: vLLM serve argv and startup/cleanup, target checkpoint consistency
- `bench.py`: OpenAI-compatible async streaming serving workload client
- `artifacts.py`: atomic receipts + hashing / durable sync
- `quality.py`: official-score import vs explicitly-labeled auxiliary proxy
- `report.py`: statuses and aggregate summaries, never invented GPU metrics
- `cli.py`: CPU plan, dataset prepare, runtime serving and benchmark entrypoints

Scientific baseline vs upstream: no copied or modified STARLIGHT code. This repo has separate history, CI and results. Never copy licensed third-party dataset/model content into Git.

## Extensibility
New GPU providers, workload datasets, vLLM backend versions, MTP depths, custom judge or metrics can be added behind explicit registries, versioned contracts, new run IDs and independent validation. No silent substitutions.
