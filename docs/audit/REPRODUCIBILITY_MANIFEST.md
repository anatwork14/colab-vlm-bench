# Reproducibility manifest — benchmark engineering audit (2026-11-09)

Companion to [NUMERIC_INTEGRITY_LEDGER.md](NUMERIC_INTEGRITY_LEDGER.md). Status: **CPU static audit only. No GPU run, no Colab session, no Drive access, no paid action, no dataset or checkpoint download.**

## 1. Identity

| Item | Value |
|---|---|
| Repository | `colab-vlm-bench` (worktree `wt/wf-bench-p8`) |
| Audited base | `cfccc4b5b3c24621298ddf93144709cace9944d2` (PR #20 head, branch `feat/qwen35-pretrained-validation`) |
| Code fixes (tests first, then fix) | `ebbf60d5bee25b6095e84fba8f9d46dd7a9ba94c` |
| Telemetry trust fields in receipts | `07b64c9` (HEAD of the code at the time of this audit; counts in §3 are at this commit) |
| Docs commit | `dd6df27`: adds `docs/audit/*` and `runbooks/GPU_RUNBOOK.md` |
| Round 2 fixes (tests first, one commit each) | F-05 `92c8b68` · F-06 `00260cc` · F-07 `4975562` · F-09 `573be47` |
| Round 2 generator and guard | item 6 `e4a30ae` (`repro_manifest.py`, `PINNED_INPUTS.md`) · item 7 `49e3d58` (`test_no_paper_targets.py`) |
| Round 2 docs | the commit that updates this file, `runbooks/GPU_RUNBOOK.md` and the ledger (see `git log`) |
| Branch | `feat/…` worktree branch; **not pushed, not merged** |

To reproduce the audit state: `git checkout ebbf60d` (or the docs commit), then run §4.

## 2. Environment used for the test results

The audit reused an existing venv on this host so that no software was installed:

| Item | Value |
|---|---|
| Interpreter | `/mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python` (CPython 3.11.16) |
| pytest | 9.2.0 |
| httpx / ijson / psutil / numpy | 0.28.1 / 3.6.0 / 7.2.2 / 2.3.5 |
| transformers / torch / huggingface_hub | 5.22.2 / 2.6.0+cu124 / 2.4.0 |
| vLLM | **not installed** (GPU path not exercised) |
| PyMuPDF (`fitz`) / pyarrow | **not installed**. The MMLong render regression test uses a fake `fitz` module. |
| ruff (lint) | 0.17.4, `--select E4,E7,E9,F821` on changed files: all checks passed |
| `PYTHONPATH` | `src` (also set by `pyproject.toml` `pythonpath = ["src"]`) |

**Declared-profile mismatch:** `requirements-agent-cpu.txt` pins `huggingface_hub>=0.35,<2`. The audit venv has 2.4.0. Results may differ on the declared profile. Re-run on a profile that matches the pins before relying on them.

## 3. Results (counts)

| Run | Result |
|---|---|
| 5 named modules, `cfccc4b` | 40 passed, 8 skipped |
| 5 named modules, `ebbf60d` | 40 passed, 8 skipped |
| Full suite, `cfccc4b` | 272 passed, 1 failed, 10 skipped |
| Full suite, `ebbf60d` | 283 passed, 1 failed, 10 skipped |
| Full suite, `49e3d58` (round 2 HEAD) | 340 passed, 1 failed, 10 skipped (same p7 failure) |
| Full suite, `dd6df27` x3 (flake check) | 283 passed, 1 failed, 10 skipped each time |
| New regression file vs `cfccc4b` archive | 10 failed, 1 passed (guard) |
| New regression file, `ebbf60d` | 11 passed |

The one failure in every full run is `tests/test_multimodal_fixture.py::test_progressive_qualification_test_c_multimodal_execution` (p7-owned, pre-existing, outside this task). The 10 skips are legacy notebook tests (8) and `test_stepwise_audit.py` (2). One full run at `ebbf60d` showed 3 unexplained failures, and four later runs did not reproduce them (see the ledger §0).

## 4. Commands

Run from the repository root with `PYTHONPATH=src` and the venv above.

```bash
PY=/mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python

# Required modules
$PY -m pytest -q tests/test_metrics.py tests/test_datasets.py tests/test_notebooks.py \
    tests/test_numeric_gate_invariants.py tests/test_numeric_artifact_provenance.py

# Regression tests added by this audit
$PY -m pytest -q tests/test_measurement_regressions.py

# Full suite
$PY -m pytest -q

# Lint (Makefile target, changed files)
$PY -m ruff check --select E4,E7,E9,F821 src tests

# CPU-only plan and validation (no GPU, no network)
PYTHONPATH=src $PY -m vlmbench.cli plan --config configs/a100_qwen_priority_pilot.json
PYTHONPATH=src $PY -m vlmbench.cli plan --config configs/a100_qwen_priority_full.json

# Refusal checks (must fail closed)
PYTHONPATH=src $PY -m vlmbench.cli run-job --config configs/a100_qwen_priority_pilot.json \
    --model qwen35_08b --mode base --dataset memlens        # PermissionError: --approve-paid
$PY scripts/colab_agent.py benchmark --session vlmb-a100 --model qwen35_08b \
    --dataset memlens --phase pilot --mode base             # PermissionError: --approve-paid
PYTHONPATH=src $PY -m vlmbench.preflight --checkpoint /nonexistent/ckpt   # CPU_PREFLIGHT_BLOCKED
```

Expected CPU plan output:

| Config | Cells | Measured requests | Warmups | Nominal output tokens | Quality jobs |
|---|---:|---:|---:|---:|---:|
| `a100_qwen_priority_pilot.json` | 48 | 2,816 | 656 | 720,896 (= 2,816 × 256) | 8 (2 models × 2 modes × 2 datasets) |
| `a100_qwen_priority_full.json` | 1,008 | 45,696 | 10,584 | 11,698,176 (= 45,696 × 256) | 8 |

Requests per cell = `max(16, 4 × CCU)`; warmups per cell = `CCU` (`config.py:158-185`).

## 5. Configuration and dependency checksums (SHA-256 at `ebbf60d`)

| File | SHA-256 |
|---|---|
| `configs/a100_qwen_priority_pilot.json` | `e98fd1a78a4948eb0a5c48b3055c410bee9fae8cc1655214f6987b1e09d3c01a` |
| `configs/a100_qwen_priority_full.json` | `f53329c66cd3d15be624421e3e51e68c3b2562dab07ed4537b6664f362cc30e3` |
| `configs/a100_campaign.json` | `2a50996b36680ecccfa40fab50a098a8e4ba149896f9cc3d095d5bde4853a702` |
| `configs/a100_qwen_7methods_pilot.json` | `81f3500ccf3af02087002f0dc36448c928e6f085af9714e32f10623058c14479` |
| `configs/a100_qwen_7methods_full.json` | `265daeb1a119b414791b6f05649af59853e7b261f326b6e0dfd8a45309493a6e` |
| `requirements-agent-cpu.txt` | `be597bd88bcc2ad9244c4bed75f658adc38d703ab03642d6a3508eab7af94b87` |
| `requirements-agent-gpu.txt` | `9f1662d57403dc76598a0372deff1379b36acddf4e82f353514838de6c92aa10` |
| `requirements-colab.txt` | `f34c503f03aeeba69f204ec373d80ba1671aa9e73aab544a121f538d58c155d6` |
| `pyproject.toml` | `f0c88046636de8ce956e495413e2052ceb74d3a79316b900cac04eddda0d35d0` |
| `uv.lock` | `382a328f6bd29f2618977de9d518a340357d4670acddfa6fda78843a9e10744d` |
| `src/vlmbench/hf_sources.py` | `e9161d66a05a75fd0316378c53b9f6c0e4d67f6c9ca35e7a23232f1f73fcb44f` (at `ebbf60d`, unchanged) |
| `scripts/colab_agent.py` | `dc0c0c57038314af42f15cdc6b0ae13af98bb7e53dd6859c1b0062468cbd5e54` |
| `scripts/colab_agent_remote.py` | `332d8a77e4dd315debeb5252cfaa6e164ce2e1470a28c04d230c2d67d4ae40d0` |

Re-verify with `sha256sum` before use. These hashes describe the audited files only.

## 6. Pins and inputs NOT present in the repository

Status tokens: **UNPINNED** = no revision in the config, so the run cannot start. **BLOCKED** = not in the config and not resolved offline; no value is invented. The generated table is [PINNED_INPUTS.md](PINNED_INPUTS.md), produced by `python -m vlmbench.repro_manifest` and checked by `tests/test_repro_manifest.py`. Note: the Hub answered a status request from this host during round 2. That is not a pin. Only an owner-approved commit SHA closes these rows.

| Input | Status | Required before any GPU action |
|---|---|---|
| HF model revision, `Qwen/Qwen3.5-0.8B` | **UNPINNED** (config `revision`; generated in `PINNED_INPUTS.md`) | Owner-approved commit SHA, written to `.hf_source.json` and to the campaign `models[].revision` |
| HF model revision, `Qwen/Qwen3.5-4B` | **UNPINNED** (same generated table) | same |
| HF dataset revision, `xiyuRenBill/MEMLENS` | **BLOCKED**: absent from configs; not resolved offline, so no value is given | same. Resolve the owner-name mismatch with the GitHub link first (ledger F-13). |
| HF dataset revision, `yubo2333/MMLongBench-Doc` | **BLOCKED**: same as above | same |
| Per-shard SHA-256 of the checkpoint | Not recorded (only size and count, `preflight.py:44-63`) | Run `scripts/audit_safetensors_payload.py --checkpoint-dir <pinned> --hash-shards` on CPU; record the output (ledger F-14) |
| Dataset file SHA-256 (`dataset_32k.json`, QA file, image archive) | Recorded by `prepare` (`hf_sources.py:165-178`) | Keep with the run manifest |
| `LICENSE-DATA` text and SHA | Not recorded | Record (F-15) |
| Drive paths | `REPLACE_DRIVE_PATH` placeholders in committed configs (intended) | Filled in a private copy only (`configs/README.md`) |
| GPU SKU | Not observed | Verify `nvidia-smi` reports A100 40 GB before any run (`config.py:63`) |
| vLLM | Pinned in config as `0.29.0`, **not installed** here | Verified in `environment.json` on the GPU session; a mismatch refuses the run (`cli.py` `_installed` check) |
| torch | Not pinned in any agent requirement file | Pin in the GPU profile before paid runs |

## 7. Runtime manifest (what a GPU run must write)

A run is reproducible only if these exist under the run root and on Drive:

| File | Must contain |
|---|---|
| `frozen_config.json` | Campaign config; fingerprint must match on resume (`cli.py` `run_job`) |
| `jobs/<job>.json` | Native and derived dataset SHA-256, `derivation` label (F-09), output contract, `config_sha256`, GPU inventory |
| `environment.json` | GPU, torch, vLLM version, httpx, Python, `repo_source_sha` (**must not be `UNRECORDED`** for a paid run) |
| `model_manifests/<model>.json` | Checkpoint HF id, revision, per-file list with SHA-256 from staging |
| `launch/<job>.json` | Server argv, `startup_status`, and the speculative config for MTP |
| `cells/<cell>.json` | Status, metrics, `sample_id_sha256`, `dataset_sha256`, `code_sha256`, `derivation`, notes (`92c8b68`, `573be47`) |
| `requests/<cell>.jsonl` | Per-request timings including `client_queue_wait_ms` |
| `preflight/<job>.workload.json` | Eligible unique prompts per text bin and CCU |
| `run_status.json` | `status`, `exit_code` (0 / 2 / 4 / 5 / 6), `counts` {planned, attempted, completed, non_completed}, `requeued_nonpass` (F-05, F-06; `00260cc`) |
| `RUNBOOK.md`, `summary.json` | Generated by `report`; non-PASS metrics are shown with their status (F-11) |

## 8. What this manifest does not establish

- No dataset counts, checksums, or licence status are verified against the publisher.
- No GPU behaviour, throughput, latency, memory, or MTP correctness is measured.
- No team-level scientific approval is claimed. Independent review is still required for the metric-definition changes (ledger §7).
