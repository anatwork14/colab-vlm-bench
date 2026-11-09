# Numeric integrity ledger — benchmark engineering (TASK C, wF:p8)

**Audited base:** `cfccc4b` (PR #20 head, branch `feat/qwen35-pretrained-validation`).
**Code fixes:** `ebbf60d` (regression tests and fixes), `07b64c9` (telemetry trust fields in cell receipts). This document: see the commit that adds it. Test counts below are at `07b64c9`.
**Scope:** `datasets/`, `quality.py`, `analysis.py`, `report.py`, `bench.py`, `telemetry.py`, `workloads.py`, `runbooks/`, `docs/METRICS.md`, plus the agent-CLI Colab path (`scripts/colab_agent*.py`, `agent_pipeline.py`, `cli.py run-job`, `preflight.py`, `hf_sources.py`, `derivation.py`, `notebook_bridge.py`). p7-owned engine and hallucination files are cited read-only.
**Verdict:** The serving-measurement and proxy-scoring code is now consistent with its stated definitions for the items fixed below. **No GPU number exists in this repository.** Nothing here is a GPU performance result, a scientific PASS, or team approval.

Status vocabulary: `FIXED` (code changed, regression test added), `OPEN-OWNER` (needs a change in a file owned by another worker or by the lead; exact diff requested in §5), `OPEN-DOC` (documentation or label issue), `NOT_TESTED` (requires GPU or real dataset files).

---

## 0. Test evidence

| Run | Command (venv `colab-vlm-bench/.venv/bin/python`, `PYTHONPATH=src`) | Result |
|---|---|---|
| Baseline, 5 named modules at `cfccc4b` | `-m pytest -q tests/test_metrics.py tests/test_datasets.py tests/test_notebooks.py tests/test_numeric_gate_invariants.py tests/test_numeric_artifact_provenance.py` | 40 passed, 8 skipped |
| Full suite at `cfccc4b` | `-m pytest -q` | 272 passed, 1 failed, 10 skipped (failure: `test_multimodal_fixture.py::test_progressive_qualification_test_c_multimodal_execution`, p7-owned, pre-existing) |
| New regression tests vs `cfccc4b` | same file, run against a `git archive` copy of `cfccc4b` in scratch (imports relaxed to `None` for the two new helpers) | **10 fail, 1 passes** (`test_stream_completion_is_service_timing_only` is a guard that passes on both by design). Five of the ten fail only because the new helper is absent; §6 says which. |
| New regression tests at `ebbf60d` | same file | 11 passed |
| 5 named modules at `ebbf60d` | same command | 40 passed, 8 skipped (unchanged) |
| Full suite at `ebbf60d` | `-m pytest -q` | **283 passed, 1 failed, 10 skipped**. The failure is the same pre-existing p7 test. |
| Full suite at `49e3d58` (round 2 HEAD) | `-m pytest -q` | **340 passed, 1 failed, 10 skipped** (588 s under load 16-21). The failure is the same p7 test (`CCU_ISOLATION_PASS` gate). |

**Flake analysis (round 2, item 5): UNEXPLAINED, not reproduced.** At `ebbf60d` one full run showed 3 failures (two in `tests/test_real_engine.py`); the tracebacks were not captured, so the cause cannot be identified from the logs. Three further full runs at `dd6df27` (same code as the round-1 tree) gave the same single deterministic failure each time: `test_multimodal_fixture.py::test_progressive_qualification_test_c_multimodal_execution` (`CCU_ISOLATION_PASS` = `GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED` != `PASS`), 283 passed, 10 skipped; wall times 453.6 s, 256.3 s and 56.4 s under host load 6-16 on 8 cores. Hypothesis checked: `tests/test_real_engine.py` calls `torch.randn` without a seed at 13 sites, and some assertions depend on the random values (`logical_selection_fraction > 0`, `grounded_citations > 0`, `max_abs_logits_diff <= 1e-5`, `physical_bytes_freed > 0`). Seeded runs of that module (a scratch-only pytest plugin sets random, numpy and torch seeds per test; no repository file changed): seed 0 10 passed (293 s), seed 1 10 passed (251 s), seed 2 10 passed (587 s). The unseeded-RNG hypothesis is therefore not supported, but 3 seeds do not rule it out. Timing hypothesis: the assertions that compare latency are `> 0.0` only, which load cannot plausibly break. No timeout or shared global state was found in the engine tests. Verdict: the 3-failure run stays **unexplained**. Owner p7 should keep the traceback capture (`-rfE`, full `-vv` output on failure) on any future full run. Treat `test_real_engine.py` as a watch item, not a known cause.

**Skips are real gaps:** the 10 skipped tests are all legacy notebook tests (`test_notebooks.py` ×8) and `test_stepwise_audit.py` ×2, skipped with "Legacy uploadable notebooks retired". CI runs `pytest -q` and reports green with these skipped. See F-19.

---

## 1. C1 — Dataset inventory (active set only)

The active set is taken from the priority launch configs (`configs/a100_qwen_priority_{pilot,full}.json`, `configs/a100_campaign.json`, `configs/a100_qwen_7methods_*.json`): **MemLens** and **MMLongBench-Doc**. AMBER is implemented (`datasets/amber.py`, 777 lines) and is accepted by `config.validate` as an alternative pair, but no active launch file uses it. It is excluded from this inventory.

**Important limit:** the repository contains **no dataset files, no dataset receipts, and no counts measured from data.** Every count below is a documentary claim, or a value that would be produced at run time. Nothing has been verified against the publisher.

| Field | MemLens (`memlens`) | MMLongBench-Doc (`mmlongbench_doc`) |
|---|---|---|
| Official source (code) | HF dataset `xiyuRenBill/MEMLENS` (`src/vlmbench/hf_sources.py:17`) | HF dataset `yubo2333/MMLongBench-Doc` (`hf_sources.py:18`) |
| Official source (docs) | GitHub `xrenaf/MEMLENS` (`README.md:77`) | GitHub `mayubo2333/MMLongBench-Doc` (`docs/PROTOCOL.md:29`) |
| Provenance check | **OPEN-DOC.** The owner name in the GitHub link differs from the HF repo owner in each case (`xrenaf` vs `xiyuRenBill`; `mayubo2333` vs `yubo2333`). Confirm which upstream is authoritative before citing either. | same |
| Revision | **Not pinned in repo.** `fetch_*` resolves the Hub `main` commit when no `--*-revision` is passed (`hf_sources.py:31-53`), and writes it to `.hf_source.json`. Campaign configs contain `REPLACE_DRIVE_PATH` placeholders; execution validation requires real Drive paths (`config.py:152-155`). | same |
| Artifact + checksum | Native JSON `dataset_32k.json`: SHA256 recorded (`hf_sources.py:173`). Image archive `release_images.tar.gz`: SHA256 recorded in `.image_unpack_complete.json` (`hf_sources.py:165-170`). Normalized JSONL: SHA256 in `source_*.json` (`agent_pipeline.py:221`). | QA file: SHA256 (`hf_sources.py:178`). Source PDFs and rendered page JPEGs: **no per-file SHA**. |
| License | `LICENSE-DATA` is downloaded (`hf_sources.py:155`) but its text and SHA are not recorded in any receipt. **OPEN-DOC.** | Recorded in docs only as "research non-commercial license" (`docs/PROTOCOL.md:29`). Not in any receipt. |
| Split | Single file, no split field. | `_qa_file` takes the first `*.parquet` (else `*.json`) and **prefers a stem containing "train"** (`hf_sources.py:134-141`). A heuristic, not an explicit split name. Recorded only as `qa_path`. |
| Eligible count (claimed) | 789 questions (`README.md:77`, `docs/AGENT_COLAB_CLI.md:59`). **Unverified.** | 1,091 QA, 135 documents (`README.md:78`). **Unverified.** |
| Native quality total (claimed) | 789 + 1,091 = 1,880 per configuration × 4 configurations = 7,520 (`docs/RESOURCE_BUDGET.md:15,28`). Arithmetic is consistent. | same |
| Serving eligible per bin | No receipt exists. Counts come only at run time: `build_serving_candidates` skip reasons (`notebook_bridge.py:170-181`), `exact_bin_pool` (`workloads.py:44-63`), and `serving_readiness` (`preflight.py:176-230`). | same |
| Missing images / PDFs | **Fail closed.** Missing or escaping image raises (`datasets/memlens.py:81-84`). No silent omission. | Missing PDF raises during normalization (`datasets/mmlong.py:60-62`). Preflight checks only the **first 16 rows** (`preflight.py:83-88`, `probe_rows=16`), so a partial sample is not a full presence check. |
| Duplicates | Duplicate `question_id` raises (`memlens.py:55-56`). | Duplicate `(doc_id, question)` raises (`mmlong.py:64-66`). Two different answers to the same doc+question are rejected as duplicates. |
| Leakage (gold in request) | Gold is stored in `reference`, outside `messages` (`memlens.py:98`). `ensure_no_gold_in_request` checks **top-level payload keys only** (`workloads.py:108-113`). Structural, not semantic. | same (`mmlong.py:103`) |
| Derived serving leakage | Derived rows keep the same question and a **tail** excerpt of history. They set `evidence_may_be_dropped: True` (`memlens.py:269`). They are load workloads, not QA. | Derived rows use the full-document text tail plus **one** page image (`notebook_bridge.py:110-115`). Not official QA. |
| Metric | Native: publisher LLM judge, **NOT_MEASURED** (`report.py:96`, `bench.py:369`). Proxy EM/F1: auxiliary only (`quality.py`). | Native: publisher F1 pipeline, **NOT_MEASURED**. Proxy EM/F1: auxiliary only. |
| Scorer | External upstream. `import_official` only wraps the payload and returns `QUALITY_OFFICIAL_IMPORTED_UNVERIFIED` (`quality.py`, end of file). It does not verify scorer parity or denominators. | same |

**C1 conclusion:** Normalization is fail-closed for missing media and duplicate IDs. The gaps are provenance (repo owner mismatch), unpinned revisions, missing license and per-file checksum receipts, a heuristic QA split, and no measured counts. None of these can be closed without the real files.

---

## 2. C2 — Numeric integrity ledger

Column key: **Def** = definition in code. **Num/Den** = numerator and denominator. **Agg** = aggregation. **Excl** = exclusions and failure handling. **Raw** = raw source. **Calc** = calculation site. **Lim** = limitations and status.

| # | Metric | Def | Num / Den | Agg | Excl / failure | Raw | Calc | Lim / status |
|---|---|---|---|---|---|---|---|---|
| 2.1 | Proxy EM | `exact_match`: NFKC-lowercased `\w+` token lists equal | matches / **all examples with a reference** | mean | Missing prediction → 0 (**FIXED F-01**). Non-`SUCCESS` → 0. Duplicate or unknown IDs raise (**FIXED F-01**). Reference-less examples counted, not scored. | `quality/*.jsonl` (`bench.py:305`) | `quality.py:48-100` | Proxy only, `not_official: True`. Not SQuAD-style normalization. Official accuracy `NOT_MEASURED` (`bench.py:369`). |
| 2.2 | Proxy token F1 | bag-of-words overlap F1 over `\w+` tokens | — | mean over the same denominator | same as 2.1 | same | `quality.py:35-46` | Order-insensitive. Empty-vs-empty = 1.0. Proxy only. |
| 2.3 | Grounding / citation / unsupported / contradiction / answer-hallucination rates | Formulas in `hallucination.py:117-138`; citation recorded only if the winning token is grounded (`grounding_evidence.py:210-218`, p7) | e.g. `UNSUPPORTED / evaluable claims`; `hallucinated answers / evaluable answers` | ratio | `None` when denominator is 0 (no zero fill) | external judge annotation JSONL | `hallucination.py` (p7) | **Not computed by the serving pipeline**; `NOT_MEASURED`. Formulas match `docs/METRICS.md`. Requires independent judge labels. p7 owns the code. |
| 2.4 | MTP proposal match rate | `mtp_matched_total / mtp_proposed_total` (`real_engine.py:1432-1434`, p7) | matched proposals / proposed | ratio | `None` if none proposed | engine trace | `real_engine.py` (p7) | **Diagnostic only**, not acceptance (documented). |
| 2.5 | MTP acceptance rate | **Hard-coded `0.0`** (`real_engine.py:1431`, p7); serialized as a float (`real_engine.py:1466`) | — | — | — | — | — | **Sentinel, not a measurement.** A report would show 0% acceptance. **OPEN-OWNER F-16** (p7): emit `null` / `NOT_MEASURED`. Bench already reports `"NOT_MEASURED"` (`bench.py:275`). |
| 2.6 | DA attention-mask restriction ratio | Disallowed keys / eligible keys, final decode row, logical only (`declarative_attention.py:231`, p7) | disallowed / eligible | ratio | — | attention mask | p7 | Logical restriction, **not KV read bytes, not time**. Physical field fixed at 0 (`declarative_attention.py:233`, `real_engine.py:1516`). Labelled as such in code and docs. |
| 2.7 | H2O logical fraction | `logical_selection_fraction = evicted_count / current_length` (`h2o.py:214`, p7) | evicted / current length | averaged over full-attention layers, last decode step | — | step trace | `adapters.py:379` (`h2o_retained_ratio = 1 - fraction`) | **OPEN-DOC F-17** (p7): the name says "selection" but the value is the **evicted** fraction. The retained ratio is arithmetically correct. |
| 2.8 | H2O retained KV / physical memory | Legacy `physical_kv_bytes_freed` = sum of per-step KV tensor payload deltas (`real_engine.py:1479-1484`, p7). Peak CUDA allocated/reserved saving = `null` | — | sum | — | — | p7 | **Not peak VRAM**. The doc calls it deprecated. The bench never reports it. Peak VRAM from sampled nvidia-smi is separate (2.13). **NOT_TESTED** on GPU. |
| 2.9 | TTFT | HTTP send → first non-empty `delta.content` chunk | — | p50/p95/p99 over `HTTP_OK` rows with a first chunk | Non-`HTTP_OK` rows excluded (`bench.py` valid filter). Warmups excluded. | per-request JSONL `requests/*.jsonl` | `bench.py:100` (`stream_completion`) | **Service-side, excludes client CCU-slot wait** (documented, `docs/METRICS.md`). Includes network, not server-side. With n=16 (CCU 1), p95/p99 are interpolated near the maximum; see F-12. |
| 2.10 | TPOT | `(ended − first chunk) / (usage.completion_tokens − 1)` | — | p50/p95 | Excluded if `output_tokens ≤ 1` or usage missing | same | `bench.py:102-106` | Uses server usage, not stream-chunk count. `ended` is after `[DONE]`. |
| 2.11 | E2E | **FIXED F-02.** Now from submission, before CCU-slot wait, to completed stream | — | p50/p95/p99 | Non-`HTTP_OK` excluded | same | `bench.py:32-45` (`admitted_stream`), `bench.py:101` | Includes client slot wait; **excludes server-side queue** (vLLM scheduler not instrumented). Before the fix, a CCU=1 second request reported 201 ms after a 200 ms queue. |
| 2.12 | Client queue wait (new) | Submission → CCU-slot acquisition | — | p50/p95 | Non-`HTTP_OK` excluded | same | `bench.py:32-45`, metric keys `client_queue_wait_p50/p95_ms` | New in `ebbf60d`. vLLM-side `queue_wait_ms` stays `NOT_MEASURED` (`bench.py:276`). |
| 2.13 | Peak GPU memory | Max over samples of `Σ memory.used` over visible devices (MiB) | — | max | **FIXED F-03**: multi-GPU output no longer dropped. `parse_failures`, `query_failures` reported. | `telemetry.py` samples | `telemetry.py:12-30`, `report()` | Sampled every 0.5 s: transient peaks between samples can be missed. Includes CUDA context and allocator cache. **Not** allocated or KV bytes. Single-GPU Colab assumed. |
| 2.14 | Throughput (output tok/s) | `Σ output_tokens(HTTP_OK, usage == 256) / (measured-phase wall time)` | valid output tokens | ratio | Failed requests' time stays in the denominator (intended: wall clock) | same | `bench.py:236,262` | Only `PASS_MEASURED` supports claims (`analysis.py:31-33`). **But** non-PASS rows still show metrics in the RUNBOOK table (`report.py:31`) — F-11. |
| 2.15 | Requests/s | `len(valid) / wall time` | — | ratio | same | same | `bench.py:263` | same |
| 2.16 | Offered CCU | Configured `concurrency` (client semaphore) | — | — | — | — | `bench.py` (`semaphore`) | Client-side offered load only. **Not** server-observed concurrency (`active_gpu_sequences` `NOT_MEASURED`). |
| 2.17 | Matched Base/MTP speedup | Per repetition: `throughput_mtp / throughput_base` on identical `sample_id_sha256`, both `PASS_MEASURED`; cohort median | pairs | median over repetitions | Non-PASS or sample-ID mismatch excluded and counted (`analysis.py:18-70`) | `cells/*.json` | `analysis.py:18-70` | Only 3 repetitions. Bootstrap CI of 3 values is **descriptive only** (documented). |
| 2.18 | Physical KV bytes; MTP acceptance (bench) | Constants `"NOT_MEASURED"` (`bench.py:275-276`) | — | — | — | — | — | Correct: not inferred. |
| 2.19 | Host RAM / local SSD | Sampled every 4th GPU sample (count-based) | — | min / peak | — | `telemetry.py:68-76` | host psutil, `/content/vlmbench` | Cadence shifts if nvidia-smi fails. Minor. |
| 2.20 | Official MemLens / MMLong score | — | — | — | — | — | — | **NOT_MEASURED** everywhere (`report.py:96-102`). Correct. |

**Simulated / CPU / theoretical numbers labelled as GPU performance:** none found in the bench, report, runbook or analysis path. Every serving metric is either a `PASS_MEASURED` receipt value or `NOT_MEASURED`. Two p7 items need labelling changes: the `0.0` acceptance sentinel (2.5) and the legacy `physical_kv_bytes_freed` field name (2.8). The DA ratio (2.6) is labelled logical. The receipt `artifacts/*.json` and `tests/test_numeric_artifact_provenance.py:25` retain an invalidated historical acceptance value (0.4), which is marked historical.

---

## 3. C3 — Fair-comparison confounders across arms

Scope: the two active arms, Base and native MTP. The five external arms (H2O, DA, ECRD and their MTP combinations) are not wired into `bench.py`/`cli.py run-job` (`config.validate` requires `modes == {base, mtp}`, `config.py:86-87`). They are not measured by this path.

| # | Confounder | Evidence | Effect | Status |
|---|---|---|---|---|
| C3-1 | **Identical request payloads**: same `request_payload`, `temperature 0`, `top_p 1`, `ignore_eos`, `min_tokens = 256`, same `include_usage`, same messages | `bench.py:115-132`, `config.py:67-70` | Controlled | Controlled |
| C3-2 | **Identical server flags except speculation**: `server_argv` differs only by `--speculative-config` for MTP | `engine.py:32-57` (p7, read-only) | Controlled. The argument list is identical other than the speculation block. | Controlled |
| C3-3 | **Same prompts per cell**: `cell_seed` is independent of mode; `sample_unique` over the same pool; `sample_id_sha256` checked in pairing | `config.py:164-167`, `workloads.py:66-71`, `analysis.py:33-36` | Paired samples | Controlled |
| C3-4 | **Pool depends on model tokenizer**: candidate bins are model-specific, so Base and MTP of the same model see the same pool, but **different models see different prompts** | `workloads.py:44-63` | Cross-model comparisons are not prompt-matched. Documented in `docs/METRICS.md`. | Documented |
| C3-5 | **Run order is not interleaved**: `run_job` runs one model×mode per invocation, in config order. Base and MTP are separate vLLM launches, run sequentially. | `cli.py:244-300`, `agent_pipeline.py:351-375` | Temporal confound (thermal state, Drive I/O, background load) is not randomized or alternated. | **OPEN-DOC**: interleave `base → mtp` per cell pair and record launch timestamps and order. |
| C3-6 | **Asymmetric failure coverage**: a cell with any receipt file is skipped on resume, **regardless of status** (F-05). A failed Base cell is never retried, while an MTP cell may fail under extra GPU memory pressure. | `cli.py:262-273` | Survivorship bias. Failure-prone arms lose cells, and `paired_speedups` drops the pair. Capacity failures are exactly what the design must report. | **OPEN-OWNER F-05** (diff in §5). |
| C3-7 | **MTP reserves extra GPU memory** under the same `gpu_memory_utilization = 0.88`. The vLLM KV-cache capacity differs between arms, so admission and queueing at CCU ≥ 8 differ. | `engine.py:46` (same flag, different free memory) | Throughput and tail latency at high CCU are confounded by KV capacity, not only by speculation. The vLLM "GPU KV cache size" log line is not parsed. | **OPEN-DOC**: parse and record KV-cache size and max concurrency per launch, from `server.log`. Not implemented. |
| C3-8 | **Greedy outputs may differ** between Base and MTP (speculation is not guaranteed bitwise-equal). Token counts are forced equal by `min_tokens`. | `docs/METRICS.md` | Latency is comparable. Output text is not guaranteed identical. | Documented |
| C3-9 | **Two MemLens derivation routes**: `memlens.py:201` joins with `" "` and keeps the **last** image (`memlens.py:194-196`); `notebook_bridge.py:90` joins with `"\n"` and keeps the **first** image. `agent_pipeline.py:198-205` picks the route by `source_mode`. Token counts for bins are computed with `" "` (`derivation.py:138`). | as cited | The same "memlens" label can mean different prompts. Bins can shift by a few tokens. | **OPEN-OWNER F-09**. |
| C3-10 | **Serving bin not enforced for AMBER** (inactive): `bench.py:154-161` ignores `input_length`, and the receipt still records `text_token_bin_requested` (`bench.py:283`). | as cited | Would silently mix bins. Inactive. | **OPEN-DOC (latent, inactive):** gate before any AMBER use (F-10). |
| C3-11 | **Client-side only**: TTFT/E2E include client and network time. Same client config for both arms. | `bench.py:32-108` | Same bias both arms | Controlled |

---

## 4. C4 — Notebook / Colab readiness (CPU static smoke)

**Notebooks are retired.** `notebook/Qwen3.5-*.ipynb` are legacy (`AGENTS.md`, `docs/AGENT_COLAB_CLI.md:3`). Their tests are skipped (`test_notebooks.py:5`). The readiness surface audited is the agent CLI path (`docs/AGENT_COLAB_CLI.md`, `docs/AGENT_EXECUTION.md`, `runbooks/`). This is a **static** CPU audit. No Colab session, no A100, no Drive mount, no model load, no paid action.

| Check | Command / evidence | Result |
|---|---|---|
| Benchmark refused without `--approve-paid` | `python scripts/colab_agent.py benchmark --session vlmb-a100 --model qwen35_08b --dataset memlens --phase pilot --mode base` | `PermissionError: GPU inference requires explicit --approve-paid in each invocation` ✓ |
| Remote run-job refused without approval | `python -m vlmbench.cli run-job --config configs/a100_qwen_priority_pilot.json --model qwen35_08b --mode base --dataset memlens` | `PermissionError: GPU execution requires owner_approved_compute=True` ✓ |
| Execution validation refuses a dry-run config | `load_config('configs/a100_qwen_priority_full.json', execution=True)` | `ConfigError: dry_run must be false for paid GPU work` ✓ (as designed) |
| CPU preflight blocks a missing checkpoint | `python -m vlmbench.preflight --checkpoint /nonexistent/ckpt --scratch <scratch>` | `CPU_PREFLIGHT_BLOCKED`, `Missing checkpoint directory` ✓ |
| Pilot plan totals | `python -m vlmbench.cli plan --config configs/a100_qwen_priority_pilot.json` | 48 cells; 2,816 measured requests; 656 warmups; 720,896 nominal output tokens (= 2,816 × 256) ✓ reconciles with `runbooks/QWEN_PRIORITY.md` |
| Confirmatory plan totals | `plan --config configs/a100_qwen_priority_full.json` | 1,008 cells; 45,696 measured; 10,584 warmups; 11,698,176 output tokens (= 45,696 × 256) ✓ |
| Remote source integrity | `scripts/colab_agent_remote.py:35-40` checks bundle SHA256; `colab_agent.py:check_git_clean` refuses a dirty tree | ✓ for the agent path |
| Remote **legacy** path | `scripts/remote_entry.py:21-32` extracts `/content/vlmbench_checkout.zip` **without a digest check** and runs `pip install -e` on the runtime | **FIXED `4975562`** (F-07) |
| Persistence | `artifacts.sync_files` copies receipts to Drive and checks SHA256 (`artifacts.py:72-87`) | ✓ |
| Safe resume: source | `prepare` writes an intent before downloading and refuses drift (`agent_pipeline.py:159-167`, `hf_sources.py:31-53`) | ✓ |
| Safe resume: annotation | `annotate_with_backend` keeps a source-SHA state file and refuses changed sources (`derivation.py:95-108`) | ✓ |
| Safe resume: quality | `run_quality_predictions` checks IDs and order, and does not re-run completed receipts (`bench.py`, quality block) | ✓ code path; **no direct test** |
| Safe resume: serving cells | `run_job` skipped any existing cell receipt, regardless of status (`cli.py:262-273`, before `92c8b68`) | **FIXED `92c8b68`** (F-05) |
| Manifest | `frozen_config.json`, `jobs/*.json` (dataset SHA256 for native and derived), `environment.json` (vLLM, torch, `VLMBENCH_SOURCE_SHA` or `UNRECORDED`), `launch/*.json` (`cli.py` `run_job`) | ✓ with `UNRECORDED` allowed when the env var is unset (direct CLI) |
| Dependencies | `requirements-agent-cpu.txt` and `requirements-agent-gpu.txt` pin ranges, not a lock; `torch` is not pinned in any agent requirement file; `vllm==0.29.0` is pinned only in GPU/colab files. The audit venv has `huggingface_hub 2.4.0`, outside the `<2` pin, and **vLLM is not installed**. | **OPEN-DOC**: the GPU path is **NOT_TESTED** here. |
| Mount / auth | `colab_agent.py mount` is interactive by design | NOT_TESTED |
| Honest "finished" status | Before `00260cc`, `run_job` returned 0 when every cell failed, so `agent_pipeline.benchmark` returned `GPU_JOB_FINISHED…` and the remote wrapper returned `ACTION_COMPLETED` (F-06) | **FIXED `00260cc`** (F-06) |
| CI | `.github/workflows/ci.yml` runs `pytest -q` with Python 3.11 and CPU torch. Legacy notebook tests skip silently (F-19). | ✓ CPU only |

---

## 5. Findings

Severity: **H** = can produce a wrong or unsafe reported number or an unauthorized paid action. **M** = can mislabel or bias, or lose evidence silently. **L** = cosmetic or latent.

| ID | Sev | Status | Location | Evidence | Fix / proposal | Test |
|---|---|---|---|---|---|---|
| F-01 | H | **FIXED** `ebbf60d` | `quality.py:48-100` (was `:33-45`) | `score_proxy([a→June], [a→June, b→May])` returned `count=1, exact_match=1.0` at `cfccc4b` | Missing predictions score 0; duplicates and unknown IDs raise; reference-less counted | `test_measurement_regressions.py::test_missing_prediction_is_counted_as_zero_not_dropped`, `…duplicate…`, `…unknown_id…`, `…without_reference…` |
| F-02 | H | **FIXED** `ebbf60d` | `bench.py:32-45`, `:101` | CCU=1 second request: `e2e_ms=201` after a 200 ms queue (reproduced on `cfccc4b`) | `admitted_stream` adds slot wait to e2e; TTFT stays HTTP-send; new `client_queue_wait_p50/p95_ms` | `test_e2e_is_measured_from_admission_including_client_queue_wait`, `test_stream_completion_is_service_timing_only` |
| F-03 | M | **FIXED** `ebbf60d` | `telemetry.py:12-80` | `"1234, 0\n5678, 3"` → `ValueError` swallowed → zero samples, peak = NOT_MEASURED with no reason | `parse_nvidia_smi_csv` sums devices; `parse_failures` / `query_failures` / `gpu_count_max` reported | `test_multi_gpu_csv_sums_devices_instead_of_dropping_sample`, `test_malformed_csv_raises…`, `test_sampler_reports_parse_failures…` |
| F-04 | M | **FIXED** `ebbf60d` | `datasets/mmlong.py:74-76` | Render cache keyed by document only; 96-dpi run reused 72-dpi pages (`renders == [72]` on `cfccc4b`) | DPI in cache directory name | `test_mmlong_render_cache_is_keyed_by_dpi` |
| F-05 | H | **FIXED** `92c8b68` | `cli.py:262-273` | Both branches `continue`. A failed or stale receipt is never re-run, and a changed `dataset_sha256` is silently accepted. Makes the Base/MTP coverage asymmetric (C3-6). | Applied in `92c8b68` (`src/vlmbench/resume.py`, `cli.py`): only COMPLETED receipts are skipped; the rest stop before GPU (`BLOCKED_NONCOMPLETE_RECEIPTS`) unless `--requeue-nonpass` quarantines them. | proposed: `tests/test_run_job_preflight.py` |
| F-06 | H | **FIXED** `00260cc` | `cli.py:302`; `agent_pipeline.py:373`; `colab_agent_remote.py:86-91` | `run_job` returns 0 after the serving loop regardless of cell status. The orchestration layer reports `ACTION_COMPLETED` / `GPU_JOB_FINISHED`. | Applied in `00260cc`: exit codes 0/2/4/5/6 with named statuses in `run_status.json`; `benchmark` raises on any non-zero exit, so the wrapper returns `ACTION_FAILED`. | proposed |
| F-07 | M | **FIXED (legacy path kept, not recommended)** `4975562` | `colab_controller.sh:74-75`; `remote_entry.py:21-32` | `--publish` runs `git commit` **and `git push origin HEAD`** from a GPU helper. The legacy path has no bundle digest and runs `pip install -e` on the paid runtime. `docs/PRIORITY_EXECUTION.md:41` still documents the controller as active. | Applied in `4975562`: controller never commits or pushes; `--publish` refused; `remote_entry.py` checks `bundle_sha256` and uses PYTHONPATH instead of runtime `pip install -e`. | proposed (`bash -n` in CI only) |
| F-08 | M | **OPEN-DOC** | `cli.py:244-300`; `engine.py:46` | C3-5 and C3-7: no interleaving; KV-cache capacity not recorded per launch | Record `vLLM GPU KV cache size` / max concurrency from `server.log`; interleave `base`/`mtp` per cell pair | proposed |
| F-09 | M | **FIXED** `573be47` | `memlens.py:194-201`; `notebook_bridge.py:90,110`; `derivation.py:138`; `agent_pipeline.py:198-205` | Two MemLens derivation routes with different joins and image selection, both labelled `memlens` (C3-9) | Applied in `573be47`: one `derivation` label per derived file, enforced before GPU; the label is in the job manifest and receipts and is part of the resume identity. AMBER rows are unlabelled, so AMBER serving is refused until they are labelled. | proposed |
| F-10 | L | **OPEN-DOC** (latent) | `bench.py:154-161` | AMBER branch ignores `input_length`; receipt records the requested bin anyway | Enforce bin membership or refuse before any AMBER campaign | proposed |
| F-11 | M | **OPEN-DOC** (owned by me; low urgency) | `report.py:31` | Metrics of `PARTIAL_FAILURE` cells are copied into the serving table next to the status. Claims are protected because `paired_speedups` uses PASS only. | Show `NOT_MEASURED` for non-PASS metrics, or mark the status column explicitly | proposed |
| F-12 | L | **OPEN-DOC** | `bench.py` quantiles; `report.py` | At CCU 1, n = 16 measured requests; p95/p99 are interpolated near the maximum | Report `n` next to every percentile; do not claim p99 below ~100 samples | proposed |
| F-13 | M | **OPEN-OWNER** (rendering fixed `e4a30ae`) | `hf_sources.py:16-18,31-53,134-141`; `README.md:77-78`; `docs/PROTOCOL.md:29` | No committed revision pins; owner-name mismatch between HF and GitHub; heuristic QA split | Generated `docs/audit/PINNED_INPUTS.md` (`e4a30ae`) renders model revisions as UNPINNED and dataset revisions as BLOCKED. Owner must supply commit SHAs. | proposed |
| F-14 | M | **OPEN-DOC** | `preflight.py:44-63`; `hf_sources.py:118-131`; `artifacts.py:46-69` | Model source receipts record file count and bytes only. Per-file SHA256 only at staging. | Run `scripts/audit_safetensors_payload.py --hash-shards` on the pinned checkpoint before any GPU run (existing tool, not run here) | proposed |
| F-15 | L | **OPEN-DOC** | `hf_sources.py:155` | `LICENSE-DATA` downloaded, not hashed or recorded | Record SHA256 and text path in `.hf_source.json` | proposed |
| F-16 | M | **OPEN-OWNER p7** | `real_engine.py:1431`, `:1466`, `:870` | `mtp_acceptance_rate = 0.0` is a sentinel stored as a float | Emit `None` / `NOT_MEASURED` | p7 |
| F-17 | L | **OPEN-OWNER p7** | `h2o.py:214`; `adapters.py:379` | `logical_selection_fraction` holds the **evicted** fraction | Rename or document | p7 |
| F-18 | L | **OPEN-DOC** | `preflight.py:213` vs `bench.py` pool; `workloads.py:49-51` | Readiness hard-codes a 256-token reserve; it accepts only `vllm_effective_token_accounting`, while the bench pool also accepts `model_processor` (`workloads.py:49-51`). The two gates can disagree on eligibility. | Share one predicate | proposed |
| F-19 | L | **OPEN-DOC** | `.github/workflows/ci.yml`; `tests/test_notebooks.py`; `tests/test_stepwise_audit.py` | 10 legacy tests skip silently in CI | Delete or move to `tests/legacy/` with a clear marker | proposed |
| F-20 | L | **OPEN-DOC** | `agent_pipeline.py:189-205` | Historical MemLens route ignores the pilot phase's two bins and builds all seven; the candidate file differs by phase | Use `lengths` on every route | proposed |
| F-21 | L | **OPEN-DOC** | `preflight.py:83-88` | MMLong PDF presence check samples the first 16 rows only | Full presence check happens in normalization (`mmlong.py:60-62`). Document as sample-only. | — |

### 5.1–5.3 Applied in round 2 (replaces the proposed diffs)

- **F-05 (`92c8b68`).** Each receipt is classified COMPLETED, MISSING, FAILED, STALE or INVALID (`src/vlmbench/resume.py`). COMPLETED requires PASS_MEASURED plus matching config, dataset, code (`VLMBENCH_SOURCE_SHA`; `UNRECORDED` never certifies), derivation, execution pass, output contract, cell, sample-set digest and measured request count. Any other receipt stops the job before GPU start (`BLOCKED_NONCOMPLETE_RECEIPTS`, rc 2) unless `--requeue-nonpass` is given. Requeue moves the receipt, local and Drive, to `quarantine/<stamp>/`. Nothing is deleted or rewritten. Tests: `tests/test_resume_policy.py`.
- **F-06 (`00260cc`).** Exit codes: 0 `COMPLETE` / `IN_PROGRESS`; 2 blocked; 4 pilot stop or repeated quality failure; 5 `ALL_ATTEMPTED_FAILED` / `QUALITY_ALL_FAILED`; 6 `PARTIAL_FAILURE` / `QUALITY_PARTIAL_FAILURE`. Counts go to `run_status.json`. `agent_pipeline.benchmark` raises with the status name on any non-zero exit, so the remote wrapper returns `ACTION_FAILED`. Tests: `tests/test_resume_policy.py` (F-06 section).
- **F-07 (`4975562`).** `colab_controller.sh` has no `git add/commit/push`. `--publish` is refused at argument parsing. The job carries `bundle_sha256` of the uploaded zip. `remote_entry.py` verifies that digest before extraction and runs with `PYTHONPATH`, with no runtime `pip install`. **Legacy gap, documented:** the path still has no independent review and is not used for paid runs. `runbooks/GPU_RUNBOOK.md` G-C says so. Tests: `tests/test_gpu_helper_safety.py`.
- **F-09 (`573be47`).** `src/vlmbench/derivation_labels.py` defines one constant per route. `run-job` refuses a derived file with zero, mixed or unlabelled rows (`BLOCKED_MIXED_DERIVATION`, rc 2) before GPU start. The label is recorded in the job manifest and in every receipt, and it is part of the resume identity, so another route's receipts are STALE. Tests: `tests/test_derivation_routes.py`, `tests/test_resume_policy.py`.

## 6. Regression tests added (`tests/test_measurement_regressions.py`)

Run against a `git archive` of `cfccc4b` (scratch copy, imports relaxed for the two new helpers): 10 fail, 1 passes.

| Test | Result at `cfccc4b` | Why |
|---|---|---|
| `test_missing_prediction_is_counted_as_zero_not_dropped` | FAIL | `count == 1`, EM 1.0 (reproduced directly) |
| `test_duplicate_prediction_id_is_rejected` | FAIL | no error raised |
| `test_prediction_for_unknown_id_is_rejected` | FAIL | no error raised |
| `test_example_without_reference_is_reported_not_silently_dropped` | FAIL | no `unscored_without_reference` key |
| `test_single_gpu_csv_parses` | FAIL | helper `parse_nvidia_smi_csv` absent (structural, not a behaviour reproduction) |
| `test_multi_gpu_csv_sums_devices_instead_of_dropping_sample` | FAIL | helper absent; the old inline parse raises on the same input (reproduced: `invalid literal for int() … '0\n5678'`) |
| `test_malformed_csv_raises_so_sampler_can_count_failures` | FAIL | helper absent |
| `test_sampler_reports_parse_failures_instead_of_silent_drop` | FAIL | no `parse_failures` key |
| `test_e2e_is_measured_from_admission_including_client_queue_wait` | FAIL | helper `admitted_stream` absent. The old semantics were reproduced separately: second request `e2e_ms = 201` after a 200 ms queue. |
| `test_stream_completion_is_service_timing_only` | **PASS** | guard for the unchanged service-timing contract |
| `test_mmlong_render_cache_is_keyed_by_dpi` | FAIL | `renders == [72]` (reproduced: the 96-dpi run reused 72-dpi pages) |

The five helper-absent failures show the tests need the new functions; the behavioural reproductions for F-01, F-02 and F-04 come from the separate runs listed above. At `ebbf60d`, all 11 pass. The timing test uses 0.2 s sleeps with 150–350 ms thresholds to avoid flakiness on a loaded host.

---

## 7. Limitations and what is still required

- **No GPU evidence exists.** Every serving metric remains `NOT_MEASURED` until an approved pilot writes `PASS_MEASURED` receipts.
- **No real dataset files were available.** Counts, checksums and bin eligibility in §1 are claims or run-time outputs, not verified values.
- **p7 and unowned files** (`cli.py`, `agent_pipeline.py`, `preflight.py`, `derivation.py`, `notebook_bridge.py`, `hf_sources.py`, `scripts/`) are read-only in this task. Their findings are diff requests (§5).
- **Independent review is required** for changes to time and metric definitions (`AGENTS.md`, "Ordering / ownership" step 2). The E2E definition change (F-02) and the proxy denominator change (F-01) should go to an independent reviewer. I have not claimed approval.
- The audit environment is not the declared CPU profile (`huggingface_hub 2.4.0`, no vLLM). Test results reflect this venv.
