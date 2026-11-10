# Baseline audit: Phase 0.1 (HEAD `e83a3c8`)

| Field | Value |
|---|---|
| Task | TASK-STARLIGHT-REVIEW-PHASE0-1 |
| Reviewer | argon (Claude Opus 5.5), independent scientific reviewer |
| Commissioned by | agy-orchestrator (wF:p1), relayed by the owner |
| Audited revision | `e83a3c8` (Merge PR #19), `main` == `origin/main` |
| Date | 2026-11-09 |
| Method | Read-only source and doc inspection; CPU test run on host (GTX 1080 Ti present); no Colab, Drive, GPU or network actions |
| Review independence | The reviewer wrote none of the code under review and will not implement fixes it later approves |

Severity: **P0** means the study design can't run or its results would be invalid. **P1** means a result or cost-safety claim can't be trusted until fixed. **P2** is a quality, efficiency or reproducibility improvement. Component verdicts are **PASS**, **CONDITIONAL** (usable with the listed conditions) or **BLOCKED**.

Several criteria below propose numeric thresholds. These are marked *(proposed)* and need owner sign-off before they become binding.

---

## 1. Summary

The HEAD code is careful about cost gates, durable receipts and fail-closed serving accounting. However, **it doesn't implement the owner-selected study design**. `docs/EXISTING_DRIVE_LAYOUT.md` §5 records that design as **AMBER (quality + serving)** plus **MemLens-derived seven text bins (context scaling)**. The code hard-wires **MemLens + MMLongBench-Doc** everywhere (`config.DATASETS`, `hf_sources.DATASETS`, `methods.validate_method_manifest`, `notebook_bridge.build_serving_candidates`). No AMBER adapter, prompt set, scorer or source pin exists. The document states this gap itself.

MTP is launched but never verified as active. Quality predictions can't distinguish natural EOS from truncation at 256 tokens.

| # | Component | Verdict | Blocking findings |
|---|---|---|---|
| 1 | Source integrity | CONDITIONAL | RV-01, RV-10 |
| 2 | GPU cost gates and resource boundaries | CONDITIONAL | RV-04 |
| 3 | Multimodal preprocessing and Qwen token accounting | CONDITIONAL | RV-06, RV-07 |
| 4 | AMBER quality and serving | **BLOCKED** | RV-01, RV-02 |
| 5 | MemLens-derived 7 context bins | CONDITIONAL | RV-06, RV-07, RV-08 |
| 6 | Serving vs quality metric contract | CONDITIONAL | RV-02, RV-11 |
| 7 | MTP activation and verification | **BLOCKED** (for any MTP claim) | RV-03 |
| 8 | Reproducibility | CONDITIONAL | RV-05, RV-09, RV-10 |

## 2. Defect matrix

| ID | Sev | Component | Finding | Evidence / reproduction | Required fix (criteria) |
|---|---|---|---|---|---|
| RV-01 | **P0** | AMBER, source | The owner-selected AMBER dataset has no registry entry, source pin, adapter, prompts or scorer. Config validation rejects any dataset set other than `{memlens, mmlongbench_doc}`. | `src/vlmbench/config.py:19` `DATASETS = {"memlens","mmlongbench_doc"}` and `:87-88`; `hf_sources.py:17-18`; `methods.py:75`; `notebook_bridge.py:38`. `grep -ri amber src/` returns nothing. `docs/EXISTING_DRIVE_LAYOUT.md:99` states the gap. | Section 4 (A1–A8). The config schema must also encode dataset *roles* (AMBER = quality + serving; MemLens = serving-only context bins). |
| RV-02 | **P1** | AMBER, contract | The quality loop stores `prediction` and `usage` but not `finish_reason`. Natural EOS can't be told apart from truncation at the 256-token cap (`max_output_tokens: 256`). Truncated AMBER generative descriptions bias the CHAIR, Cover and Hal metrics. | `src/vlmbench/bench.py:250-262`: the result dict has no `finish_reason`. Run any quality job and inspect `quality/*.jsonl`. `usage.completion_tokens == 256` is only an indirect signal, because an answer can end naturally at exactly 256. | A3: record `finish_reason` for every item, report the truncation rate, apply the threshold gate. |
| RV-03 | **P1** | MTP | MTP is requested through `--speculative-config`, but nothing checks that speculative decoding was active, measures acceptance, or compares greedy outputs between Base and MTP. A silent fallback or misconfiguration would give MTP numbers that come from non-speculative decoding. | `engine.py:50-56` only builds the argv. `bench.py:202` hard-codes `mtp_acceptance_rate: NOT_MEASURED`. `cli.py:234` records `NEEDS_BACKEND_TRACE_REVIEW`. No `/metrics` scrape exists anywhere. | Section 7 (M1–M6). |
| RV-04 | **P1** | GPU cost | The consecutive-failure circuit breaker for native quality runs applies only to `engineering_pilot`. In `full_study` and `priority_confirmatory` scopes a quality invocation runs the whole dataset in one call, with a 900 s per-request timeout and no per-invocation item or wall-clock cap. A systematic fault (for example, every item timing out) could consume up to items × 900 s of A100 time. | `bench.py:249` (`httpx.Timeout(900.0)`), `bench.py:273-274` (breaker gated on `kind == "engineering_pilot"`). Serving has `max_cells ≤ 6` per call; quality has no equivalent. | G2–G3 in section 6. |
| RV-05 | **P1** | Reproducibility | `environment.json` (torch, vLLM, repo SHA), `jobs/<job_key>.json` and `model_manifests/<model>.json` are **overwritten on every invocation** and synced over the Drive copy. A full grid needs about 7 or more sessions per model × mode × dataset (6 cells per call). Cell receipts carry only config, dataset and cell, with no environment, repo SHA or session reference. So a cell can't be traced back to the driver, vLLM build or code revision that produced it. | `cli.py:207-222` writes fixed filenames. `bench.py:116-118` and `:207-224`: the receipt has no environment or repo fields. `sync_files` replaces Drive copies whose content differs (`artifacts.py:72-87`). | R1–R2 in section 9. |
| RV-06 | **P1** | Bins, token accounting | Each derived bin request has 1 real image plus N **text** tokens. Image token counts vary per native example (MemLens image resolutions differ; no pixel cap is set, see `engine.py` with no `--mm-processor-kwargs`). At low bins the effective prompt can be dominated by vision tokens, so the "128-token bin" may differ widely in real input length across requests. That confounds context-scaling claims. The protocol allows "128 text + vision payload", but the analysis doesn't use effective lengths. | `notebook_bridge.py:110-112` (first image of the history), `:153-158`. `bench.py:188-193` records only the effective p50/p95, and the analysis/report use the text bin as the x-axis. | B2 and B3 in section 5. |
| RV-07 | P2 | Bins, preprocessing | There are two divergent derivation implementations. `derivation.generate_candidates` uses a word-count approximation, the *last k* images and IDs `__t{n}`. The agent path's `notebook_bridge.build_serving_candidates` is tokenizer-exact, uses the *first* image and IDs `__text{n}`. The CLI and agent paths therefore produce different workloads from the same source. | `derivation.py:19-70` vs `notebook_bridge.py:21-181`. `agent_pipeline.py:110` uses the latter. | B6: one canonical derivation, with its ID recorded in every cell receipt. Retire the other or reject it at load time. |
| RV-08 | P2 | Bins, sample budget | `max_per_bin = 192` leaves only 32 rows of slack above the 160 (128 measured + 32 warmup) that a CCU32 cell needs. Annotation re-counts text with a different join (`" ".join` in `derivation.py:133-134`, vs `excerpt + "\n" + question` at construction), and `exact_bin_pool` re-applies ±5%. Rows that drift out, or fail annotation, can leave CCU32 cells `INSUFFICIENT_VERIFIED_UNIQUE_PROMPTS`. Repetitions also redraw from the same ~192-row pool, so the "3 independent repetitions" share most samples. | `notebook_bridge.py:28,40`; `workloads.py:44-71`. | B3–B4. |
| RV-09 | P2 | Reproducibility | `PROTOCOL.md` requires "randomized/counterbalanced order", but `planned_cells` iterates in a fixed nested order (rep → length → CCU). Base and MTP run as separate jobs in whatever order the operator picks. Thermal, driver or session drift is therefore confounded with mode. | `config.py:165-182`; `PROTOCOL.md:11`. | R4. |
| RV-10 | P2 | Source, reproducibility | `transformers>=4.57` is unpinned. The tokenizer that defines the text bins comes from it, and its version isn't recorded in the prepare manifest. The prepare manifest also lacks the repo or bundle SHA. | `requirements-agent-*.txt:8`; `agent_pipeline.py:118-131`. | R2. |
| RV-11 | P2 | Serving metrics | `request_payload` → `resolve_messages` reads and base64-encodes images synchronously on the event loop, inside the measured window for concurrently running requests. At high CCU this can delay other streams' chunk reads, inflating TTFT and E2E for some requests. | `bench.py:146-151`; `workloads.py:79-105`. | S5: pre-build every payload before `start_cell`. |
| RV-12 | P2 | Tests | On any host with a working `nvidia-smi`, `tests/test_colab_agent.py::test_pipeline_prepare_resumes_same_manifest_without_extra_hf_download` fails with `GPU_SESSION_REFUSED_FOR_PREPARE`. | Reproduced on this host (GTX 1080 Ti). | Fixed in sol-lead's slice (`wt/sol-lead-runtime`, verdict PASS). Pending integration. |
| RV-13 | P2 | GPU cost | `sync_files` re-hashes every file under the run root after **each** quality item, which is O(n²) hashing over Drive FUSE across a full quality run. That is A100 wall-clock time spent on I/O. | `bench.py:270`; `artifacts.py:72-87`. | Sync only the changed file per item; do a full sync at the end of the cell or job. |
| RV-14 | P2 | GPU cost | Annotation is limited to ≤128 requests per call, and each call boots vLLM. 7 bins × 192 rows needs about 11 or more boots per model × dataset just for annotation. | `scripts/colab_agent.py:74`; `agent_pipeline.annotate`. | Measure boot cost in the pilot; consider a larger per-call cap gated by an explicit owner budget. |
| RV-15 | P2 | Quality denominator | `FAILED` quality items are never retried on resume (resume continues from `rows[len(results):]`). A transient failure becomes permanent. | `bench.py:239-250`. | State the denominator policy (A7). Allow an explicit, receipted retry of failed IDs only. |

## 3. Already sound (verified, no action required)

- **Paid-action gates:** `--approve-paid` is required per invocation before any git, upload or Colab contact (`scripts/colab_agent.py:67-70`, ahead of `check_git_clean()` at `:71`). The remote side re-checks `owner_approved_compute` (`colab_agent_remote.py`). `prepare` refuses a GPU session. The A100 40GB identity check runs before weight staging, and CPU resource preflight runs before staging.
- **Source transfer:** `git archive HEAD` bundle, SHA256-verified on the remote, with a traversal- and symlink-safe extract.
- **HF pinning:** the first resolved 40-hex commit is pinned, and drift is refused (`hf_sources._resolve_pinned`). Model staging hashes every file before and after copying (`staged_copy`). vLLM `0.29.0` is pinned and checked at runtime.
- **Serving accounting:** a cell is `PASS_MEASURED` only when every measured request is HTTP OK **and** has `usage.completion_tokens == 256`. Warmup happens at the offered CCU and is invalidated on any mismatch.
- **Paired sampling:** cell seeds are independent of model and mode, and `analysis.py` pairs Base and MTP only when `sample_id_sha256` matches and both are `PASS_MEASURED`.
- **Payload separation:** serving uses `fixed_output=True`; quality uses `fixed_output=False` with the configured cap. Confirmed at the call sites and by sol-lead's new test.
- **Gold isolation (structural):** references live outside `messages`, and `ensure_no_gold_in_request` checks every request.
- **Upstream method pins:** H2O, DA and ECRD are pinned by commit, checkout verifies `HEAD`, and the arms stay `BLOCKED_EXTERNAL_ENGINE_PORT_REQUIRED`.

---

## 4. AMBER acceptance criteria (component 4, currently BLOCKED)

AMBER facts below (task types, metric names, prompt strings, file names) must be confirmed against the **pinned** upstream repo before implementation. This audit doesn't assert them from memory.

- **A1 Source pin:** pin the upstream AMBER repository by commit SHA. Produce a SHA256 manifest of every image, annotation file, query file and scorer resource (for example, safe-word and relation lists), and the licence. Compare the Drive `STARLIGHT/amber/_archives` and `official_code` trees against that manifest. Report `FOLDER_FOUND` → `SOURCE_HASH_VERIFIED` → `DATASET_SEMANTICS_VERIFIED` → `SCORER_PARITY_PASS` separately, as `EXISTING_DRIVE_LAYOUT.md` §6 requires.
- **A2 Prompts and IDs:** use the upstream query files unmodified, for both generative description and discriminative yes/no queries. Keep upstream query IDs. Each record must store the image SHA256. The frozen sample manifest (all IDs, or a seeded subset for the pilot) is hashed and referenced by every quality and serving receipt.
- **A3 Quality pass, natural EOS:** `temperature=0`, `top_p=1`, `enable_thinking=False`, `stream=False`, no `ignore_eos`/`min_tokens`, `max_tokens = 256`. Every item records `finish_reason`, `usage.completion_tokens`, `usage.prompt_tokens`, the response text and the query ID.
  - Report the **truncation rate** (`finish_reason == "length"`) per model × arm × task type.
  - *(Proposed)* If the generative task exceeds 2% truncation, the quality verdict is CONDITIONAL and must be disclosed. Above 5% it is BLOCKED until the owner raises the cap for a new campaign ID.
  - A cap change applies to all arms equally.
- **A4 Scoring:** use the pinned upstream AMBER scorer unmodified, or a port proven equivalent.
  - Report the generative metrics (CHAIR, Cover, Hal, Cog), the discriminative metrics (accuracy, precision, recall, F1) and the composite AMBER score as defined upstream.
  - Pin and record scorer NLP dependencies (spaCy model version, NLTK data).
  - **Parity test:** the scorer must reproduce an upstream reference output exactly before any model score is reported.
  - Use the upstream yes/no answer-extraction rule. Unparseable answers are counted and reported, never silently dropped.
- **A5 Serving pass, fixed output:** a separate track (`track=amber_serving`) that is **never scored**. Requests use `ignore_eos=True`, `min_tokens = max_tokens = 256` and `stream=True`, and a cell passes only if every request has `completion_tokens == 256`. AMBER prompts are short text plus one image, so the input is vision-dominated. Report effective `prompt_tokens` per request, and don't place AMBER cells on the MemLens text-bin axis. *Owner decision needed:* which AMBER query subset feeds serving (generative only, or a seeded mix), and the per-cell sample count.
- **A6 Fairness:** Base and MTP use the identical query-ID manifest, image bytes, prompt strings and cap. Quality comparisons are on matched IDs only.
- **A7 Denominators:** report failed, OOM and timeout items as counts. Scores are computed on the frozen manifest with failures counted as failures, not removed. Retries are explicit and receipted.
- **A8 Separation:** reporting code must refuse to combine serving-track outputs with quality metrics, and vice versa. A test must prove this.

## 5. MemLens-derived 7 context bins (component 5, CONDITIONAL)

- **B1 Bins:** exactly `128, 512, 1024, 2048, 4096, 8192, 16384` **model-tokenized text tokens**, ±5%. They are measured with the pinned checkpoint tokenizer, using the **same join rule** at construction and at annotation (this fixes the RV-08 mismatch). Each eligible row records the tokenizer revision.
- **B2 Effective-length accounting:** every measured request records vLLM `usage.prompt_tokens`. Each cell receipt reports min, p50 and max effective tokens and the vision-token share, estimated as effective minus text minus template overhead. Effective length must not overlap between adjacent bins for the low bins. Otherwise the report plots against effective length, not the nominal bin.
- **B3 Visual payload control:** pin the image processor settings (max/min pixels) and record them in the run receipt. *(Proposed)* Use the **same source image** for a given derived row in every bin, which current code already does per native example, and restrict the pool to images whose vision-token count falls in one predeclared band. Alternatively, add a text-only control series.
- **B4 Sample budget:** at least 160 verified unique native IDs per bin per model for CCU32. *(Proposed)* Require ≥1.25× (200) after annotation and ±5% re-check, so annotation losses can't cause `INSUFFICIENT` cells. Warmup and measured sets are disjoint, with no duplicate native IDs within a cell. Report sample overlap between repetitions; if repetitions are meant to be independent, draw disjoint per-repetition subsets or describe them as repeated measures.
- **B5 Isolation:** derived rows keep `official_quality_allowed=False` and are never scored. No gold appears in requests. Base and MTP `sample_id_sha256` must match per cell, which analysis already enforces. MemLens contributes **no** quality claims in the new design.
- **B6 One canonical derivation:** see RV-07. Its `derivation` ID and code SHA go into every cell receipt.
- **B7 Source:** the MemLens `dataset_32k.json` SHA256 and the expected question count must be verified against the publisher (the doc cites 789). Image references must all resolve, which the adapter already enforces.

## 6. GPU cost gates (component 2, CONDITIONAL)

- **G1:** keep all existing gates listed in section 3, with tests asserting their order (paid check before git, upload or Colab contact).
- **G2:** every quality invocation, in **all** scopes, needs a per-call item cap and a wall-clock budget, plus a consecutive-failure circuit breaker. *(Proposed)* 2 consecutive failures, or ≥20% failures in any rolling 25 items.
- **G3:** set the per-request quality timeout from pilot p99 latency (for example, 3× p99), not a fixed 900 s.
- **G4:** report observed vLLM boot time and annotation cost in the pilot. Recalibrate the GPU-hour estimate (`PROTOCOL.md` time budget), because the current one is explicitly unverified.

## 7. MTP activation and verification (component 7, BLOCKED for claims)

Running MTP cells for engineering purposes is acceptable. **Reporting any MTP speed-up or quality result** requires all of the following:

- **M1 Activation evidence per cell:** snapshot vLLM's Prometheus `/metrics` speculative-decoding counters before and after each measured cell. Confirm the exact metric names in pinned vLLM 0.29.0. An MTP cell must show more than 0 draft tokens; a Base cell must show 0 or none. Store the server log line confirming the loaded speculative config.
- **M2 Acceptance rate:** accepted ÷ drafted tokens from the counters, per cell. Replace the hard-coded `NOT_MEASURED` with a measured value. `NOT_MEASURED` remains only when the counters are verifiably absent.
- **M3 Greedy equivalence:** on the natural-EOS AMBER quality pass, compare Base and MTP outputs per query ID. Report the exact-match rate and the first divergent token position. Divergence is reported, not hidden; GPU greedy decoding isn't guaranteed bitwise equal. *(Proposed)* Above 1% non-identical outputs, look into acceptance or verifier correctness before quality claims.
- **M4 Multimodal support:** confirm that vLLM 0.29.0 supports native MTP for the Qwen3.5 0.8B and 4B **vision-language** path with images. If not, the status is `UNSUPPORTED_MODEL`, not a slower result.
- **M5 Frozen depth:** `num_speculative_tokens = 1` (present in config) is recorded per receipt. Depth sensitivity is a separate, later ablation.
- **M6 Speed-up claims:** only on matched `PASS_MEASURED` Base/MTP pairs with M1 satisfied. This is already partly enforced in `analysis.py`; add the M1 check there.

## 8. Serving vs quality metric contract (component 6, CONDITIONAL)

- **S1:** serving payload: `ignore_eos`, `min_tokens = max_tokens = 256`, streaming with usage. Quality payload: neither field. Both use `temperature=0`, `top_p=1` and thinking off. **Verified** (call sites `bench.py:148-150`, `bench.py:252-254`), with a test in sol-lead's slice.
- **S2:** a serving pass requires all requests to have `completion_tokens == 256`. **Verified** (`bench.py:174-206`).
- **S3:** TTFT is measured to the first non-empty content chunk. TPOT = (end − first chunk) ÷ (tokens − 1). Stream chunks aren't reported as ITL. **Verified.**
- **S4:** quality receipts record `finish_reason` (RV-02). Serving receipts include the M1 counters (RV-03).
- **S5:** pre-encode all payloads before `start_cell` (RV-11).
- **S6:** reports keep the serving and quality tables physically separate. The track field is checked on load.

## 9. Reproducibility (component 8, CONDITIONAL)

- **R1:** write environment, job and model manifests to **immutable, per-invocation** files (for example, `environment/<utc>_<session>.json`). Every cell and quality receipt references them by SHA256, together with the repo SHA and bundle SHA (RV-05).
- **R2:** pin and record vLLM (pinned), torch, transformers (unpinned, RV-10), tokenizer and processor revision, and scorer dependencies. The prepare manifest records the repo SHA and transformers version.
- **R3:** seeds: cell seeds (SHA-derived, model- and mode-independent) and the quality subset seed (42) are deterministic. **Verified.** The server seed isn't needed at temperature 0, but record it.
- **R4:** randomize or counterbalance cell order per `PROTOCOL.md`, or amend the protocol to say order is fixed and record wall-clock timestamps per cell. Interleaving Base and MTP within a session is preferred (RV-09).
- **R5:** artifact durability: atomic writes with fsync, a hash-verified Drive sync and a conflict-refusing restore are **verified**. Keep them.

## 10. Owner decisions requested

1. Truncation thresholds for AMBER quality (A3) and the greedy-divergence threshold (M3), given above as proposed values.
2. Which AMBER queries make up the serving workload, and how many per cell (A5).
3. How to control visual payload in the MemLens bins: a fixed image-token band, or a text-only control series (B3).
4. Whether repetitions must use disjoint sample sets (B4).
5. Whether to commit `uv.lock`, carried over from the sol-lead review.

## 11. Next review gates

- Integration of sol-lead's slice (`wt/sol-lead-runtime`): **PASS**, given earlier. It resolves RV-12.
- A worker worktree `wt/agy-worker-data` (branch `worker/agy-data-amber-memlens`) exists. Its diff will be reviewed against A1–A8 and B1–B7 when it's submitted.
- After integration: an independent verdict on the final integration diff, re-checking every P0/P1 item above.
