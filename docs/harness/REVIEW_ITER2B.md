# Review of the ITER2 D7 changes (ITER2 TASK D7, auditor wF:p9)

**Date:** 2026-11-10. **Branch:** `iter2/p9-review`. **HEAD reviewed:** `ed8e64d` (merge of `2780f2a`; the lead records the same tree, per REVIEW_ITER2 §7).
**Scope:** (1) closure of the REVIEW_ITER2 findings; (2) new Stage 2 CPU work by wF:p7; (3) p8 MMLongBench scorer, dtype default, GPU smoke script; (4) Q5 restated.
**Method:** code reading at HEAD; mutation runs on a scratch copy outside the repo (`git show HEAD:<path>` into a scratch tree, one exact-string mutation at a time, reverted after each); targeted test runs with `nice -n 5`. No background full suite was run (per instruction). No source file in the repo was edited. Nothing was pushed or merged.
**Limits:** the lead's full-suite figure (673 passed, 10 skipped, 0 failed at `2780f2a`) was **not** re-run by me. Stage 2 tests use the official checkpoint on CPU (`cpu_pytorch_reference_not_vllm`). Nothing here is a speed, VRAM or vLLM claim.

---

## 0. Verdict

| Area | Verdict |
|---|---|
| S-1, S-3, S-4, S-5, S-6, S-7, I-8 closures | **Closed.** Each mutation that the closure targets is now detected by the tests (§2, §6). |
| S-2 matched-retention helper | **Closed**, with one residual doc issue (N-8). |
| S-8 scope wording | **Partly closed.** The scope string is on every arm, but two memory fields are unscoped or mis-scoped (N-7). |
| I-1 inventory forgery | **Not closed.** Forged environment plus an owner-signed plan gives **528 READY cells** in the committed config shape (N-1, HIGH). |
| G-1, G-2, G-4 | **Closed** (G-1 with a latent mixed-label issue, N-9). G-3 renamed at the release gate, engine flag still a construction flag (N-11). |
| Stage 2(a) serving-style `mtp` | Committed output equals same-arm greedy by reasoning and tests; acceptance is verifier-derived. Mutations A1–A4 are each detected by the tests (§6). |
| Stage 2(b) policy-state speculation | Rollback covers cache, recurrent state and policy state by reading. Labelling is honest. Mutations B1–B4 NOT_RUN; the eviction-free rule may be untested (§4b, §6). |
| Stage 2(c) per-KV-head H2O | Tracker-only, opt-in, honestly labelled. Two small defects (N-12, N-13). |
| p8 MMLongBench scorer | Upstream fetched and hashes match the header. Logic matches on read-through. Official gate is not enforced in code (N-3). Stage 2 `BLOCKED_NO_JUDGE` is honest. |
| p8 dtype default | **Unchanged** versus `7d209f5`. |
| p8 GPU smoke | **No outer hard deadline** (N-5). The stalled run wrote no receipt, which fits this. |

---

## 1. Q5 (restated)

**Yes, protection overrides low-score H2O candidates.** HC-1 is a conflict only when H2O wants to evict an entry (it is an H2O eviction candidate, meaning outside H2O's keep set because its cumulative score is low) and DA declares that entry protected. The protection wins. An entry H2O keeps is not a conflict. Full reasoning: REVIEW_ITER2 §1.3.
Still open: `docs/harness/STARLIGHT_INVARIANTS_ORACLE.md:104` (HC-1) says "H2O top-score". That wording is my error, O-1 in REVIEW_ITER2. I have not corrected it in this deliverable, because the D7 instruction is a review, not a docs pass.

---

## 2. Closure of REVIEW_ITER2 findings

| ID | Claim to close | Evidence at HEAD | Status |
|---|---|---|---|
| S-1 | Candidate sets identical across ablations | `real_engine.py` `_starlight_build_step_plans`: `da_slots`, `gr_slots`, `st_slots` computed on every arm; `n_protected` is the union. Mutation **S1-revert** (DA ablation gates the candidate budget again): `test_s1_candidate_set_is_ablation_independent_and_only_the_veto_differs` **fails (1)**. | CLOSED |
| S-3 | INFEASIBLE evicts nothing | `starlight.py` ~L175–225: `status = STATUS_INFEASIBLE` when `length - len(eligible) > budget`, then `eligible = set()`. Mutation **S3-revert** (eligible evicted again): **4 failed**. | CLOSED. Consequence still open: KV grows unbounded for infeasible configs; `kv_budget_violated` is reported (see §8 owner decision A-2). |
| S-4 | H-6 cache-length check tested | `real_engine.py` `_test_cache_len_offset` injection point. Mutation **S4-M17** (H-6 check disabled): `test_s4_cache_length_check_detects_an_injected_mismatch` **fails (1)**. | CLOSED |
| S-5 | Summary counters not vacuous | Mutation **S5-M8** (protected-eviction counter disabled): `test_s5_summary_counts_protected_evict_and_evicted_not_candidate_as_one_each` **fails (1)**. | CLOSED |
| S-6 | Unverified evidence protects nothing | `starlight.py` `evidence_validity` (`if not item.verified`). Mutation **S6-M23**: `test_s6_unverified_evidence_protects_nothing_and_verified_control_protects` **fails (1)**. | CLOSED |
| S-7 | Keyword variants tested | `valid_ids` / keyword tokenizer. Mutation **S7-M22** (bare form only): `test_s7_bare_space_prefixed_and_capitalised_keyword_forms_are_all_protected` **fails (1)**. | CLOSED |
| I-8 | `vision_start`/`vision_end` protected, never evicted | Structural source. Mutation **I8-mut** (`st_slots = {}`): **3 failed** (`test_HC13_vision_start_and_end_are_protected_and_never_evicted`, `test_vision_tokens_are_protected_and_never_evicted_under_pressure`, `test_vision_token_candidate_is_vetoed_at_helper_level`). | CLOSED |
| S-2 | Matched-retention helper | `h2o.py:451` `calibrate_h2o_ratio_to_retained` (bisection on measured mean). `tests/test_matched_retention.py`: **8 passed** (run in D7). | CLOSED, residual N-8 |
| S-8 | Memory wording states scope | Scope string `per decode step, prefill cache unbounded` at `real_engine.py:1647`, `:1737`, `:2760`. `test_memory_fields_state_the_scope…` passes. | PARTLY CLOSED, residual N-7 |
| S-9 | DA signal scripted, not model-declared | Unchanged (LOW, claims only). | OPEN (claim wording) |
| I-1 | READY not reachable from fabricated preconditions | See N-1. Earlier trivial forgeries are refused (tests pass). Stronger forgeries are not. | **NOT CLOSED** |
| I-2 | Committed inventory matches HEAD | `docs/harness/PILOT_INVENTORY.json` `head` = `08fec37…`, HEAD = `ed8e64d`. | OPEN |
| I-3 | "No READY possible" claim | Corrected in `ITER2_D2_REPORT.md:31`, but that correction says the forgery tests show READY is now refused. N-1 shows it is not. | **OPEN** (N-2 wording) |
| G-1 | MTP_CORRECTNESS_PASS not FAIL in an all-pass run | `scripts/run_progressive_qualification.py` `_engine_gate` scoped to `MTP_ARMS`. Scratch mock (`g1_check.py`): all-pass diagnostic path gives `GATED_NATIVE_MTP_VERIFIER_REQUIRED`. | CLOSED (latent issue N-9) |
| G-2 | "PASS" not "no exception"; isolation labels consistent | Gate renamed to `PYTORCH_THREAD_CITATION_LEAK_CHECK`. `token_isolation_check` added (tokens vs solo runs, not only citations). Engine keeps `CCU_ISOLATION_PASS` GATED. | CLOSED (residual N-10) |
| G-3 | `SEVEN_ARM_PRETRAINED_PASS` construction flag | Release gate is now `SEVEN_ARM_HOOK_COVERAGE_PASS` (test D status). Engine literal still at `real_engine.py:2256`. | PARTLY CLOSED (N-11) |
| G-4 | One parity tolerance | `src/vlmbench/infrastructure/parity.py` `LOGITS_PARITY_TOLERANCE = 1e-4`, imported by both scripts. `tests/test_parity_tolerance.py` checks it. | CLOSED |
| O-1 | HC-1 wording | Not corrected (§1). | OPEN (my doc) |

---

## 3. New findings from this review

Severity follows REVIEW_ITER2. Owner per the ITER2 ownership table.

### N-1 — READY is reachable with an owner-signed plan and forged environment (HIGH, owner wF:p8)
- **What.** The owner signature covers only the plan (the config bytes, through `plan_sha256`). Weights, GPU, vLLM, receipts and dataset files are outside it. Forging those five gives READY for every cell of the signed campaign.
- **Where.**
  - `scripts/harness_inventory.py:226–227` `covered = bool(files) and weight_files <= set(files)`. This is vacuously true when the model dir has **zero** `*.safetensors`, so a manifest listing only `config.json` verifies.
  - `:253–256` `probe_gpu` runs `nvidia-smi` from `PATH`, so a fake binary is accepted.
  - `:324–335` `verify_vllm` imports `vllm` and `vllm._C` from `PYTHONPATH` and compares `__version__`. A stub with an empty `_C` passes.
  - `:355–376` `load_receipts`: the index is an unsigned repo file. Whoever can write the index can list any receipt.
  - `:340–352` `probe_dataset`: `native_path` and `native_sha256` come from the same config the owner signed, so a self-written file matches.
- **Reproducer.** `scratchpad/forge2.py` (scratch only; the simulated owner key is in the scratch dir). Output, run at `ed8e64d`:
  ```
  == A: owner-signed plan + forged weights/GPU/vLLM/receipts/datasets
     authorization_ok={'pilot': True, 'full': True} vllm_installed=True gpu={'name': 'NVIDIA A100-SXM4-40GB', ...}
     weights qwen35_08b present=True manifest_pinned=True verified=True
     datasets checksum=[True, True] receipts_loaded=3528
     pilot counts READY=24 full READY=504
  ```
  The READY cell has all 14 checks PASS (`checks` dict) and an empty `blockers` list. The model dir holds `config.json` and the manifest only. Controls: B (same, unsigned owner record) gives 0 READY; C (signed, no manifest pin) gives 0 READY. So the signature gate works, and the environment gates are the hole.
- **Consequence.** The "READY" label certifies a measured environment. With these probes it certifies nothing an agent could not fabricate once the owner has signed a plan.
- **Fix (owner decision).** (a) weights: require every `*.safetensors` present and listed, and the manifest pinned by hash inside the owner-signed record; (b) GPU: probe through an absolute path or NVML, record the driver, not `PATH`; (c) vLLM: require the pinned version to match `importlib.metadata` from a verified install, and reject packages with no distribution metadata; (d) receipts: sign the index or require it to match the committed git blob; (e) dataset pins: put dataset checksums and revisions in the owner-signed record, not in the mutable config.

### N-2 — The CLI cannot produce any READY, and the READY claim text overstates (MEDIUM, owner wF:p8)
- `scripts/harness_inventory.py:549–566` `main()` does not pass `allowed_signers`, `pinned_manifest_sha256`, `owner_record` or `receipt_index`. So the CLI always has `authorization_ok=False`, `manifest_pinned=False` and no receipts. This is fail-closed, but the inventory can never show READY from the command that writes `PILOT_INVENTORY.json`, and the `ready_requires` rule cannot be met by that command.
- `docs/harness/ITER2_D2_REPORT.md:31` says the forgery tests show the forgeries are refused. N-1 shows the stronger forgeries pass. Correct the sentence, or fix N-1.
- `docs/harness/PILOT_INVENTORY.json` `head` is `08fec37…`, not HEAD (I-2, still open).

### N-3 — The official gate in the MMLongBench scorer is not enforced (MEDIUM, owner wF:p8)
- `src/vlmbench/datasets/mmlong_official_scorer.py:247–262` `ExtractedAnswer.__post_init__` accepts any non-empty `extractor_model` and `extractor_prompt_sha256`. It does not compare the hash to `EXTRACTION_PROMPT_SHA256` (line 38), and nothing uses that constant.
- `:278–296` `score_records` then returns `status = "OFFICIAL_SCORED_STAGE3_ONLY"` with `upstream_commit` set. A heuristic extraction tagged with any model name and any string passes as "official stage 3".
- **Reproducer.** Call `score_records(records, {qid: ExtractedAnswer(qid, pred, "x", "0"*64)})`. It returns the OFFICIAL status.
- **Fix.** Reject extractions whose hash is not `EXTRACTION_PROMPT_SHA256` and whose model is not the official extractor. Or make the status name say "caller-asserted extraction".

### N-4 — Differential fuzz of the scorer not run (limitation, not a defect)
- I downloaded upstream `eval/eval_score.py` (commit `d73f0dc`, sha256 `cafa2683…` matches the header) and `eval/extract_answer.py` (`fa6a747b…` matches) into scratch. A differential fuzz (20 000 random `eval_score` cases) was **blocked by the sandbox** because it would execute externally fetched code. I did not work around that. The check below is a read-through only.
- **Read-through.** `levenshtein_distance`, `anls_compute`, `is_float_equal`, `get_clean_string` (including the official no-op `rstrip` lines), `is_exact_match`, the Int/Float/Str/None branches, and `eval_acc_and_f1` match the official code. `show_results` breakdowns match in content. The documented deviation is `eval` → `ast.literal_eval` on list-looking strings, which differs only for non-literal list expressions. Labelled in the header.
- **Needed.** The differential run, in an environment the owner approves for executing the upstream file.

### N-5 — GPU smoke has no outer hard deadline and writes no receipt on a hang (MEDIUM, owner wF:p8)
- `scripts/pytorch_reference_gpu_smoke.py:90–95` `_guard` checks `WALL_LIMIT_S = 600` only **before each request** (`:127`, `:136`). Model load (`RealSevenArmEngine(...)`, ~`:113`) and any single `execute_request` run with no check. A hang there never reaches a guard.
- `:212–219`: the receipt is written only after the `try` (`:186–211`). A kill, hang or uncaught exit writes nothing. This matches the stalled p8 run with no receipt.
- `:162` `torch.cuda.get_device_capability(0)` runs before the `try`. On a CPU-only host it raises and writes no receipt.
- **Fix.** Write a `RUN_STARTED` receipt before any work; run under an outer watchdog (`signal.alarm` in the process, or `timeout -k` in the wrapper) that writes `ABORTED: wall-clock` and exits; move the capability probe inside the `try`.
- **Immediate.** Do not rerun the smoke without an outer `timeout`. Any run needs a wrapper `timeout` (for example `timeout -k 30 900`).

### N-6 — dtype default unchanged (verified, closed)
- `real_engine.py:67–85` `select_official_dtype`: no override gives bf16 on CUDA with capability ≥ 8, else fp32. Baseline `git show 7d209f5:src/vlmbench/infrastructure/real_engine.py` L312–318: `use_bf16 = cuda and cuda available and capability[0] >= 8`, `dtype = bf16 if use_bf16 else fp32`. Same result for every input. Wiring at `:337–338` gives the same capability as the baseline. `tests/test_dtype_selection.py` covers default, override, CPU fp16 refusal and cache key.

### N-7 — Memory-field scope is incomplete (LOW, lead wF:p5 for wording)
- `real_engine.py:2760` labels the block `per decode step, prefill cache unbounded`. The same block contains `final_retained_kv_payload_bytes` (`:2764`, value `final_allocated_bytes`), which is a **final-state** snapshot, not a per-decode-step quantity. The scope string does not describe it.
- `real_engine.py:2719–2720` `RealExecutionResult.physical_kv_bytes_allocated/freed` carry no scope at all.
- `real_engine.py:2862` `physical_kv_bytes_freed` sits in the arm telemetry, outside `kv_memory_accounting`.
- `real_engine.py:1605` (policy path) sets `physical_kv_bytes_freed=None` ("not measured"), while `:1722` sets `0` ("no eviction policy"). Both are fine but differ in meaning, and nothing documents it.
- The test checks only the scope string of `kv_memory_accounting`, not these fields.

### N-8 — The matched-retention helper's budget mapping holds only for text-only prompts (LOW, lead wF:p5)
- `h2o.py:451–455` docstring: budget `b` satisfies `int(prompt_len × ratio) == b`. The H2O engine budget is `h2o.py:124` `max(min_budget, int(unprotected_length × ratio))`, where `unprotected_length` excludes protected (visual and sink) tokens. The two agree only when `unprotected_length == prompt_len`, i.e. no protected tokens.
- The test (`tests/test_matched_retention.py`) uses a text-only prompt, so it passes. A caller calibrating on an image prompt gets a budget off by the protected count. The docstring should say so, or the helper should take the unprotected count.

### N-9 — `_engine_gate` returns FAIL for mixed arm labels (LOW, latent, owner wF:p8)
- `scripts/run_progressive_qualification.py:634–643`: if the MTP arms report different labels, `len(values) != 1` gives `FAIL`. Scratch check (`g1_check.py`): with `mtp` reporting `CPU_REFERENCE_SPECULATION_EXECUTED_NOT_VLLM` and `da_mtp` reporting `GATED…`, the gate is `FAIL`, although no test failed.
- Not reached today: test D uses `image_present=True`, which takes the diagnostic path, so all three report GATED. The text-only path (`real_engine.py:1593`, `:1700`) would trigger it.
- **Fix.** Map labels to a closed set, or treat differing non-PASS labels as their own status, not `FAIL`.

### N-10 — Assert-based oracles plus `guarded` give silent PASS under `python -O` (LOW, owner wF:p8)
- `scripts/run_progressive_qualification.py:665–672` `guarded` returns `PASS` whenever the test function returns. Every oracle in the tests is an `assert` (about 110 in the file). Under `python -O` all asserts are removed, so every test returns and records `PASS`.
- Nothing in the repo runs the script with `-O` today. The risk is latent.
- **Fix.** Replace the asserts with `if not …: raise AssertionError`, or refuse to run under `sys.flags.optimize`.

### N-11 — `SEVEN_ARM_PRETRAINED_PASS` still a construction flag at engine level (LOW, owner wF:p8/p7)
- `real_engine.py:2256` and the test D assert at `run_progressive_qualification.py:294`. The release gate now reads `SEVEN_ARM_HOOK_COVERAGE_PASS`, which is correctly labelled. The engine value still says "PASS" for hook attachment only. Rename the engine key or the value.

### N-12 — Per-KV-head docstring says the budget is untouched, the code sets it (LOW, owner wF:p7)
- `h2o.py:246–251` says `compute_eviction_plan_per_head` "Does NOT touch the layer's evicted mask or budget". `h2o.py:262–263` sets `layer.budget` when it is `None`, and `:258–260` pads `layer.scores`.
- `tests/test_h2o_per_head.py` sets `budget = 5` in its fixture, so the `None` path is never tested. The mask claim holds (test passes).

### N-13 — Per-KV-head flag is silently ignored on the STARLIGHT candidate path (LOW, owner wF:p7)
- `h2o.py:155–157` refuses `per_kv_head_selection=True` only in `compute_eviction_plan`. `keep_mask_without_protection` (`h2o.py:289–…`, used by the STARLIGHT candidates) does not check the flag. So a config with the flag on runs the layer-mean selection there, without an error. No engine path sets the flag today (grep: no reference in `real_engine.py`), so this is latent. The "tracker only" label is correct.

---

## 4. Stage 2 review (wF:p7)

### 4a. Serving-style `mtp` arm (`real_engine.py` `_speculative_mtp_greedy_k1`, ~L1431–1560)
- **Committed output equals same-arm greedy.** By reading: the pending token is always `argmax(verifier | prompt + committed)`. Accept (`proposal == target_after_pending`, where `target_after_pending = argmax(full_logits[0,0])`) emits the proposal and advances `pending` from `logits[0]` (the last row, correct because `logits` is `out.logits[:, -1]`, `real_engine.py:~868`). Reject restores the pre-block snapshot (`_restore_dynamic_cache`, which also restores linear-attention `conv_states`/`recurrent_states`, `:292–330`) and refeeds the pending token alone. Neither branch emits an unverified token.
- **Proposed/accepted are verifier-derived.** `accepted` increments only when the verifier's argmax equals the draft. `verifier_calls` counts prefill, each 2-row block and each reject refeed (`:1490`, `:1514`, `:1534`).
- **EOS.** A pending EOS stops before emission (`:1499–1502`). An accepted EOS is not emitted (`:1517–1523`). A rejected draft that is EOS is handled on the next iteration.
- **Caveat.** Equality rests on floating-point agreement between the 2-row block and the single-row forward. This is not bit-guaranteed for argmax near-ties. Tests cover three natural/forced prompts and 40 tokens (`tests/test_mtp_speculative_serving.py:105–207`).
- **Scope.** Text-only, official checkpoint, `cpu_greedy_k1_reference_not_vllm_mtp`. Image requests report `speculation_executed=False` (`real_engine.py:~2745`). Multi-turn and multi-request isolation are tested (`test_multi_turn_and_multi_request_isolation`).
- **MTP alignment with the official MTP forward** is not independently verified.

### 4b. Policy-state speculation for `h2o_mtp` and `starlight_s1_mtp` (`real_engine.py` ~L1757–2065)
- **Rollback coverage.** `_pol_snapshot` (`:1905`) and `_pol_restore` (`:1923`) cover the physical cache (through `_restore_dynamic_cache`), per-layer scores, evicted masks, visual masks, budgets, tracker `seq_len` and tracker visual mask, the STARLIGHT position map (`orig`) and `P`. I checked the H2O tracker's mutable state (`h2o.py` `H2OLayerState` fields, `H2OCacheTracker.__init__`): nothing else is mutated per step. The snapshot is complete for that state.
- **Eviction-free rule.** `_pol_block_is_eviction_free` (`:1864–1880`) requires, for each full-attention layer, `phys == L`, and for H2O `unprotected_count(L+2) <= budget`, for STARLIGHT `L+2 <= budget_total`. Otherwise the pending token is fed alone (`:2006`). This makes the identity plans in `_pol_block` valid.
- **Weaker invariant honesty.** Accepted-block score values are not asserted bit-identical (`95d45ca`; `STARLIGHT_STAGE1_DESIGN.md` §15b). Decisions and tokens are. This is stated.
- **Scope labels.** `arm_status` `degraded_text_only_no_evidence`; `da_mtp` non-speculative; text-only requests only. Labelled in the design and report.
- **Gap (open, not mutation-verified).** The policy tests (`tests/test_policy_mtp_speculative.py`) use prompts where eviction starts after about position 31, so a speculative block may never evict. If the eviction-free check were disabled (B4, NOT_RUN) and the tests still passed, then rollback under compaction, and the identity-plan assumption inside a block, would be untested. Owner p7 to confirm with B4 or a long-prompt test.
- **Acceptance rate** is measured on eviction-free blocks only, so it is biased toward the early, no-eviction part of a run. Do not quote it as a general rate.

### 4c. Per-KV-head H2O (`h2o.py:245–287`)
- Tracker-only, opt-in, default False (`h2o.py:34`). The shared plan refuses the flag (`:155–157`). Not referenced by `real_engine.py`. The per-head selection itself matches its description (own heavy hitters per head, shared recent window, same budget). Findings N-12, N-13.

---

## 5. p8 items

- **MMLongBench stage-3 scorer provenance.** `mmlong_official_scorer.py` header gives upstream commit `d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384` and three sha256 values. I fetched `eval/eval_score.py` and `eval/extract_answer.py` at that commit read-only; both hashes match (`cafa2683…`, `fa6a747b…`). The third (`prompt_for_answer_extraction.md`, `96a22c30…`) I did not fetch. Stage 2 is `BLOCKED_NO_JUDGE`, which is the honest status. Findings N-3, N-4.
- **Dtype default.** Unchanged (N-6).
- **GPU smoke.** No outer hard deadline (N-5). The stalled run wrote no receipt. Wrap any rerun with an external `timeout`. This needs the owner's decision on whether to run it at all.

---

## 6. Mutation results (scratch copy; each reverted)

Closure mutations (REVIEW_ITER2 items), `tests/test_starlight_*.py`, `tests/test_matched_retention.py`, from `scratchpad/mut2.log`:

| Mutation | Target | Result |
|---|---|---|
| S1-revert | S-1 | 1 failed |
| S3-revert | S-3 | 4 failed |
| S4-M17 | S-4 | 1 failed |
| S5-M8 | S-5 | 1 failed |
| S6-M23 | S-6 | 1 failed |
| S7-M22 | S-7 | 1 failed |
| I8-mut | I-8 | 3 failed |
| M1 | I-1 eligibility | 22 failed |
| M14 | position map not compacted | 31 failed |
| M16 | model never receives plan | 30 failed |

Stage 2 mutations (`scratchpad/mutations_s2.py`, runner `scratchpad/run_s2.py`, log `scratchpad/mut_s2b.log`; each mutation runs only the test file it targets, because the two files together were too slow). Baselines on the unmutated scratch copy: `tests/test_mtp_speculative_serving.py` 6 passed; `tests/test_policy_mtp_speculative.py` 6 passed.

| Mutation | Target | Result |
|---|---|---|
| A1 serving: accept every draft | 2(a) committed == greedy | 5 failed, 1 passed |
| A2 serving: no cache restore on reject | 2(a) rollback | 5 failed, 1 passed |
| A3 serving: EOS emitted | 2(a) EOS | 1 failed, 5 passed |
| A4 serving: refeed not counted as verifier call | 2(a) verifier-derived counts | 4 failed, 2 passed |
| B1 policy restore: evicted masks not restored | 2(b) rollback | NOT_RUN (batch stopped) |
| B2 policy restore: STARLIGHT position map not restored | 2(b) rollback | NOT_RUN (batch stopped) |
| B3 policy restore: seq_len not restored | 2(b) rollback | NOT_RUN (batch stopped) |
| B4 policy: eviction-free check disabled | 2(b) block safety | NOT_RUN (batch stopped) |

The B rows were not run because the lead asked for the review to be finalised without waiting for the batch. The batch was killed; the scratch copy was restored from HEAD and matches it. The 2(b) claims below are therefore code-reading results, not mutation-verified, and B4 in particular is an open question.

---

## 7. Test runs in this review

- `tests/test_matched_retention.py`: 8 passed (2.4 s).
- Targeted STARLIGHT mutation set: as §6 (run from `mut2`).
- Stage 2 serving and policy baselines: 6 passed each (§6). Serving mutations A1–A4 run; policy mutations B1–B4 NOT_RUN.
- `g1_check.py` (G-1 mock), `forge2.py` (N-1 forgeries): scratch only.
- Not run: full suite; the scorer differential fuzz (blocked, N-4).

---

## 8. Findings summary and owner decisions

| ID | Severity | Owner | Location | Summary |
|---|---|---|---|---|
| N-1 | HIGH | wF:p8 | `harness_inventory.py:226–227, 253–256, 324–335, 355–376, 340–352` | READY (528 cells) with owner-signed plan and forged env |
| N-2 | MEDIUM | wF:p8 | `harness_inventory.py:549–566`; `ITER2_D2_REPORT.md:31` | CLI cannot produce READY; claim text overstates |
| N-3 | MEDIUM | wF:p8 | `mmlong_official_scorer.py:247–262, 278–296` | Official stage-3 label not gated on extractor/prompt hash |
| N-4 | (limit) | wF:p9 | — | Scorer differential fuzz not run (sandbox blocked) |
| N-5 | MEDIUM | wF:p8 | `pytorch_reference_gpu_smoke.py:90–95, 108–136, 162, 212–219` | No outer hard deadline; no receipt on hang |
| N-6 | — | — | `real_engine.py:67–85` | dtype default unchanged (closed) |
| N-7 | LOW | wF:p5 (wording) | `real_engine.py:2719–2720, 2760–2764, 2862, 1605, 1722` | Scope wrong or missing on some memory fields |
| N-8 | LOW | wF:p5 | `h2o.py:451–455`, `h2o.py:124` | Matched-retention mapping holds only for text-only prompts |
| N-9 | LOW | wF:p8 | `run_progressive_qualification.py:634–643` | Mixed arm labels give FAIL (latent) |
| N-10 | LOW | wF:p8 | `run_progressive_qualification.py:665–672` | `assert` oracles plus `guarded` pass under `-O` |
| N-11 | LOW | wF:p8/p7 | `real_engine.py:2256`; `run_progressive_qualification.py:294` | Engine `SEVEN_ARM_PRETRAINED_PASS` is a construction flag |
| N-12 | LOW | wF:p7 | `h2o.py:246–263` | Per-head docstring says budget untouched; code sets it |
| N-13 | LOW | wF:p7 | `h2o.py:155–157, 289–…` | Per-head flag silently ignored on STARLIGHT path (latent) |
| I-2 | LOW | wF:p8 | `PILOT_INVENTORY.json` head | Committed inventory not at HEAD |
| O-1 | LOW | wF:p9 (me) | `STARLIGHT_INVARIANTS_ORACLE.md:104` | "H2O top-score" wording |

**Status at finalisation.** OPEN: I-1 (not closed; see N-1), N-1, N-2, N-3, N-5, N-7, N-8, N-9, N-10, N-11, N-12, N-13, I-2, O-1, S-9, I-3 (wording). CLOSED: N-6 (dtype default unchanged). LIMIT: N-4 (fuzz not run). Closed by mutation or test: S-1 to S-8 (except the S-8 residual N-7), G-1, G-2, G-4 (G-3 partly, see N-11).

**Owner decisions still needed:** A-2 (INFEASIBLE with unbounded KV growth, relates to S-3); A-4 (structural protection, I-8); A-5 (missing declarations, HC-7); A-6 (decode-only eviction, S-8); the budget basis for side-by-side tables (S-2); whether the GPU smoke runs at all (N-5); the signing design for N-1.

---

*Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>*
