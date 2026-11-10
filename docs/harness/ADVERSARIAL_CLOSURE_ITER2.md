# Adversarial closure review of iteration 2 (ITER3 TASK A, auditor wF:p9)

**Reviewed HEAD:** `d8ebc16` (branch `iter3/p9`). **Scope:** (1) closure matrix for the historical findings listed in the brief, plus a re-review of the D9–D11 fix round (commits `9b332bc`..`d8ebc16`, 19 commits, 31 files); (2) whether STARLIGHT's internal protection signal is evidence protection or keyword preservation.
**Method:** code read at HEAD; targeted pytest runs (one heavy job at a time, with timeouts); scratch reproducers outside the repo (`scratchpad/it3/`). No Opus review; the banners observed in the ledger are Sonnet 5.5 (lead) and Haiku 5.5 (p7, p8, p9). This is an independent review by one auditor, not a team approval.

**Status rule:** CLOSED only with a targeted test that passes at HEAD, or a mutation on a scratch copy with the failing test named. Otherwise OPEN, with an exact reproducing command.

**Targeted test runs at HEAD (all pass):**

| Command | Result |
|---|---|
| `pytest -q tests/test_harness_forgery.py tests/test_harness_cli.py tests/test_harness_live_probe.py tests/test_harness_inventory.py tests/test_inventory_freshness.py` | 67 passed (3.5 s) |
| `pytest -q tests/test_gpu_smoke_outer_deadline.py tests/test_gpu_smoke_helpers.py tests/test_mmlong_official_scorer.py tests/test_mmlong_fuzz_vs_reference.py tests/test_oracle_checks_under_O.py tests/test_qualification_gates.py tests/test_gate_label_consistency.py tests/test_h2o_per_head.py tests/test_matched_retention.py tests/test_dtype_selection.py tests/test_parity_tolerance.py tests/test_multimodal_fixture.py` | 123 passed (70.8 s) |
| `pytest -q tests/test_starlight_oracle.py tests/test_starlight_engine.py tests/test_starlight_audit_fixes.py tests/test_starlight_d4.py tests/test_starlight_policy.py tests/test_starlight_signals.py` | 91 passed (26.7 s) |
| `pytest -q tests/test_mtp_speculative_serving.py tests/test_policy_mtp_speculative.py` | **TIMEOUT, no result.** Killed by `timeout 580` (exit 124) with no pass or fail counts printed. Not a pass. Checkpoint-dependent Stage 2 tests; not re-run in this round (one heavy job at a time). |

---

## 0. Round 2 status (branch `iter3r2/p9`, base `iter2/lead` 8a6390d)

Independent re-review of the round-2 fixes. Commits this round: `4f52fc1` (red and coverage tests), `b39e081` (C-5 fix), `751d808` (C-4 strict xfail), plus the commit that adds this section. Auditor: wF:p9. Heavy jobs were run under `flock wt/heavy.lock`, one at a time. Mutations ran on a scratch copy of `8a6390d` (`git archive`), one exact-string change at a time, outside the repo.

### 0.1 Status per C-id

| C-id | Round 1 | Round 2 status | Evidence (tests pass at HEAD unless noted) | Residual |
|---|---|---|---|---|
| C-1 R1 weightless pin | OPEN | **CLOSED** | `177b308`; `test_harness_r2_regressions.py::test_R1_*` (4 tests) | none |
| C-1 R2 operator weight pins | OPEN | **CLOSED** | pins read only from the signed record; `test_R2_a_weight_manifest_pin_that_is_not_the_signed_one_is_not_a_pin`, `test_R2_the_operator_pin_flag_is_gone_from_the_cli` | signer anchor, see N-R2-1 |
| C-1 R3 unsigned receipt index | OPEN | **CLOSED** | `test_harness_pins.py::test_a_receipt_index_whose_hash_is_not_the_signed_pin_counts_nothing` | none |
| C-2 in-process vLLM stub | OPEN | **CLOSED** | `verify_vllm` runs `python -I` and needs signed version and RECORD pins; `test_harness_r2_regressions.py::test_C2_verify_vllm_runs_isolated_and_refuses_a_pythonpath_stub` | none |
| C-3 caller-asserted `verified` / source hash | OPEN | **CLOSED as labelling** (the finding allowed "do not call it verified") | `evidence_provenance` reports `verified_flag: CALLER_DECLARED_NOT_VERIFIED`; `test_r2_c3_provenance.py` | protection still honours the caller's `verified` (unchanged). `CHECKED_IMAGE_BYTES_MATCH` only means the declared hash equals the request's image hash; evidence content is not checked (LOW) |
| C-4 keyword breadth | OPEN | **OPEN (partial)** | per-keyword cap and stopword list in `3251469`; `test_r2_c4_keyword_breadth.py` passes; strict xfail `test_the_union_of_accepted_keywords_is_bounded_by_the_policy` | the cap is per keyword only; the protected union is unbounded and unreported (section 0.4) |
| C-5 image evidence protects no visual tokens | OPEN | **CLOSED** for the label and the modality bypass; protection itself is still text-only | `6084e25` (NOT_IMPLEMENTED label); `b39e081` refuses unknown modalities; `test_r2_c5_visual_evidence.py` (7 tests) | visual evidence still protects no visual tokens (by design, labelled) |
| C-6 stale doc statements | OPEN | **CLOSED** | `bc2b331`; `test_r2_c6_doc_stale_strings.py::test_no_stale_statement_contradicts_the_code_in_the_cited_docs` passes | historical review files still quote the retired label (not a claim) |
| C-7 receipt not in git | LOW | **CLOSED as disclosure** | `c43d526` states the receipt is local; `GPU_RECEIPT_INDEX.json` has 0 entries | the receipt is not versioned |
| C-8 bare-except deviation undocumented | LOW | **CLOSED** | `1c8a1a5` header states the change (read; not executed) | none |
| C-9 `compare_receipts` order | LOW | **CLOSED** | `3c41e8d`; `test_gpu_smoke_helpers.py::test_C9_two_cuda_receipts_are_refused` and the CPU-first refusal test pass | none |
| C-10 memory-block scope | LOW | **CLOSED** | `b7cf237`: block scope is `MIXED_PER_FIELD_SEE_memory_field_scopes`; `test_matched_retention.py::test_memory_block_states_no_single_scope_because_it_mixes_per_step_and_final_state` passes | none |
| C-11 engine label derivation | LOW | **not re-checked** in round 2 | | covered only by the label-consistency test |

### 0.2 F-1 (shared retained set) review

**Judgement on p7's concern.** The round-1 M4 cross-layer scenario (a veto on layer 3 only) can no longer occur under the shared set. Candidates are layer-independent and `_starlight_build_step_plans` refuses unequal lengths or differing protected slots. What has to be covered now is that the veto applies to a *shared candidate* on every full-attention layer. The round-2 F-1 scenario did not cover that: it protects position 5, whose mean score is high, so position 5 is never a candidate. Dropping the DA veto leaves that scenario green (mutation F1b below). Coverage is now provided by three new tests in `tests/test_r2_shared_candidates.py` and by the checkpoint M4 test, which fails under F1b.

**Mutation results** (exact strings, scratch copy, one at a time):

| ID | Mutation | Round-2 result (before my tests) | After my tests |
|---|---|---|---|
| F1a | candidates computed per layer | 28 failed, 67 passed (7 starlight files); F-1 scenario fails | 2 failed (`test_F1_shared_candidates_keep_every_full_layer_…`, `test_F1_shared_candidates_use_the_mean_over_full_layers_…`) |
| F1b | DA veto dropped on all layers | 7 failed (7 starlight files). **F-1 scenario passes** | `test_F1_veto_applies_to_a_shared_candidate_on_every_full_layer` fails (1 failed). Checkpoint `test_iter3_b2_mutation.py::test_M4…` fails under flock (1 failed, 88 s) |
| F1c | structural veto dropped | 3 failed (`test_HC13_…`, `test_vision_…` tests) | unchanged |
| F1d | off-by-one in the heavy-hitter rule | 6 failed | 2 failed (incl. `test_F1_shared_mask_with_one_layer_equals_…`) |
| F1e | aggregate taken from layer 3 only | **SURVIVED** (95 passed) | `test_F1_shared_candidates_use_the_mean_over_full_layers_not_one_layer` fails (1 failed) |
| F1f | unequal-length guard removed | 1 failed (`test_F1_unequal_full_layer_lengths_…`) | unchanged |
| F1g | protected-union guard removed | **SURVIVED** (95 passed) | `test_F1_protected_slots_that_differ_across_layers_raise_…` fails (1 failed) |

The checkpoint M4 mutant (F1b) fails at its baseline assertion, `baseline must pass first`: position 23 is evicted even without the mutation, because the mutation also applies to the baseline run. That is a valid detection, but the failure message is about the baseline, not the veto itself.

**Protected never evicted, evicted ⊆ candidates, positions valid.** Covered by `test_starlight_oracle.py` (I1/I2 recompute eligibility from the records, HC13, HC8), `test_iter3_b1_policy_invariants.py::test_pinv_very_small_kv_budgets_never_evict_protected_entries`, and the new tests (`evicted ⊆ range(length)`, protected retained).

### 0.3 Forgery attempts on READY (round 2)

| ID | Forgery | Result | Severity |
|---|---|---|---|
| E1 | Attacker generates an ed25519 key, writes their own `--allowed-signers`, signs a record with their own pins (A100 name and uuid), passes that record as `--owner-record`. | `verify_owner_authorization` → **verified True**. `build_report` with a real, unmocked live probe on this host → **READY 0 pilot, 0 full**; reasons: vLLM not installed, GPU name differs from the signed pin, GPU uuid differs, weights not pinned. | MEDIUM (N-R2-1). Authorization can be forged by whoever chooses the signer file. READY still needs the live probe on the signed hardware. |
| E2 | Image evidence declared with `modality="image"`. | Status read `NO_IMAGE_EVIDENCE_DECLARED` while image evidence was declared. | LOW, **fixed** by `b39e081`. |
| E3 | `nvidia-smi` shim first on PATH. | `probe_gpu()` returns `compute_cap 8.0, memory 40960` from the shim. `probe_gpu` (parent) runs `nvidia-smi` from PATH, while the live probe uses `/usr/bin/nvidia-smi`. The capability and memory gates therefore trust PATH. READY still requires the child's name and uuid to match the signed pin. | LOW (G-PATH), OPEN |
| E4 | Zero-byte and config-only manifest pins, operator pin flag, unsigned receipt index. | Refused by the existing `test_R1_*`, `test_R2_*`, `test_a_receipt_index_…` tests (all pass). | none |

READY stays unreachable on this host from any forgery tried. Reaching READY needs control of the signer file and the signed hardware, which the threat model already assumes an external anchor for.

### 0.4 C-4 keyword breadth: can it be bypassed, and is it silent?

Probe on the scratch copy (`c4_probe.py`, corrected):
- The 0.2 cap is applied per keyword. Six accepted keywords, each at 7/40 (0.175), together protect **40/40** prompt positions. The strict xfail records this.
- Common non-stopwords (`which`, `these`, `their`, `not`, `they`, `she`) are not in the 24-word list. `which` at 6/30 is accepted (at the cap); at 7/30 it is rejected.
- Stopword matching is exact on the lower-cased keyword. This is case-insensitive, so no case bypass was found.
- Not silent per keyword: `grounding_keyword_policy` lists accepted and rejected keywords with fractions. The union fraction is not reported. A bounded union would show up only as INFEASIBLE when protected entries exceed the budget, and then nothing is evicted. The earlier figure 33/40 in round 1 was a probe error (a wrong vocabulary id), withdrawn.

Verdict: the cap bounds each keyword but not what the keywords protect together. Owner policy decision (N-R2-2).

### 0.5 New findings from round 2

| ID | Severity | Owner | Status | Finding |
|---|---|---|---|---|
| N-R2-1 | MEDIUM | wF:p8 / lead | OPEN | The owner-authorization anchor is an operator-chosen `--allowed-signers` file. Nothing in the repo or the record pins the owner key fingerprint, so a local operator can forge authorization (E1). READY still fails on this host. Fix: pin the owner key fingerprint (or a hash of the signer file) in a location the operator cannot edit alongside the record. |
| N-R2-2 | MEDIUM | wF:p7 | **CLOSED** (ITER3 R3, lead ruling: union cap `MAX_UNION_PROMPT_FRACTION = 0.5`, declared-order greedy, `union_fraction` and `rejected_union` reported; the 0.5 value is owner-ratifiable) | Keyword breadth: the cap was per keyword and the protected union was unbounded and unreported (0.4). |
| G-PATH | LOW | wF:p8 | OPEN | `probe_gpu` takes `nvidia-smi` from PATH, so the capability and memory gates can be shimmed (E3). Fix: fixed binary path, or read capability and memory in the child. |
| F1-COV | LOW | wF:p9 | **CLOSED** (`4f52fc1`) | The F-1 scenario did not exercise the veto (F1b survived it). Three new tests now cover the veto, the mean aggregation and the union guard. |
| C5-MOD | LOW | wF:p7 | **CLOSED** (`b39e081`) | Unknown modality labels were accepted and hid image evidence from the status. |
| C3-FLAG | LOW | wF:p7 | accepted | `verified` is still caller-asserted for protection; it is now labelled as such. |

### 0.6 Owner decisions still open

- N-R2-1: the owner key anchor (what file or fingerprint the operator cannot choose).
- N-R2-2: a bound on the protected union of keywords, and reporting it.
- A-2, A-4, A-5 (unchanged from round 1).
- C-3: whether the `CALLER_DECLARED` labels are enough, or `verified` should be removed from protection.
- C-11: not re-checked.

### 0.7 Commands and counts (round 2)

All under `PYTHONPATH=src CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q -p no:cacheprovider` unless noted.

| Set | Result |
|---|---|
| harness: `test_harness_forgery`, `_cli`, `_live_probe`, `_inventory`, `_pins`, `_r2_regressions`, `test_inventory_freshness` (before any change) | 103 passed (15.1 s) |
| R2 + STARLIGHT + smoke: `test_r2_*`, `test_starlight_*`, `test_matched_retention`, `test_gpu_smoke_helpers`, `test_gpu_smoke_outer_deadline` (before any change) | 152 passed (16.7 s) |
| `test_r2_shared_candidates.py` after adding three tests | 7 passed |
| `test_r2_c5_visual_evidence.py` after the red test, before the fix | 1 failed (red as intended), 6 passed |
| evidence-related set after the C-5 fix (10 files) | 114 passed |
| `test_r2_c4_keyword_breadth.py` after the xfail | 8 passed, 1 xfailed (`--runxfail` fails on the union assertion) |
| combined targeted set (24 files, after both fixes) | **277 passed, 1 xfailed** (41.0 s) |
| checkpoint `test_iter3_b2_mutation.py::test_M4…` + `test_iter3_b2_mtp_rollback.py::test_F1_…` at HEAD, under flock | 2 passed (41.0 s) |
| same M4 on the F1b mutant, scratch, under flock | 1 failed (88 s), baseline assertion (section 0.2) |

Not run: the full suite (lead only); the Stage 2 serving and policy pair (timed out in round 1, not rerun); the checkpoint B1 and B2 suites beyond the two above; any slow CPU fp32 reference.

---

## 1. Closure matrix

| ID | Severity (original) | Original finding (source) | Fix commit(s) | Current code evidence | Targeted test or mutation | Status | Residual / reproducing command |
|---|---|---|---|---|---|---|---|
| **N-1** | HIGH | READY reachable with an owner-signed plan and a forged environment (`REVIEW_ITER2B.md` §3) | `3e7f70e` | `scripts/harness_inventory.py:346–470`: live probe in a separate `python -I` child, nonce echo, `importlib.metadata` vLLM check, fixed `/usr/bin/nvidia-smi`, CUDA kernel, recomputed weight SHA-256. `classify_cell` requires `live_probe` (line 144). | `test_full_bundle_with_a_signed_owner_plan_and_forged_environment_is_zero_ready`, `test_a_stub_vllm_without_distribution_metadata_is_refused`, `test_mocked_nvidia_smi_reporting_another_gpu_is_refused`, `test_a_recomputed_weight_hash_that_differs_from_the_pin_is_refused`, `test_output_replayed_from_an_earlier_probe_is_refused`; 67 passed | **OPEN (partly closed)** | Three residual forgeries. (R1) zero-weight manifest verifies: `harness_inventory.py:232` `covered = bool(files) and weight_files <= set(files)` is vacuous with no `*.safetensors`. (R2) weight pins are operator-supplied (`--pin-manifest`, line ~725), not in the owner-signed record, so the live probe recomputes against whatever pin the operator typed. (R3) receipts come from an unsigned repo index (`load_receipts`, line ~489); 0 entries today. With R1–R3 and a probe that really passes, READY = 528 (pilot 24, full 504). Repro: `PYTHONPATH=src python scratchpad/it3/fr/n1_forge.py mock_live` (live probe mocked to pass, A100 stand-in, vLLM stub on PYTHONPATH: `READY pilot 24 full 504`). Without the mock: `… real` gives READY 0 (vLLM absent). |
| **N-2** | MEDIUM | CLI cannot produce READY; "no READY" claim text (`REVIEW_ITER2B.md` §3) | `02096d4` | `main()` now passes signers, owner record, pins and receipt index, and runs the live probe unless `--plan-only` (lines ~712–740). Plan-only output states that no cell can be READY. | `test_live_mode_is_the_default_and_the_markdown_states_the_limit`, `test_the_cli_with_a_signed_owner_record_runs_the_live_probe_not_a_placeholder`, `test_plan_only_says_so_in_the_json_and_the_markdown_and_makes_no_ready` | **CLOSED** (CLI runs the same probes as the library) | The claim text in `ITER2_D2_REPORT.md:31` is still an overstatement; tracked under I-3. |
| **N-3** | MEDIUM | Official stage-3 label not gated on extractor identity (`REVIEW_ITER2B` §3) | `a882afb` | `mmlong_official_scorer.py:266–316`: `is_official_extraction` requires `extractor_model == "gpt-4o"` (upstream `extract_answer.py` default, verified by reading the fetched file) and the official prompt hash; otherwise `BLOCKED_NO_JUDGE` with no numbers. | `test_the_auditor_reproducer_is_not_labelled_official`, `test_right_model_with_a_wrong_prompt_hash_is_not_official`, `test_right_prompt_hash_with_another_model_is_not_official`, `test_the_official_extractor_and_prompt_produce_the_official_stage3_label` | **CLOSED** for the reproducer. Limit (stated in the docstring): the caller declares the model and hash, and the scorer cannot check that the extraction ran. | None needed for the label; the claim stays "stage 3 only, caller-declared stage 2". |
| **N-4** | limit | Scorer differential fuzz vs the official code not run (`REVIEW_ITER2B` §3) | `387d786` | Fuzz compares the port with a test-only transcription (`tests/mmlong_reference_transcription.py`), not with the executed upstream. The transcription header says so. | `test_fuzz_eval_score_port_agrees_with_the_transcription_on_4000_seeded_cases`, `test_fuzz_acc_and_f1_port_agrees_with_the_transcription_on_500_sample_sets`, hand-checked cases | **OPEN (limit)** | Differential against the executed upstream file is still not run; running fetched external code was blocked in my session. Static diff (this review, §3 C-8) shows the transcription matches upstream except the documented prints and `eval`, plus one undocumented change. |
| **N-5** | MEDIUM | GPU smoke: no outer hard deadline; no receipt on hang (`REVIEW_ITER2B` §3) | `9b332bc`, `f9306e2`, `960fb17` | `pytorch_reference_gpu_smoke.py:33` `WALL_LIMIT_S = {"cuda": 600, "cpu": 300}`; `run_outer` (line ~248) starts the child in its own session and `killpg`s it at the deadline; `_write_exclusive` writes only for `RUN_COMPLETED`. | `test_hanging_child_is_killed_at_the_outer_deadline`, `test_grandchildren_of_a_hanging_child_are_killed_too`, `test_no_receipt_when_the_outer_deadline_fires`, `test_no_receipt_unless_the_run_completed`, `test_cpu_reference_has_its_own_300_second_budget` | **CLOSED** (deadline enforced by the parent, no receipt on deadline). Residual: C-7, C-9. | A grandchild that calls `setsid` leaves the session group; not tested. |
| **N-7** | LOW | Memory-field scope wrong or missing (`REVIEW_ITER2B` §3) | `503d2ba` | `real_engine.py:1113–1131` `MEMORY_FIELD_SCOPES` glossary, attached as `memory_field_scopes` to every arm's telemetry. `final_retained_kv_payload_bytes` is labelled a final-state snapshot. | `test_every_memory_field_declares_its_own_scope[arm]` (3 arms) | **CLOSED** for the four byte fields. Residual (C-10): the block-level `kv_memory_accounting.scope` string still reads "per decode step, prefill cache unbounded" for the whole block. | — |
| **N-8** | LOW | Matched-retention mapping holds only for text-only prompts (`REVIEW_ITER2B` §3) | `fe6f255` | `h2o.py:466–…` `calibrate_h2o_ratio_to_retained(…, *, image_present: bool)`: required keyword, refuses image prompts with a "text-only" message. | `test_calibration_refuses_image_prompts_and_says_why`; `test_matched_retention.py` 8 passed in the targeted run | **CLOSED** | — |
| **N-9** | LOW (latent) | `_engine_gate` returns FAIL when arms disagree, including NOT_APPLICABLE (`REVIEW_ITER2B` §3) | `b5698c9` | `run_progressive_qualification.py:698–727`: NOT_APPLICABLE arms are excluded for shared gates; applicable disagreement stays FAIL by design. | `test_an_arm_reporting_not_applicable_does_not_make_a_shared_gate_fail`, `test_two_different_applicable_labels_are_still_a_fail`, `test_a_gate_that_no_arm_applies_to_is_not_tested` | **CLOSED** | Disagreement among applicable arms is still FAIL, which is intended. |
| **N-10** | LOW | Assert-based oracles plus `guarded` pass under `python -O` (`REVIEW_ITER2B` §3) | `9c1bc3e` | `run_progressive_qualification.py:665–…` `require()` raises `OracleFailure`. Repro of the old risk: `grep -rn "^\s*assert " scripts src --include=*.py` returns nothing. `require(` occurs 111 times. | `test_the_qualification_script_contains_no_assert_statements`, `test_under_python_O_a_failed_oracle_is_recorded_as_fail_not_pass` | **CLOSED** | — |
| **N-11** | LOW | Engine `SEVEN_ARM_PRETRAINED_PASS` is a construction flag (`REVIEW_ITER2B` §3) | `e97860d` | No live code uses the retired label. The gate is `SEVEN_ARM_HOOK_COVERAGE_PASS`. | `test_the_retired_seven_arm_label_is_absent_from_source_scripts_and_tests`, `test_the_engine_emits_the_hook_coverage_label`, `test_seven_arm_gate_is_named_for_what_it_measures` | **CLOSED** in code. Stale docs still name the retired label: C-6. | Repro: `grep -rn "SEVEN_ARM_PRETRAINED_PASS" docs` (4 hits outside the review files). |
| **N-12** | LOW | Per-head docstring says budget untouched; code sets it (`REVIEW_ITER2B` §3) | `c7d8061` | `h2o.py:246–…` `compute_eviction_plan_per_head` is pure: the budget is computed for the call only, and scores are padded in a local copy. | `test_per_head_plan_with_unset_budget_computes_it_without_storing_it_or_padding_scores`, `test_per_head_plan_does_not_touch_the_layer_mask_or_budget`; `test_h2o_per_head.py` 33 changed lines | **CLOSED** | — |
| **N-13** | LOW (latent) | Per-head flag silently ignored on the STARLIGHT candidate path (`REVIEW_ITER2B` §3) | `4a8523d` | `h2o.py:305–311`: `keep_mask_without_protection` raises `NotImplementedError` when the flag is set. | `test_starlight_candidate_path_refuses_the_per_head_flag_instead_of_running_the_mean` | **CLOSED** | — |
| **S-9** | LOW | DA is scripted, not model-declared; the name must say so in results (`REVIEW_ITER2.md`) | none directly | Telemetry label `da_scope_provenance: SCRIPTED_SEGMENT_DIAGNOSTIC_NOT_MODEL_DECLARED` (`real_engine.py:~2879`). Arm IDs `da_base` and `da_mtp` unchanged. | Not directly tested (repro: `grep -n SCRIPTED_SEGMENT scripts src -r`). | **OPEN (partial)** | The label is in telemetry only. Arm names and summary tables do not say "scripted". Claim wording needs the same rule. |
| **I-2** | LOW | Committed inventory not at HEAD (`REVIEW_ITER2.md`, `REVIEW_ITER2B`) | `cdc842a` | `PILOT_INVENTORY.json` `head` = `e97860d`, `script_sha256` matches the current script; freshness test fails on a stale script. | `test_the_committed_inventory_was_generated_by_the_current_script`, `test_the_recorded_head_is_a_commit_in_this_history`, `test_inventory_freshness.py` | **CLOSED** (fresh against the current script; `head` is an ancestor, not necessarily HEAD) | — |
| **I-3** | MEDIUM (claim) | "No READY possible" claim in `ITER2_D2_REPORT.md:30` (`REVIEW_ITER2.md`) | `ITER2_D2_REPORT.md` 2-line edit in the D9–D11 round | `ITER2_D2_REPORT.md:31` now says the live probe "refuses a signed plan with a forged environment". | None on the sentence itself. | **OPEN** | The sentence overstates the protection in the other direction: R1–R3 of N-1 still pass with a live probe. Fix text to list what the live probe does and does not attest. |
| **A-2** | owner ruling | No eligible candidate and budget unmet: status or error? (`STARLIGHT_INVARIANTS_ORACLE.md:33,170`) | `133d5c5` (behaviour), no ruling | `starlight.py:214–219`: budget unmet with no protection returns `STATUS_INFEASIBLE` with reason `INFEASIBLE_UNMET_AFTER_ELIGIBLE`. The status string `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET` is reused for a case where nothing is protected. A `ValueError` remains only for caller error (candidates supplied with `length <= budget`, line ~206). | `test_HC11_*` (oracle), `test_candidates_leaving_budget_unmet_return_explicit_infeasible_status_not_an_error` | **OPEN** | (a) Status naming conflates causes; the oracle proposed a separate status. (b) Design §12 and REPORT:74 still say "raises ValueError" (C-6). Owner ruling needed. |
| **A-4** | owner ruling | Structural tokens: protect or only measure? (`STARLIGHT_INVARIANTS_ORACLE.md:43,70,172`) | `d8ebc16` lineage; lead Q2 in D4 | `real_engine.py:1036` `STRUCTURAL_POLICY_LABEL = "PROTECTED_LEAD_DECISION_Q2"`; `vision_start`/`vision_end` are protected through the structural source. | Mutation `I8-mut` (D7 scratch, `st_slots = {}`): 3 failed (`test_HC13_vision_start_and_end_are_protected_and_never_evicted`, `test_vision_tokens_are_protected_and_never_evicted_under_pressure`, `test_vision_token_candidate_is_vetoed_at_helper_level`). | **OPEN (not ratified)** | Implemented as a lead decision, not owner-ratified. Stale text says "NOT protected": `STARLIGHT_STAGE1_DESIGN.md:196,219,229` and `STARLIGHT_STAGE1_REPORT.md:66,72–74` (C-6). |
| **A-5** | owner ruling | Missing declarations: fail-closed or conservative? (`STARLIGHT_INVARIANTS_ORACLE.md:112,173`) | none | `real_engine.py:2388` sets `NO_EVIDENCE_DECLARED`; the arm protects nothing. Not fail-closed. | `test_no_grounding_ablation_is_nominal_without_any_evidence` checks the nominal case only. | **OPEN** | HC-7 requires fail-closed or an explicit error. The implementation is explicit but not fail-closed. Owner must choose. Repro: `pool=None` on `starlight_s1` gives `NO_EVIDENCE_DECLARED` and protects 0 grounding positions. |
| **A-6** | owner ruling | Eviction timing: prefill or decode? (`STARLIGHT_INVARIANTS_ORACLE.md:174`) | none | Every decision record carries `"phase": "decode"` (`real_engine.py:~2200`). Prefill is never evicted. | `test_starlight_oracle.py:139` asserts `rec["phase"] == "decode"`. | **CLOSED** as to the oracle requirement (the choice is stated in each record). Policy acceptance of decode-only is still an owner call. | — |

---

## 2. Re-review of the D9–D11 fix round

The fix round is 19 commits, `9b332bc`..`d8ebc16`. Checks run:

1. **Removed assertions in tests.** The test-file diff removes 7 assertion lines. All 7 are the retired `SEVEN_ARM_PRETRAINED_PASS` label, replaced by `SEVEN_ARM_HOOK_COVERAGE_PASS`. Two replacements are stronger (`not any("PRETRAINED" in key …)`). No other test assertion was removed, and no test function was removed.
2. **Asserts in source.** The `assert` statements in `scripts/run_progressive_qualification.py` (about 110 before the fix round) are now `require()` calls (111). `grep -rn "^\s*assert " scripts src` is empty. Spot-checked the conversion: the same expressions, now raising `OracleFailure`.
3. **Live probe (N-1).** Reviewed as §1. Strength: the probe runs in a separate interpreter with `-I`, uses a fixed binary path, and checks a nonce. Weakness: R1–R3 in §1.
4. **Scorer (N-3, N-4).** Reviewed as §1. Port logic matches the transcription. The transcription matches upstream except the documented prints and `eval`, plus an undocumented `except:` → `except Exception:` (C-8).
5. **GPU smoke (N-5).** Parent-enforced deadline with `killpg`; receipt written only for `RUN_COMPLETED`. Probe script `probe_smoke_forward.py` writes no receipt. `compare_receipts` uses `zip(strict=True)`, refuses non-completed or different-SHA receipts, but does not check which receipt is CUDA and which is CPU (C-9).
6. **Per-head and matched retention (N-8, N-12, N-13).** Default path unchanged (`test_default_flag_is_off…`). Image prompts refused. STARLIGHT path refuses the flag.
7. **Gates (N-9, N-11).** Applicable disagreement still FAIL (intended). Old label absent from live code.

---

## 3. New findings from this review

| ID | Severity | Owner | Location | Finding | Reproducer | Fix |
|---|---|---|---|---|---|---|
| C-1 | HIGH | wF:p8 | `harness_inventory.py:232`, `:725`, `:489` | Residual N-1: zero-weight pinned manifest verifies; operator-chosen weight pins outside the owner record; unsigned receipt index. A passing live probe gives 528 READY. | `scratchpad/it3/fr/n1_forge.py mock_live` (with `PYTHONPATH` pointing at the vLLM stub) | Require all `*.safetensors` in the manifest; move weight pins and the receipt index into the owner-signed record; sign the index. |
| C-2 | MEDIUM | wF:p8 | `harness_inventory.py:322–340` | `verify_vllm` runs in-process without `-I`, so a `PYTHONPATH` stub with `__version__` and an empty `_C` passes. The report then says `vllm_installed: True`. The live probe still refuses it (no distribution metadata), so READY is blocked, but the reported environment is wrong. | `PYTHONPATH=<stub> python scratchpad/it3/n1_forge.py real` → `env vllm_installed: True`, live probe `vLLM distribution metadata missing` | Drop the in-process check, or run it under `-I` and require distribution metadata. |
| C-3 | MEDIUM | wF:p7 (lead for claims) | `starlight.py:424–446`, `grounding_evidence.py:43–52` | "Verified" and `source_sha256` are caller assertions. The controller never checks evidence content. For text-only requests any hash is accepted (`source_unchecked_text_only`). `EvidenceItem.verified` defaults to True. | Scratch probe `E2`: a bogus `ff…` hash on a text-only request gives `valid`. | Do not call the signal "verified". Rename to `declared_valid`, or add a real check that the evidence matches the source. |
| C-4 | MEDIUM | wF:p7 | `starlight.py:470–478` | Keyword breadth is unbounded. A declared common word protects every occurrence in the prompt. | Scratch probe `E1`: keyword `the` protects 4 of 16 prompt positions. | Require a relevance or frequency bound, or reject stopwords and high-frequency keywords. |
| C-5 | HIGH (claim) / MEDIUM (code) | wF:p7 (lead for claims) | `starlight.py:449–480`; `real_engine.py:2399–2402` | Image evidence protects no visual tokens. The matcher works on text token IDs. Bounding boxes (`bounding_box`, set in `adapters.py:488`) are never read by the protection code. Visual pad tokens cannot match a keyword. | Scratch probe `E3`: 0 image-pad positions protected; `bounding_box consumed by protection code: False`. | Either protect visual token spans from the box, or state that image-region evidence is not protected. |
| C-6 | MEDIUM (docs) | lead | `STARLIGHT_STAGE1_DESIGN.md:196,217,219,226,229`; `STARLIGHT_STAGE1_REPORT.md:66,72–74`; `MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:34`; `CLAIM_RECONCILIATION.md:78` | Stale text contradicts the code. I-8 "not protected" (code protects); A-2 "raises ValueError" (code returns INFEASIBLE); retired label `SEVEN_ARM_PRETRAINED_PASS` still named. | `grep -rn "NOT protected\|raises ValueError\|SEVEN_ARM_PRETRAINED_PASS" docs` | Update those lines, or mark them superseded by §13 of the design. |
| C-7 | LOW | wF:p8 | commit `e68945c`; `docs/harness/GPU_RECEIPT_INDEX.json` | The commit says "receipt in artifacts/runs". The receipt `20261109T200806Z_pytorch_reference_gtx1080ti_smoke_gpu_fp16.json` exists only in `wt/it2-p8/artifacts/runs` (gitignored, `.gitignore:18`). The index has 0 entries, so `gpu_smoke_receipt` stays NOT_TESTED. The receipt itself reads RUN_COMPLETED, label `…not_vllm_not_performance`, git_sha `960fb17`. | `python3 -c "import json;print(len(json.load(open('docs/harness/GPU_RECEIPT_INDEX.json'))['receipts']))"` → 0 | Commit the receipt and index it, or say it is not in version control. |
| C-8 | LOW | wF:p8 | `tests/mmlong_reference_transcription.py` vs upstream `eval/eval_score.py` (sha256 `cafa2683…`) | Undocumented change: bare `except:` became `except Exception:` in `is_float_equal`, `eval_score` and `eval_acc_and_f1`. The header lists only prints and `eval`. No result change on tested inputs (`KeyboardInterrupt` is not caught either way). | `diff` of the two files (text only; the upstream file was not executed) | Add the change to the header. |
| C-9 | LOW | wF:p8 | `pytorch_reference_gpu_smoke.py:79–101` | `compare_receipts` takes two positional receipts and checks status and SHA, but not that the first is CUDA and the second CPU. Two GPU receipts would compare without error. | `compare_receipts(gpu_receipt, gpu_receipt)` returns a comparison | Check `device` and `tag` for each argument. |
| C-10 | LOW | wF:p7 | `real_engine.py:2786–2791` | N-7 residual: the block string `kv_memory_accounting.scope` still says "per decode step, prefill cache unbounded" for a block that now contains a final-state field. The field-level glossary is correct. | Read the telemetry dict | Rename the block scope, or drop it in favour of the glossary. |
| C-11 | LOW | lead | `run_progressive_qualification.py`, `real_engine.py:2256` (not re-checked) | Not re-verified in this review: the engine literal `SEVEN_ARM_HOOK_COVERAGE_PASS` derivation. Covered by the label-consistency test only. | — | — |

---

## 4. Part 2: is STARLIGHT's internal protection evidence protection or keyword preservation?

**Answer: keyword preservation of declared text spans.** The signal does not measure evidence support.

What the code does (`starlight.py:449–480`, `real_engine.py:2387–2402`):
1. The caller declares evidence items with a boolean `verified`, a `source_sha256`, and a list of keywords (`grounding_evidence.py`).
2. `evidence_validity` marks an item valid unless `verified` is False, the request id differs, or the declared hash differs from the image hash. Text-only requests are valid with `source_unchecked_text_only`.
3. Every occurrence of each keyword (bare, space-prefixed, capitalised) in the **prompt token IDs** is protected. The protected positions are mapped to physical cache entries.
4. The protected positions veto H2O candidates (`eligible = candidates − protected`).

Assessment against the four questions:

| Question | Assessment | Evidence |
|---|---|---|
| **Trustworthy?** | No. `verified` and `source_sha256` are caller assertions. The controller never reads the evidence content, and text-only requests accept any hash. | C-3, probe E2 |
| **Relevant to the question?** | Not measured. Keywords are caller-declared. Nothing ties them to the question or the answer. `gold_answer_prohibited` is a caller flag, not checked. Relevance labels were never produced (HC-5 `NOT_MEASURED_NO_RELEVANCE_LABELS`, REVIEW_ITER2 row HC-5). | C-4, REVIEW_ITER2 HC-5 |
| **Causal for eviction?** | Yes, for text prompts. Protection changes which entries H2O may evict. The toggle is tested: `test_A_TOGGLE_grounding_signal_changes_the_retained_set_on_the_grounding_fixture` (`tests/test_starlight_oracle.py:274`) and `test_grounding_h2o_conflict_is_vetoed_not_evicted`. | Tests pass (91 passed) |
| **Evidence protection for image evidence?** | No. Visual tokens are never protected by grounding, so an image-region declaration protects nothing visual. | C-5, probe E3 |
| **Post-hoc telemetry?** | Not post-hoc: it is in the eviction path. But the telemetry (`grounding_protected` counts) reports protection, not usefulness. | `real_engine.py:2399–2402` |

**What can be claimed now (text-only prompts, CPU reference):**
- Under budget pressure, token spans of caller-declared keywords in the text prompt are excluded from H2O eviction, and the retained set changes accordingly (tested).
- Protection is recorded per entry with reason codes, and vetoes are counted.

**What cannot be claimed without support measurements:**
- That the protected evidence is trustworthy, verified, or grounded in the image or document.
- That the protected keywords are relevant to the question, or that they support the answer.
- That protection reduces hallucination, improves accuracy, or preserves answer quality. This needs quality deltas with and without grounding at matched retention (matched retention is calibrated for text prompts only, N-8). None exist in the repo.
- Anything about image-region or visual-token grounding. Bounding boxes are not used by the protection.
- That the internal signal is "evidence protection" in the sense of the owner contract. It is keyword preservation of declared prompt text.

**Required support measurements before any evidence claim:** relevance labels per protected keyword (HC-5), answer accuracy with and without grounding at matched retention, a verified evidence source check (or an explicit "declared, not verified" name), and a visual-region protection path if image evidence is to be claimed.

---

## 5. Items I could not close

- **N-4 differential against the executed upstream scorer**: not run (fetched external code was blocked in my session). The static diff is C-8.
- **N-1 R1–R3**: reproduced, not closed. Needs an owner-signed record that covers the weight pins and the receipt index.
- **A-2, A-4, A-5, A-6**: owner rulings. A-6 has its oracle requirement met.
- **Stage 2 serving and policy tests** (`test_mtp_speculative_serving.py`, `test_policy_mtp_speculative.py`): timed out at 580 s with no counts. Not closed and not evidence of a pass. Needs a rerun with a longer budget, run alone, by the lead.
- **Stage 2 mutation rerun**: not in scope for this task.
- **Full suite**: not run, per the host-load policy. The lead runs it.

## 6. Commands

```
cd /mnt/ssd/genos2/research_workspace/wt/it3-p9
PYTHONPATH=src CUDA_VISIBLE_DEVICES= /mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_harness_forgery.py tests/test_harness_cli.py tests/test_harness_live_probe.py tests/test_harness_inventory.py tests/test_inventory_freshness.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES= /mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_gpu_smoke_outer_deadline.py tests/test_gpu_smoke_helpers.py tests/test_mmlong_official_scorer.py tests/test_mmlong_fuzz_vs_reference.py tests/test_oracle_checks_under_O.py tests/test_qualification_gates.py tests/test_gate_label_consistency.py tests/test_h2o_per_head.py tests/test_matched_retention.py tests/test_dtype_selection.py tests/test_parity_tolerance.py tests/test_multimodal_fixture.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES= /mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_starlight_oracle.py tests/test_starlight_engine.py tests/test_starlight_audit_fixes.py tests/test_starlight_d4.py tests/test_starlight_policy.py tests/test_starlight_signals.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES= /mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_mtp_speculative_serving.py tests/test_policy_mtp_speculative.py
```

Scratch reproducers (outside the repo, not committed): `scratchpad/it3/n1_forge.py` (inventory forgeries; the `fr/` tree is generated), `scratchpad/it3/ground_probe.py` (grounding probes E1–E3).

Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>
