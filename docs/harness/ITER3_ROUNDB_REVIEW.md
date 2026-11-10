# ITER3 Round B closure review (wF:p9)

- **Reviewer:** wF:p9, adversarial scientific auditor (Claude Haiku 5.5; not an Opus review).
- **Reviewed HEAD:** `adcf942c2aea9d7b08a47d01b7ec06b423491c0b` (branch `iter2/lead`, integrated head).
- **Pre-round-B base:** `c65b84d`. Diff reviewed: `git diff c65b84d..adcf942` (21 files, +1984 / -174). Source and tests were read directly; worker summaries were not used as evidence.
- **Review branch:** `i3c/p9` (worktree `wt/i3c-p9`). Source edits: none. Tests added: `tests/test_r4b_review_wF_p9.py`. Mutations ran on `git archive` copies of `adcf942` in scratch space, never in the repo.
- **Environment:** CPU only; `PYTHONPATH=src CUDA_VISIBLE_DEVICES=`; interpreter `colab-vlm-bench/.venv`; heavy runs under `flock wt/heavy.lock timeout`. No GPU, no installs, no push or merge.

## Verdict

**Round B mechanisms are verified, but the STARLIGHT image arm is not a sparsity method, so image-prompt STARLIGHT results are BLOCKED.**

- The S1 gate, the visual-mapping arithmetic, the speculative route labels, the speculative rollback and the shared-set controls all behave as documented on the cases tested.
- F-1 (HIGH, pre-existing at `c65b84d`, open) means that on image prompts with the default DA scope, STARLIGHT never evicts. Its telemetry reports this explicitly, so it is not hidden, but it is undocumented and it invalidates `starlight_s1` versus `h2o_shared_base` comparisons on image tasks until an owner decision is made.
- Several round-B tests passed without testing their claim (F-2 to F-5). Each is now covered by a new test that kills the corresponding mutant. The per-layer retention telemetry remains self-reported (F-6, open).
- Text-only paths are acceptable, subject to the LOW items (F-7 to F-10), which are open and pinned as strict xfails.

## Commands and results

| # | Command (run from `wt/i3c-p9`, interpreter `colab-vlm-bench/.venv/bin/python`, `PYTHONPATH=src CUDA_VISIBLE_DEVICES=`) | Result |
|---|---|---|
| 1 | `python -m pytest ...` with the system `/usr/bin/python` | Not run: no pytest installed there. Corrected to the brief's interpreter. |
| 2 | `pytest tests/test_i3b_s2_visual_mapping.py tests/test_i3b_s2_visual_protection.py tests/test_i3b_s2_speculative_route.py tests/test_r2_c5_visual_evidence.py` | **85 passed** (6 s, no skips; real-processor tests ran against the local 0.8B processor) |
| 3 | `pytest tests/test_i3b_s2_visual_protection.py tests/test_i3b_arm_consistency.py` | **30 passed** |
| 4 | `flock heavy.lock timeout 520 pytest tests/test_i3b_s2_visual_engine.py` | **5 passed** (20 s) |
| 5 | `flock heavy.lock timeout 520 pytest tests/test_i3b_s2_combined_speculation.py tests/test_iter3_b2_mtp_rollback.py` | **14 passed** (200 s; no timeout) |
| 6 | `pytest` regression batch: `test_r3_review_c11_labels`, `test_r2_c4_keyword_breadth`, `test_r2_c3_provenance`, `test_r2_c6_doc_stale_strings`, `test_i3a_s4_cap_policy`, `test_i3a_h2_shared_base`, `test_i3a_s1_evidence_dimensions`, `test_i3a_s3_protected_budget`, `test_iter3_b3_doc_matrix`, `test_harness_inventory`, `test_harness_r2_regressions`, `test_harness_r3_regressions`, `test_harness_pins`, `test_starlight_audit_fixes`, `test_r2_shared_candidates` (heavy lock) | **165 passed** (25 s) |
| 7 | `pytest` qualification batch: `test_qualification_gates`, `test_qualification`, `test_official_qualification`, `test_iter3_b1_official_oracle`, `test_iter3_b1_policy_invariants`, `test_starlight_engine`, `test_starlight_policy`, `test_starlight_signals`, `test_starlight_oracle`, `test_starlight_d4` (heavy lock) | **191 passed** (31 s) |
| 8 | `pytest tests/test_r4b_review_wF_p9.py` (final state, heavy lock) | **6 passed, 3 xfailed** (27 s). The 3 xfails are the strict defects F-1, F-7 and F-8. |
| 9 | `pytest --runxfail tests/test_r4b_review_wF_p9.py` (earlier 6-test version) | 3 failed (the xfails) for the intended reasons: `cells == []`, `evicted_total == 0`, `speculation_executed is True`. |

The full suite was not run, per the brief. Timeouts: none. Every heavy job ran serially under the lock.

## Mutation and negative-control results

Mutants are exact single-string edits on `git archive adcf942`. The driver asserts each anchor occurs once and restores the file, verified with `cmp`. "Survived" means the named tests passed with the mutant applied.

| Mutant | Original suite | After this review |
|---|---|---|
| VM_protect_without_s1 (S1 gate removed) | KILLED | — |
| VM_inner_cells_only | KILLED | — |
| VM_multi_run_ignored | KILLED | — |
| VM_span_length_check_removed | KILLED | — |
| VM_grid_w_unchecked | KILLED | — |
| VM_source_sha_reason_removed | KILLED | — |
| EV_identity_ignored | KILLED (S1 file) | — |
| EV_accept_declared_always | KILLED (S1 file, not the mapping file) | — |
| EV_caller_flag_ignored | KILLED (S1 file) | — |
| EV_bytes_mismatch_ignored | KILLED | — |
| EV_bytes_verified_not_protecting | KILLED | — |
| SR_da_branch_removed | KILLED | — |
| SR_image_branch_removed | KILLED | — |
| SR_serving_label_swapped | KILLED | — |
| SR_disabled_check_removed | KILLED | — |
| **VM_rowmajor_to_colmajor** | **SURVIVED** (only a square box at the origin is tested, which is symmetric) | KILLED by `test_asymmetric_box_positions_follow_row_major_order` |
| **VM_eps_floor_removed** | **SURVIVED** | KILLED by `test_a_boundary_is_not_lost_from_float_rounding_down` |
| **VM_eps_ceil_removed** | **SURVIVED** | KILLED by `test_a_boundary_is_not_gained_from_float_rounding_up` |
| **VM_min_max_swapped** (`processor_params`) | **SURVIVED** all unit tests; KILLED by the heavy visual-engine test | KILLED by `test_processor_parameters_are_read_from_the_real_processor` |
| VM_location_gate_removed | SURVIVED | Equivalent mutant (see F-12) |
| **EN_visual_merge_removed** (visual positions not fed to the planner) | **SURVIVED** all 5 visual-engine tests (F-2) | KILLED by `test_visual_protection_is_retained_under_real_eviction_on_the_no_da_arm` |
| **EN_spec_planner_grounding_dropped** | **SURVIVED** `test_pinv_protected_original_positions_stay_retained...` (F-3) | KILLED by `test_protected_keyword_is_evicted_without_protection_and_kept_with_it_on_the_speculative_path` |
| EN_spec_text_grounding_arg_dropped | KILLED by `test_inferc_text_evidence_runs_policy_speculation...` (my first run targeted the wrong test and reported a survivor; corrected) | — |
| EN_policy_route_image_gate_removed | KILLED | — |
| **EN_retained_count_inflated** (per-layer telemetry always equals the protected count) | **SURVIVED** all suites (F-6) | Still SURVIVED: a limitation, see F-6 |

Distinct mutants: 26. Survivors that remain: EN_retained_count_inflated (F-6), VM_location_gate_removed (equivalent, F-12).

## Findings

Severity: HIGH blocks a claim; MEDIUM is a validity gap in a claim or test; LOW is a defect with no current effect on reported results; INFO is a note.

### F-1 HIGH — open, pre-existing at `c65b84d`: image-prompt STARLIGHT never evicts (full cache)

- **What happens:** with no evidence declared, `starlight_s1` on the fixture image (320-token prompt, 300 visual tokens at span `[4, 304)`, `h2o_budget_ratio` 0.6, budget 192) gives `evicted_total = 0`, `kv_budget_violated = True`, and 12 of 12 layer-steps INFEASIBLE. The planner log shows 321 declared-protected entries in every call, for sequence lengths 322 and 323.
- **Mechanism:** the scripted DA `FOCUS` scope is the whole visual run (`real_engine.py:1009–1016`: `focus_end = sink_end + n_visual`). With `len(protected) > budget`, the planner keeps every entry (`starlight.py:218`, `:231`).
- **Same at the pre-round-B base:** the probe run on `c65b84d` gives the same INFEASIBLE status on all 12 calls with 321 protected. This is not a round-B regression. Round B exposed it through the visual tests, which pass only because nothing is evicted (F-2).
- **Contrast:** `h2o_shared_base` on the same image evicts 792 entries with the budget met (`FIXED_BUDGET_MET`). So `starlight_s1 − h2o_shared_base` on image prompts measures eviction against no eviction, not protection.
- **Not disclosed:** `STARLIGHT_STAGE1_DESIGN.md` §4.1 maps FOCUS to `focus:scripted_visual_segment` but does not say that this protects the whole visual segment on image prompts.
- **Scope of evidence:** one image, 3 decode tokens, budget ratio 0.6 (fixture). The mechanism is structural: infeasible whenever the visual segment exceeds `max(min_budget, ratio × prompt_len)`. At the engine default ratio 0.5, that holds whenever visual tokens exceed half the prompt. The benchmark ratio was not checked.
- **Reproduction:** `probe_plan.py` and `probe_vis.py` (scratch, `i3c_b`) under the heavy lock. Strict xfail: `test_starlight_s1_on_an_image_prompt_evicts_when_the_budget_is_feasible`.
- **Owner decision needed:** whether the scripted FOCUS scope should cover the whole visual segment, and whether image STARLIGHT runs should be labelled full-cache.

### F-2 MEDIUM — closed by new test: the visual retention test is vacuous on `starlight_s1`

- `test_protected_visual_positions_are_retained_in_every_full_attention_layer_at_the_end` passes with visual positions removed from the planner (`EN_visual_merge_removed`, survived 5/5). That arm never evicts (F-1), so "retained" is trivially true. The test has no precondition that eviction occurred.
- **Closed for the no-DA arm:** `test_visual_protection_is_retained_under_real_eviction_on_the_no_da_arm` (`starlight_s1_no_da`, `evicted_total = 1062`, budget met). Pristine: 80 visual positions retained in each of 6 layers. Mutant: 64. The test kills the mutant.
- **Recommendation:** add `evicted_total > 0` to the existing test or retire it.

### F-3 MEDIUM — closed by new test: the speculative invariant test protects a token that is never at risk

- `test_pinv_protected_original_positions_stay_retained_at_every_speculative_step` protects `Paris` (token 5). Probe on the unprotected speculative run: evicted original positions are 8, 9, 11–35. Token 5 is never evicted, so the test passes with protection removed (`EN_spec_planner_grounding_dropped` survived).
- **Closed:** `test_protected_keyword_is_evicted_without_protection_and_kept_with_it_on_the_speculative_path` uses `Berlin` (token 12), with a negative control that asserts the unprotected run evicts it. Mutant: `protected_evicted == {12}`.

### F-4 MEDIUM — closed by new test: the row-major order is not tested for a non-symmetric box

- `test_positions_are_span_start_plus_row_major_index` uses a 4×4 box at the origin, which is symmetric under transposition. A transposed formula gives the same set (`VM_rowmajor_to_colmajor` survived).
- **Closed:** `test_asymmetric_box_positions_follow_row_major_order` (2×4 box). Mutant killed.

### F-5 MEDIUM — closed by new test: processor-to-parameter mapping is not checked against the processor

- `processor_params` maps `size.shortest_edge` to min and `size.longest_edge` to max. The mapping tests pass their own constants, so a swap survived every unit test (`VM_min_max_swapped`) and was caught only by the heavy engine test.
- **Closed:** `test_processor_parameters_are_read_from_the_real_processor` checks the local processor (patch 16, merge 2, min 65536, max 16777216). The code is correct today; this closes the test gap.

### F-6 MEDIUM — open (limitation): per-layer retention telemetry is self-reported

- `visual_protected_retained_final_by_layer` and `grounding_protected_retained_final_by_layer` are computed in the same code they describe. The only check is equality with the protected count. Inflating the count (`EN_retained_count_inflated`) survives every suite, including the new no-DA test.
- **This confirms p7's note** that the per-layer retention check was not mutation-tested. It is now tested, and it is not discriminating.
- **Recommendation:** check retention independently, e.g. with a spy on the planner's kept slots, as the new speculative test does.

### F-7 LOW — open, strict xfail: sub-tolerance box maps to no cell

- A positive-area box straddling a cell boundary by 1e-10 px gives `r0 > r1` and returns `[]`, against the module's positive-overlap superset contract. Strict xfail: `test_sub_tolerance_box_straddling_a_cell_boundary_keeps_its_positive_overlap` (observed `cells == []`). Real annotation boxes are far larger than the 1e-9-cell tolerance, so no reported result is affected.

### F-8 LOW — open, strict xfail: speculation ablation is not honoured on the serving `mtp` arm

- `_mtp_text_speculation_route` (`real_engine.py:1755`) does not read `_speculation_enabled`, unlike the policy route (`:1615`). The flag is set only by tests; production code never sets it. Observed with the flag off: `speculation_executed = True`. Strict xfail: `test_speculation_ablation_is_honoured_on_the_serving_mtp_arm`.

### F-9 LOW — open: telemetry overclaim on partial cells

- `visual_mapping.py:372` discards `_info`. The module docstring (line 16) says the count of partially covered cells is reported, but no telemetry key carries it.

### F-10 LOW — open: no route label on the serving `mtp` text path

- `_execute_mtp_speculative_text` (`real_engine.py` ~1764–1850) has no `speculative_route` key. The label appears on the policy text path (`:1686`) and the diagnostic path (`:2931`), so the route that actually executes on the `mtp` arm is unlabelled.

### F-11 INFO — `DISABLED_ABLATION` label on control arms

- `visual_evidence_protection` is `DISABLED_ABLATION` for arms without grounding, including the control `h2o_shared_base`, which is a control and not an ablation. The label is misleading but harmless to the protection logic.

### F-12 INFO — equivalent mutant: the location gate is redundant

- `visual_mapping.py` checks `location_valid` after `region_positions`, which already validates the same box bounds in `_box_values`. Removing the check changes nothing. It is not a defect, and the S1 location dimension is still reported.

## Closed vs unresolved

**Closed in this review (new tests, each kills its mutant):** F-2 (no-DA arm), F-3, F-4, F-5, and the two float-boundary mutants (`VM_eps_floor_removed`, `VM_eps_ceil_removed`).

**Verified without new tests (regression and mutation-killed):**
- S1: tri-state dimensions; `accept_declared` defaults to False; text-only items do not protect by default; identity, hash mismatch, caller flag and UNCERTAIN are all refused even under `accept_declared`. All EV mutants killed.
- Visual mapping: geometry matches the real processor on six sizes; patch order matches the processor; multi-image, token-run and grid mismatches are refused with reasons; S1 gate kills protection; original positions are followed through compaction on the no-DA arm.
- Speculative route statuses, `da_mtp` non-speculative, image prompts UNSUPPORTED on the combined path (all killed).
- Speculative rollback: 14 tests pass under the lock.
- `h2o_shared_base`: same budget formula and shared selection as STARLIGHT (budget-matched). See F-1 for the image confound.
- Keyword caps, stopwords, C-3/C-4/C-5 regressions, C-11 labels and qualification gates: 165 + 191 passing.
- Documentation: the added claims I checked use "not verified" and "UNKNOWN" consistently. I found no PASS overclaim in the added lines.

**Unresolved:** F-1 (owner decision), F-6 (self-reported telemetry), F-7, F-8, F-9, F-10, F-11 (LOW or INFO, open or pinned as xfail).

## Qualification limitations

- One fixture image, one question, 3 decode tokens for the STARLIGHT image runs; budget ratio 0.6 in the visual tests. F-1 was not measured on any benchmark dataset.
- `h2o_base` was not probed on the image: the probe reads the STARLIGHT telemetry block, which that arm does not produce. The `h2o_shared_base` contrast in F-1 is the only control measured on the image.
- CPU PyTorch reference on Qwen3.5-0.8B. Not vLLM. No GPU. No speed or quality claim.
- The 4B checkpoint is not present, and its processor was not checked. Processor parity is shown for the local 0.8B only.
- `verified` remains a caller declaration (S1 documents this). The semantic dimensions are NOT_ASSESSED and are never VERIFIED.
- The policy values (`accept_declared` default, keyword caps 0.2 and 0.5, the positive-overlap superset) are engineering hypotheses, not owner-ratified.
- Mutation coverage: 26 distinct mutants on this head. Not exhaustive. The round-2 and round-3 mutants were not re-run; they are covered only by the regression batch (item 6).
- The full suite was not run (lead's job per the brief).

## Process notes

- The first pytest attempt used the system interpreter, which has no pytest. Corrected to the brief's venv.
- My mutation driver's summary filter matched a traceback context line containing " failed", which briefly misreported `EN_spec_planner_grounding_dropped` as failing on its baseline. A manual run showed the real failure: `protected_evicted == {12}`. The kill is real; the reported line was not the failing assertion.
- Two defects in my own new test file (missing import; a pool bound to the wrong request id) were fixed before the commit. The fixture error had been reported as an xfail at setup, which a strict xfail would hide; after the import fix, each xfail fails on its own assertion.
- Scratch artefacts are outside the repo: `i3c_b/`, `i3c_m/`, `i3c_base/`, and `probe_*.py` under the session scratchpad.

## Addendum: repairs (wF:p9 final inspection of p7's repairs)

- **Reviewed SHA:** `1144291` (branch `i3d/p9`, integrated head incl. p7's repairs). Diff reviewed: `git diff adcf942..1144291 -- src scripts tests docs`.
- **Scope:** only the materially changed code since `adcf942`: F-1 labelling (`comparison_validity`), F-6, F-7, F-8, F-9, F-10, F-11. No source edits by this reviewer.
- **Tests added by this addendum:** one light test, `test_comparison_validity_never_calls_an_evicting_run_full_cache` (2 parametrised cases), in `tests/test_r4b_review_wF_p9.py`. It closes a surviving mutant (see N-0 below).
- **Strict xfails:** the F-7 and F-8 xfails were removed by p7 once the defects were fixed. The test bodies are unchanged (checked against my commit `951fffd`). This is a promotion, not a weakening. The F-1 strict xfail (`test_starlight_s1_on_an_image_prompt_evicts_when_the_budget_is_feasible`) is still present and still fails at HEAD.

### Repair verdicts

| Item | p7 claim | Verdict | Evidence |
|---|---|---|---|
| F-1 labelling | Image full-cache runs labelled `STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE`; DA scope unchanged | **Closed as labelling** (scope still open, owner decision) | Heavy `test_f1_full_cache...` passes; baseline shows the label. Mutant `CV_never_set_on_starlight` killed; `PAP_da_spans_dropped` killed. Light parametrised cases pass. |
| comparison_validity: no-eviction rule | Label only when `evicted_total == 0` and protection ≥ budget | **Correct** on every case tested | Mutant `CV_no_eviction_condition` survived the original light tests (no case had eviction with protection ≥ budget). Added test kills it. Mutant `CV_ge_to_gt` killed by the boundary case (protection == budget is labelled). |
| F-6 visual | Independent check from the planner's returned slot map; eviction precondition | **Works as described, with a stated limit** (N-4) | Combined mutant (planner drops visual protection + telemetry = protected count): killed, telemetry 80 vs independent 64 on layer 7. Pure telemetry +1 lie: killed (81 vs 80). Inflated telemetry whose value equals the true retention survives, correctly (not a false claim). |
| F-6 speculative | Independent check from the policy `next_orig`; eviction precondition | **Works as described** | Combined mutant (planner ignores grounding + telemetry = protected count): killed, telemetry 1 vs independent 0 on layer 23. |
| F-7 sub-tolerance | `_cell_interval` snaps within EPS, keeps raw interval when the snap collapses | **Closed** | Live test passes. Mutants killed: `F7_old_formula` (old formula) and `F7_no_collapse_guard` (always snap). Proof of `r0 ≤ r1`: for a positive-width box, `ceil(s1) − 1 ≥ floor(s0)`. |
| F-8 ablation | `_mtp_text_speculation_route` requires `_speculation_enabled` | **Closed** | Live test passes. Mutant `F8_ablation_ignored` killed (`speculation_executed` True with the flag off). |
| F-9 partial cells | `cells = {full, partial}` in visual items | **Closed** | Light test passes (9 full, 7 partial on the 256-px geometry). Mutant `F9_partial_zeroed` killed. |
| F-10 route label | Serving mtp text path carries `speculative_route` | **Closed** | Heavy test passes. Mutant `F10_label_removed` killed. |
| F-11 labels | Control arm `NOT_APPLICABLE_CONTROL_ARM_*`; ablation `DISABLED_BY_ABLATION`; `DISABLED_ABLATION` removed | **Closed** | Light tests pass. Mutant `F11_control_as_ablation` killed. `DISABLED_ABLATION` no longer appears in `src`, `tests` or `scripts`. |

### New findings (this addendum)

- **N-0 MEDIUM, closed by test:** `comparison_validity` had no case with evictions and protection ≥ budget, so a mutant that labels such a run FULL_CACHE survived. Added test; killed.
- **N-1 MEDIUM, open: the label does not reach any arm record.** `comparison_validity` is written to `result.telemetry` only. `scripts/run_progressive_qualification.py` copies `kv_memory_accounting` into each arm record (`:340`, `:480`) and not `comparison_validity`. No script in `scripts/` reads the label, and `bench.py` does not serialise engine telemetry. The record keeps `kv_memory_accounting.fixed_budget_status`, which is `NOT_A_SUCCESSFUL_FIXED_BUDGET_RUN` on the fixture image, so a reader can see the run was not a successful fixed-budget run, but not the "full cache, not comparable" reason. The oracle doc (`STARLIGHT_INVARIANTS_ORACLE.md` §5, rule 3) requires an infeasible arm to be reported as "not comparable", with the count. That rule is not met on the reporting path. Recommended: copy `comparison_validity` into `arm_info` and into the qualification record, and have the arm report read it.
- **N-2 LOW, open: the speculative text routes carry no `comparison_validity`.** The key is absent, not `NOT_APPLICABLE`, so a reader cannot tell "not computed" from "not applicable". I did not observe a full-cache speculative run. Probe: `starlight_s1_mtp` text, 60 tokens, two keywords at the per-keyword cap (24 positions protected), budget 30: retained 27–30 per step over 42 samples, no step over budget, eviction happening. Full cache on this route would need protection above the budget. The keyword union cap is 0.5 × prompt length, which equals the budget at ratio 0.5, so the keyword channel alone cannot produce it. Severity therefore LOW until a full-cache speculative run is demonstrated.
- **N-3 LOW, docs, open:** `STARLIGHT_STAGE1_DESIGN.md` §4.2 still says `h2o_shared_base` is "Comparable with … `starlight_s1`" with no F-1 qualification. On image prompts this is false for `starlight_s1` (see the control probe below). The §4.1 F-1 paragraph and §19 are correctly qualified.
- **N-4 INFO, test design:** the F-6 checks fail when the reported count disagrees with the planner's slot map. They do not fail on an inflated value that happens to equal the true count (`F6_visual_inflated` and `F6_spec_telemetry_inflated` survived with the planner intact; the true retention is full in those runs). That is correct behaviour. The limit is that both sides come from the planner's own `next_orig` bookkeeping, so a bug in original-position tracking would appear in both and would not be caught.
- **N-5 INFO, wording:** §4.1 says that whenever the visual token count "reaches" the budget every step is INFEASIBLE. The condition in the code is protection > budget (`starlight.py:218`). With protection == budget, a step can still evict. The label itself is correct because it requires `evicted_total == 0`.
- **N-6 INFO, control arm on images:** the control probe on the same fixture image: `h2o_shared_base` evicts 792 entries with the budget met (`FIXED_BUDGET_MET`). So on image prompts `starlight_s1 − h2o_shared_base` measures eviction against no eviction, not protection. This is the F-1 consequence that N-3 should state.

### Commands (addendum)

| Command | Result |
|---|---|
| Light set: `test_i3c_p7_repair.py`, `test_i3b_s2_visual_mapping.py`, `test_i3b_s2_visual_protection.py`, `test_i3b_s2_speculative_route.py`, `test_r2_c5_visual_evidence.py`, `test_i3a_s1_evidence_dimensions.py`, `test_r4b_review_wF_p9.py` (heavy-marked excluded by `-k`) | **113 passed** (5 s) |
| Heavy baseline under `flock heavy.lock timeout 520`: `test_r4b_review_wF_p9.py`, `test_i3c_p7_repair_heavy.py`, `test_i3b_s2_visual_engine.py` | **19 passed, 1 xfailed** (80 s). The xfail is the F-1 strict test. Waited about 20 min for the lead's suite to release the lock. |
| Heavy mutants (same node set): `F8_ablation_ignored`, `F10_label_removed`, `F6_visual_inflated`, `F6_spec_telemetry_inflated`, `CV_never_set_on_starlight`, `PAP_da_spans_dropped` | killed: F8 (1 failed), F10 (1 failed), CV_never_set (1 failed), PAP (1 failed). Survived: F6_visual_inflated, F6_spec_telemetry_inflated (see N-4). |
| Combined F-6 mutants (`mutD`): `F6_combined_visual`, `F6_combined_spec`, `F6_pure_telemetry_plus_one` | killed all three: 80 vs 64; 1 vs 0; 81 vs 80. |
| Light mutants (`mutC`): `F7_old_formula`, `F7_no_collapse_guard`, `F9_partial_zeroed`, `F11_control_as_ablation`, `CV_no_eviction_condition` (before and after the added test), `CV_ge_to_gt` | killed (after the anchor fix; CV_no_eviction_condition survived before the added test and is killed after it). |
| Speculative text probes (direct, `_speculative_policy_mtp_greedy`) | retained 27–30 against budget 30 over 42 samples; no full-cache run observed |

Mutants were applied to `git archive 1144291` copies in scratch only. Afterwards the mutated sources were verified byte-identical to a fresh archive (`cmp`).

### Closed and open after this addendum

- **Closed:** F-6 (with the stated limit N-4), F-7, F-8, F-9, F-10, F-11, F-1 as labelling, N-0.
- **Open:** F-1 owner decision on the FOCUS scope (still the blocker for any image STARLIGHT claim). N-1 (label not in arm records) MEDIUM. N-2 LOW. N-3 LOW (doc wording). N-4 to N-6 INFO. F-12 INFO (equivalent location gate, unchanged).

### Verdict on the repairs

The repairs do what they say, and the comparability label is correct on every case I could construct. The F-1 blocker is unchanged: image STARLIGHT is labelled, not fixed. The label is not carried into arm records, so the arm reports cannot yet apply it. Image `starlight_s1` results remain **BLOCKED** from any sparsity comparison, and `h2o_shared_base` comparisons on image prompts are confounded.
