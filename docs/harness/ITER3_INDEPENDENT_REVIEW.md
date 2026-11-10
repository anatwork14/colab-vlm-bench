# Iteration 3: independent review of the round-3 changes (auditor wF:p9)

**Reviewed code: `238c8ab`** (`i3/p9`, final iteration-2 head). The only commit added during this review is `4bdd042` (tests, C-11), followed by the commit of this file. Other workers (p7, p8) may still change code; re-review the integrated head before relying on any row below.

**Method.** Read the round-3 diffs (`90e056d`, `1dde7cd`, `7f09d30`, `0ce644d`, `ea8cfae`, `c8a352e`, `004b5b0`, `f8fd7f0`, plus the round-2 C-3/C-5 commits and the engine and qualification label code). For each mechanism, a scratch copy of `238c8ab` was made with `git archive` in a scratchpad outside the repo. Each mutation is one exact-string change, applied, tested, and restored (the restore was checked with `cmp`). A mutant counts as **killed** only if a named test fails and the same test passes unmutated at `238c8ab`. A mutant that no test fails is **SURVIVED** (the test set is vacuous for that mechanism). Heavy (checkpoint or >60 s) runs were under `flock wt/heavy.lock`, except for the deviations listed in section 7.

No PASS below is a status claim. Each PASS names the command and count that produced it.

---

## 1. Status of each round-3 item

| Item | Round-3 change | Status after this review | Evidence |
|---|---|---|---|
| **C-3** caller-asserted verified / source hash | `3f44186` labels; `evidence_provenance` reports `CALLER_DECLARED_NOT_VERIFIED` and `CHECKED_IMAGE_BYTES_MATCH` only against the request's image bytes | **CLOSED as labelling.** The protection still honours the caller's `verified`; `add_evidence` refuses `verified=False`, so the flag only matters for minimal pools. The `CHECKED` label means the declared hash equals the request's image hash. Evidence content is not checked. | 3 mutants killed (section 2) |
| **C-5** image evidence protects no visual tokens | `6084e25` label `NOT_IMPLEMENTED`; `b39e081` refuses unknown modalities | **CLOSED** for the label and the modality bypass. Protection is still text-only, and the label says so. | 3 mutants killed |
| **C-7** GPU receipt not in git | `c43d526` disclosure | **CLOSED as disclosure.** `GPU_RECEIPT_INDEX.json` has 0 entries. `artifacts/runs` is gitignored (`.gitignore:18`) and this worktree has no receipt. No test asserts this; it is documentation only, so there is nothing to mutate. | read: index count 0, `git check-ignore` hit |
| **C-11** qualification and engine labels derived from execution | Release gates from recorded test outcomes (`a26578c`, `d4e6334`); `SEVEN_ARM_HOOK_COVERAGE_PASS` label | **PARTIAL.** (a) Release gates are a function of recorded per-test status (`compute_release_gates`, `_test_status`, `_engine_gate`). Test D requires each arm's engine labels and fails on any mismatch. The release gate is therefore tied to real per-arm execution. (b) The engine label `SEVEN_ARM_HOOK_COVERAGE_PASS` is set from **load-time** tensor and parameter coverage plus logits parity (`real_engine.py` ~2296–2309 and ~645–650). It is not tied to hook invocation during the request, although per-request hook telemetry exists (`layer_telemetries`) and is not consulted (R3-2). (c) Survivors found in the engine scaffold label, the backend label and the status mapping; now killed by new tests (section 2.3). | 7 of 7 mutants now killed: 4 killed at `238c8ab`; 3 survived there and are killed by `tests/test_r3_review_c11_labels.py` |
| **N-R2-1** owner-key anchor | `90e056d` pin mechanism `OWNER_KEY_FINGERPRINT`, reported as operator-supplied | **OPEN.** The mechanism works (3 mutants killed), but the pin is `None` in the repo, so the anchor is the operator's signer file. The report says so. A pin that repo writers can edit is not an external anchor. | 3 mutants killed; `PILOT_INVENTORY.json`: `authorization_anchor: operator-supplied` |
| **G-PATH** PATH shims | `90e056d` `trusted_binary` (`/usr/bin`, `/usr/local/bin` only) for `nvidia-smi` and `ssh-keygen` | **CLOSED** for PATH shims: both bare-name mutants killed. Residual: a writable trusted directory is out of scope. | 2 mutants killed (3 tests) |
| **F-1** shared retained set and vetoes | `f2f60ba` shared candidates; my round-2 tests `4f52fc1` | **CLOSED** for the mechanisms tested: all seven round-2 F-1 mutants and the three vetoes are killed by named tests. The checkpoint-level veto drop (M4) timed out at this head (section 6), so that particular confirmation is not available here. | 8 lightweight F-1 mutants killed (section 2.4); checkpoint M4 on this head: TIMEOUT |
| **Keyword union cap** (N-R2-2) | `0ce644d`, `MAX_UNION_PROMPT_FRACTION = 0.5`, declared order, `rejected_union`, `union_fraction` | **CLOSED** for the mechanism: 5 mutants killed. Residual: the 0.5 value is an owner-ratifiable constant; the outcome is order-dependent by design; the cap is a fraction of the prompt, not of the budget. | section 2.5 |
| **Per-keyword cap** (0.2) and stopword list | `3251469` (round 2), unchanged here | **CLOSED** for the mechanism: 3 mutants killed (cap boundary, cap removed, stopword check removed). The 24-word list and 0.2 cap are provisional, as the brief states. | section 2.5 |
| **Stage 2 rejection and rollback** | Existing tests `test_iter3_b2_mutation.py` (M1–M4) and `test_iter3_b2_mtp_rollback.py` | **CLOSED** for the five source mutants run. Three are caught at the baseline assertion, so their mutant leg is not exercised (R3-3). Two are valid detections. The checkpoint M4 DA-drop mutant is not confirmed on this head (TIMEOUT). | section 2.6 |

---

## 2. Mutation results

Counts are the number of failing tests reported by pytest on the mutant. "Killed by" lists the failing test or tests shown in the run.

### 2.1 C-3 and C-5 (lightweight)

| Mutant | Change at `238c8ab` | Result | Killed by |
|---|---|---|---|
| C3_source_label_always_checked | unknown-source branch reports `CHECKED_IMAGE_BYTES_MATCH` | killed, 2 failed | `test_r2_c3_provenance.py::test_text_only_request_labels_both_the_flag_and_the_source_hash_as_caller_declared`, `::test_engine_telemetry_carries_the_provenance_labels_for_a_text_only_request` |
| C3_verified_flag_label_says_verified | `VERIFIED_FLAG_LABEL` = `"VERIFIED"` | killed, 5 failed | `test_the_verified_flag_is_never_reported_as_checked` and four others in `test_r2_c3_provenance.py` |
| C3_unverified_items_not_rejected | `evidence_validity` ignores `verified` | killed, 1 failed | `test_verified_false_is_reported_as_a_caller_declaration_not_a_check` |
| C5_status_always_no_image | status returns `NO_IMAGE_EVIDENCE_DECLARED` | killed, 3 failed | `test_status_is_not_implemented_when_image_region_evidence_is_declared`, `…_even_without_a_bounding_box`, `test_engine_telemetry_reports_visual_evidence_not_implemented…` |
| C5_image_modality_accepted | `image` added to `KNOWN_MODALITIES` | killed, 1 failed | `test_an_unknown_modality_is_refused_so_image_evidence_cannot_hide_behind_a_label` |
| C5_modality_check_removed | `add_evidence` modality check disabled | killed, 1 failed | same test |

### 2.2 N-R2-1 and G-PATH (lightweight)

| Mutant | Change | Result | Killed by |
|---|---|---|---|
| NR21_pin_filter_keeps_all_lines | pinned filter keeps every signer line | killed, 3 failed | `test_N_R2_1_pinned_refuses_an_attacker_key_listed_under_the_owner_principal`, `…attacker_signature_even_when_the_owner_key_is_also_listed`, `…owner_key_missing_from_the_signer_file_is_refused` |
| NR21_pinned_verify_uses_operator_file | pinned branch verifies against the operator's file, not the kept lines | killed, 1 failed | `test_N_R2_1_pinned_refuses_an_attacker_signature_even_when_the_owner_key_is_also_listed` |
| NR21_missing_pin_not_refused | "pinned key not in signer file" refusal removed | killed, 2 failed | `…attacker_key_listed_under_the_owner_principal`, `…owner_key_missing_from_the_signer_file_is_refused` |
| GP_gpu_probe_uses_path | `probe_gpu` runs bare `nvidia-smi` | killed, 1 failed | `test_G_PATH_probe_gpu_ignores_a_path_shim_nvidia_smi` |
| GP_ssh_keygen_uses_path | owner check runs bare `ssh-keygen` | killed, 2 failed | `test_G_PATH_a_path_shim_ssh_keygen_cannot_authorise_a_forged_record`, `…positive_control…` |

### 2.3 C-11 (engine and release-gate labels)

| Mutant | Change | Before this review | After `4bdd042` |
|---|---|---|---|
| C11_engine_scaffold_branch_reports_pass | scaffold `seven_arm_gate` = `PASS` | **SURVIVED** (92 passed across `test_real_engine.py`, `test_gate_label_consistency.py`, `test_qualification_gates.py`) | killed: `test_scaffold_engine_reports_the_exact_gated_labels_without_a_checkpoint` |
| C11_engine_partial_branch_reports_pass | partial-weights `seven_arm_gate` = `PASS` | killed, 1 failed (`test_real_engine_pretrained_checkpoint_loading`, 1.5 s) | (already killed) |
| C11_engine_backend_hook_always_pass | `ACTUAL_BACKEND_HOOK_PASS` = `PASS` | **SURVIVED** (`test_real_engine_pretrained_checkpoint_loading`: 1 passed). Its assertion at `test_real_engine.py:255` accepts `("PASS", "GATED_…")`. | killed: scaffold test asserts `GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED` exactly |
| C11_release_seven_arm_constant_pass | `SEVEN_ARM_HOOK_COVERAGE_PASS` = `"PASS"` in `compute_release_gates` | killed, 1 failed (`test_failed_seven_arm_test_fails_engine_gates`) | (already killed) |
| C11_release_status_unknown_counts_as_pass | `_test_status` maps an unknown status to `PASS` | **SURVIVED** (45 passed) | killed: 4 failed for `weird`, `""`, `pass`, `SKIPPED` in `test_a_status_that_is_not_pass_or_fail_is_never_reported_as_pass` |
| C11_engine_gate_ignores_not_applicable | `_engine_gate` no longer treats NOT_APPLICABLE as NOT_TESTED | killed, 1 failed (`test_an_mtp_arm_reporting_not_applicable_leaves_the_mtp_gate_not_tested`) | (already killed) |
| C11_engine_gate_disagreement_not_fail | disagreeing applicable arms return a value, not FAIL | killed, 3 failed | (already killed) |

### 2.4 F-1 shared selection and vetoes (lightweight)

Scratch at `238c8ab`. Targets: `test_r2_shared_candidates.py`, `test_starlight_oracle.py`, `test_starlight_engine.py`, `test_starlight_signals.py`.

| Mutant | Change | Result | Killed by (first failures) |
|---|---|---|---|
| F1_per_layer_candidates | candidates per layer | killed, 23 failed | `test_I7_surviving_entries_keep_their_original_position_ids_across_steps`, `test_long_context_pressure…`, `test_multi_request_isolation…`, `test_s4_cache_length_check…` |
| F1_drop_da_veto_all_layers | DA signal dropped for planning | killed, 8 failed | `test_F1_veto_applies_to_a_shared_candidate_on_every_full_layer` (`assert (3, 0)`), `test_HC3b…`, `test_A_TOGGLE_da_signal…`, `test_da_h2o_conflict_is_vetoed_not_evicted` |
| F1_drop_structural_veto | structural signal dropped | killed, 1 failed | `test_HC13_vision_start_and_end_are_protected_and_never_evicted` |
| F1_drop_grounding_veto | grounding signal dropped | killed, 6 failed | `test_A_TOGGLE_grounding…`, `test_grounding_h2o_conflict_is_vetoed_not_evicted` |
| F1_offbyone_heavy_budget | heavy-hitter count minus one | killed, 7 failed | `test_F1_shared_mask_with_one_layer_equals_…`, `test_keep_mask_selects_recent_window_plus_heavy_hitters_exactly` |
| F1_aggregate_first_layer_only | aggregate taken from layer 3 | killed, 1 failed | `test_F1_shared_candidates_use_the_mean_over_full_layers_not_one_layer` (added in round 2) |
| F1_length_guard_removed | unequal-length guard disabled | killed, 1 failed | `test_F1_unequal_full_layer_lengths_raise_instead_of_desynchronising` |
| F1_union_guard_removed | protected-union guard disabled | killed, 1 failed | `test_F1_protected_slots_that_differ_across_layers_raise_instead_of_planning` (added in round 2) |

**Judgement on the cross-layer veto.** Under the shared set, candidates and protection are layer-independent, so the round-1 scenario (a veto on layer 3 alone) cannot occur. What has to be covered is the veto on a shared candidate on every layer. The round-2 scenario protects a non-candidate and so does not cover it. The veto-on-candidate test added in round 2 does (F1_drop_da_veto_all_layers fails it).

### 2.5 Keyword union cap and per-keyword cap (lightweight)

Targets: `test_r2_c4_keyword_breadth.py`, `test_starlight_engine.py`, `test_starlight_signals.py`, `test_starlight_audit_fixes.py`.

| Mutant | Change | Result | Killed by |
|---|---|---|---|
| KU_union_check_removed | union cap never applied | killed, 4 failed | `test_the_union_of_accepted_keywords_is_bounded_by_the_policy`, `test_the_six_keyword_probe_reports_union_fraction_and_the_rejections`, `test_union_is_evaluated_in_declared_order_not_alphabetically`, `test_a_keyword_that_would_exceed_the_union_cap_does_not_block_later_smaller_keywords` |
| KU_union_boundary_ge | `>=` instead of `>` at the union cap | killed, 1 failed | `test_a_union_exactly_at_the_cap_is_accepted` |
| KU_union_rejection_blocks_later | `break` after a union rejection | killed, 3 failed | three of the above |
| KU_union_fraction_not_reported | `union_fraction` not written | killed, 2 failed | `test_the_six_keyword_probe…`, `test_engine_telemetry_reports_union_fraction…` |
| KU_keyword_order_sorted | sorted instead of declared order | killed, 4 failed | `test_union_is_evaluated_in_declared_order_not_alphabetically` and three others |
| KU_per_keyword_boundary_ge | `>=` at the per-keyword cap | killed, 3 failed | `test_a_keyword_exactly_at_the_cap_is_accepted`, `test_irrelevant_evidence_protection_is_measured_per_evidence_id` |
| KU_stopword_check_removed | stopword rejection disabled | killed, 5 failed | `test_the_is_rejected_as_a_stopword_even_when_it_is_rare`, `test_the_stopword_check_runs_before_any_position_is_protected`, `test_engine_telemetry_reports_the_keyword_policy_and_the_rejected_stopword` |
| KU_per_keyword_cap_removed | per-keyword cap never applied | killed, 1 failed | `test_a_keyword_over_the_prompt_fraction_cap_is_rejected_and_reported_with_its_fraction` |

### 2.6 Stage 2 rejection and rollback (checkpoint, under `flock`)

Scratch at `238c8ab`. Baselines were run unmutated at `238c8ab` in the worktree.

| Mutant | Change | Result | Baseline at HEAD | Detection |
|---|---|---|---|---|
| S2_restore_is_noop | `_restore_dynamic_cache` does nothing | killed: `test_M1…` fails (183 s) | `test_iter3_b2_mutation.py` 4 passed (8 min 20 s) | assertion text not captured. From the test code, the first assertion is the baseline `restored_cache`, so the failure is expected there, and the mutant leg is not reached (R3-3) |
| S2_gdn_state_not_restored | GDN branch skipped in restore | killed: `test_M2…` fails (163 s) | as above | assertion text not captured; from the test code, the baseline `restored_cache` assertion is first, as for M1 (R3-3) |
| S2_policy_restore_skips_tracker | `_pol_restore` skips tracker layers | killed: `test_M3…` fails "baseline must pass first" (127 s) | as above | baseline check (R3-3) |
| S2_reject_path_skips_restore | reject branch no longer calls `_pol_restore` | killed: `test_inferc_starlight_partial_acceptance_then_rejection_after_accept_matches_greedy` fails (338 s) | 1 passed (70.9 s) | **valid** detection |
| S2_policy_restore_skips_orig | `_pol_restore` does not restore `st.orig` | killed: `test_pinv_block_rollback_restores_kv_gdn_and_starlight_state_bit_for_bit` fails "rollback did not restore the STARLIGHT policy digests exactly" (6 s) | 1 passed (38.3 s) | **valid** detection |
| F1_drop_da_veto_all_layers (checkpoint M4) | DA ignored in planning | **TIMEOUT**: `timeout 520` expired (exit 124, 8 min 40 s). No result. | not run at this head | Not confirmed here. On `8a6390d` the same mutant failed M4 in 88 s. |

---

## 3. Tests that survived (vacuous at 238c8ab)

| Test | Why it is vacuous | Status |
|---|---|---|
| `test_real_engine.py::test_real_engine_pretrained_checkpoint_loading`, lines 255–257 | `ACTUAL_BACKEND_HOOK_PASS` is accepted as `"PASS"` or `GATED_…`; `REAL_MODEL_FORWARD_PASS` and `REAL_MULTIMODAL_INFERENCE_PASS` likewise accept `"PASS"` in the scaffold. A constant `PASS` passes. | Not weakened here. Exact-label coverage added in `tests/test_r3_review_c11_labels.py`. The `in (…)` lines are still there and should be tightened by their owner. |
| Scaffold `SEVEN_ARM_HOOK_COVERAGE_PASS` (no test at all) | 92 tests passed with the scaffold label forced to `PASS`. | Killed by the new test. |
| `_test_status` for unknown status strings (no test at all) | 45 tests passed with unknown statuses mapped to `PASS`. | Killed by the new test. |
| `test_iter3_b2_mutation.py::test_M1…`, `test_M2…`, `test_M3…` (source-level) | They monkeypatch the restore function, so a corrupted restore in source breaks the baseline assertion first and the mutant leg never runs. Breakage is detected, but the test cannot separate "restore broken" from "detector broken". | Recorded as R3-3. Not changed. |
| `test_harness_r3_regressions.py::test_N_R2_1_gap_documented_…` | Characterisation of the open gap. It passes by design and does not measure the gap closed. | By design. |
| `test_harness_r3_regressions.py::test_G_PATH_the_trusted_directories_are_absolute` | Only checks the constant. | Not a mutation check; the G-PATH mutants are killed by the other tests. |

---

## 4. New findings from this review

| ID | Severity | Owner | Finding | Evidence |
|---|---|---|---|---|
| **R3-1** | MEDIUM | wF:p7 (test owner) | Existing engine-label assertions are weak. `test_real_engine.py:255–257` accept either `PASS` or the `GATED_…` value. A constant `PASS` in the backend label, or in the scaffold forward and multimodal gates, passes them. | mutant C11_engine_backend_hook_always_pass survived; `test_r3_review_c11_labels.py` now pins the scaffold labels exactly. OPEN for the existing lines. |
| **R3-2** | LOW–MEDIUM | wF:p7 / lead | `SEVEN_ARM_HOOK_COVERAGE_PASS` (engine) is computed from load-time coverage and parity, not from hook invocation during the request. Per-request hook telemetry exists (`layer_telemetries`, `real_engine.py:~272`) but the gate does not read it. The release gate is tied to per-arm execution through test D, which mitigates this. | `real_engine.py` ~2296–2309 and ~645–650 (read). OPEN. |
| **R3-3** | LOW | wF:p9 (test design) | Source-level corruptions of the restore path fail the baseline assertion of M1–M3, so those detectors do not exercise their own mutant legs. Breakage is still caught, but the discrimination the tests claim is not demonstrated for source corruption. | section 2.6. OPEN. |
| **R3-4** | MEDIUM | wF:p8 / owner | N-R2-1 is not closed. The pin mechanism is correct and tested, but `OWNER_KEY_FINGERPRINT` is `None`, so authorization still depends on which signer file the operator passes. The report says so. A repo constant is editable by repo writers, so it cannot be the external anchor either. | `harness_inventory.py:OWNER_KEY_FINGERPRINT = None`; `PILOT_INVENTORY.json` `authorization_anchor: operator-supplied`. OPEN; owner decision. |
| **R3-5** | LOW | wF:p8 | The checkpoint M4 DA-drop mutant timed out on this head (520 s). Its status is UNKNOWN at `238c8ab`. | section 2.6 TIMEOUT. OPEN (rerun alone, with a longer bound). |
| **R3-6** | LOW | lead | The keyword union is order-dependent by design (declared order). The caller decides which keyword fills the union first. The 0.5 cap is provisional. | `starlight.py` `MAX_UNION_PROMPT_FRACTION`, `0ce644d` message. Owner ratification. |

No new finding is a bypass of READY. Authorization can be forged by whoever chooses the signer file (R3-4, the same as round 2), but READY still requires the live probe on the signed hardware.

---

## 5. Commands and counts

All with `PYTHONPATH=src CUDA_VISIBLE_DEVICES= /mnt/ssd/genos2/research_workspace/colab-vlm-bench/.venv/bin/python -m pytest -q -p no:cacheprovider`.

| Command (targets) | Result |
|---|---|
| Lightweight set at `238c8ab` (17 files: `test_harness_r3_regressions`, `_forgery`, `_cli`, `_live_probe`, `_inventory`, `test_inventory_freshness`, `_pins`, `_r2_regressions`, `test_r2_c3_provenance`, `_c4_keyword_breadth`, `_c5_visual_evidence`, `_c6_doc_stale_strings`, `test_r2_shared_candidates`, `test_gate_label_consistency`, `test_qualification_gates`, `test_gpu_smoke_helpers`, `test_gpu_smoke_outer_deadline`) | **215 passed** (27.7 s) |
| `tests/test_r3_review_c11_labels.py` at `4bdd042` | **6 passed** (15.5 s) |
| Same file, scratch, with each C-11 survivor mutant | each mutant: **≥1 failed** (section 2.3) |
| `tests/test_iter3_b2_mutation.py` at `238c8ab`, under `flock` | **4 passed** (8 min 20 s) |
| `tests/test_iter3_b2_mtp_rollback.py::test_inferc_starlight_partial_acceptance_then_rejection_after_accept_matches_greedy`, `flock` | **1 passed** (70.9 s) |
| `tests/test_iter3_b2_mtp_rollback.py::test_pinv_block_rollback_restores_kv_gdn_and_starlight_state_bit_for_bit`, `flock` | **1 passed** (38.3 s) |
| `tests/test_real_engine.py::test_real_engine_pretrained_checkpoint_loading` | **1 passed** (1.65 s) |
| Mutation runs (section 2), each through `mut.py`, scratch copy | counts in section 2 |
| Checkpoint M4 with DA-drop mutant, `flock`, `timeout 520` | **TIMEOUT** (exit 124) |

Mutation driver: `scratchpad/i3p9/mut.py` (scratch, not committed). Mutant strings are in its `MUT` table.

## 6. Items not closed and why

- **R3-4 / N-R2-1:** needs an owner decision on the external anchor. The repo cannot provide it.
- **R3-5 (M4 on this head):** timed out; rerun alone with a longer bound if the lead wants the checkpoint confirmation.
- **C-11 per-request derivation (R3-2):** a code change, not a test change. Not made here.
- **R3-1:** the existing weak assertions belong to their owner. I did not change them.
- **Checkpoint Stage 2 M1–M3 at source level (R3-3):** the detectors need a redesign to exercise their mutant legs. Not changed here.
- **C-7:** no receipt in this worktree; the one receipt lives in another worktree and is gitignored.
- **Full suite:** not run (lead only).

## 7. Host-load deviations (disclosed)

I ran some tests longer than 60 s without `flock`: `test_real_engine.py` (203 s, twice, once mutated) and `test_qualification_gates.py` (about 60 s). Those runs were on my scratch copy, one at a time. Every checkpoint run was under `flock`. I should have used the lock for the two long CPU files. None of those runs is cited as evidence for a PASS except the 92-test and 45-test "survived" results, which are used only to say the mutant was not caught.

Co-Authored-By: Claude Haiku 5.5 <noreply@anthropic.com>
