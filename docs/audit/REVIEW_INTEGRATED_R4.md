# Phase 3b verification of the integrated tree, round 4

- Reviewer: wF:p9 (Scientific Auditor), for lead wF:p5
- Reviewed tree: `/mnt/ssd/genos2/research_workspace/wt/wf-integrated` at **HEAD `50762b0`** (read-only; not edited)
- Lead's full suite on 50762b0: 416 passed, 10 skipped, 0 failed (not re-run here)
- My targeted run on 50762b0 (CPU, `CUDA_VISIBLE_DEVICES=`, `PYTHONPATH=<tree>/src`): `test_mtp_speculative_official.py`, `test_real_engine.py`, `test_infrastructure.py`, `test_mtp_da_upstream_contracts.py`, `test_telemetry_visibility.py`, `test_e2e_comparability.py`, `test_measurement_regressions.py`, `test_resume_policy.py`, `test_amber.py` → **162 passed, 0 skipped**, exit 0.
- Simulations, re-run against the R4 `h2o.py`: `scratchpad/h2o_r4_sim.py` (new), `scratchpad/h2o_compound_sim.py`, `scratchpad/h2o_compound_sim_visual.py`. The old `h2o_fixed_sim.py` now crashes with `KeyError: 'protected_saturated'` because R4 removed that metric key. The new script replaces it with the same checks.

## Summary

| Area | Verdict | Severity of what remains |
|---|---|---|
| 1a H2O: text survives on image prompts | PASS | — |
| 1b H2O: total retained ≤ protected + budget | PASS | — |
| 1c H2O: cumulative telemetry | PASS (identity holds); PARTIAL on reporting (no text-side retention field) | MEDIUM |
| 1d MTP: CPU label in helper-derived metrics | PASS | — |
| 1e MTP: rename to target agreement | PASS (code); PARTIAL (docs still use old name) | LOW |
| 1f DA: gate rests on measured masked weights | PASS | — |
| 1g DA: None-not-zero telemetry | PASS | — |
| 1h `forward_path_executed` telemetry | PASS | — |
| 1i p8 F-03 GPU sum | PARTIAL | MEDIUM-LOW |
| 1j p8 F-06 exit codes | PASS | — |
| 1k p8 F-02 comparability | PASS, with an over-exclusion | LOW |
| 1l p8 amber `None` | PASS | — |
| 2 Protected-outside-budget semantics | Documented deviation, defensible as an ablation; needs its own docs entry and a no-eviction oracle test | MEDIUM |
| 3 Matrix refresh | Done: `docs/audit/SOURCE_ALIGNMENT_MATRIX.md` §0, §2, §3, §5 and rows updated; new rows A2-14, A4-11 | — |
| 4 Per-arm blockers | See §4 | — |

---

## 1. Verification of the fixes for my findings

### 1a. H2O: text tokens survive on image prompts — PASS

Code: `h2o.py:110-117` (`_protected_mask`: sinks and visual tokens); `h2o.py:122-135` (`initialize_budget`: budget = `max(min_budget, int((L − protected_at_prefill) × ratio))`); `h2o.py:187-216` (plan: protected kept; unprotected fill to exactly B; the newest `recent` unprotected tokens and the highest-scoring rest). Engine: `real_engine.py` calls `set_visual_tokens` before `initialize_budget`, so the protected set is correct at prefill.

Re-run (`h2o_r4_sim.py`, one full-attention layer, 30 decode steps):

| Prompt | Protected at prefill | Budget B | Cache after step 1 / 30 | Original text alive at step 1 / 30 |
|---|---|---|---|---|
| text only, L=340 | 4 | 168 | 172 / 172 | 167 / 138 of 336 |
| image, L=340, 300 visual | 304 | 32 | 336 / 336 | 31 / 26 of 36 |
| image, L=340, 20 visual | 24 | 158 | 182 / 182 | 157 / 128 of 316 |

The old `h2o_compound_sim_visual.py` (L=300, 200 visual tokens) now stabilises at 252 = 204 protected + 48. Under the R3 code, the same prompt shape held 305 slots and dropped all text after step 2. Text survives on image prompts: PASS.

### 1b. H2O: total retained ≤ protected + budget — PASS

The plan returns `protected + B` at most (`h2o.py:201-216`). In every simulated case, `cache − (protected + budget)` was ≤ 0 at every step (max over steps = 0). The text-only compound sim also holds steady: 301 → 152 at the first step, then 153 → 152 each step to step 200. Compounding is gone.

### 1c. H2O: cumulative telemetry — PASS for correctness; PARTIAL for reporting

Engine fields (`real_engine.py:1767-1773`): `per_step_evicted_slot_rate` (last decode step only), `cumulative_evicted_slots` (all steps, full-attention layers), `cumulative_total_slots_at_prefill` (layers × prompt length), `final_retained_fraction`. Accumulation is at `real_engine.py:1608-1611`, and the final fraction at `:1713-1715`. Names say what they measure.

Identity check (per layer, from the sim): total slots ever present − final retained = cumulative evicted. Results: 370 − 172 = 198 (= 198 evicted, text-only); 370 − 336 = 34 (= 34, image-300); 370 − 182 = 188 (= 188, image-20). The counts are internally consistent.

What is still missing: a text-side retention field. On VLM prompts `final_retained_fraction` is near 1 by construction (protected visual slots dominate), which is correct but easy to misread as "H2O kept the KV". Add `unprotected_retained / unprotected_prefill` and report it next to the total. Severity MEDIUM.

### 1d. MTP: CPU label carried into helper-derived metrics — PASS

- The helper's return dict includes `"label": self.CPU_GREEDY_K1_LABEL` (`real_engine.py` `_speculative_mtp_greedy_k1`, return block, label value `cpu_greedy_k1_reference_not_vllm_mtp`).
- The serving telemetry carries the same label under `mtp_candidate_diagnostic.cpu_greedy_k1_reference` with `run_in_this_request: False` and `entry_point` (`real_engine.py:1802-1806`). The note states that the serving arm does not call the helper.
- Tests: `test_mtp_speculative_official.py` (6 tests, all ran).

### 1e. MTP: rename to target agreement — PASS for code; PARTIAL for docs

- The serving field is `proposal_target_agreement_diagnostic` (`real_engine.py:1801`), with `proposal_target_agreement_publishable: False` and `proposal_target_agreement_is_acceptance: False` (`:1797-1798`).
- `proposal_match_rate` is absent from `src/`. `test_real_engine.py:808` asserts it is not in the diagnostic dict.
- `mtp_acceptance_rate` is still a field, set to `None` (`real_engine.py:1696`, passed through at `:1743`; `None` unless the helper ran).
- Docs still use the old name: `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:13` and `:45`, and `docs/NUMERIC_INTEGRITY_AUDIT_20261109.md:27`. These are stale. LOW.

### 1f. DA gate: measured masked weights — PASS

- Measurement is in the hooked attention: `real_engine.py:219-238` records, for each layer's decode step, the count of masked (head, query, key) entries and the maximum post-softmax weight on them (`da_measured_masked_entry_count`, `da_measured_masked_weight_max`).
- Aggregation: `real_engine.py:1641-1650`.
- Gate: `_da_masking_gate` (`real_engine.py:933-952`). PASS requires all three: restriction ratio > 0, measured masked entries > 0, and measured max masked weight exactly 0.0. A constructed mask alone cannot pass. A measured weight above zero returns `FAIL_MASKED_KEY_RECEIVED_ATTENTION`.
- Telemetry (`real_engine.py:1826-1834`): `measured_source: hooked_forward_post_softmax` or `NOT_MEASURED_NO_HOOKED_DECODE_STEP`.
- Caveat (unchanged, documented as DA-6 in `docs/DA_DEVIATIONS.md`): the mask is still scripted. PASS means masking was applied on this scripted mask, not that the model's focus was used.
- Tests: `test_r4_da_gate_requires_measured_zero_masked_weights`, `test_r4_da_scaffold_path_never_passes_the_gate`.

### 1g. DA: None-not-zero telemetry — PASS

- `final_da_restriction_ratio`, `final_da_causal_area_ratio`, `da_measured_masked_entries`, `da_measured_weight_max` start as `None` (`real_engine.py:1484-1488`).
- Result dataclass fields are typed `float | None` with `None` defaults (`real_engine.py:1000-1004`).
- `max_new_tokens=1` reports `None` (test `test_r4_max_new_tokens_one_reports_none_not_literal_zero`, `test_real_engine.py:857`).
- Remaining LOW: `_round_or_none` wraps telemetry values, so a literal zero cannot appear there from a missing measurement.

### 1h. `forward_path_executed` telemetry — PASS

- `forward_path_label` (`real_engine.py:586-590`): `scaffold_reference`, `official_hf_hooked`, or `official_hf_partial_coverage_UNQUALIFIED`. It is emitted from both forward paths (`:857`, `:916`) and recorded at `:1523` and `:1754`.
- `hf_forward_active` (`real_engine.py:577-584`) drives cache layout and decode position handling, so the path that runs is the path that is labelled.
- Test `test_r4_forward_path_telemetry_names_the_path_that_ran` (`test_real_engine.py:889-917`) degrades coverage to 0.5 and asserts the label `official_hf_partial_coverage_UNQUALIFIED` and gate `PARTIAL_COVERAGE_NOT_QUALIFIED`. This closes the inconsistency I noted in round 3 (the HF forward ran while the gates said PARTIAL).

### 1i. p8 F-03: GPU memory — PARTIAL

Fixed (PASS): the query returns `index,uuid,memory.used,utilization.gpu` (`telemetry.py` `GPU_QUERY`), and `CUDA_VISIBLE_DEVICES` selects rows by index or UUID (`select_visible_rows`). Unset gives `sum_over_all_host_devices` with the device count. A visibility list matching no device gives `VisibilityError` and no sample (`visibility_mismatch`). Tests: `test_telemetry_visibility.py` (passed).

Remaining issue (MEDIUM-LOW): index matching assumes CUDA's device numbering equals `nvidia-smi`'s. CUDA's default ordering is `FASTEST_FIRST`, while `nvidia-smi` uses PCI bus order. The two agree only with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and that variable is not set anywhere in the repo (`grep` over `src`, `scripts`, `runbooks`). The UUID path does not have this problem. Fix: set or require `CUDA_DEVICE_ORDER=PCI_BUS_ID` for the sampler, or match by UUID only, and record which ordering was assumed.

### 1j. p8 F-06: exit codes — PASS

- `_serving_outcome` (`cli.py:95-125`) counts uncapped MISSING cells as non-completed. `cap_reached` is wired from the run limit (`cli.py:438`: `cap_reached=bool(limit) and done >= limit`).
- Nothing attempted with uncapped missing cells gives `INCOMPLETE_NOT_ATTEMPTED`, rc 6. A genuine cap gives `IN_PROGRESS`, rc 0.
- Test: `test_resume_policy.py` (passed).
- The status names are distinct, though rc 6 is shared with `PARTIAL_FAILURE`.

### 1k. p8 F-02: E2E comparability — PASS, with an over-exclusion

- Legacy receipts' E2E is withheld as `NOT_COMPARABLE_LEGACY_DEFINITION` (`report.py`, `analysis.e2e_definition`). `e2e_comparability` is `MIXED_NOT_COMPARABLE` when PASS rows mix definitions. Tests: `test_e2e_comparability.py`.
- LOW: `paired_speedups` excludes a Base/MTP pair when the two receipts differ in E2E definition (`analysis.py`). The speedup itself is computed from `output_tokens_per_s` (`bench.py:280` = output tokens / total time), which does not use E2E. The exclusion therefore drops valid throughput pairs. Either compare throughput without the E2E rule, or state the exclusion in the report.

### 1l. p8 amber `None` — PASS

`datasets/amber.py:587`: `coverage_rate` is `None` in the empty-annotation branch (status `NOT_MEASURED`). `tests/test_amber.py` was updated. Also PASS: quality counts are disjoint (`quality.py`: `invalid_predictions_counted_zero` and `missing_predictions`; `failures_counted_zero` is their sum).

---

## 2. The protected-outside-budget semantics, challenged scientifically

**Is it a documented deviation from upstream H2O?** Yes in the code, the telemetry and the tests. It is not documented in `docs/`. I did not re-read the upstream source in this round. My statement that upstream has no visual-token protection and fixes its budget from the sequence length rests on the reading recorded in the matrix (A4-01 to A4-04). The engine's own text says "documented deviation" (`h2o.py:127-131` in the R3 version; R4 docstrings at `h2o.py:122-146`). The telemetry is explicit (`protected_outside_budget: True`, `budget_basis: unprotected_prefill_tokens_fixed_per_layer`, `real_engine.py:1768-1769`). Recommendation: write `docs/H2O_DEVIATIONS.md` in the same style as `docs/DA_DEVIATIONS.md`, and name the arm in results as a variant (for example `h2o_visual_anchor`), so it is not read as upstream H2O.

**Is the budget definition defensible?** Partly.

- Defensible: protecting visual evidence and then letting the text budget compete with it is the failure mode the R3 code showed. Keeping the visual block whole and budgeting text is a clear, testable ablation. Protection of sinks is standard in the StreamingLLM family.
- Problem 1, the ratio is over a different population than in upstream. `budget_ratio=0.5` means "half of the non-visual prompt tokens" on VLM prompts but "half of all tokens" on text prompts. The same config therefore yields about 99% total retention on a 300-visual-token image (336 of 340 slots) and 50% on text-only prompts. The two are not one method.
- Problem 2, the 32-token floor dominates short text. On the 36-text-token image prompt, B = max(32, 18) = 32, so 89% of text is kept before any decode eviction. The effective text policy is set by `min_budget` (an unsourced constant, matrix A4-04), not by the ratio.
- Problem 3, the sinks are free. Four sink positions sit outside the budget per layer. Small, but it is one more difference from a count-everything budget.
- Problem 4, generated tokens are unprotected, so they compete with prompt text under B. That part matches the upstream spirit and is correct.

**What it implies for comparability with base.**

- On VLM prompts, `h2o_*` retains almost all KV (protected visual block plus most text). It differs from base mainly through text eviction during decode, and for short text prompts that eviction is small. A null accuracy difference from base therefore says little about KV compression. `final_retained_fraction` near 1.0 is the honest result, not a failure of the implementation.
- Any `h2o_*` claim of KV reduction on VLM prompts must cite the text-side fraction, not the total.
- The arms are not equal-KV comparisons. Equal-KV would require the budget to count all tokens, which is the R3 failure mode, or a visual budget, which is a different method.

**Oracle test I recommend.** Set `budget_ratio=1.0` (or any budget ≥ total length), so no eviction ever occurs. Then `h2o_base` must match `base` token-for-token on the same prompt. A second test: with protection off and `budget_ratio=1.0`, the tracker must return the full mask. Neither test exists in the repo, as far as I can find. The first would show that the H2O plumbing is inert when nothing is evicted, which is the baseline everything else depends on.

**Decision the owner still has to make.** Telemetry records `unprotected_original_h2o_variant: NOT_IMPLEMENTED_OWNER_DECISION`. Until the upstream-style variant (protection off, budget over all tokens) is implemented and run, no H2O result answers the question "does H2O preserve accuracy at 50% KV?".

---

## 3. Matrix refresh

`docs/audit/SOURCE_ALIGNMENT_MATRIX.md` is refreshed to 50762b0 in this commit:

- §0 Summary and §2 gate table: current verdicts, with the 24f57d3 status kept in parentheses.
- §3 recommended fixes: reordered to what remains.
- §5 history: a table of each changed item (24f57d3 → 50762b0), with file:line evidence, plus a list of items not re-verified this round.
- Row updates: A1-03, A1-05, A2-07, A2-08, A2-10, A3-02, A3-03, A4-01, A4-02, A4-04, A5-02, A5-04, G-01, G-03, G-04.
- New rows: A4-11 (protected outside budget, PARTIAL) and A2-14 (MTP draft attends only to the current position, FAIL on draft fidelity; verifier unaffected).

Current verdicts by area: A1 PARTIAL, A2 PARTIAL, A3 FAIL (paper fidelity; correctly labelled), A4 PARTIAL, A5 FAIL. Items I did not re-verify are listed in §5 of the matrix.

---

## 4. What still blocks calling each arm "method-correct" (CPU engineering layer)

"Method-correct" here means: the arm executes the named method, as specified by its source, on the official checkpoint, with the label saying what it is. Each arm's blockers are listed in order of severity.

**`base`**
- Method: greedy decoding through the official hooked forward. Parity with unhooked HF is verified at load (`real_engine.py` `_load_official_hf_bundle`, logits parity ≤ 1e-5, same top-1).
- Blocker 1: no EOS stop (A2-11). The decode loop stops only at `max_new_tokens`, so output length does not follow the protocol `natural_eos: True` (`methods.py:92-98`). MEDIUM.
- Blocker 2: the processor and ViT path depend on unpinned `transformers`/`AutoProcessor` versions. Not verified this round. MEDIUM.
- Verdict: closest to correct. Method-correct once EOS is handled and the software versions are pinned.

**`h2o_base`**
- Blocker 1: averaged scores over KV heads (A4-02, `h2o.py:208`). Upstream selects per KV head. FAIL.
- Blocker 2: budget constants unsourced (A4-04). FAIL.
- Blocker 3: protected-outside-budget is a variant (A4-11). It must be labelled as a variant, not as H2O.
- Works: budget fixed at prefill (A4-01 PASS, re-verified here). Compaction position-safety (A4-08) and recurrent exemption (A4-05) carry their 24f57d3 status; not re-verified this round.
- Verdict: not method-correct as H2O. Correct as a labelled text-side recency-and-score variant with a visual anchor, once blockers 1 and 2 are fixed or stated.

**`da_base`**
- Blocker 1: no model-emitted focus protocol (A3-01, DA-1). The paper's method is not implemented. FAIL.
- Blocker 2: LOCAL window of 8 masks context outside sink, focus and question (A3-02, `real_engine.py:983`). The paper's LOCAL is sink plus open tail. FAIL.
- Blocker 3: GLOBAL covers only sink and question (A3-03). FAIL.
- Works: the gate rests on measured zero weights; the scripted scope is labelled in every result (`da_scope_provenance: SCRIPTED_SEGMENT_DIAGNOSTIC_NOT_MODEL_DECLARED`).
- Verdict: not method-correct. It is a labelled scripted diagnostic and must not be reported as DA.

**`ecrd_base`**
- Blocker 1: the mechanism is a keyword logit bias, not ECRD (A5-01, A5-03). No evidence scoring, knee/top-k selection, or GRIT visual decider. FAIL.
- Blocker 2: alpha is condition-dependent and unsourced (A5-05). FAIL.
- Blocker 3: the evidence source is absent from the benchmark path (A5-06). With an empty pool, `ecrd_base` returns base logits, so it equals `base`. BLOCKED.
- Blocker 4: the unsupported penalty is inert (A5-04, documented as PARTIAL). Citation support is unverified (A5-03).
- Works: whole-word matching (A5-02 PASS); greedy over reweighted logits (A5-07 PASS).
- Verdict: not method-correct. Do not report `ecrd_base` as ECRD.

**`mtp`**
- Serving arm: does not speculate. It commits only target greedy tokens, so its output equals greedy `base`. No verifier runs in `execute_request`. As an "MTP arm" it is not method-correct; it is a diagnostic arm.
- Helper `_speculative_mtp_greedy_k1`: method-correct as a greedy-equivalent verifier. Verification is exact for any draft, and tests show equality with greedy for natural drafts, all-rejected drafts, and oracle accept/reject mixes (`test_mtp_speculative_official.py`).
- Blocker 1: the draft is computed with single-token attention and no MTP KV history (A2-14). Draft quality is therefore not representative of the official head, so any agreement rate is not a fidelity measure. FAIL on draft fidelity.
- Blocker 2: the serving arm must either call the helper, or be renamed as diagnostic-only. Currently it is the latter.
- Blocker 3: native vLLM MTP is unverified (A2-05, A2-09, BLOCKED).
- Verdict: the verifier is correct. The `mtp` arm is not an MTP method run; it is a diagnostic. Treat as "not method-correct" until blocker 2 is decided.

**`h2o_mtp`**
- Blockers: all of `h2o_base`'s, plus the rollback case. The H2O transaction snapshots only tracker state, not physical KV (`H2OMTPTransaction`, `h2o.py:285-369`). It is unused by the engine, so no rollback happens. If speculation is ever enabled, H2O + speculative rollback would be wrong until A4-07 is fixed (BLOCKED).
- Blocker from `mtp`: the serving arm does not speculate, so `h2o_mtp` is H2O on a greedy stream with an MTP diagnostic attached.
- Verdict: not method-correct. Equivalent to `h2o_base` plus diagnostics.

**`da_mtp`**
- Blockers: all of `da_base`'s, plus the `mtp` blockers (diagnostic serving arm; A2-14 draft fidelity).
- Verdict: not method-correct. Scripted DA on a greedy stream plus diagnostics.

**Global blockers, all arms**
- 4B: unpinned, no local `config.json` (A1-04). Any 4B result is unverifiable against its own config.
- Scaffold path: `mtp.fc` is mapped into `mtp_proj` (A2-12, `real_engine.py:677-682`). Scaffold only. Must not be reported.
- Pinned configs for the campaign: BLOCKED (A1-07).

---

## 5. Recommended next steps (short)

1. Add the no-eviction oracle tests in §2 (`budget_ratio=1.0` must equal `base` token-for-token).
2. Add a text-side retention field next to `final_retained_fraction`.
3. Write `docs/H2O_DEVIATIONS.md`; rename the protected-outside-budget arm in results.
4. Decide the `mtp` serving arm: call the verifier, or label it diagnostic-only in the arm name.
5. Set or require `CUDA_DEVICE_ORDER=PCI_BUS_ID` for the GPU sampler, or match by UUID only.
6. Update stale `proposal_match_rate` mentions in `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:13,45` and `docs/NUMERIC_INTEGRITY_AUDIT_20261109.md:27`.
7. Add the EOS stop (A2-11) before any `base` quality-track use.
