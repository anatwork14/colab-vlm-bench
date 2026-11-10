# Claim Reconciliation — ITER2 TASK D3 (Auditor, wF:p9)

- Base: 7d209f5 (worktree `it2-p9`, branch `iter2/p9-review`)
- Scope: README, docs/, docs/review/, docs/audit/, artifacts/ (receipts and readiness report), runbooks/, configs/ (status fields), scripts that write receipts, and the old PR #20 body.
- Method: each claim checked against the file at the cited line and against the receipts and code it refers to. Runs performed for this document are listed in section 3. Nothing here was edited in the reviewed files; the lead decides the edits.

## 0. Labels

- **OK** — accurate as written; no change.
- **CORRECT** — wrong or overstated as written; the text must be changed to the stated correction.
- **HISTORICAL** — true of an earlier state; may be kept only under a banner saying it is superseded, with the superseding evidence linked.
- **DELETE** — no supporting evidence exists in the repo; remove the claim (or the literal that produces it).

---

## 1. Claim table

### 1.1 Readiness report — `artifacts/starlight_benchmark_readiness_report.md`

The report has a supersession banner at line 6 ("SUPERSEDED FOR MTP/DA RESEARCH CLAIMS"). The banner does not cover the lines below, which still read as current.

| ID | Location | Quote (short) | Evidence | Label | Action |
|---|---|---|---|---|---|
| R-01 | :14 | "Test Suite Status: 243 passed, 10 skipped, 0 failures" | Count is from an older state. The current figure at a3526c0 is 445 passed / 10 skipped (`docs/audit/MISSION_QUALIFICATION_20261109.md:7`). Not re-run in this audit. | HISTORICAL | Date the count and the commit, or remove. |
| R-02 | :20 | "This report certifies the scientific qualification repair (`P0-01`, `P0-02`, `P1-03`)" | Supersession notice `artifacts/qwen35_pretrained_qualification_receipt.json:3` says `SUPERSEDED_NO_POST_REPAIR_PRETRAINED_GPU_RECEIPT`. Progressive suite Test D crashes at 7d209f5 (section 3.2). | CORRECT | Replace "certifies" with "describes the state of a pre-repair attempt; not a qualification". |
| R-03 | :24–26 | "`max_abs_logits_diff = 0.0` (`0.00e+00`)" | Committed coverage receipt gives `4.76837158203125e-06` on cuda (`artifacts/model_coverage_and_parity_receipt.json:45`). CPU rerun gives `6.198883056640625e-06` (section 3.1). The 0.0 literal is hard-coded at `scripts/run_progressive_qualification.py:633`. | CORRECT | Use the receipt value with its device; do not state 0.0. |
| R-04 | :36–38 | "`18.95%` (`0.1895`) metric ... `kv_residency_preserved_pct = 100.0`" | `0.1895` was hand-fitted (supersession notice, `old_results_not_publishable`). Banner at :6 already marks the 18.95% target. | HISTORICAL | Keep under banner; remove the "PASS" framing. |
| R-05 | :60–66 | Seven-arm table with `PASS` for `da_base`, `mtp`, `h2o_mtp`, `da_mtp` and `638,976 B` | MTP acceptance forced (supersession notice). H2O bytes are tensor-size sums, not VRAM (supersession notice). Native speculative MTP is GATED (`qwen35...receipt.json:17`). | HISTORICAL | Keep under banner only; delete the PASS column. |
| R-06 | :76 | `REAL_MODEL_FORWARD_PASS` ... **PASS** | Supersession gate: `PRETRAINED_MODEL_FORWARD_AFTER_REPAIR` = `PENDING_LOCAL_GPU_RERUN` (`qwen35...receipt.json:16`). The coverage receipt supports 488/488 coverage and one-step parity, not the gate. | CORRECT | Set to `PENDING_LOCAL_GPU_RERUN` (or "CPU parity only, see section 3.1"). |
| R-07 | :79 | `MTP_CORRECTNESS_PASS` ... "bit-exact `DynamicCache` rollback (`max_diff = 0.0`)" — **PASS** | Supersession gate `NATIVE_SPECULATIVE_MTP_CORRECTNESS` = `GATED` (`:17`). The source-alignment matrix records that "bit-exact" was removed from the module docstring (`docs/audit/SOURCE_ALIGNMENT_MATRIX.md:197`). | DELETE | Remove the row and the "bit-exact" wording. |
| R-08 | :80 | `CCU_ISOLATION_PASS` **PASS** — "CCU 1, 2, 4 verified concurrent thread isolation" | Thread-level PyTorch test (`run_progressive_qualification.py:630–638`, "not scheduler throughput"). Not a serving test. Not listed in the supersession notice. | CORRECT | Keep the scope: "PyTorch multi-thread request-evidence isolation at 1,2,4; not serving". |

### 1.2 Generated release gates — `scripts/run_progressive_qualification.py`

| ID | Location | Quote (short) | Evidence | Label | Action |
|---|---|---|---|---|---|
| G-01 | :630–638 | Hard-coded `release_gates` dict with literal `"PASS (...)"` strings, written on success | The gates are not computed from test results; they are written regardless of what the tests measured. | DELETE (literal) | Compute each gate from the test outcome; remove the literal dict. |
| G-02 | :633 | `"verified one-step logits parity max_abs_diff=0.0"` | Contradicted by the committed receipt (4.77e-06 cuda) and the CPU rerun (6.20e-06 cpu, and Test A's own 0.00e+00 on CPU; section 3.1). | DELETE (literal) | Remove the number from the literal; report the measured value. |
| G-03 | :317, :343 | `assert res.logical_selection_fraction > 0.0` | The field was removed on purpose in R4-1: `tests/test_real_engine.py:783` asserts `not hasattr(res, "logical_selection_fraction")`, and `tests/test_infrastructure.py:59` asserts the key is absent. The **script was not updated**, so Test D crashes here at 7d209f5 (section 3.2). The engine is correct; the progressive suite is stale. | CORRECT | Replace with the current fields (`attention_read_restriction_ratio` / `attention_mask_restriction_ratio`, `real_engine.py:990–1006`) and state their meaning. Do not restore the removed name. |
| G-04 | :323 | `assert da_acc["kv_residency_preserved_pct"] == 100.0` | Valid meaning (DA keeps original KV). Keep, but it is not a KV saving claim. | OK | — |
| G-05 | :453 | `assert res.mtp_acceptance_rate == 0.0` | Field documented as `None = not verified (serving arm)` (`real_engine.py:1003`). The assertion is stale and conflicts with line 455 (`0.0 <= ... <= 1.0`). Test E is not reached in the CPU rerun (section 3.2), so this is expected to fail when reached. | CORRECT | Assert `is None` for the serving arm, or remove the forced-acceptance expectation. Do not weaken it to pass. |
| G-06 | :477 | `"mtp_acceptance_rate": 0.0` in the receipt | Same as G-05; a 0.0 written for a field that is "not verified" is the forced-acceptance pattern the audit retracted (`docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:12`). | CORRECT | Write `null`. |
| G-07 | :595 | default `--receipt-path` = `artifacts/qwen35_pretrained_qualification_receipt.json` | The default path is the file that now holds the supersession notice. A rerun on a different device would overwrite the notice with an unlabelled receipt. | CORRECT | Require an explicit `--receipt-path` or write to a dated path; never overwrite the notice. |

### 1.3 Coverage generator — `scripts/verify_model_coverage_and_parity.py` and its receipt

| ID | Location | Quote (short) | Evidence | Label | Action |
|---|---|---|---|---|---|
| C-01 | :68, :86 | `"scientific_status": "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE"` / `"OFFICIAL_TRANSFORMERS_HOOK_QUALIFIED"` as literals | Labels are constants, not functions of the measured coverage. The 25.82% scaffold figure differs on rerun (C-05). | CORRECT | Derive each label from measured coverage and the parity result. |
| C-02 | :176–177 | literal `custom_scaffold_label` / `official_hooked_label` in the recommendation block | Same as C-01. | CORRECT | Same as C-01. |
| C-03 | :138 | `"status": "PASS" if max_diff < 1e-4 else "FAIL"` | Computed correctly. Note the status depends on the device used (C-06). | OK | Record the device next to the status (it already does at `:163`). |
| C-04 | `artifacts/model_coverage_and_parity_receipt.json:35` | `"scientific_status": "OFFICIAL_TRANSFORMERS_HOOK_QUALIFIED"` with `"total_accounted_tensors": 488` | Coverage is a static tensor inventory. It says nothing about per-arm execution, GPU behaviour, or MTP. The receipt's own `recommendation` (`:53`) is correct. | CORRECT | Rename to `TENSOR_INVENTORY_COVERED`; do not use "QUALIFIED" for a tensor count. |
| C-05 | `artifacts/model_coverage_and_parity_receipt.json:14–16` | `mapped_tensors: 126`, `coverage_percentage: 25.82` | CPU rerun (this audit) gives `mapped_tensors: 119`, `coverage_percentage: 24.39`, `unmapped 369` (section 3.1). Same checkpoint, same script. Cause not identified in this audit. | CORRECT (unreconciled) | Re-run on the same device to separate device dependence from nondeterminism; record both. |
| C-06 | `artifacts/model_coverage_and_parity_receipt.json:41–45` | `"device": "cuda"`, `max_logits_absolute_difference: 4.768e-06` | CPU rerun: device `cpu`, diff `6.199e-06`. Both pass the 1e-4 tolerance. Value is device-dependent and is not 0.0. | OK (the receipt itself) / CORRECT (any text quoting 0.0) | Keep receipt; correct every text that says 0.0 (R-03). |
| C-07 | `artifacts/model_coverage_and_parity_receipt.json:2` | `"timestamp": "2026-11-09 08:17:37 UTC"` | Only the parity and coverage numbers are reproducible in this audit (section 3.1); the progressive suite is not. | OK | — |
| C-08 | `artifacts/qwen35_pretrained_qualification_receipt.json:3` | `"qualification_verdict": "SUPERSEDED_NO_POST_REPAIR_PRETRAINED_GPU_RECEIPT"` | Accurate. The notice correctly says it is not a run receipt. | OK | Keep. |
| C-09 | `artifacts/historical/qwen35_pretrained_qualification_receipt_20261109_before_scientific_repairs.json:804` | `"qualification_verdict": "ALL_TESTS_QUALIFIED"` | Generated before repairs; MTP 0.4/0.5 forced, DA 0.1895 hand-fitted, H2O bytes are tensor sums (supersession notice). Code at 7d209f5 cannot reproduce it. | HISTORICAL | Keep the file; the verdict string must never be cited. Add a banner line at the top of the file if the format allows, or an index entry. |
| C-10 | historical receipt `:185, :217, :453, :485` and `:744` | `physical_kv_bytes_freed: 638976` / `physical_kv_freed_bytes: 811008` | Supersession notice: sums of KV tensor-size deltas, not VRAM. | HISTORICAL | Keep under the existing note; do not cite as memory saving. |

### 1.4 README and campaign docs

| ID | Location | Quote (short) | Evidence | Label | Action |
|---|---|---|---|---|---|
| D-01 | `README.md:61` | "Status: **ENGINEERING PREPARED / GPU NOT QUALIFIED / NO RESULTS**" | Matches the supersession notice and the config status fields. | OK | Keep. |
| D-02 | `README.md:27`, `:113` | "No real GPU results" style caveats | Accurate. | OK | Keep. |
| D-03 | `README.md:45` | "Prior two-arm Base/MTP configs remain valid and can produce early real results" | `configs/a100_qwen_priority_full.json` has `revision: "UNPINNED"` and `source_drive_path: "REPLACE_DRIVE_PATH"`; its `status` is `PRIORITY_CONFIRMATORY_NOT_RUN`. The README does not say the configs are not pinned. `runbooks/SEVEN_ARM_QWEN.md:10–11` marks counts "not GPU-qualified". | CORRECT | "Prior configs are not pinned or GPU-qualified and produce no results yet." |
| D-04 | `README.md:37` | "Seven-arm confirmatory, 3,528 planned cells" | `configs/a100_qwen_7methods_full.json` status `PLANNED_NOT_RUN`; `docs/PROTOCOL.md:11,25` and `README.md:89` give 2,520 serving cells (5 × 2 × 2 × 42 × 3). 3,528 equals 2,520 + 1,008 arithmetically, but no file states that decomposition. | CORRECT | State which cells are serving, which are quality/priority, and show the arithmetic in the README. |
| D-05 | `README.md:49`, `:52` | "1,008-cell confirmatory serving grid" | 1,008 = 2 models × 2 modes × 2 datasets × 7 bins × 6 CCU × 3 reps (= 1,008; `configs/a100_qwen_priority_full.json`). Arithmetic is consistent; it is a different protocol from the seven-arm grid. | OK (arithmetic) / CORRECT (label clash with D-04) | Use distinct names for the two grids. |
| D-06 | `README.md:89` | "2,520 serving cells" | Arithmetic consistent with `docs/PROTOCOL.md:11`. | OK | Keep. |
| D-07 | `docs/audit/MISSION_QUALIFICATION_20261109.md:63` | `PRETRAINED_MODEL_VALIDATED` scoped to 0.8B, CPU HF forward, one canonical image, 488/488 with logits parity | The scope is stated correctly. However the committed parity value comes from a GPU receipt, and the CPU re-run gives a different coverage count (C-05). The verdict is valid only at the stated scope and needs the device recorded. | CORRECT (add device and the unreconciled count) | Add "device-dependent; scaffold count unreconciled (C-05)". |
| D-08 | `docs/audit/MISSION_QUALIFICATION_20261109.md:31` | "protected tokens are outside the budget (a labelled variant, `h2o_visual_anchor_text_budget`, not upstream H2O)" | This is a budget semantics that conflicts with the STARLIGHT contract (I-4/I-5 in `docs/harness/STARLIGHT_INVARIANTS_ORACLE.md`). It is labelled, so it is honest, but any table comparing `h2o_*` to `starlight_s1` must state it. | OK (labelled) / CORRECT (in comparison tables) | Keep; add the note to the comparison. |
| D-09 | `docs/SEVEN_ARM_PROTOCOL.md:39` | "B0 and B4: code route exists, GPU method activation/correctness not proven" | Accurate. | OK | Keep. |
| D-10 | `docs/SEVEN_ARM_PROTOCOL.md:43` | "H2O/DA+SD must verify draft/target cache semantics" | Accurate requirement. | OK | Keep. |
| D-11 | `docs/THIRD_PARTY_METHODS.md:6` | Declarative Attention "**NOT QUALIFIED** for Qwen3.5 0.8B/4B and A100" | Accurate. | OK | Keep. |
| D-12 | `docs/RESOURCE_BUDGET.md:7` | "approximately **88–228 A100 GPU-hours**" | Labelled as a proportional extrapolation. | OK | Keep (the label must stay). |
| D-13 | `docs/PROTOCOL.md:25` | "~75–185 A100-GPU-hours ... (wide, **unverified estimate**)" | Labelled unverified. | OK | Keep. |
| D-14 | `docs/RESOURCE_BUDGET.md:16` | "Provisional GPU budget (unverified)" | Labelled. | OK | Keep. |
| D-15 | `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:34` | "The retired label `SEVEN_ARM_PRETRAINED_PASS` (replaced by `SEVEN_ARM_HOOK_COVERAGE_PASS`, N-11) meant pretrained model/ViT forward coverage only; it did NOT imply MTP speculative verifier or paper-faithful DA" | Correct statement about the retired label; consistent with the supersession notice. | OK | Keep. |
| D-16 | `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:28` | "current custom DAFSM ... is **not** paper-faithful `<focus magic_chunks>` model-declared DA" | Consistent with `DA_PAPER_FAITHFUL_MODEL_DECLARED: GATED`. | OK | Keep. |
| D-17 | `docs/COLAB_STEPWISE_AUDIT.md:34` | "GPU model-load/MTP ... cannot be certified by GitHub CPU tests alone" | Accurate. | OK | Keep. |
| D-18 | `docs/AGENT_EXECUTION.md:20` | "Native dataset prompt rendering has not been certified identical" | Accurate. | OK | Keep. |
| D-19 | `AGENTS.md:42–44` | Gate-separation rules (`CPU_TESTS_PASS` ≠ `GPU_MODEL_LOAD_PASS` ≠ `MTP_CORRECTNESS_PASS`) | Rule text; the readiness report (R-07) violates it. | OK | Keep. |
| D-20 | `CHANGELOG.md:9–10` | "verified output length", "Independent native-quality inference profile" | Not checked in this pass (no receipt cited). | UNCHECKED | Lead to check against receipts or reword. |

### 1.5 docs/review and docs/audit

| ID | Location | Quote (short) | Evidence | Label | Action |
|---|---|---|---|---|---|
| A-01 | `docs/audit/SOURCE_ALIGNMENT_MATRIX.md:16` (A2) | MTP "PARTIAL" with forced acceptance removed | Consistent with the code state recorded in the matrix. | OK | Keep. |
| A-02 | `docs/audit/SOURCE_ALIGNMENT_MATRIX.md:74` | "Bit-exact DynamicCache rollback" claim at `real_engine.py:27` | Row A2-10 is marked resolved at `:197`. Checked in this audit: `grep -i "bit-exact" src/vlmbench/infrastructure/real_engine.py` returns nothing at 7d209f5. The phrase survives only in the historical matrix row and in the readiness report (R-07). | OK (for the code) | Remove the readiness report row (R-07), which is the remaining source of the phrase. |
| A-03 | `docs/audit/NUMERIC_INTEGRITY_LEDGER.md:86` | "Simulated / CPU / theoretical numbers labelled as GPU performance: none found" | Audit-time statement for the bench path; not rechecked here. | UNCHECKED | Keep as dated audit statement. |
| A-04 | `docs/review/BASELINE_AUDIT_PHASE0_1.md:61` | "a cell is `PASS_MEASURED` only when ... `completion_tokens == 256`" | Protocol rule; consistent with `docs/PROTOCOL.md`. | OK | Keep. |
| A-05 | `docs/review/REVIEW_PHASE1_2_AGY_DATA.md:48` | DS-06 `requested_text_length_bin: 256` mixing output and input bins | Review finding; status not re-checked. | UNCHECKED | Lead to confirm the fix status. |
| A-06 | `docs/audit/REVIEW_INTEGRATED_R4.md` and `REVIEW_INTEGRATED.md` | Prior auditor rounds | Historical rounds of this audit. | HISTORICAL | Keep; do not update retroactively. |

### 1.6 Old PR #20 body

| ID | Location | Quote (short) | Evidence | Label | Action |
|---|---|---|---|---|---|
| P-01 | GitHub PR #20, earlier body (superseded; commit `d10cbf3` exists) | Describes 238 tests and uses "genuine kernel", "certified", "approved" | The current body (read via `gh pr view 20 --json body`) replaces it. The old text is not in the repo. Its claims match the supersession notice's "not publishable" list. | DELETE | Do not reuse any phrase from the old body. If a copy exists in any other doc, delete it. |
| P-02 | GitHub PR #20, current body | Status body rewritten by lead | Consistent with this reconciliation as far as it was read. | OK | Re-check after R-02 and R-06 are corrected so the PR body and the report match. |

---

## 2. Summary of required lead decisions

1. Remove or correct the readiness report's PASS rows R-06 to R-08 and the 0.0 parity text R-03.
2. Remove the hard-coded `release_gates` literals (G-01, G-02) and derive gates from results.
3. Fix the stale Test E assertion (G-05) and the `logical_selection_fraction` attribute (G-03) before any progressive run.
4. Rename `OFFICIAL_TRANSFORMERS_HOOK_QUALIFIED` to a tensor-inventory label (C-01, C-04).
5. Resolve the scaffold count discrepancy (C-05) by re-running on one device.
6. Correct README D-03 and D-04 and state the budget-semantics difference in any `h2o_*` vs `starlight_*` comparison (D-08).

---

## 3. Runs performed for this reconciliation (CPU only, scratch directories, nothing committed)

Environment: `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=<it2-p9>/src`, interpreter `colab-vlm-bench/.venv`. Logs are in the scratch directory, not the repo.

### 3.1 Coverage and parity generator (`scripts/verify_model_coverage_and_parity.py`)

Ran to completion on CPU. Output written to the scratch directory (the script writes a cwd-relative path). Comparison against the committed receipt:

| Field | Committed (cuda, 08:17:37 UTC) | CPU rerun (16:24:50 UTC) |
|---|---|---|
| device | cuda | cpu |
| scaffold mapped / unmapped | 126 / 362 | 119 / 369 |
| scaffold coverage % | 25.82 | 24.39 |
| official coverage % | 100.0 | 100.0 (log) |
| max logits abs diff | 4.768e-06 | 6.199e-06 |
| parity status | PASS | PASS |
| predicted next token | 18978 "Green" | 18978 "Green" |

Interpretation: the official hooked coverage and the one-step prediction reproduce. The scaffold mapped count does not reproduce (C-05). The parity value is device-dependent and is not 0.0 on either device.

### 3.2 Progressive qualification suite (`scripts/run_progressive_qualification.py --device cpu`)

- Tests A, B, C: PASSED on CPU (log lines 13, 15, 17). Test A printed `Logits Parity max_diff=0.00e+00`, which differs from the generator's 6.20e-06 on the same device (both are "one-step parity" but on different code paths). Unreconciled; record it.
- Test D: **crashed** with `AttributeError: 'RealExecutionResult' object has no attribute 'logical_selection_fraction'` at `scripts/run_progressive_qualification.py:343`, exit code 1. At 7d209f5 the suite therefore cannot produce a complete receipt, and the release gates `SEVEN_ARM_PRETRAINED_PASS` (retired label, N-11), `MTP_PROPOSAL_DIAGNOSTIC_PASS` and `CCU_REQUEST_ISOLATION_TEST_PASS` have no reproducing run.
- Tests E, F, G: not reached. Test E's expected-failure status (G-05) could not be confirmed by running it.
- No receipt was written to the repo.

Consequence for the claim table: any PASS in the readiness report for the seven-arm and CCU gates (R-05, R-08) is not reproducible from 7d209f5 as of this audit.

### 3.3 Not run

- The repo test suite was not re-run in this audit (R-01 stays HISTORICAL until it is).
- No GPU run. The GTX 1080 Ti is shared and was not used.
- No check of `real_engine.py:27` docstring text (A-02 remains CORRECT-until-confirmed).
