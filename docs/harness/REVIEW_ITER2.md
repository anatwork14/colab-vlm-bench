# Adversarial review of the iteration-2 integrated tree (ITER2 TASK D6)

- Reviewer: wF:p9 (Scientific Auditor), Haiku 5.5
- Reviewed HEAD: `8a050851717d1597f2b2a9de6e87f583f9742ff1` (Merge iter2 lead integration)
- Worktree: `it2-p9`, no source edited. All mutations were applied to a scratch copy of `src/` and `tests/` outside the repo and reverted; nothing was committed from them.
- Verdict in one line: the STARLIGHT Stage 1 policy code is sound on its own terms, but the ablations and the H2O comparison are **confounded** as implemented, several tests that the reports cite as coverage are **vacuous**, the inventory's `READY` status can be produced by a fabricated environment, and the progressive-suite gate labels have a wrong derivation.

## 0. Method and what was run

| Step | Command or method | Result |
|---|---|---|
| STARLIGHT tests (4 files) | `PYTHONPATH=src nice -n 5 pytest tests/test_starlight_{policy,signals,oracle,engine}.py` | 70 passed in 5.6 s |
| Mutation runs (24 mutations, 4 STARLIGHT test files each) | scratch copy, one mutation at a time, reverted after each | table in §6 |
| Confound reproducer | scratch test file in the copy; `_engine` + `_scored_tracker` + `_step` from `tests/test_starlight_engine.py` | §2, S-1 |
| Inventory false-READY | `scripts/harness_inventory.py::build_report` on a scratch models root, receipts dir, authorization file, stub `vllm` package and mock GPU; one precondition varied at a time | §3, I-1 |
| Gate derivation | `compute_release_gates` / `verdict_for` / `guarded` on mock test outcomes | §4, G-1..G-3 |
| Full suite at HEAD | `pytest -q` (background, nice -n 5) | see §7 |

Q5 (HC-1 wording) is answered in §1.3.

---

## 1. STARLIGHT Stage 1 against the auditor oracle

Oracle: `docs/harness/STARLIGHT_INVARIANTS_ORACLE.md`. Design under review: `docs/harness/STARLIGHT_STAGE1_DESIGN.md` (§12 is the claimed coverage matrix). Code: `src/vlmbench/infrastructure/starlight.py`, wiring in `src/vlmbench/infrastructure/real_engine.py`.

Status key: PASS = code and a test that would fail if the behaviour broke (checked by mutation where marked M#). PARTIAL = behaviour present but the test misses part of the claim. FAIL = behaviour absent or contrary to the oracle, or the test is vacuous. OPEN = depends on an owner ruling (A-1..A-6).

### 1.1 Invariants

| ID | Status | Evidence (file:line) | Note |
|---|---|---|---|
| I-1 eligibility formula | PASS | `starlight.py:185` `eligible = candidates - protected`; M1 (evict protected) → 18 failures | Test `test_I1_I2_*` recomputes from the same records the code wrote, so it is weaker than the mutation shows. |
| I-2 protected never evicted | PASS | `starlight.py:185`; M1 → 18 failed; M2 (evict protected only when INFEASIBLE) → 3 failed | Holds in both statuses. Engine wiring checked by M14 and M16. |
| I-3 no silent discard, closed enum | PASS (reading) | `starlight.py:41-46`, `:226-230`; `test_I3_*` | Not mutated. |
| I-4 infeasibility explicit, protected kept | PASS with consequence S-3 | `starlight.py:175-178`, `:187`; M7 (truncate protected) → 3 failed | Protected entries are kept. The non-protected entries are all evicted in the same step (S-3). |
| I-5 budget when feasible | PASS | `starlight.py:187-191` guard; M3 (guard removed) → 1 failed; M13 → 29 failed | Budget holds by construction (`h2o.py:236-266` keep set ∪ protected). |
| I-6 GDN exempt and labelled | PARTIAL | Layer scope from `full_attention_layers` in `real_engine.py` `_starlight_build_step_plans`; label `test_HC12_*` | The oracle's byte-identity hash of GDN state is not tested. |
| I-7 position preservation | PARTIAL | Original positions tracked (`starlight.py:da_protection_from_spans`, `compact_orig_positions`); M14 (map not compacted) → 21 failed | RoPE values of retained entries are not compared. |
| I-8 structural vision tokens never evicted | FAIL (OPEN A-4) | `real_engine.py:1672-1674` marks them structural; they are not protected; `test_HC13_*` (`tests/test_starlight_oracle.py:210`) asserts only `isinstance(..., int)` | The test cannot fail on the behaviour the oracle requires. |
| I-9 determinism | PARTIAL | `test_I9_HC15_*` runs twice in one process | Cross-process determinism is not tested. |
| I-10 complete decision records | PASS | `real_engine.py:1488-1545` record fields; M18 (DA flag forced False) → 1 failed | |

### 1.2 Hard cases (oracle §4)

| ID | Status | Note |
|---|---|---|
| HC-1 DA/H2O conflict | PARTIAL | Helper test `tests/test_starlight_engine.py:99-111` and M11/M4 detect loss of protection. The A-TOGGLE contrast is confounded (S-1). The "H2O-only negative control" is not an H2O-only arm. |
| HC-2 grounding/H2O conflict | PARTIAL | Same mechanism as S-1 applies to the grounding ablation (by code, `real_engine.py:1475`; not separately reproduced). M12 → 5 failed. |
| HC-3 protection > budget | PASS, consequence S-3 | M7 detected. |
| HC-3b protection = budget | PASS | `test_HC3b_*`; M11 → 6 failed. |
| HC-4 expired evidence | PARTIAL | Request-ID mismatch is detected (M9 → 4 failed). The oracle's step-based expiry is not implemented. |
| HC-4b stale evidence | PARTIAL | Source hash mismatch detected (M10 → 1 failed). Turn IDs do not exist. |
| HC-5 irrelevant evidence measured | FAIL (honestly reported) | `protected_irrelevant_fraction` = `NOT_MEASURED_NO_RELEVANCE_LABELS`. Oracle required a number; the report says "not covered" (design §12). |
| HC-6 long-context pressure | PASS (reading) | `test_long_context_pressure_*`. Its per-step count checks rely on the counters that M8 shows are vacuous (S-5). |
| HC-7 missing declaration | PARTIAL, OPEN A-5 | `NO_EVIDENCE_DECLARED` is explicit (`real_engine.py:1693`) but protects nothing and does not fail closed. For DA it cannot occur: DA spans are scripted from the prompt. |
| HC-8 malformed declaration | PASS | `starlight.py:340-343` raises; `test_HC8_*`, `test_malformed_*`. |
| HC-9 multi-request isolation | PARTIAL | Sequential only. `execute_request` is synchronous. |
| HC-10 multi-turn / evicted never returns | PARTIAL | Within-request only. |
| HC-11 no eligible candidates | PASS for the oracle's wording, OPEN A-2 | `starlight.py:179-183` and `:187-191` raise `ValueError` instead of reporting a status. |
| HC-12 GDN exemption | PARTIAL | Label and exemption present; hash not tested. |
| HC-13 vision span integrity | FAIL (OPEN A-4) | Same as I-8. |
| HC-14 cache consistency | FAIL | The only check is the H-6 counter (`real_engine.py:1938-1942`). Disabling it (M17) leaves all 70 tests passing (S-4). No logit recomputation. |
| HC-15 determinism | PARTIAL | As I-9. |

### 1.3 Q5 — HC-1 wording (DA/H2O conflict)

Question: does the oracle intend "protection overrides low-score H2O candidates"?

**Answer: yes.** HC-1 is a conflict only when H2O wants to evict an entry (it is an H2O candidate, i.e. outside H2O's keep set, which means low cumulative score) and DA declares that entry protected. The protection must win. An entry that H2O keeps (a high-score entry) is not a conflict, because the eligibility rule I-1 (`eligible = h2o_candidate ∧ ¬protected`) never evicts it either way.

The oracle's own HC-1 text says "H2O top-score", which is an error in the oracle written by this auditor (`docs/harness/STARLIGHT_INVARIANTS_ORACLE.md`, HC-1 row). It should read "H2O eviction candidate (low score) that is DA-declared". The implementation follows the intended meaning (design §12 says so). The oracle wording should be corrected in a later docs pass; this review does not edit it.

### 1.4 Activation assertions (oracle §3)

| ID | Status | Note |
|---|---|---|
| A-DA | PASS | M11 → `test_A_DA_activates_*` fails. |
| A-GROUND | PASS | M12 → `test_A_GROUND_*` fails. |
| A-NODA | PASS | M15 (no-DA arm leaks DA) → 2 failed. |
| A-NOGROUND | PASS (reading) | `test_A_NODA_and_A_NOGROUND_*`; not mutated. |
| A-H2O | PASS (reading) | `test_A_H2O_*`. |
| A-TOGGLE | FAIL on the oracle's "only the signal differs" rule | See S-1: the two arms differ in eviction of non-protected entries too. |
| A-MASK | PASS (reading) | `test_A_MASK_*`; not mutated. |

---

## 2. Ablations and H2O fairness (question 1, second half)

### S-1 — The ablations do not differ only by the removed signal (HIGH, owner wF:p7)

- **What happens.** The H2O candidate set is computed at `effective = budget_total − n_protected`, where `n_protected` counts the entries protected by the *enabled* sources (`real_engine.py:1474-1476`, `keep = tracker.keep_mask_without_protection(l_idx, k_len, effective)`). Removing DA therefore changes `n_protected`, which changes the keep budget, which changes which non-protected entries are candidates. The removed signal changes the H2O candidate set as well as the veto.
- **Reproducer (run in the scratch copy).** Fixture as in `tests/test_starlight_oracle.py:241-246`: scores `[0,0,5,4,3,2,1,1,1,1]`, length 10, budget 6, DA span `sink_global (0,2)`.

  ```
  FULL  (da+grounding): candidates [0, 1, 5, 6, 7, 8]  evicted [5, 6, 7, 8]  retained [0, 1, 2, 3, 4, 9]
  NO_DA (grounding):    candidates [0, 1, 6, 7]        evicted [0, 1, 6, 7]     retained [2, 3, 4, 5, 8, 9]
  ```

  Entry 5 is not DA-protected. It is evicted only in the full arm. Removing DA *keeps* it. The full-vs-no-DA difference therefore includes a change in how non-DA entries are treated.
- **Why the tests miss it.** `tests/test_starlight_policy.py:131-148` (`test_ablation_differs_only_by_removed_signal`) feeds the same `candidates` list to all three plans, so it checks the policy only, not the engine's candidate computation. The engine-level A-TOGGLE test (`tests/test_starlight_oracle.py:256-262`) asserts the negative control `{0,1} ⊆ no_da.evicted`, which holds, but it does not check that the candidate sets match.
- **Oracle §5.2** requires "the only difference is the DA signal". FAIL.
- **Owner fix options.** (a) Compute candidates once from a budget basis that does not depend on the ablation (for example always the full-arm `effective`, with protection applied only as a veto), or (b) keep the current basis and report the candidate-set delta per arm as a measured confound. In both cases add an engine-level test that the candidate sets of `starlight_s1` and each ablation are identical on the fixture, or that the difference is exactly the released protected entries.

### S-2 — H2O and STARLIGHT budgets are not the same quantity (HIGH for side-by-side reporting, owner lead wF:p5)

- **H2O** (`h2o.py` `compute_eviction_plan` docstring, lines 140-146): protected sinks and visual tokens sit outside the budget; `B = max(min_budget, int(unprotected_prefill_length × ratio))`; retained = protected + B.
- **STARLIGHT** (`real_engine.py:1664`): `budget = max(min_budget, int(len(full_prompt) × ratio))`; protection is inside the budget (retained ≤ budget).
- **Consequence.** At the same `budget_ratio`, H2O keeps `P_h2o × (1 − ratio)` more entries per layer than STARLIGHT, where `P_h2o` is the number of sink and visual entries. The protected sets also differ (sinks and visual for H2O; DA and grounding for STARLIGHT). Equal ratio is not equal retention, so quality and retention deltas at equal ratio are confounded by the budget basis.
- The design (§Open question 1) states this. It is acknowledged, not hidden. What is missing is a rule for side-by-side tables. Required: report retained entries per layer for every arm, and never compare quality at equal ratio without the retained-count column.

### S-3 — An INFEASIBLE step evicts every non-protected entry (HIGH, owner wF:p7 with lead ruling on A-2)

- **What happens.** When `n_protected > budget`, `effective = 0` (`real_engine.py:1475`), `keep_mask_without_protection` returns all False, every entry is a candidate, and every non-protected entry is evicted in that step. The result is `retained = protected` only, which exceeds the budget and is labelled INFEASIBLE.
- **Reproducer.** `tests/test_starlight_engine.py:126-133` asserts `plan.evicted == (7, 8, 9)` for length 10 with 7 protected entries. That is the whole non-protected set.
- **Why it matters.** The oracle (I-4) requires that protected entries stay and that the status is explicit. It does not authorise evicting all other context. The design says INFEASIBLE configs are "reported, not hidden", but the step's eviction is not reported separately from ordinary H2O evictions in the records (`reason` only prefixes the text). The DA sink (first `min(16, L/4)` positions, design §Open question 6) can make a config infeasible by design, so this path is reachable in normal runs, not only in edge cases.
- **Owner ruling needed.** Either keep the existing behaviour and record "evicted because infeasible" as its own reason code, or evict nothing when infeasible (protected plus all non-candidates retained, status INFEASIBLE). The current behaviour is the most destructive of the options and is not stated in the design.

### S-4 — The H-6 cache-length check is not tested (HIGH, owner wF:p7)

- The check at `real_engine.py:1938-1942` increments `starlight_cache_mismatch_steps`. `test_HC14_cache_length_counter_matches_retained_count_every_step` (`tests/test_starlight_oracle.py:214-216`) only asserts the counter equals 0, which is also the value when the check is absent.
- **Mutation M17** (check disabled, `if False:`) → **70 passed**. The check is not tested.
- The design §12 says "length counter checked every step (`test_HC14_*`)". That claim is not supported.
- Mutation M16 (model never receives the eviction plan) is caught by other tests (20 failed), not by the counter. Detection of cache drift therefore rests on tests that check token and record consistency, not on H-6.

### S-5 — Invariant counters are vacuous (MEDIUM, owner wF:p7)

- `summarize_plans` computes `protected_evicted` and `evicted_not_candidate` from the decisions (`starlight.py:264-269`), which are generated by the same policy code they audit.
- **Mutation M8** (protected-eviction counter disabled, `if False:`) → **70 passed**. No test seeds an inconsistent plan and checks that the counter reports it.
- The design §12 lists "I-2 ... engine invariant counter" as covered. It is not, in the sense the oracle means.
- Fix: a test that feeds `summarize_plans` a plan with one protected EVICT decision and expects `protected_evicted == 1`.

### S-6 — Unverified evidence can protect and no test catches it (MEDIUM, owner wF:p7)

- `evidence_validity` (`starlight.py:395-396`) returns `unverified` for `verified == False`, and `grounding_protection_from_pool` protects only `valid` items (`starlight.py:427`).
- **Mutation M23** (unverified items fall through to the valid branch) → **70 passed**. No test checks that unverified evidence protects nothing.

### S-7 — Grounding keyword variants are not tested (MEDIUM, owner wF:p7)

- `starlight.py:433` protects bare, space-prefixed, capitalised and space-capitalised keyword forms.
- **Mutation M22** (only the bare form) → **70 passed**. Under the Qwen tokenizer a keyword that appears in the prompt as ` Green` or `Green` tokenizes differently from `green`; with only the bare form, those occurrences are unprotected. This is under-protection, which the oracle forbids for grounding-declared spans. The design's "documented limitation" covers split words, not missing variants.

### S-8 — Positions and budgets bound decode only (LOW, lead wF:p5 for wording)

- Both arms initialise the H2O budget from the prefill length and evict only from the first decode step (`real_engine.py:1758-1770` for the H2O budget; the STARLIGHT plan runs inside the decode loop). The prefill cache holds the full prompt for every arm. The per-step budget is therefore not a peak-memory bound. This is shared by both arms (fair between them) but any memory sentence must say "per decode step".

### S-9 — The DA signal is scripted, not model-declared (LOW for claims, owner wF:p7)

- DA spans are computed from the prompt layout (`_da_scripted_tags`, `real_engine.py` 1676-1690). The arm name `starlight_s1` therefore means "scripted-DA protection", not paper DA. The design labels it `SCRIPTED_DIAGNOSTIC_NOT_MODEL_DECLARED`. Any result text that calls this "DA" without the qualifier overclaims.

---

## 3. Inventory: can a cell be READY without every precondition checked? (question 2)

**Yes. A false READY is constructible.** Reproducer: `scratchpad/falseready.py` (scratch, not committed). It runs `scripts/harness_inventory.py::build_report` on a scratch models root with the real `config.json`, a manifest that lists zero files, a repository-config copy with 40-hex placeholder revisions, an authorization file written by the caller, receipts whose content is only `{"status":"PASS_MEASURED","cell_id":...}`, a stub `vllm` package on `sys.path`, a mock A100 GPU, and mock datasets whose checksums are written into the config copy.

Result: **528 READY cells** (24 pilot, 504 full). The same run with 4B weights missing keeps the 4B cells blocked as expected.

Each vector is necessary. Varying one at a time from the all-in state:

| Variant | READY | Blocking status now shown |
|---|---:|---|
| all vectors | 528 | none |
| receipt removed | 0 | `NOT_TESTED` (504 + 24) |
| manifest removed | 0 | `BLOCKED_WEIGHTS_UNVERIFIED` (504 + 24) |
| revision left UNPINNED | 0 | `BLOCKED_REVISION_UNPINNED` |
| authorization denied | 0 | `BLOCKED_OWNER_AUTHORIZATION` |
| vLLM absent | 0 | `BLOCKED_VLLM_NOT_INSTALLED` |

### I-1 — `READY` is reachable from fabricated preconditions (HIGH, owner wF:p8)

| Vector | Location | Defect |
|---|---|---|
| V1 weights present = config.json only | `scripts/harness_inventory.py:199` (`probe_weights` starts at `:193`) | "present" needs `config.json`, not a weight file. |
| V1 verified with an empty manifest | `:211-225`, `:131` | A manifest with `files: {}` passes sizes and hashes vacuously. The manifest is also in the same directory as the weights, so "verified" means "matches a file the same party wrote". It is not compared with an upstream hash. |
| V2 receipt content not checked | `:275-287`, specifically `:284` | Any JSON with `status == PASS_MEASURED` and a matching filename counts. No check of model revision, GPU, head SHA, token count, timestamp, or a hash of the receipt body. Gate at `:151`. |
| V3 authorization self-attested | `:246-253` | Reads a repo-local JSON the caller can write. No signature, no owner reference. |
| V4 vLLM checked by `find_spec` only | `:256-257` | A stub package passes. Import is not attempted. |
| V5 revision checked by regex only | `:316-317` and the `SHA40` check in `classify_cell` | Any 40-hex string passes, including a placeholder. Not compared with the manifest revision. |
| V6 dataset checksum from the same config as the path | `probe_dataset`, `:260-270` | The expected SHA comes from the config that names the file, so the check is circular unless the config is pinned by an independent source. |
| V7 GPU probed on the host only | `probe_gpu`, `:229-245` | A host GPU is accepted for cells that may run on another runtime. Lower severity. |

- **Tests.** The 29 tests in `tests/test_harness_inventory.py` do not exercise V1–V6. Most call `classify_cell` with an injected `healthy_env()` dict, so the probes are bypassed. The probe-level tests (`test_probe_weights_*`, `test_probe_authorization_*`, `test_probe_gpu_*`) cover sizes, hashes, a missing manifest and CSV parsing; none covers an empty manifest, receipt content, a placeholder revision, or a stub `vllm` package. No test constructs a present-but-fabricated precondition.
- **Claims.** `docs/harness/ITER2_D2_REPORT.md:30`: "No READY cell is possible until owner inputs exist". This is false as shown. `docs/harness/PILOT_INVENTORY.md:5`: "no cell can be READY here" is true only because the receipts directory is empty on this host, not because the code prevents it.
- **Fix.** Weights: check each file listed in an upstream-pinned manifest (SHA from the revision, not from the local directory) and require the listed set to equal the set on disk. Receipts: validate the receipt body against the cell (model revision, GPU name, head, token counts, body hash) and record the receipt file's hash in the inventory. Authorization: reference an owner-signed record outside the repo. vLLM: `importlib.import_module` in a subprocess. Revision and checksum: cross-check against the pinned manifest, not the config alone.
- **Committed artifact.** `docs/harness/PILOT_INVENTORY.json` records `head: 7d209f5`, not the integrated HEAD. Regenerate at HEAD. (LOW.)

---

## 4. Progressive-suite gate derivation (question 3)

**No gate reports PASS for a test that did not run.** Checked by mock:

- Test D absent → `SEVEN_ARM_PRETRAINED_PASS`, `REAL_MULTIMODAL_INFERENCE_PASS`, `ACTUAL_BACKEND_HOOK_PASS` = `NOT_TESTED`.
- Test D raised → the same gates = `FAIL`.
- Test G raised → verdict `INCOMPLETE_OR_FAILED: test_g_ccu_concurrency`.

The defects below are real but not false PASSes.

### G-1 — `MTP_CORRECTNESS_PASS` is always FAIL in an all-passing run (MEDIUM, owner wF:p8)

- `_engine_gate` (`scripts/run_progressive_qualification.py:634-643`) collects the value of the gate across all seven arms. Non-MTP arms report `NOT_APPLICABLE` (`real_engine.py:1605`), MTP arms report `GATED_NATIVE_MTP_VERIFIER_REQUIRED`. The set has two members, so the function returns `FAIL`.
- Reproducer: `compute_release_gates` with seven arm gate dicts and all tests `PASS` → `'MTP_CORRECTNESS_PASS': 'FAIL'`.
- A correct run reports a gated state and shows a FAIL in the receipt. Anyone reading the receipt will take it as a test failure.
- Fix: derive from the MTP arms only, or treat `NOT_APPLICABLE` as a non-value.

### G-2 — "PASS" means "no exception", and the same receipt contradicts itself on isolation (MEDIUM, owner wF:p8)

- `guarded` (`:609-616`) returns `PASS` for any test that returns normally. The gate `CCU_REQUEST_ISOLATION_TEST_PASS` = `_test_status(g)` (`:656`) therefore reads PASS when Test G returns.
- In the same receipt, the engine's own value `CCU_ISOLATION_PASS` is `GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED` (`real_engine.py:1609`, comment: "A single request cannot independently certify concurrent isolation"). Test G asserts that value (`:564`), so both labels are in one receipt and they disagree.
- Test G's isolation check is weak. It compares citation evidence ids only (`:566-568`), not output tokens against solo runs, and all threads share one `engine` instance per CCU level. A leak of tokens without a citation would not be detected. The gate name should be renamed to what is measured (`PYTORCH_THREAD_CITATION_LEAK_CHECK`) and the engine label should not be contradicted.

### G-3 — `SEVEN_ARM_PRETRAINED_PASS` is a construction flag (LOW, owner wF:p8/p7)

- Test D asserts `res.gates["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"` (`:294`). The engine sets that literal from `self.is_official_hf_hooked` (`real_engine.py:1582`). The gate therefore records that the hook was attached, not that the inference was correct. Coverage and parity are covered separately by Test A, so this is a labelling issue, not a false pass.

### G-4 — Thresholds differ (LOW, owner wF:p8)

- Test A asserts parity ≤ 1e-5 (`run_progressive_qualification.py:116`). The coverage generator uses 1e-4 (`verify_model_coverage_and_parity.py:138`). Choose one value.

---

## 5. Claims in p7 and p8 reports that exceed the evidence (question 4)

| Report | Claim | Evidence | Verdict |
|---|---|---|---|
| `STARLIGHT_STAGE1_REPORT.md` | "70 STARLIGHT tests, all pass" | Reproduced: 70 passed | OK |
| `STARLIGHT_STAGE1_REPORT.md` | "Covered: I-2 ... engine invariant counter" | S-5 (M8 → 70 passed) | OVERCLAIM |
| `STARLIGHT_STAGE1_DESIGN.md` §12, HC-14 | "length counter checked every step (`test_HC14_*`)" | S-4 (M17 → 70 passed) | OVERCLAIM |
| `STARLIGHT_STAGE1_DESIGN.md` §12, A-TOGGLE | "with the H2O-only negative control" | The control is `starlight_s1_no_da`, which changes candidates (S-1) | OVERCLAIM |
| `STARLIGHT_STAGE1_DESIGN.md` §12, HC-1 | "covered" | Correct for the veto; the contrast with the ablation is confounded (S-1) | PARTIAL |
| `STARLIGHT_STAGE1_DESIGN.md` §12, HC-13 | "measurement" | Test asserts only the type of the count (`test_starlight_oracle.py:210`) | Honest label, but the test cannot show a nonzero or zero result |
| `STARLIGHT_STAGE1_REPORT.md` | "Seven-arm behaviour unchanged" | Full suite passes at HEAD (§7); the seven-arm tests are not separately checked for behaviour equality with 7d209f5 | PARTIAL |
| `ITER2_D2_REPORT.md:30` | "No READY cell is possible until owner inputs exist" | §3: READY reproduced with fabricated inputs | OVERCLAIM |
| `PILOT_INVENTORY.md:5` | "no cell can be READY here" | True only because the receipts directory is empty | HONEST BUT FOR THE WRONG REASON |
| `ITER2_D2b_REPORT.md` | "Test E asserts `mtp_acceptance_rate is None`" | Confirmed (`run_progressive_qualification.py:457`) | OK |
| `ITER2_D2b_REPORT.md` | "C-05 cause: b68cc9f exact-shape mapping; 7 tensors change status" | Checkpoint shapes confirmed: six `self_attn.q_proj` at 4096×1024 and `model.visual.merger.linear_fc2` at 1024×3072 | OK |
| `ITER2_D2b_REPORT.md` | "Pre-existing failing test in `tests/test_multimodal_fixture.py`" | Suite and the file both pass at HEAD (§7) | STALE |
| `ITER2_D2_REPORT.md` | "Full suite NOT COMPLETED" | Honest | OK |
| `PYTORCH_REFERENCE_PROBE.md` | "bf16 may come from emulation; do not assume fast" | Caveated | OK |
| `EXECUTION_LEDGER.md` | "No Opus review has occurred" | Self-declared | OK |

---

## 6. Mutation results (scratch copy; each mutation reverted)

Test set: `tests/test_starlight_{policy,signals,oracle,engine}.py` (70 tests at HEAD). "Failed" counts are failures in that run. Zero means the mutation was not detected.

| # | Mutation | Result | Interpretation |
|---|---|---|---|
| M1 | policy evicts protected (`eligible = set(candidates)`) | 18 failed | detected |
| M2 | evicts protected only when INFEASIBLE | 3 failed | detected |
| M3 | no budget overflow guard | 1 failed | detected |
| M4 | policy ignores DA | 13 failed | detected |
| M5 | policy ignores grounding | 9 failed | detected |
| M6 | vetoed count zeroed | 5 failed | detected |
| M7 | INFEASIBLE truncates retained to budget (silent discard) | 3 failed | detected |
| M8 | protected-eviction counter disabled | **0 failed** | **vacuous (S-5)** |
| M9 | expired/stale evidence protects | 4 failed | detected |
| M10 | stale SHA check removed | 1 failed | detected |
| M11 | engine DA protection removed | 6 failed | detected |
| M12 | engine grounding protection removed | 5 failed | detected |
| M13 | budget not reduced by protection | 29 failed | detected (guard raises) |
| M14 | position map not compacted | 21 failed | detected |
| M15 | no-DA ablation leaks DA | 2 failed | detected |
| M16 | model never receives eviction plan | 20 failed | detected by other tests |
| M17 | H-6 cache-length check disabled | **0 failed** | **vacuous (S-4)** |
| M18 | DA flag in records forced False | 1 failed | detected |
| M20 | internal sink protection re-enabled in keep mask | 5 failed | detected |
| M21 | DA span end inclusive | 5 failed | detected |
| M22 | grounding: only the bare keyword form | **0 failed** | **gap (S-7)** |
| M23 | unverified evidence protects | **0 failed** | **gap (S-6)** |
| M24 | text-only request treated as stale | 5 failed | detected |

---

## 7. Full test suite at HEAD

Background run `pytest -q -p no:cacheprovider` at `8a05085`, nice -n 5, log at `scratchpad/full_suite.log`.

Result at `8a05085`: **575 passed, 10 skipped, 0 failed** in 152 s (EXIT 0). The 10 skips are the retired-notebook tests (`tests/test_notebooks.py`).

Two earlier attempts did not finish: a first run hit its 900 s timeout (EXIT 124) at about 46% under host load, so it gives no result. The rerun above is the only suite result used in this review.

`tests/test_multimodal_fixture.py` passes (7 passed). The D2b report's statement that its `test_progressive_qualification_test_c_multimodal_execution` fails is therefore stale; the file asserts `CCU_ISOLATION_PASS == GATED_...` at line 101, which matches the engine.

---

## 8. Findings summary

| ID | Severity | Owner | File:line | Summary |
|---|---|---|---|---|
| S-1 | HIGH | wF:p7 | `real_engine.py:1474-1476`; `tests/test_starlight_policy.py:131` | Ablations change the H2O candidate set; "differs only by signal" fails; policy test hides it |
| S-2 | HIGH (reporting) | wF:p5 | `h2o.py:140-146`; `real_engine.py:1664` | Budget bases differ; equal ratio is not equal retention |
| S-3 | HIGH | wF:p7 / wF:p5 (A-2) | `real_engine.py:1475`; `tests/test_starlight_engine.py:126-133` | INFEASIBLE step evicts every non-protected entry |
| S-4 | HIGH | wF:p7 | `real_engine.py:1938-1942`; `tests/test_starlight_oracle.py:214-216` | H-6 check untested (M17 → 70 passed); design claims it is covered |
| S-5 | MEDIUM | wF:p7 | `starlight.py:264-269` | Invariant counters vacuous (M8) |
| S-6 | MEDIUM | wF:p7 | `starlight.py:395-396` | Unverified evidence protection untested (M23) |
| S-7 | MEDIUM | wF:p7 | `starlight.py:433` | Grounding variant forms untested (M22); under-protection possible |
| S-8 | LOW | wF:p5 | `real_engine.py:1758-1770` | Budget is per decode step, not a peak bound (shared by both arms) |
| S-9 | LOW | wF:p7 | `real_engine.py:1676-1690` | DA is scripted; name must say so in results |
| I-1 | HIGH | wF:p8 | `harness_inventory.py:199, 211-225, 246-257, 275-287, 316-317` | READY reachable with fabricated weights, receipts, authorization, vLLM, revision (528 cells reproduced) |
| I-2 | LOW | wF:p8 | `PILOT_INVENTORY.json` head | Committed inventory at 7d209f5, not HEAD |
| I-3 | MEDIUM (claim) | wF:p8 | `ITER2_D2_REPORT.md:30` | "No READY possible" is false |
| G-1 | MEDIUM | wF:p8 | `run_progressive_qualification.py:634-643` | `MTP_CORRECTNESS_PASS` derives to FAIL in all-pass runs |
| G-2 | MEDIUM | wF:p8 | `run_progressive_qualification.py:609-616, 656, 564-568` | "PASS" = no exception; receipt contradicts itself on isolation; isolation check weak |
| G-3 | LOW | wF:p8/p7 | `run_progressive_qualification.py:294`; `real_engine.py:1582` | `SEVEN_ARM_PRETRAINED_PASS` is a construction flag |
| G-4 | LOW | wF:p8 | `run_progressive_qualification.py:116`; `verify_model_coverage_and_parity.py:138` | Parity tolerance 1e-5 vs 1e-4 |
| O-1 | LOW (oracle doc) | wF:p9 (me) | `docs/harness/STARLIGHT_INVARIANTS_ORACLE.md` HC-1 | "H2O top-score" should read "H2O eviction candidate (low score)"; to correct in a docs pass |

Open owner decisions (not defects): A-2 (no-candidate budget status, relates to S-3), A-4 (structural tokens, I-8/HC-13), A-5 (missing declarations, HC-7), A-6 (decode-only eviction, S-8), and the budget basis for side-by-side tables (S-2).

Nothing above was committed from a mutation, and no source file in `src/`, `scripts/` or `tests/` was edited.
