# ITER2 D2b report (wF:p8, scripts/ and reconciliation items)

Source of findings: auditor wF:p9, docs/harness/CLAIM_RECONCILIATION.md (read-only).
No committed historical receipt was edited or regenerated. `git diff 7d209f5..HEAD -- artifacts` is empty.

## Commits (test-first; one commit per item)
| SHA | Item | Result |
|---|---|---|
| a26578c | G-01/G-02 | Release gates computed from recorded test outcomes by `compute_release_gates()`. Gate values are bare labels only. Evidence numbers are in `release_gate_evidence`. Each test runs under `guarded()`, so failures are recorded, not dropped. |
| af0897b | G-03/G-05/G-06 | Test D uses `h2o_logical_selection_accounting.final_retained_fraction` (> 0 kept). Test E asserts `mtp_acceptance_rate is None` (the old `== 0.0` recorded a fabricated zero). Reporting uses current field names; unmeasured values are null. |
| 8989061 | G-07 | Default receipt is `artifacts/runs/<UTC>_qwen35_pretrained_qualification.json`. The supersession notice is never a target. Existing receipts are refused without `--force`, and `main()` refuses before running any test. |
| d4e6334 | C-01/C-02 | Scaffold and official statuses, parity status, unmapped-module lists and labels are functions of measured coverage and parity. Official hook is QUALIFIED only at 100% coverage and parity PASS. Output follows the G-07 policy. |
| 46f0c9b | lint | `noqa: E402` on two sys.path-dependent imports (pre-existing, no behaviour change). |

## C-05: mapped_tensors 126 (committed receipt) vs 119 (HEAD CPU rerun)
- Cause: commit b68cc9f ("exact-shape weight mapping"). It is the only commit in a5dff1d..HEAD that introduces the shape check on `real_engine.py` (`git log -S`).
- Exactly 7 checkpoint tensors change status. Before b68cc9f they were slice-copied into the scaffold; after it they stay unmapped:
  - `model.language_model.layers.{3,7,11,15,19,23}.self_attn.q_proj.weight`: checkpoint 4096x1024, scaffold 2048x1024 (6 tensors).
  - `model.visual.merger.linear_fc2.weight`: checkpoint 1024x3072, scaffold 1024x1024 (1 tensor).
  - 126 - 7 = 119.
- The old count was inflated by slicing, not by real coverage. The reference Qwen3.5 attention (`transformers/models/qwen3_5/modeling_qwen3_5.py:774-776`) views `q_proj` output per head and chunks it into query and gate. Taking the first 2048 rows therefore does not select the query matrix. The 126 receipt counted weights that were loaded incorrectly.
- The official 488/488 hook path (`prefer_official_hf=True`) does not use this branch.
- Committed receipt `artifacts/model_coverage_and_parity_receipt.json` (a5dff1d) is stale for the scaffold count (25.82%). Corrected HEAD value: 119/369 (24.39%, auditor CPU rerun). Not regenerated, per instruction.

## G-level and MMLongBench-Doc metric discrepancy (item g): BLOCKED
Statements that disagree:
- `docs/METRICS.md:8`: "MMLongBench official F1 | publisher multi-stage scorer ...".
- `docs/audit/NUMERIC_INTEGRITY_LEDGER.md:52`: "Native: publisher F1 pipeline, NOT_MEASURED".
- `README.md:78`: names no metric; says the official scoring workflow "must be run separately".
- `docs/harness/NATIVE_SCORING_AUDIT.md:13`: already records the conflict (multi-stage scorer vs F1 pipeline).

Verifiable offline: no. No upstream MMLongBench-Doc scorer exists on disk. Only dataset fixtures (`*/tests/fixtures/bench/mmlongbench-doc.json`) and `src/vlmbench/datasets/mmlong.py` are present. Neither statement is marked as matching the official protocol. Resolution needs the pinned upstream scorer (owner-approved fetch), then METRICS.md:8 and the ledger line corrected to match it.

## Not in scope, observed
- tests/test_multimodal_fixture.py::test_progressive_qualification_test_c_multimodal_execution fails (asserts CCU_ISOLATION_PASS == PASS; engine reports GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED). Pre-existing, outside scripts/.

## Counts (targeted, nice -n 10; no full suite)
- tests/test_harness_inventory.py: 29
- tests/test_hallucination_not_measured.py: 5
- tests/test_qualification_gates.py: 18
- tests/test_coverage_parity_labels.py: 8
- tests/test_numeric_artifact_provenance.py: 2
- Total: 62 passed, 0 failed.
- Full suite at HEAD: NOT RUN (held per lead instruction).
