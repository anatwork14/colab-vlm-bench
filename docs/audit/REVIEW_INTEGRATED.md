# Phase 3 adversarial verification of the integrated tree

- Reviewer: wF:p9 (Scientific Auditor), round 3, for lead wF:p5
- Reviewed tree: `/mnt/ssd/genos2/research_workspace/wt/wf-integrated` at **HEAD `27ec58a`** (read-only; not edited)
- Includes p7 `c0aeb68`, p8 `49e3d58` (via merge `7c5fb58`), and audit `f508071`
- Method: read the code paths named in the brief; ran targeted tests on the integrated tree (CPU only, `CUDA_VISIBLE_DEVICES=`, `PYTHONPATH=<tree>/src`); re-ran my H2O simulations against the new `h2o.py`; ran p8's measurement regression tests on the pre-p8 tree (`ebbf60d^`) to check fail-before/pass-after.
- Not re-run: the full suite (the lead reports 383 passed, 10 skipped, 0 failed). I did not re-verify that number.

Targeted run on `27ec58a`: `tests/test_mtp_speculative_official.py tests/test_real_engine.py tests/test_infrastructure.py tests/test_mtp_da_upstream_contracts.py` → **74 passed, 0 skipped**, exit 0. The 6 official-MTP tests ran (not skipped).

## Summary

| # | Item | Verdict | Severity | Key location |
|---|------|---------|----------|--------------|
| 1 | H2O fixed budget / recurrent exclusion / saturation | PARTIAL | HIGH | `h2o.py:206-216`, `real_engine.py:1523-1540` |
| 2 | MTP greedy k=1 reference | PARTIAL | MEDIUM | `real_engine.py:1219-1315`, `real_engine.py:1616-1619`, `:1707-1718` |
| 3 | DA masking gate and exact-zero test | PARTIAL | MEDIUM | `real_engine.py:1514-1516`, `:1622-1624`; `tests/test_real_engine.py:693` |
| 4 | Gates from partial coverage or threads | PASS (one naming overclaim) | MEDIUM-LOW | `real_engine.py:1352-1382` |
| 5 | p8 metric changes F-01…F-06 | PARTIAL | MEDIUM | `telemetry.py` (F-03), `cli.py` `_serving_outcome` (F-06) |
| 6 | Literal or simulated values emitted as measurements | PARTIAL | MEDIUM-LOW | `real_engine.py:1414-1422`, `:1655`; `amber.py:587` |
| 7 | Docs claiming more than evidence | PARTIAL | MEDIUM | `real_engine.py:1-30`, `h2o.py:1-12`, ledger rows 2.5 / C3-6 / F-16 |

---

## 1. H2O fixed budget (`src/vlmbench/infrastructure/h2o.py`)

**Fixed from prefill length: PASS.** `initialize_budget` sets `layer.budget = max(min_budget, int(L_prefill * ratio))` once (`h2o.py:112-122`). The engine calls it for full-attention layers only, after prefill (`real_engine.py:1457-1460`). The budget is not re-derived from the compacted length. The fallback at `h2o.py:154-157` re-derives from the current length, but only when `initialize_budget` was never called; the engine always calls it.

**Evict only on overflow, per layer: PASS** for non-saturated cases. `h2o.py:180-194` returns the full retained mask when `current_length <= budget`. Each layer owns its budget, scores, and masks (`h2o.py:33-44`).

**Recurrent exclusion for Qwen3.5 hybrid: PASS.** Full-attention layers are `(i+1) % 4 == 0` (`model_config.py:58-59`, `:88-89`), which gives 3,7,…,23 for 0.8B, matching the `layer_types` path (`:140-141`). The tracker marks the rest as recurrent (`real_engine.py:1049-1050`). Recurrent layers return a keep-all plan (`h2o.py:140-152`), skip score updates (`h2o.py:84-85`), and skip compaction (`h2o.py:256`). The engine plans and compacts only `full_attention_layers` (`real_engine.py:1523-1534`, `:1573-1575`).

**Re-run of compound simulation against the new code: fixed.** Text-only, L=300 (`scratchpad/h2o_compound_sim.py`, run against `h2o.py` at 27ec58a): cache goes 301→150 at step 1, then 151→150 every step through step 200. The old compounding (300→150→75→38→32) no longer occurs.

**Protected tokens inside the budget: FAIL (HIGH).** The budget is not enforced when protection saturates.
- `h2o.py:206-207` flags saturation when `protected_count >= budget`.
- `h2o.py:210-212` sets `recent_size = max(1, ...)`, so the recent window is 1 token.
- `h2o.py:215-216` gives `hh_budget = 0`.
- Nothing clamps the retained set to `budget`, so the plan keeps `protected + 1` slots, which exceeds `budget`.

Simulation (`scratchpad/h2o_fixed_sim.py`, L=340, visual=300, budget=170, against the 27ec58a `h2o.py`):

```
prefill L=340 visual=300 budget=170 final_cache=305
  step  1: cache= 305 protected_saturated=True text_tokens_kept=1
  step  2: cache= 305 protected_saturated=True text_tokens_kept=0
  step 30: cache= 305 protected_saturated=True text_tokens_kept=0
prefill L=340 visual=0 budget=170 final_cache=170
  step  1: cache= 170 protected_saturated=False text_tokens_kept=150
```

Consequences:
- On image prompts, every prompt text token after the image is dropped from step 2. The model keeps the visual block, the sinks, and the newest token only.
- The cache stays at 305 against a budget of 170, so the "fixed budget" is a soft target.
- `tests/test_infrastructure.py:429-450` codifies this overrun ("Visual protected keys can exceed target budget"). No test checks that text tokens survive on image prompts.

**"Indistinguishable from base" and the "6/1824" figure.** The outputs are not indistinguishable from base; the context is changed. The telemetry is what looks near-base. Full-attention layers number 6, and their slots sum to 6 × 304 = 1824. Each step evicts one slot per layer, so 6 of 1824 is the per-step increment. The reported `logical_selection_fraction` (`real_engine.py:1537-1540`, `:1681-1686`) therefore reports about 0.3% while nearly all prompt text is gone. The field name and the `LAST_STEP_FULL_ATTENTION_LAYER_SLOT_WEIGHTED` label describe a per-step rate, not cumulative loss. The telemetry does carry a limitation string (`real_engine.py:1700-1703`), which is the honest part.

**Other H2O notes (LOW).** `H2OMTPTransaction` snapshots only tracker state, not physical KV. The real engine does not use it (only `adapters.py:222-235`, the synthetic harness, does). `h2o.py:1-12` describes it as the MTP transactional machine for the engine, which overstates.

## 2. MTP: `RealSevenArmEngine._speculative_mtp_greedy_k1`

**Verification is target greedy argmax: PASS.** At `real_engine.py:1289-1291`, the proposal is compared with `argmax(full_logits[0,0])`, the target's prediction after `pending`. Accept emits the proposal and advances. Reject restores the snapshot and re-feeds `pending` alone (`:1299-1305`). The committed stream is therefore the target's greedy stream.

**MTP offsets: PASS.** The draft is `mtp_logits[0,0]` from the last kept position, embedding the target-selected token (`real_engine.py:806-810`). `mtp_position_ids` is the target-selected token's absolute position (`:1266`), consistent with the prefill (`:1430`) and decode (`:1560`) conventions.

**Snapshot covers KV and Gated-DeltaNet conv and recurrent state: PASS.** `_snapshot_dynamic_cache` clones keys and values for attention layers and conv, recurrent, and flag dicts for linear layers (`real_engine.py:241-258`). `_restore_dynamic_cache` copies them back (`:261-280`). `test_mtp_speculative_official.py:103-132` asserts the state changes after a wrong verify and is bit-identical after restore. LOW caveat: restore does not delete dict keys created during the verify, though none appear in the tested path.

**Acceptance is verifier-derived: PASS** for the helper: `accepted / proposed` (`real_engine.py:1313`).

**The serving-style mtp arm emits something acceptance-like: PARTIAL (MEDIUM).** `execute_request` does not call the helper. It still computes `proposal_match_rate = matched / proposed` by comparing each candidate with the target's actual greedy token (`real_engine.py:1596-1600`, `:1616-1619`). That number is emitted at `:1707-1718` and gated by `MTP_PROPOSAL_DIAGNOSTIC_PASS` (`:1628-1630`). It is marked `proposal_match_rate_publishable: False` and `committed_draft_tokens: 0`, and `mtp_acceptance_rate` is `None`. The value is measured, not a literal, but it reads like an acceptance rate.

**Label `cpu_greedy_k1_reference_not_vllm_mtp` in every emitted metric: FAIL (MEDIUM).** The label appears only in the helper's return dict (`real_engine.py:1217`, `:1308`). Nothing in `execute_request`, `RealExecutionResult`, telemetry, or gates carries it. The label is not emitted at all; grep finds it only in the helper and in tests. The serving telemetry uses a different classification, `REFERENCE_DRAFT_NO_PERSISTENT_MTP_KV_NO_SPECULATIVE_SERVING` (`:1708`).

**Test not vacuous: PASS.**
- `test_mtp_speculative_official.py:72` asserts `proposed > 0` and `accepted + rejected == proposed`.
- `:76-85` forces every proposal to be wrong (`draft+1`) and asserts `accepted == 0`, `rejected == proposed`, and tokens equal to greedy. The reject case genuinely rejects.
- `:88-100` uses an oracle (even steps accept, odd steps reject) and asserts both `accepted > 0` and `rejected > 0`.
- All 6 tests ran, none skipped.
- LOW: the natural-draft test (`:67-73`) does not assert `accepted > 0`. This is correct for a correctness test, but it does not show the real head ever agrees with the target.

## 3. DA: `DA_MASKING_EXERCISED` and the exact-zero test

**Is the exact-zero test measured on the real hooked forward? PASS.** `test_p7_r3_da_constructed_official_case_masks_real_reads_to_exact_zero` (`tests/test_real_engine.py:693`) wraps `engine.model.forward`, the official hooked path, and checks `call["attn"]`, the post-softmax attention matrices returned by the hooked forward. For every masked key of the decode row it asserts `weights[:, k] == 0.0`. The mask is the scripted diagnostic mask, not a model-emitted `<focus>` mask. Hooked attention applies it as `masked_fill(-1e9)` before softmax (`real_engine.py:206-212`), so exact zero comes from float underflow in the hooked forward.

**`DA_MASKING_EXERCISED` gate: PARTIAL (MEDIUM).** The gate is `final_da_restriction_ratio > 0` (`real_engine.py:1622-1624`). That ratio is computed from the constructed mask (`:1514-1516`), not from measured attention weights. The gate can read PASS when the mask was constructed but not applied. The exact-zero test covers application, but the gate does not. The telemetry field `attention_read_restriction_ratio` (`:1655`) is mask-derived and is labelled as a read restriction.

**Scaffold test.** `test_real_da_attention_mask_restriction` (`tests/test_real_engine.py:94`) tests `QwenFullAttentionLayer` directly with a hand-built mask. It does not exercise the official path. LOW.

**Question-readable deviation: PARTIAL (MEDIUM).** The deviation is documented in code: `_da_scripted_tags` makes the question span GLOBAL on image prompts (`real_engine.py:904-906`, `:913-914`). Nothing in `docs/` records it. `docs/audit/SOURCE_ALIGNMENT_MATRIX.md:84` (my item A3-02) still says FAIL and recommends sink+tail, with no pointer to the deviation.

## 4. Gates from partial coverage or threaded tests

- **`REAL_MODEL_FORWARD_PASS`: PASS.** PASS only when `is_official_hf_hooked` (`real_engine.py:1370-1374`). That flag requires tensor coverage 1.0, parameter coverage 1.0, and verified logits parity (`:582-586`). Partial coverage gives `PARTIAL_COVERAGE_NOT_QUALIFIED`.
- **`REAL_MULTIMODAL_INFERENCE_PASS`: PASS.** Requires official hooking plus a vision encoder, an image present, and `vis_invoked` (`real_engine.py:1352-1355`). Partial coverage gives `PARTIAL_COVERAGE_NOT_QUALIFIED` (`:1360`).
- **`SEVEN_ARM_PRETRAINED_PASS`: PASS, with a naming overclaim (MEDIUM-LOW).** It is PASS for any single request once the model is officially hooked, regardless of arm or of whether all seven arms ran (`real_engine.py:1352-1354`, `:1376`). Partial coverage cannot produce PASS. `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:34` states it means forward coverage only. The name still reads as a seven-arm result.
- **`CCU_ISOLATION_PASS`: PASS.** Hard-coded `GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED` (`real_engine.py:1381`). Threaded tests set no gates.
- **Inconsistency (LOW).** The bundle path sets `self.hf_model` and `hook_ctx` even when `is_official_hf_hooked` is False (`real_engine.py:575-587`), and forward routes through `hf_model` whenever it exists (`:775`). A checkpoint with coverage below 1.0 can therefore run the official HF forward while its gates say PARTIAL. This direction is conservative, but the labels do not describe what ran.

## 5. p8 metric changes (independent review)

Verified against the pre-p8 tree: I copied the p8 test files into an export of `ebbf60d^` and ran `tests/test_measurement_regressions.py`. **10 failed, 1 passed.** The passing test is `test_stream_completion_is_service_timing_only`, a guard. The 10 failures include the F-01 missing-prediction, duplicate-ID, unknown-ID, and reference-less tests, the F-03 parse tests, the F-02 e2e test, and the F-04 dpi test. Some failures are `TypeError` or `ImportError` from the missing functions, so they show the functions are new, not only that the bug is absent. `test_resume_policy.py` and `test_derivation_routes.py` cannot be collected on the pre-p8 tree because `resume.py` and `derivation_labels.py` do not exist there. These are new modules, so those tests cannot discriminate in that tree.

- **F-01 `score_proxy` missing predictions = 0: PASS.** `quality.py:48-100` denominators are over references. A missing prediction counts as EM=0 and F1=0. Duplicate and unknown IDs now raise. That is a new hard failure, not a silent drop, and it is documented. LOW: `missing_predictions` is a subset of `failures_counted_zero`. Summing both double-counts in a reader's head, though not in the mean.
- **F-02 E2E includes client queue wait: PASS, with a definition caveat.** `admitted_stream` adds `acquired_at − submitted_at` to `e2e_ms` (`bench.py:32-48`). `submitted_at` is set before payload creation in `one()` (`bench.py:189-192`). TTFT (`bench.py:101`) and TPOT (`bench.py:103-106`) are computed from the HTTP-send origin, so there is no double count. `docs/METRICS.md:19-20` states the definition. The caveat is interpretive: with all requests created at t=0, e2e p95 is dominated by queue position, so the metric is no longer comparable to earlier receipts. `code_sha256` in receipts (`92c8b68`) partly mitigates this.
- **F-03 multi-GPU `nvidia-smi` sums: PARTIAL (MEDIUM).** Summing `memory.used` per sample and taking the max over time is the right peak definition. Utilization uses max, which is also right. The problem is the label. The report field says `gpu_memory_aggregation: "sum_over_visible_devices"` (`telemetry.py`, `report()`), but the query is `nvidia-smi --query-gpu=...` with no index filter. `nvidia-smi` enumerates every device the driver exposes, not the `CUDA_VISIBLE_DEVICES` subset. On a shared host the sum covers other tenants' GPUs. This was already device-total before p8 (`gpu_memory_scope: total_used_device`), so the number did not change, but the label now says it is filtered. Fix: filter by the process's visible indices, or rename the aggregation to "sum_over_all_host_devices" and report the device count. Malformed rows dropping a whole sample, counted in `parse_failures`, is PASS.
- **F-04 DPI-keyed render cache: PASS.** `mmlong.py:74-76` puts DPI in the directory name. `test_mmlong_render_cache_is_keyed_by_dpi` fails on the pre-p8 tree.
- **F-05 resume only completed receipts: PASS.** `resume.py` classifies receipts. COMPLETED requires PASS_MEASURED, matching identity keys, a sample-set digest, and the measured request count. `code_sha256 == UNRECORDED` yields STALE, so a run without `VLMBENCH_SOURCE_SHA` stops before GPU start. That is conservative. Quarantine moves rather than deletes. `test_resume_policy.py` passes on 27ec58a (part of the 44-passed earlier run).
- **F-06 exit codes: PARTIAL (MEDIUM-LOW).** `cli.py` `_serving_outcome` returns `IN_PROGRESS` with rc 0 when `attempted` is empty and no receipt is in a non-complete state. Receipts with status MISSING are not counted in `non_completed` (only FAILED, STALE, INVALID are carried). A job that attempts nothing and has only MISSING cells beyond the cap exits 0. That is an unlabelled exclusion. Fix: count MISSING planned cells as non-completed when `IN_PROGRESS` would otherwise be returned, unless the cap was actually reached.

## 6. Literal or simulated values emitted as measurements

Grep for `mtp_acceptance`, `kv_read`, `sparsity`, `latency`, and literal `0.0` in `src/`:

- `bench.py:293` `"mtp_acceptance_rate": "NOT_MEASURED"`: PASS.
- `real_engine.py:1616`: `mtp_acc_rate = None`, and the old literal `0.0` at the former `:1431` is gone: PASS.
- **`real_engine.py:1414-1422` (MEDIUM-LOW).** `final_logical_fraction`, `final_da_restriction_ratio`, and `final_da_causal_area_ratio` start at 0.0. If the decode loop never runs (for example `max_new_tokens=1`), or DA is not applied, `attention_read_restriction_ratio` is emitted as a literal 0.0 (`real_engine.py:1655`). The DA gate does report `NOT_EXERCISED`, which mitigates this, but the numeric field still reads as a measured zero. Fix: initialise to `None`.
- `datasets/amber.py:587` (LOW): `coverage_rate: 0.0` in the empty-annotation branch that also sets `status: NOT_MEASURED`. Use `None`.
- `infrastructure/adapters.py:319` (LOW): `ttft_ms = 0.0` is a default. The synthetic harness is labelled `SYNTHETIC_REFERENCE_POLICY_TEST_HARNESS` (`adapters.py:162-173`), so this is contained.
- `h2o.py:150`, `:192`: `logical_selection_fraction: 0.0` is computed (no eviction), so these are measured zeros, not literals: PASS.
- `declarative_attention.py:238`: `read_reduction_estimate` is an estimate and is labelled as one; `physical_kv_read_saving_claimed: False` (`:242`): PASS.

## 7. Docs claiming more than the evidence

- **`real_engine.py:1-30` module docstring (MEDIUM).** Item 5 says "Authentic 15-tensor MTP drafting, verification, and bit-exact `DynamicCache` rollback". Verification and rollback exist only in the helper that `execute_request` never calls. Item 6 says "zero state pollution", which the one non-vLLM concurrency test does not establish (it covers `h2o_base` only). Item 1 hard-codes "100% tensor coverage (488/488)" and parameter counts. I did not re-derive those counts in this review.
- **`h2o.py:1-12` (LOW-MEDIUM).** Describes the MTP transactional state machine as part of H2O. The engine does not use it; only the synthetic adapter does.
- **`docs/audit/NUMERIC_INTEGRITY_LEDGER.md` (stale, LOW-MEDIUM).** Row 2.5 (`:69`) and F-16 (`:158`) still say the MTP acceptance `0.0` is hard-coded at `real_engine.py:1431` and OPEN. The current code emits `None` at `:1616` (fixed in `d6e0b0f`). Row C3-6 (`:101`) says "OPEN-OWNER F-05", but the F-05 row (`:147`) says FIXED in `92c8b68`. The ledger contradicts itself.
- **`docs/audit/SOURCE_ALIGNMENT_MATRIX.md` (my own, stale, LOW).** Statuses are as of `24f57d3`. A3-02 (`:84`) still says FAIL, and the H2O compounding and forced-acceptance items are now fixed. It should be refreshed to the integrated state.
- **`docs/review/REVIEW_PHASE1_2_SOL_LEAD_DATA_AUDIT.md:15`** uses "Validated integrated source" for an older commit. It is historical and not a current claim. LOW.
- `docs/METRICS.md`, `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md`, `docs/THIRD_PARTY_METHODS.md` ("NOT QUALIFIED"), and `docs/AGENT_COLAB_CLI.md` ("BACKEND_UNQUALIFIED") were checked and are appropriately hedged. I found no GPU, speedup, or VRAM claims that exceed the evidence in those files.

## Recommended actions (in priority order)

1. **H2O (HIGH):** enforce the budget under protection saturation, or document and gate image-prompt H2O results. Add a test that text tokens after the image survive on image prompts. Replace or rename `logical_selection_fraction` so it does not read as cumulative loss.
2. **MTP (MEDIUM):** emit the `cpu_greedy_k1_reference_not_vllm_mtp` label in any metric that comes from the helper, or stop calling it a reference metric. Rename or drop `proposal_match_rate` from the serving arm, or label it non-acceptance.
3. **DA (MEDIUM):** make `DA_MASKING_EXERCISED` depend on measured masked-key weights, or rename it to a mask-construction gate. Record the question-span deviation in `docs/`.
4. **Telemetry (MEDIUM):** fix the `sum_over_visible_devices` label, filtering by `CUDA_VISIBLE_DEVICES` indices or renaming it.
5. **Zeros (MEDIUM-LOW):** initialise `final_*` telemetry fields to `None`. Change `amber.py:587` to `None`.
6. **Exit codes (MEDIUM-LOW):** count MISSING planned cells as non-completed when the job would return IN_PROGRESS with rc 0 and was not capped.
7. **Docs (MEDIUM):** correct the `real_engine.py` module docstring and `h2o.py` header. Refresh the ledger rows 2.5, C3-6, F-16, and F-17, and my matrix statuses.
