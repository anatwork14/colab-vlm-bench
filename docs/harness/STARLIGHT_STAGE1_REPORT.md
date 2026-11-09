# ITER2 D1 — STARLIGHT Stage 1 controller: report (wF:p7 → wF:p5)

Branch `iter2/p7-starlight-controller`, base `7d209f5`. HEAD `1d89a86`. Worktree clean.
Label for everything: `cpu_pytorch_reference_not_vllm`. Not vLLM, not GPU, no speed/VRAM/quality claim.

## Full suite at HEAD
`PYTHONPATH=src CUDA_VISIBLE_DEVICES= colab-vlm-bench/.venv/bin/python -m pytest -q` (nice -n 5, one run):
**515 passed, 10 skipped, 0 failed** (185 s). The 10 skips are the pre-existing retired notebook tests.
Baseline at 7d209f5: the first full run was killed by its own timeout under host load (no summary line),
so no baseline count is claimed here.

## Commits (test-first; red before green)
| Item | Red | Green |
|---|---|---|
| Pure policy contract | 435f2b3 | 40255fe |
| Signal sources (H2O keep set w/o protection, DA/grounding mapping, evidence validity) | fb78ea9 | dbf9a2b |
| Engine wiring, 3 arms | 7f72704 | a8bc349 |
| Oracle conformance (reason codes, records, H-6 checks, telemetry markers) | f3251a9 | 1d89a86 |

Test counts: policy 14, signals 16, engine 17, oracle 23 → 70 STARLIGHT tests, all pass.

## Files
- `src/vlmbench/infrastructure/starlight.py` (new): pure policy, signal helpers, summaries.
- `src/vlmbench/infrastructure/h2o.py`: new method `keep_mask_without_protection` (compute_eviction_plan unchanged).
- `src/vlmbench/infrastructure/grounding_evidence.py`: read-only `items()`, `keyword_index()`.
- `src/vlmbench/infrastructure/real_engine.py`: arms, `_starlight_build_step_plans`, `_starlight_step_records`, telemetry.
- `docs/harness/STARLIGHT_STAGE1_DESIGN.md`: contract, signal sources, deviations, limitations, oracle coverage matrix (section 12).

## Contract as implemented
eligible_to_evict = h2o_candidate AND NOT (DA_protected OR grounding_protected). Protected never evicted.
Candidates = H2O keep complement computed at budget − |protected| over all entries (no H2O internal protection),
so retained = keep ∪ protected ≤ budget. protected > budget → `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET`,
protected kept, overflow reported. A candidate set that would overflow the budget raises ValueError.
Recurrent (GDN) layers exempt, labelled `GDN_EXEMPT_NOT_TOKEN_ADDRESSABLE`. Full-attention layers only.
Protection follows original token positions through compaction (`compact_orig_positions`).

## Arms
`starlight_s1` (DA+grounding), `starlight_s1_no_da` (grounding), `starlight_s1_no_grounding` (DA).
Seven original arms unchanged. One existing assertion updated (`test_r5_h2o_arm_reports_method_variant_and_keeps_arm_ids`):
it compared `SUPPORTED_ARMS` to exactly the seven; it now checks the seven plus exactly the three new IDs.
Two existing STARLIGHT-adjacent assertions changed by design: verdict lists now carry a reason code
(`test_expired_evidence_*`). DA attention masking now uses exact arm membership (`DA_MASKING_ARMS`), because
the substring `"da" in arm` would have enabled a DA mask for `starlight_s1_no_da`. Seven-arm behaviour unchanged.

## Telemetry (`telemetry["starlight"]`)
measurement_label, method_variant (`starlight_stage1::<signals>`), method_variant_is_upstream_h2o=False,
signals_enabled, da_signal (`SCRIPTED_DIAGNOSTIC_NOT_MODEL_DECLARED`), grounding_signal_status,
grounding_evidence_verdicts (status, reason, oracle code), grounding_protected_by_evidence,
budget_total, layers, recurrent_layers_exempt, gdn_layer_records, steps, status_counts, infeasible_steps,
evicted_total, retained_total, max_retained_per_layer_step, h2o_candidates_vetoed,
conflicts_resolved_by_protection, conflicts_by_source, protected_count_by_source (entry×layer×step),
invariant_protected_evicted (0), invariant_evicted_not_candidate (0), cache_length_mismatch_steps (0),
structural_tokens_evicted, structural_policy (`NOT_RATIFIED_A4_POLICY_DEPENDENT`),
protected_irrelevant_fraction (`NOT_MEASURED_NO_RELEVANCE_LABELS`), text_side_retained_fraction_last_step,
decision_records (opt-in `starlight_record_decisions=True`; per step × layer, entries keyed by original position).
`h2o_logical_selection_accounting` is `NOT_APPLICABLE_STARLIGHT_ARM` on these arms.

## Auditor oracle (wF:p9 STARLIGHT_INVARIANTS_ORACLE.md): coverage
Full matrix in design doc section 12. Summary:
- **Covered:** I-1, I-2, I-3, I-4, I-5, I-9, I-10; HC-3, HC-3b, HC-4, HC-6, HC-8 (spans), HC-11, HC-15;
  A-DA, A-GROUND, A-NODA, A-NOGROUND, A-H2O, A-TOGGLE (with H2O-only negative control), A-MASK.
- **Partial:** I-6 (GDN exempt + label; byte-hash NOT tested), I-7 (survivor ids; RoPE values NOT compared),
  HC-4b (no turn IDs in this engine), HC-7, HC-9 (sequential only; engine is synchronous), HC-10 (within-request;
  no cross-call KV exists), HC-12 (no hash), HC-13, HC-14 (length counter only; no logit recomputation).
- **Not covered:** HC-5 (no relevance labels; metric reported as NOT_MEASURED), I-8 (structural vision tokens are
  measured, not protected; A-4 open).
- Two coverage tests (I-7, A-MASK) were added after the implementation, not red-first. Disclosed in the doc.

## Open questions for the owner / lead
1. **Budget basis (A-1/brief):** STARLIGHT uses total budget with protection reserved inside it; H2O keeps its R4-1
   outside-budget semantics. Side-by-side tables must state the difference (oracle §5.6).
2. **I-8 / A-4:** vision_start/vision_end are NOT protected. In the tests, structural tokens are measured
   (`structural_tokens_evicted`); the oracle wants them never evicted. Ratify or drop.
3. **A-2:** unmet budget with no eligible candidates raises ValueError in the policy instead of a status.
   Not reachable through the engine (candidates always exist when length > budget). Confirm.
4. **A-5 / HC-7:** missing evidence declaration gives explicit status `NO_EVIDENCE_DECLARED` and protects nothing.
   The oracle asks for fail-closed or an error. Owner decision.
5. **HC-1 wording:** the oracle says "H2O top-score" for a DA/H2O conflict, but its own candidate definition
   makes candidates low-score. Implemented as low-score candidate vs protection. Auditor to confirm.
6. **DA sink size:** the scripted DA sink (first min(16, L/4) positions) can exceed a small budget, making
   configs INFEASIBLE by design. Consider capping the sink protection.
7. **Protection cost:** protection costs budget, so text-side retention falls more than under H2O at the same
   ratio. Not a quality result. Quality is unmeasured.

## Not done
- No push, no merge, no paid GPU, no vLLM. Nothing qualified as GPU or vLLM evidence.
- No baseline comparison at 7d209f5 (see above).
- Interleaved multi-request decoding (HC-9 strict form) and logit recomputation (HC-14) not implemented.
