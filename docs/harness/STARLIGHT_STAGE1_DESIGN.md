# STARLIGHT Stage 1 controller: design

Status: CPU PyTorch reference (`cpu_pytorch_reference_not_vllm`). Not vLLM, not GPU, not a speed
or VRAM result. Branch `iter2/p7-starlight-controller`, base `7d209f5`.

## 1. Owner contract

```
eligible_to_evict   = h2o_candidate AND NOT internally_protected
internally_protected = DA_protected  OR  grounding_protected
external eviction   = H2O-proposed candidates only
```

- A protected entry is never evicted (`invariant_protected_evicted`, recomputed from decisions).
- Evicted entries are a subset of H2O candidates (`invariant_evicted_not_candidate`).
- A feasible plan retains at most `budget` entries per full-attention layer.
- When protected count > budget, the plan is `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET`. Protected
  entries stay, unprotected candidates are evicted, and the overflow is reported. It is never
  silently dropped.
- A candidate set that would leave more than `budget` entries retained is rejected with
  `ValueError`. The policy does not accept an inconsistent caller.

## 2. Components

| Module | Role |
|---|---|
| `src/vlmbench/infrastructure/starlight.py` | Pure policy (no torch, no numpy). `plan_starlight_layer` returns the retained and evicted partition with per-entry `EntryDecision` records (entry id, layer, H2O candidate flag, protection flag, signals fired, decision, reason). Also the signal helpers and `summarize_plans` / `combine_summaries`. |
| `src/vlmbench/infrastructure/h2o.py` | New method `keep_mask_without_protection`: H2O's recent-window + heavy-hitter keep choice with no internal protection. Existing `compute_eviction_plan` is unchanged. |
| `src/vlmbench/infrastructure/grounding_evidence.py` | Two read-only accessors, `items()` and `keyword_index()`. ECRD behaviour unchanged. |
| `src/vlmbench/infrastructure/real_engine.py` | Arms `starlight_s1`, `starlight_s1_no_da`, `starlight_s1_no_grounding` (`STARLIGHT_ARM_SIGNALS`). Per-step helper `_starlight_build_step_plans`. Telemetry under `telemetry["starlight"]`. |

Tests: `tests/test_starlight_policy.py` (14), `tests/test_starlight_signals.py` (16), `tests/test_starlight_oracle.py` (23),
`tests/test_starlight_engine.py` (17).

## 3. Candidate set and budget

- Total budget per full-attention layer: `B = max(min_budget, int(prefill_len * budget_ratio))`,
  fixed from the prefill length, as H2O fixes its budget.
- Protection is computed per layer for the current physical cache, including the new token.
- If `length <= B`, there are no candidates and nothing is evicted. This is the no-eviction case.
- Otherwise, the H2O keep set is computed at `B_eff = B - |protected|` over ALL entries with no
  H2O-internal protection (`keep_mask_without_protection`). Candidates are its complement.
- The controller then vetoes the candidates that are protected. Retained = keep ∪ protected, so
  `|retained| = B_eff + |protected ∩ H2O_keep_complement| <= B`.
- H2O's own sink and visual protection are disabled under STARLIGHT. The controller owns it.

Consequence: protection costs budget. A protected entry that H2O would have evicted is kept,
and the text-side budget shrinks by the same amount.

## 4. Signal sources

### 4.1 DA (scripted diagnostic, `DA_protection`)

- Spans come from `_da_scripted_tags`, anchored to the prefill, as in the decode-only DA arm.
- GLOBAL with `start == 0` → `sink_global`. GLOBAL with a later start → `question_global`.
- FOCUS → `focus:<focus_id>` (`focus:scripted_visual_segment`). LOCAL is a mask-only scope and
  does not protect.
- Spans are matched on ORIGINAL token positions (`da_protection_from_spans`). Each layer keeps
  an original-position array that follows its compaction (`compact_orig_positions`). A token's
  protection therefore survives eviction and reindexing.
- This is the scripted diagnostic. The model does not declare these scopes. The label
  `SCRIPTED_DIAGNOSTIC_NOT_MODEL_DECLARED` is in telemetry.
- Starlight arms do NOT build a DA attention mask. `DA_MASKING_ARMS` is exact membership, so
  `starlight_s1_no_da` does not enable masking through a substring match.

### 4.2 Grounding (evidence keyword spans)

- Source: the `EvidencePool` keyword index that ECRD matches on (`keyword_index()`). Benchmark
  answers are never read. `gold_answer_prohibited` is still enforced by the pool.
- Each keyword is tokenized in the bare, space-prefixed, and capitalized forms. Every contiguous
  token occurrence in the prompt is protected, labelled with the supporting evidence ids.
- Only `valid` evidence protects (see 4.3).
- Telemetry reports the protected prompt positions per evidence id, so a too-generic keyword is
  visible (`grounding_protected_by_evidence`). It is measured, not hidden.

### 4.3 Evidence validity

`evidence_validity` returns `(status, reason)` per evidence id:

| Status | Reason | Protects |
|---|---|---|
| `unverified` | `verified_false` | no |
| `expired` | `pool_request_id_mismatch` (pool built for another request) | no |
| `stale` | `source_sha256_mismatch` (image hash differs from the declared source) | no |
| `valid` | `source_sha256_match` | yes |
| `valid` | `source_unchecked_text_only` (no image; the gap is labelled) | yes |

`EvidenceItem` carries no timestamp, so "expired" means request-scoped, not time-scoped.

If grounding is enabled and the request has an image but no image bytes or path, grounding is
`NOT_PROTECTED_IMAGE_SOURCE_UNKNOWN` and protects nothing. A missing pool is
`NO_EVIDENCE_DECLARED`. A missing tokenizer is `UNAVAILABLE_NO_TOKENIZER`. Each is reported in
telemetry, never silently zero.

## 5. Recurrent layers

Gated-DeltaNet layers hold recurrent state, not token-addressable KV. They are exempt. They are
not planned or compacted, and telemetry lists them under `recurrent_layers_exempt` with
`RECURRENT_STATE_NOT_TOKEN_ADDRESSABLE_KV`. Only full-attention layers (3, 7, ... in the
pinned config) are planned.

## 6. Arms

| Arm | Signals | Notes |
|---|---|---|
| `starlight_s1` | DA + grounding | STARLIGHT Stage 1 on the non-speculative base |
| `starlight_s1_no_da` | grounding only | H2O candidates + grounding |
| `starlight_s1_no_grounding` | DA only | H2O candidates + DA |

The seven existing arm IDs are unchanged. `SUPPORTED_ARMS` now equals the seven plus these three
(asserted by a test). `h2o_base` is unchanged and has no `starlight` telemetry key.

## 7. Telemetry (`telemetry["starlight"]`)

`measurement_label`, `method_variant` (`starlight_stage1::<signals>`),
`method_variant_is_upstream_h2o: False`, `signals_enabled`, `da_signal`,
`grounding_signal_status`, `grounding_evidence_verdicts`, `grounding_protected_by_evidence`,
`candidate_basis`, `budget_total`, `budget_basis`, `layers`, `recurrent_layers_exempt`,
`steps`, `status_counts`, `infeasible_steps`, `evicted_total`, `retained_total`,
`max_retained_per_layer_step`, `h2o_candidates_vetoed`, `conflicts_resolved_by_protection`,
`conflicts_by_source`, `protected_count_by_source` (entry x layer x step, summed),
`invariant_protected_evicted`, `invariant_evicted_not_candidate`,
`text_side_retained_fraction_last_step`, `limitation`.

`h2o_logical_selection_accounting` is set to `NOT_APPLICABLE_STARLIGHT_ARM` for these arms, so
an H2O variant label is never attached to a STARLIGHT run.

## 8. Deviations from the brief and from H2O

1. **Budget basis.** STARLIGHT uses a total budget with protection reserved inside it. H2O's
   R4-1 semantics (protected outside the budget) are kept unchanged for `h2o_base`. Reason: the
   contract requires `retained <= budget` unless INFEASIBLE, which the outside-budget basis does
   not satisfy under vetoes.
2. **Candidate set at reduced budget.** H2O's keep choice is computed at `B - |protected|`.
   This is the mechanism that keeps the budget invariant while honouring "H2O candidates, minus
   protection".
3. **No H2O-internal protection under STARLIGHT.** Sink and visual protection are not applied by
   H2O. The controller owns protection.
4. **DA question span is GLOBAL** (inherited deviation DA-2 from the scripted geometry).
5. **Scripted DA only.** DA protection uses the scripted diagnostic scopes, not a model-declared
   `<focus>` protocol.

## 9. Invariants and how they are checked

| Invariant | Check |
|---|---|
| protected never evicted | pure policy tests (random, consistent plans); engine `invariant_protected_evicted == 0` |
| evicted ⊆ H2O candidates | pure policy tests; engine `invariant_evicted_not_candidate == 0` |
| retained ≤ budget unless INFEASIBLE | policy raises on inconsistency; engine `max_retained_per_layer_step <= budget` test |
| ablations differ only by the removed signal | pure ablation test (exact evicted sets); engine no_da has zero DA counts |
| no-eviction oracle (budget ≥ length ⇒ tokens equal base) | engine test against the `base` arm |
| multi-request isolation | engine test: second request equals a fresh engine |
| multi-turn continuity | engine test: same request id twice equals a fresh engine |

## 10. Limitations

- CPU PyTorch reference only. Nothing here is vLLM, GPU, or a speed or VRAM result.
- Keyword matching is token-level on the tokenizer's ids, so a word split across tokens in an
  unexpected way can be missed. ECRD matches on decoded words, so the two may differ.
- Image-region evidence (bounding boxes) is not mapped to visual token positions. Only text
  keyword spans protect in Stage 1.
- `EvidenceItem` has no timestamp. Expiry is request-scoped.
- Each `execute_request` is an independent request with a fresh tracker and position state. No KV
  is carried across calls, which matches the existing engine. "Multi-turn continuity" therefore
  means a repeated call equals a fresh engine, not KV reuse.
- The scripted DA sink (first `min(16, L/4)` positions) can exceed a small budget. Such configs
  are INFEASIBLE by design and are reported (`infeasible_steps`), not hidden.
- Protection costs budget, so STARLIGHT evicts more text tokens than H2O at the same ratio. This
  is a retention cost, not a quality result. Quality is not measured here.
- No claim is made about answer quality, latency, or memory savings.

## 11. Open questions for the owner

1. Budget basis: keep H2O's outside-budget protection for STARLIGHT (violates the retained ≤ budget
   invariant under vetoes), or accept protection-inside-budget as implemented?
2. Should DA sink protection be capped (e.g., a fraction of the budget), given it can exceed it?
3. Image-region evidence: map bounding boxes to visual token positions in Stage 2, or keep it
   text-only?
4. Evidence expiry: add a timestamp or turn index to `EvidenceItem`?

## 12. Coverage against the auditor oracle (wF:p9, `docs/harness/STARLIGHT_INVARIANTS_ORACLE.md`)

The oracle is a specification, not evidence. This matrix states what the tests show. A "partial"
or "not covered" item is not a pass. Test file: `tests/test_starlight_oracle.py` unless noted.
Decision records are opt-in (`starlight_record_decisions=True`).

| Oracle item | Status | Where / note |
|---|---|---|
| I-1 eligibility formula | covered | `test_I1_I2_*`, recomputed from records |
| I-2 protected never evicted | covered | `test_I1_I2_*`; policy random tests; engine invariant counter |
| I-3 no silent discard, closed reason enum | covered | `test_I3_*`; `REASON_CODES` |
| I-4 infeasibility explicit | covered | pure protection-over-budget test; engine helper test |
| I-5 budget when feasible | covered | pure random consistent plans; engine `max_retained_per_layer_step` |
| I-6 GDN exempt and labelled | partial | exempt and labelled (`test_HC12_*`). The byte-identity hash of GDN state is NOT tested. |
| I-7 position preservation | partial | survivors keep original ids (`test_I7_*`). RoPE tensor values are NOT compared. |
| I-8 vision_start/end never evicted | NOT satisfied (open A-4) | measured as `structural_tokens_evicted` and marked `NOT_RATIFIED_A4_POLICY_DEPENDENT`. Not protected. |
| I-9 determinism | covered | `test_I9_HC15_*` |
| I-10 complete decision records | covered | `test_I10_*`; records are keyed by original position |
| A-DA | covered | `test_A_DA_*` on the conflict fixture |
| A-GROUND | covered | `test_A_GROUND_*` on the conflict fixture |
| A-NODA, A-NOGROUND | covered | `test_A_NODA_and_A_NOGROUND_*` on every step |
| A-H2O | covered | `test_A_H2O_*` |
| A-TOGGLE | covered | `test_A_TOGGLE_*`, with the H2O-only negative control |
| A-MASK | covered | `test_A_MASK_*` |
| HC-1 DA/H2O conflict | covered | helper conflict test and A-DA/A-TOGGLE. Oracle wording "H2O top-score" conflicts with its own candidate definition (candidates are low-score). Implemented as low-score candidate vs protection. |
| HC-2 grounding/H2O conflict | covered | helper conflict test and A-GROUND/A-TOGGLE |
| HC-3 protection > budget | covered | pure and helper tests (INFEASIBLE, protected kept) |
| HC-3b protection = budget | covered | `test_HC3b_*` |
| HC-4 expired evidence | covered | `test_HC4_*`: not protected, reason code `EXPIRED_EVIDENCE` |
| HC-4b stale evidence | partial | source-hash mismatch covered (pure). Turn IDs do not exist in this engine. A request-ID mismatch is reported as expired. |
| HC-5 irrelevant evidence measured | NOT covered | per-evidence protected counts are measured. Relevance labels are absent, so `protected_irrelevant_fraction` = `NOT_MEASURED_NO_RELEVANCE_LABELS`. |
| HC-6 long-context pressure | covered | per-step I-3 checks on the pressure run |
| HC-7 missing declaration | partial, open A-5 | explicit `NO_EVIDENCE_DECLARED`, protects nothing. NOT fail-closed. Owner decision. |
| HC-8 malformed declaration | covered for spans and sha | `test_HC8_*` raises. No score declarations exist, so NaN scores are not applicable. |
| HC-9 multi-request isolation | partial | sequential isolation test. Interleaved batches are not possible: `execute_request` is synchronous. |
| HC-10 multi-turn / evicted never returns | partial | `test_HC10_*` covers evicted-not-returning within a request. The engine carries no KV across calls, so cross-turn KV is not applicable. |
| HC-11 no eligible candidates | covered | `test_HC11_*`. Unmet budget with no candidates raises ValueError (open A-2). |
| HC-12 GDN exemption | partial | label and exemption covered. Hash identity NOT tested. |
| HC-13 vision span integrity | partial, open A-4 | structural tokens measured; not protected |
| HC-14 cache consistency | partial | length counter checked every step (`test_HC14_*`). Logit recomputation NOT done. |
| HC-15 determinism | covered | `test_I9_HC15_*` |

Auditor ambiguities (A-1..A-6) and the choices made:

- A-1 budget granularity: per (request, FA layer), applied to both KV heads (oracle default).
- A-2 budget unmet with no eligible candidates: NOT a status. The policy raises ValueError. Open.
- A-3 per-head protection: a protected position is protected in all KV heads of the layer
  (conservative oracle default). Positions are shared across heads.
- A-4 structural tokens: not ratified. Measured, not protected. Open.
- A-5 missing declarations: explicit status, no protection. Not fail-closed. Open.
- A-6 eviction timing: decode only. Prefill is never evicted. Each record states `phase: "decode"`.

Tests added after the implementation, not red-first (coverage of existing behaviour):
`test_I7_*`, `test_A_MASK_*`, `test_HC13` (measurement), and the grounding-verdict reason-code
change in `test_expired_evidence_*`.

## 13. ITER2 D4 changes (lead rulings, auditor REVIEW_ITER2 §2)

- **Structural source (Q2 / I-8).** `vision_start` and `vision_end` are protected by an explicit
  `structural` source, with reason codes and telemetry (`structural_policy` =
  `PROTECTED_LEAD_DECISION_Q2`, `structural_tokens_evicted`). Protection is an unratified lead
  decision. HC-13 asserts both tokens are protected and never evicted on a prompt that contains them.
- **Veto precedence.** A candidate protected by several sources gets one reason code by fixed
  precedence (DA, then grounding, then structural).
- **INFEASIBLE evicts nothing (Q3 / S-3).** When protected entries exceed the budget, or the budget
  stays unmet after eligible eviction, the step retains every entry. Status
  `INFEASIBLE_PROTECTED_EXCEEDS_BUDGET`, distinct decision `RETAIN_INFEASIBLE_NO_EVICTION` and reason
  `INFEASIBLE_NO_EVICTION`, and `kv_budget_violated` is reported per layer, per step, and in
  telemetry. The DA sink is not capped (Q6).
- **Ablation-independent candidate set (S-1).** All signal sources are computed on every arm. The
  effective H2O budget comes from the full protected union. An ablation only removes protection as a
  veto, so candidate sets are identical across arms (tested on the oracle fixture).
- **Grounding declaration (Q4).** Not fail-closed. A missing or unusable declaration sets
  `grounding_declaration_missing` and the arm reports `degraded_to_h2o_plus_da` (or
  `degraded_to_h2o_only` without DA). Q5 (auditor confirmation) is pending.
- **Protection-only evidence.** Unverified evidence protects nothing (S-6). Keyword matching is
  tested over bare, space-prefixed and capitalised forms, and under-protection fails the test (S-7).
- **Testable H-6 check (S-4).** The cache-length comparison has a test-only injection point; the
  mutation `if False:` fails the test.
- **Summary invariants (S-5).** Protected-evicted and evicted-not-candidate counters are tested on a
  hand-built violating plan.

## 14. Retained entries and matched retention (S-2, S-8)

- Every arm reports `retained_entries_per_layer` (mean, min, max, samples). It is measured as the
  physical full-attention cache length after each **decode** step; prefill is excluded because the
  prefill cache is unbounded. Memory fields state the scope: "per decode step, prefill cache
  unbounded".
- **Matched retention** calibrates H2O's budget to a STARLIGHT arm's measured mean retained count by
  measurement (`calibrate_h2o_ratio_to_retained`: bisection over the integer budget, with the real
  engine as the measure callback). A budget-to-retained formula was tried first and rejected:
  measured H2O retained count is not a fixed function of its budget (H2O keeps protected entries
  outside the budget). On the tiny fixture, STARLIGHT's mean is 29.0; the calibrated H2O budget is
  28, which gives a measured mean of 29.0.

## 15. Stage 2 (CPU reference, `cpu_pytorch_reference_not_vllm`, no speed claims)

**(a) Serving-style `mtp` arm.** On text-only requests of the official checkpoint, `mtp` now commits
tokens through the verified helper `_speculative_mtp_greedy_k1` (label
`cpu_greedy_k1_reference_not_vllm_mtp`). EOS is never emitted and stops generation, including an
accepted EOS draft. `proposed`, `accepted`, `rejected`, `verifier_calls` (one per target forward,
counted in the loop) and `committed` are measured. Image requests and scaffold engines keep the
diagnostic path (`speculation_executed` = False). Tests: committed stream equals non-speculative
greedy for 40 tokens with natural drafts, all-reject, and mixed drafts; verifier calls match an
instrumented model wrapper.

**(b) Policy-state rollback for `h2o_mtp` and `starlight_s1_mtp`** (new arm, signals da, grounding,
structural).
- *Eviction-free rule.* A 2-row verify block runs only when no full-attention layer would evict for
  either row (H2O: `unprotected_count(l, L+2) <= budget`; STARLIGHT: `L+2 <= budget_total`).
  Otherwise the pending token is fed alone and no draft is verified (`fallback_steps`). `proposed`
  counts only verified blocks.
- *Rollback.* A reject restores the DynamicCache and the policy state atomically: tracker scores,
  evicted masks, visual masks, budgets, `seq_len`, and the STARLIGHT original-position map. The
  pending token is then refed through the normal single-row path.
- *Digests.* `decision_digest` covers evicted masks, budgets and position maps. `policy_digest`
  adds scores, visual masks and `seq_len`. Both are sha256 over raw bytes.
- *Exactness.* On a forced reject, the state after refeed equals the normal (non-speculative) run's
  state at the same logical position, bit-for-bit (the same code path). Measured: all-reject runs
  reproduce tokens and policy digests at every common position, for H2O and STARLIGHT. For mixed
  accept/reject runs, retained decisions are identical; score values are not asserted to be
  bit-identical, because a 2-row attention block is numerically not identical to two single-row
  steps.
- *Found and fixed by the equivalence test.* The normal STARLIGHT path builds scripted DA spans
  (`sink_global`, focus segment) even on text-only prefills. The driver mirrors them.
- *Limits.* Text-only requests; no evidence pool (grounding not declared, `arm_status`
  `degraded_text_only_no_evidence`); `da_mtp` stays non-speculative (no 2-row DA mask path); image
  requests stay on the diagnostic path; the driver does not measure bytes freed (evictions happen).

**(c) Per-KV-head score selection** (upstream fidelity A4-02). `H2OConfig.per_kv_head_selection`,
default False (default path unchanged). `compute_eviction_plan_per_head` gives each KV head its own
heavy hitters with the same budget. The shared `compute_eviction_plan` refuses the flag rather than
silently using the layer mean. **Not wired into the engine**: per-head KV compaction needs a per-head
gather in the hooked attention, which is not implemented. Tracker level only.

## 16. Open items

- Auditor confirmation of Q5 is pending (lead decision: confirm, do not change).
- The DA sink is not capped (Q6): INFEASIBLE stays explicit.
- Per-head selection is not wired into the engine (see §15c).
