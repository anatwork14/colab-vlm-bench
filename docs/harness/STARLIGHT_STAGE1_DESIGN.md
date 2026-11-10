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
- A candidate set that would leave more than `budget` entries retained is reported as
  `INFEASIBLE_UNMET_AFTER_ELIGIBLE`, never silently. Caller errors (candidates supplied when nothing
  needs eviction) raise `ValueError`.

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
- Protection is computed per layer for the current physical cache, including the new token. Protected
  original positions are layer-independent, so the protected slots agree across full-attention layers.
- If `length <= B`, there are no candidates and nothing is evicted. This is the no-eviction case.
- Otherwise, the H2O keep set is computed at `B_eff = B - |protected|` over ALL entries with no
  H2O-internal protection. Candidates are its complement.
- **ITER3 R2 — LABELLED DEVIATION FROM PER-LAYER H2O (lead ruling F-1).** The keep set is ONE set shared by
  all full-attention layers: `H2OCacheTracker.keep_mask_shared` ranks positions by the MEAN over full-attention
  layers of each layer's head-mean score. It is not per-layer H2O (`h2o_base`, `h2o_mtp` keep per-layer
  selection and are unchanged). Reason: with per-layer candidates, one layer could veto an entry another layer
  evicted, giving different retained lengths per layer and a crash (BACKEND_INTERFACE.md, F-1). With one shared
  retained set, the mask and KV lengths cannot desynchronise. The engine raises if full-attention lengths or
  protected slots differ across layers. Ablations use the same rule, so they stay matched to each other.
  Consequence: STARLIGHT decisions differ from per-layer H2O decisions on the same prompt, and the comparison
  between STARLIGHT and H2O arms measures this shared-set rule as well as protection.
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
- **Coverage on image prompts (wF:p9 F-1; ITER3 closure, scope unchanged).** The scripted FOCUS span
  `focus:scripted_visual_segment` covers the WHOLE visual segment (`focus_end = sink_end + n_visual`,
  `real_engine.py`, the prefill DA block). On an image prompt the DA protection is therefore every visual token, not
  only the evidence region. Protection at prefill is at least `n_visual`, so whenever the visual token count reaches the
  budget, every step is INFEASIBLE and nothing is evicted. The review's fixture run (640x480, 300 visual tokens) measured
  321 of 322 entries protected with no evidence declared, against a budget of 192. Image `starlight_s1` is therefore a
  FULL-CACHE run, not a sparse one, and it is not comparable to a matched-budget run. The run is labelled
  `comparison_validity = STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE` (§19). Narrowing the FOCUS scope is an owner
  research decision and is not made here.
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
- **Keyword breadth policy (ITER3 R2, C-4).** Stop-word-like keywords (`KEYWORD_STOPWORDS`, e.g.
  `the`) are rejected before any position is protected. A keyword whose protected positions exceed
  `MAX_KEYWORD_PROMPT_FRACTION = 0.2` of the prompt is rejected. Both rejections are reported in
  `grounding_keyword_policy` (`accepted_keywords`, `rejected_stopword`, `rejected_breadth` with the fraction).
- **Union cap (ITER3 R3, N-R2-2, lead ruling).** The protected union of accepted keywords is bounded by
  `MAX_UNION_PROMPT_FRACTION = 0.5` of prompt positions (owner-ratifiable). Keywords are processed in declared
  order. A keyword whose addition would push the union over the cap is rejected and listed in `rejected_union`
  (with the union fraction it would have produced); later keywords are still considered. `union_fraction` is
  reported. Accepted protection that still exceeds the budget keeps the INFEASIBLE semantics of section 3 (never silent).

### 4.3 Evidence validity

`evidence_validity` returns `(status, reason)` per evidence id:

| Status | Reason | Protects |
|---|---|---|
| `unverified` | `caller_declared_verified_false` (caller flag; never checked by the controller) | no |
| `expired` | `pool_request_id_mismatch` (pool built for another request) | no |
| `stale` | `source_sha256_mismatch` (image hash differs from the declared source) | no |
| `valid` | `source_sha256_match` | yes |
| `declared_unverified` | `no_source_bytes_checked` (text-only: there are no source bytes to compare) | no (default) |
| `declared_unverified` | `accepted_by_policy_accept_declared` (text-only, explicit `EvidencePolicy(accept_declared=True)`) | yes, reported as such |
| `uncertain` | `source_bytes_not_hashed` (image present, bytes unavailable) | no |

Before ITER3 Round A the text-only row read `valid / source_unchecked_text_only`. That label let an unchecked item
read as VALID, so it was changed (see §17, S1).

`EvidenceItem` carries no timestamp, so "expired" means request-scoped, not time-scoped.

**S1 (ITER3 Round A): tri-state verification dimensions.** The caller's `verified` flag is no longer a protection
switch. `evidence_verification.assess_evidence` gives each item separate dimensions, each `VERIFIED`,
`NOT_VERIFIED`, `UNCERTAIN` or `NOT_ASSESSED`: `source_identity` (bound to the request), `source_bytes_hash` (image
bytes against the declared hash), `location_valid`, `relevance_to_question`, `claim_support`, plus `method` and
`provenance`. The integrity dimensions are never reported as semantic verification. The controller never assesses
relevance or claim support, so those are `NOT_ASSESSED`.

Protection (`decide_protection`), in order: caller `verified=False` is refused; identity not bound is refused; a bytes
mismatch is refused; bytes `UNCERTAIN` protect nothing; bytes `VERIFIED` protect. Otherwise (text-only, bytes
`NOT_ASSESSED`) the item is `DECLARED_UNVERIFIED` and does not protect by default.

**Fallback for incomplete evidence (A-5).** Missing or unassessed evidence is never fabricated. DA and structural
signals still operate. A text-only declared item protects only under an explicit, reported, ablatable
`EvidencePolicy(accept_declared=True)`, and the policy is in telemetry as `evidence_policy`. Rationale: a caller
Boolean with no source check should not earn hard protection, because hard protection can make a step INFEASIBLE
and keep the cache above budget (§3).

If grounding is enabled and the request has an image but no image bytes or path, grounding is
`NOT_PROTECTED_IMAGE_SOURCE_UNKNOWN` and protects nothing. A missing pool is
`NO_EVIDENCE_DECLARED`. A missing tokenizer is `UNAVAILABLE_NO_TOKENIZER`. Each is reported in
telemetry, never silently zero.

### 4.4 Visual evidence (`image_region`, ITER3 Round B S2)

Visual protection is a separate channel from keyword grounding. It protects visual token positions only when all of
the following hold for an item (see §18):

- S1 protects the item: the source bytes are VERIFIED and bound to this request (`decide_protection`).
- `location_valid` is VERIFIED: the box is well formed, with `0 <= ymin < ymax <= 1` and `0 <= xmin < xmax <= 1`.
- The mapping is SUPPORTED for the request: the image is present and hashable, the processor geometry is established
  (`grid_thw` agrees with the smart-resize arithmetic), and the prompt holds exactly one image token run of the right
  length.

The mapping is `visual_mapping.py`. A box covers every merged 2x2 cell it overlaps with positive area, so it is a
superset of the box. Partially covered cells are counted (`partial_cells`). Positions are ORIGINAL prompt positions,
so they follow each token through compaction. A region that cannot be mapped is UNSUPPORTED with a reason, never a guess.

**Preserving a visual span does not prove the model used it.** Telemetry records that the cache kept those positions.
It does not show that any answer depends on them. The status says nothing about grounding or quality.

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
| `h2o_shared_base` | none (zero signals) | H2O confounder control (ITER3 Round A): the same shared selection with NO internal protection |

The seven existing arm IDs are unchanged. `SUPPORTED_ARMS` now holds twelve arm IDs: the seven originals, the four
STARLIGHT arms (`starlight_s1`, its two ablations, and `starlight_s1_mtp`) and the control `h2o_shared_base` (asserted
by tests). `h2o_base` is unchanged and has no `starlight` telemetry key.

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
- Image-region evidence protects visual tokens only under the conditions in §4.4 (S1 verified, location valid,
  mapping SUPPORTED). Otherwise the status is UNSUPPORTED with a reason, and no visual token is protected. Preserving a
  visual span does not prove the model used it.
- Visual protection is not counted in the keyword caps (§4.2), so the caps do not bound it.
- `EvidenceItem` has no timestamp. Expiry is request-scoped.
- Each `execute_request` is an independent request with a fresh tracker and position state. No KV
  is carried across calls, which matches the existing engine. "Multi-turn continuity" therefore
  means a repeated call equals a fresh engine, not KV reuse.
- The scripted DA sink (first `min(16, L/4)` positions) can exceed a small budget. Such configs
  are INFEASIBLE by design and are reported (`infeasible_steps`), not hidden.
- Protection costs budget, so STARLIGHT evicts more text tokens than H2O at the same ratio. This
  is a retention cost, not a quality result. Quality is not measured here.
- Image prompts under `starlight_s1` are full-cache runs when the visual segment reaches the budget (§4.1, F-1). Their
  retention numbers say nothing about sparsity. Read `telemetry["comparison_validity"]` before comparing any image
  STARLIGHT run to a matched-budget run.
- `comparison_validity` is set on the non-speculative execution path only. The two speculative text routes return
  before that block and do not carry the field (open item, §19).
- No claim is made about answer quality, latency, or memory savings.

## 11. Open questions for the owner

1. Budget basis: keep H2O's outside-budget protection for STARLIGHT (violates the retained ≤ budget
   invariant under vetoes), or accept protection-inside-budget as implemented?
2. Should DA sink protection be capped (e.g., a fraction of the budget), given it can exceed it?
3. Image-region evidence: map bounding boxes to visual token positions in Stage 2, or keep it
   text-only? (ITER3 Round B implemented the mapping and the gate in §4.4. Whether the gate and the
   UNSUPPORTED cases are the right policy is still the owner's decision. Not ratified.)
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
| I-8 vision_start/end never evicted | protected through the structural source (lead decision Q2, `PROTECTED_LEAD_DECISION_Q2`); owner ratification of A-4 open | measured as `structural_tokens_evicted` and marked `NOT_RATIFIED_A4_POLICY_DEPENDENT`. |
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
| HC-11 no eligible candidates | covered | `test_HC11_*`. Unmet budget after all eligible candidates returns `INFEASIBLE_UNMET_AFTER_ELIGIBLE` (open A-2 ruling). |
| HC-12 GDN exemption | partial | label and exemption covered. Hash identity NOT tested. |
| HC-13 vision span integrity | partial, open A-4 | structural tokens protected (Q2) and measured; A-4 ratification open |
| HC-14 cache consistency | partial | length counter checked every step (`test_HC14_*`). Logit recomputation NOT done. |
| HC-15 determinism | covered | `test_I9_HC15_*` |

Auditor ambiguities (A-1..A-6) and the choices made:

- A-1 budget granularity: per (request, FA layer), applied to both KV heads (oracle default).
- A-2 budget unmet with no eligible candidates: a status, `STATUS_INFEASIBLE` with `INFEASIBLE_UNMET_AFTER_ELIGIBLE` (code behaviour, 133d5c5). Owner ruling still open.
- A-3 per-head protection: a protected position is protected in all KV heads of the layer
  (conservative oracle default). Positions are shared across heads.
- A-4 structural tokens: protected through the structural source (lead decision Q2) and measured. Owner ratification still open.
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
- *Limits.* Text-only requests. Evidence no longer changes the route (ITER3 Round B, §18): text-only keyword spans
  protect on the policy path under `accept_declared` only. `da_mtp` stays non-speculative (explicit status
  `NOT_SPECULATIVE_DA_MTP`, no 2-row DA mask path). Image requests are `UNSUPPORTED_IMAGE_PROMPT` on this path and stay
  on the diagnostic path. The driver does not measure bytes freed (evictions happen).

**(c) Per-KV-head score selection** (upstream fidelity A4-02). `H2OConfig.per_kv_head_selection`,
default False (default path unchanged). `compute_eviction_plan_per_head` gives each KV head its own
heavy hitters with the same budget. The shared `compute_eviction_plan` refuses the flag rather than
silently using the layer mean. **Not wired into the engine**: per-head KV compaction needs a per-head
gather in the hooked attention, which is not implemented. Tracker level only.

## 16. Open items

- Auditor confirmation of Q5 is pending (lead decision: confirm, do not change).
- The DA sink is not capped (Q6): INFEASIBLE stays explicit.
- Per-head selection is not wired into the engine (see §15c).

## 17. ITER3 Round A (S1, S3, S4, H2)

Provisional policy values (engineering hypotheses, not owner ratification). Caps and stopwords are provisional,
configurable, versioned and ablatable, and were not tuned on confirmatory answers.

- **S1 evidence dimensions.** See §4.3. Integrity (identity, bytes hash) is kept separate from semantic dimensions,
  which stay `NOT_ASSESSED`. Default `EvidencePolicy(accept_declared=False)`: text-only declared evidence protects
  nothing. Tests: `tests/test_i3a_s1_evidence_dimensions.py` (16). The expectations pinned to the old default were
  re-derived with an explicit `ACCEPT_DECLARED` opt-in and a stated reason; no assertion was relaxed. The only
  changed expectations are the two text-only status labels, which are now `declared_unverified`.
- **S3 protected never silently evicted; the fixed-budget verdict.** An INFEASIBLE step evicts nothing, so it keeps
  every entry (§3). The run is therefore NOT budget-bounded. `starlight.fixed_budget_verdict` reports a run as a
  successful fixed-budget run only when no step is INFEASIBLE, no layer-step is a KV budget violation, and max
  retained per layer-step does not exceed the budget. Telemetry: `starlight.fixed_budget_run`,
  `fixed_budget_status` (`NOT_A_SUCCESSFUL_FIXED_BUDGET_RUN` when not met), `fixed_budget_verdict`, and
  `kv_memory_accounting.fixed_budget_status`, `kv_retention_status`
  (`NOT_BUDGET_BOUNDED_ENTRIES_ABOVE_BUDGET_RETAINED`) and `max_retained_over_budget_per_layer_step`. Non-STARLIGHT
  arms report `NOT_APPLICABLE_NO_STARLIGHT_CONTROLLER`, not a claim. Tests: `tests/test_i3a_s3_protected_budget.py` (11).
- **S4 versioned caps and stopwords.** `starlight.GroundingCapPolicy` holds `cap_policy_version` (default
  `GROUNDING_CAP_V1_ITER3_R3`), the per-keyword and union fractions (`None` = uncapped), and the stopword set. A
  changed threshold or stopword list under the default id is refused. The report carries `cap_policy`,
  `stopword_set_sha256` and a `threshold_ledger` (per threshold: `limit`, retained, rejected, `omitted_positions`).
  `omitted_positions_total` and `preservation_claim` are set so that a capped occurrence is never reported as perfect
  preservation (`perfect_preservation_claimed` is False when any capped occurrence was omitted). The default
  reproduces the previous behaviour. Tests: `tests/test_i3a_s4_cap_policy.py` (14).
- **H2 confounder control `h2o_shared_base`.** Same shared mean-over-full-attention-layers selection as STARLIGHT,
  with zero signals (`STARLIGHT_ARM_SIGNALS["h2o_shared_base"] == ()`), so no internal protection. Comparable with
  `h2o_base`, `starlight_s1_no_da`, `starlight_s1_no_grounding` and `starlight_s1` on prompt, budget formula, layers,
  generation policy and model id. It reports actual retained counts, and an infeasible configuration for STARLIGHT
  shows up as `fixed_budget_run` False while the control stays within budget. Tests: `tests/test_i3a_h2_shared_base.py` (7).
- Not done in Round A: S2 visual mapping (Round B). The `adapters.py` placeholder `source_sha256 = "a"*64`
  (around line 487) is not changed here and is an open item.

## 18. ITER3 Round B (S2 visual mapping; Stage 2 combined methods)

Provisional policy values (engineering hypotheses, not owner ratification): the visual gate in §4.4, the superset rule
for boxes, and `accept_declared` for text-only evidence on the speculative path.

- **S2 mapping.** `visual_mapping.py` (§4.4). The geometry is the local Qwen3.5-0.8B processor's: patch 16, merge 2,
  factor 32, pixel bounds 65536 to 16777216. Tests check `smart_resize` and the grid against the real processor on fixtures
  (256x256, 300x200, 1000x37, 64x64, 640x480, 2000x2000). The token order (row-major over merged cells) is checked
  against the real `pixel_values`. A column-major mutant gives max abs error 1.76, against 6e-8 for row-major, so the
  check discriminates. Files: `tests/test_i3b_s2_visual_mapping.py`, `tests/test_i3b_s2_visual_protection.py`.
- **Visual status (replaces `NOT_IMPLEMENTED`).** Per request: `SUPPORTED` or `UNSUPPORTED` with a reason (§4.4). The
  C-5 expectations in `tests/test_r2_c5_visual_evidence.py` were re-derived with a stated reason: the old fixed label
  is replaced by a status that depends on the request. A pool with image evidence and no image is now
  `UNSUPPORTED / NO_IMAGE_IN_REQUEST`.
- **Engine (official checkpoint).** The source size is recorded in `vis_trace`. Geometry comes from the processor, and the
  single image token run comes from the prompt. Verified cells join the grounding protection by original position and
  survive compaction. `visual_protected_retained_final_by_layer` is measured at the end of each request. Heavy test:
  `tests/test_i3b_s2_visual_engine.py` (fixture `tests/fixtures/000000039769.png`, 640x480, grid 30x40, 300 tokens):
  80 merged cells protected for the top-left quadrant, each retained in every full-attention layer.
- **Stage 2 route.** `speculative_route.py` gives an explicit status per arm and request (§2 of BACKEND_INTERFACE.md).
  Evidence no longer changes the route. For `h2o_mtp` and `starlight_s1_mtp` on text-only requests, the grounding
  protection enters the policy state, and it is retained under speculation. Text-only evidence protects only under
  `accept_declared`, because it has no source bytes to verify. An image prompt on a combined or serving MTP arm is
  `UNSUPPORTED_IMAGE_PROMPT`. `da_mtp` is `NOT_SPECULATIVE_DA_MTP`. The MTP text paths are CPU verifier references, not
  native vLLM MTP.
- **Rollback and invariants.** `tests/test_i3b_s2_combined_speculation.py` checks: a rejection after an accepted block
  restores the cache, the policy state, and the original-position map; protected original positions stay retained at every
  speculative step; requests are isolated. The rollback is also checked by a mutant that skips the original-position
  restore. The engine then raises a bookkeeping error (`position bookkeeping out of sync`), so a missing restore fails closed.
- **Scope.** A preserved visual or text span does NOT prove the model used it. The telemetry shows what the cache kept.
  Nothing here is a quality, speed, or vLLM result. Stage 2 numerics are a CPU reference. Accepted 2-row blocks are not
  asserted bit-identical to single-row steps (§15b).
- **Open items.** Owner ratification of the §4.4 gate and the `accept_declared` default. Image prompts on the speculative
  path (UNSUPPORTED by design, not implemented). Multi-image requests (the engine takes one image per request, and the
  mapping refuses multi-image). Native vLLM MTP (BLOCKED). The `adapters.py` placeholder from Round A (open).

## 19. ITER3 closure repair (wF:p7, responding to the wF:p9 Round-B review)

Source: `docs/harness/ITER3_ROUNDB_REVIEW.md`. Each item is test-first (a red commit, then a fix commit). Heavy checks run
under `wt/heavy.lock`. `cpu_pytorch_reference_not_vllm`. No speed or quality claim.

- **F-1 (HIGH; owner decision; scope NOT changed).** The DA FOCUS scope on image prompts covers the whole visual segment
  (§4.1). The engine now surfaces `telemetry["comparison_validity"]` on the non-speculative path, computed by
  `starlight.comparison_validity`. Status `STARLIGHT_FULL_CACHE_INFEASIBLE_NOT_COMPARABLE` when `evicted_total == 0` and
  `protected_at_prefill >= budget_total`. `protected_at_prefill` counts the distinct prompt positions protected at prefill:
  DA GLOBAL and FOCUS spans, grounding keys, and structural positions. Other statuses: `FIXED_BUDGET_RUN_MET` (comparable),
  `NOT_A_SUCCESSFUL_FIXED_BUDGET_RUN`, and `NOT_APPLICABLE_NO_STARLIGHT_CONTROLLER`. Tests: `tests/test_i3c_p7_repair.py`
  (light, parametrised) and `tests/test_i3c_p7_repair_heavy.py::test_f1_full_cache_infeasible_image_run_is_labelled_not_comparable`
  (heavy). The review's strict xfail `test_starlight_s1_on_an_image_prompt_evicts_when_the_budget_is_feasible` stays strict:
  it marks the owner decision that is still open.
- **F-6 (independent retention checks).** `visual_protected_retained_final_by_layer` and
  `grounding_protected_retained_final_by_layer` are checked against the planner's own returned slot map: the
  `_starlight_build_step_plans` output for the visual channel, and the policy `next_orig` for the speculative grounding
  channel. Neither check reads the telemetry expression. Each asserts an eviction precondition, so "retained" cannot be
  vacuous. `test_i3b_s2_visual_engine.py::test_protected_visual_positions_are_retained_in_every_full_attention_layer_at_the_end`
  now runs on `starlight_s1_no_da`, because `starlight_s1` never evicts on images (F-1).
  - **Red evidence (mutant, scratch copy only).** Two mutants were built from the pristine tree. Visual: the visual merge is
    removed from planner protection and the telemetry is replaced by the protected count. The independent check fails
    (telemetry 80 against independent 64 on a layer). Speculative: the policy planner ignores grounding protection and the
    telemetry is replaced by the protected count. The independent check fails (telemetry 1 against independent 0).
    Pristine code passes both. The telemetry-only retention assertion still passes on the visual mutant, which is why the
    independent check was added.
- **F-7 (sub-tolerance box).** `visual_mapping._cell_interval`: an endpoint within `_EPS` of a cell boundary snaps to it,
  unless the snap would collapse a positive-width interval. In that case the raw interval is kept, so the positive-overlap
  superset holds. Before the fix, a 1e-10 px box straddling a boundary gave `r0 > r1` and mapped to no cell. Test:
  `test_sub_tolerance_box_straddling_a_cell_boundary_keeps_its_positive_overlap` (strict xfail removed; body unchanged).
- **F-8 (speculation ablation).** `_mtp_text_speculation_route` requires `_speculation_enabled`, as the policy route does.
  Test: `test_speculation_ablation_is_honoured_on_the_serving_mtp_arm` (strict xfail removed; body unchanged).
- **F-9 (partial cells).** Each visual evidence item reports `cells = {"full": n, "partial": m}` when it protects, and
  `None` when refused. The counts reach telemetry through `visual_evidence_items`. Tests: `test_f9_*` in
  `tests/test_i3c_p7_repair.py`. The docstring claim in `visual_mapping.py` is now true.
- **F-10 (route label).** The serving mtp text path carries `speculative_route` (`MTP_VERIFIER_TEXT`). Test:
  `test_f10_the_serving_mtp_text_path_carries_the_speculative_route_label` (heavy).
- **F-11 (labels).** A control arm with no signal is `NOT_APPLICABLE_CONTROL_ARM_NO_GROUNDING_SIGNAL` (grounding and visual)
  or `NOT_APPLICABLE_CONTROL_ARM_NO_DA_SIGNAL` (DA). An ablation with signals that drops one is `DISABLED_BY_ABLATION`.
  `DISABLED_ABLATION` is removed. Tests: `test_f11_*` in `tests/test_i3c_p7_repair.py`.

**Open after this repair.** (1) F-1 owner decision on the FOCUS scope. (2) `comparison_validity` is not carried by the two
speculative text routes (they return before the non-speculative telemetry block). (3) F-12 (INFO, equivalent mutant) is not
changed.
