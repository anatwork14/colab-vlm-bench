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

Tests: `tests/test_starlight_policy.py` (14), `tests/test_starlight_signals.py` (16),
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
