# H2O deviations in this engine

Scope: the `h2o_base` and `h2o_mtp` arms in `src/vlmbench/infrastructure/` (`h2o.py`,
`real_engine.py`). Every H2O result from this engine carries
`method_variant: h2o_visual_anchor_text_budget` and `method_variant_is_upstream_h2o: False`
(telemetry key `h2o_logical_selection_accounting`).

**Arm IDs are unchanged** (`base, h2o_base, da_base, ecrd_base, mtp, h2o_mtp, da_mtp`). The name
`h2o_base` does not mean upstream H2O. The variant name above is what results must cite.

Nothing here establishes that this variant preserves accuracy at any KV budget.

## Deviation list

| ID | What this engine does | Deviation from upstream H2O | Effect on results |
|----|-----------------------|-----------------------------|-------------------|
| H2O-1 | Protected tokens (initial sinks and visual tokens) are never evicted and sit OUTSIDE the budget. | Upstream does not protect a visual block; its budget counts the retained cache. | Total retention on VLM prompts is near 1.0 by construction (the visual block is kept whole). Use the text-side fraction (`text_side_retained_fraction`) for any text-side statement. |
| H2O-2 | The ratio applies to the UNPROTECTED prefill population: `B = max(min_budget, int((L_prefill - protected_prefill) * ratio))`. | The ratio is over the whole cache population. | The same `budget_ratio` gives different total retention on different prompts. Example, L=340 with 300 visual tokens and ratio 0.5: B=32, total 336 (about 99%). On text-only L=340: about 50%. The two are not one method. |
| H2O-3 | `min_budget` floor (default 32, `h2o.py` `H2OConfig`) applies to the unprotected budget. | The floor's value is not sourced from any upstream constant or experiment here. | On short text it dominates the ratio. Example: 36 unprotected text tokens with ratio 0.5 give B=max(32, 18)=32, so 89% of text is kept before any decode eviction. The effective text policy is set by this constant. |
| H2O-4 | Sinks (first `protect_system_tokens`, default 4) are outside the budget. | Upstream has no separate sink allowance in the budget. | Small extra retention per layer (4 slots). |
| H2O-5 | Scores are AVERAGED over KV heads before selection (`layer.scores[:, :len].mean(axis=0)`). Query heads are averaged within each GQA group first. | Upstream selects heavy hitters per KV head. | Selection is coarser than upstream. This is a known difference, not yet changed (matrix A4-02). |
| H2O-6 | Budget is fixed from the PREFILL length per full-attention layer. Generated tokens are unprotected and compete with prompt text under B. | Upstream budget derivation is described differently in our matrix (A4-01). Not re-read from upstream source in this document. | With ratio 1.0 alone, generated tokens exceed the budget at the first decode step (test `test_r5_ratio_one_alone_still_evicts_generated_tokens`). A no-eviction setting needs `min_budget >= prompt + output`. |
| H2O-7 | Recurrent (Gated-DeltaNet) layers are never evicted; only the 6 full-attention layers are planned and compacted. | Upstream applies H2O to attention KV only. | Consistent with the hybrid architecture. Not a deviation in intent. |
| H2O-8 | Recent window = `min(B, max(1, int(B * recent_ratio)))` of the unprotected tokens, with `recent_ratio=0.2`. Ties broken toward later positions. | Upstream's recent-window sizing is not reproduced here. | Changes which text tokens survive at the margin. |

## Unprotected original-H2O variant: NOT IMPLEMENTED (owner decision)

The variant without protection, with the budget over ALL tokens (upstream-style), is not
implemented. Telemetry records `upstream_variant_status: NOT_IMPLEMENTED_OWNER_DECISION` and
`unprotected_original_h2o_variant: NOT_IMPLEMENTED_OWNER_DECISION`.

Until that variant is implemented and run, no H2O result in this repository answers
"does H2O preserve accuracy at 50% KV?"

## What is NOT established

- Any KV-memory or latency benefit. No physical VRAM saving is measured (`physical_vram_saving_claimed: False`).
- That H2O-style eviction preserves answers on any benchmark.
- Equal-KV comparability with `base`. Equal KV would need a budget that counts all tokens (the failure
  mode fixed in round 3) or a different visual budget, which is a different method.

## Confirmed by tests (no-eviction oracle, R5-1)

With the floor covering prompt plus output (`h2o_min_budget >= len(prompt) + output`), `h2o_base` is
token-identical to `base` on the official checkpoint, on a text prompt and on the default image prompt,
with `cumulative_evicted_slots == 0`. This shows the H2O plumbing is inert when nothing is evicted. It
does not validate the eviction policy.

## Owner decisions still open

1. Implement the unprotected upstream-style variant and run it (required before any accuracy-vs-KV claim).
2. Decide whether `min_budget=32` is kept, sourced, or replaced by a ratio-only budget.
3. Decide whether protection of visual tokens is part of the method being benchmarked.
4. Decide whether scores should be per KV head (upstream) or averaged (current).
