# DA (Declarative Attention) deviations in this engine

Scope: the `da_base` and `da_mtp` arms in `src/vlmbench/infrastructure/real_engine.py`
(`_da_scripted_tags`, `DA_DECODE_ONLY_LABEL`). Every DA result from this engine is labelled
`decode_only_scripted_da_diagnostic` and `original_paper_protocol_qualified: False`.
Nothing here is evidence for the original DA protocol.

## Deviation list

| ID | What this engine does | Deviation from the original DA protocol | Effect on results |
|----|-----------------------|------------------------------------------|-------------------|
| DA-1 | Read masks are SCRIPTED from prompt geometry (`_da_scripted_tags`). | The model is not asked to emit a `<da:focus magic_chunks=...>` declaration. No model-declared focus is used. | Restriction reflects this script, not the model's own focus. |
| DA-2 | On image prompts the span after the last `vision_end` token (the question/instruction) is tagged GLOBAL. | Not part of the scripted geometry. Makes the question readable by every decode query. | Fewer keys are restricted. Without it the LOCAL window hid question tokens from the newest query. |
| DA-3 | Prefill is NOT masked (`prefill_masked: False`). Masks apply to decode queries only. | The protocol restricts reads in the attention of the generated tokens. | Prefill attention is unrestricted in all arms. |
| DA-4 | Sink length = `min(16, max(1, anchor_len // 4))`. LOCAL window = 8 tokens. Visual focus = `n_visual` tokens (fallback `min(4, ...)` when no visual tokens are passed). | Constants chosen by this engine. Not fitted to any target sparsity and not taken from the paper. | Sets the size of the restricted region. Comparing restriction ratios across prompts only makes sense with these fixed constants. |
| DA-5 | Bounds for sink, focus and question are fixed from the PREFILL length (`anchor_len`). Only the LOCAL window moves with the generated suffix. | Chosen to keep prompt keys in the same tag class across steps. | None beyond DA-4. |
| DA-6 | The restriction ratio is computed from the constructed mask. `DA_MASKING_EXERCISED` PASS also requires masked weights measured in the hooked forward to be exactly 0.0 (R4-3). | Not a protocol deviation. It is a measurement rule. | PASS means masking was applied in the official hooked decode, on this scripted mask. |
| DA-7 | Text-only prompts have no question split, so DA-2 does not apply. | None. | None. |
| DA-8 | Masked entries are set to -1e9 before softmax. Exact 0.0 weights follow from float underflow. | Implementation choice. | The measured zero is a property of this mask implementation, not evidence about model behaviour. |

## What is NOT established

- That a model using scripted DA reads the same content as a model that declares its own focus.
- Any KV-read, memory, latency or accuracy benefit. DA here is a logical attention-mask restriction.
  `physical_kv_read_saving_claimed` and `hbm_traffic_reduction_claimed` are both `False`.
- Behaviour on the scaffold (unhooked) path. The gate cannot pass there (`NOT_MEASURED_NO_HOOKED_DECODE_STEP`).

## Owner decisions still open

1. Whether to replace DA-1 with a model-emitted focus protocol (needs a defined output format and a verifier).
2. Whether DA-2 should be kept as a documented deviation or dropped in favour of the scripted geometry alone.
3. Whether DA-4 constants should be calibrated, and against what target.
