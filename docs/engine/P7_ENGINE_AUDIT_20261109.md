# P7 Engine audit — 2026-11-09 (branch mission/20261109-engine-p7)

Base `cfccc4b`. Round 1 and round 2 fixes are both on the branch. Real arm execution lives in
`src/vlmbench/infrastructure/` (`real_engine.py`, `h2o.py`, `declarative_attention.py`,
`grounding_evidence.py`, `model_config.py`, `adapters.py`). `src/vlmbench/engine.py` is only the A100
vLLM launch helper with no arm logic. All runs are CPU, `PYTHONPATH=src`, using the existing
`colab-vlm-bench/.venv/bin/python`. No GPU, no vLLM, no serving evidence.

## Test counts
- Baseline `cfccc4b`: 1 failed, 272 passed, 10 skipped.
- Round 1 end (`b68cc9f`): 281 passed, 10 skipped, 0 failed.
- Round 2 end (`1050cdc`): **306 passed, 10 skipped, 0 failed.**

## Round 2 corrections (lead audit wF:p9)

### 1. H2O budget collapse: CRITICAL, fixed
- Round 1 said "H2O zero-eviction fine". That claim covered only single-step behavior. **It was wrong for multi-step decode.**
- Cause: `compute_eviction_plan` recomputed `max(min_budget, int(current_length * ratio))` from the already-compacted cache. Measured retained counts for a 300-token text prefill: 150, 75, 38, 32, 32, ... With 200 protected visual tokens the plateau was 226, leaving about 22 text tokens.
- Fix (`668f0a0`): `initialize_budget(layer, prefill_len)` fixes B = max(min_budget, int(prefill*ratio)) once per full-attention layer after prefill. Evict only when length > B. Protected tokens (sinks + visual) are **inside** B. When they alone reach B, the plan keeps protected tokens plus the newest token and sets `protected_saturated=True`.
- After fix: text-only retention holds at 150 for 8 steps. With 200 visual tokens it holds at 205 (protected + newest).
- Official VLM prompt after fix (305 tokens, 302 vision-block tokens): H2O evicts 6 of 1824 slots across 6 layers and 8 steps. The protected visual block covers nearly the whole prompt, so the arm barely differs from Base in KV retention on this fixture. Qualification consequence: **H2O arms do little on this VLM prompt**. Not measured for quality.

### 2. MTP forced acceptance: confirmed, removed
- `adapters.py` contained `accepted_count = 1 if step % 2 == 0 else 0`. The round-1 guard scanned only `real_engine.py`.
- Fix (`0919eaa`): the synthetic block scores proposals inside the transaction, always rolls back, and reports `mtp_accepted_count=None` with `NOT_VERIFIED_SYNTHETIC_NO_ACCEPTANCE_MODEL`. `SevenArmEngine` and `InfrastructureRunner` are removed from the package `__all__`, and remain importable from `vlmbench.infrastructure.adapters` labeled SYNTHETIC.
- Guard test now scans `adapters.py`, checks exports, and checks every MTP trace step for absence of acceptance.

### 3. ECRD keyword substring matching: reproduced, fixed
- Repro: keyword `red` matched token text `reduced`, `shred`, `hundred`, `rede`. Keyword `target` matched token `tar`. Cause: `kw in clean or clean in kw` substring rule.
- Fix (`9cfacfc`): keyword words must appear as a contiguous word sequence in the token text (`[a-z0-9]+`). BPE markers and punctuation are stripped. The processor no longer boosts substring tokens. Whole-word tokens still get `alpha`.
- Alpha confounder (disclosed, NOT retuned): `reweight_alpha` = 3.5 when `official_hf_hooked and image_present`, else 20.0. No rationale is recorded in the code. ECRD results are **not comparable across those conditions**. Telemetry (`ecrd_reweight`) now says `calibrated=False`, `cross_condition_comparable=False`.
- `unsupported_penalty` (beta) is **inert**: the processor never applies it. Documented in the config and locked by a test that changing it changes no logits.

### 4. DA question readability: fixed, with a decision and a finding
- The scripted LOCAL window (8) hid question tokens from the newest query. Documented deviation: for image prompts the span after the last `vision_end` token is GLOBAL. The deviation is listed in telemetry as `deviations_from_scripted_geometry: ["question_span_global"]`. Text-only prompts have no question split, so no deviation applies there.
- Decision per lead: prefill masking unchanged. DA stays decode-only. Telemetry carries `result_label = decode_only_scripted_da_diagnostic`, `prefill_masked=False`, and `nontrivial_masking_exercised`. The adapter's DA stats carry `SYNTHETIC_PROMPT_MASK_STATS_NOT_DECODE_ONLY_DIAGNOSTIC`.
- **Finding:** with the question readable, the official image prompt in a 4-token decode measures `attention_mask_restriction_ratio = 0.0`. The active focus already makes the visual segment readable, and the only remaining restriction source is the LOCAL tail beyond 8 generated tokens. So at current geometry the scripted DA restricts nothing on short official decodes. The round-1 test demanded `> 0`, and that threshold depended on hiding the question. I replaced it with an assertion that the emitted `nontrivial_masking_exercised` flag agrees with the measured ratio. **This relaxes a test; the lead should decide whether the DA diagnostic is acceptable at this geometry.**

### 5. Mask misalignment after H2O compaction: CONFIRMED, fixed; no numeric effect here
- Probe (official 0.8B, CPU, image prompt, h2o_base): prefill mask 305 = KV 305. First decode step: mask **306** (built from the pre-compaction cache), plan retains 303, so the hook sliced `[..., :303]`, which picks the wrong columns. Later steps: mask 304 = KV 304.
- Numeric effect in this path: none observed. Decode masks are all zero, batch size 1, no padding. A mask with `-inf` at retained positions would be misapplied; that case does not occur here.
- Fix (`b52d42d`): mask columns are index-selected by the same retained positions as the KV cache. Any width other than the pre-compaction KV length raises. Layer telemetry reports `attention_mask_len` and `attention_mask_len_received`. Guard test requires full-attention layers to report the key and match their KV width at every decode step.

### 6. Config pinning: fixed (`1050cdc`)
- Local `config.json` sha256 `b90b86f3…` matches `starlight_local_manifest.json` (revision `2fc06364…`). Hard-coded `qwen35_08b()` disagreed with it: heads 16 vs 8, head_dim 64 vs 256, FFN 2816 vs 3584, EOS 151645 vs 248044. Corrected to the pinned values.
- `from_checkpoint()` requires keys (no silent defaults), reads layer layout from `layer_types`, and verifies the manifest sha → `PINNED_MANIFEST_CONFIG_SHA256`.
- `qwen35_4b()`: no local pin. Marked `UNPINNED_NO_LOCAL_CONFIG_JSON`. No values were invented.
- `load_model_config()`: exact-match registry or checkpoint directory only. Unknown IDs (including substring lookalikes like `Qwen3.5-14B`) raise `ValueError`. No fallback to 0.8B.
- Still unpinned legacy defaults (not in `config.json`): `pad_token_id=151643`, `mtp_num_speculative_tokens=2`.

## Round 1 findings (still applicable)
- Partial loader: oversized projections were silently sliced and counted as mapped. Now exact-shape only (`b68cc9f`).
- DA window: negative or non-numeric window crashed or made keys unreadable. Now a clear `ValueError` (`b68cc9f`).
- H2O MTP commit: `accepted > proposed` was accepted. Now refused (`b68cc9f`).
- MTP reference path: token-identical to Base under identical weights (regression test).
- CPU multi-request concurrency matches sequential runs (regression test; non-vLLM).

## Qualification boundaries (open)
- **DA:** decode-only, scripted, not the model-emitted `<da:focus>` protocol. Restricts nothing on short official decodes (see 4). Prefill is unmasked by decision.
- **H2O on VLM prompts:** protected visual tokens dominate the prompt, so the budget barely evicts. Bytes freed are per-step tensor deltas, not VRAM savings.
- **MTP:** proposal diagnostic only. No drafts committed. Acceptance 0.0 by design. Native verifier gated. The MTP layer attends only to its own position.
- **ECRD:** alpha is condition-dependent and uncalibrated (see 3). `unsupported_penalty` is inert.
- **Partial scaffold loader:** `mtp.fc` is mapped into `mtp_proj` (approximate semantics). Scaffold only.
- **Unverified:** whether same-`request_id` concurrent requests collide on the H2O tracker (`reset=True`). Not exercised.
- **Nothing here is GPU or vLLM evidence.**
