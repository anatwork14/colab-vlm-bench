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

## Round 3 (lead directives D1, D2, R3-1, R3-2)

Test counts for round 3 are reported in the lead message at the final HEAD, not repeated here.

### D1. DA masking: flag-consistency test kept, constructed official case added
- The round-2 relaxation (accepting a readable question with zero restricted reads) is kept as a flag-consistency test only. It is not evidence that masking works.
- Constructed official-checkpoint case (`2953f6f` red, `78d1d2f` fix): 14 decode steps with a DA mask that restricts real reads. The test spies `engine.model.forward` and asserts (a) the restricted-read ratio is > 0 and (b) every masked key has attention weight exactly 0.
- Gate: `DA_MASKING_EXERCISED` = `PASS` only when restricted decode reads occurred; otherwise `NOT_EXERCISED_NO_RESTRICTED_READS`; `NOT_APPLICABLE` for non-DA arms.
- Limit: DA is decode-only and scripted. Prefill is unmasked (see Qualification boundaries).

### D2. H2O: protected-inside-budget kept; limitation recorded; alternative needs owner approval
- Semantics kept: sinks and visual tokens sit inside the fixed budget B. This is a deliberate protocol choice, not an accident.
- Telemetry (`9ccb944` red, `68390d3` fix): `h2o_logical_selection_accounting` now carries `protected_vision_tokens_in_prompt`, `multimodal_kv_compression_evidence`, and a `limitation` string stating that on VLM prompts the protected visual block dominates the budget, so KV compression is near zero and not evidence of memory savings.
- **Protocol revisions needing owner approval (NOT implemented):**
  1. Unprotected original-H2O variant: protected tokens outside the budget, so compression acts on the full cache. Telemetry field `unprotected_original_h2o_variant = "NOT_IMPLEMENTED_OWNER_DECISION"`.
  2. DA question span tagged GLOBAL (documented deviation from the scripted geometry; affects which reads are restricted).
  3. ECRD alpha: 3.5 for official+image, 20.0 otherwise. Uncalibrated, not comparable across conditions. Calibrate or fix one value.
  4. `unsupported_penalty` is inert. Implement an unsupported-token detector or delete the parameter.
  5. Unpinned legacy defaults `pad_token_id=151643` and `mtp_num_speculative_tokens=2`: not in `config.json`; need an owner value.

### R3-1. Real-forward gate fix (`7b19b87` red, `f9445b2` fix)
- `REAL_MODEL_FORWARD_PASS` is `PASS` only when `is_official_hf_hooked` is true.
- A partial checkpoint (coverage below full) emits `PARTIAL_COVERAGE_NOT_QUALIFIED`, not PASS.
- **Disclosed test changes:** earlier partial-load assertions expected PASS. They now expect `PARTIAL_COVERAGE_NOT_QUALIFIED`. A low-coverage fixture (unmapped 9600x64 visual tensor, coverage < 0.15) is added.

### R3-2. MTP: real CPU greedy k=1 speculative reference on the official checkpoint (label `cpu_greedy_k1_reference_not_vllm_mtp`)
- Implementation (`d6e0b0f`): `RealSevenArmEngine._speculative_mtp_greedy_k1`. The official MTP head proposes one draft per step. One target forward over `[pending, draft]` verifies it from a KV plus Gated-DeltaNet snapshot. Accept iff draft == target greedy argmax. On reject the snapshot is restored and `pending` is re-fed alone. The committed stream is always the target's greedy stream. `Qwen35NeuralModel.forward` gained `logits_to_keep` (scaffold path accepts only 1).
- Tests (`3f7e9ad` red, `7ed0fba` revised): `tests/test_mtp_speculative_official.py`, 6 tests on the official 0.8B checkpoint.
  - Natural MTP drafts: token-identical to greedy over 16 tokens.
  - Every draft rejected: token-identical, accepted = 0, proposed > 0.
  - Oracle accept/reject mix: token-identical, accepted > 0 and rejected > 0, proposed > 0.
  - Rejected verify rolled back: KV and DeltaNet tensors bit-identical to the snapshot. The test asserts the verify changed the state first.
  - Multi-call isolation: second turn matches a fresh greedy reference.
  - Serving-arm `mtp_acceptance_rate` is `None`.
- **Disclosed revisions to the red set (`7ed0fba`):** the original prompt was `range(100,140)`, random token ids with 0 natural accepts. It is now a real text prompt via the official tokenizer. The forced accept/reject test relied on natural MTP accepts, which is data-dependent, so it now uses oracle hooks. Both changes are disclosed in that commit message.
- **Measured (CPU, text prompt, not a benchmark):** natural MTP drafts: 3/12 accepted over 16 tokens; 8/23 accepted over 32 tokens. Token output equals greedy. These are correctness-path counts, not speed.
- **Acceptance is now `None` when unverified** (`d6e0b0f`): `mtp_acc_rate = 0.0` in `execute_request` is replaced by `None`. Literal `== 0.0` assertions in `tests/test_real_engine.py` (2) and `tests/test_mtp_da_upstream_contracts.py` (1) are changed to `is None` (disclosed). The scaffold `test_p7_mtp_reference_path_is_token_identical_to_base` is narrowed: it checks only that the serving arm commits target tokens.
- **Not done:** the serving-style `mtp` arm (`execute_request`) does not use the verifier; its acceptance is `None` and its gate remains `GATED_NATIVE_MTP_VERIFIER_REQUIRED`. Native vLLM MTP is not verified and is not claimed.
- **MTP layer position note:** the draft layer attends only to its own token (no persistent MTP KV history), so RoPE position has no effect on the draft. In a diagnostic with teacher forcing, match counts were identical under three position conventions (9/23 each). The position convention is therefore not the binding constraint here.

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
- **MTP:** serving arm proposes only; acceptance is `None` (unverified). The CPU reference (`cpu_greedy_k1_reference_not_vllm_mtp`) measures verifier-derived acceptance on the official checkpoint but is not vLLM MTP and makes no speed claim. Native vLLM verifier still gated. The MTP layer attends only to its own token, so position ids do not affect the draft.
- **ECRD:** alpha is condition-dependent and uncalibrated (see 3). `unsupported_penalty` is inert.
- **Partial scaffold loader:** `mtp.fc` is mapped into `mtp_proj` (approximate semantics). Scaffold only.
- **Unverified:** whether same-`request_id` concurrent requests collide on the H2O tracker (`reset=True`). Not exercised.
- **Nothing here is GPU or vLLM evidence.**
