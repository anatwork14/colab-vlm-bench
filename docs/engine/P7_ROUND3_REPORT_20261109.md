# P7 Engine — Round 3 report (wF:p7 → wF:p5), 2026-11-09

HEAD at full-suite run: `c0aeb68` (branch `mission/20261109-engine-p7`, base `cfccc4b`). Worktree clean.
Environment: CPU only, `PYTHONPATH=src CUDA_VISIBLE_DEVICES=`, `colab-vlm-bench/.venv/bin/python -m pytest`.
No GPU, no vLLM, no pushes, no merges. Full audit: `docs/engine/P7_ENGINE_AUDIT_20261109.md`.

## Counts
- Baseline `cfccc4b`: 1 failed, 272 passed, 10 skipped.
- Round 2 end `1050cdc`: 306 passed, 10 skipped, 0 failed.
- **Round 3 final `c0aeb68` (FULL suite): 315 passed, 10 skipped, 0 failed.**
- The 10 skips are pre-existing notebook/stepwise tests marked retired.

## Items and status
| Item | Status | Commits (red → fix) |
|---|---|---|
| D1 DA masking | Done. Constructed official case: restricted reads > 0 and masked keys exactly 0 attention. Flag test kept as consistency check only. | `2953f6f` → `78d1d2f` |
| D2 H2O | Done. Protected-inside-budget kept. Limitation in telemetry. Unprotected original-H2O variant listed as owner decision, NOT implemented. | `9ccb944` → `68390d3` |
| R3-1 real-forward gate | Done. PASS only with official hooking; partial → `PARTIAL_COVERAGE_NOT_QUALIFIED`. Disclosed test assertion changes. | `7b19b87` → `f9445b2` |
| R3-2 CPU greedy k=1 MTP | **Done, not BLOCKED.** Label `cpu_greedy_k1_reference_not_vllm_mtp`. 6 official-checkpoint tests pass. | `3f7e9ad` → `d6e0b0f` → `7ed0fba` (test revision, disclosed) |
| Audit doc round 3 | Done. | `c0aeb68` |

## R3-2 details
- `RealSevenArmEngine._speculative_mtp_greedy_k1` (`src/vlmbench/infrastructure/real_engine.py`):
  - MTP head proposes one draft per step.
  - One target forward over `[pending, draft]` verifies it from a KV + Gated-DeltaNet snapshot.
  - Accept iff draft == target greedy argmax; on reject, restore the snapshot and re-feed `pending` alone.
  - The committed stream is always the target's greedy stream. No speed claim.
- `Qwen35NeuralModel.forward` gained `logits_to_keep`. The scaffold path accepts only 1.
- Tests in `tests/test_mtp_speculative_official.py`:
  - Natural MTP drafts: token-identical to greedy over 16 tokens.
  - All drafts rejected: token-identical; accepted = 0; proposed > 0.
  - Oracle accept/reject mix: token-identical; accepted > 0 and rejected > 0; proposed > 0.
  - Rejected verify: KV and DeltaNet tensors bit-identical after rollback. The test first asserts the verify changed the state.
  - Multi-call isolation: second turn matches a fresh greedy reference.
  - Serving-arm acceptance is `None`.

### Disclosed test changes
1. The red set used a prompt of random token ids, `range(100,140)`, with 0 natural accepts. It now uses a real text prompt via the official tokenizer.
2. The forced accept/reject test relied on natural MTP accepts, which depend on the data. It now uses oracle hooks: the target's own next token (accept) or a corrupted one (reject). The real verify and rollback code still runs.
3. Literal `mtp_acceptance_rate == 0.0` → `is None` in `tests/test_real_engine.py` (2) and `tests/test_mtp_da_upstream_contracts.py` (1).
4. The scaffold `test_p7_mtp_reference_path_is_token_identical_to_base` is narrowed to "serving arm commits target tokens only". It does not test speculative MTP.

### Measured numbers (one-off CPU diagnostic runs, NOT committed as assertions; correctness-path counts, not speed)
- Text prompt, natural MTP drafts: 3/12 accepted over 16 tokens; 8/23 over 32 tokens. Output equals greedy.
- Teacher-forced MTP match on the same text: 9/23, identical under three position conventions. The draft layer attends only to its own token, so position ids do not change the draft.

### Not done / limits
- The serving-style `mtp` arm in `execute_request` runs NO verifier. Its acceptance is `None` and its gate stays `GATED_NATIVE_MTP_VERIFIER_REQUIRED`.
- Native vLLM MTP is not verified and not claimed.
- The CPU reference is not wired into serving-arm results. This was a scope choice; say if you want it wired.
- Nothing here is GPU or vLLM evidence.

## Owner decisions still open (not implemented)
1. Unprotected original-H2O variant (protected tokens outside the budget). Telemetry marks it `NOT_IMPLEMENTED_OWNER_DECISION`.
2. DA question span tagged GLOBAL: a documented deviation from the scripted geometry. Prefill remains unmasked.
3. ECRD alpha: 3.5 (official+image) vs 20.0 (otherwise). Uncalibrated and not comparable across conditions.
4. `unsupported_penalty` is inert. Implement a detector or remove the parameter.
5. Unpinned legacy defaults `pad_token_id=151643` and `mtp_num_speculative_tokens=2`, absent from `config.json`. Need an owner value.

## Gate status (this engine, CPU)
- `POLICY_HARNESS_PASS`: as before.
- `REAL_MODEL_FORWARD_PASS`: PASS only on the official hooked checkpoint. Partial → `PARTIAL_COVERAGE_NOT_QUALIFIED`.
- `MTP_CORRECTNESS_PASS`: still `GATED_NATIVE_MTP_VERIFIER_REQUIRED`. The CPU reference shows token equivalence, but that is not the native gate.
- `DA_MASKING_EXERCISED`: PASS only with restricted decode reads (the constructed case).
- `CCU_ISOLATION_PASS`: `GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED`.
- Gates are independent. A passing lower gate does not imply a higher one.
