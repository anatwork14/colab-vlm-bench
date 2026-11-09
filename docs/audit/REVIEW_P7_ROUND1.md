# Adversarial review of wF:p7 round-1 commits

- Reviewer: wF:p9 (Scientific Auditor), round 2 for lead wF:p5
- Reviewed tree: `/mnt/ssd/genos2/research_workspace/wt/wf-engine-p7`, read-only (not edited). HEAD at review time `0919eaa`.
- Round-1 commits under review: `22372dd` (tests), `b68cc9f` (fixes), `77ec005` (audit doc `docs/engine/P7_ENGINE_AUDIT_20261109.md`).
- Later p7 commits also read where they change the verdict: `3dea7a1`, `668f0a0` (H2O red tests and fixed budget), `acd31e7` (synthetic adapter red test), `0919eaa` (forced acceptance removed, `SevenArmEngine` unexported).
- Method: read diffs; ran targeted tests on the `77ec005` export and on `0919eaa` with `CUDA_VISIBLE_DEVICES=` (CPU only) and `PYTHONPATH=<tree>/src`; ran old-code vs new-code tests to check fail-before/pass-after; ran a real-shape probe (`docs/audit/repro/p7_probe_real_shapes.py`, output `docs/audit/repro/P7_PROBE_OUTPUT.txt`).
- No GPU, no installs, no pushes, no merges, no edits to the p7 worktree.

## Summary

| # | Question | Verdict | Severity |
|---|----------|---------|----------|
| a | Exact-shape mapping | Shape guard is strict and fails closed, but the pass gate it feeds is not meaningful (10% coverage reports PASS) | HIGH |
| b | DA geometry anchored to prefill | Anchor is a deliberate design choice, but the test that claims stability excludes the LOCAL window it moves | MEDIUM |
| c | Commit refuses accepted > proposed | PASS. Refusal exists, fails before and passes after | — |
| d | MTP token-identity claim | VACUOUS as an MTP correctness test: both arms are plain decoding; acceptance is hard-coded 0.0 | HIGH |
| e | Concurrency test | WEAK: passes on old and new code, one arm only, no shared-state stress | MEDIUM |

Additional findings: H2O fixed budget still saturates on image prompts; `REAL_MODEL_FORWARD_PASS` is PASS at ~10% coverage; stale citation numbers in one repo file; uncommitted p7 changes not reviewed.

## Test evidence

- `77ec005` export, targeted tests: 60 passed.
- `0919eaa` head, `tests/test_infrastructure.py` and `tests/test_mtp_da_upstream_contracts.py`: 31 passed.
- Old-code vs new-code (fail-before/pass-after): the new tests that fail on old code include
  - `tests/test_real_engine.py:414` partial loader rejects oversized `q_proj`;
  - `tests/test_real_engine.py:436` DA geometry stable across steps;
  - `tests/test_infrastructure.py:471` H2O commit refuses accepted > proposed;
  - `tests/test_infrastructure.py:488` malformed DA window raises;
  - `tests/test_infrastructure.py:501` DA window 0 handled.
  My notes record six failing new tests on old code; I did not re-derive the sixth in this write-up.
- `tests/test_real_engine.py:457` (MTP identity) and `:159` (concurrency) pass on both old and new code.
- Full suite on the `77ec005` tree (CPU, `CUDA_VISIBLE_DEVICES=`, `PYTHONPATH=<export>/src`): **281 passed, 10 skipped** in 345 s. I checked the export's `real_engine.py` and `h2o.py` hashes against git: they match `77ec005` (and `b68cc9f` for these two files), not `0919eaa`.
- Full suite on head `0919eaa` has **not** been run. Head-only checks (31 passed, above) cover `0919eaa` for the infrastructure and upstream-contract tests only. The `0919eaa` removal of forced acceptance is therefore not covered by a full-suite run.

## (a) Exact-shape mapping

Code: `src/vlmbench/infrastructure/real_engine.py` exact-shape guards at `:626` (`mtp_proj`), `:634` (visual `fc2`), `:669` (`self_attn` rows/cols), `:681` (linear `qkv` must equal `(3*r_dim, hidden)`), `:691` (`out_proj`). Expected dimensions come from the engine config (`mtp_proj` built at `:530`).

What is right:
- The guards compare each tensor's shape to the configured shape and refuse mismatches. A wrong-shaped tensor is not silently loaded; `tests/test_real_engine.py:414` checks this.

What is wrong:
- The config the expected shapes come from is the hard-coded 0.8B config (`src/vlmbench/infrastructure/model_config.py:57-66`, silent fallback `:136-146`, per my matrix A1-03). Real-checkpoint shapes are therefore checked against a config the engine chose itself, not against the checkpoint's `config.json`.
- Real-shape probe (`docs/audit/repro/p7_probe_real_shapes.py`, output `docs/audit/repro/P7_PROBE_OUTPUT.txt`): 9 real-shaped Qwen3.5-0.8B tensors loaded into `RealSevenArmEngine(method_arm="base", config=Qwen35ModelConfig.qwen35_08b(), device="cpu")`. Result: `mapped_tensors_count: 2`, `unmapped_tensors_count: 7`, `parameter_coverage_ratio: 0.100044`, `is_pretrained: True`. The unmapped set includes `linear_attn.in_proj_qkv`, `linear_attn.out_proj`, the layer-3 `self_attn` q/k/v/o, and `visual.merger.linear_fc2`. The probe output does not record a per-tensor reason, so I have not confirmed which guard rejected each one.

Verdict: the shape guard is fail-closed on mismatch (good), but the engine cannot load the real 0.8B checkpoint through it at full coverage. See the REAL_MODEL_FORWARD_PASS finding below for why this is a correctness issue and not only a coverage issue.

## (b) DA geometry anchored to prefill

Code: `real_engine.py:1250` `da_anchor_len = len(full_prompt)  # DA geometry is anchored to the prefill, not the growing KV`; scripted tags built at `:1338` via `_da_scripted_tags` (`:861`).

- The anchor is a deliberate choice: scripted DA geometry is computed once from the prompt length, so LOCAL sink and open-tail windows do not drift as the KV grows during decoding. That is a defensible reading of the paper's scripted case, but it is a choice, not something the paper's FSM derives. The upstream FSM (16-token sink, one-based segments, LOCAL = sink + open tail; upstream contract in `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md`) applies to model-emitted tags.
- The stability test `tests/test_real_engine.py:436` records tags per step with `recorded.append([t for t in tags if t.tag_type != DATagType.LOCAL])` (line 9 of the test body). It therefore excludes the LOCAL window, which is exactly the part that depends on the anchor. The test cannot detect drift in LOCAL, so it does not verify the prefill-anchored geometry it is named for.

Verdict: anchor is intentional and documented; the test does not cover the anchored part. Needs a test that asserts LOCAL windows are identical across decode steps, and a note in the audit doc that scripted DA is prefill-anchored by design.

## (c) Commit refuses accepted > proposed

Code: `src/vlmbench/infrastructure/h2o.py:323-341`. Negative `accepted_tokens` raises at `:327-328`. `proposed = max(0, self.tracker.seq_len - self._snapshot["seq_len"])` at `:329`; `accepted_tokens > proposed` raises at `:330-332`.

- Test `tests/test_infrastructure.py:471` fails on old code and passes on the new code. The refusal is in place on head `0919eaa`.
- Caveat: the guard protects the H2O tracker only. The adapter's forced-acceptance path (`adapters.py`, `accepted_count = 1 if step % 2 == 0 else 0`, my matrix item A5) is removed in `0919eaa`; I confirmed the removal in the diff.

Verdict: PASS for this claim.

## (d) MTP token-identity claim is vacuous

Code: `tests/test_real_engine.py:457-474` (`test_p7_mtp_reference_path_is_token_identical_to_base`). Also `:142-155` (`test_real_mtp_speculative_verification_and_rollback`).

- The test asserts `committed_draft_tokens == 0` and `mtp_acceptance_rate == 0.0` on the MTP arm, then asserts MTP tokens equal Base tokens.
- `real_engine.py:1454` sets `mtp_acc_rate = 0.0` as a literal, so the rate is never measured. With no draft ever committed, the MTP arm is ordinary target decoding, and equality with Base holds by construction. The test passes on old code, which confirms it does not discriminate.
- It therefore tests "MTP disabled equals Base", not "MTP speculative verification is token-identical to target greedy decoding". The second claim is untested. The `:142` test has the same problem: it asserts the zeros it should be exercising.

Verdict: vacuous as evidence for MTP correctness. Before any MTP claim is made, the arm must propose drafts, the acceptance rate must be computed from the verifier, and the test must assert the committed sequence equals the greedy target sequence while proposals are nonzero. Until then the MTP status should be "not evidenced".

## (e) Concurrency test shared state

Code: `tests/test_real_engine.py:159` (`test_ccu_concurrency_isolation`) and the `h2o_base` concurrency test (`:474` area). The test runs four distinct prompts through one `RealSevenArmEngine` from a `ThreadPoolExecutor(max_workers=4)` and compares tokens with a sequential run.

What the code shares and how it is handled:
- `self.h2o_trackers` is a plain dict keyed by request ID (`real_engine.py:1000-1011`, `_get_or_create_h2o_tracker`). It has no lock. Distinct IDs avoid collisions, but the test never uses colliding or reused IDs.
- The KV cache is created per request (`real_engine.py:754`, `DynamicCache`), so cache contents are not shared.
- Model forward passes are serialized by `_hf_lock` (`:751`, `:1115`).
- `_HookStepContext` is `threading.local` (`:60`), so hook state is per thread.

Verdict: the test is a reasonable smoke test for the cases it covers, but it passes on old and new code and only exercises `h2o_base`, not the DA, MTP, or image-prompt paths. It is not a race test. Per its own docstring it is CPU PyTorch reference evidence, not vLLM serving evidence, which is labelled correctly. To count as concurrency evidence it needs reused request IDs, all arms, and a check that `h2o_trackers` does not leak between requests.

## Additional findings

1. **H2O fixed budget saturates on image prompts** (HIGH). After `668f0a0` the H2O budget is fixed. For an image prompt with 300 visual tokens and a budget of 170, text tokens kept is 0 from decode step 2 onward (simulation on the p7 head, `h2o_fixed_sim.py`, in the scratchpad; not committed). The fix in `668f0a0` stops the per-step compounding but does not stop visual tokens from exhausting the budget. Image-prompt H2O results from this tree should not be reported.

2. **`REAL_MODEL_FORWARD_PASS` is PASS at ~10% coverage** (HIGH). `real_engine.py:1218` sets `"REAL_MODEL_FORWARD_PASS": "PASS" if self.is_pretrained else "GATED_..."`, and `is_pretrained=True` is set at `:965` as soon as a checkpoint directory is loaded. The real-shape probe loaded 2 of 9 tensors (coverage 0.100044) and still reports `is_pretrained: True`. The gate should require full mapping, or report the coverage ratio and PASS only at 1.0.

3. **Stale citation numbers** (MEDIUM). Repo files with the old v1 numbers (52.0%/31.1%) or any stale numbers, searched over `*.md`, `*.py`, `*.yaml`, `*.tex`, `*.txt` in the p7 tree:
   - `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:27` quotes 52.0% (Gemma-4-31B) and 31.1% (Qwen-3.6-27B). This is the only stale citation in the p7 tree.
   - My own matrix (`docs/audit/SOURCE_ALIGNMENT_MATRIX.md`, item A3-08) quotes the v1 numbers on purpose, to document the discrepancy. It is not in the p7 tree.

4. **Uncommitted p7 changes not reviewed** (INFO). `src/vlmbench/infrastructure/grounding_evidence.py` and `tests/test_infrastructure.py` have uncommitted modifications in the p7 worktree. I did not review them; any ECRD claim must be checked against a commit, not these working-tree changes.

## Citation confirmation (DA numbers)

The DA paper's reported decoding-attention reductions changed between versions:

| Version | Source | Gemma-4-31B | Qwen-3.6-27B | Quote |
|---------|--------|-------------|--------------|-------|
| v1 | https://arxiv.org/abs/2609.02737v1 (abstract) | 52.0% | 31.1% | "significantly reduces total attended tokens during decoding (52.0%, 31.1%…)" |
| v1 | https://arxiv.org/html/2609.02737v1 (body) | 52.0% | 31.1% | "DA reduces average decoding attention cost by 52.0% on Gemma-4-31B and 31.1% on…" |
| v2 | https://arxiv.org/abs/2609.02737 (abstract) | 45.8% | 19.6% | "reduces total attended tokens per response (45.8%, 19.6%…)" |

Verified by fetching each page in this session (HTTP 200) and matching the figures in the text. The v2 body also carries 45.8%/19.6%. Any repo text should cite v2 (45.8%/19.6%) or say which version it means.

## Recommendations

1. Make `REAL_MODEL_FORWARD_PASS` require full tensor mapping (coverage 1.0) and report the coverage ratio.
2. Take expected shapes from the checkpoint's `config.json`, not the hard-coded 0.8B fallback (`model_config.py:57-66`, `:136-146`).
3. Rewrite the MTP identity test so the MTP arm proposes drafts and the acceptance rate is measured; until then, label MTP "not evidenced".
4. Add a LOCAL-window stability assertion to `test_real_engine.py:436`.
5. Add an H2O image-prompt test that fails when text tokens kept is 0, and gate image-prompt H2O results until fixed.
6. Update `docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md:27` to v2 numbers, or mark them as v1.
7. Run the full suite on `0919eaa` (the current head) before stating any suite-level pass for it. The only full-suite result so far is for `77ec005`: 281 passed, 10 skipped.
