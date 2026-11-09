# Backend interface and method support matrix (ITER3 B3)

Status: engineering reference for `src/vlmbench/infrastructure/real_engine.py`.
Label for every number produced by these backends: `cpu_pytorch_reference_not_vllm`.
This document makes **no speed, VRAM, throughput or quality claim**. vLLM is **BLOCKED**.

Facts below are taken from the code and from the tests named in each row. Line numbers refer to
the worktree at ITER3 B2 (see `git log` for the commit).

## 1. Backends

| ID | What it is | Where | Real RoPE | Hooks available | Speculation (MTP) | Status |
|---|---|---|---|---|---|---|
| B-SCAFFOLD | Tiny synthetic model used by tests (no checkpoint). Pure PyTorch layers. | `Qwen35NeuralModel` (`real_engine.py:567`), `QwenFullAttentionLayer` (`:492`) | **No.** The full-attention layer applies no rotary embedding (`:492-566`). Position-identity claims cannot be checked here. | `h2o_eviction_plans`, `da_masks` (`:815-832`, `:912`) | No MTP head (`mtp_module` is None) | CPU reference, tests only |
| B-HF-HOOKED | Official Qwen3.5-0.8B checkpoint via `transformers` `modeling_qwen3_5`, with the engine's hooked attention. | `RealSevenArmEngine(checkpoint_dir=...)`, hooked attention (`real_engine.py` ~170-300) | **Yes** (`apply_rotary_pos_emb` runs before the cache update, ~line 191) | `h2o_eviction_plans` and `da_masks` via `hook_ctx`; `attention_matrices`; `layer_telemetries`; `compute_mtp` + `mtp_position_ids` | Yes: `mtp` (k=1 verifier), `h2o_mtp` and `starlight_s1_mtp` (policy-state, text-only) | CPU reference (fp32). Fast kernels absent: `causal_conv1d` and `flash-linear-attention` fall back to the PyTorch reference (load-time warnings) |
| B-GPU-PROBE | Piecewise GPU PyTorch reference probes on a GTX 1080 Ti (sm_61). | ledger `docs/harness/EXECUTION_LEDGER.md` | — | — | — | **Not evidence**: no receipt; `base` and `h2o_base` probes only |
| B-VLLM | vLLM serving backend. | `has_vllm_backend` (`real_engine.py:1212`, never set True in the repo); `vllm_backend_status` = `MOCKED_PYTORCH_FALLBACK` (`:2782`) | — | — | — | **BLOCKED** (see §5) |

Facts checked in this round:
- `vllm` is not importable from either the `colab-vlm-bench` or the system Python environment.
- The engine's gate `ACTUAL_BACKEND_HOOK_PASS` is `GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED` unless
  `has_vllm_backend` is True (`real_engine.py:2271`). It is never True.
- Hardware compatibility for vLLM is checked by `check_vllm_hardware_compatibility`
  (`adapters.py`, imported at `real_engine.py:59`). The local GPU is sm_61, which vLLM does not support.

## 2. Interface contract (engine level)

- `RealSevenArmEngine(method_arm, device, config | checkpoint_dir, h2o_budget_ratio, h2o_min_budget, ...)`.
  `SUPPORTED_ARMS` (`real_engine.py:1163`) lists the 11 arm IDs: the seven originals
  (`base`, `h2o_base`, `da_base`, `ecrd_base`, `mtp`, `h2o_mtp`, `da_mtp`) and the four STARLIGHT arms
  (`starlight_s1`, `starlight_s1_no_da`, `starlight_s1_no_grounding`, `starlight_s1_mtp`).
- `execute_request(request_id, prompt_token_ids, max_new_tokens, image_present, evidence_pool=None)` returns
  `RealExecutionResult` with `tokens_generated`, `physical_kv_bytes_allocated`, `physical_kv_bytes_freed`
  (semantics in `MEMORY_FIELD_SCOPES`, `real_engine.py`), `mtp_acceptance_rate`, `gates`, `telemetry`.
- Route predicates decide which path runs (these are the only places speculation is enabled):
  - `_mtp_text_speculation_route` (`real_engine.py:1690`): `mtp`, official, text-only, MTP head present.
  - `_policy_text_speculation_route` (`real_engine.py:1587`): `h2o_mtp` or `starlight_s1_mtp`, official,
    text-only, **no evidence pool**, MTP head present, and `_speculation_enabled`.
  - Every other request uses the diagnostic/normal path. The telemetry field `arm_label` then reads
    `mtp_serving_arm_diagnostic_no_speculation` for the MTP arms, and `speculation_executed` is False.
- Plan contract for physical compaction: a plan is a sorted, unique array of physical indices over the
  current KV length **including** the row being decoded (`compact_orig_positions`, `starlight.py:393`).
- Policy state (`_PolicyState`, `real_engine.py:1080`): cache, tracker, full-attention layers, `P`
  (logical position), original-position map (STARLIGHT), DA spans. Rollback = `_pol_snapshot` /
  `_pol_restore`, covering KV, Gated-DeltaNet state, tracker arrays and the original map.

## 3. Method support matrix

Columns: **CPU ref** = scaffold or official hooked forward on CPU; **HF hooked** = official checkpoint
through the hooked forward; **Telemetry** = what is reported and how it is scoped; **vLLM** = status.
"Not qualified" means the repo's own gate (AGENTS.md: no H2O/DA/ECRD qualification without verified
backend ports) has not been met. No row claims speed.

| Arm | CPU ref (scaffold) | HF hooked (official, CPU) | Speculation | Telemetry | vLLM | Tests |
|---|---|---|---|---|---|---|
| base | supported | supported (normal decode, RoPE) | none | `kv_memory_accounting`, `memory_field_scopes` | BLOCKED | `test_iter3_b1_official_oracle.py`, `test_matched_retention.py` |
| h2o_base | supported (eviction via `h2o_eviction_plans`) | supported (compaction; RoPE keys bit-identical at layer 3; logits match masked reference, max diff 8.1e-6) | none | `h2o_logical_selection_accounting`, `kv_memory_accounting` | BLOCKED; GPU probe piecewise, no receipt | `test_iter3_b1_official_oracle.py` |
| da_base | supported (DA `da_masks`, logical masking only) | supported as masking; `kv_residency_preserved_ratio` = 1.0 (no KV saving claimed) | none | `da_cost_accounting` | BLOCKED; not qualified (scripted DA geometry, `original_paper_protocol_qualified` False) | `test_starlight_oracle.py` (A_MASK) |
| ecrd_base | supported as logits reweighting (`ECRDLogitsProcessor`, normal path) | supported on the normal path; **not KV** | none | `ecrd_reweight` | BLOCKED; not qualified (no verified backend port) | `test_real_engine.py`, `test_seven_methods.py` (ECRD) |
| mtp | diagnostic only (no MTP head) | **verifier-backed k=1 speculation** on text-only requests, label `cpu_greedy_k1_reference_not_vllm_mtp`; image or evidence requests use the diagnostic path | yes (text-only) | `speculation` block (`proposed`, `accepted`, `rejected`, `verifier_calls`), `memory_field_scopes` | BLOCKED | `test_mtp_speculative_serving.py`, `test_mtp_speculative_official.py` |
| h2o_mtp | diagnostic: H2O active, speculation **not** executed (`arm_label` says so) | policy-state speculation on text-only, no-evidence requests (eviction-free 2-row blocks only; rejects roll back KV, GDN and tracker); otherwise H2O stays active on the diagnostic path | yes (text-only, no evidence) | `policy` block (`kind`, `budget_total`), `speculation`, `memory_field_scopes` | BLOCKED | `test_policy_mtp_speculative.py`, `test_iter3_b2_*`, `test_iter3_b1_policy_invariants.py` |
| da_mtp | DA masks active, speculation **not** executed | never speculative (no route); DA accounting reported | none | `da_cost_accounting`, `arm_label` | BLOCKED | `test_iter3_b1_policy_invariants.py` |
| starlight_s1 | supported (decision records per step and layer) | supported; **see F-1** (crash when a protected entry is evicted on some full-attention layers only) | none | `starlight` block, `decision_records`, invariant counters | BLOCKED | `test_starlight_*.py`, `test_iter3_b1_*.py` |
| starlight_s1_no_da | supported (DA signal removed, grounding kept) | as starlight_s1 | none | as starlight_s1 | BLOCKED | `test_starlight_engine.py`, `test_starlight_policy.py` (ablation) |
| starlight_s1_no_grounding | supported (grounding signal removed, DA kept) | as starlight_s1 | none | as starlight_s1 | BLOCKED | `test_starlight_engine.py`, `test_starlight_policy.py` (ablation) |
| starlight_s1_mtp | diagnostic | policy-state speculation on text-only, no-evidence requests; `degraded_to_h2o_plus_da` when declarations are missing (explicit, not silent) | yes (text-only, no evidence) | `starlight` block, `policy`, `speculation` | BLOCKED | `test_policy_mtp_speculative.py`, `test_iter3_b2_*` |

Measured status of "logits after compaction equal the masked reference": **yes, on the official
checkpoint, 7 teacher-forced decode steps, max |logit diff| 8.1e-6, tolerance 1e-4** (test output of
`test_iter3_b1_official_oracle.py`). This is numerical agreement only, not a quality claim.

## 4. Known limitations and findings

**F-1 — Per-layer KV length desynchronisation under protection (defect, open).**
- Symptom: `ValueError: Layer 7: attention mask width 28 does not match KV length 27; refusing a silent slice`.
  The engine raises; it does not silently corrupt. The run is lost.
- Mechanism: the STARLIGHT candidate set comes from H2O's keep choice over all positions, computed without
  protection (`keep_mask_without_protection`, `h2o.py`). Protected entries among those candidates are
  vetoed (`starlight.py:213-221`). Different layers have different scores, so one layer can veto an entry
  that another layer evicts. Retained counts then differ per full-attention layer (measured: layer 3 kept
  22 entries, layer 7 kept 21, same step). The hooked forward builds one attention mask from one layer's
  length and checks it against every layer (`real_engine.py:202`), so it refuses.
- Reproduction (official checkpoint, CPU, 19-token prompt, text TEXT in `test_iter3_b2_mtp_rollback.py`):
  inject a DA focus span `("focus:mid", 16, 17)` into the protected spans. The speculative path
  (`_speculative_policy_mtp_greedy("starlight", ...)`) and the **normal** path
  (`execute_request` with `_speculation_enabled=False`) both raise the ValueError above. So this is not a
  speculation-only defect.
- Scope of the evidence: reproduced with an injected span. Not yet measured with the real DA scopes on
  longer prompts. The default scripted spans did not trigger it in any run.
- Test: `test_iter3_b2_mtp_rollback.py::test_F1_protected_veto_that_differs_across_layers_completes`
  is a **strict xfail** (`raises=ValueError`). It flips to a failure when the defect is fixed.
- Fix direction (not applied, needs a lead ruling): build the candidate set so protected entries are
  forced into the keep set on every full-attention layer, or give all layers one shared retained set.
  Either changes decisions in the normal path too, so existing STARLIGHT expectations must be re-derived.

**F-2 — The scaffold cannot evidence RoPE.** `QwenFullAttentionLayer` applies no rotary embedding
(`real_engine.py:492-566`). Every RoPE-consistency claim is therefore made on B-HF-HOOKED only
(`test_iter3_b1_official_oracle.py`).

**F-3 — Which detector catches which rollback corruption (measured).**
- M1 (KV/GDN cache restore is a no-op), all-rejects run: the engine fails loudly before any token
  comparison with `RuntimeError: STARLIGHT position bookkeeping out of sync on layer 3: 19 tracked vs 21
  cached entries`. This is the engine's own length invariant (`real_engine.py`, position bookkeeping check),
  not the token stream.
- M2 (Gated-DeltaNet state not rolled back, KV only): no length changes, so the length invariant does not
  fire. Only the **bit-for-bit state check** (KV + GDN tensors) catches it
  (`test_iter3_b2_mutation.py::test_M2_*`).
- M3 (policy restore skips tracker and original map): the **policy digest check** fails (measured by
  `test_M3_*`). Whether the length invariant also fires was not recorded.
- M4 (plans ignore DA protection): the **protected-position spy** fires (measured by `test_M4_*`).
- Not measured: whether the greedy token stream alone would catch M2 or M3 on this prompt. The mutation
  tests assert the state checks and do not claim token-level detection.

**Not measured in this round:** how often the MTP head's natural drafts are right (the tests use forced
drafts), and any throughput, latency or memory saving.

## 5. vLLM status

**BLOCKED.** Requirements not met on this host: vLLM is not installed; the local GPU is GTX 1080 Ti (sm_61),
which vLLM does not support; the owner has not authorised a paid GPU (`B-GPU-AUTH`). Until a supported GPU and
an install are approved, no vLLM run, serving benchmark, or vLLM MTP claim exists in this repository.
The engine's own gate reports `GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED` (`real_engine.py:2271`).

## 6. How to reproduce the checks in this document

Run from the worktree root, one heavy job at a time, with a timeout:

```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES= timeout 270 python -m pytest -q tests/test_iter3_b1_policy_invariants.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES= timeout 270 python -m pytest -q -s tests/test_iter3_b1_official_oracle.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES= timeout 270 python -m pytest -q tests/test_iter3_b2_mtp_rollback.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES= timeout 270 python -m pytest -q tests/test_iter3_b2_mutation.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES= timeout 270 python -m pytest -q tests/test_iter3_b3_doc_matrix.py
```

The official-checkpoint files skip automatically when `models/Qwen/Qwen3.5-0.8B/config.json` is absent.
