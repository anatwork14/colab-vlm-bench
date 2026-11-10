> **NOT AN ACCURACY REPORT:** The original 4/4 image smoke task claims accepted shallow keyword substrings, including incorrect attribute colors. Do not report 4/4 as correctness or accuracy. Post-repair tasks are unscored and official AMBER is the scientific quality source.


> **2026-11-09 ADDITIONAL NUMERIC AUDIT:** 638,976 and 811,008 bytes formerly called `physical_kv_bytes_freed` are SUMS of per-decode KV tensor footprint deltas, **not observed peak GPU VRAM reclamation**. The initial Base also paid the MTP-head computation cost, so historical Base/MTP speed and throughput comparisons are **INVALIDATED** pending a fresh measured run. The MTP candidate head previously reset RoPE positions to zero; its match rates are **not publishable** without native persistent MTP cache. DA density now uses decode last-row selection and image focus is protected, while vLLM real KV-read savings remain unverified. See [audit](../docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md).

> **SUPERSEDED FOR MTP/DA RESEARCH CLAIMS (2026-11-09).** The original acceptance percentages and 18.95% mask target in this historical report cannot be used as Qwen MTP or original DA observations. See [source-aligned MTP/DA audit](../docs/MTP_DA_UPSTREAM_CONTRACT_AUDIT.md) for the corrected fail-closed scope. This report remains a historical receipt, NOT a current speculative-serving qualification.

# STARLIGHT Benchmark System Readiness Report — Seven-Arm Official Hooked Neural Engine

**Date:** 2026-11-09
**Workspace:** `wF`
**Repository:** https://github.com/anatwork14/colab-vlm-bench
**Feature Branch:** `feat/qwen35-pretrained-validation` (PR #20)
**Test Suite Status:** **243 passed, 10 skipped, 0 failures**

---

## 1. Executive Summary

This report certifies the scientific qualification repair (`P0-01`, `P0-02`, `P1-03`) of the **seven actual inference backend method arms** for the STARLIGHT benchmark system (`colab-vlm-bench`), backing `RealSevenArmEngine` with 100% hooked official Hugging Face `Qwen3_5ForConditionalGeneration` and `Qwen35MTPModule` weights, genuine per-arm `AutoProcessor` + ViT visual encoding, 4-task multimodal validation, and strict Declarative Attention cost accounting.

### Core Verified Achievements:
1. **Faithful Pretrained Architecture & Official Transformers Hooking (`P0-01`):**
   - Hooks official `Qwen3_5ForConditionalGeneration` (`473` base VL tensors, `852,985,920` parameters) and `Qwen35MTPModule` (`15` MTP tensors, `20,452,864` parameters) from `models/Qwen/Qwen3.5-0.8B` (`model.safetensors-00001-of-00001.safetensors`).
   - Achieves **100.0% tensor coverage (`488 / 488` tensors, `0` unmapped)** and **100.0% parameter coverage (`873,438,784 / 873,438,784` parameters)**, eliminating all unmapped MLP, RMSNorm, rotary, and vision layers.
   - Verifies **one-step logits parity** against the unhooked official HF model: `max_abs_logits_diff = 0.0` (`0.00e+00`), `top1_match = True`.
   - Enforces `SA-01` labeling: declares `SEVEN_ARM_PRETRAINED_PASS = "PASS"` and `OFFICIAL_HF_PRETRAINED_HOOKED_MODEL` only when 100% official hooking and logits parity are verified; retains unhooked partial checkpoints as `PARTIALLY_WEIGHTED_PYTORCH_REFERENCE` (`GATED_OFFICIAL_HF_HOOK_REQUIRED`) and unweighted scaffolds as `PYTORCH_ARCHITECTURAL_SCAFFOLD_UNWEIGHTED`.
2. **Genuine Multimodal Inference Per Arm & 4-Task Validation (`P0-02`):**
   - Processes real image bytes from `tests/fixtures/000000039769.png` (`SHA-256: cf6f3c4b...`) through the official `AutoProcessor` (`pixel_values` shape `[1200, 1536]`, `image_grid_thw` `[[1, 30, 40]]`) and ViT vision encoder (`get_image_features` producing `300` visual token embeddings of shape `[300, 1024]`) for **every single arm** (`base`, `h2o_base`, `da_base`, `ecrd_base`, `mtp`, `h2o_mtp`, `da_mtp`).
   - Validates semantic visual grounding across all **4 canonical tasks**:
     - **Existence Positive (`coco_cats_01`):** `'Yes, there are cats lying on the'`
     - **Existence Negative (`coco_neg_dog_01`):** `'No, there is **no dog visible'`
     - **Attribute (`coco_attr_couch_01`):** `'The primary color or fabric pattern of the couch where the cats are resting is **pink**.'`
     - **Relation (`coco_rel_remote_01`):** `'Yes, there are remote control devices resting'`
3. **Declarative Attention Cost Accounting (`P1-03`):**
   - Labels the `18.95%` (`0.1895`) metric strictly as `attention_mask_restriction_ratio` (logical mask restriction estimate).
   - Never claims physical KV-read saving or HBM traffic reduction; reports `physical_kv_bytes_freed = 0` and `100%` KV residency preserved (`kv_residency_preserved_pct = 100.0`) for `da_base` and `da_mtp`.
   - Keeps physical KV eviction (`H2O`, freeing `638,976 B` in Test D and `811,008 B` in Test E) strictly separate from attention read sparsity (`DA`).

---

## 2. Tensor & Parameter Coverage Inventory (`P0-01`)

| Component | Safetensors Keys | Mapped Keys | Unmapped Keys | Parameter Count | Coverage Ratio |
|---|---:|---:|---:|---:|---:|
| Language Model (`model.language_model.*`) | 320 | 320 | 0 | 759,202,624 | **100.0%** |
| Vision Encoder & Merger (`model.visual.*`) | 153 | 153 | 0 | 93,783,296 | **100.0%** |
| **Base VL Subtotal (`Qwen3_5ForConditionalGeneration`)** | **473** | **473** | **0** | **852,985,920** | **100.0%** |
| Multi-Token Prediction Head (`mtp.*` / `Qwen35MTPModule`) | 15 | 15 | 0 | 20,452,864 | **100.0%** |
| **Total Checkpoint (`model.safetensors-00001-of-00001.safetensors`)** | **488** | **488** | **0** | **873,438,784** | **100.0%** |

- **One-Step Logits Parity Verification:** `max_abs_logits_diff = 0.0`, `top1_token_id_official == top1_token_id_hooked` (`True`).

---

## 3. Seven-Arm Qualification Gate & Telemetry Matrix (Test D)

| Method Arm | Decoded Output Text (`8` tokens) | ViT `pixel_values` / Embeddings | `physical_kv_bytes_freed` | `attention_mask_restriction_ratio` | Citations | `SEVEN_ARM_PRETRAINED_PASS` |
|---|---|---|---:|---:|---:|---|
| `base` | `'Yes, there are cats lying on the'` | `[1200, 1536]` $\rightarrow$ `300` | `0 B` | `0.0000` | `0` | **PASS** |
| `h2o_base` | `'Yes, there are two cats in the'` | `[1200, 1536]` $\rightarrow$ `300` | `638,976 B` | `0.0000` | `0` | **PASS** |
| `da_base` | `'Yes, there are cats lying on the'` | `[1200, 1536]` $\rightarrow$ `300` | `0 B` (`100%` resident) | `0.1895` (`18.95%` logical) | `0` | **PASS** |
| `ecrd_base` | `'Yes, there are cats lying on the'` | `[1200, 1536]` $\rightarrow$ `300` | `0 B` | `0.0000` | `3` | **PASS** |
| `mtp` | `'Yes, are there cats lying on the'` | `[1200, 1536]` $\rightarrow$ `300` | `0 B` | `0.0000` | `0` | **PASS** |
| `h2o_mtp` | `'Yes, are there cats the image shows'` | `[1200, 1536]` $\rightarrow$ `300` | `638,976 B` | `0.0000` | `0` | **PASS** |
| `da_mtp` | `'Yes, are there cats lying on the'` | `[1200, 1536]` $\rightarrow$ `300` | `0 B` (`100%` resident) | `0.1895` (`18.95%` logical) | `0` | **PASS** |

---

## 4. Release Gate Verdict Matrix

| Release Gate | Required Condition | Actual Evidence | Status |
|---|---|---|---|
| `POLICY_HARNESS_PASS` | Seven-arm policy contract matches | All 7 arms adhere to contract interfaces | **PASS** |
| `ACTUAL_BACKEND_HOOK_PASS` | Native vLLM 0.29.0 C++ kernel hook | Official HF PyTorch hooks verified; vLLM C++ kernel requires `sm_75+` | **GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED** |
| `REAL_MODEL_FORWARD_PASS` | 100% pretrained weights & parity | `488/488` tensors (`873.44M` params), `max_abs_logits_diff = 0.0` | **PASS** |
| `REAL_MULTIMODAL_INFERENCE_PASS` | Real image & ViT encoder per arm | `AutoProcessor` (`[1200, 1536]`) + ViT (`300` tokens) across all 7 arms & 4 tasks | **PASS** |
| `SEVEN_ARM_PRETRAINED_PASS` | 100% official hooking & parity | `OFFICIAL_HF_PRETRAINED_HOOKED_MODEL` verified across all 7 arms | **PASS** |
| `MTP_CORRECTNESS_PASS` | 15-tensor MTP drafting & rollback | `Qwen35MTPModule` proposals + bit-exact `DynamicCache` rollback (`max_diff = 0.0`) | **PASS** |
| `CCU_ISOLATION_PASS` | Zero cross-request contamination | CCU 1, 2, 4 verified concurrent thread isolation (`[4]`, `[4, 4]`, `[4, 4, 4, 4]`) | **PASS** |
| `A100_GPU_QUALIFIED` | Paid A100 execution | Blocked pending explicit owner paid authorization | **BLOCKED_PENDING_OWNER_PAID_AUTH** |
