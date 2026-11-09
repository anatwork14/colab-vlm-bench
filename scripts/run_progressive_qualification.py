#!/usr/bin/env python3
"""Progressive Qualification Suite for Qwen3.5-0.8B Pretrained Neural Model.

Autonomous Qualification Harness for HERDR wF:
1. P0-01: Verifies 100% official Transformers hooking (`Qwen3_5ForConditionalGeneration` +
   `Qwen35MTPModule`), 488/488 safetensors tensor coverage (852,985,920 base VL params +
   20,452,864 MTP params = 873,438,784 total params, 0 unmapped), and one-step logits parity.
2. P0-02: Processes real image bytes from `000000039769.png` through official `AutoProcessor`
   (`pixel_values`, `image_grid_thw`) and ViT encoder for EVERY arm (`base`, `h2o_base`,
   `da_base`, `ecrd_base`, `mtp`, `h2o_mtp`, `da_mtp`) and validates across all 4 canonical
   multimodal tasks (Existence Positive, Existence Negative, Attribute, Relation).
3. P1-03: Enforces strict Declarative Attention cost accounting (`attention_mask_restriction_ratio`
   labeled as logical mask restriction estimate, `physical_kv_bytes_freed = 0`, 100% KV residency
   preserved, strictly separated from H2O physical KV eviction).
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from vlmbench.infrastructure import (
    Qwen35ModelConfig,
    RealSevenArmEngine,
    EvidencePool,
    EvidenceItem,
)
from vlmbench.multimodal_fixture import (
    CanonicalMultimodalFixture,
    CANONICAL_IMAGE_SHA256,
    CANONICAL_PANOPTIC_SHA256,
    get_supplementary_evaluation_suite,
)


def run_test_a_architecture(model_dir: Path, device: str = "cpu") -> dict[str, Any]:
    """Test A: Checkpoint Loading, 100% Tensor/Parameter Coverage & One-Step Logits Parity (P0-01)."""
    print("=== [Test A] Checkpoint Loading, 100% Coverage Inventory & Logits Parity ===")
    t0 = time.perf_counter()

    # 1. Config verification
    config_file = model_dir / "config.json"
    assert config_file.is_file(), f"Missing config.json in {model_dir}"
    with open(config_file, "r", encoding="utf8") as f:
        cfg_dict = json.load(f)

    tc = cfg_dict.get("text_config", cfg_dict)
    vocab_size = tc.get("vocab_size", cfg_dict.get("vocab_size"))
    num_layers = tc.get("num_hidden_layers", cfg_dict.get("num_hidden_layers"))
    layer_types = tc.get("layer_types", [])

    assert cfg_dict.get("model_type") == "qwen3_5", f"Expected qwen3_5, got {cfg_dict.get('model_type')}"
    assert vocab_size == 248320, f"Expected 248320, got {vocab_size}"
    assert num_layers == 24, f"Expected 24 layers, got {num_layers}"
    assert cfg_dict.get("image_token_id") == 248056
    assert cfg_dict.get("vision_start_token_id") == 248053
    assert cfg_dict.get("vision_end_token_id") == 248054

    full_attn = [i for i, lt in enumerate(layer_types) if lt == "full_attention"] if layer_types else [3, 7, 11, 15, 19, 23]
    assert len(full_attn) == 6, f"Expected 6 full attention layers, got {len(full_attn)}"
    recurrent_count = num_layers - len(full_attn)
    assert recurrent_count == 18, f"Expected 18 linear recurrent layers, got {recurrent_count}"

    # 2. Safetensors weights verification
    weight_file = model_dir / "model.safetensors-00001-of-00001.safetensors"
    assert weight_file.is_file(), f"Missing safetensors file in {model_dir}"
    weight_size = weight_file.stat().st_size
    assert weight_size == 1746942600, f"Unexpected weight file size: {weight_size}"

    with safe_open(str(weight_file), framework="pt") as sf:
        keys = list(sf.keys())
        mtp_keys = [k for k in keys if "mtp" in k]
        visual_keys = [k for k in keys if "visual" in k]
        lm_keys = [k for k in keys if "language_model" in k]

    assert len(keys) == 488, f"Expected 488 total safetensors keys, found {len(keys)}"
    assert len(mtp_keys) == 15, f"Expected 15 MTP keys, found {len(mtp_keys)}"
    assert len(visual_keys) == 153, f"Expected 153 vision keys, found {len(visual_keys)}"
    assert len(lm_keys) == 320, f"Expected 320 language keys, found {len(lm_keys)}"

    # 3. Verify 100% official hooking coverage & one-step logits parity via RealSevenArmEngine
    engine = RealSevenArmEngine(method_arm="base", device=device, checkpoint_dir=model_dir)
    assert engine.is_pretrained is True
    assert engine.is_official_hf_hooked is True
    inv = engine.pretrained_weight_summary
    assert inv is not None
    assert inv["checkpoint_total_tensors"] == 488
    assert inv["mapped_tensors_count"] == 488
    assert inv["unmapped_tensors_count"] == 0
    assert inv["unmapped_tensor_keys"] == []
    assert inv["base_vl_tensors_loaded"] == 473
    assert inv["mtp_tensors_loaded"] == 15
    assert inv["base_vl_parameters"] == 852985920
    assert inv["mtp_parameters"] == 20452864
    assert inv["total_checkpoint_parameters"] == 873438784
    assert inv["mapped_parameters"] == 873438784
    assert inv["unmapped_parameters"] == 0
    assert inv["tensor_coverage_ratio"] == 1.0
    assert inv["parameter_coverage_ratio"] == 1.0
    assert inv["logits_parity"]["verified"] is True
    assert inv["logits_parity"]["max_abs_logits_diff"] <= 1e-5
    assert inv["logits_parity"]["top1_match"] is True
    engine.close()

    dt = time.perf_counter() - t0
    result = {
        "duration_s": round(dt, 3),
        "total_parameters": 852985920,
        "mtp_parameters": 20452864,
        "total_checkpoint_parameters": 873438784,
        "vocab_size": 248320,
        "num_hidden_layers": 24,
        "full_attention_layers": full_attn,
        "recurrent_layers_count": recurrent_count,
        "checkpoint_total_tensors": 488,
        "base_vl_keys_count": 473,
        "language_keys_count": len(lm_keys),
        "vision_keys_count": len(visual_keys),
        "mtp_keys_count": len(mtp_keys),
        "coverage_inventory": inv,
    }
    print(
        f"Test A PASSED in {dt:.2f}s (488/488 tensors [100.0%], 852.99M VL + 20.45M MTP params, "
        f"Logits Parity max_diff={inv['logits_parity']['max_abs_logits_diff']:.2e})"
    )
    return result


def run_test_b_text_forward(model_dir: Path, device: str = "cpu") -> dict[str, Any]:
    """Test B: Single Real Neural Forward Pass (Text Only)."""
    print(f"=== [Test B] Single Real Neural Forward Pass (Text Only on {device}) ===")
    t0 = time.perf_counter()

    engine = RealSevenArmEngine(method_arm="base", device=device, checkpoint_dir=model_dir)
    proc = engine.processor
    model = engine.model.hf_model

    prompt = "Answer the question directly: What is the color of an emerald?"
    messages = [
        {"role": "user", "content": prompt},
    ]
    chat_prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[chat_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits
        assert logits.ndim == 3
        assert logits.shape[-1] == 248320
        assert not torch.isnan(logits).any(), "Logits contain NaN!"
        next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        decoded = proc.decode([next_tok]).strip()
        assert len(decoded) > 0, "Decoded token is empty!"

    engine.close()

    dt = time.perf_counter() - t0
    result = {
        "duration_s": round(dt, 3),
        "prompt": prompt,
        "input_tokens_count": int(inputs["input_ids"].shape[1]),
        "logits_shape": list(logits.shape),
        "next_token_id": next_tok,
        "next_token_text": decoded,
    }
    print(f"Test B PASSED in {dt:.2f}s (Prompt '{prompt}' -> Next token: {next_tok} '{decoded}')")
    return result


def run_test_c_multimodal_forward(model_dir: Path, fixture_path: Path, device: str = "cpu") -> dict[str, Any]:
    """Test C: Real Multimodal Forward with Image Fixture & Grounded Evidence."""
    print(f"=== [Test C] Real Multimodal Forward (Image + Vision Encoder on {device}) ===")
    t0 = time.perf_counter()

    assert fixture_path.is_file(), f"Missing fixture image: {fixture_path}"
    with open(fixture_path, "rb") as f:
        img_sha = hashlib.sha256(f.read()).hexdigest()
    assert img_sha == CANONICAL_IMAGE_SHA256, f"Cryptographic mismatch: {img_sha} != {CANONICAL_IMAGE_SHA256}"

    engine = RealSevenArmEngine(method_arm="base", device=device, checkpoint_dir=model_dir)
    proc = engine.processor
    model = engine.model.hf_model

    img = Image.open(fixture_path).convert("RGB")
    question = "Are there cats lying on the couch in the living room?"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt_text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[prompt_text], images=[img], padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits
        assert logits.ndim == 3
        assert logits.shape[-1] == 248320
        next_tok = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        decoded = proc.decode([next_tok]).strip()

    engine.close()

    dt = time.perf_counter() - t0
    result = {
        "duration_s": round(dt, 3),
        "image_fixture_sha256": img_sha,
        "question": question,
        "input_sequence_length": int(inputs["input_ids"].shape[1]),
        "vision_pixel_values_shape": list(inputs["pixel_values"].shape),
        "image_grid_thw": inputs["image_grid_thw"].detach().cpu().tolist(),
        "visual_token_embeddings_count": int((inputs["input_ids"] == 248056).sum().item()),
        "next_token_id": next_tok,
        "next_token_text": decoded,
        "grounded_verification": decoded.lower().startswith("yes"),
    }
    print(f"Test C PASSED in {dt:.2f}s (Image verified -> Output: {next_tok} '{decoded}', Grounded: {result['grounded_verification']})")
    return result


def run_test_d_seven_arms(model_dir: Path, fixture_path: Path, device: str = "cpu") -> dict[str, Any]:
    """Test D: Genuine Multimodal Inference Across All Seven Arms + 4 Canonical Tasks (P0-01, P0-02, P1-03)."""
    print(f"=== [Test D] Seven-Arm Official Hooked Multimodal Execution & 4-Task Suite on {device} ===")
    t0 = time.perf_counter()

    expected_arms = [
        "base",
        "h2o_base",
        "da_base",
        "ecrd_base",
        "mtp",
        "h2o_mtp",
        "da_mtp",
    ]

    raw_image_bytes = fixture_path.read_bytes()
    img_sha = hashlib.sha256(raw_image_bytes).hexdigest()
    assert img_sha == CANONICAL_IMAGE_SHA256

    evidence_pool = EvidencePool("test_d_arm_pool")
    evidence_pool.add_evidence(
        EvidenceItem("ev_obj_01", "image_region", "two domestic cats on sofa", CANONICAL_IMAGE_SHA256),
        keywords=["cats", "cat", "couch", "sofa", "yes", "lying"],
    )

    arm_results: dict[str, Any] = {}
    question = "Are there cats lying on the couch in the living room?"

    for arm in expected_arms:
        engine = RealSevenArmEngine(method_arm=arm, device=device, checkpoint_dir=model_dir)
        assert engine.is_pretrained is True, f"{arm}: expected is_pretrained=True"
        assert engine.is_official_hf_hooked is True, f"{arm}: expected is_official_hf_hooked=True"
        assert engine.has_vision_encoder is True, f"{arm}: expected has_vision_encoder=True"

        res = engine.execute_request(
            request_id=f"test_d_{arm}",
            max_new_tokens=8,
            evidence_pool=evidence_pool,
            image_present=True,
            image_path=fixture_path,
            image_bytes=raw_image_bytes,
            question=question,
        )

        assert res.method_arm == arm
        assert len(res.tokens_generated) == 8
        assert res.total_latency_ms > 0.0

        # Verify P0-01 official hooking & SA-01 gates
        assert res.gates["POLICY_HARNESS_PASS"] == "PASS"
        assert res.gates["ACTUAL_BACKEND_HOOK_PASS"] in ("PASS", "GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED")
        assert res.gates["REAL_MODEL_FORWARD_PASS"] == "PASS"
        assert res.gates["REAL_MULTIMODAL_INFERENCE_PASS"] == "PASS"
        assert res.gates["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"
        assert res.gates["CCU_ISOLATION_PASS"] == "GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED"
        assert res.gates["A100_GPU_QUALIFIED"] == "BLOCKED_PENDING_OWNER_PAID_AUTH"
        assert res.telemetry["architectural_status"] == "OFFICIAL_HF_PRETRAINED_HOOKED_MODEL"
        assert res.telemetry["pretrained_weights"]["mapped_tensors_count"] == 488
        assert res.telemetry["pretrained_weights"]["unmapped_tensors_count"] == 0
        assert res.telemetry["pretrained_weights"]["parameter_coverage_ratio"] == 1.0

        # Verify P0-02 visual encoder invocation trace per arm
        vis_trace = res.telemetry["visual_encoder_trace"]
        assert vis_trace["invoked"] is True, f"{arm}: ViT encoder was not invoked"
        assert vis_trace["image_sha256"] == CANONICAL_IMAGE_SHA256
        assert vis_trace["pixel_values_shape"] == [1200, 1536]
        assert vis_trace["image_grid_thw"] == [[1, 30, 40]]
        assert vis_trace["visual_token_embeddings_count"] == 300
        assert vis_trace["visual_embeddings_shape"] == [300, 1024]

        # Verify arm-specific interventions & P1-03 cost accounting
        if "h2o" in arm:
            assert res.physical_kv_bytes_freed > 0, f"{arm}: expected nonzero per-step KV tensor footprint deltas"
            assert res.logical_selection_fraction > 0.0, f"{arm} must report logical selection fraction > 0"
        if "da" in arm:
            assert res.attention_mask_restriction_ratio > 0.0
            assert res.physical_kv_bytes_freed == 0, f"{arm} (P1-03) must report physical_kv_bytes_freed == 0"
            da_acc = res.telemetry["da_cost_accounting"]
            assert da_acc["physical_kv_bytes_freed"] == 0
            assert da_acc["kv_residency_preserved_pct"] == 100.0
            assert da_acc["physical_kv_read_saving_claimed"] is False
            assert da_acc["hbm_traffic_reduction_claimed"] is False
        if "ecrd" in arm:
            assert len(res.grounded_citations) > 0, f"{arm} must produce grounded citations"
        if "mtp" in arm:
            assert res.gates["MTP_CORRECTNESS_PASS"] == "GATED_NATIVE_MTP_VERIFIER_REQUIRED"
            assert res.gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] == "PASS"

        arm_info = {
            "tokens_generated": res.tokens_generated,
            "tokens_generated_count": len(res.tokens_generated),
            "output_text": res.output_text,
            "total_latency_ms": round(res.total_latency_ms, 2),
            "ttft_ms": round(res.ttft_ms, 2),
            "tpot_ms": round(res.tpot_ms, 2),
            "physical_kv_bytes_allocated": res.physical_kv_bytes_allocated,
            "physical_kv_bytes_freed": res.physical_kv_bytes_freed,
            "kv_memory_accounting": res.telemetry["kv_memory_accounting"],
            "mtp_candidate_diagnostic": res.telemetry["mtp_candidate_diagnostic"],
            "logical_selection_fraction": round(res.logical_selection_fraction, 4),
            "attention_mask_restriction_ratio": round(res.attention_mask_restriction_ratio, 4),
            "grounded_citations_count": len(res.grounded_citations),
            "mtp_acceptance_rate": round(res.mtp_acceptance_rate, 4),
            "architectural_status": res.telemetry["architectural_status"],
            "pretrained_tensors_loaded": res.telemetry["pretrained_weights"]["loaded_tensors_count"],
            "parameter_coverage_ratio": res.telemetry["pretrained_weights"]["parameter_coverage_ratio"],
            "visual_encoder_trace": vis_trace,
            "da_cost_accounting": res.telemetry["da_cost_accounting"],
            "gates": res.gates,
        }

        engine.close()
        arm_results[arm] = arm_info
        print(
            f"  Arm [{arm:10s}] -> Text: {res.output_text!r} | TTFT: {res.ttft_ms:.1f}ms, "
            f"TPOT: {res.tpot_ms:.1f}ms, KVShapeDeltaSum(NOT VRAM): {res.physical_kv_bytes_freed}B, "
            f"DA_Mask_Ratio: {res.attention_mask_restriction_ratio:.4f}, "
            f"ViT_Tokens: {vis_trace['visual_token_embeddings_count']}, Citations: {len(res.grounded_citations)}"
        )

    # Validate across all 4 canonical tasks (P0-02: Existence Positive, Existence Negative, Attribute, Relation)
    print("  --- Validating 4 Canonical Multimodal Tasks (Existence+, Existence-, Attribute, Relation) ---")
    suite = get_supplementary_evaluation_suite(fixture_path.parent)
    task_results: dict[str, Any] = {}
    task_engine = RealSevenArmEngine(method_arm="base", device=device, checkpoint_dir=model_dir)

    for fix in suite:
        max_toks = 18 if fix.task_type == "attribute" else 8
        pool = fix.build_evidence_pool(request_id=f"task_{fix.evidence_id}")
        task_res = task_engine.execute_request(
            request_id=f"task_{fix.evidence_id}",
            max_new_tokens=max_toks,
            evidence_pool=pool,
            image_present=True,
            image_path=fix.image_path,
            image_bytes=raw_image_bytes,
            question=fix.question,
        )
        v_trace = task_res.telemetry["visual_encoder_trace"]
        assert v_trace["invoked"] is True
        assert v_trace["pixel_values_shape"] == [1200, 1536]
        assert v_trace["visual_token_embeddings_count"] == 300
        assert len(task_res.output_text) > 0

        # The four questions are only neural/ViT smoke fixtures. A substring
        # match is NOT accuracy: e.g. the attribute gold says Yellow but the
        # word "pink" or even "color" used to be counted as PASS. Never turn
        # these fixtures into reported benchmark correctness.
        inference_smoke_pass = (
            v_trace["invoked"] is True
            and len(task_res.tokens_generated) == max_toks
            and bool(task_res.output_text)
        )
        assert inference_smoke_pass

        task_results[fix.evidence_id] = {
            "task_type": fix.task_type,
            "question": fix.question,
            "ground_truth_label": fix.ground_truth_label,
            "output_text": task_res.output_text,
            "tokens_generated": task_res.tokens_generated,
            "inference_smoke_pass": True,
            "semantic_verification_pass": None,
            "ground_truth_match": None,
            "quality_score_qualified": False,
            "quality_scoring_status": "UNSCORED_SMOKE_ONLY_USE_OFFICIAL_AMBER_FOR_QUALITY",
            "visual_encoder_trace": v_trace,
        }
        print(
            f"  Task [{fix.evidence_id:18s} ({fix.task_type:9s})]"
            f" -> Inference smoke PASS; ground-truth accuracy NOT SCORED"
        )

    task_engine.close()

    dt = time.perf_counter() - t0
    result = {
        "duration_s": round(dt, 3),
        "arms_evaluated": len(expected_arms),
        "arms": arm_results,
        "canonical_four_tasks": task_results,
    }
    print(f"Test D INFERENCE SMOKE PASSED in {dt:.2f}s (7 arms + 4 image tasks; accuracy NOT SCORED)")
    return result


def run_test_e_speculative_correctness(model_dir: Path, fixture_path: Path, device: str = "cpu") -> dict[str, Any]:
    """Test E: source-aligned MTP proposal diagnostic and greedy target equivalence (not speculative serving)."""
    print(f"=== [Test E] MTP Proposal Diagnostic + Greedy Target Equivalence on {device} ===")
    t0 = time.perf_counter()

    engine = RealSevenArmEngine(method_arm="h2o_mtp", device=device, checkpoint_dir=model_dir)
    assert engine.is_pretrained is True
    assert engine.is_official_hf_hooked is True
    res = engine.execute_request(
        request_id="test_e_spec_tx",
        max_new_tokens=16,
        image_present=True,
        image_path=fixture_path,
        question="Are there cats lying on the couch in the living room?",
    )

    assert len(res.tokens_generated) == 16
    assert res.gates["REAL_MODEL_FORWARD_PASS"] == "PASS"
    assert res.gates["REAL_MULTIMODAL_INFERENCE_PASS"] == "PASS"
    assert res.gates["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"
    assert res.gates["MTP_CORRECTNESS_PASS"] == "GATED_NATIVE_MTP_VERIFIER_REQUIRED"
    assert res.gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] == "PASS"
    assert res.mtp_acceptance_rate == 0.0
    assert res.physical_kv_bytes_freed > 0, "H2O diagnostic must show KV tensor footprint compaction"
    assert 0.0 <= res.mtp_acceptance_rate <= 1.0

    engine.close()
    # Paired greedy-equivalence: the MTP diagnostic must NEVER change target
    # output tokens, even when MTP proposals do not match the target.
    baseline = RealSevenArmEngine(method_arm="h2o_base", device=device, checkpoint_dir=model_dir)
    baseline_res = baseline.execute_request(
        request_id="test_e_target_reference",
        max_new_tokens=16, image_present=True, image_path=fixture_path,
        question="Are there cats lying on the couch in the living room?",
    )
    baseline.close()
    assert res.tokens_generated == baseline_res.tokens_generated, (
        "MTP diagnostic changed the H2O target's greedy output tokens"
    )
    mtp_diag = res.telemetry["mtp_candidate_diagnostic"]
    assert mtp_diag["committed_draft_tokens"] == 0
    assert mtp_diag["proposals_verified"] > 0
    dt = time.perf_counter() - t0
    result = {
        "claim_scope": "DIAGNOSTIC_ONLY",
        "duration_s": round(dt, 3),
        "mtp_acceptance_rate": 0.0,
        "proposal_match_rate": mtp_diag["proposal_match_rate"],
        "greedy_target_equivalence": True,
        "legacy_physical_kv_freed_bytes": res.physical_kv_bytes_freed,
        "kv_memory_accounting": res.telemetry["kv_memory_accounting"],
        "peak_gpu_vram_saved_bytes": None,
        "output_text": res.output_text,
        "architectural_status": res.telemetry["architectural_status"],
        "gates": res.gates,
    }
    print(f"Test E DIAGNOSTIC PASSED in {dt:.2f}s (Proposal Match: {mtp_diag['proposal_match_rate']}, target-equivalent, no speculative token committed)")
    return result


def run_test_f_fixed_output_serving(model_dir: Path, device: str = "cpu", target_tokens: int = 256) -> dict[str, Any]:
    """Test F: Fixed-Output Serving (256 Committed Tokens, Temperature=0, Top_P=1.0)."""
    print(f"=== [Test F] Fixed-Output Serving ({target_tokens} Committed Tokens on {device}) ===")
    t0 = time.perf_counter()

    engine = RealSevenArmEngine(method_arm="base", device=device, checkpoint_dir=model_dir)
    assert engine.is_pretrained is True
    assert engine.is_official_hf_hooked is True
    res = engine.execute_request(
        request_id="test_f_fixed_256",
        prompt_token_ids=[100, 101, 102],
        max_new_tokens=target_tokens,
        image_present=False,
    )

    assert len(res.tokens_generated) == target_tokens, f"Expected {target_tokens} tokens, got {len(res.tokens_generated)}"
    assert res.gates["REAL_MODEL_FORWARD_PASS"] == "PASS"
    assert res.gates["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"
    engine.close()

    dt = time.perf_counter() - t0
    result = {
        "duration_s": round(dt, 3),
        "target_tokens": target_tokens,
        "tokens_committed": len(res.tokens_generated),
        "total_latency_ms": round(res.total_latency_ms, 2),
        "ttft_ms": round(res.ttft_ms, 2),
        "tpot_ms": round(res.tpot_ms, 2),
        "throughput_tok_per_sec": round(target_tokens / (res.total_latency_ms / 1000.0), 2),
        "throughput_scope": "SINGLE_REQUEST_LOCAL_PYTORCH_REFERENCE_NOT_VLLM_SERVING",
        "quality_score_qualified": False,
        "architectural_status": res.telemetry["architectural_status"],
    }
    print(f"Test F PYTORCH DIAGNOSTIC PASSED in {dt:.2f}s ({target_tokens} tokens, local reference rate {result['throughput_tok_per_sec']} tok/s; NOT vLLM serving)")
    return result


def run_test_g_ccu_concurrency(model_dir: Path, device: str = "cpu") -> dict[str, Any]:
    """Test G: Concurrency Isolation Regression (CCU 1, CCU 2, CCU 4)."""
    print(f"=== [Test G] Concurrency Isolation Regression (CCU 1, 2, 4 on {device}) ===")
    t0 = time.perf_counter()

    ccu_results: dict[str, Any] = {}

    for ccu in (1, 2, 4):
        ccu_t0 = time.perf_counter()
        engine = RealSevenArmEngine(method_arm="ecrd_base", device=device, checkpoint_dir=model_dir)
        assert engine.is_pretrained is True
        assert engine.is_official_hf_hooked is True

        requests = []
        for i in range(ccu):
            pool = EvidencePool(f"req_{ccu}_{i}")
            pool.add_evidence(
                EvidenceItem(f"ev_{i}", "text", f"fact for entity {i}", hashlib.sha256(f"entity_{i}".encode()).hexdigest()),
                keywords=["target", "region", "apple", "object"],
            )
            requests.append((f"req_{ccu}_{i}", [100, 101], 4, pool))

        with concurrent.futures.ThreadPoolExecutor(max_workers=ccu) as executor:
            futures = [
                executor.submit(engine.execute_request, req_id, p_tokens, max_toks, pool, False)
                for req_id, p_tokens, max_toks, pool in requests
            ]
            outputs = [f.result() for f in futures]

        assert len(outputs) == ccu
        for idx, out in enumerate(outputs):
            assert len(out.tokens_generated) == 4
            assert out.gates["REAL_MODEL_FORWARD_PASS"] == "PASS"
            assert out.gates["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"
            assert out.gates["CCU_ISOLATION_PASS"] == "GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED"
            assert len(out.grounded_citations) > 0, f"Request {idx} in CCU {ccu} must produce citations"
            for cit in out.grounded_citations:
                for ev_id in cit["evidence_ids"]:
                    assert ev_id == f"ev_{idx}", f"Isolation breach: {ev_id} appeared in request {idx}"

        engine.close()
        ccu_dt = time.perf_counter() - ccu_t0
        ccu_results[f"CCU_{ccu}"] = {
            "concurrency": ccu,
            "duration_s": round(ccu_dt, 3),
            "isolation_pass": True,
            "citations_per_request": [len(o.grounded_citations) for o in outputs],
        }
        print(f"  CCU {ccu}: Verified zero cross-request contamination in {ccu_dt:.2f}s (Citations: {[len(o.grounded_citations) for o in outputs]})")

    dt = time.perf_counter() - t0
    result = {
        "concurrency_scope": "PYTORCH_THREAD_REQUEST_ISOLATION_NOT_VLLM_SCHEDULER",
        "duration_s": round(dt, 3),
        "levels": ccu_results,
    }
    print(f"Test G PASSED in {dt:.2f}s (Strict Isolation across CCU 1, 2, 4)")
    return result


EXPECTED_TESTS = (
    "test_a_architecture",
    "test_b_text_forward",
    "test_c_multimodal_forward",
    "test_d_seven_arms",
    "test_e_speculative_correctness",
    "test_f_fixed_output_serving",
    "test_g_ccu_concurrency",
)
OK_VERDICT = "PRETRAINED_FORWARD_AND_MTP_PROPOSAL_DIAGNOSTIC_ONLY"


# ---------------------------------------------------------------------------
# Result recording and release gates (ITER2 D2b G-01/G-02).
# Every gate is derived from recorded test outcomes. Gate values are bare labels
# (PASS, FAIL, NOT_TESTED, GATED_*, BLOCKED_*); evidence numbers live separately
# in release_gate_evidence and are never embedded in a gate string.
# ---------------------------------------------------------------------------

def guarded(fn, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Run one qualification test and record its outcome. A failure is recorded, not dropped."""
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # recorded in the receipt; the run exits non-zero
        print(f"[FAIL] {fn.__name__}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)[:2000]}
    return {"status": "PASS", **result}


def _test_status(test: dict[str, Any] | None) -> str:
    if test is None:
        return "NOT_TESTED"
    status = test.get("status")
    return status if status in ("PASS", "FAIL") else "NOT_TESTED"


def _combine(*statuses: str) -> str:
    if "FAIL" in statuses:
        return "FAIL"
    if statuses and all(s == "PASS" for s in statuses):
        return "PASS"
    return "NOT_TESTED"


def _engine_gate(test_d: dict[str, Any] | None, name: str) -> str:
    """The value the engine itself reported, used only if every arm reporting it agrees."""
    status = _test_status(test_d)
    if status != "PASS":
        return status
    values = {arm["gates"][name] for arm in (test_d or {}).get("arms", {}).values() if name in arm.get("gates", {})}
    if not values:
        return "NOT_TESTED"
    return values.pop() if len(values) == 1 else "FAIL"


def compute_release_gates(tests: dict[str, dict[str, Any]], device_name: str) -> dict[str, str]:
    a, b, c = tests.get("test_a_architecture"), tests.get("test_b_text_forward"), tests.get("test_c_multimodal_forward")
    d, e, g = tests.get("test_d_seven_arms"), tests.get("test_e_speculative_correctness"), tests.get("test_g_ccu_concurrency")
    return {
        "POLICY_HARNESS_PASS": _test_status(a),
        "ACTUAL_BACKEND_HOOK_PASS": _engine_gate(d, "ACTUAL_BACKEND_HOOK_PASS"),
        "REAL_MODEL_FORWARD_PASS": _combine(_test_status(a), _test_status(b)),
        "REAL_MULTIMODAL_INFERENCE_PASS": _combine(_test_status(c), _test_status(d)),
        "SEVEN_ARM_PRETRAINED_PASS": _test_status(d),
        "MTP_CORRECTNESS_PASS": _engine_gate(d, "MTP_CORRECTNESS_PASS"),
        "MTP_PROPOSAL_DIAGNOSTIC_PASS": _test_status(e),
        "CCU_REQUEST_ISOLATION_TEST_PASS": _test_status(g),
        # This script runs no vLLM scheduler load test and no A100 run.
        "VLLM_SCHEDULER_CONCURRENCY_PASS": "NOT_TESTED",
        "A100_GPU_QUALIFIED": "NOT_TESTED" if "A100" in device_name else "BLOCKED_PENDING_OWNER_PAID_AUTH",
    }


def verdict_for(tests: dict[str, dict[str, Any]]) -> str:
    failed = [name for name in EXPECTED_TESTS if _test_status(tests.get(name)) != "PASS"]
    return OK_VERDICT if not failed else "INCOMPLETE_OR_FAILED: " + ", ".join(failed)


def gate_evidence(tests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    inv = (tests.get("test_a_architecture") or {}).get("coverage_inventory") or {}
    parity = inv.get("logits_parity") or {}
    return {
        "logits_parity_max_abs_diff": parity.get("max_abs_logits_diff"),
        "logits_parity_top1_match": parity.get("top1_match"),
        "checkpoint_total_tensors": inv.get("checkpoint_total_tensors"),
        "mapped_tensors_count": inv.get("mapped_tensors_count"),
        "tensor_coverage_ratio": inv.get("tensor_coverage_ratio"),
        "parameter_coverage_ratio": inv.get("parameter_coverage_ratio"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Progressive Qualification Harness for Qwen3.5-0.8B Pretrained Model")
    parser.add_argument("--model-dir", type=Path, default=Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B"))
    parser.add_argument("--fixture-path", type=Path, default=REPO_ROOT / "tests" / "fixtures" / "000000039769.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--receipt-path", type=Path, default=REPO_ROOT / "artifacts" / "qwen35_pretrained_qualification_receipt.json")
    args = parser.parse_args()

    overall_t0 = time.perf_counter()
    print("================================================================================")
    print("  PROGRESSIVE QUALIFICATION SUITE: REAL QWEN3.5-0.8B PRETRAINED VALIDATION")
    print(f"  Model Checkpoint: {args.model_dir}")
    print(f"  Target Device:    {args.device}")
    print(f"  Canonical Image:  {args.fixture_path}")
    print("================================================================================")

    receipt: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_id": "Qwen/Qwen3.5-0.8B",
        "pinned_revision_sha": "2fc06364715b967f1860aea9cf38778875588b17",
        "checkpoint_path": str(args.model_dir),
        "target_device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "tests": {},
    }

    tests: dict[str, dict[str, Any]] = {
        "test_a_architecture": guarded(run_test_a_architecture, args.model_dir, device=args.device),
        "test_b_text_forward": guarded(run_test_b_text_forward, args.model_dir, device=args.device),
        "test_c_multimodal_forward": guarded(run_test_c_multimodal_forward, args.model_dir, args.fixture_path, device=args.device),
        "test_d_seven_arms": guarded(run_test_d_seven_arms, args.model_dir, args.fixture_path, device=args.device),
        "test_e_speculative_correctness": guarded(run_test_e_speculative_correctness, args.model_dir, args.fixture_path, device=args.device),
        "test_f_fixed_output_serving": guarded(run_test_f_fixed_output_serving, args.model_dir, device=args.device, target_tokens=256),
        "test_g_ccu_concurrency": guarded(run_test_g_ccu_concurrency, args.model_dir, device=args.device),
    }

    receipt["tests"] = tests
    receipt["qualification_verdict"] = verdict_for(tests)
    receipt["total_duration_s"] = round(time.perf_counter() - overall_t0, 3)
    receipt["release_gates"] = compute_release_gates(tests, device_name=receipt["cuda_device_name"])
    receipt["release_gate_evidence"] = gate_evidence(tests)

    args.receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.receipt_path, "w", encoding="utf8") as f:
        json.dump(receipt, f, indent=2)

    print("\n================================================================================")
    print(f"  VERDICT: {receipt['qualification_verdict']} in {receipt['total_duration_s']:.2f}s")
    print(f"  Execution Receipt written to: {args.receipt_path}")
    print("================================================================================")
    if receipt["qualification_verdict"] != OK_VERDICT:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
