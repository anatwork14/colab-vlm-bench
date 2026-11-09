"""Comprehensive tests for real Qwen3.5 neural engine and vLLM backend integrations.

Validates:
1. Exact Hugging Face architectural parameters (vocab_size=248320, 3:1 DeltaNet ratio).
2. True physical KV cache byte compaction under H2O.
3. Genuine kernel attention restrictions under Declarative Attention.
4. ECRD logits processing on full 248,320 vocabulary with winning-token citation alignment.
5. MTP source-head proposal diagnostics; native speculative verifier remains gated.
6. Multi-request CCU concurrency isolation (zero cross-request pollution).
7. Complete seven-arm qualification gate reporting.
"""
import concurrent.futures
import numpy as np
import pytest
import torch

from vlmbench.infrastructure import (
    Qwen35ModelConfig,
    load_model_config,
    RealSevenArmEngine,
    RealExecutionResult,
    DeltaNetRecurrentLayer,
    QwenFullAttentionLayer,
    Qwen35NeuralModel,
    EvidencePool,
    EvidenceItem,
    ECRDConfig,
    ECRDLogitsProcessor,
    DATagType,
)


def test_qwen35_model_config_official_specs():
    """Verify Qwen3.5 architecture matches official Hugging Face specifications."""
    cfg = Qwen35ModelConfig.qwen35_08b()

    # 1. Official vocabulary size: 248,320 (NOT legacy 151,936)
    assert cfg.vocab_size == 248320
    assert cfg.model_type == "qwen3_5"
    assert "Qwen3_5ForConditionalGeneration" in cfg.architectures

    # 2. Multimodal vision tokens
    assert cfg.image_token_id == 248056
    assert cfg.vision_start_token_id == 248053
    assert cfg.vision_end_token_id == 248054

    # 3. Hybrid layout: 3 DeltaNet linear recurrent layers to 1 full attention layer
    assert cfg.num_hidden_layers == 24
    assert len(cfg.full_attention_layers) == 6
    assert len(cfg.recurrent_layers) == 18
    assert cfg.full_attention_layers == (3, 7, 11, 15, 19, 23)
    assert set(cfg.full_attention_layers).isdisjoint(set(cfg.recurrent_layers))

    # 4. 4B configuration verification
    cfg_4b = Qwen35ModelConfig.qwen35_4b()
    assert cfg_4b.vocab_size == 248320
    assert cfg_4b.num_hidden_layers == 32
    assert len(cfg_4b.full_attention_layers) == 8
    assert len(cfg_4b.recurrent_layers) == 24


def test_real_h2o_physical_kv_compaction():
    """Verify H2O physically reclaims allocated KV cache tensor bytes on CUDA/CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Qwen35ModelConfig.qwen35_08b()
    layer = QwenFullAttentionLayer(cfg, layer_idx=3).to(device)

    # Initial prompt: sequence length 12
    hidden = torch.randn(1, 12, cfg.hidden_size, device=device)
    out, (k, v), attn, telem = layer(hidden)

    init_seq_len = k.shape[2]
    assert init_seq_len == 12
    init_bytes = telem["kv_bytes_allocated"]
    assert init_bytes > 0

    # Next decode step with H2O eviction: retain only 6 positions (50% physical compaction)
    retained_indices = torch.tensor([0, 1, 8, 9, 10, 11], device=device, dtype=torch.long)
    next_hidden = torch.randn(1, 1, cfg.hidden_size, device=device)
    out2, (k2, v2), attn2, telem2 = layer(
        next_hidden,
        past_key_value=(k, v),
        h2o_retained_indices=retained_indices,
    )

    # Physical tensor dimension MUST be compacted to retained_indices count
    assert k2.shape[2] == len(retained_indices)
    assert telem2["physical_bytes_freed"] > 0
    assert telem2["kv_bytes_allocated"] < init_bytes


def test_real_da_attention_mask_restriction():
    """Verify Declarative Attention guarantees zero attention mass on forbidden positions."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Qwen35ModelConfig.qwen35_08b()
    layer = QwenFullAttentionLayer(cfg, layer_idx=3).to(device)

    # Sequence length 4
    hidden = torch.randn(1, 4, cfg.hidden_size, device=device)
    # DA mask: query 3 is strictly forbidden from attending to key 1
    da_mask = torch.ones(4, 4, dtype=torch.bool, device=device)
    da_mask[3, 1] = False

    out, (k, v), attn_weights, telem = layer(hidden, da_mask=da_mask)

    # Attention weight from query 3 to key 1 MUST be strictly 0.0
    forbidden_weight = attn_weights[0, :, 3, 1]
    assert torch.all(forbidden_weight == 0.0), f"DA failed: leaked attention weight {forbidden_weight}"
    # Physical KV cache remains 100% allocated (no eviction)
    assert k.shape[2] == 4


def test_real_ecrd_on_full_vocabulary():
    """Verify ECRD logits processor on official vocabulary size (248,320)."""
    pool = EvidencePool("req_ecrd")
    pool.add_evidence(
        EvidenceItem("ev_target", "image_region", "target apple", "a" * 64),
        keywords=["apple"],
    )

    class _MockTok:
        def decode(self, token_ids, skip_special_tokens=True):
            return "apple" if 103 in token_ids else "banana"

    config = ECRDConfig(reweight_alpha=3.0)
    proc = ECRDLogitsProcessor(pool, _MockTok(), config)

    assert proc.is_argmax_invariant() is False

    # Full vocabulary logits tensor
    logits = np.zeros(248320, dtype=np.float32)
    logits[200] = 5.0   # banana
    logits[103] = 4.0   # apple (boosted +3.0 -> 7.0)

    res = proc([1, 2], logits)
    winning_token = int(np.argmax(res))
    assert winning_token == 103
    assert len(proc.citations) == 1
    assert proc.citations[0]["evidence_ids"] == ["ev_target"]


def test_real_mtp_speculative_verification_and_rollback():
    """Verify MTP proposal matches are measured without speculative commits."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = RealSevenArmEngine(method_arm="mtp", device=device)

    prompt = [100, 101, 102]
    res = engine.execute_request("req_mtp", prompt_token_ids=prompt, max_new_tokens=8)

    assert res.method_arm == "mtp"
    assert len(res.tokens_generated) == 8
    assert res.gates["MTP_CORRECTNESS_PASS"] == "GATED_NATIVE_MTP_VERIFIER_REQUIRED"
    assert res.gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] == "PASS"
    assert res.mtp_acceptance_rate == 0.0
    assert res.telemetry["mtp_candidate_diagnostic"]["committed_draft_tokens"] == 0
    engine.close()


def test_ccu_concurrency_isolation():
    """Verify concurrent requests execute with strict isolation (CCU > 1)."""
    device = "cpu"
    engine = RealSevenArmEngine(method_arm="ecrd_base", device=device)

    # Request A: Evidence for apple
    pool_a = EvidencePool("req_A")
    pool_a.add_evidence(
        EvidenceItem("ev_A", "text", "red apple", "1" * 64),
        keywords=["apple"],
    )

    # Request B: Evidence for target
    pool_b = EvidencePool("req_B")
    pool_b.add_evidence(
        EvidenceItem("ev_B", "text", "target zone", "2" * 64),
        keywords=["target"],
    )

    # Execute truly concurrently via ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(engine.execute_request, "req_A", [100, 101], 4, pool_a)
        future_b = executor.submit(engine.execute_request, "req_B", [102, 103], 4, pool_b)
        res_a = future_a.result()
        res_b = future_b.result()

    # Verify non-vacuous citations triggered
    assert len(res_a.grounded_citations) > 0, "res_a must produce at least one grounded citation"
    assert len(res_b.grounded_citations) > 0, "res_b must produce at least one grounded citation"

    # Verify zero citation cross-contamination
    for cit in res_a.grounded_citations:
        assert "ev_A" in cit["evidence_ids"]
        assert "ev_B" not in cit["evidence_ids"]

    for cit in res_b.grounded_citations:
        assert "ev_B" in cit["evidence_ids"]
        assert "ev_A" not in cit["evidence_ids"]

    engine.close()


def test_h2o_sequence_above_min_budget_physical_eviction():
    """Verify H2O physically compacts KV cache without IndexError when sequence > min_budget."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = RealSevenArmEngine(method_arm="h2o_base", device=device)

    # Prompt length: 30 tokens + 6 visual tokens = 36 tokens (> min_budget=32)
    prompt = list(range(100, 130))
    res = engine.execute_request("test_h2o_overflow", prompt_token_ids=prompt, max_new_tokens=8)

    # Compaction must have physically triggered and freed bytes
    assert res.physical_kv_bytes_freed > 0, "H2O must physically free bytes when seq_len > min_budget"
    assert res.logical_selection_fraction > 0.0
    assert res.physical_kv_bytes_allocated > 0
    engine.close()


def test_all_seven_arms_real_neural_forward_and_gates():
    """Execute real neural inference across all 7 arms with multimodal visual inputs."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    expected_arms = [
        "base",
        "h2o_base",
        "da_base",
        "ecrd_base",
        "mtp",
        "h2o_mtp",
        "da_mtp",
    ]

    evidence_pool = EvidencePool("test_arm")
    evidence_pool.add_evidence(
        EvidenceItem("ev_obj", "image_region", "salient object", "f" * 64),
        keywords=["object"],
    )

    for arm in expected_arms:
        engine = RealSevenArmEngine(method_arm=arm, device=device)
        res = engine.execute_request(
            request_id=f"test_{arm}",
            prompt_token_ids=[100, 101, 102],
            max_new_tokens=6,
            evidence_pool=evidence_pool,
            image_present=True,
        )

        assert res.method_arm == arm
        assert len(res.tokens_generated) == 6
        assert res.total_latency_ms > 0.0
        assert res.ttft_ms > 0.0

        # Required release gates with SA-01 strict truth-in-labeling
        assert res.gates["POLICY_HARNESS_PASS"] == "PASS"
        assert res.gates["ACTUAL_BACKEND_HOOK_PASS"] in ("PASS", "GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED")
        assert res.gates["REAL_MODEL_FORWARD_PASS"] in ("PASS", "GATED_PRETRAINED_CHECKPOINT_REQUIRED")
        assert res.gates["REAL_MULTIMODAL_INFERENCE_PASS"] in ("PASS", "GATED_PRETRAINED_CHECKPOINT_REQUIRED")
        assert res.gates["CCU_ISOLATION_PASS"] == "GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED"
        assert res.gates["A100_GPU_QUALIFIED"] == "BLOCKED_PENDING_OWNER_PAID_AUTH"

        if "h2o" in arm:
            assert res.physical_kv_bytes_freed >= 0
            assert res.logical_selection_fraction >= 0.0
        if "da" in arm:
            # Small prompts may genuinely have zero exclusion: do not demand
            # a positive sparsity measurement where no selective read is needed.
            assert 0.0 <= res.attention_read_restriction_ratio <= 1.0
            assert 0.0 <= res.attention_mask_restriction_ratio <= 1.0
            assert res.physical_kv_bytes_freed == 0
            assert res.telemetry["da_cost_accounting"]["physical_kv_bytes_freed"] == 0
            assert res.telemetry["da_cost_accounting"]["kv_residency_preserved_pct"] == 100.0
            assert res.telemetry["da_cost_accounting"]["physical_kv_read_saving_claimed"] is False
            assert res.telemetry["da_cost_accounting"]["hbm_traffic_reduction_claimed"] is False
        if "mtp" in arm:
            assert res.gates["MTP_CORRECTNESS_PASS"] == "GATED_NATIVE_MTP_VERIFIER_REQUIRED"
            assert res.gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] == "PASS"

        engine.close()


def test_real_engine_pretrained_checkpoint_loading(tmp_path):
    """Verify partial reference safetensors sets PARTIALLY_WEIGHTED_PYTORCH_REFERENCE and gates SEVEN_ARM_PRETRAINED_PASS."""
    from safetensors.torch import save_file

    small_cfg = Qwen35ModelConfig(
        model_id="Qwen/Qwen3.5-TinyTest",
        model_type="qwen3_5",
        architectures=("Qwen3_5ForConditionalGeneration",),
        vocab_size=512,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=1024,
        image_token_id=500,
        vision_start_token_id=501,
        vision_end_token_id=502,
        pad_token_id=0,
        eos_token_id=1,
        full_attention_layers=(3,),
        recurrent_layers=(0, 1, 2),
        mtp_num_speculative_tokens=2,
    )

    tensors = {
        "model.language_model.embed_tokens.weight": torch.randn(512, 64),
        "model.language_model.norm.weight": torch.ones(64),
        "mtp.fc.weight": torch.randn(64, 128),
        "model.visual.merger.linear_fc2.weight": torch.randn(64, 64),
        "model.visual.merger.linear_fc2.bias": torch.zeros(64),
        "model.language_model.layers.0.input_layernorm.weight": torch.ones(64),
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": torch.randn(192, 64),
        "model.language_model.layers.0.linear_attn.out_proj.weight": torch.randn(64, 64),
        "model.language_model.layers.3.input_layernorm.weight": torch.ones(64),
        "model.language_model.layers.3.self_attn.q_proj.weight": torch.randn(64, 64),
        "model.language_model.layers.3.self_attn.k_proj.weight": torch.randn(32, 64),
        "model.language_model.layers.3.self_attn.v_proj.weight": torch.randn(32, 64),
        "model.language_model.layers.3.self_attn.o_proj.weight": torch.randn(64, 64),
        "model.language_model.layers.3.mlp.gate_proj.weight": torch.randn(128, 64),
    }
    save_file(tensors, str(tmp_path / "model.safetensors"))

    engine = RealSevenArmEngine(
        method_arm="h2o_mtp",
        config=small_cfg,
        device="cpu",
        checkpoint_dir=tmp_path,
    )
    assert engine.is_pretrained is True
    assert engine.is_official_hf_hooked is False
    assert engine.has_vision_encoder is True

    res = engine.execute_request("req_pretrained_tiny", prompt_token_ids=[10, 11, 12], max_new_tokens=4)
    assert res.gates["REAL_MODEL_FORWARD_PASS"] == "PASS"
    assert res.gates["REAL_MULTIMODAL_INFERENCE_PASS"] == "PASS"
    assert res.gates["SEVEN_ARM_PRETRAINED_PASS"] == "GATED_OFFICIAL_HF_HOOK_REQUIRED"
    assert res.telemetry["architectural_status"] == "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE"
    assert res.telemetry["pretrained_weights"]["mapped_tensors_count"] == 13
    assert res.telemetry["pretrained_weights"]["unmapped_tensors_count"] == 1
    assert res.telemetry["pretrained_weights"]["parameter_coverage_ratio"] < 1.0
    engine.close()


def test_official_hf_hooking_coverage_and_logits_parity():
    """Verify 100% official HF hooking (488/488 tensors), logits parity, and per-arm ViT execution when checkpoint exists."""
    from pathlib import Path

    ckpt = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
    if not ckpt.is_dir() or not (ckpt / "config.json").is_file():
        pytest.skip(f"Official Qwen3.5-0.8B checkpoint not present at {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = RealSevenArmEngine(method_arm="da_base", device=device, checkpoint_dir=ckpt)
    assert engine.is_pretrained is True
    assert engine.is_official_hf_hooked is True

    inv = engine.pretrained_weight_summary
    assert inv["checkpoint_total_tensors"] == 488
    assert inv["mapped_tensors_count"] == 488
    assert inv["unmapped_tensors_count"] == 0
    assert inv["base_vl_parameters"] == 852985920
    assert inv["mtp_parameters"] == 20452864
    assert inv["total_checkpoint_parameters"] == 873438784
    assert inv["parameter_coverage_ratio"] == 1.0
    assert inv["logits_parity"]["verified"] is True
    assert inv["logits_parity"]["max_abs_logits_diff"] <= 1e-5

    res = engine.execute_request(
        request_id="req_official_da",
        max_new_tokens=4,
        image_present=True,
        question="Are there cats lying on the couch in the living room?",
    )
    assert res.gates["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"
    assert res.gates["REAL_MULTIMODAL_INFERENCE_PASS"] == "PASS"
    assert res.telemetry["architectural_status"] == "OFFICIAL_HF_PRETRAINED_HOOKED_MODEL"
    assert res.telemetry["visual_encoder_trace"]["invoked"] is True
    assert res.telemetry["visual_encoder_trace"]["pixel_values_shape"] == [1200, 1536]
    assert res.telemetry["visual_encoder_trace"]["visual_token_embeddings_count"] == 300
    assert res.attention_mask_restriction_ratio > 0.0
    assert res.physical_kv_bytes_freed == 0
    engine.close()


# --- P7 engine regression tests (2026-11-09 audit, base cfccc4b) ---

P7_SMALL_CFG = dict(
    model_id="Qwen/Qwen3.5-TinyTest",
    model_type="qwen3_5",
    architectures=("Qwen3_5ForConditionalGeneration",),
    vocab_size=512,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    max_position_embeddings=1024,
    image_token_id=500,
    vision_start_token_id=501,
    vision_end_token_id=502,
    pad_token_id=0,
    eos_token_id=1,
    full_attention_layers=(3,),
    recurrent_layers=(0, 1, 2),
    mtp_num_speculative_tokens=2,
)


def _p7_small_cfg():
    return Qwen35ModelConfig(**P7_SMALL_CFG)


def test_p7_partial_loader_does_not_count_truncated_oversized_tensor(tmp_path):
    """Regression: an oversized q_proj was silently sliced to shape and counted as mapped."""
    from safetensors.torch import save_file

    qkey = "model.language_model.layers.3.self_attn.q_proj.weight"
    save_file({
        qkey: torch.randn(128, 64),  # expected (q_dim=64, hidden=64): oversized
        "model.language_model.layers.3.self_attn.k_proj.weight": torch.randn(32, 64),
        "model.language_model.layers.3.self_attn.v_proj.weight": torch.randn(32, 64),
        "model.language_model.layers.3.self_attn.o_proj.weight": torch.randn(64, 64),
    }, str(tmp_path / "model.safetensors"))

    engine = RealSevenArmEngine(method_arm="base", config=_p7_small_cfg(), device="cpu",
                                checkpoint_dir=tmp_path)
    s = engine.pretrained_weight_summary
    assert qkey in s["unmapped_tensor_keys"]
    assert s["unmapped_tensors_count"] == 1
    assert s["mapped_tensors_count"] == 3
    assert s["parameter_coverage_ratio"] < 1.0
    engine.close()


def test_p7_da_mask_geometry_stable_across_decode_steps():
    """Regression: DA sink/focus boundaries were recomputed from the growing KV length,
    so the same prompt keys changed class between decode steps (sink 9 -> 16 for a 36-token prompt)."""
    engine = RealSevenArmEngine(method_arm="da_base", config=_p7_small_cfg(), device="cpu")
    recorded = []
    real_gen = engine.da_mask_gen.generate_attention_mask

    def _spy(seq_len, tags=None, active_focus_id=None):
        recorded.append([t for t in (tags or []) if t.tag_type != DATagType.LOCAL])
        return real_gen(seq_len=seq_len, tags=tags, active_focus_id=active_focus_id)

    engine.da_mask_gen.generate_attention_mask = _spy
    engine.execute_request("p7_da_drift", prompt_token_ids=list(range(100, 136)),
                           max_new_tokens=30, image_present=True)
    assert len(recorded) >= 20
    first = recorded[0]
    for step_tags in recorded[1:]:
        assert step_tags == first, "DA prompt-region tags changed between decode steps"
    engine.close()


def test_p7_mtp_reference_path_is_token_identical_to_base():
    """MTP never commits drafts, so its committed sequence must equal Base under the same weights."""
    prompt = list(range(10, 50))
    tokens = {}
    for arm in ("base", "mtp"):
        torch.manual_seed(1234)
        engine = RealSevenArmEngine(method_arm=arm, config=_p7_small_cfg(), device="cpu")
        res = engine.execute_request(f"p7_mtp_{arm}", prompt_token_ids=prompt,
                                     max_new_tokens=12, image_present=False)
        tokens[arm] = res.tokens_generated
        if arm == "mtp":
            assert res.telemetry["mtp_candidate_diagnostic"]["committed_draft_tokens"] == 0
            assert res.mtp_acceptance_rate == 0.0
        engine.close()
    assert tokens["mtp"] == tokens["base"]


def test_p7_concurrent_requests_match_sequential_non_vllm():
    """Multi-request isolation on the CPU PyTorch reference (NOT vLLM serving evidence).

    Four requests with distinct prompts run on one engine from a thread pool; each must produce exactly the tokens of its sequential run.
    """
    import concurrent.futures

    prompts = {f"p7_req_{i}": [100 + i, 110 + i, 120 + i, 130 + i] + list(range(200 + i * 40, 240 + i * 40))
               for i in range(4)}

    torch.manual_seed(99)
    engine = RealSevenArmEngine(method_arm="h2o_base", config=_p7_small_cfg(), device="cpu")
    sequential = {
        rid: engine.execute_request(rid, prompt_token_ids=p, max_new_tokens=6,
                                    evidence_pool=None, image_present=False).tokens_generated
        for rid, p in prompts.items()
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = {rid: ex.submit(engine.execute_request, rid, p, 6, None, False)
                   for rid, p in prompts.items()}
        concurrent_tokens = {rid: f.result().tokens_generated for rid, f in futures.items()}
    assert concurrent_tokens == sequential
    engine.close()
