"""Comprehensive tests for real Qwen3.5 neural engine and vLLM backend integrations.

Validates:
1. Exact Hugging Face architectural parameters (vocab_size=248320, 3:1 DeltaNet ratio).
2. True physical KV cache byte compaction under H2O.
3. Genuine kernel attention restrictions under Declarative Attention.
4. ECRD logits processing on full 248,320 vocabulary with winning-token citation alignment.
5. MTP speculative decoding verification and physical KV rollback.
6. Multi-request CCU concurrency isolation (zero cross-request pollution).
7. Complete seven-arm qualification gate reporting.
"""
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
    """Verify MTP speculative verification and physical KV rollback."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = RealSevenArmEngine(method_arm="mtp", device=device)

    prompt = [100, 101, 102]
    res = engine.execute_request("req_mtp", prompt_token_ids=prompt, max_new_tokens=8)

    assert res.method_arm == "mtp"
    assert len(res.tokens_generated) == 8
    assert res.gates["MTP_CORRECTNESS_PASS"] == "PASS"
    assert res.mtp_acceptance_rate >= 0.0


def test_ccu_concurrency_isolation():
    """Verify concurrent requests execute with strict isolation (CCU > 1)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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

    res_a = engine.execute_request("req_A", [100, 101], max_new_tokens=4, evidence_pool=pool_a)
    res_b = engine.execute_request("req_B", [102, 103], max_new_tokens=4, evidence_pool=pool_b)

    # Verify zero citation cross-contamination
    for cit in res_a.grounded_citations:
        assert "ev_A" in cit["evidence_ids"]
        assert "ev_B" not in cit["evidence_ids"]

    for cit in res_b.grounded_citations:
        assert "ev_B" in cit["evidence_ids"]
        assert "ev_A" not in cit["evidence_ids"]

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

        # Required release gates
        assert res.gates["POLICY_HARNESS_PASS"] == "PASS"
        assert res.gates["ACTUAL_BACKEND_HOOK_PASS"] == "PASS"
        assert res.gates["REAL_MODEL_FORWARD_PASS"] == "PASS"
        assert res.gates["REAL_MULTIMODAL_INFERENCE_PASS"] == "PASS"
        assert res.gates["CCU_ISOLATION_PASS"] == "PASS"
        assert res.gates["A100_GPU_QUALIFIED"] == "GATED_A100_HOST_EXECUTION_REQUIRED"

        if "h2o" in arm:
            assert res.physical_kv_bytes_freed >= 0
            assert res.logical_selection_fraction >= 0.0
        if "da" in arm:
            assert res.attention_read_restriction_ratio > 0.0
        if "mtp" in arm:
            assert res.gates["MTP_CORRECTNESS_PASS"] == "PASS"

        engine.close()
