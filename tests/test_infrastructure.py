"""Tests for Seven-Arm Multimodal Infrastructure (H2O, DA, ECRD, Adapters, Hardware Gate).

Validates:
1. H2O KV-cache eviction policy, recurrent layer immunity, visual/system token protection under sequence growth.
2. H2O speculative MTP transaction state machine (begin, partial/zero commit, rollback with zero residual state, nested protection).
3. Declarative Attention (DA) FSM parsing, tag normalization, causal 2D read mask generation, and strict exclusion.
4. ECRD logits processor contracts (is_argmax_invariant() callable returning False, citation provenance, winning-token alignment).
5. Hardware gate enforcement (deterministic mocked CPU, sm_61, sm_75, sm_80 profiles; floor lowering prohibited).
6. End-to-end execution of all 7 method arms with telemetry verification under synthetic test harness labeling.
"""
from unittest.mock import patch
import numpy as np
import pytest

from vlmbench.infrastructure import (
    H2OConfig,
    H2OCacheTracker,
    H2OEvictionPolicy,
    H2OMTPTransaction,
    DAConfig,
    DATag,
    DATagType,
    DAFSM,
    DAMaskGenerator,
    DAAttentionModifier,
    ECRDConfig,
    EvidenceItem,
    EvidencePool,
    ECRDLogitsProcessor,
    HardwareGateError,
    SevenArmEngine,
    InfrastructureRunner,
    check_vllm_hardware_compatibility,
)


# --- 1. H2O KV Eviction Tests ---

def test_h2o_score_accumulation_and_budget():
    config = H2OConfig(budget_ratio=0.5, recent_ratio=0.2, min_budget=4, protect_system_tokens=2)
    tracker = H2OCacheTracker(config=config, num_layers=4, num_kv_heads=2)

    # Simulate 10 tokens of attention
    seq_len = 10
    weights = np.ones((2, 1, seq_len), dtype=np.float32)
    tracker.update_attention_scores(0, weights)

    retained, metrics = tracker.compute_eviction_plan(0, current_length=seq_len)
    assert len(retained) == seq_len
    # 50% budget of 10 tokens = 5 retained tokens
    assert metrics["retained_tokens"] == 5
    assert metrics["evicted_tokens"] == 5
    # First 2 system tokens permanently protected
    assert retained[0] and retained[1]
    # Recent tokens retained
    assert retained[-1]
    assert "logical_selection_fraction" in metrics
    assert metrics["kv_bytes_freed"] == "NOT_MEASURED_LOGICAL_MASK_ONLY"


def test_h2o_recurrent_layer_immunity():
    tracker = H2OCacheTracker(num_layers=4, num_kv_heads=1)
    tracker.register_recurrent_layer(1)  # DeltaNet linear layer

    weights = np.ones((1, 1, 8), dtype=np.float32)
    tracker.update_attention_scores(1, weights)

    # Recurrent layer should have zero scores accumulated (bypassed)
    assert tracker.layers[1].scores.shape[1] == 0

    retained, metrics = tracker.compute_eviction_plan(1, current_length=8)
    assert np.all(retained)
    assert metrics["retained_tokens"] == 8
    assert metrics["evicted_tokens"] == 0


def test_h2o_visual_token_protection_under_sequence_growth():
    config = H2OConfig(budget_ratio=0.3, recent_ratio=0.1, min_budget=4, protect_visual_tokens=True)
    tracker = H2OCacheTracker(config=config, num_layers=2, num_kv_heads=1)

    # Initial mask defines 20 positions with tokens 5..9 as visual
    visual_mask = [False] * 20
    for i in range(5, 10):
        visual_mask[i] = True
    tracker.set_visual_tokens(visual_mask)

    # SA-03 probe: Sequence grows past 20 to 25 positions
    grown_seq_len = 25
    weights = np.random.RandomState(42).rand(1, 1, grown_seq_len).astype(np.float32)
    tracker.update_attention_scores(0, weights)

    retained, metrics = tracker.compute_eviction_plan(0, current_length=grown_seq_len)
    # All visual tokens 5..9 MUST remain protected despite sequence growth
    for i in range(5, 10):
        assert retained[i], f"Visual token at index {i} was evicted after sequence growth to {grown_seq_len}!"


def test_h2o_mtp_transaction_lifecycle_and_residual_state():
    tracker = H2OCacheTracker(num_layers=2, num_kv_heads=1)
    weights = np.ones((1, 1, 4), dtype=np.float32)
    tracker.update_attention_scores(0, weights)

    orig_scores = np.copy(tracker.layers[0].scores)
    assert tracker.seq_len == 4

    mtp_tx = H2OMTPTransaction(tracker)

    # 1. Begin transaction
    mtp_tx.begin()
    assert mtp_tx.in_transaction

    # Nested begin must be blocked
    with pytest.raises(RuntimeError, match="Nested MTP transactions not allowed"):
        mtp_tx.begin()

    # Speculative proposal: propose 4 tokens (seq_len becomes 8)
    spec_weights = np.ones((1, 1, 8), dtype=np.float32) * 5.0
    tracker.update_attention_scores(0, spec_weights)
    assert tracker.seq_len == 8

    # 2. Rollback when rejected
    mtp_tx.rollback()
    assert not mtp_tx.in_transaction
    assert tracker.seq_len == 4
    np.testing.assert_array_equal(tracker.layers[0].scores[:, :4], orig_scores)

    # 3. Partial commit (SA-04 probe): proposing length 4->8 and committing 1 token
    mtp_tx.begin()
    tracker.update_attention_scores(0, spec_weights)
    mtp_tx.commit(accepted_tokens=1)
    # New sequence length must be exactly 4 + 1 = 5, not 8!
    assert tracker.seq_len == 5
    # Scores beyond index 5 must be zeroed out
    assert np.all(tracker.layers[0].scores[:, 5:] == 0.0)

    # 4. Commit 0 tokens triggers clean rollback
    mtp_tx.begin()
    tracker.update_attention_scores(0, np.ones((1, 1, 7), dtype=np.float32))
    mtp_tx.commit(accepted_tokens=0)
    assert tracker.seq_len == 5


def test_h2o_per_layer_visual_mask_survives_different_compaction_indices():
    config = H2OConfig(
        budget_ratio=0.25, recent_ratio=0.2,
        min_budget=1, protect_system_tokens=1, protect_visual_tokens=True,
    )
    tracker = H2OCacheTracker(config=config, num_layers=2, num_kv_heads=1)
    original_visual = np.array([False, True, True, False, False, False, False, False])
    tracker.set_visual_tokens(original_visual)

    # Attention layers retain different index maps; image positions move to
    # different physical indices. Global original mask must never be rewritten.
    tracker.compact_layer_state(0, np.array([0, 1, 3, 5, 7]))
    tracker.compact_layer_state(1, np.array([0, 2, 4, 6, 7]))
    np.testing.assert_array_equal(tracker.visual_token_mask, original_visual)
    assert tracker.layers[0].visual_token_mask.tolist() == [False, True, False, False, False]
    assert tracker.layers[1].visual_token_mask.tolist() == [False, True, False, False, False]

    # Next decode appends one nonvisual token per layer, then plans eviction.
    for layer_idx in (0, 1):
        retained, _ = tracker.compute_eviction_plan(layer_idx, current_length=6)
        assert retained[1], f"Layer {layer_idx} lost protected visual evidence after compaction"
        assert tracker.layers[layer_idx].visual_token_mask[-1] == False

    # Speculative rollback restores each layer's coordinates independently.
    tx = H2OMTPTransaction(tracker)
    tx.begin()
    tracker.compact_layer_state(0, np.array([0, 2, 4]))
    tracker.compact_layer_state(1, np.array([0, 1, 3, 5]))
    tx.rollback()
    for layer_idx in (0, 1):
        assert tracker.layers[layer_idx].visual_token_mask[1]
        assert len(tracker.layers[layer_idx].visual_token_mask) == 6


# --- 2. Declarative Attention (DA) Tests ---

def test_da_fsm_tag_parsing():
    fsm = DAFSM()
    raw_prompt = (
        "<da:global>Always read this context.</da:global> "
        "Normal text here. "
        "<da:focus id=\"img1\">Image caption and details.</da:focus> "
        "<da:local window=\"32\">Sliding window paragraph.</da:local>"
    )
    clean_text, spans = fsm.parse_text(raw_prompt)

    assert "<da:" not in clean_text
    assert "</da:" not in clean_text
    assert len(spans) == 3

    types = [s[2] for s in spans]
    assert types == [DATagType.GLOBAL, DATagType.FOCUS, DATagType.LOCAL]
    assert spans[1][3]["id"] == "img1"
    assert spans[2][3]["window"] == 32


def test_da_fsm_malformed_and_unclosed_tags():
    fsm = DAFSM()
    raw = "<da:global>System instructions without closing tag"
    clean, spans = fsm.parse_text(raw)
    assert len(spans) == 1
    assert spans[0][2] == DATagType.GLOBAL
    assert clean == "System instructions without closing tag"


def test_da_attention_mask_invariants():
    generator = DAMaskGenerator(DAConfig(default_local_window=2))
    seq_len = 6
    tags = [
        DATag(DATagType.GLOBAL, 0, 1),
        DATag(DATagType.FOCUS, 1, 3, focus_id="A"),
        DATag(DATagType.LOCAL, 3, 6, window=1),
    ]
    mask = generator.generate_attention_mask(seq_len=seq_len, tags=tags)

    assert mask.shape == (seq_len, seq_len)

    # Invariant 1: Causal masking (strictly lower-triangular + diagonal)
    for q in range(seq_len):
        for k in range(q + 1, seq_len):
            assert not mask[q, k], f"Query {q} attended to future key {k}"

    # Invariant 2: Global token (k=0) is attended to by all queries
    for q in range(seq_len):
        assert mask[q, 0], f"Query {q} failed to attend to global token 0"

    # Invariant 3: Focus isolation
    assert not mask[3, 1]
    assert not mask[3, 2]

    # Invariant 4: Local window 1
    assert mask[5, 5]
    assert mask[5, 4]
    assert not mask[5, 3]

    stats = generator.mask_statistics(mask)
    assert stats["read_reduction_estimate"] > 0.0
    assert stats["kv_cache_preserved_ratio"] == 1.0


def test_da_attention_modifier_strict_exclusion():
    modifier = DAAttentionModifier()
    # SA-07 probe: allowed logit 0.0, forbidden logit 100000.0
    logits = np.array([[[0.0, 100000.0]]], dtype=np.float32)
    mask = np.array([[True, False]])
    mod_logits = modifier.modify_attention_logits(logits, mask)

    # With strict exclusion (-1e9), allowed logit MUST beat the forbidden logit
    assert mod_logits[0, 0, 0] == 0.0
    assert mod_logits[0, 0, 1] <= -1e8
    assert np.argmax(mod_logits[0, 0]) == 0


# --- 3. Grounding Evidence (ECRD) Tests ---

class _MockTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        mapping = {10: "apple", 20: "banana", 30: "orange"}
        return " ".join(mapping.get(t, f"tok_{t}") for t in token_ids)


def test_ecrd_is_argmax_invariant_contract():
    pool = EvidencePool("test")
    processor = ECRDLogitsProcessor(pool, _MockTokenizer())

    # SA-05 contract: is_argmax_invariant() MUST be callable and return False
    assert callable(processor.is_argmax_invariant)
    assert processor.is_argmax_invariant() is False


def test_ecrd_evidence_provenance_and_validation():
    pool = EvidencePool("req_val")

    # 1. Unverified evidence item rejected
    with pytest.raises(ValueError, match="verified=False"):
        pool.add_evidence(
            EvidenceItem("ev_unverified", "text", "bad", "a" * 64, verified=False)
        )

    # 2. Invalid SHA256 rejected
    with pytest.raises(ValueError, match="invalid SHA256"):
        pool.add_evidence(
            EvidenceItem("ev_bad_sha", "text", "bad", "not_a_valid_sha256", verified=True)
        )

    # 3. gold_answer_prohibited=False rejected
    with pytest.raises(ValueError, match="gold_answer_prohibited=False"):
        pool.add_evidence(
            EvidenceItem("ev_leak", "text", "leak", "b" * 64, gold_answer_prohibited=False)
        )


def test_ecrd_citation_alignment_and_request_isolation():
    pool = EvidencePool("req_1")
    pool.add_evidence(
        EvidenceItem("ev_apple", "image_region", "red fruit", "a" * 64),
        keywords=["apple"],
    )

    tokenizer = _MockTokenizer()
    config = ECRDConfig(reweight_alpha=2.0)
    processor = ECRDLogitsProcessor(pool, tokenizer, config)

    # Scenario A: Apple boosted and wins -> citation recorded
    logits = np.zeros(50, dtype=np.float32)
    logits[20] = 5.0  # banana
    logits[10] = 4.0  # apple (+2.0 -> 6.0)
    res = processor([1], logits)
    assert np.argmax(res) == 10
    assert len(processor.citations) == 1
    assert processor.citations[0]["evidence_ids"] == ["ev_apple"]

    # Scenario B (SA-06 probe): Apple boosted (+2.0 -> 3.0), but banana (10.0) still wins
    processor.reset_request(pool)
    logits_b = np.zeros(50, dtype=np.float32)
    logits_b[20] = 10.0  # banana
    logits_b[10] = 1.0   # apple
    res_b = processor([1], logits_b)
    assert np.argmax(res_b) == 20
    # Winning token (banana) has NO evidence -> citations must be EMPTY!
    assert len(processor.citations) == 0

    # Scenario C: Request isolation -> reset_request clears previous state
    processor.reset_request(None)
    assert processor.evidence_pool.total_items == 0
    assert len(processor.citations) == 0


def test_ecrd_nan_safety():
    pool = EvidencePool("nan_test")
    processor = ECRDLogitsProcessor(pool, _MockTokenizer())

    # NaNs or all -inf must raise ValueError cleanly
    with pytest.raises(ValueError, match="NaN"):
        processor([], np.array([np.nan, 1.0, 2.0], dtype=np.float32))

    with pytest.raises(ValueError, match="all -inf"):
        processor([], np.array([-np.inf, -np.inf], dtype=np.float32))


# --- 4. Hardware Gate and Adapters Tests ---

def test_hardware_gate_floor_lowering_prohibited():
    # SA-08: Attempting to lower min_compute_capability below sm_75 must be blocked
    with pytest.raises(ValueError, match="Cannot lower vLLM minimum compute capability"):
        check_vllm_hardware_compatibility(target_backend="vllm", min_compute_capability=(6, 1))

    # Invalid backend name must be rejected
    with pytest.raises(ValueError, match="Unknown target_backend"):
        check_vllm_hardware_compatibility(target_backend="unknown_engine")


def test_hardware_gate_deterministic_mocks(monkeypatch):
    import sys
    # SA-09: Test hardware gate across all architectures deterministically using mocks

    # Case 1: CPU mode (no CUDA detected)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("shutil.which", lambda _: None)
    with pytest.raises(HardwareGateError, match="CUDA is not available"):
        check_vllm_hardware_compatibility(target_backend="vllm")
    cpu_prof = check_vllm_hardware_compatibility(target_backend="pytorch")
    assert cpu_prof["device_name"] == "CPU"

    # Case 2: sm_61 host (GTX 1080 Ti)
    mock_run_sm61 = lambda *args, **kwargs: type("Res", (), {
        "returncode": 0,
        "stdout": "NVIDIA GeForce GTX 1080 Ti, 6.1, 11264 MiB\n",
    })()
    monkeypatch.setattr("subprocess.run", mock_run_sm61)
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/nvidia-smi")

    # vLLM on sm_61 fails closed
    with pytest.raises(HardwareGateError, match="requires compute capability >= sm_75, but detected sm_61"):
        check_vllm_hardware_compatibility(target_backend="vllm")
    # PyTorch on sm_61 succeeds
    sm61_prof = check_vllm_hardware_compatibility(target_backend="pytorch")
    assert sm61_prof["compute_capability"] == (6, 1)
    assert sm61_prof["vllm_binary_compatible"] is False

    # Case 3: sm_80 host (A100 40GB)
    mock_run_a100 = lambda *args, **kwargs: type("Res", (), {
        "returncode": 0,
        "stdout": "NVIDIA A100-SXM4-40GB, 8.0, 40960 MiB\n",
    })()
    monkeypatch.setattr("subprocess.run", mock_run_a100)

    a100_prof = check_vllm_hardware_compatibility(target_backend="vllm")
    assert a100_prof["vllm_binary_compatible"] is True
    assert a100_prof["compute_capability"] == (8, 0)


def test_seven_arm_engine_all_arms_suite():
    runner = InfrastructureRunner(target_backend="pytorch")
    results = runner.run_all_arms_suite(max_tokens=6)

    expected_arms = {
        "base",
        "h2o_base",
        "da_base",
        "ecrd_base",
        "mtp",
        "h2o_mtp",
        "da_mtp",
    }
    assert set(results.keys()) == expected_arms

    for arm, res in results.items():
        assert res.method_arm == arm
        assert len(res.tokens_generated) == 6
        assert res.ttft_ms > 0.0
        assert res.total_latency_ms > 0.0
        assert res.execution_classification == "SYNTHETIC_REFERENCE_POLICY_TEST_HARNESS"
        assert res.qualification_status == "BACKEND_UNQUALIFIED_A100_EXECUTION_REQUIRED"

        if "h2o" in arm:
            assert "final_h2o" in res.verification_trace
            assert res.verification_trace["final_h2o"]["retained_tokens"] > 0
            assert "logical_selection_fraction" in res.verification_trace["final_h2o"]
        if "da" in arm:
            assert "final_da_read_ratio" in res.verification_trace
        if "ecrd" in arm:
            assert "ecrd_citations" in res.verification_trace
            assert "ecrd_auxiliary_calls" in res.verification_trace


def test_h2o_visual_protection_not_miscounted_as_heavy_hitters():
    """Visual protected keys can exceed target budget; none is HH-selected."""
    tracker = H2OCacheTracker(
        H2OConfig(budget_ratio=0.2, recent_ratio=0.25, min_budget=4,
                  protect_system_tokens=2, protect_visual_tokens=True),
        num_layers=2,
        num_kv_heads=1,
    )
    visual = [False] * 20
    visual[5:10] = [True] * 5
    tracker.set_visual_tokens(visual)
    tracker.update_attention_scores(
        0, np.ones((1, 1, 20), dtype=np.float32)
    )
    kept, metrics = tracker.compute_eviction_plan(0, current_length=20)
    assert np.all(kept[5:10]), "Protected visual evidence must stay resident"
    assert kept[0] and kept[1], "Sink tokens must stay resident"
    assert metrics["heavy_hitters"] == 0, (
        "Protected visual tokens cannot be counted as score-selected heavy hitters"
    )
    assert metrics["retained_tokens"] > metrics["heavy_hitters"]


def test_h2o_heavy_hitters_report_only_actual_score_selected_candidates():
    tracker = H2OCacheTracker(
        H2OConfig(budget_ratio=0.5, recent_ratio=0.2, min_budget=4,
                  protect_system_tokens=2, protect_visual_tokens=False),
        num_layers=1,
        num_kv_heads=1,
    )
    tracker.update_attention_scores(
        0, np.arange(10, dtype=np.float32)[None, None, :]
    )
    kept, metrics = tracker.compute_eviction_plan(0, current_length=10)
    # 5 total capacity: 2 sink + 1 recent + 2 score-selected heavy hitters.
    assert metrics["retained_tokens"] == 5
    assert metrics["heavy_hitters"] == 2
    assert int(kept.sum()) == 5


# --- P7 engine regression tests (2026-11-09 audit, base cfccc4b) ---

def test_p7_h2o_mtp_commit_rejects_more_tokens_than_proposed():
    """Regression: commit(accepted) larger than the proposal left seq_len past scored state."""
    tracker = H2OCacheTracker(num_layers=2, num_kv_heads=1)
    tracker.update_attention_scores(0, np.ones((1, 1, 4), dtype=np.float32))
    tx = H2OMTPTransaction(tracker)
    tx.begin()
    tracker.update_attention_scores(0, np.ones((1, 1, 4), dtype=np.float32))  # 4 proposed
    with pytest.raises(ValueError, match="exceed"):
        tx.commit(accepted_tokens=10)
    # Refused commit must leave the transaction open and rollback-able.
    assert tx.in_transaction
    tx.rollback()
    assert tracker.seq_len == 4
    assert tracker.layers[0].scores.shape[1] == 4


@pytest.mark.parametrize("window_attr", ['"-4"', '"abc"'])
def test_p7_da_rejects_negative_or_malformed_local_window(window_attr):
    """Regression: a negative window made every local key unreadable (even by itself),
    and a non-numeric window crashed with an opaque numpy int() error."""
    fsm = DAFSM()
    _, spans = fsm.parse_text(f"<da:local window={window_attr}>x</da:local>")
    tags = fsm.spans_to_token_tags(spans, [(0, 1)])
    with pytest.raises(ValueError, match="window"):
        DAMaskGenerator().generate_attention_mask(seq_len=1, tags=tags)
    with pytest.raises(ValueError, match="window"):
        DAMaskGenerator().generate_attention_mask(
            seq_len=2, tags=[DATag(DATagType.LOCAL, 0, 2, window=-1)])


def test_p7_da_zero_window_keeps_self_attention():
    """Boundary: window=0 is legal and must still let each query read itself (no empty row)."""
    mask = DAMaskGenerator().generate_attention_mask(
        seq_len=3, tags=[DATag(DATagType.LOCAL, 0, 3, window=0)])
    assert np.all(np.diag(mask))
