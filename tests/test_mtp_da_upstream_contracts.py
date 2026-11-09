"""Source-alignment regression tests for Qwen MTP and DA diagnostics."""

from pathlib import Path

from vlmbench.infrastructure.real_engine import RealSevenArmEngine


def test_mtp_cannot_commit_unverified_drafts_or_fake_acceptance():
    engine = RealSevenArmEngine(method_arm="mtp", device="cpu")
    try:
        result = engine.execute_request(
            request_id="mtp_source_alignment",
            prompt_token_ids=[100, 101, 102, 103],
            image_present=False,
            max_new_tokens=6,
        )
        diagnostic = result.telemetry["mtp_candidate_diagnostic"]
        assert diagnostic["proposals_verified"] == 5
        assert 0 <= diagnostic["proposals_matching_target"] <= 5
        assert diagnostic["committed_draft_tokens"] == 0
        assert diagnostic["speculative_speedup_claimed"] is False
        assert result.mtp_acceptance_rate == 0.0
        assert result.gates["MTP_CORRECTNESS_PASS"] == "GATED_NATIVE_MTP_VERIFIER_REQUIRED"
        assert result.gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] == "PASS"
    finally:
        engine.close()


def test_mtp_diagnostic_is_token_equivalent_to_greedy_base():
    outputs = {}
    for arm in ("base", "mtp"):
        engine = RealSevenArmEngine(method_arm=arm, device="cpu")
        try:
            outputs[arm] = engine.execute_request(
                request_id=f"match_{arm}",
                prompt_token_ids=[100, 101, 102, 103],
                image_present=False,
                max_new_tokens=6,
            ).tokens_generated
        finally:
            engine.close()
    assert outputs["base"] == outputs["mtp"]


def test_da_does_not_fit_target_sparsity():
    source = (Path(__file__).resolve().parents[1] / "src/vlmbench/infrastructure/real_engine.py").read_text()
    assert "round(0.1895 *" not in source
    assert "step % 2" not in source
    assert "SCRIPTED_SEGMENT_DIAGNOSTIC_NOT_MODEL_DECLARED" in source
    engine = RealSevenArmEngine(method_arm="da_base", device="cpu")
    try:
        result = engine.execute_request(
            request_id="da_scripted_diagnostic",
            prompt_token_ids=list(range(100, 125)),
            image_present=False,
            max_new_tokens=6,
        )
        assert result.telemetry["da_cost_accounting"]["original_paper_protocol_qualified"] is False
        assert result.telemetry["da_cost_accounting"]["physical_kv_read_saving_claimed"] is False
        assert result.physical_kv_bytes_freed == 0
    finally:
        engine.close()
