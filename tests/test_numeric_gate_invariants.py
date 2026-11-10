"""Request-level gate truth-in-labeling and validation of output-token counts."""

import pytest
import torch

from vlmbench.infrastructure.model_config import Qwen35ModelConfig
from vlmbench.infrastructure.real_engine import RealSevenArmEngine


def tiny_config():
    return Qwen35ModelConfig(
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


@pytest.mark.parametrize("invalid", [0, -1, -256, 1.5, None, False, True, "256"])
def test_invalid_token_count_rejected_before_neural_work(invalid):
    engine = RealSevenArmEngine(method_arm="base", device="cpu", config=tiny_config())
    try:
        with pytest.raises(ValueError, match="max_new_tokens must be a positive integer"):
            engine.execute_request("input_guard", [100, 101, 102], max_new_tokens=invalid,
                                   image_present=False)
    finally:
        engine.close()


@pytest.mark.parametrize("invalid", ["", " ", None, 1])
def test_invalid_request_identifier_rejected(invalid):
    engine = RealSevenArmEngine(method_arm="base", device="cpu", config=tiny_config())
    try:
        with pytest.raises(ValueError, match="request_id must be a non-empty string"):
            engine.execute_request(invalid, [100, 101, 102], max_new_tokens=1,
                                   image_present=False)
    finally:
        engine.close()


def test_single_request_does_not_certify_ccu_isolation():
    torch.manual_seed(11)
    engine = RealSevenArmEngine(method_arm="base", device="cpu", config=tiny_config())
    try:
        res = engine.execute_request("single", [100, 101, 102],
                                     max_new_tokens=1, image_present=False)
        assert len(res.tokens_generated) == 1
        assert res.gates["CCU_ISOLATION_PASS"] == "GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED"
        assert res.gates["VLLM_SCHEDULER_CONCURRENCY_PASS"] == "GATED_NATIVE_VLLM_LOAD_TEST_REQUIRED"
    finally:
        engine.close()
