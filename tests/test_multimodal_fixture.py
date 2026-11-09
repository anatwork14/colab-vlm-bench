"""Tests for Progressive Qualification Test C and CanonicalMultimodalFixture."""
from pathlib import Path
import pytest

from vlmbench.infrastructure import (
    Qwen35ModelConfig,
    RealSevenArmEngine,
)
from vlmbench.multimodal_fixture import (
    CanonicalMultimodalFixture,
    CANONICAL_IMAGE_SHA256,
)
from vlmbench.qualification import get_qualification_tokenizer


@pytest.fixture
def canonical_fixture():
    fixture_dir = Path(__file__).parent / "fixtures"
    return CanonicalMultimodalFixture.default_fixture(fixture_dir)


@pytest.fixture
def model_dir():
    return Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")


def test_fixture_integrity(canonical_fixture):
    """Verify the canonical fixture image exists and matches cryptographic SHA-256."""
    assert canonical_fixture.verify_image_integrity() is True
    assert canonical_fixture.image_sha256 == CANONICAL_IMAGE_SHA256
    assert canonical_fixture.image_size == (640, 480)
    assert canonical_fixture.question == "Are there cats lying on the couch in the living room?"


def test_progressive_qualification_test_a_model_config(model_dir):
    """Progressive Qualification Test A: Architecture & Config."""
    cfg = Qwen35ModelConfig.from_checkpoint(model_dir)
    assert cfg.model_type == "qwen3_5"
    assert "Qwen3_5ForConditionalGeneration" in cfg.architectures
    assert cfg.image_token_id == 248056
    assert cfg.vision_start_token_id == 248053
    assert cfg.vision_end_token_id == 248054
    assert cfg.num_hidden_layers == 24
    assert len(cfg.full_attention_layers) == 6
    assert len(cfg.recurrent_layers) == 18


def test_progressive_qualification_test_b_tokenizer(model_dir, canonical_fixture):
    """Progressive Qualification Test B: Tokenizer & Chat Template."""
    tok = get_qualification_tokenizer(model_dir, require_real=True)
    assert tok.is_synthetic is False
    assert tok.tokenizer_type == "pretrained_model"
    assert len(tok) == 248077

    # Format grounded prompt with chat template
    prompt_str, token_ids = canonical_fixture.format_qwen_chat_prompt(tok)
    assert "<image>" in prompt_str
    assert "Are there cats lying on the couch in the living room?" in prompt_str
    assert len(token_ids) > 0


def test_progressive_qualification_test_c_multimodal_execution(model_dir, canonical_fixture):
    """Progressive Qualification Test C: Multimodal Execution on Real Neural Engine."""
    tok = get_qualification_tokenizer(model_dir, require_real=True)
    engine = RealSevenArmEngine(method_arm="base", device="cpu")

    report = canonical_fixture.run_test_c_qualification(
        engine=engine,
        tokenizer=tok,
        request_id="test_c_qual_run",
        max_new_tokens=4,
    )

    assert report["test_tier"] == "TEST_C_MULTIMODAL_QUALIFICATION"
    assert report["status"] == "QUALIFIED"
    assert report["image_sha256"] == CANONICAL_IMAGE_SHA256
    assert len(report["tokens_generated"]) == 4
    assert report["total_latency_ms"] > 0.0
    assert report["ttft_ms"] > 0.0
    assert report["gates"]["POLICY_HARNESS_PASS"] == "PASS"
    assert report["gates"]["CCU_ISOLATION_PASS"] == "PASS"

    engine.close()
