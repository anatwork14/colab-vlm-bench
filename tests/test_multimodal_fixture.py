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
    CANONICAL_PANOPTIC_SHA256,
    get_supplementary_evaluation_suite,
)
from vlmbench.qualification import get_qualification_tokenizer


@pytest.fixture
def canonical_fixture():
    fixture_dir = Path(__file__).parent / "fixtures"
    return CanonicalMultimodalFixture.default_fixture(fixture_dir)


@pytest.fixture
def model_dir():
    p = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
    if not p.is_dir() or not (p / "config.json").is_file():
        pytest.skip(f"Pretrained checkpoint not present at {p}")
    return p


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
    assert 248056 in token_ids  # Verify Qwen3.5 vision token ID


def test_progressive_qualification_test_c_multimodal_execution(model_dir, canonical_fixture):
    """Progressive Qualification Test C: Multimodal Execution on Real Pretrained Neural Engine."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = get_qualification_tokenizer(model_dir, require_real=True)
    engine = RealSevenArmEngine(method_arm="base", device=device, checkpoint_dir=model_dir)
    assert engine.is_pretrained is True
    assert engine.is_official_hf_hooked is True
    assert engine.has_vision_encoder is True

    report = canonical_fixture.run_test_c_qualification(
        engine=engine,
        tokenizer=tok,
        request_id="test_c_qual_run",
        max_new_tokens=4,
    )

    assert report["test_tier"] == "TEST_C_MULTIMODAL_QUALIFICATION"
    assert report["status"] == "QUALIFIED_PRETRAINED"
    assert report["image_sha256"] == CANONICAL_IMAGE_SHA256
    assert len(report["tokens_generated"]) == 4
    assert report["total_latency_ms"] > 0.0
    assert report["ttft_ms"] > 0.0
    assert report["visual_encoder_trace"]["invoked"] is True
    assert report["visual_encoder_trace"]["pixel_values_shape"] == [1200, 1536]
    assert report["visual_encoder_trace"]["visual_token_embeddings_count"] == 300
    assert report["gates"]["POLICY_HARNESS_PASS"] == "PASS"
    assert report["gates"]["REAL_MODEL_FORWARD_PASS"] == "PASS"
    assert report["gates"]["REAL_MULTIMODAL_INFERENCE_PASS"] == "PASS"
    assert report["gates"]["SEVEN_ARM_PRETRAINED_PASS"] == "PASS"
    # A single request cannot certify concurrent isolation (see real_engine.py); the
    # multi-request CPU check lives in tests/test_real_engine.py (non-vLLM evidence).
    assert report["gates"]["CCU_ISOLATION_PASS"] == "GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED"

    engine.close()


def test_panoptic_mask_integrity(canonical_fixture):
    """Verify the canonical panoptic mask fixture exists and matches SHA-256."""
    assert canonical_fixture.verify_panoptic_integrity() is True
    assert canonical_fixture.panoptic_sha256 == CANONICAL_PANOPTIC_SHA256


def test_supplementary_evaluation_suite(canonical_fixture):
    """Verify supplementary 4-task evaluation suite generates valid fixtures."""
    suite = get_supplementary_evaluation_suite(canonical_fixture.image_path.parent)
    assert len(suite) == 4
    tasks = [f.task_type for f in suite]
    assert "existence" in tasks
    assert "attribute" in tasks
    assert "relation" in tasks

    # Verify existence negative fixture
    neg_fix = suite[1]
    assert neg_fix.ground_truth_label == "No"
    assert "dog" in neg_fix.keywords

    # Verify attribute fixture
    attr_fix = suite[2]
    assert attr_fix.ground_truth_label == "Yellow"

    # Verify relation fixture
    rel_fix = suite[3]
    assert rel_fix.ground_truth_label == "Yes"
    assert "remote" in rel_fix.keywords


def test_multi_turn_chat_formatting(canonical_fixture, model_dir):
    """Verify multi-turn chat template formatting for context streaming evaluation."""
    tok = get_qualification_tokenizer(model_dir, require_real=True)
    prompt_str, token_ids = canonical_fixture.format_multi_turn_qwen_chat(tok)
    assert "<image>" in prompt_str
    assert "Are there remote controls on the couch as well?" in prompt_str
    assert len(token_ids) > len(canonical_fixture.format_qwen_chat_prompt(tok)[1])
    assert 248056 in token_ids
