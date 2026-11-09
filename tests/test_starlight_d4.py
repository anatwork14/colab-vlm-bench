"""ITER2 D4 engine-level checks: structural protection (lead Q2), degraded-arm labels (lead Q4).
cpu_pytorch_reference_not_vllm.
"""

import hashlib

import numpy as np
import torch

from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.h2o import H2OCacheTracker, H2OConfig
from vlmbench.infrastructure.model_config import Qwen35ModelConfig
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

SHA = hashlib.sha256(b"no-image").hexdigest()
VS, VE = 501, 502  # vision_start / vision_end in the tiny config
# 3 is vision_start, 20 is vision_end; 'the' (7) appears at 10 and 12 for grounding
PROMPT = [100, 101, 102, VS, 103, 104, 105, 106, 107, 108, 7, 109, 7, 110, 111, 112, 113, 114, 115,
          116, VE, 117, 118, 119, 120, 121, 122, 123, 124, 125]


class _WordTok:
    VOCAB = {"the": 7}

    def encode(self, text):
        return [self.VOCAB[w] for w in text.lower().split() if w in self.VOCAB]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


def _cfg():
    return Qwen35ModelConfig(
        model_id="Qwen/Qwen3.5-StarlightD4Test", model_type="qwen3_5",
        architectures=("Qwen3_5ForConditionalGeneration",),
        vocab_size=512, hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, max_position_embeddings=1024,
        image_token_id=500, vision_start_token_id=VS, vision_end_token_id=VE,
        pad_token_id=0, eos_token_id=1, full_attention_layers=(3, 7),
        recurrent_layers=(0, 1, 2, 4, 5, 6), mtp_num_speculative_tokens=2,
    )


def _pool(request_id):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id="ev_the", modality="text_span", content="the",
                                   source_sha256=SHA, verified=True), keywords=["the"])
    return pool


def _run(arm, pool=None, tokenizer=None, record=True, budget=16, ratio=0.2, new=8):
    torch.manual_seed(1234)
    engine = RealSevenArmEngine(
        method_arm=arm, device="cpu", config=_cfg(), stop_at_eos=False, h2o_min_budget=budget,
        h2o_budget_ratio=ratio, tokenizer=tokenizer, starlight_record_decisions=record,
    )
    try:
        return engine.execute_request(request_id="d4", prompt_token_ids=PROMPT, image_present=False,
                                      max_new_tokens=new, evidence_pool=pool)
    finally:
        engine.close()


# --- Q2: structural protection ---------------------------------------------------------------

def test_arms_carry_the_structural_source_in_every_starlight_variant():
    for arm, expected in (
        ("starlight_s1", ("da", "grounding", "structural")),
        ("starlight_s1_no_da", ("grounding", "structural")),
        ("starlight_s1_no_grounding", ("da", "structural")),
    ):
        torch.manual_seed(1234)
        engine = RealSevenArmEngine(method_arm=arm, device="cpu", config=_cfg())
        try:
            assert engine.starlight_signals == expected, arm
        finally:
            engine.close()


def test_vision_tokens_are_protected_and_never_evicted_under_pressure():
    s = _run("starlight_s1", pool=_pool("d4"), tokenizer=_WordTok()).telemetry["starlight"]
    assert s["structural_tokens_evicted"] == 0
    assert s["structural_policy"] == "PROTECTED_LEAD_DECISION_Q2"
    assert s["protected_count_by_source"]["structural"] > 0
    assert s["invariant_protected_evicted"] == 0


def test_vision_token_candidate_is_vetoed_at_helper_level():
    torch.manual_seed(1234)
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg())
    try:
        tracker = H2OCacheTracker(config=H2OConfig(), num_layers=8, num_kv_heads=2)
        scores = np.asarray([0.0, 9, 9, 0.0, 9, 9, 9, 9, 9, 9], dtype=np.float32)  # slot 3 lowest
        for layer in (3, 7):
            tracker.update_attention_scores(layer, np.tile(scores, (4, 1, 1)))
        k_lens = {3: 10, 7: 10}
        orig = {3: np.arange(9), 7: np.arange(9)}
        _, _, plans = engine._starlight_build_step_plans(
            tracker, k_lens, orig, 9, 6, [], {}, ("da", "grounding", "structural"),
            structural={3: ("vision_start",)},
        )
        d = plans[0].decisions[3]
        assert d.h2o_candidate and d.decision == "RETAIN_PROTECTED_VETOED_H2O_CANDIDATE"
        assert d.reason_code == "H2O_CANDIDATE_VETOED_BY_STRUCTURAL"
        assert 3 in plans[0].retained
    finally:
        engine.close()


# --- Q4: degraded arm labels and missing-declaration flag ------------------------------------

def test_missing_evidence_declaration_degrades_starlight_s1_to_h2o_plus_da():
    s = _run("starlight_s1", pool=None).telemetry["starlight"]
    assert s["grounding_declaration_missing"] is True
    assert s["arm_status"] == "degraded_to_h2o_plus_da"
    assert s["grounding_signal_status"] == "NO_EVIDENCE_DECLARED"


def test_declared_evidence_with_tokenizer_is_nominal():
    s = _run("starlight_s1", pool=_pool("d4"), tokenizer=_WordTok()).telemetry["starlight"]
    assert s["grounding_declaration_missing"] is False
    assert s["arm_status"] == "nominal"


def test_declared_evidence_without_tokenizer_is_missing_for_the_signal_and_degrades():
    s = _run("starlight_s1", pool=_pool("d4"), tokenizer=None).telemetry["starlight"]
    assert s["grounding_signal_status"] == "UNAVAILABLE_NO_TOKENIZER"
    assert s["grounding_declaration_missing"] is True
    assert s["arm_status"] == "degraded_to_h2o_plus_da"


def test_no_da_ablation_without_evidence_degrades_to_h2o_only():
    s = _run("starlight_s1_no_da", pool=None).telemetry["starlight"]
    assert s["arm_status"] == "degraded_to_h2o_only"
    assert s["grounding_declaration_missing"] is True


def test_no_grounding_ablation_is_nominal_without_any_evidence():
    s = _run("starlight_s1_no_grounding", pool=None).telemetry["starlight"]
    assert s["arm_status"] == "nominal"
    assert s["grounding_declaration_missing"] is False  # grounding is not part of this arm
