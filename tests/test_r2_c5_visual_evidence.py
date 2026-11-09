"""ITER3 R2 C-5 (closure doc): image evidence protects NO visual tokens. This must not pass silently.

The grounding signal protects token spans of evidence KEYWORDS in the text prompt only. An image_region
evidence item (bounding box on the source image) has no route to vision tokens in the cache. Telemetry must
say so explicitly: `visual_evidence_protection` is NOT_IMPLEMENTED whenever image evidence is declared,
and the protection function must not claim visual coverage. cpu_pytorch_reference_not_vllm.
"""
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import visual_evidence_protection_status

from test_starlight_oracle import PROMPT, _cfg, _WordTok

SHA = "c" * 64


def _pool(modality, eid="ev_v", keywords=("car",), bbox=None):
    pool = EvidencePool("c5")
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality=modality, content="car", source_sha256=SHA,
                                   verified=True, bounding_box=bbox), keywords=list(keywords))
    return pool


def test_status_is_not_implemented_when_image_region_evidence_is_declared():
    pool = _pool("image_region", bbox=(0.1, 0.1, 0.5, 0.5))
    assert visual_evidence_protection_status(pool) == "NOT_IMPLEMENTED"


def test_status_is_not_implemented_even_without_a_bounding_box():
    assert visual_evidence_protection_status(_pool("image_region")) == "NOT_IMPLEMENTED"


def test_text_only_evidence_declares_no_visual_protection_need():
    assert visual_evidence_protection_status(_pool("text_span")) == "NO_IMAGE_EVIDENCE_DECLARED"


def test_no_pool_declares_nothing():
    assert visual_evidence_protection_status(None) == "NO_EVIDENCE_DECLARED"


def test_engine_telemetry_reports_visual_evidence_not_implemented_and_does_not_claim_coverage():
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="c5", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4, evidence_pool=_pool("image_region",
                                                                           bbox=(0.0, 0.0, 1.0, 1.0)))
    finally:
        engine.close()
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "NOT_IMPLEMENTED"
    assert "visual" in block["visual_evidence_protection_note"].lower()
