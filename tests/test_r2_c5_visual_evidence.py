"""ITER3 R2 C-5 (closure doc, RE-DERIVED in ITER3 Round B S2): image evidence never passes silently.

Round B replaced the single label NOT_IMPLEMENTED with an explicit per-request status:
  * SUPPORTED: the region-to-visual-position mapping is established for this request (processor geometry and one
    image token run). Protection is still gated per item by S1 (source bytes VERIFIED and bound, location VERIFIED).
  * UNSUPPORTED + unsupported_reason: the mapping cannot be established, so no visual token is protected.
  * NO_EVIDENCE_DECLARED / NO_IMAGE_EVIDENCE_DECLARED: no image_region evidence was declared.

Why the old expectations changed: NOT_IMPLEMENTED was a fixed label for every pool with image evidence. It is
replaced by a status that depends on the request (image present, hash, geometry, token run). A pool with image
evidence and no image in the request is UNSUPPORTED with reason NO_IMAGE_IN_REQUEST: the old test asserted
NOT_IMPLEMENTED for that same input, so its expectation is re-derived to the new reason.

cpu_pytorch_reference_not_vllm. Preserving a visual span does NOT prove the model used it.
"""
from vlmbench.infrastructure import visual_mapping as vm
from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

from test_starlight_oracle import PROMPT, _cfg, _WordTok

SHA = "c" * 64


def _pool(modality, eid="ev_v", keywords=("car",), bbox=None):
    pool = EvidencePool("c5")
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality=modality, content="car", source_sha256=SHA,
                                   verified=True, bounding_box=bbox), keywords=list(keywords))
    return pool


def _status(pool, **overrides):
    # No image in the request: the default context for these C-5 pins.
    args = dict(request_id="c5", image_sha256=None, image_present=False, policy=EvidencePolicy(),
                geometry=None, geometry_reason=vm.REASON_MISSING_GRID, span=None, span_reason=None)
    args.update(overrides)
    return vm.visual_evidence_report(pool, **args)


def test_image_region_evidence_without_an_image_in_the_request_is_unsupported_with_its_reason():
    rep = _status(_pool("image_region", bbox=(0.1, 0.1, 0.5, 0.5)))
    assert rep["status"] == "UNSUPPORTED"
    assert rep["unsupported_reason"] == "NO_IMAGE_IN_REQUEST"
    assert rep["visual_positions_protected"] == 0


def test_unsupported_even_without_a_bounding_box_and_never_claims_coverage():
    rep = _status(_pool("image_region"))
    assert rep["status"] == "UNSUPPORTED"
    assert rep["positions"] == {}


def test_text_only_evidence_declares_no_visual_protection_need():
    assert _status(_pool("text_span"))["status"] == "NO_IMAGE_EVIDENCE_DECLARED"


def test_no_pool_declares_nothing():
    assert _status(None)["status"] == "NO_EVIDENCE_DECLARED"


def test_engine_telemetry_reports_visual_evidence_unsupported_with_reason_and_does_not_claim_coverage():
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="c5", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4, evidence_pool=_pool("image_region",
                                                                           bbox=(0.0, 0.0, 1.0, 1.0)))
    finally:
        engine.close()
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "UNSUPPORTED"
    assert block["visual_evidence_protection_reason"] == "NO_IMAGE_IN_REQUEST"
    assert block["visual_positions_protected"] == 0
    assert "visual" in block["visual_evidence_protection_note"].lower()


# ITER3 R2 round-2 review (auditor wF:p9): the status keys on modality == "image_region". Before this fix any
# other string (e.g. "image") was accepted by add_evidence and the status then read NO_IMAGE_EVIDENCE_DECLARED
# while image evidence was declared. Unknown modalities are now refused at the pool.

def test_an_unknown_modality_is_refused_so_image_evidence_cannot_hide_behind_a_label():
    try:
        _pool("image")
    except ValueError as exc:
        assert "modality" in str(exc)
        return
    raise AssertionError("an unknown modality must be refused, not accepted as text-only evidence")


def test_the_known_modalities_are_still_accepted():
    for modality in ("image_region", "text_span", "table_cell"):
        pool = _pool(modality, bbox=(0.1, 0.1, 0.5, 0.5) if modality == "image_region" else None)
        assert len(pool.items()) == 1
