"""ITER3 R2 C-3 (closure doc, MEDIUM): evidence provenance must not overstate verification.

The controller never checks evidence content. `EvidenceItem.verified` is a caller declaration, and
`source_sha256` is compared with the request's image bytes ONLY when an image is present. Telemetry must
say so: the `verified` flag is always CALLER_DECLARED_NOT_VERIFIED, and `source_sha256` is labelled as
checked only when it was actually compared against image bytes. Text-only requests stay
CALLER_DECLARED_NOT_VERIFIED for the source hash too. cpu_pytorch_reference_not_vllm.
"""
from types import SimpleNamespace

from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.starlight import evidence_provenance, evidence_validity

SHA_A = "a" * 64
SHA_B = "b" * 64
DECLARED = "CALLER_DECLARED_NOT_VERIFIED"


def _pool(request_id="req-1", sha=SHA_A, verified=True, eid="ev_1", keywords=("car",)):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality="text_span", content=" ".join(keywords),
                                   source_sha256=sha, verified=verified), keywords=list(keywords))
    return pool


def test_text_only_request_labels_both_the_flag_and_the_source_hash_as_caller_declared():
    prov = evidence_provenance(_pool(), image_sha256=None)
    assert prov == {"ev_1": {"verified_flag": DECLARED, "source_sha256": DECLARED}}


def test_image_hash_match_is_labelled_as_checked_but_the_flag_stays_caller_declared():
    prov = evidence_provenance(_pool(sha=SHA_A), image_sha256=SHA_A)
    assert prov["ev_1"]["source_sha256"] == "CHECKED_IMAGE_BYTES_MATCH"
    assert prov["ev_1"]["verified_flag"] == DECLARED


def test_image_hash_mismatch_is_labelled_as_checked_and_mismatched():
    prov = evidence_provenance(_pool(sha=SHA_A), image_sha256=SHA_B)
    assert prov["ev_1"]["source_sha256"] == "CHECKED_IMAGE_BYTES_MISMATCH"
    assert prov["ev_1"]["verified_flag"] == DECLARED


def _fake_pool(items, request_id="req-1"):
    # EvidencePool.add_evidence refuses verified=False, so that case is driven through a minimal pool
    # with the same items()/request_id interface (what evidence_validity and provenance read).
    return SimpleNamespace(items=lambda: tuple(items), request_id=request_id, keyword_index=lambda: {})


def test_the_verified_flag_is_never_reported_as_checked():
    items = [
        EvidenceItem(evidence_id="ev_a", modality="text_span", content="car", source_sha256=SHA_A, verified=True),
        EvidenceItem(evidence_id="ev_b", modality="text_span", content="cat", source_sha256=SHA_A, verified=False),
    ]
    pool = _fake_pool(items)
    for image in (None, SHA_A, SHA_B):
        prov = evidence_provenance(pool, image_sha256=image)
        assert {p["verified_flag"] for p in prov.values()} == {DECLARED}


def test_verified_false_is_reported_as_a_caller_declaration_not_a_check():
    item = EvidenceItem(evidence_id="ev_1", modality="text_span", content="car", source_sha256=SHA_A, verified=False)
    assert evidence_validity(_fake_pool([item]), "req-1", None) == {
        "ev_1": ("unverified", "caller_declared_verified_false")}


def test_status_values_for_image_and_text_requests():
    # S1 RE-DERIVED (was test_existing_status_values_are_unchanged). The image-bound statuses are unchanged.
    # The text-only status changes by design: "valid / source_unchecked_text_only" read as VALID although no
    # source bytes were checked, so it is now DECLARED_UNVERIFIED. Its protection is gated by the explicit policy.
    assert evidence_validity(_pool(), "req-1", SHA_A) == {"ev_1": ("valid", "source_sha256_match")}
    assert evidence_validity(_pool(), "req-1", SHA_B) == {"ev_1": ("stale", "source_sha256_mismatch")}
    assert evidence_validity(_pool(), "req-1", None) == {"ev_1": ("declared_unverified", "no_source_bytes_checked")}


def test_engine_telemetry_carries_the_provenance_labels_for_a_text_only_request():
    # End-to-end on the tiny scaffold (no checkpoint): the STARLIGHT block reports the labels.
    from vlmbench.infrastructure.real_engine import RealSevenArmEngine
    from test_starlight_oracle import PROMPT, _cfg, _WordTok

    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="c3_text", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4, evidence_pool=_pool("c3_text", eid="ev_c", keywords=("car",)))
    finally:
        engine.close()
    prov = res.telemetry["starlight"]["grounding_evidence_provenance"]
    assert prov == {"ev_c": {"verified_flag": DECLARED, "source_sha256": DECLARED}}
