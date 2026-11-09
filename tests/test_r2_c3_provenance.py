"""ITER3 R2 C-3 (closure doc, MEDIUM): evidence provenance must not overstate verification.

The controller never checks evidence content. `EvidenceItem.verified` is a caller declaration, and
`source_sha256` is compared with the request's image bytes ONLY when an image is present. Telemetry must
say so: the `verified` flag is always CALLER_DECLARED_NOT_VERIFIED, and `source_sha256` is labelled as
checked only when it was actually compared against image bytes. Text-only requests stay
CALLER_DECLARED_NOT_VERIFIED for the source hash too. cpu_pytorch_reference_not_vllm.
"""
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.starlight import evidence_provenance, evidence_validity

SHA_A = "a" * 64
SHA_B = "b" * 64
DECLARED = "CALLER_DECLARED_NOT_VERIFIED"


def _pool(request_id="req-1", sha=SHA_A, verified=True, eid="ev_1"):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality="text_span", content="car",
                                   source_sha256=sha, verified=verified), keywords=["car"])
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


def test_the_verified_flag_is_never_reported_as_checked():
    pool = EvidencePool("req-1")
    pool.add_evidence(EvidenceItem(evidence_id="ev_a", modality="text_span", content="car",
                                   source_sha256=SHA_A, verified=True), keywords=["car"])
    pool.add_evidence(EvidenceItem(evidence_id="ev_b", modality="text_span", content="cat",
                                   source_sha256=SHA_A, verified=False), keywords=["cat"])
    for image in (None, SHA_A, SHA_B):
        prov = evidence_provenance(pool, image_sha256=image)
        assert {p["verified_flag"] for p in prov.values()} == {DECLARED}


def test_verified_false_is_reported_as_a_caller_declaration_not_a_check():
    pool = _pool(verified=False)
    assert evidence_validity(pool, "req-1", None) == {"ev_1": ("unverified", "caller_declared_verified_false")}


def test_existing_status_values_are_unchanged():
    # Only the provenance labels and the verified_false reason text change; statuses do not.
    assert evidence_validity(_pool(), "req-1", SHA_A) == {"ev_1": ("valid", "source_sha256_match")}
    assert evidence_validity(_pool(), "req-1", SHA_B) == {"ev_1": ("stale", "source_sha256_mismatch")}
    assert evidence_validity(_pool(), "req-1", None) == {"ev_1": ("valid", "source_unchecked_text_only")}
