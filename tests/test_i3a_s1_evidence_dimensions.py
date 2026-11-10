"""ITER3 round A, S1: evidence verification as separate tri-state dimensions (no caller-Boolean trust).

Dimensions (VERIFIED / NOT_VERIFIED / UNCERTAIN / NOT_ASSESSED):
  integrity:  source_identity (declaration bound to THIS request), source_bytes_hash (declared sha256 vs the
              request's image bytes)
  checkable:  location_valid (image_region bounding box well-formed)
  semantic:   relevance_to_question, claim_support (no assessor exists in the engine: always NOT_ASSESSED)
Source-integrity results are never reported as semantic verification. The caller's `verified` flag is a
declaration recorded in provenance; it never protects by itself.

Fallback for incomplete evidence (the provisional design, documented in STARLIGHT_STAGE1_DESIGN.md):
  * source bytes VERIFIED and identity VERIFIED  -> protects (PROTECTED_SOURCE_BYTES_VERIFIED).
  * no source bytes to check (text-only)         -> DECLARED_UNVERIFIED: no hard protection by default.
    Protection only under the explicit, reported, ablatable EvidencePolicy(accept_declared=True).
  * source bytes mismatch, or not bound to request -> rejected, never protects, even with accept_declared.
  * image present but not hashable (UNCERTAIN)  -> no protection.
  * caller verified=False                        -> refused before any protection.
A-5: missing evidence is never fabricated; DA/structural signals still operate. Cpu reference, not vLLM.
"""
import hashlib
from types import SimpleNamespace

from vlmbench.infrastructure.evidence_verification import (
    DIMENSIONS,
    SEMANTIC_DIMENSIONS,
    TRI_STATES,
    EvidencePolicy,
    assess_evidence,
    decide_protection,
)
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.starlight import grounding_protection_with_report

SHA_A = hashlib.sha256(b"image-a").hexdigest()
SHA_B = hashlib.sha256(b"image-b").hexdigest()


class _Tok:
    VOCAB = {"car": 8, "cat": 9}

    def encode(self, text):
        return [self.VOCAB[w] for w in text.lower().split() if w in self.VOCAB]


def _item(eid="ev_1", sha=SHA_A, verified=True, modality="text_span", bbox=None, content="car"):
    return EvidenceItem(evidence_id=eid, modality=modality, content=content, source_sha256=sha,
                        verified=verified, bounding_box=bbox)


def _pool(items, keywords=("car",), request_id="req-1"):
    pool = EvidencePool(request_id)
    for it in items:
        pool.add_evidence(it, keywords=list(keywords))
    return pool


def _assess(item, *, request_id="req-1", pool_request_id="req-1", image_sha=None, image_present=False):
    return assess_evidence(item, request_id=request_id, pool_request_id=pool_request_id,
                           image_sha256=image_sha, image_present=image_present)


def _protects(item, *, policy=EvidencePolicy(), image_sha=None, image_present=False, pool_request_id="req-1"):
    a = _assess(item, pool_request_id=pool_request_id, image_sha=image_sha, image_present=image_present)
    return decide_protection(a, policy, caller_verified=bool(item.verified))


def test_dimensions_are_named_and_tri_state_and_semantic_ones_exist():
    assert DIMENSIONS == ("source_identity", "source_bytes_hash", "location_valid",
                          "relevance_to_question", "claim_support")
    assert SEMANTIC_DIMENSIONS == ("relevance_to_question", "claim_support")
    assert TRI_STATES == ("VERIFIED", "NOT_VERIFIED", "UNCERTAIN", "NOT_ASSESSED")
    dims = _assess(_item()).dimensions
    assert set(dims) == set(DIMENSIONS)
    assert all(v in TRI_STATES for v in dims.values())


def test_text_only_request_has_no_source_bytes_to_check_and_no_semantic_verdict():
    dims = _assess(_item(), image_present=False).dimensions
    assert dims["source_bytes_hash"] == "NOT_ASSESSED"
    assert dims["relevance_to_question"] == "NOT_ASSESSED"
    assert dims["claim_support"] == "NOT_ASSESSED"


def test_matching_image_bytes_verify_the_source_hash_but_never_the_semantics():
    a = _assess(_item(sha=SHA_A), image_present=True, image_sha=SHA_A)
    assert a.dimensions["source_bytes_hash"] == "VERIFIED"
    assert a.dimensions["source_identity"] == "VERIFIED"
    # A matching hash says the source is the one declared. It says nothing about relevance or support.
    assert a.dimensions["relevance_to_question"] == "NOT_ASSESSED"
    assert a.dimensions["claim_support"] == "NOT_ASSESSED"


def test_no_combination_reports_a_semantic_dimension_as_verified():
    cases = [
        _assess(_item(sha=SHA_A), image_present=True, image_sha=SHA_A),
        _assess(_item(sha=SHA_A), image_present=True, image_sha=SHA_B),
        _assess(_item(), image_present=False),
        _assess(_item(), pool_request_id="other", image_present=False),
    ]
    for a in cases:
        assert all(a.dimensions[d] != "VERIFIED" for d in SEMANTIC_DIMENSIONS)


def test_caller_verified_true_alone_does_not_protect_a_text_only_item():
    protects, mode, reason = _protects(_item(verified=True))
    assert protects is False
    assert mode == "NO_PROTECTION_DECLARED_UNVERIFIED"
    assert "no source bytes" in reason


def test_accept_declared_policy_protects_text_only_items_and_says_so():
    protects, mode, _reason = _protects(_item(verified=True), policy=EvidencePolicy(accept_declared=True))
    assert protects is True
    assert mode == "PROTECTED_DECLARED_UNVERIFIED_BY_POLICY"


def test_matching_image_bytes_protect_without_the_policy_flag():
    protects, mode, _ = _protects(_item(sha=SHA_A), image_present=True, image_sha=SHA_A)
    assert protects is True
    assert mode == "PROTECTED_SOURCE_BYTES_VERIFIED"


def test_image_mismatch_is_rejected_even_with_accept_declared():
    protects, mode, _ = _protects(_item(sha=SHA_A), policy=EvidencePolicy(accept_declared=True),
                                  image_present=True, image_sha=SHA_B)
    assert protects is False
    assert mode == "REJECTED_SOURCE_BYTES_MISMATCH"


def test_evidence_not_bound_to_this_request_is_rejected_even_with_accept_declared():
    protects, mode, _ = _protects(_item(), policy=EvidencePolicy(accept_declared=True), pool_request_id="other")
    assert protects is False
    assert mode == "REJECTED_NOT_BOUND_TO_REQUEST"


def test_unhashable_image_is_uncertain_and_protects_nothing_even_with_accept_declared():
    protects, mode, _ = _protects(_item(sha=SHA_A), policy=EvidencePolicy(accept_declared=True),
                                  image_present=True, image_sha=None)
    assert protects is False
    assert mode == "UNCERTAIN_SOURCE_NOT_HASHED"
    assert _assess(_item(sha=SHA_A), image_present=True, image_sha=None).dimensions["source_bytes_hash"] == "UNCERTAIN"


def test_caller_verified_false_is_refused_before_any_protection():
    fake = SimpleNamespace(items=lambda: (_item(verified=False),), request_id="req-1",
                           keyword_index=lambda: {"car": ("ev_1",)})
    protected, report = grounding_protection_with_report(fake, [8, 9, 8], _Tok().encode, "req-1", None,
                                                         policy=EvidencePolicy(accept_declared=True))
    assert protected == {}
    assert report["assessments"]["ev_1"]["protection_mode"] == "REJECTED_CALLER_DECLARED_FALSE"


def test_method_and_provenance_are_recorded_not_asserted_as_verification():
    a = _assess(_item(verified=True, sha=SHA_A), image_present=True, image_sha=SHA_A)
    assert a.method == "CALLER_DECLARED_KEYWORD_MATCH_V1"
    assert a.provenance["caller_verified_flag"] == "TRUE"
    assert a.provenance["source_sha256_declared"] == SHA_A
    assert len(a.provenance["content_sha256"]) == 64


def test_location_valid_checks_the_bounding_box_of_image_region_evidence():
    ok = _assess(_item(modality="image_region", bbox=(0.1, 0.1, 0.5, 0.5)))
    bad = _assess(_item(modality="image_region", bbox=(0.6, 0.1, 0.2, 0.5)))
    none = _assess(_item(modality="image_region", bbox=None))
    text = _assess(_item(modality="text_span"))
    assert ok.dimensions["location_valid"] == "VERIFIED"
    assert bad.dimensions["location_valid"] == "NOT_VERIFIED"
    assert none.dimensions["location_valid"] == "NOT_ASSESSED"
    assert text.dimensions["location_valid"] == "NOT_ASSESSED"


def test_grounding_report_carries_per_item_assessments_and_the_policy():
    pool = _pool([_item("ev_1", content="car")], keywords=("car",))
    protected, report = grounding_protection_with_report(pool, [1, 8, 2], _Tok().encode, "req-1", None,
                                                         policy=EvidencePolicy(accept_declared=False))
    assert protected == {}
    entry = report["assessments"]["ev_1"]
    assert entry["protection_mode"] == "NO_PROTECTION_DECLARED_UNVERIFIED"
    assert entry["dimensions"]["source_bytes_hash"] == "NOT_ASSESSED"
    assert report["evidence_policy"] == {"accept_declared": False}
    assert report["omitted_items"] == [{"evidence_id": "ev_1", "reason": "NO_PROTECTION_DECLARED_UNVERIFIED"}]


def test_A5_missing_evidence_is_not_fabricated_and_independent_da_signal_still_protects():
    from vlmbench.infrastructure.real_engine import RealSevenArmEngine
    from test_starlight_oracle import PROMPT, _cfg, _WordTok

    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="a5", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4)
    finally:
        engine.close()
    s = res.telemetry["starlight"]
    assert s["grounding_signal_status"] == "NO_EVIDENCE_DECLARED"
    assert s["grounding_protected_by_evidence"] == {}
    assert s["protected_count_by_source"]["grounding"] == 0
    assert s["protected_count_by_source"]["da"] > 0  # the independent DA signal still operates


def test_engine_telemetry_reports_dimensions_mode_and_the_default_policy_for_text_only_evidence():
    from vlmbench.infrastructure.real_engine import RealSevenArmEngine
    from test_starlight_oracle import PROMPT, _cfg, _WordTok

    pool = _pool([_item("ev_c", content="car")], keywords=("car",), request_id="a5b")
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="a5b", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4, evidence_pool=pool)
    finally:
        engine.close()
    s = res.telemetry["starlight"]
    assert s["evidence_policy"] == {"accept_declared": False}
    entry = s["grounding_evidence_assessments"]["ev_c"]
    assert entry["protection_mode"] == "NO_PROTECTION_DECLARED_UNVERIFIED"
    assert entry["dimensions"]["source_bytes_hash"] == "NOT_ASSESSED"
    assert s["grounding_protected_by_evidence"] == {}
