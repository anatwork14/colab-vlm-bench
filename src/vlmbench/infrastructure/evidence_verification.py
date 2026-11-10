"""ITER3 round A, S1: evidence verification as separate tri-state dimensions (pure policy: no torch, no numpy).

Verification is NOT a caller Boolean. Each evidence item is assessed on named dimensions, each one of
VERIFIED / NOT_VERIFIED / UNCERTAIN / NOT_ASSESSED:

  integrity (checked by this code):
    source_identity    the declaration is bound to THIS request (pool request id equals the request id)
    source_bytes_hash  the declared source sha256 equals the sha256 of the request's image bytes
  checkable location:
    location_valid     an image_region bounding box is well formed; text spans are NOT_ASSESSED
  semantic (no assessor exists in the engine, so always NOT_ASSESSED):
    relevance_to_question, claim_support

Source-integrity results are never reported as semantic verification: a matching hash shows the source is the
declared one, nothing about relevance or support. The caller's `verified` flag is a declaration recorded in
provenance. It never protects by itself.

Fallback for incomplete evidence (provisional, owner-ratifiable; see STARLIGHT_STAGE1_DESIGN.md section 4.3):
  * caller verified=False                           -> refused (REJECTED_CALLER_DECLARED_FALSE).
  * not bound to this request                       -> rejected (REJECTED_NOT_BOUND_TO_REQUEST).
  * image present, declared hash differs            -> rejected (REJECTED_SOURCE_BYTES_MISMATCH).
  * image present, not hashable                     -> UNCERTAIN, no protection (UNCERTAIN_SOURCE_NOT_HASHED).
  * source bytes VERIFIED and bound                 -> protects (PROTECTED_SOURCE_BYTES_VERIFIED).
  * no source bytes to check (text-only request)    -> DECLARED_UNVERIFIED: no hard protection by default
    (NO_PROTECTION_DECLARED_UNVERIFIED). Protection only under the explicit, reported, ablatable
    EvidencePolicy(accept_declared=True) (PROTECTED_DECLARED_UNVERIFIED_BY_POLICY).
Why the default is conservative: a text-only item has no source-integrity check at all, so a bare caller claim
should not decide which KV entries are evicted. The flag makes the trade-off visible and ablatable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TRI_STATES: tuple[str, ...] = ("VERIFIED", "NOT_VERIFIED", "UNCERTAIN", "NOT_ASSESSED")
DIMENSIONS: tuple[str, ...] = (
    "source_identity", "source_bytes_hash", "location_valid", "relevance_to_question", "claim_support",
)
SEMANTIC_DIMENSIONS: tuple[str, ...] = ("relevance_to_question", "claim_support")
METHOD_LABEL = "CALLER_DECLARED_KEYWORD_MATCH_V1"

MODE_REJECTED_CALLER_FALSE = "REJECTED_CALLER_DECLARED_FALSE"
MODE_REJECTED_NOT_BOUND = "REJECTED_NOT_BOUND_TO_REQUEST"
MODE_REJECTED_BYTES_MISMATCH = "REJECTED_SOURCE_BYTES_MISMATCH"
MODE_UNCERTAIN = "UNCERTAIN_SOURCE_NOT_HASHED"
MODE_PROTECTED_VERIFIED = "PROTECTED_SOURCE_BYTES_VERIFIED"
MODE_NO_PROTECTION_DECLARED = "NO_PROTECTION_DECLARED_UNVERIFIED"
MODE_PROTECTED_BY_POLICY = "PROTECTED_DECLARED_UNVERIFIED_BY_POLICY"


@dataclass(frozen=True)
class EvidencePolicy:
    """Explicit evidence policy. accept_declared=False is the default and the conservative choice."""
    accept_declared: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"accept_declared": bool(self.accept_declared)}


@dataclass(frozen=True)
class EvidenceAssessment:
    evidence_id: str
    dimensions: dict[str, str]
    method: str
    provenance: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ProtectionDecision:
    protects: bool
    mode: str
    reason: str


def _bbox_well_formed(box: Any) -> bool:
    # (ymin, xmin, ymax, xmax) in [0, 1] with strictly positive extent.
    if box is None or len(box) != 4:
        return False
    ymin, xmin, ymax, xmax = (float(v) for v in box)
    return 0.0 <= ymin < ymax <= 1.0 and 0.0 <= xmin < xmax <= 1.0


def assess_evidence(
    item: Any,
    *,
    request_id: str,
    pool_request_id: str,
    image_sha256: str | None,
    image_present: bool,
) -> EvidenceAssessment:
    """Assess one evidence item on the separate dimensions. Pure: no protection decision here."""
    if pool_request_id == request_id:
        identity = "VERIFIED"
    else:
        identity = "NOT_VERIFIED"
    if not image_present:
        bytes_hash = "NOT_ASSESSED"  # text-only request: there are no source bytes to compare
    elif image_sha256 is None:
        bytes_hash = "UNCERTAIN"  # image present but its bytes could not be hashed
    elif str(item.source_sha256).lower() == str(image_sha256).lower():
        bytes_hash = "VERIFIED"
    else:
        bytes_hash = "NOT_VERIFIED"
    if item.modality == "image_region":
        if item.bounding_box is None:
            location = "NOT_ASSESSED"
        else:
            location = "VERIFIED" if _bbox_well_formed(item.bounding_box) else "NOT_VERIFIED"
    else:
        location = "NOT_ASSESSED"  # a text span has no source geometry to check here
    dimensions = {
        "source_identity": identity,
        "source_bytes_hash": bytes_hash,
        "location_valid": location,
        "relevance_to_question": "NOT_ASSESSED",  # no relevance assessor exists: never VERIFIED
        "claim_support": "NOT_ASSESSED",  # no claim-support assessor exists: never VERIFIED
    }
    provenance = {
        "caller_verified_flag": "TRUE" if bool(item.verified) else "FALSE",
        "source_sha256_declared": str(item.source_sha256).lower(),
        "content_sha256": item.content_digest(),
    }
    return EvidenceAssessment(evidence_id=item.evidence_id, dimensions=dimensions,
                              method=METHOD_LABEL, provenance=provenance)


def decide_protection(assessment: EvidenceAssessment, policy: EvidencePolicy, caller_verified: bool) -> ProtectionDecision:
    """Whether this item may hard-protect its keyword spans, and why. Order matters: first refusal wins."""
    d = assessment.dimensions
    if not caller_verified:
        return ProtectionDecision(False, MODE_REJECTED_CALLER_FALSE, "caller declared verified=False")
    if d["source_identity"] == "NOT_VERIFIED":
        return ProtectionDecision(False, MODE_REJECTED_NOT_BOUND, "pool built for another request")
    if d["source_bytes_hash"] == "NOT_VERIFIED":
        return ProtectionDecision(False, MODE_REJECTED_BYTES_MISMATCH, "declared sha256 differs from the image bytes")
    if d["source_bytes_hash"] == "UNCERTAIN":
        return ProtectionDecision(False, MODE_UNCERTAIN, "image present but its bytes could not be hashed")
    if d["source_bytes_hash"] == "VERIFIED":
        return ProtectionDecision(True, MODE_PROTECTED_VERIFIED,
                                  "declared sha256 matches the image bytes (source integrity only)")
    # NOT_ASSESSED: text-only request, no source bytes were checked.
    if policy.accept_declared:
        return ProtectionDecision(True, MODE_PROTECTED_BY_POLICY,
                                  "no source bytes checked; protected by explicit accept_declared policy")
    return ProtectionDecision(False, MODE_NO_PROTECTION_DECLARED,
                              "no source bytes to check (text-only request); declared, unverified")
