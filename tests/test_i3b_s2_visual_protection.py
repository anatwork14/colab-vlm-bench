"""ITER3 Round B S2: visual evidence protection, gated by S1 dimensions AND a supported mapping. Pure (no model).

Rules pinned here:
  * a visual span is protected ONLY when S1 protects the item (source bytes VERIFIED and bound to this request)
    AND the item's location is VERIFIED AND the region maps to positions under a supported geometry and span;
  * the status is SUPPORTED when the mapping is established for this request, and UNSUPPORTED with a reason when
    it is not. SUPPORTED does not mean protected: per-item S1 refusals are reported item by item;
  * protected positions are ORIGINAL prompt positions (the engine's coordinates before any compaction);
  * the report never protects anything under an UNSUPPORTED status.

Preserving a visual span does NOT prove the model used it. cpu_pytorch_reference_not_vllm.
"""
import hashlib

import pytest

from vlmbench.infrastructure import visual_mapping as vm
from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool

RID = "req-s2"
IMG = b"fixture image bytes for S2"
SHA = hashlib.sha256(IMG).hexdigest()
OTHER_SHA = hashlib.sha256(b"different image").hexdigest()
MIN_PIXELS = 65536
MAX_PIXELS = 16777216
GEOM = vm.build_geometry(256, 256, [1, 16, 16], patch_size=16, merge_size=2,
                         min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)[0]
SPAN = (3, 3 + 64)  # image tokens at original positions 3..66


def _pool(boxes, *, sha=SHA, request_id=RID):
    pool = EvidencePool(request_id)
    for i, box in enumerate(boxes):
        pool.add_evidence(
            EvidenceItem(evidence_id=f"v{i}", modality="image_region", content=f"region {i}",
                         source_sha256=sha, verified=True, bounding_box=box),
            keywords=[f"kw{i}"],
        )
    return pool


def _report(pool, **overrides):
    args = dict(request_id=RID, image_sha256=SHA, image_present=True, policy=EvidencePolicy(),
                geometry=GEOM, geometry_reason=None, span=SPAN, span_reason=None)
    args.update(overrides)
    return vm.visual_evidence_report(pool, **args)


# --- vocabulary ---------------------------------------------------------------------------------------

def test_no_pool_declares_no_evidence():
    rep = _report(None)
    assert rep["status"] == "NO_EVIDENCE_DECLARED"
    assert rep["positions"] == {} and rep["visual_positions_protected"] == 0


def test_text_only_pool_declares_no_image_evidence():
    pool = EvidencePool(RID)
    pool.add_evidence(EvidenceItem(evidence_id="t0", modality="text_span", content="car", source_sha256=SHA,
                                   verified=True), keywords=["car"])
    rep = _report(pool)
    assert rep["status"] == "NO_IMAGE_EVIDENCE_DECLARED"
    assert rep["positions"] == {}


# --- supported mapping, S1-gated protection -----------------------------------------------------------

def test_valid_region_with_verified_bytes_protects_its_original_positions():
    box = (0.0, 0.0, 0.5, 0.5)
    rep = _report(_pool([box]))
    assert rep["status"] == "SUPPORTED" and rep["unsupported_reason"] is None
    expected = vm.region_positions(box, GEOM, SPAN)[0]
    assert sorted(rep["positions"]) == expected
    assert all(SPAN[0] <= p < SPAN[1] for p in rep["positions"]), "positions must be original, inside the span"
    assert set(rep["positions"].values()) == {("v0",)}
    assert rep["visual_positions_protected"] == len(expected) == 16
    item = rep["items"]["v0"]
    assert item["protects"] is True and item["reason"] == "PROTECTED_SOURCE_BYTES_VERIFIED"
    assert item["dimensions"]["location_valid"] == "VERIFIED"
    assert item["dimensions"]["source_bytes_hash"] == "VERIFIED"


def test_protected_positions_are_sorted_original_positions_for_telemetry():
    rep = _report(_pool([(0.0, 0.0, 0.5, 0.5)]))
    assert rep["visual_protected_positions"] == sorted(rep["positions"])


def test_bytes_mismatch_protects_nothing_but_the_mapping_is_still_established():
    rep = _report(_pool([(0.0, 0.0, 0.5, 0.5)], sha=OTHER_SHA))
    assert rep["status"] == "SUPPORTED"
    item = rep["items"]["v0"]
    assert item["protects"] is False and item["reason"] == "REJECTED_SOURCE_BYTES_MISMATCH"
    assert rep["positions"] == {} and rep["visual_positions_protected"] == 0


def test_evidence_bound_to_another_request_protects_nothing():
    rep = _report(_pool([(0.0, 0.0, 0.5, 0.5)], request_id="someone-else"))
    item = rep["items"]["v0"]
    assert item["protects"] is False and item["reason"] == "REJECTED_NOT_BOUND_TO_REQUEST"
    assert rep["positions"] == {}


def test_out_of_range_box_is_refused_with_the_box_reason_and_protects_nothing():
    rep = _report(_pool([(0.0, 0.0, 1.2, 0.5)]))
    item = rep["items"]["v0"]
    assert item["protects"] is False and item["reason"] == vm.REASON_BOX_OUT_OF_RANGE
    assert rep["positions"] == {}


def test_malformed_box_is_refused_with_the_box_reason():
    rep = _report(_pool([(0.6, 0.0, 0.5, 0.5)]))
    assert rep["items"]["v0"]["reason"] == vm.REASON_BOX_MALFORMED
    assert rep["items"]["v0"]["protects"] is False


def test_region_without_a_bounding_box_is_not_protected():
    rep = _report(_pool([None]))
    item = rep["items"]["v0"]
    assert item["protects"] is False and item["reason"] == vm.REASON_NO_BOUNDING_BOX
    assert item["dimensions"]["location_valid"] == "NOT_ASSESSED"


def test_a_bad_region_does_not_stop_a_good_one_and_the_union_is_reported():
    rep = _report(_pool([(0.0, 0.0, 0.5, 0.5), (0.6, 0.0, 0.5, 0.5)]))
    assert rep["items"]["v0"]["protects"] is True
    assert rep["items"]["v1"]["protects"] is False
    assert set(rep["positions"].values()) == {("v0",)}


def test_overlapping_regions_record_every_supporting_item_per_position():
    rep = _report(_pool([(0.0, 0.0, 0.5, 0.5), (0.25, 0.25, 0.75, 0.75)]))
    both = [p for p, ids in rep["positions"].items() if set(ids) == {"v0", "v1"}]
    assert both, "a position covered by two valid regions must list both"


# --- UNSUPPORTED: the mapping cannot be established, so nothing is protected ----------------------

@pytest.mark.parametrize("overrides, reason", [
    (dict(image_present=False), "NO_IMAGE_IN_REQUEST"),
    (dict(image_sha256=None), "IMAGE_SOURCE_UNKNOWN"),
    (dict(geometry=None, geometry_reason=vm.REASON_MISSING_GRID), vm.REASON_MISSING_GRID),
    (dict(geometry=None, geometry_reason=vm.REASON_GRID_ARITHMETIC_MISMATCH), vm.REASON_GRID_ARITHMETIC_MISMATCH),
    (dict(geometry=None, geometry_reason=vm.REASON_TILED_OR_TEMPORAL_GRID), vm.REASON_TILED_OR_TEMPORAL_GRID),
    (dict(span=None, span_reason=vm.REASON_MULTI_IMAGE), vm.REASON_MULTI_IMAGE),
    (dict(span=None, span_reason=vm.REASON_NO_IMAGE_TOKENS), vm.REASON_NO_IMAGE_TOKENS),
])
def test_unsupported_mapping_says_why_and_protects_nothing(overrides, reason):
    rep = _report(_pool([(0.0, 0.0, 0.5, 0.5)]), **overrides)
    assert rep["status"] == "UNSUPPORTED"
    assert rep["unsupported_reason"] == reason
    assert rep["positions"] == {} and rep["visual_positions_protected"] == 0
    assert rep["items"]["v0"]["protects"] is False


def test_an_unsupported_status_never_carries_positions_even_for_a_perfect_region():
    rep = _report(_pool([(0.0, 0.0, 1.0, 1.0)]), image_present=False)
    assert rep["positions"] == {} and rep["visual_protected_positions"] == []


# --- geometry from the processor's grid list (multi-image and missing grids are refused) ------------

def test_geometry_from_a_single_grid_list_is_the_single_image_geometry():
    geom, reason = vm.geometry_from_grid_list(256, 256, [[1, 16, 16]], patch_size=16, merge_size=2,
                                              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert reason is None and geom.n_tokens == 64


def test_two_grids_are_refused_as_multi_image():
    geom, reason = vm.geometry_from_grid_list(256, 256, [[1, 16, 16], [1, 16, 16]], patch_size=16, merge_size=2,
                                              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert geom is None and reason == vm.REASON_MULTI_IMAGE


@pytest.mark.parametrize("grids", [None, []])
def test_no_grid_is_refused_as_missing(grids):
    geom, reason = vm.geometry_from_grid_list(256, 256, grids, patch_size=16, merge_size=2,
                                              min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert geom is None and reason == vm.REASON_MISSING_GRID
