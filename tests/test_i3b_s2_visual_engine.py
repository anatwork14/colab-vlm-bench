"""ITER3 Round B S2, HEAVY (run under the heavy lock): the real image path through the OFFICIAL Qwen3.5-0.8B
checkpoint and the repo image fixture tests/fixtures/000000039769.png (640x480 -> grid 30x40 -> 300 visual tokens).

What is pinned end to end:
  * the visual status is SUPPORTED for an S1-verified image region on the real processor geometry;
  * the protected positions equal the merged cells the box covers, computed here by arithmetic from the grid the
    processor produced, and they are ORIGINAL prompt positions inside the image token run;
  * the protected visual positions are still retained in every full-attention layer at the end of the request
    (they were preserved through physical compaction, not only labelled);
  * a bytes mismatch protects nothing, and a text-only request is UNSUPPORTED with its reason;
  * request isolation: a request with no evidence after a protecting one reports zero visual positions.

Preserving a visual span does NOT prove the model used it. cpu_pytorch_reference_not_vllm. No speed or quality claim.
"""
import hashlib
from pathlib import Path

import pytest

from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "000000039769.png"
pytestmark = pytest.mark.skipif(
    not ((CKPT / "config.json").is_file() and FIXTURE.is_file()),
    reason=f"official checkpoint {CKPT} or fixture {FIXTURE} not present",
)

BOX = (0.0, 0.0, 0.5, 0.5)  # ymin, xmin, ymax, xmax: the top-left quadrant of the source image
QUESTION = "What is shown in this picture?"


@pytest.fixture(scope="module")
def image_bytes():
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def sha(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


@pytest.fixture(scope="module")
def eng():
    e = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False,
                           h2o_min_budget=8, h2o_budget_ratio=0.6, evidence_policy=EvidencePolicy())
    yield e
    e.close()


def _pool(request_id, sha256, box=BOX):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id="ev_region", modality="image_region", content="top-left quadrant",
                                   source_sha256=sha256, verified=True, bounding_box=box))
    return pool


def _run(eng, request_id, *, pool, image_present=True, image_bytes=None):
    return eng.execute_request(request_id=request_id, image_present=image_present,
                               image_bytes=image_bytes, question=QUESTION, max_new_tokens=3,
                               evidence_pool=pool)


def test_verified_image_region_is_supported_and_protects_its_merged_cells(eng, image_bytes, sha):
    res = _run(eng, "s2_supported", pool=_pool("s2_supported", sha), image_bytes=image_bytes)
    block = res.telemetry["starlight"]
    vis = res.telemetry["visual_encoder_trace"]
    assert block["visual_evidence_protection"] == "SUPPORTED", block["visual_evidence_protection_reason"]
    assert block["visual_evidence_protection_reason"] is None

    grid_t, grid_h, grid_w = vis["image_grid_thw"][0]
    assert grid_t == 1 and (grid_h, grid_w) == (30, 40)
    n_tokens = (grid_h // 2) * (grid_w // 2)
    assert n_tokens == vis["visual_token_embeddings_count"] == 300
    assert block["visual_geometry"]["n_tokens"] == n_tokens

    start, end = block["visual_span"]
    assert end - start == n_tokens, "the span is the single image token run"

    # The top-left quadrant covers half the merged rows and half the merged columns (ceil on the odd side).
    merged_h, merged_w = grid_h // 2, grid_w // 2
    rows, cols = -(-merged_h // 2), -(-merged_w // 2)
    expected = sorted(start + r * merged_w + c for r in range(rows) for c in range(cols))
    assert block["visual_positions_protected"] == len(expected) == 8 * 10
    assert block["visual_protected_positions"] == expected
    assert all(start <= p < end for p in expected), "original positions inside the image token run"
    assert block["visual_evidence_items"]["ev_region"]["protects"] is True
    assert block["visual_evidence_items"]["ev_region"]["reason"] == "PROTECTED_SOURCE_BYTES_VERIFIED"


def test_protected_visual_positions_are_retained_in_every_full_attention_layer_at_the_end(eng, image_bytes, sha):
    res = _run(eng, "s2_retained", pool=_pool("s2_retained", sha), image_bytes=image_bytes)
    block = res.telemetry["starlight"]
    protected = len(block["visual_protected_positions"])
    retained = block["visual_protected_retained_final_by_layer"]
    assert retained, "the per-layer final retention must be reported"
    for layer, count in retained.items():
        assert count == protected, (layer, count, protected)


def test_bytes_mismatch_protects_no_visual_position(eng, image_bytes):
    wrong = hashlib.sha256(b"not the fixture").hexdigest()
    res = _run(eng, "s2_mismatch", pool=_pool("s2_mismatch", wrong), image_bytes=image_bytes)
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "SUPPORTED"
    assert block["visual_positions_protected"] == 0
    item = block["visual_evidence_items"]["ev_region"]
    assert item["protects"] is False and item["reason"] == "REJECTED_SOURCE_BYTES_MISMATCH"


def test_text_only_request_with_image_evidence_is_unsupported_with_its_reason(eng, sha):
    res = eng.execute_request(request_id="s2_text", prompt_token_ids=[100, 101, 102, 103], image_present=False,
                              max_new_tokens=2, evidence_pool=_pool("s2_text", sha))
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "UNSUPPORTED"
    assert block["visual_evidence_protection_reason"] == "NO_IMAGE_IN_REQUEST"
    assert block["visual_positions_protected"] == 0


def test_a_later_request_without_evidence_protects_no_visual_position(eng, image_bytes, sha):
    _run(eng, "s2_first", pool=_pool("s2_first", sha), image_bytes=image_bytes)
    res = _run(eng, "s2_second", pool=None, image_bytes=image_bytes)
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "NO_EVIDENCE_DECLARED"
    assert block["visual_positions_protected"] == 0
    assert block["visual_protected_positions"] == []
