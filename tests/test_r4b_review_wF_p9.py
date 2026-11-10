"""ITER3 Round B closure review (auditor wF:p9, reviewed HEAD adcf942). ADDED TESTS ONLY; no source is edited.

Light tests pin claims the existing suite could not falsify: row-major positions on a NON-symmetric box, and the
processor parameters read from the REAL processor. Heavy tests (real Qwen3.5-0.8B checkpoint, CPU) pin the visual
protection channel UNDER REAL EVICTION, which the existing engine test never reaches.

A strict xfail records a defect found by this review. It stays visible as xfail, never as PASS. When the defect is
fixed the test XPASSes, which fails the run, so the marker must then be removed on purpose.
cpu_pytorch_reference_not_vllm.
"""
import hashlib
from pathlib import Path

import pytest

from vlmbench.infrastructure import visual_mapping as vm
from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "000000039769.png"
HAVE_CKPT = (CKPT / "config.json").is_file()
HAVE_PROCESSOR = (CKPT / "preprocessor_config.json").is_file()
HAVE_HEAVY = HAVE_CKPT and FIXTURE.is_file()
QUESTION = "What is shown in this picture?"
TEXT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
_HEAVY = pytest.mark.skipif(not HAVE_HEAVY, reason=f"official checkpoint {CKPT} or fixture {FIXTURE} not present")


# --- light: mapping arithmetic ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def geom256():
    geom, reason = vm.build_geometry(256, 256, [1, 16, 16], patch_size=16, merge_size=2,
                                     min_pixels=65536, max_pixels=16777216)
    assert reason is None
    return geom


def test_asymmetric_box_positions_follow_row_major_order(geom256):
    # 2 merged rows x 4 merged columns, grid 8 wide. A transposed formula gives a different set. A square box at the
    # origin does NOT show this (the set is symmetric under transposition), which is why the existing test cannot.
    positions, reason, _info = vm.region_positions((0.0, 0.0, 0.25, 0.5), geom256, (10, 10 + 64))
    assert reason is None
    assert positions == sorted(10 + r * 8 + c for r in range(2) for c in range(4))


def test_a_boundary_is_not_gained_from_float_rounding_up():
    # 800 px image, merged cell 32 px. The box ends at 224 px = the start of cell 7: cell 7 has zero overlap.
    # 224/800 * 800 is 224.00000000000003 in floating point, so the tolerance is what keeps cell 7 out.
    geom, reason = vm.build_geometry(800, 800, [1, 50, 50], patch_size=16, merge_size=2,
                                     min_pixels=65536, max_pixels=16777216)
    assert reason is None
    cells, reason, _info = vm.box_cells((0.0, 0.0, 224 / 800, 0.5), geom)
    assert reason is None
    assert max(r for r, _c in cells) == 6


def test_a_boundary_is_not_lost_from_float_rounding_down():
    # 704 px image. The box starts at 480 px = the start of cell 15; 480/704 * 704 is 479.99999999999994.
    geom, reason = vm.build_geometry(704, 704, [1, 44, 44], patch_size=16, merge_size=2,
                                     min_pixels=65536, max_pixels=16777216)
    assert reason is None
    cells, reason, _info = vm.box_cells((480 / 704, 0.0, 1.0, 0.5), geom)
    assert reason is None
    assert min(r for r, _c in cells) == 15


@pytest.mark.skipif(not HAVE_PROCESSOR, reason=f"official checkpoint processor not present at {CKPT}")
def test_processor_parameters_are_read_from_the_real_processor():
    # processor_params feeds the geometry used for protection. The mapping tests above pass their own constants,
    # so only this test checks the processor-to-parameter mapping (size.shortest_edge -> min, longest_edge -> max).
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(CKPT))
    params, reason = vm.processor_params(proc.image_processor)
    assert reason is None
    assert params == {"patch_size": 16, "merge_size": 2, "min_pixels": 65536, "max_pixels": 16777216}


def test_sub_tolerance_box_straddling_a_cell_boundary_keeps_its_positive_overlap(geom256):
    e = 1e-10  # pixels: a positive-area box, far below one pixel, straddling the boundary at 32 px
    ymin, ymax = (32 - e) / 256, (32 + e) / 256
    cells, reason, _info = vm.box_cells((ymin, 0.0, ymax, 0.5), geom256)
    assert reason is None
    assert cells, "a positive-area box must map to the cells it overlaps"


# --- heavy: visual protection under real eviction (real 0.8B checkpoint, CPU) ----------------------

@pytest.fixture(scope="module")
def image_bytes():
    return FIXTURE.read_bytes()


@pytest.fixture(scope="module")
def sha(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def _image_pool(request_id, sha256, box=(0.0, 0.0, 0.5, 0.5)):
    pool = EvidencePool(request_id)  # S1 binds evidence to the request id: it must match
    pool.add_evidence(EvidenceItem(evidence_id="ev_region", modality="image_region", content="top-left quadrant",
                                   source_sha256=sha256, verified=True, bounding_box=box))
    return pool


def _engine(arm):
    return RealSevenArmEngine(method_arm=arm, device="cpu", checkpoint_dir=CKPT, stop_at_eos=False,
                              h2o_min_budget=8, h2o_budget_ratio=0.6, evidence_policy=EvidencePolicy())


@pytest.fixture(scope="module")
def no_da_engine():
    e = _engine("starlight_s1_no_da")  # no DA scope: the visual protection is the only large protected set
    yield e
    e.close()


@pytest.fixture(scope="module")
def starlight_engine():
    e = _engine("starlight_s1")
    yield e
    e.close()


@_HEAVY
def test_visual_protection_is_retained_under_real_eviction_on_the_no_da_arm(no_da_engine, image_bytes, sha):
    res = no_da_engine.execute_request(request_id="r4b_nda", image_present=True, image_bytes=image_bytes,
                                       question=QUESTION, max_new_tokens=3, evidence_pool=_image_pool("r4b_nda", sha))
    block = res.telemetry["starlight"]
    assert block["visual_evidence_protection"] == "SUPPORTED"
    # Precondition: eviction must actually happen, otherwise "retained" says nothing about protection.
    assert block["evicted_total"] > 0, "no eviction happened: the retention check would be vacuous"
    assert block["kv_budget_violated"] is False
    protected = len(block["visual_protected_positions"])
    assert protected > 0
    retained = block["visual_protected_retained_final_by_layer"]
    assert retained and all(count == protected for count in retained.values()), retained


@_HEAVY
@pytest.mark.xfail(strict=True, reason=(
    "wF:p9 R-B-HIGH-1 (PRE-EXISTING at c65b84d): on an image prompt the scripted DA FOCUS scope is the whole visual "
    "segment, so 321 of 322 entries are protected with NO evidence declared; every step is INFEASIBLE and nothing is "
    "evicted. Owner decision needed on the FOCUS scope before any image STARLIGHT result is read as sparsity."))
def test_starlight_s1_on_an_image_prompt_evicts_when_the_budget_is_feasible(starlight_engine, image_bytes):
    res = starlight_engine.execute_request(request_id="r4b_full", image_present=True, image_bytes=image_bytes,
                                           question=QUESTION, max_new_tokens=3, evidence_pool=None)
    block = res.telemetry["starlight"]
    assert block["evicted_total"] > 0
    assert block["kv_budget_violated"] is False


@_HEAVY
def test_speculation_ablation_is_honoured_on_the_serving_mtp_arm():
    e = RealSevenArmEngine(method_arm="mtp", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False)
    try:
        e._speculation_enabled = False
        ids = list(e.tokenizer.encode(TEXT))
        res = e.execute_request(request_id="r4b_mtp_off", prompt_token_ids=ids, image_present=False,
                                max_new_tokens=4, evidence_pool=None)
        assert res.telemetry["speculation_executed"] is False
    finally:
        e.close()


# --- heavy: grounding protection under speculation, with a negative control --------------------------

TEXT_IDS_NOTE = "keyword 'Berlin' (token 12) is evicted in every full-attention layer when nothing protects it"


@_HEAVY
def test_protected_keyword_is_evicted_without_protection_and_kept_with_it_on_the_speculative_path():
    # The existing invariant test protects 'Paris' (token 5). An unprotected run never evicts token 5, so that test
    # passes with protection removed. This one keeps a keyword that the unprotected run DOES evict (negative control).
    import numpy as np
    from vlmbench.infrastructure.starlight import grounding_protection_with_report
    text = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
    e = RealSevenArmEngine(method_arm="starlight_s1_mtp", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False,
                           h2o_min_budget=16, h2o_budget_ratio=0.5,
                           evidence_policy=EvidencePolicy(accept_declared=True))
    try:
        ids = list(e.tokenizer.encode(text))
        pool = EvidencePool("r4b_spec")
        pool.add_evidence(EvidenceItem(evidence_id="ev_berlin", modality="text_span",
                                       content="Berlin is the capital of Germany",
                                       source_sha256=hashlib.sha256(text.encode()).hexdigest(), verified=True),
                          keywords=["Berlin"])
        protected, _report = grounding_protection_with_report(
            pool, ids, e.tokenizer.encode, "r4b_spec", None, e.evidence_policy,
            image_present=False, cap_policy=e.grounding_cap_policy)
        wanted = set(protected)
        assert wanted, "the keyword must protect a position"

        def run(grounding):
            gone = set()
            real = e._pol_apply

            def spy(st, plans, next_orig):
                before = {l: set(np.asarray(st.orig[l]).tolist()) for l in st.full_layers}
                real(st, plans, next_orig)
                for l in st.full_layers:
                    gone.update(wanted & (before[l] - set(np.asarray(st.orig[l]).tolist())))

            e._pol_apply = spy
            try:
                e._speculative_policy_mtp_greedy("starlight", ids, 24, grounding=grounding)
            finally:
                del e._pol_apply
            return gone

        baseline_evicted = run({})
        assert baseline_evicted, f"negative control failed: nothing evicted the keyword ({TEXT_IDS_NOTE})"
        protected_evicted = run(protected)
        assert not protected_evicted, f"protected original positions were evicted under speculation: {protected_evicted}"
    finally:
        e.close()
