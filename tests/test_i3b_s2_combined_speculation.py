"""ITER3 Round B S2 Stage 2, HEAVY (run under heavy.lock): combined methods WITH evidence and WITH image input.

Arms h2o_mtp and starlight_s1_mtp on the OFFICIAL Qwen3.5-0.8B checkpoint, cpu_pytorch_reference_not_vllm.

Evidence on the text-only speculative path is DECLARED, not S1-verified: there are no source bytes to hash, so
under the default policy it protects nothing. It protects only under EvidencePolicy(accept_declared=True), which is
reported as a policy choice. A VERIFIED image region needs image bytes, and an image prompt is UNSUPPORTED on the
speculative path (explicit status and reason; the request runs the diagnostic path).

Test kinds, labelled in each name:
  * pinv_  POLICY-INVARIANT: protected original positions stay retained at every speculative step; a rollback
           restores the policy state, including the original-position map, bit for bit; requests are isolated.
  * inferc_ INFERENCE-CONSISTENCY: committed tokens equal the non-speculative run of the SAME arm and evidence.

Forced drafts (draft_override) test the verify and rollback code, not how often the MTP head is right.
Preserving a visual or text span does NOT prove the model used it.
"""
import hashlib
from pathlib import Path

import numpy as np
import pytest

from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import grounding_protection_with_report

# Labels as strings (the same values as speculative_route.py), so each test fails on its behaviour, not an import.
ROUTE_POLICY_TEXT = "POLICY_EVICTION_FREE_TEXT"
ROUTE_UNSUPPORTED_IMAGE = "UNSUPPORTED_IMAGE_PROMPT"
ROUTE_NOT_SPECULATIVE_DA = "NOT_SPECULATIVE_DA_MTP"

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "000000039769.png"
pytestmark = pytest.mark.skipif(
    not ((CKPT / "config.json").is_file() and FIXTURE.is_file()),
    reason=f"official checkpoint {CKPT} or fixture {FIXTURE} not present",
)

TEXT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
KEYWORD = "Paris"
N = 24


def _pool(request_id, *, sha=None):
    pool = EvidencePool(request_id)
    pool.add_evidence(
        EvidenceItem(evidence_id="ev_paris", modality="text_span", content="Paris is the capital of France",
                     source_sha256=sha or hashlib.sha256(TEXT.encode()).hexdigest(), verified=True),
        keywords=[KEYWORD],
    )
    return pool


def _image_pool(sha):
    pool = EvidencePool("s2c_img")
    pool.add_evidence(EvidenceItem(evidence_id="ev_region", modality="image_region", content="top-left quadrant",
                                   source_sha256=sha, verified=True, bounding_box=(0.0, 0.0, 0.5, 0.5)))
    return pool


def _engine(arm, *, accept_declared, budget=16):
    return RealSevenArmEngine(method_arm=arm, device="cpu", checkpoint_dir=CKPT, stop_at_eos=False,
                              h2o_min_budget=budget, h2o_budget_ratio=0.5,
                              evidence_policy=EvidencePolicy(accept_declared=accept_declared))


@pytest.fixture(scope="module")
def star_accept():
    e = _engine("starlight_s1_mtp", accept_declared=True)
    yield e
    e.close()


@pytest.fixture(scope="module")
def star_big():
    # A larger budget keeps the first decode steps eviction-free, so verified 2-row blocks (accepts) occur
    # before evictions begin. Without this the policy path only falls back and nothing can be rolled back.
    e = _engine("starlight_s1_mtp", accept_declared=True, budget=40)
    yield e
    e.close()


@pytest.fixture(scope="module")
def star_default():
    e = _engine("starlight_s1_mtp", accept_declared=False)
    yield e
    e.close()


@pytest.fixture(scope="module")
def h2o_default():
    e = _engine("h2o_mtp", accept_declared=False)
    yield e
    e.close()


@pytest.fixture(scope="module")
def prompt(star_accept):
    return list(star_accept.tokenizer.encode(TEXT))


def _normal(engine, prompt_ids, pool):
    """Non-speculative reference of the SAME arm and evidence (the normal decode loop).

    The request id must be the pool's own id: S1 source_identity binds the pool to its request, so a reference
    run under another id would protect nothing and would not be a fair reference.
    """
    engine._speculation_enabled = False
    request_id = pool.request_id if pool is not None else "s2c_ref"
    try:
        return list(engine.execute_request(request_id, prompt_token_ids=prompt_ids, max_new_tokens=N,
                                           image_present=False, evidence_pool=pool).tokens_generated)
    finally:
        engine._speculation_enabled = True


def _grounding(engine, prompt_ids, pool, request_id):
    protected, _report = grounding_protection_with_report(
        pool, prompt_ids, engine.tokenizer.encode, request_id, None, engine.evidence_policy,
        image_present=False, cap_policy=engine.grounding_cap_policy)
    return protected


# --- evidence on the policy speculative path ----------------------------------------------------

def test_inferc_text_evidence_runs_policy_speculation_and_matches_the_normal_run(star_accept, prompt):
    pool = _pool("s2c_ev")
    ref = _normal(star_accept, prompt, pool)
    res = star_accept.execute_request("s2c_ev", prompt_token_ids=prompt, image_present=False,
                                      max_new_tokens=N, evidence_pool=pool)
    assert res.telemetry["speculation_executed"] is True
    assert res.telemetry["speculative_route"]["status"] == ROUTE_POLICY_TEXT
    assert list(res.tokens_generated) == ref
    spec = res.telemetry["speculation"]
    assert spec["grounding_protected_positions"] > 0, "accepted declared evidence must protect its keyword spans"


def test_pinv_text_evidence_protects_only_under_accept_declared(star_default, prompt):
    res = star_default.execute_request("s2c_default", prompt_token_ids=prompt, image_present=False,
                                       max_new_tokens=N, evidence_pool=_pool("s2c_default"))
    assert res.telemetry["speculation_executed"] is True
    assert res.telemetry["speculation"]["grounding_protected_positions"] == 0
    assessment = res.telemetry["starlight"]["grounding_evidence_assessments"]["ev_paris"]
    assert assessment["status"] == "declared_unverified"


def test_pinv_h2o_mtp_with_evidence_runs_speculation_and_says_evidence_is_not_applicable(h2o_default, prompt):
    pool = _pool("s2c_h2o")
    ref = _normal(h2o_default, prompt, pool)
    res = h2o_default.execute_request("s2c_h2o", prompt_token_ids=prompt, image_present=False,
                                      max_new_tokens=N, evidence_pool=pool)
    assert res.telemetry["speculation_executed"] is True
    assert list(res.tokens_generated) == ref
    assert res.telemetry["speculation"]["evidence_protection"] == "NOT_APPLICABLE_H2O_ARM"
    assert res.telemetry["speculation"]["grounding_protected_positions"] == 0


# --- image input on the combined speculative path: explicit UNSUPPORTED ----------------------------

def test_pinv_image_prompt_on_combined_method_is_unsupported_on_the_speculative_path(star_accept):
    raw = FIXTURE.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    res = star_accept.execute_request("s2c_img", image_present=True, image_path=FIXTURE,
                                      question="What is shown in this picture?", max_new_tokens=3,
                                      evidence_pool=_image_pool(sha))
    assert res.telemetry["speculation_executed"] is False
    assert res.telemetry["speculative_route"]["status"] == ROUTE_UNSUPPORTED_IMAGE
    assert res.telemetry["speculative_route"]["reason"]
    # The diagnostic path still protects a verified visual span, on its own, S1-gated evidence.
    assert res.telemetry["starlight"]["visual_evidence_protection"] == "SUPPORTED"
    assert res.telemetry["starlight"]["visual_positions_protected"] > 0


def test_da_mtp_is_explicitly_non_speculative_on_text():
    da = RealSevenArmEngine(method_arm="da_mtp", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False)
    try:
        res = da.execute_request("s2c_da", prompt_token_ids=list(da.tokenizer.encode(TEXT)),
                                 image_present=False, max_new_tokens=3)
    finally:
        da.close()
    assert res.telemetry["speculation_executed"] is False
    assert res.telemetry["speculative_route"]["status"] == ROUTE_NOT_SPECULATIVE_DA


# --- rejection after acceptance, atomic rollback including policy state ---------------------------

def test_pinv_rejection_after_acceptance_rolls_back_policy_state_and_original_positions(star_big, prompt, monkeypatch):
    pool = _pool("s2c_rb")
    protected = _grounding(star_big, prompt, pool, "s2c_rb")
    assert protected, "fixture must protect keyword spans"
    ref = _normal(star_big, prompt, pool)
    vocab = star_big.config.vocab_size
    flip = {"i": 0}

    def alternate(draft, i, n_emitted):
        flip["i"] += 1
        return ref[n_emitted] if flip["i"] % 2 else (ref[n_emitted] + 1) % vocab

    restores = {"n": 0, "orig_mismatch": 0}
    real_restore = star_big._pol_restore

    def spy_restore(st, snap):
        real_restore(st, snap)
        restores["n"] += 1
        for layer in st.full_layers:
            if not np.array_equal(np.asarray(st.orig[layer]), np.asarray(snap["orig"][layer])):
                restores["orig_mismatch"] += 1

    monkeypatch.setattr(star_big, "_pol_restore", spy_restore)
    spec = star_big._speculative_policy_mtp_greedy(
        "starlight", prompt, N, draft_override=alternate, record_trace=True, grounding=protected)
    assert spec["accepted"] >= 1 and spec["rejected"] >= 1, "need acceptance before a rejection"
    assert restores["n"] >= 1
    assert restores["orig_mismatch"] == 0, "rollback must restore the original-position map too"
    assert spec["tokens"] == ref


# --- protection invariants under speculation -----------------------------------------------------

def test_pinv_protected_original_positions_stay_retained_at_every_speculative_step(star_accept, prompt, monkeypatch):
    pool = _pool("s2c_inv")
    protected = _grounding(star_accept, prompt, pool, "s2c_inv")
    wanted = set(protected)
    checks = {"n": 0, "missing": 0}
    real_apply = star_accept._pol_apply

    def spy_apply(st, plans, next_orig):
        real_apply(st, plans, next_orig)
        for layer in st.full_layers:
            checks["n"] += 1
            if not wanted <= set(np.asarray(st.orig[layer]).tolist()):
                checks["missing"] += 1

    monkeypatch.setattr(star_accept, "_pol_apply", spy_apply)
    star_accept._speculative_policy_mtp_greedy("starlight", prompt, N, record_trace=True, grounding=protected)
    assert checks["n"] > 0
    assert checks["missing"] == 0, "a protected original position was evicted under speculation"


# --- request-level isolation on the speculative path ---------------------------------------------

def test_pinv_request_isolation_on_the_speculative_path(star_accept, prompt):
    no_evidence_ref = _normal(star_accept, prompt, None)
    a1 = star_accept.execute_request("s2c_iso_a", prompt_token_ids=prompt, image_present=False,
                                     max_new_tokens=N, evidence_pool=_pool("s2c_iso_a"))
    b = star_accept.execute_request("s2c_iso_b", prompt_token_ids=prompt, image_present=False,
                                    max_new_tokens=N, evidence_pool=None)
    a2 = star_accept.execute_request("s2c_iso_a", prompt_token_ids=prompt, image_present=False,
                                     max_new_tokens=N, evidence_pool=_pool("s2c_iso_a"))
    assert b.telemetry["speculation"]["grounding_protected_positions"] == 0
    assert list(b.tokens_generated) == no_evidence_ref
    assert list(a2.tokens_generated) == list(a1.tokens_generated)
    assert a2.telemetry["speculation"]["grounding_protected_positions"] == a1.telemetry["speculation"][
        "grounding_protected_positions"] > 0
