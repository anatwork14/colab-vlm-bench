"""ITER3 B2: MTP speculation rollback and STARLIGHT protection under speculation, on the OFFICIAL
Qwen3.5-0.8B checkpoint (hooked forward, real Gated-DeltaNet and full-attention states).

Test kinds, labelled in each test name:
  * pinv_  POLICY-INVARIANT: rollback restores KV, Gated-DeltaNet and policy state bit-for-bit; the
           STARLIGHT protection holds at every speculative step; decisions match the non-speculative run.
  * inferc_ INFERENCE-CONSISTENCY: committed tokens equal the non-speculative greedy stream.

Partial acceptance, full rejection, rejection after accepts, and continued decoding after rollback are
exercised by forced drafts (oracle hooks: the proposal is the target's own greedy token, or a corrupted
one). This tests the verify/rollback code, not how often the MTP head is right.
Mutation checks (deliberately corrupted rollback) are in test_iter3_b2_mutation.py.
cpu_pytorch_reference_not_vllm. No speed claim, no vLLM evidence.
"""
from pathlib import Path

import pytest
import torch

from vlmbench.infrastructure.real_engine import RealSevenArmEngine, _snapshot_dynamic_cache

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
pytestmark = pytest.mark.skipif(
    not (CKPT / "config.json").is_file(), reason=f"official checkpoint not present at {CKPT}")

TEXT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
N = 40


@pytest.fixture(scope="module")
def star_engine():
    eng = RealSevenArmEngine(method_arm="starlight_s1_mtp", device="cpu", checkpoint_dir=CKPT,
                             stop_at_eos=False)
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def prompt(star_engine):
    if star_engine.tokenizer is None:
        pytest.skip("official tokenizer not available")
    return list(star_engine.tokenizer.encode(TEXT))


def _normal(engine, prompt_ids, n):
    """Non-speculative reference of the SAME arm: the normal decode loop, with a policy trace."""
    engine._speculation_enabled = False
    engine._policy_trace = []
    try:
        res = engine.execute_request("b2_ref", prompt_token_ids=prompt_ids, max_new_tokens=n,
                                     image_present=False)
        trace = list(engine._policy_trace)
    finally:
        engine._speculation_enabled = True
        engine._policy_trace = None
    return list(res.tokens_generated), trace


def _by_position(trace, key):
    out = {}
    for rec in trace:
        assert rec["P"] not in out, "one trace record per logical position"
        out[rec["P"]] = rec[key]
    return out


def _spec(engine, kind, prompt_ids, n, draft_override=None):
    return engine._speculative_policy_mtp_greedy(
        kind, prompt_ids, n, draft_override=draft_override, record_trace=True)


def _equal(x, y):
    if isinstance(x, torch.Tensor):
        return torch.equal(x, y)
    if isinstance(x, dict):
        return x.keys() == y.keys() and all(_equal(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(_equal(a, b) for a, b in zip(x, y))
    return x == y


def _block_check(eng, prompt_ids, kind="starlight"):
    """Eviction-free 2-row verify block, then rollback. Reports whether the block changed state and
    whether the rollback restored the cache (KV + Gated-DeltaNet) and the policy digests exactly."""
    st = eng._pol_new_state(kind, len(prompt_ids))
    logits, _mtp = eng._pol_prefill(st, prompt_ids)
    pending = int(torch.argmax(logits).item())
    wrong = (pending + 7) % eng.config.vocab_size
    assert eng._pol_block_is_eviction_free(st), "fixture must be eviction-free for a block"
    snap = eng._pol_snapshot(st)
    before_cache = _snapshot_dynamic_cache(st.cache)
    before_dig = eng._pol_digests(st)
    eng._pol_block(st, pending, wrong)
    mid_cache = _snapshot_dynamic_cache(st.cache)
    mid_dig = eng._pol_digests(st)
    eng._pol_restore(st, snap)
    after_cache = _snapshot_dynamic_cache(st.cache)
    after_dig = eng._pol_digests(st)
    kinds = {b[0] for b in before_cache}
    return {
        "kinds": kinds,
        "changed": (not _equal(before_cache, mid_cache)) and before_dig != mid_dig,
        "restored_cache": _equal(before_cache, after_cache),
        "restored_policy": before_dig == after_dig,
    }


def _protected_prompt_positions(st):
    """Original prompt positions protected by the scripted DA scopes (sink/focus/question)."""
    out = set()
    for _label, s, e in st.da_spans:
        out.update(range(int(s), int(e)))
    return out


def test_pinv_block_rollback_restores_kv_gdn_and_starlight_state_bit_for_bit(star_engine, prompt):
    # [POLICY-INVARIANT] One eviction-free block on the official model, then rollback. The check is only
    # meaningful if the block changed state (not vacuous) and covers both attention and Gated-DeltaNet.
    res = _block_check(star_engine, prompt)
    assert {"dynamic", "linear"} <= res["kinds"]
    assert res["changed"], "test would be vacuous: the block did not change state"
    assert res["restored_cache"], "rollback did not restore KV and Gated-DeltaNet state exactly"
    assert res["restored_policy"], "rollback did not restore the STARLIGHT policy digests exactly"


def test_inferc_starlight_partial_acceptance_then_rejection_after_accept_matches_greedy(star_engine, prompt):
    # [INFERENCE-CONSISTENCY + POLICY-INVARIANT] Half of the drafts accepted, half rejected. A rejection
    # must follow an accepted block, and the committed stream and the decisions must equal the
    # non-speculative run at every common logical position.
    ref, ref_trace = _normal(star_engine, prompt, N)
    vocab = star_engine.config.vocab_size
    flip = {"i": 0}

    def half_reject(draft, i, n_emitted):
        flip["i"] += 1
        return ref[n_emitted] if flip["i"] % 2 else (ref[n_emitted] + 1) % vocab

    spec = _spec(star_engine, "starlight", prompt, N, draft_override=half_reject)
    assert spec["accepted"] >= 1 and spec["rejected"] >= 1
    assert spec["tokens"] == ref
    via = [r["via"] for r in spec["trace"]]
    assert "accept_block" in via, "no accepted block: partial acceptance not exercised"
    assert "refeed" in via[via.index("accept_block"):], "no rejection after a prior accept"
    spec_dec = _by_position(spec["trace"], "decision_digest")
    ref_dec = _by_position(ref_trace, "decision_digest")
    common = sorted(set(spec_dec) & set(ref_dec))
    assert len(common) >= 10
    assert [p for p in common if spec_dec[p] != ref_dec[p]] == []
    assert spec["verifier_calls"] == 1 + spec["proposed"] + spec["rejected"] + spec["fallback_steps"]


def test_inferc_starlight_continued_decoding_after_repeated_rollback_matches_greedy(star_engine, prompt):
    # [INFERENCE-CONSISTENCY] Long horizon: every third verified draft is rejected, so decoding
    # continues after many rollbacks, across the eviction regime. Tokens must still equal greedy.
    n = 60
    ref, _ = _normal(star_engine, prompt, n)
    vocab = star_engine.config.vocab_size
    seen = {"i": 0}

    def reject_every_third(draft, i, n_emitted):
        seen["i"] += 1
        return (ref[n_emitted] + 1) % vocab if seen["i"] % 3 == 0 else ref[n_emitted]

    spec = _spec(star_engine, "starlight", prompt, n, draft_override=reject_every_third)
    assert spec["tokens"] == ref
    assert spec["rejected"] >= 2, "fixture must roll back more than once"
    assert spec["fallback_steps"] >= 1, "fixture must include eviction steps"


def test_pinv_speculation_never_evicts_a_protected_position(star_engine, prompt, monkeypatch):
    # [POLICY-INVARIANT] Spy on every compaction the speculative path applies (block rows and single
    # rows, accept and reject paths). A protected prompt position present before a step must survive it.
    ref, _ = _normal(star_engine, prompt, N)
    vocab = star_engine.config.vocab_size
    flip = {"i": 0}

    def half_reject(draft, i, n_emitted):
        flip["i"] += 1
        return ref[n_emitted] if flip["i"] % 2 else (ref[n_emitted] + 1) % vocab

    violations: list[tuple[int, int]] = []
    stats = {"applies": 0, "evicted_unprotected": 0, "evicted_protected_attempts": 0}
    real_apply = star_engine._pol_apply

    def spy(st, plans, next_orig):
        protected = _protected_prompt_positions(st)
        for l in st.full_layers:
            before = {int(p) for p in st.orig[l]}
            kept = {int(st.orig[l][i]) for i in plans[l] if i < len(st.orig[l])}
            removed = before - kept
            stats["applies"] += 1
            stats["evicted_unprotected"] += len(removed - protected)
            for p in (removed & protected):
                violations.append((l, p))
        return real_apply(st, plans, next_orig)

    monkeypatch.setattr(star_engine, "_pol_apply", spy)
    spec = _spec(star_engine, "starlight", prompt, N, draft_override=half_reject)
    assert spec["tokens"] == ref
    assert stats["applies"] > 0
    assert stats["evicted_unprotected"] > 0, "test would be vacuous: no eviction happened"
    assert violations == [], f"protected positions evicted under speculation: {violations[:5]}"


def test_inferc_speculative_route_reports_its_starlight_policy_and_budget(star_engine, prompt):
    # [INFERENCE-CONSISTENCY + POLICY-INVARIANT] The speculative route must run the STARLIGHT policy
    # with its real budget (not a silent fallback), and still commit the greedy stream.
    ref, _ = _normal(star_engine, prompt, N)
    res = star_engine.execute_request("b2_route", prompt_token_ids=prompt, max_new_tokens=N,
                                      image_present=False)
    assert res.telemetry["speculation_executed"] is True
    policy = res.telemetry["policy"]
    assert policy["kind"] == "starlight"
    cfg = star_engine.h2o_config
    assert policy["budget_total"] == max(cfg.min_budget, int(len(prompt) * cfg.budget_ratio))
    assert res.tokens_generated == ref


# ITER3 R2 (lead ruling F-1): the shared retained set makes the per-layer lengths identical, so this
# must now COMPLETE. It was a strict xfail in ITER3 B2 (the defect); the xfail is removed.
def test_F1_protected_veto_that_differs_across_layers_completes(star_engine, prompt, monkeypatch):
    # A DA focus span over prompt position 16. ITER3 R2: under the SHARED retained set, position 16 is a
    # candidate on every full-attention layer (measured), so the veto is applied identically on all of them.
    # The earlier per-layer defect (veto on layer 3 only) can no longer occur; this test checks completion.
    real_new_state = star_engine._pol_new_state

    def with_focus_span_on_16(kind, prompt_len):
        st = real_new_state(kind, prompt_len)
        st.da_spans = tuple(st.da_spans) + (("focus:mid", 16, 17),)
        return st

    monkeypatch.setattr(star_engine, "_pol_new_state", with_focus_span_on_16)
    res = _spec(star_engine, "starlight", prompt, N, draft_override=None)
    assert res["committed"] == N
