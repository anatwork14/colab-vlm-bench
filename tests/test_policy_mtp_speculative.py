"""ITER2 Stage 2(b): policy-state rollback for speculative MTP on the official checkpoint.

Arms h2o_mtp and starlight_s1_mtp, text-only requests, cpu_pytorch_reference_not_vllm.
Speculation runs a 2-row verify block only on EVICTION-FREE steps (no full-attention layer would
evict for either row). Otherwise the pending token is fed alone and no draft is verified
(counted as fallback_steps). Rejects restore the cache AND the policy state (tracker arrays,
STARLIGHT original-position map) bit-for-bit, then refeed the pending token through the normal
single-row path.

Exactness claims are scoped: on a forced reject the state after refeed equals the non-speculative
run's state at the same logical position bit-for-bit (same code path). Accepted blocks use a
2-row attention, so their numerics can differ; the weaker invariant (identical retained decisions)
is what the mixed test asserts.
"""
from pathlib import Path

import pytest
import torch

from vlmbench.infrastructure.real_engine import (
    RealSevenArmEngine,
    _restore_dynamic_cache,
    _snapshot_dynamic_cache,
)

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
pytestmark = pytest.mark.skipif(
    not (CKPT / "config.json").is_file(), reason=f"official checkpoint not present at {CKPT}")

TEXT = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
N = 40


@pytest.fixture(scope="module")
def h2o_engine():
    eng = RealSevenArmEngine(method_arm="h2o_mtp", device="cpu", checkpoint_dir=CKPT, stop_at_eos=False)
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def star_engine():
    eng = RealSevenArmEngine(method_arm="starlight_s1_mtp", device="cpu", checkpoint_dir=CKPT,
                             stop_at_eos=False)
    yield eng
    eng.close()


@pytest.fixture(scope="module")
def prompt(h2o_engine):
    if h2o_engine.tokenizer is None:
        pytest.skip("official tokenizer not available")
    return list(h2o_engine.tokenizer.encode(TEXT))


def _normal(engine, prompt_ids, n):
    """Non-speculative reference of the SAME arm: the normal decode loop, with a policy trace."""
    engine._speculation_enabled = False
    engine._policy_trace = []
    try:
        res = engine.execute_request("pol_ref", prompt_token_ids=prompt_ids,
                                     max_new_tokens=n, image_present=False)
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


def test_h2o_all_rejects_commit_normal_tokens_and_match_policy_state_bit_for_bit(h2o_engine, prompt):
    ref, ref_trace = _normal(h2o_engine, prompt, N)
    assert len(ref) >= 32
    vocab = h2o_engine.config.vocab_size

    def reject_every_draft(draft, i, n_emitted):
        return (ref[n_emitted] + 1) % vocab  # guaranteed != the target's next token

    spec = _spec(h2o_engine, "h2o", prompt, N, draft_override=reject_every_draft)
    assert spec["tokens"] == ref
    assert spec["accepted"] == 0 and spec["rejected"] >= 1
    assert spec["fallback_steps"] >= 1, "fixture must include eviction steps"
    spec_digests = _by_position(spec["trace"], "policy_digest")
    ref_digests = _by_position(ref_trace, "policy_digest")
    common = sorted(set(spec_digests) & set(ref_digests))
    assert len(common) >= 10
    mismatched = [p for p in common if spec_digests[p] != ref_digests[p]]
    assert mismatched == [], f"policy state differs from non-speculative run at P={mismatched[:5]}"


def test_h2o_mixed_accepts_and_rejects_keep_retained_decisions_identical(h2o_engine, prompt):
    ref, ref_trace = _normal(h2o_engine, prompt, N)
    vocab = h2o_engine.config.vocab_size
    flip = {"i": 0}

    def half_reject(draft, i, n_emitted):
        flip["i"] += 1
        return ref[n_emitted] if flip["i"] % 2 else (ref[n_emitted] + 1) % vocab

    spec = _spec(h2o_engine, "h2o", prompt, N, draft_override=half_reject)
    assert spec["accepted"] >= 1 and spec["rejected"] >= 1
    assert spec["tokens"] == ref
    spec_dec = _by_position(spec["trace"], "decision_digest")
    ref_dec = _by_position(ref_trace, "decision_digest")
    common = sorted(set(spec_dec) & set(ref_dec))
    assert len(common) >= 10
    assert [p for p in common if spec_dec[p] != ref_dec[p]] == []


def test_counts_are_verifier_derived_and_consistent(h2o_engine, prompt):
    vocab = h2o_engine.config.vocab_size
    ref, _ = _normal(h2o_engine, prompt, N)
    flip = {"i": 0}

    def alternate(draft, i, n_emitted):
        flip["i"] += 1
        return ref[n_emitted] if flip["i"] % 2 else (ref[n_emitted] + 1) % vocab

    spec = _spec(h2o_engine, "h2o", prompt, N, draft_override=alternate)
    assert spec["committed"] == len(spec["tokens"]) == len(ref)
    assert spec["proposed"] == spec["accepted"] + spec["rejected"]
    # Every target forward is counted: prefill, one per verify block, one per refeed or fallback.
    assert spec["verifier_calls"] == 1 + spec["proposed"] + spec["rejected"] + spec["fallback_steps"]
    assert spec["fallback_steps"] >= 1


def test_block_rollback_restores_cache_and_policy_state_bit_for_bit(h2o_engine, prompt):
    st = h2o_engine._pol_new_state("h2o", len(prompt))
    logits, _mtp = h2o_engine._pol_prefill(st, prompt)
    pending = int(torch.argmax(logits).item())
    snap = h2o_engine._pol_snapshot(st)
    digest_before = h2o_engine._pol_digest(st)
    cache_before = _snapshot_dynamic_cache(st.cache)
    h2o_engine._pol_block(st, pending, (pending + 7) % h2o_engine.config.vocab_size)
    assert h2o_engine._pol_digest(st) != digest_before, "the block must have changed the state"
    h2o_engine._pol_restore(st, snap)
    assert h2o_engine._pol_digest(st) == digest_before
    cache_after = _snapshot_dynamic_cache(st.cache)
    assert len(cache_after) == len(cache_before)
    for a, b in zip(cache_after, cache_before):
        assert a[0] == b[0]
        if a[0] == "dynamic":
            assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])


def test_starlight_all_rejects_match_normal_tokens_and_policy_state(star_engine, prompt):
    ref, ref_trace = _normal(star_engine, prompt, N)
    vocab = star_engine.config.vocab_size

    def reject_every_draft(draft, i, n_emitted):
        return (ref[n_emitted] + 1) % vocab

    spec = _spec(star_engine, "starlight", prompt, N, draft_override=reject_every_draft)
    assert spec["tokens"] == ref
    assert spec["fallback_steps"] >= 1
    spec_digests = _by_position(spec["trace"], "policy_digest")
    ref_digests = _by_position(ref_trace, "policy_digest")
    common = sorted(set(spec_digests) & set(ref_digests))
    assert len(common) >= 10
    assert [p for p in common if spec_digests[p] != ref_digests[p]] == []


def test_execute_request_h2o_mtp_text_only_runs_policy_speculation(h2o_engine, prompt):
    ref, _ = _normal(h2o_engine, prompt, N)
    res = h2o_engine.execute_request("pol_route", prompt_token_ids=prompt,
                                     max_new_tokens=N, image_present=False)
    assert res.telemetry["speculation_executed"] is True
    spec = res.telemetry["speculation"]
    assert spec["committed"] == len(res.tokens_generated)
    assert spec["verifier_calls"] == 1 + spec["proposed"] + spec["rejected"] + spec["fallback_steps"]
    assert res.tokens_generated == ref
    assert res.gates["MTP_CORRECTNESS_PASS"] == "CPU_REFERENCE_SPECULATION_EXECUTED_NOT_VLLM"
