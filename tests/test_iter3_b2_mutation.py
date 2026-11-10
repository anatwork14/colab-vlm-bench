"""ITER3 B2 MUTATION TESTS (POLICY-INVARIANT detectors). Each test deliberately corrupts one piece of
the speculative rollback or the STARLIGHT protection (monkeypatch, scoped to the test), then asserts that
the corresponding detector REPORTS the corruption. Each test first runs the unmutated detector and
asserts it passes, so a detector that always fails cannot pass as a mutation check.

Mutations:
  M1  KV + Gated-DeltaNet cache restore is a no-op            -> bit-for-bit cache check must fail
  M2  Gated-DeltaNet state is NOT rolled back (KV only)       -> bit-for-bit cache check must fail
  M3  Policy restore skips the tracker and the original map   -> policy digest check must fail
  M4  Plans ignore DA protection (candidates = all H2O picks) -> protected-position spy must fire

cpu_pytorch_reference_not_vllm. Official Qwen3.5-0.8B checkpoint (hooked forward).
"""
from pathlib import Path

import pytest

import vlmbench.infrastructure.real_engine as real_engine_mod
from vlmbench.infrastructure.real_engine import (
    RealSevenArmEngine,
    _restore_dynamic_cache,
    _snapshot_dynamic_cache,
)
from test_iter3_b2_mtp_rollback import (  # noqa: E402  (shared detector helpers)
    N,
    TEXT,
    _block_check,
    _protected_prompt_positions,
    _spec,
)

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
pytestmark = pytest.mark.skipif(
    not (CKPT / "config.json").is_file(), reason=f"official checkpoint not present at {CKPT}")


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


def test_M1_noop_cache_restore_is_detected_by_bit_for_bit_check(star_engine, prompt, monkeypatch):
    baseline = _block_check(star_engine, prompt)
    assert baseline["restored_cache"] and baseline["restored_policy"], "baseline must pass first"
    monkeypatch.setattr(real_engine_mod, "_restore_dynamic_cache", lambda cache, snap: None)
    mutated = _block_check(star_engine, prompt)
    assert mutated["restored_cache"] is False, "M1 not detected: no-op restore passed the cache check"


def test_M2_skipped_gdn_rollback_is_detected_by_bit_for_bit_check(star_engine, prompt, monkeypatch):
    baseline = _block_check(star_engine, prompt)
    assert {"dynamic", "linear"} <= baseline["kinds"] and baseline["restored_cache"]

    def kv_only_restore(cache, snap):
        # Keep the current (block-modified) Gated-DeltaNet state; roll back attention KV only.
        current = _snapshot_dynamic_cache(cache)
        mixed = [cur if entry[0] == "linear" else entry for entry, cur in zip(snap, current)]
        _restore_dynamic_cache(cache, mixed)

    monkeypatch.setattr(real_engine_mod, "_restore_dynamic_cache", kv_only_restore)
    mutated = _block_check(star_engine, prompt)
    assert mutated["changed"], "mutation test would be vacuous: block did not change state"
    assert mutated["restored_cache"] is False, "M2 not detected: GDN state rollback skipped, check passed"


def test_M3_policy_restore_that_skips_tracker_is_detected_by_digest_check(star_engine, prompt, monkeypatch):
    baseline = _block_check(star_engine, prompt)
    assert baseline["restored_policy"], "baseline must pass first"

    def cache_only_restore(st, snap):
        _restore_dynamic_cache(st.cache, snap["cache"])
        st.P = snap["P"]  # tracker scores, evicted masks and the STARLIGHT original map are NOT restored

    monkeypatch.setattr(star_engine, "_pol_restore", lambda st, snap: cache_only_restore(st, snap))
    mutated = _block_check(star_engine, prompt)
    assert mutated["changed"], "mutation test would be vacuous: block did not change state"
    assert mutated["restored_policy"] is False, "M3 not detected: policy digests still matched"


def test_M4_plans_that_ignore_da_protection_are_detected_by_the_protection_spy(star_engine, prompt, monkeypatch):
    # Unmutated baseline FIRST, then the mutant, on the same injected DA span.
    #
    # ITER3 R2 re-derivation (lead ruling F-1, shared retained set): the injected span was position 16,
    # which the per-layer H2O rule evicted on layer 3 only. Under the SHARED candidate set, position 16
    # is a candidate on every full-attention layer, so the mutant is no longer detected through
    # position 16 (measured: the mutant's only eviction on this 19-token prompt is generated position 23,
    # evicted on every full-attention layer at each eviction step; default DA protection already keeps
    # the other candidates). The span therefore moves to position 23. The spy checks protection
    # for every position, so generated positions are covered by the same rule.
    real_new_state = star_engine._pol_new_state
    protected_pos = 23

    def with_focus_span(kind, prompt_len, **kw):
        st = real_new_state(kind, prompt_len, **kw)
        st.da_spans = tuple(st.da_spans) + (("focus:mid", protected_pos, protected_pos + 1),)
        return st

    real_build = star_engine._starlight_build_step_plans
    real_apply = star_engine._pol_apply

    def run(mutant: bool):
        violations: list[tuple[int, int]] = []
        stats = {"evicted_unprotected": 0}

        def spy(st, plans, next_orig):
            protected = _protected_prompt_positions(st)
            for l in st.full_layers:
                before = {int(p) for p in st.orig[l]}
                kept = {int(st.orig[l][i]) for i in plans[l] if i < len(st.orig[l])}
                removed = before - kept
                stats["evicted_unprotected"] += len(removed - protected)
                violations.extend((l, p) for p in removed & protected)
            return real_apply(st, plans, next_orig)

        def ignore_da(tracker, k_lens, orig, row_pos, budget, da_spans, grounding, enabled, **kw):
            # Mutation: the DA scopes are dropped from the protection inputs; the H2O candidates are unchanged.
            return real_build(tracker, k_lens, orig, row_pos, budget, [] if mutant else da_spans,
                              grounding, enabled, **kw)

        with monkeypatch.context() as mp:
            mp.setattr(star_engine, "_pol_new_state", with_focus_span)
            mp.setattr(star_engine, "_pol_apply", spy)
            mp.setattr(star_engine, "_starlight_build_step_plans", ignore_da)
            _spec(star_engine, "starlight", prompt, N, draft_override=None)
        return violations, stats

    base_violations, base_stats = run(mutant=False)
    assert base_stats["evicted_unprotected"] > 0, "baseline would be vacuous: nothing was evicted"
    assert base_violations == [], f"baseline must pass first: protected position evicted {base_violations[:5]}"
    violations, stats = run(mutant=True)
    assert stats["evicted_unprotected"] > 0, "mutation test would be vacuous: nothing was evicted"
    assert violations, "M4 not detected: a protected position was evicted and the spy did not fire"
