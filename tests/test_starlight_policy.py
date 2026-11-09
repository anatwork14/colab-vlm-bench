"""Pure STARLIGHT Stage 1 policy contract (backend-agnostic, no torch).

Owner contract: eligible_to_evict = h2o_candidate AND NOT internally_protected, where
internal protection = DA-protected U grounding-protected. External eviction = H2O candidates.
Budget rule: a feasible plan retains at most `budget` entries. The engine computes the H2O
candidate set at budget - |protected|, so a consistent candidate set satisfies this; a
candidate set that would overflow the budget is rejected (ValueError), never accepted.
Every label is cpu_pytorch_reference_not_vllm; nothing here is GPU or vLLM evidence.
"""

import random

import pytest

from vlmbench.infrastructure.starlight import (
    DECISION_EVICT,
    DECISION_RETAIN_H2O_KEPT,
    DECISION_RETAIN_PROTECTED_VETOED,
    STATUS_INFEASIBLE,
    STATUS_OK,
    plan_starlight_layer,
    summarize_plans,
)


def _plan(length, budget, candidates, da=None, grounding=None, enabled=("da", "grounding"), layer=3):
    signals = {"da": da or {}, "grounding": grounding or {}}
    return plan_starlight_layer(
        layer=layer, length=length, budget=budget,
        h2o_candidates=candidates, signals=signals, enabled_signals=enabled,
    )


def test_no_candidates_means_no_eviction_and_status_ok():
    plan = _plan(length=10, budget=10, candidates=[])
    assert plan.status == STATUS_OK
    assert plan.evicted == ()
    assert plan.retained == tuple(range(10))
    assert all(d.decision == DECISION_RETAIN_H2O_KEPT for d in plan.decisions)


def test_eligible_is_candidate_and_not_protected():
    # Candidates 1,2,3,4. Entry 2 is DA-protected, entry 4 is grounding-protected.
    plan = _plan(
        length=8, budget=6, candidates=[1, 2, 3, 4],
        da={2: ("sink_global",)}, grounding={4: ("ev_a",)},
    )
    assert plan.evicted == (1, 3)
    assert plan.retained == (0, 2, 4, 5, 6, 7)
    assert len(plan.retained) <= plan.budget


def test_da_h2o_conflict_resolved_by_protection_and_audited():
    plan = _plan(length=8, budget=5, candidates=[0, 1, 2, 3], da={0: ("sink_global",)})
    d0 = plan.decisions[0]
    assert d0.h2o_candidate is True
    assert d0.internally_protected is True
    assert d0.decision == DECISION_RETAIN_PROTECTED_VETOED
    assert d0.signals_fired == ("da:sink_global",)
    assert "vetoed" in d0.reason
    assert 0 in plan.retained
    assert plan.h2o_candidates_vetoed == 1
    assert plan.conflicts_by_source == {"da": 1, "grounding": 0}


def test_grounding_h2o_conflict_resolved_by_protection_and_audited():
    plan = _plan(length=8, budget=6, candidates=[5, 6, 7], grounding={6: ("ev_x",)})
    d6 = plan.decisions[6]
    assert d6.decision == DECISION_RETAIN_PROTECTED_VETOED
    assert d6.signals_fired == ("grounding:ev_x",)
    assert plan.conflicts_by_source == {"da": 0, "grounding": 1}
    assert 6 in plan.retained


def test_entry_protected_by_both_sources_is_one_vetoed_entry_two_conflict_events():
    plan = _plan(length=6, budget=5, candidates=[2, 3], da={2: ("focus",)}, grounding={2: ("ev",)})
    assert plan.h2o_candidates_vetoed == 1
    assert plan.conflicts_by_source == {"da": 1, "grounding": 1}
    assert plan.decisions[2].signals_fired == ("da:focus", "grounding:ev")
    assert plan.evicted == (3,)


def test_protection_exceeding_budget_is_infeasible_and_never_evicts_protected():
    protected = {0: ("sink",), 1: ("sink",), 2: ("focus",), 3: ("focus",), 4: ("question",)}
    plan = _plan(length=10, budget=3, candidates=list(range(10)), da=protected)
    assert plan.status == STATUS_INFEASIBLE
    assert set(protected) <= set(plan.retained)
    assert not (set(protected) & set(plan.evicted))
    assert plan.retained == (0, 1, 2, 3, 4)
    assert len(plan.retained) > plan.budget  # overflow is reported, not hidden
    assert plan.evicted == (5, 6, 7, 8, 9)


def test_infeasible_with_all_entries_protected_has_no_eligible_candidates():
    plan = _plan(length=3, budget=1, candidates=[0, 1, 2], da={0: ("s",), 1: ("s",), 2: ("s",)})
    assert plan.status == STATUS_INFEASIBLE
    assert plan.evicted == ()
    assert plan.retained == (0, 1, 2)


def test_random_consistent_plans_never_evict_protected_and_respect_budget():
    rng = random.Random(7)
    for _ in range(300):
        length = rng.randint(1, 40)
        budget = rng.randint(0, length)
        protected_idx = rng.sample(range(length), rng.randint(0, budget))
        # Engine rule: the H2O keep set has size budget - |protected| over ALL entries, so the
        # candidate set is its complement. Vetoes can only come from protected entries.
        keep = set(rng.sample(range(length), budget - len(protected_idx)))
        candidates = sorted(set(range(length)) - keep) if length > budget else []
        da = {i: ("p",) for i in protected_idx}
        plan = _plan(length=length, budget=budget, candidates=candidates, da=da)
        assert plan.status == STATUS_OK
        assert len(plan.retained) <= budget
        assert not (set(protected_idx) & set(plan.evicted))
        assert set(plan.evicted) <= set(candidates)
        assert set(plan.evicted) | set(plan.retained) == set(range(length))


def test_candidate_set_that_would_overflow_budget_is_rejected():
    # 10 entries, budget 5, only 2 candidates: retained would be 8 > 5 with no infeasibility.
    with pytest.raises(ValueError, match="budget"):
        _plan(length=10, budget=5, candidates=[0, 1])


def test_candidates_when_within_budget_are_rejected():
    with pytest.raises(ValueError, match="budget"):
        _plan(length=5, budget=10, candidates=[0])


def test_ablation_differs_only_by_removed_signal():
    kwargs = dict(length=10, budget=8, candidates=[1, 2, 3, 4, 5],
                  da={1: ("s",), 2: ("s",)}, grounding={4: ("g",)})
    both = _plan(**kwargs, enabled=("da", "grounding"))
    no_da = _plan(**kwargs, enabled=("grounding",))
    no_grounding = _plan(**kwargs, enabled=("da",))
    assert both.evicted == (3, 5)
    assert no_da.evicted == (1, 2, 3, 5)
    assert no_grounding.evicted == (3, 4, 5)
    # Removing DA releases exactly the DA-only protected candidates; removing grounding
    # releases exactly the grounding-only protected candidate.
    assert set(no_da.evicted) - set(both.evicted) == {1, 2}
    assert set(no_grounding.evicted) - set(both.evicted) == {4}
    # Ablation never changes the candidate set or the non-candidate decisions.
    for plan in (both, no_da, no_grounding):
        assert set(plan.evicted) <= {1, 2, 3, 4, 5}
        assert plan.retained == tuple(sorted(set(range(10)) - set(plan.evicted)))


def test_disabled_signal_labels_are_not_reported_as_fired():
    plan = _plan(length=6, budget=4, candidates=[0, 1], da={0: ("sink",)}, enabled=("grounding",))
    assert plan.decisions[0].signals_fired == ()
    assert plan.decisions[0].decision == DECISION_EVICT


def test_malformed_declarations_fail_closed():
    with pytest.raises(ValueError):
        _plan(length=5, budget=3, candidates=[7])  # candidate index out of range
    with pytest.raises(ValueError):
        _plan(length=-1, budget=3, candidates=[])
    with pytest.raises(ValueError):
        _plan(length=5, budget=-2, candidates=[])
    # budget >= length so these fail on the intended validation, not on the budget rule.
    with pytest.raises(ValueError, match="protected"):
        _plan(length=5, budget=5, candidates=[], da={9: ("s",)})  # protected index out of range
    with pytest.raises(ValueError, match="tuple"):
        _plan(length=5, budget=5, candidates=[], da={0: "not-a-tuple"})
    with pytest.raises(ValueError, match="source"):
        _plan(length=5, budget=5, candidates=[], enabled=("unknown_source",))


def test_summary_aggregates_auditable_counts():
    p1 = _plan(length=8, budget=5, candidates=[0, 1, 2, 3], da={0: ("sink",)}, layer=3)
    p2 = _plan(length=8, budget=5, candidates=[4, 5, 6, 7], grounding={5: ("g",)}, layer=7)
    s = summarize_plans([p1, p2])
    assert s["h2o_candidates_vetoed"] == 2
    assert s["conflicts_resolved_by_protection"] == 2
    assert s["protected_count_by_source"] == {"da": 1, "grounding": 1, "union": 2}
    assert s["status_counts"] == {STATUS_OK: 2}
    assert s["evicted_total"] == len(p1.evicted) + len(p2.evicted)
    assert s["layers"] == [3, 7]
