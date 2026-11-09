"""STARLIGHT Stage 1 controller policy: pure, backend-agnostic, no torch or numpy.

Owner contract (ITER2 brief):

    eligible_to_evict = h2o_candidate AND NOT internally_protected
    internally_protected = DA-protected  OR  grounding-protected
    external eviction   = H2O-proposed candidates only

A protected entry is never evicted. A feasible plan retains at most `budget` entries. When the
protected count alone exceeds the budget, the plan is reported as
INFEASIBLE_PROTECTED_EXCEEDS_BUDGET: protected entries are still retained and the overflow is
reported, never hidden. A candidate set that would overflow the budget is rejected with
ValueError, so an inconsistent caller cannot silently break the budget.

Budget accounting: the engine computes the H2O candidate set at budget - |protected| over all
entries (H2O's own protection is disabled under STARLIGHT, so the controller owns protection).
Then retained = keep ∪ protected has size <= budget by construction. Vetoes are the H2O
candidates that the controller protects instead of evicting.

Every result is labelled cpu_pytorch_reference_not_vllm. Nothing here is GPU or vLLM evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

MEASUREMENT_LABEL = "cpu_pytorch_reference_not_vllm"

STATUS_OK = "OK"
STATUS_INFEASIBLE = "INFEASIBLE_PROTECTED_EXCEEDS_BUDGET"

DECISION_EVICT = "EVICT"
DECISION_RETAIN_PROTECTED_VETOED = "RETAIN_PROTECTED_VETOED_H2O_CANDIDATE"
DECISION_RETAIN_PROTECTED = "RETAIN_PROTECTED"
DECISION_RETAIN_H2O_KEPT = "RETAIN_H2O_KEPT"

SIGNAL_SOURCES: tuple[str, ...] = ("da", "grounding")


@dataclass(frozen=True)
class EntryDecision:
    """Auditable per-entry record: which signals fired and why the entry was kept or evicted."""

    entry_id: int
    layer: int
    h2o_candidate: bool
    internally_protected: bool
    signals_fired: tuple[str, ...]
    decision: str
    reason: str


@dataclass(frozen=True)
class StarlightLayerPlan:
    """Retained/evicted partition of one full-attention layer plus its audit trail."""

    layer: int
    length: int
    budget: int
    status: str
    retained: tuple[int, ...]
    evicted: tuple[int, ...]
    decisions: tuple[EntryDecision, ...]
    h2o_candidates_vetoed: int
    conflicts_by_source: dict[str, int] = field(default_factory=dict)
    protected_count_by_source: dict[str, int] = field(default_factory=dict)
    measurement_label: str = MEASUREMENT_LABEL

    @property
    def retained_count(self) -> int:
        return len(self.retained)

    @property
    def conflicts_resolved_by_protection(self) -> int:
        """Conflict events: one per (vetoed entry, protecting source). An entry protected by
        both DA and grounding counts once in h2o_candidates_vetoed and twice here."""
        return sum(self.conflicts_by_source.values())


def _check_int(name: str, value: Any, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _validate_labels(source: str, index: int, labels: Any) -> tuple[str, ...]:
    if isinstance(labels, (str, bytes)) or not isinstance(labels, (tuple, list)):
        raise ValueError(
            f"labels for protected index {index} from source '{source}' must be a tuple of str"
        )
    if not labels or not all(isinstance(x, str) and x for x in labels):
        raise ValueError(
            f"labels for protected index {index} from source '{source}' must be non-empty strings"
        )
    return tuple(labels)


def plan_starlight_layer(
    *,
    layer: int,
    length: int,
    budget: int,
    h2o_candidates: Iterable[int],
    signals: Mapping[str, Mapping[int, Sequence[str]]],
    enabled_signals: Iterable[str] = SIGNAL_SOURCES,
) -> StarlightLayerPlan:
    """Plan retained and evicted physical entries for one full-attention layer.

    Args:
        layer: layer index, recorded in every decision for audit.
        length: number of physical entries in the layer's KV cache.
        budget: total retained-entry budget (protected entries count toward it).
        h2o_candidates: physical indices H2O proposes to evict.
        signals: {"da": {idx: labels}, "grounding": {idx: labels}}. Labels name the fired
            signal, e.g. ("sink_global",) or ("ev_3",). Every declaration is validated.
        enabled_signals: ablation switch. A disabled source is validated but never protects.

    Raises:
        ValueError on malformed declarations, unknown sources, out-of-range indices, or a
        candidate set that would leave more than `budget` entries retained when the plan is
        feasible.
    """
    _check_int("layer", layer)
    _check_int("length", length)
    _check_int("budget", budget)

    enabled = tuple(enabled_signals)
    for src in enabled:
        if src not in SIGNAL_SOURCES:
            raise ValueError(f"unknown signal source '{src}'; expected one of {SIGNAL_SOURCES}")
    for src in signals:
        if src not in SIGNAL_SOURCES:
            raise ValueError(f"unknown signal source '{src}'; expected one of {SIGNAL_SOURCES}")

    candidates: set[int] = set()
    for c in h2o_candidates:
        ci = _check_int("h2o candidate index", c)
        if ci >= length:
            raise ValueError(f"h2o candidate index {ci} out of range for length {length}")
        candidates.add(ci)

    labels_by_source: dict[str, dict[int, tuple[str, ...]]] = {src: {} for src in SIGNAL_SOURCES}
    for src in SIGNAL_SOURCES:
        for idx, labels in dict(signals.get(src, {})).items():
            pi = _check_int("protected index", idx)
            if pi >= length:
                raise ValueError(
                    f"protected index {pi} from source '{src}' out of range for length {length}"
                )
            labels_by_source[src][pi] = _validate_labels(src, pi, labels)

    enabled_set = set(enabled)
    fired_by_index: dict[int, tuple[str, ...]] = {}
    for src in SIGNAL_SOURCES:
        if src not in enabled_set:
            continue
        for idx, labels in labels_by_source[src].items():
            fired_by_index[idx] = fired_by_index.get(idx, ()) + tuple(
                f"{src}:{label}" for label in labels
            )
    protected = set(fired_by_index)

    if len(protected) > budget:
        status = STATUS_INFEASIBLE
    else:
        status = STATUS_OK
        if length <= budget and candidates:
            raise ValueError(
                f"candidates supplied although length {length} <= budget {budget}: "
                "no eviction is allowed within budget"
            )

    eligible = candidates - protected
    retained_set = set(range(length)) - eligible
    if status == STATUS_OK and len(retained_set) > budget:
        raise ValueError(
            f"candidate set would retain {len(retained_set)} entries > budget {budget}: "
            "candidate set is inconsistent with the budget"
        )

    vetoed = candidates & protected
    conflicts = {src: 0 for src in SIGNAL_SOURCES}
    for idx in vetoed:
        for src in SIGNAL_SOURCES:
            if src in enabled_set and idx in labels_by_source[src]:
                conflicts[src] += 1

    decisions: list[EntryDecision] = []
    for i in range(length):
        is_cand = i in candidates
        is_prot = i in protected
        fired = fired_by_index.get(i, ())
        if i in eligible:
            decision, reason = DECISION_EVICT, "h2o candidate and not protected"
        elif is_cand and is_prot:
            decision = DECISION_RETAIN_PROTECTED_VETOED
            reason = f"h2o candidate vetoed: protected by {', '.join(fired)}"
        elif is_prot:
            decision = DECISION_RETAIN_PROTECTED
            reason = f"protected by {', '.join(fired)}; not an h2o candidate"
        else:
            decision = DECISION_RETAIN_H2O_KEPT
            reason = "not an h2o candidate; retained"
        if status == STATUS_INFEASIBLE:
            reason = f"INFEASIBLE (protected {len(protected)} > budget {budget}); {reason}"
        decisions.append(EntryDecision(
            entry_id=i, layer=layer, h2o_candidate=is_cand, internally_protected=is_prot,
            signals_fired=fired, decision=decision, reason=reason,
        ))

    protected_counts = {
        src: sum(1 for i in labels_by_source[src] if src in enabled_set)
        for src in SIGNAL_SOURCES
    }
    protected_counts["union"] = len(protected)

    return StarlightLayerPlan(
        layer=layer,
        length=length,
        budget=budget,
        status=status,
        retained=tuple(sorted(retained_set)),
        evicted=tuple(sorted(eligible)),
        decisions=tuple(decisions),
        h2o_candidates_vetoed=len(vetoed),
        conflicts_by_source=conflicts,
        protected_count_by_source=protected_counts,
    )


def summarize_plans(plans: Sequence[StarlightLayerPlan]) -> dict[str, Any]:
    """Aggregate per-layer plans into the telemetry fields reported by the engine."""
    conflicts: Counter[str] = Counter()
    protected: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    vetoed = 0
    evicted = 0
    retained = 0
    for p in plans:
        vetoed += p.h2o_candidates_vetoed
        evicted += len(p.evicted)
        retained += len(p.retained)
        statuses[p.status] += 1
        conflicts.update(p.conflicts_by_source)
        protected.update(p.protected_count_by_source)
    return {
        "measurement_label": MEASUREMENT_LABEL,
        "layers": [p.layer for p in plans],
        "h2o_candidates_vetoed": vetoed,
        "conflicts_resolved_by_protection": sum(conflicts.values()),
        "conflicts_by_source": {src: conflicts.get(src, 0) for src in SIGNAL_SOURCES},
        "protected_count_by_source": {
            "da": protected.get("da", 0),
            "grounding": protected.get("grounding", 0),
            "union": protected.get("union", 0),
        },
        "status_counts": dict(statuses),
        "evicted_total": evicted,
        "retained_total": retained,
    }
