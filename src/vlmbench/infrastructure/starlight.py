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
Then retained = keep union protected has size <= budget by construction. Vetoes are the H2O
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

# Closed enum of per-entry reason codes (auditor oracle I-3). Every decision carries one.
REASON_EVICT = "H2O_CANDIDATE_UNPROTECTED"
REASON_VETOED = "H2O_CANDIDATE_VETOED_BY_PROTECTION"
REASON_PROTECTED = "PROTECTED_NOT_H2O_CANDIDATE"
REASON_KEPT = "H2O_KEPT"
REASON_CODES: tuple[str, ...] = (REASON_EVICT, REASON_VETOED, REASON_PROTECTED, REASON_KEPT)


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
    reason_code: str


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
            decision, reason, code = DECISION_EVICT, "h2o candidate and not protected", REASON_EVICT
        elif is_cand and is_prot:
            decision = DECISION_RETAIN_PROTECTED_VETOED
            reason = f"h2o candidate vetoed: protected by {', '.join(fired)}"
            code = REASON_VETOED
        elif is_prot:
            decision = DECISION_RETAIN_PROTECTED
            reason = f"protected by {', '.join(fired)}; not an h2o candidate"
            code = REASON_PROTECTED
        else:
            decision = DECISION_RETAIN_H2O_KEPT
            reason = "not an h2o candidate; retained"
            code = REASON_KEPT
        if status == STATUS_INFEASIBLE:
            reason = f"INFEASIBLE (protected {len(protected)} > budget {budget}); {reason}"
        decisions.append(EntryDecision(
            entry_id=i, layer=layer, h2o_candidate=is_cand, internally_protected=is_prot,
            signals_fired=fired, decision=decision, reason=reason, reason_code=code,
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
    """Aggregate per-layer plans into the telemetry fields reported by the engine.

    `protected_evicted` and `evicted_not_candidate` are invariant counters. Both must be 0:
    they are recomputed from the decisions, not assumed from the policy.
    """
    conflicts: Counter[str] = Counter()
    protected: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    vetoed = evicted = retained = protected_evicted = evicted_not_candidate = 0
    max_retained = 0
    for p in plans:
        vetoed += p.h2o_candidates_vetoed
        evicted += len(p.evicted)
        retained += len(p.retained)
        max_retained = max(max_retained, len(p.retained))
        statuses[p.status] += 1
        conflicts.update(p.conflicts_by_source)
        protected.update(p.protected_count_by_source)
        for d in p.decisions:
            if d.decision == DECISION_EVICT and d.internally_protected:
                protected_evicted += 1
            if d.decision == DECISION_EVICT and not d.h2o_candidate:
                evicted_not_candidate += 1
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
        "max_retained_per_layer": max_retained,
        "protected_evicted": protected_evicted,
        "evicted_not_candidate": evicted_not_candidate,
    }


def combine_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-step summaries (from summarize_plans) into one request-level record."""
    total: dict[str, Any] = {
        "measurement_label": MEASUREMENT_LABEL,
        "steps": 0,
        "h2o_candidates_vetoed": 0,
        "conflicts_resolved_by_protection": 0,
        "conflicts_by_source": {src: 0 for src in SIGNAL_SOURCES},
        "protected_count_by_source": {"da": 0, "grounding": 0, "union": 0},
        "status_counts": {},
        "evicted_total": 0,
        "retained_total": 0,
        "max_retained_per_layer": 0,
        "protected_evicted": 0,
        "evicted_not_candidate": 0,
    }
    for s in summaries:
        total["steps"] += 1
        for key in ("h2o_candidates_vetoed", "conflicts_resolved_by_protection", "evicted_total",
                    "retained_total", "protected_evicted", "evicted_not_candidate"):
            total[key] += int(s[key])
        total["max_retained_per_layer"] = max(total["max_retained_per_layer"], int(s["max_retained_per_layer"]))
        for src in SIGNAL_SOURCES:
            total["conflicts_by_source"][src] += int(s["conflicts_by_source"].get(src, 0))
        for key in ("da", "grounding", "union"):
            total["protected_count_by_source"][key] += int(s["protected_count_by_source"].get(key, 0))
        for status, n in s["status_counts"].items():
            total["status_counts"][status] = total["status_counts"].get(status, 0) + int(n)
    return total


# --- Signal sources ---------------------------------------------------------------------------
# Pure helpers that turn engine-side declarations into the {idx: labels} maps the policy reads.
# No torch: each takes plain sequences, so a different backend can reuse them unchanged.


def da_protection_from_spans(
    spans: Iterable[tuple[str, int, int]], original_positions: Sequence[int],
) -> dict[int, tuple[str, ...]]:
    """DA scope spans (label, start, end) on ORIGINAL token positions -> physical-slot labels.

    `original_positions[i]` is the original absolute position of physical slot i. Spans are
    matched on that original position, so protection follows the token through KV compaction.
    Spans come from the scripted DA diagnostic (sink/GLOBAL, FOCUS, readable question); the LOCAL
    window is a mask-only scope and is not passed here.
    """
    normalized: list[tuple[str, int, int]] = []
    for label, start, end in spans:
        if not isinstance(label, str) or not label:
            raise ValueError("DA span label must be a non-empty string")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ValueError(f"malformed DA span ({label!r}, {start}, {end})")
        if end > start:
            normalized.append((label, start, end))
    protected: dict[int, tuple[str, ...]] = {}
    for slot, pos in enumerate(original_positions):
        labels = tuple(label for label, start, end in normalized if start <= pos < end)
        if labels:
            protected[slot] = labels
    return protected


def compact_orig_positions(
    original_positions: Sequence[int],
    retained: Sequence[int] | None,
    appended: int | None,
) -> list[int]:
    """Track original positions through one step: append the new token, or compact by retained.

    `retained` must be sorted, unique and in range (the same contract as the H2O tracker).
    """
    if retained is None:
        if appended is None:
            raise ValueError("either retained or appended must be given")
        return [int(p) for p in original_positions] + [int(appended)]
    idx = [int(i) for i in retained]
    if any(i < 0 or i >= len(original_positions) for i in idx):
        raise ValueError("retained index out of range")
    if any(b <= a for a, b in zip(idx, idx[1:])):
        raise ValueError("retained indices must be sorted and unique")
    return [int(original_positions[i]) for i in idx]


def match_token_spans(prompt_ids: Sequence[int], needle_ids: Sequence[int]) -> list[int]:
    """Start indices of every contiguous occurrence of needle_ids in prompt_ids."""
    needle = [int(t) for t in needle_ids]
    n = len(needle)
    if n == 0:
        return []
    prompt = [int(t) for t in prompt_ids]
    return [i for i in range(len(prompt) - n + 1) if prompt[i:i + n] == needle]


def evidence_validity(pool: Any, current_request_id: str, image_sha256: str | None) -> dict[str, tuple[str, str]]:
    """Per evidence id: (status, reason) with status in {valid, expired, stale, unverified}.

    - expired: the pool was built for a different request id (it outlived its request).
    - stale: the image source hash differs from the source the evidence was declared against.
    - text-only requests have no image hash to check; they are valid with an explicit
      `source_unchecked_text_only` reason, so the gap stays visible in telemetry.
    """
    verdicts: dict[str, tuple[str, str]] = {}
    if pool is None:
        return verdicts
    for item in pool.items():
        if not item.verified:
            verdicts[item.evidence_id] = ("unverified", "verified_false")
        elif pool.request_id != current_request_id:
            verdicts[item.evidence_id] = ("expired", "pool_request_id_mismatch")
        elif image_sha256 is None:
            verdicts[item.evidence_id] = ("valid", "source_unchecked_text_only")
        elif item.source_sha256.lower() != image_sha256.lower():
            verdicts[item.evidence_id] = ("stale", "source_sha256_mismatch")
        else:
            verdicts[item.evidence_id] = ("valid", "source_sha256_match")
    return verdicts


def grounding_protection_from_pool(
    pool: Any,
    prompt_ids: Sequence[int],
    encode: Any,
    current_request_id: str,
    image_sha256: str | None,
) -> dict[int, tuple[str, ...]]:
    """Grounding-protected prompt positions: token spans of valid evidence keywords.

    Uses the same keyword index ECRD matches on. Each keyword is tokenized in the bare,
    space-prefixed and capitalised forms, and every contiguous occurrence in the prompt is
    protected, labelled with the evidence ids it supports. Only `valid` evidence protects.
    Benchmark answers are never read: the input is the pool's declared keywords and the prompt.
    Keyword matching is token-level, so a word split across tokens in a different way can be
    missed (documented limitation).
    """
    if pool is None:
        return {}
    verdicts = evidence_validity(pool, current_request_id, image_sha256)
    valid_ids = {eid for eid, (status, _) in verdicts.items() if status == "valid"}
    protected: dict[int, set[str]] = {}
    for keyword, evidence_ids in pool.keyword_index().items():
        supporting = [eid for eid in evidence_ids if eid in valid_ids]
        if not supporting:
            continue
        variants = {keyword, f" {keyword}", keyword.capitalize(), f" {keyword.capitalize()}"}
        for variant in sorted(variants):
            tokens = list(encode(variant))
            for start in match_token_spans(prompt_ids, tokens):
                for pos in range(start, start + len(tokens)):
                    protected.setdefault(pos, set()).update(supporting)
    return {pos: tuple(sorted(ids)) for pos, ids in sorted(protected.items())}
