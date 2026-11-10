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

from .evidence_verification import (
    EvidencePolicy, MODE_NO_PROTECTION_DECLARED, MODE_PROTECTED_BY_POLICY, MODE_PROTECTED_VERIFIED,
    MODE_REJECTED_BYTES_MISMATCH, MODE_REJECTED_CALLER_FALSE, MODE_REJECTED_NOT_BOUND, MODE_UNCERTAIN,
    assess_evidence, decide_protection,
)

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

MEASUREMENT_LABEL = "cpu_pytorch_reference_not_vllm"

STATUS_OK = "OK"
STATUS_INFEASIBLE = "INFEASIBLE_PROTECTED_EXCEEDS_BUDGET"

DECISION_EVICT = "EVICT"
DECISION_RETAIN_PROTECTED_VETOED = "RETAIN_PROTECTED_VETOED_H2O_CANDIDATE"
DECISION_RETAIN_PROTECTED = "RETAIN_PROTECTED"
DECISION_RETAIN_INFEASIBLE = "RETAIN_INFEASIBLE_NO_EVICTION"
DECISION_RETAIN_H2O_KEPT = "RETAIN_H2O_KEPT"

SIGNAL_SOURCES: tuple[str, ...] = ("da", "grounding", "structural")

# Closed enum of per-entry reason codes (auditor oracle I-3). Every decision carries one.
REASON_EVICT = "H2O_CANDIDATE_UNPROTECTED"
REASON_VETOED_DA = "H2O_CANDIDATE_VETOED_BY_DA"
REASON_VETOED_GROUNDING = "H2O_CANDIDATE_VETOED_BY_GROUNDING"
REASON_VETOED_STRUCTURAL = "H2O_CANDIDATE_VETOED_BY_STRUCTURAL"
REASON_PROTECTED = "PROTECTED_NOT_H2O_CANDIDATE"
REASON_KEPT = "H2O_KEPT"
REASON_INFEASIBLE_NO_EVICTION = "INFEASIBLE_NO_EVICTION"
REASON_CODES: tuple[str, ...] = (
    REASON_EVICT, REASON_VETOED_DA, REASON_VETOED_GROUNDING, REASON_VETOED_STRUCTURAL,
    REASON_PROTECTED, REASON_KEPT,
)
# Unmet-budget reasons (lead decision Q3): explicit, never an exception.
INFEASIBLE_PROTECTED = "protected_exceeds_budget"
INFEASIBLE_UNMET_AFTER_ELIGIBLE = "budget_unmet_after_eligible_eviction"


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
    infeasible_reason: str = ""
    conflicts_by_source: dict[str, int] = field(default_factory=dict)
    protected_count_by_source: dict[str, int] = field(default_factory=dict)
    measurement_label: str = MEASUREMENT_LABEL

    @property
    def retained_count(self) -> int:
        return len(self.retained)

    @property
    def kv_budget_violated(self) -> bool:
        """True when the step is INFEASIBLE and the physical KV retains more than the budget."""
        return self.status == STATUS_INFEASIBLE and len(self.retained) > self.budget

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


def _veto_code(fired: Sequence[str]) -> str:
    """Reason code for a vetoed candidate: the highest-precedence source that fired (DA, grounding, structural)."""
    if any(sig.startswith("da:") for sig in fired):
        return REASON_VETOED_DA
    if any(sig.startswith("grounding:") for sig in fired):
        return REASON_VETOED_GROUNDING
    return REASON_VETOED_STRUCTURAL


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

    infeasible_reason = ""
    if len(protected) > budget:
        status = STATUS_INFEASIBLE
        infeasible_reason = INFEASIBLE_PROTECTED
    else:
        status = STATUS_OK
        if length <= budget and candidates:
            # Caller error: an eviction candidate was supplied although nothing needs to go.
            raise ValueError(
                f"candidates supplied although length {length} <= budget {budget}: "
                "no eviction is allowed within budget"
            )

    eligible = candidates - protected
    if status == STATUS_OK and length - len(eligible) > budget:
        # Budget unmet after evicting every eligible candidate (lead decision Q3): explicit status.
        status = STATUS_INFEASIBLE
        infeasible_reason = INFEASIBLE_UNMET_AFTER_ELIGIBLE
    if status == STATUS_INFEASIBLE:
        # ITER2 D4 S-3 (lead ruling): an INFEASIBLE step evicts NOTHING; every entry is retained.
        eligible = set()
    retained_set = set(range(length)) - eligible

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
            code = _veto_code(fired)
        elif is_prot:
            decision = DECISION_RETAIN_PROTECTED
            reason = f"protected by {', '.join(fired)}; not an h2o candidate"
            code = REASON_PROTECTED
        elif is_cand:
            # Unprotected H2O candidate that would have been evicted, but the step is INFEASIBLE.
            decision = DECISION_RETAIN_INFEASIBLE
            reason = "h2o candidate retained: step INFEASIBLE, no eviction this step"
            code = REASON_INFEASIBLE_NO_EVICTION
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
        infeasible_reason=infeasible_reason,
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
    vetoed = evicted = retained = protected_evicted = evicted_not_candidate = kv_violated = 0
    max_retained = 0
    for p in plans:
        vetoed += p.h2o_candidates_vetoed
        evicted += len(p.evicted)
        retained += len(p.retained)
        max_retained = max(max_retained, len(p.retained))
        statuses[p.status] += 1
        kv_violated += int(p.kv_budget_violated)
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
            "structural": protected.get("structural", 0),
            "union": protected.get("union", 0),
        },
        "status_counts": dict(statuses),
        "evicted_total": evicted,
        "retained_total": retained,
        "max_retained_per_layer": max_retained,
        "protected_evicted": protected_evicted,
        "evicted_not_candidate": evicted_not_candidate,
        "kv_budget_violated_layers": kv_violated,
    }


def combine_summaries(summaries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-step summaries (from summarize_plans) into one request-level record."""
    total: dict[str, Any] = {
        "measurement_label": MEASUREMENT_LABEL,
        "steps": 0,
        "h2o_candidates_vetoed": 0,
        "conflicts_resolved_by_protection": 0,
        "conflicts_by_source": {src: 0 for src in SIGNAL_SOURCES},
        "protected_count_by_source": {"da": 0, "grounding": 0, "structural": 0, "union": 0},
        "status_counts": {},
        "evicted_total": 0,
        "retained_total": 0,
        "max_retained_per_layer": 0,
        "protected_evicted": 0,
        "evicted_not_candidate": 0,
        "kv_budget_violated_layers": 0,
    }
    for s in summaries:
        total["steps"] += 1
        for key in ("h2o_candidates_vetoed", "conflicts_resolved_by_protection", "evicted_total",
                    "retained_total", "protected_evicted", "evicted_not_candidate",
                    "kv_budget_violated_layers"):
            total[key] += int(s[key])
        total["max_retained_per_layer"] = max(total["max_retained_per_layer"], int(s["max_retained_per_layer"]))
        for src in SIGNAL_SOURCES:
            total["conflicts_by_source"][src] += int(s["conflicts_by_source"].get(src, 0))
        for key in ("da", "grounding", "structural", "union"):
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


_MODE_TO_STATUS: dict[str, tuple[str, str]] = {
    MODE_REJECTED_CALLER_FALSE: ("unverified", "caller_declared_verified_false"),
    MODE_REJECTED_NOT_BOUND: ("expired", "pool_request_id_mismatch"),
    MODE_REJECTED_BYTES_MISMATCH: ("stale", "source_sha256_mismatch"),
    MODE_UNCERTAIN: ("uncertain", "source_bytes_not_hashed"),
    MODE_PROTECTED_VERIFIED: ("valid", "source_sha256_match"),
    MODE_NO_PROTECTION_DECLARED: ("declared_unverified", "no_source_bytes_checked"),
    MODE_PROTECTED_BY_POLICY: ("declared_unverified", "accepted_by_policy_accept_declared"),
}


def evidence_assessments(
    pool: Any,
    current_request_id: str,
    image_sha256: str | None,
    *,
    image_present: bool | None = None,
    policy: EvidencePolicy | None = None,
) -> dict[str, dict[str, Any]]:
    """ITER3 R3 S1: per evidence id, the tri-state dimensions, the method and provenance, and the protection decision.

    `image_present` None means: present iff an image hash was supplied (the old call convention).
    The status vocabulary: valid (source bytes VERIFIED and bound), declared_unverified (text-only, no source
    bytes checked: protects only under accept_declared), uncertain, stale, expired, unverified (caller false).
    """
    if image_present is None:
        image_present = image_sha256 is not None
    policy = policy or EvidencePolicy()
    out: dict[str, dict[str, Any]] = {}
    if pool is None:
        return out
    for item in pool.items():
        a = assess_evidence(item, request_id=current_request_id, pool_request_id=pool.request_id,
                            image_sha256=image_sha256, image_present=bool(image_present))
        d = decide_protection(a, policy, caller_verified=bool(item.verified))
        status, status_reason = _MODE_TO_STATUS[d.mode]
        out[item.evidence_id] = {
            "dimensions": dict(a.dimensions),
            "method": a.method,
            "provenance": dict(a.provenance),
            "protection_mode": d.mode,
            "protects": bool(d.protects),
            "decision_reason": d.reason,
            "status": status,
            "status_reason": status_reason,
        }
    return out


def evidence_validity(
    pool: Any,
    current_request_id: str,
    image_sha256: str | None,
    *,
    image_present: bool | None = None,
    policy: EvidencePolicy | None = None,
) -> dict[str, tuple[str, str]]:
    """Per evidence id: (status, reason). Derived from evidence_assessments (single decision path)."""
    return {
        eid: (a["status"], a["status_reason"])
        for eid, a in evidence_assessments(pool, current_request_id, image_sha256,
                                           image_present=image_present, policy=policy).items()
    }


# ITER3 R2 C-3: provenance labels. The `verified` flag is never checked by this controller, so it is always
# reported as a caller declaration. `source_sha256` is labelled as checked only when it was compared with
# the request's image bytes; text-only requests have nothing to compare against.
VERIFIED_FLAG_LABEL = "CALLER_DECLARED_NOT_VERIFIED"
SOURCE_UNCHECKED_LABEL = "CALLER_DECLARED_NOT_VERIFIED"
SOURCE_CHECKED_MATCH_LABEL = "CHECKED_IMAGE_BYTES_MATCH"
SOURCE_CHECKED_MISMATCH_LABEL = "CHECKED_IMAGE_BYTES_MISMATCH"


def evidence_provenance(pool: Any, image_sha256: str | None) -> dict[str, dict[str, str]]:
    """Per evidence id: the provenance label of its `verified` flag and of its `source_sha256`.

    `verified_flag` is always CALLER_DECLARED_NOT_VERIFIED. `source_sha256` is CHECKED_IMAGE_BYTES_MATCH or
    CHECKED_IMAGE_BYTES_MISMATCH only when an image hash exists to compare, else CALLER_DECLARED_NOT_VERIFIED.
    """
    out: dict[str, dict[str, str]] = {}
    if pool is None:
        return out
    for item in pool.items():
        if image_sha256 is None:
            source = SOURCE_UNCHECKED_LABEL
        elif item.source_sha256.lower() == image_sha256.lower():
            source = SOURCE_CHECKED_MATCH_LABEL
        else:
            source = SOURCE_CHECKED_MISMATCH_LABEL
        out[item.evidence_id] = {"verified_flag": VERIFIED_FLAG_LABEL, "source_sha256": source}
    return out


VISUAL_EVIDENCE_NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
VISUAL_EVIDENCE_NOTE = (
    "Grounding protects token spans of evidence KEYWORDS in the text prompt only. image_region evidence "
    "protects NO visual tokens (no bounding-box-to-vision-token mapping exists). Visual evidence coverage is "
    "not claimed."
)


def visual_evidence_protection_status(pool: Any) -> str:
    """ITER3 R2 C-5: explicit status of protection for image evidence. Never a silent pass.

    NOT_IMPLEMENTED when the pool declares any image_region evidence (the controller protects none of its
    visual tokens); NO_IMAGE_EVIDENCE_DECLARED when the pool has only text evidence; NO_EVIDENCE_DECLARED
    without a pool.
    """
    if pool is None:
        return "NO_EVIDENCE_DECLARED"
    if any(item.modality == "image_region" for item in pool.items()):
        return VISUAL_EVIDENCE_NOT_IMPLEMENTED
    return "NO_IMAGE_EVIDENCE_DECLARED"


def grounding_protection_from_pool(
    pool: Any,
    prompt_ids: Sequence[int],
    encode: Any,
    current_request_id: str,
    image_sha256: str | None,
    policy: EvidencePolicy | None = None,
    *,
    image_present: bool | None = None,
) -> dict[int, tuple[str, ...]]:
    """Grounding-protected prompt positions: token spans of evidence keywords whose items PROTECT.

    Uses the same keyword index ECRD matches on. Each keyword is tokenized in the bare,
    space-prefixed and capitalised forms, and every contiguous occurrence in the prompt is
    protected, labelled with the evidence ids it supports. Only `valid` evidence protects.
    Benchmark answers are never read: the input is the pool's declared keywords and the prompt.
    Keyword matching is token-level, so a word split across tokens in a different way can be
    missed (documented limitation).
    """
    protected, _report = grounding_protection_with_report(
        pool, prompt_ids, encode, current_request_id, image_sha256, policy, image_present=image_present)
    return protected


# ITER3 R2 C-4: keyword breadth policy. Reported explicitly in telemetry (grounding_keyword_policy).
# Stop-word-like keywords are rejected whatever their frequency. A keyword whose protected positions exceed
# MAX_KEYWORD_PROMPT_FRACTION of the prompt is rejected. Accepted protection that still exceeds the budget is
# handled by the existing INFEASIBLE semantics in plan_starlight_layer (never silently dropped).
KEYWORD_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "its", "of", "on",
    "or", "that", "the", "this", "to", "was", "were", "with",
})
MAX_KEYWORD_PROMPT_FRACTION = 0.2
# ITER3 R3 (N-R2-2, lead ruling): cap on the protected UNION of accepted keywords, as a fraction of prompt
# positions. Owner-ratifiable named constant. Keywords are processed in DECLARED order; a keyword whose addition
# would push the union over the cap is rejected and listed in rejected_union (later keywords still considered).
MAX_UNION_PROMPT_FRACTION = 0.5


def grounding_protection_with_report(
    pool: Any,
    prompt_ids: Sequence[int],
    encode: Any,
    current_request_id: str,
    image_sha256: str | None,
    policy: EvidencePolicy | None = None,
    *,
    image_present: bool | None = None,
) -> tuple[dict[int, tuple[str, ...]], dict[str, Any]]:
    """Grounding-protected positions AND the keyword policy report (accepted and rejected keywords).

    Same matching as grounding_protection_from_pool. Only evidence items whose PROTECTION DECISION is positive
    (evidence_verification.decide_protection) can protect. Each keyword is checked against the breadth policy
    before any of its positions is protected, so a rejected keyword protects nothing.
    """
    policy = policy or EvidencePolicy()
    report: dict[str, Any] = {
        "max_keyword_prompt_fraction": MAX_KEYWORD_PROMPT_FRACTION,
        "max_union_prompt_fraction": MAX_UNION_PROMPT_FRACTION,
        "stopword_list": sorted(KEYWORD_STOPWORDS),
        "accepted_keywords": [],
        "rejected_stopword": [],
        "rejected_breadth": [],
        "rejected_union": [],
        "union_fraction": 0.0,
        "evidence_policy": policy.as_dict(),
        "assessments": {},
        "omitted_items": [],
    }
    if pool is None:
        return {}, report
    assessments = evidence_assessments(pool, current_request_id, image_sha256,
                                       image_present=image_present, policy=policy)
    report["assessments"] = assessments
    report["omitted_items"] = [
        {"evidence_id": eid, "reason": a["protection_mode"]}
        for eid, a in assessments.items() if not a["protects"]
    ]
    valid_ids = {eid for eid, a in assessments.items() if a["protects"]}
    n_prompt = len(prompt_ids)
    protected: dict[int, set[str]] = {}
    # Declared order (the pool's insertion order), not alphabetical: the union cap is order-dependent by design.
    for keyword, evidence_ids in pool.keyword_index().items():
        supporting = [eid for eid in evidence_ids if eid in valid_ids]
        if not supporting:
            continue
        if keyword.strip().lower() in KEYWORD_STOPWORDS:
            report["rejected_stopword"].append(keyword)
            continue
        own: dict[int, set[str]] = {}
        variants = {keyword, f" {keyword}", keyword.capitalize(), f" {keyword.capitalize()}"}
        for variant in sorted(variants):
            tokens = list(encode(variant))
            for start in match_token_spans(prompt_ids, tokens):
                for pos in range(start, start + len(tokens)):
                    own.setdefault(pos, set()).update(supporting)
        fraction = (len(own) / n_prompt) if n_prompt else 0.0
        if fraction > MAX_KEYWORD_PROMPT_FRACTION:
            report["rejected_breadth"].append({"keyword": keyword, "protected_fraction": round(fraction, 6)})
            continue
        new_positions = len(set(own) - set(protected))
        union_if_added = ((len(protected) + new_positions) / n_prompt) if n_prompt else 0.0
        if union_if_added > MAX_UNION_PROMPT_FRACTION:
            report["rejected_union"].append({"keyword": keyword, "union_fraction_if_added": union_if_added})
            continue
        report["accepted_keywords"].append(keyword)
        for pos, ids in own.items():
            protected.setdefault(pos, set()).update(ids)
    report["union_fraction"] = (len(protected) / n_prompt) if n_prompt else 0.0
    return {pos: tuple(sorted(ids)) for pos, ids in sorted(protected.items())}, report
