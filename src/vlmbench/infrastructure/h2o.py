"""Heavy-Hitter Oracle (H2O) KV Cache management and eviction policy.

Pinned upstream: https://github.com/FMInference/H2O @ ac75c2a8a9e76832b2a4139b9363373b56336bfb

Implements:
1. Per-layer, per-KV-head cumulative attention score accumulation.
2. Dynamic budget split: Heavy Hitters (K_hh) + Recent Local Window (W_r).
3. Protection of special tokens and visual token blocks.
4. Immunity for linear recurrent / DeltaNet layers (only full-attention layers are evicted).
5. MTP transactional state machine (begin -> propose -> verify -> commit / rollback). Used ONLY by
   the synthetic policy harness (adapters.py). NOT used by the real engine, which keeps no H2O MTP transaction.
6. Exact memory attribution: distinguishing virtual mask savings from physical block reclamation.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np


@dataclass(frozen=True)
class H2OConfig:
    """Configuration for Heavy Hitter Oracle KV cache pruning."""
    budget_ratio: float = 0.5          # Total allowed KV cache fraction (e.g. 0.5 = 50% retained)
    recent_ratio: float = 0.2          # Fraction of budget allocated to recent local window
    min_budget: int = 32               # Minimum retained tokens before eviction triggers
    protect_visual_tokens: bool = True # Protect genuine multimodal image tokens from eviction
    protect_system_tokens: int = 4     # Number of initial anchor/system tokens permanently retained
    tie_breaking: str = "recent"       # Deterministic tie-breaking: 'recent' prefers later tokens
    # ITER2 Stage 2(c), upstream fidelity A4-02. Opt-in. False (default) keeps the layer-mean
    # selection. True selects heavy hitters PER KV HEAD via compute_eviction_plan_per_head; the
    # shared compute_eviction_plan refuses it. Not wired into the engine (needs per-head KV gather).
    per_kv_head_selection: bool = False


@dataclass
class H2OLayerState:
    """Per-layer attention score and KV occupancy tracker."""
    layer_idx: int
    is_recurrent: bool = False  # If True (e.g. DeltaNet/Linear Attention), eviction is bypassed
    num_kv_heads: int = 1
    scores: np.ndarray = field(default_factory=lambda: np.zeros((1, 0), dtype=np.float32))
    # scores shape: [num_kv_heads, seq_len]
    evicted_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    # evicted_mask[i] == True if token i is physically evicted from KV cache
    visual_token_mask: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=bool))
    # Per-layer physical-token coordinates; MUST NOT be shared between layers.
    budget: int | None = None  # Fixed total KV budget, set once from prefill length

    def __post_init__(self) -> None:
        if self.scores.shape[0] != self.num_kv_heads:
            self.scores = np.zeros((self.num_kv_heads, 0), dtype=np.float32)


class H2OCacheTracker:
    """Tracks cumulative attention scores across layers and generates eviction masks."""

    def __init__(self, config: H2OConfig | None = None, num_layers: int = 24, num_kv_heads: int = 2):
        self.config = config or H2OConfig()
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.layers: list[H2OLayerState] = [
            H2OLayerState(layer_idx=i, num_kv_heads=num_kv_heads)
            for i in range(num_layers)
        ]
        self.seq_len = 0
        self.visual_token_mask = np.zeros((0,), dtype=bool)

    def register_recurrent_layer(self, layer_idx: int) -> None:
        """Mark a layer as recurrent (e.g. DeltaNet in hybrid Qwen3.5), exempting it from KV eviction."""
        if 0 <= layer_idx < self.num_layers:
            self.layers[layer_idx].is_recurrent = True

    def set_visual_tokens(self, mask: Sequence[bool]) -> None:
        """Register boolean mask of visual token positions to preserve evidence."""
        self.visual_token_mask = np.array(mask, dtype=bool)
        for layer in self.layers:
            layer.visual_token_mask = self.visual_token_mask.copy()

    def update_attention_scores(self, layer_idx: int, attention_weights: np.ndarray) -> None:
        """Accumulate attention scores for an attention layer.

        Args:
            layer_idx: 0-based index of transformer layer.
            attention_weights: [num_kv_heads, query_len, key_len] or [query_len, key_len].
        """
        layer = self.layers[layer_idx]
        if layer.is_recurrent:
            return  # Recurrent states have fixed size; no KV eviction

        if attention_weights.ndim == 2:
            weights = attention_weights[np.newaxis, ...]
        else:
            weights = attention_weights

        num_heads, q_len, k_len = weights.shape
        if k_len > layer.scores.shape[1]:
            # Expand score array to match current key length
            pad_len = k_len - layer.scores.shape[1]
            layer.scores = np.pad(layer.scores, ((0, 0), (0, pad_len)), mode="constant")
            layer.evicted_mask = np.pad(layer.evicted_mask, (0, pad_len), mode="constant", constant_values=False)
            self.seq_len = max(self.seq_len, k_len)

        # Sum attention received by each key token across queries
        query_sum = weights.sum(axis=1)  # [num_heads, k_len]
        if query_sum.shape[0] != self.num_kv_heads:
            # Average query heads belonging to each KV head group (GQA)
            repeat = query_sum.shape[0] // self.num_kv_heads
            query_sum = query_sum.reshape(self.num_kv_heads, repeat, k_len).mean(axis=1)

        layer.scores[:, :k_len] += query_sum

    def _protected_mask(self, layer: H2OLayerState, length: int) -> np.ndarray:
        """Positions that are never evicted: initial sinks and (optionally) visual tokens."""
        protected = np.zeros(length, dtype=bool)
        protected[:min(self.config.protect_system_tokens, length)] = True
        if self.config.protect_visual_tokens and len(layer.visual_token_mask) > 0:
            active = min(length, len(layer.visual_token_mask))
            protected[:active] |= layer.visual_token_mask[:active]
        return protected

    def _budget_for_unprotected(self, unprotected_length: int) -> int:
        return max(self.config.min_budget, int(unprotected_length * self.config.budget_ratio))

    def initialize_budget(self, layer_idx: int, prefill_length: int) -> int:
        """Fix this layer's budget for UNPROTECTED tokens from the PREFILL length, once.

        R4-1 semantics: protected tokens (sinks and visual tokens) sit OUTSIDE the budget.
        budget = max(min_budget, int((prefill_length - protected_at_prefill) * budget_ratio)).
        Total retained = protected + budget (text-side), and it never exceeds that.
        The budget is never re-derived from the compacted cache length (that decayed geometrically).
        """
        layer = self.layers[layer_idx]
        if prefill_length < 0:
            raise ValueError("prefill_length must be non-negative")
        protected = int(np.sum(self._protected_mask(layer, prefill_length)))
        layer.budget = self._budget_for_unprotected(prefill_length - protected)
        return layer.budget

    def compute_eviction_plan(self, layer_idx: int, current_length: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Plan retained positions for a layer with a FIXED budget over UNPROTECTED tokens.

        Semantics (lead round 4, R4-1):
        - Protected tokens (sinks and visual tokens) are never evicted and do not count toward the budget.
        - Budget B = max(min_budget, int(unprotected_prefill_length * budget_ratio)), fixed per layer.
        - When unprotected tokens <= B, nothing is evicted.
        - Otherwise exactly B unprotected tokens are kept: the newest `recent` unprotected tokens
          (recent = min(B, max(1, int(B * recent_ratio)))) plus the highest-scoring remaining
          unprotected tokens. Total retained = protected + B, never more.

        Returns:
            (retained_mask, metrics_dict) where retained_mask is bool array of length current_length.
        """
        if self.config.per_kv_head_selection:
            raise NotImplementedError(
                "per_kv_head_selection is opt-in: use compute_eviction_plan_per_head; the shared "
                "single-mask plan would silently use the layer-mean selection"
            )
        layer = self.layers[layer_idx]
        if layer.is_recurrent:
            return np.ones(current_length, dtype=bool), {
                "total_tokens": current_length,
                "retained_tokens": current_length,
                "evicted_tokens": 0,
                "heavy_hitters": 0,
                "recent_tokens": current_length,
                "protected_tokens": 0,
                "unprotected_tokens": 0,
                "unprotected_retained": 0,
                "budget": None,
                "per_step_evicted_slot_rate": 0.0,
                "kv_bytes_freed": "NOT_MEASURED_LOGICAL_MASK_ONLY",
            }

        # Per-layer mask bookkeeping (positions move after compaction; never mutate the global mask).
        if len(layer.visual_token_mask) < current_length:
            layer.visual_token_mask = np.pad(
                layer.visual_token_mask,
                (0, current_length - len(layer.visual_token_mask)),
                constant_values=False,
            )
        if layer.evicted_mask.shape[0] < current_length:
            layer.evicted_mask = np.pad(
                layer.evicted_mask,
                (0, current_length - layer.evicted_mask.shape[0]),
                constant_values=False,
            )
        if layer.scores.shape[1] < current_length:
            layer.scores = np.pad(
                layer.scores,
                ((0, 0), (0, current_length - layer.scores.shape[1])),
                constant_values=0.0,
            )

        protected = self._protected_mask(layer, current_length)
        protected_count = int(np.sum(protected))
        if layer.budget is None:
            # Fallback only when the engine did not call initialize_budget(); the engine always does.
            layer.budget = self._budget_for_unprotected(current_length - protected_count)
        budget = layer.budget

        unprotected_idx = np.where(~protected)[0]
        unprotected_len = len(unprotected_idx)
        selected_count = 0
        if unprotected_len <= budget:
            retained = np.ones(current_length, dtype=bool)
            recent_size = unprotected_len
        else:
            retained = protected.copy()
            recent_size = min(budget, max(1, int(budget * self.config.recent_ratio))) if budget > 0 else 0
            recent_idx = unprotected_idx[unprotected_len - recent_size:]
            retained[recent_idx] = True
            hh_budget = budget - recent_size
            candidate_indices = unprotected_idx[:unprotected_len - recent_size]
            if len(candidate_indices) > 0 and hh_budget > 0:
                layer_scores = layer.scores[:, :current_length].mean(axis=0)
                candidate_scores = layer_scores[candidate_indices]
                # Deterministic: score descending, ties broken toward later (more recent) positions.
                if self.config.tie_breaking == "recent":
                    order = np.lexsort((candidate_indices, candidate_scores))[::-1]
                else:
                    order = np.argsort(-candidate_scores)
                selected_count = min(len(order), hh_budget)
                retained[candidate_indices[order[:selected_count]]] = True
        layer.evicted_mask[:current_length] = ~retained

        evicted_count = int(np.sum(~retained))
        retained_count = int(np.sum(retained))
        metrics = {
            "total_tokens": current_length,
            "retained_tokens": retained_count,
            "evicted_tokens": evicted_count,
            "heavy_hitters": selected_count,  # Protected tokens are NOT heavy hitters.
            "recent_tokens": recent_size,
            "protected_tokens": protected_count,
            "unprotected_tokens": unprotected_len,
            "unprotected_retained": retained_count - protected_count,
            "budget": budget,  # applies to UNPROTECTED tokens only (R4-1)
            "per_step_evicted_slot_rate": float(evicted_count / max(1, current_length)),
            "kv_bytes_freed": "NOT_MEASURED_LOGICAL_MASK_ONLY",
        }
        return retained, metrics

    def compute_eviction_plan_per_head(self, layer_idx: int, current_length: int) -> tuple[np.ndarray, dict[str, Any]]:
        """Per-KV-head heavy-hitter selection (upstream fidelity A4-02, tracker level).

        Same budget rule as compute_eviction_plan, applied to each KV head's own scores: protected
        tokens are kept in every head, the recent window is shared (newest unprotected tokens), and
        each head keeps its own top heavy hitters among the remaining candidates. Returns a
        (num_kv_heads, current_length) bool mask.

        Pure with respect to the layer (ITER2 D10 N-12): it writes nothing to the layer. The evicted
        mask, the stored budget and the stored scores are left unchanged. If the layer's budget is
        still None it is computed from the current unprotected count for this call only (not stored);
        scores shorter than current_length are zero-padded in a local copy only.
        """
        layer = self.layers[layer_idx]
        H = self.num_kv_heads
        if layer.is_recurrent:
            retained = np.ones((H, current_length), dtype=bool)
            return retained, {"retained_tokens_per_head": [current_length] * H, "budget": None}
        scores = layer.scores
        if scores.shape[1] < current_length:
            scores = np.pad(scores, ((0, 0), (0, current_length - scores.shape[1])), constant_values=0.0)
        protected = self._protected_mask(layer, current_length)
        if layer.budget is not None:
            budget = layer.budget
        else:
            budget = self._budget_for_unprotected(current_length - int(protected.sum()))
        unprotected_idx = np.where(~protected)[0]
        unprotected_len = len(unprotected_idx)
        retained = np.zeros((H, current_length), dtype=bool)
        if unprotected_len <= budget:
            retained[:, :] = True
        else:
            retained[:, protected] = True
            recent_size = min(budget, max(1, int(budget * self.config.recent_ratio))) if budget > 0 else 0
            recent_idx = unprotected_idx[unprotected_len - recent_size:]
            retained[:, recent_idx] = True
            hh_budget = budget - recent_size
            candidate_indices = unprotected_idx[:unprotected_len - recent_size]
            if len(candidate_indices) > 0 and hh_budget > 0:
                for h in range(H):
                    cand_scores = scores[h, candidate_indices]
                    order = np.lexsort((candidate_indices, cand_scores))[::-1]
                    retained[h, candidate_indices[order[:min(len(order), hh_budget)]]] = True
        metrics = {
            "retained_tokens_per_head": retained.sum(axis=1).tolist(),
            "budget": budget,
            "per_head": True,
        }
        return retained, metrics

    def keep_mask_without_protection(self, layer_idx: int, length: int, keep_budget: int) -> np.ndarray:
        """STARLIGHT candidate set: H2O's keep choice over ALL positions, with NO internal protection.

        The same recent-window + heavy-hitter rule as compute_eviction_plan, but sinks and visual
        tokens are not protected here. Under STARLIGHT the controller owns protection: it vetoes
        candidates that are protected instead of evicting them. Existing h2o_base behaviour is
        unchanged; this is a separate method and does not touch the layer's budget or masks.
        Returns a bool keep mask of length `length` with exactly min(keep_budget, length) True.
        """
        if length < 0 or keep_budget < 0:
            raise ValueError("length and keep_budget must be non-negative")
        keep = np.zeros(length, dtype=bool)
        if length == 0 or keep_budget == 0:
            return keep
        if length <= keep_budget:
            keep[:] = True
            return keep
        layer = self.layers[layer_idx]
        scores = np.zeros((self.num_kv_heads, length), dtype=np.float32)
        usable = min(length, layer.scores.shape[1])
        scores[:, :usable] = layer.scores[:, :usable]
        recent = min(keep_budget, max(1, int(keep_budget * self.config.recent_ratio)))
        keep[length - recent:] = True
        heavy_budget = keep_budget - recent
        older = np.arange(length - recent)
        if heavy_budget > 0 and older.size > 0:
            older_scores = scores[:, :older.size].mean(axis=0)
            # Score descending; ties toward later positions, matching compute_eviction_plan.
            order = np.lexsort((older, older_scores))[::-1]
            keep[older[order[:min(heavy_budget, older.size)]]] = True
        return keep

    def unprotected_count(self, layer_idx: int, length: int) -> int:
        """Number of UNPROTECTED (text-side) positions among the first `length` physical slots."""
        layer = self.layers[layer_idx]
        return int(length - np.sum(self._protected_mask(layer, length)))

    def compact_layer_state(self, layer_idx: int, retained_indices: np.ndarray) -> None:
        """Synchronize tracker cumulative scores and mask with physical KV cache compaction.

        Args:
            layer_idx: index of full-attention layer compacted.
            retained_indices: 1D array of retained physical indices.
        """
        layer = self.layers[layer_idx]
        if layer.is_recurrent or len(retained_indices) == 0:
            return

        indices = np.asarray(retained_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(np.diff(indices) <= 0):
            raise ValueError("Compaction indices must be nonnegative, sorted and unique")
        required = int(indices[-1]) + 1
        if layer.scores.shape[1] < required:
            layer.scores = np.pad(
                layer.scores,
                ((0, 0), (0, required - layer.scores.shape[1])),
                constant_values=0.0,
            )
        if layer.visual_token_mask.shape[0] < required:
            layer.visual_token_mask = np.pad(
                layer.visual_token_mask,
                (0, required - layer.visual_token_mask.shape[0]),
                constant_values=False,
            )
        # Eviction masks are indexed by physical positions, not original IDs.
        layer.scores = layer.scores[:, indices]
        layer.evicted_mask = np.zeros(len(indices), dtype=bool)
        layer.visual_token_mask = layer.visual_token_mask[indices]
        # self.visual_token_mask preserves ORIGINAL positions as an immutable
        # cross-layer source. Each layer owns its physical-position mask.
        self.seq_len = len(indices)


class H2OEvictionPolicy:
    """Manages multi-layer H2O eviction decisions and telemetry."""

    def __init__(self, config: H2OConfig | None = None, num_layers: int = 24, num_kv_heads: int = 2):
        self.config = config or H2OConfig()
        self.tracker = H2OCacheTracker(self.config, num_layers=num_layers, num_kv_heads=num_kv_heads)

    def step(self, layer_idx: int, attention_matrix: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        self.tracker.update_attention_scores(layer_idx, attention_matrix)
        cur_len = attention_matrix.shape[-1]
        return self.tracker.compute_eviction_plan(layer_idx, cur_len)


class H2OMTPTransaction:
    """Transactional state machine for Speculative Decoding (MTP) with H2O cache rollback."""

    def __init__(self, tracker: H2OCacheTracker):
        self.tracker = tracker
        self._snapshot: dict[str, Any] | None = None
        self.in_transaction = False

    def begin(self) -> None:
        """Snapshot current KV state and cumulative scores before speculative proposals."""
        if self.in_transaction:
            raise RuntimeError("Nested MTP transactions not allowed: transaction already active.")
        self._snapshot = {
            "seq_len": self.tracker.seq_len,
            "layers": [
                {
                    "scores": np.copy(layer.scores),
                    "evicted_mask": np.copy(layer.evicted_mask),
                    "visual_token_mask": np.copy(layer.visual_token_mask),
                }
                for layer in self.tracker.layers
            ],
            "visual_mask": np.copy(self.tracker.visual_token_mask),
        }
        self.in_transaction = True

    def commit(self, accepted_tokens: int) -> None:
        """Commit speculative steps upon verifier acceptance."""
        if not self.in_transaction or self._snapshot is None:
            raise RuntimeError("No active MTP transaction to commit")
        if accepted_tokens < 0:
            raise ValueError(f"accepted_tokens cannot be negative: {accepted_tokens}")
        proposed = max(0, self.tracker.seq_len - self._snapshot["seq_len"])
        if accepted_tokens > proposed:
            raise ValueError(
                f"accepted_tokens {accepted_tokens} exceed {proposed} proposed tokens; "
                "refusing to commit positions that were never scored"
            )

        if accepted_tokens == 0:
            self.rollback()
            return

        orig_seq_len = self._snapshot["seq_len"]
        new_seq_len = orig_seq_len + accepted_tokens
        self.tracker.seq_len = new_seq_len

        # Reset scores and mask beyond new_seq_len so rejected tokens leave zero residual state
        for layer in self.tracker.layers:
            if layer.scores.shape[1] > new_seq_len:
                layer.scores[:, new_seq_len:] = 0.0
            if len(layer.evicted_mask) > new_seq_len:
                layer.evicted_mask[new_seq_len:] = False
            if len(layer.visual_token_mask) > new_seq_len:
                layer.visual_token_mask[new_seq_len:] = False

        self._snapshot = None
        self.in_transaction = False

    def rollback(self) -> None:
        """Rollback all speculative state changes when verifier rejects proposals."""
        if not self.in_transaction or self._snapshot is None:
            raise RuntimeError("No active MTP transaction to rollback")

        self.tracker.seq_len = self._snapshot["seq_len"]
        self.tracker.visual_token_mask = self._snapshot["visual_mask"]
        for layer, snap in zip(self.tracker.layers, self._snapshot["layers"]):
            layer.scores = snap["scores"]
            layer.evicted_mask = snap["evicted_mask"]
            layer.visual_token_mask = snap["visual_token_mask"]

        self._snapshot = None
        self.in_transaction = False



def calibrate_h2o_ratio_to_retained(measure, target_mean_retained: float, prompt_len: int, min_budget: int,
                                    *, image_present: bool) -> dict:
    """Matched retention by MEASUREMENT: find the H2O budget whose measured mean retained count
    is closest to a target (ITER2 D4 S-2).

    `measure(budget_ratio) -> mean retained entries per decode step` must run the real engine.
    A budget-to-retained formula is not used: H2O's out-of-budget protected entries make the
    mapping non-linear (measured: budget 25 gives retained 26, not 29). The budget is the integer
    b with int(prompt_len * ratio) == b, ratio = (b + 0.5) / prompt_len. Bisection assumes the
    measured mean is non-decreasing in b; the best of the bisection point and its neighbours is
    returned with its measured value, so the residual is reported, never hidden.

    TEXT-ONLY PROMPTS ONLY (ITER2 D10 N-8). The engine budget is
    max(min_budget, int(unprotected_length * ratio)), and unprotected_length excludes protected
    tokens (sinks and, with protect_visual_tokens, image tokens). The mapping above equals the
    engine's budget only when no token is protected outside the budget. An image prompt therefore
    gets a budget off by its protected count, so `image_present=True` is refused.
    `image_present` is required: the caller must state it.
    """
    if type(image_present) is not bool:
        raise TypeError("image_present must be an explicit bool")
    if image_present:
        raise ValueError(
            "calibrate_h2o_ratio_to_retained is valid only for TEXT-ONLY prompts: image prompts add "
            "protected visual entries outside the budget, so the budget-to-retained mapping is off by "
            "the protected count. Calibrate on a text-only prompt or extend the helper first."
        )
    if type(prompt_len) is not int or prompt_len < 1:
        raise ValueError("prompt_len must be a positive integer")
    lo = max(1, int(min_budget))
    hi = prompt_len
    if lo > hi:
        raise ValueError("min_budget exceeds prompt_len: no calibratable budget")
    target = float(target_mean_retained)
    measured: dict[int, float] = {}

    def m(b: int) -> float:
        if b not in measured:
            measured[b] = float(measure((b + 0.5) / prompt_len))
        return measured[b]

    left, right = lo, hi
    while left < right:
        mid = (left + right) // 2
        if m(mid) < target:
            left = mid + 1
        else:
            right = mid
    candidates = [b for b in (left - 1, left, left + 1) if lo <= b <= hi]
    best = min(candidates, key=lambda b: abs(m(b) - target))
    return {
        "budget": best,
        "budget_ratio": (best + 0.5) / prompt_len,
        "measured_mean_retained": m(best),
        "target_mean_retained": target,
        "evaluations": len(measured),
        "method": "bisection_on_measured_retained_mean",
    }
