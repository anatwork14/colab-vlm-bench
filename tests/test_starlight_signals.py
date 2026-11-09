"""STARLIGHT Stage 1 signal sources (cpu_pytorch_reference_not_vllm).

- H2O candidate set without H2O's internal protection (the controller owns protection).
- DA protection mapped from scripted scope spans through ORIGINAL token positions, so it
  survives KV compaction.
- Grounding protection: token positions of evidence keyword spans from the ECRD EvidencePool
  index. No benchmark answer is ever read.
- Evidence validity: stale (source mismatch), expired (request mismatch), unverified and
  missing declarations are excluded and reported, never silently protected.
"""

import hashlib

import numpy as np
import pytest

from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.h2o import H2OCacheTracker, H2OConfig
from vlmbench.infrastructure.starlight import (
    compact_orig_positions,
    da_protection_from_spans,
    evidence_validity,
    grounding_protection_from_pool,
    match_token_spans,
)

SHA_A = hashlib.sha256(b"image-a").hexdigest()
SHA_B = hashlib.sha256(b"image-b").hexdigest()


# --- H2O keep set without internal protection ---------------------------------------------

def _tracker_with_scores(column_scores, config=None):
    t = H2OCacheTracker(config=config or H2OConfig(), num_layers=1, num_kv_heads=1)
    weights = np.asarray(column_scores, dtype=np.float32).reshape(1, 1, -1)
    t.update_attention_scores(0, weights)
    return t


def test_keep_mask_selects_recent_window_plus_heavy_hitters_exactly():
    scores = [0.1] * 10
    scores[0], scores[2], scores[4], scores[6] = 9.0, 5.0, 4.0, 3.0
    t = _tracker_with_scores(scores)
    keep = t.keep_mask_without_protection(0, length=10, keep_budget=4)
    # recent = max(1, int(4 * 0.2)) = 1 -> newest position 9; heavy = top 3 older by score.
    assert np.flatnonzero(keep).tolist() == [0, 2, 4, 9]
    assert int(keep.sum()) == 4


def test_keep_mask_has_no_internal_sink_or_visual_protection():
    # Position 1 is an attention sink under H2O's default protect_system_tokens=4, but the
    # STARLIGHT candidate set must not protect it: the controller owns protection.
    scores = [0.1] * 10
    scores[0], scores[2], scores[4], scores[6] = 9.0, 5.0, 4.0, 3.0
    # Budget set so the existing H2O plan really evicts (budget 2 over 6 unprotected tokens).
    existing = H2OConfig(min_budget=0, budget_ratio=0.4)
    t = _tracker_with_scores(scores, config=existing)
    t.initialize_budget(0, prefill_length=10)
    keep = t.keep_mask_without_protection(0, length=10, keep_budget=4)
    assert not keep[1]  # STARLIGHT candidate set: no sink protection
    retained, metrics = t.compute_eviction_plan(0, current_length=10)
    assert metrics["evicted_tokens"] > 0
    assert retained[1]  # existing h2o_base plan still protects the sink: unchanged behaviour


def test_keep_mask_is_independent_of_protection_config_and_handles_edges():
    scores = [0.1] * 8
    scores[3] = 7.0
    default = _tracker_with_scores(scores, config=H2OConfig())
    bare = _tracker_with_scores(scores, config=H2OConfig(protect_system_tokens=0, protect_visual_tokens=False))
    a = default.keep_mask_without_protection(0, length=8, keep_budget=3)
    b = bare.keep_mask_without_protection(0, length=8, keep_budget=3)
    assert np.array_equal(a, b)
    assert default.keep_mask_without_protection(0, length=8, keep_budget=0).sum() == 0
    assert default.keep_mask_without_protection(0, length=8, keep_budget=8).all()
    assert default.keep_mask_without_protection(0, length=5, keep_budget=9).all()


def test_keep_mask_ties_break_toward_later_positions():
    t = _tracker_with_scores([1.0] * 10)
    keep = t.keep_mask_without_protection(0, length=10, keep_budget=4)
    # recent = 1 (position 9); heavy = 3 among 0..8 with all-equal scores -> later ones win.
    assert np.flatnonzero(keep).tolist() == [6, 7, 8, 9]


# --- DA protection through original positions ----------------------------------------------

def test_da_protection_maps_scope_spans_through_original_positions_after_compaction():
    # After compaction, physical slot i holds original position orig[i]. Scope spans are
    # defined on ORIGINAL positions, so physical protection must follow the token, not the slot.
    orig = [0, 1, 2, 5, 9, 10, 11]
    spans = [("sink_global", 0, 2), ("focus_scripted_visual_segment", 5, 6)]
    protected = da_protection_from_spans(spans, orig)
    assert protected == {0: ("sink_global",), 1: ("sink_global",), 3: ("focus_scripted_visual_segment",)}


def test_da_protection_ignores_positions_outside_all_spans_and_empty_spans():
    assert da_protection_from_spans([("sink_global", 3, 3)], [0, 1, 2, 3]) == {}
    assert da_protection_from_spans([], [0, 1, 2]) == {}


def test_compact_orig_positions_tracks_retained_original_ids():
    orig = [0, 1, 2, 3, 4]
    after_append = compact_orig_positions(orig, retained=None, appended=5)
    assert after_append == [0, 1, 2, 3, 4, 5]
    assert compact_orig_positions(after_append, retained=[0, 2, 5], appended=None) == [0, 2, 5]
    with pytest.raises(ValueError):
        compact_orig_positions([0, 1], retained=[1, 0], appended=None)  # must be sorted/unique


# --- Grounding protection from the EvidencePool keyword index ------------------------------

class _WordTok:
    """Deterministic word tokenizer: one id per word, unknown words get fresh ids."""

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def encode(self, text):
        out = []
        for w in text.split():
            key = w.lower().strip(".,")
            if not key:
                continue
            out.append(self.vocab.setdefault(key, len(self.vocab) + 100))
        return out


def _item(eid, sha=SHA_A, verified=True, content="evidence text"):
    return EvidenceItem(evidence_id=eid, modality="text_span", content=content,
                        source_sha256=sha, verified=verified)


def test_match_token_spans_finds_all_contiguous_occurrences():
    prompt = [1, 2, 3, 2, 3, 9]
    assert match_token_spans(prompt, [2, 3]) == [1, 3]
    assert match_token_spans(prompt, [7]) == []
    assert match_token_spans(prompt, []) == []


def test_grounding_protects_token_positions_of_keyword_spans_only():
    tok = _WordTok()
    pool = EvidencePool("req-1")
    pool.add_evidence(_item("ev_red"), keywords=["red car"])
    prompt_text = "the photo shows a red car near the gate and a red car again"
    prompt = tok.encode(prompt_text)
    prot = grounding_protection_from_pool(
        pool, prompt, tok.encode, current_request_id="req-1", image_sha256=None,
    )
    # the(0) photo(1) shows(2) a(3) red(4) car(5) near(6) the(7) gate(8) and(9) a(10)
    # red(11) car(12) again(13): both "red car" spans, and nothing else.
    assert sorted(prot) == [4, 5, 11, 12]
    assert all(prot[p] == ("ev_red",) for p in prot)


def test_irrelevant_evidence_protection_is_measured_not_hidden():
    tok = _WordTok()
    pool = EvidencePool("req-1")
    pool.add_evidence(_item("ev_the"), keywords=["the"])  # a too-generic keyword
    prompt = tok.encode("the cat sat on the mat near the door")
    prot = grounding_protection_from_pool(pool, prompt, tok.encode, current_request_id="req-1",
                                          image_sha256=None)
    assert len(prot) == 3  # every "the" is protected: reported per evidence id, never dropped
    assert {labels for labels in prot.values()} == {("ev_the",)}


def test_grounding_ignores_evidence_without_a_prompt_span_match():
    tok = _WordTok()
    pool = EvidencePool("req-1")
    pool.add_evidence(_item("ev_absent"), keywords=["submarine"])
    assert grounding_protection_from_pool(pool, tok.encode("a cat on a mat"), tok.encode,
                                          current_request_id="req-1", image_sha256=None) == {}


def test_stale_source_evidence_is_excluded_and_reported():
    pool = EvidencePool("req-1")
    pool.add_evidence(_item("ev_old", sha=SHA_B), keywords=["car"])
    status = evidence_validity(pool, current_request_id="req-1", image_sha256=SHA_A)
    assert status == {"ev_old": ("stale", "source_sha256_mismatch")}
    tok = _WordTok()
    prot = grounding_protection_from_pool(pool, tok.encode("a car"), tok.encode,
                                          current_request_id="req-1", image_sha256=SHA_A)
    assert prot == {}


def test_expired_pool_from_another_request_is_excluded():
    pool = EvidencePool("req-earlier")
    pool.add_evidence(_item("ev_e"), keywords=["car"])
    status = evidence_validity(pool, current_request_id="req-now", image_sha256=None)
    assert status == {"ev_e": ("expired", "pool_request_id_mismatch")}
    tok = _WordTok()
    assert grounding_protection_from_pool(pool, tok.encode("a car"), tok.encode,
                                          current_request_id="req-now", image_sha256=None) == {}


def test_text_only_request_is_accepted_with_explicit_unchecked_label():
    pool = EvidencePool("req-1")
    pool.add_evidence(_item("ev_t"), keywords=["car"])
    assert evidence_validity(pool, current_request_id="req-1", image_sha256=None) == {
        "ev_t": ("valid", "source_unchecked_text_only"),
    }


def test_valid_image_source_is_accepted():
    pool = EvidencePool("req-1")
    pool.add_evidence(_item("ev_i", sha=SHA_A), keywords=["car"])
    assert evidence_validity(pool, current_request_id="req-1", image_sha256=SHA_A) == {
        "ev_i": ("valid", "source_sha256_match"),
    }


def test_missing_pool_declares_no_grounding_signal():
    assert grounding_protection_from_pool(None, [1, 2], lambda s: [1], current_request_id="r",
                                          image_sha256=None) == {}
