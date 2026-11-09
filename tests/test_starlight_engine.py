"""STARLIGHT Stage 1 wired into RealSevenArmEngine (cpu_pytorch_reference_not_vllm).

Hard cases from the ITER2 brief, each exercised through the engine or its per-step helper:
DA/H2O conflict, grounding/H2O conflict, protection > budget (INFEASIBLE), expired and stale
evidence, irrelevant evidence wrongly protected (measured), long-context pressure, missing and
malformed declarations, multi-request isolation, multi-turn continuity, no eligible candidates.
Invariants: protected never evicted; evicted subset of H2O candidates; retained <= budget unless
INFEASIBLE; ablations differ only by the removed signal; no-eviction oracle; recurrent exemption.
"""

import hashlib

import numpy as np
import pytest
import torch

from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.h2o import H2OCacheTracker, H2OConfig
from vlmbench.infrastructure.model_config import Qwen35ModelConfig
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import (
    STATUS_INFEASIBLE,
    STATUS_OK,
    combine_summaries,
    summarize_plans,
)

SEVEN_ARMS = frozenset(("base", "h2o_base", "da_base", "ecrd_base", "mtp", "h2o_mtp", "da_mtp"))
STARLIGHT_ARMS = frozenset(("starlight_s1", "starlight_s1_no_da", "starlight_s1_no_grounding"))
SHA = hashlib.sha256(b"no-image").hexdigest()


def _cfg():
    return Qwen35ModelConfig(
        model_id="Qwen/Qwen3.5-StarlightTest", model_type="qwen3_5",
        architectures=("Qwen3_5ForConditionalGeneration",),
        vocab_size=512, hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, max_position_embeddings=1024,
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
        pad_token_id=0, eos_token_id=1, full_attention_layers=(3, 7),
        recurrent_layers=(0, 1, 2, 4, 5, 6), mtp_num_speculative_tokens=2,
    )


class _WordTok:
    """Fake tokenizer: case- and space-insensitive word ids below the tiny vocab."""

    VOCAB = {"the": 7, "car": 8, "cat": 9}

    def encode(self, text):
        return [self.VOCAB[w] for w in text.lower().split() if w in self.VOCAB]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


def _engine(arm, **kw):
    torch.manual_seed(1234)
    return RealSevenArmEngine(method_arm=arm, device="cpu", config=_cfg(), **kw)


def _pool(request_id, keywords, eid="ev_1", sha=SHA):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality="text_span", content=" ".join(keywords),
                                   source_sha256=sha, verified=True), keywords=list(keywords))
    return pool


def _scored_tracker(scores):
    t = H2OCacheTracker(config=H2OConfig(), num_layers=8, num_kv_heads=2)
    for layer in (3, 7):
        w = np.tile(np.asarray(scores, dtype=np.float32), (4, 1, 1))  # 4 query heads, 1 query
        t.update_attention_scores(layer, w)
    return t


def _step(engine, tracker, length, budget, da_spans=(), grounding=None, enabled=("da", "grounding")):
    k_lens = {3: length, 7: length}
    orig = {3: np.arange(length - 1), 7: np.arange(length - 1)}
    return engine._starlight_build_step_plans(
        tracker, k_lens, orig, length - 1, budget, list(da_spans), dict(grounding or {}), tuple(enabled),
    )


# --- arm registry -------------------------------------------------------------------------

def test_supported_arms_adds_three_starlight_arms_and_keeps_the_seven():
    assert RealSevenArmEngine.SUPPORTED_ARMS == SEVEN_ARMS | STARLIGHT_ARMS


def test_existing_h2o_base_has_no_starlight_telemetry():
    result = _engine("h2o_base").execute_request(
        request_id="r_h2o", prompt_token_ids=list(range(100, 130)), image_present=False, max_new_tokens=4)
    assert "starlight" not in result.telemetry


# --- helper-level hard cases (one engine step, deterministic scores) ----------------------

def test_da_h2o_conflict_is_vetoed_not_evicted():
    engine = _engine("starlight_s1")
    tracker = _scored_tracker([0, 0, 5, 4, 3, 2, 1, 1, 1, 1])
    plans_t, orig_next, plans = _step(engine, tracker, length=10, budget=6,
                                      da_spans=[("sink_global", 0, 2)])
    plan = plans[0]
    assert plan.status == STATUS_OK
    assert plan.h2o_candidates_vetoed == 2            # slots 0 and 1: H2O wanted them gone
    assert plan.evicted == (5, 6, 7, 8)
    assert plan.retained == (0, 1, 2, 3, 4, 9)
    assert len(plan.retained) == 6                    # budget respected
    assert sorted(plans_t[3].tolist()) == list(plan.retained)
    assert orig_next[3].tolist() == list(plan.retained)


def test_grounding_h2o_conflict_is_vetoed_not_evicted():
    engine = _engine("starlight_s1")
    tracker = _scored_tracker([5, 4, 3, 2, 1, 0, 0, 0, 0, 1])
    _, _, plans = _step(engine, tracker, length=10, budget=6, grounding={5: ("ev_1",)})
    plan = plans[0]
    assert plan.status == STATUS_OK
    assert plan.decisions[5].decision == "RETAIN_PROTECTED_VETOED_H2O_CANDIDATE"
    assert plan.decisions[5].signals_fired == ("grounding:ev_1",)
    assert plan.evicted == (4, 6, 7, 8)
    assert plan.retained == (0, 1, 2, 3, 5, 9)


def test_protection_exceeding_budget_is_infeasible_and_protected_stay():
    engine = _engine("starlight_s1")
    tracker = _scored_tracker([1] * 10)
    _, _, plans = _step(engine, tracker, length=10, budget=6, da_spans=[("sink_global", 0, 7)])
    plan = plans[0]
    assert plan.status == STATUS_INFEASIBLE
    assert plan.retained == tuple(range(7))          # protected 7 > budget 6: reported, not hidden
    assert plan.evicted == (7, 8, 9)                 # only unprotected candidates are evicted


def test_within_budget_has_no_candidates_and_no_eviction():
    engine = _engine("starlight_s1")
    tracker = _scored_tracker([1] * 5)
    _, _, plans = _step(engine, tracker, length=5, budget=6)
    assert plans[0].status == STATUS_OK
    assert plans[0].evicted == ()
    assert plans[0].h2o_candidates_vetoed == 0


# --- engine-level runs --------------------------------------------------------------------

def test_no_eviction_oracle_matches_base_tokens_when_budget_covers_the_sequence():
    prompt = list(range(100, 140))
    base_engine = _engine("base", stop_at_eos=False)
    base_tokens = base_engine.execute_request(
        request_id="oracle", prompt_token_ids=prompt, image_present=False, max_new_tokens=8).tokens_generated
    star_engine = _engine("starlight_s1", stop_at_eos=False, h2o_min_budget=256)
    result = star_engine.execute_request(
        request_id="oracle", prompt_token_ids=prompt, image_present=False, max_new_tokens=8)
    assert result.tokens_generated == base_tokens
    assert result.telemetry["starlight"]["evicted_total"] == 0


def test_long_context_pressure_keeps_budget_and_invariants():
    prompt = list(range(100, 140))  # 40 tokens; budget 16 forces eviction from the first step
    result = _engine("starlight_s1", stop_at_eos=False, h2o_min_budget=16, h2o_budget_ratio=0.2) \
        .execute_request(request_id="pressure", prompt_token_ids=prompt, image_present=False,
                         max_new_tokens=12)
    s = result.telemetry["starlight"]
    assert s["budget_total"] == 16
    assert s["evicted_total"] > 0
    assert s["max_retained_per_layer_step"] <= 16
    assert s["infeasible_steps"] == 0
    assert s["invariant_protected_evicted"] == 0
    assert s["invariant_evicted_not_candidate"] == 0


def test_expired_evidence_protects_nothing_and_is_reported():
    tok = _WordTok()
    pool = _pool("request_earlier", ["the"])
    result = _engine("starlight_s1", tokenizer=tok, stop_at_eos=False).execute_request(
        request_id="request_now", prompt_token_ids=[7, 9, 7, 8, 7], image_present=False,
        max_new_tokens=2, evidence_pool=pool)
    s = result.telemetry["starlight"]
    assert s["grounding_protected_by_evidence"] == {}
    assert s["grounding_evidence_verdicts"] == {"ev_1": ["expired", "pool_request_id_mismatch"]}
    assert s["protected_count_by_source"]["grounding"] == 0


def test_irrelevant_evidence_protection_is_measured_per_evidence_id():
    tok = _WordTok()
    pool = _pool("req_irr", ["the"], eid="ev_the")
    result = _engine("starlight_s1", tokenizer=tok, stop_at_eos=False).execute_request(
        request_id="req_irr", prompt_token_ids=[7, 9, 7, 8, 7], image_present=False,
        max_new_tokens=2, evidence_pool=pool)
    s = result.telemetry["starlight"]
    assert s["grounding_protected_by_evidence"] == {"ev_the": 3}  # three "the" tokens, all reported
    assert s["protected_count_by_source"]["grounding"] > 0


def test_missing_declarations_are_explicit_not_silent():
    no_pool = _engine("starlight_s1", tokenizer=_WordTok(), stop_at_eos=False).execute_request(
        request_id="r_nopool", prompt_token_ids=[7, 9], image_present=False, max_new_tokens=1)
    assert no_pool.telemetry["starlight"]["grounding_signal_status"] == "NO_EVIDENCE_DECLARED"
    pool = _pool("r_notok", ["the"])
    no_tok = _engine("starlight_s1", stop_at_eos=False).execute_request(
        request_id="r_notok", prompt_token_ids=[7, 9], image_present=False, max_new_tokens=1,
        evidence_pool=pool)
    assert no_tok.telemetry["starlight"]["grounding_signal_status"] == "UNAVAILABLE_NO_TOKENIZER"
    assert no_tok.telemetry["starlight"]["protected_count_by_source"]["grounding"] == 0


def test_malformed_evidence_declaration_is_rejected_before_any_protection():
    bad = EvidenceItem(evidence_id="ev_bad", modality="text_span", content="the",
                       source_sha256="not-a-sha", verified=True)
    pool = EvidencePool("r_bad")
    with pytest.raises(ValueError, match="SHA256"):
        pool.add_evidence(bad, keywords=["the"])
    assert pool.total_items == 0


def test_multi_request_isolation_second_request_is_unaffected_by_first():
    tok = _WordTok()
    prompt = [7, 9, 7, 8, 7, 9, 8, 7, 9, 9, 8, 7]
    engine = _engine("starlight_s1", tokenizer=tok, stop_at_eos=False, h2o_min_budget=8,
                     h2o_budget_ratio=0.5)
    engine.execute_request(request_id="iso_a", prompt_token_ids=prompt, image_present=False,
                           max_new_tokens=4, evidence_pool=_pool("iso_a", ["the"], eid="ev_a"))
    second = engine.execute_request(request_id="iso_b", prompt_token_ids=prompt, image_present=False,
                                    max_new_tokens=4)
    fresh = _engine("starlight_s1", tokenizer=tok, stop_at_eos=False, h2o_min_budget=8,
                    h2o_budget_ratio=0.5).execute_request(
        request_id="iso_b", prompt_token_ids=prompt, image_present=False, max_new_tokens=4)
    assert second.tokens_generated == fresh.tokens_generated
    assert second.telemetry["starlight"]["protected_count_by_source"] == \
        fresh.telemetry["starlight"]["protected_count_by_source"]
    assert second.telemetry["starlight"]["grounding_protected_by_evidence"] == {}


def test_multi_turn_continuity_same_request_id_matches_fresh_engine():
    tok = _WordTok()
    prompt = [7, 9, 7, 8, 7, 9, 8, 7, 9, 9, 8, 7]
    engine = _engine("starlight_s1", tokenizer=tok, stop_at_eos=False, h2o_min_budget=8,
                     h2o_budget_ratio=0.5)
    pool = _pool("turns", ["the"])
    turn1 = engine.execute_request(request_id="turns", prompt_token_ids=prompt, image_present=False,
                                   max_new_tokens=3, evidence_pool=pool)
    turn2 = engine.execute_request(request_id="turns", prompt_token_ids=prompt, image_present=False,
                                   max_new_tokens=3, evidence_pool=pool)
    fresh = _engine("starlight_s1", tokenizer=tok, stop_at_eos=False, h2o_min_budget=8,
                    h2o_budget_ratio=0.5).execute_request(
        request_id="turns", prompt_token_ids=prompt, image_present=False, max_new_tokens=3,
        evidence_pool=pool)
    assert turn2.tokens_generated == fresh.tokens_generated == turn1.tokens_generated
    assert turn2.telemetry["starlight"]["evicted_total"] == fresh.telemetry["starlight"]["evicted_total"]


def test_ablation_no_da_removes_only_the_da_signal_and_does_not_enable_da_masking():
    tok = _WordTok()
    prompt = list(range(100, 140))
    pool = _pool("abl", ["the"])
    full = _engine("starlight_s1", tokenizer=tok, stop_at_eos=False, h2o_min_budget=16,
                   h2o_budget_ratio=0.2).execute_request(
        request_id="abl", prompt_token_ids=prompt, image_present=False, max_new_tokens=4, evidence_pool=pool)
    no_da = _engine("starlight_s1_no_da", tokenizer=tok, stop_at_eos=False, h2o_min_budget=16,
                    h2o_budget_ratio=0.2).execute_request(
        request_id="abl", prompt_token_ids=prompt, image_present=False, max_new_tokens=4, evidence_pool=pool)
    assert no_da.telemetry["starlight"]["signals_enabled"] == ["grounding"]
    assert no_da.telemetry["starlight"]["protected_count_by_source"]["da"] == 0
    assert no_da.telemetry["starlight"]["conflicts_by_source"]["da"] == 0
    assert full.telemetry["starlight"]["protected_count_by_source"]["da"] > 0
    assert no_da.telemetry["da_cost_accounting"]["attention_mask_restriction_ratio"] is None
    assert no_da.attention_mask_restriction_ratio is None


def test_starlight_arms_report_cpu_reference_labels_and_recurrent_exemption():
    result = _engine("starlight_s1", stop_at_eos=False, h2o_min_budget=64).execute_request(
        request_id="labels", prompt_token_ids=list(range(100, 120)), image_present=False, max_new_tokens=2)
    s = result.telemetry["starlight"]
    assert s["measurement_label"] == "cpu_pytorch_reference_not_vllm"
    assert s["method_variant"] == "starlight_stage1::da+grounding"
    assert s["method_variant_is_upstream_h2o"] is False
    assert s["recurrent_layers_exempt"] == [0, 1, 2, 4, 5, 6]
    assert s["recurrent_exemption_label"] == "RECURRENT_STATE_NOT_TOKEN_ADDRESSABLE_KV"
    assert s["layers"] == [3, 7]
    assert result.telemetry["h2o_logical_selection_accounting"]["method_variant"] == "NOT_APPLICABLE_STARLIGHT_ARM"
    assert 0.0 <= s["text_side_retained_fraction_last_step"] <= 1.0


def test_combine_summaries_adds_steps_and_keeps_invariant_counts():
    p = _scored_tracker([0, 0, 5, 4, 3, 2, 1, 1, 1, 1])
    engine = _engine("starlight_s1")
    _, _, plans = _step(engine, p, length=10, budget=6, da_spans=[("sink_global", 0, 2)])
    one = summarize_plans(plans)
    total = combine_summaries([one, one])
    assert total["steps"] == 2
    assert total["h2o_candidates_vetoed"] == 2 * one["h2o_candidates_vetoed"]
    assert total["evicted_total"] == 2 * one["evicted_total"]
    assert total["protected_evicted"] == 0 and total["evicted_not_candidate"] == 0
    assert total["status_counts"] == {STATUS_OK: 4}  # 2 steps x 2 full-attention layers
