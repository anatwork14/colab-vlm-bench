"""STARLIGHT Stage 1 checks against the auditor's independent oracle (wF:p9,
docs/harness/STARLIGHT_INVARIANTS_ORACLE.md, spec not evidence). Each test names the oracle ID it
covers. cpu_pytorch_reference_not_vllm; nothing here is GPU, vLLM or a quality result.

Oracle items NOT covered here, and why, are listed in docs/harness/STARLIGHT_STAGE1_DESIGN.md
section 12 (coverage matrix): I-6 state hash, HC-10 cross-turn KV, HC-14 logit recomputation,
HC-5 relevance labels, and the open ambiguities A-1..A-6.
"""

import hashlib

import numpy as np
import pytest

from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.h2o import H2OCacheTracker, H2OConfig
from vlmbench.infrastructure.model_config import Qwen35ModelConfig
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import (
    REASON_CODES,
    STATUS_INFEASIBLE,
    STATUS_OK,
    da_protection_from_spans,
    plan_starlight_layer,
)

SHA = hashlib.sha256(b"no-image").hexdigest()
PROMPT = list(range(100, 120)) + [7, 9, 7, 8, 7] + list(range(120, 130))  # 30 tokens, "the" at 20,22,24


def _cfg():
    return Qwen35ModelConfig(
        model_id="Qwen/Qwen3.5-StarlightOracleTest", model_type="qwen3_5",
        architectures=("Qwen3_5ForConditionalGeneration",),
        vocab_size=512, hidden_size=64, num_hidden_layers=8, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128, max_position_embeddings=1024,
        image_token_id=500, vision_start_token_id=501, vision_end_token_id=502,
        pad_token_id=0, eos_token_id=1, full_attention_layers=(3, 7),
        recurrent_layers=(0, 1, 2, 4, 5, 6), mtp_num_speculative_tokens=2,
    )


class _WordTok:
    VOCAB = {"the": 7, "car": 8, "cat": 9}

    def encode(self, text):
        return [self.VOCAB[w] for w in text.lower().split() if w in self.VOCAB]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


def _pool(request_id, keywords, eid="ev_the"):
    pool = EvidencePool(request_id)
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality="text_span", content=" ".join(keywords),
                                   source_sha256=SHA, verified=True), keywords=list(keywords))
    return pool


def _run(arm, pool=None, record=True, budget=16, ratio=0.2, new=10, request_id="oracle_run"):
    torch_seed()
    engine = RealSevenArmEngine(
        method_arm=arm, device="cpu", config=_cfg(), stop_at_eos=False, h2o_min_budget=budget,
        h2o_budget_ratio=ratio, tokenizer=_WordTok(), starlight_record_decisions=record,
    )
    try:
        return engine.execute_request(
            request_id=request_id, prompt_token_ids=PROMPT, image_present=False,
            max_new_tokens=new, evidence_pool=pool,
        )
    finally:
        engine.close()


def torch_seed():
    import torch
    torch.manual_seed(1234)


def _records(result):
    recs = result.telemetry["starlight"]["decision_records"]
    assert recs, "decision records must be recorded when starlight_record_decisions=True"
    return recs


def _scored_tracker(scores):
    t = H2OCacheTracker(config=H2OConfig(), num_layers=8, num_kv_heads=2)
    for layer in (3, 7):
        t.update_attention_scores(layer, np.tile(np.asarray(scores, dtype=np.float32), (4, 1, 1)))
    return t


def _step(engine, tracker, length, budget, da_spans=(), grounding=None, enabled=("da", "grounding")):
    k_lens = {3: length, 7: length}
    orig = {3: np.arange(length - 1), 7: np.arange(length - 1)}
    return engine._starlight_build_step_plans(
        tracker, k_lens, orig, length - 1, budget, list(da_spans), dict(grounding or {}), tuple(enabled),
    )


def _engine_helper():
    torch_seed()
    return RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg())


# --- I-1, I-2, I-3, I-10: per-entry record checks on a long-context run -------------------

def test_I1_I2_eligibility_formula_and_protection_recomputed_from_records():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]))
    checked = 0
    for rec in _records(result):
        for e in rec["entries"]:
            protected = e["da_protected"] or e["grounding_protected"] or e["structural_protected"]
            eligible = e["h2o_candidate"] and not protected  # I-1 recomputed from the record
            assert (e["decision"] == "EVICT") == eligible, e
            assert not (e["decision"] == "EVICT" and protected), e  # I-2
            checked += 1
    assert checked > 0


def test_I3_count_check_and_closed_reason_enum_at_every_step():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]))
    for rec in _records(result):
        assert rec["pre"] == rec["retained"] + rec["evicted"]  # |pre| = |R| + |evicted|
        assert rec["n_eligible"] == rec["evicted"]
        for e in rec["entries"]:
            assert e["reason_code"] in REASON_CODES
        evicted_codes = {e["reason_code"] for e in rec["entries"] if e["decision"] == "EVICT"}
        assert evicted_codes <= {"H2O_CANDIDATE_UNPROTECTED"}


def test_I10_decision_records_carry_every_required_field():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]))
    required = {"step", "phase", "layer", "budget", "pre", "retained", "evicted", "status", "n_da",
                "n_grounding", "n_h2o_candidates", "n_eligible", "n_protected",
                "n_protected_outside_budget", "entries"}
    for rec in _records(result):
        assert required <= set(rec), required - set(rec)
        assert rec["phase"] == "decode"  # H-7: eviction only at decode, stated in the record
        assert rec["layer"] in (3, 7)
        assert rec["status"] in (STATUS_OK, STATUS_INFEASIBLE)


def test_I9_HC15_identical_inputs_give_identical_records_and_tokens():
    pool = _pool("oracle_run", ["the"])
    a = _run("starlight_s1", pool=pool)
    b = _run("starlight_s1", pool=pool)
    assert a.tokens_generated == b.tokens_generated
    assert a.telemetry["starlight"]["decision_records"] == b.telemetry["starlight"]["decision_records"]


# --- HC-3b, HC-8, HC-11 (helper-level fixtures) ----------------------------------------------

def test_HC3b_protection_exactly_budget_is_ok_and_fills_budget():
    engine = _engine_helper()
    tracker = _scored_tracker([1] * 10)
    _, _, plans = _step(engine, tracker, length=10, budget=6, da_spans=[("sink_global", 0, 6)])
    plan = plans[0]
    assert plan.status == STATUS_OK
    assert len(plan.retained) == 6  # |R| = B exactly
    assert plan.retained == (0, 1, 2, 3, 4, 5)
    assert plan.evicted == (6, 7, 8, 9)


def test_HC8_malformed_da_span_is_a_raised_error_never_an_unprotected_entry():
    with pytest.raises(ValueError):
        da_protection_from_spans([("sink_global", 5, 2)], list(range(10)))  # start > end
    with pytest.raises(ValueError):
        da_protection_from_spans([("sink_global", -1, 3)], list(range(10)))  # negative start
    with pytest.raises(ValueError):
        da_protection_from_spans([("sink_global", 1.5, 3)], list(range(10)))  # non-integer
    with pytest.raises(ValueError):
        da_protection_from_spans([("", 0, 3)], list(range(10)))  # unlabelled


def test_HC11_no_eligible_candidates_reports_zero_eligible_and_no_eviction():
    result = _run("starlight_s1", pool=None, budget=64, new=4)  # budget covers the whole sequence
    recs = _records(result)
    assert all(r["evicted"] == 0 and r["n_eligible"] == 0 and r["status"] == STATUS_OK for r in recs)


# --- HC-10, HC-12, HC-13, HC-14, HC-5, HC-4, HC-7 (engine telemetry) -------------------------

def test_HC10_evicted_entry_never_returns_to_a_later_step():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]), new=12)
    by_layer: dict[int, list[dict]] = {}
    for rec in _records(result):
        by_layer.setdefault(rec["layer"], []).append(rec)
    for layer, recs in by_layer.items():
        evicted_positions: set[int] = set()
        for rec in recs:  # steps are in order per layer
            present = {e["position"] for e in rec["entries"]}
            assert not (present & evicted_positions), (layer, rec["step"])
            evicted_positions |= {e["position"] for e in rec["entries"] if e["decision"] == "EVICT"}


def test_HC12_gdn_layers_are_exempt_and_labelled_and_absent_from_records():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]))
    gdn = result.telemetry["starlight"]["gdn_layer_records"]
    assert [g["layer"] for g in gdn] == [0, 1, 2, 4, 5, 6]
    for g in gdn:
        assert g["decision"] == "GDN_EXEMPT_NOT_TOKEN_ADDRESSABLE"
        assert g["evicted"] == 0
    assert {rec["layer"] for rec in _records(result)} == {3, 7}


def test_HC13_structural_vision_tokens_are_measured_and_policy_marked_unratified():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]))
    s = result.telemetry["starlight"]
    assert isinstance(s["structural_tokens_evicted"], int)
    assert s["structural_policy"] == "PROTECTED_LEAD_DECISION_Q2"
    assert s["structural_tokens_evicted"] == 0


def test_HC14_cache_length_counter_matches_retained_count_every_step():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]), new=12)
    assert result.telemetry["starlight"]["cache_length_mismatch_steps"] == 0


def test_HC5_irrelevant_protection_metric_is_present_and_marked_unmeasured():
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]))
    assert result.telemetry["starlight"]["protected_irrelevant_fraction"] == "NOT_MEASURED_NO_RELEVANCE_LABELS"


def test_HC4_expired_evidence_carries_the_oracle_reason_code():
    pool = _pool("some_earlier_request", ["the"])
    result = _run("starlight_s1", pool=pool, request_id="oracle_now")
    verdict = result.telemetry["starlight"]["grounding_evidence_verdicts"]["ev_the"]
    assert verdict == ["expired", "pool_request_id_mismatch", "EXPIRED_EVIDENCE"]
    assert all(r["n_grounding"] == 0 for r in _records(result))


def test_HC7_missing_evidence_declaration_is_explicit_not_silent():
    result = _run("starlight_s1", pool=None)
    s = result.telemetry["starlight"]
    assert s["grounding_signal_status"] == "NO_EVIDENCE_DECLARED"
    assert all(r["n_grounding"] == 0 for r in _records(result))


# --- HC-1, HC-2 and activation / toggle assertions (negative controls included) --------------

def _da_fixture_plans(enabled):
    engine = _engine_helper()
    tracker = _scored_tracker([0, 0, 5, 4, 3, 2, 1, 1, 1, 1])
    _, _, plans = _step(engine, tracker, length=10, budget=6, da_spans=[("sink_global", 0, 2)],
                        enabled=enabled)
    return plans[0]


def _grounding_fixture_plans(enabled):
    engine = _engine_helper()
    tracker = _scored_tracker([5, 4, 3, 2, 1, 0, 0, 0, 0, 1])
    _, _, plans = _step(engine, tracker, length=10, budget=6, grounding={5: ("ev_1",)}, enabled=enabled)
    return plans[0]


def test_A_TOGGLE_da_signal_changes_the_retained_set_on_the_da_fixture():
    full = _da_fixture_plans(("da", "grounding"))
    no_da = _da_fixture_plans(("grounding",))
    assert full.retained != no_da.retained, "DA signal inert on its own fixture"
    # Negative control: without DA (H2O-only behaviour here) the DA-declared candidates are evicted.
    assert {0, 1} <= set(no_da.evicted)
    assert {0, 1}.isdisjoint(full.evicted)


def test_A_TOGGLE_grounding_signal_changes_the_retained_set_on_the_grounding_fixture():
    full = _grounding_fixture_plans(("da", "grounding"))
    no_gr = _grounding_fixture_plans(("da",))
    assert full.retained != no_gr.retained, "grounding signal inert on its own fixture"
    assert 5 in no_gr.evicted  # negative control: the grounding-declared candidate is evicted without it
    assert 5 not in full.evicted


def test_A_DA_activates_on_a_da_h2o_conflict_and_retains_the_candidate():
    plan = _da_fixture_plans(("da", "grounding"))
    vetoed_by_da = [d for d in plan.decisions
                    if d.h2o_candidate and any(s.startswith("da:") for s in d.signals_fired)]
    assert vetoed_by_da, "A-DA: no DA-protected H2O candidate on the conflict fixture"
    assert all(d.decision == "RETAIN_PROTECTED_VETOED_H2O_CANDIDATE" for d in vetoed_by_da)


def test_A_GROUND_activates_on_a_grounding_h2o_conflict_and_retains_the_candidate():
    plan = _grounding_fixture_plans(("da", "grounding"))
    vetoed = [d for d in plan.decisions
              if d.h2o_candidate and any(s.startswith("grounding:") for s in d.signals_fired)]
    assert vetoed, "A-GROUND: no grounding-protected H2O candidate on the conflict fixture"


def test_A_NODA_and_A_NOGROUND_report_zero_for_the_removed_signal_on_every_step():
    pool = _pool("oracle_run", ["the"])
    no_da = _records(_run("starlight_s1_no_da", pool=pool))
    assert all(r["n_da"] == 0 for r in no_da), "A-NODA: DA signal leaked into the no-DA ablation"
    no_gr = _records(_run("starlight_s1_no_grounding", pool=pool))
    assert all(r["n_grounding"] == 0 for r in no_gr), "A-NOGROUND: grounding leaked into the ablation"


def test_A_H2O_candidates_exist_on_the_long_context_fixture():
    recs = _records(_run("starlight_s1", pool=_pool("oracle_run", ["the"])))
    assert any(r["n_h2o_candidates"] > 0 for r in recs), "A-H2O: arm never saw an eviction candidate"


def test_A_NOT_ACTIVATED_label_is_not_needed_for_grounding_arm_on_this_fixture():
    # The grounding keyword is present in PROMPT, so n_grounding must be > 0 on the pressure run.
    recs = _records(_run("starlight_s1", pool=_pool("oracle_run", ["the"])))
    assert any(r["n_grounding"] > 0 for r in recs)


def test_I7_surviving_entries_keep_their_original_position_ids_across_steps():
    # Added after the implementation (coverage of an existing property), not red-first.
    # Records key entries by ORIGINAL position. A renumbering bug would make a survivor's id appear
    # as a smaller or new id in the next step, so every survivor must reappear with the same id.
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]), new=12)
    by_layer: dict[int, list[dict]] = {}
    for rec in _records(result):
        by_layer.setdefault(rec["layer"], []).append(rec)
    for layer, recs in by_layer.items():
        for prev, nxt in zip(recs, recs[1:]):
            survivors = {e["position"] for e in prev["entries"] if e["decision"] != "EVICT"}
            present_next = {e["position"] for e in nxt["entries"]}
            assert survivors <= present_next, (layer, nxt["step"])
            assert max(present_next) == nxt["entries"][-1]["position"]  # newest token is the last entry


def test_A_MASK_da_attention_mask_is_not_reported_as_kv_saving_on_starlight_arms():
    # Added after the implementation (coverage of an existing property). A logical DA restriction must
    # never appear as a KV-residency or byte saving for the STARLIGHT arms.
    result = _run("starlight_s1", pool=_pool("oracle_run", ["the"]))
    cost = result.telemetry["da_cost_accounting"]
    assert cost["attention_mask_restriction_ratio"] is None
    assert result.attention_mask_restriction_ratio is None
    assert cost["original_paper_protocol_qualified"] is False
