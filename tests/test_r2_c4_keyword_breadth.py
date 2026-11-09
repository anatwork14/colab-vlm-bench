"""ITER3 R2 C-4 (closure doc): bound the keyword breadth of grounding protection.

Policy (reported explicitly in telemetry):
  * stop-word-like keywords are rejected (e.g. 'the'), whatever their frequency;
  * a keyword whose protected positions exceed MAX_KEYWORD_PROMPT_FRACTION of the prompt is rejected;
  * if the accepted protection still exceeds the budget, the existing INFEASIBLE semantics apply
    (protected entries are retained, the overflow is reported, nothing is silently dropped).
cpu_pytorch_reference_not_vllm.
"""
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import (
    KEYWORD_STOPWORDS,
    MAX_KEYWORD_PROMPT_FRACTION,
    STATUS_INFEASIBLE,
    grounding_protection_with_report,
    plan_starlight_layer,
)

from test_starlight_oracle import PROMPT, _cfg, _WordTok

SHA = "d" * 64
THE, CAR, CAT, MAT, FILLER = 7, 8, 9, 12, 30


class _Tok:
    VOCAB = {"the": THE, "car": CAR, "cat": CAT, "mat": MAT}

    def encode(self, text):
        return [self.VOCAB[w] for w in text.lower().split() if w in self.VOCAB]


def _pool(keywords, eid="ev_k", rid="c4"):
    pool = EvidencePool(rid)
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality="text_span", content=" ".join(keywords),
                                   source_sha256=SHA, verified=True), keywords=list(keywords))
    return pool


def _prompt(n, placed):
    """n filler tokens with {token_id: [positions]} placed on top."""
    ids = [FILLER] * n
    for tok, positions in placed.items():
        for p in positions:
            ids[p] = tok
    return ids


def _protect(pool, prompt_ids):
    tok = _Tok()
    return grounding_protection_with_report(pool, prompt_ids, tok.encode, "c4", None)


def test_the_is_in_the_stopword_list_and_the_policy_fraction_is_stated():
    assert "the" in KEYWORD_STOPWORDS
    assert MAX_KEYWORD_PROMPT_FRACTION == 0.2


def test_the_is_rejected_as_a_stopword_even_when_it_is_rare():
    prompt = _prompt(40, {THE: [5]})  # 'the' once in 40 tokens: breadth alone would accept it
    prot, report = _protect(_pool(["the"]), prompt)
    assert prot == {}
    assert report["rejected_stopword"] == ["the"]
    assert report["accepted_keywords"] == []


def test_a_keyword_over_the_prompt_fraction_cap_is_rejected_and_reported_with_its_fraction():
    prompt = _prompt(10, {CAR: [1, 4, 7]})  # 3 of 10 = 0.3 > 0.2
    prot, report = _protect(_pool(["car"]), prompt)
    assert prot == {}
    assert report["rejected_breadth"] == [{"keyword": "car", "protected_fraction": 0.3}]


def test_a_keyword_exactly_at_the_cap_is_accepted():
    prompt = _prompt(10, {CAR: [2, 6]})  # 2 of 10 = 0.2, equal to the cap
    prot, report = _protect(_pool(["car"]), prompt)
    assert prot == {2: ("ev_k",), 6: ("ev_k",)}
    assert report["accepted_keywords"] == ["car"]
    assert report["rejected_breadth"] == []


def test_an_accepted_keyword_protects_its_occurrences_and_is_reported():
    prompt = _prompt(20, {CAT: [3, 11]})  # 2 of 20 = 0.1
    prot, report = _protect(_pool(["cat"]), prompt)
    assert prot == {3: ("ev_k",), 11: ("ev_k",)}
    assert report["accepted_keywords"] == ["cat"]


def test_the_stopword_check_runs_before_any_position_is_protected():
    prompt = _prompt(40, {THE: [5], CAR: [10]})
    prot, report = _protect(_pool(["the", "car"]), prompt)
    assert prot == {10: ("ev_k",)}
    assert report["rejected_stopword"] == ["the"]
    assert report["accepted_keywords"] == ["car"]


def test_protection_over_budget_stays_infeasible_and_is_not_silently_dropped():
    # Accepted protection that still exceeds the budget keeps the existing INFEASIBLE semantics.
    length, budget = 10, 4
    protected = {i: ("ev_k",) for i in range(6)}
    plan = plan_starlight_layer(
        layer=0, length=length, budget=budget, h2o_candidates=[],
        signals={"da": {}, "grounding": protected, "structural": {}}, enabled_signals=("grounding",),
    )
    assert plan.status == STATUS_INFEASIBLE
    assert len(plan.retained) >= 6  # protected entries are retained, the overflow is reported


def test_engine_telemetry_reports_the_keyword_policy_and_the_rejected_stopword():
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="c4", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4, evidence_pool=_pool(["the"], rid="c4"))
    finally:
        engine.close()
    policy = res.telemetry["starlight"]["grounding_keyword_policy"]
    assert policy["max_keyword_prompt_fraction"] == MAX_KEYWORD_PROMPT_FRACTION
    assert policy["rejected_stopword"] == ["the"]
    assert res.telemetry["starlight"]["grounding_protected_by_evidence"] == {}
