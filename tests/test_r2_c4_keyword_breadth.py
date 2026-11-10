"""ITER3 R2 C-4 (closure doc): bound the keyword breadth of grounding protection.

Policy (reported explicitly in telemetry):
  * stop-word-like keywords are rejected (e.g. 'the'), whatever their frequency;
  * a keyword whose protected positions exceed MAX_KEYWORD_PROMPT_FRACTION of the prompt is rejected;
  * if the accepted protection still exceeds the budget, the existing INFEASIBLE semantics apply
    (protected entries are retained, the overflow is reported, nothing is silently dropped).
cpu_pytorch_reference_not_vllm.
"""
import pytest

from vlmbench.infrastructure import starlight as starlight_mod
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


# ITER3 R3 (N-R2-2, lead ruling): the protected UNION of accepted keywords is bounded too. The default cap is
# starlight_mod.MAX_UNION_PROMPT_FRACTION = 0.5 of prompt positions (owner-ratifiable named constant). Keywords are
# processed in DECLARED order; a keyword whose addition would push the union over the cap is rejected and listed,
# and later keywords are still considered. Probe: six keywords, each 7/40 (under the per-keyword cap), together 40/40.
UNION_WORDS = ["which", "these", "their", "not", "they", "she"]
UNION_VOCAB = {"which": 21, "these": 22, "their": 23, "not": 24, "they": 25, "she": 26}
UNION_BLOCKS = [(0, 7), (7, 14), (14, 21), (21, 28), (28, 35), (35, 40)]


class _UnionTok:
    def encode(self, text):
        return [UNION_VOCAB[w] for w in text.lower().split() if w in UNION_VOCAB]


def _probe_prompt():
    prompt = [FILLER] * 40
    for (a, b), w in zip(UNION_BLOCKS, UNION_WORDS):
        for pos in range(a, b):
            prompt[pos] = UNION_VOCAB[w]
    return prompt


def _declared_pool(words, eid="ev_u"):
    pool = EvidencePool("c4u")
    pool.add_evidence(EvidenceItem(evidence_id=eid, modality="text_span", content=" ".join(words),
                                   source_sha256=SHA, verified=True), keywords=list(words))
    return pool


def _union_protect(words, prompt):
    return grounding_protection_with_report(_declared_pool(words), prompt, _UnionTok().encode, "c4u", None)


def test_the_union_of_accepted_keywords_is_bounded_by_the_policy():
    # Promoted from the ITER3 R2 strict xfail; the test was not weakened. RE-DERIVED with stated reasons:
    #  (1) the 0.2 bound was per keyword. Each accepted keyword is still checked against it (7/40 = 0.175).
    #  (2) the union bound is now the policy's MAX_UNION_PROMPT_FRACTION (0.5): six keywords together protect
    #      40/40 and cannot fit it, so the original 'all six accepted' expectation is replaced by the exact
    #      outcome: the first two in declared order (14/40) are accepted, the other four are rejected for union.
    prompt = _probe_prompt()
    protected, report = _union_protect(UNION_WORDS, prompt)
    for w in report["accepted_keywords"]:
        own, _ = _union_protect([w], prompt)
        assert len(own) / len(prompt) <= MAX_KEYWORD_PROMPT_FRACTION, w
    assert report["accepted_keywords"] == UNION_WORDS[:2]
    assert len(protected) / len(prompt) <= starlight_mod.MAX_UNION_PROMPT_FRACTION
    assert [r["keyword"] for r in report["rejected_union"]] == UNION_WORDS[2:]


def test_the_union_cap_is_the_named_half_of_the_prompt_default():
    assert starlight_mod.MAX_UNION_PROMPT_FRACTION == 0.5


def test_the_six_keyword_probe_reports_union_fraction_and_the_rejections():
    prompt = _probe_prompt()
    _protected, report = _union_protect(UNION_WORDS, prompt)
    assert report["max_union_prompt_fraction"] == 0.5
    assert report["union_fraction"] == 14 / 40
    assert report["rejected_union"] == [
        {"keyword": "their", "union_fraction_if_added": 21 / 40},
        {"keyword": "not", "union_fraction_if_added": 21 / 40},
        {"keyword": "they", "union_fraction_if_added": 21 / 40},
        {"keyword": "she", "union_fraction_if_added": 21 / 40},
    ]


def test_union_is_evaluated_in_declared_order_not_alphabetically():
    # Declared order decides which keyword fits. Reversed declaration accepts a different pair.
    prompt = _probe_prompt()
    _protected, report = _union_protect(list(reversed(UNION_WORDS)), prompt)
    assert report["accepted_keywords"] == ["she", "they"]


def test_a_keyword_that_would_exceed_the_union_cap_does_not_block_later_smaller_keywords():
    # A: 6 positions; B: 15 positions (union would be 21/40 = 0.525, rejected); C: 3 positions (union 9/40, accepted).
    words = ["alpha", "beta", "gamma"]
    vocab = {"alpha": 41, "beta": 42, "gamma": 43}
    prompt = [FILLER] * 40
    for pos in range(0, 6):
        prompt[pos] = 41
    for pos in range(6, 21):
        prompt[pos] = 42
    for pos in range(21, 24):
        prompt[pos] = 43

    class _T:
        def encode(self, text):
            return [vocab[w] for w in text.lower().split() if w in vocab]

    pool = _declared_pool(words)
    protected, report = grounding_protection_with_report(pool, prompt, _T().encode, "c4u", None)
    assert report["accepted_keywords"] == ["alpha", "gamma"]
    assert [r["keyword"] for r in report["rejected_union"]] == ["beta"]
    assert len(protected) / 40 == 9 / 40


def test_a_union_exactly_at_the_cap_is_accepted():
    # 20 of 40 positions = 0.5 exactly: at the cap is accepted, consistent with the per-keyword rule.
    prompt = [FILLER] * 40
    vocab = {"alpha": 41}
    for pos in range(0, 20):
        prompt[pos] = 41

    class _T:
        def encode(self, text):
            return [vocab[w] for w in text.lower().split() if w in vocab]

    protected, report = grounding_protection_with_report(_declared_pool(["alpha"]), prompt, _T().encode, "c4u", None)
    assert len(protected) / 40 == 0.5
    assert report["accepted_keywords"] == ["alpha"] and report["rejected_union"] == []


def test_engine_telemetry_reports_union_fraction_and_empty_union_rejections_when_under_cap():
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="c4e", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4, evidence_pool=_pool(["car", "cat"], rid="c4e"))
    finally:
        engine.close()
    policy = res.telemetry["starlight"]["grounding_keyword_policy"]
    assert policy["union_fraction"] == 2 / len(PROMPT)
    assert policy["rejected_union"] == []
    assert policy["max_union_prompt_fraction"] == 0.5
