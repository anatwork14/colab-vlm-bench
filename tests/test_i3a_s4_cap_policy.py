"""ITER3 Round A S4 (cpu_pytorch_reference_not_vllm): the grounding keyword caps and the stopword list are
configurable, VERSIONED (cap_policy_version), traceable in telemetry, and every threshold records what it retained,
rejected and omitted. Nothing is truncated silently while perfect preservation is claimed.

Provisional policy (engineering hypothesis, not owner ratification): caps and stopwords are provisional,
configurable, versioned and ablatable. They were not tuned on confirmatory answers. The numbers below are
arithmetic over the synthetic probe prompts, not measurements.
"""
import hashlib

import pytest

from vlmbench.infrastructure import starlight as starlight_mod
from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.real_engine import RealSevenArmEngine
from vlmbench.infrastructure.starlight import KEYWORD_STOPWORDS

from test_starlight_oracle import PROMPT, _cfg, _WordTok

# Text-only fixtures with caller verified=True need the explicit policy to exercise the caps (S1 default is off).
ACCEPT_DECLARED = EvidencePolicy(accept_declared=True)
SHA = "d" * 64
THE, CAR, FILLER = 7, 8, 30
VOCAB = {"the": THE, "car": CAR, "cat": 9, "mat": 12}
UNION_WORDS = ["which", "these", "their", "not", "they", "she"]
UNION_VOCAB = {"which": 21, "these": 22, "their": 23, "not": 24, "they": 25, "she": 26}
UNION_BLOCKS = [(0, 7), (7, 14), (14, 21), (21, 28), (28, 35), (35, 40)]


class _Tok:
    def encode(self, text):
        return [VOCAB[w] for w in text.lower().split() if w in VOCAB]


class _UnionTok:
    def encode(self, text):
        return [UNION_VOCAB[w] for w in text.lower().split() if w in UNION_VOCAB]


def _pool(keywords, rid="s4"):
    pool = EvidencePool(rid)
    pool.add_evidence(EvidenceItem(evidence_id="ev_s4", modality="text_span", content=" ".join(keywords),
                                   source_sha256=SHA, verified=True), keywords=list(keywords))
    return pool


def _prompt(n, placed):
    ids = [FILLER] * n
    for tok, positions in placed.items():
        for p in positions:
            ids[p] = tok
    return ids


def _cap(version, max_kw, max_union, stopwords=KEYWORD_STOPWORDS):
    return starlight_mod.GroundingCapPolicy(
        cap_policy_version=version, max_keyword_prompt_fraction=max_kw,
        max_union_prompt_fraction=max_union, stopwords=frozenset(stopwords),
    )


def _protect(keywords, prompt, tok=None, cap=None):
    return starlight_mod.grounding_protection_with_report(
        _pool(keywords), prompt, (tok or _Tok()).encode, "s4", None, policy=ACCEPT_DECLARED, cap_policy=cap,
    )


# --- versioning and traceability ------------------------------------------------------------------

def test_default_cap_policy_is_versioned_and_its_stopword_set_is_traceable_by_hash():
    default = starlight_mod.DEFAULT_GROUNDING_CAP_POLICY
    assert default.cap_policy_version
    _prot, report = _protect(["car"], _prompt(20, {CAR: [3]}))
    assert report["cap_policy_version"] == default.cap_policy_version
    expected = hashlib.sha256("\n".join(sorted(KEYWORD_STOPWORDS)).encode("utf8")).hexdigest()
    assert report["cap_policy"]["stopword_set_sha256"] == expected
    assert report["cap_policy"]["max_keyword_prompt_fraction"] == starlight_mod.MAX_KEYWORD_PROMPT_FRACTION == 0.2
    assert report["cap_policy"]["max_union_prompt_fraction"] == starlight_mod.MAX_UNION_PROMPT_FRACTION == 0.5


def test_changing_a_cap_or_the_stopwords_without_a_new_version_id_is_refused():
    default_id = starlight_mod.DEFAULT_GROUNDING_CAP_POLICY.cap_policy_version
    # A changed threshold under the default version id would make the id lie about what ran.
    with pytest.raises(ValueError):
        starlight_mod.GroundingCapPolicy(
            cap_policy_version=default_id, max_keyword_prompt_fraction=0.3, max_union_prompt_fraction=0.5,
            stopwords=KEYWORD_STOPWORDS,
        )
    # Likewise a changed stopword list.
    with pytest.raises(ValueError):
        starlight_mod.GroundingCapPolicy(
            cap_policy_version=default_id, max_keyword_prompt_fraction=0.2, max_union_prompt_fraction=0.5,
            stopwords=frozenset({"car"}),
        )


def test_invalid_fractions_are_refused():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            starlight_mod.GroundingCapPolicy(cap_policy_version="S4_TEST_BAD", max_keyword_prompt_fraction=bad,
                                             max_union_prompt_fraction=None, stopwords=frozenset())


def test_a_new_version_id_makes_a_changed_policy_legal_and_is_reported():
    cap = _cap("S4_TEST_PER_KW_0.3", 0.3, 0.5)
    _prot, report = _protect(["car"], _prompt(20, {CAR: [3]}), cap=cap)
    assert report["cap_policy_version"] == "S4_TEST_PER_KW_0.3"
    assert report["max_keyword_prompt_fraction"] == 0.3


# --- several per-keyword thresholds, including the boundary --------------------------------------

def test_per_keyword_thresholds_retain_and_reject_at_each_boundary_with_a_ledger_per_threshold():
    # 'car' occupies 4 of 20 prompt positions = 0.2 exactly.
    prompt = _prompt(20, {CAR: [1, 5, 9, 13]})
    for limit, accepted in ((0.1, False), (0.19, False), (0.2, True), (0.3, True)):
        cap = _cap(f"S4_TEST_PER_KW_{limit}", limit, None)
        prot, report = _protect(["car"], prompt, cap=cap)
        ledger = report["threshold_ledger"]["per_keyword_fraction"]
        assert ledger["limit"] == limit
        if accepted:
            assert set(prot) == {1, 5, 9, 13}, limit
            assert report["accepted_keywords"] == ["car"], limit
            assert ledger["rejected"] == [] and ledger["omitted_positions"] == 0, limit
        else:
            assert prot == {}, limit
            assert report["accepted_keywords"] == [], limit
            assert ledger["rejected"] == [{"keyword": "car", "protected_fraction": 0.2}], limit
            assert ledger["omitted_positions"] == 4, limit


# --- the uncapped case ----------------------------------------------------------------------------

def test_the_uncapped_case_protects_every_occurrence_and_records_no_rejection():
    prompt = _prompt(20, {CAR: list(range(15))})  # 15 of 20 = 0.75, which a 0.2 cap would reject
    cap = _cap("S4_TEST_UNCAPPED", None, None, stopwords=frozenset())
    prot, report = _protect(["car"], prompt, cap=cap)
    assert set(prot) == set(range(15))
    assert report["rejected_breadth"] == [] and report["rejected_union"] == []
    assert report["threshold_ledger"]["per_keyword_fraction"]["limit"] is None
    assert report["threshold_ledger"]["union_fraction"]["limit"] is None
    assert report["omitted_positions_total"] == 0
    assert report["perfect_preservation_claimed"] is True
    assert report["preservation_claim"] == "PERFECT_PRESERVATION_OF_KEYWORD_OCCURRENCES"


def test_capped_omission_is_never_described_as_perfect_preservation():
    prompt = _prompt(10, {CAR: [1, 4, 7]})  # 3 of 10 = 0.3 > 0.2: all three occurrences are omitted
    prot, report = _protect(["car"], prompt)
    assert prot == {}
    assert report["omitted_positions_total"] == 3
    assert report["perfect_preservation_claimed"] is False
    assert report["preservation_claim"] == "NOT_PERFECT_PRESERVATION_CAPPED_OCCURRENCES_OMITTED"


# --- several union thresholds on the six-keyword probe (40 positions, declared order) ------------

def _probe_prompt():
    prompt = [FILLER] * 40
    for (a, b), w in zip(UNION_BLOCKS, UNION_WORDS):
        for pos in range(a, b):
            prompt[pos] = UNION_VOCAB[w]
    return prompt


def test_union_thresholds_are_each_evaluated_in_declared_order_with_their_own_ledger():
    cases = (
        # limit, accepted in declared order, rejected in declared order, positions omitted by the union cap
        (0.4, ["which", "these"], ["their", "not", "they", "she"], 26),
        (0.6, ["which", "these", "their"], ["not", "they", "she"], 19),
        (0.9, ["which", "these", "their", "not", "they"], ["she"], 5),
    )
    for limit, accepted, rejected, omitted in cases:
        cap = _cap(f"S4_TEST_UNION_{limit}", 0.2, limit)
        prot, report = starlight_mod.grounding_protection_with_report(
            _pool(UNION_WORDS), _probe_prompt(), _UnionTok().encode, "s4", None,
            policy=ACCEPT_DECLARED, cap_policy=cap,
        )
        ledger = report["threshold_ledger"]["union_fraction"]
        assert report["accepted_keywords"] == accepted, limit
        assert [r["keyword"] for r in ledger["rejected"]] == rejected, limit
        assert ledger["omitted_positions"] == omitted, limit
        assert len(prot) / 40 <= limit, limit


def test_the_default_union_ledger_records_the_three_rejections_and_their_21_omitted_positions():
    _prot, report = starlight_mod.grounding_protection_with_report(
        _pool(UNION_WORDS), _probe_prompt(), _UnionTok().encode, "s4", None, policy=ACCEPT_DECLARED,
    )
    ledger = report["threshold_ledger"]["union_fraction"]
    assert report["accepted_keywords"] == ["which", "these", "she"]
    assert ledger["retained"] == ["which", "these", "she"]
    assert [r["keyword"] for r in ledger["rejected"]] == ["their", "not", "they"]
    assert ledger["omitted_positions"] == 21
    assert report["omitted_positions_total"] == 21
    assert report["preservation_claim"] == "NOT_PERFECT_PRESERVATION_CAPPED_OCCURRENCES_OMITTED"


def test_an_uncapped_union_accepts_all_six_probe_keywords():
    cap = _cap("S4_TEST_UNION_UNCAPPED", 0.2, None)
    prot, report = starlight_mod.grounding_protection_with_report(
        _pool(UNION_WORDS), _probe_prompt(), _UnionTok().encode, "s4", None, policy=ACCEPT_DECLARED, cap_policy=cap,
    )
    assert report["accepted_keywords"] == UNION_WORDS
    assert report["rejected_union"] == []
    assert len(prot) == 40


# --- configurable, versioned stopword list ----------------------------------------------------------

def test_the_stopword_list_is_configurable_and_changes_the_outcome_in_both_directions():
    prompt = _prompt(40, {THE: [3], CAR: [10, 20]})
    # Default list: 'the' is a stopword (rejected, its one occurrence omitted); 'car' protects its two.
    prot_default, rep_default = _protect(["the", "car"], prompt)
    assert set(prot_default) == {10, 20}
    assert rep_default["rejected_stopword"] == ["the"]
    # Custom list without 'the' but with 'car': 'the' protects its one occurrence; 'car' is the stopword.
    cap = _cap("S4_TEST_STOPWORDS_CAR", 0.2, 0.5, stopwords={"car"})
    prot_custom, rep_custom = _protect(["the", "car"], prompt, cap=cap)
    assert set(prot_custom) == {3}
    assert rep_custom["rejected_stopword"] == ["car"]
    assert rep_custom["threshold_ledger"]["stopword"]["omitted_positions"] == 2
    assert rep_custom["cap_policy_version"] == "S4_TEST_STOPWORDS_CAR"


def test_a_different_stopword_list_has_a_different_traceable_hash():
    _p, default_rep = _protect(["car"], _prompt(20, {CAR: [3]}))
    _p, custom_rep = _protect(["car"], _prompt(20, {CAR: [3]}), cap=_cap("S4_TEST_SW", 0.2, 0.5, {"car"}))
    assert default_rep["cap_policy"]["stopword_set_sha256"] != custom_rep["cap_policy"]["stopword_set_sha256"]


# --- engine level: the policy is passed through and reported --------------------------------------

def test_engine_telemetry_reports_the_configured_cap_policy_version_and_ledgers():
    uncapped = _cap("S4_TEST_ENGINE_UNCAPPED", None, None)
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok(),
                                evidence_policy=ACCEPT_DECLARED, grounding_cap_policy=uncapped)
    try:
        res = engine.execute_request(request_id="s4e", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=4, evidence_pool=_pool(["car"], rid="s4e"))
    finally:
        engine.close()
    kw = res.telemetry["starlight"]["grounding_keyword_policy"]
    assert kw["cap_policy_version"] == "S4_TEST_ENGINE_UNCAPPED"
    assert kw["threshold_ledger"]["per_keyword_fraction"]["limit"] is None
    assert kw["omitted_positions_total"] == 0
    assert kw["perfect_preservation_claimed"] is True


def test_engine_default_reports_the_default_cap_policy_version():
    engine = RealSevenArmEngine(method_arm="starlight_s1", device="cpu", config=_cfg(), stop_at_eos=False,
                                h2o_min_budget=16, h2o_budget_ratio=0.2, tokenizer=_WordTok())
    try:
        res = engine.execute_request(request_id="s4d", prompt_token_ids=PROMPT, image_present=False,
                                     max_new_tokens=2)
    finally:
        engine.close()
    kw = res.telemetry["starlight"]["grounding_keyword_policy"]
    assert kw["cap_policy_version"] == starlight_mod.DEFAULT_GROUNDING_CAP_POLICY.cap_policy_version
