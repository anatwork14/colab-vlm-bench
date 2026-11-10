"""ITER2 D4 auditor fixes S-5, S-6, S-7 (cpu_pytorch_reference_not_vllm).

These are regression tests for behaviour the policy already implements. Each has a documented
mutation that must make it fail (checked when added; see the commit message).
"""
import hashlib

from vlmbench.infrastructure.evidence_verification import EvidencePolicy
from vlmbench.infrastructure.grounding_evidence import EvidenceItem, EvidencePool
from vlmbench.infrastructure.starlight import (
    DECISION_EVICT,
    STATUS_OK,
    EntryDecision,
    StarlightLayerPlan,
    grounding_protection_from_pool,
    summarize_plans,
)

# S1 (evidence policy): these fixtures are text-only with caller verified=True. They test keyword-form
# protection, so they opt in explicitly with accept_declared=True (reported, ablatable). The unverified (caller
# verified=False) case is refused under either policy and is kept on the default.
ACCEPT_DECLARED = EvidencePolicy(accept_declared=True)

SHA = "ab" * 32


def _pool(keyword, verified=True, request_id="req_s6"):
    pool = EvidencePool(request_id)
    pool.add_evidence(
        EvidenceItem(evidence_id="ev_1", modality="text_span", content=keyword,
                     source_sha256=SHA, verified=verified),
        keywords=[keyword],
    )
    return pool


# --- S-5: summary invariant counters on a hand-built violating plan ------------------------

def test_s5_summary_counts_protected_evict_and_evicted_not_candidate_as_one_each():
    # Entry 0: protected AND evicted (a protected-evict violation).
    protected_evict = EntryDecision(
        entry_id=0, layer=3, h2o_candidate=True, internally_protected=True,
        signals_fired=("da:sink_global",), decision=DECISION_EVICT,
        reason="forced", reason_code="H2O_CANDIDATE_UNPROTECTED",
    )
    # Entry 1: evicted but NOT an H2O candidate (an evicted-not-candidate violation).
    not_candidate = EntryDecision(
        entry_id=1, layer=3, h2o_candidate=False, internally_protected=False,
        signals_fired=(), decision=DECISION_EVICT,
        reason="forced", reason_code="H2O_CANDIDATE_UNPROTECTED",
    )
    plan = StarlightLayerPlan(
        layer=3, length=2, budget=1, status=STATUS_OK, retained=(),
        evicted=(0, 1), decisions=(protected_evict, not_candidate), h2o_candidates_vetoed=0,
    )
    summary = summarize_plans([plan])
    assert summary["protected_evicted"] == 1
    assert summary["evicted_not_candidate"] == 1


# --- S-6: unverified evidence protects nothing ---------------------------------------------

class _StubItem:
    # Mirrors EvidenceItem's fields and content_digest() (S1 reads both to assess the item).
    def __init__(self, evidence_id, verified, source_sha256, content=""):
        self.evidence_id = evidence_id
        self.modality = "text_span"
        self.content = content
        self.verified = verified
        self.source_sha256 = source_sha256

    def content_digest(self):
        return hashlib.sha256(self.content.encode("utf8")).hexdigest()


class _StubPool:
    """Pool-shaped object. EvidencePool itself refuses verified=False at insertion, so an
    unverified item can only reach the policy through a pool that carries one (e.g. a loaded
    artifact). The policy must still protect nothing for it."""

    def __init__(self, keyword, verified):
        self.request_id = "req_s6"
        self._items = [_StubItem("ev_1", verified, SHA, content=keyword)]
        self._keywords = {keyword: ["ev_1"]}

    def items(self):
        return list(self._items)

    def keyword_index(self):
        return {k: list(v) for k, v in self._keywords.items()}


def test_s6_unverified_evidence_protects_nothing_and_verified_control_protects():
    table = {"paris": [9]}

    def _encode(text):
        return list(table.get(text, []))

    # ITER3 R2 C-4 RE-DERIVED: 'paris' once in 3 tokens is 1/3 > 0.2 (breadth cap); five filler tokens make it 1/8.
    prompt = [7, 9, 1, 2, 3, 4, 5, 6]
    unverified = grounding_protection_from_pool(
        _StubPool("paris", verified=False), prompt, _encode, "req_s6", None)
    assert unverified == {}, "unverified evidence must not protect any prompt position"

    verified = grounding_protection_from_pool(
        _StubPool("paris", verified=True), prompt, _encode, "req_s6", None, policy=ACCEPT_DECLARED)
    assert verified == {1: ("ev_1",)}, "positive control: the same keyword protects when verified"


# --- S-7: tokenizer-variant forms of a grounding keyword are all protected -----------------

def test_s7_bare_space_prefixed_and_capitalised_keyword_forms_are_all_protected():
    # Each surface form tokenizes to a DIFFERENT id, as real BPE tokenizers do.
    table = {
        "paris": [9],        # bare
        " paris": [42],      # space-prefixed
        "Paris": [43],       # capitalised
        " Paris": [44],      # space-prefixed and capitalised
    }

    def _encode(text):
        return list(table.get(text, []))

    # ITER3 R2 C-4 RE-DERIVED: four protected forms in 9 tokens is 4/9 > 0.2 (breadth cap). Twelve filler
    # tokens (ids outside the table) are appended, so the four positions 1,3,5,7 are unchanged and 4/21 = 0.19.
    prompt = [7, 9, 1, 42, 2, 43, 3, 44, 5] + list(range(100, 112))
    protected = grounding_protection_from_pool(
        _pool("paris"), prompt, _encode, "req_s6", None, policy=ACCEPT_DECLARED)
    # Positions of 9 (bare), 42 (space), 43 (capitalised), 44 (space+capitalised).
    assert set(protected) == {1, 3, 5, 7}, (
        "under-protection: a keyword form in the prompt was not protected")
    assert all(labels == ("ev_1",) for labels in protected.values())
