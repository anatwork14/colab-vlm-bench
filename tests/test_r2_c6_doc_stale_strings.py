"""ITER3 R2 C-6 (closure doc, MEDIUM docs): stale statements that contradict the code must not return.

Checked in the four documents the closure doc names. Two stale statement classes:
  * A-2 described as a ValueError. The code returns STATUS_INFEASIBLE with INFEASIBLE_UNMET_AFTER_ELIGIBLE
    when the budget is unmet after every eligible candidate is evicted. ValueError is only the caller-error
    guard (candidates supplied although nothing needs eviction).
  * vision_start/vision_end (structural tokens) described as NOT protected. The code protects them through the
    structural source (lead decision Q2, PROTECTED_LEAD_DECISION_Q2). Owner ratification of A-4 stays open.
Also: the retired label the old seven-arm pretrained label may appear only marked as retired or superseded.
Documentation check only. No model, no inference.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "docs" / "harness" / "STARLIGHT_STAGE1_DESIGN.md",
    ROOT / "docs" / "harness" / "STARLIGHT_STAGE1_REPORT.md",
    ROOT / "docs" / "MTP_DA_UPSTREAM_CONTRACT_AUDIT.md",
    ROOT / "docs" / "harness" / "CLAIM_RECONCILIATION.md",
]
STALE = {
    "A-2 described as a ValueError (code returns INFEASIBLE_UNMET_AFTER_ELIGIBLE)":
        re.compile(r"(unmet|overflow|budget)[^\n]*raises ValueError", re.I),
    "structural/vision tokens described as NOT protected (code protects, lead decision Q2)":
        re.compile(r"(vision_start|structural (vision )?tokens|vision span)[^\n]*not protected", re.I),
}


def _lines(doc):
    return list(enumerate(doc.read_text(encoding="utf-8").splitlines(), 1))


def test_no_stale_statement_contradicts_the_code_in_the_cited_docs():
    hits = [f"{doc.name}:{n} {label}"
            for doc in DOCS for n, line in _lines(doc)
            for label, pat in STALE.items() if pat.search(line)]
    assert hits == [], hits


def test_the_retired_label_appears_only_marked_as_retired_or_superseded():
    unmarked = [f"{doc.name}:{n}" for doc in DOCS for n, line in _lines(doc)
                if ("SEVEN_ARM_" + "PRETRAINED_PASS") in line and not re.search(r"retired|superseded", line, re.I)]
    assert unmarked == [], unmarked


def test_the_design_doc_states_the_structural_protection_it_implements():
    text = (ROOT / "docs" / "harness" / "STARLIGHT_STAGE1_DESIGN.md").read_text(encoding="utf-8")
    assert "PROTECTED_LEAD_DECISION_Q2" in text or "lead decision Q2" in text
