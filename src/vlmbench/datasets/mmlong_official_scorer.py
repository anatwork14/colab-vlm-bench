"""Native MMLongBench-Doc scorer, ported from the official evaluation code.

PROVENANCE (fetched read-only 2026-11-10, not owner-approved):
  upstream   https://github.com/mayubo2333/MMLongBench-Doc
  commit     d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384 (main)
  licence    Apache-2.0
  eval/eval_score.py       sha256 cafa268358b08521cfc9a805c67d86732d943677e4408c5751f336123d09631f
  eval/extract_answer.py   sha256 fa6a747bd6a8e96fb6884eb7201663268a5df700ad71d85db3bc5e04a157722f
  eval/prompt_for_answer_extraction.md  sha256 96a22c3003c22801f1f19717be45070b1d6b9ce76b42e171e044f63b641de659

WHAT IS PORTED: the deterministic scoring stage (`eval_score`, `eval_acc_and_f1`, and the
breakdowns of `show_results`). Functions keep the official logic, including its quirks.

Deviations, none of which changes a score on well-formed inputs:
  * `eval(...)` on list-looking strings is replaced by `ast.literal_eval`.
  * The official `print` debugging calls and file writing are removed; `show_results` returns a dict.

WHAT IS NOT PORTED (official three-stage protocol, README section 4.1):
  Stage 1 is the model's long-form response, which this module does not produce.
  Stage 2 is an LLM extraction (`extract_answer` calls GPT-4o with the prompt above). It is
  a model step, so it is an interface only. Without its output this module reports
  BLOCKED_NO_JUDGE and never a heuristic score labelled as official.
  Stage 3 is the deterministic scoring in this file.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from math import isclose
from typing import Any, Iterable, Sequence

UPSTREAM_REPO = "https://github.com/mayubo2333/MMLongBench-Doc"
UPSTREAM_COMMIT = "d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384"
EVAL_SCORE_SHA256 = "cafa268358b08521cfc9a805c67d86732d943677e4408c5751f336123d09631f"
EXTRACT_ANSWER_SHA256 = "fa6a747bd6a8e96fb6884eb7201663268a5df700ad71d85db3bc5e04a157722f"
EXTRACTION_PROMPT_SHA256 = "96a22c3003c22801f1f19717be45070b1d6b9ce76b42e171e044f63b641de659"
OFFICIAL_EXTRACTOR_MODEL = "gpt-4o"  # upstream eval/extract_answer.py at d73f0dc: model_name="gpt-4o"
NOT_ANSWERABLE = "Not answerable"


# --- ported from eval/eval_score.py -------------------------------------------------

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if "." in str(gt_ans):
            precision = len(str(gt_ans).split(".")[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except Exception:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith("mile"):
        s.rstrip("mile").strip()  # official no-op (result discarded); kept for fidelity
    if s.endswith("miles"):
        s.rstrip("miles").strip()  # official no-op; kept for fidelity
    if s.endswith("million"):
        s.rstrip("million").strip()  # official no-op; kept for fidelity
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    if "https://" in s:
        flag = True
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    if re.fullmatch(r"\b\d+(-\d+|\s\d+)?\b", s):
        flag = True
    if "a.m." in s or "p.m." in s:
        flag = True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}[-\s]\d{2}\b", s):
        flag = True
    if re.fullmatch(r"\b\d{4}[-\s]\d{2}\b", s):
        flag = True
    if re.fullmatch(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def _as_list(value):
    if isinstance(value, str) and value.startswith("["):
        value = ast.literal_eval(value)
    if not isinstance(value, list):
        value = [value]
    return value


def eval_score(gt, pred, answer_type) -> float:
    if answer_type == "Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except Exception:
            pred = ""
        score = (gt == pred)
    elif answer_type == "Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except Exception:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)
    else:
        gt = _as_list(gt)
        pred = _as_list(pred)
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score = ("-".join(gt) == "-".join(pred))
            else:
                score = min([anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])

    return float(score)


# --- ported from eval/eval_score.py: eval_acc_and_f1 and show_results breakdowns ----

def eval_acc_and_f1(samples: Sequence[dict]) -> tuple[float, float]:
    evaluated = [s for s in samples if "score" in s]
    if not evaluated:
        return 0.0, 0.0

    acc = sum(s["score"] for s in evaluated) / len(evaluated)
    try:
        answerable = [s for s in evaluated if s["answer"] != NOT_ANSWERABLE]
        predicted = [s for s in evaluated if s["pred"] != NOT_ANSWERABLE]
        recall = sum(s["score"] for s in answerable) / len(answerable)
        precision = sum(s["score"] for s in answerable) / len(predicted)
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0.0 else 0.0
    except Exception:
        f1 = 0.0
    return acc, f1


def _pages(sample: dict) -> list:
    value = sample["evidence_pages"]
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


def _sources(sample: dict) -> list:
    value = sample["evidence_sources"]
    return ast.literal_eval(value) if isinstance(value, str) else list(value)


def breakdowns(samples: Sequence[dict]) -> dict[str, Any]:
    """The categories written by the official show_results(), returned as data."""
    overall_acc, overall_f1 = eval_acc_and_f1(samples)
    single = [s for s in samples if len(_pages(s)) == 1]
    cross = [s for s in samples if len(_pages(s)) != 1 and s["answer"] != NOT_ANSWERABLE]
    neg = [s for s in samples if s["answer"] == NOT_ANSWERABLE]

    by_source: dict[str, list] = defaultdict(list)
    by_doc_type: dict[str, list] = defaultdict(list)
    for s in samples:
        for src in _sources(s):
            by_source[src].append(s)
        by_doc_type[s["doc_type"]].append(s)

    return {
        "overall_acc": overall_acc,
        "overall_f1": overall_f1,
        "question_number": len(samples),
        "single_page": {"acc": eval_acc_and_f1(single)[0], "n": len(single)},
        "cross_page": {"acc": eval_acc_and_f1(cross)[0], "n": len(cross)},
        "unanswerable": {"acc": eval_acc_and_f1(neg)[0], "n": len(neg)},
        "evidence_sources": {k: {"acc": eval_acc_and_f1(v)[0], "n": len(v)} for k, v in by_source.items()},
        "document_types": {k: {"acc": eval_acc_and_f1(v)[0], "n": len(v)} for k, v in by_doc_type.items()},
    }


# --- stage 2 interface (LLM extraction) and the blocked status ----------------------

@dataclass(frozen=True)
class ExtractedAnswer:
    """A short answer produced by the official LLM extraction stage (stage 2).

    Only values with a named extractor model and the prompt hash may be scored as official.
    """

    question_id: str
    pred: str
    extractor_model: str
    extractor_prompt_sha256: str

    def __post_init__(self):
        if not self.extractor_model or not self.extractor_prompt_sha256:
            raise ValueError(
                "extractor model and prompt hash are required: official stage 2 is an LLM step"
            )


def is_official_extraction(ext: ExtractedAnswer) -> bool:
    """True only for the official extractor model and the official extraction prompt hash."""
    return ext.extractor_model == OFFICIAL_EXTRACTOR_MODEL and ext.extractor_prompt_sha256 == EXTRACTION_PROMPT_SHA256


def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "BLOCKED_NO_JUDGE",
        "reason": reason,
        "official_acc": None,
        "official_f1": None,
        "deterministic_stage_ported": True,
    }
    out.update(extra)
    return out


def official_status(extracted: Iterable[ExtractedAnswer] | None) -> dict[str, Any]:
    """Without stage-2 output there is no official number: the status is BLOCKED_NO_JUDGE."""
    if extracted is None:
        return _blocked("official stage 2 (LLM answer extraction, eval/extract_answer.py) has not been run",
                        reason_code="NO_EXTRACTION")
    raise NotImplementedError("scoring of extracted answers is applied via score_records()")


def score_records(records: Sequence[dict], extracted: dict[str, ExtractedAnswer]) -> dict[str, Any]:
    """Stage 3 over records and stage-2 output. The official label needs official extraction for every
    question. Any other extraction, or a missing one, is BLOCKED_NO_JUDGE with no numbers.

    Limit: the scorer cannot verify that the extractor ran. It checks the model name and prompt hash the
    caller declares, so OFFICIAL_SCORED_STAGE3_ONLY is only as strong as the caller's record of stage 2.
    """
    missing = [rec["question_id"] for rec in records if rec["question_id"] not in extracted]
    if missing:
        return _blocked("no stage-2 extraction for some questions", missing_question_ids=missing)
    refused = [qid for qid in (r["question_id"] for r in records) if not is_official_extraction(extracted[qid])]
    if refused:
        return _blocked("extraction is not from the official extractor model and prompt hash",
                        refused_question_ids=refused, official_extractor_model=OFFICIAL_EXTRACTOR_MODEL,
                        official_prompt_sha256=EXTRACTION_PROMPT_SHA256)
    scored = []
    for rec in records:
        ext = extracted[rec["question_id"]]
        item = dict(rec)
        item["pred"] = ext.pred
        item["score"] = eval_score(rec["answer"], ext.pred, rec["answer_format"])
        scored.append(item)
    out = breakdowns(scored)
    out.update({
        "status": "OFFICIAL_SCORED_STAGE3_ONLY",
        "extractor_models": sorted({e.extractor_model for e in extracted.values()}),
        "extractor_prompt_sha256": sorted({e.extractor_prompt_sha256 for e in extracted.values()}),
        "upstream_commit": UPSTREAM_COMMIT,
    })
    return out
