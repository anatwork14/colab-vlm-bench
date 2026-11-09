"""Test-only transcription of the official MMLongBench-Doc scoring functions (no network, not executed upstream).

Transcribed by reading eval/eval_score.py at commit d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384
(sha256 cafa268358b08521cfc9a805c67d86732d943677e4408c5751f336123d09631f), Apache-2.0.
Logic is copied line for line. Two changes, neither of which changes a result on the inputs tested:
  * the official print() debugging calls are removed;
  * eval() on list-looking strings is replaced by ast.literal_eval (the port's documented deviation).
This is a second copy of the same logic, so agreement shows the port matches the transcription,
not that the transcription is correct. The hand-checked cases in test_mmlong_fuzz_vs_reference.py are the
independent check.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from math import isclose


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
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
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
        s.rstrip("mile").strip()
    if s.endswith("miles"):
        s.rstrip("miles").strip()
    if s.endswith("million"):
        s.rstrip("million").strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
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
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    if "a.m." in s or "p.m." in s:
        flag = True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_score(gt, pred, answer_type):
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
        if isinstance(gt, str) and gt.startswith("["):
            gt = ast.literal_eval(gt)
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            pred = ast.literal_eval(pred)
        if not isinstance(pred, list):
            pred = [pred]
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


def eval_acc_and_f1(samples):
    evaluated_samples = [sample for sample in samples if "score" in sample]
    if not evaluated_samples:
        return 0.0, 0.0

    acc = sum([sample["score"] for sample in evaluated_samples]) / len(evaluated_samples)
    try:
        recall = sum([sample["score"] for sample in evaluated_samples if sample["answer"] != "Not answerable"]) / len([sample for sample in evaluated_samples if sample["answer"] != "Not answerable"])
        precision = sum([sample["score"] for sample in evaluated_samples if sample["answer"] != "Not answerable"]) / len([sample for sample in evaluated_samples if sample["pred"] != "Not answerable"])
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0.0 else 0.0
    except Exception:
        f1 = 0.0

    return acc, f1
