"""ITER2 D5 (2): native MMLongBench-Doc scorer, ported from the official eval code.

Source: mayubo2333/MMLongBench-Doc @ d73f0dc0be7e0a2ff6a403d5fe65fcd96461f384 (Apache-2.0),
eval/eval_score.py sha256 cafa268358b08521cfc9a805c67d86732d943677e4408c5751f336123d09631f.
Expected values below are derived by hand from that code's own logic (see comments).
The LLM answer-extraction stage is not implemented here: without its output no official
score is produced (status BLOCKED_NO_JUDGE).
"""
from __future__ import annotations

import pytest

from vlmbench.datasets import mmlong_official_scorer as ms


def test_levenshtein_matches_classic_example():
    assert ms.levenshtein_distance("kitten", "sitting") == 3


def test_anls_threshold_zeroes_scores_at_or_below_half():
    # dist 1 over length 4 -> anls 0.75 (kept); dist 4 over length 4 -> anls 0.0 (zeroed)
    assert ms.anls_compute("abcd", "abce") == pytest.approx(0.75)
    assert ms.anls_compute("abcd", "wxyz") == 0.0


def test_int_answers_compare_after_int_coercion():
    assert ms.eval_score("10", "10.0", "Int") == 1.0
    assert ms.eval_score("10", "11", "Int") == 0.0
    assert ms.eval_score("10", "ten", "Int") == 0.0  # official: unparseable prediction -> ""


def test_float_answers_accept_percent_forms():
    assert ms.eval_score("12.5%", "12.5", "Float") == 1.0
    assert ms.eval_score("50", "0.5", "Float") == 1.0  # 50 also matches 50/100 = 0.5 (include_percentage)
    assert ms.eval_score("12.5", "13.0", "Float") == 0.0


def test_str_answers_use_exact_match_for_exact_types_and_anls_otherwise():
    assert ms.eval_score("Yes", "yes", "Str") == 1.0
    assert ms.eval_score("Yes", "no", "Str") == 0.0  # anls 0 (distance 3 over length 3)
    assert ms.eval_score("cats", "cat", "Str") == pytest.approx(0.75)  # distance 1 over length 4
    assert ms.eval_score("https://a.com", "https://a.com/", "Str") == 0.0  # exact-match type
    assert ms.eval_score("page 3", "page 3", "Str") == 1.0


def test_list_answers_are_order_insensitive_and_length_strict():
    assert ms.eval_score("['apple', 'banana']", ["banana", "apple"], "List") == 1.0
    assert ms.eval_score("['a', 'b']", ["a"], "List") == 0.0
    assert ms.eval_score("['1', '2']", ["2", "1"], "List") == 1.0  # numeric branch joins sorted values


def test_acc_and_f1_match_hand_computation_from_official_definitions():
    samples = [
        {"answer": "Not answerable", "pred": "Not answerable", "score": 1.0},
        {"answer": "x", "pred": "Not answerable", "score": 0.0},
        {"answer": "y", "pred": "y", "score": 1.0},
    ]
    acc, f1 = ms.eval_acc_and_f1(samples)
    # acc = (1+0+1)/3; recall = (0+1)/2 over answerable; precision = (0+1)/1 over non-"Not answerable" preds
    assert acc == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def test_unanswerable_only_run_gives_zero_f1_not_a_crash():
    acc, f1 = ms.eval_acc_and_f1([{"answer": "Not answerable", "pred": "Not answerable", "score": 1.0}])
    assert acc == 1.0
    assert f1 == 0.0


def test_no_official_score_without_extraction_output():
    status = ms.official_status(extracted=None)
    assert status["status"] == "BLOCKED_NO_JUDGE"
    assert status["official_acc"] is None and status["official_f1"] is None


def test_extracted_answers_must_record_extractor_and_prompt_hash():
    with pytest.raises(ValueError, match="extractor"):
        ms.ExtractedAnswer(question_id="q1", pred="10", extractor_model="", extractor_prompt_sha256="")
    ok = ms.ExtractedAnswer(question_id="q1", pred="10", extractor_model="gpt-4o", extractor_prompt_sha256="a" * 64)
    assert ok.pred == "10"
