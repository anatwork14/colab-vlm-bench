"""ITER2 N-4 (REVIEW_ITER2B §3): differential fuzz and hand-checked cases for the scorer port.

The reference is a test-only transcription of the official eval/eval_score.py (see
tests/mmlong_reference_transcription.py). Hand-checked cases carry their derivation in the comment.
Nothing here touches the network or executes externally fetched code.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from vlmbench.datasets import mmlong_official_scorer as port

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("mmlong_reference", ROOT / "tests/mmlong_reference_transcription.py")
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

HAND_CASES = [
    # (gt, pred, type, expected score, derivation)
    ("3", "3.0", "Int", 1.0, "int('3')==int(float('3.0'))==3"),
    ("3", "x", "Int", 0.0, "pred becomes '' after the except; 3 == '' is False"),
    ("12.5%", "12.5", "Float", 1.0, "clean -> 12.5 both; isclose(12.5, 12.5) inside the %-variants"),
    ("1000", "1.0k", "Float", 0.0, "float('1.0k') raises; pred '' makes is_float_equal return False"),
    ("page 5", "page 5", "Str", 1.0, "is_exact_match: startswith('page') -> exact equality -> True"),
    ("123", "124", "Str", 0.0, "is_exact_match: \\d+ -> exact equality -> False"),
    ("abcd", "abce", "Str", 0.75, "not exact: 1 - 1/4 = 0.75 > 0.5 kept"),
    ("abcd", "wxyz", "Str", 0.0, "not exact: 1 - 4/4 = 0 <= 0.5 zeroed"),
    ("Paris (France)", "paris", "Str", 1.0, "parenthesis removed -> 'paris' == 'paris' -> ANLS 1.0"),
    ("['apple', 'pear']", "['pear', 'apple']", "List", 1.0, "both sorted to [apple, pear]; equal; not numeric or exact"),
    ("['a', 'b']", "['a']", "List", 0.0, "length mismatch -> 0.0"),
    ("[1, 2]", "[2, 1]", "List", 1.0, "sorted ['1','2'] both; isfloat -> '1-2' == '1-2'"),
]


@pytest.mark.parametrize("gt,pred,kind,expected,why", HAND_CASES)
def test_hand_checked_case_matches_the_derivation_and_the_transcription(gt, pred, kind, expected, why):
    assert port.eval_score(gt, pred, kind) == pytest.approx(expected), why
    assert ref.eval_score(gt, pred, kind) == pytest.approx(expected), why


def test_hand_checked_sample_set_for_acc_and_f1():
    # 3 samples: s1 answerable and right; s2 unanswerable and predicted; s3 answerable and missed.
    # acc = (1+0+0)/3. answerable = [s1, s3]: recall = (1+0)/2 = 0.5.
    # predicted (pred != 'Not answerable') = [s1, s2]: precision = (1+0)/2 = 0.5. f1 = 0.5.
    samples = [
        {"answer": "a", "pred": "a", "score": 1.0},
        {"answer": "Not answerable", "pred": "x", "score": 0.0},
        {"answer": "b", "pred": "Not answerable", "score": 0.0},
    ]
    assert port.eval_acc_and_f1(samples) == pytest.approx((1 / 3, 0.5))
    assert ref.eval_acc_and_f1(samples) == pytest.approx((1 / 3, 0.5))



def test_an_empty_gold_list_raises_in_both_the_port_and_upstream_logic():
    # upstream indexes gt[0] after the length check, so "[]" gold raises IndexError. The port keeps that.
    with pytest.raises(IndexError):
        ref.eval_score("[]", "[]", "List")
    with pytest.raises(IndexError):
        port.eval_score("[]", "[]", "List")


ALPHABET = "abcdeAB 0123.%$'\"()-,"


def _rstr(rng, max_len):
    return "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, max_len)))


def _case(rng):
    kind = rng.choice(["Int", "Float", "Str", "None", "List"])
    if kind == "Int":
        n = rng.randint(-50, 50)
        gt = str(n)
        pred = rng.choice([str(n), f"{n}.0", _rstr(rng, 3), ""])
    elif kind == "Float":
        x = rng.uniform(0, 1000)
        gt = f"{x:.{rng.randint(0, 3)}f}" + rng.choice(["", "%"])
        pred = rng.choice([gt, f"{x * rng.choice([1, 0.01, 100]):.{rng.randint(0, 3)}f}", _rstr(rng, 4)])
    elif kind in ("Str", "None"):
        gt = rng.choice([_rstr(rng, 10), "page 3", "123", "2024-01-02", "user@example.com"])
        pred = rng.choice([gt, _rstr(rng, 10), gt[:-1] if gt else "", gt.upper()])
    else:
        # official gold lists are never empty: upstream indexes gt[0] (see the quirk test below)
        items = [rng.choice([_rstr(rng, 6), str(rng.randint(0, 99))]) for _ in range(rng.randint(1, 4))]
        gt = repr(items)
        shuffled = items[:]
        rng.shuffle(shuffled)
        pred = repr(rng.choice([shuffled, items[:-1], [_rstr(rng, 5)] * len(items)]))
    return gt, pred, kind


def test_fuzz_eval_score_port_agrees_with_the_transcription_on_4000_seeded_cases():
    rng = random.Random(20261110)
    mismatches, outcomes = [], set()
    for _ in range(4000):
        gt, pred, kind = _case(rng)
        mine, theirs = port.eval_score(gt, pred, kind), ref.eval_score(gt, pred, kind)
        outcomes.add("zero" if mine == 0.0 else "one" if mine == 1.0 else "fraction")
        if mine != theirs:
            mismatches.append((gt, pred, kind, mine, theirs))
    assert mismatches == [], mismatches[:5]
    # the seeded cases must not be trivial: they reach zero, one and a fractional (ANLS) score
    assert outcomes == {"zero", "one", "fraction"}


def test_fuzz_acc_and_f1_port_agrees_with_the_transcription_on_500_sample_sets():
    rng = random.Random(7)
    for _ in range(500):
        samples = []
        for _ in range(rng.randint(1, 12)):
            gt, pred, kind = _case(rng)
            unanswerable = rng.random() < 0.3
            answer = "Not answerable" if unanswerable else gt
            answer_format = "None" if unanswerable else kind  # official data: unanswerable items are format None
            p = rng.choice(["Not answerable", pred]) if rng.random() < 0.3 else pred
            score = ref.eval_score(answer, p, answer_format)
            samples.append({"answer": answer, "pred": p, "score": score, "answer_format": answer_format})
        mine, theirs = port.eval_acc_and_f1(samples), ref.eval_acc_and_f1(samples)
        assert mine == pytest.approx(theirs, abs=1e-12)
