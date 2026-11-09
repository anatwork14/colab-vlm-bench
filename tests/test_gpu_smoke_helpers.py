"""ITER2 D5 (3): helpers of the bounded GTX 1080 Ti PyTorch-reference smoke (pure functions)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/pytorch_reference_gpu_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("pytorch_reference_gpu_smoke", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


smoke = _load()


def test_parse_nvidia_smi_line():
    line = "NVIDIA GeForce GTX 1080 Ti, GPU-1234, 5090, 11264"
    assert smoke.parse_smi_csv(line) == {
        "name": "NVIDIA GeForce GTX 1080 Ti", "uuid": "GPU-1234", "free_mib": 5090, "total_mib": 11264,
    }


def test_free_memory_below_three_gib_aborts_before_loading():
    smoke.check_free_memory(3072)  # exactly 3 GiB is allowed
    with pytest.raises(smoke.SmokeAbort, match="free GPU memory"):
        smoke.check_free_memory(3071)


def test_token_comparison_reports_divergence_without_hiding_it():
    same = smoke.compare_tokens([1, 2, 3], [1, 2, 3])
    assert same["identical"] is True and same["first_divergence"] is None
    diverged = smoke.compare_tokens([1, 2, 9, 4], [1, 2, 3, 4])
    assert diverged["identical"] is False
    assert diverged["first_divergence"] == 2
    assert diverged["matching_prefix_len"] == 2
    unequal_len = smoke.compare_tokens([1, 2], [1, 2, 3])
    assert unequal_len["identical"] is False and unequal_len["first_divergence"] == 2


def test_label_and_limits_are_the_agreed_ones():
    assert smoke.LABEL == "pytorch_reference_gtx1080ti_smoke_not_vllm_not_performance"
    assert smoke.MIN_FREE_MIB == 3072
    # ITER2 D9: the GPU run and the short CPU fp32 reference are separate runs with their own budgets.
    assert smoke.WALL_LIMIT_S == {"cuda": 600, "cpu": 300}
    assert smoke.ARMS == ("base", "h2o_base")
    assert smoke.MAX_NEW_TOKENS <= 8


def _receipt(status="RUN_COMPLETED", sha="a" * 40, dtype="float16", tokens=None, inputs=("text0", "text1", "image")):
    tokens = tokens or {arm: [[1, 2]] * len(inputs) for arm in smoke.ARMS}
    return {
        "status": status, "git_sha": sha, "dtype": dtype,
        "results": {
            arm: [{"input": name, "tokens": list(t), "tokens_generated_count": len(t)}
                  for name, t in zip(inputs, tokens[arm])]
            for arm in smoke.ARMS
        },
    }


def test_compare_reports_identical_tokens_when_every_input_matches():
    out = smoke.compare_receipts(_receipt(), _receipt(dtype="float32"))
    assert out["token_agreement"] == "IDENTICAL_ALL_INPUTS"
    assert out["gpu_dtype"] == "float16" and out["reference_dtype"] == "float32"


def test_compare_names_the_first_divergence_and_does_not_hide_it():
    cpu = _receipt(dtype="float32", tokens={"base": [[1, 2], [5, 6], [7, 8]], "h2o_base": [[1, 2], [1, 2], [1, 2]]})
    out = smoke.compare_receipts(_receipt(), cpu)
    assert out["token_agreement"] == "DIVERGENCE_PRESENT_SEE_COMPARISON"
    assert out["comparison"]["base"][1]["first_divergence"] == 0
    assert out["comparison"]["h2o_base"][0]["identical"] is True


@pytest.mark.parametrize("bad", [
    {"status": "ABORTED: wall-clock"},
    {"sha": "b" * 40},
    {"inputs": ("text0", "text9", "image")},
])
def test_compare_refuses_incomplete_mismatched_or_unrelated_receipts(bad):
    cpu = _receipt(dtype="float32", **{k: v for k, v in bad.items() if k != "status"},
                   **({"status": bad["status"]} if "status" in bad else {}))
    with pytest.raises(ValueError):
        smoke.compare_receipts(_receipt(), cpu)
