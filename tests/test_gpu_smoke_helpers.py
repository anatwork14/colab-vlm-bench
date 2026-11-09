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
    assert smoke.WALL_LIMIT_S == 600
    assert smoke.ARMS == ("base", "h2o_base")
