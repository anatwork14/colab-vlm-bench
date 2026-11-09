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


def _receipt(status="RUN_COMPLETED", sha="a" * 40, dtype="float16", tokens=None, inputs=("text0", "text1", "image"),
             device=None):
    tokens = tokens or {arm: [[1, 2]] * len(inputs) for arm in smoke.ARMS}
    # C-9: the receipt names its device. Default: float32 is the CPU reference, float16 is CUDA.
    device = device or ("cpu" if dtype == "float32" else "cuda")
    return {
        "status": status, "git_sha": sha, "dtype": dtype, "device": device, "tag": smoke.TAG[device],
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


def test_request_is_complete_at_eos_or_at_the_token_cap_and_not_otherwise():
    # The engine never emits EOS (<|im_end|>), so an EOS stop is a short, complete output.
    assert smoke.request_complete({"stop_reason": "eos", "tokens_generated_count": 3}) is True
    assert smoke.request_complete({"stop_reason": "max_new_tokens", "tokens_generated_count": smoke.MAX_NEW_TOKENS}) is True
    assert smoke.request_complete({"stop_reason": "max_new_tokens", "tokens_generated_count": 3}) is False
    assert smoke.request_complete({"stop_reason": None, "tokens_generated_count": smoke.MAX_NEW_TOKENS}) is False


# --- C-9: compare_receipts requires exactly (CUDA, CPU), in that order -----------------------------

def test_C9_two_cuda_receipts_are_refused():
    with pytest.raises(ValueError, match="cpu"):
        smoke.compare_receipts(_receipt(), _receipt())


def test_C9_cpu_first_and_cuda_second_is_refused():
    with pytest.raises(ValueError, match="cuda"):
        smoke.compare_receipts(_receipt(dtype="float32"), _receipt())


def test_C9_a_cuda_receipt_in_the_cpu_slot_is_refused():
    with pytest.raises(ValueError):
        smoke.compare_receipts(_receipt(), _receipt(device="cuda", dtype="float16"))


def test_C9_the_tag_must_match_the_device():
    bad = _receipt(dtype="float32")
    bad["tag"] = "gpu_fp16"  # a CPU-labelled receipt carrying the GPU tag
    with pytest.raises(ValueError):
        smoke.compare_receipts(_receipt(), bad)


def test_C9_correct_order_compares():
    out = smoke.compare_receipts(_receipt(), _receipt(dtype="float32"))
    assert out["token_agreement"] == "IDENTICAL_ALL_INPUTS"
