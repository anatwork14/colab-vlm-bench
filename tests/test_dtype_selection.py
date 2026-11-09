"""ITER2 D5 (3): the official hooked model's dtype is explicit and cannot leak across devices.

Default behaviour is unchanged (bf16 on sm_80+ CUDA, fp32 otherwise). fp16 is an explicit
opt-in and is allowed only on CUDA. Pure-function tests: no model is loaded.
"""
from __future__ import annotations

import pytest
import torch

from vlmbench.infrastructure.real_engine import official_cache_key, select_official_dtype


def test_default_is_unchanged_fp32_on_sm61_and_cpu():
    assert select_official_dtype("cuda", True, 6) is torch.float32  # GTX 1080 Ti, sm_61
    assert select_official_dtype("cpu", False, None) is torch.float32


def test_default_is_bf16_on_sm80_and_up():
    assert select_official_dtype("cuda", True, 8) is torch.bfloat16


def test_fp16_is_an_explicit_override_on_cuda_only():
    assert select_official_dtype("cuda", True, 6, "fp16") is torch.float16
    with pytest.raises(ValueError, match="requires a CUDA device"):
        select_official_dtype("cpu", False, None, "fp16")


def test_override_wins_over_capability_and_unknown_names_are_refused():
    assert select_official_dtype("cuda", True, 8, "fp32") is torch.float32
    with pytest.raises(ValueError, match="unknown dtype override"):
        select_official_dtype("cuda", True, 6, "int8")


def test_cache_key_separates_dtypes_so_an_fp16_bundle_never_serves_fp32():
    from pathlib import Path

    p = Path("/x/ckpt")
    assert official_cache_key(p, "cuda", torch.float16) != official_cache_key(p, "cuda", torch.float32)
    assert official_cache_key(p, "cuda", torch.float16) == official_cache_key(p, "cuda", torch.float16)
