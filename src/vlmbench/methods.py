"""Seven-arm Qwen3.5 benchmark registry with fail-closed backend qualification.

References are pinned to upstream Git SHAs, but upstream availability does NOT
imply Qwen3.5 + vLLM + A100 support. No fake flag/proxy is accepted as H2O/DA/ECRD.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

METHODS = (
    "base",
    "h2o_base",
    "da_base",
    "ecrd_base",
    "mtp",
    "h2o_mtp",
    "da_mtp",
)
NATIVE_AVAILABLE = frozenset(("base", "mtp"))
UPSTREAM = {
    "h2o": {
        "repo": "https://github.com/FMInference/H2O",
        "commit": "ac75c2a8a9e76832b2a4139b9363373b56336bfb",
        "implementation": "HF/FlexGen KV heavy-hitter eviction",
        "qualification": "Qwen3.5 hybrid DeltaNet attention + vLLM v0.29 A100 port NOT_VERIFIED",
    },
    "da": {
        "repo": "https://github.com/itsnamgyu/declarative-attention",
        "commit": "24110b7b441ad9423f55952a2c02d23eac80ec52",
        "implementation": "DA tag FSM + vLLM attention mask/read selection; original KV preserved",
        "qualification": "Current upstream profiles SM100 and non-Qwen3.5-0.8B/4B; A100 port NOT_VERIFIED",
    },
    "ecrd": {
        "repo": "https://github.com/uuuuZYC/ECRD",
        "commit": "0bc209f599bfef0d28d19e9a34024464406a9295",
        "implementation": "Evidence-constrained decoding logits reweighting + optional visual decider",
        "qualification": "HF Qwen2.5-VL demo only; Qwen3.5/vLLM and auxiliary costs NOT_VERIFIED",
    },
}


@dataclass(frozen=True)
class MethodArm:
    id: str
    cache_policy: str
    attention_policy: str
    decoding: str
    target_revision_required: bool = True
    source: str | None = None
    gate: str = "NEEDS_REAL_A100_BACKEND_QUALIFICATION"


ARM_REGISTRY = {
    "base": MethodArm("base", "fullkv", "default", "greedy", gate="NATIVE_VLLM_PRELIMINARY"),
    "h2o_base": MethodArm("h2o_base", "h2o_eviction", "default", "greedy", source="h2o"),
    "da_base": MethodArm("da_base", "fullkv_preserved", "declarative_selection", "greedy", source="da"),
    "ecrd_base": MethodArm("ecrd_base", "fullkv", "default", "ecrd_greedy", source="ecrd"),
    "mtp": MethodArm("mtp", "fullkv", "default", "native_mtp_greedy", gate="NATIVE_VLLM_PRELIMINARY"),
    "h2o_mtp": MethodArm("h2o_mtp", "h2o_eviction", "default", "native_mtp_greedy", source="h2o"),
    "da_mtp": MethodArm("da_mtp", "fullkv_preserved", "declarative_selection", "native_mtp_greedy", source="da"),
}


def validate_method_manifest(m: dict) -> None:
    if m.get("schema_version") != "1.0":
        raise ValueError("Unknown method campaign schema")
    models = m.get("models")
    if models != ["qwen35_08b", "qwen35_4b"]:
        raise ValueError("Only Qwen3.5 0.8B and 4B are authorized")
    methods = m.get("methods")
    if methods != list(METHODS):
        raise ValueError("Seven frozen methods in canonical order required")
    if m.get("datasets") != ["memlens", "mmlongbench_doc"]:
        raise ValueError("Two datasets required")
    if m.get("temperature") != 0 or m.get("top_p") != 1.0:
        raise ValueError("Greedy inference: temperature=0, top_p=1")
    if m.get("gpu") != {"type": "NVIDIA A100 40GB", "dtype": "bfloat16", "tensor_parallel_size": 1}:
        raise ValueError("A100 40GB BF16 TP1 hardware contract required")
    if m.get("track") not in ("pilot", "confirmatory"):
        raise ValueError("Unknown track")
    if m.get("track") == "pilot":
        expected = ([2048, 16384], [1, 8, 32], 1)
    else:
        expected = ([128, 512, 1024, 2048, 4096, 8192, 16384],
                    [1, 2, 4, 8, 16, 32], 3)
    if (m.get("input_lengths"), m.get("ccu"), m.get("repetitions")) != expected:
        raise ValueError("Frozen method experiment grid mismatch")
    if m.get("output_tokens") != 256 or m.get("prefix_caching") is not False:
        raise ValueError("Fixed-output serving policy and no prefix caching required")
    if m.get("quality") != {
        "track": "native_full_context",
        "temperature": 0,
        "top_p": 1.0,
        "natural_eos": True,
        "scorers": ["upstream_official", "independent_evidence_judge"],
    }:
        raise ValueError("Full-context official and evidence-supported quality scoring required")
    if m.get("source_pins") != {k: v["commit"] for k, v in UPSTREAM.items()}:
        raise ValueError("Upstream source provenance changed without a new manifest version")


def plan(m: dict) -> dict:
    validate_method_manifest(m)
    count = len(m["models"]) * len(m["methods"]) * len(m["datasets"])
    grid = len(m["input_lengths"]) * len(m["ccu"]) * m["repetitions"]
    return {
        "campaign_id": m["campaign_id"],
        "status": "PLANNED_NOT_MEASURED",
        "track": m["track"],
        "model_count": len(m["models"]),
        "method_count": len(m["methods"]),
        "dataset_count": len(m["datasets"]),
        "cells_per_model_method_dataset": grid,
        "total_serving_cells": count * grid,
        "temperature": m["temperature"],
        "methods": [
            {**asdict(ARM_REGISTRY[name]),
             "status": "NOT_GPU_QUALIFIED",
             "implementation_status": (
                 "NATIVE_VLLM_CODE_EXISTS_NO_A100_PROOF"
                 if name in NATIVE_AVAILABLE
                 else "BLOCKED_EXTERNAL_ENGINE_PORT_REQUIRED"
             )}
            for name in m["methods"]
        ],
        "note": "No H2O/DA/ECRD fallback. No method is GPU-qualified by this plan.",
    }


class MethodUnqualified(RuntimeError):
    pass


def native_mode_or_block(method: str) -> str:
    """Only Base and MTP can route to the existing vLLM runner.

    Other methods require independently reviewed executable backend adapters,
    with attention/kernel hooks + trace proving actual method activation.
    """
    if method not in ARM_REGISTRY:
        raise ValueError(f"Unknown arm: {method}")
    if method not in NATIVE_AVAILABLE:
        raise MethodUnqualified(
            f"{method} is BLOCKED_EXTERNAL_ENGINE_PORT_REQUIRED. "
            f"Do not substitute vanilla vLLM or prompt engineering."
        )
    return {"base": "base", "mtp": "mtp"}[method]


def blocked_method_receipt(method: str, model: str, dataset: str) -> dict:
    if method not in ARM_REGISTRY:
        raise ValueError(f"Unknown arm {method}")
    if method in NATIVE_AVAILABLE:
        raise ValueError("Native arms must produce actual inference receipts, not blocked placeholders")
    spec = ARM_REGISTRY[method]
    return {
        "status": "BACKEND_UNQUALIFIED",
        "method": method, "model": model, "dataset": dataset,
        "source_pin": UPSTREAM[spec.source],
        "latency_ms": None, "throughput_tokens_per_s": None,
        "accuracy": None, "hallucination_rate": None,
        "reason": "No verified Qwen3.5 + A10040 + vLLM implementation/activation trace",
        "not_a_gpu_attempt": True,
    }


def manifest_from_path(path: str | Path) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    validate_method_manifest(obj)
    return obj
