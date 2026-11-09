"""Seven-Arm Infrastructure: H2O, Declarative Attention (DA), and Grounding Evidence (ECRD).

Production implementations for Qwen3.5 (0.8B/4B) multimodal serving benchmarks
consistent with vLLM 0.29.0 and PyTorch reference execution.
"""
from __future__ import annotations

from .h2o import (
    H2OConfig,
    H2OCacheTracker,
    H2OEvictionPolicy,
    H2OMTPTransaction,
)
from .declarative_attention import (
    DAConfig,
    DATag,
    DATagType,
    DAFSM,
    DAMaskGenerator,
    DAAttentionModifier,
)
from .grounding_evidence import (
    ECRDConfig,
    EvidenceItem,
    EvidencePool,
    ECRDLogitsProcessor,
)
from .adapters import (
    HardwareGateError,
    check_vllm_hardware_compatibility,
)
from .model_config import (
    Qwen35ModelConfig,
    load_model_config,
)
from .real_engine import (
    RealSevenArmEngine,
    RealExecutionResult,
    DeltaNetRecurrentLayer,
    QwenFullAttentionLayer,
    Qwen35MTPModule,
    Qwen35NeuralModel,
)

__all__ = [
    "H2OConfig",
    "H2OCacheTracker",
    "H2OEvictionPolicy",
    "H2OMTPTransaction",
    "DAConfig",
    "DATag",
    "DATagType",
    "DAFSM",
    "DAMaskGenerator",
    "DAAttentionModifier",
    "ECRDConfig",
    "EvidenceItem",
    "EvidencePool",
    "ECRDLogitsProcessor",
    "HardwareGateError",
    "check_vllm_hardware_compatibility",
    "Qwen35ModelConfig",
    "load_model_config",
    "RealSevenArmEngine",
    "RealExecutionResult",
    "DeltaNetRecurrentLayer",
    "QwenFullAttentionLayer",
    "Qwen35MTPModule",
    "Qwen35NeuralModel",
]
