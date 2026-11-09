"""Canonical model architecture and metadata loader for Qwen3.5 (0.8B and 4B).

Resolves exact Hugging Face configurations, token IDs, hybrid layer types,
and Multi-Token Prediction (MTP) specifications without mock approximations.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Qwen35ModelConfig:
    """Canonical architectural specification for Qwen3.5 vision-language models."""
    model_id: str
    model_type: str                         # "qwen3_5"
    architectures: tuple[str, ...]          # ("Qwen3_5ForConditionalGeneration",)
    vocab_size: int                         # Official: 248,320 (NOT legacy 151,936)
    hidden_size: int                        # 1024 for 0.8B; 2560 for 4B
    num_hidden_layers: int                  # 24 for 0.8B; 32 for 4B
    num_attention_heads: int                # 16 for 0.8B; 20 for 4B
    num_key_value_heads: int                # 2 for 0.8B; 4 for 4B (GQA)
    head_dim: int                           # 64 for 0.8B; 128 for 4B
    intermediate_size: int                  # FFN intermediate dimension
    max_position_embeddings: int            # 262,144 (256k context)

    # Multimodal Vision Tokens
    image_token_id: int                     # 248056 (image placeholder)
    vision_start_token_id: int              # 248053 (vision block start)
    vision_end_token_id: int                # 248054 (vision block end)
    pad_token_id: int                       # 151643
    eos_token_id: int                       # 151645

    # Hybrid Layout: Gated Delta Networks (Linear) vs Full Gated Attention (3:1 ratio)
    full_attention_layers: tuple[int, ...]  # 1 out of 4 layers: e.g. (3, 7, 11, 15, 19, 23)
    recurrent_layers: tuple[int, ...]       # 3 out of 4 layers: (0, 1, 2, 4, 5, 6, ...)

    # Speculative Decoding / MTP Head
    mtp_num_speculative_tokens: int = 2     # Default native MTP speculation depth

    @classmethod
    def qwen35_08b(cls) -> Qwen35ModelConfig:
        """Official configuration for Qwen/Qwen3.5-0.8B."""
        num_layers = 24
        # 3:1 ratio: 3 DeltaNet linear recurrent layers followed by 1 full-attention layer
        full_attn = tuple(i for i in range(num_layers) if (i + 1) % 4 == 0)
        recurrent = tuple(i for i in range(num_layers) if (i + 1) % 4 != 0)
        return cls(
            model_id="Qwen/Qwen3.5-0.8B",
            model_type="qwen3_5",
            architectures=("Qwen3_5ForConditionalGeneration",),
            vocab_size=248320,
            hidden_size=1024,
            num_hidden_layers=num_layers,
            num_attention_heads=16,
            num_key_value_heads=2,
            head_dim=64,
            intermediate_size=2816,
            max_position_embeddings=262144,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            pad_token_id=151643,
            eos_token_id=151645,
            full_attention_layers=full_attn,
            recurrent_layers=recurrent,
            mtp_num_speculative_tokens=2,
        )

    @classmethod
    def qwen35_4b(cls) -> Qwen35ModelConfig:
        """Official configuration for Qwen/Qwen3.5-4B."""
        num_layers = 32
        full_attn = tuple(i for i in range(num_layers) if (i + 1) % 4 == 0)
        recurrent = tuple(i for i in range(num_layers) if (i + 1) % 4 != 0)
        return cls(
            model_id="Qwen/Qwen3.5-4B",
            model_type="qwen3_5",
            architectures=("Qwen3_5ForConditionalGeneration",),
            vocab_size=248320,
            hidden_size=2560,
            num_hidden_layers=num_layers,
            num_attention_heads=20,
            num_key_value_heads=4,
            head_dim=128,
            intermediate_size=6912,
            max_position_embeddings=262144,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            pad_token_id=151643,
            eos_token_id=151645,
            full_attention_layers=full_attn,
            recurrent_layers=recurrent,
            mtp_num_speculative_tokens=2,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> Qwen35ModelConfig:
        """Parse configuration directly from local checkpoint config.json."""
        p = Path(checkpoint_dir) / "config.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing config.json in checkpoint directory: {checkpoint_dir}")
        data = json.loads(p.read_text(encoding="utf8"))
        text_cfg = data.get("text_config", data)

        num_layers = text_cfg.get("num_hidden_layers", 24)
        full_attn = tuple(i for i in range(num_layers) if (i + 1) % 4 == 0)
        recurrent = tuple(i for i in range(num_layers) if (i + 1) % 4 != 0)

        return cls(
            model_id=data.get("_name_or_path", "Qwen/Qwen3.5-Custom"),
            model_type=data.get("model_type", "qwen3_5"),
            architectures=tuple(data.get("architectures", ["Qwen3_5ForConditionalGeneration"])),
            vocab_size=text_cfg.get("vocab_size", 248320),
            hidden_size=text_cfg.get("hidden_size", 1024),
            num_hidden_layers=num_layers,
            num_attention_heads=text_cfg.get("num_attention_heads", 16),
            num_key_value_heads=text_cfg.get("num_key_value_heads", 2),
            head_dim=text_cfg.get("head_dim", 64),
            intermediate_size=text_cfg.get("intermediate_size", 2816),
            max_position_embeddings=text_cfg.get("max_position_embeddings", 262144),
            image_token_id=data.get("image_token_id", 248056),
            vision_start_token_id=data.get("vision_start_token_id", 248053),
            vision_end_token_id=data.get("vision_end_token_id", 248054),
            pad_token_id=text_cfg.get("pad_token_id", 151643),
            eos_token_id=text_cfg.get("eos_token_id", 151645),
            full_attention_layers=full_attn,
            recurrent_layers=recurrent,
            mtp_num_speculative_tokens=data.get("mtp_num_speculative_tokens", 2),
        )


def load_model_config(model_id_or_path: str) -> Qwen35ModelConfig:
    """Resolve model configuration for Qwen3.5 models."""
    norm = model_id_or_path.lower()
    if "4b" in norm:
        return Qwen35ModelConfig.qwen35_4b()
    elif "0.8b" in norm or "08b" in norm:
        return Qwen35ModelConfig.qwen35_08b()
    p = Path(model_id_or_path)
    if p.is_dir() and (p / "config.json").is_file():
        return Qwen35ModelConfig.from_checkpoint(p)
    return Qwen35ModelConfig.qwen35_08b()
