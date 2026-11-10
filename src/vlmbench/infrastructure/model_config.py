"""Canonical model architecture and metadata loader for Qwen3.5 (0.8B and 4B).

Resolves exact Hugging Face configurations, token IDs, hybrid layer types,
and Multi-Token Prediction (MTP) specifications without mock approximations.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIN_STATUS_MANIFEST = "PINNED_MANIFEST_CONFIG_SHA256"
PIN_STATUS_HARDCODED = "HARDCODED_NOT_FROM_LOCAL_FILE"
PIN_STATUS_UNPINNED_4B = "UNPINNED_NO_LOCAL_CONFIG_JSON"
PIN_STATUS_UNPINNED_NO_MANIFEST = "UNPINNED_NO_MANIFEST"


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

    # Provenance: how these values were obtained. Never silently default.
    pin_status: str = PIN_STATUS_UNPINNED_NO_MANIFEST
    pin_revision: str | None = None

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
            num_attention_heads=8,      # config.json text_config.num_attention_heads
            num_key_value_heads=2,      # config.json text_config.num_key_value_heads
            head_dim=256,               # config.json text_config.head_dim
            intermediate_size=3584,     # config.json text_config.intermediate_size
            max_position_embeddings=262144,
            image_token_id=248056,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            pad_token_id=151643,        # NOT in config.json; unpinned legacy value
            eos_token_id=248044,        # config.json text_config.eos_token_id
            full_attention_layers=full_attn,
            recurrent_layers=recurrent,
            mtp_num_speculative_tokens=2,  # serving depth; not a config.json field
            pin_status=PIN_STATUS_HARDCODED,
            pin_revision="2fc06364715b967f1860aea9cf38778875588b17",
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
            # No local config.json or manifest pin exists for 4B. Values are NOT verified.
            pin_status=PIN_STATUS_UNPINNED_4B,
            pin_revision=None,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> Qwen35ModelConfig:
        """Parse configuration from a local checkpoint config.json. Required keys are strict.

        Provenance: if the directory holds starlight_local_manifest.json whose config.json
        sha256 matches the local file, pin_status is PINNED_MANIFEST_CONFIG_SHA256 and the
        manifest revision is recorded. Otherwise the config is UNPINNED_NO_MANIFEST.
        """
        d = Path(checkpoint_dir)
        p = d / "config.json"
        if not p.is_file():
            raise FileNotFoundError(f"Missing config.json in checkpoint directory: {checkpoint_dir}")
        raw = p.read_bytes()
        data = json.loads(raw.decode("utf8"))
        text_cfg = data.get("text_config", data)

        def _req(src: dict, key: str) -> Any:
            if key not in src:
                raise ValueError(f"config.json in {checkpoint_dir} lacks required key '{key}'")
            return src[key]

        layer_types = _req(text_cfg, "layer_types")
        num_layers = int(_req(text_cfg, "num_hidden_layers"))
        if len(layer_types) != num_layers:
            raise ValueError("layer_types length does not match num_hidden_layers")
        full_attn = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
        recurrent = tuple(i for i, t in enumerate(layer_types) if t != "full_attention")

        pin_status, pin_revision, model_id = PIN_STATUS_UNPINNED_NO_MANIFEST, None, data.get("_name_or_path", "")
        manifest_path = d / "starlight_local_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf8"))
            entry = manifest.get("files", {}).get("config.json", {})
            if entry.get("sha256") == hashlib.sha256(raw).hexdigest():
                pin_status = PIN_STATUS_MANIFEST
                pin_revision = manifest.get("revision")
                model_id = manifest.get("model_id", model_id)

        return cls(
            model_id=model_id or "UNKNOWN_CHECKPOINT_ID",
            model_type=_req(data, "model_type"),
            architectures=tuple(_req(data, "architectures")),
            vocab_size=int(_req(text_cfg, "vocab_size")),
            hidden_size=int(_req(text_cfg, "hidden_size")),
            num_hidden_layers=num_layers,
            num_attention_heads=int(_req(text_cfg, "num_attention_heads")),
            num_key_value_heads=int(_req(text_cfg, "num_key_value_heads")),
            head_dim=int(_req(text_cfg, "head_dim")),
            intermediate_size=int(_req(text_cfg, "intermediate_size")),
            max_position_embeddings=int(_req(text_cfg, "max_position_embeddings")),
            image_token_id=int(_req(data, "image_token_id")),
            vision_start_token_id=int(_req(data, "vision_start_token_id")),
            vision_end_token_id=int(_req(data, "vision_end_token_id")),
            pad_token_id=int(text_cfg.get("pad_token_id", 151643)),  # not present in config.json
            eos_token_id=int(_req(text_cfg, "eos_token_id")),
            full_attention_layers=full_attn,
            recurrent_layers=recurrent,
            mtp_num_speculative_tokens=int(data.get("mtp_num_speculative_tokens", 2)),
            pin_status=pin_status,
            pin_revision=pin_revision,
        )


_KNOWN_MODEL_CONFIGS = {
    "qwen/qwen3.5-0.8b": Qwen35ModelConfig.qwen35_08b,
    "qwen35_08b": Qwen35ModelConfig.qwen35_08b,
    "qwen/qwen3.5-4b": Qwen35ModelConfig.qwen35_4b,
    "qwen35_4b": Qwen35ModelConfig.qwen35_4b,
}


def load_model_config(model_id_or_path: str) -> Qwen35ModelConfig:
    """Resolve a model configuration by EXACT registry key or local checkpoint directory.

    Unknown IDs raise ValueError. There is no fallback to Qwen3.5-0.8B. The 4B entry is
    returned with pin_status UNPINNED_NO_LOCAL_CONFIG_JSON, not as verified.
    """
    key = str(model_id_or_path).strip().lower()
    if key in _KNOWN_MODEL_CONFIGS:
        return _KNOWN_MODEL_CONFIGS[key]()
    p = Path(model_id_or_path)
    if p.is_dir() and (p / "config.json").is_file():
        return Qwen35ModelConfig.from_checkpoint(p)
    raise ValueError(
        f"Unknown or unpinned model '{model_id_or_path}': refusing to default to Qwen3.5-0.8B"
    )
