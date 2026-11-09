"""Real PyTorch Neural Model & Official Transformers Hooking Engine for Seven-Arm Benchmarking.

What this module does, and what it does NOT establish (see docs/engine/P7_ENGINE_AUDIT_20261109.md):
1. Official checkpoint loading (P0-01):
   - Loads the official HF `Qwen3_5ForConditionalGeneration` and the `Qwen35MTPModule` head.
   - Tensor and parameter coverage are computed at load time and reported in
     `pretrained_weight_summary`. A full-coverage load is required for `is_official_hf_hooked`.
   - Partial coverage keeps the scaffold labels (`PARTIALLY_WEIGHTED_PYTORCH_REFERENCE`) and
     never reports the real-forward gate as PASS.
2. Multimodal inputs per arm (P0-02):
   - Processes real image bytes through the official `AutoProcessor` and the ViT encoder for
     the arms that take an image.
   - Records `visual_encoder_trace` (`pixel_values_shape`, `visual_token_embeddings_count`).
3. H2O KV eviction (physical compaction on full-attention layers):
   - Computes attention weights from forward passes and compacts `DynamicCache` tensors.
   - DeltaNet recurrent layers are exempt. Protected (sink and visual) tokens sit outside the budget.
   - Reports retained and evicted slots. No physical VRAM saving is measured.
4. Declarative attention (DA) cost accounting:
   - Injects 2D boolean read masks into attention logits before softmax.
   - Reports `attention_mask_restriction_ratio` as a logical estimate. No KV-read saving or HBM
     traffic reduction is claimed.
5. MTP (multi-token prediction) in this module:
   - The serving-style `mtp` arm proposes drafts for diagnostics only and commits none. It does not
     verify them. Its candidate-vs-target agreement is not an acceptance rate.
   - Verifier-derived greedy k=1 acceptance and KV/DeltaNet-state rollback exist only in
     `RealSevenArmEngine._speculative_mtp_greedy_k1`. It is labelled
     `cpu_greedy_k1_reference_not_vllm_mtp`. `execute_request` does not call it.
   - Native vLLM MTP is not verified here.
6. ECRD logits processing: a full-vocabulary logits processor (`is_argmax_invariant() -> False`).
7. Concurrency: a CPU test covers one H2O arm against sequential runs. Hook context is thread-local
   and the official forward is serialised on a lock. Broader isolation is NOT proven.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import gc
import hashlib
import io
import math
from pathlib import Path
import threading
import time
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import Qwen35ModelConfig, load_model_config
from .h2o import H2OConfig, H2OCacheTracker, H2OMTPTransaction
from .declarative_attention import DAConfig, DAFSM, DAMaskGenerator, DATag, DATagType
from .grounding_evidence import ECRDConfig, EvidencePool, EvidenceItem, ECRDLogitsProcessor
from .starlight import (
    STATUS_INFEASIBLE, StarlightLayerPlan, combine_summaries, compact_orig_positions,
    da_protection_from_spans, evidence_validity, grounding_protection_from_pool,
    plan_starlight_layer, summarize_plans,
)
from .adapters import HardwareGateError, check_vllm_hardware_compatibility


_PRETRAINED_STATE_CACHE: dict[tuple[str, int, int, int, int], tuple[dict[str, torch.Tensor], dict[str, Any]]] = {}
_OFFICIAL_HF_BUNDLE_CACHE: dict[tuple[str, str], dict[str, Any]] = {}
_DEFAULT_FIXTURE_IMAGE = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "fixtures" / "000000039769.png"


class _HookStepContext(threading.local):
    """Thread-local intervention and telemetry collector for hooked official Qwen3.5 layers."""

    def __init__(self) -> None:
        super().__init__()
        self.da_masks: dict[int, torch.Tensor] | None = None
        self.h2o_eviction_plans: dict[int, torch.Tensor] | None = None
        self.attention_matrices: dict[int, torch.Tensor] = {}
        self.layer_telemetries: dict[int, dict[str, Any]] = {}


class Qwen35MTPModule(nn.Module):
    """Official 15-tensor Qwen3.5 Multi-Token Prediction (MTP) speculative head.

    Maps all 15 `mtp.*` tensors from `model.safetensors-00001-of-00001.safetensors`:
    - `mtp.pre_fc_norm_embedding.weight` (1024,)
    - `mtp.pre_fc_norm_hidden.weight` (1024,)
    - `mtp.fc.weight` (1024, 2048)
    - `mtp.layers.0.*` (11 full-attention decoder layer tensors)
    - `mtp.norm.weight` (1024,)
    """

    def __init__(self, text_config: Any):
        super().__init__()
        from transformers.models.qwen3_5 import modeling_qwen3_5

        mtp_cfg = copy.deepcopy(text_config)
        mtp_cfg.layer_types = ["full_attention"]
        mtp_cfg._attn_implementation = "eager"
        self.hidden_size = mtp_cfg.hidden_size
        self.pre_fc_norm_embedding = modeling_qwen3_5.Qwen3_5RMSNorm(mtp_cfg.hidden_size, eps=mtp_cfg.rms_norm_eps)
        self.pre_fc_norm_hidden = modeling_qwen3_5.Qwen3_5RMSNorm(mtp_cfg.hidden_size, eps=mtp_cfg.rms_norm_eps)
        self.fc = nn.Linear(mtp_cfg.hidden_size * 2, mtp_cfg.hidden_size, bias=False)
        self.layers = nn.ModuleList([modeling_qwen3_5.Qwen3_5DecoderLayer(mtp_cfg, layer_idx=0)])
        self.norm = modeling_qwen3_5.Qwen3_5RMSNorm(mtp_cfg.hidden_size, eps=mtp_cfg.rms_norm_eps)
        self.rotary_emb = modeling_qwen3_5.Qwen3_5TextRotaryEmbedding(config=mtp_cfg)

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        next_token_embeds: torch.Tensor,
        lm_head: nn.Module,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute one source-head candidate (diagnostic; no persistent MTP KV history)."""
        b, seq_len, _ = last_hidden_state.shape
        norm_e = self.pre_fc_norm_embedding(next_token_embeds)
        norm_h = self.pre_fc_norm_hidden(last_hidden_state)
        h = self.fc(torch.cat([norm_e, norm_h], dim=-1))
        if position_ids is None:
            raise ValueError(
                "MTP needs absolute next-token position_ids; refusing per-step RoPE position reset"
            )
        if tuple(position_ids.shape) != (3, b, seq_len):
            raise ValueError(
                f"Expected MTP position_ids [3,{b},{seq_len}], got {tuple(position_ids.shape)}"
            )
        pos_emb = self.rotary_emb(h, position_ids.to(h.device))
        h = self.layers[0](h, position_embeddings=pos_emb, attention_mask=None)
        h = self.norm(h)
        return lm_head(h)


def _install_seven_arm_attention_hooks(hf_model: Any, hook_ctx: _HookStepContext) -> None:
    """Install H2O physical KV compaction and DA logit masking hooks on official Qwen3_5Attention layers."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for layer_idx, decoder_layer in enumerate(hf_model.model.language_model.layers):
        if getattr(decoder_layer, "block_type", None) != "full_attention":
            continue
        attn_mod = decoder_layer.self_attn

        def _make_hooked_forward(mod: Any, l_idx: int):
            def _hooked_forward(
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values: Any = None,
                **kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, mod.head_dim)

                query_states, gate = torch.chunk(
                    mod.q_proj(hidden_states).view(*input_shape, -1, mod.head_dim * 2), 2, dim=-1
                )
                gate = gate.reshape(*input_shape, -1)

                query_states = mod.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = mod.k_norm(mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                cos, sin = position_embeddings
                query_states, key_states = modeling_qwen3_5.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin
                )

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(key_states, value_states, l_idx)

                kv_bytes_before = (key_states.nelement() + value_states.nelement()) * key_states.element_size()
                physical_bytes_freed = 0
                pre_compaction_len = key_states.shape[2]
                mask_len_received = None if attention_mask is None else int(attention_mask.shape[-1])

                # Attention mask invariant: HF builds the decode mask from the PRE-compaction
                # cache, so its width must equal the pre-compaction KV length exactly.
                if attention_mask is not None and mask_len_received != pre_compaction_len:
                    raise ValueError(
                        f"Layer {l_idx}: attention mask width {mask_len_received} does not match "
                        f"KV length {pre_compaction_len}; refusing a silent slice"
                    )

                # 1. H2O Physical KV Eviction & Tensor Compaction on official DynamicLayer cache
                h2o_plans = getattr(hook_ctx, "h2o_eviction_plans", None)
                h2o_retained = h2o_plans.get(l_idx) if h2o_plans else None
                if h2o_retained is not None and len(h2o_retained) < key_states.shape[2]:
                    if attention_mask is not None:
                        # Select mask columns by the SAME retained positions as the KV cache.
                        attention_mask = torch.index_select(attention_mask, dim=-1, index=h2o_retained)
                    key_states = torch.index_select(key_states, dim=2, index=h2o_retained).contiguous()
                    value_states = torch.index_select(value_states, dim=2, index=h2o_retained).contiguous()
                    if past_key_values is not None and l_idx < len(past_key_values.layers):
                        past_key_values.layers[l_idx].keys = key_states
                        past_key_values.layers[l_idx].values = value_states
                    kv_bytes_after = (key_states.nelement() + value_states.nelement()) * key_states.element_size()
                    physical_bytes_freed = kv_bytes_before - kv_bytes_after

                q_len = query_states.shape[2]
                k_len = key_states.shape[2]

                key_rep = modeling_qwen3_5.repeat_kv(key_states, mod.num_key_value_groups)
                value_rep = modeling_qwen3_5.repeat_kv(value_states, mod.num_key_value_groups)

                attn_logits = torch.matmul(query_states, key_rep.transpose(2, 3)) * mod.scaling
                da_masked_neg: torch.Tensor | None = None
                if attention_mask is not None:
                    if attention_mask.shape[-1] != k_len:
                        raise ValueError(
                            f"Layer {l_idx}: aligned attention mask width {attention_mask.shape[-1]} "
                            f"!= KV length {k_len}"
                        )
                    attn_logits = attn_logits + attention_mask

                # 2. Declarative Attention (DA) Logit Masking (preserves 100% of KV residency)
                da_masks = getattr(hook_ctx, "da_masks", None)
                da_mask = da_masks.get(l_idx) if da_masks else None
                if da_mask is not None:
                    if da_mask.shape != (q_len, k_len):
                        da_mask_slice = da_mask[-q_len:, :k_len]
                    else:
                        da_mask_slice = da_mask
                    mask_expanded = da_mask_slice.unsqueeze(0).unsqueeze(0)
                    da_masked_neg = ~mask_expanded
                    attn_logits = attn_logits.masked_fill(da_masked_neg, -1e9)

                attn_weights = nn.functional.softmax(
                    attn_logits, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)

                # R4-3: MEASURE the post-softmax weight on every masked (head, query, key) entry.
                # This is what DA_MASKING_EXERCISED is allowed to rest on, not the constructed mask.
                da_measured: dict[str, Any] = {}
                if da_masked_neg is not None:
                    neg_entries = da_masked_neg.expand_as(attn_weights)
                    n_masked_entries = int(neg_entries.sum().item())
                    da_measured = {
                        "da_measured_masked_entry_count": n_masked_entries,
                        "da_measured_masked_weight_max": (
                            float(attn_weights.masked_select(neg_entries).max().item())
                            if n_masked_entries else None
                        ),
                    }

                if hasattr(hook_ctx, "attention_matrices"):
                    hook_ctx.attention_matrices[l_idx] = attn_weights.detach()
                if hasattr(hook_ctx, "layer_telemetries"):
                    hook_ctx.layer_telemetries[l_idx] = {
                        **da_measured,
                        "kv_seq_len": k_len,
                        "attention_mask_len": None if attention_mask is None else int(attention_mask.shape[-1]),
                        "attention_mask_len_received": mask_len_received,
                        "kv_bytes_allocated": (key_states.nelement() + value_states.nelement()) * key_states.element_size(),
                        "physical_bytes_freed": physical_bytes_freed,
                    }

                attn_output = torch.matmul(attn_weights, value_rep)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate)
                attn_output = mod.o_proj(attn_output)
                return attn_output, attn_weights

            return _hooked_forward

        attn_mod.forward = _make_hooked_forward(attn_mod, layer_idx)


def _snapshot_dynamic_cache(cache: Any) -> list[Any]:
    """Create a deep tensor snapshot of an official Qwen3.5 DynamicCache for MTP rollback."""
    snap: list[Any] = []
    for layer in cache.layers:
        if hasattr(layer, "keys") and layer.keys is not None:
            snap.append(("dynamic", layer.keys.clone(), layer.values.clone()))
        elif hasattr(layer, "conv_states"):
            snap.append((
                "linear",
                {k: v.clone() for k, v in layer.conv_states.items() if v is not None},
                {k: v.clone() for k, v in layer.recurrent_states.items() if v is not None},
                dict(layer.has_previous_state),
                dict(layer.is_conv_states_initialized),
                dict(layer.is_recurrent_states_initialized),
            ))
        else:
            snap.append(("empty",))
    return snap


def _restore_dynamic_cache(cache: Any, snap: list[Any]) -> None:
    """Restore an official Qwen3.5 DynamicCache to its exact pre-speculative snapshot."""
    for layer, s in zip(cache.layers, snap):
        if s[0] == "dynamic":
            layer.keys = s[1].clone()
            layer.values = s[2].clone()
        elif s[0] == "linear":
            for k, c in s[1].items():
                if k in layer.conv_states and layer.conv_states[k] is not None and layer.conv_states[k].shape == c.shape:
                    layer.conv_states[k].copy_(c)
                else:
                    layer.conv_states[k] = c.clone()
            for k, r in s[2].items():
                if k in layer.recurrent_states and layer.recurrent_states[k] is not None and layer.recurrent_states[k].shape == r.shape:
                    layer.recurrent_states[k].copy_(r)
                else:
                    layer.recurrent_states[k] = r.clone()
            layer.has_previous_state = dict(s[3])
            layer.is_conv_states_initialized = dict(s[4])
            layer.is_recurrent_states_initialized = dict(s[5])


def _load_official_hf_bundle(ckpt_path: Path, device: str) -> dict[str, Any]:
    """Load official Qwen3_5ForConditionalGeneration + 15-tensor Qwen35MTPModule + AutoProcessor."""
    cache_key = (str(ckpt_path), str(device))
    if cache_key in _OFFICIAL_HF_BUNDLE_CACHE:
        return _OFFICIAL_HF_BUNDLE_CACHE[cache_key]

    from safetensors import safe_open
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

    # Select hardware-native dtype: BF16 on sm_80+ (A100), FP32 on sm_61/CPU
    use_bf16 = (
        str(device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    shards = sorted(ckpt_path.glob("*.safetensors"))
    all_sf_keys: dict[str, int] = {}
    mtp_sd: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as sf:
            for k in sf.keys():
                t = sf.get_tensor(k)
                all_sf_keys[k] = int(t.numel())
                if k.startswith("mtp."):
                    mtp_sd[k[len("mtp."):]] = t.to(dtype=dtype)

    hf_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(ckpt_path),
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    mtp_module = Qwen35MTPModule(hf_model.config.text_config)
    mtp_module.load_state_dict(mtp_sd, strict=True)
    mtp_module = mtp_module.to(device=device, dtype=dtype).eval()

    processor = AutoProcessor.from_pretrained(str(ckpt_path))

    # Compute exact tensor & parameter coverage inventory (P0-01)
    hf_sd_keys = set(hf_model.state_dict().keys())
    mapped_base_keys = [k for k in all_sf_keys if k in hf_sd_keys]
    mapped_mtp_keys = [k for k in all_sf_keys if k.startswith("mtp.") and k[len("mtp."):] in mtp_sd]
    mapped_keys_set = set(mapped_base_keys) | set(mapped_mtp_keys)
    unmapped_keys = sorted(k for k in all_sf_keys if k not in mapped_keys_set)

    language_keys = [k for k in mapped_base_keys if "language_model" in k]
    visual_keys = [k for k in mapped_base_keys if "visual" in k]

    base_vl_params = sum(all_sf_keys[k] for k in mapped_base_keys)
    mtp_params = sum(all_sf_keys[k] for k in mapped_mtp_keys)
    total_sf_params = sum(all_sf_keys.values())
    mapped_params = base_vl_params + mtp_params
    unmapped_params = total_sf_params - mapped_params

    # Verify one-step logits parity before and after installing attention hooks (P0-01)
    parity_ids = torch.tensor([[100, 101, 102, 103]], device=device, dtype=torch.long)
    with torch.no_grad():
        ref_out = hf_model(input_ids=parity_ids, use_cache=False)
        ref_logits = ref_out.logits.detach().clone()

    hook_ctx = _HookStepContext()
    _install_seven_arm_attention_hooks(hf_model, hook_ctx)

    with torch.no_grad():
        hook_ctx.da_masks = None
        hook_ctx.h2o_eviction_plans = None
        hook_ctx.attention_matrices = {}
        hook_ctx.layer_telemetries = {}
        hooked_out = hf_model(input_ids=parity_ids, use_cache=False)
        hooked_logits = hooked_out.logits.detach()

    max_abs_diff = float((hooked_logits - ref_logits).abs().max().item())
    top1_official = int(torch.argmax(ref_logits[:, -1, :], dim=-1).item())
    top1_hooked = int(torch.argmax(hooked_logits[:, -1, :], dim=-1).item())
    parity_verified = bool(max_abs_diff <= 1e-5 and top1_official == top1_hooked)

    summary = {
        "checkpoint_dir": str(ckpt_path),
        "architectural_mode": "OFFICIAL_HF_QWEN3_5_HOOKED",
        "is_official_hf_hooked": True,
        "checkpoint_total_tensors": len(all_sf_keys),
        "loaded_tensors_count": len(mapped_keys_set),
        "mapped_tensors_count": len(mapped_keys_set),
        "unmapped_tensors_count": len(unmapped_keys),
        "unmapped_tensor_keys": unmapped_keys,
        "base_vl_tensors_loaded": len(mapped_base_keys),
        "language_tensors_loaded": len(language_keys),
        "visual_tensors_loaded": len(visual_keys),
        "mtp_tensors_loaded": len(mapped_mtp_keys),
        "base_vl_parameters": base_vl_params,
        "mtp_parameters": mtp_params,
        "total_checkpoint_parameters": total_sf_params,
        "mapped_parameters": mapped_params,
        "unmapped_parameters": unmapped_params,
        "tensor_coverage_ratio": round(len(mapped_keys_set) / max(1, len(all_sf_keys)), 6),
        "parameter_coverage_ratio": round(mapped_params / max(1, total_sf_params), 6),
        "logits_parity": {
            "verified": parity_verified,
            "max_abs_logits_diff": max_abs_diff,
            "top1_token_id_official": top1_official,
            "top1_token_id_hooked": top1_hooked,
            "top1_match": top1_official == top1_hooked,
        },
    }

    bundle = {
        "hf_model": hf_model,
        "mtp_module": mtp_module,
        "processor": processor,
        "hook_ctx": hook_ctx,
        "lock": threading.Lock(),
        "summary": summary,
    }
    _OFFICIAL_HF_BUNDLE_CACHE[cache_key] = bundle
    return bundle


# --- 1. Reference PyTorch Qwen3.5 Architectural Layers (Scaffold & Unit-Test Fallback) ---

class DeltaNetRecurrentLayer(nn.Module):
    """Linear Attention / Gated DeltaNet layer with fixed-size recurrent state."""

    def __init__(self, config: Qwen35ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim

        self.norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        recurrent_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass updating fixed-size linear recurrent state."""
        hidden_states = self.norm(hidden_states)
        b, q_len, _ = hidden_states.shape
        if recurrent_state is None:
            recurrent_state = torch.zeros(
                (b, self.num_heads, self.head_dim, self.head_dim),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )

        k = self.k_proj(hidden_states).view(b, q_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(b, q_len, self.num_heads, self.head_dim)
        delta = torch.einsum("bnhd,bnhe->bhde", k, v)
        new_state = torch.tanh(0.9 * recurrent_state + delta * 0.05)

        q = self.q_proj(hidden_states).view(b, q_len, self.num_heads, self.head_dim)
        out = torch.einsum("bnhd,bhde->bnhe", q, new_state).reshape(b, q_len, -1)
        return torch.tanh(self.out_proj(out)), new_state


class QwenFullAttentionLayer(nn.Module):
    """Gated Full-Attention Layer with GQA and physical KV-cache management."""

    def __init__(self, config: Qwen35ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        self.norm = nn.LayerNorm(self.hidden_size)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        da_mask: torch.Tensor | None = None,
        h2o_retained_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor, dict[str, Any]]:
        """Forward pass with physical KV management and attention tracking."""
        hidden_states = self.norm(hidden_states)
        b, q_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            pk, pv = past_key_value
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        kv_bytes_before = (k.nelement() + v.nelement()) * k.element_size()
        physical_bytes_freed = 0

        if h2o_retained_indices is not None and len(h2o_retained_indices) < k.shape[2]:
            k = torch.index_select(k, dim=2, index=h2o_retained_indices).contiguous()
            v = torch.index_select(v, dim=2, index=h2o_retained_indices).contiguous()
            kv_bytes_after = (k.nelement() + v.nelement()) * k.element_size()
            physical_bytes_freed = kv_bytes_before - kv_bytes_after

        k_len = k.shape[2]

        repeat_factor = self.num_heads // self.num_kv_heads
        k_rep = k.repeat_interleave(repeat_factor, dim=1)
        v_rep = v.repeat_interleave(repeat_factor, dim=1)

        scale = 1.0 / np.sqrt(self.head_dim)
        attn_logits = torch.matmul(q, k_rep.transpose(-1, -2)) * scale

        if da_mask is not None:
            if da_mask.shape != (q_len, k_len):
                da_mask_slice = da_mask[-q_len:, :k_len]
            else:
                da_mask_slice = da_mask
            mask_expanded = da_mask_slice.unsqueeze(0).unsqueeze(0)
            attn_logits = attn_logits.masked_fill(~mask_expanded, -1e9)

        attn_weights = F.softmax(attn_logits, dim=-1)

        context = torch.matmul(attn_weights, v_rep).transpose(1, 2).reshape(b, q_len, -1)
        out = self.out_proj(context)

        telemetry = {
            "kv_seq_len": k_len,
            "kv_bytes_allocated": (k.nelement() + v.nelement()) * k.element_size(),
            "physical_bytes_freed": physical_bytes_freed,
        }
        return out, (k, v), attn_weights, telemetry


class Qwen35NeuralModel(nn.Module):
    """Qwen3.5 vision-language neural backbone supporting both official HF hooking and scaffold fallback."""

    def __init__(self, config: Qwen35ModelConfig | None = None, allocate_scaffold: bool = True):
        super().__init__()
        self.config = config or Qwen35ModelConfig.qwen35_08b()
        self.vocab_size = self.config.vocab_size

        # Official HF hooked attributes (populated when loading a full Hugging Face Qwen3.5 checkpoint)
        self.hf_model: Any | None = None
        self.mtp_module: Qwen35MTPModule | None = None
        self.processor: Any | None = None
        self.hook_ctx: _HookStepContext | None = None
        self._hf_lock: threading.Lock | None = None
        self.is_official_hf_hooked: bool = False

        # Reference PyTorch scaffold layers (only allocated when unweighted or partial synthetic checkpoint)
        if allocate_scaffold:
            self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
            self.layers = nn.ModuleList()

            for i in range(self.config.num_hidden_layers):
                if i in self.config.full_attention_layers:
                    self.layers.append(QwenFullAttentionLayer(self.config, layer_idx=i))
                else:
                    self.layers.append(DeltaNetRecurrentLayer(self.config, layer_idx=i))

            self.final_norm = nn.LayerNorm(self.config.hidden_size)
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            self.mtp_proj = nn.Linear(
                self.config.hidden_size,
                self.config.hidden_size * self.config.mtp_num_speculative_tokens,
                bias=False,
            )
            self.visual_merger = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=True)

    @property
    def hf_forward_active(self) -> bool:
        """True when forward() runs the attached official HF model (R4-4).

        A config.json checkpoint attaches the HF model even when coverage is partial. Cache layout
        and decode position handling must follow THIS flag, not the qualification flag.
        """
        return self.hf_model is not None and self.hook_ctx is not None

    @property
    def forward_path_label(self) -> str:
        if not self.hf_forward_active:
            return "scaffold_reference"
        return "official_hf_hooked" if self.is_official_hf_hooked else "official_hf_partial_coverage_UNQUALIFIED"

    def load_pretrained_weights(
        self,
        checkpoint_dir: str | Path,
        device: str = "cpu",
        prefer_official_hf: bool = True,
    ) -> dict[str, Any]:
        """Attach official hooked Qwen3_5ForConditionalGeneration + Qwen35MTPModule (100% coverage)
        when a full HF checkpoint is present, or fallback to partial reference loading for synthetic tests.
        """
        ckpt_path = Path(checkpoint_dir).resolve()
        if not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

        shards = sorted(ckpt_path.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No .safetensors weight shards found in {ckpt_path}")

        # 1. Full Official Hugging Face Qwen3.5 Checkpoint Hooking (P0-01: 473+15=488 tensors, 100% coverage)
        if prefer_official_hf and (ckpt_path / "config.json").is_file():
            bundle = _load_official_hf_bundle(ckpt_path, device=device)
            self.hf_model = bundle["hf_model"]
            self.mtp_module = bundle["mtp_module"]
            self.processor = bundle["processor"]
            self.hook_ctx = bundle["hook_ctx"]
            self._hf_lock = bundle["lock"]
            self.is_official_hf_hooked = bool(
                bundle["summary"]["tensor_coverage_ratio"] == 1.0
                and bundle["summary"]["parameter_coverage_ratio"] == 1.0
                and bundle["summary"]["logits_parity"]["verified"]
            )
            return dict(bundle["summary"])

        # 2. Partial Reference Fallback (for synthetic tiny safetensors tests without config.json)
        cache_key = (
            str(ckpt_path),
            self.config.vocab_size,
            self.config.hidden_size,
            self.config.num_hidden_layers,
            self.config.head_dim,
        )
        if cache_key in _PRETRAINED_STATE_CACHE:
            cached_sd, cached_summary = _PRETRAINED_STATE_CACHE[cache_key]
            self.load_state_dict(cached_sd, strict=True)
            self.is_official_hf_hooked = False
            return dict(cached_summary)

        from safetensors import safe_open

        language_loaded = 0
        mtp_loaded = 0
        visual_loaded = 0
        all_keys: set[str] = set()
        mapped_keys: set[str] = set()
        total_sf_params = 0
        mapped_sf_params = 0
        hidden_size = self.config.hidden_size
        dtype = self.embed_tokens.weight.dtype

        with torch.no_grad():
            for shard in shards:
                with safe_open(str(shard), framework="pt") as sf:
                    keys = set(sf.keys())
                    all_keys.update(keys)
                    for k in keys:
                        total_sf_params += int(sf.get_tensor(k).numel())

                    if "model.language_model.embed_tokens.weight" in keys:
                        emb = sf.get_tensor("model.language_model.embed_tokens.weight").to(dtype=dtype)
                        v_rows = min(self.embed_tokens.weight.shape[0], emb.shape[0])
                        h_cols = min(hidden_size, emb.shape[1])
                        self.embed_tokens.weight[:v_rows, :h_cols].copy_(emb[:v_rows, :h_cols])
                        self.lm_head.weight[:v_rows, :h_cols].copy_(emb[:v_rows, :h_cols])
                        language_loaded += 2
                        mapped_keys.add("model.language_model.embed_tokens.weight")
                        mapped_sf_params += int(emb.numel())

                    if "model.language_model.norm.weight" in keys:
                        norm_w = sf.get_tensor("model.language_model.norm.weight").to(dtype=dtype)
                        self.final_norm.weight[: min(hidden_size, len(norm_w))].copy_(norm_w[:hidden_size])
                        self.final_norm.bias.zero_()
                        language_loaded += 1
                        mapped_keys.add("model.language_model.norm.weight")
                        mapped_sf_params += int(norm_w.numel())

                    # Projection tensors are mapped ONLY on exact shape match. A mismatched
                    # tensor stays unmapped so coverage cannot be inflated by silent slicing.
                    if "mtp.fc.weight" in keys:
                        mtp_fc = sf.get_tensor("mtp.fc.weight").to(dtype=dtype)
                        mtp_t = mtp_fc.t()
                        if tuple(mtp_t.shape) == tuple(self.mtp_proj.weight.shape):
                            self.mtp_proj.weight.copy_(mtp_t)
                            mtp_loaded += 1
                            mapped_keys.add("mtp.fc.weight")
                            mapped_sf_params += int(mtp_fc.numel())

                    if "model.visual.merger.linear_fc2.weight" in keys:
                        vis_w = sf.get_tensor("model.visual.merger.linear_fc2.weight").to(dtype=dtype)
                        if tuple(vis_w.shape) == tuple(self.visual_merger.weight.shape):
                            self.visual_merger.weight.copy_(vis_w)
                            visual_loaded += 1
                            mapped_keys.add("model.visual.merger.linear_fc2.weight")
                            mapped_sf_params += int(vis_w.numel())
                    if "model.visual.merger.linear_fc2.bias" in keys:
                        vis_b = sf.get_tensor("model.visual.merger.linear_fc2.bias").to(dtype=dtype)
                        if tuple(vis_b.shape) == tuple(self.visual_merger.bias.shape):
                            self.visual_merger.bias.copy_(vis_b)
                            visual_loaded += 1
                            mapped_keys.add("model.visual.merger.linear_fc2.bias")
                            mapped_sf_params += int(vis_b.numel())

                    for i, layer in enumerate(self.layers):
                        ln_key = f"model.language_model.layers.{i}.input_layernorm.weight"
                        if ln_key in keys:
                            ln_w = sf.get_tensor(ln_key).to(dtype=dtype)
                            layer.norm.weight.copy_(ln_w[:hidden_size])
                            layer.norm.bias.zero_()
                            language_loaded += 1
                            mapped_keys.add(ln_key)
                            mapped_sf_params += int(ln_w.numel())

                        if isinstance(layer, QwenFullAttentionLayer):
                            q_dim = layer.num_heads * layer.head_dim
                            kv_dim = layer.num_kv_heads * layer.head_dim
                            for proj_attr, key_suffix, rows, cols in (
                                ("q_proj", "q_proj.weight", q_dim, hidden_size),
                                ("k_proj", "k_proj.weight", kv_dim, hidden_size),
                                ("v_proj", "v_proj.weight", kv_dim, hidden_size),
                                ("out_proj", "o_proj.weight", hidden_size, q_dim),
                            ):
                                full_k = f"model.language_model.layers.{i}.self_attn.{key_suffix}"
                                if full_k in keys:
                                    t = sf.get_tensor(full_k).to(dtype=dtype)
                                    if tuple(t.shape) != (rows, cols):
                                        continue  # shape mismatch: leave unmapped, never slice
                                    getattr(layer, proj_attr).weight.copy_(t)
                                    language_loaded += 1
                                    mapped_keys.add(full_k)
                                    mapped_sf_params += int(t.numel())
                        else:
                            r_dim = layer.num_heads * layer.head_dim
                            qkv_key = f"model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight"
                            out_key = f"model.language_model.layers.{i}.linear_attn.out_proj.weight"
                            if qkv_key in keys:
                                qkv = sf.get_tensor(qkv_key).to(dtype=dtype)
                                if tuple(qkv.shape) == (3 * r_dim, hidden_size):
                                    stride = r_dim
                                    layer.q_proj.weight.copy_(qkv[:r_dim, :])
                                    layer.k_proj.weight.copy_(qkv[stride : stride + r_dim, :])
                                    layer.v_proj.weight.copy_(qkv[2 * stride : 2 * stride + r_dim, :])
                                    language_loaded += 3
                                    mapped_keys.add(qkv_key)
                                    mapped_sf_params += int(qkv.numel())
                            if out_key in keys:
                                t_out = sf.get_tensor(out_key).to(dtype=dtype)
                                if tuple(t_out.shape) == (hidden_size, r_dim):
                                    layer.out_proj.weight.copy_(t_out)
                                    language_loaded += 1
                                    mapped_keys.add(out_key)
                                    mapped_sf_params += int(t_out.numel())

        total_loaded = language_loaded + mtp_loaded + visual_loaded
        if total_loaded == 0:
            raise ValueError(f"No matching Qwen3.5 weights found in checkpoint: {ckpt_path}")

        unmapped_keys = sorted(all_keys - mapped_keys)
        self.is_official_hf_hooked = False
        summary = {
            "checkpoint_dir": str(ckpt_path),
            "architectural_mode": "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE",
            "is_official_hf_hooked": False,
            "checkpoint_total_tensors": len(all_keys),
            "loaded_tensors_count": total_loaded,
            "mapped_tensors_count": len(mapped_keys),
            "unmapped_tensors_count": len(unmapped_keys),
            "unmapped_tensor_keys": unmapped_keys,
            "language_tensors_loaded": language_loaded,
            "mtp_tensors_loaded": mtp_loaded,
            "visual_tensors_loaded": visual_loaded,
            "total_checkpoint_parameters": total_sf_params,
            "mapped_parameters": mapped_sf_params,
            "unmapped_parameters": max(0, total_sf_params - mapped_sf_params),
            "tensor_coverage_ratio": round(len(mapped_keys) / max(1, len(all_keys)), 6),
            "parameter_coverage_ratio": round(mapped_sf_params / max(1, total_sf_params), 6),
            "logits_parity": {
                "verified": False,
                "reason": "Partial reference scaffold without full HF config/MLP weights",
            },
        }
        if len(_PRETRAINED_STATE_CACHE) < 2:
            _PRETRAINED_STATE_CACHE[cache_key] = (
                {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
                dict(summary),
            )
        return summary

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        recurrent_states: list[Any] | None = None,
        da_masks: dict[int, torch.Tensor] | None = None,
        h2o_eviction_plans: dict[int, torch.Tensor] | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        mm_encoder_outputs: dict[str, Any] | None = None,
        compute_mtp: bool = False,
        mtp_position_ids: torch.Tensor | None = None,
        logits_to_keep: int = 1,
    ) -> dict[str, Any]:
        """Full forward pass routing through hooked official HF model when attached, or scaffold layers.

        `logits_to_keep` = number of trailing positions whose logits are returned. `logits`,
        `full_logits[:, -1]` and the MTP draft all refer to the LAST kept position.
        """
        if type(logits_to_keep) is not int or logits_to_keep < 1:
            raise ValueError("logits_to_keep must be a positive integer")
        if self.hf_forward_active:
            from transformers.models.qwen3_5 import modeling_qwen3_5

            lock = self._hf_lock or threading.Lock()
            with lock:
                if past_key_values is None:
                    past_key_values = modeling_qwen3_5.DynamicCache(config=self.hf_model.config)

                self.hook_ctx.da_masks = da_masks
                self.hook_ctx.h2o_eviction_plans = h2o_eviction_plans
                self.hook_ctx.attention_matrices = {}
                self.hook_ctx.layer_telemetries = {}

                out = self.hf_model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    mm_encoder_outputs=mm_encoder_outputs,
                    logits_to_keep=logits_to_keep,
                    use_cache=True,
                    output_hidden_states=True,
                )

                logits = out.logits[:, -1, :].float()
                last_hidden = out.hidden_states[-1][:, -1:, :]
                mtp_logits = None
                if compute_mtp:
                    next_tok_id = torch.argmax(logits, dim=-1, keepdim=True)
                    next_tok_emb = self.hf_model.model.get_input_embeddings()(next_tok_id)
                    mtp_logits = self.mtp_module(
                        last_hidden, next_tok_emb, self.hf_model.lm_head,
                        position_ids=mtp_position_ids,
                    ).float()

                attn_matrices = dict(self.hook_ctx.attention_matrices)
                hook_telems = dict(self.hook_ctx.layer_telemetries)

                layer_telemetries: list[dict[str, Any]] = []
                for idx in range(self.config.num_hidden_layers):
                    if idx in hook_telems:
                        layer_telemetries.append(hook_telems[idx])
                    else:
                        layer_telemetries.append({"type": "recurrent_deltanet", "kv_bytes_allocated": 0, "physical_bytes_freed": 0})

                return {
                    "forward_path": self.forward_path_label,
                    "logits": logits,
                    "full_logits": out.logits.float(),
                    "mtp_logits": mtp_logits,
                    "past_key_values": out.past_key_values,
                    "recurrent_states": recurrent_states,
                    "attention_matrices": attn_matrices,
                    "layer_telemetries": layer_telemetries,
                }

        # Reference PyTorch scaffold path (last position only; multi-position verify is official-path only)
        if logits_to_keep != 1:
            raise ValueError("scaffold forward supports logits_to_keep=1 only")
        b, seq_len = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        vis_mask = input_ids == self.config.image_token_id
        if vis_mask.any():
            hidden = torch.where(vis_mask.unsqueeze(-1), self.visual_merger(hidden), hidden)

        new_past_kv: list[Any] = []
        new_recurrent: list[Any] = []
        layer_telemetries = []
        attention_matrices: dict[int, torch.Tensor] = {}

        for idx, layer in enumerate(self.layers):
            if isinstance(layer, QwenFullAttentionLayer):
                pk = past_key_values[idx] if past_key_values and idx < len(past_key_values) else None
                da_m = da_masks.get(idx) if da_masks else None
                h2o_indices = h2o_eviction_plans.get(idx) if h2o_eviction_plans else None

                h_out, pkv, attn_w, telem = layer(
                    hidden,
                    past_key_value=pk,
                    da_mask=da_m,
                    h2o_retained_indices=h2o_indices,
                )
                hidden = hidden + h_out
                new_past_kv.append(pkv)
                new_recurrent.append(None)
                layer_telemetries.append(telem)
                attention_matrices[idx] = attn_w
            else:
                rec_s = recurrent_states[idx] if recurrent_states and idx < len(recurrent_states) else None
                h_out, n_rec = layer(hidden, recurrent_state=rec_s)
                hidden = hidden + h_out
                new_past_kv.append(None)
                new_recurrent.append(n_rec)
                layer_telemetries.append({"type": "recurrent_deltanet", "kv_bytes_allocated": 0})

        normed_hidden = self.final_norm(hidden[:, -1, :])
        logits = self.lm_head(normed_hidden)
        mtp_logits = None
        if compute_mtp:
            mtp_hidden = self.mtp_proj(normed_hidden).view(
                b, self.config.mtp_num_speculative_tokens, self.config.hidden_size
            )
            mtp_logits = self.lm_head(mtp_hidden)

        return {
            "forward_path": self.forward_path_label,
            "logits": logits,
            "mtp_logits": mtp_logits,
            "past_key_values": new_past_kv,
            "recurrent_states": new_recurrent,
            "attention_matrices": attention_matrices,
            "layer_telemetries": layer_telemetries,
        }


DA_DECODE_ONLY_LABEL = "decode_only_scripted_da_diagnostic"


def _round_or_none(value: float | None, ndigits: int = 4) -> float | None:
    return None if value is None else round(value, ndigits)


def _da_masking_gate(
    method_arm: str,
    restriction_ratio: float | None,
    measured_masked_entries: int | None,
    measured_masked_weight_max: float | None,
) -> str:
    """DA_MASKING_EXERCISED verdict (R4-3).

    PASS needs all three: a constructed restriction (> 0), masked entries MEASURED in the hooked
    forward, and every measured masked weight exactly 0.0. A constructed mask alone never passes.
    """
    if method_arm not in DA_MASKING_ARMS:
        return "NOT_APPLICABLE"
    if restriction_ratio is not None and restriction_ratio == 0.0:
        return "NOT_EXERCISED_NO_RESTRICTED_READS"
    if restriction_ratio is None or not measured_masked_entries or measured_masked_weight_max is None:
        return "NOT_MEASURED_NO_HOOKED_DECODE_STEP"
    if measured_masked_weight_max != 0.0:
        return "FAIL_MASKED_KEY_RECEIVED_ATTENTION"
    return "PASS"


def _da_scripted_tags(
    anchor_len: int, phys_len: int, n_visual: int, image_present: bool,
    question_start: int | None = None,
) -> list[DATag]:
    """SCRIPTED DA diagnostic geometry, fixed from the prompt anchor.

    Sink, focus and visual-segment bounds are computed ONCE from the prefill length, so
    the prompt region keeps the same tag class at every decode step. Only the trailing
    LOCAL window moves with the generated suffix (phys_len). Deriving the bounds from the
    growing KV length instead reclassified prompt keys between steps.

    DOCUMENTED DEVIATION: when question_start is given (image prompts: the span after the
    last vision_end token), that question/instruction span is GLOBAL, so every query can
    read it. Without this the LOCAL window hid question tokens from the newest query.
    Text-only prompts have no question split, so no deviation is applied to them.
    """
    sink_end = min(16, max(1, anchor_len // 4))
    if not n_visual and image_present:
        n_visual = min(4, max(0, anchor_len - sink_end - 1))
    tags = [DATag(DATagType.GLOBAL, start_idx=0, end_idx=sink_end)]
    if question_start is not None and question_start < anchor_len:
        tags.append(DATag(DATagType.GLOBAL, start_idx=question_start, end_idx=anchor_len))
    if anchor_len > sink_end + 1:
        focus_end = min(anchor_len - 1, sink_end + max(1, n_visual))
        tags.append(DATag(
            DATagType.FOCUS, start_idx=sink_end, end_idx=focus_end,
            focus_id="scripted_visual_segment",
        ))
        tags.append(DATag(DATagType.LOCAL, start_idx=focus_end, end_idx=phys_len, window=8))
    return tags


# STARLIGHT Stage 1 (cpu_pytorch_reference_not_vllm): arm -> enabled controller signals.
# The seven original arms are not in this map; their behaviour is unchanged.
STARLIGHT_ARM_SIGNALS: dict[str, tuple[str, ...]] = {
    "starlight_s1": ("da", "grounding", "structural"),
    "starlight_s1_no_da": ("grounding", "structural"),
    "starlight_s1_no_grounding": ("da", "structural"),
}
STARLIGHT_MEASUREMENT_LABEL = "cpu_pytorch_reference_not_vllm"
# Arms that run the DA attention mask. Exact membership, never a substring test: the name
# "starlight_s1_no_da" contains "da" but is a protection-only ablation and must not build a mask.
DA_MASKING_ARMS: tuple[str, ...] = ("da_base", "da_mtp")
# Oracle reason codes for evidence verdicts (HC-4, HC-4b): status -> closed code.
EVIDENCE_REASON_CODES: dict[str, str] = {
    "valid": "VALID",
    "expired": "EXPIRED_EVIDENCE",
    "stale": "STALE_DECLARATION",
    "unverified": "UNVERIFIED_EVIDENCE",
}
GDN_EXEMPT_LABEL = "GDN_EXEMPT_NOT_TOKEN_ADDRESSABLE"
# Lead decision Q2: vision_start/vision_end are protected as an explicit structural source.
STRUCTURAL_POLICY_LABEL = "PROTECTED_LEAD_DECISION_Q2"


def _image_sha256(image_bytes: bytes | None, image_path: str | Path | None) -> str | None:
    """SHA256 of the request's image source, or None when no source is available."""
    if image_bytes is not None:
        return hashlib.sha256(bytes(image_bytes)).hexdigest()
    if image_path is not None:
        return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    return None


# --- 2. Production Seven-Arm Engine with True Backend Interventions ---

@dataclass
class RealExecutionResult:
    """Telemetry and outputs of real model execution across the 7 arms."""
    method_arm: str
    tokens_generated: list[int]
    output_text: str
    total_latency_ms: float
    ttft_ms: float
    tpot_ms: float
    physical_kv_bytes_allocated: int
    physical_kv_bytes_freed: int
    per_step_evicted_slot_rate: float | None  # H2O: last decode step only; None if no H2O step ran
    attention_read_restriction_ratio: float | None  # None when no decode step built a DA mask
    grounded_citations: list[dict[str, Any]]
    mtp_acceptance_rate: float | None  # None = not verified (serving arm); see _speculative_mtp_greedy_k1
    attention_mask_restriction_ratio: float | None = None
    gates: dict[str, str] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


class RealSevenArmEngine:
    """Production execution engine implementing the 7 Multimodal Benchmarking arms.

    Operates with official hooked `Qwen3_5ForConditionalGeneration` + `Qwen35MTPModule`
    (100% parameter coverage and verified logits parity) when a full checkpoint is loaded,
    or falls back to the reference PyTorch scaffold for synthetic unit tests.
    """

    SUPPORTED_ARMS = frozenset((
        "base",
        "h2o_base",
        "da_base",
        "ecrd_base",
        "mtp",
        "h2o_mtp",
        "da_mtp",
        "starlight_s1",
        "starlight_s1_no_da",
        "starlight_s1_no_grounding",
    ))

    def __init__(
        self,
        method_arm: str,
        config: Qwen35ModelConfig | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        h2o_budget_ratio: float = 0.5,
        target_backend: str = "pytorch",
        h2o_min_budget: int = 32,
        stop_at_eos: bool = True,
        eos_token_ids: Sequence[int] | None = None,
        tokenizer: Any | None = None,
        checkpoint_dir: str | Path | None = None,
        starlight_record_decisions: bool = False,
    ):
        if method_arm not in self.SUPPORTED_ARMS:
            raise ValueError(f"Unknown method arm: {method_arm}")

        self.method_arm = method_arm
        self.config = config or Qwen35ModelConfig.qwen35_08b()
        # Protocol (methods.py quality track, native_full_context): natural EOS stop. Latency
        # runs must pass stop_at_eos=False explicitly. EOS defaults to the pinned config's id.
        self.stop_at_eos = bool(stop_at_eos)
        self.eos_token_ids: tuple[int, ...] = tuple(
            int(t) for t in (eos_token_ids if eos_token_ids is not None else (self.config.eos_token_id,))
        )
        self.device = device
        self.backend = target_backend
        self.tokenizer = tokenizer
        self.checkpoint_dir = Path(checkpoint_dir).resolve() if checkpoint_dir is not None else None
        self.hw_profile = check_vllm_hardware_compatibility(target_backend=self.backend)

        # Architectural and backend status flags (SA-01 & P0-01 strict truth-in-labeling)
        self.is_pretrained: bool = False
        self.is_official_hf_hooked: bool = False
        self.has_vllm_backend: bool = False
        self.has_vision_encoder: bool = False
        self.pretrained_weight_summary: dict[str, Any] | None = None
        self.processor: Any | None = None

        is_full_hf_ckpt = self.checkpoint_dir is not None and (self.checkpoint_dir / "config.json").is_file()
        if not is_full_hf_ckpt and str(self.device).startswith("cuda") and _OFFICIAL_HF_BUNDLE_CACHE:
            # Evict cached official HF GPU bundle before allocating unweighted scaffold on shared GPU
            _OFFICIAL_HF_BUNDLE_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model = Qwen35NeuralModel(self.config, allocate_scaffold=not is_full_hf_ckpt)
        if self.checkpoint_dir is not None:
            self.pretrained_weight_summary = model.load_pretrained_weights(
                self.checkpoint_dir, device=self.device
            )
            self.is_pretrained = True
            self.is_official_hf_hooked = model.is_official_hf_hooked
            self.has_vision_encoder = bool(self.pretrained_weight_summary.get("visual_tensors_loaded", 0) > 0)
            self.processor = model.processor
            if self.tokenizer is None and self.processor is not None:
                self.tokenizer = getattr(self.processor, "tokenizer", None)

        if not self.is_official_hf_hooked:
            self.model = model.to(self.device).eval()
        else:
            self.model = model

        # Arm-specific components
        self.h2o_trackers: dict[str, H2OCacheTracker] = {}
        # min_budget is the floor of the per-layer budget. With a fixed budget it can dominate on
        # short prompts (see docs/H2O_DEVIATIONS.md). Default 32 is unsourced and kept as-is.
        self.h2o_config = H2OConfig(budget_ratio=h2o_budget_ratio, min_budget=h2o_min_budget)
        # STARLIGHT arms are protection-only ablations of the H2O tracker; None for the seven.
        self.starlight_signals: tuple[str, ...] | None = STARLIGHT_ARM_SIGNALS.get(method_arm)
        # Opt-in per-entry decision records (auditor oracle I-10). Off by default: records are large.
        self.starlight_record_decisions = bool(starlight_record_decisions)
        self.da_fsm = DAFSM() if method_arm in DA_MASKING_ARMS else None
        self.da_mask_gen = DAMaskGenerator() if method_arm in DA_MASKING_ARMS else None
        self.ecrd_processors: dict[str, ECRDLogitsProcessor] = {}

    def close(self) -> None:
        """Release trackers and temporary caches."""
        if hasattr(self, "model"):
            del self.model
        self.h2o_trackers.clear()
        self.ecrd_processors.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _get_or_create_h2o_tracker(self, request_id: str, reset: bool = True) -> H2OCacheTracker:
        if reset or request_id not in self.h2o_trackers:
            tracker = H2OCacheTracker(
                config=self.h2o_config,
                num_layers=self.config.num_hidden_layers,
                num_kv_heads=self.config.num_key_value_heads,
            )
            for l_idx in self.config.recurrent_layers:
                tracker.register_recurrent_layer(l_idx)
            self.h2o_trackers[request_id] = tracker
        return self.h2o_trackers[request_id]

    ECRD_ALPHA_RULE = "3.5 if official_hf_hooked and image_present else 20.0"

    def _ecrd_alpha(self, image_present: bool) -> float:
        """Condition-dependent ECRD boost. UNCALIBRATED: the two values differ by condition, so
        ECRD results from official+image runs and from other runs are NOT directly comparable.
        Do not retune against benchmark answers; disclosed in telemetry as the confounder."""
        return 3.5 if (self.is_official_hf_hooked and image_present) else 20.0

    def _get_or_create_ecrd_processor(
        self,
        request_id: str,
        pool: EvidencePool | None,
        image_present: bool = True,
    ) -> ECRDLogitsProcessor:
        class _LocalTokenizer:
            _WORDS = {
                100: "object",
                101: "region",
                102: "target",
                103: "apple",
                104: "cats",
                105: "cat",
                106: "couch",
                107: "sofa",
                248056: "<image>",
            }
            _INVERSE = {v: k for k, v in _WORDS.items()}

            def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
                return " ".join(self._WORDS.get(t, f"tok_{t}") for t in token_ids)

            def encode(self, text: str) -> list[int]:
                tokens: list[int] = []
                for word in text.split():
                    clean = word.strip().lower()
                    if clean in self._INVERSE:
                        tokens.append(self._INVERSE[clean])
                return tokens

        tok = self.tokenizer if self.tokenizer is not None else _LocalTokenizer()
        p = pool or EvidencePool(request_id)
        alpha = self._ecrd_alpha(image_present)
        proc = ECRDLogitsProcessor(
            evidence_pool=p,
            tokenizer=tok,
            config=ECRDConfig(reweight_alpha=alpha),
        )
        self.ecrd_processors[request_id] = proc
        return proc

    def _prepare_official_multimodal_inputs(
        self,
        prompt_token_ids: list[int],
        image_present: bool,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        question: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None, dict[str, Any]]:
        """Process real image bytes through official AutoProcessor and ViT encoder (P0-02)."""
        if not image_present or not self.is_official_hf_hooked or self.processor is None:
            full_prompt = list(prompt_token_ids)
            if image_present and self.config.image_token_id not in full_prompt:
                vision_block = (
                    [self.config.vision_start_token_id]
                    + [self.config.image_token_id] * 4
                    + [self.config.vision_end_token_id]
                )
                full_prompt = vision_block + full_prompt
            input_ids = torch.tensor([full_prompt], device=self.device, dtype=torch.long)
            return input_ids, None, None, None, {"invoked": False}

        from PIL import Image

        if image_bytes is not None:
            raw_bytes = image_bytes
            resolved_path = str(image_path) if image_path is not None else "in_memory_bytes"
        else:
            img_file = Path(image_path) if image_path is not None else _DEFAULT_FIXTURE_IMAGE
            raw_bytes = img_file.read_bytes()
            resolved_path = str(img_file)

        img_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        effective_question = question or "Are there cats lying on the couch in the living room?"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": effective_question},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        proc_inputs = self.processor(
            text=[prompt_text],
            images=[img],
            padding=True,
            return_tensors="pt",
        )

        pixel_values = proc_inputs["pixel_values"].to(
            device=self.device,
            dtype=self.model.hf_model.dtype,
        )
        image_grid_thw = proc_inputs["image_grid_thw"].to(device=self.device)

        # Invoke official ViT vision encoder (`model.visual`) on real image patches
        lock = self.model._hf_lock or threading.Lock()
        with lock, torch.no_grad():
            vis_out = self.model.hf_model.model.get_image_features(
                pixel_values, image_grid_thw, return_dict=True
            )
            vis_embeds = torch.cat(vis_out.pooler_output, dim=0)
            num_vis_tokens = int(vis_embeds.shape[0])

        if question is not None or not prompt_token_ids:
            input_ids = proc_inputs["input_ids"].to(device=self.device, dtype=torch.long)
            mm_token_type_ids = proc_inputs["mm_token_type_ids"].to(device=self.device)
            attention_mask = proc_inputs["attention_mask"].to(device=self.device)
        else:
            # Expand caller's prompt_token_ids so image placeholder count matches ViT embeddings count (300)
            clean_text_ids = [
                t for t in prompt_token_ids
                if t not in (
                    self.config.vision_start_token_id,
                    self.config.image_token_id,
                    self.config.vision_end_token_id,
                )
            ]
            full_prompt = (
                [self.config.vision_start_token_id]
                + [self.config.image_token_id] * num_vis_tokens
                + [self.config.vision_end_token_id]
                + clean_text_ids
            )
            input_ids = torch.tensor([full_prompt], device=self.device, dtype=torch.long)
            mm_token_type_ids = (input_ids == self.config.image_token_id).to(dtype=torch.int32)
            attention_mask = torch.ones_like(input_ids)

        with lock, torch.no_grad():
            pos_ids, rope_deltas = self.model.hf_model.model.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                attention_mask=attention_mask,
            )

        vis_trace = {
            "invoked": True,
            "image_path": resolved_path,
            "image_sha256": img_sha256,
            "pixel_values_shape": list(pixel_values.shape),
            "image_grid_thw": image_grid_thw.detach().cpu().tolist(),
            "visual_token_embeddings_count": num_vis_tokens,
            "visual_embeddings_shape": list(vis_embeds.shape),
            "image_placeholder_tokens_in_prompt": int((input_ids == self.config.image_token_id).sum().item()),
        }
        return input_ids, pos_ids, rope_deltas, {"image": vis_out}, vis_trace

    CPU_GREEDY_K1_LABEL = "cpu_greedy_k1_reference_not_vllm_mtp"
    # Test hook for the H-6 cache-length check (STARLIGHT). Must stay 0 outside tests.
    _test_cache_len_offset: int = 0

    def _speculative_mtp_greedy_k1(
        self,
        prompt_ids: Sequence[int],
        max_new_tokens: int,
        draft_override: Any = None,
        t0: float | None = None,
    ) -> dict[str, Any]:
        """CPU greedy k=1 speculative decoding with the official MTP head (reference, NOT vLLM MTP).

        Invariant: the committed stream is always the target's greedy stream. Each step, the MTP
        head proposes one draft d for the token after the pending token. One target forward over
        [pending, d] from a state snapshot verifies it: accept iff d == target argmax after pending.
        On reject, the snapshot (KV and Gated-DeltaNet conv/recurrent states) is restored and the
        pending token is re-fed alone, so the next committed token comes from a single-token target
        forward.

        `draft_override(draft, proposal_index, n_emitted) -> int` is a test hook that replaces the
        proposal. `n_emitted` = tokens already committed; an oracle hook can use it to propose the
        target's known next token (forced accept) or a corrupted one (forced reject).
        No speed is claimed. Acceptance is verifier-derived: accepted / proposed.
        """
        if not self.is_official_hf_hooked or getattr(self.model, "mtp_module", None) is None:
            raise RuntimeError(
                "cpu_greedy_k1 reference requires the official hooked checkpoint with its MTP head"
            )
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        ids = [int(t) for t in prompt_ids]
        if not ids:
            raise ValueError("prompt_ids must be non-empty")
        vocab = self.config.vocab_size
        from transformers.models.qwen3_5 import modeling_qwen3_5

        cache = modeling_qwen3_5.DynamicCache(config=self.model.hf_model.config)

        def _pos(start: int, n: int) -> torch.Tensor:
            return (
                torch.arange(start, start + n, device=self.device, dtype=torch.long)
                .view(1, 1, n).expand(3, 1, n).clone()
            )

        def _feed(tokens: list[int], start: int, keep: int) -> dict[str, Any]:
            # MTP position = absolute position of the next target token (after the fed chunk).
            return self.model(
                torch.tensor([tokens], device=self.device, dtype=torch.long),
                past_key_values=cache,
                position_ids=_pos(start, len(tokens)),
                compute_mtp=True,
                mtp_position_ids=_pos(start + len(tokens), 1),
                logits_to_keep=keep,
            )

        stops: set[int] = set(self.eos_token_ids) if self.stop_at_eos else set()
        t_start = time.perf_counter() if t0 is None else t0
        first_commit_ms: float | None = None
        verifier_calls = 0
        with torch.no_grad():
            out = _feed(ids, 0, 1)
            verifier_calls += 1  # prefill is a target forward
            cache_len = len(ids)  # tokens currently in the cache; `pending` is not yet fed
            pending = int(torch.argmax(out["logits"][0]).item())
            draft = int(torch.argmax(out["mtp_logits"][0, 0]).item())

            emitted: list[int] = []
            proposed = accepted = rejected = 0
            stop_reason = "max_new_tokens"
            while True:
                if pending in stops:
                    stop_reason = "eos"  # EOS is never emitted into the output
                    break
                emitted.append(pending)
                if first_commit_ms is None:
                    first_commit_ms = (time.perf_counter() - t_start) * 1000.0
                if len(emitted) >= max_new_tokens:
                    break
                proposal = draft
                if draft_override is not None:
                    proposal = int(draft_override(draft, proposed, len(emitted)))
                if not 0 <= proposal < vocab:
                    raise ValueError("draft proposal outside the vocabulary")
                proposed += 1
                snap = _snapshot_dynamic_cache(cache)
                v = _feed([pending, proposal], cache_len, 2)
                verifier_calls += 1
                target_after_pending = int(torch.argmax(v["full_logits"][0, 0]).item())
                if proposal == target_after_pending:
                    accepted += 1
                    cache_len += 2
                    if proposal in stops:
                        # The verifier agreed the next target token is EOS: stop, do not emit it.
                        stop_reason = "eos"
                        break
                    emitted.append(proposal)
                    if len(emitted) >= max_new_tokens:
                        break
                    pending = int(torch.argmax(v["logits"][0]).item())
                    draft = int(torch.argmax(v["mtp_logits"][0, 0]).item())
                else:
                    rejected += 1
                    _restore_dynamic_cache(cache, snap)
                    v1 = _feed([pending], cache_len, 1)
                    verifier_calls += 1
                    cache_len += 1
                    pending = int(torch.argmax(v1["logits"][0]).item())
                    draft = int(torch.argmax(v1["mtp_logits"][0, 0]).item())

        kv_bytes = 0
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            if keys is not None:
                kv_bytes += int(keys.numel() * keys.element_size()) + int(
                    layer.values.numel() * layer.values.element_size())
        return {
            "label": self.CPU_GREEDY_K1_LABEL,
            "tokens": emitted[:max_new_tokens],
            "committed": len(emitted[:max_new_tokens]),
            "proposed": proposed,
            "accepted": accepted,
            "rejected": rejected,
            "verifier_calls": verifier_calls,
            "stop_reason": stop_reason,
            "acceptance_rate": (accepted / proposed) if proposed else None,
            "first_commit_ms": first_commit_ms,
            "kv_bytes_allocated": kv_bytes,
            "speed_claim": None,
        }

    MTP_TEXT_SPECULATION_GATE = "CPU_REFERENCE_SPECULATION_EXECUTED_NOT_VLLM"

    def _mtp_text_speculation_route(self, image_present: bool) -> bool:
        """True when the mtp arm must run verifier-backed speculation (official, text-only)."""
        return (
            self.method_arm == "mtp"
            and self.is_official_hf_hooked
            and not image_present
            and getattr(self.model, "mtp_module", None) is not None
        )

    def _execute_mtp_speculative_text(
        self,
        *,
        request_id: str,
        full_prompt: list[int],
        max_new_tokens: int,
        t0: float,
        arch_status: str,
        gates: dict[str, str],
    ) -> "RealExecutionResult":
        """Serving-style mtp arm: committed tokens come from the verifier (CPU reference, not vLLM)."""
        spec = self._speculative_mtp_greedy_k1(full_prompt, max_new_tokens, t0=t0)
        tokens = list(spec["tokens"])
        total_ms = (time.perf_counter() - t0) * 1000.0
        ttft_ms = spec["first_commit_ms"] if spec["first_commit_ms"] is not None else total_ms
        tpot_ms = (total_ms - ttft_ms) / (len(tokens) - 1) if len(tokens) > 1 else total_ms
        output_str = ""
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            try:
                output_str = str(self.tokenizer.decode(tokens, skip_special_tokens=True)).strip()
            except Exception:
                output_str = ""
        if not output_str:
            output_str = f"Decoded {len(tokens)} committed tokens via mtp"
        gates = dict(gates)
        gates["MTP_CORRECTNESS_PASS"] = self.MTP_TEXT_SPECULATION_GATE
        gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] = "PASS" if spec["proposed"] > 0 else "INSUFFICIENT_TOKENS"
        speculation = {
            "label": spec["label"],
            "method_variant": "text_only_official_mtp_k1_verifier",
            "proposed": spec["proposed"],
            "accepted": spec["accepted"],
            "rejected": spec["rejected"],
            "verifier_calls": spec["verifier_calls"],
            "committed": spec["committed"],
            "stop_reason": spec["stop_reason"],
            "acceptance_rate": spec["acceptance_rate"],
            "speed_claim": None,
        }
        return RealExecutionResult(
            method_arm=self.method_arm,
            tokens_generated=tokens,
            output_text=output_str,
            total_latency_ms=total_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            physical_kv_bytes_allocated=spec["kv_bytes_allocated"],
            physical_kv_bytes_freed=0,  # no eviction policy on this arm
            per_step_evicted_slot_rate=None,
            attention_read_restriction_ratio=None,
            attention_mask_restriction_ratio=None,
            grounded_citations=[],
            mtp_acceptance_rate=spec["acceptance_rate"],
            gates=gates,
            telemetry={
                "request_id": request_id,
                "model_id": self.config.model_id,
                "device": self.device,
                "architectural_status": arch_status,
                "is_official_hf_hooked": self.is_official_hf_hooked,
                "measurement_label": "cpu_pytorch_reference_not_vllm",
                "speculation_executed": True,
                "arm_label": "mtp_cpu_speculative_reference_text_only_not_vllm",
                "speculation": speculation,
                "generation": {
                    "stop_at_eos": self.stop_at_eos,
                    "eos_token_ids": list(self.eos_token_ids),
                    "stop_reason": spec["stop_reason"],
                },
            },
        )

    def _starlight_build_step_plans(
        self,
        tracker: H2OCacheTracker,
        k_lens: dict[int, int],
        orig_positions: dict[int, np.ndarray],
        appended_position: int,
        budget_total: int,
        da_spans: Sequence[tuple[str, int, int]],
        grounding_positions: dict[int, tuple[str, ...]],
        enabled: Sequence[str],
        structural: dict[int, tuple[str, ...]] | None = None,
    ) -> tuple[dict[int, torch.Tensor], dict[int, np.ndarray], list[StarlightLayerPlan]]:
        """One decode step of STARLIGHT Stage 1 over every full-attention layer (CPU reference).

        Candidate set: H2O's keep choice with NO internal protection, computed at
        budget - |protected| over all physical entries. The controller then vetoes candidates
        that are protected. Recurrent layers are not in full_attention_layers and are never planned.
        `orig_positions[l]` holds each physical slot's ORIGINAL token position, so protection
        follows the token through compaction.
        """
        enabled_t = tuple(enabled)
        plans_tensor: dict[int, torch.Tensor] = {}
        next_orig: dict[int, np.ndarray] = {}
        step_plans: list[StarlightLayerPlan] = []
        for l_idx in self.config.full_attention_layers:
            k_len = int(k_lens[l_idx])
            prev = np.asarray(orig_positions[l_idx], dtype=np.int64)
            if prev.shape[0] != k_len - 1:
                raise RuntimeError(
                    f"STARLIGHT position bookkeeping out of sync on layer {l_idx}: "
                    f"{prev.shape[0]} tracked vs {k_len - 1} cached entries"
                )
            orig = np.asarray(compact_orig_positions(prev, None, appended=appended_position), dtype=np.int64)
            # S-1 (auditor HIGH): ALL signals are computed on every arm. The candidate set must not
            # depend on the ablation; an ablation only removes protection as a veto (below).
            da_slots = da_protection_from_spans(da_spans, orig)
            grounding_spans = [
                (label, pos, pos + 1) for pos, labels in grounding_positions.items() for label in labels
            ]
            gr_slots = da_protection_from_spans(grounding_spans, orig)
            structural_spans = [
                (label, pos, pos + 1) for pos, labels in (structural or {}).items() for label in labels
            ]
            st_slots = da_protection_from_spans(structural_spans, orig)
            if k_len <= budget_total:
                candidates: list[int] = []  # within budget: nothing to evict
            else:
                # Effective H2O budget from the FULL protected union (all sources, ablation-independent).
                n_protected = len(set(da_slots) | set(gr_slots) | set(st_slots))
                effective = budget_total - n_protected if n_protected <= budget_total else 0
                keep = tracker.keep_mask_without_protection(l_idx, k_len, effective)
                candidates = np.flatnonzero(~keep).tolist()
            plan = plan_starlight_layer(
                layer=l_idx, length=k_len, budget=budget_total, h2o_candidates=candidates,
                signals={"da": da_slots, "grounding": gr_slots, "structural": st_slots},
                enabled_signals=enabled_t,
            )
            retained = np.asarray(plan.retained, dtype=np.int64)
            plans_tensor[l_idx] = torch.as_tensor(retained, dtype=torch.long, device=self.device)
            next_orig[l_idx] = orig[retained]
            step_plans.append(plan)
        return plans_tensor, next_orig, step_plans

    def _starlight_step_records(
        self,
        step: int,
        appended_position: int,
        step_plans: Sequence[StarlightLayerPlan],
        orig_before: dict[int, np.ndarray],
        next_orig: dict[int, np.ndarray],
        budget_total: int,
        structural_positions: set[int],
    ) -> tuple[list[dict[str, Any]], int]:
        """Per (step, full-attention layer) decision records keyed by ORIGINAL token position.

        Returns (records, structural_tokens_evicted_this_step). Structural = vision_start/vision_end.
        Flags are read from the plan's own signal labels, not recomputed from the arm name.
        """
        records: list[dict[str, Any]] = []
        structural_evicted = 0
        for plan in step_plans:
            l_idx = plan.layer
            orig_full = [int(x) for x in np.append(orig_before[l_idx], appended_position)]
            retained_positions = {int(x) for x in next_orig[l_idx]}
            for pos in orig_full:
                if pos not in retained_positions and pos in structural_positions:
                    structural_evicted += 1
            entries = []
            for d in plan.decisions:
                entries.append({
                    "position": orig_full[d.entry_id],
                    "h2o_candidate": d.h2o_candidate,
                    "da_protected": any(sig.startswith("da:") for sig in d.signals_fired),
                    "grounding_protected": any(sig.startswith("grounding:") for sig in d.signals_fired),
                    "structural_protected": any(sig.startswith("structural:") for sig in d.signals_fired),
                    "decision": d.decision,
                    "reason_code": d.reason_code,
                })
            n_protected = sum(
                1 for e in entries
                if e["da_protected"] or e["grounding_protected"] or e["structural_protected"]
            )
            records.append({
                "step": step,
                "phase": "decode",
                "layer": l_idx,
                "budget": budget_total,
                "pre": plan.length,
                "retained": len(plan.retained),
                "evicted": len(plan.evicted),
                "status": plan.status,
                "n_da": sum(1 for e in entries if e["da_protected"]),
                "n_grounding": sum(1 for e in entries if e["grounding_protected"]),
                "n_h2o_candidates": sum(1 for e in entries if e["h2o_candidate"]),
                "n_eligible": len(plan.evicted),
                "n_protected": n_protected,
                # Protected entries that cannot fit the budget (INFEASIBLE only); 0 when feasible.
                "n_protected_outside_budget": (
                    max(0, n_protected - budget_total) if plan.status == STATUS_INFEASIBLE else 0
                ),
                "entries": entries,
            })
        return records, structural_evicted

    def execute_request(
        self,
        request_id: str,
        prompt_token_ids: list[int] | None = None,
        max_new_tokens: int = 256,
        evidence_pool: EvidencePool | None = None,
        image_present: bool = True,
        image_path: str | Path | None = None,
        image_bytes: bytes | None = None,
        question: str | None = None,
    ) -> RealExecutionResult:
        """Execute full multimodal inference across any of the 7 arms with real neural passes."""
        # A zero/negative request must fail before expensive model/image work.
        # bool is not a valid count even though bool is an int subclass.
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer")
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")

        t0 = time.perf_counter()
        raw_prompt_ids = list(prompt_token_ids) if prompt_token_ids is not None else [100, 101, 102]

        input_ids, prefill_pos_ids, rope_deltas, mm_encoder_outputs, vis_trace = (
            self._prepare_official_multimodal_inputs(
                prompt_token_ids=raw_prompt_ids,
                image_present=image_present,
                image_path=image_path,
                image_bytes=image_bytes,
                question=question,
            )
        )
        full_prompt = input_ids[0].tolist()
        vis_invoked = bool(vis_trace.get("invoked", False))

        # SA-01 & P0-01 strict truth-in-labeling gates
        if self.is_official_hf_hooked:
            arch_status = "OFFICIAL_HF_PRETRAINED_HOOKED_MODEL"
            seven_arm_gate = "PASS"
            mm_gate = "PASS" if (self.has_vision_encoder and image_present and vis_invoked) else "GATED_MULTIMODAL_VISION_ENCODER_REQUIRED"
        elif self.is_pretrained:
            arch_status = "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE"
            seven_arm_gate = "GATED_OFFICIAL_HF_HOOK_REQUIRED"
            # Partial reference weights never qualify a real multimodal forward (R3-1).
            mm_gate = "PARTIAL_COVERAGE_NOT_QUALIFIED"
        else:
            arch_status = "PYTORCH_ARCHITECTURAL_SCAFFOLD_UNWEIGHTED"
            seven_arm_gate = "GATED_PRETRAINED_CHECKPOINT_REQUIRED"
            mm_gate = "GATED_PRETRAINED_CHECKPOINT_REQUIRED"

        gates = {
            "POLICY_HARNESS_PASS": "PASS",
            "ACTUAL_BACKEND_HOOK_PASS": "PASS" if self.has_vllm_backend else "GATED_VLLM_BACKEND_ATTACHMENT_REQUIRED",
            # PASS only with official HF hooking (tensor+parameter coverage 1.0, parity verified).
            "REAL_MODEL_FORWARD_PASS": (
                "PASS" if self.is_official_hf_hooked
                else "PARTIAL_COVERAGE_NOT_QUALIFIED" if self.is_pretrained
                else "GATED_PRETRAINED_CHECKPOINT_REQUIRED"
            ),
            "REAL_MULTIMODAL_INFERENCE_PASS": mm_gate,
            "SEVEN_ARM_PRETRAINED_PASS": seven_arm_gate,
            "MTP_CORRECTNESS_PASS": "GATED_NATIVE_MTP_VERIFIER_REQUIRED" if "mtp" in self.method_arm else "NOT_APPLICABLE",
            "MTP_PROPOSAL_DIAGNOSTIC_PASS": "PENDING" if "mtp" in self.method_arm else "NOT_APPLICABLE",
            "MTP_PROPOSAL_RATE_SCIENTIFICALLY_QUALIFIED": "BLOCKED_NO_NATIVE_MTP_KV_HISTORY" if "mtp" in self.method_arm else "NOT_APPLICABLE",
            # A single request cannot independently certify concurrent isolation.
            "CCU_ISOLATION_PASS": "GATED_MULTIREQUEST_ISOLATION_TEST_REQUIRED",
            "VLLM_SCHEDULER_CONCURRENCY_PASS": "GATED_NATIVE_VLLM_LOAD_TEST_REQUIRED",
            "A100_GPU_QUALIFIED": "BLOCKED_PENDING_OWNER_PAID_AUTH",
        }

        # ITER2 Stage 2(a): the serving-style mtp arm runs verifier-backed CPU speculation on
        # text-only requests of the official checkpoint. Image requests keep the scaffold path.
        if self._mtp_text_speculation_route(image_present):
            return self._execute_mtp_speculative_text(
                request_id=request_id, full_prompt=full_prompt, max_new_tokens=max_new_tokens,
                t0=t0, arch_status=arch_status, gates=gates,
            )

        starlight_active = self.starlight_signals is not None
        tracker = (
            self._get_or_create_h2o_tracker(request_id, reset=True)
            if ("h2o" in self.method_arm or starlight_active) else None
        )
        # The H2O budget/plan path is for h2o arms only; STARLIGHT arms use their own plan.
        h2o_plan_active = tracker is not None and not starlight_active
        if tracker is not None:
            is_vis = [
                t in (
                    self.config.image_token_id,
                    self.config.vision_start_token_id,
                    self.config.vision_end_token_id,
                )
                for t in full_prompt
            ]
            tracker.set_visual_tokens(is_vis)

        ecrd_proc = (
            self._get_or_create_ecrd_processor(request_id, evidence_pool, image_present=image_present)
            if "ecrd" in self.method_arm
            else None
        )

        past_kv: Any = None
        recurrent_s: list[Any] | None = None
        da_anchor_len = len(full_prompt)  # DA geometry is anchored to the prefill, not the growing KV
        da_n_visual = int(vis_trace.get("visual_token_embeddings_count", 0))
        da_question_start: int | None = None
        if image_present and self.config.vision_end_token_id in full_prompt:
            da_question_start = len(full_prompt) - 1 - full_prompt[::-1].index(self.config.vision_end_token_id) + 1

        starlight_enabled: tuple[str, ...] = self.starlight_signals or ()
        starlight_budget = 0
        starlight_orig: dict[int, np.ndarray] = {}
        starlight_da_spans: list[tuple[str, int, int]] = []
        starlight_grounding: dict[int, tuple[str, ...]] = {}
        starlight_grounding_status = "NOT_STARTED" if "grounding" in starlight_enabled else "DISABLED_ABLATION"
        starlight_verdicts: dict[str, list[str]] = {}
        starlight_summaries: list[dict[str, Any]] = []
        starlight_max_retained = 0
        starlight_text_fraction: float | None = None
        starlight_text_positions: np.ndarray | None = None
        starlight_next_orig: dict[int, np.ndarray] = {}
        starlight_records: list[dict[str, Any]] = []
        starlight_structural: set[int] = set()
        starlight_structural_labels: dict[int, tuple[str, ...]] = {}
        starlight_structural_evicted = 0
        starlight_cache_mismatch_steps = 0
        if starlight_active:
            # Total budget per full-attention layer, fixed from the prefill length (as H2O does).
            starlight_budget = max(self.h2o_config.min_budget, int(len(full_prompt) * self.h2o_config.budget_ratio))
            starlight_orig = {l: np.arange(len(full_prompt), dtype=np.int64) for l in self.config.full_attention_layers}
            visual_ids = {
                self.config.image_token_id, self.config.vision_start_token_id, self.config.vision_end_token_id,
            }
            starlight_text_positions = np.asarray(
                [i for i, t in enumerate(full_prompt) if t not in visual_ids], dtype=np.int64,
            )
            for i, t in enumerate(full_prompt):
                if t == self.config.vision_start_token_id:
                    starlight_structural_labels[i] = ("vision_start",)
                elif t == self.config.vision_end_token_id:
                    starlight_structural_labels[i] = ("vision_end",)
            starlight_structural = set(starlight_structural_labels)
            if "da" in starlight_enabled:
                # SCRIPTED DIAGNOSTIC DA scope, anchored to the prefill. Only GLOBAL (sink, question) and
                # FOCUS spans protect; the LOCAL window is a mask-only scope and does not protect.
                for tag in _da_scripted_tags(
                    anchor_len=da_anchor_len, phys_len=da_anchor_len, n_visual=da_n_visual,
                    image_present=image_present, question_start=da_question_start,
                ):
                    if tag.tag_type == DATagType.GLOBAL:
                        label = "sink_global" if tag.start_idx == 0 else "question_global"
                    elif tag.tag_type == DATagType.FOCUS:
                        label = f"focus:{tag.focus_id}"
                    else:
                        continue
                    starlight_da_spans.append((label, tag.start_idx, tag.end_idx))
            if "grounding" in starlight_enabled:
                image_sha = _image_sha256(image_bytes, image_path) if image_present else None
                if evidence_pool is None:
                    starlight_grounding_status = "NO_EVIDENCE_DECLARED"
                else:
                    starlight_verdicts = {
                        eid: [status, reason, EVIDENCE_REASON_CODES[status]] for eid, (status, reason) in
                        evidence_validity(evidence_pool, request_id, image_sha).items()
                    }
                    if self.tokenizer is None:
                        starlight_grounding_status = "UNAVAILABLE_NO_TOKENIZER"
                    elif image_present and image_sha is None:
                        starlight_grounding_status = "NOT_PROTECTED_IMAGE_SOURCE_UNKNOWN"
                    else:
                        starlight_grounding = grounding_protection_from_pool(
                            evidence_pool, full_prompt, self.tokenizer.encode, request_id, image_sha,
                        )
                        starlight_grounding_status = "ACTIVE"
        tokens_generated: list[int] = []
        step_latencies: list[float] = []
        ttft_ms = 0.0

        total_physical_freed = 0
        final_allocated_bytes = 0
        # H2O telemetry: None until a measurement exists (never a literal zero).
        h2o_per_step_rate: float | None = None
        h2o_cumulative_evicted_slots: int | None = None  # None until a decode step runs
        h2o_slots_at_prefill: int | None = None
        h2o_last_step_retained_slots: int | None = None
        h2o_unprotected_prefill_slots: int | None = None
        h2o_last_step_unprotected_retained: int | None = None
        final_da_restriction_ratio: float | None = None  # None until a decode step builds a mask
        final_da_causal_area_ratio: float | None = None
        da_measured_masked_entries: int | None = None    # None = no hooked decode measurement
        da_measured_weight_max: float | None = None
        mtp_proposed_total = 0
        mtp_matched_total = 0
        pending_mtp_candidate: int | None = None

        # Prefill forward pass. Reference MTP candidate is evaluated at the
        # next target token's absolute 3-axis RoPE position, never position 0.
        initial_next_pos = (
            prefill_pos_ids[..., -1:] + 1
            if prefill_pos_ids is not None
            else torch.full(
                (3, 1, 1), input_ids.shape[1], device=self.device, dtype=torch.long
            )
        )
        with torch.no_grad():
            prefill_out = self.model(
                input_ids,
                position_ids=prefill_pos_ids,
                mm_encoder_outputs=mm_encoder_outputs,
                compute_mtp=("mtp" in self.method_arm),
                mtp_position_ids=initial_next_pos if "mtp" in self.method_arm else None,
            )
            past_kv = prefill_out["past_key_values"]
            recurrent_s = prefill_out["recurrent_states"]
            logits = prefill_out["logits"]
            ttft_ms = (time.perf_counter() - t0) * 1000.0
            final_allocated_bytes = sum(t.get("kv_bytes_allocated", 0) for t in prefill_out["layer_telemetries"])

        # The official MTP head consumes the hidden state at position t together
        # with the TARGET-selected embedding for t+1 and proposes token t+2.
        # The next ordinary target forward at t+1 can validate that proposal.
        # This reference path NEVER commits drafts or asserts serving speedup.
        if "mtp" in self.method_arm:
            pending_mtp_candidate = int(torch.argmax(prefill_out["mtp_logits"][0, 0]).item())

        forward_path_executed = prefill_out["forward_path"]

        # Fix each full-attention layer's H2O budget from the PREFILL length, once.
        if h2o_plan_active:
            for l_idx in self.config.full_attention_layers:
                tracker.initialize_budget(l_idx, len(full_prompt))
            h2o_slots_at_prefill = len(self.config.full_attention_layers) * len(full_prompt)
            h2o_unprotected_prefill_slots = sum(
                tracker.unprotected_count(l_idx, len(full_prompt))
                for l_idx in self.config.full_attention_layers
            )

        # Feed prefill attention scores into H2O tracker
        if tracker is not None:
            for l_idx, attn_w in prefill_out["attention_matrices"].items():
                tracker.update_attention_scores(l_idx, attn_w[0].float().cpu().numpy())

        if ecrd_proc is not None:
            logits_np0 = logits[0].float().cpu().numpy()
            reweighted_np0 = ecrd_proc(tokens_generated, logits_np0)
            current_token = int(np.argmax(reweighted_np0))
        else:
            current_token = int(torch.argmax(logits[0]).item())
        stop_reason = "max_new_tokens"
        if self.stop_at_eos and current_token in self.eos_token_ids:
            stop_reason = "eos"  # EOS is never emitted into the output
        else:
            tokens_generated.append(current_token)

        # Autoregressive generation loop for up to max_new_tokens
        curr_len = len(full_prompt) + 1
        for step in range(1, max_new_tokens):
            if stop_reason == "eos":
                break
            step_t0 = time.perf_counter()
            next_input = torch.tensor([[current_token]], device=self.device, dtype=torch.long)

            # Determine active physical KV cache length for attention layers
            first_attn_layer = self.config.full_attention_layers[0]
            if self.model.hf_forward_active and past_kv is not None:
                phys_kv_len = past_kv.layers[first_attn_layer].keys.shape[2] + 1
            elif past_kv is not None and past_kv[first_attn_layer] is not None:
                phys_kv_len = past_kv[first_attn_layer][0].shape[2] + 1
            else:
                phys_kv_len = curr_len

            # 1. Declarative Attention mask for full-attention layers (P1-03 compliant)
            da_masks: dict[int, torch.Tensor] = {}
            if self.da_mask_gen is not None:
                # SCRIPTED DIAGNOSTIC ONLY: this is NOT original DA's model-emitted
                # <focus magic_chunks="..."> protocol. No target sparsity is fitted.
                # Bounds derive from the current prompt/vision geometry, and the
                # observed mask density is calculated after actual construction.
                da_tags = _da_scripted_tags(
                    anchor_len=da_anchor_len, phys_len=phys_kv_len,
                    n_visual=da_n_visual, image_present=image_present,
                    question_start=da_question_start,
                )

                raw_mask = self.da_mask_gen.generate_attention_mask(
                    seq_len=phys_kv_len, tags=da_tags,
                    active_focus_id="scripted_visual_segment" if image_present else None,
                )
                da_tensor = torch.from_numpy(raw_mask).to(self.device)
                for l_idx in self.config.full_attention_layers:
                    da_masks[l_idx] = da_tensor
                stats = self.da_mask_gen.mask_statistics(raw_mask)
                final_da_causal_area_ratio = stats["attention_mask_restriction_ratio"]
                # For decoding, only the newest query row executes; LxL area
                # sparsity is not the actual allowed-key fraction for this step.
                final_da_restriction_ratio = (
                    1.0 - float(np.count_nonzero(raw_mask[-1])) / phys_kv_len
                )

            # 2. H2O physical eviction plans for full-attention layers coordinated with physical KV length
            h2o_eviction_plans: dict[int, torch.Tensor] = {}
            if starlight_active:
                k_lens: dict[int, int] = {}
                for l_idx in self.config.full_attention_layers:
                    if self.model.hf_forward_active and past_kv is not None:
                        k_lens[l_idx] = past_kv.layers[l_idx].keys.shape[2] + 1
                    elif past_kv is not None and past_kv[l_idx] is not None:
                        k_lens[l_idx] = past_kv[l_idx][0].shape[2] + 1
                    else:
                        k_lens[l_idx] = curr_len
                h2o_eviction_plans, starlight_next_orig, step_plans = self._starlight_build_step_plans(
                    tracker, k_lens, starlight_orig, curr_len - 1, starlight_budget,
                    starlight_da_spans, starlight_grounding, starlight_enabled,
                    structural=starlight_structural_labels,
                )
                starlight_summaries.append(summarize_plans(step_plans))
                step_records, step_structural = self._starlight_step_records(
                    step, curr_len - 1, step_plans, starlight_orig, starlight_next_orig,
                    starlight_budget, starlight_structural,
                )
                starlight_structural_evicted += step_structural
                if self.starlight_record_decisions:
                    starlight_records.extend(step_records)
                starlight_max_retained = max(starlight_max_retained, max(len(p.retained) for p in step_plans))
                if starlight_text_positions is not None and starlight_text_positions.size:
                    starlight_text_fraction = float(np.mean([
                        np.isin(starlight_next_orig[l_idx], starlight_text_positions).sum()
                        / starlight_text_positions.size
                        for l_idx in self.config.full_attention_layers
                    ]))
            if h2o_plan_active:
                step_evicted = 0
                step_total = 0
                step_retained = 0
                step_unprotected_retained = 0
                for l_idx in self.config.full_attention_layers:
                    if self.model.hf_forward_active and past_kv is not None:
                        k_phys_len = past_kv.layers[l_idx].keys.shape[2] + 1
                    elif past_kv is not None and past_kv[l_idx] is not None:
                        k_phys_len = past_kv[l_idx][0].shape[2] + 1
                    else:
                        k_phys_len = curr_len
                    retained, metrics = tracker.compute_eviction_plan(l_idx, current_length=k_phys_len)
                    retained_indices = torch.where(torch.from_numpy(retained))[0].to(self.device)
                    h2o_eviction_plans[l_idx] = retained_indices
                    step_evicted += metrics["evicted_tokens"]
                    step_total += metrics["total_tokens"]
                    step_retained += metrics["retained_tokens"]
                    step_unprotected_retained += metrics["unprotected_retained"]
                # Per-step rate over full-attention layers (last step only), plus cumulative
                # evicted slots over all steps. Layers may have different compacted lengths.
                h2o_cumulative_evicted_slots = (h2o_cumulative_evicted_slots or 0) + step_evicted
                h2o_last_step_retained_slots = step_retained
                h2o_last_step_unprotected_retained = step_unprotected_retained
                h2o_per_step_rate = step_evicted / step_total if step_total else 0.0

            # Compute 3D RoPE position_ids for decode step when using official HF model
            step_pos_ids: torch.Tensor | None = None
            if self.model.hf_forward_active:
                step_pos_ids = torch.full((3, 1, 1), curr_len - 1, device=self.device, dtype=torch.long)
                if rope_deltas is not None:
                    step_pos_ids = step_pos_ids + rope_deltas.view(1, 1, 1)

            # 3. Model forward step
            with torch.no_grad():
                out = self.model(
                    next_input,
                    past_key_values=past_kv,
                    recurrent_states=recurrent_s,
                    da_masks=da_masks if da_masks else None,
                    h2o_eviction_plans=h2o_eviction_plans if h2o_eviction_plans else None,
                    position_ids=step_pos_ids,
                    compute_mtp=("mtp" in self.method_arm),
                    mtp_position_ids=(
                        step_pos_ids + 1
                        if step_pos_ids is not None
                        else torch.full(
                            (3, 1, 1), curr_len, device=self.device, dtype=torch.long
                        )
                    ) if "mtp" in self.method_arm else None,
                )
                past_kv = out["past_key_values"]
                recurrent_s = out["recurrent_states"]
                step_logits = out["logits"][0]
                for layer_telem in out["layer_telemetries"]:
                    n_entries = layer_telem.get("da_measured_masked_entry_count")
                    if n_entries is None:
                        continue
                    da_measured_masked_entries = (da_measured_masked_entries or 0) + int(n_entries)
                    w_max = layer_telem.get("da_measured_masked_weight_max")
                    if w_max is not None:
                        da_measured_weight_max = (
                            w_max if da_measured_weight_max is None else max(da_measured_weight_max, w_max)
                        )

                if tracker is not None:
                    for l_idx, attn_w in out["attention_matrices"].items():
                        if l_idx in h2o_eviction_plans:
                            retained_np = h2o_eviction_plans[l_idx].cpu().numpy()
                            tracker.compact_layer_state(l_idx, retained_np)
                        tracker.update_attention_scores(l_idx, attn_w[0].float().cpu().numpy())
                if starlight_active:
                    # H-6 length-counter check: the physical cache must hold exactly the retained entries.
                    for l_idx in self.config.full_attention_layers:
                        if self.model.hf_forward_active:
                            cache_len = past_kv.layers[l_idx].keys.shape[2]
                        else:
                            cache_len = past_kv[l_idx][0].shape[2]
                        # ITER2 D4 S-4: test-only injection point (always 0 in production runs).
                        cache_len += self._test_cache_len_offset
                        if cache_len != len(starlight_next_orig[l_idx]):
                            starlight_cache_mismatch_steps += 1
                    starlight_orig = starlight_next_orig

                for telem in out["layer_telemetries"]:
                    if "physical_bytes_freed" in telem:
                        total_physical_freed += telem["physical_bytes_freed"]
                final_allocated_bytes = sum(t.get("kv_bytes_allocated", 0) for t in out["layer_telemetries"])

            # 4. ECRD logits processor
            if ecrd_proc is not None:
                logits_np = step_logits.float().cpu().numpy()
                reweighted_np = ecrd_proc(tokens_generated, logits_np)
                current_token = int(np.argmax(reweighted_np))
            else:
                current_token = int(torch.argmax(step_logits).item())

            if self.stop_at_eos and current_token in self.eos_token_ids:
                stop_reason = "eos"  # shared check for every arm; EOS is not emitted
                break

            # 5. P0 source-correct MTP PROPOSAL DIAGNOSTIC, not speculative serving.
            # Qwen3.5's single trained MTP layer proposes t+2 from h(t) and
            # the embedding of target-selected t+1. Verify that proposal only
            # against the TARGET logit computed at t+1 by the regular decode.
            # Critically, drafts never enter the output or KV/recurrent states.
            if "mtp" in self.method_arm:
                if pending_mtp_candidate is not None:
                    mtp_proposed_total += 1
                    mtp_matched_total += int(pending_mtp_candidate == current_token)
                pending_mtp_candidate = int(torch.argmax(out["mtp_logits"][0, 0]).item())

            tokens_generated.append(current_token)
            curr_len += 1
            step_latencies.append((time.perf_counter() - step_t0) * 1000.0)
            if len(tokens_generated) >= max_new_tokens:
                break

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        tpot_ms = float(np.mean(step_latencies)) if step_latencies else total_latency_ms

        citations = ecrd_proc.citations if ecrd_proc is not None else []
        # This serving-style arm runs NO verifier and commits only target tokens, so no
        # acceptance rate was measured. None = unverified (never a literal 0.0). The
        # verifier-derived CPU reference lives in _speculative_mtp_greedy_k1 and is
        # labelled cpu_greedy_k1_reference_not_vllm_mtp.
        mtp_acc_rate: float | None = None
        # Candidate-vs-committed-target agreement. NOT an acceptance rate: no draft is committed
        # or verified by a speculative step in this serving arm (R4-2).
        mtp_proposal_target_agreement = (
            mtp_matched_total / mtp_proposed_total if mtp_proposed_total else None
        )
        # R4-3: PASS only from measured zero masked weights plus a constructed restriction.
        gates["DA_MASKING_EXERCISED"] = _da_masking_gate(
            self.method_arm, final_da_restriction_ratio,
            da_measured_masked_entries, da_measured_weight_max,
        )
        if "mtp" in self.method_arm:
            gates["MTP_PROPOSAL_DIAGNOSTIC_PASS"] = (
                "PASS" if mtp_proposed_total > 0 else "INSUFFICIENT_TOKENS"
            )

        # Decode semantic text when official tokenizer is attached
        h2o_final_retained_fraction = (
            h2o_last_step_retained_slots / h2o_slots_at_prefill
            if h2o_last_step_retained_slots is not None and h2o_slots_at_prefill
            else None
        )
        h2o_text_side_retained_fraction = (
            h2o_last_step_unprotected_retained / h2o_unprotected_prefill_slots
            if h2o_last_step_unprotected_retained is not None and h2o_unprotected_prefill_slots
            else None
        )
        decoded_str = ""
        if self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            try:
                decoded_str = str(self.tokenizer.decode(tokens_generated, skip_special_tokens=True)).strip()
            except Exception:
                decoded_str = ""
        output_str = decoded_str if decoded_str else f"Decoded {len(tokens_generated)} committed tokens via {self.method_arm}"

        # P1-03: Strictly separate H2O physical KV eviction from DA logical attention mask restriction
        if self.method_arm in DA_MASKING_ARMS and "h2o" not in self.method_arm:
            total_physical_freed = 0

        result = RealExecutionResult(
            method_arm=self.method_arm,
            tokens_generated=tokens_generated,
            output_text=output_str,
            total_latency_ms=total_latency_ms,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            physical_kv_bytes_allocated=final_allocated_bytes,
            physical_kv_bytes_freed=total_physical_freed,
            per_step_evicted_slot_rate=h2o_per_step_rate,
            attention_read_restriction_ratio=final_da_restriction_ratio,
            attention_mask_restriction_ratio=final_da_restriction_ratio,
            grounded_citations=citations,
            mtp_acceptance_rate=mtp_acc_rate,
            gates=gates,
            telemetry={
                "model_id": self.config.model_id,
                "vocab_size": self.config.vocab_size,
                "num_layers": self.config.num_hidden_layers,
                "full_attention_layers": list(self.config.full_attention_layers),
                "recurrent_layers_count": len(self.config.recurrent_layers),
                "device": self.device,
                "architectural_status": arch_status,
                "is_official_hf_hooked": self.is_official_hf_hooked,
                "forward_path_executed": forward_path_executed,
                # R5-5: serving-style arms never speculate. The verifier-backed reference is a separate helper.
                "speculation_executed": False,
                "arm_label": (
                    "mtp_serving_arm_diagnostic_no_speculation" if "mtp" in self.method_arm
                    else "non_speculative_arm"
                ),
                "generation": {
                    "stop_at_eos": self.stop_at_eos,
                    "eos_token_ids": list(self.eos_token_ids),
                    "stop_reason": stop_reason,
                    "generated_token_count": len(tokens_generated),
                    "eos_token_excluded_from_output": True,
                    "protocol_reference": "methods.py quality track natural_eos=True; latency runs set stop_at_eos=False",
                },
                "vllm_backend_status": "MOCKED_PYTORCH_FALLBACK" if not self.has_vllm_backend else "ATTACHED_VLLM_0_29_0",
                "pretrained_weights": self.pretrained_weight_summary,
                "kv_memory_accounting": {
                    "legacy_field": "physical_kv_bytes_freed",
                    "classification": "SUM_OF_KV_TENSOR_FOOTPRINT_DELTAS_NOT_GPU_VRAM_SAVING",
                    "compaction_tensor_byte_deltas_sum": total_physical_freed,
                    "final_retained_kv_payload_bytes": final_allocated_bytes,
                    "peak_cuda_allocated_saving_bytes": None,
                    "peak_cuda_reserved_saving_bytes": None,
                    "device_hbm_reduction_measured": False,
                    "warning": "Each decode step's index_select size delta is added. The total is NOT a peak or allocated/reserved VRAM delta, and may count repeated reductions.",
                },
                "h2o_logical_selection_accounting": {
                    "method_variant": "h2o_visual_anchor_text_budget",
                    "method_variant_is_upstream_h2o": False,
                    "upstream_variant_status": "NOT_IMPLEMENTED_OWNER_DECISION",
                    "budget_basis": "unprotected_prefill_tokens_fixed_per_layer",
                    "protected_outside_budget": True,
                    "per_step_evicted_slot_rate": h2o_per_step_rate,  # last decode step only
                    "cumulative_evicted_slots": h2o_cumulative_evicted_slots,  # all decode steps, full-attention layers
                    "cumulative_total_slots_at_prefill": h2o_slots_at_prefill,  # full-attention layers x prompt length
                    "final_retained_fraction": h2o_final_retained_fraction,  # retained after last step / prefill slots
                    "unprotected_prefill_slots": h2o_unprotected_prefill_slots,  # text-side population at prefill
                    "unprotected_retained_slots_last_step": h2o_last_step_unprotected_retained,
                    "text_side_retained_fraction": h2o_text_side_retained_fraction,
                    "total_retention_near_1_expected_on_vlm_prompts": True,
                    "total_retention_interpretation": (
                        "EXPECTED on VLM prompts: the protected visual block is kept whole, so the total "
                        "final_retained_fraction is near 1.0 by construction. Use text_side_retained_fraction "
                        "for any text-side retention statement. Not a KV-compression result."
                    ),
                    "physical_vram_saving_claimed": False,
                    "protected_vision_tokens_in_prompt": int(
                        sum(1 for t in full_prompt if t in (
                            self.config.image_token_id,
                            self.config.vision_start_token_id,
                            self.config.vision_end_token_id,
                        ))
                    ),
                    # R4-1: protected tokens sit OUTSIDE the budget. The budget caps text tokens only.
                    "multimodal_kv_compression_evidence": False if (
                        any(t == self.config.image_token_id for t in full_prompt)
                    ) else "NOT_APPLICABLE_NO_VISION_TOKENS",
                    "limitation": (
                        "Protected sinks and visual tokens are never evicted and sit outside the fixed budget; "
                        "the budget caps text tokens only. No physical VRAM saving was measured. "
                        "h2o_* arms are not evidence of multimodal KV compression."
                    ),
                    "unprotected_original_h2o_variant": "NOT_IMPLEMENTED_OWNER_DECISION",
                },
                "mtp_candidate_diagnostic": {
                    "classification": "REFERENCE_DRAFT_NO_PERSISTENT_MTP_KV_NO_SPECULATIVE_SERVING",
                    "position_ids_are_absolute": self.is_official_hf_hooked,
                    "persistent_mtp_kv_history": False,
                    "proposal_target_agreement_publishable": False,
                    "proposal_target_agreement_is_acceptance": False,
                    "proposals_verified": mtp_proposed_total,
                    "proposals_matching_target": mtp_matched_total,
                    "proposal_target_agreement_diagnostic": mtp_proposal_target_agreement,
                    "cpu_greedy_k1_reference": {
                        "label": self.CPU_GREEDY_K1_LABEL,
                        "run_in_this_request": False,
                        "entry_point": "RealSevenArmEngine._speculative_mtp_greedy_k1",
                        "note": "Verifier-derived acceptance exists only in that helper; this serving arm does not call it.",
                    },
                    "committed_draft_tokens": 0,
                    "native_target_verifier_executed": False,
                    "speculative_speedup_claimed": False,
                },
                "visual_encoder_trace": vis_trace,
                "ecrd_reweight": (
                    {
                        "reweight_alpha": self._ecrd_alpha(image_present),
                        "alpha_rule": self.ECRD_ALPHA_RULE,
                        "calibrated": False,
                        "cross_condition_comparable": False,
                        "unsupported_penalty_applied": False,
                        "unsupported_penalty_note": "ECRDConfig.unsupported_penalty is inert in this processor",
                    }
                    if "ecrd" in self.method_arm else None
                ),
                "da_cost_accounting": {
                    "attention_mask_restriction_ratio": _round_or_none(final_da_restriction_ratio),
                    "decode_last_query_allowed_key_ratio": (
                        None if final_da_restriction_ratio is None else round(1.0 - final_da_restriction_ratio, 4)
                    ),
                    "full_causal_mask_area_restriction_ratio": _round_or_none(final_da_causal_area_ratio),
                    "measured_masked_entry_count": da_measured_masked_entries,
                    "measured_masked_weight_max": da_measured_weight_max,
                    "measured_source": (
                        "hooked_forward_post_softmax" if da_measured_masked_entries is not None
                        else "NOT_MEASURED_NO_HOOKED_DECODE_STEP"
                    ),
                    "focus_visual_evidence_kept": bool(image_present),
                    "da_scope_provenance": "SCRIPTED_SEGMENT_DIAGNOSTIC_NOT_MODEL_DECLARED",
                    "result_label": DA_DECODE_ONLY_LABEL,
                    "nontrivial_masking_exercised": (
                        None if final_da_restriction_ratio is None else bool(final_da_restriction_ratio > 0.0)
                    ),
                    "prefill_masked": False,
                    "deviations_from_scripted_geometry": (
                        ["question_span_global"] if da_question_start is not None else []
                    ),
                    "original_paper_protocol_qualified": False,
                    "physical_kv_bytes_freed": 0 if self.method_arm in DA_MASKING_ARMS else total_physical_freed,
                    "kv_residency_preserved_ratio": 1.0 if self.method_arm in DA_MASKING_ARMS else (
                        round(h2o_final_retained_fraction, 4) if h2o_final_retained_fraction is not None else None),
                    "kv_residency_preserved_pct": 100.0 if self.method_arm in DA_MASKING_ARMS else (
                        round(h2o_final_retained_fraction * 100.0, 2) if h2o_final_retained_fraction is not None else None),
                    "physical_kv_read_saving_claimed": False,
                    "hbm_traffic_reduction_claimed": False,
                    "accounting_note": (
                        "attention_mask_restriction_ratio is a logical mask restriction estimate only; "
                        "does NOT claim physical KV-read saving or HBM traffic reduction (physical_kv_bytes_freed=0, 100% KV residency preserved)."
                    ),
                },
            },
        )
        if starlight_active:
            summary = combine_summaries(starlight_summaries)
            # Lead decision Q4: not fail-closed. A missing or unusable grounding declaration is flagged,
            # and the arm reports the degraded configuration it actually ran.
            grounding_missing = (
                "grounding" in starlight_enabled and starlight_grounding_status != "ACTIVE"
            )
            if not grounding_missing:
                arm_status = "nominal"
            elif "da" in starlight_enabled:
                arm_status = "degraded_to_h2o_plus_da"
            else:
                arm_status = "degraded_to_h2o_only"
            grounding_by_evidence: dict[str, int] = {}
            for labels in starlight_grounding.values():
                for eid in labels:
                    grounding_by_evidence[eid] = grounding_by_evidence.get(eid, 0) + 1
            result.telemetry["h2o_logical_selection_accounting"] = {
                "method_variant": "NOT_APPLICABLE_STARLIGHT_ARM",
                "note": "STARLIGHT arms report telemetry['starlight']; H2O budget fields are not used.",
            }
            result.telemetry["starlight"] = {
                "measurement_label": STARLIGHT_MEASUREMENT_LABEL,
                "method_variant": f"starlight_stage1::{'+'.join(starlight_enabled)}",
                "method_variant_is_upstream_h2o": False,
                "signals_enabled": list(starlight_enabled),
                "da_signal": (
                    "SCRIPTED_DIAGNOSTIC_NOT_MODEL_DECLARED" if "da" in starlight_enabled else "DISABLED_ABLATION"
                ),
                "grounding_signal_status": starlight_grounding_status,
                "grounding_declaration_missing": grounding_missing,
                "arm_status": arm_status,
                "grounding_evidence_verdicts": starlight_verdicts,
                "grounding_protected_by_evidence": grounding_by_evidence,
                "candidate_basis": "H2O keep set over all positions with NO internal protection, at budget - protected",
                "budget_total": starlight_budget,
                "budget_basis": (
                    "max(min_budget, int(prefill_len * budget_ratio)) total retained entries per full-attention "
                    "layer; protected entries are reserved inside the budget"
                ),
                "layers": list(self.config.full_attention_layers),
                "recurrent_layers_exempt": list(self.config.recurrent_layers),
                "recurrent_exemption_label": "RECURRENT_STATE_NOT_TOKEN_ADDRESSABLE_KV",
                "steps": summary["steps"],
                "status_counts": summary["status_counts"],
                # ITER2 D4 S-3: an INFEASIBLE step keeps the physical KV above budget (no eviction).
                "kv_budget_violated": summary["kv_budget_violated_layers"] > 0,
                "kv_budget_violated_layers": summary["kv_budget_violated_layers"],
                "infeasible_steps": summary["status_counts"].get(STATUS_INFEASIBLE, 0),
                "evicted_total": summary["evicted_total"],
                "retained_total": summary["retained_total"],
                "max_retained_per_layer_step": starlight_max_retained,
                "h2o_candidates_vetoed": summary["h2o_candidates_vetoed"],
                "conflicts_resolved_by_protection": summary["conflicts_resolved_by_protection"],
                "conflicts_by_source": summary["conflicts_by_source"],
                "protected_count_by_source": summary["protected_count_by_source"],
                "protected_count_basis": "entry x full-attention-layer x decode-step, summed over steps",
                "invariant_protected_evicted": summary["protected_evicted"],
                "invariant_evicted_not_candidate": summary["evicted_not_candidate"],
                "text_side_retained_fraction_last_step": (
                    round(starlight_text_fraction, 4) if starlight_text_fraction is not None else None
                ),
                "decision_records": starlight_records if self.starlight_record_decisions else None,
                "gdn_layer_records": [
                    {"layer": l_idx, "decision": GDN_EXEMPT_LABEL, "evicted": 0}
                    for l_idx in self.config.recurrent_layers
                ],
                "structural_tokens_evicted": starlight_structural_evicted,
                "structural_policy": STRUCTURAL_POLICY_LABEL,
                "cache_length_mismatch_steps": starlight_cache_mismatch_steps,
                # HC-5 needs external relevance labels, which this engine does not have. Reported as such.
                "protected_irrelevant_fraction": "NOT_MEASURED_NO_RELEVANCE_LABELS",
                "physical_vram_saving_claimed": False,
                "limitation": (
                    "CPU PyTorch reference, not vLLM. Scripted DA geometry, token-level keyword matching, "
                    "text-side retention only. No speed or VRAM claim."
                ),
            }
        return result
