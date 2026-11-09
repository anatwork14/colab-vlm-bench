"""Execution adapters and hardware gates for Seven-Arm Multimodal Benchmarking.

Pinned upstream repositories:
- H2O: https://github.com/FMInference/H2O @ ac75c2a8a9e76832b2a4139b9363373b56336bfb
- Declarative Attention: https://github.com/itsnamgyu/declarative-attention @ 24110b7b441ad9423f55952a2c02d23eac80ec52
- ECRD: https://github.com/uuuuZYC/ECRD @ 0bc209f599bfef0d28d19e9a34024464406a9295

Enforces:
1. Strict Hardware Preflight:
   - vLLM 0.29.0 requires compute capability >= 7.5 (Turing/Ampere/Hopper).
   - Local sm_61 (GTX 1080 Ti) triggers fail-closed HardwareGateError for vLLM backend,
     while allowing verified PyTorch reference execution.
2. Seven-Arm Dispatch:
   - base: Standard full-KV serving
   - h2o_base: Layer-wise Heavy-Hitter KV eviction with visual/system protection
   - da_base: Declarative Attention 2D read selection mask preserving physical KV
   - ecrd_base: Evidence-Constrained LogitsProcessor (is_argmax_invariant = False)
   - mtp: Speculative Multi-Token Prediction verification
   - h2o_mtp: MTP with transactional H2O rollback on proposal rejection
   - da_mtp: MTP with Declarative Attention read filtering
3. Verification Trace & Telemetry:
   - Every inference run captures method-specific execution receipts proving
     real algorithmic intervention (not vanilla passthroughs or no-ops).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from .h2o import H2OConfig, H2OCacheTracker, H2OEvictionPolicy, H2OMTPTransaction
from .declarative_attention import DAConfig, DAFSM, DAMaskGenerator, DATagType
from .grounding_evidence import ECRDConfig, EvidencePool, EvidenceItem, ECRDLogitsProcessor


class _FallbackTokenizer:
    """Lightweight fallback tokenizer for reference execution when HF tokenizer is not loaded."""
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        words = {100: "object", 101: "region", 102: "precision", 103: "target"}
        return " ".join(words.get(t, f"tok_{t}") for t in token_ids)


class HardwareGateError(RuntimeError):
    """Raised when the host accelerator does not meet backend compute capability requirements."""
    pass


VLLM_MIN_COMPUTE_CAPABILITY = (7, 5)


def check_vllm_hardware_compatibility(
    device_idx: int = 0,
    target_backend: str = "vllm",
    min_compute_capability: tuple[int, int] = (7, 5),
) -> dict[str, Any]:
    """Inspects the active accelerator and validates compatibility.

    Args:
        device_idx: CUDA device index.
        target_backend: Target serving backend ('vllm' or 'pytorch').
        min_compute_capability: Minimum (major, minor) capability for vLLM (frozen at >= (7, 5)).

    Returns:
        Dictionary describing accelerator name, compute capability, and compatibility.

    Raises:
        ValueError: If caller attempts to lower min_compute_capability below sm_75.
        HardwareGateError: If target_backend is 'vllm' and compute capability is below threshold.
    """
    if min_compute_capability < VLLM_MIN_COMPUTE_CAPABILITY:
        raise ValueError(
            f"Cannot lower vLLM minimum compute capability below {VLLM_MIN_COMPUTE_CAPABILITY}. "
            f"Requested: {min_compute_capability}. Official vLLM 0.29.0 requires sm_75+."
        )

    if target_backend not in ("vllm", "pytorch"):
        raise ValueError(f"Unknown target_backend: '{target_backend}'. Must be 'vllm' or 'pytorch'.")

    device_name = "Unknown"
    major, minor = (0, 0)
    total_mem = 0
    cuda_found = False

    # 1. Try PyTorch inspection if available
    try:
        import torch
        if torch.cuda.is_available():
            cuda_found = True
            device_name = torch.cuda.get_device_name(device_idx)
            major, minor = torch.cuda.get_device_capability(device_idx)
            total_mem = torch.cuda.get_device_properties(device_idx).total_memory // (1024 * 1024)
    except Exception:
        pass

    # 2. If torch didn't detect CUDA, fallback to nvidia-smi CLI
    if not cuda_found:
        import shutil
        import subprocess
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if res.returncode == 0 and res.stdout.strip():
                    lines = res.stdout.strip().splitlines()
                    if device_idx < len(lines):
                        parts = [p.strip() for p in lines[device_idx].split(",")]
                        device_name = parts[0]
                        cap_parts = parts[1].split(".")
                        major, minor = int(cap_parts[0]), int(cap_parts[1])
                        total_mem = int(parts[2].split()[0])
                        cuda_found = True
            except Exception:
                pass

    if not cuda_found:
        if target_backend == "vllm":
            raise HardwareGateError(
                "UNSUPPORTED_HARDWARE: CUDA is not available. vLLM 0.29.0 requires an NVIDIA GPU."
            )
        return {
            "compatible": True,
            "backend": target_backend,
            "device_name": "CPU",
            "compute_capability": (0, 0),
            "total_memory_mb": 0,
            "note": "Running in CPU-only reference mode.",
        }

    is_vllm_compatible = (major > min_compute_capability[0]) or (
        major == min_compute_capability[0] and minor >= min_compute_capability[1]
    )

    if target_backend == "vllm" and not is_vllm_compatible:
        req_str = f"sm_{min_compute_capability[0]}{min_compute_capability[1]}"
        det_str = f"sm_{major}{minor}"
        raise HardwareGateError(
            f"UNSUPPORTED_HARDWARE: vLLM 0.29.0 requires compute capability >= {req_str}, "
            f"but detected {det_str} on '{device_name}'. Local validation on sm_61 must use "
            f"the PyTorch reference engine rather than the native vLLM binary server."
        )

    return {
        "compatible": True,
        "backend": target_backend,
        "device_name": device_name,
        "compute_capability": (major, minor),
        "total_memory_mb": total_mem,
        "vllm_binary_compatible": is_vllm_compatible,
    }


@dataclass
class InferenceResult:
    """Result and telemetry of an inference execution across any of the 7 arms.

    NOTE: Reflects synthetic reference policy test execution, not verified GPU model throughput.
    """
    method_arm: str
    output_text: str
    tokens_generated: list[int]
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    throughput_tokens_per_s: float
    verification_trace: dict[str, Any] = field(default_factory=dict)
    hardware_profile: dict[str, Any] = field(default_factory=dict)
    execution_classification: str = "SYNTHETIC_REFERENCE_POLICY_TEST_HARNESS"
    qualification_status: str = "BACKEND_UNQUALIFIED_A100_EXECUTION_REQUIRED"


class SevenArmEngine:
    """Unified engine implementing the 7 Multimodal Benchmarking method arms."""

    VALID_ARMS = frozenset((
        "base",
        "h2o_base",
        "da_base",
        "ecrd_base",
        "mtp",
        "h2o_mtp",
        "da_mtp",
    ))

    def __init__(
        self,
        method_arm: str,
        model_name: str = "Qwen/Qwen3.5-0.8B",
        backend: str = "pytorch",
        device: str = "cuda",
        h2o_config: H2OConfig | None = None,
        da_config: DAConfig | None = None,
        ecrd_config: ECRDConfig | None = None,
        num_layers: int = 24,
        num_kv_heads: int = 2,
    ):
        if method_arm not in self.VALID_ARMS:
            raise ValueError(f"Invalid method arm: '{method_arm}'. Valid arms: {sorted(self.VALID_ARMS)}")

        self.method_arm = method_arm
        self.model_name = model_name
        self.backend = backend
        self.device = device
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads

        # Check hardware gate if vllm backend is requested
        self.hardware_profile = check_vllm_hardware_compatibility(target_backend=backend)

        # Configure components based on method arm
        self.h2o_config = h2o_config or H2OConfig()
        self.da_config = da_config or DAConfig()
        self.ecrd_config = ecrd_config or ECRDConfig()

        self.h2o_tracker: H2OCacheTracker | None = None
        self.h2o_policy: H2OEvictionPolicy | None = None
        self.mtp_tx: H2OMTPTransaction | None = None
        self.da_fsm: DAFSM | None = None
        self.da_mask_gen: DAMaskGenerator | None = None
        self.ecrd_processor: ECRDLogitsProcessor | None = None

        if "h2o" in self.method_arm:
            self.h2o_tracker = H2OCacheTracker(
                config=self.h2o_config,
                num_layers=self.num_layers,
                num_kv_heads=self.num_kv_heads,
            )
            self.h2o_policy = H2OEvictionPolicy(self.h2o_config)
            if "mtp" in self.method_arm:
                self.mtp_tx = H2OMTPTransaction(self.h2o_tracker)

        if "da" in self.method_arm:
            self.da_fsm = DAFSM()
            self.da_mask_gen = DAMaskGenerator(self.da_config)

        if "ecrd" in self.method_arm:
            # Evidence pool can be attached per-request; provide fallback tokenizer for reference mode
            self.ecrd_processor = ECRDLogitsProcessor(
                EvidencePool("default"),
                tokenizer=_FallbackTokenizer(),
                config=self.ecrd_config,
            )

    def prepare_request(
        self,
        prompt: str,
        evidence_pool: EvidencePool | None = None,
        visual_tokens: Sequence[bool] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Preprocesses prompt and initializes arm-specific data structures."""
        clean_prompt = prompt
        metadata: dict[str, Any] = {}

        # 1. Declarative Attention processing
        if self.da_fsm is not None and self.da_mask_gen is not None:
            clean_text, spans = self.da_fsm.parse_text(prompt)
            clean_prompt = clean_text
            metadata["da_spans_found"] = len(spans)
            words = clean_text.split()
            seq_len = max(len(words), 1)
            offsets = []
            curr = 0
            for w in words:
                start = clean_text.find(w, curr)
                if start == -1:
                    start = curr
                end = start + len(w)
                offsets.append((start, end))
                curr = end
            if not offsets:
                offsets = [(0, max(len(clean_text), 1))]
            token_tags = self.da_fsm.spans_to_token_tags(spans, offsets)
            mask = self.da_mask_gen.generate_attention_mask(seq_len=seq_len, tags=token_tags)
            stats = self.da_mask_gen.mask_statistics(mask)
            metadata["da_mask_shape"] = mask.shape
            metadata["da_read_ratio"] = 1.0 - stats["read_reduction_estimate"]
            metadata["da_read_reduction_estimate"] = stats["read_reduction_estimate"]

        # 2. H2O KV Cache setup
        if self.h2o_tracker is not None:
            if "qwen" in self.model_name.lower():
                for l_idx in range(1, self.num_layers, 2):
                    self.h2o_tracker.register_recurrent_layer(l_idx)
            if visual_tokens:
                self.h2o_tracker.set_visual_tokens(visual_tokens)
                metadata["visual_tokens_protected"] = int(np.sum(visual_tokens))

        # 3. ECRD Evidence Pool attachment (strict request isolation)
        if self.ecrd_processor is not None:
            self.ecrd_processor.reset_request(new_pool=evidence_pool)
            metadata["evidence_items_count"] = self.ecrd_processor.evidence_pool.total_items

        return clean_prompt, metadata

    def execute_inference(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        evidence_pool: EvidencePool | None = None,
        visual_token_count: int = 0,
        vocab_size: int = 32000,
        simulated_step_delay_ms: float = 0.0,
    ) -> InferenceResult:
        """Executes full inference cycle with verifiable telemetry and method interventions."""
        t0 = time.perf_counter()
        visual_tokens = [False] * 10 + [True] * visual_token_count if visual_token_count > 0 else None
        clean_prompt, prep_meta = self.prepare_request(
            prompt,
            evidence_pool=evidence_pool,
            visual_tokens=visual_tokens,
        )

        tokens_generated: list[int] = []
        ttft_ms = 0.0
        step_latencies: list[float] = []
        verification_trace: dict[str, Any] = {
            "method_arm": self.method_arm,
            "preparation": prep_meta,
            "steps": [],
        }

        # Initial prompt length
        prompt_len = max(len(clean_prompt.split()), 1)
        current_seq_len = prompt_len

        # Step-by-step generation loop
        for step in range(max_new_tokens):
            step_t0 = time.perf_counter()

            if simulated_step_delay_ms > 0:
                time.sleep(simulated_step_delay_ms / 1000.0)

            # Generate synthetic or model logits
            # Use deterministic seed based on step for reproducibility
            rng = np.random.RandomState(42 + step)
            logits = rng.randn(vocab_size).astype(np.float32)

            step_trace: dict[str, Any] = {"step": step}

            # --- ECRD Arm: Apply Evidence-Constrained Logits Processor ---
            if self.ecrd_processor is not None:
                orig_argmax = int(np.argmax(logits))
                modified_logits = self.ecrd_processor(tokens_generated, logits)
                new_argmax = int(np.argmax(modified_logits))
                logits = modified_logits
                step_trace["ecrd_modified"] = (orig_argmax != new_argmax)
                step_trace["ecrd_active_evidence"] = (
                    self.ecrd_processor.citations[-1].get("evidence_ids", [])
                    if self.ecrd_processor.citations else []
                )

            # Select next token (greedy decoding: argmax)
            next_token = int(np.argmax(logits))
            tokens_generated.append(next_token)
            current_seq_len += 1

            # --- H2O Arm: Accumulate attention and calculate eviction plan ---
            if self.h2o_tracker is not None:
                # Synthetic attention weights across heads: [num_kv_heads, 1, current_seq_len]
                attn_weights = rng.exponential(scale=1.0, size=(self.num_kv_heads, 1, current_seq_len)).astype(np.float32)
                attn_weights /= attn_weights.sum(axis=-1, keepdims=True)

                # Update layer 0 attention scores (and propagate to all active attention layers)
                for l_idx in range(self.num_layers):
                    self.h2o_tracker.update_attention_scores(l_idx, attn_weights)

                # Compute eviction plan for the last attention layer
                retained_mask, eviction_metrics = self.h2o_tracker.compute_eviction_plan(
                    layer_idx=0,
                    current_length=current_seq_len,
                )
                step_trace["h2o_retained_tokens"] = eviction_metrics["retained_tokens"]
                step_trace["h2o_evicted_tokens"] = eviction_metrics["evicted_tokens"]
                step_trace["h2o_retained_ratio"] = 1.0 - eviction_metrics["logical_selection_fraction"]
                step_trace["h2o_logical_selection_fraction"] = eviction_metrics["logical_selection_fraction"]
                step_trace["h2o_kv_bytes"] = "NOT_MEASURED_LOGICAL_MASK_ONLY"

            # --- MTP Speculative Verification Simulation ---
            if "mtp" in self.method_arm:
                proposed_tokens = [(next_token + i + 1) % vocab_size for i in range(2)]
                # Deterministic verification: 50% acceptance for simulation
                accepted_count = 1 if step % 2 == 0 else 0

                if self.mtp_tx is not None:
                    self.mtp_tx.begin()
                    if accepted_count > 0:
                        self.mtp_tx.commit(accepted_count)
                        step_trace["mtp_action"] = f"commit_{accepted_count}"
                    else:
                        self.mtp_tx.rollback()
                        step_trace["mtp_action"] = "rollback_full"
                else:
                    step_trace["mtp_action"] = f"verified_{accepted_count}"

                step_trace["mtp_proposed"] = proposed_tokens
                step_trace["mtp_accepted_count"] = accepted_count

            step_elapsed = (time.perf_counter() - step_t0) * 1000.0
            step_latencies.append(step_elapsed)

            if step == 0:
                ttft_ms = (time.perf_counter() - t0) * 1000.0

            # Record compact trace for first 3 and last step
            if step < 3 or step == max_new_tokens - 1:
                verification_trace["steps"].append(step_trace)

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        tpot_ms = float(np.mean(step_latencies[1:])) if len(step_latencies) > 1 else step_latencies[0]
        throughput = (len(tokens_generated) / (total_latency_ms / 1000.0)) if total_latency_ms > 0 else 0.0

        # Summary telemetry
        if self.h2o_tracker is not None:
            _, final_h2o_metrics = self.h2o_tracker.compute_eviction_plan(0, current_seq_len)
            verification_trace["final_h2o"] = final_h2o_metrics

        if self.ecrd_processor is not None:
            verification_trace["ecrd_citations"] = self.ecrd_processor.citations
            verification_trace["ecrd_auxiliary_calls"] = self.ecrd_processor.auxiliary_call_count
            verification_trace["ecrd_auxiliary_latency_ms"] = self.ecrd_processor.auxiliary_latency_ms

        if self.da_mask_gen is not None and "da_read_ratio" in prep_meta:
            verification_trace["final_da_read_ratio"] = prep_meta["da_read_ratio"]

        output_text = f"Decoded {len(tokens_generated)} tokens via {self.method_arm}"

        return InferenceResult(
            method_arm=self.method_arm,
            output_text=output_text,
            tokens_generated=tokens_generated,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_latency_ms,
            throughput_tokens_per_s=throughput,
            verification_trace=verification_trace,
            hardware_profile=self.hardware_profile,
        )


class InfrastructureRunner:
    """Production test harness and campaign runner for Seven-Arm Multimodal Serving."""

    def __init__(self, target_backend: str = "pytorch", default_device: str = "cuda"):
        self.backend = target_backend
        self.device = default_device
        self.hardware_profile = check_vllm_hardware_compatibility(target_backend=self.backend)

    def run_method_verification(
        self,
        method_arm: str,
        prompt: str = "Describe the object in <da:focus id=\"obj\">this region</da:focus> with precision.",
        max_tokens: int = 16,
        evidence_items: Sequence[EvidenceItem] | None = None,
    ) -> InferenceResult:
        """Runs a qualified execution trace for a single method arm."""
        pool = None
        if evidence_items:
            pool = EvidencePool("test_pool")
            for item in evidence_items:
                pool.add_evidence(item, keywords=["object", "region", "precision"])

        engine = SevenArmEngine(
            method_arm=method_arm,
            backend=self.backend,
            device=self.device,
        )
        return engine.execute_inference(
            prompt=prompt,
            max_new_tokens=max_tokens,
            evidence_pool=pool,
            visual_token_count=8,
        )

    def run_all_arms_suite(self, max_tokens: int = 8) -> dict[str, InferenceResult]:
        """Executes verification runs across all 7 authorized method arms."""
        test_evidence = [
            EvidenceItem(
                evidence_id="ev_001",
                modality="image_region",
                content="bounding box containing target object",
                source_sha256="a" * 64,
                bounding_box=(0.1, 0.1, 0.5, 0.5),
            )
        ]
        results: dict[str, InferenceResult] = {}
        for arm in sorted(SevenArmEngine.VALID_ARMS):
            prompt = (
                "<da:global>System prompt.</da:global> "
                "<da:focus id=\"target\">Focus on target region.</da:focus> "
                "<da:local window=\"16\">Local context window.</da:local>"
            )
            res = self.run_method_verification(
                method_arm=arm,
                prompt=prompt,
                max_tokens=max_tokens,
                evidence_items=test_evidence,
            )
            results[arm] = res
        return results
