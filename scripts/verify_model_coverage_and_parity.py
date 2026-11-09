"""Scientific Verification: Checkpoint Tensor Coverage and Logits Parity.

Audits exact mapped vs unmapped source checkpoint tensors and parameter coverage
for Qwen/Qwen3.5-0.8B, verifying parity against official Hugging Face Transformers.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from vlmbench.infrastructure.real_engine import Qwen35ModelConfig, Qwen35NeuralModel


CHECKPOINT_DIR = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
INDEX_FILE = CHECKPOINT_DIR / "model.safetensors.index.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
PARITY_TOLERANCE = 1e-4
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"


# ---------------------------------------------------------------------------
# Labels are functions of measured coverage and parity (ITER2 D2b C-01/C-02).
# No status string is written by hand next to a number.
# ---------------------------------------------------------------------------

def default_out_file(stamp: str) -> Path:
    """A new per-run file; the committed receipt is never the default target."""
    return RUNS_DIR / f"{stamp}_model_coverage_and_parity_receipt.json"


def write_receipt(path: Path, receipt: dict[str, Any], force: bool = False) -> None:
    """Refuse to replace an existing receipt unless the caller explicitly forces it."""
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing receipt {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w" if force else "x", encoding="utf-8") as f:
        f.write(json.dumps(receipt, indent=2))


def parity_status(max_diff: float, tolerance: float = PARITY_TOLERANCE) -> str:
    """NaN compares false, so a non-finite difference is a FAIL."""
    return "PASS" if max_diff < tolerance else "FAIL"


def scaffold_scientific_status(coverage_pct: float) -> str:
    if coverage_pct >= 100.0:
        return "FULLY_WEIGHTED_PYTORCH_REFERENCE"
    if coverage_pct > 0.0:
        return "PARTIALLY_WEIGHTED_PYTORCH_REFERENCE"
    return "UNWEIGHTED_PYTORCH_SCAFFOLD"


def official_coverage_status(coverage_pct: float) -> str:
    return "OFFICIAL_COVERAGE_COMPLETE" if coverage_pct >= 100.0 else "OFFICIAL_COVERAGE_INCOMPLETE"


def official_hook_status(coverage_pct: float, parity: str) -> str:
    if coverage_pct >= 100.0 and parity == "PASS":
        return "OFFICIAL_TRANSFORMERS_HOOK_QUALIFIED"
    return "OFFICIAL_TRANSFORMERS_HOOK_UNQUALIFIED"


def unmapped_module_patterns(keys: list[str]) -> list[str]:
    """layers.13.mlp.gate_proj.weight -> model.language_model.layers.*.mlp.gate_proj"""
    patterns = {re.sub(r"\.(weight|bias)$", "", re.sub(r"\.\d+\.", ".*.", key)) for key in keys}
    return sorted(patterns)


def scientific_qualification_labels(scaffold_status: str, official_status: str) -> dict[str, str]:
    recommendation = (
        "Use official Transformers hooking for faithful pretrained inference"
        if official_status == "OFFICIAL_TRANSFORMERS_HOOK_QUALIFIED"
        else "No qualified hooking path in this receipt; do not report pretrained-faithful results from it"
    )
    return {
        "custom_scaffold_label": scaffold_status,
        "official_hooked_label": official_status,
        "recommendation": recommendation,
    }


def audit_checkpoint_tensors(checkpoint_dir: Path = CHECKPOINT_DIR) -> dict[str, Any]:
    """Inventory all source checkpoint tensors from safetensors index."""
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Index not found: {index_path}")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map: dict[str, str] = index_data["weight_map"]
    total_checkpoint_tensors = len(weight_map)

    # Classify tensors
    language_tensors = [k for k in weight_map if k.startswith("model.language_model.")]
    vision_tensors = [k for k in weight_map if k.startswith("model.visual.")]
    mtp_tensors = [k for k in weight_map if k.startswith("mtp.")]
    other_tensors = [k for k in weight_map if not (k.startswith("model.language_model.") or k.startswith("model.visual.") or k.startswith("mtp."))]

    # Audit custom Qwen35NeuralModel partial reference scaffold coverage (unhooked)
    cfg = Qwen35ModelConfig.qwen35_08b()
    custom_model = Qwen35NeuralModel(cfg, allocate_scaffold=True)
    custom_summary = custom_model.load_pretrained_weights(checkpoint_dir, prefer_official_hf=False)
    custom_mapped_count = custom_summary.get("loaded_tensors_count", 0)
    custom_unmapped_count = total_checkpoint_tensors - custom_mapped_count

    # Audit official hooked Qwen35NeuralModel / RealSevenArmEngine coverage (100% 473+15=488 tensors)
    hooked_model = Qwen35NeuralModel(cfg, allocate_scaffold=False)
    hooked_summary = hooked_model.load_pretrained_weights(checkpoint_dir, device="cpu", prefer_official_hf=True)
    official_total_params = hooked_summary["base_vl_parameters"]
    official_mapped_count = hooked_summary["base_vl_tensors_loaded"]
    official_mtp_count = hooked_summary["mtp_tensors_loaded"]
    official_coverage_pct = round(hooked_summary["tensor_coverage_ratio"] * 100.0, 2)
    custom_coverage_pct = round(custom_mapped_count / total_checkpoint_tensors * 100.0, 2)

    coverage_report = {
        "checkpoint_dir": str(checkpoint_dir),
        "total_source_tensors": total_checkpoint_tensors,
        "source_tensor_categories": {
            "language_model_tensors": len(language_tensors),
            "vision_encoder_tensors": len(vision_tensors),
            "mtp_speculative_tensors": len(mtp_tensors),
            "other_tensors": len(other_tensors),
        },
        "custom_neural_model_scaffold": {
            "mapped_tensors": custom_mapped_count,
            "unmapped_tensors": custom_unmapped_count,
            "coverage_percentage": custom_coverage_pct,
            "scientific_status": scaffold_scientific_status(custom_coverage_pct),
            "unmapped_modules": unmapped_module_patterns(custom_summary.get("unmapped_tensor_keys", [])),
        },
        "official_transformers_model": {
            "total_parameters": official_total_params,
            "mtp_parameters": hooked_summary["mtp_parameters"],
            "total_checkpoint_parameters": hooked_summary["total_checkpoint_parameters"],
            "direct_model_tensors": official_mapped_count,
            "mtp_speculative_tensors": official_mtp_count,
            "total_accounted_tensors": official_mapped_count + official_mtp_count,
            "unmapped_tensors": hooked_summary["unmapped_tensors_count"],
            "coverage_percentage": official_coverage_pct,
            "parameter_coverage_ratio": hooked_summary["parameter_coverage_ratio"],
            # Final hook status needs measured parity; main() sets it after the parity run.
            "scientific_status": official_coverage_status(official_coverage_pct),
            "unmapped_modules": unmapped_module_patterns(hooked_summary.get("unmapped_tensor_keys", [])),
        },
    }

    del custom_model, hooked_model
    return coverage_report


def verify_one_step_logits_parity(checkpoint_dir: Path = CHECKPOINT_DIR, device: str = "cpu") -> dict[str, Any]:
    """Execute matched one-step logits parity check between official HF model and hooked RealSevenArmEngine."""
    from vlmbench.infrastructure import RealSevenArmEngine

    use_bf16 = (
        str(device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(0)[0] >= 8
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float32

    proc = AutoProcessor.from_pretrained(str(checkpoint_dir))
    ref_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    prompt = "Answer the question directly: What is the color of an emerald?"
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[chat_prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        out_ref = ref_model(**inputs, use_cache=False)
        logits_ref = out_ref.logits[:, -1, :].float().detach().cpu()

    del ref_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Now run the hooked RealSevenArmEngine model on the exact same input_ids
    engine = RealSevenArmEngine(method_arm="base", device=device, checkpoint_dir=checkpoint_dir)
    with torch.no_grad():
        out_hooked = engine.model(inputs["input_ids"])
        logits_hooked = out_hooked["logits"].float().detach().cpu()
    engine.close()

    max_diff = float(torch.max(torch.abs(logits_ref - logits_hooked)).item())
    next_tok = int(torch.argmax(logits_hooked, dim=-1).item())
    decoded = proc.decode([next_tok]).strip()

    parity_report = {
        "status": parity_status(max_diff),
        "device": device,
        "prompt": prompt,
        "input_tokens_count": int(inputs["input_ids"].shape[1]),
        "vocab_size": int(logits_hooked.shape[-1]),
        "max_logits_absolute_difference": max_diff,
        "predicted_next_token_id": next_tok,
        "predicted_next_token_text": decoded,
        "parity_tolerance": PARITY_TOLERANCE,
    }
    return parity_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpoint coverage and one-step logits parity receipt for Qwen3.5-0.8B.")
    parser.add_argument("--out-file", type=Path, default=None,
                        help="Receipt file to create. Default: a new timestamped file under artifacts/runs/.")
    parser.add_argument("--force", action="store_true", help="Allow replacing an existing receipt file.")
    args = parser.parse_args()
    out_file = args.out_file or default_out_file(time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    if out_file.exists() and not args.force:
        parser.error(f"receipt already exists: {out_file} (pass --force to replace it)")

    print("=== [P0-01] Auditing Checkpoint Tensor Coverage ===")
    t0 = time.perf_counter()
    coverage = audit_checkpoint_tensors()
    dt_cov = time.perf_counter() - t0
    print(f"Coverage audit completed in {dt_cov:.2f}s")
    print(f"Source Checkpoint Total Tensors: {coverage['total_source_tensors']}")
    print(f"Custom Scaffold Coverage: {coverage['custom_neural_model_scaffold']['coverage_percentage']}% ({coverage['custom_neural_model_scaffold']['scientific_status']})")
    print(f"Official Transformers Coverage: {coverage['official_transformers_model']['coverage_percentage']}% ({coverage['official_transformers_model']['scientific_status']})")

    print("\n=== [P0-01] Executing One-Step Logits Parity ===")
    t1 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parity = verify_one_step_logits_parity(device=device)
    dt_par = time.perf_counter() - t1
    print(f"Parity check completed in {dt_par:.2f}s on {device}")
    print(f"Max Logits Absolute Difference: {parity['max_logits_absolute_difference']:.8f} (Status: {parity['status']})")
    print(f"Predicted: {parity['predicted_next_token_id']} '{parity['predicted_next_token_text']}'")

    official = coverage["official_transformers_model"]
    official["scientific_status"] = official_hook_status(official["coverage_percentage"], parity["status"])

    receipt = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "coverage_audit": coverage,
        "logits_parity": parity,
        "scientific_qualification": scientific_qualification_labels(
            coverage["custom_neural_model_scaffold"]["scientific_status"],
            official["scientific_status"],
        ),
    }

    write_receipt(out_file, receipt, force=args.force)
    print(f"\nSaved coverage & parity receipt to {out_file}")


if __name__ == "__main__":
    main()
