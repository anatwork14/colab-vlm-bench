"""Immutable campaign shape, fair comparison invariants, and planned cells."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

LENGTHS = (128, 512, 1024, 2048, 4096, 8192, 16384)
CONCURRENCIES = (1, 2, 4, 8, 16, 32)
EXPECTED_MODELS = {
    "qwen35_08b": "Qwen/Qwen3.5-0.8B",
    "qwen35_2b": "Qwen/Qwen3.5-2B",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "qwen35_9b": "Qwen/Qwen3.5-9B",
    "gemma4_e4b": "google/gemma-4-E4B-it",
}
DATASETS = {"memlens", "mmlongbench_doc"}
MODES = {"base", "mtp"}
ID_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{2,95}$")


class ConfigError(ValueError):
    """Configuration is unsafe, ambiguous, or not scientifically comparable."""


def canonical_json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(data: object) -> str:
    return hashlib.sha256(canonical_json(data).encode()).hexdigest()


def load_config(path: str | Path, *, execution: bool = False) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    validate(obj, execution=execution)
    return obj


def validate(c: dict, *, execution: bool = False) -> None:
    if c.get("schema_version") != "1.0":
        raise ConfigError("Unsupported campaign schema")
    if not ID_PATTERN.fullmatch(str(c.get("campaign_id", ""))):
        raise ConfigError("Invalid campaign ID")
    h, s, p, q = c["hardware"], c["serving"], c["performance"], c["quality"]
    if h["expected_gpu_substring"] != "A100" or h["min_vram_mib"] != 39000:
        raise ConfigError("Main hardware is exact A100 40GB")
    if h["dtype"] != "bfloat16" or h["tensor_parallel_size"] != 1:
        raise ConfigError("Main benchmark requires BF16 TP1")
    if s["backend"] != "vllm" or s["enable_prefix_caching"] is not False:
        raise ConfigError("vLLM with prefix caching disabled is required")
    if int(s["max_num_seqs"]) < 32 or s["max_model_len"] < 16640:
        raise ConfigError("Insufficient configured serving capacity for 16K+256")
    if s["max_num_batched_tokens"] < 1024:
        raise ConfigError("Chunked prefill budget too small")
    models = {m["id"]: m for m in c["models"]}
    if len(models) != len(c["models"]) or set(models) != set(EXPECTED_MODELS):
        raise ConfigError("Exactly five specified target checkpoints are required")
    for mid, expected_hf in EXPECTED_MODELS.items():
        m = models[mid]
        if m["hf_id"] != expected_hf or not 1 <= int(m["mtp_tokens"]) <= 8:
            raise ConfigError(f"Invalid target or MTP setup: {mid}")
        if (mid == "gemma4_e4b") != bool(m.get("assistant_drive_path")):
            raise ConfigError("Gemma E4B requires matching assistant; Qwen uses native MTP")
    if set(c["modes"]) != MODES or len(c["modes"]) != 2:
        raise ConfigError("Must run matched base and MTP")
    if len(c["datasets"]) != 2 or {d["id"] for d in c["datasets"]} != DATASETS:
        raise ConfigError("Two expected datasets required")
    if tuple(p["input_lengths"]) != LENGTHS or tuple(p["concurrencies"]) != CONCURRENCIES:
        raise ConfigError("Fixed 7x6 grid was changed")
    if p.get("input_length_unit") != "text_tokens":
        raise ConfigError("VLM bins must represent model-tokenized text; multimodal tokens reported separately")
    if p["output_tokens"] != 256 or p["repetitions"] != 3:
        raise ConfigError("Output and repetitions are frozen")
    if len(set(p["seeds"])) != 3 or len(p["seeds"]) != 3:
        raise ConfigError("Three independent fixed seeds required")
    if not p["require_real_token_counts"] or not (0.0 < p["max_bin_relative_error"] <= 0.05):
        raise ConfigError("Model-specific effective tokens must be verified")
    if q["require_full_native_context"] is not True or q["official_scorer_required"] is not True:
        raise ConfigError("Official quality evaluation cannot be truncated or proxy-only")
    if execution:
        if c.get("dry_run", True):
            raise ConfigError("dry_run must be false for paid GPU work")
        if not c["storage"]["drive_results_dir"].startswith("/content/drive/"):
            raise ConfigError("Results must be saved to mounted Drive")
        for m in c["models"]:
            if m["revision"] in ("UNPINNED", "", None) or not m["source_drive_path"].startswith("/content/drive/"):
                raise ConfigError(f"Pin checkpoint source/revision for {m['id']}")
            if m["id"] == "gemma4_e4b" and not m["assistant_drive_path"].startswith("/content/drive/"):
                raise ConfigError("Pin Gemma assistant checkpoint on Drive")
        for d in c["datasets"]:
            for key in ("native_path", "derived_path"):
                if not d[key].startswith("/content/drive/"):
                    raise ConfigError(f"Missing Drive dataset path: {d['id']}.{key}")


def request_count(ccu: int) -> int:
    if ccu not in CONCURRENCIES:
        raise ConfigError(f"Invalid CCU: {ccu}")
    return max(16, 4 * ccu)


def cell_seed(dataset: str, input_length: int, ccu: int, repetition: int, base_seed: int) -> int:
    # Deliberately independent of model and mode for paired observations.
    key = f"{dataset}|{input_length}|{ccu}|{repetition}|{base_seed}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")


def planned_cells(c: dict) -> Iterator[dict]:
    p = c["performance"]
    for ds in c["datasets"]:
        for model in c["models"]:
            for mode in c["modes"]:
                for rep in range(p["repetitions"]):
                    for n in p["input_lengths"]:
                        for ccu in p["concurrencies"]:
                            yield {
                                "dataset": ds["id"], "model": model["id"], "mode": mode,
                                "repetition": rep, "input_length": n, "concurrency": ccu,
                                "seed": cell_seed(ds["id"], n, ccu, rep, p["seeds"][rep]),
                                "warmup": ccu, "requests": request_count(ccu),
                                "output_tokens": p["output_tokens"],
                            }


def cell_id(cell: dict) -> str:
    return (
        f"{cell['dataset']}__{cell['model']}__{cell['mode']}__"
        f"r{cell['repetition']}__n{cell['input_length']}__c{cell['concurrency']}"
    )


def plan_summary(c: dict) -> dict:
    cells = list(planned_cells(c))
    return {
        "campaign_id": c["campaign_id"],
        "status": c["status"],
        "dry_run": c["dry_run"],
        "model_count": len(c["models"]),
        "mode_count": len(c["modes"]),
        "dataset_count": len(c["datasets"]),
        "cell_count": len(cells),
        "quality_jobs": len(c["models"]) * len(c["modes"]) * len(c["datasets"]),
        "measured_requests": sum(x["requests"] for x in cells),
        "warmup_requests": sum(x["warmup"] for x in cells),
        "nominal_output_tokens": sum(x["requests"] * x["output_tokens"] for x in cells),
        "grid": {"lengths": list(LENGTHS), "ccus": list(CONCURRENCIES)},
        "config_sha256": fingerprint(c),
        "note": "Plan only; CCU and context capacity must be attempted on actual A100",
    }
