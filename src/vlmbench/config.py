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
SUPPORTED_DATASETS = DATASETS | {"amber"}
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


def selected_scope(c: dict) -> dict:
    """Resolve an auditable sub-campaign without removing future model registrations."""
    p = c["performance"]
    scope = c.get("execution_scope") or {}
    all_ids = [m["id"] for m in c["models"]]
    return {
        "kind": scope.get("kind", "full_study"),
        "model_ids": scope.get("model_ids", all_ids),
        "input_lengths": scope.get("input_lengths", p["input_lengths"]),
        "concurrencies": scope.get("concurrencies", p["concurrencies"]),
        "repetitions": scope.get("repetitions", list(range(p["repetitions"]))),
    }


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
    if s.get("temperature") != 0:
        raise ConfigError("Accuracy requires greedy temperature=0 for all arms")
    if s.get("top_p", 1.0) != 1.0:
        raise ConfigError("Accuracy requires top_p=1.0 for all arms")
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
    if len(c["datasets"]) != 2 or {d["id"] for d in c["datasets"]} not in (
        DATASETS, {"memlens", "amber"}
    ):
        raise ConfigError("Two expected datasets required")
    if tuple(p["input_lengths"]) != LENGTHS or tuple(p["concurrencies"]) != CONCURRENCIES:
        raise ConfigError("Fixed 7x6 grid was changed")
    if p.get("input_length_unit") != "text_tokens":
        raise ConfigError("VLM bins must represent model-tokenized text; multimodal tokens reported separately")
    if p["output_tokens"] != 256 or p["repetitions"] != 3:
        raise ConfigError("Output and repetitions are frozen")
    if q["max_output_tokens"] != 256:
        raise ConfigError("Natural-EOS quality output cap must be 256")
    if len(set(p["seeds"])) != 3 or len(p["seeds"]) != 3:
        raise ConfigError("Three independent fixed seeds required")
    if not p["require_real_token_counts"] or not (0.0 < p["max_bin_relative_error"] <= 0.05):
        raise ConfigError("Model-specific effective tokens must be verified")
    scope = selected_scope(c)
    if scope["kind"] not in {"full_study", "engineering_pilot", "priority_confirmatory"}:
        raise ConfigError("Unknown execution scope kind")
    allowed = {"kind", "model_ids", "input_lengths", "concurrencies", "repetitions"}
    if set(c.get("execution_scope", {})) - allowed:
        raise ConfigError("Unknown execution scope fields")
    for key, population in (
        ("model_ids", [m["id"] for m in c["models"]]),
        ("input_lengths", p["input_lengths"]),
        ("concurrencies", p["concurrencies"]),
        ("repetitions", list(range(p["repetitions"]))),
    ):
        selected = scope[key]
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
            raise ConfigError(f"Nonempty unique execution scope {key} required")
        if any(x not in population for x in selected):
            raise ConfigError(f"Out-of-registry execution scope {key}")
        if selected != [x for x in population if x in selected]:
            raise ConfigError(f"Execution scope {key} must preserve canonical order")
    if scope["kind"] in {"full_study", "priority_confirmatory"}:
        if (scope["input_lengths"] != p["input_lengths"]
                or scope["concurrencies"] != p["concurrencies"]
                or scope["repetitions"] != list(range(p["repetitions"]))):
            raise ConfigError("Confirmatory scope must retain the full 42-cell x 3-rep grid")
    if scope["kind"] == "full_study" and set(scope["model_ids"]) != set(EXPECTED_MODELS):
        raise ConfigError("Full study must include all registered models")
    if q.get("max_samples", 0) is not None and (
        not isinstance(q.get("max_samples", 0), int) or q.get("max_samples", 0) < 0
    ):
        raise ConfigError("quality.max_samples must be zero (all) or a positive integer")
    if q.get("max_samples", 0) and scope["kind"] != "engineering_pilot":
        raise ConfigError("Native quality subset is restricted to engineering pilot")
    if q.get("max_model_len", 0) < 32768 or q.get("max_num_seqs") != 1:
        raise ConfigError("Quality needs independent long-context runtime profile, CCU=1")
    if q["require_full_native_context"] is not True or q["official_scorer_required"] is not True:
        raise ConfigError("Official quality evaluation cannot be truncated or proxy-only")
    if execution:
        if c.get("dry_run", True):
            raise ConfigError("dry_run must be false for paid GPU work")
        if not c["storage"]["drive_results_dir"].startswith("/content/drive/"):
            raise ConfigError("Results must be saved to mounted Drive")
        for m in c["models"]:
            if m["id"] not in scope["model_ids"]:
                continue
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
    scope = selected_scope(c)
    for ds in c["datasets"]:
        for model in c["models"]:
            if model["id"] not in scope["model_ids"]:
                continue
            for mode in c["modes"]:
                for rep in scope["repetitions"]:
                    for n in ([None] if ds["id"] == "amber" else scope["input_lengths"]):
                        for ccu in scope["concurrencies"]:
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
        "model_count": len(selected_scope(c)["model_ids"]),
        "registered_model_count": len(c["models"]),
        "execution_scope": selected_scope(c),
        "mode_count": len(c["modes"]),
        "dataset_count": len(c["datasets"]),
        "cell_count": len(cells),
        "quality_jobs": len(selected_scope(c)["model_ids"]) * len(c["modes"]) * len(c["datasets"]),
        "measured_requests": sum(x["requests"] for x in cells),
        "warmup_requests": sum(x["warmup"] for x in cells),
        "nominal_output_tokens": sum(x["requests"] * x["output_tokens"] for x in cells),
        "grid": {"lengths": list(LENGTHS), "ccus": list(CONCURRENCIES)},
        "config_sha256": fingerprint(c),
        "note": "Plan only; CCU and context capacity must be attempted on actual A100",
    }
