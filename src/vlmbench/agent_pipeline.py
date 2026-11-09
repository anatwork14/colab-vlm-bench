"""Remote Colab agent pipeline: CPU HF intake, GPU annotation, bounded benchmark."""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import atomic_json, sha256, staged_copy
from .config import load_config
from .execution_safety import require_compute_approval
from .datasets import normalize_memlens, normalize_mmlong, normalize_amber, build_memlens_context_bins
from .drive_inventory import inspect_existing_drive
from .engine import a100_inventory, server_argv, start_server, stop_server, wait_ready
from .hf_sources import DATASETS, MODELS, _qa_file, fetch_dataset, fetch_model
from .notebook_bridge import build_serving_candidates, write_local_campaign
from .preflight import assess_resources, checkpoint_inventory, host_resources
from .qualification import qualify_all_drive_assets
from .source_mapping import read_mapping, verify_mapping, drive_path, stage_amber_archive


def paths_for(job: dict) -> dict:
    root = drive_path(job["drive_root"])
    model, ds, phase = job["model"], job["dataset"], job["phase"]
    limit = job.get("qa_limit", 0)
    if model not in MODELS or ds not in {*DATASETS, "amber"} or phase not in ("pilot", "full"):
        raise ValueError("Unapproved model, dataset or phase")
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Colab Drive not mounted; call colab drivemount")
    if type(limit) is not int or limit < 0 or (phase == "full" and limit):
        raise ValueError("QA limit must be zero for full study")
    suffix = f"n{limit}" if limit else "all"
    base = root / "prepared" / model / ds / phase
    paths = {
        "root": root,
        "model_root": root / "hf" / "models" / model,
        "dataset_root": root / "hf" / "datasets" / ds,
        "native": base / f"native_{suffix}.jsonl",
        "candidates": base / f"candidates_{suffix}.jsonl",
        "annotated": base / f"annotated_{suffix}.jsonl",
        "manifest": base / f"source_{suffix}.json",
        "pdf_render": root / "prepared" / ds / "pdf_pages_96dpi",
        "suffix": suffix,
    }
    if job.get("source_mode", "historical") == "historical":
        source_map = drive_path(job.get("source_map") or str(
            root / "source_maps" / f"{model}__{ds}.json"))
        if job.get("action") == "status" and not source_map.is_file():
            return paths
        mapping = read_mapping(source_map, model, ds)
        for key in ("model_root", "dataset_root"):
            source = drive_path(mapping[key]).resolve()
            if key == "dataset_root" and mapping.get("archive_path"):
                expected = root / "source_cache" / "amber" / str(mapping.get("archive_sha256", ""))
                if source != expected.resolve():
                    raise ValueError("AMBER archive requires independent source cache mapping")
                continue
            if root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve()):
                raise ValueError("Output namespace must be separate from original sources")
        paths.update(model_root=drive_path(mapping["model_root"]),
                     dataset_root=drive_path(mapping["dataset_root"]),
                     source_map=source_map, mapping=mapping)
    return paths


def require_prepared(job: dict, paths: dict) -> dict:
    if not paths["manifest"].is_file():
        raise FileNotFoundError("CPU prepare not complete. Run agent prepare on CPU session.")
    manifest = json.loads(paths["manifest"].read_text(encoding="utf8"))
    for key in ("model", "dataset", "phase"):
        if manifest.get(key) != job.get(key):
            raise ValueError(f"Prepared source {key} does not match job")
    if manifest.get("qa_limit") != job.get("qa_limit", 0):
        raise ValueError("Prepared QA limit differs from job")
    for key, name in (("native", "native_sha256"),
                      ("candidates", "candidate_sha256")):
        if not paths[key].is_file() or sha256(paths[key]) != manifest.get(name):
            raise ValueError(f"Frozen {key} source changed or missing")
    if "source_map" in paths:
        if sha256(paths["source_map"]) != manifest.get("source_map_sha256"):
            raise ValueError("Historical source mapping changed")
        model_receipt = verify_mapping(paths["mapping"])
    else:
        model_receipt = json.loads(
            (paths["model_root"] / ".hf_source.json").read_text(encoding="utf8"))
    if model_receipt.get("revision") != manifest["model_revision"]:
        raise ValueError("Prepared model source HF revision mismatch")
    return manifest


def require_cpu_for_hf_intake() -> None:
    """Block expensive downloads/rendering if an accelerator is attached."""
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("ACCELERATOR_STATE_UNKNOWN: nvidia-smi unavailable")
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("ACCELERATOR_STATE_UNKNOWN: GPU probe failed") from exc
    if probe.returncode != 0:
        raise RuntimeError("ACCELERATOR_STATE_UNKNOWN: GPU probe failed")
    if probe.stdout.strip():
        raise RuntimeError("GPU_SESSION_REFUSED_FOR_PREPARE: download and render on CPU Colab")


def prepare(job: dict) -> dict:
    require_cpu_for_hf_intake()
    p = paths_for(job)
    if "source_map" in p:
        stage_amber_archive(p["mapping"], p["root"])
    if p["manifest"].is_file():
        return {**require_prepared(job, p),
                "status": "PREPARED_REUSED_NO_NEW_DOWNLOAD"}
    if "source_map" in p:
        model = verify_mapping(p["mapping"])
        dataset = {"revision": p["mapping"]["dataset_revision"]}
    else:
        if job["dataset"] == "amber":
            raise ValueError("AMBER requires a reviewed historical source map")
        model = fetch_model(job["model"], p["root"], revision=job.get("model_revision"))
        dataset = fetch_dataset(job["dataset"], p["root"], revision=job.get("dataset_revision"))
    intent = p["manifest"].with_suffix(".intent.json")
    expected_intent = {"model_revision": model["revision"], "dataset_revision": dataset["revision"],
                       "source_map_sha256": sha256(p["source_map"]) if "source_map" in p else None}
    if intent.exists():
        if json.loads(intent.read_text()) != expected_intent:
            raise ValueError("Partial prepared artifacts belong to different frozen sources")
    elif p["native"].exists() or p["candidates"].exists():
        raise ValueError("Unbound partial prepared artifacts require audit before reuse")
    atomic_json(intent, expected_intent)
    p["native"].parent.mkdir(parents=True, exist_ok=True)
    limit = job.get("qa_limit", 0)
    if not p["native"].is_file():
        if job["dataset"] == "amber":
            native_info = normalize_amber(
                p["mapping"]["query_path"], p["mapping"]["images_root"], p["native"],
                pass_type="natural_eos_quality", limit=limit or None)
        elif job["dataset"] == "memlens":
            native_info = normalize_memlens(
                Path(p.get("mapping", {}).get("query_path", p["dataset_root"] / "dataset_32k.json")),
                Path(p.get("mapping", {}).get("images_root", p["dataset_root"])),
                p["native"], limit=limit or None)
        else:
            native_info = normalize_mmlong(
                _qa_file(p["dataset_root"]),
                p["dataset_root"] / "documents",
                p["pdf_render"], p["native"],
                limit=limit or None, dpi=96)
    else:
        native_info = {"status": "EXISTING_NATIVE_REUSED",
                       "sha256_normalized": sha256(p["native"])}
    lengths = [2048, 16384] if job["phase"] == "pilot" else [
        128, 512, 1024, 2048, 4096, 8192, 16384
    ]
    p["candidates"].parent.mkdir(parents=True, exist_ok=True)
    if not p["candidates"].is_file():
        if job["dataset"] == "amber":
            candidate_info = normalize_amber(
                p["mapping"]["query_path"], p["mapping"]["images_root"], p["candidates"],
                pass_type="fixed_serving", limit=limit or None)
        elif job["dataset"] == "memlens" and "source_map" in p:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(str(p["model_root"]),
                                                     local_files_only=True, trust_remote_code=False)
            candidate_info = build_memlens_context_bins(
                p["native"], p["candidates"], tokenizer=tokenizer,
                input_lengths=[128, 512, 1024, 2048, 4096, 8192, 16384],
                max_per_bin=job.get("max_per_bin", 192))
        else:
            candidate_info = build_serving_candidates(
            p["native"], p["candidates"], dataset=job["dataset"],
            model_directory=p["model_root"],
            input_lengths=lengths, max_per_bin=job.get("max_per_bin", 192),
            pdf_root=p["dataset_root"] / "documents"
            if job["dataset"] == "mmlongbench_doc" else None)
    else:
        candidate_info = {"status": "EXISTING_CANDIDATES_REUSED"}
    manifest = {
        "status": "CPU_PREPARED_NEEDS_GPU_TOKEN_ANNOTATION",
        "model": job["model"], "dataset": job["dataset"],
        "phase": job["phase"], "qa_limit": limit,
        "model_revision": model["revision"],
        "dataset_revision": dataset["revision"],
        "native_sha256": sha256(p["native"]),
        "candidate_sha256": sha256(p["candidates"]),
        "native_path": str(p["native"]),
        "candidate_path": str(p["candidates"]),
        "annotated_path": str(p["annotated"]),
        "native_details": native_info,
        "serving_details": candidate_info,
        "official_quality": "UPSTREAM_SCORER_NOT_VERIFIED",
        "source_map_sha256": sha256(p["source_map"]) if "source_map" in p else None,
    }
    atomic_json(p["manifest"], manifest)
    return manifest


def campaign(job: dict, paths: dict) -> tuple[dict, Path]:
    manifest = require_prepared(job, paths)
    template = Path(__file__).resolve().parents[2] / "configs" / (
        "a100_qwen_priority_pilot.json" if job["phase"] == "pilot"
        else "a100_qwen_priority_full.json"
    )
    outfile = paths["root"] / "configs" / (
        f"{job['phase']}_{job['model']}_{job['dataset']}_{paths['suffix']}.json")
    data = {}
    campaign_datasets = ("memlens", "amber") if job["dataset"] == "amber" or "source_map" in paths else tuple(DATASETS)
    for ds in campaign_datasets:
        if ds == job["dataset"]:
            data[ds] = {"native": str(paths["native"]),
                        "derived": str(paths["annotated"])}
        else:
            data[ds] = {
                "native": str(paths["root"] / "NOT_PREPARED" / ds / "native.jsonl"),
                "derived": str(paths["root"] / "NOT_PREPARED" / ds / "annotated.jsonl"),
            }
    if not outfile.is_file():
        if "amber" in campaign_datasets:
            template_data = json.loads(template.read_text(encoding="utf8"))
            template_data["datasets"][1]["id"] = "amber"
            template_data["datasets"][1]["scorer"] = "amber_proxy_unverified"
            template_data["campaign_id"] += "-amber-memlens-v1"
            # Every context bin and CCU is explicit, including in pilot configs.
            template_data["execution_scope"]["input_lengths"] = template_data["performance"]["input_lengths"]
            template_data["execution_scope"]["concurrencies"] = [1, 2, 4, 8, 16, 32]
            template = p_template = paths["root"] / "configs" / f"amber_memlens_{job['phase']}_template.json"
            atomic_json(p_template, template_data)
        write_local_campaign(
            template, outfile, model_id=job["model"],
            drive_model=str(paths["model_root"]),
            drive_results=str(paths["root"] / "results"),
            data_paths=data, revision=manifest["model_revision"],
            campaign_suffix=f"hf-{job['dataset']}-{paths['suffix']}",
        )
    cfg = load_config(outfile, execution=True)
    selected = next(x for x in cfg["models"] if x["id"] == job["model"])
    ds = next(x for x in cfg["datasets"] if x["id"] == job["dataset"])
    if selected["revision"] != manifest["model_revision"] or (
        ds["native_path"] != str(paths["native"]) or
        ds["derived_path"] != str(paths["annotated"])
    ):
        raise ValueError("Campaign config mutated relative to pinned HF sources")
    return cfg, outfile


def stage_for_gpu(cfg: dict, job: dict, paths: dict) -> tuple[dict, dict]:
    require_compute_approval(job.get("owner_approved_compute"))
    a100_inventory()
    import importlib.metadata

    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A100 BF16 unavailable")
    if importlib.metadata.version("vllm") != cfg["serving"]["backend_version"]:
        raise RuntimeError("Pinned vLLM mismatch; qualify backend before running GPU")
    checkpoint = checkpoint_inventory(paths["model_root"])
    resources = assess_resources(host_resources(Path("/content/vlmbench")),
                                 checkpoint["total_bytes"])
    if resources["status"] != "CPU_PREFLIGHT_PASS":
        raise RuntimeError(f"Insufficient RAM/SSD: {resources['issues']}")
    staged = staged_copy(
        paths["model_root"],
        Path("/content/vlmbench/agent_models") / job["model"] / "target",
    )
    spec = next(m for m in cfg["models"] if m["id"] == job["model"])
    return spec, staged


def annotate(job: dict) -> dict:
    require_compute_approval(job.get("owner_approved_compute"))
    p = paths_for(job)
    require_prepared(job, p)
    cfg, _ = campaign(job, p)
    limit = job.get("max_requests", 8)
    if type(limit) is not int or not 1 <= limit <= 128:
        raise ValueError("Annotation budget must be 1..128")
    spec, staged = stage_for_gpu(cfg, job, p)
    import httpx

    from .bench import request_payload
    from .derivation import annotate_with_backend
    from .workloads import load_jsonl

    log = p["root"] / "logs" / (
        f"{job['model']}_{job['dataset']}_base_annotation.log")
    proc = start_server(server_argv(cfg, spec, "base", staged["local_path"]), log)
    try:
        wait_ready(proc, cfg["serving"]["port"], timeout_s=600,
                   expected_model=job["model"])
        examples = load_jsonl(p["candidates"], dataset=job["dataset"],
                              track="amber_serving" if job["dataset"] == "amber" else "derived_serving")
        if not examples:
            raise RuntimeError("No real VLM image examples for annotation")
        with httpx.Client(timeout=150) as client:
            response = client.post(
                f"http://127.0.0.1:{cfg['serving']['port']}/v1/chat/completions",
                json=request_payload(examples[0], model_id=job["model"],
                                     max_tokens=1, streaming=False,
                                     fixed_output=False))
            response.raise_for_status()
            if not response.json().get("usage", {}).get("prompt_tokens"):
                raise RuntimeError("Real image preflight returned no prompt_tokens")
        result = annotate_with_backend(
            p["candidates"], p["annotated"], job["dataset"], job["model"],
            staged["local_path"],
            f"http://127.0.0.1:{cfg['serving']['port']}",
            limit=limit, owner_approved_compute=True)
        return {**result, "model": job["model"], "dataset": job["dataset"],
                "gpu_status": "ANNOTATION_ONLY_NOT_FULL_BENCHMARK"}
    finally:
        stop_server(proc)


def benchmark(job: dict) -> dict:
    require_compute_approval(job.get("owner_approved_compute"))
    p = paths_for(job)
    require_prepared(job, p)
    if not job.get("quality", False) and not p["annotated"].is_file():
        raise FileNotFoundError("Need GPU token annotation before serving benchmark")
    cfg, config_path = campaign(job, p)
    if job.get("mode") not in ("base", "mtp"):
        raise ValueError("Only real native Base/MTP; external method ports NOT qualified")
    max_cells = job.get("max_cells", 1)
    if type(max_cells) is not int or not 1 <= max_cells <= 6:
        raise ValueError("GPU run max_cells must be between 1 and 6")
    from types import SimpleNamespace

    from .cli import run_job
    status = run_job(SimpleNamespace(
        config=config_path, model=job["model"], mode=job["mode"],
        dataset=job["dataset"], max_cells=max_cells,
        quality=bool(job.get("quality", False)), owner_approved_compute=True,
    ))
    if status:
        raise RuntimeError(f"Bench refused, code {status}. Check Drive receipts.")
    return {"status": "GPU_JOB_FINISHED_CHECK_OBSERVED_RECEIPTS",
            "campaign_id": cfg["campaign_id"], "model": job["model"],
            "mode": job["mode"], "dataset": job["dataset"]}


def report(job: dict) -> dict:
    p = paths_for(job)
    cfg, _ = campaign(job, p)
    from .report import write_report
    root = Path(cfg["storage"]["drive_results_dir"]) / cfg["campaign_id"]
    result = write_report(root, cfg)
    return {"status": "DRIVE_REPORT_WRITTEN",
            "summary": str(root / "summary.json"),
            "runbook": str(root / "RUNBOOK.md"),
            "cells": result["status_counts"],
            "campaign_id": cfg["campaign_id"]}


def execute(job: dict) -> dict:
    action = job["action"]
    if action in {"annotate", "benchmark"}:
        require_compute_approval(job.get("owner_approved_compute"))
    if action == "drive-inventory":
        # A read-only, CPU-only source discovery action: no downloads or vLLM.
        return inspect_existing_drive()
    if action == "qualify-drive":
        # A read-only, CPU-only dataset/model qualification action.
        drive_root = Path(job.get("drive_root", "/content/drive/MyDrive"))
        tolerance = float(job.get("tolerance", 0.05))
        compute_shards = bool(job.get("compute_shards_sha", False))
        return qualify_all_drive_assets(drive_root, tolerance=tolerance, compute_shards_sha=compute_shards)
    if action == "prepare":
        return prepare(job)
    if action == "annotate":
        return annotate(job)
    if action == "benchmark":
        return benchmark(job)
    if action == "report":
        return report(job)
    if action == "status":
        p = paths_for(job)
        return {"status": "REMOTE_INVENTORY_CPU",
                "resources": host_resources(Path("/content/vlmbench")),
                "prepared": p["manifest"].exists(),
                "annotated": p["annotated"].exists()}
    raise ValueError(f"Unknown remote action: {action}")


def main(argv=None) -> int:
    """Single fresh OS process for every job, avoiding Colab kernel module caching."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    job = json.loads(args.job.read_text(encoding="utf8"))
    outcome = execute(job)
    atomic_json(args.result, outcome)
    print(json.dumps({"status": "REMOTE_ACTION_DONE", "outcome": outcome},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
