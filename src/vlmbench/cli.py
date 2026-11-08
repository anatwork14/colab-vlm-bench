"""CLI entrypoints. Planning is CPU-only; run-job requires owner-authorized A100."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .artifacts import atomic_json, sha256, staged_copy, sync_files
from .config import cell_id, fingerprint, load_config, plan_summary, planned_cells, selected_scope
from .datasets import normalize_memlens, normalize_mmlong
from .engine import a100_inventory, server_argv, start_server, stop_server, wait_ready
from .quality import import_official, score_proxy
from .report import write_report
from .workloads import load_jsonl
from .bench import run_performance_cell, run_quality_predictions
from .methods import manifest_from_path, plan as methods_plan, native_mode_or_block, blocked_method_receipt, MethodUnqualified
from .hallucination import read_annotations, score_evidence_annotations

from .derivation import generate_candidates, annotate_with_backend
from .preflight import (PreflightError, assess_resources, checkpoint_inventory,
                        host_resources, serving_readiness)


def _installed(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def _persist_error(root: Path, drive_root: Path, tag: str, message: str) -> None:
    atomic_json(root / "run_status.json", {"status": tag, "error": message,
                "recorded_utc": datetime.now(timezone.utc).isoformat()})
    sync_files(root, drive_root)


def run_job(args) -> int:
    cfg = load_config(args.config, execution=True)
    model = next((m for m in cfg["models"] if m["id"] == args.model), None)
    ds = next((d for d in cfg["datasets"] if d["id"] == args.dataset), None)
    if (model is None or ds is None or args.mode not in cfg["modes"]
            or args.model not in selected_scope(cfg)["model_ids"]):
        raise ValueError("Invalid model/dataset/mode selection")
    job_key = f"{args.model}__{args.mode}__{args.dataset}__{'quality' if args.quality else 'serving'}"
    root = Path(cfg["storage"]["local_scratch_dir"]) / cfg["campaign_id"]
    drive_root = Path(cfg["storage"]["drive_results_dir"]) / cfg["campaign_id"]
    root.mkdir(parents=True, exist_ok=True)
    drive_root.mkdir(parents=True, exist_ok=True)
    # Colab local SSD disappears across sessions: restore only resumable receipts and
    # the frozen campaign manifest from durable Drive, rejecting conflicting copies.
    for old in drive_root.rglob("*"):
        if not old.is_file():
            continue
        rel = old.relative_to(drive_root)
        if rel.parts[0] not in {"cells", "quality"} and rel.name != "frozen_config.json":
            continue
        if old.is_symlink():
            raise RuntimeError(f"Unexpected symlink in Drive receipts: {old}")
        local = root / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists() and sha256(local) != sha256(old):
            raise RuntimeError(f"Conflicting remote/local receipt: {rel}")
        if not local.exists():
            shutil.copy2(old, local)
    fp = fingerprint(cfg)
    frozen_path = root / "frozen_config.json"
    if frozen_path.exists():
        prev = json.loads(frozen_path.read_text(encoding="utf8"))
        if fingerprint(prev) != fp:
            raise RuntimeError("Existing run ID has different config. Use a new campaign ID.")
    else:
        atomic_json(frozen_path, cfg)
    inventory = None
    proc = None
    try:
        native_path, derived_path = Path(ds["native_path"]), Path(ds["derived_path"])
        if args.quality and not native_path.is_file():
            raise FileNotFoundError("Quality needs complete native Drive JSONL")
        if not args.quality and not derived_path.is_file():
            raise FileNotFoundError("Serving needs derived, processor-annotated Drive JSONL")

        # Run CPU-only readiness and SSD/RAM checks BEFORE staging weights and
        # launching vLLM. Avoid billable GPU time for deterministic missing-data
        # errors. These checks do not claim a GPU inference pass.
        src_files = checkpoint_inventory(Path(model["source_drive_path"]))
        resources = assess_resources(
            host_resources(Path(cfg["storage"]["local_scratch_dir"])),
            src_files["total_bytes"],
        )
        atomic_json(root / "preflight" / (job_key + ".resources.json"), resources)
        if resources["status"] != "CPU_PREFLIGHT_PASS":
            atomic_json(root / "run_status.json", {
                "status": "BLOCKED_CPU_RESOURCES",
                "issues": resources["issues"], "job": job_key,
            })
            sync_files(root, drive_root)
            print(json.dumps({"status": "BLOCKED_CPU_RESOURCES",
                              "issues": resources["issues"]}), flush=True)
            return 2
        if not args.quality:
            scoped_cells = [
                x for x in planned_cells(cfg)
                if x["model"] == args.model and x["dataset"] == args.dataset
                and x["mode"] == args.mode
            ]
            pending_cells = [
                cell for cell in scoped_cells
                if not (root / "cells" / f"{cell_id(cell)}.json").is_file()
            ]
            if not pending_cells:
                write_report(root, cfg)
                atomic_json(root / "run_status.json", {
                    "status": "NO_PENDING_CELLS_NO_GPU_START",
                    "job": job_key, "recorded_cells": len(scoped_cells),
                })
                sync_files(root, drive_root)
                print(json.dumps({"status": "NO_PENDING_CELLS_NO_GPU_START",
                                  "recorded_cells": len(scoped_cells)}), flush=True)
                return 0
            readiness = serving_readiness(
                derived_path, dataset=ds["id"], model_id=args.model,
                cells=pending_cells,
                tolerance=cfg["performance"]["max_bin_relative_error"],
                max_model_len=cfg["serving"]["max_model_len"],
            )
            atomic_json(root / "preflight" / (job_key + ".workload.json"), readiness)
            if not readiness["ready_cells"]:
                atomic_json(root / "run_status.json", {
                    "status": "BLOCKED_NO_ELIGIBLE_CELLS",
                    "job": job_key, "eligible": readiness["eligible_by_text_bin"],
                })
                sync_files(root, drive_root)
                print(json.dumps({"status": "BLOCKED_NO_ELIGIBLE_CELLS",
                                  "eligible": readiness["eligible_by_text_bin"]}),
                      flush=True)
                return 2
        if args.quality:
            existing_receipt = root / "quality" / (
                f"{ds['id']}__{args.model}__{args.mode}.receipt.json"
            )
            existing_predictions = root / "quality" / (
                f"{ds['id']}__{args.model}__{args.mode}.jsonl"
            )
            if existing_receipt.is_file() and existing_predictions.is_file():
                prior = json.loads(existing_receipt.read_text(encoding="utf8"))
                if (prior.get("status") == "QUALITY_PREDICTIONS_UNSCORED" and
                        prior.get("prediction_sha256") == sha256(existing_predictions)):
                    write_report(root, cfg)
                    atomic_json(root / "run_status.json", {
                        "status": "QUALITY_ALREADY_RECORDED_NO_GPU_START",
                        "job": job_key, "count": prior.get("count"),
                    })
                    sync_files(root, drive_root)
                    print(json.dumps({"status": "QUALITY_ALREADY_RECORDED_NO_GPU_START",
                                      "count": prior.get("count")}), flush=True)
                    return 0
        if args.quality:
            # Check the *actual pilot subset* image count before allocating
            # the 65,536-token quality server. No silently dropped images.
            quality_probe = load_jsonl(native_path, dataset=ds["id"], track="native_quality")
            from .quality import deterministic_subset
            quality_probe = deterministic_subset(
                quality_probe, int(cfg["quality"].get("max_samples", 0)),
                int(cfg["quality"].get("sample_seed", 42)))
            max_images = cfg["serving"]["max_images_per_prompt"]
            over_limit = [
                (str(row["id"]), sum(
                    p.get("type") == "image_path"
                    for m in row["messages"] if isinstance(m["content"], list)
                    for p in m["content"]))
                for row in quality_probe
            ]
            over_limit = [(sid, count) for sid, count in over_limit if count > max_images]
            if over_limit:
                atomic_json(root / "run_status.json", {
                    "status": "BLOCKED_QUALITY_IMAGE_LIMIT",
                    "limit": max_images, "over_limit_count": len(over_limit),
                    "examples": over_limit[:8], "job": job_key,
                })
                sync_files(root, drive_root)
                print(json.dumps({"status": "BLOCKED_QUALITY_IMAGE_LIMIT",
                                  "over_limit_count": len(over_limit),
                                  "limit": max_images}), flush=True)
                return 2
        inventory = a100_inventory()
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native BF16 unsupported or no active CUDA device")
        if _installed("vllm") != cfg["serving"]["backend_version"]:
            raise RuntimeError("Pinned vLLM version mismatch")
        staging_dir = Path(cfg["storage"]["local_scratch_dir"]) / "model_stage" / args.model
        target = staged_copy(model["source_drive_path"], staging_dir / "target")
        assistant = None
        if args.mode == "mtp" and model["id"] == "gemma4_e4b":
            assistant = staged_copy(model["assistant_drive_path"], staging_dir / "assistant")
        model_meta = {
            "checkpoint": model["hf_id"], "revision": model["revision"],
            "target_files": target["files"], "assistant_files": assistant["files"] if assistant else [],
            "source_stage": target["local_path"],
        }
        atomic_json(root / "model_manifests" / (args.model + ".json"), model_meta)
        atomic_json(root / "environment.json", {
            "gpu": inventory, "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "vllm": _installed("vllm"), "httpx": _installed("httpx"),
            "python": sys.version, "repo_source_sha": os.environ.get("VLMBENCH_SOURCE_SHA", "UNRECORDED"),
            "campaign_sha256": fp,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
        })
        derived_hash = sha256(derived_path) if derived_path.is_file() else "NOT_AVAILABLE"
        native_hash = sha256(native_path) if native_path.is_file() else "NOT_AVAILABLE"
        job_manifest = {"dataset": ds["id"], "model": args.model, "mode": args.mode,
                        "dataset_native_sha256": native_hash,
                        "dataset_derived_sha256": derived_hash,
                        "config_sha256": fp, "gpu": inventory}
        atomic_json(root / "jobs" / (job_key + ".json"), job_manifest)
        sync_files(root, drive_root)
        runtime_cfg = json.loads(json.dumps(cfg))
        if args.quality:
            runtime_cfg["serving"]["max_model_len"] = cfg["quality"]["max_model_len"]
            runtime_cfg["serving"]["max_num_seqs"] = cfg["quality"]["max_num_seqs"]
        command = server_argv(runtime_cfg, model, args.mode,
                              target["local_path"],
                              assistant["local_path"] if assistant else None)
        atomic_json(root / "launch" / (job_key + ".json"), {
            "argv": command, "startup_status": "STARTING",
            "expected_method": args.mode,
            "mtp_method_qualification": "NEEDS_BACKEND_TRACE_REVIEW" if args.mode == "mtp" else "NA",
        })
        proc = start_server(command, root / "logs" / (job_key + ".server.log"))
        wait_ready(proc, cfg["serving"]["port"], expected_model=args.model)
        atomic_json(root / "launch" / (job_key + ".json"), {
            "argv": command, "startup_status": "READY",
            "expected_method": args.mode,
            "mtp_method_qualification": "NEEDS_BACKEND_TRACE_REVIEW" if args.mode == "mtp" else "NA",
        })
        rows = (load_jsonl(derived_path, dataset=ds["id"], track="derived_serving")
                if not args.quality else [])
        cells = [x for x in planned_cells(cfg)
                 if x["model"] == args.model and x["dataset"] == args.dataset and x["mode"] == args.mode]
        done = 0
        limit = args.max_cells if args.max_cells > 0 else int(cfg["performance"]["max_cells_per_session"])
        if args.quality:
            cells = []
        for cell in cells:
            # Resume independently at cell level; a full successful cell does not consume job cap.
            existing = root / "cells" / f"{cell_id(cell)}.json"
            if existing.is_file():
                old = json.loads(existing.read_text(encoding="utf8"))
                if (old.get("status") == "PASS_MEASURED"
                        and old.get("config_sha256") == fp
                        and old.get("dataset_sha256") == derived_hash
                        and old.get("cell") == cell):
                    continue
                # Prevent accidental overwrite of failures in the same campaign.
                continue
            if limit and done >= limit:
                break
            result = asyncio.run(run_performance_cell(
                cfg, cell, rows, root, drive_root, args.model, derived_hash))
            done += 1
            print(json.dumps({"cell": cell_id(cell), "status": result["status"]}), flush=True)
        if args.quality:
            quality_rows = quality_probe
            result = asyncio.run(run_quality_predictions(
                cfg, args.model, args.mode, ds["id"], quality_rows, root, drive_root))
            print("Quality predictions:", result["status"])
        write_report(root, cfg)
        sync_files(root, drive_root)
        atomic_json(root / "run_status.json", {
            "status": "JOB_PARTIAL_OR_COMPLETE",
            "model": args.model, "mode": args.mode, "dataset": ds["id"],
            "cells_attempted_this_invocation": done,
            "gpu": inventory, "note": "GPU model execution does not certify native MTP correctness",
        })
        sync_files(root, drive_root)
        return 0
    except BaseException as exc:
        _persist_error(root, drive_root, "FAILED_RUNTIME", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if proc is not None:
            stop_server(proc)


def main(argv=None):
    p = argparse.ArgumentParser(prog="vlmbench")
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("plan", "validate"):
        cmd = sub.add_parser(name, help=f"CPU-only {name}")
        cmd.add_argument("--config", required=True, type=Path)

    mm = sub.add_parser("prepare-memlens")
    mm.add_argument("--input", required=True, type=Path)
    mm.add_argument("--images-root", required=True, type=Path)
    mm.add_argument("--output", required=True, type=Path)
    mm.add_argument("--limit", type=int)

    ml = sub.add_parser("prepare-mmlong")
    ml.add_argument("--input", required=True, type=Path)
    ml.add_argument("--pdf-root", required=True, type=Path)
    ml.add_argument("--images-out", required=True, type=Path)
    ml.add_argument("--output", required=True, type=Path)
    ml.add_argument("--limit", type=int)
    ml.add_argument("--dpi", type=int, default=96)

    derived = sub.add_parser("derive-serving")
    derived.add_argument("--native", required=True, type=Path)
    derived.add_argument("--dataset", required=True, choices=["memlens", "mmlongbench_doc"])
    derived.add_argument("--output", required=True, type=Path)
    derived.add_argument("--max-images", type=int, default=1)

    annotation = sub.add_parser("annotate-serving")
    annotation.add_argument("--input", required=True, type=Path)
    annotation.add_argument("--output", required=True, type=Path)
    annotation.add_argument("--dataset", required=True, choices=["memlens", "mmlongbench_doc"])
    annotation.add_argument("--model-id", required=True)
    annotation.add_argument("--checkpoint", required=True)
    annotation.add_argument("--base-url", default="http://127.0.0.1:8000")
    annotation.add_argument("--limit", type=int, default=0)

    methods = sub.add_parser("methods-plan", help="Plan all seven method arms (CPU only, no inference)")
    methods.add_argument("--config", required=True, type=Path)

    check = sub.add_parser("methods-check", help="Refuse unqualified external methods")
    check.add_argument("--config", required=True, type=Path)
    check.add_argument("--method", required=True)
    check.add_argument("--model", required=True)
    check.add_argument("--dataset", required=True)
    check.add_argument("--blocked-out", type=Path)

    score = sub.add_parser("hallucination-score", help="Score pre-audited claim/evidence judge labels")
    score.add_argument("--annotations", required=True, type=Path)
    score.add_argument("--expected-ids-json", type=Path)
    score.add_argument("--allow-partial", action="store_true")
    score.add_argument("--output", required=True, type=Path)

    job = sub.add_parser("run-job")
    job.add_argument("--config", required=True, type=Path)
    job.add_argument("--model", required=True)
    job.add_argument("--mode", choices=["base", "mtp"], required=True)
    job.add_argument("--dataset", choices=["memlens", "mmlongbench_doc"], required=True)
    job.add_argument("--max-cells", type=int, default=0)
    job.add_argument("--quality", action="store_true")

    report = sub.add_parser("report")
    report.add_argument("--config", required=True, type=Path)
    report.add_argument("--root", required=True, type=Path)

    sc = sub.add_parser("score-proxy")
    sc.add_argument("--dataset-jsonl", required=True, type=Path)
    sc.add_argument("--predictions-jsonl", required=True, type=Path)
    sc.add_argument("--out", required=True, type=Path)

    official = sub.add_parser("import-official-score")
    official.add_argument("--dataset", required=True, choices=["memlens", "mmlongbench_doc"])
    official.add_argument("--upstream-result", required=True, type=Path)
    official.add_argument("--judge-revision", required=True)
    official.add_argument("--sample-manifest-sha256", required=True)
    official.add_argument("--out", required=True, type=Path)

    args = p.parse_args(argv)
    if args.command in ("plan", "validate"):
        cfg = load_config(args.config)
        print(json.dumps(plan_summary(cfg), indent=2))
        return 0
    if args.command == "methods-plan":
        print(json.dumps(methods_plan(manifest_from_path(args.config)), indent=2))
        return 0
    if args.command == "methods-check":
        manifest = manifest_from_path(args.config)
        if args.model not in manifest["models"] or args.dataset not in manifest["datasets"]:
            raise ValueError("Model/dataset not in frozen seven-arm manifest")
        if args.method not in manifest["methods"]:
            raise ValueError("Unknown method in seven-arm manifest")
        try:
            mode = native_mode_or_block(args.method)
        except MethodUnqualified as exc:
            receipt = blocked_method_receipt(args.method, args.model, args.dataset)
            if args.blocked_out is not None:
                atomic_json(args.blocked_out, receipt)
            print(json.dumps(receipt, indent=2))
            return 2
        print(json.dumps({"method": args.method, "native_mode": mode,
                          "status": "REQUIRES_REAL_A100_GPU_QUALIFICATION",
                          "temperature": 0, "top_p": 1.0}, indent=2))
        return 0
    if args.command == "hallucination-score":
        expected = None
        if args.expected_ids_json is not None:
            expected = json.loads(args.expected_ids_json.read_text(encoding="utf8"))
            if not isinstance(expected, list):
                raise ValueError("Expected IDs must be a JSON array")
        scores = score_evidence_annotations(
            read_annotations(args.annotations),
            expected_ids=expected,
            require_complete=not args.allow_partial)
        atomic_json(args.output, scores)
        print(json.dumps({key: val for key, val in scores.items()
                          if key != "per_answer"}, indent=2))
        return 0
    if args.command == "prepare-memlens":
        print(json.dumps(normalize_memlens(args.input, args.images_root, args.output, args.limit), indent=2))
        return 0
    if args.command == "prepare-mmlong":
        print(json.dumps(normalize_mmlong(args.input, args.pdf_root, args.images_out,
                                         args.output, args.limit, args.dpi), indent=2))
        return 0
    if args.command == "derive-serving":
        print(json.dumps(generate_candidates(args.native, args.output,
                                             args.dataset, args.max_images), indent=2))
        return 0
    if args.command == "annotate-serving":
        print(json.dumps(annotate_with_backend(args.input, args.output,
                                             args.dataset, args.model_id,
                                             args.checkpoint, args.base_url,
                                             args.limit), indent=2))
        return 0
    if args.command == "run-job":
        return run_job(args)
    if args.command == "report":
        cfg = load_config(args.config)
        print(json.dumps(write_report(args.root, cfg)["status_counts"], indent=2))
        return 0
    if args.command == "score-proxy":
        def read_jsonl(path):
            return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
        atomic_json(args.out, score_proxy(read_jsonl(args.predictions_jsonl),
                                          read_jsonl(args.dataset_jsonl)))
        print(args.out)
        return 0
    if args.command == "import-official-score":
        data = import_official(args.upstream_result, dataset=args.dataset,
                               judge_revision=args.judge_revision,
                               sample_manifest_sha256=args.sample_manifest_sha256)
        atomic_json(args.out, data)
        print(args.out)
        return 0
    raise AssertionError("Unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
