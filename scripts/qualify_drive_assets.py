#!/usr/bin/env python3
"""CPU qualification script for Google Drive models and datasets.

Verifies the integrity, internal structure, SHA-256 fingerprints, and loader
compatibility of historical STARLIGHT assets on mounted Google Drive:
- Qwen models (Qwen3.5-0.8B, Qwen3.5-4B)
- AMBER (AMBER.zip contents, CRC, query/annotation schema, normalize_amber)
- MemLens (dataset_32k.json, release_images, build_memlens_context_bins)

Enforces strict read-only access (no modifications to original Drive assets).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure repo root is on sys.path
REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from vlmbench.artifacts import atomic_json
from vlmbench.qualification import (
    qualify_all_drive_assets,
    qualify_amber_assets,
    qualify_memlens_assets,
    qualify_qwen_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and qualify historical Google Drive datasets and models."
    )
    parser.add_argument(
        "--drive-root",
        default="/content/drive/MyDrive",
        help="Path to mounted Google Drive root (default: /content/drive/MyDrive)",
    )
    parser.add_argument(
        "--report-json",
        help="Path to write the structured qualification JSON report.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Strict context token tolerance for MemLens bins (default: 0.05, max: 0.05).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of query/session items to inspect for schema validation (default: 5).",
    )
    parser.add_argument(
        "--skip-shards-sha",
        action="store_true",
        help="Skip computing SHA-256 for large safetensors weight shards.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--amber-only",
        action="store_true",
        help="Qualify only AMBER dataset assets.",
    )
    group.add_argument(
        "--memlens-only",
        action="store_true",
        help="Qualify only MemLens dataset assets.",
    )
    group.add_argument(
        "--models-only",
        action="store_true",
        help="Qualify only Qwen model checkpoints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    drive_root = Path(args.drive_root)
    compute_shards_sha = not args.skip_shards_sha

    if args.report_json:
        report_path = Path(args.report_json).resolve()
        if drive_root.is_dir() and (report_path == drive_root or drive_root in report_path.parents):
            print(
                f"Error: --report-json ({report_path}) cannot be written inside --drive-root ({drive_root}). "
                "Drive must remain strictly read-only.",
                file=sys.stderr,
            )
            return 1

    try:
        if args.amber_only:
            result = qualify_amber_assets(drive_root, sample_limit=args.sample_limit)
        elif args.memlens_only:
            result = qualify_memlens_assets(
                drive_root, sample_limit=args.sample_limit, tolerance=args.tolerance
            )
        elif args.models_only:
            result = qualify_qwen_models(
                drive_root, compute_shards_sha=compute_shards_sha
            )
        else:
            result = qualify_all_drive_assets(
                drive_root,
                tolerance=args.tolerance,
                compute_shards_sha=compute_shards_sha,
                sample_limit=args.sample_limit,
            )
    except (ValueError, TypeError, OSError, RuntimeError, PermissionError) as exc:
        print(f"Qualification execution error: {exc}", file=sys.stderr)
        return 1

    formatted_json = json.dumps(result, indent=2)
    print(formatted_json)

    if args.report_json:
        atomic_json(args.report_json, result)
        print(f"Saved qualification report to: {args.report_json}", file=sys.stderr)

    status = result.get("status", "")
    if status in ("ALL_QUALIFIED", "QUALIFIED_AMBER_SOURCE", "QUALIFIED_MEMLENS_SOURCE", "QUALIFIED_MODEL_SOURCES"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
