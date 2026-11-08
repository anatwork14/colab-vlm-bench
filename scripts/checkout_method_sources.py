"""Fetch pinned upstream method sources into a separate local directory.

Source availability is NOT Qwen3.5/vLLM compatibility. Never auto-install or
claim GPU backend qualification from a successful checkout.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from vlmbench.methods import UPSTREAM


def git(*args: str) -> str:
    p = subprocess.run(["git", *args], check=True, text=True,
                       capture_output=True, timeout=300)
    return p.stdout.strip()


def fetch_one(name: str, destination: Path, *, verify_only: bool) -> dict:
    src = UPSTREAM[name]
    root = destination / name
    if root.is_symlink():
        raise ValueError(f"Refuse symlink checkout: {root}")
    if not root.exists():
        if verify_only:
            return {"method": name, "status": "NOT_CLONED", "expected_sha": src["commit"]}
        root.parent.mkdir(parents=True, exist_ok=True)
        git("clone", "--filter=blob:none", "--no-checkout", src["repo"], str(root))
        git("-C", str(root), "checkout", "--detach", src["commit"])
    if not (root / ".git").exists():
        raise ValueError(f"Source directory is not an independent Git clone: {root}")
    head = git("-C", str(root), "rev-parse", "HEAD")
    if head != src["commit"]:
        raise RuntimeError(f"Source SHA mismatch for {name}: {head} != {src['commit']}")
    license_ok = any((root / filename).is_file() for filename in ("LICENSE", "LICENSE.md", "LICENSE.txt"))
    if not license_ok:
        raise RuntimeError(f"Source license file missing: {name}")
    return {
        "method": name, "status": "SOURCE_CHECKOUT_SHA_VERIFIED_NOT_ENGINE_QUALIFIED",
        "repo": src["repo"], "expected_sha": src["commit"],
        "actual_sha": head, "license_present": license_ok,
        "local_path": str(root), "gpu_qualified": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["h2o", "da", "ecrd", "all"], default="all")
    p.add_argument("--dest", type=Path, default=Path("/content/vlmbench/upstream_sources"))
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--report-json", type=Path)
    args = p.parse_args(argv)
    if args.dest.resolve() == Path("/"):
        raise ValueError("Checkout destination cannot be filesystem root")
    chosen = list(UPSTREAM) if args.method == "all" else [args.method]
    records = [fetch_one(name, args.dest.resolve(), verify_only=args.verify_only) for name in chosen]
    report = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "status": "UPSTREAM_SOURCE_INVENTORY_NOT_GPU_QUALIFICATION",
        "records": records,
    }
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
