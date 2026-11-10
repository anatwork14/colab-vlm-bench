"""CLI for read-only checkpoint payload reconciliation; do not use GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vlmbench.checkpoint_payload import (
    audit_safetensors_payload,
    require_exact_payload_match,
    safe_receipt_output_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile safetensors index.total_size from actual local tensor headers")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--hash-shards", action="store_true",
                        help="Also compute full SHA256 of each shard (reads whole checkpoint)")
    parser.add_argument("--json-out", type=Path,
                        help="Optional small JSON output path, never writes into original model tree")
    args = parser.parse_args()
    result = audit_safetensors_payload(args.checkpoint_dir, hash_shards=args.hash_shards)
    print(json.dumps(result, indent=2, sort_keys=True))
    require_exact_payload_match(result)
    if args.json_out:
        out_path = safe_receipt_output_path(args.checkpoint_dir, args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
