from pathlib import Path
import subprocess
import sys
import json

BASE = Path(__file__).resolve().parents[1]


def test_source_checkout_script_verify_only_does_not_use_network(tmp_path):
    p = subprocess.run([
        sys.executable, str(BASE / "scripts" / "checkout_method_sources.py"),
        "--verify-only", "--dest", str(tmp_path), "--method", "all",
    ], capture_output=True, text=True, check=True)
    payload = json.loads(p.stdout)
    assert payload["status"] == "UPSTREAM_SOURCE_INVENTORY_NOT_GPU_QUALIFICATION"
    assert len(payload["records"]) == 3
    assert all(x["status"] == "NOT_CLONED" for x in payload["records"])
    assert not list(tmp_path.iterdir())
