"""Cell-receipt classification for resumable runs (audit F-05).

Only a COMPLETED receipt may be skipped. Everything else is either flagged (default,
no GPU start) or, on an explicit requeue, moved to a quarantine folder and re-run.
Quarantine renames files; it never deletes or edits a receipt in place.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

COMPLETED = "COMPLETED"
MISSING = "MISSING"
FAILED = "FAILED"    # identity matches, status is not PASS_MEASURED
STALE = "STALE"      # identity or sample set differs from this run
INVALID = "INVALID"  # unreadable, or PASS without the expected measured request count
NON_COMPLETE = frozenset({FAILED, STALE, INVALID})

UNRECORDED = "UNRECORDED"
IDENTITY_KEYS = ("config_sha256", "dataset_sha256", "code_sha256",
                 "execution_pass", "output_contract")


def classify_cell_receipt(path: Path, identity: dict, cell: dict,
                          expected_sample_sha: str | None | Callable[[], str | None]
                          ) -> tuple[str, str]:
    """Return (state, reason). `identity` holds the expected IDENTITY_KEYS for this run.

    `expected_sample_sha` may be a zero-argument callable: it is evaluated only for PASS
    receipts, so failed receipts never require the derived dataset to be loaded.
    """
    path = Path(path)
    if not path.is_file():
        return MISSING, "no receipt"
    try:
        data = json.loads(path.read_text(encoding="utf8"))
    except (ValueError, OSError) as exc:
        return INVALID, f"unreadable receipt: {exc}"
    if not isinstance(data, dict):
        return INVALID, "receipt is not a JSON object"

    mismatched = [k for k in IDENTITY_KEYS if data.get(k) != identity.get(k)]
    if data.get("cell") != cell:
        mismatched.append("cell")
    if mismatched:
        return STALE, "identity mismatch: " + ",".join(mismatched)

    status = data.get("status")
    if status != "PASS_MEASURED":
        return FAILED, f"status={status}"

    if identity.get("code_sha256") == UNRECORDED:
        return STALE, "code identity UNRECORDED: a completed receipt cannot be certified"
    if callable(expected_sample_sha):
        expected_sample_sha = expected_sample_sha()
    if expected_sample_sha is None or data.get("sample_id_sha256") != expected_sample_sha:
        return STALE, "sample set mismatch or not verifiable for this run"
    if (not data.get("metrics") or data.get("successful_requests") != cell["requests"]
            or data.get("expected_requests") != cell["requests"]):
        return INVALID, "PASS_MEASURED without the expected measured request count"
    return COMPLETED, "matches identity, sample set and request count"


def quarantine_cell_receipts(root: Path, drive_root: Path, cell_ids: list[str],
                             stamp: str) -> list[str]:
    """Move non-completed cell receipts (local and Drive copies) into quarantine/<stamp>/.

    Drive copies are moved as well so the start-of-run restore cannot bring them back.
    Returns the list of moved paths (as strings) for the run status.
    """
    moved: list[str] = []
    for cid in cell_ids:
        for rel in (Path("cells") / f"{cid}.json", Path("requests") / f"{cid}.jsonl"):
            for base in (root, drive_root):
                src = base / rel
                if not src.is_file():
                    continue
                dst = base / "quarantine" / stamp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                moved.append(str(dst))
    return moved
