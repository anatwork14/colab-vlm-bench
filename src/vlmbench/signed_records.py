"""Signed JSON records (ITER3 C1 and C4). A record counts only when `ssh-keygen -Y verify` accepts its
signature for the expected namespace and principal, against a signer file kept outside the repository.

Namespaces keep authorities apart: a signature made for the owner's compute approval
(`vlmbench-owner-compute`) does not verify as an extractor-stage qualification (`vlmbench-extractor-qualification`),
and the reverse also fails, because each verification names the namespace and principal it expects.

Trust anchor: the signer file and the private keys. A signature proves who signed the exact bytes, not
that the content is true. A signer who lies, or a local actor who controls the signer key, defeats this.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_signed_record(record_path: Path | None, *, namespace: str, principal: str,
                         allowed_signers: Path | None, runner=subprocess.run) -> dict[str, Any]:
    out: dict[str, Any] = {"verified": False, "reason": "", "record": None, "sha256": None}
    if record_path is None or not Path(record_path).is_file():
        out["reason"] = "record file missing"
        return out
    record_path = Path(record_path)
    sig = record_path.with_name(record_path.name + ".sig")
    if not sig.is_file():
        out["reason"] = "no signature file next to the record"
        return out
    if allowed_signers is None or not Path(allowed_signers).is_file():
        out["reason"] = "no signer file (kept outside the repository)"
        return out
    data = record_path.read_bytes()
    try:
        record = json.loads(data.decode("utf8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        out["reason"] = "record is not valid JSON"
        return out
    if not isinstance(record, dict):
        out["reason"] = "record must be a JSON object"
        return out
    cmd = ["ssh-keygen", "-Y", "verify", "-f", str(allowed_signers), "-I", principal,
           "-n", namespace, "-s", str(sig)]
    try:
        done = runner(cmd, input=data, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        out["reason"] = "signature check could not run"
        return out
    if done.returncode != 0:
        out["reason"] = f"signature does not verify for namespace {namespace!r} and principal {principal!r}"
        return out
    out.update(verified=True, record=record, sha256=sha256_hex(data),
               reason=f"signed in namespace {namespace!r} by principal {principal!r}")
    return out
