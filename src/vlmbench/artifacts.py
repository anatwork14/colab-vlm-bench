"""Content-addressed provenance and fail-closed durable receipts."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".pending")
    with tmp.open("w", encoding="utf8") as fp:
        json.dump(value, fp, ensure_ascii=False, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(p)


def atomic_jsonl(path: str | Path, records: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".pending")
    with tmp.open("w", encoding="utf8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        fp.flush()
        os.fsync(fp.fileno())
    tmp.replace(p)


def staged_copy(source: str | Path, dest: str | Path) -> dict:
    """No source mutation; verify hashes of every file after copying to local SSD."""
    source, dest = Path(source).resolve(), Path(dest).resolve()
    if not source.is_dir() or not str(source).startswith("/content/drive/"):
        raise ValueError("Source must be an existing mounted Google Drive directory")
    if dest.is_relative_to(source) or source.is_relative_to(dest):
        raise ValueError("Nested source/destination forbidden")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if not files or not any(x.suffix in (".safetensors", ".bin") for x in files):
        raise ValueError("Checkpoint source has no model weights")
    manifest = []
    for original in files:
        if original.is_symlink():
            raise ValueError(f"Symlink in checkpoint tree: {original}")
        rel = original.relative_to(source)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != original.stat().st_size:
            shutil.copy2(original, target)
        digest = sha256(original)
        if sha256(target) != digest:
            raise IOError(f"Staging hash mismatch: {rel}")
        manifest.append({"path": str(rel), "bytes": original.stat().st_size, "sha256": digest})
    return {"source": str(source), "local_path": str(dest), "files": manifest}


def sync_files(source: str | Path, dest: str | Path) -> None:
    """Non-destructive copy; check receipt integrity (no deletion or cloud credentials)."""
    source, dest = Path(source), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for p in source.rglob("*"):
        if not p.is_file() or p.name.endswith(".pending"):
            continue
        target = dest / p.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == p.stat().st_size and sha256(target) == sha256(p):
            continue
        tmp = target.with_name(target.name + ".pending")
        shutil.copy2(p, tmp)
        if sha256(tmp) != sha256(p):
            raise IOError(f"Drive export mismatch for {p.name}")
        tmp.replace(target)


def verified_resume(receipt_path: Path, expected: dict) -> bool:
    if not receipt_path.exists():
        return False
    try:
        data = json.loads(receipt_path.read_text(encoding="utf8"))
    except (ValueError, OSError):
        return False
    return (
        data.get("status") == "PASS_MEASURED"
        and all(data.get(k) == v for k, v in expected.items())
        and bool(data.get("metrics"))
        and data.get("successful_requests") == data.get("expected_requests")
    )
