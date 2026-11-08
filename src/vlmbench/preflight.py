"""CPU-only Colab resource/dataset readiness checks before A100 allocation.

No model download, no torch/CUDA import, no image rendering, no GPU server.
Interpret the actual *available* RAM/SSD, not the advertised total quota.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

GiB = 1024 ** 3


class PreflightError(RuntimeError):
    """A preventable dataset/storage/host problem that should block paid GPU work."""


def host_resources(scratch: Path) -> dict:
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(scratch)
    try:
        import psutil
        ram = psutil.virtual_memory()
        ram_total, ram_free = ram.total, ram.available
    except ImportError:
        # /proc/meminfo is available on normal Colab Linux runtimes.
        table = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, rest = line.split(":", 1)
            table[key] = int(rest.strip().split()[0]) * 1024
        ram_total, ram_free = table["MemTotal"], table["MemAvailable"]
    return {
        "ram_total_gib": round(ram_total / GiB, 2),
        "ram_available_gib": round(ram_free / GiB, 2),
        "disk_total_gib": round(disk.total / GiB, 2),
        "disk_free_gib": round(disk.free / GiB, 2),
        "scratch": str(scratch),
    }


def checkpoint_inventory(source: Path) -> dict:
    source = Path(source)
    if not source.is_dir():
        raise PreflightError(f"Missing checkpoint directory: {source}")
    if any(p.is_symlink() for p in source.rglob("*")):
        raise PreflightError("Do not stage checkpoint trees with symlinks")
    files = [p for p in source.rglob("*") if p.is_file()]
    weighted = [p for p in files if p.suffix in (".safetensors", ".bin")]
    if not weighted:
        raise PreflightError("Checkpoint has no .safetensors/.bin weights")
    if not any(p.name == "config.json" for p in files):
        raise PreflightError("Checkpoint config.json missing")
    if not any("tokenizer" in p.name.lower() for p in files):
        raise PreflightError("Checkpoint tokenizer artifacts missing")
    return {
        "files": len(files),
        "weight_files": len(weighted),
        "total_bytes": sum(p.stat().st_size for p in files),
        "source": str(source),
    }


def dataset_probe_memlens(source: Path, images_root: Path) -> dict:
    source, images_root = Path(source), Path(images_root)
    if not source.is_file():
        raise PreflightError(f"MemLens JSON missing: {source}")
    if not (images_root / "release_images").is_dir():
        raise PreflightError(f"MemLens release_images directory missing under {images_root}")
    with source.open("rb") as f:
        first = f.read(256).lstrip()
    if not first.startswith(b"["):
        raise PreflightError("MemLens source is not a top-level JSON list")
    return {
        "json_bytes": source.stat().st_size,
        "images_directory": str(images_root / "release_images"),
        "status": "SOURCE_LAYOUT_PRESENT_NOT_PROMPT_PARITY_VALIDATED",
    }


def dataset_probe_mmlong(qa_path: Path, pdf_root: Path, probe_rows: int = 16) -> dict:
    qa_path, pdf_root = Path(qa_path), Path(pdf_root).resolve()
    if not qa_path.is_file():
        raise PreflightError(f"MMLongBench QA missing: {qa_path}")
    if not pdf_root.is_dir():
        raise PreflightError(f"MMLongBench PDF root missing: {pdf_root}")
    expected = {"doc_id", "question", "answer"}
    sampled = []
    if qa_path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise PreflightError("Install pyarrow before reading QA parquet") from exc
        pf = pq.ParquetFile(qa_path)
        missing = expected.difference(pf.schema_arrow.names)
        if missing:
            raise PreflightError(f"MMLongBench QA columns missing: {sorted(missing)}")
        for batch in pf.iter_batches(batch_size=min(probe_rows, 16)):
            sampled.extend(batch.to_pylist()[: max(probe_rows - len(sampled), 0)])
            if len(sampled) >= probe_rows:
                break
        count = pf.metadata.num_rows
    elif qa_path.suffix.lower() in (".json", ".jsonl"):
        if qa_path.suffix.lower() == ".jsonl":
            with qa_path.open("r", encoding="utf8") as f:
                for line in f:
                    if line.strip():
                        sampled.append(json.loads(line))
                        if len(sampled) >= probe_rows:
                            break
            count = None
        else:
            # QA JSON here is small (~250 KB); not a full document corpus.
            obj = json.loads(qa_path.read_text(encoding="utf8"))
            rows = obj if isinstance(obj, list) else obj.get("data", obj.get("questions", []))
            if not isinstance(rows, list):
                raise PreflightError("Invalid MMLongBench QA JSON list")
            count = len(rows)
            sampled = rows[:probe_rows]
    else:
        raise PreflightError("MMLongBench QA must be JSON, JSONL or Parquet")
    if not sampled:
        raise PreflightError("MMLongBench QA is empty")
    missing = expected.difference(sampled[0].keys())
    if missing:
        raise PreflightError(f"MMLongBench QA fields missing: {sorted(missing)}")
    absent, present = [], 0
    for row in sampled:
        doc = (pdf_root / str(row["doc_id"])).resolve()
        if not doc.is_relative_to(pdf_root):
            raise PreflightError(f"Unsafe PDF doc_id: {row['doc_id']}")
        if not doc.is_file():
            absent.append(str(row["doc_id"]))
        else:
            present += 1
    if absent:
        raise PreflightError(
            f"Sampled MMLongBench PDF files missing ({len(absent)}/{len(sampled)}): "
            + ", ".join(absent[:4])
        )
    return {
        "qa_bytes": qa_path.stat().st_size,
        "qa_rows": count,
        "sampled_pdf_rows": len(sampled),
        "sampled_pdf_present": present,
        "status": "SAMPLED_SOURCE_LAYOUT_PRESENT_NOT_FULL_OFFICIAL_PARITY",
    }


def assess_resources(resources: dict, model_bytes: int | None = None) -> dict:
    """Budget after staging weights plus space for dataset rendering/receipts.

    The remaining disk threshold is deliberately conservative; this does not
    guarantee downstream GPU KV capacity or total PDF image expansion.
    """
    minimum_ram = 10.0
    minimum_disk = max(18.0, 10.0 + 1.35 * (model_bytes or 0) / GiB)
    issues = []
    if resources["ram_available_gib"] < minimum_ram:
        issues.append(f"Available system RAM < {minimum_ram} GiB")
    if resources["disk_free_gib"] < minimum_disk:
        issues.append(f"Local free SSD < {minimum_disk:.1f} GiB")
    return {
        "status": "CPU_PREFLIGHT_PASS" if not issues else "CPU_PREFLIGHT_BLOCKED",
        "issues": issues,
        "required_available_ram_gib": minimum_ram,
        "required_free_local_disk_gib": round(minimum_disk, 1),
        **resources,
        "note": "RAM/SSD gates are host checks; actual A100 serving capacity requires a bounded GPU pilot",
    }



def serving_readiness(
    annotated_path: Path, *, dataset: str, model_id: str,
    cells: list[dict], tolerance: float, max_model_len: int,
) -> dict:
    """Streaming scan: count eligible UNIQUE real VLM prompts by text bin.

    This is deliberately independent of torch/vLLM, and must precede launching
    a server. It is a precondition, not a claim that A100 has enough KV memory.
    """
    path = Path(annotated_path)
    if not path.is_file():
        return {"status": "BLOCKED_NO_ANNOTATED_DATASET",
                "path": str(path), "ready_cells": [], "blocked_cells": len(cells)}
    from collections import Counter
    counts = Counter()
    seen = set()
    with path.open("r", encoding="utf8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dataset") != dataset or row.get("track") != "derived_serving":
                raise PreflightError("Annotated JSONL has wrong dataset/track")
            sid = str(row["id"])
            if sid in seen:
                raise PreflightError(f"Duplicate annotated ID: {sid}")
            seen.add(sid)
            stat = row.get("model_token_counts", {}).get(model_id, {})
            if stat.get("verified_by") != "vllm_effective_token_accounting":
                continue
            text_len, effective = stat.get("text_tokens"), stat.get("effective_total_tokens")
            if not isinstance(text_len, int) or not isinstance(effective, int):
                continue
            for n in {c["input_length"] for c in cells}:
                if (abs(text_len - n) / n <= tolerance and
                        effective < max_model_len - 256):
                    counts[n] += 1
    ready, blocked = [], []
    for cell in cells:
        n, required = cell["input_length"], cell["requests"] + cell["warmup"]
        (ready if counts[n] >= required else blocked).append({
            "text_bin": n, "ccu": cell["concurrency"],
            "eligible": counts[n], "required": required,
            "repetition": cell["repetition"],
        })
    return {
        "status": "READY_CELLS_PRESENT" if ready else "BLOCKED_NO_ELIGIBLE_CELLS",
        "dataset": dataset, "model": model_id, "path": str(path),
        "eligible_by_text_bin": {str(n): counts[n] for n in sorted({c["input_length"] for c in cells})},
        "ready_cells": ready, "blocked_cell_details": blocked,
        "blocked_cells": len(blocked),
        "note": "CPU sample-count check only; model VRAM and 256 output token generation unverified",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="CPU-only preflight, safe on Colab CPU runtime")
    p.add_argument("--scratch", type=Path, default=Path("/content/vlmbench"))
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--memlens-json", type=Path)
    p.add_argument("--memlens-images", type=Path)
    p.add_argument("--mmlong-qa", type=Path)
    p.add_argument("--mmlong-pdfs", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args(argv)
    report = {"checkpoint": None, "memlens": None, "mmlong": None}
    try:
        if a.checkpoint is not None:
            report["checkpoint"] = checkpoint_inventory(a.checkpoint)
        if a.memlens_json is not None:
            if a.memlens_images is None:
                raise PreflightError("--memlens-images required with --memlens-json")
            report["memlens"] = dataset_probe_memlens(a.memlens_json, a.memlens_images)
        if a.mmlong_qa is not None:
            if a.mmlong_pdfs is None:
                raise PreflightError("--mmlong-pdfs required with --mmlong-qa")
            report["mmlong"] = dataset_probe_mmlong(a.mmlong_qa, a.mmlong_pdfs)
        report.update(assess_resources(host_resources(a.scratch),
                       report["checkpoint"]["total_bytes"] if report["checkpoint"] else None))
    except (PreflightError, OSError, ValueError) as exc:
        report.update({"status": "CPU_PREFLIGHT_BLOCKED",
                       "issues": [f"{type(exc).__name__}: {exc}"]})
    if a.report is not None:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "CPU_PREFLIGHT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
