#!/usr/bin/env python3
"""Machine-checkable execution inventory for the pilot (48 cells) and the full matrix (3528 cells).

Cells come from the repo's own plan code: ``vlmbench.config.planned_cells`` for the pilot and
``vlmbench.methods`` for the seven-arm matrix. Each cell gets a status from real local probes:
weights (size and SHA-256 against the local manifest), model revision, owner authorization,
vLLM importability, GPU compute capability and memory, context window, dataset presence and
checksum, dataset revision, disk headroom, and a GPU smoke receipt.

Status vocabulary: READY, BLOCKED_<REASON>, UNSUPPORTED, NOT_TESTED. A cell is READY only when
every check passes. A GPU smoke receipt is required, so no cell is READY without a real run.
Nothing here starts a GPU job, installs software or writes outside the report paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vlmbench.config import cell_id, load_config, planned_cells  # noqa: E402
from vlmbench.methods import NATIVE_AVAILABLE, manifest_from_path, plan  # noqa: E402

DEFAULT_MODELS_ROOT = Path("/mnt/ssd/genos2/research_workspace/models")
DEFAULT_AUTHORIZATION = REPO / "docs/harness/OWNER_AUTHORIZATION.json"
DEFAULT_FEASIBILITY = REPO / "docs/harness/DATASET_FEASIBILITY.json"  # read-only metadata, not approval
DEFAULT_RECEIPT_INDEX = REPO / "docs/harness/GPU_RECEIPT_INDEX.json"  # committed; a receipt counts only if listed here
OWNER_NAMESPACE = "vlmbench-owner-compute"  # ssh-keygen -Y signature namespace for owner compute approval
OWNER_PRINCIPAL = "owner"
LOCAL_WEIGHTS_MANIFEST = "starlight_local_manifest.json"
REQUIRED_CAPABILITY = 8.0       # BF16 native support (Ampere, sm_80 and newer). Assumption, stated in report.
REQUIRED_VRAM_MIB = 39000       # matches vlmbench.config.validate (A100 40GB contract)
OUTPUT_TOKENS = 256
SHA40 = re.compile(r"^[0-9a-f]{40}$")

STATUS_READY = "READY"
STATUS_UNSUPPORTED = "UNSUPPORTED"
STATUS_NOT_TESTED = "NOT_TESTED"
ARM_CODE = "UNSUPPORTED_ARM_NO_QUALIFIED_BACKEND"

# Ordered: the first failing check gives the primary status. Every failing check is listed.
CHECKS = (
    ("arm_supported", ARM_CODE),
    ("weights_present", "BLOCKED_WEIGHTS_MISSING"),
    ("weights_verified", "BLOCKED_WEIGHTS_MISMATCH"),
    ("model_revision_pinned", "BLOCKED_REVISION_UNPINNED"),
    ("owner_authorization", "BLOCKED_OWNER_AUTHORIZATION"),
    ("vllm_importable", "BLOCKED_VLLM_NOT_INSTALLED"),
    ("gpu_compute_capability", "BLOCKED_GPU_CAPABILITY"),
    ("gpu_memory", "BLOCKED_GPU_MEMORY"),
    ("context_fits", "BLOCKED_CONTEXT_EXCEEDS_WINDOW"),
    ("dataset_present", "BLOCKED_DATASET_MISSING"),
    ("dataset_checksum", "BLOCKED_DATASET_UNVERIFIED"),
    ("dataset_revision_pinned", "BLOCKED_REVISION_UNPINNED"),
    ("disk_headroom", "BLOCKED_DISK_HEADROOM"),
    ("gpu_smoke_receipt", None),  # NOT_TESTED when absent; never a blocker by itself
)
STATUS_VOCABULARY = (STATUS_READY, STATUS_UNSUPPORTED, STATUS_NOT_TESTED) + tuple(
    sorted({code for _, code in CHECKS if code} - {ARM_CODE}))


# ---------------------------------------------------------------- enumeration

def enumerate_pilot_cells(config_path: Path) -> list[dict]:
    config = load_config(config_path)
    out = []
    for cell in planned_cells(config):
        out.append({"matrix": "pilot", "cell_id": cell_id(cell), "arm": cell["mode"],
                    "model": cell["model"], "dataset": cell["dataset"],
                    "input_length": cell["input_length"], "concurrency": cell["concurrency"],
                    "repetition": cell["repetition"], "output_tokens": cell["output_tokens"]})
    return out


def enumerate_matrix_cells(manifest_path: Path) -> list[dict]:
    manifest = manifest_from_path(manifest_path)
    out = []
    for model in manifest["models"]:
        for arm in manifest["methods"]:
            for dataset in manifest["datasets"]:
                for n in manifest["input_lengths"]:
                    for ccu in manifest["ccu"]:
                        for rep in range(manifest["repetitions"]):
                            cell = {"dataset": dataset, "model": model, "mode": arm,
                                    "input_length": n, "concurrency": ccu, "repetition": rep}
                            out.append({"matrix": "full", "cell_id": cell_id(cell), "arm": arm,
                                        "model": model, "dataset": dataset, "input_length": n,
                                        "concurrency": ccu, "repetition": rep,
                                        "output_tokens": manifest["output_tokens"]})
    expected = plan(manifest)["total_serving_cells"]
    if len(out) != expected:
        raise ValueError(f"matrix enumeration {len(out)} != methods.plan total {expected}")
    return out


# ---------------------------------------------------------------- classification (pure)

def _check(name: str, outcome: str) -> str:
    return outcome


def classify_cell(cell: dict, env: dict) -> dict:
    """Return status, blockers, not_tested and per-check outcomes for one cell.

    Pure function of the cell and the probed environment, so tests can inject any state.
    """
    model, dataset, arm = cell["model"], cell["dataset"], cell["arm"]
    w = env["weights"].get(model, {"present": False})
    gpu = env.get("gpu")
    ds = env["datasets"].get(dataset, {"present": False, "checksum_verified": False})
    max_pos = w.get("max_position_embeddings")
    window_limits = [x for x in (max_pos, env.get("serving_max_model_len")) if x]
    need = cell["input_length"] + cell.get("output_tokens", OUTPUT_TOKENS)
    revision = env["revisions"].get(model)
    smoke = env["receipts"].get(cell["cell_id"])

    outcomes = {}
    outcomes["arm_supported"] = "PASS" if arm in NATIVE_AVAILABLE else "FAIL"
    outcomes["weights_present"] = "PASS" if w.get("present") else "FAIL"
    if not w.get("present"):
        outcomes["weights_verified"] = "NOT_TESTED"
    elif w.get("manifest_pinned") and w.get("hashes_ok") is False:
        outcomes["weights_verified"] = "FAIL_MISMATCH"
    elif not w.get("manifest_verified"):
        # a manifest counts only when pinned by hash and covering every weight file (probe_weights)
        outcomes["weights_verified"] = "FAIL_UNVERIFIED"
    else:
        outcomes["weights_verified"] = "PASS"
    outcomes["model_revision_pinned"] = "PASS" if (revision and not is_degenerate_sha(revision)) else "FAIL"
    outcomes["owner_authorization"] = "PASS" if env["authorization_ok"] else "FAIL"
    outcomes["vllm_importable"] = "PASS" if env["vllm_installed"] else "FAIL"
    if gpu is None:
        outcomes["gpu_compute_capability"] = "FAIL"
        outcomes["gpu_memory"] = "NOT_TESTED"
    else:
        outcomes["gpu_compute_capability"] = "PASS" if gpu["compute_cap"] >= REQUIRED_CAPABILITY else "FAIL"
        outcomes["gpu_memory"] = "PASS" if gpu["memory_total_mib"] >= REQUIRED_VRAM_MIB else "FAIL"
    if max_pos is None:
        outcomes["context_fits"] = "NOT_TESTED"
    else:
        outcomes["context_fits"] = "PASS" if need <= min(window_limits) else "FAIL"
    outcomes["dataset_present"] = "PASS" if ds.get("present") else "FAIL"
    outcomes["dataset_checksum"] = ("NOT_TESTED" if not ds.get("present")
                                    else "PASS" if ds.get("checksum_verified") else "FAIL")
    ds_rev = env["dataset_revisions"].get(dataset)
    outcomes["dataset_revision_pinned"] = "PASS" if (ds_rev and not is_degenerate_sha(ds_rev)) else "FAIL"
    outcomes["disk_headroom"] = "PASS" if env["disk_free_gib"] >= env["min_free_gib"] else "FAIL"
    outcomes["gpu_smoke_receipt"] = "PASS" if smoke else "NOT_TESTED"

    blockers, not_tested = [], []
    for name, code in CHECKS:
        out = outcomes[name]
        if out.startswith("FAIL"):
            if name == "weights_verified" and out == "FAIL_UNVERIFIED":
                blockers.append("BLOCKED_WEIGHTS_UNVERIFIED")
            elif name == "weights_verified" and out == "FAIL_MISMATCH":
                blockers.append("BLOCKED_WEIGHTS_MISMATCH")
            elif name == "arm_supported":
                blockers.append(ARM_CODE)
            else:
                blockers.append(code)
        elif out == "NOT_TESTED":
            not_tested.append(name)

    if outcomes["arm_supported"] == "FAIL":
        status = STATUS_UNSUPPORTED
    elif blockers:
        status = blockers[0]
    elif not_tested:
        status = STATUS_NOT_TESTED
    else:
        status = STATUS_READY
    checks = {name: ("PASS" if outcomes[name] == "PASS" else
                     "FAIL" if outcomes[name].startswith("FAIL") else "NOT_TESTED")
              for name, _ in CHECKS}
    return {"status": status, "blockers": blockers, "not_tested": not_tested, "checks": checks}


def summarize(statuses: list[dict]) -> dict:
    counts = {s: 0 for s in STATUS_VOCABULARY}
    blocker_counts: dict[str, int] = {}
    for st in statuses:
        counts[st["status"]] = counts.get(st["status"], 0) + 1
        for code in st["blockers"]:
            blocker_counts[code] = blocker_counts.get(code, 0) + 1
    return {"cell_count": len(statuses), "counts": counts,
            "blocker_counts": dict(sorted(blocker_counts.items()))}


# ---------------------------------------------------------------- probes (I/O)

def is_degenerate_sha(value: str | None) -> bool:
    """A 40-hex string that is a placeholder (one or two repeated characters) is not a pin."""
    return not value or not SHA40.match(value) or len(set(value.lower())) <= 3


def probe_weights(model_dir: Path, manifest_path: Path | None, *, hash_files: bool,
                  pinned_manifest_sha256: str | None = None) -> dict:
    result = {"path": str(model_dir), "present": False, "manifest_present": False,
              "manifest_pinned": False, "manifest_verified": False,
              "sizes_ok": None, "hashes_ok": None, "max_position_embeddings": None}
    config = model_dir / "config.json"
    if not (model_dir.is_dir() and config.is_file()):
        return result
    result["present"] = True
    try:
        cfg = json.loads(config.read_text(encoding="utf8"))
        text = cfg.get("text_config", cfg)
        result["max_position_embeddings"] = int(text["max_position_embeddings"])
    except (KeyError, ValueError, json.JSONDecodeError):
        result["max_position_embeddings"] = None
    if manifest_path is None or not manifest_path.is_file():
        return result
    result["manifest_present"] = True
    manifest_bytes = manifest_path.read_bytes()
    result["manifest_pinned"] = bool(pinned_manifest_sha256) and sha256_hex(manifest_bytes) == pinned_manifest_sha256
    files = json.loads(manifest_bytes.decode("utf8"))["files"]
    weight_files = {p.name for p in model_dir.glob("*.safetensors")}
    covered = bool(files) and weight_files <= set(files)
    sizes_ok, hashes_ok = True, True
    for rel, rec in files.items():
        path = model_dir / rel
        if not path.is_file() or path.stat().st_size != rec["size"]:
            sizes_ok = False
            hashes_ok = False
            continue
        if hash_files:
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            if digest.hexdigest() != rec["sha256"]:
                hashes_ok = False
    result["sizes_ok"] = sizes_ok
    result["hashes_ok"] = hashes_ok if hash_files else None
    result["manifest_verified"] = bool(result["manifest_pinned"] and covered and sizes_ok
                                       and result["hashes_ok"] is True)
    return result


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def probe_gpu(runner=subprocess.run) -> dict | None:
    args = ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total",
            "--format=csv,noheader,nounits"]
    try:
        done = runner(args, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((x for x in done.stdout.splitlines() if x.strip()), "")
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return {"name": parts[0], "compute_cap": float(parts[1]), "memory_total_mib": int(float(parts[2]))}
    except ValueError:
        return None


def verify_owner_authorization(record_path: Path, campaign_id: str, plan_path: Path,
                               allowed_signers: Path | None, runner=subprocess.run) -> dict:
    """Owner approval counts only as an ssh-keygen signature over the record, by a key in an
    owner signer file kept outside the repository, naming this campaign and this plan's hash.
    A bare JSON file, however it reads, never authorizes.
    """
    out = {"verified": False, "reason": ""}
    sig = record_path.with_name(record_path.name + ".sig")
    if not record_path.is_file() or not sig.is_file():
        out["reason"] = "no signed owner record"
        return out
    if allowed_signers is None or not allowed_signers.is_file():
        out["reason"] = "no owner signer file (kept outside the repository)"
        return out
    try:
        record = json.loads(record_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        out["reason"] = "owner record is not valid JSON"
        return out
    if not plan_path.is_file():
        out["reason"] = "plan file missing"
        return out
    plan_sha = sha256_hex(plan_path.read_bytes())
    entries = [c for c in record.get("campaigns", []) if isinstance(c, dict)]
    match = [c for c in entries if c.get("campaign_id") == campaign_id and c.get("plan_sha256") == plan_sha]
    if not match:
        out["reason"] = "record does not name this campaign with this plan hash"
        return out
    if not record.get("resource") or not record.get("approved_on"):
        out["reason"] = "record lacks resource or approval date"
        return out
    cmd = ["ssh-keygen", "-Y", "verify", "-f", str(allowed_signers), "-I", OWNER_PRINCIPAL,
           "-n", OWNER_NAMESPACE, "-s", str(sig)]
    try:
        done = runner(cmd, input=record_path.read_bytes(), capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        out["reason"] = "signature check could not run"
        return out
    if done.returncode != 0:
        out["reason"] = "signature does not verify against the owner signer file"
        return out
    out["verified"] = True
    out["reason"] = "signed by an owner key for this campaign and plan"
    return out


def verify_vllm(pinned_version: str | None, runner=subprocess.run) -> dict:
    """vLLM counts only when importing it and its compiled kernels succeeds and the version equals
    the configured pin. A stub package or an unpinned install is not verified."""
    out = {"verified": False, "installed": False, "kernels_importable": False, "version": None,
           "pinned_version": pinned_version}
    if not pinned_version:
        out["reason"] = "no vLLM version pinned in config"
        return out
    code = "import vllm, vllm._C; print(vllm.__version__)"
    try:
        done = runner([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        out["reason"] = "could not run the import check"
        return out
    if done.returncode != 0:
        out["reason"] = "import of vllm or vllm._C failed"
        return out
    version = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else ""
    out.update(installed=True, kernels_importable=True, version=version)
    out["verified"] = version == pinned_version
    out["reason"] = "pinned version and kernels import" if out["verified"] else "version differs from pin"
    return out


def probe_dataset(spec: dict) -> dict:
    path = Path(spec["native_path"]) if spec.get("native_path") else None
    present = bool(path and path.exists())
    expected = spec.get("native_sha256")
    checksum_verified = False
    if present and expected and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        checksum_verified = digest.hexdigest() == expected
    return {"path": str(path) if path else None, "present": present,
            "checksum_verified": checksum_verified}


def load_receipts(receipts_dir: Path | None, index_path: Path | None = DEFAULT_RECEIPT_INDEX) -> dict:
    """A receipt counts only when the committed index lists it with a matching SHA-256 and it
    reports PASS_MEASURED for that same cell. A file in the directory alone never counts."""
    if receipts_dir is None or index_path is None or not index_path.is_file():
        return {}
    try:
        entries = json.loads(index_path.read_text(encoding="utf8")).get("receipts", [])
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    found = {}
    for entry in entries:
        path = receipts_dir / str(entry.get("file", ""))
        if not path.is_file() or sha256_hex(path.read_bytes()) != entry.get("sha256"):
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError:
            continue
        if rec.get("status") == "PASS_MEASURED" and rec.get("cell_id") == entry.get("cell_id"):
            found[entry["cell_id"]] = True
    return found


def git_head() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10, check=True)
        return done.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"


# ---------------------------------------------------------------- report

def build_report(pilot_config: Path, matrix_manifest: Path, *, models_root: Path,
                 authorization: Path, receipts_dir: Path | None, min_free_gib: float,
                 hash_weights: bool, probe_gpu_fn=probe_gpu,
                 feasibility_path: Path | None = DEFAULT_FEASIBILITY,
                 allowed_signers: Path | None = None,
                 pinned_manifest_sha256: dict[str, str] | None = None,
                 owner_record: Path | None = None,
                 receipt_index: Path | None = DEFAULT_RECEIPT_INDEX) -> dict:
    config = load_config(pilot_config)
    manifest = manifest_from_path(matrix_manifest)
    pilot_cells = enumerate_pilot_cells(pilot_config)
    matrix_cells = enumerate_matrix_cells(matrix_manifest)

    models = {m["id"]: m for m in config["models"]}
    weights, revisions = {}, {}
    for mid in sorted(set(manifest["models"]) | {c["model"] for c in pilot_cells}):
        hf_id = models[mid]["hf_id"]
        model_dir = models_root / hf_id
        manifest_file = model_dir / LOCAL_WEIGHTS_MANIFEST
        weights[mid] = probe_weights(model_dir, manifest_file if manifest_file.is_file() else None,
                                     hash_files=hash_weights,
                                     pinned_manifest_sha256=(pinned_manifest_sha256 or {}).get(mid))
        rev = models[mid].get("revision")
        revisions[mid] = rev if rev else None
    datasets = {d["id"]: probe_dataset(d) for d in config["datasets"]}
    dataset_revisions = {d["id"]: d.get("revision") or None for d in config["datasets"]}
    gpu = probe_gpu_fn()
    receipts = load_receipts(receipts_dir, receipt_index)
    disk_free = shutil.disk_usage(models_root if models_root.exists() else REPO).free / 2**30
    campaign_pilot = config["campaign_id"]
    record = owner_record or authorization
    authorized_pilot = verify_owner_authorization(record, campaign_pilot, pilot_config, allowed_signers)["verified"]
    authorized_full = verify_owner_authorization(record, manifest["campaign_id"], matrix_manifest,
                                                 allowed_signers)["verified"]
    vllm = verify_vllm(config.get("vllm_pin"))["verified"]

    def env_for(serving_max, authorized):
        return {"weights": weights, "revisions": revisions, "dataset_revisions": dataset_revisions,
                "authorization_ok": authorized, "vllm_installed": vllm, "gpu": gpu,
                "serving_max_model_len": serving_max, "datasets": datasets,
                "disk_free_gib": disk_free, "min_free_gib": min_free_gib, "receipts": receipts}

    pilot_env = env_for(config["serving"]["max_model_len"], authorized_pilot)
    full_env = env_for(None, authorized_full)
    cells = []
    for cell, env in [(c, pilot_env) for c in pilot_cells] + [(c, full_env) for c in matrix_cells]:
        verdict = classify_cell(cell, env)
        cells.append({**cell, **verdict})
    pilot_summary = summarize([c for c in cells if c["matrix"] == "pilot"])
    full_summary = summarize([c for c in cells if c["matrix"] == "full"])
    return {
        "schema": "harness_inventory/1",
        "head": git_head(),
        "status_vocabulary": list(STATUS_VOCABULARY),
        "rules": {
            "required_compute_capability": REQUIRED_CAPABILITY,
            "required_vram_mib": REQUIRED_VRAM_MIB,
            "output_tokens": OUTPUT_TOKENS,
            "min_free_gib": min_free_gib,
            "min_free_gib_is_guard_not_measurement": True,
            "ready_requires": "every check PASS, including a verified gpu_smoke_receipt",
        },
        "environment": {
            "models_root": str(models_root),
            "weights": weights,
            "revisions_in_config": revisions,
            "vllm_installed": vllm,
            "gpu": gpu,
            "disk_free_gib": round(disk_free, 3),
            "authorization_file": str(authorization),
            "authorization_ok": {"pilot": authorized_pilot, "full": authorized_full},
            "datasets": datasets,
            "dataset_revisions_in_config": dataset_revisions,
            "receipts_loaded": len(receipts),
            "weights_hashed": hash_weights,
        },
        "matrices": {
            "pilot": {"campaign_id": campaign_pilot, **pilot_summary,
                      "source": "vlmbench.config.planned_cells (" + str(pilot_config.name) + ")"},
            "full": {"campaign_id": manifest["campaign_id"], **full_summary,
                     "source": "vlmbench.methods.plan + manifest grid (" + str(matrix_manifest.name) + ")"},
        },
        "dataset_feasibility": _load_feasibility(feasibility_path),
        "cells": cells,
    }


def _load_feasibility(path: Path | None) -> dict | None:
    """Resolved revisions and sizes from read-only metadata. Never an approval, never a status input."""
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def render_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        "# Pilot and full-matrix execution inventory (ITER2 D2-1)", "",
        f"Generated by `scripts/harness_inventory.py` at HEAD `{report['head']}`. "
        "Regenerate with the command in §5. Per-cell records are in `docs/harness/PILOT_INVENTORY.json`.", "",
        "A cell is **READY** only when every check passes, including a verified GPU smoke receipt. "
        "No GPU job has run, so no cell can be READY here.", "",
        "## 1. Counts by status", "",
        "| Matrix | Cells | " + " | ".join(report["status_vocabulary"]) + " |",
        "|---|---:|" + "---:|" * len(report["status_vocabulary"]),
    ]
    for name in ("pilot", "full"):
        m = report["matrices"][name]
        row = " | ".join(str(m["counts"].get(s, 0)) for s in report["status_vocabulary"])
        lines.append(f"| {name} ({m['campaign_id']}) | {m['cell_count']} | {row} |")
    lines += ["", "## 2. Blocker counts (a cell can carry several)", "",
              "| Blocker | Pilot cells | Full-matrix cells |", "|---|---:|---:|"]
    codes = sorted(set(report["matrices"]["pilot"]["blocker_counts"]) |
                   set(report["matrices"]["full"]["blocker_counts"]))
    for code in codes:
        lines.append(f"| {code} | {report['matrices']['pilot']['blocker_counts'].get(code, 0)} | "
                     f"{report['matrices']['full']['blocker_counts'].get(code, 0)} |")
    lines += ["", "## 3. Host state observed by the probes", "",
              f"- Models root: `{env['models_root']}`",
              f"- Weights: " + "; ".join(
                  f"`{mid}` present={w['present']}, manifest={w['manifest_present']}, "
                  f"sizes_ok={w['sizes_ok']}, sha256_ok={w['hashes_ok']}, "
                  f"max_position={w['max_position_embeddings']}"
                  for mid, w in env["weights"].items()),
              f"- vLLM importable: {env['vllm_installed']}",
              f"- GPU (nvidia-smi): {env['gpu']}",
              f"- Disk free on models root: {env['disk_free_gib']} GiB (guard: {report['rules']['min_free_gib']} GiB)",
              f"- Owner authorization file `{env['authorization_file']}`: {env['authorization_ok']}",
              f"- Datasets: " + "; ".join(f"`{k}` present={v['present']}" for k, v in env["datasets"].items()),
              f"- Revisions in config: models {env['revisions_in_config']}; datasets {env['dataset_revisions_in_config']}",
              f"- GPU smoke receipts loaded: {env['receipts_loaded']}", "",
              "## 4. Assumptions and limits", "",
              f"- BF16 native support requires compute capability >= {REQUIRED_CAPABILITY} (Ampere or newer). "
              "This is an assumption about the target runtime, not a measurement.",
              f"- VRAM floor {REQUIRED_VRAM_MIB} MiB is the A100 40GB contract from `vlmbench.config.validate`.",
              f"- Disk guard {report['rules']['min_free_gib']} GiB is a conservative guard, not a measured requirement.",
              "- The 0.8B weights are checked against the local manifest (`starlight_local_manifest.json`). That manifest's "
              "revision is not an owner pin, so the config revision remains UNPINNED.",
              "- Datasets are checked at the config's `native_path` (Drive paths). None exist on this host.",
              "- Full-matrix cells use the pilot config's dataset paths and authorization record, and have no serving "
              "`max_model_len` (the 7-method manifest does not set one).",
              "- Nothing here starts a GPU job, loads a model or contacts the network.", "",
              "## 5. Regenerate", "",
              "```", f"{sys.executable} scripts/harness_inventory.py", "```", ""]
    feas = report.get("dataset_feasibility")
    if feas:
        mem, mmlong = feas["memlens"], feas["mmlongbench_doc"]
        lines += ["## 6. Dataset feasibility (read-only metadata, not owner-approved)", "",
                  f"Resolved {feas['resolved_on']}; `owner_approved` = {feas['owner_approved']}. "
                  "These revisions do not unpin the config, and no status above depends on them.", "",
                  f"- MemLens `{mem['hf_dataset']}` @ `{mem['hf_revision_sha']}`, 789 items per context config, "
                  f"{mem['total_repo_bytes']:,} bytes total, licence {mem['hf_license_card']}.",
                  f"- MMLongBench-Doc `{mmlong['hf_dataset']}` @ `{mmlong['hf_revision_sha']}`, "
                  f"{mmlong['qa_counts']['answerable'] + mmlong['qa_counts']['unanswerable_answer_Not_answerable']} QA rows, "
                  f"{mmlong['documents']['count']} documents, licence card: {mmlong['hf_license_card']}.",
                  "- Official scoring needs an LLM answer-extraction step; it is BLOCKED_NO_JUDGE. Details are in "
                  "`docs/harness/DATASET_FEASIBILITY.json`.", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--pilot-config", type=Path, default=REPO / "configs/a100_qwen_priority_pilot.json")
    p.add_argument("--matrix-manifest", type=Path, default=REPO / "configs/a100_qwen_7methods_full.json")
    p.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    p.add_argument("--authorization", type=Path, default=DEFAULT_AUTHORIZATION)
    p.add_argument("--receipts-dir", type=Path, default=None)
    p.add_argument("--min-free-gib", type=float, default=20.0)
    p.add_argument("--no-hash-weights", action="store_true", help="skip SHA-256 of weight files")
    p.add_argument("--out-json", type=Path, default=REPO / "docs/harness/PILOT_INVENTORY.json")
    p.add_argument("--out-md", type=Path, default=REPO / "docs/harness/PILOT_INVENTORY.md")
    args = p.parse_args(argv)
    report = build_report(args.pilot_config, args.matrix_manifest, models_root=args.models_root,
                          authorization=args.authorization, receipts_dir=args.receipts_dir,
                          min_free_gib=args.min_free_gib, hash_weights=not args.no_hash_weights)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf8")
    args.out_md.write_text(render_markdown(report), encoding="utf8")
    for name in ("pilot", "full"):
        m = report["matrices"][name]
        print(f"{name}: {m['cell_count']} cells {m['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
