#!/usr/bin/env python3
"""Machine-checkable execution inventory for the pilot (48 cells) and the full matrix (3528 cells).

Cells come from the repo's own plan code: ``vlmbench.config.planned_cells`` for the pilot and
``vlmbench.methods`` for the seven-arm matrix. Each cell gets a status from real local probes:
weights (size and SHA-256 against the local manifest), model revision, owner authorization,
vLLM importability, GPU compute capability and memory, context window, dataset presence and
checksum, dataset revision, disk headroom, and a GPU smoke receipt.

Status vocabulary: READY, BLOCKED_<REASON>, UNSUPPORTED, NOT_TESTED. A cell is READY only when
every check passes. A GPU smoke receipt is required, so no cell is READY without a real run.
ITER3 C1: every pin (revisions, weights manifest, dataset checksum, vLLM version and RECORD, receipt
index, GPU name and uuid) comes only from the owner-signed record. Config values and flags are never pins.
Each cell also gets a ladder state (see LADDER_RUNGS). Trust model: host-produced evidence has no external
trust anchor except the owner's signing key, which must stay off this host. See docs/harness/READINESS.md.
Nothing here starts a GPU job, installs software or writes outside the report paths.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

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
    ("live_probe", "BLOCKED_LIVE_PROBE_FAILED"),  # N-1: an executed probe in this run, not a file on disk
    ("gpu_identity", "BLOCKED_GPU_IDENTITY"),     # ITER3 C1: name AND uuid equal the owner-signed GPU pin
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

# ITER3 C1: the explicit state ladder. A cell's ladder_state is the highest rung such that every check on
# that rung and on every lower rung passes. BENCHMARK_CELL_COMPLETED is part of the vocabulary but NOTHING in
# this inventory can reach it: a finished benchmark cell is recorded by the run itself, and no attestation
# mechanism for that exists here. READY is exactly GPU_SMOKE_VERIFIED with every check passing.
LADDER_RUNGS = (
    ("PLAN_AUTHORIZED", ("owner_authorization",)),
    ("STATIC_PREREQUISITES_VALID", ("arm_supported", "weights_present", "weights_verified",
                                    "model_revision_pinned", "dataset_present", "dataset_checksum",
                                    "dataset_revision_pinned", "disk_headroom", "context_fits")),
    ("LIVE_ENVIRONMENT_VERIFIED", ("vllm_importable", "gpu_identity", "gpu_compute_capability",
                                   "gpu_memory", "live_probe")),
    ("GPU_SMOKE_VERIFIED", ("gpu_smoke_receipt",)),
)
LADDER_UNREACHABLE = "BENCHMARK_CELL_COMPLETED"
LADDER_STATES = ("NONE",) + tuple(name for name, _ in LADDER_RUNGS) + (LADDER_UNREACHABLE,)
_CHECK_NAMES = {name for name, _ in CHECKS}
if sorted(n for _, names in LADDER_RUNGS for n in names) != sorted(_CHECK_NAMES):
    raise RuntimeError("ladder rungs must cover every check exactly once")  # explicit: survives python -O


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
    outcomes["live_probe"] = "PASS" if (env.get("live_probe") or {}).get("ok") is True else "FAIL"
    # ITER3 C1: identity is the signed pin, never a config value. Name AND uuid must both match it.
    gpu_pin = env.get("gpu_pin") or {}
    if gpu is None or not gpu_pin.get("name") or not gpu_pin.get("uuid"):
        outcomes["gpu_identity"] = "FAIL"
    elif gpu.get("name") == gpu_pin["name"] and gpu.get("uuid") == gpu_pin["uuid"]:
        outcomes["gpu_identity"] = "PASS"
    else:
        outcomes["gpu_identity"] = "FAIL"
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
    return {"status": status, "blockers": blockers, "not_tested": not_tested, "checks": checks,
            "ladder_state": ladder_state(checks)}


def ladder_state(checks: dict) -> str:
    """Highest rung whose checks, and every lower rung's, all PASS. Never BENCHMARK_CELL_COMPLETED."""
    state = "NONE"
    for rung, names in LADDER_RUNGS:
        if not all(checks.get(n) == "PASS" for n in names):
            break
        state = rung
    return state


def summarize(statuses: list[dict]) -> dict:
    counts = {s: 0 for s in STATUS_VOCABULARY}
    ladder = {s: 0 for s in LADDER_STATES}
    blocker_counts: dict[str, int] = {}
    for st in statuses:
        counts[st["status"]] = counts.get(st["status"], 0) + 1
        if "ladder_state" in st:
            ladder[st["ladder_state"]] = ladder.get(st["ladder_state"], 0) + 1
        for code in st["blockers"]:
            blocker_counts[code] = blocker_counts.get(code, 0) + 1
    return {"cell_count": len(statuses), "counts": counts, "ladder_counts": ladder,
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
    # C-1 (R1): a config alone is not weights. A shard counts only when it is a non-empty safetensors file.
    weight_files = {p.name for p in model_dir.glob("*.safetensors") if p.is_file() and p.stat().st_size > 0}
    if not weight_files:
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
    # C-1 (R1): the manifest must list every shard on disk, at least one shard, and each listed shard non-empty.
    # A manifest that lists only the config (vacuously "covering" no shards) does not verify anything.
    listed_weights = {rel for rel in files if rel.endswith(".safetensors")}
    covered = (bool(weight_files) and weight_files <= listed_weights
               and all(isinstance(files[rel], dict) and files[rel].get("size", 0) > 0 for rel in listed_weights))
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
    args = ["nvidia-smi", "--query-gpu=name,uuid,compute_cap,memory.total",
            "--format=csv,noheader,nounits"]
    try:
        done = runner(args, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((x for x in done.stdout.splitlines() if x.strip()), "")
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 4:
        return None
    try:
        # Self-reported by the host: this is identity evidence only when it equals the owner-signed pin.
        return {"name": parts[0], "uuid": parts[1], "compute_cap": float(parts[2]),
                "memory_total_mib": int(float(parts[3]))}
    except ValueError:
        return None


def verify_owner_authorization(record_path: Path, campaign_id: str, plan_path: Path,
                               allowed_signers: Path | None, runner=subprocess.run) -> dict:
    """Owner approval counts only as an ssh-keygen signature over the record, by a key in an
    owner signer file kept outside the repository, naming this campaign and this plan's hash.
    A bare JSON file, however it reads, never authorizes.
    """
    out = {"verified": False, "reason": "", "resource": None, "pins": {}}
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
    out["resource"] = str(record["resource"])
    # ITER3 C1: the pins live only in the signed record. Config values and flags never supply them.
    out["pins"] = record.get("pins") if isinstance(record.get("pins"), dict) else {}
    out["reason"] = "signed by an owner key for this campaign and plan"
    return out


def signed_pin(pins: dict, *path: str) -> str | None:
    """Read one string pin from the signed record. Anything that is not a non-empty string is no pin."""
    node: Any = pins
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, str) and node else None


VLLM_CHECK_CODE = r"""
import hashlib, importlib.metadata as md, json, sys
import vllm, vllm._C
dist = md.distribution("vllm")
record = (dist.read_text("RECORD") or "").encode("utf8")
print(json.dumps({"module_version": vllm.__version__, "dist_version": dist.version,
                  "record_sha256": hashlib.sha256(record).hexdigest()}))
"""


def verify_vllm(pinned_version: str | None, runner=subprocess.run,
                pinned_record_sha256: str | None = None) -> dict:
    """vLLM counts only when importing it and its compiled kernels succeeds, the module and distribution
    versions equal the signed version pin, and the installed distribution RECORD hashes to the signed pin.
    A stub package, an unpinned install or a different build is not verified. Both pins come from the
    owner-signed record. The RECORD hash is produced on this host: it defeats a casual swap, not a
    determined local actor."""
    out = {"verified": False, "installed": False, "kernels_importable": False, "version": None,
           "record_sha256": None, "pinned_version": pinned_version,
           "pinned_record_sha256": pinned_record_sha256}
    if not pinned_version or not pinned_record_sha256:
        out["reason"] = "no signed vLLM pin (version and distribution record)"
        return out
    try:
        done = runner([sys.executable, "-I", "-c", VLLM_CHECK_CODE], capture_output=True, text=True, timeout=180)
    except (OSError, subprocess.SubprocessError):
        out["reason"] = "could not run the import check"
        return out
    if done.returncode != 0:
        out["reason"] = "import of vllm or vllm._C failed"
        return out
    try:
        info = json.loads(done.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError, AttributeError):
        out["reason"] = "import check produced no metadata"
        return out
    out.update(installed=True, kernels_importable=True, version=info.get("dist_version"),
               record_sha256=info.get("record_sha256"))
    if info.get("module_version") != pinned_version or info.get("dist_version") != pinned_version:
        out["reason"] = "version differs from the signed pin"
    elif info.get("record_sha256") != pinned_record_sha256:
        out["reason"] = "distribution RECORD differs from the signed pin"
    else:
        out["verified"] = True
        out["reason"] = "signed version and distribution record match, kernels import"
    return out


# N-1: the live probe. It runs in a separate isolated interpreter (python -I, so PYTHONPATH and user
# site packages are ignored) with a fresh nonce. It executes the checks itself; the parent evaluates the
# returned report. A determined local actor who controls this machine can still forge the output: READY
# means "preconditions probed in this run", not an attestation.
LIVE_PROBE_CODE = r"""
import hashlib, importlib.metadata as md, json, os, subprocess, sys
nonce, spec = sys.argv[1], json.loads(sys.argv[2])
out = {"nonce": nonce, "pid": os.getpid(), "python": sys.executable}
try:
    dist_obj = md.distribution("vllm")
    dist = dist_obj.version
    record = (dist_obj.read_text("RECORD") or "").encode("utf8")
    out["vllm"] = {"dist_version": dist, "record_sha256": hashlib.sha256(record).hexdigest()}
except md.PackageNotFoundError:
    out["vllm"] = {"dist_version": None, "record_sha256": None}
try:
    import vllm, vllm._C
    out["vllm"]["module_version"] = getattr(vllm, "__version__", None)
    out["vllm"]["module_file"] = getattr(vllm, "__file__", None)
except Exception as exc:
    out["vllm"]["import_error"] = type(exc).__name__ + ": " + str(exc)
smi = next((p for p in ("/usr/bin/nvidia-smi", "/usr/local/bin/nvidia-smi") if os.path.isfile(p)), None)
out["nvidia_smi"] = {"path": smi}
if smi:
    try:
        line = subprocess.run([smi, "-i", "0", "--query-gpu=name,uuid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30, check=True).stdout.strip().splitlines()[0]
        name, uuid = [part.strip() for part in line.split(",", 1)]
        out["nvidia_smi"].update(name=name, uuid=uuid)
    except Exception as exc:
        out["nvidia_smi"]["error"] = type(exc).__name__ + ": " + str(exc)
try:
    import torch
    x = torch.arange(16, dtype=torch.float32, device="cuda")
    out["cuda"] = {"device_name": torch.cuda.get_device_name(0),
                   "kernel_sum": float((x * 2 + 1).sum().item()),
                   "expected_sum": float(sum(2 * i + 1 for i in range(16))),
                   "capability": list(torch.cuda.get_device_capability(0))}
except Exception as exc:
    out["cuda"] = {"error": type(exc).__name__ + ": " + str(exc)}
weights = {}
for model_id, entry in spec.items():
    files = {}
    for rel in entry["files"]:
        try:
            digest, size = hashlib.sha256(), 0
            with open(os.path.join(entry["path"], rel), "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
                    size += len(chunk)
            files[rel] = {"sha256": digest.hexdigest(), "size": size}
        except OSError as exc:
            files[rel] = {"error": type(exc).__name__ + ": " + str(exc)}
    weights[model_id] = {"files": files}
out["weights"] = weights
print(json.dumps(out))
"""


def evaluate_probe(report: Any, *, nonce: str, parent_pid: int, resource_name: str | None,
                   vllm_pin: str | None, weight_specs: dict | None,
                   vllm_record_sha256: str | None = None, gpu_uuid: str | None = None) -> dict:
    """Pure check of one probe report against the signed plan. Every reason to refuse is listed.

    resource_name, vllm_pin, vllm_record_sha256, gpu_uuid and weight_specs must come from the owner-signed
    record. A missing value is refused, never skipped."""
    if not isinstance(report, dict):
        return {"ok": False, "reasons": ["probe output is not a JSON object"]}
    reasons: list[str] = []
    if report.get("nonce") != nonce:
        reasons.append("nonce mismatch: output is not from this probe execution")
    if report.get("pid") == parent_pid:
        reasons.append("probe ran in the same process as the parent")
    vllm = report.get("vllm") or {}
    if vllm.get("import_error"):
        reasons.append(f"vLLM import failed: {vllm['import_error']}")
    if not vllm.get("dist_version"):
        reasons.append("vLLM distribution metadata missing (stub or unmanaged package)")
    elif not vllm_pin:
        reasons.append("no signed vLLM version pin")
    elif vllm["dist_version"] != vllm_pin:
        reasons.append(f"vLLM version {vllm['dist_version']} differs from the signed pin {vllm_pin}")
    if not vllm_record_sha256:
        reasons.append("no signed vLLM distribution-record pin")
    elif vllm.get("record_sha256") != vllm_record_sha256:
        reasons.append("vLLM distribution RECORD differs from the signed pin")
    if not resource_name:
        reasons.append("no signed GPU name pin")
    smi = report.get("nvidia_smi") or {}
    if resource_name and smi.get("name") != resource_name:
        reasons.append(f"GPU name {smi.get('name')!r} differs from the signed pin {resource_name!r}")
    if not gpu_uuid:
        reasons.append("no signed GPU uuid pin")
    elif smi.get("uuid") != gpu_uuid:
        reasons.append("GPU uuid differs from the signed pin")
    cuda = report.get("cuda") or {}
    if "error" in cuda:
        reasons.append(f"CUDA kernel did not run: {cuda['error']}")
    else:
        if cuda.get("kernel_sum") != cuda.get("expected_sum"):
            reasons.append("CUDA kernel result is wrong")
        if smi.get("name") and cuda.get("device_name") != smi.get("name"):
            reasons.append("CUDA device differs from nvidia-smi device")
    if not weight_specs:
        reasons.append("weights are not pinned for every model")
    else:
        got_weights = report.get("weights") or {}
        for model_id, spec in weight_specs.items():
            got_files = (got_weights.get(model_id) or {}).get("files") or {}
            for rel, pinned in spec["files"].items():
                got = got_files.get(rel)
                if got is None:
                    reasons.append(f"{model_id}/{rel} not recomputed")
                elif "error" in got:
                    reasons.append(f"{model_id}/{rel}: {got['error']}")
                elif got.get("sha256") != pinned["sha256"]:
                    reasons.append(f"{model_id}/{rel}: recomputed sha256 differs from the pinned manifest")
                elif got.get("size") != pinned["size"]:
                    reasons.append(f"{model_id}/{rel}: size differs from the pinned manifest")
    return {"ok": not reasons, "reasons": reasons, "probe_pid": report.get("pid"),
            "nonce_echoed": report.get("nonce") == nonce}


def run_live_probe(resource_name: str | None, vllm_pin: str | None, weight_specs: dict | None, *,
                   runner=subprocess.run, timeout_s: float = 600,
                   vllm_record_sha256: str | None = None, gpu_uuid: str | None = None) -> dict:
    nonce = secrets.token_hex(16)
    cmd = [sys.executable, "-I", "-c", LIVE_PROBE_CODE, nonce, json.dumps(weight_specs or {})]
    try:
        done = runner(cmd, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "reasons": [f"live probe could not run: {type(exc).__name__}"]}
    lines = [line for line in (done.stdout or "").splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        report = None
    if report is None:
        return {"ok": False, "reasons": [f"live probe produced no JSON (exit {done.returncode})"]}
    return evaluate_probe(report, nonce=nonce, parent_pid=os.getpid(), resource_name=resource_name,
                          vllm_pin=vllm_pin, weight_specs=weight_specs,
                          vllm_record_sha256=vllm_record_sha256, gpu_uuid=gpu_uuid)


def probe_dataset(spec: dict, expected_sha256: str | None = None) -> dict:
    """The expected checksum is the owner-signed native_sha256. A checksum in the config (or written next
    to the file by the local actor) is never the expectation."""
    path = Path(spec["native_path"]) if spec.get("native_path") else None
    present = bool(path and path.exists())
    expected = expected_sha256
    checksum_verified = False
    if present and expected and path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        checksum_verified = digest.hexdigest() == expected
    return {"path": str(path) if path else None, "present": present,
            "checksum_verified": checksum_verified}


def load_receipts(receipts_dir: Path | None, index_path: Path | None = DEFAULT_RECEIPT_INDEX,
                  *, expected_index_sha256: str | None = None) -> dict:
    """A receipt counts only when the index file hashes to the owner-signed receipt_index_sha256, the index
    lists it with a matching SHA-256, and it reports PASS_MEASURED for that same cell. A file in the
    directory alone never counts, and an index that is not signed counts for nothing (ITER3 C1)."""
    if receipts_dir is None or index_path is None or not index_path.is_file() or not expected_index_sha256:
        return {}
    try:
        index_bytes = index_path.read_bytes()
    except OSError:
        return {}
    if sha256_hex(index_bytes) != expected_index_sha256:
        return {}
    try:
        entries = json.loads(index_bytes.decode("utf8")).get("receipts", [])
    except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
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
                 owner_record: Path | None = None,
                 receipt_index: Path | None = DEFAULT_RECEIPT_INDEX,
                 live_probe: bool = True) -> dict:
    config = load_config(pilot_config)
    manifest = manifest_from_path(matrix_manifest)
    pilot_cells = enumerate_pilot_cells(pilot_config)
    matrix_cells = enumerate_matrix_cells(matrix_manifest)

    models = {m["id"]: m for m in config["models"]}
    campaign_pilot = config["campaign_id"]
    record = owner_record or authorization
    auth_pilot = verify_owner_authorization(record, campaign_pilot, pilot_config, allowed_signers)
    auth_full = verify_owner_authorization(record, manifest["campaign_id"], matrix_manifest, allowed_signers)
    authorized_pilot, authorized_full = auth_pilot["verified"], auth_full["verified"]

    # ITER3 C1: every pin comes from the owner-signed record, and only from a record whose signature
    # verified. Config revisions, config checksums, config vllm_pin and command-line flags are reported
    # for transparency but never used as pins. The signature is the only external anchor, and it must
    # be checked against a signer file kept off this host.
    pins = (auth_pilot if authorized_pilot else auth_full)["pins"] if (authorized_pilot or authorized_full) else {}
    gpu_pin = {"name": signed_pin(pins, "gpu", "name"), "uuid": signed_pin(pins, "gpu", "uuid")}
    vllm_version_pin = signed_pin(pins, "vllm", "version")
    vllm_record_pin = signed_pin(pins, "vllm", "distribution_record_sha256")
    receipt_index_pin = signed_pin(pins, "receipt_index_sha256")

    weights, revisions = {}, {}
    for mid in sorted(set(manifest["models"]) | {c["model"] for c in pilot_cells}):
        hf_id = models[mid]["hf_id"]
        model_dir = models_root / hf_id
        manifest_file = model_dir / LOCAL_WEIGHTS_MANIFEST
        weights[mid] = probe_weights(model_dir, manifest_file if manifest_file.is_file() else None,
                                     hash_files=hash_weights,
                                     pinned_manifest_sha256=signed_pin(pins, "models", mid, "weights_manifest_sha256"))
        revisions[mid] = signed_pin(pins, "models", mid, "revision")
    datasets = {d["id"]: probe_dataset(d, expected_sha256=signed_pin(pins, "datasets", d["id"], "native_sha256"))
                for d in config["datasets"]}
    dataset_revisions = {d["id"]: signed_pin(pins, "datasets", d["id"], "revision") for d in config["datasets"]}
    gpu = probe_gpu_fn()
    receipts = load_receipts(receipts_dir, receipt_index, expected_index_sha256=receipt_index_pin)
    disk_free = shutil.disk_usage(models_root if models_root.exists() else REPO).free / 2**30
    vllm = verify_vllm(vllm_version_pin, pinned_record_sha256=vllm_record_pin)["verified"]

    # N-1: READY also needs a live probe executed in this run. Weight specs come only from pinned manifests.
    weight_specs = None
    if all(w.get("manifest_verified") for w in weights.values()):
        weight_specs = {}
        for mid in weights:
            model_dir = models_root / models[mid]["hf_id"]
            files = json.loads((model_dir / LOCAL_WEIGHTS_MANIFEST).read_text(encoding="utf8"))["files"]
            weight_specs[mid] = {"path": str(model_dir), "files": files}
    if not live_probe:
        live = {"ok": False, "reasons": ["plan-only mode: no live probe was run"]}
    elif not (authorized_pilot or authorized_full):
        live = {"ok": False, "reasons": ["not run: no verified owner authorization"]}
    else:
        live = run_live_probe(gpu_pin["name"], vllm_version_pin, weight_specs,
                              vllm_record_sha256=vllm_record_pin, gpu_uuid=gpu_pin["uuid"])

    def env_for(serving_max, authorized):
        return {"weights": weights, "revisions": revisions, "dataset_revisions": dataset_revisions,
                "authorization_ok": authorized, "vllm_installed": vllm, "gpu": gpu, "gpu_pin": gpu_pin,
                "serving_max_model_len": serving_max, "datasets": datasets,
                "disk_free_gib": disk_free, "min_free_gib": min_free_gib, "receipts": receipts,
                "live_probe": live}

    pilot_env = env_for(config["serving"]["max_model_len"], authorized_pilot)
    full_env = env_for(None, authorized_full)
    cells = []
    for cell, env in [(c, pilot_env) for c in pilot_cells] + [(c, full_env) for c in matrix_cells]:
        verdict = classify_cell(cell, env)
        cells.append({**cell, **verdict})
    pilot_summary = summarize([c for c in cells if c["matrix"] == "pilot"])
    full_summary = summarize([c for c in cells if c["matrix"] == "full"])
    return {
        "schema": "harness_inventory/2",
        "head": git_head(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "status_vocabulary": list(STATUS_VOCABULARY),
        "ladder": list(LADDER_STATES),
        "rules": {
            "required_compute_capability": REQUIRED_CAPABILITY,
            "required_vram_mib": REQUIRED_VRAM_MIB,
            "output_tokens": OUTPUT_TOKENS,
            "min_free_gib": min_free_gib,
            "min_free_gib_is_guard_not_measurement": True,
            "pins_source": "owner-signed record only (revisions, weights manifest, dataset native_sha256, "
                           "vLLM version and RECORD, receipt index, GPU name and uuid)",
            "ready_requires": ("every check PASS: a signed owner authorization, pins from the signed record, "
                               "weights matching the signed manifest, vLLM and GPU identity matching the signed "
                               "pins, a live probe executed in this run (nonce echo, CUDA kernel, recomputed "
                               "weight SHA-256), and a gpu_smoke_receipt listed in the signed receipt index"),
            "ladder_top_reachable": "GPU_SMOKE_VERIFIED (BENCHMARK_CELL_COMPLETED is not reachable here)",
        },
        "environment": {
            "models_root": str(models_root),
            "weights": weights,
            "signed_pins_used": pins,
            "config_values_ignored": {
                "revisions": {mid: models[mid].get("revision") for mid in models},
                "datasets_native_sha256": {d["id"]: d.get("native_sha256") for d in config["datasets"]},
                "vllm_pin": config.get("vllm_pin"),
            },
            "revisions_pinned": revisions,
            "vllm_installed": vllm,
            "gpu": gpu,
            "disk_free_gib": round(disk_free, 3),
            "authorization_file": str(authorization),
            "authorization_ok": {"pilot": authorized_pilot, "full": authorized_full},
            "datasets": datasets,
            "dataset_revisions_pinned": dataset_revisions,
            "receipts_loaded": len(receipts),
            "weights_hashed": hash_weights,
            "live_probe": live,
        },
        "mode": "live_probes" if live_probe else "plan_only",
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
        f"Generated by `scripts/harness_inventory.py` (script SHA-256 `{report['script_sha256'][:16]}…`) "
        f"at HEAD `{report['head']}`. "
        "Regenerate with the command in §5. Per-cell records are in `docs/harness/PILOT_INVENTORY.json`.", "",
        "A cell is **READY** only when every check passes, including a verified GPU smoke receipt. "
        "No GPU job has run, so no cell can be READY here.", "",
        f"**Mode:** `{report['mode']}`. " + (
            "A live probe ran in this report. READY means preconditions probed in this run, not an attestation: "
            "a determined local actor who controls the machine can forge local files and probe output."
            if report["mode"] == "live_probes"
            else "Plan-only: the live probe was skipped, so no cell can be READY in this report."), "",
        "## 1. Counts by status", "",
        "| Matrix | Cells | " + " | ".join(report["status_vocabulary"]) + " |",
        "|---|---:|" + "---:|" * len(report["status_vocabulary"]),
    ]
    for name in ("pilot", "full"):
        m = report["matrices"][name]
        row = " | ".join(str(m["counts"].get(s, 0)) for s in report["status_vocabulary"])
        lines.append(f"| {name} ({m['campaign_id']}) | {m['cell_count']} | {row} |")
    lines += ["", "Ladder state reached per cell (highest rung whose checks, and all lower ones, pass):", "",
              "| Matrix | " + " | ".join(report["ladder"]) + " |",
              "|---|" + "---:|" * len(report["ladder"])]
    for name in ("pilot", "full"):
        m = report["matrices"][name]
        row = " | ".join(str(m["ladder_counts"].get(s, 0)) for s in report["ladder"])
        lines.append(f"| {name} | {row} |")
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
              f"- Pins from the owner-signed record (only source): models {env['revisions_pinned']}; "
              f"datasets {env['dataset_revisions_pinned']}; signed pins present: {bool(env['signed_pins_used'])}",
              f"- Config revisions, checksums and vllm_pin are reported but ignored: {env['config_values_ignored']}",
              f"- GPU smoke receipts loaded: {env['receipts_loaded']}", "",
              "## 4. Assumptions and limits", "",
              f"- BF16 native support requires compute capability >= {REQUIRED_CAPABILITY} (Ampere or newer). "
              "This is an assumption about the target runtime, not a measurement.",
              f"- VRAM floor {REQUIRED_VRAM_MIB} MiB is the A100 40GB contract from `vlmbench.config.validate`.",
              f"- Disk guard {report['rules']['min_free_gib']} GiB is a conservative guard, not a measured requirement.",
              "- The weights are checked against the local manifest (`starlight_local_manifest.json`) only when that "
              "manifest's SHA-256 is pinned in the owner-signed record. A manifest on disk is not a pin.",
              "- Datasets are checked at the config's `native_path` (Drive paths) against the signed `native_sha256`. "
              "A checksum written beside a file is not an expectation.",
              "- The ladder (PLAN_AUTHORIZED < STATIC_PREREQUISITES_VALID < LIVE_ENVIRONMENT_VERIFIED < "
              "GPU_SMOKE_VERIFIED < BENCHMARK_CELL_COMPLETED) is documented in `docs/harness/READINESS.md`. "
              "This report can reach GPU_SMOKE_VERIFIED at most.",
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
    p.add_argument("--allowed-signers", type=Path, default=None,
                   help="owner signer file kept outside the repository (required for any authorization)")
    p.add_argument("--owner-record", type=Path, default=None, help="signed owner record (defaults to --authorization)")
    p.add_argument("--receipt-index", type=Path, default=DEFAULT_RECEIPT_INDEX)
    p.add_argument("--plan-only", action="store_true",
                   help="skip the live probe: the report says so and no cell can be READY")
    p.add_argument("--out-json", type=Path, default=REPO / "docs/harness/PILOT_INVENTORY.json")
    p.add_argument("--out-md", type=Path, default=REPO / "docs/harness/PILOT_INVENTORY.md")
    # ITER3 C1: there is no flag for pins. They come only from the owner-signed record (--owner-record).
    args = p.parse_args(argv)
    report = build_report(args.pilot_config, args.matrix_manifest, models_root=args.models_root,
                          authorization=args.authorization, receipts_dir=args.receipts_dir,
                          min_free_gib=args.min_free_gib, hash_weights=not args.no_hash_weights,
                          allowed_signers=args.allowed_signers,
                          owner_record=args.owner_record or args.authorization,
                          receipt_index=args.receipt_index, live_probe=not args.plan_only)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf8")
    args.out_md.write_text(render_markdown(report), encoding="utf8")
    for name in ("pilot", "full"):
        m = report["matrices"][name]
        print(f"{name}: {m['cell_count']} cells {m['counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
