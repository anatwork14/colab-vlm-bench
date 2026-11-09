#!/usr/bin/env python3
"""Agent-operated Google Colab CLI controller (host Linux/macOS only).

Only explicit commands create/terminate sessions or spend GPU. Never put tokens
in repository files. This controller uses official google-colab-cli subprocesses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SESSION_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")
DRIVE_DEFAULT = "/content/drive/MyDrive/Colab-VLM-Bench-Agent"
REQUIRE_PAID = {"annotate", "benchmark"}
SESSION_STATE = Path.home() / ".config/colab-cli/sessions.json"


def resolve_session(identifier: str) -> str:
    """Resolve an alias or registered endpoint without mutating CLI state.

    The CLI lists server assignments even when they have no local registration,
    but its status/upload/exec commands require a local alias. Never turn a
    missing registration into an implicit allocation or print proxy credentials.
    """
    verify_session(identifier)
    try:
        sessions = json.loads(SESSION_STATE.read_text(encoding="utf8"))
    except FileNotFoundError:
        sessions = {}
    except (OSError, ValueError):
        raise RuntimeError("Cannot read Colab session registry; inspect it locally") from None
    if not isinstance(sessions, dict) or any(not isinstance(v, dict) for v in sessions.values()):
        raise RuntimeError("Invalid Colab session registry; inspect it locally")
    if identifier in sessions:
        return identifier
    aliases = [name for name, data in sessions.items() if data.get("endpoint") == identifier]
    if len(aliases) > 1:
        raise RuntimeError("Colab endpoint registration is ambiguous; specify its alias")
    if aliases:
        return verify_session(aliases[0])
    raise RuntimeError(
        f"Colab session {identifier!r} is not registered locally. "
        "Use an existing local alias created by colab new. A [?] assignment "
        "requires authorized local registration; no replacement runtime was allocated.")


def command(argv: list[str], *, dry_run: bool = False) -> None:
    """No shell=True; arguments are literal, never agent-rendered shell code."""
    argv = list(argv)
    if len(argv) > 1 and argv[0] == "colab" and argv[1] != "new" and "-s" in argv:
        index = argv.index("-s") + 1
        argv[index] = resolve_session(argv[index])
    print("+", " ".join(argv), flush=True)
    if not dry_run:
        subprocess.run(argv, cwd=REPO, check=True)


def verify_session(name: str) -> str:
    if not SESSION_NAME.fullmatch(name):
        raise ValueError("Use simple explicit session name: letters/digits/._-")
    return name


def verify_drive(path: str) -> str:
    if (not path.startswith("/content/drive/MyDrive/")
            or ".." in Path(path).parts or path.endswith("/")):
        raise ValueError("Drive root must be a persistent MyDrive subdirectory")
    return path


def check_git_clean() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all",
         "--ignore-submodules=none"], cwd=REPO, text=True)
    # Editable installs generate metadata absent from git archive. Exempt only
    # known untracked metadata files, never tracked edits or arbitrary code.
    metadata = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "entry_points.txt",
                "requires.txt", "top_level.txt", "not-zip-safe", "zip-safe"}
    dirty = []
    for entry in status.split("\0"):
        if not entry:
            continue
        path = Path(entry[3:])
        if (entry.startswith("?? ") and path.parent.name.endswith(".egg-info")
                and path.name in metadata):
            continue
        dirty.append(entry)
    if dirty:
        raise RuntimeError("Commit/stash worktree before remote source transfer: "
                           + repr(dirty))
    return sha


def make_bundle(dst: Path) -> str:
    with dst.open("wb") as fp:
        subprocess.run(["git", "archive", "--format=zip", "HEAD"],
                       cwd=REPO, stdout=fp, check=True)
    digest = hashlib.sha256(dst.read_bytes()).hexdigest()
    return digest


def remote_job(args: argparse.Namespace) -> None:
    session = verify_session(args.session)
    drive = verify_drive(args.drive_root)
    # Reject paid actions BEFORE touching git, uploading data, or contacting Colab.
    if args.action in REQUIRE_PAID and not args.approve_paid:
        raise PermissionError(
            "GPU inference requires explicit --approve-paid in each invocation")
    sha = check_git_clean()
    if args.action == "prepare" and args.qa_limit and args.phase == "full":
        raise ValueError("Full confirmatory source cannot have QA subset limit")
    if args.action == "annotate" and not (1 <= args.max_requests <= 128):
        raise ValueError("Annotation per-call limit must be 1..128")
    if args.action == "benchmark" and not (1 <= args.max_cells <= 6):
        raise ValueError("Serving per-call cell limit must be 1..6")
    with tempfile.TemporaryDirectory(prefix="vlmbench-colab-agent-") as scratch:
        directory = Path(scratch)
        archive = directory / "source.zip"
        bundle_sha = make_bundle(archive)
        job = {
            "invocation_id": uuid.uuid4().hex,
            "schema_version": 1,
            "action": args.action, "session": session, "drive_root": drive,
            "model": args.model, "dataset": args.dataset, "phase": args.phase,
            "qa_limit": args.qa_limit, "max_per_bin": args.max_per_bin,
            "max_requests": args.max_requests, "max_cells": args.max_cells,
            "mode": args.mode, "quality": args.quality,
            "model_revision": args.model_revision,
            "dataset_revision": args.dataset_revision,
            "source_mode": args.source_mode, "source_map": args.source_map,
            "repo_sha": sha, "bundle_sha256": bundle_sha,
            "owner_approved_compute": bool(args.approve_paid),
        }
        path = directory / "job.json"
        path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf8")
        command(["colab", "status", "-s", session], dry_run=args.dry_run)
        command(["colab", "upload", "-s", session, str(archive),
                 "/content/vlmbench_agent_source.zip"], dry_run=args.dry_run)
        command(["colab", "upload", "-s", session, str(path),
                 "/content/vlmbench_agent_job.json"], dry_run=args.dry_run)
        try:
            command(["colab", "exec", "-s", session, "-f",
                     "scripts/colab_agent_remote.py"], dry_run=args.dry_run)
        finally:
            target = REPO / "agent_artifacts" / session / job["invocation_id"]
            if not args.dry_run:
                target.mkdir(parents=True, exist_ok=True)
                # Download status EVEN when remote execution failed; no raw data.
                for remote, name in (
                    ("/content/vlmbench_agent_result.json", "last_result.json"),
                    ("/content/vlmbench_agent_summary.zip", "last_summary.zip"),
                ):
                    try:
                        command(["colab", "download", "-s", session, remote,
                                 str(target / name)])
                    except subprocess.CalledProcessError:
                        print(f"Remote artifact unavailable: {remote}", file=sys.stderr)
        if not args.dry_run:
            try:
                result = json.loads((target / "last_result.json").read_text(encoding="utf8"))
            except (OSError, ValueError):
                raise RuntimeError("Missing or invalid remote result; inspect invocation artifacts") from None
            if not isinstance(result, dict) or result.get("status") != "ACTION_COMPLETED":
                raise RuntimeError("Remote action failed; inspect invocation artifacts")
            for key in ("invocation_id", "action", "repo_sha", "bundle_sha256"):
                if result.get(key) != job[key]:
                    raise RuntimeError(f"Remote result provenance mismatch: {key}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Agent controlled Colab CLI workflows, not notebooks")
    sub = p.add_subparsers(dest="action", required=True)

    sub.add_parser("doctor", help="Read-only Colab CLI, sessions and compute units")
    provision = sub.add_parser("provision", help="Create named CPU or approved A100 session")
    provision.add_argument("--session", required=True)
    provision.add_argument("--gpu", choices=["cpu", "A100"], default="cpu")
    provision.add_argument("--high-mem", action="store_true")
    provision.add_argument("--approve-paid", action="store_true")
    provision.add_argument("--dry-run", action="store_true")

    mount = sub.add_parser("mount", help="Mount Drive (interactive login may be required)")
    mount.add_argument("--session", required=True)
    mount.add_argument("--dry-run", action="store_true")

    bootstrap = sub.add_parser("bootstrap", help="Install CPU or A100 requirements")
    bootstrap.add_argument("--session", required=True)
    bootstrap.add_argument("--kind", choices=["cpu", "gpu"], required=True)
    bootstrap.add_argument("--approve-paid", action="store_true")
    bootstrap.add_argument("--dry-run", action="store_true")

    for name in ("drive-inventory", "qualify-drive", "prepare", "status", "annotate", "benchmark", "report"):
        sp = sub.add_parser(name, help="Execute one auditable remote job")
        sp.add_argument("--session", required=True)
        sp.add_argument("--drive-root", default=DRIVE_DEFAULT)
        sp.add_argument("--model", choices=["qwen35_08b", "qwen35_4b"],
                        default="qwen35_08b")
        sp.add_argument("--dataset", choices=["memlens", "mmlongbench_doc", "amber"],
                        default="memlens")
        sp.add_argument("--phase", choices=["pilot", "full"], default="pilot")
        sp.add_argument("--qa-limit", type=int, default=0)
        sp.add_argument("--max-per-bin", type=int, default=192)
        sp.add_argument("--model-revision")
        sp.add_argument("--dataset-revision")
        sp.add_argument("--source-mode", choices=["historical", "hf"], default="historical")
        sp.add_argument("--source-map", help="Reviewed Drive JSON with pinned source SHA256 inventory")
        sp.add_argument("--max-requests", type=int, default=8)
        sp.add_argument("--max-cells", type=int, default=1)
        sp.add_argument("--mode", choices=["base", "mtp"], default="base")
        sp.add_argument("--quality", action="store_true")
        sp.add_argument("--approve-paid", action="store_true")
        sp.add_argument("--dry-run", action="store_true")

    pull = sub.add_parser("pull", help="Download last small status + runbook bundle")
    pull.add_argument("--session", required=True)
    pull.add_argument("--output", default="agent_artifacts/latest_summary.zip")

    stop = sub.add_parser("stop", help="Explicitly release named Colab session")
    stop.add_argument("--session", required=True)
    stop.add_argument("--confirm-stop", action="store_true")
    stop.add_argument("--dry-run", action="store_true")

    args = p.parse_args(argv)
    if args.action == "doctor":
        for cmd in (["colab", "version"], ["colab", "sessions"],
                    ["colab", "usage"]):
            command(cmd)
    elif args.action == "provision":
        name = verify_session(args.session)
        cmd = ["colab", "new", "-s", name]
        if args.gpu == "A100":
            if not args.approve_paid:
                raise PermissionError("Require --approve-paid to allocate A100")
            cmd += ["--gpu", "A100"]
        if args.high_mem:
            cmd += ["--high-mem"]
        command(cmd, dry_run=args.dry_run)
    elif args.action == "mount":
        command(["colab", "drivemount", "-s", verify_session(args.session)],
                dry_run=args.dry_run)
    elif args.action == "bootstrap":
        if args.kind == "gpu" and not args.approve_paid:
            raise PermissionError("Require --approve-paid for GPU runtime installs")
        command(["colab", "install", "-s", verify_session(args.session),
                 "-r", "requirements-agent-" + args.kind + ".txt"],
                dry_run=args.dry_run)
    elif args.action in {"drive-inventory", "qualify-drive", "prepare", "status", "annotate", "benchmark", "report"}:
        remote_job(args)
    elif args.action == "pull":
        command(["colab", "download", "-s", verify_session(args.session),
                 "/content/vlmbench_agent_summary.zip", args.output])
    elif args.action == "stop":
        if not args.confirm_stop:
            raise PermissionError("Explicit --confirm-stop required")
        command(["colab", "stop", "-s", verify_session(args.session)],
                dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
