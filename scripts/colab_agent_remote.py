"""Remotely executed by official 'colab exec -s NAME -f' from the host.

The code ZIP and job JSON are uploaded by the controller. Never execute
untrusted commands read from a job; dispatch only allowlisted actions.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import stat
import sys
import traceback
import zipfile
from pathlib import Path

ARCHIVE = Path("/content/vlmbench_agent_source.zip")
JOB_FILE = Path("/content/vlmbench_agent_job.json")
RESULT = Path("/content/vlmbench_agent_result.json")
EXPORT = Path("/content/vlmbench_agent_summary.zip")
CHECKOUT = Path("/content/vlmbench_agent_checkout")
INNER = Path("/content/vlmbench_agent_inner.json")


def _atomic_result(result: dict) -> None:
    tmp = RESULT.with_suffix(".pending")
    tmp.write_text(json.dumps(result, indent=2, default=str) + "\n")
    tmp.replace(RESULT)


def _extract_source(job: dict) -> None:
    if not ARCHIVE.is_file() or not JOB_FILE.is_file():
        raise FileNotFoundError("Agent source archive/job file not uploaded")
    expected = job["bundle_sha256"]
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != expected:
        raise ValueError("Source ZIP digest does not match host job")
    checkout = CHECKOUT / str(job["repo_sha"])
    checkout.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as z:
        for item in z.infolist():
            rel = Path(item.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("Unsafe source archive traversal")
            mode = (item.external_attr >> 16)
            if stat.S_ISLNK(mode):
                raise ValueError("Source archive symlink refused")
        z.extractall(checkout)
    return checkout


def _summary_bundle(result: dict) -> None:
    with zipfile.ZipFile(EXPORT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("agent_result.json", json.dumps(result, indent=2))
        path = result.get("runbook")
        if path and Path(path).is_file():
            z.write(path, "RUNBOOK.md")
        path = result.get("summary")
        if path and Path(path).is_file():
            z.write(path, "summary.json")


def main() -> int:
    job = json.loads(JOB_FILE.read_text(encoding="utf8"))
    try:
        checkout = _extract_source(job)
        os.environ["VLMBENCH_SOURCE_SHA"] = job["repo_sha"]
        if job["action"] in ("annotate", "benchmark") and not job.get(
                "owner_approved_compute", False):
            raise PermissionError("Remote GPU job lacks owner-approved compute flag")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(checkout / "src") + os.pathsep + env.get("PYTHONPATH", "")
        INNER.unlink(missing_ok=True)
        # Each Colab CLI invocation gets fresh Python modules, while no extra
        # shells, credentials or user-supplied code are executed.
        run = subprocess.run(
            [sys.executable, "-m", "vlmbench.agent_pipeline",
             "--job", str(JOB_FILE), "--result", str(INNER)],
            env=env, check=False,
        )
        if run.returncode or not INNER.is_file():
            raise RuntimeError(f"Remote job failed (exit={run.returncode}), see Colab exec logs")
        info = json.loads(INNER.read_text(encoding="utf8"))
        result = {
            "status": "ACTION_COMPLETED", "action": job["action"],
            "repo_sha": job["repo_sha"], "outcome": info,
        }
        # Include small runbook directly when action is report.
        if job["action"] == "report":
            result["runbook"] = info.get("runbook")
            result["summary"] = info.get("summary")
        _atomic_result(result)
        _summary_bundle(result)
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as exc:
        failure = {
            "status": "ACTION_FAILED", "action": job.get("action"),
            "repo_sha": job.get("repo_sha"),
            "error_type": type(exc).__name__, "error": str(exc)[:1000],
        }
        _atomic_result(failure)
        _summary_bundle(failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
