"""CPU-only regression coverage for runtime execution boundaries."""
import asyncio
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from vlmbench import agent_pipeline as pipeline, bench, cli


@pytest.mark.parametrize("approval", [None, False, "true", "false", 1])
def test_gpu_boundaries_require_boolean_approval(monkeypatch, approval):
    monkeypatch.setattr(pipeline, "paths_for", lambda job: pytest.fail("touched sources"))
    for action in ("annotate", "benchmark"):
        with pytest.raises(PermissionError):
            pipeline.execute({"action": action, "owner_approved_compute": approval})
        with pytest.raises(PermissionError):
            getattr(pipeline, action)({"owner_approved_compute": approval})
    with pytest.raises(PermissionError):
        cli.run_job(SimpleNamespace(owner_approved_compute=approval))


@pytest.mark.parametrize("failure", [
    SimpleNamespace(returncode=1, stdout="", stderr="driver failed"),
    subprocess.TimeoutExpired("nvidia-smi", 10), OSError("probe unavailable"),
])
def test_cpu_intake_refuses_unknown_accelerator(monkeypatch, failure):
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/nvidia-smi")
    def probe(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr(subprocess, "run", probe)
    with pytest.raises(RuntimeError, match="ACCELERATOR_STATE_UNKNOWN"):
        pipeline.require_cpu_for_hf_intake()


@pytest.mark.parametrize("terminal", [None, "done", "finish"])
def test_stream_requires_explicit_terminal(terminal):
    events = [{"choices": [{"delta": {"content": "answer"}}]},
              {"choices": [], "usage": {"completion_tokens": 256}}]
    if terminal == "finish":
        events.append({"choices": [{"delta": {}, "finish_reason": "length"}]})
    text = "".join("data: " + json.dumps(e) + "\n\n" for e in events)
    if terminal == "done":
        text += "data: [DONE]\n\n"
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=text))) as client:
            return await bench.stream_completion(client, "https://test.invalid", {}, "id")
    result = asyncio.run(run())
    assert result["status"] == ("ERROR" if terminal is None else "HTTP_OK")


@pytest.mark.parametrize("bad_result", ["failed", "bundle", "invocation", "missing", "valid"])
def test_host_rejects_failed_or_stale_result(tmp_path, monkeypatch, bad_result):
    path = Path(__file__).resolve().parents[1] / "scripts/colab_agent.py"
    spec = importlib.util.spec_from_file_location("runtime_host", path)
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    monkeypatch.setattr(host, "REPO", tmp_path)
    monkeypatch.setattr(host, "check_git_clean", lambda: "source-sha")
    def bundle(path):
        path.write_bytes(b"source")
        return "bundle-sha"
    monkeypatch.setattr(host, "make_bundle", bundle)
    job = {}
    def command(argv, **kw):
        if argv[1] == "upload" and argv[-1].endswith("job.json"):
            job.update(json.loads(Path(argv[-2]).read_text()))
        if argv[1] == "download" and argv[-1].endswith("last_result.json"):
            if bad_result == "missing":
                raise subprocess.CalledProcessError(1, argv)
            result = {"status": "ACTION_COMPLETED", "action": job["action"],
                      "repo_sha": job["repo_sha"], "bundle_sha256": job["bundle_sha256"],
                      "invocation_id": job.get("invocation_id")}
            if bad_result == "failed": result["status"] = "ACTION_FAILED"
            if bad_result == "bundle": result["bundle_sha256"] = "wrong"
            if bad_result == "invocation": result["invocation_id"] = "stale"
            Path(argv[-1]).write_text(json.dumps(result))
    monkeypatch.setattr(host, "command", command)
    if bad_result == "valid":
        assert host.main(["status", "--session", "cpu-test"]) == 0
    else:
        with pytest.raises(RuntimeError):
            host.main(["status", "--session", "cpu-test"])


def test_cpu_probe_absence_is_unknown(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="ACCELERATOR_STATE_UNKNOWN"):
        pipeline.require_cpu_for_hf_intake()


@pytest.mark.parametrize("node", ["nvidia0", "nvidia7", "nvidiactl", "nvidia-uvm",
                                  "nvidia-uvm-tools", "nvidia-modeset", "nvidia-caps",
                                  "nvhost-ctrl", "nvhost-gpu"])
def test_cpu_intake_refuses_nvidia_devices_without_probe_or_torch(monkeypatch, node):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setitem(sys.modules, "torch", None)
    original_lstat = Path.lstat
    def lstat(path, *args, **kwargs):
        if str(path) == "/proc/driver/nvidia":
            raise FileNotFoundError(str(path))
        return original_lstat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "lstat", lstat)
    original_iterdir = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda path: iter([Path("/dev") / node])
                        if str(path) == "/dev" else original_iterdir(path))
    assert str(Path("/dev") / node) in pipeline._accelerator_indicators()
    with pytest.raises(RuntimeError, match="ACCELERATOR_STATE_UNKNOWN"):
        pipeline.require_cpu_for_hf_intake()


@pytest.mark.parametrize("cuda", [None, False, True, "error", "unknown", 1])
def test_no_probe_requires_verified_cuda_state(monkeypatch, cuda):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    monkeypatch.setattr(pipeline, "_accelerator_indicators", lambda: [])
    def cuda_available():
        if cuda == "error": raise RuntimeError("driver error")
        if cuda == "unknown": return None
        return cuda
    monkeypatch.setitem(sys.modules, "torch", None if cuda is None else
                        SimpleNamespace(cuda=SimpleNamespace(is_available=cuda_available)))
    if cuda is False:
        pipeline.require_cpu_for_hf_intake()
    else:
        with pytest.raises(RuntimeError, match="ACCELERATOR_STATE_UNKNOWN|GPU_SESSION_REFUSED"):
            pipeline.require_cpu_for_hf_intake()


def test_unreadable_device_scan_is_unknown(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    def unreadable(path): raise PermissionError("device scan denied")
    monkeypatch.setattr(Path, "iterdir", unreadable)
    with pytest.raises(RuntimeError, match="ACCELERATOR_STATE_UNKNOWN"):
        pipeline.require_cpu_for_hf_intake()


@pytest.mark.parametrize("indicators", [[], ["/dev/nvidiactl"]])
def test_empty_successful_gpu_probe_must_not_contradict_devices(monkeypatch, indicators):
    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout=""))
    monkeypatch.setattr(pipeline, "_accelerator_indicators", lambda: indicators)
    if indicators:
        with pytest.raises(RuntimeError, match="ACCELERATOR_STATE_UNKNOWN"):
            pipeline.require_cpu_for_hf_intake()
    else:
        pipeline.require_cpu_for_hf_intake()


def test_direct_annotation_refuses_before_loading_or_inference():
    from vlmbench.derivation import annotate_with_backend
    with pytest.raises(PermissionError):
        annotate_with_backend(None, None, "amber", "qwen35_08b", "", "", owner_approved_compute="false")


@pytest.mark.parametrize("approval", [None, False, "true", 1, True])
def test_remote_guard_requires_exact_boolean_before_extraction(tmp_path, monkeypatch, approval):
    path = Path(__file__).resolve().parents[1] / "scripts/colab_agent_remote.py"
    spec = importlib.util.spec_from_file_location("runtime_remote", path)
    remote = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(remote)
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"action": "annotate", "owner_approved_compute": approval}))
    monkeypatch.setattr(remote, "JOB_FILE", job)
    monkeypatch.setattr(remote, "_atomic_result", lambda result: None)
    monkeypatch.setattr(remote, "_summary_bundle", lambda result: None)
    def extraction(job):
        raise RuntimeError("EXTRACTION_REACHED")
    monkeypatch.setattr(remote, "_extract_source", extraction)
    if approval is True:
        with pytest.raises(RuntimeError, match="EXTRACTION_REACHED"):
            remote.main()
    else:
        with pytest.raises(PermissionError):
            remote.main()
