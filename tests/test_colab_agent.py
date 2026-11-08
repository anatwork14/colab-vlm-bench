"""Agent CLI/Hugging Face source gates: CPU only, no Colab or network actions."""
from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlmbench import agent_pipeline, hf_sources


REPO = Path(__file__).resolve().parents[1]


def host_cli():
    spec = importlib.util.spec_from_file_location(
        "colab_agent_host", REPO / "scripts" / "colab_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paid_sessions_and_gpu_jobs_refuse_missing_owner_approval(monkeypatch):
    cli = host_cli()
    called = []
    monkeypatch.setattr(cli, "command", lambda args, **kw: called.append(args))
    with pytest.raises(PermissionError, match="approve-paid"):
        cli.main(["provision", "--session", "exp-a100", "--gpu", "A100"])
    with pytest.raises(PermissionError, match="approve-paid"):
        cli.main(["bootstrap", "--session", "exp-a100", "--kind", "gpu"])
    with pytest.raises(PermissionError, match="approve-paid"):
        cli.main(["annotate", "--session", "exp-a100"])
    with pytest.raises(PermissionError, match="confirm-stop"):
        cli.main(["stop", "--session", "exp-a100"])
    assert not called


def test_host_agent_dispatch_uses_explicit_named_sessions(monkeypatch):
    cli = host_cli()
    called = []
    monkeypatch.setattr(cli, "command", lambda argv, **k: called.append(argv))
    assert cli.main(["provision", "--session", "cpu-0", "--gpu", "cpu"]) == 0
    assert called[-1] == ["colab", "new", "-s", "cpu-0"]
    assert cli.main(["mount", "--session", "cpu-0"]) == 0
    assert called[-1] == ["colab", "drivemount", "-s", "cpu-0"]
    assert cli.main(["provision", "--session", "g-a100", "--gpu", "A100",
                     "--approve-paid", "--high-mem"]) == 0
    assert called[-1] == ["colab", "new", "-s", "g-a100", "--gpu", "A100", "--high-mem"]
    assert cli.main(["stop", "--session", "g-a100", "--confirm-stop"]) == 0
    assert called[-1] == ["colab", "stop", "-s", "g-a100"]


def test_host_rejects_command_injection_and_unsafe_drive_path():
    cli = host_cli()
    with pytest.raises(ValueError):
        cli.verify_session("cpu && rm -rf /")
    with pytest.raises(ValueError):
        cli.verify_drive("/content/drive/MyDrive/../../etc")
    with pytest.raises(ValueError):
        cli.verify_drive("/tmp/not-a-drive")
    assert cli.verify_drive("/content/drive/MyDrive/Experiment") == (
        "/content/drive/MyDrive/Experiment")


def test_source_sha_is_frozen_and_model_is_not_redownloaded(tmp_path, monkeypatch):
    sha = "a" * 40
    hits = []
    class Api:
        def repo_info(self, repo_id, repo_type, revision=None):
            hits.append((repo_id, repo_type, revision))
            return SimpleNamespace(sha=sha)
    def snapshot_download(repo_id, repo_type, revision, local_dir, **kw):
        root = Path(local_dir)
        (root / "config.json").write_text("{}")
        (root / "tokenizer.json").write_text("{}")
        (root / "model.safetensors").write_bytes(b"modelweights")
        return str(root)
    monkeypatch.setattr(hf_sources, "_hub", lambda: (Api(), snapshot_download))
    one = hf_sources.fetch_model("qwen35_08b", tmp_path)
    assert one["revision"] == sha
    two = hf_sources.fetch_model("qwen35_08b", tmp_path)
    assert two["revision"] == sha
    assert hits[1][2] == sha
    with pytest.raises(ValueError):
        hf_sources.fetch_model("qwen35_2b", tmp_path)


def test_memlens_archive_safe_extract_and_traversal_refusal(tmp_path):
    ok = tmp_path / "images.tar.gz"
    with tarfile.open(ok, "w:gz") as tar:
        payload = b"image bytes"
        info = tarfile.TarInfo("release_images/needle_images/safe.jpg")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    report = hf_sources.safe_extract_memlens_images(ok, tmp_path / "data")
    assert report["images_extracted"] == 1
    assert (tmp_path / "data/release_images/needle_images/safe.jpg").read_bytes() == b"image bytes"
    bad = tmp_path / "unsafe.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        item = tarfile.TarInfo("../../escape")
        item.size = 3
        tar.addfile(item, io.BytesIO(b"bad"))
    with pytest.raises(ValueError, match="Unsafe"):
        hf_sources.safe_extract_memlens_images(bad, tmp_path / "data")


def test_hf_memlens_fetches_only_32k_and_licensed_image_archive(tmp_path, monkeypatch):
    sha = "b" * 40
    requests = []
    class Api:
        def repo_info(self, repo_id, repo_type, revision=None):
            return SimpleNamespace(sha=sha)
    def download(repo_id, repo_type, revision, local_dir, allow_patterns=None, **kw):
        requests.append((repo_id, allow_patterns, revision))
        root = Path(local_dir)
        (root / "dataset_32k.json").write_text("[]")
        (root / "LICENSE-DATA").write_text("CC-BY-4.0")
        with tarfile.open(root / "release_images.tar.gz", "w:gz") as tar:
            item = tarfile.TarInfo("release_images/haystack_images/safe.jpg")
            item.size = 4
            tar.addfile(item, io.BytesIO(b"abcd"))
    monkeypatch.setattr(hf_sources, "_hub", lambda: (Api(), download))
    result = hf_sources.fetch_dataset("memlens", tmp_path)
    assert result["revision"] == sha
    assert requests[0][1] == ["dataset_32k.json", "release_images.tar.gz", "LICENSE-DATA"]
    assert (Path(result["path"]) / "release_images/haystack_images/safe.jpg").is_file()
    assert (Path(result["path"]) / ".image_unpack_complete.json").is_file()


def test_remote_actions_never_infer_gpu_approval_or_extra_models(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        agent_pipeline.paths_for({
            "drive_root": "/tmp/unmounted", "model": "qwen35_08b",
            "dataset": "memlens", "phase": "pilot"})
    with pytest.raises(ValueError):
        agent_pipeline.paths_for({
            "drive_root": "/content/drive/MyDrive/Experiment",
            "model": "gemma4_e4b", "dataset": "memlens", "phase": "pilot",
        })


def test_pipeline_prepare_resumes_same_manifest_without_extra_hf_download(tmp_path, monkeypatch):
    p = {
        "root": tmp_path, "model_root": tmp_path / "model",
        "dataset_root": tmp_path / "dataset",
        "native": tmp_path / "data/native.jsonl",
        "candidates": tmp_path / "data/candidates.jsonl",
        "annotated": tmp_path / "data/annotated.jsonl",
        "manifest": tmp_path / "data/source.json",
        "pdf_render": tmp_path / "pages", "suffix": "all",
    }
    job = {"model": "qwen35_08b", "dataset": "memlens",
           "phase": "pilot", "qa_limit": 0, "max_per_bin": 192}
    monkeypatch.setattr(agent_pipeline, "paths_for", lambda job: p)
    hf_hits = []
    def fetch(kind):
        def f(*a, **kw):
            hf_hits.append(kind)
            return {"revision": "c" * 40}
        return f
    monkeypatch.setattr(agent_pipeline, "fetch_model", fetch("model"))
    monkeypatch.setattr(agent_pipeline, "fetch_dataset", fetch("dataset"))
    p["model_root"].mkdir()
    (p["model_root"] / ".hf_source.json").write_text(
        json.dumps({"revision": "c" * 40}))
    def normalization(*args, **kw):
        p["native"].parent.mkdir(exist_ok=True)
        p["native"].write_text('{"id": "one"}\n')
        return {"count": 1}
    def derived(*args, **kw):
        p["candidates"].write_text('{"id": "one__text2048"}\n')
        return {"count": 1}
    monkeypatch.setattr(agent_pipeline, "normalize_memlens", normalization)
    monkeypatch.setattr(agent_pipeline, "build_serving_candidates", derived)
    first = agent_pipeline.prepare(job)
    assert first["status"] == "CPU_PREPARED_NEEDS_GPU_TOKEN_ANNOTATION"
    assert len(hf_hits) == 2
    second = agent_pipeline.prepare(job)
    assert second["status"] == "PREPARED_REUSED_NO_NEW_DOWNLOAD"
    assert len(hf_hits) == 2
