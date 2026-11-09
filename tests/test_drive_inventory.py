from pathlib import Path

from vlmbench import drive_inventory
from vlmbench import agent_pipeline


def _make_model(folder):
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    (folder / "tokenizer_config.json").write_text("{}")
    (folder / "model.safetensors").write_bytes(b"dummy-shard")


def test_existing_drive_inventory_is_read_only_and_uses_actual_two_roots(tmp_path):
    my = tmp_path / "MyDrive"
    model = my / "LLM_Weights/models/Qwen/Qwen3.5-0.8B"
    _make_model(model)
    dataset = my / "Research/Topic/Speculative Decoding/STARLIGHT"
    (dataset / "memlens/release_images/needle_images").mkdir(parents=True)
    (dataset / "memlens/release_images/haystack_images").mkdir()
    (dataset / "memlens/dataset_32k.json").write_text("[]")
    (dataset / "amber/_archives").mkdir(parents=True)
    (dataset / "amber/official_code").mkdir()
    (dataset / "mmlongbench_doc/data").mkdir(parents=True)
    (dataset / "mmlongbench_doc/documents").mkdir()
    (dataset / "mmlongbench_doc/data/qa.parquet").write_bytes(b"fake")
    prior = sorted(str(x) for x in my.rglob("*"))
    result = drive_inventory.inspect_existing_drive(my)
    after = sorted(str(x) for x in my.rglob("*"))
    assert prior == after, "Inventory must not touch original Drive sources"
    assert result["status"] == "READ_ONLY_DISCOVERY_NOT_INTEGRITY_QUALIFICATION"
    assert result["legacy_model_root"].endswith("LLM_Weights/models/Qwen")
    assert result["legacy_dataset_root"].endswith("Speculative Decoding/STARLIGHT")
    assert result["legacy_models"]["qwen35_08b"]["reuse_status"].startswith("CANDIDATE_")
    assert result["legacy_models"]["qwen35_4b"]["reuse_status"] == "NOT_PRESENT"
    assert result["legacy_datasets"]["memlens"]["reuse_status"].startswith("CANDIDATE_")
    assert result["legacy_datasets"]["amber"]["reuse_status"] == "PARTIAL_OR_UNQUALIFIED_SOURCE"
    assert result["legacy_datasets"]["mmlongbench_doc"]["source_files"]["qa_filename_candidates"] == ["qa.parquet"]
    assert result["new_default_is_separate"] is True


def test_missing_mount_reports_absence_no_side_effects(tmp_path):
    missing = tmp_path / "never-created"
    result = drive_inventory.inspect_existing_drive(missing)
    assert result["status"] == "DRIVE_NOT_MOUNTED"
    assert not missing.exists()


def test_large_images_only_shallow_listed(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    for n in range(70):
        (root / f"file{n:03d}.jpg").write_bytes(b"1")
    report = drive_inventory._list_shallow(root, max_names=5)
    assert report["entries_truncated"] is True
    assert len(report["entries_sample"]) == 5


def test_agent_dispatch_drive_inventory_no_gpu_no_hf(monkeypatch):
    observed = {"status": "READ_ONLY_DISCOVERY_NOT_INTEGRITY_QUALIFICATION"}
    monkeypatch.setattr(agent_pipeline, "inspect_existing_drive", lambda: observed)
    monkeypatch.setattr(agent_pipeline, "paths_for", lambda *a: (
        (_ for _ in ()).throw(AssertionError("No benchmark campaign path lookup"))))
    assert agent_pipeline.execute({"action": "drive-inventory"}) == observed


def test_host_supports_drive_inventory_without_paid_approval(monkeypatch):
    import importlib.util
    src = Path(__file__).resolve().parents[1] / "scripts/colab_agent.py"
    spec = importlib.util.spec_from_file_location("host_colab_agent", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    called = []
    monkeypatch.setattr(module, "remote_job", lambda args: called.append(args))
    assert module.main(["drive-inventory", "--session", "cpu-existing"]) == 0
    assert len(called) == 1 and called[0].action == "drive-inventory"
    assert called[0].approve_paid is False


def test_agent_doc_mentions_legacy_roots_and_no_automatic_reuse():
    src = (Path(__file__).resolve().parents[1] / "docs/EXISTING_DRIVE_LAYOUT.md").read_text()
    assert "LLM_Weights/models/Qwen" in src
    assert "Research/Topic/Speculative Decoding/STARLIGHT" in src
    assert "does not yet automatically reuse" in src
    assert "drive-inventory" in src
