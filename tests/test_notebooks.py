from pathlib import Path
import json
import pytest

from vlmbench.config import load_config
from vlmbench.workloads import load_jsonl
from vlmbench.notebook_bridge import write_local_campaign

BASE = Path(__file__).resolve().parents[1]
NBS = BASE / "notebook"


@pytest.mark.parametrize("name,model", [
    ("Qwen3.5-0.8B.ipynb", "qwen35_08b"),
    ("Qwen3.5-4B.ipynb", "qwen35_4b"),
])
def test_notebook_json_and_python_syntax(name, model):
    obj = json.loads((NBS / name).read_text(encoding="utf8"))
    assert obj["nbformat"] == 4
    assert obj["metadata"]["vlmbench_model_id"] == model
    assert len(obj["cells"]) >= 14
    for i, cell in enumerate(obj["cells"]):
        if cell["cell_type"] == "code":
            source = cell["source"]
            if isinstance(source, list):
                source = "".join(source)
            compile(source, f"{name}:cell-{i}", "exec")
            assert not cell.get("outputs")
            assert cell.get("execution_count") is None


def test_single_model_campaign_no_deferred_revision_requirements(tmp_path):
    paths = {
        "memlens": {"native": "/content/drive/MyDrive/mem/native.jsonl",
                    "derived": "/content/drive/MyDrive/mem/serving.jsonl"},
        "mmlongbench_doc": {"native": "/content/drive/MyDrive/doc/native.jsonl",
                            "derived": "/content/drive/MyDrive/doc/serving.jsonl"},
    }
    out = tmp_path / "campaign.json"
    summary = write_local_campaign(
        BASE / "configs/a100_qwen_priority_pilot.json", out,
        model_id="qwen35_08b",
        drive_model="/content/drive/MyDrive/models/Qwen-0.8B",
        drive_results="/content/drive/MyDrive/results",
        data_paths=paths, revision="pinned_sha", campaign_suffix="test")
    assert summary["cell_count"] == 24
    assert summary["model_count"] == 1
    cfg = load_config(out, execution=True)
    assert cfg["execution_scope"]["model_ids"] == ["qwen35_08b"]
    assert cfg["dry_run"] is False


@pytest.mark.parametrize("name", ["Qwen3.5-0.8B.ipynb", "Qwen3.5-4B.ipynb"])
def test_seven_method_notebook_cells_and_hallucination_gate(name):
    notebook = json.loads((NBS / name).read_text(encoding="utf8"))
    assert len(notebook["cells"]) == 38
    combined = "\n".join(str(c.get("source", "")) for c in notebook["cells"])
    for method in ("h2o_base", "da_base", "ecrd_base", "h2o_mtp", "da_mtp"):
        assert method in combined or "METHOD_SELECTION" in combined
    assert "RUN_HALLUCINATION_SCORING" in combined
    assert "score_evidence_annotations" in combined
    assert "RUN_METHOD_MATRIX = False" in combined


def test_memlens_colab_ram_regression_and_clean_outputs():
    obj = json.loads((NBS / "Qwen3.5-0.8B.ipynb").read_text(encoding="utf8"))
    combined = "\n".join(
        "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        for c in obj["cells"] if c["cell_type"] == "code"
    )
    assert "patched_normalize_memlens" not in combined
    assert "robust_lowercase_roles" not in combined
    assert "json.load(f)" not in combined
    assert "normalize_memlens(" in combined
    assert "ijson>=3.3" in combined
    assert "pyarrow>=15" in combined
    assert 'checkout", "--detach", "FETCH_HEAD"' in combined
    assert "BENCH_SOURCE_REF" in combined
    assert "drive.mount" in combined
    for c in obj["cells"]:
        if c["cell_type"] == "code":
            assert not c.get("outputs")
            assert c.get("execution_count") is None


@pytest.mark.parametrize("name", ["Qwen3.5-0.8B.ipynb", "Qwen3.5-4B.ipynb"])
def test_full_colab_resource_and_cost_guard_before_gpu_steps(name):
    notebook = json.loads((NBS / name).read_text(encoding="utf8"))
    cells = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"]
             for c in notebook["cells"]]
    preflight = next(i for i,x in enumerate(cells) if "checkpoint_info = checkpoint_inventory" in x)
    staging = next(i for i,x in enumerate(cells) if "staged_copy(MODEL_DRIVE_DIR" in x)
    annotation = next(i for i,x in enumerate(cells) if "annotate_with_backend(" in x)
    serving = next(i for i,x in enumerate(cells) if '"run-job", "--config"' in x)
    assert preflight < staging < annotation < serving
    combined = "\n".join(cells)
    assert "CPU_PREFLIGHT_ONLY" in combined
    assert ("ANNOTATION_CALL_BUDGET = 24" if name == "Qwen3.5-0.8B.ipynb"\
            else "ANNOTATION_CALL_BUDGET = 8") in combined
    assert "serving_readiness(" in combined
    assert "ACTIVE_DATASETS" in combined
    if name == "Qwen3.5-0.8B.ipynb":
        assert "RUN_GPU_ANNOTATION = True" in combined
        assert "RUN_BASE = True" in combined
        assert "MAX_CELLS_PER_JOB = 1" in combined
        assert "ready_datasets" in combined
        assert "No verified-ready cells" in combined
    else:
        assert "RUN_GPU_ANNOTATION = False" in combined
        assert "RUN_BASE = False" in combined
    assert "RUN_MTP = False" in combined
    assert "RUN_NATIVE_QUALITY = False" in combined
