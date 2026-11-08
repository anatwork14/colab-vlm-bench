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
    assert len(notebook["cells"]) == 36
    combined = "\n".join(str(c.get("source", "")) for c in notebook["cells"])
    for method in ("h2o_base", "da_base", "ecrd_base", "h2o_mtp", "da_mtp"):
        assert method in combined or "METHOD_SELECTION" in combined
    assert "RUN_HALLUCINATION_SCORING" in combined
    assert "score_evidence_annotations" in combined
    assert "RUN_METHOD_MATRIX = False" in combined
