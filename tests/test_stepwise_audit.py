# DEPRECATED NOTEBOOK TESTS: superseded by official agent-operated Colab CLI.
# These tests refer to manual upload .ipynb files which may include Colab output
# cells; they are retained for provenance but are not part of the active runtime.
import pytest
pytestmark = pytest.mark.skip(reason="Legacy uploadable notebooks retired; agent CLI is active")

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model", ["Qwen3.5-0.8B.ipynb", "Qwen3.5-4B.ipynb"])
def test_stepwise_audit_covers_every_executable_cell(model):
    nb = json.loads((ROOT / "notebook" / model).read_text(encoding="utf8"))
    code_indices = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    assert code_indices == list(range(2, 37, 2))
    doc = (ROOT / "docs/COLAB_STEPWISE_AUDIT.md").read_text()
    for index in code_indices:
        assert f"| {index} |" in doc or f"| **{index}** |" in doc
    assert "CPU_PREFLIGHT_ONLY" in doc
    assert "NOT_MEASURED" in doc
