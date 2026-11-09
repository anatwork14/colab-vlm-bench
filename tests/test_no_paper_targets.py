"""Audit item 7: paper-reported Declarative Attention numbers must never become benchmark
targets or baselines. They may appear in documentation, which describes them as
paper-specific or retracted. Code, configs and scripts must not contain them."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER_OR_RETRACTED_NUMBERS = ("52.0", "31.1", "45.8", "19.6", "0.1895")
SCANNED = ("src", "configs", "scripts")


def test_no_paper_reported_or_retracted_numbers_in_benchmark_code():
    hits = []
    for folder in SCANNED:
        for path in (ROOT / folder).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json", ".sh", ".toml"}:
                continue
            text = path.read_text(encoding="utf8", errors="ignore")
            for number in PAPER_OR_RETRACTED_NUMBERS:
                if number in text:
                    hits.append(f"{path.relative_to(ROOT)}: {number}")
    assert hits == [], "paper-reported numbers must not be benchmark targets:\n" + "\n".join(hits)
