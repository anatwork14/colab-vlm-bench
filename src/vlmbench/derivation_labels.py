"""Derivation route labels for derived serving rows (audit F-09).

Each route that turns a native dataset into serving prompts stamps every row with one of
these labels. The labels differ on purpose: a pooled or mixed set of routes changes the
prompts, so run-job refuses to start unless a derived file carries exactly one label, and
receipts record the label so another route's receipts are never reused.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# MemLens context-bin builder (src/vlmbench/datasets/memlens.py): space-joined text,
# the last selected image, image_limit=1.
MEMLENS_7BIN_BUILDER = "memlens_7bin_builder_v2"

# Notebook bridge serving candidates (src/vlmbench/notebook_bridge.py): tokenizer-exact
# tail text, newline-joined, one source image.
SERVING_TAIL_TEXT_ONE_IMAGE = "tokenizer_exact_tail_text_one_source_image_v1"

# Generic route (src/vlmbench/derivation.py): last text words plus last k images.
LAST_WORDS_LAST_K_IMAGES = "last_text_words_plus_last_k_images_v1"

KNOWN_ROUTE_LABELS = frozenset({
    MEMLENS_7BIN_BUILDER,
    SERVING_TAIL_TEXT_ONE_IMAGE,
    LAST_WORDS_LAST_K_IMAGES,
})


def read_derivation_labels(path: str | Path) -> Counter:
    """Count the `derivation` value of every row (None when a row has no label).

    Streams the file and does not validate the rest of each row; load_jsonl does that.
    """
    counts: Counter = Counter()
    for line in Path(path).read_text(encoding="utf8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        counts[row.get("derivation") if isinstance(row, dict) else None] += 1
    return counts


def rows_label(rows: list[dict]) -> str:
    """Label of loaded rows, or UNLABELLED_OR_MIXED. run-job refuses such files before GPU use."""
    label = single_label(Counter(row.get("derivation") for row in rows))
    return label or "UNLABELLED_OR_MIXED"


def single_label(counts: Counter) -> str | None:
    """Return the one label if every row carries it, otherwise None (mixed or unlabelled)."""
    if len(counts) == 1:
        (label,) = counts
        if isinstance(label, str) and label:
            return label
    return None
