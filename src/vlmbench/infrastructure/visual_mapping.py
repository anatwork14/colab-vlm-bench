"""Processor-aware mapping from image evidence regions to Qwen3.5 visual token positions (ITER3 Round B, S2).

cpu_pytorch_reference_not_vllm. Pure functions, no model or processor is loaded here.

Geometry (checked against the REAL local AutoProcessor in tests/test_i3b_s2_visual_mapping.py):
  * Qwen2-VL-style smart resize with factor = patch_size * merge_size = 16 * 2 = 32, pixel bounds from the
    processor size dict (min 65536, max 16777216). Mirrors transformers' smart_resize, including Python round().
  * grid_thw = (1, resized_h / patch, resized_w / patch) for a single still image.
  * One visual token per merge x merge (2x2) block of patches, laid out row-major over the merged grid.
    The token run in input_ids is therefore n_tokens = grid_h * grid_w / 4 consecutive image tokens.

Span convention: a span is (start, end) with end exclusive, over the full prompt token ids.

Region convention: a bounding box is (ymin, xmin, ymax, xmax), normalised to [0, 1] of the source image.
A box maps to every merged cell it overlaps with POSITIVE area. This is a superset of the cells fully inside the
box, so protection never drops part of the region. The count of partially covered cells is reported.

Any case that cannot be established precisely returns (None, reason) and is never guessed.

Preserving a visual span does NOT prove the model used it. This module only establishes which positions the
evidence region covers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

REASON_NO_SOURCE_SIZE = "NO_SOURCE_SIZE"
REASON_MISSING_GRID = "MISSING_GRID"
REASON_TILED_OR_TEMPORAL_GRID = "TILED_OR_TEMPORAL_GRID"
REASON_RESIZE_UNDEFINED = "RESIZE_UNDEFINED"
REASON_GRID_ARITHMETIC_MISMATCH = "GRID_ARITHMETIC_MISMATCH"
REASON_NO_IMAGE_TOKENS = "NO_IMAGE_TOKENS"
REASON_MULTI_IMAGE = "MULTI_IMAGE"
REASON_TOKEN_RUN_MISMATCH = "TOKEN_RUN_MISMATCH"
REASON_NO_BOUNDING_BOX = "NO_BOUNDING_BOX"
REASON_BOX_OUT_OF_RANGE = "BOX_OUT_OF_RANGE"
REASON_BOX_MALFORMED = "BOX_MALFORMED"

_EPS = 1e-9  # tolerance in cell units, so an exact cell boundary does not gain a neighbouring cell


@dataclass(frozen=True)
class ProcessorGeometry:
    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    grid_h: int  # patches per column of the resized image
    grid_w: int  # patches per row of the resized image
    merge_size: int
    patch_size: int

    @property
    def merged_h(self) -> int:
        return self.grid_h // self.merge_size

    @property
    def merged_w(self) -> int:
        return self.grid_w // self.merge_size

    @property
    def n_tokens(self) -> int:
        return self.merged_h * self.merged_w


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Mirror of transformers' Qwen2-VL smart_resize (same branches, same Python round())."""
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def build_geometry(
    source_height, source_width, grid_thw, *, patch_size: int, merge_size: int, min_pixels: int, max_pixels: int
) -> tuple[ProcessorGeometry | None, str | None]:
    """Geometry of one still image, verified against the processor's own grid_thw.

    Returns (geometry, None) or (None, reason). The grid must agree exactly with the resize arithmetic,
    otherwise the mapping is not established and the result is UNSUPPORTED.
    """
    if not isinstance(source_height, int) or not isinstance(source_width, int) or source_height <= 0 or source_width <= 0:
        return None, REASON_NO_SOURCE_SIZE
    if (
        not isinstance(grid_thw, (list, tuple))
        or len(grid_thw) != 3
        or not all(isinstance(v, int) and not isinstance(v, bool) for v in grid_thw)
    ):
        return None, REASON_MISSING_GRID
    grid_t, grid_h, grid_w = grid_thw
    if grid_h <= 0 or grid_w <= 0:
        return None, REASON_MISSING_GRID
    if grid_t != 1:
        return None, REASON_TILED_OR_TEMPORAL_GRID
    try:
        rh, rw = smart_resize(source_height, source_width, merge_size * patch_size, min_pixels, max_pixels)
    except ValueError:
        return None, REASON_RESIZE_UNDEFINED
    if grid_h * patch_size != rh or grid_w * patch_size != rw:
        return None, REASON_GRID_ARITHMETIC_MISMATCH
    return (
        ProcessorGeometry(
            source_height=source_height,
            source_width=source_width,
            resized_height=rh,
            resized_width=rw,
            grid_h=grid_h,
            grid_w=grid_w,
            merge_size=merge_size,
            patch_size=patch_size,
        ),
        None,
    )


def find_image_span(ids, *, image_token_id: int, expected_tokens: int) -> tuple[tuple[int, int] | None, str | None]:
    """The single run of image tokens in the prompt, as (start, end_exclusive).

    Zero runs, more than one run (multi-image), or a run whose length is not the geometry's token count are all
    refused with a reason.
    """
    runs: list[tuple[int, int]] = []
    start = None
    for i, t in enumerate(ids):
        if t == image_token_id:
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ids)))
    if not runs:
        return None, REASON_NO_IMAGE_TOKENS
    if len(runs) > 1:
        return None, REASON_MULTI_IMAGE
    s, e = runs[0]
    if e - s != expected_tokens:
        return None, REASON_TOKEN_RUN_MISMATCH
    return (s, e), None


def _box_values(box):
    if box is None:
        return None, REASON_NO_BOUNDING_BOX
    try:
        vals = [float(v) for v in box]
    except (TypeError, ValueError):
        return None, REASON_BOX_MALFORMED
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals):
        return None, REASON_BOX_MALFORMED
    ymin, xmin, ymax, xmax = vals
    if not all(0.0 <= v <= 1.0 for v in vals):
        return None, REASON_BOX_OUT_OF_RANGE
    if not (ymin < ymax and xmin < xmax):
        return None, REASON_BOX_MALFORMED
    return (ymin, xmin, ymax, xmax), None


def box_cells(box, geom: ProcessorGeometry):
    """Merged cells a normalised (ymin, xmin, ymax, xmax) box overlaps with positive area.

    Returns (cells, reason, info). cells is a sorted list of (row, col) in the merged grid, or None.
    info carries full_cells and partial_cells (cells not entirely inside the box).
    """
    vals, reason = _box_values(box)
    if reason is not None:
        return None, reason, {}
    ymin, xmin, ymax, xmax = vals
    cell = float(geom.merge_size * geom.patch_size)  # 32 px in the resized image
    y0, y1 = ymin * geom.resized_height, ymax * geom.resized_height
    x0, x1 = xmin * geom.resized_width, xmax * geom.resized_width
    r0 = max(0, math.floor(y0 / cell + _EPS))
    r1 = min(geom.merged_h - 1, math.ceil(y1 / cell - _EPS) - 1)
    c0 = max(0, math.floor(x0 / cell + _EPS))
    c1 = min(geom.merged_w - 1, math.ceil(x1 / cell - _EPS) - 1)
    cells = []
    full = 0
    for r in range(r0, r1 + 1):
        cy0, cy1 = r * cell, (r + 1) * cell
        for c in range(c0, c1 + 1):
            cx0, cx1 = c * cell, (c + 1) * cell
            cells.append((r, c))
            if cy0 >= y0 - _EPS and cy1 <= y1 + _EPS and cx0 >= x0 - _EPS and cx1 <= x1 + _EPS:
                full += 1
    info = {"full_cells": full, "partial_cells": len(cells) - full, "merged_grid": (geom.merged_h, geom.merged_w)}
    return sorted(cells), None, info


def region_positions(box, geom: ProcessorGeometry, span: tuple[int, int]):
    """Original prompt positions of the visual tokens covering the box.

    Returns (positions, reason, info). Positions are span_start + row * merged_w + col (row-major over the
    merged grid), sorted, in the coordinates of the original, uncompacted prompt.
    """
    start, end = span
    if end - start != geom.n_tokens:
        return None, REASON_TOKEN_RUN_MISMATCH, {}
    cells, reason, info = box_cells(box, geom)
    if reason is not None:
        return None, reason, info
    positions = sorted(start + r * geom.merged_w + c for r, c in cells)
    return positions, None, info
