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
from typing import Any

from vlmbench.infrastructure.evidence_verification import EvidencePolicy, assess_evidence, decide_protection

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

    def as_dict(self) -> dict:
        return {
            "source_height": self.source_height,
            "source_width": self.source_width,
            "resized_height": self.resized_height,
            "resized_width": self.resized_width,
            "grid_h": self.grid_h,
            "grid_w": self.grid_w,
            "merged_h": self.merged_h,
            "merged_w": self.merged_w,
            "n_tokens": self.n_tokens,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
        }


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


def _cell_interval(a0: float, a1: float) -> tuple[float, float]:
    """Interval [a0, a1) in merged-cell units, with float noise snapped to a cell boundary.

    An endpoint within _EPS of an integer is snapped to it. If the snap would collapse a positive-width interval,
    the raw interval is kept, so every cell with positive overlap is still returned (ITER3 closure F-7: sub-EPS
    boxes straddling a boundary used to end with r0 > r1 and map to no cell).
    """
    s0, s1 = [float(round(a)) if abs(a - round(a)) <= _EPS else a for a in (a0, a1)]
    if s1 <= s0:
        return a0, a1
    return s0, s1


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
    ry0, ry1 = _cell_interval(y0 / cell, y1 / cell)
    rx0, rx1 = _cell_interval(x0 / cell, x1 / cell)
    r0 = max(0, math.floor(ry0))
    r1 = min(geom.merged_h - 1, math.ceil(ry1) - 1)
    c0 = max(0, math.floor(rx0))
    c1 = min(geom.merged_w - 1, math.ceil(rx1) - 1)
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


REASON_NO_IMAGE_IN_REQUEST = "NO_IMAGE_IN_REQUEST"
REASON_IMAGE_SOURCE_UNKNOWN = "IMAGE_SOURCE_UNKNOWN"
REASON_LOCATION_NOT_VERIFIED = "LOCATION_NOT_VERIFIED"

STATUS_NO_EVIDENCE = "NO_EVIDENCE_DECLARED"
STATUS_NO_IMAGE_EVIDENCE = "NO_IMAGE_EVIDENCE_DECLARED"
STATUS_SUPPORTED = "SUPPORTED"
STATUS_UNSUPPORTED = "UNSUPPORTED"

VISUAL_EVIDENCE_NOTE = (
    "SUPPORTED means the mapping from declared image_region evidence to visual token positions is established for "
    "this request: processor geometry verified against grid_thw and one image token run. Protection is still per "
    "item. A visual span is protected only when S1 protects the item (source bytes VERIFIED and bound to this "
    "request) AND its location is VERIFIED. UNSUPPORTED protects no visual token and gives a reason. Protecting a "
    "visual span does NOT prove the model used it. Visual protection is a separate channel and is not counted in "
    "the keyword caps."
)


def geometry_from_grid_list(
    source_height, source_width, grids, *, patch_size: int, merge_size: int, min_pixels: int, max_pixels: int
) -> tuple[ProcessorGeometry | None, str | None]:
    """Geometry from the processor's image_grid_thw list. Exactly one image grid is mappable."""
    if not grids:
        return None, REASON_MISSING_GRID
    if len(grids) != 1:
        return None, REASON_MULTI_IMAGE
    return build_geometry(source_height, source_width, grids[0], patch_size=patch_size,
                          merge_size=merge_size, min_pixels=min_pixels, max_pixels=max_pixels)


def processor_params(image_processor: Any) -> tuple[dict[str, int] | None, str | None]:
    """patch/merge size and pixel bounds from the processor object (SizeDict or dict). Missing -> reason."""
    size = getattr(image_processor, "size", None)

    def _field(name: str):
        if isinstance(size, dict):
            return size.get(name)
        return getattr(size, name, None)

    params = {
        "patch_size": getattr(image_processor, "patch_size", None),
        "merge_size": getattr(image_processor, "merge_size", None),
        "min_pixels": _field("shortest_edge"),
        "max_pixels": _field("longest_edge"),
    }
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in params.values()):
        return None, "PROCESSOR_GEOMETRY_UNKNOWN"
    return params, None


def visual_geometry_inputs(
    image_processor: Any, vis_trace: dict[str, Any], full_prompt, *, image_token_id: int
) -> tuple[Any, ...]:
    """(geometry, geometry_reason, span, span_reason) for one request, from the processor and the prompt tokens."""
    if not vis_trace.get("invoked"):
        return None, "NO_PROCESSOR_GEOMETRY", None, None
    params, reason = processor_params(image_processor)
    if params is None:
        return None, reason, None, None
    geom, reason = geometry_from_grid_list(
        vis_trace.get("source_height"), vis_trace.get("source_width"), vis_trace.get("image_grid_thw"),
        patch_size=params["patch_size"], merge_size=params["merge_size"],
        min_pixels=params["min_pixels"], max_pixels=params["max_pixels"],
    )
    if geom is None:
        return None, reason, None, None
    span, span_reason = find_image_span(full_prompt, image_token_id=image_token_id,
                                        expected_tokens=geom.n_tokens)
    return geom, None, span, span_reason


def visual_evidence_report(
    pool: Any,
    *,
    request_id: str,
    image_sha256: str | None,
    image_present: bool,
    policy: EvidencePolicy | None = None,
    geometry: ProcessorGeometry | None,
    geometry_reason: str | None,
    span: tuple[int, int] | None,
    span_reason: str | None,
) -> dict[str, Any]:
    """Per request: the visual evidence status, the per-item S1-gated decision, and the protected positions.

    The status is about the MAPPING: SUPPORTED when geometry and token run are established for this request,
    UNSUPPORTED (with a reason) when they are not. Positions are protected only for items that pass S1 AND map.
    A supported status with zero protected positions means every item was refused, and the items say why.
    Positions are ORIGINAL prompt positions (the coordinates before any compaction).
    """
    policy = policy or EvidencePolicy()
    rep: dict[str, Any] = {
        "status": None,
        "unsupported_reason": None,
        "items": {},
        "positions": {},
        "visual_positions_protected": 0,
        "visual_protected_positions": [],
        "geometry": geometry.as_dict() if geometry is not None else None,
        "span": list(span) if span is not None else None,
        "note": VISUAL_EVIDENCE_NOTE,
    }
    if pool is None:
        rep["status"] = STATUS_NO_EVIDENCE
        return rep
    image_items = [item for item in pool.items() if item.modality == "image_region"]
    if not image_items:
        rep["status"] = STATUS_NO_IMAGE_EVIDENCE
        return rep
    global_reason = None
    if not image_present:
        global_reason = REASON_NO_IMAGE_IN_REQUEST
    elif image_sha256 is None:
        global_reason = REASON_IMAGE_SOURCE_UNKNOWN
    elif geometry is None:
        global_reason = geometry_reason or REASON_MISSING_GRID
    elif span is None:
        global_reason = span_reason or REASON_NO_IMAGE_TOKENS
    if global_reason is not None:
        rep["status"] = STATUS_UNSUPPORTED
        rep["unsupported_reason"] = global_reason
        for item in image_items:
            rep["items"][item.evidence_id] = {"protects": False, "reason": global_reason, "dimensions": None,
                                              "positions": None, "cells": None}
        return rep

    rep["status"] = STATUS_SUPPORTED
    protected: dict[int, list[str]] = {}
    for item in image_items:
        a = assess_evidence(item, request_id=request_id, pool_request_id=pool.request_id,
                            image_sha256=image_sha256, image_present=True)
        d = decide_protection(a, policy, caller_verified=bool(item.verified))
        entry: dict[str, Any] = {"protects": False, "reason": d.mode, "dimensions": dict(a.dimensions),
                                 "positions": None, "cells": None}
        if d.protects:
            positions, reason, info = region_positions(item.bounding_box, geometry, span)
            if reason is None and a.dimensions["location_valid"] != "VERIFIED":
                reason = REASON_LOCATION_NOT_VERIFIED
            if reason is None:
                # ITER3 closure F-9: the partially covered cells (overlap < whole cell) are reported, not discarded.
                entry.update(protects=True, reason=d.mode, positions=positions,
                             cells={"full": info["full_cells"], "partial": info["partial_cells"]})
                for pos in positions:
                    protected.setdefault(pos, []).append(item.evidence_id)
            else:
                entry["reason"] = reason
        rep["items"][item.evidence_id] = entry
    rep["positions"] = {pos: tuple(sorted(ids)) for pos, ids in sorted(protected.items())}
    rep["visual_protected_positions"] = sorted(rep["positions"])
    rep["visual_positions_protected"] = len(rep["positions"])
    return rep
