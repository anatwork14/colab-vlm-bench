"""ITER3 Round B S2 (cpu_pytorch_reference_not_vllm): processor-aware mapping from image evidence regions to
Qwen3.5 visual token positions.

The geometry is the real processor's: Qwen2-VL-style smart resize with factor patch_size * merge_size = 32,
pixel bounds from the processor size dict, grid_thw = (1, H/16, W/16), and one visual token per 2x2 merged cell,
in row-major order over the merged grid. Every arithmetic claim is checked against the REAL processor output on
local files (no download). When a mapping cannot be established precisely the result is UNSUPPORTED with a reason,
never a guess.

A box maps to every merged cell it overlaps with positive area. This is a SUPERSET for protection purposes (it
protects a little more than the box, never less), and the partially covered cells are counted and reported.

Preserving a visual span does NOT prove the model used it. These tests check geometry and protection only.
"""
from pathlib import Path

import numpy as np
import pytest

from vlmbench.infrastructure import visual_mapping as vm

CKPT = Path("/mnt/ssd/genos2/research_workspace/models/Qwen/Qwen3.5-0.8B")
HAVE_CKPT = (CKPT / "preprocessor_config.json").is_file()
IMAGE_TOKEN = 248056
VISION_START = 248053
VISION_END = 248054
MIN_PIXELS = 65536  # processor size shortest_edge
MAX_PIXELS = 16777216  # processor size longest_edge


# --- resize arithmetic (hand-computed, then compared to the real processor) ----------------------

@pytest.mark.parametrize("h, w, expected", [
    (256, 256, (256, 256)),   # already in bounds, multiple of 32
    (64, 64, (256, 256)),     # below min_pixels: upscaled (ceil branch)
    (37, 1000, (64, 1344)),   # tall-thin below min: ceil branch on both edges
    (200, 300, (224, 320)),   # height 200, width 300: below min, aspect kept (probe's "(300,200)" was w,h)
    (480, 640, (480, 640)),   # in bounds, multiple of 32 once rounded
    (2000, 2000, (1984, 1984)),  # in bounds after rounding to 32 (no max branch)
])
def test_smart_resize_matches_the_documented_arithmetic(h, w, expected):
    assert vm.smart_resize(h, w, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS) == expected


def test_smart_resize_max_branch_keeps_the_pixel_budget():
    h, w = 9000, 9000  # rounds above max_pixels: the floor-division branch must bring it back under the bound
    rh, rw = vm.smart_resize(h, w, factor=32, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert rh * rw <= MAX_PIXELS
    assert rh % 32 == 0 and rw % 32 == 0


# --- geometry from a grid_thw ----------------------------------------------------------------------

def test_build_geometry_for_a_square_256_image():
    geom, reason = vm.build_geometry(256, 256, [1, 16, 16], patch_size=16, merge_size=2,
                                     min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert reason is None
    assert (geom.resized_height, geom.resized_width) == (256, 256)
    assert (geom.grid_h, geom.grid_w) == (16, 16)
    assert (geom.merged_h, geom.merged_w) == (8, 8)
    assert geom.n_tokens == 64


@pytest.mark.parametrize("grid, reason", [
    (None, vm.REASON_MISSING_GRID),
    ([1, 16], vm.REASON_MISSING_GRID),                 # malformed grid
    ([2, 16, 16], vm.REASON_TILED_OR_TEMPORAL_GRID),  # two temporal slices: not one still image
    ([1, 16, 16, 1], vm.REASON_MISSING_GRID),
    ([1, 15, 16], vm.REASON_GRID_ARITHMETIC_MISMATCH),  # 15 rows do not match 256 px / 16
    ([1, 16, 15], vm.REASON_GRID_ARITHMETIC_MISMATCH),
    ([1, 14, 14], vm.REASON_GRID_ARITHMETIC_MISMATCH),
])
def test_unmappable_grids_are_unsupported_with_a_reason(grid, reason):
    geom, got = vm.build_geometry(256, 256, grid, patch_size=16, merge_size=2,
                                  min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert geom is None
    assert got == reason


def test_missing_source_size_is_unsupported():
    geom, reason = vm.build_geometry(None, 256, [1, 16, 16], patch_size=16, merge_size=2,
                                     min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert geom is None and reason == vm.REASON_NO_SOURCE_SIZE


# --- real processor: grid, token count and patch order -------------------------------------------

@pytest.fixture(scope="module")
def processor():
    if not HAVE_CKPT:
        pytest.skip(f"official checkpoint processor not present at {CKPT}")
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(str(CKPT))


def _processor_run(processor, arr):
    from PIL import Image
    img = Image.fromarray(arr)
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "q?"}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[img], padding=True, return_tensors="pt")


@pytest.mark.parametrize("w, h", [(256, 256), (300, 200), (1000, 37), (64, 64), (640, 480), (2000, 2000)])
def test_geometry_matches_the_real_processor_grid_and_token_count(processor, w, h):
    rng = np.random.RandomState(0)
    arr = (rng.rand(h, w, 3) * 255).astype(np.uint8)
    out = _processor_run(processor, arr)
    grid = out["image_grid_thw"][0].tolist()
    n_real = int((out["input_ids"][0] == IMAGE_TOKEN).sum().item())
    geom, reason = vm.build_geometry(h, w, grid, patch_size=16, merge_size=2,
                                     min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert reason is None, reason
    assert geom.n_tokens == n_real
    assert geom.grid_h * geom.grid_w == out["pixel_values"].shape[0]


def test_token_order_is_row_major_over_merged_cells_against_real_patches(processor):
    # A gradient image where every patch is distinct. The processor output is compared patch by patch.
    h = w = 256  # identity resize: the processor output is exactly the source pixels, normalised
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    out = _processor_run(processor, arr)
    pv = out["pixel_values"].numpy().reshape(-1, 3, 2, 16, 16)[:, :, 0]  # first temporal slice of each patch
    norm = (arr.astype(np.float32) / 255.0 - 0.5) / 0.5
    merged_w = w // 32
    max_err = 0.0
    for r in range(h // 32):
        for c in range(merged_w):
            token = r * merged_w + c  # the claimed order: row-major over merged cells
            for i in range(2):
                for j in range(2):
                    patch_index = token * 4 + i * 2 + j
                    y0, x0 = r * 32 + i * 16, c * 32 + j * 16
                    expected = norm[y0:y0 + 16, x0:x0 + 16].transpose(2, 0, 1)
                    max_err = max(max_err, float(np.abs(pv[patch_index] - expected).max()))
    assert max_err < 1e-3, f"patch order disagrees with the processor: max abs error {max_err}"


# --- token runs -------------------------------------------------------------------------------

def test_one_image_run_is_found_with_its_span_and_matches_the_expected_count():
    ids = [1, 2, VISION_START] + [IMAGE_TOKEN] * 64 + [VISION_END, 9]
    span, reason = vm.find_image_span(ids, image_token_id=IMAGE_TOKEN, expected_tokens=64)
    assert reason is None
    assert span == (3, 67)


def test_two_image_runs_are_refused_as_multi_image():
    ids = [VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END, 5, VISION_START] + [IMAGE_TOKEN] * 4 + [VISION_END]
    span, reason = vm.find_image_span(ids, image_token_id=IMAGE_TOKEN, expected_tokens=4)
    assert span is None and reason == vm.REASON_MULTI_IMAGE


def test_a_run_whose_length_does_not_match_the_geometry_is_refused():
    ids = [VISION_START] + [IMAGE_TOKEN] * 60 + [VISION_END]
    span, reason = vm.find_image_span(ids, image_token_id=IMAGE_TOKEN, expected_tokens=64)
    assert span is None and reason == vm.REASON_TOKEN_RUN_MISMATCH


def test_no_image_tokens_is_reported_not_guessed():
    span, reason = vm.find_image_span([1, 2, 3], image_token_id=IMAGE_TOKEN, expected_tokens=64)
    assert span is None and reason == vm.REASON_NO_IMAGE_TOKENS


# --- boxes ------------------------------------------------------------------------------------

@pytest.fixture
def geom256():
    geom, reason = vm.build_geometry(256, 256, [1, 16, 16], patch_size=16, merge_size=2,
                                     min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    assert reason is None
    return geom


@pytest.mark.parametrize("box, reason", [
    (None, vm.REASON_NO_BOUNDING_BOX),
    ((-0.1, 0.0, 0.5, 0.5), vm.REASON_BOX_OUT_OF_RANGE),
    ((0.0, 0.0, 1.2, 0.5), vm.REASON_BOX_OUT_OF_RANGE),
    ((0.0, 0.0, 0.5, 1.0001), vm.REASON_BOX_OUT_OF_RANGE),
    ((0.5, 0.0, 0.5, 0.5), vm.REASON_BOX_MALFORMED),    # zero height
    ((0.6, 0.0, 0.5, 0.5), vm.REASON_BOX_MALFORMED),    # inverted
    ((0.0, 0.0, float("nan"), 0.5), vm.REASON_BOX_MALFORMED),
    ((0.0, 0.0, 0.5), vm.REASON_BOX_MALFORMED),         # wrong length
])
def test_invalid_boxes_are_refused_with_a_reason_and_no_cells(geom256, box, reason):
    cells, got, _info = vm.box_cells(box, geom256)
    assert cells is None and got == reason


def test_an_aligned_box_maps_to_exactly_its_merged_cells_with_no_partial_cells(geom256):
    cells, reason, info = vm.box_cells((0.0, 0.0, 0.5, 0.5), geom256)  # 128 px = 4 merged cells per side
    assert reason is None
    assert sorted(cells) == [(r, c) for r in range(4) for c in range(4)]
    assert info["partial_cells"] == 0


def test_an_unaligned_box_is_a_superset_and_reports_the_partially_covered_cells(geom256):
    cells, reason, info = vm.box_cells((0.01, 0.01, 0.5, 0.5), geom256)  # starts 2.56 px into row/col 0
    assert reason is None
    assert sorted(cells) == [(r, c) for r in range(4) for c in range(4)]  # superset: nothing dropped
    assert info["partial_cells"] == 7  # row 0 (4) + column 0 (4) - the corner counted once
    assert info["full_cells"] == 9


def test_a_box_inside_one_cell_maps_to_that_cell_only(geom256):
    cells, reason, info = vm.box_cells((0.5, 0.5, 0.52, 0.52), geom256)  # 128..133 px: inside merged cell (4,4)
    assert reason is None
    assert cells == [(4, 4)]
    assert info["partial_cells"] == 1


def test_positions_are_span_start_plus_row_major_index(geom256):
    span = (10, 10 + 64)
    positions, reason, _info = vm.region_positions((0.0, 0.0, 0.5, 0.5), geom256, span)
    assert reason is None
    assert positions == sorted(10 + r * 8 + c for r in range(4) for c in range(4))


def test_a_span_that_does_not_fit_the_geometry_is_refused(geom256):
    positions, reason, _info = vm.region_positions((0.0, 0.0, 0.5, 0.5), geom256, (10, 20))
    assert positions is None and reason == vm.REASON_TOKEN_RUN_MISMATCH
