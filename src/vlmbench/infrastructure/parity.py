"""The single logits-parity tolerance for the qualification and coverage scripts (ITER2 G-4).

Value 1e-4, strict: a max-absolute logits difference equal to the tolerance fails.
Rationale: the only measured FP32 CPU value on record is 6.2e-06 on the Test A fixture. 1e-4 leaves
about a 16x margin for BLAS and thread-order drift between runs, and is still far below any
difference that would indicate a numerically different model. It is a bound on accepted drift, not
a claim of exact equality.
"""
from __future__ import annotations

import math

LOGITS_PARITY_TOLERANCE = 1e-4


def logits_parity_passes(max_abs_diff: float, tolerance: float = LOGITS_PARITY_TOLERANCE) -> bool:
    """Strict comparison. NaN compares false, so a non-finite difference does not pass."""
    return not math.isnan(max_abs_diff) and max_abs_diff < tolerance
