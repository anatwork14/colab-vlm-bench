"""Explicit operator intent at every paid-compute boundary."""


def require_compute_approval(value: object) -> None:
    if value is not True:
        raise PermissionError("GPU execution requires owner_approved_compute=True (--approve-paid)")
