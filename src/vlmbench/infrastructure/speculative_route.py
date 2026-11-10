"""ITER3 Round B S2 (Stage 2): the speculative route as an explicit, labelled status per arm and request. Pure.

The route depends on the arm, whether the request has an image, and whether speculation is enabled, the
checkpoint is official, and an MTP module exists. EVIDENCE DOES NOT CHANGE THE ROUTE: evidence changes
protection (grounding positions), which the engine reports separately.

Statuses:
  POLICY_EVICTION_FREE_TEXT        h2o_mtp / starlight_s1_mtp, text-only: policy speculation, eviction-free verified
                                   blocks only. This is a CPU reference, not native vLLM MTP.
  MTP_VERIFIER_TEXT                mtp (serving-style), text-only: CPU verifier reference, not native vLLM MTP.
  UNSUPPORTED_IMAGE_PROMPT         combined or serving MTP arm, image prompt: NOT supported on the speculative path.
                                   The request runs the diagnostic path, which reports speculation as not executed.
  NOT_SPECULATIVE_DA_MTP           da_mtp is explicitly non-speculative (text or image).
  NOT_APPLICABLE_METHOD            the arm has no MTP speculation (base, h2o_base, starlight_s1, ...).
  SPECULATION_DISABLED             speculation is switched off by configuration (ablation).
  UNSUPPORTED_NOT_OFFICIAL_CHECKPOINT, UNSUPPORTED_NO_MTP_MODULE: the speculative path cannot run here.

cpu_pytorch_reference_not_vllm.
"""
from __future__ import annotations

ROUTE_POLICY_TEXT = "POLICY_EVICTION_FREE_TEXT"
ROUTE_MTP_TEXT = "MTP_VERIFIER_TEXT"
ROUTE_UNSUPPORTED_IMAGE = "UNSUPPORTED_IMAGE_PROMPT"
ROUTE_NOT_SPECULATIVE_DA = "NOT_SPECULATIVE_DA_MTP"
ROUTE_NOT_APPLICABLE = "NOT_APPLICABLE_METHOD"
ROUTE_DISABLED = "SPECULATION_DISABLED"
ROUTE_UNSUPPORTED_NOT_OFFICIAL = "UNSUPPORTED_NOT_OFFICIAL_CHECKPOINT"
ROUTE_UNSUPPORTED_NO_MTP = "UNSUPPORTED_NO_MTP_MODULE"

ROUTE_MTP_TEXT_NOTE = (
    "cpu_pytorch_reference_not_vllm_mtp: a CPU verifier reference, not native vLLM MTP. No speed claim."
)

_COMBINED_ARMS = ("h2o_mtp", "starlight_s1_mtp")
_SERVING_ARM = "mtp"


def speculative_route(
    method_arm: str, *, image_present: bool, speculation_enabled: bool, official: bool, mtp_present: bool,
) -> tuple[str, str | None]:
    """(status, reason). reason is None only for the two speculative text routes and NOT_APPLICABLE_METHOD."""
    if method_arm == "da_mtp":
        return ROUTE_NOT_SPECULATIVE_DA, "da_mtp is explicitly non-speculative: its DA scope is a scripted diagnostic"
    if method_arm not in _COMBINED_ARMS + (_SERVING_ARM,):
        return ROUTE_NOT_APPLICABLE, None
    if not speculation_enabled:
        return ROUTE_DISABLED, "speculation is disabled by configuration (ablation)"
    if not official:
        return ROUTE_UNSUPPORTED_NOT_OFFICIAL, "speculation needs the official hooked checkpoint"
    if not mtp_present:
        return ROUTE_UNSUPPORTED_NO_MTP, "the checkpoint bundle has no MTP module"
    if image_present:
        return (
            ROUTE_UNSUPPORTED_IMAGE,
            "image prompts are not supported on the speculative path (text-only verifier); the request runs the "
            "diagnostic path, which reports speculation as not executed",
        )
    if method_arm == _SERVING_ARM:
        return ROUTE_MTP_TEXT, None
    return ROUTE_POLICY_TEXT, None
