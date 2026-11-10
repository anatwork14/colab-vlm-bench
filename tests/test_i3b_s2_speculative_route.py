"""ITER3 Round B S2 (Stage 2): the speculative route is an explicit, labelled status per arm and request. Pure.

  * h2o_mtp and starlight_s1_mtp: text-only requests run the policy speculation path (eviction-free verified
    blocks only). Evidence does NOT change the route; it changes protection (see the heavy engine tests).
  * image prompts on a combined method are UNSUPPORTED on the speculative path, with a reason. They run the
    diagnostic path, which reports speculation as not executed.
  * mtp (serving-style) text runs the CPU verifier reference, which is NOT native vLLM MTP.
  * da_mtp is explicitly NON-speculative, for text and image alike.
  * Non-MTP arms are NOT_APPLICABLE_METHOD.

cpu_pytorch_reference_not_vllm.
"""
import pytest

from vlmbench.infrastructure.speculative_route import (
    ROUTE_DISABLED,
    ROUTE_MTP_TEXT,
    ROUTE_MTP_TEXT_NOTE,
    ROUTE_NOT_APPLICABLE,
    ROUTE_NOT_SPECULATIVE_DA,
    ROUTE_POLICY_TEXT,
    ROUTE_UNSUPPORTED_IMAGE,
    ROUTE_UNSUPPORTED_NO_MTP,
    ROUTE_UNSUPPORTED_NOT_OFFICIAL,
    speculative_route,
)


def _route(arm, *, image_present=False, enabled=True, official=True, mtp=True):
    return speculative_route(arm, image_present=image_present, speculation_enabled=enabled, official=official,
                             mtp_present=mtp)


@pytest.mark.parametrize("arm", ["h2o_mtp", "starlight_s1_mtp"])
def test_combined_methods_run_policy_speculation_on_text_only_requests(arm):
    status, reason = _route(arm)
    assert status == ROUTE_POLICY_TEXT and reason is None


@pytest.mark.parametrize("arm", ["h2o_mtp", "starlight_s1_mtp", "mtp"])
def test_image_prompts_are_unsupported_on_the_speculative_path_with_a_reason(arm):
    status, reason = _route(arm, image_present=True)
    assert status == ROUTE_UNSUPPORTED_IMAGE
    assert isinstance(reason, str) and "image" in reason.lower()


def test_serving_mtp_text_runs_the_cpu_verifier_reference_not_native_vllm():
    status, _ = _route("mtp")
    assert status == ROUTE_MTP_TEXT
    note = ROUTE_MTP_TEXT_NOTE.lower()
    assert "cpu" in note and "not native vllm mtp" in note


@pytest.mark.parametrize("image_present", [False, True])
def test_da_mtp_is_explicitly_non_speculative(image_present):
    status, reason = _route("da_mtp", image_present=image_present)
    assert status == ROUTE_NOT_SPECULATIVE_DA
    assert status not in (ROUTE_POLICY_TEXT, ROUTE_MTP_TEXT)
    assert reason and "da" in reason.lower()


@pytest.mark.parametrize("arm", ["base", "h2o_base", "starlight_s1"])
def test_non_mtp_arms_are_not_applicable(arm):
    assert _route(arm)[0] == ROUTE_NOT_APPLICABLE


def test_disabled_speculation_is_reported_as_disabled_not_as_a_route():
    assert _route("h2o_mtp", enabled=False)[0] == ROUTE_DISABLED


def test_a_non_official_backend_is_unsupported_for_speculation():
    status, reason = _route("starlight_s1_mtp", official=False)
    assert status == ROUTE_UNSUPPORTED_NOT_OFFICIAL and reason


def test_a_missing_mtp_module_is_unsupported_for_speculation():
    status, reason = _route("h2o_mtp", mtp=False)
    assert status == ROUTE_UNSUPPORTED_NO_MTP and reason


def test_every_unsupported_status_carries_a_reason_and_no_speculative_label():
    for kwargs in (dict(image_present=True), dict(official=False), dict(mtp=False)):
        status, reason = _route("starlight_s1_mtp", **kwargs)
        assert status.startswith("UNSUPPORTED") and reason
        assert status not in (ROUTE_POLICY_TEXT, ROUTE_MTP_TEXT)
