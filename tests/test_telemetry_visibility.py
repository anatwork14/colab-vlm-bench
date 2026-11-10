"""R3-A (review F-03): GPU memory must be summed over the devices this process can use,
not over every device on the host. With CUDA_VISIBLE_DEVICES unset the report says so."""
from __future__ import annotations

import subprocess

import pytest

from vlmbench import telemetry
from vlmbench.telemetry import (GPUSampler, VisibilityError, parse_cuda_visible_devices,
                                parse_nvidia_smi_csv, parse_nvidia_smi_rows)

UUID_A = "GPU-6f1e2a3b-0000-4000-8000-00000000000a"
UUID_B = "GPU-6f1e2a3b-0000-4000-8000-00000000000b"
TWO_GPUS = (f"0, {UUID_A}, 1000, 10\n"
            f"1, {UUID_B}, 5000, 90\n")


def test_rows_carry_index_uuid_memory_and_utilization():
    rows = parse_nvidia_smi_rows(TWO_GPUS)
    assert rows == [
        {"index": "0", "uuid": UUID_A, "used_mib": 1000, "util_percent": 10},
        {"index": "1", "uuid": UUID_B, "used_mib": 5000, "util_percent": 90},
    ]


def test_malformed_row_still_raises():
    with pytest.raises(ValueError):
        parse_nvidia_smi_rows("0, GPU-x, not-a-number, 3\n")


def test_unset_visibility_sums_all_host_devices_and_reports_count():
    assert parse_cuda_visible_devices(None) is None
    used, util, count = parse_nvidia_smi_csv(TWO_GPUS)
    assert (used, util, count) == (6000, 90, 2)


def test_index_filter_counts_only_the_visible_device():
    used, util, count = parse_nvidia_smi_csv(TWO_GPUS, parse_cuda_visible_devices("1"))
    assert (used, util, count) == (5000, 90, 1)


def test_uuid_filter_accepts_prefixed_and_bare_uuid():
    tokens = parse_cuda_visible_devices(f"{UUID_A}")
    assert parse_nvidia_smi_csv(TWO_GPUS, tokens) == (1000, 10, 1)
    bare = parse_cuda_visible_devices(UUID_A[len("GPU-"):])
    assert parse_nvidia_smi_csv(TWO_GPUS, bare) == (1000, 10, 1)


def test_visibility_matching_nothing_raises_visibility_error():
    with pytest.raises(VisibilityError):
        parse_nvidia_smi_csv(TWO_GPUS, parse_cuda_visible_devices("7"))


def test_empty_visibility_is_no_device_not_all_devices():
    tokens = parse_cuda_visible_devices("")
    assert tokens == []
    with pytest.raises(VisibilityError):
        parse_nvidia_smi_csv(TWO_GPUS, tokens)


def _fake_smi(monkeypatch, sampler, stdout, calls=2):
    n = {"c": 0}

    class Done:
        def __init__(self, text):
            self.stdout = text

    def fake_run(*args, **kwargs):
        n["c"] += 1
        if n["c"] > calls:
            sampler.stop_event.set()
            raise subprocess.SubprocessError("stop")
        return Done(stdout)

    monkeypatch.setattr(telemetry.subprocess, "run", fake_run)


def test_unset_env_report_uses_all_host_devices_label(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    sampler = GPUSampler(interval_s=0.001)
    _fake_smi(monkeypatch, sampler, TWO_GPUS)
    sampler._sample()
    report = sampler.report()
    assert report["gpu_memory_aggregation"] == "sum_over_all_host_devices"
    assert report["device_count"] == 2
    assert report["peak_total_gpu_used_mib"] == 6000
    assert report["cuda_visible_devices"] is None


def test_set_env_report_sums_only_visible_device(monkeypatch):
    # R4-A: an integer index is only meaningful when CUDA enumerates devices in PCI_BUS_ID order.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    sampler = GPUSampler(interval_s=0.001)
    _fake_smi(monkeypatch, sampler, TWO_GPUS)
    sampler._sample()
    report = sampler.report()
    assert report["gpu_memory_aggregation"] == "sum_over_cuda_visible_devices"
    assert report["device_count"] == 1
    assert report["device_indices"] == ["1"]
    assert report["peak_total_gpu_used_mib"] == 5000
    assert report["cuda_visible_devices"] == "1"


def test_visibility_mismatch_is_counted_and_no_sample_is_kept(monkeypatch):
    # R4-A: the device order must be verified first, or the ordering refusal would fire instead.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    sampler = GPUSampler(interval_s=0.001)
    _fake_smi(monkeypatch, sampler, TWO_GPUS)
    sampler._sample()
    report = sampler.report()
    assert report["samples"] == 0
    assert report["visibility_mismatch"] >= 1
    assert report["peak_total_gpu_used_mib"] is None


# ---------------------------------------------------------------- R4-A (review 1i): index ordering

def test_index_filter_without_pci_bus_order_refuses_the_sample(monkeypatch):
    # CUDA's default is FASTEST_FIRST, nvidia-smi uses PCI bus order: index "1" may be another GPU.
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    sampler = GPUSampler(interval_s=0.001)
    _fake_smi(monkeypatch, sampler, TWO_GPUS)
    sampler._sample()
    report = sampler.report()
    assert report["samples"] == 0
    assert report["peak_total_gpu_used_mib"] is None
    assert report["ordering_unverified"] >= 1
    assert report["index_ordering"] == "UNVERIFIED"
    assert report["cuda_device_order"] is None


def test_explicit_fastest_first_also_refuses_index_filter(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    sampler = GPUSampler(interval_s=0.001)
    _fake_smi(monkeypatch, sampler, TWO_GPUS)
    sampler._sample()
    assert sampler.report()["samples"] == 0
    assert sampler.report()["index_ordering"] == "UNVERIFIED"


def test_uuid_filter_does_not_depend_on_device_order(monkeypatch):
    # UUIDs identify the physical device in any enumeration order, so no ordering is needed.
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID_A)
    sampler = GPUSampler(interval_s=0.001)
    _fake_smi(monkeypatch, sampler, TWO_GPUS)
    sampler._sample()
    report = sampler.report()
    assert report["peak_total_gpu_used_mib"] == 1000
    assert report["ordering_unverified"] == 0
    assert report["index_ordering"] == "NOT_APPLICABLE"


def test_report_records_pci_bus_order_for_index_filter(monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    sampler = GPUSampler(interval_s=0.001)
    _fake_smi(monkeypatch, sampler, TWO_GPUS)
    sampler._sample()
    report = sampler.report()
    assert report["index_ordering"] == "PCI_BUS_ID"
    assert report["cuda_device_order"] == "PCI_BUS_ID"
    assert report["peak_total_gpu_used_mib"] == 5000


def test_ensure_pci_bus_order_sets_default_and_keeps_an_explicit_value():
    from vlmbench.telemetry import ensure_pci_bus_order

    env = {}
    assert ensure_pci_bus_order(env) == "PCI_BUS_ID"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    kept = {"CUDA_DEVICE_ORDER": "FASTEST_FIRST"}
    assert ensure_pci_bus_order(kept) == "FASTEST_FIRST"
    assert kept["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
