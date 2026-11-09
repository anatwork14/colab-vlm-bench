"""GPU total-memory sampling: never mislabel as physical KV cache usage.

The sampler sums memory over the devices this process can use. When CUDA_VISIBLE_DEVICES
is set, only the matching devices are summed (by index or UUID). When it is unset, every
device the host reports is summed, and the report says so. Summing all host devices
includes other tenants, so the label is explicit (review F-03).
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

from .preflight import host_resources
from time import monotonic

GPU_QUERY = ["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]

# R4-A (review 1i): CUDA's default device order is FASTEST_FIRST, while nvidia-smi lists
# devices in PCI bus order. An integer in CUDA_VISIBLE_DEVICES names the same GPU in both only
# when CUDA_DEVICE_ORDER=PCI_BUS_ID. UUIDs name the device in any order.
REQUIRED_DEVICE_ORDER = "PCI_BUS_ID"


def ensure_pci_bus_order(env: dict | None = None) -> str:
    """Default CUDA_DEVICE_ORDER to PCI_BUS_ID for a child process; an explicit value is kept.

    Call before the first CUDA use in this process or before spawning children: CUDA reads the
    variable at context creation.
    """
    target = os.environ if env is None else env
    target.setdefault("CUDA_DEVICE_ORDER", REQUIRED_DEVICE_ORDER)
    return target["CUDA_DEVICE_ORDER"]


class VisibilityError(ValueError):
    """No device on the host matches CUDA_VISIBLE_DEVICES; a sample would be meaningless."""


def parse_nvidia_smi_rows(text: str) -> list[dict]:
    """Parse ``index,uuid,memory.used,utilization.gpu`` CSV, one row per device.

    Any malformed row raises ValueError: a partial parse must never be kept as a
    device-total observation.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("nvidia-smi returned no device rows")
    rows = []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 4 or not fields[0].isdigit() or not fields[1] \
                or not fields[2].isdigit() or not fields[3].isdigit():
            raise ValueError(f"Unparseable nvidia-smi row: {line!r}")
        rows.append({"index": fields[0], "uuid": fields[1],
                     "used_mib": int(fields[2]), "util_percent": int(fields[3])})
    return rows


def parse_cuda_visible_devices(value: str | None) -> list[str] | None:
    """None when unset (sum all host devices); otherwise the list of tokens (may be empty)."""
    if value is None:
        return None
    return [token.strip() for token in value.split(",") if token.strip()]


def _strip_gpu_prefix(text: str) -> str:
    return text[4:] if text.startswith("GPU-") else text


def _token_matches(token: str, row: dict) -> bool:
    if token.isdigit():
        return token == row["index"]
    return _strip_gpu_prefix(token) == _strip_gpu_prefix(row["uuid"])


def has_index_tokens(tokens: list[str] | None) -> bool:
    return bool(tokens) and any(token.isdigit() for token in tokens)


def select_visible_rows(rows: list[dict], tokens: list[str] | None) -> list[dict]:
    if tokens is None:
        return rows
    selected = [row for row in rows if any(_token_matches(t, row) for t in tokens)]
    if not selected:
        raise VisibilityError(f"no host device matches CUDA_VISIBLE_DEVICES={','.join(tokens)!r}")
    return selected


def parse_nvidia_smi_csv(text: str, visible_tokens: list[str] | None = None) -> tuple[int, int, int]:
    """Return (memory used MiB summed over selected devices, max utilization %, device count)."""
    selected = select_visible_rows(parse_nvidia_smi_rows(text), visible_tokens)
    used_total = sum(row["used_mib"] for row in selected)
    util_max = max(row["util_percent"] for row in selected)
    return used_total, util_max, len(selected)


class GPUSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        # Visibility is read once, when the sampler is created, so a run cannot change it midway.
        self.cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.visible_tokens = parse_cuda_visible_devices(self.cuda_visible_devices)
        self.cuda_device_order = os.environ.get("CUDA_DEVICE_ORDER")
        # Index tokens are trusted only when CUDA and nvidia-smi share the PCI_BUS_ID order.
        self.index_unverified = (has_index_tokens(self.visible_tokens)
                                 and self.cuda_device_order != REQUIRED_DEVICE_ORDER)
        self.ordering_unverified = 0
        self.device_indices_seen: set[str] = set()
        # Visible failure counters: a silent drop would make a broken sampler look like
        # NOT_MEASURED with no explanation.
        self.parse_failures = 0
        self.query_failures = 0
        self.visibility_mismatch = 0
        self.gpu_count_max = 0

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                p = subprocess.run(GPU_QUERY, capture_output=True,
                                   text=True, timeout=4, check=True)
            except (OSError, subprocess.SubprocessError):
                self.query_failures += 1
                self.stop_event.wait(self.interval_s)
                continue
            if self.index_unverified:
                # Refuse to claim a filtered scope: the index may name a different GPU.
                self.ordering_unverified += 1
                self.stop_event.wait(self.interval_s)
                continue
            try:
                rows = parse_nvidia_smi_rows(p.stdout)
                selected = select_visible_rows(rows, self.visible_tokens)
            except VisibilityError:
                self.visibility_mismatch += 1
                self.stop_event.wait(self.interval_s)
                continue
            except ValueError:
                self.parse_failures += 1
                self.stop_event.wait(self.interval_s)
                continue
            used = sum(row["used_mib"] for row in selected)
            util = max(row["util_percent"] for row in selected)
            gpus = len(selected)
            self.device_indices_seen.update(row["index"] for row in selected)
            self.gpu_count_max = max(self.gpu_count_max, gpus)
            sample = {"t_monotonic": monotonic(), "used_mib": used, "util_percent": util,
                      "gpu_count": gpus}
            try:
                # Host sampling is cheap but less frequent than GPU queries.
                # RAM (~80GiB) and local SSD (~200GiB) are separate from GPU VRAM.
                if len(self.samples) % 4 == 0:
                    try:
                        r = host_resources(Path("/content/vlmbench"))
                        sample["ram_available_gib"] = r["ram_available_gib"]
                        sample["ram_used_gib"] = round(
                            r["ram_total_gib"] - r["ram_available_gib"], 2)
                        sample["disk_free_gib"] = r["disk_free_gib"]
                    except (OSError, ValueError):
                        pass
                self.samples.append(sample)
            except (OSError, ValueError):
                pass
            self.stop_event.wait(self.interval_s)

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def report(self):
        visible = self.visible_tokens is not None
        if not has_index_tokens(self.visible_tokens):
            index_ordering = "NOT_APPLICABLE"
        elif self.index_unverified:
            index_ordering = "UNVERIFIED"
        else:
            index_ordering = REQUIRED_DEVICE_ORDER
        return {
            "peak_total_gpu_used_mib": max((x["used_mib"] for x in self.samples), default=None),
            "samples": len(self.samples), "sample_period_s": self.interval_s,
            "parse_failures": self.parse_failures, "query_failures": self.query_failures,
            "visibility_mismatch": self.visibility_mismatch,
            "ordering_unverified": self.ordering_unverified,
            "cuda_device_order": self.cuda_device_order,
            "index_ordering": index_ordering,
            "gpu_count_max": self.gpu_count_max,
            "device_count": self.gpu_count_max,
            "device_indices": sorted(self.device_indices_seen, key=int),
            "cuda_visible_devices": self.cuda_visible_devices,
            # Label states exactly what was summed. Unset env means every host device,
            # which can include other tenants.
            "gpu_memory_aggregation": ("sum_over_cuda_visible_devices" if visible
                                       else "sum_over_all_host_devices"),
            "physical_kv_bytes": "NOT_MEASURED", "gpu_memory_scope": "total_used_device",
            "minimum_available_system_ram_gib": min(
                (x["ram_available_gib"] for x in self.samples if "ram_available_gib" in x),
                default=None),
            "peak_system_ram_used_gib": max(
                (x["ram_used_gib"] for x in self.samples if "ram_used_gib" in x),
                default=None),
            "minimum_free_local_ssd_gib": min(
                (x["disk_free_gib"] for x in self.samples if "disk_free_gib" in x),
                default=None),
        }
