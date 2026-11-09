"""GPU total-memory sampling: never mislabel as physical KV cache usage."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from .preflight import host_resources
from time import monotonic


def parse_nvidia_smi_csv(text: str) -> tuple[int, int, int]:
    """Parse ``memory.used,utilization.gpu`` CSV (one line per device).

    Returns (total used MiB across all devices, max utilization %, device count).
    Any malformed row raises ValueError: a partial parse must never be kept as a
    device-total observation.
    """
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        raise ValueError("nvidia-smi returned no device rows")
    used_total, util_max = 0, 0
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 2 or not all(f.isdigit() for f in fields):
            raise ValueError(f"Unparseable nvidia-smi row: {line!r}")
        used_total += int(fields[0])
        util_max = max(util_max, int(fields[1]))
    return used_total, util_max, len(lines)


class GPUSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        # Visible failure counters: a silent drop would make a broken sampler look like
        # NOT_MEASURED with no explanation.
        self.parse_failures = 0
        self.query_failures = 0
        self.gpu_count_max = 0

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                p = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, timeout=4, check=True,
                )
            except (OSError, subprocess.SubprocessError):
                self.query_failures += 1
                self.stop_event.wait(self.interval_s)
                continue
            try:
                used, util, gpus = parse_nvidia_smi_csv(p.stdout)
            except ValueError:
                self.parse_failures += 1
                self.stop_event.wait(self.interval_s)
                continue
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
        return {
            "peak_total_gpu_used_mib": max((x["used_mib"] for x in self.samples), default=None),
            "samples": len(self.samples), "sample_period_s": self.interval_s,
            "parse_failures": self.parse_failures, "query_failures": self.query_failures,
            "gpu_count_max": self.gpu_count_max,
            "gpu_memory_aggregation": "sum_over_visible_devices",
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
