"""GPU total-memory sampling: never mislabel as physical KV cache usage."""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from .preflight import host_resources
from time import monotonic


class GPUSampler:
    def __init__(self, interval_s: float = 0.5):
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                p = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True,
                    text=True, timeout=4, check=True,
                )
                used, util = (int(s.strip()) for s in p.stdout.strip().split(",")[:2])
                sample = {"t_monotonic": monotonic(), "used_mib": used, "util_percent": util}
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
            except (OSError, ValueError, subprocess.SubprocessError):
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
