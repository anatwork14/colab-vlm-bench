"""GPU total-memory sampling: never mislabel as physical KV cache usage."""
from __future__ import annotations

import subprocess
import threading
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
                self.samples.append({"t_monotonic": monotonic(), "used_mib": used, "util_percent": util})
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
        }
