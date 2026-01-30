import csv
import gc
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from typing import Optional

import psutil

from neurogc.utils import calculate_percentiles


@dataclass
class ProfileMetrics:
    time: float = 0.0
    cpu: float = 0.0
    mem: float = 0.0
    disk_read: float = 0.0
    disk_write: float = 0.0
    net_sent: float = 0.0
    net_recv: float = 0.0
    rps: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    gc_triggered: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileMetrics":
        return cls(
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        )


class Profiler:
    def __init__(self, profile_interval: float = 1.0):
        self.profile_interval = profile_interval
        self._running = False
        self._lock = threading.Lock()

        self._latencies: deque = deque(maxlen=10000)
        self._request_count = 0
        self._last_rps_time = time.time()
        self._last_request_count = 0

        self._gc_count_before = 0
        self._gc_triggered = False

        self._disk_io_start: Optional[psutil._common.sdiskio] = None
        self._net_io_start: Optional[psutil._common.snetio] = None
        self._last_disk_io: Optional[psutil._common.sdiskio] = None
        self._last_net_io: Optional[psutil._common.snetio] = None
        self._last_snapshot_time = time.time()

        self._current_metrics = ProfileMetrics()

    def start(self) -> None:
        with self._lock:
            self._running = True
            self._gc_count_before = self._get_gc_count()
            self._disk_io_start = psutil.disk_io_counters()
            self._net_io_start = psutil.net_io_counters()
            self._last_disk_io = self._disk_io_start
            self._last_net_io = self._net_io_start
            self._last_snapshot_time = time.time()
            self._last_rps_time = time.time()

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def _get_gc_count(self) -> int:
        stats = gc.get_stats()
        return sum(s.get("collections", 0) for s in stats)

    def record_request(self, latency_ms: float) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            self._request_count += 1

    def check_gc_triggered(self) -> bool:
        current_gc_count = self._get_gc_count()
        triggered = current_gc_count > self._gc_count_before
        self._gc_count_before = current_gc_count
        return triggered

    def snapshot(self) -> ProfileMetrics:
        current_time = time.time()

        with self._lock:
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_percent = psutil.virtual_memory().percent

            disk_io = psutil.disk_io_counters()
            time_delta = current_time - self._last_snapshot_time
            if time_delta > 0 and self._last_disk_io:
                disk_read = (
                    disk_io.read_bytes - self._last_disk_io.read_bytes
                ) / time_delta
                disk_write = (
                    disk_io.write_bytes - self._last_disk_io.write_bytes
                ) / time_delta
            else:
                disk_read = 0.0
                disk_write = 0.0

            self._last_disk_io = disk_io

            net_io = psutil.net_io_counters()
            if time_delta > 0 and self._last_net_io:
                net_sent = (
                    net_io.bytes_sent - self._last_net_io.bytes_sent
                ) / time_delta
                net_recv = (
                    net_io.bytes_recv - self._last_net_io.bytes_recv
                ) / time_delta
            else:
                net_sent = 0.0
                net_recv = 0.0

            self._last_net_io = net_io

            rps_time_delta = current_time - self._last_rps_time
            if rps_time_delta >= self.profile_interval:
                rps = (
                    self._request_count - self._last_request_count
                ) / rps_time_delta
                self._last_request_count = self._request_count
                self._last_rps_time = current_time
            else:
                rps = self._current_metrics.rps

            p95, p99 = calculate_percentiles(list(self._latencies))

            gc_triggered = self.check_gc_triggered()

            self._last_snapshot_time = current_time

            self._current_metrics = ProfileMetrics(
                time=current_time,
                cpu=cpu_percent,
                mem=mem_percent,
                disk_read=disk_read,
                disk_write=disk_write,
                net_sent=net_sent,
                net_recv=net_recv,
                rps=rps,
                p95=p95,
                p99=p99,
                gc_triggered=gc_triggered,
            )

            return self._current_metrics

    def get_metrics(self) -> ProfileMetrics:
        return self.snapshot()

    def get_metrics_dict(self) -> dict:
        return self.get_metrics().to_dict()

    def clear_latencies(self) -> None:
        with self._lock:
            self._latencies.clear()

    def save_to_csv(self, filepath: str, overwrite: bool = False) -> None:
        metrics = self.get_metrics()
        metrics_dict = metrics.to_dict()

        file_exists = os.path.exists(filepath)
        file_mode = "w" if overwrite else "a"

        with open(filepath, file_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            if not file_exists or overwrite:
                writer.writeheader()
            writer.writerow(metrics_dict)

    @staticmethod
    def load_from_csv(filepath: str) -> list[ProfileMetrics]:
        metrics_list = []

        if not os.path.exists(filepath):
            return metrics_list

        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                converted = {
                    "time": float(row["time"]),
                    "cpu": float(row["cpu"]),
                    "mem": float(row["mem"]),
                    "disk_read": float(row["disk_read"]),
                    "disk_write": float(row["disk_write"]),
                    "net_sent": float(row["net_sent"]),
                    "net_recv": float(row["net_recv"]),
                    "rps": float(row["rps"]),
                    "p95": float(row["p95"]),
                    "p99": float(row["p99"]),
                    "gc_triggered": row["gc_triggered"].lower() == "true",
                }
                metrics_list.append(ProfileMetrics.from_dict(converted))

        return metrics_list


class BackgroundProfiler(Profiler):
    def __init__(
        self, profile_interval: float = 1.0, csv_path: Optional[str] = None
    ):
        super().__init__(profile_interval)
        self._csv_path = csv_path
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        super().start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        super().stop()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while self._running:
            self.snapshot()
            if self._csv_path:
                self.save_to_csv(self._csv_path)
            time.sleep(self.profile_interval)


def get_system_snapshot() -> ProfileMetrics:
    profiler = Profiler()
    profiler.start()
    time.sleep(0.1)
    metrics = profiler.snapshot()
    profiler.stop()
    return metrics


if __name__ == "__main__":
    import random

    print("Starting profiler demo...")

    profiler = BackgroundProfiler(
        profile_interval=1.0, csv_path="demo_metrics.csv"
    )
    profiler.start()

    for i in range(10):
        profiler.record_request(random.uniform(10, 100))
        time.sleep(0.5)

    metrics = profiler.get_metrics()
    print("\nCurrent Metrics:\n")
    print(f"  CPU: {metrics.cpu:.1f}%")
    print(f"  Memory: {metrics.mem:.1f}%")
    print(f"  Disk Read: {metrics.disk_read:.0f} B/s")
    print(f"  Disk Write: {metrics.disk_write:.0f} B/s")
    print(f"  Net Sent: {metrics.net_sent:.0f} B/s")
    print(f"  Net Recv: {metrics.net_recv:.0f} B/s")
    print(f"  RPS: {metrics.rps:.2f}")
    print(f"  P95: {metrics.p95:.2f} ms")
    print(f"  P99: {metrics.p99:.2f} ms")
    print(f"  GC Triggered: {metrics.gc_triggered}")

    profiler.stop()
    print("\nDemo complete. Check demo_metrics.csv for saved data\n")
