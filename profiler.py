#!/usr/bin/env python3
from neurogc.profiler import (
    BackgroundProfiler,
    ProfileMetrics,
    Profiler,
    get_system_snapshot,
)

__all__ = [
    "Profiler",
    "BackgroundProfiler",
    "ProfileMetrics",
    "get_system_snapshot",
]

if __name__ == "__main__":
    import random
    import time

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
