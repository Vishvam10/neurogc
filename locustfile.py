import csv
import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from locust import HttpUser, between, events, tag, task
from locust.runners import WorkerRunner

from neurogc.profiler import Profiler
from neurogc.utils import calculate_percentiles


def load_config(config_path: str = "config.json") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    # Default config if file doesn't exist
    return {
        "profile_interval": 1.0,
        "locust": {
            "host_with_gc": "http://localhost:8001",
            "host_without_gc": "http://localhost:8002",
            "metrics_server_url": "http://localhost:8003",
        },
    }


config = load_config()

# Determine which servers to test based on environment variable
TARGET_SERVERS = os.environ.get("TARGET_SERVERS", "both").lower()


# Collects profiling metrics and posts them to the metrics server. Runs in a
# background thread, collecting metrics every profile_interval seconds.
class ProfileCollector:
    def __init__(
        self,
        profile_interval: float = 1.0,
        metrics_server_url: str = "http://localhost:8003",
    ):
        self.profile_interval = profile_interval
        self.metrics_server_url = metrics_server_url
        self.profiler = Profiler(profile_interval=profile_interval)

        # Server URLs for fetching metrics directly from each server
        self.server_with_gc_url = config.get("locust", {}).get(
            "host_with_gc", "http://localhost:8001"
        )
        self.server_without_gc_url = config.get("locust", {}).get(
            "host_without_gc", "http://localhost:8002"
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Request tracking for RPS and latency calculations
        self._request_latencies_with_gc: deque = deque(maxlen=10000)
        self._request_latencies_without_gc: deque = deque(maxlen=10000)
        self._request_count_with_gc = 0
        self._request_count_without_gc = 0
        self._last_count_with_gc = 0
        self._last_count_without_gc = 0
        self._last_time = time.time()

        # GC tracking
        self._gc_events_with_gc: list[float] = []
        self._gc_events_without_gc: list[float] = []

        # Cached server metrics (fetched directly from servers)
        self._server_metrics_with_gc: dict = {}
        self._server_metrics_without_gc: dict = {}

        # Store all metrics for CSV export
        self._all_metrics: list[dict] = []

    def start(self) -> None:
        self._running = True
        self.profiler.start()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(
            f"[ProfileCollector] Started. Posting to {self.metrics_server_url}"
        )

    def stop(self) -> None:
        self._running = False
        self.profiler.stop()
        if self._thread:
            self._thread.join(timeout=2.0)
        print("[ProfileCollector] Stopped")

    def save_to_csv(self, filepath: str = "benchmark.csv") -> None:
        if not self._all_metrics:
            print("[ProfileCollector] No metrics to save")
            return

        fieldnames = [
            "timestamp",
            "server",
            "cpu",
            "mem",
            "disk_read",
            "disk_write",
            "net_sent",
            "net_recv",
            "rps",
            "p95",
            "p99",
            "gc_triggered",
        ]

        with self._lock:
            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for m in self._all_metrics:
                    writer.writerow(
                        {
                            "timestamp": m.get("time", 0),
                            "server": m.get("server", ""),
                            "cpu": m.get("cpu", 0),
                            "mem": m.get("mem", 0),
                            "disk_read": m.get("disk_read", 0),
                            "disk_write": m.get("disk_write", 0),
                            "net_sent": m.get("net_sent", 0),
                            "net_recv": m.get("net_recv", 0),
                            "rps": m.get("rps", 0),
                            "p95": m.get("p95", 0),
                            "p99": m.get("p99", 0),
                            "gc_triggered": m.get("gc_triggered", False),
                        }
                    )

        print(
            f"[ProfileCollector] Saved {len(self._all_metrics)} metrics to {filepath}"
        )

    def record_request(
        self, server: str, latency_ms: float, gc_triggered: bool = False
    ) -> None:
        with self._lock:
            if server == "with_gc":
                self._request_latencies_with_gc.append(latency_ms)
                self._request_count_with_gc += 1
                if gc_triggered:
                    self._gc_events_with_gc.append(time.time())
            else:
                self._request_latencies_without_gc.append(latency_ms)
                self._request_count_without_gc += 1
                if gc_triggered:
                    self._gc_events_without_gc.append(time.time())

    def _calculate_percentiles(self, latencies: deque) -> tuple[float, float]:
        return calculate_percentiles(list(latencies))

    def _fetch_server_metrics(self) -> None:
        try:
            with httpx.Client(timeout=2.0) as client:
                try:
                    resp = client.get(f"{self.server_with_gc_url}/metrics")
                    if resp.status_code == 200:
                        self._server_metrics_with_gc = resp.json()
                except Exception:
                    pass

                try:
                    resp = client.get(f"{self.server_without_gc_url}/metrics")
                    if resp.status_code == 200:
                        self._server_metrics_without_gc = resp.json()
                except Exception:
                    pass
        except Exception:
            pass

    def _get_metrics_for_server(self, server: str) -> dict:
        current_time = time.time()

        with self._lock:
            time_delta = current_time - self._last_time

            if server == "with_gc":
                latencies = self._request_latencies_with_gc
                count = self._request_count_with_gc
                last_count = self._last_count_with_gc
                gc_events = self._gc_events_with_gc
                server_metrics = self._server_metrics_with_gc
            else:
                latencies = self._request_latencies_without_gc
                count = self._request_count_without_gc
                last_count = self._last_count_without_gc
                gc_events = self._gc_events_without_gc
                server_metrics = self._server_metrics_without_gc

            rps = (count - last_count) / time_delta if time_delta > 0 else 0.0

            p95, p99 = self._calculate_percentiles(latencies)

            # Check for recent GC events
            gc_triggered = any(
                t > current_time - self.profile_interval for t in gc_events
            )

            # Use server's actual metrics for cpu, mem, disk, net
            return {
                "time": current_time,
                "cpu": server_metrics.get("cpu", 0.0),
                "mem": server_metrics.get("mem", 0.0),
                "disk_read": server_metrics.get("disk_read", 0.0),
                "disk_write": server_metrics.get("disk_write", 0.0),
                "net_sent": server_metrics.get("net_sent", 0.0),
                "net_recv": server_metrics.get("net_recv", 0.0),
                "rps": rps,
                "p95": p95,
                "p99": p99,
                "gc_triggered": server_metrics.get(
                    "gc_triggered", gc_triggered
                ),
                "server": server,
            }

    def _run_loop(self) -> None:
        while self._running:
            try:
                # Fetch actual metrics from each server
                self._fetch_server_metrics()

                metrics_with_gc = self._get_metrics_for_server("with_gc")
                metrics_without_gc = self._get_metrics_for_server("without_gc")

                # Store metrics for CSV export
                with self._lock:
                    self._all_metrics.append(metrics_with_gc.copy())
                    self._all_metrics.append(metrics_without_gc.copy())
                    self._last_count_with_gc = self._request_count_with_gc
                    self._last_count_without_gc = self._request_count_without_gc
                    self._last_time = time.time()

                try:
                    with httpx.Client(timeout=5.0) as client:
                        client.post(
                            f"{self.metrics_server_url}/api/metrics",
                            json=metrics_with_gc,
                        )
                        client.post(
                            f"{self.metrics_server_url}/api/metrics",
                            json=metrics_without_gc,
                        )
                except Exception:
                    pass

            except Exception as e:
                print(f"[ProfileCollector] Error: {e}")

            time.sleep(self.profile_interval)


profile_collector: Optional[ProfileCollector] = None


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    global profile_collector

    # Only start collector on master or standalone mode (not on workers)
    if not isinstance(environment.runner, WorkerRunner):
        profile_interval = config.get("profile_interval", 1.0)
        metrics_server_url = config.get("locust", {}).get(
            "metrics_server_url", "http://localhost:8003"
        )

        profile_collector = ProfileCollector(
            profile_interval=profile_interval,
            metrics_server_url=metrics_server_url,
        )
        profile_collector.start()


def get_benchmark_output_path() -> Path:
    model_name = os.environ.get("NEUROGC_MODEL", config.get("default_model", "unknown"))
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
    benchmark_dir = Path("benchmarks") / model_name / timestamp
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    return benchmark_dir / "benchmark.csv"


@events.quitting.add_listener
def on_locust_quit(environment, **kwargs):
    global profile_collector

    if profile_collector:
        output_path = get_benchmark_output_path()
        profile_collector.save_to_csv(str(output_path))
        print(f"[Locust] Benchmark saved to: {output_path}")
        profile_collector.stop()


# Record request metrics for profiling. This hook is called for every request
# made by Locust
@events.request.add_listener
def on_request(
    request_type,
    name,
    response_time,
    response_length,
    response,
    context,
    exception,
    **kwargs,
):
    global profile_collector

    if profile_collector and response_time is not None:
        if response and hasattr(response, "url"):
            url = str(response.url)
            if "8001" in url or "with_gc" in name:
                server = "with_gc"
            else:
                server = "without_gc"

            # Check if GC was triggered (from response if available)
            gc_triggered = False
            if response and hasattr(response, "json"):
                try:
                    data = response.json()
                    gc_triggered = data.get("gc_triggered", False)
                except Exception:
                    pass

            profile_collector.record_request(
                server, response_time, gc_triggered
            )


@tag("with_gc")
class ServerWithGCUser(HttpUser):

    host = config.get("locust", {}).get("host_with_gc", "http://localhost:8001")
    wait_time = between(0.1, 0.5)

    # Skip this user class if TARGET_SERVERS is set to "without_gc"
    weight = 0 if TARGET_SERVERS == "without_gc" else 1

    @task(3)
    def cpu_heavy(self):
        with self.client.get(
            "/cpu-heavy", catch_response=True, name="[with_gc] /cpu-heavy"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def memory_heavy(self):
        with self.client.get(
            "/memory-heavy", catch_response=True, name="[with_gc] /memory-heavy"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def network_heavy(self):
        with self.client.get(
            "/network-heavy",
            catch_response=True,
            name="[with_gc] /network-heavy",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def io_heavy(self):
        with self.client.get(
            "/io-heavy", catch_response=True, name="[with_gc] /io-heavy"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def health_check(self):
        with self.client.get(
            "/health", catch_response=True, name="[with_gc] /health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")


@tag("without_gc")
class ServerWithoutGCUser(HttpUser):

    host = config.get("locust", {}).get(
        "host_without_gc", "http://localhost:8002"
    )
    wait_time = between(0.1, 0.5)

    # Skip this user class if TARGET_SERVERS is set to "with_gc"
    weight = 0 if TARGET_SERVERS == "with_gc" else 1

    @task(3)
    def cpu_heavy(self):
        with self.client.get(
            "/cpu-heavy", catch_response=True, name="[without_gc] /cpu-heavy"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(3)
    def memory_heavy(self):
        with self.client.get(
            "/memory-heavy",
            catch_response=True,
            name="[without_gc] /memory-heavy",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def network_heavy(self):
        with self.client.get(
            "/network-heavy",
            catch_response=True,
            name="[without_gc] /network-heavy",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(2)
    def io_heavy(self):
        with self.client.get(
            "/io-heavy", catch_response=True, name="[without_gc] /io-heavy"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")

    @task(1)
    def health_check(self):
        with self.client.get(
            "/health", catch_response=True, name="[without_gc] /health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code: {response.status_code}")
