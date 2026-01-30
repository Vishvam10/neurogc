import gc
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, Sequence


def calculate_percentiles(latencies: Sequence[float]) -> tuple[float, float]:
    if not latencies:
        return 0.0, 0.0

    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)

    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)

    return sorted_latencies[p95_idx], sorted_latencies[p99_idx]


class GCTracker:
    def __init__(self):
        self._last_count = self._get_gc_count()
        self._triggered_flag = False

    @staticmethod
    def _get_gc_count() -> int:
        counts = gc.get_count()
        return sum(counts)

    def check_occurred(self) -> bool:
        current = self._get_gc_count()
        occurred = current != self._last_count
        self._last_count = current
        return occurred

    def set_triggered(self) -> None:
        self._triggered_flag = True

    def get_and_reset_triggered(self) -> bool:
        was_triggered = self._triggered_flag
        self._triggered_flag = False
        return was_triggered


class RateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last_time = 0.0

    def should_proceed(self) -> bool:
        now = time.time()
        if now - self._last_time >= self.min_interval:
            self._last_time = now
            return True
        return False

    def wait_if_needed(self) -> None:
        now = time.time()
        elapsed = now - self._last_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_time = time.time()


class MovingAverage:
    def __init__(self, window_size: int = 10):
        self._window: deque[float] = deque(maxlen=window_size)
        self._sum = 0.0

    def add(self, value: float) -> float:
        if len(self._window) == self._window.maxlen:
            self._sum -= self._window[0]
        self._window.append(value)
        self._sum += value
        return self.average

    @property
    def average(self) -> float:
        if not self._window:
            return 0.0
        return self._sum / len(self._window)

    @property
    def count(self) -> int:
        return len(self._window)


@dataclass
class TimingResult:
    elapsed_ms: float
    result: any


@contextmanager
def timed_operation() -> Generator[TimingResult, None, None]:
    result = TimingResult(elapsed_ms=0.0, result=None)
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.elapsed_ms = (time.perf_counter() - start) * 1000


def format_bytes(bytes_value: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(bytes_value) < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


INPUT_FEATURES = [
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
