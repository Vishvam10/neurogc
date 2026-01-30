import hashlib
import json
import os
import random
import tempfile
import time
from typing import Callable, Optional

import httpx
from fastapi import FastAPI

from neurogc.profiler import Profiler


def create_app(title: str, description: str) -> FastAPI:
    return FastAPI(title=title, description=description)


def register_common_endpoints(
    app: FastAPI,
    profiler: Profiler,
    server_name: str,
    gc_flag_getter: Optional[Callable[[], bool]] = None,
    extra_health_info: Optional[Callable[[], dict]] = None,
    extra_metrics_info: Optional[Callable[[], dict]] = None,
) -> None:
    def record_request_timing(start_time: float) -> None:
        latency_ms = (time.time() - start_time) * 1000
        profiler.record_request(latency_ms)

    def get_gc_flag() -> bool:
        if gc_flag_getter:
            return gc_flag_getter()
        return False

    @app.get("/health")
    async def health_check():
        response = {
            "status": "healthy",
            "server": server_name,
        }
        if extra_health_info:
            response.update(extra_health_info())
        return response

    @app.get("/metrics")
    async def get_metrics():
        metrics = profiler.get_metrics()
        response = metrics.to_dict()
        if extra_metrics_info:
            response.update(extra_metrics_info())
        return response

    @app.get("/cpu-heavy")
    async def cpu_heavy():
        start_time = time.time()
        result = Workloads.cpu_heavy()
        record_request_timing(start_time)

        return {
            "operation": "cpu_heavy",
            **result,
            "duration_ms": (time.time() - start_time) * 1000,
            "gc_triggered": get_gc_flag(),
        }

    @app.get("/memory-heavy")
    async def memory_heavy():
        start_time = time.time()
        result = Workloads.memory_heavy()
        record_request_timing(start_time)

        return {
            "operation": "memory_heavy",
            **result,
            "duration_ms": (time.time() - start_time) * 1000,
            "gc_triggered": get_gc_flag(),
        }

    @app.get("/network-heavy")
    async def network_heavy():
        start_time = time.time()
        result = await Workloads.network_heavy()
        record_request_timing(start_time)

        return {
            "operation": "network_heavy",
            **result,
            "duration_ms": (time.time() - start_time) * 1000,
            "gc_triggered": get_gc_flag(),
        }

    @app.get("/io-heavy")
    async def io_heavy():
        start_time = time.time()
        result = Workloads.io_heavy()
        record_request_timing(start_time)

        return {
            "operation": "io_heavy",
            **result,
            "duration_ms": (time.time() - start_time) * 1000,
            "gc_triggered": get_gc_flag(),
        }


class Workloads:
    @staticmethod
    def cpu_heavy() -> dict:
        def is_prime(n: int) -> bool:
            if n < 2:
                return False
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return False
            return True

        primes = [n for n in range(2, 10000) if is_prime(n)]

        data = "".join(str(p) for p in primes[:100])
        for _ in range(1000):
            data = hashlib.sha256(data.encode()).hexdigest()

        return {
            "primes_found": len(primes),
            "hash_result": data[:16],
        }

    @staticmethod
    def memory_heavy() -> dict:
        data_list = []

        for _ in range(100):
            large_list = [random.random() for _ in range(10000)]
            data_list.append(large_list)

        large_dict = {f"key_{i}": [random.random() for _ in range(100)] for i in range(1000)}

        nested = {
            "level1": {
                f"item_{i}": {
                    "data": [random.random() for _ in range(500)],
                    "metadata": {"index": i, "random": random.random()},
                }
                for i in range(100)
            }
        }

        total_elements = sum(len(lst) for lst in data_list) + len(large_dict) * 100

        return {
            "total_elements": total_elements,
            "lists_created": len(data_list),
            "dict_keys": len(large_dict),
        }

    @staticmethod
    async def network_heavy() -> dict:
        import asyncio

        await asyncio.sleep(random.uniform(0.05, 0.15))

        results = []

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("https://httpbin.org/get", timeout=2.0)
                results.append({"status": response.status_code, "source": "httpbin"})
            except Exception as e:
                results.append({"error": str(e), "source": "httpbin"})

        fake_payload = {"data": [random.random() for _ in range(1000)]}
        json_str = json.dumps(fake_payload)
        _ = json.loads(json_str)

        return {
            "external_calls": len(results),
            "results": results,
            "payload_size": len(json_str),
        }

    @staticmethod
    def io_heavy() -> dict:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True, suffix=".txt") as f:
            data_to_write = "\n".join(
                [f"Line {i}: {'x' * random.randint(100, 500)}" for i in range(1000)]
            )
            f.write(data_to_write)
            f.flush()

            f.seek(0)
            read_data = f.read()

            lines = read_data.split("\n")
            total_chars = sum(len(line) for line in lines)

        with tempfile.TemporaryDirectory() as tmpdir:
            files_created = 0
            total_bytes = 0

            for i in range(10):
                filepath = os.path.join(tmpdir, f"file_{i}.dat")
                content = os.urandom(1024 * 10)

                with open(filepath, "wb") as f:
                    f.write(content)

                with open(filepath, "rb") as f:
                    _ = f.read()

                files_created += 1
                total_bytes += len(content)

        return {
            "lines_written": len(lines),
            "total_chars": total_chars,
            "files_created": files_created,
            "total_bytes": total_bytes,
        }
