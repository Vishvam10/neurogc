import asyncio
import gc
import hashlib
import json
import os
import random
import tempfile
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from profiler import Profiler, ProfileMetrics


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


config = load_config()
profiler = Profiler(profile_interval=config.get("profile_interval", 1.0))
csv_path = config.get("csv_path", "profiler_data.csv")

# Track GC count to detect automatic GC events
_last_gc_count = 0


def get_gc_count() -> int:
    stats = gc.get_stats()
    return sum(s.get("collections", 0) for s in stats)


def check_gc_occurred() -> bool:
    """Check if automatic GC has occurred since last check."""
    global _last_gc_count
    current = get_gc_count()
    occurred = current > _last_gc_count
    _last_gc_count = current
    return occurred


async def profiler_save_loop():
    profile_interval = config.get("profile_interval", 1.0)
    
    while True:
        try:
            profiler.save_to_csv(csv_path)
        except Exception as e:
            print(f"[Control] Error saving to CSV: {e}")
        
        await asyncio.sleep(profile_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    profiler.start()
    
    # Start background CSV saving task
    save_task = asyncio.create_task(profiler_save_loop())

    print(f"[Control] Server starting on port {config['server_ports']['without_neurogc']}")
    print(f"[Control] Saving metrics to {csv_path}")
    print("[Control] Using Python's default garbage collection")

    yield

    save_task.cancel()
    profiler.stop()
    print("[Control] Server shutting down")


app = FastAPI(
    title="Control Server (without NeuroGC)",
    description="FastAPI server with default Python garbage collection",
    lifespan=lifespan,
)


def record_request_timing(start_time: float) -> None:
    latency_ms = (time.time() - start_time) * 1000
    profiler.record_request(latency_ms)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "server": "without_neurogc", "gc_collections": get_gc_count()}


@app.get("/metrics")
async def get_metrics():
    metrics = profiler.get_metrics()
    return {**metrics.to_dict(), "gc_collections": get_gc_count()}


@app.get("/cpu-heavy")
async def cpu_heavy():
    start_time = time.time()

    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    primes = [n for n in range(2, 10000) if is_prime(n)]

    # Also do some hashing
    data = "".join(str(p) for p in primes[:100])
    for _ in range(1000):
        data = hashlib.sha256(data.encode()).hexdigest()

    record_request_timing(start_time)

    return {
        "operation": "cpu_heavy",
        "primes_found": len(primes),
        "hash_result": data[:16],
        "duration_ms": (time.time() - start_time) * 1000,
        "gc_triggered": check_gc_occurred(),
    }


@app.get("/memory-heavy")
async def memory_heavy():
    start_time = time.time()

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

    record_request_timing(start_time)

    return {
        "operation": "memory_heavy",
        "total_elements": total_elements,
        "lists_created": len(data_list),
        "dict_keys": len(large_dict),
        "duration_ms": (time.time() - start_time) * 1000,
        "gc_triggered": check_gc_occurred(),
    }


@app.get("/network-heavy")
async def network_heavy():
    start_time = time.time()

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

    record_request_timing(start_time)

    return {
        "operation": "network_heavy",
        "external_calls": len(results),
        "results": results,
        "payload_size": len(json_str),
        "duration_ms": (time.time() - start_time) * 1000,
        "gc_triggered": check_gc_occurred(),
    }


@app.get("/io-heavy")
async def io_heavy():
    start_time = time.time()

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
            # 10KB random data
            content = os.urandom(1024 * 10)

            with open(filepath, "wb") as f:
                f.write(content)

            with open(filepath, "rb") as f:
                _ = f.read()

            files_created += 1
            total_bytes += len(content)

    record_request_timing(start_time)

    return {
        "operation": "io_heavy",
        "lines_written": len(lines),
        "total_chars": total_chars,
        "files_created": files_created,
        "total_bytes": total_bytes,
        "duration_ms": (time.time() - start_time) * 1000,
        "gc_triggered": check_gc_occurred(),
    }


if __name__ == "__main__":
    import uvicorn

    port = config["server_ports"]["without_neurogc"]
    uvicorn.run(app, host="0.0.0.0", port=port)
