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
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


# Global state
config = load_config()
profiler = Profiler(profile_interval=config.get('profile_interval', 1.0))
gc_predictor = None
gc_trigger_count = 0
model_available = False


def init_model():
    global gc_predictor, model_available
    
    model_path = config.get('model_path', 'gc_model.pth')
    
    if os.path.exists(model_path):
        try:
            from model import load_model
            gc_predictor = load_model(model_path)
            model_available = True
            print(f"[NeuroGC] Model loaded from {model_path}")
        except Exception as e:
            print(f"[NeuroGC] Failed to load model: {e}")
            model_available = False
    else:
        print(f"[NeuroGC] Model not found at {model_path}. Running without ML-based GC.")
        model_available = False


async def gc_check_loop():
    """Background task for intelligent GC triggering."""
    global gc_trigger_count
    
    profile_interval = config.get('profile_interval', 1.0)
    gc_threshold = config.get('gc_threshold', 0.7)
    
    while True:
        try:
            metrics = profiler.get_metrics()
            metrics_dict = metrics.to_dict()
            
            if model_available and gc_predictor is not None:
                gc_predictor.add_metrics(metrics_dict)
                
                if gc_predictor.can_predict():
                    urgency = gc_predictor.predict()
                    
                    if urgency > gc_threshold:
                        gc.collect()
                        gc_trigger_count += 1
                        print(f"[NeuroGC] GC triggered (urgency: {urgency:.4f}, threshold: {gc_threshold})")

            await asyncio.sleep(profile_interval)
            
        except Exception as e:
            print(f"[NeuroGC] Error in GC check loop: {e}")
            await asyncio.sleep(profile_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    profiler.start()
    init_model()
    
    gc_task = asyncio.create_task(gc_check_loop())
    
    print(f"[NeuroGC] Server starting on port {config['server_ports']['with_neurogc']}")
    
    yield

    gc_task.cancel()
    profiler.stop()
    print("[NeuroGC] Server shutting down")


app = FastAPI(
    title="NeuroGC Server (with intelligent GC)",
    description="FastAPI server with ML-based garbage collection optimization",
    lifespan=lifespan
)


def record_request_timing(start_time: float) -> None:
    latency_ms = (time.time() - start_time) * 1000
    profiler.record_request(latency_ms)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "server": "with_neurogc",
        "model_loaded": model_available,
        "gc_trigger_count": gc_trigger_count
    }


@app.get("/metrics")
async def get_metrics():
    metrics = profiler.get_metrics()
    return {
        **metrics.to_dict(),
        "gc_trigger_count": gc_trigger_count,
        "model_available": model_available
    }


@app.get("/cpu-heavy")
async def cpu_heavy():

    start_time = time.time()
    
    def is_prime(n: int) -> bool:
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
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
        "duration_ms": (time.time() - start_time) * 1000
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
                "metadata": {"index": i, "random": random.random()}
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
        "duration_ms": (time.time() - start_time) * 1000
    }


@app.get("/network-heavy")
async def network_heavy():
    start_time = time.time()
    
    await asyncio.sleep(random.uniform(0.05, 0.15))
    
    results = []
    
    async with httpx.AsyncClient() as client:
        try:
            # Just ping a reliable endpoint
            response = await client.get(
                "https://httpbin.org/get",
                timeout=2.0
            )
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
        "duration_ms": (time.time() - start_time) * 1000
    }


@app.get("/io-heavy")
async def io_heavy():

    start_time = time.time()
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.txt') as f:
        data_to_write = "\n".join([
            f"Line {i}: {'x' * random.randint(100, 500)}"
            for i in range(1000)
        ])
        f.write(data_to_write)
        f.flush()
        
        f.seek(0)
        read_data = f.read()
        
        lines = read_data.split('\n')
        total_chars = sum(len(line) for line in lines)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        files_created = 0
        total_bytes = 0
        
        for i in range(10):
            filepath = os.path.join(tmpdir, f"file_{i}.dat")

            # 10KB random data
            content = os.urandom(1024 * 10)
            
            with open(filepath, 'wb') as f:
                f.write(content)
            
            # Read it back
            with open(filepath, 'rb') as f:
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
        "duration_ms": (time.time() - start_time) * 1000
    }


if __name__ == "__main__":
    import uvicorn
    
    port = config['server_ports']['with_neurogc']
    uvicorn.run(app, host="0.0.0.0", port=port)
