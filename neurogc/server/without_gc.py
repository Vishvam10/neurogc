import argparse
import asyncio
import gc
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from neurogc.config import get_config, load_config
from neurogc.profiler import Profiler
from neurogc.server.base import create_app, register_common_endpoints
from neurogc.utils import GCTracker


_profiler: Optional[Profiler] = None
_gc_tracker: Optional[GCTracker] = None
_csv_path: Optional[str] = None


def get_gc_count() -> int:
    stats = gc.get_stats()
    return sum(s.get("collections", 0) for s in stats)


def check_gc_occurred() -> bool:
    if _gc_tracker:
        return _gc_tracker.check_occurred()
    return False


async def profiler_save_loop() -> None:
    config = get_config()
    profile_interval = config.profile_interval

    while True:
        try:
            if _profiler and _csv_path:
                _profiler.save_to_csv(_csv_path)
        except Exception as e:
            print(f"[Control] Error saving to CSV: {e}")

        await asyncio.sleep(profile_interval)


def create_control_server() -> FastAPI:
    global _profiler, _gc_tracker, _csv_path

    config = get_config()
    _csv_path = config.csv_path
    _profiler = Profiler(profile_interval=config.profile_interval)
    _gc_tracker = GCTracker()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _profiler

        _profiler.start()

        save_task = asyncio.create_task(profiler_save_loop())

        port = config.server_ports.without_neurogc
        print(f"[Control] Server starting on port {port}")
        print(f"[Control] Saving metrics to {_csv_path}")
        print("[Control] Using Python's default garbage collection")

        yield

        save_task.cancel()
        _profiler.stop()
        print("[Control] Server shutting down")

    app = create_app(
        title="Control Server (without NeuroGC)",
        description="FastAPI server with default Python garbage collection",
    )
    app.router.lifespan_context = lifespan

    register_common_endpoints(
        app=app,
        profiler=_profiler,
        server_name="without_neurogc",
        gc_flag_getter=check_gc_occurred,
        extra_health_info=lambda: {
            "gc_collections": get_gc_count(),
        },
        extra_metrics_info=lambda: {
            "gc_collections": get_gc_count(),
        },
    )

    return app


def main():
    parser = argparse.ArgumentParser(description="Control Server with default Python GC")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run server on (overrides config)",
    )

    args = parser.parse_args()

    load_config(args.config)
    config = get_config()

    app = create_control_server()

    import uvicorn

    port = args.port or config.server_ports.without_neurogc
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
