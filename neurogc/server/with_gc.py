import argparse
import asyncio
import gc
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from neurogc.config import get_config, load_config
from neurogc.models import get_model
from neurogc.models.base import BaseGCPredictor
from neurogc.profiler import Profiler
from neurogc.server.base import create_app, register_common_endpoints

_profiler: Optional[Profiler] = None
_predictor: Optional[BaseGCPredictor] = None
_gc_trigger_count = 0
_model_available = False
_last_gc_triggered = False
_csv_path: Optional[str] = None


def get_and_reset_gc_flag() -> bool:
    global _last_gc_triggered
    flag = _last_gc_triggered
    _last_gc_triggered = False
    return flag


def init_model(
    model_name: str = "lstm", model_path: Optional[str] = None
) -> None:
    global _predictor, _model_available

    config = get_config()
    path = model_path or config.model_path

    if not os.path.exists(path):
        print(
            f"[NeuroGC] Model not found at {path}. Running without ML-based GC."
        )
        _model_available = False
        return

    try:
        model_class = get_model(model_name)
        _predictor = model_class()
        _predictor.load(Path(path))
        _model_available = True
        print(f"[NeuroGC] Loaded {model_name} model from {path}")
    except Exception as e:
        print(f"[NeuroGC] Failed to load model: {e}")
        _model_available = False


async def gc_check_loop() -> None:
    global _gc_trigger_count, _last_gc_triggered

    config = get_config()
    profile_interval = config.profile_interval
    gc_threshold = config.gc_threshold

    while True:
        try:
            if _profiler is None:
                await asyncio.sleep(profile_interval)
                continue

            metrics = _profiler.get_metrics()
            metrics_dict = metrics.to_dict()

            if _csv_path:
                _profiler.save_to_csv(_csv_path)

            if _model_available and _predictor is not None:
                _predictor.add_metrics(metrics_dict)

                if _predictor.can_predict():
                    urgency = _predictor.predict()

                    if urgency > gc_threshold:
                        gc.collect()
                        _gc_trigger_count += 1
                        _last_gc_triggered = True
                        print(
                            f"[NeuroGC] GC triggered (urgency: {urgency:.4f}, threshold: {gc_threshold})"
                        )

            await asyncio.sleep(profile_interval)

        except Exception as e:
            print(f"[NeuroGC] Error in GC check loop: {e}")
            await asyncio.sleep(profile_interval)


def create_neurogc_server(
    model_name: str = "lstm",
    model_path: Optional[str] = None,
) -> FastAPI:
    global _profiler, _csv_path

    config = get_config()
    _csv_path = config.csv_path.replace(".csv", "_with_gc.csv")
    _profiler = Profiler(profile_interval=config.profile_interval)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _profiler

        _profiler.start()
        init_model(model_name, model_path)

        gc_task = asyncio.create_task(gc_check_loop())

        port = config.server_ports.with_neurogc
        print(f"[NeuroGC] Server starting on port {port}")
        print(f"[NeuroGC] Using model: {model_name}")
        print(f"[NeuroGC] Saving metrics to {_csv_path}")

        yield

        gc_task.cancel()
        _profiler.stop()
        print("[NeuroGC] Server shutting down")

    app = create_app(
        title="NeuroGC Server (with intelligent GC)",
        description="FastAPI server with ML-based garbage collection optimization",
    )
    app.router.lifespan_context = lifespan

    register_common_endpoints(
        app=app,
        profiler=_profiler,
        server_name="with_neurogc",
        gc_flag_getter=get_and_reset_gc_flag,
        extra_health_info=lambda: {
            "model_loaded": _model_available,
            "gc_trigger_count": _gc_trigger_count,
        },
        extra_metrics_info=lambda: {
            "gc_trigger_count": _gc_trigger_count,
            "model_available": _model_available,
        },
    )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="NeuroGC Server with ML-driven GC"
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="lstm",
        choices=["lstm", "transformer", "feedforward", "classical"],
        help="Model type to use for GC prediction",
    )
    parser.add_argument(
        "--model-path",
        "-p",
        type=str,
        default=None,
        help="Path to model file (uses config default if not specified)",
    )
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

    app = create_neurogc_server(
        model_name=args.model,
        model_path=args.model_path,
    )

    import uvicorn

    port = args.port or config.server_ports.with_neurogc
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
