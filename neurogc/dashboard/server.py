import argparse
import asyncio
import csv
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from neurogc.config import get_config, load_config


class MetricsData(BaseModel):
    server: str
    time: float
    cpu: float
    mem: float
    disk_read: float
    disk_write: float
    net_sent: float
    net_recv: float
    rps: float
    p95: float
    p99: float
    gc_triggered: bool


@dataclass
class MetricsStore:
    max_history: int = 300

    with_gc_history: list = field(default_factory=list)
    without_gc_history: list = field(default_factory=list)

    with_gc_latest: Optional[dict] = None
    without_gc_latest: Optional[dict] = None

    latest_loss: float = 0.0

    gc_count_with: int = 0
    gc_count_without: int = 0

    def add_metrics(self, metrics: dict) -> None:
        server = metrics.get("server", "unknown")

        if server == "with_gc":
            self.with_gc_latest = metrics
            self.with_gc_history.append(metrics)
            if len(self.with_gc_history) > self.max_history:
                self.with_gc_history = self.with_gc_history[-self.max_history :]
            if metrics.get("gc_triggered"):
                self.gc_count_with += 1
        elif server == "without_gc":
            self.without_gc_latest = metrics
            self.without_gc_history.append(metrics)
            if len(self.without_gc_history) > self.max_history:
                self.without_gc_history = self.without_gc_history[
                    -self.max_history :
                ]
            if metrics.get("gc_triggered"):
                self.gc_count_without += 1

    def get_latest(self) -> dict:
        return {
            "with_gc": self.with_gc_latest,
            "without_gc": self.without_gc_latest,
            "gc_count_with": self.gc_count_with,
            "gc_count_without": self.gc_count_without,
            "latest_loss": self.latest_loss,
            "timestamp": time.time(),
        }

    def get_history(self, limit: int = 60) -> dict:
        return {
            "with_gc": self.with_gc_history[-limit:],
            "without_gc": self.without_gc_history[-limit:],
            "gc_count_with": self.gc_count_with,
            "gc_count_without": self.gc_count_without,
            "latest_loss": self.latest_loss,
        }

    def reset(self) -> None:
        self.with_gc_history.clear()
        self.without_gc_history.clear()
        self.with_gc_latest = None
        self.without_gc_latest = None
        self.gc_count_with = 0
        self.gc_count_without = 0


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        print(
            f"[Dashboard] Client connected. Total: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(
            f"[Dashboard] Client disconnected. Total: {len(self.active_connections)}"
        )

    async def broadcast(self, message: dict) -> None:
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


def load_csv_metrics(filepath: str) -> list[dict]:
    metrics = []
    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics.append(
                {
                    "time": float(row["timestamp"]),
                    "server": row["server"],
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
            )
    return metrics


def create_dashboard_server(
    replay_file: Optional[str] = None,
) -> tuple[FastAPI, MetricsStore, ConnectionManager]:
    config = get_config()
    metrics_store = MetricsStore()
    connection_manager = ConnectionManager()

    async def broadcast_metrics_loop():
        profile_interval = config.profile_interval

        while True:
            try:
                latest = metrics_store.get_latest()
                await connection_manager.broadcast(
                    {"type": "metrics_update", "data": latest}
                )
            except Exception as e:
                print(f"[Dashboard] Broadcast error: {e}")

            await asyncio.sleep(profile_interval)

    async def replay_from_csv(filepath: str):
        profile_interval = config.profile_interval

        print(f"[Dashboard] Loading metrics from {filepath}")
        all_metrics = load_csv_metrics(filepath)
        print(f"[Dashboard] Loaded {len(all_metrics)} entries")

        if not all_metrics:
            print("[Dashboard] No metrics to replay")
            return

        idx = 0
        total = len(all_metrics)

        while idx < total:
            metrics_pair = []
            current_time = all_metrics[idx]["time"]

            while (
                idx < total
                and abs(all_metrics[idx]["time"] - current_time) < 0.5
            ):
                metrics_pair.append(all_metrics[idx])
                idx += 1

            for m in metrics_pair:
                metrics_store.add_metrics(m)

            try:
                latest = metrics_store.get_latest()
                await connection_manager.broadcast(
                    {"type": "metrics_update", "data": latest}
                )
            except Exception as e:
                print(f"[Dashboard] Broadcast error: {e}")

            await asyncio.sleep(profile_interval)

        print("[Dashboard] Replay finished. Metrics will remain visible.")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        port = config.server_ports.metrics_server

        if replay_file:
            if not os.path.exists(replay_file):
                print(
                    f"[Dashboard] Error: Replay file not found: {replay_file}"
                )
                yield
                return

            print(f"[Dashboard] Starting in REPLAY mode on port {port}")
            print(f"[Dashboard] Replaying from: {replay_file}")
            background_task = asyncio.create_task(replay_from_csv(replay_file))
        else:
            print(f"[Dashboard] Starting in LIVE mode on port {port}")
            background_task = asyncio.create_task(broadcast_metrics_loop())

        yield

        background_task.cancel()
        print("[Dashboard] Shutting down")

    app = FastAPI(
        title="NeuroGC Metrics Dashboard",
        description="Real-time metrics aggregation and streaming server",
        lifespan=lifespan,
    )

    ui_paths = [
        Path(__file__).parent / "ui.html",
        Path.cwd() / "neurogc" / "dashboard" / "ui.html",
        Path.cwd() / "ui.html",
    ]

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        for ui_path in ui_paths:
            if ui_path.exists():
                return HTMLResponse(content=ui_path.read_text())
        return HTMLResponse(
            content="<h1>UI not found</h1><p>ui.html is missing</p>"
        )

    @app.get("/config")
    async def get_config_endpoint():
        cfg = get_config()
        return {
            "profile_interval": cfg.profile_interval,
            "server_ports": {
                "with_neurogc": cfg.server_ports.with_neurogc,
                "without_neurogc": cfg.server_ports.without_neurogc,
                "metrics_server": cfg.server_ports.metrics_server,
            },
            "gc_threshold": cfg.gc_threshold,
            "thresholds": {
                "cpu": cfg.thresholds.cpu,
                "memory": cfg.thresholds.memory,
                "disk_read": cfg.thresholds.disk_read,
                "disk_write": cfg.thresholds.disk_write,
                "net_sent": cfg.thresholds.net_sent,
                "net_recv": cfg.thresholds.net_recv,
                "p95": cfg.thresholds.p95,
                "p99": cfg.thresholds.p99,
            },
        }

    @app.post("/api/metrics")
    async def receive_metrics(metrics: MetricsData):
        metrics_dict = metrics.model_dump()
        metrics_store.add_metrics(metrics_dict)
        return {"status": "ok", "server": metrics.server}

    @app.get("/api/metrics/latest")
    async def get_latest_metrics():
        return metrics_store.get_latest()

    @app.get("/api/metrics/history")
    async def get_metrics_history(limit: int = 60):
        return metrics_store.get_history(limit)

    @app.post("/api/loss")
    async def update_loss(loss: float):
        metrics_store.latest_loss = loss
        return {"status": "ok", "loss": loss}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await connection_manager.connect(websocket)

        try:
            cfg = get_config()
            config_dict = {
                "profile_interval": cfg.profile_interval,
                "gc_threshold": cfg.gc_threshold,
            }
            await websocket.send_json(
                {
                    "type": "initial",
                    "data": metrics_store.get_history(60),
                    "config": config_dict,
                }
            )

            while True:
                try:
                    data = await asyncio.wait_for(
                        websocket.receive_text(), timeout=30.0
                    )

                    try:
                        message = json.loads(data)
                        if message.get("type") == "get_history":
                            limit = message.get("limit", 60)
                            await websocket.send_json(
                                {
                                    "type": "history",
                                    "data": metrics_store.get_history(limit),
                                }
                            )
                    except json.JSONDecodeError:
                        pass

                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"type": "ping"})
                    except Exception:
                        break

        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[Dashboard] WebSocket error: {e}")
        finally:
            connection_manager.disconnect(websocket)

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "server": "metrics_dashboard",
            "connected_clients": len(connection_manager.active_connections),
            "metrics_count": {
                "with_gc": len(metrics_store.with_gc_history),
                "without_gc": len(metrics_store.without_gc_history),
            },
        }

    return app, metrics_store, connection_manager


def main():
    parser = argparse.ArgumentParser(description="NeuroGC Metrics Dashboard")
    parser.add_argument(
        "--replay",
        "-r",
        type=str,
        default=None,
        help="Path to CSV file for replay mode (e.g., benchmark.csv)",
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

    app, _, _ = create_dashboard_server(replay_file=args.replay)

    import uvicorn

    port = args.port or config.server_ports.metrics_server
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
