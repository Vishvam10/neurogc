import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


def load_config(config_path: str = "config.json") -> dict:
    with open(config_path, 'r') as f:
        return json.load(f)


config = load_config()


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
    
    # Keep 5 minutes of data at 1s intervals
    max_history: int = 300
    
    # Metrics history for each server
    with_gc_history: list = field(default_factory=list)
    without_gc_history: list = field(default_factory=list)
    
    # Latest metrics
    with_gc_latest: Optional[dict] = None
    without_gc_latest: Optional[dict] = None
    
    # Model loss tracking
    latest_loss: float = 0.0
    
    # GC event counts
    gc_count_with: int = 0
    gc_count_without: int = 0

    def add_metrics(self, metrics: dict) -> None:
        server = metrics.get('server', 'unknown')
        
        if server == 'with_gc':
            self.with_gc_latest = metrics
            self.with_gc_history.append(metrics)
            if len(self.with_gc_history) > self.max_history:
                self.with_gc_history = self.with_gc_history[-self.max_history:]
            if metrics.get('gc_triggered'):
                self.gc_count_with += 1
        elif server == 'without_gc':
            self.without_gc_latest = metrics
            self.without_gc_history.append(metrics)
            if len(self.without_gc_history) > self.max_history:
                self.without_gc_history = self.without_gc_history[-self.max_history:]
            if metrics.get('gc_triggered'):
                self.gc_count_without += 1

    def get_latest(self) -> dict:
        return {
            'with_gc': self.with_gc_latest,
            'without_gc': self.without_gc_latest,
            'gc_count_with': self.gc_count_with,
            'gc_count_without': self.gc_count_without,
            'latest_loss': self.latest_loss,
            'timestamp': time.time()
        }

    def get_history(self, limit: int = 60) -> dict:
        return {
            'with_gc': self.with_gc_history[-limit:],
            'without_gc': self.without_gc_history[-limit:],
            'gc_count_with': self.gc_count_with,
            'gc_count_without': self.gc_count_without,
            'latest_loss': self.latest_loss
        }


class ConnectionManager:

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[MetricsServer] Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[MetricsServer] Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict) -> None:
        if not self.active_connections:
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# Global instances
metrics_store = MetricsStore()
connection_manager = ConnectionManager()


async def broadcast_metrics_loop():
    """Background task to broadcast metrics to WebSocket clients."""
    profile_interval = config.get('profile_interval', 1.0)
    
    while True:
        try:
            # Broadcast latest metrics to all connected clients
            latest = metrics_store.get_latest()
            await connection_manager.broadcast({
                'type': 'metrics_update',
                'data': latest
            })
        except Exception as e:
            print(f"[MetricsServer] Broadcast error: {e}")
        
        await asyncio.sleep(profile_interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    broadcast_task = asyncio.create_task(broadcast_metrics_loop())
    print(f"[MetricsServer] Starting on port {config['server_ports']['metrics_server']}")
    
    yield
    
    broadcast_task.cancel()
    print("[MetricsServer] Shutting down")


app = FastAPI(
    title="NeuroGC Metrics Server",
    description="Real-time metrics aggregation and streaming server",
    lifespan=lifespan
)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
    if os.path.exists(ui_path):
        with open(ui_path, 'r') as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>UI not found</h1><p>ui.html is missing</p>")


@app.get("/config")
async def get_config():
    return config


# Receive metrics from Locust profile collector. This endpoint is called 
# periodically by the ProfileCollector in locustfile.py.
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
        # Send initial data
        await websocket.send_json({
            'type': 'initial',
            'data': metrics_store.get_history(60),
            'config': config
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                
                try:
                    message = json.loads(data)
                    if message.get('type') == 'get_history':
                        limit = message.get('limit', 60)
                        await websocket.send_json({
                            'type': 'history',
                            'data': metrics_store.get_history(limit)
                        })
                except json.JSONDecodeError:
                    pass
                    
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
                try:
                    await websocket.send_json({'type': 'ping'})
                except Exception:
                    break
                    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[MetricsServer] WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "server": "metrics_server",
        "connected_clients": len(connection_manager.active_connections),
        "metrics_count": {
            "with_gc": len(metrics_store.with_gc_history),
            "without_gc": len(metrics_store.without_gc_history)
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    port = config['server_ports']['metrics_server']
    uvicorn.run(app, host="0.0.0.0", port=port)
