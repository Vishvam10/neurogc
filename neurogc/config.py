import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LSTMParams:
    input_size: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    sequence_length: int = 10
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32


@dataclass
class TransformerParams:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    sequence_length: int = 10
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32


@dataclass
class FeedforwardParams:
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 32])
    lookback: int = 5
    epochs: int = 100
    learning_rate: float = 0.001
    batch_size: int = 32


@dataclass
class ClassicalParams:
    algorithm: str = "random_forest"
    n_estimators: int = 100
    max_depth: Optional[int] = None
    lookback: int = 5


@dataclass
class ServerPorts:
    with_neurogc: int = 8001
    without_neurogc: int = 8002
    metrics_server: int = 8003


@dataclass
class Thresholds:
    cpu: float = 80.0
    memory: float = 75.0
    disk_read: float = 100_000_000
    disk_write: float = 100_000_000
    net_sent: float = 10_000_000
    net_recv: float = 10_000_000
    p95: float = 500.0
    p99: float = 1000.0


@dataclass
class LocustConfig:
    host_with_gc: str = "http://localhost:8001"
    host_without_gc: str = "http://localhost:8002"
    metrics_server_url: str = "http://localhost:8003"


@dataclass
class ModelsConfig:
    lstm: LSTMParams = field(default_factory=LSTMParams)
    transformer: TransformerParams = field(default_factory=TransformerParams)
    feedforward: FeedforwardParams = field(default_factory=FeedforwardParams)
    classical: ClassicalParams = field(default_factory=ClassicalParams)


@dataclass
class Config:
    profile_interval: float = 1.0
    server_ports: ServerPorts = field(default_factory=ServerPorts)
    gc_threshold: float = 0.7
    thresholds: Thresholds = field(default_factory=Thresholds)
    csv_path: str = "profiler_data.csv"
    model_path: str = "gc_model.pth"
    locust: LocustConfig = field(default_factory=LocustConfig)
    default_model: str = "lstm"
    models: ModelsConfig = field(default_factory=ModelsConfig)

    def get_model_config(self, model_name: str) -> Any:
        model_configs = {
            "lstm": self.models.lstm,
            "transformer": self.models.transformer,
            "feedforward": self.models.feedforward,
            "classical": self.models.classical,
        }
        if model_name not in model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        return model_configs[model_name]


_config: Optional[Config] = None
_config_path: Optional[str] = None


def _parse_server_ports(data: dict) -> ServerPorts:
    return ServerPorts(
        with_neurogc=data.get("with_neurogc", 8001),
        without_neurogc=data.get("without_neurogc", 8002),
        metrics_server=data.get("metrics_server", 8003),
    )


def _parse_thresholds(data: dict) -> Thresholds:
    return Thresholds(
        cpu=data.get("cpu", 80.0),
        memory=data.get("memory", 75.0),
        disk_read=data.get("disk_read", 100_000_000),
        disk_write=data.get("disk_write", 100_000_000),
        net_sent=data.get("net_sent", 10_000_000),
        net_recv=data.get("net_recv", 10_000_000),
        p95=data.get("p95", 500.0),
        p99=data.get("p99", 1000.0),
    )


def _parse_locust(data: dict) -> LocustConfig:
    return LocustConfig(
        host_with_gc=data.get("host_with_gc", "http://localhost:8001"),
        host_without_gc=data.get("host_without_gc", "http://localhost:8002"),
        metrics_server_url=data.get(
            "metrics_server_url", "http://localhost:8003"
        ),
    )


def _parse_lstm_params(data: dict) -> LSTMParams:
    return LSTMParams(
        input_size=data.get("input_size", 10),
        hidden_size=data.get("hidden_size", 64),
        num_layers=data.get("num_layers", 2),
        sequence_length=data.get("sequence_length", 10),
        epochs=data.get("epochs", 100),
        learning_rate=data.get("learning_rate", 0.001),
        batch_size=data.get("batch_size", 32),
    )


def _parse_transformer_params(data: dict) -> TransformerParams:
    return TransformerParams(
        d_model=data.get("d_model", 64),
        nhead=data.get("nhead", 4),
        num_layers=data.get("num_layers", 2),
        sequence_length=data.get("sequence_length", 10),
        epochs=data.get("epochs", 100),
        learning_rate=data.get("learning_rate", 0.001),
        batch_size=data.get("batch_size", 32),
    )


def _parse_feedforward_params(data: dict) -> FeedforwardParams:
    return FeedforwardParams(
        hidden_sizes=data.get("hidden_sizes", [64, 32]),
        lookback=data.get("lookback", 5),
        epochs=data.get("epochs", 100),
        learning_rate=data.get("learning_rate", 0.001),
        batch_size=data.get("batch_size", 32),
    )


def _parse_classical_params(data: dict) -> ClassicalParams:
    return ClassicalParams(
        algorithm=data.get("algorithm", "random_forest"),
        n_estimators=data.get("n_estimators", 100),
        max_depth=data.get("max_depth"),
        lookback=data.get("lookback", 5),
    )


def _parse_models_config(data: dict) -> ModelsConfig:
    return ModelsConfig(
        lstm=_parse_lstm_params(data.get("lstm", {})),
        transformer=_parse_transformer_params(data.get("transformer", {})),
        feedforward=_parse_feedforward_params(data.get("feedforward", {})),
        classical=_parse_classical_params(data.get("classical", {})),
    )


def load_config(path: str = "config.json") -> Config:
    global _config, _config_path

    if _config is not None and _config_path == path:
        return _config

    config_path = Path(path)
    if not config_path.is_absolute():
        if not config_path.exists():
            module_dir = Path(__file__).parent.parent
            alt_path = module_dir / path
            if alt_path.exists():
                config_path = alt_path

    if not config_path.exists():
        print(
            f"[Config] Warning: Config file not found at {path}, using defaults"
        )
        _config = Config()
        _config_path = path
        return _config

    with open(config_path, "r") as f:
        data = json.load(f)

    models_data = data.get("models", {})
    models = _parse_models_config(models_data)

    _config = Config(
        profile_interval=data.get("profile_interval", 1.0),
        server_ports=_parse_server_ports(data.get("server_ports", {})),
        gc_threshold=data.get("gc_threshold", 0.7),
        thresholds=_parse_thresholds(data.get("thresholds", {})),
        csv_path=data.get("csv_path", "profiler_data.csv"),
        model_path=data.get("model_path", "gc_model.pth"),
        locust=_parse_locust(data.get("locust", {})),
        default_model=data.get("default_model", "lstm"),
        models=models,
    )

    _config_path = path
    return _config


def get_config() -> Config:
    global _config
    if _config is None:
        return load_config()
    return _config


def reload_config(path: Optional[str] = None) -> Config:
    global _config, _config_path
    _config = None
    return load_config(path or _config_path or "config.json")
