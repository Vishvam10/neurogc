# neurogc/models/model.py

from typing import Callable

from neurogc.models.lstm import (
    train_model as train_lstm,
    load_model as load_lstm,
)
from neurogc.models.classical import (
    train_model as train_classical,
    load_model as load_classical,
)
from neurogc.models.feedforward import (
    train_model as train_feedforward,
    load_model as load_feedforward,
)
from neurogc.models.transformer import (
    train_model as train_transformer,
    load_model as load_transformer,
)

TRAINERS: dict[str, Callable] = {
    "lstm": train_lstm,
    "classical": train_classical,
    "feedforward": train_feedforward,
    "transformer": train_transformer,
}

LOADERS: dict[str, Callable] = {
    "lstm": load_lstm,
    "classical": load_classical,
    "feedforward": load_feedforward,
    "transformer": load_transformer,
}


def train_model(
    arch: str,
    csv_path: str,
    config_path: str,
    model_path: str,
):
    if arch not in TRAINERS:
        raise ValueError(f"Unknown architecture: {arch}")

    return TRAINERS[arch](csv_path, config_path, model_path)


def load_model(
    arch: str,
    model_path: str,
):
    if arch not in LOADERS:
        raise ValueError(f"Unknown architecture: {arch}")

    return LOADERS[arch](model_path)


def predict(
    metrics: dict,
    arch: str = "lstm",
    model_path: str = "gc_model.pth",
) -> float:
    wrapper = load_model(arch, model_path)

    lookback = (
        wrapper.config.sequence_length
        if hasattr(wrapper.config, "sequence_length")
        else wrapper.config.lookback
    )

    for _ in range(lookback):
        wrapper.add_metrics(metrics)

    return wrapper.predict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or test GC predictor model"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="lstm",
        choices=["lstm", "classical", "feedforward", "transformer"],
        help="Model architecture",
    )
    parser.add_argument(
        "--train", type=str, help="Path to CSV file for training"
    )
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model", type=str, default="gc_model.pth")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.train:
        print(f"Training {args.arch} model from {args.train} ...")
        _, loss, _ = train_model(args.arch, args.train, args.config, args.model)
        print(f"Training complete. Final loss : {loss:.6f}")

    elif args.test:
        sample_metrics = {
            "cpu": 45.0,
            "mem": 60.0,
            "disk_read": 1e6,
            "disk_write": 5e5,
            "net_sent": 1e5,
            "net_recv": 2e5,
            "rps": 100.0,
            "p95": 50.0,
            "p99": 100.0,
            "gc_triggered": False,
        }

        prediction = predict(
            sample_metrics,
            arch=args.arch,
            model_path=args.model,
        )
        print(f"GC Urgency Prediction : {prediction:.4f}")
