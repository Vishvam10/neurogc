import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neurogc.config import LSTMParams, get_config
from neurogc.models import register_model
from neurogc.models.base import BaseGCPredictor, TrainingResult
from neurogc.utils import INPUT_FEATURES


class GCDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        sequence_length: int = 10,
        feature_columns: Optional[list[str]] = None,
        normalize: bool = True,
    ):
        self.sequence_length = sequence_length
        self.normalize = normalize
        self.feature_columns = feature_columns or INPUT_FEATURES.copy()

        self.df = pd.read_csv(csv_path)

        if "gc_triggered" in self.df.columns:
            self.df["gc_triggered"] = self.df["gc_triggered"].astype(int)

        self.features = self.df[self.feature_columns].values.astype(np.float32)

        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        if self.normalize:
            self.feature_means = self.features.mean(axis=0)
            self.feature_stds = self.features.std(axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0
            self.features = (
                self.features - self.feature_means
            ) / self.feature_stds

        self._create_targets()

    def _create_targets(self) -> None:
        df = self.df
        mem_pressure = df["mem"].values / 100.0
        cpu_factor = df["cpu"].values / 100.0
        gc_recent = df["gc_triggered"].astype(float).values

        self.targets = np.clip(
            0.4 * mem_pressure + 0.3 * cpu_factor + 0.3 * (1 - gc_recent * 0.5),
            0.0,
            1.0,
        ).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.features) - self.sequence_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx : idx + self.sequence_length])
        y = torch.tensor([self.targets[idx + self.sequence_length - 1]])
        return x, y

    def get_normalization_params(
        self,
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.feature_means, self.feature_stds


class LSTMNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        out = self.sigmoid(out)
        return out


@register_model("lstm")
class LSTMPredictor(BaseGCPredictor):
    def __init__(self, config: Optional[LSTMParams] = None):
        super().__init__()

        if config is None:
            try:
                config = get_config().models.lstm
            except Exception:
                config = LSTMParams()

        self.config = config
        self._model: Optional[LSTMNetwork] = None
        self._buffer: deque = deque(maxlen=config.sequence_length)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def train(self, data_path: Path, **kwargs) -> TrainingResult:
        start_time = time.time()

        epochs = kwargs.get("epochs", self.config.epochs)
        learning_rate = kwargs.get("learning_rate", self.config.learning_rate)
        batch_size = kwargs.get("batch_size", self.config.batch_size)

        dataset = GCDataset(
            str(data_path), sequence_length=self.config.sequence_length
        )

        if len(dataset) == 0:
            actual_rows = len(dataset.df) if hasattr(dataset, "df") else 0
            raise ValueError(
                f"Dataset too small for training. "
                f"Found {actual_rows} rows, need at least {self.config.sequence_length + 1}. "
                f"Collect more data or reduce sequence_length in config."
            )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model = LSTMNetwork(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
        ).to(self._device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

        self._model.train()
        best_loss = float("inf")
        final_loss = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                outputs = self._model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            final_loss = avg_loss
            best_loss = min(best_loss, avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.6f}")

        self._feature_means, self._feature_stds = (
            dataset.get_normalization_params()
        )
        self._is_loaded = True

        training_time = time.time() - start_time

        return TrainingResult(
            epochs=epochs,
            final_loss=final_loss,
            best_loss=best_loss,
            training_time_seconds=training_time,
            metrics={"dataset_size": len(dataset), "batch_size": batch_size},
        )

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save. Train or load a model first.")

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": {
                    "input_size": self.config.input_size,
                    "hidden_size": self.config.hidden_size,
                    "num_layers": self.config.num_layers,
                },
                "norm_params": {
                    "feature_means": (
                        self._feature_means.tolist()
                        if self._feature_means is not None
                        else None
                    ),
                    "feature_stds": (
                        self._feature_stds.tolist()
                        if self._feature_stds is not None
                        else None
                    ),
                    "sequence_length": self.config.sequence_length,
                },
            },
            str(path),
        )
        print(f"[LSTM] Model saved to {path}")

    def load(self, path: Path) -> None:
        checkpoint = torch.load(
            str(path), map_location=self._device, weights_only=False
        )

        model_config = checkpoint["config"]
        norm_params = checkpoint.get("norm_params", {})

        self._model = LSTMNetwork(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config["num_layers"],
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        feature_means = norm_params.get("feature_means")
        feature_stds = norm_params.get("feature_stds")
        sequence_length = norm_params.get(
            "sequence_length", self.config.sequence_length
        )

        if feature_means:
            self._feature_means = np.array(feature_means, dtype=np.float32)
        if feature_stds:
            self._feature_stds = np.array(feature_stds, dtype=np.float32)

        self._buffer = deque(maxlen=sequence_length)
        self._is_loaded = True

        print(f"[LSTM] Model loaded from {path}")

    def predict(self) -> float:
        if not self.can_predict():
            return 0.0

        if self._model is None:
            return 0.0

        features = np.array(list(self._buffer))

        if self._feature_means is not None and self._feature_stds is not None:
            features = (features - self._feature_means) / self._feature_stds

        x = (
            torch.tensor(features, dtype=torch.float32)
            .unsqueeze(0)
            .to(self._device)
        )

        with torch.no_grad():
            output = self._model(x)

        return output.item()

    def can_predict(self) -> bool:
        return (
            len(self._buffer) >= self.config.sequence_length
            and self._model is not None
        )

    def add_metrics(self, metrics: dict) -> None:
        features = np.array(
            [float(metrics.get(k, 0)) for k in INPUT_FEATURES], dtype=np.float32
        )
        self._buffer.append(features)

    def reset(self) -> None:
        self._buffer.clear()


def train_model(
    csv_path: str,
    config_path: str = "config.json",
    model_save_path: Optional[str] = None,
) -> tuple[LSTMNetwork, float, dict]:
    from neurogc.config import load_config as load_cfg

    config = load_cfg(config_path)
    predictor = LSTMPredictor(config.models.lstm)

    result = predictor.train(Path(csv_path))

    save_path = model_save_path or config.model_path
    predictor.save(Path(save_path))

    norm_params = {
        "feature_means": (
            predictor._feature_means.tolist()
            if predictor._feature_means is not None
            else None
        ),
        "feature_stds": (
            predictor._feature_stds.tolist()
            if predictor._feature_stds is not None
            else None
        ),
        "sequence_length": predictor.config.sequence_length,
    }

    return predictor._model, result.final_loss, norm_params


def load_model(model_path: str, device: str = "cpu") -> LSTMPredictor:
    predictor = LSTMPredictor()
    predictor._device = device
    predictor.load(Path(model_path))
    return predictor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or test LSTM GC predictor"
    )
    parser.add_argument(
        "--train", type=str, help="Path to CSV file for training"
    )
    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to config file"
    )
    parser.add_argument(
        "--model", type=str, default="gc_model.pth", help="Path to model file"
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the model with sample data"
    )

    args = parser.parse_args()

    if args.train:
        print(f"Training LSTM model from {args.train}...")
        model, loss, params = train_model(args.train, args.config, args.model)
        print(f"Training complete. Final loss: {loss:.6f}")

    elif args.test:
        print("Testing model with sample metrics...")
        sample_metrics = {
            "cpu": 45.0,
            "mem": 60.0,
            "disk_read": 1000000.0,
            "disk_write": 500000.0,
            "net_sent": 100000.0,
            "net_recv": 200000.0,
            "rps": 100.0,
            "p95": 50.0,
            "p99": 100.0,
            "gc_triggered": False,
        }

        try:
            predictor = load_model(args.model)
            for _ in range(predictor.config.sequence_length):
                predictor.add_metrics(sample_metrics)
            prediction = predictor.predict()
            print(f"GC Urgency Prediction: {prediction:.4f}")
        except FileNotFoundError:
            print(f"Model file {args.model} not found. Train a model first.")

    else:
        parser.print_help()
