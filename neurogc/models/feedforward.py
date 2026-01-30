import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neurogc.config import FeedforwardParams, get_config
from neurogc.models import register_model
from neurogc.models.base import BaseGCPredictor, ModelMetadata, TrainingResult
from neurogc.utils import INPUT_FEATURES


class FeedforwardNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] = [64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.input_size = input_size

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FeedforwardDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        lookback: int = 5,
        feature_columns: Optional[list[str]] = None,
        normalize: bool = True,
    ):
        self.lookback = lookback
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
            self.features = (self.features - self.feature_means) / self.feature_stds

        self._create_targets()

    def _create_targets(self) -> None:
        df = self.df
        mem_pressure = df["mem"].values / 100.0
        cpu_factor = df["cpu"].values / 100.0
        gc_recent = df["gc_triggered"].astype(float).values

        self.targets = np.clip(
            0.4 * mem_pressure + 0.3 * cpu_factor + 0.3 * (1 - gc_recent * 0.5), 0.0, 1.0
        ).astype(np.float32)

    def __len__(self) -> int:
        return max(0, len(self.features) - self.lookback)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx : idx + self.lookback].flatten())
        y = torch.tensor([self.targets[idx + self.lookback - 1]])
        return x, y

    def get_normalization_params(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.feature_means, self.feature_stds


@register_model("feedforward")
class FeedforwardPredictor(BaseGCPredictor):
    def __init__(self, config: Optional[FeedforwardParams] = None):
        super().__init__()

        if config is None:
            try:
                config = get_config().models.feedforward
            except Exception:
                config = FeedforwardParams()

        self.config = config
        self._model: Optional[FeedforwardNetwork] = None
        self._buffer: deque = deque(maxlen=config.lookback)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="feedforward",
            version="1.0.0",
            input_features=INPUT_FEATURES.copy(),
            sequence_length=self.config.lookback,
            description="Feedforward MLP for fast GC prediction",
            supports_training=True,
            requires_sequence=False,
        )

    def train(self, data_path: Path, **kwargs) -> TrainingResult:
        start_time = time.time()

        epochs = kwargs.get("epochs", self.config.epochs)
        learning_rate = kwargs.get("learning_rate", self.config.learning_rate)
        batch_size = kwargs.get("batch_size", self.config.batch_size)

        dataset = FeedforwardDataset(str(data_path), lookback=self.config.lookback)

        if len(dataset) == 0:
            actual_rows = len(dataset.df) if hasattr(dataset, "df") else 0
            raise ValueError(
                f"Dataset too small. Found {actual_rows} rows, need at least {self.config.lookback + 1}."
            )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_size = len(INPUT_FEATURES) * self.config.lookback
        self._model = FeedforwardNetwork(
            input_size=input_size,
            hidden_sizes=self.config.hidden_sizes,
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

        self._feature_means, self._feature_stds = dataset.get_normalization_params()
        self._is_loaded = True

        return TrainingResult(
            epochs=epochs,
            final_loss=final_loss,
            best_loss=best_loss,
            training_time_seconds=time.time() - start_time,
            metrics={"dataset_size": len(dataset), "input_size": input_size},
        )

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": {
                    "input_size": self._model.input_size,
                    "hidden_sizes": self.config.hidden_sizes,
                    "lookback": self.config.lookback,
                },
                "norm_params": {
                    "feature_means": (
                        self._feature_means.tolist() if self._feature_means is not None else None
                    ),
                    "feature_stds": (
                        self._feature_stds.tolist() if self._feature_stds is not None else None
                    ),
                },
            },
            str(path),
        )
        print(f"[Feedforward] Model saved to {path}")

    def load(self, path: Path) -> None:
        checkpoint = torch.load(str(path), map_location=self._device, weights_only=False)

        model_config = checkpoint["config"]
        norm_params = checkpoint.get("norm_params", {})

        self._model = FeedforwardNetwork(
            input_size=model_config["input_size"],
            hidden_sizes=model_config["hidden_sizes"],
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self._device)
        self._model.eval()

        lookback = model_config.get("lookback", self.config.lookback)
        self._buffer = deque(maxlen=lookback)

        feature_means = norm_params.get("feature_means")
        feature_stds = norm_params.get("feature_stds")

        if feature_means:
            self._feature_means = np.array(feature_means, dtype=np.float32)
        if feature_stds:
            self._feature_stds = np.array(feature_stds, dtype=np.float32)

        self._is_loaded = True
        print(f"[Feedforward] Model loaded from {path}")

    def predict(self) -> float:
        if not self.can_predict():
            return 0.0

        if self._model is None:
            return 0.0

        features = np.array(list(self._buffer))

        if self._feature_means is not None and self._feature_stds is not None:
            features = (features - self._feature_means) / self._feature_stds

        x = torch.tensor(features.flatten(), dtype=torch.float32).unsqueeze(0).to(self._device)

        with torch.no_grad():
            output = self._model(x)

        return output.item()

    def can_predict(self) -> bool:
        return len(self._buffer) >= self.config.lookback and self._model is not None

    def add_metrics(self, metrics: dict) -> None:
        features = np.array(
            [float(metrics.get(k, 0)) for k in INPUT_FEATURES], dtype=np.float32
        )
        self._buffer.append(features)

    def reset(self) -> None:
        self._buffer.clear()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test Feedforward GC predictor")
    parser.add_argument("--train", type=str, help="Path to CSV file for training")
    parser.add_argument("--model", type=str, default="gc_model_feedforward.pth", help="Model path")
    parser.add_argument("--test", action="store_true", help="Test the model")

    args = parser.parse_args()

    if args.train:
        print(f"Training Feedforward model from {args.train}...")
        predictor = FeedforwardPredictor()
        result = predictor.train(Path(args.train))
        predictor.save(Path(args.model))
        print(f"Training complete. Final loss: {result.final_loss:.6f}")

    elif args.test:
        print("Testing model...")
        try:
            predictor = FeedforwardPredictor()
            predictor.load(Path(args.model))
            sample = {"cpu": 45.0, "mem": 60.0, "disk_read": 1e6, "disk_write": 5e5,
                      "net_sent": 1e5, "net_recv": 2e5, "rps": 100.0, "p95": 50.0,
                      "p99": 100.0, "gc_triggered": False}
            for _ in range(predictor.config.lookback):
                predictor.add_metrics(sample)
            print(f"GC Urgency Prediction: {predictor.predict():.4f}")
        except FileNotFoundError:
            print(f"Model file {args.model} not found.")

    else:
        parser.print_help()
