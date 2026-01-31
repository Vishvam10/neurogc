import math
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neurogc.config import TransformerParams, get_config
from neurogc.models import register_model
from neurogc.models.base import BaseGCPredictor, ModelMetadata, TrainingResult
from neurogc.utils import INPUT_FEATURES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)].transpose(0, 1)
        return self.dropout(x)


class TransformerNetwork(nn.Module):
    def __init__(
        self,
        input_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


class TransformerDataset(Dataset):
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


@register_model("transformer")
class TransformerPredictor(BaseGCPredictor):
    def __init__(self, config: Optional[TransformerParams] = None):
        super().__init__()

        if config is None:
            try:
                config = get_config().models.transformer
            except Exception:
                config = TransformerParams()

        self.config = config
        self._model: Optional[TransformerNetwork] = None
        self._buffer: deque = deque(maxlen=config.sequence_length)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="transformer",
            version="1.0.0",
            input_features=INPUT_FEATURES.copy(),
            sequence_length=self.config.sequence_length,
            description="Transformer-based GC predictor with self-attention",
            supports_training=True,
            requires_sequence=True,
        )

    def train(self, data_path: Path, **kwargs) -> TrainingResult:
        start_time = time.time()

        epochs = kwargs.get("epochs", self.config.epochs)
        learning_rate = kwargs.get("learning_rate", self.config.learning_rate)
        batch_size = kwargs.get("batch_size", self.config.batch_size)

        dataset = TransformerDataset(
            str(data_path), sequence_length=self.config.sequence_length
        )

        if len(dataset) == 0:
            actual_rows = len(dataset.df) if hasattr(dataset, "df") else 0
            raise ValueError(
                f"Dataset too small. Found {actual_rows} rows, need at least {self.config.sequence_length + 1}."
            )

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self._model = TransformerNetwork(
            input_size=len(INPUT_FEATURES),
            d_model=self.config.d_model,
            nhead=self.config.nhead,
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

        return TrainingResult(
            epochs=epochs,
            final_loss=final_loss,
            best_loss=best_loss,
            training_time_seconds=time.time() - start_time,
            metrics={"dataset_size": len(dataset)},
        )


    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")

        torch.save(
            {
                "model_state_dict": self._model.state_dict(),
                "config": {
                    "input_size": len(INPUT_FEATURES),
                    "d_model": self.config.d_model,
                    "nhead": self.config.nhead,
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
        print(f"[Transformer] Model saved to {path}")

    def load(self, path: Path) -> None:
        checkpoint = torch.load(
            str(path), map_location=self._device, weights_only=False
        )

        model_config = checkpoint["config"]
        norm_params = checkpoint.get("norm_params", {})

        self._model = TransformerNetwork(
            input_size=model_config["input_size"],
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
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

        print(f"[Transformer] Model loaded from {path}")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train or test Transformer GC predictor"
    )
    parser.add_argument(
        "--train", type=str, help="Path to CSV file for training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gc_model_transformer.pth",
        help="Model path",
    )
    parser.add_argument("--test", action="store_true", help="Test the model")

    args = parser.parse_args()

    if args.train:
        print(f"Training Transformer model from {args.train}...")
        predictor = TransformerPredictor()
        result = predictor.train(Path(args.train))
        predictor.save(Path(args.model))
        print(f"Training complete. Final loss: {result.final_loss:.6f}")

    elif args.test:
        print("Testing model...")
        try:
            predictor = TransformerPredictor()
            predictor.load(Path(args.model))
            sample = {
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
            for _ in range(predictor.config.sequence_length):
                predictor.add_metrics(sample)
            print(f"GC Urgency Prediction: {predictor.predict():.4f}")
        except FileNotFoundError:
            print(f"Model file {args.model} not found.")

    else:
        parser.print_help()
