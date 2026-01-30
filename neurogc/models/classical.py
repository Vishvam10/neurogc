import pickle
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from neurogc.config import ClassicalParams, get_config
from neurogc.models import register_model
from neurogc.models.base import BaseGCPredictor, ModelMetadata, TrainingResult
from neurogc.utils import INPUT_FEATURES

sklearn = None


def _import_sklearn():
    global sklearn
    if sklearn is None:
        try:
            import sklearn as sk
            sklearn = sk
        except ImportError:
            raise ImportError(
                "scikit-learn is required for classical models. "
                "Install with: pip install scikit-learn"
            )
    return sklearn


class ClassicalDataset:
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

        self.raw_features = self.df[self.feature_columns].values.astype(np.float32)

        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None

        if self.normalize:
            self.feature_means = self.raw_features.mean(axis=0)
            self.feature_stds = self.raw_features.std(axis=0)
            self.feature_stds[self.feature_stds == 0] = 1.0
            self.raw_features = (self.raw_features - self.feature_means) / self.feature_stds

        self._create_features_and_targets()

    def _create_features_and_targets(self) -> None:
        df = self.df

        mem_pressure = df["mem"].values / 100.0
        cpu_factor = df["cpu"].values / 100.0
        gc_recent = df["gc_triggered"].astype(float).values
        targets = np.clip(
            0.4 * mem_pressure + 0.3 * cpu_factor + 0.3 * (1 - gc_recent * 0.5), 0.0, 1.0
        ).astype(np.float32)

        X_list = []
        y_list = []

        for i in range(self.lookback, len(self.raw_features)):
            window = self.raw_features[i - self.lookback : i]

            flat_features = window.flatten()
            mean_features = window.mean(axis=0)
            std_features = window.std(axis=0)
            min_features = window.min(axis=0)
            max_features = window.max(axis=0)
            trend_features = window[-1] - window[0]

            all_features = np.concatenate([
                flat_features,
                mean_features,
                std_features,
                min_features,
                max_features,
                trend_features,
            ])

            X_list.append(all_features)
            y_list.append(targets[i])

        self.X = np.array(X_list)
        self.y = np.array(y_list)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        return self.X, self.y

    def get_normalization_params(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        return self.feature_means, self.feature_stds


@register_model("classical")
class ClassicalPredictor(BaseGCPredictor):
    SUPPORTED_ALGORITHMS = ["random_forest", "gradient_boosting", "xgboost"]

    def __init__(self, config: Optional[ClassicalParams] = None):
        super().__init__()

        if config is None:
            try:
                config = get_config().models.classical
            except Exception:
                config = ClassicalParams()

        self.config = config
        self._model = None
        self._buffer: deque = deque(maxlen=config.lookback)
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="classical",
            version="1.0.0",
            input_features=INPUT_FEATURES.copy(),
            sequence_length=self.config.lookback,
            description=f"Classical ML predictor ({self.config.algorithm})",
            supports_training=True,
            requires_sequence=False,
        )

    def _create_model(self):
        _import_sklearn()
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

        algorithm = self.config.algorithm

        if algorithm == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=42,
                n_jobs=-1,
            )
        elif algorithm == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth or 3,
                random_state=42,
            )
        elif algorithm == "xgboost":
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth or 6,
                    random_state=42,
                    n_jobs=-1,
                )
            except ImportError:
                raise ImportError(
                    "XGBoost is required for xgboost algorithm. "
                    "Install with: pip install xgboost"
                )
        else:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: {self.SUPPORTED_ALGORITHMS}"
            )

    def _engineer_features(self, window: np.ndarray) -> np.ndarray:
        flat_features = window.flatten()
        mean_features = window.mean(axis=0)
        std_features = window.std(axis=0)
        min_features = window.min(axis=0)
        max_features = window.max(axis=0)
        trend_features = window[-1] - window[0]

        return np.concatenate([
            flat_features,
            mean_features,
            std_features,
            min_features,
            max_features,
            trend_features,
        ])

    def train(self, data_path: Path, **kwargs) -> TrainingResult:
        start_time = time.time()

        dataset = ClassicalDataset(str(data_path), lookback=self.config.lookback)
        X, y = dataset.get_data()

        if len(X) == 0:
            raise ValueError(
                f"Dataset too small. Need at least {self.config.lookback + 1} rows."
            )

        self._model = self._create_model()
        self._model.fit(X, y)

        predictions = self._model.predict(X)
        mse = np.mean((predictions - y) ** 2)

        self._feature_means, self._feature_stds = dataset.get_normalization_params()
        self._is_loaded = True

        training_time = time.time() - start_time

        print(f"[Classical] Trained {self.config.algorithm} on {len(X)} samples")

        return TrainingResult(
            epochs=1,
            final_loss=mse,
            best_loss=mse,
            training_time_seconds=training_time,
            metrics={
                "dataset_size": len(X),
                "algorithm": self.config.algorithm,
                "n_estimators": self.config.n_estimators,
            },
        )

    def save(self, path: Path) -> None:
        if self._model is None:
            raise RuntimeError("No model to save.")

        data = {
            "model": self._model,
            "config": {
                "algorithm": self.config.algorithm,
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth,
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
        }

        with open(str(path), "wb") as f:
            pickle.dump(data, f)

        print(f"[Classical] Model saved to {path}")

    def load(self, path: Path) -> None:
        with open(str(path), "rb") as f:
            data = pickle.load(f)

        self._model = data["model"]

        config = data.get("config", {})
        lookback = config.get("lookback", self.config.lookback)
        self._buffer = deque(maxlen=lookback)

        norm_params = data.get("norm_params", {})
        feature_means = norm_params.get("feature_means")
        feature_stds = norm_params.get("feature_stds")

        if feature_means:
            self._feature_means = np.array(feature_means, dtype=np.float32)
        if feature_stds:
            self._feature_stds = np.array(feature_stds, dtype=np.float32)

        self._is_loaded = True
        print(f"[Classical] Model loaded from {path}")

    def predict(self) -> float:
        if not self.can_predict():
            return 0.0

        if self._model is None:
            return 0.0

        window = np.array(list(self._buffer))

        if self._feature_means is not None and self._feature_stds is not None:
            window = (window - self._feature_means) / self._feature_stds

        features = self._engineer_features(window)

        prediction = self._model.predict([features])[0]

        return float(np.clip(prediction, 0.0, 1.0))

    def can_predict(self) -> bool:
        return len(self._buffer) >= self.config.lookback and self._model is not None

    def add_metrics(self, metrics: dict) -> None:
        features = np.array(
            [float(metrics.get(k, 0)) for k in INPUT_FEATURES], dtype=np.float32
        )
        self._buffer.append(features)

    def reset(self) -> None:
        self._buffer.clear()

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        if self._model is None or not hasattr(self._model, "feature_importances_"):
            return None

        importances = self._model.feature_importances_

        lookback = self.config.lookback

        feature_names = []

        for t in range(lookback):
            for f in INPUT_FEATURES:
                feature_names.append(f"{f}_t-{lookback-t}")

        for stat in ["mean", "std", "min", "max", "trend"]:
            for f in INPUT_FEATURES:
                feature_names.append(f"{f}_{stat}")

        base_importance = {f: 0.0 for f in INPUT_FEATURES}

        for name, imp in zip(feature_names, importances):
            for f in INPUT_FEATURES:
                if name.startswith(f):
                    base_importance[f] += imp
                    break

        total = sum(base_importance.values())
        if total > 0:
            base_importance = {k: v / total for k, v in base_importance.items()}

        return base_importance


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test Classical ML GC predictor")
    parser.add_argument("--train", type=str, help="Path to CSV file for training")
    parser.add_argument("--model", type=str, default="gc_model_classical.pkl", help="Model path")
    parser.add_argument("--algorithm", type=str, default="random_forest",
                        choices=ClassicalPredictor.SUPPORTED_ALGORITHMS, help="Algorithm")
    parser.add_argument("--test", action="store_true", help="Test the model")

    args = parser.parse_args()

    if args.train:
        print(f"Training {args.algorithm} model from {args.train}...")
        config = ClassicalParams(algorithm=args.algorithm)
        predictor = ClassicalPredictor(config)
        result = predictor.train(Path(args.train))
        predictor.save(Path(args.model))
        print(f"Training complete. MSE: {result.final_loss:.6f}")

        importance = predictor.get_feature_importance()
        if importance:
            print("\nFeature Importance:")
            for k, v in sorted(importance.items(), key=lambda x: -x[1]):
                print(f"  {k}: {v:.4f}")

    elif args.test:
        print("Testing model...")
        try:
            predictor = ClassicalPredictor()
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
