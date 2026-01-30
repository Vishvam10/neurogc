from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from neurogc.utils import INPUT_FEATURES


@dataclass
class ModelMetadata:
    name: str
    version: str
    input_features: list[str] = field(
        default_factory=lambda: INPUT_FEATURES.copy()
    )
    sequence_length: int = 1
    description: str = ""
    supports_training: bool = True
    requires_sequence: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "input_features": self.input_features,
            "sequence_length": self.sequence_length,
            "description": self.description,
            "supports_training": self.supports_training,
            "requires_sequence": self.requires_sequence,
        }


@dataclass
class TrainingResult:
    epochs: int
    final_loss: float
    best_loss: float
    training_time_seconds: float
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "training_time_seconds": self.training_time_seconds,
            "metrics": self.metrics,
        }


class BaseGCPredictor(ABC):
    def __init__(self):
        self._is_loaded = False

    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        pass

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @abstractmethod
    def train(self, data_path: Path, **kwargs) -> TrainingResult:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass

    @abstractmethod
    def predict(self) -> float:
        pass

    @abstractmethod
    def can_predict(self) -> bool:
        pass

    def add_metrics(self, metrics: dict) -> None:
        pass

    def reset(self) -> None:
        pass

    def get_feature_importance(self) -> Optional[dict[str, float]]:
        return None


class DummyPredictor(BaseGCPredictor):
    def __init__(self, fixed_value: float = 0.5):
        super().__init__()
        self._fixed_value = fixed_value
        self._is_loaded = True

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            name="dummy",
            version="1.0.0",
            description="Dummy predictor for testing",
            supports_training=False,
            requires_sequence=False,
        )

    def train(self, data_path: Path, **kwargs) -> TrainingResult:
        raise NotImplementedError("DummyPredictor does not support training")

    def save(self, path: Path) -> None:
        pass

    def load(self, path: Path) -> None:
        self._is_loaded = True

    def predict(self) -> float:
        return self._fixed_value

    def can_predict(self) -> bool:
        return True
