from typing import Dict, Type

from neurogc.models.base import (
    BaseGCPredictor,
    DummyPredictor,
    TrainingResult,
)

_registry: Dict[str, Type[BaseGCPredictor]] = {}


def register_model(name: str):
    def decorator(cls: Type[BaseGCPredictor]) -> Type[BaseGCPredictor]:
        if not issubclass(cls, BaseGCPredictor):
            raise TypeError(f"{cls.__name__} must inherit from BaseGCPredictor")
        _registry[name] = cls
        return cls

    return decorator


def get_model(name: str) -> Type[BaseGCPredictor]:
    if name not in _registry:
        available = ", ".join(_registry.keys()) or "none"
        raise ValueError(
            f"Unknown model: '{name}'. Available models: {available}"
        )
    return _registry[name]


def list_models() -> list[str]:
    return list(_registry.keys())


register_model("dummy")(DummyPredictor)

# ruff: noqa: E402
from neurogc.models.classical import ClassicalPredictor
from neurogc.models.feedforward import FeedforwardPredictor
from neurogc.models.lstm import LSTMPredictor
from neurogc.models.transformer import TransformerPredictor

__all__ = [
    "register_model",
    "get_model",
    "list_models",
    "BaseGCPredictor",
    "TrainingResult",
    "LSTMPredictor",
    "TransformerPredictor",
    "FeedforwardPredictor",
    "ClassicalPredictor",
    "DummyPredictor",
]
