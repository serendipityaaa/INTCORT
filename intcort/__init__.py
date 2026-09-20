"""Public API for the Qwen2-VL INTCORT implementation."""

from .types import PredictionResult

__all__ = ["INTCORT", "PredictionResult"]


def __getattr__(name: str):
    if name == "INTCORT":
        from .inference import INTCORT

        return INTCORT
    raise AttributeError(name)
