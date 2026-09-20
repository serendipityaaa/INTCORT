"""Serializable result types used by the public inference API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BranchResult:
    text: str | None = None
    raw_prediction: str | None = None
    prediction: str | None = None
    confidence: float = -1e9
    prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionResult:
    route: str
    prediction: str | None
    direct_text: str
    branches: dict[str, BranchResult] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "prediction": self.prediction,
            "direct_text": self.direct_text,
            "branches": {
                name: branch.to_dict() for name, branch in self.branches.items()
            },
        }

