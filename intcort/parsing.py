"""Relation extraction and confidence-weighted voting."""

from __future__ import annotations

import collections
import math
import re
from collections.abc import Iterable, Sequence


PATTERNS = {
    "left": r"\bleft\b",
    "right": r"\bright\b",
    "top": r"\b(?:top|above|upper)\b",
    "bottom": r"\b(?:bottom|below|lower)\b",
    "above": r"\babove\b",
    "below": r"\b(?:below|under|underneath)\b",
    "on": r"\bon(?:\s+top\s+of)?\b(?!\s+the\s+(?:left|right))",
    "under": r"\b(?:under|underneath)\b",
    "front": r"\b(?:front|in\s+front|foreground)\b",
    "behind": r"\b(?:behind|background|rear)\b",
}


def _search_text(text: str) -> str:
    low = text.lower().strip()
    final = re.findall(r"(?:final\s+answer|answer)\s*[:：]\s*([^\n]+)", low)
    return final[-1] if final else low


def parse_relation(
    text: str,
    candidate_relations: Sequence[str],
) -> str | None:
    """Extract one candidate relation from a generated response.

    Candidate order is respected only for deciding which patterns are
    available; the latest textual hit is returned, matching the existing
    benchmark parser behavior.
    """

    search = _search_text(text)
    hits: list[tuple[int, str]] = []
    for label in candidate_relations:
        normalized = label.strip().lower()
        pattern = PATTERNS.get(normalized, rf"\b{re.escape(normalized)}\b")
        for match in re.finditer(pattern, search):
            hits.append((match.start(), normalized))
    return max(hits)[1] if hits else None


def confidence_vote(
    predictions: Iterable[str | None],
    confidences: Iterable[float],
    fallback: str | None,
) -> str | None:
    """Accumulate exp(clipped log-confidence) for each relation label."""

    scores: collections.defaultdict[str, float] = collections.defaultdict(float)
    for prediction, confidence in zip(predictions, confidences):
        if prediction is not None:
            scores[prediction] += math.exp(max(-20.0, float(confidence)))
    return max(scores, key=scores.get) if scores else fallback


def route_unary(
    direct: str | None,
    horizontal: str | None,
    vertical: str | None,
    direct_conf: float,
    horizontal_conf: float,
    vertical_conf: float,
) -> str | None:
    """Apply the original unary routing rules without modification."""

    if direct is None:
        return confidence_vote(
            [direct, horizontal, vertical],
            [direct_conf, horizontal_conf, vertical_conf],
            None,
        )
    if direct in ("left", "right"):
        return direct
    if direct in ("top", "bottom"):
        return horizontal
    if direct in ("front", "behind"):
        return horizontal if horizontal_conf >= vertical_conf else vertical
    return confidence_vote(
        [direct, horizontal, vertical],
        [direct_conf, horizontal_conf, vertical_conf],
        direct,
    )


def route_binary(
    direct: str | None,
    horizontal: str | None,
    reverse: str | None,
    direct_conf: float,
    horizontal_conf: float,
    reverse_conf: float,
) -> str | None:
    """Apply the original binary routing rules without modification."""

    if direct in ("front", "behind") and reverse is not None:
        return reverse
    return confidence_vote(
        [direct, horizontal, reverse],
        [direct_conf, horizontal_conf, reverse_conf],
        direct,
    )

