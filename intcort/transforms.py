"""Input transformations and relation-space inverse mappings."""

from __future__ import annotations

import re
from dataclasses import dataclass

from PIL import Image, ImageOps


HORIZONTAL = {"left": "right", "right": "left"}
VERTICAL = {
    "top": "bottom",
    "bottom": "top",
    "above": "below",
    "below": "above",
    "on": "under",
    "under": "on",
}
REVERSE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "front": "behind",
    "behind": "front",
    "on": "under",
    "under": "on",
}


@dataclass(frozen=True)
class ReversedQuestion:
    """A query after swapping its target and reference entities."""

    text: str


def mapped(prediction: str | None, mapping: dict[str, str]) -> str | None:
    """Map a transformed prediction into the original relation space."""

    if prediction is None:
        return None
    return mapping.get(prediction, prediction)


def reverse_question(question: str) -> ReversedQuestion:
    """Swap the two entities in the supported spatial-query templates.

    The two templates are the ones used by the existing Qwen2 benchmark
    evaluator. Keeping the suffix intact preserves answer-choice instructions.
    """

    match = re.fullmatch(
        r"Where\s+(is|are)\s+(.+?)\s+in relation to\s+(.+?)\?(.*)",
        question.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        auxiliary, subject, reference, suffix = match.groups()
        return ReversedQuestion(
            f"Where {auxiliary} {reference.strip()} in relation to "
            f"{subject.strip()}?{suffix}"
        )

    match = re.fullmatch(
        r"(.*?\bwhere\s+)(is|are)\s+(.+?)\s+located with respect to\s+(.+?)(\?\s*Select.*)",
        question.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if match:
        lead, auxiliary, subject, reference, suffix = match.groups()
        return ReversedQuestion(
            f"{lead}{auxiliary} {reference.strip()} located with respect to "
            f"{subject.strip()}{suffix}"
        )

    raise ValueError(f"cannot reverse spatial question: {question!r}")


def horizontal_image(image: Image.Image) -> Image.Image:
    return ImageOps.mirror(image)


def vertical_image(image: Image.Image) -> Image.Image:
    return ImageOps.flip(image)

