"""End-to-end Qwen2-VL INTCORT inference."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .model import Qwen2VLBackend
from .parsing import parse_relation, route_binary, route_unary
from .router import Qwen25Router
from .transforms import (
    HORIZONTAL,
    REVERSE,
    VERTICAL,
    horizontal_image,
    mapped,
    reverse_question,
    vertical_image,
)
from .types import BranchResult, PredictionResult


def _prompt_with_candidates(
    question: str,
    candidate_relations: list[str] | tuple[str, ...] | None,
) -> str:
    question = question.strip()
    if not candidate_relations:
        return question
    if "answer with" in question.lower() or "select from" in question.lower():
        return question
    labels = [str(label).strip() for label in candidate_relations]
    return f"{question} Answer with {', '.join(labels[:-1])} or {labels[-1]}."


class INTCORT:
    """Qwen2-VL-7B implementation with the Qwen2.5-1.5B router.

    The method uses direct inference plus independent input-side branches:
    horizontal flip for every spatial query, vertical flip for unary queries,
    and target/reference reversal for binary queries. Non-spatial queries use
    the direct branch only.
    """

    def __init__(
        self,
        vlm_path: str,
        router_path: str,
        device: str = "cuda",
        max_new_tokens: int = 48,
    ) -> None:
        self.vlm = Qwen2VLBackend(vlm_path, device=device)
        self.router = Qwen25Router(router_path, device=device)
        self.max_new_tokens = max_new_tokens

    def predict(
        self,
        image: str | Path | Image.Image,
        question: str,
        candidate_relations: list[str] | tuple[str, ...] | None = None,
    ) -> PredictionResult:
        """Run one sample without using its ground-truth answer.

        ``candidate_relations`` is used only to construct the answer prompt and
        parse spatial relation labels; it is never used to select the route.
        """

        source = image if isinstance(image, Image.Image) else Image.open(image)
        image_rgb = source.convert("RGB")
        prompt = _prompt_with_candidates(question, candidate_relations)
        route = self.router.classify(question)

        direct_text, direct_conf = self.vlm.generate(
            image_rgb, prompt, self.max_new_tokens
        )
        direct_prediction = (
            parse_relation(direct_text, candidate_relations)
            if candidate_relations
            else None
        )
        branches = {
            "direct": BranchResult(
                text=direct_text,
                prediction=direct_prediction,
                confidence=direct_conf,
                prompt=prompt,
            )
        }

        if route == "other":
            return PredictionResult(
                route=route,
                prediction=direct_prediction or direct_text,
                direct_text=direct_text,
                branches=branches,
            )

        reversed_prompt = None
        if route == "binary":
            try:
                reversed_prompt = reverse_question(prompt).text
            except ValueError:
                return PredictionResult(
                    route=route,
                    prediction=direct_prediction or direct_text,
                    direct_text=direct_text,
                    branches=branches,
                )

        horizontal_text, horizontal_conf = self.vlm.generate(
            horizontal_image(image_rgb), prompt, self.max_new_tokens
        )
        horizontal_raw = (
            parse_relation(horizontal_text, candidate_relations)
            if candidate_relations
            else None
        )
        horizontal_prediction = mapped(horizontal_raw, HORIZONTAL)
        branches["horizontal"] = BranchResult(
            text=horizontal_text,
            raw_prediction=horizontal_raw,
            prediction=horizontal_prediction,
            confidence=horizontal_conf,
            prompt=prompt,
        )

        if route == "unary":
            vertical_text, vertical_conf = self.vlm.generate(
                vertical_image(image_rgb), prompt, self.max_new_tokens
            )
            vertical_raw = (
                parse_relation(vertical_text, candidate_relations)
                if candidate_relations
                else None
            )
            vertical_prediction = mapped(vertical_raw, VERTICAL)
            branches["vertical"] = BranchResult(
                text=vertical_text,
                raw_prediction=vertical_raw,
                prediction=vertical_prediction,
                confidence=vertical_conf,
                prompt=prompt,
            )
            final_prediction = route_unary(
                direct_prediction,
                horizontal_prediction,
                vertical_prediction,
                direct_conf,
                horizontal_conf,
                vertical_conf,
            )
        else:
            reverse_text, reverse_conf = self.vlm.generate(
                image_rgb, reversed_prompt, self.max_new_tokens
            )
            reverse_raw = (
                parse_relation(reverse_text, candidate_relations)
                if candidate_relations
                else None
            )
            reverse_prediction = mapped(reverse_raw, REVERSE)
            branches["reverse"] = BranchResult(
                text=reverse_text,
                raw_prediction=reverse_raw,
                prediction=reverse_prediction,
                confidence=reverse_conf,
                prompt=reversed_prompt,
            )
            final_prediction = route_binary(
                direct_prediction,
                horizontal_prediction,
                reverse_prediction,
                direct_conf,
                horizontal_conf,
                reverse_conf,
            )

        return PredictionResult(
            route=route,
            prediction=final_prediction,
            direct_text=direct_text,
            branches=branches,
        )
