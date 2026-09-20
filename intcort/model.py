"""Qwen2-VL-7B generation backend with relation-token confidence."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from qwen_vl_utils import process_vision_info


def generated_confidence(
    scores: tuple[torch.Tensor, ...],
    generated: torch.Tensor,
    eos_token_ids: list[int],
) -> float:
    """Return the minimum non-EOS generated-token log probability.

    This intentionally matches the existing Qwen2 implementation, which uses
    every generated non-EOS token rather than changing the score definition.
    """

    values: list[float] = []
    eos = set(eos_token_ids)
    for index, logits in enumerate(scores):
        token = int(generated[0, index])
        if token in eos:
            break
        values.append(float(F.log_softmax(logits[0].float(), dim=-1)[token]))
    return min(values) if values else -1e9


class Qwen2VLBackend:
    """Frozen Qwen2-VL-7B model used for all inference branches."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).eval().to(self.device)
        eos = self.processor.tokenizer.eos_token_id
        self.eos_token_ids = eos if isinstance(eos, list) else [eos]

    def _build_inputs(self, image: Image.Image, prompt: str) -> Any:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages, image_patch_size=14)
        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 48,
    ) -> tuple[str, float]:
        inputs = self._build_inputs(image, prompt)
        output = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated = output.sequences[:, inputs.input_ids.shape[1] :]
        text = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        confidence = generated_confidence(
            output.scores,
            generated,
            self.eos_token_ids,
        )
        return text, confidence

