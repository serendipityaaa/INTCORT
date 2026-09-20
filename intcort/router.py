"""Qwen2.5-1.5B question-structure router."""

from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROUTER_PROMPT = """Classify only the structure of the question. Output exactly one label: UNARY, BINARY, or OTHER.
UNARY means one object is located relative to the image or photo itself.
BINARY means one object is located relative to a second named object.
OTHER means counting, color, identity, OCR, existence, or another non-spatial task.
Do not answer the question. Ignore the image and answer choices.
Examples:
Q: Where is the dog in the photograph? A: UNARY
Q: Where is a red ball in the picture? A: UNARY
Q: Where is the dog in relation to a cat? A: BINARY
Q: Where is a bicycle with respect to a car? A: BINARY
Q: How many apples are visible? A: OTHER
Q: What color is the apple? A: OTHER
Now classify:
Q: """

SYSTEM_PROMPT = "You are a strict question classifier."
_LABEL_PATTERN = re.compile(r"\b(UNARY|BINARY|OTHER)\b", re.IGNORECASE)


class Qwen25Router:
    """Frozen three-way router used before Qwen2-VL inference."""

    def __init__(self, model_path: str, device: str = "cuda") -> None:
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(self.device).eval()

    def _prompt(self, question: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ROUTER_PROMPT + question.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @staticmethod
    def parse(output: str) -> str:
        match = _LABEL_PATTERN.search(output.upper())
        return match.group(1).lower() if match else "other"

    @torch.inference_mode()
    def classify(self, question: str) -> str:
        encoded = self.tokenizer(
            [self._prompt(question)],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        generated = self.model.generate(
            **encoded,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        continuation = generated[:, encoded.input_ids.shape[1] :]
        output = self.tokenizer.decode(
            continuation[0], skip_special_tokens=True
        ).strip()
        return self.parse(output)

