#!/usr/bin/env python3
"""Run INTCORT on one image/question pair."""

from __future__ import annotations

import argparse
import json

from intcort import INTCORT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--router-path", required=True)
    parser.add_argument("--candidate-relations", nargs="*")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    engine = INTCORT(
        vlm_path=args.vlm_path,
        router_path=args.router_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    result = engine.predict(
        image=args.image,
        question=args.question,
        candidate_relations=args.candidate_relations,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

