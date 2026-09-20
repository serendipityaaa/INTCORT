#!/usr/bin/env python3
"""Evaluate a JSONL manifest with the Qwen2 INTCORT implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from intcort import INTCORT


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_correct(result, row: dict) -> bool:
    answer = str(row.get("answer", row.get("gold", ""))).strip().lower()
    if result.route != "other":
        return result.prediction == answer
    return answer in result.direct_text.lower().split()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vlm-path", required=True)
    parser.add_argument("--router-path", required=True)
    parser.add_argument("--image-root", default=".")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()

    engine = INTCORT(
        vlm_path=args.vlm_path,
        router_path=args.router_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    rows = _load_rows(Path(args.manifest))
    output_rows = []
    correct = 0
    for row in tqdm(rows, desc="INTCORT"):
        image = Path(args.image_root) / row["image"]
        result = engine.predict(
            image=image,
            question=row["question"],
            candidate_relations=row.get("candidate_relations"),
        )
        item = {
            "id": row.get("id"),
            "route": result.route,
            "prediction": result.prediction,
            "answer": row.get("answer", row.get("gold")),
            "correct": _is_correct(result, row),
            "result": result.to_dict(),
        }
        correct += int(item["correct"])
        output_rows.append(item)

    payload = {
        "samples": len(output_rows),
        "accuracy": correct / max(1, len(output_rows)),
        "results": output_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: payload[key] for key in ("samples", "accuracy")}, indent=2))


if __name__ == "__main__":
    main()

