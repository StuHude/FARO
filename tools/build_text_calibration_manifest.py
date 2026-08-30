"""Build a frozen positive/negative text-abstention calibration manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive-schema", required=True)
    parser.add_argument("--positive-jsonl")
    parser.add_argument("--negative-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=256)
    args = parser.parse_args()

    positive = []
    for row in rows(Path(args.positive_schema)):
        image = row.get("image_path") or row.get("image")
        prompt = row.get("query") or row.get("prompt")
        if not image or not prompt:
            continue
        positive.append(
            {
                "id": f"positive-{len(positive):04d}-{row.get('id', 'unknown')}",
                "image_path": image,
                "prompt": prompt,
                "answer": "Target exists.",
                "class": "positive",
                "source": row.get("source", "refseg"),
            }
        )
        if len(positive) >= args.per_class:
            break
    if len(positive) < args.per_class and args.positive_jsonl:
        for row in rows(Path(args.positive_jsonl)):
            image = row.get("target_image") or row.get("source_image") or row.get("image_path")
            category = row.get("category_name") or row.get("category") or "the described object"
            if not image:
                continue
            positive.append(
                {
                    "id": f"positive-{len(positive):04d}-{row.get('pair_id', 'gres')}",
                    "image_path": image,
                    "prompt": f"Please segment {category}.",
                    "answer": "Target exists.",
                    "class": "positive",
                    "source": row.get("source", "gres_positive"),
                }
            )
            if len(positive) >= args.per_class:
                break

    negative = []
    for row in rows(Path(args.negative_jsonl)):
        image = row.get("image_path") or row.get("image")
        prompt = row.get("prompt") or row.get("query")
        if not image or not prompt:
            continue
        negative.append(
            {
                "id": f"negative-{len(negative):04d}",
                "image_path": image,
                "prompt": prompt,
                "answer": row.get("answer", "No target."),
                "class": "negative",
                "source": row.get("source", "gres_no_target"),
            }
        )
        if len(negative) >= args.per_class:
            break

    if len(positive) < args.per_class or len(negative) < args.per_class:
        raise RuntimeError(f"insufficient rows: positive={len(positive)} negative={len(negative)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in positive + negative) + "\n", encoding="utf-8")
    print(f"wrote {output} positive={len(positive)} negative={len(negative)}")


if __name__ == "__main__":
    main()
