"""Build a deterministic image-shuffle control for selective grounding.

Queries, labels, masks, and metadata remain attached to their original rows.
Only the visual input is replaced by a deranged image and resized to the
original row's spatial dimensions so existing mask-or-null evaluation remains
shape compatible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def derangement(size: int, seed: int) -> list[int]:
    if size < 2:
        raise ValueError("image-shuffle control requires at least two rows")
    rng = random.Random(seed)
    indices = list(range(size))
    for _ in range(10_000):
        rng.shuffle(indices)
        if all(source != target for target, source in enumerate(indices)):
            return indices
    raise RuntimeError("failed to construct a derangement")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    image_dir = Path(args.image_dir)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    permutation = derangement(len(rows), args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    shuffled_rows = []
    for target_index, source_index in enumerate(permutation):
        row = dict(rows[target_index])
        source = rows[source_index]
        height, width = (int(value) for value in row["mask"]["size"])
        image_output = image_dir / f"{target_index:05d}.jpg"
        with Image.open(source["image_path"]) as image:
            image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC).save(
                image_output,
                format="JPEG",
                quality=90,
            )
        meta = dict(row.get("meta") or {})
        meta.update({
            "image_shuffle": True,
            "image_shuffle_seed": args.seed,
            "original_image_path": row["image_path"],
            "shuffled_from_image_path": source["image_path"],
            "shuffled_from_no_target": bool((source.get("meta") or {}).get("no_target", False)),
        })
        row["image_path"] = str(image_output)
        row["meta"] = meta
        shuffled_rows.append(row)

    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in shuffled_rows) + "\n",
        encoding="utf-8",
    )
    report = {
        "rows": len(shuffled_rows),
        "seed": args.seed,
        "fixed_points": sum(i == source for i, source in enumerate(permutation)),
        "label_preserved": all(
            bool((before.get("meta") or {}).get("no_target", False))
            == bool((after.get("meta") or {}).get("no_target", False))
            for before, after in zip(rows, shuffled_rows)
        ),
        "source_label_matches": sum(
            bool((rows[i].get("meta") or {}).get("no_target", False))
            == bool((rows[source].get("meta") or {}).get("no_target", False))
            for i, source in enumerate(permutation)
        ),
        "output": str(output_path),
        "image_dir": str(image_dir),
    }
    output_path.with_suffix(output_path.suffix + ".audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
