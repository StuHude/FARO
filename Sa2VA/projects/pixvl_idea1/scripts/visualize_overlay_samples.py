#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from projects.pixvl_idea1.datasets.overlay_utils import build_overlay_image
from projects.pixvl_idea1.datasets.schema import decode_rle_mask, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.schema_file)[: args.num_samples]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows):
        image = Image.open(row["image_path"]).convert("RGB")
        mask = decode_rle_mask(row["mask"])
        overlay = build_overlay_image(image, mask)
        overlay.save(output_dir / f"overlay_{idx:02d}.jpg")


if __name__ == "__main__":
    main()

