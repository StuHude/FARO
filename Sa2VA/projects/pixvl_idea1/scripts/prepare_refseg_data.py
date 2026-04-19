#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from projects.pixvl_idea1.datasets.adapters_refcoco import iter_refcoco_records
from projects.pixvl_idea1.datasets.schema import encode_binary_mask, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/sa2va_training/ref_seg",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    full_rows_written = False

    for split in ("train", "val", "testA", "testB", "test"):
        rows = []
        for dataset_name in ("refcoco", "refcoco+", "refcocog"):
            try:
                rows.extend(iter_refcoco_records(args.data_root, dataset_name, split))
            except Exception:
                continue
        if rows:
            write_jsonl(output_root / f"refseg_{split}.jsonl", rows)
            print(f"{split}: {len(rows)} records")
            full_rows_written = True

    if full_rows_written:
        return

    example_root = data_root.parent.parent / "sa2va_finetune_example"
    annotations_path = example_root / "my_data" / "annotations.json"
    images_root = example_root / "my_data" / "images"
    if not annotations_path.exists():
        print("No full refseg data and no finetune example found.")
        return

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    rows = []
    cap_rows = []
    for item_idx, item in enumerate(annotations):
        image_path = images_root / item["image"]
        for mask_idx, (mask_polys, text) in enumerate(zip(item["mask"], item["text"])):
            width = height = None
            # infer size lazily from polygons if possible; actual decode uses image size later if needed
            from PIL import Image
            with Image.open(image_path) as image:
                width, height = image.size
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in mask_polys:
                rles = mask_utils.frPyObjects([seg], height, width)
                binary_mask |= mask_utils.decode(rles).astype(np.uint8).squeeze()
            rows.append(
                {
                    "id": f"example_{item_idx}_{mask_idx}",
                    "task": "refseg",
                    "source": "sa2va_finetune_example",
                    "image_path": str(image_path),
                    "mask": encode_binary_mask(binary_mask),
                    "query": str(text).strip().lower(),
                    "caption": None,
                    "split": "train",
                    "meta": {
                        "dataset_name": "sa2va_finetune_example",
                        "width": width,
                        "height": height,
                    },
                }
            )
            cap_rows.append(
                {
                    "id": f"example_cap_{item_idx}_{mask_idx}",
                    "task": "maskcap",
                    "source": "sa2va_finetune_example",
                    "image_path": str(image_path),
                    "mask": encode_binary_mask(binary_mask),
                    "query": None,
                    "caption": str(text).strip().lower(),
                    "split": "train",
                    "meta": {
                        "dataset_name": "sa2va_finetune_example",
                        "width": width,
                        "height": height,
                    },
                }
            )
    write_jsonl(output_root / "refseg_train.jsonl", rows)
    write_jsonl(output_root / "refseg_val.jsonl", rows[: min(len(rows), 32)])
    write_jsonl(output_root / "dam_train.jsonl", cap_rows)
    print(f"fallback example train: {len(rows)} records")


if __name__ == "__main__":
    main()
