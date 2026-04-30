from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils

from projects.pixvl_idea1.datasets.schema import encode_binary_mask, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grefs-json", required=True)
    parser.add_argument("--instances-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle)
    else:
        polygons = [poly for poly in segmentation if len(poly) >= 6 and len(poly) % 2 == 0]
        if not polygons:
            return np.zeros((height, width), dtype=np.uint8)
        rle = mask_utils.frPyObjects(polygons, height, width)
        mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.sum(axis=2)
    return (mask > 0).astype(np.uint8)


def merge_ann_masks(ann_ids: list[int], anns: dict[int, dict[str, Any]], height: int, width: int) -> np.ndarray:
    if not ann_ids or ann_ids == [-1]:
        return np.zeros((height, width), dtype=np.uint8)
    masks = [decode_segmentation(anns[ann_id]["segmentation"], height, width) for ann_id in ann_ids]
    merged = np.sum(masks, axis=0)
    return (merged > 0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    refs = json.loads(Path(args.grefs_json).read_text(encoding="utf-8"))
    instances = json.loads(Path(args.instances_json).read_text(encoding="utf-8"))
    anns = {ann["id"]: ann for ann in instances["annotations"]}
    imgs = {img["id"]: img for img in instances["images"]}
    image_root = Path(args.image_root)

    rows: list[dict[str, Any]] = []
    for ref in refs:
        if ref.get("split") != args.split:
            continue
        image = imgs[ref["image_id"]]
        ann_ids = ref["ann_id"] if isinstance(ref["ann_id"], list) else [ref["ann_id"]]
        mask = merge_ann_masks(ann_ids, anns, image["height"], image["width"])
        image_path = image_root / image["file_name"]
        mask_rle = encode_binary_mask(mask)
        for sent in ref["sentences"]:
            rows.append(
                {
                    "id": f"grefcoco_{args.split}_{ref['ref_id']}_{sent['sent_id']}",
                    "task": "refseg",
                    "source": "grefcoco",
                    "image_path": str(image_path),
                    "mask": mask_rle,
                    "query": sent["sent"].strip().lower(),
                    "caption": None,
                    "split": args.split,
                    "meta": {
                        "dataset_name": "grefcoco",
                        "ref_id": ref["ref_id"],
                        "sent_id": sent["sent_id"],
                        "image_id": ref["image_id"],
                        "width": image["width"],
                        "height": image["height"],
                    },
                }
            )

    write_jsonl(args.output, rows)
    print(f"{args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
