from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
from pycocotools import mask as mask_utils

from .schema import encode_binary_mask


SPLIT_FILE = {
    "refcoco": "refs(unc).p",
    "refcoco+": "refs(unc).p",
    "refcocog": "refs(umd).p",
}


def _decode_annotation_mask(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle)
    else:
        polygons = [poly for poly in segmentation if len(poly) >= 6 and len(poly) % 2 == 0]
        if len(polygons) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        rle = mask_utils.frPyObjects(polygons, height, width)
        mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask.sum(axis=2)
    return (mask > 0).astype(np.uint8)


def iter_refcoco_records(data_root: str, dataset_name: str, split: str) -> Iterator[dict]:
    dataset_dir = Path(data_root) / dataset_name
    split_path = dataset_dir / SPLIT_FILE[dataset_name]
    instances_path = dataset_dir / "instances.json"
    if not split_path.exists() or not instances_path.exists():
        raise FileNotFoundError(f"Missing files for {dataset_name}: {split_path} / {instances_path}")

    refs = pickle.loads(split_path.read_bytes())
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    anns = {ann["id"]: ann for ann in instances["annotations"]}
    imgs = {img["id"]: img for img in instances["images"]}

    for ref in refs:
        if ref.get("split") != split:
            continue
        image = imgs[ref["image_id"]]
        ann = anns[ref["ann_id"]]
        image_path = dataset_dir / "coco2014" / "train2014" / image["file_name"]
        mask = _decode_annotation_mask(ann["segmentation"], image["height"], image["width"])
        rle = encode_binary_mask(mask)
        for sent in ref["sentences"]:
            yield {
                "id": f"{dataset_name}_{ref['ref_id']}_{sent['sent_id']}",
                "task": "refseg",
                "source": dataset_name.replace("+", "plus"),
                "image_path": str(image_path),
                "mask": rle,
                "query": sent["sent"].strip().lower(),
                "caption": None,
                "split": split,
                "meta": {
                    "dataset_name": dataset_name,
                    "ref_id": ref["ref_id"],
                    "sent_id": sent["sent_id"],
                    "width": image["width"],
                    "height": image["height"],
                },
            }
