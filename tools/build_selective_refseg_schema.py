"""Convert a same-source existence schema into mask-or-null refseg rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils



def encode_binary_mask(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)[:, :, None]))[0]
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return {
        "format": "rle",
        "counts": encoded["counts"],
        "size": [int(mask.shape[0]), int(mask.shape[1])],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existence-schema", required=True)
    parser.add_argument("--grefs", required=True)
    parser.add_argument("--instances", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def annotation_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    segmentation = annotation.get("segmentation") or []
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
    else:
        polygons = [p for p in segmentation if len(p) >= 6 and len(p) % 2 == 0]
        if not polygons:
            return np.zeros((height, width), dtype=np.uint8)
        rle = mask_utils.frPyObjects(polygons, height, width)
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.max(axis=2)
    return (decoded > 0).astype(np.uint8)


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.existence_schema).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refs = {str(ref.get("ref_id")): ref for ref in json.loads(Path(args.grefs).read_text(encoding="utf-8"))}
    instances = json.loads(Path(args.instances).read_text(encoding="utf-8"))
    images = {str(image["id"]): image for image in instances["images"]}
    annotations = {int(ann["id"]): ann for ann in instances["annotations"]}

    output_rows = []
    for row in rows:
        no_target = "no target" in str(row["answer"]).lower()
        meta = dict(row.get("meta") or {})
        image = images[str(meta["source_image_id"])]
        height, width = int(image["height"]), int(image["width"])
        binary = np.zeros((height, width), dtype=np.uint8)
        if not no_target:
            ref = refs[str(meta["source_ref_id"])]
            ann_ids = ref.get("ann_id") or []
            if isinstance(ann_ids, int):
                ann_ids = [ann_ids]
            for ann_id in ann_ids:
                binary |= annotation_mask(annotations[int(ann_id)], height, width)
            if binary.sum() == 0:
                raise ValueError(f"Positive row has an empty mask: {row['id']}")
        output_rows.append({
            "id": row["id"].replace("existence-", "selective-refseg-"),
            "pair_id": row["pair_id"],
            "task": "refseg",
            "source": "grefcoco_selective",
            "split": row.get("split", "train"),
            "image_path": row["image_path"],
            "query": row["query"],
            "mask": encode_binary_mask(binary),
            "meta": {**meta, "no_target": no_target, "height": height, "width": width},
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(output_rows),
        "positive": sum(not row["meta"]["no_target"] for row in output_rows),
        "no_target": sum(row["meta"]["no_target"] for row in output_rows),
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
