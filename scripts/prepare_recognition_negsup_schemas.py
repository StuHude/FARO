#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dlc-bench-json", required=True)
    parser.add_argument("--class-names-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out-anchor", required=True)
    parser.add_argument("--out-negsup", required=True)
    parser.add_argument("--neg-threshold", type=float, default=0.7)
    return parser.parse_args()


def wrap_mask(seg: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "rle",
        "counts": seg["counts"],
        "size": seg["size"],
    }


def main() -> None:
    args = parse_args()
    bench = json.load(open(args.dlc_bench_json, "r", encoding="utf-8"))
    class_names = json.load(open(args.class_names_json, "r", encoding="utf-8"))
    preds = json.load(open(args.pred_json, "r", encoding="utf-8"))
    eval_payload = json.load(open(args.eval_json, "r", encoding="utf-8"))
    details = eval_payload["details"]
    image_root = Path(args.image_root)

    anchor_rows = []
    negsup_rows = []
    for item in bench:
        image_path = image_root / item["image_name"]
        for sample in item["mask_samples"]:
            ann_id = str(sample["ann_id"])
            mask_obj = wrap_mask(sample["segmentation"])
            class_name = class_names.get(ann_id, sample.get("class_name", "object")).strip().lower()
            anchor_rows.append(
                {
                    "id": f"dlc_recognition_{ann_id}",
                    "task": "maskcap",
                    "source": "recognition_anchor",
                    "image_path": str(image_path),
                    "mask": mask_obj,
                    "query": None,
                    "caption": class_name,
                    "split": "train",
                    "meta": {
                        "prompt_key": "recognition_anchor",
                        "ann_id": ann_id,
                    },
                }
            )

            info = details.get(ann_id)
            if info is None:
                continue
            score_neg = info.get("score_neg")
            if score_neg is None or score_neg > args.neg_threshold:
                continue
            pred_caption = preds.get(ann_id, info.get("pred", "")).strip()
            if not pred_caption:
                continue
            negsup_rows.append(
                {
                    "id": f"dlc_negsup_{ann_id}",
                    "task": "maskcap",
                    "source": "calibration_pseudo",
                    "image_path": str(image_path),
                    "mask": mask_obj,
                    "query": None,
                    "caption": pred_caption,
                    "split": "train",
                    "meta": {
                        "prompt_key": "calibration_caption",
                        "ann_id": ann_id,
                        "score_neg": score_neg,
                        "recognition_result": info.get("recognition_result"),
                    },
                }
            )

    for out_path, rows in [
        (Path(args.out_anchor), anchor_rows),
        (Path(args.out_negsup), negsup_rows),
    ]:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
