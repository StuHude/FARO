from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from projects.pixvl_idea1.datasets.mask_codec import SAMTokMaskCodec
from projects.pixvl_idea1.trainers.common import (
    build_model_bundle,
    build_prompt_and_answer_ids,
    generate_answer,
    load_config,
    move_inputs_to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--parquet-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    return parser.parse_args()


def mask_to_bbox_xyxy(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def bbox_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    model, processor = build_model_bundle(cfg, trainable=False, adapter_path=args.adapter_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    codec = SAMTokMaskCodec(
        model_name_or_path=cfg["model"]["processor_name_or_path"],
        mask_tokenizer_path=cfg["model"]["mask_tokenizer_path"],
        sam2_ckpt_path=cfg["model"]["sam2_ckpt_path"],
        device=device,
    )

    table = pq.read_table(args.parquet_file)
    rows = table.to_pylist()
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    results = []
    for idx, row in enumerate(rows):
        image_bytes = row["image"]["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        query = row["normal_caption"].strip().lower()
        prompt_text = cfg["data"]["prompts"]["refseg"].format(query=query)
        prompt_inputs, _ = build_prompt_and_answer_ids(processor, image, prompt_text, "")
        prompt_inputs = move_inputs_to_device(prompt_inputs, next(model.parameters()).device)
        _, sample_text = generate_answer(model, processor, prompt_inputs, cfg["generation"]["refseg"])
        pred_codes = codec.text_to_codes(sample_text)
        pred_mask = codec.decode_codes(image, pred_codes)
        pred_box = mask_to_bbox_xyxy(pred_mask)
        gt_box = [float(v) for v in row["solution"]]
        iou = bbox_iou(pred_box, gt_box)
        results.append(
            {
                "id": int(row["row_idx"]),
                "file_name": row["file_name"],
                "query": query,
                "bbox_iou": iou,
                "pred_box": pred_box,
                "gt_box": gt_box,
            }
        )

    payload = {
        "num_samples": len(results),
        "mean_bbox_iou": sum(item["bbox_iou"] for item in results) / max(len(results), 1),
        "acc50": sum(item["bbox_iou"] >= 0.5 for item in results) / max(len(results), 1),
        "acc75": sum(item["bbox_iou"] >= 0.75 for item in results) / max(len(results), 1),
        "acc90": sum(item["bbox_iou"] >= 0.9 for item in results) / max(len(results), 1),
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
