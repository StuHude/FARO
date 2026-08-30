#!/usr/bin/env python
"""Paired statistics for two official RefCOCO prediction directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_official_grefcoco_pair import decode_compressed_counts, rle_pair_stats


def rle_bbox(rle: dict) -> tuple[float, float, float, float]:
    """Match torchvision ``masks_to_boxes`` without materializing the mask."""
    height = int(rle["size"][0])
    offset = 0
    min_x = min_y = None
    max_x = max_y = None
    for index, length in enumerate(decode_compressed_counts(str(rle["counts"]))):
        if index % 2 and length:
            start = offset
            end = offset + length - 1
            start_x, start_y = divmod(start, height)
            end_x, end_y = divmod(end, height)
            run_min_y, run_max_y = (start_y, end_y) if start_x == end_x else (0, height - 1)
            min_x = start_x if min_x is None else min(min_x, start_x)
            max_x = end_x if max_x is None else max(max_x, end_x)
            min_y = run_min_y if min_y is None else min(min_y, run_min_y)
            max_y = run_max_y if max_y is None else max(max_y, run_max_y)
        offset += length
    if min_x is None:
        return (0.0, 0.0, 0.0, 0.0)
    return (float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y))


def bbox_iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1, y1, width1, height1 = left
    x2, y2, width2, height2 = right
    intersection_width = max(0.0, min(x1 + width1, x2 + width2) - max(x1, x2))
    intersection_height = max(0.0, min(y1 + height1, y2 + height2) - max(y1, y2))
    intersection = intersection_width * intersection_height
    union = width1 * height1 + width2 * height2 - intersection
    return intersection / union if union else 0.0


def load_metrics(directory: Path) -> dict[str, tuple[float, float, float]]:
    rows: dict[str, tuple[float, float, float]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        gt_masks = payload.get("gt_masks")
        pred_masks = payload.get("prediction_masks")
        if not isinstance(gt_masks, list) or len(gt_masks) != 1:
            raise ValueError(f"{path}: expected exactly one ground-truth mask")
        if not isinstance(pred_masks, list) or len(pred_masks) != 1:
            raise ValueError(f"{path}: expected exactly one prediction mask")
        _, _, intersection, union = rle_pair_stats(gt_masks[0], pred_masks[0])
        rows[path.name] = (
            bbox_iou(rle_bbox(gt_masks[0]), rle_bbox(pred_masks[0])),
            float(intersection / max(union, 1)),
        )
    if not rows:
        raise FileNotFoundError(f"No JSON records in {directory}")
    return rows


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "ap50": float((values[:, 0] >= 0.5).mean()),
        "ciou": float(values[:, 1].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_rows = load_metrics(args.left)
    right_rows = load_metrics(args.right)
    if set(left_rows) != set(right_rows):
        raise ValueError(
            f"Record names differ: left={len(left_rows)} right={len(right_rows)} "
            f"paired={len(set(left_rows) & set(right_rows))}"
        )
    names = sorted(left_rows)
    left = np.asarray([left_rows[name] for name in names], dtype=np.float64)
    right = np.asarray([right_rows[name] for name in names], dtype=np.float64)
    left_summary = summarize(left)
    right_summary = summarize(right)

    rng = np.random.default_rng(args.seed)
    ap50_delta = np.empty(args.repeats, dtype=np.float64)
    ciou_delta = np.empty(args.repeats, dtype=np.float64)
    for start in range(0, args.repeats, args.chunk_size):
        stop = min(start + args.chunk_size, args.repeats)
        draw = rng.integers(0, len(names), size=(stop - start, len(names)))
        ap50_delta[start:stop] = (
            (right[draw, 0] >= 0.5).mean(axis=1)
            - (left[draw, 0] >= 0.5).mean(axis=1)
        )
        ciou_delta[start:stop] = (
            right[draw, 1].mean(axis=1) - left[draw, 1].mean(axis=1)
        )

    report = {
        "num_paired": len(names),
        "left": left_summary,
        "right": right_summary,
        "right_minus_left": {
            "ap50": {
                "mean": right_summary["ap50"] - left_summary["ap50"],
                "ci95": [float(np.quantile(ap50_delta, 0.025)), float(np.quantile(ap50_delta, 0.975))],
            },
            "ciou": {
                "mean": right_summary["ciou"] - left_summary["ciou"],
                "ci95": [float(np.quantile(ciou_delta, 0.025)), float(np.quantile(ciou_delta, 0.975))],
            },
        },
        "bootstrap_repeats": args.repeats,
        "seed": args.seed,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
