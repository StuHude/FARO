#!/usr/bin/env python
"""Paired statistics for two official GRefCOCO merged-output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def decode_compressed_counts(text: str) -> list[int]:
    """Decode COCO's compressed RLE counts without requiring pycocotools."""
    counts: list[int] = []
    position = 0
    while position < len(text):
        value = 0
        shift = 0
        while True:
            code = ord(text[position]) - 48
            position += 1
            value |= (code & 0x1F) << shift
            shift += 5
            if not (code & 0x20):
                if code & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError(f"Invalid negative RLE run: {value}")
        counts.append(value)
    return counts


def rle_pair_stats(left: dict, right: dict) -> tuple[int, int, int, int]:
    """Return left area, right area, intersection, and union."""
    if left["size"] != right["size"]:
        raise ValueError(f"RLE sizes differ: {left['size']} != {right['size']}")
    left_counts = decode_compressed_counts(str(left["counts"]))
    right_counts = decode_compressed_counts(str(right["counts"]))
    left_index = right_index = 0
    left_remaining = left_counts[0]
    right_remaining = right_counts[0]
    left_value = right_value = 0
    left_area = right_area = intersection = union = 0
    consumed = 0
    total = int(left["size"][0]) * int(left["size"][1])

    while left_index < len(left_counts) and right_index < len(right_counts):
        run = min(left_remaining, right_remaining)
        if run == 0:
            if left_remaining == 0:
                left_index += 1
                left_value ^= 1
                if left_index < len(left_counts):
                    left_remaining = left_counts[left_index]
            if right_remaining == 0:
                right_index += 1
                right_value ^= 1
                if right_index < len(right_counts):
                    right_remaining = right_counts[right_index]
            continue
        consumed += run
        left_area += run * left_value
        right_area += run * right_value
        intersection += run * (left_value & right_value)
        union += run * (left_value | right_value)
        left_remaining -= run
        right_remaining -= run

    if consumed != total:
        raise ValueError(f"RLE covers {consumed} pixels, expected {total}")
    return left_area, right_area, intersection, union


def load_rows(directory: Path) -> dict[str, dict]:
    files = sorted(directory.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON records in {directory}")
    return {path.name: json.loads(path.read_text(encoding="utf-8")) for path in files}


def metric_arrays(rows: dict[str, dict], names: list[str]) -> dict[str, np.ndarray]:
    gt_empty: list[bool] = []
    pred_empty: list[bool] = []
    giou: list[float] = []
    intersections: list[float] = []
    unions: list[float] = []
    for name in names:
        row = rows[name]
        gt_area, pred_area, intersection, union = rle_pair_stats(
            row["gt_masks"], row["pred_masks"]
        )
        is_gt_empty = gt_area == 0
        is_pred_empty = pred_area == 0
        gt_empty.append(is_gt_empty)
        pred_empty.append(is_pred_empty)
        giou.append(1.0 if is_gt_empty and is_pred_empty else intersection / max(union, 1))
        intersections.append(float(intersection))
        unions.append(float(union))
    return {
        "gt_empty": np.asarray(gt_empty, dtype=bool),
        "pred_empty": np.asarray(pred_empty, dtype=bool),
        "giou": np.asarray(giou, dtype=np.float64),
        "intersection": np.asarray(intersections, dtype=np.float64),
        "union": np.asarray(unions, dtype=np.float64),
    }


def summarize(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    negative = metrics["gt_empty"]
    positive = ~negative
    return {
        "N_acc": 100.0 * float(metrics["pred_empty"][negative].mean()),
        "T_acc": 100.0 * float((~metrics["pred_empty"][positive]).mean()),
        "g_iou": 100.0 * float(metrics["giou"].mean()),
        "c_iou": 100.0 * float(metrics["intersection"].sum() / metrics["union"].sum()),
    }


def percentile_report(values: np.ndarray, point: float) -> dict[str, float | list[float]]:
    return {
        "mean": float(point),
        "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
    }


def paired_bootstrap(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    repeats: int,
    seed: int,
    chunk_size: int = 128,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(left["gt_empty"])
    positive = np.flatnonzero(~left["gt_empty"])
    all_indices = np.arange(len(left["gt_empty"]))
    samples = {key: np.empty(repeats, dtype=np.float64) for key in ("N_acc", "T_acc", "g_iou", "c_iou")}

    for start in range(0, repeats, chunk_size):
        stop = min(start + chunk_size, repeats)
        count = stop - start
        neg_draw = rng.choice(negative, size=(count, len(negative)), replace=True)
        pos_draw = rng.choice(positive, size=(count, len(positive)), replace=True)
        all_draw = rng.choice(all_indices, size=(count, len(all_indices)), replace=True)
        samples["N_acc"][start:stop] = 100.0 * (
            right["pred_empty"][neg_draw].mean(axis=1)
            - left["pred_empty"][neg_draw].mean(axis=1)
        )
        samples["T_acc"][start:stop] = 100.0 * (
            (~right["pred_empty"][pos_draw]).mean(axis=1)
            - (~left["pred_empty"][pos_draw]).mean(axis=1)
        )
        samples["g_iou"][start:stop] = 100.0 * (
            right["giou"][all_draw].mean(axis=1) - left["giou"][all_draw].mean(axis=1)
        )
        left_ciou = left["intersection"][all_draw].sum(axis=1) / left["union"][all_draw].sum(axis=1)
        right_ciou = right["intersection"][all_draw].sum(axis=1) / right["union"][all_draw].sum(axis=1)
        samples["c_iou"][start:stop] = 100.0 * (right_ciou - left_ciou)

    left_summary = summarize(left)
    right_summary = summarize(right)
    return {
        key: percentile_report(samples[key], right_summary[key] - left_summary[key])
        for key in samples
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path, help="Baseline merged-output directory")
    parser.add_argument("right", type=Path, help="Candidate merged-output directory")
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_rows = load_rows(args.left)
    right_rows = load_rows(args.right)
    if set(left_rows) != set(right_rows):
        raise ValueError(
            f"Record names differ: left={len(left_rows)} right={len(right_rows)} "
            f"paired={len(set(left_rows) & set(right_rows))}"
        )
    names = sorted(left_rows)
    left = metric_arrays(left_rows, names)
    right = metric_arrays(right_rows, names)
    if not np.array_equal(left["gt_empty"], right["gt_empty"]):
        raise ValueError("Ground-truth empty/non-empty labels differ between paired outputs")

    report = {
        "num_paired": len(names),
        "num_no_target": int(left["gt_empty"].sum()),
        "num_target": int((~left["gt_empty"]).sum()),
        "left": summarize(left),
        "right": summarize(right),
        "right_minus_left": paired_bootstrap(
            left, right, repeats=args.repeats, seed=args.seed
        ),
        "bootstrap_repeats": args.repeats,
        "seed": args.seed,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
