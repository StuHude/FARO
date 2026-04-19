from __future__ import annotations

import numpy as np


def compute_ciou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    union = float((pred | gt).sum())
    if union == 0.0:
        return 1.0
    inter = float((pred & gt).sum())
    return inter / union


def compute_exact_pair(pred_codes: list[int], gt_codes: list[int]) -> float:
    if len(pred_codes) != len(gt_codes):
        return 0.0
    return 1.0 if list(pred_codes) == list(gt_codes) else 0.0


def compute_seg_reward(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    pred_codes: list[int] | None = None,
    gt_codes: list[int] | None = None,
    exact_pair_weight: float = 0.2,
) -> float:
    ciou = compute_ciou(pred_mask, gt_mask)
    if pred_codes is None or gt_codes is None:
        return ciou
    return (1.0 - exact_pair_weight) * ciou + exact_pair_weight * compute_exact_pair(pred_codes, gt_codes)


def is_seg_failure(reward_or_ciou: float, threshold: float = 0.5) -> bool:
    return reward_or_ciou < threshold

