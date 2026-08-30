from __future__ import annotations

import numpy as np


def ciou(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def area_similarity(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_area = float(np.asarray(prediction).astype(bool).sum())
    target_area = float(np.asarray(target).astype(bool).sum())
    if pred_area == 0.0 and target_area == 0.0:
        return 1.0
    if pred_area == 0.0 or target_area == 0.0:
        return 0.0
    return float(min(pred_area, target_area) / max(pred_area, target_area))


def geometry_reward(prediction: np.ndarray, target: np.ndarray, *, area_weight: float = 0.2) -> float:
    area_weight = float(area_weight)
    if not 0.0 <= area_weight <= 1.0:
        raise ValueError("area_weight must be in [0, 1]")
    return (1.0 - area_weight) * ciou(prediction, target) + area_weight * area_similarity(prediction, target)
