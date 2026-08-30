from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BOUNDARY_WIDTH = 2
FIFO_CAPACITY = 16
SENTINEL_SIZE = 32
FIFO_INIT_SIZE = 16
ANCHOR_BUFFER_SIZE = 64
SCHEDULE_PER_STRATUM = 80
SHUFFLE_SEED = 1907


def _square_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_or.reduce(
            [padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
             for dy in range(3) for dx in range(3)]
        )
    return result


def _square_erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool)
    for _ in range(iterations):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = np.logical_and.reduce(
            [padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
             for dy in range(3) for dx in range(3)]
        )
    return result


def mask_boundary(mask: np.ndarray, width: int = BOUNDARY_WIDTH) -> np.ndarray:
    if type(width) is not int or width < 1:
        raise ValueError("boundary width must be a positive integer")
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError("boundary masks must be two-dimensional")
    return _square_dilate(binary, width) & ~_square_erode(binary, width)


def boundary_iou(
    prediction: np.ndarray, target: np.ndarray, width: int = BOUNDARY_WIDTH
) -> float:
    predicted_boundary = mask_boundary(prediction, width)
    target_boundary = mask_boundary(target, width)
    union = np.logical_or(predicted_boundary, target_boundary).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted_boundary, target_boundary).sum() / union)


@dataclass(frozen=True)
class GeometryFeatures:
    area_ratio: float
    compactness: float
    boundary_density: float


def geometry_features(mask: np.ndarray) -> GeometryFeatures:
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or binary.size == 0:
        raise ValueError("geometry mask must be a nonempty two-dimensional array")
    area = int(binary.sum())
    if area == 0:
        raise ValueError("positive training masks must have nonzero area")
    boundary = mask_boundary(binary)
    perimeter = int(boundary.sum())
    return GeometryFeatures(
        area_ratio=float(area / binary.size),
        compactness=float(4.0 * math.pi * area / max(perimeter * perimeter, 1)),
        boundary_density=float(perimeter / max(math.sqrt(area), 1.0)),
    )


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _linear_quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("quantile values must be a finite nonempty vector")
    return float(np.quantile(array, probability, method="linear"))


def build_geometry_registry(
    rows: Sequence[dict[str, Any]], *, area_stratified: bool = False,
    boundary_stratified: bool = False,
) -> dict[str, Any]:
    # Keep this module importable on the CPU login node, which intentionally
    # lacks the Torch/pycocotools runtime used by GPU jobs.
    from .mask_codec import decode_rle_mask

    positive = [row for row in rows if not bool(row["meta"].get("no_target", False))]
    features: dict[str, GeometryFeatures] = {}
    for row in positive:
        pair_id = str(row["pair_id"])
        if pair_id in features:
            raise ValueError(f"duplicate positive pair id: {pair_id}")
        features[pair_id] = geometry_features(decode_rle_mask(row["mask"]))
    thresholds = {
        "area_ratio_q25": _linear_quantile(
            [item.area_ratio for item in features.values()], 0.25
        ),
        "compactness_q25": _linear_quantile(
            [item.compactness for item in features.values()], 0.25
        ),
        "boundary_density_q75": _linear_quantile(
            [item.boundary_density for item in features.values()], 0.75
        ),
    }
    if area_stratified:
        thresholds["area_ratio_q75"] = _linear_quantile(
            [item.area_ratio for item in features.values()], 0.75
        )
    records: dict[str, dict[str, Any]] = {}
    for pair_id, item in features.items():
        flags = {
            "small": item.area_ratio <= thresholds["area_ratio_q25"],
            "thin": item.compactness <= thresholds["compactness_q25"],
            "boundary_hard": item.boundary_density
            >= thresholds["boundary_density_q75"],
        }
        records[pair_id] = {
            **asdict(item),
            **flags,
            "hard_geometry": any(flags.values()),
        }
        if area_stratified:
            if item.area_ratio <= thresholds["area_ratio_q25"]:
                area_stratum = "small"
            elif item.area_ratio >= thresholds["area_ratio_q75"]:
                area_stratum = "large"
            else:
                area_stratum = "medium"
            records[pair_id]["area_stratum"] = area_stratum
    hard_count = sum(bool(item["hard_geometry"]) for item in records.values())
    ordinary_count = len(records) - hard_count
    if hard_count < len(records) * 0.25 or ordinary_count < len(records) * 0.25:
        raise ValueError("geometry registry needs at least 25% hard and ordinary IDs")
    payload = {
        "boundary_width": BOUNDARY_WIDTH,
        "quantile_method": "linear",
        "thresholds": thresholds,
        "records": dict(sorted(records.items())),
        "hard_count": hard_count,
        "ordinary_count": ordinary_count,
    }
    if area_stratified:
        payload["area_stratified"] = True
    if boundary_stratified:
        payload["boundary_stratified"] = True
    payload["registry_sha256"] = stable_hash(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return payload


def select_registered_ids(
    registry: dict[str, Any], *, schedule_per_stratum: int = SCHEDULE_PER_STRATUM,
    area_stratified: bool = False, boundary_stratified: bool = False,
) -> dict[str, Any]:
    records = registry["records"]
    pair_ids = sorted(records, key=stable_hash)
    sentinel_ids = pair_ids[:SENTINEL_SIZE]
    eligible = [pair_id for pair_id in pair_ids if pair_id not in set(sentinel_ids)]
    fifo_ids = eligible[:FIFO_INIT_SIZE]
    hard = [pair_id for pair_id in eligible if records[pair_id]["hard_geometry"]]
    ordinary = [pair_id for pair_id in eligible if not records[pair_id]["hard_geometry"]]
    if area_stratified:
        # Keep the hard/ordinary balance used by all previous stages, while
        # cycling through training-only target-area strata inside each arm.
        def interleave_by_area(pair_ids: list[str]) -> list[str]:
            if any("area_stratum" not in records[pair_id] for pair_id in pair_ids):
                raise ValueError("area-stratified schedule requires area-stratum registry metadata")
            groups = {
                name: sorted(
                    (pair_id for pair_id in pair_ids if records[pair_id]["area_stratum"] == name),
                    key=stable_hash,
                )
                for name in ("small", "medium", "large")
            }
            output: list[str] = []
            while len(output) < len(pair_ids):
                progressed = False
                for name in ("small", "medium", "large"):
                    if groups[name]:
                        output.append(groups[name].pop(0))
                        progressed = True
                if not progressed:
                    break
            return output

        hard = interleave_by_area(hard)
        ordinary = interleave_by_area(ordinary)
    if boundary_stratified:
        # Make the boundary-focused mixture disjoint and deterministic.  This
        # avoids double-counting thin+boundary-hard examples while preserving
        # a fixed 2 ordinary : 1 thin : 1 boundary-hard batch ratio.
        boundary_hard = [
            pair_id for pair_id in eligible if records[pair_id]["boundary_hard"]
        ]
        thin = [pair_id for pair_id in eligible if records[pair_id]["thin"]]
        # The geometry quantiles can make the two flags overlap almost
        # completely. Assign overlap by stable hash parity, rather than
        # silently dropping one stratum.
        overlap = set(thin) & set(boundary_hard)
        overlap_order = sorted(overlap, key=stable_hash)
        overlap_thin = set(overlap_order[::2])
        overlap_boundary = set(overlap_order[1::2])
        thin = [pair_id for pair_id in thin if pair_id not in overlap_boundary]
        boundary_hard = [
            pair_id for pair_id in boundary_hard if pair_id not in overlap_thin
        ]
        ordinary = [
            pair_id for pair_id in eligible
            if not records[pair_id]["thin"] and not records[pair_id]["boundary_hard"]
        ]
        boundary_hard = sorted(boundary_hard, key=stable_hash)
        thin = sorted(thin, key=stable_hash)
        ordinary = sorted(ordinary, key=stable_hash)
        if min(len(boundary_hard), len(thin), len(ordinary)) < schedule_per_stratum:
            raise ValueError(
                "boundary-stratified schedule needs at least "
                f"{schedule_per_stratum} disjoint IDs per stratum"
            )
        boundary_hard = boundary_hard[:schedule_per_stratum]
        thin = thin[:schedule_per_stratum]
        ordinary = ordinary[: 2 * schedule_per_stratum]
        batches = [
            [ordinary[2 * i], ordinary[2 * i + 1], thin[i], boundary_hard[i]]
            for i in range(schedule_per_stratum)
        ]
        schedule_ids = [pair_id for batch in batches for pair_id in batch]
        payload = {
            "sentinel_pair_ids": sentinel_ids,
            "fifo_init_pair_ids": fifo_ids,
            "batches": batches,
            "schedule_pair_ids": schedule_ids,
            "pairs_per_batch": 4,
            "ordinary_per_batch": 2,
            "thin_per_batch": 1,
            "boundary_hard_per_batch": 1,
            "boundary_stratified": True,
            "boundary_stratum_assignment": {
                **{pair_id: "thin" for pair_id in thin},
                **{pair_id: "boundary_hard" for pair_id in boundary_hard},
                **{pair_id: "ordinary" for pair_id in ordinary},
            },
        }
        payload["schedule_sha256"] = stable_hash(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        return payload
    if len(hard) < schedule_per_stratum or len(ordinary) < schedule_per_stratum:
        raise ValueError(
            f"need {schedule_per_stratum} eligible IDs per stratum, got "
            f"hard={len(hard)} ordinary={len(ordinary)}"
        )
    hard = hard[:schedule_per_stratum]
    ordinary = ordinary[:schedule_per_stratum]
    batches: list[list[str]] = []
    for offset in range(0, schedule_per_stratum, 2):
        batch = [hard[offset], ordinary[offset], hard[offset + 1], ordinary[offset + 1]]
        batches.append(batch)
    schedule_ids = [pair_id for batch in batches for pair_id in batch]
    payload = {
        "sentinel_pair_ids": sentinel_ids,
        "fifo_init_pair_ids": fifo_ids,
        "batches": batches,
        "schedule_pair_ids": schedule_ids,
        "pairs_per_batch": 4,
        "hard_per_batch": 2,
        "ordinary_per_batch": 2,
    }
    # Area metadata is an R22-only extension. Keep the legacy registry and
    # schedule payload compatible for earlier stages, whose records do not
    # carry ``area_stratum``.
    if area_stratified:
        payload["area_stratified"] = True
        payload["area_strata"] = {
            name: sum(
                records[pair_id]["area_stratum"] == name for pair_id in schedule_ids
            )
            for name in ("small", "medium", "large")
        }
    payload["schedule_sha256"] = stable_hash(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
    )
    return payload


def select_anchor_buffer_pair_ids(
    registry: dict[str, Any], schedule: dict[str, Any], *, total_rows: int = ANCHOR_BUFFER_SIZE
) -> list[str]:
    """Select a deterministic, target-present geometry-stratified train buffer.

    The registered schedule already excludes the no-target sentinel and balances
    hard/ordinary geometry.  Cycling its small/thin/boundary bins gives the
    anchor KL a fixed, disjoint training-only support without consulting a
    holdout or introducing another sampler.
    """
    if type(total_rows) is not int or total_rows < 1:
        raise ValueError("anchor buffer size must be a positive integer")
    records = registry.get("records")
    if not isinstance(records, dict):
        raise ValueError("anchor buffer requires geometry registry records")
    schedule_ids = [str(value) for value in schedule.get("schedule_pair_ids", [])]
    if not schedule_ids or len(set(schedule_ids)) != len(schedule_ids):
        raise ValueError("anchor buffer requires unique registered schedule IDs")
    bins = [
        [pair_id for pair_id in schedule_ids if records[pair_id].get("small")],
        [pair_id for pair_id in schedule_ids if records[pair_id].get("thin")],
        [pair_id for pair_id in schedule_ids if records[pair_id].get("boundary_hard")],
        [pair_id for pair_id in schedule_ids if not records[pair_id].get("hard_geometry")],
    ]
    selected: list[str] = []
    seen: set[str] = set()
    cursor = [0] * len(bins)
    while len(selected) < min(total_rows, len(schedule_ids)):
        progressed = False
        for index, candidates in enumerate(bins):
            while cursor[index] < len(candidates) and candidates[cursor[index]] in seen:
                cursor[index] += 1
            if cursor[index] < len(candidates):
                pair_id = candidates[cursor[index]]
                cursor[index] += 1
                selected.append(pair_id)
                seen.add(pair_id)
                progressed = True
                if len(selected) == total_rows:
                    break
        if not progressed:
            break
    if len(selected) < min(total_rows, len(schedule_ids)):
        selected.extend(pair_id for pair_id in schedule_ids if pair_id not in seen)
        selected = selected[:total_rows]
    if len(selected) != total_rows:
        raise ValueError(
            f"anchor buffer needs {total_rows} registered target rows, got {len(selected)}"
        )
    return selected


def shuffled_hard_flags(
    pair_ids: Sequence[str], true_flags: dict[str, bool], seed: int = SHUFFLE_SEED
) -> dict[str, bool]:
    ordered_ids = sorted(str(pair_id) for pair_id in pair_ids)
    values = [bool(true_flags[pair_id]) for pair_id in ordered_ids]
    destinations = sorted(
        ordered_ids, key=lambda pair_id: stable_hash(f"{pair_id}|seed={seed}")
    )
    return {pair_id: value for pair_id, value in zip(destinations, values)}


class FIFOEmpiricalRank:
    def __init__(self, values: Iterable[float], capacity: int = FIFO_CAPACITY) -> None:
        values = [float(np.float32(value)) for value in values]
        if type(capacity) is not int or capacity < 1:
            raise ValueError("FIFO capacity must be a positive integer")
        if len(values) != capacity or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
        ):
            raise ValueError("FIFO must start full with finite values in [0, 1]")
        self.capacity = capacity
        self._values: deque[float] = deque(values, maxlen=capacity)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(self._values)

    def midrank(self, value: float) -> float:
        scalar = float(np.float32(value))
        if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
            raise ValueError("rank value must be finite and in [0, 1]")
        snapshot = self.values
        less = sum(item < scalar for item in snapshot)
        equal = sum(item == scalar for item in snapshot)
        return float((less + 0.5 * equal) / len(snapshot))

    def append_group(self, values: Sequence[float]) -> None:
        if len(values) != 4:
            raise ValueError("FIFO updates require exactly K=4 values")
        for value in values:
            scalar = float(np.float32(value))
            if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                raise ValueError("FIFO update value must be finite and in [0, 1]")
            self._values.append(scalar)

    def sha256(self) -> str:
        return stable_hash(json.dumps(self.values, separators=(",", ":")))


def write_registry(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
