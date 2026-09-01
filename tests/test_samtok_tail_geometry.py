from __future__ import annotations

import numpy as np
import pytest

from projects.samtok_selective.tail_geometry import (
    FIFOEmpiricalRank,
    boundary_iou,
    geometry_features,
    mask_boundary,
    select_registered_ids,
    shuffled_hard_flags,
)


def test_boundary_iou_empty_full_and_shifted_masks():
    empty = np.zeros((9, 9), dtype=np.uint8)
    full = np.ones((9, 9), dtype=np.uint8)
    point = empty.copy()
    point[4, 4] = 1
    shifted = empty.copy()
    shifted[4, 5] = 1
    assert boundary_iou(empty, empty) == 1.0
    assert boundary_iou(empty, full) == 0.0
    assert boundary_iou(point, point) == 1.0
    assert 0.0 < boundary_iou(point, shifted) < 1.0
    assert mask_boundary(point).dtype == np.bool_


def test_geometry_features_reject_empty_positive():
    with pytest.raises(ValueError, match="nonzero area"):
        geometry_features(np.zeros((4, 4), dtype=np.uint8))
    features = geometry_features(np.eye(8, dtype=np.uint8))
    assert 0.0 < features.area_ratio <= 1.0
    assert features.compactness > 0.0
    assert features.boundary_density > 0.0


def test_fifo_midrank_freezes_group_then_evicts_four():
    queue = FIFOEmpiricalRank([index / 15 for index in range(16)])
    before = queue.values
    ranks = [queue.midrank(value) for value in (0.0, 0.5, 1.0, 0.5)]
    assert queue.values == before
    assert ranks[0] == pytest.approx(0.03125)
    assert ranks[2] == pytest.approx(0.96875)
    queue.append_group((0.2, 0.3, 0.4, 0.5))
    assert len(queue.values) == 16
    assert queue.values[:12] == before[4:]
    assert queue.values[-4:] == pytest.approx((0.2, 0.3, 0.4, 0.5))


def test_shuffled_flags_preserve_counts_and_are_deterministic():
    pair_ids = [f"pair-{index}" for index in range(16)]
    flags = {pair_id: index < 8 for index, pair_id in enumerate(pair_ids)}
    first = shuffled_hard_flags(pair_ids, flags)
    second = shuffled_hard_flags(pair_ids, flags)
    assert first == second
    assert sum(first.values()) == sum(flags.values()) == 8
    assert first != flags


def test_full_data_schedule_covers_every_pair_including_sentinel():
    records = {
        f"pair-{index:04d}": {"hard_geometry": index % 2 == 0}
        for index in range(2560)
    }
    schedule = select_registered_ids({"records": records}, include_sentinel=True)
    assert schedule["full_data_schedule"] is True
    assert schedule["scheduled_pair_count"] == 2560
    assert schedule["scheduled_row_count"] == 5120
    assert len(schedule["batches"]) == 640
    assert len(set(schedule["schedule_pair_ids"])) == 2560
    assert set(schedule["sentinel_pair_ids"]).issubset(schedule["schedule_pair_ids"])
