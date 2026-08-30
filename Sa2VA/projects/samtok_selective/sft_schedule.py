from __future__ import annotations

from typing import Any


def validate_updates_per_batch(value: Any) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("updates_per_batch must be a positive integer")
    return value


def batch_update_indices(step: int, max_steps: int, updates_per_batch: int) -> range:
    updates_per_batch = validate_updates_per_batch(updates_per_batch)
    if step < 0 or max_steps < 1 or step > max_steps:
        raise ValueError("Invalid SFT optimizer step bounds")
    return range(min(updates_per_batch, max_steps - step))
