#!/usr/bin/env python3
"""Reject training submissions that violate the registered minimum budget."""

from __future__ import annotations

import argparse
import math
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SA2VA_ROOT = ROOT / "Sa2VA"
for path in (SA2VA_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _nonempty_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def validate(config_path: Path, data_path: Path, *, min_rows: int = 5000, min_steps: int = 10) -> dict[str, int]:
    if not config_path.is_file():
        raise ValueError(f"missing config: {config_path}")
    if not data_path.is_file():
        raise ValueError(f"missing data: {data_path}")
    try:
        namespace = runpy.run_path(str(config_path))
        config = namespace["config"]
    except Exception as exc:
        raise ValueError(f"unable to load config: {config_path}: {exc}") from exc
    configured_rows = int((config.get("data") or {}).get("expected_rows", 0) or 0)
    configured_steps = int((config.get("optimizer") or {}).get("max_steps", 0) or 0)
    data_config = config.get("data") or {}
    runtime_config = config.get("runtime") or {}
    method_config = config.get("tail_gppo") or {}
    actual_rows = _nonempty_rows(data_path)
    if actual_rows < min_rows:
        raise ValueError(f"training requires at least {min_rows} rows, got {actual_rows}")
    if configured_rows < min_rows:
        raise ValueError(f"config expected_rows must be at least {min_rows}, got {configured_rows}")
    if configured_steps < min_steps:
        raise ValueError(f"config optimizer.max_steps must be at least {min_steps}, got {configured_steps}")
    result = {
        "actual_rows": actual_rows,
        "configured_rows": configured_rows,
        "configured_steps": configured_steps,
    }
    if method_config.get("full_data_schedule") is True:
        if actual_rows != configured_rows:
            raise ValueError(
                "full-data schedule requires the dataset row count to equal "
                f"config expected_rows ({configured_rows}), got {actual_rows}"
            )
        if actual_rows % 2:
            raise ValueError("full-data schedule requires an even row count (positive/no-target pairs)")
        pairs_per_batch = int(data_config.get("pairs_per_device_batch", 0) or 0)
        world_size = int(runtime_config.get("expected_world_size", 0) or 0)
        if pairs_per_batch < 1 or world_size < 1:
            raise ValueError(
                "full-data schedule requires positive pairs_per_device_batch "
                "and runtime.expected_world_size"
            )
        minimum_pairs = int(
            method_config.get("minimum_consumed_pairs", actual_rows // 2) or 0
        )
        minimum_rows = int(
            method_config.get("minimum_consumed_rows", minimum_pairs * 2) or 0
        )
        if minimum_rows > actual_rows or minimum_pairs * 2 > actual_rows:
            raise ValueError(
                "full-data schedule minimum coverage exceeds the available dataset"
            )
        # Registered batches are partitioned across processes by Accelerate;
        # this is the lower bound for one complete pass over all pair IDs.
        required_global_batches = math.ceil(minimum_pairs / pairs_per_batch)
        required_local_steps = math.ceil(required_global_batches / world_size)
        if configured_steps < max(min_steps, required_local_steps):
            raise ValueError(
                "full-data schedule max_steps is too small for the required pair "
                f"coverage: need at least {required_local_steps}, got {configured_steps}"
            )
        result.update(
            {
                "full_data_required_local_steps": required_local_steps,
                "full_data_minimum_rows": minimum_rows,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.config, args.data)
    except ValueError as exc:
        parser.error(str(exc))
    print("training_budget_ok " + " ".join(f"{k}={v}" for k, v in result.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
