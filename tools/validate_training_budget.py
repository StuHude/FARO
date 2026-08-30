#!/usr/bin/env python3
"""Reject training submissions that violate the registered minimum budget."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


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
    actual_rows = _nonempty_rows(data_path)
    if actual_rows < min_rows:
        raise ValueError(f"training requires at least {min_rows} rows, got {actual_rows}")
    if configured_rows < min_rows:
        raise ValueError(f"config expected_rows must be at least {min_rows}, got {configured_rows}")
    if configured_steps < min_steps:
        raise ValueError(f"config optimizer.max_steps must be at least {min_steps}, got {configured_steps}")
    return {"actual_rows": actual_rows, "configured_rows": configured_rows, "configured_steps": configured_steps}


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
