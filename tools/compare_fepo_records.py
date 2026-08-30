#!/usr/bin/env python3
"""Compare two FEPO evaluator outputs with paired bootstrap intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.bootstrap_metrics import bootstrap_paired_delta


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


def _records(payload: dict[str, Any], section: str, metric: str) -> dict[str, float]:
    rows = payload.get(section, {}).get("records", [])
    if not isinstance(rows, list):
        raise ValueError(f"{section} has no records list")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or "id" not in row or metric not in row:
            continue
        result[str(row["id"])] = float(row[metric])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--sections", nargs="+", default=["geometry", "semantic", "refseg_overall"])
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    left = _load(args.left)
    right = _load(args.right)
    output = {}
    for section in args.sections:
        metric = "reward" if section == "semantic" else "ciou"
        lhs = _records(left, section, metric)
        rhs = _records(right, section, metric)
        keys = sorted(set(lhs) & set(rhs))
        if not keys:
            output[section] = {"error": "no paired records"}
            continue
        output[section] = bootstrap_paired_delta(
            [lhs[key] for key in keys],
            [rhs[key] for key in keys],
            repeats=args.repeats,
            seed=args.seed,
        )
        output[section]["num_paired"] = len(keys)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
