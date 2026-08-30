#!/usr/bin/env python3
"""Bootstrap confidence intervals for FEPO per-example evaluator records.

The evaluator should write ``{"records": [{"slice": ..., "ciou": ...}]}``
or a bare list of records. This utility intentionally uses only the standard
library so it can run in the lightweight analysis environment.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute a quantile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(q)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def bootstrap_mean(
    values: Sequence[float], *, repeats: int = 5000, seed: int = 0
) -> dict[str, Any]:
    """Return mean and percentile bootstrap interval for one metric."""

    clean = [float(value) for value in values]
    if not clean:
        raise ValueError("metric has no observations")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = random.Random(seed)
    n = len(clean)
    means = [sum(clean[rng.randrange(n)] for _ in range(n)) / n for _ in range(repeats)]
    return {
        "n": n,
        "mean": sum(clean) / n,
        "ci95": [_quantile(means, 0.025), _quantile(means, 0.975)],
        "bootstrap_repeats": repeats,
        "seed": seed,
    }


def bootstrap_paired_delta(
    left: Sequence[float],
    right: Sequence[float],
    *,
    repeats: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Return CI for ``right - left`` using paired resampling."""

    if len(left) != len(right):
        raise ValueError("paired metrics must have equal lengths")
    deltas = [float(r) - float(l) for l, r in zip(left, right)]
    result = bootstrap_mean(deltas, repeats=repeats, seed=seed)
    result["comparison"] = "right-minus-left"
    return result


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("records", payload.get("samples"))
    else:
        records = None
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("input must contain a list of object records")
    return records


def summarize_records(
    records: Iterable[dict[str, Any]],
    *,
    metric: str,
    repeats: int,
    seed: int,
    group_key: str | None = None,
) -> dict[str, Any]:
    rows = list(records)
    if group_key is None:
        return {"all": bootstrap_mean([row[metric] for row in rows], repeats=repeats, seed=seed)}
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(str(row.get(group_key, "unknown")), []).append(float(row[metric]))
    return {
        group: bootstrap_mean(values, repeats=repeats, seed=seed + index)
        for index, (group, values) in enumerate(sorted(groups.items()))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--group-key")
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    print(json.dumps(summarize_records(_records(payload), metric=args.metric, repeats=args.repeats, seed=args.seed, group_key=args.group_key), indent=2))


if __name__ == "__main__":
    main()
