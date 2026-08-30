#!/usr/bin/env python3
"""Build exact-multiset curriculum and shuffled selective-RL schemas."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("curriculum", type=Path)
    parser.add_argument("control", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line
    ]
    positive = sorted(
        (row for row in rows if not bool((row.get("meta") or {}).get("no_target", False))),
        key=lambda row: str(row["id"]),
    )[:150]
    negative = sorted(
        (row for row in rows if bool((row.get("meta") or {}).get("no_target", False))),
        key=lambda row: str(row["id"]),
    )[:50]
    if len(positive) != 150 or len(negative) != 50:
        raise ValueError("input must provide at least 150 positive and 50 negative rows")

    rng = random.Random(args.seed)
    stage_a = positive[:100]
    stage_b = positive[100:] + negative
    rng.shuffle(stage_b)
    curriculum = stage_a + stage_b
    control = curriculum.copy()
    random.Random(args.seed + 1).shuffle(control)

    curriculum_ids = [str(row["id"]) for row in curriculum]
    control_ids = [str(row["id"]) for row in control]
    if len(set(curriculum_ids)) != 200 or set(curriculum_ids) != set(control_ids):
        raise AssertionError("curriculum and control must contain the same 200 unique IDs")
    write_jsonl(args.curriculum, curriculum)
    write_jsonl(args.control, control)
    audit = {
        "seed": args.seed,
        "num_rows": 200,
        "num_positive": 150,
        "num_negative": 50,
        "stage_a_positive": 100,
        "stage_a_negative": 0,
        "stage_b_positive": 50,
        "stage_b_negative": 50,
        "same_id_multiset": True,
        "unique_ids": len(set(curriculum_ids)),
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
