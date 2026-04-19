#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path

from projects.pixvl_idea1.datasets.schema import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--schema-files",
        nargs="+",
        default=[
            "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/refseg_train.jsonl",
            "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/dam_train.jsonl",
            "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/seed_dataset_train.jsonl",
            "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/fine_grained_dataset_train.jsonl",
        ],
    )
    parser.add_argument("--output-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/smoke")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    rows = []
    for schema_file in args.schema_files:
        path = Path(schema_file)
        if path.exists():
            rows.extend(load_jsonl(path))
    random.shuffle(rows)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for size in (32, 128, 512):
        subset = rows[: min(size, len(rows))]
        write_jsonl(output_root / f"smoke_{size}.jsonl", subset)
        print(f"smoke_{size}: {len(subset)} samples")


if __name__ == "__main__":
    main()

