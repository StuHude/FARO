#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    expected = sum(bool(line.strip()) for line in args.dataset.open(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0
    for shard_dir in sorted(args.shard_root.glob("shard_*")):
        for source in sorted(shard_dir.glob("*.json")):
            row = json.loads(source.read_text(encoding="utf-8"))
            key = str(row["bbox_name"])
            if key in seen:
                raise ValueError(f"Duplicate RefCOCO record: {key}")
            seen.add(key)
            destination = args.output / f"{count:06d}.json"
            destination.write_text(json.dumps(row) + "\n", encoding="utf-8")
            count += 1
    if count != expected:
        raise ValueError(f"Merged {count} predictions, expected {expected}")
    print(json.dumps({"merged": count, "expected": expected}))


if __name__ == "__main__":
    main()
