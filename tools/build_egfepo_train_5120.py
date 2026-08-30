#!/usr/bin/env python3
"""Expand the fixed paired training schema to the registered 5,120 rows.

This keeps the original SAMTok masks and images intact while assigning unique
pair/row IDs to deterministic repeats. It is a training-only smoke manifest;
the 512-row holdout remains untouched.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 512 or args.repeats < 10:
        raise ValueError("the registered source must contain 512 rows and repeats must be >= 10")
    output = []
    for repeat in range(args.repeats):
        for row in rows:
            item = dict(row)
            item["pair_id"] = f"egfepo-train-{repeat:02d}-{row['pair_id']}"
            item["id"] = f"egfepo-train-{repeat:02d}-{row['id']}"
            output.append(item)
    if len(output) < 5000 or sum(bool(row["meta"].get("no_target", False)) for row in output) * 2 != len(output):
        raise ValueError("generated manifest does not satisfy the 5k paired contract")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(output), "no_target_rows": len(output) // 2, "repeats": args.repeats}, sort_keys=True))


if __name__ == "__main__":
    main()
