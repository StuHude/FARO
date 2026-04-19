from __future__ import annotations

import argparse
from pathlib import Path

from projects.pixvl_idea1.datasets.adapters_refcoco import iter_refcoco_records
from projects.pixvl_idea1.datasets.schema import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/ref_seg/ref_seg",
    )
    parser.add_argument(
        "--output",
        default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/refseg_refcocog_train.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(iter_refcoco_records(args.data_root, "refcocog", "train"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    print(f"Exported {len(rows)} RefCOCOg geometry train samples to {output}")


if __name__ == "__main__":
    main()
