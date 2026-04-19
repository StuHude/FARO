#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from projects.pixvl_idea1.datasets.adapters_gar import export_gar_local_arrow_records, export_gar_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-names",
        nargs="+",
        default=["Fine-Grained-Dataset-Part1"],
    )
    parser.add_argument("--dataset-name", default="HaochenWang/Grasp-Any-Region-Dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas")
    parser.add_argument("--image-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/gar_images")
    parser.add_argument("--dataset-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/gar")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--file-start", type=int, default=0)
    parser.add_argument("--file-stride", type=int, default=1)
    parser.add_argument("--output-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for config_name in args.config_names:
        normalized = config_name.lower().replace("-", "_")
        output_path = Path(args.output_path) if args.output_path else output_root / f"{normalized}_train.jsonl"
        if args.local_only:
            total = export_gar_local_arrow_records(
                output_path=str(output_path),
                dataset_root=args.dataset_root,
                part_names=[config_name],
                image_root=args.image_root,
                max_files=args.max_files,
                max_rows=args.max_rows,
                file_start=args.file_start,
                file_stride=args.file_stride,
            )
        else:
            total = export_gar_records(
                output_path=str(output_path),
                config_name=config_name,
                dataset_name=args.dataset_name,
                split=args.split,
                image_root=args.image_root,
            )
        print(f"Exported {total} GAR samples ({config_name}) to {output_path}")


if __name__ == "__main__":
    main()
