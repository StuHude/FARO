#!/usr/bin/env python3

from __future__ import annotations

import argparse

from projects.pixvl_idea1.datasets.adapters_dam import (
    export_dam_local_records,
    export_dam_records,
    export_dlc_bench_local_records,
    export_dlc_bench_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/dam_train.jsonl")
    parser.add_argument("--dataset-name", default="nvidia/describe-anything-dataset")
    parser.add_argument("--split", default="train")
    parser.add_argument("--image-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/dam_images")
    parser.add_argument("--skip-dam", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--dataset-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam")
    parser.add_argument("--subset-names", nargs="+", default=["COCOStuff", "LVIS", "PACO"])
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--export-dlc-bench", action="store_true")
    parser.add_argument("--dlc-output-path", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/dlc_bench_train.jsonl")
    parser.add_argument("--dlc-dataset-name", default="nvidia/DLC-Bench")
    parser.add_argument("--dlc-image-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/dlc_bench_images")
    parser.add_argument("--dlc-dataset-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench")
    parser.add_argument("--dlc-local-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_dam:
        if args.local_only:
            total = export_dam_local_records(
                output_path=args.output_path,
                dataset_root=args.dataset_root,
                subset_names=args.subset_names,
                image_root=args.image_root,
                max_rows=args.max_rows,
            )
        else:
            total = export_dam_records(
                output_path=args.output_path,
                dataset_name=args.dataset_name,
                split=args.split,
                image_root=args.image_root,
            )
        print(f"Exported {total} DAM samples to {args.output_path}")
    if args.export_dlc_bench:
        if args.dlc_local_only:
            total = export_dlc_bench_local_records(
                output_path=args.dlc_output_path,
                dataset_root=args.dlc_dataset_root,
            )
        else:
            total = export_dlc_bench_records(
                output_path=args.dlc_output_path,
                dataset_name=args.dlc_dataset_name,
                split="train",
                image_root=args.dlc_image_root,
            )
        print(f"Exported {total} DLC-Bench samples to {args.dlc_output_path}")


if __name__ == "__main__":
    main()
