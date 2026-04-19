from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from projects.pixvl_idea1.datasets.schema import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-jsonl",
        default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/refseg_val.jsonl",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/data/PaDT-MLLM/RefCOCO",
    )
    parser.add_argument("--source", default="refcoco")
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [row for row in load_jsonl(args.input_jsonl) if row["source"] == args.source]
    if args.max_items is not None:
        rows = sorted(
            rows,
            key=lambda row: int(hashlib.sha1(row["id"].encode()).hexdigest()[:12], 16),
        )[: args.max_items]

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rec_root = output_root / "rec_jsons_processed"
    rec_root.mkdir(parents=True, exist_ok=True)

    padt_items = []
    rec_items = []
    for row in rows:
        image_path = Path(row["image_path"])
        image_name = image_path.name
        image_id = int(image_name.split("_")[-1].split(".")[0])
        phrase = row["query"]
        rle = {
            "counts": row["mask"]["counts"],
            "size": row["mask"]["size"],
        }
        padt_items.append(
            {
                "id": image_id,
                "image": image_name,
                "objects": [
                    {
                        "label": phrase,
                        "rle": rle,
                    }
                ],
            }
        )
        rec_items.append(
            {
                "image": image_name,
                "normal_caption": phrase,
            }
        )

    suffix = "" if args.max_items is None else f"_{args.max_items}"
    dataset_path = output_root / f"refcoco_val{suffix}.json"
    rec_path = rec_root / f"refcoco_val{suffix}.json"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for item in padt_items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    with rec_path.open("w", encoding="utf-8") as handle:
        json.dump(rec_items, handle, ensure_ascii=False, indent=2)

    print(f"Exported {len(padt_items)} PaDT-style RefCOCO val items to {dataset_path}")


if __name__ == "__main__":
    main()
