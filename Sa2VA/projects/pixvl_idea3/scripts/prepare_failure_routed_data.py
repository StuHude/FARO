from __future__ import annotations

import argparse
import json
from pathlib import Path

from projects.pixvl_idea1.datasets.schema import load_jsonl, write_jsonl
from projects.pixvl_idea3.routing import annotate_record, stable_hash_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas")
    parser.add_argument("--output-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas")
    parser.add_argument("--semantic-holdout-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    refseg_train = [
        annotate_record(record, include_slice_tags=False, include_geometry_metrics=False)
        for record in load_jsonl(input_root / "refseg_train.jsonl")
    ]
    refcocog_train_path = input_root / "refseg_refcocog_train.jsonl"
    if refcocog_train_path.exists():
        refseg_train.extend(
            annotate_record(record, include_slice_tags=False, include_geometry_metrics=False)
            for record in load_jsonl(refcocog_train_path)
        )
    refseg_val = [
        annotate_record(record, include_geometry_metrics=False)
        for record in load_jsonl(input_root / "refseg_val.jsonl")
    ]
    dam_train = [annotate_record(record) for record in load_jsonl(input_root / "dam_train.jsonl")]
    gar_fg = [annotate_record(record) for record in load_jsonl(input_root / "fine_grained_dataset_part1_train.jsonl")]
    relation_path = input_root / "relation_dataset_train.jsonl"
    gar_relation = [annotate_record(record) for record in load_jsonl(relation_path)] if relation_path.exists() else []
    dlc_eval = [annotate_record(record) for record in load_jsonl(input_root / "dlc_bench_train.jsonl")]

    semantic_train: list[dict] = []
    semantic_eval: list[dict] = []
    for record in dam_train + gar_fg + gar_relation:
        ratio = stable_hash_ratio(record["id"])
        tags = set((record.get("meta") or {}).get("failure_slice_tags", []))
        if "semantic" in tags and ratio < args.semantic_holdout_ratio:
            semantic_eval.append(record)
        else:
            semantic_train.append(record)

    relation_eval = [
        record
        for record in refseg_val
        if "relation" in set((record.get("meta") or {}).get("failure_slice_tags", []))
    ]
    geometry_eval = [
        record
        for record in refseg_val
        if "geometry" in set((record.get("meta") or {}).get("failure_slice_tags", []))
    ]
    refcoco_m_val_path = input_root / "refcoco_m_val.jsonl"
    if refcoco_m_val_path.exists():
        geometry_eval.extend(
            annotate_record(record, include_geometry_metrics=False)
            for record in load_jsonl(refcoco_m_val_path)
        )

    outputs = {
        "refseg_train_routed.jsonl": refseg_train,
        "refseg_val_routed.jsonl": refseg_val,
        "maskcap_train_routed.jsonl": semantic_train,
        "semantic_slice_eval.jsonl": semantic_eval,
        "relation_slice_eval.jsonl": relation_eval,
        "geometry_slice_eval.jsonl": geometry_eval,
        "dlc_eval.jsonl": dlc_eval,
    }
    for filename, rows in outputs.items():
        write_jsonl(output_root / filename, rows)

    manifest = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "semantic_holdout_ratio": args.semantic_holdout_ratio,
        "counts": {name: len(rows) for name, rows in outputs.items()},
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
