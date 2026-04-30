from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from projects.pixvl_idea1.datasets.schema import load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas")
    parser.add_argument("--output-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas_shuffled")
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def shuffled_labels(rows: list[dict], seed: int) -> list[str]:
    labels = [str((row.get("meta") or {}).get("failure_route", "default")) for row in rows]
    rng = random.Random(seed)
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    return shuffled


def apply_labels(rows: list[dict], labels: list[str]) -> list[dict]:
    updated_rows: list[dict] = []
    for row, label in zip(rows, labels):
        updated = dict(row)
        meta = dict(updated.get("meta") or {})
        tags = set(meta.get("failure_slice_tags", []))
        old_label = str(meta.get("failure_route", "default"))
        if old_label in tags:
            tags.remove(old_label)
        tags.add(label)
        meta["failure_route"] = label
        meta["failure_slice_tags"] = sorted(tags)
        updated["meta"] = meta
        updated["route_bucket"] = label
        updated_rows.append(updated)
    return updated_rows


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    outputs = {}
    manifest = {"seed": args.seed, "counts": {}, "label_distributions": {}}

    for filename in ["refseg_train_routed.jsonl", "maskcap_train_routed.jsonl", "refseg_val_routed.jsonl"]:
        rows = load_jsonl(input_root / filename)
        labels = shuffled_labels(rows, seed=args.seed + hash(filename) % 1000)
        updated = apply_labels(rows, labels)
        outputs[filename] = updated
        manifest["counts"][filename] = len(updated)
        manifest["label_distributions"][filename] = dict(Counter(labels))

    # Eval slices remain unchanged; training labels only are shuffled.
    for filename in [
        "semantic_slice_eval.jsonl",
        "relation_slice_eval.jsonl",
        "geometry_slice_eval.jsonl",
        "dlc_eval.jsonl",
    ]:
        rows = load_jsonl(input_root / filename)
        outputs[filename] = rows
        manifest["counts"][filename] = len(rows)

    for filename, rows in outputs.items():
        write_jsonl(output_root / filename, rows)

    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
