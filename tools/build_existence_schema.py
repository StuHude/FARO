"""Build an image-disjoint positive/negative existence schema.

The preferred positive source is GRefCOCO's public referring expressions and
COCO annotations.  The negative source is a GRES no-target JSONL.  This tool
only writes an existence schema; it does not load models or training data.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
from pathlib import Path
from typing import Any


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--positive-grefs", help="GRefCOCO grefs(unc).json")
    p.add_argument("--positive-instances", help="COCO instances.json")
    p.add_argument("--positive-image-root", help="COCO image directory")
    p.add_argument("--positive", help="Legacy JSONL positive source")
    p.add_argument("--negative", help="Legacy GRES no-target JSONL")
    p.add_argument("--negative-grefs", help="GRefCOCO file providing same-source no-target expressions")
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="train")
    p.add_argument("--exclude-schema", help="JSONL whose image IDs must not appear in this output")
    return p.parse_args()


def normalize_query(text: str) -> str:
    text = re.sub(r"^\s*please\s+segment\s+", "", text, flags=re.I)
    text = re.split(r"\s*\bif\s+no\s+described\s+target\s+exists\b", text, maxsplit=1, flags=re.I)[0]
    return re.sub(r"\s+", " ", text).strip()


def image_key(path: str) -> str:
    return Path(path).name.lower()


def load_gref_positive(grefs_path: str, instances_path: str, image_root: str, split: str) -> list[dict[str, Any]]:
    if not Path(image_root).is_dir():
        raise FileNotFoundError(f"Missing GRefCOCO image root: {image_root}")
    grefs = json.loads(Path(grefs_path).read_text(encoding="utf-8"))
    instances = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in instances["images"]}
    anns = {int(x["id"]): x for x in instances["annotations"]}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ref in grefs:
        if ref.get("split") != split or ref.get("no_target"):
            continue
        image = images.get(int(ref["image_id"]))
        if image is None:
            continue
        ann_ids = ref.get("ann_id") or []
        if isinstance(ann_ids, int):
            ann_ids = [ann_ids]
        if not ann_ids or ann_ids == [-1] or any(int(a) not in anns for a in ann_ids):
            continue
        path = str(Path(image_root) / image["file_name"])
        for sent in ref.get("sentences", []):
            query = normalize_query(str(sent.get("sent", "")))
            key = (image_key(path), query.lower())
            if query and key not in seen:
                seen.add(key)
                rows.append({
                    "image_path": path,
                    "query": query,
                    "source": "grefcoco_positive",
                    "source_image_id": str(image["id"]),
                    "source_ref_id": str(ref.get("ref_id", "")),
                    "source_sent_id": str(sent.get("sent_id", "")),
                })
    return rows


def load_gref_negative(grefs_path: str, instances_path: str, image_root: str, split: str) -> list[dict[str, Any]]:
    if not Path(image_root).is_dir():
        raise FileNotFoundError(f"Missing GRefCOCO image root: {image_root}")
    grefs = json.loads(Path(grefs_path).read_text(encoding="utf-8"))
    instances = json.loads(Path(instances_path).read_text(encoding="utf-8"))
    images = {int(x["id"]): x for x in instances["images"]}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for ref in grefs:
        ann_ids = ref.get("ann_id") or []
        if ref.get("split") != split or not (ref.get("no_target") or ann_ids == [-1]):
            continue
        image = images.get(int(ref["image_id"]))
        if image is None:
            continue
        path = str(Path(image_root) / image["file_name"])
        for sent in ref.get("sentences", []):
            query = normalize_query(str(sent.get("sent", "")))
            key = (image_key(path), query.lower())
            if query and key not in seen:
                seen.add(key)
                rows.append({
                    "image_path": path,
                    "query": query,
                    "source": "grefcoco_no_target",
                    "source_image_id": str(image["id"]),
                    "source_ref_id": str(ref.get("ref_id", "")),
                    "source_sent_id": str(sent.get("sent_id", "")),
                })
    return rows


def load_legacy_positive(path: str) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = row.get("image_path") or row.get("image")
        query = row.get("prompt") or row.get("query")
        conversations = row.get("conversations") or []
        if not query and conversations:
            query = str(conversations[0].get("value", "")).replace("<image>", "").strip()
        if image and query:
            rows.append({"image_path": str(image), "query": normalize_query(str(query)), "source_image_id": image_key(str(image)), "source": "refcoco_positive"})
    return rows


def load_negative(path: str) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image = row.get("image_path") or row.get("image")
        query = row.get("prompt") or row.get("query")
        answer = str(row.get("answer") or row.get("target") or "")
        if image and query and "no target" in answer.lower():
            rows.append({"image_path": str(image), "query": normalize_query(str(query)), "source_image_id": image_key(str(image))})
    return rows


def choose_length_matched(positives: list[dict[str, Any]], negatives: list[dict[str, Any]], limit: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    chosen_p: list[dict[str, Any]] = []
    chosen_n: list[dict[str, Any]] = []
    used_p: set[str] = set()
    used_n: set[str] = set()
    indexed_positives = sorted(
        (len(p["query"]), image_key(p["image_path"]), p["query"], index, p)
        for index, p in enumerate(positives)
    )
    positive_lengths = [item[0] for item in indexed_positives]
    for n in negatives:
        nid = image_key(n["image_path"])
        if nid in used_n or nid in used_p:
            continue
        center = bisect.bisect_left(positive_lengths, len(n["query"]))
        left, right = center - 1, center
        p = None
        while left >= 0 or right < len(indexed_positives):
            if left < 0:
                candidate_index = right
                right += 1
            elif right >= len(indexed_positives):
                candidate_index = left
                left -= 1
            else:
                left_delta = abs(indexed_positives[left][0] - len(n["query"]))
                right_delta = abs(indexed_positives[right][0] - len(n["query"]))
                if left_delta <= right_delta:
                    candidate_index = left
                    left -= 1
                else:
                    candidate_index = right
                    right += 1
            candidate = indexed_positives[candidate_index][4]
            pid = image_key(candidate["image_path"])
            if pid not in used_p and pid not in used_n and pid != nid:
                p = candidate
                break
        if p is None:
            break
        chosen_p.append(p)
        chosen_n.append(n)
        used_p.add(image_key(p["image_path"]))
        used_n.add(nid)
        if len(chosen_p) >= limit:
            break
    return chosen_p, chosen_n


def main() -> None:
    a = args()
    if a.positive_grefs or a.positive_instances or a.positive_image_root:
        if not (a.positive_grefs and a.positive_instances and a.positive_image_root):
            raise SystemExit("--positive-grefs, --positive-instances and --positive-image-root are required together")
        positives = load_gref_positive(a.positive_grefs, a.positive_instances, a.positive_image_root, a.split)
    elif a.positive:
        positives = load_legacy_positive(a.positive)
    else:
        raise SystemExit("provide the preferred GRefCOCO positive arguments or legacy --positive")
    if a.negative_grefs:
        if not (a.positive_instances and a.positive_image_root):
            raise SystemExit("--positive-instances and --positive-image-root are required with --negative-grefs")
        negatives = load_gref_negative(a.negative_grefs, a.positive_instances, a.positive_image_root, a.split)
    elif a.negative:
        negatives = load_negative(a.negative)
    else:
        raise SystemExit("provide preferred --negative-grefs or legacy --negative")
    if a.exclude_schema:
        excluded = {
            image_key(str(row.get("image_path") or row.get("image")))
            for row in (
                json.loads(line)
                for line in Path(a.exclude_schema).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        positives = [row for row in positives if image_key(row["image_path"]) not in excluded]
        negatives = [row for row in negatives if image_key(row["image_path"]) not in excluded]
    positives, negatives = choose_length_matched(positives, negatives, a.limit, a.seed)
    if len(positives) < a.limit:
        raise SystemExit(f"only {len(positives)} disjoint pairs available, requested {a.limit}")

    rows: list[dict[str, Any]] = []
    for i, (pos, neg) in enumerate(zip(positives, negatives)):
        pair = f"existence-{i:05d}"
        rows.append({"id": f"{pair}-positive", "pair_id": pair, "task": "existence", "source": pos.get("source", "positive"), "split": a.split, "image_path": pos["image_path"], "query": pos["query"], "answer": "Target exists.", "meta": {k: v for k, v in pos.items() if k not in {"image_path", "query", "source"}}})
        rows.append({"id": f"{pair}-negative", "pair_id": pair, "task": "existence", "source": neg.get("source", "no_target"), "split": a.split, "image_path": neg["image_path"], "query": neg["query"], "answer": "No target.", "meta": {k: v for k, v in neg.items() if k not in {"image_path", "query", "source"}}})
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    pos_ids = {image_key(r["image_path"]) for r in rows if r["answer"] == "Target exists."}
    neg_ids = {image_key(r["image_path"]) for r in rows if r["answer"] == "No target."}
    lengths = [abs(len(rows[i]["query"]) - len(rows[i + 1]["query"])) for i in range(0, len(rows), 2)]
    parent_dirs = {
        "positive": sorted({str(Path(r["image_path"]).parent) for r in rows if r["answer"] == "Target exists."}),
        "negative": sorted({str(Path(r["image_path"]).parent) for r in rows if r["answer"] == "No target."}),
    }
    report = {"rows": len(rows), "positive": len(positives), "negative": len(negatives), "image_disjoint": not (pos_ids & neg_ids), "excluded_schema": a.exclude_schema, "mean_abs_char_length_delta": sum(lengths) / len(lengths), "max_abs_char_length_delta": max(lengths), "in_this_image_positive": sum("in this image" in r["query"].lower() for r in rows if r["answer"] == "Target exists."), "in_this_image_negative": sum("in this image" in r["query"].lower() for r in rows if r["answer"] == "No target."), "parent_dirs": parent_dirs, "seed": a.seed, "split": a.split}
    out.with_suffix(out.suffix + ".audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
