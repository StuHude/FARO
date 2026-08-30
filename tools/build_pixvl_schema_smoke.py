#!/usr/bin/env python3
"""Decode a small current PixVL self-supervision slice into Idea3 schema rows."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from trackcycle.modeling.samtok_qwen3vl import SAMTokQwen3VLAdapter, decode_generated_mt_tags


MASK_RE = re.compile(r"<\|mt_start\|><\|mt_\d{4}\|><\|mt_\d{4}\|><\|mt_end\|>")


def _conversation(row: dict) -> tuple[str, str]:
    turns = row.get("conversations") or []
    human = next((str(x.get("value", "")) for x in turns if x.get("from") == "human"), "")
    answer = next((str(x.get("value", "")) for x in turns if x.get("from") == "gpt"), "")
    return human.replace("<image>", "").strip(), answer.strip()


def _mask_obj(mask: np.ndarray) -> dict:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(encoded, list):
        encoded = encoded[0]
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"format": "rle", "counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def convert(path: Path, task: str, adapter: SAMTokQwen3VLAdapter, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            row = json.loads(line)
            prompt, answer = _conversation(row)
            match = MASK_RE.search(answer if task == "refseg" else prompt)
            if not match:
                continue
            codes = decode_generated_mt_tags(match.group(0))
            if len(codes) != 2:
                continue
            image_path = str(row["image"])
            image = np.asarray(Image.open(image_path).convert("RGB"))
            decoded = adapter._decode_mask_tokens(image, codes)
            mask_obj = _mask_obj(decoded)
            if task == "refseg":
                query = prompt
                if query.lower().endswith("in this image."):
                    query = query[:-len("in this image.")].strip()
                rows.append({
                    "id": f"pixvl-smoke-refseg-{len(rows):05d}",
                    "task": "refseg", "source": "pixvl_smoke", "split": "train",
                    "image_path": image_path, "mask": mask_obj, "query": query,
                    "meta": {"failure_route": "geometry", "pixvl_source": str(path)},
                })
            else:
                rows.append({
                    "id": f"pixvl-smoke-maskcap-{len(rows):05d}",
                    "task": "maskcap", "source": "pixvl_smoke", "split": "train",
                    "image_path": image_path, "mask": mask_obj,
                    "caption": answer, "meta": {"failure_route": "semantic", "pixvl_source": str(path)},
                })
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seg-input", required=True, type=Path)
    parser.add_argument("--maskcap-input", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    adapter = SAMTokQwen3VLAdapter(device="cuda")
    adapter._ensure_loaded()
    refseg = convert(args.seg_input, "refseg", adapter, args.limit)
    maskcap = convert(args.maskcap_input, "maskcap", adapter, args.limit)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("refseg_train_routed.jsonl", refseg), ("maskcap_train_routed.jsonl", maskcap)):
        with (args.output_root / name).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"refseg": len(refseg), "maskcap": len(maskcap), "output_root": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
