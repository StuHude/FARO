from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from pycocotools import mask as mask_utils


@dataclass(frozen=True)
class MaskRecord:
    format: str
    counts: Any
    size: list[int]


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(mask[:, :, None]))[0]
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return {
        "format": "rle",
        "counts": encoded["counts"],
        "size": [int(mask.shape[0]), int(mask.shape[1])],
    }


def decode_rle_mask(mask_obj: dict[str, Any]) -> np.ndarray:
    rle = {
        "counts": mask_obj["counts"],
        "size": mask_obj["size"],
    }
    mask = mask_utils.decode(rle)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.uint8)


def mask_area(mask_obj: dict[str, Any]) -> int:
    return int(decode_rle_mask(mask_obj).sum())


def mask_bbox_xyxy(mask_obj: dict[str, Any]) -> list[int]:
    mask = decode_rle_mask(mask_obj)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

