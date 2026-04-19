from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.ipc as ipc
from PIL import Image

from .schema import encode_binary_mask


CAPTION_KEYS = ("caption", "description", "text", "answer", "region_caption")
MASK_KEYS = ("segmentation", "mask", "region_mask", "rle")
IMAGE_KEYS = ("image", "image_path", "img", "img_path")


def _first_present(example: dict[str, Any], candidates: Iterable[str]) -> Any:
    for key in candidates:
        if key in example and example[key] is not None:
            return example[key]
    raise KeyError(f"无法从字段中推断数据列，可用字段: {sorted(example.keys())}")


def _materialize_image(value: Any, image_root: Path, row_id: str) -> str:
    if isinstance(value, str) and Path(value).exists():
        return value
    image_root.mkdir(parents=True, exist_ok=True)
    image_path = image_root / f"{row_id}.jpg"
    if image_path.exists():
        return str(image_path)
    if isinstance(value, dict) and "path" in value and value["path"] and Path(value["path"]).exists():
        return value["path"]
    if isinstance(value, dict) and "bytes" in value:
        Image.open(io.BytesIO(value["bytes"])).convert("RGB").save(image_path)
        return str(image_path)
    if isinstance(value, Image.Image):
        value.convert("RGB").save(image_path)
        return str(image_path)
    if isinstance(value, np.ndarray):
        Image.fromarray(value).convert("RGB").save(image_path)
        return str(image_path)
    raise TypeError(f"不支持的图像字段类型: {type(value)}")


def _coerce_mask(mask_value: Any) -> dict[str, Any]:
    if isinstance(mask_value, dict) and "counts" in mask_value and "size" in mask_value:
        return {
            "format": "rle",
            "counts": mask_value["counts"],
            "size": list(mask_value["size"]),
        }
    if isinstance(mask_value, np.ndarray):
        return encode_binary_mask(mask_value.astype(np.uint8))
    if isinstance(mask_value, list):
        return encode_binary_mask(np.asarray(mask_value, dtype=np.uint8))
    raise TypeError(f"不支持的 mask 字段类型: {type(mask_value)}")


def export_gar_records(
    output_path: str,
    config_name: str,
    dataset_name: str = "HaochenWang/Grasp-Any-Region-Dataset",
    split: str = "train",
    image_root: str = "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/gar_images",
) -> int:
    ds = load_dataset(dataset_name, name=config_name, split=split)
    image_root_path = Path(image_root)
    rows = []
    source_name = config_name.lower().replace("-", "_")
    for idx, example in enumerate(ds):
        caption = _first_present(example, CAPTION_KEYS)
        mask = _coerce_mask(_first_present(example, MASK_KEYS))
        image = _first_present(example, IMAGE_KEYS)
        image_path = _materialize_image(image, image_root_path, f"{source_name}_{idx:08d}")
        rows.append(
            {
                "id": f"{source_name}_{idx:08d}",
                "task": "maskcap",
                "source": source_name,
                "image_path": image_path,
                "mask": mask,
                "query": None,
                "caption": str(caption).strip(),
                "split": split,
                "meta": {
                    "dataset_name": "gar",
                    "config_name": config_name,
                },
            }
        )
    from .schema import write_jsonl

    write_jsonl(output_path, rows)
    return len(rows)


def export_gar_local_arrow_records(
    output_path: str,
    dataset_root: str,
    part_names: list[str],
    image_root: str = "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/gar_images",
    max_files: int | None = None,
    max_rows: int | None = None,
    file_start: int = 0,
    file_stride: int = 1,
) -> int:
    image_root_path = Path(image_root)
    output_target = Path(output_path)
    output_target.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with output_target.open("w", encoding="utf-8") as writer:
        for part_name in part_names:
            part_dir = Path(dataset_root) / part_name
            arrow_files = sorted(part_dir.glob("*.arrow"))
            if file_stride > 1:
                arrow_files = arrow_files[file_start::file_stride]
            elif file_start > 0:
                arrow_files = arrow_files[file_start:]
            if max_files is not None:
                arrow_files = arrow_files[:max_files]
            for arrow_file in arrow_files:
                with arrow_file.open("rb") as handle:
                    table = ipc.open_stream(handle).read_all()
                for idx, example in enumerate(table.to_pylist()):
                    image = example["image"]
                    image_path = _materialize_image(image, image_root_path, f"{part_name}_{arrow_file.stem}_{idx:08d}")
                    caption = example["conversations"][1]["value"].strip()
                    raw_masks = example["mask_rle"]
                    if isinstance(raw_masks, list):
                        if not raw_masks:
                            continue
                        mask = {
                            "format": "rle",
                            "counts": raw_masks[0]["counts"],
                            "size": list(raw_masks[0]["size"]),
                        }
                        aux_mask = {
                            "format": "rle",
                            "counts": raw_masks[1]["counts"],
                            "size": list(raw_masks[1]["size"]),
                        } if len(raw_masks) > 1 else None
                        caption = (
                            caption.replace("<Prompt0>", "Region A")
                            .replace("<Prompt1>", "Region B")
                            .strip()
                        )
                    else:
                        mask = {
                            "format": "rle",
                            "counts": raw_masks["counts"],
                            "size": list(raw_masks["size"]),
                        }
                        aux_mask = None
                    meta = {
                        "dataset_name": "gar",
                        "part_name": part_name,
                        "category": example.get("catagory"),
                    }
                    if aux_mask is not None:
                        meta["aux_mask"] = aux_mask
                    row = {
                        "id": f"{part_name}_{arrow_file.stem}_{idx:08d}",
                        "task": "maskcap",
                        "source": part_name.lower().replace("-", "_"),
                        "image_path": image_path,
                        "mask": mask,
                        "query": None,
                        "caption": caption,
                        "split": "train",
                        "meta": meta,
                    }
                    writer.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += 1
                    if max_rows is not None and total_rows >= max_rows:
                        return total_rows
    return total_rows
