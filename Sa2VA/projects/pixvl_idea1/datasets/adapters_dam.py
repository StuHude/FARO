from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset
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


def export_dam_records(
    output_path: str,
    dataset_name: str = "nvidia/describe-anything-dataset",
    split: str = "train",
    image_root: str = "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/dam_images",
) -> int:
    ds = load_dataset(dataset_name, split=split)
    image_root_path = Path(image_root)
    rows = []
    for idx, example in enumerate(ds):
        caption = _first_present(example, CAPTION_KEYS)
        mask = _coerce_mask(_first_present(example, MASK_KEYS))
        image = _first_present(example, IMAGE_KEYS)
        image_path = _materialize_image(image, image_root_path, f"dam_{idx:08d}")
        rows.append(
            {
                "id": f"dam_{idx:08d}",
                "task": "maskcap",
                "source": "dam",
                "image_path": image_path,
                "mask": mask,
                "query": None,
                "caption": str(caption).strip(),
                "split": split,
                "meta": {
                    "dataset_name": "dam",
                },
            }
        )
    from .schema import write_jsonl

    write_jsonl(output_path, rows)
    return len(rows)


def export_dlc_bench_records(
    output_path: str,
    dataset_name: str = "nvidia/DLC-Bench",
    split: str = "train",
    image_root: str = "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/dlc_bench_images",
) -> int:
    ds = load_dataset(dataset_name, split=split)
    image_root_path = Path(image_root)
    rows = []
    for idx, example in enumerate(ds):
        image = _first_present(example, IMAGE_KEYS)
        image_path = _materialize_image(image, image_root_path, f"dlc_{idx:08d}")
        mask_samples = example.get("mask_samples")
        if not mask_samples:
            continue
        for mask_idx, sample in enumerate(mask_samples):
            caption = _first_present(sample, CAPTION_KEYS)
            mask = _coerce_mask(_first_present(sample, MASK_KEYS))
            rows.append(
                {
                    "id": f"dlc_{idx:08d}_{mask_idx}",
                    "task": "maskcap",
                    "source": "dlc_bench",
                    "image_path": image_path,
                    "mask": mask,
                    "query": None,
                    "caption": str(caption).strip(),
                    "split": split,
                    "meta": {
                        "dataset_name": "dlc_bench",
                    },
                }
            )
    from .schema import write_jsonl

    write_jsonl(output_path, rows)
    return len(rows)


def export_dlc_bench_local_records(
    output_path: str,
    dataset_root: str = "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dlc_bench",
) -> int:
    dataset_root_path = Path(dataset_root)
    annotations = json.loads((dataset_root_path / "annotations.json").read_text(encoding="utf-8"))
    images = {item["id"]: item for item in annotations["images"]}
    categories = {item["id"]: item["name"] for item in annotations["categories"]}

    rows = []
    for ann in annotations["annotations"]:
        image_info = images[ann["image_id"]]
        image_path = dataset_root_path / "images" / image_info["file_name"]
        rows.append(
            {
                "id": f"dlc_{ann['id']}",
                "task": "maskcap",
                "source": "dlc_bench",
                "image_path": str(image_path),
                "mask": {
                    "format": "rle",
                    "counts": ann["segmentation"]["counts"],
                    "size": list(ann["segmentation"]["size"]),
                },
                "query": None,
                "caption": categories[ann["category_id"]],
                "split": "train",
                "meta": {
                    "dataset_name": "dlc_bench",
                    "image_id": ann["image_id"],
                    "annotation_id": ann["id"],
                },
            }
        )
    from .schema import write_jsonl

    write_jsonl(output_path, rows)
    return len(rows)


def export_dam_local_records(
    output_path: str,
    dataset_root: str,
    subset_names: list[str],
    image_root: str = "/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/dam_images",
    max_rows: int | None = None,
) -> int:
    dataset_root_path = Path(dataset_root)
    image_root_path = Path(image_root)
    tar_handles: dict[Path, tarfile.TarFile] = {}
    tar_member_cache: dict[Path, dict[str, str]] = {}
    rows = []

    def materialize_from_tar(subset_dir: Path, tar_name: str, image_rel: str, row_id: str) -> str:
        target_path = image_root_path / row_id
        target_path = target_path.with_suffix(Path(image_rel).suffix or ".jpg")
        if target_path.exists():
            return str(target_path)
        tar_path = subset_dir / "images" / tar_name
        if not tar_path.exists():
            raise FileNotFoundError(f"Missing tar shard: {tar_path}")
        if tar_path not in tar_handles:
            tar_handles[tar_path] = tarfile.open(tar_path, "r")
            tar_member_cache[tar_path] = {
                Path(member.name).name: member.name
                for member in tar_handles[tar_path].getmembers()
                if member.isfile()
            }
        basename = Path(image_rel).name
        member_name = tar_member_cache[tar_path].get(basename)
        if member_name is None:
            stem = Path(basename).stem
            normalized = str(int(stem)) + Path(basename).suffix if stem.isdigit() else basename
            member_name = tar_member_cache[tar_path].get(normalized)
        if member_name is None:
            raise FileNotFoundError(f"Missing {basename} in {tar_path}")
        extracted = tar_handles[tar_path].extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"Missing {basename} in {tar_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(extracted.read())).convert("RGB").save(target_path)
        return str(target_path)

    for subset_name in subset_names:
        subset_dir = dataset_root_path / subset_name
        annotations = json.loads((subset_dir / "annotations.json").read_text(encoding="utf-8"))
        for img_id, region_items in annotations.items():
            for region in region_items:
                row_id = f"{subset_name}_{img_id}_{region['ann_id']}"
                try:
                    image_path = materialize_from_tar(subset_dir, region["tar_file"], region["image"], row_id)
                except FileNotFoundError:
                    continue
                rows.append(
                    {
                        "id": row_id,
                        "task": "maskcap",
                        "source": subset_name.lower(),
                        "image_path": image_path,
                        "mask": {
                            "format": "rle",
                            "counts": region["mask_rle"]["counts"],
                            "size": list(region["mask_rle"]["size"]),
                        },
                        "query": None,
                        "caption": region["caption"].strip(),
                        "split": "train",
                        "meta": {
                            "dataset_name": "describe_anything",
                            "subset_name": subset_name,
                            "img_id": region["img_id"],
                            "ann_id": region["ann_id"],
                            "tar_file": region["tar_file"],
                        },
                    }
                )
                if max_rows is not None and len(rows) >= max_rows:
                    for handle in tar_handles.values():
                        handle.close()
                    from .schema import write_jsonl

                    write_jsonl(output_path, rows)
                    return len(rows)

    for handle in tar_handles.values():
        handle.close()

    from .schema import write_jsonl

    write_jsonl(output_path, rows)
    return len(rows)
