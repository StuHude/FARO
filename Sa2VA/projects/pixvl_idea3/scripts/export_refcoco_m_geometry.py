from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from PIL import Image

from projects.pixvl_idea1.datasets.schema import decode_rle_mask, encode_binary_mask, write_jsonl


def _normalize_mask(mask_value: Any) -> np.ndarray:
    if isinstance(mask_value, dict) and "counts" in mask_value and "size" in mask_value:
        return decode_rle_mask(mask_value)
    if isinstance(mask_value, Image.Image):
        return (np.asarray(mask_value) > 0).astype(np.uint8)
    array = np.asarray(mask_value)
    if array.ndim == 3:
        array = array[..., 0]
    return (array > 0).astype(np.uint8)


def _materialize_image(image_value: Any, output_dir: Path, sample_id: str) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{sample_id}.jpg"
    if image_path.exists():
        return str(image_path)
    if isinstance(image_value, Image.Image):
        image_value.convert("RGB").save(image_path)
        return str(image_path)
    if isinstance(image_value, dict) and image_value.get("path"):
        path = Path(image_value["path"])
        if path.exists():
            return str(path)
    if isinstance(image_value, dict) and image_value.get("bytes"):
        Image.open(io.BytesIO(image_value["bytes"])).convert("RGB").save(image_path)
        return str(image_path)
    Image.fromarray(np.asarray(image_value)).convert("RGB").save(image_path)
    return str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="moondream/refcoco-m")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    parser.add_argument("--output", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/schemas/refcoco_m_val.jsonl")
    parser.add_argument("--image-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/raw/refcoco_m_images")
    parser.add_argument("--cache-dir", default="/mnt/pfs/xiaoyicheng/.cache/huggingface")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.hf_endpoint
    ds = load_dataset(
        args.dataset_name,
        split=args.split,
        cache_dir=args.cache_dir,
    )
    image_root = Path(args.image_root)
    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(ds):
        image = item["image"]
        samples = item.get("samples") or []
        for sample_idx, sample in enumerate(samples):
            mask = _normalize_mask(sample["mask"])
            base_id = f"refcoco_m_{item_idx:06d}_{sample_idx:02d}"
            image_path = _materialize_image(image, image_root, base_id)
            sentences = sample.get("sentences") or []
            for sent_idx, sentence in enumerate(sentences):
                query = str(sentence).strip().lower()
                if not query:
                    continue
                rows.append(
                    {
                        "id": f"{base_id}_{sent_idx:02d}",
                        "task": "refseg",
                        "source": "refcoco_m",
                        "image_path": image_path,
                        "mask": encode_binary_mask(mask),
                        "query": query,
                        "caption": None,
                        "split": args.split,
                        "meta": {
                            "dataset_name": "refcoco_m",
                            "sample_idx": sample_idx,
                            "sent_idx": sent_idx,
                            "image_meta": item.get("image_meta"),
                        },
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    print(f"Exported {len(rows)} RefCOCO-M geometry eval samples to {output}")


if __name__ == "__main__":
    main()
