#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official SAMTok RefCOCO evaluator on one pre-split shard."
    )
    parser.add_argument("--pixvl-root", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--vq-sam2-path", type=Path, required=True)
    parser.add_argument("--sam2-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    return parser.parse_args()


def load_official_module(pixvl_root: Path):
    sa2va_root = pixvl_root / "third_party" / "Sa2VA"
    for path in (pixvl_root, sa2va_root, pixvl_root / "third_party" / "transformers" / "src"):
        sys.path.insert(0, str(path))
    module_path = (
        sa2va_root
        / "projects"
        / "samtok"
        / "evaluation"
        / "qwen3vl"
        / "qwen3vl_refcoco_padt_style_eval.py"
    )
    spec = importlib.util.spec_from_file_location("official_samtok_refcoco_eval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official evaluator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_workspace(save_dir: Path, image_root: Path, pixvl_root: Path) -> None:
    image_link = save_dir / "data" / "glamm_data" / "images" / "coco2014" / "train2014"
    image_link.parent.mkdir(parents=True, exist_ok=True)
    if not image_link.exists():
        image_link.symlink_to(image_root, target_is_directory=True)
    padt_link = save_dir / "data" / "PaDT-MLLM"
    padt_link.parent.mkdir(parents=True, exist_ok=True)
    padt_source = pixvl_root / "third_party" / "Sa2VA" / "data" / "PaDT-MLLM"
    if not padt_link.exists():
        padt_link.symlink_to(padt_source, target_is_directory=True)


def validate_adapter(adapter: Path, base_model: Path) -> None:
    config_path = adapter / "adapter_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    configured_base = str(config.get("base_model_name_or_path", ""))
    if "samtok" not in configured_base.lower() or "samtok" not in str(base_model).lower():
        raise ValueError(
            f"Adapter/base violates the SAMTok lineage contract: {configured_base} / {base_model}"
        )


def main() -> None:
    args = parse_args()
    for path in (args.base_model, args.vq_sam2_path, args.sam2_path, args.dataset, args.image_root):
        if not path.exists():
            raise FileNotFoundError(path)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    prepare_workspace(args.save_dir, args.image_root, args.pixvl_root)
    module = load_official_module(args.pixvl_root)

    if args.adapter is not None:
        validate_adapter(args.adapter, args.base_model)
        from peft import PeftModel

        original_loader = module.Qwen3VLForConditionalGeneration.from_pretrained
        adapter_path = str(args.adapter)

        class AdapterAwareModel:
            @classmethod
            def from_pretrained(cls, model_path, **kwargs):
                base = original_loader(model_path, **kwargs)
                return PeftModel.from_pretrained(base, adapter_path)

        module.Qwen3VLForConditionalGeneration = AdapterAwareModel

    parsed = SimpleNamespace(
        model_path=str(args.base_model),
        vq_sam2_path=str(args.vq_sam2_path),
        sam2_path=str(args.sam2_path),
        dataset=str(args.dataset),
        save_dir=str(args.save_dir),
        task_id=0,
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(args.save_dir)
        module.parse_args = lambda: parsed
        module.main()
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
