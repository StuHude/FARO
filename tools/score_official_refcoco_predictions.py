#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixvl-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    sa2va_root = args.pixvl_root / "third_party" / "Sa2VA"
    for path in (args.pixvl_root, sa2va_root, args.pixvl_root / "third_party" / "transformers" / "src"):
        sys.path.insert(0, str(path))
    module_path = (
        sa2va_root
        / "projects"
        / "samtok"
        / "evaluation"
        / "qwen3vl"
        / "qwen3vl_refcoco_padt_style_eval.py"
    )
    spec = importlib.util.spec_from_file_location("official_samtok_refcoco_metric", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data_link = args.workspace / "data" / "PaDT-MLLM"
    data_link.parent.mkdir(parents=True, exist_ok=True)
    if not data_link.exists():
        data_link.symlink_to(sa2va_root / "data" / "PaDT-MLLM", target_is_directory=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(args.workspace)
        module.metric(str(args.predictions))
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
