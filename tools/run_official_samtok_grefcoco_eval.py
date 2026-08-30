#!/usr/bin/env python
"""Run the official GRefCOCO evaluator on SAMTok base plus optional LoRA."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


FARO_ROOT = Path(__file__).resolve().parents[1]
PIXVL_ROOT = Path(
    os.environ.get(
        "FARO_PIXVL_ROOT",
        "/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/"
        "Nemotrontiaozheng/PixVL_ailab",
    )
)
UPSTREAM = PIXVL_ROOT / "tools" / "run_official_qwen3_grefcoco_eval.py"
EVALUATOR = (
    FARO_ROOT
    / "Sa2VA/projects/samtok/evaluation/qwen3vl/qwen3vl_gres_eval.py"
)
SHARD_RUNNER = FARO_ROOT / "tools" / "run_official_samtok_grefcoco_shard.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Official GRefCOCO evaluation for the original SAMTok base and PEFT adapters."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="none")
    parser.add_argument("--vq-sam2-path", required=True)
    parser.add_argument("--sam2-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "samtok" not in args.model_path.lower():
        raise ValueError("--model-path must be an original SAMTok checkpoint")
    adapter = "" if args.adapter_path.lower() == "none" else args.adapter_path
    if adapter:
        adapter_config = Path(adapter) / "adapter_config.json"
        if not adapter_config.exists():
            raise FileNotFoundError(f"Missing PEFT adapter config: {adapter_config}")
        os.environ["FARO_PEFT_ADAPTER"] = adapter
    else:
        os.environ.pop("FARO_PEFT_ADAPTER", None)

    sys.path.insert(0, str(UPSTREAM.parent))
    spec = importlib.util.spec_from_file_location("pixvl_official_grefcoco_eval", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.MODULE_PATH = EVALUATOR
    module.SHARD_RUNNER = SHARD_RUNNER
    module.parse_args = lambda: args
    module.main()


if __name__ == "__main__":
    main()
