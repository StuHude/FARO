#!/usr/bin/env python
"""Run one official GRefCOCO shard with FARO's adapter-aware evaluator."""

from __future__ import annotations

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
UPSTREAM = PIXVL_ROOT / "tools" / "run_official_qwen3_grefcoco_shard.py"
EVALUATOR = (
    FARO_ROOT
    / "Sa2VA/projects/samtok/evaluation/qwen3vl/qwen3vl_gres_eval.py"
)


def main() -> None:
    sys.path.insert(0, str(UPSTREAM.parent))
    spec = importlib.util.spec_from_file_location("pixvl_official_grefcoco_shard", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.MODULE_PATH = EVALUATOR
    module.main()


if __name__ == "__main__":
    main()
