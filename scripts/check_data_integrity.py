#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_dir(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    files = [p for p in path.rglob("*") if p.is_file()]
    return {
        "exists": True,
        "files": len(files),
        "examples": [str(p) for p in files[:5]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1")
    args = parser.parse_args()

    root = Path(args.root)
    summary = {
        "root": str(root),
        "hf": summarize_dir(root / "hf"),
        "raw": summarize_dir(root / "raw"),
        "schemas": summarize_dir(root / "schemas"),
        "smoke": summarize_dir(root / "smoke"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
