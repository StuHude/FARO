#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="nvidia/describe-anything-dataset")
    parser.add_argument("--output-root", default="/mnt/pfs/xiaoyicheng/data/pixvl_idea1/hf/dam")
    parser.add_argument("--subset-names", nargs="+", default=["COCOStuff", "LVIS", "PACO"])
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    return parser.parse_args()


def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not output.with_suffix(output.suffix + ".aria2").exists():
        return
    while True:
        if shutil.which("aria2c"):
            cmd = [
                "aria2c",
                "--continue=true",
                "--max-connection-per-server=16",
                "--split=16",
                "--min-split-size=1M",
                "--max-tries=0",
                "--timeout=120",
                "--file-allocation=none",
                "--summary-interval=0",
                "--dir",
                str(output.parent),
                "--out",
                output.name,
                url,
            ]
        else:
            cmd = [
                "wget",
                "-c",
                "--retry-connrefused",
                "--waitretry=5",
                "--read-timeout=120",
                "--timeout=120",
                "-O",
                str(output),
                url,
            ]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            return
        time.sleep(5)


def main() -> None:
    args = parse_args()
    api = HfApi(endpoint=args.hf_endpoint)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # fetch small root files
    for path in ("README.md", ".gitattributes"):
        download_file(
            f"{args.hf_endpoint}/datasets/{args.repo_id}/resolve/main/{path}",
            output_root / path,
        )

    tree = api.list_repo_tree(repo_id=args.repo_id, repo_type="dataset", recursive=True, expand=True)
    files = []
    for item in tree:
        path = getattr(item, "path", "") or ""
        if any(path.startswith(f"{subset}/") for subset in args.subset_names):
            if getattr(item, "size", None) is not None:
                files.append(path)

    for path in files:
        download_file(
            f"{args.hf_endpoint}/datasets/{args.repo_id}/resolve/main/{path}",
            output_root / path,
        )


if __name__ == "__main__":
    main()
