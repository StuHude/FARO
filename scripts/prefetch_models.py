#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="zhouyik/Qwen3-VL-4B-SAMTok-co")
    parser.add_argument("--local-dir", default="/mnt/pfs/xiaoyicheng/models/Qwen3-VL-4B-SAMTok-co")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    last_error = None
    for attempt in range(1, 4):
        try:
            snapshot_download(
                repo_id=args.repo_id,
                local_dir=args.local_dir,
                local_dir_use_symlinks=False,
                max_workers=1,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            print(f"[attempt {attempt}/3] failed: {exc}")
            time.sleep(5 * attempt)
    if last_error is not None:
        raise last_error
    print(f"Downloaded {args.repo_id} -> {args.local_dir}")


if __name__ == "__main__":
    main()

