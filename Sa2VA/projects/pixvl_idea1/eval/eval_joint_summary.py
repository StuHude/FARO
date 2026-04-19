from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refseg", required=True)
    parser.add_argument("--maskcap", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refseg = json.loads(Path(args.refseg).read_text(encoding="utf-8"))
    maskcap = json.loads(Path(args.maskcap).read_text(encoding="utf-8"))
    summary = {
        "refseg_mean_ciou": refseg.get("mean_ciou"),
        "maskcap_mean_reward": maskcap.get("mean_reward"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
