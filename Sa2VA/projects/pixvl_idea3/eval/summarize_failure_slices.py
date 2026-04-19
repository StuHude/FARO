from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--relation", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--refseg-overall", default=None)
    parser.add_argument("--maskcap-overall", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_json(path: str | None) -> dict | None:
    if not path:
        return None
    target = Path(path)
    if not target.exists():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    semantic = read_json(args.semantic) or {}
    relation = read_json(args.relation) or {}
    geometry = read_json(args.geometry) or {}
    refseg_overall = read_json(args.refseg_overall)
    maskcap_overall = read_json(args.maskcap_overall)

    summary = {
        "semantic_mean_reward": semantic.get("mean_reward"),
        "relation_mean_ciou": relation.get("mean_ciou"),
        "geometry_mean_ciou": geometry.get("mean_ciou"),
        "refseg_overall_mean_ciou": None if refseg_overall is None else refseg_overall.get("mean_ciou"),
        "maskcap_overall_mean_reward": None if maskcap_overall is None else maskcap_overall.get("mean_reward"),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
