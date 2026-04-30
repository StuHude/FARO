#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

COLOR_WORDS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "gray", "grey",
    "orange", "pink", "purple", "gold", "silver", "beige", "tan",
}
MATERIAL_WORDS = {
    "wood", "wooden", "metal", "metallic", "plastic", "glass", "fabric", "cloth",
    "leather", "paper", "rubber", "stone", "concrete", "ceramic",
}
PART_WORDS = {
    "handle", "wheel", "door", "window", "head", "tail", "wing", "leg", "arm",
    "face", "nose", "eye", "ear", "roof", "strap", "lid", "screen", "seat",
}
REL_WORDS = {
    "left", "right", "behind", "front", "next", "between", "under", "over",
    "near", "beside", "above", "below", "on", "in", "holding",
}
UNCERTAINTY_WORDS = {
    "maybe", "possibly", "likely", "appears", "seems", "suggests", "probably",
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def ratio_count(items: list[str], vocab: set[str]) -> float:
    if not items:
        return 0.0
    return sum(1 for token in items if token in vocab) / len(items)


def compute_stats(path: Path) -> dict:
    data = json.load(open(path, "r", encoding="utf-8"))
    details = data["details"]
    vals = list(details.values())
    recog_true = [x for x in vals if x.get("recognition_result") is True]
    recog_false = [x for x in vals if x.get("recognition_result") is not True]
    neg_true = [x["score_neg"] for x in recog_true if x.get("score_neg") is not None]
    pos_true = [x["score_pos"] for x in recog_true if x.get("score_pos") is not None]

    captions = [x.get("pred", "") for x in vals]
    toks = [tokenize(c) for c in captions]
    lengths = [len(t) for t in toks]
    repeated_prefix_rate = 0.0
    if toks:
        prefixes = [" ".join(t[:8]) for t in toks]
        repeated_prefix_rate = 1.0 - len(set(prefixes)) / len(prefixes)

    return {
        "num_samples": len(vals),
        "recognition_error_rate": len(recog_false) / len(vals) if vals else 0.0,
        "avg_pos_all": data["avg_pos"],
        "avg_neg_all": data["avg_neg"],
        "avg_all": data["avg"],
        "avg_pos_recognition_true": mean(pos_true),
        "avg_neg_recognition_true": mean(neg_true),
        "mean_caption_length": mean(lengths),
        "color_token_ratio": mean([ratio_count(t, COLOR_WORDS) for t in toks]),
        "material_token_ratio": mean([ratio_count(t, MATERIAL_WORDS) for t in toks]),
        "part_token_ratio": mean([ratio_count(t, PART_WORDS) for t in toks]),
        "relation_token_ratio": mean([ratio_count(t, REL_WORDS) for t in toks]),
        "uncertainty_token_ratio": mean([ratio_count(t, UNCERTAINTY_WORDS) for t in toks]),
        "repeated_prefix_rate": repeated_prefix_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("pairs", nargs="+", help="name=path_to_eval_json")
    args = parser.parse_args()

    summary = {}
    for pair in args.pairs:
        name, raw_path = pair.split("=", 1)
        summary[name] = compute_stats(Path(raw_path))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
