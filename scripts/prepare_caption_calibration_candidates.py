#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-eval-json", required=True)
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--qa-json", required=True)
    parser.add_argument("--class-names-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-neg-threshold", type=float, default=0.65)
    parser.add_argument("--max-recognition-error", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    official = json.load(open(args.official_eval_json, "r", encoding="utf-8"))
    preds = json.load(open(args.pred_json, "r", encoding="utf-8"))
    qa = json.load(open(args.qa_json, "r", encoding="utf-8"))
    class_names = json.load(open(args.class_names_json, "r", encoding="utf-8"))

    rows = []
    for key, info in official["details"].items():
        recognition_error = 0 if info.get("recognition_result") else 1
        score_neg = info.get("score_neg")
        if recognition_error > args.max_recognition_error:
            continue
        if score_neg is None or score_neg > args.max_neg_threshold:
            continue
        rows.append(
            {
                "id": key,
                "pred_caption": preds.get(key, info.get("pred", "")),
                "class_name": class_names.get(key),
                "qa": qa.get(key, []),
                "recognition_result": info.get("recognition_result"),
                "score": info.get("score"),
                "score_pos": info.get("score_pos"),
                "score_neg": score_neg,
                "neg_valid_num": info.get("neg_valid_num"),
                "details_negatives": info.get("details_negatives"),
                "details_positives": info.get("details_positives"),
                "details_recognition": info.get("details_recognition"),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out} rows={len(rows)}")


if __name__ == "__main__":
    main()
