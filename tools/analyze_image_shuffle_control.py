"""Analyze visual dependence of selective mask-or-null predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.bootstrap_metrics import bootstrap_paired_delta
except ModuleNotFoundError:
    from bootstrap_metrics import bootstrap_paired_delta


def records(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["refseg_overall"]["records"]}


def correctness(row: dict) -> float:
    return float(bool(row["explicit_null"]) != bool(row["truth_exists"]))


def summarize(original: dict[str, dict], shuffled: dict[str, dict], seed: int) -> dict:
    ids = sorted(set(original) & set(shuffled))
    if len(ids) != len(original) or len(ids) != len(shuffled):
        raise ValueError("original and shuffled evaluation IDs differ")
    positives = [key for key in ids if original[key]["truth_exists"]]
    negatives = [key for key in ids if not original[key]["truth_exists"]]
    for key in ids:
        if original[key]["truth_exists"] != shuffled[key]["truth_exists"]:
            raise ValueError(f"label changed under image shuffle: {key}")

    original_correct = [correctness(original[key]) for key in ids]
    shuffled_correct = [correctness(shuffled[key]) for key in ids]
    return {
        "num_paired": len(ids),
        "original_existence_accuracy": sum(original_correct) / len(ids),
        "shuffled_existence_accuracy": sum(shuffled_correct) / len(ids),
        "shuffled_minus_original_accuracy": bootstrap_paired_delta(
            original_correct,
            shuffled_correct,
            repeats=20_000,
            seed=seed,
        ),
        "positive_abstain_rate": {
            "original": sum(bool(original[key]["explicit_null"]) for key in positives) / len(positives),
            "shuffled": sum(bool(shuffled[key]["explicit_null"]) for key in positives) / len(positives),
        },
        "no_target_recall": {
            "original": sum(bool(original[key]["explicit_null"]) for key in negatives) / len(negatives),
            "shuffled": sum(bool(shuffled[key]["explicit_null"]) for key in negatives) / len(negatives),
        },
        "prediction_flip_rate": sum(
            bool(original[key]["explicit_null"]) != bool(shuffled[key]["explicit_null"])
            for key in ids
        ) / len(ids),
        "positive_prediction_flips": sum(
            bool(original[key]["explicit_null"]) != bool(shuffled[key]["explicit_null"])
            for key in positives
        ),
        "negative_prediction_flips": sum(
            bool(original[key]["explicit_null"]) != bool(shuffled[key]["explicit_null"])
            for key in negatives
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-original", type=Path, required=True)
    parser.add_argument("--base-shuffled", type=Path, required=True)
    parser.add_argument("--candidate-original", type=Path, required=True)
    parser.add_argument("--candidate-shuffled", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = {
        "base": summarize(records(args.base_original), records(args.base_shuffled), args.seed),
        "candidate": summarize(
            records(args.candidate_original),
            records(args.candidate_shuffled),
            args.seed + 1,
        ),
        "note": "Mask cIoU on shuffled images is intentionally not reported.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
