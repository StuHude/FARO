#!/usr/bin/env python
"""Analyze sharded SAMTok action margins against deterministic null flips."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from tools.action_margin_contract import (
    EXPECTED_ROWS,
    assert_disjoint_source_images,
    load_aligned_eval,
    validate_margin_shards,
    validate_policy_alignment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--margin-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--continued-eval", required=True)
    parser.add_argument("--candidate-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def auc(labels: list[bool], scores: list[float]) -> float:
    positives = [score for label, score in zip(labels, scores) if label]
    negatives = [score for label, score in zip(labels, scores) if not label]
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    if not positives or not negatives:
        raise ValueError("AUROC requires both labels")
    return wins / (len(positives) * len(negatives))


def fit_youden_threshold(records: list[dict]) -> tuple[float, float]:
    candidates = sorted({float(row["margin"]) for row in records})
    best = (-1.0, candidates[0])
    for threshold in candidates:
        positives = sum(bool(row["no_target"]) for row in records)
        negatives = len(records) - positives
        tpr = sum(bool(row["no_target"]) and float(row["margin"]) >= threshold for row in records) / positives
        tnr = sum((not bool(row["no_target"])) and float(row["margin"]) < threshold for row in records) / negatives
        score = tpr + tnr - 1.0
        if score > best[0]:
            best = (score, threshold)
    return best[1], best[0]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def permutation_p_less(
    left: list[float], right: list[float], repeats: int, seed: int
) -> tuple[float, float]:
    observed = mean(left) - mean(right)
    values = left + right
    left_size = len(left)
    rng = random.Random(seed)
    at_most = 1
    for _ in range(repeats):
        rng.shuffle(values)
        delta = mean(values[:left_size]) - mean(values[left_size:])
        at_most += int(delta <= observed)
    return observed, at_most / (repeats + 1)


def permutation_p_greater(
    left: list[float], right: list[float], repeats: int, seed: int
) -> tuple[float, float]:
    observed, p_value = permutation_p_less(
        [-value for value in left], [-value for value in right], repeats, seed
    )
    return -observed, p_value


def main() -> None:
    args = parse_args()
    root = Path(args.margin_root)
    loaded = {
        (split, policy): validate_margin_shards(root, split, policy, args.num_shards)
        for split in ("train", "holdout")
        for policy in ("continued", "candidate")
    }
    margins = {key: value[0] for key, value in loaded.items()}
    for split in ("train", "holdout"):
        validate_policy_alignment(
            margins[(split, "continued")], margins[(split, "candidate")], split
        )
        if loaded[(split, "continued")][1].sha256 != loaded[(split, "candidate")][1].sha256:
            raise ValueError(f"cross-policy schema SHA mismatch for {split}")
        if loaded[(split, "continued")][2]["scoring"] != loaded[(split, "candidate")][2]["scoring"]:
            raise ValueError(f"cross-policy scoring metadata mismatch for {split}")
        if loaded[(split, "continued")][2]["model"] != loaded[(split, "candidate")][2]["model"]:
            raise ValueError(f"cross-policy base-model metadata mismatch for {split}")
    for policy in ("continued", "candidate"):
        train_metadata = loaded[("train", policy)][2]
        holdout_metadata = loaded[("holdout", policy)][2]
        for key in (
            "model",
            "adapter",
            "adapter_metrics_sha256",
            "adapter_provenance_sha256",
            "scoring",
        ):
            if train_metadata[key] != holdout_metadata[key]:
                raise ValueError(f"train/holdout {key} metadata mismatch for {policy}")
    train_schema = loaded[("train", "continued")][1]
    holdout_schema = loaded[("holdout", "continued")][1]
    assert_disjoint_source_images(train_schema, holdout_schema)
    indexed = {
        key: {str(row["id"]): row for row in rows} for key, rows in margins.items()
    }
    train_continued = margins[("train", "continued")]
    holdout_continued = margins[("holdout", "continued")]
    holdout_candidate = indexed[("holdout", "candidate")]
    threshold, train_youden = fit_youden_threshold(train_continued)
    holdout_auc = auc(
        [bool(row["no_target"]) for row in holdout_continued],
        [float(row["margin"]) for row in holdout_continued],
    )

    continued_eval = load_aligned_eval(args.continued_eval, holdout_schema)
    candidate_eval = load_aligned_eval(args.candidate_eval, holdout_schema)
    flip_ids: list[str] = []
    retained_ids: list[str] = []
    repair_ids: list[str] = []
    persistent_error_ids: list[str] = []
    for row in holdout_continued:
        row_id = str(row["id"])
        if not bool(row["no_target"]):
            continue
        base_correct = bool(continued_eval[row_id]["explicit_null"])
        candidate_correct = bool(candidate_eval[row_id]["explicit_null"])
        if base_correct and not candidate_correct:
            flip_ids.append(row_id)
        elif base_correct and candidate_correct:
            retained_ids.append(row_id)
        elif not base_correct and candidate_correct:
            repair_ids.append(row_id)
        else:
            persistent_error_ids.append(row_id)

    continued_index = indexed[("holdout", "continued")]
    deltas = {
        row_id: float(holdout_candidate[row_id]["margin"])
        - float(continued_index[row_id]["margin"])
        for row_id in flip_ids + retained_ids + repair_ids + persistent_error_ids
    }
    if flip_ids and retained_ids:
        observed, p_value = permutation_p_less(
            [deltas[row_id] for row_id in flip_ids],
            [deltas[row_id] for row_id in retained_ids],
            args.permutations,
            args.seed,
        )
    else:
        observed, p_value = None, 1.0
    predicted_flip_ids = [
        row_id
        for row_id in flip_ids
        if float(continued_index[row_id]["margin"]) >= threshold
        and float(holdout_candidate[row_id]["margin"]) < threshold
    ]
    flips_predicted = len(predicted_flip_ids)
    repair_reference_ids = retained_ids + persistent_error_ids
    if repair_ids and repair_reference_ids:
        repair_observed, repair_p_value = permutation_p_greater(
            [deltas[row_id] for row_id in repair_ids],
            [deltas[row_id] for row_id in repair_reference_ids],
            args.permutations,
            args.seed + 1,
        )
    else:
        repair_observed, repair_p_value = None, 1.0
    predicted_repair_ids = [
        row_id
        for row_id in repair_ids
        if float(continued_index[row_id]["margin"]) < threshold
        and float(holdout_candidate[row_id]["margin"]) >= threshold
    ]
    gate = {
        "auc_at_least_0.80": holdout_auc >= 0.80,
        "flip_shift_one_sided_p_below_0.05": p_value < 0.05,
        "at_least_five_flips_predicted": flips_predicted >= 5,
    }
    payload = {
        "num_rows": EXPECTED_ROWS,
        "train_schema_sha256": train_schema.sha256,
        "holdout_schema_sha256": holdout_schema.sha256,
        "continued_adapter_metrics_sha256": loaded[("holdout", "continued")][2][
            "adapter_metrics_sha256"
        ],
        "continued_adapter_provenance_sha256": loaded[("holdout", "continued")][2][
            "adapter_provenance_sha256"
        ],
        "candidate_adapter_metrics_sha256": loaded[("holdout", "candidate")][2][
            "adapter_metrics_sha256"
        ],
        "candidate_adapter_provenance_sha256": loaded[("holdout", "candidate")][2][
            "adapter_provenance_sha256"
        ],
        "train_fitted_threshold": threshold,
        "train_youden_j": train_youden,
        "continued_holdout_auroc": holdout_auc,
        "num_candidate_only_null_flips": len(flip_ids),
        "flip_ids": flip_ids,
        "retained_null_count": len(retained_ids),
        "flip_minus_retained_margin_shift": observed,
        "one_sided_permutation_p": p_value,
        "flips_predicted_by_train_threshold": flips_predicted,
        "predicted_flip_ids": predicted_flip_ids,
        "num_candidate_only_null_repairs": len(repair_ids),
        "repair_ids": repair_ids,
        "persistent_null_error_count": len(persistent_error_ids),
        "repair_minus_other_margin_shift": repair_observed,
        "repair_one_sided_permutation_p": repair_p_value,
        "repairs_predicted_by_train_threshold": len(predicted_repair_ids),
        "predicted_repair_ids": predicted_repair_ids,
        "gate": gate,
        "promotion_gate": all(gate.values()) and len(flip_ids) >= 5,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
