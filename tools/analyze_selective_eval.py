"""Paired analysis for mask-or-null selective grounding evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.bootstrap_metrics import bootstrap_paired_delta


def records(path: Path, section: str) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload[section]["records"]}


def summarize(rows: list[dict]) -> dict[str, float]:
    positive = [row for row in rows if row["truth_exists"]]
    negative = [row for row in rows if not row["truth_exists"]]
    boundary = [float(row["boundary_iou"]) for row in positive if row.get("boundary_iou") is not None]
    return {
        "selective_utility": sum(float(row["ciou"]) for row in rows) / max(len(rows), 1),
        "positive_ciou": sum(float(row["ciou"]) for row in positive) / max(len(positive), 1),
        "positive_mask_rate": sum(bool(row["pred_exists"]) for row in positive) / max(len(positive), 1),
        "no_target_explicit_recall": sum(bool(row["explicit_null"]) for row in negative) / max(len(negative), 1),
        "invalid_output_rate": sum(not (row["valid_mask_tokens"] or row["explicit_null"]) for row in rows) / max(len(rows), 1),
        "positive_boundary_iou": sum(boundary) / max(len(boundary), 1),
    }


def summarize_slices(rows: list[dict]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = {"all": rows}
    for row in rows:
        metadata = row.get("slice_metadata") or {}
        for name in ("small", "thin", "boundary_hard"):
            if metadata.get(name):
                groups.setdefault(name, []).append(row)
    report = {}
    for name, group in sorted(groups.items()):
        positive = [row for row in group if row.get("truth_exists")]
        boundary = [float(row["boundary_iou"]) for row in positive if row.get("boundary_iou") is not None]
        report[name] = {
            "num_samples": len(group),
            "positive_num_samples": len(positive),
            "mean_ciou": sum(float(row["ciou"]) for row in group) / max(len(group), 1),
            "positive_ciou": sum(float(row["ciou"]) for row in positive) / max(len(positive), 1),
            "positive_boundary_iou": sum(boundary) / max(len(boundary), 1),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--section", default="refseg_overall")
    parser.add_argument("--repeats", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noninferiority-margin", type=float, default=0.01)
    parser.add_argument("--min-utility-delta", type=float, default=None)
    parser.add_argument(
        "--min-utility-ci-lower",
        type=float,
        default=None,
        help="Require the paired utility bootstrap CI lower bound to meet this threshold.",
    )
    parser.add_argument("--require-utility-ci-positive", action="store_true")
    parser.add_argument("--min-positive-ciou-delta", type=float, default=None)
    parser.add_argument(
        "--min-positive-ciou-ci-lower",
        type=float,
        default=None,
        help="Require the paired positive-cIoU bootstrap CI lower bound to meet this threshold.",
    )
    parser.add_argument("--positive-ciou-ci-positive-alternative", action="store_true")
    parser.add_argument("--min-negative-ci-lower", type=float, default=None)
    parser.add_argument("--min-positive-mask-rate", type=float, default=0.95)
    parser.add_argument("--max-invalid-output-rate", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    base, candidate = records(args.base, args.section), records(args.candidate, args.section)
    ids = sorted(set(base) & set(candidate))
    if len(ids) != len(base) or len(ids) != len(candidate):
        raise ValueError(f"Evaluation IDs differ: base={len(base)} candidate={len(candidate)} paired={len(ids)}")
    positive_ids = [key for key in ids if base[key]["truth_exists"]]
    negative_ids = [key for key in ids if not base[key]["truth_exists"]]

    all_ci = bootstrap_paired_delta(
        [base[key]["ciou"] for key in ids],
        [candidate[key]["ciou"] for key in ids],
        repeats=args.repeats,
        seed=args.seed,
    )
    positive_ci = bootstrap_paired_delta(
        [base[key]["ciou"] for key in positive_ids],
        [candidate[key]["ciou"] for key in positive_ids],
        repeats=args.repeats,
        seed=args.seed + 1,
    )
    negative_recall = bootstrap_paired_delta(
        [float(base[key]["explicit_null"]) for key in negative_ids],
        [float(candidate[key]["explicit_null"]) for key in negative_ids],
        repeats=args.repeats,
        seed=args.seed + 2,
    )
    discordant = {
        "candidate_only_correct": sum(not base[key]["explicit_null"] and candidate[key]["explicit_null"] for key in negative_ids),
        "base_only_correct": sum(base[key]["explicit_null"] and not candidate[key]["explicit_null"] for key in negative_ids),
    }
    candidate_summary = summarize([candidate[key] for key in ids])
    # Keep the historical gate for reproducibility, but expose a CI-based gate
    # for promotion decisions.  A positive bootstrap mean alone is not enough
    # evidence for a paired improvement, especially on the 256-row subsets.
    ci_corrected_gate = (
        all_ci["ci95"][0] >= (args.min_utility_ci_lower if args.min_utility_ci_lower is not None else 0.0)
        and positive_ci["ci95"][0] >= (
            args.min_positive_ciou_ci_lower
            if args.min_positive_ciou_ci_lower is not None
            else -abs(args.noninferiority_margin)
        )
        and negative_recall["ci95"][0] >= (
            args.min_negative_ci_lower
            if args.min_negative_ci_lower is not None
            else -abs(args.noninferiority_margin)
        )
        and candidate_summary["positive_mask_rate"] >= args.min_positive_mask_rate
        and candidate_summary["invalid_output_rate"] <= args.max_invalid_output_rate
    )
    if args.min_utility_delta is None:
        promotion_gate = (
            negative_recall["mean"] > 0.0
            and positive_ci["ci95"][0] >= -abs(args.noninferiority_margin)
            and candidate_summary["positive_mask_rate"] >= args.min_positive_mask_rate
            and candidate_summary["invalid_output_rate"] <= args.max_invalid_output_rate
        )
        gate_definition = "legacy"
    else:
        positive_gain = (
            positive_ci["mean"] >= (args.min_positive_ciou_delta or 0.0)
            or (
                args.positive_ciou_ci_positive_alternative
                and positive_ci["ci95"][0] > 0.0
            )
        )
        utility_ci_ok = (
            args.min_utility_ci_lower is None
            or all_ci["ci95"][0] >= args.min_utility_ci_lower
        )
        positive_ci_lower_ok = (
            args.min_positive_ciou_ci_lower is None
            or positive_ci["ci95"][0] >= args.min_positive_ciou_ci_lower
        )
        promotion_gate = (
            all_ci["mean"] >= args.min_utility_delta
            and (not args.require_utility_ci_positive or all_ci["ci95"][0] > 0.0)
            and utility_ci_ok
            and positive_gain
            and positive_ci_lower_ok
            and (
                args.min_negative_ci_lower is None
                or negative_recall["ci95"][0] > args.min_negative_ci_lower
            )
            and candidate_summary["positive_mask_rate"] >= args.min_positive_mask_rate
            and candidate_summary["invalid_output_rate"] <= args.max_invalid_output_rate
        )
        gate_definition = {
            "min_utility_delta": args.min_utility_delta,
            "min_utility_ci_lower": args.min_utility_ci_lower,
            "require_utility_ci_positive": args.require_utility_ci_positive,
            "min_positive_ciou_delta": args.min_positive_ciou_delta,
            "min_positive_ciou_ci_lower": args.min_positive_ciou_ci_lower,
            "positive_ciou_ci_positive_alternative": args.positive_ciou_ci_positive_alternative,
            "min_negative_ci_lower": args.min_negative_ci_lower,
            "min_positive_mask_rate": args.min_positive_mask_rate,
            "max_invalid_output_rate": args.max_invalid_output_rate,
        }
    report = {
        "num_paired": len(ids),
        "base": summarize([base[key] for key in ids]),
        "candidate": candidate_summary,
        "base_slices": summarize_slices([base[key] for key in ids]),
        "candidate_slices": summarize_slices([candidate[key] for key in ids]),
        "selective_utility_delta": all_ci,
        "positive_ciou_delta": positive_ci,
        "no_target_explicit_recall_delta": negative_recall,
        "no_target_discordant": discordant,
        "positive_ciou_noninferior": positive_ci["ci95"][0] >= -abs(args.noninferiority_margin),
        "promotion_gate": promotion_gate,
        "ci_corrected_promotion_gate": ci_corrected_gate,
        "ci_corrected_promotion_gate_definition": {
            "utility_ci_lower": args.min_utility_ci_lower if args.min_utility_ci_lower is not None else 0.0,
            "positive_ciou_ci_lower": args.min_positive_ciou_ci_lower if args.min_positive_ciou_ci_lower is not None else -abs(args.noninferiority_margin),
            "negative_recall_ci_lower": args.min_negative_ci_lower if args.min_negative_ci_lower is not None else -abs(args.noninferiority_margin),
            "min_positive_mask_rate": args.min_positive_mask_rate,
            "max_invalid_output_rate": args.max_invalid_output_rate,
        },
        "promotion_gate_definition": gate_definition,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
