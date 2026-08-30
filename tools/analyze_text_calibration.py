"""Compare matched text-calibration outputs with a paired bootstrap."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_records(path: str) -> dict[str, dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(record["id"]): record for record in payload.get("records", [])}


def row_class(record: dict[str, object]) -> str:
    explicit = record.get("class")
    if explicit:
        return str(explicit)
    row_id = str(record.get("id", ""))
    if row_id.startswith("positive-"):
        return "positive"
    if row_id.startswith("negative-"):
        return "negative"
    return "unknown"


def correct(record: dict[str, object]) -> int:
    return int(bool(record["abstention"]) == bool(record["target_abstention"]))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    index = round((len(values) - 1) * probability)
    return sorted(values)[index]


def summarize(
    ids: list[str],
    baseline: dict[str, dict[str, object]],
    candidate: dict[str, dict[str, object]],
    *,
    bootstrap_samples: int,
    rng: random.Random,
) -> dict[str, object]:
    baseline_scores = [correct(baseline[row_id]) for row_id in ids]
    candidate_scores = [correct(candidate[row_id]) for row_id in ids]
    deltas = [right - left for left, right in zip(baseline_scores, candidate_scores)]
    n = len(ids)
    bootstrap = [
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(bootstrap_samples)
    ] if n else []
    return {
        "num_samples": n,
        "baseline_accuracy": sum(baseline_scores) / max(n, 1),
        "candidate_accuracy": sum(candidate_scores) / max(n, 1),
        "paired_delta": sum(deltas) / max(n, 1),
        "bootstrap_ci95": [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)],
        "discordant": {
            "baseline_only_correct": sum(left == 1 and right == 0 for left, right in zip(baseline_scores, candidate_scores)),
            "candidate_only_correct": sum(left == 0 and right == 1 for left, right in zip(baseline_scores, candidate_scores)),
        },
    }


def main() -> None:
    args = parse_args()
    baseline = load_records(args.baseline)
    candidate = load_records(args.candidate)
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:10]
        missing_baseline = sorted(set(candidate) - set(baseline))[:10]
        raise ValueError(
            "record IDs do not match: "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )
    ids = sorted(baseline)
    rng = random.Random(args.seed)
    report = {
        "baseline": args.baseline,
        "candidate": args.candidate,
        "seed": args.seed,
        "bootstrap_samples": args.bootstrap_samples,
        "slices": {},
    }
    for name in ("all", "positive", "negative", "unknown"):
        slice_ids = ids if name == "all" else [row_id for row_id in ids if row_class(baseline[row_id]) == name]
        if slice_ids:
            report["slices"][name] = summarize(
                slice_ids,
                baseline,
                candidate,
                bootstrap_samples=args.bootstrap_samples,
                rng=rng,
            )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
