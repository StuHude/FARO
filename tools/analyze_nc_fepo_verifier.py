"""Analyze a frozen SAMTok NC-FEPO target/null margin run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _is_calibration(row_id: str, fraction: float) -> bool:
    bucket = int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < fraction


def _balanced_accuracy(rows, threshold: float) -> float:
    recalls = []
    for target in (True, False):
        subset = [r for r in rows if bool(r["target_exists"]) == target]
        if subset:
            recalls.append(sum(int((float(r["target_margin"]) >= threshold) == target) for r in subset) / len(subset))
    return sum(recalls) / max(len(recalls), 1)


def select_threshold(rows) -> float:
    margins = sorted({float(r["target_margin"]) for r in rows})
    candidates = [margins[0] - 1e-6] + [(a + b) / 2 for a, b in zip(margins, margins[1:])] + [margins[-1] + 1e-6]
    return max(candidates, key=lambda t: (_balanced_accuracy(rows, t), -abs(t)))


def summarize(rows, threshold: float) -> dict:
    result = {"num_samples": len(rows), "threshold": threshold, "balanced_accuracy": _balanced_accuracy(rows, threshold)}
    for source in sorted({str(r.get("source")) for r in rows}):
        subset = [r for r in rows if str(r.get("source")) == source]
        result[source] = {
            "n": len(subset),
            "accuracy": sum(int((float(r["target_margin"]) >= threshold) == bool(r["target_exists"])) for r in subset) / len(subset),
            "mean_margin": sum(float(r["target_margin"]) for r in subset) / len(subset),
        }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output")
    p.add_argument("--calibration-fraction", type=float, default=0.5)
    args = p.parse_args()
    if not 0 < args.calibration_fraction < 1:
        raise ValueError("calibration-fraction must be between 0 and 1")
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows = payload["records"]
    calibration = [r for r in rows if _is_calibration(str(r["id"]), args.calibration_fraction)]
    holdout = [r for r in rows if not _is_calibration(str(r["id"]), args.calibration_fraction)]
    threshold = select_threshold(calibration)
    result = {
        "calibration": summarize(calibration, threshold),
        "holdout": summarize(holdout, threshold),
        "zero_threshold_holdout": summarize(holdout, 0.0),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
