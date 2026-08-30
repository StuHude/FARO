#!/usr/bin/env python3
"""Close PV-FEPO from its pre-registered training-only support gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def decide(metrics: dict) -> dict:
    steps = metrics.get("steps") or []
    fractions = [
        float(step["paired_view_joint_positive_fraction"])
        for step in steps
        if "paired_view_joint_positive_fraction" in step
    ]
    correlations = [
        float(step["paired_view_reward_correlation"])
        for step in steps
        if "paired_view_reward_correlation" in step
    ]
    finite = bool(correlations) and all(math.isfinite(value) for value in correlations)
    mean_fraction = sum(fractions) / len(fractions) if fractions else float("nan")
    support_passed = bool(fractions) and math.isfinite(mean_fraction) and mean_fraction >= 0.20
    contract_passed = (
        metrics.get("status") == "finished"
        and int(metrics.get("steps_completed", 0)) >= 10
        and int(metrics.get("optimizer_updates_completed", 0)) >= 10
        and len(fractions) >= 10
    )
    return {
        "candidate": "PV-FEPO",
        "decision": "open" if contract_passed and finite and support_passed else "closed_training_gate",
        "reason": "joint_positive_fraction_below_preregistered_0.20"
        if contract_passed and finite and not support_passed
        else "training_contract_or_finite_correlation_not_satisfied",
        "training_contract_passed": contract_passed,
        "correlation_finite": finite,
        "joint_positive_fraction_threshold": 0.20,
        "joint_positive_fraction_mean": mean_fraction,
        "joint_positive_fraction_min": min(fractions) if fractions else None,
        "joint_positive_fraction_max": max(fractions) if fractions else None,
        "num_optimizer_records": len(fractions),
        "source_stage": metrics.get("stage"),
        "holdout_used": False,
        "transform_or_threshold_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    result = decide(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
