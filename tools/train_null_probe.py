"""Fit and evaluate a tiny linear null probe on frozen SAMTok features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def is_calibration(row_id: str) -> bool:
    return int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    for path in sorted(Path(args.input_dir).glob("part_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8")).get("records", []))
    rows.sort(key=lambda r: str(r["id"]))
    if not rows:
        raise RuntimeError("no probe features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([r["feature"] for r in rows], dtype=torch.float32, device=device)
    y = torch.tensor([float(r["target_exists"]) for r in rows], dtype=torch.float32, device=device)
    mean, std = x.mean(0, keepdim=True), x.std(0, keepdim=True).clamp_min(1e-5)
    x = (x - mean) / std
    cal = torch.tensor([is_calibration(str(r["id"])) for r in rows], dtype=torch.bool, device=device)
    weight = torch.zeros(x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros((), device=device, requires_grad=True)
    opt = torch.optim.Adam([weight, bias], lr=0.03, weight_decay=1e-3)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x[cal] @ weight + bias, y[cal])
        loss.backward(); opt.step()
    scores = (x @ weight + bias).detach().cpu().tolist()
    cal_rows = [r for r, c in zip(rows, cal.cpu().tolist()) if c]
    cal_scores = [s for s, c in zip(scores, cal.cpu().tolist()) if c]
    candidates = sorted(set(cal_scores))
    def ba(threshold, subset, subset_scores):
        recalls = []
        for label in (True, False):
            pairs = [(r, s) for r, s in zip(subset, subset_scores) if bool(r["target_exists"]) == label]
            if pairs:
                recalls.append(sum(int((s >= threshold) == label) for r, s in pairs) / len(pairs))
        return sum(recalls) / max(len(recalls), 1)
    threshold = max(candidates, key=lambda t: ba(t, cal_rows, cal_scores))
    for row, score in zip(rows, scores):
        row.pop("feature", None)
        row["probe_score"] = score
        row["predicted_exists"] = bool(score >= threshold)
    hold = [r for r in rows if not is_calibration(str(r["id"]))]
    result = {
        "num_samples": len(rows),
        "threshold": threshold,
        "calibration_balanced_accuracy": ba(threshold, cal_rows, cal_scores),
        "holdout_balanced_accuracy": ba(threshold, hold, [r["probe_score"] for r in hold]),
        "records": rows,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
