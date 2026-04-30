#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval
PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python

while true; do
  if [[ -f "$ROOT/refcoco/metric.log" && \
        -f "$ROOT/dlc_official/eval.json" && \
        -f "$ROOT/split/relation.json" && \
        -f "$ROOT/split/geometry.json" && \
        -f "$ROOT/split/semantic.json" && \
        -f "$ROOT/split/overall.json" && \
        -f "$ROOT/split/dlc_reward.json" ]]; then
    break
  fi
  sleep 60
done

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/caption_calibration_eval")

def parse_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    line = [x.strip() for x in text.splitlines() if "REC AP_50:" in x][-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

ap50, ciou = parse_metric(root / "refcoco" / "metric.log")
relation = json.load(open(root / "split" / "relation.json", "r", encoding="utf-8"))
geometry = json.load(open(root / "split" / "geometry.json", "r", encoding="utf-8"))
semantic = json.load(open(root / "split" / "semantic.json", "r", encoding="utf-8"))
overall = json.load(open(root / "split" / "overall.json", "r", encoding="utf-8"))
dlc_reward = json.load(open(root / "split" / "dlc_reward.json", "r", encoding="utf-8"))
dlc_official = json.load(open(root / "dlc_official" / "eval.json", "r", encoding="utf-8"))

summary = {
    "refcoco_val_ap50": ap50,
    "refcoco_val_ciou": ciou,
    "relation_ciou": relation["mean_ciou"],
    "geometry_ciou": geometry["mean_ciou"],
    "semantic_reward": semantic["mean_reward"],
    "overall_refseg_ciou": overall["mean_ciou"],
    "dlc_reward": dlc_reward["mean_reward"],
    "dlc_official_avg_pos": dlc_official["avg_pos"],
    "dlc_official_avg_neg": dlc_official["avg_neg"],
    "dlc_official_avg": dlc_official["avg"],
}

(root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(root / "summary.json")
PY
