#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8
PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python

while true; do
  if [[ -f "$ROOT/refcoco/no_bucket/metric.log" && \
        -f "$ROOT/refcoco/shuffled/metric.log" && \
        -f "$ROOT/split/no_bucket/relation.json" && \
        -f "$ROOT/split/no_bucket/geometry.json" && \
        -f "$ROOT/split/no_bucket/semantic.json" && \
        -f "$ROOT/split/no_bucket/overall.json" && \
        -f "$ROOT/split/no_bucket/dlc_reward.json" && \
        -f "$ROOT/split/shuffled/relation.json" && \
        -f "$ROOT/split/shuffled/geometry.json" && \
        -f "$ROOT/split/shuffled/semantic.json" && \
        -f "$ROOT/split/shuffled/overall.json" && \
        -f "$ROOT/split/shuffled/dlc_reward.json" && \
        -f "$ROOT/dlc_official/no_bucket/eval.json" && \
        -f "$ROOT/dlc_official/shuffled/eval.json" ]]; then
    break
  fi
  sleep 60
done

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_mixed8")

def parse_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    line = [x.strip() for x in text.splitlines() if "REC AP_50:" in x][-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

summary = {}
for name in ["no_bucket", "shuffled"]:
    ap50, ciou = parse_metric(root / "refcoco" / name / "metric.log")
    split_root = root / "split" / name
    relation = json.load(open(split_root / "relation.json", "r", encoding="utf-8"))
    geometry = json.load(open(split_root / "geometry.json", "r", encoding="utf-8"))
    semantic = json.load(open(split_root / "semantic.json", "r", encoding="utf-8"))
    overall = json.load(open(split_root / "overall.json", "r", encoding="utf-8"))
    dlc_reward = json.load(open(split_root / "dlc_reward.json", "r", encoding="utf-8"))
    dlc_official = json.load(open(root / "dlc_official" / name / "eval.json", "r", encoding="utf-8"))
    summary[name] = {
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

out = root / "summary_ckpt1000.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY

"$PY" /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/analyze_dlc_official_breakdown.py \
  --output "$ROOT/dlc_breakdown_ablations.json" \
  no_bucket="$ROOT/dlc_official/no_bucket/eval.json" \
  shuffled="$ROOT/dlc_official/shuffled/eval.json"

while true; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    sleep 60
  else
    break
  fi
done

CUDA_VISIBLE_DEVICES=0 bash /mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_dlc_calibration_test.sh \
  > /mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/dlc_calibration_test_routed_opd_1500/run.log 2>&1
