#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval
PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
PACKED_SCRIPT=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/scripts/run_ablation_split_eval_packed.sh

while true; do
  if [[ -f "$ROOT/refcoco_no_bucket/metric.log" && \
        -f "$ROOT/refcoco_shuffled/metric.log" && \
        -f "$ROOT/dlc_official_no_bucket/eval.json" && \
        -f "$ROOT/dlc_official_shuffled/eval.json" ]]; then
    break
  fi
  sleep 60
done

# Stop the currently serialized split relation jobs so the remaining split eval can
# be relaunched in packed parallel mode.
pkill -f "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval/split_no_bucket/relation.json" || true
pkill -f "/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval/split_shuffled/relation.json" || true

bash "$PACKED_SCRIPT" > "$ROOT/packed.log" 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval")
packed = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/ablation_ckpt1000_eval_packed")

def parse_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    line = [x.strip() for x in text.splitlines() if "REC AP_50:" in x][-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

summary = {}
for name in ["no_bucket", "shuffled"]:
    ap50, ciou = parse_metric(root / f"refcoco_{name}" / "metric.log")
    split_root = packed / name
    relation = json.load(open(split_root / "relation.json", "r", encoding="utf-8"))
    geometry = json.load(open(split_root / "geometry.json", "r", encoding="utf-8"))
    semantic = json.load(open(split_root / "semantic.json", "r", encoding="utf-8"))
    overall = json.load(open(split_root / "overall.json", "r", encoding="utf-8"))
    dlc_reward = json.load(open(split_root / "dlc_reward.json", "r", encoding="utf-8"))
    dlc_official = json.load(open(root / f"dlc_official_{name}" / "eval.json", "r", encoding="utf-8"))
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
