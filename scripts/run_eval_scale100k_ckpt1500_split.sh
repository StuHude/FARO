#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
OUT_ROOT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1500_eval_split
SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_2000

CFG_UNIFIED=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_2gpu_unified_opd_rl.py
CFG_ROUTED_RL=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_rl.py
CFG_ROUTED_OPD=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_mvp_scale100k_3gpu_routed_opd_rl.py

ADAPTER_UNIFIED=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_unified_opd_rl/checkpoint-step-1500/adapter
ADAPTER_ROUTED_RL=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_rl/checkpoint-step-1500/adapter
ADAPTER_ROUTED_OPD=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_3gpu_routed_opd_rl/checkpoint-step-1500/adapter

mkdir -p "$OUT_ROOT"
rm -rf "$OUT_ROOT"/*

launch_triplet() {
  local cmd0="$1"
  local cmd1="$2"
  local cmd2="$3"
  bash -lc "$cmd0" &
  p0=$!
  bash -lc "$cmd1" &
  p1=$!
  bash -lc "$cmd2" &
  p2=$!
  wait "$p0"
  wait "$p1"
  wait "$p2"
}

# Wave 1: relation
launch_triplet \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=0 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_UNIFIED --adapter-path $ADAPTER_UNIFIED --schema-file $SUBSET_ROOT/relation_2000.jsonl --output $OUT_ROOT/unified_relation.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=1 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_ROUTED_RL --adapter-path $ADAPTER_ROUTED_RL --schema-file $SUBSET_ROOT/relation_2000.jsonl --output $OUT_ROOT/routed_rl_relation.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=2 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_ROUTED_OPD --adapter-path $ADAPTER_ROUTED_OPD --schema-file $SUBSET_ROOT/relation_2000.jsonl --output $OUT_ROOT/routed_opd_relation.json"

# Wave 2: geometry
launch_triplet \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=0 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_UNIFIED --adapter-path $ADAPTER_UNIFIED --schema-file $SUBSET_ROOT/geometry_2000.jsonl --output $OUT_ROOT/unified_geometry.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=1 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_ROUTED_RL --adapter-path $ADAPTER_ROUTED_RL --schema-file $SUBSET_ROOT/geometry_2000.jsonl --output $OUT_ROOT/routed_rl_geometry.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=2 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_ROUTED_OPD --adapter-path $ADAPTER_ROUTED_OPD --schema-file $SUBSET_ROOT/geometry_2000.jsonl --output $OUT_ROOT/routed_opd_geometry.json"

# Wave 3: semantic
launch_triplet \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=0 && $PY -m projects.pixvl_idea1.eval.eval_dlc --config $CFG_UNIFIED --adapter-path $ADAPTER_UNIFIED --schema-file $SUBSET_ROOT/semantic_2000.jsonl --output $OUT_ROOT/unified_semantic.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=1 && $PY -m projects.pixvl_idea1.eval.eval_dlc --config $CFG_ROUTED_RL --adapter-path $ADAPTER_ROUTED_RL --schema-file $SUBSET_ROOT/semantic_2000.jsonl --output $OUT_ROOT/routed_rl_semantic.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=2 && $PY -m projects.pixvl_idea1.eval.eval_dlc --config $CFG_ROUTED_OPD --adapter-path $ADAPTER_ROUTED_OPD --schema-file $SUBSET_ROOT/semantic_2000.jsonl --output $OUT_ROOT/routed_opd_semantic.json"

# Wave 4: overall refseg
launch_triplet \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=0 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_UNIFIED --adapter-path $ADAPTER_UNIFIED --schema-file $SUBSET_ROOT/refseg_val_2000.jsonl --output $OUT_ROOT/unified_overall.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=1 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_ROUTED_RL --adapter-path $ADAPTER_ROUTED_RL --schema-file $SUBSET_ROOT/refseg_val_2000.jsonl --output $OUT_ROOT/routed_rl_overall.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=2 && $PY -m projects.pixvl_idea1.eval.eval_refseg --config $CFG_ROUTED_OPD --adapter-path $ADAPTER_ROUTED_OPD --schema-file $SUBSET_ROOT/refseg_val_2000.jsonl --output $OUT_ROOT/routed_opd_overall.json"

# Wave 5: dlc reward proxy
launch_triplet \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=0 && $PY -m projects.pixvl_idea1.eval.eval_dlc --config $CFG_UNIFIED --adapter-path $ADAPTER_UNIFIED --schema-file $SUBSET_ROOT/dlc_eval_100.jsonl --output $OUT_ROOT/unified_dlc.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=1 && $PY -m projects.pixvl_idea1.eval.eval_dlc --config $CFG_ROUTED_RL --adapter-path $ADAPTER_ROUTED_RL --schema-file $SUBSET_ROOT/dlc_eval_100.jsonl --output $OUT_ROOT/routed_rl_dlc.json" \
  "cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA && export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:\${PYTHONPATH:-} && export PIXVL_TEXT_SIM_DEVICE=cpu && export PIXVL_TEXT_SIM_LOCAL_ONLY=1 && export CUDA_VISIBLE_DEVICES=2 && $PY -m projects.pixvl_idea1.eval.eval_dlc --config $CFG_ROUTED_OPD --adapter-path $ADAPTER_ROUTED_OPD --schema-file $SUBSET_ROOT/dlc_eval_100.jsonl --output $OUT_ROOT/routed_opd_dlc.json"

"$PY" - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1500_eval")
split_root = Path("/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_ckpt1500_eval_split")

def parse_refcoco_metric(path: Path):
    text = path.read_text(encoding="utf-8")
    line = [x.strip() for x in text.splitlines() if "REC AP_50:" in x][-1]
    ap50 = float(line.split("REC AP_50:")[1].split("|")[0].strip())
    ciou = float(line.split("RES CIoU:")[1].strip())
    return ap50, ciou

summary = {}
for name in ["unified", "routed_rl", "routed_opd_rl"]:
    ap50, ciou = parse_refcoco_metric(root / f"refcoco_{name}" / "metric.log")
    relation = json.load(open(split_root / f"{name}_relation.json"))
    geometry = json.load(open(split_root / f"{name}_geometry.json"))
    semantic = json.load(open(split_root / f"{name}_semantic.json"))
    overall = json.load(open(split_root / f"{name}_overall.json"))
    dlc = json.load(open(split_root / f"{name}_dlc.json"))
    summary[name] = {
        "refcoco_val_ap50": ap50,
        "refcoco_val_ciou": ciou,
        "relation_ciou": relation["mean_ciou"],
        "geometry_ciou": geometry["mean_ciou"],
        "semantic_reward": semantic["mean_reward"],
        "overall_refseg_ciou": overall["mean_ciou"],
        "dlc_reward": dlc["mean_reward"],
    }

out = split_root / "summary_1500.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
