#!/usr/bin/env bash
set -euo pipefail

cd /mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA
export PYTHONPATH=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA:${PYTHONPATH:-}
export PIXVL_TEXT_SIM_DEVICE=cpu
export PIXVL_TEXT_SIM_LOCAL_ONLY=1
export TORCHDYNAMO_DISABLE=1

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_sft_8gpu/adapter
CONFIG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_recognition_negsup_sft_8gpu.py
SUBSET_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/eval_subsets_formal_2000
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/recognition_negsup_eval/split

mkdir -p "$OUT"

shard_refseg() {
  local schema="$1"
  local prefix="$2"
  for task in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
      --config "$CONFIG" \
      --adapter-path "$ADAPTER" \
      --schema-file "$schema" \
      --output "$OUT/${prefix}.part${task}.json" \
      --task-id "$task" \
      --num-tasks 8 &
  done
  wait
  "$PY" - <<PY
import json
from pathlib import Path
root = Path("$OUT")
parts = sorted(root.glob("${prefix}.part*.json"))
total_n = 0
total_sum = 0.0
all_results = []
for p in parts:
    d = json.load(open(p, "r", encoding="utf-8"))
    n = d["num_samples"]
    total_n += n
    total_sum += d["mean_ciou"] * n
    all_results.extend(d.get("results", []))
payload = {"num_samples": total_n, "mean_ciou": total_sum / max(total_n, 1), "results": all_results}
(root / "${prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

shard_maskcap() {
  local schema="$1"
  local prefix="$2"
  for task in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$task" "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
      --config "$CONFIG" \
      --adapter-path "$ADAPTER" \
      --schema-file "$schema" \
      --output "$OUT/${prefix}.part${task}.json" \
      --task-id "$task" \
      --num-tasks 8 &
  done
  wait
  "$PY" - <<PY
import json
from pathlib import Path
root = Path("$OUT")
parts = sorted(root.glob("${prefix}.part*.json"))
total_n = 0
total_sum = 0.0
all_results = []
for p in parts:
    d = json.load(open(p, "r", encoding="utf-8"))
    n = d["num_samples"]
    total_n += n
    total_sum += d["mean_reward"] * n
    all_results.extend(d.get("results", []))
payload = {"num_samples": total_n, "mean_reward": total_sum / max(total_n, 1), "results": all_results}
(root / "${prefix}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

shard_refseg "$SUBSET_ROOT/relation_2000.jsonl" relation
shard_refseg "$SUBSET_ROOT/geometry_2000.jsonl" geometry
shard_maskcap "$SUBSET_ROOT/semantic_2000.jsonl" semantic
shard_refseg "$SUBSET_ROOT/refseg_val_2000.jsonl" overall
shard_maskcap "$SUBSET_ROOT/dlc_eval_100.jsonl" dlc_reward
