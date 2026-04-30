#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PY=/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=4
CONFIG=/mnt/pfs/xiaoyicheng/BRIDGE-OPD/Sa2VA/projects/pixvl_idea3/configs/idea3_semcovcal_routed_opd_rl_8gpu_500.py
ADAPTER=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/semcovcal_routed_opd_rl_8gpu_100_run7_fast/checkpoint-step-100/adapter
SCHEMA_ROOT=/mnt/pfs/xiaoyicheng/data/pixvl_idea3/schemas
OUT=/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/run7_ckpt100_eval/split
LOG=$OUT/launcher.log
mkdir -p "$OUT"

log(){ printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG" ; }

run_refseg_split() {
  name="$1"
  schema="$2"
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name"/part*.json
  PIDS=()
  for rank in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES="$rank" "$PY" -m projects.pixvl_idea1.eval.eval_refseg \
      --config "$CONFIG" \
      --adapter-path "$ADAPTER" \
      --schema-file "$schema" \
      --output "$OUT/$name/part${rank}.json" \
      --task-id "$rank" \
      --num-tasks 8 \
      > "$OUT/${name}_gpu${rank}.log" 2>&1 &
    PIDS+=($!)
    sleep 3
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY > "$OUT/${name}.json"
import glob, json
results=[]
for p in sorted(glob.glob("$OUT/$name/part*.json")):
    results.extend(json.load(open(p)).get("results", []))
num=len(results)
mean=sum(x["ciou"] for x in results)/max(num,1)
ap50=sum(1 for x in results if x["ciou"] >= 0.5)/max(num,1)
json.dump({"num_samples":num,"mean_ciou":mean,"ap50":ap50}, open("/dev/stdout","w"), indent=2)
PY
}

run_maskcap_split() {
  name="$1"
  schema="$2"
  mkdir -p "$OUT/$name"
  rm -f "$OUT/$name"/part*.json
  PIDS=()
  for rank in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES="$rank" "$PY" -m projects.pixvl_idea1.eval.eval_dlc \
      --config "$CONFIG" \
      --adapter-path "$ADAPTER" \
      --schema-file "$schema" \
      --output "$OUT/$name/part${rank}.json" \
      --task-id "$rank" \
      --num-tasks 8 \
      > "$OUT/${name}_gpu${rank}.log" 2>&1 &
    PIDS+=($!)
    sleep 3
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid"
  done
  CUDA_VISIBLE_DEVICES=0 "$PY" - <<PY > "$OUT/${name}.json"
import glob, json
results=[]
for p in sorted(glob.glob("$OUT/$name/part*.json")):
    results.extend(json.load(open(p)).get("results", []))
num=len(results)
mean=sum(x["reward"] for x in results)/max(num,1)
json.dump({"num_samples":num,"mean_reward":mean}, open("/dev/stdout","w"), indent=2)
PY
}

log "start split semantic"
run_maskcap_split semantic "$SCHEMA_ROOT/semantic_slice_eval.jsonl"
log "start split relation"
run_refseg_split relation "$SCHEMA_ROOT/relation_slice_eval.jsonl"
log "start split geometry"
run_refseg_split geometry "$SCHEMA_ROOT/geometry_slice_eval.jsonl"
log "start split refseg_overall"
run_refseg_split refseg_overall "$SCHEMA_ROOT/refseg_val_routed.jsonl"
log "start split dlc_reward"
run_maskcap_split dlc_reward "$SCHEMA_ROOT/dlc_eval.jsonl"

CUDA_VISIBLE_DEVICES=0 "$PY" -m projects.pixvl_idea3.eval.summarize_failure_slices \
  --semantic "$OUT/semantic.json" \
  --relation "$OUT/relation.json" \
  --geometry "$OUT/geometry.json" \
  --refseg-overall "$OUT/refseg_overall.json" \
  --maskcap-overall "$OUT/dlc_reward.json" \
  --output "$OUT/summary.json" \
  > "$OUT/summary.log" 2>&1

log "split eval done"
