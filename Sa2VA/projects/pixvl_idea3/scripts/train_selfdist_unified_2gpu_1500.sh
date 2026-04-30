#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MAIN_PROCESS_PORT:-29581}"
CONFIG_PATH="$REPO_ROOT/projects/pixvl_idea3/configs/idea3_mvp_scale100k_2gpu_unified_opd_rl_selfdist1500.py"
LOG_DIR="${LOG_DIR:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/scale100k_2gpu_unified_opd_rl_selfdist1500/manual_logs}"
mkdir -p "$LOG_DIR"
PIDS=()
cleanup(){ for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM
for idx in 0 1; do
  gpu="$idx"
  CUDA_VISIBLE_DEVICES="$gpu" MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" WORLD_SIZE=2 RANK="$idx" LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    "$PYTHON_BIN" -u -m projects.pixvl_idea1.trainers.joint_opd_rl_trainer \
    --config "$CONFIG_PATH" > "$LOG_DIR/rank${idx}.log" 2>&1 &
  PIDS+=($!)
done
status=0
finished=0
while (( finished < ${#PIDS[@]} )); do
  if wait -n "${PIDS[@]}"; then
    finished=$((finished+1))
  else
    status=$?
    cleanup
    wait || true
    break
  fi
done
exit "$status"
