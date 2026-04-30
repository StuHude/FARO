#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python}"
WORLD_SIZE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MAIN_PROCESS_PORT:-29571}"
CONFIG_PATH="$REPO_ROOT/projects/pixvl_idea3/configs/idea3_atom_semcovcal_routed_opd_rl_8gpu_1epoch.py"
LOG_DIR="${LOG_DIR:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/atom_semcovcal_routed_opd_rl_8gpu_1epoch_sample_conditioned/manual_logs}"
mkdir -p "$LOG_DIR"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-7200}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM

for ((rank=0; rank<WORLD_SIZE; rank++)); do
  CUDA_VISIBLE_DEVICES="$rank" \
  MASTER_ADDR="$MASTER_ADDR" \
  MASTER_PORT="$MASTER_PORT" \
  WORLD_SIZE="$WORLD_SIZE" \
  RANK="$rank" \
  LOCAL_RANK=0 \
  LOCAL_WORLD_SIZE=1 \
  "$PYTHON_BIN" -u -m projects.pixvl_idea3.trainers.joint_routed_opd_rl_trainer \
    --config "$CONFIG_PATH" \
    "$@" \
    > "$LOG_DIR/rank${rank}.log" 2>&1 &
  PIDS+=($!)
done

status=0
finished=0
while (( finished < ${#PIDS[@]} )); do
  if wait -n "${PIDS[@]}"; then
    finished=$((finished + 1))
  else
    status=$?
    cleanup
    wait || true
    break
  fi
done
exit "$status"
