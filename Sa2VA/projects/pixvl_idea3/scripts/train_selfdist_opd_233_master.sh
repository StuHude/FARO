#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-/mnt/pfs/xiaoyicheng/outputs/pixvl_idea3/selfdist_233_master_logs}"
mkdir -p "$LOG_DIR"
PIDS=()
cleanup(){ for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done; }
trap cleanup INT TERM

bash "$REPO_ROOT/projects/pixvl_idea3/scripts/train_selfdist_unified_2gpu_1500.sh" > "$LOG_DIR/unified.launch.log" 2>&1 &
PIDS+=($!)
bash "$REPO_ROOT/projects/pixvl_idea3/scripts/train_selfdist_routed_rl_3gpu_1500.sh" > "$LOG_DIR/routed_rl.launch.log" 2>&1 &
PIDS+=($!)
bash "$REPO_ROOT/projects/pixvl_idea3/scripts/train_selfdist_routed_opd_rl_3gpu_1500.sh" > "$LOG_DIR/routed_opd_rl.launch.log" 2>&1 &
PIDS+=($!)

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
