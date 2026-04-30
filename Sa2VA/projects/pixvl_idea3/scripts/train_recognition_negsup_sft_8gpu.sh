#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/accelerate}"

"$ACCELERATE_BIN" launch \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  --num_processes "${NPROC_PER_NODE:-8}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  -m projects.pixvl_idea1.trainers.joint_sft_trainer \
  --config "$REPO_ROOT/projects/pixvl_idea3/configs/idea3_recognition_negsup_sft_8gpu.py" \
  "$@"
