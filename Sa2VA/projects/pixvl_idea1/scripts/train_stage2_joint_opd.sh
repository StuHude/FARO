#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIXVL_TEXT_SIM_DEVICE="${PIXVL_TEXT_SIM_DEVICE:-cpu}"
export PIXVL_TEXT_SIM_LOCAL_ONLY="${PIXVL_TEXT_SIM_LOCAL_ONLY:-1}"
export PIXVL_DATASET_DEBUG="${PIXVL_DATASET_DEBUG:-0}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/accelerate}"

"$ACCELERATE_BIN" launch \
  --main_process_port "${MAIN_PROCESS_PORT:-0}" \
  --num_processes "${NPROC_PER_NODE:-8}" \
  --mixed_precision "${MIXED_PRECISION:-bf16}" \
  -m projects.pixvl_idea1.trainers.joint_opd_trainer \
  --config "$REPO_ROOT/projects/pixvl_idea1/configs/idea1_joint_opd.py" \
  "$@"
