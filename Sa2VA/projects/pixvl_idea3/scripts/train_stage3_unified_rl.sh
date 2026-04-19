#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
PYTHON_BIN="${PYTHON_BIN:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1_fa/bin/python}"

"$PYTHON_BIN" -m projects.pixvl_idea1.trainers.joint_opd_rl_trainer \
  --config "$REPO_ROOT/projects/pixvl_idea3/configs/idea3_mvp_unified_rl.py"
