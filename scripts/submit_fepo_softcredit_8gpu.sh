#!/usr/bin/env bash
set -euo pipefail

# One matched FEPO run. The delegated submitter enforces the 8--24 GPU bound
# and the dna- name contract; this wrapper only selects the new credit mode.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export GPU_COUNT=${GPU_COUNT:-8}
export JOB_NAME=${JOB_NAME:-dna-fepo-relation-softcredit-seed17-8gpu-r1}
export CONFIG=${CONFIG:-${SCRIPT_DIR}/../Sa2VA/projects/pixvl_idea3/configs/fepo_relation_softcredit_seed17_8gpu.py}
exec "${SCRIPT_DIR}/submit_fepo_schema_train_smoke.sh" "$@"
