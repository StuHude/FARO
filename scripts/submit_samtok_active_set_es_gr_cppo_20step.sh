#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
export FARO_ROOT
export CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_active_set_es_gr_cppo_20step_2gpu.py}
export JOB_NAME=${JOB_NAME:-dna-samtok-fepo-active-set-es-20step-2g}
exec bash "$SCRIPT_DIR/submit_samtok_active_set_es_gr_cppo_one_step.sh"
