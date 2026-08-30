#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
export FARO_ROOT
export CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_gain_preference_one_step_2gpu.py}
export JOB_NAME=${JOB_NAME:-dna-samtok-fepo-gain-pref-one-step-2g}
exec bash "$SCRIPT_DIR/submit_samtok_boundary_credit_gr_cppo.sh"
