#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
export FARO_ROOT
export CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/projector_plastic_sft_control40.py}
export JOB_NAME=${JOB_NAME:-dna-samtok-projector-plastic-sft-control40-2g}
export SAMTOK_STANDALONE_ADAPTER=${SAMTOK_STANDALONE_ADAPTER:-$FARO_ROOT/outputs/samtok_selective/continued_sft_to500/adapter}
exec bash "$SCRIPT_DIR/submit_samtok_standalone_sft.sh"
