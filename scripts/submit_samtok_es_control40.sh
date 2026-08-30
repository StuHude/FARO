#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
EXPECTED_ANCHOR=$FARO_ROOT/outputs/samtok_selective/continued_sft_to500/adapter
ADAPTER=${SAMTOK_STANDALONE_ADAPTER:-$EXPECTED_ANCHOR}

[[ "$(realpath "$ADAPTER")" == "$(realpath "$EXPECTED_ANCHOR")" ]] || {
  echo "ES control40 must use the frozen total-500 SAMTok anchor" >&2
  exit 2
}

export FARO_ROOT
export SAMTOK_STANDALONE_ADAPTER=$ADAPTER
export CONFIG=$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/continued_sft_es_control40.py
export JOB_NAME=${JOB_NAME:-dna-samtok-standalone-es-control40-2g}
export GPU_COUNT=2
export TAGS_FILE=$FARO_ROOT/rjob_tags.txt

exec bash "$SCRIPT_DIR/submit_samtok_standalone_sft.sh"
