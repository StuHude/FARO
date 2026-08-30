#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
export FARO_ROOT
export CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_evidence_gated_20step_2gpu.py}
export SAMTOK_SELECTIVE_DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
export JOB_NAME=${JOB_NAME:-dna-samtok-evidence-gated-20step-2g}
export TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
export FEPO_EVIDENCE_MODE=${FEPO_EVIDENCE_MODE:-view_drop}
exec bash "$SCRIPT_DIR/submit_samtok_es_gr_cppo_one_step.sh"
