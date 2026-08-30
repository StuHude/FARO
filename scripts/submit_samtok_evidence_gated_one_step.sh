#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
export FARO_ROOT
export CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_evidence_gated_one_step_2gpu.py}
export SAMTOK_SELECTIVE_DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
export JOB_NAME=${JOB_NAME:-dna-samtok-evidence-gated-one-step-2g}
export TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
export FEPO_EVIDENCE_MODE=${FEPO_EVIDENCE_MODE:-view_drop}
echo "The registered minimum is 5,120 rows and 10 optimizer steps; use submit_samtok_evidence_gated_10step.sh" >&2
exit 2
