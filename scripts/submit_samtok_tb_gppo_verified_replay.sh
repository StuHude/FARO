#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_verified_replay_10step_2gpu.py}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
JOB_NAME=${JOB_NAME:-dna-fepo-verified-replay-10step-2g}
export FARO_ROOT CONFIG SAMTOK_SELECTIVE_DATA="$DATA" JOB_NAME

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -f "$DATA" ]] || { echo "Missing training data: $DATA" >&2; exit 2; }
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "R9 requires at least 5000 rows, got $rows" >&2; exit 2; }

exec bash "$SCRIPT_DIR/submit_samtok_tb_gppo.sh"
