#!/usr/bin/env bash
set -euo pipefail

# R13 is a two-GPU screen; the generic wrapper enforces dnacoding positive
# tags, the dna-* job name, and the SAMTok-only initialization contract.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
FARO_ROOT=${FARO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/fepo_tb_gppo_plain_rank_unified_native_rank_pareto_10step_2gpu.py}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/egfepo_train_5120.jsonl}
JOB_NAME=${JOB_NAME:-dna-fepo-native-rank-pareto-10step-2g}

[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -f "$DATA" ]] || { echo "Missing training data: $DATA" >&2; exit 2; }
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "R13 requires at least 5000 rows, got $rows" >&2; exit 2; }

export CONFIG SAMTOK_SELECTIVE_DATA="$DATA" JOB_NAME
exec bash "$SCRIPT_DIR/submit_samtok_tb_gppo.sh"
