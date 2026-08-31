#!/usr/bin/env bash
set -euo pipefail
FARO_ROOT=${FARO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
STATE=${STATE:-$FARO_ROOT/logs/pes_shuffled_submit}
INTERVAL=${INTERVAL:-300}
PROXY_SETUP_URL=${PROXY_SETUP_URL:-http://deploy.i.h.pjlab.org.cn/infra/scripts/setup_proxy.sh}
PES_METRICS=$FARO_ROOT/outputs/samtok_selective/fepo_tb_gppo_plain_rank_unified_predicted_evidence_scope_10step_2gpu/metrics.json
mkdir -p "$STATE"
exec 9>"$STATE/.lock"
flock -n 9 || exit 0
printf '%s\n' "$$" > "$STATE/pid"
cleanup_pid() {
  [[ "$(sed -n '1p' "$STATE/pid" 2>/dev/null || true)" == "$$" ]] && rm -f "$STATE/pid"
}
trap cleanup_pid EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready() {
  python3 - "$PES_METRICS" <<'PY'
import json,sys
from pathlib import Path
try: d=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
except (OSError,ValueError): raise SystemExit(1)
gate = d.get('validity_gate') or {}
ready = (
    d.get('status') == 'finished'
    and len(d.get('steps') or []) >= 10
    and gate.get('passed') is True
    and gate.get('effective_support_gate_passed') is True
    and gate.get('tail_risk_gate_passed') is True
    and gate.get('pes_coverage_gate_passed') is True
)
raise SystemExit(0 if ready else 1)
PY
}
refresh_proxy_best_effort() {
  command -v curl >/dev/null 2>&1 || return 0
  local setup
  setup=$(curl -fsSL --max-time 20 "$PROXY_SETUP_URL" 2>/dev/null) || return 0
  [[ -n "$setup" ]] || return 0
  source /dev/stdin <<<"$setup" 2>/dev/null || true
}
marker=$STATE/submitted
while [[ ! -s "$marker" ]]; do
  ready || { sleep "$INTERVAL"; continue; }
  set +e; listing=$(rjob list --namespace=ailab-dnacoding 2>&1); rc=$?; set -e
  if (( rc != 0 )); then
    printf '%s control_plane_unavailable status=%s\n' "$(date -Is)" "$rc" >> "$STATE/submit.log"
    refresh_proxy_best_effort
    sleep "$INTERVAL"; continue
  fi
  existing=$(printf '%s\n' "$listing" | grep -E 'dna-fepo-predicted-evidence-shuffled-10step-2g' | head -1 || true)
  if [[ -n "$existing" ]]; then printf '%s\n' "$(printf '%s\n' "$existing" | awk '{print $1}')" > "$marker"; break; fi
  job_name="dna-fepo-predicted-evidence-shuffled-10step-2g-$(date +%s)"
  set +e; output=$(JOB_NAME="$job_name" bash "$FARO_ROOT/scripts/submit_samtok_tb_gppo_predicted_evidence_scope_shuffled.sh" 2>&1); rc=$?; set -e
  printf '%s\n' "$output" >> "$STATE/submit.log"
  if (( rc == 0 )); then printf '%s\n' "$job_name" > "$marker"; break; fi
  sleep "$INTERVAL"
done
