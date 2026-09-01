#!/usr/bin/env bash
set -euo pipefail

# Submit an evaluation at 8 GPUs and downgrade queued jobs every five minutes.
# The final 1-GPU job is intentionally left queued indefinitely.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
JOB_PREFIX=${JOB_PREFIX:-dna-fepo-adaptive-eval}
ADAPTER=${ADAPTER:?ADAPTER is required}
ANCHOR_ADAPTER=${ANCHOR_ADAPTER:-}
OUTPUT=${OUTPUT:?OUTPUT is required}
[[ -n "$ADAPTER" && -n "$OUTPUT" ]] || { echo "ADAPTER and OUTPUT must be non-empty" >&2; exit 2; }
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_LEVELS=(8 6 4 2 1)

# The login workspace exports a proxy that is not routable to the kube-brain
# API. Keep every control-plane call proxy-free even when this script is
# launched by a long-lived monitor that inherited the workspace environment.
rjob_clean() {
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
      -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY rjob "$@"
}

for gpu in "${GPU_LEVELS[@]}"; do
  job_name="${JOB_PREFIX}-${gpu}g-$(date +%s)"
  echo "SUBMIT gpu=${gpu} job=${job_name}"
  set +e
  submit_log=$(JOB_NAME="$job_name" GPU_COUNT="$gpu" ADAPTER="$ADAPTER" ANCHOR_ADAPTER="$ANCHOR_ADAPTER" OUTPUT="$OUTPUT" \
    CONFIG="${CONFIG:-}" REFSEG_SCHEMA="${REFSEG_SCHEMA:-}" MASKCAP_SCHEMA="${MASKCAP_SCHEMA:-}" \
    GEOMETRY_SCHEMA="${GEOMETRY_SCHEMA:-}" EXISTENCE_SCHEMA="${EXISTENCE_SCHEMA:-}" \
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY \
        bash "$SCRIPT_DIR/submit_fepo_eval_sharded.sh" 2>&1)
  submit_status=$?
  set -e
  printf '%s\n' "$submit_log"
  if (( submit_status != 0 )); then
    echo "SUBMIT_FAILED gpu=${gpu} status=${submit_status}" >&2
    exit "$submit_status"
  fi
  query_name=$(printf '%s\n' "$submit_log" | sed -n 's/.*created rjob_name: //p' | tail -1)
  query_name=${query_name:-$job_name}

  # The one-GPU stage is the terminal fallback and is never cancelled.
  (( gpu == 1 )) && exit 0

  sleep "$POLL_SECONDS"
  # A transient control-plane/DNS failure is not evidence that the replica is
  # still queued. Keep the submitted stage alive and retry until the API is
  # reachable; otherwise the fallback chain could stop a valid 8/6/4/2-GPU
  # evaluation merely because the status query timed out.
  while :; do
    set +e
    status=$(rjob_clean list "$query_name" --namespace=ailab-dnacoding 2>&1)
    status_rc=$?
    set -e
    echo "$status"
    if (( status_rc == 0 )); then
      break
    fi
    echo "STATUS_QUERY_UNAVAILABLE job=${query_name}; retrying in 60s" >&2
    sleep 60
  done
  # rjob currently spells the terminal success state ``Succeed`` (not
  # ``Succeeded``); accept both spellings for compatibility with older CLI
  # versions. Any allocated or terminal job stops the downgrade chain.
  # Some queued replicas are reported as ``STARTING,`` with an empty node.
  # Only a STARTING replica with a concrete GPU node has left the queue.
  if printf '%s\n' "$status" | grep -qE 'Running|Succeed(ed)?|Failed|Stopped' || \
     printf '%s\n' "$status" | grep -qE 'STARTING, +gpu-[^[:space:]]+'; then
    echo "JOB_${job_name}_LEFT_QUEUE"
    exit 0
  fi

  echo "JOB_${job_name}_STILL_QUEUED_DOWNGRADE"
  rjob_clean stop "$query_name" --namespace=ailab-dnacoding || true
done
