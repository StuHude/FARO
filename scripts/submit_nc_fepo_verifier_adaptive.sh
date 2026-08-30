#!/usr/bin/env bash
set -euo pipefail

# Frozen SAMTok-only NC-FEPO verifier. The only PixVL dependency is the
# evaluation/data mount; MODEL must remain the released SAMTok checkpoint.
ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
ADAPTER=${ADAPTER:-}
INPUT=${INPUT:?INPUT is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
OUTPUT=$(realpath -m "$OUTPUT")
case "$OUTPUT" in "$FARO_ROOT"/*) ;; *) echo "OUTPUT must be under FARO_ROOT: $FARO_ROOT" >&2; exit 2 ;; esac
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/pixvl_idea3/configs/fepo_eval_deterministic.py}
JOB_PREFIX=${JOB_PREFIX:-dna-nc-fepo-verifier}
POLL_SECONDS=${POLL_SECONDS:-300}
GPU_LEVELS=(8 6 4 2 1)
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
ALL_POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$ALL_POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS_64=$ALL_POSITIVE_TAGS
POSITIVE_TAGS_16=$ALL_POSITIVE_TAGS

[[ "${MODEL,,}" == *samtok* ]] || { echo "MODEL must be the original SAMTok checkpoint" >&2; exit 2; }
[[ "$JOB_PREFIX" == dna-* ]] || { echo "JOB_PREFIX must start with dna-" >&2; exit 2; }
if [[ -n "$ADAPTER" && -f "$ADAPTER/adapter_config.json" ]]; then
  base=$(sed -n 's/.*"base_model_name_or_path": "\([^"]*\)".*/\1/p' "$ADAPTER/adapter_config.json" | head -1)
  [[ "${base,,}" == *samtok* ]] || { echo "ADAPTER is not SAMTok-derived: $base" >&2; exit 2; }
fi

for gpu in "${GPU_LEVELS[@]}"; do
  job_name="${JOB_PREFIX}-${gpu}g-$(date +%s)"
  tags="$POSITIVE_TAGS_16"; (( gpu >= 8 )) && tags="$POSITIVE_TAGS_64"
  echo "SUBMIT gpu=$gpu job=$job_name"
  submit_log=$(rjob submit --name="$job_name" --namespace=ailab-dnacoding \
    --cpu="$((gpu * 10))" --gpu="$gpu" --memory="$((gpu * 80000))" --positive-tags="$tags" \
    --charged-group=dnacoding_gpu --private-machine=group \
    --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
    --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
    --image=registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
    --custom-resources=brainpp.cn/fuse=1 --enable-sshd -- bash -lc "
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
cd /opt; mkdir -p /opt/vlm
if [ -f /opt/vlm_env.tar.gz ]; then tar -xzf /opt/vlm_env.tar.gz -C /opt/vlm; rm -f /opt/vlm_env.tar.gz; elif [ -f vlm_env.tar.gz ]; then tar -xzf vlm_env.tar.gz -C /opt/vlm; rm -f vlm_env.tar.gz; fi
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack || true
export PATH=/opt/vlm/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/compat/bin:\$PATH
export LD_LIBRARY_PATH=/opt/vlm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
export PYTHONPATH='$ROOT/.runtime_deps/huggingface_hub_1_21:$ROOT/third_party/transformers/src:$ROOT/third_party/Sa2VA:$ROOT:$FARO_ROOT/Sa2VA:$FARO_ROOT':\${PYTHONPATH:-}
export FARO_PROJECT_ROOT='$ROOT'; export FARO_WORKSPACE_ROOT='$FARO_ROOT'; export FARO_SAMTOK_MODEL='$MODEL'
OUT='$OUTPUT'; TMP=\"\${OUT}.shards\"; rm -rf \"\$TMP\"; mkdir -p \"\$TMP\"
for i in \$(seq 0 $((gpu - 1))); do
  CUDA_VISIBLE_DEVICES=\$i /opt/vlm/bin/python -m projects.pixvl_idea3.eval.eval_null_verifier --config '$CONFIG' --adapter-path '$ADAPTER' --input '$INPUT' --task-id \$i --num-tasks '$gpu' --output \"\$TMP/part_\$i.json\" > \"\$TMP/log_\$i.txt\" 2>&1 &
done
wait
/opt/vlm/bin/python - \"\$TMP\" \"\$OUT\" <<'PY'
import json, sys
from pathlib import Path
parts = [json.loads(p.read_text()) for p in sorted(Path(sys.argv[1]).glob('part_*.json'))]
records = [r for part in parts for r in part.get('records', [])]
records.sort(key=lambda r: r['id'])
correct = sum(int(bool(r['predicted_exists']) == bool(r['target_exists'])) for r in records)
Path(sys.argv[2]).write_text(json.dumps({'num_samples': len(records), 'accuracy': correct / max(len(records), 1), 'records': records}, indent=2), encoding='utf-8')
PY
" 2>&1)
  printf '%s\n' "$submit_log"
  query_name=$(printf '%s\n' "$submit_log" | sed -n 's/.*created rjob_name: //p' | tail -1)
  query_name=${query_name:-$job_name}
  (( gpu == 1 )) && exit 0
  sleep "$POLL_SECONDS"
  while :; do
    set +e
    status=$(rjob list "$query_name" --namespace=ailab-dnacoding 2>&1)
    status_rc=$?
    set -e
    printf '%s\n' "$status"
    if (( status_rc == 0 )); then break; fi
    echo "STATUS_QUERY_UNAVAILABLE job=${query_name}; retrying in 60s" >&2
    sleep 60
  done
  if printf '%s\n' "$status" | grep -qE 'Running|Succeed(ed)?|Failed|Stopped' || \
     printf '%s\n' "$status" | grep -qE 'STARTING, +gpu-[^[:space:]]+'; then
    echo "JOB_${job_name}_LEFT_QUEUE"; exit 0
  fi
  rjob stop "$query_name" --namespace=ailab-dnacoding || true
done
