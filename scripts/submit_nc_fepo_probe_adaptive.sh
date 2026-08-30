#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
INPUT=${INPUT:?INPUT is required}; OUTPUT=${OUTPUT:?OUTPUT is required}
OUTPUT=$(realpath -m "$OUTPUT")
case "$OUTPUT" in "$FARO_ROOT"/*) ;; *) echo "OUTPUT must be under FARO_ROOT: $FARO_ROOT" >&2; exit 2 ;; esac
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/pixvl_idea3/configs/fepo_eval_deterministic.py}
JOB_PREFIX=${JOB_PREFIX:-dna-nc-fepo-linear-probe}; POLL_SECONDS=${POLL_SECONDS:-300}
GPU_LEVELS=(8 6 4 2 1)
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
ALL_POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$ALL_POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
TAGS8=$ALL_POSITIVE_TAGS
TAGS1=$ALL_POSITIVE_TAGS
[[ "${MODEL,,}" == *samtok* && "$JOB_PREFIX" == dna-* ]] || exit 2
for gpu in "${GPU_LEVELS[@]}"; do
  name="${JOB_PREFIX}-${gpu}g-$(date +%s)"; tags="$TAGS1"; ((gpu>=8)) && tags="$TAGS8"
  echo "SUBMIT gpu=$gpu job=$name"
  log=$(rjob submit --name="$name" --namespace=ailab-dnacoding --cpu="$((gpu*10))" --gpu="$gpu" --memory="$((gpu*80000))" --positive-tags="$tags" --charged-group=dnacoding_gpu --private-machine=group --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng --image=registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 --custom-resources=brainpp.cn/fuse=1 --enable-sshd -- bash -lc "
set -euo pipefail; unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY; cd /opt; mkdir -p /opt/vlm
if [ -f /opt/vlm_env.tar.gz ]; then tar -xzf /opt/vlm_env.tar.gz -C /opt/vlm; rm -f /opt/vlm_env.tar.gz; elif [ -f vlm_env.tar.gz ]; then tar -xzf vlm_env.tar.gz -C /opt/vlm; rm -f vlm_env.tar.gz; fi
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack || true; export PATH=/opt/vlm/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/compat/bin:\$PATH; export LD_LIBRARY_PATH=/opt/vlm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}; export FARO_PROJECT_ROOT='$ROOT'; export FARO_WORKSPACE_ROOT='$FARO_ROOT'; export FARO_SAMTOK_MODEL='$MODEL'; export PYTHONPATH='$ROOT/.runtime_deps/huggingface_hub_1_21:$ROOT/third_party/transformers/src:$ROOT/third_party/Sa2VA:$ROOT:$FARO_ROOT/Sa2VA:$FARO_ROOT':\${PYTHONPATH:-}; OUT='$OUTPUT'; TMP=\"\${OUT}.probe_shards\"; rm -rf \"\$TMP\"; mkdir -p \"\$TMP\"
for i in \$(seq 0 $((gpu-1))); do CUDA_VISIBLE_DEVICES=\$i /opt/vlm/bin/python -m projects.pixvl_idea3.eval.extract_null_probe_features --config '$CONFIG' --input '$INPUT' --task-id \$i --num-tasks '$gpu' --output \"\$TMP/part_\$i.json\" > \"\$TMP/log_\$i.txt\" 2>&1 & done; wait
/opt/vlm/bin/python '$FARO_ROOT/tools/train_null_probe.py' --input-dir \"\$TMP\" --output \"\$OUT\"
" 2>&1)
  printf '%s\n' "$log"
  q=$(printf '%s\n' "$log" | sed -n 's/.*created rjob_name: //p' | tail -1)
  q=${q:-$name}
  ((gpu==1)) && exit 0
  sleep "$POLL_SECONDS"
  while :; do
    set +e
    st=$(rjob list "$q" --namespace=ailab-dnacoding 2>&1)
    status_rc=$?
    set -e
    printf '%s\n' "$st"
    if (( status_rc == 0 )); then break; fi
    echo "STATUS_QUERY_UNAVAILABLE job=${q}; retrying in 60s" >&2
    sleep 60
  done
  if printf '%s\n' "$st" | grep -qE 'Running|Succeed(ed)?|Failed|Stopped' || \
     printf '%s\n' "$st" | grep -qE 'STARTING, +gpu-[^[:space:]]+'; then exit 0; fi
  rjob stop "$q" --namespace=ailab-dnacoding || true
done
