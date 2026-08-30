#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
ADAPTER=${ADAPTER:-none}
DATASET=${DATASET:?DATASET is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
OUTPUT=$(realpath -m "$OUTPUT")
case "$OUTPUT" in "$FARO_ROOT"/*) ;; *) echo "OUTPUT must be under FARO_ROOT: $FARO_ROOT" >&2; exit 2 ;; esac
JOB_PREFIX=${JOB_PREFIX:-dna-official-refcoco}
POLL_SECONDS=${POLL_SECONDS:-300}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT/data/gcg_sft_20260726/extracted/glamm_data/images/coco2014/train2014}
GPU_LEVELS_OVERRIDE=${GPU_LEVELS_OVERRIDE:-"8 6 4 2 1"}
read -r -a GPU_LEVELS <<< "$GPU_LEVELS_OVERRIDE"
(( ${#GPU_LEVELS[@]} > 0 )) || { echo "GPU_LEVELS_OVERRIDE must not be empty" >&2; exit 2; }
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }

case "$JOB_PREFIX" in dna-*) ;; *) echo "JOB_PREFIX must start with dna-" >&2; exit 2 ;; esac
[[ "${MODEL,,}" == *samtok* ]] || { echo "MODEL must be an original SAMTok checkpoint" >&2; exit 2; }
[[ -f "$DATASET" ]] || { echo "Missing dataset: $DATASET" >&2; exit 2; }
[[ -d "$IMAGE_ROOT" ]] || { echo "Missing image root: $IMAGE_ROOT" >&2; exit 2; }
adapter_arg=""
if [[ "${ADAPTER,,}" != none ]]; then
  [[ -f "$ADAPTER/adapter_config.json" ]] || { echo "Missing adapter: $ADAPTER" >&2; exit 2; }
  adapter_base=$(sed -n 's/.*"base_model_name_or_path": "\([^"]*\)".*/\1/p' "$ADAPTER/adapter_config.json" | head -1)
  [[ "${adapter_base,,}" == *samtok* ]] || { echo "Adapter is not SAMTok-derived: $adapter_base" >&2; exit 2; }
  adapter_arg="--adapter '$ADAPTER'"
fi

for gpu in "${GPU_LEVELS[@]}"; do
  job_name="${JOB_PREFIX}-${gpu}g-$(date +%s)"
  echo "SUBMIT gpu=$gpu job=$job_name"
  submit_log=$(rjob submit --name="$job_name" --namespace=ailab-dnacoding \
    --cpu="$((gpu * 10))" --gpu="$gpu" --memory="$((gpu * 80000))" \
    --positive-tags="$POSITIVE_TAGS" --charged-group=dnacoding_gpu --private-machine=group \
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
OUT='$OUTPUT'; TMP=\"\${OUT}.tmp.$job_name\"; rm -rf \"\$TMP\"; mkdir -p \"\$TMP/datasets\" \"\$TMP/shards\"
split -d -a 2 -n l/'$gpu' --additional-suffix=.jsonl '$DATASET' \"\$TMP/datasets/shard_\"
pids=()
for i in \$(seq 0 $((gpu - 1))); do
  suffix=\$(printf '%02d' \"\$i\")
  mkdir -p \"\$TMP/shards/shard_\$suffix\"
  CUDA_VISIBLE_DEVICES=\$i /opt/vlm/bin/python '$FARO_ROOT/tools/run_official_samtok_refcoco_shard.py' \
    --pixvl-root '$ROOT' --base-model '$MODEL' \
    --vq-sam2-path '$MODEL/mask_tokenizer_256x2.pth' --sam2-path '$MODEL/sam2.1_hiera_large.pt' \
    --dataset \"\$TMP/datasets/shard_\$suffix.jsonl\" --image-root '$IMAGE_ROOT' \
    --save-dir \"\$TMP/shards/shard_\$suffix\" $adapter_arg \
    > \"\$TMP/log_\$i.txt\" 2>&1 & pids+=(\$!)
done
failed=0; for pid in \"\${pids[@]}\"; do wait \"\$pid\" || failed=1; done
(( failed == 0 )) || { echo 'RefCOCO shard failed' >&2; exit 1; }
/opt/vlm/bin/python '$FARO_ROOT/tools/merge_official_refcoco_shards.py' \
  --dataset '$DATASET' --shard-root \"\$TMP/shards\" --output \"\$TMP/merged\"
/opt/vlm/bin/python '$FARO_ROOT/tools/score_official_refcoco_predictions.py' \
  --pixvl-root '$ROOT' --predictions \"\$TMP/merged\" --workspace \"\$TMP/metric_workspace\" \
  > \"\$TMP/metric.txt\"
mkdir -p \"\$(dirname \"\$OUT\")\"; rm -rf \"\$OUT\"; mkdir -p \"\$OUT\"
mv \"\$TMP/merged\" \"\$OUT/predictions\"
cp \"\$TMP/metric.txt\" \"\$OUT/metric.txt\"
cp \"\$TMP\"/log_*.txt \"\$OUT/\"
/opt/vlm/bin/python - \"\$OUT/metric.txt\" \"\$OUT/summary.json\" <<'PY'
import json, re, sys
text = open(sys.argv[1], encoding='utf-8').read()
matches = re.findall(r'REC AP_50:\s*([0-9.]+)\s*\\| RES CIoU:\s*([0-9.]+)', text)
if not matches:
    raise SystemExit('official metric output missing')
out = {'val_ap50': float(matches[0][0]), 'val_ciou': float(matches[0][1])}
if len(matches) > 1:
    out.update({'vlmr1_ap50': float(matches[1][0]), 'vlmr1_ciou': float(matches[1][1])})
json.dump(out, open(sys.argv[2], 'w', encoding='utf-8'), indent=2)
PY
rm -rf \"\$TMP\"
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
     printf '%s\n' "$status" | grep -qE 'STARTING, +gpu-[^[:space:]]+'; then exit 0; fi
  rjob stop "$query_name" --namespace=ailab-dnacoding || true
done
