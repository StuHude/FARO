#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
JOB_NAME=${JOB_NAME:-dna-fepo-eval-r1}
ADAPTER=${ADAPTER:?ADAPTER is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
OUTPUT=$(realpath -m "$OUTPUT")
case "$OUTPUT" in "$FARO_ROOT"/*) ;; *) echo "OUTPUT must be under FARO_ROOT: $FARO_ROOT" >&2; exit 2 ;; esac
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/pixvl_idea3/configs/fepo_schema_smoke_2gpu.py}
GPU_COUNT=${GPU_COUNT:-8}
CPU_COUNT=${CPU_COUNT:-$((GPU_COUNT * 10))}
MEMORY_MB=${MEMORY_MB:-$((GPU_COUNT * 80000))}
if (( GPU_COUNT < 1 || GPU_COUNT > 24 )); then
  echo "GPU_COUNT must be between 1 and 24" >&2
  exit 2
fi
REFSEG_SCHEMA=${REFSEG_SCHEMA:-$ROOT/data/pixvl_idea3/schemas/refseg_train_routed.jsonl}
MASKCAP_SCHEMA=${MASKCAP_SCHEMA:-$ROOT/data/pixvl_idea3/schemas/maskcap_train_routed.jsonl}
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
rjob submit \
  --name="$JOB_NAME" --namespace=ailab-dnacoding --cpu="$CPU_COUNT" --gpu="$GPU_COUNT" --memory="$MEMORY_MB" --positive-tags="$POSITIVE_TAGS" \
  --charged-group=dnacoding_gpu --private-machine=group \
  --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
  --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
  --image=registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
  --custom-resources=brainpp.cn/fuse=1 --enable-sshd \
  -- bash -lc "
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
cd /opt; mkdir -p /opt/vlm
if [ -f /opt/vlm_env.tar.gz ]; then tar -xzf /opt/vlm_env.tar.gz -C /opt/vlm; rm -f /opt/vlm_env.tar.gz; elif [ -f vlm_env.tar.gz ]; then tar -xzf vlm_env.tar.gz -C /opt/vlm; rm -f vlm_env.tar.gz; fi
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack || true
export PATH=/opt/vlm/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/compat/bin:\$PATH
export LD_LIBRARY_PATH=/opt/vlm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
export FARO_PROJECT_ROOT='$ROOT'
export FARO_WORKSPACE_ROOT='$FARO_ROOT'
export FARO_SAMTOK_MODEL='$MODEL'
export FARO_BASE_CONFIG='$FARO_ROOT/Sa2VA/projects/pixvl_idea1/configs/idea1_joint_sft.py'
export PYTHONPATH='$ROOT/.runtime_deps/huggingface_hub_1_21:$ROOT/third_party/transformers/src:$ROOT/third_party/Sa2VA:$ROOT:$FARO_ROOT/Sa2VA:$FARO_ROOT':\${PYTHONPATH:-}
mkdir -p \"$(dirname '$OUTPUT')\"
/opt/vlm/bin/python -m projects.pixvl_idea3.eval.eval_mvp_bundle \
  --config '$CONFIG' \
  --adapter-path '$ADAPTER' \
  --refseg-overall-schema '$REFSEG_SCHEMA' \
  --geometry-schema '$REFSEG_SCHEMA' \
  --semantic-schema '$MASKCAP_SCHEMA' \
  --output '$OUTPUT'
"
