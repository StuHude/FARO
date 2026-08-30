#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
ANCHOR_ADAPTER=${ANCHOR_ADAPTER:?ANCHOR_ADAPTER is required}
VISUAL_ADAPTER=${VISUAL_ADAPTER:?VISUAL_ADAPTER is required}
OUTPUT=${OUTPUT:?OUTPUT is required}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/pixvl_idea3/configs/samtok_selective_refseg_eval.py}
SCHEMA=${SCHEMA:-$FARO_ROOT/data/fepo_existence/grefcoco_selective_holdout_256.jsonl}
JOB_NAME=${JOB_NAME:-dna-samtok-visual-probe-$(date +%s)}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
TAGS_FILE=$FARO_ROOT/rjob_tags.txt

case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
[[ "${MODEL,,}" == *samtok* ]] || { echo "MODEL must be original SAMTok" >&2; exit 2; }
[[ -f "$ANCHOR_ADAPTER/adapter_model.safetensors" ]] || { echo "Missing anchor weights" >&2; exit 2; }
[[ -f "$VISUAL_ADAPTER/adapter_model.safetensors" ]] || { echo "Missing visual weights" >&2; exit 2; }
ANCHOR_ADAPTER=$(realpath "$ANCHOR_ADAPTER")
VISUAL_ADAPTER=$(realpath "$VISUAL_ADAPTER")
SCHEMA=$(realpath "$SCHEMA")
OUTPUT=$(realpath -m "$OUTPUT")
case "$(realpath -m "$OUTPUT")" in "$FARO_ROOT"/evals/*) ;; *) echo "OUTPUT must be under FARO/evals" >&2; exit 2 ;; esac

rjob submit --name="$JOB_NAME" --namespace=ailab-dnacoding --cpu=10 --gpu=1 --memory=80000 \
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
export FARO_PROJECT_ROOT='$ROOT'
export SAMTOK_SA2VA_ROOT='$ROOT/third_party/Sa2VA'
export FARO_SAMTOK_MODEL='$MODEL'
export PYTHONPATH='$ROOT/.runtime_deps/huggingface_hub_1_21:$ROOT/third_party/transformers/src:$ROOT/third_party/Sa2VA:$ROOT:$FARO_ROOT/Sa2VA:$FARO_ROOT':\${PYTHONPATH:-}
mkdir -p \"\$(dirname '$OUTPUT')\"
/opt/vlm/bin/python '$FARO_ROOT/tools/probe_visual_adapter_effect.py' --config '$CONFIG' --anchor-adapter '$ANCHOR_ADAPTER' --visual-adapter '$VISUAL_ADAPTER' --schema '$SCHEMA' --output '$OUTPUT'
"
