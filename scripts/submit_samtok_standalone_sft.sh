#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${SAMTOK_BASE_CHECKPOINT:?Set SAMTOK_BASE_CHECKPOINT to the approved original SAMTok checkpoint}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/grefcoco_selective_train_256.jsonl}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/samtok_selective/configs/sft_smoke_2gpu.py}
ADAPTER=${SAMTOK_STANDALONE_ADAPTER:-}
JOB_NAME=${JOB_NAME:-dna-samtok-standalone-sft-smoke-2g}
GPU_COUNT=${GPU_COUNT:-2}
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}

case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
[[ "$GPU_COUNT" == "2" ]] || { echo "The registered standalone smoke requires GPU_COUNT=2" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 2; }
[[ -f "$DATA" ]] || { echo "Missing data: $DATA" >&2; exit 2; }
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "Training requires at least 5000 rows, got $rows" >&2; exit 2; }
case "$JOB_NAME" in *smoke*|*one-step*|*one_step*)
  echo "Smoke/one-step training jobs are disabled; use a >=10-step job" >&2
exit 2;; esac
export PYTHONPATH="$FARO_ROOT/Sa2VA:$FARO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SAMTOK_BASE_CHECKPOINT="$MODEL" SAMTOK_SELECTIVE_DATA="$DATA" SAMTOK_STANDALONE_ADAPTER="$ADAPTER"
python3 "$FARO_ROOT/tools/validate_training_budget.py" --config "$CONFIG" --data "$DATA"
if [[ -n "$ADAPTER" ]]; then
  [[ -f "$ADAPTER/adapter_config.json" && -f "$ADAPTER/adapter_model.safetensors" ]] || {
    echo "Invalid standalone continued-SFT adapter: $ADAPTER" >&2; exit 2;
  }
fi
if [[ "$(basename "$CONFIG")" == continued_* && -z "$ADAPTER" ]]; then
  echo "continued-SFT config requires SAMTOK_STANDALONE_ADAPTER" >&2
  exit 2
fi
for artifact in config.json model.safetensors.index.json sam2.1_hiera_large.pt mask_tokenizer_256x2.pth; do
  [[ -f "$MODEL/$artifact" ]] || { echo "Missing SAMTok artifact: $MODEL/$artifact" >&2; exit 2; }
done
[[ ! -f "$MODEL/adapter_config.json" ]] || { echo "MODEL must be a base checkpoint, not an adapter" >&2; exit 2; }
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }

PYTHONPATH="$FARO_ROOT/Sa2VA:$FARO_ROOT" python3 -m projects.samtok_selective.manifests guard

rjob submit --name="$JOB_NAME" --namespace=ailab-dnacoding \
  --cpu=20 --gpu=2 --memory=160000 \
  --positive-tags="$POSITIVE_TAGS" --charged-group=dnacoding_gpu --private-machine=group \
  --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
  --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
  --image=registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
  --custom-resources=brainpp.cn/fuse=1 --enable-sshd -- bash -lc "
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
mkdir -p /opt/vlm
if [ -f /opt/vlm_env.tar.gz ]; then tar -xzf /opt/vlm_env.tar.gz -C /opt/vlm; rm -f /opt/vlm_env.tar.gz; fi
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack || true
/opt/vlm/bin/pip install --no-index --no-deps --force-reinstall '$FARO_ROOT'/vendor/wheels/*.whl >/dev/null
/opt/vlm/bin/pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
export PATH=/opt/vlm/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/compat/bin:\$PATH
export LD_LIBRARY_PATH=/opt/vlm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
export PYTHONPATH='$FARO_ROOT/third_party/transformers/src:$FARO_ROOT/Sa2VA:$FARO_ROOT'
export SAMTOK_BASE_CHECKPOINT='$MODEL'
export SAMTOK_SELECTIVE_DATA='$DATA'
export SAMTOK_STANDALONE_ADAPTER='$ADAPTER'
mkdir -p '$FARO_ROOT/logs/samtok_selective'
cd '$FARO_ROOT/Sa2VA'
/opt/vlm/bin/torchrun --standalone --nproc_per_node=2 -m projects.samtok_selective.sft_trainer --config '$CONFIG' 2>&1 | tee '$FARO_ROOT/logs/samtok_selective/$JOB_NAME.log'
"
