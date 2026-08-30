#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
# Workers mount the shared dnacoding roots, not submit-host paths such as
# /mnt/pfs. Default to the approved read-only SAMTok release on that mount.
APPROVED_MODEL=/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co
MODEL=${SAMTOK_BASE_CHECKPOINT:-$APPROVED_MODEL}
DATA=${SAMTOK_SELECTIVE_DATA:-$FARO_ROOT/data/fepo_existence/grefcoco_selective_train_256.jsonl}
CONFIG=${CONFIG:?Set CONFIG to one preregistered TB-GPPO config}
ADAPTER=${SAMTOK_STANDALONE_ADAPTER:-$FARO_ROOT/outputs/samtok_selective/continued_sft_to500/adapter}
JOB_NAME=${JOB_NAME:-dna-samtok-fepo-tb-gppo-2g}
TAGS_FILE=$FARO_ROOT/rjob_tags.txt

case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
for path in "$CONFIG" "$DATA" "$TAGS_FILE"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
rows=$(awk 'NF {n++} END {print n+0}' "$DATA")
(( rows >= 5000 )) || { echo "Training requires at least 5000 rows, got $rows" >&2; exit 2; }
case "$CONFIG" in
  *_one_step_*.py|*_one-step-*.py)
    echo "Training configs with one-step budgets are disabled; use a >=10-step config" >&2
    exit 2
    ;;
esac
export PYTHONPATH="$FARO_ROOT/Sa2VA:$FARO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SAMTOK_BASE_CHECKPOINT="$MODEL" SAMTOK_SELECTIVE_DATA="$DATA" SAMTOK_STANDALONE_ADAPTER="$ADAPTER"
python3 "$FARO_ROOT/tools/validate_training_budget.py" --config "$CONFIG" --data "$DATA"
MODEL=$(realpath "$MODEL")
[[ "$MODEL" == "$APPROVED_MODEL" ]] || {
  echo "SAMTOK_BASE_CHECKPOINT must be the worker-visible approved path: $APPROVED_MODEL" >&2
  echo "Refusing submit-host-only checkpoint path: $MODEL" >&2
  exit 2
}
if [[ "${SKIP_LOCAL_MODEL_PREFLIGHT:-0}" != "1" ]]; then
  for artifact in config.json model.safetensors.index.json sam2.1_hiera_large.pt mask_tokenizer_256x2.pth; do
    [[ -f "$MODEL/$artifact" ]] || { echo "Missing SAMTok artifact: $MODEL/$artifact" >&2; exit 2; }
  done
  # Transformers/SAMTok releases use both names for the processor metadata.
  if [[ ! -f "$MODEL/processor_config.json" && ! -f "$MODEL/preprocessor_config.json" ]]; then
    echo "Missing SAMTok processor metadata: expected processor_config.json or preprocessor_config.json under $MODEL" >&2
    exit 2
  fi
fi
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }

export SAMTOK_BASE_CHECKPOINT="$MODEL"
export SAMTOK_SELECTIVE_DATA="$DATA"
export SAMTOK_STANDALONE_ADAPTER="$ADAPTER"
python3 -m projects.samtok_selective.manifests guard
python3 -m projects.samtok_selective.tail_gppo_contract \
  --repo-root "$FARO_ROOT" --adapter "$ADAPTER" --skip-model-hash

rjob submit --name="$JOB_NAME" --namespace=ailab-dnacoding \
  --cpu=20 --gpu=2 --memory=240000 \
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
/opt/vlm/bin/torchrun --standalone --nproc_per_node=2 \
  -m projects.samtok_selective.fepo_gr_cppo_trainer --config '$CONFIG' \
  2>&1 | tee '$FARO_ROOT/logs/samtok_selective/$JOB_NAME.log'
"
