#!/usr/bin/env bash
set -euo pipefail

FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
PIXVL_ROOT=${PIXVL_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
MODEL=${MODEL:-$PIXVL_ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
CONTINUED=${CONTINUED:?set CONTINUED to a finished FARO standalone adapter}
CANDIDATE=${CANDIDATE:-${V5:-}}
[[ -n "$CANDIDATE" ]] || { echo "Set CANDIDATE to a finished FARO standalone adapter" >&2; exit 2; }
TRAIN_SCHEMA=${TRAIN_SCHEMA:-$FARO_ROOT/data/fepo_existence/grefcoco_selective_train_256.jsonl}
HOLDOUT_SCHEMA=${HOLDOUT_SCHEMA:-$FARO_ROOT/data/fepo_existence/grefcoco_selective_holdout_256.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$FARO_ROOT/evals/action_margin_diag}
GPU_COUNT=${GPU_COUNT:-8}
JOB_NAME=${JOB_NAME:-dna-samtok-action-margin-${GPU_COUNT}g}
# The action-margin diagnostic may run on the 16-GPU partition when it is
# downgraded below eight GPUs.  Its positive-tag allowlist is distinct from
# the regular namespace list; do not let a stale caller override that choice.
if (( GPU_COUNT < 8 )); then
  TAGS_FILE=$FARO_ROOT/rjob_tags_16gpu_partition.txt
else
  TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
fi

case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
if (( GPU_COUNT < 1 || GPU_COUNT > 8 )); then echo "GPU_COUNT must be 1..8" >&2; exit 2; fi
for path in "$MODEL/config.json" "$TRAIN_SCHEMA" "$HOLDOUT_SCHEMA" "$TAGS_FILE"; do
  [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
python3 "$FARO_ROOT/tools/action_margin_contract.py" validate-adapters \
  --output-root "$FARO_ROOT/outputs/samtok_selective" "$CONTINUED" "$CANDIDATE"
python3 "$FARO_ROOT/tools/action_margin_contract.py" validate-schemas \
  --train "$TRAIN_SCHEMA" --holdout "$HOLDOUT_SCHEMA"
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }

rjob submit --name="$JOB_NAME" --namespace=ailab-dnacoding \
  --cpu="$((GPU_COUNT * 8))" --gpu="$GPU_COUNT" --memory="$((GPU_COUNT * 60000))" \
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
mkdir -p '$OUTPUT_ROOT'
pids=()
for rank in \$(seq 0 $((GPU_COUNT - 1))); do
  (
    export CUDA_VISIBLE_DEVICES=\$rank
    for split in train holdout; do
      if [ \"\$split\" = train ]; then schema='$TRAIN_SCHEMA'; else schema='$HOLDOUT_SCHEMA'; fi
      for policy in continued candidate; do
        if [ \"\$policy\" = continued ]; then adapter='$CONTINUED'; else adapter='$CANDIDATE'; fi
        /opt/vlm/bin/python '$FARO_ROOT/tools/score_samtok_action_margin.py' \
          --model '$MODEL' --adapter \"\$adapter\" --schema \"\$schema\" \
          --policy \"\$policy\" \
          --task-id \$rank --num-tasks '$GPU_COUNT' \
          --output '$OUTPUT_ROOT'/\${split}_\${policy}_part_\${rank}.json
      done
    done
  ) > '$OUTPUT_ROOT'/rank_\${rank}.log 2>&1 &
  pids+=(\$!)
done
for pid in \"\${pids[@]}\"; do wait \$pid; done
touch '$OUTPUT_ROOT/_SUCCESS_${GPU_COUNT}g'
"
