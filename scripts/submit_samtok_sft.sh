#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MSG'
Legacy PixVL trainer submission is disabled. Use a SAMTok-only standalone
SFT or FEPO entry point with the registered 5,120-row/10-step contract.
MSG
exit 2

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
CONFIG=${CONFIG:-$FARO_ROOT/Sa2VA/projects/pixvl_idea3/configs/samtok_selective_refseg_sft_2gpu.py}
JOB_NAME=${JOB_NAME:-dna-samtok-selective-sft-2g-r1}
GPU_COUNT=${GPU_COUNT:-2}
OUTPUT=${OUTPUT:-$ROOT/outputs/pixvl_idea3/samtok_selective_refseg_sft_2gpu}
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
[[ "${MODEL,,}" == *samtok* ]] || { echo "MODEL must be original SAMTok" >&2; exit 2; }
if (( GPU_COUNT < 1 || GPU_COUNT > 24 )); then echo "GPU_COUNT must be 1..24" >&2; exit 2; fi
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
gpu_csv=$(seq -s, 0 $((GPU_COUNT - 1)))
rjob submit --name="$JOB_NAME" --namespace=ailab-dnacoding \
  --cpu="$((GPU_COUNT * 10))" --gpu="$GPU_COUNT" --memory="$((GPU_COUNT * 80000))" \
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
export FARO_PROJECT_ROOT='$ROOT'; export FARO_WORKSPACE_ROOT='$ROOT'; export FARO_SAMTOK_MODEL='$MODEL'
export PYTHONPATH='$ROOT/.runtime_deps/huggingface_hub_1_21:$ROOT/third_party/transformers/src:$ROOT/third_party/Sa2VA:$ROOT:$FARO_ROOT/Sa2VA:$FARO_ROOT':\${PYTHONPATH:-}
mkdir -p '$ROOT/logs'
cd '$FARO_ROOT'
/opt/vlm/bin/torchrun --standalone --nproc_per_node='$GPU_COUNT' -m projects.pixvl_idea1.trainers.joint_sft_trainer --config '$CONFIG' 2>&1 | tee '$ROOT/logs/$JOB_NAME.log'
"
