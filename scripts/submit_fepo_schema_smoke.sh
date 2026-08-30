#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
JOB_NAME=${JOB_NAME:-dna-fepo-schema-smoke-r1}
LIMIT=${LIMIT:-16}
SCHEMA_ROOT=${SCHEMA_ROOT:-$FARO_ROOT/data/pixvl_idea3/schemas}
SCHEMA_ROOT=$(realpath -m "$SCHEMA_ROOT")
case "$SCHEMA_ROOT" in "$FARO_ROOT"/*) ;; *) echo "SCHEMA_ROOT must be under FARO_ROOT: $FARO_ROOT" >&2; exit 2 ;; esac
GPU_COUNT=${GPU_COUNT:-8}
CPU_COUNT=${CPU_COUNT:-$((GPU_COUNT * 20))}
MEMORY_MB=${MEMORY_MB:-$((GPU_COUNT * 120000))}
if (( GPU_COUNT < 8 || GPU_COUNT > 24 )); then
  echo "GPU_COUNT must be between 8 and 24" >&2
  exit 2
fi
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
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
cat > /etc/apt/sources.list <<'APT__EOF'
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-security main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-backports main restricted universe multiverse
APT__EOF
apt update; apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 || true
export PATH=/opt/vlm/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/compat/bin:\$PATH
export LD_LIBRARY_PATH=/opt/vlm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
cd '$ROOT'
export SAMTOK_MODEL_PATH='$MODEL'
export SAMTOK_MASK_TOKENIZER_PATH='$MODEL/mask_tokenizer_256x2.pth'
export SAMTOK_SAM2_PATH='$MODEL/sam2.1_hiera_large.pt'
export SAMTOK_SA2VA_ROOT='$FARO_ROOT/Sa2VA'
export PYTHONPATH='$ROOT/.runtime_deps/huggingface_hub_1_21:$ROOT/third_party/transformers/src:$ROOT/third_party/Sa2VA:$ROOT:$FARO_ROOT/Sa2VA:$FARO_ROOT':\${PYTHONPATH:-}
mkdir -p '$FARO_ROOT/logs'; echo FEPO_SCHEMA_START | tee '$FARO_ROOT/logs/$JOB_NAME.log'
/opt/vlm/bin/python '$FARO_ROOT/tools/build_pixvl_schema_smoke.py' \
  --seg-input '$ROOT/data/baseline_three_stage/selfsup_seg.jsonl' \
  --maskcap-input '$ROOT/data/baseline_three_stage/selfsup_maskcap.jsonl' \
  --output-root '$SCHEMA_ROOT' --limit '$LIMIT' 2>&1 | tee -a '$FARO_ROOT/logs/$JOB_NAME.log'
"
