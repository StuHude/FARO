#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
JOB_NAME=${JOB_NAME:-dna-fepo-contract-smoke}
CPU=${CPU:-8}
GPU=${GPU:-8}
MEMORY=${MEMORY:-32768}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
if (( GPU < 8 || GPU > 24 )); then
  echo "GPU must be between 8 and 24" >&2
  exit 2
fi

case "$JOB_NAME" in
  dna-*) ;;
  *) echo "JOB_NAME must start with dna-: $JOB_NAME" >&2; exit 2 ;;
esac

rjob submit \
  --name "$JOB_NAME" \
  --namespace=ailab-dnacoding \
  --cpu="$CPU" \
  --gpu="$GPU" \
  --memory="$MEMORY" \
  --positive-tags="$POSITIVE_TAGS" \
  --charged-group=dnacoding_gpu \
  --private-machine=group \
  --mount=gpfs://gpfs1/dnacoding:/mnt/shared-storage-user/dnacoding \
  --mount=gpfs://gpfs1/wuyucheng:/mnt/shared-storage-user/wuyucheng \
  --image registry.h.pjlab.org.cn/ailab-dnacoding/wuyucheng:test1 \
  --custom-resources brainpp.cn/fuse=1 \
  --enable-sshd \
  -- bash -lc "
set -euo pipefail
set -x
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
cd /opt
mkdir -p /opt/vlm
if [ -f /opt/vlm_env.tar.gz ]; then
  tar -xzf /opt/vlm_env.tar.gz -C /opt/vlm
  rm -f /opt/vlm_env.tar.gz
elif [ -f vlm_env.tar.gz ]; then
  tar -xzf vlm_env.tar.gz -C /opt/vlm
  rm -f vlm_env.tar.gz
fi
/opt/vlm/bin/python /opt/vlm/bin/conda-unpack || true
cat > /etc/apt/sources.list <<'APT__EOF'
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-security main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb http://mirrors.h.pjlab.org.cn/ubuntu/ jammy-backports main restricted universe multiverse
APT__EOF
apt update
apt install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 || true
export PATH=/opt/vlm/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin:/opt/tensorrt/bin:\$PATH
export LD_LIBRARY_PATH=/opt/vlm/lib:/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/lib/python3.12/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:\${LD_LIBRARY_PATH:-}
/opt/vlm/bin/python -c \"import torch, transformers; print('ok')\"
cd '$ROOT'
export PYTHONPATH='$ROOT/Sa2VA:$ROOT':\${PYTHONPATH:-}
/opt/vlm/bin/python tools/smoke_fepo.py
"
