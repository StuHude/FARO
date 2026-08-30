#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/PixVL_ailab}
MODEL=${MODEL:-$ROOT/checkpoints/SAMTok/Qwen3-VL-4B-SAMTok-co}
FARO_ROOT=${FARO_ROOT:-/mnt/shared-storage-user/dnacoding/wuyucheng/workspace/Nemotrontiaozheng/Faro_ailab/FARO}
IMAGE=${IMAGE:-/mnt/shared-storage-user/dnacoding/wuyucheng/dataset/refcoco/train2014/COCO_train2014_000000526754.jpg}
JOB_NAME=${JOB_NAME:-dna-fepo-evidence-sweep-r1}
GPU_COUNT=${GPU_COUNT:-8}
CPU_COUNT=${CPU_COUNT:-$((GPU_COUNT * 20))}
MEMORY_MB=${MEMORY_MB:-$((GPU_COUNT * 120000))}
TAGS_FILE=${TAGS_FILE:-$FARO_ROOT/rjob_tags.txt}
[[ -f "$TAGS_FILE" ]] || { echo "Missing positive-tag file: $TAGS_FILE" >&2; exit 2; }
POSITIVE_TAGS=$(sed '/^[[:space:]]*#/d;/^[[:space:]]*$/d' "$TAGS_FILE" | paste -sd, -)
[[ -n "$POSITIVE_TAGS" ]] || { echo "No positive tags in $TAGS_FILE" >&2; exit 2; }
if (( GPU_COUNT < 8 || GPU_COUNT > 24 )); then
  echo "GPU_COUNT must be between 8 and 24" >&2
  exit 2
fi
case "$JOB_NAME" in dna-*) ;; *) echo "JOB_NAME must start with dna-" >&2; exit 2 ;; esac

rjob submit \
  --name="$JOB_NAME" \
  --namespace=ailab-dnacoding \
  --cpu="$CPU_COUNT" --gpu="$GPU_COUNT" --memory="$MEMORY_MB" \
  --positive-tags="$POSITIVE_TAGS" \
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
mkdir -p '$FARO_ROOT/logs'; echo FEPO_EVIDENCE_SWEEP_START | tee '$FARO_ROOT/logs/$JOB_NAME.log'
/opt/vlm/bin/python - <<'PY' 2>&1 | tee -a '$FARO_ROOT/logs/$JOB_NAME.log'
import json
from pathlib import Path
import numpy as np
from PIL import Image
from trackcycle.modeling.hf_compat import ensure_huggingface_hub_compat
ensure_huggingface_hub_compat()
from trackcycle.modeling.samtok_qwen3vl import SAMTokQwen3VLAdapter
from projects.pixvl_idea3.failure_evidence import predicted_only_evidence_route

image = np.asarray(Image.open(Path('$IMAGE')).convert('RGB'))
adapter = SAMTokQwen3VLAdapter(device='cuda')
adapter._ensure_loaded()
cases = [
    ('refseg_geometry', 'Please segment the zebra creature front and center in this image.', 'refseg', 0.25),
    ('refseg_relation', 'Please segment the object to the left of the zebra creature.', 'refseg', 0.50),
    ('maskcap_semantic', 'Given a detailed description of this region, describe its color, material, and object category.', 'maskcap', 0.70),
    ('maskcap_relation', 'Given a detailed description of this region, state whether it is left of or behind another object.', 'maskcap', 0.55),
    ('no_target', 'Please segment the nonexistent transparent unicorn that is not present.', 'refseg', 0.20),
]
rows = []
for name, prompt, task, conf in cases:
    text = adapter._generate_text(image, prompt, max_new_tokens=32, do_sample=False)
    route = predicted_only_evidence_route(prompt_text=prompt, predicted_text=text, task=task, answer_confidence=conf)
    rows.append({'case': name, 'prompt': prompt, 'text': text, 'route': route})
    print('FEPO_EVIDENCE_CASE', json.dumps(rows[-1], ensure_ascii=False), flush=True)
print('FEPO_EVIDENCE_SWEEP_OK', json.dumps(rows, ensure_ascii=False), flush=True)
PY
"
