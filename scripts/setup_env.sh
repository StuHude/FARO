#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${ENV_PREFIX:-/mnt/pfs/xiaoyicheng/envs/pixvl_idea1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
CLONE_FROM_ENV="${CLONE_FROM_ENV:-/mnt/pfs/miniconda3/envs/qwen35}"

export HF_HOME="${HF_HOME:-/mnt/pfs/xiaoyicheng/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/mnt/pfs/xiaoyicheng/.cache/pip}"
export TMPDIR="${TMPDIR:-/mnt/pfs/xiaoyicheng/.tmp}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" "$TMPDIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda 未安装，无法继续。"
  exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  if [[ -d "$CLONE_FROM_ENV" ]]; then
    conda create -y -p "$ENV_PREFIX" --clone "$CLONE_FROM_ENV"
  else
    conda create -y -p "$ENV_PREFIX" "python=${PYTHON_VERSION}" pip
  fi
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

python -m pip install --upgrade pip wheel setuptools

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  python -m pip install \
    --extra-index-url "$CUDA_INDEX_URL" \
    "torch>=2.6.0" \
    "torchvision>=0.21.0"
fi

python -m pip install \
  accelerate \
  datasets \
  huggingface_hub \
  peft \
  pycocoevalcap \
  transformers==4.57.1 \
  qwen_vl_utils \
  pycocotools \
  rouge-score \
  sentence-transformers \
  pillow \
  numpy \
  scipy \
  safetensors \
  pyyaml \
  tqdm \
  jsonlines \
  opencv-python-headless \
  scikit-image \
  xlsxwriter \
  tabulate

python - <<PY
from pathlib import Path
import site

repo_root = Path("$ROOT_DIR/Sa2VA").resolve()
pth_path = Path(site.getsitepackages()[0]) / "sa2va_local_repo.pth"
pth_path.write_text(str(repo_root) + "\n", encoding="utf-8")
print(f"Wrote {pth_path} -> {repo_root}")
PY

cat <<EOF
环境安装完成。

激活方式：
  eval "\$(conda shell.bash hook)"
  conda activate "$ENV_PREFIX"

建议同时设置：
  export HF_HOME=$HF_HOME
  export TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
  export HF_DATASETS_CACHE=$HF_DATASETS_CACHE
EOF
