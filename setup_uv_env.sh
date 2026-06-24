#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed."
  echo "Install it first, for example:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  echo "  source ~/.local/bin/env"
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found. Set CUDA_HOME to a CUDA Toolkit path, for example:" >&2
  echo "  export CUDA_HOME=/usr/local/cuda" >&2
  echo "  export PATH=\$CUDA_HOME/bin:\$PATH" >&2
  exit 1
fi

uv python install "${PYTHON_VERSION}"
uv venv .venv --python "${PYTHON_VERSION}"

uv pip install --python .venv/bin/python --upgrade pip setuptools wheel ninja
uv pip install --python .venv/bin/python \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "${PYTORCH_INDEX_URL}"

uv pip install --python .venv/bin/python plyfile tqdm opencv-python joblib
uv pip install --python .venv/bin/python \
  submodules/diff-gaussian-rasterization \
  submodules/simple-knn \
  submodules/fused-ssim

.venv/bin/python - <<'PY'
import torch
import diff_gaussian_rasterization
import simple_knn
import fused_ssim

print("Python/uv environment is ready.")
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Activate with:"
echo "  source .venv/bin/activate"
