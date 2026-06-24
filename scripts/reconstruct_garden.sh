#!/usr/bin/env bash
set -euo pipefail

MIPNERF360_ROOT="${1:-data/mipnerf360}"
OUTPUT_ROOT="${2:-output/mipnerf360}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))
SCENE_DIR="${MIPNERF360_ROOT}/garden"
MODEL_DIR="${OUTPUT_ROOT}/garden"

if [[ ! -d "${SCENE_DIR}" ]]; then
  echo "Missing garden scene: ${SCENE_DIR}" >&2
  echo "Download it first with: bash scripts/download_mipnerf360.sh ${MIPNERF360_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SCENE_DIR}/images_4" ]]; then
  echo "Missing ${SCENE_DIR}/images_4; garden is an outdoor Mip-NeRF 360 scene and official evaluation uses images_4." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Warning: nvidia-smi was not found. Official 3DGS training requires an NVIDIA CUDA GPU." >&2
fi

mkdir -p "${OUTPUT_ROOT}"

python train.py \
  -s "${SCENE_DIR}" \
  -i images_4 \
  -m "${MODEL_DIR}" \
  --eval \
  --disable_viewer \
  --test_iterations 7000 30000 \
  --save_iterations 7000 30000 \
  "$@"

python render.py \
  -s "${SCENE_DIR}" \
  -m "${MODEL_DIR}" \
  --eval

python metrics.py -m "${MODEL_DIR}"

echo "Garden reconstruction written to ${MODEL_DIR}"
