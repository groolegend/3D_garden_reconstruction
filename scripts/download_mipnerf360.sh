#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-data/mipnerf360}"
SCENE="${2:-garden}"
ARCHIVE="${ROOT_DIR}/${SCENE}.zip"
URL="https://storage.googleapis.com/gresearch/refraw360/${SCENE}.zip"

mkdir -p "${ROOT_DIR}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "Downloading Mip-NeRF 360 ${SCENE} scene to ${ARCHIVE}"
  curl -L "${URL}" -o "${ARCHIVE}"
else
  echo "Archive already exists: ${ARCHIVE}"
fi

echo "Extracting ${ARCHIVE}"
unzip -n "${ARCHIVE}" -d "${ROOT_DIR}"

if [[ -d "${ROOT_DIR}/${SCENE}" ]]; then
  echo "${SCENE} scene is ready at ${ROOT_DIR}/${SCENE}"
else
  echo "Could not find ${ROOT_DIR}/${SCENE} after extraction" >&2
  exit 1
fi
