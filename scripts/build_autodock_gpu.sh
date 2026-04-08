#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}/external/AutoDock-GPU"

module load cuda/12.8 2>/dev/null || true

make clean 2>/dev/null || true
make DEVICE=CUDA NUMWI=128 \
  GPU_INCLUDE_PATH="${CUDA_HOME}/include" \
  GPU_LIBRARY_PATH="${CUDA_HOME}/lib64" \
  -j8
mkdir -p "${ROOT_DIR}/.local/bin"
cp bin/autodock_gpu_128wi "${ROOT_DIR}/.local/bin/"
echo "AutoDock-GPU built: ${ROOT_DIR}/.local/bin/autodock_gpu_128wi"
