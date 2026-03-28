#!/bin/bash
# Bochen runner — sets up CUDA paths for JAX and runs the given command
VENV_DIR="$(dirname "$0")/.venv2"
NVIDIA_LIBS="$VENV_DIR/lib/python3.12/site-packages/nvidia"

export LD_LIBRARY_PATH="$NVIDIA_LIBS/cublas/lib:$NVIDIA_LIBS/cusparse/lib:$NVIDIA_LIBS/cuda_runtime/lib:$NVIDIA_LIBS/cusolver/lib:$NVIDIA_LIBS/cufft/lib:$NVIDIA_LIBS/cudnn/lib:$NVIDIA_LIBS/nccl/lib:$NVIDIA_LIBS/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

exec "$VENV_DIR/bin/python" "$@"
