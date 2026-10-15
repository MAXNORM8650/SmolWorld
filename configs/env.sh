# Source before running anything in this repo:  source configs/env.sh
# Paths for this machine; edit the four roots below when moving to another host.
export SW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SW_WM_ROOT="${SW_WM_ROOT:-/media/mbzuaiser/SSD1/Komal/Documents/WM}"          # worldlab + flex-pi + Lumos checkouts
export SW_PY="${SW_PY:-/nvme-data/flexpi-eval-venv/bin/python}"                     # torch 2.6, transformers 5.17, TensorRT 10.9
export SW_LIBERO_DATA="${SW_LIBERO_DATA:-/nvme-data/flexpi_repro/data/libero_mujoco3.3.2_depth}"

export SW_CKPT="$SW_ROOT/model/smolworld_100k.pt"
export SW_COSMOS="$SW_ROOT/model/cosmos_tokenizer"
export SW_INTRINSICS="$SW_ROOT/model/camera_intrinsics_libero.json"

export FLEXPI_ROOT="$SW_WM_ROOT/flex-pi"
export PYTHONPATH="$SW_ROOT:$SW_WM_ROOT/worldlab:$SW_WM_ROOT/worldlab/scripts/eval:$SW_WM_ROOT/Lumos/lumos-1/lumos-1:$FLEXPI_ROOT/experiments/libero:$FLEXPI_ROOT/third_party/LIBERO:$FLEXPI_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/media/mbzuaiser/SSD1/.cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
