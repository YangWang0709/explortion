# Environment Record

## Conda

- Active project environment: `env_isaaclab`

## Probed Package State

- Python: `Python 3.11.15`
- Torch: `2.7.0+cu128`
- Torch CUDA available: `True`
- IsaacSim module path: `/home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/__init__.py`

Probe commands used:

```bash
python -V
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import isaacsim; print(isaacsim.__file__)"
```

## Headless IsaacSim Environment Variables

Expected headless variables:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY
unset WAYLAND_DISPLAY
unset XAUTHORITY
unset GNOME_SETUP_DISPLAY
```

## Current Gates

- `human_visual_inspection_done=false`
- `formal_expert_sampling_ready=false`
- `full_expert_dataset_ready=false`
- `stage4a66d_executed=false`
- `stage4a67_executed=false`
