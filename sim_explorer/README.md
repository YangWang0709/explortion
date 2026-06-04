# Stage 4A-1 Simulator Smoke

This directory contains the minimal Isaac depth-observation smoke test for
continuous exploration.

Scope:

- Launch Isaac Lab/Isaac Sim headlessly.
- Build a tiny indoor-like scene.
- Capture `distance_to_image_plane` depth at three fixed camera poses.
- Convert depth frames into an observed voxel map with states:
  `UNKNOWN = -1`, `FREE = 0`, `OCCUPIED = 1`.

Out of scope:

- SSCNet / PredictionLayer integration.
- Paper expert scoring.
- RL, PPO, imitation-learning training, or behavior cloning training.
- Writing predictions into `observed_map`.

Typical commands:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/sc_explorer_ws/sim_explorer

python test_depth_to_voxel.py

cd /home/ubuntu22/IsaacLab
export TERM=xterm
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY

python /home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py \
  --headless --enable_cameras

cd /home/ubuntu22/sc_explorer_ws/sim_explorer
python depth_to_voxel.py \
  --input_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke
```
