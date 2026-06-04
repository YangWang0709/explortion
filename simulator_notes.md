# Simulator Notes

Updated: 2026-05-30

## 1. env_isaaclab Status

All Python checks for Stage 4A-1 were run inside `env_isaaclab`.

```text
CONDA_DEFAULT_ENV=env_isaaclab
python: /home/ubuntu22/miniconda3/envs/env_isaaclab/bin/python
Python: 3.11.15
pip: /home/ubuntu22/miniconda3/envs/env_isaaclab/bin/pip
pip: 26.1.1
torch: OK 2.7.0+cu128
isaaclab: OK 0.54.3
isaacsim: OK, package metadata 5.1.0.0
omni: OK
pxr: direct pre-launch import failed, but pxr modules load after Isaac/Kit startup
GPU: NVIDIA GeForce RTX 5080, driver 580.159.03, CUDA runtime 13.0
```

Raw log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a1_env_status.log
```

## 2. Isaac Installation Discovery

Isaac Lab repository:

```text
/home/ubuntu22/IsaacLab
commit: 090aed18163b2194d5551c7919f7539283677743
status: clean
launcher: /home/ubuntu22/IsaacLab/isaaclab.sh
```

Isaac Sim is installed as pip packages in `env_isaaclab`:

```text
path: /home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim
package metadata: isaacsim 5.1.0.0, isaacsim-rl 5.1.0.0, isaacsim-kernel 5.1.0.0
VERSION file: 5.1.0-rc.19+release.26219.9c81211b.gl
ISAAC_PATH: /home/ubuntu22/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim
```

Relevant official examples found:

```text
/home/ubuntu22/IsaacLab/scripts/tutorials/00_sim/create_empty.py
/home/ubuntu22/IsaacLab/scripts/tutorials/04_sensors/run_usd_camera.py
/home/ubuntu22/IsaacLab/scripts/tutorials/04_sensors/add_sensors_on_robot.py
/home/ubuntu22/IsaacLab/scripts/tutorials/04_sensors/run_ray_caster_camera.py
/home/ubuntu22/IsaacLab/scripts/demos/sensors/cameras.py
```

## 3. Isaac Smoke Test Results

Empty scene smoke:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/IsaacLab
export TERM=xterm
export PYTHONUNBUFFERED=1
timeout --kill-after=20s 60s ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

Result:

```text
headless IsaacLab app launched
experience: /home/ubuntu22/IsaacLab/apps/isaaclab.python.headless.kit
observed marker: [INFO]: Setup complete...
log: /home/ubuntu22/sc_explorer_ws/logs/isaac_empty_scene_smoke.log
```

The official tutorial is an infinite loop, so it was stopped after proving setup.

Camera/depth sensor smoke:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/IsaacLab
export TERM=xterm
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY
timeout --kill-after=20s 100s python scripts/tutorials/04_sensors/run_usd_camera.py --headless --enable_cameras
```

Result:

```text
headless rendering experience launched
experience: /home/ubuntu22/IsaacLab/apps/isaaclab.python.headless.rendering.kit
depth key: distance_to_image_plane
official depth shape: torch.Size([2, 480, 640, 1])
log: /home/ubuntu22/sc_explorer_ws/logs/isaac_sensor_smoke.log
```

Important startup note:

```text
Using the inherited desktop DISPLAY/WAYLAND variables caused GLXBadFBConfig.
The working headless camera command unsets DISPLAY/WAYLAND/XAUTHORITY and pins
Vulkan to /usr/share/vulkan/icd.d/nvidia_icd.json.
```

## 4. Minimal Depth Scene

Created external workspace scripts, without modifying IsaacLab source:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/README.md
/home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_depth_to_voxel.py
```

Minimal scene command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/IsaacLab
export TERM=xterm
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY

timeout --kill-after=30s 240s python /home/ubuntu22/sc_explorer_ws/sim_explorer/minimal_depth_scene.py \
  --headless --enable_cameras
```

Result:

```text
exit status: 0
scene: 8m x 8m floor, two side walls, back wall, two cuboid obstacles
camera data type: distance_to_image_plane
camera resolution: 160 x 120
camera max depth: 5m
poses:
  pose 0: center, yaw 0 deg
  pose 1: center, yaw 90 deg
  pose 2: shifted forward, yaw 0 deg
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a1_minimal_depth_scene.log
```

Depth outputs:

```text
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke
depth_000.npy: shape (120, 160), dtype float32, min 1.3499999, max 3.9250004
depth_001.npy: shape (120, 160), dtype float32, min 1.6134452, max 3.9250004
depth_002.npy: shape (120, 160), dtype float32, min 0.3499999, max 2.9250004
camera_info.json
pose_000.json
pose_001.json
pose_002.json
scene_metadata.json
```

## 5. Depth to Observed Voxel Map

The observed map implementation is pure Python/Numpy and independent of
PredictionLayer, SSCNet, target labels, expert scoring, and ground truth.

Command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/sc_explorer_ws/sim_explorer
python depth_to_voxel.py \
  --input_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke \
  --voxel_size 0.1 \
  --pixel_stride 2
```

Result:

```text
voxel_size: 0.1m
map bounds: x [-4, 4], y [-4, 4], z [0, 3]
observed map shape: (80, 80, 30)
UNKNOWN = -1, FREE = 0, OCCUPIED = 1
unknown_count: 143335
free_count: 44435
occupied_count: 4230
observed_ratio: 0.2534635416666667
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel.log
```

Outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step0.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step1.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_summary.json
```

Pure Python test:

```text
command: python test_depth_to_voxel.py
result: depth_to_voxel tests passed
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a1_depth_to_voxel_test.log
```

## 6. Stage 4A-1 Results

Completed:

```text
Isaac Lab and Isaac Sim availability checked in env_isaaclab.
IsaacLab repo and Isaac Sim pip install paths identified.
Official headless empty scene launched.
Official USD camera tutorial produced distance_to_image_plane depth.
Minimal indoor-like depth scene created outside IsaacLab source.
Three fixed camera poses saved depth images and pose/camera metadata.
Depth frames fused into a simple measured-only observed voxel map.
Smoke outputs and logs saved.
```

Boundaries preserved:

```text
No RL or PPO.
No imitation-learning or behavior-cloning training.
No SSCNet / PredictionLayer integration.
No expert scorer integration.
No prediction writes into observed_map.
No target_lr, target_hr, or ground-truth map use.
No AirSim or Unreal.
No full online planner.
```

Decision:

```text
Stage 4A-1 only validates simulator depth observation and observed_map update.
Prediction and expert integration are Stage 4A-2.
```

## 7. Stage 4A-1 Scene Visualization

Stage 4A-1-scene-viz is complete. This step renders the scripted Isaac
minimal scene itself, not only the observed voxel map. It does not modify
`observed_state_step*.npy`.

Render script:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/render_minimal_scene_views.py
```

Command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/IsaacLab
export TERM=xterm
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY

python /home/ubuntu22/sc_explorer_ws/sim_explorer/render_minimal_scene_views.py \
  --headless \
  --enable_cameras \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_scene_viz \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage4a1_scene_viz.log
```

Output directory:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_scene_viz
```

Generated scene images:

```text
camera_rgb_000.png
camera_rgb_001.png
camera_rgb_002.png
camera_depth_color_000.png
camera_depth_color_001.png
camera_depth_color_002.png
scene_overview_rgb.png
scene_overview_depth_color.png
scene_layout_topdown.png
scene_metadata.json
scene_viz_summary.json
```

Camera data keys:

```text
RGB key used: rgb
Depth key used: distance_to_image_plane
Main camera RGB shape: 120 x 160 x 3
Overview RGB shape: 480 x 640 x 3
Main camera max depth: 5m
Overview camera max depth: 12m
```

Scene summary:

```text
floor: 8m x 8m, x/y bounds [-4, 4]
walls: left y=-4, right y=4, back x=4, height 2m
obstacles:
  center cuboid size [0.7, 0.7, 1.2], position [1.7, 0.35, 0.6]
  back cuboid size [0.8, 1.2, 1.5], position [3.2, -1.2, 0.75]
camera poses:
  pose 0: position [0, 0, 1.2], yaw 0 deg
  pose 1: position [0, 0, 1.2], yaw 90 deg
  pose 2: position [1, 0, 1.2], yaw 0 deg
overview:
  position [0, -6.5, 5], target [1.5, 0, 0.7]
```

Validation:

```text
Required files exist.
RGB PNGs are true RGB mode.
scene_viz_summary.json contains no NaN.
Camera RGB stats are non-uniform:
  camera_rgb_000 std 77.74
  camera_rgb_001 std 87.15
  camera_rgb_002 std 51.52
  scene_overview_rgb std 66.26
Depth color images include matplotlib colorbars and min/max titles.
```

Headless log:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a1_scene_viz.log
```

Observed warnings:

```text
Isaac headless startup still prints GLFW/default-display warnings after
DISPLAY/WAYLAND are unset. Rendering succeeds through Vulkan with the NVIDIA
ICD, and the RGB/depth outputs are non-empty.
```

Boundaries:

```text
No RL or PPO.
No behavior-cloning or imitation-learning training.
No SSCNet / PredictionLayer integration.
No expert scoring.
No observed_map or observed_state modification.
```

## 8. Stage 4A-1 Observed Map Visualization

Stage 4A-1-viz is complete. This step only visualized existing Stage 4A-1
outputs and did not modify any `observed_state_step*.npy` file.

Visualization script:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_observed_map.py
```

Command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/sc_explorer_ws/sim_explorer

python visualize_observed_map.py \
  --input_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke_viz \
  --max_points 5000 \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage4a1_visualize_observed_map.log
```

Output directory:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke_viz
```

Generated visualizations:

```text
depth_000.png
depth_001.png
depth_002.png
depth_grid.png
observed_topdown_step0.png
observed_topdown_step1.png
observed_topdown_step2.png
observed_topdown_compare.png
occupied_voxels_3d_step2.png
free_occupied_voxels_3d_step2.png
slices_step2.png
index.html
viz_summary.json
```

Observed map counts:

```text
shape: (80, 80, 30)
step0: unknown=170910, free=19335, occupied=1755, observed_ratio=0.10984375
step1: unknown=143439, free=44515, occupied=4046, observed_ratio=0.252921875
step2: unknown=143335, free=44435, occupied=4230, observed_ratio=0.2534635416666667
```

Open3D / PLY:

```text
observed_step2_pointcloud.ply was skipped because open3d is not installed:
ModuleNotFoundError("No module named 'open3d'")
```

Limitations:

```text
Visualization uses matplotlib static PNGs and a small HTML index.
3D scatter plots subsample free voxels to at most 5000 points.
This step does not render a new Isaac camera image and does not connect
SSCNet, PredictionLayer, expert scoring, RL, or training.
```

## 9. Stage 4A-2 Simulator Observed Map Expert Step

Stage 4A-2 is complete. This stage connects one Isaac measured-only observed
voxel map to a SC-Explorer-style paper expert candidate scorer:

```text
observed_state_step2.npy
  -> frontier / candidate generation
  -> observed-map raycast visibility
  -> paper gain scoring with EmptyPredictionLayer
  -> best next viewpoint selection
  -> top-N decision and visualization
```

New scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
```

Input:

```text
observed_state: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy
observed_summary: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_summary.json
camera_info: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/camera_info.json
pose_json: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/pose_002.json
scene_metadata: /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/scene_metadata.json
```

Coordinate convention:

```text
observed_state[i, j, k] axis order is x, y, z.
i -> x, j -> y, k -> z.
voxel center = min_bound + (index + 0.5) * voxel_size.
map bounds: x [-4, 4], y [-4, 4], z [0, 3].
voxel_size: 0.1m.
pose_002.json contains position [1.0, 0.0, 1.2] and yaw_rad 0.0, so no yaw fallback was needed.
```

Run command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/sc_explorer_ws/sim_explorer

python run_sim_expert_step.py \
  --observed_state /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy \
  --observed_summary /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_summary.json \
  --camera_info /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/camera_info.json \
  --pose_json /home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/pose_002.json \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke \
  --num_candidates 64 \
  --top_n 16 \
  --gain_mode hybrid \
  --prediction_mode empty \
  --print_topn \
  --save_viz \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_step.log
```

Observed map stats:

```text
shape: (80, 80, 30)
unknown_count: 143335
free_count: 44435
occupied_count: 4230
observed_ratio: 0.2534635416666667
current pose world: [1.0, 0.0, 1.2]
current pose grid: [50, 40, 11]
current yaw: 0.0 rad
```

Frontier / candidates:

```text
frontier_count: 5929
frontier_adjacent_free_count: 5876
candidates: 64
top_n: 16
raycast_mode: observed_conservative_unknown_blocking
prediction_mode: empty
gain_mode: hybrid
```

Best candidate:

```text
expert_action: 0
candidate id: 63
best score: 88.83270299135849
gain_exp: 73.0
gain_sc: 0.0
gain_hybrid: 73.0
gain_occ: 0.0
gain_conf: 0.0
path_cost: 0.8217694333482268
grid position: [51, 38, 14]
world position: [1.15, -0.15, 1.45]
yaw: -0.7030942394487684 rad
```

Outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_step_decision.npz
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_step_decision.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_step_candidates.jsonl
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_topdown.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke/expert_score_bar.png
```

Visualization:

```text
expert_topdown.png: 1620 x 1440 PNG.
expert_score_bar.png: 1800 x 900 PNG.
Topdown shows unknown/free/occupied, current pose, frontier voxels,
sampled candidates, top-N candidates, best candidate, and current-to-best arrow.
Score bar shows top-N final_score values and includes gain_mode/prediction_mode.
```

Smoke test:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/sc_explorer_ws/sim_explorer

python test_sim_paper_expert.py \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage4a2_sim_expert_test.log
```

Test result:

```text
Stage 4A-2 simulator expert smoke test passed.
frontier_count: 5929
frontier_adjacent_free_count: 5876
candidate_count: 64
expert_action: 0
observed_state_modified: no
rl_or_optimizer_or_policy_training_run: no
```

Strict boundaries:

```text
Default prediction layer is EmptyPredictionLayer.
gain_sc, gain_occ, and gain_conf are 0 under EmptyPredictionLayer.
gain_hybrid equals gain_exp under EmptyPredictionLayer.
Prediction is never written into observed_state.
observed_state_step2.npy was hash-checked and not modified.
No SSCNet inference on Isaac depth was run.
No NYU target_lr/target_hr was read.
No scene ground truth or simulator ground truth was used.
No RL, PPO, optimizer step, policy training, behavior cloning, or imitation-learning training was run.
```

Limitations:

```text
This is a one-step expert decision only, not a continuous rollout.
PredictionLayer / SSCNet connection remains future work.
Candidates are sampled from observed FREE frontier-adjacent voxels, with FREE fallback.
Path cost is Euclidean grid distance plus yaw time; no A* or collision-checked path planner yet.
Raycast uses conservative UNKNOWN blocking through the measured observed map.
```

## 10. Stage 4A-3 Empty-Prediction Expert Rollout

Stage 4A-3 is complete. It turns the Stage 4A-2 one-step empty-prediction
expert into a deterministic multi-step Isaac rollout:

```text
current camera pose -> capture Isaac RGB/depth -> update measured observed_map
  -> run sim_paper_expert -> select best candidate -> teleport camera
  -> repeat
```

New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
```

The depth module now exposes `update_observed_state_from_depth(...)`, a
rollout-facing wrapper around the existing measured-only single-frame depth
fusion. The original depth-to-voxel CLI remains available.

Run command:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/IsaacLab
export TERM=xterm
export PYTHONUNBUFFERED=1
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
unset DISPLAY WAYLAND_DISPLAY XAUTHORITY GNOME_SETUP_DISPLAY

timeout --kill-after=30s 600s python /home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py \
  --headless --enable_cameras \
  --output_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred \
  --episode_id minimal_room_empty_pred_000 \
  --max_steps 10 \
  --num_candidates 64 \
  --top_n 16 \
  --gain_mode hybrid \
  --prediction_mode empty \
  --motion_mode planar \
  --camera_height 1.2 \
  --voxel_size 0.1 \
  --pixel_stride 2 \
  --save_rgb \
  --save_depth \
  --save_viz \
  --print_steps \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_empty_pred.log
```

Rollout setup:

```text
scene: same minimal indoor-like room as minimal_depth_scene.py
prediction mode: empty / EmptyPredictionLayer
motion mode: planar teleport camera
camera height: 1.2m
max_steps: 10
voxel_size: 0.1m
pixel_stride: 2
map bounds: x/y [-4,4], z [0,3]
```

Rollout result:

```text
episode id: minimal_room_empty_pred_000
steps completed: 10
done reason: max_steps
observed_ratio start: 0.0
observed_ratio end: 0.21754166666666666
total delta observed_ratio: 0.21754166666666666
final unknown/free/occupied: 150232 / 35873 / 5895
final pose: [3.549999952316284, 3.25, 1.2000000476837158]
final yaw: -2.8477112304002925 rad
repeated_pose_count: 2
average frontier_count: 4525.6
average candidates: 64.0
best_score min/mean/max: 29.41531522194122 / 105.48766454499457 / 190.10038228379815
gain_exp min/mean/max: 37.0 / 63.8 / 89.0
gain_sc min/mean/max: 0.0 / 0.0 / 0.0
path_cost min/mean/max: 0.2 / 0.8726943498167937 / 2.6516798957102083
```

Outputs:

```text
episode dir:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000

per-step transitions:
step_000.npz ... step_009.npz

jsonl:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/transitions.jsonl

summary:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/episode_summary.json

final observed map:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/observed_state_final.npy

visualizations:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/rollout_topdown_path.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/observed_ratio_curve.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/frontier_count_curve.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/step_topdown_000.png ... step_topdown_009.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000/rollout_index.html

global manifest:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/manifest.jsonl
```

Smoke test:

```bash
source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu22/sc_explorer_ws/sim_explorer

python test_sim_expert_rollout.py \
  --episode_dir /home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000 \
  2>&1 | tee /home/ubuntu22/sc_explorer_ws/logs/stage4a3_rollout_test.log
```

Test result:

```text
Stage 4A-3 rollout smoke test passed.
synthetic_transition_serialization: ok
real_episode_validation: ok, steps 10
observed_ratio_non_decreasing: yes
gain_sc_empty_prediction: zero
prediction_writes_observed_map: no
rl_optimizer_bc_training_run: no
```

Strict boundaries:

```text
observed_state is initialized all UNKNOWN.
FREE/OCCUPIED updates come only from Isaac depth ray marching.
candidate generation uses only observed_state.
EmptyPredictionLayer is the only prediction layer used.
gain_sc is 0 and gain_hybrid equals gain_exp under EmptyPredictionLayer.
Prediction is never written into observed_state.
No SSCNet inference on Isaac depth was run.
No NYU target_lr/target_hr was used.
No scene ground truth or simulator ground truth was used.
No RL, PPO, optimizer step, policy training, behavior cloning training, or
imitation-learning training was run.
No A* or physical collision-checked robot path execution was run.
```

Limitations:

```text
Camera motion is planar teleport motion, not physical robot execution.
No A* over observed FREE space yet.
No full SC-Explorer RRT tree planner yet.
Prediction remains EmptyPredictionLayer only.
The scene is still the minimal synthetic room.
```

## 11. Stage 4A-3.2 Medium-Complexity Scripted Scene

Stage 4A-3.2 is complete. It increases the scripted Isaac scene complexity
before scaling rollouts, while keeping the pipeline limited to scene
construction, fixed-pose depth/RGB capture, measured-only observed-map fusion,
visualization, and an optional one-step EmptyPredictionLayer expert smoke.

New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/scene_factory.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/medium_complex_depth_scene.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/render_medium_complex_scene_views.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_medium_complex_scene_metadata.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/depth_to_voxel.py
```

Scene summary:

```text
scene_id: medium_complex_three_rooms
variant: three_rooms
seed: 0
bounds: x/y [-6, 6], z [0, 3]
floor: 12m x 12m
wall height: 2.2m
rooms: 3
corridors: 1
openings: 3
walls: 13 cuboid wall segments
obstacles: 13 cuboid furniture/obstacle boxes
camera poses: 5 fixed smoke poses
camera height: 1.2m
main camera: 160 x 120
overview camera: 640 x 480
depth max range: 8m for main camera
```

Smoke outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke
depth_000.npy ... depth_004.npy
rgb_000.png ... rgb_004.png
pose_000.json ... pose_004.json
camera_info.json
scene_metadata.json
observed_state_step0.npy ... observed_state_step4.npy
observed_state_final.npy
observed_summary.json
```

Depth and observed-map result:

```text
depth shape: (120, 160)
observed map shape: (120, 120, 30)
voxel_size: 0.1m
pixel_stride: 2
map bounds: x/y [-6,6], z [0,3]
unknown/free/occupied: 339813 / 86064 / 6123
observed_ratio: 0.21339583333333334
```

Visualization outputs:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_viz
scene_overview_rgb.png
scene_overview_depth_color.png
scene_layout_topdown.png
camera_rgb_000.png ... camera_rgb_004.png
camera_depth_color_000.png ... camera_depth_color_004.png
camera_rgb_grid.png
camera_depth_grid.png
observed_topdown_compare.png
free_occupied_voxels_3d_final.png
slices_final.png
scene_viz_summary.json
```

Optional one-step expert smoke:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_smoke
observed_state: observed_state_step4.npy
prediction_mode: empty
gain_mode: hybrid
frontier_count: 20919
frontier_adjacent_free_count: 21637
candidates: 64
top_n: 16
best score: 53.62160777611031
best candidate grid: [64, 91, 13]
best candidate world: [0.45, 3.15, 1.35]
best yaw: -3.093456734308408
gain_sc: 0.0
```

Logs:

```text
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_metadata_test.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_depth_to_voxel.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_viz.log
/home/ubuntu22/sc_explorer_ws/logs/stage4a32_medium_scene_expert_step.log
```

Validation:

```text
metadata test: passed
py_compile: passed for new/updated simulator files
RGB images: nonblank
depth images: finite positive values
observed map: contains UNKNOWN/FREE/OCCUPIED
observed_ratio: 0.21339583333333334 > 0.05
obstacle count: 13 >= 10
room count: 3 >= 3
opening count: 3 >= 3
```

Strict boundaries:

```text
No RL, PPO, behavior-cloning training, imitation-learning training,
policy optimization, optimizer step, or model save was run.
No SSCNet training or SSCNet inference on Isaac depth was run.
No PredictionLayer / SSCNet map_predict was connected.
No prediction was written into observed_map.
No NYU target_lr or target_hr was used.
No scene ground truth or simulator ground truth was used for exploration.
Observed_map remains measured-only from Isaac depth.
The optional expert step used EmptyPredictionLayer only and did not run a
multi-step rollout.
```

## 12. Stage 4A-3.5 A* Observed-Free Path Cost

Stage 4A-3.5 is implemented. It adds 2D A* path-cost scoring over a
traversability grid derived only from the current measured `observed_state`.
The actual simulator motion remains planar teleport-to-selected-viewpoint.

New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

A* design:

```text
traversability source: observed_state only
FREE: traversable support in robot-height z band
OCCUPIED: blocked and inflated in x/y
UNKNOWN: not traversable
robot height: 1.2m
clearance height: 0.6m
robot radius: 0.2m
neighbor mode: 8-neighbor A*
path_cost: A* path length / v_max + yaw time
fallback: none; unreachable candidates are invalid
prediction mode: EmptyPredictionLayer only
```

One-step medium A* expert:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_smoke
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_expert_step_astar.log
observed_state: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke/observed_state_final.npy
shape: (120, 120, 30)
unknown/free/occupied: 339813 / 86064 / 6123
traversable/blocked/unknown 2D cells: 4316 / 1907 / 8177
candidates requested: 64
reachable candidates: 12
unreachable candidates: 52
top_n saved: 12
best candidate id: 20
best score: 51.651363679237036
best gain_exp: 110.0
best gain_sc: 0.0
best path cost: 2.129663036258191
best A* path length: 1.2656854249492382m
best A* expanded cells: 36
best grid: [64, 91, 13]
best world: [0.45, 3.15, 1.35]
Euclidean comparison: same best candidate as the prior medium one-step smoke,
but Euclidean path_cost was 2.051412 and top_n had 16 because it did not mark
wall/unknown-disconnected candidates unreachable.
```

One-step visualizations:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_smoke/expert_topdown.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_smoke/expert_score_bar.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_smoke/traversability_topdown.png
```

Medium A* rollout:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred
episode: medium_three_rooms_astar_empty_pred_000
episode dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_medium_rollout_astar_empty_pred.log
scene_variant: medium_three_rooms
path_cost_mode: astar
prediction_mode: empty
steps_completed: 5
done_reason: no_valid_candidate
observed_ratio: 0.0 -> 0.04308796296296296
final unknown/free/occupied: 413386 / 15863 / 2751
final pose: [0.6500000000000004, 0.6500000000000004, 1.2]
final yaw: -0.9638482032376592
average reachable candidates: 18.4
average best path cost: 0.9421159585855353
```

Rollout blocker:

```text
The rollout stopped at expert selection step 5 because all 64 sampled
candidates were unreachable under conservative observed-free traversability:
traversable=338, blocked=918, unknown=13144. No Euclidean fallback was used.
This is expected to need careful handling in the next stage, either by
improving candidate sampling around the reachable observed-free component or by
making traversability less brittle without allowing UNKNOWN as traversable.
```

Rollout visualizations:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000/rollout_topdown_path.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000/observed_ratio_curve.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000/frontier_count_curve.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000/reachable_candidates_curve.png
/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000/step_topdown_000.png ... step_topdown_004.png
```

Tests:

```text
py_compile: passed
test_astar_planner.py: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_astar_planner_test.log
test_sim_expert_astar.py: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_sim_expert_astar_test.log
Euclidean one-step regression: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_paper_expert_euclidean_test.log
Euclidean rollout regression: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a35_regression_sim_expert_rollout_test.log
```

Strict boundaries:

```text
No RL, PPO, behavior-cloning training, imitation-learning training,
optimizer step, policy training, model save, SSCNet retraining, or SSCNet
inference on Isaac depth was run.
No PredictionLayer / SSCNet map_predict was connected.
No prediction was written into observed_map.
No NYU target_lr or target_hr was used.
No scene ground truth or simulator ground truth was used for scoring.
Observed_map remains measured-only from Isaac depth.
A* is used only for expert path-cost scoring.
Motion still teleports; no physical path execution was performed.
No full SC-Explorer RRT tree planner was implemented.
```

## 14. Stage 4A-4 Multi-Episode Empty-Prediction A* Rollout Dataset

Stage 4A-4 generated a deterministic sequential expert rollout dataset using
the Stage 4A-3.6 reachable-frontier A* expert.

New files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_batch.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_rollout_dataset.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_rollout_dataset_batch.py
```

Updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

Batch setup:

```text
output root: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar
scene_variant: medium_three_rooms
scene_seeds: 0, 1, 2
start_variants: start_room_a, start_corridor, start_room_b
intended episodes: 9
actual ok episodes: 9
failed episodes: 0
max_steps: 10
num_candidates: 64
top_n: 16
gain_mode: hybrid
prediction_mode: empty
path_cost_mode: astar
candidate_sampling_mode: reachable_frontier
motion_mode: planar
camera_height: 1.2
voxel_size: 0.1
pixel_stride: 2
```

Dataset result:

```text
total transitions: 90
steps_completed min/mean/max: 10 / 10 / 10
done_reason counts: max_steps=9
observed_ratio_end min/mean/max:
  0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223
total_delta_observed_ratio min/mean/max:
  0.08587037037037037 / 0.11455118312757204 / 0.16534722222222223
no_valid_candidate episodes: 0
no_valid_candidate steps total: 0
average reachable candidates: 64.0
average reachable component count: 570.3444444444444
average best_score: 163.2387554327081
average gain_exp: 49.15555555555556
average gain_sc: 0.0
average path_cost: 0.45623051832594874
```

Outputs:

```text
manifest: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/manifest.jsonl
dataset_summary.json: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/dataset_summary.json
dataset_summary.md: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/dataset_summary.md
HTML index: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/rollout_dataset_index.html
aggregate plots:
  aggregate_observed_ratio_curve.png
  aggregate_observed_ratio_end_bar.png
  aggregate_steps_completed_bar.png
  aggregate_steps_hist.png
  aggregate_done_reasons.png
  aggregate_reachable_candidates_curve.png
  aggregate_no_valid_candidate_stats.png
```

Validation:

```text
batch log: /home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_batch_empty_pred_astar.log
summary log: /home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_summary.log
test log: /home/ubuntu22/sc_explorer_ws/logs/stage4a4_rollout_dataset_test.log
py_compile: passed
test_reachable_candidate_sampling.py: passed
test_astar_planner.py: passed
test_sim_expert_rollout.py regression: passed
test_rollout_dataset_batch.py: passed
observed_ratio non-decreasing: yes
gain_sc with EmptyPredictionLayer: zero
prediction writes observed_map: no
target/ground-truth fields: none
RL/optimizer/BC/IL training: no
UNKNOWN traversability shortcut: no
Euclidean fallback: no
```

Strict boundaries:

```text
No RL, PPO, behavior-cloning training, imitation-learning training,
optimizer step, policy training, model save, SSCNet retraining, or SSCNet
inference on Isaac depth was run.
No PredictionLayer / SSCNet map_predict was connected.
EmptyPredictionLayer remains the only prediction layer.
No prediction was written into observed_map.
No NYU target_lr or target_hr was used.
No scene ground truth or simulator ground truth was used for expert scoring.
Observed_map remains measured-only from Isaac depth.
UNKNOWN remained non-traversable.
A* was used only for expert path-cost scoring.
Motion still teleports; no physical path execution was performed.
No full SC-Explorer RRT tree planner was implemented.
```

## 13. Stage 4A-3.6 Reachability-Aware A* Candidate Sampling

Stage 4A-3.6 is complete. It fixes the Stage 4A-3.5 medium-rollout blocker by
sampling A* candidates from the connected component of observed FREE space
reachable from the current pose. UNKNOWN remains non-traversable, and no
Euclidean fallback is used.

New / updated files:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_reachable_candidate_sampling.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_astar_planner.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_astar.py
```

Implementation:

```text
connected_component_from_start(...)
nearest_traversable_cell(...)
frontier_reachable_candidate_mask(...)
compute_reachable_frontier_candidate_cells(...)
--candidate_sampling_mode frontier|reachable_frontier|auto
--snap_start_to_traversable
--max_snap_radius_cells
```

Sampling behavior:

```text
Euclidean auto mode: old frontier sampling.
A* auto mode: reachable_frontier sampling.
reachable_frontier mode:
  build traversability from observed_state only
  optionally snap current xy to nearest traversable cell
  compute reachable connected component
  sample reachable frontier-adjacent FREE cells first
  fall back only to reachable FREE cells if needed
  exclude the current/snap start cell when other candidates exist
  still run A* per candidate for exact path cost
UNKNOWN traversable: no
Euclidean fallback: no
```

One-step medium reachable A* expert:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_reachable_smoke
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_expert_step_astar_reachable.log
traversable/blocked/unknown 2D cells: 4316 / 1907 / 8177
reachable component count: 1196
reachable frontier-adjacent count: 1196
candidate source: reachable_frontier
snapped current: false
snapped current xy: [60, 80]
reachable/unreachable candidates: 64 / 0
top_n saved: 16
best score: 88.24634362636618
best gain_exp: 66.0
best gain_sc: 0.0
best path_cost: 0.7479063413600806
best A* path length: 0.28284271247461906m
best grid/world: [58, 82, 11] / [-0.15, 2.25, 1.15]
comparison vs Stage 4A-3.5: reachable candidates improved from 12/64 to
64/64, unreachable candidates dropped from 52 to 0, and top_n returned to 16.
```

Medium reachable A* rollout:

```text
output root: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_reachable_empty_pred
episode: medium_three_rooms_astar_reachable_empty_pred_000
episode dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_reachable_empty_pred/episodes/medium_three_rooms_astar_reachable_empty_pred_000
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_medium_rollout_astar_reachable_empty_pred.log
steps_completed: 10
done_reason: max_steps
observed_ratio: 0.0 -> 0.10147453703703704
final unknown/free/occupied: 388163 / 36017 / 7820
final pose: [0.550000011920929, -0.05000000074505806, 1.2000000476837158]
final yaw: 1.6619787310236784
average reachable component count: 238.8
average reachable frontier-adjacent count: 238.8
average reachable candidates: 64.0
candidate source counts: reachable_frontier=10
no_valid_candidate_steps: []
```

Visualizations:

```text
one-step:
  expert_topdown.png
  expert_score_bar.png
  traversability_topdown.png
rollout:
  rollout_topdown_path.png
  observed_ratio_curve.png
  frontier_count_curve.png
  reachable_candidates_curve.png
  reachable_component_count_curve.png
  step_topdown_000.png ... step_topdown_009.png
  rollout_index.html
```

Validation:

```text
py_compile: passed
test_reachable_candidate_sampling.py: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_reachable_candidate_sampling_test.log
test_astar_planner.py: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_astar_planner_regression_test.log
test_sim_expert_astar.py: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_sim_expert_astar_reachable_test.log
Euclidean one-step regression: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_paper_expert_euclidean_test.log
Euclidean rollout regression: passed
  log: /home/ubuntu22/sc_explorer_ws/logs/stage4a36_regression_sim_expert_rollout_test.log
observed_ratio non-decreasing: yes
gain_sc with EmptyPredictionLayer: zero
prediction writes observed_map: no
target/ground-truth fields: none
RL/optimizer/BC/IL training: no
```

Strict boundaries:

```text
No RL, PPO, behavior-cloning training, imitation-learning training,
optimizer step, policy training, model save, SSCNet retraining, or SSCNet
inference on Isaac depth was run.
No PredictionLayer / SSCNet map_predict was connected.
No prediction was written into observed_map.
No NYU target_lr or target_hr was used.
No scene ground truth or simulator ground truth was used for scoring.
Observed_map remains measured-only from Isaac depth.
A* is used only for expert path-cost scoring.
Motion still teleports; no physical path execution was performed.
No full SC-Explorer RRT tree planner was implemented.
```

## 15. Stage 4A-5 Isaac Single-Frame map_predict Alignment Smoke

Stage 4A-5 is complete. It introduces SSCNet `map_predict` only as a
single-frame, read-only simulator prediction layer. The prediction is not
written into `observed_state`, is not used for A* traversability/collision, and
is not used by the expert or a rollout.

Files added:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_prediction_layer.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_isaac_map_predict_single.py
```

Input:

```text
dataset: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar
episode: medium_three_rooms_seed0_start_room_a_empty_astar
step: 0
depth: depth_000.npy
pose: pose_000.json
observed_state: observed_state_step000.npy
checkpoint: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
```

Position convention check:

```text
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_position_convention_check.log
NYU position shape: (480, 640)
NYU position dtype: int32
valid index range: [0, 240*144*240)
invalid/out-of-volume convention: position 0
Project2Dto3D convention: scatter into flat (240,144,240), then permute to
  network axes (z_forward, y_up, x_right)
Isaac convention: smoke-only local volume x_right [-2.4,2.4],
  y_up [-1.44,1.44], z_forward [0,4.8]
```

Run output:

```text
output dir: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single.log
test log: /home/ubuntu22/sc_explorer_ws/logs/stage4a5_isaac_map_predict_single_test.log
depth input shape: (480, 640)
position shape: (480, 640)
valid position pixels: 166888
logits shape: (1, 12, 60, 36, 60)
local prediction shape: (60, 36, 60)
observed_state/global prediction shape: (120, 120, 30)
global valid prediction voxels: 56602
global predicted occupied voxels at tau=0.1: 15664
predicted_unmeasured voxels at tau=0.1: 39400
inference time: 0.1617s
observed_state modified: no
```

Generated artifacts:

```text
sscnet_input_debug.npz
sscnet_depth_input.npy
sscnet_position.npy
valid_position_mask.npy
local_prediction.npz
global_prediction_layer.npz
prediction_alignment_summary.json
isaac_depth_input.png
local_prediction_slices.png
global_prediction_topdown.png
observed_vs_prediction_topdown.png
prediction_not_measured_topdown.png
```

Validation:

```text
py_compile: passed
test_isaac_map_predict_single.py: passed
observed_state hash unchanged: yes
SimPredictionLayer API: ok
prediction layer shape matches observed_state: yes
no target_lr/target_hr/ground_truth fields in prediction artifacts: yes
RL/optimizer/BC/IL training: no
expert/rollout used prediction: no
prediction used for traversability/collision/A*: no
prediction writeback to observed_state: no
```

## 16. Stage 4A-5.1 One-Step SC-Aware Expert Scoring

Stage 4A-5.1 is complete. The read-only Stage 4A-5
`SimPredictionLayer` is now connected to the one-step simulator expert scorer
for paper-style prediction gain only. Prediction does not write into
`observed_state`, does not affect candidate sampling, A* traversability, A*
path validity, collision checking, or ray blocking.

Files updated/added:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_expert_with_prediction.py
```

Input:

```text
episode: medium_three_rooms_seed0_start_room_a_empty_astar
step: 0
observed_state: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/observed_state_step000.npy
pose: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar/pose_000.json
prediction: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke/global_prediction_layer.npz
tau: 0.1
```

Commands/logs:

```text
empty baseline log:
/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_empty_baseline.log

SC prediction log:
/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_sc_prediction.log

test log:
/home/ubuntu22/sc_explorer_ws/logs/stage4a51_expert_with_prediction_test.log
```

Empty baseline:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/empty_baseline
best id: 11
best score: 331.3448560321166
best gain_exp: 55.0
best gain_sc: 0.0
best gain_hybrid: 55.0
best path_cost: 0.1659902032541859
best grid: [13, 13, 11]
best world: [-4.65, -4.65, 1.15]
```

SC prediction expert:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/sc_prediction
best id: 11
best score: 662.6897120642332
best gain_exp: 55.0
best gain_sc: 55.0
best gain_hybrid: 110.0
best gain_occ: 13.0
best gain_conf: 19.406008422374725
best path_cost: 0.1659902032541859
best grid: [13, 13, 11]
best world: [-4.65, -4.65, 1.15]
candidates with gain_sc > 0: 64 / 64
max gain_sc: 174.0
mean gain_sc: 71.59375
total predicted_unmeasured visible count: 4582
```

Comparison:

```text
best candidate changed: false
score delta: 331.3448560321166
gain_hybrid delta: 55.0
top-N overlap: 16 / 16
comparison summary:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.json
/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_smoke/comparison_summary.md
```

Visualizations:

```text
empty baseline:
expert_topdown.png
expert_score_bar.png
traversability_topdown.png

SC prediction:
expert_topdown.png
expert_score_bar.png
traversability_topdown.png
prediction_overlay_topdown.png
predicted_unmeasured_visible_topdown.png

comparison:
empty_vs_prediction_best_candidate.png
gain_comparison_bar.png
```

Validation:

```text
py_compile: passed
test_sim_expert_with_prediction.py: passed
prediction layer shape == observed_state shape: yes
observed_state hash unchanged: yes
empty mode gain_sc == 0: yes
prediction mode gain_sc nonzero: yes
gain_hybrid == gain_exp + gain_sc: yes
gain_occ finite: yes
gain_conf finite: yes
prediction used for traversability: no
prediction used for collision: no
prediction used for A*: no
prediction blocks rays: no
prediction writeback to observed_map: no
target_lr/target_hr/ground_truth leakage: no
RL/optimizer/BC/IL training: no
rollout run: no
```

Limitations:

```text
single-frame prediction only
provisional Isaac-to-SSCNet preprocessing
NYU-to-Isaac domain shift
no prediction fusion
no hierarchical map
no multi-step SC-aware rollout
no full RRT tree planner
```

## 17. Stage 4A-6 Short Multi-Step SC-Aware Rollout

Stage 4A-6 is complete. It extends the Stage 4A-5.1 one-step read-only
`SimPredictionLayer` expert scoring into a short 5-step simulator rollout.
At each step the loop captures Isaac depth, updates the measured-only
`observed_state`, runs dynamic map_predict on the current depth/pose, aligns
the local SSCNet prediction back to the global simulator map shape, and passes
the resulting read-only `SimPredictionLayer` to the expert for information gain
only.

Files added/updated:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/compare_sc_pred_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sim_sc_aware_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_rollout_utils.py
```

Run setup:

```text
scene_variant: medium_three_rooms
scene_seed: 0
start_variant: start_room_a
max_steps: 5
prediction_mode: sim_dynamic
path_cost_mode: astar
candidate_sampling_mode: reachable_frontier
motion_mode: planar
checkpoint: /home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke
episode: medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000
```

Rollout result:

```text
steps_completed: 5
done_reason: max_steps
observed_ratio: 0.000000 -> 0.05899768518518519
final unknown/free/occupied: 406513 / 21226 / 4261
final pose: [-4.25, -4.150000095367432, 1.2000000476837158]
average gain_exp: 49.6
average gain_sc: 49.4
average gain_hybrid: 99.0
average gain_occ: 8.8
average gain_conf: 16.96283725500107
average best_score: 441.9845465468916
candidates_with_gain_sc_positive min/mean/max: 63 / 63.6 / 64
no_valid_candidate_steps: []
```

map_predict performance:

```text
model_loaded_once: true
average preprocess_time: 0.05369079960000818 s
average inference_time: 0.020522295199771178 s
average alignment_time: 0.03251961960013432 s
average preprocess+inference+alignment total: 0.14326694260016665 s
average expert_time: 1.026360238399866 s
total_wall_time: 19.86559214800036 s
gpu_memory_peak: 794354176 bytes
hardware: AMD Ryzen 9 9950X3D, 32 CPU threads, 32GB RAM, NVIDIA RTX 5080
```

Comparison against Stage 4A-4 matching empty baseline:

```text
baseline episode: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/medium_three_rooms_seed0_start_room_a_empty_astar
compared_steps: 5
empty final observed_ratio: 0.06896296296296296
SC final observed_ratio: 0.05899768518518519
SC-empty observed_ratio delta: -0.009965277777777774
changed selected actions: 5
mean score delta SC-empty: 233.79287700349096
mean gain_exp delta SC-empty: -5.199999999999996
mean SC gain_sc: 49.4
mean path_cost delta SC-empty: -0.09316736073162746
```

Interpretation: the SC-aware rollout underperformed the matching measured-only
baseline on observed_ratio at the 5-step horizon. This is not treated as a
performance win; Stage 4A-6 success is integration correctness, dynamic
prediction plumbing, and safety/leakage preservation.

Generated outputs:

```text
episode_summary.json
transitions.jsonl
step_000.npz ... step_004.npz
observed_state_step000.npy ... observed_state_step004.npy
observed_state_final.npy
depth_000.npy ... depth_004.npy
rgb_000.png ... rgb_004.png
prediction_step000/ ... prediction_step004/
rollout_topdown_path.png
observed_ratio_curve.png
frontier_count_curve.png
gain_exp_gain_sc_curve.png
best_score_curve.png
map_predict_timing_curve.png
prediction_valid_count_curve.png
predicted_unmeasured_count_curve.png
step_topdown_000.png ... step_topdown_004.png
rollout_index.html
comparison_to_empty_baseline/
```

Validation:

```text
py_compile: passed
test_sim_sc_aware_rollout.py: passed
required episode files present: yes
required comparison files present: yes
required output files zero-size: no
observed_ratio non-decreasing: yes
gain_sc nonzero: yes
observed_state hash unchanged by prediction: yes
prediction used for traversability: no
prediction used for collision: no
prediction used for A*: no
prediction blocks rays: no
prediction writeback: no
checkpoint modified: no
RL/optimizer/BC/IL/SSCNet training: no
log scan Traceback/Error/CUDA unavailable/current blocker: no
```

Per-step selected action metrics:

```text
step 0: ratio 0.000000000 -> 0.042416667, gain_exp/sc/hybrid 53/53/106, score 639.954941, path_cost 0.165637, pred_valid 56602, pred_unmeasured 39375
step 1: ratio 0.042416667 -> 0.051043981, gain_exp/sc/hybrid 53/52/105, score 212.544531, path_cost 0.494014, pred_valid 57382, pred_unmeasured 37537
step 2: ratio 0.051043981 -> 0.055990741, gain_exp/sc/hybrid 48/48/96, score 329.455103, path_cost 0.291390, pred_valid 57356, pred_unmeasured 33943
step 3: ratio 0.055990741 -> 0.058018519, gain_exp/sc/hybrid 53/53/106, score 746.162117, path_cost 0.142060, pred_valid 57616, pred_unmeasured 33149
step 4: ratio 0.058018519 -> 0.058997685, gain_exp/sc/hybrid 41/41/82, score 281.806041, path_cost 0.290980, pred_valid 56394, pred_unmeasured 31587
```

Fixed issue:

```text
initial failure: prediction visualization raised KeyError('observed_state_source') at step 0
cause: prediction_alignment_summary.json did not include observed_state_source
fix: isaac_map_predictor.py now writes observed_state_source, depth_source, pose_source, and camera_info_source
final status: fixed; rerun completed 5 steps and validation passed
```

Boundary decisions:

```text
Stage 4A-6 uses dynamic per-step map_predict for a short SC-aware rollout.
Prediction is information-gain-only and read-only.
Prediction does not affect observed_map, traversability, collision, A*, candidate reachability, or ray blocking.
SSCNet checkpoint is loaded once per rollout and reused at each step.
No target_lr/target_hr, scene ground truth, simulator ground truth, RL, PPO, BC, IL, or SSCNet training is used.
```

Limitations:

```text
short single episode only
provisional Isaac-to-SSCNet preprocessing
NYU-to-Isaac domain shift
no prediction fusion yet
no hierarchical map yet
teleport motion
A* scoring only, no physical path execution
no full RRT tree planner
no RL/IL training
```

Next recommendation:

```text
Stage 4A-6.1 should be analysis/ablation/tuning, not RL:
static prediction ablation, gain_sc weighting, tau sweep, possible prediction-gain cap,
overlay/action inspection, and only then a 10-step SC-aware rollout compared
against the measured-only baseline.
```

## 18. Stage 4A-6.1 SC-Aware Rollout Analysis and Ablation

Stage 4A-6.1 is complete. It analyzed why the 5-step dynamic SC-aware rollout
underperformed the matching measured-only empty baseline, then ran five small
5-step scoring ablations without changing the prediction safety boundary.

Code added / updated:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_sc_rollout_behavior.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_pred_ablation_sweep.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_pred_ablation.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_pred_ablation.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/sim_paper_expert.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_step.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_sim_rollout.py
```

New scorer fields:

```text
sc_gain_weight
sc_gain_cap
score_gain_mode: hybrid_raw | hybrid_weighted
weighted_gain_sc
gain_hybrid_weighted
utility_hybrid_weighted
```

Default behavior remains compatible with Stage 4A-6:

```text
score_gain_mode=hybrid_raw
sc_gain_weight=1.0
sc_gain_cap=None
final_score remains the raw gain_hybrid/path_cost score by default
```

Existing SC vs empty analysis:

```text
analysis output:
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/existing_sc_vs_empty

empty final observed_ratio at 5 steps: 0.06896296296296296
original SC final observed_ratio: 0.05899768518518519
SC-empty delta: -0.009965277777777774
changed selected actions: 5 / 5
first step where SC lags empty: step 1
mean path_cost empty / SC: 0.36998367643136965 / 0.2768163156997422
mean gain_exp empty / SC: 54.8 / 49.6
mean SC gain_sc: 49.4
mean best_score empty / SC: 208.19166954340062 / 441.9845465468916
dense gain_sc candidate steps: 0, 1, 2, 3, 4
SC selected nearest revisit distance mean/min: 0.1960404596788693 / 0.09999990463256836 m
```

Ablation configs completed:

```text
dynamic_w025_tau01: final observed_ratio 0.05899768518518519
dynamic_w05_tau01: final observed_ratio 0.05899768518518519
dynamic_w1_tau03: final observed_ratio 0.05899768518518519
dynamic_w1_tau01_cap50: final observed_ratio 0.05899768518518519
static_step0_weight_1p0_tau_0p1: final observed_ratio 0.05899768518518519
```

All five ablations selected the same 5/5 actions as the original SC rollout.
Weighting, a tau increase to 0.3, capping gain_sc at 50, and reusing the static
step0 prediction changed scores/timing but did not change behavior or measured
coverage in this scene/start.

Performance and resource summary:

```text
dynamic_w025_tau01: wall_time 30.416639261 s, avg inference 0.03013971940004012 s, GPU peak 794296320 bytes
dynamic_w05_tau01: wall_time 27.75568118299998 s, avg inference 0.021613561999947704 s, GPU peak 794296320 bytes
dynamic_w1_tau03: wall_time 29.056840359000034 s, avg inference 0.02078229319995444 s, GPU peak 794296320 bytes
dynamic_w1_tau01_cap50: wall_time 28.657331081000166 s, avg inference 0.02070385000006354 s, GPU peak 794296320 bytes
static_step0_weight_1p0_tau_0p1: wall_time 22.448690754999916 s, avg inference 0.0 s, GPU peak None
```

Summary output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_ablation/summary
/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection
```

Validation:

```text
py_compile: passed
test_sc_pred_ablation.py: passed
completed ablation configs: 5
observed_ratio non-decreasing: yes
weighted gain formula: yes
prediction writeback: no
prediction used for traversability: no
prediction used for collision: no
prediction used for A*: no
prediction blocks rays: no
UNKNOWN traversable: no
Euclidean fallback: no
target/ground-truth leakage: no
RL/PPO/BC/IL/optimizer/SSCNet training: no
checkpoint modified: no
log scan blockers: none found
```

Interpretation:

```text
This is not a plumbing failure. SC-aware scoring is active and safe, but the
prediction gain is dense and not sufficiently discriminative under the current
Isaac-to-SSCNet preprocessing/alignment. In this seed/start, the SC scorer is
pulled toward nearby low-path-cost local viewpoints that score highly but add
less measured coverage than the empty baseline. The immediate next step should
inspect preprocessing, alignment, confidence calibration, and NYU-to-Isaac
domain shift before scaling rollout length.
```

## 19. Stage 4A-6.2 map_predict Preprocessing / Alignment / Calibration Diagnostics

Stage 4A-6.2 completed as an offline prediction diagnostic stage. It did not
start Isaac, did not rerun long rollout, did not run RL/IL/training, did not
modify the SSCNet checkpoint, and did not write predictions into
`observed_state`.

Created scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/diagnose_prediction_global_alignment.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/evaluate_prediction_against_future_observed.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_map_predict_alignment_variant_sweep.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_map_predict_diagnostics.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_map_predict_diagnostics.py
```

Diagnostics root:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics
```

Key outputs:

```text
preprocess: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/preprocess_stats
alignment: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/global_alignment
future eval: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/future_observed_eval
variant sweep: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/alignment_variant_sweep
candidate decomposition: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/candidate_score_decomposition
summary: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a62_diagnostics/summary
```

Preprocessing comparison:

```text
Isaac mean depth: 2.532350206375122 m
NYU mean depth: 2.8481276988983155 m
Isaac depth p99: 4.373712015151978 m
NYU depth p99: 5.255490036964427 m
Isaac valid position ratio: 0.565763671875
NYU position nonzero proxy ratio: 0.74495458984375
suspicious difference: Isaac valid position ratio differs strongly from NYU
```

Global alignment sanity:

```text
mean valid in-front ratio: 0.9977230302200312
mean inside global bounds ratio: 0.8669629629629629
mean valid inside expected local-volume ratio: 0.9922601247618591
mean local-to-global duplicate rate: 0.492071105189736
local voxels below floor before clipping: 10800.0
likely default axis/yaw issue from direct sanity: false
z/bounds issue: true, due local volume extending below floor from camera height
```

Future observed evaluation used future measured maps only as post-hoc delayed
sensor validation, not planning input or expert scoring:

```text
tau=0.1 mean predicted_unmeasured: 35118.2
tau=0.1 mean later measured fraction: 0.059004217437215026
tau=0.1 occupied precision: 0.25632042463242544
tau=0.1 free precision: 0.9323112341592195
tau=0.1 occupied Brier: 0.2786559495144023
ECE-like occupied calibration: 0.3405436085907938
tau=0.1 too dense: true
tau sweep did not reduce density meaningfully: true
```

Alignment variant sweep:

```text
variants tested: current_default, yaw_sign_flipped, x_right_sign_flipped,
z_forward_sign_flipped, x_right_and_yaw_flipped, xz_swap_variant,
z_up_sign_variant, local_origin_shift_forward_half_voxel,
local_origin_shift_backward_half_voxel
best diagnostic variant: xz_swap_variant
default variant rank: 7
Brier improvement vs default: 0.0735458940774611
likely alignment bug: true
```

Candidate-score decomposition:

```text
gain_exp/gain_sc correlation: 0.9647202023737985
final_score vs inverse path_cost correlation: 0.9713818732156227
final_score vs gain_hybrid correlation: -0.5071519566890417
gain_sc duplicates gain_exp: true
path_cost dominance: true
all five Stage 4A-6.1 ablations matched original SC actions: 5/5
```

Final diagnostic recommendation:

```text
Primary suspected issue: alignment convention
Recommendation: fix alignment convention and rerun Stage 4A-5/5.1/6 smoke
Secondary issue: confidence calibration is too dense/unselective at tau=0.1
Do not jump to RL/IL or scale SC-aware rollouts yet.
```

Validation:

```text
py_compile: passed
test_map_predict_diagnostics.py: passed
observed_state modified: no
prediction writeback: no
prediction used for traversability/collision/A*/candidate reachability/ray blocking: no
future observations: post-hoc evaluation only
checkpoint modified: no
diagnostics ran without Isaac startup: yes
log scan blockers: none found
```

## 20. Stage 4A-6.3 SSCNet Alignment Convention Fix

Stage 4A-6.3 reconciled SSCNet local prediction axes with the Isaac global
map axes. The stage did not run RL/PPO/BC/IL, did not train SSCNet, did not
modify the checkpoint, did not use future observations for planning, and did
not write prediction into `observed_state`.

Created scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/document_sscnet_axis_convention.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/fix_prediction_alignment_convention.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_alignment_convention_fix.py
```

Modified scripts:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_sscnet_preprocess.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/isaac_map_predictor.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_isaac_map_predict_single.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sim_expert_rollout_sc_pred.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/visualize_isaac_prediction_alignment.py
```

Axis audit result:

```text
Project2Dto3D: scatter flat -> view(W,H,D) -> permute(D,H,W)
raw Python dataloader branch: np.ravel_multi_index((x,y,z),(240,144,240))
C++/ROS projection path: z*(240*144)+y*240+x
dataloader target_lr.T: reverses stored axes before loss flattening
Stage 4A-5 assumption: followed the raw Python branch and interpreted output
  as (z_forward,y_up,x_right)
code_consistent_v1: follows C++/ROS/repackaged projection convention, with
  input position flatten (z_forward,y_up,x_right) and output axes
  (x_right,y_up,z_forward)
```

Convention evaluation output:

```text
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/axis_convention_audit
/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_stage4a63_alignment_fix/convention_eval
```

At tau `0.1`, using future measured maps only as post-hoc delayed sensor
validation:

```text
current_default_v0 occupied Brier: 0.2786559495144023
code_consistent_v1 occupied Brier: 0.20511005543694122
Brier improvement: 0.0735458940774611
current_default_v0 ECE-like: 0.3405436085907937
code_consistent_v1 ECE-like: 0.22427722861569463
later measured fraction default/fixed: 0.059004217437215026 / 0.059004217437215026
best diagnostic convention: xz_swap_diagnostic
recommended fixed convention: code_consistent_v1
```

Fixed single-frame map_predict smoke:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke_alignment_fixed
alignment_convention: code_consistent_v1
global_valid_prediction_count: 56602
global_predicted_occupied_count: 16792
predicted_unmeasured_count: 39400
local confidence mean: 0.715299665927887
local occupied_prob mean: 0.36329972743988037
observed_state modified: no
```

Fixed one-step expert smoke:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_sc_pred_alignment_fixed_smoke
best candidate id: 11
best gain_exp/gain_sc/gain_hybrid: 55.0 / 55.0 / 110.0
best score: 662.6897120642332
candidates remain dense with SC gain
observed_state modified: no
prediction affects information gain only
```

Fixed 5-step rollout smoke:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_alignment_fixed_smoke
steps_completed: 5
done_reason: max_steps
observed_ratio: 0.0 -> 0.05899768518518519
empty baseline final observed_ratio at 5 steps: 0.06896296296296296
original SC final observed_ratio at 5 steps: 0.05899768518518519
changed actions vs empty: 5
changed actions vs original SC: 0
mean gain_sc: 49.4
candidates_with_gain_sc_positive mean: 63.6 / 64
```

Interpretation:

```text
The alignment convention issue is real and code-explainable: the diagnostic
x/z swap is the output-side symptom of the C++/ROS projection index convention.
`code_consistent_v1` should be used for future Isaac map_predict smoke runs.
However, the fixed 5-step rollout still matches the original SC actions and
still underperforms the empty baseline, so the remaining blocker is the dense
and poorly selective prediction gain/calibration rather than axis alignment.
```

Validation:

```text
py_compile: passed
test_alignment_convention_fix.py: passed
observed_state modified: no
prediction writeback: no
prediction used for traversability/collision/A*/candidate reachability/ray blocking: no
future observations: post-hoc evaluation only
target/ground-truth leakage: no
RL/PPO/BC/IL/optimizer/SSCNet training: no
checkpoint modified: no
log scan blockers: none found, aside from explicit false safety lines in the
  validation log
```

## 21. Stage 4A-6.4 Calibrated / Confidence-Gated I_sc

Stage 4A-6.4 added selective simulator prediction gain while keeping
prediction read-only and information-gain-only. Prediction still does not
write `observed_state`, does not participate in A*, traversability, collision,
candidate reachability, or ray blocking, and no RL/PPO/BC/IL/optimizer or
SSCNet training was run.

Implemented / validated:

```text
/home/ubuntu22/sc_explorer_ws/sim_explorer/calibrate_prediction_gain.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/run_sc_gain_gating_ablation.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/summarize_sc_gain_gating.py
/home/ubuntu22/sc_explorer_ws/sim_explorer/test_sc_gain_gating.py
```

Calibration:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/calibration
sample_count: 11175
occupied_prob weighted bin correlation: 0.8699543518514645
confidence weighted bin correlation: 0.893222674245022
recommended occupied/confidence thresholds: 0.9 / 0.9
calibrated_occupied usable: true
future observations: post-hoc reliability estimation only
```

5-step gated ablation:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/ablation
completed configs: occupied_only_occ07, occupied_only_occ08,
  occupied_margin_occ06_w05, confidence_weighted_conf05_cap30
failed configs: []
empty baseline observed_ratio at 5 steps: 0.06896296296296296
fixed raw SC observed_ratio at 5 steps: 0.05899768518518519
all gated configs observed_ratio at 5 steps: 0.05899768518518519
changed actions vs fixed raw SC: 0/5
```

Selectivity:

```text
mean raw gain_sc: 49.4
mean effective_gain_sc occupied_only_occ07: 4.2
mean effective_gain_sc occupied_only_occ08: 3.2
mean effective_gain_sc occupied_margin_occ06_w05: 1.7860426306724548
mean effective_gain_sc confidence_weighted_conf05_cap30: 36.19095666408539
```

Summary:

```text
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a64_gain_gating/summary
recommendation: No completed gating config changed raw SC behavior enough;
prediction gain remains insufficiently selective for this scene.
```

Validation:

```text
py_compile: passed
test_sc_gain_gating.py: passed
observed_ratio non-decreasing: yes
raw/effective/weighted gain logged: yes
prediction read-only: yes
prediction writeback/traversability/collision/A*/ray blocking: no
future observations used for planning/scoring: no
target/ground-truth leakage: no
RL/PPO/BC/IL/optimizer/SSCNet training: no
checkpoint modified: no
```

Conclusion:

```text
Selective SC gain works numerically, but candidate ranking did not change in
the current 5-step medium_three_rooms seed/start. The next simulator-side
debug step should examine candidate-level rank decomposition and spatial
placement of selected/rejected candidates under the gated formulas before any
rollout scaling or policy training.
```

## 22. Stage 4A-6.5a Candidate Rank Sensitivity Diagnosis

Stage 4A-6.5a performed an offline-only candidate rank decomposition over
existing steps `0..4`; it did not launch Isaac or run new rollouts.

```text
script: /home/ubuntu22/sc_explorer_ws/sim_explorer/analyze_candidate_rank_sensitivity_small.py
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65a_rank_sensitivity
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a65a_rank_sensitivity.log
configs: empty_baseline, fixed_raw_sc, occupied_only_occ07,
  occupied_only_occ08, occupied_margin_occ06_w05,
  confidence_weighted_conf05_cap30
steps: 0..4
```

Main result:

```text
gated selected candidate ids identical: yes
gated selected positions identical: yes
top-1 stable vs fixed raw SC: yes
mean top-5 Jaccard vs raw SC: 0.9166666666666666
mean top-16 Jaccard vs raw SC: 0.869934640522876
final_score vs inverse path_cost Pearson: 0.8919154707376216
final_score vs gain_exp Pearson: -0.46732096565152237
final_score vs effective_gain_sc Pearson: 0.03806071813182923
selected low-path-cost rank mean: 1.0333333333333334
selected gain_exp rank mean: 14.4
diagnosis: path_cost dominates top-1; gating changes lower ranks only.
recommended next small task: offline counterfactual score analysis
```

## 23. Stage 4A-6.5b Offline Counterfactual Score Analysis

Stage 4A-6.5b used only the existing Stage 4A-6.5a candidate rank table and
summary files. It did not launch Isaac, run a rollout, rerun map_predict, or
modify expert runtime scoring.

```text
script: /home/ubuntu22/sc_explorer_ws/sim_explorer/offline_score_counterfactuals_small.py
output: /home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65b_counterfactual_scores
log: /home/ubuntu22/sc_explorer_ws/logs/stage4a65b_counterfactual_scores.log
formula variants executed: 94
candidate rows: 480
```

Main result:

```text
exp_only_no_cost changed top-1 groups: 30/30
over-cost formulas changed top-1 groups: 0
alpha=0.5 changed vs alpha=1 grouped sweeps: 80
sc_only changed executable groups: 10/20
SC-specific lambda threshold: min 0.1, median 0.5
plausible later one-step smoke candidate: decoupled_sc_lambda0p5
counterfactual formulas are offline-only; no observed_ratio improvement is
claimed.
```
