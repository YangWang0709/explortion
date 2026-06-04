# Artifact Inventory

This file records important local artifacts that are intentionally not tracked by ordinary Git.

## Required Local Paths

- Source scene USD: `/home/ubuntu22/sc_explorer_ws/building_scene.usd`
- Isaac dependency package: `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/dependencies`
- Fixed localized USD: `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd`
- Corrected visual package: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_camera_pose_fix`

## Not Tracked By Git

- Checkpoints: `/home/ubuntu22/sc_explorer_ws/checkpoints`
- Outputs: `/home/ubuntu22/sc_explorer_ws/outputs`
- Logs: `/home/ubuntu22/sc_explorer_ws/logs`
- Datasets: `/home/ubuntu22/sc_explorer_ws/data`
- Isaac dependencies: `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment/dependencies`
- Localized Isaac asset packages: `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized`
- Fixed localized Isaac asset package: `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized_defaultprim`
- Root source USD: `/home/ubuntu22/sc_explorer_ws/building_scene.usd`

## Large Files Found During Initialization

The initial scan found 23 files larger than 50 MB. All of them are intentionally excluded from ordinary Git:

- `data/real_nyu_npz/NYUtest_npz.zip`
- `data/real_nyu_npz/NYUtrain_npz.zip`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_final.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step000.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step001.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step002.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step003.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step004.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step005.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step006.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step007.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step008.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step009.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step010.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step011.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step012.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step013.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step014.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step015.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step016.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step017.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step018.npy`
- `outputs/isaac_stage4a66c_usd_defaultprim_fix/observed_state_step019.npy`

## Restore Guidance

Place or restore local artifacts at the exact paths above before running scene validation or later expert-sampling work. If these artifacts need to be shared remotely, use Git LFS or external artifact storage instead of adding them directly to this repository.
