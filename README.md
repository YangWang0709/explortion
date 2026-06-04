# SC-Explorer IsaacSim Workspace

This repository is the lightweight project workspace for SC-Explorer / SSCNet / IsaacSim active exploration reproduction and extension work.

The active scene context is user-provided through `building_scene.usd`, but large USD files, Isaac dependency packages, visual outputs, datasets, checkpoints, logs, and generated runtime artifacts are intentionally not tracked in ordinary Git.

## Current Project State

- Workspace path: `/home/ubuntu22/sc_explorer_ws`
- Main conda environment: `env_isaaclab`
- Active fixed environment USD: `/home/ubuntu22/sc_explorer_ws/assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd`
- Corrected camera-pose validation output: `/home/ubuntu22/sc_explorer_ws/outputs/isaac_stage4a66c_usd_camera_pose_fix`
- Current next step: Stage 4A-6.6d USD scene audit + human visual review
- Do not jump directly to Stage 4A-6.7 expert sampling.

## Version-Control Policy

Git tracks source code, lightweight configs, project context notes, and reproducibility documentation.

Git does not track:

- `outputs/`
- `logs/`
- `checkpoints/`
- `data/`
- large USD/source assets
- Isaac dependency packages
- PNG/MP4/NPY/NPZ visual or runtime products
- model checkpoints and binary weights

If future work needs large assets in a remote repository, configure Git LFS or external artifact storage explicitly before adding them.
