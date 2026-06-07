# Agent Log

## Phase 0

- Main Coordinator resolved PI_WORKSPACE=/home/ubuntu22/pi.
- SSH Experiment audited git, system, conda env, GPU, ROS2, USD, source files, Isaac imports, and headless startup.
- IsaacSim Scene and Sensor found a blocker: primary USD crashes during Isaac/Omniverse USD context open.
- GitHub Sync initialized the PI_WORKSPACE repository and prepared reports for YangWang0709/sss.
- Critic confirmed no training, RL, checkpoint, or rollout occurred.

## Phase 1 scene-load pre-smoke

- Created scenes/minimal_indoor_smoke.usda as a fallback scene without modifying building_scene.usd.
- Added scripts/phase1_scene_load_probe.py.
- Fallback scene-load smoke status: passed.
- Smoke log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase1_scene_smoke/minimal_scene_load.log

## Phase 1 camera sensor smoke

- Added scripts/phase1_camera_sensor_smoke.py.
- Ran RGB-D camera smoke against scenes/minimal_indoor_smoke.usda.
- Camera sensor smoke status: passed.
- Smoke log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase1_sensor_smoke/camera_sensor_smoke.log

## Phase 1 robot pose smoke

- Added scripts/phase1_robot_pose_smoke.py.
- Validated robot marker pose read/write against scenes/minimal_indoor_smoke.usda.
- Robot pose smoke status: passed after synchronous retry.
- Smoke log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase1_robot_pose_smoke/robot_pose_smoke.log

## Phase 1 robot pose smoke retry

- Replaced robot pose smoke with synchronous omni.usd/app.update implementation.
- Robot pose smoke retry status: passed.
- Retry log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase1_robot_pose_smoke/robot_pose_smoke_retry.log

## Phase 1 point-cloud smoke

- Added scripts/phase1_pointcloud_smoke.py.
- Ran point-cloud metadata smoke against scenes/minimal_indoor_smoke.usda.
- Point-cloud smoke status: passed.
- Smoke log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase1_pointcloud_smoke/pointcloud_smoke.log

## Phase 2 partial map smoke

- Added scripts/phase2_partial_map_smoke.py.
- Generated BEV occupancy map summary from fallback scene geometry and robot marker path.
- Partial map smoke status: failed.
- Smoke artifacts: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase2_mapping_smoke/

## Phase 2 partial map smoke retry

- Fixed CSV first_seen_pose_id field handling in scripts/phase2_partial_map_smoke.py.
- Partial map smoke retry status: passed.
- Smoke artifacts: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase2_mapping_smoke/

## Phase 3 candidate gain smoke

- Added scripts/phase3_candidate_gain_smoke.py.
- Sampled candidate viewpoints around the current robot marker pose.
- Candidate gain smoke status: passed.
- Smoke artifacts: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase3_candidate_smoke/

## Phase 4 closed-loop smoke

- Added scripts/phase4_closed_loop_smoke.py.
- Ran bounded selector loop on fallback BEV map.
- Closed-loop smoke status: passed.
- Smoke artifacts: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase4_closed_loop_smoke/

## Phase 5 fallback long rollout

- Added scripts/phase5_fallback_long_rollout.py.
- Generated 10-start bounded fallback rollout metadata packet.
- Fallback long rollout status: passed.
- Rollout artifacts: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/
