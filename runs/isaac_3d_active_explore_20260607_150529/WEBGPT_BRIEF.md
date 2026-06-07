# WebGPT Brief

## Current Phase

Phase 0: environment and project audit, blocked on primary USD load.

## Completed

- Confirmed PI_WORKSPACE: /home/ubuntu22/pi
- Audited git, Ubuntu system, conda env_isaaclab, GPU, ROS2 availability, USD assets, source files, Isaac imports, and Isaac headless startup.
- Confirmed Isaac headless starts.
- Confirmed primary USD exists but crashes during Isaac/Omniverse USD context loading.

## Key Results

- Primary USD: /home/ubuntu22/pi/building_scene.usd
- Primary USD sha256: 11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b
- Primary USD file type: USD crate, version 0.8.0
- IsaacSim Python module found: true
- Isaac headless started: true
- Primary USD opened through Isaac/Omniverse USD context: false, core dump exit code 134
- Existing source file count: 0
- Files over 50MB: 0

## Workspace

- PI_WORKSPACE: /home/ubuntu22/pi
- conda_env: env_isaaclab
- env_activation_command: source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab

## Key Paths

- run_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- logs: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs
- reports: /home/ubuntu22/pi/ENVIRONMENT_AUDIT.md, /home/ubuntu22/pi/runs/CRITIC_REPORT.md, /home/ubuntu22/pi/runs/FAILURE_DIAGNOSIS.md
- data: no rollout data collected in Phase 0

## Metrics

- start_count: 0
- actions_per_start: 0
- coverage: not measured yet
- failure_count: 1 Phase 0 scene-load blocker
- stuck_count: 0
- candidate_count: 0
- average_information_gain: not measured yet

## Current Issues

- The primary USDC scene crashes Isaac/Omniverse when opened.
- PI_WORKSPACE has no robot/sensor/mapping/rollout source code yet.

## Critic Review Conclusion

- continue: only with minimal scene-load repair or fallback.
- deviated_from_goal: no
- human_review_needed: no rollout review yet; human review comes after long rollout packet exists.

## Questions For Web ChatGPT

1. Is it acceptable to preserve /home/ubuntu22/pi/building_scene.usd and add a separate minimal indoor USDA smoke scene to unblock Phase 1?
2. Should the original USDC be treated as corrupted or incompatible until converted or repaired?
