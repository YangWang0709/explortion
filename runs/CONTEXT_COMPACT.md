# Context Compact

- Current goal: Isaac Sim 3D active exploration long rollout data collection in PI_WORKSPACE.
- PI_WORKSPACE: /home/ubuntu22/pi
- Conda env: env_isaaclab
- Activation command: source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
- Current phase: Phase 0 audit complete, blocked on primary USD load crash.
- Primary USD: /home/ubuntu22/pi/building_scene.usd
- Primary USD sha256: 11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b
- Key blocker: Isaac headless starts, but omni.usd open_stage of the primary USD aborts with exit code 134.
- Key files: ENVIRONMENT_AUDIT.md, runs/ACTIVE_TASK_BOARD.md, runs/WEBGPT_BRIEF.md, runs/CRITIC_REPORT.md, runs/FAILURE_DIAGNOSIS.md, runs/LONG_ROLLOUT_STATUS.md.
- Latest run dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- Do not train PI/openpi/VLM, do not run RL, do not create checkpoints before human-approved rollout data exists.
