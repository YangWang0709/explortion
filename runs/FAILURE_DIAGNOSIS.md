# Failure Diagnosis

- phase: Phase 0
- blocking_failure: true
- failure_type: USD scene load crash
- primary_usd: /home/ubuntu22/pi/building_scene.usd
- primary_usd_sha256: 11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b
- primary_usd_file_type: USD crate, version 0.8.0
- failing_command_log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs/phase0_isaac_usd_context_probe.log
- exit_code: 134

## What Happened

Isaac headless SimulationApp starts successfully, but opening /home/ubuntu22/pi/building_scene.usd through omni.usd.get_context().open_stage aborts the Python process with a core dump. The crash stack is inside USD crate and layer loading.

## Minimal Repair Plan

1. Do not delete or overwrite /home/ubuntu22/pi/building_scene.usd.
2. Add a separate minimal indoor USDA smoke scene under PI_WORKSPACE for controlled Phase 1 validation.
3. If the fallback scene loads, use it to build the robot and sensor smoke pipeline while separately diagnosing whether the original USDC can be converted or repaired.
4. Do not start training, RL, checkpointing, or long rollout until the scene and sensor smoke path is reliable.
