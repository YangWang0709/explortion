# Phase 1 Scene Load Smoke Report

- phase: Phase 1 pre-smoke scene-load validation
- generated_at: 2026-06-07T15:14:02.852581+08:00
- PI_WORKSPACE: /home/ubuntu22/pi
- fallback_scene: /home/ubuntu22/pi/scenes/minimal_indoor_smoke.usda
- original_scene_preserved: true
- original_scene: /home/ubuntu22/pi/building_scene.usd
- smoke_exit_code: 0
- simulation_app_started: None
- fallback_stage_available: None
- root_prims: None
- prim_count: None
- camera_count: None
- cube_count: None
- error: summary missing

## Scope

This smoke only validates that Isaac headless can open a stable fallback indoor USD scene. It does not run sensors, robot control, rollout, training, checkpointing, or RL.

## Next Step

If this smoke passes, Phase 1 can continue by adding RGB-D / LiDAR / pose smoke instrumentation against the fallback scene while the original USDC crash is diagnosed separately.
