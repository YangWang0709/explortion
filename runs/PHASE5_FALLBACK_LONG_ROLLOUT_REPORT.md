# Phase 5 Fallback Long Rollout Report

- phase: Phase 5 bounded fallback long-rollout metadata packet
- generated_at: 2026-06-07T15:30:59.219580+08:00
- PI_WORKSPACE: /home/ubuntu22/pi
- rollout_exit_code: 0
- fallback_scene: scenes/minimal_indoor_smoke.usda
- primary_usd_used: false
- primary_usd_status: load crash, preserved for diagnosis
- start_count: 10
- max_actions: 16
- total_step_rows: 149
- total_action_count: 143
- candidate_rows: 3576
- starts_with_failures: 6
- average_final_known_ratio: 0.8947

## Artifacts

- rollout_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout
- rollout_summary: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/rollout_summary.json
- rollout_steps_csv: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/rollout_steps.csv
- rollout_steps_jsonl: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/rollout_steps.jsonl
- candidate_summary_csv: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/candidate_summary.csv
- failure_summary_csv: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout/failure_summary.csv

## Scope

This packet is a bounded fallback-scene rollout metadata packet using the deterministic BEV selector. It is for pipeline/review scaffolding only until the primary USD crash is repaired. It does not train, does not checkpoint, and does not run RL.

## Human Review Note

This packet can be inspected for action/coverage schema sanity, but it should not be promoted as final Isaac primary-scene data because /home/ubuntu22/pi/building_scene.usd still crashes on Isaac/Omniverse load.

## Per-Start Directories

The packet also includes per-start CSV directories start_000 through start_009 under:

- /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase5_fallback_long_rollout

Each start directory contains trajectory.csv and candidates.csv.
