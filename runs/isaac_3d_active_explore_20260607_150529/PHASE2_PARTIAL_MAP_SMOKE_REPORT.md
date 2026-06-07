# Phase 2 Partial Map Smoke Report

- phase: Phase 2 lightweight partial-map smoke
- generated_at: 2026-06-07T15:26:54.847903+08:00
- PI_WORKSPACE: /home/ubuntu22/pi
- smoke_exit_code: 0
- resolution: 0.25
- grid_width: 49
- grid_height: 33
- total_cells: 1617
- path_pose_count: 5
- sensor_radius: 2.8
- counts: {"frontier": 82, "known_free": 780, "occupied": 75, "unknown": 762}
- known_ratio: 0.5288

## Artifacts

- grid_csv: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase2_mapping_smoke/partial_map_grid.csv
- ascii_map: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase2_mapping_smoke/partial_map_ascii.txt
- summary_json: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase2_mapping_smoke/partial_map_summary.json

## Scope

This smoke establishes the BEV/occupancy data contract for known_free, occupied, unknown, observed_count, and frontier. It uses deterministic fallback-scene geometry and the robot marker path. It does not run rollout, training, checkpointing, or RL.

## Next Step

Phase 3 can add candidate viewpoint sampling and information-gain scoring on top of this partial-map representation.
