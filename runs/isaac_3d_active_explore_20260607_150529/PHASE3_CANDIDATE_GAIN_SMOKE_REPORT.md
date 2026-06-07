# Phase 3 Candidate Gain Smoke Report

- phase: Phase 3 candidate viewpoint and information-gain smoke
- generated_at: 2026-06-07T15:28:05.957309+08:00
- PI_WORKSPACE: /home/ubuntu22/pi
- smoke_exit_code: 0
- start_pose: {'x': 3.0, 'y': -0.2}
- candidate_count: 24
- valid_count: 11
- invalid_count: 13
- alpha: 2.0
- sensor_radius: 2.8
- candidate_radius: 2.0
- best_candidate: {"candidate_id": 5, "collision": false, "in_bounds": true, "information_gain": 147, "path_blocked": false, "path_cost": 2.0, "score": 143.0, "valid": true, "x": 3.5176, "y": 1.7319, "yaw": 1.309}

## Artifacts

- candidate_table: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase3_candidate_smoke/candidate_table.csv
- candidate_summary: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase3_candidate_smoke/candidate_summary.json

## Scope

This smoke samples candidate viewpoints around the robot marker pose and computes score = information_gain - alpha * path_cost. It performs no training, no checkpointing, no RL, and no rollout execution.

## Next Step

Phase 4 can use the candidate selector in a bounded closed-loop smoke, moving the robot marker through selected goals and updating the BEV map after each step.
