# Phase 4 Closed-Loop Smoke Report

- phase: Phase 4 bounded baseline closed-loop smoke
- generated_at: 2026-06-07T15:29:24.686909+08:00
- PI_WORKSPACE: /home/ubuntu22/pi
- smoke_exit_code: 0
- action_count: 8
- step_count: 8
- final_pose: {'x': -0.0247, 'y': -1.5894}
- final_counts: {"known_free": 910, "known_ratio": 0.6338899196042053, "occupied": 115, "unknown": 592}
- coverage_increased: True

## Artifacts

- closed_loop_steps: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase4_closed_loop_smoke/closed_loop_steps.csv
- closed_loop_summary: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/phase4_closed_loop_smoke/closed_loop_summary.json

## Scope

This is a bounded closed-loop selector smoke on the fallback BEV map. It uses score = information_gain - alpha * path_cost and updates map coverage after each selected action. It is not the required 10-start long rollout and does not train, checkpoint, or run RL.

## Next Step

Phase 5 can collect a bounded fallback long-rollout packet with at least 10 starts, then prepare a human review packet. The original primary USDC crash remains a separate blocker for using building_scene.usd directly.
