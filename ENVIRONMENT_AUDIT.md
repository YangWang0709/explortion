# Phase 0 Environment Audit

- phase: Phase 0 environment and project audit
- generated_at: 2026-06-07T15:12:36+08:00
- host: ubuntu22
- PI_WORKSPACE: /home/ubuntu22/pi
- run_dir: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529
- conda_env: env_isaaclab
- env_activation_command: source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
- git_root: /home/ubuntu22/pi
- remote_origin: git@github.com:YangWang0709/sss.git

## Summary

| Check | Result |
| --- | --- |
| PI_WORKSPACE exists | yes |
| GPU command nvidia-smi ran | yes |
| USD asset count | 1 |
| Primary USD | /home/ubuntu22/pi/building_scene.usd |
| Primary USD sha256 | 11e4a3f55af816bc8b9dba3888498612295e6635e29198c6e5e40d6131bc7b8b |
| Primary USD file type | USD crate, version 0.8.0 |
| env_isaaclab has isaacsim module | true |
| env_isaaclab direct pxr import before Isaac startup | false |
| Isaac headless SimulationApp started | true |
| Isaac/Omniverse USD context opened primary USD | false |
| USD context failure | core dump / abort, exit code 134 |
| Existing robot/sensor/mapping/rollout source code | not found yet |
| Files over 50MB | 0 |
| Phase 0 gate | blocked on primary USD load crash |

## Current Project Capability

- Environment can activate env_isaaclab.
- Isaac headless starts on the Ubuntu host with the RTX 5080 visible.
- The current scene file exists and is a USDC crate, but opening it through Isaac/Omniverse crashes the process.
- PI_WORKSPACE currently has no robot/sensor/mapping/rollout implementation files.

## Minimal Fix Direction

1. Preserve the original /home/ubuntu22/pi/building_scene.usd exactly as-is.
2. Add a separate minimal indoor USDA smoke scene for Phase 1, or repair/convert the current USDC if a reliable converter path becomes available.
3. Run Phase 1 only after at least one USD scene can be loaded without crashing Isaac.

## Key Logs

- initial audit log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs/phase0_command_outputs.log
- corrected import probe: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs/phase0_command_outputs_fix.log
- Isaac USD context crash log: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs/phase0_isaac_usd_context_probe.log
- USD file inspection: /home/ubuntu22/pi/runs/isaac_3d_active_explore_20260607_150529/logs/phase0_usd_file_inspection.log

## Constraints Confirmed

- No PI/openpi/VLM training was run.
- No RL was run.
- No checkpoint was created.
- No rollout was started.
- Original USD scene was not deleted.
- Large generated data is excluded by .gitignore patterns.
