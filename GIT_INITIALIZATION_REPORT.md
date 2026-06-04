# Git Initialization Report

## Summary

- Workspace: `/home/ubuntu22/sc_explorer_ws`
- Already a Git repository before this task: `false`
- Repository initialized during this task: `true`
- Branch after initialization: `master`
- `.gitignore`: created at repository root
- Tracked file count at initial commit: `498`
- Commit message: `Initialize SC-Explorer IsaacSim project repository`
- Commit hash: generated after commit in `git_initial_commit_hash.txt`

The final commit hash cannot be embedded into a file inside the same commit without changing that commit hash. The post-commit hash file is generated separately for reproducibility.

## Scans

- Large files over 50 MB: `23`
- Large files over 200 MB: `23`
- Sensitive candidate paths: `12`
- Sensitive candidate note: all detected candidates are local asset paths matching the broad `*secret*` pattern through `SecretaryDesk`; they are still excluded from Git with the asset/dependency ignore rules.
- Large staged files: `0`
- Forbidden artifact paths staged: `0`
- Embedded repository gitlinks staged: `0`

## Tracked Content Policy

Tracked content is limited to source code, lightweight configs/scripts, project context Markdown, and reproducibility documentation.

Excluded content:

- runtime outputs
- logs
- checkpoints
- datasets
- root `building_scene.usd`
- Isaac dependency packages
- localized USD packages
- PNG/MP4/NPY/NPZ products
- binary model weights

## Safety Artifacts

- Tracked file list: `git_tracked_files.txt`
- Ignored summary: `git_ignored_summary.txt`
- Safety check script: `scripts/check_git_repo_safety.sh`
- Safety check log: `logs/git_repo_safety_check.log`

## Negative Scope

- no Isaac startup
- no capture
- no rollout
- no expert sampling
- no map_predict
- no SSCNet inference
- no RL/GDPO/PPO/BC/IL
- no prediction NPZ generation
- no training
- no checkpoint modification
- no source USD modification
