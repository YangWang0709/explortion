#!/usr/bin/env python3
"""Validate Stage 4A-6.5n Isaac two-frame tree smoke outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from offline_mini_rrt_tree import sha256_file


CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)
REQUIRED_OUTPUTS = [
    "frame001_rgb.png",
    "frame001_depth.npy",
    "frame001_depth.png",
    "frame001_pose.json",
    "observed_state_frame001.npy",
    "frame001_tree_decision.json",
    "frame001_tree_decision.md",
    "frame001_tree_segments.jsonl",
    "frame001_gain_cost_value_table.csv",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_depth.png",
    "frame002_pose.json",
    "observed_state_frame002.npy",
    "frame002_tree_decision.json",
    "frame002_tree_decision.md",
    "frame002_tree_segments.jsonl",
    "frame002_gain_cost_value_table.csv",
    "two_frame_tree_summary.json",
    "two_frame_tree_summary.md",
    "observed_ratio_two_frame.json",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "frame001_tree_topdown.png",
    "frame002_tree_topdown.png",
    "two_frame_path_topdown.png",
    "recommended_next_faithful_step.md",
    "observed_state_hashes.json",
]
PROHIBITED_NAMES = [
    "transitions.jsonl",
    "observed_ratio_curve.png",
    "rollout_topdown_path.png",
    "rollout_index.html",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_status_short(path: Path) -> str:
    if not path.exists():
        return "missing"
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(path),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip()}"
    return completed.stdout.strip()


def _scan_map_predict_artifacts(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ("*map_predict*", "*prediction*.npz", "*class_prob*", "*logits*.npy"):
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    for frame in ("frame001", "frame002"):
        depth = np.load(output_dir / f"{frame}_depth.npy")
        positive = depth[np.isfinite(depth) & (depth > 0.0)]
        _assert(depth.ndim == 2, f"{frame} depth should be HxW, got {depth.shape}")
        _assert(positive.size > 0, f"{frame} depth has no finite positive values")
        rgb = Image.open(output_dir / f"{frame}_rgb.png")
        _assert(rgb.size[0] > 0 and rgb.size[1] > 0, f"{frame} RGB image has invalid dimensions")
        observed = np.load(output_dir / f"observed_state_{frame}.npy")
        _assert(observed.ndim == 3, f"{frame} observed_state should be 3D")
    return {"passed": True, "required_outputs": len(REQUIRED_OUTPUTS)}


def test_profile_and_decisions(output_dir: Path) -> dict[str, Any]:
    checklist = _load_json(output_dir / "source_protection_checklist.json")
    frame1 = _load_json(output_dir / "frame001_tree_decision.json")
    frame2 = _load_json(output_dir / "frame002_tree_decision.json")
    summary = _load_json(output_dir / "two_frame_tree_summary.json")

    crop = checklist["mechanisms"]["crop_min_length_min_path_length"]
    _assert(crop["active"] is True, "crop_min_length should be active")
    _assert(abs(float(crop["value_m"]) - 0.25) <= 1.0e-9, f"wrong crop value: {crop}")
    _assert(checklist["prediction"]["enabled"] is False, "prediction should be disabled")
    _assert(checklist["prediction"]["map_predict_used"] is False, "map_predict should be disabled")
    _assert(checklist["profile_parameters"]["gain_mode"] == "exp", "gain mode should be measured-only exp")

    _assert(frame1["built_successfully"] is True, "frame1 tree decision failed")
    _assert(frame2["built_successfully"] is True, "frame2 tree decision failed")
    _assert(frame1["extra"]["matches_reference_one_step_exactly"] is True, "frame1 did not reproduce one-step")
    _assert(frame1["nonlocal_branch"] is True, "frame1 should be nonlocal")
    _assert(frame2["nonlocal_branch"] is True, "frame2 should remain nonlocal")
    _assert(summary["answers"]["map_predict_or_prediction_used"] is False, "prediction/map_predict should be absent")
    _assert(summary["answers"]["ready_for_rollout"] is False, "summary should not mark rollout ready")
    return {
        "passed": True,
        "frame1_selected": frame1["selected_child"].get("segment_id"),
        "frame1_best": frame1["best_descendant"].get("segment_id"),
        "frame2_selected": frame2["selected_child"].get("segment_id"),
        "frame2_best": frame2["best_descendant"].get("segment_id"),
    }


def test_observed_hashes_and_safety(output_dir: Path) -> dict[str, Any]:
    hashes = _load_json(output_dir / "observed_state_hashes.json")
    ratios = _load_json(output_dir / "observed_ratio_two_frame.json")
    summary = _load_json(output_dir / "two_frame_tree_summary.json")
    safety = summary["safety"]

    prior_path = Path(hashes["episode_prior_observed_state"])
    _assert(prior_path.is_file(), f"prior observed_state missing: {prior_path}")
    _assert(hashes["episode_prior_hash_unchanged"] is True, "episode prior observed_state hash changed")
    _assert(sha256_file(prior_path) == hashes["episode_prior_sha256_after"], "current prior hash mismatch")
    _assert(hashes["frame001_prior_hash_unchanged_during_frame2_update"] is True, "frame001 observed_state was modified during frame2 update")

    frame1 = np.load(output_dir / "observed_state_frame001.npy")
    frame2 = np.load(output_dir / "observed_state_frame002.npy")
    _assert(frame1.shape == frame2.shape, "frame observed_state shapes differ")
    _assert(int(np.count_nonzero(frame1 != frame2)) > 0, "frame2 observed_state did not change from frame1")
    _assert(int(ratios["frame002_delta_observed_count"]) > 0, "frame2 did not add measured observed voxels")
    _assert(ratios["prediction_used"] is False, "ratios report prediction use")
    _assert(ratios["map_predict_used"] is False, "ratios report map_predict use")

    _assert(safety["isaac_startup"] is True, "Isaac startup flag should be true")
    _assert(int(safety["frames_captured"]) == 2, "frames captured should be exactly 2")
    _assert(int(safety["selected_action_execution_count"]) == 1, "selected action should execute exactly once")
    for key in (
        "rollout",
        "online_open_ended_loop",
        "map_predict_rerun",
        "sscnet_inference_or_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "target_lr_target_hr_ground_truth_scoring",
        "external_source_modified_or_built_by_stage",
    ):
        _assert(not bool(safety.get(key, False)), f"safety flag should be false: {key}")

    checkpoint_hash = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    if checkpoint_hash is not None:
        _assert(safety["checkpoint_sha256_before"] == safety["checkpoint_sha256_after"], "checkpoint summary hash changed")
        _assert(safety["checkpoint_sha256_after"] == checkpoint_hash, "checkpoint current hash mismatch")

    external_status = _git_status_short(EXTERNAL_SOURCE_DIR)
    _assert(external_status == safety["external_source_git_status_after"], "external source status changed after summary")
    _assert(
        safety["external_source_git_status_before"] == safety["external_source_git_status_after"],
        "external source status changed during run",
    )
    return {"passed": True, "frame002_delta_observed_count": int(ratios["frame002_delta_observed_count"])}


def test_absence_of_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    for name in PROHIBITED_NAMES:
        _assert(not (output_dir / name).exists(), f"prohibited output exists: {name}")
    _assert(not list(output_dir.rglob("transitions.jsonl")), "transitions.jsonl rollout manifest exists")
    _assert(not list(output_dir.rglob("step_*.npz")), "rollout step npz outputs exist")
    _assert(not list(output_dir.rglob("step_topdown_*.png")), "rollout step topdown outputs exist")
    _assert(not list(output_dir.rglob("observed_ratio_curve.png")), "observed ratio curve exists")
    _assert(not list(output_dir.rglob("rollout_topdown_path.png")), "rollout topdown path exists")
    _assert(not list(output_dir.rglob("frame003*")), "third frame artifacts exist")
    map_predict_artifacts = _scan_map_predict_artifacts(output_dir)
    _assert(not map_predict_artifacts, f"map_predict artifacts found: {map_predict_artifacts[:5]}")
    return {"passed": True}


def run_tests(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "profile_and_decisions": test_profile_and_decisions(output_dir),
        "observed_hashes_and_safety": test_observed_hashes_and_safety(output_dir),
        "absence_of_forbidden_outputs": test_absence_of_forbidden_outputs(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    run_tests(Path(parsed.output_dir))
