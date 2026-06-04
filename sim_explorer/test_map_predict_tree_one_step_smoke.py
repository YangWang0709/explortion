#!/usr/bin/env python3
"""Validate Stage 4A-6.5o map_predict + source-protected tree one-step outputs."""

from __future__ import annotations

import argparse
import csv
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
    "frame001_camera_info.json",
    "observed_state_frame001.npy",
    "observed_state_update_summary.json",
    "observed_state_hashes.json",
    "prediction_stats.json",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "measured_only_tree_decision.json",
    "measured_only_tree_segments.jsonl",
    "measured_only_gain_cost_value_table.csv",
    "measured_only_tree_topdown.png",
    "sc_tree_decision.json",
    "sc_tree_tree_segments.jsonl",
    "sc_tree_gain_cost_value_table.csv",
    "sc_tree_tree_topdown.png",
    "tree_decision_comparison.json",
    "tree_decision_comparison.md",
    "tree_decision_comparison_topdown.png",
    "predicted_unmeasured_visible_topdown.png",
    "map_predict_tree_one_step_summary.json",
    "map_predict_tree_one_step_summary.md",
    "capture_scene_metadata.json",
    "recommended_next_faithful_step.md",
    "map_predict/global_prediction_layer.npz",
    "map_predict/local_prediction.npz",
    "map_predict/prediction_alignment_summary.json",
]
PROHIBITED_PATTERNS = [
    "transitions.jsonl",
    "step_*.npz",
    "step_topdown_*.png",
    "observed_ratio_curve.png",
    "rollout_topdown_path.png",
    "rollout_index.html",
    "frame002*",
    "frame003*",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def _candidate_value(row: dict[str, Any], key: str) -> float:
    if key in row:
        return float(row[key] or 0.0)
    gains = row.get("gains", {})
    if key in gains:
        return float(gains[key] or 0.0)
    return 0.0


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")

    depth = np.load(output_dir / "frame001_depth.npy")
    positive = depth[np.isfinite(depth) & (depth > 0.0)]
    _assert(depth.ndim == 2, f"frame001 depth should be HxW, got {depth.shape}")
    _assert(positive.size > 0, "frame001 depth has no finite positive values")
    rgb = Image.open(output_dir / "frame001_rgb.png")
    _assert(rgb.size[0] > 0 and rgb.size[1] > 0, "frame001 RGB image has invalid dimensions")
    observed = np.load(output_dir / "observed_state_frame001.npy")
    _assert(observed.ndim == 3, "observed_state should be 3D")
    return {"passed": True, "required_outputs": len(REQUIRED_OUTPUTS)}


def test_prediction_and_tree_outputs(output_dir: Path) -> dict[str, Any]:
    summary = _load_json(output_dir / "map_predict_tree_one_step_summary.json")
    prediction_stats = _load_json(output_dir / "prediction_stats.json")
    checklist = _load_json(output_dir / "source_protection_checklist.json")
    sc_decision = _load_json(output_dir / "sc_tree_decision.json")
    measured_decision = _load_json(output_dir / "measured_only_tree_decision.json")

    answers = summary["answers"]
    _assert(answers["isaac_one_frame_capture_success"] is True, "Isaac capture should succeed")
    _assert(answers["measured_only_observed_state_update_success"] is True, "measured update should succeed")
    _assert(answers["map_predict_success"] is True, "map_predict should succeed")
    _assert(answers["prediction_layer_shape_aligned_to_observed_state"] is True, "prediction shape should align")
    _assert(answers["source_protected_tree_ran_with_prediction_mode"] is True, "SC tree should run")
    _assert(answers["gain_sc_nonzero"] is True, "SC gain should be nonzero")
    _assert(answers["gain_hybrid_identity_passed"] is True, "hybrid identity should pass")
    _assert(answers["prediction_did_not_write_observed_state"] is True, "prediction should not write observed_state")
    _assert(answers["prediction_used_for_traversability_collision_ray_blocking"] is False, "prediction leakage flag true")
    _assert(answers["ready_for_rollout"] is False, "one-step smoke must not mark rollout ready")

    _assert(prediction_stats["shape_aligned_to_observed_state"] is True, "prediction stats shape mismatch")
    _assert(prediction_stats["alignment_convention"] == "code_consistent_v1", "wrong alignment convention")
    _assert(int(prediction_stats["prediction_valid_count"]) > 0, "no valid predictions")
    _assert(int(prediction_stats["predicted_unmeasured_count"]) > 0, "no predicted-unmeasured voxels")

    pred = checklist["prediction"]
    _assert(pred["enabled"] is True, "prediction should be enabled")
    _assert(pred["map_predict_used"] is True, "map_predict should be used")
    _assert(pred["prediction_used_for_information_gain_only"] is True, "prediction should be info-gain-only")
    _assert(pred["prediction_writeback"] is False, "prediction writeback should be false")
    _assert(pred["prediction_used_for_collision_traversability"] is False, "prediction planning leakage")
    _assert(pred["prediction_blocks_rays"] is False, "prediction ray blocking leakage")

    _assert(sc_decision["decision"]["built_successfully"] is True, "SC tree decision failed")
    _assert(measured_decision["decision"]["built_successfully"] is True, "measured tree decision failed")
    _assert(int(sc_decision["gain_stats"]["nodes_with_gain_sc_positive"]) > 0, "no SC tree nodes with gain_sc")
    return {
        "passed": True,
        "selected_changed": bool(answers["selected_child_differs_from_measured_only_tree"]),
        "sc_nodes_with_gain_sc": int(sc_decision["gain_stats"]["nodes_with_gain_sc_positive"]),
    }


def test_gain_records(output_dir: Path) -> dict[str, Any]:
    rows = _load_jsonl(output_dir / "sc_tree_tree_segments.jsonl")
    non_root = [row for row in rows if row.get("segment_id") != "root"]
    _assert(non_root, "SC tree segments are empty")
    for row in non_root:
        for key in ("gain_exp", "gain_sc", "gain_hybrid", "gain_occ", "gain_conf"):
            _assert(key in row, f"missing {key} in segment {row.get('segment_id')}")
            _assert(np.isfinite(float(row[key])), f"non-finite {key} in segment {row.get('segment_id')}")

    max_error = 0.0
    with (output_dir / "sc_tree_gain_cost_value_table.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("segment_id") == "root":
                continue
            gain_exp = float(row.get("gain_exp") or 0.0)
            gain_sc = float(row.get("gain_sc") or 0.0)
            gain_hybrid = float(row.get("gain_hybrid") or 0.0)
            max_error = max(max_error, abs(gain_hybrid - (gain_exp + gain_sc)))
    _assert(max_error <= 1.0e-5, f"gain_hybrid identity error too high: {max_error}")

    selected = _load_json(output_dir / "sc_tree_decision.json")["decision"]["selected"]
    for key in ("gain_exp", "gain_sc", "gain_hybrid", "gain_occ", "gain_conf"):
        _assert(key in selected, f"selected child missing {key}")
    return {"passed": True, "segments_checked": len(non_root), "max_hybrid_error": max_error}


def test_observed_hashes_and_safety(output_dir: Path) -> dict[str, Any]:
    hashes = _load_json(output_dir / "observed_state_hashes.json")
    summary = _load_json(output_dir / "map_predict_tree_one_step_summary.json")
    safety = summary["safety"]

    prior_path = Path(hashes["episode_prior_observed_state"])
    _assert(prior_path.is_file(), f"prior observed_state missing: {prior_path}")
    _assert(hashes["episode_prior_hash_unchanged"] is True, "episode prior observed_state hash changed")
    _assert(sha256_file(prior_path) == hashes["episode_prior_sha256_after"], "current prior hash mismatch")
    observed_path = Path(hashes["new_observed_state"])
    _assert(sha256_file(observed_path) == hashes["new_observed_state_sha256_after_prediction_and_tree"], "observed hash mismatch")
    _assert(
        hashes["new_observed_state_sha256_before_prediction_and_tree"]
        == hashes["new_observed_state_sha256_after_prediction_and_tree"],
        "new observed_state was modified after creation",
    )

    _assert(safety["isaac_startup"] is True, "Isaac startup flag should be true")
    _assert(int(safety["frames_captured"]) == 1, "frames captured should be exactly 1")
    _assert(int(safety["selected_action_execution_count"]) == 0, "selected action should not execute")
    for key in (
        "two_frame",
        "selected_action_execution",
        "rollout",
        "online_open_ended_loop",
        "sscnet_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_writeback",
        "prediction_used_for_traversability_collision",
        "prediction_blocks_rays",
        "target_lr_target_hr_ground_truth_scoring",
        "external_source_modified_or_built_by_stage",
        "coverage_improvement_claimed",
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
    return {"passed": True}


def test_absence_of_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    for pattern in PROHIBITED_PATTERNS:
        found = list(output_dir.rglob(pattern))
        _assert(not found, f"prohibited outputs for {pattern}: {[str(path) for path in found[:5]]}")
    with np.load(output_dir / "map_predict/local_prediction.npz", allow_pickle=False) as data:
        _assert("class_prob" not in data.files, "large dense class_prob should not be saved in default smoke")
    return {"passed": True}


def run_tests(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "prediction_and_tree_outputs": test_prediction_and_tree_outputs(output_dir),
        "gain_records": test_gain_records(output_dir),
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
