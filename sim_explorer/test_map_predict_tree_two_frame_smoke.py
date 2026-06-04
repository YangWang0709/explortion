#!/usr/bin/env python3
"""Validate Stage 4A-6.5p map_predict + source-protected tree two-frame outputs."""

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
    "observed_state_frame001.npy",
    "frame001_observed_summary.json",
    "frame001_prediction/global_prediction_layer.npz",
    "frame001_prediction/local_prediction.npz",
    "frame001_prediction/prediction_alignment_summary.json",
    "frame001_prediction/global_prediction_topdown.png",
    "frame001_prediction/observed_vs_prediction_topdown.png",
    "frame001_measured_tree_decision.json",
    "frame001_measured_tree_decision.md",
    "frame001_sc_tree_decision.json",
    "frame001_sc_tree_decision.md",
    "frame001_sc_tree_segments.jsonl",
    "frame001_sc_gain_cost_value_table.csv",
    "frame001_sc_node_gain_breakdown.csv",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_depth.png",
    "frame002_pose.json",
    "observed_state_frame002.npy",
    "frame002_observed_summary.json",
    "frame002_prediction/global_prediction_layer.npz",
    "frame002_prediction/local_prediction.npz",
    "frame002_prediction/prediction_alignment_summary.json",
    "frame002_prediction/global_prediction_topdown.png",
    "frame002_prediction/observed_vs_prediction_topdown.png",
    "frame002_measured_tree_decision.json",
    "frame002_measured_tree_decision.md",
    "frame002_sc_tree_decision.json",
    "frame002_sc_tree_decision.md",
    "frame002_sc_tree_segments.jsonl",
    "frame002_sc_gain_cost_value_table.csv",
    "frame002_sc_node_gain_breakdown.csv",
    "map_predict_tree_two_frame_summary.json",
    "map_predict_tree_two_frame_summary.md",
    "frame001_measured_vs_sc_comparison.json",
    "frame001_measured_vs_sc_comparison.md",
    "frame002_measured_vs_sc_comparison.json",
    "frame002_measured_vs_sc_comparison.md",
    "frame001_vs_frame002_sc_tree_comparison.json",
    "frame001_vs_frame002_sc_tree_comparison.md",
    "prediction_safety_checklist.json",
    "prediction_safety_checklist.md",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "observed_ratio_two_frame.json",
    "recommended_next_faithful_step.md",
    "frame001_measured_vs_sc_tree_topdown.png",
    "frame002_measured_vs_sc_tree_topdown.png",
    "two_frame_sc_path_topdown.png",
    "prediction_overlay_frame001_topdown.png",
    "prediction_overlay_frame002_topdown.png",
    "sc_gain_vs_exp_gain_frame001.png",
    "sc_gain_vs_exp_gain_frame002.png",
    "observed_state_hashes.json",
]
PROHIBITED_PATTERNS = [
    "transitions.jsonl",
    "step_*.npz",
    "step_topdown_*.png",
    "observed_ratio_curve.png",
    "rollout_topdown_path.png",
    "rollout_index.html",
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


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    for prefix in ("frame001", "frame002"):
        depth = np.load(output_dir / f"{prefix}_depth.npy")
        positive = depth[np.isfinite(depth) & (depth > 0.0)]
        _assert(depth.ndim == 2, f"{prefix} depth should be HxW, got {depth.shape}")
        _assert(positive.size > 0, f"{prefix} depth has no finite positive values")
        rgb = Image.open(output_dir / f"{prefix}_rgb.png")
        _assert(rgb.size[0] > 0 and rgb.size[1] > 0, f"{prefix} RGB image has invalid dimensions")
        observed = np.load(output_dir / f"observed_state_{prefix}.npy")
        _assert(observed.ndim == 3, f"{prefix} observed_state should be 3D")
    return {"passed": True, "required_outputs": len(REQUIRED_OUTPUTS)}


def test_profile_prediction_and_counts(output_dir: Path) -> dict[str, Any]:
    summary = _load_json(output_dir / "map_predict_tree_two_frame_summary.json")
    source = _load_json(output_dir / "source_protection_checklist.json")
    prediction = _load_json(output_dir / "prediction_safety_checklist.json")
    crop = source["mechanisms"]["crop_min_length_min_path_length"]

    _assert(crop["active"] is True, "crop_min_length should be active")
    _assert(abs(float(crop["value_m"]) - 0.25) <= 1.0e-9, f"wrong crop value: {crop}")
    _assert(source["profile_parameters"]["alignment_convention"] == "code_consistent_v1", "wrong alignment convention")
    _assert(source["prediction"]["enabled"] is True, "prediction should be enabled")
    _assert(source["prediction"]["prediction_used_for_information_gain_only"] is True, "prediction should be information-gain-only")
    _assert(source["prediction"]["prediction_writeback"] is False, "prediction writeback should be false")
    _assert(source["prediction"]["prediction_used_for_collision_traversability"] is False, "prediction collision/traversability leakage")
    _assert(source["prediction"]["prediction_blocks_rays"] is False, "prediction ray blocking leakage")

    _assert(int(prediction["map_predict_predictions"]) == 2, "expected exactly two map_predict predictions")
    _assert(int(prediction["predictor_steps_predicted"]) == 2, "predictor should report two predicted steps")
    _assert(prediction["checkpoint_loaded_once"] is True, "checkpoint/model should be loaded once")
    _assert(summary["safety"]["frames_captured"] == 2, "expected exactly two frames")
    _assert(summary["safety"]["selected_action_execution_count"] == 1, "expected exactly one selected action")
    _assert(summary["answers"]["ready_for_rollout"] is False, "two-frame smoke should not mark rollout ready")
    return {"passed": True}


def test_prediction_outputs_and_gains(output_dir: Path) -> dict[str, Any]:
    results: dict[str, Any] = {"passed": True}
    for prefix in ("frame001", "frame002"):
        decision = _load_json(output_dir / f"{prefix}_sc_tree_decision.json")
        stats = decision["gain_stats"]
        _assert(decision["built_successfully"] is True, f"{prefix} SC tree failed")
        _assert(int(stats["nodes_with_gain_sc_positive"]) > 0, f"{prefix} has no gain_sc-positive nodes")
        _assert(float(decision["accumulated_gain_sc"]) > 0.0, f"{prefix} accumulated gain_sc should be nonzero")
        _assert(
            abs(float(decision["accumulated_gain_hybrid"]) - (float(decision["accumulated_gain_exp"]) + float(decision["accumulated_gain_sc"])))
            <= 1.0e-5,
            f"{prefix} accumulated hybrid identity failed",
        )
        with np.load(output_dir / f"{prefix}_prediction/global_prediction_layer.npz", allow_pickle=False) as data:
            _assert("class_prob" not in data.files, f"{prefix} global prediction should not save class_prob")
            shape = tuple(int(v) for v in data["global_pred_class"].shape)
        observed_shape = tuple(int(v) for v in np.load(output_dir / f"observed_state_{prefix}.npy").shape)
        _assert(shape == observed_shape, f"{prefix} prediction shape {shape} != observed {observed_shape}")
        with (output_dir / f"{prefix}_sc_gain_cost_value_table.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            checked = 0
            for row in reader:
                if row.get("segment_id") == "root":
                    continue
                gain_exp = float(row.get("gain_exp") or 0.0)
                gain_sc = float(row.get("gain_sc") or 0.0)
                gain_hybrid = float(row.get("gain_hybrid") or 0.0)
                _assert(abs(gain_hybrid - (gain_exp + gain_sc)) <= 1.0e-5, f"{prefix} node hybrid identity failed")
                checked += 1
        _assert(checked > 0, f"{prefix} gain table was empty")
        segments = [row for row in _load_jsonl(output_dir / f"{prefix}_sc_tree_segments.jsonl") if row.get("segment_id") != "root"]
        _assert(segments, f"{prefix} segments missing")
        for key in ("gain_exp", "gain_sc", "gain_hybrid", "gain_occ", "gain_conf"):
            _assert(key in segments[0], f"{prefix} segment missing {key}")
        results[f"{prefix}_nodes_with_gain_sc"] = int(stats["nodes_with_gain_sc_positive"])
    return results


def test_observed_hashes_safety_and_absence(output_dir: Path) -> dict[str, Any]:
    hashes = _load_json(output_dir / "observed_state_hashes.json")
    summary = _load_json(output_dir / "map_predict_tree_two_frame_summary.json")
    safety = summary["safety"]
    ratios = _load_json(output_dir / "observed_ratio_two_frame.json")

    prior_path = Path(hashes["episode_prior_observed_state"])
    _assert(prior_path.is_file(), f"prior observed_state missing: {prior_path}")
    _assert(hashes["episode_prior_hash_unchanged"] is True, "episode prior observed_state hash changed")
    _assert(sha256_file(prior_path) == hashes["episode_prior_sha256_after"], "current prior hash mismatch")
    _assert(hashes["frame001_sha256_before_prediction_and_tree"] == hashes["frame001_sha256_after_prediction_and_tree"], "frame001 observed_state changed after prediction/tree")
    _assert(hashes["frame002_sha256_before_prediction_and_tree"] == hashes["frame002_sha256_after_prediction_and_tree"], "frame002 observed_state changed after prediction/tree")
    _assert(hashes["frame001_prior_hash_unchanged_during_frame2_update"] is True, "frame001 observed_state changed during frame2 update")

    frame1 = np.load(output_dir / "observed_state_frame001.npy")
    frame2 = np.load(output_dir / "observed_state_frame002.npy")
    _assert(frame1.shape == frame2.shape, "frame observed_state shapes differ")
    _assert(int(np.count_nonzero(frame1 != frame2)) > 0, "frame2 observed_state did not differ from frame1")
    _assert(int(ratios["frame002_delta_observed_count"]) > 0, "frame2 did not add measured observed voxels")

    _assert(safety["isaac_startup"] is True, "Isaac startup should be true")
    _assert(int(safety["frames_captured"]) == 2, "frames captured should be exactly 2")
    _assert(int(safety["map_predict_predictions"]) == 2, "map_predict predictions should be exactly 2")
    _assert(int(safety["selected_action_execution_count"]) == 1, "selected action should execute once")
    for key in (
        "rollout",
        "online_open_ended_loop",
        "sscnet_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_blocks_rays",
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
    _assert(safety["external_source_git_status_before"] == safety["external_source_git_status_after"], "external source changed during run")

    for pattern in PROHIBITED_PATTERNS:
        found = list(output_dir.rglob(pattern))
        _assert(not found, f"prohibited outputs for {pattern}: {[str(path) for path in found[:5]]}")
    return {"passed": True}


def test_no_target_ground_truth_fields(output_dir: Path) -> dict[str, Any]:
    summary = _load_json(output_dir / "map_predict_tree_two_frame_summary.json")
    safety = summary["safety"]
    _assert(safety["target_lr_target_hr_ground_truth_scoring"] is False, "target/ground-truth scoring flag true")
    _assert(safety["prediction_writeback"] is False, "prediction writeback flag true")
    _assert(safety["prediction_used_for_collision_traversability"] is False, "prediction planning leakage flag true")
    for prefix in ("frame001", "frame002"):
        decision = _load_json(output_dir / f"{prefix}_sc_tree_decision.json")
        _assert(
            decision["safety"]["target_lr_target_hr_ground_truth_scoring"] is False,
            f"{prefix} target/ground-truth scoring flag true",
        )
    return {"passed": True}


def run_tests(output_dir: Path, reference_one_step_sc_dir: Path | None = None, reference_no_pred_two_frame_dir: Path | None = None) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "profile_prediction_and_counts": test_profile_prediction_and_counts(output_dir),
        "prediction_outputs_and_gains": test_prediction_outputs_and_gains(output_dir),
        "observed_hashes_safety_and_absence": test_observed_hashes_safety_and_absence(output_dir),
        "no_target_ground_truth_fields": test_no_target_ground_truth_fields(output_dir),
    }
    payload = {
        "all_passed": all(item["passed"] for item in results.values()),
        "tests": results,
        "reference_one_step_sc_dir": str(reference_one_step_sc_dir) if reference_one_step_sc_dir else None,
        "reference_no_pred_two_frame_dir": str(reference_no_pred_two_frame_dir) if reference_no_pred_two_frame_dir else None,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_one_step_sc_dir", default="")
    parser.add_argument("--reference_no_pred_two_frame_dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    run_tests(
        Path(parsed.output_dir),
        Path(parsed.reference_one_step_sc_dir) if parsed.reference_one_step_sc_dir else None,
        Path(parsed.reference_no_pred_two_frame_dir) if parsed.reference_no_pred_two_frame_dir else None,
    )
