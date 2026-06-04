#!/usr/bin/env python3
"""Smoke tests for Stage 4A-3 empty-prediction expert rollout outputs."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from depth_to_voxel import FREE, OCCUPIED, UNKNOWN, create_observed_grid, update_observed_state_from_depth
from sim_paper_expert import FEATURE_NAMES, select_sim_expert_action
from sim_rollout_utils import (
    build_transition,
    compute_next_pose_from_candidate,
    empty_prediction_layer_for,
    leakage_checks,
    load_json,
    load_jsonl,
    observed_summary,
    pose_dict,
    write_transition_records,
)

FORBIDDEN_EXACT_FIELDS = {"target_lr", "target_hr", "ground_truth", "gt"}


def _assert_no_forbidden_exact_fields(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN_EXACT_FIELDS.intersection(value.keys())
        assert not overlap, f"forbidden exact fields at {path}: {sorted(overlap)}"
        for key, child in value.items():
            _assert_no_forbidden_exact_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _assert_no_forbidden_exact_fields(child, f"{path}[{idx}]")


def _synthetic_depth_case() -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any], float]:
    depth = np.full((24, 32), 1.8, dtype=np.float32)
    depth[:4, :] = 2.6
    pose = pose_dict([0.0, 0.0, 1.0], 0.0)
    camera_info = {"fx": 22.0, "fy": 22.0, "cx": 15.5, "cy": 11.5, "max_depth": 3.0}
    bounds = {"x": [-1.0, 3.0], "y": [-2.0, 2.0], "z": [0.0, 2.4]}
    return depth, pose, camera_info, bounds, 0.2


def test_synthetic_transition_serialization() -> None:
    depth, pose, camera_info, bounds, voxel_size = _synthetic_depth_case()
    observed = create_observed_grid(bounds, voxel_size=voxel_size)
    before = observed_summary(observed)
    observed = update_observed_state_from_depth(
        observed_state=observed,
        depth=depth,
        camera_pose=pose,
        camera_info=camera_info,
        bounds=bounds,
        voxel_size=voxel_size,
        pixel_stride=2,
    )
    after = observed_summary(observed)
    assert after["observed_count"] > before["observed_count"], (before, after)
    assert set(np.unique(observed).tolist()).issubset({int(UNKNOWN), int(FREE), int(OCCUPIED)})

    before_expert = observed.copy()
    result = select_sim_expert_action(
        observed_state=observed,
        current_pose_world=pose,
        bounds=bounds,
        voxel_size=voxel_size,
        prediction_layer=empty_prediction_layer_for(observed),
        prediction_mode="empty",
        num_candidates=8,
        top_n=4,
        gain_mode="hybrid",
        seed=7,
        max_range_voxels=20,
        num_yaw=8,
        num_pitch=3,
    )
    assert np.array_equal(before_expert, observed), "expert modified observed_state"
    for candidate in result["top_candidates"]:
        assert candidate.gain_sc == 0.0
        assert candidate.gain_hybrid == candidate.gain_exp

    next_pose = compute_next_pose_from_candidate(
        result["best_candidate"],
        bounds=bounds,
        voxel_size=voxel_size,
        observed_shape=tuple(observed.shape),
        motion_mode="planar",
        camera_height=1.0,
    )
    transition = build_transition(
        episode_id="synthetic_rollout_test",
        step=0,
        result=result,
        current_pose_world=pose,
        next_pose=next_pose,
        summary_before=before,
        summary_after=after,
        bounds=bounds,
        voxel_size=voxel_size,
        prediction_mode="empty",
        gain_mode="hybrid",
        path_cost_mode="euclidean",
        motion_mode="planar",
        done=False,
        done_reason="",
        prediction_wrote_observed_map=False,
    )
    assert transition["delta_observed_ratio"] > 0.0
    assert transition["expert_action"] == 0
    assert transition["candidate_features"].shape[1] == len(FEATURE_NAMES)
    assert transition["leakage_checks"] == leakage_checks(False, a_star_planner=False)

    with tempfile.TemporaryDirectory() as tmp:
        episode_dir = Path(tmp)
        paths = write_transition_records(episode_dir, transition)
        assert Path(paths["npz"]).exists()
        assert Path(paths["jsonl"]).exists()
        records = load_jsonl(paths["jsonl"])
        assert len(records) == 1
        _assert_no_forbidden_exact_fields(records[0])
        with np.load(paths["npz"], allow_pickle=False) as data:
            required = {
                "candidate_features",
                "feature_names",
                "candidate_positions_grid",
                "candidate_positions_world",
                "candidate_yaws",
                "valid_mask",
                "expert_action",
                "expert_scores",
                "selected_next_pose_world",
                "selected_next_pose_grid",
                "best_gain_sc",
                "prediction_wrote_observed_map",
                "rl_or_ppo_training",
                "optimizer_step",
            }
            assert required.issubset(set(data.files)), sorted(set(data.files))
            assert int(data["expert_action"]) == 0
            assert data["candidate_features"].shape[1] == len(FEATURE_NAMES)
            assert np.allclose(data["candidate_features"][:, 1], 0.0)
            assert float(data["best_gain_sc"]) == 0.0
            assert bool(data["prediction_wrote_observed_map"]) is False
            assert bool(data["rl_or_ppo_training"]) is False
            assert bool(data["optimizer_step"]) is False
            assert FORBIDDEN_EXACT_FIELDS.isdisjoint(set(data.files))


def validate_real_episode(episode_dir: Path) -> dict[str, Any]:
    summary_path = episode_dir / "episode_summary.json"
    transitions_path = episode_dir / "transitions.jsonl"
    final_map_path = episode_dir / "observed_state_final.npy"

    assert summary_path.exists(), f"missing episode_summary.json: {summary_path}"
    assert transitions_path.exists(), f"missing transitions.jsonl: {transitions_path}"
    assert final_map_path.exists(), f"missing final observed map: {final_map_path}"
    assert final_map_path.stat().st_size > 0, f"empty final observed map: {final_map_path}"

    summary = load_json(summary_path)
    transitions = load_jsonl(transitions_path)
    assert len(transitions) >= 2, f"expected at least 2 rollout steps, got {len(transitions)}"
    _assert_no_forbidden_exact_fields(summary)
    for transition in transitions:
        _assert_no_forbidden_exact_fields(transition)

    ratios = [float(t["observed_ratio_after"]) for t in transitions]
    assert all(b + 1e-12 >= a for a, b in zip(ratios, ratios[1:])), ratios
    for transition in transitions:
        assert float(transition["observed_ratio_after"]) + 1e-12 >= float(transition["observed_ratio_before"])

    moved = False
    for transition in transitions:
        step = int(transition["step"])
        step_npz = episode_dir / f"step_{step:03d}.npz"
        observed_step = episode_dir / f"observed_state_step{step:03d}.npy"
        assert step_npz.exists(), f"missing per-step npz: {step_npz}"
        assert observed_step.exists(), f"missing per-step observed map: {observed_step}"

        with np.load(step_npz, allow_pickle=False) as data:
            assert "candidate_features" in data.files
            assert "expert_action" in data.files
            assert "expert_scores" in data.files
            assert "valid_mask" in data.files
            assert "best_gain_sc" in data.files
            assert "prediction_wrote_observed_map" in data.files
            assert FORBIDDEN_EXACT_FIELDS.isdisjoint(set(data.files))

            candidate_features = data["candidate_features"]
            expert_action = int(data["expert_action"])
            valid_mask = data["valid_mask"].astype(bool)
            assert candidate_features.ndim == 2
            assert 0 <= expert_action < candidate_features.shape[0], expert_action
            assert bool(valid_mask[expert_action]), (expert_action, valid_mask)
            assert np.isfinite(candidate_features).all()
            assert np.allclose(candidate_features[:, 1], 0.0)
            assert float(data["best_gain_sc"]) == 0.0
            assert bool(data["prediction_wrote_observed_map"]) is False

        current = np.asarray(transition["current_pose_world"], dtype=np.float64)
        selected = np.asarray(transition["selected_next_pose_world"], dtype=np.float64)
        if np.linalg.norm(current - selected) > 1e-6:
            moved = True

        checks = transition["leakage_checks"]
        assert checks["prediction_mode"] == "empty"
        assert checks["prediction_layer"] == "EmptyPredictionLayer"
        assert checks["prediction_wrote_observed_map"] is False
        assert checks["optimizer_step"] is False
        assert checks["rl_or_ppo_training"] is False
        assert checks["behavior_cloning_training"] is False
        assert checks["imitation_learning_training"] is False
        assert checks["scene_ground_truth_used"] is False
        assert checks["simulator_ground_truth_used"] is False
        assert checks["target_lr_used"] is False
        assert checks["target_hr_used"] is False
        assert transition["prediction_mode"] == "empty"
        assert transition["gain_mode"] == "hybrid"

    assert moved, "camera pose never changed"
    final_state = np.load(final_map_path)
    assert set(np.unique(final_state).tolist()).issubset({int(UNKNOWN), int(FREE), int(OCCUPIED)})

    summary_checks = summary["leakage_checks"]
    assert summary_checks["prediction_wrote_observed_map"] is False
    assert summary_checks["optimizer_step"] is False
    assert summary_checks["rl_or_ppo_training"] is False
    assert summary_checks["behavior_cloning_training"] is False
    assert summary_checks["imitation_learning_training"] is False
    assert summary["prediction_mode"] == "empty"
    assert summary["prediction_layer"] == "EmptyPredictionLayer"
    assert int(summary["steps_completed"]) == len(transitions)

    return {
        "steps": len(transitions),
        "observed_ratio_start": float(summary["observed_ratio_start"]),
        "observed_ratio_end": float(summary["observed_ratio_end"]),
        "done_reason": summary["done_reason"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test Stage 4A-3 simulator expert rollout outputs.")
    parser.add_argument(
        "--episode_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_synthetic_transition_serialization()
    episode_dir = Path(args.episode_dir)
    real_summary = None
    if episode_dir.exists():
        real_summary = validate_real_episode(episode_dir)

    print("Stage 4A-3 rollout smoke test passed.")
    print("synthetic_transition_serialization: ok")
    if real_summary is None:
        print(f"real_episode_validation: skipped, episode_dir not found: {episode_dir}")
    else:
        print(f"real_episode_validation: ok {real_summary}")
    print("observed_ratio_non_decreasing: yes")
    print("gain_sc_empty_prediction: zero")
    print("prediction_writes_observed_map: no")
    print("rl_optimizer_bc_training_run: no")


if __name__ == "__main__":
    main()
