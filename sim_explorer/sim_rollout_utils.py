#!/usr/bin/env python3
"""Utilities for Stage 4A-3 deterministic simulator expert rollouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from depth_to_voxel import create_observed_grid, summarize_observed_grid
from sim_paper_expert import (
    EmptyPredictionLayer,
    FEATURE_NAMES,
    grid_to_world,
    normalize_bounds,
    world_to_grid,
)

ROLLOUT_LIMITATION_NOTES = [
    "Stage 4A-3 uses teleport camera motion, not physical robot execution.",
    "Stage 4A-3 uses EmptyPredictionLayer only; no SSCNet map_predict is used.",
    "Prediction is read-only and is never written into observed_state.",
    "Observed_state updates come only from Isaac depth ray marching.",
    "No NYU target_lr/target_hr, scene ground truth, or simulator ground truth is used.",
    "No RL, PPO, behavior cloning training, imitation learning training, or optimizer step is run.",
    "Stage 4A-3.6 can use A* with reachability-aware candidate sampling, but no full SC-Explorer RRT tree planner is run.",
    "A* scoring does not execute a physical path; rollout motion remains teleport-based.",
]


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(record), sort_keys=True, allow_nan=False) + "\n")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def pose_dict(position: list[float] | tuple[float, float, float] | np.ndarray, yaw_rad: float) -> dict[str, Any]:
    xyz = np.asarray(position, dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {xyz.shape}")
    yaw = wrap_angle(float(yaw_rad))
    return {
        "position": [float(v) for v in xyz.tolist()],
        "yaw_rad": yaw,
        "yaw_deg": float(math.degrees(yaw)),
    }


def pose_target(position: list[float] | np.ndarray, yaw_rad: float) -> list[float]:
    xyz = np.asarray(position, dtype=np.float64)
    return [
        float(xyz[0] + math.cos(float(yaw_rad))),
        float(xyz[1] + math.sin(float(yaw_rad))),
        float(xyz[2]),
    ]


def clamp_position_to_bounds(
    position: list[float] | np.ndarray,
    bounds: dict[str, Any],
    margin: float = 0.05,
) -> list[float]:
    normalized = normalize_bounds(bounds)
    xyz = np.asarray(position, dtype=np.float64).copy()
    for axis_id, axis in enumerate(("x", "y", "z")):
        lo, hi = normalized[axis]
        if hi <= lo:
            raise ValueError(f"invalid bounds for {axis}: {(lo, hi)}")
        axis_margin = min(float(margin), max((hi - lo) * 0.25, 0.0))
        xyz[axis_id] = float(np.clip(xyz[axis_id], lo + axis_margin, hi - axis_margin))
    return [float(v) for v in xyz.tolist()]


def initialize_observed_state(bounds: dict[str, Any], voxel_size: float) -> np.ndarray:
    return create_observed_grid(normalize_bounds(bounds), voxel_size=float(voxel_size))


def observed_summary(observed_state: np.ndarray) -> dict[str, Any]:
    return summarize_observed_grid(observed_state)


def leakage_checks(
    prediction_wrote_observed_map: bool = False,
    a_star_planner: bool = False,
    prediction_mode: str = "empty",
    prediction_layer: str = "EmptyPredictionLayer",
) -> dict[str, Any]:
    return {
        "target_lr_used": False,
        "target_hr_used": False,
        "scene_ground_truth_used": False,
        "simulator_ground_truth_used": False,
        "prediction_mode": str(prediction_mode),
        "prediction_layer": str(prediction_layer),
        "prediction_wrote_observed_map": bool(prediction_wrote_observed_map),
        "optimizer_step": False,
        "rl_or_ppo_training": False,
        "behavior_cloning_training": False,
        "imitation_learning_training": False,
        "robot_physics_or_path_execution": False,
        "a_star_planner": bool(a_star_planner),
    }


def empty_prediction_layer_for(observed_state: np.ndarray) -> EmptyPredictionLayer:
    return EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape))


def compute_next_pose_from_candidate(
    candidate: Any,
    bounds: dict[str, Any],
    voxel_size: float,
    observed_shape: tuple[int, int, int],
    motion_mode: str = "planar",
    camera_height: float = 1.2,
) -> dict[str, Any]:
    """Convert a selected candidate into the next camera pose."""
    if motion_mode not in ("planar", "voxel3d"):
        raise ValueError("motion_mode must be 'planar' or 'voxel3d'")

    candidate_world = np.asarray(candidate.world_position, dtype=np.float64)
    if motion_mode == "planar":
        next_position = [float(candidate_world[0]), float(candidate_world[1]), float(camera_height)]
    else:
        next_position = [float(v) for v in candidate_world.tolist()]

    next_position = clamp_position_to_bounds(next_position, bounds)
    next_yaw = wrap_angle(float(candidate.yaw))
    next_grid = world_to_grid(
        next_position,
        bounds,
        voxel_size,
        shape=tuple(int(v) for v in observed_shape),
        clip=True,
    )
    return {
        "world": pose_dict(next_position, next_yaw),
        "grid": [int(v) for v in next_grid],
        "motion_mode": str(motion_mode),
        "source_candidate_id": int(candidate.id),
    }


def pose_repeat_key(
    pose: dict[str, Any],
    bounds: dict[str, Any],
    voxel_size: float,
    observed_shape: tuple[int, int, int],
    yaw_bin_deg: float = 5.0,
) -> tuple[int, int, int, int]:
    grid = world_to_grid(
        pose["position"],
        bounds,
        voxel_size,
        shape=tuple(int(v) for v in observed_shape),
        clip=True,
    )
    yaw_deg = math.degrees(wrap_angle(float(pose.get("yaw_rad", 0.0))))
    yaw_bin = int(round(yaw_deg / float(yaw_bin_deg)))
    return int(grid[0]), int(grid[1]), int(grid[2]), yaw_bin


def build_transition(
    episode_id: str,
    step: int,
    result: dict[str, Any],
    current_pose_world: dict[str, Any],
    next_pose: dict[str, Any],
    summary_before: dict[str, Any],
    summary_after: dict[str, Any],
    bounds: dict[str, Any],
    voxel_size: float,
    prediction_mode: str,
    gain_mode: str,
    path_cost_mode: str,
    motion_mode: str,
    done: bool,
    done_reason: str,
    prediction_wrote_observed_map: bool,
) -> dict[str, Any]:
    arrays = result["feature_arrays"]
    best = result["best_candidate"]
    current_pose_grid = np.asarray(result["current_grid"], dtype=np.int32)
    current_pose_world_array = np.asarray(current_pose_world["position"], dtype=np.float32)
    selected_next_pose_world = np.asarray(next_pose["world"]["position"], dtype=np.float32)
    selected_next_pose_grid = np.asarray(next_pose["grid"], dtype=np.int32)
    diagnostics = result["diagnostics"]
    snapped_xy = diagnostics.get("snapped_current_xy")
    all_candidates = result.get("all_candidates", [])
    gain_sc_values = np.asarray([float(getattr(candidate, "gain_sc", 0.0)) for candidate in all_candidates], dtype=np.float64)
    effective_gain_sc_values = np.asarray(
        [float(getattr(candidate, "effective_gain_sc", getattr(candidate, "gain_sc", 0.0))) for candidate in all_candidates],
        dtype=np.float64,
    )
    weighted_gain_sc_values = np.asarray(
        [float(getattr(candidate, "weighted_gain_sc", 0.0)) for candidate in all_candidates],
        dtype=np.float64,
    )
    predicted_unmeasured_visible_counts = np.asarray(
        [int(getattr(candidate, "predicted_unmeasured_visible_count", 0)) for candidate in all_candidates],
        dtype=np.int64,
    )
    prediction_layer_name = "EmptyPredictionLayer" if str(prediction_mode) == "empty" else "SimPredictionLayer"

    transition = {
        "episode_id": str(episode_id),
        "step": int(step),
        "current_pose_world": current_pose_world_array,
        "current_pose_grid": current_pose_grid,
        "current_yaw": float(current_pose_world["yaw_rad"]),
        "candidate_features": arrays["candidate_features"],
        "feature_names": np.asarray(FEATURE_NAMES),
        "candidate_positions_grid": arrays["candidate_positions_grid"],
        "candidate_positions_world": arrays["candidate_positions_world"],
        "candidate_yaws": arrays["candidate_yaws"],
        "valid_mask": arrays["valid_mask"],
        "expert_action": int(result["expert_action"]),
        "expert_scores": arrays["expert_scores"],
        "selected_next_pose_world": selected_next_pose_world,
        "selected_next_pose_grid": selected_next_pose_grid,
        "selected_next_yaw": float(next_pose["world"]["yaw_rad"]),
        "best_candidate_id": int(best.id),
        "best_score": float(best.final_score),
        "best_gain_exp": float(best.gain_exp),
        "best_gain_sc": float(best.gain_sc),
        "best_gain_hybrid": float(best.gain_hybrid),
        "best_raw_gain_sc": float(getattr(best, "raw_gain_sc", best.gain_sc)),
        "best_effective_gain_sc": float(getattr(best, "effective_gain_sc", best.gain_sc)),
        "best_gain_hybrid_effective": float(getattr(best, "gain_hybrid_effective", best.gain_hybrid)),
        "best_weighted_gain_sc": float(getattr(best, "weighted_gain_sc", best.gain_sc)),
        "best_gain_hybrid_weighted": float(getattr(best, "gain_hybrid_weighted", best.gain_hybrid)),
        "best_utility_effective_sc": float(getattr(best, "utility_effective_sc", best.utility_sc)),
        "best_utility_hybrid_effective": float(getattr(best, "utility_hybrid_effective", best.utility_hybrid)),
        "best_utility_hybrid_weighted": float(getattr(best, "utility_hybrid_weighted", best.utility_hybrid)),
        "best_gain_occ": float(getattr(best, "gain_occ", 0.0)),
        "best_gain_conf": float(getattr(best, "gain_conf", 0.0)),
        "best_path_cost": float(best.path_cost),
        "gain_exp": float(best.gain_exp),
        "gain_sc": float(best.gain_sc),
        "gain_hybrid": float(best.gain_hybrid),
        "raw_gain_sc": float(getattr(best, "raw_gain_sc", best.gain_sc)),
        "effective_gain_sc": float(getattr(best, "effective_gain_sc", best.gain_sc)),
        "gain_hybrid_effective": float(getattr(best, "gain_hybrid_effective", best.gain_hybrid)),
        "weighted_gain_sc": float(getattr(best, "weighted_gain_sc", best.gain_sc)),
        "gain_hybrid_weighted": float(getattr(best, "gain_hybrid_weighted", best.gain_hybrid)),
        "utility_effective_sc": float(getattr(best, "utility_effective_sc", best.utility_sc)),
        "utility_hybrid_effective": float(getattr(best, "utility_hybrid_effective", best.utility_hybrid)),
        "utility_hybrid_weighted": float(getattr(best, "utility_hybrid_weighted", best.utility_hybrid)),
        "gain_occ": float(getattr(best, "gain_occ", 0.0)),
        "gain_conf": float(getattr(best, "gain_conf", 0.0)),
        "path_cost": float(best.path_cost),
        "final_score": float(best.final_score),
        "tau": float(diagnostics.get("tau", 0.1)),
        "sc_gain_formula": str(diagnostics.get("sc_gain_formula", "raw_count")),
        "sc_occ_threshold": float(diagnostics.get("sc_occ_threshold", 0.7)),
        "sc_conf_threshold": float(diagnostics.get("sc_conf_threshold", 0.3)),
        "sc_count_mode": str(diagnostics.get("sc_count_mode", "raw_count")),
        "score_gain_mode": str(diagnostics.get("score_gain_mode", "hybrid_raw")),
        "sc_gain_weight": float(diagnostics.get("sc_gain_weight", getattr(best, "sc_gain_weight", 1.0))),
        "sc_gain_cap": diagnostics.get("sc_gain_cap"),
        "sc_gain_cap_value": float(diagnostics.get("sc_gain_cap_value", getattr(best, "sc_gain_cap_value", -1.0))),
        "best_astar_reachable": bool(best.astar_reachable),
        "best_astar_path_length_m": float(best.astar_path_length_m if np.isfinite(best.astar_path_length_m) else 0.0),
        "best_astar_num_expanded": int(best.astar_num_expanded),
        "best_astar_path_xy": np.asarray(best.astar_path_xy, dtype=np.int32)
        if best.astar_path_xy
        else np.zeros((0, 2), dtype=np.int32),
        "unknown_count_before": int(summary_before["unknown_count"]),
        "free_count_before": int(summary_before["free_count"]),
        "occupied_count_before": int(summary_before["occupied_count"]),
        "observed_ratio_before": float(summary_before["observed_ratio"]),
        "unknown_count_after": int(summary_after["unknown_count"]),
        "free_count_after": int(summary_after["free_count"]),
        "occupied_count_after": int(summary_after["occupied_count"]),
        "observed_ratio_after": float(summary_after["observed_ratio"]),
        "delta_observed_ratio": float(summary_after["observed_ratio"] - summary_before["observed_ratio"]),
        "frontier_count": int(result["diagnostics"]["frontier_count"]),
        "frontier_adjacent_free_count": int(diagnostics.get("frontier_adjacent_free_count") or 0),
        "candidate_count": int(result["diagnostics"]["num_candidates"]),
        "candidates_with_gain_sc_positive": int(np.count_nonzero(gain_sc_values > 0.0)),
        "candidates_with_effective_gain_sc_positive": int(np.count_nonzero(effective_gain_sc_values > 0.0)),
        "mean_gain_sc": float(np.mean(gain_sc_values)) if gain_sc_values.size else 0.0,
        "max_gain_sc": float(np.max(gain_sc_values)) if gain_sc_values.size else 0.0,
        "mean_effective_gain_sc": float(np.mean(effective_gain_sc_values)) if effective_gain_sc_values.size else 0.0,
        "max_effective_gain_sc": float(np.max(effective_gain_sc_values)) if effective_gain_sc_values.size else 0.0,
        "mean_weighted_gain_sc": float(np.mean(weighted_gain_sc_values)) if weighted_gain_sc_values.size else 0.0,
        "max_weighted_gain_sc": float(np.max(weighted_gain_sc_values)) if weighted_gain_sc_values.size else 0.0,
        "predicted_unmeasured_visible_count_best": int(
            getattr(best, "predicted_unmeasured_visible_count", 0)
        ),
        "predicted_unmeasured_visible_count_total": int(np.sum(predicted_unmeasured_visible_counts))
        if predicted_unmeasured_visible_counts.size
        else 0,
        "prediction_valid_voxels": int(diagnostics.get("prediction_valid_voxels") or 0),
        "prediction_predicted_voxels": int(diagnostics.get("prediction_predicted_voxels") or 0),
        "predicted_occupied_voxels": int(diagnostics.get("prediction_predicted_occupied_voxels") or 0),
        "predicted_unmeasured_voxels": int(diagnostics.get("prediction_predicted_unmeasured_voxels") or 0),
        "prediction_used_for_information_gain": bool(diagnostics.get("prediction_used_for_information_gain", False)),
        "prediction_used_for_traversability": bool(diagnostics.get("prediction_used_for_traversability", False)),
        "prediction_used_for_collision": bool(diagnostics.get("prediction_used_for_collision", False)),
        "prediction_used_for_a_star": bool(diagnostics.get("prediction_used_for_astar", False)),
        "prediction_used_for_astar": bool(diagnostics.get("prediction_used_for_astar", False)),
        "prediction_blocks_rays": bool(diagnostics.get("prediction_blocks_rays", False)),
        "prediction_written_to_observed_state": bool(diagnostics.get("prediction_written_to_observed_state", False)),
        "future_observations_used_for_planning": bool(diagnostics.get("future_observations_used_for_planning", False)),
        "future_observations_used_for_scoring": bool(diagnostics.get("future_observations_used_for_scoring", False)),
        "candidate_sampling_mode": str(diagnostics.get("candidate_sampling_mode", "frontier")),
        "candidate_source": str(diagnostics.get("candidate_source", "frontier")),
        "candidate_source_counts": diagnostics.get("candidate_source_counts", {}),
        "reachable_component_count": int(diagnostics.get("reachable_component_count") or 0),
        "reachable_frontier_adjacent_count": int(diagnostics.get("reachable_frontier_adjacent_count") or 0),
        "reachable_free_fallback_count": int(diagnostics.get("reachable_free_fallback_count") or 0),
        "snapped_current": bool(diagnostics.get("snapped_current", False)),
        "snapped_current_xy": [-1, -1] if snapped_xy is None else [int(v) for v in snapped_xy],
        "snap_distance_cells": -1.0
        if diagnostics.get("snap_distance_cells") is None
        else float(diagnostics.get("snap_distance_cells")),
        "start_traversable": None
        if diagnostics.get("start_traversable") is None
        else bool(diagnostics.get("start_traversable")),
        "no_valid_candidate_reason": str(diagnostics.get("no_valid_candidate_reason", "")),
        "prediction_mode": str(prediction_mode),
        "gain_mode": str(gain_mode),
        "path_cost_mode": str(path_cost_mode),
        "motion_mode": str(motion_mode),
        "done": bool(done),
        "done_reason": str(done_reason),
        "bounds": {axis: [float(v) for v in normalize_bounds(bounds)[axis]] for axis in ("x", "y", "z")},
        "voxel_size": float(voxel_size),
        "leakage_checks": leakage_checks(
            prediction_wrote_observed_map,
            a_star_planner=str(path_cost_mode) == "astar",
            prediction_mode=str(prediction_mode),
            prediction_layer=prediction_layer_name,
        ),
    }
    return transition


def save_transition_npz(path: str | Path, transition: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        episode_id=np.array(str(transition["episode_id"])),
        step=np.array(int(transition["step"]), dtype=np.int64),
        current_pose_world=np.asarray(transition["current_pose_world"], dtype=np.float32),
        current_pose_grid=np.asarray(transition["current_pose_grid"], dtype=np.int32),
        current_yaw=np.array(float(transition["current_yaw"]), dtype=np.float32),
        candidate_features=np.asarray(transition["candidate_features"], dtype=np.float32),
        feature_names=np.asarray(transition["feature_names"]),
        candidate_positions_grid=np.asarray(transition["candidate_positions_grid"], dtype=np.int32),
        candidate_positions_world=np.asarray(transition["candidate_positions_world"], dtype=np.float32),
        candidate_yaws=np.asarray(transition["candidate_yaws"], dtype=np.float32),
        valid_mask=np.asarray(transition["valid_mask"], dtype=bool),
        expert_action=np.array(int(transition["expert_action"]), dtype=np.int64),
        expert_scores=np.asarray(transition["expert_scores"], dtype=np.float32),
        selected_next_pose_world=np.asarray(transition["selected_next_pose_world"], dtype=np.float32),
        selected_next_pose_grid=np.asarray(transition["selected_next_pose_grid"], dtype=np.int32),
        selected_next_yaw=np.array(float(transition["selected_next_yaw"]), dtype=np.float32),
        best_score=np.array(float(transition["best_score"]), dtype=np.float32),
        best_gain_exp=np.array(float(transition["best_gain_exp"]), dtype=np.float32),
        best_gain_sc=np.array(float(transition["best_gain_sc"]), dtype=np.float32),
        best_gain_hybrid=np.array(float(transition["best_gain_hybrid"]), dtype=np.float32),
        best_raw_gain_sc=np.array(float(transition.get("best_raw_gain_sc", transition["best_gain_sc"])), dtype=np.float32),
        best_effective_gain_sc=np.array(
            float(transition.get("best_effective_gain_sc", transition["best_gain_sc"])),
            dtype=np.float32,
        ),
        best_gain_hybrid_effective=np.array(
            float(transition.get("best_gain_hybrid_effective", transition["best_gain_hybrid"])),
            dtype=np.float32,
        ),
        best_weighted_gain_sc=np.array(float(transition.get("best_weighted_gain_sc", 0.0)), dtype=np.float32),
        best_gain_hybrid_weighted=np.array(float(transition.get("best_gain_hybrid_weighted", 0.0)), dtype=np.float32),
        best_utility_effective_sc=np.array(float(transition.get("best_utility_effective_sc", 0.0)), dtype=np.float32),
        best_utility_hybrid_effective=np.array(
            float(transition.get("best_utility_hybrid_effective", 0.0)),
            dtype=np.float32,
        ),
        best_utility_hybrid_weighted=np.array(float(transition.get("best_utility_hybrid_weighted", 0.0)), dtype=np.float32),
        best_gain_occ=np.array(float(transition.get("best_gain_occ", 0.0)), dtype=np.float32),
        best_gain_conf=np.array(float(transition.get("best_gain_conf", 0.0)), dtype=np.float32),
        best_path_cost=np.array(float(transition["best_path_cost"]), dtype=np.float32),
        gain_exp=np.array(float(transition["gain_exp"]), dtype=np.float32),
        gain_sc=np.array(float(transition["gain_sc"]), dtype=np.float32),
        gain_hybrid=np.array(float(transition["gain_hybrid"]), dtype=np.float32),
        raw_gain_sc=np.array(float(transition.get("raw_gain_sc", transition["gain_sc"])), dtype=np.float32),
        effective_gain_sc=np.array(float(transition.get("effective_gain_sc", transition["gain_sc"])), dtype=np.float32),
        gain_hybrid_effective=np.array(
            float(transition.get("gain_hybrid_effective", transition["gain_hybrid"])),
            dtype=np.float32,
        ),
        weighted_gain_sc=np.array(float(transition.get("weighted_gain_sc", 0.0)), dtype=np.float32),
        gain_hybrid_weighted=np.array(float(transition.get("gain_hybrid_weighted", 0.0)), dtype=np.float32),
        utility_effective_sc=np.array(float(transition.get("utility_effective_sc", 0.0)), dtype=np.float32),
        utility_hybrid_effective=np.array(float(transition.get("utility_hybrid_effective", 0.0)), dtype=np.float32),
        utility_hybrid_weighted=np.array(float(transition.get("utility_hybrid_weighted", 0.0)), dtype=np.float32),
        gain_occ=np.array(float(transition.get("gain_occ", 0.0)), dtype=np.float32),
        gain_conf=np.array(float(transition.get("gain_conf", 0.0)), dtype=np.float32),
        path_cost=np.array(float(transition["path_cost"]), dtype=np.float32),
        final_score=np.array(float(transition["final_score"]), dtype=np.float32),
        tau=np.array(float(transition.get("tau", 0.1)), dtype=np.float32),
        sc_gain_formula=np.array(str(transition.get("sc_gain_formula", "raw_count"))),
        sc_occ_threshold=np.array(float(transition.get("sc_occ_threshold", 0.7)), dtype=np.float32),
        sc_conf_threshold=np.array(float(transition.get("sc_conf_threshold", 0.3)), dtype=np.float32),
        sc_count_mode=np.array(str(transition.get("sc_count_mode", "raw_count"))),
        score_gain_mode=np.array(str(transition.get("score_gain_mode", "hybrid_raw"))),
        sc_gain_weight=np.array(float(transition.get("sc_gain_weight", 1.0)), dtype=np.float32),
        sc_gain_cap_value=np.array(float(transition.get("sc_gain_cap_value", -1.0)), dtype=np.float32),
        best_astar_reachable=np.array(bool(transition["best_astar_reachable"])),
        best_astar_path_length_m=np.array(float(transition["best_astar_path_length_m"]), dtype=np.float32),
        best_astar_num_expanded=np.array(int(transition["best_astar_num_expanded"]), dtype=np.int64),
        best_astar_path_xy=np.asarray(transition["best_astar_path_xy"], dtype=np.int32),
        unknown_count_before=np.array(int(transition["unknown_count_before"]), dtype=np.int64),
        free_count_before=np.array(int(transition["free_count_before"]), dtype=np.int64),
        occupied_count_before=np.array(int(transition["occupied_count_before"]), dtype=np.int64),
        observed_ratio_before=np.array(float(transition["observed_ratio_before"]), dtype=np.float32),
        unknown_count_after=np.array(int(transition["unknown_count_after"]), dtype=np.int64),
        free_count_after=np.array(int(transition["free_count_after"]), dtype=np.int64),
        occupied_count_after=np.array(int(transition["occupied_count_after"]), dtype=np.int64),
        observed_ratio_after=np.array(float(transition["observed_ratio_after"]), dtype=np.float32),
        delta_observed_ratio=np.array(float(transition["delta_observed_ratio"]), dtype=np.float32),
        frontier_count=np.array(int(transition["frontier_count"]), dtype=np.int64),
        frontier_adjacent_free_count=np.array(int(transition.get("frontier_adjacent_free_count", 0)), dtype=np.int64),
        candidate_count=np.array(int(transition["candidate_count"]), dtype=np.int64),
        candidate_sampling_mode=np.array(str(transition.get("candidate_sampling_mode", "frontier"))),
        candidate_source=np.array(str(transition.get("candidate_source", "frontier"))),
        reachable_component_count=np.array(int(transition.get("reachable_component_count", 0)), dtype=np.int64),
        reachable_frontier_adjacent_count=np.array(
            int(transition.get("reachable_frontier_adjacent_count", 0)),
            dtype=np.int64,
        ),
        reachable_free_fallback_count=np.array(int(transition.get("reachable_free_fallback_count", 0)), dtype=np.int64),
        snapped_current=np.array(bool(transition.get("snapped_current", False))),
        snapped_current_xy=np.asarray(transition.get("snapped_current_xy", [-1, -1]), dtype=np.int32),
        snap_distance_cells=np.array(float(transition.get("snap_distance_cells", -1.0)), dtype=np.float32),
        reachable_candidates=np.array(int(transition.get("reachable_candidates", 0)), dtype=np.int64),
        unreachable_candidates=np.array(int(transition.get("unreachable_candidates", 0)), dtype=np.int64),
        candidates_with_gain_sc_positive=np.array(
            int(transition.get("candidates_with_gain_sc_positive", 0)),
            dtype=np.int64,
        ),
        candidates_with_effective_gain_sc_positive=np.array(
            int(transition.get("candidates_with_effective_gain_sc_positive", 0)),
            dtype=np.int64,
        ),
        mean_gain_sc=np.array(float(transition.get("mean_gain_sc", 0.0)), dtype=np.float32),
        max_gain_sc=np.array(float(transition.get("max_gain_sc", 0.0)), dtype=np.float32),
        mean_effective_gain_sc=np.array(float(transition.get("mean_effective_gain_sc", 0.0)), dtype=np.float32),
        max_effective_gain_sc=np.array(float(transition.get("max_effective_gain_sc", 0.0)), dtype=np.float32),
        mean_weighted_gain_sc=np.array(float(transition.get("mean_weighted_gain_sc", 0.0)), dtype=np.float32),
        max_weighted_gain_sc=np.array(float(transition.get("max_weighted_gain_sc", 0.0)), dtype=np.float32),
        predicted_unmeasured_visible_count_best=np.array(
            int(transition.get("predicted_unmeasured_visible_count_best", 0)),
            dtype=np.int64,
        ),
        predicted_unmeasured_visible_count_total=np.array(
            int(transition.get("predicted_unmeasured_visible_count_total", 0)),
            dtype=np.int64,
        ),
        prediction_valid_voxels=np.array(int(transition.get("prediction_valid_voxels", 0)), dtype=np.int64),
        predicted_occupied_voxels=np.array(int(transition.get("predicted_occupied_voxels", 0)), dtype=np.int64),
        predicted_unmeasured_voxels=np.array(int(transition.get("predicted_unmeasured_voxels", 0)), dtype=np.int64),
        prediction_used_for_information_gain=np.array(
            bool(transition.get("prediction_used_for_information_gain", False))
        ),
        prediction_used_for_traversability=np.array(bool(transition.get("prediction_used_for_traversability", False))),
        prediction_used_for_collision=np.array(bool(transition.get("prediction_used_for_collision", False))),
        prediction_used_for_a_star=np.array(bool(transition.get("prediction_used_for_a_star", False))),
        prediction_blocks_rays=np.array(bool(transition.get("prediction_blocks_rays", False))),
        prediction_written_to_observed_state=np.array(
            bool(transition.get("prediction_written_to_observed_state", False))
        ),
        map_predict_preprocess_time=np.array(float(transition.get("map_predict_preprocess_time", 0.0)), dtype=np.float32),
        map_predict_inference_time=np.array(float(transition.get("map_predict_inference_time", 0.0)), dtype=np.float32),
        map_predict_alignment_time=np.array(float(transition.get("map_predict_alignment_time", 0.0)), dtype=np.float32),
        map_predict_total_time=np.array(float(transition.get("map_predict_total_time", 0.0)), dtype=np.float32),
        expert_time=np.array(float(transition.get("expert_time", 0.0)), dtype=np.float32),
        step_total_time=np.array(float(transition.get("step_total_time", 0.0)), dtype=np.float32),
        observed_state_hash_before_prediction=np.array(
            str(transition.get("observed_state_hash_before_prediction", ""))
        ),
        observed_state_hash_after_prediction=np.array(str(transition.get("observed_state_hash_after_prediction", ""))),
        observed_state_prediction_modified=np.array(
            bool(transition.get("observed_state_prediction_modified", False))
        ),
        prediction_npz=np.array(str(transition.get("prediction_npz", ""))),
        local_prediction_npz=np.array(str(transition.get("local_prediction_npz", ""))),
        prediction_summary_json=np.array(str(transition.get("prediction_summary_json", ""))),
        gpu_memory_peak=np.array(-1 if transition.get("gpu_memory_peak") is None else int(transition.get("gpu_memory_peak"))),
        prediction_mode=np.array(str(transition["prediction_mode"])),
        gain_mode=np.array(str(transition["gain_mode"])),
        path_cost_mode=np.array(str(transition["path_cost_mode"])),
        motion_mode=np.array(str(transition["motion_mode"])),
        done=np.array(bool(transition["done"])),
        done_reason=np.array(str(transition["done_reason"])),
        prediction_wrote_observed_map=np.array(bool(transition["leakage_checks"]["prediction_wrote_observed_map"])),
        rl_or_ppo_training=np.array(bool(transition["leakage_checks"]["rl_or_ppo_training"])),
        optimizer_step=np.array(bool(transition["leakage_checks"]["optimizer_step"])),
    )


def write_transition_records(
    episode_dir: str | Path,
    transition: dict[str, Any],
) -> dict[str, str]:
    episode_dir = Path(episode_dir)
    step = int(transition["step"])
    npz_path = episode_dir / f"step_{step:03d}.npz"
    jsonl_path = episode_dir / "transitions.jsonl"
    save_transition_npz(npz_path, transition)
    append_jsonl(jsonl_path, transition)
    return {"npz": str(npz_path), "jsonl": str(jsonl_path)}


def transition_score_stats(transitions: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = np.asarray([float(t[key]) for t in transitions if key in t and np.isfinite(float(t[key]))], dtype=np.float64)
    if values.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}
