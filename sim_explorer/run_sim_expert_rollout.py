#!/usr/bin/env python3
"""Stage 4A-3 Isaac rollout script for the empty-prediction expert."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 4A-3 multi-step simulator expert rollout.")
parser.add_argument(
    "--output_dir",
    default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred",
    help="Root output directory for rollout episodes and manifest.",
)
parser.add_argument("--episode_id", default="minimal_room_empty_pred_000")
parser.add_argument("--max_steps", type=int, default=10)
parser.add_argument("--coverage_stop", type=float, default=0.75)
parser.add_argument("--min_score", type=float, default=1e-6)
parser.add_argument("--repeat_pose_patience", type=int, default=3)
parser.add_argument("--num_candidates", type=int, default=64)
parser.add_argument("--top_n", type=int, default=16)
parser.add_argument("--gain_mode", choices=("exp", "sc", "hybrid", "occ", "conf"), default="hybrid")
parser.add_argument("--prediction_mode", choices=("empty",), default="empty")
parser.add_argument("--path_cost_mode", choices=("euclidean", "astar"), default="euclidean")
parser.add_argument("--candidate_sampling_mode", choices=("frontier", "reachable_frontier", "auto"), default="auto")
parser.add_argument("--snap_start_to_traversable", action="store_true")
parser.add_argument("--max_snap_radius_cells", type=int, default=5)
parser.add_argument("--motion_mode", choices=("planar", "voxel3d"), default="planar")
parser.add_argument("--scene_variant", choices=("minimal", "medium_three_rooms"), default="minimal")
parser.add_argument("--scene_seed", type=int, default=0)
parser.add_argument("--start_variant", default="custom")
parser.add_argument("--obstacle_jitter_m", type=float, default=0.0)
parser.add_argument("--map_bound_mode", choices=("scene_metadata", "explicit"), default="scene_metadata")
parser.add_argument("--camera_height", type=float, default=1.2)
parser.add_argument("--voxel_size", type=float, default=0.1)
parser.add_argument("--pixel_stride", type=int, default=2)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_sensor_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=5.0)
parser.add_argument("--settle_steps", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_range_voxels", type=int, default=50)
parser.add_argument("--num_yaw", type=int, default=32)
parser.add_argument("--num_pitch", type=int, default=7)
parser.add_argument("--fov_yaw_deg", type=float, default=90.0)
parser.add_argument("--fov_pitch_deg", type=float, default=60.0)
parser.add_argument("--start_x", type=float, default=0.0)
parser.add_argument("--start_y", type=float, default=0.0)
parser.add_argument("--start_yaw_deg", type=float, default=0.0)
parser.add_argument("--x_min", type=float, default=None)
parser.add_argument("--x_max", type=float, default=None)
parser.add_argument("--y_min", type=float, default=None)
parser.add_argument("--y_max", type=float, default=None)
parser.add_argument("--z_min", type=float, default=None)
parser.add_argument("--z_max", type=float, default=None)
parser.add_argument("--save_rgb", action="store_true")
parser.add_argument("--save_depth", action="store_true")
parser.add_argument("--save_viz", action="store_true")
parser.add_argument("--print_steps", action="store_true")
parser.add_argument("--no_manifest", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

from depth_to_voxel import DEFAULT_MAP_BOUNDS, update_observed_state_from_depth
from scene_factory import build_medium_complex_scene, build_minimal_scene
from sim_paper_expert import select_sim_expert_action
from sim_rollout_utils import (
    ROLLOUT_LIMITATION_NOTES,
    append_jsonl,
    build_transition,
    compute_next_pose_from_candidate,
    empty_prediction_layer_for,
    initialize_observed_state,
    leakage_checks,
    load_jsonl,
    observed_summary,
    pose_dict,
    pose_repeat_key,
    pose_target,
    save_json,
    transition_score_stats,
    write_transition_records,
)

DEPTH_KEY = "distance_to_image_plane"


def _spawn_box(path: str, size: tuple[float, float, float], position: tuple[float, float, float], color) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(path, cfg, translation=position)


def _build_scene_metadata(spawn: bool) -> dict[str, Any]:
    if str(args_cli.scene_variant) == "minimal":
        return build_minimal_scene(spawn=spawn, sim_utils_module=sim_utils if spawn else None)
    if str(args_cli.scene_variant) == "medium_three_rooms":
        return build_medium_complex_scene(
            seed=int(args_cli.scene_seed),
            variant="three_rooms",
            obstacle_jitter_m=float(args_cli.obstacle_jitter_m),
            spawn=spawn,
            sim_utils_module=sim_utils if spawn else None,
        )
    raise ValueError(f"Unsupported scene_variant: {args_cli.scene_variant}")


def design_scene() -> tuple[Camera, dict[str, Any]]:
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    scene_metadata = _build_scene_metadata(spawn=True)

    data_types = [DEPTH_KEY]
    if args_cli.save_rgb:
        data_types.insert(0, "rgb")

    sim_utils.create_prim("/World/CameraRig", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CameraRig/CameraSensor",
        update_period=0.0,
        height=args_cli.camera_sensor_height,
        width=args_cli.camera_width,
        data_types=data_types,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, args_cli.max_depth),
        ),
    )
    return Camera(cfg=camera_cfg), scene_metadata


def _set_camera_pose(camera: Camera, sim: sim_utils.SimulationContext, pose: dict[str, Any]) -> None:
    position = [float(v) for v in pose["position"]]
    target = pose_target(position, float(pose["yaw_rad"]))
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )


def _step_camera(camera: Camera, sim: sim_utils.SimulationContext, settle_steps: int) -> None:
    for _ in range(max(int(settle_steps), 1)):
        sim.step()
        camera.update(dt=sim.get_physics_dt())


def _camera_info_from_camera(camera: Camera) -> dict[str, Any]:
    intrinsic_matrix = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
    return {
        "sensor_api_depth_key": DEPTH_KEY,
        "depth_units": "meters",
        "width": int(args_cli.camera_width),
        "height": int(args_cli.camera_sensor_height),
        "max_depth": float(args_cli.max_depth),
        "near_depth": 0.05,
        "horizontal_fov_deg": 90.0,
        "intrinsic_matrix": intrinsic_matrix.tolist(),
        "fx": float(intrinsic_matrix[0, 0]),
        "fy": float(intrinsic_matrix[1, 1]),
        "cx": float(intrinsic_matrix[0, 2]),
        "cy": float(intrinsic_matrix[1, 2]),
    }


def _extract_depth(camera: Camera) -> np.ndarray:
    depth_tensor = camera.data.output.get(DEPTH_KEY)
    if depth_tensor is None:
        raise KeyError(f"Camera output missing {DEPTH_KEY}. Keys: {list(camera.data.output.keys())}")
    depth = depth_tensor[0].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _save_rgb_if_requested(camera: Camera, path: Path) -> str | None:
    if not args_cli.save_rgb:
        return None
    tensor = camera.data.output.get("rgb")
    if tensor is None:
        tensor = camera.data.output.get("rgba")
    if tensor is None:
        raise KeyError(f"Camera output missing rgb/rgba. Keys: {list(camera.data.output.keys())}")
    image = tensor[0].detach().cpu().numpy()
    rgb = _normalize_rgb(image)
    Image.fromarray(rgb).save(path)
    return str(path)


def _depth_stats(depth: np.ndarray) -> dict[str, Any]:
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    return {
        "shape": [int(v) for v in depth.shape],
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
    }


def _explicit_bounds_from_args() -> dict[str, tuple[float, float]] | None:
    values = {
        "x": (args_cli.x_min, args_cli.x_max),
        "y": (args_cli.y_min, args_cli.y_max),
        "z": (args_cli.z_min, args_cli.z_max),
    }
    provided = [value is not None for pair in values.values() for value in pair]
    if not any(provided):
        return None
    missing = [
        f"{axis}_{suffix}"
        for axis, pair in values.items()
        for suffix, value in zip(("min", "max"), pair)
        if value is None
    ]
    if missing:
        raise ValueError(f"explicit bounds require complete min/max pairs; missing {missing}")
    bounds = {axis: (float(pair[0]), float(pair[1])) for axis, pair in values.items()}
    for axis, pair in bounds.items():
        if pair[1] <= pair[0]:
            raise ValueError(f"invalid {axis} bounds: {pair}")
    return bounds


def _resolve_rollout_bounds(scene_metadata: dict[str, Any]) -> dict[str, tuple[float, float]]:
    if str(args_cli.map_bound_mode) == "explicit":
        explicit = _explicit_bounds_from_args()
        if explicit is None:
            raise ValueError("--map_bound_mode explicit requires --x_min/--x_max/--y_min/--y_max/--z_min/--z_max")
        return explicit

    raw_bounds = scene_metadata.get("map_bounds") or DEFAULT_MAP_BOUNDS
    return {axis: tuple(float(v) for v in raw_bounds[axis]) for axis in ("x", "y", "z")}


def _transition_metric_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    if not transitions:
        return {}
    def stats_optional(key: str) -> dict[str, float | None]:
        values = np.asarray(
            [float(t[key]) for t in transitions if key in t and t[key] is not None and np.isfinite(float(t[key]))],
            dtype=np.float64,
        )
        if values.size == 0:
            return {"min": None, "mean": None, "max": None}
        return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}

    summary = {
        "average_frontier_count": float(np.mean([float(t["frontier_count"]) for t in transitions])),
        "average_candidate_count": float(np.mean([float(t["candidate_count"]) for t in transitions])),
        "best_score": transition_score_stats(transitions, "best_score"),
        "gain_exp": transition_score_stats(transitions, "best_gain_exp"),
        "gain_sc": transition_score_stats(transitions, "best_gain_sc"),
        "gain_hybrid": transition_score_stats(transitions, "best_gain_hybrid"),
        "path_cost": transition_score_stats(transitions, "best_path_cost"),
    }
    if any("reachable_candidates" in t for t in transitions):
        summary["average_reachable_candidates"] = float(
            np.mean([float(t.get("reachable_candidates", 0.0)) for t in transitions])
        )
    if any("best_astar_path_length_m" in t for t in transitions):
        summary["best_astar_path_length_m"] = transition_score_stats(transitions, "best_astar_path_length_m")
    if any("reachable_component_count" in t for t in transitions):
        summary["reachable_component_count"] = stats_optional("reachable_component_count")
        values = [float(t.get("reachable_component_count", 0.0) or 0.0) for t in transitions]
        summary["average_reachable_component_count"] = float(np.mean(values))
    if any("reachable_frontier_adjacent_count" in t for t in transitions):
        summary["reachable_frontier_adjacent_count"] = stats_optional("reachable_frontier_adjacent_count")
        values = [float(t.get("reachable_frontier_adjacent_count", 0.0) or 0.0) for t in transitions]
        summary["average_reachable_frontier_adjacent_count"] = float(np.mean(values))
    source_counts: dict[str, int] = {}
    for transition in transitions:
        source = str(transition.get("candidate_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    summary["candidate_source_counts"] = source_counts
    summary["no_valid_candidate_steps"] = [
        int(t["step"])
        for t in transitions
        if str(t.get("no_valid_candidate_reason", "") or "") or str(t.get("done_reason", "")) == "no_valid_candidate"
    ]
    return summary


def run_rollout() -> dict[str, Any]:
    output_root = Path(args_cli.output_dir).resolve()
    episode_dir = output_root / "episodes" / args_cli.episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    transitions_jsonl = episode_dir / "transitions.jsonl"
    if transitions_jsonl.exists():
        transitions_jsonl.unlink()

    planned_scene_metadata = _build_scene_metadata(spawn=False)
    bounds = _resolve_rollout_bounds(planned_scene_metadata)
    observed_state = initialize_observed_state(bounds, voxel_size=args_cli.voxel_size)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([5.0, -5.0, 4.0], [0.0, 0.0, 0.5])
    camera, spawned_scene_metadata = design_scene()
    sim.reset()
    print(f"[INFO]: Stage 4A-3 rollout scene setup complete: {args_cli.scene_variant}")

    current_pose = pose_dict(
        [args_cli.start_x, args_cli.start_y, args_cli.camera_height],
        math.radians(float(args_cli.start_yaw_deg)),
    )
    start_pose = {
        "variant": str(args_cli.start_variant),
        "world": pose_dict(current_pose["position"], float(current_pose["yaw_rad"])),
        "source": "scripted_layout_initial_condition",
        "ground_truth_used_for_scoring": False,
    }
    camera_info_saved = False
    camera_info: dict[str, Any] | None = None
    step_records: list[dict[str, Any]] = []
    depth_summaries: list[dict[str, Any]] = []
    pose_visits: dict[tuple[int, int, int, int], int] = {}
    start_key = pose_repeat_key(current_pose, bounds, args_cli.voxel_size, tuple(observed_state.shape))
    pose_visits[start_key] = 1
    repeated_pose_count = 1
    done_reason = "max_steps"

    scene_metadata = {
        **spawned_scene_metadata,
        "stage": "Stage 4A-3.6" if str(args_cli.path_cost_mode) == "astar" else "Stage 4A-3",
        "scene": spawned_scene_metadata.get("scene", str(args_cli.scene_variant)),
        "scene_variant": str(args_cli.scene_variant),
        "scene_seed": int(args_cli.scene_seed),
        "obstacle_jitter_m": float(args_cli.obstacle_jitter_m),
        "start_variant": str(args_cli.start_variant),
        "start_pose": start_pose,
        "camera_height_m": float(args_cli.camera_height),
        "map_bounds": {axis: [float(v) for v in bounds[axis]] for axis in ("x", "y", "z")},
        "voxel_size": float(args_cli.voxel_size),
        "prediction_mode": str(args_cli.prediction_mode),
        "path_cost_mode": str(args_cli.path_cost_mode),
        "candidate_sampling_mode": str(args_cli.candidate_sampling_mode),
        "snap_start_to_traversable": bool(args_cli.snap_start_to_traversable),
        "max_snap_radius_cells": int(args_cli.max_snap_radius_cells),
        "motion_mode": str(args_cli.motion_mode),
        "limitations": ROLLOUT_LIMITATION_NOTES,
        "leakage_checks": leakage_checks(False, a_star_planner=str(args_cli.path_cost_mode) == "astar"),
    }
    save_json(episode_dir / "scene_metadata.json", scene_metadata)

    for step in range(int(args_cli.max_steps)):
        summary_before = observed_summary(observed_state)

        _set_camera_pose(camera, sim, current_pose)
        _step_camera(camera, sim, args_cli.settle_steps)

        if not camera_info_saved:
            camera_info = _camera_info_from_camera(camera)
            save_json(episode_dir / "camera_info.json", camera_info)
            camera_info_saved = True
        assert camera_info is not None

        depth = _extract_depth(camera)
        depth_stats = _depth_stats(depth)
        if args_cli.save_depth:
            depth_path = episode_dir / f"depth_{step:03d}.npy"
            np.save(depth_path, depth)
            depth_stats["depth_file"] = str(depth_path)
        rgb_path = _save_rgb_if_requested(camera, episode_dir / f"rgb_{step:03d}.png")
        if rgb_path is not None:
            depth_stats["rgb_file"] = rgb_path
        depth_summaries.append({"step": int(step), **depth_stats})

        pose_record = {
            "index": int(step),
            "position": [float(v) for v in current_pose["position"]],
            "yaw_rad": float(current_pose["yaw_rad"]),
            "yaw_deg": float(current_pose["yaw_deg"]),
            "target": pose_target(current_pose["position"], current_pose["yaw_rad"]),
            "motion_mode": str(args_cli.motion_mode),
        }
        save_json(episode_dir / f"pose_{step:03d}.json", pose_record)

        observed_state = update_observed_state_from_depth(
            observed_state=observed_state,
            depth=depth,
            camera_pose=current_pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=float(args_cli.voxel_size),
            pixel_stride=int(args_cli.pixel_stride),
        )
        observed_state_path = episode_dir / f"observed_state_step{step:03d}.npy"
        np.save(observed_state_path, observed_state)
        summary_after = observed_summary(observed_state)

        before_expert = observed_state.copy()
        try:
            result = select_sim_expert_action(
                observed_state=observed_state,
                current_pose_world=current_pose,
                bounds=bounds,
                voxel_size=float(args_cli.voxel_size),
                prediction_layer=empty_prediction_layer_for(observed_state),
                prediction_mode=str(args_cli.prediction_mode),
                num_candidates=int(args_cli.num_candidates),
                top_n=int(args_cli.top_n),
                gain_mode=str(args_cli.gain_mode),
                seed=int(args_cli.seed) + int(step),
                max_range_voxels=int(args_cli.max_range_voxels),
                num_yaw=int(args_cli.num_yaw),
                num_pitch=int(args_cli.num_pitch),
                fov_yaw_deg=float(args_cli.fov_yaw_deg),
                fov_pitch_deg=float(args_cli.fov_pitch_deg),
                path_cost_mode=str(args_cli.path_cost_mode),
                robot_height_m=float(args_cli.camera_height),
                candidate_sampling_mode=str(args_cli.candidate_sampling_mode),
                snap_start_to_traversable=bool(args_cli.snap_start_to_traversable),
                max_snap_radius_cells=int(args_cli.max_snap_radius_cells),
            )
        except ValueError as exc:
            error_text = str(exc)
            done_reason = "no_reachable_free_component" if "no_reachable_free_component" in error_text else "no_valid_candidate"
            step_records.append(
                {
                    "step": int(step),
                    "done_reason": done_reason,
                    "error": error_text,
                    "no_valid_candidate_reason": done_reason,
                    "observed_ratio_before": float(summary_before["observed_ratio"]),
                    "observed_ratio_after": float(summary_after["observed_ratio"]),
                }
            )
            print(f"[WARN]: stopping at step {step}: {error_text}")
            break

        prediction_wrote_observed_map = not bool(np.array_equal(before_expert, observed_state))
        best = result["best_candidate"]
        next_pose = compute_next_pose_from_candidate(
            candidate=best,
            bounds=bounds,
            voxel_size=float(args_cli.voxel_size),
            observed_shape=tuple(int(v) for v in observed_state.shape),
            motion_mode=str(args_cli.motion_mode),
            camera_height=float(args_cli.camera_height),
        )

        done = False
        done_reason = ""
        next_key = pose_repeat_key(next_pose["world"], bounds, args_cli.voxel_size, tuple(observed_state.shape))
        pose_visits[next_key] = pose_visits.get(next_key, 0) + 1
        repeated_pose_count = max(repeated_pose_count, pose_visits[next_key])

        if float(summary_after["observed_ratio"]) >= float(args_cli.coverage_stop):
            done = True
            done_reason = "coverage_stop"
        elif float(best.final_score) <= float(args_cli.min_score):
            done = True
            done_reason = "best_score_below_min_score"
        elif pose_visits[next_key] >= int(args_cli.repeat_pose_patience):
            done = True
            done_reason = "repeat_pose_patience"
        elif step == int(args_cli.max_steps) - 1:
            done = True
            done_reason = "max_steps"

        transition = build_transition(
            episode_id=str(args_cli.episode_id),
            step=int(step),
            result=result,
            current_pose_world=current_pose,
            next_pose=next_pose,
            summary_before=summary_before,
            summary_after=summary_after,
            bounds=bounds,
            voxel_size=float(args_cli.voxel_size),
            prediction_mode=str(args_cli.prediction_mode),
            gain_mode=str(args_cli.gain_mode),
            path_cost_mode=str(args_cli.path_cost_mode),
            motion_mode=str(args_cli.motion_mode),
            done=done,
            done_reason=done_reason,
            prediction_wrote_observed_map=prediction_wrote_observed_map,
        )
        transition["reachable_candidates"] = int(result["diagnostics"].get("reachable_candidates") or 0)
        transition["unreachable_candidates"] = int(result["diagnostics"].get("unreachable_candidates") or 0)
        paths = write_transition_records(episode_dir, transition)
        step_record = {
            "step": int(step),
            "observed_state_file": str(observed_state_path),
            "transition_npz": paths["npz"],
            "observed_ratio_before": float(summary_before["observed_ratio"]),
            "observed_ratio_after": float(summary_after["observed_ratio"]),
            "delta_observed_ratio": float(transition["delta_observed_ratio"]),
            "frontier_count": int(transition["frontier_count"]),
            "candidate_count": int(transition["candidate_count"]),
            "best_score": float(transition["best_score"]),
            "best_gain_exp": float(transition["best_gain_exp"]),
            "best_gain_sc": float(transition["best_gain_sc"]),
            "best_gain_hybrid": float(transition["best_gain_hybrid"]),
            "best_path_cost": float(transition["best_path_cost"]),
            "best_astar_path_length_m": float(transition["best_astar_path_length_m"]),
            "best_astar_num_expanded": int(transition["best_astar_num_expanded"]),
            "reachable_candidates": int(transition["reachable_candidates"]),
            "unreachable_candidates": int(transition["unreachable_candidates"]),
            "reachable_component_count": int(transition.get("reachable_component_count") or 0),
            "reachable_frontier_adjacent_count": int(transition.get("reachable_frontier_adjacent_count") or 0),
            "candidate_source": str(transition.get("candidate_source", "")),
            "snapped_current": bool(transition.get("snapped_current", False)),
            "snapped_current_xy": transition.get("snapped_current_xy"),
            "no_valid_candidate_reason": str(transition.get("no_valid_candidate_reason", "")),
            "current_pose_world": [float(v) for v in transition["current_pose_world"]],
            "selected_next_pose_world": [float(v) for v in transition["selected_next_pose_world"]],
            "done": bool(done),
            "done_reason": str(done_reason),
        }
        step_records.append(step_record)

        if args_cli.print_steps:
            print(
                "[STEP "
                f"{step:03d}] ratio {summary_before['observed_ratio']:.6f} -> {summary_after['observed_ratio']:.6f} "
                f"frontiers={transition['frontier_count']} candidates={transition['candidate_count']} "
                f"score={transition['best_score']:.6f} gain_exp={transition['best_gain_exp']:.1f} "
                f"gain_sc={transition['best_gain_sc']:.1f} path_cost={transition['best_path_cost']:.3f} "
                f"reachable={step_record['reachable_candidates']} "
                f"component={step_record['reachable_component_count']} "
                f"source={step_record['candidate_source']} next={step_record['selected_next_pose_world']} "
                f"done={done} reason={done_reason or 'continue'}"
            )

        current_pose = next_pose["world"]
        if done:
            break

    np.save(episode_dir / "observed_state_final.npy", observed_state)
    final_summary = observed_summary(observed_state)
    transitions = load_jsonl(transitions_jsonl)
    steps_completed = len(transitions)
    terminal_done_reason = str(done_reason or "")
    if transitions:
        if terminal_done_reason == "no_valid_candidate":
            final_pose = current_pose["position"]
            final_yaw = current_pose["yaw_rad"]
        else:
            final_pose = transitions[-1]["selected_next_pose_world"]
            final_yaw = transitions[-1]["selected_next_yaw"]
        transition_done_reason = str(transitions[-1].get("done_reason", ""))
        if transition_done_reason:
            terminal_done_reason = transition_done_reason
        start_ratio = transitions[0]["observed_ratio_before"]
        end_ratio = float(final_summary["observed_ratio"])
    else:
        final_pose = current_pose["position"]
        final_yaw = current_pose["yaw_rad"]
        start_ratio = 0.0
        end_ratio = float(final_summary["observed_ratio"])

    output_paths = {
        "episode_dir": str(episode_dir),
        "transitions": str(transitions_jsonl),
        "summary": str(episode_dir / "episode_summary.json"),
        "final_observed_map": str(episode_dir / "observed_state_final.npy"),
    }

    if args_cli.save_viz and transitions:
        from visualize_sim_rollout import save_rollout_visualizations

        viz_paths = save_rollout_visualizations(episode_dir, save_steps=True)
        output_paths.update({k: v for k, v in viz_paths.items() if isinstance(v, str)})

    metric_summary = _transition_metric_summary(transitions)
    episode_summary = {
        "stage": "Stage 4A-3.6" if str(args_cli.path_cost_mode) == "astar" else "Stage 4A-3",
        "episode_id": str(args_cli.episode_id),
        "episode_dir": str(episode_dir),
        "scene": spawned_scene_metadata.get("scene", str(args_cli.scene_variant)),
        "scene_variant": str(args_cli.scene_variant),
        "scene_seed": int(args_cli.scene_seed),
        "obstacle_jitter_m": float(args_cli.obstacle_jitter_m),
        "start_variant": str(args_cli.start_variant),
        "start_pose": start_pose,
        "prediction_mode": str(args_cli.prediction_mode),
        "prediction_layer": "EmptyPredictionLayer",
        "gain_mode": str(args_cli.gain_mode),
        "path_cost_mode": str(args_cli.path_cost_mode),
        "candidate_sampling_mode": str(args_cli.candidate_sampling_mode),
        "snap_start_to_traversable": bool(args_cli.snap_start_to_traversable),
        "max_snap_radius_cells": int(args_cli.max_snap_radius_cells),
        "motion_mode": str(args_cli.motion_mode),
        "max_steps": int(args_cli.max_steps),
        "steps_completed": int(steps_completed),
        "done_reason": str(terminal_done_reason or "no_transition"),
        "coverage_stop": float(args_cli.coverage_stop),
        "min_score": float(args_cli.min_score),
        "repeat_pose_patience": int(args_cli.repeat_pose_patience),
        "repeated_pose_count": int(repeated_pose_count),
        "voxel_size": float(args_cli.voxel_size),
        "pixel_stride": int(args_cli.pixel_stride),
        "map_bounds": {axis: [float(v) for v in bounds[axis]] for axis in ("x", "y", "z")},
        "observed_ratio_start": float(start_ratio),
        "observed_ratio_end": float(end_ratio),
        "total_delta_observed_ratio": float(end_ratio - start_ratio),
        "final_counts": final_summary,
        "final_pose": [float(v) for v in final_pose],
        "final_yaw": float(final_yaw),
        "metrics": metric_summary,
        "step_records": step_records,
        "depth_summaries": depth_summaries,
        "output_paths": output_paths,
        "leakage_checks": leakage_checks(
            any(bool(t["leakage_checks"]["prediction_wrote_observed_map"]) for t in transitions),
            a_star_planner=str(args_cli.path_cost_mode) == "astar",
        ),
        "limitations": ROLLOUT_LIMITATION_NOTES,
    }
    save_json(episode_dir / "episode_summary.json", episode_summary)

    manifest_record = {
        "episode_id": str(args_cli.episode_id),
        "episode_dir": str(episode_dir),
        "status": "ok",
        "steps_completed": int(steps_completed),
        "done_reason": str(episode_summary["done_reason"]),
        "scene_seed": int(args_cli.scene_seed),
        "start_variant": str(args_cli.start_variant),
        "observed_ratio_start": float(episode_summary["observed_ratio_start"]),
        "observed_ratio_end": float(episode_summary["observed_ratio_end"]),
        "total_delta_observed_ratio": float(episode_summary["total_delta_observed_ratio"]),
        "no_valid_candidate_steps": metric_summary.get("no_valid_candidate_steps", []),
        "prediction_mode": str(args_cli.prediction_mode),
        "path_cost_mode": str(args_cli.path_cost_mode),
        "candidate_sampling_mode": str(args_cli.candidate_sampling_mode),
        "motion_mode": str(args_cli.motion_mode),
        "summary": str(episode_dir / "episode_summary.json"),
        "leakage_checks": leakage_checks(False, a_star_planner=str(args_cli.path_cost_mode) == "astar"),
    }
    if not args_cli.no_manifest:
        append_jsonl(output_root / "manifest.jsonl", manifest_record)
    return episode_summary


def main() -> None:
    summary = run_rollout()
    print(
        "Stage 4A-3.6 simulator expert rollout complete."
        if summary.get("path_cost_mode") == "astar"
        else "Stage 4A-3 simulator expert rollout complete."
    )
    print(f"episode_id: {summary['episode_id']}")
    print(f"episode_dir: {summary['episode_dir']}")
    print(f"scene_variant: {summary['scene_variant']}")
    print(f"path_cost_mode: {summary['path_cost_mode']}")
    print(f"candidate_sampling_mode: {summary.get('candidate_sampling_mode')}")
    print(f"steps_completed: {summary['steps_completed']}")
    print(f"done_reason: {summary['done_reason']}")
    print(
        "observed_ratio: "
        f"{summary['observed_ratio_start']:.6f} -> {summary['observed_ratio_end']:.6f} "
        f"delta={summary['total_delta_observed_ratio']:.6f}"
    )
    print(f"final_counts: {summary['final_counts']}")
    for key, path in summary["output_paths"].items():
        print(f"output_{key}: {path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
