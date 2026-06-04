#!/usr/bin/env python3
"""Stage 4A-6.5p map_predict + source-protected tree two-frame smoke.

This runner starts Isaac once, captures exactly two RGB/depth frames in the
deterministic medium scene, updates measured-only observed maps, runs SSCNet
map_predict once per frame with one persistent model instance, and evaluates a
measured-only baseline tree plus a read-only SC/hybrid tree per frame. It
executes exactly one selected-child move between frames. Prediction is used
only for information gain and never for observed_state writeback,
traversability, collision, or ray blocking.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


BASE_PROFILE_NAME = "source_like_crop_min_length_0p25"
PROFILE_NAME = BASE_PROFILE_NAME
DEFAULT_SELECTED_CASE = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65c_decoupled_one_step_smoke/selected_case.json"
)
DEFAULT_REFERENCE_ONE_STEP_SC_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke"
)
DEFAULT_REFERENCE_NO_PRED_TWO_FRAME_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65n_two_frame_tree_smoke"
)
DEFAULT_EPISODE_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_medium_rollout_sc_pred_alignment_fixed_smoke/episodes/"
    "medium_three_rooms_seed0_start_room_a_sc_pred_alignment_fixed_000"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke"
)
DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)


def profile_name_for_seed(seed: int) -> str:
    return BASE_PROFILE_NAME if int(seed) == 0 else f"{BASE_PROFILE_NAME}_seed{int(seed)}"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--selected_case_json", default=DEFAULT_SELECTED_CASE)
parser.add_argument("--reference_one_step_sc_dir", default=DEFAULT_REFERENCE_ONE_STEP_SC_DIR)
parser.add_argument("--reference_no_pred_two_frame_dir", default=DEFAULT_REFERENCE_NO_PRED_TWO_FRAME_DIR)
parser.add_argument("--episode_dir", default=DEFAULT_EPISODE_DIR)
parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
parser.add_argument("--scene_variant", default="medium_three_rooms")
parser.add_argument("--scene_seed", type=int, default=0)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=5.0)
parser.add_argument("--settle_steps", type=int, default=12)
parser.add_argument("--pixel_stride", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num_nodes", type=int, default=256)
parser.add_argument("--max_extension_m", type=float, default=0.5)
parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
parser.add_argument("--prediction_mode", choices=["sim_dynamic", "sim_npz"], default="sim_dynamic")
parser.add_argument("--prediction_npz", default="")
parser.add_argument("--gain_mode", choices=["hybrid", "sc"], default="hybrid")
parser.add_argument("--baseline_gain_mode", choices=["exp"], default="exp")
parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
parser.add_argument("--v_max", type=float, default=1.0)
parser.add_argument("--robot_radius_m", type=float, default=0.2)
parser.add_argument("--voxel_size", type=float, default=0.1)
parser.add_argument("--raycast_stride", type=int, default=2)
parser.add_argument("--num_yaw_samples", type=int, default=8)
parser.add_argument("--max_ray_length_m", type=float, default=4.8)
parser.add_argument("--short_edge_policy", choices=["crop"], default="crop")
parser.add_argument("--crop_min_length_m", type=float, default=0.25)
parser.add_argument("--min_edge_length_m", type=float, default=0.0)
parser.add_argument("--min_root_child_length_m", type=float, default=0.0)
parser.add_argument("--min_root_distance_m", type=float, default=0.0)
parser.add_argument("--density_radius_m", type=float, default=0.0)
parser.add_argument("--max_nodes_per_density_radius", type=int, default=0)
parser.add_argument("--tau", type=float, default=0.1)
parser.add_argument(
    "--sc_gain_formula",
    choices=[
        "raw_count",
        "weight_0p5",
        "weight_1p0",
        "cap25",
        "cap50",
        "confidence_weighted",
        "occupied_only",
        "confidence_weighted_cap25",
    ],
    default="raw_count",
)
parser.add_argument("--alignment_convention", choices=["code_consistent_v1"], default="code_consistent_v1")
parser.add_argument("--torch_num_threads", type=int, default=8)
parser.add_argument("--variant_name", default="map_predict_source_like_crop_min_length_0p25_two_frame")
parser.add_argument("--save_viz", action="store_true")
parser.add_argument("--save_probs", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "headless"):
    args_cli.headless = True
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

SSC_EXPLORATION_DIR = Path("/home/ubuntu22/sc_explorer_ws/ssc_exploration")
SSC_NETWORK_DIR = SSC_EXPLORATION_DIR / "ssc_network"
for _path in (SSC_EXPLORATION_DIR, SSC_NETWORK_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from depth_to_voxel import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    create_observed_grid,
    normalize_map_bounds,
    summarize_observed_grid,
    update_observed_state_from_depth,
)
from isaac_map_predictor import IsaacMapPredictor
from offline_mini_rrt_tree import run as run_mini_rrt
from offline_mini_rrt_tree import sha256_file, to_jsonable
from scene_factory import build_medium_complex_scene
from sim_prediction_layer import SimPredictionLayer


DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")
CAMERA_HEIGHT_M = 1.2
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def _pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def _scene_variant_name(scene_variant: str) -> str:
    if scene_variant in {"medium_three_rooms", "three_rooms"}:
        return "three_rooms"
    raise ValueError(f"unsupported scene_variant: {scene_variant}")


def _episode_bounds(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    episode_summary = Path(args.episode_dir).resolve() / "episode_summary.json"
    if episode_summary.is_file():
        data = _load_json(episode_summary)
        raw = data.get("map_bounds") or data.get("bounds")
        if raw is not None:
            return normalize_map_bounds(raw)
    return normalize_map_bounds({"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]})


def _initial_pose(args: argparse.Namespace) -> dict[str, Any]:
    episode_dir = Path(args.episode_dir).resolve()
    pose_path = episode_dir / "pose_001.json"
    source: Path | None = None
    if pose_path.is_file():
        source = pose_path
        raw = _load_json(pose_path)
        position = [float(v) for v in raw["position"]]
        yaw_rad = float(raw.get("yaw_rad", math.radians(float(raw.get("yaw_deg", 0.0)))))
    else:
        position = [-4.65, -4.65, CAMERA_HEIGHT_M]
        yaw_rad = 0.38710316317995463
    return {
        "index": 1,
        "frame": 1,
        "position": position,
        "yaw_rad": yaw_rad,
        "yaw_deg": float(math.degrees(yaw_rad)),
        "target": _pose_target(position, yaw_rad),
        "source_pose_file": str(source) if source is not None else None,
        "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
    }


def _pose_from_selected_child(selected: dict[str, Any]) -> dict[str, Any]:
    world = selected.get("end_world")
    if not world or len(world) < 2:
        raise ValueError(f"selected child has no usable end_world: {selected}")
    yaw_rad = float(selected.get("yaw", 0.0))
    position = [float(world[0]), float(world[1]), CAMERA_HEIGHT_M]
    return {
        "index": 2,
        "frame": 2,
        "position": position,
        "yaw_rad": yaw_rad,
        "yaw_deg": float(math.degrees(yaw_rad)),
        "target": _pose_target(position, yaw_rad),
        "source": "frame001_sc_selected_child",
        "selected_child_segment_id": selected.get("segment_id"),
        "selected_child_grid": selected.get("end_grid"),
        "selected_child_world": selected.get("end_world"),
        "motion_mode": "planar_teleport_once_to_sc_selected_child_xy_fixed_camera_height",
        "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
    }


def _make_camera(args: argparse.Namespace) -> Camera:
    sim_utils.create_prim("/World/CameraRig", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CameraRig/CameraSensor",
        update_period=0.0,
        height=int(args.camera_height),
        width=int(args.camera_width),
        data_types=["rgb", DEPTH_KEY],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, float(args.max_depth)),
        ),
    )
    return Camera(cfg=camera_cfg)


def _add_lighting() -> None:
    dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.82, 0.84, 0.80))
    dome_cfg.func("/World/Light", dome_cfg)


def _set_camera_pose(camera: Camera, sim: sim_utils.SimulationContext, pose: dict[str, Any]) -> None:
    position = [float(v) for v in pose["position"]]
    target = [float(v) for v in pose["target"]]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )


def _settle(camera: Camera, sim: sim_utils.SimulationContext, steps: int) -> None:
    for _ in range(max(int(steps), 1)):
        sim.step()
        camera.update(dt=sim.get_physics_dt())


def _normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _extract_rgb(camera: Camera) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEY_CANDIDATES:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        source = tensor[0].detach().cpu().numpy()
        if source.ndim != 3 or source.shape[-1] not in (3, 4):
            raise ValueError(f"expected RGB/RGBA image, got {source.shape}")
        rgb = _normalize_rgb(source)
        stats = {
            "key": key,
            "shape": [int(v) for v in rgb.shape],
            "dtype_after_save_conversion": str(rgb.dtype),
            "min": int(rgb.min()) if rgb.size else None,
            "max": int(rgb.max()) if rgb.size else None,
            "mean": float(rgb.mean()) if rgb.size else None,
            "std": float(rgb.std()) if rgb.size else None,
        }
        if rgb.size == 0 or stats["max"] is None or int(stats["max"]) <= 2 or float(stats["std"] or 0.0) < 1.0:
            raise ValueError(f"RGB image appears blank or nearly uniform: {stats}")
        return rgb, key, stats
    raise KeyError(f"camera output missing RGB/RGBA. Keys: {list(camera.data.output.keys())}")


def _extract_depth(camera: Camera) -> tuple[np.ndarray, dict[str, Any]]:
    tensor = camera.data.output.get(DEPTH_KEY)
    if tensor is None:
        raise KeyError(f"camera output missing {DEPTH_KEY}. Keys: {list(camera.data.output.keys())}")
    depth = tensor[0].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        raise ValueError("captured depth has no finite positive values")
    return depth, {
        "shape": [int(v) for v in depth.shape],
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()),
        "max": float(positive.max()),
        "mean": float(positive.mean()),
    }


def _camera_info(camera: Camera, args: argparse.Namespace) -> dict[str, Any]:
    intrinsic_matrix = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
    return {
        "sensor_api_depth_key": DEPTH_KEY,
        "depth_units": "meters",
        "width": int(args.camera_width),
        "height": int(args.camera_height),
        "max_depth": float(args.max_depth),
        "near_depth": 0.05,
        "horizontal_fov_deg": 90.0,
        "intrinsic_matrix": intrinsic_matrix.tolist(),
        "fx": float(intrinsic_matrix[0, 0]),
        "fy": float(intrinsic_matrix[1, 1]),
        "cx": float(intrinsic_matrix[0, 2]),
        "cy": float(intrinsic_matrix[1, 2]),
    }


def _save_depth_png(path: Path, depth: np.ndarray, title: str) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError("cannot save depth PNG: no positive finite depth")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _capture_frame(
    args: argparse.Namespace,
    output_dir: Path,
    camera: Camera,
    sim: sim_utils.SimulationContext,
    pose: dict[str, Any],
    frame_index: int,
) -> dict[str, Any]:
    prefix = f"frame{frame_index:03d}"
    _set_camera_pose(camera, sim, pose)
    _settle(camera, sim, int(args.settle_steps))
    depth, depth_stats = _extract_depth(camera)
    rgb, rgb_key, rgb_stats = _extract_rgb(camera)
    camera_info = _camera_info(camera, args)

    pose_path = output_dir / f"{prefix}_pose.json"
    depth_path = output_dir / f"{prefix}_depth.npy"
    depth_png_path = output_dir / f"{prefix}_depth.png"
    rgb_path = output_dir / f"{prefix}_rgb.png"
    camera_info_path = output_dir / f"{prefix}_camera_info.json"

    _save_json(pose_path, pose)
    np.save(depth_path, depth)
    _save_depth_png(depth_png_path, depth, f"Isaac map_predict tree depth {frame_index:03d}")
    Image.fromarray(rgb).save(rgb_path)
    _save_json(camera_info_path, camera_info)

    return {
        "frame_index": int(frame_index),
        "prefix": prefix,
        "depth": depth,
        "depth_path": str(depth_path),
        "depth_png_path": str(depth_png_path),
        "rgb_path": str(rgb_path),
        "pose": pose,
        "pose_path": str(pose_path),
        "camera_info": camera_info,
        "camera_info_path": str(camera_info_path),
        "depth_stats": depth_stats,
        "rgb_stats": rgb_stats,
        "rgb_key_used": rgb_key,
        "camera_output_keys": list(camera.data.output.keys()),
    }


def _start_scene(args: argparse.Namespace) -> tuple[sim_utils.SimulationContext, Camera, dict[str, Any]]:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([7.0, -8.0, 6.0], [0.0, -0.2, 0.8])
    _add_lighting()
    scene_metadata = build_medium_complex_scene(
        seed=int(args.scene_seed),
        variant=_scene_variant_name(str(args.scene_variant)),
        obstacle_jitter_m=0.0,
        spawn=True,
        sim_utils_module=sim_utils,
    )
    camera = _make_camera(args)
    sim.reset()
    return sim, camera, scene_metadata


def _update_observed_state(
    args: argparse.Namespace,
    output_dir: Path,
    capture: dict[str, Any],
    prior_path: Path,
    frame_index: int,
) -> dict[str, Any]:
    bounds = _episode_bounds(args)
    fallback_empty = False
    if prior_path.is_file():
        prior = np.load(prior_path)
    else:
        fallback_empty = True
        prior = create_observed_grid(bounds, voxel_size=float(args.voxel_size))
        prior_path = output_dir / f"observed_state_frame{frame_index - 1:03d}_empty_fallback.npy"
        np.save(prior_path, prior)

    prior_hash_before = sha256_file(prior_path)
    prior_summary_before = summarize_observed_grid(prior)
    updated = np.array(prior, dtype=np.int8, copy=True)
    updated = update_observed_state_from_depth(
        observed_state=updated,
        depth=np.asarray(capture["depth"], dtype=np.float32),
        camera_pose=capture["pose"],
        camera_info=capture["camera_info"],
        bounds=bounds,
        voxel_size=float(args.voxel_size),
        pixel_stride=int(args.pixel_stride),
    )
    prior_hash_after = sha256_file(prior_path)

    output_observed = output_dir / f"observed_state_frame{frame_index:03d}.npy"
    np.save(output_observed, updated)
    updated_summary = summarize_observed_grid(updated)
    delta_observed_count = int(updated_summary["observed_count"] - prior_summary_before["observed_count"])
    delta_observed_ratio = float(updated_summary["observed_ratio"] - prior_summary_before["observed_ratio"])
    payload = {
        "frame_index": int(frame_index),
        "prior_observed_state": str(prior_path),
        "prior_sha256_before": prior_hash_before,
        "prior_sha256_after": prior_hash_after,
        "prior_hash_unchanged": prior_hash_before == prior_hash_after,
        "fallback_empty_prior_used": fallback_empty,
        "new_observed_state": str(output_observed),
        "new_observed_state_sha256": sha256_file(output_observed),
        "prior_summary_before": prior_summary_before,
        "updated_summary": updated_summary,
        "delta_observed_count": delta_observed_count,
        "delta_observed_ratio": delta_observed_ratio,
        "bounds": {axis: [float(bounds[axis][0]), float(bounds[axis][1])] for axis in ("x", "y", "z")},
        "voxel_size": float(args.voxel_size),
        "pixel_stride": int(args.pixel_stride),
        "capture_depth_file": capture["depth_path"],
        "capture_pose_file": capture["pose_path"],
        "capture_camera_info_file": capture["camera_info_path"],
        "measured_only": True,
        "prediction_used": False,
        "map_predict_used": False,
        "target_lr_target_hr_ground_truth_used": False,
        "prior_observed_state_modified": prior_hash_before != prior_hash_after,
    }
    _save_json(output_dir / f"frame{frame_index:03d}_observed_summary.json", payload)
    return payload


def _tree_args(
    args: argparse.Namespace,
    tree_dir: Path,
    observed_path: str,
    capture: dict[str, Any],
    prediction_npz: str,
    gain_mode: str,
    variant_suffix: str,
) -> argparse.Namespace:
    episode_dir = Path(args.episode_dir).resolve()
    return argparse.Namespace(
        case_json=str(Path(args.selected_case_json).resolve()),
        episode_dir=str(episode_dir),
        observed_state=str(observed_path),
        pose_json=str(capture["pose_path"]),
        camera_info=str(capture["camera_info_path"]),
        episode_summary=str(episode_dir / "episode_summary.json"),
        prediction_npz=str(prediction_npz or ""),
        output_dir=str(tree_dir),
        seed=int(args.seed),
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        gain_mode=str(gain_mode),
        path_cost_mode=str(args.path_cost_mode),
        v_max=float(args.v_max),
        yaw_rate=1.0,
        robot_radius_m=float(args.robot_radius_m),
        voxel_size=float(args.voxel_size),
        raycast_stride=int(args.raycast_stride),
        num_yaw_samples=int(args.num_yaw_samples),
        max_ray_length_m=float(args.max_ray_length_m),
        sc_gain_formula=str(args.sc_gain_formula),
        tau=float(args.tau),
        save_viz=bool(args.save_viz),
        profile=True,
        min_edge_length_m=float(args.min_edge_length_m),
        min_root_child_length_m=float(args.min_root_child_length_m),
        min_root_distance_m=float(args.min_root_distance_m),
        crop_min_length_m=float(args.crop_min_length_m),
        short_edge_policy=str(args.short_edge_policy),
        density_radius_m=float(args.density_radius_m),
        max_nodes_per_density_radius=int(args.max_nodes_per_density_radius),
        variant_name=f"{args.variant_name}_{variant_suffix}",
    )


def _first_float(values: list[Any]) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _decision_parts(summary: dict[str, Any]) -> dict[str, Any]:
    decision = summary.get("decision", {})
    comparison = summary.get("comparison", {})
    mini = comparison.get("mini_rrt", {})
    selected = decision.get("selected_child") or mini.get("selected_child") or {}
    best = decision.get("best_descendant") or mini.get("best_descendant") or {}
    selected_distance = _first_float(
        [
            mini.get("selected_child_distance_from_root_m"),
            selected.get("accumulated_cost") if selected.get("parent_id") == "root" else None,
            selected.get("segment_length_m") if selected.get("parent_id") == "root" else None,
        ]
    )
    best_distance = _first_float([mini.get("best_descendant_distance_from_root_m")])
    return {
        "selected": selected,
        "best": best,
        "selected_distance_m": selected_distance,
        "best_distance_m": best_distance,
        "value": selected.get("value"),
        "accumulated_gain": best.get("accumulated_gain"),
        "accumulated_cost": best.get("accumulated_cost"),
        "accepted_nodes": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
        "rejected_samples": summary.get("tree", {}).get("rejected_samples"),
        "built_successfully": bool(summary.get("tree", {}).get("built_successfully")),
        "root": summary.get("root", {}),
        "raw_decision": decision,
    }


def _same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def _close(a: Any, b: Any, tol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _euclidean(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        av = [float(v) for v in a]
        bv = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if len(av) != len(bv):
        return None
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv))))


def _load_segments(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _min_mean_max(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(finite.min()), "mean": float(finite.mean()), "max": float(finite.max())}


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xa) & np.isfinite(ya)
    if int(np.count_nonzero(mask)) < 2:
        return None
    xa = xa[mask]
    ya = ya[mask]
    if float(np.std(xa)) <= 1.0e-12 or float(np.std(ya)) <= 1.0e-12:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _tree_gain_stats(tree_dir: Path) -> dict[str, Any]:
    segments = _load_segments(tree_dir / "mini_rrt_tree_segments.jsonl")
    non_root = [row for row in segments if row.get("segment_id") != "root"]
    gain_exp = [float(row.get("gain_exp", 0.0) or 0.0) for row in non_root]
    gain_sc = [float(row.get("gain_sc", 0.0) or 0.0) for row in non_root]
    gain_hybrid = [float(row.get("gain_hybrid", 0.0) or 0.0) for row in non_root]
    gain_occ = [float(row.get("gain_occ", 0.0) or 0.0) for row in non_root]
    gain_conf = [float(row.get("gain_conf", 0.0) or 0.0) for row in non_root]
    positive_sc = [row for row in non_root if float(row.get("gain_sc", 0.0) or 0.0) > 0.0]
    return {
        "node_count_excluding_root": int(len(non_root)),
        "nodes_with_gain_sc_positive": int(len(positive_sc)),
        "gain_sc_density": float(len(positive_sc) / len(non_root)) if non_root else 0.0,
        "gain_sc_min_mean_max": _min_mean_max(gain_sc),
        "gain_exp_min_mean_max": _min_mean_max(gain_exp),
        "gain_hybrid_min_mean_max": _min_mean_max(gain_hybrid),
        "gain_occ_min_mean_max": _min_mean_max(gain_occ),
        "gain_conf_min_mean_max": _min_mean_max(gain_conf),
        "gain_exp_gain_sc_pearson": _pearson(gain_exp, gain_sc),
        "positive_gain_sc_segment_ids_sample": [str(row.get("segment_id")) for row in positive_sc[:16]],
    }


def _path_sums(tree_dir: Path, best_segment_id: str | None) -> dict[str, Any]:
    segments = _load_segments(tree_dir / "mini_rrt_tree_segments.jsonl")
    by_id = {str(row.get("segment_id")): row for row in segments}
    if not best_segment_id or best_segment_id not in by_id:
        return {
            "path_segment_ids": [],
            "gain_exp": None,
            "gain_sc": None,
            "gain_hybrid": None,
            "cost": None,
        }
    path: list[str] = []
    current: str | None = str(best_segment_id)
    while current and current in by_id:
        path.append(current)
        current = by_id[current].get("parent_id")
    path.reverse()
    non_root = [seg_id for seg_id in path if seg_id != "root"]
    return {
        "path_segment_ids": non_root,
        "gain_exp": float(sum(float(by_id[seg_id].get("gain_exp", 0.0) or 0.0) for seg_id in non_root)),
        "gain_sc": float(sum(float(by_id[seg_id].get("gain_sc", 0.0) or 0.0) for seg_id in non_root)),
        "gain_hybrid": float(sum(float(by_id[seg_id].get("gain_hybrid", 0.0) or 0.0) for seg_id in non_root)),
        "cost": float(sum(float(by_id[seg_id].get("cost", 0.0) or 0.0) for seg_id in non_root)),
    }


def _prediction_stats(prediction_npz: str, observed_state: np.ndarray, tau: float) -> dict[str, Any]:
    layer = SimPredictionLayer.from_npz(prediction_npz)
    with np.load(prediction_npz, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occupied_prob = np.asarray(data["global_occupied_prob"], dtype=np.float32)
        alignment_convention = str(np.asarray(data["alignment_convention"]).item())
    valid_tau = valid & (confidence >= float(tau))
    predicted_unmeasured = valid_tau & (observed_state == UNKNOWN)
    predicted_occupied = valid_tau & (occupied_prob >= 0.5)
    return {
        "prediction_npz": str(prediction_npz),
        "shape": [int(v) for v in layer.shape()],
        "observed_state_shape": [int(v) for v in observed_state.shape],
        "shape_aligned_to_observed_state": tuple(layer.shape()) == tuple(observed_state.shape),
        "alignment_convention": alignment_convention,
        "tau": float(tau),
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "prediction_valid_tau_count": int(np.count_nonzero(valid_tau)),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "predicted_occupied_count": int(np.count_nonzero(predicted_occupied)),
        "confidence_valid_tau_min_mean_max": _min_mean_max(confidence[valid_tau].astype(float).tolist()),
        "occupied_prob_valid_tau_min_mean_max": _min_mean_max(occupied_prob[valid_tau].astype(float).tolist()),
    }


def _copy_alias(src: Path, dst: Path) -> str:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(to_jsonable(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key, "")
                    for key in fields
                }
            )


def _write_node_gain_breakdown(src_jsonl: Path, dst_csv: Path) -> str:
    rows = []
    for row in _load_segments(src_jsonl):
        if row.get("segment_id") == "root":
            continue
        rows.append(
            {
                "segment_id": row.get("segment_id"),
                "parent_id": row.get("parent_id"),
                "depth": row.get("depth"),
                "end_grid": row.get("end_grid"),
                "end_world": row.get("end_world"),
                "gain_exp": row.get("gain_exp"),
                "gain_sc": row.get("gain_sc"),
                "gain_hybrid": row.get("gain_hybrid"),
                "gain_occ": row.get("gain_occ"),
                "gain_conf": row.get("gain_conf"),
                "cost": row.get("cost"),
                "value": row.get("value"),
                "accumulated_gain": row.get("accumulated_gain"),
                "accumulated_cost": row.get("accumulated_cost"),
                "best_descendant_id": row.get("best_descendant_id"),
            }
        )
    _write_csv(
        dst_csv,
        rows,
        [
            "segment_id",
            "parent_id",
            "depth",
            "end_grid",
            "end_world",
            "gain_exp",
            "gain_sc",
            "gain_hybrid",
            "gain_occ",
            "gain_conf",
            "cost",
            "value",
            "accumulated_gain",
            "accumulated_cost",
            "best_descendant_id",
        ],
    )
    return str(dst_csv)


def _alias_tree_outputs(output_dir: Path, tree_dir: Path, prefix: str) -> dict[str, str]:
    aliases = {
        f"{prefix}_tree_segments.jsonl": _copy_alias(
            tree_dir / "mini_rrt_tree_segments.jsonl", output_dir / f"{prefix}_tree_segments.jsonl"
        ),
        f"{prefix}_gain_cost_value_table.csv": _copy_alias(
            tree_dir / "gain_cost_value_table.csv", output_dir / f"{prefix}_gain_cost_value_table.csv"
        ),
        f"{prefix}_node_gain_breakdown.csv": _write_node_gain_breakdown(
            tree_dir / "mini_rrt_tree_segments.jsonl", output_dir / f"{prefix}_node_gain_breakdown.csv"
        ),
    }
    topdown = tree_dir / "mini_rrt_tree_topdown.png"
    if topdown.is_file():
        aliases[f"{prefix}_tree_topdown.png"] = _copy_alias(topdown, output_dir / f"{prefix}_tree_topdown.png")
    return aliases


def _check_hybrid_identity(csv_path: Path) -> dict[str, Any]:
    if not csv_path.is_file():
        return {"passed": False, "reason": f"missing {csv_path}"}
    checked = 0
    max_abs_error = 0.0
    failures: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("segment_id") == "root":
                continue
            gain_exp = float(row.get("gain_exp") or 0.0)
            gain_sc = float(row.get("gain_sc") or 0.0)
            gain_hybrid = float(row.get("gain_hybrid") or 0.0)
            error = abs(gain_hybrid - (gain_exp + gain_sc))
            max_abs_error = max(max_abs_error, error)
            checked += 1
            if error > 1.0e-5:
                failures.append({"segment_id": row.get("segment_id"), "error": error})
    return {
        "passed": len(failures) == 0 and checked > 0,
        "checked_segments": checked,
        "max_abs_error": float(max_abs_error),
        "failures_sample": failures[:8],
    }


def _make_source_protection_checklist(args: argparse.Namespace) -> dict[str, Any]:
    profile_name = profile_name_for_seed(int(args.seed))
    return {
        "profile_name": profile_name,
        "variant_name": str(args.variant_name),
        "profile_parameters": {
            "short_edge_policy": str(args.short_edge_policy),
            "crop_min_length_m": float(args.crop_min_length_m),
            "min_edge_length_m": float(args.min_edge_length_m),
            "min_root_child_length_m": float(args.min_root_child_length_m),
            "min_root_distance_m": float(args.min_root_distance_m),
            "density_radius_m": float(args.density_radius_m),
            "max_nodes_per_density_radius": int(args.max_nodes_per_density_radius),
            "num_nodes": int(args.num_nodes),
            "max_extension_m": float(args.max_extension_m),
            "sample_mode": str(args.sample_mode),
            "baseline_gain_mode": str(args.baseline_gain_mode),
            "gain_mode": str(args.gain_mode),
            "sc_gain_formula": str(args.sc_gain_formula),
            "path_cost_mode": str(args.path_cost_mode),
            "v_max": float(args.v_max),
            "robot_radius_m": float(args.robot_radius_m),
            "voxel_size": float(args.voxel_size),
            "raycast_stride": int(args.raycast_stride),
            "num_yaw_samples": int(args.num_yaw_samples),
            "max_ray_length_m": float(args.max_ray_length_m),
            "tau": float(args.tau),
            "alignment_convention": str(args.alignment_convention),
            "seed": int(args.seed),
        },
        "mechanisms": {
            "crop_min_length_min_path_length": {
                "implemented": True,
                "active": str(args.short_edge_policy) == "crop" and float(args.crop_min_length_m) > 0.0,
                "value_m": float(args.crop_min_length_m),
            },
            "density_limiting_max_density_range": {
                "implemented": True,
                "active": False,
                "density_radius_m": float(args.density_radius_m),
                "max_nodes_per_density_radius": int(args.max_nodes_per_density_radius),
                "reason": "Stage 4A-6.5k showed density limiting at radius 0.25 / max nodes 1 was too restrictive",
            },
            "continuous_yaw": {
                "implemented_approximation": True,
                "active": int(args.num_yaw_samples) > 0,
                "num_yaw_samples": int(args.num_yaw_samples),
            },
            "root_rewiring_reinsert": {
                "full_implementation": False,
                "active": False,
                "reason": "not part of this two-frame smoke",
            },
            "optional_parent_visible_clearing": {
                "active": False,
                "reason": "optional source evidence only",
            },
            "root_visible_filtering_near_root_discount": {
                "active": False,
                "reason": "no mandatory source evidence",
            },
        },
        "prediction": {
            "enabled": True,
            "prediction_mode": str(args.prediction_mode),
            "map_predict_used": str(args.prediction_mode) == "sim_dynamic",
            "prediction_writeback": False,
            "prediction_used_for_information_gain_only": True,
            "prediction_used_for_collision_traversability": False,
            "prediction_blocks_rays": False,
        },
    }


def _write_source_protection_md(path: Path, checklist: dict[str, Any]) -> None:
    mech = checklist["mechanisms"]
    pred = checklist["prediction"]
    lines = [
        "# Source Protection Checklist",
        "",
        f"- profile: `{checklist['profile_name']}`",
        f"- variant: `{checklist['variant_name']}`",
        f"- crop_min_length / min_path_length: active `{mech['crop_min_length_min_path_length']['active']}`, value `{mech['crop_min_length_min_path_length']['value_m']}` m.",
        f"- density limiting / max_density_range: active `{mech['density_limiting_max_density_range']['active']}`.",
        f"- continuous yaw approximation: active `{mech['continuous_yaw']['active']}`, samples `{mech['continuous_yaw']['num_yaw_samples']}`.",
        f"- prediction enabled: `{pred['enabled']}`; map_predict used: `{pred['map_predict_used']}`.",
        "- prediction writeback / traversability / collision / ray blocking: `False` / `False` / `False` / `False`.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _make_prediction_safety_checklist(args: argparse.Namespace, predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "map_predict_predictions": int(len(predictions)),
        "prediction_mode": str(args.prediction_mode),
        "alignment_convention": str(args.alignment_convention),
        "tau": float(args.tau),
        "checkpoint_loaded_once": all(bool(item.get("model_loaded_once", False)) for item in predictions),
        "predictor_steps_predicted": int(predictions[-1].get("steps_predicted", len(predictions))) if predictions else 0,
        "prediction_writeback": False,
        "prediction_used_for_information_gain_only": True,
        "prediction_used_for_collision": False,
        "prediction_used_for_traversability": False,
        "prediction_blocks_rays": False,
        "target_lr_target_hr_ground_truth_scoring": False,
        "observed_state_modified_by_prediction": False,
        "large_dense_class_prob_saved": bool(args.save_probs),
    }


def _write_prediction_safety_md(path: Path, checklist: dict[str, Any]) -> None:
    lines = [
        "# Prediction Safety Checklist",
        "",
        f"- map_predict predictions: `{checklist['map_predict_predictions']}`",
        f"- checkpoint loaded once: `{checklist['checkpoint_loaded_once']}`",
        f"- predictor steps predicted: `{checklist['predictor_steps_predicted']}`",
        "- prediction writeback: `False`",
        "- prediction used for information gain only: `True`",
        "- prediction used for collision/traversability/ray blocking: `False` / `False` / `False`",
        "- target/ground-truth scoring: `False`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _topdown_image(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def _grid_point(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    return float(grid[0]) + 0.5, float(grid[1]) + 0.5


def _save_tree_comparison_plot(
    path: Path,
    observed_state: np.ndarray,
    measured: dict[str, Any],
    sc_tree: dict[str, Any],
    title: str,
) -> None:
    image = _topdown_image(observed_state)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    root = _grid_point(sc_tree.get("root", {}).get("grid") or measured.get("root", {}).get("grid"))
    points = [
        ("measured selected", measured.get("selected", {}).get("end_grid"), "#f97316", "o", "-"),
        ("measured best", measured.get("best", {}).get("end_grid"), "#fb923c", "*", "--"),
        ("SC selected", sc_tree.get("selected", {}).get("end_grid"), "#3b82f6", "o", "-"),
        ("SC best", sc_tree.get("best", {}).get("end_grid"), "#2563eb", "*", "--"),
    ]
    if root:
        ax.scatter([root[0]], [root[1]], c="#ffffff", edgecolors="#111111", s=90, label="root")
    for label, grid, color, marker, style in points:
        point = _grid_point(grid)
        if point:
            ax.scatter([point[0]], [point[1]], c=color, marker=marker, s=130 if marker == "*" else 75, label=label)
            if root:
                ax.plot([root[0], point[0]], [root[1], point[1]], color=color, linestyle=style, linewidth=1.7)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_two_frame_sc_path_plot(path: Path, observed_state: np.ndarray, frame1: dict[str, Any], frame2: dict[str, Any]) -> None:
    image = _topdown_image(observed_state)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    for label, parts, color in (("frame1", frame1, "#d1495b"), ("frame2", frame2, "#3f88c5")):
        root = _grid_point(parts.get("root", {}).get("grid"))
        selected = _grid_point(parts.get("selected", {}).get("end_grid"))
        best = _grid_point(parts.get("best", {}).get("end_grid"))
        if root:
            ax.scatter([root[0]], [root[1]], c="#ffffff", edgecolors=color, s=85, label=f"{label} root")
        if selected:
            ax.scatter([selected[0]], [selected[1]], c=color, s=75, label=f"{label} selected")
        if best:
            ax.scatter([best[0]], [best[1]], c=color, marker="*", s=140, label=f"{label} best")
        if root and selected:
            ax.plot([root[0], selected[0]], [root[1], selected[1]], color=color, linewidth=2.3)
        if root and best:
            ax.plot([root[0], best[0]], [root[1], best[1]], color=color, linewidth=1.1, linestyle="--")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Two-frame SC selected path")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_prediction_overlay(path: Path, observed_state: np.ndarray, prediction_npz: str, tau: float, title: str) -> None:
    with np.load(prediction_npz, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occupied_prob = np.asarray(data["global_occupied_prob"], dtype=np.float32)
    predicted = valid & (confidence >= float(tau)) & (observed_state == UNKNOWN)
    pred_occ = predicted & (occupied_prob >= 0.5)
    pred_top = np.any(predicted, axis=2)
    pred_occ_top = np.any(pred_occ, axis=2)
    image = _topdown_image(observed_state)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.6, 6.8), constrained_layout=True)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    overlay = np.ma.masked_where(~pred_top.T, pred_top.T.astype(float))
    ax.imshow(overlay, origin="lower", cmap="magma", alpha=0.45, interpolation="nearest")
    occ = np.argwhere(pred_occ_top)
    if len(occ):
        ax.scatter(occ[:, 0] + 0.5, occ[:, 1] + 0.5, c="#fde047", s=2, alpha=0.45, label="pred occupied")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    if len(occ):
        ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_gain_scatter(path: Path, tree_dir: Path, title: str) -> None:
    rows = [row for row in _load_segments(tree_dir / "mini_rrt_tree_segments.jsonl") if row.get("segment_id") != "root"]
    gain_exp = [float(row.get("gain_exp", 0.0) or 0.0) for row in rows]
    gain_sc = [float(row.get("gain_sc", 0.0) or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(6.8, 5.2), constrained_layout=True)
    ax.scatter(gain_exp, gain_sc, s=18, alpha=0.72, color="#2563eb")
    ax.set_xlabel("gain_exp")
    ax.set_ylabel("gain_sc")
    ax.set_title(title)
    corr = _pearson(gain_exp, gain_sc)
    if corr is not None:
        ax.text(0.02, 0.98, f"pearson={corr:.4f}", transform=ax.transAxes, ha="left", va="top")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _scan_prohibited_rollout_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in (
        "transitions.jsonl",
        "step_*.npz",
        "observed_ratio_curve.png",
        "rollout_topdown_path.png",
        "rollout_index.html",
        "step_topdown_*.png",
        "frame003*",
    ):
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def _run_prediction(
    args: argparse.Namespace,
    predictor: IsaacMapPredictor | None,
    output_dir: Path,
    capture: dict[str, Any],
    observed: dict[str, Any],
    frame_index: int,
) -> dict[str, Any]:
    observed_state = np.load(observed["new_observed_state"])
    prefix = f"frame{frame_index:03d}"
    if str(args.prediction_mode) == "sim_npz":
        if not args.prediction_npz:
            raise ValueError("--prediction_npz is required when --prediction_mode sim_npz")
        prediction_npz = str(Path(args.prediction_npz).resolve())
        return {
            "prediction_layer": SimPredictionLayer.from_npz(prediction_npz),
            "prediction_npz": prediction_npz,
            "global_prediction_npz": prediction_npz,
            "local_prediction_npz": None,
            "summary": {"map_predict_executed": False, "prediction_npz": prediction_npz},
            "summary_json": None,
            "timing": {},
            "model_loaded_once": False,
            "steps_predicted": int(frame_index),
            "checkpoint_unchanged": True,
            "map_predict_success": False,
        }
    if predictor is None:
        raise ValueError("sim_dynamic prediction requires a predictor")
    result = predictor.predict_step(
        depth=np.asarray(capture["depth"], dtype=np.float32),
        pose=capture["pose"],
        camera_info=capture["camera_info"],
        observed_state=observed_state,
        map_bounds=observed["bounds"],
        voxel_size=float(args.voxel_size),
        output_dir=output_dir / f"{prefix}_prediction",
        step=int(frame_index),
        save_probs=bool(args.save_probs),
        save_viz=bool(args.save_viz),
        observed_state_path=observed["new_observed_state"],
        depth_source=capture["depth_path"],
        pose_source=capture["pose_path"],
        camera_info_source=capture["camera_info_path"],
    )
    result["model_loaded_once"] = bool(predictor.model_loaded_once)
    result["steps_predicted"] = int(predictor.steps_predicted)
    result["checkpoint_unchanged"] = bool(predictor.checkpoint_unchanged())
    result["map_predict_success"] = True
    return result


def _compare_decisions(measured: dict[str, Any], sc_tree: dict[str, Any]) -> dict[str, Any]:
    measured_selected = measured.get("selected", {})
    sc_selected = sc_tree.get("selected", {})
    measured_best = measured.get("best", {})
    sc_best = sc_tree.get("best", {})
    selected_same_grid = _same_grid(measured_selected.get("end_grid"), sc_selected.get("end_grid"))
    best_same_grid = _same_grid(measured_best.get("end_grid"), sc_best.get("end_grid"))
    selected_delta = _euclidean(measured_selected.get("end_world"), sc_selected.get("end_world"))
    best_delta = _euclidean(measured_best.get("end_world"), sc_best.get("end_world"))
    measured_value = float(measured.get("value") or 0.0)
    sc_value = float(sc_tree.get("value") or 0.0)
    selected_changed = not selected_same_grid
    return {
        "selected_child_same_grid": selected_same_grid,
        "selected_child_differs_from_measured_only": selected_changed,
        "selected_child_world_delta_m": selected_delta,
        "selected_child_spatially_meaningful_change": bool(
            selected_changed and selected_delta is not None and float(selected_delta) >= 0.25
        ),
        "best_descendant_same_grid": best_same_grid,
        "best_descendant_differs_from_measured_only": not best_same_grid,
        "best_descendant_world_delta_m": best_delta,
        "value_delta_sc_minus_measured": float(sc_value - measured_value),
        "measured_selected_child": measured_selected,
        "sc_selected_child": sc_selected,
        "measured_best_descendant": measured_best,
        "sc_best_descendant": sc_best,
    }


def _compare_to_reference_sc(frame_parts: dict[str, Any], reference_dir: Path) -> dict[str, Any]:
    decision_path = reference_dir / "sc_tree_decision.json"
    if not decision_path.is_file():
        return {"available": False, "path": str(decision_path)}
    payload = _load_json(decision_path)
    ref = payload.get("decision", {})
    ref_selected = ref.get("selected", {})
    ref_best = ref.get("best", {})
    return {
        "available": True,
        "path": str(decision_path),
        "selected_child_same_id": frame_parts["selected"].get("segment_id") == ref_selected.get("segment_id"),
        "selected_child_same_grid": _same_grid(frame_parts["selected"].get("end_grid"), ref_selected.get("end_grid")),
        "best_descendant_same_id": frame_parts["best"].get("segment_id") == ref_best.get("segment_id"),
        "best_descendant_same_grid": _same_grid(frame_parts["best"].get("end_grid"), ref_best.get("end_grid")),
        "accumulated_gain_same": _close(frame_parts.get("accumulated_gain"), ref.get("accumulated_gain")),
        "accumulated_cost_same": _close(frame_parts.get("accumulated_cost"), ref.get("accumulated_cost")),
        "value_same": _close(frame_parts.get("value"), ref.get("value")),
        "reference_accumulated_gain": ref.get("accumulated_gain"),
        "reference_accumulated_cost": ref.get("accumulated_cost"),
        "reference_value": ref.get("value"),
    }


def _compare_to_no_pred_frame2(frame_parts: dict[str, Any], reference_dir: Path) -> dict[str, Any]:
    decision_path = reference_dir / "frame002_tree_decision.json"
    if not decision_path.is_file():
        return {"available": False, "path": str(decision_path)}
    ref = _load_json(decision_path)
    return {
        "available": True,
        "path": str(decision_path),
        "selected_child_same_id": frame_parts["selected"].get("segment_id") == ref.get("selected_child", {}).get("segment_id"),
        "selected_child_same_grid": _same_grid(frame_parts["selected"].get("end_grid"), ref.get("selected_child", {}).get("end_grid")),
        "best_descendant_same_id": frame_parts["best"].get("segment_id") == ref.get("best_descendant", {}).get("segment_id"),
        "best_descendant_same_grid": _same_grid(frame_parts["best"].get("end_grid"), ref.get("best_descendant", {}).get("end_grid")),
        "accumulated_gain_same": _close(frame_parts.get("accumulated_gain"), ref.get("accumulated_gain")),
        "accumulated_cost_same": _close(frame_parts.get("accumulated_cost"), ref.get("accumulated_cost")),
        "value_same": _close(frame_parts.get("value"), ref.get("value")),
        "reference_accumulated_gain": ref.get("accumulated_gain"),
        "reference_accumulated_cost": ref.get("accumulated_cost"),
        "reference_value": ref.get("value"),
    }


def _decision_payload(
    frame_index: int,
    label: str,
    args: argparse.Namespace,
    parts: dict[str, Any],
    gain_stats: dict[str, Any],
    accumulated_by_formula: dict[str, Any],
    observed_update: dict[str, Any],
    capture: dict[str, Any],
    prediction_stats: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_name = profile_name_for_seed(int(args.seed))
    return {
        "stage": "Stage 4A-6.5p map_predict + source-protected tree two-frame smoke",
        "frame_index": int(frame_index),
        "tree_label": label,
        "profile_name": profile_name,
        "variant_name": str(args.variant_name),
        "profile": {
            "short_edge_policy": str(args.short_edge_policy),
            "crop_min_length_m": float(args.crop_min_length_m),
            "num_nodes": int(args.num_nodes),
            "max_extension_m": float(args.max_extension_m),
            "sample_mode": str(args.sample_mode),
            "path_cost_mode": str(args.path_cost_mode),
            "v_max": float(args.v_max),
            "robot_radius_m": float(args.robot_radius_m),
            "voxel_size": float(args.voxel_size),
            "raycast_stride": int(args.raycast_stride),
            "num_yaw_samples": int(args.num_yaw_samples),
            "max_ray_length_m": float(args.max_ray_length_m),
            "seed": int(args.seed),
            "gain_mode": str(args.baseline_gain_mode if label == "measured" else args.gain_mode),
            "sc_gain_formula": str(args.sc_gain_formula),
            "tau": float(args.tau),
        },
        "selected_child": parts["selected"],
        "selected_child_distance_from_root_m": parts["selected_distance_m"],
        "best_descendant": parts["best"],
        "best_descendant_distance_from_root_m": parts["best_distance_m"],
        "value": parts["value"],
        "accumulated_gain": parts["accumulated_gain"],
        "accumulated_cost": parts["accumulated_cost"],
        "accumulated_gain_exp": accumulated_by_formula.get("gain_exp"),
        "accumulated_gain_sc": accumulated_by_formula.get("gain_sc"),
        "accumulated_gain_hybrid": accumulated_by_formula.get("gain_hybrid"),
        "accumulated_cost_from_path_sum": accumulated_by_formula.get("cost"),
        "path_segment_ids": accumulated_by_formula.get("path_segment_ids"),
        "accepted_nodes": parts["accepted_nodes"],
        "rejected_samples": parts["rejected_samples"],
        "built_successfully": parts["built_successfully"],
        "root": parts["root"],
        "raw_decision": parts["raw_decision"],
        "gain_stats": gain_stats,
        "capture": {
            "rgb_path": capture["rgb_path"],
            "depth_path": capture["depth_path"],
            "pose_path": capture["pose_path"],
            "camera_info_path": capture["camera_info_path"],
            "depth_stats": capture["depth_stats"],
            "rgb_stats": capture["rgb_stats"],
        },
        "observed_state": {
            "path": observed_update["new_observed_state"],
            "sha256": observed_update["new_observed_state_sha256"],
            "prior_path": observed_update["prior_observed_state"],
            "prior_sha256_before": observed_update["prior_sha256_before"],
            "prior_sha256_after": observed_update["prior_sha256_after"],
            "prior_unchanged": observed_update["prior_hash_unchanged"],
            "delta_observed_count": observed_update["delta_observed_count"],
            "delta_observed_ratio": observed_update["delta_observed_ratio"],
            "updated_summary": observed_update["updated_summary"],
        },
        "prediction_stats": prediction_stats,
        "safety": {
            "prediction_used": label == "sc",
            "map_predict_used": label == "sc" and str(args.prediction_mode) == "sim_dynamic",
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "prediction_blocks_rays": False,
            "target_lr_target_hr_ground_truth_scoring": False,
        },
        "extra": extra or {},
    }


def _write_decision_md(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_child"]
    best = payload["best_descendant"]
    lines = [
        f"# Frame {payload['frame_index']:03d} {payload['tree_label']} Tree Decision",
        "",
        f"- profile: `{payload['profile_name']}`",
        f"- selected child: `{selected.get('segment_id')}` grid `{selected.get('end_grid')}` world `{selected.get('end_world')}`",
        f"- best descendant: `{best.get('segment_id')}` grid `{best.get('end_grid')}` world `{best.get('end_world')}`",
        f"- selected gains exp/sc/hybrid/occ/conf: `{selected.get('gain_exp')}` / `{selected.get('gain_sc')}` / `{selected.get('gain_hybrid')}` / `{selected.get('gain_occ')}` / `{selected.get('gain_conf')}`",
        f"- best gains exp/sc/hybrid/occ/conf: `{best.get('gain_exp')}` / `{best.get('gain_sc')}` / `{best.get('gain_hybrid')}` / `{best.get('gain_occ')}` / `{best.get('gain_conf')}`",
        f"- accumulated gain_exp/gain_sc/gain_hybrid/cost: `{payload['accumulated_gain_exp']}` / `{payload['accumulated_gain_sc']}` / `{payload['accumulated_gain_hybrid']}` / `{payload['accumulated_cost']}`",
        f"- value: `{payload['value']}`",
        f"- nodes with gain_sc > 0: `{payload['gain_stats']['nodes_with_gain_sc_positive']}` / `{payload['gain_stats']['node_count_excluding_root']}`",
        "- prediction writeback / collision / traversability / ray blocking: `False` / `False` / `False` / `False`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_comparison_md(path: Path, comparison: dict[str, Any], frame_index: int) -> None:
    lines = [
        f"# Frame {frame_index:03d} Measured Vs SC Tree Comparison",
        "",
        f"- selected child differs: `{comparison['selected_child_differs_from_measured_only']}`",
        f"- selected child world delta m: `{comparison['selected_child_world_delta_m']}`",
        f"- selected child spatially meaningful: `{comparison['selected_child_spatially_meaningful_change']}`",
        f"- best descendant differs: `{comparison['best_descendant_differs_from_measured_only']}`",
        f"- best descendant world delta m: `{comparison['best_descendant_world_delta_m']}`",
        f"- value delta SC - measured: `{comparison['value_delta_sc_minus_measured']}`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _run_tree_pair(
    args: argparse.Namespace,
    output_dir: Path,
    capture: dict[str, Any],
    observed: dict[str, Any],
    prediction: dict[str, Any],
    prediction_stats: dict[str, Any],
) -> dict[str, Any]:
    frame_index = int(capture["frame_index"])
    prefix = f"frame{frame_index:03d}"
    measured_tree_dir = output_dir / f"{prefix}_measured_tree_raw"
    measured_summary = run_mini_rrt(
        _tree_args(
            args,
            measured_tree_dir,
            observed["new_observed_state"],
            capture,
            prediction_npz="",
            gain_mode=str(args.baseline_gain_mode),
            variant_suffix=f"{prefix}_measured_only",
        )
    )
    measured_parts = _decision_parts(measured_summary)
    measured_gain_stats = _tree_gain_stats(measured_tree_dir)
    measured_path_sums = _path_sums(measured_tree_dir, measured_parts["best"].get("segment_id"))

    sc_tree_dir = output_dir / f"{prefix}_sc_tree_raw"
    sc_summary = run_mini_rrt(
        _tree_args(
            args,
            sc_tree_dir,
            observed["new_observed_state"],
            capture,
            prediction_npz=str(prediction["prediction_npz"]),
            gain_mode=str(args.gain_mode),
            variant_suffix=f"{prefix}_{args.prediction_mode}_{args.gain_mode}",
        )
    )
    sc_parts = _decision_parts(sc_summary)
    sc_gain_stats = _tree_gain_stats(sc_tree_dir)
    sc_path_sums = _path_sums(sc_tree_dir, sc_parts["best"].get("segment_id"))
    comparison = _compare_decisions(measured_parts, sc_parts)
    hybrid_identity = _check_hybrid_identity(sc_tree_dir / "gain_cost_value_table.csv")

    aliases = {}
    aliases.update(_alias_tree_outputs(output_dir, measured_tree_dir, f"{prefix}_measured"))
    aliases.update(_alias_tree_outputs(output_dir, sc_tree_dir, f"{prefix}_sc"))

    measured_payload = _decision_payload(
        frame_index,
        "measured",
        args,
        measured_parts,
        measured_gain_stats,
        measured_path_sums,
        observed,
        capture,
        prediction_stats=None,
    )
    sc_payload = _decision_payload(
        frame_index,
        "sc",
        args,
        sc_parts,
        sc_gain_stats,
        sc_path_sums,
        observed,
        capture,
        prediction_stats=prediction_stats,
        extra={"hybrid_identity": hybrid_identity},
    )
    _save_json(output_dir / f"{prefix}_measured_tree_decision.json", measured_payload)
    _write_decision_md(output_dir / f"{prefix}_measured_tree_decision.md", measured_payload)
    _save_json(output_dir / f"{prefix}_sc_tree_decision.json", sc_payload)
    _write_decision_md(output_dir / f"{prefix}_sc_tree_decision.md", sc_payload)
    _save_json(output_dir / f"{prefix}_measured_vs_sc_comparison.json", comparison)
    _write_comparison_md(output_dir / f"{prefix}_measured_vs_sc_comparison.md", comparison, frame_index)

    observed_state = np.load(observed["new_observed_state"])
    _save_tree_comparison_plot(
        output_dir / f"{prefix}_measured_vs_sc_tree_topdown.png",
        observed_state,
        measured_parts,
        sc_parts,
        f"Frame {frame_index:03d} measured-only vs SC tree",
    )
    _save_prediction_overlay(
        output_dir / f"prediction_overlay_{prefix}_topdown.png",
        observed_state,
        str(prediction["prediction_npz"]),
        float(args.tau),
        f"Frame {frame_index:03d} prediction overlay",
    )
    _save_gain_scatter(
        output_dir / f"sc_gain_vs_exp_gain_{prefix}.png",
        sc_tree_dir,
        f"Frame {frame_index:03d} gain_sc vs gain_exp",
    )
    return {
        "frame_index": frame_index,
        "measured_summary": measured_summary,
        "sc_summary": sc_summary,
        "measured_parts": measured_parts,
        "sc_parts": sc_parts,
        "measured_gain_stats": measured_gain_stats,
        "sc_gain_stats": sc_gain_stats,
        "measured_path_sums": measured_path_sums,
        "sc_path_sums": sc_path_sums,
        "comparison": comparison,
        "hybrid_identity": hybrid_identity,
        "aliases": aliases,
        "measured_payload": measured_payload,
        "sc_payload": sc_payload,
        "measured_tree_dir": str(measured_tree_dir),
        "sc_tree_dir": str(sc_tree_dir),
    }


def _frame1_vs_frame2_comparison(frame1: dict[str, Any], frame2: dict[str, Any]) -> dict[str, Any]:
    f1 = frame1["sc_parts"]
    f2 = frame2["sc_parts"]
    return {
        "frame1_selected_child": f1["selected"],
        "frame2_selected_child": f2["selected"],
        "selected_child_same_grid": _same_grid(f1["selected"].get("end_grid"), f2["selected"].get("end_grid")),
        "selected_child_world_delta_m": _euclidean(f1["selected"].get("end_world"), f2["selected"].get("end_world")),
        "frame1_best_descendant": f1["best"],
        "frame2_best_descendant": f2["best"],
        "best_descendant_same_grid": _same_grid(f1["best"].get("end_grid"), f2["best"].get("end_grid")),
        "best_descendant_world_delta_m": _euclidean(f1["best"].get("end_world"), f2["best"].get("end_world")),
        "value_delta_frame2_minus_frame1": float(float(f2.get("value") or 0.0) - float(f1.get("value") or 0.0)),
    }


def _write_frame_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Frame 001 Vs Frame 002 SC Tree Comparison",
        "",
        f"- selected child same grid: `{comparison['selected_child_same_grid']}`",
        f"- selected child world delta m: `{comparison['selected_child_world_delta_m']}`",
        f"- best descendant same grid: `{comparison['best_descendant_same_grid']}`",
        f"- best descendant world delta m: `{comparison['best_descendant_world_delta_m']}`",
        f"- value delta frame2 - frame1: `{comparison['value_delta_frame2_minus_frame1']}`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _next_step(frame1_tree: dict[str, Any], frame2_tree: dict[str, Any]) -> str:
    f1_selected = bool(frame1_tree["comparison"]["selected_child_differs_from_measured_only"])
    f2_selected = bool(frame2_tree["comparison"]["selected_child_differs_from_measured_only"])
    f1_best = bool(frame1_tree["comparison"]["best_descendant_differs_from_measured_only"])
    f2_best = bool(frame2_tree["comparison"]["best_descendant_differs_from_measured_only"])
    meaningful = bool(
        frame1_tree["comparison"]["selected_child_spatially_meaningful_change"]
        or frame2_tree["comparison"]["selected_child_spatially_meaningful_change"]
    )
    if not f1_selected and not f2_selected:
        return "SC-tree gain selectivity/rank diagnosis"
    if (f1_best or f2_best) and not (f1_selected or f2_selected):
        return "SC tree branch-depth analysis"
    if meaningful:
        return "controlled gated SC tree one-step smoke or repeated two-frame smoke"
    return "SC tree branch-depth analysis"


def _make_summary(
    args: argparse.Namespace,
    output_dir: Path,
    scene_metadata: dict[str, Any],
    capture1: dict[str, Any],
    capture2: dict[str, Any],
    observed1: dict[str, Any],
    observed2: dict[str, Any],
    prediction1: dict[str, Any],
    prediction2: dict[str, Any],
    prediction_stats1: dict[str, Any],
    prediction_stats2: dict[str, Any],
    frame1_tree: dict[str, Any],
    frame2_tree: dict[str, Any],
    reference_frame1_sc: dict[str, Any],
    reference_frame2_no_pred: dict[str, Any],
    frame_sc_comparison: dict[str, Any],
    source_checklist: dict[str, Any],
    prediction_checklist: dict[str, Any],
    checkpoint_before: str | None,
    checkpoint_after: str | None,
    external_before: str,
    external_after: str,
    ssc_before: str,
    ssc_after: str,
) -> dict[str, Any]:
    frame1_observed_hash_after = sha256_file(observed1["new_observed_state"])
    frame2_observed_hash_after = sha256_file(observed2["new_observed_state"])
    frame1_sc = frame1_tree["sc_parts"]
    frame2_sc = frame2_tree["sc_parts"]
    gain_sc_nonzero_both = (
        int(frame1_tree["sc_gain_stats"]["nodes_with_gain_sc_positive"]) > 0
        and int(frame2_tree["sc_gain_stats"]["nodes_with_gain_sc_positive"]) > 0
    )
    dense_unselective = bool(
        frame1_tree["sc_gain_stats"]["gain_sc_density"] >= 0.9
        and frame2_tree["sc_gain_stats"]["gain_sc_density"] >= 0.9
        and not frame1_tree["comparison"]["selected_child_differs_from_measured_only"]
        and not frame2_tree["comparison"]["selected_child_differs_from_measured_only"]
    )
    next_step = _next_step(frame1_tree, frame2_tree)
    return {
        "stage": "Stage 4A-6.5p map_predict + source-protected tree two-frame smoke",
        "output_dir": str(output_dir),
        "profile_name": profile_name_for_seed(int(args.seed)),
        "scene": {
            "scene_variant": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "scene_metadata": scene_metadata,
        },
        "answers": {
            "frame1_capture_observed_map_predict_sc_tree_success": bool(
                capture1["depth_stats"]["positive_count"] > 0
                and int(observed1["delta_observed_count"]) > 0
                and bool(prediction1["map_predict_success"])
                and bool(frame1_sc["built_successfully"])
            ),
            "frame1_reproduces_stage4a65o": bool(
                reference_frame1_sc.get("available")
                and reference_frame1_sc.get("selected_child_same_grid")
                and reference_frame1_sc.get("best_descendant_same_grid")
            ),
            "frame1_sc_changed_measured_selected_child": bool(
                frame1_tree["comparison"]["selected_child_differs_from_measured_only"]
            ),
            "frame2_capture_observed_map_predict_sc_tree_success": bool(
                capture2["depth_stats"]["positive_count"] > 0
                and int(observed2["delta_observed_count"]) > 0
                and bool(prediction2["map_predict_success"])
                and bool(frame2_sc["built_successfully"])
            ),
            "frame2_sc_changed_measured_selected_child": bool(
                frame2_tree["comparison"]["selected_child_differs_from_measured_only"]
            ),
            "frame2_sc_changed_measured_best_descendant": bool(
                frame2_tree["comparison"]["best_descendant_differs_from_measured_only"]
            ),
            "gain_sc_nonzero_both_frames": gain_sc_nonzero_both,
            "prediction_dense_unselective": dense_unselective,
            "prediction_read_only": (
                observed1["new_observed_state_sha256"] == frame1_observed_hash_after
                and observed2["new_observed_state_sha256"] == frame2_observed_hash_after
            ),
            "prediction_used_for_traversability_collision_ray_blocking": False,
            "enough_for_sc_tree_gain_selectivity_gating_diagnosis": True,
            "ready_for_rollout": False,
        },
        "frames": {
            "frame001": {
                "capture": {k: v for k, v in capture1.items() if k != "depth"},
                "observed_state": observed1,
                "prediction": {
                    "result_summary": prediction1.get("summary", {}),
                    "summary_json": prediction1.get("summary_json"),
                    "prediction_npz": prediction1.get("prediction_npz"),
                    "local_prediction_npz": prediction1.get("local_prediction_npz"),
                    "timing": prediction1.get("timing", {}),
                    "stats": prediction_stats1,
                },
                "tree": frame1_tree,
                "comparison_to_stage4a65o": reference_frame1_sc,
            },
            "frame002": {
                "capture": {k: v for k, v in capture2.items() if k != "depth"},
                "observed_state": observed2,
                "prediction": {
                    "result_summary": prediction2.get("summary", {}),
                    "summary_json": prediction2.get("summary_json"),
                    "prediction_npz": prediction2.get("prediction_npz"),
                    "local_prediction_npz": prediction2.get("local_prediction_npz"),
                    "timing": prediction2.get("timing", {}),
                    "stats": prediction_stats2,
                },
                "tree": frame2_tree,
                "comparison_to_stage4a65n_no_prediction_frame2": reference_frame2_no_pred,
            },
        },
        "move_once": {
            "executed_action": True,
            "execution_count": 1,
            "selected_from": "frame001_sc_tree",
            "selected_child_segment_id": frame1_sc["selected"].get("segment_id"),
            "selected_child_grid": frame1_sc["selected"].get("end_grid"),
            "selected_child_world": frame1_sc["selected"].get("end_world"),
            "new_pose": capture2["pose"],
            "motion_mode": "planar_teleport_once_to_sc_selected_child_xy_fixed_camera_height",
            "safety_note": "exactly one action was executed between frame 1 and frame 2; no rollout loop was entered",
        },
        "comparisons": {
            "frame001_measured_vs_sc": frame1_tree["comparison"],
            "frame002_measured_vs_sc": frame2_tree["comparison"],
            "frame001_vs_frame002_sc_tree": frame_sc_comparison,
        },
        "diagnosis": {
            "gain_sc_density": {
                "frame001": frame1_tree["sc_gain_stats"]["gain_sc_density"],
                "frame002": frame2_tree["sc_gain_stats"]["gain_sc_density"],
            },
            "gain_exp_gain_sc_correlation": {
                "frame001": frame1_tree["sc_gain_stats"]["gain_exp_gain_sc_pearson"],
                "frame002": frame2_tree["sc_gain_stats"]["gain_exp_gain_sc_pearson"],
            },
            "value_delta_sc_minus_measured": {
                "frame001": frame1_tree["comparison"]["value_delta_sc_minus_measured"],
                "frame002": frame2_tree["comparison"]["value_delta_sc_minus_measured"],
            },
            "prediction_dense_unselective": dense_unselective,
            "interpretation": (
                "gain_sc is dense and did not change the selected child in either frame"
                if dense_unselective
                else "SC tree produced a changed branch or less-dense prediction signal; inspect frame comparisons"
            ),
        },
        "source_protection_checklist": source_checklist,
        "prediction_safety_checklist": prediction_checklist,
        "safety": {
            "isaac_startup": True,
            "frames_captured": 2,
            "selected_action_execution": True,
            "selected_action_execution_count": 1,
            "rollout": False,
            "online_open_ended_loop": False,
            "map_predict_predictions": int(prediction_checklist["map_predict_predictions"]),
            "sscnet_training": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_modified": checkpoint_before != checkpoint_after,
            "existing_observed_state_modified": not bool(observed1["prior_hash_unchanged"]),
            "new_observed_state_written_under_output_dir_only": True,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "prediction_blocks_rays": False,
            "target_lr_target_hr_ground_truth_scoring": False,
            "external_source_git_status_before": external_before,
            "external_source_git_status_after": external_after,
            "external_source_modified_or_built_by_stage": external_before != external_after,
            "ssc_exploration_git_status_before": ssc_before,
            "ssc_exploration_git_status_after": ssc_after,
            "ssc_exploration_modified_by_stage": ssc_before != ssc_after,
            "prohibited_rollout_outputs": _scan_prohibited_rollout_outputs(output_dir),
            "coverage_improvement_claimed": False,
        },
        "recommended_next_faithful_step": next_step,
        "still_not_next": [
            "rollout",
            "online open-ended loop",
            "RL/PPO/BC/IL training",
            "prediction writeback",
            "observed_map prediction fusion",
            "target/ground-truth scoring",
            "checkpoint changes",
            "coverage improvement claim",
            "external source build",
        ],
    }


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    f1 = summary["frames"]["frame001"]
    f2 = summary["frames"]["frame002"]
    f1_sc = f1["tree"]["sc_parts"]
    f2_sc = f2["tree"]["sc_parts"]
    lines = [
        "# Stage 4A-6.5p Map Predict Tree Two-Frame Summary",
        "",
        f"1. Frame 1 capture / observed update / map_predict / SC tree successful? `{answers['frame1_capture_observed_map_predict_sc_tree_success']}`.",
        f"2. Frame 1 reproduced Stage 4A-6.5o? `{answers['frame1_reproduces_stage4a65o']}`.",
        f"3. Frame 1 SC changed measured-only selected child? `{answers['frame1_sc_changed_measured_selected_child']}`.",
        f"4. Frame 2 capture / observed update / map_predict / SC tree successful? `{answers['frame2_capture_observed_map_predict_sc_tree_success']}`.",
        f"5. Frame 2 SC changed measured-only selected child? `{answers['frame2_sc_changed_measured_selected_child']}`.",
        f"6. Frame 2 SC changed measured-only best descendant? `{answers['frame2_sc_changed_measured_best_descendant']}`.",
        f"7. gain_sc nonzero in both frames? `{answers['gain_sc_nonzero_both_frames']}`.",
        f"8. prediction still dense/unselective? `{answers['prediction_dense_unselective']}`.",
        f"9. prediction read-only? `{answers['prediction_read_only']}`.",
        f"10. prediction used for traversability / collision / ray blocking? `{answers['prediction_used_for_traversability_collision_ray_blocking']}`.",
        f"11. enough for SC tree gain selectivity / gating diagnosis? `{answers['enough_for_sc_tree_gain_selectivity_gating_diagnosis']}`.",
        f"12. ready for rollout? `{answers['ready_for_rollout']}`.",
        "",
        "## Frame 1 SC Tree",
        f"- selected child: `{f1_sc['selected'].get('segment_id')}` grid `{f1_sc['selected'].get('end_grid')}`.",
        f"- best descendant: `{f1_sc['best'].get('segment_id')}` grid `{f1_sc['best'].get('end_grid')}`.",
        f"- accumulated exp/sc/hybrid/cost: `{f1['tree']['sc_path_sums']['gain_exp']}` / `{f1['tree']['sc_path_sums']['gain_sc']}` / `{f1['tree']['sc_path_sums']['gain_hybrid']}` / `{f1_sc['accumulated_cost']}`.",
        f"- value: `{f1_sc['value']}`.",
        "",
        "## Frame 2 SC Tree",
        f"- selected child: `{f2_sc['selected'].get('segment_id')}` grid `{f2_sc['selected'].get('end_grid')}`.",
        f"- best descendant: `{f2_sc['best'].get('segment_id')}` grid `{f2_sc['best'].get('end_grid')}`.",
        f"- accumulated exp/sc/hybrid/cost: `{f2['tree']['sc_path_sums']['gain_exp']}` / `{f2['tree']['sc_path_sums']['gain_sc']}` / `{f2['tree']['sc_path_sums']['gain_hybrid']}` / `{f2_sc['accumulated_cost']}`.",
        f"- value: `{f2_sc['value']}`.",
        "",
        "## Diagnosis",
        f"- gain_sc density frame001/frame002: `{summary['diagnosis']['gain_sc_density']['frame001']}` / `{summary['diagnosis']['gain_sc_density']['frame002']}`.",
        f"- gain_exp/gain_sc correlation frame001/frame002: `{summary['diagnosis']['gain_exp_gain_sc_correlation']['frame001']}` / `{summary['diagnosis']['gain_exp_gain_sc_correlation']['frame002']}`.",
        f"- interpretation: {summary['diagnosis']['interpretation']}.",
        "",
        f"Recommended next faithful step: {summary['recommended_next_faithful_step']}. Still no rollout.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_recommended_next(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {summary['recommended_next_faithful_step']}",
        f"- reason: {summary['diagnosis']['interpretation']}",
        "- still not next: rollout, online open-ended loop, RL/PPO/BC/IL training, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, coverage-improvement claims, or external source build.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_before = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    external_before = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_before = _git_status_short(SSC_EXPLORATION_DIR)
    source_checklist = _make_source_protection_checklist(args)

    predictor: IsaacMapPredictor | None = None
    if str(args.prediction_mode) == "sim_dynamic":
        predictor = IsaacMapPredictor(
            checkpoint=args.checkpoint,
            device="cuda",
            tau=float(args.tau),
            torch_num_threads=int(args.torch_num_threads),
            alignment_convention=str(args.alignment_convention),
        )

    sim, camera, scene_metadata = _start_scene(args)
    initial_pose = _initial_pose(args)
    capture1 = _capture_frame(args, output_dir, camera, sim, initial_pose, 1)

    episode_dir = Path(args.episode_dir).resolve()
    prior0_path = episode_dir / "observed_state_step000.npy"
    observed1 = _update_observed_state(args, output_dir, capture1, prior0_path, 1)
    prediction1 = _run_prediction(args, predictor, output_dir, capture1, observed1, 1)
    observed_state1 = np.load(observed1["new_observed_state"])
    prediction_stats1 = _prediction_stats(str(prediction1["prediction_npz"]), observed_state1, float(args.tau))
    frame1_tree = _run_tree_pair(args, output_dir, capture1, observed1, prediction1, prediction_stats1)

    pose2 = _pose_from_selected_child(frame1_tree["sc_parts"]["selected"])
    capture2 = _capture_frame(args, output_dir, camera, sim, pose2, 2)
    observed2 = _update_observed_state(args, output_dir, capture2, Path(observed1["new_observed_state"]), 2)
    prediction2 = _run_prediction(args, predictor, output_dir, capture2, observed2, 2)
    observed_state2 = np.load(observed2["new_observed_state"])
    prediction_stats2 = _prediction_stats(str(prediction2["prediction_npz"]), observed_state2, float(args.tau))
    frame2_tree = _run_tree_pair(args, output_dir, capture2, observed2, prediction2, prediction_stats2)

    reference_frame1_sc = _compare_to_reference_sc(frame1_tree["sc_parts"], Path(args.reference_one_step_sc_dir).resolve())
    reference_frame2_no_pred = _compare_to_no_pred_frame2(
        frame2_tree["measured_parts"],
        Path(args.reference_no_pred_two_frame_dir).resolve(),
    )
    frame_sc_comparison = _frame1_vs_frame2_comparison(frame1_tree, frame2_tree)
    _save_json(output_dir / "frame001_vs_frame002_sc_tree_comparison.json", frame_sc_comparison)
    _write_frame_comparison_md(output_dir / "frame001_vs_frame002_sc_tree_comparison.md", frame_sc_comparison)

    prediction_checklist = _make_prediction_safety_checklist(args, [prediction1, prediction2])
    _save_json(output_dir / "source_protection_checklist.json", source_checklist)
    _write_source_protection_md(output_dir / "source_protection_checklist.md", source_checklist)
    _save_json(output_dir / "prediction_safety_checklist.json", prediction_checklist)
    _write_prediction_safety_md(output_dir / "prediction_safety_checklist.md", prediction_checklist)

    observed_ratio = {
        "prior_step000": observed1["prior_summary_before"],
        "frame001": observed1["updated_summary"],
        "frame002": observed2["updated_summary"],
        "frame001_delta_observed_count": observed1["delta_observed_count"],
        "frame001_delta_observed_ratio": observed1["delta_observed_ratio"],
        "frame002_delta_observed_count": observed2["delta_observed_count"],
        "frame002_delta_observed_ratio": observed2["delta_observed_ratio"],
        "measured_only_observed_state_updates": True,
        "prediction_used": False,
        "map_predict_used": str(args.prediction_mode) == "sim_dynamic",
    }
    _save_json(output_dir / "observed_ratio_two_frame.json", observed_ratio)

    hashes = {
        "episode_prior_observed_state": str(prior0_path),
        "episode_prior_sha256_before": observed1["prior_sha256_before"],
        "episode_prior_sha256_after": observed1["prior_sha256_after"],
        "episode_prior_hash_unchanged": observed1["prior_hash_unchanged"],
        "frame001_observed_state": observed1["new_observed_state"],
        "frame001_sha256_before_prediction_and_tree": observed1["new_observed_state_sha256"],
        "frame001_sha256_after_prediction_and_tree": sha256_file(observed1["new_observed_state"]),
        "frame001_prior_hash_unchanged_during_frame2_update": observed2["prior_hash_unchanged"],
        "frame002_observed_state": observed2["new_observed_state"],
        "frame002_sha256_before_prediction_and_tree": observed2["new_observed_state_sha256"],
        "frame002_sha256_after_prediction_and_tree": sha256_file(observed2["new_observed_state"]),
    }
    _save_json(output_dir / "observed_state_hashes.json", hashes)

    _save_two_frame_sc_path_plot(
        output_dir / "two_frame_sc_path_topdown.png",
        observed_state2,
        frame1_tree["sc_parts"],
        frame2_tree["sc_parts"],
    )

    _save_json(
        output_dir / "capture_scene_metadata.json",
        {
            "scene_variant": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "scene_metadata": scene_metadata,
            "frames_captured": 2,
            "selected_action_execution_count": 1,
            "prediction_used": True,
            "map_predict_used": str(args.prediction_mode) == "sim_dynamic",
            "rollout": False,
        },
    )

    checkpoint_after = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    external_after = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_after = _git_status_short(SSC_EXPLORATION_DIR)

    summary = _make_summary(
        args,
        output_dir,
        scene_metadata,
        capture1,
        capture2,
        observed1,
        observed2,
        prediction1,
        prediction2,
        prediction_stats1,
        prediction_stats2,
        frame1_tree,
        frame2_tree,
        reference_frame1_sc,
        reference_frame2_no_pred,
        frame_sc_comparison,
        source_checklist,
        prediction_checklist,
        checkpoint_before,
        checkpoint_after,
        external_before,
        external_after,
        ssc_before,
        ssc_after,
    )
    _save_json(output_dir / "map_predict_tree_two_frame_summary.json", summary)
    _write_summary_md(output_dir / "map_predict_tree_two_frame_summary.md", summary)
    _write_recommended_next(output_dir / "recommended_next_faithful_step.md", summary)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    try:
        run(args_cli)
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
