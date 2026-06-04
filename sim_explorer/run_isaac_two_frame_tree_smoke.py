#!/usr/bin/env python3
"""Stage 4A-6.5n Isaac two-frame source-protected tree smoke.

This runner starts Isaac once, captures exactly two RGB/depth frames in the
deterministic medium scene, updates measured-only observed maps, and runs the
source-protected no-prediction mini-RRT tree decision after each frame. It
executes exactly one selected action between the two frames by moving the
camera to the first decision's selected child pose. It does not run rollout,
map_predict, SSCNet inference/training, RL, PPO, BC, IL, prediction writeback,
or external source builds.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


PROFILE_NAME = "source_like_crop_min_length_0p25"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--selected_case_json", required=True)
parser.add_argument("--reference_one_step_dir", required=True)
parser.add_argument("--episode_dir", required=True)
parser.add_argument("--output_dir", required=True)
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
parser.add_argument("--gain_mode", choices=["exp"], default="exp")
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
parser.add_argument("--variant_name", default=f"{PROFILE_NAME}_two_frame")
parser.add_argument("--save_viz", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "headless"):
    args_cli.headless = True
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import shutil
import subprocess
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

from depth_to_voxel import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    create_observed_grid,
    normalize_map_bounds,
    summarize_observed_grid,
    update_observed_state_from_depth,
)
from offline_mini_rrt_tree import (
    run as run_mini_rrt,
    sha256_file,
    to_jsonable,
)
from scene_factory import build_medium_complex_scene


DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")
EXPECTED_FRAME1_SELECTED_ID = "n0001"
EXPECTED_FRAME1_SELECTED_GRID = [18, 12, 11]
EXPECTED_FRAME1_BEST_ID = "n0249"
EXPECTED_FRAME1_BEST_GRID = [39, 19, 11]
EXPECTED_FRAME1_GAIN = 645.0
EXPECTED_FRAME1_COST = 4.565369444959812
CAMERA_HEIGHT_M = 1.2
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)
SSC_EXPLORATION_DIR = Path("/home/ubuntu22/sc_explorer_ws/ssc_exploration")


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
        "source": "frame001_selected_child",
        "selected_child_segment_id": selected.get("segment_id"),
        "selected_child_grid": selected.get("end_grid"),
        "selected_child_world": selected.get("end_world"),
        "motion_mode": "planar_teleport_once_to_selected_child_xy_fixed_camera_height",
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
    stats = {
        "shape": [int(v) for v in depth.shape],
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()),
        "max": float(positive.max()),
        "mean": float(positive.mean()),
    }
    return depth, stats


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
    _save_depth_png(depth_png_path, depth, f"Isaac two-frame capture depth {frame_index:03d}")
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

    episode_dir = Path(args.episode_dir).resolve()
    saved_step_path = episode_dir / f"observed_state_step{frame_index:03d}.npy"
    saved_step_comparison: dict[str, Any] = {"available": False}
    if saved_step_path.is_file():
        saved = np.load(saved_step_path)
        saved_step_comparison = {
            "available": True,
            "path": str(saved_step_path),
            "sha256": sha256_file(saved_step_path),
            "same_shape": list(saved.shape) == list(updated.shape),
            "different_voxel_count": int(np.count_nonzero(saved != updated)) if saved.shape == updated.shape else None,
            "equal_to_saved_step": bool(np.array_equal(saved, updated)) if saved.shape == updated.shape else False,
            "saved_summary": summarize_observed_grid(saved),
        }

    return {
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
        "saved_step_comparison": saved_step_comparison,
        "measured_only": True,
        "prediction_used": False,
        "map_predict_used": False,
        "target_lr_target_hr_ground_truth_used": False,
        "prior_observed_state_modified": prior_hash_before != prior_hash_after,
    }


def _tree_args(
    args: argparse.Namespace,
    tree_dir: Path,
    observed_path: str,
    capture: dict[str, Any],
) -> argparse.Namespace:
    episode_dir = Path(args.episode_dir).resolve()
    return argparse.Namespace(
        case_json=str(Path(args.selected_case_json).resolve()),
        episode_dir=str(episode_dir),
        observed_state=str(observed_path),
        pose_json=str(capture["pose_path"]),
        camera_info=str(capture["camera_info_path"]),
        episode_summary=str(episode_dir / "episode_summary.json"),
        prediction_npz="",
        output_dir=str(tree_dir),
        seed=int(args.seed),
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        gain_mode=str(args.gain_mode),
        path_cost_mode=str(args.path_cost_mode),
        v_max=float(args.v_max),
        yaw_rate=1.0,
        robot_radius_m=float(args.robot_radius_m),
        voxel_size=float(args.voxel_size),
        raycast_stride=int(args.raycast_stride),
        num_yaw_samples=int(args.num_yaw_samples),
        max_ray_length_m=float(args.max_ray_length_m),
        tau=0.1,
        save_viz=bool(args.save_viz),
        profile=True,
        min_edge_length_m=float(args.min_edge_length_m),
        min_root_child_length_m=float(args.min_root_child_length_m),
        min_root_distance_m=float(args.min_root_distance_m),
        crop_min_length_m=float(args.crop_min_length_m),
        short_edge_policy=str(args.short_edge_policy),
        density_radius_m=float(args.density_radius_m),
        max_nodes_per_density_radius=int(args.max_nodes_per_density_radius),
        variant_name=f"{args.variant_name}_frame{capture['frame_index']:03d}",
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


def _is_nonlocal(parts: dict[str, Any]) -> bool:
    selected_distance = parts.get("selected_distance_m")
    best_distance = parts.get("best_distance_m")
    return bool(
        (selected_distance is not None and float(selected_distance) >= 0.5)
        or (best_distance is not None and float(best_distance) >= 1.0)
    )


def _copy_alias(src: Path, dst: Path) -> str:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def _alias_tree_outputs(output_dir: Path, tree_dir: Path, frame_index: int) -> dict[str, str]:
    prefix = f"frame{frame_index:03d}"
    aliases = {
        f"{prefix}_tree_segments.jsonl": _copy_alias(
            tree_dir / "mini_rrt_tree_segments.jsonl", output_dir / f"{prefix}_tree_segments.jsonl"
        ),
        f"{prefix}_gain_cost_value_table.csv": _copy_alias(
            tree_dir / "gain_cost_value_table.csv", output_dir / f"{prefix}_gain_cost_value_table.csv"
        ),
    }
    topdown = tree_dir / "mini_rrt_tree_topdown.png"
    if topdown.is_file():
        aliases[f"{prefix}_tree_topdown.png"] = _copy_alias(topdown, output_dir / f"{prefix}_tree_topdown.png")
    return aliases


def _make_source_protection_checklist(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile_name": PROFILE_NAME,
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
            "gain_mode": str(args.gain_mode),
            "path_cost_mode": str(args.path_cost_mode),
            "v_max": float(args.v_max),
            "robot_radius_m": float(args.robot_radius_m),
            "voxel_size": float(args.voxel_size),
            "raycast_stride": int(args.raycast_stride),
            "num_yaw_samples": int(args.num_yaw_samples),
            "max_ray_length_m": float(args.max_ray_length_m),
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
                "reason": "requires a multi-step planner/root change and is not part of this two-frame smoke",
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
            "enabled": False,
            "prediction_npz": None,
            "map_predict_used": False,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
        },
    }


def _write_source_protection_md(path: Path, checklist: dict[str, Any]) -> None:
    mech = checklist["mechanisms"]
    lines = [
        "# Source Protection Checklist",
        "",
        f"- profile: `{checklist['profile_name']}`",
        f"- variant: `{checklist['variant_name']}`",
        f"- crop_min_length / min_path_length: implemented `{mech['crop_min_length_min_path_length']['implemented']}`, active `{mech['crop_min_length_min_path_length']['active']}`, value `{mech['crop_min_length_min_path_length']['value_m']}` m.",
        f"- density limiting / max_density_range: implemented `{mech['density_limiting_max_density_range']['implemented']}`, active `{mech['density_limiting_max_density_range']['active']}`; {mech['density_limiting_max_density_range']['reason']}.",
        f"- continuous yaw: approximation implemented `{mech['continuous_yaw']['implemented_approximation']}`, active `{mech['continuous_yaw']['active']}`, samples `{mech['continuous_yaw']['num_yaw_samples']}`.",
        f"- root rewiring / reinsert: full implementation `{mech['root_rewiring_reinsert']['full_implementation']}`, active `{mech['root_rewiring_reinsert']['active']}`; {mech['root_rewiring_reinsert']['reason']}.",
        f"- optional parent visible clearing: active `{mech['optional_parent_visible_clearing']['active']}`; {mech['optional_parent_visible_clearing']['reason']}.",
        f"- root-visible filtering / near-root discount: active `{mech['root_visible_filtering_near_root_discount']['active']}`; {mech['root_visible_filtering_near_root_discount']['reason']}.",
        "- prediction / map_predict / writeback: `False` / `False` / `False`.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _load_reference_decision(reference_one_step_dir: Path) -> dict[str, Any]:
    decision_path = reference_one_step_dir / "source_protected_tree_decision.json"
    if not decision_path.is_file():
        raise FileNotFoundError(decision_path)
    payload = _load_json(decision_path)
    return {
        "path": str(decision_path),
        "selected": payload.get("selected_child") or {},
        "best": payload.get("best_descendant") or {},
        "selected_distance_m": payload.get("selected_child_distance_from_root_m"),
        "best_distance_m": payload.get("best_descendant_distance_from_root_m"),
        "value": payload.get("value"),
        "accumulated_gain": payload.get("accumulated_gain"),
        "accumulated_cost": payload.get("accumulated_cost"),
        "accepted_nodes": payload.get("accepted_nodes"),
        "rejected_samples": payload.get("rejected_samples"),
    }


def _compare_frame1(parts: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    selected = parts["selected"]
    best = parts["best"]
    ref_selected = reference["selected"]
    ref_best = reference["best"]
    exact = bool(
        selected.get("segment_id") == ref_selected.get("segment_id")
        and best.get("segment_id") == ref_best.get("segment_id")
        and _same_grid(selected.get("end_grid"), ref_selected.get("end_grid"))
        and _same_grid(best.get("end_grid"), ref_best.get("end_grid"))
        and _close(parts.get("selected_distance_m"), reference.get("selected_distance_m"))
        and _close(parts.get("best_distance_m"), reference.get("best_distance_m"))
        and _close(parts.get("value"), reference.get("value"))
        and _close(parts.get("accumulated_gain"), reference.get("accumulated_gain"))
        and _close(parts.get("accumulated_cost"), reference.get("accumulated_cost"))
    )
    expected_exact = bool(
        selected.get("segment_id") == EXPECTED_FRAME1_SELECTED_ID
        and _same_grid(selected.get("end_grid"), EXPECTED_FRAME1_SELECTED_GRID)
        and best.get("segment_id") == EXPECTED_FRAME1_BEST_ID
        and _same_grid(best.get("end_grid"), EXPECTED_FRAME1_BEST_GRID)
        and _close(parts.get("accumulated_gain"), EXPECTED_FRAME1_GAIN)
        and _close(parts.get("accumulated_cost"), EXPECTED_FRAME1_COST)
    )
    return {
        "matches_reference_one_step_exactly": exact,
        "matches_expected_stage4a65m_values": expected_exact,
        "selected_child_world_delta_m": _euclidean(selected.get("end_world"), ref_selected.get("end_world")),
        "best_descendant_world_delta_m": _euclidean(best.get("end_world"), ref_best.get("end_world")),
    }


def _decision_payload(
    frame_index: int,
    args: argparse.Namespace,
    parts: dict[str, Any],
    observed_update: dict[str, Any],
    capture: dict[str, Any],
    checklist: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": "Stage 4A-6.5n no-prediction two-frame source-protected tree smoke",
        "frame_index": int(frame_index),
        "profile_name": PROFILE_NAME,
        "variant_name": str(args.variant_name),
        "profile": checklist["profile_parameters"],
        "selected_child": parts["selected"],
        "selected_child_distance_from_root_m": parts["selected_distance_m"],
        "best_descendant": parts["best"],
        "best_descendant_distance_from_root_m": parts["best_distance_m"],
        "value": parts["value"],
        "accumulated_gain": parts["accumulated_gain"],
        "accumulated_cost": parts["accumulated_cost"],
        "accepted_nodes": parts["accepted_nodes"],
        "rejected_samples": parts["rejected_samples"],
        "built_successfully": parts["built_successfully"],
        "root": parts["root"],
        "raw_decision": parts["raw_decision"],
        "nonlocal_branch": _is_nonlocal(parts),
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
        "safety": {
            "prediction_used": False,
            "map_predict_used": False,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "target_lr_target_hr_ground_truth_scoring": False,
        },
        "extra": extra or {},
    }


def _write_decision_md(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_child"]
    best = payload["best_descendant"]
    lines = [
        f"# Frame {payload['frame_index']:03d} Source-Protected Tree Decision",
        "",
        f"- profile: `{payload['profile_name']}`",
        f"- selected child: `{selected.get('segment_id')}` grid `{selected.get('end_grid')}` world `{selected.get('end_world')}`",
        f"- selected child distance: `{payload['selected_child_distance_from_root_m']}` m",
        f"- best descendant: `{best.get('segment_id')}` grid `{best.get('end_grid')}` world `{best.get('end_world')}`",
        f"- best descendant distance: `{payload['best_descendant_distance_from_root_m']}` m",
        f"- value: `{payload['value']}`",
        f"- accumulated gain/cost: `{payload['accumulated_gain']}` / `{payload['accumulated_cost']}`",
        f"- accepted/rejected: `{payload['accepted_nodes']}` / `{payload['rejected_samples']}`",
        f"- nonlocal branch: `{payload['nonlocal_branch']}`",
        f"- observed delta count/ratio: `{payload['observed_state']['delta_observed_count']}` / `{payload['observed_state']['delta_observed_ratio']}`",
        "- prediction / map_predict used: `False` / `False`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _topdown_image(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def _save_two_frame_path_plot(
    path: Path,
    observed_state: np.ndarray,
    frame1_parts: dict[str, Any],
    frame2_parts: dict[str, Any],
) -> None:
    image = _topdown_image(observed_state)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")

    root1 = frame1_parts.get("root", {}).get("grid")
    sel1 = frame1_parts.get("selected", {}).get("end_grid")
    best1 = frame1_parts.get("best", {}).get("end_grid")
    root2 = frame2_parts.get("root", {}).get("grid")
    sel2 = frame2_parts.get("selected", {}).get("end_grid")
    best2 = frame2_parts.get("best", {}).get("end_grid")

    def point(grid: Any) -> tuple[float, float] | None:
        if grid is None:
            return None
        return float(grid[0]) + 0.5, float(grid[1]) + 0.5

    p_root1 = point(root1)
    p_sel1 = point(sel1)
    p_best1 = point(best1)
    p_root2 = point(root2)
    p_sel2 = point(sel2)
    p_best2 = point(best2)
    if p_root1:
        ax.scatter([p_root1[0]], [p_root1[1]], c="#ffffff", edgecolors="#111111", s=80, label="frame1 root")
    if p_sel1:
        ax.scatter([p_sel1[0]], [p_sel1[1]], c="#d1495b", s=70, label="executed child")
    if p_root1 and p_sel1:
        ax.plot([p_root1[0], p_sel1[0]], [p_root1[1], p_sel1[1]], color="#d1495b", linewidth=2.6)
    if p_root1 and p_best1:
        ax.plot([p_root1[0], p_best1[0]], [p_root1[1], p_best1[1]], color="#d1495b", linewidth=1.1, linestyle="--")
    if p_root2:
        ax.scatter([p_root2[0]], [p_root2[1]], c="#f4d35e", edgecolors="#111111", s=70, label="frame2 root")
    if p_sel2:
        ax.scatter([p_sel2[0]], [p_sel2[1]], c="#3f88c5", s=60, label="frame2 selected")
    if p_root2 and p_sel2:
        ax.plot([p_root2[0], p_sel2[0]], [p_root2[1], p_sel2[1]], color="#3f88c5", linewidth=2.0)
    if p_root2 and p_best2:
        ax.plot([p_root2[0], p_best2[0]], [p_root2[1], p_best2[1]], color="#3f88c5", linewidth=1.1, linestyle="--")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Two-frame source-protected path")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _scan_map_predict_artifacts(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ("*map_predict*", "*prediction*.npz", "*class_prob*", "*logits*.npy"):
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def _scan_prohibited_rollout_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in (
        "transitions.jsonl",
        "step_*.npz",
        "observed_ratio_curve.png",
        "rollout_topdown_path.png",
        "rollout_index.html",
        "step_topdown_*.png",
    ):
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def _make_summary(
    args: argparse.Namespace,
    output_dir: Path,
    scene_metadata: dict[str, Any],
    capture1: dict[str, Any],
    capture2: dict[str, Any],
    observed1: dict[str, Any],
    observed2: dict[str, Any],
    frame1_parts: dict[str, Any],
    frame2_parts: dict[str, Any],
    frame1_compare: dict[str, Any],
    checklist: dict[str, Any],
    checkpoint_before: str | None,
    checkpoint_after: str | None,
    external_before: str,
    external_after: str,
    ssc_before: str,
    ssc_after: str,
) -> dict[str, Any]:
    frame1_ok = bool(frame1_parts["built_successfully"] and frame1_compare["matches_reference_one_step_exactly"])
    frame2_ok = bool(frame2_parts["built_successfully"])
    frame2_nonlocal = _is_nonlocal(frame2_parts)
    measured_increment = int(observed2["delta_observed_count"]) > 0
    ready_map_predict_one_step = bool(
        frame1_ok
        and frame2_ok
        and frame2_nonlocal
        and measured_increment
        and observed1["prior_hash_unchanged"]
        and observed2["prior_hash_unchanged"]
        and checkpoint_before == checkpoint_after
        and external_before == external_after
    )
    if ready_map_predict_one_step:
        next_step = "map_predict + source-protected tree one-step smoke"
    elif not frame1_ok or not measured_increment:
        next_step = "depth-to-voxel two-frame replay debug"
    else:
        next_step = "two-frame sampling/min-length/yaw diagnosis"
    return {
        "stage": "Stage 4A-6.5n no-prediction two-frame source-protected tree smoke",
        "output_dir": str(output_dir),
        "scene": {
            "scene_variant": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "scene_metadata": scene_metadata,
        },
        "answers": {
            "frame1_capture_success": True,
            "frame1_tree_decision_reproduces_stage4a65m": frame1_compare["matches_reference_one_step_exactly"],
            "selected_child_executed_once": True,
            "frame2_capture_success": True,
            "observed_state_measured_only_increment_frame1_to_frame2": measured_increment,
            "frame2_tree_decision_success": frame2_ok,
            "frame2_still_finds_nonlocal_branch": frame2_nonlocal,
            "map_predict_or_prediction_used": False,
            "enough_for_map_predict_source_protected_tree_one_step_smoke": ready_map_predict_one_step,
            "ready_for_rollout": False,
        },
        "frames": {
            "frame001": {
                "capture": {k: v for k, v in capture1.items() if k != "depth"},
                "observed_state": observed1,
                "tree_decision": frame1_parts,
                "comparison_to_stage4a65m": frame1_compare,
            },
            "frame002": {
                "capture": {k: v for k, v in capture2.items() if k != "depth"},
                "observed_state": observed2,
                "tree_decision": frame2_parts,
            },
        },
        "move_once": {
            "executed_action": True,
            "execution_count": 1,
            "selected_child_segment_id": frame1_parts["selected"].get("segment_id"),
            "selected_child_grid": frame1_parts["selected"].get("end_grid"),
            "selected_child_world": frame1_parts["selected"].get("end_world"),
            "new_pose": capture2["pose"],
            "motion_mode": "planar_teleport_once_to_selected_child_xy_fixed_camera_height",
            "safety_note": "exactly one action was executed between frame 1 and frame 2; no rollout loop was entered",
        },
        "source_protection_checklist": checklist,
        "safety": {
            "isaac_startup": True,
            "frames_captured": 2,
            "selected_action_execution": True,
            "selected_action_execution_count": 1,
            "rollout": False,
            "online_open_ended_loop": False,
            "map_predict_rerun": False,
            "sscnet_inference_or_training": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_modified": checkpoint_before != checkpoint_after,
            "existing_observed_state_modified": not bool(observed1["prior_hash_unchanged"]),
            "new_observed_state_written_under_output_dir_only": True,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "target_lr_target_hr_ground_truth_scoring": False,
            "external_source_git_status_before": external_before,
            "external_source_git_status_after": external_after,
            "external_source_modified_or_built_by_stage": external_before != external_after,
            "ssc_exploration_git_status_before": ssc_before,
            "ssc_exploration_git_status_after": ssc_after,
            "ssc_exploration_modified_by_stage": ssc_before != ssc_after,
            "map_predict_artifacts_in_output_dir": _scan_map_predict_artifacts(output_dir),
            "prohibited_rollout_outputs": _scan_prohibited_rollout_outputs(output_dir),
            "coverage_improvement_claimed": False,
        },
        "recommended_next_faithful_step": next_step,
        "still_not_next": [
            "rollout",
            "RL/PPO/BC/IL training",
            "prediction writeback",
            "observed_map prediction fusion",
            "target/ground-truth scoring",
            "checkpoint changes",
            "external source build",
        ],
    }


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    f1 = summary["frames"]["frame001"]["tree_decision"]
    f2 = summary["frames"]["frame002"]["tree_decision"]
    lines = [
        "# Stage 4A-6.5n Two-Frame Tree Summary",
        "",
        f"1. frame 1 capture successful? `{answers['frame1_capture_success']}`.",
        f"2. frame 1 tree decision reproduced Stage 4A-6.5m? `{answers['frame1_tree_decision_reproduces_stage4a65m']}`.",
        f"3. selected child executed once? `{answers['selected_child_executed_once']}`.",
        f"4. frame 2 capture successful? `{answers['frame2_capture_success']}`.",
        f"5. measured-only observed_state increment from frame 1 to frame 2? `{answers['observed_state_measured_only_increment_frame1_to_frame2']}`.",
        f"6. frame 2 tree decision successful? `{answers['frame2_tree_decision_success']}`.",
        f"7. frame 2 still found a nonlocal branch? `{answers['frame2_still_finds_nonlocal_branch']}`.",
        f"8. map_predict / prediction used? `{answers['map_predict_or_prediction_used']}`.",
        f"9. enough for map_predict + source-protected tree one-step smoke? `{answers['enough_for_map_predict_source_protected_tree_one_step_smoke']}`.",
        f"10. ready for rollout? `{answers['ready_for_rollout']}`.",
        "",
        "## Frame 1",
        f"- selected child: `{f1['selected'].get('segment_id')}` grid `{f1['selected'].get('end_grid')}` world `{f1['selected'].get('end_world')}`.",
        f"- best descendant: `{f1['best'].get('segment_id')}` grid `{f1['best'].get('end_grid')}` world `{f1['best'].get('end_world')}`.",
        f"- value: `{f1['value']}`; accumulated gain/cost: `{f1['accumulated_gain']}` / `{f1['accumulated_cost']}`.",
        "",
        "## Move Once",
        f"- executed action: `{summary['move_once']['selected_child_segment_id']}` grid `{summary['move_once']['selected_child_grid']}`.",
        f"- new pose: `{summary['move_once']['new_pose']['position']}`, yaw `{summary['move_once']['new_pose']['yaw_rad']}`.",
        "",
        "## Frame 2",
        f"- selected child: `{f2['selected'].get('segment_id')}` grid `{f2['selected'].get('end_grid')}` world `{f2['selected'].get('end_world')}`.",
        f"- best descendant: `{f2['best'].get('segment_id')}` grid `{f2['best'].get('end_grid')}` world `{f2['best'].get('end_world')}`.",
        f"- value: `{f2['value']}`; accumulated gain/cost: `{f2['accumulated_gain']}` / `{f2['accumulated_cost']}`.",
        f"- observed map delta count/ratio: `{summary['frames']['frame002']['observed_state']['delta_observed_count']}` / `{summary['frames']['frame002']['observed_state']['delta_observed_ratio']}`.",
        "",
        f"Recommended next faithful step: {summary['recommended_next_faithful_step']}. Still no rollout.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_recommended_next(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    if answers["enough_for_map_predict_source_protected_tree_one_step_smoke"]:
        reason = "two-frame measured-only tree smoke succeeded and frame 2 remained nonlocal"
    elif not answers["observed_state_measured_only_increment_frame1_to_frame2"]:
        reason = "frame 2 did not add measured observed voxels over frame 1"
    else:
        reason = "frame 2 tree behavior did not remain a stable nonlocal branch"
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {summary['recommended_next_faithful_step']}",
        f"- reason: {reason}",
        "- still not next: rollout, RL/PPO/BC/IL training, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, or external source build.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_before = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_before = _git_status_short(SSC_EXPLORATION_DIR)
    checklist = _make_source_protection_checklist(args)

    sim, camera, scene_metadata = _start_scene(args)
    initial_pose = _initial_pose(args)
    capture1 = _capture_frame(args, output_dir, camera, sim, initial_pose, 1)

    episode_dir = Path(args.episode_dir).resolve()
    prior0_path = episode_dir / "observed_state_step000.npy"
    observed1 = _update_observed_state(args, output_dir, capture1, prior0_path, 1)

    tree1_dir = output_dir / "frame001_tree_raw"
    mini1 = run_mini_rrt(_tree_args(args, tree1_dir, observed1["new_observed_state"], capture1))
    frame1_parts = _decision_parts(mini1)
    reference = _load_reference_decision(Path(args.reference_one_step_dir).resolve())
    frame1_compare = _compare_frame1(frame1_parts, reference)
    aliases1 = _alias_tree_outputs(output_dir, tree1_dir, 1)

    pose2 = _pose_from_selected_child(frame1_parts["selected"])
    capture2 = _capture_frame(args, output_dir, camera, sim, pose2, 2)
    observed2 = _update_observed_state(args, output_dir, capture2, Path(observed1["new_observed_state"]), 2)

    tree2_dir = output_dir / "frame002_tree_raw"
    mini2 = run_mini_rrt(_tree_args(args, tree2_dir, observed2["new_observed_state"], capture2))
    frame2_parts = _decision_parts(mini2)
    aliases2 = _alias_tree_outputs(output_dir, tree2_dir, 2)

    checkpoint_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_after = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_after = _git_status_short(SSC_EXPLORATION_DIR)

    frame1_decision = _decision_payload(1, args, frame1_parts, observed1, capture1, checklist, frame1_compare)
    frame2_decision = _decision_payload(2, args, frame2_parts, observed2, capture2, checklist)
    _save_json(output_dir / "frame001_tree_decision.json", frame1_decision)
    _write_decision_md(output_dir / "frame001_tree_decision.md", frame1_decision)
    _save_json(output_dir / "frame002_tree_decision.json", frame2_decision)
    _write_decision_md(output_dir / "frame002_tree_decision.md", frame2_decision)

    _save_json(output_dir / "source_protection_checklist.json", checklist)
    _write_source_protection_md(output_dir / "source_protection_checklist.md", checklist)

    observed_ratio = {
        "prior_step000": observed1["prior_summary_before"],
        "frame001": observed1["updated_summary"],
        "frame002": observed2["updated_summary"],
        "frame001_delta_observed_count": observed1["delta_observed_count"],
        "frame001_delta_observed_ratio": observed1["delta_observed_ratio"],
        "frame002_delta_observed_count": observed2["delta_observed_count"],
        "frame002_delta_observed_ratio": observed2["delta_observed_ratio"],
        "measured_only": True,
        "prediction_used": False,
        "map_predict_used": False,
    }
    _save_json(output_dir / "observed_ratio_two_frame.json", observed_ratio)

    hashes = {
        "episode_prior_observed_state": str(prior0_path),
        "episode_prior_sha256_before": observed1["prior_sha256_before"],
        "episode_prior_sha256_after": observed1["prior_sha256_after"],
        "episode_prior_hash_unchanged": observed1["prior_hash_unchanged"],
        "frame001_observed_state": observed1["new_observed_state"],
        "frame001_sha256": observed1["new_observed_state_sha256"],
        "frame001_prior_hash_unchanged_during_frame2_update": observed2["prior_hash_unchanged"],
        "frame002_observed_state": observed2["new_observed_state"],
        "frame002_sha256": observed2["new_observed_state_sha256"],
    }
    _save_json(output_dir / "observed_state_hashes.json", hashes)

    final_observed = np.load(observed2["new_observed_state"])
    _save_two_frame_path_plot(output_dir / "two_frame_path_topdown.png", final_observed, frame1_parts, frame2_parts)

    _save_json(
        output_dir / "capture_scene_metadata.json",
        {
            "scene_variant": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "scene_metadata": scene_metadata,
            "frames_captured": 2,
            "selected_action_execution_count": 1,
            "prediction_used": False,
            "map_predict_used": False,
            "rollout": False,
        },
    )

    summary = _make_summary(
        args,
        output_dir,
        scene_metadata,
        capture1,
        capture2,
        observed1,
        observed2,
        frame1_parts,
        frame2_parts,
        frame1_compare,
        checklist,
        checkpoint_before,
        checkpoint_after,
        external_before,
        external_after,
        ssc_before,
        ssc_after,
    )
    summary["generated_aliases"] = {**aliases1, **aliases2}
    _save_json(output_dir / "two_frame_tree_summary.json", summary)
    _write_summary_md(output_dir / "two_frame_tree_summary.md", summary)
    _write_recommended_next(output_dir / "recommended_next_faithful_step.md", summary)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    try:
        run(args_cli)
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
