#!/usr/bin/env python3
"""Stage 4A-6.5o map_predict + source-protected tree one-step smoke.

This runner starts Isaac once, captures exactly one RGB/depth frame in the
deterministic medium scene, updates a new measured-only observed_state, runs
SSCNet map_predict read-only, and evaluates one source-protected mini-RRT tree
decision with prediction used only for information gain. It does not execute
the selected action, run rollout/two-frame loops, train, write prediction into
observed_state, or use prediction for traversability/collision/ray blocking.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


PROFILE_NAME = "source_like_crop_min_length_0p25"
DEFAULT_SELECTED_CASE = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65c_decoupled_one_step_smoke/selected_case.json"
)
DEFAULT_REFERENCE_ONE_STEP_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65l_source_protected_one_step_tree_smoke"
)
DEFAULT_EPISODE_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_medium_rollout_sc_pred_alignment_fixed_smoke/episodes/"
    "medium_three_rooms_seed0_start_room_a_sc_pred_alignment_fixed_000"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65o_map_predict_tree_one_step_smoke"
)
DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--selected_case_json", default=DEFAULT_SELECTED_CASE)
parser.add_argument("--reference_one_step_dir", default=DEFAULT_REFERENCE_ONE_STEP_DIR)
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
parser.add_argument("--sc_gain_formula", choices=["raw_count"], default="raw_count")
parser.add_argument("--alignment_convention", choices=["code_consistent_v1"], default="code_consistent_v1")
parser.add_argument("--torch_num_threads", type=int, default=8)
parser.add_argument("--variant_name", default=f"{PROFILE_NAME}_map_predict_one_step")
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
import time
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
) -> dict[str, Any]:
    prefix = "frame001"
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
    _save_depth_png(depth_png_path, depth, "Isaac one-frame map_predict tree capture")
    Image.fromarray(rgb).save(rgb_path)
    _save_json(camera_info_path, camera_info)

    return {
        "frame_index": 1,
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
) -> dict[str, Any]:
    bounds = _episode_bounds(args)
    fallback_empty = False
    if prior_path.is_file():
        prior = np.load(prior_path)
    else:
        fallback_empty = True
        prior = create_observed_grid(bounds, voxel_size=float(args.voxel_size))
        prior_path = output_dir / "observed_state_empty_fallback.npy"
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

    output_observed = output_dir / "observed_state_frame001.npy"
    np.save(output_observed, updated)
    updated_summary = summarize_observed_grid(updated)
    delta_observed_count = int(updated_summary["observed_count"] - prior_summary_before["observed_count"])
    delta_observed_ratio = float(updated_summary["observed_ratio"] - prior_summary_before["observed_ratio"])

    return {
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
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _tree_gain_stats(tree_dir: Path) -> dict[str, Any]:
    segments = _load_segments(tree_dir / "mini_rrt_tree_segments.jsonl")
    non_root = [row for row in segments if row.get("segment_id") != "root"]
    positive_sc = [row for row in non_root if float(row.get("gain_sc", 0.0) or 0.0) > 0.0]
    positive_occ = [row for row in non_root if float(row.get("gain_occ", 0.0) or 0.0) > 0.0]
    positive_conf = [row for row in non_root if float(row.get("gain_conf", 0.0) or 0.0) > 0.0]
    return {
        "node_count_excluding_root": int(len(non_root)),
        "nodes_with_gain_sc_positive": int(len(positive_sc)),
        "nodes_with_gain_occ_positive": int(len(positive_occ)),
        "nodes_with_gain_conf_positive": int(len(positive_conf)),
        "gain_sc_min_mean_max": _min_mean_max([float(row.get("gain_sc", 0.0) or 0.0) for row in non_root]),
        "gain_exp_min_mean_max": _min_mean_max([float(row.get("gain_exp", 0.0) or 0.0) for row in non_root]),
        "gain_hybrid_min_mean_max": _min_mean_max([float(row.get("gain_hybrid", 0.0) or 0.0) for row in non_root]),
        "gain_occ_min_mean_max": _min_mean_max([float(row.get("gain_occ", 0.0) or 0.0) for row in non_root]),
        "gain_conf_min_mean_max": _min_mean_max([float(row.get("gain_conf", 0.0) or 0.0) for row in non_root]),
        "positive_gain_sc_segment_ids_sample": [str(row.get("segment_id")) for row in positive_sc[:16]],
    }


def _min_mean_max(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(finite.min()), "mean": float(finite.mean()), "max": float(finite.max())}


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


def _alias_tree_outputs(output_dir: Path, tree_dir: Path, prefix: str) -> dict[str, str]:
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


def _make_source_protection_checklist(args: argparse.Namespace, prediction_npz: str | None) -> dict[str, Any]:
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
                "reason": "not part of this one-step smoke",
            },
        },
        "prediction": {
            "enabled": True,
            "prediction_mode": str(args.prediction_mode),
            "prediction_npz": prediction_npz,
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
        f"- crop_min_length / min_path_length: implemented `{mech['crop_min_length_min_path_length']['implemented']}`, active `{mech['crop_min_length_min_path_length']['active']}`, value `{mech['crop_min_length_min_path_length']['value_m']}` m.",
        f"- density limiting / max_density_range: implemented `{mech['density_limiting_max_density_range']['implemented']}`, active `{mech['density_limiting_max_density_range']['active']}`.",
        f"- continuous yaw: approximation implemented `{mech['continuous_yaw']['implemented_approximation']}`, active `{mech['continuous_yaw']['active']}`, samples `{mech['continuous_yaw']['num_yaw_samples']}`.",
        f"- prediction enabled for information gain only: `{pred['prediction_used_for_information_gain_only']}`.",
        f"- map_predict used: `{pred['map_predict_used']}`.",
        "- prediction writeback / traversability / collision / ray blocking: `False` / `False` / `False` / `False`.",
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
) -> None:
    image = _topdown_image(observed_state)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")

    root = _grid_point(sc_tree.get("root", {}).get("grid") or measured.get("root", {}).get("grid"))
    measured_selected = _grid_point(measured.get("selected", {}).get("end_grid"))
    measured_best = _grid_point(measured.get("best", {}).get("end_grid"))
    sc_selected = _grid_point(sc_tree.get("selected", {}).get("end_grid"))
    sc_best = _grid_point(sc_tree.get("best", {}).get("end_grid"))

    if root:
        ax.scatter([root[0]], [root[1]], c="#ffffff", edgecolors="#111111", s=90, label="root")
    if measured_selected:
        ax.scatter([measured_selected[0]], [measured_selected[1]], c="#f97316", s=70, label="measured selected")
    if measured_best:
        ax.scatter([measured_best[0]], [measured_best[1]], c="#fb923c", marker="*", s=130, label="measured best")
    if sc_selected:
        ax.scatter([sc_selected[0]], [sc_selected[1]], c="#3b82f6", s=80, label="SC selected")
    if sc_best:
        ax.scatter([sc_best[0]], [sc_best[1]], c="#2563eb", marker="*", s=150, label="SC best")
    for point, color, style in (
        (measured_selected, "#f97316", "-"),
        (measured_best, "#f97316", "--"),
        (sc_selected, "#3b82f6", "-"),
        (sc_best, "#3b82f6", "--"),
    ):
        if root and point:
            ax.plot([root[0], point[0]], [root[1], point[1]], color=color, linestyle=style, linewidth=1.8)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Measured-only vs SC source-protected tree")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_prediction_unmeasured_topdown(
    path: Path,
    observed_state: np.ndarray,
    prediction_npz: str,
    tau: float,
) -> None:
    with np.load(prediction_npz, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
    predicted_unmeasured = valid & (confidence >= float(tau)) & (observed_state == UNKNOWN)
    topdown = np.any(predicted_unmeasured, axis=2)
    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", cmap="magma", interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Predicted unmeasured voxels, topdown")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _scan_prohibited_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in (
        "transitions.jsonl",
        "step_*.npz",
        "step_topdown_*.png",
        "observed_ratio_curve.png",
        "rollout_topdown_path.png",
        "rollout_index.html",
        "frame002*",
        "frame003*",
    ):
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def _run_prediction(args: argparse.Namespace, output_dir: Path, capture: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    observed_state = np.load(observed["new_observed_state"])
    if str(args.prediction_mode) == "sim_npz":
        if not args.prediction_npz:
            raise ValueError("--prediction_npz is required when --prediction_mode sim_npz")
        prediction_npz = str(Path(args.prediction_npz).resolve())
        layer = SimPredictionLayer.from_npz(prediction_npz)
        return {
            "prediction_layer": layer,
            "prediction_npz": prediction_npz,
            "global_prediction_npz": prediction_npz,
            "local_prediction_npz": None,
            "summary": {"map_predict_executed": False, "prediction_npz": prediction_npz},
            "summary_json": None,
            "timing": {},
            "model_loaded_once": False,
            "checkpoint_unchanged": True,
            "map_predict_success": False,
        }

    predictor = IsaacMapPredictor(
        checkpoint=args.checkpoint,
        device="cuda",
        tau=float(args.tau),
        torch_num_threads=int(args.torch_num_threads),
        alignment_convention=str(args.alignment_convention),
    )
    predict_dir = output_dir / "map_predict"
    result = predictor.predict_step(
        depth=np.asarray(capture["depth"], dtype=np.float32),
        pose=capture["pose"],
        camera_info=capture["camera_info"],
        observed_state=observed_state,
        map_bounds=observed["bounds"],
        voxel_size=float(args.voxel_size),
        output_dir=predict_dir,
        step=1,
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
        "measured_selected_child": measured_selected,
        "sc_selected_child": sc_selected,
        "measured_best_descendant": measured_best,
        "sc_best_descendant": sc_best,
    }


def _make_summary(
    args: argparse.Namespace,
    output_dir: Path,
    scene_metadata: dict[str, Any],
    capture: dict[str, Any],
    observed: dict[str, Any],
    prediction_result: dict[str, Any],
    prediction_stats: dict[str, Any],
    measured_summary: dict[str, Any],
    sc_summary: dict[str, Any],
    measured_parts: dict[str, Any],
    sc_parts: dict[str, Any],
    measured_gain_stats: dict[str, Any],
    sc_gain_stats: dict[str, Any],
    comparison: dict[str, Any],
    checklist: dict[str, Any],
    checkpoint_before: str | None,
    checkpoint_after: str | None,
    external_before: str,
    external_after: str,
    ssc_before: str,
    ssc_after: str,
    generated_aliases: dict[str, str],
) -> dict[str, Any]:
    observed_state_path = Path(observed["new_observed_state"])
    observed_hash_after_all = sha256_file(observed_state_path)
    sc_gain_nonzero = int(sc_gain_stats["nodes_with_gain_sc_positive"]) > 0
    hybrid_identity = _check_hybrid_identity(Path(output_dir / "sc_tree_gain_cost_value_table.csv"))
    enough_for_two_frame = bool(
        capture["depth_stats"]["positive_count"] > 0
        and int(observed["delta_observed_count"]) > 0
        and bool(prediction_result["map_predict_success"])
        and bool(prediction_stats["shape_aligned_to_observed_state"])
        and bool(sc_summary.get("tree", {}).get("built_successfully"))
        and sc_gain_nonzero
        and hybrid_identity["passed"]
        and observed["new_observed_state_sha256"] == observed_hash_after_all
        and not checklist["prediction"]["prediction_writeback"]
        and not checklist["prediction"]["prediction_used_for_collision_traversability"]
        and not checklist["prediction"]["prediction_blocks_rays"]
        and checkpoint_before == checkpoint_after
    )
    return {
        "stage": "Stage 4A-6.5o map_predict + source-protected tree one-step smoke",
        "output_dir": str(output_dir),
        "profile_name": PROFILE_NAME,
        "scene": {
            "scene_variant": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "scene_metadata": scene_metadata,
        },
        "answers": {
            "isaac_one_frame_capture_success": True,
            "measured_only_observed_state_update_success": int(observed["delta_observed_count"]) > 0,
            "map_predict_success": bool(prediction_result["map_predict_success"]),
            "prediction_layer_shape_aligned_to_observed_state": bool(
                prediction_stats["shape_aligned_to_observed_state"]
            ),
            "source_protected_tree_ran_with_prediction_mode": bool(sc_summary.get("tree", {}).get("built_successfully")),
            "tree_prediction_mode": str(args.prediction_mode),
            "gain_sc_nonzero": sc_gain_nonzero,
            "gain_hybrid_formula": "gain_hybrid = gain_exp + gain_sc",
            "gain_hybrid_identity_passed": bool(hybrid_identity["passed"]),
            "selected_child_differs_from_measured_only_tree": bool(
                comparison["selected_child_differs_from_measured_only"]
            ),
            "selected_child_change_spatially_meaningful": bool(
                comparison["selected_child_spatially_meaningful_change"]
            ),
            "prediction_did_not_write_observed_state": observed["new_observed_state_sha256"] == observed_hash_after_all,
            "prediction_used_for_traversability_collision_ray_blocking": False,
            "enough_for_map_predict_source_protected_tree_two_frame_smoke": enough_for_two_frame,
            "ready_for_rollout": False,
        },
        "capture": {k: v for k, v in capture.items() if k != "depth"},
        "observed_state": {
            **observed,
            "new_observed_state_sha256_after_all": observed_hash_after_all,
            "new_observed_state_unchanged_after_prediction_and_tree": observed["new_observed_state_sha256"]
            == observed_hash_after_all,
        },
        "prediction": {
            "mode": str(args.prediction_mode),
            "map_predict_summary": prediction_result.get("summary", {}),
            "map_predict_summary_json": prediction_result.get("summary_json"),
            "prediction_npz": prediction_result.get("prediction_npz"),
            "global_prediction_npz": prediction_result.get("global_prediction_npz"),
            "local_prediction_npz": prediction_result.get("local_prediction_npz"),
            "model_loaded_once": bool(prediction_result.get("model_loaded_once", False)),
            "steps_predicted": int(prediction_result.get("steps_predicted", 0)),
            "checkpoint_unchanged_according_to_predictor": bool(prediction_result.get("checkpoint_unchanged", False)),
            "timing": prediction_result.get("timing", {}),
            "stats": prediction_stats,
        },
        "trees": {
            "measured_only": {
                "tree_dir": str(output_dir / "measured_only_tree_raw"),
                "summary": measured_summary,
                "decision": measured_parts,
                "gain_stats": measured_gain_stats,
            },
            "sc_tree": {
                "tree_dir": str(output_dir / "sc_tree_raw"),
                "summary": sc_summary,
                "decision": sc_parts,
                "gain_stats": sc_gain_stats,
            },
            "comparison": comparison,
            "hybrid_identity": hybrid_identity,
        },
        "source_protection_checklist": checklist,
        "safety": {
            "isaac_startup": True,
            "frames_captured": 1,
            "two_frame": False,
            "selected_action_execution": False,
            "selected_action_execution_count": 0,
            "rollout": False,
            "online_open_ended_loop": False,
            "map_predict_used": str(args.prediction_mode) == "sim_dynamic",
            "sscnet_inference": str(args.prediction_mode) == "sim_dynamic",
            "sscnet_training": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_modified": checkpoint_before != checkpoint_after,
            "existing_observed_state_modified": not bool(observed["prior_hash_unchanged"]),
            "new_observed_state_written_under_output_dir_only": True,
            "prediction_writeback": False,
            "prediction_used_for_traversability_collision": False,
            "prediction_blocks_rays": False,
            "target_lr_target_hr_ground_truth_scoring": False,
            "external_source_git_status_before": external_before,
            "external_source_git_status_after": external_after,
            "external_source_modified_or_built_by_stage": external_before != external_after,
            "ssc_exploration_git_status_before": ssc_before,
            "ssc_exploration_git_status_after": ssc_after,
            "ssc_exploration_modified_by_stage": ssc_before != ssc_after,
            "prohibited_rollout_or_two_frame_outputs": _scan_prohibited_outputs(output_dir),
            "coverage_improvement_claimed": False,
        },
        "generated_aliases": generated_aliases,
        "recommended_next_faithful_step": (
            "map_predict + source-protected tree two-frame smoke"
            if enough_for_two_frame
            else "inspect map_predict tree one-step gain/selection diagnostics"
        ),
        "still_not_next": [
            "rollout",
            "online open-ended loop",
            "RL/PPO/BC/IL training",
            "prediction writeback",
            "observed_map prediction fusion",
            "target/ground-truth scoring",
            "checkpoint changes",
            "external source build",
        ],
    }


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


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    comparison = summary["trees"]["comparison"]
    sc = summary["trees"]["sc_tree"]["decision"]
    measured = summary["trees"]["measured_only"]["decision"]
    lines = [
        "# Stage 4A-6.5o Map Predict Tree One-Step Summary",
        "",
        f"1. Isaac one-frame capture successful? `{answers['isaac_one_frame_capture_success']}`.",
        f"2. measured-only observed_state update successful? `{answers['measured_only_observed_state_update_success']}`.",
        f"3. map_predict successful? `{answers['map_predict_success']}`.",
        f"4. prediction shape aligned? `{answers['prediction_layer_shape_aligned_to_observed_state']}`.",
        f"5. source-protected tree ran with prediction? `{answers['source_protected_tree_ran_with_prediction_mode']}`.",
        f"6. gain_sc nonzero? `{answers['gain_sc_nonzero']}`.",
        f"7. hybrid formula: `{answers['gain_hybrid_formula']}`, identity passed `{answers['gain_hybrid_identity_passed']}`.",
        f"8. selected child differs from measured-only? `{answers['selected_child_differs_from_measured_only_tree']}`.",
        f"9. selected change spatially meaningful? `{answers['selected_child_change_spatially_meaningful']}`.",
        f"10. prediction did not write observed_state? `{answers['prediction_did_not_write_observed_state']}`.",
        f"11. prediction used for traversability/collision/ray blocking? `{answers['prediction_used_for_traversability_collision_ray_blocking']}`.",
        f"12. enough for two-frame smoke? `{answers['enough_for_map_predict_source_protected_tree_two_frame_smoke']}`.",
        f"13. ready for rollout? `{answers['ready_for_rollout']}`.",
        "",
        "## Measured-Only Tree",
        f"- selected child: `{measured['selected'].get('segment_id')}` grid `{measured['selected'].get('end_grid')}` world `{measured['selected'].get('end_world')}`.",
        f"- best descendant: `{measured['best'].get('segment_id')}` grid `{measured['best'].get('end_grid')}` world `{measured['best'].get('end_world')}`.",
        f"- value: `{measured['value']}`; accumulated gain/cost: `{measured['accumulated_gain']}` / `{measured['accumulated_cost']}`.",
        "",
        "## SC Tree",
        f"- selected child: `{sc['selected'].get('segment_id')}` grid `{sc['selected'].get('end_grid')}` world `{sc['selected'].get('end_world')}`.",
        f"- best descendant: `{sc['best'].get('segment_id')}` grid `{sc['best'].get('end_grid')}` world `{sc['best'].get('end_world')}`.",
        f"- selected gains exp/sc/hybrid/occ/conf: `{sc['selected'].get('gain_exp')}` / `{sc['selected'].get('gain_sc')}` / `{sc['selected'].get('gain_hybrid')}` / `{sc['selected'].get('gain_occ')}` / `{sc['selected'].get('gain_conf')}`.",
        f"- best gains exp/sc/hybrid/occ/conf: `{sc['best'].get('gain_exp')}` / `{sc['best'].get('gain_sc')}` / `{sc['best'].get('gain_hybrid')}` / `{sc['best'].get('gain_occ')}` / `{sc['best'].get('gain_conf')}`.",
        f"- value: `{sc['value']}`; accumulated gain/cost: `{sc['accumulated_gain']}` / `{sc['accumulated_cost']}`.",
        "",
        "## Comparison",
        f"- selected child world delta: `{comparison['selected_child_world_delta_m']}` m.",
        f"- best descendant world delta: `{comparison['best_descendant_world_delta_m']}` m.",
        f"- prediction valid / predicted-unmeasured count: `{summary['prediction']['stats']['prediction_valid_count']}` / `{summary['prediction']['stats']['predicted_unmeasured_count']}`.",
        f"- nodes with gain_sc > 0: `{summary['trees']['sc_tree']['gain_stats']['nodes_with_gain_sc_positive']}`.",
        "",
        f"Recommended next faithful step: {summary['recommended_next_faithful_step']}. Still no rollout.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Measured-Only Vs SC Tree Comparison",
        "",
        f"- selected child differs: `{comparison['selected_child_differs_from_measured_only']}`",
        f"- selected child world delta m: `{comparison['selected_child_world_delta_m']}`",
        f"- selected child spatially meaningful: `{comparison['selected_child_spatially_meaningful_change']}`",
        f"- best descendant differs: `{comparison['best_descendant_differs_from_measured_only']}`",
        f"- best descendant world delta m: `{comparison['best_descendant_world_delta_m']}`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_recommended_next(path: Path, summary: dict[str, Any]) -> None:
    if summary["answers"]["enough_for_map_predict_source_protected_tree_two_frame_smoke"]:
        reason = "map_predict connected read-only to source-protected tree, gain_sc was nonzero, and safety checks passed"
    else:
        reason = "one or more one-step map_predict/tree diagnostics need inspection before two-frame"
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {summary['recommended_next_faithful_step']}",
        f"- reason: {reason}",
        "- still not next: rollout, online open-ended loop, RL/PPO/BC/IL training, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, or external source build.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_before = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    external_before = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_before = _git_status_short(SSC_EXPLORATION_DIR)

    sim, camera, scene_metadata = _start_scene(args)
    initial_pose = _initial_pose(args)
    capture = _capture_frame(args, output_dir, camera, sim, initial_pose)

    episode_dir = Path(args.episode_dir).resolve()
    prior0_path = episode_dir / "observed_state_step000.npy"
    observed = _update_observed_state(args, output_dir, capture, prior0_path)
    _save_json(output_dir / "observed_state_update_summary.json", observed)

    prediction_result = _run_prediction(args, output_dir, capture, observed)
    observed_state = np.load(observed["new_observed_state"])
    prediction_stats = _prediction_stats(prediction_result["prediction_npz"], observed_state, float(args.tau))
    _save_json(output_dir / "prediction_stats.json", prediction_stats)

    measured_tree_dir = output_dir / "measured_only_tree_raw"
    measured_summary = run_mini_rrt(
        _tree_args(
            args,
            measured_tree_dir,
            observed["new_observed_state"],
            capture,
            prediction_npz="",
            gain_mode=str(args.baseline_gain_mode),
            variant_suffix="measured_only",
        )
    )
    measured_parts = _decision_parts(measured_summary)
    measured_gain_stats = _tree_gain_stats(measured_tree_dir)

    sc_tree_dir = output_dir / "sc_tree_raw"
    sc_summary = run_mini_rrt(
        _tree_args(
            args,
            sc_tree_dir,
            observed["new_observed_state"],
            capture,
            prediction_npz=str(prediction_result["prediction_npz"]),
            gain_mode=str(args.gain_mode),
            variant_suffix=f"{args.prediction_mode}_{args.gain_mode}",
        )
    )
    sc_parts = _decision_parts(sc_summary)
    sc_gain_stats = _tree_gain_stats(sc_tree_dir)

    generated_aliases = {}
    generated_aliases.update(_alias_tree_outputs(output_dir, measured_tree_dir, "measured_only"))
    generated_aliases.update(_alias_tree_outputs(output_dir, sc_tree_dir, "sc_tree"))
    comparison = _compare_decisions(measured_parts, sc_parts)
    checklist = _make_source_protection_checklist(args, str(prediction_result["prediction_npz"]))

    _save_json(
        output_dir / "measured_only_tree_decision.json",
        {"decision": measured_parts, "gain_stats": measured_gain_stats},
    )
    _save_json(output_dir / "sc_tree_decision.json", {"decision": sc_parts, "gain_stats": sc_gain_stats})
    _save_json(output_dir / "tree_decision_comparison.json", comparison)
    _write_comparison_md(output_dir / "tree_decision_comparison.md", comparison)
    _save_json(output_dir / "source_protection_checklist.json", checklist)
    _write_source_protection_md(output_dir / "source_protection_checklist.md", checklist)

    _save_tree_comparison_plot(output_dir / "tree_decision_comparison_topdown.png", observed_state, measured_parts, sc_parts)
    _save_prediction_unmeasured_topdown(
        output_dir / "predicted_unmeasured_visible_topdown.png",
        observed_state,
        str(prediction_result["prediction_npz"]),
        float(args.tau),
    )

    checkpoint_after = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    external_after = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_after = _git_status_short(SSC_EXPLORATION_DIR)

    hashes = {
        "episode_prior_observed_state": str(prior0_path),
        "episode_prior_sha256_before": observed["prior_sha256_before"],
        "episode_prior_sha256_after": observed["prior_sha256_after"],
        "episode_prior_hash_unchanged": observed["prior_hash_unchanged"],
        "new_observed_state": observed["new_observed_state"],
        "new_observed_state_sha256_before_prediction_and_tree": observed["new_observed_state_sha256"],
        "new_observed_state_sha256_after_prediction_and_tree": sha256_file(observed["new_observed_state"]),
    }
    _save_json(output_dir / "observed_state_hashes.json", hashes)

    _save_json(
        output_dir / "capture_scene_metadata.json",
        {
            "scene_variant": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "scene_metadata": scene_metadata,
            "frames_captured": 1,
            "selected_action_execution_count": 0,
            "prediction_used": True,
            "map_predict_used": str(args.prediction_mode) == "sim_dynamic",
            "rollout": False,
            "two_frame": False,
        },
    )

    summary = _make_summary(
        args,
        output_dir,
        scene_metadata,
        capture,
        observed,
        prediction_result,
        prediction_stats,
        measured_summary,
        sc_summary,
        measured_parts,
        sc_parts,
        measured_gain_stats,
        sc_gain_stats,
        comparison,
        checklist,
        checkpoint_before,
        checkpoint_after,
        external_before,
        external_after,
        ssc_before,
        ssc_after,
        generated_aliases,
    )
    _save_json(output_dir / "map_predict_tree_one_step_summary.json", summary)
    _write_summary_md(output_dir / "map_predict_tree_one_step_summary.md", summary)
    _write_recommended_next(output_dir / "recommended_next_faithful_step.md", summary)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    try:
        run(args_cli)
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
