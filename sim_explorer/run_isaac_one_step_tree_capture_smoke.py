#!/usr/bin/env python3
"""Stage 4A-6.5m Isaac one-frame capture plus source-protected tree smoke.

This runner starts Isaac once, captures a single RGB/depth frame in the
deterministic medium scene, fuses that measured depth into a new observed map,
and runs the no-prediction source-protected mini-RRT tree decision on that new
map. It does not execute the selected action, run rollout, call map_predict,
run SSCNet inference, train, modify checkpoints, or modify existing observed
state files.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


PROFILE_NAME = "source_like_crop_min_length_0p25"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--selected_case_json", required=True)
parser.add_argument("--reference_tree_dir", required=True)
parser.add_argument("--episode_dir", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--scene_variant", default="medium_three_rooms")
parser.add_argument("--scene_seed", type=int, default=0)
parser.add_argument("--capture_step", type=int, default=1)
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
parser.add_argument("--variant_name", default=f"{PROFILE_NAME}_isaac_capture")
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
    save_json,
    sha256_file,
    to_jsonable,
)
from scene_factory import build_medium_complex_scene


DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")
OLD_SHORT_EDGE_CHILD_ID = "n0140"
ONE_STEP_BASELINE_GRID = [15, 16, 11]
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


def _resolve_capture_pose(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], Path | None]:
    episode_dir = Path(args.episode_dir).resolve()
    pose_path = episode_dir / f"pose_{int(args.capture_step):03d}.json"
    if pose_path.is_file():
        pose = _load_json(pose_path)
        source = pose_path
    else:
        selected_case = _load_json(Path(args.selected_case_json).resolve())
        pose_candidate = selected_case.get("pose_json") or selected_case.get("pose")
        if pose_candidate and Path(str(pose_candidate)).is_file():
            source = Path(str(pose_candidate)).resolve()
            pose = _load_json(source)
        else:
            source = None
            pose = {
                "index": int(args.capture_step),
                "position": [-4.65, -4.65, 1.2],
                "yaw_rad": 0.38710316317995463,
                "yaw_deg": math.degrees(0.38710316317995463),
                "fallback_pose": True,
            }

    position = [float(v) for v in pose["position"]]
    yaw_rad = float(pose.get("yaw_rad", math.radians(float(pose.get("yaw_deg", 0.0)))))
    pose_record = {
        "index": int(args.capture_step),
        "position": position,
        "yaw_rad": yaw_rad,
        "yaw_deg": float(math.degrees(yaw_rad)),
        "target": _pose_target(position, yaw_rad),
        "source_pose_file": str(source) if source is not None else None,
        "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
    }
    if source is None:
        pose_record["fallback_pose_used"] = True
    _save_json(output_dir / f"capture_pose_{int(args.capture_step):03d}.json", pose_record)
    return pose_record, source


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


def _save_depth_png(path: Path, depth: np.ndarray) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError("cannot save depth PNG: no positive finite depth")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_title("Isaac one-frame capture depth")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scene_variant_name(scene_variant: str) -> str:
    if scene_variant in {"medium_three_rooms", "three_rooms"}:
        return "three_rooms"
    raise ValueError(f"unsupported scene_variant: {scene_variant}")


def _capture_one_frame(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
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
    pose_record, pose_source = _resolve_capture_pose(args, output_dir)
    camera = _make_camera(args)
    sim.reset()

    _set_camera_pose(camera, sim, pose_record)
    _settle(camera, sim, int(args.settle_steps))
    depth, depth_stats = _extract_depth(camera)
    rgb, rgb_key, rgb_stats = _extract_rgb(camera)
    camera_info = _camera_info(camera, args)

    step = int(args.capture_step)
    depth_path = output_dir / f"capture_depth_{step:03d}.npy"
    depth_png_path = output_dir / f"capture_depth_{step:03d}.png"
    rgb_path = output_dir / f"capture_rgb_{step:03d}.png"
    camera_info_path = output_dir / "capture_camera_info.json"
    np.save(depth_path, depth)
    _save_depth_png(depth_png_path, depth)
    Image.fromarray(rgb).save(rgb_path)
    _save_json(camera_info_path, camera_info)
    _save_json(
        output_dir / "capture_scene_metadata.json",
        {
            "scene_variant": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "scene_metadata": scene_metadata,
            "single_frame_capture_only": True,
            "pose_source": str(pose_source) if pose_source is not None else None,
            "prediction_used": False,
            "map_predict_used": False,
            "rollout": False,
        },
    )
    return {
        "depth": depth,
        "depth_path": str(depth_path),
        "depth_png_path": str(depth_png_path),
        "rgb_path": str(rgb_path),
        "camera_info": camera_info,
        "camera_info_path": str(camera_info_path),
        "pose": pose_record,
        "pose_path": str(output_dir / f"capture_pose_{step:03d}.json"),
        "depth_stats": depth_stats,
        "rgb_stats": rgb_stats,
        "rgb_key_used": rgb_key,
        "camera_output_keys": list(camera.data.output.keys()),
    }


def _episode_bounds(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    episode_summary = Path(args.episode_dir).resolve() / "episode_summary.json"
    if episode_summary.is_file():
        data = _load_json(episode_summary)
        raw = data.get("map_bounds") or data.get("bounds")
        if raw is not None:
            return normalize_map_bounds(raw)
    return normalize_map_bounds({"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]})


def _update_observed_state(
    args: argparse.Namespace,
    output_dir: Path,
    capture: dict[str, Any],
) -> dict[str, Any]:
    step = int(args.capture_step)
    prior_step = max(step - 1, 0)
    episode_dir = Path(args.episode_dir).resolve()
    prior_path = episode_dir / f"observed_state_step{prior_step:03d}.npy"
    fallback_empty = False
    bounds = _episode_bounds(args)
    if prior_path.is_file():
        prior = np.load(prior_path)
    else:
        fallback_empty = True
        prior = create_observed_grid(bounds, voxel_size=float(args.voxel_size))
        prior_path = output_dir / "observed_state_prior_empty_fallback.npy"
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
    output_observed = output_dir / f"observed_state_isaac_capture_step{step:03d}.npy"
    np.save(output_observed, updated)
    output_hash = sha256_file(output_observed)
    updated_summary = summarize_observed_grid(updated)

    saved_step_path = episode_dir / f"observed_state_step{step:03d}.npy"
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

    hash_payload = {
        "prior_observed_state": str(prior_path),
        "prior_sha256_before": prior_hash_before,
        "prior_sha256_after": prior_hash_after,
        "prior_hash_unchanged": prior_hash_before == prior_hash_after,
        "fallback_empty_prior_used": fallback_empty,
        "new_observed_state": str(output_observed),
        "new_observed_state_sha256": output_hash,
    }
    _save_json(output_dir / "observed_state_prior_hash.json", hash_payload)

    capture_summary = {
        "stage": "Stage 4A-6.5m measured-only depth update",
        "single_frame_capture_only": True,
        "prior_observed_state": hash_payload,
        "prior_summary_before": prior_summary_before,
        "updated_summary": updated_summary,
        "delta_observed_count": int(updated_summary["observed_count"] - prior_summary_before["observed_count"]),
        "delta_observed_ratio": float(updated_summary["observed_ratio"] - prior_summary_before["observed_ratio"]),
        "bounds": {axis: [float(bounds[axis][0]), float(bounds[axis][1])] for axis in ("x", "y", "z")},
        "voxel_size": float(args.voxel_size),
        "pixel_stride": int(args.pixel_stride),
        "capture_depth_file": capture["depth_path"],
        "capture_pose_file": capture["pose_path"],
        "capture_camera_info_file": capture["camera_info_path"],
        "saved_step_comparison": saved_step_comparison,
        "prediction_used": False,
        "map_predict_used": False,
        "target_lr_target_hr_ground_truth_used": False,
        "existing_observed_state_modified": prior_hash_before != prior_hash_after,
    }
    _save_json(output_dir / "observed_state_capture_summary.json", capture_summary)
    return {
        "observed_path": str(output_observed),
        "observed_hash": output_hash,
        "prior_hash_payload": hash_payload,
        "capture_summary": capture_summary,
    }


def _tree_args(args: argparse.Namespace, output_dir: Path, observed_path: str, capture: dict[str, Any]) -> argparse.Namespace:
    episode_dir = Path(args.episode_dir).resolve()
    return argparse.Namespace(
        case_json=str(Path(args.selected_case_json).resolve()),
        episode_dir=str(episode_dir),
        observed_state=str(observed_path),
        pose_json=str(capture["pose_path"]),
        camera_info=str(capture["camera_info_path"]),
        episode_summary=str(episode_dir / "episode_summary.json"),
        prediction_npz="",
        output_dir=str(output_dir),
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
        variant_name=str(args.variant_name),
    )


def _decision_parts(summary: dict[str, Any]) -> dict[str, Any]:
    comparison = summary.get("comparison", {}).get("mini_rrt", {})
    decision = summary.get("decision", {})
    selected = decision.get("selected_child") or comparison.get("selected_child") or {}
    best = decision.get("best_descendant") or comparison.get("best_descendant") or {}
    return {
        "selected": selected,
        "best": best,
        "selected_distance_m": comparison.get("selected_child_distance_from_root_m"),
        "best_distance_m": comparison.get("best_descendant_distance_from_root_m"),
        "value": selected.get("value"),
        "accumulated_gain": best.get("accumulated_gain"),
        "accumulated_cost": best.get("accumulated_cost"),
        "accepted_nodes": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
        "rejected_samples": summary.get("tree", {}).get("rejected_samples"),
    }


def _load_reference_decision(reference_tree_dir: Path) -> dict[str, Any]:
    decision_path = reference_tree_dir / "source_protected_tree_decision.json"
    if decision_path.is_file():
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
    summary_path = reference_tree_dir / "mini_rrt_tree_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"reference tree decision not found in {reference_tree_dir}")
    summary = _load_json(summary_path)
    return {"path": str(summary_path), **_decision_parts(summary)}


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


def _close(a: Any, b: Any, tol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _copy_alias(output_dir: Path, src_name: str, dst_name: str) -> str | None:
    src = output_dir / src_name
    dst = output_dir / dst_name
    if not src.is_file():
        return None
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return str(dst)


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
                "reason": "needs root change / multi-step planner, not a one-frame smoke",
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
    }


def _write_source_protection_md(path: Path, checklist: dict[str, Any]) -> None:
    mech = checklist["mechanisms"]
    lines = [
        "# Source Protection Checklist",
        "",
        f"- profile: `{checklist['profile_name']}`",
        f"- variant: `{checklist['variant_name']}`",
        f"- A. crop_min_length / min_path_length: implemented `{mech['crop_min_length_min_path_length']['implemented']}`, active `{mech['crop_min_length_min_path_length']['active']}`, value `{mech['crop_min_length_min_path_length']['value_m']}` m.",
        f"- B. density limiting / max_density_range: implemented `{mech['density_limiting_max_density_range']['implemented']}`, active `{mech['density_limiting_max_density_range']['active']}`; {mech['density_limiting_max_density_range']['reason']}.",
        f"- C. continuous yaw: implemented approximation `{mech['continuous_yaw']['implemented_approximation']}`, active `{mech['continuous_yaw']['active']}`, value `{mech['continuous_yaw']['num_yaw_samples']}` fixed yaw samples.",
        f"- D. root rewiring / reinsert: full implementation `{mech['root_rewiring_reinsert']['full_implementation']}`, active `{mech['root_rewiring_reinsert']['active']}`; {mech['root_rewiring_reinsert']['reason']}.",
        f"- E. optional parent visible clearing: active `{mech['optional_parent_visible_clearing']['active']}`; {mech['optional_parent_visible_clearing']['reason']}.",
        f"- F. root-visible filtering / near-root discount: active `{mech['root_visible_filtering_near_root_discount']['active']}`; {mech['root_visible_filtering_near_root_discount']['reason']}.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _make_comparison_to_saved(
    live_parts: dict[str, Any],
    reference: dict[str, Any],
    mini_summary: dict[str, Any],
    observed_update: dict[str, Any],
) -> dict[str, Any]:
    live_selected = live_parts["selected"]
    live_best = live_parts["best"]
    ref_selected = reference["selected"]
    ref_best = reference["best"]
    selected_delta = _euclidean(live_selected.get("end_world"), ref_selected.get("end_world"))
    best_delta = _euclidean(live_best.get("end_world"), ref_best.get("end_world"))
    exact = bool(
        live_selected.get("segment_id") == ref_selected.get("segment_id")
        and live_best.get("segment_id") == ref_best.get("segment_id")
        and _same_grid(live_selected.get("end_grid"), ref_selected.get("end_grid"))
        and _same_grid(live_best.get("end_grid"), ref_best.get("end_grid"))
        and _close(live_parts.get("selected_distance_m"), reference.get("selected_distance_m"))
        and _close(live_parts.get("best_distance_m"), reference.get("best_distance_m"))
        and _close(live_parts.get("accumulated_gain"), reference.get("accumulated_gain"))
        and _close(live_parts.get("accumulated_cost"), reference.get("accumulated_cost"))
        and _close(live_parts.get("value"), reference.get("value"))
    )
    spatially_close = bool(
        selected_delta is not None
        and selected_delta <= 0.75
        and best_delta is not None
        and best_delta <= 1.25
    )
    selected_distance = live_parts.get("selected_distance_m")
    best_distance = live_parts.get("best_distance_m")
    nonlocal_branch = bool(
        float(selected_distance or 0.0) >= 0.5
        or float(best_distance or 0.0) >= 1.0
    )
    moved_off_n0140 = bool(live_selected.get("segment_id") != OLD_SHORT_EDGE_CHILD_ID)
    baseline_differs = not _same_grid(live_selected.get("end_grid"), ONE_STEP_BASELINE_GRID)
    return {
        "stage": "Stage 4A-6.5m comparison to Stage 4A-6.5l saved-map tree smoke",
        "reference_tree_dir": str(Path(args_cli.reference_tree_dir).resolve()),
        "reference_decision": reference,
        "live_capture_decision": live_parts,
        "judgement": {
            "exact_match_with_stage4a65l_saved_map": exact,
            "selected_child_spatial_delta_m": selected_delta,
            "best_descendant_spatial_delta_m": best_delta,
            "spatially_close_to_saved_map": spatially_close,
            "moved_off_old_short_edge_n0140": moved_off_n0140,
            "selected_child_differs_from_one_step_baseline_grid": baseline_differs,
            "nonlocal_branch_found": nonlocal_branch,
            "measured_only_depth_update": True,
            "prediction_used": False,
            "map_predict_used": False,
            "expected_difference_if_not_exact": (
                "Live Isaac depth can differ slightly from saved depth/replay timing; exact match is not required."
            ),
            "saved_step_observed_state_equal_to_live_capture": observed_update["capture_summary"]
            .get("saved_step_comparison", {})
            .get("equal_to_saved_step"),
            "tree_built_successfully": bool(mini_summary.get("tree", {}).get("built_successfully")),
        },
    }


def _write_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    live = comparison["live_capture_decision"]
    ref = comparison["reference_decision"]
    judge = comparison["judgement"]
    lines = [
        "# Comparison To Stage 4A-6.5l Saved-Map Tree Smoke",
        "",
        f"- exact match: `{judge['exact_match_with_stage4a65l_saved_map']}`",
        f"- selected child spatial delta: `{judge['selected_child_spatial_delta_m']}` m",
        f"- best descendant spatial delta: `{judge['best_descendant_spatial_delta_m']}` m",
        f"- spatially close: `{judge['spatially_close_to_saved_map']}`",
        f"- moved off `n0140`: `{judge['moved_off_old_short_edge_n0140']}`",
        f"- nonlocal branch found: `{judge['nonlocal_branch_found']}`",
        f"- measured-only / no prediction: `{judge['measured_only_depth_update']}` / `{not judge['prediction_used']}`",
        "",
        "## Live Capture",
        f"- selected child: `{live['selected'].get('segment_id')}` grid `{live['selected'].get('end_grid')}` world `{live['selected'].get('end_world')}` distance `{live['selected_distance_m']}` m",
        f"- best descendant: `{live['best'].get('segment_id')}` grid `{live['best'].get('end_grid')}` world `{live['best'].get('end_world')}` distance `{live['best_distance_m']}` m",
        f"- value: `{live['value']}`; accumulated gain/cost: `{live['accumulated_gain']}` / `{live['accumulated_cost']}`",
        f"- accepted/rejected: `{live['accepted_nodes']}` / `{live['rejected_samples']}`",
        "",
        "## Saved Reference",
        f"- selected child: `{ref['selected'].get('segment_id')}` grid `{ref['selected'].get('end_grid')}` world `{ref['selected'].get('end_world')}` distance `{ref['selected_distance_m']}` m",
        f"- best descendant: `{ref['best'].get('segment_id')}` grid `{ref['best'].get('end_grid')}` world `{ref['best'].get('end_world')}` distance `{ref['best_distance_m']}` m",
        f"- value: `{ref['value']}`; accumulated gain/cost: `{ref['accumulated_gain']}` / `{ref['accumulated_cost']}`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _make_decision_payload(
    args: argparse.Namespace,
    mini_summary: dict[str, Any],
    live_parts: dict[str, Any],
    checklist: dict[str, Any],
    observed_update: dict[str, Any],
    comparison: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": "Stage 4A-6.5m no-prediction Isaac one-step capture tree decision",
        "profile_name": PROFILE_NAME,
        "variant_name": str(args.variant_name),
        "profile": checklist["profile_parameters"],
        "selected_child": live_parts["selected"],
        "selected_child_distance_from_root_m": live_parts["selected_distance_m"],
        "best_descendant": live_parts["best"],
        "best_descendant_distance_from_root_m": live_parts["best_distance_m"],
        "value": live_parts["value"],
        "accumulated_gain": live_parts["accumulated_gain"],
        "accumulated_cost": live_parts["accumulated_cost"],
        "accepted_nodes": live_parts["accepted_nodes"],
        "rejected_samples": live_parts["rejected_samples"],
        "raw_decision": mini_summary.get("decision", {}),
        "comparison_to_saved_tree_smoke": comparison["judgement"],
        "observed_state": {
            "path": observed_update["observed_path"],
            "sha256": observed_update["observed_hash"],
            "prior_path": observed_update["prior_hash_payload"]["prior_observed_state"],
            "prior_sha256_before": observed_update["prior_hash_payload"]["prior_sha256_before"],
            "prior_sha256_after": observed_update["prior_hash_payload"]["prior_sha256_after"],
            "prior_unchanged": observed_update["prior_hash_payload"]["prior_hash_unchanged"],
        },
        "safety": {
            "isaac_startup": True,
            "one_frame_capture_only": True,
            "rollout": False,
            "selected_action_execution": False,
            "online_multi_step_loop": False,
            "map_predict_rerun": False,
            "sscnet_inference_or_training": False,
            "training_rl_ppo_bc_il": False,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "target_lr_target_hr_ground_truth_scoring": False,
        },
    }


def _write_decision_md(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_child"]
    best = payload["best_descendant"]
    lines = [
        "# Source-Protected Tree Decision",
        "",
        f"- profile: `{payload['profile_name']}`",
        f"- variant: `{payload['variant_name']}`",
        f"- selected child: `{selected.get('segment_id')}`",
        f"- selected child grid/world: `{selected.get('end_grid')}` / `{selected.get('end_world')}`",
        f"- selected child distance: `{payload['selected_child_distance_from_root_m']}` m",
        f"- best descendant: `{best.get('segment_id')}`",
        f"- best descendant grid/world: `{best.get('end_grid')}` / `{best.get('end_world')}`",
        f"- best descendant distance: `{payload['best_descendant_distance_from_root_m']}` m",
        f"- value: `{payload['value']}`",
        f"- accumulated gain/cost: `{payload['accumulated_gain']}` / `{payload['accumulated_cost']}`",
        f"- accepted/rejected: `{payload['accepted_nodes']}` / `{payload['rejected_samples']}`",
        f"- prior observed_state unchanged: `{payload['observed_state']['prior_unchanged']}`",
        "- prediction / map_predict used: `False` / `False`",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _topdown_image(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def _save_observed_topdown(path: Path, observed_state: np.ndarray, root_grid: list[int] | None) -> None:
    image = _topdown_image(observed_state)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    if root_grid is not None:
        ax.scatter([root_grid[0] + 0.5], [root_grid[1] + 0.5], c="#ffffff", edgecolors="#111111", s=70, label="root")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Captured observed map topdown")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_saved_vs_capture_plot(
    path: Path,
    observed_state: np.ndarray,
    live_parts: dict[str, Any],
    reference: dict[str, Any],
) -> None:
    image = _topdown_image(observed_state)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")

    def draw_branch(parts: dict[str, Any], color: str, label: str) -> None:
        selected = parts["selected"]
        best = parts["best"]
        start = selected.get("start_grid")
        selected_grid = selected.get("end_grid")
        best_grid = best.get("end_grid")
        if start and selected_grid:
            ax.plot(
                [start[0] + 0.5, selected_grid[0] + 0.5],
                [start[1] + 0.5, selected_grid[1] + 0.5],
                color=color,
                linewidth=2.0,
                label=f"{label} selected",
            )
            ax.scatter([selected_grid[0] + 0.5], [selected_grid[1] + 0.5], c=color, s=60)
        if start and best_grid:
            ax.plot(
                [start[0] + 0.5, best_grid[0] + 0.5],
                [start[1] + 0.5, best_grid[1] + 0.5],
                color=color,
                linewidth=1.2,
                linestyle="--",
                alpha=0.85,
                label=f"{label} best",
            )
            ax.scatter([best_grid[0] + 0.5], [best_grid[1] + 0.5], c=color, marker="x", s=90)

    draw_branch(live_parts, "#d1495b", "capture")
    draw_branch(reference, "#3f88c5", "saved")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Saved-map vs live-capture tree decision")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _create_alias_outputs(output_dir: Path) -> dict[str, Any]:
    return {
        "source_protected_tree_segments.jsonl": _copy_alias(
            output_dir, "mini_rrt_tree_segments.jsonl", "source_protected_tree_segments.jsonl"
        ),
        "source_protected_gain_cost_value_table.csv": _copy_alias(
            output_dir, "gain_cost_value_table.csv", "source_protected_gain_cost_value_table.csv"
        ),
        "source_protected_sampled_nodes.csv": _copy_alias(
            output_dir, "sampled_nodes.csv", "source_protected_sampled_nodes.csv"
        ),
        "source_protected_rejected_samples.csv": _copy_alias(
            output_dir, "rejected_samples.csv", "source_protected_rejected_samples.csv"
        ),
        "tree_capture_topdown.png": _copy_alias(output_dir, "mini_rrt_tree_topdown.png", "tree_capture_topdown.png"),
        "selected_branch_capture_topdown.png": _copy_alias(
            output_dir, "selected_branch_topdown.png", "selected_branch_capture_topdown.png"
        ),
        "gain_cost_value_capture_scatter.png": _copy_alias(
            output_dir, "gain_cost_scatter.png", "gain_cost_value_capture_scatter.png"
        ),
    }


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
        found.extend(str(path) for path in sorted(output_dir.glob(pattern)))
    return found


def _make_summary(
    args: argparse.Namespace,
    output_dir: Path,
    capture: dict[str, Any],
    observed_update: dict[str, Any],
    live_parts: dict[str, Any],
    comparison: dict[str, Any],
    checklist: dict[str, Any],
    checkpoint_before: str | None,
    checkpoint_after: str | None,
    external_before: str,
    external_after: str,
    ssc_before: str,
    ssc_after: str,
) -> dict[str, Any]:
    judge = comparison["judgement"]
    ready_two_frame = bool(
        (judge["exact_match_with_stage4a65l_saved_map"] or judge["spatially_close_to_saved_map"])
        and judge["moved_off_old_short_edge_n0140"]
        and judge["nonlocal_branch_found"]
        and observed_update["prior_hash_payload"]["prior_hash_unchanged"]
        and not judge["prediction_used"]
    )
    rollout_outputs = _scan_prohibited_rollout_outputs(output_dir)
    map_predict_artifacts = _scan_map_predict_artifacts(output_dir)
    return {
        "stage": "Stage 4A-6.5m no-prediction Isaac one-step capture + tree decision smoke",
        "output_dir": str(output_dir),
        "answers": {
            "isaac_one_frame_capture_succeeded": True,
            "captured_depth_updated_new_observed_state": True,
            "old_observed_state_unmodified": bool(observed_update["prior_hash_payload"]["prior_hash_unchanged"]),
            "source_protected_tree_decision_succeeded": True,
            "source_like_protections_active": bool(checklist["mechanisms"]["crop_min_length_min_path_length"]["active"])
            and bool(checklist["mechanisms"]["continuous_yaw"]["active"]),
            "prediction_or_map_predict_used": False,
            "matches_saved_map_stage4a65l": judge["exact_match_with_stage4a65l_saved_map"],
            "spatially_close_if_not_exact": judge["spatially_close_to_saved_map"],
            "moved_off_n0140": judge["moved_off_old_short_edge_n0140"],
            "nonlocal_branch_found": judge["nonlocal_branch_found"],
            "enough_for_no_prediction_two_frame_smoke": ready_two_frame,
            "ready_for_rollout": False,
        },
        "capture": {
            "scene": str(args.scene_variant),
            "scene_seed": int(args.scene_seed),
            "pose": capture["pose"],
            "depth_file": capture["depth_path"],
            "rgb_file": capture["rgb_path"],
            "camera_info": capture["camera_info_path"],
            "depth_stats": capture["depth_stats"],
            "rgb_stats": capture["rgb_stats"],
        },
        "observed_state": observed_update["capture_summary"],
        "tree_decision": live_parts,
        "comparison_to_saved_tree_smoke": comparison,
        "source_protection_checklist": checklist,
        "safety": {
            "isaac_startup": True,
            "one_frame_capture_only": True,
            "rollout": False,
            "selected_action_execution": False,
            "online_multi_step_loop": False,
            "map_predict_rerun": False,
            "sscnet_inference": False,
            "sscnet_training": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_modified": checkpoint_before != checkpoint_after,
            "existing_observed_state_modified": not observed_update["prior_hash_payload"]["prior_hash_unchanged"],
            "new_observed_state_output_written": True,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "target_lr_target_hr_ground_truth_scoring": False,
            "external_source_git_status_before": external_before,
            "external_source_git_status_after": external_after,
            "external_source_modified_or_built_by_stage": external_before != external_after,
            "ssc_exploration_git_status_before": ssc_before,
            "ssc_exploration_git_status_after": ssc_after,
            "ssc_exploration_modified_by_stage": ssc_before != ssc_after,
            "prohibited_rollout_outputs": rollout_outputs,
            "map_predict_artifacts_in_output_dir": map_predict_artifacts,
            "coverage_improvement_claimed": False,
        },
        "recommended_next_faithful_step": (
            "no-prediction two-frame tree smoke"
            if ready_two_frame
            else "capture/replay reproducibility or tree sampling stability debug"
        ),
        "still_not_next": [
            "rollout",
            "RL/PPO/BC/IL training",
            "map_predict tree integration",
            "prediction writeback",
            "checkpoint changes",
            "external source build",
        ],
    }


def _write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    tree = summary["tree_decision"]
    selected = tree["selected"]
    best = tree["best"]
    judge = summary["comparison_to_saved_tree_smoke"]["judgement"]
    lines = [
        "# Stage 4A-6.5m Isaac One-Step Tree Capture Summary",
        "",
        f"1. Isaac one-frame capture succeeded? `{answers['isaac_one_frame_capture_succeeded']}`.",
        f"2. Captured depth updated a new observed_state? `{answers['captured_depth_updated_new_observed_state']}`.",
        f"3. Old observed_state unmodified? `{answers['old_observed_state_unmodified']}`.",
        f"4. Source-protected tree decision ran? `{answers['source_protected_tree_decision_succeeded']}`.",
        f"5. crop_min_length / source-like protections active? `{answers['source_like_protections_active']}`.",
        f"6. prediction / map_predict used? `{answers['prediction_or_map_predict_used']}`.",
        f"7. Live capture decision matches saved-map Stage 4A-6.5l? `{answers['matches_saved_map_stage4a65l']}`.",
        f"8. If not exact, spatially close? `{answers['spatially_close_if_not_exact']}`.",
        f"9. Moved off `n0140`? `{answers['moved_off_n0140']}`.",
        f"10. Nonlocal branch found? `{answers['nonlocal_branch_found']}`.",
        f"11. Enough for no-prediction two-frame tree smoke? `{answers['enough_for_no_prediction_two_frame_smoke']}`.",
        f"12. Ready for rollout? `{answers['ready_for_rollout']}`.",
        "",
        "## Decision",
        f"- selected child: `{selected.get('segment_id')}` grid `{selected.get('end_grid')}` world `{selected.get('end_world')}` distance `{tree['selected_distance_m']}` m.",
        f"- best descendant: `{best.get('segment_id')}` grid `{best.get('end_grid')}` world `{best.get('end_world')}` distance `{tree['best_distance_m']}` m.",
        f"- value: `{tree['value']}`; accumulated gain/cost: `{tree['accumulated_gain']}` / `{tree['accumulated_cost']}`.",
        f"- accepted/rejected: `{tree['accepted_nodes']}` / `{tree['rejected_samples']}`.",
        "",
        "## Comparison",
        f"- selected child spatial delta: `{judge['selected_child_spatial_delta_m']}` m.",
        f"- best descendant spatial delta: `{judge['best_descendant_spatial_delta_m']}` m.",
        "",
        f"Recommended next faithful step: {summary['recommended_next_faithful_step']}. Still no rollout.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _write_recommended_next(path: Path, summary: dict[str, Any]) -> None:
    reason = (
        "live capture matched or stayed spatially close to the saved source-protected tree decision and remained nonlocal"
        if summary["answers"]["enough_for_no_prediction_two_frame_smoke"]
        else "live capture was not both close/stable and nonlocal"
    )
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {summary['recommended_next_faithful_step']}",
        f"- reason: {reason}",
        "- still not next: rollout, RL, PPO, BC/IL training, map_predict tree integration, prediction writeback, target/ground-truth scoring, checkpoint changes, or external source build.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_before = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_before = _git_status_short(SSC_EXPLORATION_DIR)

    capture = _capture_one_frame(args, output_dir)
    observed_update = _update_observed_state(args, output_dir, capture)

    mini_summary = run_mini_rrt(
        _tree_args(args, output_dir, observed_update["observed_path"], capture)
    )
    live_parts = _decision_parts(mini_summary)
    reference = _load_reference_decision(Path(args.reference_tree_dir).resolve())
    checklist = _make_source_protection_checklist(args)
    comparison = _make_comparison_to_saved(live_parts, reference, mini_summary, observed_update)
    aliases = _create_alias_outputs(output_dir)

    observed_state = np.load(observed_update["observed_path"])
    root_grid = mini_summary.get("root", {}).get("grid")
    _save_observed_topdown(output_dir / "observed_capture_topdown.png", observed_state, root_grid)
    _save_saved_vs_capture_plot(
        output_dir / "saved_vs_capture_tree_decision_topdown.png",
        observed_state,
        live_parts,
        reference,
    )

    checkpoint_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_after = _git_status_short(EXTERNAL_SOURCE_DIR)
    ssc_after = _git_status_short(SSC_EXPLORATION_DIR)

    decision_payload = _make_decision_payload(
        args,
        mini_summary,
        live_parts,
        checklist,
        observed_update,
        comparison,
    )
    summary = _make_summary(
        args,
        output_dir,
        capture,
        observed_update,
        live_parts,
        comparison,
        checklist,
        checkpoint_before,
        checkpoint_after,
        external_before,
        external_after,
        ssc_before,
        ssc_after,
    )
    summary["generated_aliases"] = aliases

    _save_json(output_dir / "source_protected_tree_decision.json", decision_payload)
    _write_decision_md(output_dir / "source_protected_tree_decision.md", decision_payload)
    _save_json(output_dir / "source_protection_checklist.json", checklist)
    _write_source_protection_md(output_dir / "source_protection_checklist.md", checklist)
    _save_json(output_dir / "comparison_to_saved_tree_smoke.json", comparison)
    _write_comparison_md(output_dir / "comparison_to_saved_tree_smoke.md", comparison)
    _save_json(output_dir / "isaac_one_step_tree_capture_summary.json", summary)
    _write_summary_md(output_dir / "isaac_one_step_tree_capture_summary.md", summary)
    _write_recommended_next(output_dir / "recommended_next_faithful_step.md", summary)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    try:
        run(args_cli)
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
