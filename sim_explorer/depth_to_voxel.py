#!/usr/bin/env python3
"""Convert Isaac depth images into a simple observed voxel map.

This module is intentionally independent from SSCNet, PredictionLayer, expert
scoring, target labels, and ground-truth maps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)

DEFAULT_MAP_BOUNDS = {
    "x": (-4.0, 4.0),
    "y": (-4.0, 4.0),
    "z": (0.0, 3.0),
}


def normalize_map_bounds(map_bounds: dict[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    """Return map bounds as ordered float tuples."""
    raw = DEFAULT_MAP_BOUNDS if map_bounds is None else map_bounds
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def grid_shape(map_bounds: dict[str, Any] | None, voxel_size: float) -> tuple[int, int, int]:
    bounds = normalize_map_bounds(map_bounds)
    return tuple(
        int(math.ceil((bounds[axis][1] - bounds[axis][0]) / float(voxel_size))) for axis in ("x", "y", "z")
    )


def create_observed_grid(
    map_bounds: dict[str, Any] | None = None,
    voxel_size: float = 0.1,
) -> np.ndarray:
    """Create an all-unknown observed map with axis order `[x, y, z]`."""
    return np.full(grid_shape(map_bounds, voxel_size), UNKNOWN, dtype=np.int8)


def summarize_observed_grid(observed_state: np.ndarray) -> dict[str, Any]:
    unknown_count = int(np.count_nonzero(observed_state == UNKNOWN))
    free_count = int(np.count_nonzero(observed_state == FREE))
    occupied_count = int(np.count_nonzero(observed_state == OCCUPIED))
    total = int(observed_state.size)
    observed_count = free_count + occupied_count
    return {
        "shape": list(observed_state.shape),
        "total_count": total,
        "unknown_count": unknown_count,
        "free_count": free_count,
        "occupied_count": occupied_count,
        "observed_count": observed_count,
        "observed_ratio": float(observed_count / total) if total else 0.0,
    }


def _world_to_grid_index(
    point: np.ndarray,
    map_bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    shape: tuple[int, int, int],
) -> tuple[int, int, int] | None:
    mins = np.array([map_bounds["x"][0], map_bounds["y"][0], map_bounds["z"][0]], dtype=np.float64)
    idx = np.floor((point - mins) / float(voxel_size)).astype(np.int64)
    if np.any(idx < 0) or np.any(idx >= np.asarray(shape, dtype=np.int64)):
        return None
    return int(idx[0]), int(idx[1]), int(idx[2])


def _camera_basis_from_yaw(yaw_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return forward/right/up basis for a level camera.

    Yaw zero points along +X, positive yaw rotates toward +Y.
    """
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float64)
    right = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return forward, right, up


def _intrinsics_from_dict(intrinsics: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    if "intrinsic_matrix" in intrinsics:
        matrix = np.asarray(intrinsics["intrinsic_matrix"], dtype=np.float64)
        if matrix.shape == (3, 3):
            return float(matrix[0, 0]), float(matrix[1, 1]), float(matrix[0, 2]), float(matrix[1, 2])
    if all(k in intrinsics for k in ("fx", "fy", "cx", "cy")):
        return float(intrinsics["fx"]), float(intrinsics["fy"]), float(intrinsics["cx"]), float(intrinsics["cy"])
    horizontal_fov_deg = float(intrinsics.get("horizontal_fov_deg", 90.0))
    fx = (float(width) * 0.5) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    fy = fx
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return fx, fy, cx, cy


def _pose_to_origin_yaw(camera_pose: dict[str, Any]) -> tuple[np.ndarray, float]:
    if "position" not in camera_pose:
        raise KeyError("camera_pose must contain 'position'")
    origin = np.asarray(camera_pose["position"], dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError(f"camera_pose position must have shape (3,), got {origin.shape}")

    if "yaw_rad" in camera_pose:
        yaw_rad = float(camera_pose["yaw_rad"])
    elif "yaw_deg" in camera_pose:
        yaw_rad = math.radians(float(camera_pose["yaw_deg"]))
    else:
        raise KeyError("camera_pose must contain 'yaw_rad' or 'yaw_deg'")
    return origin, yaw_rad


def integrate_depth_frame(
    observed_state: np.ndarray,
    depth_image: np.ndarray,
    camera_pose: dict[str, Any],
    intrinsics: dict[str, Any],
    map_bounds: dict[str, Any] | None = None,
    voxel_size: float = 0.1,
    pixel_stride: int = 2,
    max_depth: float | None = None,
) -> np.ndarray:
    """Fuse one depth image into an observed voxel map.

    The output only contains measured occupancy evidence from the depth sensor.
    It never consumes predictions, target labels, or ground truth.
    """
    if observed_state.dtype != np.int8:
        raise ValueError(f"observed_state must be int8, got {observed_state.dtype}")
    if pixel_stride < 1:
        raise ValueError("pixel_stride must be >= 1")

    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"depth_image must be HxW or HxWx1, got {depth_image.shape}")

    height, width = depth.shape
    fx, fy, cx, cy = _intrinsics_from_dict(intrinsics, width=width, height=height)
    origin, yaw_rad = _pose_to_origin_yaw(camera_pose)
    forward, right, up = _camera_basis_from_yaw(yaw_rad)

    bounds = normalize_map_bounds(map_bounds)
    expected_shape = grid_shape(bounds, voxel_size)
    if tuple(observed_state.shape) != expected_shape:
        raise ValueError(f"observed_state shape {observed_state.shape} does not match expected {expected_shape}")

    effective_max_depth = float(max_depth if max_depth is not None else intrinsics.get("max_depth", 5.0))
    free_step = max(float(voxel_size) * 0.5, 1e-3)

    for v in range(0, height, pixel_stride):
        y_norm = (float(v) + 0.5 - cy) / fy
        for u in range(0, width, pixel_stride):
            depth_val = float(depth[v, u])
            if not np.isfinite(depth_val) or depth_val <= 0.0:
                continue

            clamped_depth = min(depth_val, effective_max_depth)
            camera_ray = forward + ((float(u) + 0.5 - cx) / fx) * right - y_norm * up
            hit_vector = clamped_depth * camera_ray
            hit_distance = float(np.linalg.norm(hit_vector))
            if hit_distance <= 1e-6:
                continue
            ray_dir = hit_vector / hit_distance

            free_limit = max(hit_distance - free_step, 0.0)
            for distance in np.arange(0.0, free_limit, free_step):
                point = origin + ray_dir * distance
                idx = _world_to_grid_index(point, bounds, float(voxel_size), tuple(observed_state.shape))
                if idx is not None and observed_state[idx] == UNKNOWN:
                    observed_state[idx] = FREE

            hit_is_surface = depth_val < (effective_max_depth - 1e-3)
            if hit_is_surface:
                hit_point = origin + hit_vector
                hit_idx = _world_to_grid_index(hit_point, bounds, float(voxel_size), tuple(observed_state.shape))
                if hit_idx is not None:
                    observed_state[hit_idx] = OCCUPIED

    return observed_state


def update_observed_state_from_depth(
    observed_state: np.ndarray,
    depth: np.ndarray,
    camera_pose: dict[str, Any],
    camera_info: dict[str, Any],
    bounds: dict[str, Any] | None = None,
    voxel_size: float = 0.1,
    pixel_stride: int = 2,
) -> np.ndarray:
    """Fuse one Isaac depth frame into the measured-only observed map.

    This is the Stage 4A-3 rollout-facing wrapper around
    `integrate_depth_frame`. It keeps the same boundary as Stage 4A-1: FREE
    and OCCUPIED updates come only from depth ray marching.
    """
    return integrate_depth_frame(
        observed_state=observed_state,
        depth_image=depth,
        camera_pose=camera_pose,
        intrinsics=camera_info,
        map_bounds=bounds,
        voxel_size=voxel_size,
        pixel_stride=pixel_stride,
        max_depth=float(camera_info.get("max_depth", 5.0)),
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_depth_sequence(
    input_dir: Path,
    output_dir: Path,
    voxel_size: float = 0.1,
    pixel_stride: int = 2,
    map_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    camera_info = _load_json(input_dir / "camera_info.json")
    scene_metadata_path = input_dir / "scene_metadata.json"
    scene_metadata = _load_json(scene_metadata_path) if scene_metadata_path.exists() else {}
    bounds = normalize_map_bounds(map_bounds or scene_metadata.get("map_bounds") or DEFAULT_MAP_BOUNDS)
    observed_state = create_observed_grid(bounds, voxel_size)

    output_dir.mkdir(parents=True, exist_ok=True)
    depth_paths = sorted(input_dir.glob("depth_*.npy"))
    if not depth_paths:
        raise FileNotFoundError(f"No depth_*.npy files found in {input_dir}")

    step_summaries = []
    for step, depth_path in enumerate(depth_paths):
        pose_path = input_dir / f"pose_{step:03d}.json"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing pose file for {depth_path.name}: {pose_path}")
        depth = np.load(depth_path)
        pose = _load_json(pose_path)
        observed_state = integrate_depth_frame(
            observed_state=observed_state,
            depth_image=depth,
            camera_pose=pose,
            intrinsics=camera_info,
            map_bounds=bounds,
            voxel_size=voxel_size,
            pixel_stride=pixel_stride,
            max_depth=float(camera_info.get("max_depth", 5.0)),
        )
        step_path = output_dir / f"observed_state_step{step}.npy"
        np.save(step_path, observed_state)
        step_summary = summarize_observed_grid(observed_state)
        step_summary.update(
            {
                "step": step,
                "depth_file": str(depth_path),
                "pose_file": str(pose_path),
                "observed_state_file": str(step_path),
            }
        )
        step_summaries.append(step_summary)

    final_observed_path = output_dir / "observed_state_final.npy"
    np.save(final_observed_path, observed_state)

    summary = summarize_observed_grid(observed_state)
    summary.update(
        {
            "voxel_size": float(voxel_size),
            "map_bounds": {axis: [bounds[axis][0], bounds[axis][1]] for axis in ("x", "y", "z")},
            "pixel_stride": int(pixel_stride),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "num_depth_frames": len(depth_paths),
            "steps": step_summaries,
            "final_observed_state_file": str(final_observed_path),
            "prediction_used": False,
            "target_lr_used": False,
            "ground_truth_used": False,
        }
    )
    summary_path = output_dir / "observed_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Isaac depth smoke outputs to an observed voxel map.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke"),
        help="Directory containing depth_*.npy, pose_*.json, and camera_info.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke"),
        help="Directory to write observed_state_step*.npy and observed_summary.json.",
    )
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--pixel_stride", type=int, default=2)
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument("--z_min", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    return parser.parse_args()


def _bounds_from_cli(args: argparse.Namespace) -> dict[str, tuple[float, float]] | None:
    values = {
        "x": (args.x_min, args.x_max),
        "y": (args.y_min, args.y_max),
        "z": (args.z_min, args.z_max),
    }
    provided = [value is not None for pair in values.values() for value in pair]
    if not any(provided):
        return None
    missing = [f"{axis}_{suffix}" for axis, pair in values.items() for suffix, value in zip(("min", "max"), pair) if value is None]
    if missing:
        raise ValueError(f"Bounds CLI arguments must be provided as complete min/max pairs; missing {missing}")
    bounds = {axis: (float(pair[0]), float(pair[1])) for axis, pair in values.items()}
    for axis, pair in bounds.items():
        if pair[1] <= pair[0]:
            raise ValueError(f"Invalid {axis} bounds: {pair}")
    return bounds


def main() -> None:
    args = parse_args()
    summary = run_depth_sequence(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        voxel_size=args.voxel_size,
        pixel_stride=args.pixel_stride,
        map_bounds=_bounds_from_cli(args),
    )
    print("Depth-to-voxel smoke complete")
    print(f"observed map shape: {tuple(summary['shape'])}")
    print(f"unknown_count: {summary['unknown_count']}")
    print(f"free_count: {summary['free_count']}")
    print(f"occupied_count: {summary['occupied_count']}")
    print(f"observed_ratio: {summary['observed_ratio']:.6f}")
    print(f"summary: {Path(args.output_dir) / 'observed_summary.json'}")


if __name__ == "__main__":
    main()
