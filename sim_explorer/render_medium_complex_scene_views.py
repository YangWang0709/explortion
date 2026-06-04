#!/usr/bin/env python3
"""Render Stage 4A-3.2 medium scene visualizations.

This is a visualization/smoke script only. It rebuilds the scripted cuboid
scene from deterministic metadata, renders RGB/depth views, and visualizes the
measured-only observed map saved by depth_to_voxel.py.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 4A-3.2 medium-complexity scene visualization.")
parser.add_argument(
    "--input_dir",
    type=str,
    default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke",
    help="Directory containing medium scene smoke outputs and observed maps.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_viz",
    help="Directory for rendered scene and observed-map visualizations.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--variant", default="three_rooms")
parser.add_argument("--obstacle_jitter_m", type=float, default=0.0)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=8.0)
parser.add_argument("--overview_width", type=int, default=640)
parser.add_argument("--overview_height", type=int, default=480)
parser.add_argument("--overview_max_depth", type=float, default=18.0)
parser.add_argument("--settle_steps", type=int, default=12)
parser.add_argument("--max_points", type=int, default=7000)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

from scene_factory import build_medium_complex_scene

DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
STATE_CMAP = ListedColormap(["#30343b", "#76b7c4", "#c95c5c"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _add_lighting() -> None:
    dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.82, 0.84, 0.80))
    dome_cfg.func("/World/Light", dome_cfg)


def _make_camera(prim_path: str, width: int, height: int, max_depth: float, focal_length: float) -> Camera:
    camera_cfg = CameraCfg(
        prim_path=prim_path,
        update_period=0.0,
        height=int(height),
        width=int(width),
        data_types=["rgb", DEPTH_KEY],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(focal_length),
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, float(max_depth)),
        ),
    )
    return Camera(cfg=camera_cfg)


def _pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def _set_camera_view(
    camera: Camera,
    sim: sim_utils.SimulationContext,
    position: list[float],
    target: list[float],
) -> None:
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )


def _settle_and_update(sim: sim_utils.SimulationContext, cameras: list[Camera], steps: int) -> None:
    for _ in range(max(int(steps), 1)):
        sim.step()
        for camera in cameras:
            camera.update(dt=sim.get_physics_dt())


def _normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _extract_rgb(camera: Camera, label: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEY_CANDIDATES:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        source = tensor[0].detach().cpu().numpy()
        if source.ndim != 3 or source.shape[-1] not in (3, 4):
            raise ValueError(f"{label}: expected RGB/RGBA image, got {source.shape}")
        rgb = _normalize_rgb(source)
        stats = {
            "key": key,
            "shape": [int(v) for v in rgb.shape],
            "min": int(rgb.min()) if rgb.size else None,
            "max": int(rgb.max()) if rgb.size else None,
            "mean": float(rgb.mean()) if rgb.size else None,
            "std": float(rgb.std()) if rgb.size else None,
        }
        if rgb.size == 0 or stats["max"] is None or int(stats["max"]) <= 2 or float(stats["std"] or 0.0) < 1.0:
            raise ValueError(f"{label}: RGB image appears blank or nearly uniform: {stats}")
        return rgb, key, stats
    raise KeyError(f"{label}: camera output missing rgb/rgba. Keys: {list(camera.data.output.keys())}")


def _extract_depth(camera: Camera, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    tensor = camera.data.output.get(DEPTH_KEY)
    if tensor is None:
        raise KeyError(f"{label}: camera output missing {DEPTH_KEY}. Keys: {list(camera.data.output.keys())}")
    depth = tensor[0].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        raise ValueError(f"{label}: depth has no finite positive values")
    return depth, {
        "shape": [int(v) for v in depth.shape],
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()),
        "max": float(positive.max()),
        "mean": float(positive.mean()),
    }


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def _save_depth_color_png(path: Path, depth: np.ndarray, title: str) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError(f"No finite positive depth values for {path}")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_title(f"{title}: min {float(finite.min()):.3f} m, max {float(finite.max()):.3f} m")
    ax.set_xlabel("u pixel")
    ax.set_ylabel("v pixel")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_rgb_grid(path: Path, rgbs: list[np.ndarray], labels: list[str]) -> None:
    fig, axes = plt.subplots(1, len(rgbs), figsize=(4.2 * len(rgbs), 3.2), constrained_layout=True)
    if len(rgbs) == 1:
        axes = [axes]
    for ax, rgb, label in zip(axes, rgbs, labels):
        ax.imshow(rgb)
        ax.set_title(label)
        ax.axis("off")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _save_depth_grid(path: Path, depths: list[np.ndarray], labels: list[str]) -> None:
    finite_values = [depth[np.isfinite(depth) & (depth > 0.0)] for depth in depths]
    vmin = min(float(values.min()) for values in finite_values if values.size)
    vmax = max(float(values.max()) for values in finite_values if values.size)
    fig, axes = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 3.6), constrained_layout=True)
    if len(depths) == 1:
        axes = [axes]
    last_image = None
    for ax, depth, label in zip(axes, depths, labels):
        masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
        last_image = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_xlabel("u")
        ax.set_ylabel("v")
    fig.colorbar(last_image, ax=axes, label="depth (m)", shrink=0.75)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _box_xy(spec: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, _ = (float(v) for v in spec["position"])
    sx, sy, _ = (float(v) for v in spec["size"])
    return x - 0.5 * sx, y - 0.5 * sy, sx, sy


def _draw_camera_arrow(ax, pose: dict[str, Any], color: str) -> None:
    position = [float(v) for v in pose["position"]]
    yaw_rad = float(pose.get("yaw_rad", math.radians(float(pose["yaw_deg"]))))
    target = _pose_target(position, yaw_rad)
    dx = target[0] - position[0]
    dy = target[1] - position[1]
    length = math.hypot(dx, dy)
    if length > 1.0e-6:
        dx, dy = dx / length * 0.55, dy / length * 0.55
    ax.arrow(position[0], position[1], dx, dy, color=color, width=0.025, head_width=0.18, length_includes_head=True)
    ax.scatter([position[0]], [position[1]], c=[color], s=36, zorder=5)
    ax.text(position[0] + 0.10, position[1] + 0.10, f"cam {pose['index']}", color=color, fontsize=8, weight="bold")


def _category_color(category: str) -> str:
    return {
        "table_like_low_box": "#c58d48",
        "cabinet_like_tall_box": "#587d66",
        "shelf_like_long_box": "#8b7652",
        "sofa_like_block": "#4e6795",
    }.get(category, "#9b6b4a")


def save_layout_topdown(path: Path, metadata: dict[str, Any]) -> None:
    bounds = metadata["map_bounds"]
    fig, ax = plt.subplots(figsize=(9.0, 9.0), constrained_layout=True)
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    ax.add_patch(
        Rectangle(
            (float(x0), float(y0)),
            float(x1) - float(x0),
            float(y1) - float(y0),
            facecolor="#edf1f4",
            edgecolor="#1f2937",
            linewidth=1.8,
            label="floor",
        )
    )

    for room in metadata.get("rooms", []):
        rb = room["bounds"]
        cx = 0.5 * (float(rb["x"][0]) + float(rb["x"][1]))
        cy = 0.5 * (float(rb["y"][0]) + float(rb["y"][1]))
        ax.text(cx, cy, room["label"], ha="center", va="center", fontsize=12, weight="bold", color="#374151")
    for corridor in metadata.get("corridors", []):
        cb = corridor["bounds"]
        cx = 0.5 * (float(cb["x"][0]) + float(cb["x"][1]))
        cy = 0.5 * (float(cb["y"][0]) + float(cb["y"][1]))
        ax.text(cx, cy, corridor["label"], ha="center", va="center", fontsize=10, weight="bold", color="#0f766e")

    for wall in metadata["walls"]:
        ax.add_patch(
            Rectangle(
                _box_xy(wall)[:2],
                _box_xy(wall)[2],
                _box_xy(wall)[3],
                facecolor="#777f89",
                edgecolor="#374151",
                alpha=0.92,
                label="walls" if wall["name"] == "outer_west" else None,
            )
        )

    for door in metadata["openings"]:
        rect = door["clear_rect"]
        rx0, rx1 = float(rect["x"][0]), float(rect["x"][1])
        ry0, ry1 = float(rect["y"][0]), float(rect["y"][1])
        ax.add_patch(
            Rectangle(
                (rx0, ry0),
                rx1 - rx0,
                ry1 - ry0,
                facecolor="#d6f0d2",
                edgecolor="#16803c",
                linewidth=1.8,
                alpha=0.95,
                label="door openings" if door["name"] == metadata["openings"][0]["name"] else None,
            )
        )
        ax.text(float(door["center"][0]) + 0.10, float(door["center"][1]) + 0.10, door["name"].replace("door_", ""), fontsize=7, color="#166534")

    for obstacle in metadata["obstacles"]:
        ax.add_patch(
            Rectangle(
                _box_xy(obstacle)[:2],
                _box_xy(obstacle)[2],
                _box_xy(obstacle)[3],
                facecolor=_category_color(str(obstacle.get("category", ""))),
                edgecolor="#111827",
                alpha=0.88,
                label="obstacles" if obstacle["name"] == metadata["obstacles"][0]["name"] else None,
            )
        )
        ox, oy, _ = (float(v) for v in obstacle["position"])
        ax.text(ox, oy, obstacle["name"].replace("_", "\n"), ha="center", va="center", fontsize=5.8, color="white")

    for pose in metadata["camera_poses"]:
        _draw_camera_arrow(ax, pose, "#111827")

    ax.set_title("Stage 4A-3.2 medium-complexity scripted layout")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(x0) - 0.4, float(x1) + 0.4)
    ax.set_ylim(float(y0) - 0.4, float(y1) + 0.4)
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.75)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _state_counts(grid: np.ndarray) -> dict[str, int | float | list[int]]:
    unknown = int(np.count_nonzero(grid == UNKNOWN))
    free = int(np.count_nonzero(grid == FREE))
    occupied = int(np.count_nonzero(grid == OCCUPIED))
    observed = free + occupied
    total = int(grid.size)
    return {
        "shape": [int(v) for v in grid.shape],
        "unknown_count": unknown,
        "free_count": free,
        "occupied_count": occupied,
        "observed_count": observed,
        "total_count": total,
        "observed_ratio": float(observed / total) if total else 0.0,
    }


def _project_topdown(grid: np.ndarray) -> np.ndarray:
    occupied = np.any(grid == OCCUPIED, axis=2)
    free = np.any(grid == FREE, axis=2)
    topdown = np.full(grid.shape[:2], UNKNOWN, dtype=np.int8)
    topdown[free] = FREE
    topdown[occupied] = OCCUPIED
    return topdown


def _draw_state_colorbar(fig, ax) -> None:
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=STATE_NORM, cmap=STATE_CMAP),
        ax=ax,
        ticks=[UNKNOWN, FREE, OCCUPIED],
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(["unknown", "free", "occupied"])


def _parse_step(path: Path) -> int:
    match = re.search(r"observed_state_step(\d+)\.npy$", path.name)
    if match is None:
        return -1
    return int(match.group(1))


def _observed_paths(input_dir: Path) -> list[Path]:
    paths = sorted(input_dir.glob("observed_state_step*.npy"), key=_parse_step)
    if len(paths) < 1:
        raise FileNotFoundError(f"No observed_state_step*.npy files found in {input_dir}")
    return paths


def _bounds_from_summary(summary: dict[str, Any], metadata: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = summary.get("map_bounds") or metadata.get("map_bounds")
    if raw is None:
        raise KeyError("Observed summary or scene metadata must contain map_bounds")
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def _voxel_size_from_summary(summary: dict[str, Any], metadata: dict[str, Any], grid: np.ndarray) -> float:
    if "voxel_size" in summary:
        return float(summary["voxel_size"])
    if "voxel_size_recommended" in metadata:
        return float(metadata["voxel_size_recommended"])
    bounds = _bounds_from_summary(summary, metadata)
    return float((bounds["x"][1] - bounds["x"][0]) / grid.shape[0])


def _save_observed_topdown_compare(
    path: Path,
    grids: list[np.ndarray],
    labels: list[str],
    bounds: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, axes = plt.subplots(1, len(grids), figsize=(4.7 * len(grids), 4.8), constrained_layout=True)
    if len(grids) == 1:
        axes = [axes]
    summaries = []
    for ax, grid, label in zip(axes, grids, labels):
        counts = _state_counts(grid)
        summaries.append({"label": label, **counts})
        ax.imshow(_project_topdown(grid).T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nobs {counts['observed_ratio']:.3f}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    _draw_state_colorbar(fig, axes[-1])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return summaries


def _sample_indices(indices: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(indices) <= int(max_points):
        return indices
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(indices), size=int(max_points), replace=False)
    return indices[np.sort(choice)]


def _voxel_centers(indices: np.ndarray, bounds: dict[str, tuple[float, float]], voxel_size: float) -> np.ndarray:
    mins = np.array([bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=np.float64)
    return mins + (indices.astype(np.float64) + 0.5) * float(voxel_size)


def _set_3d_axes(ax, bounds: dict[str, tuple[float, float]]) -> None:
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=25, azim=-48)


def _save_final_3d_scatter(
    path: Path,
    grid: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    max_points: int,
) -> dict[str, Any]:
    occupied_idx = np.argwhere(grid == OCCUPIED)
    free_idx = np.argwhere(grid == FREE)
    if occupied_idx.size == 0 or free_idx.size == 0:
        raise ValueError("Final observed map must contain both FREE and OCCUPIED voxels for 3D scatter")
    occupied_sample = _sample_indices(occupied_idx, max_points=max_points, seed=23)
    free_sample = _sample_indices(free_idx, max_points=max_points, seed=29)
    occupied_xyz = _voxel_centers(occupied_sample, bounds, voxel_size)
    free_xyz = _voxel_centers(free_sample, bounds, voxel_size)
    fig = plt.figure(figsize=(8.5, 7.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(free_xyz[:, 0], free_xyz[:, 1], free_xyz[:, 2], s=2, c="#76b7c4", alpha=0.16, depthshade=False, label="free")
    ax.scatter(occupied_xyz[:, 0], occupied_xyz[:, 1], occupied_xyz[:, 2], s=6, c="#c95c5c", alpha=0.95, depthshade=False, label="occupied")
    _set_3d_axes(ax, bounds)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Final observed map\nfree {len(free_idx)} ({len(free_sample)} shown), "
        f"occupied {len(occupied_idx)} ({len(occupied_sample)} shown)"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return {
        "free_total": int(len(free_idx)),
        "free_shown": int(len(free_sample)),
        "occupied_total": int(len(occupied_idx)),
        "occupied_shown": int(len(occupied_sample)),
    }


def _save_final_slices(
    path: Path,
    grid: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    z_meters: list[float],
) -> list[dict[str, Any]]:
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, axes = plt.subplots(1, len(z_meters), figsize=(4.8 * len(z_meters), 4.6), constrained_layout=True)
    if len(z_meters) == 1:
        axes = [axes]
    summaries = []
    for ax, z_meter in zip(axes, z_meters):
        z_idx = int(round((float(z_meter) - bounds["z"][0]) / float(voxel_size) - 0.5))
        z_idx = int(np.clip(z_idx, 0, grid.shape[2] - 1))
        actual_z = bounds["z"][0] + (z_idx + 0.5) * float(voxel_size)
        slice_xy = grid[:, :, z_idx]
        summaries.append({"z_requested_m": float(z_meter), "z_index": z_idx, "z_center_m": float(actual_z), **_state_counts(slice_xy[:, :, None])})
        ax.imshow(slice_xy.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"z={actual_z:.2f} m")
    _draw_state_colorbar(fig, axes[-1])
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return summaries


def save_observed_map_visualizations(
    input_dir: Path,
    output_dir: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    summary_path = input_dir / "observed_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing observed_summary.json in {input_dir}")
    observed_summary = _load_json(summary_path)
    paths = _observed_paths(input_dir)
    grids = [np.load(path) for path in paths]
    if len({grid.shape for grid in grids}) != 1:
        raise ValueError(f"Observed state shapes do not match: {[grid.shape for grid in grids]}")
    final_grid = grids[-1]
    bounds = _bounds_from_summary(observed_summary, metadata)
    voxel_size = _voxel_size_from_summary(observed_summary, metadata, final_grid)

    topdown_path = output_dir / "observed_topdown_compare.png"
    labels = [f"step {_parse_step(path)}" for path in paths]
    topdown_counts = _save_observed_topdown_compare(topdown_path, grids, labels, bounds)

    scatter_path = output_dir / "free_occupied_voxels_3d_final.png"
    scatter_summary = _save_final_3d_scatter(scatter_path, final_grid, bounds, voxel_size, int(args_cli.max_points))

    slices_path = output_dir / "slices_final.png"
    slice_summary = _save_final_slices(slices_path, final_grid, bounds, voxel_size, [0.2, 0.8, 1.2, 1.8])

    final_counts = _state_counts(final_grid)
    return (
        {
            "observed_topdown_compare": str(topdown_path),
            "free_occupied_voxels_3d_final": str(scatter_path),
            "slices_final": str(slices_path),
        },
        {
            "observed_summary_file": str(summary_path),
            "observed_state_files": [str(path) for path in paths],
            "observed_map_shape": [int(v) for v in final_grid.shape],
            "map_bounds": {axis: [bounds[axis][0], bounds[axis][1]] for axis in ("x", "y", "z")},
            "voxel_size": float(voxel_size),
            "counts_by_step": topdown_counts,
            "final_counts": final_counts,
            "scatter_sampling": scatter_summary,
            "slice_summary": slice_summary,
        },
    )


def _capture_view(
    camera: Camera,
    sim: sim_utils.SimulationContext,
    all_cameras: list[Camera],
    position: list[float],
    target: list[float],
    label: str,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any], dict[str, Any]]:
    _set_camera_view(camera, sim, position, target)
    _settle_and_update(sim, all_cameras, int(args_cli.settle_steps))
    rgb, rgb_key, rgb_stats = _extract_rgb(camera, label)
    depth, depth_stats = _extract_depth(camera, label)
    return rgb, depth, rgb_key, rgb_stats, depth_stats


def main() -> None:
    input_dir = Path(args_cli.input_dir).resolve()
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_metadata_path = input_dir / "scene_metadata.json"
    input_metadata = _load_json(input_metadata_path) if input_metadata_path.exists() else {}
    seed = int(input_metadata.get("seed", args_cli.seed))
    variant = str(input_metadata.get("variant", args_cli.variant))
    obstacle_jitter_m = float(input_metadata.get("obstacle_jitter_m", args_cli.obstacle_jitter_m))

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([7.0, -8.0, 6.0], [0.0, -0.2, 0.8])

    _add_lighting()
    scene_metadata = build_medium_complex_scene(
        seed=seed,
        variant=variant,
        obstacle_jitter_m=obstacle_jitter_m,
        spawn=True,
        sim_utils_module=sim_utils,
    )
    if input_metadata.get("camera_poses"):
        scene_metadata["camera_poses"] = input_metadata["camera_poses"]
    if input_metadata.get("overview_pose"):
        scene_metadata["overview_pose"] = input_metadata["overview_pose"]

    sim_utils.create_prim("/World/CameraRig", "Xform")
    main_camera = _make_camera(
        "/World/CameraRig/CameraSensor",
        width=int(args_cli.camera_width),
        height=int(args_cli.camera_height),
        max_depth=float(args_cli.max_depth),
        focal_length=18.0,
    )
    sim_utils.create_prim("/World/OverviewCameraRig", "Xform")
    overview_camera = _make_camera(
        "/World/OverviewCameraRig/CameraSensor",
        width=int(args_cli.overview_width),
        height=int(args_cli.overview_height),
        max_depth=float(args_cli.overview_max_depth),
        focal_length=22.0,
    )
    sim.reset()
    print("[INFO]: Stage 4A-3.2 medium scene visualization setup complete.")

    all_cameras = [main_camera, overview_camera]
    generated_files: dict[str, str] = {}
    camera_records: list[dict[str, Any]] = []
    camera_rgbs: list[np.ndarray] = []
    camera_depths: list[np.ndarray] = []
    labels: list[str] = []

    for pose in scene_metadata["camera_poses"]:
        idx = int(pose["index"])
        position = [float(v) for v in pose["position"]]
        target = _pose_target(position, float(pose["yaw_rad"]))
        rgb, depth, rgb_key, rgb_stats, depth_stats = _capture_view(
            main_camera,
            sim,
            all_cameras,
            position,
            target,
            f"camera pose {idx}",
        )
        rgb_path = output_dir / f"camera_rgb_{idx:03d}.png"
        depth_color_path = output_dir / f"camera_depth_color_{idx:03d}.png"
        _save_rgb_png(rgb_path, rgb)
        _save_depth_color_png(depth_color_path, depth, f"Camera depth {idx:03d}")
        generated_files[f"camera_rgb_{idx:03d}"] = str(rgb_path)
        generated_files[f"camera_depth_color_{idx:03d}"] = str(depth_color_path)
        camera_rgbs.append(rgb)
        camera_depths.append(depth)
        labels.append(f"cam {idx}")
        record = {
            "index": idx,
            "position": position,
            "yaw_rad": float(pose["yaw_rad"]),
            "yaw_deg": float(pose["yaw_deg"]),
            "note": str(pose.get("note", "")),
            "room": str(pose.get("room", "")),
            "rgb_file": str(rgb_path),
            "depth_color_file": str(depth_color_path),
            "rgb_key_used": rgb_key,
            "rgb_stats": rgb_stats,
            "depth_stats": depth_stats,
            "camera_output_keys": list(main_camera.data.output.keys()),
        }
        camera_records.append(record)
        print(
            f"[INFO]: saved camera {idx:03d} rgb={rgb_path.name} depth={depth_color_path.name} "
            f"depth_min={depth_stats['min']:.3f} depth_max={depth_stats['max']:.3f}"
        )

    rgb_grid_path = output_dir / "camera_rgb_grid.png"
    depth_grid_path = output_dir / "camera_depth_grid.png"
    _save_rgb_grid(rgb_grid_path, camera_rgbs, labels)
    _save_depth_grid(depth_grid_path, camera_depths, labels)
    generated_files["camera_rgb_grid"] = str(rgb_grid_path)
    generated_files["camera_depth_grid"] = str(depth_grid_path)

    overview_pose = scene_metadata["overview_pose"]
    overview_position = [float(v) for v in overview_pose["position"]]
    overview_target = [float(v) for v in overview_pose["target"]]
    overview_rgb, overview_depth, overview_rgb_key, overview_rgb_stats, overview_depth_stats = _capture_view(
        overview_camera,
        sim,
        all_cameras,
        overview_position,
        overview_target,
        "overview camera",
    )
    overview_rgb_path = output_dir / "scene_overview_rgb.png"
    overview_depth_path = output_dir / "scene_overview_depth_color.png"
    _save_rgb_png(overview_rgb_path, overview_rgb)
    _save_depth_color_png(overview_depth_path, overview_depth, "Overview depth")
    generated_files["scene_overview_rgb"] = str(overview_rgb_path)
    generated_files["scene_overview_depth_color"] = str(overview_depth_path)
    print(
        f"[INFO]: saved overview rgb={overview_rgb_path.name} depth={overview_depth_path.name} "
        f"depth_min={overview_depth_stats['min']:.3f} depth_max={overview_depth_stats['max']:.3f}"
    )

    layout_path = output_dir / "scene_layout_topdown.png"
    save_layout_topdown(layout_path, scene_metadata)
    generated_files["scene_layout_topdown"] = str(layout_path)

    observed_files, observed_viz_summary = save_observed_map_visualizations(input_dir, output_dir, scene_metadata)
    generated_files.update(observed_files)

    scene_metadata_path = output_dir / "scene_metadata.json"
    _save_json(scene_metadata_path, scene_metadata)
    generated_files["scene_metadata"] = str(scene_metadata_path)

    summary_path = output_dir / "scene_viz_summary.json"
    generated_files["scene_viz_summary"] = str(summary_path)
    summary = {
        "stage": "Stage 4A-3.2-scene-viz",
        "completed": True,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": seed,
        "variant": variant,
        "generated_files": generated_files,
        "camera_records": camera_records,
        "overview_record": {
            "position": overview_position,
            "target": overview_target,
            "rgb_file": str(overview_rgb_path),
            "depth_color_file": str(overview_depth_path),
            "rgb_key_used": overview_rgb_key,
            "rgb_stats": overview_rgb_stats,
            "depth_stats": overview_depth_stats,
            "camera_output_keys": list(overview_camera.data.output.keys()),
        },
        "observed_map": observed_viz_summary,
        "scene_counts": {
            "rooms": len(scene_metadata.get("rooms", [])),
            "corridors": len(scene_metadata.get("corridors", [])),
            "openings": len(scene_metadata.get("openings", [])),
            "walls": len(scene_metadata.get("walls", [])),
            "obstacles": len(scene_metadata.get("obstacles", [])),
            "camera_poses": len(scene_metadata.get("camera_poses", [])),
        },
        "leakage_checks": {
            "prediction_used": False,
            "prediction_wrote_observed_map": False,
            "target_lr_used": False,
            "target_hr_used": False,
            "ground_truth_used_for_exploration": False,
            "rl_or_ppo_training": False,
            "behavior_cloning_training": False,
            "imitation_learning_training": False,
            "sscnet_training": False,
        },
        "limitations": [
            "headless render only",
            "synthetic cuboid-only scripted scene",
            "no expert rollout, no prediction, no RL/PPO, no imitation-learning training",
        ],
    }
    _save_json(summary_path, summary)
    print(f"[INFO]: wrote medium scene visualizations to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
