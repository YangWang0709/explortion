#!/usr/bin/env python3
"""Render Stage 4A-1 minimal Isaac scene RGB/depth visualizations.

This script visualizes the scripted Isaac scene itself. It does not call
SSCNet, PredictionLayer, expert scoring, RL, or imitation-learning code.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 4A-1 minimal Isaac scene visualization.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_scene_viz",
    help="Directory for rendered scene visualizations and metadata.",
)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=5.0)
parser.add_argument("--overview_width", type=int, default=640)
parser.add_argument("--overview_height", type=int, default=480)
parser.add_argument("--overview_max_depth", type=float, default=12.0)
parser.add_argument("--settle_steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")

FLOOR_SPEC: dict[str, Any] = {
    "name": "floor",
    "size": [8.0, 8.0],
    "bounds": {"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z": [0.0, 0.0]},
}

WALL_SPECS: list[dict[str, Any]] = [
    {
        "name": "left",
        "path": "/World/Walls/Left",
        "size": [8.0, 0.15, 2.0],
        "position": [0.0, -4.0, 1.0],
        "color": [0.55, 0.58, 0.62],
    },
    {
        "name": "right",
        "path": "/World/Walls/Right",
        "size": [8.0, 0.15, 2.0],
        "position": [0.0, 4.0, 1.0],
        "color": [0.55, 0.58, 0.62],
    },
    {
        "name": "back",
        "path": "/World/Walls/Back",
        "size": [0.15, 8.0, 2.0],
        "position": [4.0, 0.0, 1.0],
        "color": [0.50, 0.54, 0.58],
    },
]

OBSTACLE_SPECS: list[dict[str, Any]] = [
    {
        "name": "center",
        "path": "/World/Obstacles/Center",
        "size": [0.7, 0.7, 1.2],
        "position": [1.7, 0.35, 0.6],
        "color": [0.8, 0.32, 0.24],
    },
    {
        "name": "back",
        "path": "/World/Obstacles/Back",
        "size": [0.8, 1.2, 1.5],
        "position": [3.2, -1.2, 0.75],
        "color": [0.24, 0.50, 0.82],
    },
]

CAMERA_POSES: list[dict[str, Any]] = [
    {"index": 0, "position": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
    {"index": 1, "position": [0.0, 0.0, 1.2], "yaw_deg": 90.0},
    {"index": 2, "position": [1.0, 0.0, 1.2], "yaw_deg": 0.0},
]

OVERVIEW_POSE: dict[str, Any] = {
    "name": "overview",
    "position": [0.0, -6.5, 5.0],
    "target": [1.5, 0.0, 0.7],
}


def _spawn_box(path: str, size: list[float], position: list[float], color: list[float]) -> None:
    cfg = sim_utils.CuboidCfg(
        size=tuple(float(v) for v in size),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(float(v) for v in color)),
    )
    cfg.func(path, cfg, translation=tuple(float(v) for v in position))


def _make_camera(
    prim_path: str,
    width: int,
    height: int,
    max_depth: float,
    focal_length: float,
    horizontal_aperture: float,
) -> Camera:
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
            horizontal_aperture=float(horizontal_aperture),
            clipping_range=(0.05, float(max_depth)),
        ),
    )
    return Camera(cfg=camera_cfg)


def design_scene() -> tuple[Camera, Camera]:
    ground_cfg = sim_utils.GroundPlaneCfg(size=tuple(FLOOR_SPEC["size"]))
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Walls", "Xform")
    for wall in WALL_SPECS:
        _spawn_box(wall["path"], wall["size"], wall["position"], wall["color"])

    sim_utils.create_prim("/World/Obstacles", "Xform")
    for obstacle in OBSTACLE_SPECS:
        _spawn_box(obstacle["path"], obstacle["size"], obstacle["position"], obstacle["color"])

    sim_utils.create_prim("/World/CameraRig", "Xform")
    main_camera = _make_camera(
        "/World/CameraRig/CameraSensor",
        width=args_cli.camera_width,
        height=args_cli.camera_height,
        max_depth=args_cli.max_depth,
        focal_length=18.0,
        horizontal_aperture=36.0,
    )

    sim_utils.create_prim("/World/OverviewCameraRig", "Xform")
    overview_camera = _make_camera(
        "/World/OverviewCameraRig/CameraSensor",
        width=args_cli.overview_width,
        height=args_cli.overview_height,
        max_depth=args_cli.overview_max_depth,
        focal_length=20.0,
        horizontal_aperture=36.0,
    )
    return main_camera, overview_camera


def _pose_target(position: list[float], yaw_deg: float) -> list[float]:
    yaw = math.radians(float(yaw_deg))
    return [
        float(position[0]) + math.cos(yaw),
        float(position[1]) + math.sin(yaw),
        float(position[2]),
    ]


def _save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def _finite_tensor_list(tensor: torch.Tensor) -> list[float] | None:
    values = tensor.detach().cpu().numpy().astype(float)
    if not np.all(np.isfinite(values)):
        return None
    return values.tolist()


def _to_numpy_image(data: torch.Tensor) -> np.ndarray:
    image = data[0].detach().cpu().numpy()
    if image.ndim != 3 or image.shape[-1] not in (3, 4):
        raise ValueError(f"Expected RGB/RGBA image shape (H, W, 3/4), got {image.shape}")
    return image


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        max_value = float(finite.max()) if finite.size else 0.0
        if max_value <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _rgb_stats(image: np.ndarray, key: str, source_image: np.ndarray) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "key": key,
        "shape": list(image.shape),
        "dtype_after_save_conversion": str(image.dtype),
        "min": int(image.min()) if image.size else None,
        "max": int(image.max()) if image.size else None,
        "mean": float(image.mean()) if image.size else None,
        "std": float(image.std()) if image.size else None,
    }
    if key == "rgba" and source_image.shape[-1] == 4:
        alpha = source_image[..., 3]
        stats["alpha_min"] = float(np.nanmin(alpha))
        stats["alpha_max"] = float(np.nanmax(alpha))
    return stats


def _validate_rgb(image: np.ndarray, stats: dict[str, Any], label: str) -> None:
    if image.size == 0:
        raise ValueError(f"{label} RGB image is empty.")
    if stats["max"] is None or stats["min"] is None:
        raise ValueError(f"{label} RGB image has invalid stats: {stats}")
    if stats["max"] <= 2:
        raise ValueError(f"{label} RGB image appears all black: {stats}")
    if stats["min"] >= 253:
        raise ValueError(f"{label} RGB image appears all white: {stats}")
    if stats["std"] is not None and stats["std"] < 1.0:
        raise ValueError(f"{label} RGB image appears nearly uniform: {stats}")


def _extract_rgb(camera: Camera, label: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEY_CANDIDATES:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        source = _to_numpy_image(tensor)
        rgb = _normalize_rgb(source)
        stats = _rgb_stats(rgb, key, source)
        _validate_rgb(rgb, stats, label)
        return rgb, key, stats
    raise KeyError(f"Camera output missing RGB/RGBA data for {label}. Keys: {list(camera.data.output.keys())}")


def _extract_depth(camera: Camera, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    tensor = camera.data.output.get(DEPTH_KEY)
    if tensor is None:
        raise KeyError(f"Camera output missing {DEPTH_KEY} for {label}. Keys: {list(camera.data.output.keys())}")
    depth = tensor[0].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    if positive.size == 0:
        raise ValueError(f"{label} depth has no finite positive values.")
    stats = {
        "key": DEPTH_KEY,
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()),
        "max": float(positive.max()),
        "mean": float(positive.mean()),
    }
    return depth, stats


def _save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


def _save_depth_color_png(path: Path, depth: np.ndarray, title_prefix: str) -> None:
    finite_positive = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite_positive.size == 0:
        raise ValueError(f"Cannot render depth color image for {path}; no finite positive depth values.")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    vmin = float(finite_positive.min())
    vmax = float(finite_positive.max())
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax)
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_title(f"{title_prefix}: min {vmin:.3f} m, max {vmax:.3f} m")
    ax.set_xlabel("u pixel")
    ax.set_ylabel("v pixel")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _set_camera_view(camera: Camera, sim: sim_utils.SimulationContext, position: list[float], target: list[float]) -> None:
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )


def _settle_and_update(sim: sim_utils.SimulationContext, cameras: list[Camera], steps: int) -> None:
    for _ in range(max(int(steps), 1)):
        sim.step()
        for camera in cameras:
            camera.update(dt=sim.get_physics_dt())


def _box_xy(spec: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, _ = (float(v) for v in spec["position"])
    sx, sy, _ = (float(v) for v in spec["size"])
    return x - sx / 2.0, y - sy / 2.0, sx, sy


def _draw_camera_arrow(ax, position: list[float], target: list[float], label: str, color: str) -> None:
    px, py = float(position[0]), float(position[1])
    tx, ty = float(target[0]), float(target[1])
    dx, dy = tx - px, ty - py
    length = math.hypot(dx, dy)
    if length > 1.0e-6:
        dx, dy = dx / length * 0.45, dy / length * 0.45
    ax.arrow(px, py, dx, dy, color=color, width=0.025, head_width=0.16, length_includes_head=True)
    ax.scatter([px], [py], c=[color], s=35, zorder=5)
    ax.text(px + 0.08, py + 0.08, label, color=color, fontsize=9, weight="bold")


def save_layout_topdown(path: Path) -> None:
    floor = FLOOR_SPEC["bounds"]
    fig, ax = plt.subplots(figsize=(8.0, 8.0), constrained_layout=True)
    floor_x0, floor_x1 = floor["x"]
    floor_y0, floor_y1 = floor["y"]
    ax.add_patch(
        Rectangle(
            (floor_x0, floor_y0),
            floor_x1 - floor_x0,
            floor_y1 - floor_y0,
            facecolor="#edf2f7",
            edgecolor="#1f2937",
            linewidth=1.6,
            label="8m x 8m floor",
        )
    )

    for wall in WALL_SPECS:
        ax.add_patch(
            Rectangle(
                _box_xy(wall)[:2],
                _box_xy(wall)[2],
                _box_xy(wall)[3],
                facecolor="#8b949e",
                edgecolor="#4b5563",
                alpha=0.85,
                label="walls" if wall["name"] == "left" else None,
            )
        )

    obstacle_colors = {"center": "#d46f52", "back": "#4479b6"}
    for obstacle in OBSTACLE_SPECS:
        ax.add_patch(
            Rectangle(
                _box_xy(obstacle)[:2],
                _box_xy(obstacle)[2],
                _box_xy(obstacle)[3],
                facecolor=obstacle_colors.get(obstacle["name"], "#c08457"),
                edgecolor="#111827",
                alpha=0.88,
                label="obstacles" if obstacle["name"] == "center" else None,
            )
        )
        ax.text(
            float(obstacle["position"][0]),
            float(obstacle["position"][1]),
            obstacle["name"],
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
        )

    for pose in CAMERA_POSES:
        target = _pose_target(pose["position"], float(pose["yaw_deg"]))
        _draw_camera_arrow(ax, pose["position"], target, f"cam {pose['index']}", "#111827")

    _draw_camera_arrow(ax, OVERVIEW_POSE["position"], OVERVIEW_POSE["target"], "overview", "#0f766e")

    ax.set_title("Stage 4A-1 scripted Isaac scene layout")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-4.6, 4.8)
    ax.set_ylim(-7.2, 4.7)
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper left", frameon=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _capture_view(
    camera: Camera,
    sim: sim_utils.SimulationContext,
    all_cameras: list[Camera],
    position: list[float],
    target: list[float],
    label: str,
) -> tuple[np.ndarray, np.ndarray, str, dict[str, Any], dict[str, Any]]:
    _set_camera_view(camera, sim, position, target)
    _settle_and_update(sim, all_cameras, args_cli.settle_steps)
    rgb, rgb_key, rgb_stats = _extract_rgb(camera, label)
    depth, depth_stats = _extract_depth(camera, label)
    return rgb, depth, rgb_key, rgb_stats, depth_stats


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([5.0, -5.0, 4.0], [0.0, 0.0, 0.5])

    main_camera, overview_camera = design_scene()
    sim.reset()
    print("[INFO]: Stage 4A-1 scene visualization setup complete.")

    all_cameras = [main_camera, overview_camera]
    generated_files: dict[str, str] = {}
    camera_records: list[dict[str, Any]] = []

    for pose in CAMERA_POSES:
        idx = int(pose["index"])
        position = [float(v) for v in pose["position"]]
        target = _pose_target(position, float(pose["yaw_deg"]))
        rgb, depth, rgb_key, rgb_stats, depth_stats = _capture_view(
            main_camera, sim, all_cameras, position, target, f"camera pose {idx}"
        )

        rgb_path = output_dir / f"camera_rgb_{idx:03d}.png"
        depth_color_path = output_dir / f"camera_depth_color_{idx:03d}.png"
        _save_rgb_png(rgb_path, rgb)
        _save_depth_color_png(depth_color_path, depth, f"Camera depth {idx:03d}")
        generated_files[f"camera_rgb_{idx:03d}"] = str(rgb_path)
        generated_files[f"camera_depth_color_{idx:03d}"] = str(depth_color_path)

        record = {
            "index": idx,
            "position": position,
            "yaw_deg": float(pose["yaw_deg"]),
            "target": target,
            "rgb_file": str(rgb_path),
            "depth_color_file": str(depth_color_path),
            "rgb_key_used": rgb_key,
            "camera_output_keys": list(main_camera.data.output.keys()),
            "rgb_stats": rgb_stats,
            "depth_stats": depth_stats,
        }
        if hasattr(main_camera.data, "pos_w"):
            pos_w = _finite_tensor_list(main_camera.data.pos_w[0])
            if pos_w is not None:
                record["isaac_camera_pos_w"] = pos_w
        if hasattr(main_camera.data, "quat_w_ros"):
            quat_w_ros = _finite_tensor_list(main_camera.data.quat_w_ros[0])
            if quat_w_ros is not None:
                record["isaac_camera_quat_w_ros"] = quat_w_ros
        camera_records.append(record)
        print(
            f"[INFO]: saved camera {idx:03d} rgb={rgb_path.name} depth={depth_color_path.name} "
            f"rgb_key={rgb_key} depth_min={depth_stats['min']:.3f} depth_max={depth_stats['max']:.3f}"
        )

    overview_position = [float(v) for v in OVERVIEW_POSE["position"]]
    overview_target = [float(v) for v in OVERVIEW_POSE["target"]]
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
        f"rgb_key={overview_rgb_key} depth_min={overview_depth_stats['min']:.3f} "
        f"depth_max={overview_depth_stats['max']:.3f}"
    )

    layout_path = output_dir / "scene_layout_topdown.png"
    save_layout_topdown(layout_path)
    generated_files["scene_layout_topdown"] = str(layout_path)

    intrinsic_matrix = main_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
    overview_intrinsic_matrix = overview_camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
    scene_metadata = {
        "stage": "Stage 4A-1 Scene Visualization",
        "scene": "minimal indoor-like Isaac scripted scene",
        "floor": FLOOR_SPEC,
        "walls": WALL_SPECS,
        "obstacles": OBSTACLE_SPECS,
        "camera": {
            "width": int(args_cli.camera_width),
            "height": int(args_cli.camera_height),
            "max_depth": float(args_cli.max_depth),
            "near_depth": 0.05,
            "data_types_requested": ["rgb", DEPTH_KEY],
            "intrinsic_matrix": intrinsic_matrix.tolist(),
            "poses": CAMERA_POSES,
        },
        "overview_camera": {
            "width": int(args_cli.overview_width),
            "height": int(args_cli.overview_height),
            "max_depth": float(args_cli.overview_max_depth),
            "near_depth": 0.05,
            "pose": OVERVIEW_POSE,
            "rgb_key_used": overview_rgb_key,
            "rgb_stats": overview_rgb_stats,
            "depth_stats": overview_depth_stats,
            "intrinsic_matrix": overview_intrinsic_matrix.tolist(),
        },
        "prediction_used": False,
        "expert_used": False,
        "rl_or_il_training_used": False,
        "observed_map_modified": False,
    }
    scene_metadata_path = output_dir / "scene_metadata.json"
    _save_json(scene_metadata_path, scene_metadata)
    generated_files["scene_metadata"] = str(scene_metadata_path)

    summary_path = output_dir / "scene_viz_summary.json"
    generated_files["scene_viz_summary"] = str(summary_path)
    summary = {
        "stage": "Stage 4A-1-scene-viz",
        "completed": True,
        "output_dir": str(output_dir),
        "render_command_context": {
            "headless": bool(args_cli.headless),
            "enable_cameras": bool(args_cli.enable_cameras),
            "device": args_cli.device,
        },
        "generated_files": generated_files,
        "camera_records": camera_records,
        "overview_record": {
            "position": overview_position,
            "target": overview_target,
            "rgb_file": str(overview_rgb_path),
            "depth_color_file": str(overview_depth_path),
            "rgb_key_used": overview_rgb_key,
            "camera_output_keys": list(overview_camera.data.output.keys()),
            "rgb_stats": overview_rgb_stats,
            "depth_stats": overview_depth_stats,
        },
        "scene_metadata_file": str(scene_metadata_path),
        "limitations": [
            "headless render only",
            "minimal synthetic scene",
            "no expert, prediction, RL, PPO, or imitation-learning training",
        ],
    }
    _save_json(summary_path, summary)
    print(f"[INFO]: wrote scene visualization outputs to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
