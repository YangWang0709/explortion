#!/usr/bin/env python3
"""Minimal Isaac depth scene for Stage 4A-1.

This script only validates simulator depth observation. It does not call
SSCNet, PredictionLayer, the expert scorer, RL, or IL training code.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 4A-1 minimal Isaac depth scene.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke",
    help="Directory for depth images, poses, and camera metadata.",
)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=5.0)
parser.add_argument("--settle_steps", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg


def _spawn_box(path: str, size: tuple[float, float, float], position: tuple[float, float, float], color) -> None:
    cfg = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    cfg.func(path, cfg, translation=position)


def design_scene() -> Camera:
    ground_cfg = sim_utils.GroundPlaneCfg(size=(8.0, 8.0))
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim_utils.create_prim("/World/Walls", "Xform")
    _spawn_box("/World/Walls/Left", (8.0, 0.15, 2.0), (0.0, -4.0, 1.0), (0.55, 0.58, 0.62))
    _spawn_box("/World/Walls/Right", (8.0, 0.15, 2.0), (0.0, 4.0, 1.0), (0.55, 0.58, 0.62))
    _spawn_box("/World/Walls/Back", (0.15, 8.0, 2.0), (4.0, 0.0, 1.0), (0.50, 0.54, 0.58))

    sim_utils.create_prim("/World/Obstacles", "Xform")
    _spawn_box("/World/Obstacles/Center", (0.7, 0.7, 1.2), (1.7, 0.35, 0.6), (0.8, 0.32, 0.24))
    _spawn_box("/World/Obstacles/Back", (0.8, 1.2, 1.5), (3.2, -1.2, 0.75), (0.24, 0.50, 0.82))

    sim_utils.create_prim("/World/CameraRig", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CameraRig/CameraSensor",
        update_period=0.0,
        height=args_cli.camera_height,
        width=args_cli.camera_width,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, args_cli.max_depth),
        ),
    )
    return Camera(cfg=camera_cfg)


def _pose_target(position: tuple[float, float, float], yaw_deg: float) -> tuple[float, float, float]:
    yaw = math.radians(yaw_deg)
    return (
        position[0] + math.cos(yaw),
        position[1] + math.sin(yaw),
        position[2],
    )


def _depth_stats(depth: np.ndarray) -> dict[str, float | int | list[int]]:
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    return {
        "shape": list(depth.shape),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
    }


def _save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main() -> None:
    output_dir = Path(args_cli.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([5.0, -5.0, 4.0], [0.0, 0.0, 0.5])

    camera = design_scene()
    sim.reset()
    print("[INFO]: Minimal depth scene setup complete.")

    poses = [
        {"index": 0, "position": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
        {"index": 1, "position": [0.0, 0.0, 1.2], "yaw_deg": 90.0},
        {"index": 2, "position": [1.0, 0.0, 1.2], "yaw_deg": 0.0},
    ]
    depth_summaries = []

    for pose in poses:
        position = tuple(float(v) for v in pose["position"])
        yaw_deg = float(pose["yaw_deg"])
        target = _pose_target(position, yaw_deg)
        camera.set_world_poses_from_view(
            eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
            targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
        )
        for _ in range(max(args_cli.settle_steps, 1)):
            sim.step()
            camera.update(dt=sim.get_physics_dt())

        depth_tensor = camera.data.output.get("distance_to_image_plane")
        if depth_tensor is None:
            raise KeyError(f"Camera output missing distance_to_image_plane. Keys: {list(camera.data.output.keys())}")
        depth = depth_tensor[0].detach().cpu().numpy().astype(np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        idx = int(pose["index"])
        depth_path = output_dir / f"depth_{idx:03d}.npy"
        pose_path = output_dir / f"pose_{idx:03d}.json"
        np.save(depth_path, depth)

        pose_record = {
            "index": idx,
            "position": [float(v) for v in position],
            "yaw_deg": yaw_deg,
            "yaw_rad": math.radians(yaw_deg),
            "target": [float(v) for v in target],
            "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
        }
        if hasattr(camera.data, "pos_w"):
            pose_record["isaac_camera_pos_w"] = camera.data.pos_w[0].detach().cpu().tolist()
        if hasattr(camera.data, "quat_w_ros"):
            pose_record["isaac_camera_quat_w_ros"] = camera.data.quat_w_ros[0].detach().cpu().tolist()
        _save_json(pose_path, pose_record)

        stats = _depth_stats(depth)
        stats.update({"index": idx, "depth_file": str(depth_path), "pose_file": str(pose_path)})
        depth_summaries.append(stats)
        print(
            f"[INFO]: saved depth_{idx:03d}.npy shape={tuple(depth.shape)} "
            f"dtype={depth.dtype} min={stats['min']} max={stats['max']}"
        )

    intrinsic_matrix = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
    camera_info = {
        "sensor_api_depth_key": "distance_to_image_plane",
        "depth_units": "meters",
        "width": int(args_cli.camera_width),
        "height": int(args_cli.camera_height),
        "max_depth": float(args_cli.max_depth),
        "near_depth": 0.05,
        "horizontal_fov_deg": 90.0,
        "intrinsic_matrix": intrinsic_matrix.tolist(),
        "fx": float(intrinsic_matrix[0, 0]),
        "fy": float(intrinsic_matrix[1, 1]),
        "cx": float(intrinsic_matrix[0, 2]),
        "cy": float(intrinsic_matrix[1, 2]),
    }
    _save_json(output_dir / "camera_info.json", camera_info)

    scene_metadata = {
        "stage": "Stage 4A-1",
        "scene": "minimal indoor-like Isaac depth smoke",
        "floor_size_m": [8.0, 8.0],
        "camera_height_m": 1.2,
        "poses": poses,
        "depth_summaries": depth_summaries,
        "map_bounds": {"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z": [0.0, 3.0]},
        "voxel_size_recommended": 0.1,
        "prediction_used": False,
        "expert_used": False,
        "rl_or_il_training_used": False,
    }
    _save_json(output_dir / "scene_metadata.json", scene_metadata)
    print(f"[INFO]: Wrote smoke outputs to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
