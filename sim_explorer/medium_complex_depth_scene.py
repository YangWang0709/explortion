#!/usr/bin/env python3
"""Stage 4A-3.2 medium-complexity Isaac depth scene smoke capture.

This script only builds a scripted scene and captures measured RGB/depth from
fixed camera poses. It does not call SSCNet, PredictionLayer, RL, PPO, behavior
cloning, imitation learning, or expert rollout code.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 4A-3.2 medium-complexity Isaac depth scene.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke",
    help="Directory for RGB/depth, poses, camera metadata, and scene metadata.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--variant", default="three_rooms")
parser.add_argument("--obstacle_jitter_m", type=float, default=0.0)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=8.0)
parser.add_argument("--settle_steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

from scene_factory import build_medium_complex_scene

DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def _make_camera() -> Camera:
    sim_utils.create_prim("/World/CameraRig", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CameraRig/CameraSensor",
        update_period=0.0,
        height=int(args_cli.camera_height),
        width=int(args_cli.camera_width),
        data_types=["rgb", DEPTH_KEY],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, float(args_cli.max_depth)),
        ),
    )
    return Camera(cfg=camera_cfg)


def _add_lighting() -> None:
    dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.82, 0.84, 0.80))
    dome_cfg.func("/World/Light", dome_cfg)


def _set_camera_pose(camera: Camera, sim: sim_utils.SimulationContext, pose: dict[str, Any]) -> None:
    position = [float(v) for v in pose["position"]]
    yaw_rad = float(pose.get("yaw_rad", math.radians(float(pose["yaw_deg"]))))
    target = _pose_target(position, yaw_rad)
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )


def _settle(camera: Camera, sim: sim_utils.SimulationContext, steps: int) -> None:
    for _ in range(max(int(steps), 1)):
        sim.step()
        camera.update(dt=sim.get_physics_dt())


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
            "dtype_after_save_conversion": str(rgb.dtype),
            "min": int(rgb.min()) if rgb.size else None,
            "max": int(rgb.max()) if rgb.size else None,
            "mean": float(rgb.mean()) if rgb.size else None,
            "std": float(rgb.std()) if rgb.size else None,
        }
        if rgb.size == 0 or stats["max"] is None or int(stats["max"]) <= 2 or float(stats["std"] or 0.0) < 1.0:
            raise ValueError(f"{label}: RGB image appears blank or nearly uniform: {stats}")
        return rgb, key, stats
    raise KeyError(f"{label}: camera output missing RGB/RGBA. Keys: {list(camera.data.output.keys())}")


def _camera_info(camera: Camera) -> dict[str, Any]:
    intrinsic_matrix = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
    return {
        "sensor_api_depth_key": DEPTH_KEY,
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


def main() -> None:
    output_dir = Path(args_cli.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([7.0, -8.0, 6.0], [0.0, -0.2, 0.8])

    _add_lighting()
    scene_metadata = build_medium_complex_scene(
        seed=int(args_cli.seed),
        variant=str(args_cli.variant),
        obstacle_jitter_m=float(args_cli.obstacle_jitter_m),
        spawn=True,
        sim_utils_module=sim_utils,
    )
    camera = _make_camera()
    sim.reset()
    print("[INFO]: Stage 4A-3.2 medium-complexity scene setup complete.")

    capture_records: list[dict[str, Any]] = []
    pose_records: list[dict[str, Any]] = []
    for pose in scene_metadata["camera_poses"]:
        idx = int(pose["index"])
        _set_camera_pose(camera, sim, pose)
        _settle(camera, sim, args_cli.settle_steps)

        depth, depth_stats = _extract_depth(camera, f"camera pose {idx}")
        rgb, rgb_key, rgb_stats = _extract_rgb(camera, f"camera pose {idx}")

        depth_path = output_dir / f"depth_{idx:03d}.npy"
        rgb_path = output_dir / f"rgb_{idx:03d}.png"
        pose_path = output_dir / f"pose_{idx:03d}.json"
        np.save(depth_path, depth)
        Image.fromarray(rgb).save(rgb_path)

        pose_record = {
            "index": idx,
            "position": [float(v) for v in pose["position"]],
            "yaw_rad": float(pose["yaw_rad"]),
            "yaw_deg": float(pose["yaw_deg"]),
            "target": _pose_target([float(v) for v in pose["position"]], float(pose["yaw_rad"])),
            "note": str(pose.get("note", "")),
            "room": str(pose.get("room", "")),
            "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
        }
        if hasattr(camera.data, "pos_w"):
            pose_record["isaac_camera_pos_w"] = camera.data.pos_w[0].detach().cpu().numpy().astype(float).tolist()
        if hasattr(camera.data, "quat_w_ros"):
            pose_record["isaac_camera_quat_w_ros"] = (
                camera.data.quat_w_ros[0].detach().cpu().numpy().astype(float).tolist()
            )
        _save_json(pose_path, pose_record)
        pose_records.append(pose_record)

        capture = {
            "index": idx,
            "depth_file": str(depth_path),
            "rgb_file": str(rgb_path),
            "pose_file": str(pose_path),
            "rgb_key_used": rgb_key,
            "camera_output_keys": list(camera.data.output.keys()),
            "depth_stats": depth_stats,
            "rgb_stats": rgb_stats,
        }
        capture_records.append(capture)
        print(
            f"[INFO]: saved pose {idx:03d} depth={depth_path.name} rgb={rgb_path.name} "
            f"depth_min={depth_stats['min']:.3f} depth_max={depth_stats['max']:.3f}"
        )

    camera_info = _camera_info(camera)
    _save_json(output_dir / "camera_info.json", camera_info)

    scene_metadata.update(
        {
            "stage": "Stage 4A-3.2",
            "output_dir": str(output_dir),
            "camera": {
                **camera_info,
                "data_types_requested": ["rgb", DEPTH_KEY],
                "poses": pose_records,
            },
            "poses": pose_records,
            "depth_summaries": [record["depth_stats"] | {"index": record["index"]} for record in capture_records],
            "capture_records": capture_records,
            "prediction_used": False,
            "expert_used": False,
            "observed_map_modified": False,
            "rl_or_il_training_used": False,
            "sscnet_training_used": False,
        }
    )
    _save_json(output_dir / "scene_metadata.json", scene_metadata)
    print(f"[INFO]: wrote medium-complexity smoke outputs to {output_dir}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
