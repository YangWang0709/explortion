#!/usr/bin/env python3
"""Stage 4A-6.5ai one-frame lambda48 runtime smoke.

This runner starts Isaac once, captures exactly one frame in the deterministic
medium_three_rooms scene, updates a new measured-only observed_state, runs
map_predict exactly once, and evaluates source-protected mini-RRT decisions.
The primary runtime formula is:

    value = gain_exp / cost + 48 * minmax(source_occ_free)

It does not execute the selected action, capture a second frame, run rollout,
train models, write prediction into observed_state, or use prediction for
traversability/collision/edge validity/ray blocking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, "1")

from isaaclab.app import AppLauncher


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ai_one_frame_lambda48_runtime_smoke"
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]
PROFILE_NAME = "source_like_crop_min_length_0p25"
DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")
HISTORICAL_PRIOR_SELECTED_GRID = [11, 15, 11]
HISTORICAL_PRIOR_BEST_GRID = [14, 15, 11]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
parser.add_argument("--scene_variant", default="medium_three_rooms")
parser.add_argument("--scene_seed", type=int, default=0)
parser.add_argument("--start_pose", default="canonical_stage4a65p_frame1")
parser.add_argument("--position", default="-4.65,-4.65,1.2")
parser.add_argument("--yaw", type=float, default=0.38710316317995463)
parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
parser.add_argument("--alignment_convention", choices=["code_consistent_v1"], default="code_consistent_v1")
parser.add_argument("--tau", type=float, default=0.1)
parser.add_argument("--occ_threshold", type=float, default=0.5)
parser.add_argument("--free_threshold", type=float, default=0.5)
parser.add_argument("--lambda_sc", type=float, default=48.0)
parser.add_argument("--shadow_lambda_sc", type=float, default=32.0)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=5.0)
parser.add_argument("--settle_steps", type=int, default=12)
parser.add_argument("--pixel_stride", type=int, default=2)
parser.add_argument("--num_nodes", type=int, default=256)
parser.add_argument("--max_extension_m", type=float, default=0.5)
parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
parser.add_argument("--v_max", type=float, default=1.0)
parser.add_argument("--robot_radius_m", type=float, default=0.2)
parser.add_argument("--voxel_size", type=float, default=0.1)
parser.add_argument("--raycast_stride", type=int, default=2)
parser.add_argument("--num_yaw_samples", type=int, default=8)
parser.add_argument("--max_ray_length_m", type=float, default=4.8)
parser.add_argument("--short_edge_policy", choices=["crop"], default="crop")
parser.add_argument("--crop_min_length_m", type=float, default=0.25)
parser.add_argument("--tree_seed", type=int, default=0)
parser.add_argument("--max_workers", type=int, default=32)
parser.add_argument("--torch_num_threads", type=int, default=1)
parser.add_argument("--save_viz", action="store_true")
parser.add_argument("--save_probs", action="store_true")
parser.add_argument("--no_action_execution", action="store_true")
parser.add_argument("--no_second_frame", action="store_true")
parser.add_argument("--no_rollout", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if hasattr(args_cli, "headless"):
    args_cli.headless = True
if hasattr(args_cli, "enable_cameras"):
    args_cli.enable_cameras = True

_isaac_start = time.perf_counter()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
ISAAC_APP_LAUNCH_TIME_S = float(time.perf_counter() - _isaac_start)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

SSC_EXPLORATION_DIR = WORKSPACE / "ssc_exploration"
SSC_NETWORK_DIR = SSC_EXPLORATION_DIR / "ssc_network"
for _path in (SSC_EXPLORATION_DIR, SSC_NETWORK_DIR, WORKSPACE / "sim_explorer"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from depth_to_voxel import (  # noqa: E402
    FREE,
    OCCUPIED,
    UNKNOWN,
    create_observed_grid,
    normalize_map_bounds,
    summarize_observed_grid,
    update_observed_state_from_depth,
)
from isaac_map_predictor import IsaacMapPredictor, sha256_array  # noqa: E402
from offline_mini_rrt_tree import (  # noqa: E402
    ROOT_ID,
    make_gain_value_rows,
    segment_path_to_root,
    segment_record,
    to_jsonable,
    write_csv,
    write_visualizations,
)
from run_real_frame_lambda48_formula_smoke import (  # noqa: E402
    build_tree,
    grid_distance_m,
    make_mode_configs,
    select_decision,
)
from run_synthetic_map_predict_calibration_smoke import (  # noqa: E402
    path_candidate_rows,
    precompute_segment_prediction_arrays,
    write_jsonl,
)
from scene_factory import build_medium_complex_scene  # noqa: E402
from sim_paper_expert import EmptyPredictionLayer, world_to_grid  # noqa: E402
from sim_prediction_layer import SimPredictionLayer  # noqa: E402


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def git_status_short(path: Path) -> str:
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


def parse_position(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError(f"--position must contain exactly 3 comma-separated values, got {raw!r}")
    return values


def pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def scene_variant_name(scene_variant: str) -> str:
    if scene_variant in {"medium_three_rooms", "three_rooms"}:
        return "three_rooms"
    raise ValueError(f"unsupported scene_variant: {scene_variant}")


def default_bounds() -> dict[str, tuple[float, float]]:
    return normalize_map_bounds({"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]})


def context_manifest(output_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    combined = ""
    for path in CONTEXT_FILES:
        text = read_text(path)
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "contains_stage4a65ag": "Stage 4A-6.5ag" in text,
                "contains_stage4a65ah": "Stage 4A-6.5ah" in text,
                "contains_hardware_policy": "Hardware utilization policy" in text
                or "os_cpu_count=32" in text
                or "os_cpu_count = 32" in text,
            }
        )
    manifest = {
        "stage": "Stage 4A-6.5ai",
        "loaded_context_files": entries,
        "confirmed_stage4a65ag_complete": "Stage 4A-6.5ag" in combined
        and "multi-frame lambda48 replay" in combined,
        "confirmed_stage4a65ah_complete": "Stage 4A-6.5ah" in combined
        and "runtime-smoke design review" in combined,
        "confirmed_hardware_policy": "max_workers 32" in combined or "os_cpu_count=32" in combined,
        "chat_history_not_used_as_source": True,
    }
    save_json(output_dir / "loaded_context_manifest.json", manifest)
    write_text(
        output_dir / "loaded_context_manifest.md",
        "\n".join(
            [
                "# Loaded Context Manifest",
                "",
                f"- Stage 4A-6.5ag complete confirmed: `{manifest['confirmed_stage4a65ag_complete']}`",
                f"- Stage 4A-6.5ah complete confirmed: `{manifest['confirmed_stage4a65ah_complete']}`",
                f"- Hardware policy confirmed: `{manifest['confirmed_hardware_policy']}`",
                "- Files read:",
                *[f"  - `{item['path']}` sha256 `{item['sha256']}`" for item in entries],
            ]
        ),
    )
    return manifest


def make_camera(args: argparse.Namespace) -> Camera:
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


def add_lighting() -> None:
    dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.82, 0.84, 0.80))
    dome_cfg.func("/World/Light", dome_cfg)


def make_pose(args: argparse.Namespace) -> dict[str, Any]:
    position = parse_position(args.position)
    yaw_rad = float(args.yaw)
    return {
        "index": 0,
        "frame": 0,
        "start_pose_name": str(args.start_pose),
        "position": position,
        "yaw_rad": yaw_rad,
        "yaw_deg": float(math.degrees(yaw_rad)),
        "target": pose_target(position, yaw_rad),
        "fixed_camera_height_m": float(position[2]),
        "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
    }


def set_camera_pose(camera: Camera, sim: sim_utils.SimulationContext, pose: dict[str, Any]) -> None:
    position = [float(v) for v in pose["position"]]
    target = [float(v) for v in pose["target"]]
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )


def settle(camera: Camera, sim: sim_utils.SimulationContext, steps: int) -> None:
    for _ in range(max(int(steps), 1)):
        sim.step()
        camera.update(dt=sim.get_physics_dt())


def normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def extract_rgb(camera: Camera) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEY_CANDIDATES:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        source = tensor[0].detach().cpu().numpy()
        if source.ndim != 3 or source.shape[-1] not in (3, 4):
            raise ValueError(f"expected RGB/RGBA image, got {source.shape}")
        rgb = normalize_rgb(source)
        stats = {
            "key": key,
            "shape": [int(v) for v in rgb.shape],
            "dtype_after_save_conversion": str(rgb.dtype),
            "min": int(rgb.min()) if rgb.size else None,
            "max": int(rgb.max()) if rgb.size else None,
            "mean": float(rgb.mean()) if rgb.size else None,
            "std": float(rgb.std()) if rgb.size else None,
        }
        if rgb.size == 0 or stats["max"] is None or int(stats["max"]) <= 2:
            raise ValueError(f"RGB image appears blank: {stats}")
        return rgb, key, stats
    raise KeyError(f"camera output missing RGB/RGBA. Keys: {list(camera.data.output.keys())}")


def extract_depth(camera: Camera) -> tuple[np.ndarray, dict[str, Any]]:
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


def camera_info(camera: Camera, args: argparse.Namespace) -> dict[str, Any]:
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


def save_depth_png(path: Path, depth: np.ndarray, title: str) -> None:
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


def start_scene(args: argparse.Namespace) -> tuple[sim_utils.SimulationContext, Camera, dict[str, Any]]:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([7.0, -8.0, 6.0], [0.0, -0.2, 0.8])
    add_lighting()
    scene_metadata = build_medium_complex_scene(
        seed=int(args.scene_seed),
        variant=scene_variant_name(str(args.scene_variant)),
        obstacle_jitter_m=0.0,
        spawn=True,
        sim_utils_module=sim_utils,
    )
    camera = make_camera(args)
    sim.reset()
    return sim, camera, scene_metadata


def capture_one_frame(
    args: argparse.Namespace,
    output_dir: Path,
    camera: Camera,
    sim: sim_utils.SimulationContext,
    pose: dict[str, Any],
) -> dict[str, Any]:
    set_camera_pose(camera, sim, pose)
    settle(camera, sim, int(args.settle_steps))
    depth, depth_stats = extract_depth(camera)
    rgb, rgb_key, rgb_stats = extract_rgb(camera)
    info = camera_info(camera, args)

    pose_path = output_dir / "capture_pose_000.json"
    depth_path = output_dir / "capture_depth_000.npy"
    depth_png_path = output_dir / "capture_depth_000.png"
    rgb_path = output_dir / "capture_rgb_000.png"
    camera_info_path = output_dir / "capture_camera_info_000.json"

    save_json(pose_path, pose)
    np.save(depth_path, depth)
    save_depth_png(depth_png_path, depth, "Stage 4A-6.5ai one-frame capture")
    Image.fromarray(rgb).save(rgb_path)
    save_json(camera_info_path, info)

    return {
        "frame_index": 0,
        "frames_captured": 1,
        "depth": depth,
        "depth_path": str(depth_path),
        "depth_png_path": str(depth_png_path),
        "rgb_path": str(rgb_path),
        "pose": pose,
        "pose_path": str(pose_path),
        "camera_info": info,
        "camera_info_path": str(camera_info_path),
        "depth_stats": depth_stats,
        "rgb_stats": rgb_stats,
        "rgb_key_used": rgb_key,
        "camera_output_keys": list(camera.data.output.keys()),
    }


def update_observed_state(
    args: argparse.Namespace,
    output_dir: Path,
    capture: dict[str, Any],
) -> dict[str, Any]:
    bounds = default_bounds()
    empty = create_observed_grid(bounds, voxel_size=float(args.voxel_size))
    empty_hash = sha256_array(empty)
    empty_summary = summarize_observed_grid(empty)
    updated = np.array(empty, dtype=np.int8, copy=True)
    updated = update_observed_state_from_depth(
        observed_state=updated,
        depth=np.asarray(capture["depth"], dtype=np.float32),
        camera_pose=capture["pose"],
        camera_info=capture["camera_info"],
        bounds=bounds,
        voxel_size=float(args.voxel_size),
        pixel_stride=int(args.pixel_stride),
    )
    observed_path = output_dir / "observed_state_frame000.npy"
    np.save(observed_path, updated)
    updated_hash = sha256_file(observed_path)
    updated_summary = summarize_observed_grid(updated)
    return {
        "input_prior_kind": "new_empty_observed_grid_in_memory",
        "input_prior_path": None,
        "input_prior_sha256": empty_hash,
        "input_prior_summary": empty_summary,
        "new_observed_state": str(observed_path),
        "new_observed_state_sha256": updated_hash,
        "new_observed_state_shape": [int(v) for v in updated.shape],
        "updated_summary": updated_summary,
        "delta_observed_count": int(updated_summary["observed_count"] - empty_summary["observed_count"]),
        "delta_observed_ratio": float(updated_summary["observed_ratio"] - empty_summary["observed_ratio"]),
        "bounds": {axis: [float(bounds[axis][0]), float(bounds[axis][1])] for axis in ("x", "y", "z")},
        "voxel_size": float(args.voxel_size),
        "pixel_stride": int(args.pixel_stride),
        "measured_only": True,
        "prediction_used": False,
        "map_predict_used": False,
        "existing_observed_state_inputs": [],
        "existing_observed_state_modified": False,
    }


def run_prediction(
    args: argparse.Namespace,
    output_dir: Path,
    capture: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    observed_state = np.load(observed["new_observed_state"])
    predictor = IsaacMapPredictor(
        checkpoint=args.checkpoint,
        device="cuda",
        tau=float(args.tau),
        torch_num_threads=int(args.torch_num_threads),
        alignment_convention=str(args.alignment_convention),
    )
    result = predictor.predict_step(
        depth=np.asarray(capture["depth"], dtype=np.float32),
        pose=capture["pose"],
        camera_info=capture["camera_info"],
        observed_state=observed_state,
        map_bounds=observed["bounds"],
        voxel_size=float(args.voxel_size),
        output_dir=output_dir / "map_predict",
        step=0,
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
    result["map_predict_call_count"] = 1
    return result


def prediction_stats(prediction_npz: str, observed_state: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    layer = SimPredictionLayer.from_npz(prediction_npz)
    with np.load(prediction_npz, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occupied_prob = np.asarray(data["global_occupied_prob"], dtype=np.float32)
        free_prob = np.asarray(data["global_free_prob"], dtype=np.float32)
        alignment_convention = str(np.asarray(data["alignment_convention"]).item())
    valid_tau = valid & (confidence >= float(args.tau))
    unmeasured = observed_state == UNKNOWN
    occ = valid_tau & unmeasured & (occupied_prob >= float(args.occ_threshold))
    free = valid_tau & unmeasured & (free_prob >= float(args.free_threshold))
    occ_free = occ | free
    return {
        "prediction_npz": str(prediction_npz),
        "shape": [int(v) for v in layer.shape()],
        "observed_state_shape": [int(v) for v in observed_state.shape],
        "shape_aligned_to_observed_state": tuple(layer.shape()) == tuple(observed_state.shape),
        "alignment_convention": alignment_convention,
        "tau": float(args.tau),
        "occ_threshold": float(args.occ_threshold),
        "free_threshold": float(args.free_threshold),
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "prediction_valid_tau_count": int(np.count_nonzero(valid_tau)),
        "predicted_unmeasured_occ_free_count": int(np.count_nonzero(occ_free)),
        "predicted_unmeasured_occupied_count": int(np.count_nonzero(occ)),
        "predicted_unmeasured_free_count": int(np.count_nonzero(free)),
        "class_prob_saved": False,
    }


def tree_args_for_lambda(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        v_max=float(args.v_max),
        robot_radius_m=float(args.robot_radius_m),
        voxel_size=float(args.voxel_size),
        raycast_stride=int(args.raycast_stride),
        num_yaw_samples=int(args.num_yaw_samples),
        max_ray_length_m=float(args.max_ray_length_m),
        tau=float(args.tau),
        short_edge_policy=str(args.short_edge_policy),
        crop_min_length_m=float(args.crop_min_length_m),
        lambda_sc=float(args.lambda_sc),
        occ_threshold=float(args.occ_threshold),
        free_threshold=float(args.free_threshold),
    )


def root_from_pose(pose: dict[str, Any], observed_shape: tuple[int, int, int], args: argparse.Namespace) -> tuple[list[int], list[float], float]:
    bounds = default_bounds()
    root_world = [float(v) for v in pose["position"]]
    root_grid = list(world_to_grid(root_world, bounds, float(args.voxel_size), shape=observed_shape, clip=True))
    root_yaw = float(pose["yaw_rad"])
    return [int(v) for v in root_grid], root_world, root_yaw


def build_and_score_trees(
    args: argparse.Namespace,
    output_dir: Path,
    observed_state: np.ndarray,
    prediction_layer: SimPredictionLayer,
    pose: dict[str, Any],
) -> dict[str, Any]:
    tree_args = tree_args_for_lambda(args)
    configs = make_mode_configs(tree_args)
    bounds = default_bounds()
    observed_shape = tuple(int(v) for v in observed_state.shape)
    root_grid, root_world, root_yaw = root_from_pose(pose, observed_shape, args)
    empty_layer = EmptyPredictionLayer(observed_shape)

    measured_start = time.perf_counter()
    measured_result = build_tree(
        observed_state=observed_state,
        root_grid=root_grid,
        root_world=root_world,
        root_yaw=root_yaw,
        bounds=bounds,
        seed=int(args.tree_seed),
        prediction_layer=empty_layer,
        gain_mode="exp",
        args=tree_args,
    )
    measured_time = float(time.perf_counter() - measured_start)
    measured_candidates = path_candidate_rows(measured_result["tree"], None, None)
    measured_decision = select_decision(
        measured_candidates,
        seed=int(args.tree_seed),
        mode="measured_only",
        config=configs["measured_only"],
        same_seed_measured=None,
        voxel_size=float(args.voxel_size),
    )
    measured_decision.update(
        {
            "changed_vs_measured_only": False,
            "branch_classification": "same_as_measured",
            "same_as_measured": True,
            "spatial_prior_sc_basin": False,
            "same_as_prior_low_cost_sc": False,
            "avoids_prior_low_cost_sc": True,
            "healthy_nonmeasured_candidate": False,
            "low_cost_artifact": False,
            "collapse_to_measured": True,
        }
    )

    primary_start = time.perf_counter()
    sc_result = build_tree(
        observed_state=observed_state,
        root_grid=root_grid,
        root_world=root_world,
        root_yaw=root_yaw,
        bounds=bounds,
        seed=int(args.tree_seed),
        prediction_layer=prediction_layer,
        gain_mode="hybrid",
        args=tree_args,
    )
    hidden_mask = np.zeros(observed_shape, dtype=bool)
    frontier_local_mask = np.zeros(observed_shape, dtype=bool)
    segment_arrays = precompute_segment_prediction_arrays(
        sc_result["tree"],
        observed_state,
        prediction_layer,
        hidden_mask,
        frontier_local_mask,
        tree_args,
    )
    sc_candidates = path_candidate_rows(sc_result["tree"], segment_arrays, configs["map_predict_lambda48"])
    lambda48_decision = select_decision(
        sc_candidates,
        seed=int(args.tree_seed),
        mode="map_predict_lambda48",
        config=configs["map_predict_lambda48"],
        same_seed_measured=measured_decision,
        voxel_size=float(args.voxel_size),
    )
    lambda48_time = float(time.perf_counter() - primary_start)

    lambda32_decision: dict[str, Any] | None = None
    lambda32_time = 0.0
    if float(args.shadow_lambda_sc) > 0.0:
        config32 = dict(configs["map_predict_lambda32"])
        config32["lambda"] = float(args.shadow_lambda_sc)
        config32["formula"] = f"gain_exp / cost + {float(args.shadow_lambda_sc):g} * minmax(source_occ_free)"
        start32 = time.perf_counter()
        lambda32_decision = select_decision(
            path_candidate_rows(sc_result["tree"], segment_arrays, config32),
            seed=int(args.tree_seed),
            mode="map_predict_lambda32",
            config=config32,
            same_seed_measured=measured_decision,
            voxel_size=float(args.voxel_size),
        )
        lambda32_time = float(time.perf_counter() - start32)

    raw_root = output_dir / "raw_trees"
    for label, result in (("measured_only", measured_result), ("map_predict_raw_count", sc_result)):
        tree_dir = raw_root / label
        write_jsonl(tree_dir / "mini_rrt_tree_segments.jsonl", [segment_record(seg) for seg in result["tree"].values()])
        save_json(tree_dir / "mini_rrt_tree_summary.json", summarize_tree_result(result))
        write_csv(tree_dir / "gain_cost_value_table.csv", make_gain_value_rows(result["tree"]))
        if bool(args.save_viz):
            write_visualizations(tree_dir, result["tree"], result)

    save_json(
        output_dir / "measured_shadow_tree_decision.json",
        {
            "decision": measured_decision,
            "tree_profile": tree_profile(measured_result),
            "tree_dir": str(raw_root / "measured_only"),
            "timing_s": measured_time,
        },
    )
    write_decision_md(output_dir / "measured_shadow_tree_decision.md", "Measured-Only Shadow Tree Decision", measured_decision)

    save_json(
        output_dir / "lambda48_primary_tree_decision.json",
        {
            "decision": lambda48_decision,
            "tree_profile": tree_profile(sc_result),
            "tree_dir": str(raw_root / "map_predict_raw_count"),
            "timing_s": lambda48_time,
        },
    )
    write_decision_md(output_dir / "lambda48_primary_tree_decision.md", "Lambda48 Primary Tree Decision", lambda48_decision)

    if lambda32_decision is not None:
        save_json(
            output_dir / "lambda32_shadow_tree_decision.json",
            {
                "decision": lambda32_decision,
                "tree_profile": tree_profile(sc_result),
                "timing_s": lambda32_time,
            },
        )
        write_decision_md(output_dir / "lambda32_shadow_tree_decision.md", "Lambda32 Shadow Tree Decision", lambda32_decision)
    else:
        write_text(output_dir / "lambda32_shadow_skipped.md", "# Lambda32 Shadow Skipped\n\n- reason: `--shadow_lambda_sc <= 0`")

    return {
        "tree_args": vars(tree_args),
        "root_grid": root_grid,
        "root_world": root_world,
        "root_yaw": root_yaw,
        "measured_result": measured_result,
        "sc_result": sc_result,
        "measured_decision": measured_decision,
        "lambda48_decision": lambda48_decision,
        "lambda32_decision": lambda32_decision,
        "sc_candidates": sc_candidates,
        "timing": {
            "measured_shadow_tree_time_s": measured_time,
            "lambda48_primary_tree_time_s": lambda48_time,
            "lambda32_shadow_tree_time_s": lambda32_time,
        },
    }


def summarize_tree_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": result.get("profile", {}),
        "decision": result.get("decision", {}),
        "accepted_rows_count": len(result.get("accepted_rows", [])),
        "rejected_rows_count": len(result.get("rejected_rows", [])),
        "utility_warnings": result.get("utility_warnings", []),
    }


def tree_profile(result: dict[str, Any]) -> dict[str, Any]:
    profile = dict(result.get("profile", {}))
    profile["accepted_nodes_excluding_root"] = int(profile.get("accepted_nodes_excluding_root", 0))
    profile["rejected_samples"] = int(profile.get("rejected_samples", 0))
    profile["attempts"] = int(profile.get("attempts", 0))
    return profile


def write_decision_md(path: Path, title: str, decision: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- formula: `{decision.get('formula')}`",
        f"- selected child: `{decision.get('selected_child_id')}` grid `{decision.get('selected_child_grid')}` world `{decision.get('selected_child_world')}`",
        f"- best descendant: `{decision.get('best_descendant_id')}` grid `{decision.get('best_descendant_grid')}` world `{decision.get('best_descendant_world')}`",
        f"- gain_exp/source_occ_free/cost: `{decision.get('gain_exp')}` / `{decision.get('source_occ_free_count')}` / `{decision.get('cost')}`",
        f"- base_exp_value/normalized_sc/sc_bonus/final_value: `{decision.get('base_exp_value')}` / `{decision.get('normalized_sc')}` / `{decision.get('sc_bonus')}` / `{decision.get('final_value')}`",
        f"- branch classification: `{decision.get('branch_classification')}`",
        f"- low-cost artifact: `{decision.get('low_cost_artifact')}`",
    ]
    write_text(path, "\n".join(lines))


def compare_decisions(measured: dict[str, Any], lambda48: dict[str, Any], lambda32: dict[str, Any] | None) -> dict[str, Any]:
    lambda32_compare = None
    if lambda32 is not None:
        lambda32_compare = {
            "selected_child_id": lambda32.get("selected_child_id"),
            "best_descendant_id": lambda32.get("best_descendant_id"),
            "branch_classification": lambda32.get("branch_classification"),
            "same_selected_child_as_lambda48": lambda32.get("selected_child_id") == lambda48.get("selected_child_id"),
            "same_best_descendant_as_lambda48": lambda32.get("best_descendant_id") == lambda48.get("best_descendant_id"),
        }
    return {
        "measured_selected_child_id": measured.get("selected_child_id"),
        "measured_selected_child_grid": measured.get("selected_child_grid"),
        "measured_best_descendant_id": measured.get("best_descendant_id"),
        "measured_best_descendant_grid": measured.get("best_descendant_grid"),
        "lambda48_selected_child_id": lambda48.get("selected_child_id"),
        "lambda48_selected_child_grid": lambda48.get("selected_child_grid"),
        "lambda48_best_descendant_id": lambda48.get("best_descendant_id"),
        "lambda48_best_descendant_grid": lambda48.get("best_descendant_grid"),
        "selected_child_distance_from_same_seed_measured_m": lambda48.get(
            "selected_child_distance_from_same_seed_measured_m"
        ),
        "changed_vs_measured_only": bool(lambda48.get("changed_vs_measured_only")),
        "same_as_measured": bool(lambda48.get("same_as_measured")),
        "branch_classification": lambda48.get("branch_classification"),
        "healthy_nonmeasured_candidate": bool(lambda48.get("healthy_nonmeasured_candidate")),
        "low_cost_artifact": bool(lambda48.get("low_cost_artifact")),
        "spatial_prior_sc_basin": bool(lambda48.get("spatial_prior_sc_basin")),
        "lambda32_shadow": lambda32_compare,
    }


def low_cost_diagnosis(measured: dict[str, Any], lambda48: dict[str, Any]) -> dict[str, Any]:
    return {
        "definition": (
            "true if selected branch has lower gain_exp, lower source_occ_free, lower cost, "
            "and changed mainly by lower cost/formula amplification"
        ),
        "low_cost_artifact": bool(lambda48.get("low_cost_artifact")),
        "lambda48_gain_exp": lambda48.get("gain_exp"),
        "lambda48_source_occ_free": lambda48.get("source_occ_free_count"),
        "lambda48_cost": lambda48.get("cost"),
        "base_exp_selected_child_id": lambda48.get("base_exp_selected_child_id"),
        "base_exp_gain_exp": lambda48.get("base_exp_selected_gain_exp"),
        "base_exp_source_occ_free": lambda48.get("base_exp_selected_source_occ_free"),
        "base_exp_cost": lambda48.get("base_exp_selected_cost"),
        "measured_selected_child_id": measured.get("selected_child_id"),
        "measured_gain_exp": measured.get("gain_exp"),
        "measured_cost": measured.get("cost"),
    }


def source_protection_checklist(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "profile_name": PROFILE_NAME,
        "source_protected_profile": {
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
            "tree_seed": int(args.tree_seed),
        },
        "prediction_safety": {
            "prediction_information_gain_only": True,
            "prediction_writeback": False,
            "prediction_fused_into_observed_state": False,
            "prediction_used_for_traversability": False,
            "prediction_used_for_collision": False,
            "prediction_ray_blocking": False,
            "prediction_used_for_candidate_sampling": False,
            "prediction_used_for_edge_validity": False,
            "target_ground_truth_future_observed_planning_scoring": False,
        },
        "runtime_prohibited": {
            "selected_action_execution": False,
            "second_frame": False,
            "two_frame_runtime": False,
            "rollout": False,
            "open_ended_loop": False,
            "over_cost_primary": False,
        },
    }


def write_source_protection_md(path: Path, checklist: dict[str, Any]) -> None:
    profile = checklist["source_protected_profile"]
    safety = checklist["prediction_safety"]
    lines = [
        "# Source Protection Checklist",
        "",
        f"- profile: `{checklist['profile_name']}`",
        f"- short_edge_policy: `{profile['short_edge_policy']}`, crop_min_length_m `{profile['crop_min_length_m']}`",
        f"- num_nodes/max_extension/sample_mode: `{profile['num_nodes']}` / `{profile['max_extension_m']}` / `{profile['sample_mode']}`",
        f"- prediction information-gain only: `{safety['prediction_information_gain_only']}`",
        f"- prediction writeback/fusion: `{safety['prediction_writeback']}` / `{safety['prediction_fused_into_observed_state']}`",
        f"- prediction traversability/collision/ray blocking: `{safety['prediction_used_for_traversability']}` / `{safety['prediction_used_for_collision']}` / `{safety['prediction_ray_blocking']}`",
        f"- prediction candidate sampling/edge validity: `{safety['prediction_used_for_candidate_sampling']}` / `{safety['prediction_used_for_edge_validity']}`",
    ]
    write_text(path, "\n".join(lines))


def formula_definition(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "stage": "Stage 4A-6.5ai",
        "primary_formula_name": "map_predict_source_occ_free_decoupled_minmax_lambda48",
        "primary_formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
        "lambda_sc": float(args.lambda_sc),
        "shadow_lambda_sc": float(args.shadow_lambda_sc),
        "shadow_formula": "gain_exp / cost + 32 * minmax(source_occ_free)",
        "tau": float(args.tau),
        "occ_threshold": float(args.occ_threshold),
        "free_threshold": float(args.free_threshold),
        "source_occ_free": {
            "counts": "predicted OCCUPIED plus predicted FREE",
            "validity": "prediction-valid, confidence-thresholded, unmeasured voxels only",
            "excludes": ["predicted unknown", "invalid prediction", "already measured voxels"],
        },
        "minmax": {
            "scope": "per tree over valid root-to-descendant path accumulated source_occ_free",
            "formula": "(sc - min_sc) / (max_sc - min_sc), or 0 when max_sc == min_sc",
        },
        "prohibited_runtime_primary_formulas": [
            "(gain_exp + 48 * source_occ_free) / cost",
            "(gain_exp + source_occ_free) / cost",
        ],
        "over_cost_runtime_primary_executed": False,
    }


def write_formula_md(path: Path, definition: dict[str, Any]) -> None:
    lines = [
        "# Formula Definition",
        "",
        f"- primary: `{definition['primary_formula']}`",
        f"- lambda: `{definition['lambda_sc']}`",
        f"- tau: `{definition['tau']}`",
        f"- occ/free thresholds: `{definition['occ_threshold']}` / `{definition['free_threshold']}`",
        "- SC bonus is outside the cost denominator.",
        "- Over-cost is diagnostic-only history and was not executed as runtime primary.",
    ]
    write_text(path, "\n".join(lines))


def topdown_observed(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def grid_xy(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    return float(grid[0]) + 0.5, float(grid[1]) + 0.5


def plot_observed(path: Path, observed_state: np.ndarray) -> None:
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 6.8), constrained_layout=True)
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Measured-only observed_state after one frame")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_prediction_overlay(path: Path, observed_state: np.ndarray, prediction_npz: str, args: argparse.Namespace) -> None:
    base = topdown_observed(observed_state)
    with np.load(prediction_npz, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occupied_prob = np.asarray(data["global_occupied_prob"], dtype=np.float32)
        free_prob = np.asarray(data["global_free_prob"], dtype=np.float32)
    valid_tau = valid & (confidence >= float(args.tau)) & (observed_state == UNKNOWN)
    occ = np.any(valid_tau & (occupied_prob >= float(args.occ_threshold)), axis=2)
    free = np.any(valid_tau & (free_prob >= float(args.free_threshold)), axis=2)
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 6.8), constrained_layout=True)
    ax.imshow(base.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.contour(free.T, levels=[0.5], colors=["#38bdf8"], linewidths=0.8)
    ax.contour(occ.T, levels=[0.5], colors=["#facc15"], linewidths=0.8)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Prediction overlay on unmeasured voxels")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_tree_comparison(
    path: Path,
    observed_state: np.ndarray,
    measured: dict[str, Any],
    lambda48: dict[str, Any],
) -> None:
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    root = grid_xy(lambda48.get("root_grid") or measured.get("root_grid"))
    measured_selected = grid_xy(measured.get("selected_child_grid"))
    measured_best = grid_xy(measured.get("best_descendant_grid"))
    lambda_selected = grid_xy(lambda48.get("selected_child_grid"))
    lambda_best = grid_xy(lambda48.get("best_descendant_grid"))
    if root:
        ax.scatter([root[0]], [root[1]], c="#ffffff", edgecolors="#111111", s=90, label="root")
    for point, color, label, marker, size in (
        (measured_selected, "#f97316", "measured selected", "o", 70),
        (measured_best, "#fb923c", "measured best", "*", 130),
        (lambda_selected, "#3b82f6", "lambda48 selected", "o", 80),
        (lambda_best, "#2563eb", "lambda48 best", "*", 150),
    ):
        if point:
            ax.scatter([point[0]], [point[1]], c=color, marker=marker, s=size, label=label)
            if root:
                ax.plot([root[0], point[0]], [root[1], point[1]], color=color, linewidth=1.8, alpha=0.85)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Measured-only shadow vs lambda48 primary")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_lambda48_branch(path: Path, observed_state: np.ndarray, tree: dict[str, Any], decision: dict[str, Any]) -> None:
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    branch_ids = segment_path_to_root(tree, decision.get("best_descendant_id"))
    for a, b in zip(branch_ids, branch_ids[1:]):
        parent = tree[a]
        child = tree[b]
        ax.plot(
            [parent.end_grid[0] + 0.5, child.end_grid[0] + 0.5],
            [parent.end_grid[1] + 0.5, child.end_grid[1] + 0.5],
            color="#2563eb",
            linewidth=2.4,
        )
    for segment_id in branch_ids:
        segment = tree[segment_id]
        ax.scatter(segment.end_grid[0] + 0.5, segment.end_grid[1] + 0.5, s=42, c="#2563eb")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Lambda48 selected best-descendant branch")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_value_components(path: Path, decision: dict[str, Any]) -> None:
    labels = ["gain_exp/cost", "lambda48 minmax SC", "final value"]
    values = [
        float(decision.get("base_exp_value") or 0.0),
        float(decision.get("sc_bonus") or 0.0),
        float(decision.get("final_value") or 0.0),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.bar(labels, values, color=["#0f766e", "#2563eb", "#7c3aed"])
    ax.set_ylabel("value")
    ax.set_title("Lambda48 selected branch value components")
    ax.tick_params(axis="x", rotation=15)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_low_cost(path: Path, diagnosis: dict[str, Any]) -> None:
    labels = ["lambda48 cost", "base-exp cost", "lambda48 gain", "base-exp gain", "lambda48 SC", "base-exp SC"]
    values = [
        float(diagnosis.get("lambda48_cost") or 0.0),
        float(diagnosis.get("base_exp_cost") or 0.0),
        float(diagnosis.get("lambda48_gain_exp") or 0.0),
        float(diagnosis.get("base_exp_gain_exp") or 0.0),
        float(diagnosis.get("lambda48_source_occ_free") or 0.0),
        float(diagnosis.get("base_exp_source_occ_free") or 0.0),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    colors = ["#0891b2", "#67e8f9", "#16a34a", "#86efac", "#7c3aed", "#c4b5fd"]
    ax.bar(labels, values, color=colors)
    ax.set_title(f"Low-cost artifact: {diagnosis.get('low_cost_artifact')}")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def generate_plots(
    output_dir: Path,
    observed_state: np.ndarray,
    prediction_npz: str,
    tree_result: dict[str, Any],
    measured_decision: dict[str, Any],
    lambda48_decision: dict[str, Any],
    low_cost: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    generated: dict[str, str] = {}
    skipped: dict[str, str] = {}
    plotters = {
        "observed_runtime_topdown.png": lambda p: plot_observed(p, observed_state),
        "prediction_overlay_topdown.png": lambda p: plot_prediction_overlay(p, observed_state, prediction_npz, args),
        "measured_vs_lambda48_tree_topdown.png": lambda p: plot_tree_comparison(
            p, observed_state, measured_decision, lambda48_decision
        ),
        "lambda48_selected_branch_topdown.png": lambda p: plot_lambda48_branch(
            p, observed_state, tree_result["tree"], lambda48_decision
        ),
        "tree_value_components_lambda48.png": lambda p: plot_value_components(p, lambda48_decision),
        "low_cost_artifact_runtime.png": lambda p: plot_low_cost(p, low_cost),
    }
    for name, plotter in plotters.items():
        path = output_dir / name
        try:
            plotter(path)
            generated[name] = str(path)
        except Exception as exc:  # pragma: no cover - reported as artifact
            reason_path = output_dir / f"{Path(name).stem}_skipped_reason.md"
            write_text(reason_path, f"# Plot Skipped\n\n- plot: `{name}`\n- reason: `{exc}`")
            skipped[name] = str(reason_path)
    return {"generated": generated, "skipped": skipped}


def hardware_report(args: argparse.Namespace, timings: dict[str, Any]) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    actual_workers = min(int(args.max_workers), int(cpu_count))
    report = {
        "os_cpu_count": int(cpu_count),
        "requested_max_workers": int(args.max_workers),
        "actual_max_workers": int(actual_workers),
        "parallel_backend": "single_process_runtime_stage_no_process_pool",
        "task_count": 1,
        "worker_process_thread_mode": "single Isaac/runtime process; no CPU process pool needed for one-frame single-seed smoke",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "torch_num_threads": int(torch.get_num_threads()),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "timing": timings,
    }
    return report


def write_hardware_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Hardware Utilization Report",
        "",
        f"- os_cpu_count: `{report['os_cpu_count']}`",
        f"- requested_max_workers: `{report['requested_max_workers']}`",
        f"- actual_max_workers: `{report['actual_max_workers']}`",
        f"- parallel_backend: `{report['parallel_backend']}`",
        f"- OMP/OPENBLAS/MKL/NUMEXPR/VECLIB: `{report['OMP_NUM_THREADS']}` / `{report['OPENBLAS_NUM_THREADS']}` / `{report['MKL_NUM_THREADS']}` / `{report['NUMEXPR_NUM_THREADS']}` / `{report['VECLIB_MAXIMUM_THREADS']}`",
        f"- torch_num_threads: `{report['torch_num_threads']}`",
        f"- CUDA device: `{report['cuda_device_name']}`",
        f"- total wall time: `{report['timing'].get('total_wall_time_s')}` s",
    ]
    write_text(path, "\n".join(lines))


def write_generic_md(path: Path, title: str, rows: list[str]) -> None:
    write_text(path, "\n".join([f"# {title}", "", *rows]))


def make_recommendation(lambda48: dict[str, Any], map_predict_success: bool, capture_success: bool) -> tuple[str, str]:
    if not capture_success:
        return "runtime Isaac environment/debug only", "Isaac startup or capture failed."
    if not map_predict_success:
        return "runtime map_predict failure diagnosis", "map_predict failed in the one-frame runtime path."
    if bool(lambda48.get("low_cost_artifact")) or bool(lambda48.get("spatial_prior_sc_basin")):
        return "offline artifact diagnosis before any more runtime", "lambda48 touched a prior basin or low-cost artifact risk."
    if bool(lambda48.get("healthy_nonmeasured_candidate")):
        return (
            "Stage 4A-6.5aj staged two-frame one-action lambda48 runtime smoke design review only",
            "the one-frame runtime was safe and lambda48 selected a healthy distinct non-measured branch.",
        )
    if bool(lambda48.get("same_as_measured")):
        return (
            "Stage 4A-6.5aj one-action/two-frame lambda48 runtime design review only, or controlled capture-only additional saved-frame collection",
            "the one-frame runtime was safe and lambda48 behaved conservatively.",
        )
    return "controlled capture-only additional saved-frame collection", "the one-frame outcome was safe but weak/ambiguous."


def scan_forbidden_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in (
        "transitions.jsonl",
        "rollout_topdown_path.png",
        "observed_ratio_curve.png",
        "rollout_index.html",
        "episode_manifest*",
        "frame002*",
        "capture_*_001.*",
        "step_*.npz",
    ):
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not bool(args.no_action_execution):
        raise ValueError("--no_action_execution is required for Stage 4A-6.5ai")
    if not bool(args.no_second_frame):
        raise ValueError("--no_second_frame is required for Stage 4A-6.5ai")
    if not bool(args.no_rollout):
        raise ValueError("--no_rollout is required for Stage 4A-6.5ai")

    total_start = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, Any] = {"isaac_app_launch_time_s": ISAAC_APP_LAUNCH_TIME_S}

    context = context_manifest(output_dir)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_hash_before = sha256_file(checkpoint_path)
    external_before = git_status_short(WORKSPACE / "external_src/active_3d_planning_inspection/mav_active_3d_planning")
    workspace_before = git_status_short(WORKSPACE)

    scene_start = time.perf_counter()
    sim, camera, scene_metadata = start_scene(args)
    pose = make_pose(args)
    capture = capture_one_frame(args, output_dir, camera, sim, pose)
    timings["isaac_scene_start_and_capture_time_s"] = float(time.perf_counter() - scene_start)

    capture_summary = {
        "isaac_startup_count": 1,
        "frames_captured": 1,
        "rgb_frames_captured": 1,
        "depth_frames_captured": 1,
        "second_frame_captured": False,
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "start_pose": pose,
        "scene_metadata": scene_metadata,
        **{k: v for k, v in capture.items() if k != "depth"},
    }
    save_json(output_dir / "runtime_capture_summary.json", capture_summary)
    write_generic_md(
        output_dir / "runtime_capture_summary.md",
        "Runtime Capture Summary",
        [
            "- Isaac startup count: `1`",
            "- RGB/depth frames captured: `1 / 1`",
            "- second frame captured: `False`",
            f"- scene: `{args.scene_variant}` seed `{args.scene_seed}`",
            f"- start pose: `{pose['position']}`, yaw `{pose['yaw_rad']}`",
        ],
    )

    observed_start = time.perf_counter()
    observed = update_observed_state(args, output_dir, capture)
    timings["observed_state_update_time_s"] = float(time.perf_counter() - observed_start)
    save_json(output_dir / "observed_state_update_summary.json", observed)
    write_generic_md(
        output_dir / "observed_state_update_summary.md",
        "Observed State Update Summary",
        [
            "- measured-only update count: `1`",
            f"- observed_state: `{observed['new_observed_state']}`",
            f"- shape: `{observed['new_observed_state_shape']}`",
            f"- observed ratio: `{observed['updated_summary']['observed_ratio']}`",
            "- prediction used in update: `False`",
        ],
    )

    predict_start = time.perf_counter()
    prediction_result = run_prediction(args, output_dir, capture, observed)
    timings["map_predict_total_time_s"] = float(time.perf_counter() - predict_start)
    observed_state = np.load(observed["new_observed_state"])
    pred_stats = prediction_stats(str(prediction_result["prediction_npz"]), observed_state, args)
    prediction_hash_after_creation = sha256_file(prediction_result["prediction_npz"])
    map_summary = {
        "map_predict_call_count": 1,
        "checkpoint": str(checkpoint_path),
        "checkpoint_unchanged_according_to_predictor": bool(prediction_result["checkpoint_unchanged"]),
        "alignment_convention": str(args.alignment_convention),
        "tau": float(args.tau),
        "prediction_layer_shape": pred_stats["shape"],
        "observed_state_shape": pred_stats["observed_state_shape"],
        "prediction_layer_shape_aligned_to_observed_state": pred_stats["shape_aligned_to_observed_state"],
        "prediction_npz": prediction_result["prediction_npz"],
        "summary_json": prediction_result["summary_json"],
        "timing": prediction_result.get("timing", {}),
        "stats": pred_stats,
    }
    save_json(output_dir / "map_predict_runtime_summary.json", map_summary)
    write_generic_md(
        output_dir / "map_predict_runtime_summary.md",
        "Map Predict Runtime Summary",
        [
            "- map_predict calls: `1`",
            f"- alignment convention: `{args.alignment_convention}`",
            f"- prediction shape aligned: `{pred_stats['shape_aligned_to_observed_state']}`",
            f"- predicted unmeasured occ+free count: `{pred_stats['predicted_unmeasured_occ_free_count']}`",
        ],
    )

    formula = formula_definition(args)
    save_json(output_dir / "formula_definition.json", formula)
    write_formula_md(output_dir / "formula_definition.md", formula)

    checklist = source_protection_checklist(args)
    save_json(output_dir / "source_protection_checklist.json", checklist)
    write_source_protection_md(output_dir / "source_protection_checklist.md", checklist)

    tree_start = time.perf_counter()
    tree = build_and_score_trees(
        args,
        output_dir,
        observed_state,
        prediction_result["prediction_layer"],
        capture["pose"],
    )
    timings["tree_decision_total_time_s"] = float(time.perf_counter() - tree_start)
    timings.update(tree["timing"])

    comparison = compare_decisions(tree["measured_decision"], tree["lambda48_decision"], tree["lambda32_decision"])
    save_json(output_dir / "tree_decision_comparison.json", comparison)
    write_generic_md(
        output_dir / "tree_decision_comparison.md",
        "Tree Decision Comparison",
        [
            f"- changed vs measured-only: `{comparison['changed_vs_measured_only']}`",
            f"- branch classification: `{comparison['branch_classification']}`",
            f"- selected child distance from measured: `{comparison['selected_child_distance_from_same_seed_measured_m']}` m",
            f"- lambda32 same selected child as lambda48: `{(comparison['lambda32_shadow'] or {}).get('same_selected_child_as_lambda48')}`",
        ],
    )
    with (output_dir / "tree_decision_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mode", "selected_child_id", "best_descendant_id", "branch_classification", "final_value"])
        writer.writeheader()
        for mode, decision in (
            ("measured_only", tree["measured_decision"]),
            ("map_predict_lambda48", tree["lambda48_decision"]),
            ("map_predict_lambda32", tree["lambda32_decision"] or {}),
        ):
            writer.writerow(
                {
                    "mode": mode,
                    "selected_child_id": decision.get("selected_child_id"),
                    "best_descendant_id": decision.get("best_descendant_id"),
                    "branch_classification": decision.get("branch_classification"),
                    "final_value": decision.get("final_value"),
                }
            )

    branch = {
        "classification": tree["lambda48_decision"].get("branch_classification"),
        "same_as_measured": bool(tree["lambda48_decision"].get("same_as_measured")),
        "distinct_nonmeasured_branch": tree["lambda48_decision"].get("branch_classification")
        == "distinct_nonmeasured_branch",
        "local_jitter": tree["lambda48_decision"].get("branch_classification") == "local_jitter",
        "spatial_prior_sc_basin": bool(tree["lambda48_decision"].get("spatial_prior_sc_basin")),
        "healthy_nonmeasured_candidate": bool(tree["lambda48_decision"].get("healthy_nonmeasured_candidate")),
        "selected_child_distance_from_same_seed_measured_m": tree["lambda48_decision"].get(
            "selected_child_distance_from_same_seed_measured_m"
        ),
        "historical_prior_selected_grid": HISTORICAL_PRIOR_SELECTED_GRID,
        "historical_prior_best_grid": HISTORICAL_PRIOR_BEST_GRID,
        "selected_child_distance_from_historical_prior_m": grid_distance_m(
            tree["lambda48_decision"].get("selected_child_grid"),
            HISTORICAL_PRIOR_SELECTED_GRID,
            float(args.voxel_size),
        ),
        "best_descendant_distance_from_historical_prior_m": grid_distance_m(
            tree["lambda48_decision"].get("best_descendant_grid"),
            HISTORICAL_PRIOR_BEST_GRID,
            float(args.voxel_size),
        ),
    }
    save_json(output_dir / "branch_classification.json", branch)
    write_generic_md(
        output_dir / "branch_classification.md",
        "Branch Classification",
        [
            f"- lambda48 classification: `{branch['classification']}`",
            f"- healthy non-measured: `{branch['healthy_nonmeasured_candidate']}`",
            f"- historical prior basin: `{branch['spatial_prior_sc_basin']}`",
        ],
    )

    low_cost = low_cost_diagnosis(tree["measured_decision"], tree["lambda48_decision"])
    save_json(output_dir / "low_cost_artifact_diagnosis.json", low_cost)
    write_generic_md(
        output_dir / "low_cost_artifact_diagnosis.md",
        "Low-Cost Artifact Diagnosis",
        [
            f"- low_cost_artifact: `{low_cost['low_cost_artifact']}`",
            f"- lambda48 cost / base-exp cost: `{low_cost['lambda48_cost']}` / `{low_cost['base_exp_cost']}`",
            f"- lambda48 source_occ_free / base-exp source_occ_free: `{low_cost['lambda48_source_occ_free']}` / `{low_cost['base_exp_source_occ_free']}`",
        ],
    )

    plot_report = generate_plots(
        output_dir,
        observed_state,
        prediction_result["prediction_npz"],
        tree["sc_result"],
        tree["measured_decision"],
        tree["lambda48_decision"],
        low_cost,
        args,
    )

    prediction_hash_after_tree = sha256_file(prediction_result["prediction_npz"])
    observed_hash_after_prediction_and_tree = sha256_file(observed["new_observed_state"])
    checkpoint_hash_after = sha256_file(checkpoint_path)
    hash_checks = {
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": checkpoint_hash_before,
            "sha256_after": checkpoint_hash_after,
            "unchanged": checkpoint_hash_before == checkpoint_hash_after,
        },
        "new_observed_state": {
            "path": observed["new_observed_state"],
            "sha256_after_update": observed["new_observed_state_sha256"],
            "sha256_after_prediction_and_tree": observed_hash_after_prediction_and_tree,
            "unchanged_after_update": observed["new_observed_state_sha256"] == observed_hash_after_prediction_and_tree,
        },
        "prediction_npz": {
            "path": prediction_result["prediction_npz"],
            "sha256_after_creation": prediction_hash_after_creation,
            "sha256_after_tree": prediction_hash_after_tree,
            "unchanged_after_creation": prediction_hash_after_creation == prediction_hash_after_tree,
        },
        "existing_observed_state_inputs": [],
        "existing_prediction_npz_inputs": [],
    }
    save_json(output_dir / "hash_checks.json", hash_checks)

    prediction_safety = {
        "prediction_read_only": True,
        "prediction_written_to_observed_state": False,
        "prediction_fused_into_observed_state": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_ray_blocking": False,
        "prediction_used_for_candidate_sampling": False,
        "prediction_used_for_edge_validity": False,
        "target_lr_target_hr_ground_truth_used_for_planning_scoring": False,
        "future_observed_used_for_planning_scoring": False,
        "observed_state_hash_unchanged_after_prediction_and_tree": hash_checks["new_observed_state"][
            "unchanged_after_update"
        ],
    }
    save_json(output_dir / "prediction_safety_report.json", prediction_safety)
    write_generic_md(
        output_dir / "prediction_safety_report.md",
        "Prediction Safety Report",
        [
            "- prediction read-only: `True`",
            "- prediction writeback/fusion: `False / False`",
            "- prediction traversability/collision/ray blocking: `False / False / False`",
            "- prediction candidate sampling/edge validity: `False / False`",
        ],
    )

    no_action = {
        "selected_action_execution": False,
        "selected_action_execution_count": 0,
        "camera_pose_after_decision_equals_captured_frame_pose": True,
        "captured_frame_pose": capture["pose"],
        "second_frame_captured": False,
        "rollout": False,
    }
    save_json(output_dir / "no_action_execution_report.json", no_action)
    write_generic_md(
        output_dir / "no_action_execution_report.md",
        "No Action Execution Report",
        [
            "- selected action execution count: `0`",
            "- camera moved to selected child: `False`",
            "- second frame captured: `False`",
            "- rollout: `False`",
        ],
    )

    prohibited = scan_forbidden_outputs(output_dir)
    required_files = [
        "loaded_context_manifest.json",
        "loaded_context_manifest.md",
        "hardware_utilization_report.json",
        "hardware_utilization_report.md",
        "runtime_capture_summary.json",
        "runtime_capture_summary.md",
        "observed_state_update_summary.json",
        "observed_state_update_summary.md",
        "map_predict_runtime_summary.json",
        "map_predict_runtime_summary.md",
        "formula_definition.json",
        "formula_definition.md",
        "source_protection_checklist.json",
        "source_protection_checklist.md",
        "measured_shadow_tree_decision.json",
        "measured_shadow_tree_decision.md",
        "lambda48_primary_tree_decision.json",
        "lambda48_primary_tree_decision.md",
        "tree_decision_comparison.json",
        "tree_decision_comparison.md",
        "branch_classification.json",
        "branch_classification.md",
        "low_cost_artifact_diagnosis.json",
        "low_cost_artifact_diagnosis.md",
        "prediction_safety_report.json",
        "prediction_safety_report.md",
        "hash_checks.json",
        "missing_fields_report.json",
        "missing_fields_report.md",
        "no_action_execution_report.json",
        "no_action_execution_report.md",
        "stage4a65ai_one_frame_lambda48_runtime_summary.json",
        "stage4a65ai_one_frame_lambda48_runtime_summary.md",
        "recommended_next_faithful_step.md",
        "capture_rgb_000.png",
        "capture_depth_000.npy",
        "capture_depth_000.png",
        "observed_state_frame000.npy",
        "map_predict/global_prediction_layer.npz",
        "map_predict/prediction_alignment_summary.json",
        "observed_runtime_topdown.png",
        "prediction_overlay_topdown.png",
        "measured_vs_lambda48_tree_topdown.png",
        "lambda48_selected_branch_topdown.png",
        "tree_value_components_lambda48.png",
        "low_cost_artifact_runtime.png",
    ]
    if tree["lambda32_decision"] is not None:
        required_files.append("lambda32_shadow_tree_decision.json")
    else:
        required_files.append("lambda32_shadow_skipped.md")

    missing_files = [name for name in required_files if not (output_dir / name).is_file()]
    missing_report = {
        "missing_required_files": missing_files,
        "plot_skipped_reasons": plot_report["skipped"],
        "prohibited_artifacts_found": prohibited,
    }
    save_json(output_dir / "missing_fields_report.json", missing_report)
    write_generic_md(
        output_dir / "missing_fields_report.md",
        "Missing Fields Report",
        [
            f"- missing required files: `{missing_files}`",
            f"- skipped plots: `{plot_report['skipped']}`",
            f"- prohibited artifacts found: `{prohibited}`",
        ],
    )

    recommendation, recommendation_reason = make_recommendation(tree["lambda48_decision"], True, True)
    readiness = {
        "runtime_executed": True,
        "one_frame_runtime_smoke_complete": True,
        "two_frame_runtime_ready": False,
        "two_frame_runtime_ready_reason": "not marked ready by this one-frame no-action smoke; next is design review only",
        "rollout_ready": False,
        "rollout_ready_reason": "one-frame no-action smoke cannot establish rollout readiness",
    }
    total_wall_time = float(time.perf_counter() - total_start)
    timings["total_wall_time_s"] = total_wall_time
    report = hardware_report(args, timings)
    save_json(output_dir / "hardware_utilization_report.json", report)
    write_hardware_md(output_dir / "hardware_utilization_report.md", report)

    external_after = git_status_short(WORKSPACE / "external_src/active_3d_planning_inspection/mav_active_3d_planning")
    workspace_after = git_status_short(WORKSPACE)
    summary = {
        "stage": "Stage 4A-6.5ai staged one-frame lambda48 runtime smoke",
        "output_dir": str(output_dir),
        "context_loaded": context,
        "scene": {"scene_variant": str(args.scene_variant), "scene_seed": int(args.scene_seed)},
        "runtime_setup": {
            "isaac_startup_count": 1,
            "frames_captured": 1,
            "map_predict_calls": 1,
            "selected_action_execution_count": 0,
            "second_frame": False,
            "two_frame_runtime": False,
            "rollout": False,
        },
        "formula": formula,
        "map_predict": map_summary,
        "results": {
            "measured_only_shadow": tree["measured_decision"],
            "lambda48_primary": tree["lambda48_decision"],
            "lambda32_shadow": tree["lambda32_decision"],
            "branch_classification": branch,
            "low_cost_artifact": low_cost,
            "prediction_stats": pred_stats,
        },
        "readiness": readiness,
        "recommendation": {
            "next_small_task": recommendation,
            "why": recommendation_reason,
            "do_not_recommend_rollout_directly": True,
        },
        "safety": {
            "isaac_startup": True,
            "frames_captured": 1,
            "map_predict_calls": 1,
            "selected_action_execution": False,
            "selected_action_execution_count": 0,
            "second_frame": False,
            "two_frame_runtime": False,
            "rollout": False,
            "open_ended_loop": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_modified": checkpoint_hash_before != checkpoint_hash_after,
            "existing_observed_state_modified": False,
            "prediction_npz_modified_after_creation": prediction_hash_after_creation != prediction_hash_after_tree,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "prediction_ray_blocking": False,
            "prediction_used_for_candidate_sampling_edge_validity": False,
            "target_ground_truth_future_observed_planning_scoring": False,
            "external_source_modified_built": external_before != external_after,
            "over_cost_runtime_primary": False,
            "coverage_improvement_claim": False,
            "workspace_git_status_before": workspace_before,
            "workspace_git_status_after": workspace_after,
            "external_source_git_status_before": external_before,
            "external_source_git_status_after": external_after,
            "prohibited_artifacts_found": prohibited,
        },
        "hardware": report,
        "missing_fields_report": missing_report,
    }
    save_json(output_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.json", summary)
    write_summary_md(output_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.md", summary)
    write_recommendation(output_dir / "recommended_next_faithful_step.md", recommendation, recommendation_reason)

    final_missing_files = [name for name in required_files if not (output_dir / name).is_file()]
    missing_report = {
        "missing_required_files": final_missing_files,
        "plot_skipped_reasons": plot_report["skipped"],
        "prohibited_artifacts_found": scan_forbidden_outputs(output_dir),
    }
    save_json(output_dir / "missing_fields_report.json", missing_report)
    write_generic_md(
        output_dir / "missing_fields_report.md",
        "Missing Fields Report",
        [
            f"- missing required files: `{final_missing_files}`",
            f"- skipped plots: `{plot_report['skipped']}`",
            f"- prohibited artifacts found: `{missing_report['prohibited_artifacts_found']}`",
        ],
    )
    summary["missing_fields_report"] = missing_report
    summary["safety"]["prohibited_artifacts_found"] = missing_report["prohibited_artifacts_found"]
    save_json(output_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.json", summary)
    write_summary_md(output_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.md", summary)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    setup = summary["runtime_setup"]
    results = summary["results"]
    safety = summary["safety"]
    readiness = summary["readiness"]
    lines = [
        "# Stage 4A-6.5ai One-Frame Lambda48 Runtime Summary",
        "",
        f"1. Successfully read Stage 4A-6.5ag/ah context? `{summary['context_loaded']['confirmed_stage4a65ag_complete'] and summary['context_loaded']['confirmed_stage4a65ah_complete']}`.",
        f"2. Isaac started exactly once? `{setup['isaac_startup_count'] == 1}`.",
        f"3. Captured exactly one frame? `{setup['frames_captured'] == 1}`.",
        f"4. No second frame? `{not setup['second_frame']}`.",
        f"5. No selected action execution? `{setup['selected_action_execution_count'] == 0}`.",
        f"6. No rollout? `{not setup['rollout']}`.",
        f"7. map_predict exactly once? `{setup['map_predict_calls'] == 1}`.",
        f"8. map_predict used code_consistent_v1? `{summary['map_predict']['alignment_convention'] == 'code_consistent_v1'}`.",
        f"9. prediction layer shape aligned? `{summary['map_predict']['prediction_layer_shape_aligned_to_observed_state']}`.",
        "10. prediction read-only? `True`.",
        "11. prediction not written to observed_state? `True`.",
        "12. prediction not used for traversability/collision/ray blocking/edge validity? `True`.",
        f"13. lambda48 formula exact? `{summary['formula']['primary_formula']}`.",
        f"14. measured-only shadow selected: `{results['measured_only_shadow'].get('selected_child_id')}` -> `{results['measured_only_shadow'].get('best_descendant_id')}`.",
        f"15. lambda48 primary selected: `{results['lambda48_primary'].get('selected_child_id')}` -> `{results['lambda48_primary'].get('best_descendant_id')}`.",
        f"16. lambda32 shadow selected: `{(results['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{(results['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"17. lambda48 branch classification: `{results['branch_classification']['classification']}`.",
        f"18. low-cost artifact? `{results['low_cost_artifact']['low_cost_artifact']}`.",
        f"19. historical prior bad branch hit? `{results['branch_classification']['spatial_prior_sc_basin']}`.",
        f"20. consistent with 6.5ag saved-frame evidence? `safety-clean one-frame runtime; behavior comparison is recorded, but this does not replace 6.5ag aggregate evidence`.",
        f"21. enough for rollout? `{readiness['rollout_ready']}`.",
        f"22. enough for two-frame runtime? `{readiness['two_frame_runtime_ready']}`.",
        f"23. next recommended step: `{summary['recommendation']['next_small_task']}`.",
        "",
        "## Safety",
        f"- checkpoint modified: `{safety['checkpoint_modified']}`",
        f"- existing observed_state modified: `{safety['existing_observed_state_modified']}`",
        f"- prediction writeback: `{safety['prediction_writeback']}`",
        f"- coverage improvement claimed: `{safety['coverage_improvement_claim']}`",
    ]
    write_text(path, "\n".join(lines))


def write_recommendation(path: Path, next_step: str, reason: str) -> None:
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {next_step}",
        f"- why: {reason}",
        "- still not next: rollout, online open-ended loop, RL/PPO/BC/IL, prediction writeback/fusion, prediction traversability/collision/ray blocking, target/ground-truth/future-observed scoring, checkpoint changes, external source build, Pareto gate implementation, runtime planner implementation, or over-cost runtime promotion.",
        "- future offline analysis commands should continue to include `--max_workers 32`.",
    ]
    write_text(path, "\n".join(lines))


if __name__ == "__main__":
    try:
        run(args_cli)
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
