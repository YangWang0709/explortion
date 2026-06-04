#!/usr/bin/env python3
"""Stage 4A-6 short multi-step SC-aware rollout with read-only map_predict."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)
DEFAULT_OUTPUT_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke"
DEFAULT_EPISODE_ID = "medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000"

parser = argparse.ArgumentParser(description="Stage 4A-6 short SC-aware simulator expert rollout.")
parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
parser.add_argument("--episode_id", default=DEFAULT_EPISODE_ID)
parser.add_argument("--scene_variant", choices=("minimal", "medium_three_rooms"), default="medium_three_rooms")
parser.add_argument("--scene_seed", type=int, default=0)
parser.add_argument("--start_variant", default="start_room_a")
parser.add_argument("--max_steps", type=int, default=5)
parser.add_argument("--coverage_stop", type=float, default=0.75)
parser.add_argument("--min_score", type=float, default=1e-6)
parser.add_argument("--repeat_pose_patience", type=int, default=3)
parser.add_argument("--num_candidates", type=int, default=64)
parser.add_argument("--top_n", type=int, default=16)
parser.add_argument("--gain_mode", choices=("exp", "sc", "hybrid", "occ", "conf"), default="hybrid")
parser.add_argument("--prediction_mode", choices=("sim_dynamic", "sim_static_npz"), default="sim_dynamic")
parser.add_argument("--static_prediction_npz", default=None)
parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
parser.add_argument("--tau", type=float, default=0.1)
parser.add_argument(
    "--alignment_convention",
    choices=("current_default_v0", "code_consistent_v1", "xz_swap_diagnostic"),
    default="code_consistent_v1",
)
parser.add_argument("--sc_gain_formula", choices=("raw_count", "occupied_only", "occupied_margin", "confidence_weighted", "entropy_weighted", "calibrated_occupied", "novelty_discounted"), default="raw_count")
parser.add_argument("--sc_occ_threshold", type=float, default=0.7)
parser.add_argument("--sc_conf_threshold", type=float, default=0.3)
parser.add_argument("--sc_count_mode", choices=("raw_count", "selective"), default="raw_count")
parser.add_argument("--calibration_table", default=None)
parser.add_argument("--sc_gain_weight", type=float, default=1.0)
parser.add_argument("--sc_gain_cap", type=float, default=None)
parser.add_argument("--score_gain_mode", choices=("hybrid_raw", "hybrid_weighted"), default="hybrid_raw")
parser.add_argument("--static_prediction_step", type=int, default=None)
parser.add_argument("--path_cost_mode", choices=("astar",), default="astar")
parser.add_argument("--candidate_sampling_mode", choices=("reachable_frontier",), default="reachable_frontier")
parser.add_argument("--snap_start_to_traversable", action="store_true")
parser.add_argument("--max_snap_radius_cells", type=int, default=5)
parser.add_argument("--motion_mode", choices=("planar",), default="planar")
parser.add_argument("--camera_height", type=float, default=1.2)
parser.add_argument("--voxel_size", type=float, default=0.1)
parser.add_argument("--pixel_stride", type=int, default=2)
parser.add_argument("--camera_width", type=int, default=160)
parser.add_argument("--camera_sensor_height", type=int, default=120)
parser.add_argument("--max_depth", type=float, default=5.0)
parser.add_argument("--settle_steps", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max_range_voxels", type=int, default=50)
parser.add_argument("--num_yaw", type=int, default=32)
parser.add_argument("--num_pitch", type=int, default=7)
parser.add_argument("--fov_yaw_deg", type=float, default=90.0)
parser.add_argument("--fov_pitch_deg", type=float, default=60.0)
parser.add_argument("--start_x", type=float, default=None)
parser.add_argument("--start_y", type=float, default=None)
parser.add_argument("--start_yaw_deg", type=float, default=None)
parser.add_argument("--obstacle_jitter_m", type=float, default=0.0)
parser.add_argument("--save_rgb", action="store_true")
parser.add_argument("--save_depth", action="store_true")
parser.add_argument("--save_prediction", action="store_true")
parser.add_argument("--save_viz", action="store_true")
parser.add_argument("--save_probs", action="store_true")
parser.add_argument("--print_steps", action="store_true")
parser.add_argument("--profile", action="store_true")
parser.add_argument("--torch_num_threads", type=int, default=8)
parser.add_argument("--sscnet_device", default="cuda")
parser.add_argument("--no_manifest", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import hashlib
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

from depth_to_voxel import DEFAULT_MAP_BOUNDS, update_observed_state_from_depth
from isaac_map_predictor import IsaacMapPredictor, sha256_array
from run_sim_expert_rollout_batch import default_start_pose
from scene_factory import build_medium_complex_scene, build_minimal_scene
from sim_paper_expert import select_sim_expert_action
from sim_prediction_layer import SimPredictionLayer
from sim_rollout_utils import (
    ROLLOUT_LIMITATION_NOTES,
    append_jsonl,
    build_transition,
    compute_next_pose_from_candidate,
    initialize_observed_state,
    leakage_checks,
    load_jsonl,
    observed_summary,
    pose_dict,
    pose_repeat_key,
    pose_target,
    save_json,
    transition_score_stats,
    write_transition_records,
)

DEPTH_KEY = "distance_to_image_plane"


STAGE4A6_LIMITATIONS = [
    "Stage 4A-6 is a short single-episode smoke rollout.",
    "Prediction is recomputed per step and used only for information gain.",
    "Prediction is read-only and never written into observed_state.",
    "Observed_state updates come only from Isaac depth ray marching.",
    "A* traversability, collision bookkeeping, candidate reachability, and ray blocking use observed_state only.",
    "UNKNOWN is not traversable; no Euclidean fallback is used.",
    "No RL, PPO, behavior cloning training, imitation learning training, optimizer step, or SSCNet training is run.",
] + ROLLOUT_LIMITATION_NOTES


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_stat(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).resolve()
    stat = checkpoint.stat()
    return {
        "path": str(checkpoint),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(checkpoint),
    }


def _build_scene_metadata(spawn: bool) -> dict[str, Any]:
    if str(args_cli.scene_variant) == "minimal":
        return build_minimal_scene(spawn=spawn, sim_utils_module=sim_utils if spawn else None)
    if str(args_cli.scene_variant) == "medium_three_rooms":
        return build_medium_complex_scene(
            seed=int(args_cli.scene_seed),
            variant="three_rooms",
            obstacle_jitter_m=float(args_cli.obstacle_jitter_m),
            spawn=spawn,
            sim_utils_module=sim_utils if spawn else None,
        )
    raise ValueError(f"Unsupported scene_variant: {args_cli.scene_variant}")


def design_scene() -> tuple[Camera, dict[str, Any]]:
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    scene_metadata = _build_scene_metadata(spawn=True)

    data_types = [DEPTH_KEY]
    if args_cli.save_rgb:
        data_types.insert(0, "rgb")

    sim_utils.create_prim("/World/CameraRig", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/CameraRig/CameraSensor",
        update_period=0.0,
        height=int(args_cli.camera_sensor_height),
        width=int(args_cli.camera_width),
        data_types=data_types,
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=36.0,
            clipping_range=(0.05, float(args_cli.max_depth)),
        ),
    )
    return Camera(cfg=camera_cfg), scene_metadata


def _set_camera_pose(camera: Camera, sim: sim_utils.SimulationContext, pose: dict[str, Any]) -> None:
    position = [float(v) for v in pose["position"]]
    target = pose_target(position, float(pose["yaw_rad"]))
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )


def _step_camera(camera: Camera, sim: sim_utils.SimulationContext, settle_steps: int) -> None:
    for _ in range(max(int(settle_steps), 1)):
        sim.step()
        camera.update(dt=sim.get_physics_dt())


def _camera_info_from_camera(camera: Camera) -> dict[str, Any]:
    intrinsic_matrix = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
    return {
        "sensor_api_depth_key": DEPTH_KEY,
        "depth_units": "meters",
        "width": int(args_cli.camera_width),
        "height": int(args_cli.camera_sensor_height),
        "max_depth": float(args_cli.max_depth),
        "near_depth": 0.05,
        "horizontal_fov_deg": 90.0,
        "intrinsic_matrix": intrinsic_matrix.tolist(),
        "fx": float(intrinsic_matrix[0, 0]),
        "fy": float(intrinsic_matrix[1, 1]),
        "cx": float(intrinsic_matrix[0, 2]),
        "cy": float(intrinsic_matrix[1, 2]),
    }


def _extract_depth(camera: Camera) -> np.ndarray:
    depth_tensor = camera.data.output.get(DEPTH_KEY)
    if depth_tensor is None:
        raise KeyError(f"Camera output missing {DEPTH_KEY}. Keys: {list(camera.data.output.keys())}")
    depth = depth_tensor[0].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    return depth


def _normalize_rgb(image: np.ndarray) -> np.ndarray:
    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _save_rgb_if_requested(camera: Camera, path: Path) -> str | None:
    if not args_cli.save_rgb:
        return None
    tensor = camera.data.output.get("rgb")
    if tensor is None:
        tensor = camera.data.output.get("rgba")
    if tensor is None:
        raise KeyError(f"Camera output missing rgb/rgba. Keys: {list(camera.data.output.keys())}")
    image = tensor[0].detach().cpu().numpy()
    Image.fromarray(_normalize_rgb(image)).save(path)
    return str(path)


def _depth_stats(depth: np.ndarray) -> dict[str, Any]:
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    return {
        "shape": [int(v) for v in depth.shape],
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
    }


def _resolve_rollout_bounds(scene_metadata: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw_bounds = scene_metadata.get("map_bounds") or DEFAULT_MAP_BOUNDS
    return {axis: tuple(float(v) for v in raw_bounds[axis]) for axis in ("x", "y", "z")}


def _resolve_start_pose() -> tuple[dict[str, Any], dict[str, Any]]:
    if args_cli.start_x is not None or args_cli.start_y is not None or args_cli.start_yaw_deg is not None:
        if args_cli.start_x is None or args_cli.start_y is None:
            raise ValueError("--start_x and --start_y must be provided together for explicit starts")
        position = [float(args_cli.start_x), float(args_cli.start_y), float(args_cli.camera_height)]
        yaw_deg = float(args_cli.start_yaw_deg if args_cli.start_yaw_deg is not None else 0.0)
        note = "explicit CLI start pose"
    else:
        pose = default_start_pose(str(args_cli.scene_variant), str(args_cli.start_variant), float(args_cli.camera_height))
        position = [float(v) for v in pose["position"]]
        yaw_deg = float(pose["yaw_deg"])
        note = str(pose.get("note", "scripted start pose"))

    current_pose = pose_dict(position, math.radians(yaw_deg))
    start_pose = {
        "variant": str(args_cli.start_variant),
        "world": pose_dict(current_pose["position"], float(current_pose["yaw_rad"])),
        "source": note,
        "ground_truth_used_for_scoring": False,
    }
    return current_pose, start_pose


def _transition_metric_summary(transitions: list[dict[str, Any]]) -> dict[str, Any]:
    if not transitions:
        return {}

    def stats_optional(key: str) -> dict[str, float | None]:
        values = np.asarray(
            [float(t[key]) for t in transitions if key in t and t[key] is not None and np.isfinite(float(t[key]))],
            dtype=np.float64,
        )
        if values.size == 0:
            return {"min": None, "mean": None, "max": None}
        return {"min": float(values.min()), "mean": float(values.mean()), "max": float(values.max())}

    summary = {
        "average_frontier_count": float(np.mean([float(t["frontier_count"]) for t in transitions])),
        "average_candidate_count": float(np.mean([float(t["candidate_count"]) for t in transitions])),
        "best_score": transition_score_stats(transitions, "best_score"),
        "gain_exp": transition_score_stats(transitions, "best_gain_exp"),
        "gain_sc": transition_score_stats(transitions, "best_gain_sc"),
        "gain_hybrid": transition_score_stats(transitions, "best_gain_hybrid"),
        "gain_occ": transition_score_stats(transitions, "best_gain_occ"),
        "gain_conf": transition_score_stats(transitions, "best_gain_conf"),
        "weighted_gain_sc": transition_score_stats(transitions, "best_weighted_gain_sc"),
        "effective_gain_sc": transition_score_stats(transitions, "best_effective_gain_sc"),
        "gain_hybrid_weighted": transition_score_stats(transitions, "best_gain_hybrid_weighted"),
        "gain_hybrid_effective": transition_score_stats(transitions, "best_gain_hybrid_effective"),
        "utility_hybrid_weighted": transition_score_stats(transitions, "best_utility_hybrid_weighted"),
        "utility_hybrid_effective": transition_score_stats(transitions, "best_utility_hybrid_effective"),
        "path_cost": transition_score_stats(transitions, "best_path_cost"),
        "map_predict_preprocess_time": stats_optional("map_predict_preprocess_time"),
        "map_predict_inference_time": stats_optional("map_predict_inference_time"),
        "map_predict_alignment_time": stats_optional("map_predict_alignment_time"),
        "map_predict_total_time": stats_optional("map_predict_total_time"),
        "expert_time": stats_optional("expert_time"),
        "step_total_time": stats_optional("step_total_time"),
        "candidates_with_gain_sc_positive": stats_optional("candidates_with_gain_sc_positive"),
        "candidates_with_effective_gain_sc_positive": stats_optional("candidates_with_effective_gain_sc_positive"),
        "predicted_unmeasured_voxels": stats_optional("predicted_unmeasured_voxels"),
        "prediction_valid_voxels": stats_optional("prediction_valid_voxels"),
        "no_valid_candidate_steps": [
            int(t["step"])
            for t in transitions
            if str(t.get("no_valid_candidate_reason", "") or "") or str(t.get("done_reason", "")) == "no_valid_candidate"
        ],
    }
    if any("reachable_candidates" in t for t in transitions):
        summary["average_reachable_candidates"] = float(
            np.mean([float(t.get("reachable_candidates", 0.0)) for t in transitions])
        )
    if any("reachable_component_count" in t for t in transitions):
        summary["reachable_component_count"] = stats_optional("reachable_component_count")
    return summary


def _build_failure_record(
    episode_dir: Path,
    done_reason: str,
    failure_stage: str,
    exc: BaseException,
    step_records: list[dict[str, Any]],
    observed_state: np.ndarray | None,
    bounds: dict[str, tuple[float, float]],
    checkpoint_before: dict[str, Any],
) -> dict[str, Any]:
    final_summary = observed_summary(observed_state) if observed_state is not None else {}
    summary = {
        "stage": "Stage 4A-6 short multi-step SC-aware rollout",
        "episode_id": str(args_cli.episode_id),
        "episode_dir": str(episode_dir),
        "status": "failed",
        "done_reason": str(done_reason),
        "failure_stage": str(failure_stage),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "scene_variant": str(args_cli.scene_variant),
        "scene_seed": int(args_cli.scene_seed),
        "start_variant": str(args_cli.start_variant),
        "prediction_mode": str(args_cli.prediction_mode),
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "tau": float(args_cli.tau),
        "alignment_convention": str(args_cli.alignment_convention),
        "sc_gain_formula": str(args_cli.sc_gain_formula),
        "sc_occ_threshold": float(args_cli.sc_occ_threshold),
        "sc_conf_threshold": float(args_cli.sc_conf_threshold),
        "sc_count_mode": str(args_cli.sc_count_mode),
        "calibration_table": str(Path(args_cli.calibration_table).resolve()) if args_cli.calibration_table else None,
        "sc_gain_weight": float(args_cli.sc_gain_weight),
        "sc_gain_cap": None if args_cli.sc_gain_cap is None else float(args_cli.sc_gain_cap),
        "score_gain_mode": str(args_cli.score_gain_mode),
        "checkpoint_stat_before": checkpoint_before,
        "checkpoint_stat_after": checkpoint_stat(args_cli.checkpoint) if Path(args_cli.checkpoint).exists() else None,
        "max_steps": int(args_cli.max_steps),
        "steps_completed": int(len(load_jsonl(episode_dir / "transitions.jsonl"))),
        "map_bounds": {axis: [float(v) for v in bounds[axis]] for axis in ("x", "y", "z")},
        "final_counts": final_summary,
        "step_records": step_records,
        "strict_no_prediction_write": True,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_a_star": False,
        "prediction_blocks_rays": False,
        "rl_optimizer_training_run": False,
        "limitations": STAGE4A6_LIMITATIONS,
    }
    save_json(episode_dir / "episode_summary.json", summary)
    return summary


def _static_prediction_step(
    static_layer: SimPredictionLayer,
    step_prediction_dir: Path,
    observed_state: np.ndarray,
    step: int,
) -> dict[str, Any]:
    step_prediction_dir.mkdir(parents=True, exist_ok=True)
    layer_path = step_prediction_dir / "global_prediction_layer.npz"
    np.savez_compressed(
        layer_path,
        global_pred_class=static_layer.pred_class,
        global_confidence=static_layer.confidence,
        global_free_prob=static_layer.free_prob,
        global_occupied_prob=static_layer.occupied_prob,
        global_prediction_valid=static_layer.valid,
        observed_state_source=f"step_{int(step):03d}_observed_state",
        local_prediction_source=str(static_layer.source_npz or ""),
        checkpoint=str(args_cli.checkpoint),
        strict_no_observed_write=np.array(True, dtype=bool),
        read_only_note="static prediction layer copied read-only for ablation",
    )
    tau = float(args_cli.tau)
    predicted = static_layer.valid & (static_layer.confidence >= tau)
    predicted_occupied = predicted & (static_layer.occupied_prob >= 0.5)
    predicted_unmeasured = predicted & (observed_state == -1)
    summary = {
        "stage": "Stage 4A-6 static prediction ablation step",
        "step": int(step),
        "global_valid_prediction_count": int(np.count_nonzero(static_layer.valid)),
        "global_predicted_occupied_count": int(np.count_nonzero(predicted_occupied)),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "timing": {"preprocess_time": 0.0, "inference_time": 0.0, "alignment_time": 0.0, "total_time": 0.0},
        "paths": {"global_prediction_layer": str(layer_path)},
        "strict_no_observed_write": True,
    }
    save_json(step_prediction_dir / "prediction_alignment_summary.json", summary)
    return {
        "prediction_layer": static_layer,
        "prediction_npz": str(layer_path),
        "global_prediction_npz": str(layer_path),
        "local_prediction_npz": "",
        "summary": summary,
        "summary_json": str(step_prediction_dir / "prediction_alignment_summary.json"),
        "timing": summary["timing"],
        "visualizations": {},
    }


def run_rollout() -> dict[str, Any]:
    output_root = Path(args_cli.output_dir).resolve()
    episode_dir = output_root / "episodes" / args_cli.episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    transitions_jsonl = episode_dir / "transitions.jsonl"
    if transitions_jsonl.exists():
        transitions_jsonl.unlink()

    checkpoint_before = checkpoint_stat(args_cli.checkpoint)
    planned_scene_metadata = _build_scene_metadata(spawn=False)
    bounds = _resolve_rollout_bounds(planned_scene_metadata)
    observed_state: np.ndarray | None = initialize_observed_state(bounds, voxel_size=float(args_cli.voxel_size))
    step_records: list[dict[str, Any]] = []
    depth_summaries: list[dict[str, Any]] = []
    prediction_summaries: list[dict[str, Any]] = []
    failure_stage = "setup"

    try:
        predictor: IsaacMapPredictor | None = None
        static_prediction_layer: SimPredictionLayer | None = None
        if str(args_cli.prediction_mode) == "sim_dynamic":
            failure_stage = "map_predict_model_load"
            predictor = IsaacMapPredictor(
                checkpoint=args_cli.checkpoint,
                device=str(args_cli.sscnet_device),
                tau=float(args_cli.tau),
                torch_num_threads=int(args_cli.torch_num_threads),
                alignment_convention=str(args_cli.alignment_convention),
            )
        else:
            if not args_cli.static_prediction_npz:
                raise ValueError("--static_prediction_npz is required when --prediction_mode sim_static_npz")
            static_prediction_layer = SimPredictionLayer.from_npz(args_cli.static_prediction_npz)

        failure_stage = "isaac_scene_setup"
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([5.0, -5.0, 4.0], [0.0, 0.0, 0.5])
        camera, spawned_scene_metadata = design_scene()
        sim.reset()
        print(f"[INFO]: Stage 4A-6 scene setup complete: {args_cli.scene_variant}")

        current_pose, start_pose = _resolve_start_pose()
        camera_info_saved = False
        camera_info: dict[str, Any] | None = None
        pose_visits: dict[tuple[int, int, int, int], int] = {}
        assert observed_state is not None
        start_key = pose_repeat_key(current_pose, bounds, args_cli.voxel_size, tuple(observed_state.shape))
        pose_visits[start_key] = 1
        repeated_pose_count = 1
        done_reason = "max_steps"
        total_wall_start = time.perf_counter()

        scene_metadata = {
            **spawned_scene_metadata,
            "stage": "Stage 4A-6",
            "scene": spawned_scene_metadata.get("scene", str(args_cli.scene_variant)),
            "scene_variant": str(args_cli.scene_variant),
            "scene_seed": int(args_cli.scene_seed),
            "obstacle_jitter_m": float(args_cli.obstacle_jitter_m),
            "start_variant": str(args_cli.start_variant),
            "start_pose": start_pose,
            "camera_height_m": float(args_cli.camera_height),
            "map_bounds": {axis: [float(v) for v in bounds[axis]] for axis in ("x", "y", "z")},
            "voxel_size": float(args_cli.voxel_size),
        "prediction_mode": str(args_cli.prediction_mode),
        "checkpoint": str(Path(args_cli.checkpoint).resolve()),
        "tau": float(args_cli.tau),
        "alignment_convention": str(args_cli.alignment_convention),
        "sc_gain_formula": str(args_cli.sc_gain_formula),
        "sc_occ_threshold": float(args_cli.sc_occ_threshold),
        "sc_conf_threshold": float(args_cli.sc_conf_threshold),
        "sc_count_mode": str(args_cli.sc_count_mode),
        "calibration_table": str(Path(args_cli.calibration_table).resolve()) if args_cli.calibration_table else None,
        "sc_gain_weight": float(args_cli.sc_gain_weight),
        "sc_gain_cap": None if args_cli.sc_gain_cap is None else float(args_cli.sc_gain_cap),
        "score_gain_mode": str(args_cli.score_gain_mode),
        "static_prediction_step": None if args_cli.static_prediction_step is None else int(args_cli.static_prediction_step),
        "path_cost_mode": str(args_cli.path_cost_mode),
            "candidate_sampling_mode": str(args_cli.candidate_sampling_mode),
            "snap_start_to_traversable": bool(args_cli.snap_start_to_traversable),
            "max_snap_radius_cells": int(args_cli.max_snap_radius_cells),
            "motion_mode": str(args_cli.motion_mode),
            "hardware_context": {
                "cpu": "AMD Ryzen 9 9950X3D",
                "cpu_threads": 32,
                "ram_gb": 32,
                "gpu": "NVIDIA RTX 5080",
                "sscnet_device": str(args_cli.sscnet_device),
                "torch_num_threads": int(args_cli.torch_num_threads),
            },
            "limitations": STAGE4A6_LIMITATIONS,
            "leakage_checks": leakage_checks(
                False,
                a_star_planner=True,
                prediction_mode=str(args_cli.prediction_mode),
                prediction_layer="SimPredictionLayer",
            ),
        }
        save_json(episode_dir / "scene_metadata.json", scene_metadata)

        for step in range(int(args_cli.max_steps)):
            step_start = time.perf_counter()
            summary_before = observed_summary(observed_state)

            failure_stage = "isaac_render_camera_capture"
            _set_camera_pose(camera, sim, current_pose)
            _step_camera(camera, sim, int(args_cli.settle_steps))

            if not camera_info_saved:
                camera_info = _camera_info_from_camera(camera)
                save_json(episode_dir / "camera_info.json", camera_info)
                camera_info_saved = True
            assert camera_info is not None

            depth = _extract_depth(camera)
            depth_stats = _depth_stats(depth)
            depth_path = episode_dir / f"depth_{step:03d}.npy"
            if args_cli.save_depth:
                np.save(depth_path, depth)
                depth_stats["depth_file"] = str(depth_path)
            rgb_path = _save_rgb_if_requested(camera, episode_dir / f"rgb_{step:03d}.png")
            if rgb_path is not None:
                depth_stats["rgb_file"] = rgb_path
            depth_summaries.append({"step": int(step), **depth_stats})

            pose_record = {
                "index": int(step),
                "position": [float(v) for v in current_pose["position"]],
                "yaw_rad": float(current_pose["yaw_rad"]),
                "yaw_deg": float(current_pose["yaw_deg"]),
                "target": pose_target(current_pose["position"], current_pose["yaw_rad"]),
                "motion_mode": str(args_cli.motion_mode),
            }
            pose_path = episode_dir / f"pose_{step:03d}.json"
            save_json(pose_path, pose_record)

            failure_stage = "observed_map_update"
            observed_state = update_observed_state_from_depth(
                observed_state=observed_state,
                depth=depth,
                camera_pose=current_pose,
                camera_info=camera_info,
                bounds=bounds,
                voxel_size=float(args_cli.voxel_size),
                pixel_stride=int(args_cli.pixel_stride),
            )
            observed_state_path = episode_dir / f"observed_state_step{step:03d}.npy"
            np.save(observed_state_path, observed_state)
            summary_after = observed_summary(observed_state)

            observed_hash_before_prediction = sha256_array(observed_state)
            prediction_dir = episode_dir / f"prediction_step{step:03d}"
            if str(args_cli.prediction_mode) == "sim_dynamic":
                assert predictor is not None
                failure_stage = "map_predict_preprocessing_inference_alignment"
                prediction_result = predictor.predict_step(
                    depth=depth,
                    pose=pose_record,
                    camera_info=camera_info,
                    observed_state=observed_state,
                    map_bounds=bounds,
                    voxel_size=float(args_cli.voxel_size),
                    output_dir=prediction_dir,
                    step=int(step),
                    save_probs=bool(args_cli.save_probs),
                    save_viz=bool(args_cli.save_viz),
                    observed_state_path=observed_state_path,
                    depth_source=depth_path if args_cli.save_depth else None,
                    pose_source=pose_path,
                    camera_info_source=episode_dir / "camera_info.json",
                )
            else:
                assert static_prediction_layer is not None
                failure_stage = "static_prediction_layer_loading"
                prediction_result = _static_prediction_step(
                    static_layer=static_prediction_layer,
                    step_prediction_dir=prediction_dir,
                    observed_state=observed_state,
                    step=int(step),
                )

            prediction_layer = prediction_result["prediction_layer"]
            if tuple(prediction_layer.shape()) != tuple(observed_state.shape):
                raise ValueError(
                    f"prediction layer shape {prediction_layer.shape()} differs from observed_state {observed_state.shape}"
                )
            observed_hash_after_prediction = sha256_array(observed_state)
            observed_state_prediction_modified = observed_hash_before_prediction != observed_hash_after_prediction
            if observed_state_prediction_modified:
                raise RuntimeError("map_predict modified observed_state, violating Stage 4A-6 read-only boundary")

            failure_stage = "expert_scoring"
            before_expert = observed_state.copy()
            expert_start = time.perf_counter()
            try:
                result = select_sim_expert_action(
                    observed_state=observed_state,
                    current_pose_world=current_pose,
                    bounds=bounds,
                    voxel_size=float(args_cli.voxel_size),
                    prediction_layer=prediction_layer,
                    prediction_mode="sim_npz",
                    num_candidates=int(args_cli.num_candidates),
                    top_n=int(args_cli.top_n),
                    gain_mode=str(args_cli.gain_mode),
                    sc_gain_formula=str(args_cli.sc_gain_formula),
                    sc_occ_threshold=float(args_cli.sc_occ_threshold),
                    sc_conf_threshold=float(args_cli.sc_conf_threshold),
                    sc_count_mode=str(args_cli.sc_count_mode),
                    score_gain_mode=str(args_cli.score_gain_mode),
                    sc_gain_weight=float(args_cli.sc_gain_weight),
                    sc_gain_cap=args_cli.sc_gain_cap,
                    calibration_table=str(args_cli.calibration_table) if args_cli.calibration_table else None,
                    seed=int(args_cli.seed) + int(step),
                    tau=float(args_cli.tau),
                    max_range_voxels=int(args_cli.max_range_voxels),
                    num_yaw=int(args_cli.num_yaw),
                    num_pitch=int(args_cli.num_pitch),
                    fov_yaw_deg=float(args_cli.fov_yaw_deg),
                    fov_pitch_deg=float(args_cli.fov_pitch_deg),
                    path_cost_mode=str(args_cli.path_cost_mode),
                    robot_height_m=float(args_cli.camera_height),
                    candidate_sampling_mode=str(args_cli.candidate_sampling_mode),
                    snap_start_to_traversable=bool(args_cli.snap_start_to_traversable),
                    max_snap_radius_cells=int(args_cli.max_snap_radius_cells),
                )
            except ValueError as exc:
                done_reason = "no_reachable_free_component" if "no_reachable_free_component" in str(exc) else "no_valid_candidate"
                step_records.append(
                    {
                        "step": int(step),
                        "done_reason": done_reason,
                        "error": str(exc),
                        "failure_stage": failure_stage,
                        "no_valid_candidate_reason": done_reason,
                        "observed_ratio_before": float(summary_before["observed_ratio"]),
                        "observed_ratio_after": float(summary_after["observed_ratio"]),
                    }
                )
                print(f"[WARN]: stopping at step {step}: {exc}")
                break
            expert_time = time.perf_counter() - expert_start

            result["diagnostics"]["rollout_prediction_mode"] = str(args_cli.prediction_mode)
            result["diagnostics"]["prediction_used_for_astar"] = False
            prediction_wrote_observed_map = not bool(np.array_equal(before_expert, observed_state))
            if prediction_wrote_observed_map:
                raise RuntimeError("expert or prediction modified observed_state")

            best = result["best_candidate"]
            next_pose = compute_next_pose_from_candidate(
                candidate=best,
                bounds=bounds,
                voxel_size=float(args_cli.voxel_size),
                observed_shape=tuple(int(v) for v in observed_state.shape),
                motion_mode=str(args_cli.motion_mode),
                camera_height=float(args_cli.camera_height),
            )

            done = False
            done_reason = ""
            next_key = pose_repeat_key(next_pose["world"], bounds, args_cli.voxel_size, tuple(observed_state.shape))
            pose_visits[next_key] = pose_visits.get(next_key, 0) + 1
            repeated_pose_count = max(repeated_pose_count, pose_visits[next_key])
            if float(summary_after["observed_ratio"]) >= float(args_cli.coverage_stop):
                done = True
                done_reason = "coverage_stop"
            elif float(best.final_score) <= float(args_cli.min_score):
                done = True
                done_reason = "best_score_below_min_score"
            elif pose_visits[next_key] >= int(args_cli.repeat_pose_patience):
                done = True
                done_reason = "repeat_pose_patience"
            elif step == int(args_cli.max_steps) - 1:
                done = True
                done_reason = "max_steps"

            transition = build_transition(
                episode_id=str(args_cli.episode_id),
                step=int(step),
                result=result,
                current_pose_world=current_pose,
                next_pose=next_pose,
                summary_before=summary_before,
                summary_after=summary_after,
                bounds=bounds,
                voxel_size=float(args_cli.voxel_size),
                prediction_mode=str(args_cli.prediction_mode),
                gain_mode=str(args_cli.gain_mode),
                path_cost_mode=str(args_cli.path_cost_mode),
                motion_mode=str(args_cli.motion_mode),
                done=done,
                done_reason=done_reason,
                prediction_wrote_observed_map=prediction_wrote_observed_map,
            )
            transition.update(
                {
                    "reachable_candidates": int(result["diagnostics"].get("reachable_candidates") or 0),
                    "unreachable_candidates": int(result["diagnostics"].get("unreachable_candidates") or 0),
                    "prediction_npz": str(prediction_result["prediction_npz"]),
                    "local_prediction_npz": str(prediction_result.get("local_prediction_npz", "")),
                    "prediction_summary_json": str(prediction_result["summary_json"]),
                    "map_predict_preprocess_time": float(prediction_result["timing"].get("preprocess_time", 0.0)),
                    "map_predict_inference_time": float(prediction_result["timing"].get("inference_time", 0.0)),
                    "map_predict_alignment_time": float(prediction_result["timing"].get("alignment_time", 0.0)),
                    "map_predict_total_time": float(prediction_result["timing"].get("total_time", 0.0)),
                    "expert_time": float(expert_time),
                    "step_total_time": float(time.perf_counter() - step_start),
                    "observed_state_hash_before_prediction": observed_hash_before_prediction,
                    "observed_state_hash_after_prediction": observed_hash_after_prediction,
                    "observed_state_prediction_modified": bool(observed_state_prediction_modified),
                    "strict_no_prediction_write": True,
                    "prediction_used_for_traversability": False,
                    "prediction_used_for_collision": False,
                    "prediction_used_for_a_star": False,
                    "prediction_blocks_rays": False,
                    "prediction_writeback": False,
                    "prediction_mode_expert_internal": "sim_npz",
                    "gpu_memory_peak": prediction_result["summary"].get("gpu_memory_peak"),
                    "checkpoint_unchanged": bool(prediction_result["summary"].get("checkpoint_unchanged", True)),
                }
            )
            paths = write_transition_records(episode_dir, transition)
            prediction_summaries.append(
                {
                    "step": int(step),
                    "prediction_dir": str(prediction_dir),
                    "prediction_npz": str(prediction_result["prediction_npz"]),
                    "summary_json": str(prediction_result["summary_json"]),
                    "global_valid_prediction_count": int(
                        prediction_result["summary"].get("global_valid_prediction_count", 0)
                    ),
                    "predicted_unmeasured_count": int(prediction_result["summary"].get("predicted_unmeasured_count", 0)),
                    "gpu_memory_peak": prediction_result["summary"].get("gpu_memory_peak"),
                    "timing": prediction_result["timing"],
                }
            )
            step_record = {
                "step": int(step),
                "observed_state_file": str(observed_state_path),
                "transition_npz": paths["npz"],
                "prediction_npz": str(prediction_result["prediction_npz"]),
                "observed_ratio_before": float(summary_before["observed_ratio"]),
                "observed_ratio_after": float(summary_after["observed_ratio"]),
                "delta_observed_ratio": float(transition["delta_observed_ratio"]),
                "frontier_count": int(transition["frontier_count"]),
                "candidate_count": int(transition["candidate_count"]),
                "best_score": float(transition["best_score"]),
                "best_gain_exp": float(transition["best_gain_exp"]),
                "best_gain_sc": float(transition["best_gain_sc"]),
                    "best_gain_hybrid": float(transition["best_gain_hybrid"]),
                    "best_raw_gain_sc": float(transition.get("best_raw_gain_sc", transition["best_gain_sc"])),
                    "best_effective_gain_sc": float(transition.get("best_effective_gain_sc", transition["best_gain_sc"])),
                    "best_gain_hybrid_effective": float(
                        transition.get("best_gain_hybrid_effective", transition["best_gain_hybrid"])
                    ),
                    "best_weighted_gain_sc": float(transition.get("best_weighted_gain_sc", 0.0)),
                    "best_gain_hybrid_weighted": float(transition.get("best_gain_hybrid_weighted", 0.0)),
                    "best_utility_hybrid_weighted": float(transition.get("best_utility_hybrid_weighted", 0.0)),
                    "best_gain_occ": float(transition["best_gain_occ"]),
                "best_gain_conf": float(transition["best_gain_conf"]),
                "best_path_cost": float(transition["best_path_cost"]),
                "best_astar_path_length_m": float(transition["best_astar_path_length_m"]),
                "best_astar_num_expanded": int(transition["best_astar_num_expanded"]),
                "candidates_with_gain_sc_positive": int(transition["candidates_with_gain_sc_positive"]),
                    "candidates_with_effective_gain_sc_positive": int(
                        transition.get("candidates_with_effective_gain_sc_positive", 0)
                    ),
                    "mean_gain_sc": float(transition["mean_gain_sc"]),
                    "max_gain_sc": float(transition["max_gain_sc"]),
                    "mean_effective_gain_sc": float(transition.get("mean_effective_gain_sc", 0.0)),
                    "max_effective_gain_sc": float(transition.get("max_effective_gain_sc", 0.0)),
                    "mean_weighted_gain_sc": float(transition.get("mean_weighted_gain_sc", 0.0)),
                    "max_weighted_gain_sc": float(transition.get("max_weighted_gain_sc", 0.0)),
                "predicted_unmeasured_voxels": int(transition["predicted_unmeasured_voxels"]),
                "prediction_valid_voxels": int(transition["prediction_valid_voxels"]),
                "reachable_candidates": int(transition["reachable_candidates"]),
                "unreachable_candidates": int(transition["unreachable_candidates"]),
                "reachable_component_count": int(transition.get("reachable_component_count") or 0),
                "reachable_frontier_adjacent_count": int(transition.get("reachable_frontier_adjacent_count") or 0),
                "candidate_source": str(transition.get("candidate_source", "")),
                "current_pose_world": [float(v) for v in transition["current_pose_world"]],
                "selected_next_pose_world": [float(v) for v in transition["selected_next_pose_world"]],
                "map_predict_preprocess_time": float(transition["map_predict_preprocess_time"]),
                "map_predict_inference_time": float(transition["map_predict_inference_time"]),
                "map_predict_alignment_time": float(transition["map_predict_alignment_time"]),
                "map_predict_total_time": float(transition["map_predict_total_time"]),
                "expert_time": float(transition["expert_time"]),
                "step_total_time": float(transition["step_total_time"]),
                "done": bool(done),
                "done_reason": str(done_reason),
            }
            step_records.append(step_record)

            if args_cli.print_steps:
                print(
                    "[STEP "
                    f"{step:03d}] ratio {summary_before['observed_ratio']:.6f} -> {summary_after['observed_ratio']:.6f} "
                    f"score={transition['best_score']:.6f} gain_exp={transition['best_gain_exp']:.1f} "
                    f"gain_sc={transition['best_gain_sc']:.1f} gain_hybrid={transition['best_gain_hybrid']:.1f} "
                    f"effective_sc={transition.get('best_effective_gain_sc', transition['best_gain_sc']):.3f} "
                    f"weighted_sc={transition.get('best_weighted_gain_sc', 0.0):.1f} "
                    f"path_cost={transition['best_path_cost']:.3f} "
                    f"sc_pos={transition['candidates_with_gain_sc_positive']}/{transition['candidate_count']} "
                    f"pred_t={transition['map_predict_total_time']:.3f}s expert_t={transition['expert_time']:.3f}s "
                    f"next={step_record['selected_next_pose_world']} done={done} reason={done_reason or 'continue'}"
                )

            current_pose = next_pose["world"]
            if done:
                break

        np.save(episode_dir / "observed_state_final.npy", observed_state)
        final_summary = observed_summary(observed_state)
        transitions = load_jsonl(transitions_jsonl)
        steps_completed = len(transitions)
        terminal_done_reason = str(done_reason or "")
        if transitions:
            final_pose = transitions[-1]["selected_next_pose_world"]
            final_yaw = transitions[-1]["selected_next_yaw"]
            transition_done_reason = str(transitions[-1].get("done_reason", ""))
            if transition_done_reason:
                terminal_done_reason = transition_done_reason
            start_ratio = float(transitions[0]["observed_ratio_before"])
            end_ratio = float(final_summary["observed_ratio"])
        else:
            final_pose = current_pose["position"]
            final_yaw = current_pose["yaw_rad"]
            start_ratio = 0.0
            end_ratio = float(final_summary["observed_ratio"])

        output_paths = {
            "episode_dir": str(episode_dir),
            "transitions": str(transitions_jsonl),
            "summary": str(episode_dir / "episode_summary.json"),
            "final_observed_map": str(episode_dir / "observed_state_final.npy"),
        }

        if args_cli.save_viz and transitions:
            from visualize_sim_rollout import save_rollout_visualizations

            viz_paths = save_rollout_visualizations(episode_dir, save_steps=True)
            output_paths.update({k: v for k, v in viz_paths.items() if isinstance(v, str)})

        metric_summary = _transition_metric_summary(transitions)
        checkpoint_after = checkpoint_stat(args_cli.checkpoint)
        gpu_memory_peaks = [
            int(t["gpu_memory_peak"])
            for t in transitions
            if t.get("gpu_memory_peak") is not None and str(t.get("gpu_memory_peak")) != "None"
        ]
        prediction_total_times = [float(t.get("map_predict_total_time", 0.0)) for t in transitions]
        episode_summary = {
            "stage": "Stage 4A-6 short multi-step SC-aware rollout",
            "episode_id": str(args_cli.episode_id),
            "episode_dir": str(episode_dir),
            "scene": spawned_scene_metadata.get("scene", str(args_cli.scene_variant)),
            "scene_variant": str(args_cli.scene_variant),
            "scene_seed": int(args_cli.scene_seed),
            "obstacle_jitter_m": float(args_cli.obstacle_jitter_m),
            "start_variant": str(args_cli.start_variant),
            "start_pose": start_pose,
            "prediction_mode": str(args_cli.prediction_mode),
            "prediction_layer": "SimPredictionLayer",
            "checkpoint": str(Path(args_cli.checkpoint).resolve()),
            "checkpoint_stat_before": checkpoint_before,
            "checkpoint_stat_after": checkpoint_after,
            "checkpoint_modified": checkpoint_before != checkpoint_after,
            "tau": float(args_cli.tau),
            "alignment_convention": str(args_cli.alignment_convention),
            "sc_gain_formula": str(args_cli.sc_gain_formula),
            "sc_occ_threshold": float(args_cli.sc_occ_threshold),
            "sc_conf_threshold": float(args_cli.sc_conf_threshold),
            "sc_count_mode": str(args_cli.sc_count_mode),
            "calibration_table": str(Path(args_cli.calibration_table).resolve()) if args_cli.calibration_table else None,
            "sc_gain_weight": float(args_cli.sc_gain_weight),
            "sc_gain_cap": None if args_cli.sc_gain_cap is None else float(args_cli.sc_gain_cap),
            "sc_gain_cap_value": -1.0 if args_cli.sc_gain_cap is None else float(args_cli.sc_gain_cap),
            "score_gain_mode": str(args_cli.score_gain_mode),
            "static_prediction_step": None if args_cli.static_prediction_step is None else int(args_cli.static_prediction_step),
            "gain_mode": str(args_cli.gain_mode),
            "path_cost_mode": str(args_cli.path_cost_mode),
            "candidate_sampling_mode": str(args_cli.candidate_sampling_mode),
            "snap_start_to_traversable": bool(args_cli.snap_start_to_traversable),
            "max_snap_radius_cells": int(args_cli.max_snap_radius_cells),
            "motion_mode": str(args_cli.motion_mode),
            "max_steps": int(args_cli.max_steps),
            "steps_completed": int(steps_completed),
            "done_reason": str(terminal_done_reason or "no_transition"),
            "coverage_stop": float(args_cli.coverage_stop),
            "min_score": float(args_cli.min_score),
            "repeat_pose_patience": int(args_cli.repeat_pose_patience),
            "repeated_pose_count": int(repeated_pose_count),
            "voxel_size": float(args_cli.voxel_size),
            "pixel_stride": int(args_cli.pixel_stride),
            "map_bounds": {axis: [float(v) for v in bounds[axis]] for axis in ("x", "y", "z")},
            "total_wall_time": float(time.perf_counter() - total_wall_start),
            "average_map_predict_inference_time": metric_summary.get("map_predict_inference_time", {}).get("mean"),
            "average_map_predict_total_time": float(np.mean(prediction_total_times)) if prediction_total_times else None,
            "average_expert_time": metric_summary.get("expert_time", {}).get("mean"),
            "observed_ratio_start": float(start_ratio),
            "observed_ratio_end": float(end_ratio),
            "observed_ratio_delta": float(end_ratio - start_ratio),
            "total_delta_observed_ratio": float(end_ratio - start_ratio),
            "final_counts": final_summary,
            "final_pose": [float(v) for v in final_pose],
            "final_yaw": float(final_yaw),
            "metrics": metric_summary,
            "best_score": metric_summary.get("best_score", {}),
            "gain_exp": metric_summary.get("gain_exp", {}),
            "gain_sc": metric_summary.get("gain_sc", {}),
            "gain_hybrid": metric_summary.get("gain_hybrid", {}),
            "gain_occ": metric_summary.get("gain_occ", {}),
            "gain_conf": metric_summary.get("gain_conf", {}),
            "effective_gain_sc": metric_summary.get("effective_gain_sc", {}),
            "gain_hybrid_effective": metric_summary.get("gain_hybrid_effective", {}),
            "weighted_gain_sc": metric_summary.get("weighted_gain_sc", {}),
            "gain_hybrid_weighted": metric_summary.get("gain_hybrid_weighted", {}),
            "utility_hybrid_effective": metric_summary.get("utility_hybrid_effective", {}),
            "utility_hybrid_weighted": metric_summary.get("utility_hybrid_weighted", {}),
            "candidates_with_gain_sc_positive": metric_summary.get("candidates_with_gain_sc_positive", {}),
            "candidates_with_effective_gain_sc_positive": metric_summary.get(
                "candidates_with_effective_gain_sc_positive",
                {},
            ),
            "predicted_unmeasured_voxels": metric_summary.get("predicted_unmeasured_voxels", {}),
            "no_valid_candidate_steps": metric_summary.get("no_valid_candidate_steps", []),
            "prediction_summaries": prediction_summaries,
            "step_records": step_records,
            "depth_summaries": depth_summaries,
            "output_paths": output_paths,
            "model_loaded_once": bool(predictor.model_loaded_once) if predictor is not None else False,
            "model_load_time": float(predictor.model_load_time) if predictor is not None else None,
            "gpu_name": predictor.gpu_name if predictor is not None else None,
            "gpu_memory_peak": int(max(gpu_memory_peaks)) if gpu_memory_peaks else None,
            "hardware_context": {
                "cpu": "AMD Ryzen 9 9950X3D",
                "cpu_threads": 32,
                "ram_gb": 32,
                "gpu": "NVIDIA RTX 5080",
                "torch_device": str(args_cli.sscnet_device),
                "torch_num_threads": int(args_cli.torch_num_threads),
            },
            "strict_no_prediction_write": True,
            "prediction_used_for_traversability": False,
            "prediction_used_for_collision": False,
            "prediction_used_for_a_star": False,
            "prediction_blocks_rays": False,
            "prediction_writeback": False,
            "prediction_used_for_candidate_reachability": False,
            "prediction_used_for_collision_checking": False,
            "prediction_used_for_a_star_traversability": False,
            "rl_optimizer_training_run": False,
            "rl_optimizer_bc_il_training_run": False,
            "rl_or_ppo_training": False,
            "optimizer_step": False,
            "behavior_cloning_training": False,
            "imitation_learning_training": False,
            "sscnet_training": False,
            "leakage_checks": leakage_checks(
                any(bool(t["leakage_checks"]["prediction_wrote_observed_map"]) for t in transitions),
                a_star_planner=True,
                prediction_mode=str(args_cli.prediction_mode),
                prediction_layer="SimPredictionLayer",
            ),
            "limitations": STAGE4A6_LIMITATIONS,
        }
        save_json(episode_dir / "episode_summary.json", episode_summary)

        manifest_record = {
            "episode_id": str(args_cli.episode_id),
            "episode_dir": str(episode_dir),
            "status": "ok",
            "steps_completed": int(steps_completed),
            "done_reason": str(episode_summary["done_reason"]),
            "scene_seed": int(args_cli.scene_seed),
            "start_variant": str(args_cli.start_variant),
            "observed_ratio_start": float(episode_summary["observed_ratio_start"]),
            "observed_ratio_end": float(episode_summary["observed_ratio_end"]),
            "total_delta_observed_ratio": float(episode_summary["total_delta_observed_ratio"]),
            "no_valid_candidate_steps": metric_summary.get("no_valid_candidate_steps", []),
            "prediction_mode": str(args_cli.prediction_mode),
            "tau": float(args_cli.tau),
            "alignment_convention": str(args_cli.alignment_convention),
            "sc_gain_formula": str(args_cli.sc_gain_formula),
            "sc_occ_threshold": float(args_cli.sc_occ_threshold),
            "sc_conf_threshold": float(args_cli.sc_conf_threshold),
            "sc_count_mode": str(args_cli.sc_count_mode),
            "calibration_table": str(Path(args_cli.calibration_table).resolve()) if args_cli.calibration_table else None,
            "sc_gain_weight": float(args_cli.sc_gain_weight),
            "sc_gain_cap": None if args_cli.sc_gain_cap is None else float(args_cli.sc_gain_cap),
            "score_gain_mode": str(args_cli.score_gain_mode),
            "path_cost_mode": str(args_cli.path_cost_mode),
            "candidate_sampling_mode": str(args_cli.candidate_sampling_mode),
            "motion_mode": str(args_cli.motion_mode),
            "summary": str(episode_dir / "episode_summary.json"),
            "leakage_checks": episode_summary["leakage_checks"],
        }
        if not args_cli.no_manifest:
            append_jsonl(output_root / "manifest.jsonl", manifest_record)
        return episode_summary
    except BaseException as exc:
        assert observed_state is not None
        summary = _build_failure_record(
            episode_dir=episode_dir,
            done_reason="failed",
            failure_stage=failure_stage,
            exc=exc,
            step_records=step_records,
            observed_state=observed_state,
            bounds=bounds,
            checkpoint_before=checkpoint_before,
        )
        append_jsonl(
            output_root / "manifest.jsonl",
            {
                "episode_id": str(args_cli.episode_id),
                "episode_dir": str(episode_dir),
                "status": "failed",
                "steps_completed": int(summary.get("steps_completed", 0)),
                "done_reason": "failed",
                "failure_stage": failure_stage,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def main() -> None:
    summary = run_rollout()
    print("Stage 4A-6 SC-aware simulator expert rollout complete.")
    print(f"episode_id: {summary['episode_id']}")
    print(f"episode_dir: {summary['episode_dir']}")
    print(f"scene_variant: {summary['scene_variant']}")
    print(f"prediction_mode: {summary['prediction_mode']}")
    print(f"tau: {summary.get('tau')}")
    print(f"alignment_convention: {summary.get('alignment_convention')}")
    print(f"sc_gain_formula: {summary.get('sc_gain_formula')}")
    print(f"sc_occ_threshold: {summary.get('sc_occ_threshold')}")
    print(f"sc_conf_threshold: {summary.get('sc_conf_threshold')}")
    print(f"score_gain_mode: {summary.get('score_gain_mode')}")
    print(f"sc_gain_weight: {summary.get('sc_gain_weight')}")
    print(f"sc_gain_cap: {summary.get('sc_gain_cap')}")
    print(f"path_cost_mode: {summary['path_cost_mode']}")
    print(f"candidate_sampling_mode: {summary.get('candidate_sampling_mode')}")
    print(f"steps_completed: {summary['steps_completed']}")
    print(f"done_reason: {summary['done_reason']}")
    print(
        "observed_ratio: "
        f"{summary['observed_ratio_start']:.6f} -> {summary['observed_ratio_end']:.6f} "
        f"delta={summary['total_delta_observed_ratio']:.6f}"
    )
    print(f"model_loaded_once: {summary.get('model_loaded_once')}")
    print(f"average_map_predict_total_time: {summary.get('average_map_predict_total_time')}")
    print(f"gpu_memory_peak: {summary.get('gpu_memory_peak')}")
    for key, path in summary["output_paths"].items():
        print(f"output_{key}: {path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
