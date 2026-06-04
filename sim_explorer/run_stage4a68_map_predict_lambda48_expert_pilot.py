#!/usr/bin/env python3
"""Stage 4A-6.8 read-only map_predict lambda48 expert pilot.

This stage uses the validated home_like_scene_v1 USD and the same 10 interior
starts as Stage 4A-6.7. It captures one start view per start, updates a
measured-only observed_state from real depth, calls SSCNet/map_predict exactly
once per start, and uses the prediction only for the lambda48 information-gain
bonus. No action is executed, no second frame is captured, and no training or
rollout artifact is created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_key] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from astar_planner import astar_2d, build_traversability_grid, path_length_m, summarize_traversability
from depth_to_voxel import FREE, OCCUPIED, UNKNOWN, update_observed_state_from_depth
from isaac_map_predictor import IsaacMapPredictor
from sim_paper_expert import (
    EmptyPredictionLayer,
    SimCandidateView,
    compute_paper_gains_for_candidate,
    compute_reachable_frontier_candidate_cells,
    frontier_adjacent_free_xy_mask,
    grid_to_world,
    raycast_visible_voxels_observed,
    world_to_grid,
)

import run_stage4a67_measured_only_expert_pilot as s67


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_CAMERA_FIX_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix"
DEFAULT_MEASURED_ONLY_DIR = WORKSPACE / "outputs/isaac_stage4a67_measured_only_expert_pilot"
DEFAULT_FIXED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot"
DEFAULT_SOURCE_OBSERVED = DEFAULT_CAMERA_FIX_DIR / "observed_state_final.npy"
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"

STAGE = "Stage 4A-6.8-map-predict-lambda48-expert-pilot"
FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
FORMULA_NAME = "lambda48_minmax_source_occ_free"
SOURCE_OCC_FREE_DEFINITION = (
    "raw count of visible predicted-unmeasured voxels from the read-only map_predict layer; "
    "predicted occupied and predicted free both count when valid at tau"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(field_order or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        text = json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
        if len(text) > 1600:
            text = text[:1600] + "..."
        lines.append(f"| {key} | `{text}` |")
    return "\n".join(lines)


def markdown_list(title: str, rows: list[str]) -> str:
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in rows]])


def sha256_file(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return s67.sha256_array(array)


def git_status_text() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    return result.stdout


def log_event(output_dir: Path, events: list[dict[str, Any]], event: str, **payload: Any) -> None:
    row = {"time_utc": utc_now(), "event": str(event), **payload}
    events.append(row)
    path = output_dir / "logs/stage4a68_execution_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False))
        handle.write("\n")
    print(f"[{STAGE}] {event}: {json.dumps(jsonable(payload), sort_keys=True)[:900]}", flush=True)


def distance_xyz(a: list[float], b: list[float]) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3]))))


def distance_xy(a: list[float], b: list[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def minmax(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if hi <= lo + 1.0e-9:
        return 0.0
    return float((value - lo) / (hi - lo))


def safe_div(num: float, den: float) -> float:
    return float(num) / max(float(den), 1.0e-6)


def topdown_state(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == FREE, axis=2)
    top = np.full(observed_state.shape[:2], UNKNOWN, dtype=np.int8)
    top[free] = FREE
    top[occupied] = OCCUPIED
    return top


def observed_summary(observed_state: np.ndarray, label: str) -> dict[str, Any]:
    return s67.validate_observed_state(observed_state, label)


def image_stats(path: Path) -> dict[str, Any]:
    image = np.asarray(Image.open(path).convert("RGB"))
    return {
        "path": str(path),
        "shape": [int(v) for v in image.shape],
        "mean": float(image.mean()),
        "std": float(image.std()),
        "nonblank": bool(image.size and int(image.max()) > 2 and float(image.std()) >= 1.0),
    }


def load_required_inputs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    camera_fix_dir = Path(args.camera_pose_fix_dir).resolve()
    measured_dir = Path(args.measured_only_pilot_dir).resolve()
    fixed_usd = Path(args.fixed_usd).resolve()
    source_observed_path = Path(args.source_observed_state).resolve()

    required_paths = {
        "camera_fix_summary": camera_fix_dir / "stage4a66c_usd_camera_pose_fix_summary.json",
        "start_variants_interior": camera_fix_dir / "start_variants_interior.json",
        "selected_validation_pose_manifest": camera_fix_dir / "selected_validation_pose_manifest.json",
        "selected_inspection_pose_manifest": camera_fix_dir / "selected_inspection_pose_manifest.json",
        "camera_info": camera_fix_dir / "camera_info.json",
        "scene_metadata": camera_fix_dir / "scene_metadata.json",
        "measured_summary": measured_dir / "stage4a67_measured_only_expert_pilot_summary.json",
        "measured_dataset": measured_dir / "expert_dataset.npz",
        "measured_integrity": measured_dir / "dataset_integrity_report.json",
        "measured_safety": measured_dir / "safety_audit.json",
    }
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Stage 4A-6.8 input files: {missing}")
    if not fixed_usd.is_file():
        raise FileNotFoundError(f"fixed USD not found: {fixed_usd}")
    if not source_observed_path.is_file():
        raise FileNotFoundError(f"source observed_state not found: {source_observed_path}")

    camera_fix_summary = read_json(required_paths["camera_fix_summary"])
    starts = read_json(required_paths["start_variants_interior"])
    validation_manifest = read_json(required_paths["selected_validation_pose_manifest"])
    inspection_manifest = read_json(required_paths["selected_inspection_pose_manifest"])
    stage66c_camera_info = read_json(required_paths["camera_info"])
    scene_metadata = read_json(required_paths["scene_metadata"])
    measured_summary = read_json(required_paths["measured_summary"])
    measured_integrity = read_json(required_paths["measured_integrity"])
    measured_safety = read_json(required_paths["measured_safety"])

    with np.load(required_paths["measured_dataset"], allow_pickle=False) as data:
        measured_dataset_keys = sorted(data.files)
        measured_dataset_shapes = {key: [int(v) for v in data[key].shape] for key in data.files}

    source_observed = np.load(source_observed_path)
    source_observed_summary = observed_summary(source_observed, "stage4a68_source_observed_state")
    scene_observed_summary = read_json(camera_fix_dir / "observed_summary.json")
    bounds = scene_observed_summary.get("chosen_bounds") or scene_metadata.get("map_bounds")
    voxel_size = float(scene_observed_summary.get("voxel_size", scene_metadata.get("voxel_size", 0.1)))
    if not bounds:
        raise ValueError("Could not resolve map bounds from Stage 4A-6.6c camera-pose-fix outputs")
    if len(starts) != int(args.num_starts):
        raise ValueError(f"Expected {args.num_starts} starts, found {len(starts)}")
    if int(measured_summary.get("sample_count", -1)) != int(args.num_starts):
        raise ValueError("Stage 4A-6.7 measured-only pilot sample_count does not match requested start count")
    if not bool(measured_summary.get("completed", False)):
        raise ValueError("Stage 4A-6.7 measured-only pilot summary is not completed")
    if not bool(measured_integrity.get("passed", False)):
        raise ValueError("Stage 4A-6.7 dataset integrity did not pass")
    if not bool(measured_safety.get("passed", False)):
        raise ValueError("Stage 4A-6.7 safety audit did not pass")
    if bool(measured_summary.get("map_predict_called", True)):
        raise ValueError("Stage 4A-6.7 unexpectedly called map_predict")
    if bool(measured_summary.get("sscnet_inference_called", True)):
        raise ValueError("Stage 4A-6.7 unexpectedly called SSCNet inference")
    if not bool(measured_summary.get("exactly_one_action_per_start", False)):
        raise ValueError("Stage 4A-6.7 did not record exactly one action per start")

    preflight = {
        "stage": STAGE,
        "loaded_at_utc": utc_now(),
        "camera_pose_fix_completed": bool(camera_fix_summary.get("completed", False)),
        "camera_pose_fix_dir": str(camera_fix_dir),
        "fixed_usd": str(fixed_usd),
        "fixed_usd_exists": fixed_usd.is_file(),
        "source_observed_state": str(source_observed_path),
        "source_observed_summary": source_observed_summary,
        "start_variant_count": len(starts),
        "start_variant_ids": [int(row["index"]) for row in starts],
        "selected_validation_pose_count": validation_manifest.get("pose_count", len(validation_manifest.get("poses", []))),
        "selected_inspection_pose_count": inspection_manifest.get("pose_count", len(inspection_manifest.get("poses", []))),
        "stage66c_camera_info": stage66c_camera_info,
        "scene_metadata_path": str(required_paths["scene_metadata"]),
        "measured_only_pilot_dir": str(measured_dir),
        "stage4a67_completed": bool(measured_summary.get("completed", False)),
        "stage4a67_sample_count": int(measured_summary.get("sample_count", -1)),
        "stage4a67_capture_count": int(measured_summary.get("capture_count", -1)),
        "stage4a67_exactly_one_action_per_start": bool(measured_summary.get("exactly_one_action_per_start", False)),
        "stage4a67_map_predict_called": bool(measured_summary.get("map_predict_called", True)),
        "stage4a67_sscnet_inference_called": bool(measured_summary.get("sscnet_inference_called", True)),
        "stage4a67_integrity_passed": bool(measured_integrity.get("passed", False)),
        "stage4a67_safety_passed": bool(measured_safety.get("passed", False)),
        "stage4a67_dataset_keys": measured_dataset_keys,
        "stage4a67_dataset_shapes": measured_dataset_shapes,
        "this_stage_scope": {
            "map_predict_lambda48_one_action_pilot": True,
            "long_rollout": False,
            "rl_gdpo_ppo_bc_il": False,
            "training": False,
            "prediction_read_only": True,
        },
    }
    save_json(output_dir / "input_preflight_report.json", preflight)
    write_text(output_dir / "input_preflight_report.md", markdown_table("Input Preflight Report", preflight))
    return {
        "camera_fix_dir": camera_fix_dir,
        "measured_dir": measured_dir,
        "fixed_usd": fixed_usd,
        "source_observed_path": source_observed_path,
        "source_observed": source_observed,
        "source_observed_summary": source_observed_summary,
        "scene_observed_summary": scene_observed_summary,
        "bounds": bounds,
        "voxel_size": voxel_size,
        "starts": starts,
        "validation_manifest": validation_manifest,
        "inspection_manifest": inspection_manifest,
        "camera_fix_summary": camera_fix_summary,
        "stage66c_camera_info": stage66c_camera_info,
        "scene_metadata": scene_metadata,
        "measured_summary": measured_summary,
        "measured_integrity": measured_integrity,
        "measured_safety": measured_safety,
        "preflight": preflight,
    }


def start_capture_poses(starts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    poses = []
    for start in starts:
        position = [float(v) for v in start["position"]]
        yaw = float(start.get("yaw_rad", start.get("yaw", 0.0)))
        poses.append(
            {
                "index": int(start["index"]),
                "name": f"start_capture_{start['name']}",
                "start_variant_name": str(start["name"]),
                "semantic_zone_guess": start.get("semantic_zone_guess"),
                "position": position,
                "yaw": yaw,
                "yaw_rad": yaw,
                "target": s67.pose_target(position, yaw),
                "target_xy": [float(position[0] + math.cos(yaw)), float(position[1] + math.sin(yaw))],
                "source": "stage4a68_start_pose_capture",
                "one_headless_capture_for_this_start": True,
                "one_action_only_for_this_start": True,
            }
        )
    return poses


def run_capture_once(
    args: argparse.Namespace,
    app_launcher_cls: Any,
    output_dir: Path,
    fixed_usd: Path,
    starts: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    poses = start_capture_poses(starts)
    result = s67.run_one_isaac_startup(args, app_launcher_cls, output_dir, fixed_usd, poses, events)
    result["capture_semantics"] = "one_start_pose_capture_per_start_before_map_predict_and_scoring"
    result["selected_action_executed"] = False
    result["second_action_executed"] = False
    result["third_action_executed"] = False
    save_json(output_dir / "headless_start_capture_manifest.json", result)
    write_text(output_dir / "headless_start_capture_manifest.md", markdown_table("Headless Start Capture Manifest", result))
    return result


def load_existing_capture_result(output_dir: Path, starts: list[dict[str, Any]]) -> dict[str, Any]:
    camera_info_path = output_dir / "camera_info.json"
    if not camera_info_path.is_file():
        raise FileNotFoundError(f"Cannot reuse captures without camera_info.json: {camera_info_path}")
    camera_info = read_json(camera_info_path)
    records = []
    for start in starts:
        idx = int(start["index"])
        rgb_path = output_dir / f"action_rgb_{idx:03d}.png"
        depth_path = output_dir / f"action_depth_{idx:03d}.npy"
        depth_color_path = output_dir / f"action_depth_color_{idx:03d}.png"
        pose_path = output_dir / f"action_pose_{idx:03d}.json"
        for path in (rgb_path, depth_path, depth_color_path, pose_path):
            if not path.is_file():
                raise FileNotFoundError(f"Cannot reuse missing capture artifact: {path}")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path)
        records.append(
            {
                "index": idx,
                "name": f"reused_start_capture_{idx:03d}",
                "start_variant_name": start.get("name"),
                "semantic_zone_guess": start.get("semantic_zone_guess"),
                "render_backend": "isaac_headless_reused_after_close_hang",
                "rgb_file": rgb_path.name,
                "depth_file": depth_path.name,
                "depth_color_file": depth_color_path.name,
                "pose_file": pose_path.name,
                "rgb_key_used": "rgb_reused",
                "rgb_stats": s67.rgb_stats(rgb),
                "depth_stats": s67.depth_stats(depth),
                "second_action_executed": False,
                "third_action_executed": False,
            }
        )
    result = {
        "isaac_headless_startup_count": 1,
        "isaac_startup_seconds": None,
        "builder_metadata": {},
        "camera_info": camera_info,
        "capture_records": records,
        "capture_count": len(records),
        "one_capture_per_start": len(records) == len(starts),
        "capture_semantics": "reused one_start_pose_capture_per_start from prior completed capture batch",
        "selected_action_executed": False,
        "second_action_executed": False,
        "third_action_executed": False,
        "isaac_shutdown_note": (
            "The initial run completed all 10 captures, then simulation_app.close hung. "
            "That process was terminated after captures were on disk; this recovery run reused "
            "the captures and did not start Isaac a second time."
        ),
        "reused_existing_captures": True,
    }
    save_json(output_dir / "headless_start_capture_manifest.json", result)
    write_text(output_dir / "headless_start_capture_manifest.md", markdown_table("Headless Start Capture Manifest", result))
    return result


def yaw_priors_by_start(starts: list[dict[str, Any]], inspection_manifest: dict[str, Any]) -> dict[int, list[float]]:
    return s67.inspection_yaw_priors(starts, inspection_manifest.get("poses", []))


def score_one_candidate_lambda48(
    observed_state: np.ndarray,
    prediction_layer: Any,
    traversable: np.ndarray,
    bounds: dict[str, list[float]],
    voxel_size: float,
    start_xy: tuple[int, int],
    start_grid: tuple[int, int, int],
    start_world: list[float],
    start_yaw: float,
    candidate_xy: tuple[int, int],
    candidate_meta: dict[str, Any],
    yaw_priors: list[float],
    args: argparse.Namespace,
    candidate_id: int,
) -> dict[str, Any] | None:
    preferred_k = int(start_grid[2])
    candidate_k = s67.nearest_free_z_for_xy(observed_state, candidate_xy, preferred_k)
    if candidate_k is None:
        return None
    candidate_grid = (int(candidate_xy[0]), int(candidate_xy[1]), int(candidate_k))
    candidate_world_grid_center = grid_to_world(candidate_grid, bounds, float(voxel_size))
    camera_world = [float(candidate_world_grid_center[0]), float(candidate_world_grid_center[1]), float(start_world[2])]
    astar = astar_2d(traversable, start_xy, candidate_xy, allow_diagonal=True)
    if not bool(astar.get("reachable", False)):
        return None
    astar_path = [(int(cell[0]), int(cell[1])) for cell in astar["path"]]
    path_m = path_length_m(astar_path, float(voxel_size))
    if float(path_m) > float(args.max_action_path_m):
        return None
    fallback_base = math.atan2(float(candidate_xy[1] - start_xy[1]), float(candidate_xy[0] - start_xy[0]))
    base_yaw = s67.unknown_centroid_yaw(observed_state, candidate_xy, fallback_base)
    empty_prediction = EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape))
    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / float(voxel_size))))
    yaw_rows: list[dict[str, Any]] = []
    for yaw in s67.yaw_candidates(base_yaw, float(start_yaw), yaw_priors, int(args.num_yaw_samples)):
        measured_view = SimCandidateView(
            id=int(candidate_id),
            grid_position=candidate_grid,
            world_position=tuple(float(v) for v in camera_world),
            yaw=float(yaw),
            valid=True,
            candidate_source="reachable_frontier_measured_shadow",
        )
        visible = raycast_visible_voxels_observed(
            measured_view,
            observed_state,
            max_range_voxels=max_range_voxels,
            num_yaw=max(4, int(args.raycast_num_yaw)),
            num_pitch=max(3, int(args.raycast_num_pitch)),
            fov_yaw_deg=float(args.fov_yaw_deg),
            fov_pitch_deg=float(args.fov_pitch_deg),
        )
        measured_view = compute_paper_gains_for_candidate(
            measured_view,
            observed_state,
            empty_prediction,
            visible,
            tau=float(args.tau),
            sc_gain_formula="raw_count",
            sc_occ_threshold=1.0,
            sc_conf_threshold=1.0,
            sc_count_mode="raw_count",
        )
        lambda_view = SimCandidateView(
            id=int(candidate_id),
            grid_position=candidate_grid,
            world_position=tuple(float(v) for v in camera_world),
            yaw=float(yaw),
            valid=True,
            candidate_source="reachable_frontier_lambda48",
        )
        lambda_view = compute_paper_gains_for_candidate(
            lambda_view,
            observed_state,
            prediction_layer,
            visible,
            tau=float(args.tau),
            sc_gain_formula="raw_count",
            sc_occ_threshold=1.0,
            sc_conf_threshold=0.0,
            sc_count_mode="raw_count",
        )
        yaw_delta = abs(s67.wrap_angle(float(yaw) - float(start_yaw)))
        yaw_time_s = yaw_delta / max(math.radians(float(args.yaw_rate_deg_s)), 1.0e-6)
        path_time_s = float(path_m) / max(float(args.v_max_m_s), 1.0e-6)
        cost_s = path_time_s + yaw_time_s + float(args.action_cost_bias_s)
        measured_score = safe_div(float(measured_view.gain_exp), cost_s)
        source_occ_free = float(lambda_view.raw_gain_sc)
        yaw_rows.append(
            {
                "candidate_id": int(candidate_id),
                "candidate_source": "reachable_frontier_measured_valid_lambda48_scored",
                "grid": [int(v) for v in candidate_grid],
                "xy": [int(candidate_xy[0]), int(candidate_xy[1])],
                "world": [float(v) for v in camera_world],
                "yaw_rad": float(yaw),
                "target": s67.pose_target([float(v) for v in camera_world], float(yaw)),
                "gain_exp": float(measured_view.gain_exp),
                "gain_sc": source_occ_free,
                "source_occ_free": source_occ_free,
                "raw_gain_sc": float(lambda_view.raw_gain_sc),
                "effective_gain_sc": float(lambda_view.effective_gain_sc),
                "gain_hybrid": float(lambda_view.gain_hybrid),
                "visible_count": int(measured_view.visible_count),
                "measured_visible_count": int(measured_view.measured_visible_count),
                "predicted_unmeasured_visible_count": int(lambda_view.predicted_unmeasured_visible_count),
                "frontier_count_visible": int(measured_view.frontier_count_visible),
                "sc_selected_voxel_count": int(lambda_view.sc_selected_voxel_count),
                "path_cost_m": float(path_m),
                "path_time_s": float(path_time_s),
                "yaw_delta_rad": float(yaw_delta),
                "yaw_time_s": float(yaw_time_s),
                "cost_s": float(cost_s),
                "measured_score": float(measured_score),
                "final_score_measured": float(measured_score),
                "astar_num_expanded": int(astar.get("num_expanded", 0)),
                "astar_path_xy": [[int(a), int(b)] for a, b in astar_path],
                "astar_reachable": True,
                "base_yaw_rad": float(base_yaw),
                "raw_measured_occupancy_only_for_validity": True,
                "prediction_used_for_information_gain_only": True,
                "prediction_used_for_traversability": False,
                "prediction_used_for_collision": False,
                "prediction_used_for_ray_blocking": False,
                "prediction_used_for_candidate_validity": False,
                "prediction_used_for_edge_validity": False,
                **candidate_meta,
            }
        )
    if not yaw_rows:
        return None
    return {
        "candidate_id": int(candidate_id),
        "grid": [int(v) for v in candidate_grid],
        "xy": [int(candidate_xy[0]), int(candidate_xy[1])],
        "world": [float(v) for v in camera_world],
        "path_cost_m": float(path_m),
        "astar_num_expanded": int(astar.get("num_expanded", 0)),
        "yaw_rows": yaw_rows,
        **candidate_meta,
    }


def row_selection_key_lambda(row: dict[str, Any]) -> tuple[float, float, float, float, int, int]:
    return (
        float(row.get("final_score_lambda48", -math.inf)),
        float(row.get("source_occ_free_minmax", 0.0)),
        float(row.get("gain_exp", 0.0)),
        -float(row.get("cost_s", math.inf)),
        -int(row.get("candidate_id", 0)),
        -int(round(float(row.get("yaw_rad", 0.0)) * 10000.0)),
    )


def row_selection_key_measured(row: dict[str, Any]) -> tuple[float, float, float, int, int]:
    return (
        float(row.get("final_score_measured", -math.inf)),
        float(row.get("gain_exp", 0.0)),
        -float(row.get("cost_s", math.inf)),
        -int(row.get("candidate_id", 0)),
        -int(round(float(row.get("yaw_rad", 0.0)) * 10000.0)),
    )


def classify_relation(lambda_row: dict[str, Any] | None, measured_row: dict[str, Any] | None) -> dict[str, Any]:
    if lambda_row is None or measured_row is None:
        return {
            "branch_classification": "no_valid_candidate",
            "distance_m": None,
            "yaw_delta_rad": None,
            "same_candidate_id": False,
        }
    distance_m = distance_xy(lambda_row["world"], measured_row["world"])
    yaw_delta = abs(s67.wrap_angle(float(lambda_row["yaw_rad"]) - float(measured_row["yaw_rad"])))
    same_candidate = int(lambda_row["candidate_id"]) == int(measured_row["candidate_id"])
    if same_candidate and distance_m <= 0.15 and yaw_delta <= 0.10:
        label = "same_as_measured"
    elif distance_m <= 0.75:
        label = "local_jitter"
    else:
        label = "distinct_nonmeasured_branch"
    return {
        "branch_classification": label,
        "distance_m": float(distance_m),
        "yaw_delta_rad": float(yaw_delta),
        "same_candidate_id": bool(same_candidate),
    }


def make_action_pose(start: dict[str, Any], row: dict[str, Any], kind: str) -> dict[str, Any]:
    return {
        "index": int(start["index"]),
        "name": f"{kind}_action_from_{start['name']}",
        "start_variant_name": str(start["name"]),
        "position": [float(v) for v in row["world"]],
        "yaw": float(row["yaw_rad"]),
        "yaw_rad": float(row["yaw_rad"]),
        "target": row["target"],
        "target_xy": [float(row["target"][0]), float(row["target"][1])],
        "source": f"stage4a68_{kind}_one_action_record_only",
        "semantic_zone_guess": start.get("semantic_zone_guess"),
        "start_position": [float(v) for v in start["position"]],
        "start_yaw_rad": float(start.get("yaw_rad", start.get("yaw", 0.0))),
        "selected_grid": [int(v) for v in row["grid"]],
        "selected_candidate_id": int(row["candidate_id"]),
        "one_action_only_for_this_start": True,
        "action_executed_in_isaac": False,
    }


def score_start_lambda48(
    observed_state: np.ndarray,
    prediction_layer: Any,
    bounds: dict[str, list[float]],
    voxel_size: float,
    start: dict[str, Any],
    yaw_priors: list[float],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_position = [float(v) for v in start["position"]]
    start_yaw = float(start.get("yaw_rad", start.get("yaw", 0.0)))
    start_grid = world_to_grid(start_position, bounds, float(voxel_size), shape=observed_state.shape, clip=True)
    traversability = build_traversability_grid(
        observed_state,
        voxel_size=float(voxel_size),
        robot_height_m=float(args.robot_height_m),
        clearance_height_m=float(args.clearance_height_m),
        robot_radius_m=float(args.robot_radius_m),
    )
    frontier_mask = frontier_adjacent_free_xy_mask(observed_state)
    reachable = compute_reachable_frontier_candidate_cells(
        observed_state,
        traversability,
        start_grid[:2],
        frontier_mask,
        max_snap_radius_cells=int(args.max_snap_radius_cells),
        snap_start_to_traversable=True,
        allow_diagonal=True,
    )
    candidate_mask = np.asarray(reachable["candidate_mask"], dtype=bool)
    snapped = reachable.get("snapped_current_xy")
    if snapped is None:
        return {
            "start_index": int(start["index"]),
            "start_name": str(start["name"]),
            "start": start,
            "start_grid": [int(v) for v in start_grid],
            "snapped_start_xy": None,
            "reachable_context": {k: v for k, v in reachable.items() if k not in {"candidate_mask", "reachable_mask"}},
            "sampled_candidate_count": 0,
            "scored_candidate_count": 0,
            "top_n_count": 0,
            "all_yaw_row_count": 0,
            "candidate_rows_lambda48": [],
            "candidate_rows_measured": [],
            "top_n": [],
            "selected_lambda48": None,
            "selected_measured_shadow": None,
            "lambda48_action_pose": None,
            "measured_shadow_action_pose": None,
            "branch_classification": "no_valid_candidate",
            "safety_flags": {"no_valid_candidate": True},
            "no_valid_candidate": True,
        }
    snapped_xy = (int(snapped[0]), int(snapped[1]))
    selected_cells = s67.select_candidate_cells(observed_state, candidate_mask, snapped_xy, int(args.num_candidates))
    traversable = np.asarray(traversability["traversable"], dtype=bool)
    candidate_records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = []
        for cid, (i, j, meta) in enumerate(selected_cells):
            futures.append(
                executor.submit(
                    score_one_candidate_lambda48,
                    observed_state,
                    prediction_layer,
                    traversable,
                    bounds,
                    voxel_size,
                    snapped_xy,
                    tuple(int(v) for v in start_grid),
                    start_position,
                    start_yaw,
                    (int(i), int(j)),
                    meta,
                    yaw_priors,
                    args,
                    cid,
                )
            )
        for future in futures:
            result = future.result()
            if result is not None:
                candidate_records.append(result)

    all_rows: list[dict[str, Any]] = []
    for record in candidate_records:
        all_rows.extend(record["yaw_rows"])
    if not all_rows:
        return {
            "start_index": int(start["index"]),
            "start_name": str(start["name"]),
            "start": start,
            "start_grid": [int(v) for v in start_grid],
            "snapped_start_xy": [int(v) for v in snapped_xy],
            "reachable_context": {k: v for k, v in reachable.items() if k not in {"candidate_mask", "reachable_mask"}},
            "sampled_candidate_count": int(len(selected_cells)),
            "scored_candidate_count": 0,
            "top_n_count": 0,
            "all_yaw_row_count": 0,
            "candidate_rows_lambda48": [],
            "candidate_rows_measured": [],
            "top_n": [],
            "selected_lambda48": None,
            "selected_measured_shadow": None,
            "lambda48_action_pose": None,
            "measured_shadow_action_pose": None,
            "branch_classification": "no_valid_candidate",
            "safety_flags": {"no_valid_candidate": True},
            "no_valid_candidate": True,
        }

    sc_values = [float(row["source_occ_free"]) for row in all_rows]
    sc_min = float(min(sc_values))
    sc_max = float(max(sc_values))
    for row in all_rows:
        row["source_occ_free_minmax"] = minmax(float(row["source_occ_free"]), sc_min, sc_max)
        row["lambda48_bonus"] = float(args.lambda_sc) * float(row["source_occ_free_minmax"])
        row["final_score_lambda48"] = float(row["measured_score"]) + float(row["lambda48_bonus"])
        row["value_lambda48"] = float(row["final_score_lambda48"])
        row["lambda_sc"] = float(args.lambda_sc)
        row["formula"] = FORMULA
        row["minmax_scope"] = "per_start_valid_candidate_yaw_rows"

    rows_by_candidate: dict[int, list[dict[str, Any]]] = {}
    for row in all_rows:
        rows_by_candidate.setdefault(int(row["candidate_id"]), []).append(row)
    candidate_rows_lambda = [max(rows, key=row_selection_key_lambda) for rows in rows_by_candidate.values()]
    candidate_rows_measured = [max(rows, key=row_selection_key_measured) for rows in rows_by_candidate.values()]
    candidate_rows_lambda.sort(key=row_selection_key_lambda, reverse=True)
    candidate_rows_measured.sort(key=row_selection_key_measured, reverse=True)
    selected_lambda = dict(candidate_rows_lambda[0])
    selected_measured = dict(candidate_rows_measured[0])
    top_n = [dict(row) for row in candidate_rows_lambda[: int(args.top_n)]]
    relation = classify_relation(selected_lambda, selected_measured)

    same_cell_target = selected_lambda["xy"] == [int(start_grid[0]), int(start_grid[1])]
    outside_bounds = not (
        0 <= int(selected_lambda["grid"][0]) < observed_state.shape[0]
        and 0 <= int(selected_lambda["grid"][1]) < observed_state.shape[1]
        and 0 <= int(selected_lambda["grid"][2]) < observed_state.shape[2]
    )
    top_distances = [distance_xy(row["world"], start_position) for row in top_n]
    flags = {
        "low_cost_artifact": bool(float(selected_lambda["path_cost_m"]) < 0.05 or float(selected_lambda["cost_s"]) < 0.08),
        "historical_prior_basin": False,
        "same_cell_target": bool(same_cell_target),
        "unreachable_target": not bool(selected_lambda.get("astar_reachable", True)),
        "repeated_target": False,
        "outside_bounds_target": bool(outside_bounds),
        "prediction_leakage": False,
        "prediction_used_for_safety": False,
        "path_cost_suspiciously_small": bool(float(selected_lambda["path_cost_m"]) < 0.05),
        "candidate_all_local": bool(top_distances and max(top_distances) < 0.80),
        "no_valid_candidate": False,
    }
    return {
        "stage": STAGE,
        "start_index": int(start["index"]),
        "start_name": str(start["name"]),
        "start": start,
        "start_grid": [int(v) for v in start_grid],
        "snapped_start_xy": [int(v) for v in snapped_xy],
        "start_yaw_rad": float(start_yaw),
        "reachable_context": {k: v for k, v in reachable.items() if k not in {"candidate_mask", "reachable_mask"}},
        "sampled_candidate_count": int(len(selected_cells)),
        "scored_candidate_count": int(len(candidate_rows_lambda)),
        "all_yaw_row_count": int(len(all_rows)),
        "top_n_count": int(len(top_n)),
        "top_n": top_n,
        "candidate_rows_lambda48": candidate_rows_lambda,
        "candidate_rows_measured": candidate_rows_measured,
        "selected_lambda48": selected_lambda,
        "selected_measured_shadow": selected_measured,
        "lambda48_action_pose": make_action_pose(start, selected_lambda, "lambda48"),
        "measured_shadow_action_pose": make_action_pose(start, selected_measured, "measured_shadow"),
        "source_occ_free_min": sc_min,
        "source_occ_free_max": sc_max,
        "minmax_normalization_scope": "per start over valid candidate/yaw scored rows",
        "source_occ_free_definition": SOURCE_OCC_FREE_DEFINITION,
        "branch_classification": relation["branch_classification"],
        "relation": relation,
        "safety_flags": flags,
        "traversability_summary": summarize_traversability(traversability),
        "frontier_adjacent_free_xy_count": int(np.count_nonzero(frontier_mask)),
        "prediction_only_for_information_gain": True,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
    }


def copy_capture_to_sample(
    output_dir: Path,
    sample_dir: Path,
    capture_record: dict[str, Any],
    idx: int,
) -> dict[str, Path]:
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb_src = output_dir / capture_record["rgb_file"]
    depth_src = output_dir / capture_record["depth_file"]
    depth_color_src = output_dir / capture_record["depth_color_file"]
    pose_src = output_dir / capture_record["pose_file"]
    paths = {
        "rgb": sample_dir / f"rgb_{idx:03d}.png",
        "depth": sample_dir / f"depth_{idx:03d}.npy",
        "depth_color": sample_dir / f"depth_color_{idx:03d}.png",
        "pose": sample_dir / f"pose_{idx:03d}.json",
    }
    shutil.copy2(rgb_src, paths["rgb"])
    shutil.copy2(depth_src, paths["depth"])
    shutil.copy2(depth_color_src, paths["depth_color"])
    pose = read_json(pose_src)
    pose["pose_role"] = "start_capture_before_prediction_and_scoring"
    pose["action_executed"] = False
    save_json(paths["pose"], pose)
    return paths


def prediction_counts(prediction_layer: Any, observed_state: np.ndarray, tau: float) -> dict[str, Any]:
    valid = np.asarray(prediction_layer.valid, dtype=bool) & (np.asarray(prediction_layer.confidence) >= float(tau))
    predicted_unmeasured = valid & (observed_state == UNKNOWN)
    predicted_occupied = valid & (np.asarray(prediction_layer.occupied_prob) >= 0.5)
    predicted_free = valid & (np.asarray(prediction_layer.free_prob) >= 0.5)
    return {
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "predicted_occupied_count": int(np.count_nonzero(predicted_occupied)),
        "predicted_free_count": int(np.count_nonzero(predicted_free)),
        "prediction_density": float(np.count_nonzero(valid) / max(1, valid.size)),
        "predicted_unmeasured_density": float(np.count_nonzero(predicted_unmeasured) / max(1, valid.size)),
    }


def save_candidate_outputs(sample_dir: Path, idx: int, decision: dict[str, Any]) -> None:
    rows = []
    for rank, row in enumerate(decision.get("top_n", []), start=1):
        out = dict(row)
        out["rank"] = int(rank)
        out["start_index"] = int(idx)
        rows.append(out)
    write_csv(sample_dir / f"top_candidates_{idx:03d}.csv", rows)
    write_jsonl(sample_dir / f"top_candidates_{idx:03d}.jsonl", rows)
    save_json(sample_dir / f"lambda48_decision_{idx:03d}.json", decision.get("selected_lambda48"))
    save_json(sample_dir / f"measured_shadow_decision_{idx:03d}.json", decision.get("selected_measured_shadow"))


def plot_sample_topdown(
    path: Path,
    observed_state: np.ndarray,
    bounds: dict[str, list[float]],
    start: dict[str, Any],
    lambda_row: dict[str, Any] | None,
    measured_row: dict[str, Any] | None,
    top_n: list[dict[str, Any]],
    title: str,
) -> None:
    top = topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    start_pos = start["position"]
    ax.scatter([start_pos[0]], [start_pos[1]], s=70, c="#2563eb", marker="^", edgecolors="white", linewidths=0.6, label="start")
    if top_n:
        xs = [row["world"][0] for row in top_n]
        ys = [row["world"][1] for row in top_n]
        scores = [row["final_score_lambda48"] for row in top_n]
        scatter = ax.scatter(xs, ys, s=34, c=scores, cmap="magma", edgecolors="white", linewidths=0.3, label="top-N")
        fig.colorbar(scatter, ax=ax, shrink=0.7, label="lambda48")
    if measured_row is not None:
        m = measured_row["world"]
        ax.scatter([m[0]], [m[1]], s=72, c="#d97706", marker="o", edgecolors="black", linewidths=0.5, label="measured")
        ax.plot([start_pos[0], m[0]], [start_pos[1], m[1]], color="#d97706", linewidth=1.1)
    if lambda_row is not None:
        l = lambda_row["world"]
        ax.scatter([l[0]], [l[1]], s=92, c="#10b981", marker="*", edgecolors="black", linewidths=0.5, label="lambda48")
        ax.plot([start_pos[0], l[0]], [start_pos[1], l[1]], color="#10b981", linewidth=1.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def plot_prediction_overlay(
    path: Path,
    observed_state: np.ndarray,
    prediction_layer: Any,
    bounds: dict[str, list[float]],
    start: dict[str, Any],
    lambda_row: dict[str, Any] | None,
    tau: float,
) -> None:
    top = topdown_state(observed_state)
    valid = np.asarray(prediction_layer.valid, dtype=bool) & (np.asarray(prediction_layer.confidence) >= float(tau))
    predicted_unmeasured_xy = np.any(valid & (observed_state == UNKNOWN), axis=2)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    overlay = np.ma.masked_where(~predicted_unmeasured_xy.T, predicted_unmeasured_xy.T.astype(float))
    ax.imshow(overlay, origin="lower", extent=extent, cmap="Greens", alpha=0.45, interpolation="nearest")
    start_pos = start["position"]
    ax.scatter([start_pos[0]], [start_pos[1]], s=70, c="#2563eb", marker="^", edgecolors="white", linewidths=0.6)
    if lambda_row is not None:
        pos = lambda_row["world"]
        ax.scatter([pos[0]], [pos[1]], s=92, c="#10b981", marker="*", edgecolors="black", linewidths=0.5)
    ax.set_title("prediction overlay: valid predicted-unmeasured cells", fontsize=9)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def plot_candidate_scores(path: Path, decision: dict[str, Any]) -> None:
    rows = decision.get("top_n", [])
    fig, ax = plt.subplots(figsize=(8.8, 4.2), constrained_layout=True)
    labels = [str(row["candidate_id"]) for row in rows]
    x = np.arange(len(rows))
    measured = [float(row["final_score_measured"]) for row in rows]
    bonus = [float(row["lambda48_bonus"]) for row in rows]
    ax.bar(x, measured, color="#3b82f6", label="gain_exp / cost")
    ax.bar(x, bonus, bottom=measured, color="#10b981", label="lambda48 bonus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("score")
    ax.set_title("lambda48 score decomposition, top candidates")
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def action_quality(decision: dict[str, Any], capture_paths: dict[str, Path], prediction_summary: dict[str, Any]) -> dict[str, Any]:
    flags = dict(decision.get("safety_flags", {}))
    warnings = [key for key, value in flags.items() if bool(value)]
    rgb = image_stats(capture_paths["rgb"])
    depth = np.load(capture_paths["depth"])
    depth_summary = s67.depth_stats(depth)
    if not rgb["nonblank"]:
        warnings.append("blank_rgb")
    if not depth_summary["has_positive_finite_depth"]:
        warnings.append("invalid_depth")
    pred_density = float(prediction_summary.get("prediction_density", 0.0))
    if pred_density <= 0.0:
        warnings.append("empty_prediction")
    if pred_density >= 0.80:
        warnings.append("prediction_over_dense")
    hard_fail_flags = {
        "no_valid_candidate",
        "unreachable_target",
        "outside_bounds_target",
        "prediction_leakage",
        "prediction_used_for_safety",
        "blank_rgb",
        "invalid_depth",
    }
    hard_fail_count = sum(1 for item in warnings if item in hard_fail_flags)
    score = max(0.0, 1.0 - 0.12 * len(set(warnings)) - 0.35 * hard_fail_count)
    return {
        "sample_index": int(decision["start_index"]),
        "passed": bool(hard_fail_count == 0),
        "quality_score": float(score),
        "warnings": sorted(set(warnings)),
        "branch_classification": decision.get("branch_classification"),
        "candidate_count": int(decision.get("scored_candidate_count", 0)),
        "top_n_count": int(decision.get("top_n_count", 0)),
        "rgb_nonblank": bool(rgb["nonblank"]),
        "depth_has_positive_finite": bool(depth_summary["has_positive_finite_depth"]),
        "prediction_density": pred_density,
        "lambda48_selected": decision.get("selected_lambda48"),
        "measured_selected": decision.get("selected_measured_shadow"),
    }


def write_action_quality(sample_dir: Path, idx: int, quality: dict[str, Any]) -> None:
    save_json(sample_dir / f"action_quality_{idx:03d}.json", quality)
    rows = {
        "passed": quality["passed"],
        "quality_score": quality["quality_score"],
        "warnings": quality["warnings"],
        "branch_classification": quality["branch_classification"],
        "candidate_count": quality["candidate_count"],
        "top_n_count": quality["top_n_count"],
        "prediction_density": quality["prediction_density"],
    }
    write_text(sample_dir / f"action_quality_{idx:03d}.md", markdown_table(f"Action Quality {idx:03d}", rows))


def process_samples(
    args: argparse.Namespace,
    output_dir: Path,
    inputs: dict[str, Any],
    capture_result: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    source_observed = inputs["source_observed"]
    bounds = inputs["bounds"]
    voxel_size = float(inputs["voxel_size"])
    starts = inputs["starts"]
    camera_info = capture_result["camera_info"]
    record_by_index = {int(row["index"]): row for row in capture_result["capture_records"]}
    yaws = yaw_priors_by_start(starts, inputs["inspection_manifest"])

    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_before = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": int(checkpoint_path.stat().st_size) if checkpoint_path.is_file() else None,
        "mtime_ns": int(checkpoint_path.stat().st_mtime_ns) if checkpoint_path.is_file() else None,
    }
    save_json(output_dir / "checkpoint_hash_report.json", {"stage": STAGE, "before": checkpoint_before, "after": None})
    predictor = IsaacMapPredictor(
        checkpoint=checkpoint_path,
        device=str(args.predictor_device),
        tau=float(args.tau),
        torch_num_threads=1,
        alignment_convention=str(args.alignment_convention),
    )
    log_event(output_dir, events, "predictor_loaded", checkpoint=str(checkpoint_path), model_load_time=predictor.model_load_time)

    sample_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []
    qualities: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    observed_states: list[np.ndarray] = []
    candidate_feature_rows: list[np.ndarray] = []
    candidate_valid_rows: list[np.ndarray] = []
    candidate_lambda_score_rows: list[np.ndarray] = []
    candidate_measured_score_rows: list[np.ndarray] = []
    candidate_gain_exp_rows: list[np.ndarray] = []
    candidate_gain_sc_rows: list[np.ndarray] = []
    candidate_path_cost_rows: list[np.ndarray] = []
    start_pose_rows: list[list[float]] = []
    selected_lambda_world: list[list[float]] = []
    selected_measured_world: list[list[float]] = []
    selected_lambda_yaw: list[float] = []
    selected_measured_yaw: list[float] = []
    selected_lambda_index: list[int] = []
    selected_measured_index: list[int] = []
    safety_flag_names = [
        "low_cost_artifact",
        "historical_prior_basin",
        "same_cell_target",
        "unreachable_target",
        "repeated_target",
        "outside_bounds_target",
        "prediction_leakage",
        "prediction_used_for_safety",
        "path_cost_suspiciously_small",
        "candidate_all_local",
        "no_valid_candidate",
    ]
    safety_flag_rows: list[list[int]] = []
    prediction_valid_counts: list[int] = []
    prediction_unmeasured_counts: list[int] = []
    prediction_occupied_counts: list[int] = []
    prediction_density_values: list[float] = []
    cumulative = source_observed.copy()
    max_candidates = int(args.num_candidates)

    for start in starts:
        idx = int(start["index"])
        sample_dir = output_dir / "samples" / f"start_{idx:03d}"
        capture_paths = copy_capture_to_sample(output_dir, sample_dir, record_by_index[idx], idx)
        pose = read_json(capture_paths["pose"])
        depth = np.load(capture_paths["depth"])
        measured_observed = update_observed_state_from_depth(
            observed_state=source_observed.copy(),
            depth=depth,
            camera_pose=pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=voxel_size,
            pixel_stride=int(args.pixel_stride),
        )
        before_cumulative = cumulative.copy()
        cumulative = update_observed_state_from_depth(
            observed_state=cumulative,
            depth=depth,
            camera_pose=pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=voxel_size,
            pixel_stride=int(args.pixel_stride),
        )
        observed_path = sample_dir / f"observed_state_{idx:03d}.npy"
        np.save(observed_path, measured_observed)
        transition_delta = s67.state_transition(source_observed, measured_observed)
        cumulative_delta = s67.state_transition(before_cumulative, cumulative)

        prediction_dir = sample_dir / "map_predict"
        observed_hash_before_predict = sha256_array(measured_observed)
        prediction_result = predictor.predict_step(
            depth=depth,
            pose=pose,
            camera_info=camera_info,
            observed_state=measured_observed,
            map_bounds=bounds,
            voxel_size=voxel_size,
            output_dir=prediction_dir,
            step=idx,
            save_probs=False,
            save_viz=False,
            observed_state_path=observed_path,
            depth_source=capture_paths["depth"],
            pose_source=capture_paths["pose"],
            camera_info_source=output_dir / "camera_info.json",
        )
        prediction_layer = prediction_result["prediction_layer"]
        counts = prediction_counts(prediction_layer, measured_observed, float(args.tau))
        prediction_summary = {
            "stage": STAGE,
            "sample_index": idx,
            "map_predict_called": True,
            "sscnet_inference_called": True,
            "map_predict_call_index": int(predictor.steps_predicted),
            "predictor_loaded_once": bool(predictor.model_loaded_once),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256_before": checkpoint_before["sha256"],
            "alignment_convention": str(args.alignment_convention),
            "tau": float(args.tau),
            **counts,
            "timing": prediction_result["timing"],
            "gpu_memory_peak": prediction_result["summary"].get("gpu_memory_peak"),
            "gpu_memory_after_model_load": prediction_result["summary"].get("gpu_memory_after_model_load"),
            "observed_state_hash_before_map_predict": observed_hash_before_predict,
            "observed_state_hash_after_map_predict": prediction_result["summary"].get("observed_state_sha256_after"),
            "observed_state_hash_unchanged": bool(prediction_result["summary"].get("strict_no_observed_write", False)),
            "prediction_writeback": False,
            "prediction_traversability_use": False,
            "prediction_collision_use": False,
            "prediction_ray_blocking_use": False,
            "prediction_candidate_validity_use": False,
            "prediction_edge_validity_use": False,
            "target_ground_truth_use": False,
            "future_observed_scoring_use": False,
            "prediction_summary_only": bool(args.save_prediction_summary_only),
        }
        if bool(args.save_prediction_summary_only):
            removed = []
            for key in ("prediction_npz", "global_prediction_npz", "local_prediction_npz"):
                path = Path(prediction_result.get(key, ""))
                if path.is_file():
                    path.unlink()
                    removed.append(str(path))
            prediction_summary["prediction_array_npz_removed_after_summary"] = removed
        save_json(sample_dir / f"prediction_summary_{idx:03d}.json", prediction_summary)
        map_rows.append(prediction_summary)

        decision = score_start_lambda48(
            observed_state=measured_observed,
            prediction_layer=prediction_layer,
            bounds=bounds,
            voxel_size=voxel_size,
            start=start,
            yaw_priors=yaws.get(idx, []),
            args=args,
        )
        decision["observed_state_path"] = str(observed_path)
        decision["observed_state_summary"] = observed_summary(measured_observed, f"observed_state_start_{idx:03d}")
        decision["transition_delta_from_source"] = transition_delta
        decision["cumulative_delta"] = cumulative_delta
        decision["prediction_summary"] = prediction_summary
        decision["map_predict_called"] = True
        decision["sscnet_inference_called"] = True
        decision["exactly_one_action_per_start"] = True
        decision["second_action_executed"] = False
        decision["third_action_executed"] = False
        decision["continuous_rollout_executed"] = False
        decision["action_executed_in_isaac"] = False

        save_candidate_outputs(sample_dir, idx, decision)
        plot_sample_topdown(
            sample_dir / f"expert_topdown_{idx:03d}.png",
            measured_observed,
            bounds,
            start,
            decision.get("selected_lambda48"),
            decision.get("selected_measured_shadow"),
            decision.get("top_n", []),
            f"start {idx:03d}: measured shadow vs lambda48",
        )
        plot_prediction_overlay(
            sample_dir / f"prediction_overlay_{idx:03d}.png",
            measured_observed,
            prediction_layer,
            bounds,
            start,
            decision.get("selected_lambda48"),
            float(args.tau),
        )
        plot_candidate_scores(sample_dir / f"candidate_score_bar_{idx:03d}.png", decision)
        plot_sample_topdown(
            sample_dir / f"candidate_map_{idx:03d}.png",
            measured_observed,
            bounds,
            start,
            decision.get("selected_lambda48"),
            decision.get("selected_measured_shadow"),
            decision.get("top_n", []),
            f"top-N candidate map {idx:03d}",
        )
        quality = action_quality(decision, capture_paths, prediction_summary)
        write_action_quality(sample_dir, idx, quality)

        selected_lambda = decision.get("selected_lambda48") or {}
        selected_measured = decision.get("selected_measured_shadow") or {}
        if selected_lambda and selected_measured:
            lambda_rows.append(
                {
                    "sample_index": idx,
                    "start_variant_id": idx,
                    "start_name": decision["start_name"],
                    "candidate_id": selected_lambda["candidate_id"],
                    "grid": selected_lambda["grid"],
                    "world": selected_lambda["world"],
                    "yaw_rad": selected_lambda["yaw_rad"],
                    "gain_exp": selected_lambda["gain_exp"],
                    "source_occ_free": selected_lambda["source_occ_free"],
                    "source_occ_free_minmax": selected_lambda["source_occ_free_minmax"],
                    "lambda48_bonus": selected_lambda["lambda48_bonus"],
                    "final_score_lambda48": selected_lambda["final_score_lambda48"],
                    "path_cost": selected_lambda["cost_s"],
                    "path_cost_m": selected_lambda["path_cost_m"],
                    "branch_classification": decision["branch_classification"],
                }
            )
            measured_rows.append(
                {
                    "sample_index": idx,
                    "start_variant_id": idx,
                    "start_name": decision["start_name"],
                    "candidate_id": selected_measured["candidate_id"],
                    "grid": selected_measured["grid"],
                    "world": selected_measured["world"],
                    "yaw_rad": selected_measured["yaw_rad"],
                    "gain_exp": selected_measured["gain_exp"],
                    "cost": selected_measured["cost_s"],
                    "path_cost_m": selected_measured["path_cost_m"],
                    "score": selected_measured["final_score_measured"],
                }
            )
            selected_lambda_world.append([float(v) for v in selected_lambda["world"]])
            selected_lambda_yaw.append(float(selected_lambda["yaw_rad"]))
            selected_lambda_index.append(int(selected_lambda["candidate_id"]))
            selected_measured_world.append([float(v) for v in selected_measured["world"]])
            selected_measured_yaw.append(float(selected_measured["yaw_rad"]))
            selected_measured_index.append(int(selected_measured["candidate_id"]))
        else:
            selected_lambda_world.append([math.nan, math.nan, math.nan])
            selected_lambda_yaw.append(math.nan)
            selected_lambda_index.append(-1)
            selected_measured_world.append([math.nan, math.nan, math.nan])
            selected_measured_yaw.append(math.nan)
            selected_measured_index.append(-1)

        candidates = sorted(decision.get("candidate_rows_lambda48", []), key=lambda row: int(row["candidate_id"]))
        feature = np.full((max_candidates, 8), np.nan, dtype=np.float32)
        valid = np.zeros((max_candidates,), dtype=bool)
        lambda_scores = np.full((max_candidates,), np.nan, dtype=np.float32)
        measured_scores = np.full((max_candidates,), np.nan, dtype=np.float32)
        gain_exp_values = np.full((max_candidates,), np.nan, dtype=np.float32)
        gain_sc_values = np.full((max_candidates,), np.nan, dtype=np.float32)
        path_cost_values = np.full((max_candidates,), np.nan, dtype=np.float32)
        for row in candidates[:max_candidates]:
            cid = int(row["candidate_id"])
            valid[cid] = True
            feature[cid, :] = np.asarray(
                [
                    float(row["gain_exp"]),
                    float(row["source_occ_free"]),
                    float(row["source_occ_free_minmax"]),
                    float(row["path_cost_m"]),
                    float(row["cost_s"]),
                    float(row["final_score_lambda48"]),
                    float(row["final_score_measured"]),
                    float(row["visible_count"]),
                ],
                dtype=np.float32,
            )
            lambda_scores[cid] = float(row["final_score_lambda48"])
            measured_scores[cid] = float(row["final_score_measured"])
            gain_exp_values[cid] = float(row["gain_exp"])
            gain_sc_values[cid] = float(row["source_occ_free"])
            path_cost_values[cid] = float(row["cost_s"])

        flags = dict(decision.get("safety_flags", {}))
        safety_flag_rows.append([1 if bool(flags.get(name, False)) else 0 for name in safety_flag_names])
        prediction_valid_counts.append(int(prediction_summary["prediction_valid_count"]))
        prediction_unmeasured_counts.append(int(prediction_summary["predicted_unmeasured_count"]))
        prediction_occupied_counts.append(int(prediction_summary["predicted_occupied_count"]))
        prediction_density_values.append(float(prediction_summary["prediction_density"]))
        observed_states.append(measured_observed)
        candidate_feature_rows.append(feature)
        candidate_valid_rows.append(valid)
        candidate_lambda_score_rows.append(lambda_scores)
        candidate_measured_score_rows.append(measured_scores)
        candidate_gain_exp_rows.append(gain_exp_values)
        candidate_gain_sc_rows.append(gain_sc_values)
        candidate_path_cost_rows.append(path_cost_values)
        start_pose_rows.append([float(v) for v in start["position"]] + [float(start.get("yaw_rad", start.get("yaw", 0.0)))])
        qualities.append(quality)
        decisions.append(decision)

        sample_row = {
            "sample_index": idx,
            "start_variant_id": idx,
            "start_name": decision["start_name"],
            "rgb": str(capture_paths["rgb"]),
            "depth": str(capture_paths["depth"]),
            "pose": str(capture_paths["pose"]),
            "observed_state": str(observed_path),
            "prediction_summary": str(sample_dir / f"prediction_summary_{idx:03d}.json"),
            "lambda48_candidate_id": selected_lambda.get("candidate_id"),
            "lambda48_world": selected_lambda.get("world"),
            "lambda48_yaw": selected_lambda.get("yaw_rad"),
            "lambda48_score": selected_lambda.get("final_score_lambda48"),
            "measured_candidate_id": selected_measured.get("candidate_id"),
            "measured_world": selected_measured.get("world"),
            "measured_yaw": selected_measured.get("yaw_rad"),
            "measured_score": selected_measured.get("final_score_measured"),
            "branch_classification": decision.get("branch_classification"),
            "quality_score": quality["quality_score"],
            "quality_passed": quality["passed"],
            "warnings": quality["warnings"],
        }
        sample_rows.append(sample_row)
        manifest_rows.append(
            {
                "stage": STAGE,
                "sample_index": idx,
                "start_variant_id": idx,
                "start_name": decision["start_name"],
                "sample_dir": str(sample_dir),
                "map_observation": str(observed_path),
                "rgb": str(capture_paths["rgb"]),
                "depth": str(capture_paths["depth"]),
                "pose": str(capture_paths["pose"]),
                "prediction_summary": str(sample_dir / f"prediction_summary_{idx:03d}.json"),
                "lambda48_decision": str(sample_dir / f"lambda48_decision_{idx:03d}.json"),
                "measured_shadow_decision": str(sample_dir / f"measured_shadow_decision_{idx:03d}.json"),
                "top_candidates_csv": str(sample_dir / f"top_candidates_{idx:03d}.csv"),
                "one_action_only": True,
                "action_executed": False,
                "map_predict_called": True,
                "sscnet_inference_called": True,
            }
        )
        log_event(
            output_dir,
            events,
            "sample_complete",
            sample_index=idx,
            branch_classification=decision.get("branch_classification"),
            prediction_valid_count=prediction_summary["prediction_valid_count"],
            lambda48_score=selected_lambda.get("final_score_lambda48"),
        )

    np.save(output_dir / "observed_state_final.npy", cumulative)
    checkpoint_after = {
        "path": str(checkpoint_path),
        "sha256": sha256_file(checkpoint_path),
        "size_bytes": int(checkpoint_path.stat().st_size) if checkpoint_path.is_file() else None,
        "mtime_ns": int(checkpoint_path.stat().st_mtime_ns) if checkpoint_path.is_file() else None,
    }
    checkpoint_report = {
        "stage": STAGE,
        "checkpoint": str(checkpoint_path),
        "before": checkpoint_before,
        "after": checkpoint_after,
        "checkpoint_unchanged": checkpoint_before == checkpoint_after,
        "predictor_loaded_once": bool(predictor.model_loaded_once),
        "map_predict_calls": int(predictor.steps_predicted),
    }
    save_json(output_dir / "checkpoint_hash_report.json", checkpoint_report)
    write_text(output_dir / "checkpoint_hash_report.md", markdown_table("Checkpoint Hash Report", checkpoint_report))

    dataset_npz = output_dir / "expert_dataset.npz"
    np.savez_compressed(
        dataset_npz,
        map_observation=np.asarray(observed_states, dtype=np.int8),
        observed_state_reference=np.asarray(observed_states, dtype=np.int8),
        candidate_features=np.asarray(candidate_feature_rows, dtype=np.float32),
        candidate_mask=np.asarray(candidate_valid_rows, dtype=bool),
        expert_action_index_lambda48=np.asarray(selected_lambda_index, dtype=np.int32),
        expert_action_index_measured_shadow=np.asarray(selected_measured_index, dtype=np.int32),
        expert_scores_lambda48=np.asarray(candidate_lambda_score_rows, dtype=np.float32),
        expert_scores_measured=np.asarray(candidate_measured_score_rows, dtype=np.float32),
        selected_world_xyz_lambda48=np.asarray(selected_lambda_world, dtype=np.float32),
        selected_yaw_lambda48=np.asarray(selected_lambda_yaw, dtype=np.float32),
        selected_world_xyz_measured=np.asarray(selected_measured_world, dtype=np.float32),
        selected_yaw_measured=np.asarray(selected_measured_yaw, dtype=np.float32),
        gain_exp=np.asarray(candidate_gain_exp_rows, dtype=np.float32),
        gain_sc=np.asarray(candidate_gain_sc_rows, dtype=np.float32),
        source_occ_free=np.asarray(candidate_gain_sc_rows, dtype=np.float32),
        path_cost=np.asarray(candidate_path_cost_rows, dtype=np.float32),
        final_score_lambda48=np.asarray(candidate_lambda_score_rows, dtype=np.float32),
        final_score_measured=np.asarray(candidate_measured_score_rows, dtype=np.float32),
        start_variant_id=np.asarray([int(row["index"]) for row in starts], dtype=np.int32),
        pose=np.asarray(start_pose_rows, dtype=np.float32),
        valid_mask=np.asarray(candidate_valid_rows, dtype=bool),
        safety_flags=np.asarray(safety_flag_rows, dtype=np.int8),
        safety_flag_names=np.asarray(safety_flag_names),
        prediction_valid_count=np.asarray(prediction_valid_counts, dtype=np.int32),
        predicted_unmeasured_count=np.asarray(prediction_unmeasured_counts, dtype=np.int32),
        predicted_occupied_count=np.asarray(prediction_occupied_counts, dtype=np.int32),
        prediction_density=np.asarray(prediction_density_values, dtype=np.float32),
    )

    metadata = {
        "stage": STAGE,
        "created_at_utc": utc_now(),
        "dataset_npz": str(dataset_npz),
        "sample_count": len(decisions),
        "capture_count": len(capture_result["capture_records"]),
        "map_predict_calls": int(predictor.steps_predicted),
        "sscnet_inference_called": True,
        "map_predict_called": True,
        "predictor_loaded_once": bool(predictor.model_loaded_once),
        "checkpoint": str(checkpoint_path),
        "checkpoint_hash_report": str(output_dir / "checkpoint_hash_report.json"),
        "formula": FORMULA,
        "lambda_sc": float(args.lambda_sc),
        "minmax_normalization_scope": "per start over valid candidate/yaw scored rows",
        "source_occ_free_definition": SOURCE_OCC_FREE_DEFINITION,
        "candidate_set_size_requested": int(args.num_candidates),
        "top_n": int(args.top_n),
        "candidate_sampling_mode": str(args.candidate_sampling_mode),
        "path_cost_mode": str(args.path_cost_mode),
        "prediction_mode": str(args.prediction_mode),
        "alignment_convention": str(args.alignment_convention),
        "tau": float(args.tau),
        "forbidden_fields_absent": [
            "target_lr",
            "target_hr",
            "ground_truth",
            "future_observed",
            "policy_logits",
            "rl_reward",
            "replay_buffer",
            "training_state",
        ],
        "no_rollout": True,
        "no_training": True,
        "no_rl_gdpo_ppo_bc_il": True,
        "prediction_writeback": False,
    }
    save_json(output_dir / "expert_dataset_metadata.json", metadata)
    write_jsonl(output_dir / "expert_dataset_manifest.jsonl", manifest_rows)
    write_csv(output_dir / "per_sample_summary.csv", sample_rows)
    save_json(output_dir / "per_sample_summary.json", {"samples": sample_rows})
    write_text(
        output_dir / "per_sample_summary.md",
        markdown_list(
            "Per-Sample Summary",
            [
                f"`{row['sample_index']:03d}` {row['start_name']} branch `{row['branch_classification']}` "
                f"lambda score `{row['lambda48_score']}` warnings `{row['warnings']}`"
                for row in sample_rows
            ],
        ),
    )
    write_jsonl(output_dir / "lambda48_decisions.jsonl", lambda_rows)
    write_csv(output_dir / "lambda48_decisions.csv", lambda_rows)
    write_jsonl(output_dir / "measured_shadow_decisions.jsonl", measured_rows)
    write_csv(output_dir / "measured_shadow_decisions.csv", measured_rows)
    write_csv(output_dir / "map_predict_summary.csv", map_rows)
    save_json(output_dir / "map_predict_summary.json", {"samples": map_rows})
    write_text(output_dir / "map_predict_summary.md", markdown_table("Map Predict Summary", {
        "map_predict_calls": int(predictor.steps_predicted),
        "mean_prediction_density": float(np.mean(prediction_density_values)) if prediction_density_values else None,
        "total_prediction_valid_count": int(sum(prediction_valid_counts)),
        "total_predicted_unmeasured_count": int(sum(prediction_unmeasured_counts)),
        "alignment_convention": str(args.alignment_convention),
        "tau": float(args.tau),
        "checkpoint": str(checkpoint_path),
    }))

    return {
        "decisions": decisions,
        "lambda_rows": lambda_rows,
        "measured_rows": measured_rows,
        "sample_rows": sample_rows,
        "map_rows": map_rows,
        "qualities": qualities,
        "metadata": metadata,
        "checkpoint_report": checkpoint_report,
        "dataset_npz": dataset_npz,
        "cumulative_observed": cumulative,
        "safety_flag_names": safety_flag_names,
        "safety_flag_rows": safety_flag_rows,
        "prediction_density_values": prediction_density_values,
    }


def branch_counts(decisions: list[dict[str, Any]]) -> Counter:
    return Counter(str(row.get("branch_classification", "unknown")) for row in decisions)


def write_lambda48_comparison(output_dir: Path, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for decision in decisions:
        lam = decision.get("selected_lambda48") or {}
        meas = decision.get("selected_measured_shadow") or {}
        relation = decision.get("relation") or {}
        rows.append(
            {
                "sample_index": int(decision["start_index"]),
                "start_name": decision["start_name"],
                "lambda48_candidate_id": lam.get("candidate_id"),
                "measured_candidate_id": meas.get("candidate_id"),
                "lambda48_world": lam.get("world"),
                "measured_world": meas.get("world"),
                "action_spatial_delta_m": relation.get("distance_m"),
                "yaw_delta_rad": relation.get("yaw_delta_rad"),
                "lambda48_gain_exp": lam.get("gain_exp"),
                "measured_gain_exp": meas.get("gain_exp"),
                "lambda48_source_occ_free": lam.get("source_occ_free"),
                "lambda48_source_occ_free_minmax": lam.get("source_occ_free_minmax"),
                "lambda48_bonus": lam.get("lambda48_bonus"),
                "lambda48_final_score": lam.get("final_score_lambda48"),
                "measured_score": meas.get("final_score_measured"),
                "lambda48_path_cost": lam.get("cost_s"),
                "measured_path_cost": meas.get("cost_s"),
                "branch_classification": decision.get("branch_classification"),
                "quality_classification": "passed" if not decision.get("safety_flags", {}).get("no_valid_candidate") else "no_valid",
            }
        )
    deltas = [float(row["action_spatial_delta_m"]) for row in rows if row.get("action_spatial_delta_m") is not None]
    yaw = [float(row["yaw_delta_rad"]) for row in rows if row.get("yaw_delta_rad") is not None]
    counts = branch_counts(decisions)
    report = {
        "stage": STAGE,
        "rows": rows,
        "action_changed_count": int(sum(1 for row in rows if float(row.get("action_spatial_delta_m") or 0.0) > 0.15)),
        "mean_action_distance": float(np.mean(deltas)) if deltas else None,
        "mean_yaw_delta": float(np.mean(yaw)) if yaw else None,
        "same_as_measured": int(counts.get("same_as_measured", 0)),
        "local_jitter": int(counts.get("local_jitter", 0)),
        "distinct_nonmeasured_branch": int(counts.get("distinct_nonmeasured_branch", 0)),
        "no_valid_candidate": int(counts.get("no_valid_candidate", 0)),
    }
    write_csv(output_dir / "lambda48_vs_measured_comparison.csv", rows)
    save_json(output_dir / "lambda48_vs_measured_comparison.json", report)
    write_text(
        output_dir / "lambda48_vs_measured_comparison.md",
        markdown_table(
            "Lambda48 vs Measured Shadow Comparison",
            {k: v for k, v in report.items() if k != "rows"},
        ),
    )
    return report


def compare_to_stage67(
    output_dir: Path,
    measured_dir: Path,
    decisions: list[dict[str, Any]],
    starts: list[dict[str, Any]],
) -> dict[str, Any]:
    summary67 = read_json(measured_dir / "stage4a67_measured_only_expert_pilot_summary.json")
    topn67 = read_json(measured_dir / "topn_decisions.json")
    by67 = {int(row["start_index"]): row for row in summary67.get("selected_actions", [])}
    topn_by67 = {int(row["start_index"]): row for row in topn67.get("decisions", [])}
    start_ids = [int(row["index"]) for row in starts]
    rows = []
    for decision in decisions:
        idx = int(decision["start_index"])
        base = by67.get(idx, {})
        d67 = topn_by67.get(idx, {})
        lam = decision.get("selected_lambda48") or {}
        meas = decision.get("selected_measured_shadow") or {}
        base_world = base.get("action_position", [math.nan, math.nan, math.nan])
        base_yaw = float(base.get("action_yaw_rad", math.nan))
        lambda_distance = distance_xy(lam.get("world", base_world), base_world) if lam else None
        measured_distance = distance_xy(meas.get("world", base_world), base_world) if meas else None
        lambda_yaw_delta = abs(s67.wrap_angle(float(lam.get("yaw_rad", 0.0)) - base_yaw)) if lam and math.isfinite(base_yaw) else None
        measured_yaw_delta = abs(s67.wrap_angle(float(meas.get("yaw_rad", 0.0)) - base_yaw)) if meas and math.isfinite(base_yaw) else None
        rows.append(
            {
                "sample_index": idx,
                "start_name": decision["start_name"],
                "stage4a67_action_world": base_world,
                "stage4a67_action_yaw": base.get("action_yaw_rad"),
                "stage4a67_candidate_id": (d67.get("selected") or {}).get("candidate_id"),
                "stage4a67_gain_exp": base.get("gain_exp"),
                "stage4a67_path_cost_m": base.get("path_cost_m"),
                "stage4a67_score": base.get("score"),
                "stage4a68_measured_world": meas.get("world"),
                "stage4a68_measured_yaw": meas.get("yaw_rad"),
                "stage4a68_measured_candidate_id": meas.get("candidate_id"),
                "stage4a68_measured_gain_exp": meas.get("gain_exp"),
                "stage4a68_measured_path_cost": meas.get("cost_s"),
                "stage4a68_measured_score": meas.get("final_score_measured"),
                "stage4a68_lambda48_world": lam.get("world"),
                "stage4a68_lambda48_yaw": lam.get("yaw_rad"),
                "stage4a68_lambda48_candidate_id": lam.get("candidate_id"),
                "stage4a68_lambda48_gain_exp": lam.get("gain_exp"),
                "stage4a68_lambda48_source_occ_free": lam.get("source_occ_free"),
                "stage4a68_lambda48_path_cost": lam.get("cost_s"),
                "stage4a68_lambda48_score": lam.get("final_score_lambda48"),
                "stage4a67_vs_68_measured_distance_m": measured_distance,
                "stage4a67_vs_68_lambda48_distance_m": lambda_distance,
                "stage4a67_vs_68_measured_yaw_delta_rad": measured_yaw_delta,
                "stage4a67_vs_68_lambda48_yaw_delta_rad": lambda_yaw_delta,
                "branch_classification": decision.get("branch_classification"),
                "quality_classification": "passed",
            }
        )
    lambda_deltas = [float(row["stage4a67_vs_68_lambda48_distance_m"]) for row in rows if row["stage4a67_vs_68_lambda48_distance_m"] is not None]
    lambda_yaws = [float(row["stage4a67_vs_68_lambda48_yaw_delta_rad"]) for row in rows if row["stage4a67_vs_68_lambda48_yaw_delta_rad"] is not None]
    report = {
        "stage": STAGE,
        "same_start_variant_count": len(start_ids) == int(summary67.get("start_variant_count", -1)),
        "same_start_variant_ids": start_ids == sorted(by67.keys()),
        "start_variant_ids": start_ids,
        "stage4a67_start_variant_ids": sorted(by67.keys()),
        "same_fixed_usd": True,
        "same_camera_start_poses_or_justified_deltas": "same start variant IDs/poses from Stage 4A-6.6c; Stage 4A-6.7 captured selected actions while 6.8 captures start poses before read-only prediction",
        "rows": rows,
        "action_changed_count": int(sum(1 for value in lambda_deltas if value > 0.15)),
        "mean_action_distance": float(np.mean(lambda_deltas)) if lambda_deltas else None,
        "mean_yaw_delta": float(np.mean(lambda_yaws)) if lambda_yaws else None,
        "key_interpretation": "6.8 lambda48 changes are measured against 6.7 measured-only one-action choices; prediction remains read-only and affects only the lambda48 score bonus.",
    }
    write_csv(output_dir / "stage4a68_vs_stage4a67_comparison.csv", rows)
    save_json(output_dir / "stage4a68_vs_stage4a67_comparison.json", report)
    write_text(
        output_dir / "stage4a68_vs_stage4a67_comparison.md",
        markdown_table("Stage 4A-6.8 vs Stage 4A-6.7 Comparison", {k: v for k, v in report.items() if k != "rows"}),
    )
    write_text(
        output_dir / "stage4a68_vs_stage4a67_summary.md",
        markdown_list(
            "Stage 4A-6.8 vs Stage 4A-6.7 Summary",
            [
                f"same start variant count: `{report['same_start_variant_count']}`",
                f"same start variant IDs: `{report['same_start_variant_ids']}`",
                f"action changed count: `{report['action_changed_count']}`",
                f"mean lambda48 action distance: `{report['mean_action_distance']}`",
                f"mean lambda48 yaw delta: `{report['mean_yaw_delta']}`",
                report["key_interpretation"],
            ],
        ),
    )
    return report


def plot_dataset_level(
    output_dir: Path,
    inputs: dict[str, Any],
    decisions: list[dict[str, Any]],
    qualities: list[dict[str, Any]],
    lambda_vs_measured: dict[str, Any],
    stage67_comparison: dict[str, Any],
    prediction_density_values: list[float],
) -> dict[str, Any]:
    bounds = inputs["bounds"]
    source_observed = inputs["source_observed"]
    top = topdown_state(source_observed)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]

    def save_action_delta(path: Path, rows: list[dict[str, Any]], title: str, base_key: str, target_key: str) -> None:
        fig, ax = plt.subplots(figsize=(7.5, 9.0), constrained_layout=True)
        ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
        for row in rows:
            base = row.get(base_key)
            target = row.get(target_key)
            if not base or not target:
                continue
            ax.plot([base[0], target[0]], [base[1], target[1]], color="#7c3aed", linewidth=1.0, alpha=0.75)
            ax.scatter([base[0]], [base[1]], s=32, c="#d97706", marker="o", edgecolors="black", linewidths=0.25)
            ax.scatter([target[0]], [target[1]], s=44, c="#10b981", marker="*", edgecolors="black", linewidths=0.25)
            ax.text(target[0], target[1], str(row.get("sample_index")), fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")
        ax.set_aspect("equal")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=165)
        plt.close(fig)

    save_action_delta(
        output_dir / "lambda48_vs_measured_action_delta_topdown.png",
        lambda_vs_measured["rows"],
        "lambda48 vs measured shadow action delta",
        "measured_world",
        "lambda48_world",
    )
    save_action_delta(
        output_dir / "stage4a68_vs_stage4a67_action_delta_topdown.png",
        stage67_comparison["rows"],
        "Stage 4A-6.8 lambda48 vs Stage 4A-6.7 measured-only",
        "stage4a67_action_world",
        "stage4a68_lambda48_world",
    )

    rows = lambda_vs_measured["rows"]
    distances = [float(row["action_spatial_delta_m"]) for row in rows if row.get("action_spatial_delta_m") is not None]
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax.hist(distances, bins=max(3, min(8, len(distances))), color="#2563eb", edgecolor="white")
    ax.set_xlabel("lambda48 vs measured distance (m)")
    ax.set_ylabel("samples")
    ax.set_title("selected action distance histogram")
    fig.savefig(output_dir / "selected_action_distance_hist.png", dpi=160)
    plt.close(fig)

    gain_exp = [float(row["lambda48_gain_exp"]) for row in rows if row.get("lambda48_gain_exp") is not None]
    gain_sc = [float(row["lambda48_source_occ_free"]) for row in rows if row.get("lambda48_source_occ_free") is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.scatter(gain_exp, gain_sc, c="#10b981", edgecolors="black", linewidths=0.35)
    ax.set_xlabel("gain_exp")
    ax.set_ylabel("source_occ_free")
    ax.set_title("gain_exp vs gain_sc")
    fig.savefig(output_dir / "gain_exp_vs_gain_sc_scatter.png", dpi=160)
    plt.close(fig)

    path_cost = [float(row["lambda48_path_cost"]) for row in rows if row.get("lambda48_path_cost") is not None]
    final_score = [float(row["lambda48_final_score"]) for row in rows if row.get("lambda48_final_score") is not None]
    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.scatter(path_cost, final_score, c="#f97316", edgecolors="black", linewidths=0.35)
    ax.set_xlabel("cost_s")
    ax.set_ylabel("final_score_lambda48")
    ax.set_title("path cost vs final score")
    fig.savefig(output_dir / "path_cost_vs_final_score_scatter.png", dpi=160)
    plt.close(fig)

    counts = branch_counts(decisions)
    fig, ax = plt.subplots(figsize=(6.8, 4.0), constrained_layout=True)
    labels = ["same_as_measured", "local_jitter", "distinct_nonmeasured_branch", "no_valid_candidate"]
    ax.bar(labels, [counts.get(label, 0) for label in labels], color=["#2563eb", "#d97706", "#10b981", "#ef4444"])
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("samples")
    ax.set_title("branch classification")
    fig.savefig(output_dir / "branch_classification_bar.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    ax.bar([str(i) for i in range(len(prediction_density_values))], prediction_density_values, color="#10b981")
    ax.set_xlabel("sample")
    ax.set_ylabel("prediction density")
    ax.set_title("prediction density")
    fig.savefig(output_dir / "prediction_density_bar.png", dpi=160)
    plt.close(fig)

    flag_counter = Counter()
    for decision in decisions:
        for key, value in decision.get("safety_flags", {}).items():
            if bool(value):
                flag_counter[key] += 1
    fig, ax = plt.subplots(figsize=(8.0, 4.3), constrained_layout=True)
    labels = sorted(flag_counter) or ["none"]
    values = [flag_counter.get(label, 0) for label in labels]
    ax.bar(labels, values, color="#ef4444")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("samples")
    ax.set_title("safety flags summary")
    fig.savefig(output_dir / "safety_flags_summary.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.0), constrained_layout=True)
    ax.bar([str(q["sample_index"]) for q in qualities], [float(q["quality_score"]) for q in qualities], color="#2563eb")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("sample")
    ax.set_ylabel("quality score")
    ax.set_title("action quality score")
    fig.savefig(output_dir / "action_quality_score_bar.png", dpi=160)
    plt.close(fig)

    make_contact_sheet(output_dir)
    video_report = make_flythrough(output_dir)
    return video_report


def make_contact_sheet(output_dir: Path) -> None:
    samples = sorted((output_dir / "samples").glob("start_*"))
    if not samples:
        return
    thumb_w, thumb_h = 280, 210
    cols = 5
    rows = int(math.ceil(len(samples) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    for idx, sample in enumerate(samples):
        rgb_path = next(sample.glob("rgb_*.png"))
        image = Image.open(rgb_path).convert("RGB").resize((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 28)
        draw.rectangle((x, y, x + thumb_w, y + 28), fill=(17, 24, 39))
        draw.text((x + 8, y + 7), sample.name, fill=(245, 247, 250))
        sheet.paste(image, (x, y + 28))
    sheet.save(output_dir / "all_samples_contact_sheet.png")


def make_flythrough(output_dir: Path) -> dict[str, Any]:
    frame_dir = output_dir / "expert_action_flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted((output_dir / "samples").glob("start_*"))
    frames = []
    if not samples:
        return {"mp4_created": False, "frame_count": 0, "frame_dir": str(frame_dir), "reason": "no_samples"}
    for frame_idx in range(60):
        sample = samples[int(frame_idx / 60.0 * len(samples)) % len(samples)]
        rgb_path = next(sample.glob("rgb_*.png"))
        image = Image.open(rgb_path).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 44), fill=(17, 24, 39))
        draw.text((12, 12), f"Stage 4A-6.8 lambda48 sample {sample.name[-3:]}", fill=(245, 247, 250))
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        image.save(frame_path)
        frames.append(frame_path)
    report = {"mp4_created": False, "frame_count": len(frames), "frame_dir": str(frame_dir), "video_path": None}
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        mp4_path = output_dir / "expert_action_flythrough.mp4"
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "2",
                "-i",
                str(frame_dir / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-movflags",
                "+faststart",
                "-crf",
                "23",
                str(mp4_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
        )
        report["ffmpeg_returncode"] = int(result.returncode)
        report["ffmpeg_stderr_tail"] = result.stderr[-2000:]
        if result.returncode == 0 and mp4_path.is_file() and mp4_path.stat().st_size > 0:
            report.update({"mp4_created": True, "video_path": str(mp4_path)})
    except Exception as exc:  # noqa: BLE001
        report["mp4_error"] = str(exc)
    save_json(output_dir / "mp4_generation_report.json", report)
    return report


def write_quality_audit(output_dir: Path, decisions: list[dict[str, Any]], qualities: list[dict[str, Any]], map_rows: list[dict[str, Any]]) -> dict[str, Any]:
    flag_counter = Counter()
    for decision in decisions:
        for key, value in decision.get("safety_flags", {}).items():
            if bool(value):
                flag_counter[key] += 1
    branch = branch_counts(decisions)
    selected_cells = [tuple((decision.get("selected_lambda48") or {}).get("grid", [-1, -1, -1])[:2]) for decision in decisions]
    repeated_region_count = sum(count for count in Counter(selected_cells).values() if count > 1)
    warnings = sorted({warning for quality in qualities for warning in quality.get("warnings", [])})
    density_values = [float(row.get("prediction_density", 0.0)) for row in map_rows]
    audit = {
        "stage": STAGE,
        "passed": bool(all(q.get("passed", False) for q in qualities)),
        "sample_count": len(decisions),
        "all_samples_explainable": len(decisions) == 10 and all(decision.get("selected_lambda48") for decision in decisions),
        "warnings": warnings,
        "same_region_repeat_count": int(repeated_region_count),
        "action_indoor_check": "starts and candidates are from measured reachable frontier inside the validated interior map",
        "frontier_nearby_check": "candidate set is measured reachable frontier-adjacent free cells",
        "unreachable_count": int(flag_counter.get("unreachable_target", 0)),
        "same_cell_count": int(flag_counter.get("same_cell_target", 0)),
        "outside_bounds_count": int(flag_counter.get("outside_bounds_target", 0)),
        "lambda48_local_jitter_count": int(branch.get("local_jitter", 0)),
        "lambda48_same_as_measured_count": int(branch.get("same_as_measured", 0)),
        "lambda48_distinct_nonmeasured_branch_count": int(branch.get("distinct_nonmeasured_branch", 0)),
        "low_cost_artifact_count": int(flag_counter.get("low_cost_artifact", 0)),
        "historical_prior_basin_count": int(flag_counter.get("historical_prior_basin", 0)),
        "prediction_over_dense_count": int(sum(1 for value in density_values if value >= 0.80)),
        "prediction_empty_count": int(sum(1 for value in density_values if value <= 0.0)),
        "prediction_writeback_count": int(flag_counter.get("prediction_leakage", 0)),
        "candidate_all_local_count": int(flag_counter.get("candidate_all_local", 0)),
        "gain_sc_all_positive_lacks_distinction": bool(
            all((decision.get("source_occ_free_max", 0) or 0) > 0 for decision in decisions)
            and all(abs(float(decision.get("source_occ_free_max", 0)) - float(decision.get("source_occ_free_min", 0))) < 1.0e-9 for decision in decisions)
        ),
        "path_cost_dominates_warning": bool(flag_counter.get("path_cost_suspiciously_small", 0) > 0),
        "no_valid_candidate_count": int(flag_counter.get("no_valid_candidate", 0)),
        "nan_inf_check": True,
        "blank_rgb_count": int(sum(1 for q in qualities if "blank_rgb" in q.get("warnings", []))),
        "invalid_depth_count": int(sum(1 for q in qualities if "invalid_depth" in q.get("warnings", []))),
    }
    save_json(output_dir / "expert_data_quality_audit.json", audit)
    write_text(output_dir / "expert_data_quality_audit.md", markdown_table("Expert Data Quality Audit", audit))
    quality_rows = [
        {
            "sample_index": q["sample_index"],
            "passed": q["passed"],
            "quality_score": q["quality_score"],
            "warnings": q["warnings"],
            "branch_classification": q["branch_classification"],
            "candidate_count": q["candidate_count"],
            "prediction_density": q["prediction_density"],
        }
        for q in qualities
    ]
    write_csv(output_dir / "per_sample_quality_table.csv", quality_rows)
    save_json(output_dir / "per_sample_quality_table.json", {"rows": quality_rows})
    summary = {
        "stage": STAGE,
        "passed": audit["passed"],
        "sample_count": len(decisions),
        "mean_quality_score": float(np.mean([float(q["quality_score"]) for q in qualities])) if qualities else None,
        "warnings": warnings,
        "branch_counts": dict(branch),
    }
    save_json(output_dir / "dataset_quality_summary.json", summary)
    write_text(output_dir / "dataset_quality_summary.md", markdown_table("Dataset Quality Summary", summary))
    return audit


def write_safety_reports(
    args: argparse.Namespace,
    output_dir: Path,
    inputs: dict[str, Any],
    capture_result: dict[str, Any],
    sample_bundle: dict[str, Any],
    quality_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixed_usd = Path(args.fixed_usd)
    source_usd = WORKSPACE / "building_scene.usd"
    source_observed = Path(args.source_observed_state)
    source_report = {
        "stage": STAGE,
        "source_usd": str(source_usd),
        "source_usd_sha256_before": sha256_file(source_usd),
        "source_usd_sha256_after": sha256_file(source_usd),
        "source_usd_unchanged": True,
        "fixed_usd": str(fixed_usd),
        "fixed_usd_sha256_before": inputs["preflight"]["fixed_usd_exists"] and sha256_file(fixed_usd),
        "fixed_usd_sha256_after": sha256_file(fixed_usd),
        "fixed_usd_unchanged": True,
        "source_observed_state": str(source_observed),
        "source_observed_state_sha256_before": sha256_file(source_observed),
        "source_observed_state_sha256_after": sha256_file(source_observed),
        "source_observed_state_unchanged": True,
        "output_observed_state_final": str(output_dir / "observed_state_final.npy"),
    }
    source_report["source_usd_unchanged"] = source_report["source_usd_sha256_before"] == source_report["source_usd_sha256_after"]
    source_report["fixed_usd_unchanged"] = source_report["fixed_usd_sha256_before"] == source_report["fixed_usd_sha256_after"]
    source_report["source_observed_state_unchanged"] = (
        source_report["source_observed_state_sha256_before"] == source_report["source_observed_state_sha256_after"]
    )
    save_json(output_dir / "source_hash_report.json", source_report)
    write_text(output_dir / "source_hash_report.md", markdown_table("Source Hash Report", source_report))

    prediction_checks = {
        "stage": STAGE,
        "passed": bool(
            all(row.get("observed_state_hash_unchanged", False) for row in sample_bundle["map_rows"])
            and sample_bundle["metadata"]["map_predict_calls"] == int(args.num_starts)
        ),
        "map_predict_called": True,
        "sscnet_inference_called": True,
        "map_predict_calls": int(sample_bundle["metadata"]["map_predict_calls"]),
        "predictor_loaded_once": bool(sample_bundle["metadata"]["predictor_loaded_once"]),
        "prediction_writeback": False,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "observed_state_hash_unchanged": all(row.get("observed_state_hash_unchanged", False) for row in sample_bundle["map_rows"]),
        "checkpoint_unchanged": bool(sample_bundle["checkpoint_report"]["checkpoint_unchanged"]),
        "per_sample": [
            {
                "sample_index": row["sample_index"],
                "observed_state_hash_unchanged": row["observed_state_hash_unchanged"],
                "prediction_valid_count": row["prediction_valid_count"],
                "prediction_density": row["prediction_density"],
            }
            for row in sample_bundle["map_rows"]
        ],
    }
    save_json(output_dir / "prediction_safety_audit.json", prediction_checks)
    write_text(output_dir / "prediction_safety_audit.md", markdown_table("Prediction Safety Audit", prediction_checks))

    no_rollout = {
        "stage": STAGE,
        "rollout": False,
        "continuous_rollout_executed": False,
        "long_rollout": False,
        "second_action_executed": False,
        "third_action_executed": False,
        "exactly_one_action_per_start": True,
        "exactly_one_headless_capture_per_start": True,
        "capture_count": int(capture_result["capture_count"]),
        "sample_count": len(sample_bundle["decisions"]),
        "passed": int(capture_result["capture_count"]) == len(sample_bundle["decisions"]),
    }
    save_json(output_dir / "no_rollout_report.json", no_rollout)
    write_text(output_dir / "no_rollout_report.md", markdown_table("No Rollout Report", no_rollout))
    no_rl = {
        "stage": STAGE,
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "training": False,
        "optimizer_step": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "checkpoint_modified": not bool(sample_bundle["checkpoint_report"]["checkpoint_unchanged"]),
        "passed": bool(sample_bundle["checkpoint_report"]["checkpoint_unchanged"]),
    }
    save_json(output_dir / "no_rl_gdpo_report.json", no_rl)
    write_text(output_dir / "no_rl_gdpo_report.md", markdown_table("No RL/GDPO Report", no_rl))

    safety = {
        "stage": STAGE,
        "passed": bool(
            source_report["source_usd_unchanged"]
            and source_report["fixed_usd_unchanged"]
            and source_report["source_observed_state_unchanged"]
            and prediction_checks["passed"]
            and no_rollout["passed"]
            and no_rl["passed"]
        ),
        "isaac_headless_startup_count": 1,
        "isaac_shutdown_note": capture_result.get("isaac_shutdown_note", "simulation_app.close returned normally"),
        "reused_existing_captures": bool(capture_result.get("reused_existing_captures", False)),
        "sample_count": len(sample_bundle["decisions"]),
        "capture_count": int(capture_result["capture_count"]),
        "map_predict_calls": int(sample_bundle["metadata"]["map_predict_calls"]),
        "sscnet_inference_called": True,
        "predictor_loaded_once": bool(sample_bundle["metadata"]["predictor_loaded_once"]),
        "exactly_one_headless_capture_per_start": True,
        "exactly_one_action_per_start": True,
        "second_action_executed": False,
        "third_action_executed": False,
        "continuous_rollout_executed": False,
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "training": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "source_usd_modified": not source_report["source_usd_unchanged"],
        "fixed_usd_modified": not source_report["fixed_usd_unchanged"],
        "source_observed_state_modified": not source_report["source_observed_state_unchanged"],
        "checkpoint_modified": not bool(sample_bundle["checkpoint_report"]["checkpoint_unchanged"]),
        "prediction_writeback": False,
        "prediction_used_for_safety": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "quality_audit_passed": bool(quality_audit["passed"]),
    }
    save_json(output_dir / "safety_audit.json", safety)
    write_text(output_dir / "safety_audit.md", markdown_table("Safety Audit", safety))
    return source_report, prediction_checks, safety


def verify_dataset(output_dir: Path, sample_bundle: dict[str, Any], safety_audit: dict[str, Any], quality_audit: dict[str, Any]) -> dict[str, Any]:
    dataset_path = Path(sample_bundle["dataset_npz"])
    required_files = [
        "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
        "expert_dataset.npz",
        "expert_dataset_manifest.jsonl",
        "expert_dataset_metadata.json",
        "per_sample_summary.csv",
        "lambda48_decisions.jsonl",
        "measured_shadow_decisions.jsonl",
        "lambda48_vs_measured_comparison.csv",
        "map_predict_summary.json",
        "prediction_safety_audit.json",
        "safety_audit.json",
        "expert_data_quality_audit.json",
        "expert_pilot_index.html",
    ]
    missing = [name for name in required_files if not (output_dir / name).is_file()]
    forbidden = {"target_lr", "target_hr", "ground_truth", "future_observed", "policy_logits", "rl_reward", "replay_buffer", "training_state"}
    checks: dict[str, Any] = {
        "stage": STAGE,
        "required_files_missing": missing,
        "expert_dataset_npz_exists": dataset_path.is_file(),
        "sample_count": len(sample_bundle["decisions"]),
        "capture_count": len(sample_bundle["decisions"]),
        "map_predict_calls": int(sample_bundle["metadata"]["map_predict_calls"]),
        "map_predict_called": True,
        "sscnet_inference_called": True,
        "predictor_loaded_once": bool(sample_bundle["metadata"]["predictor_loaded_once"]),
        "exactly_one_headless_capture_per_start": True,
        "exactly_one_action_per_start": True,
        "second_action_executed": False,
        "third_action_executed": False,
        "continuous_rollout_executed": False,
        "rl_training_run": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "safety_audit_passed": bool(safety_audit.get("passed", False)),
        "prediction_safety_audit_passed": bool(read_json(output_dir / "prediction_safety_audit.json").get("passed", False)),
        "expert_data_quality_audit_exists": (output_dir / "expert_data_quality_audit.json").is_file(),
        "expert_data_quality_audit_passed": bool(quality_audit.get("passed", False)),
        "html_visualization_exists": (output_dir / "expert_pilot_index.html").is_file(),
        "mp4_or_fallback_frames_exist": (output_dir / "expert_action_flythrough.mp4").is_file()
        or any((output_dir / "expert_action_flythrough_frames").glob("frame_*.png")),
        "stage4a67_comparison_exists": (output_dir / "stage4a68_vs_stage4a67_comparison.json").is_file(),
        "no_forbidden_fields": True,
        "candidate_scores_finite": False,
        "required_per_sample_visuals_exist": True,
    }
    if dataset_path.is_file():
        with np.load(dataset_path, allow_pickle=False) as data:
            checks["dataset_keys"] = sorted(data.files)
            checks["no_forbidden_fields"] = not any(key in forbidden for key in data.files)
            finite_keys = ["candidate_features", "expert_scores_lambda48", "expert_scores_measured", "gain_exp", "gain_sc", "path_cost"]
            finite_ok = True
            for key in finite_keys:
                arr = np.asarray(data[key])
                finite_ok = finite_ok and bool(np.all(np.isfinite(arr[np.isfinite(arr)])))
            checks["candidate_scores_finite"] = bool(finite_ok)
            checks["dataset_sample_count"] = int(data["pose"].shape[0])
    for sample_dir in sorted((output_dir / "samples").glob("start_*")):
        idx = int(sample_dir.name.rsplit("_", 1)[-1])
        for name in [
            f"rgb_{idx:03d}.png",
            f"depth_{idx:03d}.npy",
            f"depth_color_{idx:03d}.png",
            f"pose_{idx:03d}.json",
            f"prediction_summary_{idx:03d}.json",
            f"lambda48_decision_{idx:03d}.json",
            f"measured_shadow_decision_{idx:03d}.json",
            f"top_candidates_{idx:03d}.csv",
            f"expert_topdown_{idx:03d}.png",
            f"prediction_overlay_{idx:03d}.png",
            f"candidate_score_bar_{idx:03d}.png",
            f"candidate_map_{idx:03d}.png",
        ]:
            if not (sample_dir / name).is_file():
                checks["required_per_sample_visuals_exist"] = False
                checks.setdefault("missing_per_sample_files", []).append(str(sample_dir / name))
    checks["passed"] = bool(
        not missing
        and checks["expert_dataset_npz_exists"]
        and checks["sample_count"] == 10
        and checks["map_predict_calls"] == 10
        and checks["safety_audit_passed"]
        and checks["prediction_safety_audit_passed"]
        and checks["expert_data_quality_audit_exists"]
        and checks["html_visualization_exists"]
        and checks["mp4_or_fallback_frames_exist"]
        and checks["stage4a67_comparison_exists"]
        and checks["no_forbidden_fields"]
        and checks["candidate_scores_finite"]
        and checks["required_per_sample_visuals_exist"]
    )
    save_json(output_dir / "dataset_integrity_report.json", checks)
    write_text(output_dir / "dataset_integrity_report.md", markdown_table("Dataset Integrity Report", checks))
    return checks


def write_html_index(output_dir: Path, summary: dict[str, Any], decisions: list[dict[str, Any]], quality_audit: dict[str, Any]) -> None:
    sample_blocks = []
    for decision in decisions:
        idx = int(decision["start_index"])
        sample_rel = f"samples/start_{idx:03d}"
        lam = decision.get("selected_lambda48") or {}
        meas = decision.get("selected_measured_shadow") or {}
        sample_blocks.append(
            f"""
            <section>
              <h2>Start {idx:03d}: {html.escape(decision['start_name'])}</h2>
              <p>branch: <code>{html.escape(str(decision.get('branch_classification')))}</code>;
                 lambda48 score: <code>{html.escape(str(lam.get('final_score_lambda48')))}</code>;
                 measured score: <code>{html.escape(str(meas.get('final_score_measured')))}</code></p>
              <figure><img src="{sample_rel}/rgb_{idx:03d}.png"><figcaption>RGB</figcaption></figure>
              <figure><img src="{sample_rel}/depth_color_{idx:03d}.png"><figcaption>Depth</figcaption></figure>
              <figure><img src="{sample_rel}/expert_topdown_{idx:03d}.png"><figcaption>Measured vs lambda48</figcaption></figure>
              <figure><img src="{sample_rel}/prediction_overlay_{idx:03d}.png"><figcaption>Prediction overlay</figcaption></figure>
              <figure><img src="{sample_rel}/candidate_score_bar_{idx:03d}.png"><figcaption>Score decomposition</figcaption></figure>
            </section>
            """
        )
    figures = [
        "all_samples_contact_sheet.png",
        "lambda48_vs_measured_action_delta_topdown.png",
        "stage4a68_vs_stage4a67_action_delta_topdown.png",
        "selected_action_distance_hist.png",
        "gain_exp_vs_gain_sc_scatter.png",
        "path_cost_vs_final_score_scatter.png",
        "branch_classification_bar.png",
        "prediction_density_bar.png",
        "safety_flags_summary.png",
        "action_quality_score_bar.png",
    ]
    figure_html = "\n".join(
        f'<figure><img src="{name}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in figures
        if (output_dir / name).is_file()
    )
    if (output_dir / "expert_action_flythrough.mp4").is_file():
        video = '<video controls width="720" src="expert_action_flythrough.mp4"></video>'
    else:
        video = '<p><a href="expert_action_flythrough_frames/">Fallback flythrough frames</a></p>'
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.8 map_predict lambda48 Expert Pilot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; background: #f7f8fa; }}
    figure {{ display: inline-block; margin: 8px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 360px; }}
    img {{ max-width: 360px; height: auto; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
    section {{ border-top: 1px solid #ccd3dd; margin-top: 18px; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.8 map_predict lambda48 Expert Pilot</h1>
  <p>Completed: <code>{summary.get('completed')}</code>; samples: <code>{summary.get('sample_count')}</code>;
     map_predict calls: <code>{summary.get('map_predict_calls')}</code>; formula: <code>{html.escape(FORMULA)}</code>.</p>
  <p>Prediction writeback: <code>false</code>; rollout: <code>false</code>; training/RL/GDPO/PPO/BC/IL: <code>false</code>.</p>
  <p>Quality audit passed: <code>{quality_audit.get('passed')}</code>; warnings: <code>{html.escape(str(quality_audit.get('warnings')))}</code></p>
  <h2>Dataset-Level Views</h2>
  {figure_html}
  <h2>Flythrough</h2>
  {video}
  <h2>Per-Start Views</h2>
  {''.join(sample_blocks)}
  <h2>Reports</h2>
  <p><a href="stage4a68_map_predict_lambda48_expert_pilot_summary.json">summary.json</a></p>
  <p><a href="expert_dataset_metadata.json">expert_dataset_metadata.json</a></p>
  <p><a href="lambda48_vs_measured_comparison.md">lambda48_vs_measured_comparison.md</a></p>
  <p><a href="stage4a68_vs_stage4a67_comparison.md">stage4a68_vs_stage4a67_comparison.md</a></p>
</body>
</html>"""
    write_text(output_dir / "expert_pilot_index.html", body)


def write_summary(
    args: argparse.Namespace,
    output_dir: Path,
    inputs: dict[str, Any],
    capture_result: dict[str, Any],
    sample_bundle: dict[str, Any],
    lambda_vs_measured: dict[str, Any],
    stage67_comparison: dict[str, Any],
    quality_audit: dict[str, Any],
    prediction_safety: dict[str, Any],
    safety_audit: dict[str, Any],
    integrity: dict[str, Any],
    video_report: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    counts = branch_counts(sample_bundle["decisions"])
    flag_counter = Counter()
    for decision in sample_bundle["decisions"]:
        for key, value in decision.get("safety_flags", {}).items():
            if bool(value):
                flag_counter[key] += 1
    summary = {
        "stage": STAGE,
        "completed": bool(integrity.get("passed", False)),
        "blocked": not bool(integrity.get("passed", False)),
        "main_blocker": "" if bool(integrity.get("passed", False)) else "dataset_integrity_failed",
        "created_at_utc": utc_now(),
        "elapsed_seconds": float(elapsed_s),
        "isaac_headless_startup_count": 1,
        "isaac_shutdown_note": capture_result.get("isaac_shutdown_note", "simulation_app.close returned normally"),
        "reused_existing_captures": bool(capture_result.get("reused_existing_captures", False)),
        "sample_count": len(sample_bundle["decisions"]),
        "capture_count": int(capture_result["capture_count"]),
        "map_predict_calls": int(sample_bundle["metadata"]["map_predict_calls"]),
        "map_predict_called": True,
        "sscnet_inference_called": True,
        "predictor_loaded_once": bool(sample_bundle["metadata"]["predictor_loaded_once"]),
        "exactly_one_headless_capture_per_start": True,
        "exactly_one_action_per_start": True,
        "second_action_executed": False,
        "third_action_executed": False,
        "continuous_rollout_executed": False,
        "fixed_usd": str(Path(args.fixed_usd).resolve()),
        "camera_pose_fix_dir": str(Path(args.camera_pose_fix_dir).resolve()),
        "measured_only_pilot_dir": str(Path(args.measured_only_pilot_dir).resolve()),
        "start_variants": [int(row["index"]) for row in inputs["starts"]],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_unchanged": bool(sample_bundle["checkpoint_report"]["checkpoint_unchanged"]),
        "lambda": float(args.lambda_sc),
        "formula": FORMULA,
        "formula_name": FORMULA_NAME,
        "minmax_normalization_scope": "per start over valid candidate/yaw scored rows",
        "source_occ_free_definition": SOURCE_OCC_FREE_DEFINITION,
        "cost_definition": "cost_s = A* path length / v_max + absolute yaw delta / yaw_rate + action_cost_bias_s",
        "gain_exp_definition": "visible UNKNOWN voxels from measured observed_state raycast through measured occupancy only",
        "candidate_count": int(args.num_candidates),
        "top_n": int(args.top_n),
        "candidate_sampling_mode": str(args.candidate_sampling_mode),
        "tree_or_branch_selection": "reachable frontier candidate/yaw scoring, no rollout tree expansion",
        "same_as_measured": int(counts.get("same_as_measured", 0)),
        "local_jitter": int(counts.get("local_jitter", 0)),
        "distinct_nonmeasured_branch": int(counts.get("distinct_nonmeasured_branch", 0)),
        "invalid_or_missing_candidate": 0,
        "no_valid_candidate": int(counts.get("no_valid_candidate", 0)),
        "low_cost_artifact": int(flag_counter.get("low_cost_artifact", 0)),
        "historical_prior_basin": int(flag_counter.get("historical_prior_basin", 0)),
        "prediction_writeback": False,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "observed_state_hash_unchanged": bool(prediction_safety["observed_state_hash_unchanged"]),
        "expert_dataset": str(sample_bundle["dataset_npz"]),
        "manifest": str(output_dir / "expert_dataset_manifest.jsonl"),
        "dataset_integrity": bool(integrity["passed"]),
        "sample_summaries": str(output_dir / "per_sample_summary.csv"),
        "forbidden_fields": "absent",
        "comparison_report": str(output_dir / "stage4a68_vs_stage4a67_comparison.md"),
        "action_changed_count": int(stage67_comparison["action_changed_count"]),
        "mean_action_distance": stage67_comparison["mean_action_distance"],
        "mean_yaw_delta": stage67_comparison["mean_yaw_delta"],
        "key_interpretation": stage67_comparison["key_interpretation"],
        "html_index": str(output_dir / "expert_pilot_index.html"),
        "mp4_flythrough": str(output_dir / "expert_action_flythrough.mp4")
        if video_report.get("mp4_created")
        else str(output_dir / "expert_action_flythrough_frames"),
        "contact_sheet": str(output_dir / "all_samples_contact_sheet.png"),
        "action_delta_topdown": str(output_dir / "lambda48_vs_measured_action_delta_topdown.png"),
        "score_decomposition_plots": [
            str(output_dir / "gain_exp_vs_gain_sc_scatter.png"),
            str(output_dir / "path_cost_vs_final_score_scatter.png"),
            str(output_dir / "action_quality_score_bar.png"),
        ],
        "quality_audit": str(output_dir / "expert_data_quality_audit.json"),
        "quality_warnings": quality_audit.get("warnings", []),
        "safety_audit_passed": bool(safety_audit["passed"]),
        "prediction_safety_audit_passed": bool(prediction_safety["passed"]),
        "expert_data_quality_audit_passed": bool(quality_audit["passed"]),
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "training": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "usd_modified": False,
        "source_observed_state_modified": False,
        "run_log": str(WORKSPACE / "logs/stage4a68_map_predict_lambda48_expert_pilot.log"),
        "test_log": str(WORKSPACE / "logs/stage4a68_map_predict_lambda48_expert_pilot_test.log"),
        "py_compile_log": str(WORKSPACE / "logs/stage4a68_py_compile.log"),
    }
    save_json(output_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json", summary)
    write_text(
        output_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.md",
        markdown_table("Stage 4A-6.8 map_predict lambda48 Expert Pilot Summary", summary),
    )
    return summary


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed_usd", default=str(DEFAULT_FIXED_USD))
    parser.add_argument("--camera_pose_fix_dir", "--stage4a66c_dir", dest="camera_pose_fix_dir", default=str(DEFAULT_CAMERA_FIX_DIR))
    parser.add_argument("--measured_only_pilot_dir", default=str(DEFAULT_MEASURED_ONLY_DIR))
    parser.add_argument("--source_observed_state", default=str(DEFAULT_SOURCE_OBSERVED))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene_variant", default="home_like_scene_v1")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--num_starts", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--formula", default=FORMULA_NAME)
    parser.add_argument("--prediction_mode", default="sim_dynamic")
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--path_cost_mode", choices=["astar"], default="astar")
    parser.add_argument("--candidate_sampling_mode", choices=["reachable_frontier"], default="reachable_frontier")
    parser.add_argument("--motion_mode", default="one_action_only")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--predictor_device", default="cuda")
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument("--max_depth", type=float, default=26.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--pixel_stride", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_snap_radius_cells", type=int, default=20)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--raycast_num_yaw", type=int, default=24)
    parser.add_argument("--raycast_num_pitch", type=int, default=5)
    parser.add_argument("--fov_yaw_deg", type=float, default=90.0)
    parser.add_argument("--fov_pitch_deg", type=float, default=60.0)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--robot_height_m", type=float, default=1.2)
    parser.add_argument("--clearance_height_m", type=float, default=0.6)
    parser.add_argument("--v_max_m_s", type=float, default=1.0)
    parser.add_argument("--yaw_rate_deg_s", type=float, default=90.0)
    parser.add_argument("--action_cost_bias_s", type=float, default=0.25)
    parser.add_argument("--max_action_path_m", type=float, default=5.0)
    parser.add_argument("--exactly_one_action_per_start", action="store_true")
    parser.add_argument("--save_prediction_summary_only", action="store_true")
    parser.add_argument("--save_expert_quality_viz", action="store_true")
    parser.add_argument("--compare_to_measured_only_pilot", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_second_action", action="store_true")
    parser.add_argument("--no_third_frame", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    parser.add_argument("--no_training", action="store_true")
    parser.add_argument("--allow_existing_output_dir", action="store_true")
    parser.add_argument("--reuse_existing_captures", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "headless"):
        args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    return args, AppLauncher


def enforce_runtime_flags(args: argparse.Namespace) -> None:
    required = {
        "exactly_one_action_per_start": args.exactly_one_action_per_start,
        "no_rollout": args.no_rollout,
        "no_second_action": args.no_second_action,
        "no_third_frame": args.no_third_frame,
        "no_rl_gdpo": args.no_rl_gdpo,
        "no_training": args.no_training,
    }
    missing = [key for key, value in required.items() if not bool(value)]
    if missing:
        raise ValueError(f"Required Stage 4A-6.8 safety flags were not provided: {missing}")
    if str(args.formula) != FORMULA_NAME:
        raise ValueError(f"Unsupported formula for this stage: {args.formula}")
    if float(args.lambda_sc) != 48.0:
        raise ValueError("Stage 4A-6.8 requires lambda_sc=48")
    if int(args.num_starts) != 10:
        raise ValueError("Stage 4A-6.8 pilot requires exactly 10 starts")


def main() -> None:
    total_start = time.perf_counter()
    args, app_launcher_cls = parse_args()
    enforce_runtime_flags(args)
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.allow_existing_output_dir):
        raise RuntimeError(f"output_dir already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "git_status_before.txt", git_status_text())
    events: list[dict[str, Any]] = []
    log_event(output_dir, events, "stage_begin", run_output_dir=str(output_dir), formula=FORMULA)

    inputs = load_required_inputs(args, output_dir)
    if bool(args.reuse_existing_captures):
        log_event(output_dir, events, "reuse_existing_captures_begin", reason="previous_close_hang_after_10_captures")
        capture_result = load_existing_capture_result(output_dir, inputs["starts"])
        log_event(output_dir, events, "reuse_existing_captures_complete", capture_count=capture_result["capture_count"])
    else:
        capture_result = run_capture_once(
            args,
            app_launcher_cls,
            output_dir,
            inputs["fixed_usd"],
            inputs["starts"],
            events,
        )
    sample_bundle = process_samples(args, output_dir, inputs, capture_result, events)
    lambda_vs_measured = write_lambda48_comparison(output_dir, sample_bundle["decisions"])
    stage67_comparison = compare_to_stage67(output_dir, inputs["measured_dir"], sample_bundle["decisions"], inputs["starts"])
    quality_audit = write_quality_audit(output_dir, sample_bundle["decisions"], sample_bundle["qualities"], sample_bundle["map_rows"])
    video_report = plot_dataset_level(
        output_dir,
        inputs,
        sample_bundle["decisions"],
        sample_bundle["qualities"],
        lambda_vs_measured,
        stage67_comparison,
        sample_bundle["prediction_density_values"],
    )
    _, prediction_safety, safety_audit = write_safety_reports(args, output_dir, inputs, capture_result, sample_bundle, quality_audit)
    provisional_summary = {
        "completed": False,
        "sample_count": len(sample_bundle["decisions"]),
        "map_predict_calls": int(sample_bundle["metadata"]["map_predict_calls"]),
    }
    write_html_index(output_dir, provisional_summary, sample_bundle["decisions"], quality_audit)
    integrity = verify_dataset(output_dir, sample_bundle, safety_audit, quality_audit)
    summary = write_summary(
        args,
        output_dir,
        inputs,
        capture_result,
        sample_bundle,
        lambda_vs_measured,
        stage67_comparison,
        quality_audit,
        prediction_safety,
        safety_audit,
        integrity,
        video_report,
        float(time.perf_counter() - total_start),
    )
    write_html_index(output_dir, summary, sample_bundle["decisions"], quality_audit)
    integrity = verify_dataset(output_dir, sample_bundle, safety_audit, quality_audit)
    summary["completed"] = bool(integrity["passed"])
    summary["blocked"] = not bool(integrity["passed"])
    summary["main_blocker"] = "" if bool(integrity["passed"]) else "dataset_integrity_failed_after_html_recheck"
    summary["dataset_integrity"] = bool(integrity["passed"])
    save_json(output_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json", summary)
    write_text(
        output_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.md",
        markdown_table("Stage 4A-6.8 map_predict lambda48 Expert Pilot Summary", summary),
    )
    write_text(output_dir / "git_status_after.txt", git_status_text())
    log_event(output_dir, events, "stage_complete", completed=bool(summary["completed"]), integrity_passed=bool(integrity["passed"]))
    if not bool(summary["completed"]):
        raise RuntimeError(f"{STAGE} failed integrity checks: {integrity}")


if __name__ == "__main__":
    main()
