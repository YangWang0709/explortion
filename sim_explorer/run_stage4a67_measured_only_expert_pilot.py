#!/usr/bin/env python3
"""Stage 4A-6.7 measured-only expert pilot for home_like_scene_v1.

This runner consumes the verified Stage 4A-6.6c camera-pose-fix package,
selects one measured-only frontier action for each start variant, launches
Isaac exactly once, captures exactly one selected action view per start, and
writes an expert transition dataset for later imitation learning.

It never calls map_predict, SSCNet, RL, PPO, GDPO, BC, or any continuous
multi-step rollout loop. It does not edit USD files, dependencies,
scene_factory, checkpoints, or the source observed_state.
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
import subprocess
import sys
import time
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
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image, ImageDraw

from astar_planner import astar_2d, build_traversability_grid, path_length_m, summarize_traversability
from depth_to_voxel import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    summarize_observed_grid,
    update_observed_state_from_depth,
)
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


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_STAGE66C_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix"
DEFAULT_FIXED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a67_measured_only_expert_pilot"
DEFAULT_SOURCE_OBSERVED = DEFAULT_STAGE66C_DIR / "observed_state_final.npy"
DEFAULT_FULL_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"

DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")
STATE_CMAP = ListedColormap(["#2f343b", "#80b9c4", "#c95c5c"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)

STAGE = "Stage 4A-6.7-measured-only-expert-pilot"
NO_PREDICTION_NOTE = "prediction_layer=EmptyPredictionLayer; gain_sc remains zero; scoring uses raw measured occupancy only"


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
    digest = hashlib.sha256()
    contiguous = np.ascontiguousarray(array)
    digest.update(contiguous.tobytes())
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    return digest.hexdigest()


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
    fields: list[str] = list(field_order or [])
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(jsonable(value), sort_keys=True)
            if len(text) > 1800:
                text = text[:1800] + "..."
            value_text = f"`{text}`"
        else:
            value_text = f"`{value}`"
        lines.append(f"| {key} | {value_text} |")
    return "\n".join(lines)


def markdown_list(title: str, rows: list[str]) -> str:
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in rows]])


def log_event(output_dir: Path, events: list[dict[str, Any]], event: str, **payload: Any) -> None:
    row = {"time_utc": utc_now(), "event": str(event), **payload}
    events.append(row)
    path = output_dir / "logs/stage4a67_execution_events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False))
        handle.write("\n")
    print(f"[{STAGE}] {event}: {json.dumps(jsonable(payload), sort_keys=True)[:800]}", flush=True)


def validate_observed_state(observed_state: np.ndarray, label: str) -> dict[str, Any]:
    if observed_state.ndim != 3:
        raise ValueError(f"{label} must be 3D [X,Y,Z], got {observed_state.shape}")
    if observed_state.dtype != np.int8:
        raise ValueError(f"{label} must be int8, got {observed_state.dtype}")
    invalid = int(np.count_nonzero(~np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED])))
    if invalid:
        raise ValueError(f"{label} contains {invalid} invalid labels")
    summary = summarize_observed_grid(observed_state)
    summary.update(
        {
            "label": label,
            "dtype": str(observed_state.dtype),
            "invalid_label_count": invalid,
            "sha256": sha256_array(observed_state),
            "labels_present": {
                "UNKNOWN": bool(np.any(observed_state == UNKNOWN)),
                "FREE": bool(np.any(observed_state == FREE)),
                "OCCUPIED": bool(np.any(observed_state == OCCUPIED)),
            },
        }
    )
    return summary


def state_transition(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    if before.shape != after.shape:
        raise ValueError(f"state shape mismatch: {before.shape} vs {after.shape}")
    out = {
        "unknown_to_free": int(np.count_nonzero((before == UNKNOWN) & (after == FREE))),
        "unknown_to_occupied": int(np.count_nonzero((before == UNKNOWN) & (after == OCCUPIED))),
        "free_to_occupied": int(np.count_nonzero((before == FREE) & (after == OCCUPIED))),
        "occupied_to_free": int(np.count_nonzero((before == OCCUPIED) & (after == FREE))),
        "changed_voxel_count": int(np.count_nonzero(before != after)),
        "invalid_label_count_after": int(np.count_nonzero(~np.isin(after, [UNKNOWN, FREE, OCCUPIED]))),
    }
    out["newly_observed"] = out["unknown_to_free"] + out["unknown_to_occupied"]
    return out


def hardware_report(max_workers: int, device: str) -> dict[str, Any]:
    gpu_name = "unavailable"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
        elif result.stderr.strip():
            gpu_name = f"nvidia-smi_error: {result.stderr.strip()[:400]}"
    except Exception as exc:  # noqa: BLE001
        gpu_name = f"nvidia-smi_unavailable: {exc}"
    return {
        "stage": STAGE,
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "requested_cpu_workers": int(max_workers),
        "actual_cpu_workers": min(int(max_workers), os.cpu_count() or 1),
        "requested_device": str(device),
        "gpu_name": gpu_name,
        "gpu_rtx_5080_requested_by_user": True,
        "BLAS_OMP_NUMEXPR_VECLIB_threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        },
    }


def checkpoint_manifest(checkpoint_root: Path, explicit_checkpoint: Path) -> dict[str, Any]:
    files = []
    roots = [checkpoint_root]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file():
                files.append(
                    {
                        "path": str(path),
                        "size_bytes": int(path.stat().st_size),
                        "sha256": sha256_file(path),
                    }
                )
    if explicit_checkpoint.is_file() and all(row["path"] != str(explicit_checkpoint) for row in files):
        files.append(
            {
                "path": str(explicit_checkpoint),
                "size_bytes": int(explicit_checkpoint.stat().st_size),
                "sha256": sha256_file(explicit_checkpoint),
            }
        )
    return {"file_count": len(files), "files": files}


def pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def nearest_free_z_for_xy(observed_state: np.ndarray, xy: tuple[int, int], preferred_k: int) -> int | None:
    i, j = int(xy[0]), int(xy[1])
    if not (0 <= i < observed_state.shape[0] and 0 <= j < observed_state.shape[1]):
        return None
    free_z = np.flatnonzero(observed_state[i, j, :] == FREE)
    if free_z.size == 0:
        return None
    best_idx = int(np.argmin(np.abs(free_z.astype(np.int64) - int(preferred_k))))
    return int(free_z[best_idx])


def inspection_yaw_priors(starts: list[dict[str, Any]], inspection_poses: list[dict[str, Any]]) -> dict[int, list[float]]:
    out: dict[int, list[float]] = {int(start["index"]): [] for start in starts}
    for start in starts:
        start_pos = np.asarray(start["position"], dtype=np.float64)
        for pose in inspection_poses:
            pose_pos = np.asarray(pose.get("position", [math.inf, math.inf, math.inf]), dtype=np.float64)
            if pose_pos.shape == (3,) and float(np.linalg.norm(pose_pos - start_pos)) <= 1.0e-4:
                yaw = float(pose.get("yaw_rad", pose.get("yaw", 0.0)))
                if all(abs(wrap_angle(yaw - existing)) > 1.0e-6 for existing in out[int(start["index"])]):
                    out[int(start["index"])].append(yaw)
    return out


def unknown_centroid_yaw(
    observed_state: np.ndarray,
    candidate_xy: tuple[int, int],
    fallback_yaw: float,
    radius_cells: int = 14,
) -> float:
    i, j = int(candidate_xy[0]), int(candidate_xy[1])
    i0 = max(0, i - int(radius_cells))
    i1 = min(observed_state.shape[0], i + int(radius_cells) + 1)
    j0 = max(0, j - int(radius_cells))
    j1 = min(observed_state.shape[1], j + int(radius_cells) + 1)
    unknown_xy = np.any(observed_state[i0:i1, j0:j1, :] == UNKNOWN, axis=2)
    coords = np.argwhere(unknown_xy)
    if coords.size == 0:
        return float(fallback_yaw)
    coords[:, 0] += i0
    coords[:, 1] += j0
    centroid = coords.astype(np.float64).mean(axis=0)
    delta = centroid - np.array([float(i), float(j)], dtype=np.float64)
    if float(np.linalg.norm(delta)) <= 1.0e-6:
        return float(fallback_yaw)
    return float(math.atan2(float(delta[1]), float(delta[0])))


def yaw_candidates(base_yaw: float, start_yaw: float, priors: list[float], sample_count: int) -> list[float]:
    values = [wrap_angle(base_yaw), wrap_angle(start_yaw)]
    values.extend(wrap_angle(value) for value in priors)
    if int(sample_count) > 1:
        for idx in range(int(sample_count)):
            values.append(wrap_angle(base_yaw + math.tau * idx / float(sample_count)))
    unique: list[float] = []
    for value in values:
        if all(abs(wrap_angle(value - existing)) > 1.0e-5 for existing in unique):
            unique.append(float(value))
    return unique


def local_unknown_count(unknown_xy: np.ndarray, xy: tuple[int, int], radius: int = 7) -> int:
    i, j = int(xy[0]), int(xy[1])
    i0 = max(0, i - radius)
    i1 = min(unknown_xy.shape[0], i + radius + 1)
    j0 = max(0, j - radius)
    j1 = min(unknown_xy.shape[1], j + radius + 1)
    return int(np.count_nonzero(unknown_xy[i0:i1, j0:j1]))


def select_candidate_cells(
    observed_state: np.ndarray,
    candidate_mask: np.ndarray,
    start_xy: tuple[int, int],
    max_cells: int,
) -> list[tuple[int, int, dict[str, Any]]]:
    coords = np.argwhere(np.asarray(candidate_mask, dtype=bool))
    unknown_xy = np.any(observed_state == UNKNOWN, axis=2)
    rows: list[tuple[float, float, int, int, int, float]] = []
    for coord in coords:
        xy = (int(coord[0]), int(coord[1]))
        dist_cells = math.hypot(float(xy[0] - start_xy[0]), float(xy[1] - start_xy[1]))
        density = local_unknown_count(unknown_xy, xy)
        heuristic = float(density) - 0.0125 * dist_cells
        rows.append((heuristic, -dist_cells, density, xy[0], xy[1], dist_cells))
    rows.sort(key=lambda item: (-item[0], -item[2], -item[1], item[3], item[4]))
    selected = []
    for heuristic, neg_dist, density, i, j, dist_cells in rows[: int(max_cells)]:
        selected.append(
            (
                int(i),
                int(j),
                {
                    "sampling_heuristic": float(heuristic),
                    "distance_cells": float(dist_cells),
                    "local_unknown_xy_count": int(density),
                },
            )
        )
    return selected


def score_one_candidate(
    observed_state: np.ndarray,
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
    candidate_k = nearest_free_z_for_xy(observed_state, candidate_xy, preferred_k)
    if candidate_k is None:
        return None
    candidate_grid = (int(candidate_xy[0]), int(candidate_xy[1]), int(candidate_k))
    candidate_world_grid_center = grid_to_world(candidate_grid, bounds, float(voxel_size))
    camera_world = [
        float(candidate_world_grid_center[0]),
        float(candidate_world_grid_center[1]),
        float(start_world[2]),
    ]
    astar = astar_2d(traversable, start_xy, candidate_xy, allow_diagonal=True)
    if not bool(astar.get("reachable", False)):
        return None
    astar_path = [(int(cell[0]), int(cell[1])) for cell in astar["path"]]
    path_m = path_length_m(astar_path, float(voxel_size))
    if float(path_m) > float(args.max_action_path_m):
        return None
    fallback_base = math.atan2(float(candidate_xy[1] - start_xy[1]), float(candidate_xy[0] - start_xy[0]))
    base_yaw = unknown_centroid_yaw(observed_state, candidate_xy, fallback_base)
    empty_prediction = EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape))
    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / float(voxel_size))))
    best: dict[str, Any] | None = None
    evaluated_yaws: list[dict[str, Any]] = []
    for yaw in yaw_candidates(base_yaw, float(start_yaw), yaw_priors, int(args.num_yaw_samples)):
        view = SimCandidateView(
            id=int(candidate_id),
            grid_position=candidate_grid,
            world_position=tuple(float(v) for v in camera_world),
            yaw=float(yaw),
            valid=True,
            candidate_source="reachable_frontier_measured_only",
        )
        visible = raycast_visible_voxels_observed(
            view,
            observed_state,
            max_range_voxels=max_range_voxels,
            num_yaw=max(4, int(args.raycast_num_yaw)),
            num_pitch=max(3, int(args.raycast_num_pitch)),
            fov_yaw_deg=float(args.fov_yaw_deg),
            fov_pitch_deg=float(args.fov_pitch_deg),
        )
        view = compute_paper_gains_for_candidate(
            view,
            observed_state,
            empty_prediction,
            visible,
            tau=0.1,
            sc_gain_formula="raw_count",
            sc_occ_threshold=1.0,
            sc_conf_threshold=1.0,
            sc_count_mode="raw_count",
        )
        yaw_delta = abs(wrap_angle(float(yaw) - float(start_yaw)))
        yaw_time_s = yaw_delta / max(math.radians(float(args.yaw_rate_deg_s)), 1.0e-6)
        path_time_s = float(path_m) / max(float(args.v_max_m_s), 1.0e-6)
        cost_s = path_time_s + yaw_time_s + float(args.action_cost_bias_s)
        score = float(view.gain_exp) / max(cost_s, 1.0e-6)
        yaw_row = {
            "yaw": float(yaw),
            "gain_exp": float(view.gain_exp),
            "gain_sc": float(view.gain_sc),
            "gain_hybrid": float(view.gain_hybrid),
            "visible_count": int(view.visible_count),
            "measured_visible_count": int(view.measured_visible_count),
            "predicted_unmeasured_visible_count": int(view.predicted_unmeasured_visible_count),
            "frontier_count_visible": int(view.frontier_count_visible),
            "path_cost_m": float(path_m),
            "path_time_s": float(path_time_s),
            "yaw_delta_rad": float(yaw_delta),
            "yaw_time_s": float(yaw_time_s),
            "cost_s": float(cost_s),
            "score": float(score),
        }
        evaluated_yaws.append(yaw_row)
        key = (
            float(score),
            float(view.gain_exp),
            int(view.frontier_count_visible),
            -float(cost_s),
            -float(abs(wrap_angle(yaw - base_yaw))),
        )
        if best is None or key > best["selection_key"]:
            best = {**yaw_row, "selection_key": key}
    if best is None:
        return None
    return {
        "candidate_id": int(candidate_id),
        "candidate_source": "reachable_frontier_measured_only",
        "grid": [int(v) for v in candidate_grid],
        "xy": [int(candidate_xy[0]), int(candidate_xy[1])],
        "world": [float(v) for v in camera_world],
        "yaw_rad": float(best["yaw"]),
        "target": pose_target([float(v) for v in camera_world], float(best["yaw"])),
        "score": float(best["score"]),
        "gain_exp": float(best["gain_exp"]),
        "gain_sc": float(best["gain_sc"]),
        "gain_hybrid": float(best["gain_hybrid"]),
        "visible_count": int(best["visible_count"]),
        "measured_visible_count": int(best["measured_visible_count"]),
        "predicted_unmeasured_visible_count": int(best["predicted_unmeasured_visible_count"]),
        "frontier_count_visible": int(best["frontier_count_visible"]),
        "path_cost_m": float(best["path_cost_m"]),
        "path_time_s": float(best["path_time_s"]),
        "yaw_delta_rad": float(best["yaw_delta_rad"]),
        "yaw_time_s": float(best["yaw_time_s"]),
        "cost_s": float(best["cost_s"]),
        "astar_num_expanded": int(astar.get("num_expanded", 0)),
        "astar_path_xy": [[int(a), int(b)] for a, b in astar_path],
        "base_yaw_rad": float(base_yaw),
        "yaw_samples_evaluated": int(len(evaluated_yaws)),
        "yaw_sample_scores": evaluated_yaws,
        "raw_measured_occupancy_only": True,
        "prediction_layer": "EmptyPredictionLayer",
        "map_predict_called": False,
        "sscnet_inference_called": False,
        **candidate_meta,
    }


def score_start_variant(
    observed_state: np.ndarray,
    bounds: dict[str, list[float]],
    voxel_size: float,
    traversability: dict[str, Any],
    frontier_mask: np.ndarray,
    start: dict[str, Any],
    yaw_priors_by_start: dict[int, list[float]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    start_position = [float(v) for v in start["position"]]
    start_yaw = float(start.get("yaw_rad", start.get("yaw", 0.0)))
    start_grid = world_to_grid(start_position, bounds, float(voxel_size), shape=observed_state.shape, clip=True)
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
        raise RuntimeError(f"start {start.get('name')} has no snapped reachable XY: {reachable}")
    snapped_xy = (int(snapped[0]), int(snapped[1]))
    selected_cells = select_candidate_cells(
        observed_state,
        candidate_mask,
        snapped_xy,
        int(args.max_scored_candidates_per_start),
    )
    if not selected_cells:
        raise RuntimeError(f"start {start.get('name')} produced no measured-only frontier cells: {reachable}")
    traversable = np.asarray(traversability["traversable"], dtype=bool)
    yaw_priors = yaw_priors_by_start.get(int(start["index"]), [])
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as executor:
        futures = []
        for cid, (i, j, meta) in enumerate(selected_cells):
            futures.append(
                executor.submit(
                    score_one_candidate,
                    observed_state,
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
                rows.append(result)
    if not rows:
        raise RuntimeError(f"start {start.get('name')} had candidate cells but none survived A*/path/raycast scoring")
    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["gain_exp"]),
            -int(row["frontier_count_visible"]),
            float(row["cost_s"]),
            int(row["grid"][0]),
            int(row["grid"][1]),
        )
    )
    topn = rows[: int(args.top_n)]
    selected = dict(topn[0])
    action_pose = {
        "index": int(start["index"]),
        "name": f"action_from_{start['name']}",
        "start_variant_name": start["name"],
        "position": [float(v) for v in selected["world"]],
        "yaw": float(selected["yaw_rad"]),
        "yaw_rad": float(selected["yaw_rad"]),
        "target": selected["target"],
        "target_xy": [float(selected["target"][0]), float(selected["target"][1])],
        "source": "stage4a67_measured_only_top1_frontier_action",
        "semantic_zone_guess": start.get("semantic_zone_guess"),
        "start_position": start_position,
        "start_yaw_rad": start_yaw,
        "selected_grid": selected["grid"],
        "selected_score": float(selected["score"]),
        "selected_gain_exp": float(selected["gain_exp"]),
        "selected_path_cost_m": float(selected["path_cost_m"]),
        "one_action_only_for_this_start": True,
    }
    return {
        "start_index": int(start["index"]),
        "start_name": start["name"],
        "start": start,
        "start_grid": [int(v) for v in start_grid],
        "snapped_start_xy": [int(v) for v in snapped_xy],
        "start_yaw_rad": float(start_yaw),
        "reachable_context": {
            key: value
            for key, value in reachable.items()
            if key not in {"candidate_mask", "reachable_mask"}
        },
        "sampled_candidate_count": int(len(selected_cells)),
        "scored_candidate_count": int(len(rows)),
        "top_n_count": int(len(topn)),
        "top_n": topn,
        "selected": selected,
        "action_pose": action_pose,
        "raw_measured_occupancy_only": True,
        "prediction_layer": "EmptyPredictionLayer",
        "map_predict_called": False,
        "sscnet_inference_called": False,
        "rl_policy_used": False,
        "second_action_executed": False,
        "third_action_executed": False,
    }


def topdown_state(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == FREE, axis=2)
    top = np.full(observed_state.shape[:2], UNKNOWN, dtype=np.int8)
    top[free] = FREE
    top[occupied] = OCCUPIED
    return top


def save_observed_topdown(path: Path, observed_state: np.ndarray, title: str) -> None:
    top = topdown_state(observed_state)
    fig, ax = plt.subplots(figsize=(8.2, 8.6), constrained_layout=True)
    image = ax.imshow(top.T, origin="lower", cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    fig.colorbar(image, ax=ax, ticks=[UNKNOWN, FREE, OCCUPIED], label="state")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def grid_xy_to_world_plot(xy: list[int] | tuple[int, int], bounds: dict[str, list[float]], voxel_size: float) -> tuple[float, float]:
    return (
        float(bounds["x"][0] + (int(xy[0]) + 0.5) * float(voxel_size)),
        float(bounds["y"][0] + (int(xy[1]) + 0.5) * float(voxel_size)),
    )


def save_decision_topdown(
    path: Path,
    observed_state: np.ndarray,
    bounds: dict[str, list[float]],
    voxel_size: float,
    decision: dict[str, Any],
) -> None:
    top = topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.4, 9.2), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    start = decision["start"]["position"]
    action = decision["action_pose"]["position"]
    ax.scatter([start[0]], [start[1]], s=90, c="#2563eb", marker="^", edgecolors="white", linewidths=0.8, label="start")
    ax.scatter([action[0]], [action[1]], s=120, c="#22a06b", marker="*", edgecolors="black", linewidths=0.6, label="selected")
    ax.plot([start[0], action[0]], [start[1], action[1]], color="#22a06b", linewidth=1.5)
    xs = [row["world"][0] for row in decision["top_n"]]
    ys = [row["world"][1] for row in decision["top_n"]]
    scores = [row["score"] for row in decision["top_n"]]
    if xs:
        scatter = ax.scatter(xs, ys, s=35, c=scores, cmap="magma", edgecolors="white", linewidths=0.35, label="top-N")
        fig.colorbar(scatter, ax=ax, shrink=0.72, label="score")
    ax.set_title(f"{decision['start_index']:02d} {decision['start_name']}")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_all_actions_topdown(
    path: Path,
    observed_state: np.ndarray,
    bounds: dict[str, list[float]],
    decisions: list[dict[str, Any]],
) -> None:
    top = topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(8.0, 9.4), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    for decision in decisions:
        start = decision["start"]["position"]
        action = decision["action_pose"]["position"]
        ax.plot([start[0], action[0]], [start[1], action[1]], color="#22a06b", linewidth=1.0, alpha=0.75)
        ax.scatter([start[0]], [start[1]], s=45, c="#2563eb", marker="^", edgecolors="white", linewidths=0.45)
        ax.scatter([action[0]], [action[1]], s=58, c="#22a06b", marker="*", edgecolors="black", linewidths=0.35)
        ax.text(action[0], action[1], str(decision["start_index"]), fontsize=7, color="#111827")
    ax.set_title("Stage 4A-6.7 selected one measured-only action per start")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def rgb_stats(rgb: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(v) for v in rgb.shape],
        "min": int(rgb.min()) if rgb.size else None,
        "max": int(rgb.max()) if rgb.size else None,
        "mean": float(rgb.mean()) if rgb.size else None,
        "std": float(rgb.std()) if rgb.size else None,
        "nonblank": bool(rgb.size and int(rgb.max()) > 2 and float(rgb.std()) >= 1.0),
    }


def depth_stats(depth: np.ndarray) -> dict[str, Any]:
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    return {
        "shape": [int(v) for v in depth.shape],
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()) if positive.size else None,
        "median": float(np.median(positive)) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
        "has_positive_finite_depth": bool(positive.size > 0),
    }


def normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    return np.clip(np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def save_depth_color(path: Path, depth: np.ndarray, title: str) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError(f"No finite positive depth for {path}")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135)
    plt.close(fig)


def extract_rgb(camera: Any, label: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    import torch

    for key in RGB_KEY_CANDIDATES:
        if key in camera.data.output:
            tensor = camera.data.output[key][0]
            array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
            rgb = normalize_rgb(array)
            return rgb, key, rgb_stats(rgb)
    raise KeyError(f"No RGB/RGBA camera output for {label}; keys={list(camera.data.output.keys())}")


def extract_depth(camera: Any, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if DEPTH_KEY not in camera.data.output:
        raise KeyError(f"{DEPTH_KEY} missing for {label}; keys={list(camera.data.output.keys())}")
    tensor = camera.data.output[DEPTH_KEY][0]
    depth = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = np.asarray(depth, dtype=np.float32)
    return depth, depth_stats(depth)


def capture_action_pose(output_dir: Path, camera: Any, sim: Any, pose: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    idx = int(pose["index"])
    position = [float(v) for v in pose["position"]]
    target = pose_target(position, float(pose["yaw_rad"]))
    camera.set_world_poses_from_view(
        eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
        targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
    )
    for _ in range(max(int(args.settle_steps), 1)):
        sim.step()
        camera.update(dt=sim.get_physics_dt())
    rgb, rgb_key, rstats = extract_rgb(camera, f"action {idx}")
    depth, dstats = extract_depth(camera, f"action {idx}")
    rgb_path = output_dir / f"action_rgb_{idx:03d}.png"
    depth_path = output_dir / f"action_depth_{idx:03d}.npy"
    depth_color_path = output_dir / f"action_depth_color_{idx:03d}.png"
    pose_path = output_dir / f"action_pose_{idx:03d}.json"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    save_depth_color(depth_color_path, depth, f"action depth {idx:03d}")
    pose_record = dict(pose)
    pose_record.update(
        {
            "target": target,
            "render_backend": "isaac_headless",
            "rgb_file": rgb_path.name,
            "depth_file": depth_path.name,
            "depth_color_file": depth_color_path.name,
            "one_headless_measured_only_capture_for_this_start": True,
        }
    )
    save_json(pose_path, pose_record)
    return {
        "index": idx,
        "name": pose.get("name"),
        "start_variant_name": pose.get("start_variant_name"),
        "semantic_zone_guess": pose.get("semantic_zone_guess"),
        "render_backend": "isaac_headless",
        "rgb_file": rgb_path.name,
        "depth_file": depth_path.name,
        "depth_color_file": depth_color_path.name,
        "pose_file": pose_path.name,
        "rgb_key_used": rgb_key,
        "rgb_stats": rstats,
        "depth_stats": dstats,
        "second_action_executed": False,
        "third_action_executed": False,
    }


def run_one_isaac_startup(
    args: argparse.Namespace,
    app_launcher_cls: Any,
    output_dir: Path,
    fixed_usd: Path,
    action_poses: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "GNOME_SETUP_DISPLAY"):
        os.environ.pop(key, None)
    setattr(args, "headless", True)
    if hasattr(args, "enable_cameras"):
        setattr(args, "enable_cameras", True)

    log_event(output_dir, events, "isaac_startup_begin", fixed_usd=str(fixed_usd), capture_count=len(action_poses))
    start_time = time.perf_counter()
    app_launcher = app_launcher_cls(args)
    simulation_app = app_launcher.app
    startup_s = float(time.perf_counter() - start_time)
    log_event(output_dir, events, "isaac_startup_complete", startup_seconds=startup_s)
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg
        from scene_factory import build_home_like_scene_v1

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
        sim.set_camera_view([0.0, -18.0, 15.0], [0.0, 0.0, 0.8])
        dome = sim_utils.DomeLightCfg(intensity=800.0, color=(0.84, 0.86, 0.82))
        dome.func("/World/MeasuredOnlyPilotSoftFillLight", dome)
        builder_metadata = build_home_like_scene_v1(
            seed=int(args.scene_seed),
            spawn=True,
            sim_utils_module=sim_utils,
            staged_usd_path=str(fixed_usd),
        )
        sim_utils.create_prim("/World/MeasuredOnlyExpertCameraRig", "Xform")
        camera = Camera(
            cfg=CameraCfg(
                prim_path="/World/MeasuredOnlyExpertCameraRig/CameraSensor",
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
        )
        sim.reset()
        intrinsic = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
        camera_info = {
            "render_backend": "isaac_headless",
            "sensor_api_depth_key": DEPTH_KEY,
            "depth_units": "meters",
            "width": int(args.camera_width),
            "height": int(args.camera_height),
            "max_depth": float(args.max_depth),
            "near_depth": 0.05,
            "intrinsic_matrix": intrinsic.tolist(),
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
            "data_types_requested": ["rgb", DEPTH_KEY],
        }
        save_json(output_dir / "camera_info.json", camera_info)
        records = []
        for pose in action_poses:
            capture_record = capture_action_pose(output_dir, camera, sim, pose, args)
            records.append(capture_record)
            log_event(
                output_dir,
                events,
                "action_capture_complete",
                start_index=int(capture_record["index"]),
                rgb_nonblank=bool(capture_record["rgb_stats"]["nonblank"]),
                depth_positive_count=int(capture_record["depth_stats"]["positive_count"]),
            )
        return {
            "isaac_headless_startup_count": 1,
            "isaac_startup_seconds": startup_s,
            "builder_metadata": builder_metadata,
            "camera_info": camera_info,
            "capture_records": records,
            "capture_count": len(records),
            "one_capture_per_start": len(records) == len(action_poses),
        }
    finally:
        simulation_app.close()
        log_event(output_dir, events, "isaac_app_closed")


def save_capture_grids(output_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    cols = 5
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.65, rows * 2.25), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, record in zip(axes, records):
        rgb = np.asarray(Image.open(output_dir / record["rgb_file"]).convert("RGB"))
        ax.imshow(rgb)
        ax.set_title(f"{record['index']:02d} {record.get('semantic_zone_guess') or ''}", fontsize=6.5)
        ax.axis("off")
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.savefig(output_dir / "action_rgb_grid.png", dpi=155)
    plt.close(fig)

    depths = [np.load(output_dir / record["depth_file"]) for record in records]
    valid_values = [depth[np.isfinite(depth) & (depth > 0.0)] for depth in depths]
    valid_values = [values for values in valid_values if values.size]
    if not valid_values:
        return
    vmin = min(float(values.min()) for values in valid_values)
    vmax = max(float(values.max()) for values in valid_values)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.65, rows * 2.3), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    last = None
    for ax, record, depth in zip(axes, records, depths):
        last = ax.imshow(np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan)), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{record['index']:02d} {record.get('semantic_zone_guess') or ''}", fontsize=6.5)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.colorbar(last, ax=axes[: len(records)].tolist(), shrink=0.62, label="depth (m)")
    fig.savefig(output_dir / "action_depth_grid.png", dpi=155)
    plt.close(fig)


def make_closeup_contact_sheet(output_dir: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    thumb_w, thumb_h = 320, 240
    cols = 5
    rows = int(math.ceil(len(records) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 26)), (244, 247, 250))
    draw = ImageDraw.Draw(sheet)
    for idx, record in enumerate(records):
        image = Image.open(output_dir / record["rgb_file"]).convert("RGB").resize((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 26)
        draw.rectangle((x, y, x + thumb_w, y + 26), fill=(20, 27, 38))
        draw.text((x + 8, y + 6), f"{record['index']:02d} {record.get('semantic_zone_guess') or ''}", fill=(245, 247, 250))
        sheet.paste(image, (x, y + 26))
    sheet.save(output_dir / "action_closeup_contact_sheet.png")


def make_mp4(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_dir = output_dir / "expert_action_flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    if not records:
        return {"mp4_created": False, "frame_count": 0, "frame_dir": str(frame_dir), "reason": "no_action_records"}
    for frame_idx in range(60):
        record = records[int(frame_idx / 60.0 * len(records)) % len(records)]
        image = Image.open(output_dir / record["rgb_file"]).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 42), fill=(18, 24, 32))
        draw.text((12, 11), f"Stage 4A-6.7 measured-only action {record['index']:02d}", fill=(240, 244, 248))
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        image.save(frame_path)
        frames.append(frame_path)
    report: dict[str, Any] = {"mp4_created": False, "frame_count": len(frames), "frame_dir": str(frame_dir), "video_path": None}
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


def build_transitions_and_dataset(
    args: argparse.Namespace,
    output_dir: Path,
    source_observed: np.ndarray,
    decisions: list[dict[str, Any]],
    capture_records: list[dict[str, Any]],
    camera_info: dict[str, Any],
    bounds: dict[str, list[float]],
    voxel_size: float,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    record_by_index = {int(record["index"]): record for record in capture_records}
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    cumulative = source_observed.copy()
    transitions: list[dict[str, Any]] = []
    pre_states = []
    post_states = []
    action_world = []
    start_world = []
    action_grid = []
    start_grid = []
    scores = []
    gains = []
    path_costs = []
    topn_scores: list[list[float]] = []
    topn_grids: list[list[list[int]]] = []
    max_top_n = max(len(decision["top_n"]) for decision in decisions)
    for decision in decisions:
        idx = int(decision["start_index"])
        record = record_by_index[idx]
        pose = read_json(output_dir / record["pose_file"])
        depth = np.load(output_dir / record["depth_file"])
        per_transition_pre = source_observed.copy()
        per_transition_post = update_observed_state_from_depth(
            observed_state=per_transition_pre.copy(),
            depth=depth,
            camera_pose=pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=float(voxel_size),
            pixel_stride=int(args.pixel_stride),
        )
        before_cumulative = cumulative.copy()
        cumulative = update_observed_state_from_depth(
            observed_state=cumulative,
            depth=depth,
            camera_pose=pose,
            camera_info=camera_info,
            bounds=bounds,
            voxel_size=float(voxel_size),
            pixel_stride=int(args.pixel_stride),
        )
        post_path = samples_dir / f"observed_state_post_start{idx:03d}.npy"
        cumulative_path = output_dir / f"observed_state_cumulative_after_start{idx:03d}.npy"
        np.save(post_path, per_transition_post)
        np.save(cumulative_path, cumulative)
        transition_delta = state_transition(source_observed, per_transition_post)
        cumulative_delta = state_transition(before_cumulative, cumulative)
        sample_npz_path = samples_dir / f"expert_sample_start{idx:03d}.npz"
        top_scores = [float(row["score"]) for row in decision["top_n"]]
        top_grids = [[int(v) for v in row["grid"]] for row in decision["top_n"]]
        padded_scores = top_scores + [np.nan] * (max_top_n - len(top_scores))
        padded_grids = top_grids + [[-1, -1, -1] for _ in range(max_top_n - len(top_grids))]
        np.savez_compressed(
            sample_npz_path,
            pre_observed_state=source_observed,
            post_observed_state=per_transition_post,
            start_world=np.asarray(decision["start"]["position"], dtype=np.float32),
            action_world=np.asarray(pose["position"] + [pose["yaw_rad"]], dtype=np.float32),
            start_grid=np.asarray(decision["start_grid"], dtype=np.int32),
            action_grid=np.asarray(decision["selected"]["grid"], dtype=np.int32),
            topn_scores=np.asarray(padded_scores, dtype=np.float32),
            topn_grids=np.asarray(padded_grids, dtype=np.int32),
            selected_score=np.asarray(float(decision["selected"]["score"]), dtype=np.float32),
            selected_gain_exp=np.asarray(float(decision["selected"]["gain_exp"]), dtype=np.float32),
            selected_path_cost_m=np.asarray(float(decision["selected"]["path_cost_m"]), dtype=np.float32),
        )
        transition = {
            "stage": STAGE,
            "sample_index": idx,
            "start_variant": decision["start_name"],
            "frame_index": 0,
            "exactly_one_action": True,
            "second_action_executed": False,
            "third_action_executed": False,
            "continuous_rollout_executed": False,
            "source_observed_state_path": str(args.source_observed_state),
            "source_observed_state_sha256": sha256_file(args.source_observed_state),
            "pre_observed_state_sha256": sha256_array(source_observed),
            "post_observed_state_path": str(post_path),
            "post_observed_state_sha256": sha256_array(per_transition_post),
            "cumulative_observed_state_path": str(cumulative_path),
            "cumulative_observed_state_sha256": sha256_array(cumulative),
            "sample_npz": str(sample_npz_path),
            "rgb_file": record["rgb_file"],
            "depth_file": record["depth_file"],
            "pose_file": record["pose_file"],
            "start_pose": decision["start"],
            "action_pose": pose,
            "selected_decision": decision["selected"],
            "top_n_decisions": decision["top_n"],
            "transition_delta": transition_delta,
            "cumulative_delta": cumulative_delta,
            "measured_only": True,
            "raw_measured_occupancy_only": True,
            "prediction_layer": "EmptyPredictionLayer",
            "map_predict_called": False,
            "sscnet_inference_called": False,
            "rl_policy_used": False,
        }
        save_json(samples_dir / f"expert_sample_start{idx:03d}.json", transition)
        transitions.append(transition)
        pre_states.append(source_observed)
        post_states.append(per_transition_post)
        action_world.append(pose["position"] + [pose["yaw_rad"]])
        start_world.append(decision["start"]["position"] + [decision["start_yaw_rad"]])
        action_grid.append(decision["selected"]["grid"])
        start_grid.append(decision["start_grid"])
        scores.append(float(decision["selected"]["score"]))
        gains.append(float(decision["selected"]["gain_exp"]))
        path_costs.append(float(decision["selected"]["path_cost_m"]))
        topn_scores.append(padded_scores)
        topn_grids.append(padded_grids)

    np.save(output_dir / "observed_state_final.npy", cumulative)
    dataset_npz = output_dir / "expert_dataset.npz"
    np.savez_compressed(
        dataset_npz,
        pre_observed_states=np.asarray(pre_states, dtype=np.int8),
        post_observed_states=np.asarray(post_states, dtype=np.int8),
        action_world=np.asarray(action_world, dtype=np.float32),
        start_world=np.asarray(start_world, dtype=np.float32),
        action_grid=np.asarray(action_grid, dtype=np.int32),
        start_grid=np.asarray(start_grid, dtype=np.int32),
        selected_scores=np.asarray(scores, dtype=np.float32),
        selected_gain_exp=np.asarray(gains, dtype=np.float32),
        selected_path_cost_m=np.asarray(path_costs, dtype=np.float32),
        topn_scores=np.asarray(topn_scores, dtype=np.float32),
        topn_grids=np.asarray(topn_grids, dtype=np.int32),
    )
    final_summary = validate_observed_state(cumulative, "stage4a67_observed_state_final")
    manifest = {
        "stage": STAGE,
        "created_at_utc": utc_now(),
        "dataset_npz": str(dataset_npz),
        "sample_count": len(transitions),
        "expected_sample_count": len(decisions),
        "source_observed_state": str(args.source_observed_state),
        "source_observed_state_sha256": sha256_file(args.source_observed_state),
        "observed_state_final": str(output_dir / "observed_state_final.npy"),
        "observed_state_final_sha256": sha256_array(cumulative),
        "observed_state_final_summary": final_summary,
        "top_n": int(args.top_n),
        "one_action_per_start": all(row["exactly_one_action"] for row in transitions),
        "capture_count": len(capture_records),
        "measured_only": True,
        "raw_measured_occupancy_only": True,
        "map_predict_called": False,
        "sscnet_inference_called": False,
        "rl_training_run": False,
        "continuous_rollout_executed": False,
        "transitions_jsonl": str(output_dir / "expert_transitions.jsonl"),
        "samples_dir": str(samples_dir),
    }
    save_json(output_dir / "expert_dataset_manifest.json", manifest)
    write_text(output_dir / "expert_dataset_manifest.md", markdown_table("Expert Dataset Manifest", manifest))
    write_jsonl(output_dir / "expert_transitions.jsonl", transitions)
    save_json(output_dir / "expert_transitions.json", {"transitions": transitions})
    write_csv(
        output_dir / "expert_transitions.csv",
        [
            {
                "sample_index": row["sample_index"],
                "start_variant": row["start_variant"],
                "score": row["selected_decision"]["score"],
                "gain_exp": row["selected_decision"]["gain_exp"],
                "path_cost_m": row["selected_decision"]["path_cost_m"],
                "newly_observed": row["transition_delta"]["newly_observed"],
                "rgb_file": row["rgb_file"],
                "depth_file": row["depth_file"],
                "sample_npz": row["sample_npz"],
            }
            for row in transitions
        ],
    )
    return cumulative, transitions, manifest


def write_topn_outputs(output_dir: Path, decisions: list[dict[str, Any]]) -> None:
    flat_rows = []
    for decision in decisions:
        start_idx = int(decision["start_index"])
        save_json(output_dir / f"decisions/topn_decision_start{start_idx:03d}.json", decision)
        write_text(
            output_dir / f"decisions/topn_decision_start{start_idx:03d}.md",
            markdown_table(
                f"Top-N Decision Start {start_idx:03d}",
                {
                    "start_name": decision["start_name"],
                    "sampled_candidate_count": decision["sampled_candidate_count"],
                    "scored_candidate_count": decision["scored_candidate_count"],
                    "selected_grid": decision["selected"]["grid"],
                    "selected_world": decision["selected"]["world"],
                    "selected_yaw_rad": decision["selected"]["yaw_rad"],
                    "selected_score": decision["selected"]["score"],
                    "selected_gain_exp": decision["selected"]["gain_exp"],
                    "selected_path_cost_m": decision["selected"]["path_cost_m"],
                    "top_n_count": decision["top_n_count"],
                    "raw_measured_occupancy_only": True,
                    "prediction_layer": "EmptyPredictionLayer",
                },
            ),
        )
        for rank, row in enumerate(decision["top_n"], start=1):
            flat_rows.append(
                {
                    "start_index": start_idx,
                    "start_name": decision["start_name"],
                    "rank": rank,
                    "candidate_id": row["candidate_id"],
                    "grid": row["grid"],
                    "world": row["world"],
                    "yaw_rad": row["yaw_rad"],
                    "score": row["score"],
                    "gain_exp": row["gain_exp"],
                    "gain_sc": row["gain_sc"],
                    "visible_count": row["visible_count"],
                    "measured_visible_count": row["measured_visible_count"],
                    "predicted_unmeasured_visible_count": row["predicted_unmeasured_visible_count"],
                    "frontier_count_visible": row["frontier_count_visible"],
                    "path_cost_m": row["path_cost_m"],
                    "cost_s": row["cost_s"],
                    "raw_measured_occupancy_only": True,
                    "map_predict_called": False,
                    "sscnet_inference_called": False,
                }
            )
    save_json(output_dir / "topn_decisions.json", {"decisions": decisions, "flat_topn_rows": flat_rows})
    write_jsonl(output_dir / "topn_decisions.jsonl", flat_rows)
    write_csv(output_dir / "topn_decisions.csv", flat_rows)
    lines = [
        f"`{row['start_index']:02d}` rank `{row['rank']:02d}` score `{row['score']:.4f}` gain `{row['gain_exp']:.1f}` path `{row['path_cost_m']:.2f}` grid `{row['grid']}`"
        for row in flat_rows[: max(1, min(80, len(flat_rows)))]
    ]
    write_text(output_dir / "topn_decisions.md", markdown_list("Top-N Decisions", lines))


def write_gate_reports(output_dir: Path) -> dict[str, Any]:
    reports = {
        "no_map_predict_report": {
            "map_predict_called": False,
            "sscnet_inference_called": False,
            "prediction_npz_created": False,
            "prediction_written_to_observed_state": False,
            "prediction_layer_used_for_scoring": "EmptyPredictionLayer only",
        },
        "no_rl_report": {
            "rl_run": False,
            "gdpo_run": False,
            "ppo_run": False,
            "behavior_cloning_run": False,
            "imitation_learning_training_run": False,
            "policy_checkpoint_created": False,
            "checkpoint_modified": False,
        },
        "no_continuous_rollout_report": {
            "continuous_rollout_run": False,
            "second_action_executed": False,
            "third_action_executed": False,
            "open_ended_loop_run": False,
            "exactly_one_action_per_start": True,
        },
    }
    for stem, report in reports.items():
        report["stage"] = STAGE
        save_json(output_dir / f"{stem}.json", report)
        write_text(output_dir / f"{stem}.md", markdown_table(stem.replace("_", " ").title(), report))
    combined = {
        "stage": STAGE,
        "gates_closed": True,
        "no_map_predict": reports["no_map_predict_report"],
        "no_rl": reports["no_rl_report"],
        "no_continuous_rollout": reports["no_continuous_rollout_report"],
    }
    save_json(output_dir / "gates_closed_report.json", combined)
    write_text(output_dir / "gates_closed_report.md", markdown_table("Gates Closed Report", combined))
    return combined


def write_html_index(
    output_dir: Path,
    summary: dict[str, Any],
    video_report: dict[str, Any],
    decisions: list[dict[str, Any]],
) -> None:
    image_names = [
        "expert_actions_topdown.png",
        "observed_state_source_topdown.png",
        "observed_state_final_topdown.png",
        "action_rgb_grid.png",
        "action_depth_grid.png",
        "action_closeup_contact_sheet.png",
        "input_validation_camera_poses_topdown.png",
        "input_inspection_camera_poses_topdown.png",
    ]
    figures = "\n".join(
        f'<figure><img src="{html.escape(name)}" width="380"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in image_names
        if (output_dir / name).is_file()
    )
    if video_report.get("mp4_created") and (output_dir / "expert_action_flythrough.mp4").is_file():
        video_html = '<video controls width="720" src="expert_action_flythrough.mp4"></video>'
    else:
        video_html = '<p><a href="expert_action_flythrough_frames/">Fallback flythrough frames</a></p>'
    start_rows = "\n".join(
        "<li>"
        f"<code>{decision['start_index']:02d}</code> "
        f"{html.escape(decision['start_name'])}: score "
        f"<code>{decision['selected']['score']:.4f}</code>, gain "
        f"<code>{decision['selected']['gain_exp']:.1f}</code>, action "
        f"<code>{html.escape(str(decision['action_pose']['position']))}</code>"
        "</li>"
        for decision in decisions
    )
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.7 Measured-Only Expert Pilot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #17202a; background: #f7f8fa; }}
    figure {{ display: inline-block; margin: 10px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 380px; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.7 Measured-Only Expert Pilot</h1>
  <p>Fixed USD: <code>{html.escape(summary['fixed_usd'])}</code></p>
  <p>Source observed_state: <code>{html.escape(summary['source_observed_state'])}</code></p>
  <p>Samples: <code>{summary['sample_count']}</code>; Isaac startups: <code>{summary['isaac_headless_startup_count']}</code>; capture count: <code>{summary['capture_count']}</code>.</p>
  <p>map_predict: <code>false</code>; SSCNet: <code>false</code>; RL/GDPO/PPO/BC/IL training: <code>false</code>; continuous rollout: <code>false</code>.</p>
  <h2>Selected Actions</h2>
  <ul>{start_rows}</ul>
  <h2>Visuals</h2>
  {figures}
  <h2>Flythrough</h2>
  {video_html}
  <h2>Dataset</h2>
  <p><a href="expert_dataset_manifest.json">expert_dataset_manifest.json</a></p>
  <p><a href="expert_transitions.jsonl">expert_transitions.jsonl</a></p>
  <p><a href="topn_decisions.csv">topn_decisions.csv</a></p>
</body>
</html>"""
    write_text(output_dir / "expert_pilot_index.html", body)


def copy_input_visual_refs(stage66c_dir: Path, output_dir: Path) -> None:
    mapping = {
        "corrected_validation_camera_poses_topdown.png": "input_validation_camera_poses_topdown.png",
        "corrected_inspection_camera_poses_topdown.png": "input_inspection_camera_poses_topdown.png",
        "corrected_start_variants_topdown.png": "input_start_variants_topdown.png",
    }
    for src_name, dst_name in mapping.items():
        src = stage66c_dir / src_name
        if src.is_file():
            Image.open(src).save(output_dir / dst_name)


def write_input_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    starts: list[dict[str, Any]],
    validation_manifest: dict[str, Any],
    inspection_manifest: dict[str, Any],
    scene_metadata: dict[str, Any],
    observed_summary: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "stage": STAGE,
        "created_at_utc": utc_now(),
        "fixed_usd": str(Path(args.fixed_usd).resolve()),
        "fixed_usd_sha256_before": sha256_file(args.fixed_usd),
        "source_observed_state": str(Path(args.source_observed_state).resolve()),
        "source_observed_state_sha256_before": sha256_file(args.source_observed_state),
        "stage4a66c_dir": str(Path(args.stage4a66c_dir).resolve()),
        "start_variant_count": len(starts),
        "start_variants": starts,
        "validation_pose_count": validation_manifest.get("pose_count", len(validation_manifest.get("poses", []))),
        "inspection_pose_count": inspection_manifest.get("pose_count", len(inspection_manifest.get("poses", []))),
        "validation_pose_manifest": str(Path(args.stage4a66c_dir) / "validation_pose_manifest.json"),
        "inspection_pose_manifest": str(Path(args.stage4a66c_dir) / "inspection_pose_manifest.json"),
        "scene_metadata_path": str(Path(args.stage4a66c_dir) / "scene_metadata.json"),
        "scene_metadata": scene_metadata,
        "observed_summary": observed_summary,
        "user_confirmed_visuals_from_6_6d": True,
        "no_usd_edit": True,
        "no_asset_download": True,
        "no_procedural_fallback": True,
        "larger_complex_scene_v1_touched": False,
        "source_observed_state_modified": False,
    }
    save_json(output_dir / "input_manifest.json", manifest)
    write_text(output_dir / "input_manifest.md", markdown_table("Input Manifest", manifest))
    return manifest


def write_safety_audit(
    args: argparse.Namespace,
    output_dir: Path,
    input_manifest: dict[str, Any],
    checkpoint_before: dict[str, Any],
    checkpoint_after: dict[str, Any],
    isaac_result: dict[str, Any],
    dataset_manifest: dict[str, Any],
    gate_report: dict[str, Any],
) -> dict[str, Any]:
    fixed_usd = Path(args.fixed_usd)
    source_observed = Path(args.source_observed_state)
    scene_factory_path = WORKSPACE / "sim_explorer/scene_factory.py"
    audit = {
        "stage": STAGE,
        "created_at_utc": utc_now(),
        "fixed_usd": str(fixed_usd),
        "fixed_usd_sha256_before": input_manifest["fixed_usd_sha256_before"],
        "fixed_usd_sha256_after": sha256_file(fixed_usd),
        "fixed_usd_modified": input_manifest["fixed_usd_sha256_before"] != sha256_file(fixed_usd),
        "source_observed_state": str(source_observed),
        "source_observed_state_sha256_before": input_manifest["source_observed_state_sha256_before"],
        "source_observed_state_sha256_after": sha256_file(source_observed),
        "source_observed_state_modified": input_manifest["source_observed_state_sha256_before"] != sha256_file(source_observed),
        "scene_factory_path": str(scene_factory_path),
        "scene_factory_sha256": sha256_file(scene_factory_path),
        "scene_factory_modified_by_this_stage": False,
        "observed_state_final": dataset_manifest["observed_state_final"],
        "observed_state_final_is_new_output": True,
        "checkpoint_before": checkpoint_before,
        "checkpoint_after": checkpoint_after,
        "checkpoint_modified": checkpoint_before != checkpoint_after,
        "isaac_headless_startup_count": isaac_result["isaac_headless_startup_count"],
        "capture_count": isaac_result["capture_count"],
        "start_variant_count": dataset_manifest["sample_count"],
        "exactly_one_headless_capture_per_start": isaac_result["capture_count"] == dataset_manifest["sample_count"],
        "exactly_one_action_per_start": dataset_manifest["one_action_per_start"],
        "second_action_executed": False,
        "third_action_executed": False,
        "continuous_rollout_executed": False,
        "map_predict_called": False,
        "sscnet_inference_called": False,
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "usd_written": False,
        "dependency_written": False,
        "asset_download_attempted": False,
        "procedural_fallback_generated": False,
        "larger_complex_scene_v1_touched": False,
        "gate_report": gate_report,
    }
    audit["passed"] = bool(
        not audit["fixed_usd_modified"]
        and not audit["source_observed_state_modified"]
        and not audit["checkpoint_modified"]
        and audit["isaac_headless_startup_count"] == 1
        and audit["exactly_one_headless_capture_per_start"]
        and audit["exactly_one_action_per_start"]
        and not audit["map_predict_called"]
        and not audit["sscnet_inference_called"]
        and not audit["rl_training_run"]
        and not audit["continuous_rollout_executed"]
    )
    save_json(output_dir / "safety_audit.json", audit)
    write_text(output_dir / "safety_audit.md", markdown_table("Safety Audit", audit))
    return audit


def verify_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    decisions: list[dict[str, Any]],
    capture_records: list[dict[str, Any]],
    dataset_manifest: dict[str, Any],
    video_report: dict[str, Any],
    safety_audit: dict[str, Any],
) -> dict[str, Any]:
    dataset_path = Path(dataset_manifest["dataset_npz"])
    observed_final_path = Path(dataset_manifest["observed_state_final"])
    checks: dict[str, Any] = {
        "stage": STAGE,
        "py_compile_checked_externally": True,
        "expert_dataset_npz_exists": dataset_path.is_file(),
        "expert_dataset_manifest_exists": (output_dir / "expert_dataset_manifest.json").is_file(),
        "expert_transitions_jsonl_exists": (output_dir / "expert_transitions.jsonl").is_file(),
        "observed_state_final_exists": observed_final_path.is_file(),
        "topdown_exists": (output_dir / "expert_actions_topdown.png").is_file()
        and (output_dir / "observed_state_final_topdown.png").is_file(),
        "closeups_exist": (output_dir / "action_closeup_contact_sheet.png").is_file(),
        "html_exists": (output_dir / "expert_pilot_index.html").is_file(),
        "mp4_exists": (output_dir / "expert_action_flythrough.mp4").is_file(),
        "mp4_created": bool(video_report.get("mp4_created", False)),
        "safety_audit_passed": bool(safety_audit.get("passed", False)),
        "capture_count": len(capture_records),
        "expected_capture_count": len(decisions),
        "one_capture_per_start": len(capture_records) == len(decisions),
        "rgb_capture_valid_count": sum(1 for record in capture_records if record["rgb_stats"]["nonblank"]),
        "depth_capture_valid_count": sum(1 for record in capture_records if record["depth_stats"]["has_positive_finite_depth"]),
        "map_predict_called": False,
        "sscnet_inference_called": False,
        "rl_training_run": False,
        "continuous_rollout_executed": False,
    }
    if dataset_path.is_file():
        with np.load(dataset_path, allow_pickle=False) as data:
            checks["dataset_keys"] = sorted(data.files)
            checks["dataset_sample_count"] = int(data["action_world"].shape[0])
            checks["dataset_pre_shape"] = [int(v) for v in data["pre_observed_states"].shape]
            checks["dataset_post_shape"] = [int(v) for v in data["post_observed_states"].shape]
            checks["dataset_selected_scores_finite"] = bool(np.all(np.isfinite(data["selected_scores"])))
    if observed_final_path.is_file():
        final_state = np.load(observed_final_path)
        checks["observed_state_final_validation"] = validate_observed_state(final_state, "observed_state_final")
    checks["passed"] = bool(
        checks["expert_dataset_npz_exists"]
        and checks["expert_dataset_manifest_exists"]
        and checks["expert_transitions_jsonl_exists"]
        and checks["observed_state_final_exists"]
        and checks["topdown_exists"]
        and checks["closeups_exist"]
        and checks["html_exists"]
        and checks["mp4_exists"]
        and checks["mp4_created"]
        and checks["safety_audit_passed"]
        and checks["one_capture_per_start"]
        and checks["rgb_capture_valid_count"] == len(decisions)
        and checks["depth_capture_valid_count"] == len(decisions)
        and not checks["map_predict_called"]
        and not checks["sscnet_inference_called"]
        and not checks["rl_training_run"]
        and not checks["continuous_rollout_executed"]
        and bool(checks.get("dataset_selected_scores_finite", False))
    )
    save_json(output_dir / "dataset_integrity_report.json", checks)
    write_text(output_dir / "dataset_integrity_report.md", markdown_table("Dataset Integrity Report", checks))
    return checks


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    input_manifest: dict[str, Any],
    hardware: dict[str, Any],
    decisions: list[dict[str, Any]],
    isaac_result: dict[str, Any],
    dataset_manifest: dict[str, Any],
    video_report: dict[str, Any],
    safety_audit: dict[str, Any],
    integrity: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    summary = {
        "stage": STAGE,
        "completed": bool(integrity.get("passed", False)),
        "created_at_utc": utc_now(),
        "elapsed_seconds": float(elapsed_s),
        "fixed_usd": input_manifest["fixed_usd"],
        "source_observed_state": input_manifest["source_observed_state"],
        "source_observed_state_sha256": input_manifest["source_observed_state_sha256_before"],
        "observed_state_final": dataset_manifest["observed_state_final"],
        "expert_dataset_npz": dataset_manifest["dataset_npz"],
        "expert_dataset_manifest": str(output_dir / "expert_dataset_manifest.json"),
        "sample_count": dataset_manifest["sample_count"],
        "start_variant_count": len(decisions),
        "top_n": int(args.top_n),
        "isaac_headless_startup_count": isaac_result["isaac_headless_startup_count"],
        "capture_count": isaac_result["capture_count"],
        "exactly_one_headless_capture_per_start": isaac_result["capture_count"] == len(decisions),
        "exactly_one_action_per_start": dataset_manifest["one_action_per_start"],
        "second_action_executed": False,
        "third_action_executed": False,
        "continuous_rollout_executed": False,
        "measured_only_frontier_scoring": True,
        "raw_measured_occupancy_only": True,
        "prediction_layer": "EmptyPredictionLayer",
        "map_predict_called": False,
        "sscnet_inference_called": False,
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "usd_modified": safety_audit["fixed_usd_modified"],
        "source_observed_state_modified": safety_audit["source_observed_state_modified"],
        "checkpoint_modified": safety_audit["checkpoint_modified"],
        "safety_audit_passed": safety_audit["passed"],
        "integrity_passed": integrity["passed"],
        "hardware": hardware,
        "topn_decisions": str(output_dir / "topn_decisions.json"),
        "expert_transitions_jsonl": str(output_dir / "expert_transitions.jsonl"),
        "html": str(output_dir / "expert_pilot_index.html"),
        "mp4": str(output_dir / "expert_action_flythrough.mp4"),
        "video_report": video_report,
        "selected_actions": [
            {
                "start_index": decision["start_index"],
                "start_name": decision["start_name"],
                "action_position": decision["action_pose"]["position"],
                "action_yaw_rad": decision["action_pose"]["yaw_rad"],
                "score": decision["selected"]["score"],
                "gain_exp": decision["selected"]["gain_exp"],
                "path_cost_m": decision["selected"]["path_cost_m"],
            }
            for decision in decisions
        ],
    }
    save_json(output_dir / "stage4a67_measured_only_expert_pilot_summary.json", summary)
    lines = [
        "# Stage 4A-6.7 Measured-Only Expert Pilot Summary",
        "",
        f"- completed: `{summary['completed']}`",
        f"- fixed USD: `{summary['fixed_usd']}`",
        f"- source observed_state: `{summary['source_observed_state']}`",
        f"- expert dataset: `{summary['expert_dataset_npz']}`",
        f"- samples: `{summary['sample_count']}`",
        f"- Isaac startup count: `{summary['isaac_headless_startup_count']}`",
        f"- capture count: `{summary['capture_count']}`",
        f"- exactly one capture/action per start: `{summary['exactly_one_headless_capture_per_start']}` / `{summary['exactly_one_action_per_start']}`",
        f"- map_predict / SSCNet / RL: `{summary['map_predict_called']}` / `{summary['sscnet_inference_called']}` / `{summary['rl_training_run']}`",
        f"- continuous rollout / second action / third action: `{summary['continuous_rollout_executed']}` / `{summary['second_action_executed']}` / `{summary['third_action_executed']}`",
        f"- safety audit passed: `{summary['safety_audit_passed']}`",
        f"- integrity passed: `{summary['integrity_passed']}`",
        f"- HTML: `{summary['html']}`",
        f"- MP4: `{summary['mp4']}`",
    ]
    write_text(output_dir / "stage4a67_measured_only_expert_pilot_summary.md", "\n".join(lines))
    return summary


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a66c_dir", default=str(DEFAULT_STAGE66C_DIR))
    parser.add_argument("--fixed_usd", default=str(DEFAULT_FIXED_USD))
    parser.add_argument("--source_observed_state", default=str(DEFAULT_SOURCE_OBSERVED))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument("--max_depth", type=float, default=26.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--pixel_stride", type=int, default=5)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--max_scored_candidates_per_start", type=int, default=96)
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
    parser.add_argument("--checkpoint_for_audit", default=str(DEFAULT_FULL_CHECKPOINT))
    parser.add_argument("--allow_existing_output_dir", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "headless"):
        args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    return args, AppLauncher


def main() -> None:
    total_start = time.perf_counter()
    args, app_launcher_cls = parse_args()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(args.allow_existing_output_dir):
        raise RuntimeError(f"output_dir already exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    log_event(output_dir, events, "stage_begin", run_output_dir=str(output_dir))

    stage66c_dir = Path(args.stage4a66c_dir).resolve()
    fixed_usd = Path(args.fixed_usd).resolve()
    source_observed_path = Path(args.source_observed_state).resolve()
    if not stage66c_dir.is_dir():
        raise FileNotFoundError(stage66c_dir)
    if not fixed_usd.is_file():
        raise FileNotFoundError(fixed_usd)
    if not source_observed_path.is_file():
        raise FileNotFoundError(source_observed_path)

    source_observed = np.load(source_observed_path)
    source_observed_summary = validate_observed_state(source_observed, "stage4a66c_source_observed_state")
    observed_summary = read_json(stage66c_dir / "observed_summary.json")
    scene_metadata = read_json(stage66c_dir / "scene_metadata.json")
    starts = read_json(stage66c_dir / "start_variants.json")
    validation_manifest = read_json(stage66c_dir / "validation_pose_manifest.json")
    inspection_manifest = read_json(stage66c_dir / "inspection_pose_manifest.json")
    bounds = observed_summary.get("chosen_bounds") or scene_metadata.get("map_bounds")
    voxel_size = float(observed_summary.get("voxel_size", 0.1))
    if not bounds:
        raise ValueError("Could not resolve map bounds from Stage 4A-6.6c observed_summary/scene_metadata")
    if list(source_observed.shape) != list(observed_summary.get("shape", source_observed.shape)):
        raise ValueError("source observed_state shape does not match Stage 4A-6.6c observed summary")
    log_event(
        output_dir,
        events,
        "inputs_loaded",
        start_count=len(starts),
        validation_pose_count=validation_manifest.get("pose_count"),
        inspection_pose_count=inspection_manifest.get("pose_count"),
        observed_shape=list(source_observed.shape),
    )

    hardware = hardware_report(int(args.max_workers), str(getattr(args, "device", "unknown")))
    save_json(output_dir / "hardware_report.json", hardware)
    write_text(output_dir / "hardware_report.md", markdown_table("Hardware Report", hardware))
    checkpoint_before = checkpoint_manifest(WORKSPACE / "checkpoints", Path(args.checkpoint_for_audit).resolve())
    input_manifest = write_input_manifest(
        args,
        output_dir,
        starts,
        validation_manifest,
        inspection_manifest,
        scene_metadata,
        {**observed_summary, "source_validation": source_observed_summary},
    )
    copy_input_visual_refs(stage66c_dir, output_dir)

    traversability = build_traversability_grid(
        source_observed,
        voxel_size=voxel_size,
        robot_height_m=float(args.robot_height_m),
        clearance_height_m=float(args.clearance_height_m),
        robot_radius_m=float(args.robot_radius_m),
    )
    frontier_mask = frontier_adjacent_free_xy_mask(source_observed)
    traversal_summary = {
        "stage": STAGE,
        "traversability": summarize_traversability(traversability),
        "frontier_adjacent_free_xy_count": int(np.count_nonzero(frontier_mask)),
        "measured_only": True,
        "raw_measured_occupancy_only": True,
    }
    save_json(output_dir / "frontier_sampling_context.json", traversal_summary)
    write_text(output_dir / "frontier_sampling_context.md", markdown_table("Frontier Sampling Context", traversal_summary))

    yaw_priors = inspection_yaw_priors(starts, inspection_manifest.get("poses", []))
    decisions: list[dict[str, Any]] = []
    for start in starts:
        decision = score_start_variant(
            source_observed,
            bounds,
            voxel_size,
            traversability,
            frontier_mask,
            start,
            yaw_priors,
            args,
        )
        decisions.append(decision)
        save_decision_topdown(
            output_dir / f"topdown/start{int(start['index']):03d}_topn_decision.png",
            source_observed,
            bounds,
            voxel_size,
            decision,
        )
        log_event(
            output_dir,
            events,
            "topn_decision_complete",
            start_index=int(start["index"]),
            selected_score=decision["selected"]["score"],
            top_n_count=decision["top_n_count"],
        )
    write_topn_outputs(output_dir, decisions)
    save_observed_topdown(output_dir / "observed_state_source_topdown.png", source_observed, "Stage 4A-6.6c source observed_state")
    save_all_actions_topdown(output_dir / "expert_actions_topdown.png", source_observed, bounds, decisions)

    action_poses = [decision["action_pose"] for decision in decisions]
    save_json(output_dir / "selected_action_poses.json", {"poses": action_poses})
    write_text(
        output_dir / "selected_action_poses.md",
        markdown_list(
            "Selected Action Poses",
            [
                f"`{pose['index']:02d}` `{pose['name']}` pos `{pose['position']}` yaw `{pose['yaw_rad']:.4f}`"
                for pose in action_poses
            ],
        ),
    )

    isaac_result = run_one_isaac_startup(args, app_launcher_cls, output_dir, fixed_usd, action_poses, events)
    save_json(output_dir / "headless_action_capture_manifest.json", isaac_result)
    write_text(output_dir / "headless_action_capture_manifest.md", markdown_table("Headless Action Capture Manifest", isaac_result))
    capture_records = isaac_result["capture_records"]
    save_capture_grids(output_dir, capture_records)
    make_closeup_contact_sheet(output_dir, capture_records)
    video_report = make_mp4(output_dir, capture_records)

    final_state, transitions, dataset_manifest = build_transitions_and_dataset(
        args,
        output_dir,
        source_observed,
        decisions,
        capture_records,
        isaac_result["camera_info"],
        bounds,
        voxel_size,
    )
    save_observed_topdown(output_dir / "observed_state_final_topdown.png", final_state, "Stage 4A-6.7 measured-only observed_state final")

    capture_validation = {
        "stage": STAGE,
        "action_capture_count": len(capture_records),
        "expected_start_count": len(starts),
        "one_capture_per_start": len(capture_records) == len(starts),
        "nonblank_rgb_count": sum(1 for row in capture_records if row["rgb_stats"]["nonblank"]),
        "finite_positive_depth_count": sum(1 for row in capture_records if row["depth_stats"]["has_positive_finite_depth"]),
        "failed_views": [
            row["index"]
            for row in capture_records
            if not row["rgb_stats"]["nonblank"] or not row["depth_stats"]["has_positive_finite_depth"]
        ],
        "records": capture_records,
    }
    save_json(output_dir / "capture_validation.json", capture_validation)
    write_text(output_dir / "capture_validation.md", markdown_table("Capture Validation", capture_validation))

    gate_report = write_gate_reports(output_dir)
    checkpoint_after = checkpoint_manifest(WORKSPACE / "checkpoints", Path(args.checkpoint_for_audit).resolve())
    safety_audit = write_safety_audit(
        args,
        output_dir,
        input_manifest,
        checkpoint_before,
        checkpoint_after,
        isaac_result,
        dataset_manifest,
        gate_report,
    )
    integrity = verify_outputs(args, output_dir, decisions, capture_records, dataset_manifest, video_report, safety_audit)
    summary = write_summary(
        output_dir,
        args,
        input_manifest,
        hardware,
        decisions,
        isaac_result,
        dataset_manifest,
        video_report,
        safety_audit,
        integrity,
        float(time.perf_counter() - total_start),
    )
    write_html_index(output_dir, summary, video_report, decisions)
    # Re-run the HTML check after writing the page.
    integrity = verify_outputs(args, output_dir, decisions, capture_records, dataset_manifest, video_report, safety_audit)
    summary["integrity_passed"] = bool(integrity["passed"])
    summary["completed"] = bool(integrity["passed"])
    save_json(output_dir / "stage4a67_measured_only_expert_pilot_summary.json", summary)
    log_event(output_dir, events, "stage_complete", completed=bool(summary["completed"]), integrity_passed=bool(integrity["passed"]))
    if not bool(summary["completed"]):
        raise RuntimeError(f"{STAGE} completed with failing integrity checks: {integrity}")


if __name__ == "__main__":
    main()
