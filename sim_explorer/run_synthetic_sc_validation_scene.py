#!/usr/bin/env python3
"""Stage 4A-6.5aa controlled synthetic SC validation scene smoke.

This runner creates one deterministic hidden-room frontier scene, captures one
synthetic RGB/depth frame, builds a measured-only observed_state, writes a
read-only oracle prediction layer, optionally tries one real map_predict pass,
and evaluates source-protected one-step mini-RRT tree decisions. It never
executes a selected action, runs a rollout/two-frame loop, writes prediction
into observed_state, or uses prediction for traversability/collision/ray
blocking.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image

from depth_to_voxel import FREE, OCCUPIED, UNKNOWN, summarize_observed_grid
from isaac_map_predictor import IsaacMapPredictor
from offline_mini_rrt_tree import (
    ROOT_ID,
    build_mini_rrt_tree,
    segment_path_to_root,
    segment_record,
    sha256_file,
    to_jsonable,
)
from scene_factory import build_synthetic_hidden_room_frontier_scene
from sim_paper_expert import (
    EmptyPredictionLayer,
    SimCandidateView,
    grid_to_world,
    normalize_bounds,
    raycast_visible_voxels_observed,
    world_to_grid,
)
from sim_prediction_layer import SimPredictionLayer


DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65aa_synthetic_sc_validation"
)
DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)
EPS = 1.0e-9


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(to_jsonable(value), sort_keys=True)
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


def parse_ints(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not values:
        raise ValueError("at least one seed is required")
    return values


def bounds_from_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    return normalize_bounds(
        {
            "x": [float(args.x_min), float(args.x_max)],
            "y": [float(args.y_min), float(args.y_max)],
            "z": [float(args.z_min), float(args.z_max)],
        }
    )


def grid_shape(bounds: dict[str, tuple[float, float]], voxel_size: float) -> tuple[int, int, int]:
    return tuple(
        int(round((float(bounds[axis][1]) - float(bounds[axis][0])) / float(voxel_size)))
        for axis in ("x", "y", "z")
    )


def region_to_slices(
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    shape: tuple[int, int, int],
    region: dict[str, list[float]],
) -> tuple[slice, slice, slice]:
    slices = []
    for axis, limit in zip(("x", "y", "z"), shape):
        lo, hi = sorted((float(region[axis][0]), float(region[axis][1])))
        start = int(math.floor((lo - float(bounds[axis][0])) / float(voxel_size)))
        stop = int(math.ceil((hi - float(bounds[axis][0])) / float(voxel_size)))
        slices.append(slice(max(0, start), min(int(limit), stop)))
    return tuple(slices)  # type: ignore[return-value]


def fill_region(
    array: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    region: dict[str, list[float]],
    value: Any,
    *,
    only_unknown: bool = False,
) -> None:
    slc = region_to_slices(bounds, voxel_size, tuple(int(v) for v in array.shape), region)
    if only_unknown:
        view = array[slc]
        view[view == UNKNOWN] = value
    else:
        array[slc] = value


def box_to_region(box: dict[str, Any]) -> dict[str, list[float]]:
    pos = [float(v) for v in box["position"]]
    size = [float(v) for v in box["size"]]
    return {
        "x": [pos[0] - 0.5 * size[0], pos[0] + 0.5 * size[0]],
        "y": [pos[1] - 0.5 * size[1], pos[1] + 0.5 * size[1]],
        "z": [pos[2] - 0.5 * size[2], pos[2] + 0.5 * size[2]],
    }


def region_mask(
    shape: tuple[int, int, int],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    region: dict[str, list[float]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[region_to_slices(bounds, voxel_size, shape, region)] = True
    return mask


def make_observed_state(
    scene_metadata: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = grid_shape(bounds, voxel_size)
    observed = np.full(shape, UNKNOWN, dtype=np.int8)
    regions = scene_metadata["diagnostic_regions"]

    for name in ("measured_start_room", "measured_side_frontier", "measured_doorway_corridor"):
        fill_region(observed, bounds, voxel_size, regions[name], int(FREE))

    skip_wall_names = {
        "hidden_room_south_wall",
        "hidden_room_north_wall",
        "hidden_room_east_wall",
    }
    occupied_names: list[str] = []
    for wall in scene_metadata["walls"]:
        if wall["name"] in skip_wall_names:
            continue
        fill_region(observed, bounds, voxel_size, box_to_region(wall), int(OCCUPIED))
        occupied_names.append(str(wall["name"]))
    for obstacle in scene_metadata["obstacles"]:
        if str(obstacle.get("category", "")).startswith("oracle_hidden"):
            continue
        fill_region(observed, bounds, voxel_size, box_to_region(obstacle), int(OCCUPIED))
        occupied_names.append(str(obstacle["name"]))

    pose = scene_metadata["camera_poses"][0]
    root_grid = world_to_grid(pose["position"], bounds, voxel_size, shape=shape, clip=True)
    for di in range(-2, 3):
        for dj in range(-2, 3):
            for dk in range(-1, 2):
                idx = (
                    int(np.clip(root_grid[0] + di, 0, shape[0] - 1)),
                    int(np.clip(root_grid[1] + dj, 0, shape[1] - 1)),
                    int(np.clip(root_grid[2] + dk, 0, shape[2] - 1)),
                )
                observed[idx] = FREE

    summary = summarize_observed_grid(observed)
    summary.update(
        {
            "construction": "measured-only synthetic ray-visible state",
            "measured_regions": [
                "measured_start_room",
                "measured_side_frontier",
                "measured_doorway_corridor",
            ],
            "observed_occupied_specs": occupied_names,
            "prediction_writeback": False,
            "target_or_ground_truth_used_for_planning": False,
        }
    )
    return observed, summary


def camera_info(width: int, height: int, max_depth: float) -> dict[str, Any]:
    horizontal_fov_deg = 90.0
    fx = (float(width) * 0.5) / math.tan(math.radians(horizontal_fov_deg) * 0.5)
    fy = fx
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return {
        "sensor_api_depth_key": "synthetic_distance_to_image_plane",
        "depth_units": "meters",
        "width": int(width),
        "height": int(height),
        "max_depth": float(max_depth),
        "near_depth": 0.05,
        "horizontal_fov_deg": float(horizontal_fov_deg),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "intrinsic_matrix": [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
        "capture_backend": "scripted_scene_factory_depth_raycast",
    }


def camera_basis(yaw_rad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.array([math.cos(yaw_rad), math.sin(yaw_rad), 0.0], dtype=np.float64)
    right = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return forward, right, up


def ray_box_intersection(origin: np.ndarray, direction: np.ndarray, box: dict[str, Any]) -> float | None:
    pos = np.asarray(box["position"], dtype=np.float64)
    size = np.asarray(box["size"], dtype=np.float64)
    mins = pos - 0.5 * size
    maxs = pos + 0.5 * size
    tmin = -math.inf
    tmax = math.inf
    for axis in range(3):
        d = float(direction[axis])
        if abs(d) <= 1.0e-12:
            if float(origin[axis]) < mins[axis] or float(origin[axis]) > maxs[axis]:
                return None
            continue
        t1 = (mins[axis] - float(origin[axis])) / d
        t2 = (maxs[axis] - float(origin[axis])) / d
        near = min(t1, t2)
        far = max(t1, t2)
        tmin = max(tmin, near)
        tmax = min(tmax, far)
        if tmax < tmin:
            return None
    if tmax <= 0.0:
        return None
    return float(tmin if tmin > 0.0 else tmax)


def render_synthetic_frame(
    scene_metadata: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    pose = dict(scene_metadata["camera_poses"][0])
    width = int(args.camera_width)
    height = int(args.camera_height)
    info = camera_info(width, height, float(args.max_depth))
    origin = np.asarray(pose["position"], dtype=np.float64)
    forward, right, up = camera_basis(float(pose["yaw_rad"]))
    fx, fy, cx, cy = float(info["fx"]), float(info["fy"]), float(info["cx"]), float(info["cy"])
    boxes = list(scene_metadata["walls"]) + list(scene_metadata["obstacles"])
    depth = np.full((height, width), float(args.max_depth), dtype=np.float32)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    sky = np.array([34, 42, 52], dtype=np.uint8)
    floor_color = np.array([152, 158, 142], dtype=np.uint8)
    rgb[:, :] = sky

    for v in range(height):
        y_norm = (float(v) + 0.5 - cy) / fy
        for u in range(width):
            x_norm = (float(u) + 0.5 - cx) / fx
            direction = forward + x_norm * right - y_norm * up
            best_t = float(args.max_depth)
            best_color = sky
            if float(direction[2]) < -1.0e-9:
                floor_t = (0.0 - float(origin[2])) / float(direction[2])
                if 0.0 < floor_t < best_t:
                    best_t = float(floor_t)
                    best_color = floor_color
            for box in boxes:
                t = ray_box_intersection(origin, direction, box)
                if t is not None and 0.05 < float(t) < best_t:
                    best_t = float(t)
                    color = np.asarray(box.get("color", [0.5, 0.5, 0.5]), dtype=np.float64)
                    best_color = np.clip(color * 255.0, 0, 255).astype(np.uint8)
            depth[v, u] = np.float32(best_t)
            shade = max(0.45, 1.0 - 0.06 * best_t)
            rgb[v, u] = np.clip(best_color.astype(np.float64) * shade, 0, 255).astype(np.uint8)

    stats = {
        "backend": "scripted_scene_factory_depth_raycast",
        "depth_shape": [int(v) for v in depth.shape],
        "rgb_shape": [int(v) for v in rgb.shape],
        "depth_min": float(np.min(depth)),
        "depth_max": float(np.max(depth)),
        "depth_mean": float(np.mean(depth)),
        "rgb_min": int(np.min(rgb)),
        "rgb_max": int(np.max(rgb)),
        "rgb_mean": float(np.mean(rgb)),
        "frames_captured": 1,
    }
    return depth, rgb, pose, info | {"render_stats": stats}


def make_oracle_prediction(
    output_path: Path,
    observed_state: np.ndarray,
    scene_metadata: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    alignment_convention: str,
) -> dict[str, Any]:
    shape = tuple(int(v) for v in observed_state.shape)
    pred_class = np.zeros(shape, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)
    occupied_prob = np.full(shape, 0.5, dtype=np.float32)
    free_prob = np.full(shape, 0.5, dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    hidden_region = scene_metadata["diagnostic_regions"]["oracle_hidden_room"]

    hidden_mask = region_mask(shape, bounds, voxel_size, hidden_region)
    valid[hidden_mask] = True
    confidence[hidden_mask] = 0.95
    free_prob[hidden_mask] = 0.90
    occupied_prob[hidden_mask] = 0.10
    pred_class[hidden_mask] = 0

    occupied_specs: list[str] = []
    for wall in scene_metadata["walls"]:
        if str(wall["name"]).startswith("hidden_room"):
            fill_region(pred_class, bounds, voxel_size, box_to_region(wall), 1)
            fill_region(confidence, bounds, voxel_size, box_to_region(wall), 0.95)
            fill_region(occupied_prob, bounds, voxel_size, box_to_region(wall), 0.90)
            fill_region(free_prob, bounds, voxel_size, box_to_region(wall), 0.10)
            fill_region(valid, bounds, voxel_size, box_to_region(wall), 1)
            occupied_specs.append(str(wall["name"]))
    for obstacle in scene_metadata["obstacles"]:
        if str(obstacle.get("category", "")).startswith("oracle_hidden"):
            fill_region(pred_class, bounds, voxel_size, box_to_region(obstacle), 1)
            fill_region(confidence, bounds, voxel_size, box_to_region(obstacle), 0.95)
            fill_region(occupied_prob, bounds, voxel_size, box_to_region(obstacle), 0.90)
            fill_region(free_prob, bounds, voxel_size, box_to_region(obstacle), 0.10)
            fill_region(valid, bounds, voxel_size, box_to_region(obstacle), 1)
            occupied_specs.append(str(obstacle["name"]))

    valid &= observed_state == UNKNOWN
    confidence[~valid] = 0.0
    occupied_prob[~valid] = 0.5
    free_prob[~valid] = 0.5
    pred_class[~valid] = 0
    occupied = valid & (occupied_prob >= 0.5)
    free = valid & (free_prob >= 0.5)

    np.savez_compressed(
        output_path,
        global_pred_class=pred_class,
        global_confidence=confidence,
        global_occupied_prob=occupied_prob,
        global_free_prob=free_prob,
        global_prediction_valid=valid,
        alignment_convention=np.array(str(alignment_convention)),
        oracle_prediction=np.array(True),
        diagnostic_only=np.array(True),
        not_map_predict=np.array(True),
        not_ground_truth_runtime_planning=np.array(True),
        prediction_writeback=np.array(False),
        prediction_used_for_traversability=np.array(False),
        prediction_used_for_collision=np.array(False),
        prediction_blocks_rays=np.array(False),
    )
    summary = {
        "prediction_npz": str(output_path),
        "shape": [int(v) for v in shape],
        "alignment_convention": str(alignment_convention),
        "oracle_prediction": True,
        "diagnostic_only": True,
        "not_map_predict": True,
        "not_ground_truth_runtime_planning": True,
        "prediction_writeback": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_blocks_rays": False,
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "predicted_occupied_count": int(np.count_nonzero(occupied)),
        "predicted_free_count": int(np.count_nonzero(free)),
        "hidden_region_valid_count": int(np.count_nonzero(valid & hidden_mask)),
        "hidden_region_occupied_count": int(np.count_nonzero(occupied & hidden_mask)),
        "hidden_region_free_count": int(np.count_nonzero(free & hidden_mask)),
        "occupied_specs": occupied_specs,
        "confidence": {"value": 0.95, "min": 0.95 if np.count_nonzero(valid) else None},
    }
    return summary


def prediction_summary_from_npz(
    path: Path,
    observed_state: np.ndarray,
    hidden_mask: np.ndarray,
    tau: float,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occ = np.asarray(data["global_occupied_prob"], dtype=np.float32)
        free = np.asarray(data["global_free_prob"], dtype=np.float32)
        alignment = str(np.asarray(data["alignment_convention"]).item()) if "alignment_convention" in data.files else ""
    valid_tau = valid & (confidence >= float(tau)) & (observed_state == UNKNOWN)
    occ_mask = valid_tau & (occ >= 0.5)
    free_mask = valid_tau & (free >= 0.5)
    return {
        "prediction_npz": str(path),
        "shape": [int(v) for v in valid.shape],
        "alignment_convention": alignment,
        "tau": float(tau),
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "prediction_valid_tau_unmeasured_count": int(np.count_nonzero(valid_tau)),
        "predicted_occupied_count": int(np.count_nonzero(occ_mask)),
        "predicted_free_count": int(np.count_nonzero(free_mask)),
        "hidden_region_valid_tau_unmeasured_count": int(np.count_nonzero(valid_tau & hidden_mask)),
        "hidden_region_predicted_occupied_count": int(np.count_nonzero(occ_mask & hidden_mask)),
        "hidden_region_predicted_free_count": int(np.count_nonzero(free_mask & hidden_mask)),
    }


def write_prediction_md(path: Path, title: str, summary: dict[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"- prediction npz: `{summary.get('prediction_npz')}`",
        f"- shape: `{summary.get('shape')}`",
        f"- valid count: `{summary.get('prediction_valid_count')}`",
        f"- predicted occupied count: `{summary.get('predicted_occupied_count')}`",
        f"- predicted free count: `{summary.get('predicted_free_count')}`",
        f"- hidden valid count: `{summary.get('hidden_region_valid_count', summary.get('hidden_region_valid_tau_unmeasured_count'))}`",
        f"- diagnostic oracle: `{summary.get('oracle_prediction', False)}`",
        f"- prediction writeback: `{summary.get('prediction_writeback', False)}`",
        f"- traversability/collision/ray blocking: `{summary.get('prediction_used_for_traversability', False)}` / `{summary.get('prediction_used_for_collision', False)}` / `{summary.get('prediction_blocks_rays', False)}`",
    ]
    write_text(path, "\n".join(lines))


def topdown_observed(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def grid_point(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    return float(grid[0]) + 0.5, float(grid[1]) + 0.5


def plot_observed(path: Path, observed_state: np.ndarray, title: str) -> None:
    cmap = ListedColormap(["#2f343f", "#8cc7bb", "#c94a44"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.4, 6.8), constrained_layout=True)
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_scene_layout(path: Path, scene_metadata: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 7.0), constrained_layout=True)
    ax.set_aspect("equal")
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-6.0, 6.0)
    regions = scene_metadata["diagnostic_regions"]
    for name, region in regions.items():
        x0, x1 = region["x"]
        y0, y1 = region["y"]
        color = "#4f86c6" if "hidden" in name else "#82b366"
        alpha = 0.16 if "hidden" in name else 0.20
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, alpha=alpha, edgecolor=color))
        ax.text(x0 + 0.05, y1 - 0.18, name, fontsize=7, color=color)
    for box in scene_metadata["walls"]:
        region = box_to_region(box)
        x0, x1 = region["x"]
        y0, y1 = region["y"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#565b63", edgecolor="#333840"))
    for box in scene_metadata["obstacles"]:
        region = box_to_region(box)
        x0, x1 = region["x"]
        y0, y1 = region["y"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#b5654d", edgecolor="#743b2f"))
    pose = scene_metadata["camera_poses"][0]
    ax.scatter([pose["position"][0]], [pose["position"][1]], c="#111111", s=60, label="camera/root")
    ax.arrow(
        pose["position"][0],
        pose["position"][1],
        math.cos(float(pose["yaw_rad"])) * 0.8,
        math.sin(float(pose["yaw_rad"])) * 0.8,
        width=0.035,
        color="#111111",
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("synthetic_hidden_room_frontier layout")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_prediction_topdown(
    path: Path,
    prediction_npz: Path,
    observed_state: np.ndarray,
    tau: float,
    title: str,
    *,
    overlay: bool = False,
) -> None:
    with np.load(prediction_npz, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occ = np.asarray(data["global_occupied_prob"], dtype=np.float32)
        free = np.asarray(data["global_free_prob"], dtype=np.float32)
    pred = valid & (confidence >= float(tau)) & (observed_state == UNKNOWN)
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(pred & (free >= 0.5), axis=2)] = 1
    image[np.any(pred & (occ >= 0.5), axis=2)] = 2
    fig, ax = plt.subplots(figsize=(7.4, 6.8), constrained_layout=True)
    if overlay:
        ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap="Greys", alpha=0.28, interpolation="nearest")
    cmap = ListedColormap(["#20242b", "#4fba7a", "#e05a47"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, alpha=0.85, interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def min_mean_max(values: list[float]) -> dict[str, float | None]:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(clean.min()), "mean": float(clean.mean()), "max": float(clean.max())}


def euclidean(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    av = [float(v) for v in a]
    bv = [float(v) for v in b]
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv))))


def visible_set_for_segment(
    segment: Any,
    observed_state: np.ndarray,
    args: argparse.Namespace,
) -> set[tuple[int, int, int]]:
    candidate = SimCandidateView(
        id=-1,
        grid_position=tuple(int(v) for v in segment.end_grid),
        world_position=tuple(float(v) for v in segment.end_world),
        yaw=float(segment.yaw),
        valid=True,
        candidate_source="synthetic_sc_validation",
    )
    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / float(args.voxel_size))))
    num_yaw = max(4, int(math.ceil(32 / max(1, int(args.raycast_stride)))))
    num_pitch = max(3, int(math.ceil(7 / max(1, int(args.raycast_stride)))))
    return {
        tuple(int(v) for v in voxel)
        for voxel in raycast_visible_voxels_observed(
            candidate,
            observed_state,
            max_range_voxels=max_range_voxels,
            num_yaw=num_yaw,
            num_pitch=num_pitch,
        )
    }


def sample_fov_directions(
    yaw_center: float,
    num_yaw: int,
    num_pitch: int,
    fov_yaw_deg: float = 90.0,
    fov_pitch_deg: float = 60.0,
) -> list[np.ndarray]:
    yaw_offsets = np.linspace(
        -0.5 * math.radians(float(fov_yaw_deg)),
        0.5 * math.radians(float(fov_yaw_deg)),
        max(1, int(num_yaw)),
    )
    pitch_offsets = np.linspace(
        -0.5 * math.radians(float(fov_pitch_deg)),
        0.5 * math.radians(float(fov_pitch_deg)),
        max(1, int(num_pitch)),
    )
    directions: list[np.ndarray] = []
    for yaw_offset in yaw_offsets:
        yaw = float(yaw_center) + float(yaw_offset)
        horizontal = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)
        for pitch in pitch_offsets:
            direction = np.array(
                [
                    horizontal[0] * math.cos(float(pitch)),
                    horizontal[1] * math.cos(float(pitch)),
                    math.sin(float(pitch)),
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                directions.append(direction / norm)
    return directions


def prediction_visible_set_for_segment(
    segment: Any,
    observed_state: np.ndarray,
    args: argparse.Namespace,
) -> set[tuple[int, int, int]]:
    """Visibility set for SC completion counts.

    This source-count ray pass is blocked only by measured OCCUPIED voxels.
    UNKNOWN does not block because this diagnostic asks whether an ideal SC
    completion signal can be used. Prediction still never blocks rays.
    """

    origin = np.asarray(segment.end_grid, dtype=np.float64) + 0.5
    start_voxel = tuple(int(v) for v in segment.end_grid)
    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / float(args.voxel_size))))
    num_yaw = max(4, int(math.ceil(32 / max(1, int(args.raycast_stride)))))
    num_pitch = max(3, int(math.ceil(7 / max(1, int(args.raycast_stride)))))
    directions = sample_fov_directions(float(segment.yaw), num_yaw=num_yaw, num_pitch=num_pitch)
    visible: set[tuple[int, int, int]] = set()
    shape = tuple(int(v) for v in observed_state.shape)
    for direction in directions:
        distance = 0.5
        last_voxel: tuple[int, int, int] | None = None
        while distance <= float(max_range_voxels):
            point = origin + direction * distance
            voxel = tuple(int(math.floor(float(v))) for v in point)
            if not (0 <= voxel[0] < shape[0] and 0 <= voxel[1] < shape[1] and 0 <= voxel[2] < shape[2]):
                break
            if voxel == last_voxel:
                distance += 0.5
                continue
            last_voxel = voxel
            if voxel == start_voxel:
                distance += 0.5
                continue
            visible.add(voxel)
            if observed_state[voxel] == OCCUPIED:
                break
            distance += 0.5
    return visible


def component_for_segment(
    segment: Any,
    observed_state: np.ndarray,
    prediction: EmptyPredictionLayer | SimPredictionLayer,
    hidden_mask: np.ndarray,
    args: argparse.Namespace,
    cache: dict[str, set[tuple[int, int, int]]],
) -> dict[str, Any]:
    if segment.segment_id in cache:
        visible = set(cache[segment.segment_id])
    else:
        visible = prediction_visible_set_for_segment(segment, observed_state, args)
        cache[segment.segment_id] = set(visible)
    source_threshold = float(args.ssc_confidence_threshold)
    occ_threshold = 0.5 + source_threshold
    free_threshold = 0.5 + source_threshold
    source_occ: set[tuple[int, int, int]] = set()
    source_free: set[tuple[int, int, int]] = set()
    source_unknown: set[tuple[int, int, int]] = set()
    for voxel in visible:
        if observed_state[voxel] != UNKNOWN:
            continue
        try:
            if not bool(prediction.valid[voxel]):  # type: ignore[attr-defined]
                continue
            confidence = float(prediction.confidence[voxel])  # type: ignore[attr-defined]
            occupied_prob = float(prediction.occupied_prob[voxel])  # type: ignore[attr-defined]
            free_prob = float(prediction.free_prob[voxel])  # type: ignore[attr-defined]
        except AttributeError:
            continue
        if confidence < source_threshold:
            continue
        if occupied_prob >= occ_threshold:
            source_occ.add(voxel)
        elif free_prob >= free_threshold:
            source_free.add(voxel)
        else:
            source_unknown.add(voxel)
    source_occ_free = source_occ | source_free
    hidden_visible = {voxel for voxel in source_occ_free if bool(hidden_mask[voxel])}
    return {
        "visible": visible,
        "source_occ": source_occ,
        "source_free": source_free,
        "source_unknown": source_unknown,
        "source_occ_free": source_occ_free,
        "visible_count": int(len(visible)),
        "source_occ_count": int(len(source_occ)),
        "source_free_count": int(len(source_free)),
        "source_unknown_count": int(len(source_unknown)),
        "source_occ_free_count": int(len(source_occ_free)),
        "hidden_region_visible_count": int(len(hidden_visible)),
    }


def root_child_for_path(path_ids: list[str]) -> str | None:
    for node_id in path_ids:
        if node_id != ROOT_ID:
            return str(node_id)
    return None


def rank_values(rows: list[dict[str, Any]], field: str, *, descending: bool) -> dict[str, int]:
    def value(row: dict[str, Any]) -> float:
        raw = row.get(field)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("-inf") if descending else float("inf")

    ordered = sorted(rows, key=value, reverse=descending)
    ranks: dict[str, int] = {}
    last_value: float | None = None
    last_rank = 0
    for idx, row in enumerate(ordered, start=1):
        current = value(row)
        if last_value is None or abs(current - last_value) > 1.0e-12:
            last_rank = idx
            last_value = current
        ranks[str(row["node_id"])] = int(last_rank)
    return ranks


def path_candidates(
    tree: dict[str, Any],
    observed_state: np.ndarray,
    prediction: EmptyPredictionLayer | SimPredictionLayer,
    hidden_mask: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    visible_cache: dict[str, set[tuple[int, int, int]]] = {}
    root_visible: set[tuple[int, int, int]] = set()
    if ROOT_ID in tree:
        root_visible = visible_set_for_segment(tree[ROOT_ID], observed_state, args)
    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        components[node_id] = component_for_segment(segment, observed_state, prediction, hidden_mask, args, visible_cache)

    rows: list[dict[str, Any]] = []
    root = tree[ROOT_ID]
    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        path_ids = segment_path_to_root(tree, node_id)
        no_root = [item for item in path_ids if item != ROOT_ID]
        if not no_root:
            continue
        gain_exp = float(sum(float(tree[item].gain_exp) for item in no_root))
        cost = float(sum(float(tree[item].cost) for item in no_root))
        source_occ = 0.0
        source_free = 0.0
        source_unknown = 0.0
        source_occ_free = 0.0
        hidden_visible = 0.0
        union_visible: set[tuple[int, int, int]] = set()
        for item in no_root:
            comp = components[item]
            source_occ += float(comp["source_occ_count"])
            source_free += float(comp["source_free_count"])
            source_unknown += float(comp["source_unknown_count"])
            source_occ_free += float(comp["source_occ_free_count"])
            hidden_visible += float(comp["hidden_region_visible_count"])
            union_visible |= set(comp["visible"])
        root_overlap = len(union_visible & root_visible)
        selected_child_id = root_child_for_path(path_ids)
        selected = tree.get(selected_child_id) if selected_child_id is not None else None
        rows.append(
            {
                "node_id": str(node_id),
                "selected_child_id": selected_child_id,
                "selected_child_grid": selected.end_grid if selected is not None else None,
                "selected_child_world": selected.end_world if selected is not None else None,
                "best_descendant_id": str(node_id),
                "best_descendant_grid": segment.end_grid,
                "best_descendant_world": segment.end_world,
                "root_grid": root.end_grid,
                "root_world": root.end_world,
                "path_node_ids": no_root,
                "branch_depth": int(len(no_root)),
                "accumulated_gain_exp": float(gain_exp),
                "accumulated_source_occ_free": float(source_occ_free),
                "accumulated_source_occ": float(source_occ),
                "accumulated_source_free": float(source_free),
                "accumulated_source_unknown": float(source_unknown),
                "accumulated_cost": float(cost),
                "base_exp_value": float(gain_exp / max(cost, EPS)),
                "oracle_hidden_region_visible_count": int(hidden_visible),
                "map_predict_hidden_region_visible_count": int(hidden_visible),
                "root_visible_overlap_count": int(root_overlap),
                "root_visible_overlap_fraction": float(root_overlap / max(1, len(union_visible))),
                "selected_child_distance_from_root_m": euclidean(root.end_world, selected.end_world)
                if selected is not None
                else None,
                "best_descendant_distance_from_root_m": euclidean(root.end_world, segment.end_world),
                "tree_total_nodes": int(len(tree)),
            }
        )

    if rows:
        source_ranks = rank_values(rows, "accumulated_source_occ_free", descending=True)
        gain_ranks = rank_values(rows, "accumulated_gain_exp", descending=True)
        cost_ranks = rank_values(rows, "accumulated_cost", descending=False)
        source_values = [float(row["accumulated_source_occ_free"]) for row in rows]
        min_sc = min(source_values)
        max_sc = max(source_values)
        denom = max(max_sc - min_sc, EPS)
        for row in rows:
            node_id = str(row["node_id"])
            row["source_occ_free_rank"] = source_ranks[node_id]
            row["gain_exp_rank"] = gain_ranks[node_id]
            row["cost_rank"] = cost_ranks[node_id]
            row["normalized_source_occ_free"] = float((float(row["accumulated_source_occ_free"]) - min_sc) / denom)
    return rows


def classify_branch(row: dict[str, Any]) -> str:
    root = row.get("root_world") or [0.0, 0.0, 0.0]
    best = row.get("best_descendant_world") or row.get("selected_child_world")
    if best is None:
        return "other"
    dx = float(best[0]) - float(root[0])
    dy = float(best[1]) - float(root[1])
    if dx > 0.60 and abs(float(best[1])) <= 1.20:
        return "toward_hidden_room"
    if dy > 0.65 or float(best[1]) > 1.25:
        return "toward_measured_frontier"
    return "other"


def select_decision(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    mode: str,
    prediction_source: str,
    lambda_value: float | None,
    value_kind: str,
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    for row in candidates:
        cost = float(row["accumulated_cost"])
        if cost <= EPS:
            continue
        item = dict(row)
        if value_kind == "measured":
            sc_bonus = 0.0
            final_value = float(item["base_exp_value"])
        elif value_kind == "over_cost":
            sc_bonus = float(item["accumulated_source_occ_free"]) / max(cost, EPS)
            final_value = float((float(item["accumulated_gain_exp"]) + float(item["accumulated_source_occ_free"])) / max(cost, EPS))
        elif value_kind == "decoupled_minmax":
            lam = float(lambda_value or 0.0)
            sc_bonus = lam * float(item.get("normalized_source_occ_free", 0.0))
            final_value = float(item["base_exp_value"]) + sc_bonus
        else:
            raise ValueError(f"unsupported value_kind: {value_kind}")
        item["sc_bonus_value"] = float(sc_bonus)
        item["final_value"] = float(final_value)
        scored.append(item)
    if not scored:
        raise RuntimeError(f"no scored candidates for seed={seed} mode={mode}")

    best_by_child: dict[str, dict[str, Any]] = {}
    for row in scored:
        child_id = str(row.get("selected_child_id"))
        current = best_by_child.get(child_id)
        if current is None or float(row["final_value"]) > float(current["final_value"]):
            best_by_child[child_id] = row
    ranked_children = sorted(best_by_child.values(), key=lambda item: float(item["final_value"]), reverse=True)
    winner = dict(ranked_children[0])
    runner = ranked_children[1] if len(ranked_children) > 1 else None
    runner_value = float(runner["final_value"]) if runner is not None else None
    margin = None if runner_value is None else float(float(winner["final_value"]) - runner_value)
    normalized_margin = None if margin is None else float(margin / max(abs(float(winner["final_value"])), EPS))
    branch_direction = classify_branch(winner)
    cost_rank = int(winner.get("cost_rank", 10**9))
    source_rank = int(winner.get("source_occ_free_rank", 10**9))
    low_cost_artifact = bool(
        cost_rank <= max(3, int(math.ceil(0.10 * len(scored))))
        and source_rank > max(3, int(math.ceil(0.25 * len(scored))))
        and float(winner.get("selected_child_distance_from_root_m") or 0.0) < 0.85
    )
    return {
        "seed": int(seed),
        "mode": str(mode),
        "prediction_source": str(prediction_source),
        "lambda": lambda_value,
        "value_kind": str(value_kind),
        "status": "completed",
        "selected_child_id": winner.get("selected_child_id"),
        "selected_child_grid": winner.get("selected_child_grid"),
        "selected_child_world": winner.get("selected_child_world"),
        "best_descendant_id": winner.get("best_descendant_id"),
        "best_descendant_grid": winner.get("best_descendant_grid"),
        "best_descendant_world": winner.get("best_descendant_world"),
        "selected_child_distance_from_root_m": winner.get("selected_child_distance_from_root_m"),
        "best_descendant_distance_from_root_m": winner.get("best_descendant_distance_from_root_m"),
        "branch_direction_label": branch_direction,
        "accumulated_gain_exp": winner.get("accumulated_gain_exp"),
        "accumulated_source_occ_free": winner.get("accumulated_source_occ_free"),
        "accumulated_source_occ": winner.get("accumulated_source_occ"),
        "accumulated_source_free": winner.get("accumulated_source_free"),
        "accumulated_cost": winner.get("accumulated_cost"),
        "base_exp_value": winner.get("base_exp_value"),
        "sc_bonus_value": winner.get("sc_bonus_value"),
        "final_value": winner.get("final_value"),
        "runner_up_value": runner_value,
        "margin": margin,
        "normalized_margin": normalized_margin,
        "branch_depth": winner.get("branch_depth"),
        "path_node_ids": winner.get("path_node_ids"),
        "source_occ_free_rank": winner.get("source_occ_free_rank"),
        "gain_exp_rank": winner.get("gain_exp_rank"),
        "cost_rank": winner.get("cost_rank"),
        "low_cost_artifact": low_cost_artifact,
        "oracle_hidden_region_visible_count": winner.get("oracle_hidden_region_visible_count")
        if prediction_source == "oracle"
        else None,
        "map_predict_hidden_region_visible_count": winner.get("map_predict_hidden_region_visible_count")
        if prediction_source == "map_predict"
        else None,
        "root_visible_overlap_count": winner.get("root_visible_overlap_count"),
        "root_visible_overlap_fraction": winner.get("root_visible_overlap_fraction"),
        "root_grid": winner.get("root_grid"),
        "root_world": winner.get("root_world"),
        "tree_total_nodes": winner.get("tree_total_nodes"),
        "candidate_count": len(scored),
    }


def write_tree_artifacts(tree_dir: Path, tree: dict[str, Any], summary: dict[str, Any]) -> None:
    tree_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(tree_dir / "mini_rrt_tree_segments.jsonl", [segment_record(tree[key]) for key in tree])
    save_json(tree_dir / "mini_rrt_tree_summary.json", summary)


def run_tree(
    observed_state: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    pose: dict[str, Any],
    args: argparse.Namespace,
    *,
    seed: int,
    prediction_layer: EmptyPredictionLayer | SimPredictionLayer,
    gain_mode: str,
) -> dict[str, Any]:
    root_grid = list(world_to_grid(pose["position"], bounds, float(args.voxel_size), shape=observed_state.shape, clip=True))
    root_world = [float(v) for v in pose["position"]]
    return build_mini_rrt_tree(
        observed_state=observed_state,
        root_grid=root_grid,
        root_world=root_world,
        root_yaw=float(pose.get("yaw_rad", 0.0)),
        bounds=bounds,
        seed=int(seed),
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        gain_mode=str(gain_mode),
        v_max=float(args.v_max),
        robot_radius_m=float(args.robot_radius_m),
        voxel_size=float(args.voxel_size),
        raycast_stride=int(args.raycast_stride),
        num_yaw_samples=int(args.num_yaw_samples),
        max_ray_length_m=float(args.max_ray_length_m),
        sc_gain_formula="raw_count",
        prediction_layer=prediction_layer,
        tau=float(args.tau),
        profile=True,
        crop_min_length_m=float(args.crop_min_length_m),
        short_edge_policy=str(args.short_edge_policy),
        density_radius_m=0.0,
        max_nodes_per_density_radius=0,
    )


def decision_summary_by_mode(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row["mode"]) if row.get("lambda") is None else f"{row['mode']}_lambda_{row['lambda']}"
        groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        total = len(items)
        counts = Counter(str(item["branch_direction_label"]) for item in items)
        out.append(
            {
                "mode_key": key,
                "total": total,
                "hidden_room_fraction": counts.get("toward_hidden_room", 0) / max(1, total),
                "measured_frontier_fraction": counts.get("toward_measured_frontier", 0) / max(1, total),
                "other_fraction": counts.get("other", 0) / max(1, total),
                "low_cost_artifact_fraction": sum(bool(item.get("low_cost_artifact")) for item in items) / max(1, total),
                "margin_min_mean_max": min_mean_max([float(item.get("margin") or 0.0) for item in items]),
                "hidden_prediction_count_min_mean_max": min_mean_max(
                    [
                        float(
                            item.get("oracle_hidden_region_visible_count")
                            if item.get("oracle_hidden_region_visible_count") is not None
                            else item.get("map_predict_hidden_region_visible_count") or 0.0
                        )
                        for item in items
                    ]
                ),
            }
        )
    return out


def compare_to_measured(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = {int(row["seed"]): row for row in rows if row["mode"] == "measured_only"}
    comparisons: list[dict[str, Any]] = []
    for row in rows:
        if row["mode"] == "measured_only":
            continue
        seed = int(row["seed"])
        base = measured.get(seed)
        if base is None:
            continue
        same_child = row.get("selected_child_grid") == base.get("selected_child_grid")
        same_label = row.get("branch_direction_label") == base.get("branch_direction_label")
        comparisons.append(
            {
                "seed": seed,
                "mode": row["mode"],
                "lambda": row.get("lambda"),
                "prediction_source": row.get("prediction_source"),
                "measured_branch_direction": base.get("branch_direction_label"),
                "mode_branch_direction": row.get("branch_direction_label"),
                "same_selected_grid_as_measured": bool(same_child),
                "same_direction_as_measured": bool(same_label),
                "changed_meaningfully_from_measured": bool(not same_label),
            }
        )
    by_mode = decision_summary_by_mode([row for row in rows if row["mode"] != "measured_only"])
    return {"per_seed_mode": comparisons, "summary_by_mode": by_mode}


def compare_map_to_oracle(rows: list[dict[str, Any]], map_predict_available: bool) -> dict[str, Any]:
    if not map_predict_available:
        return {
            "status": "skipped",
            "reason": "map_predict output unavailable; see map_predict_skipped_or_failed_reason.md",
            "per_seed_mode": [],
        }
    oracle = {
        (int(row["seed"]), str(row["mode"]), str(row.get("lambda"))): row
        for row in rows
        if row.get("prediction_source") == "oracle"
    }
    pairs = []
    for row in rows:
        if row.get("prediction_source") != "map_predict":
            continue
        oracle_mode = str(row["mode"]).replace("map_predict_", "oracle_")
        key = (int(row["seed"]), oracle_mode, str(row.get("lambda")))
        ref = oracle.get(key)
        if ref is None:
            continue
        pairs.append(
            {
                "seed": int(row["seed"]),
                "map_mode": row["mode"],
                "oracle_mode": ref["mode"],
                "lambda": row.get("lambda"),
                "same_direction": row.get("branch_direction_label") == ref.get("branch_direction_label"),
                "map_direction": row.get("branch_direction_label"),
                "oracle_direction": ref.get("branch_direction_label"),
                "same_selected_grid": row.get("selected_child_grid") == ref.get("selected_child_grid"),
            }
        )
    return {
        "status": "completed",
        "per_seed_mode": pairs,
        "agreement_fraction": sum(bool(row["same_direction"]) for row in pairs) / max(1, len(pairs)),
    }


def write_branch_summary_md(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = ["# Branch Direction Summary", ""]
    for row in summary_rows:
        lines.append(
            f"- `{row['mode_key']}`: hidden `{row['hidden_room_fraction']:.3f}`, measured-frontier `{row['measured_frontier_fraction']:.3f}`, low-cost `{row['low_cost_artifact_fraction']:.3f}`."
        )
    write_text(path, "\n".join(lines))


def write_comparison_md(path: Path, title: str, comparison: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    if comparison.get("status") == "skipped":
        lines.append(f"- status: `skipped`")
        lines.append(f"- reason: {comparison.get('reason')}")
    for row in comparison.get("summary_by_mode", []):
        lines.append(
            f"- `{row['mode_key']}` hidden fraction `{row['hidden_room_fraction']:.3f}`, measured-frontier fraction `{row['measured_frontier_fraction']:.3f}`."
        )
    if comparison.get("agreement_fraction") is not None:
        lines.append(f"- map/oracle direction agreement fraction: `{comparison['agreement_fraction']}`")
    write_text(path, "\n".join(lines))


def write_low_cost_md(path: Path, rows: list[dict[str, Any]]) -> None:
    flagged = [row for row in rows if bool(row.get("low_cost_artifact"))]
    lines = [
        "# Low-Cost Artifact Diagnosis",
        "",
        f"- selected decision rows: `{len(rows)}`",
        f"- low-cost artifact flags: `{len(flagged)}`",
        f"- fraction: `{len(flagged) / max(1, len(rows))}`",
        "- flag heuristic: selected path is very low cost, short from root, and not in the top source OCC+FREE quartile.",
    ]
    write_text(path, "\n".join(lines))


def plot_branch_comparison(path: Path, observed_state: np.ndarray, rows: list[dict[str, Any]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.8), constrained_layout=True)
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap="Greys", alpha=0.35, interpolation="nearest")
    colors = {
        "measured_only": "#f97316",
        "oracle_source_occ_free_over_cost": "#2563eb",
        "oracle_decoupled_source_minmax": "#7c3aed",
        "map_predict_source_occ_free_over_cost": "#059669",
        "map_predict_decoupled_source_minmax": "#14b8a6",
    }
    for row in rows:
        root = grid_point(row.get("root_grid"))
        best = grid_point(row.get("best_descendant_grid"))
        selected = grid_point(row.get("selected_child_grid"))
        color = colors.get(str(row.get("mode")), "#111827")
        if root and best:
            ax.plot([root[0], best[0]], [root[1], best[1]], color=color, linewidth=1.2, alpha=0.55)
        if selected:
            ax.scatter([selected[0]], [selected[1]], color=color, s=28, alpha=0.85)
        if best:
            ax.scatter([best[0]], [best[1]], color=color, marker="*", s=55, alpha=0.90)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_hidden_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = str(row["mode"]) if row.get("lambda") is None else f"{row['mode']} l={row['lambda']}"
        value = row.get("oracle_hidden_region_visible_count")
        if value is None:
            value = row.get("map_predict_hidden_region_visible_count")
        groups[key].append(float(value or 0.0))
    labels = list(groups.keys())
    means = [float(np.mean(groups[label])) for label in labels]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.48 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(np.arange(len(labels)), means, color="#4f86c6")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("avg hidden-region visible source count")
    ax.set_title("Hidden-region prediction counts")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_value_components(path: Path, rows: list[dict[str, Any]]) -> None:
    selected = rows[:]
    labels = [
        f"{row['mode'].replace('_source_occ_free', '').replace('_source_minmax', '')}\ns{row['seed']}"
        for row in selected
    ]
    exp_values = [float(row.get("base_exp_value") or 0.0) for row in selected]
    sc_values = [float(row.get("sc_bonus_value") or 0.0) for row in selected]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.32 * len(labels)), 5.0), constrained_layout=True)
    ax.bar(x, exp_values, label="gain_exp/cost", color="#7aa6c2")
    ax.bar(x, sc_values, bottom=exp_values, label="SC term", color="#f2b84b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("selected value components")
    ax.set_title("Branch value components")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_map_predict_if_requested(
    args: argparse.Namespace,
    output_dir: Path,
    depth: np.ndarray,
    pose: dict[str, Any],
    info: dict[str, Any],
    observed_state: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    hidden_mask: np.ndarray,
) -> tuple[Path | None, dict[str, Any]]:
    if not bool(args.run_map_predict):
        reason = "map_predict was not requested"
        write_text(output_dir / "map_predict_skipped_or_failed_reason.md", f"# map_predict skipped\n\n- reason: {reason}")
        return None, {"status": "skipped", "reason": reason}
    try:
        predictor = IsaacMapPredictor(
            checkpoint=args.checkpoint,
            device="cuda",
            tau=float(args.tau),
            torch_num_threads=int(args.torch_num_threads),
            alignment_convention=str(args.alignment_convention),
        )
        predict_dir = output_dir / "map_predict"
        result = predictor.predict_step(
            depth=np.asarray(depth, dtype=np.float32),
            pose=pose,
            camera_info=info,
            observed_state=observed_state,
            map_bounds=bounds,
            voxel_size=float(args.voxel_size),
            output_dir=predict_dir,
            step=0,
            save_probs=False,
            save_viz=False,
            observed_state_path=output_dir / "observed_state_synthetic_frame000.npy",
            depth_source=output_dir / "depth_000.npy",
            pose_source=output_dir / "pose_000.json",
            camera_info_source=output_dir / "camera_info.json",
        )
        prediction_npz = Path(result["global_prediction_npz"])
        summary = prediction_summary_from_npz(prediction_npz, observed_state, hidden_mask, float(args.tau))
        summary.update(
            {
                "status": "completed",
                "map_predict_executed": True,
                "checkpoint_unchanged": bool(predictor.checkpoint_unchanged()),
                "model_loaded_once": bool(predictor.model_loaded_once),
                "timing": result.get("timing", {}),
            }
        )
        save_json(output_dir / "map_predict_summary.json", summary)
        write_prediction_md(output_dir / "map_predict_summary.md", "map_predict Summary", summary)
        plot_prediction_topdown(
            output_dir / "map_predict_overlay_topdown.png",
            prediction_npz,
            observed_state,
            float(args.tau),
            "map_predict prediction overlay",
            overlay=True,
        )
        plot_prediction_topdown(
            output_dir / "map_predict_prediction_overlay_topdown.png",
            prediction_npz,
            observed_state,
            float(args.tau),
            "map_predict prediction overlay",
            overlay=True,
        )
        return prediction_npz, summary
    except Exception as exc:  # map_predict is diagnostic and must not block Oracle validation.
        reason = f"{type(exc).__name__}: {exc}"
        trace = traceback.format_exc()
        write_text(
            output_dir / "map_predict_skipped_or_failed_reason.md",
            "\n".join(["# map_predict skipped or failed", "", f"- reason: {reason}", "", "```text", trace, "```"]),
        )
        return None, {"status": "failed", "reason": reason}


def write_synthetic_scene_md(path: Path, scene_metadata: dict[str, Any], observed_summary: dict[str, Any]) -> None:
    topo = scene_metadata["topology_summary"]
    lines = [
        "# Synthetic Scene Summary",
        "",
        f"- scene variant: `{scene_metadata['variant']}`",
        f"- rooms/corridors/openings: `{topo['room_count']}` / `{topo['corridor_count']}` / `{topo['opening_count']}`",
        f"- observed shape: `{observed_summary['shape']}`",
        f"- observed unknown/free/occupied: `{observed_summary['unknown_count']}` / `{observed_summary['free_count']}` / `{observed_summary['occupied_count']}`",
        "- camera frames captured: `1`",
        "- selected action executed: `False`",
        "- rollout/two-frame runtime: `False` / `False`",
    ]
    write_text(path, "\n".join(lines))


def make_final_summary(
    args: argparse.Namespace,
    output_dir: Path,
    scene_metadata: dict[str, Any],
    observed_summary: dict[str, Any],
    oracle_summary: dict[str, Any],
    map_summary: dict[str, Any],
    decision_rows: list[dict[str, Any]],
    branch_summary_rows: list[dict[str, Any]],
    oracle_vs_measured: dict[str, Any],
    map_vs_oracle: dict[str, Any],
) -> dict[str, Any]:
    measured_rows = [row for row in decision_rows if row["mode"] == "measured_only"]
    oracle_rows = [row for row in decision_rows if row.get("prediction_source") == "oracle"]
    map_rows = [row for row in decision_rows if row.get("prediction_source") == "map_predict"]
    measured_hidden_fraction = sum(row["branch_direction_label"] == "toward_hidden_room" for row in measured_rows) / max(
        1, len(measured_rows)
    )
    oracle_hidden_fraction = sum(row["branch_direction_label"] == "toward_hidden_room" for row in oracle_rows) / max(
        1, len(oracle_rows)
    )
    map_hidden_count = int(map_summary.get("hidden_region_valid_tau_unmeasured_count", 0) or 0)
    map_hidden_fraction = (
        sum(row["branch_direction_label"] == "toward_hidden_room" for row in map_rows) / max(1, len(map_rows))
        if map_rows
        else 0.0
    )
    oracle_effective = oracle_hidden_fraction > measured_hidden_fraction and oracle_hidden_fraction >= 0.40
    map_available = bool(map_rows)
    map_effective = bool(map_available and map_hidden_fraction >= 0.40 and map_hidden_count > 0)
    if oracle_effective and not map_effective:
        bottleneck = "utility pipeline can use a good SC signal; map_predict/domain/preprocess/calibration is the likely bottleneck"
        next_step = "inspect map_predict hidden-region calibration/domain mismatch on this synthetic frame"
    elif oracle_effective and map_effective:
        bottleneck = "oracle and map_predict both produced useful hidden-room signal in this diagnostic"
        next_step = "repeat a tiny controlled map_predict calibration smoke before any runtime smoke"
    else:
        bottleneck = "tree utility / SC integration still needs inspection because oracle did not robustly steer hidden-room selection"
        next_step = "debug oracle source OCC+FREE visibility and tree selection on the synthetic scene"

    return {
        "stage": "Stage 4A-6.5aa controlled synthetic SC validation scene smoke",
        "output_dir": str(output_dir),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "synthetic_scene_constructed": True,
        "capture_backend": "scripted_scene_factory_depth_raycast",
        "frames_captured": 1,
        "selected_action_executed": False,
        "selected_action_execution_count": 0,
        "two_frame_runtime": False,
        "rollout": False,
        "online_open_ended_loop": False,
        "training_rl_ppo_bc_il": False,
        "checkpoint_modified": False,
        "prediction_writeback": False,
        "prediction_used_for_traversability_collision_ray_blocking": False,
        "target_or_ground_truth_used_for_planning_scoring": False,
        "coverage_improvement_claimed": False,
        "scene": scene_metadata,
        "observed_state": observed_summary,
        "oracle_prediction": oracle_summary,
        "map_predict": map_summary,
        "tree": {
            "seed_count": len(set(int(row["seed"]) for row in decision_rows)),
            "decision_row_count": len(decision_rows),
            "branch_summary_by_mode": branch_summary_rows,
            "measured_hidden_fraction": measured_hidden_fraction,
            "oracle_hidden_fraction": oracle_hidden_fraction,
            "map_predict_hidden_fraction": map_hidden_fraction,
            "oracle_low_cost_artifact_fraction": sum(bool(row.get("low_cost_artifact")) for row in oracle_rows)
            / max(1, len(oracle_rows)),
        },
        "comparisons": {
            "oracle_vs_measured": oracle_vs_measured,
            "map_predict_vs_oracle": map_vs_oracle,
        },
        "answers": {
            "synthetic_scene_successfully_constructed": True,
            "measured_only_direction_counts": dict(Counter(row["branch_direction_label"] for row in measured_rows)),
            "oracle_prediction_steered_hidden_room_direction": bool(oracle_effective),
            "oracle_over_cost_and_decoupled_consistent": _oracle_modes_consistent(oracle_rows),
            "oracle_low_cost_artifact_detected": any(bool(row.get("low_cost_artifact")) for row in oracle_rows),
            "map_predict_produced_hidden_region_occ_free": bool(map_hidden_count > 0),
            "map_predict_selected_or_reinforced_oracle_direction": bool(map_effective),
            "bottleneck_interpretation": bottleneck,
            "runtime_smoke_supported_now": False,
            "rollout_supported_now": False,
            "recommended_next": next_step,
        },
        "recommended_next_faithful_step": next_step,
        "still_not_next": [
            "runtime smoke",
            "rollout",
            "online open-ended loop",
            "Pareto dominance gate",
            "runtime planner implementation",
            "RL/PPO/BC/IL training",
            "prediction writeback",
            "observed_map prediction fusion",
            "target/ground-truth scoring",
            "checkpoint changes",
            "coverage-improvement claims",
            "external source build",
        ],
    }


def _oracle_modes_consistent(oracle_rows: list[dict[str, Any]]) -> bool:
    by_seed: dict[int, set[str]] = defaultdict(set)
    for row in oracle_rows:
        by_seed[int(row["seed"])].add(str(row["branch_direction_label"]))
    return bool(by_seed) and all(len(labels) == 1 for labels in by_seed.values())


def write_final_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    tree = summary["tree"]
    lines = [
        "# Synthetic SC Validation Summary",
        "",
        f"1. synthetic scene constructed? `{answers['synthetic_scene_successfully_constructed']}`.",
        f"2. measured-only direction counts: `{answers['measured_only_direction_counts']}`.",
        f"3. Oracle steered toward hidden room / doorway? `{answers['oracle_prediction_steered_hidden_room_direction']}`.",
        f"4. Oracle over-cost and decoupled consistent? `{answers['oracle_over_cost_and_decoupled_consistent']}`.",
        f"5. Oracle low-cost artifact detected? `{answers['oracle_low_cost_artifact_detected']}`; fraction `{tree['oracle_low_cost_artifact_fraction']}`.",
        f"6. map_predict hidden-region OCC/FREE produced? `{answers['map_predict_produced_hidden_region_occ_free']}`.",
        f"7. map_predict selected or reinforced Oracle direction? `{answers['map_predict_selected_or_reinforced_oracle_direction']}`.",
        f"8. bottleneck: {answers['bottleneck_interpretation']}.",
        "9. If Oracle is ineffective, inspect tree utility / SC integration before runtime use.",
        f"10. runtime smoke supported now? `{answers['runtime_smoke_supported_now']}`.",
        f"11. rollout supported now? `{answers['rollout_supported_now']}`.",
        f"12. next recommended step: {answers['recommended_next']}.",
        "",
        "No coverage improvement is claimed.",
    ]
    write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_time = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if str(args.scene_variant) != "synthetic_hidden_room_frontier":
        raise ValueError("Stage 4A-6.5aa only supports --scene_variant synthetic_hidden_room_frontier")

    bounds = bounds_from_args(args)
    scene_metadata = build_synthetic_hidden_room_frontier_scene(seed=int(args.scene_seed), spawn=False)
    scene_metadata["map_bounds"] = {axis: [float(bounds[axis][0]), float(bounds[axis][1])] for axis in ("x", "y", "z")}
    save_json(output_dir / "scene_metadata.json", scene_metadata)

    observed_state, observed_summary = make_observed_state(scene_metadata, bounds, float(args.voxel_size))
    observed_path = output_dir / "observed_state_synthetic_frame000.npy"
    np.save(observed_path, observed_state)
    observed_hash = sha256_file(observed_path)
    observed_summary.update({"observed_state_path": str(observed_path), "sha256": observed_hash})
    save_json(output_dir / "observed_state_summary.json", observed_summary)

    depth, rgb, pose, info = render_synthetic_frame(scene_metadata, args)
    np.save(output_dir / "depth_000.npy", depth)
    Image.fromarray(rgb).save(output_dir / "rgb_000.png")
    save_json(output_dir / "pose_000.json", pose)
    save_json(output_dir / "camera_info.json", info)

    synthetic_scene_summary = {
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "bounds": {axis: [float(bounds[axis][0]), float(bounds[axis][1])] for axis in ("x", "y", "z")},
        "voxel_size": float(args.voxel_size),
        "frames_captured": 1,
        "capture_backend": "scripted_scene_factory_depth_raycast",
        "selected_action_executed": False,
        "rollout": False,
        "two_frame_runtime": False,
        "observed_state_summary": observed_summary,
        "render_stats": info.get("render_stats", {}),
    }
    save_json(output_dir / "synthetic_scene_summary.json", synthetic_scene_summary)
    write_synthetic_scene_md(output_dir / "synthetic_scene_summary.md", scene_metadata, observed_summary)

    plot_scene_layout(output_dir / "scene_layout_topdown.png", scene_metadata)
    plot_observed(output_dir / "observed_topdown.png", observed_state, "measured-only observed_state")

    hidden_mask = region_mask(
        tuple(int(v) for v in observed_state.shape),
        bounds,
        float(args.voxel_size),
        scene_metadata["diagnostic_regions"]["oracle_hidden_room"],
    )
    oracle_npz = output_dir / "oracle_global_prediction_layer.npz"
    oracle_summary = make_oracle_prediction(
        oracle_npz,
        observed_state,
        scene_metadata,
        bounds,
        float(args.voxel_size),
        str(args.alignment_convention),
    )
    save_json(output_dir / "oracle_prediction_summary.json", oracle_summary)
    write_prediction_md(output_dir / "oracle_prediction_summary.md", "Oracle Prediction Summary", oracle_summary)
    plot_prediction_topdown(output_dir / "oracle_prediction_topdown.png", oracle_npz, observed_state, float(args.tau), "oracle prediction")
    plot_prediction_topdown(
        output_dir / "oracle_prediction_overlay_topdown.png",
        oracle_npz,
        observed_state,
        float(args.tau),
        "oracle prediction overlay",
        overlay=True,
    )

    map_npz, map_summary = run_map_predict_if_requested(
        args,
        output_dir,
        depth,
        pose,
        info,
        observed_state,
        bounds,
        hidden_mask,
    )

    seeds = parse_ints(args.seeds)
    oracle_layer = SimPredictionLayer.from_npz(oracle_npz)
    map_layer = SimPredictionLayer.from_npz(map_npz) if map_npz is not None else None
    empty_layer = EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape))
    decision_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    raw_root = output_dir / "raw_trees"

    for seed in seeds:
        measured_result = run_tree(
            observed_state,
            bounds,
            pose,
            args,
            seed=seed,
            prediction_layer=empty_layer,
            gain_mode="exp",
        )
        measured_tree = measured_result["tree"]
        measured_dir = raw_root / f"seed_{seed:03d}" / "measured_only"
        write_tree_artifacts(measured_dir, measured_tree, {"profile": measured_result["profile"], "decision": measured_result["decision"]})
        measured_candidates = path_candidates(measured_tree, observed_state, empty_layer, hidden_mask, args)
        measured_decision = select_decision(
            measured_candidates,
            seed=seed,
            mode="measured_only",
            prediction_source="none",
            lambda_value=None,
            value_kind="measured",
        )
        decision_rows.append(measured_decision)
        manifest_rows.append(
            {
                "seed": seed,
                "mode": "measured_only",
                "prediction_source": "none",
                "tree_dir": str(measured_dir),
                "status": "completed",
            }
        )

        oracle_result = run_tree(
            observed_state,
            bounds,
            pose,
            args,
            seed=seed,
            prediction_layer=oracle_layer,
            gain_mode="hybrid",
        )
        oracle_tree = oracle_result["tree"]
        oracle_dir = raw_root / f"seed_{seed:03d}" / "oracle_raw_count"
        write_tree_artifacts(oracle_dir, oracle_tree, {"profile": oracle_result["profile"], "decision": oracle_result["decision"]})
        oracle_candidates = path_candidates(oracle_tree, observed_state, oracle_layer, hidden_mask, args)
        decision_rows.append(
            select_decision(
                oracle_candidates,
                seed=seed,
                mode="oracle_source_occ_free_over_cost",
                prediction_source="oracle",
                lambda_value=None,
                value_kind="over_cost",
            )
        )
        manifest_rows.append(
            {
                "seed": seed,
                "mode": "oracle_source_occ_free_over_cost",
                "prediction_source": "oracle",
                "tree_dir": str(oracle_dir),
                "status": "completed",
            }
        )
        for lam in (8.0, 16.0, 32.0):
            decision_rows.append(
                select_decision(
                    oracle_candidates,
                    seed=seed,
                    mode="oracle_decoupled_source_minmax",
                    prediction_source="oracle",
                    lambda_value=lam,
                    value_kind="decoupled_minmax",
                )
            )
            manifest_rows.append(
                {
                    "seed": seed,
                    "mode": "oracle_decoupled_source_minmax",
                    "lambda": lam,
                    "prediction_source": "oracle",
                    "tree_dir": str(oracle_dir),
                    "status": "completed",
                }
            )

        if map_layer is not None:
            map_result = run_tree(
                observed_state,
                bounds,
                pose,
                args,
                seed=seed,
                prediction_layer=map_layer,
                gain_mode="hybrid",
            )
            map_tree = map_result["tree"]
            map_dir = raw_root / f"seed_{seed:03d}" / "map_predict_raw_count"
            write_tree_artifacts(map_dir, map_tree, {"profile": map_result["profile"], "decision": map_result["decision"]})
            map_candidates = path_candidates(map_tree, observed_state, map_layer, hidden_mask, args)
            decision_rows.append(
                select_decision(
                    map_candidates,
                    seed=seed,
                    mode="map_predict_source_occ_free_over_cost",
                    prediction_source="map_predict",
                    lambda_value=None,
                    value_kind="over_cost",
                )
            )
            manifest_rows.append(
                {
                    "seed": seed,
                    "mode": "map_predict_source_occ_free_over_cost",
                    "prediction_source": "map_predict",
                    "tree_dir": str(map_dir),
                    "status": "completed",
                }
            )
            for lam in (8.0, 16.0, 32.0):
                decision_rows.append(
                    select_decision(
                        map_candidates,
                        seed=seed,
                        mode="map_predict_decoupled_source_minmax",
                        prediction_source="map_predict",
                        lambda_value=lam,
                        value_kind="decoupled_minmax",
                    )
                )
                manifest_rows.append(
                    {
                        "seed": seed,
                        "mode": "map_predict_decoupled_source_minmax",
                        "lambda": lam,
                        "prediction_source": "map_predict",
                        "tree_dir": str(map_dir),
                        "status": "completed",
                    }
                )

    write_jsonl(output_dir / "tree_decision_manifest.jsonl", manifest_rows)
    save_json(output_dir / "per_seed_mode_decisions.json", decision_rows)
    write_csv(output_dir / "per_seed_mode_decisions.csv", decision_rows)

    branch_rows = [
        {
            "seed": row["seed"],
            "mode": row["mode"],
            "lambda": row.get("lambda"),
            "prediction_source": row.get("prediction_source"),
            "branch_direction_label": row.get("branch_direction_label"),
            "selected_child_grid": row.get("selected_child_grid"),
            "best_descendant_grid": row.get("best_descendant_grid"),
        }
        for row in decision_rows
    ]
    save_json(output_dir / "branch_direction_classification.json", branch_rows)
    write_csv(output_dir / "branch_direction_classification.csv", branch_rows)
    branch_summary_rows = decision_summary_by_mode(decision_rows)
    write_branch_summary_md(output_dir / "branch_direction_summary.md", branch_summary_rows)

    oracle_vs_measured = compare_to_measured([row for row in decision_rows if row.get("prediction_source") in {"none", "oracle"}])
    save_json(output_dir / "oracle_vs_measured_comparison.json", oracle_vs_measured)
    write_comparison_md(output_dir / "oracle_vs_measured_comparison.md", "Oracle Vs Measured Comparison", oracle_vs_measured)

    map_vs_oracle = compare_map_to_oracle(decision_rows, map_layer is not None)
    save_json(output_dir / "map_predict_vs_oracle_comparison.json", map_vs_oracle)
    write_comparison_md(output_dir / "map_predict_vs_oracle_comparison.md", "map_predict Vs Oracle Comparison", map_vs_oracle)

    write_csv(output_dir / "low_cost_artifact_diagnosis.csv", decision_rows)
    write_low_cost_md(output_dir / "low_cost_artifact_diagnosis.md", decision_rows)

    lines = ["# Per-Seed Mode Decisions", ""]
    for row in decision_rows:
        lam = "" if row.get("lambda") is None else f" lambda `{row['lambda']}`"
        lines.append(
            f"- seed `{row['seed']}` `{row['mode']}`{lam}: `{row['branch_direction_label']}`, selected `{row['selected_child_id']}` -> best `{row['best_descendant_id']}`, value `{row['final_value']}`."
        )
    write_text(output_dir / "per_seed_mode_decisions.md", "\n".join(lines))

    plot_branch_comparison(
        output_dir / "measured_vs_oracle_selected_branches_topdown.png",
        observed_state,
        [row for row in decision_rows if row.get("prediction_source") in {"none", "oracle"}],
        "measured-only vs oracle selected branches",
    )
    plot_branch_comparison(
        output_dir / "selected_branch_by_seed_topdown.png",
        observed_state,
        decision_rows,
        "selected branch by seed/mode",
    )
    plot_hidden_counts(output_dir / "hidden_region_prediction_counts.png", decision_rows)
    plot_value_components(output_dir / "branch_value_components.png", decision_rows)

    summary = make_final_summary(
        args,
        output_dir,
        scene_metadata,
        observed_summary,
        oracle_summary,
        map_summary,
        decision_rows,
        branch_summary_rows,
        oracle_vs_measured,
        map_vs_oracle,
    )
    summary["runtime_s"] = float(time.perf_counter() - start_time)
    save_json(output_dir / "synthetic_sc_validation_summary.json", summary)
    write_final_summary_md(output_dir / "synthetic_sc_validation_summary.md", summary)
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {summary['recommended_next_faithful_step']}",
                "- reason: this stage is diagnostic-only and does not justify runtime smoke or rollout by itself.",
                "- still not next: runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training, prediction writeback, target/ground-truth scoring, checkpoint changes, or coverage claims.",
            ]
        ),
    )

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene_variant", default="synthetic_hidden_room_frontier")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--map_bound_mode", default="medium")
    parser.add_argument("--x_min", type=float, default=-6.0)
    parser.add_argument("--x_max", type=float, default=6.0)
    parser.add_argument("--y_min", type=float, default=-6.0)
    parser.add_argument("--y_max", type=float, default=6.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", choices=["crop"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--ssc_confidence_threshold", type=float, default=0.05)
    parser.add_argument("--alignment_convention", choices=["code_consistent_v1"], default="code_consistent_v1")
    parser.add_argument("--run_map_predict", action="store_true")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--camera_width", type=int, default=160)
    parser.add_argument("--camera_height", type=int, default=120)
    parser.add_argument("--max_depth", type=float, default=6.0)
    parser.add_argument("--torch_num_threads", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
