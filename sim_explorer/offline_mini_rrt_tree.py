#!/usr/bin/env python3
"""Stage 4A-6.5i offline mini-RRT tree builder on a saved observed map.

This script is intentionally offline. It reads an existing observed_state,
builds a small measured-only RRT-like tree in memory, computes the source-style
GlobalNormalizedGain/SubsequentBest utility, and writes diagnostics. It does
not launch Isaac, run rollout, rerun map_predict, train models, modify
observed_state, or modify/build external active_3d_planning source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from astar_planner import (
    build_traversability_grid,
    connected_component_from_start,
    nearest_traversable_cell,
    summarize_traversability,
)
from offline_tree_utility_prototype import compute_global_normalized_gain, select_subsequent_best
from sim_paper_expert import (
    FREE,
    UNKNOWN,
    EmptyPredictionLayer,
    SimCandidateView,
    compute_paper_gains_for_candidate,
    frontier_adjacent_free_xy_mask,
    grid_to_world,
    normalize_bounds,
    raycast_visible_voxels_observed,
    world_to_grid,
)
from sim_prediction_layer import SimPredictionLayer


EPS = 1.0e-6
ROOT_ID = "root"
ROLLOUT_LIKE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]


@dataclass
class MiniRRTSegment:
    segment_id: str
    parent_id: str | None
    children: list[str] = field(default_factory=list)
    start_grid: list[int] | None = None
    start_world: list[float] | None = None
    end_grid: list[int] | None = None
    end_world: list[float] | None = None
    yaw: float = 0.0
    gain: float = 0.0
    gain_exp: float = 0.0
    gain_sc: float = 0.0
    gain_hybrid: float = 0.0
    effective_gain_sc: float = 0.0
    gain_hybrid_effective: float = 0.0
    gain_occ: float = 0.0
    gain_conf: float = 0.0
    sc_gain_formula: str = "raw_count"
    cost: float = 0.0
    value: float = float("-inf")
    best_descendant_id: str | None = None
    accumulated_gain: float = 0.0
    accumulated_cost: float = 0.0
    depth: int = 0
    segment_length_m: float = 0.0
    yaw_delta: float = 0.0
    yaw_time: float = 0.0
    local_visibility_stats: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path) == "":
        return {}
    json_path = Path(path)
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True))
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def euclidean(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def distance_xy(a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def in_bounds_xy(xy: tuple[int, int] | list[int], shape: tuple[int, int]) -> bool:
    return 0 <= int(xy[0]) < shape[0] and 0 <= int(xy[1]) < shape[1]


def line_cells_xy(start_xy: tuple[int, int] | list[int], end_xy: tuple[int, int] | list[int]) -> list[tuple[int, int]]:
    """Return deterministic integer cells along a 2D segment."""

    x0, y0 = int(start_xy[0]), int(start_xy[1])
    x1, y1 = int(end_xy[0]), int(end_xy[1])
    steps = max(abs(x1 - x0), abs(y1 - y0))
    if steps == 0:
        return [(x0, y0)]
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for step in range(steps + 1):
        t = float(step) / float(steps)
        cell = (int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)))
        if cell not in seen:
            seen.add(cell)
            cells.append(cell)
    return cells


def line_is_traversable(
    traversable: np.ndarray,
    start_xy: tuple[int, int] | list[int],
    end_xy: tuple[int, int] | list[int],
    reachable_mask: np.ndarray | None = None,
) -> tuple[bool, str, list[tuple[int, int]]]:
    grid = np.asarray(traversable, dtype=bool)
    reachable = None if reachable_mask is None else np.asarray(reachable_mask, dtype=bool)
    shape = (int(grid.shape[0]), int(grid.shape[1]))
    cells = line_cells_xy(start_xy, end_xy)
    for cell in cells:
        if not in_bounds_xy(cell, shape):
            return False, "edge_out_of_bounds", cells
        if not bool(grid[cell]):
            return False, "edge_non_traversable_or_unknown", cells
        if reachable is not None and not bool(reachable[cell]):
            return False, "edge_leaves_reachable_component", cells
    return True, "ok", cells


def nearest_free_z_for_xy(observed_state: np.ndarray, xy: tuple[int, int] | list[int], preferred_k: int) -> int | None:
    i, j = int(xy[0]), int(xy[1])
    if not (0 <= i < observed_state.shape[0] and 0 <= j < observed_state.shape[1]):
        return None
    free_z = np.flatnonzero(observed_state[i, j, :] == FREE)
    if free_z.size == 0:
        return None
    best_idx = int(np.argmin(np.abs(free_z.astype(np.int64) - int(preferred_k))))
    return int(free_z[best_idx])


def add_tree_child(tree: dict[str, MiniRRTSegment], parent_id: str, segment: MiniRRTSegment) -> None:
    if parent_id not in tree:
        raise KeyError(f"missing parent: {parent_id}")
    tree[segment.segment_id] = segment
    tree[parent_id].children.append(segment.segment_id)


def segment_record(segment: MiniRRTSegment) -> dict[str, Any]:
    record = asdict(segment)
    record["value"] = segment.value if math.isfinite(segment.value) else None
    return record


def segment_brief(segment: MiniRRTSegment | None) -> dict[str, Any] | None:
    if segment is None:
        return None
    return {
        "segment_id": segment.segment_id,
        "parent_id": segment.parent_id,
        "start_grid": segment.start_grid,
        "end_grid": segment.end_grid,
        "start_world": segment.start_world,
        "end_world": segment.end_world,
        "yaw": segment.yaw,
        "gain": segment.gain,
        "gain_exp": segment.gain_exp,
        "gain_sc": segment.gain_sc,
        "gain_hybrid": segment.gain_hybrid,
        "effective_gain_sc": segment.effective_gain_sc,
        "gain_hybrid_effective": segment.gain_hybrid_effective,
        "gain_occ": segment.gain_occ,
        "gain_conf": segment.gain_conf,
        "sc_gain_formula": segment.sc_gain_formula,
        "cost": segment.cost,
        "value": segment.value if math.isfinite(segment.value) else None,
        "best_descendant_id": segment.best_descendant_id,
        "accumulated_gain": segment.accumulated_gain,
        "accumulated_cost": segment.accumulated_cost,
        "depth": segment.depth,
        "segment_length_m": segment.segment_length_m,
    }


def parse_formula_number(raw: str) -> float:
    text = str(raw).strip().replace("p", ".")
    if text.startswith("_"):
        text = text[1:]
    value = float(text)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"formula value must be finite and non-negative: {raw}")
    return value


def effective_sc_gain_from_formula(
    *,
    raw_gain_sc: float,
    gain_occ: float,
    gain_conf: float,
    formula: str,
) -> float:
    """Return the gated SC gain used by the tree utility.

    The raw aggregate fields stay recorded separately. This helper only changes
    the optional prediction term that is added to measured gain in SC modes.
    """

    name = str(formula or "raw_count")
    raw = float(raw_gain_sc)
    occ = float(gain_occ)
    conf = float(gain_conf)
    if name in {"measured_only", "none", "exp"}:
        return 0.0
    if name in {"raw_count", "weight_1p0", "weight_1.0"}:
        return raw
    if name.startswith("weight_"):
        return parse_formula_number(name[len("weight_") :]) * raw
    if name.startswith("cap_"):
        return min(raw, parse_formula_number(name[len("cap_") :]))
    if name.startswith("cap"):
        return min(raw, parse_formula_number(name[len("cap") :]))
    if name in {"occupied_only", "occupied_only_gain_occ"}:
        return occ
    if name in {"confidence_weighted", "confidence_weighted_gain_conf"}:
        return conf
    if name in {"confidence_weighted_cap25", "confidence_weighted_gain_conf_cap25"}:
        return min(conf, 25.0)
    raise ValueError(
        "unsupported sc_gain_formula: "
        f"{formula}; expected raw_count, weight_*, cap*, confidence_weighted, occupied_only, or measured_only"
    )


def segment_path_to_root(tree: dict[str, MiniRRTSegment], segment_id: str | None) -> list[str]:
    if segment_id is None or segment_id not in tree:
        return []
    path: list[str] = []
    current: str | None = segment_id
    while current is not None and current in tree:
        path.append(current)
        current = tree[current].parent_id
    path.reverse()
    return path


def load_npz_scalar_or_array(path: Path, key: str) -> Any:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if key not in data.files:
            return None
        value = data[key]
        if value.shape == ():
            return value.item()
        return np.array(value)


def resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    case_json = Path(args.case_json).resolve() if args.case_json else None
    case = read_json(case_json)
    case_step = int(case.get("step", 1))
    explicit_episode = bool(args.episode_dir)
    episode_dir = Path(args.episode_dir or case.get("episode_dir", "")).resolve()

    def step_file(name: str, suffix: str) -> Path | None:
        if not episode_dir or str(episode_dir) == ".":
            return None
        return episode_dir / f"{name}_{case_step:03d}.{suffix}"

    observed_state = Path(args.observed_state).resolve() if args.observed_state else None
    if observed_state is None:
        if explicit_episode:
            observed_state = episode_dir / f"observed_state_step{case_step:03d}.npy"
        if observed_state is None or not observed_state.is_file():
            observed_state = Path(case.get("observed_state", "")).resolve()
    if not observed_state or not observed_state.is_file():
        raise FileNotFoundError(f"observed_state not found: {observed_state}")

    pose_json = Path(args.pose_json).resolve() if args.pose_json else None
    if pose_json is None:
        if explicit_episode and step_file("pose", "json") is not None:
            pose_json = step_file("pose", "json")
        if pose_json is None or not pose_json.is_file():
            pose_json = Path(case.get("pose_json", "")).resolve()

    camera_info = Path(args.camera_info).resolve() if args.camera_info else None
    if camera_info is None:
        if explicit_episode:
            camera_info = episode_dir / "camera_info.json"
        if camera_info is None or not camera_info.is_file():
            camera_info = Path(case.get("camera_info", "")).resolve()

    episode_summary = Path(args.episode_summary).resolve() if args.episode_summary else None
    if episode_summary is None:
        if explicit_episode:
            episode_summary = episode_dir / "episode_summary.json"
        if episode_summary is None or not episode_summary.is_file():
            episode_summary = Path(case.get("episode_summary", "")).resolve()

    step_npz = Path(case.get("step_npz", "")).resolve()
    if explicit_episode:
        candidate_step_npz = step_file("step", "npz")
        if candidate_step_npz is not None and candidate_step_npz.is_file():
            step_npz = candidate_step_npz

    prediction_npz = Path(args.prediction_npz).resolve() if args.prediction_npz else None
    case_prediction = Path(case.get("prediction_npz", "")).resolve()
    if prediction_npz is None and args.gain_mode in {"hybrid", "sc"} and case_prediction.is_file():
        prediction_npz = case_prediction

    one_step_comparison = None
    if case_json is not None:
        candidate = case_json.parent / "one_step_comparison.json"
        if candidate.is_file():
            one_step_comparison = candidate

    return {
        "case_json": case_json,
        "case": case,
        "case_step": case_step,
        "episode_dir": episode_dir if episode_dir.exists() else None,
        "observed_state": observed_state,
        "pose_json": pose_json if pose_json and pose_json.is_file() else None,
        "camera_info": camera_info if camera_info and camera_info.is_file() else None,
        "episode_summary": episode_summary if episode_summary and episode_summary.is_file() else None,
        "step_npz": step_npz if step_npz and step_npz.is_file() else None,
        "prediction_npz": prediction_npz if prediction_npz and prediction_npz.is_file() else None,
        "one_step_comparison": one_step_comparison,
    }


def load_bounds_and_root(inputs: dict[str, Any], observed_shape: tuple[int, int, int], voxel_size: float) -> dict[str, Any]:
    episode_summary = read_json(inputs.get("episode_summary"))
    raw_bounds = episode_summary.get("map_bounds") or episode_summary.get("bounds")
    if raw_bounds is None:
        raw_bounds = {
            "x": (-0.5 * observed_shape[0] * voxel_size, 0.5 * observed_shape[0] * voxel_size),
            "y": (-0.5 * observed_shape[1] * voxel_size, 0.5 * observed_shape[1] * voxel_size),
            "z": (0.0, observed_shape[2] * voxel_size),
        }
    bounds = normalize_bounds(raw_bounds)

    pose = read_json(inputs.get("pose_json"))
    step_npz = inputs.get("step_npz")
    root_world: list[float] | None = None
    root_grid: list[int] | None = None
    root_yaw = 0.0

    if step_npz is not None:
        value = load_npz_scalar_or_array(step_npz, "current_pose_world")
        if value is not None:
            root_world = [float(v) for v in np.asarray(value, dtype=np.float64).tolist()]
        value = load_npz_scalar_or_array(step_npz, "current_pose_grid")
        if value is not None:
            root_grid = [int(v) for v in np.asarray(value, dtype=np.int64).tolist()]
        value = load_npz_scalar_or_array(step_npz, "current_yaw")
        if value is not None:
            root_yaw = float(value)

    if root_world is None and pose.get("position") is not None:
        root_world = [float(v) for v in pose["position"]]
    if pose.get("yaw_rad") is not None:
        root_yaw = float(pose["yaw_rad"])
    if root_grid is None and root_world is not None:
        root_grid = list(world_to_grid(root_world, bounds, voxel_size, shape=observed_shape, clip=True))
    if root_grid is None:
        raise ValueError("Could not resolve root grid from step_npz or pose_json")
    if root_world is None:
        root_world = list(grid_to_world(root_grid, bounds, voxel_size))

    return {
        "bounds": bounds,
        "episode_summary": episode_summary,
        "pose": pose,
        "root_grid": [int(v) for v in root_grid],
        "root_world": [float(v) for v in root_world],
        "root_yaw": float(root_yaw),
    }


def build_sampling_context(
    observed_state: np.ndarray,
    root_grid: list[int],
    voxel_size: float,
    robot_radius_m: float,
) -> dict[str, Any]:
    traversability = build_traversability_grid(
        observed_state,
        voxel_size=float(voxel_size),
        robot_height_m=1.2,
        clearance_height_m=0.6,
        robot_radius_m=float(robot_radius_m),
    )
    traversable = np.asarray(traversability["traversable"], dtype=bool)
    root_xy = (int(root_grid[0]), int(root_grid[1]))
    snap = nearest_traversable_cell(traversable, root_xy, max_radius_cells=5)
    start_xy = root_xy
    if not (in_bounds_xy(root_xy, traversable.shape) and bool(traversable[root_xy])):
        if not bool(snap.get("found", False)):
            raise ValueError(f"Root is not traversable and could not be snapped: {snap}")
        cell = snap["cell"]
        start_xy = (int(cell[0]), int(cell[1]))

    component = connected_component_from_start(traversable, start_xy, allow_diagonal=True)
    reachable_mask = np.asarray(component["reachable_mask"], dtype=bool)
    frontier_xy = frontier_adjacent_free_xy_mask(observed_state)
    reachable_frontier_mask = reachable_mask & frontier_xy
    return {
        "traversability": traversability,
        "traversable": traversable,
        "root_xy": [int(root_xy[0]), int(root_xy[1])],
        "snapped_root_xy": [int(start_xy[0]), int(start_xy[1])],
        "root_snapped": start_xy != root_xy,
        "root_snap_result": snap,
        "reachable_mask": reachable_mask,
        "reachable_frontier_mask": reachable_frontier_mask,
        "reachable_free_count": int(np.count_nonzero(reachable_mask)),
        "reachable_frontier_count": int(np.count_nonzero(reachable_frontier_mask)),
        "frontier_adjacent_free_count": int(np.count_nonzero(frontier_xy)),
        "component": {
            "start_valid": bool(component.get("start_valid", False)),
            "reason": str(component.get("reason", "")),
            "reachable_count": int(component.get("reachable_count", 0)),
            "start_xy": component.get("start_xy"),
        },
        "traversability_summary": summarize_traversability(traversability),
    }


def choose_sample_pool(context: dict[str, Any], sample_mode: str, rng: np.random.Generator) -> tuple[str, np.ndarray]:
    reachable = np.asarray(context["reachable_mask"], dtype=bool)
    frontier = np.asarray(context["reachable_frontier_mask"], dtype=bool)
    if sample_mode == "reachable_frontier":
        source = "reachable_frontier"
        mask = frontier
    elif sample_mode == "reachable_free":
        source = "reachable_free"
        mask = reachable
    elif sample_mode == "mixed":
        use_frontier = bool(np.count_nonzero(frontier)) and float(rng.random()) < 0.7
        source = "reachable_frontier" if use_frontier else "reachable_free"
        mask = frontier if use_frontier else reachable
    else:
        raise ValueError("sample_mode must be reachable_frontier, reachable_free, or mixed")
    pool = np.argwhere(mask).astype(np.int32, copy=False)
    return source, pool


def score_pose_gain(
    observed_state: np.ndarray,
    grid: tuple[int, int, int],
    world: tuple[float, float, float],
    yaw: float,
    gain_mode: str,
    prediction_layer: EmptyPredictionLayer | SimPredictionLayer,
    sc_gain_formula: str,
    tau: float,
    raycast_stride: int,
    max_ray_length_m: float,
    voxel_size: float,
) -> dict[str, Any]:
    candidate = SimCandidateView(
        id=-1,
        grid_position=(int(grid[0]), int(grid[1]), int(grid[2])),
        world_position=(float(world[0]), float(world[1]), float(world[2])),
        yaw=float(yaw),
        valid=True,
        candidate_source="mini_rrt",
    )
    stride = max(1, int(raycast_stride))
    max_range_voxels = max(1, int(round(float(max_ray_length_m) / float(voxel_size))))
    visible = raycast_visible_voxels_observed(
        candidate,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=max(4, int(math.ceil(32 / stride))),
        num_pitch=max(3, int(math.ceil(7 / stride))),
    )
    candidate = compute_paper_gains_for_candidate(
        candidate,
        observed_state,
        prediction_layer,
        visible,
        tau=float(tau),
        sc_gain_formula="raw_count",
        sc_occ_threshold=0.7,
        sc_conf_threshold=0.3,
        sc_count_mode="raw_count",
    )
    effective_gain_sc = effective_sc_gain_from_formula(
        raw_gain_sc=float(candidate.gain_sc),
        gain_occ=float(candidate.gain_occ),
        gain_conf=float(candidate.gain_conf),
        formula=str(sc_gain_formula),
    )
    gain_hybrid_effective = float(candidate.gain_exp + effective_gain_sc)
    if gain_mode == "exp":
        gain = float(candidate.gain_exp)
    elif gain_mode == "hybrid":
        gain = float(gain_hybrid_effective)
    elif gain_mode == "sc":
        gain = float(effective_gain_sc)
    else:
        raise ValueError("gain_mode must be exp, hybrid, or sc")
    return {
        "gain": gain,
        "gain_exp": float(candidate.gain_exp),
        "gain_sc": float(candidate.gain_sc),
        "gain_hybrid": float(candidate.gain_hybrid),
        "effective_gain_sc": float(effective_gain_sc),
        "gain_hybrid_effective": float(gain_hybrid_effective),
        "gain_occ": float(candidate.gain_occ),
        "gain_conf": float(candidate.gain_conf),
        "sc_gain_formula": str(sc_gain_formula),
        "visible_count": int(candidate.visible_count),
        "measured_visible_count": int(candidate.measured_visible_count),
        "predicted_unmeasured_visible_count": int(candidate.predicted_unmeasured_visible_count),
        "frontier_count_visible": int(candidate.frontier_count_visible),
    }


def choose_best_yaw(
    observed_state: np.ndarray,
    grid: tuple[int, int, int],
    world: tuple[float, float, float],
    parent_yaw: float,
    base_yaw: float,
    num_yaw_samples: int,
    gain_mode: str,
    prediction_layer: EmptyPredictionLayer | SimPredictionLayer,
    sc_gain_formula: str,
    tau: float,
    raycast_stride: int,
    max_ray_length_m: float,
    voxel_size: float,
) -> dict[str, Any]:
    sample_count = max(1, int(num_yaw_samples))
    if sample_count == 1:
        yaw_values = [wrap_angle(base_yaw)]
    else:
        yaw_values = [wrap_angle(base_yaw + 2.0 * math.pi * i / sample_count) for i in range(sample_count)]

    best: dict[str, Any] | None = None
    evaluated: list[dict[str, Any]] = []
    for yaw in yaw_values:
        stats = score_pose_gain(
            observed_state=observed_state,
            grid=grid,
            world=world,
            yaw=float(yaw),
            gain_mode=gain_mode,
            prediction_layer=prediction_layer,
            sc_gain_formula=sc_gain_formula,
            tau=tau,
            raycast_stride=raycast_stride,
            max_ray_length_m=max_ray_length_m,
            voxel_size=voxel_size,
        )
        stats["yaw"] = float(yaw)
        stats["abs_yaw_delta_from_parent"] = abs(wrap_angle(float(yaw) - float(parent_yaw)))
        evaluated.append(stats)
        if best is None:
            best = stats
            continue
        key = (float(stats["gain"]), -float(stats["abs_yaw_delta_from_parent"]))
        best_key = (float(best["gain"]), -float(best["abs_yaw_delta_from_parent"]))
        if key > best_key:
            best = stats
    assert best is not None
    best["yaw_samples_evaluated"] = len(evaluated)
    best["yaw_sample_stats"] = evaluated
    return best


def build_mini_rrt_tree(
    observed_state: np.ndarray,
    root_grid: list[int],
    root_world: list[float],
    root_yaw: float,
    bounds: dict[str, tuple[float, float]],
    *,
    seed: int,
    num_nodes: int,
    max_extension_m: float,
    sample_mode: str,
    gain_mode: str,
    v_max: float,
    robot_radius_m: float,
    voxel_size: float,
    raycast_stride: int,
    num_yaw_samples: int,
    max_ray_length_m: float,
    sc_gain_formula: str = "raw_count",
    prediction_layer: EmptyPredictionLayer | SimPredictionLayer | None = None,
    tau: float = 0.1,
    profile: bool = False,
    min_edge_length_m: float = 0.0,
    min_root_child_length_m: float = 0.0,
    min_root_distance_m: float = 0.0,
    crop_min_length_m: float = 0.0,
    short_edge_policy: str = "allow",
    density_radius_m: float = 0.0,
    max_nodes_per_density_radius: int = 0,
) -> dict[str, Any]:
    if int(num_nodes) <= 1:
        raise ValueError("num_nodes must be greater than 1")
    if float(max_extension_m) <= 0.0:
        raise ValueError("max_extension_m must be positive")
    if float(v_max) <= 0.0:
        raise ValueError("v_max must be positive")
    if short_edge_policy not in {"allow", "reject", "crop"}:
        raise ValueError("short_edge_policy must be allow, reject, or crop")

    start_time = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    observed_state = np.asarray(observed_state)
    context = build_sampling_context(observed_state, root_grid, voxel_size, robot_radius_m)
    if prediction_layer is None:
        prediction_layer = EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape))
    if tuple(prediction_layer.shape()) != tuple(observed_state.shape):
        raise ValueError(f"prediction shape {prediction_layer.shape()} != observed_state {observed_state.shape}")

    root_xy = tuple(int(v) for v in context["snapped_root_xy"])
    root_k = nearest_free_z_for_xy(observed_state, root_xy, int(root_grid[2]))
    if root_k is None:
        root_k = int(np.clip(root_grid[2], 0, observed_state.shape[2] - 1))
    resolved_root_grid = [int(root_xy[0]), int(root_xy[1]), int(root_k)]
    resolved_root_world = [
        float(root_world[0]),
        float(root_world[1]),
        float(root_world[2]),
    ]

    tree: dict[str, MiniRRTSegment] = {
        ROOT_ID: MiniRRTSegment(
            segment_id=ROOT_ID,
            parent_id=None,
            start_grid=resolved_root_grid,
            start_world=resolved_root_world,
            end_grid=resolved_root_grid,
            end_world=resolved_root_world,
            yaw=float(root_yaw),
            local_visibility_stats={"role": "root"},
            info={"root_snapped": bool(context["root_snapped"]), "original_root_grid": root_grid},
        )
    }
    occupied_xy: set[tuple[int, int]] = {(int(resolved_root_grid[0]), int(resolved_root_grid[1]))}
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    max_attempts = max(int(num_nodes) * 60, int(num_nodes) + 100)
    attempts = 0
    score_time = 0.0
    while len(tree) < int(num_nodes) and attempts < max_attempts:
        attempts += 1
        sample_source, pool = choose_sample_pool(context, sample_mode, rng)
        if len(pool) == 0:
            rejected_rows.append({"attempt": attempts, "reason": "empty_sample_pool", "sample_source": sample_source})
            break
        target_xy_arr = pool[int(rng.integers(0, len(pool)))]
        target_xy = (int(target_xy_arr[0]), int(target_xy_arr[1]))
        target_world = grid_to_world((target_xy[0], target_xy[1], resolved_root_grid[2]), bounds, voxel_size)

        nearest_id = min(
            tree.keys(),
            key=lambda segment_id: distance_xy(tree[segment_id].end_world or [0.0, 0.0, 0.0], target_world),
        )
        parent = tree[nearest_id]
        assert parent.end_world is not None and parent.end_grid is not None
        parent_world = np.asarray(parent.end_world, dtype=np.float64)
        parent_grid = np.asarray(parent.end_grid, dtype=np.int64)
        delta_xy = np.asarray(target_world[:2], dtype=np.float64) - parent_world[:2]
        distance_to_target = float(np.linalg.norm(delta_xy))
        if distance_to_target <= EPS:
            rejected_rows.append(
                {
                    "attempt": attempts,
                    "reason": "target_same_as_nearest",
                    "sample_source": sample_source,
                    "target_xy": [int(target_xy[0]), int(target_xy[1])],
                    "nearest_id": nearest_id,
                }
            )
            continue

        step_length = min(float(max_extension_m), distance_to_target)
        traversable = np.asarray(context["traversable"], dtype=bool)
        reachable_mask = np.asarray(context["reachable_mask"], dtype=bool)

        def propose_child(candidate_step_length: float, trial_label: str) -> dict[str, Any]:
            new_xy_world = parent_world[:2] + delta_xy / distance_to_target * float(candidate_step_length)
            candidate_grid_raw = world_to_grid(
                (float(new_xy_world[0]), float(new_xy_world[1]), float(parent_world[2])),
                bounds,
                voxel_size,
                shape=observed_state.shape,
                clip=True,
            )
            candidate_xy = (int(candidate_grid_raw[0]), int(candidate_grid_raw[1]))
            snapped = False
            snap_result: dict[str, Any] | None = None
            if not (in_bounds_xy(candidate_xy, traversable.shape) and bool(traversable[candidate_xy])):
                snap_result = nearest_traversable_cell(
                    traversable,
                    candidate_xy,
                    max_radius_cells=max(2, int(math.ceil(float(max_extension_m) / float(voxel_size)))),
                )
                if not bool(snap_result.get("found", False)):
                    return {
                        "ok": False,
                        "reason": "steer_not_traversable_and_snap_failed",
                        "candidate_xy": [int(candidate_xy[0]), int(candidate_xy[1])],
                        "snap_result": snap_result,
                        "trial_label": trial_label,
                        "trial_step_length_m": float(candidate_step_length),
                    }
                cell = snap_result["cell"]
                candidate_xy = (int(cell[0]), int(cell[1]))
                snapped = True

            if not bool(reachable_mask[candidate_xy]):
                return {
                    "ok": False,
                    "reason": "steered_cell_not_in_reachable_component",
                    "candidate_xy": [int(candidate_xy[0]), int(candidate_xy[1])],
                    "snapped": snapped,
                    "trial_label": trial_label,
                    "trial_step_length_m": float(candidate_step_length),
                }

            child_k = nearest_free_z_for_xy(observed_state, candidate_xy, int(parent_grid[2]))
            if child_k is None:
                return {
                    "ok": False,
                    "reason": "no_free_z_at_candidate_xy",
                    "candidate_xy": [int(candidate_xy[0]), int(candidate_xy[1])],
                    "trial_label": trial_label,
                    "trial_step_length_m": float(candidate_step_length),
                }
            child_grid = (int(candidate_xy[0]), int(candidate_xy[1]), int(child_k))

            edge_ok, edge_reason, edge_cells = line_is_traversable(
                traversable,
                (int(parent_grid[0]), int(parent_grid[1])),
                candidate_xy,
                reachable_mask=reachable_mask,
            )
            if not edge_ok:
                return {
                    "ok": False,
                    "reason": edge_reason,
                    "candidate_grid": list(child_grid),
                    "edge_cell_count": len(edge_cells),
                    "trial_label": trial_label,
                    "trial_step_length_m": float(candidate_step_length),
                }

            child_world = grid_to_world(child_grid, bounds, voxel_size)
            segment_length = euclidean(parent.end_world, list(child_world))
            root_distance = euclidean(tree[ROOT_ID].end_world or [0.0, 0.0, 0.0], list(child_world))
            density_count = 0
            if float(density_radius_m) > 0.0 and int(max_nodes_per_density_radius) > 0:
                density_count = sum(
                    1
                    for existing in tree.values()
                    if existing.end_world is not None
                    and euclidean(existing.end_world, list(child_world)) <= float(density_radius_m)
                )
            return {
                "ok": True,
                "candidate_xy": candidate_xy,
                "child_grid": child_grid,
                "child_world": child_world,
                "snapped": snapped,
                "snap_result": snap_result,
                "edge_cells": edge_cells,
                "segment_length_m": float(segment_length),
                "root_distance_m": float(root_distance),
                "density_neighbor_count": int(density_count),
                "trial_label": trial_label,
                "trial_step_length_m": float(candidate_step_length),
            }

        crop_threshold = max(0.0, float(crop_min_length_m))
        crop_step = min(float(max_extension_m), crop_threshold) if crop_threshold > EPS else step_length
        trial_steps: list[tuple[str, float]] = [("initial", step_length)]
        if short_edge_policy == "crop" and crop_threshold > EPS and crop_step > step_length + EPS:
            trial_steps.append(("crop_min_length", crop_step))

        selected_proposal: dict[str, Any] | None = None
        last_failure: dict[str, Any] | None = None
        for trial_label, trial_step in trial_steps:
            proposal = propose_child(trial_step, trial_label)
            if not proposal.get("ok", False):
                last_failure = proposal
                continue

            child_grid = proposal["child_grid"]
            candidate_xy = proposal["candidate_xy"]
            segment_length = float(proposal["segment_length_m"])
            root_distance = float(proposal["root_distance_m"])
            can_retry_crop = (
                short_edge_policy == "crop"
                and trial_label == "initial"
                and len(trial_steps) > 1
            )

            if child_grid == tuple(int(v) for v in parent_grid.tolist()):
                last_failure = {**proposal, "reason": "same_grid_as_parent"}
                if can_retry_crop:
                    continue
                break
            if candidate_xy in occupied_xy:
                last_failure = {**proposal, "reason": "duplicate_xy"}
                if can_retry_crop:
                    continue
                break
            if short_edge_policy == "reject" and float(min_edge_length_m) > 0.0 and segment_length + EPS < float(min_edge_length_m):
                last_failure = {
                    **proposal,
                    "reason": "short_edge_rejected",
                    "min_edge_length_m": float(min_edge_length_m),
                }
                break
            if short_edge_policy == "crop" and crop_threshold > 0.0 and segment_length + EPS < crop_threshold:
                last_failure = {
                    **proposal,
                    "reason": "crop_result_shorter_than_min",
                    "crop_min_length_m": crop_threshold,
                }
                if can_retry_crop:
                    continue
                break
            if nearest_id == ROOT_ID and float(min_root_child_length_m) > 0.0 and segment_length + EPS < float(min_root_child_length_m):
                last_failure = {
                    **proposal,
                    "reason": "root_child_too_short",
                    "min_root_child_length_m": float(min_root_child_length_m),
                }
                break
            if float(min_root_distance_m) > 0.0 and root_distance + EPS < float(min_root_distance_m):
                last_failure = {
                    **proposal,
                    "reason": "root_distance_too_short",
                    "min_root_distance_m": float(min_root_distance_m),
                }
                break
            if (
                float(density_radius_m) > 0.0
                and int(max_nodes_per_density_radius) > 0
                and int(proposal["density_neighbor_count"]) >= int(max_nodes_per_density_radius)
            ):
                last_failure = {
                    **proposal,
                    "reason": "density_limit_exceeded",
                    "density_radius_m": float(density_radius_m),
                    "max_nodes_per_density_radius": int(max_nodes_per_density_radius),
                }
                break
            selected_proposal = proposal
            break

        if selected_proposal is None:
            failure = last_failure or {"reason": "candidate_selection_failed"}
            rejected_rows.append(
                {
                    "attempt": attempts,
                    "reason": failure.get("reason", "candidate_selection_failed"),
                    "sample_source": sample_source,
                    "target_xy": [int(target_xy[0]), int(target_xy[1])],
                    "nearest_id": nearest_id,
                    "distance_to_sample_m": distance_to_target,
                    "steer_step_length_m": step_length,
                    "short_edge_policy": short_edge_policy,
                    "min_edge_length_m": float(min_edge_length_m),
                    "min_root_child_length_m": float(min_root_child_length_m),
                    "min_root_distance_m": float(min_root_distance_m),
                    "crop_min_length_m": float(crop_min_length_m),
                    **{
                        key: value
                        for key, value in failure.items()
                        if key not in {"ok", "reason", "edge_cells"}
                    },
                }
            )
            continue

        candidate_xy = selected_proposal["candidate_xy"]
        child_grid = selected_proposal["child_grid"]
        child_world = selected_proposal["child_world"]
        snapped = bool(selected_proposal["snapped"])
        snap_result = selected_proposal["snap_result"]
        edge_cells = selected_proposal["edge_cells"]
        segment_length = float(selected_proposal["segment_length_m"])
        base_yaw = math.atan2(float(child_world[1] - parent_world[1]), float(child_world[0] - parent_world[0]))
        score_start = time.perf_counter()
        yaw_stats = choose_best_yaw(
            observed_state=observed_state,
            grid=child_grid,
            world=child_world,
            parent_yaw=float(parent.yaw),
            base_yaw=float(base_yaw),
            num_yaw_samples=int(num_yaw_samples),
            gain_mode=gain_mode,
            prediction_layer=prediction_layer,
            sc_gain_formula=str(sc_gain_formula),
            tau=float(tau),
            raycast_stride=int(raycast_stride),
            max_ray_length_m=float(max_ray_length_m),
            voxel_size=float(voxel_size),
        )
        score_time += time.perf_counter() - score_start

        cost = float(segment_length / max(float(v_max), EPS))
        yaw_delta = abs(wrap_angle(float(yaw_stats["yaw"]) - float(parent.yaw)))
        yaw_time = yaw_delta / max(1.0, EPS)
        child_id = f"n{len(tree):04d}"
        segment = MiniRRTSegment(
            segment_id=child_id,
            parent_id=nearest_id,
            start_grid=[int(v) for v in parent.end_grid],
            start_world=[float(v) for v in parent.end_world],
            end_grid=[int(v) for v in child_grid],
            end_world=[float(v) for v in child_world],
            yaw=float(yaw_stats["yaw"]),
            gain=float(yaw_stats["gain"]),
            gain_exp=float(yaw_stats["gain_exp"]),
            gain_sc=float(yaw_stats["gain_sc"]),
            gain_hybrid=float(yaw_stats["gain_hybrid"]),
            effective_gain_sc=float(yaw_stats["effective_gain_sc"]),
            gain_hybrid_effective=float(yaw_stats["gain_hybrid_effective"]),
            gain_occ=float(yaw_stats["gain_occ"]),
            gain_conf=float(yaw_stats["gain_conf"]),
            sc_gain_formula=str(yaw_stats["sc_gain_formula"]),
            cost=cost,
            depth=int(parent.depth + 1),
            segment_length_m=float(segment_length),
            yaw_delta=float(yaw_delta),
            yaw_time=float(yaw_time),
            local_visibility_stats={
                "visible_count": int(yaw_stats["visible_count"]),
                "measured_visible_count": int(yaw_stats["measured_visible_count"]),
                "predicted_unmeasured_visible_count": int(yaw_stats["predicted_unmeasured_visible_count"]),
                "frontier_count_visible": int(yaw_stats["frontier_count_visible"]),
                "yaw_samples_evaluated": int(yaw_stats["yaw_samples_evaluated"]),
            },
            info={
                "sample_source": sample_source,
                "target_xy": [int(target_xy[0]), int(target_xy[1])],
                "target_world": [float(v) for v in target_world],
                "nearest_id": nearest_id,
                "distance_to_sample_m": distance_to_target,
                "steer_step_length_m": step_length,
                "trial_label": selected_proposal["trial_label"],
                "trial_step_length_m": selected_proposal["trial_step_length_m"],
                "snapped": bool(snapped),
                "snap_result": snap_result,
                "edge_cell_count": len(edge_cells),
                "edge_cells": [[int(a), int(b)] for a, b in edge_cells],
                "base_yaw": float(base_yaw),
                "root_distance_m": float(selected_proposal["root_distance_m"]),
                "density_neighbor_count": int(selected_proposal["density_neighbor_count"]),
                "short_edge_policy": short_edge_policy,
                "min_edge_length_m": float(min_edge_length_m),
                "min_root_child_length_m": float(min_root_child_length_m),
                "min_root_distance_m": float(min_root_distance_m),
                "crop_min_length_m": float(crop_min_length_m),
            },
        )
        add_tree_child(tree, nearest_id, segment)
        occupied_xy.add(candidate_xy)
        accepted_rows.append(
            {
                "attempt": attempts,
                "segment_id": child_id,
                "parent_id": nearest_id,
                "sample_source": sample_source,
                "target_xy": [int(target_xy[0]), int(target_xy[1])],
                "end_grid": list(child_grid),
                "end_world": [float(v) for v in child_world],
                "yaw": float(segment.yaw),
                "gain": float(segment.gain),
                "gain_exp": float(segment.gain_exp),
                "gain_sc": float(segment.gain_sc),
                "gain_hybrid": float(segment.gain_hybrid),
                "effective_gain_sc": float(segment.effective_gain_sc),
                "gain_hybrid_effective": float(segment.gain_hybrid_effective),
                "gain_occ": float(segment.gain_occ),
                "gain_conf": float(segment.gain_conf),
                "sc_gain_formula": str(segment.sc_gain_formula),
                "cost": float(segment.cost),
                "segment_length_m": float(segment.segment_length_m),
                "root_distance_m": float(selected_proposal["root_distance_m"]),
                "density_neighbor_count": int(selected_proposal["density_neighbor_count"]),
                "trial_label": selected_proposal["trial_label"],
                "trial_step_length_m": selected_proposal["trial_step_length_m"],
                "depth": int(segment.depth),
                "visible_count": int(segment.local_visibility_stats["visible_count"]),
                "snapped": bool(snapped),
            }
        )

    warnings = compute_global_normalized_gain(tree, ROOT_ID)
    decision = select_subsequent_best(tree, ROOT_ID)
    local_child_scores = {
        child_id: (tree[child_id].gain / max(tree[child_id].cost, EPS))
        for child_id in tree[ROOT_ID].children
        if tree[child_id].cost > EPS
    }
    root_local_best = max(local_child_scores, key=local_child_scores.get) if local_child_scores else None
    root_min_cost = min(tree[ROOT_ID].children, key=lambda child_id: tree[child_id].cost) if tree[ROOT_ID].children else None
    root_max_gain = max(tree[ROOT_ID].children, key=lambda child_id: tree[child_id].gain) if tree[ROOT_ID].children else None

    profile_info = {
        "total_time_s": float(time.perf_counter() - start_time),
        "gain_scoring_time_s": float(score_time),
        "attempts": int(attempts),
        "accepted_nodes_excluding_root": int(len(tree) - 1),
        "rejected_samples": int(len(rejected_rows)),
    }
    if not profile:
        profile_info = {"profile_enabled": False, **profile_info}
    else:
        profile_info["profile_enabled"] = True

    return {
        "tree": tree,
        "sampling_context": context,
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "utility_warnings": warnings,
        "decision": decision,
        "root_local_best_child_id": root_local_best,
        "root_min_cost_child_id": root_min_cost,
        "root_max_gain_child_id": root_max_gain,
        "root_local_child_scores": local_child_scores,
        "profile": profile_info,
    }


def load_one_step_reference(inputs: dict[str, Any]) -> dict[str, Any]:
    comparison = read_json(inputs.get("one_step_comparison"))
    case = inputs.get("case") or {}
    case_json = inputs.get("case_json")
    base_dir = case_json.parent if case_json is not None else None
    baseline_candidates = read_jsonl(base_dir / "baseline_runtime" / "expert_step_candidates.jsonl") if base_dir else []
    valid_candidates = [row for row in baseline_candidates if bool(row.get("valid", False))]

    def candidate_brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": row.get("id"),
            "key": row.get("key") or _candidate_key(row.get("grid_position")),
            "grid": row.get("grid_position") or row.get("candidate_grid"),
            "world": row.get("world_position") or row.get("candidate_world"),
            "yaw": row.get("yaw"),
            "score": row.get("score") or row.get("final_score"),
            "gain_exp": row.get("gain_exp"),
            "gain_sc": row.get("gain_sc"),
            "effective_gain_sc": row.get("effective_gain_sc"),
            "path_cost": row.get("path_cost"),
        }

    one_step_baseline = comparison.get("baseline", {}).get("best_candidate")
    decoupled = comparison.get("decoupled_sc", {}).get("best_candidate")
    min_path_cost = min(valid_candidates, key=lambda row: float(row.get("path_cost", math.inf))) if valid_candidates else None
    max_gain = max(valid_candidates, key=lambda row: float(row.get("gain_exp", float("-inf")))) if valid_candidates else None
    max_effective_sc = (
        max(valid_candidates, key=lambda row: float(row.get("effective_gain_sc", float("-inf"))))
        if valid_candidates
        else None
    )
    selected_case_top = None
    if case.get("source_counterfactual_row"):
        selected_case_top = {
            "grid_position": case["source_counterfactual_row"].get("top_candidate_grid"),
            "world_position": case["source_counterfactual_row"].get("top_candidate_world"),
            "score": case["source_counterfactual_row"].get("top_score"),
            "gain_exp": case["source_counterfactual_row"].get("top_gain_exp"),
            "gain_sc": case["source_counterfactual_row"].get("top_gain_sc"),
            "effective_gain_sc": case["source_counterfactual_row"].get("top_effective_gain_sc"),
            "path_cost": case["source_counterfactual_row"].get("top_path_cost"),
            "key": case["source_counterfactual_row"].get("top_candidate_key"),
        }

    return {
        "one_step_comparison_path": str(inputs.get("one_step_comparison")) if inputs.get("one_step_comparison") else None,
        "baseline_candidate_jsonl": str(base_dir / "baseline_runtime" / "expert_step_candidates.jsonl") if base_dir else None,
        "baseline_valid_candidate_count": len(valid_candidates),
        "baseline_selected": candidate_brief(one_step_baseline),
        "decoupled_selected": candidate_brief(decoupled or selected_case_top),
        "min_path_cost_candidate": candidate_brief(min_path_cost),
        "max_gain_exp_candidate": candidate_brief(max_gain),
        "max_effective_sc_candidate": candidate_brief(max_effective_sc),
    }


def _candidate_key(grid: Any) -> str | None:
    if grid is None:
        return None
    try:
        values = [int(round(float(v))) for v in grid]
    except (TypeError, ValueError):
        return None
    return "grid:" + ",".join(str(v) for v in values)


def segment_vs_reference(segment: MiniRRTSegment | None, reference: dict[str, Any] | None) -> dict[str, Any]:
    if segment is None or reference is None:
        return {"available": False}
    ref_grid = reference.get("grid") or reference.get("grid_position")
    ref_world = reference.get("world") or reference.get("world_position")
    grid_equal = None
    distance = None
    if ref_grid is not None:
        try:
            grid_equal = [int(v) for v in segment.end_grid or []] == [int(round(float(v))) for v in ref_grid]
        except (TypeError, ValueError):
            grid_equal = None
    if ref_world is not None and segment.end_world is not None:
        try:
            distance = euclidean(segment.end_world, [float(v) for v in ref_world])
        except (TypeError, ValueError):
            distance = None
    return {
        "available": True,
        "grid_equal": grid_equal,
        "distance_m": distance,
        "segment_grid": segment.end_grid,
        "reference_grid": ref_grid,
        "segment_world": segment.end_world,
        "reference_world": ref_world,
    }


def make_comparison(
    tree: dict[str, MiniRRTSegment],
    result: dict[str, Any],
    one_step: dict[str, Any],
) -> dict[str, Any]:
    decision = result["decision"]
    selected_child = tree.get(decision.get("selected_child_id") or "")
    best_descendant = tree.get(decision.get("selected_child_best_descendant_id") or "")
    root = tree[ROOT_ID]
    root_local_best = tree.get(result.get("root_local_best_child_id") or "")
    root_min_cost = tree.get(result.get("root_min_cost_child_id") or "")
    root_max_gain = tree.get(result.get("root_max_gain_child_id") or "")

    baseline_cmp = segment_vs_reference(selected_child, one_step.get("baseline_selected"))
    decoupled_cmp = segment_vs_reference(selected_child, one_step.get("decoupled_selected"))
    best_baseline_cmp = segment_vs_reference(best_descendant, one_step.get("baseline_selected"))
    best_decoupled_cmp = segment_vs_reference(best_descendant, one_step.get("decoupled_selected"))

    selected_distance = euclidean(root.end_world or [0, 0, 0], selected_child.end_world) if selected_child else None
    best_distance = euclidean(root.end_world or [0, 0, 0], best_descendant.end_world) if best_descendant else None
    local_changed = selected_child is not None and root_local_best is not None and selected_child.segment_id != root_local_best.segment_id
    nonlocal_branch = bool(best_distance is not None and best_distance >= 1.0 and best_descendant is not None and selected_child is not None and best_descendant.segment_id != selected_child.segment_id)

    return {
        "one_step": one_step,
        "mini_rrt": {
            "selected_child": segment_brief(selected_child),
            "best_descendant": segment_brief(best_descendant),
            "root_local_best_child": segment_brief(root_local_best),
            "root_min_cost_child": segment_brief(root_min_cost),
            "root_max_gain_child": segment_brief(root_max_gain),
            "selected_child_distance_from_root_m": selected_distance,
            "best_descendant_distance_from_root_m": best_distance,
            "selected_differs_from_root_local_best": local_changed,
            "best_descendant_nonlocal": nonlocal_branch,
            "selected_branch_path": segment_path_to_root(tree, best_descendant.segment_id if best_descendant else None),
        },
        "comparisons": {
            "selected_child_vs_one_step_baseline": baseline_cmp,
            "selected_child_vs_decoupled": decoupled_cmp,
            "best_descendant_vs_one_step_baseline": best_baseline_cmp,
            "best_descendant_vs_decoupled": best_decoupled_cmp,
            "selected_child_differs_from_one_step_baseline": not bool(baseline_cmp.get("grid_equal", False)),
            "selected_child_differs_from_decoupled": not bool(decoupled_cmp.get("grid_equal", False)),
        },
        "interpretation": {
            "did_mini_rrt_find_nonlocal_branch": nonlocal_branch,
            "did_tree_utility_reduce_local_path_cost_dominance": bool(local_changed),
            "limitation": "offline static-map mini-RRT; no root rewiring, no online planner loop, no map updates along hypothetical branches",
        },
    }


def rejected_reason_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("reason", "unknown")) for row in rows).items()))


def scan_rollout_like_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        found.extend(str(path) for path in sorted(output_dir.glob(pattern)))
    return found


def write_formula_reference(path: Path) -> None:
    lines = [
        "# Offline Mini-RRT Formula Reference",
        "",
        "- `TrajectorySegment` analogue: each `MiniRRTSegment` stores parent, children, start/end grid/world, yaw, gain, cost, value, best descendant, accumulated gain/cost, and local visibility stats.",
        "- `SegmentTime` analogue: `cost = segment_length_m / v_max`; yaw delta/time is recorded but not included by default.",
        "- `GlobalNormalizedGain`: value is the best accumulated root-to-descendant `gain / cost` ratio in a segment subtree.",
        "- `SubsequentBest`: the selected action is the root immediate child whose subtree contains the highest-value descendant.",
        "- `ContinuousYawPlanningEvaluator` approximation: each new node evaluates a fixed number of yaw samples and keeps the yaw with maximum local gain.",
        "- Traversability and edge validity use only measured `observed_state`; UNKNOWN is not traversable and prediction never blocks rays or edges.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_missing_features(path: Path, gain_mode: str, prediction_loaded: bool) -> None:
    prediction_note = (
        "Optional prediction gain was loaded read-only for local gain only."
        if prediction_loaded
        else "Prediction gain was disabled; default run used measured-only gain_exp."
    )
    lines = [
        "# Missing Or Limited Features",
        "",
        "- This is not a full online planner reproduction.",
        "- No Isaac startup, rollout, online expert loop, map_predict rerun, SSCNet inference, training, RL, PPO, BC, or IL is run.",
        "- No root rewiring or branch reinsertion is implemented.",
        "- No dynamic map updates occur along hypothetical branches; gains are scored on the saved observed_state only.",
        "- Edge validity is a 2D observed-free line check, not full source ESDF collision checking.",
        "- Segment cost records yaw time but defaults to source-like segment length / v_max only.",
        "- Yaw search is a fixed discrete approximation, not the full continuous yaw evaluator.",
        f"- Gain mode: `{gain_mode}`. {prediction_note}",
        "- Prediction is never used for traversability, collision, A*, or ray blocking.",
        "- No target_lr, target_hr, scene ground truth, or simulator ground truth is used for scoring.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommendation(path: Path, comparison: dict[str, Any], tree_ok: bool) -> str:
    interp = comparison.get("interpretation", {})
    if not tree_ok:
        next_step = "tree sampling / edge-validity debug"
        reason = "tree construction failed or produced no valid selected child"
    elif bool(interp.get("did_mini_rrt_find_nonlocal_branch", False)):
        next_step = "no-prediction online one-step tree smoke, still no rollout"
        reason = "offline mini-RRT selected a nonlocal branch with a best descendant beyond the immediate local move"
    elif not bool(interp.get("did_tree_utility_reduce_local_path_cost_dominance", False)):
        next_step = "inspect gain/raycast or sampling strategy"
        reason = "tree utility still agrees with the root-local best child"
    else:
        next_step = "inspect whether the branch is robust under no-prediction one-step tree smoke"
        reason = "tree utility changed the root child but the branch is not clearly nonlocal"
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {next_step}",
        f"- reason: {reason}",
        "- still not next: rollout, RL, PPO, BC/IL training, map_predict rerun, SSCNet training, or full planner claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return next_step


def write_decision_md(path: Path, decision: dict[str, Any], tree: dict[str, MiniRRTSegment]) -> None:
    child = tree.get(decision.get("selected_child_id") or "")
    desc = tree.get(decision.get("selected_child_best_descendant_id") or "")
    lines = [
        "# SubsequentBest Decision",
        "",
        f"- selected immediate child: `{decision.get('selected_child_id')}`",
        f"- best descendant: `{decision.get('selected_child_best_descendant_id')}`",
        f"- selected child value: `{decision.get('selected_child_value')}`",
        f"- selected child accumulated gain/cost: `{decision.get('selected_child_accumulated_gain')}` / `{decision.get('selected_child_accumulated_cost')}`",
        f"- best descendant accumulated gain/cost: `{decision.get('best_descendant_accumulated_gain')}` / `{decision.get('best_descendant_accumulated_cost')}`",
        f"- reason: {decision.get('reason')}",
    ]
    if child is not None:
        lines.extend(
            [
                "",
                "## Selected Child",
                f"- grid/world: `{child.end_grid}` / `{child.end_world}`",
                f"- gain/cost/value: `{child.gain}` / `{child.cost}` / `{child.value if math.isfinite(child.value) else None}`",
            ]
        )
    if desc is not None:
        lines.extend(
            [
                "",
                "## Best Descendant",
                f"- grid/world: `{desc.end_grid}` / `{desc.end_world}`",
                f"- gain/cost/value: `{desc.gain}` / `{desc.cost}` / `{desc.value if math.isfinite(desc.value) else None}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    one = comparison.get("one_step", {})
    mini = comparison.get("mini_rrt", {})
    cmp = comparison.get("comparisons", {})
    lines = [
        "# Tree Vs One-Step Comparison",
        "",
        "## One-Step References",
        f"- baseline selected: `{one.get('baseline_selected')}`",
        f"- decoupled selected: `{one.get('decoupled_selected')}`",
        f"- min path-cost candidate: `{one.get('min_path_cost_candidate')}`",
        f"- max gain_exp candidate: `{one.get('max_gain_exp_candidate')}`",
        "",
        "## Mini-RRT",
        f"- SubsequentBest selected child: `{mini.get('selected_child')}`",
        f"- best descendant: `{mini.get('best_descendant')}`",
        f"- selected child distance from root: `{mini.get('selected_child_distance_from_root_m')}`",
        f"- best descendant distance from root: `{mini.get('best_descendant_distance_from_root_m')}`",
        f"- selected differs from root-local best: `{mini.get('selected_differs_from_root_local_best')}`",
        "",
        "## Distances",
        f"- selected child vs one-step baseline: `{cmp.get('selected_child_vs_one_step_baseline')}`",
        f"- selected child vs decoupled: `{cmp.get('selected_child_vs_decoupled')}`",
        f"- best descendant vs one-step baseline: `{cmp.get('best_descendant_vs_one_step_baseline')}`",
        f"- best descendant vs decoupled: `{cmp.get('best_descendant_vs_decoupled')}`",
        "",
        "## Interpretation",
        f"- did mini-RRT find a nonlocal branch: `{comparison.get('interpretation', {}).get('did_mini_rrt_find_nonlocal_branch')}`",
        f"- did tree utility reduce local path-cost dominance: `{comparison.get('interpretation', {}).get('did_tree_utility_reduce_local_path_cost_dominance')}`",
        f"- limitation: {comparison.get('interpretation', {}).get('limitation')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    tree = summary["tree"]
    root = summary["root"]
    decision = summary["decision"]
    comparison = summary["comparison"]
    lines = [
        "# Stage 4A-6.5i Offline Mini-RRT Tree",
        "",
        f"1. Tree built successfully? `{tree['built_successfully']}`",
        f"2. Number of accepted nodes? `{tree['accepted_nodes_excluding_root']}` excluding root, `{tree['total_nodes']}` total.",
        f"3. Number of rejected samples and reasons? `{tree['rejected_samples']}`; `{tree['rejected_reason_counts']}`.",
        f"4. Root pose/grid/world? grid `{root['grid']}`, world `{root['world']}`, yaw `{root['yaw']}`.",
        f"5. Sampling mode? `{summary['parameters']['sample_mode']}`.",
        f"6. Gain mode? `{summary['parameters']['gain_mode']}`.",
        f"7. Cost mode? `{summary['parameters']['path_cost_mode']}`.",
        f"8. Did GlobalNormalizedGain compute valid values? `{tree['global_normalized_gain_valid']}`.",
        f"9. What immediate child did SubsequentBest select? `{decision.get('selected_child_id')}`.",
        f"10. What best descendant made that child win? `{decision.get('selected_child_best_descendant_id')}`.",
        f"11. How far is selected child from root? `{comparison['mini_rrt'].get('selected_child_distance_from_root_m')}` m.",
        f"12. How far is best descendant from root? `{comparison['mini_rrt'].get('best_descendant_distance_from_root_m')}` m.",
        f"13. Does selected immediate child differ from one-step expert action? `{comparison['comparisons'].get('selected_child_differs_from_one_step_baseline')}`.",
        f"14. Does best descendant point toward a nonlocal high-gain branch? `{comparison['interpretation'].get('did_mini_rrt_find_nonlocal_branch')}`.",
        "15. Is this still offline-only? `True`.",
        f"16. What is missing before online use? {summary['limitations_short']}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_gain_value_rows(tree: dict[str, MiniRRTSegment]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment_id in sorted(tree.keys(), key=lambda item: (0 if item == ROOT_ID else 1, item)):
        segment = tree[segment_id]
        rows.append(
            {
                "segment_id": segment.segment_id,
                "parent_id": segment.parent_id,
                "depth": segment.depth,
                "end_grid": segment.end_grid,
                "end_world": segment.end_world,
                "gain": segment.gain,
                "gain_exp": segment.gain_exp,
                "gain_sc": segment.gain_sc,
                "gain_hybrid": segment.gain_hybrid,
                "effective_gain_sc": segment.effective_gain_sc,
                "gain_hybrid_effective": segment.gain_hybrid_effective,
                "gain_occ": segment.gain_occ,
                "gain_conf": segment.gain_conf,
                "sc_gain_formula": segment.sc_gain_formula,
                "cost": segment.cost,
                "local_gain_over_cost": segment.gain / max(segment.cost, EPS) if segment.cost > EPS else None,
                "local_effective_hybrid_over_cost": (
                    segment.gain_hybrid_effective / max(segment.cost, EPS) if segment.cost > EPS else None
                ),
                "accumulated_gain": segment.accumulated_gain,
                "accumulated_cost": segment.accumulated_cost,
                "value": segment.value if math.isfinite(segment.value) else None,
                "best_descendant_id": segment.best_descendant_id,
                "children_count": len(segment.children),
                "visible_count": segment.local_visibility_stats.get("visible_count"),
                "frontier_count_visible": segment.local_visibility_stats.get("frontier_count_visible"),
            }
        )
    return rows


def write_visualizations(output_dir: Path, tree: dict[str, MiniRRTSegment], result: dict[str, Any]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    context = result["sampling_context"]
    traversable = np.asarray(context["traversable"], dtype=bool)
    reachable = np.asarray(context["reachable_mask"], dtype=bool)
    frontier = np.asarray(context["reachable_frontier_mask"], dtype=bool)
    image = np.zeros(traversable.shape, dtype=np.int8)
    image[reachable] = 1
    image[frontier] = 2
    image[~traversable] = 0

    def plot_base(title: str) -> tuple[Any, Any]:
        cmap = ListedColormap(["#2f343f", "#9fc7d1", "#e5b84d"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
        ax.imshow(image.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel("grid x")
        ax.set_ylabel("grid y")
        ax.set_title(title)
        return fig, ax

    generated: dict[str, str] = {}
    fig, ax = plot_base("Offline mini-RRT tree")
    for segment in tree.values():
        if segment.parent_id is None or segment.start_grid is None or segment.end_grid is None:
            continue
        ax.plot(
            [segment.start_grid[0] + 0.5, segment.end_grid[0] + 0.5],
            [segment.start_grid[1] + 0.5, segment.end_grid[1] + 0.5],
            color="#355c7d",
            linewidth=0.7,
            alpha=0.55,
        )
    nodes = np.asarray([segment.end_grid[:2] for segment in tree.values() if segment.end_grid is not None])
    if len(nodes):
        ax.scatter(nodes[:, 0] + 0.5, nodes[:, 1] + 0.5, s=8, c="#1b4d89", alpha=0.75)
    root = tree[ROOT_ID]
    ax.scatter(root.end_grid[0] + 0.5, root.end_grid[1] + 0.5, s=80, c="#ffffff", edgecolors="#000000", label="root")
    ax.legend(loc="upper right", fontsize=8)
    path = output_dir / "mini_rrt_tree_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    generated[path.name] = str(path)

    fig, ax = plot_base("Selected branch")
    decision = result["decision"]
    branch_ids = segment_path_to_root(tree, decision.get("selected_child_best_descendant_id"))
    for a, b in zip(branch_ids, branch_ids[1:]):
        parent = tree[a]
        child = tree[b]
        ax.plot(
            [parent.end_grid[0] + 0.5, child.end_grid[0] + 0.5],
            [parent.end_grid[1] + 0.5, child.end_grid[1] + 0.5],
            color="#d1495b",
            linewidth=2.4,
        )
    for segment_id in branch_ids:
        segment = tree[segment_id]
        ax.scatter(segment.end_grid[0] + 0.5, segment.end_grid[1] + 0.5, s=35, c="#d1495b")
    path = output_dir / "selected_branch_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    generated[path.name] = str(path)

    non_root = [segment for segment in tree.values() if segment.segment_id != ROOT_ID]
    if non_root:
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        gains = np.asarray([segment.gain for segment in non_root], dtype=np.float64)
        costs = np.asarray([segment.cost for segment in non_root], dtype=np.float64)
        values = np.asarray([segment.value if math.isfinite(segment.value) else 0.0 for segment in non_root])
        scatter = ax.scatter(costs, gains, c=values, s=18, cmap="viridis", alpha=0.85)
        ax.set_xlabel("segment cost")
        ax.set_ylabel("local gain")
        ax.set_title("Gain / cost / subtree value")
        fig.colorbar(scatter, ax=ax, label="subtree value")
        path = output_dir / "gain_cost_scatter.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        generated[path.name] = str(path)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        depths = np.asarray([segment.depth for segment in non_root], dtype=np.int64)
        ax.hist(depths, bins=np.arange(depths.min(), depths.max() + 2) - 0.5, color="#4c78a8", edgecolor="white")
        ax.set_xlabel("tree depth")
        ax.set_ylabel("node count")
        ax.set_title("Value-depth histogram")
        path = output_dir / "value_depth_histogram.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        generated[path.name] = str(path)
    return generated


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_inputs(args)
    observed_path = Path(inputs["observed_state"])
    observed_hash_before = sha256_file(observed_path)
    observed_state = np.load(observed_path)
    observed_state.setflags(write=False)
    if observed_state.ndim != 3:
        raise ValueError(f"observed_state must be 3D, got {observed_state.shape}")
    resolved = load_bounds_and_root(inputs, tuple(int(v) for v in observed_state.shape), args.voxel_size)

    prediction_layer: EmptyPredictionLayer | SimPredictionLayer | None = None
    prediction_loaded = False
    prediction_npz = inputs.get("prediction_npz")
    if args.gain_mode in {"hybrid", "sc"} and prediction_npz is not None:
        prediction_layer = SimPredictionLayer.from_npz(prediction_npz)
        prediction_loaded = True
    elif args.gain_mode in {"hybrid", "sc"}:
        prediction_layer = EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape))
    else:
        prediction_layer = EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape))

    result = build_mini_rrt_tree(
        observed_state,
        resolved["root_grid"],
        resolved["root_world"],
        resolved["root_yaw"],
        resolved["bounds"],
        seed=args.seed,
        num_nodes=args.num_nodes,
        max_extension_m=args.max_extension_m,
        sample_mode=args.sample_mode,
        gain_mode=args.gain_mode,
        v_max=args.v_max,
        robot_radius_m=args.robot_radius_m,
        voxel_size=args.voxel_size,
        raycast_stride=args.raycast_stride,
        num_yaw_samples=args.num_yaw_samples,
        max_ray_length_m=args.max_ray_length_m,
        sc_gain_formula=str(getattr(args, "sc_gain_formula", "raw_count")),
        prediction_layer=prediction_layer,
        tau=args.tau,
        profile=args.profile,
        min_edge_length_m=args.min_edge_length_m,
        min_root_child_length_m=args.min_root_child_length_m,
        min_root_distance_m=args.min_root_distance_m,
        crop_min_length_m=args.crop_min_length_m,
        short_edge_policy=args.short_edge_policy,
        density_radius_m=args.density_radius_m,
        max_nodes_per_density_radius=args.max_nodes_per_density_radius,
    )
    tree: dict[str, MiniRRTSegment] = result["tree"]
    one_step = load_one_step_reference(inputs)
    comparison = make_comparison(tree, result, one_step)
    next_step = write_recommendation(
        output_dir / "recommended_next_faithful_step.md",
        comparison,
        tree_ok=bool(result["decision"].get("selected_child_id")),
    )

    observed_hash_after = sha256_file(observed_path)
    decision = dict(result["decision"])
    selected_child = tree.get(decision.get("selected_child_id") or "")
    best_descendant = tree.get(decision.get("selected_child_best_descendant_id") or "")
    if selected_child is not None:
        decision["selected_child"] = segment_brief(selected_child)
    if best_descendant is not None:
        decision["best_descendant"] = segment_brief(best_descendant)

    total = int(observed_state.size)
    unknown = int(np.count_nonzero(observed_state == UNKNOWN))
    free = int(np.count_nonzero(observed_state == FREE))
    occupied = int(total - unknown - free)
    utility_valid = any(
        segment.segment_id != ROOT_ID and math.isfinite(segment.value)
        for segment in tree.values()
    )
    summary = {
        "stage": "Stage 4A-6.5i/6.5k offline mini-RRT tree builder",
        "variant_name": str(args.variant_name or "default"),
        "inputs": {
            "case_json": str(inputs.get("case_json")) if inputs.get("case_json") else None,
            "episode_dir": str(inputs.get("episode_dir")) if inputs.get("episode_dir") else None,
            "observed_state": str(observed_path),
            "pose_json": str(inputs.get("pose_json")) if inputs.get("pose_json") else None,
            "camera_info": str(inputs.get("camera_info")) if inputs.get("camera_info") else None,
            "episode_summary": str(inputs.get("episode_summary")) if inputs.get("episode_summary") else None,
            "step_npz": str(inputs.get("step_npz")) if inputs.get("step_npz") else None,
            "prediction_npz": str(prediction_npz) if prediction_npz else None,
        },
        "parameters": {
            "seed": int(args.seed),
            "num_nodes": int(args.num_nodes),
            "max_extension_m": float(args.max_extension_m),
            "sample_mode": args.sample_mode,
            "gain_mode": args.gain_mode,
            "sc_gain_formula": str(getattr(args, "sc_gain_formula", "raw_count")),
            "path_cost_mode": args.path_cost_mode,
            "v_max": float(args.v_max),
            "yaw_rate": float(args.yaw_rate),
            "robot_radius_m": float(args.robot_radius_m),
            "voxel_size": float(args.voxel_size),
            "raycast_stride": int(args.raycast_stride),
            "num_yaw_samples": int(args.num_yaw_samples),
            "max_ray_length_m": float(args.max_ray_length_m),
            "tau": float(args.tau),
            "min_edge_length_m": float(args.min_edge_length_m),
            "min_root_child_length_m": float(args.min_root_child_length_m),
            "min_root_distance_m": float(args.min_root_distance_m),
            "crop_min_length_m": float(args.crop_min_length_m),
            "short_edge_policy": str(args.short_edge_policy),
            "density_radius_m": float(args.density_radius_m),
            "max_nodes_per_density_radius": int(args.max_nodes_per_density_radius),
            "variant_name": str(args.variant_name or "default"),
        },
        "map": {
            "bounds": resolved["bounds"],
            "observed_shape": [int(v) for v in observed_state.shape],
            "observed_state_sha256_before": observed_hash_before,
            "observed_state_sha256_after": observed_hash_after,
            "observed_state_hash_unchanged": observed_hash_before == observed_hash_after,
            "unknown_count": unknown,
            "free_count": free,
            "occupied_count": occupied,
            "observed_ratio": float((free + occupied) / total),
        },
        "root": {
            "grid": resolved["root_grid"],
            "world": resolved["root_world"],
            "yaw": float(resolved["root_yaw"]),
            "resolved_grid": tree[ROOT_ID].end_grid,
            "resolved_world": tree[ROOT_ID].end_world,
            "sampling_root_snapped": bool(result["sampling_context"]["root_snapped"]),
            "snapped_root_xy": result["sampling_context"]["snapped_root_xy"],
        },
        "tree": {
            "built_successfully": bool(len(tree) > 1 and result["decision"].get("selected_child_id") is not None),
            "total_nodes": int(len(tree)),
            "accepted_nodes_excluding_root": int(len(tree) - 1),
            "target_total_nodes": int(args.num_nodes),
            "rejected_samples": int(len(result["rejected_rows"])),
            "rejected_reason_counts": rejected_reason_counts(result["rejected_rows"]),
            "attempts": int(result["profile"]["attempts"]),
            "global_normalized_gain_valid": bool(utility_valid),
            "utility_warning_count": len(result["utility_warnings"]),
            "utility_warnings_sample": result["utility_warnings"][:10],
            "sampling": {
                "reachable_free_count": result["sampling_context"]["reachable_free_count"],
                "reachable_frontier_count": result["sampling_context"]["reachable_frontier_count"],
                "frontier_adjacent_free_count": result["sampling_context"]["frontier_adjacent_free_count"],
                "component": result["sampling_context"]["component"],
                "traversability": result["sampling_context"]["traversability_summary"],
            },
        },
        "decision": decision,
        "comparison": comparison,
        "profile": result["profile"],
        "source_faithful_components": {
            "trajectory_segment_fields": True,
            "segment_time_like_cost": "cost = segment_length_m / v_max; yaw time recorded but not mixed by default",
            "global_normalized_gain": "best accumulated root-to-descendant gain / cost in subtree",
            "subsequent_best": "root immediate child whose subtree contains highest-value segment",
            "continuous_yaw_approximation": f"{args.num_yaw_samples} fixed yaw samples per new node",
        },
        "limitations_short": "offline static-map prototype; missing online planner loop, root rewiring, dynamic branch map updates, full ESDF collision checks, and full continuous yaw planning",
        "recommended_next_faithful_step": next_step,
        "safety": {
            "isaac_startup": False,
            "rollout": False,
            "online_expert_loop": False,
            "map_predict_rerun": False,
            "sscnet_inference_or_training": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_modified": False,
            "observed_state_modified": observed_hash_before != observed_hash_after,
            "prediction_writeback": False,
            "prediction_used_for_traversability_collision": False,
            "prediction_blocks_rays": False,
            "target_lr_target_hr_ground_truth_scoring": False,
            "external_source_modified_or_built": False,
        },
    }

    generated_viz: dict[str, str] = {}
    if args.save_viz:
        generated_viz = write_visualizations(output_dir, tree, result)
    summary["generated_files"] = {
        "output_dir": str(output_dir),
        "visualizations": generated_viz,
        "rollout_like_files_created_in_output_dir": scan_rollout_like_outputs(output_dir),
    }

    write_jsonl(output_dir / "mini_rrt_tree_segments.jsonl", [segment_record(tree[key]) for key in tree])
    save_json(output_dir / "mini_rrt_tree_summary.json", summary)
    write_summary_md(output_dir / "mini_rrt_tree_summary.md", summary)
    save_json(output_dir / "subsequent_best_decision.json", decision)
    write_decision_md(output_dir / "subsequent_best_decision.md", decision, tree)
    save_json(output_dir / "tree_vs_one_step_comparison.json", comparison)
    write_comparison_md(output_dir / "tree_vs_one_step_comparison.md", comparison)
    write_csv(output_dir / "sampled_nodes.csv", result["accepted_rows"])
    write_csv(output_dir / "rejected_samples.csv", result["rejected_rows"])
    write_csv(output_dir / "gain_cost_value_table.csv", make_gain_value_rows(tree))
    write_formula_reference(output_dir / "tree_formula_reference.md")
    write_missing_features(output_dir / "missing_or_limited_features.md", args.gain_mode, prediction_loaded)

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_json", default="")
    parser.add_argument("--episode_dir", default="")
    parser.add_argument("--observed_state", default="")
    parser.add_argument("--pose_json", default="")
    parser.add_argument("--camera_info", default="")
    parser.add_argument("--episode_summary", default="")
    parser.add_argument("--prediction_npz", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--gain_mode", choices=["exp", "hybrid", "sc"], default="exp")
    parser.add_argument("--sc_gain_formula", default="raw_count")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--yaw_rate", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--min_edge_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_child_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_distance_m", type=float, default=0.0)
    parser.add_argument("--crop_min_length_m", type=float, default=0.0)
    parser.add_argument("--short_edge_policy", choices=["reject", "crop", "allow"], default="allow")
    parser.add_argument("--density_radius_m", type=float, default=0.0)
    parser.add_argument("--max_nodes_per_density_radius", type=int, default=0)
    parser.add_argument("--variant_name", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
