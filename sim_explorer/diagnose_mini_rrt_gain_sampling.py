#!/usr/bin/env python3
"""Stage 4A-6.5j offline mini-RRT gain/raycast/sampling diagnosis.

This script is diagnostic-only. It reads saved Stage 4A-6.5i mini-RRT outputs,
the saved observed_state, and external source text. It does not launch Isaac,
run rollout, rerun map_predict, train anything, modify checkpoints, modify
observed_state, modify runtime planner code, or build external source.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sim_paper_expert import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    EmptyPredictionLayer,
    SimCandidateView,
    compute_paper_gains_for_candidate,
    grid_to_world,
    normalize_bounds,
    raycast_visible_voxels_observed,
)


EPS = 1.0e-9
ROOT_ID = "root"
MIN_LENGTH_FILTERS = (0.0, 0.15, 0.25, 0.35, 0.5)
MIN_ROOT_DISTANCE_FILTERS = (0.0, 0.25, 0.5, 1.0)
COST_EXPONENTS = (0.5, 0.75, 1.0)
NOVELTY_BASES = ("raw_gain", "parent_novel_gain", "root_novel_gain")
ROLLOUT_LIKE_PATTERNS = (
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
)
REQUIRED_CONTEXT = (
    "/home/ubuntu22/sc_explorer_ws/.project_context/CURRENT_STATE.md",
    "/home/ubuntu22/sc_explorer_ws/.project_context/CODEX_LOG.md",
    "/home/ubuntu22/sc_explorer_ws/.project_context/TODO.md",
)


SOURCE_SEARCHES: dict[str, list[str]] = {
    "minimum_segment_length": [
        r"min_path_length",
        r"crop_min_length",
        r"minimum length",
        r"check min length",
        r"check minimum length",
    ],
    "maximum_extension_or_density": [
        r"max_extension_range",
        r"max_density_range",
        r"semilocal_sampling_radius",
    ],
    "gain_cache_or_visible_sets": [
        r"visible_voxels",
        r"TrajectoryInfo",
        r"SimulatedSensorInfo",
        r"storeTrajectoryInformation",
    ],
    "parent_visible_filtering": [
        r"clear_from_parents",
        r"previously seen by parent",
        r"old_voxels",
    ],
    "root_gain_discount_or_overlap": [
        r"root.visible",
        r"root[-_ ]visible",
        r"gain discount",
        r"discount near root",
    ],
    "root_rewiring": [
        r"rewireRoot",
        r"rewire_root",
        r"reinsert_root",
        r"rewireIntermediate",
        r"update_subsequent",
    ],
    "evaluator_update_or_prune": [
        r"EvaluatorUpdater",
        r"PruneDirect",
        r"ConstrainedUpdater",
        r"minimum_gain",
        r"maximum_cost",
    ],
    "trajectory_metadata": [
        r"tg_visited",
        r"\binfo\b",
        r"TrajectorySegment",
    ],
    "continuous_yaw": [
        r"ContinuousYawPlanningEvaluator",
        r"sampleYaw",
        r"n_directions",
        r"n_sections_fov",
    ],
}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        return csv_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (list, tuple, dict, np.ndarray)):
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_literal(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] in "[{(":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return None
    return value


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def finite_array(values: list[Any] | np.ndarray) -> np.ndarray:
    arr = np.asarray([as_float(v) for v in list(values)], dtype=np.float64)
    return arr[np.isfinite(arr)]


def quantile_stats(values: list[Any] | np.ndarray) -> dict[str, Any]:
    arr = finite_array(values)
    if arr.size == 0:
        return {
            "count": 0,
            "min": None,
            "p1": None,
            "p5": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def pearson(x_values: list[Any], y_values: list[Any]) -> float | None:
    pairs: list[tuple[float, float]] = []
    for x_raw, y_raw in zip(x_values, y_values):
        x = as_float(x_raw)
        y = as_float(y_raw)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y_arr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= EPS or float(np.std(y_arr)) <= EPS:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def euclidean(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(aa - bb))


def state_name(value: Any) -> str:
    ivalue = int(value)
    if ivalue == int(UNKNOWN):
        return "unknown"
    if ivalue == int(FREE):
        return "free"
    if ivalue == int(OCCUPIED):
        return "occupied"
    return f"state_{ivalue}"


def git_status_short(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "status": None, "error": "missing"}
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        return {"available": False, "path": str(path), "status": None, "error": str(exc)}
    return {
        "available": proc.returncode == 0,
        "path": str(path),
        "status": proc.stdout.splitlines(),
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def context_read_summary() -> dict[str, Any]:
    result: dict[str, Any] = {}
    patterns = {
        "stage_4a65i_complete": "Stage 4A-6.5i",
        "current_next_task": "inspect gain/raycast or sampling strategy",
        "no_rollout_boundary": "no rollout",
    }
    for raw_path in REQUIRED_CONTEXT:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        result[path.name] = {
            "read": bool(text),
            "path": str(path),
            "stage_4a65i_found": patterns["stage_4a65i_complete"] in text,
            "next_task_found": patterns["current_next_task"] in text,
            "hard_boundary_hint_found": patterns["no_rollout_boundary"].lower() in text.lower(),
        }
    return result


def load_segments(mini_rrt_dir: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = read_jsonl(mini_rrt_dir / "mini_rrt_tree_segments.jsonl")
    by_id = {str(row.get("segment_id")): row for row in rows}
    return rows, by_id


def root_child_for(segment_id: str, segments: dict[str, dict[str, Any]]) -> str | None:
    current_id: str | None = segment_id
    last_id: str | None = None
    while current_id and current_id in segments:
        parent = segments[current_id].get("parent_id")
        if parent == ROOT_ID:
            return current_id
        if parent is None:
            return last_id
        last_id = current_id
        current_id = str(parent)
    return None


def path_to_root(segment_id: str, segments: dict[str, dict[str, Any]]) -> list[str]:
    path: list[str] = []
    current_id: str | None = segment_id
    seen: set[str] = set()
    while current_id and current_id in segments and current_id not in seen:
        seen.add(current_id)
        path.append(current_id)
        parent = segments[current_id].get("parent_id")
        current_id = None if parent is None else str(parent)
    path.reverse()
    return path


def make_segment_metrics(segments: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    root = segments[ROOT_ID]
    root_world = root.get("end_world") or root.get("start_world")
    rows: list[dict[str, Any]] = []
    for segment_id, segment in sorted(segments.items()):
        if segment_id == ROOT_ID:
            continue
        length = as_float(segment.get("segment_length_m")) or 0.0
        cost = as_float(segment.get("cost")) or 0.0
        gain = as_float(segment.get("gain")) or 0.0
        value = as_float(segment.get("value"))
        end_world = segment.get("end_world") or [math.nan, math.nan, math.nan]
        distance_from_root = euclidean(end_world, root_world)
        rows.append(
            {
                "segment_id": segment_id,
                "parent_id": segment.get("parent_id"),
                "root_child_id": root_child_for(segment_id, segments),
                "is_root_child": segment.get("parent_id") == ROOT_ID,
                "depth": as_int(segment.get("depth")),
                "start_grid": segment.get("start_grid"),
                "end_grid": segment.get("end_grid"),
                "start_world": segment.get("start_world"),
                "end_world": segment.get("end_world"),
                "yaw": as_float(segment.get("yaw")),
                "gain": gain,
                "gain_exp": as_float(segment.get("gain_exp")) or 0.0,
                "gain_sc": as_float(segment.get("gain_sc")) or 0.0,
                "gain_hybrid": as_float(segment.get("gain_hybrid")) or 0.0,
                "cost": cost,
                "segment_length_m": length,
                "inverse_segment_length": None if length <= EPS else 1.0 / length,
                "local_gain_over_cost": None if cost <= EPS else gain / cost,
                "accumulated_gain": as_float(segment.get("accumulated_gain")),
                "accumulated_cost": as_float(segment.get("accumulated_cost")),
                "value": value,
                "best_descendant_id": segment.get("best_descendant_id"),
                "children_count": len(segment.get("children") or []),
                "visible_count": (segment.get("local_visibility_stats") or {}).get("visible_count"),
                "frontier_count_visible": (segment.get("local_visibility_stats") or {}).get("frontier_count_visible"),
                "distance_from_root_m": distance_from_root,
            }
        )
    return rows


def add_rank(rows: list[dict[str, Any]], key: str, rank_name: str, descending: bool) -> None:
    candidates = [row for row in rows if as_float(row.get(key)) is not None]
    candidates.sort(key=lambda row: as_float(row.get(key)) or 0.0, reverse=descending)
    for rank, row in enumerate(candidates, start=1):
        row[rank_name] = rank
    for row in rows:
        row.setdefault(rank_name, None)


def best_row(rows: list[dict[str, Any]], key: str, descending: bool = True) -> dict[str, Any] | None:
    candidates = [row for row in rows if as_float(row.get(key)) is not None]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: as_float(row.get(key)) or 0.0, reverse=descending)[0]


def segment_length_cost_diagnosis(
    metrics: list[dict[str, Any]],
    selected_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    for key, rank_name, desc in (
        ("segment_length_m", "rank_segment_length_ascending", False),
        ("gain", "rank_local_gain_descending", True),
        ("local_gain_over_cost", "rank_local_gain_over_cost_descending", True),
        ("value", "rank_value_descending", True),
        ("depth", "rank_depth_ascending", False),
        ("distance_from_root_m", "rank_distance_from_root_ascending", False),
    ):
        add_rank(metrics, key, rank_name, desc)
    for row in metrics:
        row["is_selected_child"] = row["segment_id"] == selected_id

    thresholds = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
    tiny_counts = {
        f"lt_{threshold:.2f}m".replace(".", "p"): int(
            sum((as_float(row.get("segment_length_m")) or 0.0) < threshold for row in metrics)
        )
        for threshold in thresholds
    }
    selected = next((row for row in metrics if row["segment_id"] == selected_id), None)
    root_children = [row for row in metrics if row.get("parent_id") == ROOT_ID]
    root_children_by_value = sorted(
        [row for row in root_children if as_float(row.get("value")) is not None],
        key=lambda row: as_float(row.get("value")) or -math.inf,
        reverse=True,
    )
    root_children_by_gain_cost = sorted(
        [row for row in root_children if as_float(row.get("local_gain_over_cost")) is not None],
        key=lambda row: as_float(row.get("local_gain_over_cost")) or -math.inf,
        reverse=True,
    )
    selected_root_value_rank = next(
        (idx for idx, row in enumerate(root_children_by_value, start=1) if row["segment_id"] == selected_id),
        None,
    )
    selected_root_gain_cost_rank = next(
        (idx for idx, row in enumerate(root_children_by_gain_cost, start=1) if row["segment_id"] == selected_id),
        None,
    )
    filtered_root_children = [
        row for row in root_children if (as_float(row.get("segment_length_m")) or 0.0) >= 0.20
    ]
    filtered_all = [row for row in metrics if (as_float(row.get("segment_length_m")) or 0.0) >= 0.20]
    summary = {
        "distributions": {
            "segment_length_m": quantile_stats([row.get("segment_length_m") for row in metrics]),
            "cost": quantile_stats([row.get("cost") for row in metrics]),
            "gain": quantile_stats([row.get("gain") for row in metrics]),
            "value": quantile_stats([row.get("value") for row in metrics]),
            "local_gain_over_cost": quantile_stats([row.get("local_gain_over_cost") for row in metrics]),
            "distance_from_root_m": quantile_stats([row.get("distance_from_root_m") for row in metrics]),
            "tree_depth": quantile_stats([row.get("depth") for row in metrics]),
        },
        "tiny_edge_counts": tiny_counts,
        "correlations": {
            "value_vs_inverse_segment_length": pearson(
                [row.get("value") for row in metrics],
                [row.get("inverse_segment_length") for row in metrics],
            ),
            "value_vs_gain": pearson([row.get("value") for row in metrics], [row.get("gain") for row in metrics]),
            "value_vs_distance_from_root": pearson(
                [row.get("value") for row in metrics],
                [row.get("distance_from_root_m") for row in metrics],
            ),
            "local_gain_over_cost_vs_inverse_segment_length": pearson(
                [row.get("local_gain_over_cost") for row in metrics],
                [row.get("inverse_segment_length") for row in metrics],
            ),
        },
        "selected": selected,
        "root_child_scope": {
            "root_child_count": len(root_children),
            "best_root_child_by_value": root_children_by_value[0] if root_children_by_value else None,
            "best_root_child_by_local_gain_over_cost": root_children_by_gain_cost[0]
            if root_children_by_gain_cost
            else None,
            "selected_root_child_value_rank": selected_root_value_rank,
            "selected_root_child_local_gain_over_cost_rank": selected_root_gain_cost_rank,
        },
        "filter_lt_0p2m": {
            "root_child_best_by_value_after_filter": best_row(filtered_root_children, "value"),
            "all_node_best_by_value_after_filter": best_row(filtered_all, "value"),
            "selected_removed": selected is not None
            and (as_float(selected.get("segment_length_m")) or 0.0) < 0.20,
        },
        "answers": {
            "many_tiny_edges": bool(tiny_counts["lt_0p20m"] >= max(5, int(0.05 * len(metrics)))),
            "selected_is_tiny_edge_amplified": bool(
                selected
                and (as_float(selected.get("segment_length_m")) or 0.0) < 0.20
                and selected_root_gain_cost_rank == 1
            ),
            "selected_is_root_local_best_child": bool(selected_root_value_rank == 1 or selected_root_gain_cost_rank == 1),
        },
    }
    write_csv(output_dir / "segment_length_cost_diagnosis.csv", metrics)
    save_json(output_dir / "segment_length_cost_diagnosis.json", summary)
    return summary


def make_candidate(segment: dict[str, Any], candidate_id: int = -1, yaw_override: float | None = None) -> SimCandidateView:
    return SimCandidateView(
        id=int(candidate_id),
        grid_position=tuple(int(v) for v in segment["end_grid"]),
        world_position=tuple(float(v) for v in segment["end_world"]),
        yaw=float(segment.get("yaw") if yaw_override is None else yaw_override),
        valid=True,
        candidate_source="mini_rrt_diagnosis",
    )


def raycast_sets(
    segment: dict[str, Any],
    observed_state: np.ndarray,
    *,
    max_range_voxels: int,
    num_yaw: int,
    num_pitch: int,
    yaw_override: float | None = None,
) -> dict[str, Any]:
    candidate = make_candidate(segment, yaw_override=yaw_override)
    visible = raycast_visible_voxels_observed(
        candidate,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
    )
    visible_set = {tuple(int(v) for v in voxel) for voxel in visible}
    unknown_set = {voxel for voxel in visible_set if observed_state[voxel] == UNKNOWN}
    free_set = {voxel for voxel in visible_set if observed_state[voxel] == FREE}
    occupied_set = {voxel for voxel in visible_set if observed_state[voxel] == OCCUPIED}
    scored = compute_paper_gains_for_candidate(
        candidate,
        observed_state,
        EmptyPredictionLayer(tuple(int(v) for v in observed_state.shape)),
        sorted(visible_set),
        tau=0.1,
    )
    return {
        "visible": visible_set,
        "unknown": unknown_set,
        "free": free_set,
        "occupied": occupied_set,
        "gain_exp": float(scored.gain_exp),
        "frontier_count_visible": int(scored.frontier_count_visible),
    }


def distance_histogram(distances: list[float]) -> dict[str, Any]:
    bins = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.8, math.inf]
    labels = ["0_0p5", "0p5_1p0", "1p0_1p5", "1p5_2p0", "2p0_3p0", "3p0_4p8", "gt_4p8"]
    counts = {label: 0 for label in labels}
    for value in distances:
        for idx, label in enumerate(labels):
            if bins[idx] <= value < bins[idx + 1]:
                counts[label] += 1
                break
    return {"bins_m": labels, "counts": counts, "stats": quantile_stats(distances)}


def selected_node_raycast_audit(
    segments: dict[str, dict[str, Any]],
    selected_id: str,
    observed_state: np.ndarray,
    bounds: dict[str, Any],
    voxel_size: float,
    raycast_stride: int,
    max_ray_length_m: float,
    camera_info: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected = segments[selected_id]
    root = segments[ROOT_ID]
    parent = segments[str(selected.get("parent_id"))]
    max_range_voxels = max(1, int(round(max_ray_length_m / voxel_size)))
    num_yaw = max(4, int(math.ceil(32 / max(1, raycast_stride))))
    num_pitch = max(3, int(math.ceil(7 / max(1, raycast_stride))))

    selected_sets = raycast_sets(
        selected,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
    )
    root_sets = raycast_sets(
        root,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
    )
    root_same_yaw_sets = raycast_sets(
        root,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
        yaw_override=as_float(selected.get("yaw")),
    )
    parent_sets = raycast_sets(
        parent,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
    )

    selected_unknown = selected_sets["unknown"]
    root_overlap = selected_unknown & root_sets["unknown"]
    root_same_yaw_overlap = selected_unknown & root_same_yaw_sets["unknown"]
    parent_overlap = selected_unknown & parent_sets["unknown"]
    visible_rows: list[dict[str, Any]] = []
    distances: list[float] = []
    selected_grid = np.asarray(selected["end_grid"], dtype=np.float64)
    for voxel in sorted(selected_sets["visible"]):
        state = observed_state[voxel]
        world = grid_to_world(voxel, bounds, voxel_size)
        distance_m = float(np.linalg.norm((np.asarray(voxel, dtype=np.float64) - selected_grid) * voxel_size))
        if state == UNKNOWN:
            distances.append(distance_m)
        visible_rows.append(
            {
                "i": voxel[0],
                "j": voxel[1],
                "k": voxel[2],
                "world_x": world[0],
                "world_y": world[1],
                "world_z": world[2],
                "state": int(state),
                "state_name": state_name(state),
                "distance_from_selected_m": distance_m,
                "is_unknown": bool(state == UNKNOWN),
                "root_visible_unknown": bool(voxel in root_sets["unknown"]),
                "root_same_yaw_visible_unknown": bool(voxel in root_same_yaw_sets["unknown"]),
                "parent_visible_unknown": bool(voxel in parent_sets["unknown"]),
            }
        )
    write_csv(output_dir / "selected_node_visible_voxels.csv", visible_rows)
    plot_selected_visible_topdown(
        output_dir / "selected_node_visible_topdown.png",
        observed_state,
        selected,
        root,
        selected_unknown,
        root_overlap,
        parent_overlap,
    )
    logged_gain = as_float(selected.get("gain_exp"))
    recomputed_gain = selected_sets["gain_exp"]
    audit = {
        "selected_node": {
            "segment_id": selected_id,
            "parent_id": selected.get("parent_id"),
            "grid": selected.get("end_grid"),
            "world": selected.get("end_world"),
            "yaw": as_float(selected.get("yaw")),
            "distance_from_root_m": euclidean(selected.get("end_world"), root.get("end_world")),
        },
        "raycast_parameters": {
            "raycast_stride": int(raycast_stride),
            "max_ray_length_m": float(max_ray_length_m),
            "max_range_voxels": int(max_range_voxels),
            "num_yaw": int(num_yaw),
            "num_pitch": int(num_pitch),
            "fov_yaw_deg": 90.0,
            "fov_pitch_deg": 60.0,
            "camera_info_horizontal_fov_deg": camera_info.get("horizontal_fov_deg"),
            "camera_info_max_depth": camera_info.get("max_depth"),
        },
        "counts": {
            "visible_voxel_count": int(len(selected_sets["visible"])),
            "visible_unknown_count": int(len(selected_unknown)),
            "visible_free_count": int(len(selected_sets["free"])),
            "visible_occupied_count": int(len(selected_sets["occupied"])),
            "gain_exp_recomputed": float(recomputed_gain),
            "original_logged_gain_exp": logged_gain,
            "gain_match": bool(logged_gain is not None and abs(float(logged_gain) - recomputed_gain) <= 1.0e-6),
            "root_visible_unknown_count": int(len(root_sets["unknown"])),
            "parent_visible_unknown_count": int(len(parent_sets["unknown"])),
            "root_overlap_unknown_count": int(len(root_overlap)),
            "root_same_yaw_overlap_unknown_count": int(len(root_same_yaw_overlap)),
            "parent_overlap_unknown_count": int(len(parent_overlap)),
            "root_visible_overlap_ratio": float(len(root_overlap) / max(1, len(selected_unknown))),
            "root_same_yaw_overlap_ratio": float(len(root_same_yaw_overlap) / max(1, len(selected_unknown))),
            "parent_visible_overlap_ratio": float(len(parent_overlap) / max(1, len(selected_unknown))),
            "novel_unknown_gain_vs_root": int(len(selected_unknown - root_sets["unknown"])),
            "novel_unknown_gain_vs_root_same_yaw": int(len(selected_unknown - root_same_yaw_sets["unknown"])),
            "novel_unknown_gain_vs_parent": int(len(selected_unknown - parent_sets["unknown"])),
        },
        "visible_unknown_distance_histogram_m": distance_histogram(distances),
        "interpretation": {
            "gain_reproducible": bool(logged_gain is not None and abs(float(logged_gain) - recomputed_gain) <= 1.0e-6),
            "gain_is_mostly_root_repeated": bool(len(root_overlap) / max(1, len(selected_unknown)) >= 0.5),
            "gain_is_mostly_parent_repeated": bool(len(parent_overlap) / max(1, len(selected_unknown)) >= 0.5),
            "raycast_has_parent_root_novelty_filter": False,
        },
    }
    save_json(output_dir / "selected_node_raycast_audit.json", audit)
    return audit, {
        selected_id: selected_sets,
        ROOT_ID: root_sets,
        f"{ROOT_ID}_same_selected_yaw": root_same_yaw_sets,
        str(selected.get("parent_id")): parent_sets,
    }


def plot_selected_visible_topdown(
    path: Path,
    observed_state: np.ndarray,
    selected: dict[str, Any],
    root: dict[str, Any],
    selected_unknown: set[tuple[int, int, int]],
    root_overlap: set[tuple[int, int, int]],
    parent_overlap: set[tuple[int, int, int]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    topdown = np.full(observed_state.shape[:2], -1, dtype=np.int8)
    topdown[np.any(observed_state == FREE, axis=2)] = 0
    topdown[np.any(observed_state == OCCUPIED, axis=2)] = 1
    cmap = ListedColormap(["#30343f", "#b6d7e0", "#d75f4f"])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    novel = np.asarray([voxel[:2] for voxel in selected_unknown - root_overlap - parent_overlap], dtype=np.float64)
    root_rep = np.asarray([voxel[:2] for voxel in root_overlap], dtype=np.float64)
    parent_rep = np.asarray([voxel[:2] for voxel in parent_overlap - root_overlap], dtype=np.float64)
    if len(root_rep):
        ax.scatter(root_rep[:, 0] + 0.5, root_rep[:, 1] + 0.5, s=10, c="#6f4aa8", label="root overlap")
    if len(parent_rep):
        ax.scatter(parent_rep[:, 0] + 0.5, parent_rep[:, 1] + 0.5, s=10, c="#b07aa1", label="parent overlap")
    if len(novel):
        ax.scatter(novel[:, 0] + 0.5, novel[:, 1] + 0.5, s=12, c="#f2c94c", label="novel unknown")
    root_grid = root["end_grid"]
    selected_grid = selected["end_grid"]
    ax.scatter(root_grid[0] + 0.5, root_grid[1] + 0.5, s=90, c="#ffffff", edgecolors="#000000", label="root")
    ax.scatter(selected_grid[0] + 0.5, selected_grid[1] + 0.5, s=90, c="#1b4d89", edgecolors="#ffffff", label="selected")
    ax.set_title("Selected node visible unknown overlap")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def compute_visibility_cache(
    segments: dict[str, dict[str, Any]],
    observed_state: np.ndarray,
    raycast_stride: int,
    max_ray_length_m: float,
    voxel_size: float,
    existing_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    cache = dict(existing_cache or {})
    max_range_voxels = max(1, int(round(max_ray_length_m / voxel_size)))
    num_yaw = max(4, int(math.ceil(32 / max(1, raycast_stride))))
    num_pitch = max(3, int(math.ceil(7 / max(1, raycast_stride))))
    for segment_id, segment in segments.items():
        if segment_id in cache:
            continue
        cache[segment_id] = raycast_sets(
            segment,
            observed_state,
            max_range_voxels=max_range_voxels,
            num_yaw=num_yaw,
            num_pitch=num_pitch,
        )
    return cache


def novelty_overlap_analysis(
    segments: dict[str, dict[str, Any]],
    metrics: list[dict[str, Any]],
    visibility_cache: dict[str, dict[str, Any]],
    selected_id: str,
    top_k: int,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    metric_by_id = {row["segment_id"]: row for row in metrics}
    root_unknown = visibility_cache[ROOT_ID]["unknown"]
    novelty_by_id: dict[str, dict[str, Any]] = {}
    for row in metrics:
        segment_id = row["segment_id"]
        parent_id = str(row.get("parent_id"))
        node_unknown = visibility_cache[segment_id]["unknown"]
        parent_unknown = visibility_cache.get(parent_id, {"unknown": set()})["unknown"]
        parent_overlap = node_unknown & parent_unknown
        root_overlap = node_unknown & root_unknown
        cost = as_float(row.get("cost")) or 0.0
        path_ids = [item for item in path_to_root(segment_id, segments) if item != ROOT_ID]
        union_path_unknown: set[tuple[int, int, int]] = set()
        acc_cost = 0.0
        acc_parent_novel = 0
        for path_id in path_ids:
            union_path_unknown |= visibility_cache[path_id]["unknown"]
            path_row = metric_by_id.get(path_id, {})
            acc_cost += as_float(path_row.get("cost")) or 0.0
        novelty = {
            "segment_id": segment_id,
            "parent_id": parent_id,
            "root_child_id": row.get("root_child_id"),
            "depth": row.get("depth"),
            "segment_length_m": row.get("segment_length_m"),
            "distance_from_root_m": row.get("distance_from_root_m"),
            "cost": cost,
            "local_gain": int(len(node_unknown)),
            "logged_gain": row.get("gain"),
            "visible_unknown_count": int(len(node_unknown)),
            "parent_overlap_unknown_count": int(len(parent_overlap)),
            "root_overlap_unknown_count": int(len(root_overlap)),
            "parent_overlap_unknown_ratio": float(len(parent_overlap) / max(1, len(node_unknown))),
            "root_overlap_unknown_ratio": float(len(root_overlap) / max(1, len(node_unknown))),
            "novel_unknown_gain_vs_parent": int(len(node_unknown - parent_unknown)),
            "novel_unknown_gain_vs_root": int(len(node_unknown - root_unknown)),
            "local_gain_over_cost": None if cost <= EPS else float(len(node_unknown) / cost),
            "novel_parent_gain_over_cost": None if cost <= EPS else float(len(node_unknown - parent_unknown) / cost),
            "novel_root_gain_over_cost": None if cost <= EPS else float(len(node_unknown - root_unknown) / cost),
            "original_value": row.get("value"),
            "is_selected_child": segment_id == selected_id,
            "accumulated_cost": row.get("accumulated_cost"),
            "accumulated_unique_unknown_vs_root": int(len(union_path_unknown - root_unknown)),
            "accumulated_unique_unknown_vs_root_over_cost": None
            if acc_cost <= EPS
            else float(len(union_path_unknown - root_unknown) / acc_cost),
        }
        novelty_by_id[segment_id] = novelty
    for novelty in novelty_by_id.values():
        path_ids = [item for item in path_to_root(novelty["segment_id"], segments) if item != ROOT_ID]
        acc_parent_novel = sum(novelty_by_id[path_id]["novel_unknown_gain_vs_parent"] for path_id in path_ids)
        acc_cost = sum((as_float(metric_by_id[path_id].get("cost")) or 0.0) for path_id in path_ids)
        novelty["accumulated_parent_novel_gain"] = int(acc_parent_novel)
        novelty["accumulated_parent_novel_gain_over_cost"] = None if acc_cost <= EPS else float(acc_parent_novel / acc_cost)

    novelty_rows = list(novelty_by_id.values())
    add_rank(novelty_rows, "original_value", "rank_original_value_descending", True)
    add_rank(novelty_rows, "novel_parent_gain_over_cost", "rank_parent_novel_over_cost_descending", True)
    add_rank(novelty_rows, "novel_root_gain_over_cost", "rank_root_novel_over_cost_descending", True)
    add_rank(novelty_rows, "accumulated_unique_unknown_vs_root_over_cost", "rank_accumulated_root_novel_descending", True)
    top_ids = {
        row["segment_id"]
        for row in sorted(novelty_rows, key=lambda item: as_float(item.get("original_value")) or -math.inf, reverse=True)[
            : max(1, top_k)
        ]
    }
    top_ids.add(selected_id)
    table_rows = [row for row in novelty_rows if row["segment_id"] in top_ids]
    table_rows.sort(key=lambda row: as_float(row.get("rank_original_value_descending")) or math.inf)
    write_csv(output_dir / "node_novelty_overlap_table.csv", table_rows)

    selected = novelty_by_id.get(selected_id)
    best_original = best_row(novelty_rows, "original_value")
    best_parent_novel = best_row(novelty_rows, "novel_parent_gain_over_cost")
    best_root_novel = best_row(novelty_rows, "novel_root_gain_over_cost")
    best_accumulated = best_row(novelty_rows, "accumulated_unique_unknown_vs_root_over_cost")
    min_len_rows = [row for row in novelty_rows if (as_float(row.get("segment_length_m")) or 0.0) >= 0.20]
    best_min_len = best_row(min_len_rows, "local_gain_over_cost")
    nonlocal_rows = [row for row in novelty_rows if (as_float(row.get("distance_from_root_m")) or 0.0) >= 1.0]
    summary = {
        "scope": {
            "top_k_by_value_written": int(max(1, top_k)),
            "all_nodes_analyzed": len(novelty_rows),
        },
        "selected": selected,
        "reranks": {
            "best_original_value": best_original,
            "best_parent_novel_gain_over_cost": best_parent_novel,
            "best_root_novel_gain_over_cost": best_root_novel,
            "best_gain_over_cost_with_min_segment_length_0p2m": best_min_len,
            "best_accumulated_unique_root_novel_over_cost": best_accumulated,
        },
        "answers": {
            "selected_wins_parent_novel": bool(best_parent_novel and best_parent_novel["segment_id"] == selected_id),
            "selected_wins_root_novel": bool(best_root_novel and best_root_novel["segment_id"] == selected_id),
            "top_value_mean_root_overlap_ratio": float(
                np.mean([as_float(row.get("root_overlap_unknown_ratio")) or 0.0 for row in table_rows])
            )
            if table_rows
            else None,
            "top_value_nodes_with_root_overlap_gt_0p5": int(
                sum((as_float(row.get("root_overlap_unknown_ratio")) or 0.0) > 0.5 for row in table_rows)
            ),
            "best_nonlocal_by_root_novel_gain_over_cost": best_row(nonlocal_rows, "novel_root_gain_over_cost"),
            "best_nonlocal_by_parent_novel_gain_over_cost": best_row(nonlocal_rows, "novel_parent_gain_over_cost"),
        },
    }
    save_json(output_dir / "novelty_rerank_summary.json", summary)
    write_novelty_md(output_dir / "novelty_rerank_summary.md", summary)
    return summary, novelty_by_id


def write_novelty_md(path: Path, summary: dict[str, Any]) -> None:
    selected = summary.get("selected") or {}
    reranks = summary.get("reranks") or {}
    answers = summary.get("answers") or {}
    lines = [
        "# Novelty Rerank Summary",
        "",
        f"- all nodes analyzed: `{summary.get('scope', {}).get('all_nodes_analyzed')}`",
        f"- selected node: `{selected.get('segment_id')}`",
        f"- selected local/root/parent novel gains: `{selected.get('local_gain')}` / `{selected.get('novel_unknown_gain_vs_root')}` / `{selected.get('novel_unknown_gain_vs_parent')}`",
        f"- selected root overlap ratio: `{selected.get('root_overlap_unknown_ratio')}`",
        f"- selected parent overlap ratio: `{selected.get('parent_overlap_unknown_ratio')}`",
        "",
        "## Reranks",
        f"- best original value: `{(reranks.get('best_original_value') or {}).get('segment_id')}`",
        f"- best parent-novel gain/cost: `{(reranks.get('best_parent_novel_gain_over_cost') or {}).get('segment_id')}`",
        f"- best root-novel gain/cost: `{(reranks.get('best_root_novel_gain_over_cost') or {}).get('segment_id')}`",
        f"- best min-length 0.2m local gain/cost: `{(reranks.get('best_gain_over_cost_with_min_segment_length_0p2m') or {}).get('segment_id')}`",
        f"- best accumulated unique root-novel/cost: `{(reranks.get('best_accumulated_unique_root_novel_over_cost') or {}).get('segment_id')}`",
        "",
        "## Answers",
        f"- selected still wins parent-novel rerank: `{answers.get('selected_wins_parent_novel')}`",
        f"- selected still wins root-novel rerank: `{answers.get('selected_wins_root_novel')}`",
        f"- top-value mean root overlap ratio: `{answers.get('top_value_mean_root_overlap_ratio')}`",
        f"- nonlocal best by root novelty: `{(answers.get('best_nonlocal_by_root_novel_gain_over_cost') or {}).get('segment_id')}`",
        "",
        "All reranks are diagnostic-only and do not change planner runtime.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sampling_steering_diagnosis(
    segments: dict[str, dict[str, Any]],
    rejected_rows_raw: list[dict[str, str]],
    root_world: list[float],
    bounds: dict[str, Any],
    voxel_size: float,
    output_dir: Path,
) -> dict[str, Any]:
    root_grid = segments[ROOT_ID]["end_grid"]
    rows: list[dict[str, Any]] = []
    accepted_segments = [segment for sid, segment in sorted(segments.items()) if sid != ROOT_ID]
    for segment in accepted_segments:
        info = segment.get("info") or {}
        target_world = info.get("target_world")
        snap_result = info.get("snap_result") or {}
        snap_distance_m = None
        if isinstance(snap_result, dict) and snap_result.get("distance_cells") is not None:
            snap_distance_m = (as_float(snap_result.get("distance_cells")) or 0.0) * voxel_size
        row = {
            "status": "accepted",
            "segment_id": segment.get("segment_id"),
            "parent_id": segment.get("parent_id"),
            "sample_source": info.get("sample_source"),
            "target_xy": info.get("target_xy"),
            "target_world": target_world,
            "target_distance_to_root_m": euclidean(target_world, root_world) if target_world else None,
            "nearest_node_distance_to_target_m": info.get("distance_to_sample_m"),
            "steer_length_before_snap_m": info.get("steer_step_length_m"),
            "segment_length_m": segment.get("segment_length_m"),
            "snap_distance_m": snap_distance_m,
            "snapped": info.get("snapped"),
            "end_grid": segment.get("end_grid"),
            "end_world": segment.get("end_world"),
            "distance_from_root_m": euclidean(segment.get("end_world"), root_world),
            "depth": segment.get("depth"),
            "reason": "accepted",
        }
        rows.append(row)
    for rejected in rejected_rows_raw:
        target_xy = parse_literal(rejected.get("target_xy"))
        target_world = None
        target_distance = None
        if isinstance(target_xy, list) and len(target_xy) >= 2:
            target_world = grid_to_world((int(target_xy[0]), int(target_xy[1]), int(root_grid[2])), bounds, voxel_size)
            target_distance = euclidean(target_world, root_world)
        rows.append(
            {
                "status": "rejected",
                "segment_id": rejected.get("segment_id"),
                "parent_id": rejected.get("nearest_id"),
                "sample_source": rejected.get("sample_source"),
                "target_xy": target_xy,
                "target_world": target_world,
                "target_distance_to_root_m": target_distance,
                "nearest_node_distance_to_target_m": None,
                "steer_length_before_snap_m": None,
                "segment_length_m": None,
                "snap_distance_m": None,
                "snapped": None,
                "end_grid": parse_literal(rejected.get("candidate_grid")),
                "end_world": None,
                "distance_from_root_m": None,
                "depth": None,
                "reason": rejected.get("reason"),
            }
        )
    write_csv(output_dir / "sampling_steering_diagnosis.csv", rows)

    accepted_rows = [row for row in rows if row["status"] == "accepted"]
    rejected_rows = [row for row in rows if row["status"] == "rejected"]
    sample_counts = {
        "accepted": dict(Counter(str(row.get("sample_source")) for row in accepted_rows)),
        "rejected": dict(Counter(str(row.get("sample_source")) for row in rejected_rows)),
        "all": dict(Counter(str(row.get("sample_source")) for row in rows)),
    }
    rejection_counts = dict(Counter(str(row.get("reason")) for row in rejected_rows))
    endpoints = np.asarray(
        [parse_literal(row.get("end_grid"))[:2] for row in accepted_rows if parse_literal(row.get("end_grid"))],
        dtype=np.float64,
    )
    nearest_distances: list[float] = []
    if len(endpoints) > 1:
        for idx, point in enumerate(endpoints):
            deltas = endpoints - point
            dists = np.linalg.norm(deltas, axis=1) * voxel_size
            dists[idx] = math.inf
            nearest_distances.append(float(np.min(dists)))
    x_values = [row["end_world"][0] for row in accepted_rows if row.get("end_world")]
    y_values = [row["end_world"][1] for row in accepted_rows if row.get("end_world")]
    depth_counts = dict(Counter(str(row.get("depth")) for row in accepted_rows))
    summary = {
        "counts": {
            "accepted_nodes": len(accepted_rows),
            "rejected_samples": len(rejected_rows),
            "sample_mode_counts": sample_counts,
            "rejected_reason_counts": rejection_counts,
        },
        "distributions": {
            "target_distance_to_root_m": quantile_stats([row.get("target_distance_to_root_m") for row in rows]),
            "nearest_node_distance_to_target_m": quantile_stats(
                [row.get("nearest_node_distance_to_target_m") for row in accepted_rows]
            ),
            "steer_length_before_snap_m": quantile_stats([row.get("steer_length_before_snap_m") for row in accepted_rows]),
            "final_segment_length_after_snap_m": quantile_stats([row.get("segment_length_m") for row in accepted_rows]),
            "snap_distance_m": quantile_stats([row.get("snap_distance_m") for row in accepted_rows]),
            "accepted_radius_from_root_m": quantile_stats([row.get("distance_from_root_m") for row in accepted_rows]),
            "nearest_endpoint_spacing_m": quantile_stats(nearest_distances),
            "tree_depth": quantile_stats([row.get("depth") for row in accepted_rows]),
        },
        "spatial_spread": {
            "world_x_min": min(x_values) if x_values else None,
            "world_x_max": max(x_values) if x_values else None,
            "world_y_min": min(y_values) if y_values else None,
            "world_y_max": max(y_values) if y_values else None,
            "max_distance_from_root_m": max(
                [as_float(row.get("distance_from_root_m")) or 0.0 for row in accepted_rows],
                default=None,
            ),
            "median_distance_from_root_m": quantile_stats([row.get("distance_from_root_m") for row in accepted_rows])[
                "median"
            ],
        },
        "duplicates": {
            "exact_duplicate_accepted_endpoints": len(accepted_rows) - len({json.dumps(row.get("end_grid")) for row in accepted_rows}),
            "accepted_endpoint_nearest_spacing_lt_0p15m": int(sum(value < 0.15 for value in nearest_distances)),
            "target_same_as_nearest_rejections": int(rejection_counts.get("target_same_as_nearest", 0)),
        },
        "depth_counts": depth_counts,
        "answers": {
            "far_targets_shrink_to_short_edges": bool(
                (quantile_stats([row.get("target_distance_to_root_m") for row in accepted_rows])["median"] or 0.0) > 1.0
                and (quantile_stats([row.get("segment_length_m") for row in accepted_rows])["median"] or 0.0) < 0.55
            ),
            "same_as_nearest_issue": bool(rejection_counts.get("target_same_as_nearest", 0) >= 10),
            "short_edges_common": bool(
                sum((as_float(row.get("segment_length_m")) or 0.0) < 0.20 for row in accepted_rows) >= 5
            ),
        },
    }
    save_json(output_dir / "sampling_steering_diagnosis.json", summary)
    write_sampling_md(output_dir / "sampling_rejection_summary.md", summary)
    plot_sampling_spread(output_dir / "sampling_spread_topdown.png", segments)
    return summary


def write_sampling_md(path: Path, summary: dict[str, Any]) -> None:
    counts = summary.get("counts") or {}
    answers = summary.get("answers") or {}
    lines = [
        "# Sampling And Rejection Summary",
        "",
        f"- accepted nodes: `{counts.get('accepted_nodes')}`",
        f"- rejected samples: `{counts.get('rejected_samples')}`",
        f"- sample mode counts: `{counts.get('sample_mode_counts')}`",
        f"- rejection counts: `{counts.get('rejected_reason_counts')}`",
        f"- target distance to root distribution: `{summary.get('distributions', {}).get('target_distance_to_root_m')}`",
        f"- final segment length distribution: `{summary.get('distributions', {}).get('final_segment_length_after_snap_m')}`",
        f"- accepted spatial spread: `{summary.get('spatial_spread')}`",
        "",
        "## Answers",
        f"- far targets shrink to short edges: `{answers.get('far_targets_shrink_to_short_edges')}`",
        f"- same-as-nearest issue: `{answers.get('same_as_nearest_issue')}`",
        f"- short edges common: `{answers.get('short_edges_common')}`",
        "",
        "This diagnosis is offline-only and does not modify sampling code.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_sampling_spread(path: Path, segments: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = segments[ROOT_ID]
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for segment in segments.values():
        if segment.get("segment_id") == ROOT_ID or not segment.get("start_grid") or not segment.get("end_grid"):
            continue
        start = segment["start_grid"]
        end = segment["end_grid"]
        ax.plot([start[0], end[0]], [start[1], end[1]], color="#6b8ba4", linewidth=0.7, alpha=0.55)
    pts = np.asarray([seg["end_grid"][:2] for sid, seg in segments.items() if sid != ROOT_ID], dtype=np.float64)
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#1b4d89", alpha=0.7)
    ax.scatter(root["end_grid"][0], root["end_grid"][1], s=90, c="#d1495b", edgecolors="#000000", label="root")
    ax.set_title("Accepted mini-RRT node spread")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def offline_filter_rerank(
    metrics: list[dict[str, Any]],
    novelty_by_id: dict[str, dict[str, Any]],
    selected_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for row in metrics:
        novelty = novelty_by_id.get(row["segment_id"], {})
        out = dict(row)
        out["raw_gain"] = row.get("gain")
        out["parent_novel_gain"] = novelty.get("novel_unknown_gain_vs_parent")
        out["root_novel_gain"] = novelty.get("novel_unknown_gain_vs_root")
        out["accumulated_unique_unknown_vs_root_over_cost"] = novelty.get("accumulated_unique_unknown_vs_root_over_cost")
        enriched.append(out)

    for min_len in MIN_LENGTH_FILTERS:
        for min_root_dist in MIN_ROOT_DISTANCE_FILTERS:
            for basis in NOVELTY_BASES:
                for alpha in COST_EXPONENTS:
                    candidates: list[dict[str, Any]] = []
                    for row in enriched:
                        if (as_float(row.get("segment_length_m")) or 0.0) < min_len:
                            continue
                        if (as_float(row.get("distance_from_root_m")) or 0.0) < min_root_dist:
                            continue
                        gain = as_float(row.get(basis)) or 0.0
                        cost = as_float(row.get("cost")) or 0.0
                        if cost <= EPS:
                            continue
                        score = gain / (cost ** alpha)
                        candidate = dict(row)
                        candidate["diagnostic_score"] = score
                        candidates.append(candidate)
                    best = best_row(candidates, "diagnostic_score")
                    rows.append(
                        {
                            "min_segment_length_m": min_len,
                            "min_distance_from_root_m": min_root_dist,
                            "novelty_basis": basis,
                            "cost_exponent_alpha": alpha,
                            "candidate_count": len(candidates),
                            "best_segment_id": None if best is None else best.get("segment_id"),
                            "best_root_child_id": None if best is None else best.get("root_child_id"),
                            "best_parent_id": None if best is None else best.get("parent_id"),
                            "best_depth": None if best is None else best.get("depth"),
                            "best_segment_length_m": None if best is None else best.get("segment_length_m"),
                            "best_distance_from_root_m": None if best is None else best.get("distance_from_root_m"),
                            "best_gain_basis_value": None if best is None else best.get(basis),
                            "best_cost": None if best is None else best.get("cost"),
                            "best_diagnostic_score": None if best is None else best.get("diagnostic_score"),
                            "best_is_original_selected_child": bool(best and best.get("segment_id") == selected_id),
                            "best_is_immediate_root_child": bool(best and best.get("parent_id") == ROOT_ID),
                            "best_nonlocal_ge_1m": bool(best and (as_float(best.get("distance_from_root_m")) or 0.0) >= 1.0),
                        }
                    )
    write_csv(output_dir / "offline_filter_rerank_table.csv", rows)

    def row_for(min_len: float, min_dist: float, basis: str, alpha: float) -> dict[str, Any] | None:
        for row in rows:
            if (
                abs(float(row["min_segment_length_m"]) - min_len) <= EPS
                and abs(float(row["min_distance_from_root_m"]) - min_dist) <= EPS
                and row["novelty_basis"] == basis
                and abs(float(row["cost_exponent_alpha"]) - alpha) <= EPS
            ):
                return row
        return None

    root_children = [row for row in enriched if row.get("parent_id") == ROOT_ID]

    def root_child_best_for(min_len: float, min_dist: float, basis: str, alpha: float) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for row in root_children:
            if (as_float(row.get("segment_length_m")) or 0.0) < min_len:
                continue
            if (as_float(row.get("distance_from_root_m")) or 0.0) < min_dist:
                continue
            gain = as_float(row.get(basis)) or 0.0
            cost = as_float(row.get("cost")) or 0.0
            if cost <= EPS:
                continue
            candidate = dict(row)
            candidate["diagnostic_score"] = gain / (cost ** alpha)
            candidates.append(candidate)
        best = best_row(candidates, "diagnostic_score")
        if best is None:
            return None
        return {
            "best_segment_id": best.get("segment_id"),
            "best_root_child_id": best.get("root_child_id"),
            "best_segment_length_m": best.get("segment_length_m"),
            "best_distance_from_root_m": best.get("distance_from_root_m"),
            "best_gain_basis_value": best.get(basis),
            "best_cost": best.get("cost"),
            "best_diagnostic_score": best.get("diagnostic_score"),
            "best_is_original_selected_child": best.get("segment_id") == selected_id,
            "candidate_count": len(candidates),
            "scope": "immediate_root_children",
            "min_segment_length_m": min_len,
            "min_distance_from_root_m": min_dist,
            "novelty_basis": basis,
            "cost_exponent_alpha": alpha,
        }

    first_len_changes = None
    for min_len in MIN_LENGTH_FILTERS:
        row = row_for(min_len, 0.0, "raw_gain", 1.0)
        if row and row.get("best_segment_id") != selected_id:
            first_len_changes = row
            break
    first_dist_changes = None
    for min_dist in MIN_ROOT_DISTANCE_FILTERS:
        row = row_for(0.0, min_dist, "raw_gain", 1.0)
        if row and row.get("best_segment_id") != selected_id:
            first_dist_changes = row
            break
    root_first_len_changes = None
    for min_len in MIN_LENGTH_FILTERS:
        row = root_child_best_for(min_len, 0.0, "raw_gain", 1.0)
        if row and row.get("best_segment_id") != selected_id:
            root_first_len_changes = row
            break
    root_first_dist_changes = None
    for min_dist in MIN_ROOT_DISTANCE_FILTERS:
        row = root_child_best_for(0.0, min_dist, "raw_gain", 1.0)
        if row and row.get("best_segment_id") != selected_id:
            root_first_dist_changes = row
            break
    summary = {
        "baseline_raw_cost_alpha1": row_for(0.0, 0.0, "raw_gain", 1.0),
        "min_length_first_changes_selected": first_len_changes,
        "min_distance_first_changes_selected": first_dist_changes,
        "parent_novel_no_length": row_for(0.0, 0.0, "parent_novel_gain", 1.0),
        "root_novel_no_length": row_for(0.0, 0.0, "root_novel_gain", 1.0),
        "raw_gain_cost_alpha_0p5": row_for(0.0, 0.0, "raw_gain", 0.5),
        "raw_gain_cost_alpha_0p75": row_for(0.0, 0.0, "raw_gain", 0.75),
        "raw_gain_min_len_0p5": row_for(0.5, 0.0, "raw_gain", 1.0),
        "root_child_scope": {
            "baseline_raw_cost_alpha1": root_child_best_for(0.0, 0.0, "raw_gain", 1.0),
            "min_length_first_changes_selected": root_first_len_changes,
            "min_distance_first_changes_selected": root_first_dist_changes,
            "parent_novel_no_length": root_child_best_for(0.0, 0.0, "parent_novel_gain", 1.0),
            "root_novel_no_length": root_child_best_for(0.0, 0.0, "root_novel_gain", 1.0),
            "raw_gain_cost_alpha_0p5": root_child_best_for(0.0, 0.0, "raw_gain", 0.5),
            "raw_gain_min_len_0p5": root_child_best_for(0.5, 0.0, "raw_gain", 1.0),
        },
        "answers": {
            "minimum_length_needed_to_move_off_selected": None
            if first_len_changes is None
            else first_len_changes["min_segment_length_m"],
            "minimum_distance_needed_to_move_off_selected": None
            if first_dist_changes is None
            else first_dist_changes["min_distance_from_root_m"],
            "parent_novel_moves_off_selected": bool(
                (row_for(0.0, 0.0, "parent_novel_gain", 1.0) or {}).get("best_segment_id") != selected_id
            ),
            "root_novel_moves_off_selected": bool(
                (row_for(0.0, 0.0, "root_novel_gain", 1.0) or {}).get("best_segment_id") != selected_id
            ),
            "alpha_0p5_moves_off_selected": bool(
                (row_for(0.0, 0.0, "raw_gain", 0.5) or {}).get("best_segment_id") != selected_id
            ),
            "root_child_minimum_length_needed_to_move_off_selected": None
            if root_first_len_changes is None
            else root_first_len_changes["min_segment_length_m"],
            "root_child_minimum_distance_needed_to_move_off_selected": None
            if root_first_dist_changes is None
            else root_first_dist_changes["min_distance_from_root_m"],
            "root_child_parent_novel_moves_off_selected": bool(
                (root_child_best_for(0.0, 0.0, "parent_novel_gain", 1.0) or {}).get("best_segment_id") != selected_id
            ),
            "root_child_root_novel_moves_off_selected": bool(
                (root_child_best_for(0.0, 0.0, "root_novel_gain", 1.0) or {}).get("best_segment_id") != selected_id
            ),
            "root_child_alpha_0p5_moves_off_selected": bool(
                (root_child_best_for(0.0, 0.0, "raw_gain", 0.5) or {}).get("best_segment_id") != selected_id
            ),
        },
        "interpretation": (
            "Source-like min_path_length/crop_min_length is the least ad hoc direction if tiny-edge domination is primary; "
            "parent/root novelty remains a diagnostic unless source config enables clear_from_parents or an equivalent updater."
        ),
    }
    save_json(output_dir / "offline_filter_rerank_summary.json", summary)
    write_filter_md(output_dir / "offline_filter_rerank_summary.md", summary)
    plot_rerank_comparison(output_dir / "rerank_selected_child_comparison.png", summary)
    return summary


def write_filter_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary.get("answers") or {}
    root_scope = summary.get("root_child_scope") or {}
    lines = [
        "# Offline Filter Rerank Summary",
        "",
        f"- baseline raw gain/cost: `{summary.get('baseline_raw_cost_alpha1')}`",
        f"- first min segment length that moves off selected: `{answers.get('minimum_length_needed_to_move_off_selected')}`",
        f"- first min root distance that moves off selected: `{answers.get('minimum_distance_needed_to_move_off_selected')}`",
        f"- parent novelty moves off selected: `{answers.get('parent_novel_moves_off_selected')}`",
        f"- root novelty moves off selected: `{answers.get('root_novel_moves_off_selected')}`",
        f"- cost alpha 0.5 moves off selected: `{answers.get('alpha_0p5_moves_off_selected')}`",
        "",
        "## Immediate Root-Child Scope",
        f"- baseline root-child raw gain/cost: `{root_scope.get('baseline_raw_cost_alpha1')}`",
        f"- root-child first min segment length that moves off selected: `{answers.get('root_child_minimum_length_needed_to_move_off_selected')}`",
        f"- root-child first min root distance that moves off selected: `{answers.get('root_child_minimum_distance_needed_to_move_off_selected')}`",
        f"- root-child parent novelty moves off selected: `{answers.get('root_child_parent_novel_moves_off_selected')}`",
        f"- root-child root novelty moves off selected: `{answers.get('root_child_root_novel_moves_off_selected')}`",
        f"- root-child cost alpha 0.5 moves off selected: `{answers.get('root_child_alpha_0p5_moves_off_selected')}`",
        f"- source-faithful interpretation: {summary.get('interpretation')}",
        "",
        "All filters and reranks are diagnostic-only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rerank_comparison(path: Path, summary: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    variants = [
        ("raw", summary.get("baseline_raw_cost_alpha1") or {}),
        ("minlen0.5", summary.get("raw_gain_min_len_0p5") or {}),
        ("parentNovel", summary.get("parent_novel_no_length") or {}),
        ("rootNovel", summary.get("root_novel_no_length") or {}),
        ("alpha0.5", summary.get("raw_gain_cost_alpha_0p5") or {}),
    ]
    scores = [as_float(row.get("best_diagnostic_score")) or 0.0 for _, row in variants]
    labels = [f"{name}\n{row.get('best_segment_id')}" for name, row in variants]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(labels, scores, color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"])
    ax.set_ylabel("best diagnostic score")
    ax.set_title("Diagnostic rerank winners")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def source_anti_local_check(
    external_source_dir: Path,
    external_inspection_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    file_exts = {".cpp", ".cc", ".cxx", ".h", ".hpp", ".yaml", ".yml", ".xml", ".md", ".txt"}
    hits: list[dict[str, Any]] = []
    compiled = {
        category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
        for category, patterns in SOURCE_SEARCHES.items()
    }
    if external_source_dir.exists():
        for path in sorted(external_source_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in file_exts:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue
            rel = path.relative_to(external_source_dir)
            for lineno, line in enumerate(lines, start=1):
                for category, patterns in compiled.items():
                    for regex in patterns:
                        if regex.search(line):
                            hits.append(
                                {
                                    "category": category,
                                    "pattern": regex.pattern,
                                    "file": str(rel),
                                    "line": lineno,
                                    "snippet": line.strip()[:240],
                                }
                            )
    write_csv(output_dir / "source_anti_local_hits.csv", hits)
    found_categories = sorted({hit["category"] for hit in hits})
    external_summary = read_json(external_inspection_dir / "external_tree_utility_summary.json")
    mechanism_summary = {
        "source_dir": str(external_source_dir),
        "inspection_dir": str(external_inspection_dir),
        "hit_count": len(hits),
        "found_categories": found_categories,
        "source_evidence": {
            "minimum_length_or_cropping": "minimum_segment_length" in found_categories,
            "max_extension_or_density": "maximum_extension_or_density" in found_categories,
            "parent_visible_filtering_optional": "parent_visible_filtering" in found_categories,
            "root_rewiring": "root_rewiring" in found_categories,
            "visible_voxel_info_cache": "gain_cache_or_visible_sets" in found_categories,
            "continuous_yaw": "continuous_yaw" in found_categories,
            "evaluator_update_or_prune": "evaluator_update_or_prune" in found_categories,
        },
        "not_found_or_not_proven": {
            "root_visible_overlap_filtering": "root_gain_discount_or_overlap" not in found_categories,
            "near_root_gain_discount": "root_gain_discount_or_overlap" not in found_categories,
            "mandatory_parent_visible_dedup_in_configs": not any(
                hit["category"] == "parent_visible_filtering" and "clear_from_parents: true" in hit["snippet"]
                for hit in hits
            ),
        },
        "external_inspection_context": {
            "has_summary": bool(external_summary),
            "global_normalized_gain_formula": (
                external_summary.get("answers", {})
                .get("global_normalized_gain", {})
                .get("formula")
            ),
            "subsequent_best_logic": (
                external_summary.get("answers", {})
                .get("subsequent_best", {})
                .get("best_node_branch_logic")
            ),
        },
    }
    write_source_md(output_dir / "source_anti_local_mechanisms.md", mechanism_summary, hits)
    return mechanism_summary


def write_source_md(path: Path, summary: dict[str, Any], hits: list[dict[str, Any]]) -> None:
    evidence = summary.get("source_evidence", {})
    not_found = summary.get("not_found_or_not_proven", {})

    def first_hit(category: str) -> str:
        for hit in hits:
            if hit["category"] == category:
                return f"{hit['file']}:{hit['line']} `{hit['snippet']}`"
        return "not found"

    lines = [
        "# Source Anti-Local Mechanisms",
        "",
        "## Source Evidence",
        f"- minimum segment/path length or cropping: `{evidence.get('minimum_length_or_cropping')}`; {first_hit('minimum_segment_length')}",
        f"- max extension/density controls: `{evidence.get('max_extension_or_density')}`; {first_hit('maximum_extension_or_density')}",
        f"- optional parent visible filtering: `{evidence.get('parent_visible_filtering_optional')}`; {first_hit('parent_visible_filtering')}",
        f"- root rewiring/reinsert mechanisms: `{evidence.get('root_rewiring')}`; {first_hit('root_rewiring')}",
        f"- visible voxel info storage/cache: `{evidence.get('visible_voxel_info_cache')}`; {first_hit('gain_cache_or_visible_sets')}",
        f"- continuous yaw evaluator: `{evidence.get('continuous_yaw')}`; {first_hit('continuous_yaw')}",
        f"- updater/prune mechanisms: `{evidence.get('evaluator_update_or_prune')}`; {first_hit('evaluator_update_or_prune')}",
        "",
        "## Inference",
        "- Source evidence supports min path length/cropped segment rejection and RRTStar density/rewiring as anti-local mechanisms.",
        "- Source evidence supports an optional parent-history visible-voxel clearing mechanism, but this diagnosis did not prove it is enabled in the SC-Explorer config used here.",
        "- GlobalNormalizedGain and SubsequentBest are branch utility mechanisms, not by themselves tiny-edge penalties.",
        "",
        "## Not Found Or Not Proven",
        f"- root-visible overlap filtering: `{not_found.get('root_visible_overlap_filtering')}`",
        f"- near-root gain discount: `{not_found.get('near_root_gain_discount')}`",
        f"- mandatory parent-visible dedup in configs: `{not_found.get('mandatory_parent_visible_dedup_in_configs')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_optional_plots(
    output_dir: Path,
    metrics: list[dict[str, Any]],
    novelty_by_id: dict[str, dict[str, Any]],
    selected_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = [as_float(row.get("segment_length_m")) or 0.0 for row in metrics]
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(lengths, bins=30, color="#4c78a8", edgecolor="white")
    selected = next((row for row in metrics if row["segment_id"] == selected_id), None)
    if selected:
        ax.axvline(as_float(selected.get("segment_length_m")) or 0.0, color="#d1495b", linewidth=2, label="selected")
    ax.set_xlabel("segment length m")
    ax.set_ylabel("count")
    ax.set_title("Segment length histogram")
    ax.legend()
    fig.savefig(output_dir / "segment_length_histogram.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    costs = [as_float(row.get("cost")) or 0.0 for row in metrics]
    gains = [as_float(row.get("gain")) or 0.0 for row in metrics]
    values = [as_float(row.get("value")) or 0.0 for row in metrics]
    scatter = ax.scatter(costs, gains, c=values, cmap="viridis", s=22, alpha=0.8)
    if selected:
        ax.scatter([selected.get("cost")], [selected.get("gain")], s=100, c="#d1495b", edgecolors="#ffffff", label="selected")
    ax.set_xlabel("cost")
    ax.set_ylabel("gain")
    ax.set_title("Gain vs cost")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="value")
    fig.savefig(output_dir / "gain_vs_cost_scatter_annotated.png", dpi=170)
    plt.close(fig)

    novelty_rows = list(novelty_by_id.values())
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    overlaps = [as_float(row.get("root_overlap_unknown_ratio")) or 0.0 for row in novelty_rows]
    vals = [as_float(row.get("original_value")) or 0.0 for row in novelty_rows]
    dists = [as_float(row.get("distance_from_root_m")) or 0.0 for row in novelty_rows]
    scatter = ax.scatter(overlaps, vals, c=dists, cmap="plasma", s=22, alpha=0.8)
    selected_novelty = novelty_by_id.get(selected_id)
    if selected_novelty:
        ax.scatter(
            [selected_novelty.get("root_overlap_unknown_ratio")],
            [selected_novelty.get("original_value")],
            s=100,
            c="#d1495b",
            edgecolors="#ffffff",
            label="selected",
        )
    ax.set_xlabel("root overlap unknown ratio")
    ax.set_ylabel("original value")
    ax.set_title("Root overlap vs value")
    ax.legend()
    fig.colorbar(scatter, ax=ax, label="distance from root m")
    fig.savefig(output_dir / "root_overlap_vs_value.png", dpi=170)
    plt.close(fig)


def scan_rollout_like_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        found.extend(str(path) for path in sorted(output_dir.glob(pattern)))
    return found


def write_recommended_next(path: Path, summary: dict[str, Any]) -> str:
    segment = summary["segment_length_cost"]["answers"]
    filter_answers = summary["offline_filter_rerank"]["answers"]
    audit_counts = summary["selected_node_audit"]["counts"]
    source_evidence = summary["source_anti_local"]["source_evidence"]
    sampling_answers = summary["sampling_steering"]["answers"]
    if bool(segment.get("selected_is_tiny_edge_amplified")) and source_evidence.get("minimum_length_or_cropping"):
        next_step = "offline mini-RRT minimum-edge-length variant, no Isaac"
        reason = "selected value is amplified by a 0.1118m edge and source evidence includes min_path_length/crop_min_length controls"
    elif bool(filter_answers.get("parent_novel_moves_off_selected")) or bool(filter_answers.get("root_novel_moves_off_selected")):
        next_step = "offline novelty-gain variant, no Isaac"
        reason = "parent/root novelty rerank changes the diagnostic winner"
    elif not bool(audit_counts.get("gain_match")):
        next_step = "fix/review raycast implementation, no rollout"
        reason = "selected node gain did not reproduce from saved observed_state"
    elif bool(sampling_answers.get("same_as_nearest_issue")):
        next_step = "offline sampling strategy variant"
        reason = "sampling produced many same-as-nearest rejections and short edges"
    else:
        next_step = "inspect source more before any online smoke"
        reason = "no single diagnostic isolated a faithful offline fix"
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {next_step}",
        f"- reason: {reason}",
        "- still not next: rollout, RL, PPO, BC/IL training, map_predict rerun, SSCNet training, or coverage claims.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return next_step


def write_main_summary_md(path: Path, summary: dict[str, Any]) -> None:
    selected = summary["selected_node_audit"]["selected_node"]
    audit = summary["selected_node_audit"]["counts"]
    length = summary["segment_length_cost"]
    sampling = summary["sampling_steering"]
    rerank = summary["offline_filter_rerank"]
    rerank_answers = rerank.get("answers", {})
    source = summary["source_anti_local"]
    lines = [
        "# Stage 4A-6.5j Gain/Raycast/Sampling Summary",
        "",
        "## Required Answers",
        f"1. selected child `n0140` won because it has logged gain `{audit.get('original_logged_gain_exp')}` over a very short cost/length `{selected.get('distance_from_root_m')}` m, producing the largest gain/cost value.",
        f"2. `gain=32` reproducible: `{audit.get('gain_match')}`; recomputed gain `{audit.get('gain_exp_recomputed')}`.",
        f"3. selected visible unknown overlap: root `{audit.get('root_overlap_unknown_count')}` / `{audit.get('visible_unknown_count')}`, parent `{audit.get('parent_overlap_unknown_count')}` / `{audit.get('visible_unknown_count')}`.",
        "4. current raycast has no parent/root novelty filtering; it counts visible unknowns from the current pose on the saved map.",
        f"5. tiny-edge domination: `{length.get('answers', {}).get('selected_is_tiny_edge_amplified')}`; segment-length stats `{length.get('distributions', {}).get('segment_length_m')}`.",
        f"6. sampling same-as-nearest/short-edge issue: `{sampling.get('answers', {}).get('same_as_nearest_issue')}` / `{sampling.get('answers', {}).get('short_edges_common')}`.",
        f"7. min segment length filter first moves the immediate root-child winner off selected at `{rerank_answers.get('root_child_minimum_length_needed_to_move_off_selected')}` m.",
        f"8. parent/root novelty root-child rerank moves off selected: `{rerank_answers.get('root_child_parent_novel_moves_off_selected')}` / `{rerank_answers.get('root_child_root_novel_moves_off_selected')}`.",
        f"9. source evidence for anti-local mechanisms: `{source.get('source_evidence')}`; not found/proven `{source.get('not_found_or_not_proven')}`.",
        f"10. recommended next: `{summary.get('recommended_next_step')}`.",
        "",
        "## Diagnosis",
        f"- cause of root-local collapse: {summary.get('main_diagnosis', {}).get('cause_of_root_local_collapse')}",
        f"- raycast/gain suspect: `{summary.get('main_diagnosis', {}).get('raycast_gain_suspect')}`",
        f"- sampling suspect: `{summary.get('main_diagnosis', {}).get('sampling_suspect')}`",
        f"- cost/value suspect: `{summary.get('main_diagnosis', {}).get('cost_value_suspect')}`",
        f"- recommended fix direction: {summary.get('main_diagnosis', {}).get('recommended_fix_direction')}",
        "",
        "All rerank/filter results are diagnostic-only. This stage does not modify planner runtime and does not claim coverage improvement.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini_rrt_dir", required=True)
    parser.add_argument("--case_json", required=True)
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--external_source_dir", required=True)
    parser.add_argument("--external_inspection_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k_nodes", type=int, default=50)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    mini_rrt_dir = Path(args.mini_rrt_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    external_source_dir = Path(args.external_source_dir).resolve()
    external_inspection_dir = Path(args.external_inspection_dir).resolve()
    external_status_before = git_status_short(external_source_dir)

    context_summary = context_read_summary()
    mini_summary = read_json(mini_rrt_dir / "mini_rrt_tree_summary.json")
    decision = read_json(mini_rrt_dir / "subsequent_best_decision.json")
    _, segments = load_segments(mini_rrt_dir)
    if ROOT_ID not in segments:
        raise ValueError("mini-RRT segments missing root")
    selected_id = str(decision.get("selected_child_id") or "n0140")
    if selected_id not in segments:
        raise ValueError(f"selected segment missing from tree: {selected_id}")

    observed_path = Path(mini_summary["inputs"]["observed_state"]).resolve()
    observed_hash_before = sha256_file(observed_path)
    observed_state = np.load(observed_path)
    observed_state.setflags(write=False)
    params = mini_summary.get("parameters") or {}
    voxel_size = float(params.get("voxel_size", 0.1))
    bounds = normalize_bounds(mini_summary.get("map", {}).get("bounds"))
    camera_info_path = Path(mini_summary.get("inputs", {}).get("camera_info") or Path(args.episode_dir) / "camera_info.json")
    camera_info = read_json(camera_info_path)

    metrics = make_segment_metrics(segments)
    segment_diag = segment_length_cost_diagnosis(metrics, selected_id, output_dir)
    selected_audit, seed_visibility = selected_node_raycast_audit(
        segments,
        selected_id,
        observed_state,
        bounds,
        voxel_size,
        int(args.raycast_stride),
        float(args.max_ray_length_m),
        camera_info,
        output_dir,
    )
    visibility_cache = compute_visibility_cache(
        segments,
        observed_state,
        int(args.raycast_stride),
        float(args.max_ray_length_m),
        voxel_size,
        existing_cache=seed_visibility,
    )
    novelty_summary, novelty_by_id = novelty_overlap_analysis(
        segments,
        metrics,
        visibility_cache,
        selected_id,
        int(args.top_k_nodes),
        output_dir,
    )
    rejected_rows = read_csv(mini_rrt_dir / "rejected_samples.csv")
    root_world = [float(v) for v in segments[ROOT_ID]["end_world"]]
    sampling_summary = sampling_steering_diagnosis(
        segments,
        rejected_rows,
        root_world,
        bounds,
        voxel_size,
        output_dir,
    )
    filter_summary = offline_filter_rerank(metrics, novelty_by_id, selected_id, output_dir)
    source_summary = source_anti_local_check(external_source_dir, external_inspection_dir, output_dir)
    write_optional_plots(output_dir, metrics, novelty_by_id, selected_id)

    observed_hash_after = sha256_file(observed_path)
    external_status_after = git_status_short(external_source_dir)
    summary: dict[str, Any] = {
        "stage": "Stage 4A-6.5j offline mini-RRT gain/raycast and sampling diagnosis",
        "context_read": context_summary,
        "inputs": {
            "mini_rrt_dir": str(mini_rrt_dir),
            "case_json": str(Path(args.case_json).resolve()),
            "episode_dir": str(Path(args.episode_dir).resolve()),
            "observed_state": str(observed_path),
            "external_source_dir": str(external_source_dir),
            "external_inspection_dir": str(external_inspection_dir),
        },
        "parameters": {
            "top_k_nodes": int(args.top_k_nodes),
            "raycast_stride": int(args.raycast_stride),
            "max_ray_length_m": float(args.max_ray_length_m),
            "voxel_size": float(voxel_size),
        },
        "selected_node_audit": selected_audit,
        "segment_length_cost": segment_diag,
        "novelty_rerank": novelty_summary,
        "sampling_steering": sampling_summary,
        "offline_filter_rerank": filter_summary,
        "source_anti_local": source_summary,
        "safety": {
            "isaac_startup": False,
            "rollout": False,
            "online_expert_loop": False,
            "map_predict_rerun": False,
            "sscnet_inference_or_training": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_modified": False,
            "observed_state_modified": observed_hash_before != observed_hash_after,
            "observed_state_sha256_before": observed_hash_before,
            "observed_state_sha256_after": observed_hash_after,
            "prediction_writeback": False,
            "prediction_used_for_collision_traversability": False,
            "target_lr_target_hr_ground_truth_scoring": False,
            "external_source_modified_or_built": external_status_before.get("status") != external_status_after.get("status"),
            "external_source_git_status_before": external_status_before,
            "external_source_git_status_after": external_status_after,
            "rollout_like_outputs": scan_rollout_like_outputs(output_dir),
        },
    }
    summary["recommended_next_step"] = write_recommended_next(output_dir / "recommended_next_faithful_step.md", summary)
    selected_is_tiny = bool(segment_diag.get("answers", {}).get("selected_is_tiny_edge_amplified"))
    gain_mismatch = not bool(selected_audit.get("counts", {}).get("gain_match"))
    sampling_suspect = bool(sampling_summary.get("answers", {}).get("same_as_nearest_issue")) or bool(
        sampling_summary.get("answers", {}).get("short_edges_common")
    )
    summary["main_diagnosis"] = {
        "cause_of_root_local_collapse": (
            "the selected child combines a reproducible local unknown gain with an extremely short segment cost, "
            "and the current offline mini-RRT does not apply source-like min_path_length/crop_min_length or novelty clearing"
        ),
        "raycast_gain_suspect": bool(gain_mismatch),
        "sampling_suspect": sampling_suspect,
        "cost_value_suspect": selected_is_tiny,
        "recommended_fix_direction": summary["recommended_next_step"],
    }
    save_json(output_dir / "stage4a65j_gain_raycast_sampling_summary.json", summary)
    write_main_summary_md(output_dir / "stage4a65j_gain_raycast_sampling_summary.md", summary)
    print(json.dumps(to_jsonable({
        "stage": summary["stage"],
        "selected": selected_audit["selected_node"],
        "gain_match": selected_audit["counts"]["gain_match"],
        "recommended_next_step": summary["recommended_next_step"],
        "observed_state_modified": summary["safety"]["observed_state_modified"],
        "external_source_modified_or_built": summary["safety"]["external_source_modified_or_built"],
    }), indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    run(parse_args())
