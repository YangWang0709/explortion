#!/usr/bin/env python3
"""Stage 4A-6.5ab synthetic map_predict calibration smoke.

This runner is offline-only. It reads the saved Stage 4A-6.5aa synthetic
hidden-room frame, read-only Oracle/map_predict NPZ prediction layers, and the
saved 6.5aa raw mini-RRT trees. It replays tree decisions under threshold and
utility variants, then writes compact diagnostics. It does not start Isaac,
capture frames, rerun map_predict, run SSCNet inference, execute actions, or
modify any existing observed_state/prediction artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import MiniRRTSegment, ROOT_ID, segment_path_to_root
from sim_prediction_layer import SimPredictionLayer


DEFAULT_STAGE4A65AA_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65aa_synthetic_sc_validation"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ab_synthetic_calibration_smoke"
)

UNKNOWN = -1
OCCUPIED = 1
EPS = 1.0e-9


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
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
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
    fields_out = list(field_order or [])
    for row in rows:
        for key in row:
            if key not in fields_out:
                fields_out.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields_out:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields_out)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields_out})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("at least one integer is required")
    return values


def parse_floats(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("at least one float is required")
    return values


def normalize_bounds(raw: dict[str, Any]) -> dict[str, tuple[float, float]]:
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def region_to_slices(
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    shape: tuple[int, int, int],
    region: dict[str, list[float]],
) -> tuple[slice, slice, slice]:
    slices: list[slice] = []
    for axis, limit in zip(("x", "y", "z"), shape):
        lo, hi = sorted((float(region[axis][0]), float(region[axis][1])))
        start = int(math.floor((lo - bounds[axis][0]) / float(voxel_size)))
        stop = int(math.ceil((hi - bounds[axis][0]) / float(voxel_size)))
        slices.append(slice(max(0, start), min(int(limit), stop)))
    return tuple(slices)  # type: ignore[return-value]


def region_mask(
    shape: tuple[int, int, int],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    region: dict[str, list[float]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[region_to_slices(bounds, voxel_size, shape, region)] = True
    return mask


def make_frontier_local_mask(
    shape: tuple[int, int, int],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    scene_metadata: dict[str, Any],
) -> np.ndarray:
    regions = scene_metadata.get("diagnostic_regions", {})
    mask = np.zeros(shape, dtype=bool)
    if "measured_doorway_corridor" in regions:
        mask |= region_mask(shape, bounds, voxel_size, regions["measured_doorway_corridor"])
    # Source-inspired diagnostic zone around the hidden doorway, not a runtime
    # ground-truth formula. It stays in summary files as diagnostic/source-inspired.
    doorway = {
        "x": [-1.25, 1.55],
        "y": [-1.25, 1.25],
        "z": [0.0, 2.2],
    }
    mask |= region_mask(shape, bounds, voxel_size, doorway)
    return mask


def load_tree(tree_dir: Path) -> dict[str, MiniRRTSegment]:
    rows = read_jsonl(tree_dir / "mini_rrt_tree_segments.jsonl")
    if not rows:
        raise FileNotFoundError(f"missing tree segments: {tree_dir}")
    allowed = {field.name for field in fields(MiniRRTSegment)}
    tree: dict[str, MiniRRTSegment] = {}
    for row in rows:
        clean = {key: value for key, value in row.items() if key in allowed}
        if clean.get("value") is None:
            clean["value"] = float("-inf")
        segment = MiniRRTSegment(**clean)
        tree[str(segment.segment_id)] = segment
    if ROOT_ID not in tree:
        raise ValueError(f"tree missing root: {tree_dir}")
    return tree


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


def prediction_visible_voxels(segment: MiniRRTSegment, observed_state: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Visible voxels for prediction-count diagnostics.

    Measured OCCUPIED voxels block rays. UNKNOWN does not block this diagnostic
    visibility pass. Prediction never blocks rays.
    """

    origin = np.asarray(segment.end_grid, dtype=np.float64) + 0.5
    start_voxel = tuple(int(v) for v in segment.end_grid or (0, 0, 0))
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
            if voxel != start_voxel:
                visible.add(voxel)
            if observed_state[voxel] == OCCUPIED:
                break
            distance += 0.5
    if not visible:
        return np.zeros((0, 3), dtype=np.int16)
    return np.asarray(sorted(visible), dtype=np.int16)


def precompute_segment_prediction_arrays(
    tree: dict[str, MiniRRTSegment],
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    hidden_mask: np.ndarray,
    frontier_local_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        voxels = prediction_visible_voxels(segment, observed_state, args)
        if voxels.size == 0:
            out[node_id] = {
                "confidence": np.zeros(0, dtype=np.float32),
                "occupied_prob": np.zeros(0, dtype=np.float32),
                "free_prob": np.zeros(0, dtype=np.float32),
                "hidden": np.zeros(0, dtype=bool),
                "frontier_local": np.zeros(0, dtype=bool),
            }
            continue
        x, y, z = voxels[:, 0], voxels[:, 1], voxels[:, 2]
        candidate_mask = (observed_state[x, y, z] == UNKNOWN) & prediction.valid[x, y, z]
        x = x[candidate_mask]
        y = y[candidate_mask]
        z = z[candidate_mask]
        out[node_id] = {
            "confidence": np.asarray(prediction.confidence[x, y, z], dtype=np.float32),
            "occupied_prob": np.asarray(prediction.occupied_prob[x, y, z], dtype=np.float32),
            "free_prob": np.asarray(prediction.free_prob[x, y, z], dtype=np.float32),
            "hidden": np.asarray(hidden_mask[x, y, z], dtype=bool),
            "frontier_local": np.asarray(frontier_local_mask[x, y, z], dtype=bool),
        }
    return out


def segment_metric(
    arrays: dict[str, np.ndarray],
    confidence_threshold: float,
    occ_threshold: float,
    free_threshold: float,
    basis: str,
) -> dict[str, float]:
    conf = arrays["confidence"]
    if conf.size == 0:
        return {
            "sc_gain": 0.0,
            "source_occ_count": 0.0,
            "source_free_count": 0.0,
            "source_occ_free_count": 0.0,
            "confidence_sum": 0.0,
            "confidence_mean": 0.0,
            "hidden_region_predicted_count": 0.0,
            "frontier_local_predicted_count": 0.0,
        }
    keep = conf >= float(confidence_threshold)
    occ = keep & (arrays["occupied_prob"] >= float(occ_threshold))
    free = keep & (arrays["free_prob"] >= float(free_threshold))
    occ_free = occ | free
    hidden = arrays["hidden"]
    frontier = arrays["frontier_local"]

    source_occ_count = float(np.count_nonzero(occ))
    source_free_count = float(np.count_nonzero(free))
    source_occ_free_count = float(np.count_nonzero(occ_free))
    hidden_count = float(np.count_nonzero(occ_free & hidden))
    frontier_count = float(np.count_nonzero(occ_free & frontier))
    confidence_sum = float(np.sum(conf[occ_free], dtype=np.float64))
    confidence_mean = float(confidence_sum / max(1.0, source_occ_free_count))

    if basis == "source_occ_free":
        sc_gain = source_occ_free_count
    elif basis == "source_occ_only":
        sc_gain = source_occ_count
    elif basis == "source_free_only":
        sc_gain = source_free_count
    elif basis == "confidence_weighted_occ_free":
        sc_gain = confidence_sum
    elif basis == "hidden_region_occ_free":
        sc_gain = hidden_count
    elif basis == "frontier_local_occ_free":
        sc_gain = frontier_count
    else:
        raise ValueError(f"unsupported SC basis: {basis}")

    return {
        "sc_gain": float(sc_gain),
        "source_occ_count": source_occ_count,
        "source_free_count": source_free_count,
        "source_occ_free_count": source_occ_free_count,
        "confidence_sum": confidence_sum,
        "confidence_mean": confidence_mean,
        "hidden_region_predicted_count": hidden_count,
        "frontier_local_predicted_count": frontier_count,
    }


def rank_values(rows: list[dict[str, Any]], field_name: str, *, descending: bool) -> dict[str, int]:
    def value(row: dict[str, Any]) -> float:
        try:
            return float(row.get(field_name))
        except (TypeError, ValueError):
            return float("-inf") if descending else float("inf")

    ordered = sorted(rows, key=value, reverse=descending)
    ranks: dict[str, int] = {}
    last_value: float | None = None
    last_rank = 0
    for index, row in enumerate(ordered, start=1):
        current = value(row)
        if last_value is None or abs(current - last_value) > 1.0e-12:
            last_value = current
            last_rank = index
        ranks[str(row["node_id"])] = int(last_rank)
    return ranks


def classify_branch(row: dict[str, Any]) -> str:
    root = row.get("root_world") or [0.0, 0.0, 0.0]
    best = row.get("best_descendant_world") or row.get("selected_child_world")
    if best is None:
        return "unknown"
    dx = float(best[0]) - float(root[0])
    dy = float(best[1]) - float(root[1])
    dist = math.hypot(dx, dy)
    if dx > 0.60 and abs(float(best[1])) <= 1.20:
        return "toward_hidden_room"
    if dy > 0.65 or float(best[1]) > 1.25:
        return "toward_measured_frontier"
    if dist < 0.85:
        return "local_jitter"
    return "other"


def path_candidate_rows(
    tree: dict[str, MiniRRTSegment],
    segment_arrays: dict[str, dict[str, np.ndarray]] | None,
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    root = tree[ROOT_ID]
    rows: list[dict[str, Any]] = []
    metric_cache: dict[str, dict[str, float]] = {}
    if config is not None and segment_arrays is not None:
        for node_id, arrays in segment_arrays.items():
            metric_cache[node_id] = segment_metric(
                arrays,
                float(config["confidence_threshold"]),
                float(config["occ_threshold"]),
                float(config["free_threshold"]),
                str(config["sc_basis"]),
            )

    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        path_ids = [item for item in segment_path_to_root(tree, node_id) if item != ROOT_ID]
        if not path_ids:
            continue
        gain_exp = float(sum(float(tree[item].gain_exp) for item in path_ids))
        cost = float(sum(float(tree[item].cost) for item in path_ids))
        metrics = {
            "sc_gain": 0.0,
            "source_occ_count": 0.0,
            "source_free_count": 0.0,
            "source_occ_free_count": 0.0,
            "confidence_sum": 0.0,
            "hidden_region_predicted_count": 0.0,
            "frontier_local_predicted_count": 0.0,
        }
        if config is not None:
            for item in path_ids:
                for key, value in metric_cache.get(item, {}).items():
                    if key == "confidence_mean":
                        continue
                    metrics[key] = float(metrics.get(key, 0.0) + float(value))
        confidence_mean = float(metrics["confidence_sum"] / max(1.0, metrics["source_occ_free_count"]))
        selected_child_id = next((item for item in path_ids if item != ROOT_ID), None)
        selected = tree.get(str(selected_child_id)) if selected_child_id is not None else None
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
                "path_node_ids": path_ids,
                "branch_depth": int(len(path_ids)),
                "gain_exp": gain_exp,
                "sc_gain": float(metrics["sc_gain"]),
                "source_occ_count": float(metrics["source_occ_count"]),
                "source_free_count": float(metrics["source_free_count"]),
                "source_occ_free_count": float(metrics["source_occ_free_count"]),
                "confidence_sum": float(metrics["confidence_sum"]),
                "confidence_mean": confidence_mean,
                "hidden_region_predicted_count": float(metrics["hidden_region_predicted_count"]),
                "frontier_local_predicted_count": float(metrics["frontier_local_predicted_count"]),
                "cost": cost,
                "base_exp_value": float(gain_exp / max(cost, EPS)),
                "selected_child_distance_from_root_m": euclidean(root.end_world, selected.end_world)
                if selected is not None
                else None,
                "best_descendant_distance_from_root_m": euclidean(root.end_world, segment.end_world),
                "tree_total_nodes": int(len(tree)),
            }
        )

    if not rows:
        return rows
    for field_name, descending in (
        ("cost", False),
        ("gain_exp", True),
        ("sc_gain", True),
        ("hidden_region_predicted_count", True),
    ):
        ranks = rank_values(rows, field_name, descending=descending)
        rank_field = {
            "cost": "cost_rank",
            "gain_exp": "gain_exp_rank",
            "sc_gain": "sc_gain_rank",
            "hidden_region_predicted_count": "hidden_region_count_rank",
        }[field_name]
        for row in rows:
            row[rank_field] = ranks[str(row["node_id"])]
    return rows


def euclidean(a: list[float] | tuple[float, ...] | None, b: list[float] | tuple[float, ...] | None) -> float | None:
    if a is None or b is None:
        return None
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def best_by_child(scored: list[dict[str, Any]], value_field: str) -> list[dict[str, Any]]:
    by_child: dict[str, dict[str, Any]] = {}
    for row in scored:
        child = str(row.get("selected_child_id"))
        current = by_child.get(child)
        if current is None or float(row[value_field]) > float(current[value_field]):
            by_child[child] = row
    return sorted(by_child.values(), key=lambda item: float(item[value_field]), reverse=True)


def select_decision(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    config: dict[str, Any],
    measured_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("candidate rows are empty")
    sc_values = [float(row["sc_gain"]) for row in candidates]
    min_sc = min(sc_values)
    max_sc = max(sc_values)
    denom = max(max_sc - min_sc, EPS)
    max_sc_for_log = max(max_sc, EPS)

    scored: list[dict[str, Any]] = []
    for row in candidates:
        cost = float(row["cost"])
        if cost <= EPS:
            continue
        item = dict(row)
        sc_gain = float(item["sc_gain"])
        utility = str(config["utility_mode"])
        lam = config.get("lambda")
        if utility == "measured":
            sc_bonus = 0.0
            value = float(item["base_exp_value"])
        elif utility == "over_cost":
            sc_bonus = float(sc_gain / max(cost, EPS))
            value = float((float(item["gain_exp"]) + sc_gain) / max(cost, EPS))
        elif utility == "decoupled_minmax":
            normalized = float((sc_gain - min_sc) / denom)
            sc_bonus = float(float(lam or 0.0) * normalized)
            value = float(item["base_exp_value"] + sc_bonus)
        elif utility == "decoupled_log":
            normalized = float(math.log1p(sc_gain) / math.log1p(max_sc_for_log))
            sc_bonus = float(float(lam or 0.0) * normalized)
            value = float(item["base_exp_value"] + sc_bonus)
        else:
            raise ValueError(f"unsupported utility mode: {utility}")
        item["normalized_sc_gain_minmax"] = float((sc_gain - min_sc) / denom)
        item["normalized_sc_gain_log"] = float(math.log1p(sc_gain) / math.log1p(max_sc_for_log))
        item["sc_bonus"] = sc_bonus
        item["final_value"] = value
        scored.append(item)
    if not scored:
        raise RuntimeError("no scored candidates")

    base_ranked = best_by_child(scored, "base_exp_value")
    base_winner = base_ranked[0]
    ranked = best_by_child(scored, "final_value")
    winner = dict(ranked[0])
    runner_value = float(ranked[1]["final_value"]) if len(ranked) > 1 else None
    margin = None if runner_value is None else float(float(winner["final_value"]) - runner_value)
    normalized_margin = None if margin is None else float(margin / max(abs(float(winner["final_value"])), EPS))
    direction = classify_branch(winner)
    measured_direction = measured_reference.get("selected_direction") if measured_reference else None
    measured_frontier_selected = direction == "toward_measured_frontier"
    hidden_room_selected = direction == "toward_hidden_room"
    changed_vs_measured = bool(measured_direction is not None and direction != measured_direction)
    low_cost_artifact = bool(
        changed_vs_measured
        and winner.get("selected_child_id") != base_winner.get("selected_child_id")
        and float(winner.get("gain_exp", 0.0)) < float(base_winner.get("gain_exp", 0.0))
        and float(winner.get("sc_gain", 0.0)) < float(base_winner.get("sc_gain", 0.0))
        and float(winner.get("cost", 0.0)) < float(base_winner.get("cost", 0.0))
        and int(winner.get("cost_rank", 10**9)) <= max(3, int(math.ceil(0.10 * len(scored))))
    )

    return {
        "seed": int(seed),
        "prediction_source": str(config["prediction_source"]),
        "formula_name": str(config["formula_name"]),
        "config_key": str(config["config_key"]),
        "sc_basis": str(config.get("sc_basis", "none")),
        "utility_mode": str(config["utility_mode"]),
        "lambda": config.get("lambda"),
        "confidence_threshold": config.get("confidence_threshold"),
        "occ_threshold": config.get("occ_threshold"),
        "free_threshold": config.get("free_threshold"),
        "status": "completed",
        "selected_child_id": winner.get("selected_child_id"),
        "selected_child_grid": winner.get("selected_child_grid"),
        "selected_child_world": winner.get("selected_child_world"),
        "best_descendant_id": winner.get("best_descendant_id"),
        "best_descendant_grid": winner.get("best_descendant_grid"),
        "best_descendant_world": winner.get("best_descendant_world"),
        "selected_direction": direction,
        "best_descendant_direction": direction,
        "hidden_room_selected": hidden_room_selected,
        "measured_frontier_selected": measured_frontier_selected,
        "changed_vs_measured_only": changed_vs_measured,
        "agreement_with_oracle_same_formula": None,
        "gain_exp": winner.get("gain_exp"),
        "sc_gain": winner.get("sc_gain"),
        "source_occ_count": winner.get("source_occ_count"),
        "source_free_count": winner.get("source_free_count"),
        "source_occ_free_count": winner.get("source_occ_free_count"),
        "confidence_sum": winner.get("confidence_sum"),
        "confidence_mean": winner.get("confidence_mean"),
        "hidden_region_predicted_count": winner.get("hidden_region_predicted_count"),
        "frontier_local_predicted_count": winner.get("frontier_local_predicted_count"),
        "cost": winner.get("cost"),
        "final_value": winner.get("final_value"),
        "base_exp_value": winner.get("base_exp_value"),
        "sc_bonus": winner.get("sc_bonus"),
        "margin": margin,
        "normalized_margin": normalized_margin,
        "low_cost_artifact": low_cost_artifact,
        "selected_cost_rank": winner.get("cost_rank"),
        "selected_gain_exp_rank": winner.get("gain_exp_rank"),
        "selected_sc_gain_rank": winner.get("sc_gain_rank"),
        "selected_hidden_region_count_rank": winner.get("hidden_region_count_rank"),
        "path_node_ids": winner.get("path_node_ids"),
        "branch_depth": winner.get("branch_depth"),
        "root_grid": winner.get("root_grid"),
        "root_world": winner.get("root_world"),
        "selected_child_distance_from_root_m": winner.get("selected_child_distance_from_root_m"),
        "best_descendant_distance_from_root_m": winner.get("best_descendant_distance_from_root_m"),
        "tree_total_nodes": winner.get("tree_total_nodes"),
        "candidate_count": len(scored),
        "same_child_as_base_exp": winner.get("selected_child_id") == base_winner.get("selected_child_id"),
        "base_exp_selected_child_id": base_winner.get("selected_child_id"),
        "base_exp_selected_direction": classify_branch(base_winner),
        "base_exp_selected_cost": base_winner.get("cost"),
        "base_exp_selected_gain_exp": base_winner.get("gain_exp"),
        "base_exp_selected_sc_gain": base_winner.get("sc_gain"),
    }


def measured_decision_from_tree(seed: int, tree: dict[str, MiniRRTSegment]) -> dict[str, Any]:
    config = {
        "prediction_source": "none",
        "formula_name": "measured_only",
        "config_key": "none|measured_only",
        "sc_basis": "none",
        "utility_mode": "measured",
        "lambda": None,
        "confidence_threshold": None,
        "occ_threshold": None,
        "free_threshold": None,
    }
    rows = path_candidate_rows(tree, None, None)
    decision = select_decision(rows, seed=seed, config=config, measured_reference=None)
    decision["changed_vs_measured_only"] = False
    return decision


def format_float_tag(value: float | None) -> str:
    if value is None:
        return "none"
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return text


def make_config(
    prediction_source: str,
    basis: str,
    utility_mode: str,
    confidence_threshold: float,
    occ_threshold: float,
    free_threshold: float,
    lambda_value: float | None = None,
    source: str = "required",
) -> dict[str, Any]:
    if utility_mode == "over_cost":
        utility_label = "over_cost"
    elif utility_mode == "decoupled_minmax":
        utility_label = f"decoupled_minmax_lambda{format_float_tag(lambda_value)}"
    elif utility_mode == "decoupled_log":
        utility_label = f"decoupled_log_lambda{format_float_tag(lambda_value)}"
    else:
        utility_label = utility_mode
    formula_name = f"{basis}_{utility_label}"
    config_key = (
        f"{prediction_source}|{basis}|{utility_label}|"
        f"tau{format_float_tag(confidence_threshold)}|"
        f"occ{format_float_tag(occ_threshold)}|free{format_float_tag(free_threshold)}"
    )
    return {
        "prediction_source": prediction_source,
        "sc_basis": basis,
        "utility_mode": utility_mode,
        "lambda": None if lambda_value is None else float(lambda_value),
        "confidence_threshold": float(confidence_threshold),
        "occ_threshold": float(occ_threshold),
        "free_threshold": float(free_threshold),
        "formula_name": formula_name,
        "config_key": config_key,
        "sweep_source": source,
    }


def build_calibration_configs(args: argparse.Namespace, map_predict_available: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    confidence_thresholds = parse_floats(args.confidence_thresholds)
    occ_thresholds = parse_floats(args.occ_thresholds)
    free_thresholds = parse_floats(args.free_thresholds)
    lambdas = parse_floats(args.lambdas)
    sources = ["oracle"] + (["map_predict"] if map_predict_available else [])
    configs: dict[str, dict[str, Any]] = {}

    def add(config: dict[str, Any]) -> None:
        configs[str(config["config_key"])] = config

    for source in sources:
        for tau in confidence_thresholds:
            add(make_config(source, "source_occ_free", "over_cost", tau, 0.5, 0.5))
        for tau in (0.1, 0.4, 0.8):
            for lam in lambdas:
                add(make_config(source, "source_occ_free", "decoupled_minmax", tau, 0.5, 0.5, lam))
            for lam in (32.0, 48.0):
                add(make_config(source, "source_occ_free", "decoupled_log", tau, 0.5, 0.5, lam))
        for tau in (0.1, 0.4, 0.8):
            add(make_config(source, "confidence_weighted_occ_free", "over_cost", tau, 0.5, 0.5))
        for basis in ("source_occ_only", "source_free_only"):
            add(make_config(source, basis, "over_cost", 0.1, 0.5, 0.5))
        for basis in ("hidden_region_occ_free", "frontier_local_occ_free"):
            add(make_config(source, basis, "over_cost", 0.1, 0.5, 0.5))
            add(make_config(source, basis, "decoupled_minmax", 0.1, 0.5, 0.5, 32.0))

    if map_predict_available:
        for occ in occ_thresholds:
            for free in free_thresholds:
                add(make_config("map_predict", "source_occ_free", "over_cost", 0.1, occ, free, source="optional_occ_free"))
                add(
                    make_config(
                        "map_predict",
                        "source_occ_free",
                        "decoupled_minmax",
                        0.1,
                        occ,
                        free,
                        32.0,
                        source="optional_occ_free",
                    )
                )

    skipped = [
        {
            "reason": "avoided full Cartesian product",
            "details": (
                "Only required minimal sweep plus map_predict source_occ_free occ/free threshold sweep was run. "
                "Other basis/utility threshold Cartesian products were intentionally skipped."
            ),
            "not_run_examples": [
                "oracle source_occ_free full occ/free grid",
                "source_occ_only/free_only all confidence thresholds",
                "hidden_region_occ_free all confidence/occ/free thresholds",
                "frontier_local_occ_free all confidence/occ/free thresholds",
                "confidence_weighted_occ_free decoupled variants",
            ],
        }
    ]
    if not map_predict_available:
        skipped.append({"reason": "map_predict NPZ missing", "details": "map_predict source configs skipped."})
    return sorted(configs.values(), key=lambda item: str(item["config_key"])), skipped


def median(values: list[float]) -> float | None:
    finite = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return float(finite[mid])
    return float(0.5 * (finite[mid - 1] + finite[mid]))


def min_median_max(values: list[float]) -> dict[str, float | None]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"min": None, "median": None, "max": None}
    return {"min": float(min(finite)), "median": median(finite), "max": float(max(finite))}


def summarize_by_config(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["prediction_source"] == "none":
            continue
        groups[str(row["config_key"])].append(row)
    summaries: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        first = items[0]
        total = len(items)
        hidden = sum(bool(row.get("hidden_room_selected")) for row in items)
        measured = sum(bool(row.get("measured_frontier_selected")) for row in items)
        low_cost = sum(bool(row.get("low_cost_artifact")) for row in items)
        agreement_rows = [row for row in items if row.get("agreement_with_oracle_same_formula") is not None]
        agreement = (
            sum(bool(row.get("agreement_with_oracle_same_formula")) for row in agreement_rows) / len(agreement_rows)
            if agreement_rows
            else None
        )
        hidden_fraction = hidden / max(1, total)
        low_cost_fraction = low_cost / max(1, total)
        recommended_tag = "not_recommended"
        if str(first["sc_basis"]) == "hidden_region_occ_free" and hidden_fraction >= 0.8:
            recommended_tag = "diagnostic_only"
        elif hidden_fraction >= 0.8 and low_cost_fraction == 0.0:
            if str(first["utility_mode"]) == "over_cost":
                recommended_tag = "useful_but_risky"
            else:
                recommended_tag = "recommended_saved_frame_candidate"
        summaries.append(
            {
                "config_key": key,
                "prediction_source": first["prediction_source"],
                "formula_name": first["formula_name"],
                "sc_basis": first["sc_basis"],
                "utility_mode": first["utility_mode"],
                "lambda": first.get("lambda"),
                "confidence_threshold": first.get("confidence_threshold"),
                "occ_threshold": first.get("occ_threshold"),
                "free_threshold": first.get("free_threshold"),
                "seed_count": total,
                "hidden_room_selection_fraction": hidden_fraction,
                "measured_frontier_selection_fraction": measured / max(1, total),
                "oracle_map_predict_agreement_fraction": agreement,
                "low_cost_artifact_fraction": low_cost_fraction,
                "margin": min_median_max([float(row.get("margin") or 0.0) for row in items]),
                "mean_selected_cost": mean([row.get("cost") for row in items]),
                "mean_selected_sc_gain": mean([row.get("sc_gain") for row in items]),
                "mean_hidden_region_count": mean([row.get("hidden_region_predicted_count") for row in items]),
                "mean_selected_cost_rank": mean([row.get("selected_cost_rank") for row in items]),
                "mean_selected_sc_gain_rank": mean([row.get("selected_sc_gain_rank") for row in items]),
                "robust_hidden_score": float(hidden_fraction - low_cost_fraction),
                "recommended_tag": recommended_tag,
            }
        )
    return summaries


def mean(values: list[Any]) -> float | None:
    finite: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def fill_oracle_agreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    oracle = {
        (
            int(row["seed"]),
            str(row["formula_name"]),
            str(row.get("confidence_threshold")),
            str(row.get("occ_threshold")),
            str(row.get("free_threshold")),
            str(row.get("lambda")),
        ): row
        for row in rows
        if row.get("prediction_source") == "oracle"
    }
    for row in rows:
        if row.get("prediction_source") == "oracle":
            row["agreement_with_oracle_same_formula"] = True
        elif row.get("prediction_source") == "map_predict":
            key = (
                int(row["seed"]),
                str(row["formula_name"]),
                str(row.get("confidence_threshold")),
                str(row.get("occ_threshold")),
                str(row.get("free_threshold")),
                str(row.get("lambda")),
            )
            ref = oracle.get(key)
            row["agreement_with_oracle_same_formula"] = (
                None if ref is None else row.get("selected_direction") == ref.get("selected_direction")
            )
    return rows


def oracle_map_predict_agreement(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("prediction_source") == "map_predict" and row.get("agreement_with_oracle_same_formula") is not None:
            key = (
                row["formula_name"],
                row["sc_basis"],
                row["utility_mode"],
                row.get("lambda"),
                row.get("confidence_threshold"),
                row.get("occ_threshold"),
                row.get("free_threshold"),
            )
            groups[key].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        out.append(
            {
                "formula_name": key[0],
                "sc_basis": key[1],
                "utility_mode": key[2],
                "lambda": key[3],
                "confidence_threshold": key[4],
                "occ_threshold": key[5],
                "free_threshold": key[6],
                "seed_count": len(items),
                "agreement_fraction": sum(bool(row["agreement_with_oracle_same_formula"]) for row in items)
                / max(1, len(items)),
                "map_hidden_room_fraction": sum(bool(row["hidden_room_selected"]) for row in items) / max(1, len(items)),
            }
        )
    return out


def threshold_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("prediction_source") == "none":
            continue
        if row.get("sc_basis") != "source_occ_free":
            continue
        if row.get("utility_mode") not in {"over_cost", "decoupled_minmax"}:
            continue
        key = (
            row["prediction_source"],
            row["utility_mode"],
            row.get("lambda"),
            row.get("confidence_threshold"),
            row.get("occ_threshold"),
            row.get("free_threshold"),
        )
        groups[key].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        out.append(
            {
                "prediction_source": key[0],
                "utility_mode": key[1],
                "lambda": key[2],
                "confidence_threshold": key[3],
                "occ_threshold": key[4],
                "free_threshold": key[5],
                "seed_count": len(items),
                "hidden_room_selection_fraction": sum(bool(row["hidden_room_selected"]) for row in items)
                / max(1, len(items)),
                "mean_sc_gain": mean([row.get("sc_gain") for row in items]),
                "mean_hidden_region_count": mean([row.get("hidden_region_predicted_count") for row in items]),
                "low_cost_artifact_fraction": sum(bool(row["low_cost_artifact"]) for row in items) / max(1, len(items)),
            }
        )
    return out


def lambda_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("prediction_source") == "none":
            continue
        if row.get("utility_mode") not in {"decoupled_minmax", "decoupled_log"}:
            continue
        key = (
            row["prediction_source"],
            row["sc_basis"],
            row["utility_mode"],
            row.get("lambda"),
            row.get("confidence_threshold"),
            row.get("occ_threshold"),
            row.get("free_threshold"),
        )
        groups[key].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        out.append(
            {
                "prediction_source": key[0],
                "sc_basis": key[1],
                "utility_mode": key[2],
                "lambda": key[3],
                "confidence_threshold": key[4],
                "occ_threshold": key[5],
                "free_threshold": key[6],
                "seed_count": len(items),
                "hidden_room_selection_fraction": sum(bool(row["hidden_room_selected"]) for row in items)
                / max(1, len(items)),
                "mean_sc_gain": mean([row.get("sc_gain") for row in items]),
                "mean_margin": mean([row.get("margin") for row in items]),
                "low_cost_artifact_fraction": sum(bool(row["low_cost_artifact"]) for row in items) / max(1, len(items)),
            }
        )
    return out


def hidden_region_signal(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("prediction_source") == "none":
            continue
        key = (
            row["prediction_source"],
            row["sc_basis"],
            row.get("confidence_threshold"),
            row.get("occ_threshold"),
            row.get("free_threshold"),
        )
        groups[key].append(row)
    out = []
    for key, items in sorted(groups.items(), key=lambda item: str(item[0])):
        out.append(
            {
                "prediction_source": key[0],
                "sc_basis": key[1],
                "confidence_threshold": key[2],
                "occ_threshold": key[3],
                "free_threshold": key[4],
                "rows": len(items),
                "mean_selected_hidden_region_predicted_count": mean(
                    [row.get("hidden_region_predicted_count") for row in items]
                ),
                "mean_selected_frontier_local_predicted_count": mean(
                    [row.get("frontier_local_predicted_count") for row in items]
                ),
                "hidden_room_selection_fraction": sum(bool(row["hidden_room_selected"]) for row in items)
                / max(1, len(items)),
            }
        )
    return out


def best_candidates(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {row["config_key"]: row for row in summary_rows}
    candidates: list[dict[str, Any]] = []
    for row in summary_rows:
        if row["prediction_source"] != "map_predict":
            continue
        if row["sc_basis"] == "hidden_region_occ_free":
            continue
        if float(row["hidden_room_selection_fraction"]) < 0.8:
            continue
        if float(row["low_cost_artifact_fraction"]) != 0.0:
            continue
        oracle_key = str(row["config_key"]).replace("map_predict|", "oracle|", 1)
        oracle = by_key.get(oracle_key)
        if oracle is None or float(oracle["hidden_room_selection_fraction"]) < 0.8:
            continue
        agreement = row.get("oracle_map_predict_agreement_fraction")
        if agreement is None or float(agreement) < 0.8:
            continue
        item = dict(row)
        item["oracle_hidden_room_selection_fraction"] = oracle["hidden_room_selection_fraction"]
        item["runtime_readiness"] = "saved-frame-only-ready"
        if row["utility_mode"] == "over_cost":
            item["runtime_readiness"] = "useful-but-risky-offline"
        candidates.append(item)

    def sort_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        utility_bonus = 1.0 if row["utility_mode"] != "over_cost" else 0.0
        risky_penalty = -1.0 if row["recommended_tag"] == "useful_but_risky" else 0.0
        return (
            utility_bonus,
            float(row["hidden_room_selection_fraction"]),
            float(row.get("oracle_map_predict_agreement_fraction") or 0.0),
            float(row["robust_hidden_score"]),
            risky_penalty,
        )

    candidates = sorted(candidates, key=sort_key, reverse=True)
    recommended = candidates[0] if candidates else None
    return {"recommended": recommended, "candidates": candidates[:20], "candidate_count": len(candidates)}


def write_formula_definitions(output_dir: Path) -> None:
    definitions = {
        "source_occ_free": "count predicted OCC plus predicted FREE visible from each path segment",
        "source_occ_only": "count predicted OCC only",
        "source_free_only": "count predicted FREE only",
        "confidence_weighted_occ_free": "sum confidence over predicted OCC/FREE voxels",
        "hidden_region_occ_free": "diagnostic-only count of predicted OCC/FREE inside synthetic hidden-region mask",
        "frontier_local_occ_free": "source-inspired diagnostic count of predicted OCC/FREE near doorway/frontier-local mask",
        "over_cost": "(gain_exp + sc_gain) / cost",
        "decoupled_minmax": "gain_exp / cost + lambda * minmax(sc_gain)",
        "decoupled_log": "gain_exp / cost + lambda * log1p(sc_gain)/log1p(max_sc_gain)",
    }
    save_json(output_dir / "calibration_formula_definitions.json", definitions)
    lines = ["# Calibration Formula Definitions", ""]
    for key, value in definitions.items():
        lines.append(f"- `{key}`: {value}.")
    write_text(output_dir / "calibration_formula_definitions.md", "\n".join(lines))


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    best = summary.get("best_candidate_config") or {}
    lines = [
        "# Stage 4A-6.5ab Synthetic Calibration Summary",
        "",
        f"1. Loaded 6.5aa outputs: `{answers['loaded_stage4a65aa_outputs']}`.",
        f"2. No Isaac / new capture / map_predict rerun: `{answers['no_isaac_no_capture_no_map_predict_rerun']}`.",
        f"3. Tested seeds/configs/decision rows: `{answers['seed_count']}` / `{answers['config_count']}` / `{answers['decision_row_count']}`.",
        f"4. measured-only direction: `{answers['measured_only_direction_counts']}`.",
        f"5. Oracle stable hidden-room configs: `{answers['oracle_stable_hidden_room_configs']}`.",
        f"6. map_predict stable hidden-room configs: `{answers['map_predict_stable_hidden_room_configs']}`.",
        f"7. Highest map/oracle agreement config: `{answers['highest_agreement_config']}`.",
        f"8. over_cost behavior: {answers['over_cost_behavior']}",
        f"9. decoupled lambda32/lambda48 stability: {answers['decoupled_lambda_32_48_stability']}",
        f"10. confidence-threshold effect: {answers['confidence_threshold_effect']}",
        f"11. occ/free-threshold effect: {answers['occ_free_threshold_effect']}",
        f"12. low-cost artifacts: `{answers['low_cost_artifact_count']}` / `{answers['decision_row_count']}`.",
        f"13. recommended candidate config: `{best.get('config_key')}`.",
        f"14. readiness: `{answers['best_config_readiness']}`.",
        f"15. saved-frame one-step formula smoke next? `{answers['saved_frame_one_step_formula_smoke_ready']}`.",
        f"16. rollout readiness: `{answers['rollout_readiness']}`.",
        "",
        "This is an offline calibration result only; it does not claim coverage improvement.",
    ]
    write_text(path, "\n".join(lines))


def write_simple_md_table(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str], limit: int = 40) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("- no rows")
        write_text(path, "\n".join(lines))
        return
    header = "| " + " | ".join(fields) + " |"
    sep = "| " + " | ".join(["---"] * len(fields)) + " |"
    lines.extend([header, sep])
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(to_jsonable(row.get(field, ""))) for field in fields) + " |")
    if len(rows) > limit:
        lines.append(f"\nShowing `{limit}` of `{len(rows)}` rows.")
    write_text(path, "\n".join(lines))


def write_decisions_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Per-Seed Calibration Decisions", ""]
    for row in rows:
        if row["prediction_source"] == "none":
            lines.append(
                f"- seed `{row['seed']}` measured-only: `{row['selected_direction']}`, best `{row['best_descendant_id']}`."
            )
        else:
            lines.append(
                f"- seed `{row['seed']}` `{row['prediction_source']}` `{row['formula_name']}` "
                f"tau `{row['confidence_threshold']}` occ/free `{row['occ_threshold']}/{row['free_threshold']}`: "
                f"`{row['selected_direction']}`, value `{row['final_value']}`, low-cost `{row['low_cost_artifact']}`."
            )
    write_text(path, "\n".join(lines))


def topdown_observed(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == 0, axis=2)
    image = np.zeros(observed_state.shape[:2], dtype=np.float32)
    image[free] = 0.45
    image[occupied] = 0.95
    return image


def plot_hidden_room_selection(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    rows = sorted(
        summary_rows,
        key=lambda row: (float(row["hidden_room_selection_fraction"]), float(row["robust_hidden_score"])),
        reverse=True,
    )[:28]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    labels = [f"{r['prediction_source']} {r['formula_name']}\nt{r['confidence_threshold']}" for r in rows]
    values = [float(r["hidden_room_selection_fraction"]) for r in rows]
    ax.bar(range(len(rows)), values, color="#2563eb")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hidden-room selection fraction")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_title("Hidden-room selection by config")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_agreement(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = sorted(rows, key=lambda row: float(row["agreement_fraction"]), reverse=True)[:32]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    labels = [f"{r['formula_name']}\nt{r['confidence_threshold']}" for r in rows]
    values = [float(r["agreement_fraction"]) for r in rows]
    ax.bar(range(len(rows)), values, color="#059669")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("map/oracle direction agreement")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_title("Oracle vs map_predict agreement")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_tau_sensitivity(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for source, color in (("oracle", "#2563eb"), ("map_predict", "#059669")):
        points = [
            row
            for row in rows
            if row["prediction_source"] == source
            and row["utility_mode"] == "over_cost"
            and row["lambda"] is None
            and float(row["occ_threshold"]) == 0.5
            and float(row["free_threshold"]) == 0.5
        ]
        points = sorted(points, key=lambda row: float(row["confidence_threshold"]))
        if points:
            ax.plot(
                [float(row["confidence_threshold"]) for row in points],
                [float(row["hidden_room_selection_fraction"]) for row in points],
                marker="o",
                label=source,
                color=color,
            )
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("hidden-room fraction")
    ax.set_title("Tau sensitivity")
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_lambda_sensitivity(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for source, color in (("oracle", "#2563eb"), ("map_predict", "#059669")):
        points = [
            row
            for row in rows
            if row["prediction_source"] == source
            and row["sc_basis"] == "source_occ_free"
            and row["utility_mode"] == "decoupled_minmax"
            and float(row["confidence_threshold"]) == 0.1
            and float(row["occ_threshold"]) == 0.5
            and float(row["free_threshold"]) == 0.5
        ]
        points = sorted(points, key=lambda row: float(row["lambda"]))
        if points:
            ax.plot(
                [float(row["lambda"]) for row in points],
                [float(row["hidden_room_selection_fraction"]) for row in points],
                marker="o",
                label=source,
                color=color,
            )
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("lambda")
    ax.set_ylabel("hidden-room fraction")
    ax.set_title("Decoupled minmax lambda sensitivity at tau=0.1")
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_low_cost(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    rows = sorted(summary_rows, key=lambda row: float(row["low_cost_artifact_fraction"]), reverse=True)[:28]
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    labels = [f"{r['prediction_source']} {r['formula_name']}\nt{r['confidence_threshold']}" for r in rows]
    values = [float(r["low_cost_artifact_fraction"]) for r in rows]
    ax.bar(range(len(rows)), values, color="#dc2626")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("low-cost artifact fraction")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_title("Low-cost artifact fraction")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_selected_topdown(path: Path, observed_state: np.ndarray, decision_rows: list[dict[str, Any]], config_key: str) -> None:
    rows = [row for row in decision_rows if row.get("config_key") == config_key]
    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap="Greys", alpha=0.5, interpolation="nearest")
    colors = {"oracle": "#2563eb", "map_predict": "#059669", "none": "#f97316"}
    for row in rows:
        root = row.get("root_grid")
        best = row.get("best_descendant_grid")
        selected = row.get("selected_child_grid")
        color = colors.get(str(row.get("prediction_source")), "#111827")
        if root and best:
            ax.plot([root[0], best[0]], [root[1], best[1]], color=color, linewidth=1.4, alpha=0.6)
        if selected:
            ax.scatter([selected[0]], [selected[1]], color=color, s=30)
        if best:
            ax.scatter([best[0]], [best[1]], color=color, marker="*", s=65)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title("Selected branches for best config")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_hidden_signal(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for source, color in (("oracle", "#2563eb"), ("map_predict", "#059669")):
        points = [
            row
            for row in rows
            if row["prediction_source"] == source
            and row["sc_basis"] == "source_occ_free"
            and float(row["occ_threshold"]) == 0.5
            and float(row["free_threshold"]) == 0.5
        ]
        by_tau: dict[float, list[float]] = defaultdict(list)
        for row in points:
            by_tau[float(row["confidence_threshold"])].append(
                float(row.get("mean_selected_hidden_region_predicted_count") or 0.0)
            )
        xs = sorted(by_tau)
        if xs:
            ax.plot(xs, [sum(by_tau[x]) / max(1, len(by_tau[x])) for x in xs], marker="o", color=color, label=source)
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("mean selected hidden-region predicted count")
    ax.set_title("Hidden-region signal by threshold")
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_value_components(path: Path, decision_rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    if best is None:
        rows = [row for row in decision_rows if row.get("prediction_source") != "none"][:10]
    else:
        rows = [row for row in decision_rows if row.get("config_key") == best.get("config_key")]
    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    labels = [f"{row['prediction_source']} s{row['seed']}" for row in rows]
    gain_exp = [float(row.get("gain_exp") or 0.0) for row in rows]
    sc_gain = [float(row.get("sc_gain") or 0.0) for row in rows]
    costs = [float(row.get("cost") or 0.0) for row in rows]
    x = np.arange(len(rows))
    ax.bar(x, gain_exp, label="gain_exp", color="#f97316")
    ax.bar(x, sc_gain, bottom=gain_exp, label="sc_gain", color="#2563eb")
    ax.plot(x, costs, color="#111827", marker="o", label="cost")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Value components for best config rows")
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_prediction_overlay(path: Path, observed_state: np.ndarray, oracle_npz: Path, map_npz: Path | None) -> None:
    sources: list[tuple[str, Path]] = [("oracle", oracle_npz)]
    if map_npz is not None and map_npz.is_file():
        sources.append(("map_predict", map_npz))
    fig, axes = plt.subplots(1, len(sources), figsize=(6 * len(sources), 5.2), constrained_layout=True)
    if len(sources) == 1:
        axes = [axes]  # type: ignore[assignment]
    base = topdown_observed(observed_state)
    for ax, (name, path) in zip(axes, sources):
        pred = SimPredictionLayer.from_npz(path)
        occ = np.any((pred.valid & (pred.occupied_prob >= 0.5)), axis=2)
        free = np.any((pred.valid & (pred.free_prob >= 0.5)), axis=2)
        ax.imshow(base.T, origin="lower", cmap="Greys", alpha=0.45, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~free.T, free.T), origin="lower", cmap="Greens", alpha=0.28, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~occ.T, occ.T), origin="lower", cmap="Reds", alpha=0.38, interpolation="nearest")
        ax.set_title(f"{name} prediction overlay")
        ax.set_xlabel("grid x")
        ax.set_ylabel("grid y")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_plots(
    output_dir: Path,
    observed_state: np.ndarray,
    summary_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    lambda_rows: list[dict[str, Any]],
    hidden_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
    oracle_npz: Path,
    map_npz: Path | None,
    missing: dict[str, Any],
) -> None:
    plotters = [
        ("hidden_room_selection_by_config.png", lambda p: plot_hidden_room_selection(p, summary_rows)),
        ("oracle_vs_map_predict_agreement.png", lambda p: plot_agreement(p, agreement_rows)),
        ("tau_sensitivity_hidden_room_fraction.png", lambda p: plot_tau_sensitivity(p, threshold_rows)),
        ("lambda_sensitivity_hidden_room_fraction.png", lambda p: plot_lambda_sensitivity(p, lambda_rows)),
        ("low_cost_artifact_fraction.png", lambda p: plot_low_cost(p, summary_rows)),
        (
            "selected_branch_topdown_by_best_config.png",
            lambda p: plot_selected_topdown(p, observed_state, decision_rows, str((best or {}).get("config_key", ""))),
        ),
        ("hidden_region_signal_by_threshold.png", lambda p: plot_hidden_signal(p, hidden_rows)),
        ("value_component_stack_best_configs.png", lambda p: plot_value_components(p, decision_rows, best)),
        (
            "map_predict_vs_oracle_prediction_overlay_topdown.png",
            lambda p: plot_prediction_overlay(p, observed_state, oracle_npz, map_npz),
        ),
    ]
    for name, func in plotters:
        path = output_dir / name
        try:
            func(path)
        except Exception as exc:  # pragma: no cover - recorded for operational smoke robustness
            reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
            write_text(reason, f"# Plot Skipped\n\n- plot: `{name}`\n- reason: `{type(exc).__name__}: {exc}`")
            missing.setdefault("plot_failures", []).append({"plot": name, "reason": f"{type(exc).__name__}: {exc}"})


def summarize_prediction_npz(path: Path, observed_state: np.ndarray, hidden_mask: np.ndarray) -> dict[str, Any]:
    pred = SimPredictionLayer.from_npz(path)
    valid = pred.valid
    occ = valid & (pred.occupied_prob >= 0.5)
    free = valid & (pred.free_prob >= 0.5)
    unmeasured = observed_state == UNKNOWN
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(pred.shape()),
        "valid_count": int(np.count_nonzero(valid)),
        "predicted_occupied_count": int(np.count_nonzero(occ & unmeasured)),
        "predicted_free_count": int(np.count_nonzero(free & unmeasured)),
        "hidden_valid_count": int(np.count_nonzero(valid & hidden_mask & unmeasured)),
        "hidden_predicted_occupied_count": int(np.count_nonzero(occ & hidden_mask & unmeasured)),
        "hidden_predicted_free_count": int(np.count_nonzero(free & hidden_mask & unmeasured)),
    }


def update_summary_agreement(summary_rows: list[dict[str, Any]], agreement_rows: list[dict[str, Any]]) -> None:
    by_key = {
        (
            row["formula_name"],
            row["sc_basis"],
            row["utility_mode"],
            str(row.get("lambda")),
            str(row.get("confidence_threshold")),
            str(row.get("occ_threshold")),
            str(row.get("free_threshold")),
        ): row
        for row in agreement_rows
    }
    for row in summary_rows:
        key = (
            row["formula_name"],
            row["sc_basis"],
            row["utility_mode"],
            str(row.get("lambda")),
            str(row.get("confidence_threshold")),
            str(row.get("occ_threshold")),
            str(row.get("free_threshold")),
        )
        if row["prediction_source"] == "map_predict" and key in by_key:
            row["oracle_map_predict_agreement_fraction"] = by_key[key]["agreement_fraction"]
        if row["prediction_source"] == "oracle" and key in by_key:
            row["oracle_map_predict_agreement_fraction"] = by_key[key]["agreement_fraction"]


def make_answers(
    *,
    seeds: list[int],
    configs: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    agreement_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    lambda_rows: list[dict[str, Any]],
    best: dict[str, Any] | None,
) -> dict[str, Any]:
    measured = [row for row in decisions if row["prediction_source"] == "none"]
    measured_counts = dict(Counter(str(row["selected_direction"]) for row in measured))
    oracle_stable = [
        row["config_key"]
        for row in summary_rows
        if row["prediction_source"] == "oracle" and float(row["hidden_room_selection_fraction"]) >= 0.8
    ][:12]
    map_stable = [
        row["config_key"]
        for row in summary_rows
        if row["prediction_source"] == "map_predict" and float(row["hidden_room_selection_fraction"]) >= 0.8
    ][:12]
    highest_agreement = max(agreement_rows, key=lambda row: float(row["agreement_fraction"]), default=None)
    over_cost_rows = [
        row for row in summary_rows if row["utility_mode"] == "over_cost" and row["sc_basis"] == "source_occ_free"
    ]
    over_cost_hidden = {
        row["config_key"]: row["hidden_room_selection_fraction"]
        for row in over_cost_rows
        if float(row["hidden_room_selection_fraction"]) >= 0.8
    }
    decoupled_rows = [
        row
        for row in lambda_rows
        if row["sc_basis"] == "source_occ_free"
        and row["utility_mode"] == "decoupled_minmax"
        and float(row["occ_threshold"]) == 0.5
        and float(row["free_threshold"]) == 0.5
        and float(row["lambda"]) in {32.0, 48.0}
    ]
    tau_points = [
        row
        for row in threshold_rows
        if row["utility_mode"] == "over_cost"
        and row["lambda"] is None
        and float(row["occ_threshold"]) == 0.5
        and float(row["free_threshold"]) == 0.5
    ]
    map_occ_free = [
        row
        for row in threshold_rows
        if row["prediction_source"] == "map_predict"
        and row["utility_mode"] in {"over_cost", "decoupled_minmax"}
        and float(row["confidence_threshold"]) == 0.1
    ]
    low_cost_count = sum(bool(row.get("low_cost_artifact")) for row in decisions)
    saved_frame_ready = bool(best and best.get("utility_mode") != "over_cost" and best.get("sc_basis") != "hidden_region_occ_free")
    if best and best.get("utility_mode") == "over_cost":
        best_readiness = "useful-but-risky-offline"
    elif saved_frame_ready:
        best_readiness = "saved-frame-only-ready"
    elif best:
        best_readiness = "diagnostic-only"
    else:
        best_readiness = "not-ready"

    return {
        "loaded_stage4a65aa_outputs": True,
        "no_isaac_no_capture_no_map_predict_rerun": True,
        "seed_count": len(seeds),
        "config_count": len(configs),
        "decision_row_count": len(decisions),
        "measured_only_direction_counts": measured_counts,
        "oracle_stable_hidden_room_configs": oracle_stable,
        "map_predict_stable_hidden_room_configs": map_stable,
        "highest_agreement_config": highest_agreement,
        "over_cost_behavior": f"stable configs: {len(over_cost_hidden)}; useful but risky because SC remains inside the cost denominator",
        "decoupled_lambda_32_48_stability": decoupled_rows,
        "confidence_threshold_effect": tau_points,
        "occ_free_threshold_effect": map_occ_free,
        "low_cost_artifact_count": int(low_cost_count),
        "best_config_readiness": best_readiness,
        "saved_frame_one_step_formula_smoke_ready": saved_frame_ready,
        "rollout_readiness": False,
    }


def write_recommendation(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    best = summary.get("best_candidate_config")
    if answers["saved_frame_one_step_formula_smoke_ready"]:
        next_step = "saved-frame one-step formula smoke only"
        why = "map_predict has a non-over_cost robust candidate with Oracle agreement and zero low-cost artifacts in this offline calibration."
    elif best and best.get("utility_mode") == "over_cost":
        next_step = "offline decoupled lambda/normalization refinement"
        why = "over_cost works, but it keeps SC inside the cost denominator and should remain offline/risky."
    elif summary.get("oracle_has_robust_config") and not summary.get("map_predict_has_robust_config"):
        next_step = "synthetic map_predict calibration/domain/preprocess diagnosis"
        why = "Oracle is stable but map_predict is not stable enough under non-diagnostic formulas."
    else:
        next_step = "debug logging/calibration rerun only"
        why = "no robust non-diagnostic candidate config was found."
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {next_step}",
        f"- why: {why}",
        "- runtime smoke readiness: no",
        "- rollout readiness: no",
        "- do not run RL/PPO/BC/IL.",
    ]
    write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    stage_dir = Path(args.stage4a65aa_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)

    observed_path = stage_dir / "observed_state_synthetic_frame000.npy"
    oracle_npz = stage_dir / "oracle_global_prediction_layer.npz"
    map_npz = stage_dir / "map_predict" / "global_prediction_layer.npz"
    pose_json = stage_dir / "pose_000.json"
    camera_info = stage_dir / "camera_info.json"
    scene_metadata_path = stage_dir / "scene_metadata.json"
    validation_summary_path = stage_dir / "synthetic_sc_validation_summary.json"
    per_seed_65aa_path = stage_dir / "per_seed_mode_decisions.csv"
    branch_65aa_path = stage_dir / "branch_direction_classification.csv"

    required = [observed_path, oracle_npz, pose_json, camera_info, scene_metadata_path, validation_summary_path]
    missing_required = [str(path) for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(f"missing required Stage 4A-6.5aa inputs: {missing_required}")

    observed_hash_before = sha256_file(observed_path)
    oracle_hash_before = sha256_file(oracle_npz)
    map_available = map_npz.is_file()
    map_hash_before = sha256_file(map_npz) if map_available else None

    observed_state = np.load(observed_path)
    scene_metadata = read_json(scene_metadata_path)
    bounds = normalize_bounds(scene_metadata["map_bounds"])
    hidden_mask = region_mask(
        tuple(int(v) for v in observed_state.shape),
        bounds,
        float(args.voxel_size),
        scene_metadata["diagnostic_regions"]["oracle_hidden_room"],
    )
    frontier_local_mask = make_frontier_local_mask(
        tuple(int(v) for v in observed_state.shape),
        bounds,
        float(args.voxel_size),
        scene_metadata,
    )

    loaded_manifest = {
        "stage4a65aa_dir": str(stage_dir),
        "observed_state": {"path": str(observed_path), "sha256": observed_hash_before, "shape": list(observed_state.shape)},
        "oracle_prediction": summarize_prediction_npz(oracle_npz, observed_state, hidden_mask),
        "map_predict_prediction": summarize_prediction_npz(map_npz, observed_state, hidden_mask) if map_available else None,
        "pose_json": {"path": str(pose_json), "sha256": sha256_file(pose_json)},
        "camera_info": {"path": str(camera_info), "sha256": sha256_file(camera_info)},
        "scene_metadata": {"path": str(scene_metadata_path), "sha256": sha256_file(scene_metadata_path)},
        "synthetic_sc_validation_summary": {
            "path": str(validation_summary_path),
            "sha256": sha256_file(validation_summary_path),
        },
        "per_seed_mode_decisions": {"path": str(per_seed_65aa_path), "exists": per_seed_65aa_path.is_file()},
        "branch_direction_classification": {"path": str(branch_65aa_path), "exists": branch_65aa_path.is_file()},
        "map_predict_available": map_available,
    }
    save_json(output_dir / "loaded_stage4a65aa_manifest.json", loaded_manifest)
    write_text(
        output_dir / "loaded_stage4a65aa_manifest.md",
        "\n".join(
            [
                "# Loaded Stage 4A-6.5aa Manifest",
                "",
                f"- Stage 4A-6.5aa dir: `{stage_dir}`",
                f"- observed_state: `{observed_path}`",
                f"- observed_state sha256: `{observed_hash_before}`",
                f"- oracle prediction: `{oracle_npz}`",
                f"- map_predict prediction: `{map_npz if map_available else 'missing'}`",
                f"- scene metadata: `{scene_metadata_path}`",
                "- Isaac startup: `false`",
                "- map_predict rerun: `false`",
            ]
        ),
    )

    if not map_available:
        write_text(
            output_dir / "map_predict_missing_skipped_reason.md",
            "# map_predict Missing\n\n- The 6.5aa map_predict NPZ was missing, so map_predict calibration rows were skipped. Oracle calibration continued.",
        )

    write_formula_definitions(output_dir)
    configs, skipped = build_calibration_configs(args, map_available)
    save_json(output_dir / "skipped_combinations.json", skipped)

    prediction_layers: dict[str, SimPredictionLayer] = {"oracle": SimPredictionLayer.from_npz(oracle_npz)}
    if map_available:
        prediction_layers["map_predict"] = SimPredictionLayer.from_npz(map_npz)

    decision_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    measured_refs: dict[int, dict[str, Any]] = {}
    segment_array_cache: dict[tuple[int, str], dict[str, dict[str, np.ndarray]]] = {}

    for seed in seeds:
        measured_tree_dir = stage_dir / "raw_trees" / f"seed_{seed:03d}" / "measured_only"
        measured_tree = load_tree(measured_tree_dir)
        measured_decision = measured_decision_from_tree(seed, measured_tree)
        measured_refs[seed] = measured_decision
        decision_rows.append(measured_decision)
        manifest_rows.append(
            {
                "seed": seed,
                "prediction_source": "none",
                "formula_name": "measured_only",
                "status": "completed",
                "tree_dir": str(measured_tree_dir),
                "decision_rows": 1,
            }
        )

        trees_by_source: dict[str, dict[str, MiniRRTSegment]] = {}
        for source in prediction_layers:
            tree_dir = stage_dir / "raw_trees" / f"seed_{seed:03d}" / f"{source}_raw_count"
            trees_by_source[source] = load_tree(tree_dir)
            segment_array_cache[(seed, source)] = precompute_segment_prediction_arrays(
                trees_by_source[source],
                observed_state,
                prediction_layers[source],
                hidden_mask,
                frontier_local_mask,
                args,
            )

        for config in configs:
            source = str(config["prediction_source"])
            if source not in trees_by_source:
                continue
            candidates = path_candidate_rows(trees_by_source[source], segment_array_cache[(seed, source)], config)
            decision = select_decision(candidates, seed=seed, config=config, measured_reference=measured_refs.get(seed))
            decision_rows.append(decision)

    decision_rows = fill_oracle_agreement(decision_rows)
    summary_rows = summarize_by_config(decision_rows)
    agreement_rows = oracle_map_predict_agreement(decision_rows)
    update_summary_agreement(summary_rows, agreement_rows)
    # Recompute tags that depend on agreement after agreement fields are filled.
    summary_rows = summarize_by_config(decision_rows)
    update_summary_agreement(summary_rows, agreement_rows)
    threshold_rows = threshold_sensitivity(decision_rows)
    lambda_rows = lambda_sensitivity(decision_rows)
    hidden_rows = hidden_region_signal(decision_rows)
    best_info = best_candidates(summary_rows)
    best = best_info.get("recommended")

    for config in configs:
        count = sum(1 for row in decision_rows if row.get("config_key") == config["config_key"])
        manifest_rows.append({**config, "status": "completed", "decision_rows": count})
    write_jsonl(output_dir / "calibration_sweep_manifest.jsonl", manifest_rows)

    decision_fields = [
        "seed",
        "prediction_source",
        "formula_name",
        "config_key",
        "sc_basis",
        "utility_mode",
        "lambda",
        "confidence_threshold",
        "occ_threshold",
        "free_threshold",
        "selected_child_id",
        "selected_child_grid",
        "selected_child_world",
        "best_descendant_id",
        "best_descendant_grid",
        "best_descendant_world",
        "selected_direction",
        "best_descendant_direction",
        "hidden_room_selected",
        "measured_frontier_selected",
        "changed_vs_measured_only",
        "agreement_with_oracle_same_formula",
        "gain_exp",
        "sc_gain",
        "source_occ_count",
        "source_free_count",
        "confidence_sum",
        "confidence_mean",
        "hidden_region_predicted_count",
        "frontier_local_predicted_count",
        "cost",
        "final_value",
        "base_exp_value",
        "sc_bonus",
        "margin",
        "normalized_margin",
        "low_cost_artifact",
        "selected_cost_rank",
        "selected_gain_exp_rank",
        "selected_sc_gain_rank",
        "selected_hidden_region_count_rank",
    ]
    save_json(output_dir / "per_seed_calibration_decisions.json", decision_rows)
    write_csv(output_dir / "per_seed_calibration_decisions.csv", decision_rows, decision_fields)
    write_decisions_md(output_dir / "per_seed_calibration_decisions.md", decision_rows)

    save_json(output_dir / "calibration_summary_by_config.json", summary_rows)
    write_csv(output_dir / "calibration_summary_by_config.csv", summary_rows)
    write_simple_md_table(
        output_dir / "calibration_summary_by_config.md",
        "Calibration Summary By Config",
        summary_rows,
        [
            "prediction_source",
            "formula_name",
            "confidence_threshold",
            "occ_threshold",
            "free_threshold",
            "hidden_room_selection_fraction",
            "oracle_map_predict_agreement_fraction",
            "low_cost_artifact_fraction",
            "recommended_tag",
        ],
    )

    save_json(output_dir / "threshold_sensitivity_summary.json", threshold_rows)
    write_csv(output_dir / "threshold_sensitivity_summary.csv", threshold_rows)
    write_simple_md_table(
        output_dir / "threshold_sensitivity_summary.md",
        "Threshold Sensitivity Summary",
        threshold_rows,
        [
            "prediction_source",
            "utility_mode",
            "lambda",
            "confidence_threshold",
            "occ_threshold",
            "free_threshold",
            "hidden_room_selection_fraction",
            "mean_sc_gain",
            "low_cost_artifact_fraction",
        ],
    )

    save_json(output_dir / "lambda_sensitivity_summary.json", lambda_rows)
    write_csv(output_dir / "lambda_sensitivity_summary.csv", lambda_rows)
    write_simple_md_table(
        output_dir / "lambda_sensitivity_summary.md",
        "Lambda Sensitivity Summary",
        lambda_rows,
        [
            "prediction_source",
            "sc_basis",
            "utility_mode",
            "lambda",
            "confidence_threshold",
            "hidden_room_selection_fraction",
            "mean_sc_gain",
            "mean_margin",
        ],
    )

    save_json(output_dir / "oracle_map_predict_agreement.json", agreement_rows)
    write_csv(output_dir / "oracle_map_predict_agreement.csv", agreement_rows)
    write_simple_md_table(
        output_dir / "oracle_map_predict_agreement.md",
        "Oracle Map Predict Agreement",
        agreement_rows,
        [
            "formula_name",
            "confidence_threshold",
            "occ_threshold",
            "free_threshold",
            "lambda",
            "agreement_fraction",
            "map_hidden_room_fraction",
        ],
    )

    low_cost_rows = [row for row in decision_rows if row["prediction_source"] != "none"]
    save_json(output_dir / "low_cost_artifact_diagnosis.json", low_cost_rows)
    write_csv(output_dir / "low_cost_artifact_diagnosis.csv", low_cost_rows)
    flagged = [row for row in low_cost_rows if bool(row.get("low_cost_artifact"))]
    write_text(
        output_dir / "low_cost_artifact_diagnosis.md",
        "\n".join(
            [
                "# Low-Cost Artifact Diagnosis",
                "",
                f"- decision rows checked: `{len(low_cost_rows)}`",
                f"- low-cost artifact rows: `{len(flagged)}`",
                f"- fraction: `{len(flagged) / max(1, len(low_cost_rows))}`",
                "- definition: selected branch changes vs measured-only and has lower gain_exp, lower sc_gain, and lower cost than the same-tree base-exp winner.",
            ]
        ),
    )

    save_json(output_dir / "hidden_region_signal_summary.json", hidden_rows)
    write_csv(output_dir / "hidden_region_signal_summary.csv", hidden_rows)
    write_simple_md_table(
        output_dir / "hidden_region_signal_summary.md",
        "Hidden Region Signal Summary",
        hidden_rows,
        [
            "prediction_source",
            "sc_basis",
            "confidence_threshold",
            "occ_threshold",
            "free_threshold",
            "mean_selected_hidden_region_predicted_count",
            "hidden_room_selection_fraction",
        ],
    )

    save_json(output_dir / "best_config_candidates.json", best_info)
    write_simple_md_table(
        output_dir / "best_config_candidates.md",
        "Best Config Candidates",
        best_info.get("candidates", []),
        [
            "prediction_source",
            "formula_name",
            "config_key",
            "hidden_room_selection_fraction",
            "oracle_map_predict_agreement_fraction",
            "low_cost_artifact_fraction",
            "runtime_readiness",
        ],
    )

    missing: dict[str, Any] = {
        "map_predict_npz_missing": not map_available,
        "map_predict_missing_reason_file": str(output_dir / "map_predict_missing_skipped_reason.md") if not map_available else None,
        "plot_failures": [],
        "fields_missing": [],
    }
    write_plots(
        output_dir,
        observed_state,
        summary_rows,
        agreement_rows,
        threshold_rows,
        lambda_rows,
        hidden_rows,
        decision_rows,
        best,
        oracle_npz,
        map_npz if map_available else None,
        missing,
    )

    observed_hash_after = sha256_file(observed_path)
    oracle_hash_after = sha256_file(oracle_npz)
    map_hash_after = sha256_file(map_npz) if map_available else None
    safety = {
        "isaac_startup": False,
        "new_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "two_frame_runtime": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_or_rl": False,
        "checkpoint_modified": False,
        "existing_observed_state_modified": observed_hash_before != observed_hash_after,
        "prediction_npz_modified": oracle_hash_before != oracle_hash_after or map_hash_before != map_hash_after,
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_planning_scoring": False,
        "external_source_modified_or_built": False,
        "coverage_improvement_claim": False,
        "observed_state_sha256_before": observed_hash_before,
        "observed_state_sha256_after": observed_hash_after,
        "oracle_npz_sha256_before": oracle_hash_before,
        "oracle_npz_sha256_after": oracle_hash_after,
        "map_predict_npz_sha256_before": map_hash_before,
        "map_predict_npz_sha256_after": map_hash_after,
    }
    missing["safety"] = safety
    save_json(output_dir / "missing_fields_report.json", missing)

    oracle_has_robust = any(
        row["prediction_source"] == "oracle"
        and row["sc_basis"] != "hidden_region_occ_free"
        and float(row["hidden_room_selection_fraction"]) >= 0.8
        and float(row["low_cost_artifact_fraction"]) == 0.0
        for row in summary_rows
    )
    map_has_robust = any(
        row["prediction_source"] == "map_predict"
        and row["sc_basis"] != "hidden_region_occ_free"
        and float(row["hidden_room_selection_fraction"]) >= 0.8
        and float(row["low_cost_artifact_fraction"]) == 0.0
        for row in summary_rows
    )
    answers = make_answers(
        seeds=seeds,
        configs=configs,
        decisions=decision_rows,
        summary_rows=summary_rows,
        agreement_rows=agreement_rows,
        threshold_rows=threshold_rows,
        lambda_rows=lambda_rows,
        best=best,
    )
    summary = {
        "stage": "Stage 4A-6.5ab",
        "status": "completed",
        "stage4a65aa_dir": str(stage_dir),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "config_count": len(configs),
        "decision_row_count": len(decision_rows),
        "map_predict_available": map_available,
        "best_candidate_config": best,
        "best_candidate_count": best_info.get("candidate_count", 0),
        "oracle_has_robust_config": oracle_has_robust,
        "map_predict_has_robust_config": map_has_robust,
        "answers": answers,
        "safety": safety,
        "runtime_s": float(time.perf_counter() - start),
        "coverage_improvement_claimed": False,
    }
    save_json(output_dir / "stage4a65ab_synthetic_calibration_summary.json", summary)
    write_summary_md(output_dir / "stage4a65ab_synthetic_calibration_summary.md", summary)
    write_recommendation(output_dir / "recommended_next_faithful_step.md", summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65aa_dir", default=DEFAULT_STAGE4A65AA_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--confidence_thresholds", default="0.05,0.1,0.2,0.4,0.6,0.8")
    parser.add_argument("--occ_thresholds", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--free_thresholds", default="0.3,0.5,0.7,0.9")
    parser.add_argument("--lambdas", default="16,24,32,48")
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", default="mixed")
    parser.add_argument("--path_cost_mode", default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--save_viz", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
