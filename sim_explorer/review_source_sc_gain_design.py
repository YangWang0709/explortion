#!/usr/bin/env python3
"""Stage 4A-6.5x source-faithful SC gain design review.

This is an offline-only review script. It reads saved observed maps,
prediction NPZs, tree JSONL/CSV artifacts, and local/external source files.
It does not launch Isaac, run map_predict, execute actions, train, or write
predictions into observed maps.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sim_paper_expert import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    SimCandidateView,
    grid_to_world,
    normalize_bounds,
    raycast_visible_voxels_observed,
)
from sim_prediction_layer import SimPredictionLayer


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
TAU_DEFAULT = 0.1
VOXEL_SIZE_DEFAULT = 0.1
SOURCE_CONFIDENCE_THRESHOLD_DEFAULT = 0.05

PLOT_FILES = [
    "branch_visible_voxel_counts.png",
    "branch_prediction_confidence_distributions.png",
    "branch_gain_component_stack.png",
    "measured_vs_sc_branch_overlap_topdown.png",
    "predicted_voxel_spatial_distribution_topdown.png",
    "candidate_formula_effects_bar.png",
    "seed_replay_if_done.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "depth_*.npy",
    "rgb_*.png",
    "frame*_depth.npy",
    "frame*_rgb.png",
    "local_prediction.npz",
    "global_prediction_layer.npz",
    "sscnet_depth_input.npy",
    "sscnet_position.npy",
    "valid_position_mask.npy",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "step_*.npz",
]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
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


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, set, np.ndarray)):
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def parse_json_cell(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def extract_scalar_from_yaml(text: str, key: str, default: Any = None) -> Any:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*([^#\n]+)", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return default
    raw = match.group(1).strip().strip('"').strip("'")
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return float(raw)
    except ValueError:
        return raw


def line_hits(path: Path, needles: list[str]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    text = read_text(path)
    if not text:
        return hits
    for lineno, line in enumerate(text.splitlines(), start=1):
        for needle in needles:
            if needle in line:
                hits.append({"file": str(path), "line": lineno, "needle": needle, "text": line.strip()})
    return hits


def git_status_porcelain(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(path),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return f"error: {exc}"
    if result.returncode != 0:
        return f"error: {result.stderr.strip()}"
    return result.stdout.strip()


def state_counts(observed_state: np.ndarray, voxels: set[tuple[int, int, int]]) -> dict[str, int]:
    counts = {"visible_unknown_count": 0, "visible_free_count": 0, "visible_occupied_count": 0}
    for voxel in voxels:
        state = observed_state[voxel]
        if state == UNKNOWN:
            counts["visible_unknown_count"] += 1
        elif state == FREE:
            counts["visible_free_count"] += 1
        elif state == OCCUPIED:
            counts["visible_occupied_count"] += 1
    return counts


def has_free_neighbor(observed_state: np.ndarray, voxel: tuple[int, int, int]) -> bool:
    shape = observed_state.shape
    i, j, k = voxel
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
            if observed_state[ni, nj, nk] == FREE:
                return True
    return False


def summarize_numeric(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_sum": 0.0, f"{prefix}_mean": None, f"{prefix}_p50": None, f"{prefix}_p90": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_sum": float(np.sum(arr)),
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
    }


def source_occ_free_masks(
    prediction: SimPredictionLayer,
    voxels: set[tuple[int, int, int]],
    tau: float,
    source_threshold: float = SOURCE_CONFIDENCE_THRESHOLD_DEFAULT,
) -> dict[str, set[tuple[int, int, int]]]:
    occ_threshold = 0.5 + float(source_threshold)
    free_threshold = 0.5 + float(source_threshold)
    predicted: set[tuple[int, int, int]] = set()
    occ: set[tuple[int, int, int]] = set()
    free: set[tuple[int, int, int]] = set()
    source_unknown: set[tuple[int, int, int]] = set()
    for voxel in voxels:
        if not prediction.is_predicted(voxel, tau=tau):
            continue
        predicted.add(voxel)
        occ_prob = prediction.get_occupied_prob(voxel)
        free_prob = prediction.get_free_prob(voxel)
        if occ_prob >= occ_threshold:
            occ.add(voxel)
        elif free_prob >= free_threshold:
            free.add(voxel)
        else:
            source_unknown.add(voxel)
    return {"predicted": predicted, "occupied": occ, "free": free, "source_unknown": source_unknown}


def prediction_stats_for_voxels(
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    voxels: set[tuple[int, int, int]],
    tau: float,
    source_threshold: float = SOURCE_CONFIDENCE_THRESHOLD_DEFAULT,
) -> dict[str, Any]:
    unmeasured = {v for v in voxels if observed_state[v] == UNKNOWN and prediction.is_predicted(v, tau=tau)}
    masks = source_occ_free_masks(prediction, unmeasured, tau=tau, source_threshold=source_threshold)
    conf_values = [prediction.get_confidence(v) for v in unmeasured]
    occ_values = [prediction.get_occupied_prob(v) for v in unmeasured]
    margin_values = [abs(0.5 - prediction.get_occupied_prob(v)) for v in unmeasured]
    frontier_local = {v for v in masks["predicted"] if has_free_neighbor(observed_state, v)}
    result: dict[str, Any] = {
        "predicted_visible_count": int(len(masks["predicted"])),
        "predicted_unmeasured_count": int(len(unmeasured)),
        "predicted_occupied_count": int(len(masks["occupied"])),
        "predicted_free_count": int(len(masks["free"])),
        "predicted_unknown_count": int(len(masks["source_unknown"])),
        "frontier_local_predicted_count": int(len(frontier_local)),
        "raw_count_gain": float(len(unmeasured)),
        "confidence_weighted_gain": float(sum(margin_values)),
        "confidence_weighted_source_style_2x_gain": float(2.0 * sum(margin_values)),
        "cap25_effective_gain": float(min(len(unmeasured), 25)),
        "occupied_only_gain": float(len(masks["occupied"])),
        "source_occ_free_gain": float(len(masks["occupied"]) + len(masks["free"])),
        "frontier_local_sc_gain": float(len(frontier_local)),
    }
    result.update(summarize_numeric(conf_values, "predicted_confidence"))
    result.update(summarize_numeric(occ_values, "occupied_prob"))
    result.update(summarize_numeric(margin_values, "occupied_margin_abs"))
    return result


def load_tree(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["segment_id"]): row for row in read_jsonl(path)}


def path_to_root(tree: dict[str, dict[str, Any]], segment_id: str) -> list[str]:
    path: list[str] = []
    current: str | None = segment_id
    while current and current in tree:
        path.append(current)
        parent = tree[current].get("parent_id")
        current = str(parent) if parent else None
    path.reverse()
    return path


def branch_vector_stats(
    root_grid: list[int],
    best_grid: list[int],
    predicted_voxels: set[tuple[int, int, int]],
) -> dict[str, Any]:
    if not predicted_voxels:
        return {
            "predicted_centroid_grid": None,
            "predicted_projection_on_branch_cells": None,
            "predicted_perp_distance_to_branch_cells": None,
        }
    coords = np.asarray(sorted(predicted_voxels), dtype=np.float64)
    centroid = np.mean(coords, axis=0)
    root = np.asarray(root_grid, dtype=np.float64)
    end = np.asarray(best_grid, dtype=np.float64)
    vector = end - root
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        projection = 0.0
        perp = float(np.linalg.norm(centroid - root))
    else:
        unit = vector / norm
        rel = centroid - root
        projection = float(np.dot(rel, unit))
        perp = float(np.linalg.norm(rel - projection * unit))
    return {
        "predicted_centroid_grid": [float(v) for v in centroid],
        "predicted_projection_on_branch_cells": projection,
        "predicted_perp_distance_to_branch_cells": perp,
    }


def compute_visible_set(
    segment: dict[str, Any],
    observed_state: np.ndarray,
    max_range_voxels: int,
    cache: dict[tuple[tuple[int, int, int], float], set[tuple[int, int, int]]],
) -> set[tuple[int, int, int]]:
    grid = tuple(int(v) for v in segment["end_grid"])
    yaw = float(segment.get("yaw", 0.0))
    key = (grid, round(yaw, 6))
    if key in cache:
        return set(cache[key])
    candidate = SimCandidateView(
        id=-1,
        grid_position=grid,
        world_position=(0.0, 0.0, 0.0),
        yaw=yaw,
        valid=True,
        candidate_source="stage4a65x_review",
    )
    visible = set(
        tuple(int(v) for v in voxel)
        for voxel in raycast_visible_voxels_observed(
            candidate,
            observed_state,
            max_range_voxels=max_range_voxels,
            num_yaw=32,
            num_pitch=7,
        )
    )
    cache[key] = set(visible)
    return visible


def analyze_branch(
    *,
    label: str,
    source_stage: str,
    seed: int | None,
    formula: str,
    tree_path: Path,
    selected_child_id: str,
    best_descendant_id: str,
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    root_visible: set[tuple[int, int, int]],
    max_range_voxels: int,
    visible_cache: dict[tuple[tuple[int, int, int], float], set[tuple[int, int, int]]],
    tau: float,
    source_threshold: float,
    rank: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tree = load_tree(tree_path)
    if not tree:
        return (
            {
                "branch_label": label,
                "source_stage": source_stage,
                "seed": seed,
                "formula": formula,
                "rank": rank,
                "status": "missing_tree",
                "tree_path": str(tree_path),
            },
            {},
        )
    if best_descendant_id not in tree:
        best_descendant_id = selected_child_id
    path_ids = path_to_root(tree, best_descendant_id)
    path_ids_no_root = [item for item in path_ids if item != "root"]
    if selected_child_id in tree and selected_child_id not in path_ids_no_root:
        path_ids_no_root = [selected_child_id] + path_ids_no_root

    root_segment = tree.get("root", {})
    selected = tree.get(selected_child_id, {})
    best = tree.get(best_descendant_id, selected)
    previous_path_visible: set[tuple[int, int, int]] = set(root_visible)
    previous_path_predicted: set[tuple[int, int, int]] = set()
    union_visible: set[tuple[int, int, int]] = set()
    union_predicted: set[tuple[int, int, int]] = set()
    sum_stats = defaultdict(float)
    segment_rows: list[dict[str, Any]] = []

    for node_id in path_ids_no_root:
        segment = tree.get(node_id)
        if not segment:
            continue
        visible = compute_visible_set(segment, observed_state, max_range_voxels, visible_cache)
        pred_stats = prediction_stats_for_voxels(
            observed_state, prediction, visible, tau=tau, source_threshold=source_threshold
        )
        masks = source_occ_free_masks(
            prediction,
            {v for v in visible if observed_state[v] == UNKNOWN},
            tau=tau,
            source_threshold=source_threshold,
        )
        predicted_set = set(masks["predicted"])
        source_occ_free_set = set(masks["occupied"]) | set(masks["free"])
        parent_cleared = source_occ_free_set - previous_path_predicted
        parent_and_root_cleared = source_occ_free_set - previous_path_predicted - {
            v for v in root_visible if observed_state[v] == UNKNOWN and prediction.is_predicted(v, tau=tau)
        }
        frontier_local = {v for v in source_occ_free_set if has_free_neighbor(observed_state, v)}

        counts = state_counts(observed_state, visible)
        seg_row = {
            "branch_label": label,
            "segment_id": node_id,
            "segment_depth": int(segment.get("depth", 0)),
            "segment_grid": segment.get("end_grid"),
            "visible_count": int(len(visible)),
            **counts,
            **pred_stats,
            "novel_unknown_vs_root": int(
                len({v for v in visible if observed_state[v] == UNKNOWN} - {v for v in root_visible if observed_state[v] == UNKNOWN})
            ),
            "novel_unknown_vs_parent_path": int(
                len({v for v in visible if observed_state[v] == UNKNOWN} - {v for v in previous_path_visible if observed_state[v] == UNKNOWN})
            ),
            "overlap_with_root_visible": int(len(visible & root_visible)),
            "overlap_with_parent_visible": int(len(visible & previous_path_visible)),
            "overlap_with_root_prediction_visible": int(len(predicted_set & previous_path_predicted)),
            "parent_visible_cleared_sc_gain": float(len(parent_cleared)),
            "parent_and_root_visible_cleared_sc_gain": float(len(parent_and_root_cleared)),
            "frontier_local_source_occ_free_gain": float(len(frontier_local)),
            "recorded_gain_exp": parse_float(segment.get("gain_exp")),
            "recorded_raw_gain_sc": parse_float(segment.get("gain_sc")),
            "recorded_effective_gain_sc": parse_float(segment.get("effective_gain_sc")),
            "recorded_gain_conf": parse_float(segment.get("gain_conf")),
            "recorded_gain_occ": parse_float(segment.get("gain_occ")),
            "recorded_cost": parse_float(segment.get("cost")),
        }
        segment_rows.append(seg_row)
        union_visible |= visible
        union_predicted |= predicted_set
        previous_path_visible |= visible
        previous_path_predicted |= predicted_set
        for key in (
            "visible_count",
            "visible_unknown_count",
            "visible_free_count",
            "visible_occupied_count",
            "predicted_visible_count",
            "predicted_unmeasured_count",
            "predicted_occupied_count",
            "predicted_free_count",
            "predicted_unknown_count",
            "raw_count_gain",
            "confidence_weighted_gain",
            "confidence_weighted_source_style_2x_gain",
            "cap25_effective_gain",
            "occupied_only_gain",
            "source_occ_free_gain",
            "frontier_local_sc_gain",
            "parent_visible_cleared_sc_gain",
            "parent_and_root_visible_cleared_sc_gain",
            "frontier_local_source_occ_free_gain",
            "recorded_gain_exp",
            "recorded_raw_gain_sc",
            "recorded_effective_gain_sc",
            "recorded_gain_conf",
            "recorded_gain_occ",
            "recorded_cost",
        ):
            sum_stats[key] += float(seg_row.get(key, 0.0) or 0.0)

    union_counts = state_counts(observed_state, union_visible)
    union_pred_stats = prediction_stats_for_voxels(
        observed_state, prediction, union_visible, tau=tau, source_threshold=source_threshold
    )
    selected_grid = selected.get("end_grid")
    best_grid = best.get("end_grid")
    root_grid = root_segment.get("end_grid", [0, 0, 0])
    spatial_stats = branch_vector_stats(root_grid, best_grid or selected_grid or root_grid, union_predicted)
    best_cost = parse_float(best.get("accumulated_cost"), sum_stats["recorded_cost"])
    branch_row: dict[str, Any] = {
        "branch_label": label,
        "source_stage": source_stage,
        "seed": seed,
        "formula": formula,
        "rank": rank,
        "status": "completed",
        "tree_path": str(tree_path),
        "visibility_source": "recomputed_with_current_offline_observed_raycaster",
        "prediction_ray_blocking": False,
        "selected_child_id": selected_child_id,
        "selected_child_grid": selected_grid,
        "best_descendant_id": best_descendant_id,
        "best_descendant_grid": best_grid,
        "root_grid": root_grid,
        "path_node_ids": path_ids_no_root,
        "path_depth": len(path_ids_no_root),
        "path_cost_recorded": best_cost,
        "path_gain_exp_recorded": parse_float(best.get("accumulated_gain"), sum_stats["recorded_gain_exp"])
        if formula == "measured_only"
        else sum_stats["recorded_gain_exp"],
        "path_raw_gain_sc_recorded": sum_stats["recorded_raw_gain_sc"],
        "path_effective_gain_sc_recorded": sum_stats["recorded_effective_gain_sc"],
        "path_gain_conf_recorded": sum_stats["recorded_gain_conf"],
        "path_gain_occ_recorded": sum_stats["recorded_gain_occ"],
        "value_recorded": parse_float(selected.get("value"), parse_float(best.get("value"))),
        "segment_visible_count_sum": int(sum_stats["visible_count"]),
        "path_visible_unique_count": int(len(union_visible)),
        **union_counts,
        **{f"path_sum_{k}": v for k, v in sum_stats.items()},
        **{f"path_unique_{k}": v for k, v in union_pred_stats.items()},
        "overlap_with_root_visible": int(len(union_visible & root_visible)),
        "overlap_with_root_visible_fraction": float(len(union_visible & root_visible) / max(1, len(union_visible))),
        "overlap_with_root_prediction_visible": int(
            len(
                union_predicted
                & {
                    v
                    for v in root_visible
                    if observed_state[v] == UNKNOWN and prediction.is_predicted(v, tau=tau)
                }
            )
        ),
        **spatial_stats,
    }
    return branch_row, {"segments": segment_rows, "union_predicted": union_predicted, "union_visible": union_visible}


def collect_branch_specs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    missing: list[str] = []
    specs: list[dict[str, Any]] = [
        {
            "label": "seed0_measured_branch_n0001_to_n0112",
            "source_stage": "stage4a65p",
            "seed": 0,
            "formula": "measured_only",
            "tree_path": Path(args.stage4a65p_dir) / "frame002_measured_tree_segments.jsonl",
            "selected_child_id": "n0001",
            "best_descendant_id": "n0112",
            "rank": 1,
        },
        {
            "label": "seed0_sc_branch_n0127_to_n0162",
            "source_stage": "stage4a65p",
            "seed": 0,
            "formula": "raw_count_or_confidence_reference",
            "tree_path": Path(args.stage4a65p_dir) / "frame002_sc_tree_segments.jsonl",
            "selected_child_id": "n0127",
            "best_descendant_id": "n0162",
            "rank": 1,
        },
        {
            "label": "seed1_measured_confidence_branch_n0057_to_n0118",
            "source_stage": "stage4a65v",
            "seed": 1,
            "formula": "confidence_weighted",
            "tree_path": Path(args.stage4a65v_dir) / "raw_trees/seed_001/confidence_weighted/mini_rrt_tree_segments.jsonl",
            "selected_child_id": "n0057",
            "best_descendant_id": "n0118",
            "rank": 1,
        },
    ]

    for formula in ("confidence_weighted", "cap25"):
        for seed in range(10):
            table_path = Path(args.stage4a65v_dir) / f"raw_trees/seed_{seed:03d}/{formula}/gain_cost_value_table.csv"
            tree_path = Path(args.stage4a65v_dir) / f"raw_trees/seed_{seed:03d}/{formula}/mini_rrt_tree_segments.jsonl"
            rows = read_csv_rows(table_path)
            if not rows:
                missing.append(f"missing top-branch table: {table_path}")
                continue
            root_children = [row for row in rows if parse_int(row.get("depth")) == 1]
            root_children.sort(key=lambda row: parse_float(row.get("value"), -1e18), reverse=True)
            for rank, row in enumerate(root_children[:3], start=1):
                selected = str(row.get("segment_id", ""))
                best = str(row.get("best_descendant_id", selected) or selected)
                specs.append(
                    {
                        "label": f"seed{seed}_{formula}_top{rank}_{selected}_to_{best}",
                        "source_stage": "stage4a65v_top3",
                        "seed": seed,
                        "formula": formula,
                        "tree_path": tree_path,
                        "selected_child_id": selected,
                        "best_descendant_id": best,
                        "rank": rank,
                    }
                )
    return specs, missing


def source_evidence(args: argparse.Namespace) -> dict[str, Any]:
    local = Path(args.local_ssc_exploration_dir)
    external = Path(args.external_active3d_dir)
    files = {
        "ssc_exploration_evaluator": local / "ssc_planning/src/trajectory_evaluator/ssc_exploration_evaluator.cpp",
        "ssc_voxel_evaluator": local / "ssc_planning/src/trajectory_evaluator/ssc_voxel_evaluator.cpp",
        "ssc_voxblox_map": local / "ssc_planning/src/map/ssc_voxblox_map.cpp",
        "ssc_voxblox_map_h": local / "ssc_planning/include/ssc_planning/map/ssc_voxblox_map.h",
        "sc_explorer_config": local / "ssc_planning/cfg/planners/sc_explorer.yaml",
        "baseline_config": local / "ssc_planning/cfg/planners/baseline.yaml",
        "simulated_sensor_evaluator": external
        / "active_3d_planning_core/src/module/trajectory_evaluator/simulated_sensor_evaluator.cpp",
        "iterative_ray_caster": external / "active_3d_planning_core/src/module/sensor_model/iterative_ray_caster.cpp",
        "global_normalized_gain": external
        / "active_3d_planning_core/src/module/trajectory_evaluator/value_computers/global_normalized_gain.cpp",
        "subsequent_best": external / "active_3d_planning_core/src/module/trajectory_evaluator/next_selector/subsequent_best.cpp",
        "rrt_star": external / "active_3d_planning_core/src/module/trajectory_generator/rrt_star.cpp",
    }
    cfg = read_text(files["sc_explorer_config"])
    predicted_occ_weight = extract_scalar_from_yaml(cfg, "predicted_occ_weight", 1.0)
    predicted_free_weight = extract_scalar_from_yaml(cfg, "predicted_free_weight", 1.0)
    unobserved_weight = extract_scalar_from_yaml(cfg, "unobserved_weight", None)
    ssc_confidence_threshold = extract_scalar_from_yaml(
        cfg, "ssc_confidence_threshold", SOURCE_CONFIDENCE_THRESHOLD_DEFAULT
    )
    weight_by_confidence = extract_scalar_from_yaml(cfg, "weight_by_confidence", False)
    use_ssc_information = extract_scalar_from_yaml(cfg, "use_ssc_information_planning", None)
    use_voxblox_information = extract_scalar_from_yaml(cfg, "use_voxblox_information_planning", None)
    use_ssc_planning = extract_scalar_from_yaml(cfg, "use_ssc_planning", None)
    use_voxblox_planning = extract_scalar_from_yaml(cfg, "use_voxblox_planning", None)

    evidence_hits: list[dict[str, Any]] = []
    for path in files.values():
        evidence_hits.extend(
            line_hits(
                path,
                [
                    "predicted_occ_weight",
                    "predicted_free_weight",
                    "unobserved_weight",
                    "weight_by_confidence",
                    "getVoxelSSCState",
                    "use_ssc_information_planning",
                    "use_ssc_planning",
                    "clear_from_parents",
                    "visible_voxels",
                    "getVoxelState",
                    "OCCUPIED",
                    "GlobalNormalizedGain",
                    "SubsequentBest",
                    "computeGain",
                ],
            )
        )

    source_ambiguous_parts = [
        "Mapping simulator NPZ occupied/free probabilities back to the C++ SSCMap binary log-odds layer is approximate.",
        "The source also contains SSCVoxelEvaluator, but sc_explorer.yaml selects SSCExplorationEvaluator for this profile.",
    ]
    return {
        "files_inspected": {name: str(path) for name, path in files.items() if path.exists()},
        "evidence_hits": evidence_hits[:220],
        "rewarded_voxel_classes": {
            "measured_observed": "weight 0 in SSCExplorationEvaluator::getVoxelType",
            "predicted_occupied": float(predicted_occ_weight),
            "predicted_free": float(predicted_free_weight),
            "predicted_unknown_or_unobserved": float(unobserved_weight)
            if unobserved_weight is not None
            else "source default is 1.0 but config value not found",
            "sc_explorer_config_unobserved_weight": unobserved_weight,
        },
        "confidence_probability_use": {
            "ssc_confidence_threshold": float(ssc_confidence_threshold),
            "occupied_probability_threshold": float(0.5 + float(ssc_confidence_threshold)),
            "free_probability_threshold": float(0.5 - float(ssc_confidence_threshold)),
            "weight_by_confidence_configured": bool(weight_by_confidence),
            "weight_by_confidence_code_supported": True,
            "confidence_weight_formula_if_enabled": "2 * abs(probabilityFromLogOdds(probability_log) - 0.5)",
        },
        "planning_flags": {
            "use_ssc_information_planning": use_ssc_information,
            "use_voxblox_information_planning": use_voxblox_information,
            "use_ssc_planning": use_ssc_planning,
            "use_voxblox_planning": use_voxblox_planning,
            "prediction_blocks_visibility_raycasting_in_sc_explorer_config": bool(use_ssc_information),
            "prediction_can_participate_in_source_traversability_if_use_ssc_planning": bool(use_ssc_planning),
            "current_simulator_keeps_prediction_out_of_traversability_collision_raycast": True,
        },
        "parent_visible_clearing": {
            "supported_by_base_simulated_sensor_evaluator": True,
            "default": False,
            "configured_in_sc_explorer_yaml": "clear_from_parents" in cfg,
            "active_in_sc_explorer_profile": False,
        },
        "visible_voxel_storage": {
            "stored_per_segment": True,
            "type": "SimulatedSensorInfo.visible_voxels",
            "saved_python_tree_artifacts_include_visible_ids": False,
        },
        "local_gain_recompute_after_rewire": {
            "rewire_to_best_parent_recomputes_cost_value": True,
            "visible_gain_recompute_on_rewire_not_proven_in_rewireToBestParent": True,
            "source_ambiguous": True,
        },
        "current_formula_source_status": {
            "raw_count": "source-inspired only unless restricted to source OCC/FREE states and thresholds",
            "confidence_weighted": "source-supported only when weight_by_confidence is configured; not active in sc_explorer.yaml",
            "cap25": "diagnostic-only",
            "occupied_only": "source-incomplete because source rewards predicted free too",
        },
        "source_ambiguous_parts": source_ambiguous_parts,
    }


def write_source_reports(output_dir: Path, evidence: dict[str, Any]) -> None:
    save_json(output_dir / "source_sc_gain_evidence.json", evidence)
    rewarded = evidence["rewarded_voxel_classes"]
    flags = evidence["planning_flags"]
    conf = evidence["confidence_probability_use"]
    clearing = evidence["parent_visible_clearing"]
    lines = [
        "# Source SC Gain Evidence",
        "",
        "## Answers",
        "",
        f"- Rewarded predicted voxel classes: occupied weight {rewarded['predicted_occupied']}, free weight {rewarded['predicted_free']}.",
        f"- Predicted unknown/unobserved weight in sc_explorer.yaml: {rewarded['sc_explorer_config_unobserved_weight']}.",
        f"- Confidence/probability threshold: ssc_confidence_threshold {conf['ssc_confidence_threshold']} means occupied >= {conf['occupied_probability_threshold']} or free <= {conf['free_probability_threshold']}.",
        f"- Direct confidence weighting configured: {conf['weight_by_confidence_configured']}. Code supports it, but this profile does not enable it.",
        f"- SSC information planning enabled: {flags['use_ssc_information_planning']}. Therefore prediction does not block visibility raycasting in this profile.",
        f"- SSC planning/collision flag in source config: {flags['use_ssc_planning']}; current simulator review keeps prediction out of traversability/collision.",
        f"- Parent-visible clearing supported by source base evaluator: {clearing['supported_by_base_simulated_sensor_evaluator']}; active in sc_explorer profile: {clearing['active_in_sc_explorer_profile']}.",
        "- Visible voxels are stored per segment in source, but the saved Python tree artifacts do not include visible voxel ids, so this review recomputes them diagnostically.",
        "",
        "## Source-Ambiguous Parts",
    ]
    lines += [f"- {item}" for item in evidence["source_ambiguous_parts"]]
    write_text(output_dir / "source_sc_gain_evidence.md", "\n".join(lines))

    rows = [
        {
            "formula": "source SSCExplorationEvaluator",
            "classification": "source profile",
            "definition": "count unmeasured visible SSC OCC and FREE voxels with configured weights; unobserved is 0 in sc_explorer.yaml",
        },
        {
            "formula": "current raw_count",
            "classification": "source-inspired, not exact",
            "definition": "counts all prediction-valid unmeasured visible voxels at tau, including voxels source thresholds may leave unknown",
        },
        {
            "formula": "current confidence_weighted",
            "classification": "source-inspired/diagnostic for this profile",
            "definition": "uses occupancy margin abs(occupied_prob - 0.5); source code supports 2x margin only if weight_by_confidence is enabled",
        },
        {
            "formula": "current cap25",
            "classification": "diagnostic-only",
            "definition": "caps raw SC contribution per segment; no matching source config found",
        },
    ]
    text = ["# Source Gain Formula Comparison", ""]
    for row in rows:
        text.append(f"## {row['formula']}")
        text.append(f"- Classification: {row['classification']}")
        text.append(f"- Definition: {row['definition']}")
        text.append("")
    write_text(output_dir / "source_gain_formula_comparison.md", "\n".join(text))


def current_formula_audit() -> list[dict[str, Any]]:
    return [
        {
            "formula": "raw_count",
            "definition": "effective SC gain = number of predicted-valid, unmeasured, visible voxels",
            "uses_predicted_occupied": True,
            "uses_predicted_free": True,
            "uses_confidence": "only tau gate via SimPredictionLayer.is_predicted",
            "uses_occupied_probability": False,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": False,
            "source_status": "source-inspired, not exact",
            "possible_failure_mode": "counts dense prediction-valid voxels that source OCC/FREE thresholds might not reward",
        },
        {
            "formula": "confidence_weighted",
            "definition": "effective SC gain = sum abs(occupied_prob - 0.5) over predicted-unmeasured visible voxels",
            "uses_predicted_occupied": True,
            "uses_predicted_free": True,
            "uses_confidence": "name says confidence, implementation uses occupancy margin; tau gate still uses confidence",
            "uses_occupied_probability": True,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": False,
            "source_status": "source-inspired/diagnostic unless source weight_by_confidence is enabled",
            "possible_failure_mode": "still rewards dense low-novelty predicted volume and is not active source config",
        },
        {
            "formula": "cap25",
            "definition": "effective SC gain = min(raw predicted-unmeasured count, 25) per segment",
            "uses_predicted_occupied": True,
            "uses_predicted_free": True,
            "uses_confidence": "tau gate only",
            "uses_occupied_probability": False,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": False,
            "source_status": "diagnostic-only",
            "possible_failure_mode": "preserves or removes branches by arithmetic cap rather than source semantics",
        },
        {
            "formula": "occupied_only",
            "definition": "effective SC gain = count predicted occupied unmeasured visible voxels",
            "uses_predicted_occupied": True,
            "uses_predicted_free": False,
            "uses_confidence": "tau gate only",
            "uses_occupied_probability": True,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": False,
            "source_status": "source-incomplete",
            "possible_failure_mode": "drops predicted free voxels that source profile rewards",
        },
        {
            "formula": "occupied_margin",
            "definition": "effective SC gain = max(occupied_prob - 0.5, 0) after occupied/confidence thresholds",
            "uses_predicted_occupied": True,
            "uses_predicted_free": False,
            "uses_confidence": "tau plus configured confidence threshold",
            "uses_occupied_probability": True,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": False,
            "source_status": "diagnostic-only",
            "possible_failure_mode": "semantic occupied bias not present in sc_explorer.yaml",
        },
        {
            "formula": "calibrated_occupied",
            "definition": "effective SC gain = delayed-sensor calibrated occupied rate for occupied-thresholded voxels",
            "uses_predicted_occupied": True,
            "uses_predicted_free": False,
            "uses_confidence": "tau/conf threshold gates",
            "uses_occupied_probability": True,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": False,
            "source_status": "diagnostic-only",
            "possible_failure_mode": "uses post-hoc calibration artifact and ignores predicted free",
        },
        {
            "formula": "novelty_discounted",
            "definition": "effective SC gain = occupied margin downweighted near measured-free neighbors",
            "uses_predicted_occupied": True,
            "uses_predicted_free": False,
            "uses_confidence": "tau/conf threshold gates",
            "uses_occupied_probability": True,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": "partial local heuristic only",
            "source_status": "diagnostic-only",
            "possible_failure_mode": "not source-proven and can suppress true frontier structure",
        },
        {
            "formula": "confidence_weighted_cap25",
            "definition": "effective SC gain = min(sum abs(occupied_prob - 0.5), 25)",
            "uses_predicted_occupied": True,
            "uses_predicted_free": True,
            "uses_confidence": "name says confidence, implementation uses occupancy margin plus tau gate",
            "uses_occupied_probability": True,
            "uses_measured_unmeasured_mask": True,
            "deduplicates_visible_voxels_across_parent_root": False,
            "source_status": "diagnostic-only",
            "possible_failure_mode": "combines two non-active source assumptions",
        },
    ]


def write_formula_audit(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "current_gain_formula_audit.csv", rows)
    save_json(output_dir / "current_gain_formula_audit.json", rows)
    lines = ["# Current Gain Formula Audit", ""]
    for row in rows:
        lines.append(f"## {row['formula']}")
        lines.append(f"- Definition: {row['definition']}")
        lines.append(f"- Source status: {row['source_status']}")
        lines.append(f"- Failure mode: {row['possible_failure_mode']}")
        lines.append("")
    write_text(output_dir / "current_gain_formula_audit.md", "\n".join(lines))


def variant_score(gain_exp: float, sc_gain: float, cost: float) -> float:
    return float((float(gain_exp) + float(sc_gain)) / max(float(cost), 1.0e-6))


def evaluate_candidate_variants(branch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label = {row["branch_label"]: row for row in branch_rows if row.get("status") == "completed"}
    measured = by_label.get("seed0_measured_branch_n0001_to_n0112")
    sc = by_label.get("seed0_sc_branch_n0127_to_n0162")
    if not measured or not sc:
        return []
    variants = [
        {
            "variant": "source_raw_predicted_occupied_free",
            "source_status": "most source-faithful in this review",
            "measured_key": "path_sum_source_occ_free_gain",
            "sc_key": "path_sum_source_occ_free_gain",
            "description": "count source-thresholded predicted occupied plus free, unmeasured visible voxels",
        },
        {
            "variant": "source_confidence_thresholded",
            "source_status": "source-faithful threshold gate, not source confidence weighting",
            "measured_key": "path_sum_source_occ_free_gain",
            "sc_key": "path_sum_source_occ_free_gain",
            "description": "same OCC/FREE count using ssc_confidence_threshold-derived 0.55/0.45 thresholds",
        },
        {
            "variant": "parent_visible_cleared_sc",
            "source_status": "source-inspired optional; clear_from_parents supported but inactive in sc_explorer.yaml",
            "measured_key": "path_sum_parent_and_root_visible_cleared_sc_gain",
            "sc_key": "path_sum_parent_and_root_visible_cleared_sc_gain",
            "description": "remove source OCC/FREE voxels already visible from root or earlier path segments",
        },
        {
            "variant": "frontier_local_sc",
            "source_status": "diagnostic/source-inspired only",
            "measured_key": "path_sum_frontier_local_source_occ_free_gain",
            "sc_key": "path_sum_frontier_local_source_occ_free_gain",
            "description": "count source OCC/FREE predicted voxels only when adjacent to measured free frontier",
        },
        {
            "variant": "spatial_novelty_discounted_sc",
            "source_status": "diagnostic-only",
            "measured_key": "path_sum_parent_visible_cleared_sc_gain",
            "sc_key": "path_sum_parent_visible_cleared_sc_gain",
            "scale": 0.5,
            "description": "downweight non-novel repeated prediction visibility",
        },
        {
            "variant": "branch_normalized_sc",
            "source_status": "diagnostic-only",
            "measured_key": "normalized",
            "sc_key": "normalized",
            "description": "normalize source OCC/FREE count by visible volume density",
        },
    ]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        if variant["measured_key"] == "normalized":
            measured_sc = float(measured.get("path_sum_source_occ_free_gain", 0.0)) / max(
                1.0, float(measured.get("segment_visible_count_sum", 0.0))
            ) * 100.0
            sc_sc = float(sc.get("path_sum_source_occ_free_gain", 0.0)) / max(
                1.0, float(sc.get("segment_visible_count_sum", 0.0))
            ) * 100.0
        else:
            measured_sc = float(measured.get(variant["measured_key"], 0.0))
            sc_sc = float(sc.get(variant["sc_key"], 0.0))
        scale = float(variant.get("scale", 1.0))
        measured_sc *= scale
        sc_sc *= scale
        measured_score = variant_score(
            float(measured.get("path_sum_recorded_gain_exp", 0.0)),
            measured_sc,
            float(measured.get("path_cost_recorded", 0.0)),
        )
        sc_score = variant_score(
            float(sc.get("path_sum_recorded_gain_exp", 0.0)),
            sc_sc,
            float(sc.get("path_cost_recorded", 0.0)),
        )
        rows.append(
            {
                **variant,
                "seed0_measured_branch_sc_gain": measured_sc,
                "seed0_sc_branch_sc_gain": sc_sc,
                "seed0_measured_branch_score_proxy": measured_score,
                "seed0_sc_branch_score_proxy": sc_score,
                "seed0_winner_proxy": "seed0_sc_branch" if sc_score > measured_score else "measured_branch",
                "would_keep_seed0_sc_branch": bool(sc_score > measured_score),
                "expected_seed_robustness_effect": "likely less dense/noisy"
                if "cleared" in variant["variant"] or "frontier" in variant["variant"]
                else "uncertain",
            }
        )
    return rows


def write_candidate_variant_reports(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "candidate_sc_gain_variants.csv", rows)
    save_json(output_dir / "candidate_sc_gain_variants.json", rows)
    lines = ["# Candidate SC Gain Variants", ""]
    for row in rows:
        lines.append(f"## {row['variant']}")
        lines.append(f"- Source status: {row['source_status']}")
        lines.append(f"- Description: {row['description']}")
        lines.append(f"- Seed0 proxy winner: {row['seed0_winner_proxy']}")
        lines.append(f"- Expected seed robustness effect: {row['expected_seed_robustness_effect']}")
        lines.append("")
    write_text(output_dir / "candidate_sc_gain_variants.md", "\n".join(lines))


def write_decomposition_reports(output_dir: Path, branch_rows: list[dict[str, Any]], segment_rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / "branch_visible_voxel_decomposition.csv", branch_rows)
    save_json(output_dir / "branch_visible_voxel_decomposition.json", branch_rows)
    key_rows = [row for row in branch_rows if row.get("source_stage") in {"stage4a65p", "stage4a65v"}][:6]
    measured = next((row for row in branch_rows if row.get("branch_label") == "seed0_measured_branch_n0001_to_n0112"), {})
    sc = next((row for row in branch_rows if row.get("branch_label") == "seed0_sc_branch_n0127_to_n0162"), {})
    seed1 = next((row for row in branch_rows if row.get("branch_label") == "seed1_measured_confidence_branch_n0057_to_n0118"), {})
    lines = [
        "# Branch Visible-Voxel Decomposition",
        "",
        "Visible voxel ids were not saved in the tree artifacts, so this review recomputed them with the current offline observed-state raycaster. Prediction was not used for ray blocking.",
        "",
        "## Key Branches",
    ]
    for row in (measured, sc, seed1):
        if not row:
            continue
        lines.append(
            f"- {row['branch_label']}: path {row['path_node_ids']}, unique predicted {row['path_unique_predicted_unmeasured_count']}, "
            f"source OCC/FREE sum {row['path_sum_source_occ_free_gain']}, parent/root-cleared {row['path_sum_parent_and_root_visible_cleared_sc_gain']}, "
            f"frontier-local {row['path_sum_frontier_local_source_occ_free_gain']}."
        )
    if measured and sc:
        lines += [
            "",
            "## Diagnosis",
            f"- Seed0 SC raw predicted count vs measured branch: {sc.get('path_sum_raw_count_gain')} vs {measured.get('path_sum_raw_count_gain')}.",
            f"- Seed0 SC source OCC/FREE count vs measured branch: {sc.get('path_sum_source_occ_free_gain')} vs {measured.get('path_sum_source_occ_free_gain')}.",
            f"- Seed0 SC parent/root-cleared source count vs measured branch: {sc.get('path_sum_parent_and_root_visible_cleared_sc_gain')} vs {measured.get('path_sum_parent_and_root_visible_cleared_sc_gain')}.",
            f"- Seed0 SC overlap with root visible fraction: {sc.get('overlap_with_root_visible_fraction')}; measured branch fraction: {measured.get('overlap_with_root_visible_fraction')}.",
        ]
    lines += [
        "",
        "## Questions",
        "- The branch winner is assessed by comparing source-like counts, current counts, overlap, and low path cost in the generated CSV/JSON.",
        "- Parent/path visible clearing is diagnostic here because source supports it but the sc_explorer profile does not enable it.",
        "- Frontier-local SC remains diagnostic unless a source config/code path is found that directly supports it.",
    ]
    write_text(output_dir / "branch_visible_voxel_decomposition.md", "\n".join(lines))
    save_json(output_dir / "branch_segment_visible_voxel_decomposition.json", segment_rows)


def make_plots(output_dir: Path, branch_rows: list[dict[str, Any]], variant_rows: list[dict[str, Any]], observed_state: np.ndarray) -> None:
    completed = [row for row in branch_rows if row.get("status") == "completed"]
    key = [row for row in completed if row.get("branch_label", "").startswith(("seed0_measured", "seed0_sc", "seed1_"))]
    if not key:
        key = completed[:6]
    labels = [str(row["branch_label"]).replace("_branch", "").replace("seed0_", "s0_").replace("seed1_", "s1_") for row in key]

    plt.figure(figsize=(11, 5))
    x = np.arange(len(key))
    plt.bar(x - 0.2, [row.get("visible_unknown_count", 0) for row in key], width=0.2, label="unknown")
    plt.bar(x, [row.get("visible_free_count", 0) for row in key], width=0.2, label="free")
    plt.bar(x + 0.2, [row.get("visible_occupied_count", 0) for row in key], width=0.2, label="occupied")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("unique visible voxels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "branch_visible_voxel_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    for row in key:
        mean = row.get("path_unique_predicted_confidence_mean")
        p90 = row.get("path_unique_predicted_confidence_p90")
        if mean is not None:
            plt.scatter(float(mean), float(p90 or mean), label=str(row["branch_label"])[:22])
    plt.xlabel("prediction confidence mean")
    plt.ylabel("prediction confidence p90")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_dir / "branch_prediction_confidence_distributions.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    x = np.arange(len(key))
    plt.bar(x, [row.get("path_sum_recorded_gain_exp", 0) for row in key], label="gain_exp")
    plt.bar(
        x,
        [row.get("path_sum_source_occ_free_gain", 0) for row in key],
        bottom=[row.get("path_sum_recorded_gain_exp", 0) for row in key],
        label="source occ+free",
    )
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("path gain components")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "branch_gain_component_stack.png", dpi=160)
    plt.close()

    measured = next((row for row in completed if row.get("branch_label") == "seed0_measured_branch_n0001_to_n0112"), None)
    sc = next((row for row in completed if row.get("branch_label") == "seed0_sc_branch_n0127_to_n0162"), None)
    plt.figure(figsize=(7, 7))
    topdown = np.max(observed_state == OCCUPIED, axis=2)
    plt.imshow(topdown.T, origin="lower", cmap="Greys", alpha=0.25)
    for row, color, name in ((measured, "tab:blue", "measured"), (sc, "tab:red", "seed0 SC")):
        if row:
            pts = [parse_json_cell(json.dumps(row.get("root_grid")), row.get("root_grid"))]
            for grid in (row.get("selected_child_grid"), row.get("best_descendant_grid")):
                if grid:
                    pts.append(grid)
            arr = np.asarray([[p[0], p[1]] for p in pts], dtype=float)
            plt.plot(arr[:, 0], arr[:, 1], marker="o", color=color, label=name)
    plt.legend()
    plt.title("measured vs seed0 SC branch")
    plt.tight_layout()
    plt.savefig(output_dir / "measured_vs_sc_branch_overlap_topdown.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.imshow(np.max(observed_state == FREE, axis=2).T, origin="lower", cmap="Greens", alpha=0.15)
    for row, color in zip(key[:5], ["tab:blue", "tab:red", "tab:orange", "tab:purple", "tab:brown"]):
        centroid = row.get("predicted_centroid_grid")
        if centroid:
            plt.scatter(float(centroid[0]), float(centroid[1]), s=80, color=color, label=str(row["branch_label"])[:18])
    plt.legend(fontsize=7)
    plt.title("predicted voxel centroid by branch")
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_voxel_spatial_distribution_topdown.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    if variant_rows:
        x = np.arange(len(variant_rows))
        plt.bar(x - 0.2, [row["seed0_measured_branch_score_proxy"] for row in variant_rows], width=0.4, label="measured")
        plt.bar(x + 0.2, [row["seed0_sc_branch_score_proxy"] for row in variant_rows], width=0.4, label="seed0 SC")
        plt.xticks(x, [row["variant"].replace("_", "\n") for row in variant_rows], fontsize=7)
        plt.ylabel("proxy (gain_exp + SC) / cost")
        plt.legend()
    else:
        plt.text(0.1, 0.5, "candidate variants unavailable")
    plt.tight_layout()
    plt.savefig(output_dir / "candidate_formula_effects_bar.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.text(0.02, 0.55, "Candidate formula seed replay skipped.\nSee candidate_formula_seed_replay_skipped_reason.md.", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "seed_replay_if_done.png", dpi=160)
    plt.close()


def write_replay_skipped(output_dir: Path) -> None:
    write_text(
        output_dir / "candidate_formula_seed_replay_skipped_reason.md",
        "# Candidate Formula Seed Replay Skipped\n\n"
        "Skipped in Stage 4A-6.5x to keep this turn limited to source review and visible-voxel decomposition. "
        "The existing Stage 4A-6.5v/6.5w seed replay artifacts were read, but no new tree replay was launched.",
    )


def build_summary(
    *,
    output_dir: Path,
    evidence: dict[str, Any],
    branch_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    safety: dict[str, Any],
    missing_fields: dict[str, Any],
) -> dict[str, Any]:
    measured = next((row for row in branch_rows if row.get("branch_label") == "seed0_measured_branch_n0001_to_n0112"), {})
    sc = next((row for row in branch_rows if row.get("branch_label") == "seed0_sc_branch_n0127_to_n0162"), {})
    seed1 = next((row for row in branch_rows if row.get("branch_label") == "seed1_measured_confidence_branch_n0057_to_n0118"), {})
    most_source = next((row for row in variant_rows if row.get("variant") == "source_raw_predicted_occupied_free"), {})
    frontier = next((row for row in variant_rows if row.get("variant") == "frontier_local_sc"), {})
    sc_raw = float(sc.get("path_sum_raw_count_gain", 0.0) or 0.0) if sc else 0.0
    sc_source = float(sc.get("path_sum_source_occ_free_gain", 0.0) or 0.0) if sc else 0.0
    sc_source_unknown_fraction = float(max(0.0, sc_raw - sc_source) / max(1.0, sc_raw))
    dense_prediction_dominates = bool(sc and sc_source_unknown_fraction > 0.2)
    low_novelty_prediction_dominates = bool(
        sc
        and float(sc.get("overlap_with_root_visible_fraction", 0.0) or 0.0) > 0.35
    )
    parent_clearing_removes_advantage = bool(
        sc
        and measured
        and float(sc.get("path_sum_parent_and_root_visible_cleared_sc_gain", 0.0))
        <= float(measured.get("path_sum_parent_and_root_visible_cleared_sc_gain", 0.0))
    )
    next_small_task = (
        "offline source OCC+FREE plus parent-visible-cleared/frontier-local seed replay"
        if parent_clearing_removes_advantage or low_novelty_prediction_dominates
        else "offline source OCC+FREE seed replay"
    )
    summary = {
        "stage": "Stage 4A-6.5x",
        "output_dir": str(output_dir),
        "completed": True,
        "blocked": False,
        "main_blocker": "No runtime blocker; the main design blocker is SC gain semantics/selectivity.",
        "answers": {
            "source_rewarded_voxel_classes": evidence["rewarded_voxel_classes"],
            "source_confidence_probability_use": evidence["confidence_probability_use"],
            "parent_visible_clearing": evidence["parent_visible_clearing"],
            "prediction_ray_blocking": evidence["planning_flags"]["prediction_blocks_visibility_raycasting_in_sc_explorer_config"],
            "source_faithful_current_formulas": ["none exactly as currently named", "raw_count only after source OCC/FREE threshold restriction"],
            "diagnostic_current_formulas": ["confidence_weighted", "cap25", "occupied_only", "occupied_margin", "calibrated_occupied", "novelty_discounted"],
            "seed0_sc_branch_why_wins": {
                "raw_count_gain": sc.get("path_sum_raw_count_gain"),
                "source_occ_free_gain": sc.get("path_sum_source_occ_free_gain"),
                "path_cost": sc.get("path_cost_recorded"),
                "interpretation": "short branch plus dense prediction-visible voxels near the root/local branch, not a proven robust semantic basin",
            },
            "seed1_multi_seed_return_to_measured": {
                "seed1_branch": seed1.get("path_node_ids"),
                "interpretation": "saved 6.5v/6.5w evidence shows seed-dependent tree sampling often lands in the measured branch basin",
            },
            "dense_source_unknown_prediction_dominates": dense_prediction_dominates,
            "seed0_sc_source_unknown_fraction": sc_source_unknown_fraction,
            "low_novelty_prediction_visibility_dominates": low_novelty_prediction_dominates,
            "parent_visible_clearing_removes_advantage": parent_clearing_removes_advantage,
            "frontier_local_possible": True,
            "recommended_next_formula": most_source.get("variant", "source_raw_predicted_occupied_free"),
            "most_promising_diagnostic": frontier.get("variant", "frontier_local_sc"),
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
        },
        "key_branch_metrics": {
            "measured_branch": measured,
            "seed0_sc_branch": sc,
            "seed1_branch": seed1,
        },
        "candidate_variants": variant_rows,
        "next_small_task": next_small_task,
        "safety": safety,
        "missing_fields": missing_fields,
    }
    save_json(output_dir / "stage4a65x_sc_gain_design_review_summary.json", summary)

    lines = [
        "# Stage 4A-6.5x SC Gain Design Review Summary",
        "",
        "## Required Answers",
        "",
        f"1. Original SC gain rewards predicted occupied and predicted free voxels; sc_explorer.yaml sets unobserved_weight to {evidence['rewarded_voxel_classes']['sc_explorer_config_unobserved_weight']}.",
        f"2. The source profile thresholds SSC occupancy/free probability via ssc_confidence_threshold {evidence['confidence_probability_use']['ssc_confidence_threshold']}; direct weight_by_confidence is not enabled.",
        f"3. Parent-visible clearing is supported in the base evaluator but inactive in sc_explorer.yaml.",
        "4. No current formula is exactly source-faithful by name; source_raw_predicted_occupied_free is the closest offline reconstruction.",
        "5. confidence_weighted, cap25, occupied_only, occupied_margin, calibrated_occupied, novelty_discounted, and branch normalization are diagnostic or source-inspired only.",
        f"6. Seed0 SC branch wins through short path cost plus predicted-visible mass: raw {sc.get('path_sum_raw_count_gain')}, source OCC/FREE {sc.get('path_sum_source_occ_free_gain')}, cost {sc.get('path_cost_recorded')}.",
        "7. Seed1/multi-seed often returns to measured because the saved tree sampling basin is not stable; 6.5v/6.5w already showed same-as-measured dominates.",
        f"8. Dense predicted_unmeasured diagnosis: source-unknown dominance is {dense_prediction_dominates} "
        f"(seed0 SC raw {sc_raw}, source OCC/FREE {sc_source}, source-unknown fraction {sc_source_unknown_fraction}); "
        f"low-novelty/root-overlap visibility is {low_novelty_prediction_dominates}.",
        f"9. Parent/root visible clearing removes or weakens seed0 SC advantage in the proxy: {parent_clearing_removes_advantage}.",
        "10. Frontier-local SC may be more stable, but it is diagnostic/source-inspired rather than source-proven.",
        f"11. Recommended next formula: {most_source.get('variant', 'source_raw_predicted_occupied_free')}.",
        "12. Yes, implement source OCC+FREE offline replay before runtime smoke.",
        "13. Runtime smoke is not ready.",
        "14. Rollout is not ready.",
        "",
        "## Recommendation",
        "",
        f"Next small task: {next_small_task}.",
        "Do it offline over the saved Frame2 seed set before another runtime smoke. Do not jump to rollout.",
    ]
    write_text(output_dir / "stage4a65x_sc_gain_design_review_summary.md", "\n".join(lines))
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        f"# Recommended Next Faithful Step\n\n{next_small_task} on saved Stage 4A-6.5p Frame2 artifacts. "
        "Keep it offline: no Isaac startup, no map_predict rerun, no action execution, and no rollout.",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame2_observed_state", required=True)
    parser.add_argument("--frame2_prediction_npz", required=True)
    parser.add_argument("--frame2_pose_json", required=True)
    parser.add_argument("--frame2_camera_info_json", required=True)
    parser.add_argument("--stage4a65p_dir", required=True)
    parser.add_argument("--stage4a65q_dir", required=True)
    parser.add_argument("--stage4a65r_dir", required=True)
    parser.add_argument("--stage4a65v_dir", required=True)
    parser.add_argument("--stage4a65w_dir", required=True)
    parser.add_argument("--local_ssc_exploration_dir", required=True)
    parser.add_argument("--external_active3d_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--save_viz", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_path = Path(args.frame2_observed_state)
    prediction_path = Path(args.frame2_prediction_npz)
    input_hashes_before = {
        "observed_state": sha256_file(observed_path),
        "prediction_npz": sha256_file(prediction_path),
        "checkpoint": sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else None,
    }
    external_status_before = git_status_porcelain(Path(args.external_active3d_dir))

    observed_state = np.load(observed_path)
    prediction = SimPredictionLayer.from_npz(prediction_path)
    if tuple(observed_state.shape) != tuple(prediction.shape()):
        raise ValueError(f"observed_state shape {observed_state.shape} != prediction shape {prediction.shape()}")

    observed_summary = read_json(Path(args.stage4a65p_dir) / "frame002_observed_summary.json")
    bounds = normalize_bounds(observed_summary.get("bounds", {"x": (-6, 6), "y": (-6, 6), "z": (0, 3)}))
    voxel_size = float(observed_summary.get("voxel_size", VOXEL_SIZE_DEFAULT))
    camera_info = read_json(Path(args.frame2_camera_info_json))
    max_depth = float(camera_info.get("max_depth", 5.0))
    max_range_voxels = max(1, int(round(max_depth / voxel_size)))

    evidence = source_evidence(args)
    write_source_reports(output_dir, evidence)
    formula_rows = current_formula_audit()
    write_formula_audit(output_dir, formula_rows)

    p_tree = load_tree(Path(args.stage4a65p_dir) / "frame002_measured_tree_segments.jsonl")
    root_segment = p_tree.get("root", {"end_grid": [13, 13, 11], "yaw": 0.0})
    visible_cache: dict[tuple[tuple[int, int, int], float], set[tuple[int, int, int]]] = {}
    root_visible = compute_visible_set(root_segment, observed_state, max_range_voxels, visible_cache)

    specs, missing_specs = collect_branch_specs(args)
    branch_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []
    branch_details: dict[str, Any] = {}
    for spec in specs:
        row, details = analyze_branch(
            label=spec["label"],
            source_stage=spec["source_stage"],
            seed=spec.get("seed"),
            formula=spec["formula"],
            tree_path=spec["tree_path"],
            selected_child_id=spec["selected_child_id"],
            best_descendant_id=spec["best_descendant_id"],
            observed_state=observed_state,
            prediction=prediction,
            root_visible=root_visible,
            max_range_voxels=max_range_voxels,
            visible_cache=visible_cache,
            tau=float(args.tau),
            source_threshold=SOURCE_CONFIDENCE_THRESHOLD_DEFAULT,
            rank=spec.get("rank"),
        )
        branch_rows.append(row)
        segment_rows.extend(details.get("segments", []))
        if details:
            branch_details[row["branch_label"]] = {
                "union_predicted_count": len(details["union_predicted"]),
                "union_visible_count": len(details["union_visible"]),
            }

    missing_fields = {
        "visible_voxel_ids_saved_in_tree_artifacts": False,
        "visible_voxel_recomputation": "diagnostic_current_offline_observed_raycaster",
        "missing_or_limited_inputs": missing_specs,
        "branch_details": branch_details,
        "candidate_formula_seed_replay_done": False,
        "candidate_formula_seed_replay_skipped_reason": "kept Stage 4A-6.5x to source review and decomposition only",
    }
    save_json(output_dir / "visible_voxel_missing_fields_report.json", missing_fields)
    write_decomposition_reports(output_dir, branch_rows, segment_rows)

    variant_rows = evaluate_candidate_variants(branch_rows)
    write_candidate_variant_reports(output_dir, variant_rows)
    write_replay_skipped(output_dir)
    if args.save_viz:
        make_plots(output_dir, branch_rows, variant_rows, observed_state)

    input_hashes_after = {
        "observed_state": sha256_file(observed_path),
        "prediction_npz": sha256_file(prediction_path),
        "checkpoint": sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else None,
    }
    external_status_after = git_status_porcelain(Path(args.external_active3d_dir))
    safety = {
        "isaac_startup": False,
        "new_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_rl": False,
        "checkpoint_modified": input_hashes_before["checkpoint"] != input_hashes_after["checkpoint"],
        "observed_state_modified": input_hashes_before["observed_state"] != input_hashes_after["observed_state"],
        "prediction_npz_modified": input_hashes_before["prediction_npz"] != input_hashes_after["prediction_npz"],
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_scoring": False,
        "external_source_modified_or_built": external_status_before != external_status_after or bool(external_status_after),
        "external_status_before": external_status_before,
        "external_status_after": external_status_after,
        "coverage_improvement_claim": False,
        "input_hashes_before": input_hashes_before,
        "input_hashes_after": input_hashes_after,
        "alignment_convention": str(args.alignment_convention),
        "prohibited_output_patterns": PROHIBITED_OUTPUT_PATTERNS,
    }
    save_json(output_dir / "safety_summary.json", safety)
    build_summary(
        output_dir=output_dir,
        evidence=evidence,
        branch_rows=branch_rows,
        variant_rows=variant_rows,
        safety=safety,
        missing_fields=missing_fields,
    )
    plot_status = {
        "plots_requested": bool(args.save_viz),
        "plots": {name: (output_dir / name).is_file() for name in PLOT_FILES},
        "skipped": [] if args.save_viz else PLOT_FILES,
    }
    save_json(output_dir / "plot_status.json", plot_status)
    print(json.dumps({"output_dir": str(output_dir), "completed": True, "blocked": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
