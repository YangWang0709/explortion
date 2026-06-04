#!/usr/bin/env python3
"""Stage 4A-6.5y offline source-gain seed replay.

This runner is intentionally offline. It reads the saved Frame 2 observed map
and saved prediction NPZ, replays current mini-RRT formulas, and post-hoc
rescoring variants that count source OCC+FREE prediction gain. It never starts
Isaac, captures frames, reruns map_predict/SSCNet, executes actions, trains, or
writes prediction into observed_state.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import ROOT_ID, run as run_mini_rrt, sha256_file, to_jsonable
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


DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65y_source_gain_seed_replay"
)
DEFAULT_SELECTED_CASE = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65c_decoupled_one_step_smoke/selected_case.json"
)
DEFAULT_EPISODE_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_medium_rollout_sc_pred_alignment_fixed_smoke/episodes/"
    "medium_three_rooms_seed0_start_room_a_sc_pred_alignment_fixed_000"
)
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)

REFERENCE_SC_SELECTED_GRID = [11, 15, 11]
REFERENCE_SC_BEST_GRID = [14, 15, 11]
REFERENCE_MEASURED_SELECTED_GRID = [17, 16, 11]
REFERENCE_MEASURED_BEST_GRID = [8, 27, 11]
SEED0_SC_SOURCE_OCC_FREE_REFERENCE = 135.0
SEED0_SC_GAIN_EXP_REFERENCE = 76.0
SEED0_SC_COST_REFERENCE = 0.5872281406276059

CURRENT_FORMULA_TO_MINI = {
    "measured_only": "measured_only",
    "current_confidence_weighted": "confidence_weighted",
    "current_cap25": "cap25",
    "current_raw_count": "raw_count",
}

SOURCE_FORMULAS = {
    "source_occ_free",
    "source_occ_free_thresholded",
    "parent_visible_cleared_source_occ_free",
    "root_visible_cleared_source_occ_free",
    "frontier_local_source_occ_free",
    "parent_cleared_frontier_local_source_occ_free",
    "branch_normalized_source_occ_free",
}

FORMULA_SOURCE_STATUS = {
    "measured_only": {
        "source_faithfulness": "measured-only-baseline",
        "description": "No prediction gain; gain is measured exploration gain only.",
    },
    "current_confidence_weighted": {
        "source_faithfulness": "diagnostic-only",
        "description": "Current Python confidence/margin-weighted diagnostic baseline; not active source-faithful.",
    },
    "current_cap25": {
        "source_faithfulness": "diagnostic-only",
        "description": "Current Python capped raw-count diagnostic baseline; not active source-faithful.",
    },
    "current_raw_count": {
        "source_faithfulness": "source-inspired",
        "description": "Counts prediction-valid unmeasured voxels; source-like only after OCC/FREE restriction.",
    },
    "source_occ_free": {
        "source_faithfulness": "source-faithful-approx",
        "description": "Counts predicted OCC+FREE unmeasured voxels with active source weights and threshold proxy.",
    },
    "source_occ_free_thresholded": {
        "source_faithfulness": "source-faithful-approx",
        "description": "Same OCC+FREE count with explicit probability threshold mapping from simulator NPZ.",
    },
    "parent_visible_cleared_source_occ_free": {
        "source_faithfulness": "source-inspired",
        "description": "OCC+FREE with root/path-visible clearing; supported by source base evaluator but inactive profile.",
    },
    "root_visible_cleared_source_occ_free": {
        "source_faithfulness": "diagnostic-only",
        "description": "OCC+FREE after removing voxels visible from the root only.",
    },
    "frontier_local_source_occ_free": {
        "source_faithfulness": "source-inspired-diagnostic",
        "description": "OCC+FREE restricted to measured-frontier-local prediction voxels.",
    },
    "parent_cleared_frontier_local_source_occ_free": {
        "source_faithfulness": "diagnostic-only",
        "description": "Combines parent/root visible clearing with frontier-local filtering.",
    },
    "branch_normalized_source_occ_free": {
        "source_faithfulness": "diagnostic-only",
        "description": "Normalizes source OCC+FREE by visible volume; not active source-faithful.",
    },
}

REQUIRED_FILES = [
    "prediction_npz_field_inventory.json",
    "prediction_npz_field_inventory.md",
    "source_occ_free_mapping_report.json",
    "source_occ_free_mapping_report.md",
    "source_gain_replay_manifest.jsonl",
    "per_seed_formula_decisions.csv",
    "per_seed_formula_decisions.json",
    "per_seed_formula_decisions.md",
    "per_seed_formula_gain_components.csv",
    "per_seed_formula_gain_components.json",
    "branch_classification_by_formula_seed.csv",
    "branch_classification_by_formula_seed.json",
    "branch_classification_summary_by_formula.json",
    "branch_classification_summary_by_formula.md",
    "margin_summary_by_formula.csv",
    "margin_summary_by_formula.json",
    "margin_summary_by_formula.md",
    "overlap_novelty_summary_by_formula.csv",
    "overlap_novelty_summary_by_formula.json",
    "overlap_novelty_summary_by_formula.md",
    "frontier_local_summary_by_formula.csv",
    "frontier_local_summary_by_formula.json",
    "frontier_local_summary_by_formula.md",
    "formula_source_faithfulness_table.csv",
    "formula_source_faithfulness_table.json",
    "formula_source_faithfulness_table.md",
    "missing_fields_report.json",
    "stage4a65y_source_gain_seed_replay_summary.json",
    "stage4a65y_source_gain_seed_replay_summary.md",
    "recommended_next_faithful_step.md",
]

REQUIRED_PLOTS = [
    "formula_branch_classification_bar.png",
    "formula_same_as_measured_fraction.png",
    "formula_seed0_sc_basin_fraction.png",
    "formula_avoids_short_local_sc_fraction.png",
    "winner_source_occ_free_by_formula.png",
    "winner_cost_by_formula.png",
    "winner_gain_cost_stack_by_formula.png",
    "root_overlap_fraction_by_formula.png",
    "frontier_local_fraction_by_formula.png",
    "selected_children_by_formula_topdown.png",
    "best_descendants_by_formula_topdown.png",
    "value_vs_source_occ_free_by_formula.png",
    "value_vs_cost_by_formula.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "sscnet_*",
    "map_predict*",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_topdown_path.png",
    "rollout_*.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_literal_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        import ast

        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return None
    return list(parsed) if isinstance(parsed, (list, tuple)) else None


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1.0e-9:
        return None
    return float(numerator) / float(denominator)


def euclidean(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        aa = [float(v) for v in a]
        bb = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    dims = min(len(aa), len(bb))
    if dims == 0:
        return None
    return float(math.sqrt(sum((aa[idx] - bb[idx]) ** 2 for idx in range(dims))))


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def percentile_summary(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def min_mean_max(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(clean), "mean": statistics.fmean(clean), "max": max(clean)}


def pearson_pairs(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= 1.0e-9 or float(np.std(y_arr)) <= 1.0e-9:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def fraction(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(1 for row in rows if bool(row.get(key)))) / float(len(rows))


def default_bounds(shape: tuple[int, int, int], voxel_size: float) -> dict[str, tuple[float, float]]:
    return normalize_bounds(
        {
            "x": (-0.5 * shape[0] * voxel_size, 0.5 * shape[0] * voxel_size),
            "y": (-0.5 * shape[1] * voxel_size, 0.5 * shape[1] * voxel_size),
            "z": (0.0, shape[2] * voxel_size),
        }
    )


def reference_worlds(observed_state: np.ndarray, voxel_size: float) -> dict[str, Any]:
    bounds = default_bounds(tuple(int(v) for v in observed_state.shape), voxel_size)
    return {
        "seed0_sc_selected_grid": REFERENCE_SC_SELECTED_GRID,
        "seed0_sc_best_grid": REFERENCE_SC_BEST_GRID,
        "seed0_measured_selected_grid": REFERENCE_MEASURED_SELECTED_GRID,
        "seed0_measured_best_grid": REFERENCE_MEASURED_BEST_GRID,
        "seed0_sc_selected_world": list(grid_to_world(REFERENCE_SC_SELECTED_GRID, bounds, voxel_size)),
        "seed0_sc_best_world": list(grid_to_world(REFERENCE_SC_BEST_GRID, bounds, voxel_size)),
        "seed0_measured_selected_world": list(grid_to_world(REFERENCE_MEASURED_SELECTED_GRID, bounds, voxel_size)),
        "seed0_measured_best_world": list(grid_to_world(REFERENCE_MEASURED_BEST_GRID, bounds, voxel_size)),
    }


def has_free_neighbor(observed_state: np.ndarray, voxel: tuple[int, int, int]) -> bool:
    shape = observed_state.shape
    i, j, k = voxel
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        ni, nj, nk = i + di, j + dj, k + dk
        if 0 <= ni < shape[0] and 0 <= nj < shape[1] and 0 <= nk < shape[2]:
            if observed_state[ni, nj, nk] == FREE:
                return True
    return False


def parse_tree_table(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "depth",
        "gain",
        "gain_exp",
        "gain_sc",
        "gain_hybrid",
        "effective_gain_sc",
        "gain_hybrid_effective",
        "gain_occ",
        "gain_conf",
        "cost",
        "accumulated_gain",
        "accumulated_cost",
        "value",
        "children_count",
        "visible_count",
        "frontier_count_visible",
    }
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        row: dict[str, Any] = dict(raw)
        for key in ("end_grid", "end_world", "start_grid", "start_world"):
            if key in row:
                row[key] = parse_literal_list(row.get(key))
        for key in numeric:
            if key in row:
                row[key] = as_float(row.get(key), float("nan"))
        rows.append(row)
    return rows


def table_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("segment_id")): row for row in rows if row.get("segment_id") is not None}


def load_tree_segments(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(row["segment_id"]): row for row in rows if row.get("segment_id") is not None}


def path_to_root(tree: dict[str, dict[str, Any]], node_id: str | None) -> list[str]:
    if not node_id or node_id not in tree:
        return []
    path: list[str] = []
    seen: set[str] = set()
    current: str | None = str(node_id)
    while current and current in tree and current not in seen:
        seen.add(current)
        path.append(current)
        parent = tree[current].get("parent_id")
        current = str(parent) if parent else None
    path.reverse()
    return path


def root_child_for_path(path_ids: list[str]) -> str | None:
    for node_id in path_ids:
        if node_id != ROOT_ID:
            return node_id
    return None


def root_children(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    children = [row for row in rows if str(row.get("parent_id")) == ROOT_ID]
    children.sort(key=lambda row: as_float(row.get("value"), float("-inf")), reverse=True)
    return children


def margin_from_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    children = root_children(rows)
    winner = children[0] if children else None
    runner = children[1] if len(children) > 1 else None
    winner_value = as_float(winner.get("value"), float("nan")) if winner else None
    runner_value = as_float(runner.get("value"), float("nan")) if runner else None
    margin = None
    normalized = None
    if winner_value is not None and runner_value is not None:
        margin = float(winner_value - runner_value)
        normalized = safe_ratio(margin, abs(float(winner_value)))
    return {
        "winner_id": winner.get("segment_id") if winner else None,
        "winner_best_descendant_id": winner.get("best_descendant_id") if winner else None,
        "winner_value": winner_value,
        "runner_up_id": runner.get("segment_id") if runner else None,
        "runner_up_best_descendant_id": runner.get("best_descendant_id") if runner else None,
        "runner_up_value": runner_value,
        "winner_margin": margin,
        "normalized_margin": normalized,
        "root_child_count": len(children),
    }


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


def compute_visible_set(
    segment: dict[str, Any],
    observed_state: np.ndarray,
    max_range_voxels: int,
    num_yaw: int,
    num_pitch: int,
    cache: dict[tuple[tuple[int, int, int], float, int, int], set[tuple[int, int, int]]],
) -> set[tuple[int, int, int]]:
    grid = tuple(int(v) for v in segment.get("end_grid") or segment.get("grid") or [0, 0, 0])
    yaw = float(segment.get("yaw", 0.0))
    key = (grid, round(yaw, 6), int(num_yaw), int(num_pitch))
    if key in cache:
        return set(cache[key])
    candidate = SimCandidateView(
        id=-1,
        grid_position=grid,
        world_position=(0.0, 0.0, 0.0),
        yaw=yaw,
        valid=True,
        candidate_source="stage4a65y_source_gain_replay",
    )
    visible = set(
        tuple(int(v) for v in voxel)
        for voxel in raycast_visible_voxels_observed(
            candidate,
            observed_state,
            max_range_voxels=max_range_voxels,
            num_yaw=max(4, int(num_yaw)),
            num_pitch=max(3, int(num_pitch)),
        )
    )
    cache[key] = set(visible)
    return visible


def prediction_masks_for_visible(
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    visible: set[tuple[int, int, int]],
    source_threshold: float,
) -> dict[str, set[tuple[int, int, int]]]:
    occ_threshold = 0.5 + float(source_threshold)
    free_threshold = 0.5 + float(source_threshold)
    predicted: set[tuple[int, int, int]] = set()
    occ: set[tuple[int, int, int]] = set()
    free: set[tuple[int, int, int]] = set()
    source_unknown: set[tuple[int, int, int]] = set()
    for voxel in visible:
        if observed_state[voxel] != UNKNOWN:
            continue
        if not bool(prediction.valid[voxel]):
            continue
        if float(prediction.confidence[voxel]) < float(source_threshold):
            continue
        predicted.add(voxel)
        occ_prob = float(prediction.occupied_prob[voxel])
        free_prob = float(prediction.free_prob[voxel])
        if occ_prob >= occ_threshold:
            occ.add(voxel)
        elif free_prob >= free_threshold:
            free.add(voxel)
        else:
            source_unknown.add(voxel)
    return {"predicted": predicted, "occupied": occ, "free": free, "source_unknown": source_unknown}


def confidence_stats(prediction: SimPredictionLayer, voxels: set[tuple[int, int, int]], prefix: str) -> dict[str, Any]:
    if not voxels:
        return {
            f"{prefix}_sum": 0.0,
            f"{prefix}_mean": None,
            f"{prefix}_p90": None,
        }
    values = np.asarray([float(prediction.confidence[v]) for v in voxels], dtype=np.float64)
    return {
        f"{prefix}_sum": float(np.sum(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def source_component_for_segment(
    segment: dict[str, Any],
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    root_visible: set[tuple[int, int, int]],
    max_range_voxels: int,
    num_yaw: int,
    num_pitch: int,
    source_threshold: float,
    visible_cache: dict[tuple[tuple[int, int, int], float, int, int], set[tuple[int, int, int]]],
) -> dict[str, Any]:
    visible = compute_visible_set(segment, observed_state, max_range_voxels, num_yaw, num_pitch, visible_cache)
    masks = prediction_masks_for_visible(observed_state, prediction, visible, source_threshold=source_threshold)
    source_set = set(masks["occupied"]) | set(masks["free"])
    frontier_local = {voxel for voxel in source_set if has_free_neighbor(observed_state, voxel)}
    counts = state_counts(observed_state, visible)
    root_overlap = len(visible & root_visible)
    return {
        "visible": visible,
        "source_predicted": set(masks["predicted"]),
        "source_occ": set(masks["occupied"]),
        "source_free": set(masks["free"]),
        "source_unknown": set(masks["source_unknown"]),
        "source_occ_free": source_set,
        "frontier_local": frontier_local,
        "visible_count": int(len(visible)),
        **counts,
        "source_predicted_count": int(len(masks["predicted"])),
        "source_occ_count": int(len(masks["occupied"])),
        "source_free_count": int(len(masks["free"])),
        "source_unknown_count": int(len(masks["source_unknown"])),
        "source_occ_free_count": int(len(source_set)),
        "frontier_local_count": int(len(frontier_local)),
        "root_visible_overlap_count": int(root_overlap),
        "root_visible_overlap_fraction": float(root_overlap / max(1, len(visible))),
    }


def local_formula_sc(formula: str, comp: dict[str, Any], root_source: set[tuple[int, int, int]], previous_source: set[tuple[int, int, int]]) -> float:
    source_set = set(comp["source_occ_free"])
    if formula in {"source_occ_free", "source_occ_free_thresholded"}:
        return float(len(source_set))
    if formula == "parent_visible_cleared_source_occ_free":
        return float(len(source_set - previous_source))
    if formula == "root_visible_cleared_source_occ_free":
        return float(len(source_set - root_source))
    if formula == "frontier_local_source_occ_free":
        return float(len(set(comp["frontier_local"])))
    if formula == "parent_cleared_frontier_local_source_occ_free":
        return float(len((source_set - previous_source) & set(comp["frontier_local"])))
    if formula == "branch_normalized_source_occ_free":
        return float(100.0 * len(source_set) / max(1, int(comp["visible_count"])))
    raise ValueError(f"unsupported source formula: {formula}")


def path_source_summary(
    *,
    path_ids: list[str],
    tree: dict[str, dict[str, Any]],
    node_components: dict[str, dict[str, Any]],
    root_visible: set[tuple[int, int, int]],
    root_source: set[tuple[int, int, int]],
    prediction: SimPredictionLayer,
    formula: str,
) -> dict[str, Any]:
    ids = [node_id for node_id in path_ids if node_id != ROOT_ID and node_id in tree]
    previous_visible = set(root_visible)
    previous_source = set(root_source)
    union_visible: set[tuple[int, int, int]] = set()
    union_source: set[tuple[int, int, int]] = set()
    union_occ: set[tuple[int, int, int]] = set()
    union_free: set[tuple[int, int, int]] = set()
    union_unknown: set[tuple[int, int, int]] = set()
    union_frontier: set[tuple[int, int, int]] = set()
    summed = defaultdict(float)
    parent_path_overlap_sum = 0
    visible_count_sum = 0
    local_formula_values: list[float] = []
    for node_id in ids:
        comp = node_components[node_id]
        source_set = set(comp["source_occ_free"])
        frontier_set = set(comp["frontier_local"])
        local_sc = local_formula_sc(formula, comp, root_source, previous_source)
        local_formula_values.append(float(local_sc))
        summed["source_occ_free"] += float(len(source_set))
        summed["source_occ"] += float(len(comp["source_occ"]))
        summed["source_free"] += float(len(comp["source_free"]))
        summed["source_unknown"] += float(len(comp["source_unknown"]))
        summed["parent_root_cleared_source_occ_free"] += float(len(source_set - previous_source))
        summed["root_cleared_source_occ_free"] += float(len(source_set - root_source))
        summed["frontier_local_source_occ_free"] += float(len(frontier_set))
        summed["parent_cleared_frontier_local_source_occ_free"] += float(len((source_set - previous_source) & frontier_set))
        summed["branch_normalized_source_occ_free"] += float(100.0 * len(source_set) / max(1, int(comp["visible_count"])))
        summed["formula_effective_sc"] += float(local_sc)
        parent_path_overlap_sum += len(set(comp["visible"]) & previous_visible)
        visible_count_sum += int(comp["visible_count"])
        union_visible |= set(comp["visible"])
        union_source |= source_set
        union_occ |= set(comp["source_occ"])
        union_free |= set(comp["source_free"])
        union_unknown |= set(comp["source_unknown"])
        union_frontier |= frontier_set
        previous_visible |= set(comp["visible"])
        previous_source |= source_set
    root_overlap_count = len(union_visible & root_visible)
    result = {
        "accumulated_source_occ_free": float(summed["source_occ_free"]),
        "accumulated_source_occ": float(summed["source_occ"]),
        "accumulated_source_free": float(summed["source_free"]),
        "accumulated_source_unknown": float(summed["source_unknown"]),
        "accumulated_parent_root_cleared_source_occ_free": float(summed["parent_root_cleared_source_occ_free"]),
        "accumulated_root_cleared_source_occ_free": float(summed["root_cleared_source_occ_free"]),
        "accumulated_frontier_local_source_occ_free": float(summed["frontier_local_source_occ_free"]),
        "accumulated_parent_cleared_frontier_local_source_occ_free": float(
            summed["parent_cleared_frontier_local_source_occ_free"]
        ),
        "accumulated_branch_normalized_source_occ_free": float(summed["branch_normalized_source_occ_free"]),
        "accumulated_formula_effective_sc": float(summed["formula_effective_sc"]),
        "path_visible_count_sum": int(visible_count_sum),
        "path_visible_unique_count": int(len(union_visible)),
        "visible_predicted_occ_count": int(len(union_occ)),
        "visible_predicted_free_count": int(len(union_free)),
        "visible_predicted_unknown_count": int(len(union_unknown)),
        "visible_source_occ_free_unique_count": int(len(union_source)),
        "root_visible_overlap_count": int(root_overlap_count),
        "root_visible_overlap_fraction": float(root_overlap_count / max(1, len(union_visible))),
        "parent_path_visible_overlap_count": int(parent_path_overlap_sum),
        "parent_path_visible_overlap_fraction": float(parent_path_overlap_sum / max(1, visible_count_sum)),
        "frontier_local_unique_count": int(len(union_frontier)),
        "frontier_local_fraction": float(len(union_frontier) / max(1, len(union_source))),
        "local_formula_effective_sc_min_mean_max": min_mean_max(local_formula_values),
    }
    result.update(confidence_stats(prediction, union_source, "visible_prediction_confidence"))
    return result


def source_tree_decision(
    *,
    seed: int,
    formula: str,
    tree_dir: Path,
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    args: argparse.Namespace,
    root_visible: set[tuple[int, int, int]],
    root_source: set[tuple[int, int, int]],
    visible_cache: dict[tuple[tuple[int, int, int], float, int, int], set[tuple[int, int, int]]],
    missing_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    tree = load_tree_segments(tree_dir / "mini_rrt_tree_segments.jsonl")
    table_rows = parse_tree_table(tree_dir / "gain_cost_value_table.csv")
    rows_by_id = table_by_id(table_rows)
    if not tree or not table_rows:
        raise FileNotFoundError(f"missing base tree artifacts for {seed} {formula}: {tree_dir}")

    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / float(args.voxel_size))))
    num_yaw = max(4, int(math.ceil(32 / max(1, int(args.raycast_stride)))))
    num_pitch = max(3, int(math.ceil(7 / max(1, int(args.raycast_stride)))))
    node_components: dict[str, dict[str, Any]] = {}
    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        node_components[node_id] = source_component_for_segment(
            segment,
            observed_state,
            prediction,
            root_visible,
            max_range_voxels,
            num_yaw,
            num_pitch,
            float(args.ssc_confidence_threshold),
            visible_cache,
        )

    candidate_rows: list[dict[str, Any]] = []
    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        path_ids = path_to_root(tree, node_id)
        no_root_path = [item for item in path_ids if item != ROOT_ID]
        source_summary = path_source_summary(
            path_ids=path_ids,
            tree=tree,
            node_components=node_components,
            root_visible=root_visible,
            root_source=root_source,
            prediction=prediction,
            formula=formula,
        )
        gain_exp = float(sum(as_float(tree[path_id].get("gain_exp")) for path_id in no_root_path if path_id in tree))
        raw_sc = float(sum(as_float(tree[path_id].get("gain_sc")) for path_id in no_root_path if path_id in tree))
        cost = float(sum(as_float(tree[path_id].get("cost")) for path_id in no_root_path if path_id in tree))
        effective_sc = float(source_summary["accumulated_formula_effective_sc"])
        hybrid = float(gain_exp + effective_sc)
        value = safe_ratio(hybrid, cost) if cost > 1.0e-9 else None
        selected_child_id = root_child_for_path(path_ids)
        candidate_rows.append(
            {
                "node_id": node_id,
                "selected_child_id": selected_child_id,
                "path_node_ids": no_root_path,
                "branch_depth": len(no_root_path),
                "accumulated_gain_exp": gain_exp,
                "accumulated_current_raw_gain_sc": raw_sc,
                "accumulated_formula_effective_sc": effective_sc,
                "accumulated_hybrid_effective": hybrid,
                "accumulated_cost": cost,
                "value": value,
                **source_summary,
            }
        )

    if not candidate_rows:
        raise RuntimeError(f"no source candidate rows for seed={seed}, formula={formula}")

    best_by_child: dict[str, dict[str, Any]] = {}
    for row in candidate_rows:
        child_id = row.get("selected_child_id")
        if not child_id:
            continue
        current = best_by_child.get(str(child_id))
        if current is None or as_float(row.get("value"), float("-inf")) > as_float(current.get("value"), float("-inf")):
            best_by_child[str(child_id)] = row

    ranked_children = sorted(best_by_child.values(), key=lambda row: as_float(row.get("value"), float("-inf")), reverse=True)
    winner = ranked_children[0]
    runner = ranked_children[1] if len(ranked_children) > 1 else None
    selected_id = str(winner["selected_child_id"])
    best_id = str(winner["node_id"])
    selected = rows_by_id.get(selected_id) or tree.get(selected_id, {})
    best = rows_by_id.get(best_id) or tree.get(best_id, {})
    root = rows_by_id.get(ROOT_ID) or tree.get(ROOT_ID, {})
    winner_value = as_float(winner.get("value"), float("nan"))
    runner_value = as_float(runner.get("value"), float("nan")) if runner else None
    margin = None if runner_value is None else float(winner_value - runner_value)
    normalized_margin = safe_ratio(margin, abs(winner_value)) if margin is not None else None

    values = [as_float(row.get("value"), float("nan")) for row in candidate_rows]
    costs = [as_float(row.get("accumulated_cost"), float("nan")) for row in candidate_rows]
    inverse_cost = [safe_ratio(1.0, cost) or float("nan") for cost in costs]
    gain_exp_values = [as_float(row.get("accumulated_gain_exp"), float("nan")) for row in candidate_rows]
    formula_sc_values = [as_float(row.get("accumulated_formula_effective_sc"), float("nan")) for row in candidate_rows]
    node_local_sc = [
        local_formula_sc(formula, comp, root_source, root_source)
        for comp in node_components.values()
    ]

    for field, value in {
        "selected_child_grid": selected.get("end_grid"),
        "best_descendant_grid": best.get("end_grid"),
        "root_world": root.get("end_world"),
    }.items():
        if value is None:
            missing_fields.append({"seed": seed, "formula": formula, "field": field, "severity": "derived_missing"})

    return {
        "seed": int(seed),
        "formula": formula,
        "status": "completed_posthoc_source_rescore",
        "gain_mode": "hybrid_posthoc",
        "sc_gain_formula": formula,
        "tree_dir": str(tree_dir),
        "source_tree_basis_formula": "current_raw_count",
        "selected_child_id": selected_id,
        "selected_child_grid": selected.get("end_grid"),
        "selected_child_world": selected.get("end_world"),
        "best_descendant_id": best_id,
        "best_descendant_grid": best.get("end_grid"),
        "best_descendant_world": best.get("end_world"),
        "root_grid": root.get("end_grid"),
        "root_world": root.get("end_world"),
        "selected_child_distance_from_root_m": euclidean(root.get("end_world"), selected.get("end_world")),
        "best_descendant_distance_from_root_m": euclidean(root.get("end_world"), best.get("end_world")),
        "runner_up_id": runner.get("selected_child_id") if runner else None,
        "runner_up_best_descendant_id": runner.get("node_id") if runner else None,
        "runner_up_value": runner_value,
        "winner_margin": margin,
        "normalized_margin": normalized_margin,
        "root_child_count": len(ranked_children),
        "branch_depth": winner["branch_depth"],
        "path_node_ids": winner["path_node_ids"],
        "nodes_accepted": len([node_id for node_id in tree if node_id != ROOT_ID]),
        "nodes_rejected": read_json(tree_dir / "mini_rrt_tree_summary.json").get("tree", {}).get("rejected_samples"),
        "nodes_with_formula_sc_positive": sum(1 for value in node_local_sc if value > 0.0),
        "formula_effective_sc_min_mean_max": min_mean_max(node_local_sc),
        "gain_exp_formula_effective_sc_correlation": pearson_pairs(gain_exp_values, formula_sc_values),
        "value_formula_effective_sc_correlation": pearson_pairs(values, formula_sc_values),
        "value_cost_correlation": pearson_pairs(values, costs),
        "value_inverse_cost_correlation": pearson_pairs(values, inverse_cost),
        **{
            key: value
            for key, value in winner.items()
            if key
            not in {
                "node_id",
                "selected_child_id",
                "path_node_ids",
                "branch_depth",
                "value",
            }
        },
        "value": winner_value,
    }


def actual_tree_decision(
    *,
    seed: int,
    formula: str,
    tree_dir: Path,
    prediction: SimPredictionLayer,
    node_components: dict[str, dict[str, Any]],
    root_visible: set[tuple[int, int, int]],
    root_source: set[tuple[int, int, int]],
    tree_segments: dict[str, dict[str, Any]],
    missing_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    table_rows = parse_tree_table(tree_dir / "gain_cost_value_table.csv")
    rows_by_id = table_by_id(table_rows)
    summary = read_json(tree_dir / "mini_rrt_tree_summary.json")
    decision = summary.get("decision", {})
    selected_payload = decision.get("selected_child") if isinstance(decision.get("selected_child"), dict) else {}
    best_payload = decision.get("best_descendant") if isinstance(decision.get("best_descendant"), dict) else {}
    selected_id = decision.get("selected_child_id") or selected_payload.get("segment_id")
    best_id = decision.get("selected_child_best_descendant_id") or best_payload.get("segment_id") or selected_id
    selected = rows_by_id.get(str(selected_id), {}) if selected_id else {}
    best = rows_by_id.get(str(best_id), {}) if best_id else {}
    root = rows_by_id.get(ROOT_ID, {})
    path_ids = path_to_root(tree_segments, str(best_id) if best_id else None)
    no_root_path = [node_id for node_id in path_ids if node_id != ROOT_ID]
    path_rows = [rows_by_id[node_id] for node_id in no_root_path if node_id in rows_by_id]
    margin = margin_from_table(table_rows)
    non_root = [row for row in table_rows if row.get("segment_id") != ROOT_ID]
    values = [as_float(row.get("value"), float("nan")) for row in non_root]
    costs = [as_float(row.get("accumulated_cost"), float("nan")) for row in non_root]
    inverse_cost = [safe_ratio(1.0, cost) or float("nan") for cost in costs]
    gain_exp_values = [as_float(row.get("gain_exp"), float("nan")) for row in non_root]
    effective_sc_values = [as_float(row.get("effective_gain_sc"), float("nan")) for row in non_root]
    source_summary = path_source_summary(
        path_ids=path_ids,
        tree=tree_segments,
        node_components=node_components,
        root_visible=root_visible,
        root_source=root_source,
        prediction=prediction,
        formula="source_occ_free",
    )

    for field, value in {
        "selected_child_id": selected_id,
        "selected_child_grid": selected.get("end_grid") or selected_payload.get("end_grid"),
        "best_descendant_id": best_id,
        "best_descendant_grid": best.get("end_grid") or best_payload.get("end_grid"),
    }.items():
        if value is None:
            missing_fields.append({"seed": seed, "formula": formula, "field": field, "severity": "derived_missing"})

    accumulated_gain_exp = float(sum(as_float(row.get("gain_exp")) for row in path_rows))
    accumulated_raw_sc = float(sum(as_float(row.get("gain_sc")) for row in path_rows))
    accumulated_effective_sc = 0.0 if formula == "measured_only" else float(
        sum(as_float(row.get("effective_gain_sc")) for row in path_rows)
    )
    return {
        "seed": int(seed),
        "formula": formula,
        "status": "completed",
        "gain_mode": "exp" if formula == "measured_only" else "hybrid",
        "sc_gain_formula": CURRENT_FORMULA_TO_MINI.get(formula, formula),
        "tree_dir": str(tree_dir),
        "source_tree_basis_formula": None,
        "selected_child_id": selected_id,
        "selected_child_grid": selected.get("end_grid") or selected_payload.get("end_grid"),
        "selected_child_world": selected.get("end_world") or selected_payload.get("end_world"),
        "best_descendant_id": best_id,
        "best_descendant_grid": best.get("end_grid") or best_payload.get("end_grid"),
        "best_descendant_world": best.get("end_world") or best_payload.get("end_world"),
        "root_grid": root.get("end_grid") or summary.get("root", {}).get("grid"),
        "root_world": root.get("end_world") or summary.get("root", {}).get("world"),
        "selected_child_distance_from_root_m": euclidean(root.get("end_world"), selected.get("end_world")),
        "best_descendant_distance_from_root_m": euclidean(root.get("end_world"), best.get("end_world")),
        "accumulated_gain_exp": accumulated_gain_exp,
        "accumulated_current_raw_gain_sc": accumulated_raw_sc,
        "accumulated_formula_effective_sc": accumulated_effective_sc,
        "accumulated_hybrid_effective": float(accumulated_gain_exp + accumulated_effective_sc),
        "accumulated_cost": float(sum(as_float(row.get("cost")) for row in path_rows)),
        "value": as_float(selected.get("value"), None),
        "runner_up_value": margin.get("runner_up_value"),
        "winner_margin": margin.get("winner_margin"),
        "normalized_margin": margin.get("normalized_margin"),
        "winner_id": margin.get("winner_id"),
        "runner_up_id": margin.get("runner_up_id"),
        "root_child_count": margin.get("root_child_count"),
        "branch_depth": len(no_root_path),
        "path_node_ids": no_root_path,
        "nodes_accepted": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
        "nodes_rejected": summary.get("tree", {}).get("rejected_samples"),
        "nodes_with_formula_sc_positive": sum(1 for value in effective_sc_values if value > 0.0),
        "formula_effective_sc_min_mean_max": min_mean_max(effective_sc_values),
        "gain_exp_formula_effective_sc_correlation": pearson_pairs(gain_exp_values, effective_sc_values),
        "value_formula_effective_sc_correlation": pearson_pairs(values, effective_sc_values),
        "value_cost_correlation": pearson_pairs(values, costs),
        "value_inverse_cost_correlation": pearson_pairs(values, inverse_cost),
        **{key: value for key, value in source_summary.items() if key != "accumulated_formula_effective_sc"},
    }


def resolve_replay_context(stage4a65v_dir: Path) -> dict[str, str]:
    summary = read_json(stage4a65v_dir / "raw_trees/seed_000/confidence_weighted/mini_rrt_tree_summary.json")
    inputs = summary.get("inputs", {}) if isinstance(summary.get("inputs"), dict) else {}
    case_json = str(inputs.get("case_json") or DEFAULT_SELECTED_CASE)
    episode_dir = str(inputs.get("episode_dir") or DEFAULT_EPISODE_DIR)
    episode_summary = str(inputs.get("episode_summary") or (Path(episode_dir) / "episode_summary.json"))
    return {
        "case_json": case_json if Path(case_json).is_file() else "",
        "episode_dir": episode_dir if Path(episode_dir).is_dir() else "",
        "episode_summary": episode_summary if Path(episode_summary).is_file() else "",
    }


def mini_tree_args(args: argparse.Namespace, seed: int, formula: str, tree_dir: Path, replay_context: dict[str, str]) -> argparse.Namespace:
    mini_formula = CURRENT_FORMULA_TO_MINI[formula]
    measured = mini_formula == "measured_only"
    return argparse.Namespace(
        case_json=replay_context["case_json"],
        episode_dir=replay_context["episode_dir"],
        observed_state=str(Path(args.observed_state).resolve()),
        pose_json=str(Path(args.pose_json).resolve()),
        camera_info=str(Path(args.camera_info_json).resolve()),
        episode_summary=replay_context["episode_summary"],
        prediction_npz="" if measured else str(Path(args.prediction_npz).resolve()),
        output_dir=str(tree_dir),
        seed=int(seed),
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        gain_mode="exp" if measured else "hybrid",
        sc_gain_formula=mini_formula,
        path_cost_mode=str(args.path_cost_mode),
        v_max=float(args.v_max),
        yaw_rate=1.0,
        robot_radius_m=float(args.robot_radius_m),
        voxel_size=float(args.voxel_size),
        raycast_stride=int(args.raycast_stride),
        num_yaw_samples=int(args.num_yaw_samples),
        max_ray_length_m=float(args.max_ray_length_m),
        tau=float(args.tau),
        save_viz=False,
        profile=True,
        min_edge_length_m=0.0,
        min_root_child_length_m=0.0,
        min_root_distance_m=0.0,
        crop_min_length_m=float(args.crop_min_length_m),
        short_edge_policy=str(args.short_edge_policy),
        density_radius_m=0.0,
        max_nodes_per_density_radius=0,
        variant_name=f"stage4a65y_seed{int(seed):03d}_{formula}",
    )


def inventory_prediction_npz(path: Path) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            arr = data[key]
            item: dict[str, Any] = {
                "shape": [int(v) for v in arr.shape],
                "dtype": str(arr.dtype),
                "size": int(arr.size),
            }
            if arr.shape == ():
                item["scalar_value"] = arr.item() if arr.dtype.kind in {"b", "i", "u", "f", "U", "S"} else str(arr.item())
            elif arr.dtype.kind in {"b", "i", "u", "f"} and arr.size:
                item.update(
                    {
                        "min": float(np.min(arr)) if arr.dtype.kind == "f" else int(np.min(arr)),
                        "max": float(np.max(arr)) if arr.dtype.kind == "f" else int(np.max(arr)),
                        "nonzero_count": int(np.count_nonzero(arr)),
                    }
                )
                if key == "global_pred_class":
                    vals, counts = np.unique(arr, return_counts=True)
                    item["unique_counts"] = {
                        str(int(v)): int(c)
                        for v, c in zip(vals.tolist(), counts.tolist())
                    }
            elif arr.dtype.kind in {"U", "S", "O"}:
                sample = arr.reshape(-1)[:5].tolist()
                item["sample_values"] = [str(v) for v in sample]
            fields[key] = item
    return {"path": str(path), "fields": fields}


def write_inventory_md(path: Path, inventory: dict[str, Any]) -> None:
    lines = [
        "# Prediction NPZ Field Inventory",
        "",
        f"- source: `{inventory['path']}`",
        "",
        "| key | shape | dtype | min | max | notes |",
        "|---|---|---|---:|---:|---|",
    ]
    for key, item in inventory["fields"].items():
        notes = ""
        if "unique_counts" in item:
            notes = f"unique_counts={item['unique_counts']}"
        elif "sample_values" in item:
            notes = f"sample={item['sample_values']}"
        lines.append(
            f"| `{key}` | `{item['shape']}` | `{item['dtype']}` | "
            f"{item.get('min', '')} | {item.get('max', '')} | {notes} |"
        )
    write_text(path, "\n".join(lines))


def build_mapping_report(inventory: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    fields = inventory["fields"]
    required = [
        "global_pred_class",
        "global_confidence",
        "global_free_prob",
        "global_occupied_prob",
        "global_prediction_valid",
    ]
    missing = [key for key in required if key not in fields]
    pred_unique = fields.get("global_pred_class", {}).get("unique_counts", {})
    return {
        "mapping_status": "source-faithful-approx" if not missing else "missing-required-fields",
        "required_fields_present": not missing,
        "missing_required_fields": missing,
        "simulator_npz_fields_used": required,
        "source_occ_free_definition": {
            "prediction_set": "global_prediction_valid and observed_state UNKNOWN",
            "occupied_proxy": f"global_occupied_prob >= {0.5 + float(args.ssc_confidence_threshold)}",
            "free_proxy": f"global_free_prob >= {0.5 + float(args.ssc_confidence_threshold)}",
            "weights": {"occupied": 1.0, "free": 1.0, "unobserved_or_unknown": 0.0},
            "confidence_threshold": float(args.ssc_confidence_threshold),
            "weight_by_confidence": False,
            "prediction_ray_blocking": False,
            "prediction_collision_or_traversability": False,
        },
        "pred_class_observation": {
            "unique_counts": pred_unique,
            "interpretation": (
                "class 0 is free-like in this NPZ, class 255 is invalid/unpredicted, and nonzero valid classes are "
                "occupied semantic classes; probability thresholds are used as the replay's OCC/FREE proxy."
            ),
            "class_mapping_exactness": "approximate because the saved simulator NPZ is not the C++ SSCMap log-odds layer.",
        },
        "formula_mapping": {
            "source_occ_free": "same probability-threshold OCC+FREE proxy as source_occ_free_thresholded in this replay.",
            "source_occ_free_thresholded": "explicit OCC/FREE probability threshold proxy.",
            "parent_visible_cleared_source_occ_free": "source-inspired optional clearing; inactive in active sc_explorer.yaml.",
            "frontier_local_source_occ_free": "diagnostic/source-inspired frontier locality filter.",
        },
    }


def write_mapping_md(path: Path, report: dict[str, Any]) -> None:
    definition = report["source_occ_free_definition"]
    lines = [
        "# Source OCC+FREE Mapping Report",
        "",
        f"- mapping status: `{report['mapping_status']}`",
        f"- required fields present: `{report['required_fields_present']}`",
        f"- missing fields: `{report['missing_required_fields']}`",
        f"- prediction set: {definition['prediction_set']}",
        f"- occupied proxy: `{definition['occupied_proxy']}`",
        f"- free proxy: `{definition['free_proxy']}`",
        f"- weights: `{definition['weights']}`",
        f"- confidence threshold: `{definition['confidence_threshold']}`",
        f"- weight by confidence: `{definition['weight_by_confidence']}`",
        f"- prediction ray blocking: `{definition['prediction_ray_blocking']}`",
        "",
        "## Pred Class Observation",
        "",
        f"- unique counts: `{report['pred_class_observation']['unique_counts']}`",
        f"- interpretation: {report['pred_class_observation']['interpretation']}",
        f"- exactness: {report['pred_class_observation']['class_mapping_exactness']}",
    ]
    write_text(path, "\n".join(lines))


def classify_row(row: dict[str, Any], measured: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    selected_to_measured = euclidean(row.get("selected_child_world"), measured.get("selected_child_world"))
    best_to_measured = euclidean(row.get("best_descendant_world"), measured.get("best_descendant_world"))
    selected_to_seed0 = euclidean(row.get("selected_child_world"), refs["seed0_sc_selected_world"])
    best_to_seed0 = euclidean(row.get("best_descendant_world"), refs["seed0_sc_best_world"])
    missing = any(
        row.get(key) is None
        for key in ("selected_child_grid", "selected_child_world", "best_descendant_grid", "best_descendant_world")
    )
    missing = missing or measured.get("selected_child_world") is None
    exact = same_grid(row.get("selected_child_grid"), refs["seed0_sc_selected_grid"]) and same_grid(
        row.get("best_descendant_grid"), refs["seed0_sc_best_grid"]
    )
    spatial = False if missing or selected_to_seed0 is None or best_to_seed0 is None else bool(
        selected_to_seed0 <= 0.25 and best_to_seed0 <= 0.75
    )
    same_as_measured = False if missing or selected_to_measured is None else bool(
        same_grid(row.get("selected_child_grid"), measured.get("selected_child_grid"))
        or selected_to_measured <= 0.15
    )
    measured_basin = bool(same_as_measured and spatial)
    local_jitter = False if missing or selected_to_measured is None else bool(
        not same_as_measured and selected_to_measured < 0.25
    )
    distinct = False if missing or selected_to_measured is None else bool(selected_to_measured >= 0.25 and not spatial)
    avoids_short = bool(
        not spatial
        and (
            as_float(row.get("accumulated_source_occ_free"), 0.0) > SEED0_SC_SOURCE_OCC_FREE_REFERENCE
            or as_float(row.get("accumulated_gain_exp"), 0.0) > SEED0_SC_GAIN_EXP_REFERENCE
        )
        and as_float(row.get("accumulated_cost"), 0.0) > SEED0_SC_COST_REFERENCE
    )
    source_measured = bool(
        same_as_measured
        and as_float(measured.get("accumulated_source_occ_free"), 0.0) >= SEED0_SC_SOURCE_OCC_FREE_REFERENCE
    )
    if missing:
        primary = "unstable_or_missing"
    elif exact:
        primary = "exact_seed0_sc"
    elif measured_basin:
        primary = "measured_but_seed0_sc_basin"
    elif spatial:
        primary = "spatial_seed0_sc_basin"
    elif same_as_measured:
        primary = "same_as_measured_for_seed"
    elif distinct:
        primary = "distinct_sc_branch"
    elif local_jitter:
        primary = "local_jitter"
    else:
        primary = "unstable_or_missing"
    return {
        "seed": row["seed"],
        "formula": row["formula"],
        "primary_classification": primary,
        "exact_seed0_sc": exact,
        "spatial_seed0_sc_basin": spatial,
        "same_as_measured_for_seed": same_as_measured,
        "measured_but_seed0_sc_basin": measured_basin,
        "distinct_sc_branch": distinct,
        "local_jitter": local_jitter,
        "unstable_or_missing": bool(missing),
        "avoids_short_local_sc": avoids_short,
        "source_measured_preferred": source_measured,
        "selected_to_same_seed_measured_m": selected_to_measured,
        "best_to_same_seed_measured_m": best_to_measured,
        "selected_to_seed0_sc_reference_m": selected_to_seed0,
        "best_to_seed0_sc_reference_m": best_to_seed0,
        "selected_child_id": row.get("selected_child_id"),
        "selected_child_grid": row.get("selected_child_grid"),
        "best_descendant_id": row.get("best_descendant_id"),
        "best_descendant_grid": row.get("best_descendant_grid"),
    }


def summarize_classifications(class_rows: list[dict[str, Any]], formulas: list[str]) -> dict[str, Any]:
    keys = [
        "exact_seed0_sc",
        "spatial_seed0_sc_basin",
        "same_as_measured_for_seed",
        "measured_but_seed0_sc_basin",
        "distinct_sc_branch",
        "local_jitter",
        "unstable_or_missing",
        "avoids_short_local_sc",
        "source_measured_preferred",
    ]
    summary: dict[str, Any] = {}
    for formula in formulas:
        rows = [row for row in class_rows if row["formula"] == formula]
        summary[formula] = {
            "seed_count": len(rows),
            "counts": {key: sum(1 for row in rows if bool(row.get(key))) for key in keys},
            "fractions": {key: fraction(rows, key) for key in keys},
            "primary_counts": {
                label: sum(1 for row in rows if row.get("primary_classification") == label)
                for label in sorted({str(row.get("primary_classification")) for row in rows})
            },
        }
    return summary


def summary_rows_by_formula(decision_rows: list[dict[str, Any]], formulas: list[str], key: str) -> list[float]:
    return [as_float(row.get(key), float("nan")) for row in decision_rows if row["formula"] in formulas]


def build_margin_summary(decisions: list[dict[str, Any]], formulas: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for formula in formulas:
        subset = [row for row in decisions if row["formula"] == formula]
        margins = [as_float(row.get("normalized_margin"), float("nan")) for row in subset]
        clean = [value for value in margins if math.isfinite(value)]
        rows.append(
            {
                "formula": formula,
                "seed_count": len(subset),
                "winner_margin": percentile_summary(
                    [as_float(row.get("winner_margin"), float("nan")) for row in subset]
                ),
                "normalized_margin": percentile_summary(margins),
                "narrow_margin_fraction_normalized_lt_0p02": (
                    None if not clean else sum(1 for value in clean if value < 0.02) / len(clean)
                ),
            }
        )
    return rows


def build_overlap_summary(decisions: list[dict[str, Any]], formulas: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for formula in formulas:
        subset = [row for row in decisions if row["formula"] == formula]
        rows.append(
            {
                "formula": formula,
                "seed_count": len(subset),
                "root_visible_overlap_fraction": percentile_summary(
                    [as_float(row.get("root_visible_overlap_fraction"), float("nan")) for row in subset]
                ),
                "parent_path_visible_overlap_fraction": percentile_summary(
                    [as_float(row.get("parent_path_visible_overlap_fraction"), float("nan")) for row in subset]
                ),
                "winner_source_occ_free": percentile_summary(
                    [as_float(row.get("accumulated_source_occ_free"), float("nan")) for row in subset]
                ),
                "winner_formula_effective_sc": percentile_summary(
                    [as_float(row.get("accumulated_formula_effective_sc"), float("nan")) for row in subset]
                ),
            }
        )
    return rows


def build_frontier_summary(decisions: list[dict[str, Any]], formulas: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for formula in formulas:
        subset = [row for row in decisions if row["formula"] == formula]
        rows.append(
            {
                "formula": formula,
                "seed_count": len(subset),
                "frontier_local_count": percentile_summary(
                    [as_float(row.get("frontier_local_unique_count"), float("nan")) for row in subset]
                ),
                "frontier_local_fraction": percentile_summary(
                    [as_float(row.get("frontier_local_fraction"), float("nan")) for row in subset]
                ),
                "accumulated_frontier_local_source_occ_free": percentile_summary(
                    [as_float(row.get("accumulated_frontier_local_source_occ_free"), float("nan")) for row in subset]
                ),
            }
        )
    return rows


def formula_source_rows(formulas: list[str], mapping_status: str) -> list[dict[str, Any]]:
    rows = []
    for formula in formulas:
        info = FORMULA_SOURCE_STATUS.get(formula, {"source_faithfulness": "unknown", "description": ""})
        status = info["source_faithfulness"]
        if formula in {"source_occ_free", "source_occ_free_thresholded"} and mapping_status != "source-faithful-approx":
            status = mapping_status
        rows.append(
            {
                "formula": formula,
                "source_faithfulness": status,
                "prediction_used_for_ray_blocking": False,
                "prediction_used_for_traversability_collision": False,
                "prediction_weighted_by_confidence": formula == "current_confidence_weighted",
                "description": info["description"],
            }
        )
    return rows


def topdown_projection(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def plot_base_map(ax: plt.Axes, observed_state: np.ndarray) -> None:
    proj = topdown_projection(observed_state)
    colors = np.asarray(
        [
            [0.84, 0.84, 0.84, 1.0],
            [0.90, 0.97, 0.98, 1.0],
            [0.58, 0.12, 0.12, 1.0],
        ],
        dtype=np.float64,
    )
    ax.imshow(colors[proj].transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.grid(color="#111827", alpha=0.12, linewidth=0.4)


def formula_colors(formulas: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {formula: cmap(idx % 20) for idx, formula in enumerate(formulas)}


def grid_xy(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    try:
        return float(grid[0]), float(grid[1])
    except (TypeError, ValueError, IndexError):
        return None


def make_plots(
    output_dir: Path,
    observed_state: np.ndarray,
    decisions: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    class_summary: dict[str, Any],
    formulas: list[str],
) -> dict[str, str]:
    colors = formula_colors(formulas)
    plots: dict[str, str] = {}
    class_keys = [
        "exact_seed0_sc",
        "spatial_seed0_sc_basin",
        "same_as_measured_for_seed",
        "distinct_sc_branch",
        "avoids_short_local_sc",
    ]
    x = np.arange(len(formulas))
    bottom = np.zeros(len(formulas), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.2, 6.2), constrained_layout=True)
    for key in class_keys:
        values = [class_summary.get(formula, {}).get("fractions", {}).get(key) or 0.0 for formula in formulas]
        ax.bar(x, values, bottom=bottom, label=key)
        bottom += np.asarray(values)
    ax.set_xticks(x)
    ax.set_xticklabels(formulas, rotation=32, ha="right")
    ax.set_ylabel("fraction")
    ax.set_title("Branch classification fractions by formula")
    ax.legend(fontsize=8)
    path = output_dir / "formula_branch_classification_bar.png"
    fig.savefig(path, dpi=165)
    plt.close(fig)
    plots[path.name] = str(path)

    for filename, key, title in [
        ("formula_same_as_measured_fraction.png", "same_as_measured_for_seed", "Same-as-measured fraction"),
        ("formula_seed0_sc_basin_fraction.png", "spatial_seed0_sc_basin", "Seed0 SC basin fraction"),
        ("formula_avoids_short_local_sc_fraction.png", "avoids_short_local_sc", "Avoids short local SC fraction"),
    ]:
        fig, ax = plt.subplots(figsize=(10.8, 4.8), constrained_layout=True)
        vals = [class_summary.get(formula, {}).get("fractions", {}).get(key) or 0.0 for formula in formulas]
        ax.bar(formulas, vals, color=[colors[f] for f in formulas])
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=32)
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / filename
        fig.savefig(path, dpi=165)
        plt.close(fig)
        plots[path.name] = str(path)

    box_specs = [
        ("winner_source_occ_free_by_formula.png", "accumulated_source_occ_free", "Winner source OCC+FREE"),
        ("winner_cost_by_formula.png", "accumulated_cost", "Winner path cost"),
        ("root_overlap_fraction_by_formula.png", "root_visible_overlap_fraction", "Root visible overlap fraction"),
        ("frontier_local_fraction_by_formula.png", "frontier_local_fraction", "Frontier-local fraction"),
    ]
    for filename, key, title in box_specs:
        fig, ax = plt.subplots(figsize=(10.8, 5.0), constrained_layout=True)
        data = [
            [as_float(row.get(key), float("nan")) for row in decisions if row["formula"] == formula]
            for formula in formulas
        ]
        ax.boxplot(data, tick_labels=formulas, showmeans=True)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=32)
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / filename
        fig.savefig(path, dpi=165)
        plt.close(fig)
        plots[path.name] = str(path)

    means = []
    for formula in formulas:
        subset = [row for row in decisions if row["formula"] == formula]
        means.append(
            [
                statistics.fmean([as_float(row.get("accumulated_gain_exp"), 0.0) for row in subset]) if subset else 0.0,
                statistics.fmean([as_float(row.get("accumulated_formula_effective_sc"), 0.0) for row in subset]) if subset else 0.0,
                statistics.fmean([as_float(row.get("accumulated_cost"), 0.0) for row in subset]) if subset else 0.0,
            ]
        )
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    arr = np.asarray(means, dtype=np.float64)
    ax.bar(x - 0.25, arr[:, 0], width=0.25, label="gain_exp")
    ax.bar(x, arr[:, 1], width=0.25, label="formula_sc")
    ax.bar(x + 0.25, arr[:, 2], width=0.25, label="cost")
    ax.set_xticks(x)
    ax.set_xticklabels(formulas, rotation=32, ha="right")
    ax.set_title("Winner gain / SC / cost mean by formula")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "winner_gain_cost_stack_by_formula.png"
    fig.savefig(path, dpi=165)
    plt.close(fig)
    plots[path.name] = str(path)

    for filename, grid_key, title, ref_grid in [
        ("selected_children_by_formula_topdown.png", "selected_child_grid", "Selected children by formula", REFERENCE_SC_SELECTED_GRID),
        ("best_descendants_by_formula_topdown.png", "best_descendant_grid", "Best descendants by formula", REFERENCE_SC_BEST_GRID),
    ]:
        fig, ax = plt.subplots(figsize=(9.0, 7.6), constrained_layout=True)
        plot_base_map(ax, observed_state)
        for row in decisions:
            xy = grid_xy(row.get(grid_key))
            if xy is None:
                continue
            formula = str(row["formula"])
            label = formula if formula not in ax.get_legend_handles_labels()[1] else None
            ax.scatter(
                [xy[0]],
                [xy[1]],
                s=46,
                c=[colors[formula]],
                edgecolor="#111827",
                linewidth=0.4,
                alpha=0.78,
                label=label,
            )
        ref_xy = grid_xy(ref_grid)
        if ref_xy is not None:
            ax.scatter([ref_xy[0]], [ref_xy[1]], s=130, c="none", edgecolor="#111827", linewidth=1.6, label="seed0 SC ref")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=7)
        path = output_dir / filename
        fig.savefig(path, dpi=165)
        plt.close(fig)
        plots[path.name] = str(path)

    for filename, xkey, xlabel in [
        ("value_vs_source_occ_free_by_formula.png", "accumulated_source_occ_free", "accumulated source OCC+FREE"),
        ("value_vs_cost_by_formula.png", "accumulated_cost", "accumulated cost"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
        for formula in formulas:
            subset = [row for row in decisions if row["formula"] == formula]
            ax.scatter(
                [as_float(row.get(xkey), float("nan")) for row in subset],
                [as_float(row.get("value"), float("nan")) for row in subset],
                s=62,
                color=colors[formula],
                edgecolor="#111827",
                linewidth=0.45,
                label=formula,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("winner value")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        path = output_dir / filename
        fig.savefig(path, dpi=165)
        plt.close(fig)
        plots[path.name] = str(path)
    return plots


def write_decisions_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 4A-6.5y Per-Seed Formula Decisions",
        "",
        "| seed | formula | selected | selected grid | best | best grid | source OCC+FREE | formula SC | cost | value | class |",
        "|---:|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | `{row['formula']}` | `{row.get('selected_child_id')}` | "
            f"`{row.get('selected_child_grid')}` | `{row.get('best_descendant_id')}` | "
            f"`{row.get('best_descendant_grid')}` | {row.get('accumulated_source_occ_free')} | "
            f"{row.get('accumulated_formula_effective_sc')} | {row.get('accumulated_cost')} | "
            f"{row.get('value')} | `{row.get('classification', '')}` |"
        )
    write_text(path, "\n".join(lines))


def write_classification_md(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Branch Classification Summary By Formula", ""]
    for formula, info in summary.items():
        lines.extend([f"## {formula}", "", f"- seed count: `{info['seed_count']}`"])
        for key, value in info["counts"].items():
            lines.append(f"- {key}: `{value}` fraction `{info['fractions'][key]}`")
        lines.append("")
    write_text(path, "\n".join(lines))


def write_simple_table_md(path: Path, title: str, rows: list[dict[str, Any]]) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        write_text(path, "\n".join(lines))
        return
    keys = list(rows[0].keys())
    lines.append("| " + " | ".join(keys) + " |")
    lines.append("|" + "|".join("---" for _ in keys) + "|")
    for row in rows:
        lines.append("| " + " | ".join(f"`{row.get(key)}`" for key in keys) + " |")
    write_text(path, "\n".join(lines))


def write_formula_source_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Formula Source Faithfulness Table",
        "",
        "| formula | source faithfulness | prediction ray blocking | confidence weighting | description |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['formula']}` | `{row['source_faithfulness']}` | "
            f"`{row['prediction_used_for_ray_blocking']}` | `{row['prediction_weighted_by_confidence']}` | "
            f"{row['description']} |"
        )
    write_text(path, "\n".join(lines))


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    results = summary["key_results"]
    safety = summary["safety"]
    lines = [
        "# Stage 4A-6.5y Source-Gain Seed Replay Summary",
        "",
        f"1. Seeds run: `{summary['seed_count']}` (`{summary['seeds']}`).",
        f"2. Formulas run: `{summary['formulas']}`.",
        f"3. source_occ_free seed0 selection: `{results.get('source_occ_free_seed0_selected')}` -> `{results.get('source_occ_free_seed0_best')}`.",
        f"4. current_confidence_weighted seed0 reference match: `{results.get('seed0_current_confidence_matches_reference')}`.",
        f"5. source_occ_free spatial seed0 SC basin fraction: `{results.get('source_occ_free_spatial_seed0_sc_basin_fraction')}`.",
        f"6. parent-visible-cleared spatial seed0 SC basin fraction: `{results.get('parent_visible_cleared_spatial_seed0_sc_basin_fraction')}`.",
        f"7. frontier-local spatial seed0 SC basin fraction: `{results.get('frontier_local_spatial_seed0_sc_basin_fraction')}`.",
        f"8. recommended next small task: `{summary['recommended_next_faithful_step']}`.",
        f"9. runtime smoke ready: `{summary['answers']['runtime_smoke_readiness']}`.",
        f"10. rollout ready: `{summary['answers']['rollout_readiness']}`.",
        "",
        "## Safety",
        "",
    ]
    for key in [
        "isaac_startup",
        "new_capture",
        "map_predict_rerun",
        "sscnet_inference",
        "selected_action_execution",
        "rollout",
        "training_or_rl",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_scoring",
        "external_source_modified_or_built",
        "coverage_improvement_claim",
    ]:
        lines.append(f"- {key}: `{safety[key]}`")
    write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    formulas = [item.strip() for item in str(args.formulas).split(",") if item.strip()]
    required = {"measured_only", "source_occ_free", "source_occ_free_thresholded", "parent_visible_cleared_source_occ_free", "frontier_local_source_occ_free"}
    missing_required = sorted(required - set(formulas))
    if missing_required:
        raise ValueError(f"formulas missing required Stage 4A-6.5y entries: {missing_required}")
    unsupported = sorted(set(formulas) - set(CURRENT_FORMULA_TO_MINI) - SOURCE_FORMULAS)
    if unsupported:
        raise ValueError(f"unsupported formulas: {unsupported}")

    observed_path = Path(args.observed_state).resolve()
    prediction_path = Path(args.prediction_npz).resolve()
    pose_path = Path(args.pose_json).resolve()
    camera_path = Path(args.camera_info_json).resolve()
    for input_path in (observed_path, prediction_path, pose_path, camera_path):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    observed_hash_before = sha256_file(observed_path)
    prediction_hash_before = sha256_file(prediction_path)
    checkpoint_hash_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    observed_state = np.load(observed_path)
    observed_state.setflags(write=False)
    prediction = SimPredictionLayer.from_npz(prediction_path)
    if tuple(prediction.shape()) != tuple(observed_state.shape):
        raise ValueError(f"prediction shape {prediction.shape()} != observed_state {observed_state.shape}")

    inventory = inventory_prediction_npz(prediction_path)
    mapping_report = build_mapping_report(inventory, args)
    save_json(output_dir / "prediction_npz_field_inventory.json", inventory)
    write_inventory_md(output_dir / "prediction_npz_field_inventory.md", inventory)
    save_json(output_dir / "source_occ_free_mapping_report.json", mapping_report)
    write_mapping_md(output_dir / "source_occ_free_mapping_report.md", mapping_report)

    refs = reference_worlds(observed_state, float(args.voxel_size))
    replay_context = resolve_replay_context(Path(args.stage4a65v_dir).resolve())
    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / float(args.voxel_size))))
    num_yaw = max(4, int(math.ceil(32 / max(1, int(args.raycast_stride)))))
    num_pitch = max(3, int(math.ceil(7 / max(1, int(args.raycast_stride)))))
    visible_cache: dict[tuple[tuple[int, int, int], float, int, int], set[tuple[int, int, int]]] = {}

    manifest_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    missing_fields: list[dict[str, Any]] = []
    root_visible_by_seed: dict[int, set[tuple[int, int, int]]] = {}
    root_source_by_seed: dict[int, set[tuple[int, int, int]]] = {}
    raw_tree_components_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    raw_tree_segments_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    current_tree_dirs: dict[tuple[int, str], Path] = {}

    current_formulas_to_run = [formula for formula in formulas if formula in CURRENT_FORMULA_TO_MINI]
    if "current_raw_count" not in current_formulas_to_run and SOURCE_FORMULAS & set(formulas):
        current_formulas_to_run.append("current_raw_count")

    for seed in seeds:
        for formula in current_formulas_to_run:
            tree_dir = output_dir / "raw_trees" / f"seed_{seed:03d}" / formula
            mini_args = mini_tree_args(args, seed, formula, tree_dir, replay_context)
            before = time.perf_counter()
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                run_mini_rrt(mini_args)
            elapsed = time.perf_counter() - before
            current_tree_dirs[(seed, formula)] = tree_dir
            manifest_rows.append(
                {
                    "seed": seed,
                    "formula": formula,
                    "status": "mini_rrt_completed",
                    "tree_dir": str(tree_dir),
                    "elapsed_s": elapsed,
                    "suppressed_mini_rrt_stdout_chars": len(stdout_capture.getvalue()),
                }
            )
            print(json.dumps({"seed": seed, "formula": formula, "stage": "mini_rrt", "elapsed_s": round(elapsed, 3)}, sort_keys=True))

        raw_tree_dir = current_tree_dirs[(seed, "current_raw_count")]
        raw_tree = load_tree_segments(raw_tree_dir / "mini_rrt_tree_segments.jsonl")
        raw_tree_segments_by_seed[seed] = raw_tree
        root_segment = raw_tree[ROOT_ID]
        root_visible = compute_visible_set(root_segment, observed_state, max_range_voxels, num_yaw, num_pitch, visible_cache)
        root_masks = prediction_masks_for_visible(observed_state, prediction, root_visible, float(args.ssc_confidence_threshold))
        root_source = set(root_masks["occupied"]) | set(root_masks["free"])
        root_visible_by_seed[seed] = root_visible
        root_source_by_seed[seed] = root_source
        comps: dict[str, dict[str, Any]] = {}
        for node_id, segment in raw_tree.items():
            if node_id == ROOT_ID:
                continue
            comps[node_id] = source_component_for_segment(
                segment,
                observed_state,
                prediction,
                root_visible,
                max_range_voxels,
                num_yaw,
                num_pitch,
                float(args.ssc_confidence_threshold),
                visible_cache,
            )
        raw_tree_components_by_seed[seed] = comps

        for formula in [item for item in formulas if item in CURRENT_FORMULA_TO_MINI]:
            tree_dir = current_tree_dirs[(seed, formula)]
            tree_segments = load_tree_segments(tree_dir / "mini_rrt_tree_segments.jsonl")
            components = raw_tree_components_by_seed[seed] if formula == "current_raw_count" else {}
            if formula != "current_raw_count":
                root_segment_for_formula = tree_segments[ROOT_ID]
                root_visible_for_formula = compute_visible_set(
                    root_segment_for_formula, observed_state, max_range_voxels, num_yaw, num_pitch, visible_cache
                )
                root_masks_formula = prediction_masks_for_visible(
                    observed_state, prediction, root_visible_for_formula, float(args.ssc_confidence_threshold)
                )
                root_source_for_formula = set(root_masks_formula["occupied"]) | set(root_masks_formula["free"])
                for node_id, segment in tree_segments.items():
                    if node_id == ROOT_ID:
                        continue
                    components[node_id] = source_component_for_segment(
                        segment,
                        observed_state,
                        prediction,
                        root_visible_for_formula,
                        max_range_voxels,
                        num_yaw,
                        num_pitch,
                        float(args.ssc_confidence_threshold),
                        visible_cache,
                    )
            else:
                root_visible_for_formula = root_visible_by_seed[seed]
                root_source_for_formula = root_source_by_seed[seed]
            row = actual_tree_decision(
                seed=seed,
                formula=formula,
                tree_dir=tree_dir,
                prediction=prediction,
                node_components=components,
                root_visible=root_visible_for_formula,
                root_source=root_source_for_formula,
                tree_segments=tree_segments,
                missing_fields=missing_fields,
            )
            decision_rows.append(row)

        for formula in [item for item in formulas if item in SOURCE_FORMULAS]:
            before = time.perf_counter()
            row = source_tree_decision(
                seed=seed,
                formula=formula,
                tree_dir=raw_tree_dir,
                observed_state=observed_state,
                prediction=prediction,
                args=args,
                root_visible=root_visible_by_seed[seed],
                root_source=root_source_by_seed[seed],
                visible_cache=visible_cache,
                missing_fields=missing_fields,
            )
            elapsed = time.perf_counter() - before
            row["elapsed_s"] = elapsed
            decision_rows.append(row)
            manifest_rows.append(
                {
                    "seed": seed,
                    "formula": formula,
                    "status": "posthoc_source_rescore_completed",
                    "tree_dir": str(raw_tree_dir),
                    "elapsed_s": elapsed,
                    "selected_child_id": row.get("selected_child_id"),
                    "best_descendant_id": row.get("best_descendant_id"),
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "formula": formula,
                        "stage": "source_rescore",
                        "selected": row.get("selected_child_id"),
                        "best": row.get("best_descendant_id"),
                        "elapsed_s": round(elapsed, 3),
                    },
                    sort_keys=True,
                )
            )

    by_key = {(row["seed"], row["formula"]): row for row in decision_rows}
    class_rows: list[dict[str, Any]] = []
    for row in decision_rows:
        measured = by_key.get((row["seed"], "measured_only"), {})
        classification = classify_row(row, measured, refs)
        row["classification"] = classification["primary_classification"]
        row["selected_to_seed0_sc_reference_m"] = classification["selected_to_seed0_sc_reference_m"]
        row["best_to_seed0_sc_reference_m"] = classification["best_to_seed0_sc_reference_m"]
        row["selected_to_same_seed_measured_m"] = classification["selected_to_same_seed_measured_m"]
        row["avoids_short_local_sc"] = classification["avoids_short_local_sc"]
        row["source_measured_preferred"] = classification["source_measured_preferred"]
        class_rows.append(classification)

    class_summary = summarize_classifications(class_rows, formulas)
    margin_summary = build_margin_summary(decision_rows, formulas)
    overlap_summary = build_overlap_summary(decision_rows, formulas)
    frontier_summary = build_frontier_summary(decision_rows, formulas)
    source_rows = formula_source_rows(formulas, mapping_report["mapping_status"])
    plots = make_plots(output_dir, observed_state, decision_rows, class_rows, class_summary, formulas)

    seed0_conf = by_key.get((0, "current_confidence_weighted"), {})
    seed0_source = by_key.get((0, "source_occ_free"), {})
    seed0_conf_match = same_grid(seed0_conf.get("selected_child_grid"), REFERENCE_SC_SELECTED_GRID) and same_grid(
        seed0_conf.get("best_descendant_grid"), REFERENCE_SC_BEST_GRID
    )
    observed_hash_after = sha256_file(observed_path)
    prediction_hash_after = sha256_file(prediction_path)
    checkpoint_hash_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    prohibited = {
        pattern: sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
        for pattern in PROHIBITED_OUTPUT_PATTERNS
    }
    prohibited = {key: value for key, value in prohibited.items() if value}
    safety = {
        "isaac_startup": False,
        "new_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_or_rl": False,
        "checkpoint_modified": checkpoint_hash_before != checkpoint_hash_after,
        "observed_state_modified": observed_hash_before != observed_hash_after,
        "prediction_npz_modified": prediction_hash_before != prediction_hash_after,
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_scoring": False,
        "external_source_modified_or_built": False,
        "coverage_improvement_claim": False,
        "observed_state_sha256_before": observed_hash_before,
        "observed_state_sha256_after": observed_hash_after,
        "prediction_npz_sha256_before": prediction_hash_before,
        "prediction_npz_sha256_after": prediction_hash_after,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "prohibited_artifacts_in_output": prohibited,
    }
    key_results = {
        "seed0_current_confidence_matches_reference": bool(seed0_conf_match),
        "source_occ_free_seed0_selected": seed0_source.get("selected_child_id"),
        "source_occ_free_seed0_best": seed0_source.get("best_descendant_id"),
        "source_occ_free_seed0_selects_measured": bool(
            same_grid(seed0_source.get("selected_child_grid"), REFERENCE_MEASURED_SELECTED_GRID)
        ),
        "source_occ_free_spatial_seed0_sc_basin_fraction": class_summary.get("source_occ_free", {})
        .get("fractions", {})
        .get("spatial_seed0_sc_basin"),
        "parent_visible_cleared_spatial_seed0_sc_basin_fraction": class_summary.get(
            "parent_visible_cleared_source_occ_free", {}
        )
        .get("fractions", {})
        .get("spatial_seed0_sc_basin"),
        "frontier_local_spatial_seed0_sc_basin_fraction": class_summary.get("frontier_local_source_occ_free", {})
        .get("fractions", {})
        .get("spatial_seed0_sc_basin"),
    }
    next_step = "inspect source OCC/FREE mapping and source-inspired novelty filters offline before any runtime smoke"
    summary = {
        "stage": "Stage 4A-6.5y",
        "output_dir": str(output_dir),
        "seed_count": len(seeds),
        "seeds": seeds,
        "formulas": formulas,
        "inputs": {
            "observed_state": str(observed_path),
            "prediction_npz": str(prediction_path),
            "pose_json": str(pose_path),
            "camera_info_json": str(camera_path),
            "stage4a65p_dir": str(Path(args.stage4a65p_dir).resolve()),
            "stage4a65r_dir": str(Path(args.stage4a65r_dir).resolve()),
            "stage4a65v_dir": str(Path(args.stage4a65v_dir).resolve()),
            "stage4a65x_dir": str(Path(args.stage4a65x_dir).resolve()),
            "replay_case_json": replay_context["case_json"],
            "replay_episode_dir": replay_context["episode_dir"],
            "replay_episode_summary": replay_context["episode_summary"],
        },
        "parameters": {
            "num_nodes": int(args.num_nodes),
            "max_extension_m": float(args.max_extension_m),
            "sample_mode": args.sample_mode,
            "path_cost_mode": args.path_cost_mode,
            "v_max": float(args.v_max),
            "robot_radius_m": float(args.robot_radius_m),
            "voxel_size": float(args.voxel_size),
            "raycast_stride": int(args.raycast_stride),
            "num_yaw_samples": int(args.num_yaw_samples),
            "max_ray_length_m": float(args.max_ray_length_m),
            "short_edge_policy": args.short_edge_policy,
            "crop_min_length_m": float(args.crop_min_length_m),
            "tau": float(args.tau),
            "ssc_confidence_threshold": float(args.ssc_confidence_threshold),
            "alignment_convention": str(args.alignment_convention),
            "max_workers_requested": int(args.max_workers),
        },
        "mapping_report": mapping_report,
        "reference": refs,
        "branch_classification_summary": class_summary,
        "margin_summary_by_formula": margin_summary,
        "overlap_novelty_summary_by_formula": overlap_summary,
        "frontier_local_summary_by_formula": frontier_summary,
        "formula_source_faithfulness": source_rows,
        "key_results": key_results,
        "answers": {
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
            "coverage_improvement_claimed": False,
        },
        "recommended_next_faithful_step": next_step,
        "safety": safety,
        "plots": plots,
        "required_outputs": REQUIRED_FILES + REQUIRED_PLOTS,
        "elapsed_s": time.perf_counter() - started,
    }

    gain_component_rows = [
        {
            key: row.get(key)
            for key in [
                "seed",
                "formula",
                "accumulated_gain_exp",
                "accumulated_current_raw_gain_sc",
                "accumulated_source_occ_free",
                "accumulated_source_occ",
                "accumulated_source_free",
                "accumulated_source_unknown",
                "accumulated_parent_root_cleared_source_occ_free",
                "accumulated_root_cleared_source_occ_free",
                "accumulated_frontier_local_source_occ_free",
                "accumulated_parent_cleared_frontier_local_source_occ_free",
                "accumulated_branch_normalized_source_occ_free",
                "accumulated_formula_effective_sc",
                "accumulated_hybrid_effective",
                "accumulated_cost",
                "visible_predicted_occ_count",
                "visible_predicted_free_count",
                "visible_predicted_unknown_count",
                "visible_prediction_confidence_sum",
                "visible_prediction_confidence_mean",
                "visible_prediction_confidence_p90",
                "root_visible_overlap_fraction",
                "parent_path_visible_overlap_fraction",
                "frontier_local_fraction",
            ]
        }
        for row in decision_rows
    ]

    write_jsonl(output_dir / "source_gain_replay_manifest.jsonl", manifest_rows)
    write_csv(output_dir / "per_seed_formula_decisions.csv", decision_rows)
    save_json(output_dir / "per_seed_formula_decisions.json", decision_rows)
    write_decisions_md(output_dir / "per_seed_formula_decisions.md", decision_rows)
    write_csv(output_dir / "per_seed_formula_gain_components.csv", gain_component_rows)
    save_json(output_dir / "per_seed_formula_gain_components.json", gain_component_rows)
    write_csv(output_dir / "branch_classification_by_formula_seed.csv", class_rows)
    save_json(output_dir / "branch_classification_by_formula_seed.json", class_rows)
    save_json(output_dir / "branch_classification_summary_by_formula.json", class_summary)
    write_classification_md(output_dir / "branch_classification_summary_by_formula.md", class_summary)
    write_csv(output_dir / "margin_summary_by_formula.csv", margin_summary)
    save_json(output_dir / "margin_summary_by_formula.json", {"rows": margin_summary})
    write_simple_table_md(output_dir / "margin_summary_by_formula.md", "Margin Summary By Formula", margin_summary)
    write_csv(output_dir / "overlap_novelty_summary_by_formula.csv", overlap_summary)
    save_json(output_dir / "overlap_novelty_summary_by_formula.json", {"rows": overlap_summary})
    write_simple_table_md(output_dir / "overlap_novelty_summary_by_formula.md", "Overlap Novelty Summary By Formula", overlap_summary)
    write_csv(output_dir / "frontier_local_summary_by_formula.csv", frontier_summary)
    save_json(output_dir / "frontier_local_summary_by_formula.json", {"rows": frontier_summary})
    write_simple_table_md(output_dir / "frontier_local_summary_by_formula.md", "Frontier Local Summary By Formula", frontier_summary)
    write_csv(output_dir / "formula_source_faithfulness_table.csv", source_rows)
    save_json(output_dir / "formula_source_faithfulness_table.json", source_rows)
    write_formula_source_md(output_dir / "formula_source_faithfulness_table.md", source_rows)
    save_json(output_dir / "missing_fields_report.json", {"missing_fields": missing_fields, "count": len(missing_fields)})
    save_json(output_dir / "stage4a65y_source_gain_seed_replay_summary.json", summary)
    write_summary_md(output_dir / "stage4a65y_source_gain_seed_replay_summary.md", summary)
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {next_step}",
                "- still not next: runtime smoke, rollout, online open-ended loop, 3-frame smoke, RL/PPO/BC/IL training, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, coverage-improvement claims, or external source build.",
            ]
        ),
    )

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed_state", required=True)
    parser.add_argument("--prediction_npz", required=True)
    parser.add_argument("--pose_json", required=True)
    parser.add_argument("--camera_info_json", required=True)
    parser.add_argument("--stage4a65p_dir", required=True)
    parser.add_argument("--stage4a65r_dir", required=True)
    parser.add_argument("--stage4a65v_dir", required=True)
    parser.add_argument("--stage4a65x_dir", required=True)
    parser.add_argument("--local_ssc_exploration_dir", required=True)
    parser.add_argument("--external_active3d_dir", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument(
        "--formulas",
        default=(
            "measured_only,current_confidence_weighted,current_cap25,current_raw_count,"
            "source_occ_free,source_occ_free_thresholded,parent_visible_cleared_source_occ_free,"
            "root_visible_cleared_source_occ_free,frontier_local_source_occ_free,"
            "parent_cleared_frontier_local_source_occ_free,branch_normalized_source_occ_free"
        ),
    )
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
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--ssc_confidence_threshold", type=float, default=0.05)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_viz", action="store_true")
    return parser


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
