#!/usr/bin/env python3
"""Stage 4A-6.5v multi-seed offline replay for gated SC mini-RRT trees.

This runner is intentionally offline. It reads one saved Frame 2
observed_state and one saved prediction NPZ, rebuilds source-protected gated
trees for multiple tree seeds, and writes stability summaries. It does not
start Isaac, capture new RGB/depth, rerun map_predict or SSCNet inference,
execute actions, run rollout, train, modify checkpoints, or write prediction
into observed_state.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import io
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import ROOT_ID, run as run_mini_rrt, sha256_file, to_jsonable
from sim_paper_expert import FREE, OCCUPIED, UNKNOWN, grid_to_world, normalize_bounds


DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65v_multi_seed_offline_replay"
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

REQUIRED_FILES = [
    "multi_seed_replay_manifest.jsonl",
    "per_seed_formula_decisions.csv",
    "per_seed_formula_decisions.json",
    "per_seed_formula_decisions.md",
    "per_seed_rank_margin.csv",
    "per_seed_rank_margin.json",
    "branch_classification_by_seed.csv",
    "branch_classification_by_seed.json",
    "branch_classification_summary.json",
    "branch_classification_summary.md",
    "confidence_vs_cap25_agreement.csv",
    "confidence_vs_cap25_agreement.json",
    "spatial_basin_summary.csv",
    "spatial_basin_summary.json",
    "spatial_basin_summary.md",
    "missing_fields_report.json",
    "stage4a65v_multi_seed_replay_summary.json",
    "stage4a65v_multi_seed_replay_summary.md",
    "recommended_next_faithful_step.md",
]

REQUIRED_PLOTS = [
    "selected_children_by_seed_topdown.png",
    "best_descendants_by_seed_topdown.png",
    "seed_classification_bar.png",
    "margin_distribution_by_formula.png",
    "selected_delta_to_seed0_sc.png",
    "confidence_vs_cap25_selected_delta.png",
    "value_vs_effective_sc_by_seed.png",
    "value_vs_cost_by_seed.png",
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

FORMULA_COLORS = {
    "measured_only": "#2563eb",
    "confidence_weighted": "#f97316",
    "cap25": "#7c3aed",
    "raw_count": "#dc2626",
}


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
    path.write_text(text, encoding="utf-8")


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


def parse_literal_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    return list(parsed) if isinstance(parsed, (list, tuple)) else None


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def euclidean(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        aa = [float(v) for v in a]
        bb = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if len(aa) < 2 or len(bb) < 2:
        return None
    dims = min(len(aa), len(bb))
    return float(math.sqrt(sum((aa[idx] - bb[idx]) ** 2 for idx in range(dims))))


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1.0e-9:
        return None
    return float(numerator) / float(denominator)


def min_mean_max(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(clean), "mean": statistics.fmean(clean), "max": max(clean)}


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


def pearson_pairs(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= 1.0e-9 or float(np.std(y_arr)) <= 1.0e-9:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def parse_csv_table(path: Path) -> list[dict[str, Any]]:
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
        "local_gain_over_cost",
        "local_effective_hybrid_over_cost",
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
        for key in ("end_grid", "end_world"):
            row[key] = parse_literal_list(row.get(key))
        for key in numeric:
            if key in row:
                row[key] = as_float(row.get(key), float("nan"))
        rows.append(row)
    return rows


def table_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("segment_id")): row for row in rows if row.get("segment_id") is not None}


def path_to_root(rows_by_id: dict[str, dict[str, Any]], node_id: str | None) -> list[str]:
    if not node_id or node_id not in rows_by_id:
        return []
    path: list[str] = []
    seen: set[str] = set()
    current: str | None = node_id
    while current and current in rows_by_id and current not in seen:
        seen.add(current)
        path.append(current)
        parent = rows_by_id[current].get("parent_id")
        current = str(parent) if parent else None
    path.reverse()
    return [node for node in path if node != ROOT_ID]


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


def tree_args(args: argparse.Namespace, seed: int, formula: str, tree_dir: Path) -> argparse.Namespace:
    measured = formula == "measured_only"
    return argparse.Namespace(
        case_json=str(getattr(args, "replay_case_json", "")),
        episode_dir=str(getattr(args, "replay_episode_dir", "")),
        observed_state=str(Path(args.observed_state).resolve()),
        pose_json=str(Path(args.pose_json).resolve()),
        camera_info=str(Path(args.camera_info_json).resolve()),
        episode_summary=str(getattr(args, "replay_episode_summary", "")),
        prediction_npz="" if measured else str(Path(args.prediction_npz).resolve()),
        output_dir=str(tree_dir),
        seed=int(seed),
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        gain_mode="exp" if measured else "hybrid",
        sc_gain_formula="measured_only" if measured else str(formula),
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
        variant_name=f"stage4a65v_seed{int(seed):03d}_{formula}",
    )


def summarize_tree_result(
    *,
    seed: int,
    formula: str,
    tree_dir: Path,
    summary: dict[str, Any],
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    table_rows = parse_csv_table(tree_dir / "gain_cost_value_table.csv")
    rows_by_id = table_by_id(table_rows)
    root = rows_by_id.get(ROOT_ID, {})
    decision = summary.get("decision", {})
    selected_payload = decision.get("selected_child") if isinstance(decision.get("selected_child"), dict) else {}
    best_payload = decision.get("best_descendant") if isinstance(decision.get("best_descendant"), dict) else {}

    selected_id = decision.get("selected_child_id") or selected_payload.get("segment_id")
    best_id = decision.get("selected_child_best_descendant_id") or best_payload.get("segment_id")
    selected = rows_by_id.get(str(selected_id), {}) if selected_id is not None else {}
    best = rows_by_id.get(str(best_id), {}) if best_id is not None else {}

    selected_grid = selected.get("end_grid") or selected_payload.get("end_grid")
    selected_world = selected.get("end_world") or selected_payload.get("end_world")
    best_grid = best.get("end_grid") or best_payload.get("end_grid")
    best_world = best.get("end_world") or best_payload.get("end_world")
    root_world = root.get("end_world") or summary.get("root", {}).get("resolved_world") or summary.get("root", {}).get("world")
    root_grid = root.get("end_grid") or summary.get("root", {}).get("resolved_grid") or summary.get("root", {}).get("grid")

    for field, value in {
        "selected_child_id": selected_id,
        "selected_child_grid": selected_grid,
        "selected_child_world": selected_world,
        "best_descendant_id": best_id,
        "best_descendant_grid": best_grid,
        "best_descendant_world": best_world,
        "root_world": root_world,
    }.items():
        if value is None:
            missing.append(
                {
                    "seed": seed,
                    "formula": formula,
                    "tree_dir": str(tree_dir),
                    "field": field,
                    "severity": "derived_missing",
                }
            )

    path_ids = path_to_root(rows_by_id, str(best_id) if best_id is not None else None)
    path_rows = [rows_by_id[node_id] for node_id in path_ids if node_id in rows_by_id]
    non_root = [row for row in table_rows if row.get("segment_id") != ROOT_ID]
    values = [as_float(row.get("value"), float("nan")) for row in non_root]
    costs = [as_float(row.get("accumulated_cost"), float("nan")) for row in non_root]
    inverse_cost = [safe_ratio(1.0, cost) or float("nan") for cost in costs]
    gain_exp = [as_float(row.get("gain_exp"), float("nan")) for row in non_root]
    effective_sc = [as_float(row.get("effective_gain_sc"), float("nan")) for row in non_root]
    raw_sc = [as_float(row.get("gain_sc"), float("nan")) for row in non_root]
    margin = margin_from_table(table_rows)

    selected_value = as_float(selected.get("value"), float("nan"))
    best_value = as_float(best.get("value"), float("nan"))
    return {
        "seed": int(seed),
        "formula": str(formula),
        "status": "completed" if bool(summary.get("tree", {}).get("built_successfully")) else "built_limited",
        "gain_mode": "exp" if formula == "measured_only" else "hybrid",
        "sc_gain_formula": "measured_only" if formula == "measured_only" else formula,
        "tree_dir": str(tree_dir),
        "selected_child_id": selected_id,
        "selected_child_grid": selected_grid,
        "selected_child_world": selected_world,
        "best_descendant_id": best_id,
        "best_descendant_grid": best_grid,
        "best_descendant_world": best_world,
        "root_grid": root_grid,
        "root_world": root_world,
        "selected_child_distance_from_root_m": euclidean(root_world, selected_world),
        "best_descendant_distance_from_root_m": euclidean(root_world, best_world),
        "accumulated_gain_exp": float(sum(as_float(row.get("gain_exp")) for row in path_rows)),
        "accumulated_raw_gain_sc": float(sum(as_float(row.get("gain_sc")) for row in path_rows)),
        "accumulated_effective_gain_sc": float(sum(as_float(row.get("effective_gain_sc")) for row in path_rows)),
        "accumulated_hybrid_effective": float(sum(as_float(row.get("gain_hybrid_effective")) for row in path_rows)),
        "accumulated_cost": float(sum(as_float(row.get("cost")) for row in path_rows)),
        "value": selected_value if math.isfinite(selected_value) else None,
        "best_descendant_value": best_value if math.isfinite(best_value) else None,
        "runner_up_value": margin.get("runner_up_value"),
        "winner_margin": margin.get("winner_margin"),
        "normalized_margin": margin.get("normalized_margin"),
        "winner_id": margin.get("winner_id"),
        "runner_up_id": margin.get("runner_up_id"),
        "root_child_count": margin.get("root_child_count"),
        "branch_depth": len(path_ids),
        "path_node_ids": path_ids,
        "nodes_with_effective_gain_sc_positive": sum(1 for value in effective_sc if value > 0.0),
        "effective_gain_sc_min_mean_max": min_mean_max(effective_sc),
        "raw_gain_sc_min_mean_max": min_mean_max(raw_sc),
        "gain_exp_effective_gain_sc_correlation": pearson_pairs(gain_exp, effective_sc),
        "value_effective_gain_sc_correlation": pearson_pairs(values, effective_sc),
        "value_cost_correlation": pearson_pairs(values, costs),
        "value_inverse_cost_correlation": pearson_pairs(values, inverse_cost),
        "accepted_nodes": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
        "rejected_samples": summary.get("tree", {}).get("rejected_samples"),
    }


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
    distinct = False if missing or selected_to_measured is None else bool(
        selected_to_measured >= 0.25 and not spatial
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
        "selected_to_same_seed_measured_m": selected_to_measured,
        "best_to_same_seed_measured_m": best_to_measured,
        "selected_to_seed0_sc_reference_m": selected_to_seed0,
        "best_to_seed0_sc_reference_m": best_to_seed0,
        "selected_child_id": row.get("selected_child_id"),
        "selected_child_grid": row.get("selected_child_grid"),
        "best_descendant_id": row.get("best_descendant_id"),
        "best_descendant_grid": row.get("best_descendant_grid"),
    }


def fraction(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(1 for row in rows if bool(row.get(key)))) / float(len(rows))


def summarize_classifications(class_rows: list[dict[str, Any]], formulas: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    keys = [
        "exact_seed0_sc",
        "spatial_seed0_sc_basin",
        "same_as_measured_for_seed",
        "measured_but_seed0_sc_basin",
        "distinct_sc_branch",
        "local_jitter",
        "unstable_or_missing",
    ]
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


def build_agreement_rows(decisions: list[dict[str, Any]], seeds: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {(row["seed"], row["formula"]): row for row in decisions}
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        measured = by_key.get((seed, "measured_only"), {})
        conf = by_key.get((seed, "confidence_weighted"), {})
        cap = by_key.get((seed, "cap25"), {})
        raw = by_key.get((seed, "raw_count"), {})
        conf_cap_delta = euclidean(conf.get("selected_child_world"), cap.get("selected_child_world"))
        conf_measured_delta = euclidean(conf.get("selected_child_world"), measured.get("selected_child_world"))
        cap_measured_delta = euclidean(cap.get("selected_child_world"), measured.get("selected_child_world"))
        raw_measured_delta = euclidean(raw.get("selected_child_world"), measured.get("selected_child_world"))
        rows.append(
            {
                "seed": seed,
                "confidence_selected_child_id": conf.get("selected_child_id"),
                "cap25_selected_child_id": cap.get("selected_child_id"),
                "measured_selected_child_id": measured.get("selected_child_id"),
                "raw_count_selected_child_id": raw.get("selected_child_id"),
                "confidence_cap25_exact_grid_agree": same_grid(
                    conf.get("selected_child_grid"), cap.get("selected_child_grid")
                ),
                "confidence_cap25_spatial_agree_0p15m": (
                    conf_cap_delta is not None and conf_cap_delta <= 0.15
                ),
                "confidence_measured_exact_grid_agree": same_grid(
                    conf.get("selected_child_grid"), measured.get("selected_child_grid")
                ),
                "confidence_measured_spatial_agree_0p15m": (
                    conf_measured_delta is not None and conf_measured_delta <= 0.15
                ),
                "cap25_measured_exact_grid_agree": same_grid(
                    cap.get("selected_child_grid"), measured.get("selected_child_grid")
                ),
                "cap25_measured_spatial_agree_0p15m": cap_measured_delta is not None and cap_measured_delta <= 0.15,
                "raw_count_measured_exact_grid_agree": same_grid(
                    raw.get("selected_child_grid"), measured.get("selected_child_grid")
                ),
                "confidence_cap25_selected_delta_m": conf_cap_delta,
                "confidence_measured_selected_delta_m": conf_measured_delta,
                "cap25_measured_selected_delta_m": cap_measured_delta,
                "raw_count_measured_selected_delta_m": raw_measured_delta,
            }
        )

    def rate(key: str) -> float | None:
        return fraction(rows, key)

    summary = {
        "seed_count": len(rows),
        "confidence_vs_cap25_exact_grid_agreement_rate": rate("confidence_cap25_exact_grid_agree"),
        "confidence_vs_cap25_spatial_0p15m_agreement_rate": rate("confidence_cap25_spatial_agree_0p15m"),
        "confidence_vs_measured_exact_grid_agreement_rate": rate("confidence_measured_exact_grid_agree"),
        "confidence_vs_measured_spatial_0p15m_agreement_rate": rate("confidence_measured_spatial_agree_0p15m"),
        "cap25_vs_measured_exact_grid_agreement_rate": rate("cap25_measured_exact_grid_agree"),
        "cap25_vs_measured_spatial_0p15m_agreement_rate": rate("cap25_measured_spatial_agree_0p15m"),
        "raw_count_vs_measured_exact_grid_agreement_rate": rate("raw_count_measured_exact_grid_agree"),
        "confidence_cap25_selected_delta_m": percentile_summary(
            [as_float(row.get("confidence_cap25_selected_delta_m"), float("nan")) for row in rows]
        ),
    }
    return rows, summary


def build_spatial_basin_rows(class_rows: list[dict[str, Any]], formulas: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for formula in formulas:
        subset = [row for row in class_rows if row["formula"] == formula]
        selected = [as_float(row.get("selected_to_seed0_sc_reference_m"), float("nan")) for row in subset]
        best = [as_float(row.get("best_to_seed0_sc_reference_m"), float("nan")) for row in subset]
        clean_selected = [value for value in selected if math.isfinite(value)]
        clean_best = [value for value in best if math.isfinite(value)]
        rows.append(
            {
                "formula": formula,
                "seed_count": len(subset),
                "exact_seed0_sc_fraction": fraction(subset, "exact_seed0_sc"),
                "spatial_seed0_sc_basin_fraction": fraction(subset, "spatial_seed0_sc_basin"),
                "same_as_measured_for_seed_fraction": fraction(subset, "same_as_measured_for_seed"),
                "measured_but_seed0_sc_basin_fraction": fraction(subset, "measured_but_seed0_sc_basin"),
                "distinct_sc_branch_fraction": fraction(subset, "distinct_sc_branch"),
                "local_jitter_fraction": fraction(subset, "local_jitter"),
                "selected_delta_to_seed0_sc_mean_m": statistics.fmean(clean_selected) if clean_selected else None,
                "selected_delta_to_seed0_sc_median_m": statistics.median(clean_selected) if clean_selected else None,
                "best_delta_to_seed0_sc_mean_m": statistics.fmean(clean_best) if clean_best else None,
                "best_delta_to_seed0_sc_median_m": statistics.median(clean_best) if clean_best else None,
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
    agreement_rows: list[dict[str, Any]],
    formulas: list[str],
) -> dict[str, str]:
    plots: dict[str, str] = {}
    formula_markers = {
        "measured_only": "o",
        "confidence_weighted": "^",
        "cap25": "s",
        "raw_count": "X",
    }

    fig, ax = plt.subplots(figsize=(8.5, 7.4), constrained_layout=True)
    plot_base_map(ax, observed_state)
    for row in decisions:
        xy = grid_xy(row.get("selected_child_grid"))
        if xy is None:
            continue
        formula = str(row["formula"])
        label = formula if formula not in ax.get_legend_handles_labels()[1] else None
        ax.scatter(
            [xy[0]],
            [xy[1]],
            s=54,
            c=FORMULA_COLORS.get(formula, "#64748b"),
            marker=formula_markers.get(formula, "o"),
            edgecolor="#111827",
            linewidth=0.55,
            alpha=0.78,
            label=label,
        )
        if formula == "confidence_weighted":
            ax.text(xy[0] + 0.35, xy[1] + 0.35, str(row["seed"]), fontsize=7, color="#111827")
    ref_xy = grid_xy(REFERENCE_SC_SELECTED_GRID)
    if ref_xy is not None:
        ax.scatter([ref_xy[0]], [ref_xy[1]], s=120, c="none", edgecolor="#111827", linewidth=1.6, label="seed0 SC ref")
    ax.set_title("Selected children by seed")
    ax.legend(loc="upper right", fontsize=8)
    path = output_dir / "selected_children_by_seed_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(8.5, 7.4), constrained_layout=True)
    plot_base_map(ax, observed_state)
    for row in decisions:
        xy = grid_xy(row.get("best_descendant_grid"))
        if xy is None:
            continue
        formula = str(row["formula"])
        label = formula if formula not in ax.get_legend_handles_labels()[1] else None
        ax.scatter(
            [xy[0]],
            [xy[1]],
            s=54,
            c=FORMULA_COLORS.get(formula, "#64748b"),
            marker=formula_markers.get(formula, "o"),
            edgecolor="#111827",
            linewidth=0.55,
            alpha=0.78,
            label=label,
        )
        if formula == "confidence_weighted":
            ax.text(xy[0] + 0.35, xy[1] + 0.35, str(row["seed"]), fontsize=7, color="#111827")
    ref_xy = grid_xy(REFERENCE_SC_BEST_GRID)
    if ref_xy is not None:
        ax.scatter([ref_xy[0]], [ref_xy[1]], s=120, c="none", edgecolor="#111827", linewidth=1.6, label="seed0 SC ref")
    ax.set_title("Best descendants by seed")
    ax.legend(loc="upper right", fontsize=8)
    path = output_dir / "best_descendants_by_seed_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    class_keys = [
        "exact_seed0_sc",
        "spatial_seed0_sc_basin",
        "same_as_measured_for_seed",
        "measured_but_seed0_sc_basin",
        "distinct_sc_branch",
        "local_jitter",
    ]
    x = np.arange(len(class_keys))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.2, 5.6), constrained_layout=True)
    for offset, formula in [(-0.18, "confidence_weighted"), (0.18, "cap25")]:
        subset = [row for row in class_rows if row["formula"] == formula]
        counts = [sum(1 for row in subset if row.get(key)) for key in class_keys]
        ax.bar(x + offset, counts, width=width, label=formula, color=FORMULA_COLORS.get(formula))
    ax.set_xticks(x)
    ax.set_xticklabels(class_keys, rotation=28, ha="right")
    ax.set_ylabel("seed count")
    ax.set_title("Branch classification counts")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "seed_classification_bar.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    data = [
        [
            as_float(row.get("normalized_margin"), float("nan"))
            for row in decisions
            if row["formula"] == formula and row.get("normalized_margin") is not None
        ]
        for formula in formulas
    ]
    ax.boxplot(data, tick_labels=formulas, showmeans=True)
    ax.axhline(0.02, color="#dc2626", linestyle="--", linewidth=1.0, label="narrow < 0.02")
    ax.set_ylabel("normalized winner margin")
    ax.set_title("Margin distribution by formula")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    path = output_dir / "margin_distribution_by_formula.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for formula in formulas:
        rows = sorted([row for row in class_rows if row["formula"] == formula], key=lambda row: row["seed"])
        ax.plot(
            [row["seed"] for row in rows],
            [as_float(row.get("selected_to_seed0_sc_reference_m"), float("nan")) for row in rows],
            marker=formula_markers.get(formula, "o"),
            color=FORMULA_COLORS.get(formula, "#64748b"),
            label=formula,
        )
    ax.axhline(0.25, color="#111827", linestyle="--", linewidth=1.0, label="0.25m basin")
    ax.set_xlabel("tree seed")
    ax.set_ylabel("selected-child delta to seed0 SC ref (m)")
    ax.set_title("Selected delta to seed0 SC reference")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / "selected_delta_to_seed0_sc.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.bar(
        [row["seed"] for row in agreement_rows],
        [as_float(row.get("confidence_cap25_selected_delta_m"), float("nan")) for row in agreement_rows],
        color="#7c3aed",
    )
    ax.axhline(0.15, color="#111827", linestyle="--", linewidth=1.0, label="0.15m")
    ax.set_xlabel("tree seed")
    ax.set_ylabel("confidence vs cap25 selected delta (m)")
    ax.set_title("Confidence / cap25 selected-child delta")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "confidence_vs_cap25_selected_delta.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    for formula in formulas:
        rows = [row for row in decisions if row["formula"] == formula]
        ax.scatter(
            [as_float(row.get("accumulated_effective_gain_sc"), float("nan")) for row in rows],
            [as_float(row.get("value"), float("nan")) for row in rows],
            s=62,
            marker=formula_markers.get(formula, "o"),
            c=FORMULA_COLORS.get(formula, "#64748b"),
            edgecolor="#111827",
            linewidth=0.45,
            label=formula,
        )
    ax.set_xlabel("winning-path accumulated effective SC")
    ax.set_ylabel("selected root-child value")
    ax.set_title("Value vs effective SC by seed")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / "value_vs_effective_sc_by_seed.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), constrained_layout=True)
    for formula in formulas:
        rows = [row for row in decisions if row["formula"] == formula]
        ax.scatter(
            [as_float(row.get("accumulated_cost"), float("nan")) for row in rows],
            [as_float(row.get("value"), float("nan")) for row in rows],
            s=62,
            marker=formula_markers.get(formula, "o"),
            c=FORMULA_COLORS.get(formula, "#64748b"),
            edgecolor="#111827",
            linewidth=0.45,
            label=formula,
        )
    ax.set_xlabel("winning-path accumulated cost")
    ax.set_ylabel("selected root-child value")
    ax.set_title("Value vs cost by seed")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / "value_vs_cost_by_seed.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)
    return plots


def extract_reference_decision(stage_dir: Path, mode: str) -> dict[str, Any]:
    candidates = [
        stage_dir / f"frame002_{mode}_tree_decision.json",
        stage_dir / f"frame002_{mode}_subsequent_best_decision.json",
    ]
    data: dict[str, Any] = {}
    for path in candidates:
        if path.is_file():
            data = read_json(path)
            break
    if isinstance(data.get("decision"), dict):
        raw = data["decision"]
    elif isinstance(data.get("source_payload"), dict):
        raw = data["source_payload"]
    elif isinstance(data.get("raw_decision"), dict):
        raw = data["raw_decision"]
    else:
        raw = data
    selected = raw.get("selected_child") if isinstance(raw.get("selected_child"), dict) else {}
    best = raw.get("best_descendant") if isinstance(raw.get("best_descendant"), dict) else {}
    return {
        "selected_child_id": raw.get("selected_child_id") or selected.get("segment_id"),
        "selected_child_grid": raw.get("selected_child_grid") or selected.get("end_grid"),
        "selected_child_world": raw.get("selected_child_world") or selected.get("end_world"),
        "best_descendant_id": raw.get("selected_child_best_descendant_id") or best.get("segment_id"),
        "best_descendant_grid": raw.get("best_descendant_grid") or best.get("end_grid"),
        "best_descendant_world": raw.get("best_descendant_world") or best.get("end_world"),
    }


def resolve_replay_context(seed0_reference_dir: Path) -> dict[str, str]:
    """Reuse the saved wrapper context that fixes the source-protected root.

    Stage 4A-6.5p/6.5s passed a selected-case JSON and episode dir through to
    offline_mini_rrt_tree. The observed_state/prediction remain the canonical
    explicit inputs, but the episode step resolves the same root grid used by
    the reference artifacts.
    """

    summary = read_json(seed0_reference_dir / "frame002_confidence_weighted_mini_rrt_tree_summary.json")
    inputs = summary.get("inputs", {}) if isinstance(summary.get("inputs"), dict) else {}
    case_json = str(inputs.get("case_json") or DEFAULT_SELECTED_CASE)
    episode_dir = str(inputs.get("episode_dir") or DEFAULT_EPISODE_DIR)
    episode_summary = str(inputs.get("episode_summary") or (Path(episode_dir) / "episode_summary.json"))
    return {
        "case_json": case_json if Path(case_json).is_file() else "",
        "episode_dir": episode_dir if Path(episode_dir).is_dir() else "",
        "episode_summary": episode_summary if Path(episode_summary).is_file() else "",
    }


def compare_to_reference(row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    selected_delta = euclidean(row.get("selected_child_world"), ref.get("selected_child_world"))
    best_delta = euclidean(row.get("best_descendant_world"), ref.get("best_descendant_world"))
    return {
        "row_seed": row.get("seed"),
        "row_formula": row.get("formula"),
        "reference_selected_child_id": ref.get("selected_child_id"),
        "reference_best_descendant_id": ref.get("best_descendant_id"),
        "exact_grid_match": same_grid(row.get("selected_child_grid"), ref.get("selected_child_grid"))
        and same_grid(row.get("best_descendant_grid"), ref.get("best_descendant_grid")),
        "spatial_match": selected_delta is not None
        and best_delta is not None
        and selected_delta <= 0.25
        and best_delta <= 0.75,
        "selected_delta_m": selected_delta,
        "best_delta_m": best_delta,
    }


def choose_recommendation(class_summary: dict[str, Any], confidence_margins: list[float]) -> tuple[str, str, dict[str, Any]]:
    conf = class_summary.get("confidence_weighted", {})
    fractions = conf.get("fractions", {})
    spatial = as_float(fractions.get("spatial_seed0_sc_basin"), 0.0)
    same = as_float(fractions.get("same_as_measured_for_seed"), 0.0)
    narrow_values = [value for value in confidence_margins if math.isfinite(float(value))]
    narrow_count = sum(1 for value in narrow_values if float(value) < 0.02)
    mostly_narrow = bool(narrow_values and narrow_count / len(narrow_values) >= 0.5)
    if spatial >= 0.70 and same >= 0.70:
        next_step = "branch basin / sampling stability diagnosis"
        reason = "confidence_weighted is usually in the seed0 SC spatial basin, but same-seed measured agreement dominates."
    elif spatial >= 0.70 and mostly_narrow:
        next_step = "margin/ranking stabilization before 3-frame"
        reason = "confidence_weighted is spatially stable, but normalized margins are mostly narrow."
    elif spatial >= 0.70:
        next_step = "another start/scene seed repeated gated two-frame smoke"
        reason = "confidence_weighted is spatially stable across seeds and margins are not mostly narrow."
    else:
        next_step = "tree sampling stabilization or SC gain design review"
        reason = "confidence_weighted diverges widely across tree seeds."
    return next_step, reason, {
        "confidence_spatial_fraction": spatial,
        "confidence_same_as_measured_fraction": same,
        "confidence_margin_count": len(narrow_values),
        "narrow_margin_count_normalized_lt_0p02": narrow_count,
        "margins_mostly_narrow": mostly_narrow,
    }


def write_decisions_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 4A-6.5v Per-Seed Formula Decisions",
        "",
        "| seed | formula | selected | selected grid | best | best grid | value | margin | norm margin | class |",
        "|---:|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {seed} | `{formula}` | `{selected}` | `{selected_grid}` | `{best}` | `{best_grid}` | {value} | {margin} | {norm} | `{classification}` |".format(
                seed=row["seed"],
                formula=row["formula"],
                selected=row.get("selected_child_id"),
                selected_grid=row.get("selected_child_grid"),
                best=row.get("best_descendant_id"),
                best_grid=row.get("best_descendant_grid"),
                value=row.get("value"),
                margin=row.get("winner_margin"),
                norm=row.get("normalized_margin"),
                classification=row.get("classification", ""),
            )
        )
    lines.append("")
    write_text(path, "\n".join(lines))


def write_classification_md(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Branch Classification Summary", ""]
    for formula, info in summary.items():
        lines.append(f"## {formula}")
        lines.append("")
        lines.append(f"- seed count: `{info['seed_count']}`")
        for key, value in info["counts"].items():
            lines.append(f"- {key}: `{value}` fraction `{info['fractions'][key]}`")
        lines.append("")
    write_text(path, "\n".join(lines))


def write_spatial_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Spatial Basin Summary",
        "",
        "| formula | spatial fraction | same-as-measured fraction | selected mean m | selected median m | best mean m | best median m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['formula']}` | {row.get('spatial_seed0_sc_basin_fraction')} | "
            f"{row.get('same_as_measured_for_seed_fraction')} | "
            f"{row.get('selected_delta_to_seed0_sc_mean_m')} | "
            f"{row.get('selected_delta_to_seed0_sc_median_m')} | "
            f"{row.get('best_delta_to_seed0_sc_mean_m')} | "
            f"{row.get('best_delta_to_seed0_sc_median_m')} |"
        )
    lines.append("")
    write_text(path, "\n".join(lines))


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    result = summary["multi_seed_results"]
    stability = summary["stability"]
    interpretation = summary["interpretation"]
    safety = summary["safety"]
    lines = [
        "# Stage 4A-6.5v Multi-Seed Offline Replay Summary",
        "",
        f"1. Tree seeds run: `{summary['seed_count']}` (`{summary['seeds']}`).",
        f"2. confidence_weighted exact seed0 SC: `{result['confidence_exact_seed0_sc_fraction']}`.",
        f"3. confidence_weighted spatial seed0 SC basin: `{result['confidence_spatial_seed0_sc_basin_fraction']}`.",
        f"4. confidence_weighted same-as-measured: `{result['confidence_same_as_measured_fraction']}`.",
        f"5. cap25/confidence agreement rate: `{result['confidence_cap25_agreement_rate']}`.",
        f"6. raw_count behavior: {interpretation['raw_count_behavior']}",
        f"7. Margins mostly narrow? `{stability['margins_mostly_narrow']}`; narrow seeds `{stability['confidence_narrow_margin_seeds_normalized_lt_0p02']}`.",
        f"8. seed0/seed1 difference: {interpretation['seed0_seed1_difference']}",
        f"9. Current support for 3-frame gated smoke: `{interpretation['ready_for_3_frame']}`.",
        f"10. Current support for another start/scene two-frame smoke: `{interpretation['ready_for_another_start_scene_two_frame']}`.",
        f"11. Current support for rollout: `{interpretation['ready_for_rollout']}`.",
        "",
        f"- recommended next small task: `{summary['recommended_next_faithful_step']}`",
        f"- why: {summary['recommendation_reason']}",
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
        "source_modified_built",
        "coverage_improvement_claim",
    ]:
        lines.append(f"- {key}: `{safety[key]}`")
    lines.append("")
    write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    formulas = [item.strip() for item in str(args.formulas).split(",") if item.strip()]
    required_formulas = {"measured_only", "confidence_weighted", "cap25"}
    missing_required = sorted(required_formulas - set(formulas))
    if missing_required:
        raise ValueError(f"formulas must include {missing_required}")

    observed_path = Path(args.observed_state).resolve()
    prediction_path = Path(args.prediction_npz).resolve()
    pose_path = Path(args.pose_json).resolve()
    camera_path = Path(args.camera_info_json).resolve()
    for path in [observed_path, prediction_path, pose_path, camera_path]:
        if not path.is_file():
            raise FileNotFoundError(path)

    observed_hash_before = sha256_file(observed_path)
    prediction_hash_before = sha256_file(prediction_path)
    checkpoint_hash_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    observed_state = np.load(observed_path)
    observed_state.setflags(write=False)
    refs = reference_worlds(observed_state, float(args.voxel_size))
    replay_context = resolve_replay_context(Path(args.seed0_reference_dir).resolve())
    args.replay_case_json = replay_context["case_json"]
    args.replay_episode_dir = replay_context["episode_dir"]
    args.replay_episode_summary = replay_context["episode_summary"]

    manifest_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    missing_fields: list[dict[str, Any]] = []

    for seed in seeds:
        for formula in formulas:
            tree_dir = output_dir / "raw_trees" / f"seed_{seed:03d}" / formula
            before = time.perf_counter()
            mini_args = tree_args(args, seed, formula, tree_dir)
            stdout_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                mini_summary = run_mini_rrt(mini_args)
            elapsed = time.perf_counter() - before
            row = summarize_tree_result(
                seed=seed,
                formula=formula,
                tree_dir=tree_dir,
                summary=mini_summary,
                missing=missing_fields,
            )
            row["elapsed_s"] = elapsed
            decision_rows.append(row)
            manifest_rows.append(
                {
                    "seed": seed,
                    "formula": formula,
                    "status": row["status"],
                    "tree_dir": str(tree_dir),
                    "selected_child_id": row.get("selected_child_id"),
                    "best_descendant_id": row.get("best_descendant_id"),
                    "elapsed_s": elapsed,
                    "suppressed_mini_rrt_stdout_chars": len(stdout_capture.getvalue()),
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "formula": formula,
                        "selected": row.get("selected_child_id"),
                        "best": row.get("best_descendant_id"),
                        "classification_pending": True,
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
        class_rows.append(classification)

    class_summary = summarize_classifications(class_rows, formulas)
    agreement_rows, agreement_summary = build_agreement_rows(decision_rows, seeds)
    spatial_rows = build_spatial_basin_rows(class_rows, formulas)
    rank_margin_rows = [
        {
            "seed": row["seed"],
            "formula": row["formula"],
            "winner_id": row.get("winner_id"),
            "runner_up_id": row.get("runner_up_id"),
            "winner_value": row.get("value"),
            "runner_up_value": row.get("runner_up_value"),
            "winner_margin": row.get("winner_margin"),
            "normalized_margin": row.get("normalized_margin"),
            "margin_narrow_normalized_lt_0p02": (
                row.get("normalized_margin") is not None and as_float(row.get("normalized_margin")) < 0.02
            ),
            "gain_exp_effective_gain_sc_correlation": row.get("gain_exp_effective_gain_sc_correlation"),
            "value_effective_gain_sc_correlation": row.get("value_effective_gain_sc_correlation"),
            "value_cost_correlation": row.get("value_cost_correlation"),
            "value_inverse_cost_correlation": row.get("value_inverse_cost_correlation"),
        }
        for row in decision_rows
    ]

    plots = make_plots(output_dir, observed_state, decision_rows, class_rows, agreement_rows, formulas)

    seed0_ref = extract_reference_decision(Path(args.seed0_reference_dir).resolve(), "confidence_weighted")
    seed1_ref = extract_reference_decision(Path(args.seed1_reference_dir).resolve(), "confidence_weighted")
    seed0_conf = by_key.get((0, "confidence_weighted"), {})
    seed1_conf = by_key.get((1, "confidence_weighted"), {})
    reference_checks = {
        "seed0_confidence_vs_stage4a65s": compare_to_reference(seed0_conf, seed0_ref) if seed0_conf else {},
        "seed1_confidence_vs_stage4a65t": compare_to_reference(seed1_conf, seed1_ref) if seed1_conf else {},
    }

    confidence_rows = [row for row in class_rows if row["formula"] == "confidence_weighted"]
    confidence_margins = [
        as_float(row.get("normalized_margin"), float("nan"))
        for row in decision_rows
        if row["formula"] == "confidence_weighted"
    ]
    next_step, reason, recommendation_stats = choose_recommendation(class_summary, confidence_margins)
    confidence_class = class_summary["confidence_weighted"]["fractions"]
    cap25_class = class_summary.get("cap25", {}).get("fractions", {})
    raw_class = class_summary.get("raw_count", {}).get("fractions", {})
    raw_behavior = (
        "raw_count exact/spatial behavior matched confidence closely."
        if raw_class.get("spatial_seed0_sc_basin") == confidence_class.get("spatial_seed0_sc_basin")
        else "raw_count was more aggressive or less stable than confidence_weighted by spatial-basin fraction."
    )
    if "raw_count" not in formulas:
        raw_behavior = "raw_count was not requested."

    same_dominates = as_float(confidence_class.get("same_as_measured_for_seed"), 0.0) >= 0.70
    spatial_stable = as_float(confidence_class.get("spatial_seed0_sc_basin"), 0.0) >= 0.70
    seed0_seed1_diff = (
        "score/rank remains seed-sensitive because same-as-measured dominates while the spatial basin remains similar."
        if same_dominates and spatial_stable
        else "tree sampling changes branch rank/score enough that spatial stability is not yet sufficient."
    )
    ready_another_two_frame = next_step == "another start/scene seed repeated gated two-frame smoke"

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
        "source_modified_built": False,
        "coverage_improvement_claim": False,
        "observed_state_sha256_before": observed_hash_before,
        "observed_state_sha256_after": observed_hash_after,
        "prediction_npz_sha256_before": prediction_hash_before,
        "prediction_npz_sha256_after": prediction_hash_after,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "prohibited_artifacts_in_output": prohibited,
    }

    summary = {
        "stage": "Stage 4A-6.5v",
        "output_dir": str(output_dir),
        "seed_count": len(seeds),
        "seeds": seeds,
        "formulas": formulas,
        "inputs": {
            "observed_state": str(observed_path),
            "prediction_npz": str(prediction_path),
            "pose_json": str(pose_path),
            "camera_info_json": str(camera_path),
            "seed0_reference_dir": str(Path(args.seed0_reference_dir).resolve()),
            "seed1_reference_dir": str(Path(args.seed1_reference_dir).resolve()),
            "stage4a65u_dir": str(Path(args.stage4a65u_dir).resolve()),
            "replay_case_json": replay_context["case_json"],
            "replay_episode_dir": replay_context["episode_dir"],
            "replay_episode_summary": replay_context["episode_summary"],
        },
        "source_protected_profile": {
            "short_edge_policy": args.short_edge_policy,
            "crop_min_length_m": float(args.crop_min_length_m),
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
            "tau": float(args.tau),
            "alignment_convention": str(args.alignment_convention),
        },
        "reference": refs,
        "reference_checks": reference_checks,
        "branch_classification_summary": class_summary,
        "agreement_summary": agreement_summary,
        "spatial_basin_summary": spatial_rows,
        "multi_seed_results": {
            "confidence_exact_seed0_sc_fraction": confidence_class.get("exact_seed0_sc"),
            "confidence_spatial_seed0_sc_basin_fraction": confidence_class.get("spatial_seed0_sc_basin"),
            "confidence_same_as_measured_fraction": confidence_class.get("same_as_measured_for_seed"),
            "confidence_measured_but_seed0_sc_basin_fraction": confidence_class.get("measured_but_seed0_sc_basin"),
            "confidence_distinct_sc_branch_fraction": confidence_class.get("distinct_sc_branch"),
            "confidence_local_jitter_fraction": confidence_class.get("local_jitter"),
            "cap25_spatial_seed0_sc_basin_fraction": cap25_class.get("spatial_seed0_sc_basin"),
            "confidence_cap25_agreement_rate": agreement_summary.get(
                "confidence_vs_cap25_exact_grid_agreement_rate"
            ),
            "confidence_vs_measured_agreement_rate": agreement_summary.get(
                "confidence_vs_measured_exact_grid_agreement_rate"
            ),
            "cap25_vs_measured_agreement_rate": agreement_summary.get(
                "cap25_vs_measured_exact_grid_agreement_rate"
            ),
            "raw_count_spatial_seed0_sc_basin_fraction": raw_class.get("spatial_seed0_sc_basin"),
            "raw_count_same_as_measured_fraction": raw_class.get("same_as_measured_for_seed"),
        },
        "stability": {
            **recommendation_stats,
            "confidence_narrow_margin_seeds_normalized_lt_0p02": [
                row["seed"]
                for row in decision_rows
                if row["formula"] == "confidence_weighted"
                and row.get("normalized_margin") is not None
                and as_float(row.get("normalized_margin")) < 0.02
            ],
            "confidence_margin_distribution": percentile_summary(confidence_margins),
            "cost_value_correlation_distribution": percentile_summary(
                [
                    as_float(row.get("value_inverse_cost_correlation"), float("nan"))
                    for row in decision_rows
                    if row["formula"] == "confidence_weighted"
                ]
            ),
            "effective_sc_value_correlation_distribution": percentile_summary(
                [
                    as_float(row.get("value_effective_gain_sc_correlation"), float("nan"))
                    for row in decision_rows
                    if row["formula"] == "confidence_weighted"
                ]
            ),
        },
        "interpretation": {
            "raw_count_behavior": raw_behavior,
            "seed0_seed1_difference": seed0_seed1_diff,
            "seed_robustness": "spatially stable" if spatial_stable else "not spatially stable",
            "branch_basin_stability": "same basin dominates" if spatial_stable else "divergent basins",
            "score_rank_stability": "same-as-measured dominates" if same_dominates else "SC branch rank remains distinct often enough",
            "ready_for_3_frame": False,
            "ready_for_another_start_scene_two_frame": ready_another_two_frame,
            "ready_for_rollout": False,
        },
        "recommended_next_faithful_step": next_step,
        "recommendation_reason": reason,
        "safety": safety,
        "plots": plots,
        "required_outputs": REQUIRED_FILES + REQUIRED_PLOTS,
        "elapsed_s": time.perf_counter() - started,
    }

    write_jsonl(output_dir / "multi_seed_replay_manifest.jsonl", manifest_rows)
    write_csv(
        output_dir / "per_seed_formula_decisions.csv",
        decision_rows,
        [
            "seed",
            "formula",
            "status",
            "selected_child_id",
            "selected_child_grid",
            "best_descendant_id",
            "best_descendant_grid",
            "selected_child_distance_from_root_m",
            "best_descendant_distance_from_root_m",
            "accumulated_gain_exp",
            "accumulated_raw_gain_sc",
            "accumulated_effective_gain_sc",
            "accumulated_hybrid_effective",
            "accumulated_cost",
            "value",
            "runner_up_value",
            "winner_margin",
            "normalized_margin",
            "branch_depth",
            "path_node_ids",
            "nodes_with_effective_gain_sc_positive",
            "classification",
        ],
    )
    save_json(output_dir / "per_seed_formula_decisions.json", decision_rows)
    write_decisions_md(output_dir / "per_seed_formula_decisions.md", decision_rows)
    write_csv(output_dir / "per_seed_rank_margin.csv", rank_margin_rows)
    save_json(output_dir / "per_seed_rank_margin.json", rank_margin_rows)
    write_csv(output_dir / "branch_classification_by_seed.csv", class_rows)
    save_json(output_dir / "branch_classification_by_seed.json", class_rows)
    save_json(output_dir / "branch_classification_summary.json", class_summary)
    write_classification_md(output_dir / "branch_classification_summary.md", class_summary)
    write_csv(output_dir / "confidence_vs_cap25_agreement.csv", agreement_rows)
    save_json(output_dir / "confidence_vs_cap25_agreement.json", agreement_summary | {"rows": agreement_rows})
    write_csv(output_dir / "spatial_basin_summary.csv", spatial_rows)
    save_json(output_dir / "spatial_basin_summary.json", {"rows": spatial_rows})
    write_spatial_md(output_dir / "spatial_basin_summary.md", spatial_rows)
    save_json(output_dir / "missing_fields_report.json", {"missing_fields": missing_fields, "count": len(missing_fields)})
    save_json(output_dir / "stage4a65v_multi_seed_replay_summary.json", summary)
    write_summary_md(output_dir / "stage4a65v_multi_seed_replay_summary.md", summary)
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {next_step}",
                f"- why: {reason}",
                "- still not next: rollout, RL/PPO/BC/IL training, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, coverage-improvement claims, or external source build.",
                "",
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
    parser.add_argument("--seed0_reference_dir", required=True)
    parser.add_argument("--seed1_reference_dir", required=True)
    parser.add_argument("--stage4a65u_dir", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--formulas", default="measured_only,confidence_weighted,cap25,raw_count")
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
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--save_viz", action="store_true")
    return parser


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
