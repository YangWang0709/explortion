#!/usr/bin/env python3
"""Stage 4A-6.5z offline decoupled SC utility sweep.

This runner is intentionally offline. It reads saved Stage 4A-6.5p Frame 2
observed/prediction artifacts and saved Stage 4A-6.5y mini-RRT trees, then
rescored candidate paths with:

    value = gain_exp / cost + lambda * normalized_sc

Prediction stays read-only and only contributes to the information-gain bonus.
The runner never starts Isaac, captures frames, reruns map_predict/SSCNet,
executes actions, trains, writes prediction into observed_state, or changes
traversability/collision/raycast blocking.
"""

from __future__ import annotations

import argparse
import csv
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

from offline_mini_rrt_tree import ROOT_ID, sha256_file, to_jsonable
from run_source_gain_seed_replay import (
    CHECKPOINT_PATH,
    REFERENCE_MEASURED_BEST_GRID,
    REFERENCE_MEASURED_SELECTED_GRID,
    REFERENCE_SC_BEST_GRID,
    REFERENCE_SC_SELECTED_GRID,
    SEED0_SC_COST_REFERENCE,
    SEED0_SC_GAIN_EXP_REFERENCE,
    SEED0_SC_SOURCE_OCC_FREE_REFERENCE,
    as_float,
    build_mapping_report,
    classify_row,
    compute_visible_set,
    euclidean,
    fraction,
    inventory_prediction_npz,
    load_tree_segments,
    path_source_summary,
    path_to_root,
    percentile_summary,
    prediction_masks_for_visible,
    read_json,
    reference_worlds,
    root_child_for_path,
    safe_ratio,
    same_grid,
    source_component_for_segment,
    summarize_classifications,
    write_csv,
    write_inventory_md,
    write_jsonl,
    write_mapping_md,
    write_text,
)
from sim_prediction_layer import SimPredictionLayer


DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65z_decoupled_sc_utility_sweep"
)
DEFAULT_STAGE4A65P_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke"
)
DEFAULT_STAGE4A65V_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65v_multi_seed_offline_replay"
)
DEFAULT_STAGE4A65X_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65x_sc_gain_design_review"
)
DEFAULT_STAGE4A65Y_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65y_source_gain_seed_replay"
)
DEFAULT_OBSERVED_STATE = f"{DEFAULT_STAGE4A65P_DIR}/observed_state_frame002.npy"
DEFAULT_PREDICTION_NPZ = f"{DEFAULT_STAGE4A65P_DIR}/frame002_prediction/global_prediction_layer.npz"
DEFAULT_POSE_JSON = f"{DEFAULT_STAGE4A65P_DIR}/frame002_pose.json"
DEFAULT_CAMERA_INFO_JSON = f"{DEFAULT_STAGE4A65P_DIR}/frame002_camera_info.json"

DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"
DEFAULT_SC_BASES = (
    "source_occ_free,"
    "parent_visible_cleared_source_occ_free,"
    "frontier_local_source_occ_free"
)
DEFAULT_FIXED_LAMBDAS = "0,1,2,4,8,12,16,24,32"
DEFAULT_ADAPTIVE_SCALES = "0.25,0.5,1.0,2.0"

REQUIRED_FILES = [
    "prediction_npz_field_inventory.json",
    "prediction_npz_field_inventory.md",
    "decoupled_sc_mapping_report.json",
    "decoupled_sc_mapping_report.md",
    "normalization_summary_by_seed_basis.csv",
    "normalization_summary_by_seed_basis.json",
    "adaptive_lambda_values.csv",
    "adaptive_lambda_values.json",
    "decoupled_candidate_topk.csv",
    "decoupled_candidate_topk.json",
    "decoupled_sc_sweep_decisions.csv",
    "decoupled_sc_sweep_decisions.json",
    "decoupled_sc_sweep_decisions.md",
    "branch_classification_by_formula_seed.csv",
    "branch_classification_by_formula_seed.json",
    "branch_classification_summary_by_formula.json",
    "branch_classification_summary_by_basis_variant.csv",
    "branch_classification_summary_by_basis_variant.json",
    "lambda_sweep_summary_by_basis_variant.csv",
    "lambda_sweep_summary_by_basis_variant.json",
    "seed0_base_gap_report.json",
    "seed0_base_gap_report.md",
    "safety_summary.json",
    "stage4a65z_decoupled_sc_utility_sweep_summary.json",
    "stage4a65z_decoupled_sc_utility_sweep_summary.md",
    "recommended_next_diagnostic_step.md",
    "fixed_lambda_seed0_sc_basin_fraction.png",
    "fixed_lambda_same_as_measured_fraction.png",
    "fixed_lambda_source_occ_free_heatmap.png",
    "adaptive_lambda_values.png",
    "seed0_source_occ_free_value_components.png",
    "selected_children_fixed_lambda_topdown.png",
]

FORBIDDEN_PATTERNS = [
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


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    values: list[float] = []
    for item in str(raw).split(","):
        text = item.strip()
        if not text:
            continue
        value = float(text)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"lambda value must be finite and non-negative: {item}")
        values.append(value)
    return values


def label_float(value: float) -> str:
    if abs(float(value) - round(float(value))) <= 1.0e-9:
        return str(int(round(float(value))))
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def formula_label(sc_basis: str, lambda_family: str, lambda_label: str) -> str:
    return f"decoupled_{sc_basis}_{lambda_family}_{lambda_label}"


def load_stage4a65y_decisions(stage4a65y_dir: Path) -> list[dict[str, Any]]:
    path = stage4a65y_dir / "per_seed_formula_decisions.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage 4A-6.5y decisions: {path}")
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Stage 4A-6.5y decisions must be a list: {path}")
    return rows


def load_stage4a65y_summary(stage4a65y_dir: Path) -> dict[str, Any]:
    path = stage4a65y_dir / "stage4a65y_source_gain_seed_replay_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing Stage 4A-6.5y summary: {path}")
    return read_json(path)


def load_stage4a65x_summary(stage4a65x_dir: Path) -> dict[str, Any]:
    path = stage4a65x_dir / "stage4a65x_sc_gain_design_review_summary.json"
    return read_json(path)


def resolve_raw_tree_dir(stage4a65y_dir: Path, seed: int) -> Path:
    tree_dir = stage4a65y_dir / "raw_trees" / f"seed_{int(seed):03d}" / "current_raw_count"
    required = tree_dir / "mini_rrt_tree_segments.jsonl"
    if not required.is_file():
        raise FileNotFoundError(f"missing saved raw mini-RRT tree for seed {seed}: {required}")
    return tree_dir


def value_percentile(values: list[float], percentile: float) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return 0.0
    return float(np.percentile(np.asarray(clean, dtype=np.float64), float(percentile)))


def topdown_projection(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == 0, axis=2)] = 1
    image[np.any(observed_state > 0, axis=2)] = 2
    return image


def plot_base_map(ax: plt.Axes, observed_state: np.ndarray) -> None:
    proj = topdown_projection(observed_state)
    colors = np.asarray(
        [
            [0.83, 0.83, 0.83, 1.0],
            [0.91, 0.97, 0.96, 1.0],
            [0.62, 0.16, 0.17, 1.0],
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


def build_node_components(
    tree: dict[str, dict[str, Any]],
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    args: argparse.Namespace,
    visible_cache: dict[tuple[tuple[int, int, int], float, int, int], set[tuple[int, int, int]]],
) -> tuple[
    set[tuple[int, int, int]],
    set[tuple[int, int, int]],
    dict[str, dict[str, Any]],
]:
    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / float(args.voxel_size))))
    num_yaw = max(4, int(math.ceil(32 / max(1, int(args.raycast_stride)))))
    num_pitch = max(3, int(math.ceil(7 / max(1, int(args.raycast_stride)))))
    root_visible = compute_visible_set(tree[ROOT_ID], observed_state, max_range_voxels, num_yaw, num_pitch, visible_cache)
    root_masks = prediction_masks_for_visible(
        observed_state,
        prediction,
        root_visible,
        source_threshold=float(args.ssc_confidence_threshold),
    )
    root_source = set(root_masks["occupied"]) | set(root_masks["free"])

    components: dict[str, dict[str, Any]] = {}
    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        components[node_id] = source_component_for_segment(
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
    return root_visible, root_source, components


def build_candidate_rows(
    *,
    seed: int,
    sc_basis: str,
    tree_dir: Path,
    tree: dict[str, dict[str, Any]],
    node_components: dict[str, dict[str, Any]],
    root_visible: set[tuple[int, int, int]],
    root_source: set[tuple[int, int, int]],
    prediction: SimPredictionLayer,
    normalization_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = tree[ROOT_ID]
    for node_id, segment in tree.items():
        if node_id == ROOT_ID:
            continue
        path_ids = path_to_root(tree, node_id)
        no_root_path = [path_id for path_id in path_ids if path_id != ROOT_ID and path_id in tree]
        selected_child_id = root_child_for_path(path_ids)
        if selected_child_id is None or selected_child_id not in tree:
            continue
        source_summary = path_source_summary(
            path_ids=path_ids,
            tree=tree,
            node_components=node_components,
            root_visible=root_visible,
            root_source=root_source,
            prediction=prediction,
            formula=sc_basis,
        )
        gain_exp = float(sum(as_float(tree[path_id].get("gain_exp")) for path_id in no_root_path))
        raw_sc = float(sum(as_float(tree[path_id].get("gain_sc")) for path_id in no_root_path))
        cost = float(sum(as_float(tree[path_id].get("cost")) for path_id in no_root_path))
        base_value = safe_ratio(gain_exp, cost)
        if base_value is None:
            continue
        rows.append(
            {
                "seed": int(seed),
                "sc_basis": sc_basis,
                "node_id": node_id,
                "selected_child_id": selected_child_id,
                "path_node_ids": no_root_path,
                "branch_depth": len(no_root_path),
                "root_grid": root.get("end_grid"),
                "root_world": root.get("end_world"),
                "selected_child_grid": tree[selected_child_id].get("end_grid"),
                "selected_child_world": tree[selected_child_id].get("end_world"),
                "best_descendant_id": node_id,
                "best_descendant_grid": segment.get("end_grid"),
                "best_descendant_world": segment.get("end_world"),
                "selected_child_distance_from_root_m": euclidean(root.get("end_world"), tree[selected_child_id].get("end_world")),
                "best_descendant_distance_from_root_m": euclidean(root.get("end_world"), segment.get("end_world")),
                "accumulated_gain_exp": gain_exp,
                "accumulated_current_raw_gain_sc": raw_sc,
                "accumulated_formula_effective_sc": float(source_summary["accumulated_formula_effective_sc"]),
                "accumulated_cost": cost,
                "base_exp_value": float(base_value),
                "source_tree_basis_formula": "current_raw_count",
                "tree_dir": str(tree_dir),
                **source_summary,
            }
        )

    if not rows:
        raise RuntimeError(f"no decoupled candidate rows for seed={seed} basis={sc_basis}")

    sc_values = [max(0.0, as_float(row.get("accumulated_formula_effective_sc"), 0.0)) for row in rows]
    if normalization_mode == "max":
        denom = max(sc_values)
    elif normalization_mode == "p95":
        denom = value_percentile(sc_values, 95)
    elif normalization_mode == "p90":
        denom = value_percentile(sc_values, 90)
    else:
        raise ValueError(f"unsupported normalization mode: {normalization_mode}")
    if denom <= 1.0e-9 or not math.isfinite(float(denom)):
        denom = 1.0

    normalized_values: list[float] = []
    for row in rows:
        normalized = max(0.0, as_float(row.get("accumulated_formula_effective_sc"), 0.0)) / float(denom)
        normalized = min(1.0, normalized)
        row["normalized_sc"] = float(normalized)
        row["normalization_denominator"] = float(denom)
        row["normalization_mode"] = normalization_mode
        normalized_values.append(float(normalized))

    base_values = [as_float(row.get("base_exp_value"), float("nan")) for row in rows]
    lambda_base = max(0.0, value_percentile(base_values, 90) - value_percentile(base_values, 50))
    norm_summary = {
        "seed": int(seed),
        "sc_basis": sc_basis,
        "normalization_mode": normalization_mode,
        "normalization_denominator": float(denom),
        "lambda_base_p90_minus_p50_base_exp_value": float(lambda_base),
        "candidate_count": len(rows),
        "base_exp_value": percentile_summary(base_values),
        "raw_sc": percentile_summary(sc_values),
        "normalized_sc": percentile_summary(normalized_values),
    }
    return rows, norm_summary


def select_decoupled_decision(
    *,
    seed: int,
    sc_basis: str,
    lambda_family: str,
    lambda_label: str,
    lambda_value: float,
    lambda_scale: float | None,
    lambda_base: float,
    candidate_rows: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        value = as_float(row.get("base_exp_value"), float("-inf")) + float(lambda_value) * as_float(
            row.get("normalized_sc"), 0.0
        )
        scored = dict(row)
        scored["value"] = float(value)
        scored["decoupled_value"] = float(value)
        scored["lambda_value"] = float(lambda_value)
        scored["lambda_family"] = lambda_family
        scored["lambda_label"] = lambda_label
        scored["lambda_scale"] = lambda_scale
        scored["lambda_base"] = float(lambda_base)
        scored["utility_formula"] = "gain_exp / cost + lambda * normalized_sc"
        scored["sc_inside_cost_division"] = False
        scored["lambda_sc_bonus"] = float(lambda_value) * as_float(row.get("normalized_sc"), 0.0)
        scored_rows.append(scored)

    best_by_child: dict[str, dict[str, Any]] = {}
    for row in scored_rows:
        child_id = str(row["selected_child_id"])
        current = best_by_child.get(child_id)
        if current is None:
            best_by_child[child_id] = row
            continue
        current_key = (
            as_float(current.get("value"), float("-inf")),
            as_float(current.get("base_exp_value"), float("-inf")),
            as_float(current.get("normalized_sc"), float("-inf")),
            str(current.get("node_id")),
        )
        row_key = (
            as_float(row.get("value"), float("-inf")),
            as_float(row.get("base_exp_value"), float("-inf")),
            as_float(row.get("normalized_sc"), float("-inf")),
            str(row.get("node_id")),
        )
        if row_key > current_key:
            best_by_child[child_id] = row

    ranked_children = sorted(
        best_by_child.values(),
        key=lambda row: (
            as_float(row.get("value"), float("-inf")),
            as_float(row.get("base_exp_value"), float("-inf")),
            as_float(row.get("normalized_sc"), float("-inf")),
            str(row.get("selected_child_id")),
        ),
        reverse=True,
    )
    if not ranked_children:
        raise RuntimeError(f"no ranked children for seed={seed} basis={sc_basis} lambda={lambda_label}")

    winner = ranked_children[0]
    runner = ranked_children[1] if len(ranked_children) > 1 else None
    winner_value = as_float(winner.get("value"), float("nan"))
    runner_value = as_float(runner.get("value"), float("nan")) if runner else None
    margin = None if runner_value is None else float(winner_value - runner_value)
    normalized_margin = safe_ratio(margin, abs(winner_value)) if margin is not None else None
    values = [as_float(row.get("value"), float("nan")) for row in scored_rows]
    normalized_sc_values = [as_float(row.get("normalized_sc"), float("nan")) for row in scored_rows]
    base_values = [as_float(row.get("base_exp_value"), float("nan")) for row in scored_rows]
    costs = [as_float(row.get("accumulated_cost"), float("nan")) for row in scored_rows]

    formula = formula_label(sc_basis, lambda_family, lambda_label)
    decision = {
        "seed": int(seed),
        "formula": formula,
        "sc_basis": sc_basis,
        "lambda_family": lambda_family,
        "lambda_label": lambda_label,
        "lambda_value": float(lambda_value),
        "lambda_scale": lambda_scale,
        "lambda_base": float(lambda_base),
        "status": "completed_decoupled_posthoc_rescore",
        "gain_mode": "decoupled_sc_bonus",
        "sc_gain_formula": sc_basis,
        "utility_formula": "gain_exp / cost + lambda * normalized_sc",
        "sc_inside_cost_division": False,
        "source_tree_basis_formula": "current_raw_count",
        "tree_dir": winner.get("tree_dir"),
        "selected_child_id": winner.get("selected_child_id"),
        "selected_child_grid": winner.get("selected_child_grid"),
        "selected_child_world": winner.get("selected_child_world"),
        "best_descendant_id": winner.get("best_descendant_id"),
        "best_descendant_grid": winner.get("best_descendant_grid"),
        "best_descendant_world": winner.get("best_descendant_world"),
        "root_grid": winner.get("root_grid"),
        "root_world": winner.get("root_world"),
        "selected_child_distance_from_root_m": winner.get("selected_child_distance_from_root_m"),
        "best_descendant_distance_from_root_m": winner.get("best_descendant_distance_from_root_m"),
        "accumulated_gain_exp": winner.get("accumulated_gain_exp"),
        "accumulated_current_raw_gain_sc": winner.get("accumulated_current_raw_gain_sc"),
        "accumulated_formula_effective_sc": winner.get("accumulated_formula_effective_sc"),
        "accumulated_cost": winner.get("accumulated_cost"),
        "base_exp_value": winner.get("base_exp_value"),
        "normalized_sc": winner.get("normalized_sc"),
        "normalization_denominator": winner.get("normalization_denominator"),
        "normalization_mode": winner.get("normalization_mode"),
        "lambda_sc_bonus": winner.get("lambda_sc_bonus"),
        "value": winner_value,
        "decoupled_value": winner_value,
        "runner_up_id": runner.get("selected_child_id") if runner else None,
        "runner_up_best_descendant_id": runner.get("best_descendant_id") if runner else None,
        "runner_up_value": runner_value,
        "winner_margin": margin,
        "normalized_margin": normalized_margin,
        "root_child_count": len(ranked_children),
        "branch_depth": winner.get("branch_depth"),
        "path_node_ids": winner.get("path_node_ids"),
        "candidate_path_count": len(scored_rows),
        "value_normalized_sc_correlation": pearson_pairs(values, normalized_sc_values),
        "value_base_exp_correlation": pearson_pairs(values, base_values),
        "value_cost_correlation": pearson_pairs(values, costs),
    }
    for key, value in winner.items():
        if key.startswith("accumulated_") or key in {
            "path_visible_count_sum",
            "path_visible_unique_count",
            "visible_predicted_occ_count",
            "visible_predicted_free_count",
            "visible_predicted_unknown_count",
            "visible_source_occ_free_unique_count",
            "root_visible_overlap_count",
            "root_visible_overlap_fraction",
            "parent_path_visible_overlap_count",
            "parent_path_visible_overlap_fraction",
            "frontier_local_unique_count",
            "frontier_local_fraction",
            "visible_prediction_confidence_sum",
            "visible_prediction_confidence_mean",
            "visible_prediction_confidence_p90",
        }:
            decision.setdefault(key, value)

    top_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked_children[: max(1, int(top_k))], start=1):
        top_rows.append(
            {
                "seed": int(seed),
                "formula": formula,
                "sc_basis": sc_basis,
                "lambda_family": lambda_family,
                "lambda_label": lambda_label,
                "lambda_value": float(lambda_value),
                "rank": rank,
                "selected_child_id": row.get("selected_child_id"),
                "selected_child_grid": row.get("selected_child_grid"),
                "best_descendant_id": row.get("best_descendant_id"),
                "best_descendant_grid": row.get("best_descendant_grid"),
                "base_exp_value": row.get("base_exp_value"),
                "normalized_sc": row.get("normalized_sc"),
                "lambda_sc_bonus": row.get("lambda_sc_bonus"),
                "value": row.get("value"),
                "accumulated_gain_exp": row.get("accumulated_gain_exp"),
                "accumulated_formula_effective_sc": row.get("accumulated_formula_effective_sc"),
                "accumulated_cost": row.get("accumulated_cost"),
            }
        )
    return decision, top_rows


def pearson_pairs(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= 1.0e-9 or float(np.std(y_arr)) <= 1.0e-9:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def group_summary_rows(decisions: list[dict[str, Any]], class_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    class_by_formula = {row["formula"]: row for row in class_rows}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        grouped[(str(row["sc_basis"]), str(row["lambda_family"]), str(row["lambda_label"]))].append(row)

    rows: list[dict[str, Any]] = []
    for (basis, family, label), subset in sorted(grouped.items()):
        formulas = {str(row["formula"]) for row in subset}
        cls = [class_by_formula[formula] for formula in formulas if formula in class_by_formula]
        rows.append(
            {
                "sc_basis": basis,
                "lambda_family": family,
                "lambda_label": label,
                "lambda_value_min": min(as_float(row.get("lambda_value")) for row in subset),
                "lambda_value_max": max(as_float(row.get("lambda_value")) for row in subset),
                "seed_count": len(subset),
                "spatial_seed0_sc_basin_fraction": fraction(cls, "spatial_seed0_sc_basin"),
                "same_as_measured_fraction": fraction(cls, "same_as_measured_for_seed"),
                "distinct_sc_branch_fraction": fraction(cls, "distinct_sc_branch"),
                "avoids_short_local_sc_fraction": fraction(cls, "avoids_short_local_sc"),
                "winner_base_exp_value": percentile_summary(
                    [as_float(row.get("base_exp_value"), float("nan")) for row in subset]
                ),
                "winner_normalized_sc": percentile_summary(
                    [as_float(row.get("normalized_sc"), float("nan")) for row in subset]
                ),
                "winner_lambda_sc_bonus": percentile_summary(
                    [as_float(row.get("lambda_sc_bonus"), float("nan")) for row in subset]
                ),
                "winner_value": percentile_summary([as_float(row.get("value"), float("nan")) for row in subset]),
                "winner_gain_exp": percentile_summary(
                    [as_float(row.get("accumulated_gain_exp"), float("nan")) for row in subset]
                ),
                "winner_sc": percentile_summary(
                    [as_float(row.get("accumulated_formula_effective_sc"), float("nan")) for row in subset]
                ),
                "winner_cost": percentile_summary(
                    [as_float(row.get("accumulated_cost"), float("nan")) for row in subset]
                ),
            }
        )
    return rows


def build_seed0_gap_report(stage4a65y_decisions: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(int(row["seed"]), str(row["formula"])): row for row in stage4a65y_decisions}
    measured = by_key[(0, "measured_only")]
    source = by_key[(0, "source_occ_free")]
    measured_base = safe_ratio(
        as_float(measured.get("accumulated_gain_exp")),
        as_float(measured.get("accumulated_cost")),
    )
    source_base = safe_ratio(
        as_float(source.get("accumulated_gain_exp")),
        as_float(source.get("accumulated_cost")),
    )
    return {
        "measured_branch": {
            "selected_child_id": measured.get("selected_child_id"),
            "best_descendant_id": measured.get("best_descendant_id"),
            "gain_exp": measured.get("accumulated_gain_exp"),
            "stage4a65y_row_source_occ_free_sc": measured.get("accumulated_source_occ_free"),
            "stage4a65x_context_source_occ_free_sc": 569.0,
            "cost": measured.get("accumulated_cost"),
            "base_exp_value": measured_base,
        },
        "seed0_sc_branch": {
            "selected_child_id": source.get("selected_child_id"),
            "best_descendant_id": source.get("best_descendant_id"),
            "gain_exp": source.get("accumulated_gain_exp"),
            "stage4a65y_row_source_occ_free_sc": source.get("accumulated_source_occ_free"),
            "stage4a65x_context_source_occ_free_sc": SEED0_SC_SOURCE_OCC_FREE_REFERENCE,
            "cost": source.get("accumulated_cost"),
            "base_exp_value": source_base,
        },
        "base_value_gap_measured_minus_sc": None
        if measured_base is None or source_base is None
        else float(measured_base - source_base),
        "context_reference": {
            "expected_measured_gain_exp": 323.0,
            "expected_measured_cost": 2.315392939101747,
            "expected_sc_gain_exp": SEED0_SC_GAIN_EXP_REFERENCE,
            "expected_sc_source_occ_free": SEED0_SC_SOURCE_OCC_FREE_REFERENCE,
            "expected_sc_cost": SEED0_SC_COST_REFERENCE,
            "note": (
                "The Stage 4A-6.5x design review recorded source OCC+FREE SC 135. "
                "The Stage 4A-6.5y saved posthoc row can carry a different recomputed "
                "path field because this sweep reads the saved raw tree artifacts."
            ),
        },
    }


def write_seed0_gap_md(path: Path, report: dict[str, Any]) -> None:
    measured = report["measured_branch"]
    sc = report["seed0_sc_branch"]
    lines = [
        "# Seed0 Base Value Gap",
        "",
        "| branch | selected -> best | gain_exp | source OCC+FREE SC | cost | gain_exp / cost |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| measured | `{measured['selected_child_id']} -> {measured['best_descendant_id']}` | "
            f"{measured['gain_exp']} | {measured['stage4a65x_context_source_occ_free_sc']} "
            f"(6.5y row `{measured['stage4a65y_row_source_occ_free_sc']}`) | "
            f"{measured['cost']} | {measured['base_exp_value']} |"
        ),
        (
            f"| seed0 SC | `{sc['selected_child_id']} -> {sc['best_descendant_id']}` | "
            f"{sc['gain_exp']} | {sc['stage4a65x_context_source_occ_free_sc']} "
            f"(6.5y row `{sc['stage4a65y_row_source_occ_free_sc']}`) | "
            f"{sc['cost']} | {sc['base_exp_value']} |"
        ),
        "",
        f"- base value gap measured minus SC: `{report['base_value_gap_measured_minus_sc']}`",
        "- This is why lambda values above the small 0.25-4 range are swept.",
    ]
    write_text(path, "\n".join(lines))


def write_decisions_md(path: Path, rows: list[dict[str, Any]]) -> None:
    preview = rows[:60]
    lines = [
        "# Decoupled SC Sweep Decisions",
        "",
        "| seed | basis | lambda | selected -> best | base | norm_sc | bonus | value | class |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in preview:
        lines.append(
            f"| {row['seed']} | `{row['sc_basis']}` | {row['lambda_value']} | "
            f"`{row['selected_child_id']} -> {row['best_descendant_id']}` | "
            f"{row.get('base_exp_value')} | {row.get('normalized_sc')} | "
            f"{row.get('lambda_sc_bonus')} | {row.get('value')} | `{row.get('classification')}` |"
        )
    if len(rows) > len(preview):
        lines.append("")
        lines.append(f"Only the first {len(preview)} of {len(rows)} rows are shown here; see CSV/JSON for all rows.")
    write_text(path, "\n".join(lines))


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    results = summary["key_results"]
    safety = summary["safety"]
    lines = [
        "# Stage 4A-6.5z Decoupled SC Utility Sweep Summary",
        "",
        f"- seeds: `{summary['seeds']}`",
        f"- SC bases: `{summary['sc_bases']}`",
        f"- fixed lambdas: `{summary['fixed_lambdas']}`",
        f"- adaptive scales: `{summary['adaptive_lambda_scales']}`",
        f"- decision rows: `{summary['decision_row_count']}`",
        f"- lambda formula: `{summary['utility_formula']}`",
        f"- normalized_sc definition: `{summary['normalized_sc_definition']}`",
        f"- diagnostic/source-faithful status: `{summary['diagnostic_status']}`",
        "",
        "## Key Results",
        "",
        f"- seed0 fixed lambda 0 source_occ_free: `{results.get('seed0_source_occ_free_fixed_lambda0')}`",
        f"- source_occ_free fixed lambda 0 spatial seed0 SC basin fraction: `{results.get('source_occ_free_fixed_lambda0_spatial_seed0_sc_basin_fraction')}`",
        f"- source_occ_free fixed lambda 0 same-as-measured fraction: `{results.get('source_occ_free_fixed_lambda0_same_as_measured_fraction')}`",
        f"- minimum source_occ_free fixed spatial basin fraction: `{results.get('source_occ_free_fixed_min_spatial_seed0_sc_basin_fraction')}`",
        f"- runtime smoke readiness: `{summary['answers']['runtime_smoke_readiness']}`",
        f"- rollout readiness: `{summary['answers']['rollout_readiness']}`",
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
        "two_frame_runtime",
        "rollout",
        "training_or_rl",
        "observed_state_modified",
        "prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_scoring",
        "coverage_improvement_claim",
    ]:
        lines.append(f"- {key}: `{safety[key]}`")
    write_text(path, "\n".join(lines))


def plot_fixed_fraction(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    fixed_lambdas: list[float],
    key: str,
    filename: str,
    ylabel: str,
) -> str:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    bases = sorted({row["sc_basis"] for row in summary_rows if row["lambda_family"] == "fixed"})
    for basis in bases:
        values = []
        for lam in fixed_lambdas:
            label = label_float(lam)
            row = next(
                (
                    item
                    for item in summary_rows
                    if item["sc_basis"] == basis and item["lambda_family"] == "fixed" and item["lambda_label"] == label
                ),
                None,
            )
            values.append(float(row.get(key) or 0.0) if row else 0.0)
        ax.plot(fixed_lambdas, values, marker="o", linewidth=2, label=basis)
    ax.set_xlabel("lambda")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    path = output_dir / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_heatmap(output_dir: Path, summary_rows: list[dict[str, Any]], fixed_lambdas: list[float]) -> str:
    bases = sorted({row["sc_basis"] for row in summary_rows if row["lambda_family"] == "fixed"})
    data = np.zeros((len(bases), len(fixed_lambdas)), dtype=np.float64)
    for i, basis in enumerate(bases):
        for j, lam in enumerate(fixed_lambdas):
            label = label_float(lam)
            row = next(
                (
                    item
                    for item in summary_rows
                    if item["sc_basis"] == basis and item["lambda_family"] == "fixed" and item["lambda_label"] == label
                ),
                None,
            )
            data[i, j] = float(row.get("spatial_seed0_sc_basin_fraction") or 0.0) if row else 0.0
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    im = ax.imshow(data, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(fixed_lambdas)))
    ax.set_xticklabels([label_float(v) for v in fixed_lambdas])
    ax.set_yticks(range(len(bases)))
    ax.set_yticklabels(bases)
    ax.set_xlabel("fixed lambda")
    ax.set_title("Spatial seed0 SC basin fraction")
    fig.colorbar(im, ax=ax)
    path = output_dir / "fixed_lambda_source_occ_free_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_adaptive_lambdas(output_dir: Path, adaptive_rows: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in adaptive_rows:
        grouped[(str(row["sc_basis"]), str(row["adaptive_scale_label"]))].append(as_float(row.get("lambda_value")))
    labels = [f"{basis}\n{scale}" for basis, scale in grouped]
    values = [statistics.fmean(vals) if vals else 0.0 for vals in grouped.values()]
    fig, ax = plt.subplots(figsize=(11, 4.5), constrained_layout=True)
    ax.bar(range(len(labels)), values, color="#376a8a")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean adaptive lambda")
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "adaptive_lambda_values.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_seed0_components(output_dir: Path, decisions: list[dict[str, Any]], fixed_lambdas: list[float]) -> str:
    rows = [
        row
        for row in decisions
        if int(row["seed"]) == 0
        and row["sc_basis"] == "source_occ_free"
        and row["lambda_family"] == "fixed"
    ]
    by_label = {row["lambda_label"]: row for row in rows}
    lambdas = []
    base = []
    bonus = []
    total = []
    for lam in fixed_lambdas:
        row = by_label.get(label_float(lam))
        if row is None:
            continue
        lambdas.append(lam)
        base.append(as_float(row.get("base_exp_value")))
        bonus.append(as_float(row.get("lambda_sc_bonus")))
        total.append(as_float(row.get("value")))
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.plot(lambdas, base, marker="o", label="gain_exp / cost")
    ax.plot(lambdas, bonus, marker="o", label="lambda * normalized_sc")
    ax.plot(lambdas, total, marker="o", label="total value")
    ax.set_xlabel("fixed lambda")
    ax.set_ylabel("seed0 winning branch value component")
    ax.grid(alpha=0.25)
    ax.legend()
    path = output_dir / "seed0_source_occ_free_value_components.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def plot_selected_topdown(
    output_dir: Path,
    observed_state: np.ndarray,
    decisions: list[dict[str, Any]],
    fixed_lambdas: list[float],
) -> str:
    keep_lambdas = {label_float(value) for value in fixed_lambdas if value in {0.0, 8.0, 16.0, 32.0}}
    rows = [
        row
        for row in decisions
        if row["lambda_family"] == "fixed"
        and row["lambda_label"] in keep_lambdas
        and row["sc_basis"] == "source_occ_free"
    ]
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    plot_base_map(ax, observed_state)
    colors = plt.get_cmap("tab10")
    for idx, label in enumerate(sorted(keep_lambdas, key=lambda item: float(item.replace("p", ".")))):
        subset = [row for row in rows if row["lambda_label"] == label]
        points = [grid_xy(row.get("selected_child_grid")) for row in subset]
        points = [point for point in points if point is not None]
        if not points:
            continue
        arr = np.asarray(points, dtype=np.float64)
        ax.scatter(arr[:, 0], arr[:, 1], s=42, color=colors(idx), label=f"lambda {label}", alpha=0.9)
    refs = [
        (REFERENCE_MEASURED_SELECTED_GRID, "measured ref", "s", "#1f2937"),
        (REFERENCE_SC_SELECTED_GRID, "seed0 SC ref", "X", "#dc2626"),
    ]
    for grid, label, marker, color in refs:
        point = grid_xy(grid)
        if point:
            ax.scatter([point[0]], [point[1]], s=110, marker=marker, color=color, label=label)
    ax.set_title("source_occ_free selected children, fixed lambdas")
    ax.legend(fontsize=8, loc="upper right")
    path = output_dir / "selected_children_fixed_lambda_topdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def make_plots(
    output_dir: Path,
    observed_state: np.ndarray,
    decisions: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    adaptive_rows: list[dict[str, Any]],
    fixed_lambdas: list[float],
) -> dict[str, str]:
    plots = {
        "fixed_lambda_seed0_sc_basin_fraction": plot_fixed_fraction(
            output_dir,
            summary_rows,
            fixed_lambdas,
            "spatial_seed0_sc_basin_fraction",
            "fixed_lambda_seed0_sc_basin_fraction.png",
            "spatial seed0 SC basin fraction",
        ),
        "fixed_lambda_same_as_measured_fraction": plot_fixed_fraction(
            output_dir,
            summary_rows,
            fixed_lambdas,
            "same_as_measured_fraction",
            "fixed_lambda_same_as_measured_fraction.png",
            "same-as-measured fraction",
        ),
        "fixed_lambda_source_occ_free_heatmap": plot_heatmap(output_dir, summary_rows, fixed_lambdas),
        "adaptive_lambda_values": plot_adaptive_lambdas(output_dir, adaptive_rows),
        "seed0_source_occ_free_value_components": plot_seed0_components(output_dir, decisions, fixed_lambdas),
        "selected_children_fixed_lambda_topdown": plot_selected_topdown(output_dir, observed_state, decisions, fixed_lambdas),
    }
    return plots


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_ints(args.seeds)
    sc_bases = [item.strip() for item in str(args.sc_bases).split(",") if item.strip()]
    fixed_lambdas = parse_floats(args.fixed_lambdas)
    adaptive_scales = parse_floats(args.adaptive_lambda_scales)
    if fixed_lambdas != [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]:
        raise ValueError("Stage 4A-6.5z requires fixed lambdas: 0,1,2,4,8,12,16,24,32")
    if adaptive_scales != [0.25, 0.5, 1.0, 2.0]:
        raise ValueError("Stage 4A-6.5z requires adaptive scales: 0.25,0.5,1.0,2.0")

    observed_path = Path(args.observed_state).resolve()
    prediction_path = Path(args.prediction_npz).resolve()
    pose_path = Path(args.pose_json).resolve()
    camera_path = Path(args.camera_info_json).resolve()
    stage4a65p_dir = Path(args.stage4a65p_dir).resolve()
    stage4a65v_dir = Path(args.stage4a65v_dir).resolve()
    stage4a65x_dir = Path(args.stage4a65x_dir).resolve()
    stage4a65y_dir = Path(args.stage4a65y_dir).resolve()
    for path in (observed_path, prediction_path, pose_path, camera_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    observed_hash_before = sha256_file(observed_path)
    prediction_hash_before = sha256_file(prediction_path)
    pose_hash_before = sha256_file(pose_path)
    camera_hash_before = sha256_file(camera_path)
    checkpoint_hash_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    observed_state = np.load(observed_path)
    observed_state.setflags(write=False)
    prediction = SimPredictionLayer.from_npz(prediction_path)
    if tuple(prediction.shape()) != tuple(observed_state.shape):
        raise ValueError(f"prediction shape {prediction.shape()} != observed_state shape {observed_state.shape}")

    inventory = inventory_prediction_npz(prediction_path)
    mapping_report = build_mapping_report(inventory, args)
    mapping_report["decoupled_formula"] = {
        "value": "gain_exp / cost + lambda * normalized_sc",
        "normalized_sc": f"SC path bonus divided by per-seed/per-basis {args.normalization_mode} across valid paths, clipped to [0, 1].",
        "sc_inside_cost_division": False,
        "diagnostic_status": "diagnostic/engineering utility review, not source-faithful planner behavior",
    }
    save_json(output_dir / "prediction_npz_field_inventory.json", inventory)
    write_inventory_md(output_dir / "prediction_npz_field_inventory.md", inventory)
    save_json(output_dir / "decoupled_sc_mapping_report.json", mapping_report)
    write_mapping_md(output_dir / "decoupled_sc_mapping_report.md", mapping_report)

    stage4a65y_summary = load_stage4a65y_summary(stage4a65y_dir)
    stage4a65x_summary = load_stage4a65x_summary(stage4a65x_dir)
    stage4a65y_decisions = load_stage4a65y_decisions(stage4a65y_dir)
    y_by_key = {(int(row["seed"]), str(row["formula"])): row for row in stage4a65y_decisions}
    refs = reference_worlds(observed_state, float(args.voxel_size))

    decision_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    adaptive_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    visible_cache: dict[tuple[tuple[int, int, int], float, int, int], set[tuple[int, int, int]]] = {}

    for seed in seeds:
        measured = y_by_key.get((seed, "measured_only"))
        if measured is None:
            raise ValueError(f"missing Stage 4A-6.5y measured_only row for seed {seed}")
        tree_dir = resolve_raw_tree_dir(stage4a65y_dir, seed)
        before_seed = time.perf_counter()
        tree = load_tree_segments(tree_dir / "mini_rrt_tree_segments.jsonl")
        root_visible, root_source, components = build_node_components(
            tree,
            observed_state,
            prediction,
            args,
            visible_cache,
        )
        manifest_rows.append(
            {
                "seed": seed,
                "stage": "load_saved_6p5y_raw_tree_and_components",
                "tree_dir": str(tree_dir),
                "node_count": len(tree),
                "elapsed_s": time.perf_counter() - before_seed,
            }
        )
        for sc_basis in sc_bases:
            before_basis = time.perf_counter()
            candidates, norm_summary = build_candidate_rows(
                seed=seed,
                sc_basis=sc_basis,
                tree_dir=tree_dir,
                tree=tree,
                node_components=components,
                root_visible=root_visible,
                root_source=root_source,
                prediction=prediction,
                normalization_mode=str(args.normalization_mode),
            )
            normalization_rows.append(norm_summary)
            lambda_base = as_float(norm_summary["lambda_base_p90_minus_p50_base_exp_value"])
            lambda_specs: list[tuple[str, str, float, float | None]] = [
                ("fixed", label_float(value), value, None) for value in fixed_lambdas
            ]
            for scale in adaptive_scales:
                lambda_value = float(scale) * lambda_base
                lambda_specs.append(("adaptive", f"{label_float(scale)}x", lambda_value, float(scale)))
                adaptive_rows.append(
                    {
                        "seed": seed,
                        "sc_basis": sc_basis,
                        "adaptive_scale": float(scale),
                        "adaptive_scale_label": f"{label_float(scale)}x",
                        "lambda_base": float(lambda_base),
                        "lambda_value": float(lambda_value),
                    }
                )

            for family, label, lambda_value, scale in lambda_specs:
                decision, top_rows = select_decoupled_decision(
                    seed=seed,
                    sc_basis=sc_basis,
                    lambda_family=family,
                    lambda_label=label,
                    lambda_value=float(lambda_value),
                    lambda_scale=scale,
                    lambda_base=float(lambda_base),
                    candidate_rows=candidates,
                    top_k=int(args.top_k),
                )
                classification = classify_row(decision, measured, refs)
                decision["classification"] = classification["primary_classification"]
                decision["selected_to_seed0_sc_reference_m"] = classification["selected_to_seed0_sc_reference_m"]
                decision["best_to_seed0_sc_reference_m"] = classification["best_to_seed0_sc_reference_m"]
                decision["selected_to_same_seed_measured_m"] = classification["selected_to_same_seed_measured_m"]
                decision["avoids_short_local_sc"] = classification["avoids_short_local_sc"]
                decision["source_measured_preferred"] = classification["source_measured_preferred"]
                decision_rows.append(decision)
                class_rows.append(classification)
                topk_rows.extend(top_rows)
            manifest_rows.append(
                {
                    "seed": seed,
                    "sc_basis": sc_basis,
                    "stage": "decoupled_rescore_completed",
                    "candidate_count": len(candidates),
                    "lambda_base": lambda_base,
                    "elapsed_s": time.perf_counter() - before_basis,
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "sc_basis": sc_basis,
                        "candidate_count": len(candidates),
                        "lambda_base": round(lambda_base, 6),
                        "elapsed_s": round(time.perf_counter() - before_basis, 3),
                    },
                    sort_keys=True,
                )
            )

    formulas = [row["formula"] for row in decision_rows]
    class_summary = summarize_classifications(class_rows, formulas)
    grouped_summary = group_summary_rows(decision_rows, class_rows)
    plots = make_plots(output_dir, observed_state, decision_rows, grouped_summary, adaptive_rows, fixed_lambdas)
    seed0_gap = build_seed0_gap_report(stage4a65y_decisions)

    grouped_by_key = {
        (row["sc_basis"], row["lambda_family"], row["lambda_label"]): row for row in grouped_summary
    }
    source_lambda0 = grouped_by_key.get(("source_occ_free", "fixed", "0"), {})
    source_fixed = [row for row in grouped_summary if row["sc_basis"] == "source_occ_free" and row["lambda_family"] == "fixed"]
    source_min_spatial = min(
        [as_float(row.get("spatial_seed0_sc_basin_fraction"), float("inf")) for row in source_fixed],
        default=None,
    )
    seed0_source_l0 = next(
        (
            row
            for row in decision_rows
            if row["seed"] == 0
            and row["sc_basis"] == "source_occ_free"
            and row["lambda_family"] == "fixed"
            and row["lambda_label"] == "0"
        ),
        {},
    )

    observed_hash_after = sha256_file(observed_path)
    prediction_hash_after = sha256_file(prediction_path)
    pose_hash_after = sha256_file(pose_path)
    camera_hash_after = sha256_file(camera_path)
    checkpoint_hash_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    prohibited = {
        pattern: sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
        for pattern in FORBIDDEN_PATTERNS
    }
    prohibited = {key: value for key, value in prohibited.items() if value}
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
        "checkpoint_modified": checkpoint_hash_before != checkpoint_hash_after,
        "observed_state_modified": observed_hash_before != observed_hash_after,
        "prediction_npz_modified": prediction_hash_before != prediction_hash_after,
        "pose_json_modified": pose_hash_before != pose_hash_after,
        "camera_info_modified": camera_hash_before != camera_hash_after,
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
        "pose_json_sha256_before": pose_hash_before,
        "pose_json_sha256_after": pose_hash_after,
        "camera_info_sha256_before": camera_hash_before,
        "camera_info_sha256_after": camera_hash_after,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "prohibited_artifacts_in_output": prohibited,
    }

    next_step = "inspect decoupled sweep tables offline; do not proceed to runtime smoke or rollout from this diagnostic alone"
    summary = {
        "stage": "Stage 4A-6.5z",
        "output_dir": str(output_dir),
        "diagnostic_status": "diagnostic/engineering utility review, not source-faithful",
        "utility_formula": "value = gain_exp / cost + lambda * normalized_sc",
        "normalized_sc_definition": (
            f"SC path bonus divided by per-seed/per-basis {args.normalization_mode} across valid raw-tree paths, clipped to [0,1]"
        ),
        "seed_count": len(seeds),
        "seeds": seeds,
        "sc_bases": sc_bases,
        "fixed_lambdas": fixed_lambdas,
        "adaptive_lambda_scales": adaptive_scales,
        "decision_row_count": len(decision_rows),
        "candidate_topk_row_count": len(topk_rows),
        "inputs": {
            "observed_state": str(observed_path),
            "prediction_npz": str(prediction_path),
            "pose_json": str(pose_path),
            "camera_info_json": str(camera_path),
            "stage4a65p_dir": str(stage4a65p_dir),
            "stage4a65v_dir": str(stage4a65v_dir),
            "stage4a65x_dir": str(stage4a65x_dir),
            "stage4a65y_dir": str(stage4a65y_dir),
        },
        "parameters": {
            "voxel_size": float(args.voxel_size),
            "raycast_stride": int(args.raycast_stride),
            "max_ray_length_m": float(args.max_ray_length_m),
            "ssc_confidence_threshold": float(args.ssc_confidence_threshold),
            "normalization_mode": str(args.normalization_mode),
            "top_k": int(args.top_k),
            "max_workers_requested": int(args.max_workers),
        },
        "context_confirmation": {
            "stage4a65x_complete": bool(stage4a65x_summary),
            "stage4a65y_complete": bool(stage4a65y_summary),
            "stage4a65y_decision_rows": len(stage4a65y_decisions),
            "stage4a65y_seed0_confidence_reproduced": bool(
                same_grid(
                    y_by_key.get((0, "current_confidence_weighted"), {}).get("selected_child_grid"),
                    REFERENCE_SC_SELECTED_GRID,
                )
                and same_grid(
                    y_by_key.get((0, "current_confidence_weighted"), {}).get("best_descendant_grid"),
                    REFERENCE_SC_BEST_GRID,
                )
            ),
            "stage4a65y_source_occ_free_seed0_selected": y_by_key.get((0, "source_occ_free"), {}).get("selected_child_id"),
            "stage4a65y_source_occ_free_spatial_basin_fraction": stage4a65y_summary.get("key_results", {}).get(
                "source_occ_free_spatial_seed0_sc_basin_fraction"
            ),
            "runtime_smoke_readiness_before_this_task": False,
            "rollout_readiness_before_this_task": False,
        },
        "seed0_base_gap_report": seed0_gap,
        "normalization_summary": normalization_rows,
        "adaptive_lambda_values": adaptive_rows,
        "branch_classification_summary_by_formula": class_summary,
        "lambda_sweep_summary_by_basis_variant": grouped_summary,
        "key_results": {
            "seed0_source_occ_free_fixed_lambda0": {
                "selected_child_id": seed0_source_l0.get("selected_child_id"),
                "best_descendant_id": seed0_source_l0.get("best_descendant_id"),
                "selected_child_grid": seed0_source_l0.get("selected_child_grid"),
                "best_descendant_grid": seed0_source_l0.get("best_descendant_grid"),
                "classification": seed0_source_l0.get("classification"),
                "base_exp_value": seed0_source_l0.get("base_exp_value"),
                "normalized_sc": seed0_source_l0.get("normalized_sc"),
                "lambda_sc_bonus": seed0_source_l0.get("lambda_sc_bonus"),
                "value": seed0_source_l0.get("value"),
            },
            "source_occ_free_fixed_lambda0_spatial_seed0_sc_basin_fraction": source_lambda0.get(
                "spatial_seed0_sc_basin_fraction"
            ),
            "source_occ_free_fixed_lambda0_same_as_measured_fraction": source_lambda0.get(
                "same_as_measured_fraction"
            ),
            "source_occ_free_fixed_min_spatial_seed0_sc_basin_fraction": source_min_spatial,
        },
        "answers": {
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
            "source_faithful": False,
            "coverage_improvement_claimed": False,
        },
        "recommended_next_diagnostic_step": next_step,
        "safety": safety,
        "plots": plots,
        "required_outputs": REQUIRED_FILES,
        "elapsed_s": time.perf_counter() - started,
    }

    write_jsonl(output_dir / "decoupled_sc_sweep_manifest.jsonl", manifest_rows)
    write_csv(output_dir / "normalization_summary_by_seed_basis.csv", normalization_rows)
    save_json(output_dir / "normalization_summary_by_seed_basis.json", normalization_rows)
    write_csv(output_dir / "adaptive_lambda_values.csv", adaptive_rows)
    save_json(output_dir / "adaptive_lambda_values.json", adaptive_rows)
    write_csv(output_dir / "decoupled_candidate_topk.csv", topk_rows)
    save_json(output_dir / "decoupled_candidate_topk.json", topk_rows)
    write_csv(output_dir / "decoupled_sc_sweep_decisions.csv", decision_rows)
    save_json(output_dir / "decoupled_sc_sweep_decisions.json", decision_rows)
    write_decisions_md(output_dir / "decoupled_sc_sweep_decisions.md", decision_rows)
    write_csv(output_dir / "branch_classification_by_formula_seed.csv", class_rows)
    save_json(output_dir / "branch_classification_by_formula_seed.json", class_rows)
    save_json(output_dir / "branch_classification_summary_by_formula.json", class_summary)
    write_csv(output_dir / "branch_classification_summary_by_basis_variant.csv", grouped_summary)
    save_json(output_dir / "branch_classification_summary_by_basis_variant.json", grouped_summary)
    write_csv(output_dir / "lambda_sweep_summary_by_basis_variant.csv", grouped_summary)
    save_json(output_dir / "lambda_sweep_summary_by_basis_variant.json", grouped_summary)
    save_json(output_dir / "seed0_base_gap_report.json", seed0_gap)
    write_seed0_gap_md(output_dir / "seed0_base_gap_report.md", seed0_gap)
    save_json(output_dir / "safety_summary.json", safety)
    save_json(output_dir / "stage4a65z_decoupled_sc_utility_sweep_summary.json", summary)
    write_summary_md(output_dir / "stage4a65z_decoupled_sc_utility_sweep_summary.md", summary)
    write_text(
        output_dir / "recommended_next_diagnostic_step.md",
        "\n".join(
            [
                "# Recommended Next Diagnostic Step",
                "",
                f"- next small task: {next_step}",
                "- still not next: runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL training, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, coverage-improvement claims, or external source build.",
            ]
        ),
    )

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed_state", default=DEFAULT_OBSERVED_STATE)
    parser.add_argument("--prediction_npz", default=DEFAULT_PREDICTION_NPZ)
    parser.add_argument("--pose_json", default=DEFAULT_POSE_JSON)
    parser.add_argument("--camera_info_json", default=DEFAULT_CAMERA_INFO_JSON)
    parser.add_argument("--stage4a65p_dir", default=DEFAULT_STAGE4A65P_DIR)
    parser.add_argument("--stage4a65v_dir", default=DEFAULT_STAGE4A65V_DIR)
    parser.add_argument("--stage4a65x_dir", default=DEFAULT_STAGE4A65X_DIR)
    parser.add_argument("--stage4a65y_dir", default=DEFAULT_STAGE4A65Y_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--sc_bases", default=DEFAULT_SC_BASES)
    parser.add_argument("--fixed_lambdas", default=DEFAULT_FIXED_LAMBDAS)
    parser.add_argument("--adaptive_lambda_scales", default=DEFAULT_ADAPTIVE_SCALES)
    parser.add_argument("--normalization_mode", choices=["max", "p95", "p90"], default="max")
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--ssc_confidence_threshold", type=float, default=0.05)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=8)
    return parser


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
