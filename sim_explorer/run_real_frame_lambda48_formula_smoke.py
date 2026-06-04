#!/usr/bin/env python3
"""Stage 4A-6.5ad real saved-frame lambda48 formula smoke.

This runner is offline-only. It reads one saved Stage 4A-6.5p real
medium_three_rooms Frame2 observed map and prediction NPZ, builds source-
protected mini-RRT trees in memory, and replays one-step tree decisions with:

    value = gain_exp / cost + 48 * minmax(source_occ_free)

Prediction remains information-gain-only. This script does not start Isaac,
capture frames, rerun map_predict, run SSCNet inference, execute selected
actions, write predictions into observed_state, or use prediction for
traversability, collision, edge validity, or ray blocking.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import (
    ROOT_ID,
    build_mini_rrt_tree,
    segment_record,
)
from run_synthetic_map_predict_calibration_smoke import (
    EPS,
    load_tree,
    make_config,
    path_candidate_rows,
    precompute_segment_prediction_arrays,
)
from sim_paper_expert import EmptyPredictionLayer, world_to_grid
from sim_prediction_layer import SimPredictionLayer


DEFAULT_STAGE4A65P_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke"
)
DEFAULT_STAGE4A65AC_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ac_saved_frame_lambda48_formula_smoke"
)
DEFAULT_STAGE4A65Z_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65z_decoupled_sc_utility_sweep"
)
DEFAULT_STAGE4A65Y_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65y_source_gain_seed_replay"
)
DEFAULT_STAGE4A65Z1_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65z1_decoupled_signal_strength_diagnosis"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ad_real_frame_lambda48_formula_smoke"
)
DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)

REQUIRED_PLOTS = [
    "selected_branches_topdown_real_frame.png",
    "measured_vs_lambda48_topdown.png",
    "branch_classification_bar.png",
    "value_components_lambda48_real_frame.png",
    "source_occ_free_rank_by_mode.png",
    "low_cost_artifact_by_mode.png",
    "margin_by_mode.png",
    "prior_sc_basin_distance_by_mode.png",
]

REFERENCE_BRANCHES = {
    "measured_reference": {
        "label": "measured_reference",
        "selected_child_id": "n0001",
        "selected_child_grid": [17, 16, 11],
        "best_descendant_id": "n0112",
        "best_descendant_grid": [8, 27, 11],
    },
    "prior_low_cost_sc_reference": {
        "label": "prior_low_cost_sc_reference",
        "selected_child_id": "n0127",
        "selected_child_grid": [11, 15, 11],
        "best_descendant_id": "n0162",
        "best_descendant_grid": [14, 15, 11],
    },
    "seed1_near_sc_basin": {
        "label": "seed1_near_sc_basin",
        "selected_child_id": "n0057",
        "selected_child_grid": [12, 16, 11],
        "best_descendant_id": "n0118",
        "best_descendant_grid": [12, 19, 11],
    },
}


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


def read_json(path: Path | str | None) -> Any:
    if path is None or str(path) == "":
        return {}
    json_path = Path(path)
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("at least one seed is required")
    return values


def mode_sort_key(mode: str) -> tuple[int, str]:
    order = {
        "measured_only": 0,
        "map_predict_lambda32": 1,
        "map_predict_lambda48": 2,
        "source_occ_free_over_cost": 3,
        "raw_hybrid_over_cost": 4,
        "source_occ_free_no_cost": 5,
    }
    return order.get(str(mode), 99), str(mode)


def min_median_max(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "min": None, "median": None, "max": None}
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def mean(values: list[Any]) -> float | None:
    finite: list[float] = []
    for value in values:
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            finite.append(v)
    return None if not finite else float(sum(finite) / len(finite))


def grid_distance_m(a: Any, b: Any, voxel_size: float) -> float | None:
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape[0] < 3 or bv.shape[0] < 3:
        return None
    return float(np.linalg.norm((av[:3] - bv[:3]) * float(voxel_size)))


def topdown_observed(observed_state: np.ndarray) -> np.ndarray:
    observed = np.asarray(observed_state)
    occupied = np.any(observed == 1, axis=2)
    free = np.any(observed == 0, axis=2)
    image = np.full(observed.shape[:2], 0.55, dtype=np.float32)
    image[free] = 0.82
    image[occupied] = 0.12
    return image


def normalize_bounds(raw: dict[str, Any], observed_shape: tuple[int, int, int], voxel_size: float) -> dict[str, tuple[float, float]]:
    if raw:
        source = raw.get("bounds", raw)
        if all(axis in source for axis in ("x", "y", "z")):
            return {axis: (float(source[axis][0]), float(source[axis][1])) for axis in ("x", "y", "z")}
    return {
        "x": (-0.5 * observed_shape[0] * voxel_size, 0.5 * observed_shape[0] * voxel_size),
        "y": (-0.5 * observed_shape[1] * voxel_size, 0.5 * observed_shape[1] * voxel_size),
        "z": (0.0, observed_shape[2] * voxel_size),
    }


def summarize_prediction_npz(path: Path, observed_state: np.ndarray, tau: float) -> dict[str, Any]:
    layer = SimPredictionLayer.from_npz(path)
    valid = np.asarray(layer.valid, dtype=bool)
    tau_mask = valid & (layer.confidence >= float(tau))
    unmeasured = np.asarray(observed_state) == -1
    occ = tau_mask & (layer.occupied_prob >= 0.5)
    free = tau_mask & (layer.free_prob >= 0.5)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(layer.shape()),
        "shape_aligned_to_observed_state": tuple(layer.shape()) == tuple(observed_state.shape),
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "prediction_valid_tau_count": int(np.count_nonzero(tau_mask)),
        "predicted_occupied_count": int(np.count_nonzero(occ)),
        "predicted_free_count": int(np.count_nonzero(free)),
        "predicted_unmeasured_count": int(np.count_nonzero(tau_mask & unmeasured)),
        "tau": float(tau),
    }


def make_mode_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    tau = float(args.tau)
    occ = float(args.occ_threshold)
    free = float(args.free_threshold)
    lambda_sc = float(args.lambda_sc)

    measured = {
        "mode": "measured_only",
        "prediction_source": "none",
        "formula_name": "measured_only",
        "formula": "gain_exp / cost",
        "config_key": "none|measured_only",
        "sc_basis": "none",
        "utility_mode": "measured",
        "lambda": None,
        "confidence_threshold": None,
        "tau": None,
        "occ_threshold": None,
        "free_threshold": None,
    }
    map32 = make_config("map_predict", "source_occ_free", "decoupled_minmax", tau, occ, free, 32.0)
    map32.update(
        {
            "mode": "map_predict_lambda32",
            "formula": "gain_exp / cost + 32 * minmax(source_occ_free)",
        }
    )
    map48 = make_config("map_predict", "source_occ_free", "decoupled_minmax", tau, occ, free, lambda_sc)
    map48.update(
        {
            "mode": "map_predict_lambda48",
            "formula": f"gain_exp / cost + {lambda_sc:g} * minmax(source_occ_free)",
        }
    )
    over = make_config("map_predict", "source_occ_free", "over_cost", tau, occ, free)
    over.update(
        {
            "mode": "source_occ_free_over_cost",
            "formula": "(gain_exp + source_occ_free) / cost",
        }
    )
    raw_hybrid = make_config("map_predict", "source_occ_free", "over_cost", tau, occ, free)
    raw_hybrid.update(
        {
            "mode": "raw_hybrid_over_cost",
            "formula": "raw_hybrid_over_cost = (gain_exp + source_occ_free) / cost",
            "utility_mode": "raw_hybrid_over_cost",
        }
    )
    no_cost = make_config("map_predict", "source_occ_free", "over_cost", tau, occ, free)
    no_cost.update(
        {
            "mode": "source_occ_free_no_cost",
            "formula": "source_occ_free",
            "utility_mode": "sc_no_cost",
        }
    )
    return {
        "measured_only": measured,
        "map_predict_lambda32": map32,
        "map_predict_lambda48": map48,
        "source_occ_free_over_cost": over,
        "raw_hybrid_over_cost": raw_hybrid,
        "source_occ_free_no_cost": no_cost,
    }


def best_by_child(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_child: dict[str, dict[str, Any]] = {}
    for row in scored:
        child_id = str(row.get("selected_child_id"))
        current = by_child.get(child_id)
        if current is None or float(row["final_value"]) > float(current["final_value"]):
            by_child[child_id] = row
    return sorted(by_child.values(), key=lambda row: float(row["final_value"]), reverse=True)


def score_candidates(candidates: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    sc_values = [float(row.get("sc_gain") or 0.0) for row in candidates]
    min_sc = min(sc_values) if sc_values else 0.0
    max_sc = max(sc_values) if sc_values else 0.0
    denom = max(max_sc - min_sc, EPS)
    scored: list[dict[str, Any]] = []
    for row in candidates:
        cost = float(row.get("cost") or 0.0)
        if cost <= EPS:
            continue
        item = dict(row)
        gain_exp = float(item.get("gain_exp") or 0.0)
        source_occ_free = float(item.get("source_occ_free_count") or item.get("sc_gain") or 0.0)
        base_exp_value = gain_exp / max(cost, EPS)
        normalized_sc = float((source_occ_free - min_sc) / denom) if max_sc > min_sc else 0.0
        utility = str(config["utility_mode"])
        lambda_value = config.get("lambda")
        if utility == "measured":
            sc_bonus = 0.0
            final_value = base_exp_value
        elif utility == "decoupled_minmax":
            sc_bonus = float(lambda_value or 0.0) * normalized_sc
            final_value = base_exp_value + sc_bonus
        elif utility in {"over_cost", "raw_hybrid_over_cost"}:
            sc_bonus = source_occ_free / max(cost, EPS)
            final_value = (gain_exp + source_occ_free) / max(cost, EPS)
        elif utility == "sc_no_cost":
            sc_bonus = source_occ_free
            final_value = source_occ_free
        else:
            raise ValueError(f"unsupported utility mode: {utility}")
        item.update(
            {
                "base_exp_value": float(base_exp_value),
                "normalized_sc": float(normalized_sc),
                "sc_bonus": float(sc_bonus),
                "final_value": float(final_value),
                "min_sc": float(min_sc),
                "max_sc": float(max_sc),
            }
        )
        scored.append(item)
    return scored


def classify_branch(
    row: dict[str, Any],
    same_seed_measured: dict[str, Any] | None,
    voxel_size: float,
) -> dict[str, Any]:
    selected = row.get("selected_child_grid")
    best = row.get("best_descendant_grid")
    measured_ref = REFERENCE_BRANCHES["measured_reference"]
    prior_ref = REFERENCE_BRANCHES["prior_low_cost_sc_reference"]
    seed1_ref = REFERENCE_BRANCHES["seed1_near_sc_basin"]
    measured_seed_selected = None if same_seed_measured is None else same_seed_measured.get("selected_child_grid")

    selected_to_fixed_measured = grid_distance_m(selected, measured_ref["selected_child_grid"], voxel_size)
    selected_to_prior = grid_distance_m(selected, prior_ref["selected_child_grid"], voxel_size)
    best_to_fixed_measured = grid_distance_m(best, measured_ref["best_descendant_grid"], voxel_size)
    best_to_prior = grid_distance_m(best, prior_ref["best_descendant_grid"], voxel_size)
    selected_to_same_seed_measured = grid_distance_m(selected, measured_seed_selected, voxel_size)
    selected_to_seed1 = grid_distance_m(selected, seed1_ref["selected_child_grid"], voxel_size)
    best_to_seed1 = grid_distance_m(best, seed1_ref["best_descendant_grid"], voxel_size)

    same_as_prior_exact = bool(
        row.get("selected_child_id") == prior_ref["selected_child_id"]
        and row.get("best_descendant_id") == prior_ref["best_descendant_id"]
    )
    spatial_prior = bool(
        selected_to_prior is not None
        and selected_to_prior <= 0.25 + 1.0e-9
        and best_to_prior is not None
        and best_to_prior <= 0.75 + 1.0e-9
    )
    same_as_seed1 = bool(
        selected_to_seed1 is not None
        and selected_to_seed1 <= 0.25 + 1.0e-9
        and best_to_seed1 is not None
        and best_to_seed1 <= 0.75 + 1.0e-9
    )
    same_as_measured = bool(
        selected_to_same_seed_measured is not None and selected_to_same_seed_measured <= 0.15 + 1.0e-9
    )
    if same_as_measured:
        primary = "same_as_measured"
    elif same_as_prior_exact:
        primary = "same_as_prior_low_cost_sc"
    elif spatial_prior:
        primary = "spatial_prior_sc_basin"
    elif same_as_seed1:
        primary = "same_as_seed1_near_sc_basin"
    elif selected_to_same_seed_measured is not None and selected_to_same_seed_measured < 0.25:
        primary = "local_jitter"
    elif selected_to_same_seed_measured is not None and selected_to_same_seed_measured >= 0.25:
        primary = "distinct_nonmeasured_branch"
    else:
        primary = "unknown"

    return {
        "branch_classification": primary,
        "same_as_measured": same_as_measured,
        "same_as_prior_low_cost_sc": same_as_prior_exact,
        "spatial_prior_sc_basin": spatial_prior,
        "same_as_seed1_near_sc_basin": same_as_seed1,
        "avoids_prior_low_cost_sc": not (same_as_prior_exact or spatial_prior),
        "selected_child_distance_from_measured_reference_m": selected_to_fixed_measured,
        "selected_child_distance_from_prior_low_cost_sc_reference_m": selected_to_prior,
        "best_descendant_distance_from_measured_reference_m": best_to_fixed_measured,
        "best_descendant_distance_from_prior_low_cost_sc_reference_m": best_to_prior,
        "selected_child_distance_from_same_seed_measured_m": selected_to_same_seed_measured,
        "selected_child_distance_from_seed1_near_sc_basin_m": selected_to_seed1,
        "best_descendant_distance_from_seed1_near_sc_basin_m": best_to_seed1,
    }


def select_decision(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    mode: str,
    config: dict[str, Any],
    same_seed_measured: dict[str, Any] | None,
    voxel_size: float,
) -> dict[str, Any]:
    scored = score_candidates(candidates, config)
    if not scored:
        raise RuntimeError(f"no scored candidates for seed {seed} mode {mode}")
    ranked = best_by_child(scored)
    winner = dict(ranked[0])
    runner = ranked[1] if len(ranked) > 1 else None
    runner_value = None if runner is None else float(runner["final_value"])
    margin = None if runner_value is None else float(float(winner["final_value"]) - runner_value)
    normalized_margin = None if margin is None else float(margin / max(abs(float(winner["final_value"])), EPS))
    base_ranked = best_by_child(sorted(scored, key=lambda row: float(row["base_exp_value"]), reverse=True))
    base_winner = base_ranked[0]
    branch = classify_branch(winner, same_seed_measured, voxel_size)

    changed_vs_measured = not bool(branch["same_as_measured"])
    low_cost_artifact = bool(
        changed_vs_measured
        and winner.get("selected_child_id") != base_winner.get("selected_child_id")
        and float(winner.get("gain_exp") or 0.0) < float(base_winner.get("gain_exp") or 0.0)
        and float(winner.get("source_occ_free_count") or winner.get("sc_gain") or 0.0)
        < float(base_winner.get("source_occ_free_count") or base_winner.get("sc_gain") or 0.0)
        and float(winner.get("cost") or 0.0) < float(base_winner.get("cost") or 0.0)
    )
    healthy_nonmeasured = bool(
        changed_vs_measured
        and bool(branch["avoids_prior_low_cost_sc"])
        and not low_cost_artifact
        and (
            float(winner.get("normalized_sc") or 0.0) >= 0.5
            or int(winner.get("sc_gain_rank") or 10**9) <= max(3, int(math.ceil(0.25 * len(scored))))
        )
        and float(winner.get("best_descendant_distance_from_root_m") or 0.0) >= 0.5
    )
    source_occ_free = float(winner.get("source_occ_free_count") or winner.get("sc_gain") or 0.0)
    return {
        "seed": int(seed),
        "mode": str(mode),
        "prediction_source": str(config["prediction_source"]),
        "formula": str(config["formula"]),
        "formula_name": str(config["formula_name"]),
        "config_key": str(config["config_key"]),
        "sc_basis": str(config.get("sc_basis", "none")),
        "utility_mode": str(config["utility_mode"]),
        "lambda": config.get("lambda"),
        "tau": config.get("confidence_threshold"),
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
        "branch_classification": branch["branch_classification"],
        "changed_vs_measured_only": changed_vs_measured,
        "same_as_measured": bool(branch["same_as_measured"]),
        "same_as_prior_low_cost_sc": bool(branch["same_as_prior_low_cost_sc"]),
        "spatial_prior_sc_basin": bool(branch["spatial_prior_sc_basin"]),
        "same_as_seed1_near_sc_basin": bool(branch["same_as_seed1_near_sc_basin"]),
        "avoids_prior_low_cost_sc": bool(branch["avoids_prior_low_cost_sc"]),
        "collapse_to_measured": bool(branch["same_as_measured"]),
        "healthy_nonmeasured_candidate": healthy_nonmeasured,
        "gain_exp": winner.get("gain_exp"),
        "source_occ_free": source_occ_free,
        "source_occ_free_count": source_occ_free,
        "source_occ_count": winner.get("source_occ_count"),
        "source_free_count": winner.get("source_free_count"),
        "sc_gain": winner.get("sc_gain"),
        "confidence_sum": winner.get("confidence_sum"),
        "confidence_mean": winner.get("confidence_mean"),
        "cost": winner.get("cost"),
        "base_exp_value": winner.get("base_exp_value"),
        "normalized_sc": winner.get("normalized_sc"),
        "sc_bonus": winner.get("sc_bonus"),
        "final_value": winner.get("final_value"),
        "runner_up_value": runner_value,
        "margin": margin,
        "normalized_margin": normalized_margin,
        "branch_depth": winner.get("branch_depth"),
        "path_node_ids": winner.get("path_node_ids"),
        "selected_cost_rank": winner.get("cost_rank"),
        "selected_gain_exp_rank": winner.get("gain_exp_rank"),
        "selected_source_occ_free_rank": winner.get("sc_gain_rank"),
        "min_sc": winner.get("min_sc"),
        "max_sc": winner.get("max_sc"),
        "root_grid": winner.get("root_grid"),
        "root_world": winner.get("root_world"),
        "selected_child_distance_from_root_m": winner.get("selected_child_distance_from_root_m"),
        "best_descendant_distance_from_root_m": winner.get("best_descendant_distance_from_root_m"),
        "tree_total_nodes": winner.get("tree_total_nodes"),
        "candidate_count": len(scored),
        "selected_child_distance_from_measured_reference_m": branch[
            "selected_child_distance_from_measured_reference_m"
        ],
        "selected_child_distance_from_prior_low_cost_sc_reference_m": branch[
            "selected_child_distance_from_prior_low_cost_sc_reference_m"
        ],
        "best_descendant_distance_from_measured_reference_m": branch[
            "best_descendant_distance_from_measured_reference_m"
        ],
        "best_descendant_distance_from_prior_low_cost_sc_reference_m": branch[
            "best_descendant_distance_from_prior_low_cost_sc_reference_m"
        ],
        "selected_child_distance_from_same_seed_measured_m": branch[
            "selected_child_distance_from_same_seed_measured_m"
        ],
        "low_cost_artifact": low_cost_artifact,
        "low_cost_artifact_vs_base_exp": low_cost_artifact,
        "base_exp_selected_child_id": base_winner.get("selected_child_id"),
        "base_exp_best_descendant_id": base_winner.get("best_descendant_id"),
        "base_exp_selected_child_grid": base_winner.get("selected_child_grid"),
        "base_exp_best_descendant_grid": base_winner.get("best_descendant_grid"),
        "base_exp_selected_gain_exp": base_winner.get("gain_exp"),
        "base_exp_selected_source_occ_free": base_winner.get("source_occ_free_count"),
        "base_exp_selected_cost": base_winner.get("cost"),
        "prediction_safety_flags": {
            "prediction_writeback": False,
            "prediction_used_for_traversability": False,
            "prediction_used_for_collision": False,
            "prediction_ray_blocking": False,
            "target_ground_truth_planning_scoring": False,
        },
    }


def rows_for_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("mode") == mode]


def fraction(rows: list[dict[str, Any]], key: str) -> float | None:
    return None if not rows else float(sum(bool(row.get(key)) for row in rows) / len(rows))


def summarize_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["mode"])].append(row)
    summary: list[dict[str, Any]] = []
    for mode in sorted(groups, key=mode_sort_key):
        items = groups[mode]
        summary.append(
            {
                "mode": mode,
                "prediction_source": items[0].get("prediction_source"),
                "formula": items[0].get("formula"),
                "lambda": items[0].get("lambda"),
                "tau": items[0].get("tau"),
                "occ_threshold": items[0].get("occ_threshold"),
                "free_threshold": items[0].get("free_threshold"),
                "seed_count": len(items),
                "branch_classification_counts": dict(Counter(str(row.get("branch_classification")) for row in items)),
                "same_as_measured_fraction": fraction(items, "same_as_measured"),
                "spatial_prior_sc_basin_fraction": fraction(items, "spatial_prior_sc_basin"),
                "avoids_prior_low_cost_sc_fraction": fraction(items, "avoids_prior_low_cost_sc"),
                "healthy_nonmeasured_fraction": fraction(items, "healthy_nonmeasured_candidate"),
                "low_cost_artifact_fraction": fraction(items, "low_cost_artifact"),
                "margin": min_median_max([float(row.get("margin") or 0.0) for row in items]),
                "mean_selected_source_occ_free": mean([row.get("source_occ_free_count") for row in items]),
                "mean_selected_normalized_sc": mean([row.get("normalized_sc") for row in items]),
                "mean_selected_source_occ_free_rank": mean([row.get("selected_source_occ_free_rank") for row in items]),
                "seed0_selected_child_id": next((row.get("selected_child_id") for row in items if int(row["seed"]) == 0), None),
                "seed0_best_descendant_id": next((row.get("best_descendant_id") for row in items if int(row["seed"]) == 0), None),
                "seed0_branch_classification": next(
                    (row.get("branch_classification") for row in items if int(row["seed"]) == 0), None
                ),
            }
        )
    return summary


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str], limit: int = 120) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("- no rows")
        write_text(path, "\n".join(lines))
        return
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(to_jsonable(row.get(field, ""))) for field in fields) + " |")
    if len(rows) > limit:
        lines.append(f"\nShowing `{limit}` of `{len(rows)}` rows.")
    write_text(path, "\n".join(lines))


def build_tree(
    *,
    observed_state: np.ndarray,
    root_grid: list[int],
    root_world: list[float],
    root_yaw: float,
    bounds: dict[str, tuple[float, float]],
    seed: int,
    prediction_layer: EmptyPredictionLayer | SimPredictionLayer,
    gain_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return build_mini_rrt_tree(
        observed_state,
        root_grid,
        root_world,
        root_yaw,
        bounds,
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
        short_edge_policy=str(args.short_edge_policy),
        crop_min_length_m=float(args.crop_min_length_m),
    )


def load_saved_tree_result(tree_dir: Path) -> dict[str, Any]:
    tree = load_tree(tree_dir)
    summary = read_json(tree_dir / "mini_rrt_tree_summary.json")
    decision = read_json(tree_dir / "subsequent_best_decision.json")
    profile = summary.get("profile", {}) if isinstance(summary, dict) else {}
    if not isinstance(profile, dict):
        profile = {}
    if not decision:
        decision = {
            "selected_child_id": None,
            "selected_child_best_descendant_id": None,
            "best_descendant_accumulated_gain": None,
            "best_descendant_accumulated_cost": None,
        }
    return {
        "tree": tree,
        "summary": summary,
        "decision": decision,
        "profile": profile,
        "tree_dir": str(tree_dir),
        "loaded_saved_tree": True,
    }


def write_formula_definition(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    definition = {
        "stage": "Stage 4A-6.5ad",
        "recommended_formula_name": "map_predict_source_occ_free_decoupled_minmax_lambda48",
        "recommended_formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
        "lambda": float(args.lambda_sc),
        "lambda32_diagnostic_formula": "gain_exp / cost + 32 * minmax(source_occ_free)",
        "tau": float(args.tau),
        "occ_threshold": float(args.occ_threshold),
        "free_threshold": float(args.free_threshold),
        "source_occ_free": {
            "counts": "predicted OCC plus predicted FREE",
            "confidence_threshold": float(args.tau),
            "occ_threshold": float(args.occ_threshold),
            "free_threshold": float(args.free_threshold),
            "validity": "prediction-valid and unmeasured voxels only",
            "prediction_unknown_counted": False,
        },
        "minmax": {
            "scope": "per seed/tree over valid root-to-descendant path accumulated source_occ_free",
            "formula": "(sc - min_sc) / (max_sc - min_sc)",
            "flat_case": "0 for all candidates when max_sc == min_sc",
        },
        "diagnostic_only": {
            "source_occ_free_over_cost": "(gain_exp + source_occ_free) / cost",
            "raw_hybrid_over_cost": "(gain_exp + source_occ_free) / cost",
            "source_occ_free_no_cost": "source_occ_free",
        },
        "prediction_information_gain_only": True,
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
    }
    save_json(output_dir / "formula_definition.json", definition)
    lines = [
        "# Formula Definition",
        "",
        f"- formula: `{definition['recommended_formula']}`",
        f"- lambda: `{definition['lambda']}`",
        f"- tau: `{definition['tau']}`",
        f"- occ/free thresholds: `{definition['occ_threshold']}` / `{definition['free_threshold']}`",
        "- SC is outside the cost denominator for lambda48 and lambda32.",
        "- over-cost formulas are diagnostic only.",
        "- prediction is information-gain-only; it is not used for traversability, collision, ray blocking, or observed-map writeback.",
    ]
    write_text(output_dir / "formula_definition.md", "\n".join(lines))
    return definition


def write_reference_branches(output_dir: Path, voxel_size: float) -> None:
    refs = dict(REFERENCE_BRANCHES)
    refs["spatial_rules"] = {
        "voxel_size": float(voxel_size),
        "spatial_prior_sc_basin": "selected child <=0.25m from [11,15,11] and best descendant <=0.75m from [14,15,11]",
        "same_as_measured": "selected child <=0.15m from same-seed measured selected child",
        "distinct_nonmeasured_branch": "selected child differs from measured by >=0.25m and is not the prior SC basin",
    }
    save_json(output_dir / "real_frame_reference_branches.json", refs)
    lines = [
        "# Real Frame Reference Branches",
        "",
        "- measured reference: `n0001 -> n0112`, grids `[17,16,11] -> [8,27,11]`",
        "- prior low-cost SC reference: `n0127 -> n0162`, grids `[11,15,11] -> [14,15,11]`",
        "- seed1 near SC basin: `n0057 -> n0118`, grids `[12,16,11] -> [12,19,11]`",
        "- this real-frame smoke does not use hidden-room labels.",
    ]
    write_text(output_dir / "real_frame_reference_branches.md", "\n".join(lines))


def summarize_lambda48(decisions: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    mode_summary = summarize_modes(decisions)
    by_mode = {row["mode"]: row for row in mode_summary}
    map48_rows = rows_for_mode(decisions, "map_predict_lambda48")
    measured_rows = rows_for_mode(decisions, "measured_only")
    seed0_measured = next((row for row in measured_rows if int(row["seed"]) == 0), {})
    seed0_map48 = next((row for row in map48_rows if int(row["seed"]) == 0), {})
    counts = Counter(str(row.get("branch_classification")) for row in map48_rows)
    if not map48_rows:
        behavior = "missing"
    elif counts.get("spatial_prior_sc_basin", 0) or counts.get("same_as_prior_low_cost_sc", 0):
        behavior = "prior_low_cost_sc"
    elif counts.get("distinct_nonmeasured_branch", 0) or counts.get("same_as_seed1_near_sc_basin", 0):
        behavior = "distinct_or_near_nonmeasured"
    elif counts.get("same_as_measured", 0) == len(map48_rows):
        behavior = "collapse_to_measured"
    else:
        behavior = "mixed_seed_sensitive"
    return {
        "stage": "Stage 4A-6.5ad",
        "status": "completed",
        "seed_count": len(seeds),
        "mode_count": len({row["mode"] for row in decisions}),
        "decision_row_count": len(decisions),
        "map_predict_lambda48_behavior": behavior,
        "measured_only": by_mode.get("measured_only"),
        "map_predict_lambda32": by_mode.get("map_predict_lambda32"),
        "map_predict_lambda48": by_mode.get("map_predict_lambda48"),
        "source_occ_free_over_cost": by_mode.get("source_occ_free_over_cost"),
        "raw_hybrid_over_cost": by_mode.get("raw_hybrid_over_cost"),
        "source_occ_free_no_cost": by_mode.get("source_occ_free_no_cost"),
        "map_predict_lambda48_branch_classification_counts": dict(counts),
        "map_predict_lambda48_same_as_measured_fraction": fraction(map48_rows, "same_as_measured"),
        "map_predict_lambda48_spatial_prior_sc_basin_fraction": fraction(map48_rows, "spatial_prior_sc_basin"),
        "map_predict_lambda48_healthy_nonmeasured_fraction": fraction(map48_rows, "healthy_nonmeasured_candidate"),
        "map_predict_lambda48_low_cost_artifact_fraction": fraction(map48_rows, "low_cost_artifact"),
        "map_predict_lambda48_avoids_prior_low_cost_sc_fraction": fraction(map48_rows, "avoids_prior_low_cost_sc"),
        "seed0_measured_reference_reproduced": bool(
            seed0_measured.get("selected_child_id") == "n0001"
            and seed0_measured.get("best_descendant_id") == "n0112"
        ),
        "seed0_map_predict_lambda48": {
            "selected_child_id": seed0_map48.get("selected_child_id"),
            "best_descendant_id": seed0_map48.get("best_descendant_id"),
            "selected_child_grid": seed0_map48.get("selected_child_grid"),
            "best_descendant_grid": seed0_map48.get("best_descendant_grid"),
            "branch_classification": seed0_map48.get("branch_classification"),
            "low_cost_artifact": seed0_map48.get("low_cost_artifact"),
            "healthy_nonmeasured_candidate": seed0_map48.get("healthy_nonmeasured_candidate"),
        },
        "formula_components_logged": all(
            key in row for row in map48_rows for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value")
        ),
        "saved_frame_only_readiness": True,
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
    }


def write_lambda48_summary_md(path: Path, summary: dict[str, Any]) -> None:
    mp = summary.get("map_predict_lambda48") or {}
    lines = [
        "# Lambda48 Behavior Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- seeds/modes/decision rows: `{summary['seed_count']}` / `{summary['mode_count']}` / `{summary['decision_row_count']}`",
        f"- seed0 measured reference reproduced: `{summary['seed0_measured_reference_reproduced']}`",
        f"- lambda48 behavior: `{summary['map_predict_lambda48_behavior']}`",
        f"- lambda48 branch counts: `{summary['map_predict_lambda48_branch_classification_counts']}`",
        f"- lambda48 same-as-measured fraction: `{summary['map_predict_lambda48_same_as_measured_fraction']}`",
        f"- lambda48 prior SC basin fraction: `{summary['map_predict_lambda48_spatial_prior_sc_basin_fraction']}`",
        f"- lambda48 healthy non-measured fraction: `{summary['map_predict_lambda48_healthy_nonmeasured_fraction']}`",
        f"- lambda48 low-cost artifact fraction: `{summary['map_predict_lambda48_low_cost_artifact_fraction']}`",
        f"- mean selected source_occ_free: `{mp.get('mean_selected_source_occ_free')}`",
        "- readiness: saved-frame-only; not runtime-smoke-ready; not rollout-ready.",
    ]
    write_text(path, "\n".join(lines))


def compare_to_stage4a65z_z1(
    stage4a65z_dir: Path,
    stage4a65z1_dir: Path,
    lambda_summary: dict[str, Any],
) -> dict[str, Any]:
    z_summary = read_json(stage4a65z_dir / "stage4a65z_decoupled_sc_utility_sweep_summary.json")
    z_lambda_summary = read_json(stage4a65z_dir / "lambda_sweep_summary_by_basis_variant.json")
    z1_summary = read_json(stage4a65z1_dir / "stage4a65z1_decoupled_signal_strength_summary.json")
    z1_required = read_json(stage4a65z1_dir / "required_lambda_to_flip.json")
    branch_by_formula = read_json(stage4a65z_dir / "branch_classification_summary_by_formula.json")
    source32 = None
    source_fixed = [
        row
        for row in z_lambda_summary
        if row.get("sc_basis") == "source_occ_free"
        and row.get("lambda_family") == "fixed"
        and str(row.get("lambda_label")) == "32"
    ]
    if source_fixed:
        source32 = source_fixed[0]
    corrected32 = branch_by_formula.get("decoupled_source_occ_free_fixed_32") if isinstance(branch_by_formula, dict) else None
    map48 = lambda_summary.get("map_predict_lambda48") or {}
    same48 = lambda_summary.get("map_predict_lambda48_same_as_measured_fraction")
    prior48 = lambda_summary.get("map_predict_lambda48_spatial_prior_sc_basin_fraction")
    healthy48 = lambda_summary.get("map_predict_lambda48_healthy_nonmeasured_fraction")
    same32 = None
    prior32 = None
    if corrected32:
        same32 = (corrected32.get("fractions") or {}).get("same_as_measured_for_seed")
        prior32 = (corrected32.get("fractions") or {}).get("spatial_seed0_sc_basin")
    elif source32:
        same32 = source32.get("same_as_measured_fraction")
        prior32 = source32.get("spatial_seed0_sc_basin_fraction")
    return {
        "stage4a65z_dir": str(stage4a65z_dir),
        "stage4a65z1_dir": str(stage4a65z1_dir),
        "stage4a65z_decision_rows": (z_summary.get("answers") or {}).get("decision_rows")
        or z_summary.get("decision_row_count"),
        "stage4a65z_original_source_occ_free_fixed32": source32,
        "stage4a65z_corrected_source_occ_free_fixed32": corrected32,
        "stage4a65z1_answers": z1_summary.get("answers"),
        "stage4a65z1_required_lambda_distribution": (
            z1_summary.get("required_lambda_result")
            or {
                "finite_p50_p90_max": "229.31585862120286 / 627.9926880897762 / 34462.89245592027",
                "source": "summary text fallback",
            }
        ),
        "stage4a65z1_required_lambda_row_count": len(z1_required) if isinstance(z1_required, list) else None,
        "lambda48_same_as_measured_fraction": same48,
        "lambda32_same_as_measured_fraction_from_6p5z": same32,
        "lambda48_prior_sc_basin_fraction": prior48,
        "lambda32_prior_sc_basin_fraction_from_6p5z": prior32,
        "did_lambda48_exceed_previous_lambda32_behavior": (
            None if same32 is None or same48 is None else float(same48) < float(same32)
        ),
        "did_lambda48_still_collapse_to_measured": bool(same48 == 1.0),
        "did_lambda48_pick_only_old_bad_branch": bool(prior48 == 1.0),
        "did_lambda48_find_distinct_candidate": bool((healthy48 or 0.0) > 0.0),
        "interpretation": (
            "Compared against 6.5z/z.1 saved-frame diagnostics only. 6.5z used saved raw trees, "
            "while 6.5ad regenerates source-protected trees on the same saved real Frame2."
        ),
    }


def write_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    lines = [
        "# Comparison To Stage 4A-6.5z / z.1",
        "",
        f"- lambda48 same-as-measured fraction: `{comparison['lambda48_same_as_measured_fraction']}`",
        f"- 6.5z lambda32 same-as-measured fraction: `{comparison['lambda32_same_as_measured_fraction_from_6p5z']}`",
        f"- lambda48 prior SC basin fraction: `{comparison['lambda48_prior_sc_basin_fraction']}`",
        f"- 6.5z lambda32 prior SC basin fraction: `{comparison['lambda32_prior_sc_basin_fraction_from_6p5z']}`",
        f"- did lambda48 exceed previous lambda32 behavior: `{comparison['did_lambda48_exceed_previous_lambda32_behavior']}`",
        f"- did lambda48 still collapse to measured: `{comparison['did_lambda48_still_collapse_to_measured']}`",
        f"- did lambda48 pick only old bad branch: `{comparison['did_lambda48_pick_only_old_bad_branch']}`",
        f"- did lambda48 find a distinct candidate: `{comparison['did_lambda48_find_distinct_candidate']}`",
        "",
        "This comparison is offline-only and does not claim coverage improvement.",
    ]
    write_text(path, "\n".join(lines))


def recommendation_from_summary(summary: dict[str, Any]) -> tuple[str, str]:
    same = float(summary.get("map_predict_lambda48_same_as_measured_fraction") or 0.0)
    prior = float(summary.get("map_predict_lambda48_spatial_prior_sc_basin_fraction") or 0.0)
    low_cost = float(summary.get("map_predict_lambda48_low_cost_artifact_fraction") or 0.0)
    healthy = float(summary.get("map_predict_lambda48_healthy_nonmeasured_fraction") or 0.0)
    counts = summary.get("map_predict_lambda48_branch_classification_counts") or {}
    if prior > 0.0 or low_cost > 0.0:
        return (
            "low-cost artifact / dominance-gate diagnosis, still offline",
            "lambda48 touched the prior SC basin or triggered low-cost artifact risk on the saved real frame.",
        )
    if healthy > 0.0 and same < 1.0 and len(counts) <= 2:
        return (
            "saved-frame formula smoke on another real medium frame only",
            "lambda48 found at least one healthy non-measured branch while avoiding the prior low-cost SC basin.",
        )
    if same >= 0.8:
        return (
            "saved-frame lambda48 replay on additional real medium frames, still offline",
            "lambda48 mostly collapses to measured on this real frame, so another saved real frame should be checked before runtime.",
        )
    return (
        "multi-seed/multi-frame saved-frame replay only",
        "lambda48 behavior is seed-sensitive on this real frame and needs offline replay breadth before runtime.",
    )


def write_recommendation(path: Path, summary: dict[str, Any]) -> tuple[str, str]:
    next_step, why = recommendation_from_summary(summary)
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
    return next_step, why


def write_safety_report(output_dir: Path, safety: dict[str, Any]) -> None:
    save_json(output_dir / "prediction_safety_report.json", safety)
    lines = [
        "# Prediction Safety Report",
        "",
        "- Isaac startup: `false`",
        "- new capture: `false`",
        "- map_predict rerun: `false`",
        "- SSCNet inference: `false`",
        "- selected action execution: `false`",
        "- two-frame runtime: `false`",
        "- rollout: `false`",
        "- prediction writeback: `false`",
        "- prediction used for traversability/collision: `false`",
        "- prediction ray blocking: `false`",
        "- target/ground-truth planning/scoring: `false`",
        f"- observed_state unchanged: `{not safety['existing_observed_state_modified']}`",
        f"- prediction NPZ unchanged: `{not safety['prediction_npz_modified']}`",
        f"- checkpoint unchanged: `{not safety['checkpoint_modified']}`",
    ]
    write_text(output_dir / "prediction_safety_report.md", "\n".join(lines))


def write_loaded_manifest_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Loaded Inputs Manifest",
        "",
        f"- Stage 4A-6.5p dir: `{manifest['stage4a65p_dir']}`",
        f"- Stage 4A-6.5ac dir: `{manifest['stage4a65ac_dir']}`",
        f"- Stage 4A-6.5y dir: `{manifest['stage4a65y_dir']}`",
        f"- observed_state: `{manifest['observed_state']['path']}`",
        f"- prediction NPZ: `{manifest['prediction_npz']['path']}`",
        f"- pose/camera: `{manifest['pose_json']['path']}` / `{manifest['camera_info_json']['path']}`",
        f"- observed shape: `{manifest['observed_state']['shape']}`",
        f"- bounds: `{manifest['bounds']}`",
        f"- tree root/source: `{manifest['tree_root_source']}`",
        "- no Isaac startup, no new capture, no map_predict rerun.",
    ]
    write_text(path, "\n".join(lines))


def plot_selected_branches(
    path: Path,
    observed_state: np.ndarray,
    rows: list[dict[str, Any]],
    title: str,
    modes: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.5), constrained_layout=True)
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap="Greys", alpha=0.55, interpolation="nearest")
    colors = {
        "measured_only": "#f97316",
        "map_predict_lambda32": "#7c3aed",
        "map_predict_lambda48": "#059669",
        "source_occ_free_over_cost": "#dc2626",
        "raw_hybrid_over_cost": "#0f766e",
        "source_occ_free_no_cost": "#2563eb",
    }
    for ref_name, ref in REFERENCE_BRANCHES.items():
        grid = ref["selected_child_grid"]
        marker = "x" if ref_name != "measured_reference" else "+"
        ax.scatter([grid[0]], [grid[1]], marker=marker, s=110, c="#111827", linewidths=1.7)
        ax.text(grid[0] + 0.4, grid[1] + 0.4, ref_name.replace("_reference", ""), fontsize=7, color="#111827")
    for row in rows:
        if row.get("mode") not in modes:
            continue
        root = row.get("root_grid")
        best = row.get("best_descendant_grid")
        selected = row.get("selected_child_grid")
        color = colors.get(str(row.get("mode")), "#111827")
        label = f"{row.get('mode')} s{row.get('seed')}"
        if root and best:
            ax.plot([root[0], best[0]], [root[1], best[1]], color=color, linewidth=1.1, alpha=0.48)
        if selected:
            ax.scatter([selected[0]], [selected[1]], color=color, s=25, alpha=0.9)
        if best:
            ax.scatter([best[0]], [best[1]], color=color, marker="*", s=64, alpha=0.9, label=label)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:14], labels[:14], fontsize=6, loc="upper right")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_classification_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        groups[str(row["mode"])][str(row["branch_classification"])] += 1
    labels = sorted(groups, key=mode_sort_key)
    classes = sorted({cls for counter in groups.values() for cls in counter})
    bottoms = np.zeros(len(labels), dtype=np.float64)
    palette = ["#f97316", "#059669", "#2563eb", "#dc2626", "#7c3aed", "#0f766e", "#64748b"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for idx, cls in enumerate(classes):
        values = np.asarray([groups[label].get(cls, 0) for label in labels], dtype=np.float64)
        ax.bar(range(len(labels)), values, bottom=bottoms, label=cls, color=palette[idx % len(palette)])
        bottoms += values
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("seed count")
    ax.set_title("Branch classification by mode")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_value_components(path: Path, rows: list[dict[str, Any]]) -> None:
    data = [row for row in rows if row.get("mode") in {"map_predict_lambda32", "map_predict_lambda48"}]
    data = sorted(data, key=lambda row: (str(row["mode"]), int(row["seed"])))
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    x = np.arange(len(data))
    base = [float(row.get("base_exp_value") or 0.0) for row in data]
    bonus = [float(row.get("sc_bonus") or 0.0) for row in data]
    labels = [f"{row['mode'].replace('map_predict_', '')}\ns{row['seed']}" for row in data]
    ax.bar(x, base, color="#f97316", label="base_exp_value")
    ax.bar(x, bonus, bottom=base, color="#2563eb", label="sc_bonus")
    ax.plot(x, [float(row.get("final_value") or 0.0) for row in data], color="#111827", marker="o", label="final_value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("value")
    ax.set_title("Lambda value components on real Frame2")
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_metric_by_mode(
    path: Path,
    rows: list[dict[str, Any]],
    field: str,
    ylabel: str,
    title: str,
    color: str,
) -> None:
    summary = summarize_modes(rows)
    labels = [row["mode"] for row in summary]
    values = [float(row.get(field) or 0.0) for row in summary]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.bar(range(len(labels)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_margin(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("margin") is not None:
            groups[str(row["mode"])].append(float(row["margin"]))
    labels = sorted(groups, key=mode_sort_key)
    data = [groups[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.tick_params(axis="x", labelrotation=32)
    ax.set_ylabel("margin")
    ax.set_title("Decision margin by mode")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_prior_distance(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("selected_child_distance_from_prior_low_cost_sc_reference_m")
        if value is not None:
            groups[str(row["mode"])].append(float(value))
    labels = sorted(groups, key=mode_sort_key)
    data = [groups[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0.25, color="#dc2626", linestyle="--", linewidth=1.1, label="prior basin threshold")
    ax.tick_params(axis="x", labelrotation=32)
    ax.set_ylabel("selected child distance to prior SC basin (m)")
    ax.set_title("Prior SC basin distance by mode")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_plots(
    output_dir: Path,
    observed_state: np.ndarray,
    decisions: list[dict[str, Any]],
    missing: dict[str, Any],
) -> None:
    plotters = {
        "selected_branches_topdown_real_frame.png": lambda p: plot_selected_branches(
            p,
            observed_state,
            decisions,
            "Selected branches on real Frame2",
            [
                "measured_only",
                "map_predict_lambda32",
                "map_predict_lambda48",
                "source_occ_free_over_cost",
                "raw_hybrid_over_cost",
                "source_occ_free_no_cost",
            ],
        ),
        "measured_vs_lambda48_topdown.png": lambda p: plot_selected_branches(
            p,
            observed_state,
            decisions,
            "Measured-only vs map_predict lambda48",
            ["measured_only", "map_predict_lambda48"],
        ),
        "branch_classification_bar.png": lambda p: plot_classification_bar(p, decisions),
        "value_components_lambda48_real_frame.png": lambda p: plot_value_components(p, decisions),
        "source_occ_free_rank_by_mode.png": lambda p: plot_metric_by_mode(
            p,
            decisions,
            "mean_selected_source_occ_free_rank",
            "mean selected source_occ_free rank",
            "Source OCC+FREE rank by mode",
            "#059669",
        ),
        "low_cost_artifact_by_mode.png": lambda p: plot_metric_by_mode(
            p,
            decisions,
            "low_cost_artifact_fraction",
            "low-cost artifact fraction",
            "Low-cost artifact by mode",
            "#dc2626",
        ),
        "margin_by_mode.png": lambda p: plot_margin(p, decisions),
        "prior_sc_basin_distance_by_mode.png": lambda p: plot_prior_distance(p, decisions),
    }
    summary_rows = summarize_modes(decisions)
    for name, plotter in plotters.items():
        try:
            if name in {"source_occ_free_rank_by_mode.png", "low_cost_artifact_by_mode.png"}:
                # These plotters use mode summaries but share the common signature.
                if name == "source_occ_free_rank_by_mode.png":
                    labels = [row["mode"] for row in summary_rows]
                    values = [float(row.get("mean_selected_source_occ_free_rank") or 0.0) for row in summary_rows]
                    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
                    ax.bar(range(len(labels)), values, color="#059669")
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=32, ha="right")
                    ax.set_ylabel("mean selected source_occ_free rank")
                    ax.set_title("Source OCC+FREE rank by mode")
                    fig.savefig(output_dir / name, dpi=170)
                    plt.close(fig)
                else:
                    labels = [row["mode"] for row in summary_rows]
                    values = [float(row.get("low_cost_artifact_fraction") or 0.0) for row in summary_rows]
                    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
                    ax.bar(range(len(labels)), values, color="#dc2626")
                    ax.set_ylim(0.0, max(1.0, max(values, default=0.0) + 0.1))
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=32, ha="right")
                    ax.set_ylabel("low-cost artifact fraction")
                    ax.set_title("Low-cost artifact by mode")
                    fig.savefig(output_dir / name, dpi=170)
                    plt.close(fig)
            else:
                plotter(output_dir / name)
        except Exception as exc:  # pragma: no cover - operational diagnostics
            reason_path = output_dir / f"{Path(name).stem}_skipped_reason.md"
            write_text(reason_path, f"# Plot Skipped\n\n- plot: `{name}`\n- reason: `{type(exc).__name__}: {exc}`")
            missing.setdefault("plot_failures", []).append({"plot": name, "reason": f"{type(exc).__name__}: {exc}"})


def write_final_summary(
    output_dir: Path,
    *,
    answers: dict[str, Any],
    lambda_summary: dict[str, Any],
    comparison: dict[str, Any],
    next_step: str,
    why: str,
) -> dict[str, Any]:
    summary = {
        "stage": "Stage 4A-6.5ad",
        "status": "completed",
        "answers": answers,
        "lambda48_behavior_summary": lambda_summary,
        "comparison_to_stage4a65z_z1": comparison,
        "recommended_next_faithful_step": next_step,
        "recommendation_reason": why,
        "readiness": {
            "saved_frame_only": True,
            "runtime_smoke": False,
            "rollout": False,
        },
        "coverage_improvement_claimed": False,
    }
    save_json(output_dir / "stage4a65ad_real_frame_lambda48_formula_summary.json", summary)
    lines = [
        "# Stage 4A-6.5ad Real Frame Lambda48 Formula Summary",
        "",
        f"1. Loaded Stage 4A-6.5p real Frame2 inputs: `{answers['loaded_stage4a65p_frame2_inputs']}`.",
        f"2. No Isaac / new capture / map_predict rerun: `{answers['no_isaac_no_capture_no_map_predict_rerun']}`.",
        f"3. Seeds / modes / decision rows: `{answers['seed_count']}` / `{answers['mode_count']}` / `{answers['decision_row_count']}`.",
        f"4. measured_only reproduced real Frame2 measured reference: `{answers['measured_only_reproduced_frame2_reference']}`.",
        f"5. map_predict lambda48 selected branch: `{answers['map_predict_lambda48_seed0_branch']}` for seed0; counts `{answers['map_predict_lambda48_branch_counts']}`.",
        f"6. lambda48 classification: `{answers['map_predict_lambda48_behavior']}`.",
        f"7. lambda48 selected `n0127 -> n0162` or spatial prior SC basin: `{answers['lambda48_prior_low_cost_sc_or_spatial_basin']}`.",
        f"8. lambda48 low-cost artifact fraction: `{answers['lambda48_low_cost_artifact_fraction']}`.",
        f"9. lambda48 healthy non-measured fraction: `{answers['lambda48_healthy_nonmeasured_fraction']}`.",
        f"10. lambda32 vs lambda48: `{answers['lambda32_vs_lambda48']}`.",
        f"11. over-cost diagnostic: `{answers['over_cost_diagnostic']}`.",
        f"12. Consistent with Stage 4A-6.5z / z.1: `{answers['comparison_to_stage4a65z_z1']}`.",
        f"13. Supported next step: `{next_step}`.",
        "14. Runtime two-frame / rollout readiness: `false / false`.",
        "",
        f"Why: {why}",
        "",
        "This is a saved-frame-only classification smoke and does not claim coverage improvement.",
    ]
    write_text(output_dir / "stage4a65ad_real_frame_lambda48_formula_summary.md", "\n".join(lines))
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage4a65p_dir = Path(args.stage4a65p_dir).resolve()
    stage4a65ac_dir = Path(args.stage4a65ac_dir).resolve()
    stage4a65y_dir = Path(args.stage4a65y_dir).resolve()
    stage4a65z_dir = Path(args.stage4a65z_dir).resolve()
    stage4a65z1_dir = Path(args.stage4a65z1_dir).resolve()
    observed_path = Path(args.observed_state).resolve()
    prediction_path = Path(args.prediction_npz).resolve()
    pose_path = Path(args.pose_json).resolve()
    camera_info_path = Path(args.camera_info_json).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    seeds = parse_ints(args.seeds)

    required_inputs = [observed_path, prediction_path, pose_path, camera_info_path]
    missing_inputs = [str(path) for path in required_inputs if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"missing required inputs: {missing_inputs}")

    observed_hash_before = sha256_file(observed_path)
    prediction_hash_before = sha256_file(prediction_path)
    checkpoint_hash_before = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    observed_state = np.load(observed_path)
    observed_shape = tuple(int(v) for v in observed_state.shape)
    observed_summary = read_json(stage4a65p_dir / "frame002_observed_summary.json")
    bounds = normalize_bounds(observed_summary, observed_shape, float(args.voxel_size))
    pose = read_json(pose_path)
    root_world = [float(v) for v in pose.get("position", [])]
    if len(root_world) != 3:
        raise ValueError(f"pose JSON missing 3D position: {pose_path}")
    root_grid = list(world_to_grid(root_world, bounds, float(args.voxel_size), shape=observed_shape, clip=True))
    root_yaw = float(pose.get("yaw_rad", 0.0))
    prediction_layer = SimPredictionLayer.from_npz(prediction_path)
    empty_layer = EmptyPredictionLayer(observed_shape)
    if tuple(prediction_layer.shape()) != observed_shape:
        raise ValueError(f"prediction shape {prediction_layer.shape()} != observed_state {observed_shape}")
    saved_tree_status = []
    for seed in seeds:
        measured_tree_file = (
            stage4a65y_dir
            / "raw_trees"
            / f"seed_{seed:03d}"
            / "measured_only"
            / "mini_rrt_tree_segments.jsonl"
        )
        sc_tree_file = (
            stage4a65y_dir
            / "raw_trees"
            / f"seed_{seed:03d}"
            / "current_raw_count"
            / "mini_rrt_tree_segments.jsonl"
        )
        saved_tree_status.append(
            {
                "seed": int(seed),
                "measured_tree": str(measured_tree_file),
                "measured_tree_exists": measured_tree_file.is_file(),
                "current_raw_count_tree": str(sc_tree_file),
                "current_raw_count_tree_exists": sc_tree_file.is_file(),
            }
        )
    use_saved_trees = bool(
        args.prefer_saved_stage4a65y_trees
        and all(row["measured_tree_exists"] and row["current_raw_count_tree_exists"] for row in saved_tree_status)
    )

    loaded_manifest = {
        "stage": "Stage 4A-6.5ad",
        "stage4a65p_dir": str(stage4a65p_dir),
        "stage4a65ac_dir": str(stage4a65ac_dir),
        "stage4a65y_dir": str(stage4a65y_dir),
        "stage4a65z_dir": str(stage4a65z_dir),
        "stage4a65z1_dir": str(stage4a65z1_dir),
        "observed_state": {
            "path": str(observed_path),
            "sha256": observed_hash_before,
            "shape": list(observed_shape),
        },
        "prediction_npz": summarize_prediction_npz(prediction_path, observed_state, float(args.tau)),
        "pose_json": {"path": str(pose_path), "sha256": sha256_file(pose_path), "position": root_world, "yaw_rad": root_yaw},
        "camera_info_json": {"path": str(camera_info_path), "sha256": sha256_file(camera_info_path)},
        "bounds": bounds,
        "pose_root_grid": root_grid,
        "pose_root_world": root_world,
        "tree_root_source": "stage4a65y_saved_raw_trees" if use_saved_trees else "rebuilt_from_pose_json",
        "saved_stage4a65y_tree_status": saved_tree_status,
        "reference_stage4a65ac_summary": {
            "path": str(stage4a65ac_dir / "stage4a65ac_saved_frame_lambda48_formula_summary.json"),
            "exists": (stage4a65ac_dir / "stage4a65ac_saved_frame_lambda48_formula_summary.json").is_file(),
        },
        "no_isaac_startup": True,
        "no_new_capture": True,
        "no_map_predict_rerun": True,
    }
    save_json(output_dir / "loaded_inputs_manifest.json", loaded_manifest)
    write_loaded_manifest_md(output_dir / "loaded_inputs_manifest.md", loaded_manifest)
    formula_definition = write_formula_definition(output_dir, args)
    write_reference_branches(output_dir, float(args.voxel_size))

    configs = make_mode_configs(args)
    hidden_mask = np.zeros(observed_shape, dtype=bool)
    frontier_local_mask = np.zeros(observed_shape, dtype=bool)
    decisions: list[dict[str, Any]] = []
    tree_generation_rows: list[dict[str, Any]] = []
    measured_by_seed: dict[int, dict[str, Any]] = {}
    tree_root = output_dir / "raw_tree_summaries"
    tree_root.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        if use_saved_trees:
            measured_result = load_saved_tree_result(
                stage4a65y_dir / "raw_trees" / f"seed_{seed:03d}" / "measured_only"
            )
        else:
            measured_result = build_tree(
                observed_state=observed_state,
                root_grid=root_grid,
                root_world=root_world,
                root_yaw=root_yaw,
                bounds=bounds,
                seed=seed,
                prediction_layer=empty_layer,
                gain_mode="exp",
                args=args,
            )
        measured_tree = measured_result["tree"]
        measured_candidates = path_candidate_rows(measured_tree, None, None)
        measured_decision = select_decision(
            measured_candidates,
            seed=seed,
            mode="measured_only",
            config=configs["measured_only"],
            same_seed_measured=None,
            voxel_size=float(args.voxel_size),
        )
        measured_decision["changed_vs_measured_only"] = False
        measured_decision["branch_classification"] = "same_as_measured"
        measured_decision["same_as_measured"] = True
        measured_decision["spatial_prior_sc_basin"] = False
        measured_decision["same_as_prior_low_cost_sc"] = False
        measured_decision["avoids_prior_low_cost_sc"] = True
        measured_decision["healthy_nonmeasured_candidate"] = False
        measured_decision["low_cost_artifact"] = False
        measured_decision["collapse_to_measured"] = True
        measured_by_seed[seed] = measured_decision
        decisions.append(measured_decision)

        if use_saved_trees:
            sc_result = load_saved_tree_result(
                stage4a65y_dir / "raw_trees" / f"seed_{seed:03d}" / "current_raw_count"
            )
        else:
            sc_result = build_tree(
                observed_state=observed_state,
                root_grid=root_grid,
                root_world=root_world,
                root_yaw=root_yaw,
                bounds=bounds,
                seed=seed,
                prediction_layer=prediction_layer,
                gain_mode="hybrid",
                args=args,
            )
        sc_tree = sc_result["tree"]
        segment_arrays = precompute_segment_prediction_arrays(
            sc_tree,
            observed_state,
            prediction_layer,
            hidden_mask,
            frontier_local_mask,
            args,
        )

        for mode in (
            "map_predict_lambda32",
            "map_predict_lambda48",
            "source_occ_free_over_cost",
            "raw_hybrid_over_cost",
            "source_occ_free_no_cost",
        ):
            config = configs[mode]
            candidates = path_candidate_rows(sc_tree, segment_arrays, config)
            decision = select_decision(
                candidates,
                seed=seed,
                mode=mode,
                config=config,
                same_seed_measured=measured_by_seed[seed],
                voxel_size=float(args.voxel_size),
            )
            decisions.append(decision)

        for label, result in (("measured_only", measured_result), ("map_predict_raw_count", sc_result)):
            decision = result["decision"]
            tree = result["tree"]
            tree_generation_rows.append(
                {
                    "seed": int(seed),
                    "tree_label": label,
                    "node_count": int(len(tree)),
                    "accepted_nodes_excluding_root": int(result["profile"].get("accepted_nodes_excluding_root", 0)),
                    "rejected_samples": int(result["profile"].get("rejected_samples", 0)),
                    "attempts": int(result["profile"].get("attempts", 0)),
                    "selected_child_id": decision.get("selected_child_id"),
                    "selected_child_best_descendant_id": decision.get("selected_child_best_descendant_id"),
                    "best_descendant_accumulated_gain": decision.get("best_descendant_accumulated_gain"),
                    "best_descendant_accumulated_cost": decision.get("best_descendant_accumulated_cost"),
                }
            )
            if bool(args.save_raw_tree_summaries):
                seed_dir = tree_root / f"seed_{seed:03d}" / label
                write_jsonl(seed_dir / "mini_rrt_tree_segments.jsonl", [segment_record(seg) for seg in tree.values()])

    decisions = sorted(decisions, key=lambda row: (int(row["seed"]), mode_sort_key(str(row["mode"]))))
    mode_summary = summarize_modes(decisions)
    lambda_summary = summarize_lambda48(decisions, seeds)
    comparison = compare_to_stage4a65z_z1(stage4a65z_dir, stage4a65z1_dir, lambda_summary)

    decision_fields = [
        "seed",
        "mode",
        "prediction_source",
        "formula",
        "lambda",
        "tau",
        "occ_threshold",
        "free_threshold",
        "selected_child_id",
        "selected_child_grid",
        "selected_child_world",
        "best_descendant_id",
        "best_descendant_grid",
        "best_descendant_world",
        "branch_classification",
        "changed_vs_measured_only",
        "same_as_measured",
        "same_as_prior_low_cost_sc",
        "spatial_prior_sc_basin",
        "same_as_seed1_near_sc_basin",
        "avoids_prior_low_cost_sc",
        "healthy_nonmeasured_candidate",
        "gain_exp",
        "source_occ_free_count",
        "source_occ_count",
        "source_free_count",
        "cost",
        "base_exp_value",
        "normalized_sc",
        "sc_bonus",
        "final_value",
        "runner_up_value",
        "margin",
        "normalized_margin",
        "branch_depth",
        "path_node_ids",
        "selected_cost_rank",
        "selected_gain_exp_rank",
        "selected_source_occ_free_rank",
        "min_sc",
        "max_sc",
        "selected_child_distance_from_measured_reference_m",
        "selected_child_distance_from_prior_low_cost_sc_reference_m",
        "best_descendant_distance_from_measured_reference_m",
        "best_descendant_distance_from_prior_low_cost_sc_reference_m",
        "low_cost_artifact",
        "prediction_safety_flags",
    ]
    save_json(output_dir / "per_seed_mode_decisions.json", decisions)
    write_csv(output_dir / "per_seed_mode_decisions.csv", decisions, decision_fields)
    write_md_table(
        output_dir / "per_seed_mode_decisions.md",
        "Per-Seed Mode Decisions",
        decisions,
        [
            "seed",
            "mode",
            "selected_child_id",
            "best_descendant_id",
            "branch_classification",
            "gain_exp",
            "source_occ_free_count",
            "cost",
            "base_exp_value",
            "normalized_sc",
            "sc_bonus",
            "final_value",
            "margin",
            "low_cost_artifact",
        ],
    )
    value_rows = [
        {
            key: row.get(key)
            for key in (
                "seed",
                "mode",
                "formula",
                "lambda",
                "tau",
                "occ_threshold",
                "free_threshold",
                "gain_exp",
                "source_occ_free_count",
                "source_occ_count",
                "source_free_count",
                "cost",
                "base_exp_value",
                "normalized_sc",
                "sc_bonus",
                "final_value",
                "runner_up_value",
                "margin",
                "normalized_margin",
                "min_sc",
                "max_sc",
                "path_node_ids",
            )
        }
        for row in decisions
    ]
    save_json(output_dir / "per_seed_value_components.json", value_rows)
    write_csv(output_dir / "per_seed_value_components.csv", value_rows)

    branch_rows = [
        {
            key: row.get(key)
            for key in (
                "seed",
                "mode",
                "selected_child_id",
                "best_descendant_id",
                "selected_child_grid",
                "best_descendant_grid",
                "branch_classification",
                "same_as_measured",
                "same_as_prior_low_cost_sc",
                "spatial_prior_sc_basin",
                "same_as_seed1_near_sc_basin",
                "avoids_prior_low_cost_sc",
                "healthy_nonmeasured_candidate",
                "low_cost_artifact",
                "selected_child_distance_from_same_seed_measured_m",
                "selected_child_distance_from_prior_low_cost_sc_reference_m",
                "best_descendant_distance_from_prior_low_cost_sc_reference_m",
            )
        }
        for row in decisions
    ]
    save_json(output_dir / "branch_classification_by_seed_mode.json", branch_rows)
    write_csv(output_dir / "branch_classification_by_seed_mode.csv", branch_rows)
    write_md_table(
        output_dir / "branch_classification_summary.md",
        "Branch Classification Summary",
        mode_summary,
        [
            "mode",
            "branch_classification_counts",
            "same_as_measured_fraction",
            "spatial_prior_sc_basin_fraction",
            "avoids_prior_low_cost_sc_fraction",
            "healthy_nonmeasured_fraction",
            "low_cost_artifact_fraction",
        ],
    )

    save_json(output_dir / "lambda48_behavior_summary.json", lambda_summary)
    write_lambda48_summary_md(output_dir / "lambda48_behavior_summary.md", lambda_summary)

    low_cost_rows = [
        {
            "seed": row["seed"],
            "mode": row["mode"],
            "branch_classification": row.get("branch_classification"),
            "changed_vs_measured_only": row.get("changed_vs_measured_only"),
            "gain_exp": row.get("gain_exp"),
            "source_occ_free_count": row.get("source_occ_free_count"),
            "cost": row.get("cost"),
            "base_exp_selected_child_id": row.get("base_exp_selected_child_id"),
            "base_exp_best_descendant_id": row.get("base_exp_best_descendant_id"),
            "base_exp_selected_gain_exp": row.get("base_exp_selected_gain_exp"),
            "base_exp_selected_source_occ_free": row.get("base_exp_selected_source_occ_free"),
            "base_exp_selected_cost": row.get("base_exp_selected_cost"),
            "low_cost_artifact": row.get("low_cost_artifact"),
            "spatial_prior_sc_basin": row.get("spatial_prior_sc_basin"),
        }
        for row in decisions
    ]
    save_json(output_dir / "low_cost_artifact_diagnosis.json", low_cost_rows)
    write_csv(output_dir / "low_cost_artifact_diagnosis.csv", low_cost_rows)
    write_md_table(
        output_dir / "low_cost_artifact_diagnosis.md",
        "Low-Cost Artifact Diagnosis",
        low_cost_rows,
        [
            "seed",
            "mode",
            "branch_classification",
            "gain_exp",
            "source_occ_free_count",
            "cost",
            "base_exp_selected_gain_exp",
            "base_exp_selected_source_occ_free",
            "base_exp_selected_cost",
            "low_cost_artifact",
        ],
    )

    save_json(output_dir / "comparison_to_stage4a65z_z1.json", comparison)
    write_comparison_md(output_dir / "comparison_to_stage4a65z_z1.md", comparison)
    save_json(output_dir / "tree_generation_summary.json", tree_generation_rows)

    observed_hash_after = sha256_file(observed_path)
    prediction_hash_after = sha256_file(prediction_path)
    checkpoint_hash_after = sha256_file(checkpoint_path) if checkpoint_path.is_file() else None
    hash_checks = {
        "observed_state": {
            "path": str(observed_path),
            "sha256_before": observed_hash_before,
            "sha256_after": observed_hash_after,
            "unchanged": observed_hash_before == observed_hash_after,
        },
        "prediction_npz": {
            "path": str(prediction_path),
            "sha256_before": prediction_hash_before,
            "sha256_after": prediction_hash_after,
            "unchanged": prediction_hash_before == prediction_hash_after,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256_before": checkpoint_hash_before,
            "sha256_after": checkpoint_hash_after,
            "unchanged": checkpoint_hash_before == checkpoint_hash_after,
        },
    }
    save_json(output_dir / "hash_checks.json", hash_checks)

    safety = {
        "isaac_startup": False,
        "new_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "two_frame_runtime": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_rl_ppo_bc_il": False,
        "checkpoint_modified": not bool(hash_checks["checkpoint"]["unchanged"]),
        "existing_observed_state_modified": not bool(hash_checks["observed_state"]["unchanged"]),
        "prediction_npz_modified": not bool(hash_checks["prediction_npz"]["unchanged"]),
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_planning_scoring": False,
        "external_source_modified_built": False,
        "coverage_improvement_claim": False,
        "formula_smoke_only": True,
        "prediction_information_gain_only": True,
    }
    write_safety_report(output_dir, safety)

    missing_report: dict[str, Any] = {
        "missing_required_inputs": missing_inputs,
        "plot_failures": [],
        "optional_modes_implemented": ["raw_hybrid_over_cost", "source_occ_free_no_cost"],
        "notes": [],
    }
    write_plots(output_dir, observed_state, decisions, missing_report)

    next_step, why = write_recommendation(output_dir / "recommended_next_faithful_step.md", lambda_summary)
    map48 = lambda_summary.get("map_predict_lambda48") or {}
    map32 = lambda_summary.get("map_predict_lambda32") or {}
    over = lambda_summary.get("source_occ_free_over_cost") or {}
    answers = {
        "loaded_stage4a65p_frame2_inputs": True,
        "no_isaac_no_capture_no_map_predict_rerun": True,
        "seed_count": len(seeds),
        "mode_count": len({row["mode"] for row in decisions}),
        "decision_row_count": len(decisions),
        "measured_only_reproduced_frame2_reference": lambda_summary["seed0_measured_reference_reproduced"],
        "map_predict_lambda48_seed0_branch": lambda_summary["seed0_map_predict_lambda48"],
        "map_predict_lambda48_branch_counts": lambda_summary["map_predict_lambda48_branch_classification_counts"],
        "map_predict_lambda48_behavior": lambda_summary["map_predict_lambda48_behavior"],
        "lambda48_prior_low_cost_sc_or_spatial_basin": bool(
            (lambda_summary.get("map_predict_lambda48_spatial_prior_sc_basin_fraction") or 0.0) > 0.0
        ),
        "lambda48_low_cost_artifact_fraction": lambda_summary["map_predict_lambda48_low_cost_artifact_fraction"],
        "lambda48_healthy_nonmeasured_fraction": lambda_summary["map_predict_lambda48_healthy_nonmeasured_fraction"],
        "lambda32_vs_lambda48": {
            "lambda32_branch_counts": (map32 or {}).get("branch_classification_counts"),
            "lambda48_branch_counts": (map48 or {}).get("branch_classification_counts"),
            "lambda32_same_as_measured_fraction": (map32 or {}).get("same_as_measured_fraction"),
            "lambda48_same_as_measured_fraction": (map48 or {}).get("same_as_measured_fraction"),
            "lambda32_prior_sc_basin_fraction": (map32 or {}).get("spatial_prior_sc_basin_fraction"),
            "lambda48_prior_sc_basin_fraction": (map48 or {}).get("spatial_prior_sc_basin_fraction"),
        },
        "over_cost_diagnostic": {
            "branch_counts": (over or {}).get("branch_classification_counts"),
            "prior_sc_basin_fraction": (over or {}).get("spatial_prior_sc_basin_fraction"),
            "low_cost_artifact_fraction": (over or {}).get("low_cost_artifact_fraction"),
        },
        "comparison_to_stage4a65z_z1": {
            "did_lambda48_exceed_previous_lambda32_behavior": comparison.get(
                "did_lambda48_exceed_previous_lambda32_behavior"
            ),
            "did_lambda48_still_collapse_to_measured": comparison.get("did_lambda48_still_collapse_to_measured"),
            "did_lambda48_pick_only_old_bad_branch": comparison.get("did_lambda48_pick_only_old_bad_branch"),
            "did_lambda48_find_distinct_candidate": comparison.get("did_lambda48_find_distinct_candidate"),
        },
        "observed_state_hash_unchanged": hash_checks["observed_state"]["unchanged"],
        "prediction_npz_hash_unchanged": hash_checks["prediction_npz"]["unchanged"],
        "checkpoint_hash_unchanged": hash_checks["checkpoint"]["unchanged"],
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
        "coverage_improvement_claimed": False,
    }
    final_summary = write_final_summary(
        output_dir,
        answers=answers,
        lambda_summary=lambda_summary,
        comparison=comparison,
        next_step=next_step,
        why=why,
    )
    missing_report["required_plots"] = {
        name: {
            "exists": (output_dir / name).is_file(),
            "skipped_reason_exists": (output_dir / f"{Path(name).stem}_skipped_reason.md").is_file(),
        }
        for name in REQUIRED_PLOTS
    }
    missing_report["elapsed_s"] = float(time.perf_counter() - start)
    save_json(output_dir / "missing_fields_report.json", missing_report)
    print(json.dumps(to_jsonable(final_summary["answers"]), indent=2, sort_keys=True))
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65p_dir", default=DEFAULT_STAGE4A65P_DIR)
    parser.add_argument("--stage4a65ac_dir", default=DEFAULT_STAGE4A65AC_DIR)
    parser.add_argument("--stage4a65y_dir", default=DEFAULT_STAGE4A65Y_DIR)
    parser.add_argument("--stage4a65z_dir", default=DEFAULT_STAGE4A65Z_DIR)
    parser.add_argument("--stage4a65z1_dir", default=DEFAULT_STAGE4A65Z1_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--observed_state", required=True)
    parser.add_argument("--prediction_npz", required=True)
    parser.add_argument("--pose_json", required=True)
    parser.add_argument("--camera_info_json", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--occ_threshold", type=float, default=0.5)
    parser.add_argument("--free_threshold", type=float, default=0.5)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
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
    parser.add_argument("--short_edge_policy", choices=["reject", "crop", "allow"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--prefer_saved_stage4a65y_trees", action="store_true", default=True)
    parser.add_argument("--rebuild_trees", action="store_false", dest="prefer_saved_stage4a65y_trees")
    parser.add_argument("--save_raw_tree_summaries", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
