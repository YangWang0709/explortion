#!/usr/bin/env python3
"""Stage 4A-6.5ac saved-frame lambda48 formula smoke.

This runner is offline-only. It reads the saved Stage 4A-6.5aa synthetic
frame, read-only Oracle/map_predict prediction NPZs, and saved mini-RRT trees,
then replays one-step tree decisions with the Stage 4A-6.5ab recommended
formula:

    value = gain_exp / cost + 48 * minmax(source_occ_free)

It does not start Isaac, capture frames, rerun map_predict, run SSCNet
inference, execute actions, write predictions into observed_state, or use
prediction for traversability/collision/ray blocking.
"""

from __future__ import annotations

import argparse
import csv
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

from run_synthetic_map_predict_calibration_smoke import (
    EPS,
    load_tree,
    make_config,
    make_frontier_local_mask,
    mean,
    min_median_max,
    normalize_bounds,
    parse_ints,
    path_candidate_rows,
    precompute_segment_prediction_arrays,
    read_json,
    region_mask,
    save_json,
    sha256_file,
    summarize_prediction_npz,
    to_jsonable,
    topdown_observed,
    write_csv,
    write_text,
)
from sim_prediction_layer import SimPredictionLayer


DEFAULT_STAGE4A65AA_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65aa_synthetic_sc_validation"
)
DEFAULT_STAGE4A65AB_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ab_synthetic_calibration_smoke"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ac_saved_frame_lambda48_formula_smoke"
)

REQUIRED_PLOTS = [
    "selected_branches_topdown_lambda48.png",
    "measured_vs_lambda48_topdown.png",
    "oracle_vs_map_predict_lambda48_topdown.png",
    "value_components_lambda48.png",
    "source_occ_free_rank_by_mode.png",
    "low_cost_artifact_by_mode.png",
    "margin_by_mode.png",
]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mode_sort_key(mode: str) -> tuple[int, str]:
    order = {
        "measured_only": 0,
        "oracle_lambda48": 1,
        "map_predict_lambda48": 2,
        "map_predict_lambda32": 3,
        "oracle_over_cost": 4,
        "map_predict_over_cost": 5,
    }
    return order.get(str(mode), 99), str(mode)


def classify_point(root_world: list[float] | None, point_world: list[float] | None) -> str:
    if root_world is None or point_world is None:
        return "unknown"
    dx = float(point_world[0]) - float(root_world[0])
    dy = float(point_world[1]) - float(root_world[1])
    dist = math.hypot(dx, dy)
    if dx > 0.60 and abs(float(point_world[1])) <= 1.20:
        return "toward_hidden_room"
    if dy > 0.65 or float(point_world[1]) > 1.25:
        return "toward_measured_frontier"
    if dist < 0.85:
        return "local_jitter"
    return "other"


def best_by_child(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_child: dict[str, dict[str, Any]] = {}
    for row in scored:
        child_id = str(row.get("selected_child_id"))
        current = by_child.get(child_id)
        if current is None or float(row["final_value"]) > float(current["final_value"]):
            by_child[child_id] = row
    return sorted(by_child.values(), key=lambda row: float(row["final_value"]), reverse=True)


def build_mode_configs(args: argparse.Namespace, map_predict_available: bool) -> list[dict[str, Any]]:
    tau = float(args.tau)
    occ = float(args.occ_threshold)
    free = float(args.free_threshold)
    lambda_sc = float(args.lambda_sc)
    configs: list[dict[str, Any]] = []

    oracle48 = make_config("oracle", "source_occ_free", "decoupled_minmax", tau, occ, free, lambda_sc)
    oracle48.update(
        {
            "mode": "oracle_lambda48",
            "formula": f"gain_exp / cost + {lambda_sc:g} * minmax(source_occ_free)",
            "recommended": True,
        }
    )
    configs.append(oracle48)

    if map_predict_available:
        map48 = make_config("map_predict", "source_occ_free", "decoupled_minmax", tau, occ, free, lambda_sc)
        map48.update(
            {
                "mode": "map_predict_lambda48",
                "formula": f"gain_exp / cost + {lambda_sc:g} * minmax(source_occ_free)",
                "recommended": True,
            }
        )
        configs.append(map48)

        map32 = make_config("map_predict", "source_occ_free", "decoupled_minmax", tau, occ, free, 32.0)
        map32.update(
            {
                "mode": "map_predict_lambda32",
                "formula": "gain_exp / cost + 32 * minmax(source_occ_free)",
                "recommended": False,
                "diagnostic_purpose": "confirm lambda32 was less stable in Stage 4A-6.5ab",
            }
        )
        configs.append(map32)

        map_over = make_config("map_predict", "source_occ_free", "over_cost", tau, occ, free)
        map_over.update(
            {
                "mode": "map_predict_over_cost",
                "formula": "(gain_exp + source_occ_free) / cost",
                "recommended": False,
                "diagnostic_purpose": "source-style useful-but-risky comparison only",
            }
        )
        configs.append(map_over)

    oracle_over = make_config("oracle", "source_occ_free", "over_cost", tau, occ, free)
    oracle_over.update(
        {
            "mode": "oracle_over_cost",
            "formula": "(gain_exp + source_occ_free) / cost",
            "recommended": False,
            "diagnostic_purpose": "Oracle source-style comparison only",
        }
    )
    configs.append(oracle_over)
    return configs


def select_formula_decision(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    mode: str,
    config: dict[str, Any],
    measured_reference: dict[str, Any] | None,
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError(f"candidate rows are empty for seed {seed} mode {mode}")

    utility = str(config["utility_mode"])
    lambda_value = config.get("lambda")
    sc_values = [float(row.get("sc_gain") or 0.0) for row in candidates]
    min_sc = min(sc_values)
    max_sc = max(sc_values)
    sc_denom = max(float(max_sc - min_sc), EPS)

    scored: list[dict[str, Any]] = []
    for row in candidates:
        cost = float(row.get("cost") or 0.0)
        if cost <= EPS:
            continue
        item = dict(row)
        source_occ_free = float(item.get("sc_gain") or 0.0)
        base_exp_value = float(item.get("gain_exp") or 0.0) / max(cost, EPS)
        normalized_sc = float((source_occ_free - min_sc) / sc_denom) if max_sc > min_sc else 0.0

        if utility == "measured":
            sc_bonus = 0.0
            final_value = base_exp_value
        elif utility == "decoupled_minmax":
            sc_bonus = float(lambda_value or 0.0) * normalized_sc
            final_value = base_exp_value + sc_bonus
        elif utility == "over_cost":
            sc_bonus = source_occ_free / max(cost, EPS)
            final_value = (float(item.get("gain_exp") or 0.0) + source_occ_free) / max(cost, EPS)
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

    if not scored:
        raise RuntimeError(f"no scored candidates for seed {seed} mode {mode}")

    ranked = best_by_child(scored)
    winner = dict(ranked[0])
    runner = ranked[1] if len(ranked) > 1 else None
    runner_value = None if runner is None else float(runner["final_value"])
    margin = None if runner_value is None else float(float(winner["final_value"]) - runner_value)
    normalized_margin = None if margin is None else float(margin / max(abs(float(winner["final_value"])), EPS))

    base_ranked = sorted(scored, key=lambda row: float(row["base_exp_value"]), reverse=True)
    base_winner = base_ranked[0]

    root_world = winner.get("root_world")
    selected_child_direction = classify_point(root_world, winner.get("selected_child_world"))
    best_descendant_direction = classify_point(root_world, winner.get("best_descendant_world"))
    measured_direction = None if measured_reference is None else measured_reference.get("best_descendant_direction")
    changed_vs_measured = bool(measured_direction is not None and best_descendant_direction != measured_direction)

    low_cost_vs_measured = False
    if measured_reference is not None and changed_vs_measured:
        low_cost_vs_measured = bool(
            float(winner.get("gain_exp") or 0.0) < float(measured_reference.get("gain_exp") or 0.0)
            and float(winner.get("source_occ_free_count") or winner.get("sc_gain") or 0.0)
            < float(measured_reference.get("source_occ_free_count") or measured_reference.get("sc_gain") or 0.0)
            and float(winner.get("cost") or 0.0) < float(measured_reference.get("cost") or 0.0)
        )
    low_cost_vs_base_exp = bool(
        winner.get("selected_child_id") != base_winner.get("selected_child_id")
        and float(winner.get("gain_exp") or 0.0) < float(base_winner.get("gain_exp") or 0.0)
        and float(winner.get("sc_gain") or 0.0) < float(base_winner.get("sc_gain") or 0.0)
        and float(winner.get("cost") or 0.0) < float(base_winner.get("cost") or 0.0)
    )
    low_cost_artifact = bool(low_cost_vs_measured or low_cost_vs_base_exp)

    formula = str(config.get("formula", "gain_exp / cost"))
    return {
        "seed": int(seed),
        "mode": str(mode),
        "prediction_source": str(config["prediction_source"]),
        "formula": formula,
        "formula_name": str(config["formula_name"]),
        "config_key": str(config["config_key"]),
        "sc_basis": str(config.get("sc_basis", "none")),
        "utility_mode": utility,
        "lambda": lambda_value,
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
        "selected_child_direction": selected_child_direction,
        "best_descendant_direction": best_descendant_direction,
        "branch_direction_label": best_descendant_direction,
        "hidden_room_selected": best_descendant_direction == "toward_hidden_room",
        "measured_frontier_selected": best_descendant_direction == "toward_measured_frontier",
        "changed_vs_measured_only": changed_vs_measured,
        "changed_selected_child_vs_measured_only": bool(
            measured_reference is not None and winner.get("selected_child_id") != measured_reference.get("selected_child_id")
        ),
        "oracle_agreement": None,
        "agreement_with_oracle_same_formula": None,
        "gain_exp": winner.get("gain_exp"),
        "source_occ_free": winner.get("source_occ_free_count"),
        "source_occ_free_count": winner.get("source_occ_free_count"),
        "source_occ_count": winner.get("source_occ_count"),
        "source_free_count": winner.get("source_free_count"),
        "sc_gain": winner.get("sc_gain"),
        "confidence_sum": winner.get("confidence_sum"),
        "confidence_mean": winner.get("confidence_mean"),
        "hidden_region_predicted_count": winner.get("hidden_region_predicted_count"),
        "frontier_local_predicted_count": winner.get("frontier_local_predicted_count"),
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
        "selected_hidden_region_count_rank": winner.get("hidden_region_count_rank"),
        "selected_hidden_region_count": winner.get("hidden_region_predicted_count"),
        "low_cost_artifact": low_cost_artifact,
        "low_cost_artifact_vs_measured_only": low_cost_vs_measured,
        "low_cost_artifact_vs_base_exp": low_cost_vs_base_exp,
        "min_sc": winner.get("min_sc"),
        "max_sc": winner.get("max_sc"),
        "root_grid": winner.get("root_grid"),
        "root_world": winner.get("root_world"),
        "selected_child_distance_from_root_m": winner.get("selected_child_distance_from_root_m"),
        "best_descendant_distance_from_root_m": winner.get("best_descendant_distance_from_root_m"),
        "tree_total_nodes": winner.get("tree_total_nodes"),
        "candidate_count": len(scored),
        "subtree_selection_policy": "SubsequentBest over root immediate children",
        "prediction_safety_flags": {
            "prediction_writeback": False,
            "prediction_used_for_traversability": False,
            "prediction_used_for_collision": False,
            "prediction_ray_blocking": False,
            "target_ground_truth_planning_scoring": False,
        },
        "base_exp_selected_child_id": base_winner.get("selected_child_id"),
        "base_exp_selected_direction": classify_point(base_winner.get("root_world"), base_winner.get("best_descendant_world")),
        "base_exp_selected_cost": base_winner.get("cost"),
        "base_exp_selected_gain_exp": base_winner.get("gain_exp"),
        "base_exp_selected_source_occ_free": base_winner.get("source_occ_free_count"),
    }


def fill_oracle_agreement(rows: list[dict[str, Any]]) -> None:
    oracle_by_seed_mode = {
        (int(row["seed"]), "lambda48"): row
        for row in rows
        if row.get("mode") == "oracle_lambda48"
    }
    oracle_over_by_seed = {
        int(row["seed"]): row
        for row in rows
        if row.get("mode") == "oracle_over_cost"
    }
    for row in rows:
        if row.get("mode") == "oracle_lambda48":
            row["oracle_agreement"] = True
            row["agreement_with_oracle_same_formula"] = True
        elif row.get("mode") == "map_predict_lambda48":
            ref = oracle_by_seed_mode.get((int(row["seed"]), "lambda48"))
            agreement = None if ref is None else row.get("best_descendant_direction") == ref.get("best_descendant_direction")
            row["oracle_agreement"] = agreement
            row["agreement_with_oracle_same_formula"] = agreement
        elif row.get("mode") == "map_predict_over_cost":
            ref = oracle_over_by_seed.get(int(row["seed"]))
            agreement = None if ref is None else row.get("best_descendant_direction") == ref.get("best_descendant_direction")
            row["oracle_agreement"] = agreement
            row["agreement_with_oracle_same_formula"] = agreement


def rows_for_mode(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("mode") == mode]


def fraction(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(bool(row.get(key)) for row in rows) / len(rows))


def summarize_modes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["mode"])].append(row)
    out: list[dict[str, Any]] = []
    for mode in sorted(groups, key=mode_sort_key):
        items = groups[mode]
        agreement_items = [row for row in items if row.get("oracle_agreement") is not None]
        out.append(
            {
                "mode": mode,
                "prediction_source": items[0].get("prediction_source"),
                "formula": items[0].get("formula"),
                "lambda": items[0].get("lambda"),
                "tau": items[0].get("tau"),
                "occ_threshold": items[0].get("occ_threshold"),
                "free_threshold": items[0].get("free_threshold"),
                "seed_count": len(items),
                "direction_counts": dict(Counter(str(row.get("best_descendant_direction")) for row in items)),
                "hidden_room_selection_fraction": fraction(items, "hidden_room_selected"),
                "measured_frontier_selection_fraction": fraction(items, "measured_frontier_selected"),
                "oracle_agreement_fraction": (
                    sum(bool(row.get("oracle_agreement")) for row in agreement_items) / len(agreement_items)
                    if agreement_items
                    else None
                ),
                "low_cost_artifact_fraction": fraction(items, "low_cost_artifact"),
                "margin": min_median_max([float(row.get("margin") or 0.0) for row in items]),
                "mean_selected_source_occ_free": mean([row.get("source_occ_free_count") for row in items]),
                "mean_selected_hidden_region_count": mean([row.get("hidden_region_predicted_count") for row in items]),
                "mean_selected_cost_rank": mean([row.get("selected_cost_rank") for row in items]),
                "mean_selected_source_occ_free_rank": mean([row.get("selected_source_occ_free_rank") for row in items]),
            }
        )
    return out


def write_md_table(path: Path, title: str, rows: list[dict[str, Any]], fields: list[str], limit: int = 80) -> None:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("- no rows")
        write_text(path, "\n".join(lines))
        return
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join(["---"] * len(fields)) + " |")
    for row in rows[:limit]:
        values = [str(to_jsonable(row.get(field, ""))) for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > limit:
        lines.append(f"\nShowing `{limit}` of `{len(rows)}` rows.")
    write_text(path, "\n".join(lines))


def make_formula_definition(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "recommended_formula_name": "source_occ_free_decoupled_minmax_lambda48",
        "recommended_formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
        "lambda": float(args.lambda_sc),
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
        "explicitly_not_recommended": {
            "over_cost": "(gain_exp + source_occ_free) / cost",
            "scaled_over_cost": "(gain_exp + 48 * source_occ_free) / cost",
        },
        "prediction_information_gain_only": True,
    }


def write_formula_definition(output_dir: Path, definition: dict[str, Any]) -> None:
    save_json(output_dir / "formula_definition.json", definition)
    lines = [
        "# Formula Definition",
        "",
        f"- recommended formula: `{definition['recommended_formula']}`",
        f"- lambda: `{definition['lambda']}`",
        f"- tau: `{definition['tau']}`",
        f"- occ/free thresholds: `{definition['occ_threshold']}` / `{definition['free_threshold']}`",
        "- SC is outside the cost denominator.",
        "- over-cost formulas are diagnostic only and not recommended.",
        "- prediction is information-gain-only; it is not used for traversability, collision, ray blocking, or observed-map writeback.",
    ]
    write_text(output_dir / "formula_definition.md", "\n".join(lines))


def summarize_lambda48(
    *,
    seeds: list[int],
    decisions: list[dict[str, Any]],
    mode_summary: list[dict[str, Any]],
) -> dict[str, Any]:
    by_mode = {row["mode"]: row for row in mode_summary}
    measured = rows_for_mode(decisions, "measured_only")
    oracle48 = rows_for_mode(decisions, "oracle_lambda48")
    map48 = rows_for_mode(decisions, "map_predict_lambda48")
    agreement_rows = [row for row in map48 if row.get("oracle_agreement") is not None]
    agreement_fraction = (
        sum(bool(row.get("oracle_agreement")) for row in agreement_rows) / len(agreement_rows)
        if agreement_rows
        else None
    )
    map_low_cost = sum(bool(row.get("low_cost_artifact")) for row in map48)
    success = bool(
        len(measured) >= len(seeds)
        and sum(bool(row.get("measured_frontier_selected")) for row in measured) == len(seeds)
        and len(map48) >= len(seeds)
        and sum(bool(row.get("hidden_room_selected")) for row in map48) == len(seeds)
        and len(oracle48) >= len(seeds)
        and sum(bool(row.get("hidden_room_selected")) for row in oracle48) >= min(4, len(seeds))
        and agreement_fraction is not None
        and agreement_fraction >= 0.8
        and map_low_cost == 0
    )
    return {
        "stage": "Stage 4A-6.5ac",
        "status": "completed" if success else "completed_with_reproduction_warning",
        "seed_count": len(seeds),
        "mode_count": len(by_mode),
        "decision_row_count": len(decisions),
        "measured_only": by_mode.get("measured_only"),
        "oracle_lambda48": by_mode.get("oracle_lambda48"),
        "map_predict_lambda48": by_mode.get("map_predict_lambda48"),
        "map_predict_oracle_lambda48_agreement_fraction": agreement_fraction,
        "map_predict_lambda48_low_cost_artifact_count": map_low_cost,
        "map_predict_lambda48_low_cost_artifact_fraction": None if not map48 else map_low_cost / len(map48),
        "formula_components_logged": all(
            key in row for row in decisions for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value")
        ),
        "success_criteria": {
            "offline_saved_frame_only": True,
            "measured_only_measured_frontier_5of5": sum(bool(row.get("measured_frontier_selected")) for row in measured)
            == len(seeds),
            "map_predict_lambda48_hidden_room_5of5": sum(bool(row.get("hidden_room_selected")) for row in map48)
            == len(seeds),
            "oracle_lambda48_hidden_room_at_least_4of5": sum(bool(row.get("hidden_room_selected")) for row in oracle48)
            >= min(4, len(seeds)),
            "map_predict_oracle_agreement_at_least_0p8": agreement_fraction is not None and agreement_fraction >= 0.8,
            "map_predict_lambda48_low_cost_artifact_zero": map_low_cost == 0,
            "formula_components_logged": True,
            "prediction_safety_preserved": True,
        },
        "saved_frame_only_readiness": success,
        "runtime_smoke_readiness": False,
        "rollout_readiness": False,
    }


def write_lambda48_summary_md(path: Path, summary: dict[str, Any]) -> None:
    measured = summary.get("measured_only") or {}
    oracle = summary.get("oracle_lambda48") or {}
    mp = summary.get("map_predict_lambda48") or {}
    lines = [
        "# Stage 4A-6.5ac Lambda48 Reproduction Summary",
        "",
        f"- status: `{summary['status']}`",
        f"- seeds/modes/decision rows: `{summary['seed_count']}` / `{summary['mode_count']}` / `{summary['decision_row_count']}`",
        f"- measured_only measured-frontier fraction: `{measured.get('measured_frontier_selection_fraction')}`",
        f"- oracle_lambda48 hidden-room fraction: `{oracle.get('hidden_room_selection_fraction')}`",
        f"- map_predict_lambda48 hidden-room fraction: `{mp.get('hidden_room_selection_fraction')}`",
        f"- map_predict/oracle lambda48 agreement: `{summary['map_predict_oracle_lambda48_agreement_fraction']}`",
        f"- map_predict lambda48 low-cost artifact fraction: `{summary['map_predict_lambda48_low_cost_artifact_fraction']}`",
        "- readiness: saved-frame-only; not runtime-smoke-ready; not rollout-ready.",
    ]
    write_text(path, "\n".join(lines))


def oracle_vs_map_predict_lambda48(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    oracle = {int(row["seed"]): row for row in decisions if row.get("mode") == "oracle_lambda48"}
    map_rows = [row for row in decisions if row.get("mode") == "map_predict_lambda48"]
    per_seed: list[dict[str, Any]] = []
    for row in sorted(map_rows, key=lambda item: int(item["seed"])):
        ref = oracle.get(int(row["seed"]))
        per_seed.append(
            {
                "seed": int(row["seed"]),
                "oracle_best_descendant_direction": None if ref is None else ref.get("best_descendant_direction"),
                "map_predict_best_descendant_direction": row.get("best_descendant_direction"),
                "same_direction": None if ref is None else ref.get("best_descendant_direction") == row.get("best_descendant_direction"),
                "same_selected_child_id": None if ref is None else ref.get("selected_child_id") == row.get("selected_child_id"),
                "oracle_best_descendant_grid": None if ref is None else ref.get("best_descendant_grid"),
                "map_predict_best_descendant_grid": row.get("best_descendant_grid"),
                "oracle_final_value": None if ref is None else ref.get("final_value"),
                "map_predict_final_value": row.get("final_value"),
            }
        )
    comparable = [row for row in per_seed if row["same_direction"] is not None]
    return {
        "seed_count": len(per_seed),
        "comparable_seed_count": len(comparable),
        "agreement_fraction": (
            sum(bool(row["same_direction"]) for row in comparable) / len(comparable) if comparable else None
        ),
        "selected_child_agreement_fraction": (
            sum(bool(row["same_selected_child_id"]) for row in comparable) / len(comparable) if comparable else None
        ),
        "per_seed": per_seed,
    }


def compare_to_stage4a65ab(stage4a65ab_dir: Path, mode_summary: list[dict[str, Any]]) -> dict[str, Any]:
    best_path = stage4a65ab_dir / "best_config_candidates.json"
    summary_path = stage4a65ab_dir / "stage4a65ab_synthetic_calibration_summary.json"
    best = read_json(best_path)
    ab_summary = read_json(summary_path)
    recommended = best.get("recommended") or {}
    map48 = next((row for row in mode_summary if row.get("mode") == "map_predict_lambda48"), {})
    return {
        "stage4a65ab_dir": str(stage4a65ab_dir),
        "best_config_candidates_path": str(best_path),
        "stage4a65ab_summary_path": str(summary_path),
        "stage4a65ab_recommended_config_key": recommended.get("config_key"),
        "stage4a65ab_hidden_room_fraction": recommended.get("hidden_room_selection_fraction"),
        "stage4a65ab_oracle_map_predict_agreement_fraction": recommended.get("oracle_map_predict_agreement_fraction"),
        "stage4a65ab_low_cost_artifact_fraction": recommended.get("low_cost_artifact_fraction"),
        "stage4a65ab_median_margin": (recommended.get("margin") or {}).get("median"),
        "stage4a65ac_map_predict_lambda48_hidden_room_fraction": map48.get("hidden_room_selection_fraction"),
        "stage4a65ac_map_predict_lambda48_oracle_agreement_fraction": map48.get("oracle_agreement_fraction"),
        "stage4a65ac_map_predict_lambda48_low_cost_artifact_fraction": map48.get("low_cost_artifact_fraction"),
        "stage4a65ac_map_predict_lambda48_median_margin": (map48.get("margin") or {}).get("median"),
        "same_hidden_room_fraction": recommended.get("hidden_room_selection_fraction")
        == map48.get("hidden_room_selection_fraction"),
        "same_agreement_fraction": recommended.get("oracle_map_predict_agreement_fraction")
        == map48.get("oracle_agreement_fraction"),
        "same_low_cost_artifact_fraction": recommended.get("low_cost_artifact_fraction")
        == map48.get("low_cost_artifact_fraction"),
        "stage4a65ab_decision_rows": (ab_summary.get("answers") or {}).get("decision_row_count"),
        "stage4a65ab_runtime_readiness": recommended.get("runtime_readiness"),
    }


def write_safety_report(output_dir: Path, safety: dict[str, Any]) -> None:
    save_json(output_dir / "prediction_safety_report.json", safety)
    lines = [
        "# Prediction Safety Report",
        "",
        "- prediction writeback: `false`",
        "- prediction used for traversability/collision: `false`",
        "- prediction ray blocking: `false`",
        "- target/ground-truth planning or scoring: `false`",
        "- Isaac startup: `false`",
        "- new capture / map_predict rerun / SSCNet inference: `false`",
        f"- observed_state unchanged: `{not safety['existing_observed_state_modified']}`",
        f"- prediction NPZ unchanged: `{not safety['prediction_npz_modified']}`",
    ]
    write_text(output_dir / "prediction_safety_report.md", "\n".join(lines))


def write_loaded_manifest_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Loaded Inputs Manifest",
        "",
        f"- Stage 4A-6.5aa dir: `{manifest['stage4a65aa_dir']}`",
        f"- Stage 4A-6.5ab dir: `{manifest['stage4a65ab_dir']}`",
        f"- observed_state: `{manifest['observed_state']['path']}`",
        f"- oracle prediction: `{manifest['oracle_prediction']['path']}`",
        f"- map_predict prediction: `{manifest['map_predict_prediction']['path']}`",
        f"- scene metadata: `{manifest['scene_metadata']['path']}`",
        f"- best config candidates: `{manifest['stage4a65ab_best_config_candidates']['path']}`",
        "- no Isaac startup, no new capture, no map_predict rerun.",
    ]
    write_text(path, "\n".join(lines))


def write_recommendation(path: Path, reproduction: dict[str, Any]) -> None:
    success = bool(reproduction.get("saved_frame_only_readiness"))
    oracle = reproduction.get("oracle_lambda48") or {}
    mp = reproduction.get("map_predict_lambda48") or {}
    low_cost = float(reproduction.get("map_predict_lambda48_low_cost_artifact_fraction") or 0.0)
    agreement = reproduction.get("map_predict_oracle_lambda48_agreement_fraction")
    if success:
        next_step = "saved-frame formula smoke on one real medium_three_rooms frame only"
        why = "map_predict lambda48 reproduced hidden-room 5/5 with Oracle agreement >= 0.8 and zero low-cost artifacts."
    elif float((oracle or {}).get("hidden_room_selection_fraction") or 0.0) >= 0.8 and float(
        (mp or {}).get("hidden_room_selection_fraction") or 0.0
    ) < 1.0:
        next_step = "map_predict NPZ threshold/mapping debug"
        why = "Oracle lambda48 succeeds but map_predict lambda48 does not fully reproduce the synthetic target."
    elif low_cost > 0.0:
        next_step = "low-cost artifact diagnosis"
        why = "A low-cost artifact appeared before real-frame testing."
    else:
        next_step = "formula implementation / minmax normalization debug"
        why = "lambda48 did not meet the saved-frame reproduction criteria."
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {next_step}",
        f"- why: {why}",
        f"- map_predict/oracle agreement: `{agreement}`",
        "- runtime smoke readiness: no",
        "- rollout readiness: no",
        "- do not run RL/PPO/BC/IL.",
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
    ax.imshow(topdown_observed(observed_state).T, origin="lower", cmap="Greys", alpha=0.45, interpolation="nearest")
    colors = {
        "measured_only": "#f97316",
        "oracle_lambda48": "#2563eb",
        "map_predict_lambda48": "#059669",
        "map_predict_lambda32": "#7c3aed",
        "oracle_over_cost": "#0f766e",
        "map_predict_over_cost": "#dc2626",
    }
    for row in rows:
        if row.get("mode") not in modes:
            continue
        root = row.get("root_grid")
        best = row.get("best_descendant_grid")
        selected = row.get("selected_child_grid")
        color = colors.get(str(row.get("mode")), "#111827")
        label = f"{row.get('mode')} s{row.get('seed')}"
        if root and best:
            ax.plot([root[0], best[0]], [root[1], best[1]], color=color, linewidth=1.2, alpha=0.55)
        if selected:
            ax.scatter([selected[0]], [selected[1]], color=color, s=24, alpha=0.85)
        if best:
            ax.scatter([best[0]], [best[1]], color=color, marker="*", s=64, alpha=0.9, label=label)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:12], labels[:12], fontsize=6, loc="upper right")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_value_components(path: Path, rows: list[dict[str, Any]]) -> None:
    rows = [row for row in rows if row.get("mode") in {"oracle_lambda48", "map_predict_lambda48"}]
    rows = sorted(rows, key=lambda row: (str(row["mode"]), int(row["seed"])))
    fig, ax = plt.subplots(figsize=(9.5, 5.2), constrained_layout=True)
    x = np.arange(len(rows))
    base = [float(row.get("base_exp_value") or 0.0) for row in rows]
    bonus = [float(row.get("sc_bonus") or 0.0) for row in rows]
    labels = [f"{row['mode'].replace('_lambda48', '')}\ns{row['seed']}" for row in rows]
    ax.bar(x, base, color="#f97316", label="base_exp_value")
    ax.bar(x, bonus, bottom=base, color="#2563eb", label="sc_bonus")
    ax.plot(x, [float(row.get("final_value") or 0.0) for row in rows], color="#111827", marker="o", label="final_value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("value")
    ax.set_title("Lambda48 value components")
    ax.legend()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_rank_by_mode(path: Path, rows: list[dict[str, Any]]) -> None:
    summary = summarize_modes(rows)
    labels = [row["mode"] for row in summary]
    values = [float(row.get("mean_selected_source_occ_free_rank") or 0.0) for row in summary]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(range(len(labels)), values, color="#059669")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("mean selected source_occ_free rank")
    ax.set_title("Source OCC+FREE rank by mode")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_low_cost(path: Path, summary: list[dict[str, Any]]) -> None:
    labels = [row["mode"] for row in summary]
    values = [float(row.get("low_cost_artifact_fraction") or 0.0) for row in summary]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.bar(range(len(labels)), values, color="#dc2626")
    ax.set_ylim(0.0, max(1.0, max(values, default=0.0) + 0.1))
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("low-cost artifact fraction")
    ax.set_title("Low-cost artifact by mode")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def plot_margin(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("margin") is not None:
            groups[str(row["mode"])].append(float(row["margin"]))
    labels = sorted(groups, key=mode_sort_key)
    data = [groups[label] for label in labels]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.boxplot(data, labels=labels, vert=True, showmeans=True)
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylabel("margin")
    ax.set_title("Decision margin by mode")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_plots(
    output_dir: Path,
    observed_state: np.ndarray,
    decisions: list[dict[str, Any]],
    mode_summary: list[dict[str, Any]],
    missing: dict[str, Any],
) -> None:
    plotters = {
        "selected_branches_topdown_lambda48.png": lambda p: plot_selected_branches(
            p,
            observed_state,
            decisions,
            "Selected branches: measured vs lambda48",
            ["measured_only", "oracle_lambda48", "map_predict_lambda48"],
        ),
        "measured_vs_lambda48_topdown.png": lambda p: plot_selected_branches(
            p,
            observed_state,
            decisions,
            "Measured-only vs map_predict lambda48",
            ["measured_only", "map_predict_lambda48"],
        ),
        "oracle_vs_map_predict_lambda48_topdown.png": lambda p: plot_selected_branches(
            p,
            observed_state,
            decisions,
            "Oracle lambda48 vs map_predict lambda48",
            ["oracle_lambda48", "map_predict_lambda48"],
        ),
        "value_components_lambda48.png": lambda p: plot_value_components(p, decisions),
        "source_occ_free_rank_by_mode.png": lambda p: plot_rank_by_mode(p, decisions),
        "low_cost_artifact_by_mode.png": lambda p: plot_low_cost(p, mode_summary),
        "margin_by_mode.png": lambda p: plot_margin(p, decisions),
    }
    for name, plotter in plotters.items():
        try:
            plotter(output_dir / name)
        except Exception as exc:  # pragma: no cover - operational diagnostics
            reason_path = output_dir / f"{Path(name).stem}_skipped_reason.md"
            write_text(reason_path, f"# Plot Skipped\n\n- plot: `{name}`\n- reason: `{type(exc).__name__}: {exc}`")
            missing.setdefault("plot_failures", []).append({"plot": name, "reason": f"{type(exc).__name__}: {exc}"})


def write_final_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    lines = [
        "# Stage 4A-6.5ac Saved-Frame Lambda48 Formula Summary",
        "",
        f"1. Loaded Stage 4A-6.5aa and 6.5ab inputs: `{answers['loaded_inputs']}`.",
        f"2. No Isaac / new capture / map_predict rerun: `{answers['no_isaac_no_capture_no_map_predict_rerun']}`.",
        f"3. Tested seeds/modes/decision rows: `{answers['seed_count']}` / `{answers['mode_count']}` / `{answers['decision_row_count']}`.",
        f"4. measured_only reproduced measured-frontier: `{answers['measured_only_measured_frontier']}`.",
        f"5. map_predict lambda48 reproduced hidden-room 5/5: `{answers['map_predict_lambda48_hidden_room_5of5']}`.",
        f"6. Oracle lambda48 selected hidden-room: `{answers['oracle_lambda48_hidden_room_fraction']}`.",
        f"7. map_predict / Oracle agreement: `{answers['map_predict_oracle_lambda48_agreement']}`.",
        f"8. low-cost artifact for map_predict lambda48: `{answers['map_predict_lambda48_low_cost_artifact_fraction']}`.",
        f"9. formula used: `{answers['formula']}`.",
        f"10. over-cost used as recommended config: `{answers['over_cost_recommended']}`.",
        f"11. Consistent with 6.5ab best_config_candidates: `{answers['consistent_with_stage4a65ab']}`.",
        f"12. readiness: `{answers['readiness']}`.",
        f"13. next recommendation: `{answers['next_recommendation']}`.",
        "",
        "This is saved-frame-only evidence and does not claim coverage improvement.",
    ]
    write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    stage4a65aa_dir = Path(args.stage4a65aa_dir).resolve()
    stage4a65ab_dir = Path(args.stage4a65ab_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_ints(args.seeds)

    observed_path = stage4a65aa_dir / "observed_state_synthetic_frame000.npy"
    oracle_npz = stage4a65aa_dir / "oracle_global_prediction_layer.npz"
    map_npz = stage4a65aa_dir / "map_predict" / "global_prediction_layer.npz"
    pose_json = stage4a65aa_dir / "pose_000.json"
    camera_info = stage4a65aa_dir / "camera_info.json"
    scene_metadata_path = stage4a65aa_dir / "scene_metadata.json"
    aa_decisions_path = stage4a65aa_dir / "per_seed_mode_decisions.csv"
    aa_branch_path = stage4a65aa_dir / "branch_direction_classification.csv"
    ab_best_path = stage4a65ab_dir / "best_config_candidates.json"
    ab_best_md = stage4a65ab_dir / "best_config_candidates.md"
    ab_summary_path = stage4a65ab_dir / "stage4a65ab_synthetic_calibration_summary.json"

    required = [
        observed_path,
        oracle_npz,
        map_npz,
        pose_json,
        camera_info,
        scene_metadata_path,
        aa_decisions_path,
        aa_branch_path,
        ab_best_path,
        ab_best_md,
        ab_summary_path,
    ]
    missing_required = [str(path) for path in required if not path.is_file()]
    if missing_required:
        raise FileNotFoundError(f"missing required inputs: {missing_required}")

    observed_hash_before = sha256_file(observed_path)
    oracle_hash_before = sha256_file(oracle_npz)
    map_hash_before = sha256_file(map_npz)
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
        "stage4a65aa_dir": str(stage4a65aa_dir),
        "stage4a65ab_dir": str(stage4a65ab_dir),
        "observed_state": {"path": str(observed_path), "sha256": observed_hash_before, "shape": list(observed_state.shape)},
        "oracle_prediction": summarize_prediction_npz(oracle_npz, observed_state, hidden_mask),
        "map_predict_prediction": summarize_prediction_npz(map_npz, observed_state, hidden_mask),
        "pose_json": {"path": str(pose_json), "sha256": sha256_file(pose_json)},
        "camera_info": {"path": str(camera_info), "sha256": sha256_file(camera_info)},
        "scene_metadata": {"path": str(scene_metadata_path), "sha256": sha256_file(scene_metadata_path)},
        "stage4a65aa_per_seed_mode_decisions": {"path": str(aa_decisions_path), "rows": len(read_csv_rows(aa_decisions_path))},
        "stage4a65aa_branch_direction_classification": {"path": str(aa_branch_path), "rows": len(read_csv_rows(aa_branch_path))},
        "stage4a65ab_best_config_candidates": {"path": str(ab_best_path), "sha256": sha256_file(ab_best_path)},
        "stage4a65ab_best_config_candidates_md": {"path": str(ab_best_md), "sha256": sha256_file(ab_best_md)},
        "stage4a65ab_summary": {"path": str(ab_summary_path), "sha256": sha256_file(ab_summary_path)},
        "map_predict_available": True,
    }
    save_json(output_dir / "loaded_inputs_manifest.json", loaded_manifest)
    write_loaded_manifest_md(output_dir / "loaded_inputs_manifest.md", loaded_manifest)

    formula_definition = make_formula_definition(args)
    write_formula_definition(output_dir, formula_definition)

    configs = build_mode_configs(args, map_predict_available=True)
    prediction_layers = {
        "oracle": SimPredictionLayer.from_npz(oracle_npz),
        "map_predict": SimPredictionLayer.from_npz(map_npz),
    }
    decisions: list[dict[str, Any]] = []
    measured_refs: dict[int, dict[str, Any]] = {}

    for seed in seeds:
        measured_tree_dir = stage4a65aa_dir / "raw_trees" / f"seed_{seed:03d}" / "measured_only"
        measured_tree = load_tree(measured_tree_dir)
        measured_config = {
            "prediction_source": "none",
            "formula_name": "measured_only",
            "config_key": "none|measured_only",
            "sc_basis": "none",
            "utility_mode": "measured",
            "lambda": None,
            "confidence_threshold": None,
            "occ_threshold": None,
            "free_threshold": None,
            "formula": "gain_exp / cost",
        }
        measured_candidates = path_candidate_rows(measured_tree, None, None)
        measured_decision = select_formula_decision(
            measured_candidates,
            seed=seed,
            mode="measured_only",
            config=measured_config,
            measured_reference=None,
        )
        measured_decision["changed_vs_measured_only"] = False
        measured_refs[seed] = measured_decision
        decisions.append(measured_decision)

        trees: dict[str, Any] = {}
        arrays: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        for source, prediction in prediction_layers.items():
            tree_dir = stage4a65aa_dir / "raw_trees" / f"seed_{seed:03d}" / f"{source}_raw_count"
            trees[source] = load_tree(tree_dir)
            arrays[source] = precompute_segment_prediction_arrays(
                trees[source],
                observed_state,
                prediction,
                hidden_mask,
                frontier_local_mask,
                args,
            )

        for config in configs:
            source = str(config["prediction_source"])
            candidates = path_candidate_rows(trees[source], arrays[source], config)
            decision = select_formula_decision(
                candidates,
                seed=seed,
                mode=str(config["mode"]),
                config=config,
                measured_reference=measured_refs[seed],
            )
            decisions.append(decision)

    fill_oracle_agreement(decisions)
    decisions = sorted(decisions, key=lambda row: (int(row["seed"]), mode_sort_key(str(row["mode"]))))
    mode_summary = summarize_modes(decisions)
    reproduction_summary = summarize_lambda48(seeds=seeds, decisions=decisions, mode_summary=mode_summary)
    oracle_map = oracle_vs_map_predict_lambda48(decisions)
    comparison = compare_to_stage4a65ab(stage4a65ab_dir, mode_summary)

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
        "selected_child_direction",
        "best_descendant_direction",
        "hidden_room_selected",
        "measured_frontier_selected",
        "changed_vs_measured_only",
        "oracle_agreement",
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
        "selected_hidden_region_count",
        "low_cost_artifact",
        "min_sc",
        "max_sc",
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
            "best_descendant_direction",
            "hidden_room_selected",
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
                "prediction_source",
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
                "prediction_source",
                "selected_child_direction",
                "best_descendant_direction",
                "hidden_room_selected",
                "measured_frontier_selected",
                "changed_vs_measured_only",
                "oracle_agreement",
                "low_cost_artifact",
                "selected_child_grid",
                "best_descendant_grid",
            )
        }
        for row in decisions
    ]
    save_json(output_dir / "branch_direction_classification.json", branch_rows)
    write_csv(output_dir / "branch_direction_classification.csv", branch_rows)
    write_md_table(
        output_dir / "branch_direction_summary.md",
        "Branch Direction Summary",
        mode_summary,
        [
            "mode",
            "direction_counts",
            "hidden_room_selection_fraction",
            "measured_frontier_selection_fraction",
            "oracle_agreement_fraction",
            "low_cost_artifact_fraction",
        ],
    )

    save_json(output_dir / "lambda48_reproduction_summary.json", reproduction_summary)
    write_lambda48_summary_md(output_dir / "lambda48_reproduction_summary.md", reproduction_summary)
    save_json(output_dir / "oracle_vs_map_predict_lambda48.json", oracle_map)
    write_md_table(
        output_dir / "oracle_vs_map_predict_lambda48.md",
        "Oracle vs map_predict Lambda48",
        oracle_map["per_seed"],
        ["seed", "oracle_best_descendant_direction", "map_predict_best_descendant_direction", "same_direction"],
    )

    low_cost_rows = [
        {
            "seed": row["seed"],
            "mode": row["mode"],
            "changed_vs_measured_only": row.get("changed_vs_measured_only"),
            "gain_exp": row.get("gain_exp"),
            "source_occ_free_count": row.get("source_occ_free_count"),
            "cost": row.get("cost"),
            "base_exp_selected_gain_exp": row.get("base_exp_selected_gain_exp"),
            "base_exp_selected_source_occ_free": row.get("base_exp_selected_source_occ_free"),
            "base_exp_selected_cost": row.get("base_exp_selected_cost"),
            "low_cost_artifact_vs_measured_only": row.get("low_cost_artifact_vs_measured_only"),
            "low_cost_artifact_vs_base_exp": row.get("low_cost_artifact_vs_base_exp"),
            "low_cost_artifact": row.get("low_cost_artifact"),
        }
        for row in decisions
        if row["mode"] != "measured_only"
    ]
    save_json(output_dir / "low_cost_artifact_diagnosis.json", low_cost_rows)
    write_csv(output_dir / "low_cost_artifact_diagnosis.csv", low_cost_rows)
    write_text(
        output_dir / "low_cost_artifact_diagnosis.md",
        "\n".join(
            [
                "# Low-Cost Artifact Diagnosis",
                "",
                f"- rows checked: `{len(low_cost_rows)}`",
                f"- flagged rows: `{sum(bool(row['low_cost_artifact']) for row in low_cost_rows)}`",
                "- recommended map_predict lambda48 low-cost artifact fraction: "
                f"`{reproduction_summary['map_predict_lambda48_low_cost_artifact_fraction']}`",
                "- definition: changed branch with lower gain_exp, lower source_occ_free, and lower cost versus measured-only or the same-tree base-exp winner.",
            ]
        ),
    )

    save_json(output_dir / "comparison_to_stage4a65ab.json", comparison)
    write_text(
        output_dir / "comparison_to_stage4a65ab.md",
        "\n".join(
            [
                "# Comparison To Stage 4A-6.5ab",
                "",
                f"- 6.5ab best config: `{comparison['stage4a65ab_recommended_config_key']}`",
                f"- hidden-room fraction same: `{comparison['same_hidden_room_fraction']}`",
                f"- oracle/map_predict agreement same: `{comparison['same_agreement_fraction']}`",
                f"- low-cost artifact fraction same: `{comparison['same_low_cost_artifact_fraction']}`",
                f"- 6.5ab median margin: `{comparison['stage4a65ab_median_margin']}`",
                f"- 6.5ac median margin: `{comparison['stage4a65ac_map_predict_lambda48_median_margin']}`",
            ]
        ),
    )

    missing: dict[str, Any] = {
        "plot_failures": [],
        "required_plots": REQUIRED_PLOTS,
        "missing_required_inputs": [],
        "fields_missing": [],
    }
    if args.save_viz:
        write_plots(output_dir, observed_state, decisions, mode_summary, missing)
    else:
        for name in REQUIRED_PLOTS:
            reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
            write_text(reason, f"# Plot Skipped\n\n- plot: `{name}`\n- reason: `--save_viz was not passed`")

    observed_hash_after = sha256_file(observed_path)
    oracle_hash_after = sha256_file(oracle_npz)
    map_hash_after = sha256_file(map_npz)
    hash_checks = {
        "observed_state": {
            "path": str(observed_path),
            "before": observed_hash_before,
            "after": observed_hash_after,
            "unchanged": observed_hash_before == observed_hash_after,
        },
        "oracle_prediction_npz": {
            "path": str(oracle_npz),
            "before": oracle_hash_before,
            "after": oracle_hash_after,
            "unchanged": oracle_hash_before == oracle_hash_after,
        },
        "map_predict_prediction_npz": {
            "path": str(map_npz),
            "before": map_hash_before,
            "after": map_hash_after,
            "unchanged": map_hash_before == map_hash_after,
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
        "leakage": False,
        "observed_state_sha256_before": observed_hash_before,
        "observed_state_sha256_after": observed_hash_after,
        "oracle_npz_sha256_before": oracle_hash_before,
        "oracle_npz_sha256_after": oracle_hash_after,
        "map_predict_npz_sha256_before": map_hash_before,
        "map_predict_npz_sha256_after": map_hash_after,
    }
    write_safety_report(output_dir, safety)

    missing["safety"] = safety
    missing["all_formula_component_fields_present"] = bool(reproduction_summary["formula_components_logged"])
    save_json(output_dir / "missing_fields_report.json", missing)

    recommendation_success = bool(reproduction_summary.get("saved_frame_only_readiness"))
    next_recommendation = (
        "saved-frame formula smoke on one real medium_three_rooms frame only"
        if recommendation_success
        else "formula implementation / minmax normalization debug"
    )
    answers = {
        "loaded_inputs": True,
        "no_isaac_no_capture_no_map_predict_rerun": True,
        "seed_count": len(seeds),
        "mode_count": len(mode_summary),
        "decision_row_count": len(decisions),
        "measured_only_measured_frontier": (reproduction_summary["measured_only"] or {}).get(
            "measured_frontier_selection_fraction"
        ),
        "map_predict_lambda48_hidden_room_5of5": reproduction_summary["success_criteria"][
            "map_predict_lambda48_hidden_room_5of5"
        ],
        "oracle_lambda48_hidden_room_fraction": (reproduction_summary["oracle_lambda48"] or {}).get(
            "hidden_room_selection_fraction"
        ),
        "map_predict_oracle_lambda48_agreement": reproduction_summary[
            "map_predict_oracle_lambda48_agreement_fraction"
        ],
        "map_predict_lambda48_low_cost_artifact_fraction": reproduction_summary[
            "map_predict_lambda48_low_cost_artifact_fraction"
        ],
        "formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
        "over_cost_recommended": False,
        "consistent_with_stage4a65ab": bool(
            comparison["same_hidden_room_fraction"]
            and comparison["same_agreement_fraction"]
            and comparison["same_low_cost_artifact_fraction"]
        ),
        "readiness": {
            "saved_frame_only": recommendation_success,
            "runtime_smoke_ready": False,
            "rollout_ready": False,
        },
        "next_recommendation": next_recommendation,
    }
    summary = {
        "stage": "Stage 4A-6.5ac",
        "status": reproduction_summary["status"],
        "stage4a65aa_dir": str(stage4a65aa_dir),
        "stage4a65ab_dir": str(stage4a65ab_dir),
        "output_dir": str(output_dir),
        "seeds": seeds,
        "modes": [row["mode"] for row in mode_summary],
        "decision_row_count": len(decisions),
        "mode_summary": mode_summary,
        "lambda48_reproduction_summary": reproduction_summary,
        "oracle_vs_map_predict_lambda48": oracle_map,
        "comparison_to_stage4a65ab": comparison,
        "answers": answers,
        "safety": safety,
        "runtime_s": float(time.perf_counter() - start),
        "coverage_improvement_claimed": False,
    }
    save_json(output_dir / "stage4a65ac_saved_frame_lambda48_formula_summary.json", summary)
    write_final_summary_md(output_dir / "stage4a65ac_saved_frame_lambda48_formula_summary.md", summary)
    write_recommendation(output_dir / "recommended_next_faithful_step.md", reproduction_summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65aa_dir", default=DEFAULT_STAGE4A65AA_DIR)
    parser.add_argument("--stage4a65ab_dir", default=DEFAULT_STAGE4A65AB_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--occ_threshold", type=float, default=0.5)
    parser.add_argument("--free_threshold", type=float, default=0.5)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
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
