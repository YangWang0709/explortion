#!/usr/bin/env python3
"""Stage 4A-6.5z.1 decoupled SC signal-strength diagnosis.

This script is offline-only. It reads saved Stage 4A-6.5z tables and the saved
Stage 4A-6.5y raw mini-RRT trees, then recomputes diagnostic per-path SC
components from the saved Frame 2 observed map and prediction NPZ.

It never starts Isaac, captures frames, reruns map_predict/SSCNet inference,
executes actions, trains, writes prediction into observed_state, or uses
prediction for traversability/collision/ray blocking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import sha256_file, to_jsonable
from run_decoupled_sc_utility_sweep import (
    DEFAULT_CAMERA_INFO_JSON,
    DEFAULT_OBSERVED_STATE,
    DEFAULT_POSE_JSON,
    DEFAULT_PREDICTION_NPZ,
    DEFAULT_STAGE4A65P_DIR,
    DEFAULT_STAGE4A65X_DIR,
    DEFAULT_STAGE4A65Y_DIR,
    build_candidate_rows,
    build_node_components,
    label_float,
    plot_base_map,
    resolve_raw_tree_dir,
)
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
    classify_row,
    default_bounds,
    euclidean,
    load_tree_segments,
    percentile_summary,
    read_json,
    same_grid,
)
from sim_paper_expert import grid_to_world
from sim_prediction_layer import SimPredictionLayer


DEFAULT_STAGE4A65Z_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65z_decoupled_sc_utility_sweep"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65z1_decoupled_signal_strength_diagnosis"
)
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"
DEFAULT_SC_BASES = (
    "source_occ_free,"
    "parent_visible_cleared_source_occ_free,"
    "frontier_local_source_occ_free"
)

REQUIRED_OUTPUTS = [
    "loaded_6p5z_inputs_manifest.json",
    "loaded_6p5z_inputs_manifest.md",
    "near_miss_branch_table.csv",
    "near_miss_branch_table.json",
    "near_miss_branch_summary.md",
    "required_lambda_to_flip.csv",
    "required_lambda_to_flip.json",
    "required_lambda_to_flip_summary.md",
    "normalization_diagnostics_by_seed.csv",
    "normalization_diagnostics_by_seed.json",
    "normalization_diagnostics_summary.md",
    "adaptive_lambda_gap_analysis.csv",
    "adaptive_lambda_gap_analysis.json",
    "adaptive_lambda_gap_analysis.md",
    "measured_vs_nonmeasured_sc_rank.csv",
    "measured_vs_nonmeasured_sc_rank.json",
    "measured_vs_nonmeasured_sc_rank.md",
    "impossible_under_positive_lambda.csv",
    "impossible_under_positive_lambda.json",
    "low_cost_artifact_followup.csv",
    "low_cost_artifact_followup.json",
    "low_cost_artifact_followup.md",
    "debug_tree_regeneration_report.json",
    "debug_tree_regeneration_report.md",
    "missing_fields_report.json",
    "stage4a65z1_decoupled_signal_strength_summary.json",
    "stage4a65z1_decoupled_signal_strength_summary.md",
    "recommended_next_faithful_step.md",
]

REQUIRED_PLOTS = [
    "required_lambda_distribution.png",
    "required_lambda_by_seed.png",
    "adaptive_lambda_vs_required_lambda.png",
    "measured_vs_best_nonmeasured_base_exp.png",
    "measured_vs_best_nonmeasured_normalized_sc.png",
    "measured_vs_best_nonmeasured_final_value_gap.png",
    "normalized_sc_distribution_by_seed.png",
    "sc_rank_of_measured_winner.png",
    "near_miss_topdown.png",
    "value_component_near_miss_stack.png",
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

EXPECTED_6P5Z_INPUTS = {
    "formula_definitions.json": None,
    "formula_definitions.md": None,
    "sc_component_mapping_report.json": "decoupled_sc_mapping_report.json",
    "sc_component_mapping_report.md": "decoupled_sc_mapping_report.md",
    "adaptive_lambda_report.csv": "adaptive_lambda_values.csv",
    "adaptive_lambda_report.json": "adaptive_lambda_values.json",
    "adaptive_lambda_report.md": None,
    "per_seed_formula_decisions.csv": "decoupled_sc_sweep_decisions.csv",
    "per_seed_formula_decisions.json": "decoupled_sc_sweep_decisions.json",
    "per_seed_formula_decisions.md": "decoupled_sc_sweep_decisions.md",
    "per_seed_formula_value_components.csv": "decoupled_candidate_topk.csv",
    "per_seed_formula_value_components.json": "decoupled_candidate_topk.json",
    "branch_classification_by_formula_seed.csv": "branch_classification_by_formula_seed.csv",
    "branch_classification_by_formula_seed.json": "branch_classification_by_formula_seed.json",
    "branch_classification_summary_by_formula.json": "branch_classification_summary_by_formula.json",
    "branch_classification_summary_by_formula.md": None,
    "low_cost_artifact_diagnosis.csv": None,
    "low_cost_artifact_diagnosis.json": "seed0_base_gap_report.json",
    "low_cost_artifact_diagnosis.md": "seed0_base_gap_report.md",
    "lambda_sensitivity_summary.csv": "lambda_sweep_summary_by_basis_variant.csv",
    "lambda_sensitivity_summary.json": "lambda_sweep_summary_by_basis_variant.json",
    "lambda_sensitivity_summary.md": None,
    "margin_summary_by_formula.csv": None,
    "margin_summary_by_formula.json": None,
    "margin_summary_by_formula.md": None,
    "best_formula_summary.json": None,
    "best_formula_summary.md": None,
    "stage4a65z_decoupled_sc_utility_summary.json": "stage4a65z_decoupled_sc_utility_sweep_summary.json",
    "stage4a65z_decoupled_sc_utility_summary.md": "stage4a65z_decoupled_sc_utility_sweep_summary.md",
}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


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


def load_json_any(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def parse_grid(raw: str | list[int]) -> list[int]:
    if isinstance(raw, list):
        return [int(v) for v in raw]
    return [int(item.strip()) for item in str(raw).split(",") if item.strip()]


def custom_reference_worlds(
    observed_state: np.ndarray,
    voxel_size: float,
    sc_selected: list[int],
    sc_best: list[int],
    measured_selected: list[int],
    measured_best: list[int],
) -> dict[str, Any]:
    bounds = default_bounds(tuple(int(v) for v in observed_state.shape), voxel_size)
    return {
        "seed0_sc_selected_grid": sc_selected,
        "seed0_sc_best_grid": sc_best,
        "seed0_measured_selected_grid": measured_selected,
        "seed0_measured_best_grid": measured_best,
        "seed0_sc_selected_world": list(grid_to_world(sc_selected, bounds, voxel_size)),
        "seed0_sc_best_world": list(grid_to_world(sc_best, bounds, voxel_size)),
        "seed0_measured_selected_world": list(grid_to_world(measured_selected, bounds, voxel_size)),
        "seed0_measured_best_world": list(grid_to_world(measured_best, bounds, voxel_size)),
    }


def rank_desc(rows: list[dict[str, Any]], key: str, target_index: int) -> int | None:
    ranked = sorted(
        rows,
        key=lambda row: (as_float(row.get(key), float("-inf")), str(row.get("node_id"))),
        reverse=True,
    )
    for idx, row in enumerate(ranked, start=1):
        if int(row["candidate_index"]) == int(target_index):
            return idx
    return None


def finite_percentiles(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def add_log_normalized_sc(rows: list[dict[str, Any]]) -> None:
    raw_values = [max(0.0, as_float(row.get("accumulated_formula_effective_sc"), 0.0)) for row in rows]
    denom = math.log1p(max(raw_values) if raw_values else 0.0)
    if denom <= 1.0e-12 or not math.isfinite(denom):
        denom = 1.0
    for row in rows:
        raw = max(0.0, as_float(row.get("accumulated_formula_effective_sc"), 0.0))
        row["log_normalized_sc"] = float(math.log1p(raw) / denom)


def scored_copy(row: dict[str, Any], lambda_value: float) -> dict[str, Any]:
    scored = dict(row)
    normalized = as_float(row.get("normalized_sc"), 0.0)
    base = as_float(row.get("base_exp_value"), float("-inf"))
    scored["lambda_value"] = float(lambda_value)
    scored["lambda_sc_bonus"] = float(lambda_value) * normalized
    scored["final_value"] = base + float(lambda_value) * normalized
    scored["value"] = scored["final_value"]
    return scored


def ranked_children_for_lambda(rows: list[dict[str, Any]], lambda_value: float) -> list[dict[str, Any]]:
    best_by_child: dict[str, dict[str, Any]] = {}
    for row in rows:
        scored = scored_copy(row, lambda_value)
        child = str(scored.get("selected_child_id"))
        current = best_by_child.get(child)
        row_key = (
            as_float(scored.get("final_value"), float("-inf")),
            as_float(scored.get("base_exp_value"), float("-inf")),
            as_float(scored.get("normalized_sc"), float("-inf")),
            str(scored.get("node_id")),
        )
        if current is None:
            best_by_child[child] = scored
            continue
        current_key = (
            as_float(current.get("final_value"), float("-inf")),
            as_float(current.get("base_exp_value"), float("-inf")),
            as_float(current.get("normalized_sc"), float("-inf")),
            str(current.get("node_id")),
        )
        if row_key > current_key:
            best_by_child[child] = scored
    return sorted(
        best_by_child.values(),
        key=lambda row: (
            as_float(row.get("final_value"), float("-inf")),
            as_float(row.get("base_exp_value"), float("-inf")),
            as_float(row.get("normalized_sc"), float("-inf")),
            str(row.get("selected_child_id")),
        ),
        reverse=True,
    )


def classify_candidate(
    candidate: dict[str, Any],
    measured_by_seed: dict[int, dict[str, Any]],
    refs: dict[str, Any],
) -> dict[str, Any]:
    measured = measured_by_seed[int(candidate["seed"])]
    row = dict(candidate)
    row["formula"] = row.get("formula", f"diagnostic_{row.get('sc_basis')}")
    return classify_row(row, measured, refs)


def required_lambda(
    branch_base: float,
    branch_norm: float,
    measured_base: float,
    measured_norm: float,
) -> tuple[float | None, bool, bool]:
    if branch_base > measured_base:
        return 0.0, False, True
    if branch_norm <= measured_norm + 1.0e-12:
        return None, True, False
    return max(0.0, (measured_base - branch_base) / (branch_norm - measured_norm)), False, False


def collect_input_manifest(stage4a65z_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for expected, alias in EXPECTED_6P5Z_INPUTS.items():
        expected_path = stage4a65z_dir / expected
        alias_path = stage4a65z_dir / alias if alias else None
        loaded_path = expected_path if expected_path.is_file() else alias_path if alias_path and alias_path.is_file() else None
        status = "loaded_expected" if expected_path.is_file() else "loaded_alias" if loaded_path else "missing"
        row = {
            "expected_name": expected,
            "alias_used": alias,
            "status": status,
            "loaded_path": str(loaded_path) if loaded_path else None,
            "sha256": sha256_file(loaded_path) if loaded_path else None,
            "size_bytes": loaded_path.stat().st_size if loaded_path else None,
        }
        manifest_rows.append(row)
        if status == "missing":
            missing.append(
                {
                    "field": expected,
                    "reason": "Stage 4A-6.5z output did not contain this exact expected file or a known alias.",
                    "severity": "info",
                }
            )
    manifest = {
        "stage4a65z_dir": str(stage4a65z_dir),
        "inputs": manifest_rows,
        "missing_count": len(missing),
        "loaded_count": sum(1 for row in manifest_rows if row["status"] != "missing"),
    }
    return manifest, missing


def write_manifest_md(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Loaded 6.5z Inputs Manifest",
        "",
        f"- input dir: `{manifest['stage4a65z_dir']}`",
        f"- loaded files: `{manifest['loaded_count']}`",
        f"- missing exact/alias files: `{manifest['missing_count']}`",
        "",
        "| expected | status | alias | loaded path |",
        "|---|---|---|---|",
    ]
    for row in manifest["inputs"]:
        lines.append(
            f"| `{row['expected_name']}` | `{row['status']}` | `{row.get('alias_used')}` | "
            f"`{row.get('loaded_path')}` |"
        )
    write_text(path, "\n".join(lines))


def summarize_formula_rows(class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in class_rows:
        by_formula[str(row["formula"])].append(row)
    for formula, rows in sorted(by_formula.items()):
        summary[formula] = {
            "seed_count": len(rows),
            "same_as_measured_fraction": sum(bool(r.get("same_as_measured_for_seed")) for r in rows) / max(1, len(rows)),
            "spatial_seed0_sc_basin_fraction": sum(bool(r.get("spatial_seed0_sc_basin")) for r in rows) / max(1, len(rows)),
            "primary_counts": dict(Counter(str(r.get("primary_classification")) for r in rows)),
        }
    return summary


def regenerate_candidate_tables(
    *,
    args: argparse.Namespace,
    observed_state: np.ndarray,
    prediction: SimPredictionLayer,
    seeds: list[int],
    sc_bases: list[str],
    measured_by_seed: dict[int, dict[str, Any]],
    refs: dict[str, Any],
) -> tuple[dict[tuple[int, str], list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    stage4a65y_dir = Path(args.stage4a65y_dir).resolve()
    visible_cache: dict[tuple[tuple[int, int, int], float, int, int], set[tuple[int, int, int]]] = {}
    candidate_by_key: dict[tuple[int, str], list[dict[str, Any]]] = {}
    normalization_rows: list[dict[str, Any]] = []
    regen_rows: list[dict[str, Any]] = []
    candidate_index = 0
    started = time.perf_counter()

    for seed in seeds:
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
        regen_rows.append(
            {
                "seed": seed,
                "stage": "load_saved_raw_tree_and_recompute_source_components",
                "tree_dir": str(tree_dir),
                "node_count": len(tree),
                "root_visible_count": len(root_visible),
                "root_source_count": len(root_source),
                "elapsed_s": time.perf_counter() - before_seed,
            }
        )
        for sc_basis in sc_bases:
            before_basis = time.perf_counter()
            rows, norm_summary = build_candidate_rows(
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
            add_log_normalized_sc(rows)
            for row in rows:
                row["candidate_index"] = candidate_index
                candidate_index += 1
                cls = classify_candidate(row, measured_by_seed, refs)
                row.update(
                    {
                        "primary_classification": cls["primary_classification"],
                        "same_as_measured_for_seed": cls["same_as_measured_for_seed"],
                        "spatial_seed0_sc_basin": cls["spatial_seed0_sc_basin"],
                        "exact_seed0_sc": cls["exact_seed0_sc"],
                        "distinct_sc_branch": cls["distinct_sc_branch"],
                        "measured_but_seed0_sc_basin": cls["measured_but_seed0_sc_basin"],
                        "selected_to_same_seed_measured_m": cls["selected_to_same_seed_measured_m"],
                        "best_to_same_seed_measured_m": cls["best_to_same_seed_measured_m"],
                        "selected_to_seed0_sc_reference_m": cls["selected_to_seed0_sc_reference_m"],
                        "best_to_seed0_sc_reference_m": cls["best_to_seed0_sc_reference_m"],
                        "is_reference_seed0_sc_branch": bool(
                            same_grid(row.get("selected_child_grid"), REFERENCE_SC_SELECTED_GRID)
                            and same_grid(row.get("best_descendant_grid"), REFERENCE_SC_BEST_GRID)
                        ),
                    }
                )
            candidate_by_key[(seed, sc_basis)] = rows
            normalization_rows.append(norm_summary)
            regen_rows.append(
                {
                    "seed": seed,
                    "sc_basis": sc_basis,
                    "stage": "candidate_rows_recomputed",
                    "candidate_path_count": len(rows),
                    "normalization_denominator": norm_summary.get("normalization_denominator"),
                    "lambda_base": norm_summary.get("lambda_base_p90_minus_p50_base_exp_value"),
                    "elapsed_s": time.perf_counter() - before_basis,
                }
            )

    report = {
        "recomputed_debug_tree_nodes": True,
        "reason": "6.5z saved only top-k candidate rows; full per-node/per-path rows are needed for required-lambda and rank diagnostics.",
        "seeds": seeds,
        "sc_bases": sc_bases,
        "candidate_path_rows": candidate_index,
        "visible_cache_entries": len(visible_cache),
        "rows": regen_rows,
        "elapsed_s": time.perf_counter() - started,
    }
    return candidate_by_key, normalization_rows, report


def measured_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    same = [row for row in rows if bool(row.get("same_as_measured_for_seed"))]
    pool = same or rows
    return max(
        pool,
        key=lambda row: (
            as_float(row.get("base_exp_value"), float("-inf")),
            as_float(row.get("normalized_sc"), float("-inf")),
            str(row.get("node_id")),
        ),
    )


def build_near_miss_rows(
    decisions: list[dict[str, Any]],
    candidate_by_key: dict[tuple[int, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for decision in decisions:
        seed = int(decision["seed"])
        sc_basis = str(decision["sc_basis"])
        lambda_value = as_float(decision.get("lambda_value"), 0.0)
        candidates = candidate_by_key.get((seed, sc_basis), [])
        if not candidates:
            continue
        ranked = ranked_children_for_lambda(candidates, lambda_value)
        if not ranked:
            continue
        winner = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        non_measured = [row for row in ranked if not bool(row.get("same_as_measured_for_seed"))]
        seed0_basin = [row for row in ranked if bool(row.get("spatial_seed0_sc_basin"))]
        scored_all = [scored_copy(row, lambda_value) for row in candidates]
        categories: list[tuple[str, dict[str, Any] | None]] = [
            ("selected_measured_winner", winner),
            ("runner_up_by_final_value", runner),
            ("best_non_measured_branch", non_measured[0] if non_measured else None),
            ("best_seed0_sc_basin_branch", seed0_basin[0] if seed0_basin else None),
            (
                "best_by_normalized_sc",
                max(scored_all, key=lambda row: as_float(row.get("normalized_sc"), float("-inf"))),
            ),
            (
                "best_by_raw_source_occ_free",
                max(scored_all, key=lambda row: as_float(row.get("accumulated_source_occ_free"), float("-inf"))),
            ),
            (
                "best_by_sc_bonus",
                max(scored_all, key=lambda row: as_float(row.get("lambda_sc_bonus"), float("-inf"))),
            ),
            (
                "best_by_base_exp_value",
                max(scored_all, key=lambda row: as_float(row.get("base_exp_value"), float("-inf"))),
            ),
        ]
        seen: set[tuple[str, int]] = set()
        for category, row in categories:
            if row is None:
                continue
            key = (category, int(row["candidate_index"]))
            if key in seen:
                continue
            seen.add(key)
            final_value = as_float(row.get("final_value"), as_float(row.get("value"), float("nan")))
            output.append(
                {
                    "seed": seed,
                    "formula": decision.get("formula"),
                    "sc_basis": sc_basis,
                    "lambda_family": decision.get("lambda_family"),
                    "lambda_label": decision.get("lambda_label"),
                    "lambda_value": lambda_value,
                    "near_miss_type": category,
                    "candidate_index": row.get("candidate_index"),
                    "node_id": row.get("node_id"),
                    "selected_child_id": row.get("selected_child_id"),
                    "best_descendant_id": row.get("best_descendant_id"),
                    "selected_child_grid": row.get("selected_child_grid"),
                    "selected_child_world": row.get("selected_child_world"),
                    "best_descendant_grid": row.get("best_descendant_grid"),
                    "best_descendant_world": row.get("best_descendant_world"),
                    "branch_depth": row.get("branch_depth"),
                    "final_value": final_value,
                    "selected_winner_value": as_float(winner.get("final_value"), float("nan")),
                    "margin_to_winner": as_float(winner.get("final_value"), float("nan")) - final_value,
                    "base_exp_value": row.get("base_exp_value"),
                    "source_occ_free": row.get("accumulated_source_occ_free"),
                    "raw_basis_sc": row.get("accumulated_formula_effective_sc"),
                    "normalized_sc": row.get("normalized_sc"),
                    "log_normalized_sc": row.get("log_normalized_sc"),
                    "sc_bonus": row.get("lambda_sc_bonus"),
                    "cost": row.get("accumulated_cost"),
                    "gain_exp": row.get("accumulated_gain_exp"),
                    "same_as_measured_for_seed": row.get("same_as_measured_for_seed"),
                    "spatial_seed0_sc_basin": row.get("spatial_seed0_sc_basin"),
                    "exact_seed0_sc": row.get("exact_seed0_sc"),
                    "primary_classification": row.get("primary_classification"),
                    "low_cost_artifact": bool(
                        as_float(row.get("accumulated_cost"), float("inf")) <= SEED0_SC_COST_REFERENCE + 1.0e-9
                        and as_float(row.get("accumulated_gain_exp"), 0.0) <= SEED0_SC_GAIN_EXP_REFERENCE
                    ),
                }
            )
    return output


def build_required_lambda_rows(
    candidate_by_key: dict[tuple[int, str], list[dict[str, Any]]],
    adaptive_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    adaptive_2x_by_key = {
        (int(row["seed"]), str(row["sc_basis"])): as_float(row.get("lambda_value"), 0.0)
        for row in adaptive_rows
        if str(row.get("adaptive_scale_label")) == "2x"
    }
    required_rows: list[dict[str, Any]] = []
    impossible_rows: list[dict[str, Any]] = []
    for (seed, sc_basis), rows in sorted(candidate_by_key.items()):
        measured = measured_candidate(rows)
        measured_base = as_float(measured.get("base_exp_value"), float("nan"))
        measured_norm = as_float(measured.get("normalized_sc"), float("nan"))
        measured_log = as_float(measured.get("log_normalized_sc"), float("nan"))
        ranked_base = ranked_children_for_lambda(rows, 0.0)
        runner = ranked_base[1] if len(ranked_base) > 1 else measured
        runner_base = as_float(runner.get("base_exp_value"), float("nan"))
        runner_norm = as_float(runner.get("normalized_sc"), float("nan"))
        adaptive_2x = adaptive_2x_by_key.get((seed, sc_basis))
        for row in rows:
            if bool(row.get("same_as_measured_for_seed")):
                continue
            branch_base = as_float(row.get("base_exp_value"), float("nan"))
            branch_norm = as_float(row.get("normalized_sc"), float("nan"))
            branch_log = as_float(row.get("log_normalized_sc"), float("nan"))
            req, impossible, already = required_lambda(branch_base, branch_norm, measured_base, measured_norm)
            req_runner, impossible_runner, already_runner = required_lambda(
                branch_base,
                branch_norm,
                runner_base,
                runner_norm,
            )
            req_log, impossible_log, already_log = required_lambda(branch_base, branch_log, measured_base, measured_log)
            out = {
                "seed": seed,
                "sc_basis": sc_basis,
                "candidate_index": row.get("candidate_index"),
                "node_id": row.get("node_id"),
                "selected_child_id": row.get("selected_child_id"),
                "best_descendant_id": row.get("best_descendant_id"),
                "selected_child_grid": row.get("selected_child_grid"),
                "best_descendant_grid": row.get("best_descendant_grid"),
                "branch_depth": row.get("branch_depth"),
                "branch_base_exp_value": branch_base,
                "branch_normalized_sc": branch_norm,
                "branch_log_normalized_sc": branch_log,
                "branch_source_occ_free": row.get("accumulated_source_occ_free"),
                "branch_raw_basis_sc": row.get("accumulated_formula_effective_sc"),
                "branch_gain_exp": row.get("accumulated_gain_exp"),
                "branch_cost": row.get("accumulated_cost"),
                "measured_candidate_index": measured.get("candidate_index"),
                "measured_selected_child_id": measured.get("selected_child_id"),
                "measured_best_descendant_id": measured.get("best_descendant_id"),
                "measured_base_exp_value": measured_base,
                "measured_normalized_sc": measured_norm,
                "measured_log_normalized_sc": measured_log,
                "runner_up_base_exp_value": runner_base,
                "runner_up_normalized_sc": runner_norm,
                "required_lambda_to_beat_measured": req,
                "required_lambda_to_beat_runner_up": req_runner,
                "required_lambda_log_to_beat_measured": req_log,
                "impossible_under_positive_lambda": impossible,
                "impossible_under_positive_lambda_log": impossible_log,
                "already_beats_measured_at_lambda0": already,
                "already_beats_runner_up_at_lambda0": already_runner,
                "impossible_to_beat_runner_up_under_positive_lambda": impossible_runner,
                "adaptive_2x_lambda": adaptive_2x,
                "required_lambda_le_32": False if req is None else bool(req <= 32.0),
                "required_lambda_le_adaptive_2x": False
                if req is None or adaptive_2x is None
                else bool(req <= adaptive_2x),
                "is_reference_seed0_sc_branch": row.get("is_reference_seed0_sc_branch"),
                "spatial_seed0_sc_basin": row.get("spatial_seed0_sc_basin"),
                "primary_classification": row.get("primary_classification"),
            }
            required_rows.append(out)
            if impossible:
                impossible_rows.append(out)
    return required_rows, impossible_rows


def build_normalization_diagnostics(
    candidate_by_key: dict[tuple[int, str], list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_out: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    for (seed, sc_basis), rows in sorted(candidate_by_key.items()):
        measured = measured_candidate(rows)
        non_measured = [row for row in rows if not bool(row.get("same_as_measured_for_seed"))]
        best_non = max(
            non_measured,
            key=lambda row: (
                as_float(row.get("normalized_sc"), float("-inf")),
                as_float(row.get("base_exp_value"), float("-inf")),
            ),
            default=None,
        )
        seed0_sc = next((row for row in rows if bool(row.get("is_reference_seed0_sc_branch"))), None)
        raw_values = [as_float(row.get("accumulated_formula_effective_sc"), float("nan")) for row in rows]
        source_values = [as_float(row.get("accumulated_source_occ_free"), float("nan")) for row in rows]
        norm_values = [as_float(row.get("normalized_sc"), float("nan")) for row in rows]
        log_values = [as_float(row.get("log_normalized_sc"), float("nan")) for row in rows]
        norm_summary = percentile_summary(norm_values)
        norm_iqr = as_float(norm_summary.get("q75"), 0.0) - as_float(norm_summary.get("q25"), 0.0)
        row = {
            "seed": seed,
            "sc_basis": sc_basis,
            "candidate_count": len(rows),
            "source_occ_free_min": percentile_summary(source_values).get("min"),
            "source_occ_free_p25": percentile_summary(source_values).get("q25"),
            "source_occ_free_p50": percentile_summary(source_values).get("median"),
            "source_occ_free_p75": percentile_summary(source_values).get("q75"),
            "source_occ_free_p90": float(np.percentile(np.asarray(source_values, dtype=np.float64), 90)),
            "source_occ_free_max": percentile_summary(source_values).get("max"),
            "raw_basis_sc_min": percentile_summary(raw_values).get("min"),
            "raw_basis_sc_p25": percentile_summary(raw_values).get("q25"),
            "raw_basis_sc_p50": percentile_summary(raw_values).get("median"),
            "raw_basis_sc_p75": percentile_summary(raw_values).get("q75"),
            "raw_basis_sc_p90": float(np.percentile(np.asarray(raw_values, dtype=np.float64), 90)),
            "raw_basis_sc_max": percentile_summary(raw_values).get("max"),
            "normalized_sc_min": norm_summary.get("min"),
            "normalized_sc_p25": norm_summary.get("q25"),
            "normalized_sc_p50": norm_summary.get("median"),
            "normalized_sc_p75": norm_summary.get("q75"),
            "normalized_sc_p90": float(np.percentile(np.asarray(norm_values, dtype=np.float64), 90)),
            "normalized_sc_max": norm_summary.get("max"),
            "log_normalized_sc_min": percentile_summary(log_values).get("min"),
            "log_normalized_sc_p25": percentile_summary(log_values).get("q25"),
            "log_normalized_sc_p50": percentile_summary(log_values).get("median"),
            "log_normalized_sc_p75": percentile_summary(log_values).get("q75"),
            "log_normalized_sc_p90": float(np.percentile(np.asarray(log_values, dtype=np.float64), 90)),
            "log_normalized_sc_max": percentile_summary(log_values).get("max"),
            "measured_candidate_index": measured.get("candidate_index"),
            "measured_selected_child_id": measured.get("selected_child_id"),
            "measured_best_descendant_id": measured.get("best_descendant_id"),
            "measured_base_exp_value": measured.get("base_exp_value"),
            "measured_source_occ_free": measured.get("accumulated_source_occ_free"),
            "measured_normalized_sc": measured.get("normalized_sc"),
            "measured_log_normalized_sc": measured.get("log_normalized_sc"),
            "measured_normalized_sc_rank": rank_desc(rows, "normalized_sc", int(measured["candidate_index"])),
            "measured_source_occ_free_rank": rank_desc(rows, "accumulated_source_occ_free", int(measured["candidate_index"])),
            "seed0_sc_branch_present": bool(seed0_sc is not None),
            "seed0_sc_normalized_sc": None if seed0_sc is None else seed0_sc.get("normalized_sc"),
            "seed0_sc_source_occ_free": None if seed0_sc is None else seed0_sc.get("accumulated_source_occ_free"),
            "seed0_sc_normalized_sc_rank": None
            if seed0_sc is None
            else rank_desc(rows, "normalized_sc", int(seed0_sc["candidate_index"])),
            "best_nonmeasured_candidate_index": None if best_non is None else best_non.get("candidate_index"),
            "best_nonmeasured_selected_child_id": None if best_non is None else best_non.get("selected_child_id"),
            "best_nonmeasured_best_descendant_id": None if best_non is None else best_non.get("best_descendant_id"),
            "best_nonmeasured_base_exp_value": None if best_non is None else best_non.get("base_exp_value"),
            "best_nonmeasured_source_occ_free": None if best_non is None else best_non.get("accumulated_source_occ_free"),
            "best_nonmeasured_normalized_sc": None if best_non is None else best_non.get("normalized_sc"),
            "best_nonmeasured_log_normalized_sc": None if best_non is None else best_non.get("log_normalized_sc"),
            "best_nonmeasured_normalized_sc_rank": None
            if best_non is None
            else rank_desc(rows, "normalized_sc", int(best_non["candidate_index"])),
            "best_nonmeasured_source_occ_free_rank": None
            if best_non is None
            else rank_desc(rows, "accumulated_source_occ_free", int(best_non["candidate_index"])),
            "measured_winner_already_high_sc": bool(
                rank_desc(rows, "normalized_sc", int(measured["candidate_index"])) is not None
                and rank_desc(rows, "normalized_sc", int(measured["candidate_index"])) <= max(1, math.ceil(0.25 * len(rows)))
            ),
            "normalization_compressed_near_zero": bool(as_float(norm_summary.get("q90"), 0.0) < 0.2)
            if "q90" in norm_summary
            else bool(float(np.percentile(np.asarray(norm_values, dtype=np.float64), 90)) < 0.2),
            "normalization_compressed_near_one": bool(as_float(norm_summary.get("q25"), 0.0) > 0.8),
            "normalization_iqr": norm_iqr,
            "normalization_flat_iqr_lt_0p10": bool(norm_iqr < 0.10),
        }
        max_norm = max(rows, key=lambda item: as_float(item.get("normalized_sc"), float("-inf")))
        row["max_sc_belongs_to_measured_branch"] = bool(max_norm.get("same_as_measured_for_seed"))
        row["max_sc_candidate_index"] = max_norm.get("candidate_index")
        row["max_sc_selected_child_id"] = max_norm.get("selected_child_id")
        rows_out.append(row)
        rank_rows.append(
            {
                "seed": seed,
                "sc_basis": sc_basis,
                "measured_normalized_sc_rank": row["measured_normalized_sc_rank"],
                "measured_source_occ_free_rank": row["measured_source_occ_free_rank"],
                "measured_normalized_sc": row["measured_normalized_sc"],
                "best_nonmeasured_normalized_sc_rank": row["best_nonmeasured_normalized_sc_rank"],
                "best_nonmeasured_source_occ_free_rank": row["best_nonmeasured_source_occ_free_rank"],
                "best_nonmeasured_normalized_sc": row["best_nonmeasured_normalized_sc"],
                "normalized_sc_delta_best_nonmeasured_minus_measured": None
                if best_non is None
                else as_float(best_non.get("normalized_sc"), 0.0) - as_float(measured.get("normalized_sc"), 0.0),
                "source_occ_free_delta_best_nonmeasured_minus_measured": None
                if best_non is None
                else as_float(best_non.get("accumulated_source_occ_free"), 0.0)
                - as_float(measured.get("accumulated_source_occ_free"), 0.0),
                "max_sc_belongs_to_measured_branch": row["max_sc_belongs_to_measured_branch"],
                "measured_winner_already_high_sc": row["measured_winner_already_high_sc"],
            }
        )
    return rows_out, rank_rows


def build_adaptive_gap_rows(
    adaptive_rows: list[dict[str, Any]],
    required_rows: list[dict[str, Any]],
    candidate_by_key: dict[tuple[int, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    adaptive_by_key: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    base_by_key: dict[tuple[int, str], float] = {}
    for row in adaptive_rows:
        key = (int(row["seed"]), str(row["sc_basis"]))
        adaptive_by_key[key][str(row.get("adaptive_scale_label"))] = as_float(row.get("lambda_value"), 0.0)
        base_by_key[key] = as_float(row.get("lambda_base"), 0.0)

    req_by_key: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in required_rows:
        req_by_key[(int(row["seed"]), str(row["sc_basis"]))].append(row)

    output: list[dict[str, Any]] = []
    for key, candidates in sorted(candidate_by_key.items()):
        seed, sc_basis = key
        reqs = req_by_key.get(key, [])
        finite_positive = [
            as_float(row.get("required_lambda_to_beat_measured"), float("nan"))
            for row in reqs
            if row.get("required_lambda_to_beat_measured") is not None
            and as_float(row.get("required_lambda_to_beat_measured"), float("nan")) > 1.0e-9
        ]
        finite_any = [
            as_float(row.get("required_lambda_to_beat_measured"), float("nan"))
            for row in reqs
            if row.get("required_lambda_to_beat_measured") is not None
        ]
        impossible = sum(bool(row.get("impossible_under_positive_lambda")) for row in reqs)
        adaptive = adaptive_by_key.get(key, {})
        nearest = min(finite_positive, default=min(finite_any, default=None))
        adaptive_2x = adaptive.get("2x")
        output.append(
            {
                "seed": seed,
                "sc_basis": sc_basis,
                "candidate_count": len(candidates),
                "nonmeasured_candidate_count": len(reqs),
                "p50_base_exp_value": float(
                    np.percentile([as_float(row.get("base_exp_value"), 0.0) for row in candidates], 50)
                ),
                "p90_base_exp_value": float(
                    np.percentile([as_float(row.get("base_exp_value"), 0.0) for row in candidates], 90)
                ),
                "lambda_base": base_by_key.get(key),
                "adaptive_0p25x": adaptive.get("0p25x"),
                "adaptive_0p5x": adaptive.get("0p5x"),
                "adaptive_1x": adaptive.get("1x"),
                "adaptive_2x": adaptive_2x,
                "nearest_required_lambda_to_flip_nonmeasured": nearest,
                "adaptive_2x_minus_nearest_required": None
                if nearest is None or adaptive_2x is None
                else adaptive_2x - nearest,
                "adaptive_2x_theoretically_too_small": bool(
                    nearest is not None and adaptive_2x is not None and adaptive_2x < nearest
                ),
                "impossible_under_positive_lambda_count": impossible,
                "impossible_under_positive_lambda_fraction": impossible / max(1, len(reqs)),
                "finite_required_lambda_count": len(finite_any),
                "finite_required_lambda_le_32_count": sum(v <= 32.0 for v in finite_any),
                "finite_required_lambda_le_adaptive_2x_count": 0
                if adaptive_2x is None
                else sum(v <= adaptive_2x for v in finite_any),
            }
        )
    return output


def build_low_cost_rows(required_rows: list[dict[str, Any]], near_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in required_rows:
        is_low = (
            as_float(row.get("branch_cost"), float("inf")) <= SEED0_SC_COST_REFERENCE + 1.0e-9
            and as_float(row.get("branch_gain_exp"), 0.0) <= SEED0_SC_GAIN_EXP_REFERENCE
        )
        if is_low or bool(row.get("is_reference_seed0_sc_branch")) or bool(row.get("spatial_seed0_sc_basin")):
            out = dict(row)
            out["low_cost_artifact"] = is_low
            out["cost_vs_seed0_sc_reference"] = as_float(row.get("branch_cost"), 0.0) - SEED0_SC_COST_REFERENCE
            out["gain_exp_vs_seed0_sc_reference"] = as_float(row.get("branch_gain_exp"), 0.0) - SEED0_SC_GAIN_EXP_REFERENCE
            selected.append(out)
    if selected:
        return selected
    fallback = [row for row in near_rows if row.get("near_miss_type") == "best_non_measured_branch"]
    return [
        {
            "seed": row.get("seed"),
            "sc_basis": row.get("sc_basis"),
            "selected_child_id": row.get("selected_child_id"),
            "best_descendant_id": row.get("best_descendant_id"),
            "branch_cost": row.get("cost"),
            "branch_gain_exp": row.get("gain_exp"),
            "branch_base_exp_value": row.get("base_exp_value"),
            "branch_normalized_sc": row.get("normalized_sc"),
            "low_cost_artifact": bool(row.get("low_cost_artifact")),
            "source": "fallback_best_non_measured_near_miss",
        }
        for row in fallback[:60]
    ]


def plot_required_lambda_distribution(output_dir: Path, required_rows: list[dict[str, Any]]) -> None:
    values = [
        as_float(row.get("required_lambda_to_beat_measured"), float("nan"))
        for row in required_rows
        if row.get("required_lambda_to_beat_measured") is not None
    ]
    values = [v for v in values if math.isfinite(v) and v <= 200.0]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.hist(values, bins=30, color="#3f7f93", edgecolor="white")
    ax.axvline(32.0, color="#9f1239", linestyle="--", label="lambda 32")
    ax.set_xlabel("required lambda to beat measured")
    ax.set_ylabel("candidate paths")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "required_lambda_distribution.png", dpi=160)
    plt.close(fig)


def plot_required_lambda_by_seed(output_dir: Path, required_rows: list[dict[str, Any]]) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in required_rows:
        value = row.get("required_lambda_to_beat_measured")
        if value is not None and math.isfinite(float(value)) and float(value) <= 200.0:
            grouped[int(row["seed"])].append(float(value))
    seeds = sorted(grouped)
    data = [grouped[seed] for seed in seeds]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.boxplot(data, labels=[str(seed) for seed in seeds], showfliers=False)
    ax.axhline(32.0, color="#9f1239", linestyle="--", label="lambda 32")
    ax.set_xlabel("seed")
    ax.set_ylabel("required lambda")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "required_lambda_by_seed.png", dpi=160)
    plt.close(fig)


def plot_adaptive_vs_required(output_dir: Path, adaptive_gap_rows: list[dict[str, Any]]) -> None:
    xs = [as_float(row.get("adaptive_2x"), float("nan")) for row in adaptive_gap_rows]
    ys = [as_float(row.get("nearest_required_lambda_to_flip_nonmeasured"), float("nan")) for row in adaptive_gap_rows]
    colors = [int(row["seed"]) for row in adaptive_gap_rows]
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(xs, ys, c=colors, cmap="viridis", s=55, edgecolor="white")
    lim = max([v for v in xs + ys if math.isfinite(v)] + [32.0])
    ax.plot([0, lim], [0, lim], color="#374151", linestyle="--", linewidth=1)
    ax.set_xlabel("adaptive 2x lambda")
    ax.set_ylabel("nearest required lambda")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / "adaptive_lambda_vs_required_lambda.png", dpi=160)
    plt.close(fig)


def plot_measured_vs_nonmeasured(output_dir: Path, norm_rows: list[dict[str, Any]], key: str, filename: str, label: str) -> None:
    xs = [as_float(row.get(f"measured_{key}"), float("nan")) for row in norm_rows]
    ys = [as_float(row.get(f"best_nonmeasured_{key}"), float("nan")) for row in norm_rows]
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(xs, ys, s=55, color="#3f7f93", edgecolor="white")
    lim_values = [v for v in xs + ys if math.isfinite(v)]
    if lim_values:
        lo, hi = min(lim_values), max(lim_values)
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#374151", linewidth=1)
    ax.set_xlabel(f"measured {label}")
    ax.set_ylabel(f"best non-measured {label}")
    ax.grid(alpha=0.25)
    fig.savefig(output_dir / filename, dpi=160)
    plt.close(fig)


def plot_final_value_gap(output_dir: Path, near_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in near_rows if row.get("near_miss_type") == "best_non_measured_branch"]
    labels = [f"{row['seed']}\n{str(row['sc_basis']).split('_')[0]}" for row in rows]
    values = [as_float(row.get("margin_to_winner"), 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(range(len(rows)), values, color="#537a5a")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("winner minus best non-measured value")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "measured_vs_best_nonmeasured_final_value_gap.png", dpi=160)
    plt.close(fig)


def plot_normalized_distribution(output_dir: Path, candidate_by_key: dict[tuple[int, str], list[dict[str, Any]]]) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for (seed, _basis), rows in candidate_by_key.items():
        grouped[int(seed)].extend(as_float(row.get("normalized_sc"), 0.0) for row in rows)
    seeds = sorted(grouped)
    data = [grouped[seed] for seed in seeds]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax.boxplot(data, labels=[str(seed) for seed in seeds], showfliers=False)
    ax.set_xlabel("seed")
    ax.set_ylabel("normalized_sc")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "normalized_sc_distribution_by_seed.png", dpi=160)
    plt.close(fig)


def plot_sc_rank(output_dir: Path, rank_rows: list[dict[str, Any]]) -> None:
    labels = [f"{row['seed']}\n{str(row['sc_basis']).split('_')[0]}" for row in rank_rows]
    values = [as_float(row.get("measured_normalized_sc_rank"), float("nan")) for row in rank_rows]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(range(len(values)), values, color="#6b7280")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("measured normalized_sc rank, 1 = highest")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_dir / "sc_rank_of_measured_winner.png", dpi=160)
    plt.close(fig)


def plot_near_miss_topdown(output_dir: Path, observed_state: np.ndarray, near_rows: list[dict[str, Any]]) -> None:
    rows = [
        row
        for row in near_rows
        if row.get("near_miss_type") in {"selected_measured_winner", "best_non_measured_branch"}
        and str(row.get("lambda_family")) == "fixed"
        and str(row.get("lambda_label")) in {"0", "32"}
    ]
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    plot_base_map(ax, observed_state)
    for row in rows:
        grid = row.get("selected_child_grid")
        if not grid:
            continue
        color = "#1f77b4" if row.get("near_miss_type") == "selected_measured_winner" else "#d62728"
        marker = "o" if str(row.get("lambda_label")) == "0" else "^"
        ax.scatter([grid[0]], [grid[1]], s=55, color=color, marker=marker, alpha=0.85)
    for grid, label, marker, color in [
        (REFERENCE_MEASURED_SELECTED_GRID, "measured ref", "s", "#111827"),
        (REFERENCE_SC_SELECTED_GRID, "seed0 SC ref", "X", "#dc2626"),
    ]:
        ax.scatter([grid[0]], [grid[1]], s=110, color=color, marker=marker, label=label)
    ax.set_title("Near-miss selected children, fixed lambda 0/32")
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "near_miss_topdown.png", dpi=160)
    plt.close(fig)


def plot_value_component_stack(output_dir: Path, near_rows: list[dict[str, Any]]) -> None:
    rows = [
        row
        for row in near_rows
        if row.get("near_miss_type") in {"selected_measured_winner", "best_non_measured_branch"}
        and str(row.get("lambda_family")) == "fixed"
        and str(row.get("lambda_label")) == "32"
    ][:24]
    labels = [f"{row['seed']}-{row['near_miss_type'].replace('_branch', '').replace('_', '-')}" for row in rows]
    base = [as_float(row.get("base_exp_value"), 0.0) for row in rows]
    bonus = [as_float(row.get("sc_bonus"), 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar(range(len(rows)), base, label="base_exp_value", color="#426b69")
    ax.bar(range(len(rows)), bonus, bottom=base, label="lambda * normalized_sc", color="#d99a4e")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("value components at lambda 32")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output_dir / "value_component_near_miss_stack.png", dpi=160)
    plt.close(fig)


def make_plots(
    output_dir: Path,
    observed_state: np.ndarray,
    required_rows: list[dict[str, Any]],
    adaptive_gap_rows: list[dict[str, Any]],
    norm_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
    near_rows: list[dict[str, Any]],
    candidate_by_key: dict[tuple[int, str], list[dict[str, Any]]],
) -> dict[str, str]:
    plot_required_lambda_distribution(output_dir, required_rows)
    plot_required_lambda_by_seed(output_dir, required_rows)
    plot_adaptive_vs_required(output_dir, adaptive_gap_rows)
    plot_measured_vs_nonmeasured(
        output_dir,
        norm_rows,
        "base_exp_value",
        "measured_vs_best_nonmeasured_base_exp.png",
        "base_exp_value",
    )
    plot_measured_vs_nonmeasured(
        output_dir,
        norm_rows,
        "normalized_sc",
        "measured_vs_best_nonmeasured_normalized_sc.png",
        "normalized_sc",
    )
    plot_final_value_gap(output_dir, near_rows)
    plot_normalized_distribution(output_dir, candidate_by_key)
    plot_sc_rank(output_dir, rank_rows)
    plot_near_miss_topdown(output_dir, observed_state, near_rows)
    plot_value_component_stack(output_dir, near_rows)
    return {name[:-4]: str(output_dir / name) for name in REQUIRED_PLOTS}


def write_near_summary(path: Path, near_rows: list[dict[str, Any]]) -> None:
    best_non = [row for row in near_rows if row.get("near_miss_type") == "best_non_measured_branch"]
    margins = [as_float(row.get("margin_to_winner"), float("nan")) for row in best_non]
    lines = [
        "# Near-Miss Branch Summary",
        "",
        f"- near-miss rows: `{len(near_rows)}`",
        f"- best non-measured rows: `{len(best_non)}`",
        f"- best non-measured margin summary: `{finite_percentiles(margins)}`",
        "",
        "The table records selected winners, runner-ups, best non-measured branches, seed0-SC-basin branches, and SC/base component leaders for each 6.5z decision formula.",
    ]
    write_text(path, "\n".join(lines))


def write_required_summary(path: Path, rows: list[dict[str, Any]], impossible_rows: list[dict[str, Any]]) -> None:
    finite = [
        as_float(row.get("required_lambda_to_beat_measured"), float("nan"))
        for row in rows
        if row.get("required_lambda_to_beat_measured") is not None
    ]
    lines = [
        "# Required Lambda To Flip Summary",
        "",
        f"- non-measured candidate rows: `{len(rows)}`",
        f"- impossible under positive lambda: `{len(impossible_rows)}`",
        f"- finite required lambda distribution: `{finite_percentiles(finite)}`",
        f"- finite required lambda <= 32: `{sum(v <= 32.0 for v in finite)}`",
        f"- finite required lambda <= adaptive 2x: `{sum(bool(row.get('required_lambda_le_adaptive_2x')) for row in rows)}`",
    ]
    seed0 = [row for row in rows if bool(row.get("is_reference_seed0_sc_branch"))]
    if seed0:
        lines.append(f"- seed0 reference SC branch required-lambda rows: `{seed0[:3]}`")
    write_text(path, "\n".join(lines))


def write_normalization_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    high = sum(bool(row.get("measured_winner_already_high_sc")) for row in rows)
    max_measured = sum(bool(row.get("max_sc_belongs_to_measured_branch")) for row in rows)
    flat = sum(bool(row.get("normalization_flat_iqr_lt_0p10")) for row in rows)
    lines = [
        "# Normalization Diagnostics Summary",
        "",
        f"- seed/basis rows: `{len(rows)}`",
        f"- measured winner already in top SC quartile: `{high}`",
        f"- max normalized SC belongs to measured branch: `{max_measured}`",
        f"- flat normalized-sc IQR < 0.10: `{flat}`",
    ]
    write_text(path, "\n".join(lines))


def write_rank_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Measured vs Non-Measured SC Rank",
        "",
        "| seed | basis | measured rank | best non-measured rank | norm delta | max SC measured? |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | `{row['sc_basis']}` | {row.get('measured_normalized_sc_rank')} | "
            f"{row.get('best_nonmeasured_normalized_sc_rank')} | "
            f"{row.get('normalized_sc_delta_best_nonmeasured_minus_measured')} | "
            f"`{row.get('max_sc_belongs_to_measured_branch')}` |"
        )
    write_text(path, "\n".join(lines))


def write_adaptive_md(path: Path, rows: list[dict[str, Any]]) -> None:
    too_small = sum(bool(row.get("adaptive_2x_theoretically_too_small")) for row in rows)
    impossible_dominant = sum(as_float(row.get("impossible_under_positive_lambda_fraction"), 0.0) >= 0.5 for row in rows)
    lines = [
        "# Adaptive Lambda Gap Analysis",
        "",
        f"- seed/basis rows: `{len(rows)}`",
        f"- adaptive 2x theoretically too small: `{too_small}`",
        f"- rows where >=50% non-measured paths are impossible under positive lambda: `{impossible_dominant}`",
        "",
        "| seed | basis | lambda_base | adaptive 2x | nearest required | 2x - required | impossible frac |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | `{row['sc_basis']}` | {row.get('lambda_base')} | "
            f"{row.get('adaptive_2x')} | {row.get('nearest_required_lambda_to_flip_nonmeasured')} | "
            f"{row.get('adaptive_2x_minus_nearest_required')} | "
            f"{row.get('impossible_under_positive_lambda_fraction')} |"
        )
    write_text(path, "\n".join(lines))


def write_low_cost_md(path: Path, rows: list[dict[str, Any]]) -> None:
    low = sum(bool(row.get("low_cost_artifact")) for row in rows)
    seed0_ref = sum(bool(row.get("is_reference_seed0_sc_branch")) for row in rows)
    lines = [
        "# Low-Cost Artifact Follow-Up",
        "",
        f"- rows: `{len(rows)}`",
        f"- low-cost artifact rows: `{low}`",
        f"- seed0 reference SC branch rows: `{seed0_ref}`",
        "",
        "Rows here are diagnostic only. They identify branches that are seed0-SC-like or cheaper/lower-gain than the seed0 short branch reference.",
    ]
    write_text(path, "\n".join(lines))


def write_regen_md(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Debug Tree Regeneration Report",
        "",
        f"- recomputed debug tree nodes: `{report['recomputed_debug_tree_nodes']}`",
        f"- reason: {report['reason']}",
        f"- candidate path rows: `{report['candidate_path_rows']}`",
        f"- visible cache entries: `{report['visible_cache_entries']}`",
        f"- elapsed seconds: `{report['elapsed_s']}`",
    ]
    write_text(path, "\n".join(lines))


def choose_recommendation(
    required_rows: list[dict[str, Any]],
    adaptive_gap_rows: list[dict[str, Any]],
    low_cost_rows: list[dict[str, Any]],
    rank_rows: list[dict[str, Any]],
) -> tuple[str, str]:
    finite = [
        as_float(row.get("required_lambda_to_beat_measured"), float("nan"))
        for row in required_rows
        if row.get("required_lambda_to_beat_measured") is not None
    ]
    finite_positive = [v for v in finite if v > 1.0e-9]
    le32 = sum(v <= 32.0 for v in finite_positive)
    impossible_fraction = (
        sum(bool(row.get("impossible_under_positive_lambda")) for row in required_rows) / max(1, len(required_rows))
    )
    low_cost_fraction = sum(bool(row.get("low_cost_artifact")) for row in low_cost_rows) / max(1, len(low_cost_rows))
    measured_max_fraction = sum(bool(row.get("max_sc_belongs_to_measured_branch")) for row in rank_rows) / max(
        1, len(rank_rows)
    )
    if low_cost_fraction > 0.5 and le32 == 0:
        return (
            "Pareto dominance gate / branch-level utility review",
            "non-measured wins are dominated by low-gain/low-cost artifacts rather than useful SC signal",
        )
    if finite_positive and le32 / max(1, len(finite_positive)) >= 0.20:
        return (
            "offline lambda refinement around the required-lambda range",
            "many non-measured candidates have finite required lambda at or below 32, so the next evidence-preserving step is still offline lambda refinement",
        )
    if finite_positive:
        return (
            "larger offline lambda diagnostic sweep",
            "non-measured candidates can theoretically flip, but most require lambda values above 32",
        )
    if impossible_fraction >= 0.50 or measured_max_fraction >= 0.50:
        return (
            "controlled synthetic SC validation scene",
            "most non-measured candidates cannot be helped by positive lambda or measured branches already own the strongest normalized SC",
        )
    return (
        "controlled synthetic SC validation scene",
        "decoupled scoring is conservative on this saved frame and does not provide a clean branch-selective SC signal",
    )


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    answers = summary["answers"]
    lines = [
        "# Stage 4A-6.5z.1 Decoupled Signal-Strength Summary",
        "",
        f"- seeds: `{summary['seeds']}`",
        f"- SC bases: `{summary['sc_bases']}`",
        f"- regenerated candidate path rows: `{summary['debug_tree_regeneration']['candidate_path_rows']}`",
        f"- required-lambda rows: `{summary['required_lambda_row_count']}`",
        f"- original lambda-sweep summary all measured-only: `{summary['stage4a65z_input_validation']['lambda_sweep_summary_claims_all_measured_only']}`",
        f"- corrected decision/classification rows all measured-only: `{summary['stage4a65z_input_validation']['decision_rows_claim_all_measured_only']}`",
        "",
        "## Answers",
        "",
        f"- lambda too small: {answers['lambda_too_small']}",
        f"- measured winner high SC: {answers['measured_winner_already_high_sc']}",
        f"- non-measured lower/equal normalized SC often blocks positive lambda: {answers['nonmeasured_lower_or_equal_sc_blocks_positive_lambda']}",
        f"- source OCC+FREE aligned with measured exploration: {answers['source_occ_free_aligned_with_measured_exploration']}",
        f"- current Frame2 signal too weak/insufficiently branch-selective: {answers['current_frame2_map_predict_signal_too_weak_or_not_branch_selective']}",
        f"- plausible non-measured branch close enough for more offline lambda work: {answers['plausible_nonmeasured_branch_close_enough_for_offline_lambda_refinement']}",
        "",
        f"Recommended next small task: **{summary['recommended_next_small_task']}**.",
        f"Reason: {summary['recommendation_reason']}.",
    ]
    write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_ints(args.seeds)
    stage4a65z_dir = Path(args.stage4a65z_dir).resolve()
    stage4a65y_dir = Path(args.stage4a65y_dir).resolve()
    observed_path = Path(args.observed_state).resolve()
    prediction_path = Path(args.prediction_npz).resolve()
    pose_path = Path(args.pose_json).resolve()
    camera_path = Path(args.camera_info_json).resolve()
    stage4a65x_dir = Path(args.stage4a65x_dir).resolve()
    stage4a65p_dir = Path(args.stage4a65p_dir).resolve()
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

    refs = custom_reference_worlds(
        observed_state,
        float(args.voxel_size),
        parse_grid(args.reference_sc_selected_grid),
        parse_grid(args.reference_sc_best_grid),
        parse_grid(args.reference_measured_selected_grid),
        parse_grid(args.reference_measured_best_grid),
    )

    manifest, missing_fields = collect_input_manifest(stage4a65z_dir)
    save_json(output_dir / "loaded_6p5z_inputs_manifest.json", manifest)
    write_manifest_md(output_dir / "loaded_6p5z_inputs_manifest.md", manifest)

    decisions = load_json_any(stage4a65z_dir / "decoupled_sc_sweep_decisions.json", [])
    class_rows = load_json_any(stage4a65z_dir / "branch_classification_by_formula_seed.json", [])
    lambda_summary = load_json_any(stage4a65z_dir / "lambda_sweep_summary_by_basis_variant.json", [])
    adaptive_rows = load_json_any(stage4a65z_dir / "adaptive_lambda_values.json", [])
    stage4a65z_summary = load_json_any(stage4a65z_dir / "stage4a65z_decoupled_sc_utility_sweep_summary.json", {})
    seed0_gap = load_json_any(stage4a65z_dir / "seed0_base_gap_report.json", {})
    stage4a65y_decisions = load_json_any(stage4a65y_dir / "per_seed_formula_decisions.json", [])
    stage4a65y_summary = load_json_any(stage4a65y_dir / "stage4a65y_source_gain_seed_replay_summary.json", {})
    stage4a65x_summary = load_json_any(stage4a65x_dir / "stage4a65x_sc_gain_design_review_summary.json", {})
    measured_by_seed = {
        int(row["seed"]): row for row in stage4a65y_decisions if str(row.get("formula")) == "measured_only"
    }
    if set(seeds) - set(measured_by_seed):
        raise ValueError(f"missing measured_only rows for seeds: {sorted(set(seeds) - set(measured_by_seed))}")

    sc_bases = list(stage4a65z_summary.get("sc_bases") or [item for item in DEFAULT_SC_BASES.split(",") if item])
    fixed_lambdas = list(stage4a65z_summary.get("fixed_lambdas") or [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0])
    adaptive_scales = list(stage4a65z_summary.get("adaptive_lambda_scales") or [0.25, 0.5, 1.0, 2.0])

    corrected_class_summary = summarize_formula_rows(class_rows)
    lambda_sweep_summary_claims_all_measured = bool(
        lambda_summary
        and all(as_float(row.get("same_as_measured_fraction"), 0.0) == 1.0 for row in lambda_summary)
        and all(as_float(row.get("spatial_seed0_sc_basin_fraction"), 1.0) == 0.0 for row in lambda_summary)
    )
    decision_rows_claim_all_measured = bool(
        corrected_class_summary
        and all(as_float(row.get("same_as_measured_fraction"), 0.0) == 1.0 for row in corrected_class_summary.values())
        and all(as_float(row.get("spatial_seed0_sc_basin_fraction"), 1.0) == 0.0 for row in corrected_class_summary.values())
    )

    candidate_by_key, _norm_from_regen, regen_report = regenerate_candidate_tables(
        args=args,
        observed_state=observed_state,
        prediction=prediction,
        seeds=seeds,
        sc_bases=sc_bases,
        measured_by_seed=measured_by_seed,
        refs=refs,
    )
    save_json(output_dir / "debug_tree_regeneration_report.json", regen_report)
    write_regen_md(output_dir / "debug_tree_regeneration_report.md", regen_report)

    near_rows = build_near_miss_rows(decisions, candidate_by_key)
    required_rows, impossible_rows = build_required_lambda_rows(candidate_by_key, adaptive_rows)
    norm_rows, rank_rows = build_normalization_diagnostics(candidate_by_key)
    adaptive_gap_rows = build_adaptive_gap_rows(adaptive_rows, required_rows, candidate_by_key)
    low_cost_rows = build_low_cost_rows(required_rows, near_rows)

    write_csv(output_dir / "near_miss_branch_table.csv", near_rows)
    save_json(output_dir / "near_miss_branch_table.json", near_rows)
    write_near_summary(output_dir / "near_miss_branch_summary.md", near_rows)
    write_csv(output_dir / "required_lambda_to_flip.csv", required_rows)
    save_json(output_dir / "required_lambda_to_flip.json", required_rows)
    write_required_summary(output_dir / "required_lambda_to_flip_summary.md", required_rows, impossible_rows)
    write_csv(output_dir / "normalization_diagnostics_by_seed.csv", norm_rows)
    save_json(output_dir / "normalization_diagnostics_by_seed.json", norm_rows)
    write_normalization_summary(output_dir / "normalization_diagnostics_summary.md", norm_rows)
    write_csv(output_dir / "adaptive_lambda_gap_analysis.csv", adaptive_gap_rows)
    save_json(output_dir / "adaptive_lambda_gap_analysis.json", adaptive_gap_rows)
    write_adaptive_md(output_dir / "adaptive_lambda_gap_analysis.md", adaptive_gap_rows)
    write_csv(output_dir / "measured_vs_nonmeasured_sc_rank.csv", rank_rows)
    save_json(output_dir / "measured_vs_nonmeasured_sc_rank.json", rank_rows)
    write_rank_md(output_dir / "measured_vs_nonmeasured_sc_rank.md", rank_rows)
    write_csv(output_dir / "impossible_under_positive_lambda.csv", impossible_rows)
    save_json(output_dir / "impossible_under_positive_lambda.json", impossible_rows)
    write_csv(output_dir / "low_cost_artifact_followup.csv", low_cost_rows)
    save_json(output_dir / "low_cost_artifact_followup.json", low_cost_rows)
    write_low_cost_md(output_dir / "low_cost_artifact_followup.md", low_cost_rows)

    plot_paths = make_plots(
        output_dir,
        observed_state,
        required_rows,
        adaptive_gap_rows,
        norm_rows,
        rank_rows,
        near_rows,
        candidate_by_key,
    )

    finite_req = [
        as_float(row.get("required_lambda_to_beat_measured"), float("nan"))
        for row in required_rows
        if row.get("required_lambda_to_beat_measured") is not None
    ]
    finite_positive_req = [value for value in finite_req if value > 1.0e-9]
    impossible_fraction = len(impossible_rows) / max(1, len(required_rows))
    measured_high_fraction = sum(bool(row.get("measured_winner_already_high_sc")) for row in norm_rows) / max(
        1, len(norm_rows)
    )
    max_sc_measured_fraction = sum(bool(row.get("max_sc_belongs_to_measured_branch")) for row in norm_rows) / max(
        1, len(norm_rows)
    )
    lambda_le32_fraction = sum(value <= 32.0 for value in finite_positive_req) / max(1, len(finite_positive_req))
    adaptive_too_small_count = sum(bool(row.get("adaptive_2x_theoretically_too_small")) for row in adaptive_gap_rows)
    recommendation, recommendation_reason = choose_recommendation(required_rows, adaptive_gap_rows, low_cost_rows, rank_rows)

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

    validation = {
        "seeds_include_expected_0_to_9": sorted(seeds) == list(range(10)),
        "fixed_lambdas": fixed_lambdas,
        "adaptive_scales": adaptive_scales,
        "stage4a65y_seed0_confidence_reproduces_reference": bool(
            any(
                int(row.get("seed")) == 0
                and str(row.get("formula")) == "current_confidence_weighted"
                and same_grid(row.get("selected_child_grid"), REFERENCE_SC_SELECTED_GRID)
                and same_grid(row.get("best_descendant_grid"), REFERENCE_SC_BEST_GRID)
                for row in stage4a65y_decisions
            )
        ),
        "stage4a65y_source_occ_free_seed0_reproduces_reference": bool(
            any(
                int(row.get("seed")) == 0
                and str(row.get("formula")) == "source_occ_free"
                and same_grid(row.get("selected_child_grid"), REFERENCE_SC_SELECTED_GRID)
                and same_grid(row.get("best_descendant_grid"), REFERENCE_SC_BEST_GRID)
                for row in stage4a65y_decisions
            )
        ),
        "seed0_base_gap_measured_minus_sc": seed0_gap.get("base_value_gap_measured_minus_sc"),
        "lambda_sweep_summary_claims_all_measured_only": lambda_sweep_summary_claims_all_measured,
        "decision_rows_claim_all_measured_only": decision_rows_claim_all_measured,
        "corrected_class_summary": corrected_class_summary,
    }
    missing_fields.extend(
        [
            {
                "field": "lambda_sweep_summary_by_basis_variant",
                "reason": "The 6.5z lambda-sweep summary claims all measured-only, while per-formula classification rows do not. This output preserves both and uses corrected row-level diagnostics.",
                "severity": "warning",
            }
        ]
        if lambda_sweep_summary_claims_all_measured and not decision_rows_claim_all_measured
        else []
    )

    answers = {
        "lambda_too_small": bool(finite_positive_req and lambda_le32_fraction >= 0.20),
        "measured_winner_already_high_sc": bool(measured_high_fraction >= 0.50 or max_sc_measured_fraction >= 0.35),
        "nonmeasured_lower_or_equal_sc_blocks_positive_lambda": bool(impossible_fraction >= 0.50),
        "source_occ_free_aligned_with_measured_exploration": bool(max_sc_measured_fraction >= 0.35),
        "current_frame2_map_predict_signal_too_weak_or_not_branch_selective": bool(
            impossible_fraction >= 0.50 or measured_high_fraction >= 0.50
        ),
        "plausible_nonmeasured_branch_close_enough_for_offline_lambda_refinement": bool(lambda_le32_fraction >= 0.20),
        "adaptive_2x_too_small_seed_basis_count": adaptive_too_small_count,
        "runtime_two_frame_readiness": False,
        "rollout_readiness": False,
        "coverage_improvement_claimed": False,
    }

    summary = {
        "stage": "Stage 4A-6.5z.1",
        "output_dir": str(output_dir),
        "diagnostic_status": "offline-only table/saved-tree diagnosis",
        "seeds": seeds,
        "sc_bases": sc_bases,
        "fixed_lambdas": fixed_lambdas,
        "adaptive_lambda_scales": adaptive_scales,
        "inputs": {
            "stage4a65z_dir": str(stage4a65z_dir),
            "stage4a65y_dir": str(stage4a65y_dir),
            "stage4a65x_dir": str(stage4a65x_dir),
            "stage4a65p_dir": str(stage4a65p_dir),
            "observed_state": str(observed_path),
            "prediction_npz": str(prediction_path),
            "pose_json": str(pose_path),
            "camera_info_json": str(camera_path),
        },
        "context_confirmation": {
            "stage4a65x_complete": bool(stage4a65x_summary),
            "stage4a65y_complete": bool(stage4a65y_summary),
            "stage4a65z_complete": bool(stage4a65z_summary),
            "stage4a65y_decision_rows": len(stage4a65y_decisions),
            "stage4a65z_decision_rows": len(decisions),
            "stage4a65z_classification_rows": len(class_rows),
        },
        "stage4a65z_input_validation": validation,
        "debug_tree_regeneration": regen_report,
        "near_miss_row_count": len(near_rows),
        "required_lambda_row_count": len(required_rows),
        "impossible_under_positive_lambda_row_count": len(impossible_rows),
        "finite_required_lambda_distribution": finite_percentiles(finite_req),
        "finite_positive_required_lambda_le_32_fraction": lambda_le32_fraction,
        "impossible_under_positive_lambda_fraction": impossible_fraction,
        "measured_high_sc_fraction": measured_high_fraction,
        "max_sc_belongs_to_measured_fraction": max_sc_measured_fraction,
        "adaptive_2x_too_small_seed_basis_count": adaptive_too_small_count,
        "answers": answers,
        "recommended_next_small_task": recommendation,
        "recommendation_reason": recommendation_reason,
        "not_recommended_next": [
            "runtime two-frame smoke",
            "rollout",
            "online open-ended loop",
            "Pareto dominance gate implementation in this turn",
            "new runtime planner implementation in this turn",
            "RL/PPO/BC/IL",
        ],
        "safety": safety,
        "plots": plot_paths,
        "required_outputs": REQUIRED_OUTPUTS + REQUIRED_PLOTS,
        "elapsed_s": time.perf_counter() - started,
    }
    save_json(output_dir / "missing_fields_report.json", {"count": len(missing_fields), "missing_fields": missing_fields})
    save_json(output_dir / "stage4a65z1_decoupled_signal_strength_summary.json", summary)
    write_summary_md(output_dir / "stage4a65z1_decoupled_signal_strength_summary.md", summary)
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {recommendation}",
                f"- reason: {recommendation_reason}",
                "- still not next: runtime smoke, rollout, online open-ended loop, RL/PPO/BC/IL, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, coverage-improvement claims, or external source build.",
            ]
        ),
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65z_dir", default=DEFAULT_STAGE4A65Z_DIR)
    parser.add_argument("--stage4a65y_dir", default=DEFAULT_STAGE4A65Y_DIR)
    parser.add_argument("--stage4a65x_dir", default=DEFAULT_STAGE4A65X_DIR)
    parser.add_argument("--stage4a65p_dir", default=DEFAULT_STAGE4A65P_DIR)
    parser.add_argument("--observed_state", default=DEFAULT_OBSERVED_STATE)
    parser.add_argument("--prediction_npz", default=DEFAULT_PREDICTION_NPZ)
    parser.add_argument("--pose_json", default=DEFAULT_POSE_JSON)
    parser.add_argument("--camera_info_json", default=DEFAULT_CAMERA_INFO_JSON)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--normalization_mode", choices=["max", "p95", "p90"], default="max")
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--ssc_confidence_threshold", type=float, default=0.05)
    parser.add_argument("--reference_sc_selected_grid", default=",".join(str(v) for v in REFERENCE_SC_SELECTED_GRID))
    parser.add_argument("--reference_sc_best_grid", default=",".join(str(v) for v in REFERENCE_SC_BEST_GRID))
    parser.add_argument(
        "--reference_measured_selected_grid", default=",".join(str(v) for v in REFERENCE_MEASURED_SELECTED_GRID)
    )
    parser.add_argument("--reference_measured_best_grid", default=",".join(str(v) for v in REFERENCE_MEASURED_BEST_GRID))
    parser.add_argument("--save_viz", action="store_true")
    return parser


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
