#!/usr/bin/env python3
"""Run Stage 4A-6.5k offline mini-RRT minimum-edge variants.

This runner is offline-only. It calls the saved-map mini-RRT builder with
source-inspired minimum-length, crop-min-length, and density settings. It does
not start Isaac, run rollout, rerun map_predict, train, modify observed_state,
or modify/build external active_3d_planning source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from offline_mini_rrt_tree import read_jsonl, run as run_mini_rrt, save_json, sha256_file, to_jsonable, write_csv, write_jsonl


EPS = 1.0e-6
ONE_STEP_BASELINE_GRID = [15, 16, 11]
DECOUPLED_GRID = [14, 18, 11]
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)
ROLLOUT_LIKE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]


VARIANTS: list[dict[str, Any]] = [
    {
        "variant_name": "baseline_allow",
        "short_edge_policy": "allow",
        "min_edge_length_m": 0.0,
        "min_root_child_length_m": 0.0,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.0,
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    {
        "variant_name": "reject_min_edge_0p15",
        "short_edge_policy": "reject",
        "min_edge_length_m": 0.15,
        "min_root_child_length_m": 0.0,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.0,
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    {
        "variant_name": "reject_min_edge_0p25",
        "short_edge_policy": "reject",
        "min_edge_length_m": 0.25,
        "min_root_child_length_m": 0.0,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.0,
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    {
        "variant_name": "reject_root_child_0p25",
        "short_edge_policy": "allow",
        "min_edge_length_m": 0.0,
        "min_root_child_length_m": 0.25,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.0,
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    {
        "variant_name": "reject_root_distance_0p25",
        "short_edge_policy": "allow",
        "min_edge_length_m": 0.0,
        "min_root_child_length_m": 0.0,
        "min_root_distance_m": 0.25,
        "crop_min_length_m": 0.0,
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    {
        "variant_name": "crop_min_length_0p15",
        "short_edge_policy": "crop",
        "min_edge_length_m": 0.0,
        "min_root_child_length_m": 0.0,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.15,
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    {
        "variant_name": "crop_min_length_0p25",
        "short_edge_policy": "crop",
        "min_edge_length_m": 0.0,
        "min_root_child_length_m": 0.0,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.25,
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    {
        "variant_name": "density_limited",
        "short_edge_policy": "allow",
        "min_edge_length_m": 0.0,
        "min_root_child_length_m": 0.0,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.0,
        "density_radius_m": 0.25,
        "max_nodes_per_density_radius": 1,
    },
    {
        "variant_name": "combined_source_like",
        "short_edge_policy": "crop",
        "min_edge_length_m": 0.0,
        "min_root_child_length_m": 0.25,
        "min_root_distance_m": 0.0,
        "crop_min_length_m": 0.15,
        "density_radius_m": 0.25,
        "max_nodes_per_density_radius": 1,
    },
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
        av = [float(v) for v in a]
        bv = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv))))


def corr_or_none(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(ys) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
    if x.size < 3 or float(np.std(x)) <= EPS or float(np.std(y)) <= EPS:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def git_status_short(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"error: {exc}"
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip()}"
    return completed.stdout.strip()


def scan_rollout_like_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def copy_variant_viz(variant_dir: Path) -> dict[str, str]:
    mapping = {
        "selected_branch_topdown.png": "variant_selected_branch_topdown.png",
        "mini_rrt_tree_topdown.png": "variant_tree_topdown.png",
        "gain_cost_scatter.png": "variant_gain_cost_scatter.png",
    }
    copied: dict[str, str] = {}
    for src_name, dst_name in mapping.items():
        src = variant_dir / src_name
        dst = variant_dir / dst_name
        if src.is_file():
            shutil.copyfile(src, dst)
            copied[dst_name] = str(dst)
    return copied


def build_run_args(base_args: argparse.Namespace, variant: dict[str, Any], variant_dir: Path) -> argparse.Namespace:
    values = {
        "case_json": base_args.case_json,
        "episode_dir": base_args.episode_dir,
        "observed_state": "",
        "pose_json": "",
        "camera_info": "",
        "episode_summary": "",
        "prediction_npz": "",
        "output_dir": str(variant_dir),
        "seed": base_args.seed,
        "num_nodes": base_args.num_nodes,
        "max_extension_m": base_args.max_extension_m,
        "sample_mode": base_args.sample_mode,
        "gain_mode": base_args.gain_mode,
        "path_cost_mode": base_args.path_cost_mode,
        "v_max": base_args.v_max,
        "yaw_rate": 1.0,
        "robot_radius_m": base_args.robot_radius_m,
        "voxel_size": base_args.voxel_size,
        "raycast_stride": base_args.raycast_stride,
        "num_yaw_samples": base_args.num_yaw_samples,
        "max_ray_length_m": base_args.max_ray_length_m,
        "tau": 0.1,
        "save_viz": bool(base_args.save_viz),
        "profile": True,
    }
    values.update(variant)
    return argparse.Namespace(**values)


def summarize_variant(
    *,
    variant_name: str,
    variant_dir: Path,
    summary: dict[str, Any],
    segments: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    non_root = [row for row in segments if row.get("segment_id") != "root"]
    root_world = summary.get("root", {}).get("resolved_world") or summary.get("root", {}).get("world")
    decision = summary.get("decision", {})
    selected = decision.get("selected_child") or {}
    best = decision.get("best_descendant") or {}
    selected_grid = selected.get("end_grid")
    best_grid = best.get("end_grid")
    selected_world = selected.get("end_world")
    best_world = best.get("end_world")
    selected_distance = summary.get("comparison", {}).get("mini_rrt", {}).get("selected_child_distance_from_root_m")
    if selected_distance is None:
        selected_distance = euclidean(root_world, selected_world)
    best_distance = summary.get("comparison", {}).get("mini_rrt", {}).get("best_descendant_distance_from_root_m")
    if best_distance is None:
        best_distance = euclidean(root_world, best_world)

    lengths = [float(row.get("segment_length_m", 0.0)) for row in non_root if row.get("segment_length_m") is not None]
    depths = [int(row.get("depth", 0)) for row in non_root if row.get("depth") is not None]
    spread = [
        euclidean(root_world, row.get("end_world"))
        for row in non_root
        if row.get("end_world") is not None and euclidean(root_world, row.get("end_world")) is not None
    ]
    spread_values = [float(v) for v in spread if v is not None]
    local_ratios: list[float] = []
    inverse_lengths: list[float] = []
    for row in non_root:
        gain = float(row.get("gain", 0.0) or 0.0)
        cost = float(row.get("cost", 0.0) or 0.0)
        length = float(row.get("segment_length_m", 0.0) or 0.0)
        if cost > EPS and length > EPS:
            local_ratios.append(gain / cost)
            inverse_lengths.append(1.0 / length)
    correlation = corr_or_none(local_ratios, inverse_lengths)
    selected_child_is_n0140 = selected.get("segment_id") == "n0140"
    selected_differs_from_one_step = not same_grid(selected_grid, ONE_STEP_BASELINE_GRID)
    selected_differs_from_decoupled = not same_grid(selected_grid, DECOUPLED_GRID)
    nonlocal_branch = bool(
        (selected_distance is not None and float(selected_distance) >= 0.5)
        or (best_distance is not None and float(best_distance) >= 1.0)
    )

    row = {
        "variant_name": variant_name,
        "random_seed": int(summary.get("parameters", {}).get("seed", 0)),
        "status": "completed" if bool(summary.get("tree", {}).get("built_successfully", False)) else "failed",
        "output_dir": str(variant_dir),
        "accepted_nodes": int(summary.get("tree", {}).get("accepted_nodes_excluding_root", 0)),
        "rejected_samples": int(summary.get("tree", {}).get("rejected_samples", 0)),
        "rejection_reason_counts": summary.get("tree", {}).get("rejected_reason_counts", {}),
        "selected_child_id": selected.get("segment_id"),
        "selected_child_grid": selected_grid,
        "selected_child_world": selected_world,
        "selected_child_distance_from_root_m": selected_distance,
        "selected_child_segment_length_m": selected.get("segment_length_m"),
        "selected_child_local_gain": selected.get("gain"),
        "selected_child_cost": selected.get("cost"),
        "selected_child_value": selected.get("value"),
        "best_descendant_id": best.get("segment_id"),
        "best_descendant_grid": best_grid,
        "best_descendant_world": best_world,
        "best_descendant_distance_from_root_m": best_distance,
        "best_descendant_accumulated_gain": best.get("accumulated_gain"),
        "best_descendant_accumulated_cost": best.get("accumulated_cost"),
        "best_descendant_depth": best.get("depth"),
        "selected_child_is_n0140": selected_child_is_n0140,
        "selected_child_differs_from_one_step_baseline_grid": selected_differs_from_one_step,
        "selected_child_differs_from_decoupled_grid": selected_differs_from_decoupled,
        "segment_length_median": percentile(lengths, 50),
        "segment_length_p25": percentile(lengths, 25),
        "segment_length_p75": percentile(lengths, 75),
        "segment_length_min": min(lengths) if lengths else None,
        "segment_length_max": max(lengths) if lengths else None,
        "local_gain_cost_vs_inverse_length_correlation": correlation,
        "tree_depth_max": max(depths) if depths else None,
        "tree_depth_median": percentile([float(v) for v in depths], 50),
        "spatial_spread_radius_max_m": max(spread_values) if spread_values else None,
        "spatial_spread_radius_median_m": percentile(spread_values, 50),
        "nonlocal_branch_found": nonlocal_branch,
        "runtime_seconds": summary.get("profile", {}).get("total_time_s"),
        "observed_state_hash_unchanged": summary.get("map", {}).get("observed_state_hash_unchanged"),
        "prediction_writeback": summary.get("safety", {}).get("prediction_writeback"),
        "prediction_used_for_traversability_collision": summary.get("safety", {}).get(
            "prediction_used_for_traversability_collision"
        ),
    }

    length_rows = [
        {
            "variant_name": variant_name,
            "segment_id": item.get("segment_id"),
            "parent_id": item.get("parent_id"),
            "depth": item.get("depth"),
            "segment_length_m": item.get("segment_length_m"),
            "gain": item.get("gain"),
            "cost": item.get("cost"),
            "value": item.get("value"),
            "is_selected_child": item.get("segment_id") == selected.get("segment_id"),
            "is_best_descendant": item.get("segment_id") == best.get("segment_id"),
        }
        for item in non_root
    ]
    corr_row = {
        "variant_name": variant_name,
        "local_gain_cost_vs_inverse_length_correlation": correlation,
        "n": len(local_ratios),
        "segment_length_median": row["segment_length_median"],
        "selected_child_segment_length_m": row["selected_child_segment_length_m"],
    }
    return row, length_rows, corr_row


def choose_best_variant(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    completed = [row for row in rows if row.get("status") == "completed"]
    if not completed:
        return None
    preferred_order = [
        "combined_source_like",
        "crop_min_length_0p25",
        "crop_min_length_0p15",
        "reject_root_child_0p25",
        "reject_root_distance_0p25",
        "reject_min_edge_0p25",
        "reject_min_edge_0p15",
        "density_limited",
        "baseline_allow",
    ]
    viable = [
        row
        for row in completed
        if not bool(row.get("selected_child_is_n0140", False))
        and int(row.get("accepted_nodes", 0)) >= max(16, int(0.2 * max(int(row.get("accepted_nodes", 0)), 256)))
    ]
    nonlocal_viable = [row for row in viable if bool(row.get("nonlocal_branch_found", False))]
    pool = nonlocal_viable or viable or completed
    by_name = {row["variant_name"]: row for row in pool}
    for name in preferred_order:
        if name in by_name:
            return by_name[name]
    return max(
        pool,
        key=lambda row: (
            float(row.get("best_descendant_distance_from_root_m") or 0.0),
            float(row.get("selected_child_distance_from_root_m") or 0.0),
            int(row.get("accepted_nodes") or 0),
        ),
    )


def add_baseline_diffs(rows: list[dict[str, Any]]) -> None:
    baseline = next((row for row in rows if row.get("variant_name") == "baseline_allow"), None)
    baseline_grid = baseline.get("selected_child_grid") if baseline else None
    for row in rows:
        row["selected_child_differs_from_baseline_allow"] = not same_grid(row.get("selected_child_grid"), baseline_grid)


def write_source_reference(path: Path) -> None:
    lines = [
        "# Source Min-Length / Density Reference",
        "",
        "- External source inspected read-only:",
        f"  `{EXTERNAL_SOURCE_DIR}`",
        "- `rrt.cpp` reads `crop_min_length` and `min_path_length` parameters.",
        "- `connectPoses(...)` rejects segments shorter than `p_min_path_length_` before collision checking.",
        "- `adjustGoalPosition(...)` rejects short directions, caps max extension, and discards cropped segments at or below `p_crop_min_length_`.",
        "- `rrt_star.cpp` applies `max_density_range` by rejecting a new goal if an existing tree point is inside that radius.",
        "- Example reconstruction/exploration configs use `crop_segments: true`, `crop_min_length: 0.5`, `min_path_length: 0.5`, and `max_density_range: 1.0`.",
        "",
        "This Stage 4A-6.5k runner approximates those mechanisms offline on the saved observed map. It does not modify or build the external source.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def write_recommendation(path: Path, best: dict[str, Any] | None, rows: list[dict[str, Any]]) -> str:
    moved = [row["variant_name"] for row in rows if row.get("selected_child_differs_from_baseline_allow")]
    nonlocal_names = [row["variant_name"] for row in rows if row.get("nonlocal_branch_found")]
    if best is None:
        next_step = "debug offline mini-RRT variant execution"
        reason = "no variant completed"
    elif nonlocal_names:
        next_step = "no-prediction online one-step tree smoke, still no rollout"
        reason = f"offline variant(s) found nonlocal branches: {', '.join(nonlocal_names)}"
    elif moved:
        next_step = "offline sampling strategy or gain/raycast refinement before online use"
        reason = "minimum-length variants moved off n0140 but still selected local branches"
    else:
        next_step = "offline sampling/steering diagnosis"
        reason = "minimum-length variants did not change the selected child enough"
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- best variant: `{best.get('variant_name') if best else None}`",
        f"- next small task: {next_step}",
        f"- reason: {reason}",
        "- still offline-only evidence; do not claim coverage or observed_ratio improvement.",
        "- still not next: rollout, RL, PPO, BC/IL training, map_predict tree integration, or prediction writeback.",
    ]
    write_text(path, "\n".join(lines) + "\n")
    return next_step


def write_comparison_md(path: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None, next_step: str) -> None:
    moved = [row["variant_name"] for row in rows if row.get("selected_child_differs_from_baseline_allow")]
    off_n0140 = [row["variant_name"] for row in rows if not bool(row.get("selected_child_is_n0140", False))]
    nonlocal_names = [row["variant_name"] for row in rows if row.get("nonlocal_branch_found")]
    lines = [
        "# Stage 4A-6.5k Variant Comparison",
        "",
        f"- variants completed: `{sum(1 for row in rows if row.get('status') == 'completed')}` / `{len(rows)}`",
        f"- baseline selected child: `{next((row.get('selected_child_id') for row in rows if row.get('variant_name') == 'baseline_allow'), None)}`",
        f"- variants moving off baseline grid: `{moved}`",
        f"- variants moving off `n0140`: `{off_n0140}`",
        f"- variants finding nonlocal branch: `{nonlocal_names}`",
        f"- best variant: `{best.get('variant_name') if best else None}`",
        f"- recommended next: {next_step}",
        "",
        "## Per Variant",
    ]
    for row in rows:
        lines.append(
            "- `{}`: child `{}` grid `{}` distance `{}` m, best `{}` distance `{}` m, accepted `{}`, rejected `{}`, nonlocal `{}`.".format(
                row.get("variant_name"),
                row.get("selected_child_id"),
                row.get("selected_child_grid"),
                row.get("selected_child_distance_from_root_m"),
                row.get("best_descendant_id"),
                row.get("best_descendant_distance_from_root_m"),
                row.get("accepted_nodes"),
                row.get("rejected_samples"),
                row.get("nonlocal_branch_found"),
            )
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Offline-only mini-RRT variants.",
            "- Prediction disabled for these runs; measured-only `gain_exp`.",
            "- No observed_state writeback and no coverage-improvement claim.",
            "- No rollout, RL, PPO, BC, IL, map_predict rerun, or external source build.",
        ]
    )
    write_text(path, "\n".join(lines) + "\n")


def write_stage_summary_md(path: Path, payload: dict[str, Any]) -> None:
    best = payload.get("best_variant") or {}
    baseline = payload.get("baseline") or {}
    lines = [
        "# Stage 4A-6.5k Minimum-Edge Variant Summary",
        "",
        f"1. baseline_allow reproduced `n0140`? `{baseline.get('selected_child_id') == 'n0140'}`; selected `{baseline.get('selected_child_id')}`.",
        f"2. Variants moving selected child off `n0140`: `{payload.get('variants_moving_off_n0140')}`.",
        f"3. Variants finding nonlocal branch: `{payload.get('variants_finding_nonlocal_branch')}`.",
        f"4. New selected child is just another local point? `{payload.get('new_selected_child_local_interpretation')}`.",
        f"5. Min-edge thresholds collapsed accepted nodes? `{payload.get('accepted_node_count_collapsed')}`.",
        f"6. Crop policy more source-like / stable than reject? `{payload.get('crop_policy_interpretation')}`.",
        f"7. Density limiting reduced tiny-edge clustering? `{payload.get('density_limited_interpretation')}`.",
        f"8. Gain/cost still short-edge dominated? `{payload.get('short_edge_dominance_interpretation')}`.",
        f"9. Best next candidate: `{best.get('variant_name')}`.",
        f"10. Recommended next faithful direction: {payload.get('recommended_next_faithful_step')}",
        "",
        "This remains offline-only. Prediction was not used. No observed_ratio improvement is claimed. Direct rollout and RL are not recommended from this result.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def write_plots(output_dir: Path, rows: list[dict[str, Any]], length_rows: list[dict[str, Any]]) -> dict[str, str]:
    generated: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting fallback
        return {"plot_error": str(exc)}

    names = [row["variant_name"] for row in rows]
    x = np.arange(len(names))

    def save_current(name: str) -> None:
        path = output_dir / name
        plt.savefig(path, dpi=170, bbox_inches="tight")
        plt.close()
        generated[name] = str(path)

    plt.figure(figsize=(11, 4))
    plt.bar(x, [float(row.get("selected_child_distance_from_root_m") or 0.0) for row in rows], color="#4c78a8")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("selected child distance (m)")
    plt.title("Selected child distance by variant")
    save_current("selected_child_distance_by_variant.png")

    plt.figure(figsize=(11, 4))
    plt.bar(x, [float(row.get("segment_length_median") or 0.0) for row in rows], color="#72b7b2")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("median segment length (m)")
    plt.title("Segment length median by variant")
    save_current("segment_length_median_by_variant.png")

    plt.figure(figsize=(7, 5))
    for row in rows:
        plt.scatter(
            [float(row.get("selected_child_distance_from_root_m") or 0.0)],
            [float(row.get("selected_child_value") or 0.0)],
            label=row["variant_name"],
            s=35,
        )
    plt.xlabel("selected child distance from root (m)")
    plt.ylabel("selected child value")
    plt.title("Value vs distance")
    plt.legend(fontsize=7, loc="best")
    save_current("value_vs_distance_by_variant.png")

    reason_counts: Counter[str] = Counter()
    per_variant_reasons: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = row.get("rejection_reason_counts") or {}
        per_variant_reasons[row["variant_name"]] = {str(k): int(v) for k, v in counts.items()}
        reason_counts.update(per_variant_reasons[row["variant_name"]])
    top_reasons = [reason for reason, _ in reason_counts.most_common(8)]
    if top_reasons:
        bottom = np.zeros(len(rows), dtype=np.float64)
        plt.figure(figsize=(12, 5))
        for reason in top_reasons:
            values = np.asarray([per_variant_reasons[name].get(reason, 0) for name in names], dtype=np.float64)
            plt.bar(x, values, bottom=bottom, label=reason)
            bottom += values
        plt.xticks(x, names, rotation=35, ha="right")
        plt.ylabel("rejections")
        plt.title("Top rejection reasons by variant")
        plt.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        save_current("rejection_reasons_by_variant.png")
    return generated


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    all_length_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []

    external_status_before = git_status_short(EXTERNAL_SOURCE_DIR)
    for variant in VARIANTS:
        variant_name = str(variant["variant_name"])
        variant_dir = output_dir / variant_name
        variant_dir.mkdir(parents=True, exist_ok=True)
        print(f"[stage4a65k] running variant {variant_name}", flush=True)
        run_args = build_run_args(args, variant, variant_dir)
        status = "completed"
        error = None
        try:
            summary = run_mini_rrt(run_args)
            if args.save_viz:
                copy_variant_viz(variant_dir)
            segments = read_jsonl(variant_dir / "mini_rrt_tree_segments.jsonl")
            row, length_rows, corr_row = summarize_variant(
                variant_name=variant_name,
                variant_dir=variant_dir,
                summary=summary,
                segments=segments,
            )
        except Exception as exc:  # keep aggregate diagnostics even if one variant fails
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            row = {
                "variant_name": variant_name,
                "status": status,
                "output_dir": str(variant_dir),
                "error": error,
                "accepted_nodes": 0,
                "rejected_samples": 0,
            }
            length_rows = []
            corr_row = {"variant_name": variant_name, "error": error}
        row.update({"status": status, "error": error})
        row.update({key: variant[key] for key in variant if key != "variant_name"})
        rows.append(row)
        all_length_rows.extend(length_rows)
        corr_rows.append(corr_row)
        manifest.append(
            {
                "variant_name": variant_name,
                "status": status,
                "error": error,
                "output_dir": str(variant_dir),
                "parameters": variant,
                "selected_child_id": row.get("selected_child_id"),
                "selected_child_grid": row.get("selected_child_grid"),
                "accepted_nodes": row.get("accepted_nodes"),
                "rejected_samples": row.get("rejected_samples"),
            }
        )

    add_baseline_diffs(rows)
    baseline = next((row for row in rows if row.get("variant_name") == "baseline_allow"), None)
    best = choose_best_variant(rows)
    external_status_after = git_status_short(EXTERNAL_SOURCE_DIR)
    moved_off_n0140 = [row["variant_name"] for row in rows if not bool(row.get("selected_child_is_n0140", False))]
    nonlocal_names = [row["variant_name"] for row in rows if bool(row.get("nonlocal_branch_found", False))]
    completed_count = sum(1 for row in rows if row.get("status") == "completed")
    accepted_counts = [int(row.get("accepted_nodes") or 0) for row in rows if row.get("status") == "completed"]
    accepted_node_count_collapsed = bool(accepted_counts and min(accepted_counts) < max(16, int(0.25 * args.num_nodes)))
    crop_rows = [row for row in rows if str(row.get("short_edge_policy")) == "crop" and row.get("status") == "completed"]
    reject_rows = [row for row in rows if str(row.get("short_edge_policy")) == "reject" and row.get("status") == "completed"]
    density_row = next((row for row in rows if row.get("variant_name") == "density_limited"), None)
    short_corrs = [
        float(row["local_gain_cost_vs_inverse_length_correlation"])
        for row in rows
        if row.get("local_gain_cost_vs_inverse_length_correlation") is not None
    ]

    next_step = write_recommendation(output_dir / "recommended_next_faithful_step.md", best, rows)
    plot_paths = write_plots(output_dir, rows, all_length_rows) if args.save_viz else {}
    payload = {
        "stage": "Stage 4A-6.5k offline mini-RRT minimum-edge variant",
        "output_dir": str(output_dir),
        "inputs": {
            "case_json": args.case_json,
            "episode_dir": args.episode_dir,
            "diagnosis_dir": args.diagnosis_dir,
        },
        "variant_count": len(VARIANTS),
        "completed_count": completed_count,
        "failed_count": len(VARIANTS) - completed_count,
        "baseline": baseline,
        "best_variant": best,
        "variants": rows,
        "variants_moving_off_n0140": moved_off_n0140,
        "variants_finding_nonlocal_branch": nonlocal_names,
        "new_selected_child_local_interpretation": (
            "yes; no variant reached the nonlocal threshold"
            if not nonlocal_names and moved_off_n0140
            else "no; at least one variant reached the nonlocal threshold"
            if nonlocal_names
            else "no movement from baseline was observed"
        ),
        "accepted_node_count_collapsed": accepted_node_count_collapsed,
        "crop_policy_interpretation": (
            "crop variants completed and are closer to source crop-min-length controls than simple reject"
            if crop_rows
            else "crop variants did not complete"
        ),
        "density_limited_interpretation": (
            f"density_limited accepted {density_row.get('accepted_nodes')} nodes and selected {density_row.get('selected_child_id')}"
            if density_row
            else "density_limited missing"
        ),
        "short_edge_dominance_interpretation": (
            f"correlations remain high in at least one variant (max {max(short_corrs)})"
            if short_corrs and max(short_corrs) >= 0.75
            else "correlations dropped below the previous short-edge-dominance range"
            if short_corrs
            else "correlation unavailable"
        ),
        "recommended_next_faithful_step": next_step,
        "runtime_seconds": float(time.perf_counter() - started),
        "external_source_git_status_before": external_status_before,
        "external_source_git_status_after": external_status_after,
        "external_source_modified_or_built": external_status_before != external_status_after,
        "rollout_like_outputs_created": scan_rollout_like_outputs(output_dir),
        "plots": plot_paths,
        "safety": {
            "isaac_startup": False,
            "rollout": False,
            "online_expert_loop": False,
            "map_predict_rerun": False,
            "sscnet_inference_or_training": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_modified": False,
            "observed_state_modified": any(
                row.get("observed_state_hash_unchanged") is False for row in rows if row.get("status") == "completed"
            ),
            "prediction_writeback": any(bool(row.get("prediction_writeback")) for row in rows),
            "prediction_used_for_traversability_collision": any(
                bool(row.get("prediction_used_for_traversability_collision")) for row in rows
            ),
            "target_lr_target_hr_ground_truth_scoring": False,
            "external_source_modified_or_built": external_status_before != external_status_after,
        },
    }

    write_jsonl(output_dir / "variants_manifest.jsonl", manifest)
    write_csv(output_dir / "variants_summary.csv", rows)
    save_json(output_dir / "variants_summary.json", {"variants": rows})
    write_comparison_md(output_dir / "variants_comparison.md", rows, best, next_step)
    selected_fields = [
        "variant_name",
        "selected_child_id",
        "selected_child_grid",
        "selected_child_world",
        "selected_child_distance_from_root_m",
        "selected_child_segment_length_m",
        "selected_child_local_gain",
        "selected_child_cost",
        "selected_child_value",
        "best_descendant_id",
        "best_descendant_grid",
        "best_descendant_distance_from_root_m",
        "selected_child_is_n0140",
        "selected_child_differs_from_baseline_allow",
        "selected_child_differs_from_one_step_baseline_grid",
        "selected_child_differs_from_decoupled_grid",
        "nonlocal_branch_found",
    ]
    write_csv(output_dir / "selected_child_comparison.csv", rows, selected_fields)
    rejection_rows: list[dict[str, Any]] = []
    for row in rows:
        counts = row.get("rejection_reason_counts") or {}
        for reason, count in counts.items():
            rejection_rows.append({"variant_name": row["variant_name"], "reason": reason, "count": int(count)})
    write_csv(output_dir / "rejection_reason_comparison.csv", rejection_rows, ["variant_name", "reason", "count"])
    write_csv(output_dir / "segment_length_distribution_by_variant.csv", all_length_rows)
    write_csv(output_dir / "gain_cost_correlation_by_variant.csv", corr_rows)
    write_source_reference(output_dir / "source_min_length_reference.md")
    save_json(output_dir / "stage4a65k_min_edge_variant_summary.json", payload)
    write_stage_summary_md(output_dir / "stage4a65k_min_edge_variant_summary.md", payload)

    print(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_json", required=True)
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--diagnosis_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--gain_mode", choices=["exp", "hybrid", "sc"], default="exp")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--save_viz", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
