#!/usr/bin/env python3
"""Stage 4A-6.5r gated SC tree one-step smoke on saved Frame 2 artifacts.

This runner is offline only. It reads the Stage 4A-6.5p Frame 2 saved
observed_state and saved global_prediction_layer.npz, then evaluates the
source-protected mini-RRT tree once per gated SC formula. It does not launch
Isaac, rerun map_predict/SSCNet inference, execute an action, run rollout,
train, modify checkpoints, modify observed_state, or use prediction for
traversability/collision/ray blocking.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import (
    ROOT_ID,
    run as run_mini_rrt,
    scan_rollout_like_outputs,
    sha256_file,
    to_jsonable,
)
from sim_paper_expert import FREE, OCCUPIED, UNKNOWN
from sim_prediction_layer import SimPredictionLayer


PROFILE_NAME = "source_like_crop_min_length_0p25"
DEFAULT_STAGE4A65P_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke"
)
DEFAULT_STAGE4A65Q_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65q_sc_tree_branch_change_diagnosis"
)
DEFAULT_CALIBRATION_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a64_gain_gating/calibration"
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
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65r_gated_sc_tree_one_step_smoke"
)
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)

FORMULAS = [
    "measured_only",
    "raw_count",
    "weight_0p5",
    "weight_1p0",
    "cap25",
    "cap50",
    "confidence_weighted",
    "occupied_only",
    "confidence_weighted_cap25",
]

EXPECTED = {
    "measured_only": {"selected_child_id": "n0001", "selected_child_grid": [17, 16, 11]},
    "raw_count": {
        "selected_child_id": "n0127",
        "selected_child_grid": [11, 15, 11],
        "best_descendant_id": "n0162",
        "best_descendant_grid": [14, 15, 11],
    },
    "weight_0p5": {"selected_child_id": "n0001", "selected_child_grid": [17, 16, 11]},
    "weight_1p0": {"selected_child_id": "n0127", "selected_child_grid": [11, 15, 11]},
    "cap25": {"selected_child_id": "n0127", "selected_child_grid": [11, 15, 11]},
    "cap50": {"selected_child_id": "n0127", "selected_child_grid": [11, 15, 11]},
    "confidence_weighted": {"selected_child_id": "n0127", "selected_child_grid": [11, 15, 11]},
    "occupied_only": {"selected_child_id": "n0001", "selected_child_grid": [17, 16, 11]},
    "confidence_weighted_cap25": {"selected_child_id": "n0127", "selected_child_grid": [11, 15, 11]},
}

REQUIRED_STAGE4A65P_INPUTS = [
    "observed_state_frame002.npy",
    "frame002_prediction/global_prediction_layer.npz",
    "frame002_pose.json",
    "frame002_camera_info.json",
    "frame002_sc_tree_decision.json",
    "frame002_measured_tree_decision.json",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_*.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
            writer.writerow(
                {
                    key: json.dumps(to_jsonable(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key, "")
                    for key in fields
                }
            )


def git_status_short(path: Path) -> str:
    if not path.exists():
        return "missing"
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(path),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip()}"
    return completed.stdout.strip()


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
    if len(av) != len(bv):
        return None
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv))))


def min_mean_max(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(finite.min()), "mean": float(finite.mean()), "max": float(finite.max())}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def tree_args(args: argparse.Namespace, formula: str, tree_dir: Path, observed_path: Path, prediction_path: Path) -> argparse.Namespace:
    episode_dir = Path(args.episode_dir).resolve()
    is_measured = formula == "measured_only"
    return argparse.Namespace(
        case_json=str(Path(args.selected_case_json).resolve()),
        episode_dir=str(episode_dir),
        observed_state=str(observed_path),
        pose_json=str(Path(args.stage4a65p_dir).resolve() / "frame002_pose.json"),
        camera_info=str(Path(args.stage4a65p_dir).resolve() / "frame002_camera_info.json"),
        episode_summary=str(episode_dir / "episode_summary.json"),
        prediction_npz="" if is_measured else str(prediction_path),
        output_dir=str(tree_dir),
        seed=int(args.seed),
        num_nodes=int(args.num_nodes),
        max_extension_m=float(args.max_extension_m),
        sample_mode=str(args.sample_mode),
        gain_mode="exp" if is_measured else "hybrid",
        sc_gain_formula="measured_only" if is_measured else str(formula),
        path_cost_mode=str(args.path_cost_mode),
        v_max=float(args.v_max),
        yaw_rate=1.0,
        robot_radius_m=float(args.robot_radius_m),
        voxel_size=float(args.voxel_size),
        raycast_stride=int(args.raycast_stride),
        num_yaw_samples=int(args.num_yaw_samples),
        max_ray_length_m=float(args.max_ray_length_m),
        tau=float(args.tau),
        save_viz=bool(args.save_viz),
        profile=True,
        min_edge_length_m=float(args.min_edge_length_m),
        min_root_child_length_m=float(args.min_root_child_length_m),
        min_root_distance_m=float(args.min_root_distance_m),
        crop_min_length_m=float(args.crop_min_length_m),
        short_edge_policy=str(args.short_edge_policy),
        density_radius_m=float(args.density_radius_m),
        max_nodes_per_density_radius=int(args.max_nodes_per_density_radius),
        variant_name=f"stage4a65r_{formula}_{PROFILE_NAME}",
    )


def decision_parts(summary: dict[str, Any]) -> dict[str, Any]:
    decision = summary.get("decision", {})
    selected = decision.get("selected_child") or {}
    best = decision.get("best_descendant") or {}
    return {
        "raw_decision": decision,
        "selected_child_id": decision.get("selected_child_id"),
        "selected_child": selected,
        "selected_child_grid": selected.get("end_grid"),
        "selected_child_world": selected.get("end_world"),
        "selected_child_value": decision.get("selected_child_value") or selected.get("value"),
        "best_descendant_id": decision.get("selected_child_best_descendant_id"),
        "best_descendant": best,
        "best_descendant_grid": best.get("end_grid"),
        "best_descendant_world": best.get("end_world"),
        "best_descendant_value": best.get("value"),
        "root": summary.get("root", {}),
        "built_successfully": bool(summary.get("tree", {}).get("built_successfully")),
        "accepted_nodes": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
        "rejected_samples": summary.get("tree", {}).get("rejected_samples"),
    }


def path_sums(tree_dir: Path, best_segment_id: str | None) -> dict[str, Any]:
    segments = load_jsonl(tree_dir / "mini_rrt_tree_segments.jsonl")
    by_id = {str(row.get("segment_id")): row for row in segments}
    if not best_segment_id or best_segment_id not in by_id:
        return {"path_segment_ids": [], "available": False}
    path: list[str] = []
    current: str | None = str(best_segment_id)
    while current and current in by_id:
        path.append(current)
        current = by_id[current].get("parent_id")
    path.reverse()
    non_root = [segment_id for segment_id in path if segment_id != ROOT_ID]
    return {
        "available": True,
        "path_segment_ids": non_root,
        "gain_used": float(sum(safe_float(by_id[s].get("gain")) for s in non_root)),
        "gain_exp": float(sum(safe_float(by_id[s].get("gain_exp")) for s in non_root)),
        "raw_gain_sc": float(sum(safe_float(by_id[s].get("gain_sc")) for s in non_root)),
        "effective_gain_sc": float(sum(safe_float(by_id[s].get("effective_gain_sc")) for s in non_root)),
        "gain_hybrid_raw": float(sum(safe_float(by_id[s].get("gain_hybrid")) for s in non_root)),
        "gain_hybrid_effective": float(sum(safe_float(by_id[s].get("gain_hybrid_effective")) for s in non_root)),
        "gain_occ": float(sum(safe_float(by_id[s].get("gain_occ")) for s in non_root)),
        "gain_conf": float(sum(safe_float(by_id[s].get("gain_conf")) for s in non_root)),
        "cost": float(sum(safe_float(by_id[s].get("cost")) for s in non_root)),
    }


def gain_stats(tree_dir: Path) -> dict[str, Any]:
    segments = [row for row in load_jsonl(tree_dir / "mini_rrt_tree_segments.jsonl") if row.get("segment_id") != ROOT_ID]
    raw_sc = [safe_float(row.get("gain_sc")) for row in segments]
    eff_sc = [safe_float(row.get("effective_gain_sc")) for row in segments]
    gain_conf = [safe_float(row.get("gain_conf")) for row in segments]
    gain_occ = [safe_float(row.get("gain_occ")) for row in segments]
    return {
        "node_count_excluding_root": len(segments),
        "nodes_with_raw_gain_sc_positive": sum(1 for value in raw_sc if value > 0.0),
        "nodes_with_effective_gain_sc_positive": sum(1 for value in eff_sc if value > 0.0),
        "raw_gain_sc_min_mean_max": min_mean_max(raw_sc),
        "effective_gain_sc_min_mean_max": min_mean_max(eff_sc),
        "gain_occ_min_mean_max": min_mean_max(gain_occ),
        "gain_conf_min_mean_max": min_mean_max(gain_conf),
    }


def prediction_stats(prediction_path: Path, observed_state: np.ndarray, tau: float) -> dict[str, Any]:
    layer = SimPredictionLayer.from_npz(prediction_path)
    with np.load(prediction_path, allow_pickle=False) as data:
        valid = np.asarray(data["global_prediction_valid"], dtype=bool)
        confidence = np.asarray(data["global_confidence"], dtype=np.float32)
        occupied_prob = np.asarray(data["global_occupied_prob"], dtype=np.float32)
        alignment = str(np.asarray(data["alignment_convention"]).item())
        files = list(data.files)
    valid_tau = valid & (confidence >= float(tau))
    predicted_unmeasured = valid_tau & (observed_state == UNKNOWN)
    predicted_occupied = valid_tau & (occupied_prob >= 0.5)
    return {
        "prediction_npz": str(prediction_path),
        "files": files,
        "shape": [int(v) for v in layer.shape()],
        "observed_state_shape": [int(v) for v in observed_state.shape],
        "shape_aligned_to_observed_state": tuple(layer.shape()) == tuple(observed_state.shape),
        "alignment_convention": alignment,
        "tau": float(tau),
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "prediction_valid_tau_count": int(np.count_nonzero(valid_tau)),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "predicted_occupied_count": int(np.count_nonzero(predicted_occupied)),
        "large_dense_class_prob_saved": "class_prob" in files,
    }


def copy_tree_aliases(output_dir: Path, tree_dir: Path, formula: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for src_name, dst_name in [
        ("mini_rrt_tree_segments.jsonl", f"{formula}_tree_segments.jsonl"),
        ("gain_cost_value_table.csv", f"{formula}_gain_cost_value_table.csv"),
        ("subsequent_best_decision.json", f"{formula}_subsequent_best_decision.json"),
        ("mini_rrt_tree_summary.json", f"{formula}_mini_rrt_tree_summary.json"),
    ]:
        src = tree_dir / src_name
        if src.is_file():
            dst = output_dir / dst_name
            shutil.copyfile(src, dst)
            aliases[dst_name] = str(dst)
    for src_name, dst_name in [
        ("mini_rrt_tree_topdown.png", f"{formula}_tree_topdown.png"),
        ("selected_branch_topdown.png", f"{formula}_selected_branch_topdown.png"),
        ("gain_cost_scatter.png", f"{formula}_gain_cost_scatter.png"),
    ]:
        src = tree_dir / src_name
        if src.is_file():
            dst = output_dir / dst_name
            shutil.copyfile(src, dst)
            aliases[dst_name] = str(dst)
    return aliases


def topdown_projection(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def plot_base_map(ax: plt.Axes, observed_state: np.ndarray) -> None:
    proj = topdown_projection(observed_state)
    colors = np.array(
        [
            [0.86, 0.86, 0.86, 1.0],
            [0.93, 0.98, 0.99, 1.0],
            [0.62, 0.12, 0.12, 1.0],
        ]
    )
    ax.imshow(colors[proj].transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.grid(color="#111827", alpha=0.12, linewidth=0.4)


def xy_from_grid(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    try:
        return float(grid[0]), float(grid[1])
    except (TypeError, ValueError, IndexError):
        return None


def make_plots(output_dir: Path, observed_state: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, str]:
    plots: dict[str, str] = {}
    palette = {
        "measured_only": "#2563eb",
        "raw_count": "#dc2626",
        "weight_0p5": "#059669",
        "weight_1p0": "#b91c1c",
        "cap25": "#7c3aed",
        "cap50": "#9333ea",
        "confidence_weighted": "#f97316",
        "occupied_only": "#0891b2",
        "confidence_weighted_cap25": "#be123c",
    }

    fig, ax = plt.subplots(figsize=(8.4, 7.4), constrained_layout=True)
    plot_base_map(ax, observed_state)
    seen: set[str] = set()
    for row in rows:
        grid = row.get("selected_child_grid")
        xy = xy_from_grid(grid)
        if xy is None:
            continue
        label = f"{row['formula']} {row['selected_child_id']}"
        key = f"{row['selected_child_id']}:{grid}"
        marker = "o" if key not in seen else "x"
        seen.add(key)
        ax.scatter(
            [xy[0]],
            [xy[1]],
            s=78,
            c=palette.get(str(row["formula"]), "#4b5563"),
            marker=marker,
            edgecolor="#111827",
            linewidth=0.8,
            label=label,
            zorder=4,
        )
    root_grid = rows[0].get("root_grid") if rows else None
    root_xy = xy_from_grid(root_grid)
    if root_xy is not None:
        ax.scatter([root_xy[0]], [root_xy[1]], s=90, c="#111827", marker="s", label="root", zorder=5)
    ax.set_title("Stage 4A-6.5r gated formula selected children")
    ax.legend(loc="upper right", fontsize=7)
    path = output_dir / "gated_formula_selected_children_topdown.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["gated_formula_selected_children_topdown"] = str(path)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    labels = [str(row["formula"]) for row in rows]
    values = [safe_float(row.get("best_descendant_value")) for row in rows]
    colors = [palette.get(label, "#64748b") for label in labels]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("GlobalNormalizedGain value")
    ax.set_title("Selected branch value by formula")
    ax.tick_params(axis="x", rotation=32)
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "gated_formula_value_bar.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["gated_formula_value_bar"] = str(path)

    raw_segments = load_jsonl(output_dir / "raw_count_tree_segments.jsonl")
    non_root = [row for row in raw_segments if row.get("segment_id") != ROOT_ID]
    fig, ax = plt.subplots(figsize=(7.2, 5.8), constrained_layout=True)
    ax.scatter(
        [safe_float(row.get("gain_exp")) for row in non_root],
        [safe_float(row.get("effective_gain_sc")) for row in non_root],
        s=16,
        color="#64748b",
        alpha=0.62,
    )
    ax.set_xlabel("local gain_exp")
    ax.set_ylabel("local effective_gain_sc (raw_count)")
    ax.set_title("Frame 2 raw-count local gains")
    ax.grid(alpha=0.25)
    path = output_dir / "raw_count_gain_exp_vs_effective_sc.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots["raw_count_gain_exp_vs_effective_sc"] = str(path)

    return plots


def expected_result(formula: str, row: dict[str, Any]) -> dict[str, Any]:
    expected = EXPECTED.get(formula, {})
    selected_id_ok = row.get("selected_child_id") == expected.get("selected_child_id")
    selected_grid_ok = same_grid(row.get("selected_child_grid"), expected.get("selected_child_grid"))
    best_id_expected = expected.get("best_descendant_id")
    best_grid_expected = expected.get("best_descendant_grid")
    best_id_ok = True if best_id_expected is None else row.get("best_descendant_id") == best_id_expected
    best_grid_ok = True if best_grid_expected is None else same_grid(row.get("best_descendant_grid"), best_grid_expected)
    return {
        "expected": expected,
        "selected_child_id_ok": selected_id_ok,
        "selected_child_grid_ok": selected_grid_ok,
        "best_descendant_id_ok": best_id_ok,
        "best_descendant_grid_ok": best_grid_ok,
        "passed": bool(selected_id_ok and selected_grid_ok and best_id_ok and best_grid_ok),
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 4A-6.5r Gated SC Tree One-Step Smoke",
        "",
        f"- output: `{summary['output_dir']}`",
        f"- formulas: `{summary['formulas']}`",
        f"- all expected selections passed: `{summary['all_expected_selection_checks_passed']}`",
        f"- measured-only selected: `{summary['reference']['measured_only_selected_child_id']}` grid `{summary['reference']['measured_only_selected_child_grid']}`",
        f"- raw-count selected: `{summary['reference']['raw_count_selected_child_id']}` grid `{summary['reference']['raw_count_selected_child_grid']}`",
        f"- formulas preserving raw SC: `{summary['formula_groups']['preserving_raw_sc_selected_child']}`",
        f"- formulas returning to measured: `{summary['formula_groups']['returning_to_measured_selected_child']}`",
        f"- no Isaac/map_predict/SSCNet/rollout/training/action execution: `{summary['safety']['offline_saved_frame_only']}`",
        f"- prediction writeback / traversability / collision / ray blocking: `{summary['safety']['prediction_writeback']}` / `{summary['safety']['prediction_used_for_traversability_collision']}` / `{summary['safety']['prediction_used_for_traversability_collision']}` / `{summary['safety']['prediction_blocks_rays']}`",
        "",
        "Per-formula details are in `gated_formula_decisions.csv` and `gated_formula_decisions.json`.",
        "",
    ]
    write_text(path, "\n".join(lines))


def write_safety_md(path: Path, checklist: dict[str, Any]) -> None:
    lines = [
        "# Prediction And Source Safety",
        "",
        f"- Isaac startup: `{checklist['isaac_startup']}`",
        f"- map_predict rerun: `{checklist['map_predict_rerun']}`",
        f"- SSCNet inference/training: `{checklist['sscnet_inference_or_training']}`",
        f"- selected action execution: `{checklist['selected_action_execution']}`",
        f"- rollout: `{checklist['rollout']}`",
        f"- prediction writeback: `{checklist['prediction_writeback']}`",
        f"- prediction collision/traversability/ray blocking: `{checklist['prediction_used_for_traversability_collision']}` / `{checklist['prediction_blocks_rays']}`",
        f"- observed_state hash unchanged: `{checklist['observed_state_hash_unchanged']}`",
        f"- prediction hash unchanged: `{checklist['prediction_npz_hash_unchanged']}`",
        f"- checkpoint hash unchanged: `{checklist['checkpoint_hash_unchanged']}`",
        f"- external source status unchanged: `{checklist['external_source_status_unchanged']}`",
        "",
    ]
    write_text(path, "\n".join(lines))


def write_source_md(path: Path, checklist: dict[str, Any]) -> None:
    params = checklist["profile_parameters"]
    prediction = checklist["prediction"]
    lines = [
        "# Source Protection Checklist",
        "",
        f"- profile: `{checklist['profile_name']}`",
        f"- short_edge_policy / crop_min_length_m: `{params['short_edge_policy']}` / `{params['crop_min_length_m']}`",
        f"- nodes / max_extension_m / sample_mode: `{params['num_nodes']}` / `{params['max_extension_m']}` / `{params['sample_mode']}`",
        f"- cost mode / v_max: `{params['path_cost_mode']}` / `{params['v_max']}`",
        f"- raycast_stride / yaw samples / max ray length: `{params['raycast_stride']}` / `{params['num_yaw_samples']}` / `{params['max_ray_length_m']}`",
        f"- prediction NPZ: `{prediction['prediction_npz']}`",
        "- prediction is information-gain-only and is not used for writeback, traversability, collision, or ray blocking.",
        "",
    ]
    write_text(path, "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage_dir = Path(args.stage4a65p_dir).resolve()
    q_dir = Path(args.stage4a65q_dir).resolve()
    calibration_dir = Path(args.calibration_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED_STAGE4A65P_INPUTS if not (stage_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing Stage 4A-6.5p inputs: {missing}")

    observed_path = stage_dir / "observed_state_frame002.npy"
    prediction_path = stage_dir / "frame002_prediction/global_prediction_layer.npz"
    observed_hash_before = sha256_file(observed_path)
    prediction_hash_before = sha256_file(prediction_path)
    checkpoint_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_status_before = git_status_short(EXTERNAL_SOURCE_DIR)

    observed_state = np.load(observed_path)
    observed_state.setflags(write=False)
    pred_stats = prediction_stats(prediction_path, observed_state, float(args.tau))

    measured_ref = load_json(stage_dir / "frame002_measured_tree_decision.json")
    raw_ref = load_json(stage_dir / "frame002_sc_tree_decision.json")
    q_summary = load_json(q_dir / "frame002_gated_replay_summary.json") if (q_dir / "frame002_gated_replay_summary.json").is_file() else {}

    rows: list[dict[str, Any]] = []
    aliases: dict[str, dict[str, str]] = {}
    for formula in FORMULAS:
        tree_dir = output_dir / f"{formula}_tree_raw"
        summary = run_mini_rrt(tree_args(args, formula, tree_dir, observed_path, prediction_path))
        parts = decision_parts(summary)
        sums = path_sums(tree_dir, parts["best_descendant_id"])
        stats = gain_stats(tree_dir)
        expected = expected_result(formula, {**parts})
        row = {
            "formula": formula,
            "gain_mode": "exp" if formula == "measured_only" else "hybrid",
            "tree_dir": str(tree_dir),
            "selected_child_id": parts["selected_child_id"],
            "selected_child_grid": parts["selected_child_grid"],
            "selected_child_world": parts["selected_child_world"],
            "best_descendant_id": parts["best_descendant_id"],
            "best_descendant_grid": parts["best_descendant_grid"],
            "best_descendant_world": parts["best_descendant_world"],
            "best_descendant_value": parts["best_descendant_value"],
            "root_grid": parts["root"].get("resolved_grid") or parts["root"].get("grid"),
            "root_world": parts["root"].get("resolved_world") or parts["root"].get("world"),
            "accepted_nodes": parts["accepted_nodes"],
            "rejected_samples": parts["rejected_samples"],
            "path_segment_ids": sums.get("path_segment_ids"),
            "accumulated_gain_used": sums.get("gain_used"),
            "accumulated_gain_exp": sums.get("gain_exp"),
            "accumulated_raw_gain_sc": sums.get("raw_gain_sc"),
            "accumulated_effective_gain_sc": sums.get("effective_gain_sc"),
            "accumulated_gain_hybrid_raw": sums.get("gain_hybrid_raw"),
            "accumulated_gain_hybrid_effective": sums.get("gain_hybrid_effective"),
            "accumulated_gain_occ": sums.get("gain_occ"),
            "accumulated_gain_conf": sums.get("gain_conf"),
            "accumulated_cost": sums.get("cost"),
            "nodes_with_raw_gain_sc_positive": stats["nodes_with_raw_gain_sc_positive"],
            "nodes_with_effective_gain_sc_positive": stats["nodes_with_effective_gain_sc_positive"],
            "expected_selection": expected,
            "matches_expected": bool(expected["passed"]),
        }
        rows.append(row)
        aliases[formula] = copy_tree_aliases(output_dir, tree_dir, formula)
        save_json(output_dir / f"{formula}_tree_decision.json", row)

    measured_row = next(row for row in rows if row["formula"] == "measured_only")
    raw_row = next(row for row in rows if row["formula"] == "raw_count")
    measured_selected = str(measured_row["selected_child_id"])
    raw_selected = str(raw_row["selected_child_id"])
    for row in rows:
        row["returns_to_measured_selected_child"] = row["selected_child_id"] == measured_selected
        row["preserves_raw_sc_selected_child"] = row["selected_child_id"] == raw_selected
        row["selected_child_delta_from_measured_m"] = euclidean(
            row.get("selected_child_world"), measured_row.get("selected_child_world")
        )
        row["best_descendant_delta_from_measured_m"] = euclidean(
            row.get("best_descendant_world"), measured_row.get("best_descendant_world")
        )

    write_csv(
        output_dir / "gated_formula_decisions.csv",
        rows,
        [
            "formula",
            "gain_mode",
            "selected_child_id",
            "selected_child_grid",
            "best_descendant_id",
            "best_descendant_grid",
            "best_descendant_value",
            "accumulated_gain_exp",
            "accumulated_raw_gain_sc",
            "accumulated_effective_gain_sc",
            "accumulated_gain_hybrid_effective",
            "accumulated_cost",
            "returns_to_measured_selected_child",
            "preserves_raw_sc_selected_child",
            "selected_child_delta_from_measured_m",
            "matches_expected",
        ],
    )
    save_json(output_dir / "gated_formula_decisions.json", rows)
    plots = make_plots(output_dir, observed_state, rows)

    observed_hash_after = sha256_file(observed_path)
    prediction_hash_after = sha256_file(prediction_path)
    checkpoint_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_status_after = git_status_short(EXTERNAL_SOURCE_DIR)

    source_checklist = {
        "profile_name": PROFILE_NAME,
        "profile_parameters": {
            "short_edge_policy": str(args.short_edge_policy),
            "crop_min_length_m": float(args.crop_min_length_m),
            "min_edge_length_m": float(args.min_edge_length_m),
            "min_root_child_length_m": float(args.min_root_child_length_m),
            "min_root_distance_m": float(args.min_root_distance_m),
            "density_radius_m": float(args.density_radius_m),
            "max_nodes_per_density_radius": int(args.max_nodes_per_density_radius),
            "num_nodes": int(args.num_nodes),
            "max_extension_m": float(args.max_extension_m),
            "sample_mode": str(args.sample_mode),
            "path_cost_mode": str(args.path_cost_mode),
            "v_max": float(args.v_max),
            "robot_radius_m": float(args.robot_radius_m),
            "voxel_size": float(args.voxel_size),
            "raycast_stride": int(args.raycast_stride),
            "num_yaw_samples": int(args.num_yaw_samples),
            "max_ray_length_m": float(args.max_ray_length_m),
            "seed": int(args.seed),
        },
        "prediction": {
            "enabled_for_information_gain_only": True,
            "prediction_npz": str(prediction_path),
            "prediction_writeback": False,
            "prediction_used_for_traversability_collision": False,
            "prediction_blocks_rays": False,
            "map_predict_rerun": False,
        },
    }

    safety = {
        "offline_saved_frame_only": True,
        "isaac_startup": False,
        "rgb_depth_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference_or_training": False,
        "two_frame": False,
        "selected_action_execution": False,
        "rollout": False,
        "online_open_ended_loop": False,
        "training_rl_ppo_bc_il": False,
        "checkpoint_modified": checkpoint_before != checkpoint_after,
        "checkpoint_hash_unchanged": checkpoint_before == checkpoint_after,
        "existing_observed_state_modified": observed_hash_before != observed_hash_after,
        "observed_state_hash_unchanged": observed_hash_before == observed_hash_after,
        "prediction_npz_modified": prediction_hash_before != prediction_hash_after,
        "prediction_npz_hash_unchanged": prediction_hash_before == prediction_hash_after,
        "prediction_writeback": False,
        "prediction_used_for_traversability_collision": False,
        "prediction_blocks_rays": False,
        "target_lr_target_hr_ground_truth_scoring": False,
        "external_source_modified_or_built": external_status_before != external_status_after,
        "external_source_status_unchanged": external_status_before == external_status_after,
        "coverage_improvement_claimed": False,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "external_source_git_status_before": external_status_before,
        "external_source_git_status_after": external_status_after,
    }
    prohibited = {
        pattern: sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
        for pattern in PROHIBITED_OUTPUT_PATTERNS
    }
    prohibited = {key: value for key, value in prohibited.items() if value}

    observed_hashes = {
        "observed_state_frame002": str(observed_path),
        "observed_state_sha256_before": observed_hash_before,
        "observed_state_sha256_after": observed_hash_after,
        "observed_state_hash_unchanged": observed_hash_before == observed_hash_after,
        "prediction_npz": str(prediction_path),
        "prediction_npz_sha256_before": prediction_hash_before,
        "prediction_npz_sha256_after": prediction_hash_after,
        "prediction_npz_hash_unchanged": prediction_hash_before == prediction_hash_after,
    }

    summary = {
        "stage": "Stage 4A-6.5r gated SC tree one-step smoke",
        "output_dir": str(output_dir),
        "formulas": FORMULAS,
        "inputs": {
            "stage4a65p_dir": str(stage_dir),
            "stage4a65q_dir": str(q_dir),
            "calibration_dir": str(calibration_dir),
            "selected_case_json": str(Path(args.selected_case_json).resolve()),
            "episode_dir": str(Path(args.episode_dir).resolve()),
            "observed_state_frame002": str(observed_path),
            "prediction_npz": str(prediction_path),
            "pose_json": str(stage_dir / "frame002_pose.json"),
            "camera_info": str(stage_dir / "frame002_camera_info.json"),
        },
        "prediction_stats": pred_stats,
        "reference": {
            "measured_only_selected_child_id": measured_ref.get("raw_decision", measured_ref).get("selected_child_id"),
            "measured_only_selected_child_grid": measured_ref.get("selected_child", {}).get("end_grid"),
            "raw_count_selected_child_id": raw_ref.get("raw_decision", raw_ref).get("selected_child_id"),
            "raw_count_selected_child_grid": raw_ref.get("selected_child", {}).get("end_grid"),
            "raw_count_best_descendant_id": raw_ref.get("raw_decision", raw_ref).get("selected_child_best_descendant_id"),
            "raw_count_best_descendant_grid": raw_ref.get("best_descendant", {}).get("end_grid"),
            "stage4a65q_gated_replay_summary": q_summary,
        },
        "formula_groups": {
            "preserving_raw_sc_selected_child": [
                row["formula"] for row in rows if row.get("preserves_raw_sc_selected_child")
            ],
            "returning_to_measured_selected_child": [
                row["formula"] for row in rows if row.get("returns_to_measured_selected_child")
            ],
        },
        "all_expected_selection_checks_passed": all(bool(row["matches_expected"]) for row in rows),
        "formula_decisions": rows,
        "source_protection_checklist": source_checklist,
        "prediction_safety_checklist": safety,
        "safety": safety,
        "observed_state_hashes": observed_hashes,
        "generated_files": {
            "formula_aliases": aliases,
            "plots": plots,
            "rollout_like_files_created_in_output_dir": scan_rollout_like_outputs(output_dir),
            "prohibited_output_matches": prohibited,
        },
        "conclusion": {
            "raw_count_reproduced_stage4a65p_frame2": raw_row["selected_child_id"] == "n0127"
            and raw_row["best_descendant_id"] == "n0162",
            "weight_0p5_returns_to_measured": next(row for row in rows if row["formula"] == "weight_0p5")[
                "selected_child_id"
            ]
            == measured_selected,
            "confidence_weighted_preserves_sc_branch": next(
                row for row in rows if row["formula"] == "confidence_weighted"
            )["selected_child_id"]
            == raw_selected,
            "occupied_only_returns_to_measured": next(row for row in rows if row["formula"] == "occupied_only")[
                "selected_child_id"
            ]
            == measured_selected,
            "still_not_rollout": True,
        },
    }

    save_json(output_dir / "source_protection_checklist.json", source_checklist)
    save_json(output_dir / "prediction_safety_checklist.json", safety)
    save_json(output_dir / "observed_state_hashes.json", observed_hashes)
    save_json(output_dir / "gated_sc_tree_one_step_summary.json", summary)
    write_summary_md(output_dir / "gated_sc_tree_one_step_summary.md", summary)
    write_safety_md(output_dir / "prediction_safety_checklist.md", safety)
    write_source_md(output_dir / "source_protection_checklist.md", source_checklist)
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                "Use the gated one-step result to choose a conservative formula for a later staged test.",
                "Do not jump directly to rollout from this saved-frame smoke.",
                "",
            ]
        ),
    )

    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65p_dir", default=DEFAULT_STAGE4A65P_DIR)
    parser.add_argument("--stage4a65q_dir", default=DEFAULT_STAGE4A65Q_DIR)
    parser.add_argument("--calibration_dir", default=DEFAULT_CALIBRATION_DIR)
    parser.add_argument("--selected_case_json", default=DEFAULT_SELECTED_CASE)
    parser.add_argument("--episode_dir", default=DEFAULT_EPISODE_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
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
    parser.add_argument("--min_edge_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_child_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_distance_m", type=float, default=0.0)
    parser.add_argument("--density_radius_m", type=float, default=0.0)
    parser.add_argument("--max_nodes_per_density_radius", type=int, default=0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--save_viz", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
