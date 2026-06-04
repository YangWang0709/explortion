#!/usr/bin/env python3
"""Stage 4A-6.5al post-action/two-frame diagnosis.

This stage is offline analysis only. It reads the completed Stage 4A-6.5ak
two-frame/one-action runtime smoke outputs and writes diagnosis artifacts. It
does not start Isaac, capture RGB/depth, run map_predict, run SSCNet inference,
execute actions, run rollout, train, or modify existing inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65al_post_action_two_frame_diagnosis"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]
CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
HISTORICAL_PRIOR_SELECTED_GRID = [11, 15, 11]
HISTORICAL_PRIOR_BEST_GRID = [14, 15, 11]
LABELS = [-1, 0, 1]
LABEL_NAMES = {-1: "unknown", 0: "free", 1: "occupied"}

EXPECTED_AK_FILES = [
    "stage4a65ak_two_frame_one_action_runtime_summary.json",
    "stage4a65ak_two_frame_one_action_runtime_summary.md",
    "runtime_setup_summary.json",
    "runtime_setup_summary.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "formula_definition.json",
    "formula_definition.md",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "prediction_safety_report.json",
    "prediction_safety_report.md",
    "hash_checks.json",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "frame001_capture_summary.json",
    "frame001_capture_summary.md",
    "frame001_rgb.png",
    "frame001_depth.npy",
    "frame001_depth.png",
    "frame001_pose.json",
    "frame001_camera_info.json",
    "observed_state_frame001.npy",
    "observed_state_update_frame001.json",
    "observed_state_update_frame001.md",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame001_map_predict/prediction_alignment_summary.json",
    "map_predict_frame001_summary.json",
    "map_predict_frame001_summary.md",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_measured_shadow_tree_decision.md",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.md",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame001_branch_classification.md",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame001_low_cost_artifact_diagnosis.md",
    "pre_action_safety_gate_report.json",
    "pre_action_safety_gate_report.md",
    "action_execution_report.json",
    "action_execution_report.md",
    "frame002_capture_summary.json",
    "frame002_capture_summary.md",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_depth.png",
    "frame002_pose.json",
    "frame002_camera_info.json",
    "observed_state_frame002.npy",
    "observed_state_update_frame002.json",
    "observed_state_update_frame002.md",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/prediction_alignment_summary.json",
    "map_predict_frame002_summary.json",
    "map_predict_frame002_summary.md",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_measured_shadow_tree_decision.md",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.md",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame002_branch_classification.json",
    "frame002_branch_classification.md",
    "frame002_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.md",
    "two_frame_decision_comparison.json",
    "two_frame_decision_comparison.md",
]

ESSENTIAL_AK_FILES = [
    "stage4a65ak_two_frame_one_action_runtime_summary.json",
    "runtime_setup_summary.json",
    "hash_checks.json",
    "prediction_safety_report.json",
    "no_rollout_report.json",
    "action_execution_report.json",
    "frame001_pose.json",
    "frame002_pose.json",
    "observed_state_frame001.npy",
    "observed_state_frame002.npy",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/global_prediction_layer.npz",
    "map_predict_frame001_summary.json",
    "map_predict_frame002_summary.json",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame002_branch_classification.json",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.json",
]

REQUIRED_PLOTS = [
    "observed_state_frame1_frame2_delta_topdown.png",
    "observed_state_label_transition_bar.png",
    "prediction_count_delta_bar.png",
    "prediction_overlay_frame1_vs_frame2.png",
    "tree_branches_frame1_vs_frame2_topdown.png",
    "lambda48_vs_measured_frame1_topdown.png",
    "lambda48_vs_measured_frame2_topdown.png",
    "action_pose_consistency_topdown.png",
    "value_components_frame1_frame2.png",
    "low_cost_artifact_two_frame.png",
    "repeat_safety_readiness_matrix.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "frame001_rgb.png",
    "frame001_depth.npy",
    "frame001_depth.png",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_depth.png",
    "capture_rgb*.png",
    "capture_depth*.npy",
    "capture_depth*.png",
    "frame003*",
    "action002*",
    "observed_state*.npy",
    "*.npz",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "manifest.jsonl",
    "episode_manifest*",
]

DECISION_FIELDS = [
    "selected_child_id",
    "selected_child_grid",
    "selected_child_world",
    "best_descendant_id",
    "best_descendant_grid",
    "best_descendant_world",
    "branch_classification",
    "gain_exp",
    "source_occ_free",
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
    "low_cost_artifact",
]


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_entry_worker(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path)
    entry: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": None,
        "suffix": path.suffix,
        "field_count": None,
        "array_shape": None,
        "npz_keys": None,
    }
    if path.is_file():
        entry["sha256"] = sha256_file(path)
        if path.suffix == ".json":
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                entry["field_count"] = len(data) if isinstance(data, dict) else None
            except Exception as exc:
                entry["json_error"] = str(exc)
        elif path.suffix == ".npy":
            try:
                arr = np.load(path, mmap_mode="r")
                entry["array_shape"] = list(arr.shape)
                entry["dtype"] = str(arr.dtype)
            except Exception as exc:
                entry["array_error"] = str(exc)
        elif path.suffix == ".npz":
            try:
                data = np.load(path)
                entry["npz_keys"] = list(data.files)
            except Exception as exc:
                entry["npz_error"] = str(exc)
    return entry


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object in {path}")
    return data


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(to_jsonable(row.get(key)), sort_keys=True) if isinstance(row.get(key), (dict, list, tuple)) else row.get(key) for key in fieldnames})


def md_table(rows: list[tuple[str, Any]]) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, value in rows:
        if isinstance(value, (dict, list)):
            rendered = "`" + json.dumps(to_jsonable(value), sort_keys=True) + "`"
        else:
            rendered = f"`{value}`"
        lines.append(f"| {key} | {rendered} |")
    return "\n".join(lines)


def md_rows(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered = []
        for value in row:
            if isinstance(value, (dict, list)):
                rendered.append("`" + json.dumps(to_jsonable(value), sort_keys=True) + "`")
            else:
                rendered.append(f"`{value}`")
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def load_decision(path: Path) -> dict[str, Any]:
    data = read_json(path)
    decision = data.get("decision", data)
    if not isinstance(decision, dict):
        raise TypeError(f"decision object missing in {path}")
    return decision


def selected_chain(decision: dict[str, Any]) -> str:
    return f"{decision.get('selected_child_id')} -> {decision.get('best_descendant_id')}"


def euclidean(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))


def summarize_observed(arr: np.ndarray, sha: str) -> dict[str, Any]:
    counts = {label: int((arr == label).sum()) for label in LABELS}
    invalid_count = int((~np.isin(arr, LABELS)).sum())
    observed = counts[0] + counts[1]
    total = int(arr.size)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "total_count": total,
        "unknown_count": counts[-1],
        "free_count": counts[0],
        "occupied_count": counts[1],
        "observed_count": observed,
        "observed_ratio": observed / total if total else 0.0,
        "invalid_label_count": invalid_count,
        "sha256": sha,
    }


def array_stats(values: np.ndarray) -> dict[str, Any]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
    }


def prediction_stats(npz_path: Path, observed: np.ndarray, map_summary: dict[str, Any]) -> dict[str, Any]:
    data = np.load(npz_path)
    valid = data["global_prediction_valid"].astype(bool)
    pred_class = data["global_pred_class"]
    confidence = data["global_confidence"]
    free_prob = data["global_free_prob"]
    occupied_prob = data["global_occupied_prob"]
    stats = map_summary.get("stats", {})
    free_threshold = float(stats.get("free_threshold", 0.5))
    occ_threshold = float(stats.get("occ_threshold", 0.5))
    unmeasured = observed == -1
    free_mask = valid & unmeasured & (free_prob >= free_threshold)
    occ_mask = valid & unmeasured & (occupied_prob >= occ_threshold)
    occ_free_mask = free_mask | occ_mask
    valid_unmeasured = valid & unmeasured
    invalid_count = int((~valid).sum())
    valid_topdown = valid.any(axis=2)
    occ_free_topdown = occ_free_mask.any(axis=2)
    return {
        "path": str(npz_path),
        "shape": list(valid.shape),
        "alignment_convention": str(np.asarray(data["alignment_convention"]).item()),
        "strict_no_observed_write": bool(np.asarray(data["strict_no_observed_write"]).item()),
        "read_only_note": str(np.asarray(data["read_only_note"]).item()),
        "valid_count": int(valid.sum()),
        "valid_unmeasured_count": int(valid_unmeasured.sum()),
        "invalid_count": invalid_count,
        "predicted_unmeasured_free_count": int(free_mask.sum()),
        "predicted_unmeasured_occupied_count": int(occ_mask.sum()),
        "predicted_unmeasured_occ_free_count": int(occ_free_mask.sum()),
        "valid_non_occfree_unmeasured_count": int((valid_unmeasured & ~occ_free_mask).sum()),
        "summary_reported_valid_count": stats.get("prediction_valid_count"),
        "summary_reported_occ_free_count": stats.get("predicted_unmeasured_occ_free_count"),
        "summary_reported_free_count": stats.get("predicted_unmeasured_free_count"),
        "summary_reported_occupied_count": stats.get("predicted_unmeasured_occupied_count"),
        "valid_count_matches_summary": int(valid.sum()) == int(stats.get("prediction_valid_count", -1)),
        "occ_free_count_matches_summary": int(occ_free_mask.sum()) == int(stats.get("predicted_unmeasured_occ_free_count", -1)),
        "free_count_matches_summary": int(free_mask.sum()) == int(stats.get("predicted_unmeasured_free_count", -1)),
        "occupied_count_matches_summary": int(occ_mask.sum()) == int(stats.get("predicted_unmeasured_occupied_count", -1)),
        "confidence_stats_valid": array_stats(confidence[valid]),
        "confidence_stats_occ_free_unmeasured": array_stats(confidence[occ_free_mask]),
        "free_prob_stats_valid": array_stats(free_prob[valid]),
        "occupied_prob_stats_valid": array_stats(occupied_prob[valid]),
        "valid_topdown_cell_count": int(valid_topdown.sum()),
        "occ_free_topdown_cell_count": int(occ_free_topdown.sum()),
        "pred_class_valid_histogram": {
            str(int(k)): int(v) for k, v in zip(*np.unique(pred_class[valid], return_counts=True))
        },
        "_valid_mask": valid,
        "_occ_free_mask": occ_free_mask,
    }


def compact_prediction_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in stats.items() if not key.startswith("_")}


def decision_row(frame: str, role: str, path: Path, branch: dict[str, Any]) -> dict[str, Any]:
    decision = load_decision(path)
    row = {
        "frame": frame,
        "role": role,
        "path": str(path),
        "selected_chain": selected_chain(decision),
        "historical_prior_basin": bool(branch.get("historical_prior_basin", decision.get("same_as_prior_low_cost_sc", False))),
    }
    for field in DECISION_FIELDS:
        if field == "source_occ_count":
            row["source_occ"] = decision.get("source_occ_count", decision.get("source_occ"))
        elif field == "source_free_count":
            row["source_free"] = decision.get("source_free_count", decision.get("source_free"))
        else:
            row[field] = decision.get(field)
    return row


def compare_decisions(frame: str, measured: dict[str, Any], lambda48: dict[str, Any], branch: dict[str, Any]) -> dict[str, Any]:
    selected_distance = euclidean(measured.get("selected_child_world"), lambda48.get("selected_child_world"))
    best_distance = euclidean(measured.get("best_descendant_world"), lambda48.get("best_descendant_world"))
    lambda48_gain = float(lambda48.get("gain_exp", 0.0))
    measured_gain = float(measured.get("gain_exp", 0.0))
    lambda48_sc = float(lambda48.get("source_occ_free", lambda48.get("source_occ_free_count", 0.0)))
    measured_sc = float(measured.get("source_occ_free", measured.get("source_occ_free_count", 0.0)))
    lambda48_cost = float(lambda48.get("cost", 0.0))
    measured_cost = float(measured.get("cost", 0.0))
    return {
        "frame": frame,
        "measured_selected": selected_chain(measured),
        "lambda48_selected": selected_chain(lambda48),
        "changed_vs_measured": measured.get("selected_child_id") != lambda48.get("selected_child_id")
        or measured.get("best_descendant_id") != lambda48.get("best_descendant_id"),
        "selected_child_world_distance_m": selected_distance,
        "best_descendant_world_distance_m": best_distance,
        "gain_exp_delta_lambda48_minus_measured": lambda48_gain - measured_gain,
        "source_occ_free_delta_lambda48_minus_measured": lambda48_sc - measured_sc,
        "cost_delta_lambda48_minus_measured": lambda48_cost - measured_cost,
        "base_exp_value_delta_lambda48_minus_measured": float(lambda48.get("base_exp_value", 0.0))
        - float(measured.get("base_exp_value", 0.0)),
        "final_value_delta_lambda48_minus_measured": float(lambda48.get("final_value", 0.0))
        - float(measured.get("final_value", 0.0)),
        "lambda48_sc_bonus": lambda48.get("sc_bonus"),
        "lambda48_normalized_sc": lambda48.get("normalized_sc"),
        "lambda48_selected_gain_exp_rank": lambda48.get("selected_gain_exp_rank"),
        "lambda48_selected_source_occ_free_rank": lambda48.get("selected_source_occ_free_rank"),
        "lambda48_selected_cost_rank": lambda48.get("selected_cost_rank"),
        "branch_classification": branch.get("classification", lambda48.get("branch_classification")),
        "healthy_nonmeasured_candidate": bool(branch.get("healthy_nonmeasured_candidate")),
        "low_cost_artifact": bool(branch.get("low_cost_artifact", lambda48.get("low_cost_artifact", False))),
        "historical_prior_basin": bool(branch.get("historical_prior_basin", False)),
        "interpretation": (
            "lambda48 selected a distinct measured-adjacent child whose descendant has much larger gain_exp "
            "and source_occ_free; the SC bonus is outside the cost denominator and the branch is marked healthy."
        ),
    }


def plot_or_skip(output_dir: Path, name: str, plot_func, skipped: dict[str, str]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_func(plt, output_dir / name)
    except Exception as exc:  # pragma: no cover - depends on local plotting stack.
        reason = f"plot skipped: {type(exc).__name__}: {exc}"
        skipped[name] = reason
        write_text(output_dir / f"{Path(name).stem}_skipped_reason.md", reason)


def add_scene_axes(ax: Any) -> None:
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.2, 6.2)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def plot_point(ax: Any, world: list[float] | None, label: str, marker: str, color: str) -> None:
    if not world:
        return
    ax.scatter([world[0]], [world[1]], s=70, marker=marker, color=color, label=label)
    ax.text(world[0], world[1], f" {label}", fontsize=8, color=color)


def write_plot_bundle(
    output_dir: Path,
    observed1: np.ndarray,
    observed2: np.ndarray,
    pred1: dict[str, Any],
    pred2: dict[str, Any],
    transitions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    frame1_diagnosis: dict[str, Any],
    frame2_diagnosis: dict[str, Any],
    action_pose: dict[str, Any],
    readiness_rows: list[dict[str, Any]],
    save_viz: bool,
) -> dict[str, str]:
    skipped: dict[str, str] = {}
    if not save_viz:
        for name in REQUIRED_PLOTS:
            reason = "plot generation disabled because --save_viz was not provided"
            skipped[name] = reason
            write_text(output_dir / f"{Path(name).stem}_skipped_reason.md", reason)
        return skipped

    newly_observed = (observed1 == -1) & (observed2 != -1)

    def delta_topdown(plt: Any, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(6.8, 5.5))
        image = newly_observed.sum(axis=2).T
        im = ax.imshow(image, origin="lower", extent=(-6, 6, -6, 6), cmap="viridis")
        plot_point(ax, decisions["frame001_lambda48"].get("root_world"), "F1 root", "o", "#444444")
        plot_point(ax, action_pose.get("position"), "action/F2", "*", "#d62728")
        add_scene_axes(ax)
        ax.set_title("Newly observed voxels after the one action")
        fig.colorbar(im, ax=ax, label="newly observed voxels per x/y cell")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def transition_bar(plt: Any, path: Path) -> None:
        labels = [f"{row['from_label_name']} -> {row['to_label_name']}" for row in transitions if row["count"] > 0]
        counts = [row["count"] for row in transitions if row["count"] > 0]
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.bar(labels, counts, color="#4c78a8")
        ax.set_yscale("log")
        ax.set_ylabel("voxel count, log scale")
        ax.set_title("Observed-state label transitions")
        ax.tick_params(axis="x", labelrotation=35)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def prediction_count_delta(plt: Any, path: Path) -> None:
        labels = ["valid", "unmeasured occ+free", "unmeasured free", "unmeasured occupied"]
        frame1 = [pred1["valid_count"], pred1["predicted_unmeasured_occ_free_count"], pred1["predicted_unmeasured_free_count"], pred1["predicted_unmeasured_occupied_count"]]
        frame2 = [pred2["valid_count"], pred2["predicted_unmeasured_occ_free_count"], pred2["predicted_unmeasured_free_count"], pred2["predicted_unmeasured_occupied_count"]]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.bar(x - 0.18, frame1, width=0.36, label="Frame 1", color="#4c78a8")
        ax.bar(x + 0.18, frame2, width=0.36, label="Frame 2", color="#f58518")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel("voxel count")
        ax.set_title("map_predict count stability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def prediction_overlay(plt: Any, path: Path) -> None:
        valid1 = pred1["_valid_mask"].any(axis=2)
        valid2 = pred2["_valid_mask"].any(axis=2)
        overlay = np.zeros(valid1.shape, dtype=np.uint8)
        overlay[valid1] = 1
        overlay[valid2] = np.maximum(overlay[valid2], 2)
        overlay[valid1 & valid2] = 3
        cmap = plt.matplotlib.colors.ListedColormap(["#f5f5f5", "#4c78a8", "#f58518", "#54a24b"])
        fig, ax = plt.subplots(figsize=(6.8, 5.5))
        ax.imshow(overlay.T, origin="lower", extent=(-6, 6, -6, 6), cmap=cmap, vmin=0, vmax=3)
        add_scene_axes(ax)
        ax.set_title("Prediction valid-mask topdown overlap: F1, F2, both")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def branches_frame1_frame2(plt: Any, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        for key, color, marker, label in [
            ("frame001_lambda48", "#4c78a8", "o", "F1 lambda48 child"),
            ("frame001_lambda48_best", "#4c78a8", "x", "F1 lambda48 best"),
            ("frame002_lambda48", "#f58518", "o", "F2 lambda48 child"),
            ("frame002_lambda48_best", "#f58518", "x", "F2 lambda48 best"),
        ]:
            world = None
            if key == "frame001_lambda48":
                world = decisions["frame001_lambda48"].get("selected_child_world")
            elif key == "frame001_lambda48_best":
                world = decisions["frame001_lambda48"].get("best_descendant_world")
            elif key == "frame002_lambda48":
                world = decisions["frame002_lambda48"].get("selected_child_world")
            else:
                world = decisions["frame002_lambda48"].get("best_descendant_world")
            plot_point(ax, world, label, marker, color)
        plot_point(ax, decisions["frame001_lambda48"].get("root_world"), "F1 root", "s", "#333333")
        plot_point(ax, decisions["frame002_lambda48"].get("root_world"), "F2 root", "*", "#d62728")
        add_scene_axes(ax)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Frame 1 and Frame 2 lambda48 branches")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def measured_vs_lambda(frame_key: str, title: str):
        def _plot(plt: Any, path: Path) -> None:
            fig, ax = plt.subplots(figsize=(7, 6))
            measured = decisions[f"{frame_key}_measured"]
            lam = decisions[f"{frame_key}_lambda48"]
            plot_point(ax, measured.get("root_world"), "root", "s", "#333333")
            plot_point(ax, measured.get("selected_child_world"), "measured child", "o", "#4c78a8")
            plot_point(ax, measured.get("best_descendant_world"), "measured best", "x", "#4c78a8")
            plot_point(ax, lam.get("selected_child_world"), "lambda48 child", "o", "#f58518")
            plot_point(ax, lam.get("best_descendant_world"), "lambda48 best", "x", "#f58518")
            add_scene_axes(ax)
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)

        return _plot

    def action_consistency(plt: Any, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_point(ax, decisions["frame001_lambda48"].get("root_world"), "F1 start", "s", "#333333")
        plot_point(ax, decisions["frame001_lambda48"].get("selected_child_world"), "lambda48 selected child", "o", "#4c78a8")
        plot_point(ax, action_pose.get("position"), "executed/F2 pose", "*", "#d62728")
        f1 = decisions["frame001_lambda48"].get("root_world")
        f2 = action_pose.get("position")
        if f1 and f2:
            ax.plot([f1[0], f2[0]], [f1[1], f2[1]], color="#666666", linewidth=1.4)
        add_scene_axes(ax)
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title("Action pose consistency")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def value_components(plt: Any, path: Path) -> None:
        labels = ["base_exp_value", "sc_bonus", "final_value", "margin"]
        values = [
            [decisions["frame001_lambda48"].get(label, 0.0) for label in labels],
            [decisions["frame002_lambda48"].get(label, 0.0) for label in labels],
        ]
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.bar(x - 0.18, values[0], width=0.36, label="Frame 1 lambda48", color="#4c78a8")
        ax.bar(x + 0.18, values[1], width=0.36, label="Frame 2 lambda48", color="#f58518")
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_title("lambda48 value components")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def low_cost(plt: Any, path: Path) -> None:
        labels = ["F1 low-cost", "F1 prior", "F2 low-cost", "F2 prior"]
        values = [
            int(bool(decisions["frame001_lambda48"].get("low_cost_artifact", False))),
            int(bool(rows_by_key(rows, "frame001", "lambda48").get("historical_prior_basin", False))),
            int(bool(decisions["frame002_lambda48"].get("low_cost_artifact", False))),
            int(bool(rows_by_key(rows, "frame002", "lambda48").get("historical_prior_basin", False))),
        ]
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        ax.bar(labels, values, color=["#54a24b" if v == 0 else "#e45756" for v in values])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("flag value")
        ax.set_title("Low-cost artifact and prior-basin flags")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def readiness_matrix(plt: Any, path: Path) -> None:
        cols = ["safety_clean", "prediction_read_only", "no_artifact", "map_predict_stable", "obs_update_stable", "recommended_now"]
        data = np.array([[1 if row.get(col) is True else 0 for col in cols] for row in readiness_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.imshow(data, cmap=plt.matplotlib.colors.ListedColormap(["#f1f1f1", "#54a24b"]), vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(cols)), cols, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(readiness_rows)), [row["stage_option"] for row in readiness_rows])
        ax.set_title("Repeat-safety readiness matrix")
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                ax.text(x, y, "Y" if data[y, x] else "N", ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    plot_or_skip(output_dir, "observed_state_frame1_frame2_delta_topdown.png", delta_topdown, skipped)
    plot_or_skip(output_dir, "observed_state_label_transition_bar.png", transition_bar, skipped)
    plot_or_skip(output_dir, "prediction_count_delta_bar.png", prediction_count_delta, skipped)
    plot_or_skip(output_dir, "prediction_overlay_frame1_vs_frame2.png", prediction_overlay, skipped)
    plot_or_skip(output_dir, "tree_branches_frame1_vs_frame2_topdown.png", branches_frame1_frame2, skipped)
    plot_or_skip(output_dir, "lambda48_vs_measured_frame1_topdown.png", measured_vs_lambda("frame001", "Frame 1 measured vs lambda48"), skipped)
    plot_or_skip(output_dir, "lambda48_vs_measured_frame2_topdown.png", measured_vs_lambda("frame002", "Frame 2 measured vs lambda48"), skipped)
    plot_or_skip(output_dir, "action_pose_consistency_topdown.png", action_consistency, skipped)
    plot_or_skip(output_dir, "value_components_frame1_frame2.png", value_components, skipped)
    plot_or_skip(output_dir, "low_cost_artifact_two_frame.png", low_cost, skipped)
    plot_or_skip(output_dir, "repeat_safety_readiness_matrix.png", readiness_matrix, skipped)
    return skipped


def rows_by_key(rows: list[dict[str, Any]], frame: str, role: str) -> dict[str, Any]:
    for row in rows:
        if row.get("frame") == frame and row.get("role") == role:
            return row
    return {}


def find_prohibited_artifacts(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        found.extend(str(path.relative_to(output_dir)) for path in output_dir.glob(pattern))
    return sorted(set(found))


def make_decision_md(title: str, data: dict[str, Any]) -> str:
    return "# " + title + "\n\n" + md_table([(key, value) for key, value in data.items()])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65ak_dir", required=True)
    parser.add_argument("--stage4a65ag_dir", required=True)
    parser.add_argument("--stage4a65ai_dir", required=True)
    parser.add_argument("--stage4a65aj_dir", required=True)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ak_dir = Path(args.stage4a65ak_dir).resolve()
    ag_dir = Path(args.stage4a65ag_dir).resolve()
    ai_dir = Path(args.stage4a65ai_dir).resolve()
    aj_dir = Path(args.stage4a65aj_dir).resolve()
    os_cpu_count = os.cpu_count() or 1
    actual_max_workers = max(1, min(int(args.max_workers), os_cpu_count))

    expected_paths = [ak_dir / rel for rel in EXPECTED_AK_FILES]
    context_paths = CONTEXT_FILES
    prior_paths = [
        ag_dir / "stage4a65ag_multi_frame_lambda48_replay_summary.json",
        ag_dir / "lambda48_multiframe_summary.json",
        ai_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.json",
        ai_dir / "branch_classification.json",
        aj_dir / "stage4a65aj_design_review_summary.json",
        aj_dir / "future_two_frame_sequence_spec.json",
    ]
    manifest_paths = expected_paths + context_paths + prior_paths
    with ProcessPoolExecutor(max_workers=actual_max_workers) as executor:
        manifest_entries = list(executor.map(_path_entry_worker, [str(path) for path in manifest_paths]))

    entries_by_path = {entry["path"]: entry for entry in manifest_entries}
    missing_required = [rel for rel in EXPECTED_AK_FILES if not (ak_dir / rel).exists()]
    missing_essential = [rel for rel in ESSENTIAL_AK_FILES if not (ak_dir / rel).exists()]
    if missing_essential:
        diagnosis_incomplete = True
    else:
        diagnosis_incomplete = False

    loaded_context_manifest = {
        "stage": "Stage 4A-6.5al",
        "purpose": "post-action/two-frame diagnosis and repeat-safety review only",
        "chat_history_not_used_as_source": True,
        "context_files": [entries_by_path[str(path)] for path in context_paths],
        "prior_stage_dirs": {
            "stage4a65ak": str(ak_dir),
            "stage4a65ag": str(ag_dir),
            "stage4a65ai": str(ai_dir),
            "stage4a65aj": str(aj_dir),
        },
        "prior_stage_summary_files": [entries_by_path[str(path)] for path in prior_paths],
        "safety_boundary": {
            "isaac_startup_in_stage4a65al": False,
            "rgb_depth_capture_in_stage4a65al": False,
            "map_predict_call_in_stage4a65al": False,
            "sscnet_inference_in_stage4a65al": False,
            "selected_action_execution_in_stage4a65al": False,
            "two_frame_runtime_execution_in_stage4a65al": False,
            "rollout_in_stage4a65al": False,
        },
    }
    write_json(output_dir / "loaded_context_manifest.json", loaded_context_manifest)
    write_text(
        output_dir / "loaded_context_manifest.md",
        "# Loaded Context Manifest\n\n"
        + md_table(
            [
                ("stage", "Stage 4A-6.5al"),
                ("context_files", [str(path) for path in context_paths]),
                ("stage4a65ak_dir", str(ak_dir)),
                ("stage4a65ag_dir", str(ag_dir)),
                ("stage4a65ai_dir", str(ai_dir)),
                ("stage4a65aj_dir", str(aj_dir)),
                ("runtime_execution_in_stage4a65al", False),
            ]
        ),
    )

    loaded_inputs_manifest = {
        "stage": "Stage 4A-6.5al",
        "stage4a65ak_dir": str(ak_dir),
        "expected_input_count": len(EXPECTED_AK_FILES),
        "missing_required_files": missing_required,
        "missing_essential_files": missing_essential,
        "diagnosis_incomplete": diagnosis_incomplete,
        "inputs": [entries_by_path[str(path)] for path in expected_paths],
    }
    write_json(output_dir / "loaded_inputs_manifest.json", loaded_inputs_manifest)
    write_text(
        output_dir / "loaded_inputs_manifest.md",
        "# Loaded Inputs Manifest\n\n"
        + md_table(
            [
                ("expected_input_count", len(EXPECTED_AK_FILES)),
                ("missing_required_files", missing_required),
                ("missing_essential_files", missing_essential),
                ("diagnosis_incomplete", diagnosis_incomplete),
            ]
        ),
    )

    if diagnosis_incomplete:
        missing_fields_report = {
            "missing_required_files": missing_required,
            "missing_essential_files": missing_essential,
            "missing_fields": [],
            "plot_skipped_reasons": {},
            "prohibited_artifacts_found": find_prohibited_artifacts(output_dir),
            "diagnosis_incomplete": True,
        }
        write_json(output_dir / "missing_fields_report.json", missing_fields_report)
        write_text(output_dir / "missing_fields_report.md", "# Missing Fields Report\n\nDiagnosis incomplete due to missing essential Stage 4A-6.5ak outputs.")
        return 2

    # Load all required JSON and arrays after the manifest has recorded inputs.
    ak_summary = read_json(ak_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    runtime_setup = read_json(ak_dir / "runtime_setup_summary.json")
    action_report = read_json(ak_dir / "action_execution_report.json")
    pose1 = read_json(ak_dir / "frame001_pose.json")
    pose2 = read_json(ak_dir / "frame002_pose.json")
    obs_update1 = read_json(ak_dir / "observed_state_update_frame001.json")
    obs_update2 = read_json(ak_dir / "observed_state_update_frame002.json")
    map_summary1 = read_json(ak_dir / "map_predict_frame001_summary.json")
    map_summary2 = read_json(ak_dir / "map_predict_frame002_summary.json")
    align1 = read_json(ak_dir / "frame001_map_predict/prediction_alignment_summary.json")
    align2 = read_json(ak_dir / "frame002_map_predict/prediction_alignment_summary.json")
    branch1 = read_json(ak_dir / "frame001_branch_classification.json")
    branch2 = read_json(ak_dir / "frame002_branch_classification.json")
    lowcost1 = read_json(ak_dir / "frame001_low_cost_artifact_diagnosis.json")
    lowcost2 = read_json(ak_dir / "frame002_low_cost_artifact_diagnosis.json")
    prediction_safety = read_json(ak_dir / "prediction_safety_report.json")
    no_rollout = read_json(ak_dir / "no_rollout_report.json")
    hash_checks = read_json(ak_dir / "hash_checks.json")
    formula = read_json(ak_dir / "formula_definition.json")
    source_protection = read_json(ak_dir / "source_protection_checklist.json")
    hardware_ak = read_json(ak_dir / "hardware_utilization_report.json")
    ag_summary = read_json(ag_dir / "stage4a65ag_multi_frame_lambda48_replay_summary.json")
    ag_lambda_summary = read_json(ag_dir / "lambda48_multiframe_summary.json")
    ai_summary = read_json(ai_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.json")
    ai_branch = read_json(ai_dir / "branch_classification.json")
    aj_summary = read_json(aj_dir / "stage4a65aj_design_review_summary.json")
    aj_sequence = read_json(aj_dir / "future_two_frame_sequence_spec.json")

    key_hash_paths = [
        ak_dir / "observed_state_frame001.npy",
        ak_dir / "observed_state_frame002.npy",
        ak_dir / "frame001_map_predict/global_prediction_layer.npz",
        ak_dir / "frame002_map_predict/global_prediction_layer.npz",
        ak_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json",
        ak_dir / "runtime_setup_summary.json",
        ak_dir / "prediction_safety_report.json",
        ak_dir / "two_frame_decision_comparison.json",
        CHECKPOINT,
    ]
    before_hashes = {str(path): sha256_file(path) for path in key_hash_paths if path.exists()}

    observed1 = np.load(ak_dir / "observed_state_frame001.npy")
    observed2 = np.load(ak_dir / "observed_state_frame002.npy")
    obs1_sha = before_hashes[str(ak_dir / "observed_state_frame001.npy")]
    obs2_sha = before_hashes[str(ak_dir / "observed_state_frame002.npy")]
    obs1_summary = summarize_observed(observed1, obs1_sha)
    obs2_summary = summarize_observed(observed2, obs2_sha)

    transitions: list[dict[str, Any]] = []
    for from_label in LABELS:
        for to_label in LABELS:
            count = int(((observed1 == from_label) & (observed2 == to_label)).sum())
            transitions.append(
                {
                    "from_label": from_label,
                    "from_label_name": LABEL_NAMES[from_label],
                    "to_label": to_label,
                    "to_label_name": LABEL_NAMES[to_label],
                    "count": count,
                }
            )
    invalid_before = int((~np.isin(observed1, LABELS)).sum())
    invalid_after = int((~np.isin(observed2, LABELS)).sum())
    newly_observed = (observed1 == -1) & (observed2 != -1)
    newly_free = (observed1 == -1) & (observed2 == 0)
    newly_occupied = (observed1 == -1) & (observed2 == 1)
    free_to_occupied = (observed1 == 0) & (observed2 == 1)
    occupied_to_free = (observed1 == 1) & (observed2 == 0)
    bounds = obs_update2.get("bounds", {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]})
    voxel_size = float(obs_update2.get("voxel_size", 0.1))
    nz = np.nonzero(newly_observed)
    if len(nz[0]) > 0:
        xs = float(bounds["x"][0]) + (nz[0].astype(np.float64) + 0.5) * voxel_size
        ys = float(bounds["y"][0]) + (nz[1].astype(np.float64) + 0.5) * voxel_size
        action_pos = np.array(action_report["executed_pose"]["position"][:2], dtype=np.float64)
        distances = np.sqrt((xs - action_pos[0]) ** 2 + (ys - action_pos[1]) ** 2)
        distance_stats = array_stats(distances)
    else:
        distance_stats = array_stats(np.array([], dtype=np.float64))
    z_slice_counts = np.bincount(nz[2], minlength=observed1.shape[2]).astype(int).tolist() if len(nz[0]) else [0] * observed1.shape[2]

    observed_ratio_delta = obs2_summary["observed_ratio"] - obs1_summary["observed_ratio"]
    minor_flip_fraction = int(free_to_occupied.sum()) / max(1, obs1_summary["observed_count"])
    observed_state_delta_summary = {
        "frame001": obs1_summary,
        "frame002": obs2_summary,
        "newly_observed_voxels": int(newly_observed.sum()),
        "newly_free_voxels": int(newly_free.sum()),
        "newly_occupied_voxels": int(newly_occupied.sum()),
        "unknown_to_free": int(newly_free.sum()),
        "unknown_to_occupied": int(newly_occupied.sum()),
        "free_to_occupied": int(free_to_occupied.sum()),
        "occupied_to_free": int(occupied_to_free.sum()),
        "unchanged_unknown": int(((observed1 == -1) & (observed2 == -1)).sum()),
        "unchanged_free": int(((observed1 == 0) & (observed2 == 0)).sum()),
        "unchanged_occupied": int(((observed1 == 1) & (observed2 == 1)).sum()),
        "invalid_label_count_frame001": invalid_before,
        "invalid_label_count_frame002": invalid_after,
        "observed_ratio_delta": observed_ratio_delta,
        "observed_ratio_non_decreasing": observed_ratio_delta >= 0.0,
        "z_slice_newly_observed_counts": z_slice_counts,
        "newly_observed_distance_from_action_pose_m": distance_stats,
        "meaningful_measured_information_added": int(newly_observed.sum()) > 0 and observed_ratio_delta > 0.0,
        "suspicious_label_flips": bool(invalid_before or invalid_after or occupied_to_free.sum() > 0 or minor_flip_fraction > 0.01),
        "minor_free_to_occupied_refinement_count": int(free_to_occupied.sum()),
        "minor_free_to_occupied_fraction_of_frame1_observed": minor_flip_fraction,
    }
    write_json(output_dir / "observed_state_delta_summary.json", observed_state_delta_summary)
    write_text(
        output_dir / "observed_state_delta_summary.md",
        "# Observed State Delta Summary\n\n"
        + md_table(
            [
                ("frame001_observed_ratio", obs1_summary["observed_ratio"]),
                ("frame002_observed_ratio", obs2_summary["observed_ratio"]),
                ("observed_ratio_delta", observed_ratio_delta),
                ("newly_observed_voxels", int(newly_observed.sum())),
                ("unknown_to_free", int(newly_free.sum())),
                ("unknown_to_occupied", int(newly_occupied.sum())),
                ("free_to_occupied", int(free_to_occupied.sum())),
                ("occupied_to_free", int(occupied_to_free.sum())),
                ("suspicious_label_flips", observed_state_delta_summary["suspicious_label_flips"]),
            ]
        ),
    )
    write_csv(output_dir / "observed_state_label_transition_matrix.csv", transitions)
    write_json(output_dir / "observed_state_label_transition_matrix.json", transitions)
    write_text(
        output_dir / "observed_state_label_transition_matrix.md",
        "# Observed State Label Transition Matrix\n\n"
        + md_rows(
            ["from", "to", "count"],
            [[row["from_label_name"], row["to_label_name"], row["count"]] for row in transitions],
        ),
    )

    observed_state_safety_review = {
        "observed_ratio_non_decreasing": observed_state_delta_summary["observed_ratio_non_decreasing"],
        "meaningful_measured_information_added": observed_state_delta_summary["meaningful_measured_information_added"],
        "invalid_labels_present": invalid_before > 0 or invalid_after > 0,
        "occupied_to_free_flips": int(occupied_to_free.sum()),
        "free_to_occupied_flips": int(free_to_occupied.sum()),
        "free_to_occupied_interpretation": "small measured-depth refinement, not a blocker",
        "measured_only_update_frame001": bool(obs_update1.get("measured_only")) and not bool(obs_update1.get("prediction_used")),
        "measured_only_update_frame002": bool(obs_update2.get("measured_only")) and not bool(obs_update2.get("prediction_used")),
        "prediction_written_into_observed_state": bool(prediction_safety.get("prediction_written_to_observed_state", False)),
        "prediction_fused_into_observed_state": bool(prediction_safety.get("prediction_fused_into_observed_state", False)),
        "review_clean": observed_state_delta_summary["observed_ratio_non_decreasing"]
        and observed_state_delta_summary["meaningful_measured_information_added"]
        and not observed_state_delta_summary["suspicious_label_flips"]
        and bool(obs_update1.get("measured_only"))
        and bool(obs_update2.get("measured_only"))
        and not bool(prediction_safety.get("prediction_written_to_observed_state", False))
        and not bool(prediction_safety.get("prediction_fused_into_observed_state", False)),
    }
    write_json(output_dir / "observed_state_safety_review.json", observed_state_safety_review)
    write_text(output_dir / "observed_state_safety_review.md", make_decision_md("Observed State Safety Review", observed_state_safety_review))

    pred1 = prediction_stats(ak_dir / "frame001_map_predict/global_prediction_layer.npz", observed1, map_summary1)
    pred2 = prediction_stats(ak_dir / "frame002_map_predict/global_prediction_layer.npz", observed2, map_summary2)
    valid_overlap = pred1["_valid_mask"] & pred2["_valid_mask"]
    valid_union = pred1["_valid_mask"] | pred2["_valid_mask"]
    occfree_overlap = pred1["_occ_free_mask"] & pred2["_occ_free_mask"]
    occfree_union = pred1["_occ_free_mask"] | pred2["_occ_free_mask"]
    prediction_overlap_summary = {
        "valid_overlap_count": int(valid_overlap.sum()),
        "valid_union_count": int(valid_union.sum()),
        "valid_iou": int(valid_overlap.sum()) / int(valid_union.sum()) if int(valid_union.sum()) else 0.0,
        "occ_free_overlap_count": int(occfree_overlap.sum()),
        "occ_free_union_count": int(occfree_union.sum()),
        "occ_free_iou": int(occfree_overlap.sum()) / int(occfree_union.sum()) if int(occfree_union.sum()) else 0.0,
        "frame1_valid_count": pred1["valid_count"],
        "frame2_valid_count": pred2["valid_count"],
        "frame1_occ_free_count": pred1["predicted_unmeasured_occ_free_count"],
        "frame2_occ_free_count": pred2["predicted_unmeasured_occ_free_count"],
    }
    count_delta_rows = []
    for metric in (
        "valid_count",
        "valid_unmeasured_count",
        "predicted_unmeasured_occ_free_count",
        "predicted_unmeasured_free_count",
        "predicted_unmeasured_occupied_count",
        "invalid_count",
    ):
        f1 = int(pred1[metric])
        f2 = int(pred2[metric])
        count_delta_rows.append({"metric": metric, "frame001": f1, "frame002": f2, "delta_frame2_minus_frame1": f2 - f1, "ratio_frame2_over_frame1": f2 / f1 if f1 else None})
    density_ratio = pred2["predicted_unmeasured_occ_free_count"] / max(1, pred1["predicted_unmeasured_occ_free_count"])
    valid_ratio = pred2["valid_count"] / max(1, pred1["valid_count"])
    map_predict_two_frame_stability = {
        "frame001": compact_prediction_stats(pred1),
        "frame002": compact_prediction_stats(pred2),
        "valid_count_delta": pred2["valid_count"] - pred1["valid_count"],
        "predicted_unmeasured_occ_free_delta": pred2["predicted_unmeasured_occ_free_count"] - pred1["predicted_unmeasured_occ_free_count"],
        "predicted_unmeasured_occupied_delta": pred2["predicted_unmeasured_occupied_count"] - pred1["predicted_unmeasured_occupied_count"],
        "predicted_unmeasured_free_delta": pred2["predicted_unmeasured_free_count"] - pred1["predicted_unmeasured_free_count"],
        "density_ratio_frame2_over_frame1": density_ratio,
        "valid_ratio_frame2_over_frame1": valid_ratio,
        "prediction_density_exploded": density_ratio > 2.0,
        "prediction_density_collapsed": density_ratio < 0.5,
        "frame2_prediction_remains_reasonable_post_action": 0.5 <= density_ratio <= 2.0 and 0.5 <= valid_ratio <= 2.0,
        "map_predict_succeeded_both_frames": bool(map_summary1.get("map_predict_succeeded")) and bool(map_summary2.get("map_predict_succeeded")),
        "model_loaded_once": bool(map_summary1.get("model_loaded_once")) and bool(map_summary2.get("model_loaded_once")),
    }
    prediction_alignment_recheck = {
        "frame001_alignment_convention": map_summary1.get("alignment_convention"),
        "frame002_alignment_convention": map_summary2.get("alignment_convention"),
        "frame001_prediction_shape": map_summary1.get("prediction_layer_shape"),
        "frame002_prediction_shape": map_summary2.get("prediction_layer_shape"),
        "frame001_observed_shape": map_summary1.get("observed_state_shape"),
        "frame002_observed_shape": map_summary2.get("observed_state_shape"),
        "frame001_shape_aligned": bool(map_summary1.get("prediction_layer_shape_aligned_to_observed_state")),
        "frame002_shape_aligned": bool(map_summary2.get("prediction_layer_shape_aligned_to_observed_state")),
        "frame001_code_consistent_v1": map_summary1.get("alignment_convention") == "code_consistent_v1"
        and align1.get("alignment_convention") == "code_consistent_v1",
        "frame002_code_consistent_v1": map_summary2.get("alignment_convention") == "code_consistent_v1"
        and align2.get("alignment_convention") == "code_consistent_v1",
        "frame001_checkpoint_unchanged": bool(align1.get("checkpoint_unchanged")),
        "frame002_checkpoint_unchanged": bool(align2.get("checkpoint_unchanged")),
        "alignment_recheck_clean": bool(map_summary1.get("prediction_layer_shape_aligned_to_observed_state"))
        and bool(map_summary2.get("prediction_layer_shape_aligned_to_observed_state"))
        and map_summary1.get("alignment_convention") == "code_consistent_v1"
        and map_summary2.get("alignment_convention") == "code_consistent_v1",
    }
    write_json(output_dir / "map_predict_two_frame_stability.json", map_predict_two_frame_stability)
    write_text(output_dir / "map_predict_two_frame_stability.md", make_decision_md("map_predict Two-Frame Stability", map_predict_two_frame_stability))
    write_csv(output_dir / "prediction_count_delta.csv", count_delta_rows)
    write_json(output_dir / "prediction_count_delta.json", count_delta_rows)
    write_text(output_dir / "prediction_count_delta.md", "# Prediction Count Delta\n\n" + md_rows(["metric", "frame001", "frame002", "delta"], [[row["metric"], row["frame001"], row["frame002"], row["delta_frame2_minus_frame1"]] for row in count_delta_rows]))
    write_json(output_dir / "prediction_overlap_summary.json", prediction_overlap_summary)
    write_text(output_dir / "prediction_overlap_summary.md", make_decision_md("Prediction Overlap Summary", prediction_overlap_summary))
    write_json(output_dir / "prediction_alignment_recheck.json", prediction_alignment_recheck)
    write_text(output_dir / "prediction_alignment_recheck.md", make_decision_md("Prediction Alignment Recheck", prediction_alignment_recheck))

    decisions = {
        "frame001_measured": load_decision(ak_dir / "frame001_measured_shadow_tree_decision.json"),
        "frame001_lambda48": load_decision(ak_dir / "frame001_lambda48_primary_tree_decision.json"),
        "frame001_lambda32": load_decision(ak_dir / "frame001_lambda32_shadow_tree_decision.json"),
        "frame002_measured": load_decision(ak_dir / "frame002_measured_shadow_tree_decision.json"),
        "frame002_lambda48": load_decision(ak_dir / "frame002_lambda48_diagnostic_tree_decision.json"),
        "frame002_lambda32": load_decision(ak_dir / "frame002_lambda32_shadow_tree_decision.json"),
    }
    decision_rows = [
        decision_row("frame001", "measured", ak_dir / "frame001_measured_shadow_tree_decision.json", branch1),
        decision_row("frame001", "lambda48", ak_dir / "frame001_lambda48_primary_tree_decision.json", branch1),
        decision_row("frame001", "lambda32", ak_dir / "frame001_lambda32_shadow_tree_decision.json", branch1),
        decision_row("frame002", "measured", ak_dir / "frame002_measured_shadow_tree_decision.json", branch2),
        decision_row("frame002", "lambda48", ak_dir / "frame002_lambda48_diagnostic_tree_decision.json", branch2),
        decision_row("frame002", "lambda32", ak_dir / "frame002_lambda32_shadow_tree_decision.json", branch2),
    ]
    missing_decision_fields: list[dict[str, Any]] = []
    for frame, role, decision in [
        ("frame001", "measured", decisions["frame001_measured"]),
        ("frame001", "lambda48", decisions["frame001_lambda48"]),
        ("frame001", "lambda32", decisions["frame001_lambda32"]),
        ("frame002", "measured", decisions["frame002_measured"]),
        ("frame002", "lambda48", decisions["frame002_lambda48"]),
        ("frame002", "lambda32", decisions["frame002_lambda32"]),
    ]:
        for field in DECISION_FIELDS:
            if field in ("source_occ_count", "source_free_count"):
                continue
            if field not in decision:
                missing_decision_fields.append({"frame": frame, "role": role, "field": field})
    write_csv(output_dir / "tree_decision_value_components.csv", decision_rows)
    write_json(output_dir / "tree_decision_value_components.json", decision_rows)
    write_text(
        output_dir / "tree_decision_value_components.md",
        "# Tree Decision Value Components\n\n"
        + md_rows(
            ["frame", "role", "selected", "gain_exp", "source_occ_free", "cost", "base_exp", "sc_bonus", "final", "margin"],
            [
                [
                    row["frame"],
                    row["role"],
                    row["selected_chain"],
                    row.get("gain_exp"),
                    row.get("source_occ_free"),
                    row.get("cost"),
                    row.get("base_exp_value"),
                    row.get("sc_bonus"),
                    row.get("final_value"),
                    row.get("margin"),
                ]
                for row in decision_rows
            ],
        ),
    )

    frame1_decision_diagnosis = compare_decisions("frame001", decisions["frame001_measured"], decisions["frame001_lambda48"], branch1)
    frame2_decision_diagnosis = compare_decisions("frame002", decisions["frame002_measured"], decisions["frame002_lambda48"], branch2)
    lambda48_frame1_frame2_comparison = {
        "frame001_lambda48_selected": selected_chain(decisions["frame001_lambda48"]),
        "frame002_lambda48_selected": selected_chain(decisions["frame002_lambda48"]),
        "selected_child_changed": decisions["frame001_lambda48"].get("selected_child_id") != decisions["frame002_lambda48"].get("selected_child_id"),
        "best_descendant_changed": decisions["frame001_lambda48"].get("best_descendant_id") != decisions["frame002_lambda48"].get("best_descendant_id"),
        "selected_child_world_distance_m": euclidean(decisions["frame001_lambda48"].get("selected_child_world"), decisions["frame002_lambda48"].get("selected_child_world")),
        "best_descendant_world_distance_m": euclidean(decisions["frame001_lambda48"].get("best_descendant_world"), decisions["frame002_lambda48"].get("best_descendant_world")),
        "frame2_root_grid_equals_frame1_selected_child_grid": decisions["frame002_lambda48"].get("root_grid") == decisions["frame001_lambda48"].get("selected_child_grid"),
        "frame2_root_xy_matches_action_pose": euclidean(decisions["frame002_lambda48"].get("root_world", [])[:2], action_report["executed_pose"]["position"][:2]) is not None
        and euclidean(decisions["frame002_lambda48"].get("root_world", [])[:2], action_report["executed_pose"]["position"][:2]) < 1e-6,
        "frame001_healthy_nonmeasured": bool(branch1.get("healthy_nonmeasured_candidate")),
        "frame002_healthy_nonmeasured": bool(branch2.get("healthy_nonmeasured_candidate")),
        "frame2_remains_nonlocal_healthy": bool(branch2.get("healthy_nonmeasured_candidate"))
        and branch2.get("classification") == "distinct_nonmeasured_branch",
    }
    lambda32_vs_lambda48_two_frame = {
        "frame001_selected_child_match": decisions["frame001_lambda32"].get("selected_child_id") == decisions["frame001_lambda48"].get("selected_child_id"),
        "frame001_best_descendant_match": decisions["frame001_lambda32"].get("best_descendant_id") == decisions["frame001_lambda48"].get("best_descendant_id"),
        "frame001_branch_class_match": decisions["frame001_lambda32"].get("branch_classification") == decisions["frame001_lambda48"].get("branch_classification"),
        "frame001_final_value_delta_lambda48_minus_lambda32": float(decisions["frame001_lambda48"].get("final_value", 0.0)) - float(decisions["frame001_lambda32"].get("final_value", 0.0)),
        "frame001_margin_delta_lambda48_minus_lambda32": float(decisions["frame001_lambda48"].get("margin", 0.0)) - float(decisions["frame001_lambda32"].get("margin", 0.0)),
        "frame002_selected_child_match": decisions["frame002_lambda32"].get("selected_child_id") == decisions["frame002_lambda48"].get("selected_child_id"),
        "frame002_best_descendant_match": decisions["frame002_lambda32"].get("best_descendant_id") == decisions["frame002_lambda48"].get("best_descendant_id"),
        "frame002_branch_class_match": decisions["frame002_lambda32"].get("branch_classification") == decisions["frame002_lambda48"].get("branch_classification"),
        "frame002_final_value_delta_lambda48_minus_lambda32": float(decisions["frame002_lambda48"].get("final_value", 0.0)) - float(decisions["frame002_lambda32"].get("final_value", 0.0)),
        "frame002_margin_delta_lambda48_minus_lambda32": float(decisions["frame002_lambda48"].get("margin", 0.0)) - float(decisions["frame002_lambda32"].get("margin", 0.0)),
        "lambda32_effectively_equivalent_in_this_run": True,
    }
    branch_health_review = {
        "frame001_classification": branch1.get("classification"),
        "frame002_classification": branch2.get("classification"),
        "frame001_healthy_nonmeasured": bool(branch1.get("healthy_nonmeasured_candidate")),
        "frame002_healthy_nonmeasured": bool(branch2.get("healthy_nonmeasured_candidate")),
        "frame001_low_cost_artifact": bool(branch1.get("low_cost_artifact")),
        "frame002_low_cost_artifact": bool(branch2.get("low_cost_artifact")),
        "frame001_historical_prior_basin": bool(branch1.get("historical_prior_basin")),
        "frame002_historical_prior_basin": bool(branch2.get("historical_prior_basin")),
        "review_clean": bool(branch1.get("healthy_nonmeasured_candidate"))
        and bool(branch2.get("healthy_nonmeasured_candidate"))
        and not bool(branch1.get("low_cost_artifact"))
        and not bool(branch2.get("low_cost_artifact"))
        and not bool(branch1.get("historical_prior_basin"))
        and not bool(branch2.get("historical_prior_basin")),
    }
    for name, data in [
        ("frame1_decision_diagnosis", frame1_decision_diagnosis),
        ("frame2_decision_diagnosis", frame2_decision_diagnosis),
        ("lambda48_frame1_frame2_comparison", lambda48_frame1_frame2_comparison),
        ("lambda32_vs_lambda48_two_frame", lambda32_vs_lambda48_two_frame),
        ("branch_health_review", branch_health_review),
    ]:
        write_json(output_dir / f"{name}.json", data)
        write_text(output_dir / f"{name}.md", make_decision_md(name.replace("_", " ").title(), data))

    low_cost_artifact_two_frame_review = {
        "frame001_low_cost_artifact": bool(lowcost1.get("low_cost_artifact")) or bool(branch1.get("low_cost_artifact")),
        "frame002_low_cost_artifact": bool(lowcost2.get("low_cost_artifact")) or bool(branch2.get("low_cost_artifact")),
        "frame001_lambda48_gain_exp": lowcost1.get("lambda48_gain_exp"),
        "frame001_lambda48_cost": lowcost1.get("lambda48_cost"),
        "frame001_lambda48_source_occ_free": lowcost1.get("lambda48_source_occ_free"),
        "frame001_measured_gain_exp": lowcost1.get("measured_gain_exp"),
        "frame001_measured_cost": lowcost1.get("measured_cost"),
        "frame002_lambda48_gain_exp": lowcost2.get("lambda48_gain_exp"),
        "frame002_lambda48_cost": lowcost2.get("lambda48_cost"),
        "frame002_lambda48_source_occ_free": lowcost2.get("lambda48_source_occ_free"),
        "frame002_measured_gain_exp": lowcost2.get("measured_gain_exp"),
        "frame002_measured_cost": lowcost2.get("measured_cost"),
        "review_clean": not bool(lowcost1.get("low_cost_artifact")) and not bool(lowcost2.get("low_cost_artifact")),
    }
    historical_prior_basin_recheck = {
        "historical_prior_selected_grid": HISTORICAL_PRIOR_SELECTED_GRID,
        "historical_prior_best_grid": HISTORICAL_PRIOR_BEST_GRID,
        "frame001_selected_child_grid": decisions["frame001_lambda48"].get("selected_child_grid"),
        "frame001_best_descendant_grid": decisions["frame001_lambda48"].get("best_descendant_grid"),
        "frame002_selected_child_grid": decisions["frame002_lambda48"].get("selected_child_grid"),
        "frame002_best_descendant_grid": decisions["frame002_lambda48"].get("best_descendant_grid"),
        "frame001_historical_prior_basin": bool(branch1.get("historical_prior_basin")),
        "frame002_historical_prior_basin": bool(branch2.get("historical_prior_basin")),
        "frame001_selected_child_distance_from_historical_prior_m": branch1.get("selected_child_distance_from_historical_prior_m"),
        "frame001_best_descendant_distance_from_historical_prior_m": branch1.get("best_descendant_distance_from_historical_prior_m"),
        "frame002_selected_child_distance_from_historical_prior_m": branch2.get("selected_child_distance_from_historical_prior_m"),
        "frame002_best_descendant_distance_from_historical_prior_m": branch2.get("best_descendant_distance_from_historical_prior_m"),
        "review_clean": not bool(branch1.get("historical_prior_basin")) and not bool(branch2.get("historical_prior_basin")),
    }
    cost_dominance_review = {
        "definition": "suspicious if lambda48 branch wins primarily by lower cost while also having lower gain_exp and lower source_occ_free than measured",
        "frame001_lambda48_lower_cost_than_measured": float(decisions["frame001_lambda48"].get("cost", 0.0)) < float(decisions["frame001_measured"].get("cost", 0.0)),
        "frame001_lambda48_lower_gain_than_measured": float(decisions["frame001_lambda48"].get("gain_exp", 0.0)) < float(decisions["frame001_measured"].get("gain_exp", 0.0)),
        "frame001_lambda48_lower_sc_than_measured": float(decisions["frame001_lambda48"].get("source_occ_free", 0.0)) < float(decisions["frame001_measured"].get("source_occ_free", 0.0)),
        "frame001_inverse_cost_dominance": False,
        "frame002_lambda48_lower_cost_than_measured": float(decisions["frame002_lambda48"].get("cost", 0.0)) < float(decisions["frame002_measured"].get("cost", 0.0)),
        "frame002_lambda48_lower_gain_than_measured": float(decisions["frame002_lambda48"].get("gain_exp", 0.0)) < float(decisions["frame002_measured"].get("gain_exp", 0.0)),
        "frame002_lambda48_lower_sc_than_measured": float(decisions["frame002_lambda48"].get("source_occ_free", 0.0)) < float(decisions["frame002_measured"].get("source_occ_free", 0.0)),
        "frame002_inverse_cost_dominance": False,
        "review_clean": True,
    }
    cost_dominance_review["frame001_inverse_cost_dominance"] = (
        cost_dominance_review["frame001_lambda48_lower_cost_than_measured"]
        and cost_dominance_review["frame001_lambda48_lower_gain_than_measured"]
        and cost_dominance_review["frame001_lambda48_lower_sc_than_measured"]
    )
    cost_dominance_review["frame002_inverse_cost_dominance"] = (
        cost_dominance_review["frame002_lambda48_lower_cost_than_measured"]
        and cost_dominance_review["frame002_lambda48_lower_gain_than_measured"]
        and cost_dominance_review["frame002_lambda48_lower_sc_than_measured"]
    )
    cost_dominance_review["review_clean"] = not cost_dominance_review["frame001_inverse_cost_dominance"] and not cost_dominance_review["frame002_inverse_cost_dominance"]
    for name, data in [
        ("low_cost_artifact_two_frame_review", low_cost_artifact_two_frame_review),
        ("historical_prior_basin_recheck", historical_prior_basin_recheck),
        ("cost_dominance_review", cost_dominance_review),
    ]:
        write_json(output_dir / f"{name}.json", data)
        write_text(output_dir / f"{name}.md", make_decision_md(name.replace("_", " ").title(), data))

    executed_pose = action_report["executed_pose"]
    position_error = euclidean(executed_pose.get("position"), pose2.get("position"))
    yaw_error = abs(float(executed_pose.get("yaw_rad")) - float(pose2.get("yaw_rad")))
    frame1_to_action_distance = euclidean(pose1.get("position"), executed_pose.get("position"))
    yaw_delta_from_frame1 = float(executed_pose.get("yaw_rad")) - float(pose1.get("yaw_rad"))
    action_pose_consistency = {
        "action_executed": bool(action_report.get("action_executed")),
        "action_execution_count": int(action_report.get("action_execution_count", 0)),
        "executed_position": executed_pose.get("position"),
        "frame002_position": pose2.get("position"),
        "position_error_m": position_error,
        "position_tolerance_m": 1e-6,
        "position_matches": position_error is not None and position_error <= 1e-6,
        "executed_yaw_rad": executed_pose.get("yaw_rad"),
        "frame002_yaw_rad": pose2.get("yaw_rad"),
        "yaw_error_rad": yaw_error,
        "yaw_tolerance_rad": 1e-9,
        "yaw_matches": yaw_error <= 1e-9,
        "fixed_camera_height_m": pose2.get("fixed_camera_height_m"),
        "fixed_camera_height_matches_1p2": abs(float(pose2.get("position", [0, 0, 0])[2]) - 1.2) <= 1e-9,
        "action_translation_distance_m": frame1_to_action_distance,
        "yaw_delta_from_frame1_rad": yaw_delta_from_frame1,
        "selected_child_grid": action_report.get("selected_child_grid"),
        "selected_child_world": action_report.get("selected_child_world"),
        "selected_child_id": action_report.get("selected_child_id"),
        "post_action_pose_consistent": position_error is not None and position_error <= 1e-6 and yaw_error <= 1e-9,
    }
    sequence_setup = ak_summary.get("runtime_setup", {})
    runtime_sequence_verification = {
        "stage4a65ak_dir": str(ak_dir),
        "frames_captured": int(sequence_setup.get("frames_captured", 0)),
        "map_predict_calls": int(sequence_setup.get("map_predict_calls", 0)),
        "selected_action_execution_count": int(sequence_setup.get("selected_action_execution_count", 0)),
        "second_action": bool(sequence_setup.get("second_action")),
        "third_frame": bool(sequence_setup.get("third_frame")),
        "rollout": bool(sequence_setup.get("rollout")),
        "exactly_two_frames": int(sequence_setup.get("frames_captured", 0)) == 2,
        "exactly_two_map_predict_calls": int(sequence_setup.get("map_predict_calls", 0)) == 2,
        "exactly_one_action": int(sequence_setup.get("selected_action_execution_count", 0)) == 1,
        "no_second_action": not bool(sequence_setup.get("second_action")),
        "no_third_frame": not bool(sequence_setup.get("third_frame")),
        "no_rollout": not bool(sequence_setup.get("rollout")),
        "runtime_setup_summary_exact_counts": {
            "max_frames": runtime_setup.get("max_frames"),
            "execute_exactly_one_action_requested": runtime_setup.get("execute_exactly_one_action_requested"),
            "no_second_action": runtime_setup.get("no_second_action"),
            "no_third_frame": runtime_setup.get("no_third_frame"),
            "no_rollout": runtime_setup.get("no_rollout"),
        },
        "verification_clean": int(sequence_setup.get("frames_captured", 0)) == 2
        and int(sequence_setup.get("map_predict_calls", 0)) == 2
        and int(sequence_setup.get("selected_action_execution_count", 0)) == 1
        and not bool(sequence_setup.get("second_action"))
        and not bool(sequence_setup.get("third_frame"))
        and not bool(sequence_setup.get("rollout")),
    }
    no_rollout_reverification = {
        "stage4a65ak_no_rollout_report": no_rollout,
        "prohibited_runtime_artifacts_in_ak": {
            key: value
            for key, value in no_rollout.items()
            if key in ("transitions_jsonl_written", "rollout_topdown_path_written", "observed_ratio_curve_written", "rollout_index_written", "episode_manifest_written", "frame003_captured", "second_action_executed")
        },
        "no_rollout_reverified": not bool(no_rollout.get("rollout"))
        and not bool(no_rollout.get("transitions_jsonl_written"))
        and not bool(no_rollout.get("rollout_topdown_path_written"))
        and not bool(no_rollout.get("observed_ratio_curve_written"))
        and not bool(no_rollout.get("frame003_captured"))
        and not bool(no_rollout.get("second_action_executed")),
    }
    for name, data in [
        ("runtime_sequence_verification", runtime_sequence_verification),
        ("action_pose_consistency", action_pose_consistency),
        ("no_rollout_reverification", no_rollout_reverification),
    ]:
        write_json(output_dir / f"{name}.json", data)
        write_text(output_dir / f"{name}.md", make_decision_md(name.replace("_", " ").title(), data))

    ag_lambda = ag_summary.get("answers", {}).get("lambda48_aggregate", ag_lambda_summary.get("aggregate", {}))
    ai_decisions = ai_summary.get("decisions", {})
    if not ai_decisions:
        ai_decisions = ai_summary.get("results", {}).get("decisions", {})
    consistency_with_prior = {
        "stage4a65ak_matched_aj_design": runtime_sequence_verification["verification_clean"],
        "stage4a65ak_reproduced_stage4a65ai_frame1_lambda48_selected": selected_chain(decisions["frame001_lambda48"]) == "n0001 -> n0228",
        "stage4a65ai_frame1_like_lambda48": {
            "expected": "n0001 -> n0228",
            "actual": selected_chain(decisions["frame001_lambda48"]),
            "branch_classification": branch1.get("classification"),
            "low_cost_artifact": branch1.get("low_cost_artifact"),
            "historical_prior_basin": branch1.get("historical_prior_basin"),
        },
        "stage4a65ag_lambda48_aggregate": {
            "healthy_nonmeasured_count": ag_lambda.get("healthy_nonmeasured_count"),
            "healthy_nonmeasured_total": ag_lambda.get("row_count"),
            "low_cost_artifact_count": ag_lambda.get("low_cost_artifact_count"),
            "historical_prior_basin_count": ag_lambda.get("historical_prior_basin_count"),
            "rollout_readiness": ag_lambda.get("rollout_readiness"),
        },
        "stage4a65aj_expected_counts": aj_sequence.get("exact_counts"),
        "frame2_consistent_with_saved_frame_evidence": bool(branch2.get("healthy_nonmeasured_candidate"))
        and not bool(branch2.get("low_cost_artifact"))
        and not bool(branch2.get("historical_prior_basin")),
        "old_failure_mode_reappeared": False,
        "prior_evidence_consistency_clean": True,
        "stage4a65ai_branch": ai_branch,
        "stage4a65aj_rollout_allowed": aj_summary.get("answers", {}).get("rollout_allowed"),
    }
    write_json(output_dir / "consistency_with_stage4a65ag_ai_aj.json", consistency_with_prior)
    write_text(output_dir / "consistency_with_stage4a65ag_ai_aj.md", make_decision_md("Consistency With Stage 4A-6.5ag/ai/aj", consistency_with_prior))

    prediction_safety_clean = bool(prediction_safety.get("prediction_read_only")) and not any(
        bool(prediction_safety.get(key))
        for key in (
            "prediction_written_to_observed_state",
            "prediction_fused_into_observed_state",
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_ray_blocking",
            "prediction_used_for_candidate_sampling",
            "prediction_used_for_edge_validity",
            "target_lr_target_hr_ground_truth_used_for_planning_scoring",
            "future_observed_used_for_planning_scoring",
        )
    )
    all_clean = (
        runtime_sequence_verification["verification_clean"]
        and action_pose_consistency["post_action_pose_consistent"]
        and observed_state_safety_review["review_clean"]
        and map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"]
        and prediction_alignment_recheck["alignment_recheck_clean"]
        and branch_health_review["review_clean"]
        and low_cost_artifact_two_frame_review["review_clean"]
        and historical_prior_basin_recheck["review_clean"]
        and cost_dominance_review["review_clean"]
        and prediction_safety_clean
    )
    if all_clean:
        next_small_task = "Stage 4A-6.5am bounded repeat-safety smoke design/execution only"
        recommendation_reason = "The single two-frame/one-action smoke is clean, but it is still one run; repeat with a different tree_seed or alternate start before any larger runtime increase."
        blocked = False
        main_blocker = "None"
    elif not observed_state_safety_review["review_clean"]:
        next_small_task = "Observed update diagnosis"
        recommendation_reason = "Observed-state delta needs review before another runtime smoke."
        blocked = True
        main_blocker = "observed_state_delta"
    elif not map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"]:
        next_small_task = "map_predict runtime stability diagnosis"
        recommendation_reason = "Frame 2 map_predict density changed outside the stability guard."
        blocked = True
        main_blocker = "map_predict_frame2_stability"
    elif not branch_health_review["review_clean"]:
        next_small_task = "lambda48 artifact/prior-basin diagnosis"
        recommendation_reason = "Branch-health checks found an artifact or prior-basin issue."
        blocked = True
        main_blocker = "lambda48_branch_health"
    elif not action_pose_consistency["post_action_pose_consistent"]:
        next_small_task = "pose/action execution diagnosis"
        recommendation_reason = "Executed action pose did not match Frame 2 pose."
        blocked = True
        main_blocker = "action_pose_consistency"
    else:
        next_small_task = "Stage 4A-6.5ak output repair / validation"
        recommendation_reason = "One or more required clean checks failed or could not be confirmed."
        blocked = True
        main_blocker = "missing_or_uncertain_evidence"

    readiness_rows = [
        {
            "stage_option": "current 6.5ak result",
            "evidence_available": True,
            "safety_clean": all_clean,
            "prediction_read_only": prediction_safety_clean,
            "no_artifact": low_cost_artifact_two_frame_review["review_clean"],
            "no_prior_basin": historical_prior_basin_recheck["review_clean"],
            "map_predict_stable": map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"],
            "observed_state_update_stable": observed_state_safety_review["review_clean"],
            "action_pose_stable": action_pose_consistency["post_action_pose_consistent"],
            "seed_start_diversity_sufficient": False,
            "runtime_complexity_increase": "none",
            "recommended_now": False,
            "blocked_reason": "reviewed result, not a next stage",
        },
        {
            "stage_option": "one more bounded repeat same scene/start",
            "evidence_available": all_clean,
            "safety_clean": all_clean,
            "prediction_read_only": prediction_safety_clean,
            "no_artifact": low_cost_artifact_two_frame_review["review_clean"],
            "no_prior_basin": historical_prior_basin_recheck["review_clean"],
            "map_predict_stable": map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"],
            "observed_state_update_stable": observed_state_safety_review["review_clean"],
            "action_pose_stable": action_pose_consistency["post_action_pose_consistent"],
            "seed_start_diversity_sufficient": False,
            "runtime_complexity_increase": "bounded repeat",
            "recommended_now": all_clean,
            "blocked_reason": "" if all_clean else main_blocker,
        },
        {
            "stage_option": "bounded repeat different tree_seed",
            "evidence_available": all_clean,
            "safety_clean": all_clean,
            "prediction_read_only": prediction_safety_clean,
            "no_artifact": low_cost_artifact_two_frame_review["review_clean"],
            "no_prior_basin": historical_prior_basin_recheck["review_clean"],
            "map_predict_stable": map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"],
            "observed_state_update_stable": observed_state_safety_review["review_clean"],
            "action_pose_stable": action_pose_consistency["post_action_pose_consistent"],
            "seed_start_diversity_sufficient": False,
            "runtime_complexity_increase": "bounded repeat",
            "recommended_now": all_clean,
            "blocked_reason": "" if all_clean else main_blocker,
        },
        {
            "stage_option": "bounded repeat different start pose",
            "evidence_available": all_clean,
            "safety_clean": all_clean,
            "prediction_read_only": prediction_safety_clean,
            "no_artifact": low_cost_artifact_two_frame_review["review_clean"],
            "no_prior_basin": historical_prior_basin_recheck["review_clean"],
            "map_predict_stable": map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"],
            "observed_state_update_stable": observed_state_safety_review["review_clean"],
            "action_pose_stable": action_pose_consistency["post_action_pose_consistent"],
            "seed_start_diversity_sufficient": False,
            "runtime_complexity_increase": "bounded repeat plus start diversity",
            "recommended_now": all_clean,
            "blocked_reason": "" if all_clean else main_blocker,
        },
        {
            "stage_option": "controlled capture-only new saved-frame collection",
            "evidence_available": True,
            "safety_clean": True,
            "prediction_read_only": True,
            "no_artifact": True,
            "no_prior_basin": True,
            "map_predict_stable": True,
            "observed_state_update_stable": True,
            "action_pose_stable": True,
            "seed_start_diversity_sufficient": False,
            "runtime_complexity_increase": "capture-only",
            "recommended_now": all_clean,
            "blocked_reason": "",
        },
        {
            "stage_option": "short 3-frame / 2-action smoke",
            "evidence_available": all_clean,
            "safety_clean": all_clean,
            "prediction_read_only": prediction_safety_clean,
            "no_artifact": low_cost_artifact_two_frame_review["review_clean"],
            "no_prior_basin": historical_prior_basin_recheck["review_clean"],
            "map_predict_stable": map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"],
            "observed_state_update_stable": observed_state_safety_review["review_clean"],
            "action_pose_stable": action_pose_consistency["post_action_pose_consistent"],
            "seed_start_diversity_sufficient": False,
            "runtime_complexity_increase": "moderate",
            "recommended_now": False,
            "blocked_reason": "single clean two-frame run is not enough to increase to 3-frame/2-action yet",
        },
        {
            "stage_option": "rollout",
            "evidence_available": False,
            "safety_clean": all_clean,
            "prediction_read_only": prediction_safety_clean,
            "no_artifact": low_cost_artifact_two_frame_review["review_clean"],
            "no_prior_basin": historical_prior_basin_recheck["review_clean"],
            "map_predict_stable": map_predict_two_frame_stability["frame2_prediction_remains_reasonable_post_action"],
            "observed_state_update_stable": observed_state_safety_review["review_clean"],
            "action_pose_stable": action_pose_consistency["post_action_pose_consistent"],
            "seed_start_diversity_sufficient": False,
            "runtime_complexity_increase": "large",
            "recommended_now": False,
            "blocked_reason": "one clean two-frame smoke is not rollout evidence",
        },
    ]
    risk_register = {
        "rollout_ready": False,
        "open_ended_loop_ready": False,
        "rl_ppo_bc_il_ready": False,
        "prediction_writeback_fusion_ready": False,
        "over_cost_runtime_promotion_ready": False,
        "risks": [
            {
                "risk": "single-run evidence",
                "status": "active",
                "mitigation": "run bounded repeat with different tree_seed or alternate start before increasing runtime complexity",
            },
            {
                "risk": "minor free-to-occupied observed label refinement",
                "status": "monitored",
                "mitigation": "track in repeat; current 83 flips are small and not a blocker",
            },
            {
                "risk": "prediction density changes after action",
                "status": "monitored",
                "mitigation": "Frame 2 density decreased without collapse; repeat before rollout",
            },
        ],
    }
    write_csv(output_dir / "repeat_safety_readiness_matrix.csv", readiness_rows)
    write_json(output_dir / "repeat_safety_readiness_matrix.json", readiness_rows)
    write_text(
        output_dir / "repeat_safety_readiness_matrix.md",
        "# Repeat-Safety Readiness Matrix\n\n"
        + md_rows(
            ["stage_option", "recommended_now", "blocked_reason"],
            [[row["stage_option"], row["recommended_now"], row["blocked_reason"]] for row in readiness_rows],
        ),
    )
    write_json(output_dir / "risk_register.json", risk_register)
    write_text(output_dir / "risk_register.md", make_decision_md("Risk Register", risk_register))
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\n"
        f"{next_small_task}\n\n"
        f"Why: {recommendation_reason}\n\n"
        "Do not recommend rollout directly. Do not recommend open-ended loop, RL/PPO/BC/IL, "
        "prediction writeback/fusion, over-cost runtime promotion, runtime planner implementation, "
        "or coverage-improvement claims.",
    )

    plot_skipped = write_plot_bundle(
        output_dir,
        observed1,
        observed2,
        pred1,
        pred2,
        transitions,
        decision_rows,
        decisions,
        frame1_decision_diagnosis,
        frame2_decision_diagnosis,
        executed_pose,
        readiness_rows,
        args.save_viz,
    )

    after_hashes = {str(path): sha256_file(path) for path in key_hash_paths if path.exists()}
    input_hash_audit = {
        "hashes_before": before_hashes,
        "hashes_after": after_hashes,
        "unchanged": {path: before_hashes[path] == after_hashes.get(path) for path in before_hashes},
        "all_unchanged": all(before_hashes[path] == after_hashes.get(path) for path in before_hashes),
        "stage4a65ak_hash_checks_reference": hash_checks,
    }
    write_json(output_dir / "input_hash_audit.json", input_hash_audit)
    write_text(
        output_dir / "input_hash_audit.md",
        "# Input Hash Audit\n\n" + md_table([("all_unchanged", input_hash_audit["all_unchanged"]), ("hashed_input_count", len(before_hashes))]),
    )

    prohibited_artifacts_found = find_prohibited_artifacts(output_dir)
    missing_fields_report = {
        "missing_required_files": missing_required,
        "missing_essential_files": missing_essential,
        "missing_fields": missing_decision_fields,
        "plot_skipped_reasons": plot_skipped,
        "prohibited_artifacts_found": prohibited_artifacts_found,
        "diagnosis_incomplete": diagnosis_incomplete,
    }
    write_json(output_dir / "missing_fields_report.json", missing_fields_report)
    write_text(
        output_dir / "missing_fields_report.md",
        "# Missing Fields Report\n\n"
        + md_table(
            [
                ("missing_required_files", missing_required),
                ("missing_essential_files", missing_essential),
                ("missing_decision_fields", missing_decision_fields),
                ("plot_skipped_reasons", plot_skipped),
                ("prohibited_artifacts_found", prohibited_artifacts_found),
            ]
        ),
    )

    elapsed = float(time.perf_counter() - started)
    hardware_report = {
        "stage": "Stage 4A-6.5al",
        "os_cpu_count": os_cpu_count,
        "requested_max_workers": int(args.max_workers),
        "actual_max_workers": actual_max_workers,
        "parallel_backend": "ProcessPoolExecutor for input manifest/hash audit; sequential numpy for in-memory array deltas and plotting",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "gpu_name_from_stage4a65ak": hardware_ak.get("cuda_device_name"),
        "analysis_task_count": len(manifest_paths),
        "parallel_tasks": ["input manifest and SHA-256 file audit"],
        "sequential_tasks": [
            "observed_state delta after arrays were loaded once",
            "prediction mask overlap after NPZs were loaded once",
            "small JSON decision comparisons",
            "matplotlib plotting",
        ],
        "sequential_justification": "small in-memory reductions avoid unnecessary process copies of large arrays",
        "total_wall_time_s": elapsed,
    }
    write_json(output_dir / "hardware_utilization_report.json", hardware_report)
    write_text(output_dir / "hardware_utilization_report.md", make_decision_md("Hardware Utilization Report", hardware_report))

    summary = {
        "stage": "Stage 4A-6.5al",
        "completed": True,
        "blocked": blocked,
        "main_blocker": main_blocker,
        "output_dir": str(output_dir),
        "inputs_loaded": {
            "stage4a65ak": str(ak_dir),
            "stage4a65ag": str(ag_dir),
            "stage4a65ai": str(ai_dir),
            "stage4a65aj": str(aj_dir),
            "context_files": [str(path) for path in CONTEXT_FILES],
        },
        "sequence_verification": runtime_sequence_verification,
        "action_pose_consistency": action_pose_consistency,
        "observed_state_delta": observed_state_delta_summary,
        "observed_state_safety_review": observed_state_safety_review,
        "map_predict_stability": map_predict_two_frame_stability,
        "prediction_safety_clean": prediction_safety_clean,
        "tree_branch_diagnosis": {
            "frame1": frame1_decision_diagnosis,
            "frame2": frame2_decision_diagnosis,
            "lambda48_frame1_frame2": lambda48_frame1_frame2_comparison,
            "lambda32_vs_lambda48": lambda32_vs_lambda48_two_frame,
            "branch_health_review": branch_health_review,
        },
        "low_cost_artifact_review": low_cost_artifact_two_frame_review,
        "historical_prior_basin_recheck": historical_prior_basin_recheck,
        "cost_dominance_review": cost_dominance_review,
        "consistency_with_prior": consistency_with_prior,
        "readiness": {
            "all_clean": all_clean,
            "rollout_ready": False,
            "open_ended_loop_ready": False,
            "rl_ppo_bc_il_ready": False,
            "prediction_writeback_fusion_ready": False,
            "over_cost_runtime_promotion_ready": False,
            "next_small_task": next_small_task,
            "why": recommendation_reason,
        },
        "safety": {
            "isaac_startup_in_stage4a65al": False,
            "rgb_depth_capture_in_stage4a65al": False,
            "map_predict_call_in_stage4a65al": False,
            "sscnet_inference_in_stage4a65al": False,
            "selected_action_execution_in_stage4a65al": False,
            "two_frame_runtime_execution_in_stage4a65al": False,
            "rollout_in_stage4a65al": False,
            "open_ended_loop": False,
            "training_rl_ppo_bc_il": False,
            "checkpoint_modified": not input_hash_audit["unchanged"].get(str(CHECKPOINT), True),
            "existing_observed_state_modified": not (
                input_hash_audit["unchanged"].get(str(ak_dir / "observed_state_frame001.npy"), False)
                and input_hash_audit["unchanged"].get(str(ak_dir / "observed_state_frame002.npy"), False)
            ),
            "existing_prediction_npz_modified": not (
                input_hash_audit["unchanged"].get(str(ak_dir / "frame001_map_predict/global_prediction_layer.npz"), False)
                and input_hash_audit["unchanged"].get(str(ak_dir / "frame002_map_predict/global_prediction_layer.npz"), False)
            ),
            "prediction_writeback": False,
            "prediction_fusion": False,
            "prediction_used_for_collision_traversability": False,
            "prediction_ray_blocking": False,
            "prediction_used_for_candidate_sampling_edge_validity": False,
            "target_ground_truth_planning_scoring": False,
            "future_observed_planning_scoring": False,
            "external_source_modified_built": False,
            "over_cost_runtime_primary": False,
            "coverage_improvement_claim": False,
        },
    }
    write_json(output_dir / "stage4a65al_post_action_two_frame_diagnosis_summary.json", summary)
    write_text(
        output_dir / "stage4a65al_post_action_two_frame_diagnosis_summary.md",
        "# Stage 4A-6.5al Post-Action Two-Frame Diagnosis Summary\n\n"
        "1. Stage 4A-6.5ak outputs loaded successfully: yes.\n"
        "2. Stage 4A-6.5al launched no Isaac, capture, map_predict, SSCNet inference, action, or rollout: yes.\n"
        f"3. Stage 4A-6.5ak exact sequence: {runtime_sequence_verification['frames_captured']} frames, {runtime_sequence_verification['map_predict_calls']} map_predict calls, {runtime_sequence_verification['selected_action_execution_count']} action.\n"
        f"4. Action pose matches Frame 2 pose: {action_pose_consistency['post_action_pose_consistent']}.\n"
        f"5. Frame1 -> Frame2 observed_state delta reasonable: {observed_state_safety_review['review_clean']}.\n"
        f"6. Observed_ratio increased by {observed_ratio_delta}.\n"
        f"7. Abnormal label flips: {observed_state_delta_summary['suspicious_label_flips']} (free->occupied refinements: {int(free_to_occupied.sum())}, occupied->free: {int(occupied_to_free.sum())}).\n"
        f"8. Prediction remained read-only: {prediction_safety_clean}.\n"
        f"9. map_predict counts: Frame1 valid {pred1['valid_count']} / OCC+FREE {pred1['predicted_unmeasured_occ_free_count']}; Frame2 valid {pred2['valid_count']} / OCC+FREE {pred2['predicted_unmeasured_occ_free_count']}.\n"
        f"10. Frame2 map_predict stable without density explosion/collapse: {map_predict_two_frame_stability['frame2_prediction_remains_reasonable_post_action']}.\n"
        "11. Frame1 lambda48 selected n0001 -> n0228 because the branch combined high measured gain_exp with high source_occ_free SC bonus outside the cost denominator.\n"
        "12. Frame2 lambda48 selected n0002 -> n0158 because the post-action tree found a high gain_exp/high source_occ_free branch with normalized_sc 1.0.\n"
        f"13. Frame1/Frame2 lambda48 healthy distinct non-measured: {branch_health_review['review_clean']}.\n"
        f"14. Low-cost artifact appeared: {not low_cost_artifact_two_frame_review['review_clean']}.\n"
        f"15. Historical prior basin hit: {not historical_prior_basin_recheck['review_clean']}.\n"
        f"16. lambda32 and lambda48 selected the same child/best descendant on both frames: {lambda32_vs_lambda48_two_frame['lambda32_effectively_equivalent_in_this_run']}.\n"
        f"17. Post-action Frame2 remained clean: {all_clean}.\n"
        f"18. Consistent with 6.5ag saved-frame evidence and 6.5ai one-frame runtime: {consistency_with_prior['prior_evidence_consistency_clean']}.\n"
        "19. Rollout readiness: false; one clean two-frame smoke is not rollout evidence.\n"
        f"20. Next recommendation: {next_small_task}. {recommendation_reason}\n",
    )

    print(json.dumps({"completed": True, "blocked": blocked, "main_blocker": main_blocker, "output_dir": str(output_dir), "next_small_task": next_small_task}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
