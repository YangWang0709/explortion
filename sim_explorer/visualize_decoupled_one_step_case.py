#!/usr/bin/env python3
"""Stage 4A-6.5d spatial visualization for one decoupled SC step.

This script is intentionally offline and read-only for case inputs. It loads
saved observed_state, prediction npz, decisions, and candidates, then writes
topdown plots plus a compact spatial summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)

STATE_CMAP = ListedColormap(["#343a46", "#b9d6d8", "#c95d63"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)
STATE_TICKS = [int(UNKNOWN), int(FREE), int(OCCUPIED)]
STATE_TICKLABELS = ["unknown", "free", "occupied"]

BASELINE_COLOR = "#2563eb"
DECOUPLED_COLOR = "#d97706"
CURRENT_COLOR = "#111827"
MAX_GAIN_COLOR = "#7c3aed"
MAX_SC_COLOR = "#db2777"
MIN_COST_COLOR = "#059669"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def path_from_case(selected: dict[str, Any], key: str, fallback: Path | None = None) -> Path | None:
    value = selected.get(key)
    if value:
        return Path(value)
    runtime_args = selected.get("runtime_args") or {}
    value = runtime_args.get(key)
    if value:
        return Path(value)
    return fallback


def normalize_bounds(raw: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    raw = raw or {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]}
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def resolve_bounds(
    selected: dict[str, Any],
    comparison: dict[str, Any],
    baseline_decision: dict[str, Any] | None,
    episode_summary: dict[str, Any] | None,
) -> dict[str, tuple[float, float]]:
    if baseline_decision:
        diagnostics = baseline_decision.get("diagnostics") or {}
        if diagnostics.get("bounds"):
            return normalize_bounds(diagnostics["bounds"])
    if episode_summary and episode_summary.get("map_bounds"):
        return normalize_bounds(episode_summary["map_bounds"])
    if selected.get("map_bounds"):
        return normalize_bounds(selected["map_bounds"])
    baseline = comparison.get("baseline") or {}
    best = baseline.get("best_candidate") or {}
    if best.get("diagnostics", {}).get("bounds"):
        return normalize_bounds(best["diagnostics"]["bounds"])
    return normalize_bounds(None)


def voxel_size_from_bounds(bounds: dict[str, tuple[float, float]], observed_shape: tuple[int, int, int]) -> float:
    return float((bounds["x"][1] - bounds["x"][0]) / int(observed_shape[0]))


def grid_to_world_xy(
    xy: tuple[int, int] | list[int] | np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
) -> tuple[float, float]:
    arr = np.asarray(xy, dtype=np.float64)
    x = bounds["x"][0] + (arr[0] + 0.5) * voxel_size
    y = bounds["y"][0] + (arr[1] + 0.5) * voxel_size
    return float(x), float(y)


def project_topdown(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == FREE, axis=2)
    topdown = np.full(observed_state.shape[:2], UNKNOWN, dtype=np.int8)
    topdown[free] = FREE
    topdown[occupied] = OCCUPIED
    return topdown


def project_mask_topdown(mask: np.ndarray) -> np.ndarray:
    return np.any(np.asarray(mask, dtype=bool), axis=2)


def extent_from_bounds(bounds: dict[str, tuple[float, float]]) -> list[float]:
    return [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]


def draw_observed_background(
    ax: plt.Axes,
    topdown: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    title: str,
) -> None:
    ax.imshow(
        topdown.T,
        origin="lower",
        extent=extent_from_bounds(bounds),
        cmap=STATE_CMAP,
        norm=STATE_NORM,
        interpolation="nearest",
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)


def add_state_colorbar(fig: plt.Figure, ax: plt.Axes) -> None:
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=STATE_NORM, cmap=STATE_CMAP),
        ax=ax,
        ticks=STATE_TICKS,
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(STATE_TICKLABELS)


def draw_mask_overlay(
    ax: plt.Axes,
    mask2d: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    color: tuple[float, float, float],
    alpha: float,
) -> None:
    rgba = np.zeros((mask2d.shape[1], mask2d.shape[0], 4), dtype=np.float32)
    rgba[mask2d.T] = (float(color[0]), float(color[1]), float(color[2]), float(alpha))
    ax.imshow(rgba, origin="lower", extent=extent_from_bounds(bounds), interpolation="nearest")


def metric(record: dict[str, Any], names: tuple[str, ...], default: float | None = None) -> float | None:
    for name in names:
        if name in record and record[name] is not None:
            try:
                return float(record[name])
            except (TypeError, ValueError):
                continue
    return default


def grid_position(record: dict[str, Any]) -> list[int]:
    raw = record.get("grid_position") or record.get("candidate_grid") or []
    return [int(round(float(v))) for v in raw[:3]]


def world_position(record: dict[str, Any]) -> list[float]:
    raw = record.get("world_position") or record.get("candidate_world") or []
    return [float(v) for v in raw[:3]]


def candidate_key(record: dict[str, Any]) -> str:
    if record.get("key"):
        return str(record["key"])
    if record.get("candidate_key"):
        return str(record["candidate_key"])
    grid = grid_position(record)
    if grid:
        return f"grid:{grid[0]},{grid[1]},{grid[2]}"
    return ""


def candidate_by_key(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    for record in records:
        if candidate_key(record) == key:
            return record
    return None


def text_for_candidate(name: str, record: dict[str, Any]) -> str:
    score = metric(record, ("score", "final_score", "final_score_decoupled_sc"), 0.0)
    gain_exp = metric(record, ("gain_exp",), 0.0)
    effective_sc = metric(record, ("effective_gain_sc",), 0.0)
    cost = metric(record, ("path_cost",), 0.0)
    return (
        f"{name}\n"
        f"score {score:.3f}\n"
        f"gain_exp {gain_exp:.1f}, eff_sc {effective_sc:.3f}\n"
        f"path_cost {cost:.3f}"
    )


def draw_pose(ax: plt.Axes, pose: dict[str, Any], label: str = "current pose") -> None:
    world = pose.get("world") or pose.get("position") or []
    if len(world) < 2:
        return
    x, y = float(world[0]), float(world[1])
    yaw = float(pose.get("yaw", pose.get("yaw_rad", 0.0)))
    ax.scatter(x, y, s=160, c=CURRENT_COLOR, marker="^", edgecolors="white", linewidths=1.0, label=label, zorder=6)
    ax.arrow(
        x,
        y,
        0.28 * math.cos(yaw),
        0.28 * math.sin(yaw),
        width=0.012,
        head_width=0.07,
        head_length=0.09,
        color=CURRENT_COLOR,
        length_includes_head=True,
        zorder=7,
    )


def draw_candidate(
    ax: plt.Axes,
    record: dict[str, Any],
    label: str,
    color: str,
    marker: str,
    annotate: bool = True,
    offset: tuple[float, float] = (0.12, 0.12),
) -> None:
    world = world_position(record)
    if len(world) < 2:
        return
    ax.scatter(
        world[0],
        world[1],
        s=230,
        c=color,
        marker=marker,
        edgecolors="black",
        linewidths=0.8,
        label=label,
        zorder=8,
    )
    yaw = float(record.get("yaw", 0.0))
    ax.arrow(
        world[0],
        world[1],
        0.22 * math.cos(yaw),
        0.22 * math.sin(yaw),
        width=0.01,
        head_width=0.06,
        head_length=0.08,
        color=color,
        length_includes_head=True,
        zorder=9,
    )
    if annotate:
        ax.annotate(
            text_for_candidate(label, record),
            xy=(world[0], world[1]),
            xytext=(world[0] + offset[0], world[1] + offset[1]),
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": color, "alpha": 0.92},
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            zorder=10,
        )


def path_xy_to_world(
    path_xy: list[Any],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
) -> np.ndarray:
    if not path_xy:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([grid_to_world_xy(xy, bounds, voxel_size) for xy in path_xy], dtype=np.float32)


def draw_candidate_path(
    ax: plt.Axes,
    record: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    color: str,
    label: str,
) -> None:
    path = path_xy_to_world(record.get("astar_path_xy", []), bounds, voxel_size)
    if len(path):
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.1, alpha=0.95, label=label, zorder=5)


def value_array(records: list[dict[str, Any]], field_names: tuple[str, ...]) -> np.ndarray:
    values = []
    for record in records:
        value = metric(record, field_names, np.nan)
        values.append(float(value) if value is not None else np.nan)
    return np.asarray(values, dtype=np.float64)


def records_xy(records: list[dict[str, Any]]) -> np.ndarray:
    points = []
    for record in records:
        world = world_position(record)
        if len(world) >= 2:
            points.append([world[0], world[1]])
    return np.asarray(points, dtype=np.float64)


def best_by_field(records: list[dict[str, Any]], field: str, maximize: bool) -> dict[str, Any] | None:
    best_record = None
    best_value = -np.inf if maximize else np.inf
    for record in records:
        value = metric(record, (field,), None)
        if value is None or not np.isfinite(value):
            continue
        if (maximize and value > best_value) or ((not maximize) and value < best_value):
            best_value = value
            best_record = record
    return best_record


def draw_special_record(
    ax: plt.Axes,
    record: dict[str, Any] | None,
    label: str,
    marker: str,
    color: str,
    size: int = 145,
) -> None:
    if record is None:
        return
    world = world_position(record)
    if len(world) < 2:
        return
    ax.scatter(
        world[0],
        world[1],
        s=size,
        marker=marker,
        facecolors="none" if marker in {"s", "D"} else color,
        edgecolors=color,
        linewidths=2.0,
        label=label,
        zorder=9,
    )


def missing_candidate_fields(records: list[dict[str, Any]], fields: list[str]) -> dict[str, int]:
    return {field: int(sum(1 for record in records if field not in record or record[field] is None)) for field in fields}


def load_prediction(prediction_path: Path | None, observed_shape: tuple[int, int, int]) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    missing: list[str] = []
    if prediction_path is None or not prediction_path.exists():
        return None, ["prediction_npz"]
    data = np.load(prediction_path, mmap_mode="r")
    required = ["global_prediction_valid", "global_confidence", "global_occupied_prob"]
    for key in required:
        if key not in data.files:
            missing.append(key)
    if missing:
        return None, missing
    pred = {
        "valid": np.asarray(data["global_prediction_valid"], dtype=bool),
        "confidence": np.asarray(data["global_confidence"], dtype=np.float32),
        "occupied_prob": np.asarray(data["global_occupied_prob"], dtype=np.float32),
    }
    for key, array in pred.items():
        if array.shape != observed_shape:
            raise ValueError(f"Prediction {key} shape {array.shape} != observed_state shape {observed_shape}")
    return pred, []


def save_observed_baseline_decoupled(
    path: Path,
    topdown: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    current_pose: dict[str, Any],
    baseline: dict[str, Any],
    decoupled: dict[str, Any],
    baseline_decision_best: dict[str, Any],
    decoupled_decision_best: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.2), constrained_layout=True)
    draw_observed_background(ax, topdown, bounds, "Observed map with baseline vs decoupled one-step choices")
    draw_candidate_path(ax, baseline_decision_best, bounds, voxel_size, BASELINE_COLOR, "baseline A* path")
    draw_candidate_path(ax, decoupled_decision_best, bounds, voxel_size, DECOUPLED_COLOR, "decoupled A* path")
    draw_pose(ax, current_pose)
    draw_candidate(ax, baseline, "baseline", BASELINE_COLOR, "*", offset=(0.12, -0.45))
    draw_candidate(ax, decoupled, "decoupled", DECOUPLED_COLOR, "P", offset=(0.16, 0.16))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    add_state_colorbar(fig, ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_prediction_overlay(
    path: Path,
    topdown: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    current_pose: dict[str, Any],
    baseline: dict[str, Any],
    decoupled: dict[str, Any],
    prediction: dict[str, np.ndarray] | None,
    sc_conf_threshold: float,
    sc_occ_threshold: float,
) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(9.5, 8.2), constrained_layout=True)
    draw_observed_background(ax, topdown, bounds, "Prediction overlay on observed map")
    pred_stats: dict[str, Any] = {"prediction_loaded": prediction is not None}
    if prediction is not None:
        valid = prediction["valid"]
        confidence = prediction["confidence"]
        occupied_prob = prediction["occupied_prob"]
        valid2d = project_mask_topdown(valid)
        selected_occupied = valid & (confidence >= float(sc_conf_threshold)) & (occupied_prob >= float(sc_occ_threshold))
        occupied2d = project_mask_topdown(selected_occupied)
        draw_mask_overlay(ax, valid2d, bounds, (0.17, 0.38, 0.95), 0.28)
        draw_mask_overlay(ax, occupied2d, bounds, (0.95, 0.08, 0.12), 0.58)
        ax.scatter([], [], c="#2b60f0", alpha=0.45, s=55, label="prediction valid")
        ax.scatter(
            [],
            [],
            c="#f0141d",
            alpha=0.70,
            s=55,
            label=f"pred occ p>={sc_occ_threshold:g}, conf>={sc_conf_threshold:g}",
        )
        pred_stats.update(
            {
                "valid_voxel_count": int(np.count_nonzero(valid)),
                "valid_topdown_cell_count": int(np.count_nonzero(valid2d)),
                "selected_predicted_occupied_voxel_count": int(np.count_nonzero(selected_occupied)),
                "selected_predicted_occupied_topdown_cell_count": int(np.count_nonzero(occupied2d)),
            }
        )
    else:
        ax.text(
            0.5,
            0.5,
            "prediction npz unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#777777", "alpha": 0.9},
        )
    draw_pose(ax, current_pose)
    draw_candidate(ax, baseline, "baseline", BASELINE_COLOR, "*", annotate=False)
    draw_candidate(ax, decoupled, "decoupled", DECOUPLED_COLOR, "P", annotate=False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    add_state_colorbar(fig, ax)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return pred_stats


def scatter_candidates_on_axis(
    ax: plt.Axes,
    topdown: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    records: list[dict[str, Any]],
    score_fields: tuple[str, ...],
    title: str,
    baseline_best: dict[str, Any],
    decoupled_best: dict[str, Any],
    component_records: list[dict[str, Any]],
) -> None:
    draw_observed_background(ax, topdown, bounds, title)
    xy = records_xy(records)
    values = value_array(records, score_fields)
    finite_values = values[np.isfinite(values)]
    if len(xy) and len(finite_values):
        norm = Normalize(vmin=float(np.nanmin(finite_values)), vmax=float(np.nanmax(finite_values)))
        sizes = 35 + 80 * (values - norm.vmin) / max(norm.vmax - norm.vmin, 1e-6)
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=values,
            s=np.clip(sizes, 35, 115),
            cmap="viridis",
            norm=norm,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
            label="candidates",
            zorder=6,
        )
        plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="final_score")
    elif len(xy):
        ax.scatter(xy[:, 0], xy[:, 1], c="#4b5563", s=45, alpha=0.78, label="candidates", zorder=6)

    draw_candidate(ax, baseline_best, "baseline top-1", BASELINE_COLOR, "*", annotate=False)
    draw_candidate(ax, decoupled_best, "decoupled top-1", DECOUPLED_COLOR, "P", annotate=False)
    draw_special_record(
        ax,
        best_by_field(component_records, "gain_exp", maximize=True),
        "max gain_exp",
        "D",
        MAX_GAIN_COLOR,
    )
    draw_special_record(
        ax,
        best_by_field(component_records, "effective_gain_sc", maximize=True),
        "max effective_gain_sc",
        "s",
        MAX_SC_COLOR,
    )
    draw_special_record(
        ax,
        best_by_field(component_records, "path_cost", maximize=False),
        "min path_cost",
        "X",
        MIN_COST_COLOR,
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.92)


def save_candidate_score_components(
    path: Path,
    topdown: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    baseline_records: list[dict[str, Any]],
    decoupled_records: list[dict[str, Any]],
    baseline_best: dict[str, Any],
    decoupled_best: dict[str, Any],
) -> dict[str, Any]:
    source_records = decoupled_records or baseline_records
    fields = ["world_position", "grid_position", "final_score", "gain_exp", "effective_gain_sc", "path_cost"]
    missing = {
        "baseline": missing_candidate_fields(baseline_records, fields) if baseline_records else {"candidate_jsonl": 1},
        "decoupled": missing_candidate_fields(decoupled_records, fields) if decoupled_records else {"candidate_jsonl": 1},
    }

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.7), constrained_layout=True)
    scatter_candidates_on_axis(
        axes[0],
        topdown,
        bounds,
        baseline_records,
        ("final_score", "score"),
        "Baseline candidates colored by final_score",
        baseline_best,
        decoupled_best,
        source_records,
    )
    scatter_candidates_on_axis(
        axes[1],
        topdown,
        bounds,
        decoupled_records,
        ("final_score", "final_score_decoupled_sc", "score"),
        "Decoupled candidates colored by final_score",
        baseline_best,
        decoupled_best,
        source_records,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return {
        "baseline_candidate_count": len(baseline_records),
        "decoupled_candidate_count": len(decoupled_records),
        "missing_fields": missing,
        "max_gain_exp_key": candidate_key(best_by_field(source_records, "gain_exp", maximize=True) or {}),
        "max_effective_gain_sc_key": candidate_key(best_by_field(source_records, "effective_gain_sc", maximize=True) or {}),
        "min_path_cost_key": candidate_key(best_by_field(source_records, "path_cost", maximize=False) or {}),
    }


def save_local_zoom(
    path: Path,
    topdown: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    current_pose: dict[str, Any],
    baseline: dict[str, Any],
    decoupled: dict[str, Any],
    baseline_decision_best: dict[str, Any],
    decoupled_decision_best: dict[str, Any],
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.5), constrained_layout=True)
    draw_observed_background(ax, topdown, bounds, "Local zoom around current pose and selected candidates")
    draw_candidate_path(ax, baseline_decision_best, bounds, voxel_size, BASELINE_COLOR, "baseline A* path")
    draw_candidate_path(ax, decoupled_decision_best, bounds, voxel_size, DECOUPLED_COLOR, "decoupled A* path")
    draw_pose(ax, current_pose)
    draw_candidate(ax, baseline, "baseline", BASELINE_COLOR, "*", offset=(0.12, -0.34))
    draw_candidate(ax, decoupled, "decoupled", DECOUPLED_COLOR, "P", offset=(0.12, 0.14))

    points = []
    for record in (baseline, decoupled):
        world = world_position(record)
        if len(world) >= 2:
            points.append(world[:2])
    pose_world = current_pose.get("world") or current_pose.get("position") or []
    if len(pose_world) >= 2:
        points.append([float(pose_world[0]), float(pose_world[1])])
    if points:
        pts = np.asarray(points, dtype=np.float64)
        margin = max(0.75, 5.0 * voxel_size)
        ax.set_xlim(float(np.min(pts[:, 0]) - margin), float(np.max(pts[:, 0]) + margin))
        ax.set_ylim(float(np.min(pts[:, 1]) - margin), float(np.max(pts[:, 1]) + margin))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    add_state_colorbar(fig, ax)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def displacement_summary(baseline: dict[str, Any], decoupled: dict[str, Any]) -> dict[str, Any]:
    b_grid = np.asarray(grid_position(baseline), dtype=np.float64)
    d_grid = np.asarray(grid_position(decoupled), dtype=np.float64)
    b_world = np.asarray(world_position(baseline), dtype=np.float64)
    d_world = np.asarray(world_position(decoupled), dtype=np.float64)
    cell_delta = d_grid - b_grid
    world_delta = d_world - b_world
    return {
        "delta_grid": [int(v) for v in cell_delta.tolist()],
        "distance_cells": float(np.linalg.norm(cell_delta)),
        "delta_world_m": [float(v) for v in world_delta.tolist()],
        "distance_m": float(np.linalg.norm(world_delta)),
    }


def qualitative_conclusion(
    displacement: dict[str, Any],
    baseline: dict[str, Any],
    decoupled: dict[str, Any],
) -> dict[str, Any]:
    distance_m = float(displacement["distance_m"])
    higher_gain = bool(metric(decoupled, ("gain_exp",), 0.0) > metric(baseline, ("gain_exp",), 0.0))
    higher_effective_sc = bool(
        metric(decoupled, ("effective_gain_sc",), 0.0) > metric(baseline, ("effective_gain_sc",), 0.0)
    )
    higher_cost = bool(metric(decoupled, ("path_cost",), 0.0) > metric(baseline, ("path_cost",), 0.0))
    if distance_m < 0.35:
        spatial = "distinct logged candidates, but only an adjacent local shift"
        meaningful = False
        next_task = "A. candidate generation/path-level utility diagnosis"
    elif distance_m < 0.8:
        spatial = "local but visibly different viewpoint"
        meaningful = True
        next_task = "B. one-step comparison of a few score formulas"
    else:
        spatial = "spatially meaningfully different viewpoint"
        meaningful = True
        next_task = "B. one-step comparison of a few score formulas"
    conclusion = (
        f"The decoupled top-1 is {distance_m:.3f} m from the baseline top-1. "
        f"It has higher gain_exp={higher_gain}, higher effective_gain_sc={higher_effective_sc}, "
        f"and higher path_cost={higher_cost}. This makes it plausible as a one-step formula "
        "smoke, but it is not enough to justify rollout."
    )
    return {
        "are_candidates_spatially_distinct": bool(distance_m > 0.0),
        "is_spatially_meaningful": meaningful,
        "qualitative_difference": spatial,
        "decoupled_moved_toward_higher_gain": higher_gain or higher_effective_sc,
        "is_plausible_for_future_one_step_formula_comparison": True,
        "is_enough_to_justify_rollout": False,
        "recommended_next_small_task": next_task,
        "conclusion": conclusion,
    }


def build_md(summary: dict[str, Any]) -> str:
    b = summary["baseline"]
    d = summary["decoupled"]
    disp = summary["displacement"]
    interp = summary["interpretation"]
    safety = summary["safety"]
    plots = summary["generated_files"]["plots"]
    lines = [
        "# Stage 4A-6.5d Spatial Summary",
        "",
        "## Inputs",
        f"- selected case: `{summary['inputs']['selected_case']}`",
        f"- observed_state: `{summary['inputs']['observed_state']}`",
        f"- prediction: `{summary['inputs']['prediction_npz']}`",
        f"- baseline output: `{summary['inputs']['baseline_output_dir']}`",
        f"- decoupled output: `{summary['inputs']['decoupled_output_dir']}`",
        "",
        "## Spatial Comparison",
        f"- baseline grid/world: `{b['grid']}` / `{b['world']}`",
        f"- decoupled grid/world: `{d['grid']}` / `{d['world']}`",
        f"- displacement cells: `{disp['delta_grid']}`, distance `{disp['distance_cells']:.3f}` cells",
        f"- displacement meters: `{disp['delta_world_m']}`, distance `{disp['distance_m']:.3f}` m",
        f"- baseline score/gain/path_cost: `{b['score']:.6f}` / `{b['gain_exp']:.6f}` / `{b['path_cost']:.6f}`",
        f"- decoupled score/gain/path_cost: `{d['score']:.6f}` / `{d['gain_exp']:.6f}` / `{d['path_cost']:.6f}`",
        f"- higher gain_exp: `{summary['comparisons']['decoupled_has_higher_gain_exp']}`",
        f"- higher effective_gain_sc: `{summary['comparisons']['decoupled_has_higher_effective_gain_sc']}`",
        f"- higher path_cost: `{summary['comparisons']['decoupled_has_higher_path_cost']}`",
        f"- qualitative difference: {interp['qualitative_difference']}",
        "",
        "## Answers",
        f"- Are the two candidates spatially distinct? `{interp['are_candidates_spatially_distinct']}`.",
        "- Did decoupled move toward higher gain or just tiny local jitter? "
        f"`{interp['decoupled_moved_toward_higher_gain']}`, but the displacement is local.",
        "- Plausible enough for future one-step formula comparison? "
        f"`{interp['is_plausible_for_future_one_step_formula_comparison']}`.",
        f"- Enough to justify rollout? `{interp['is_enough_to_justify_rollout']}`.",
        f"- Recommended next small task: {interp['recommended_next_small_task']}.",
        "",
        "## Interpretation",
        f"- {interp['conclusion']}",
        "",
        "## Files",
        *[f"- `{plot}`" for plot in plots],
        f"- `{summary['generated_files']['summary_json']}`",
        f"- `{summary['generated_files']['summary_md']}`",
        "",
        "## Safety",
        f"- Isaac startup: `{safety['isaac_startup']}`",
        f"- rollout: `{safety['rollout']}`",
        f"- map_predict rerun: `{safety['map_predict_rerun']}`",
        f"- training/RL: `{safety['training_rl_bc_il']}`",
        f"- checkpoint modified: `{safety['checkpoint_modified']}`",
        f"- observed_state modified: `{safety['observed_state_modified']}`",
        f"- prediction writeback: `{safety['prediction_writeback']}`",
        f"- leakage: `{safety['leakage']}`",
    ]
    return "\n".join(lines) + "\n"


def best_candidate_pair(
    comparison: dict[str, Any],
    baseline_decision: dict[str, Any],
    decoupled_decision: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = ((comparison.get("baseline") or {}).get("best_candidate") or {}).copy()
    decoupled = ((comparison.get("decoupled_sc") or {}).get("best_candidate") or {}).copy()
    if not baseline:
        baseline = dict(baseline_decision.get("best_candidate") or {})
    if not decoupled:
        decoupled = dict(decoupled_decision.get("best_candidate") or {})
    return baseline, decoupled


def enrich_best_from_candidates(best: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    key = candidate_key(best)
    record = candidate_by_key(records, key) if key else None
    if record is None:
        return best
    enriched = dict(record)
    enriched.update(best)
    return enriched


def compact_candidate(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": candidate_key(record),
        "grid": grid_position(record),
        "world": world_position(record),
        "score": float(metric(record, ("score", "final_score", "final_score_decoupled_sc"), 0.0) or 0.0),
        "gain_exp": float(metric(record, ("gain_exp",), 0.0) or 0.0),
        "raw_gain_sc": float(metric(record, ("raw_gain_sc", "gain_sc"), 0.0) or 0.0),
        "effective_gain_sc": float(metric(record, ("effective_gain_sc",), 0.0) or 0.0),
        "path_cost": float(metric(record, ("path_cost",), 0.0) or 0.0),
    }


def ensure_no_rollout_files(output_dir: Path) -> bool:
    forbidden_names = {"transitions.jsonl", "episode_summary.json", "observed_state_final.npy"}
    forbidden_suffixes = (".npz", ".npy")
    for path in output_dir.iterdir():
        if path.name in forbidden_names:
            return False
        if path.name.startswith("step_") and path.suffix in forbidden_suffixes:
            return False
        if path.name.startswith("observed_state") and path.suffix == ".npy":
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = case_dir / "selected_case.json"
    comparison_path = case_dir / "one_step_comparison.json"
    selected = load_json(selected_path)
    comparison = load_json(comparison_path)

    baseline_dir = Path((comparison.get("baseline") or {}).get("output_dir") or case_dir / "baseline_runtime")
    decoupled_dir = Path(
        (comparison.get("decoupled_sc") or {}).get("output_dir") or case_dir / "decoupled_sc_lambda0p5"
    )
    baseline_decision_path = baseline_dir / "expert_step_decision.json"
    decoupled_decision_path = decoupled_dir / "expert_step_decision.json"
    baseline_candidates_path = baseline_dir / "expert_step_candidates.jsonl"
    decoupled_candidates_path = decoupled_dir / "expert_step_candidates.jsonl"

    baseline_decision = load_json(baseline_decision_path) if baseline_decision_path.exists() else {}
    decoupled_decision = load_json(decoupled_decision_path) if decoupled_decision_path.exists() else {}
    baseline_records = load_jsonl(baseline_candidates_path)
    decoupled_records = load_jsonl(decoupled_candidates_path)

    observed_path = path_from_case(selected, "observed_state")
    if observed_path is None or not observed_path.exists():
        raise FileNotFoundError(f"Missing observed_state from selected_case.json: {observed_path}")
    observed_sha_before = sha256_file(observed_path)
    observed_state = np.load(observed_path, mmap_mode="r")
    observed_read_only = bool(not observed_state.flags.writeable)
    if observed_state.ndim != 3:
        raise ValueError(f"observed_state must be 3D, got {observed_state.shape}")
    if not np.all(np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED])):
        raise ValueError(f"observed_state has unsupported values: {np.unique(observed_state).tolist()}")

    episode_summary_path = path_from_case(selected, "episode_summary")
    episode_summary = load_json(episode_summary_path) if episode_summary_path and episode_summary_path.exists() else None
    bounds = resolve_bounds(selected, comparison, baseline_decision, episode_summary)
    voxel_size = voxel_size_from_bounds(bounds, tuple(int(v) for v in observed_state.shape))
    topdown = project_topdown(observed_state)

    prediction_path = path_from_case(selected, "prediction_npz")
    prediction, missing_prediction_fields = load_prediction(prediction_path, tuple(int(v) for v in observed_state.shape))

    pose_path = path_from_case(selected, "pose", path_from_case(selected, "pose_json"))
    pose_json = load_json(pose_path) if pose_path and pose_path.exists() else {}
    current_pose = (baseline_decision.get("current_pose") or decoupled_decision.get("current_pose") or {}).copy()
    if not current_pose and pose_json:
        current_pose = {
            "world": pose_json.get("position", []),
            "yaw": pose_json.get("yaw_rad", pose_json.get("yaw", 0.0)),
        }

    baseline_best, decoupled_best = best_candidate_pair(comparison, baseline_decision, decoupled_decision)
    baseline_best = enrich_best_from_candidates(baseline_best, baseline_records)
    decoupled_best = enrich_best_from_candidates(decoupled_best, decoupled_records)
    baseline_decision_best = baseline_decision.get("best_candidate") or baseline_best
    decoupled_decision_best = decoupled_decision.get("best_candidate") or decoupled_best

    sc_thresholds = selected.get("thresholds") or {}
    runtime_args = selected.get("runtime_args") or {}
    sc_conf_threshold = float(
        sc_thresholds.get("sc_conf_threshold", runtime_args.get("sc_conf_threshold", selected.get("sc_conf_threshold", 0.5)))
    )
    sc_occ_threshold = float(
        sc_thresholds.get("sc_occ_threshold", runtime_args.get("sc_occ_threshold", selected.get("sc_occ_threshold", 0.7)))
    )

    observed_plot = output_dir / "observed_baseline_decoupled_topdown.png"
    prediction_plot = output_dir / "prediction_overlay_topdown.png"
    candidate_plot = output_dir / "candidate_score_components_topdown.png"
    zoom_plot = output_dir / "baseline_vs_decoupled_local_zoom.png"
    summary_json_path = output_dir / "stage4a65d_spatial_summary.json"
    summary_md_path = output_dir / "stage4a65d_spatial_summary.md"

    save_observed_baseline_decoupled(
        observed_plot,
        topdown,
        bounds,
        voxel_size,
        current_pose,
        baseline_best,
        decoupled_best,
        baseline_decision_best,
        decoupled_decision_best,
    )
    prediction_stats = save_prediction_overlay(
        prediction_plot,
        topdown,
        bounds,
        current_pose,
        baseline_best,
        decoupled_best,
        prediction,
        sc_conf_threshold,
        sc_occ_threshold,
    )
    candidate_stats = save_candidate_score_components(
        candidate_plot,
        topdown,
        bounds,
        baseline_records,
        decoupled_records,
        baseline_best,
        decoupled_best,
    )
    save_local_zoom(
        zoom_plot,
        topdown,
        bounds,
        voxel_size,
        current_pose,
        baseline_best,
        decoupled_best,
        baseline_decision_best,
        decoupled_decision_best,
    )

    observed_sha_after = sha256_file(observed_path)
    baseline_compact = compact_candidate(baseline_best)
    decoupled_compact = compact_candidate(decoupled_best)
    displacement = displacement_summary(baseline_best, decoupled_best)
    interpretation = qualitative_conclusion(displacement, baseline_best, decoupled_best)

    pngs = sorted(str(path) for path in output_dir.glob("*.png"))
    validation = {
        "output_dir_exists": output_dir.exists(),
        "png_count": len(pngs),
        "at_least_two_png_files_exist": len(pngs) >= 2,
        "summary_json_exists": False,
        "summary_md_exists": False,
        "observed_state_loaded_read_only": observed_read_only,
        "observed_state_hash_unchanged": observed_sha_before == observed_sha_after,
        "no_rollout_files_created": ensure_no_rollout_files(output_dir),
    }

    summary = {
        "stage": "Stage 4A-6.5d decoupled SC one-step spatial visualization",
        "selected_config": selected.get("config"),
        "step": int(selected.get("step", runtime_args.get("step", 1))),
        "formula": selected.get("formula", (selected.get("decoupled_sc") or {}).get("score_gain_mode")),
        "lambda": selected.get("lambda", selected.get("sc_gain_weight")),
        "inputs": {
            "selected_case": str(selected_path),
            "one_step_comparison": str(comparison_path),
            "observed_state": str(observed_path),
            "pose": str(pose_path) if pose_path else None,
            "camera_info": str(path_from_case(selected, "camera_info")),
            "episode_summary": str(episode_summary_path) if episode_summary_path else None,
            "prediction_npz": str(prediction_path) if prediction_path else None,
            "baseline_output_dir": str(baseline_dir),
            "decoupled_output_dir": str(decoupled_dir),
            "baseline_candidates": str(baseline_candidates_path),
            "decoupled_candidates": str(decoupled_candidates_path),
        },
        "map": {
            "bounds": bounds,
            "voxel_size": voxel_size,
            "observed_shape": list(observed_state.shape),
            "observed_state_sha256_before": observed_sha_before,
            "observed_state_sha256_after": observed_sha_after,
            "observed_state_loaded_read_only": observed_read_only,
            "unknown_count": int(np.count_nonzero(observed_state == UNKNOWN)),
            "free_count": int(np.count_nonzero(observed_state == FREE)),
            "occupied_count": int(np.count_nonzero(observed_state == OCCUPIED)),
        },
        "prediction": {
            **prediction_stats,
            "missing_fields": missing_prediction_fields,
            "sc_conf_threshold": sc_conf_threshold,
            "sc_occ_threshold": sc_occ_threshold,
        },
        "candidate_logging": candidate_stats,
        "current_pose": current_pose,
        "baseline": baseline_compact,
        "decoupled": decoupled_compact,
        "displacement": displacement,
        "comparisons": {
            "decoupled_has_higher_gain_exp": decoupled_compact["gain_exp"] > baseline_compact["gain_exp"],
            "decoupled_has_higher_effective_gain_sc": decoupled_compact["effective_gain_sc"]
            > baseline_compact["effective_gain_sc"],
            "decoupled_has_higher_path_cost": decoupled_compact["path_cost"] > baseline_compact["path_cost"],
        },
        "interpretation": interpretation,
        "generated_files": {
            "output_dir": str(output_dir),
            "plots": pngs,
            "summary_json": str(summary_json_path),
            "summary_md": str(summary_md_path),
        },
        "validation": validation,
        "safety": {
            "isaac_startup": False,
            "rollout": False,
            "map_predict_rerun": False,
            "sscnet_training": False,
            "training_rl_bc_il": False,
            "checkpoint_modified": False,
            "observed_state_modified": observed_sha_before != observed_sha_after,
            "prediction_writeback": False,
            "prediction_used_for_astar_traversability_collision_or_ray_blocking": False,
            "future_observations_used_for_planning": False,
            "target_lr_target_hr_ground_truth_used_for_scoring": False,
            "leakage": "none: offline read-only visualization; no scoring, planning, writeback, target, or ground-truth use",
        },
    }

    save_json(summary_json_path, summary)
    summary_md_path.write_text(build_md(summary), encoding="utf-8")

    pngs_after = sorted(str(path) for path in output_dir.glob("*.png"))
    summary["generated_files"]["plots"] = pngs_after
    summary["validation"] = {
        "output_dir_exists": output_dir.exists(),
        "png_count": len(pngs_after),
        "at_least_two_png_files_exist": len(pngs_after) >= 2,
        "summary_json_exists": summary_json_path.exists(),
        "summary_md_exists": summary_md_path.exists(),
        "observed_state_loaded_read_only": observed_read_only,
        "observed_state_hash_unchanged": sha256_file(observed_path) == observed_sha_before,
        "no_rollout_files_created": ensure_no_rollout_files(output_dir),
    }
    save_json(summary_json_path, summary)
    summary_md_path.write_text(build_md(summary), encoding="utf-8")

    if len(pngs_after) < 2:
        raise RuntimeError(f"Expected at least two png files, found {len(pngs_after)}")
    if not summary_json_path.exists() or not summary_md_path.exists():
        raise RuntimeError("Missing required summary JSON/MD outputs")
    if not observed_read_only:
        raise RuntimeError("observed_state was not loaded read-only")
    if sha256_file(observed_path) != observed_sha_before:
        raise RuntimeError("observed_state hash changed during visualization")
    if not ensure_no_rollout_files(output_dir):
        raise RuntimeError("Unexpected rollout-like files found in output_dir")

    print("Stage 4A-6.5d spatial visualization complete")
    print(f"output_dir: {output_dir}")
    print(f"png_count: {len(pngs_after)}")
    print(f"summary_json: {summary_json_path}")
    print(f"summary_md: {summary_md_path}")
    print(f"conclusion: {interpretation['conclusion']}")


if __name__ == "__main__":
    main()
