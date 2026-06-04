#!/usr/bin/env python3
"""Visualize Stage 4A-3 simulator expert rollout episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from sim_paper_expert import FREE, OCCUPIED, UNKNOWN, normalize_bounds
from sim_rollout_utils import load_json, load_jsonl, save_json

STATE_CMAP = ListedColormap(["#303742", "#86c5da", "#c75656"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)
STATE_TICKS = [int(UNKNOWN), int(FREE), int(OCCUPIED)]
STATE_TICKLABELS = ["unknown", "free", "occupied"]


def project_topdown(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == FREE, axis=2)
    topdown = np.full(observed_state.shape[:2], UNKNOWN, dtype=np.int8)
    topdown[free] = FREE
    topdown[occupied] = OCCUPIED
    return topdown


def _draw_state_colorbar(fig: plt.Figure, ax: plt.Axes) -> None:
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=STATE_NORM, cmap=STATE_CMAP),
        ax=ax,
        ticks=STATE_TICKS,
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(STATE_TICKLABELS)


def _extent(bounds: dict[str, Any]) -> list[float]:
    normalized = normalize_bounds(bounds)
    return [normalized["x"][0], normalized["x"][1], normalized["y"][0], normalized["y"][1]]


def _plot_pose_arrow(ax: plt.Axes, xy: np.ndarray, yaw: float, color: str, label: str | None = None) -> None:
    ax.scatter(xy[0], xy[1], s=90, c=color, marker="^", edgecolors="white", linewidths=0.7, label=label)
    ax.arrow(
        xy[0],
        xy[1],
        0.35 * np.cos(float(yaw)),
        0.35 * np.sin(float(yaw)),
        width=0.015,
        head_width=0.10,
        head_length=0.12,
        color=color,
        length_includes_head=True,
    )


def _path_xy_to_world(path_xy: Any, bounds: dict[str, Any], shape_xy: tuple[int, int]) -> np.ndarray:
    if path_xy is None:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(path_xy, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = arr.reshape((-1, 2))
    normalized = normalize_bounds(bounds)
    voxel_size_x = (normalized["x"][1] - normalized["x"][0]) / float(shape_xy[0])
    voxel_size_y = (normalized["y"][1] - normalized["y"][0]) / float(shape_xy[1])
    world = np.zeros_like(arr, dtype=np.float64)
    world[:, 0] = normalized["x"][0] + (arr[:, 0] + 0.5) * voxel_size_x
    world[:, 1] = normalized["y"][0] + (arr[:, 1] + 0.5) * voxel_size_y
    return world


def save_observed_ratio_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    ratios = np.asarray([float(t["observed_ratio_after"]) for t in transitions], dtype=np.float64)
    before0 = float(transitions[0]["observed_ratio_before"]) if transitions else 0.0

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if transitions:
        ax.plot(np.concatenate([[steps[0] - 1], steps]), np.concatenate([[before0], ratios]), marker="o", color="#2563eb")
    ax.set_xlabel("step")
    ax.set_ylabel("observed_ratio")
    ax.set_title("Measured observed ratio")
    ax.grid(alpha=0.25)
    out_path = episode_dir / "observed_ratio_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_frontier_count_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    frontier_counts = np.asarray([int(t["frontier_count"]) for t in transitions], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if transitions:
        ax.plot(steps, frontier_counts, marker="o", color="#d97706")
    ax.set_xlabel("step")
    ax.set_ylabel("frontier_count")
    ax.set_title("Frontier count")
    ax.grid(alpha=0.25)
    out_path = episode_dir / "frontier_count_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_reachable_candidates_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    reachable = np.asarray([int(t.get("reachable_candidates", 0)) for t in transitions], dtype=np.int64)
    unreachable = np.asarray([int(t.get("unreachable_candidates", 0)) for t in transitions], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if transitions:
        ax.plot(steps, reachable, marker="o", color="#15803d", label="reachable")
        ax.plot(steps, unreachable, marker="o", color="#b91c1c", label="unreachable")
    ax.set_xlabel("step")
    ax.set_ylabel("candidate count")
    ax.set_title("A* candidate reachability")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "reachable_candidates_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_reachable_component_count_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    component = np.asarray([int(t.get("reachable_component_count", 0)) for t in transitions], dtype=np.int64)
    frontier_adjacent = np.asarray(
        [int(t.get("reachable_frontier_adjacent_count", 0)) for t in transitions],
        dtype=np.int64,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if transitions:
        ax.plot(steps, component, marker="o", color="#0f766e", label="reachable component")
        ax.plot(steps, frontier_adjacent, marker="o", color="#d97706", label="reachable frontier-adjacent")
    ax.set_xlabel("step")
    ax.set_ylabel("2D cell count")
    ax.set_title("Reachable observed-free component")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "reachable_component_count_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_gain_exp_gain_sc_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    gain_exp = np.asarray([float(t.get("best_gain_exp", 0.0)) for t in transitions], dtype=np.float64)
    gain_sc = np.asarray([float(t.get("best_gain_sc", 0.0)) for t in transitions], dtype=np.float64)
    gain_hybrid = np.asarray([float(t.get("best_gain_hybrid", 0.0)) for t in transitions], dtype=np.float64)
    weighted_gain_sc = np.asarray([float(t.get("best_weighted_gain_sc", np.nan)) for t in transitions], dtype=np.float64)
    gain_hybrid_weighted = np.asarray(
        [float(t.get("best_gain_hybrid_weighted", np.nan)) for t in transitions],
        dtype=np.float64,
    )
    first = transitions[0] if transitions else {}
    weight = first.get("sc_gain_weight")
    cap = first.get("sc_gain_cap")
    cap_value = first.get("sc_gain_cap_value")
    tau = first.get("tau")

    fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
    if transitions:
        ax.plot(steps, gain_exp, marker="o", color="#0f766e", label="gain_exp")
        ax.plot(steps, gain_sc, marker="o", color="#7c3aed", label="gain_sc")
        ax.plot(steps, gain_hybrid, marker="o", color="#2563eb", label="gain_hybrid")
        if np.isfinite(weighted_gain_sc).any():
            ax.plot(steps, weighted_gain_sc, marker="o", color="#c2410c", label="weighted_gain_sc")
        if np.isfinite(gain_hybrid_weighted).any():
            ax.plot(
                steps,
                gain_hybrid_weighted,
                marker="o",
                color="#be123c",
                label="gain_hybrid_weighted",
            )
    ax.set_xlabel("step")
    ax.set_ylabel("best candidate gain")
    title = "Best candidate measured vs SC gain"
    if weight is not None or cap is not None or tau is not None:
        cap_text = cap
        if cap_text is None and cap_value is not None and float(cap_value) >= 0.0:
            cap_text = cap_value
        title += f" (tau={tau}, w={weight}, cap={cap_text})"
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "gain_exp_gain_sc_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_raw_vs_effective_gain_sc_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    raw_gain = np.asarray([float(t.get("best_gain_sc", 0.0)) for t in transitions], dtype=np.float64)
    effective_gain = np.asarray(
        [float(t.get("best_effective_gain_sc", t.get("best_weighted_gain_sc", t.get("best_gain_sc", 0.0)))) for t in transitions],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    if transitions:
        ax.plot(steps, raw_gain, marker="o", color="#7c3aed", label="raw gain_sc")
        ax.plot(steps, effective_gain, marker="o", color="#0f766e", label="effective gain_sc")
    ax.set_xlabel("step")
    ax.set_ylabel("best candidate SC gain")
    ax.set_title("Raw vs effective SC gain")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "raw_gain_sc_vs_effective_gain_sc_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_effective_gain_hybrid_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    gain_exp = np.asarray([float(t.get("best_gain_exp", 0.0)) for t in transitions], dtype=np.float64)
    effective_hybrid = np.asarray(
        [
            float(
                t.get(
                    "best_gain_hybrid_effective",
                    float(t.get("best_gain_exp", 0.0))
                    + float(t.get("best_effective_gain_sc", t.get("best_gain_sc", 0.0))),
                )
            )
            for t in transitions
        ],
        dtype=np.float64,
    )
    weighted_hybrid = np.asarray(
        [float(t.get("best_gain_hybrid_weighted", t.get("best_gain_hybrid", 0.0))) for t in transitions],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.4, 4.6), constrained_layout=True)
    if transitions:
        ax.plot(steps, gain_exp, marker="o", color="#0f766e", label="gain_exp")
        ax.plot(steps, effective_hybrid, marker="o", color="#2563eb", label="gain_hybrid_effective")
        ax.plot(steps, weighted_hybrid, marker="o", color="#be123c", label="gain_hybrid_weighted")
    ax.set_xlabel("step")
    ax.set_ylabel("best candidate gain")
    ax.set_title("Effective hybrid gain")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "effective_gain_hybrid_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_sc_selected_voxel_count_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    raw_pos = np.asarray([int(t.get("candidates_with_gain_sc_positive", 0)) for t in transitions], dtype=np.int64)
    eff_pos = np.asarray(
        [int(t.get("candidates_with_effective_gain_sc_positive", t.get("candidates_with_gain_sc_positive", 0))) for t in transitions],
        dtype=np.int64,
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    if transitions:
        ax.plot(steps, raw_pos, marker="o", color="#7c3aed", label="raw gain_sc > 0")
        ax.plot(steps, eff_pos, marker="o", color="#0f766e", label="effective gain_sc > 0")
    ax.set_xlabel("step")
    ax.set_ylabel("candidate count")
    ax.set_title("SC-positive candidate count")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "sc_selected_voxel_count_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_prediction_selectivity_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    candidate_count = np.asarray([max(1, int(t.get("candidate_count", 1))) for t in transitions], dtype=np.float64)
    raw_pos = np.asarray([int(t.get("candidates_with_gain_sc_positive", 0)) for t in transitions], dtype=np.float64)
    eff_pos = np.asarray(
        [int(t.get("candidates_with_effective_gain_sc_positive", t.get("candidates_with_gain_sc_positive", 0))) for t in transitions],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    if transitions:
        ax.plot(steps, raw_pos / candidate_count, marker="o", color="#7c3aed", label="raw positive fraction")
        ax.plot(steps, eff_pos / candidate_count, marker="o", color="#0f766e", label="effective positive fraction")
    ax.set_xlabel("step")
    ax.set_ylabel("fraction of candidates")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Prediction gain selectivity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "prediction_selectivity_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_best_score_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    scores = np.asarray([float(t.get("best_score", 0.0)) for t in transitions], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if transitions:
        ax.plot(steps, scores, marker="o", color="#1d4ed8")
    ax.set_xlabel("step")
    ax.set_ylabel("best_score")
    ax.set_title("Best candidate score")
    ax.grid(alpha=0.25)
    out_path = episode_dir / "best_score_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_map_predict_timing_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    preprocess = np.asarray([float(t.get("map_predict_preprocess_time", 0.0)) for t in transitions], dtype=np.float64)
    inference = np.asarray([float(t.get("map_predict_inference_time", 0.0)) for t in transitions], dtype=np.float64)
    alignment = np.asarray([float(t.get("map_predict_alignment_time", 0.0)) for t in transitions], dtype=np.float64)
    total = np.asarray([float(t.get("map_predict_total_time", 0.0)) for t in transitions], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.5, 4.7), constrained_layout=True)
    if transitions:
        ax.plot(steps, preprocess, marker="o", color="#0f766e", label="preprocess")
        ax.plot(steps, inference, marker="o", color="#dc2626", label="inference")
        ax.plot(steps, alignment, marker="o", color="#7c3aed", label="alignment")
        ax.plot(steps, total, marker="o", color="#2563eb", label="total")
    ax.set_xlabel("step")
    ax.set_ylabel("seconds")
    ax.set_title("map_predict timing")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    out_path = episode_dir / "map_predict_timing_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_prediction_valid_count_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    counts = np.asarray([int(t.get("prediction_valid_voxels", 0)) for t in transitions], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if transitions:
        ax.plot(steps, counts, marker="o", color="#0369a1")
    ax.set_xlabel("step")
    ax.set_ylabel("valid prediction voxels")
    ax.set_title("Prediction valid voxel count")
    ax.grid(alpha=0.25)
    out_path = episode_dir / "prediction_valid_count_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_predicted_unmeasured_count_curve(transitions: list[dict[str, Any]], episode_dir: Path) -> str:
    steps = np.asarray([int(t["step"]) for t in transitions], dtype=np.int32)
    counts = np.asarray([int(t.get("predicted_unmeasured_voxels", 0)) for t in transitions], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if transitions:
        ax.plot(steps, counts, marker="o", color="#7c3aed")
    ax.set_xlabel("step")
    ax.set_ylabel("predicted unmeasured voxels")
    ax.set_title("Predicted but unmeasured voxel count")
    ax.grid(alpha=0.25)
    out_path = episode_dir / "predicted_unmeasured_count_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_rollout_topdown_path(
    observed_state: np.ndarray,
    transitions: list[dict[str, Any]],
    bounds: dict[str, Any],
    episode_dir: Path,
) -> str:
    topdown = project_topdown(observed_state)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", extent=_extent(bounds), cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")

    if transitions:
        path_xy = [np.asarray(t["current_pose_world"][:2], dtype=np.float64) for t in transitions]
        path_xy.append(np.asarray(transitions[-1]["selected_next_pose_world"][:2], dtype=np.float64))
        path = np.stack(path_xy, axis=0)
        ax.plot(path[:, 0], path[:, 1], "-o", color="#2563eb", linewidth=2.0, markersize=4.5, label="camera path")

        selected = np.asarray([t["selected_next_pose_world"][:2] for t in transitions], dtype=np.float64)
        ax.scatter(
            selected[:, 0],
            selected[:, 1],
            s=90,
            c="#22a06b",
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            label="selected waypoints",
        )
        _plot_pose_arrow(
            ax,
            np.asarray(transitions[0]["current_pose_world"][:2], dtype=np.float64),
            float(transitions[0]["current_yaw"]),
            "#1d4ed8",
            "start",
        )
        _plot_pose_arrow(
            ax,
            np.asarray(transitions[-1]["selected_next_pose_world"][:2], dtype=np.float64),
            float(transitions[-1]["selected_next_yaw"]),
            "#15803d",
            "final selected pose",
        )
        astar_label_used = False
        for transition in transitions:
            path_world = _path_xy_to_world(
                transition.get("best_astar_path_xy"),
                bounds,
                tuple(int(v) for v in observed_state.shape[:2]),
            )
            if len(path_world):
                ax.plot(
                    path_world[:, 0],
                    path_world[:, 1],
                    color="#22a06b",
                    linewidth=1.3,
                    alpha=0.55,
                    label="selected A* segments" if not astar_label_used else None,
                )
                astar_label_used = True

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Stage 4A-3 rollout path on final measured map")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    _draw_state_colorbar(fig, ax)
    out_path = episode_dir / "rollout_topdown_path.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_step_topdown(
    observed_state: np.ndarray,
    transition: dict[str, Any],
    bounds: dict[str, Any],
    episode_dir: Path,
) -> str:
    step = int(transition["step"])
    topdown = project_topdown(observed_state)
    fig, ax = plt.subplots(figsize=(7.8, 6.8), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", extent=_extent(bounds), cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")

    prediction_path = episode_dir / f"prediction_step{step:03d}" / "global_prediction_layer.npz"
    if prediction_path.exists():
        with np.load(prediction_path) as pred:
            valid = np.asarray(pred["global_prediction_valid"], dtype=bool)
            confidence = np.asarray(pred["global_confidence"], dtype=np.float32)
            tau = float(transition.get("tau", 0.1))
            valid_tau = valid & (confidence >= tau)
            predicted_unmeasured = valid_tau & (observed_state == UNKNOWN)
            valid_top = np.any(valid_tau, axis=2)
            unmeasured_top = np.any(predicted_unmeasured, axis=2)
        ax.imshow(
            np.ma.masked_where(~valid_top.T, valid_top.T),
            origin="lower",
            extent=_extent(bounds),
            cmap=ListedColormap(["#2563eb"]),
            alpha=0.16,
            interpolation="nearest",
        )
        ax.imshow(
            np.ma.masked_where(~unmeasured_top.T, unmeasured_top.T),
            origin="lower",
            extent=_extent(bounds),
            cmap=ListedColormap(["#7c3aed"]),
            alpha=0.22,
            interpolation="nearest",
        )

    candidates = np.asarray(transition["candidate_positions_world"], dtype=np.float64)
    if candidates.size:
        ax.scatter(
            candidates[:, 0],
            candidates[:, 1],
            s=35,
            c="#111827",
            alpha=0.65,
            edgecolors="white",
            linewidths=0.4,
            label="top candidates",
        )

    current_xy = np.asarray(transition["current_pose_world"][:2], dtype=np.float64)
    selected_xy = np.asarray(transition["selected_next_pose_world"][:2], dtype=np.float64)
    astar_path = _path_xy_to_world(transition.get("best_astar_path_xy"), bounds, tuple(int(v) for v in observed_state.shape[:2]))
    if len(astar_path):
        ax.plot(astar_path[:, 0], astar_path[:, 1], color="#22a06b", linewidth=2.2, label="selected A* path")
    _plot_pose_arrow(ax, current_xy, float(transition["current_yaw"]), "#2563eb", "current pose")
    ax.scatter(
        selected_xy[0],
        selected_xy[1],
        s=170,
        c="#22a06b",
        marker="*",
        edgecolors="black",
        linewidths=0.6,
        label="best candidate",
    )
    ax.annotate(
        "",
        xy=(selected_xy[0], selected_xy[1]),
        xytext=(current_xy[0], current_xy[1]),
        arrowprops={"arrowstyle": "->", "lw": 1.9, "color": "#22a06b"},
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"Step {step:03d}: ratio {float(transition['observed_ratio_after']):.4f}, "
        f"score {float(transition['best_score']):.3f}, gain_sc {float(transition.get('best_gain_sc', 0.0)):.1f}, "
        f"w {float(transition.get('sc_gain_weight', 1.0)):.2f}, "
        f"cap {float(transition.get('sc_gain_cap_value', -1.0)):.1f}, "
        f"tau {float(transition.get('tau', 0.1)):.2f}"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    _draw_state_colorbar(fig, ax)
    out_path = episode_dir / f"step_topdown_{step:03d}.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_html_index(episode_dir: Path, generated: dict[str, Any]) -> str:
    image_names: list[str] = []
    for value in generated.values():
        if isinstance(value, list):
            image_names.extend([Path(v).name for v in value])
        elif isinstance(value, str) and value.endswith(".png"):
            image_names.append(Path(value).name)

    html_lines = [
        "<!doctype html>",
        "<html>",
        "<head><meta charset=\"utf-8\"><title>Simulator Expert Rollout</title></head>",
        "<body>",
        "<h1>Simulator Expert Rollout</h1>",
        "<ul>",
    ]
    for image_name in image_names:
        html_lines.append(f'<li><a href="{image_name}">{image_name}</a></li>')
    html_lines.extend(["</ul>"])
    for image_name in image_names:
        html_lines.append(f'<h2>{image_name}</h2><img src="{image_name}" style="max-width: 980px; width: 100%;">')
    html_lines.extend(["</body>", "</html>"])

    out_path = episode_dir / "rollout_index.html"
    out_path.write_text("\n".join(html_lines) + "\n", encoding="utf-8")
    return str(out_path)


def save_rollout_visualizations(
    episode_dir: str | Path,
    save_steps: bool = True,
) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    transitions = load_jsonl(episode_dir / "transitions.jsonl")
    if not transitions:
        raise FileNotFoundError(f"No transitions found in {episode_dir / 'transitions.jsonl'}")

    summary_path = episode_dir / "episode_summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    bounds = normalize_bounds(summary.get("map_bounds") or transitions[0].get("bounds"))

    final_map_path = episode_dir / "observed_state_final.npy"
    if not final_map_path.exists():
        final_step = max(int(t["step"]) for t in transitions)
        final_map_path = episode_dir / f"observed_state_step{final_step:03d}.npy"
    observed_final = np.load(final_map_path)

    generated: dict[str, Any] = {}
    generated["rollout_topdown_path"] = save_rollout_topdown_path(observed_final, transitions, bounds, episode_dir)
    generated["observed_ratio_curve"] = save_observed_ratio_curve(transitions, episode_dir)
    generated["frontier_count_curve"] = save_frontier_count_curve(transitions, episode_dir)
    generated["gain_exp_gain_sc_curve"] = save_gain_exp_gain_sc_curve(transitions, episode_dir)
    generated["raw_gain_sc_vs_effective_gain_sc_curve"] = save_raw_vs_effective_gain_sc_curve(transitions, episode_dir)
    generated["effective_gain_hybrid_curve"] = save_effective_gain_hybrid_curve(transitions, episode_dir)
    generated["sc_selected_voxel_count_curve"] = save_sc_selected_voxel_count_curve(transitions, episode_dir)
    generated["prediction_selectivity_curve"] = save_prediction_selectivity_curve(transitions, episode_dir)
    generated["best_score_curve"] = save_best_score_curve(transitions, episode_dir)
    if any("reachable_candidates" in t for t in transitions):
        generated["reachable_candidates_curve"] = save_reachable_candidates_curve(transitions, episode_dir)
    if any("reachable_component_count" in t for t in transitions):
        generated["reachable_component_count_curve"] = save_reachable_component_count_curve(transitions, episode_dir)
    if any("map_predict_total_time" in t for t in transitions):
        generated["map_predict_timing_curve"] = save_map_predict_timing_curve(transitions, episode_dir)
    if any("prediction_valid_voxels" in t for t in transitions):
        generated["prediction_valid_count_curve"] = save_prediction_valid_count_curve(transitions, episode_dir)
    if any("predicted_unmeasured_voxels" in t for t in transitions):
        generated["predicted_unmeasured_count_curve"] = save_predicted_unmeasured_count_curve(transitions, episode_dir)

    if save_steps:
        step_paths = []
        for transition in transitions:
            step = int(transition["step"])
            observed_path = episode_dir / f"observed_state_step{step:03d}.npy"
            if observed_path.exists():
                observed_state = np.load(observed_path)
                step_paths.append(save_step_topdown(observed_state, transition, bounds, episode_dir))
        generated["step_topdown"] = step_paths

    generated["rollout_index"] = save_html_index(episode_dir, generated)
    save_json(episode_dir / "viz_summary.json", {"episode_dir": str(episode_dir), "generated": generated})
    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a Stage 4A-3 simulator expert rollout.")
    parser.add_argument(
        "--episode_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_rollout_empty_pred/episodes/minimal_room_empty_pred_000",
    )
    parser.add_argument("--no_step_images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = save_rollout_visualizations(args.episode_dir, save_steps=not args.no_step_images)
    print("Stage 4A-3 rollout visualization complete.")
    for key, value in generated.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
