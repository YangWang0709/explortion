#!/usr/bin/env python3
"""Stage 4A-6.1 step-level SC-aware rollout behavior analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim_rollout_utils import load_json, load_jsonl, save_json
from visualize_sim_rollout import STATE_CMAP, STATE_NORM, project_topdown


def _mean(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _corr(xs: list[float], ys: list[float]) -> float | None:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(mask)) < 2:
        return None
    if float(np.std(x[mask])) <= 1e-12 or float(np.std(y[mask])) <= 1e-12:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _pos(record: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(record.get(key, [np.nan, np.nan, np.nan]), dtype=np.float64)


def _actions_changed(empty: dict[str, Any], sc: dict[str, Any]) -> bool:
    empty_pos = _pos(empty, "selected_next_pose_world")
    sc_pos = _pos(sc, "selected_next_pose_world")
    empty_yaw = float(empty.get("selected_next_yaw", 0.0))
    sc_yaw = float(sc.get("selected_next_yaw", 0.0))
    return bool(np.linalg.norm(empty_pos - sc_pos) > 1e-4 or abs(empty_yaw - sc_yaw) > 1e-4)


def _row(step: int, empty: dict[str, Any], sc: dict[str, Any]) -> dict[str, Any]:
    empty_after = float(empty.get("observed_ratio_after", 0.0))
    sc_after = float(sc.get("observed_ratio_after", 0.0))
    empty_delta = float(empty.get("delta_observed_ratio", 0.0))
    sc_delta = float(sc.get("delta_observed_ratio", 0.0))
    empty_pos = _pos(empty, "selected_next_pose_world")
    sc_pos = _pos(sc, "selected_next_pose_world")
    distance = float(np.linalg.norm(empty_pos[:2] - sc_pos[:2]))
    return {
        "step": int(step),
        "empty_observed_ratio_after": empty_after,
        "sc_observed_ratio_after": sc_after,
        "observed_ratio_delta_sc_minus_empty": float(sc_after - empty_after),
        "empty_delta_observed_ratio": empty_delta,
        "sc_delta_observed_ratio": sc_delta,
        "delta_observed_ratio_sc_minus_empty": float(sc_delta - empty_delta),
        "selected_action_changed": _actions_changed(empty, sc),
        "selected_xy_distance_m": distance,
        "empty_selected_world_x": float(empty_pos[0]),
        "empty_selected_world_y": float(empty_pos[1]),
        "empty_selected_world_z": float(empty_pos[2]),
        "sc_selected_world_x": float(sc_pos[0]),
        "sc_selected_world_y": float(sc_pos[1]),
        "sc_selected_world_z": float(sc_pos[2]),
        "empty_path_cost": float(empty.get("best_path_cost", empty.get("path_cost", 0.0))),
        "sc_path_cost": float(sc.get("best_path_cost", sc.get("path_cost", 0.0))),
        "path_cost_delta_sc_minus_empty": float(sc.get("best_path_cost", 0.0)) - float(empty.get("best_path_cost", 0.0)),
        "empty_best_score": float(empty.get("best_score", empty.get("final_score", 0.0))),
        "sc_best_score": float(sc.get("best_score", sc.get("final_score", 0.0))),
        "score_delta_sc_minus_empty": float(sc.get("best_score", 0.0)) - float(empty.get("best_score", 0.0)),
        "empty_gain_exp": float(empty.get("best_gain_exp", empty.get("gain_exp", 0.0))),
        "sc_gain_exp": float(sc.get("best_gain_exp", sc.get("gain_exp", 0.0))),
        "empty_gain_sc": float(empty.get("best_gain_sc", empty.get("gain_sc", 0.0))),
        "sc_gain_sc": float(sc.get("best_gain_sc", sc.get("gain_sc", 0.0))),
        "sc_weighted_gain_sc": float(sc.get("best_weighted_gain_sc", sc.get("weighted_gain_sc", sc.get("best_gain_sc", 0.0)))),
        "empty_gain_hybrid": float(empty.get("best_gain_hybrid", empty.get("gain_hybrid", 0.0))),
        "sc_gain_hybrid": float(sc.get("best_gain_hybrid", sc.get("gain_hybrid", 0.0))),
        "sc_gain_hybrid_weighted": float(
            sc.get("best_gain_hybrid_weighted", sc.get("gain_hybrid_weighted", sc.get("best_gain_hybrid", 0.0)))
        ),
        "sc_predicted_unmeasured_voxels": int(sc.get("predicted_unmeasured_voxels", 0)),
        "sc_candidates_with_gain_sc_positive": int(sc.get("candidates_with_gain_sc_positive", 0)),
        "empty_frontier_count": int(empty.get("frontier_count", 0)),
        "sc_frontier_count": int(sc.get("frontier_count", 0)),
        "empty_reachable_component_count": int(empty.get("reachable_component_count", 0)),
        "sc_reachable_component_count": int(sc.get("reachable_component_count", 0)),
        "empty_reachable_candidates": int(empty.get("reachable_candidates", 0)),
        "sc_reachable_candidates": int(sc.get("reachable_candidates", 0)),
        "empty_candidate_count": int(empty.get("candidate_count", 0)),
        "sc_candidate_count": int(sc.get("candidate_count", 0)),
        "sc_prediction_npz": str(sc.get("prediction_npz", "")),
        "sc_prediction_summary_json": str(sc.get("prediction_summary_json", "")),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _series(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)


def _plot_observed_ratio(rows: list[dict[str, Any]], out_path: Path) -> str:
    steps = _series(rows, "step")
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(steps, _series(rows, "empty_observed_ratio_after"), marker="o", color="#f97316", label="empty")
    ax.plot(steps, _series(rows, "sc_observed_ratio_after"), marker="o", color="#2563eb", label="SC")
    ax.bar(
        steps,
        _series(rows, "observed_ratio_delta_sc_minus_empty"),
        width=0.32,
        color="#64748b",
        alpha=0.35,
        label="SC-empty delta",
    )
    ax.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("step")
    ax.set_ylabel("observed_ratio_after")
    ax.set_title("Measured coverage: SC vs empty")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_gain_score(rows: list[dict[str, Any]], out_path: Path) -> str:
    steps = _series(rows, "step")
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2), constrained_layout=True, sharex=True)
    axes[0].plot(steps, _series(rows, "empty_gain_exp"), marker="o", color="#f97316", label="empty gain_exp")
    axes[0].plot(steps, _series(rows, "sc_gain_exp"), marker="o", color="#0f766e", label="SC gain_exp")
    axes[0].plot(steps, _series(rows, "sc_gain_sc"), marker="o", color="#7c3aed", label="SC gain_sc")
    axes[0].plot(steps, _series(rows, "sc_gain_hybrid"), marker="o", color="#2563eb", label="SC gain_hybrid")
    axes[0].set_ylabel("gain")
    axes[0].set_title("Selected gain terms")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(steps, _series(rows, "empty_best_score"), marker="o", color="#f97316", label="empty score")
    axes[1].plot(steps, _series(rows, "sc_best_score"), marker="o", color="#2563eb", label="SC score")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("best_score")
    axes[1].set_title("Selected utility score")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_path_cost(rows: list[dict[str, Any]], out_path: Path) -> str:
    steps = _series(rows, "step")
    fig, ax = plt.subplots(figsize=(8.2, 4.6), constrained_layout=True)
    ax.plot(steps, _series(rows, "empty_path_cost"), marker="o", color="#f97316", label="empty")
    ax.plot(steps, _series(rows, "sc_path_cost"), marker="o", color="#2563eb", label="SC")
    ax.set_xlabel("step")
    ax.set_ylabel("selected path_cost")
    ax.set_title("Selected A* path cost")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_prediction_counts(rows: list[dict[str, Any]], out_path: Path) -> str:
    steps = _series(rows, "step")
    fig, ax1 = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax1.plot(
        steps,
        _series(rows, "sc_predicted_unmeasured_voxels"),
        marker="o",
        color="#7c3aed",
        label="predicted_unmeasured_voxels",
    )
    ax1.set_xlabel("step")
    ax1.set_ylabel("voxel count")
    ax1.grid(alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(
        steps,
        _series(rows, "sc_candidates_with_gain_sc_positive"),
        marker="s",
        color="#2563eb",
        label="candidates with gain_sc > 0",
    )
    ax2.set_ylabel("candidate count")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title("SC prediction density and candidate exposure")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _bounds(summary: dict[str, Any], transitions: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    raw = summary.get("map_bounds") or (transitions[0].get("bounds") if transitions else None)
    if raw is None:
        raw = {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]}
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def _extent(bounds: dict[str, tuple[float, float]]) -> list[float]:
    return [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]


def _world_path(transitions: list[dict[str, Any]], max_steps: int) -> np.ndarray:
    if not transitions:
        return np.zeros((0, 2), dtype=np.float64)
    path = [np.asarray(transitions[0]["current_pose_world"][:2], dtype=np.float64)]
    path.extend(np.asarray(t["selected_next_pose_world"][:2], dtype=np.float64) for t in transitions[:max_steps])
    return np.stack(path, axis=0)


def _grid_path_to_world(path_xy: Any, bounds: dict[str, tuple[float, float]], shape_xy: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(path_xy, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = arr.reshape((-1, 2))
    world = np.empty_like(arr, dtype=np.float64)
    world[:, 0] = bounds["x"][0] + (arr[:, 0] + 0.5) * ((bounds["x"][1] - bounds["x"][0]) / float(shape_xy[0]))
    world[:, 1] = bounds["y"][0] + (arr[:, 1] + 0.5) * ((bounds["y"][1] - bounds["y"][0]) / float(shape_xy[1]))
    return world


def _plot_paths(
    empty_episode: Path,
    sc_episode: Path,
    empty_summary: dict[str, Any],
    sc_summary: dict[str, Any],
    empty_transitions: list[dict[str, Any]],
    sc_transitions: list[dict[str, Any]],
    max_steps: int,
    out_path: Path,
) -> str:
    map_path = sc_episode / "observed_state_final.npy"
    if not map_path.exists():
        map_path = empty_episode / "observed_state_final.npy"
    observed = np.load(map_path)
    bounds = _bounds(sc_summary or empty_summary, sc_transitions or empty_transitions)
    fig, ax = plt.subplots(figsize=(8.2, 7.3), constrained_layout=True)
    ax.imshow(
        project_topdown(observed).T,
        origin="lower",
        extent=_extent(bounds),
        cmap=STATE_CMAP,
        norm=STATE_NORM,
        interpolation="nearest",
    )
    for label, transitions, color in (
        ("empty", empty_transitions, "#f97316"),
        ("SC", sc_transitions, "#2563eb"),
    ):
        path = _world_path(transitions, max_steps)
        if len(path):
            ax.plot(path[:, 0], path[:, 1], "-o", linewidth=2.0, markersize=4.5, color=color, label=label)
        for transition in transitions[:max_steps]:
            astar = _grid_path_to_world(transition.get("best_astar_path_xy", []), bounds, observed.shape[:2])
            if len(astar):
                ax.plot(astar[:, 0], astar[:, 1], color=color, linewidth=1.0, alpha=0.25)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Selected path and A* segment comparison")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _plot_action_divergence(
    empty_episode: Path,
    sc_episode: Path,
    empty_summary: dict[str, Any],
    sc_summary: dict[str, Any],
    empty_transitions: list[dict[str, Any]],
    sc_transitions: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    max_steps: int,
    out_path: Path,
) -> str:
    map_path = sc_episode / "observed_state_final.npy"
    observed = np.load(map_path if map_path.exists() else empty_episode / "observed_state_final.npy")
    bounds = _bounds(sc_summary or empty_summary, sc_transitions or empty_transitions)
    fig, ax = plt.subplots(figsize=(8.2, 7.3), constrained_layout=True)
    ax.imshow(project_topdown(observed).T, origin="lower", extent=_extent(bounds), cmap=STATE_CMAP, norm=STATE_NORM)
    empty_selected = np.asarray([_pos(t, "selected_next_pose_world")[:2] for t in empty_transitions[:max_steps]], dtype=np.float64)
    sc_selected = np.asarray([_pos(t, "selected_next_pose_world")[:2] for t in sc_transitions[:max_steps]], dtype=np.float64)
    if len(empty_selected):
        ax.scatter(empty_selected[:, 0], empty_selected[:, 1], s=70, color="#f97316", label="empty selected")
    if len(sc_selected):
        ax.scatter(sc_selected[:, 0], sc_selected[:, 1], s=70, color="#2563eb", label="SC selected")
    for idx, row in enumerate(rows[:max_steps]):
        if not bool(row["selected_action_changed"]):
            continue
        e = empty_selected[idx]
        s = sc_selected[idx]
        ax.annotate("", xy=(s[0], s[1]), xytext=(e[0], e[1]), arrowprops={"arrowstyle": "->", "color": "#111827"})
        ax.text((e[0] + s[0]) * 0.5, (e[1] + s[1]) * 0.5, str(idx), fontsize=8, color="#111827")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Action divergence: arrows point empty -> SC")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _diagnosis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_lag_step = None
    for row in rows:
        if float(row["observed_ratio_delta_sc_minus_empty"]) < 0.0:
            first_lag_step = int(row["step"])
            break
    sc_lower_path_steps = [int(r["step"]) for r in rows if float(r["path_cost_delta_sc_minus_empty"]) < 0.0]
    dense_steps = [
        int(r["step"])
        for r in rows
        if int(r["sc_candidates_with_gain_sc_positive"]) >= max(1, int(r["sc_candidate_count"]) - 1)
    ]
    revisit_distances = []
    sc_positions = np.asarray([[r["sc_selected_world_x"], r["sc_selected_world_y"]] for r in rows], dtype=np.float64)
    if len(sc_positions) > 1:
        for i in range(1, len(sc_positions)):
            revisit_distances.append(float(np.min(np.linalg.norm(sc_positions[:i] - sc_positions[i], axis=1))))
    return {
        "first_step_sc_observed_ratio_lags_empty": first_lag_step,
        "changed_selected_actions": int(sum(bool(r["selected_action_changed"]) for r in rows)),
        "compared_steps": int(len(rows)),
        "sc_lower_path_cost_steps": sc_lower_path_steps,
        "mean_empty_path_cost": _mean([float(r["empty_path_cost"]) for r in rows]),
        "mean_sc_path_cost": _mean([float(r["sc_path_cost"]) for r in rows]),
        "mean_empty_gain_exp": _mean([float(r["empty_gain_exp"]) for r in rows]),
        "mean_sc_gain_exp": _mean([float(r["sc_gain_exp"]) for r in rows]),
        "mean_sc_gain_sc": _mean([float(r["sc_gain_sc"]) for r in rows]),
        "mean_empty_best_score": _mean([float(r["empty_best_score"]) for r in rows]),
        "mean_sc_best_score": _mean([float(r["sc_best_score"]) for r in rows]),
        "gain_sc_vs_delta_observed_ratio_corr": _corr(
            [float(r["sc_gain_sc"]) for r in rows],
            [float(r["sc_delta_observed_ratio"]) for r in rows],
        ),
        "predicted_unmeasured_vs_delta_observed_ratio_corr": _corr(
            [float(r["sc_predicted_unmeasured_voxels"]) for r in rows],
            [float(r["sc_delta_observed_ratio"]) for r in rows],
        ),
        "dense_gain_sc_candidate_steps": dense_steps,
        "mean_sc_revisit_nearest_distance_m": _mean(revisit_distances),
        "min_sc_revisit_nearest_distance_m": None if not revisit_distances else float(np.min(revisit_distances)),
    }


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    d = summary["diagnosis"]
    lines = [
        "# Stage 4A-6.1 Existing SC vs Empty Analysis",
        "",
        f"- compared steps: {summary['compared_steps']}",
        f"- empty final observed_ratio: {summary['empty_final_observed_ratio']:.9f}",
        f"- SC final observed_ratio: {summary['sc_final_observed_ratio']:.9f}",
        f"- delta SC-empty: {summary['observed_ratio_delta_sc_minus_empty']:.9f}",
        f"- first step where SC lags empty: {d['first_step_sc_observed_ratio_lags_empty']}",
        f"- changed selected actions: {d['changed_selected_actions']} / {d['compared_steps']}",
        f"- mean path_cost empty / SC: {d['mean_empty_path_cost']:.6f} / {d['mean_sc_path_cost']:.6f}",
        f"- mean gain_exp empty / SC: {d['mean_empty_gain_exp']:.6f} / {d['mean_sc_gain_exp']:.6f}",
        f"- mean SC gain_sc: {d['mean_sc_gain_sc']:.6f}",
        f"- mean best_score empty / SC: {d['mean_empty_best_score']:.6f} / {d['mean_sc_best_score']:.6f}",
        f"- gain_sc vs SC delta_observed_ratio corr: {d['gain_sc_vs_delta_observed_ratio_corr']}",
        f"- predicted_unmeasured vs SC delta_observed_ratio corr: {d['predicted_unmeasured_vs_delta_observed_ratio_corr']}",
        f"- SC lower-path-cost steps: {d['sc_lower_path_cost_steps']}",
        f"- dense gain_sc candidate steps: {d['dense_gain_sc_candidate_steps']}",
        f"- nearest revisit distance mean/min (SC selected xy): {d['mean_sc_revisit_nearest_distance_m']} / {d['min_sc_revisit_nearest_distance_m']}",
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    sc_episode = Path(args.sc_episode_dir).resolve()
    empty_episode = Path(args.empty_episode_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sc_summary = load_json(sc_episode / "episode_summary.json")
    empty_summary = load_json(empty_episode / "episode_summary.json")
    sc_transitions = load_jsonl(sc_episode / "transitions.jsonl")
    empty_transitions = load_jsonl(empty_episode / "transitions.jsonl")
    compared_steps = min(int(args.max_steps), len(sc_transitions), len(empty_transitions))
    if compared_steps <= 0:
        raise RuntimeError("No overlapping transitions to analyze")

    rows = [_row(i, empty_transitions[i], sc_transitions[i]) for i in range(compared_steps)]
    _write_csv(output_dir / "step_comparison.csv", rows)
    save_json(output_dir / "step_comparison.json", {"rows": rows})

    outputs = {
        "observed_ratio_step_comparison": _plot_observed_ratio(rows, output_dir / "observed_ratio_step_comparison.png"),
        "selected_path_comparison": _plot_paths(
            empty_episode,
            sc_episode,
            empty_summary,
            sc_summary,
            empty_transitions,
            sc_transitions,
            compared_steps,
            output_dir / "selected_path_comparison.png",
        ),
        "gain_score_comparison": _plot_gain_score(rows, output_dir / "gain_score_comparison.png"),
        "path_cost_comparison": _plot_path_cost(rows, output_dir / "path_cost_comparison.png"),
        "prediction_counts_comparison": _plot_prediction_counts(rows, output_dir / "prediction_counts_comparison.png"),
        "action_divergence_topdown": _plot_action_divergence(
            empty_episode,
            sc_episode,
            empty_summary,
            sc_summary,
            empty_transitions,
            sc_transitions,
            rows,
            compared_steps,
            output_dir / "action_divergence_topdown.png",
        ),
    }
    diagnosis = _diagnosis(rows)
    interpretation_parts = []
    if diagnosis["dense_gain_sc_candidate_steps"]:
        interpretation_parts.append(
            "gain_sc is dense: nearly every sampled reachable candidate receives positive SC gain on several steps."
        )
    if diagnosis["mean_sc_path_cost"] is not None and diagnosis["mean_empty_path_cost"] is not None:
        if float(diagnosis["mean_sc_path_cost"]) < float(diagnosis["mean_empty_path_cost"]):
            interpretation_parts.append(
                "SC scoring tends to prefer lower path-cost local moves, which can amplify score without improving measured coverage."
            )
    corr = diagnosis["gain_sc_vs_delta_observed_ratio_corr"]
    if corr is not None and corr < 0.0:
        interpretation_parts.append("selected gain_sc is negatively correlated with measured per-step coverage gain.")
    interpretation_parts.append(
        "Prediction remains read-only; the underperformance is a scoring/behavior issue, not observed-map writeback."
    )
    interpretation = " ".join(interpretation_parts)

    summary = {
        "stage": "Stage 4A-6.1 existing SC-aware rollout underperformance analysis",
        "sc_episode_dir": str(sc_episode),
        "empty_episode_dir": str(empty_episode),
        "output_dir": str(output_dir),
        "compared_steps": int(compared_steps),
        "empty_final_observed_ratio": float(rows[-1]["empty_observed_ratio_after"]),
        "sc_final_observed_ratio": float(rows[-1]["sc_observed_ratio_after"]),
        "observed_ratio_delta_sc_minus_empty": float(rows[-1]["observed_ratio_delta_sc_minus_empty"]),
        "diagnosis": diagnosis,
        "interpretation": interpretation,
        "safety": {
            "prediction_writeback": False,
            "prediction_used_for_traversability": False,
            "prediction_used_for_collision": False,
            "prediction_used_for_a_star": False,
            "prediction_blocks_rays": False,
            "target_or_ground_truth_scoring": False,
            "rl_or_training": False,
        },
        "outputs": outputs,
    }
    save_json(output_dir / "analysis_summary.json", summary)
    _write_md(output_dir / "analysis_summary.md", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stage 4A-6 SC rollout underperformance.")
    parser.add_argument("--sc_episode_dir", required=True)
    parser.add_argument("--empty_episode_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    summary = run_analysis(parse_args())
    print("Stage 4A-6.1 existing SC analysis complete.")
    print(f"compared_steps: {summary['compared_steps']}")
    print(f"empty_final_observed_ratio: {summary['empty_final_observed_ratio']:.9f}")
    print(f"sc_final_observed_ratio: {summary['sc_final_observed_ratio']:.9f}")
    print(f"observed_ratio_delta_sc_minus_empty: {summary['observed_ratio_delta_sc_minus_empty']:.9f}")
    print(f"analysis_summary_json: {Path(summary['output_dir']) / 'analysis_summary.json'}")
    print(f"analysis_summary_md: {Path(summary['output_dir']) / 'analysis_summary.md'}")


if __name__ == "__main__":
    main()
