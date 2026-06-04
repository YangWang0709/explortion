#!/usr/bin/env python3
"""Compare a Stage 4A-6 SC-aware rollout against the Stage 4A-4 empty baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim_rollout_utils import load_json, load_jsonl, save_json
from visualize_sim_rollout import STATE_CMAP, STATE_NORM, project_topdown


def _series(transitions: list[dict[str, Any]], key: str, max_steps: int) -> list[float]:
    return [float(t.get(key, 0.0)) for t in transitions[: int(max_steps)]]


def _positions(transitions: list[dict[str, Any]], max_steps: int) -> list[list[float]]:
    return [[float(v) for v in t["selected_next_pose_world"]] for t in transitions[: int(max_steps)]]


def _actions_changed(empty: list[dict[str, Any]], sc: list[dict[str, Any]], max_steps: int) -> tuple[int, list[bool]]:
    changed: list[bool] = []
    for e, s in zip(empty[: int(max_steps)], sc[: int(max_steps)]):
        e_pos = np.asarray(e["selected_next_pose_world"], dtype=np.float64)
        s_pos = np.asarray(s["selected_next_pose_world"], dtype=np.float64)
        e_yaw = float(e.get("selected_next_yaw", 0.0))
        s_yaw = float(s.get("selected_next_yaw", 0.0))
        changed.append(bool(np.linalg.norm(e_pos - s_pos) > 1e-4 or abs(e_yaw - s_yaw) > 1e-4))
    return int(sum(changed)), changed


def _plot_two_series(
    steps: np.ndarray,
    empty_values: list[float],
    sc_values: list[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.7), constrained_layout=True)
    ax.plot(steps[: len(empty_values)], empty_values, marker="o", color="#f97316", label="empty baseline")
    ax.plot(steps[: len(sc_values)], sc_values, marker="o", color="#2563eb", label="SC dynamic")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_gain_comparison(
    steps: np.ndarray,
    empty: list[dict[str, Any]],
    sc: list[dict[str, Any]],
    max_steps: int,
    out_path: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    empty_gain_exp = _series(empty, "best_gain_exp", max_steps)
    sc_gain_exp = _series(sc, "best_gain_exp", max_steps)
    sc_gain_sc = _series(sc, "best_gain_sc", max_steps)
    sc_gain_hybrid = _series(sc, "best_gain_hybrid", max_steps)
    ax.plot(steps[: len(empty_gain_exp)], empty_gain_exp, marker="o", color="#f97316", label="empty gain_exp")
    ax.plot(steps[: len(sc_gain_exp)], sc_gain_exp, marker="o", color="#0f766e", label="SC gain_exp")
    ax.plot(steps[: len(sc_gain_sc)], sc_gain_sc, marker="o", color="#7c3aed", label="SC gain_sc")
    ax.plot(steps[: len(sc_gain_hybrid)], sc_gain_hybrid, marker="o", color="#2563eb", label="SC gain_hybrid")
    ax.set_xlabel("step")
    ax.set_ylabel("best candidate gain")
    ax.set_title("Gain comparison")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _bounds_from_summary(summary: dict[str, Any], transition: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = summary.get("map_bounds") or transition.get("bounds")
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def _plot_path_comparison(
    empty_episode_dir: Path,
    sc_episode_dir: Path,
    empty_summary: dict[str, Any],
    sc_summary: dict[str, Any],
    empty: list[dict[str, Any]],
    sc: list[dict[str, Any]],
    max_steps: int,
    out_path: Path,
) -> str:
    map_path = sc_episode_dir / "observed_state_final.npy"
    if not map_path.exists():
        map_path = empty_episode_dir / "observed_state_final.npy"
    observed = np.load(map_path)
    bounds = _bounds_from_summary(sc_summary or empty_summary, sc[0] if sc else empty[0])
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    topdown = project_topdown(observed)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    for label, transitions, color in (
        ("empty baseline", empty, "#f97316"),
        ("SC dynamic", sc, "#2563eb"),
    ):
        if not transitions:
            continue
        path_xy = [np.asarray(transitions[0]["current_pose_world"][:2], dtype=np.float64)]
        path_xy.extend(np.asarray(t["selected_next_pose_world"][:2], dtype=np.float64) for t in transitions[: int(max_steps)])
        path = np.stack(path_xy, axis=0)
        ax.plot(path[:, 0], path[:, 1], "-o", color=color, linewidth=2.0, markersize=4.5, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Selected path comparison")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    sc_episode_dir = Path(args.sc_episode_dir).resolve()
    empty_episode_dir = Path(args.empty_episode_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(args.max_steps)

    sc_summary = load_json(sc_episode_dir / "episode_summary.json")
    empty_summary = load_json(empty_episode_dir / "episode_summary.json")
    sc_transitions = load_jsonl(sc_episode_dir / "transitions.jsonl")
    empty_transitions = load_jsonl(empty_episode_dir / "transitions.jsonl")
    compared_steps = int(min(max_steps, len(sc_transitions), len(empty_transitions)))
    if compared_steps <= 0:
        raise RuntimeError("No overlapping transitions to compare")

    sc_slice = sc_transitions[:compared_steps]
    empty_slice = empty_transitions[:compared_steps]
    changed_count, changed_flags = _actions_changed(empty_slice, sc_slice, compared_steps)

    empty_final_ratio = float(empty_slice[-1]["observed_ratio_after"])
    sc_final_ratio = float(sc_slice[-1]["observed_ratio_after"])
    empty_path_cost = _series(empty_slice, "best_path_cost", compared_steps)
    sc_path_cost = _series(sc_slice, "best_path_cost", compared_steps)
    empty_best_score = _series(empty_slice, "best_score", compared_steps)
    sc_best_score = _series(sc_slice, "best_score", compared_steps)
    sc_gain_sc = _series(sc_slice, "best_gain_sc", compared_steps)

    steps = np.arange(compared_steps, dtype=np.int32)
    outputs = {
        "observed_ratio_comparison": _plot_two_series(
            steps,
            _series(empty_slice, "observed_ratio_after", compared_steps),
            _series(sc_slice, "observed_ratio_after", compared_steps),
            "observed_ratio_after",
            "Observed ratio comparison",
            output_dir / "observed_ratio_comparison.png",
        ),
        "best_score_comparison": _plot_two_series(
            steps,
            empty_best_score,
            sc_best_score,
            "best_score",
            "Best score comparison",
            output_dir / "best_score_comparison.png",
        ),
        "gain_comparison": _plot_gain_comparison(
            steps,
            empty_slice,
            sc_slice,
            compared_steps,
            output_dir / "gain_comparison.png",
        ),
        "path_topdown_comparison": _plot_path_comparison(
            empty_episode_dir,
            sc_episode_dir,
            empty_summary,
            sc_summary,
            empty_slice,
            sc_slice,
            compared_steps,
            output_dir / "path_topdown_comparison.png",
        ),
    }

    summary = {
        "stage": "Stage 4A-6 comparison to Stage 4A-4 empty baseline",
        "sc_episode_dir": str(sc_episode_dir),
        "empty_episode_dir": str(empty_episode_dir),
        "output_dir": str(output_dir),
        "compared_steps": int(compared_steps),
        "empty_episode_id": empty_summary.get("episode_id", empty_episode_dir.name),
        "sc_episode_id": sc_summary.get("episode_id", sc_episode_dir.name),
        "empty_prediction_mode": empty_summary.get("prediction_mode"),
        "sc_prediction_mode": sc_summary.get("prediction_mode"),
        "empty_observed_ratio": _series(empty_slice, "observed_ratio_after", compared_steps),
        "sc_observed_ratio": _series(sc_slice, "observed_ratio_after", compared_steps),
        "empty_final_observed_ratio": empty_final_ratio,
        "sc_final_observed_ratio": sc_final_ratio,
        "observed_ratio_delta_sc_minus_empty": float(sc_final_ratio - empty_final_ratio),
        "empty_total_delta_observed_ratio": float(empty_final_ratio - float(empty_slice[0]["observed_ratio_before"])),
        "sc_total_delta_observed_ratio": float(sc_final_ratio - float(sc_slice[0]["observed_ratio_before"])),
        "empty_selected_positions": _positions(empty_slice, compared_steps),
        "sc_selected_positions": _positions(sc_slice, compared_steps),
        "selected_action_changed": changed_flags,
        "number_of_changed_selected_actions": int(changed_count),
        "path_difference": {
            "empty_path_cost": empty_path_cost,
            "sc_path_cost": sc_path_cost,
            "mean_path_cost_delta_sc_minus_empty": float(np.mean(sc_path_cost) - np.mean(empty_path_cost)),
        },
        "score_difference": {
            "empty_best_score": empty_best_score,
            "sc_best_score": sc_best_score,
            "mean_score_delta_sc_minus_empty": float(np.mean(sc_best_score) - np.mean(empty_best_score)),
        },
        "gain_difference": {
            "empty_gain_exp": _series(empty_slice, "best_gain_exp", compared_steps),
            "sc_gain_exp": _series(sc_slice, "best_gain_exp", compared_steps),
            "sc_gain_sc": sc_gain_sc,
            "sc_gain_hybrid": _series(sc_slice, "best_gain_hybrid", compared_steps),
            "mean_sc_gain_sc": float(np.mean(sc_gain_sc)) if sc_gain_sc else 0.0,
        },
        "no_valid_candidate": {
            "empty": [
                int(t["step"]) for t in empty_slice if str(t.get("done_reason", "")) == "no_valid_candidate"
            ],
            "sc": [int(t["step"]) for t in sc_slice if str(t.get("done_reason", "")) == "no_valid_candidate"],
        },
        "outputs": outputs,
    }
    save_json(output_dir / "comparison_summary.json", summary)

    md_lines = [
        "# Stage 4A-6 Comparison To Empty Baseline",
        "",
        f"- empty episode: {summary['empty_episode_id']}",
        f"- SC episode: {summary['sc_episode_id']}",
        f"- compared steps: {compared_steps}",
        f"- empty final observed_ratio: {empty_final_ratio:.6f}",
        f"- SC final observed_ratio: {sc_final_ratio:.6f}",
        f"- observed_ratio delta SC-empty: {summary['observed_ratio_delta_sc_minus_empty']:.6f}",
        f"- changed selected actions: {changed_count}",
        f"- mean score delta SC-empty: {summary['score_difference']['mean_score_delta_sc_minus_empty']:.6f}",
        f"- mean SC gain_sc: {summary['gain_difference']['mean_sc_gain_sc']:.6f}",
        "",
        "If paths match, this stage still passes: the goal is integration correctness, not proving improvement.",
    ]
    (output_dir / "comparison_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SC-aware rollout to empty-prediction baseline.")
    parser.add_argument("--sc_episode_dir", required=True)
    parser.add_argument("--empty_episode_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    summary = run_compare(parse_args())
    print("Stage 4A-6 comparison complete.")
    print(f"compared_steps: {summary['compared_steps']}")
    print(f"empty_final_observed_ratio: {summary['empty_final_observed_ratio']:.6f}")
    print(f"sc_final_observed_ratio: {summary['sc_final_observed_ratio']:.6f}")
    print(f"number_of_changed_selected_actions: {summary['number_of_changed_selected_actions']}")
    print(f"comparison_summary_json: {Path(summary['output_dir']) / 'comparison_summary.json'}")
    print(f"comparison_summary_md: {Path(summary['output_dir']) / 'comparison_summary.md'}")


if __name__ == "__main__":
    main()
