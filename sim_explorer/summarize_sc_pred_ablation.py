#!/usr/bin/env python3
"""Summarize Stage 4A-6.1 SC prediction ablation rollouts."""

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
from PIL import Image, ImageDraw

from sim_rollout_utils import load_json, load_jsonl, save_json
from visualize_sim_rollout import STATE_CMAP, STATE_NORM, project_topdown


def _mean(values: list[float]) -> float | None:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else None


def _load_manifest(ablation_dir: Path) -> list[dict[str, Any]]:
    path = ablation_dir / "ablation_manifest.jsonl"
    if not path.exists():
        path = ablation_dir / "manifest.jsonl"
    return load_jsonl(path)


def _transition_series(transitions: list[dict[str, Any]], key: str, max_steps: int) -> list[float]:
    return [float(t.get(key, 0.0)) for t in transitions[:max_steps]]


def _final_ratio_at(transitions: list[dict[str, Any]], max_steps: int) -> float:
    if not transitions:
        return 0.0
    idx = min(int(max_steps), len(transitions)) - 1
    return float(transitions[idx].get("observed_ratio_after", 0.0))


def _changed_actions(empty: list[dict[str, Any]], other: list[dict[str, Any]], max_steps: int) -> int:
    count = 0
    for e, o in zip(empty[:max_steps], other[:max_steps]):
        e_pos = np.asarray(e.get("selected_next_pose_world", [0, 0, 0]), dtype=np.float64)
        o_pos = np.asarray(o.get("selected_next_pose_world", [0, 0, 0]), dtype=np.float64)
        if float(np.linalg.norm(e_pos - o_pos)) > 1e-4 or abs(float(e.get("selected_next_yaw", 0.0)) - float(o.get("selected_next_yaw", 0.0))) > 1e-4:
            count += 1
    return int(count)


def _episode_row(
    label: str,
    episode_dir: Path,
    summary: dict[str, Any],
    transitions: list[dict[str, Any]],
    empty_transitions: list[dict[str, Any]],
    empty_final: float,
    original_sc_final: float,
    max_steps: int,
    status: str = "ok",
) -> dict[str, Any]:
    final = _final_ratio_at(transitions, max_steps)
    return {
        "config": label,
        "status": status,
        "episode_id": summary.get("episode_id", episode_dir.name),
        "episode_dir": str(episode_dir),
        "prediction_mode": summary.get("prediction_mode"),
        "tau": summary.get("tau"),
        "score_gain_mode": summary.get("score_gain_mode", "hybrid_raw"),
        "sc_gain_weight": summary.get("sc_gain_weight", 1.0),
        "sc_gain_cap": summary.get("sc_gain_cap"),
        "steps_completed": int(summary.get("steps_completed", len(transitions))),
        "done_reason": summary.get("done_reason", ""),
        "final_observed_ratio": final,
        "delta_vs_empty": float(final - empty_final),
        "delta_vs_original_sc": float(final - original_sc_final),
        "mean_gain_exp": _mean(_transition_series(transitions, "best_gain_exp", max_steps)),
        "mean_gain_sc": _mean(_transition_series(transitions, "best_gain_sc", max_steps)),
        "mean_weighted_gain_sc": _mean(_transition_series(transitions, "best_weighted_gain_sc", max_steps)),
        "mean_path_cost": _mean(_transition_series(transitions, "best_path_cost", max_steps)),
        "mean_best_score": _mean(_transition_series(transitions, "best_score", max_steps)),
        "changed_actions_vs_empty": _changed_actions(empty_transitions, transitions, max_steps),
        "no_valid_candidate": bool(summary.get("no_valid_candidate_steps")),
        "no_valid_candidate_steps": summary.get("no_valid_candidate_steps", []),
        "total_wall_time": summary.get("total_wall_time"),
        "average_map_predict_inference_time": summary.get("average_map_predict_inference_time"),
        "average_map_predict_total_time": summary.get("average_map_predict_total_time"),
        "gpu_memory_peak": summary.get("gpu_memory_peak"),
        "checkpoint_modified": bool(summary.get("checkpoint_modified", False)),
        "prediction_writeback": bool(summary.get("prediction_writeback", False)),
        "prediction_used_for_traversability": bool(summary.get("prediction_used_for_traversability", False)),
        "prediction_used_for_collision": bool(summary.get("prediction_used_for_collision", False)),
        "prediction_used_for_a_star": bool(summary.get("prediction_used_for_a_star", False)),
        "prediction_blocks_rays": bool(summary.get("prediction_blocks_rays", False)),
        "prediction_used_for_candidate_reachability": bool(summary.get("prediction_used_for_candidate_reachability", False)),
        "rl_optimizer_training_run": bool(summary.get("rl_optimizer_training_run", False)),
        "sscnet_training": bool(summary.get("sscnet_training", False)),
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


def _plot_final_bar(rows: list[dict[str, Any]], empty_final: float, original_sc_final: float, out_path: Path) -> str:
    labels = [str(r["config"]) for r in rows]
    values = [float(r["final_observed_ratio"]) for r in rows]
    colors = ["#f97316" if r["config"] == "empty" else "#2563eb" if r["config"] == "original_sc" else "#0f766e" for r in rows]
    fig, ax = plt.subplots(figsize=(max(8.5, 1.2 * len(rows)), 4.8), constrained_layout=True)
    ax.bar(labels, values, color=colors)
    ax.axhline(empty_final, color="#f97316", linestyle="--", linewidth=1.2, label="empty baseline")
    ax.axhline(original_sc_final, color="#2563eb", linestyle=":", linewidth=1.2, label="original SC")
    ax.set_ylabel("final observed_ratio")
    ax.set_title("Final measured coverage by config")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_ratio_curves(
    curves: dict[str, list[float]],
    empty_curve: list[float],
    out_path: Path,
) -> str:
    fig, ax = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    steps = np.arange(len(empty_curve), dtype=np.int32)
    ax.plot(steps, empty_curve, marker="o", linewidth=2.2, color="#f97316", label="empty")
    palette = ["#2563eb", "#0f766e", "#7c3aed", "#be123c", "#0369a1", "#a16207"]
    for idx, (label, values) in enumerate(curves.items()):
        ax.plot(np.arange(len(values)), values, marker="o", color=palette[idx % len(palette)], label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("observed_ratio_after")
    ax.set_title("Measured coverage curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_scatter(rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if x is None or y is None:
            continue
        ax.scatter(float(x), float(y), s=90, label=str(row["config"]))
        ax.text(float(x), float(y), str(row["config"]), fontsize=7, ha="left", va="bottom")
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_action_change(rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [str(r["config"]) for r in rows if r["config"] != "empty"]
    values = [int(r["changed_actions_vs_empty"]) for r in rows if r["config"] != "empty"]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.2 * len(labels)), 4.5), constrained_layout=True)
    ax.bar(labels, values, color="#2563eb")
    ax.set_ylabel("changed actions vs empty")
    ax.set_title("Action divergence from measured-only baseline")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _bounds(summary: dict[str, Any], transitions: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    raw = summary.get("map_bounds") or (transitions[0].get("bounds") if transitions else None)
    if raw is None:
        raw = {"x": [-6, 6], "y": [-6, 6], "z": [0, 3]}
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def _plot_path_all(
    rows: list[dict[str, Any]],
    episode_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    max_steps: int,
    out_path: Path,
) -> str:
    base_label = "original_sc" if "original_sc" in episode_data else next(iter(episode_data))
    base_summary, base_transitions = episode_data[base_label]
    base_dir = Path(rows[[r["config"] for r in rows].index(base_label)]["episode_dir"]) if base_label in [r["config"] for r in rows] else None
    map_path = base_dir / "observed_state_final.npy" if base_dir else None
    if map_path is None or not map_path.exists():
        for row in rows:
            candidate = Path(row["episode_dir"]) / "observed_state_final.npy"
            if candidate.exists():
                map_path = candidate
                break
    if map_path is None or not map_path.exists():
        return ""
    observed = np.load(map_path)
    bounds = _bounds(base_summary, base_transitions)
    fig, ax = plt.subplots(figsize=(8.8, 7.8), constrained_layout=True)
    ax.imshow(project_topdown(observed).T, origin="lower", extent=[bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]], cmap=STATE_CMAP, norm=STATE_NORM)
    palette = ["#f97316", "#2563eb", "#0f766e", "#7c3aed", "#be123c", "#0369a1", "#a16207"]
    for idx, row in enumerate(rows):
        label = str(row["config"])
        if label not in episode_data:
            continue
        _, transitions = episode_data[label]
        if not transitions:
            continue
        path = [np.asarray(transitions[0]["current_pose_world"][:2], dtype=np.float64)]
        path.extend(np.asarray(t["selected_next_pose_world"][:2], dtype=np.float64) for t in transitions[:max_steps])
        arr = np.stack(path, axis=0)
        ax.plot(arr[:, 0], arr[:, 1], "-o", linewidth=1.8, markersize=3.7, color=palette[idx % len(palette)], label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Selected paths across ablations")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _open_image(path: Path, size: tuple[int, int]) -> Image.Image:
    if path.exists():
        img = Image.open(path).convert("RGB")
        img.thumbnail(size)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(img, ((size[0] - img.width) // 2, (size[1] - img.height) // 2))
        return canvas
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), f"missing\n{path.name}", fill=(0, 0, 0))
    return canvas


def _contact_sheet(items: list[tuple[str, Path]], out_path: Path, thumb: tuple[int, int] = (320, 260), columns: int = 5) -> str:
    if not items:
        return ""
    rows = int(math.ceil(len(items) / float(columns)))
    sheet = Image.new("RGB", (columns * thumb[0], rows * (thumb[1] + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, (label, path) in enumerate(items):
        col = idx % columns
        row = idx // columns
        x = col * thumb[0]
        y = row * (thumb[1] + 28)
        draw.text((x + 6, y + 6), label, fill=(0, 0, 0))
        sheet.paste(_open_image(path, thumb), (x, y + 28))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return str(out_path)


def _qualitative(
    qualitative_dir: Path,
    empty_dir: Path,
    original_sc_dir: Path,
    best_dir: Path | None,
    rows: list[dict[str, Any]],
    max_steps: int,
) -> dict[str, str]:
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    best_name = best_dir.name if best_dir else "none"
    topdown_items: list[tuple[str, Path]] = []
    for step in range(max_steps):
        topdown_items.append((f"empty s{step}", empty_dir / f"step_topdown_{step:03d}.png"))
        topdown_items.append((f"orig SC s{step}", original_sc_dir / f"step_topdown_{step:03d}.png"))
        if best_dir is not None:
            topdown_items.append((f"best s{step}", best_dir / f"step_topdown_{step:03d}.png"))
    prediction_items: list[tuple[str, Path]] = []
    unmeasured_items: list[tuple[str, Path]] = []
    for label, ep in (("orig", original_sc_dir), ("best", best_dir)):
        if ep is None:
            continue
        for step in range(max_steps):
            prediction_items.append((f"{label} s{step}", ep / f"prediction_step{step:03d}" / "observed_vs_prediction_topdown.png"))
            unmeasured_items.append((f"{label} s{step}", ep / f"prediction_step{step:03d}" / "prediction_not_measured_topdown.png"))
    selected_items = []
    for label, ep in (("empty", empty_dir), ("orig SC", original_sc_dir), ("best", best_dir)):
        if ep is not None:
            selected_items.append((label, ep / "rollout_topdown_path.png"))

    outputs = {
        "step0_to_step4_empty_vs_sc_topdown": _contact_sheet(
            topdown_items,
            qualitative_dir / "step0_to_step4_empty_vs_sc_topdown.png",
            thumb=(300, 250),
            columns=5,
        ),
        "step0_to_step4_prediction_overlay": _contact_sheet(
            prediction_items,
            qualitative_dir / "step0_to_step4_prediction_overlay.png",
            thumb=(300, 250),
            columns=5,
        ),
        "step0_to_step4_selected_actions": _contact_sheet(
            selected_items,
            qualitative_dir / "step0_to_step4_selected_actions.png",
            thumb=(360, 310),
            columns=3,
        ),
        "step0_to_step4_predicted_unmeasured": _contact_sheet(
            unmeasured_items,
            qualitative_dir / "step0_to_step4_predicted_unmeasured.png",
            thumb=(300, 250),
            columns=5,
        ),
    }

    best_row = next((r for r in rows if best_dir is not None and Path(r["episode_dir"]) == best_dir), None)
    original = next((r for r in rows if r["config"] == "original_sc"), None)
    empty = next((r for r in rows if r["config"] == "empty"), None)
    note_lines = [
        "# Stage 4A-6.1 Qualitative Notes",
        "",
        f"- best inspected ablation: {best_name}",
        f"- empty final observed_ratio: {empty.get('final_observed_ratio') if empty else None}",
        f"- original SC final observed_ratio: {original.get('final_observed_ratio') if original else None}",
        f"- best ablation final observed_ratio: {best_row.get('final_observed_ratio') if best_row else None}",
        "",
        "## Findings",
        "",
        "- SC often assigns positive prediction gain to most reachable candidates, so gain_sc can become dense rather than selective.",
        "- Raw hybrid can duplicate exploration gain because an unknown voxel may count once as gain_exp and again as gain_sc.",
        "- Low path_cost local candidates can receive very high utility when gain_hybrid is inflated.",
        "- If tuned weight/cap/tau improves coverage, the issue is likely over-strong prediction scoring rather than rollout plumbing.",
        "- If all tuned settings still underperform, preprocessing/alignment/domain shift should be inspected before longer rollouts.",
        "",
        "Prediction was only used for information gain in these outputs; it was not used for observed_state, A*, collision, reachability, or ray blocking.",
    ]
    notes_path = qualitative_dir / "qualitative_notes.md"
    notes_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    outputs["qualitative_notes"] = str(notes_path)
    return outputs


def _write_md(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    lines = [
        "# Stage 4A-6.1 Ablation Summary",
        "",
        f"- empty baseline final observed_ratio: {summary['empty_final_observed_ratio']:.9f}",
        f"- original SC final observed_ratio: {summary['original_sc_final_observed_ratio']:.9f}",
        f"- best config: {summary['best_config']}",
        f"- any ablation beats empty: {summary['any_ablation_beats_empty']}",
        f"- any ablation improves over original SC: {summary['any_ablation_improves_over_original_sc']}",
        "",
        "| config | status | final observed_ratio | delta vs empty | delta vs original SC | mean gain_exp | mean gain_sc | mean weighted_gain_sc | mean path_cost | changed actions | no_valid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['status']} | {float(row['final_observed_ratio']):.9f} | "
            f"{float(row['delta_vs_empty']):.9f} | {float(row['delta_vs_original_sc']):.9f} | "
            f"{row['mean_gain_exp']} | {row['mean_gain_sc']} | {row['mean_weighted_gain_sc']} | "
            f"{row['mean_path_cost']} | {row['changed_actions_vs_empty']} | {row['no_valid_candidate']} |"
        )
    lines.extend(["", "## Recommendation", "", summary["recommendation"]])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_summary(args: argparse.Namespace) -> dict[str, Any]:
    ablation_dir = Path(args.ablation_dir).resolve()
    empty_dir = Path(args.empty_episode_dir).resolve()
    original_sc_dir = Path(args.existing_sc_episode_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(args.max_steps)

    empty_summary = load_json(empty_dir / "episode_summary.json")
    original_sc_summary = load_json(original_sc_dir / "episode_summary.json")
    empty_transitions = load_jsonl(empty_dir / "transitions.jsonl")
    original_sc_transitions = load_jsonl(original_sc_dir / "transitions.jsonl")
    empty_final = _final_ratio_at(empty_transitions, max_steps)
    original_sc_final = _final_ratio_at(original_sc_transitions, max_steps)

    episode_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
        "empty": (empty_summary, empty_transitions),
        "original_sc": (original_sc_summary, original_sc_transitions),
    }
    rows = [
        _episode_row(
            "empty",
            empty_dir,
            empty_summary,
            empty_transitions,
            empty_transitions,
            empty_final,
            original_sc_final,
            max_steps,
        ),
        _episode_row(
            "original_sc",
            original_sc_dir,
            original_sc_summary,
            original_sc_transitions,
            empty_transitions,
            empty_final,
            original_sc_final,
            max_steps,
        ),
    ]

    manifest = _load_manifest(ablation_dir)
    for record in manifest:
        if str(record.get("status")) != "ok":
            rows.append(
                {
                    "config": record.get("config_name", record.get("episode_id", "failed")),
                    "status": str(record.get("status", "failed")),
                    "episode_id": record.get("episode_id"),
                    "episode_dir": record.get("episode_dir"),
                    "prediction_mode": record.get("prediction_mode"),
                    "tau": record.get("tau"),
                    "score_gain_mode": record.get("score_gain_mode"),
                    "sc_gain_weight": record.get("sc_gain_weight"),
                    "sc_gain_cap": record.get("sc_gain_cap"),
                    "steps_completed": int(record.get("steps_completed", 0)),
                    "done_reason": record.get("done_reason", "failed"),
                    "final_observed_ratio": float(record.get("observed_ratio_end", 0.0)),
                    "delta_vs_empty": float(record.get("observed_ratio_end", 0.0)) - empty_final,
                    "delta_vs_original_sc": float(record.get("observed_ratio_end", 0.0)) - original_sc_final,
                    "mean_gain_exp": None,
                    "mean_gain_sc": None,
                    "mean_weighted_gain_sc": None,
                    "mean_path_cost": None,
                    "mean_best_score": None,
                    "changed_actions_vs_empty": 0,
                    "no_valid_candidate": True,
                    "no_valid_candidate_steps": [],
                    "total_wall_time": record.get("wall_time"),
                    "average_map_predict_inference_time": record.get("average_map_predict_inference_time"),
                    "average_map_predict_total_time": record.get("average_map_predict_total_time"),
                    "gpu_memory_peak": record.get("gpu_memory_peak"),
                    "checkpoint_modified": bool(record.get("checkpoint_modified", False)),
                    "prediction_writeback": False,
                    "prediction_used_for_traversability": False,
                    "prediction_used_for_collision": False,
                    "prediction_used_for_a_star": False,
                    "prediction_blocks_rays": False,
                    "prediction_used_for_candidate_reachability": False,
                    "rl_optimizer_training_run": False,
                    "sscnet_training": False,
                }
            )
            continue
        ep_dir = Path(record["episode_dir"]).resolve()
        summary = load_json(ep_dir / "episode_summary.json")
        transitions = load_jsonl(ep_dir / "transitions.jsonl")
        label = str(record.get("config_name", summary.get("episode_id", ep_dir.name)))
        episode_data[label] = (summary, transitions)
        rows.append(_episode_row(label, ep_dir, summary, transitions, empty_transitions, empty_final, original_sc_final, max_steps))

    ablation_rows = [r for r in rows if r["config"] not in ("empty", "original_sc") and r["status"] == "ok"]
    best_row = max(ablation_rows, key=lambda r: float(r["final_observed_ratio"]), default=None)
    any_beats_empty = any(float(r["final_observed_ratio"]) > empty_final for r in ablation_rows)
    any_improves_original = any(float(r["final_observed_ratio"]) > original_sc_final for r in ablation_rows)

    if best_row is None:
        recommendation = "No ablation completed; fix sweep execution before interpreting tuning."
    elif any_beats_empty:
        recommendation = (
            f"{best_row['config']} beats the 5-step empty baseline here. Next should be a 10-step tuned SC-aware rollout, still without RL."
        )
    elif any_improves_original:
        recommendation = (
            f"{best_row['config']} improves over the original SC rollout but does not beat empty. Run one cautious 10-step tuned check or inspect alignment before scaling."
        )
    else:
        recommendation = (
            "All completed SC settings remain below the empty baseline. Stage 4A-6.2 should inspect map_predict preprocessing, alignment, and domain shift before longer rollouts."
        )

    _write_csv(output_dir / "ablation_table.csv", rows)
    curves = {
        "original_sc": _transition_series(original_sc_transitions, "observed_ratio_after", max_steps),
    }
    for row in ablation_rows:
        label = str(row["config"])
        _, transitions = episode_data[label]
        curves[label] = _transition_series(transitions, "observed_ratio_after", max_steps)

    outputs = {
        "observed_ratio_final_bar": _plot_final_bar(rows, empty_final, original_sc_final, output_dir / "observed_ratio_final_bar.png"),
        "observed_ratio_curve_all": _plot_ratio_curves(
            curves,
            _transition_series(empty_transitions, "observed_ratio_after", max_steps),
            output_dir / "observed_ratio_curve_all.png",
        ),
        "gain_sc_vs_delta_observed": _plot_scatter(
            rows,
            "mean_gain_sc",
            "delta_vs_empty",
            "Mean gain_sc vs final coverage delta",
            output_dir / "gain_sc_vs_delta_observed.png",
        ),
        "score_vs_observed_ratio": _plot_scatter(
            rows,
            "mean_best_score",
            "final_observed_ratio",
            "Mean best_score vs final observed_ratio",
            output_dir / "score_vs_observed_ratio.png",
        ),
        "action_change_count_bar": _plot_action_change(rows, output_dir / "action_change_count_bar.png"),
        "path_topdown_all": _plot_path_all(rows, episode_data, max_steps, output_dir / "path_topdown_all.png"),
    }

    qualitative_dir = Path(args.qualitative_dir).resolve()
    best_dir = Path(best_row["episode_dir"]).resolve() if best_row is not None and Path(best_row["episode_dir"]).exists() else None
    outputs["qualitative"] = _qualitative(qualitative_dir, empty_dir, original_sc_dir, best_dir, rows, max_steps)

    summary = {
        "stage": "Stage 4A-6.1 SC prediction ablation summary",
        "ablation_dir": str(ablation_dir),
        "output_dir": str(output_dir),
        "qualitative_dir": str(qualitative_dir),
        "empty_final_observed_ratio": float(empty_final),
        "original_sc_final_observed_ratio": float(original_sc_final),
        "rows": rows,
        "best_config": None if best_row is None else best_row["config"],
        "best_row": best_row,
        "any_ablation_beats_empty": bool(any_beats_empty),
        "any_ablation_improves_over_original_sc": bool(any_improves_original),
        "completed_ablation_count": int(len(ablation_rows)),
        "prediction_safety_flags": {
            "prediction_writeback_any": any(bool(r.get("prediction_writeback", False)) for r in rows),
            "prediction_used_for_traversability_any": any(bool(r.get("prediction_used_for_traversability", False)) for r in rows),
            "prediction_used_for_collision_any": any(bool(r.get("prediction_used_for_collision", False)) for r in rows),
            "prediction_used_for_a_star_any": any(bool(r.get("prediction_used_for_a_star", False)) for r in rows),
            "prediction_blocks_rays_any": any(bool(r.get("prediction_blocks_rays", False)) for r in rows),
            "rl_or_training_any": any(
                bool(r.get("rl_optimizer_training_run", False)) or bool(r.get("sscnet_training", False)) for r in rows
            ),
            "checkpoint_modified_any": any(bool(r.get("checkpoint_modified", False)) for r in rows),
        },
        "recommendation": recommendation,
        "outputs": outputs,
    }
    save_json(output_dir / "ablation_summary.json", summary)
    _write_md(output_dir / "ablation_summary.md", summary)
    (output_dir / "recommendation_for_next_rollout.md").write_text(recommendation + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 4A-6.1 ablation outputs.")
    parser.add_argument("--ablation_dir", required=True)
    parser.add_argument("--empty_episode_dir", required=True)
    parser.add_argument("--existing_sc_episode_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument(
        "--qualitative_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a61_analysis/qualitative_inspection",
    )
    return parser.parse_args()


def main() -> None:
    summary = run_summary(parse_args())
    print("Stage 4A-6.1 ablation summary complete.")
    print(f"empty_final_observed_ratio: {summary['empty_final_observed_ratio']:.9f}")
    print(f"original_sc_final_observed_ratio: {summary['original_sc_final_observed_ratio']:.9f}")
    print(f"best_config: {summary['best_config']}")
    print(f"any_ablation_beats_empty: {summary['any_ablation_beats_empty']}")
    print(f"summary_json: {Path(summary['output_dir']) / 'ablation_summary.json'}")
    print(f"recommendation: {Path(summary['output_dir']) / 'recommendation_for_next_rollout.md'}")


if __name__ == "__main__":
    main()
