#!/usr/bin/env python3
"""Summarize Stage 4A-6.4 SC gain-gating one-step and rollout ablations."""

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


def _transition_series(transitions: list[dict[str, Any]], key: str, max_steps: int) -> list[float]:
    return [float(t.get(key, 0.0)) for t in transitions[:max_steps]]


def _final_ratio_at(transitions: list[dict[str, Any]], max_steps: int) -> float:
    if not transitions:
        return 0.0
    idx = min(int(max_steps), len(transitions)) - 1
    return float(transitions[idx].get("observed_ratio_after", 0.0))


def _changed_actions(reference: list[dict[str, Any]], other: list[dict[str, Any]], max_steps: int) -> int:
    count = 0
    for ref, item in zip(reference[:max_steps], other[:max_steps]):
        ref_pos = np.asarray(ref.get("selected_next_pose_world", [0, 0, 0]), dtype=np.float64)
        item_pos = np.asarray(item.get("selected_next_pose_world", [0, 0, 0]), dtype=np.float64)
        yaw_delta = abs(float(ref.get("selected_next_yaw", 0.0)) - float(item.get("selected_next_yaw", 0.0)))
        if float(np.linalg.norm(ref_pos - item_pos)) > 1e-4 or yaw_delta > 1e-4:
            count += 1
    return int(count)


def _load_manifest(ablation_dir: Path) -> list[dict[str, Any]]:
    for name in ("ablation_manifest.jsonl", "manifest.jsonl"):
        path = ablation_dir / name
        if path.exists():
            return load_jsonl(path)
    return []


def _episode_row(
    label: str,
    episode_dir: Path,
    summary: dict[str, Any],
    transitions: list[dict[str, Any]],
    empty_transitions: list[dict[str, Any]],
    raw_transitions: list[dict[str, Any]],
    empty_final: float,
    raw_final: float,
    max_steps: int,
    status: str = "ok",
) -> dict[str, Any]:
    final = _final_ratio_at(transitions, max_steps)
    return {
        "config": label,
        "status": status,
        "episode_id": summary.get("episode_id", episode_dir.name),
        "episode_dir": str(episode_dir),
        "sc_gain_formula": summary.get("sc_gain_formula", "raw_count"),
        "sc_occ_threshold": summary.get("sc_occ_threshold"),
        "sc_conf_threshold": summary.get("sc_conf_threshold"),
        "score_gain_mode": summary.get("score_gain_mode", "hybrid_raw"),
        "sc_gain_weight": summary.get("sc_gain_weight", 1.0),
        "sc_gain_cap": summary.get("sc_gain_cap"),
        "alignment_convention": summary.get("alignment_convention"),
        "steps_completed": int(summary.get("steps_completed", len(transitions))),
        "done_reason": summary.get("done_reason", ""),
        "final_observed_ratio": final,
        "delta_vs_empty": float(final - empty_final),
        "delta_vs_raw_sc": float(final - raw_final),
        "changed_actions_vs_empty": _changed_actions(empty_transitions, transitions, max_steps),
        "changed_actions_vs_raw_sc": _changed_actions(raw_transitions, transitions, max_steps),
        "mean_raw_gain_sc": _mean(_transition_series(transitions, "best_gain_sc", max_steps)),
        "mean_effective_gain_sc": _mean(_transition_series(transitions, "best_effective_gain_sc", max_steps)),
        "mean_weighted_gain_sc": _mean(_transition_series(transitions, "best_weighted_gain_sc", max_steps)),
        "mean_gain_exp": _mean(_transition_series(transitions, "best_gain_exp", max_steps)),
        "mean_path_cost": _mean(_transition_series(transitions, "best_path_cost", max_steps)),
        "mean_best_score": _mean(_transition_series(transitions, "best_score", max_steps)),
        "mean_candidates_with_raw_gain_sc_positive": _mean(
            _transition_series(transitions, "candidates_with_gain_sc_positive", max_steps)
        ),
        "mean_candidates_with_effective_gain_sc_positive": _mean(
            _transition_series(transitions, "candidates_with_effective_gain_sc_positive", max_steps)
        ),
        "no_valid_candidate": bool(summary.get("no_valid_candidate_steps")),
        "no_valid_candidate_steps": summary.get("no_valid_candidate_steps", []),
        "checkpoint_modified": bool(summary.get("checkpoint_modified", False)),
        "prediction_writeback": bool(summary.get("prediction_writeback", False)),
        "prediction_used_for_traversability": bool(summary.get("prediction_used_for_traversability", False)),
        "prediction_used_for_collision": bool(summary.get("prediction_used_for_collision", False)),
        "prediction_used_for_a_star": bool(summary.get("prediction_used_for_a_star", False)),
        "prediction_blocks_rays": bool(summary.get("prediction_blocks_rays", False)),
        "prediction_used_for_candidate_reachability": bool(summary.get("prediction_used_for_candidate_reachability", False)),
        "rl_optimizer_training_run": bool(summary.get("rl_optimizer_training_run", False)),
        "rl_or_ppo_training": bool(summary.get("rl_or_ppo_training", False)),
        "optimizer_step": bool(summary.get("optimizer_step", False)),
        "behavior_cloning_training": bool(summary.get("behavior_cloning_training", False)),
        "imitation_learning_training": bool(summary.get("imitation_learning_training", False)),
        "sscnet_training": bool(summary.get("sscnet_training", False)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_final_bar(rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [str(r["config"]) for r in rows]
    values = [float(r["final_observed_ratio"]) for r in rows]
    colors = ["#f97316" if r["config"] == "empty" else "#2563eb" if r["config"] == "raw_fixed_sc" else "#0f766e" for r in rows]
    fig, ax = plt.subplots(figsize=(max(8.5, 1.25 * len(rows)), 4.8), constrained_layout=True)
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("final observed_ratio")
    ax.set_title("Final measured coverage by gating config")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_ratio_curves(curves: dict[str, list[float]], out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    palette = ["#f97316", "#2563eb", "#0f766e", "#7c3aed", "#be123c", "#0369a1", "#a16207"]
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


def _plot_effective_gain(curves: dict[str, list[float]], out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    palette = ["#2563eb", "#0f766e", "#7c3aed", "#be123c", "#0369a1", "#a16207"]
    for idx, (label, values) in enumerate(curves.items()):
        ax.plot(np.arange(len(values)), values, marker="o", color=palette[idx % len(palette)], label=label)
    ax.set_xlabel("step")
    ax.set_ylabel("best effective gain_sc")
    ax.set_title("Effective SC gain curves")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_raw_vs_effective(rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [str(r["config"]) for r in rows if r["config"] not in ("empty",)]
    raw = [0.0 if r.get("mean_raw_gain_sc") is None else float(r["mean_raw_gain_sc"]) for r in rows if r["config"] != "empty"]
    eff = [0.0 if r.get("mean_effective_gain_sc") is None else float(r["mean_effective_gain_sc"]) for r in rows if r["config"] != "empty"]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.5, 1.2 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(x - width / 2, raw, width=width, color="#7c3aed", label="raw")
    ax.bar(x + width / 2, eff, width=width, color="#0f766e", label="effective")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean selected-candidate gain_sc")
    ax.set_title("Raw vs effective SC gain")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_changed_actions(rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [str(r["config"]) for r in rows if r["config"] not in ("empty", "raw_fixed_sc")]
    empty = [int(r["changed_actions_vs_empty"]) for r in rows if r["config"] not in ("empty", "raw_fixed_sc")]
    raw = [int(r["changed_actions_vs_raw_sc"]) for r in rows if r["config"] not in ("empty", "raw_fixed_sc")]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.5, 1.2 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(x - width / 2, empty, width=width, color="#f97316", label="vs empty")
    ax.bar(x + width / 2, raw, width=width, color="#2563eb", label="vs raw SC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("changed actions")
    ax.set_title("Action changes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _plot_selectivity(rows: list[dict[str, Any]], out_path: Path) -> str:
    labels = [str(r["config"]) for r in rows if r["config"] not in ("empty",)]
    raw = [
        0.0 if r.get("mean_candidates_with_raw_gain_sc_positive") is None else float(r["mean_candidates_with_raw_gain_sc_positive"])
        for r in rows
        if r["config"] != "empty"
    ]
    eff = [
        0.0 if r.get("mean_candidates_with_effective_gain_sc_positive") is None else float(r["mean_candidates_with_effective_gain_sc_positive"])
        for r in rows
        if r["config"] != "empty"
    ]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(8.5, 1.2 * len(labels)), 4.8), constrained_layout=True)
    ax.bar(x - width / 2, raw, width=width, color="#7c3aed", label="raw positive candidates")
    ax.bar(x + width / 2, eff, width=width, color="#0f766e", label="effective positive candidates")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("mean candidate count")
    ax.set_title("Prediction gain selectivity")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
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
    map_path: Path | None = None
    for row in rows:
        candidate = Path(row["episode_dir"]) / "observed_state_final.npy"
        if candidate.exists():
            map_path = candidate
            break
    if map_path is None:
        return ""
    observed = np.load(map_path)
    first_label = next(iter(episode_data))
    bounds = _bounds(episode_data[first_label][0], episode_data[first_label][1])
    fig, ax = plt.subplots(figsize=(8.8, 7.8), constrained_layout=True)
    ax.imshow(
        project_topdown(observed).T,
        origin="lower",
        extent=[bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]],
        cmap=STATE_CMAP,
        norm=STATE_NORM,
    )
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
    ax.set_title("Selected paths across gating configs")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def _candidate_value(candidate: dict[str, Any], key: str) -> float:
    if key in candidate:
        return float(candidate[key])
    for group in ("gains", "utilities"):
        values = candidate.get(group, {})
        if key in values:
            return float(values[key])
    return 0.0


def _load_candidates(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize_one_step(one_step_dir: Path, output_dir: Path) -> dict[str, Any]:
    if not one_step_dir.exists():
        return {"available": False, "rows": []}
    cases: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for decision_path in sorted(one_step_dir.glob("*/expert_step_decision.json")):
        label = decision_path.parent.name
        cases[label] = (load_json(decision_path), _load_candidates(decision_path.parent / "expert_step_candidates.jsonl"))
    if not cases:
        return {"available": False, "rows": []}
    empty_best = cases.get("empty_baseline", (None, []))[0]
    raw_best = cases.get("raw_count_regression", (None, []))[0]
    empty_id = int(empty_best["best_candidate"]["id"]) if empty_best else None
    raw_id = int(raw_best["best_candidate"]["id"]) if raw_best else None
    empty_top = {int(c["id"]) for c in empty_best.get("top_candidates", [])} if empty_best else set()
    raw_top = {int(c["id"]) for c in raw_best.get("top_candidates", [])} if raw_best else set()
    rows: list[dict[str, Any]] = []
    for label, (decision, candidates) in sorted(cases.items()):
        best = decision["best_candidate"]
        gain_sc = np.asarray([_candidate_value(c, "gain_sc") for c in candidates], dtype=np.float64)
        effective = np.asarray(
            [_candidate_value(c, "effective_gain_sc") if "effective_gain_sc" in c or "gains" in c else _candidate_value(c, "gain_sc") for c in candidates],
            dtype=np.float64,
        )
        top_ids = {int(c["id"]) for c in decision.get("top_candidates", [])}
        row = {
            "formula": label,
            "sc_gain_formula": decision.get("sc_gain_formula", decision.get("diagnostics", {}).get("sc_gain_formula")),
            "best_candidate_id": int(best["id"]),
            "best_score": float(best["final_score"]),
            "best_gain_exp": _candidate_value(best, "gain_exp"),
            "best_raw_gain_sc": _candidate_value(best, "gain_sc"),
            "best_effective_gain_sc": _candidate_value(best, "effective_gain_sc"),
            "raw_gain_sc_mean": float(np.mean(gain_sc)) if gain_sc.size else 0.0,
            "raw_gain_sc_max": float(np.max(gain_sc)) if gain_sc.size else 0.0,
            "effective_gain_sc_mean": float(np.mean(effective)) if effective.size else 0.0,
            "effective_gain_sc_max": float(np.max(effective)) if effective.size else 0.0,
            "candidates_with_effective_gain_sc_gt_zero": int(np.count_nonzero(effective > 0.0)),
            "changed_vs_empty": None if empty_id is None else int(best["id"]) != empty_id,
            "changed_vs_raw_sc": None if raw_id is None else int(best["id"]) != raw_id,
            "top_n_overlap_with_empty": len(top_ids & empty_top) if empty_top else None,
            "top_n_overlap_with_raw_sc": len(top_ids & raw_top) if raw_top else None,
            "output_dir": str(one_step_dir / label),
        }
        rows.append(row)
    _write_csv(output_dir / "one_step_gating_table.csv", rows)
    save_json(output_dir / "one_step_gating_summary.json", {"available": True, "rows": rows})
    md = ["# One-Step SC Gain Gating Summary", ""]
    md.append("| formula | best | raw mean/max | effective mean/max | eff>0 | changed empty | changed raw |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        md.append(
            f"| {row['formula']} | {row['best_candidate_id']} | "
            f"{row['raw_gain_sc_mean']} / {row['raw_gain_sc_max']} | "
            f"{row['effective_gain_sc_mean']} / {row['effective_gain_sc_max']} | "
            f"{row['candidates_with_effective_gain_sc_gt_zero']} | "
            f"{row['changed_vs_empty']} | {row['changed_vs_raw_sc']} |"
        )
    (output_dir / "one_step_gating_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"available": True, "rows": rows}


def run_summary(args: argparse.Namespace) -> dict[str, Any]:
    gating_root = Path(args.gating_root).resolve()
    ablation_dir = gating_root / "ablation"
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_steps = int(args.max_steps)

    empty_dir = Path(args.empty_episode_dir).resolve()
    raw_dir = Path(args.original_sc_episode_dir).resolve()
    empty_summary = load_json(empty_dir / "episode_summary.json")
    raw_summary = load_json(raw_dir / "episode_summary.json")
    empty_transitions = load_jsonl(empty_dir / "transitions.jsonl")
    raw_transitions = load_jsonl(raw_dir / "transitions.jsonl")
    empty_final = _final_ratio_at(empty_transitions, max_steps)
    raw_final = _final_ratio_at(raw_transitions, max_steps)

    episode_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {
        "empty": (empty_summary, empty_transitions),
        "raw_fixed_sc": (raw_summary, raw_transitions),
    }
    rows = [
        _episode_row("empty", empty_dir, empty_summary, empty_transitions, empty_transitions, raw_transitions, empty_final, raw_final, max_steps),
        _episode_row("raw_fixed_sc", raw_dir, raw_summary, raw_transitions, empty_transitions, raw_transitions, empty_final, raw_final, max_steps),
    ]

    manifest = _load_manifest(ablation_dir)
    for record in manifest:
        label = str(record.get("config_name", record.get("episode_id", "unknown")))
        episode_dir = Path(record.get("episode_dir", "")).resolve()
        status = str(record.get("status", "failed"))
        if status == "ok" and (episode_dir / "episode_summary.json").exists():
            summary = load_json(episode_dir / "episode_summary.json")
            transitions = load_jsonl(episode_dir / "transitions.jsonl")
            rows.append(
                _episode_row(
                    label,
                    episode_dir,
                    summary,
                    transitions,
                    empty_transitions,
                    raw_transitions,
                    empty_final,
                    raw_final,
                    max_steps,
                    status=status,
                )
            )
            episode_data[label] = (summary, transitions)
        else:
            rows.append(
                {
                    "config": label,
                    "status": status,
                    "episode_id": record.get("episode_id"),
                    "episode_dir": str(episode_dir),
                    "final_observed_ratio": None,
                    "delta_vs_empty": None,
                    "delta_vs_raw_sc": None,
                    "changed_actions_vs_empty": None,
                    "changed_actions_vs_raw_sc": None,
                    "no_valid_candidate": True,
                    "error": record.get("error"),
                }
            )

    completed = [row for row in rows if row.get("status") == "ok" and row["config"] not in ("empty", "raw_fixed_sc")]
    improving_raw = [row for row in completed if float(row.get("delta_vs_raw_sc") or 0.0) > 1e-12]
    beating_empty = [row for row in completed if float(row.get("delta_vs_empty") or 0.0) > 1e-12]
    changed = [row for row in completed if int(row.get("changed_actions_vs_raw_sc") or 0) > 0]
    best_row = max(completed, key=lambda r: float(r.get("final_observed_ratio") or -1.0), default=None)
    if best_row is None:
        recommendation = "No gating config completed; fix runner failures before the next stage."
    elif beating_empty:
        winner = max(beating_empty, key=lambda r: float(r["final_observed_ratio"]))
        recommendation = f"{winner['config']} beats the empty baseline in this 5-step smoke; run one cautious 10-step gated SC-aware smoke next."
    elif improving_raw:
        winner = max(improving_raw, key=lambda r: float(r["final_observed_ratio"]))
        recommendation = f"{winner['config']} improves over raw SC but does not beat empty; inspect qualitative selected actions before rollout scaling."
    elif changed:
        recommendation = "Gating changes at least one action but still underperforms; inspect qualitative behavior before scaling."
    else:
        recommendation = "No completed gating config changed raw SC behavior enough; prediction gain remains insufficiently selective for this scene."

    curves = {
        label: [float(t.get("observed_ratio_after", 0.0)) for t in transitions[:max_steps]]
        for label, (_, transitions) in episode_data.items()
    }
    eff_curves = {
        label: [float(t.get("best_effective_gain_sc", t.get("best_gain_sc", 0.0))) for t in transitions[:max_steps]]
        for label, (_, transitions) in episode_data.items()
        if label != "empty"
    }
    plot_paths = {
        "observed_ratio_curve_all": _plot_ratio_curves(curves, output_dir / "observed_ratio_curve_all.png"),
        "final_observed_ratio_bar": _plot_final_bar(rows, output_dir / "final_observed_ratio_bar.png"),
        "effective_gain_sc_curve_all": _plot_effective_gain(eff_curves, output_dir / "effective_gain_sc_curve_all.png"),
        "raw_vs_effective_gain_sc": _plot_raw_vs_effective(rows, output_dir / "raw_vs_effective_gain_sc.png"),
        "changed_actions_bar": _plot_changed_actions(rows, output_dir / "changed_actions_bar.png"),
        "selectivity_bar": _plot_selectivity(rows, output_dir / "selectivity_bar.png"),
        "path_topdown_all": _plot_path_all(rows, episode_data, max_steps, output_dir / "path_topdown_all.png"),
    }

    one_step_summary = summarize_one_step(gating_root / "one_step", output_dir)
    _write_csv(output_dir / "gain_gating_table.csv", rows)
    result = {
        "stage": "Stage 4A-6.4 SC gain gating summary",
        "gating_root": str(gating_root),
        "empty_episode_dir": str(empty_dir),
        "original_sc_episode_dir": str(raw_dir),
        "max_steps": max_steps,
        "empty_final_observed_ratio": empty_final,
        "raw_fixed_sc_final_observed_ratio": raw_final,
        "rows": rows,
        "completed_configs": [row["config"] for row in completed],
        "any_config_improves_over_raw_sc": bool(improving_raw),
        "any_config_beats_empty": bool(beating_empty),
        "any_config_changes_actions_vs_raw_sc": bool(changed),
        "best_config": None if best_row is None else best_row["config"],
        "best_config_final_observed_ratio": None if best_row is None else best_row["final_observed_ratio"],
        "recommendation": recommendation,
        "one_step": one_step_summary,
        "outputs": {
            **plot_paths,
            "gain_gating_summary_json": str(output_dir / "gain_gating_summary.json"),
            "gain_gating_summary_md": str(output_dir / "gain_gating_summary.md"),
            "gain_gating_table_csv": str(output_dir / "gain_gating_table.csv"),
            "recommendation_next_step": str(output_dir / "recommendation_next_step.md"),
        },
        "prediction_safety_flags": {
            key: any(bool(row.get(key, False)) for row in rows)
            for key in (
                "checkpoint_modified",
                "prediction_writeback",
                "prediction_used_for_traversability",
                "prediction_used_for_collision",
                "prediction_used_for_a_star",
                "prediction_blocks_rays",
                "prediction_used_for_candidate_reachability",
                "rl_optimizer_training_run",
                "rl_or_ppo_training",
                "optimizer_step",
                "behavior_cloning_training",
                "imitation_learning_training",
                "sscnet_training",
            )
        },
    }
    save_json(output_dir / "gain_gating_summary.json", result)
    md = [
        "# Stage 4A-6.4 Gain Gating Summary",
        "",
        f"- empty baseline final observed_ratio: `{empty_final}`",
        f"- fixed raw SC final observed_ratio: `{raw_final}`",
        f"- any config improves over raw SC: `{bool(improving_raw)}`",
        f"- any config beats empty baseline: `{bool(beating_empty)}`",
        f"- any config changes actions vs raw SC: `{bool(changed)}`",
        f"- recommendation: {recommendation}",
        "",
        "| config | status | final observed_ratio | delta vs empty | delta vs raw SC | changed vs empty | changed vs raw | mean raw gain_sc | mean effective gain_sc | mean path_cost | no_valid |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        md.append(
            f"| {row.get('config')} | {row.get('status')} | {row.get('final_observed_ratio')} | "
            f"{row.get('delta_vs_empty')} | {row.get('delta_vs_raw_sc')} | "
            f"{row.get('changed_actions_vs_empty')} | {row.get('changed_actions_vs_raw_sc')} | "
            f"{row.get('mean_raw_gain_sc')} | {row.get('mean_effective_gain_sc')} | "
            f"{row.get('mean_path_cost')} | {row.get('no_valid_candidate')} |"
        )
    (output_dir / "gain_gating_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (output_dir / "recommendation_next_step.md").write_text(recommendation + "\n", encoding="utf-8")
    print("Stage 4A-6.4 gain gating summary complete.")
    print(f"output_dir: {output_dir}")
    print(f"recommendation: {recommendation}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gating_root", required=True)
    parser.add_argument("--empty_episode_dir", required=True)
    parser.add_argument("--original_sc_episode_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_summary(parse_args())


if __name__ == "__main__":
    main()
