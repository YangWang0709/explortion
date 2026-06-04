#!/usr/bin/env python3
"""Summarize Stage 4A-4 multi-episode rollout datasets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATASET_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar"


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {"min": float(arr.min()), "mean": float(arr.mean()), "max": float(arr.max())}


def value_from_transition(transition: dict[str, Any], preferred: str, fallback: str | None = None) -> float | None:
    if preferred in transition:
        return float(transition[preferred])
    if fallback and fallback in transition:
        return float(transition[fallback])
    return None


def dedupe_manifest(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for record in records:
        episode_id = str(record.get("episode_id", ""))
        if episode_id:
            deduped[episode_id] = record
    return deduped


def episode_dirs(dataset_dir: Path) -> list[Path]:
    root = dataset_dir / "episodes"
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def collect_episodes(dataset_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_by_id = dedupe_manifest(load_jsonl(dataset_dir / "manifest.jsonl"))
    ok_episodes: list[dict[str, Any]] = []
    seen_ok: set[str] = set()

    for episode_dir in episode_dirs(dataset_dir):
        summary_path = episode_dir / "episode_summary.json"
        transitions_path = episode_dir / "transitions.jsonl"
        if not summary_path.exists() or not transitions_path.exists():
            continue
        summary = load_json(summary_path)
        transitions = load_jsonl(transitions_path)
        episode_id = str(summary.get("episode_id", episode_dir.name))
        manifest_row = manifest_by_id.get(episode_id, {})
        status = str(manifest_row.get("status", "ok"))
        if status != "ok":
            continue
        ok_episodes.append(
            {
                "episode_id": episode_id,
                "episode_dir": str(episode_dir),
                "summary": summary,
                "transitions": transitions,
                "manifest": manifest_row,
            }
        )
        seen_ok.add(episode_id)

    failed: list[dict[str, Any]] = []
    for episode_id, record in manifest_by_id.items():
        if str(record.get("status", "ok")) != "ok":
            failed.append(record)
        elif episode_id not in seen_ok and not (Path(record.get("episode_dir", "")) / "episode_summary.json").exists():
            failed.append(
                {
                    **record,
                    "status": "failed",
                    "error": record.get("error") or "manifest ok row has no readable episode_summary.json",
                }
            )
    return ok_episodes, failed


def save_observed_ratio_curve(ok_episodes: list[dict[str, Any]], dataset_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    series: list[np.ndarray] = []
    for item in ok_episodes:
        transitions = item["transitions"]
        if not transitions:
            continue
        y = [float(transitions[0]["observed_ratio_before"])]
        y.extend(float(t["observed_ratio_after"]) for t in transitions)
        x = np.arange(len(y), dtype=np.int32) - 1
        arr = np.asarray(y, dtype=np.float64)
        series.append(arr)
        ax.plot(x, arr, marker="o", linewidth=1.1, alpha=0.45, label=item["episode_id"])
    if series:
        max_len = max(len(row) for row in series)
        padded = np.full((len(series), max_len), np.nan, dtype=np.float64)
        for idx, row in enumerate(series):
            padded[idx, : len(row)] = row
        mean = np.nanmean(padded, axis=0)
        ax.plot(np.arange(max_len) - 1, mean, marker="o", linewidth=2.6, color="#111827", label="mean")
    ax.set_xlabel("step")
    ax.set_ylabel("observed_ratio")
    ax.set_title("Aggregate measured observed ratio")
    ax.grid(alpha=0.25)
    if len(ok_episodes) <= 10:
        ax.legend(fontsize=7, loc="best")
    out_path = dataset_dir / "aggregate_observed_ratio_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_reachable_candidates_curve(ok_episodes: list[dict[str, Any]], dataset_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    series: list[np.ndarray] = []
    for item in ok_episodes:
        transitions = item["transitions"]
        if not transitions:
            continue
        y = np.asarray([int(t.get("reachable_candidates", 0)) for t in transitions], dtype=np.float64)
        series.append(y)
        ax.plot(np.arange(len(y)), y, marker="o", linewidth=1.1, alpha=0.45, label=item["episode_id"])
    if series:
        max_len = max(len(row) for row in series)
        padded = np.full((len(series), max_len), np.nan, dtype=np.float64)
        for idx, row in enumerate(series):
            padded[idx, : len(row)] = row
        mean = np.nanmean(padded, axis=0)
        ax.plot(np.arange(max_len), mean, marker="o", linewidth=2.6, color="#111827", label="mean")
    ax.set_xlabel("step")
    ax.set_ylabel("reachable_candidates")
    ax.set_title("Aggregate reachable A* candidates")
    ax.grid(alpha=0.25)
    if len(ok_episodes) <= 10:
        ax.legend(fontsize=7, loc="best")
    out_path = dataset_dir / "aggregate_reachable_candidates_curve.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_bar_plot(labels: list[str], values: list[float], title: str, ylabel: str, out_path: Path, color: str) -> str:
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.75), 4.8), constrained_layout=True)
    ax.bar(np.arange(len(labels)), values, color=color)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_done_reasons(counter: Counter[str], dataset_dir: Path) -> str:
    labels = list(counter.keys()) or ["none"]
    values = [counter[label] for label in labels] or [0]
    return save_bar_plot(labels, values, "Done reasons", "episodes", dataset_dir / "aggregate_done_reasons.png", "#2563eb")


def save_steps_hist(steps: list[int], dataset_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    if steps:
        bins = np.arange(min(steps), max(steps) + 2) - 0.5
        ax.hist(steps, bins=bins, color="#0f766e", edgecolor="white")
    ax.set_xlabel("steps_completed")
    ax.set_ylabel("episodes")
    ax.set_title("Steps completed histogram")
    ax.grid(axis="y", alpha=0.2)
    out_path = dataset_dir / "aggregate_steps_hist.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_no_valid_stats(ok_episodes: list[dict[str, Any]], dataset_dir: Path) -> str:
    labels = [item["episode_id"] for item in ok_episodes]
    values = []
    for item in ok_episodes:
        metrics = item["summary"].get("metrics") or {}
        steps = metrics.get("no_valid_candidate_steps") or []
        if not steps and str(item["summary"].get("done_reason")) == "no_valid_candidate":
            steps = [int(item["summary"].get("steps_completed", 0))]
        values.append(len(steps))
    return save_bar_plot(
        labels,
        values,
        "No-valid-candidate steps by episode",
        "step count",
        dataset_dir / "aggregate_no_valid_candidate_stats.png",
        "#b91c1c",
    )


def save_html_index(summary: dict[str, Any], plot_paths: dict[str, str], ok_episodes: list[dict[str, Any]], dataset_dir: Path) -> str:
    lines = [
        "<!doctype html>",
        "<html>",
        "<head><meta charset=\"utf-8\"><title>Stage 4A-4 Rollout Dataset</title></head>",
        "<body>",
        "<h1>Stage 4A-4 Empty-Prediction A* Rollout Dataset</h1>",
        "<h2>Summary</h2>",
        "<ul>",
    ]
    for key in (
        "total_episodes",
        "ok_episodes",
        "failed_episodes",
        "total_transitions",
        "gain_sc_nonzero_count",
    ):
        lines.append(f"<li>{key}: {summary.get(key)}</li>")
    lines.extend(["</ul>", "<h2>Plots</h2>", "<ul>"])
    for key, value in plot_paths.items():
        name = Path(value).name
        lines.append(f'<li>{key}: <a href="{name}">{name}</a></li>')
    lines.extend(["</ul>"])
    for key, value in plot_paths.items():
        name = Path(value).name
        if name.endswith(".png"):
            lines.append(f'<h3>{key}</h3><img src="{name}" style="max-width: 980px; width: 100%;">')
    lines.extend(["<h2>Episodes</h2>", "<ul>"])
    for item in ok_episodes:
        rel = Path("episodes") / Path(item["episode_dir"]).name / "rollout_index.html"
        lines.append(f'<li><a href="{rel.as_posix()}">{item["episode_id"]}</a></li>')
    lines.extend(["</ul>", "</body>", "</html>"])
    out_path = dataset_dir / "rollout_dataset_index.html"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(out_path)


def save_markdown(summary: dict[str, Any], dataset_dir: Path) -> str:
    path = dataset_dir / "dataset_summary.md"
    lines = [
        "# Stage 4A-4 Rollout Dataset Summary",
        "",
        f"- total episodes: {summary['total_episodes']}",
        f"- ok episodes: {summary['ok_episodes']}",
        f"- failed episodes: {summary['failed_episodes']}",
        f"- total transitions: {summary['total_transitions']}",
        f"- steps min/mean/max: {summary['steps_completed']}",
        f"- done reasons: {summary['done_reason_counts']}",
        f"- observed_ratio_end min/mean/max: {summary['observed_ratio_end']}",
        f"- total_delta_observed_ratio min/mean/max: {summary['total_delta_observed_ratio']}",
        f"- average reachable candidates: {summary['transition_averages']['reachable_candidates']}",
        f"- average gain_sc: {summary['transition_averages']['gain_sc']}",
        f"- gain_sc nonzero count: {summary['gain_sc_nonzero_count']}",
        "",
        "No RL, PPO, behavior-cloning training, imitation-learning training, SSCNet training, map_predict connection, "
        "PredictionLayer connection, target labels, ground truth scoring, prediction writeback, UNKNOWN traversability "
        "shortcut, or Euclidean fallback is part of this dataset stage.",
        "",
    ]
    if summary["failed_episode_records"]:
        lines.append("## Failed Episodes")
        for record in summary["failed_episode_records"]:
            lines.append(f"- {record.get('episode_id')}: {record.get('error') or record.get('done_reason')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def summarize_dataset(dataset_dir: Path) -> dict[str, Any]:
    ok_episodes, failed = collect_episodes(dataset_dir)
    all_transitions = [transition for item in ok_episodes for transition in item["transitions"]]
    done_counts = Counter(str(item["summary"].get("done_reason", "")) for item in ok_episodes)
    steps_completed = [int(item["summary"].get("steps_completed", len(item["transitions"]))) for item in ok_episodes]
    observed_end = [float(item["summary"].get("observed_ratio_end", 0.0)) for item in ok_episodes]
    observed_delta = [float(item["summary"].get("total_delta_observed_ratio", 0.0)) for item in ok_episodes]
    no_valid_episode_count = 0
    no_valid_steps_total = 0
    repeated_pose_counts = []
    validation_errors: list[str] = []

    for item in ok_episodes:
        summary = item["summary"]
        metrics = summary.get("metrics") or {}
        no_valid_steps = metrics.get("no_valid_candidate_steps") or []
        if not no_valid_steps and str(summary.get("done_reason")) == "no_valid_candidate":
            no_valid_steps = [int(summary.get("steps_completed", 0))]
        if no_valid_steps:
            no_valid_episode_count += 1
            no_valid_steps_total += len(no_valid_steps)
        repeated_pose_counts.append(int(summary.get("repeated_pose_count", 0)))

    gain_sc_values = []
    transition_values: dict[str, list[float]] = {
        "reachable_candidates": [],
        "reachable_component_count": [],
        "best_score": [],
        "gain_exp": [],
        "gain_sc": [],
        "path_cost": [],
    }
    for transition in all_transitions:
        gain_sc = value_from_transition(transition, "gain_sc", "best_gain_sc")
        if gain_sc is not None:
            gain_sc_values.append(gain_sc)
            transition_values["gain_sc"].append(gain_sc)
        mappings = {
            "reachable_candidates": ("reachable_candidates", None),
            "reachable_component_count": ("reachable_component_count", None),
            "best_score": ("final_score", "best_score"),
            "gain_exp": ("gain_exp", "best_gain_exp"),
            "path_cost": ("path_cost", "best_path_cost"),
        }
        for out_key, (preferred, fallback) in mappings.items():
            value = value_from_transition(transition, preferred, fallback)
            if value is not None:
                transition_values[out_key].append(value)

    gain_sc_nonzero_count = sum(1 for value in gain_sc_values if abs(float(value)) > 1e-9)
    if gain_sc_nonzero_count:
        validation_errors.append(f"EmptyPredictionLayer expected gain_sc=0, got {gain_sc_nonzero_count} nonzero values")

    plot_paths = {
        "aggregate_observed_ratio_curve": save_observed_ratio_curve(ok_episodes, dataset_dir),
        "aggregate_observed_ratio_end_bar": save_bar_plot(
            [item["episode_id"] for item in ok_episodes],
            observed_end,
            "Final observed ratio by episode",
            "observed_ratio_end",
            dataset_dir / "aggregate_observed_ratio_end_bar.png",
            "#2563eb",
        ),
        "aggregate_steps_completed_bar": save_bar_plot(
            [item["episode_id"] for item in ok_episodes],
            [float(v) for v in steps_completed],
            "Steps completed by episode",
            "steps",
            dataset_dir / "aggregate_steps_completed_bar.png",
            "#0f766e",
        ),
        "aggregate_steps_hist": save_steps_hist(steps_completed, dataset_dir),
        "aggregate_done_reasons": save_done_reasons(done_counts, dataset_dir),
        "aggregate_reachable_candidates_curve": save_reachable_candidates_curve(ok_episodes, dataset_dir),
        "aggregate_no_valid_candidate_stats": save_no_valid_stats(ok_episodes, dataset_dir),
    }

    summary: dict[str, Any] = {
        "stage": "Stage 4A-4",
        "dataset_dir": str(dataset_dir),
        "total_episodes": int(len(ok_episodes) + len(failed)),
        "ok_episodes": int(len(ok_episodes)),
        "failed_episodes": int(len(failed)),
        "total_transitions": int(len(all_transitions)),
        "steps_completed": stats([float(v) for v in steps_completed]),
        "done_reason_counts": dict(done_counts),
        "observed_ratio_end": stats(observed_end),
        "total_delta_observed_ratio": stats(observed_delta),
        "no_valid_candidate_episode_count": int(no_valid_episode_count),
        "no_valid_candidate_steps_total": int(no_valid_steps_total),
        "transition_averages": {
            key: (float(np.mean(values)) if values else None) for key, values in transition_values.items()
        },
        "repeated_pose_count": stats([float(v) for v in repeated_pose_counts]),
        "gain_sc_nonzero_count": int(gain_sc_nonzero_count),
        "validation_errors": validation_errors,
        "failed_episode_records": failed,
        "episode_ids": [item["episode_id"] for item in ok_episodes],
        "plot_paths": plot_paths,
        "leakage_summary": {
            "prediction_mode": "empty",
            "prediction_layer": "EmptyPredictionLayer",
            "prediction_wrote_observed_map": False,
            "target_lr_used": False,
            "target_hr_used": False,
            "ground_truth_used_for_scoring": False,
            "rl_ppo_bc_il_training_run": False,
            "unknown_traversability_shortcut": False,
            "euclidean_fallback": False,
        },
    }
    summary["dataset_summary_md"] = save_markdown(summary, dataset_dir)
    summary["rollout_dataset_index"] = save_html_index(summary, plot_paths, ok_episodes, dataset_dir)
    save_json(dataset_dir / "dataset_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Stage 4A-4 rollout dataset.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    summary = summarize_dataset(dataset_dir)
    print("Stage 4A-4 rollout dataset summary complete.")
    print(f"dataset_dir: {summary['dataset_dir']}")
    print(f"ok_episodes: {summary['ok_episodes']}")
    print(f"failed_episodes: {summary['failed_episodes']}")
    print(f"total_transitions: {summary['total_transitions']}")
    print(f"steps_completed: {summary['steps_completed']}")
    print(f"done_reason_counts: {summary['done_reason_counts']}")
    print(f"observed_ratio_end: {summary['observed_ratio_end']}")
    print(f"total_delta_observed_ratio: {summary['total_delta_observed_ratio']}")
    print(f"average_reachable_candidates: {summary['transition_averages']['reachable_candidates']}")
    print(f"average_gain_sc: {summary['transition_averages']['gain_sc']}")
    print(f"gain_sc_nonzero_count: {summary['gain_sc_nonzero_count']}")
    print(f"dataset_summary_json: {dataset_dir / 'dataset_summary.json'}")
    print(f"dataset_summary_md: {summary['dataset_summary_md']}")
    print(f"rollout_dataset_index: {summary['rollout_dataset_index']}")


if __name__ == "__main__":
    main()
