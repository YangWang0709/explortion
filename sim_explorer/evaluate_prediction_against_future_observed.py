#!/usr/bin/env python3
"""Evaluate read-only Isaac SSCNet predictions against later sensor maps.

Future observed maps are delayed sensor validation only. They are never used
for planning, expert scoring, map writeback, traversability, collision checks,
candidate reachability, A*, or ray blocking.
"""

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

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)
EVAL_ONLY_NOTE = (
    "Future observations are post-hoc sensor validation only, not used for planning "
    "or expert scoring."
)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_taus(raw: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(raw, str):
        values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        values = [float(v) for v in raw]
    if not values:
        raise ValueError("At least one tau is required")
    return sorted(dict.fromkeys(values))


def step_from_path(path: Path) -> int:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse step from {path}")
    return int(digits)


def observed_state_paths(episode_dir: str | Path) -> dict[int, Path]:
    episode = Path(episode_dir)
    return {step_from_path(path): path for path in sorted(episode.glob("observed_state_step*.npy"))}


def prediction_step_dirs(episode_dir: str | Path) -> dict[int, Path]:
    episode = Path(episode_dir)
    dirs: dict[int, Path] = {}
    for path in sorted(episode.glob("prediction_step*")):
        if path.is_dir():
            dirs[step_from_path(path)] = path
    return dirs


def load_prediction_arrays(path: str | Path) -> dict[str, np.ndarray]:
    pred_path = Path(path)
    with np.load(pred_path) as data:
        required = (
            "global_prediction_valid",
            "global_confidence",
            "global_occupied_prob",
            "global_free_prob",
        )
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(f"{pred_path} missing prediction fields: {missing}")
        return {
            "valid": np.asarray(data["global_prediction_valid"], dtype=bool),
            "confidence": np.asarray(data["global_confidence"], dtype=np.float32),
            "occupied_prob": np.asarray(data["global_occupied_prob"], dtype=np.float32),
            "free_prob": np.asarray(data["global_free_prob"], dtype=np.float32),
        }


def future_measurement(
    observed_t: np.ndarray,
    future_paths: dict[int, Path],
    step: int,
    max_horizon: int,
) -> tuple[np.ndarray, list[int]]:
    """Return first later measured FREE/OCCUPIED value for voxels unknown at step."""

    observed_t = np.asarray(observed_t)
    combined = np.full(observed_t.shape, UNKNOWN, dtype=np.int8)
    still_unset = observed_t == UNKNOWN
    used_steps: list[int] = []
    for future_step in sorted(future_paths):
        if future_step <= int(step):
            continue
        if future_step > int(step) + int(max_horizon):
            continue
        future_state = np.asarray(np.load(future_paths[future_step]), dtype=np.int8)
        if future_state.shape != observed_t.shape:
            continue
        measured = still_unset & ((future_state == FREE) | (future_state == OCCUPIED))
        if np.any(measured):
            combined[measured] = future_state[measured]
            still_unset[measured] = False
        used_steps.append(int(future_step))
    return combined, used_steps


def _safe_div(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def roc_auc_score_np(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    mask = np.isfinite(scores) & np.isin(labels, [0, 1])
    scores = scores[mask]
    labels = labels[mask]
    pos = int(np.count_nonzero(labels == 1))
    neg = int(np.count_nonzero(labels == 0))
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = float(np.mean(np.arange(start + 1, end + 1, dtype=np.float64)))
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_pos = float(np.sum(ranks[labels == 1]))
    auc = (rank_sum_pos - pos * (pos + 1) / 2.0) / float(pos * neg)
    return float(auc)


def evaluate_prediction_arrays(
    prediction: dict[str, np.ndarray],
    observed_t: np.ndarray,
    future_state: np.ndarray,
    tau: float,
) -> dict[str, Any]:
    valid = np.asarray(prediction["valid"], dtype=bool)
    confidence = np.asarray(prediction["confidence"], dtype=np.float32)
    occupied_prob = np.asarray(prediction["occupied_prob"], dtype=np.float32)
    free_prob = np.asarray(prediction["free_prob"], dtype=np.float32)
    observed_t = np.asarray(observed_t, dtype=np.int8)
    future_state = np.asarray(future_state, dtype=np.int8)

    if valid.shape != observed_t.shape:
        raise ValueError(f"prediction shape {valid.shape} != observed shape {observed_t.shape}")

    unmeasured_t = observed_t == UNKNOWN
    future_measured = unmeasured_t & ((future_state == FREE) | (future_state == OCCUPIED))
    predicted_unmeasured = valid & (confidence >= float(tau)) & unmeasured_t
    evaluated = predicted_unmeasured & future_measured

    predicted_count = int(np.count_nonzero(predicted_unmeasured))
    future_count = int(np.count_nonzero(future_measured))
    later_count = int(np.count_nonzero(evaluated))

    row: dict[str, Any] = {
        "tau": float(tau),
        "predicted_unmeasured_count": predicted_count,
        "future_measured_count": future_count,
        "later_measured_count": later_count,
        "later_measured_fraction": _safe_div(later_count, predicted_count),
        "future_measured_coverage": _safe_div(later_count, future_count),
    }

    if later_count == 0:
        row.update(
            {
                "occupied_precision": None,
                "occupied_recall": None,
                "free_precision": None,
                "free_recall": None,
                "brier_occupied": None,
                "mean_occupied_prob": None,
                "mean_confidence": None,
                "accuracy_at_0p5": None,
                "confidence_correct_mean": None,
                "confidence_incorrect_mean": None,
                "roc_auc_occupied_prob": None,
            }
        )
        return row

    truth_occ = (future_state[evaluated] == OCCUPIED).astype(np.int8)
    occ_scores = occupied_prob[evaluated].astype(np.float64)
    conf_values = confidence[evaluated].astype(np.float64)
    pred_occ = occ_scores >= 0.5
    pred_free = free_prob[evaluated] >= 0.5
    truth_free = truth_occ == 0
    correct = pred_occ == truth_occ.astype(bool)

    tp_occ = int(np.count_nonzero(pred_occ & (truth_occ == 1)))
    pred_occ_count = int(np.count_nonzero(pred_occ))
    future_occ_total = int(np.count_nonzero(future_measured & (future_state == OCCUPIED)))
    tp_free = int(np.count_nonzero(pred_free & truth_free))
    pred_free_count = int(np.count_nonzero(pred_free))
    future_free_total = int(np.count_nonzero(future_measured & (future_state == FREE)))

    row.update(
        {
            "future_occupied_count": future_occ_total,
            "future_free_count": future_free_total,
            "evaluated_occupied_count": int(np.count_nonzero(truth_occ == 1)),
            "evaluated_free_count": int(np.count_nonzero(truth_free)),
            "predicted_occupied_count_evaluated": pred_occ_count,
            "predicted_free_count_evaluated": pred_free_count,
            "occupied_precision": _safe_div(tp_occ, pred_occ_count),
            "occupied_recall": _safe_div(tp_occ, future_occ_total),
            "occupied_recall_within_evaluated": _safe_div(tp_occ, int(np.count_nonzero(truth_occ == 1))),
            "free_precision": _safe_div(tp_free, pred_free_count),
            "free_recall": _safe_div(tp_free, future_free_total),
            "free_recall_within_evaluated": _safe_div(tp_free, int(np.count_nonzero(truth_free))),
            "brier_occupied": float(np.mean((occ_scores - truth_occ.astype(np.float64)) ** 2)),
            "mean_occupied_prob": float(np.mean(occ_scores)),
            "mean_confidence": float(np.mean(conf_values)),
            "accuracy_at_0p5": float(np.mean(correct.astype(np.float64))),
            "confidence_correct_mean": float(np.mean(conf_values[correct])) if np.any(correct) else None,
            "confidence_incorrect_mean": float(np.mean(conf_values[~correct])) if np.any(~correct) else None,
            "roc_auc_occupied_prob": roc_auc_score_np(occ_scores, truth_occ),
        }
    )
    return row


def calibration_bins_for_arrays(
    prediction: dict[str, np.ndarray],
    observed_t: np.ndarray,
    future_state: np.ndarray,
    tau: float,
    bins: int = 10,
    prefix: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    valid = np.asarray(prediction["valid"], dtype=bool)
    confidence = np.asarray(prediction["confidence"], dtype=np.float32)
    occupied_prob = np.asarray(prediction["occupied_prob"], dtype=np.float32)
    observed_t = np.asarray(observed_t, dtype=np.int8)
    future_state = np.asarray(future_state, dtype=np.int8)
    mask = valid & (confidence >= float(tau)) & (observed_t == UNKNOWN)
    mask &= (future_state == FREE) | (future_state == OCCUPIED)
    probs = occupied_prob[mask].astype(np.float64)
    conf = confidence[mask].astype(np.float64)
    truth = (future_state[mask] == OCCUPIED).astype(np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: list[dict[str, Any]] = []
    for idx in range(int(bins)):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == int(bins) - 1:
            bin_mask = (probs >= lo) & (probs <= hi)
        else:
            bin_mask = (probs >= lo) & (probs < hi)
        count = int(np.count_nonzero(bin_mask))
        row = dict(prefix or {})
        row.update(
            {
                "bin": idx,
                "prob_min": float(lo),
                "prob_max": float(hi),
                "count": count,
                "mean_occupied_prob": float(np.mean(probs[bin_mask])) if count else None,
                "empirical_occupied_rate": float(np.mean(truth[bin_mask])) if count else None,
                "mean_confidence": float(np.mean(conf[bin_mask])) if count else None,
            }
        )
        rows.append(row)
    return rows, probs, truth, conf


def aggregate_numeric(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def project_topdown(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr.shape}")
    return np.max(arr, axis=2)


def save_overlay(
    path: Path,
    observed_t: np.ndarray,
    prediction: dict[str, np.ndarray],
    future_state: np.ndarray,
    tau: float,
    title: str,
) -> str:
    unmeasured_t = observed_t == UNKNOWN
    predicted = prediction["valid"] & (prediction["confidence"] >= float(tau)) & unmeasured_t
    future = unmeasured_t & ((future_state == FREE) | (future_state == OCCUPIED))
    hit = predicted & future
    rgb = np.zeros((*observed_t.shape[:2], 3), dtype=np.float32)
    measured_top = project_topdown(observed_t != UNKNOWN)
    pred_top = project_topdown(predicted)
    future_top = project_topdown(future)
    hit_top = project_topdown(hit)
    rgb[measured_top] = (0.70, 0.70, 0.70)
    rgb[pred_top] = (0.20, 0.40, 0.95)
    rgb[future_top] = (0.95, 0.60, 0.15)
    rgb[hit_top] = (0.10, 0.75, 0.35)
    fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("world x grid")
    ax.set_ylabel("world y grid")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def plot_tau_metric(rows: list[dict[str, Any]], key: str, path: Path, ylabel: str, title: str) -> str:
    grouped: dict[str, dict[float, list[float]]] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        source = str(row.get("future_source", "future"))
        tau = float(row["tau"])
        grouped.setdefault(source, {}).setdefault(tau, []).append(float(value))
    fig, ax = plt.subplots(figsize=(7.8, 4.6), constrained_layout=True)
    palette = ["#2563eb", "#f97316", "#0f766e"]
    for idx, (source, by_tau) in enumerate(sorted(grouped.items())):
        taus = sorted(by_tau)
        means = [float(np.mean(by_tau[tau])) for tau in taus]
        ax.plot(taus, means, marker="o", linewidth=2.0, color=palette[idx % len(palette)], label=source)
    ax.set_xlabel("tau")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def plot_reliability(rows: list[dict[str, Any]], path: Path) -> str:
    filtered = [row for row in rows if row.get("count", 0)]
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
    if filtered:
        xs = [float(row["mean_occupied_prob"]) for row in filtered if row["mean_occupied_prob"] is not None]
        ys = [float(row["empirical_occupied_rate"]) for row in filtered if row["empirical_occupied_rate"] is not None]
        sizes = [max(25, math.sqrt(float(row["count"])) * 8.0) for row in filtered if row["mean_occupied_prob"] is not None]
        ax.scatter(xs, ys, s=sizes, color="#2563eb", alpha=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted occupied probability")
    ax.set_ylabel("empirical future occupied rate")
    ax.set_title("Reliability against later sensor measurements")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def plot_confidence_correctness(rows: list[dict[str, Any]], path: Path) -> str:
    labels: list[str] = []
    correct_vals: list[float] = []
    incorrect_vals: list[float] = []
    for row in rows:
        if row.get("future_source") != "sc_future":
            continue
        if row.get("confidence_correct_mean") is None and row.get("confidence_incorrect_mean") is None:
            continue
        labels.append(f"s{row['step']}")
        correct_vals.append(float(row["confidence_correct_mean"] or 0.0))
        incorrect_vals.append(float(row["confidence_incorrect_mean"] or 0.0))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(7.0, len(labels) * 0.8), 4.5), constrained_layout=True)
    ax.bar(x - 0.17, correct_vals, width=0.34, color="#0f766e", label="correct")
    ax.bar(x + 0.17, incorrect_vals, width=0.34, color="#be123c", label="incorrect")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean confidence")
    ax.set_title("Confidence vs later correctness")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    sc_episode = Path(args.sc_episode_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    taus = parse_taus(args.taus)
    default_tau = taus[0]
    max_horizon = int(args.max_horizon)

    sc_observed_paths = observed_state_paths(sc_episode)
    pred_dirs = prediction_step_dirs(sc_episode)
    future_sources: list[tuple[str, Path, dict[int, Path]]] = [
        ("sc_future", sc_episode, sc_observed_paths),
    ]
    if args.empty_episode_dir:
        empty_episode = Path(args.empty_episode_dir).resolve()
        if empty_episode.is_dir():
            future_sources.append(("empty_future", empty_episode, observed_state_paths(empty_episode)))

    per_step_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    all_probs: list[np.ndarray] = []
    all_truth: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []
    overlay_paths: list[str] = []

    for step, pred_dir in sorted(pred_dirs.items()):
        if step not in sc_observed_paths:
            continue
        pred_path = pred_dir / "global_prediction_layer.npz"
        if not pred_path.is_file():
            continue
        prediction = load_prediction_arrays(pred_path)
        observed_t = np.asarray(np.load(sc_observed_paths[step]), dtype=np.int8)
        for source_name, source_dir, future_paths in future_sources:
            future_state, used_steps = future_measurement(observed_t, future_paths, step, max_horizon)
            for tau in taus:
                metrics = evaluate_prediction_arrays(prediction, observed_t, future_state, tau)
                row = {
                    "step": int(step),
                    "future_source": source_name,
                    "future_episode_dir": str(source_dir),
                    "future_steps_used": json.dumps(used_steps),
                    "prediction_npz": str(pred_path),
                    "observed_state_step": str(sc_observed_paths[step]),
                    **metrics,
                    "evaluation_only_note": EVAL_ONLY_NOTE,
                }
                tau_rows.append(row)
                if float(tau) == float(default_tau):
                    per_step_rows.append(row)
                    if source_name == "sc_future":
                        bins, probs, truth, conf = calibration_bins_for_arrays(
                            prediction,
                            observed_t,
                            future_state,
                            tau=tau,
                            prefix={"step": int(step), "future_source": source_name, "tau": float(tau)},
                        )
                        calibration_rows.extend(bins)
                        if probs.size:
                            all_probs.append(probs)
                            all_truth.append(truth)
                            all_conf.append(conf)
                        overlay_paths.append(
                            save_overlay(
                                output_dir / f"future_measured_overlay_step{step:03d}.png",
                                observed_t,
                                prediction,
                                future_state,
                                tau,
                                title=f"step {step}: prediction vs later sensor measurements",
                            )
                        )

    write_csv(output_dir / "future_eval_per_step.csv", per_step_rows)
    write_csv(output_dir / "future_eval_tau_sweep.csv", tau_rows)
    write_csv(output_dir / "calibration_bins.csv", calibration_rows)

    plot_paths = {
        "reliability_diagram": plot_reliability(calibration_rows, output_dir / "reliability_diagram.png"),
        "tau_vs_predicted_unmeasured": plot_tau_metric(
            tau_rows,
            "predicted_unmeasured_count",
            output_dir / "tau_vs_predicted_unmeasured.png",
            "predicted unmeasured voxels",
            "Tau vs prediction density",
        ),
        "tau_vs_later_measured_fraction": plot_tau_metric(
            tau_rows,
            "later_measured_fraction",
            output_dir / "tau_vs_later_measured_fraction.png",
            "later measured fraction",
            "Tau vs delayed sensor overlap",
        ),
        "tau_vs_brier": plot_tau_metric(
            tau_rows,
            "brier_occupied",
            output_dir / "tau_vs_brier.png",
            "occupied Brier score",
            "Tau vs occupied calibration",
        ),
        "confidence_vs_correctness": plot_confidence_correctness(
            [row for row in per_step_rows if float(row["tau"]) == float(default_tau)],
            output_dir / "confidence_vs_correctness.png",
        ),
    }

    primary_default = [
        row
        for row in tau_rows
        if row.get("future_source") == "sc_future" and float(row.get("tau", -1.0)) == float(default_tau)
    ]
    aggregate_keys = [
        "predicted_unmeasured_count",
        "future_measured_count",
        "later_measured_count",
        "later_measured_fraction",
        "future_measured_coverage",
        "occupied_precision",
        "occupied_recall",
        "free_precision",
        "free_recall",
        "brier_occupied",
        "mean_occupied_prob",
        "mean_confidence",
        "accuracy_at_0p5",
        "roc_auc_occupied_prob",
    ]
    primary_summary = aggregate_numeric(primary_default, aggregate_keys)

    tau_summary_rows: list[dict[str, Any]] = []
    for tau in taus:
        tau_group = [
            row for row in tau_rows if row.get("future_source") == "sc_future" and float(row["tau"]) == float(tau)
        ]
        summary = aggregate_numeric(tau_group, aggregate_keys)
        tau_summary_rows.append({"tau": float(tau), **summary})

    ece = None
    if all_probs:
        probs = np.concatenate(all_probs)
        truth = np.concatenate(all_truth)
        edges = np.linspace(0.0, 1.0, 11)
        total = max(1, probs.size)
        ece_val = 0.0
        for idx in range(10):
            lo, hi = edges[idx], edges[idx + 1]
            mask = (probs >= lo) & (probs <= hi if idx == 9 else probs < hi)
            count = int(np.count_nonzero(mask))
            if count:
                ece_val += (count / total) * abs(float(np.mean(probs[mask])) - float(np.mean(truth[mask])))
        ece = float(ece_val)

    counts = [row.get("predicted_unmeasured_count") for row in primary_default]
    later_fracs = [row.get("later_measured_fraction") for row in primary_default if row.get("later_measured_fraction") is not None]
    briers = [row.get("brier_occupied") for row in primary_default if row.get("brier_occupied") is not None]
    tau_01_dense = bool(counts and float(np.mean([float(v) for v in counts])) > 25000.0)
    tau_reduces_density = None
    if len(tau_summary_rows) >= 2:
        first = tau_summary_rows[0].get("predicted_unmeasured_count")
        last = tau_summary_rows[-1].get("predicted_unmeasured_count")
        if first is not None and last is not None:
            tau_reduces_density = bool(float(last) < 0.5 * float(first))

    summary = {
        "stage": "Stage 4A-6.2 future observed prediction evaluation",
        "sc_episode_dir": str(sc_episode),
        "empty_episode_dir": str(Path(args.empty_episode_dir).resolve()) if args.empty_episode_dir else "",
        "output_dir": str(output_dir),
        "taus": taus,
        "default_tau": float(default_tau),
        "max_horizon": max_horizon,
        "future_observations_usage": EVAL_ONLY_NOTE,
        "primary_source": "sc_future",
        "primary_default_tau_summary": primary_summary,
        "tau_summary_rows": tau_summary_rows,
        "ece_occupied_prob": ece,
        "tau_0p1_too_dense": tau_01_dense,
        "tau_reduces_density_meaningfully": tau_reduces_density,
        "mean_later_measured_fraction_default": float(np.mean(later_fracs)) if later_fracs else None,
        "mean_brier_default": float(np.mean(briers)) if briers else None,
        "outputs": {
            **plot_paths,
            "future_eval_per_step": str(output_dir / "future_eval_per_step.csv"),
            "future_eval_tau_sweep": str(output_dir / "future_eval_tau_sweep.csv"),
            "calibration_bins": str(output_dir / "calibration_bins.csv"),
            "future_measured_overlays": overlay_paths,
        },
        "planning_or_training_used": False,
        "prediction_writeback": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_a_star": False,
        "prediction_blocks_rays": False,
    }
    save_json(output_dir / "future_eval_summary.json", summary)

    md = [
        "# Stage 4A-6.2 Future Observed Evaluation",
        "",
        f"- {EVAL_ONLY_NOTE}",
        f"- Primary source: `sc_future`, tau `{default_tau}`.",
        f"- Mean predicted unmeasured count: `{primary_summary.get('predicted_unmeasured_count')}`.",
        f"- Mean later measured fraction: `{primary_summary.get('later_measured_fraction')}`.",
        f"- Mean future measured coverage: `{primary_summary.get('future_measured_coverage')}`.",
        f"- Mean occupied precision: `{primary_summary.get('occupied_precision')}`.",
        f"- Mean free precision: `{primary_summary.get('free_precision')}`.",
        f"- Mean occupied Brier: `{primary_summary.get('brier_occupied')}`.",
        f"- ECE-like occupied calibration: `{ece}`.",
        f"- Tau 0.1 dense flag: `{tau_01_dense}`.",
        f"- Tau reduces density meaningfully by highest tested tau: `{tau_reduces_density}`.",
        "",
        "No planning, expert scoring, observed-map writes, traversability, collision checks, A*, ray blocking, or training used these future observations.",
    ]
    (output_dir / "future_eval_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Future observed evaluation complete: {output_dir}")
    print(EVAL_ONLY_NOTE)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc_episode_dir", type=Path, required=True)
    parser.add_argument("--empty_episode_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--taus", default="0.1,0.2,0.3,0.5,0.7,0.9")
    parser.add_argument("--max_horizon", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_evaluation(parse_args())


if __name__ == "__main__":
    main()
