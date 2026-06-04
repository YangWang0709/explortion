#!/usr/bin/env python3
"""Build diagnostic calibration tables for selective simulator SC gain.

The inputs are Stage 4A-6.2/6.3 post-hoc evaluation outputs. Future observed
maps are used only to estimate reliability tables; runtime expert scoring
never reads future observations directly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from evaluate_prediction_against_future_observed import EVAL_ONLY_NOTE, future_measurement, observed_state_paths

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)


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
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    parsed = _safe_float(value, None)
    return default if parsed is None else int(parsed)


def _bin_rows(values: np.ndarray, truth: np.ndarray, extra: np.ndarray | None, bins: int, table: str) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if extra is not None:
        extra = np.asarray(extra, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows: list[dict[str, Any]] = []
    for idx in range(int(bins)):
        lo = float(edges[idx])
        hi = float(edges[idx + 1])
        mask = (values >= lo) & ((values <= hi) if idx == int(bins) - 1 else (values < hi))
        count = int(np.count_nonzero(mask))
        row = {
            "table": table,
            "bin": int(idx),
            "prob_min" if table == "occupied_prob" else "confidence_min": lo,
            "prob_max" if table == "occupied_prob" else "confidence_max": hi,
            "count": count,
            "future_observations_usage": EVAL_ONLY_NOTE,
        }
        if count:
            row["mean_occupied_prob" if table == "occupied_prob" else "mean_confidence"] = float(np.mean(values[mask]))
            if table == "occupied_prob":
                row["empirical_occupied_rate"] = float(np.mean(truth[mask]))
                row["mean_confidence"] = float(np.mean(extra[mask])) if extra is not None else None
            else:
                row["empirical_correct_rate"] = float(np.mean(truth[mask]))
                row["mean_occupied_prob"] = float(np.mean(extra[mask])) if extra is not None else None
        else:
            row["mean_occupied_prob" if table == "occupied_prob" else "mean_confidence"] = None
            row["empirical_occupied_rate" if table == "occupied_prob" else "empirical_correct_rate"] = None
            row["mean_confidence" if table == "occupied_prob" else "mean_occupied_prob"] = None
        rows.append(row)
    return rows


def _ece(rows: list[dict[str, Any]], value_key: str, rate_key: str) -> float | None:
    total = sum(int(row.get("count") or 0) for row in rows)
    if total <= 0:
        return None
    acc = 0.0
    for row in rows:
        count = int(row.get("count") or 0)
        mean_value = row.get(value_key)
        rate = row.get(rate_key)
        if not count or mean_value is None or rate is None:
            continue
        acc += (count / total) * abs(float(mean_value) - float(rate))
    return float(acc)


def _weighted_corr(rows: list[dict[str, Any]], x_key: str, y_key: str) -> float | None:
    xs: list[float] = []
    ys: list[float] = []
    weights: list[float] = []
    for row in rows:
        count = int(row.get("count") or 0)
        x = row.get(x_key)
        y = row.get(y_key)
        if count <= 0 or x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
        weights.append(float(count))
    if len(xs) < 2:
        return None
    x_arr = np.asarray(xs, dtype=np.float64)
    y_arr = np.asarray(ys, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    wx = float(np.average(x_arr, weights=w_arr))
    wy = float(np.average(y_arr, weights=w_arr))
    cov = float(np.average((x_arr - wx) * (y_arr - wy), weights=w_arr))
    vx = float(np.average((x_arr - wx) ** 2, weights=w_arr))
    vy = float(np.average((y_arr - wy) ** 2, weights=w_arr))
    if vx <= 0.0 or vy <= 0.0:
        return None
    return float(cov / np.sqrt(vx * vy))


def _plot_reliability(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6.3, 5.6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
    filtered = [row for row in rows if int(row.get("count") or 0) > 0 and row.get(x_key) is not None and row.get(y_key) is not None]
    if filtered:
        xs = [float(row[x_key]) for row in filtered]
        ys = [float(row[y_key]) for row in filtered]
        sizes = [max(28.0, np.sqrt(float(row["count"])) * 7.0) for row in filtered]
        ax.scatter(xs, ys, s=sizes, color="#2563eb", alpha=0.82)
        ax.plot(xs, ys, color="#2563eb", alpha=0.55)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def _pick_thresholds(occ_rows: list[dict[str, Any]], conf_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(int(row.get("count") or 0) for row in occ_rows)
    global_occ_rate = (
        sum(int(row.get("count") or 0) * float(row.get("empirical_occupied_rate") or 0.0) for row in occ_rows)
        / max(1, total)
    )
    occ_candidates: list[dict[str, Any]] = []
    for row in occ_rows:
        count = int(row.get("count") or 0)
        rate = row.get("empirical_occupied_rate")
        lo = row.get("prob_min")
        if count >= 50 and rate is not None and lo is not None:
            gain = float(rate) - float(global_occ_rate)
            if gain > 0.02:
                occ_candidates.append(
                    {
                        "threshold": float(lo),
                        "empirical_occupied_rate": float(rate),
                        "count": count,
                        "rate_lift_vs_global": float(gain),
                    }
                )
    occ_candidates = sorted(occ_candidates, key=lambda r: (-float(r["empirical_occupied_rate"]), float(r["threshold"])))
    recommended_occ = 0.7
    for candidate in occ_candidates:
        threshold = float(candidate["threshold"])
        if threshold >= 0.7:
            recommended_occ = threshold
            break

    conf_candidates: list[dict[str, Any]] = []
    total_conf = sum(int(row.get("count") or 0) for row in conf_rows)
    global_correct = (
        sum(int(row.get("count") or 0) * float(row.get("empirical_correct_rate") or 0.0) for row in conf_rows)
        / max(1, total_conf)
    )
    for row in conf_rows:
        count = int(row.get("count") or 0)
        rate = row.get("empirical_correct_rate")
        lo = row.get("confidence_min")
        if count >= 50 and rate is not None and lo is not None:
            gain = float(rate) - float(global_correct)
            if gain > 0.02:
                conf_candidates.append(
                    {
                        "threshold": float(lo),
                        "empirical_correct_rate": float(rate),
                        "count": count,
                        "rate_lift_vs_global": float(gain),
                    }
                )
    conf_candidates = sorted(conf_candidates, key=lambda r: (-float(r["empirical_correct_rate"]), float(r["threshold"])))
    recommended_conf = 0.3
    for candidate in conf_candidates:
        threshold = float(candidate["threshold"])
        if threshold >= 0.5:
            recommended_conf = threshold
            break

    return {
        "recommended_occ_threshold": float(recommended_occ),
        "recommended_conf_threshold": float(recommended_conf),
        "useful_occupied_prob_threshold_candidates": occ_candidates,
        "useful_confidence_threshold_candidates": conf_candidates,
        "global_empirical_occupied_rate": float(global_occ_rate),
        "global_empirical_correct_rate": float(global_correct),
    }


def _load_code_consistent_voxels(alignment_eval_dir: Path, tau: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metrics_rows = read_csv(alignment_eval_dir / "convention_metrics.csv")
    selected = [
        row
        for row in metrics_rows
        if row.get("convention") == "code_consistent_v1"
        and row.get("future_source") == "sc_future"
        and abs(float(row.get("tau") or -1.0) - float(tau)) < 1e-9
    ]
    probs: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    occ_truth: list[np.ndarray] = []
    correct_truth: list[np.ndarray] = []
    for row in selected:
        pred_path = Path(row.get("projected_prediction_npz") or "")
        if not pred_path.is_file():
            continue
        step = int(row.get("step") or 0)
        with np.load(pred_path, allow_pickle=False) as pred:
            valid = np.asarray(pred["global_prediction_valid"], dtype=bool)
            confidence = np.asarray(pred["global_confidence"], dtype=np.float32)
            occupied_prob = np.asarray(pred["global_occupied_prob"], dtype=np.float32)
            observed_path = Path(str(np.asarray(pred["observed_state_source"]).item()))
        if not observed_path.is_file():
            continue
        observed_t = np.asarray(np.load(observed_path), dtype=np.int8)
        future_episode = Path(row.get("future_episode_dir") or "")
        if not future_episode.is_dir():
            continue
        future_state, used_steps = future_measurement(
            observed_t,
            observed_state_paths(future_episode),
            step=step,
            max_horizon=5,
        )
        if not used_steps:
            continue
        mask = valid & (confidence >= float(tau)) & (observed_t == UNKNOWN)
        mask &= (future_state == FREE) | (future_state == OCCUPIED)
        if not np.any(mask):
            continue
        occ_scores = occupied_prob[mask].astype(np.float64)
        conf_scores = confidence[mask].astype(np.float64)
        truth_occ = (future_state[mask] == OCCUPIED).astype(np.float64)
        pred_occ = occ_scores >= 0.5
        correct = (pred_occ == truth_occ.astype(bool)).astype(np.float64)
        probs.append(occ_scores)
        confs.append(conf_scores)
        occ_truth.append(truth_occ)
        correct_truth.append(correct)
    if not probs:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    return np.concatenate(probs), np.concatenate(confs), np.concatenate(occ_truth), np.concatenate(correct_truth)


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    future_eval_dir = Path(args.future_eval_dir).resolve()
    alignment_eval_dir = Path(args.alignment_eval_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tau = float(args.tau)

    future_summary = load_json(future_eval_dir / "future_eval_summary.json")
    convention_summary = load_json(alignment_eval_dir / "convention_metrics.json")
    probs, confs, truth_occ, truth_correct = _load_code_consistent_voxels(alignment_eval_dir, tau=tau)
    if probs.size == 0:
        raise RuntimeError("No code_consistent_v1 delayed sensor validation voxels available for calibration")

    occupied_bins = _bin_rows(probs, truth_occ, confs, bins=int(args.bins), table="occupied_prob")
    confidence_bins = _bin_rows(confs, truth_correct, probs, bins=int(args.bins), table="confidence")
    threshold_summary = _pick_thresholds(occupied_bins, confidence_bins)

    occupied_corr = _weighted_corr(occupied_bins, "mean_occupied_prob", "empirical_occupied_rate")
    confidence_corr = _weighted_corr(confidence_bins, "mean_confidence", "empirical_correct_rate")
    occupied_ece = _ece(occupied_bins, "mean_occupied_prob", "empirical_occupied_rate")
    confidence_ece = _ece(confidence_bins, "mean_confidence", "empirical_correct_rate")
    occupied_reliable = bool(occupied_corr is not None and occupied_corr > 0.25)
    confidence_reliable = bool(confidence_corr is not None and confidence_corr > 0.20)
    max_occ_rate = max((float(row.get("empirical_occupied_rate") or 0.0) for row in occupied_bins), default=0.0)
    calibration_usable = bool(occupied_reliable and max_occ_rate >= 0.20 and probs.size >= 500)

    table = {
        "stage": "Stage 4A-6.4 calibrated / confidence-gated I_sc",
        "source": "Stage 4A-6.3 code_consistent_v1 convention eval, sc_future delayed sensor validation",
        "future_eval_dir": str(future_eval_dir),
        "alignment_eval_dir": str(alignment_eval_dir),
        "future_observations_usage": EVAL_ONLY_NOTE,
        "runtime_planning_usage": "Fixed table only; future observations are not read by expert scoring.",
        "tau": tau,
        "bin_count": int(args.bins),
        "sample_count": int(probs.size),
        "occupied_prob_bins": occupied_bins,
        "confidence_bins": confidence_bins,
        "recommended_thresholds": threshold_summary,
        "occupied_prob_usefulness": {
            "weighted_bin_correlation": occupied_corr,
            "ece_like": occupied_ece,
            "reliable_enough": occupied_reliable,
            "max_empirical_occupied_rate": float(max_occ_rate),
        },
        "confidence_usefulness": {
            "weighted_bin_correlation": confidence_corr,
            "ece_like": confidence_ece,
            "reliable_enough": confidence_reliable,
        },
        "calibrated_occupied_usable": calibration_usable,
        "stage4a62_reference": {
            "mean_brier_default": future_summary.get("mean_brier_default"),
            "ece_occupied_prob": future_summary.get("ece_occupied_prob"),
            "tau_0p1_too_dense": future_summary.get("tau_0p1_too_dense"),
        },
        "stage4a63_reference": {
            "recommended_convention": convention_summary.get("recommended_convention"),
            "brier_improvement_code_consistent_vs_default": convention_summary.get(
                "brier_improvement_code_consistent_vs_default"
            ),
            "summary_rows": convention_summary.get("summary_rows", []),
        },
        "planning_or_training_used": False,
        "prediction_writeback": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_a_star": False,
        "prediction_blocks_rays": False,
    }

    save_json(output_dir / "calibration_table.json", table)
    write_csv(output_dir / "calibration_table.csv", occupied_bins + confidence_bins)
    save_json(output_dir / "recommended_thresholds.json", threshold_summary | {
        "occupied_prob_reliable_enough": occupied_reliable,
        "confidence_reliable_enough": confidence_reliable,
        "calibrated_occupied_usable": calibration_usable,
    })
    occ_png = _plot_reliability(
        occupied_bins,
        "mean_occupied_prob",
        "empirical_occupied_rate",
        output_dir / "occupied_prob_reliability.png",
        "Occupied probability reliability",
        "mean predicted occupied probability",
        "empirical future occupied rate",
    )
    conf_png = _plot_reliability(
        confidence_bins,
        "mean_confidence",
        "empirical_correct_rate",
        output_dir / "confidence_reliability.png",
        "Confidence reliability",
        "mean prediction confidence",
        "empirical future correctness",
    )

    md = [
        "# Stage 4A-6.4 Calibration Summary",
        "",
        f"- source: {table['source']}",
        f"- future observations usage: {EVAL_ONLY_NOTE}",
        f"- samples: `{probs.size}`",
        f"- occupied_prob weighted bin correlation: `{occupied_corr}`",
        f"- occupied_prob ECE-like: `{occupied_ece}`",
        f"- occupied_prob reliable enough: `{occupied_reliable}`",
        f"- confidence weighted bin correlation: `{confidence_corr}`",
        f"- confidence ECE-like: `{confidence_ece}`",
        f"- confidence reliable enough: `{confidence_reliable}`",
        f"- recommended occ threshold: `{threshold_summary['recommended_occ_threshold']}`",
        f"- recommended conf threshold: `{threshold_summary['recommended_conf_threshold']}`",
        f"- calibrated_occupied usable: `{calibration_usable}`",
        "",
        "Caveat: reliability is estimated from delayed sensor observations in one scene/seed/start, not from full ground truth.",
        "",
        "No planning, expert scoring with future maps, observed-map writes, traversability, collision checks, A*, ray blocking, optimizer, RL, PPO, BC, IL, or SSCNet training used these future observations.",
    ]
    (output_dir / "calibration_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    table["outputs"] = {
        "calibration_table_json": str(output_dir / "calibration_table.json"),
        "calibration_table_csv": str(output_dir / "calibration_table.csv"),
        "calibration_summary_md": str(output_dir / "calibration_summary.md"),
        "occupied_prob_reliability": occ_png,
        "confidence_reliability": conf_png,
        "recommended_thresholds": str(output_dir / "recommended_thresholds.json"),
    }
    save_json(output_dir / "calibration_table.json", table)
    print("Stage 4A-6.4 prediction-gain calibration complete.")
    print(f"output_dir: {output_dir}")
    print(f"occupied_prob_reliable_enough: {occupied_reliable}")
    print(f"confidence_reliable_enough: {confidence_reliable}")
    print(f"calibrated_occupied_usable: {calibration_usable}")
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future_eval_dir", type=Path, required=True)
    parser.add_argument("--alignment_eval_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--bins", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    run_calibration(parse_args())


if __name__ == "__main__":
    main()
