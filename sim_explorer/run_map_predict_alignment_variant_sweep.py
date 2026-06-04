#!/usr/bin/env python3
"""Sweep diagnostic-only local prediction to global alignment variants."""

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

from evaluate_prediction_against_future_observed import (
    EVAL_ONLY_NOTE,
    evaluate_prediction_arrays,
    future_measurement,
    observed_state_paths,
    parse_taus,
)
from isaac_sscnet_preprocess import DEFAULT_LOCAL_VOLUME_M, load_json
from sim_paper_expert import UNKNOWN

VARIANTS = (
    "current_default",
    "yaw_sign_flipped",
    "x_right_sign_flipped",
    "z_forward_sign_flipped",
    "x_right_and_yaw_flipped",
    "xz_swap_variant",
    "z_up_sign_variant",
    "local_origin_shift_forward_half_voxel",
    "local_origin_shift_backward_half_voxel",
)


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


def normalize_bounds(raw: dict[str, Any] | None) -> dict[str, tuple[float, float]]:
    if raw is None:
        raw = {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]}
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def pose_origin_yaw(pose: dict[str, Any], variant: str) -> tuple[np.ndarray, float]:
    origin = np.asarray(pose["position"], dtype=np.float64)
    yaw = float(pose.get("yaw_rad", math.radians(float(pose.get("yaw_deg", 0.0)))))
    if variant in {"yaw_sign_flipped", "x_right_and_yaw_flipped"}:
        yaw = -yaw
    return origin, yaw


def step_from_path(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse step from {path}")
    return int(digits)


def load_local_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "pred_class": np.asarray(data["pred_class"], dtype=np.uint8),
            "confidence": np.asarray(data["confidence"], dtype=np.float32),
            "free_prob": np.asarray(data["free_prob"], dtype=np.float32),
            "occupied_prob": np.asarray(data["occupied_prob"], dtype=np.float32),
        }


def local_coords(shape: tuple[int, int, int], variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    iz, iy, ix = np.indices(shape, dtype=np.float32)
    x_right = -0.5 * DEFAULT_LOCAL_VOLUME_M[0] + (ix + 0.5) * 0.08
    y_up = -0.5 * DEFAULT_LOCAL_VOLUME_M[1] + (iy + 0.5) * 0.08
    z_forward = (iz + 0.5) * 0.08
    if variant in {"x_right_sign_flipped", "x_right_and_yaw_flipped"}:
        x_right = -x_right
    if variant == "z_forward_sign_flipped":
        z_forward = -z_forward
    if variant == "xz_swap_variant":
        x_old = x_right.copy()
        x_right = z_forward - 2.4
        z_forward = x_old + 2.4
    if variant == "z_up_sign_variant":
        y_up = -y_up
    if variant == "local_origin_shift_forward_half_voxel":
        z_forward = z_forward + 0.04
    if variant == "local_origin_shift_backward_half_voxel":
        z_forward = z_forward - 0.04
    return x_right, y_up, z_forward


def project_variant(
    local: dict[str, np.ndarray],
    pose: dict[str, Any],
    observed_shape: tuple[int, int, int],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    variant: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    shape = tuple(int(v) for v in local["pred_class"].shape)
    x_right, y_up, z_forward = local_coords(shape, variant)
    origin, yaw = pose_origin_yaw(pose, variant)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = origin[0] + cos_yaw * z_forward - sin_yaw * x_right
    world_y = origin[1] + sin_yaw * z_forward + cos_yaw * x_right
    world_z = origin[2] + y_up
    gx = np.floor((world_x - bounds["x"][0]) / float(voxel_size)).astype(np.int64)
    gy = np.floor((world_y - bounds["y"][0]) / float(voxel_size)).astype(np.int64)
    gz = np.floor((world_z - bounds["z"][0]) / float(voxel_size)).astype(np.int64)
    shape_arr = np.asarray(observed_shape, dtype=np.int64)
    inside = (
        (gx >= 0)
        & (gx < shape_arr[0])
        & (gy >= 0)
        & (gy < shape_arr[1])
        & (gz >= 0)
        & (gz < shape_arr[2])
    )

    pred_class = np.full(observed_shape, 255, dtype=np.uint8)
    confidence = np.zeros(observed_shape, dtype=np.float32)
    free_prob = np.full(observed_shape, 0.5, dtype=np.float32)
    occupied_prob = np.full(observed_shape, 0.5, dtype=np.float32)
    valid = np.zeros(observed_shape, dtype=bool)

    if np.any(inside):
        flat = np.ravel_multi_index((gx[inside], gy[inside], gz[inside]), observed_shape)
        conf_inside = local["confidence"][inside]
        order = np.argsort(-conf_inside, kind="mergesort")
        flat_ordered = flat[order]
        _, first_positions = np.unique(flat_ordered, return_index=True)
        selected = order[first_positions]
        selected_flat = flat[selected]
        idx = np.unravel_index(selected_flat, observed_shape)
        pred_inside = local["pred_class"][inside]
        free_inside = local["free_prob"][inside]
        occ_inside = local["occupied_prob"][inside]
        pred_class[idx] = pred_inside[selected].astype(np.uint8)
        confidence[idx] = conf_inside[selected].astype(np.float32)
        free_prob[idx] = free_inside[selected].astype(np.float32)
        occupied_prob[idx] = occ_inside[selected].astype(np.float32)
        valid[idx] = True

    meta = {
        "local_voxels_total": int(np.prod(shape)),
        "local_voxels_inside_global": int(np.count_nonzero(inside)),
        "global_valid_prediction_count": int(np.count_nonzero(valid)),
        "in_front_local_ratio": float(np.count_nonzero(z_forward > 0.0) / max(1, z_forward.size)),
        "below_floor_local_count": int(np.count_nonzero(world_z < bounds["z"][0])),
        "above_ceiling_local_count": int(np.count_nonzero(world_z >= bounds["z"][1])),
        "diagnostic_only": True,
    }
    return {
        "pred_class": pred_class,
        "confidence": confidence,
        "free_prob": free_prob,
        "occupied_prob": occupied_prob,
        "valid": valid,
    }, meta


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def project_topdown(mask: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(mask), axis=2)


def plot_variant_topdown(step0_predictions: dict[str, dict[str, np.ndarray]], output_path: Path) -> str:
    if not step0_predictions:
        output_path.write_text("no variants", encoding="utf-8")
        return str(output_path)
    names = list(step0_predictions.keys())
    cols = 3
    rows = int(math.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.7), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_arr, names):
        pred = step0_predictions[name]
        image = np.zeros((*pred["valid"].shape[:2], 3), dtype=np.float32)
        image[project_topdown(pred["valid"])] = (0.2, 0.4, 0.95)
        image[project_topdown(pred["valid"] & (pred["occupied_prob"] >= 0.5))] = (0.9, 0.1, 0.1)
        ax.imshow(np.transpose(image, (1, 0, 2)), origin="lower", interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(names) :]:
        ax.axis("off")
    fig.suptitle("Step 0 diagnostic alignment variants")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_variant_metric(summary_rows: list[dict[str, Any]], key: str, path: Path, title: str, ylabel: str) -> str:
    rows = [row for row in summary_rows if row.get(key) is not None]
    fig, ax = plt.subplots(figsize=(max(8.5, len(rows) * 0.95), 4.8), constrained_layout=True)
    labels = [str(row["variant"]) for row in rows]
    values = [float(row[key]) for row in rows]
    ax.bar(labels, values, color="#2563eb")
    ax.tick_params(axis="x", labelrotation=35)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    sc_episode = Path(args.sc_episode_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    taus = parse_taus(args.taus)
    episode_summary = load_json(sc_episode / "episode_summary.json")
    bounds = normalize_bounds(episode_summary.get("map_bounds"))
    voxel_size = float(episode_summary.get("voxel_size", 0.1))
    sc_future_paths = observed_state_paths(sc_episode)
    future_sources: list[tuple[str, Path, dict[int, Path]]] = [("sc_future", sc_episode, sc_future_paths)]
    if args.empty_episode_dir:
        empty_episode = Path(args.empty_episode_dir).resolve()
        if empty_episode.is_dir():
            future_sources.append(("empty_future", empty_episode, observed_state_paths(empty_episode)))

    rows: list[dict[str, Any]] = []
    step0_predictions: dict[str, dict[str, np.ndarray]] = {}

    for pred_dir in sorted(sc_episode.glob("prediction_step*")):
        if not pred_dir.is_dir():
            continue
        step = step_from_path(pred_dir)
        observed_path = sc_episode / f"observed_state_step{step:03d}.npy"
        pose_path = sc_episode / f"pose_{step:03d}.json"
        local_path = pred_dir / "local_prediction.npz"
        if not observed_path.is_file() or not pose_path.is_file() or not local_path.is_file():
            continue
        observed_t = np.asarray(np.load(observed_path), dtype=np.int8)
        pose = load_json(pose_path)
        local = load_local_prediction(local_path)
        for variant in VARIANTS:
            prediction, meta = project_variant(local, pose, observed_t.shape, bounds, voxel_size, variant)
            if step == 0:
                step0_predictions[variant] = prediction
            for source_name, source_dir, future_paths in future_sources:
                future_state, used_steps = future_measurement(observed_t, future_paths, step, max_horizon=5)
                for tau in taus:
                    metrics = evaluate_prediction_arrays(prediction, observed_t, future_state, tau)
                    rows.append(
                        {
                            "variant": variant,
                            "step": int(step),
                            "future_source": source_name,
                            "future_episode_dir": str(source_dir),
                            "future_steps_used": json.dumps(used_steps),
                            "local_prediction_npz": str(local_path),
                            **meta,
                            **metrics,
                            "diagnostic_only": True,
                            "future_observations_usage": EVAL_ONLY_NOTE,
                        }
                    )

    write_csv(output_dir / "variant_metrics.csv", rows)

    default_tau = taus[0]
    summary_rows: list[dict[str, Any]] = []
    keys = [
        "global_valid_prediction_count",
        "local_voxels_inside_global",
        "in_front_local_ratio",
        "below_floor_local_count",
        "above_ceiling_local_count",
        "predicted_unmeasured_count",
        "later_measured_count",
        "later_measured_fraction",
        "future_measured_coverage",
        "occupied_precision",
        "free_precision",
        "brier_occupied",
        "accuracy_at_0p5",
        "roc_auc_occupied_prob",
    ]
    for variant in VARIANTS:
        group = [
            row
            for row in rows
            if row["variant"] == variant
            and row["future_source"] == "sc_future"
            and float(row["tau"]) == float(default_tau)
        ]
        agg = aggregate(group, keys)
        summary_rows.append({"variant": variant, "tau": float(default_tau), **agg})

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
        brier = row.get("brier_occupied")
        later = row.get("later_measured_fraction")
        front = row.get("in_front_local_ratio")
        return (
            float(brier) if brier is not None else 999.0,
            -float(later) if later is not None else 0.0,
            -float(front) if front is not None else 0.0,
        )

    ranked = sorted(summary_rows, key=rank_key)
    for idx, row in enumerate(ranked, start=1):
        row["rank_by_brier_then_overlap"] = int(idx)
    rank_lookup = {row["variant"]: row["rank_by_brier_then_overlap"] for row in ranked}
    for row in summary_rows:
        row["rank_by_brier_then_overlap"] = rank_lookup[row["variant"]]
    default_row = next((row for row in summary_rows if row["variant"] == "current_default"), None)
    best_row = ranked[0] if ranked else None
    default_rank = default_row.get("rank_by_brier_then_overlap") if default_row else None
    improvement = None
    if default_row and best_row and default_row.get("brier_occupied") is not None and best_row.get("brier_occupied") is not None:
        improvement = float(default_row["brier_occupied"]) - float(best_row["brier_occupied"])

    likely_alignment_bug = bool(
        best_row
        and best_row["variant"] != "current_default"
        and improvement is not None
        and improvement > 0.02
        and float(best_row.get("in_front_local_ratio") or 0.0) > 0.8
    )

    plot_paths = {
        "variant_topdown_grid_step0": plot_variant_topdown(step0_predictions, output_dir / "variant_topdown_grid_step0.png"),
        "variant_reliability_comparison": plot_variant_metric(
            summary_rows,
            "brier_occupied",
            output_dir / "variant_reliability_comparison.png",
            "Alignment variant occupied Brier",
            "Brier score",
        ),
        "variant_future_eval_comparison": plot_variant_metric(
            summary_rows,
            "later_measured_fraction",
            output_dir / "variant_future_eval_comparison.png",
            "Alignment variant later-measured fraction",
            "later measured fraction",
        ),
    }

    summary = {
        "stage": "Stage 4A-6.2 diagnostic alignment variant sweep",
        "sc_episode_dir": str(sc_episode),
        "empty_episode_dir": str(Path(args.empty_episode_dir).resolve()) if args.empty_episode_dir else "",
        "output_dir": str(output_dir),
        "variants": list(VARIANTS),
        "taus": taus,
        "default_tau_for_ranking": float(default_tau),
        "summary_rows": summary_rows,
        "best_variant": best_row["variant"] if best_row else None,
        "default_variant_rank": default_rank,
        "brier_improvement_vs_default": improvement,
        "likely_alignment_bug": likely_alignment_bug,
        "future_observations_usage": EVAL_ONLY_NOTE,
        "outputs": {
            **plot_paths,
            "variant_metrics_csv": str(output_dir / "variant_metrics.csv"),
            "variant_metrics_json": str(output_dir / "variant_metrics.json"),
        },
        "planning_or_training_used": False,
        "prediction_writeback": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_a_star": False,
        "prediction_blocks_rays": False,
    }
    save_json(output_dir / "variant_metrics.json", summary)

    recommendation = (
        "A non-default diagnostic transform fits later sensor measurements materially better; inspect axis/yaw convention before rerunning SC-aware smoke."
        if likely_alignment_bug
        else "No diagnostic-only axis/yaw variant materially beats the default projection; alignment is not the primary suspect from this sweep."
    )
    md = [
        "# Stage 4A-6.2 Alignment Variant Sweep",
        "",
        f"- {EVAL_ONLY_NOTE}",
        f"- Variants tested: `{', '.join(VARIANTS)}`.",
        f"- Best diagnostic variant: `{summary['best_variant']}`.",
        f"- Default variant rank: `{default_rank}`.",
        f"- Brier improvement vs default: `{improvement}`.",
        f"- Likely alignment bug: `{likely_alignment_bug}`.",
        "",
        recommendation,
    ]
    (output_dir / "variant_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (output_dir / "recommendation_alignment.md").write_text(recommendation + "\n", encoding="utf-8")

    print(f"Alignment variant sweep complete: {output_dir}")
    print("Diagnostic variants were not used for planning or scoring.")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc_episode_dir", type=Path, required=True)
    parser.add_argument("--empty_episode_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--taus", default="0.1,0.3,0.5")
    return parser.parse_args()


def main() -> None:
    run_sweep(parse_args())


if __name__ == "__main__":
    main()
