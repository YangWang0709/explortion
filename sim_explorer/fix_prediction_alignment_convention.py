#!/usr/bin/env python3
"""Evaluate named SSCNet local-to-global alignment conventions.

The script reprojects existing local_prediction.npz files when available. It
uses later observed maps only as delayed sensor validation and never writes
predictions into observed_state or planning inputs.
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

from evaluate_prediction_against_future_observed import (
    EVAL_ONLY_NOTE,
    calibration_bins_for_arrays,
    evaluate_prediction_arrays,
    future_measurement,
    observed_state_paths,
    parse_taus,
)
from isaac_sscnet_preprocess import (
    DEFAULT_LOCAL_VOLUME_M,
    alignment_convention_metadata,
    camera_coords_to_world,
    canonical_alignment_convention,
    local_index_to_camera_coords,
    world_to_global_grid,
)
from run_isaac_map_predict_single import align_local_prediction_to_global, normalize_bounds
from sim_paper_expert import FREE, OCCUPIED, UNKNOWN


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


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def save_projected_npz(
    path: Path,
    aligned: dict[str, Any],
    convention: str,
    observed_state_path: Path,
    local_prediction_path: Path,
) -> str:
    np.savez_compressed(
        path,
        global_pred_class=aligned["global_pred_class"].astype(np.uint8),
        global_confidence=aligned["global_confidence"].astype(np.float32),
        global_free_prob=aligned["global_free_prob"].astype(np.float32),
        global_occupied_prob=aligned["global_occupied_prob"].astype(np.float32),
        global_prediction_valid=aligned["global_prediction_valid"].astype(bool),
        observed_state_source=str(observed_state_path),
        local_prediction_source=str(local_prediction_path),
        alignment_convention=str(convention),
        alignment_stats_json=json.dumps(aligned["align_stats"], sort_keys=True),
        strict_no_observed_write=np.array(True, dtype=bool),
        read_only_note="Stage 4A-6.3 convention eval reprojection only; not used for planning",
    )
    return str(path)


def prediction_dict(aligned: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "valid": aligned["global_prediction_valid"],
        "confidence": aligned["global_confidence"],
        "occupied_prob": aligned["global_occupied_prob"],
        "free_prob": aligned["global_free_prob"],
    }


def observed_overlap(prediction: dict[str, np.ndarray], observed_state: np.ndarray, tau: float) -> dict[str, int]:
    predicted = prediction["valid"] & (prediction["confidence"] >= float(tau))
    return {
        "predicted_valid_overlap_observed_unknown": int(np.count_nonzero(predicted & (observed_state == UNKNOWN))),
        "predicted_valid_overlap_observed_free": int(np.count_nonzero(predicted & (observed_state == FREE))),
        "predicted_valid_overlap_observed_occupied": int(np.count_nonzero(predicted & (observed_state == OCCUPIED))),
    }


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
        out[key] = float(np.mean(values)) if values else None
    return out


def ece_from_bins(rows: list[dict[str, Any]]) -> float | None:
    total = sum(int(row.get("count") or 0) for row in rows)
    if total <= 0:
        return None
    acc = 0.0
    for row in rows:
        count = int(row.get("count") or 0)
        if not count:
            continue
        mean_prob = row.get("mean_occupied_prob")
        rate = row.get("empirical_occupied_rate")
        if mean_prob is None or rate is None:
            continue
        acc += (count / total) * abs(float(mean_prob) - float(rate))
    return float(acc)


def project_topdown(mask: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(mask), axis=2)


def plot_topdown_compare(step: int, projections: dict[str, dict[str, Any]], output_path: Path, tau: float) -> str:
    names = list(projections)
    cols = len(names)
    fig, axes = plt.subplots(1, cols, figsize=(max(4.2 * cols, 4.2), 4.3), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_arr, names):
        pred = prediction_dict(projections[name])
        valid = pred["valid"] & (pred["confidence"] >= float(tau))
        occ = valid & (pred["occupied_prob"] >= 0.5)
        image = np.zeros((*valid.shape[:2], 3), dtype=np.float32)
        image[project_topdown(valid)] = (0.20, 0.40, 0.95)
        image[project_topdown(occ)] = (0.95, 0.12, 0.12)
        ax.imshow(np.transpose(image, (1, 0, 2)), origin="lower", interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Step {step:03d} convention reprojection")
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_bar(summary_rows: list[dict[str, Any]], key: str, output_path: Path, title: str, ylabel: str) -> str:
    rows = [row for row in summary_rows if row.get(key) is not None]
    fig, ax = plt.subplots(figsize=(max(7.5, len(rows) * 1.5), 4.5), constrained_layout=True)
    labels = [str(row["convention"]) for row in rows]
    vals = [float(row[key]) for row in rows]
    ax.bar(labels, vals, color="#2563eb")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_reliability(calibration_rows: list[dict[str, Any]], output_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color="#111827", linestyle="--", linewidth=1.0, alpha=0.65)
    colors = {"current_default_v0": "#2563eb", "xz_swap_diagnostic": "#f97316", "code_consistent_v1": "#0f766e"}
    for convention in sorted({str(row["convention"]) for row in calibration_rows}):
        rows = [row for row in calibration_rows if row["convention"] == convention and int(row.get("count") or 0) > 0]
        xs = [float(row["mean_occupied_prob"]) for row in rows if row.get("mean_occupied_prob") is not None]
        ys = [float(row["empirical_occupied_rate"]) for row in rows if row.get("empirical_occupied_rate") is not None]
        if xs and ys:
            ax.plot(xs, ys, marker="o", linewidth=1.6, color=colors.get(convention, None), label=convention)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("mean predicted occupied probability")
    ax.set_ylabel("empirical future occupied rate")
    ax.set_title("Reliability by alignment convention")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def inverse_local_index_for_coord(
    local_coord: tuple[float, float, float],
    convention: str,
    voxel_size: float = 0.08,
) -> tuple[int, int, int]:
    x_right, y_up, z_forward = local_coord
    semantic_idx = {
        "x_right": int(np.floor((x_right + DEFAULT_LOCAL_VOLUME_M[0] * 0.5) / voxel_size)),
        "y_up": int(np.floor((y_up + DEFAULT_LOCAL_VOLUME_M[1] * 0.5) / voxel_size)),
        "z_forward": int(np.floor(z_forward / voxel_size)),
    }
    order = alignment_convention_metadata(convention)["output_axis_order"]
    return tuple(int(semantic_idx[axis]) for axis in order)


def run_synthetic_blob_test(output_dir: Path, conventions: list[str]) -> dict[str, Any]:
    pose = {"position": [0.0, 0.0, 1.2], "yaw_rad": 0.0}
    bounds = {"x": [-6.0, 6.0], "y": [-6.0, 6.0], "z": [0.0, 3.0]}
    voxel_size = 0.1
    local_coord = (1.0, 0.0, 2.0)
    rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    colors = ["#2563eb", "#f97316", "#0f766e", "#7c3aed"]
    for idx, convention in enumerate(conventions):
        raw_index = inverse_local_index_for_coord(local_coord, convention)
        coords = local_index_to_camera_coords(raw_index, convention=convention)
        world = camera_coords_to_world(coords, pose)
        grid = world_to_global_grid(world, bounds, voxel_size)
        row = {
            "convention": convention,
            "blob_local_x_right": local_coord[0],
            "blob_local_y_up": local_coord[1],
            "blob_local_z_forward": local_coord[2],
            "raw_output_index0": int(raw_index[0]),
            "raw_output_index1": int(raw_index[1]),
            "raw_output_index2": int(raw_index[2]),
            "world_x": float(np.asarray(world[0])),
            "world_y": float(np.asarray(world[1])),
            "world_z": float(np.asarray(world[2])),
            "grid_x": int(np.asarray(grid[0])),
            "grid_y": int(np.asarray(grid[1])),
            "grid_z": int(np.asarray(grid[2])),
        }
        rows.append(row)
        ax.scatter(row["world_x"], row["world_y"], s=90, color=colors[idx % len(colors)], label=convention)
        ax.text(row["world_x"] + 0.05, row["world_y"] + 0.05, convention, fontsize=8)
    ax.scatter([2.0], [1.0], s=30, color="#111827", marker="x", label="expected yaw0")
    ax.set_title("Synthetic blob projection")
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.legend(loc="best", fontsize=7)
    image_path = output_dir / "synthetic_blob_projection.png"
    fig.savefig(image_path, dpi=170)
    plt.close(fig)

    csv_path = output_dir / "synthetic_blob_projection.csv"
    write_csv(csv_path, rows)
    md = [
        "# Synthetic Alignment Test",
        "",
        "- Blob local coordinate: x_right=+1.0m, y_up=0.0m, z_forward=+2.0m.",
        "- Pose: world position [0,0,1.2], yaw 0.",
        "- Expected world coordinate under simulator yaw convention: [2.0, 1.0, 1.2].",
        "- The raw output index differs by convention; the semantic local coordinate lands at the expected world point when interpreted with that convention.",
    ]
    (output_dir / "synthetic_alignment_test.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return {"rows": rows, "csv": str(csv_path), "png": str(image_path), "md": str(output_dir / "synthetic_alignment_test.md")}


def run_eval(args: argparse.Namespace) -> dict[str, Any]:
    sc_episode = Path(args.sc_episode_dir).resolve()
    empty_episode = Path(args.empty_episode_dir).resolve() if args.empty_episode_dir else None
    axis_audit_json = Path(args.axis_audit_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conventions = [canonical_alignment_convention(item.strip()) for item in str(args.conventions).split(",") if item.strip()]
    conventions = list(dict.fromkeys(conventions))
    taus = parse_taus(args.taus)
    default_tau = float(taus[0])
    max_horizon = int(args.max_horizon)

    axis_audit = load_json(axis_audit_json) if axis_audit_json.is_file() else {}
    episode_summary = load_json(sc_episode / "episode_summary.json")
    bounds = normalize_bounds(episode_summary.get("map_bounds"))
    voxel_size = float(episode_summary.get("voxel_size", 0.1))
    sc_future_paths = observed_state_paths(sc_episode)
    future_sources: list[tuple[str, Path, dict[int, Path]]] = [("sc_future", sc_episode, sc_future_paths)]
    if empty_episode and empty_episode.is_dir():
        future_sources.append(("empty_future", empty_episode, observed_state_paths(empty_episode)))

    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    per_step_paths: dict[str, list[str]] = {name: [] for name in conventions}
    step_topdown_paths: list[str] = []

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
        projections: dict[str, dict[str, Any]] = {}
        for convention in conventions:
            aligned = align_local_prediction_to_global(
                pred_class=local["pred_class"],
                confidence=local["confidence"],
                free_prob=local["free_prob"],
                occupied_prob=local["occupied_prob"],
                observed_shape=tuple(int(v) for v in observed_t.shape),
                pose=pose,
                map_bounds=bounds,
                global_voxel_size=voxel_size,
                alignment_convention=convention,
            )
            projections[convention] = aligned
            projected_path = output_dir / f"global_prediction_layer_step{step:03d}_{convention}.npz"
            save_projected_npz(projected_path, aligned, convention, observed_path, local_path)
            per_step_paths[convention].append(str(projected_path))
            pred = prediction_dict(aligned)
            overlap = observed_overlap(pred, observed_t, default_tau)
            for source_name, source_dir, future_paths in future_sources:
                future_state, used_steps = future_measurement(observed_t, future_paths, step, max_horizon=max_horizon)
                for tau in taus:
                    metrics = evaluate_prediction_arrays(pred, observed_t, future_state, tau)
                    row = {
                        "convention": convention,
                        "step": int(step),
                        "future_source": source_name,
                        "future_episode_dir": str(source_dir),
                        "future_steps_used": json.dumps(used_steps),
                        "local_prediction_npz": str(local_path),
                        "projected_prediction_npz": str(projected_path),
                        **aligned["align_stats"],
                        **overlap,
                        **metrics,
                        "future_observations_usage": EVAL_ONLY_NOTE,
                        "planning_or_training_used": False,
                        "prediction_writeback": False,
                        "prediction_used_for_traversability": False,
                        "prediction_used_for_collision": False,
                        "prediction_used_for_a_star": False,
                        "prediction_blocks_rays": False,
                    }
                    rows.append(row)
                    if source_name == "sc_future" and float(tau) == default_tau:
                        bins, _, _, _ = calibration_bins_for_arrays(
                            pred,
                            observed_t,
                            future_state,
                            tau=tau,
                            prefix={"convention": convention, "step": int(step), "future_source": source_name, "tau": float(tau)},
                        )
                        calibration_rows.extend(bins)
        step_topdown_paths.append(
            plot_topdown_compare(step, projections, output_dir / f"topdown_compare_conventions_step{step:03d}.png", default_tau)
        )

    write_csv(output_dir / "convention_metrics.csv", rows)
    write_csv(output_dir / "convention_calibration_bins.csv", calibration_rows)

    metric_keys = [
        "global_valid_prediction_count",
        "local_voxels_inside_global",
        "inside_bounds_ratio",
        "in_front_local_ratio",
        "below_floor_local_count",
        "above_ceiling_local_count",
        "predicted_valid_overlap_observed_unknown",
        "predicted_valid_overlap_observed_free",
        "predicted_valid_overlap_observed_occupied",
        "predicted_unmeasured_count",
        "later_measured_count",
        "later_measured_fraction",
        "future_measured_coverage",
        "occupied_precision",
        "free_precision",
        "brier_occupied",
        "accuracy_at_0p5",
        "mean_occupied_prob",
        "mean_confidence",
        "roc_auc_occupied_prob",
    ]
    summary_rows: list[dict[str, Any]] = []
    for convention in conventions:
        group = [
            row
            for row in rows
            if row["convention"] == convention
            and row["future_source"] == "sc_future"
            and float(row["tau"]) == default_tau
        ]
        agg = aggregate(group, metric_keys)
        agg["ece_like_occupied_prob"] = ece_from_bins(
            [row for row in calibration_rows if row["convention"] == convention and row["future_source"] == "sc_future"]
        )
        summary_rows.append({"convention": convention, "tau": default_tau, **agg})

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float]:
        brier = row.get("brier_occupied")
        later = row.get("later_measured_fraction")
        inside = row.get("inside_bounds_ratio")
        return (
            float(brier) if brier is not None else 999.0,
            -float(later) if later is not None else 0.0,
            -float(inside) if inside is not None else 0.0,
        )

    ranked = sorted(summary_rows, key=rank_key)
    for idx, row in enumerate(ranked, start=1):
        row["rank_by_brier_then_overlap"] = int(idx)
    ranks = {row["convention"]: row["rank_by_brier_then_overlap"] for row in ranked}
    for row in summary_rows:
        row["rank_by_brier_then_overlap"] = ranks[row["convention"]]

    default_row = next((row for row in summary_rows if row["convention"] == "current_default_v0"), None)
    code_row = next((row for row in summary_rows if row["convention"] == "code_consistent_v1"), None)
    best_row = ranked[0] if ranked else None
    code_supported = str(axis_audit.get("recommended_code_consistent_convention", "")) == "code_consistent_v1"
    fixed_not_worse = bool(
        default_row
        and code_row
        and code_row.get("brier_occupied") is not None
        and default_row.get("brier_occupied") is not None
        and float(code_row["brier_occupied"]) <= float(default_row["brier_occupied"]) + 1.0e-9
    )
    fixed_improvement = None
    if default_row and code_row and default_row.get("brier_occupied") is not None and code_row.get("brier_occupied") is not None:
        fixed_improvement = float(default_row["brier_occupied"]) - float(code_row["brier_occupied"])
    recommend_fixed = bool(code_supported and fixed_not_worse and fixed_improvement is not None and fixed_improvement > 0.02)
    recommendation = "code_consistent_v1" if recommend_fixed else "controlled_synthetic_validation_before_default_change"

    plot_paths = {
        "reliability_compare_conventions": plot_reliability(calibration_rows, output_dir / "reliability_compare_conventions.png"),
        "brier_compare_conventions": plot_bar(
            summary_rows,
            "brier_occupied",
            output_dir / "brier_compare_conventions.png",
            "Occupied Brier by alignment convention",
            "Brier score",
        ),
        "later_measured_fraction_compare": plot_bar(
            summary_rows,
            "later_measured_fraction",
            output_dir / "later_measured_fraction_compare.png",
            "Later measured fraction by alignment convention",
            "later measured fraction",
        ),
    }
    synthetic = run_synthetic_blob_test(output_dir, conventions)

    summary = {
        "stage": "Stage 4A-6.3 SSCNet alignment convention evaluation",
        "sc_episode_dir": str(sc_episode),
        "empty_episode_dir": str(empty_episode) if empty_episode else "",
        "axis_audit_json": str(axis_audit_json),
        "output_dir": str(output_dir),
        "conventions": conventions,
        "taus": taus,
        "default_tau_for_ranking": default_tau,
        "summary_rows": summary_rows,
        "best_diagnostic_convention": best_row["convention"] if best_row else None,
        "default_rank": default_row.get("rank_by_brier_then_overlap") if default_row else None,
        "code_consistent_rank": code_row.get("rank_by_brier_then_overlap") if code_row else None,
        "brier_improvement_code_consistent_vs_default": fixed_improvement,
        "code_consistent_supported_by_audit": code_supported,
        "recommend_fixed": recommend_fixed,
        "recommended_convention": recommendation,
        "per_step_projected_npz": per_step_paths,
        "outputs": {
            "convention_metrics_csv": str(output_dir / "convention_metrics.csv"),
            "convention_metrics_json": str(output_dir / "convention_metrics.json"),
            "convention_summary_md": str(output_dir / "convention_summary.md"),
            "topdown_compare_conventions": step_topdown_paths,
            **plot_paths,
            "synthetic_blob_projection_csv": synthetic["csv"],
            "synthetic_blob_projection_png": synthetic["png"],
            "synthetic_alignment_test_md": synthetic["md"],
        },
        "future_observations_usage": EVAL_ONLY_NOTE,
        "planning_or_training_used": False,
        "prediction_writeback": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_a_star": False,
        "prediction_blocks_rays": False,
        "target_or_ground_truth_used_for_scoring": False,
        "rl_or_ppo_training": False,
        "optimizer_step": False,
        "behavior_cloning_training": False,
        "imitation_learning_training": False,
        "sscnet_training": False,
    }
    save_json(output_dir / "convention_metrics.json", summary)

    md = [
        "# Stage 4A-6.3 Alignment Convention Evaluation",
        "",
        f"- Future observations usage: {EVAL_ONLY_NOTE}",
        f"- Conventions tested: `{', '.join(conventions)}`.",
        f"- Best diagnostic convention: `{summary['best_diagnostic_convention']}`.",
        f"- Default rank: `{summary['default_rank']}`.",
        f"- Code-consistent rank: `{summary['code_consistent_rank']}`.",
        f"- Brier improvement code-consistent vs default: `{fixed_improvement}`.",
        f"- Recommendation: `{recommendation}`.",
        "",
        "## Summary Rows",
    ]
    for row in sorted(summary_rows, key=lambda r: int(r["rank_by_brier_then_overlap"])):
        md.append(
            f"- rank {row['rank_by_brier_then_overlap']}: `{row['convention']}` "
            f"brier={row.get('brier_occupied')} later={row.get('later_measured_fraction')} "
            f"ece={row.get('ece_like_occupied_prob')}"
        )
    md.extend(
        [
            "",
            "The evaluation reprojects existing local predictions. For the full `code_consistent_v1` fix, new smoke runs must also rerun preprocessing/inference so the input `position` flattening uses the same convention.",
        ]
    )
    (output_dir / "convention_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (output_dir / "recommendation_alignment_fix.md").write_text(
        f"recommended_convention: {recommendation}\n"
        f"recommend_fixed: {recommend_fixed}\n"
        f"brier_improvement_code_consistent_vs_default: {fixed_improvement}\n",
        encoding="utf-8",
    )
    print("Stage 4A-6.3 convention evaluation complete.")
    print(f"best_diagnostic_convention: {summary['best_diagnostic_convention']}")
    print(f"recommended_convention: {recommendation}")
    print(f"output_dir: {output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc_episode_dir", type=Path, required=True)
    parser.add_argument("--empty_episode_dir", type=Path, default=None)
    parser.add_argument("--axis_audit_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--conventions", default="current_default_v0,xz_swap_diagnostic,code_consistent_v1")
    parser.add_argument("--taus", default="0.1,0.3,0.5")
    parser.add_argument("--max_horizon", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    run_eval(parse_args())


if __name__ == "__main__":
    main()
