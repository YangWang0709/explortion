#!/usr/bin/env python3
"""Diagnose local SSCNet prediction to global observed-map alignment."""

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

from isaac_sscnet_preprocess import DEFAULT_LOCAL_VOLUME_M, load_json
from sim_paper_expert import UNKNOWN

EVAL_ONLY_NOTE = "Global alignment diagnostics are read-only and do not affect planning or scoring."


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


def pose_origin_yaw(pose: dict[str, Any]) -> tuple[np.ndarray, float]:
    origin = np.asarray(pose["position"], dtype=np.float64)
    yaw = float(pose.get("yaw_rad", math.radians(float(pose.get("yaw_deg", 0.0)))))
    return origin, yaw


def step_from_path(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse step from {path}")
    return int(digits)


def local_grid_world(
    pose: dict[str, Any],
    local_shape: tuple[int, int, int] = (60, 36, 60),
    local_volume_m: tuple[float, float, float] = DEFAULT_LOCAL_VOLUME_M,
    local_lowres_voxel_size: float = 0.08,
) -> dict[str, np.ndarray]:
    iz, iy, ix = np.indices(local_shape, dtype=np.float32)
    x_right = -0.5 * local_volume_m[0] + (ix + 0.5) * float(local_lowres_voxel_size)
    y_up = -0.5 * local_volume_m[1] + (iy + 0.5) * float(local_lowres_voxel_size)
    z_forward = (iz + 0.5) * float(local_lowres_voxel_size)
    origin, yaw = pose_origin_yaw(pose)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    world_x = origin[0] + cos_yaw * z_forward - sin_yaw * x_right
    world_y = origin[1] + sin_yaw * z_forward + cos_yaw * x_right
    world_z = origin[2] + y_up
    return {
        "x_right": x_right,
        "y_up": y_up,
        "z_forward": z_forward,
        "world_x": world_x,
        "world_y": world_y,
        "world_z": world_z,
    }


def global_centers(shape: tuple[int, int, int], bounds: dict[str, tuple[float, float]], voxel_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gx, gy, gz = np.indices(shape, dtype=np.float32)
    wx = bounds["x"][0] + (gx + 0.5) * float(voxel_size)
    wy = bounds["y"][0] + (gy + 0.5) * float(voxel_size)
    wz = bounds["z"][0] + (gz + 0.5) * float(voxel_size)
    return wx, wy, wz


def inverse_local_from_world(
    wx: np.ndarray,
    wy: np.ndarray,
    wz: np.ndarray,
    pose: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin, yaw = pose_origin_yaw(pose)
    dx = wx - origin[0]
    dy = wy - origin[1]
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    z_forward = dx * cos_yaw + dy * sin_yaw
    x_right = -dx * sin_yaw + dy * cos_yaw
    y_up = wz - origin[2]
    return x_right, y_up, z_forward


def inside_local_volume(x_right: np.ndarray, y_up: np.ndarray, z_forward: np.ndarray) -> np.ndarray:
    return (
        (x_right >= -2.4)
        & (x_right < 2.4)
        & (y_up >= -1.44)
        & (y_up < 1.44)
        & (z_forward >= 0.0)
        & (z_forward < 4.8)
    )


def stats(values: np.ndarray) -> dict[str, float | None]:
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"min": None, "p5": None, "p50": None, "p95": None, "max": None, "mean": None, "std": None}
    pct = np.percentile(vals.astype(np.float64), [5, 50, 95])
    return {
        "min": float(vals.min()),
        "p5": float(pct[0]),
        "p50": float(pct[1]),
        "p95": float(pct[2]),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
    }


def load_global_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "pred_class": np.asarray(data["global_pred_class"], dtype=np.uint8),
            "confidence": np.asarray(data["global_confidence"], dtype=np.float32),
            "free_prob": np.asarray(data["global_free_prob"], dtype=np.float32),
            "occupied_prob": np.asarray(data["global_occupied_prob"], dtype=np.float32),
            "valid": np.asarray(data["global_prediction_valid"], dtype=bool),
        }


def load_local_prediction(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {
            "pred_class": np.asarray(data["pred_class"], dtype=np.uint8),
            "confidence": np.asarray(data["confidence"], dtype=np.float32),
            "free_prob": np.asarray(data["free_prob"], dtype=np.float32),
            "occupied_prob": np.asarray(data["occupied_prob"], dtype=np.float32),
        }


def project_topdown(mask: np.ndarray) -> np.ndarray:
    return np.max(np.asarray(mask), axis=2)


def observed_rgb(observed: np.ndarray) -> np.ndarray:
    rgb = np.ones((*observed.shape[:2], 3), dtype=np.float32)
    rgb[project_topdown(observed == UNKNOWN)] = (0.94, 0.94, 0.94)
    rgb[project_topdown(observed == 0)] = (0.75, 0.80, 0.84)
    rgb[project_topdown(observed == 1)] = (0.10, 0.10, 0.10)
    return rgb


def save_topdown(path: Path, observed: np.ndarray, overlay: np.ndarray, title: str, color: tuple[float, float, float]) -> str:
    rgb = observed_rgb(observed)
    top = project_topdown(overlay)
    rgb[top] = color
    fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("world x grid")
    ax.set_ylabel("world y grid")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def frustum_corners_world(pose: dict[str, Any]) -> np.ndarray:
    origin, yaw = pose_origin_yaw(pose)
    corners_local = np.array(
        [
            [-2.4, 0.0],
            [2.4, 0.0],
            [2.4, 4.8],
            [-2.4, 4.8],
            [-2.4, 0.0],
        ],
        dtype=np.float64,
    )
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    x_right = corners_local[:, 0]
    z_forward = corners_local[:, 1]
    wx = origin[0] + cos_yaw * z_forward - sin_yaw * x_right
    wy = origin[1] + sin_yaw * z_forward + cos_yaw * x_right
    return np.stack([wx, wy], axis=1)


def save_frustum(path: Path, observed: np.ndarray, bounds: dict[str, tuple[float, float]], pose: dict[str, Any], title: str) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 6.5), constrained_layout=True)
    ax.imshow(
        np.transpose(observed_rgb(observed), (1, 0, 2)),
        origin="lower",
        extent=[bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]],
        interpolation="nearest",
    )
    corners = frustum_corners_world(pose)
    origin, yaw = pose_origin_yaw(pose)
    ax.plot(corners[:, 0], corners[:, 1], color="#2563eb", linewidth=2.0, label="local SSC volume footprint")
    ax.scatter([origin[0]], [origin[1]], color="#be123c", s=55, label="camera")
    ax.arrow(origin[0], origin[1], math.cos(yaw) * 0.65, math.sin(yaw) * 0.65, color="#be123c", width=0.025)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def save_slices(path: Path, observed: np.ndarray, prediction: dict[str, np.ndarray], tau: float, title: str) -> str:
    ks = [max(0, min(observed.shape[2] - 1, k)) for k in (8, 12, 16)]
    fig, axes = plt.subplots(2, len(ks), figsize=(4.0 * len(ks), 7.0), constrained_layout=True)
    valid_tau = prediction["valid"] & (prediction["confidence"] >= float(tau))
    occupied = valid_tau & (prediction["occupied_prob"] >= 0.5)
    for col, k in enumerate(ks):
        obs_slice = observed[:, :, k]
        axes[0, col].imshow(obs_slice.T, origin="lower", cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        axes[0, col].set_title(f"observed z={k}")
        overlay = np.zeros((*obs_slice.shape, 3), dtype=np.float32)
        overlay[valid_tau[:, :, k]] = (0.2, 0.4, 0.95)
        overlay[occupied[:, :, k]] = (0.9, 0.1, 0.1)
        axes[1, col].imshow(np.transpose(overlay, (1, 0, 2)), origin="lower", interpolation="nearest")
        axes[1, col].set_title(f"prediction z={k}")
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def save_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> str:
    vals = np.asarray(values)
    vals = vals[np.isfinite(vals)]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), constrained_layout=True)
    if vals.size:
        ax.hist(vals, bins=40, color="#2563eb", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return str(path)


def row_for_step(
    step: int,
    sc_episode: Path,
    output_dir: Path,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    tau: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    observed = np.asarray(np.load(sc_episode / f"observed_state_step{step:03d}.npy"), dtype=np.int8)
    pose = load_json(sc_episode / f"pose_{step:03d}.json")
    pred_dir = sc_episode / f"prediction_step{step:03d}"
    global_pred = load_global_prediction(pred_dir / "global_prediction_layer.npz")
    local_pred = load_local_prediction(pred_dir / "local_prediction.npz")
    summary_path = pred_dir / "prediction_alignment_summary.json"
    alignment_summary = load_json(summary_path) if summary_path.is_file() else {}

    valid = global_pred["valid"]
    confidence = global_pred["confidence"]
    occupied_prob = global_pred["occupied_prob"]
    valid_tau = valid & (confidence >= float(tau))
    predicted_unmeasured = valid_tau & (observed == UNKNOWN)
    predicted_occupied = valid_tau & (occupied_prob >= 0.5)

    wx, wy, wz = global_centers(observed.shape, bounds, voxel_size)
    x_right, y_up, z_forward = inverse_local_from_world(wx, wy, wz, pose)
    in_front = z_forward > 0.0
    in_volume = inside_local_volume(x_right, y_up, z_forward)

    local_world = local_grid_world(pose)
    local_inside_global = (
        (local_world["world_x"] >= bounds["x"][0])
        & (local_world["world_x"] < bounds["x"][1])
        & (local_world["world_y"] >= bounds["y"][0])
        & (local_world["world_y"] < bounds["y"][1])
        & (local_world["world_z"] >= bounds["z"][0])
        & (local_world["world_z"] < bounds["z"][1])
    )
    below_floor = local_world["world_z"] < bounds["z"][0]
    above_ceiling = local_world["world_z"] >= bounds["z"][1]

    valid_count = int(np.count_nonzero(valid))
    local_inside_count = int(np.count_nonzero(local_inside_global))
    duplicate_rate = 1.0 - (valid_count / max(1, local_inside_count))
    row: dict[str, Any] = {
        "step": int(step),
        "observed_state": str(sc_episode / f"observed_state_step{step:03d}.npy"),
        "prediction_npz": str(pred_dir / "global_prediction_layer.npz"),
        "local_prediction_npz": str(pred_dir / "local_prediction.npz"),
        "prediction_summary_json": str(summary_path),
        "tau": float(tau),
        "local_voxels_total": int(np.prod(local_pred["pred_class"].shape)),
        "local_voxels_inside_global": local_inside_count,
        "inside_global_bounds_ratio": float(local_inside_count / max(1, int(np.prod(local_pred["pred_class"].shape)))),
        "global_valid_prediction_count": valid_count,
        "local_to_global_duplicate_rate": float(duplicate_rate),
        "valid_voxels_in_front_ratio": float(np.count_nonzero(valid & in_front) / max(1, valid_count)),
        "valid_voxels_inside_expected_local_volume_ratio": float(np.count_nonzero(valid & in_volume) / max(1, valid_count)),
        "local_voxels_below_floor_count": int(np.count_nonzero(below_floor)),
        "local_voxels_above_ceiling_count": int(np.count_nonzero(above_ceiling)),
        "valid_voxels_below_floor_count": int(np.count_nonzero(valid & (wz < bounds["z"][0]))),
        "valid_voxels_above_ceiling_count": int(np.count_nonzero(valid & (wz >= bounds["z"][1]))),
        "predicted_unmeasured_count": int(np.count_nonzero(predicted_unmeasured)),
        "predicted_unmeasured_in_current_frustum_count": int(np.count_nonzero(predicted_unmeasured & in_volume & in_front)),
        "predicted_unmeasured_in_current_frustum_ratio": float(
            np.count_nonzero(predicted_unmeasured & in_volume & in_front) / max(1, int(np.count_nonzero(predicted_unmeasured)))
        ),
        "predicted_occupied_count": int(np.count_nonzero(predicted_occupied)),
        "predicted_occupied_ratio_valid_tau": float(np.count_nonzero(predicted_occupied) / max(1, int(np.count_nonzero(valid_tau)))),
        "confidence_mean_valid": float(np.mean(confidence[valid])) if valid_count else None,
        "confidence_std_valid": float(np.std(confidence[valid])) if valid_count else None,
        "occupied_prob_mean_valid": float(np.mean(occupied_prob[valid])) if valid_count else None,
        "occupied_prob_std_valid": float(np.std(occupied_prob[valid])) if valid_count else None,
        "pred_valid_observed_free_count": int(np.count_nonzero(valid & (observed == 0))),
        "pred_valid_observed_occupied_count": int(np.count_nonzero(valid & (observed == 1))),
        "pred_valid_observed_unknown_count": int(np.count_nonzero(valid & (observed == UNKNOWN))),
        "alignment_summary_valid_count": alignment_summary.get("global_valid_prediction_count"),
        "evaluation_only_note": EVAL_ONLY_NOTE,
    }
    z_stats = stats(wz[valid])
    row.update({f"valid_world_z_{key}": value for key, value in z_stats.items()})

    paths = {
        "frustum": save_frustum(
            output_dir / f"frustum_topdown_step{step:03d}.png",
            observed,
            bounds,
            pose,
            f"step {step}: camera and local SSC volume footprint",
        ),
        "valid": save_topdown(
            output_dir / f"prediction_valid_topdown_step{step:03d}.png",
            observed,
            valid_tau,
            f"step {step}: valid prediction (tau={tau})",
            (0.20, 0.40, 0.95),
        ),
        "occupied": save_topdown(
            output_dir / f"prediction_occupied_topdown_step{step:03d}.png",
            observed,
            predicted_occupied,
            f"step {step}: predicted occupied (tau={tau})",
            (0.90, 0.10, 0.10),
        ),
        "predicted_unmeasured": save_topdown(
            output_dir / f"predicted_unmeasured_topdown_step{step:03d}.png",
            observed,
            predicted_unmeasured,
            f"step {step}: predicted and unmeasured",
            (0.10, 0.75, 0.35),
        ),
        "slices": save_slices(
            output_dir / f"observed_vs_prediction_slices_step{step:03d}.png",
            observed,
            global_pred,
            tau,
            f"step {step}: observed slices vs prediction",
        ),
        "confidence_hist": save_hist(
            output_dir / f"confidence_hist_step{step:03d}.png",
            confidence[valid],
            f"step {step}: confidence histogram",
            "confidence",
        ),
        "occupied_prob_hist": save_hist(
            output_dir / f"occupied_prob_hist_step{step:03d}.png",
            occupied_prob[valid],
            f"step {step}: occupied probability histogram",
            "occupied probability",
        ),
        "z_distribution": save_hist(
            output_dir / f"z_distribution_step{step:03d}.png",
            wz[valid],
            f"step {step}: valid prediction world-z distribution",
            "world z (m)",
        ),
    }
    return row, paths


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else None


def write_index(output_dir: Path, image_paths: dict[int, dict[str, str]]) -> str:
    html = ["<html><body><h1>Stage 4A-6.2 Alignment Diagnostics</h1>"]
    for step, paths in sorted(image_paths.items()):
        html.append(f"<h2>Step {step}</h2><ul>")
        for name, path in sorted(paths.items()):
            rel = Path(path).name
            html.append(f'<li><a href="{rel}">{name}</a></li>')
        html.append("</ul>")
    html.append("</body></html>")
    out = output_dir / "alignment_grid_index.html"
    out.write_text("\n".join(html) + "\n", encoding="utf-8")
    return str(out)


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    sc_episode = Path(args.sc_episode_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_summary = load_json(sc_episode / "episode_summary.json")
    bounds = normalize_bounds(episode_summary.get("map_bounds"))
    voxel_size = float(episode_summary.get("voxel_size", 0.1))
    tau = float(args.tau)

    rows: list[dict[str, Any]] = []
    image_paths: dict[int, dict[str, str]] = {}
    for pred_dir in sorted(sc_episode.glob("prediction_step*")):
        if not pred_dir.is_dir():
            continue
        step = step_from_path(pred_dir)
        observed_path = sc_episode / f"observed_state_step{step:03d}.npy"
        if not observed_path.is_file():
            continue
        row, paths = row_for_step(step, sc_episode, output_dir, bounds, voxel_size, tau)
        rows.append(row)
        image_paths[step] = paths

    write_csv(output_dir / "alignment_per_step.csv", rows)
    index_path = write_index(output_dir, image_paths)

    mean_in_front = mean(rows, "valid_voxels_in_front_ratio")
    mean_inside_bounds = mean(rows, "inside_global_bounds_ratio")
    mean_in_volume = mean(rows, "valid_voxels_inside_expected_local_volume_ratio")
    mean_duplicate = mean(rows, "local_to_global_duplicate_rate")
    mean_below = mean(rows, "local_voxels_below_floor_count")
    mean_above = mean(rows, "local_voxels_above_ceiling_count")
    axis_yaw_issue = bool(
        (mean_in_front is not None and mean_in_front < 0.85)
        or (mean_in_volume is not None and mean_in_volume < 0.85)
        or any(int(row.get("valid_voxels_below_floor_count", 0)) > 0 for row in rows)
    )
    z_issue = bool(mean_below is not None and mean_below > 0)
    summary = {
        "stage": "Stage 4A-6.2 global prediction alignment diagnostics",
        "sc_episode_dir": str(sc_episode),
        "output_dir": str(output_dir),
        "tau": tau,
        "step_count": len(rows),
        "mean_valid_voxels_in_front_ratio": mean_in_front,
        "mean_inside_global_bounds_ratio": mean_inside_bounds,
        "mean_valid_voxels_inside_expected_local_volume_ratio": mean_in_volume,
        "mean_local_to_global_duplicate_rate": mean_duplicate,
        "mean_local_voxels_below_floor_count": mean_below,
        "mean_local_voxels_above_ceiling_count": mean_above,
        "likely_axis_or_yaw_issue": axis_yaw_issue,
        "z_bounds_issue": z_issue,
        "frustum_overlay_conclusion": (
            "Default projection places global valid voxels in the forward local volume."
            if not axis_yaw_issue
            else "At least one coarse alignment sanity metric is suspicious; inspect overlays and variant sweep."
        ),
        "evaluation_only_note": EVAL_ONLY_NOTE,
        "outputs": {
            "alignment_per_step_csv": str(output_dir / "alignment_per_step.csv"),
            "alignment_grid_index_html": index_path,
            "image_paths": image_paths,
        },
        "planning_or_training_used": False,
        "prediction_writeback": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_a_star": False,
        "prediction_blocks_rays": False,
    }
    save_json(output_dir / "alignment_summary.json", summary)
    md = [
        "# Stage 4A-6.2 Global Alignment Diagnostics",
        "",
        f"- {EVAL_ONLY_NOTE}",
        f"- Steps: `{len(rows)}`.",
        f"- Mean in-front ratio: `{mean_in_front}`.",
        f"- Mean inside global bounds ratio: `{mean_inside_bounds}`.",
        f"- Mean expected local-volume ratio: `{mean_in_volume}`.",
        f"- Mean duplicate rate after global projection: `{mean_duplicate}`.",
        f"- Mean local voxels below floor before clipping: `{mean_below}`.",
        f"- Mean local voxels above ceiling before clipping: `{mean_above}`.",
        f"- Likely axis/yaw issue: `{axis_yaw_issue}`.",
        f"- Z/bounds issue: `{z_issue}`.",
        "",
        summary["frustum_overlay_conclusion"],
    ]
    (output_dir / "alignment_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Global alignment diagnostics complete: {output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc_episode_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
