#!/usr/bin/env python3
"""Visualize Stage 4A-5 Isaac observed_map and read-only prediction alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap


UNKNOWN = -1
FREE = 0
OCCUPIED = 1

STATE_CMAP = ListedColormap(["#30343b", "#7fb7be", "#d65f5f"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _finite_depth_limits(*depths: np.ndarray) -> tuple[float, float]:
    values = []
    for depth in depths:
        valid = depth[np.isfinite(depth) & (depth > 0.0)]
        if valid.size:
            values.append(valid)
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    return float(merged.min()), float(merged.max())


def _display_depth(depth: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(depth) & (depth > 0.0), depth, np.nan)


def _bounds(summary: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = summary["alignment_stats"]["map_bounds"]
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def _state_topdown(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == FREE, axis=2)
    topdown = np.full(observed_state.shape[:2], UNKNOWN, dtype=np.int8)
    topdown[free] = FREE
    topdown[occupied] = OCCUPIED
    return topdown


def _valid_max_topdown(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    masked = np.where(valid, values, -np.inf)
    topdown = np.max(masked, axis=2)
    topdown[~np.isfinite(topdown)] = 0.0
    return topdown.astype(np.float32)


def _imshow_world(ax, image: np.ndarray, bounds: dict[str, tuple[float, float]], **kwargs: Any):
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    return ax.imshow(image.T, origin="lower", extent=extent, interpolation="nearest", **kwargs)


def save_depth_input_image(output_dir: Path, debug: Any) -> str:
    original_depth = np.array(debug["original_depth"], dtype=np.float32)
    resized_depth = np.array(debug["sscnet_depth_input"], dtype=np.float32)
    vmin, vmax = _finite_depth_limits(original_depth, resized_depth)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(_display_depth(original_depth), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Isaac depth raw {original_depth.shape}")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("v")
    im1 = axes[1].imshow(_display_depth(resized_depth), cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"SSCNet depth input {resized_depth.shape}")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("v")
    fig.colorbar(im1, ax=axes, label="depth (m)", shrink=0.82)
    out_path = output_dir / "isaac_depth_input.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_local_prediction_slices(output_dir: Path, local: Any) -> str:
    occupied = np.array(local["occupied_prob"], dtype=np.float32)
    confidence = np.array(local["confidence"], dtype=np.float32)
    z_indices = [10, 30, 50]
    z_indices = [min(max(0, idx), occupied.shape[0] - 1) for idx in z_indices]

    fig, axes = plt.subplots(2, len(z_indices), figsize=(5 * len(z_indices), 8), constrained_layout=True)
    for col, z_idx in enumerate(z_indices):
        im = axes[0, col].imshow(occupied[z_idx], origin="lower", cmap="magma", vmin=0.0, vmax=1.0)
        axes[0, col].set_title(f"occupied_prob z={z_idx}")
        axes[0, col].set_xlabel("x_right index")
        axes[0, col].set_ylabel("y_up index")
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        im = axes[1, col].imshow(confidence[z_idx], origin="lower", cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, col].set_title(f"confidence z={z_idx}")
        axes[1, col].set_xlabel("x_right index")
        axes[1, col].set_ylabel("y_up index")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    out_path = output_dir / "local_prediction_slices.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_global_prediction_topdown(
    output_dir: Path,
    global_layer: Any,
    bounds: dict[str, tuple[float, float]],
    tau: float,
    alignment_convention: str = "",
) -> str:
    valid = np.array(global_layer["global_prediction_valid"], dtype=bool)
    confidence = np.array(global_layer["global_confidence"], dtype=np.float32)
    occupied = np.array(global_layer["global_occupied_prob"], dtype=np.float32)
    valid_tau = valid & (confidence >= float(tau))

    valid_top = np.any(valid_tau, axis=2).astype(np.float32)
    occupied_top = _valid_max_topdown(occupied, valid_tau)
    confidence_top = _valid_max_topdown(confidence, valid_tau)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    im = _imshow_world(axes[0], valid_top, bounds, cmap="Greys", vmin=0.0, vmax=1.0)
    suffix = f"\n{alignment_convention}" if alignment_convention else ""
    axes[0].set_title("prediction valid" + suffix)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    im = _imshow_world(axes[1], occupied_top, bounds, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("occupied_prob max")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    im = _imshow_world(axes[2], confidence_top, bounds, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2].set_title("confidence max")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("world x (m)")
        ax.set_ylabel("world y (m)")

    out_path = output_dir / "global_prediction_topdown.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def save_observed_vs_prediction_topdown(
    output_dir: Path,
    observed_state: np.ndarray,
    global_layer: Any,
    bounds: dict[str, tuple[float, float]],
    tau: float,
    alignment_convention: str = "",
) -> str:
    valid = np.array(global_layer["global_prediction_valid"], dtype=bool)
    confidence = np.array(global_layer["global_confidence"], dtype=np.float32)
    occupied = np.array(global_layer["global_occupied_prob"], dtype=np.float32)
    valid_tau = valid & (confidence >= float(tau))
    predicted_occupied = valid_tau & (occupied >= 0.5)

    observed_top = _state_topdown(observed_state)
    valid_top = np.any(valid_tau, axis=2)
    occupied_top = np.any(predicted_occupied, axis=2)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    _imshow_world(ax, observed_top, bounds, cmap=STATE_CMAP, norm=STATE_NORM)
    _imshow_world(ax, np.ma.masked_where(~valid_top, valid_top), bounds, cmap=ListedColormap(["#4a90e2"]), alpha=0.24)
    _imshow_world(
        ax,
        np.ma.masked_where(~occupied_top, occupied_top),
        bounds,
        cmap=ListedColormap(["#ff2d55"]),
        alpha=0.46,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    suffix = f"\n{alignment_convention}" if alignment_convention else ""
    ax.set_title("Observed state with read-only prediction overlay" + suffix)
    out_path = output_dir / "observed_vs_prediction_topdown.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_prediction_not_measured_topdown(
    output_dir: Path,
    observed_state: np.ndarray,
    global_layer: Any,
    bounds: dict[str, tuple[float, float]],
    tau: float,
    alignment_convention: str = "",
) -> tuple[str, int]:
    valid = np.array(global_layer["global_prediction_valid"], dtype=bool)
    confidence = np.array(global_layer["global_confidence"], dtype=np.float32)
    valid_tau = valid & (confidence >= float(tau))
    prediction_not_measured = valid_tau & (observed_state == UNKNOWN)
    topdown = np.any(prediction_not_measured, axis=2).astype(np.float32)
    count = int(np.count_nonzero(prediction_not_measured))

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = _imshow_world(ax, topdown, bounds, cmap="plasma", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    convention_text = f" convention={alignment_convention}" if alignment_convention else ""
    ax.set_title(f"Prediction valid and observed UNKNOWN\nvoxels={count}{convention_text}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = output_dir / "prediction_not_measured_topdown.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path), count


def generate_visualizations(output_dir: str | Path, tau: float = 0.1) -> dict[str, Any]:
    output_path = Path(output_dir).resolve()
    summary_path = output_path / "prediction_alignment_summary.json"
    debug_path = output_path / "sscnet_input_debug.npz"
    local_path = output_path / "local_prediction.npz"
    global_path = output_path / "global_prediction_layer.npz"

    for path in (summary_path, debug_path, local_path, global_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing Stage 4A-5 artifact for visualization: {path}")

    summary = load_json(summary_path)
    observed_state = np.load(summary["observed_state_source"])
    bounds = _bounds(summary)
    alignment_convention = str(summary.get("alignment_convention", ""))

    generated: dict[str, str] = {}
    with np.load(debug_path) as debug, np.load(local_path) as local, np.load(global_path) as global_layer:
        generated["isaac_depth_input"] = save_depth_input_image(output_path, debug)
        generated["local_prediction_slices"] = save_local_prediction_slices(output_path, local)
        generated["global_prediction_topdown"] = save_global_prediction_topdown(
            output_path,
            global_layer,
            bounds,
            tau=tau,
            alignment_convention=alignment_convention,
        )
        generated["observed_vs_prediction_topdown"] = save_observed_vs_prediction_topdown(
            output_path,
            observed_state,
            global_layer,
            bounds,
            tau=tau,
            alignment_convention=alignment_convention,
        )
        prediction_not_measured_path, prediction_not_measured_count = save_prediction_not_measured_topdown(
            output_path,
            observed_state,
            global_layer,
            bounds,
            tau=tau,
            alignment_convention=alignment_convention,
        )
        generated["prediction_not_measured_topdown"] = prediction_not_measured_path

    viz_summary = {
        "output_dir": str(output_path),
        "tau": float(tau),
        "alignment_convention": alignment_convention,
        "generated_images": generated,
        "prediction_not_measured_count": int(prediction_not_measured_count),
        "observed_state_modified": False,
        "prediction_read_only": True,
        "expert_or_rollout_invoked": False,
        "rl_or_training_used": False,
    }

    summary["visualization_outputs"] = viz_summary
    save_json(summary_path, summary)
    return viz_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one Stage 4A-5 Isaac prediction alignment output.")
    parser.add_argument("--output_dir", type=Path, default=Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_map_predict_single_smoke"))
    parser.add_argument("--tau", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_visualizations(output_dir=args.output_dir, tau=float(args.tau))
    print("Stage 4A-5 visualization complete")
    for name, path in summary["generated_images"].items():
        print(f"{name}: {path}")
    print(f"prediction_not_measured_count: {summary['prediction_not_measured_count']}")


if __name__ == "__main__":
    main()
