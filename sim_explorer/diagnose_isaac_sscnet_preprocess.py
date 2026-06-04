#!/usr/bin/env python3
"""Compare Isaac SSCNet preprocessing debug tensors with real NYU samples."""

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

from isaac_sscnet_preprocess import DEFAULT_HIGHRES_DIMS, load_json

EVAL_ONLY_NOTE = "Preprocessing diagnostics are read-only and do not run planning or training."


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


def percentiles(values: np.ndarray, prefix: str) -> dict[str, float | None]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {f"{prefix}_{name}": None for name in ("min", "p1", "p5", "p50", "p95", "p99", "max", "mean")}
    pct = np.percentile(values.astype(np.float64), [1, 5, 50, 95, 99])
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p1": float(pct[0]),
        f"{prefix}_p5": float(pct[1]),
        f"{prefix}_p50": float(pct[2]),
        f"{prefix}_p95": float(pct[3]),
        f"{prefix}_p99": float(pct[4]),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_mean": float(np.mean(values)),
    }


def position_axis_stats(position: np.ndarray, valid_mask: np.ndarray | None, prefix: str) -> dict[str, Any]:
    position = np.asarray(position)
    if valid_mask is None:
        mask = position.reshape(-1) > 0
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    flat = position.reshape(-1).astype(np.int64)
    flat_valid = flat[mask]
    out: dict[str, Any] = {
        f"{prefix}_valid_position_pixels": int(flat_valid.size),
        f"{prefix}_valid_position_ratio": float(flat_valid.size / max(1, flat.size)),
        f"{prefix}_position_zero_count": int(np.count_nonzero(flat == 0)),
        f"{prefix}_position_nonzero_ratio": float(np.count_nonzero(flat != 0) / max(1, flat.size)),
        f"{prefix}_position_min": int(flat.min()) if flat.size else None,
        f"{prefix}_position_max": int(flat.max()) if flat.size else None,
    }
    if flat_valid.size:
        dims = tuple(int(v) for v in DEFAULT_HIGHRES_DIMS)
        clipped = flat_valid[(flat_valid >= 0) & (flat_valid < int(np.prod(dims)))]
        if clipped.size:
            ix, iy, iz = np.unravel_index(clipped, dims)
            for axis_name, vals in (("x_right_idx", ix), ("y_up_idx", iy), ("z_forward_idx", iz)):
                vals64 = vals.astype(np.float64)
                out[f"{prefix}_{axis_name}_min"] = float(vals64.min())
                out[f"{prefix}_{axis_name}_p50"] = float(np.percentile(vals64, 50))
                out[f"{prefix}_{axis_name}_max"] = float(vals64.max())
                out[f"{prefix}_{axis_name}_mean"] = float(vals64.mean())
    return out


def coarse_hist(values: np.ndarray, bins: int = 24, value_range: tuple[float, float] | None = None) -> list[int]:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0 for _ in range(bins)]
    hist, _ = np.histogram(values, bins=bins, range=value_range)
    return [int(v) for v in hist.tolist()]


def step_from_path(path: Path) -> int:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot parse step from {path}")
    return int(digits)


def load_preprocess_debug(step_dir: Path) -> dict[str, np.ndarray | str]:
    debug_path = step_dir / "sscnet_input_debug.npz"
    if not debug_path.is_file():
        raise FileNotFoundError(debug_path)
    with np.load(debug_path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def isaac_step_rows(sc_episode_dir: Path) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    camera_info_path = sc_episode_dir / "camera_info.json"
    camera_info = load_json(camera_info_path) if camera_info_path.is_file() else {}
    rows: list[dict[str, Any]] = []
    depth_images: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    max_flat = int(np.prod(DEFAULT_HIGHRES_DIMS)) - 1

    for depth_path in sorted(sc_episode_dir.glob("depth_*.npy")):
        step = step_from_path(depth_path)
        step_dir = sc_episode_dir / f"prediction_step{step:03d}"
        if not step_dir.is_dir():
            continue
        pose_path = sc_episode_dir / f"pose_{step:03d}.json"
        pose = load_json(pose_path) if pose_path.is_file() else {}
        debug = load_preprocess_debug(step_dir)
        original_depth = np.asarray(debug["original_depth"], dtype=np.float32)
        depth_input = np.asarray(debug["sscnet_depth_input"], dtype=np.float32)
        position = np.asarray(debug["sscnet_position"], dtype=np.int64)
        valid_pos = np.asarray(debug["valid_position_mask"], dtype=bool)
        valid_depth = np.asarray(debug["resized_valid_depth_mask"], dtype=bool)
        local_x = np.asarray(debug.get("local_x_right", np.zeros_like(depth_input)), dtype=np.float32)
        local_y = np.asarray(debug.get("local_y_up", np.zeros_like(depth_input)), dtype=np.float32)
        local_z = np.asarray(debug.get("local_z_forward", np.zeros_like(depth_input)), dtype=np.float32)

        finite_raw = np.isfinite(original_depth) & (original_depth > 0.0)
        finite_input = np.isfinite(depth_input) & (depth_input > 0.0)
        row: dict[str, Any] = {
            "step": int(step),
            "depth_path": str(depth_path),
            "debug_npz": str(step_dir / "sscnet_input_debug.npz"),
            "pose_path": str(pose_path),
            "camera_info_path": str(camera_info_path),
            "camera_info_keys": ",".join(sorted(camera_info.keys())),
            "yaw_rad": float(pose.get("yaw_rad", np.nan)),
            "original_depth_shape": "x".join(str(v) for v in original_depth.shape),
            "depth_input_shape": "x".join(str(v) for v in depth_input.shape),
            "position_shape": "x".join(str(v) for v in position.shape),
            "position_dtype": str(position.dtype),
            "valid_depth_pixels_raw": int(np.count_nonzero(finite_raw)),
            "valid_depth_pixels_resized": int(np.count_nonzero(valid_depth)),
            "valid_depth_ratio_raw": float(np.count_nonzero(finite_raw) / original_depth.size),
            "valid_depth_ratio_resized": float(np.count_nonzero(valid_depth) / valid_depth.size),
            "position_index_hist_json": json.dumps(
                coarse_hist(position[valid_pos], bins=24, value_range=(0, max_flat))
            ),
            "position_valid_ratio_minus_nonzero_ratio": float(
                (np.count_nonzero(valid_pos) - np.count_nonzero(position != 0)) / max(1, position.size)
            ),
        }
        row.update(percentiles(original_depth[finite_raw], "raw_depth_m"))
        row.update(percentiles(depth_input[finite_input], "depth_input_m"))
        row.update(position_axis_stats(position, valid_pos, "isaac"))
        for name, values in (("local_x_right_m", local_x), ("local_y_up_m", local_y), ("local_z_forward_m", local_z)):
            row.update(percentiles(values[valid_pos], name))
        rows.append(row)
        depth_images.append(depth_input)
        valid_masks.append(valid_pos)
    return rows, depth_images, valid_masks


def nyu_rows(nyu_test_dir: Path, max_samples: int) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    max_flat = int(np.prod(DEFAULT_HIGHRES_DIMS)) - 1
    for idx, sample_path in enumerate(sorted(nyu_test_dir.glob("*.npz"))[: int(max_samples)]):
        with np.load(sample_path, allow_pickle=False) as data:
            depth = np.asarray(data["depth"], dtype=np.float32)
            if depth.ndim == 3 and depth.shape[0] == 1:
                depth = depth[0]
            position = np.asarray(data["position"], dtype=np.int64)
        valid_depth = np.isfinite(depth) & (depth > 0.0)
        valid_pos_proxy = position > 0
        row: dict[str, Any] = {
            "sample_index": int(idx),
            "sample_path": str(sample_path),
            "depth_shape": "x".join(str(v) for v in depth.shape),
            "position_shape": "x".join(str(v) for v in position.shape),
            "position_dtype": str(position.dtype),
            "valid_depth_pixels": int(np.count_nonzero(valid_depth)),
            "valid_depth_ratio": float(np.count_nonzero(valid_depth) / depth.size),
            "position_index_hist_json": json.dumps(
                coarse_hist(position[valid_pos_proxy], bins=24, value_range=(0, max_flat))
            ),
        }
        row.update(percentiles(depth[valid_depth], "depth_m"))
        row.update(position_axis_stats(position, valid_pos_proxy, "nyu"))
        rows.append(row)
        depths.append(depth)
        masks.append(valid_pos_proxy)
    return rows, depths, masks


def mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(float(row[key]))]
    return float(np.mean(vals)) if vals else None


def plot_depth_hist(isaac_rows: list[dict[str, Any]], nyu_rows_: list[dict[str, Any]], out_path: Path) -> str:
    isaac_values: list[float] = []
    nyu_values: list[float] = []
    # Use percentiles as compact samples to avoid reloading all arrays here.
    for row in isaac_rows:
        for key in ("depth_input_m_p1", "depth_input_m_p5", "depth_input_m_p50", "depth_input_m_p95", "depth_input_m_p99"):
            if row.get(key) is not None:
                isaac_values.append(float(row[key]))
    for row in nyu_rows_:
        for key in ("depth_m_p1", "depth_m_p5", "depth_m_p50", "depth_m_p95", "depth_m_p99"):
            if row.get(key) is not None:
                nyu_values.append(float(row[key]))
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    ax.hist(isaac_values, bins=20, alpha=0.65, color="#2563eb", label="Isaac percentile samples")
    ax.hist(nyu_values, bins=20, alpha=0.65, color="#f97316", label="NYU percentile samples")
    ax.set_xlabel("depth (m)")
    ax.set_ylabel("count")
    ax.set_title("Depth distribution summary")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def plot_valid_ratio(isaac_rows: list[dict[str, Any]], nyu_rows_: list[dict[str, Any]], out_path: Path) -> str:
    labels = ["Isaac valid position", "Isaac nonzero position", "NYU nonzero position", "Isaac valid depth", "NYU valid depth"]
    vals = [
        mean_of(isaac_rows, "isaac_valid_position_ratio"),
        mean_of(isaac_rows, "isaac_position_nonzero_ratio"),
        mean_of(nyu_rows_, "nyu_position_nonzero_ratio"),
        mean_of(isaac_rows, "valid_depth_ratio_resized"),
        mean_of(nyu_rows_, "valid_depth_ratio"),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.bar(labels, [0.0 if v is None else float(v) for v in vals], color=["#2563eb", "#38bdf8", "#f97316", "#0f766e", "#a16207"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ratio")
    ax.set_title("Valid depth and position ratios")
    ax.tick_params(axis="x", labelrotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def plot_position_hist(isaac_rows: list[dict[str, Any]], nyu_rows_: list[dict[str, Any]], out_path: Path) -> str:
    def summed(rows: list[dict[str, Any]]) -> np.ndarray:
        acc = None
        for row in rows:
            hist = np.asarray(json.loads(str(row["position_index_hist_json"])), dtype=np.float64)
            acc = hist if acc is None else acc + hist
        if acc is None:
            return np.zeros(24, dtype=np.float64)
        return acc / max(1.0, float(np.sum(acc)))

    isaac_hist = summed(isaac_rows)
    nyu_hist = summed(nyu_rows_)
    x = np.arange(len(isaac_hist))
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.plot(x, isaac_hist, marker="o", color="#2563eb", label="Isaac")
    ax.plot(x, nyu_hist, marker="o", color="#f97316", label="NYU")
    ax.set_xlabel("coarse flat position-index bin")
    ax.set_ylabel("normalized count")
    ax.set_title("Position index distribution")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def image_grid(images: list[np.ndarray], out_path: Path, title: str, cmap: str = "viridis") -> str:
    if not images:
        out_path.write_text("no images", encoding="utf-8")
        return str(out_path)
    n = len(images)
    cols = min(5, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, image in zip(axes_arr, images):
        ax.imshow(image, cmap=cmap, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_arr[len(images) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def run_diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    sc_episode = Path(args.sc_episode_dir).resolve()
    nyu_dir = Path(args.nyu_test_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    isaac_rows_, depth_inputs, valid_masks = isaac_step_rows(sc_episode)
    nyu_rows_, nyu_depths, nyu_masks = nyu_rows(nyu_dir, int(args.max_nyu_samples))

    write_csv(output_dir / "preprocess_stats_isaac_steps.csv", isaac_rows_)
    write_csv(output_dir / "preprocess_stats_nyu_samples.csv", nyu_rows_)

    outputs = {
        "depth_hist_isaac_vs_nyu": plot_depth_hist(isaac_rows_, nyu_rows_, output_dir / "depth_hist_isaac_vs_nyu.png"),
        "position_valid_ratio_isaac_vs_nyu": plot_valid_ratio(
            isaac_rows_, nyu_rows_, output_dir / "position_valid_ratio_isaac_vs_nyu.png"
        ),
        "position_index_hist_isaac_vs_nyu": plot_position_hist(
            isaac_rows_, nyu_rows_, output_dir / "position_index_hist_isaac_vs_nyu.png"
        ),
        "valid_position_mask_grid": image_grid(
            [mask.astype(np.float32) for mask in valid_masks[:5]] + [mask.astype(np.float32) for mask in nyu_masks[:5]],
            output_dir / "valid_position_mask_grid.png",
            "Valid position masks: Isaac steps then NYU samples",
            cmap="gray",
        ),
        "depth_input_grid": image_grid(
            [arr for arr in depth_inputs[:5]] + [arr for arr in nyu_depths[:5]],
            output_dir / "depth_input_grid.png",
            "Depth inputs: Isaac steps then NYU samples",
            cmap="magma",
        ),
    }

    isaac_depth_mean = mean_of(isaac_rows_, "depth_input_m_mean")
    nyu_depth_mean = mean_of(nyu_rows_, "depth_m_mean")
    isaac_depth_p99 = mean_of(isaac_rows_, "depth_input_m_p99")
    nyu_depth_p99 = mean_of(nyu_rows_, "depth_m_p99")
    isaac_valid_pos = mean_of(isaac_rows_, "isaac_valid_position_ratio")
    nyu_valid_pos = mean_of(nyu_rows_, "nyu_valid_position_ratio")
    isaac_zero = mean_of(isaac_rows_, "isaac_position_zero_count")
    nyu_zero = mean_of(nyu_rows_, "nyu_position_zero_count")
    depth_mean_delta = None
    if isaac_depth_mean is not None and nyu_depth_mean is not None:
        depth_mean_delta = float(isaac_depth_mean - nyu_depth_mean)
    valid_ratio_delta = None
    if isaac_valid_pos is not None and nyu_valid_pos is not None:
        valid_ratio_delta = float(isaac_valid_pos - nyu_valid_pos)

    suspicious: list[str] = []
    if isaac_depth_p99 is not None and nyu_depth_p99 is not None and isaac_depth_p99 > nyu_depth_p99 * 1.25:
        suspicious.append("Isaac depth upper tail is substantially longer than NYU.")
    if valid_ratio_delta is not None and abs(valid_ratio_delta) > 0.15:
        suspicious.append("Isaac valid position ratio differs strongly from NYU nonzero-position ratio.")
    if isaac_valid_pos is not None and isaac_valid_pos < 0.45:
        suspicious.append("Isaac valid position mask is sparse for a nominal 480x640 depth input.")
    if not suspicious:
        suspicious.append("No single preprocessing statistic alone proves a scale/convention bug.")

    summary = {
        "stage": "Stage 4A-6.2 Isaac vs NYU SSCNet preprocessing diagnostics",
        "sc_episode_dir": str(sc_episode),
        "nyu_test_dir": str(nyu_dir),
        "output_dir": str(output_dir),
        "max_nyu_samples": int(args.max_nyu_samples),
        "isaac_step_count": len(isaac_rows_),
        "nyu_sample_count": len(nyu_rows_),
        "isaac_depth_mean_m": isaac_depth_mean,
        "nyu_depth_mean_m": nyu_depth_mean,
        "depth_mean_delta_isaac_minus_nyu_m": depth_mean_delta,
        "isaac_depth_p99_m": isaac_depth_p99,
        "nyu_depth_p99_m": nyu_depth_p99,
        "valid_position_ratio_isaac": isaac_valid_pos,
        "valid_position_ratio_nyu_proxy": nyu_valid_pos,
        "valid_position_ratio_delta": valid_ratio_delta,
        "position_zero_count_isaac_mean": isaac_zero,
        "position_zero_count_nyu_mean": nyu_zero,
        "suspicious_differences": suspicious,
        "position_convention_check": (
            "Both Isaac and NYU position maps are flat high-resolution voxel indices consumed by "
            "Project2Dto3D scatter_max and later permuted to D,H,W."
        ),
        "evaluation_only_note": EVAL_ONLY_NOTE,
        "outputs": outputs,
    }
    save_json(output_dir / "preprocess_comparison_summary.json", summary)

    md = [
        "# Stage 4A-6.2 Preprocessing Diagnostics",
        "",
        f"- {EVAL_ONLY_NOTE}",
        f"- Isaac steps: `{len(isaac_rows_)}`.",
        f"- NYU samples: `{len(nyu_rows_)}`.",
        f"- Isaac mean depth: `{isaac_depth_mean}` m.",
        f"- NYU mean depth: `{nyu_depth_mean}` m.",
        f"- Isaac depth p99: `{isaac_depth_p99}` m.",
        f"- NYU depth p99: `{nyu_depth_p99}` m.",
        f"- Isaac valid position ratio: `{isaac_valid_pos}`.",
        f"- NYU position nonzero proxy ratio: `{nyu_valid_pos}`.",
        f"- Position ratio delta: `{valid_ratio_delta}`.",
        "",
        "Suspicious differences:",
    ]
    md.extend(f"- {item}" for item in suspicious)
    md.extend(
        [
            "",
            "Conclusion: preprocessing remains provisional; this diagnostic compares distributions only and does not modify the default map_predict behavior.",
        ]
    )
    (output_dir / "preprocess_comparison_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Preprocess diagnostics complete: {output_dir}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sc_episode_dir", type=Path, required=True)
    parser.add_argument("--nyu_test_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_nyu_samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    run_diagnostics(parse_args())


if __name__ == "__main__":
    main()
