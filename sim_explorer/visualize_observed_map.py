#!/usr/bin/env python3
"""Visualize Stage 4A-1 Isaac depth and observed voxel outputs.

This script is read-only with respect to the observed maps. It does not train
models, run RL, call PredictionLayer, or run expert scoring.
"""

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

STATE_CMAP = ListedColormap(["#3b4252", "#88c0d0", "#bf616a"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)
STATE_TICKS = [UNKNOWN, FREE, OCCUPIED]
STATE_TICKLABELS = ["unknown", "free", "occupied"]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def ensure_inputs(input_dir: Path) -> tuple[list[Path], list[Path]]:
    depth_paths = sorted(input_dir.glob("depth_*.npy"))
    observed_paths = sorted(input_dir.glob("observed_state_step*.npy"))
    if len(depth_paths) < 3:
        raise FileNotFoundError(f"Expected at least 3 depth_*.npy files in {input_dir}, found {len(depth_paths)}")
    if len(observed_paths) < 3:
        raise FileNotFoundError(
            f"Expected at least 3 observed_state_step*.npy files in {input_dir}, found {len(observed_paths)}"
        )
    for required in ("camera_info.json", "scene_metadata.json", "pose_000.json", "pose_001.json", "pose_002.json"):
        path = input_dir / required
        if not path.exists():
            raise FileNotFoundError(f"Missing required metadata file: {path}")
    return depth_paths[:3], observed_paths[:3]


def count_states(grid: np.ndarray) -> dict[str, int]:
    return {
        "unknown_count": int(np.count_nonzero(grid == UNKNOWN)),
        "free_count": int(np.count_nonzero(grid == FREE)),
        "occupied_count": int(np.count_nonzero(grid == OCCUPIED)),
        "observed_count": int(np.count_nonzero(grid != UNKNOWN)),
        "total_count": int(grid.size),
    }


def observed_ratio(counts: dict[str, int]) -> float:
    return float(counts["observed_count"] / counts["total_count"]) if counts["total_count"] else 0.0


def depth_stats(depth: np.ndarray) -> dict[str, Any]:
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    return {
        "shape": list(depth.shape),
        "dtype": str(depth.dtype),
        "min": float(positive.min()) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
    }


def map_bounds_from_metadata(scene_metadata: dict[str, Any]) -> dict[str, tuple[float, float]]:
    raw = scene_metadata.get("map_bounds") or {"x": [-4.0, 4.0], "y": [-4.0, 4.0], "z": [0.0, 3.0]}
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def voxel_size_from_metadata(scene_metadata: dict[str, Any], grid: np.ndarray) -> float:
    if "voxel_size_recommended" in scene_metadata:
        return float(scene_metadata["voxel_size_recommended"])
    bounds = map_bounds_from_metadata(scene_metadata)
    return float((bounds["x"][1] - bounds["x"][0]) / grid.shape[0])


def state_colorbar(fig, ax):
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=STATE_NORM, cmap=STATE_CMAP),
        ax=ax,
        ticks=STATE_TICKS,
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(STATE_TICKLABELS)
    return cbar


def save_depth_images(depth_paths: list[Path], output_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    depth_summaries = []
    generated = []
    depths = []

    for idx, depth_path in enumerate(depth_paths):
        depth = np.load(depth_path)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        stats = depth_stats(depth)
        depths.append((depth, stats))
        out_path = output_dir / f"depth_{idx:03d}.png"

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        image = ax.imshow(depth, cmap="viridis")
        fig.colorbar(image, ax=ax, label="depth (m)")
        ax.set_title(f"Depth {idx:03d}: min {stats['min']:.3f} m, max {stats['max']:.3f} m")
        ax.set_xlabel("u pixel")
        ax.set_ylabel("v pixel")
        fig.savefig(out_path, dpi=160)
        plt.close(fig)

        stats.update({"index": idx, "input_file": str(depth_path), "image_file": str(out_path)})
        depth_summaries.append(stats)
        generated.append(str(out_path))

    fig, axes = plt.subplots(1, len(depths), figsize=(6 * len(depths), 5), constrained_layout=True)
    if len(depths) == 1:
        axes = [axes]
    vmin = min(stats["min"] for _, stats in depths if stats["min"] is not None)
    vmax = max(stats["max"] for _, stats in depths if stats["max"] is not None)
    last_image = None
    for idx, (ax, (depth, stats)) in enumerate(zip(axes, depths)):
        last_image = ax.imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"Depth {idx:03d}\nmin {stats['min']:.3f} m, max {stats['max']:.3f} m")
        ax.set_xlabel("u pixel")
        ax.set_ylabel("v pixel")
    fig.colorbar(last_image, ax=axes, label="depth (m)", shrink=0.78)
    grid_path = output_dir / "depth_grid.png"
    fig.savefig(grid_path, dpi=160)
    plt.close(fig)
    generated.append(str(grid_path))

    return depth_summaries, generated


def project_topdown(grid: np.ndarray) -> np.ndarray:
    occupied = np.any(grid == OCCUPIED, axis=2)
    free = np.any(grid == FREE, axis=2)
    topdown = np.full(grid.shape[:2], UNKNOWN, dtype=np.int8)
    topdown[free] = FREE
    topdown[occupied] = OCCUPIED
    return topdown


def draw_topdown(ax, topdown: np.ndarray, bounds: dict[str, tuple[float, float]], title: str) -> None:
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    ax.imshow(topdown.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)


def save_topdown_images(
    observed_paths: list[Path],
    grids: list[np.ndarray],
    bounds: dict[str, tuple[float, float]],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    step_summaries = []
    generated = []

    for idx, (observed_path, grid) in enumerate(zip(observed_paths, grids)):
        counts = count_states(grid)
        counts["observed_ratio"] = observed_ratio(counts)
        topdown = project_topdown(grid)
        out_path = output_dir / f"observed_topdown_step{idx}.png"
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        draw_topdown(
            ax,
            topdown,
            bounds,
            (
                f"Observed topdown step {idx}\n"
                f"unknown {counts['unknown_count']}, free {counts['free_count']}, occupied {counts['occupied_count']}"
            ),
        )
        state_colorbar(fig, ax)
        fig.savefig(out_path, dpi=170)
        plt.close(fig)

        step_summary = {
            "step": idx,
            "input_file": str(observed_path),
            "image_file": str(out_path),
            "shape": list(grid.shape),
            **counts,
        }
        step_summaries.append(step_summary)
        generated.append(str(out_path))

    fig, axes = plt.subplots(1, len(grids), figsize=(6 * len(grids), 5.5), constrained_layout=True)
    if len(grids) == 1:
        axes = [axes]
    for idx, (ax, grid) in enumerate(zip(axes, grids)):
        counts = step_summaries[idx]
        draw_topdown(
            ax,
            project_topdown(grid),
            bounds,
            f"Step {idx}\nfree {counts['free_count']} occ {counts['occupied_count']}",
        )
    state_colorbar(fig, axes[-1])
    compare_path = output_dir / "observed_topdown_compare.png"
    fig.savefig(compare_path, dpi=170)
    plt.close(fig)
    generated.append(str(compare_path))

    return step_summaries, generated


def voxel_centers(indices: np.ndarray, bounds: dict[str, tuple[float, float]], voxel_size: float) -> np.ndarray:
    mins = np.array([bounds["x"][0], bounds["y"][0], bounds["z"][0]], dtype=np.float64)
    return mins + (indices.astype(np.float64) + 0.5) * float(voxel_size)


def sample_indices(indices: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if len(indices) <= max_points:
        return indices
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(indices), size=max_points, replace=False)
    return indices[np.sort(choice)]


def set_3d_axes(ax, bounds: dict[str, tuple[float, float]]) -> None:
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_zlim(bounds["z"])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=24, azim=-52)


def save_3d_scatters(
    grid: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    output_dir: Path,
    max_points: int,
) -> tuple[list[str], dict[str, int]]:
    occupied_idx = np.argwhere(grid == OCCUPIED)
    free_idx = np.argwhere(grid == FREE)
    if occupied_idx.size == 0:
        raise ValueError("No OCCUPIED voxels found for 3D scatter.")
    if free_idx.size == 0:
        raise ValueError("No FREE voxels found for 3D scatter.")

    occupied_sample = sample_indices(occupied_idx, max_points=max_points, seed=7)
    free_sample = sample_indices(free_idx, max_points=max_points, seed=11)
    occupied_xyz = voxel_centers(occupied_sample, bounds, voxel_size)
    free_xyz = voxel_centers(free_sample, bounds, voxel_size)

    generated = []

    occupied_path = output_dir / "occupied_voxels_3d_step2.png"
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(occupied_xyz[:, 0], occupied_xyz[:, 1], occupied_xyz[:, 2], s=5, c="#bf616a", depthshade=False)
    set_3d_axes(ax, bounds)
    ax.set_title(f"Occupied voxels step 2: {len(occupied_idx)} total, {len(occupied_sample)} shown")
    fig.savefig(occupied_path, dpi=180)
    plt.close(fig)
    generated.append(str(occupied_path))

    combined_path = output_dir / "free_occupied_voxels_3d_step2.png"
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(free_xyz[:, 0], free_xyz[:, 1], free_xyz[:, 2], s=2, c="#88c0d0", alpha=0.18, depthshade=False, label="free")
    ax.scatter(
        occupied_xyz[:, 0],
        occupied_xyz[:, 1],
        occupied_xyz[:, 2],
        s=6,
        c="#bf616a",
        alpha=0.95,
        depthshade=False,
        label="occupied",
    )
    set_3d_axes(ax, bounds)
    ax.legend(loc="upper right")
    ax.set_title(
        f"Free + occupied step 2\nfree {len(free_idx)} ({len(free_sample)} shown), "
        f"occupied {len(occupied_idx)} ({len(occupied_sample)} shown)"
    )
    fig.savefig(combined_path, dpi=180)
    plt.close(fig)
    generated.append(str(combined_path))

    sampling_summary = {
        "free_total": int(len(free_idx)),
        "free_shown": int(len(free_sample)),
        "occupied_total": int(len(occupied_idx)),
        "occupied_shown": int(len(occupied_sample)),
    }
    return generated, sampling_summary


def save_slices(
    grid: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    output_dir: Path,
    z_meters: list[float],
) -> str:
    fig, axes = plt.subplots(1, len(z_meters), figsize=(5.2 * len(z_meters), 5), constrained_layout=True)
    if len(z_meters) == 1:
        axes = [axes]
    z_min = bounds["z"][0]
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]

    for ax, z_meter in zip(axes, z_meters):
        z_idx = int(round((z_meter - z_min) / voxel_size - 0.5))
        z_idx = int(np.clip(z_idx, 0, grid.shape[2] - 1))
        actual_z = z_min + (z_idx + 0.5) * voxel_size
        slice_xy = grid[:, :, z_idx]
        ax.imshow(slice_xy.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"z index {z_idx}\nz={actual_z:.2f} m")
    state_colorbar(fig, axes[-1])
    out_path = output_dir / "slices_step2.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return str(out_path)


def maybe_write_open3d_ply(
    grid: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    output_dir: Path,
    max_points: int,
) -> dict[str, Any]:
    try:
        import open3d as o3d  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return {"generated": False, "path": None, "reason": f"open3d unavailable: {exc!r}"}

    occupied_idx = np.argwhere(grid == OCCUPIED)
    free_idx = np.argwhere(grid == FREE)
    occupied_sample = sample_indices(occupied_idx, max_points=max_points, seed=17)
    free_sample = sample_indices(free_idx, max_points=max_points, seed=19)
    occupied_xyz = voxel_centers(occupied_sample, bounds, voxel_size)
    free_xyz = voxel_centers(free_sample, bounds, voxel_size)
    points = np.vstack([free_xyz, occupied_xyz])
    colors = np.vstack(
        [
            np.tile(np.array([[0.53, 0.75, 0.82]], dtype=np.float64), (len(free_xyz), 1)),
            np.tile(np.array([[0.75, 0.38, 0.42]], dtype=np.float64), (len(occupied_xyz), 1)),
        ]
    )

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    out_path = output_dir / "observed_step2_pointcloud.ply"
    ok = bool(o3d.io.write_point_cloud(str(out_path), pcd))
    if not ok:
        raise RuntimeError(f"open3d failed to write point cloud: {out_path}")
    return {"generated": True, "path": str(out_path), "point_count": int(len(points))}


def write_index_html(output_dir: Path, image_paths: list[str], ply_summary: dict[str, Any]) -> str:
    html_path = output_dir / "index.html"
    rows = []
    for image_path in image_paths:
        rel = Path(image_path).name
        rows.append(f"<section><h2>{rel}</h2><img src=\"{rel}\" alt=\"{rel}\" /></section>")
    ply_text = (
        f"<p>PLY: <a href=\"{Path(ply_summary['path']).name}\">{Path(ply_summary['path']).name}</a></p>"
        if ply_summary.get("generated")
        else f"<p>PLY skipped: {ply_summary.get('reason')}</p>"
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Stage 4A-1 Observed Map Visualization</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #20242a; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; }}
    section {{ margin-bottom: 32px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-1 Observed Map Visualization</h1>
  {ply_text}
  {''.join(rows)}
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return str(html_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Stage 4A-1 Isaac depth and observed voxel outputs.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke"),
        help="Stage 4A-1 output directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke_viz"),
        help="Visualization output directory.",
    )
    parser.add_argument("--max_points", type=int, default=5000, help="Maximum sampled free/occupied points per 3D plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_paths, observed_paths = ensure_inputs(input_dir)
    scene_metadata = load_json(input_dir / "scene_metadata.json")
    camera_info = load_json(input_dir / "camera_info.json")
    poses = [load_json(input_dir / f"pose_{idx:03d}.json") for idx in range(3)]

    grids = [np.load(path) for path in observed_paths]
    if len({grid.shape for grid in grids}) != 1:
        raise ValueError(f"Observed map shapes do not match: {[grid.shape for grid in grids]}")
    bounds = map_bounds_from_metadata(scene_metadata)
    voxel_size = voxel_size_from_metadata(scene_metadata, grids[-1])

    generated_images: list[str] = []
    depth_summaries, depth_images = save_depth_images(depth_paths, output_dir)
    generated_images.extend(depth_images)

    step_summaries, topdown_images = save_topdown_images(observed_paths, grids, bounds, output_dir)
    generated_images.extend(topdown_images)

    scatter_images, scatter_sampling = save_3d_scatters(
        grids[-1],
        bounds,
        voxel_size=voxel_size,
        output_dir=output_dir,
        max_points=int(args.max_points),
    )
    generated_images.extend(scatter_images)

    slices_path = save_slices(
        grids[-1],
        bounds,
        voxel_size=voxel_size,
        output_dir=output_dir,
        z_meters=[0.2, 0.8, 1.2, 1.8],
    )
    generated_images.append(slices_path)

    ply_summary = maybe_write_open3d_ply(
        grids[-1],
        bounds,
        voxel_size=voxel_size,
        output_dir=output_dir,
        max_points=int(args.max_points),
    )
    index_html = write_index_html(output_dir, generated_images, ply_summary)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "depth_files": [str(path) for path in depth_paths],
        "observed_state_files": [str(path) for path in observed_paths],
        "camera_info_file": str(input_dir / "camera_info.json"),
        "scene_metadata_file": str(input_dir / "scene_metadata.json"),
        "pose_files": [str(input_dir / f"pose_{idx:03d}.json") for idx in range(3)],
        "camera_info": camera_info,
        "poses": poses,
        "map_bounds": {axis: [bounds[axis][0], bounds[axis][1]] for axis in ("x", "y", "z")},
        "voxel_size": float(voxel_size),
        "observed_map_shape": list(grids[-1].shape),
        "depth_summaries": depth_summaries,
        "counts_by_step": step_summaries,
        "scatter_sampling": scatter_sampling,
        "generated_image_paths": generated_images,
        "index_html": index_html,
        "ply": ply_summary,
        "observed_state_modified": False,
        "prediction_used": False,
        "expert_used": False,
        "rl_or_training_used": False,
    }
    summary_path = output_dir / "viz_summary.json"
    save_json(summary_path, summary)

    print("Stage 4A-1 visualization complete")
    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"observed map shape: {tuple(grids[-1].shape)}")
    for step in step_summaries:
        print(
            f"step{step['step']} counts: unknown={step['unknown_count']} "
            f"free={step['free_count']} occupied={step['occupied_count']} "
            f"observed_ratio={step['observed_ratio']:.6f}"
        )
    print(f"generated images: {len(generated_images)}")
    print(f"summary json: {summary_path}")
    print(f"index html: {index_html}")
    if ply_summary.get("generated"):
        print(f"ply: {ply_summary['path']}")
    else:
        print(f"ply skipped: {ply_summary.get('reason')}")


if __name__ == "__main__":
    main()
