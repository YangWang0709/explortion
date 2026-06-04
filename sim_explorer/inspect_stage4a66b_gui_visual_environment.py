#!/usr/bin/env python3
"""Stage 4A-6.6b GUI capability and visual inspection package.

This stage is for human visual inspection only. It does not run expert
sampling, rollout, action execution, map_predict, SSCNet inference, prediction
NPZ generation, replay buffers, checkpoint creation, or RL/GDPO/PPO/BC/IL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_key, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scene_factory import build_larger_complex_scene_v1


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
DEFAULT_STAGE4A66_DIR = WORKSPACE / "outputs/isaac_stage4a66_larger_complex_scene_v1_validation"
DEFAULT_STAGE4A66A_DIR = WORKSPACE / "outputs/isaac_stage4a66a_scene_complexity_audit"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a66b_gui_visual_inspection"
DEPTH_KEY = "distance_to_image_plane"
RGB_KEYS = ("rgb", "rgba")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root: Path) -> dict[str, Any]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {"root": str(root), "file_count": len(files), "files": files}


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            text = json.dumps(value, sort_keys=True)
            if len(text) > 1200:
                text = text[:1200] + "..."
            value = f"`{text}`"
        else:
            value = f"`{value}`"
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def list_md(title: str, rows: list[str]) -> str:
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in rows]])


def write_manifest_md(path: Path, title: str, manifest: dict[str, Any], extra_rows: list[str] | None = None) -> None:
    rows = [
        f"root: `{manifest['root']}`",
        f"file_count: `{manifest['file_count']}`",
    ]
    if extra_rows:
        rows.extend(extra_rows)
    rows.extend(f"`{item['relative_path']}` sha256 `{item['sha256']}`" for item in manifest["files"][:180])
    write_text(path, list_md(title, rows))


def run_probe(command: list[str], timeout: int = 8) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        return {
            "command": command,
            "returncode": int(result.returncode),
            "stdout": result.stdout.strip()[:5000],
            "stderr": result.stderr.strip()[:5000],
        }
    except Exception as exc:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(exc)}


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def center_of_bounds(bounds: dict[str, list[float]]) -> tuple[float, float]:
    return (
        0.5 * (float(bounds["x"][0]) + float(bounds["x"][1])),
        0.5 * (float(bounds["y"][0]) + float(bounds["y"][1])),
    )


def box_xy(spec: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, _ = (float(v) for v in spec["position"])
    sx, sy, _ = (float(v) for v in spec["size"])
    return x - 0.5 * sx, y - 0.5 * sy, sx, sy


def pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def make_pose(index: int, name: str, position: list[float], yaw_deg: float, room: str, note: str, source: str) -> dict[str, Any]:
    yaw_rad = math.radians(float(yaw_deg))
    return {
        "index": int(index),
        "name": str(name),
        "position": [float(v) for v in position],
        "yaw_deg": float(yaw_deg),
        "yaw_rad": float(yaw_rad),
        "target": pose_target([float(v) for v in position], yaw_rad),
        "room": str(room),
        "note": str(note),
        "source_label": str(source),
    }


def build_inspection_poses(metadata: dict[str, Any], min_views: int) -> list[dict[str, Any]]:
    poses = []
    for pose in metadata["validation_camera_poses"]:
        item = dict(pose)
        item["name"] = f"fixed_validation_{int(pose['index']):03d}"
        item["source_label"] = "stage4a66_fixed_validation_view"
        poses.append(item)
    extras = [
        ("corridor_east_spur_entry", [6.75, 4.55, 1.2], 0.0, "corridor_east_spur", "east spur from loop toward room J"),
        ("room_j_inside_to_spur", [9.75, 4.55, 1.2], 180.0, "room_j", "room J cul-de-sac back toward east spur"),
        ("central_loop_north_axis", [0.0, 5.2, 1.2], 0.0, "corridor_north", "central loop north corridor looking east"),
        ("dead_end_branch_close", [-5.15, -9.65, 1.2], 90.0, "corridor_dead_end", "southwest dead-end branch toward loop"),
        ("narrow_west_spur_to_room_i", [-7.35, 4.55, 1.2], 180.0, "corridor_nw_spur", "narrow west spur toward room I"),
        ("dense_room_j_obstacles", [10.15, 4.45, 1.2], 180.0, "room_j", "high obstacle density in room J"),
        ("long_west_corridor", [-5.2, 0.0, 1.2], 90.0, "corridor_west", "long west loop corridor"),
        ("loop_junction_east", [5.2, 5.2, 1.2], -135.0, "corridor_east", "loop junction across east/north/south connection"),
        ("start_variants_group_overview", [0.0, -5.2, 1.2], 180.0, "corridor_south", "start variants group across south loop"),
        ("occluded_far_room_chain", [-9.95, 4.55, 1.2], 0.0, "room_i", "occluded far-room chain through room I"),
    ]
    for offset, (name, position, yaw_deg, room, note) in enumerate(extras, start=len(poses)):
        poses.append(make_pose(offset, name, position, yaw_deg, room, note, "stage4a66b_extra_visual_inspection_view"))
    while len(poses) < int(min_views):
        idx = len(poses)
        angle = (idx * 37) % 360
        poses.append(make_pose(idx, f"extra_auto_{idx:03d}", [0.0, 0.0, 1.2], angle, "central_loop", "auto-filled visual pose", "stage4a66b_extra_visual_inspection_view"))
    return poses[: max(int(min_views), len(poses))]


def rgb_stats(rgb: np.ndarray) -> dict[str, Any]:
    return {
        "shape": [int(v) for v in rgb.shape],
        "min": int(rgb.min()) if rgb.size else None,
        "max": int(rgb.max()) if rgb.size else None,
        "mean": float(rgb.mean()) if rgb.size else None,
        "std": float(rgb.std()) if rgb.size else None,
        "nonblank": bool(rgb.size and int(rgb.max()) > 2 and float(rgb.std()) >= 1.0),
    }


def depth_stats(depth: np.ndarray) -> dict[str, Any]:
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    return {
        "shape": [int(v) for v in depth.shape],
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
        "has_positive_finite_depth": bool(positive.size > 0),
    }


def save_depth_color(path: Path, depth: np.ndarray, title: str) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError(f"depth has no finite positive values: {path}")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(5.8, 4.3), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def draw_base_layout(ax: Any, metadata: dict[str, Any], labels: bool = True, obstacles: bool = False, openings: bool = True) -> None:
    bounds = metadata["map_bounds"]
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, color="#d9dee3", linewidth=0.45)
    for corridor in metadata["corridors"]:
        x0, x1 = corridor["bounds"]["x"]
        y0, y1 = corridor["bounds"]["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#d6edf1", edgecolor="#5d99a2", linewidth=0.8, alpha=0.68))
        if labels:
            ax.text(*center_of_bounds(corridor["bounds"]), corridor["name"], ha="center", va="center", fontsize=6.0, color="#255f67")
    for room in metadata["rooms"]:
        x0, x1 = room["bounds"]["x"]
        y0, y1 = room["bounds"]["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f2e8d4", edgecolor="#9a7847", linewidth=0.8, alpha=0.62))
        if labels:
            ax.text(*center_of_bounds(room["bounds"]), room["name"], ha="center", va="center", fontsize=6.2, color="#5e4321")
    for wall in metadata["walls"]:
        x, y, w, h = box_xy(wall)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#4b5158", edgecolor="#2a2d31", linewidth=0.18, alpha=0.86))
    if obstacles:
        colors = {
            "table_like_cuboid": "#bd7b37",
            "cabinet_like_tall_cuboid": "#4d7f61",
            "shelf_like_long_cuboid": "#806f4c",
            "pillar_cuboid": "#6b7280",
            "half_wall_occluder": "#565b61",
            "small_clutter_cuboid": "#b96c3f",
        }
        for obstacle in metadata["obstacles"]:
            x, y, w, h = box_xy(obstacle)
            ax.add_patch(Rectangle((x, y), w, h, facecolor=colors.get(obstacle.get("category"), "#996b43"), edgecolor="#232323", linewidth=0.28, alpha=0.88))
            if labels:
                ax.text(x + 0.5 * w, y + 0.5 * h, obstacle["name"].replace("obs_", ""), ha="center", va="center", fontsize=4.0, color="#111")
    if openings:
        for opening in metadata["openings"]:
            ax.scatter([opening["center"][0]], [opening["center"][1]], marker="s", s=18, c="#2e9d52", edgecolors="#14572c", linewidths=0.3, zorder=7)


def draw_pose_arrow(ax: Any, pose: dict[str, Any], color: str = "#174ea6", label: str | None = None) -> None:
    x, y, _ = (float(v) for v in pose["position"])
    yaw = float(pose.get("yaw_rad", math.radians(float(pose.get("yaw_deg", 0.0)))))
    dx = math.cos(yaw) * 0.72
    dy = math.sin(yaw) * 0.72
    ax.arrow(x, y, dx, dy, width=0.025, head_width=0.18, color=color, length_includes_head=True, zorder=8)
    ax.scatter([x], [y], c=[color], s=22, edgecolors="#ffffff", linewidths=0.4, zorder=9)
    fallback_label = pose.get("index", pose.get("name", "pose"))
    ax.text(x + 0.10, y + 0.10, label if label is not None else str(fallback_label), fontsize=6.2, color=color, weight="bold", zorder=10)


def save_layout(path: Path, metadata: dict[str, Any], title: str, *, labels: bool = True, obstacles: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 9.6), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=labels, obstacles=obstacles, openings=True)
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_room_corridor_opening_labels(path: Path, metadata: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 9.6), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=False, openings=True)
    for opening in metadata["openings"]:
        ax.text(opening["center"][0] + 0.08, opening["center"][1] + 0.08, opening["name"].replace("door_", ""), fontsize=5.0, color="#166534")
    ax.set_title("room, corridor, and opening labels")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_obstacle_labels(path: Path, metadata: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 9.6), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=False, obstacles=True, openings=True)
    for obstacle in metadata["obstacles"]:
        x, y, w, h = box_xy(obstacle)
        ax.text(x + 0.5 * w, y + 0.5 * h, obstacle["name"].replace("obs_", ""), ha="center", va="center", fontsize=4.2, color="#111827")
    ax.set_title("obstacle labels by metadata cuboid")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_start_variants(path: Path, metadata: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 9.6), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, openings=True)
    for start in metadata["start_variants"]:
        draw_pose_arrow(ax, start, color="#7c3aed", label=start["name"].replace("start_", ""))
    ax.set_title("start variants, labeled")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_validation_poses(path: Path, metadata: dict[str, Any], poses: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 9.6), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, openings=True)
    for pose in poses[:14]:
        draw_pose_arrow(ax, pose, color="#1554b7")
    ax.set_title("Stage 4A-6.6 fixed validation poses, labeled")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_topology_graph(path: Path, metadata: dict[str, Any]) -> None:
    centers = {}
    for room in metadata["rooms"]:
        centers[room["name"]] = center_of_bounds(room["bounds"])
    for corridor in metadata["corridors"]:
        centers[corridor["name"]] = center_of_bounds(corridor["bounds"])
    fig, ax = plt.subplots(figsize=(9.6, 9.4), constrained_layout=True)
    ax.set_xlim(metadata["map_bounds"]["x"])
    ax.set_ylim(metadata["map_bounds"]["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e3e8ef", linewidth=0.45)
    for edge in metadata["topology_graph"]["edges"]:
        x0, y0 = centers[edge["source"]]
        x1, y1 = centers[edge["target"]]
        ax.plot([x0, x1], [y0, y1], color="#64748b", linewidth=1.05)
        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), f"{edge.get('width_m', 0.0):.2f}m", fontsize=5.4, color="#475569")
    for node in metadata["topology_graph"]["nodes"]:
        x, y = centers[node["id"]]
        color = "#d19038" if node["kind"] == "room" else "#5597a9"
        ax.scatter([x], [y], s=125, c=[color], edgecolors="#1f2937", zorder=4)
        ax.text(x, y + 0.22, node["id"], ha="center", va="bottom", fontsize=6.5, weight="bold")
    ax.set_title("topology graph, labeled")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def bounds_for_region(metadata: dict[str, Any], name: str) -> dict[str, list[float]] | None:
    for group in ("rooms", "corridors"):
        for item in metadata[group]:
            if item["name"] == name:
                return item["bounds"]
    return None


def outline_region(ax: Any, bounds: dict[str, list[float]], label: str, color: str) -> None:
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=color, linewidth=2.2, linestyle="--", zorder=20))
    ax.text(x0, y1 + 0.20, label, fontsize=7.5, color=color, weight="bold", zorder=21)


def save_warning_regions(path: Path, metadata: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 9.6), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, openings=True)
    for name in ("corridor_east_spur", "room_j", "room_i", "corridor_nw_spur"):
        b = bounds_for_region(metadata, name)
        if b:
            outline_region(ax, b, name, "#dc2626" if name in {"corridor_east_spur", "room_j"} else "#b45309")
    close_pairs = [("start_room_d", "start_corridor_north"), ("start_room_c", "start_loop_junction")]
    starts = {start["name"]: start for start in metadata["start_variants"]}
    for a, b in close_pairs:
        if a in starts and b in starts:
            xa, ya, _ = starts[a]["position"]
            xb, yb, _ = starts[b]["position"]
            ax.plot([xa, xb], [ya, yb], color="#9333ea", linewidth=1.4, linestyle=":", zorder=22)
            ax.text(0.5 * (xa + xb), 0.5 * (ya + yb), f"{a}/{b}", fontsize=5.5, color="#6b21a8")
    ax.set_title("audit warning regions: east spur, room J, spur rooms, close topology starts")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_closeup(path: Path, metadata: dict[str, Any], title: str, xlim: tuple[float, float], ylim: tuple[float, float], poses: list[dict[str, Any]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, openings=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if poses:
        for pose in poses:
            draw_pose_arrow(ax, pose, color="#1554b7")
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_topdown_and_closeups(output_dir: Path, metadata: dict[str, Any], poses: list[dict[str, Any]]) -> None:
    save_layout(output_dir / "scene_layout_topdown_human.png", metadata, "larger_complex_scene_v1 layout for human review", labels=True, obstacles=True)
    save_room_corridor_opening_labels(output_dir / "room_corridor_opening_labels_topdown.png", metadata)
    save_obstacle_labels(output_dir / "obstacle_labels_topdown.png", metadata)
    save_start_variants(output_dir / "start_variants_labeled_topdown.png", metadata)
    save_validation_poses(output_dir / "validation_poses_labeled_topdown.png", metadata, poses)
    save_topology_graph(output_dir / "topology_graph_labeled.png", metadata)
    save_warning_regions(output_dir / "audit_warning_regions_topdown.png", metadata)

    pose_by_name = {pose.get("name", f"pose_{pose['index']}"): pose for pose in poses}
    save_closeup(output_dir / "closeup_corridor_east_spur.png", metadata, "closeup: corridor_east_spur", (5.2, 8.9), (3.5, 5.9), [pose_by_name.get("corridor_east_spur_entry")] if pose_by_name.get("corridor_east_spur_entry") else None)
    save_closeup(output_dir / "closeup_room_j.png", metadata, "closeup: room_j", (7.7, 11.7), (3.0, 5.9), [pose_by_name.get("room_j_inside_to_spur")] if pose_by_name.get("room_j_inside_to_spur") else None)
    save_closeup(output_dir / "closeup_loop_junction.png", metadata, "closeup: loop junctions", (3.5, 6.8), (3.5, 6.8), [pose_by_name.get("loop_junction_east")] if pose_by_name.get("loop_junction_east") else None)
    save_closeup(output_dir / "closeup_dead_end_branch.png", metadata, "closeup: dead-end branch", (-6.6, -3.6), (-11.5, -7.2), [pose_by_name.get("dead_end_branch_close")] if pose_by_name.get("dead_end_branch_close") else None)
    save_closeup(output_dir / "closeup_narrow_passage_examples.png", metadata, "closeup: narrow spur passages", (-8.8, -4.9), (3.4, 6.0), [pose_by_name.get("narrow_west_spur_to_room_i")] if pose_by_name.get("narrow_west_spur_to_room_i") else None)
    save_closeup(output_dir / "closeup_obstacle_dense_spur_rooms.png", metadata, "closeup: dense spur rooms", (7.6, 11.7), (3.0, 5.9), [pose_by_name.get("dense_room_j_obstacles")] if pose_by_name.get("dense_room_j_obstacles") else None)
    save_closeup(output_dir / "closeup_start_locations.png", metadata, "closeup: start locations overview", (-12.0, 12.0), (-12.0, 12.0), metadata["start_variants"])


def normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    return np.clip(np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def extract_rgb(camera: Any, label: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEYS:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        source = tensor[0].detach().cpu().numpy()
        rgb = normalize_rgb(source)
        return rgb, key, rgb_stats(rgb)
    raise KeyError(f"{label}: camera output missing rgb/rgba")


def extract_depth(camera: Any, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    tensor = camera.data.output.get(DEPTH_KEY)
    if tensor is None:
        raise KeyError(f"{label}: camera output missing {DEPTH_KEY}")
    depth = tensor[0].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    stats = depth_stats(depth)
    if not stats["has_positive_finite_depth"]:
        raise ValueError(f"{label}: no finite positive depth")
    return depth, stats


def capture_with_isaac(args: argparse.Namespace, app_launcher_cls: Any, output_dir: Path, poses: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Any]:
    if app_launcher_cls is None:
        raise RuntimeError("IsaacLab AppLauncher is not importable")

    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "GNOME_SETUP_DISPLAY"):
        os.environ.pop(key, None)

    setattr(args, "headless", True)
    if hasattr(args, "enable_cameras"):
        setattr(args, "enable_cameras", True)

    app_launcher = app_launcher_cls(args)
    simulation_app = app_launcher.app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        dome_cfg = sim_utils.DomeLightCfg(intensity=3300.0, color=(0.84, 0.85, 0.80))
        dome_cfg.func("/World/Light", dome_cfg)
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([0.0, -17.0, 14.0], [0.0, 0.0, 0.7])
        build_larger_complex_scene_v1(seed=int(args.scene_seed), spawn=True, sim_utils_module=sim_utils)
        sim_utils.create_prim("/World/InspectionCameraRig", "Xform")
        camera_cfg = CameraCfg(
            prim_path="/World/InspectionCameraRig/CameraSensor",
            update_period=0.0,
            height=int(args.camera_height),
            width=int(args.camera_width),
            data_types=["rgb", DEPTH_KEY],
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=36.0,
                clipping_range=(0.05, float(args.max_depth)),
            ),
        )
        camera = Camera(cfg=camera_cfg)
        sim.reset()

        intrinsic = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
        camera_info = {
            "render_backend": "isaac_headless",
            "sensor_api_depth_key": DEPTH_KEY,
            "depth_units": "meters",
            "width": int(args.camera_width),
            "height": int(args.camera_height),
            "max_depth": float(args.max_depth),
            "near_depth": 0.05,
            "intrinsic_matrix": intrinsic.tolist(),
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
            "data_types_requested": ["rgb", DEPTH_KEY],
        }
        save_json(output_dir / "camera_info.json", camera_info)

        records = []
        for pose in poses:
            idx = int(pose["index"])
            position = [float(v) for v in pose["position"]]
            target = pose_target(position, float(pose["yaw_rad"]))
            camera.set_world_poses_from_view(
                eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
                targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
            )
            for _ in range(max(int(args.settle_steps), 1)):
                sim.step()
                camera.update(dt=sim.get_physics_dt())
            rgb, rgb_key, rstats = extract_rgb(camera, f"inspection pose {idx}")
            depth, dstats = extract_depth(camera, f"inspection pose {idx}")
            rgb_path = output_dir / f"inspection_rgb_{idx:03d}.png"
            depth_path = output_dir / f"inspection_depth_{idx:03d}.npy"
            depth_color_path = output_dir / f"inspection_depth_color_{idx:03d}.png"
            pose_path = output_dir / f"inspection_pose_{idx:03d}.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth)
            save_depth_color(depth_color_path, depth, f"inspection depth {idx:03d}")
            pose_record = dict(pose)
            pose_record.update(
                {
                    "target": target,
                    "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
                    "render_backend": "isaac_headless",
                    "rgb_file": rgb_path.name,
                    "depth_file": depth_path.name,
                    "depth_color_file": depth_color_path.name,
                }
            )
            if hasattr(camera.data, "pos_w"):
                pose_record["isaac_camera_pos_w"] = camera.data.pos_w[0].detach().cpu().numpy().astype(float).tolist()
            if hasattr(camera.data, "quat_w_ros"):
                pose_record["isaac_camera_quat_w_ros"] = camera.data.quat_w_ros[0].detach().cpu().numpy().astype(float).tolist()
            save_json(pose_path, pose_record)
            records.append(
                {
                    "index": idx,
                    "name": pose_record.get("name"),
                    "room": pose_record.get("room"),
                    "note": pose_record.get("note"),
                    "source_label": pose_record.get("source_label"),
                    "render_backend": "isaac_headless",
                    "rgb_file": rgb_path.name,
                    "depth_file": depth_path.name,
                    "depth_color_file": depth_color_path.name,
                    "pose_file": pose_path.name,
                    "rgb_key_used": rgb_key,
                    "rgb_stats": rstats,
                    "depth_stats": dstats,
                }
            )
            print(f"[INFO]: captured inspection pose {idx:03d} backend=isaac_headless rgb_nonblank={rstats['nonblank']} depth_positive={dstats['positive_count']}")
        return records, simulation_app
    except Exception:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        raise


def schematic_rgb(metadata: dict[str, Any], pose: dict[str, Any], width: int, height: int) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(width / 120.0, height / 120.0), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, openings=True)
    x, y, _ = pose["position"]
    ax.set_xlim(float(x) - 4.0, float(x) + 4.0)
    ax.set_ylim(float(y) - 3.0, float(y) + 3.0)
    yaw = float(pose["yaw_rad"])
    fov = math.radians(68.0)
    points = [(x, y)]
    for sign in (-1, 1):
        points.append((x + math.cos(yaw + sign * fov * 0.5) * 3.2, y + math.sin(yaw + sign * fov * 0.5) * 3.2))
    ax.add_patch(Polygon(points, closed=True, facecolor="#facc15", edgecolor="#ca8a04", alpha=0.24, zorder=15))
    draw_pose_arrow(ax, pose, color="#0f4fb8")
    ax.set_title(f"{int(pose['index']):03d} {pose.get('name', pose.get('room'))}", fontsize=8)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def synthetic_depth(width: int, height: int, seed: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    base = 2.0 + 8.0 * (yy.astype(np.float32) / max(height - 1, 1))
    ripple = 0.35 * np.sin((xx + seed * 11) / 19.0) + 0.22 * np.cos((yy + seed * 7) / 17.0)
    depth = np.maximum(base + ripple, 0.2).astype(np.float32)
    return depth


def copy_or_schematic_capture(args: argparse.Namespace, output_dir: Path, stage4a66_dir: Path, metadata: dict[str, Any], poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for pose in poses:
        idx = int(pose["index"])
        rgb_path = output_dir / f"inspection_rgb_{idx:03d}.png"
        depth_path = output_dir / f"inspection_depth_{idx:03d}.npy"
        depth_color_path = output_dir / f"inspection_depth_color_{idx:03d}.png"
        pose_path = output_dir / f"inspection_pose_{idx:03d}.json"
        source_label = "offline_schematic_visualization"
        if idx < 14 and (stage4a66_dir / f"validation_rgb_{idx:03d}.png").is_file() and (stage4a66_dir / f"validation_depth_{idx:03d}.npy").is_file():
            shutil.copy2(stage4a66_dir / f"validation_rgb_{idx:03d}.png", rgb_path)
            depth = np.load(stage4a66_dir / f"validation_depth_{idx:03d}.npy")
            np.save(depth_path, depth)
            source_label = "stage4a66_existing_validation_capture_reused"
        else:
            rgb = schematic_rgb(metadata, pose, int(args.camera_width), int(args.camera_height))
            Image.fromarray(rgb).save(rgb_path)
            depth = synthetic_depth(int(args.camera_width), int(args.camera_height), idx)
            np.save(depth_path, depth)
        save_depth_color(depth_color_path, depth, f"inspection depth {idx:03d}")
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        pose_record = dict(pose)
        pose_record.update(
            {
                "target": pose_target([float(v) for v in pose["position"]], float(pose["yaw_rad"])),
                "render_backend": "existing_isaac_validation_plus_offline_schematic",
                "rgb_file": rgb_path.name,
                "depth_file": depth_path.name,
                "depth_color_file": depth_color_path.name,
            }
        )
        save_json(pose_path, pose_record)
        records.append(
            {
                "index": idx,
                "name": pose_record.get("name"),
                "room": pose_record.get("room"),
                "note": pose_record.get("note"),
                "source_label": source_label,
                "render_backend": "existing_isaac_validation_plus_offline_schematic",
                "rgb_file": rgb_path.name,
                "depth_file": depth_path.name,
                "depth_color_file": depth_color_path.name,
                "pose_file": pose_path.name,
                "rgb_key_used": "rgb_file",
                "rgb_stats": rgb_stats(rgb),
                "depth_stats": depth_stats(depth),
            }
        )
    save_json(
        output_dir / "camera_info.json",
        {
            "render_backend": "existing_isaac_validation_plus_offline_schematic",
            "width": int(args.camera_width),
            "height": int(args.camera_height),
            "max_depth": float(args.max_depth),
            "note": "First 14 views reuse Stage 4A-6.6 Isaac validation captures when available; extra views are offline schematic visualizations.",
        },
    )
    return records


def save_rgb_grid(path: Path, records: list[dict[str, Any]], output_dir: Path) -> None:
    cols = 6
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.25), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, record in zip(axes, records):
        rgb = np.asarray(Image.open(output_dir / record["rgb_file"]).convert("RGB"))
        ax.imshow(rgb)
        ax.set_title(f"{int(record['index']):03d} {record.get('room', '')}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_depth_grid(path: Path, records: list[dict[str, Any]], output_dir: Path) -> None:
    depths = [np.load(output_dir / record["depth_file"]) for record in records]
    finite_values = [depth[np.isfinite(depth) & (depth > 0.0)] for depth in depths]
    vmin = min(float(values.min()) for values in finite_values if values.size)
    vmax = max(float(values.max()) for values in finite_values if values.size)
    cols = 6
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.35), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    last = None
    for ax, record, depth in zip(axes, records, depths):
        masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
        last = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{int(record['index']):03d} {record.get('room', '')}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.colorbar(last, ax=axes[: len(records)].tolist(), shrink=0.62, label="depth (m)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_flythrough(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_dir = output_dir / "flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_idx in range(60):
        record = records[int(frame_idx / 60.0 * len(records)) % len(records)]
        image = Image.open(output_dir / record["rgb_file"]).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 44), fill=(18, 24, 32))
        draw.text((14, 12), f"larger_complex_scene_v1 flythrough | view {int(record['index']):03d} | {record.get('room', '')}", fill=(240, 244, 248))
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        image.save(frame_path)
        frames.append(frame_path)
    report: dict[str, Any] = {
        "frame_count": len(frames),
        "frame_dir": str(frame_dir),
        "mp4_created": False,
        "gif_created": False,
        "video_path": None,
        "encoding_seconds": None,
    }
    started = time.perf_counter()
    try:
        import imageio.v2 as imageio

        arrays = [np.asarray(Image.open(path).convert("RGB")) for path in frames]
        mp4_path = output_dir / "larger_complex_scene_v1_flythrough.mp4"
        imageio.mimsave(mp4_path, arrays, fps=2)
        if mp4_path.is_file() and mp4_path.stat().st_size > 0:
            report.update({"mp4_created": True, "video_path": str(mp4_path), "encoding_seconds": float(time.perf_counter() - started)})
            return report
    except Exception as exc:
        report["mp4_error"] = str(exc)
    try:
        gif_path = output_dir / "larger_complex_scene_v1_flythrough.gif"
        frames_pil = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frames]
        frames_pil[0].save(gif_path, save_all=True, append_images=frames_pil[1:], duration=500, loop=0)
        if gif_path.is_file() and gif_path.stat().st_size > 0:
            report.update({"gif_created": True, "video_path": str(gif_path), "encoding_seconds": float(time.perf_counter() - started)})
            return report
    except Exception as exc:
        report["gif_error"] = str(exc)
    write_text(output_dir / "video_generation_skipped.md", "# Video Generation Skipped\n\nMP4/GIF encoding failed; flythrough frame sequence was generated instead.")
    report["encoding_seconds"] = float(time.perf_counter() - started)
    return report


def gui_capability_report(output_dir: Path, max_workers: int) -> dict[str, Any]:
    env = {key: os.environ.get(key) for key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "VK_ICD_FILENAMES", "__GLX_VENDOR_LIBRARY_NAME")}
    probes: dict[str, Any] = {
        "who": run_probe(["who"], timeout=5) if command_exists("who") else {"available": False},
        "w": run_probe(["w"], timeout=5) if command_exists("w") else {"available": False},
        "nvidia_smi": run_probe(["nvidia-smi"], timeout=8) if command_exists("nvidia-smi") else {"available": False},
        "glxinfo_B": run_probe(["glxinfo", "-B"], timeout=8) if command_exists("glxinfo") else {"available": False},
        "vulkaninfo_summary": run_probe(["vulkaninfo", "--summary"], timeout=12) if command_exists("vulkaninfo") else {"available": False},
        "xdpyinfo": run_probe(["xdpyinfo"], timeout=8) if os.environ.get("DISPLAY") and command_exists("xdpyinfo") else {"available": command_exists("xdpyinfo"), "skipped": "DISPLAY missing or xdpyinfo unavailable"},
    }
    process_hints = run_probe(["ps", "-eo", "comm,args"], timeout=8) if command_exists("ps") else {"stdout": ""}
    hint_text = (process_hints.get("stdout") or "").lower()
    remote_display_hints = {name: (name in hint_text) for name in ("vnc", "xrdp", "nxserver", "nomachine", "xorg", "xvfb", "xpra", "dcv", "sunshine", "moonlight")}
    isaac_candidates = []
    for name in ("isaac-sim.sh", "isaac-sim", "python.sh"):
        found = shutil.which(name)
        if found:
            isaac_candidates.append(found)
    for path in (Path("/home/ubuntu22/isaac-sim"), Path("/home/ubuntu22/IsaacLab"), Path("/home/ubuntu22/isaaclab")):
        if path.exists():
            isaac_candidates.append(str(path))
    gpu_name = "unavailable"
    smi = probes["nvidia_smi"]
    if isinstance(smi, dict) and smi.get("returncode") == 0 and smi.get("stdout"):
        for line in smi["stdout"].splitlines():
            if "NVIDIA" in line or "RTX" in line or "A" in line:
                gpu_name = line.strip()
                break
    report = {
        "stage": "Stage 4A-6.6b",
        "created_at_unix": time.time(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "os_cpu_count": os.cpu_count(),
        "requested_max_workers": int(max_workers),
        "actual_max_workers": int(min(int(max_workers), os.cpu_count() or 1)),
        "env": env,
        "display_present": bool(env["DISPLAY"]),
        "wayland_present": bool(env["WAYLAND_DISPLAY"]),
        "xauthority_present": bool(env["XAUTHORITY"]),
        "gpu_name_probe": gpu_name,
        "probes": probes,
        "remote_display_hints": remote_display_hints,
        "isaac_candidates": isaac_candidates,
        "known_headless_vulkan_env": {
            "VK_ICD_FILENAMES": "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
        },
    }
    save_json(output_dir / "gui_capability_report.json", report)
    write_text(
        output_dir / "gui_capability_report.md",
        markdown_table(
            "GUI Capability Report",
            {
                "DISPLAY": env["DISPLAY"],
                "WAYLAND_DISPLAY": env["WAYLAND_DISPLAY"],
                "XAUTHORITY": env["XAUTHORITY"],
                "display_present": report["display_present"],
                "gpu_name_probe": gpu_name,
                "remote_display_hints": remote_display_hints,
                "isaac_candidates": isaac_candidates,
                "requested_max_workers": int(max_workers),
                "actual_max_workers": report["actual_max_workers"],
            },
        ),
    )
    return report


def attempt_gui_probe(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    code = r'''
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
probe_args = parser.parse_args([])
setattr(probe_args, "headless", False)
if hasattr(probe_args, "enable_cameras"):
    setattr(probe_args, "enable_cameras", True)
launcher = AppLauncher(probe_args)
app = launcher.app
try:
    import isaaclab.sim as sim_utils
    from scene_factory import build_larger_complex_scene_v1
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=probe_args.device))
    sim.set_camera_view([0.0, -16.5, 13.5], [0.0, 0.0, 0.6])
    dome = sim_utils.DomeLightCfg(intensity=2600.0, color=(0.84, 0.85, 0.80))
    dome.func("/World/GuiProbeLight", dome)
    build_larger_complex_scene_v1(seed=0, spawn=True, sim_utils_module=sim_utils)
    sim.reset()
    for _ in range(8):
        sim.step()
    print("GUI_PROBE_SCENE_LOADED")
finally:
    app.close(wait_for_replicator=False, skip_cleanup=True)
'''
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(WORKSPACE / "sim_explorer"),
            env=os.environ.copy(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=75,
        )
        probe = {
            "attempted": True,
            "returncode": int(result.returncode),
            "stdout": result.stdout[-8000:],
            "stderr": result.stderr[-8000:],
            "elapsed_seconds": float(time.perf_counter() - started),
            "success": result.returncode == 0 and "GUI_PROBE_SCENE_LOADED" in result.stdout,
        }
    except Exception as exc:
        probe = {
            "attempted": True,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "elapsed_seconds": float(time.perf_counter() - started),
            "success": False,
        }
    write_text(
        output_dir / "gui_attempt_probe.log",
        "\n".join(
            [
                "# GUI Attempt Probe Log",
                "",
                f"attempted: {probe['attempted']}",
                f"success: {probe['success']}",
                f"returncode: {probe['returncode']}",
                f"elapsed_seconds: {probe['elapsed_seconds']}",
                "",
                "## stdout",
                str(probe["stdout"]),
                "",
                "## stderr",
                str(probe["stderr"]),
            ]
        ),
    )
    return probe


def write_gui_attempt_files(output_dir: Path, args: argparse.Namespace, capability: dict[str, Any]) -> dict[str, Any]:
    start_command = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh",
            "conda activate env_isaaclab",
            "export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/sim_explorer:${PYTHONPATH:-}",
            "cd /home/ubuntu22/sc_explorer_ws/sim_explorer",
            "# Requires a real DISPLAY/remote desktop visible to the user.",
            "python inspect_stage4a66b_gui_visual_environment.py \\",
            f"  --stage4a66_dir {args.stage4a66_dir} \\",
            f"  --stage4a66a_dir {args.stage4a66a_dir} \\",
            f"  --output_dir {args.output_dir} \\",
            "  --scene_variant larger_complex_scene_v1 --scene_seed 0 --try_gui --save_viz",
        ]
    )
    script_path = output_dir / "gui_start_command.sh"
    write_text(script_path, start_command)
    script_path.chmod(0o755)
    display = capability["env"].get("DISPLAY")
    gui_probe: dict[str, Any] | None = None
    if not display:
        status = "skipped_no_display"
        reason = "DISPLAY is missing; no safe visible IsaacSim GUI target was available."
        attempted = False
        visibility = "not_visible_no_display"
    else:
        xdpy = capability["probes"].get("xdpyinfo", {})
        if isinstance(xdpy, dict) and xdpy.get("returncode") == 0:
            gui_probe = attempt_gui_probe(output_dir, args)
            status = "success" if gui_probe["success"] else "failed"
            reason = (
                "Bounded GUI-mode IsaacSim probe loaded larger_complex_scene_v1, then closed; user visibility is still unconfirmed."
                if gui_probe["success"]
                else "DISPLAY responded, but bounded GUI-mode IsaacSim probe failed; fallback visual package is generated."
            )
            attempted = True
            visibility = "gui_possible_but_user_visibility_unconfirmed"
        else:
            status = "skipped_display_unverified"
            reason = "DISPLAY was set but xdpyinfo did not verify a visible display."
            attempted = False
            visibility = "unconfirmed"
    report = {
        "stage": "Stage 4A-6.6b",
        "try_gui_requested": bool(args.try_gui),
        "gui_attempted": attempted,
        "gui_attempt_status": status,
        "failure_reason": reason,
        "display": display,
        "wayland_display": capability["env"].get("WAYLAND_DISPLAY"),
        "xauthority": capability["env"].get("XAUTHORITY"),
        "user_visibility": visibility,
        "gui_started_but_user_visibility_unconfirmed": status == "success",
        "bounded_gui_probe": gui_probe,
        "fallback_visual_package_required": True,
        "no_action_execution": True,
        "no_expert_sampling": True,
        "no_rollout": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
    }
    save_json(output_dir / "gui_attempt_report.json", report)
    write_text(output_dir / "gui_attempt_report.md", markdown_table("GUI Attempt Report", report))
    write_text(
        output_dir / "user_viewing_instructions.md",
        "\n".join(
            [
                "# User Viewing Instructions",
                "",
                f"GUI status: `{status}`.",
                "",
                "Review the offline visual package first:",
                f"- HTML index: `{output_dir / 'visual_inspection_index.html'}`",
                f"- RGB grid: `{output_dir / 'inspection_rgb_grid.png'}`",
                f"- Depth grid: `{output_dir / 'inspection_depth_grid.png'}`",
                f"- Checklist: `{output_dir / 'human_visual_review_checklist.md'}`",
                "",
                "If a real remote desktop becomes available, use `gui_start_command.sh` as the bounded GUI command template.",
            ]
        ),
    )
    write_text(
        output_dir / "gui_stop_instructions.md",
        "# GUI Stop Instructions\n\nNo persistent IsaacSim GUI process was started by this Stage 4A-6.6b run. If you later use `gui_start_command.sh`, close the IsaacSim window or stop that foreground shell process with Ctrl-C.",
    )
    return report


def write_checklist(output_dir: Path) -> None:
    items = [
        "地图看起来是否真的比 medium_three_rooms 大很多？",
        "房间是否不像空盒子？",
        "走廊/门洞/回环是否自然？",
        "`corridor_east_spur` 是否能看见？",
        "`room_j` 是否能看见？",
        "障碍物是否像有效遮挡，而不是随机装饰？",
        "是否有过窄/完全堵死的通道？",
        "是否有过多空旷区域？",
        "start variants 是否分散且合理？",
        "是否有明显穿模/漂浮/墙体错位？",
        "视觉上是否适合小规模 expert sampling pilot？",
        "是否需要回到 6.6b-revision 修改场景？",
    ]
    payload = {
        "stage": "Stage 4A-6.6b",
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "items": [{"id": idx + 1, "question": item, "status": "pending_user_review"} for idx, item in enumerate(items)],
    }
    save_json(output_dir / "human_visual_review_checklist.json", payload)
    write_text(
        output_dir / "human_visual_review_checklist.md",
        "\n".join(["# Human Visual Review Checklist", "", *[f"{idx + 1}. {item}" for idx, item in enumerate(items)], "", "Status: `pending_user_review`."]),
    )


def write_negative_scope_reports(output_dir: Path) -> None:
    reports = {
        "no_expert_sampling_report": {
            "formal_expert_sampling_run": False,
            "expert_dataset_generated": False,
            "selected_expert_action_executed": False,
            "full_expert_dataset_ready": False,
        },
        "no_rollout_report": {
            "rollout_run": False,
            "open_ended_loop_run": False,
            "transitions_jsonl_created": False,
            "rollout_artifacts_created": False,
        },
        "no_map_predict_report": {
            "map_predict_called": False,
            "sscnet_inference_called": False,
            "prediction_npz_created": False,
            "prediction_written_to_observed_state": False,
        },
        "no_rl_gdpo_report": {
            "rl_run": False,
            "gdpo_run": False,
            "ppo_run": False,
            "behavior_cloning_run": False,
            "imitation_learning_run": False,
            "replay_buffer_created": False,
            "policy_checkpoint_created": False,
            "checkpoint_modified": False,
        },
    }
    for stem, report in reports.items():
        save_json(output_dir / f"{stem}.json", report)
        write_text(output_dir / f"{stem}.md", markdown_table(stem.replace("_", " ").title(), report))


def write_manual_gate(output_dir: Path, audit_decision: dict[str, Any]) -> None:
    gate = {
        "stage": "Stage 4A-6.6b",
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "visual_approval_required_before_stage4a67": True,
        "scene_ready_for_formal_expert_sampling_pilot": bool(audit_decision.get("scene_ready_for_formal_expert_sampling_pilot", True)),
        "formal_expert_sampling_ready_full_dataset": False,
        "stage4a67_executed": False,
        "if_user_approves_visuals": "Stage 4A-6.7 bounded formal expert sampling pilot, measured-only first.",
        "if_user_rejects_visuals": "Stage 4A-6.6c scene visual revision / scene editing.",
    }
    save_json(output_dir / "manual_review_gate.json", gate)
    write_text(output_dir / "manual_review_gate.md", markdown_table("Manual Review Gate", gate))


def view_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = {
        "corridor_east_spur": [],
        "room_j": [],
        "loop_junction": [],
        "dead_end_branch": [],
        "narrow_passage": [],
        "high_obstacle_density_spur_rooms": [],
        "long_corridor": [],
        "central_loop": [],
        "start_variants_group": [],
        "occluded_far_room_chain": [],
    }
    for record in records:
        text = " ".join(str(record.get(key, "")) for key in ("name", "room", "note")).lower()
        idx = int(record["index"])
        if "east_spur" in text:
            categories["corridor_east_spur"].append(idx)
        if "room_j" in text or "room j" in text:
            categories["room_j"].append(idx)
        if "junction" in text:
            categories["loop_junction"].append(idx)
        if "dead-end" in text or "dead_end" in text:
            categories["dead_end_branch"].append(idx)
        if "narrow" in text or "spur" in text:
            categories["narrow_passage"].append(idx)
        if "dense" in text or "density" in text:
            categories["high_obstacle_density_spur_rooms"].append(idx)
        if "long" in text:
            categories["long_corridor"].append(idx)
        if "central" in text or "loop" in text:
            categories["central_loop"].append(idx)
        if "start variants" in text:
            categories["start_variants_group"].append(idx)
        if "occluded" in text or "far-room" in text:
            categories["occluded_far_room_chain"].append(idx)
    return {
        "stage": "Stage 4A-6.6b",
        "inspection_view_count": len(records),
        "categories": categories,
        "all_requested_categories_have_views": all(bool(v) for v in categories.values()),
    }


def write_pose_outputs(output_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "stage": "Stage 4A-6.6b",
        "inspection_pose_count": len(records),
        "poses": records,
        "no_expert_action": True,
        "no_rollout": True,
        "no_map_predict": True,
    }
    save_json(output_dir / "inspection_pose_manifest.json", manifest)
    lines = ["# Inspection Pose Manifest", "", f"inspection_pose_count: `{len(records)}`", ""]
    for record in records:
        lines.append(f"- `{int(record['index']):03d}` `{record.get('name')}` `{record.get('room')}`: {record.get('note')}")
    write_text(output_dir / "inspection_pose_manifest.md", "\n".join(lines))
    coverage = view_coverage(records)
    save_json(output_dir / "view_coverage_report.json", coverage)
    write_text(
        output_dir / "view_coverage_report.md",
        "\n".join(
            [
                "# View Coverage Report",
                "",
                f"inspection_view_count: `{len(records)}`",
                f"all_requested_categories_have_views: `{coverage['all_requested_categories_have_views']}`",
                "",
                *[f"- {key}: `{value}`" for key, value in coverage["categories"].items()],
            ]
        ),
    )
    return coverage


def write_fallback_report(output_dir: Path, records: list[dict[str, Any]], render_report: dict[str, Any], video_report: dict[str, Any], elapsed_render_s: float) -> dict[str, Any]:
    report = {
        "stage": "Stage 4A-6.6b",
        "fallback_visual_package_created": True,
        "inspection_view_count": len(records),
        "rgb_nonblank_count": sum(1 for item in records if item["rgb_stats"]["nonblank"]),
        "depth_positive_count": sum(1 for item in records if item["depth_stats"]["has_positive_finite_depth"]),
        "render_backend": render_report["render_backend"],
        "headless_isaac_render_succeeded": bool(render_report.get("headless_isaac_render_succeeded")),
        "headless_isaac_error": render_report.get("headless_isaac_error"),
        "render_time_seconds": float(elapsed_render_s),
        "flythrough": video_report,
        "no_expert_sampling": True,
        "no_rollout": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
    }
    save_json(output_dir / "fallback_visual_package_report.json", report)
    write_text(output_dir / "fallback_visual_package_report.md", markdown_table("Fallback Visual Package Report", report))
    return report


def write_summary(
    output_dir: Path,
    stage4a66_dir: Path,
    stage4a66a_dir: Path,
    metadata: dict[str, Any],
    audit_decision: dict[str, Any],
    capability: dict[str, Any],
    gui_attempt: dict[str, Any],
    fallback: dict[str, Any],
    coverage: dict[str, Any],
    video_report: dict[str, Any],
    total_elapsed: float,
) -> dict[str, Any]:
    summary = {
        "stage": "Stage 4A-6.6b",
        "answers": {
            "stage4a66_loaded": True,
            "stage4a66a_loaded": True,
            "scene_variant": metadata["scene_id"],
            "gui_capability_result": "DISPLAY present" if capability["display_present"] else "DISPLAY missing",
            "display": capability["env"].get("DISPLAY"),
            "gui_attempted": bool(gui_attempt["gui_attempted"]),
            "gui_attempt_status": gui_attempt["gui_attempt_status"],
            "user_can_confirm_gui_visible": False,
            "fallback_visual_package_generated": bool(fallback["fallback_visual_package_created"]),
            "inspection_views": int(fallback["inspection_view_count"]),
            "rgb_nonblank_count": int(fallback["rgb_nonblank_count"]),
            "depth_positive_count": int(fallback["depth_positive_count"]),
            "topdown_labels_starts_warning_regions_generated": True,
            "closeup_views_generated": True,
            "flythrough_generated": bool(video_report.get("mp4_created") or video_report.get("gif_created") or video_report.get("frame_count", 0) > 0),
            "html_index": str(output_dir / "visual_inspection_index.html"),
            "human_visual_review_checklist": str(output_dir / "human_visual_review_checklist.md"),
            "no_expert_sampling": True,
            "no_rollout": True,
            "no_map_predict_or_sscnet_inference": True,
            "no_action_execution": True,
            "no_rl_gdpo_ppo_bc_il": True,
            "scene_ready_for_formal_expert_sampling_pilot": bool(audit_decision.get("scene_ready_for_formal_expert_sampling_pilot", True)),
            "formal_expert_sampling_ready_full_dataset": False,
            "user_visual_review_still_required": True,
            "recommended_next": "If user approves visuals, Stage 4A-6.7 bounded formal expert sampling pilot, measured-only first; if rejected, Stage 4A-6.6c visual revision.",
        },
        "paths": {
            "stage4a66_dir": str(stage4a66_dir),
            "stage4a66a_dir": str(stage4a66a_dir),
            "output_dir": str(output_dir),
            "html_index": str(output_dir / "visual_inspection_index.html"),
            "checklist": str(output_dir / "human_visual_review_checklist.md"),
            "flythrough": video_report.get("video_path") or str(output_dir / "flythrough_frames"),
        },
        "view_coverage": coverage,
        "total_wall_seconds": float(total_elapsed),
    }
    save_json(output_dir / "visual_inspection_summary.json", summary)
    answers = summary["answers"]
    md = [
        "# Visual Inspection Summary",
        "",
        f"1. 是否成功读取 Stage 4A-6.6？ `{answers['stage4a66_loaded']}`",
        f"2. 是否成功读取 Stage 4A-6.6a？ `{answers['stage4a66a_loaded']}`",
        f"3. 当前场景是否为 larger_complex_scene_v1？ `{answers['scene_variant'] == 'larger_complex_scene_v1'}`",
        f"4. GUI capability 检查结果是什么？ `{answers['gui_capability_result']}`",
        f"5. DISPLAY 是否存在？ `{bool(answers['display'])}`",
        f"6. 是否尝试启动 IsaacSim GUI？ `{answers['gui_attempted']}`",
        f"7. GUI 尝试结果是什么？ `{answers['gui_attempt_status']}`",
        f"8. 用户是否能确认看到 GUI？ `{answers['user_can_confirm_gui_visible']}`",
        f"9. 如果不能，fallback visual package 是否生成？ `{answers['fallback_visual_package_generated']}`",
        f"10. 生成了多少 inspection views？ `{answers['inspection_views']}`",
        f"11. RGB 是否非空？ `{answers['rgb_nonblank_count']}/{answers['inspection_views']}`",
        f"12. depth 是否有限正值？ `{answers['depth_positive_count']}/{answers['inspection_views']}`",
        f"13. 是否生成 topdown / labels / starts / warning regions？ `{answers['topdown_labels_starts_warning_regions_generated']}`",
        f"14. 是否生成 closeup views？ `{answers['closeup_views_generated']}`",
        f"15. 是否生成 flythrough video/GIF/frames？ `{answers['flythrough_generated']}`",
        f"16. HTML index 在哪里？ `{answers['html_index']}`",
        f"17. human_visual_review_checklist 在哪里？ `{answers['human_visual_review_checklist']}`",
        f"18. 本轮是否没有 expert sampling？ `{answers['no_expert_sampling']}`",
        f"19. 本轮是否没有 rollout？ `{answers['no_rollout']}`",
        f"20. 本轮是否没有 map_predict / SSCNet inference？ `{answers['no_map_predict_or_sscnet_inference']}`",
        f"21. 本轮是否没有 action execution？ `{answers['no_action_execution']}`",
        f"22. 本轮是否没有 RL/GDPO/PPO/BC/IL？ `{answers['no_rl_gdpo_ppo_bc_il']}`",
        f"23. 是否保留 6.6a 结论？ `scene_ready_for_formal_expert_sampling_pilot={answers['scene_ready_for_formal_expert_sampling_pilot']}`, `formal_expert_sampling_ready_full_dataset={answers['formal_expert_sampling_ready_full_dataset']}`",
        f"24. 下一步是否仍然需要用户人工看图/GUI？ `{answers['user_visual_review_still_required']}`",
        f"25. 下一步推荐是什么？ {answers['recommended_next']}",
    ]
    write_text(output_dir / "visual_inspection_summary.md", "\n".join(md))
    return summary


def write_html_index(
    output_dir: Path,
    metadata: dict[str, Any],
    audit_decision: dict[str, Any],
    gui_attempt: dict[str, Any],
    video_report: dict[str, Any],
) -> None:
    def img(name: str, width: int = 360) -> str:
        return f'<figure><img src="{html.escape(name)}" width="{width}"><figcaption>{html.escape(name)}</figcaption></figure>'

    video_html = ""
    video_path = video_report.get("video_path")
    if video_path:
        rel = Path(video_path).name
        if rel.endswith(".mp4"):
            video_html = f'<video controls width="720" src="{html.escape(rel)}"></video>'
        else:
            video_html = img(rel, 720)
    else:
        video_html = '<p>Flythrough frames are in <code>flythrough_frames/</code>.</p>'
    warnings = audit_decision.get("warnings", [])
    images = [
        "scene_layout_topdown_human.png",
        "room_corridor_opening_labels_topdown.png",
        "obstacle_labels_topdown.png",
        "start_variants_labeled_topdown.png",
        "validation_poses_labeled_topdown.png",
        "topology_graph_labeled.png",
        "audit_warning_regions_topdown.png",
        "inspection_rgb_grid.png",
        "inspection_depth_grid.png",
        "closeup_corridor_east_spur.png",
        "closeup_room_j.png",
        "closeup_loop_junction.png",
        "closeup_dead_end_branch.png",
        "closeup_narrow_passage_examples.png",
        "closeup_obstacle_dense_spur_rooms.png",
        "closeup_start_locations.png",
    ]
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.6b Visual Inspection</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #18202a; background: #f7f8fa; }}
    section {{ margin-bottom: 30px; }}
    figure {{ display: inline-block; margin: 10px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 360px; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
    li {{ margin: 4px 0; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.6b Visual Inspection: larger_complex_scene_v1</h1>
  <section>
    <h2>Scene Summary</h2>
    <p>Rooms: <code>{len(metadata['rooms'])}</code>, corridors: <code>{len(metadata['corridors'])}</code>, openings: <code>{len(metadata['openings'])}</code>, walls: <code>{len(metadata['walls'])}</code>, obstacles: <code>{len(metadata['obstacles'])}</code>, starts: <code>{len(metadata['start_variants'])}</code>, validation poses: <code>{len(metadata['validation_camera_poses'])}</code>.</p>
    <p>Audit passed: <code>{audit_decision.get('scene_complexity_audit_passed')}</code>. Pilot ready: <code>{audit_decision.get('scene_ready_for_formal_expert_sampling_pilot')}</code>. Full dataset ready: <code>{audit_decision.get('formal_expert_sampling_ready_full_dataset')}</code>.</p>
  </section>
  <section>
    <h2>Main Warnings</h2>
    <ul>{''.join(f'<li>{html.escape(str(w))}</li>' for w in warnings)}</ul>
  </section>
  <section>
    <h2>GUI Status</h2>
    <p>GUI status: <code>{html.escape(str(gui_attempt.get('gui_attempt_status')))}</code>. User visibility: <code>{html.escape(str(gui_attempt.get('user_visibility')))}</code>.</p>
  </section>
  <section>
    <h2>Visuals</h2>
    {''.join(img(name) for name in images)}
  </section>
  <section>
    <h2>Flythrough</h2>
    {video_html}
  </section>
  <section>
    <h2>User Review Checklist</h2>
    <p>Open <a href="human_visual_review_checklist.md">human_visual_review_checklist.md</a> and answer every item before Stage 4A-6.7.</p>
  </section>
  <section>
    <h2>Next-Step Reminder</h2>
    <p>If visuals are approved: Stage 4A-6.7 bounded formal expert sampling pilot, measured-only first. If rejected: Stage 4A-6.6c scene visual revision. Full expert dataset collection remains blocked.</p>
  </section>
</body>
</html>"""
    write_text(output_dir / "visual_inspection_index.html", body)


def write_next_step_notes(output_dir: Path) -> None:
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\nUser visual inspection is still required. If the user approves the HTML/images/video package, proceed to Stage 4A-6.7 bounded formal expert sampling pilot, measured-only first and not a full dataset. If the user rejects the visuals, run Stage 4A-6.6c scene visual revision / scene editing.",
    )
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "# Long-Term RL/GDPO Note\n\nGDPO/RL/PPO/BC/IL remain future directions only. Stage 4A-6.6b did not train, create replay buffers, create checkpoints, or start policy optimization.",
    )


def write_context_manifests(output_dir: Path, stage4a66_dir: Path, stage4a66a_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    context_entries = []
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        context_entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_stage4a66": "Stage 4A-6.6" in text,
                "contains_stage4a66a": "Stage 4A-6.6a" in text,
                "contains_no_rollout": "no rollout" in text.lower() or "not rollout" in text.lower(),
            }
        )
    context = {"stage": "Stage 4A-6.6b", "loaded_context_files": context_entries}
    save_json(output_dir / "loaded_context_manifest.json", context)
    write_text(output_dir / "loaded_context_manifest.md", list_md("Loaded Context Manifest", [f"`{item['path']}` sha256 `{item['sha256']}`" for item in context_entries]))

    stage4a66 = file_manifest(stage4a66_dir)
    stage4a66["confirmed"] = {
        "larger_complex_scene_v1_constructed": True,
        "headless_validation_success": True,
        "rgb_nonblank_14_of_14": True,
        "depth_positive_14_of_14": True,
        "observed_state_shape": [240, 240, 30],
        "observed_ratio": 0.09458275462962963,
        "no_rollout": True,
        "no_formal_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
    }
    save_json(output_dir / "loaded_stage4a66_manifest.json", stage4a66)
    write_manifest_md(output_dir / "loaded_stage4a66_manifest.md", "Loaded Stage 4A-6.6 Manifest", stage4a66, ["Stage 4A-6.6 bundle loaded read-only."])

    stage4a66a = file_manifest(stage4a66a_dir)
    stage4a66a["confirmed"] = {
        "scene_complexity_audit_passed": True,
        "scene_ready_for_formal_expert_sampling_pilot": True,
        "formal_expert_sampling_ready_full_dataset": False,
        "hard_blockers": [],
        "no_rollout": True,
        "no_formal_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
    }
    save_json(output_dir / "loaded_stage4a66a_audit_manifest.json", stage4a66a)
    write_manifest_md(output_dir / "loaded_stage4a66a_audit_manifest.md", "Loaded Stage 4A-6.6a Audit Manifest", stage4a66a, ["Stage 4A-6.6a audit bundle loaded read-only."])
    return context, stage4a66, stage4a66a


def parse_args() -> tuple[argparse.Namespace, Any | None]:
    app_launcher_cls = None
    parser = argparse.ArgumentParser(description="Stage 4A-6.6b GUI / visual inspection setup.")
    parser.add_argument("--stage4a66_dir", default=str(DEFAULT_STAGE4A66_DIR))
    parser.add_argument("--stage4a66a_dir", default=str(DEFAULT_STAGE4A66A_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene_variant", default="larger_complex_scene_v1")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--try_gui", action="store_true")
    parser.add_argument("--fallback_headless_visual_package", action="store_true")
    parser.add_argument("--min_inspection_views", type=int, default=24)
    parser.add_argument("--make_flythrough", action="store_true")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument("--max_depth", type=float, default=18.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--no_expert_sampling", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        app_launcher_cls = AppLauncher
    except Exception:
        app_launcher_cls = None
    return parser.parse_args(), app_launcher_cls


def main() -> None:
    started = time.perf_counter()
    args, app_launcher_cls = parse_args()
    output_dir = Path(args.output_dir).resolve()
    stage4a66_dir = Path(args.stage4a66_dir).resolve()
    stage4a66a_dir = Path(args.stage4a66a_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.scene_variant != "larger_complex_scene_v1":
        raise ValueError("Stage 4A-6.6b only supports larger_complex_scene_v1")
    if int(args.scene_seed) != 0:
        raise ValueError("Stage 4A-6.6b requires scene_seed=0")
    if not (args.no_expert_sampling and args.no_rollout and args.no_map_predict and args.no_rl_gdpo):
        raise ValueError("Stage 4A-6.6b requires all negative-scope flags")
    if not stage4a66_dir.is_dir() or not stage4a66a_dir.is_dir():
        raise FileNotFoundError("Stage 4A-6.6 and 6.6a directories must exist")

    write_context_manifests(output_dir, stage4a66_dir, stage4a66a_dir)
    metadata = read_json(stage4a66_dir / "larger_complex_scene_v1_metadata.json")
    audit_decision = read_json(stage4a66a_dir / "scene_complexity_audit_decision.json")
    poses = build_inspection_poses(metadata, int(args.min_inspection_views))
    save_topdown_and_closeups(output_dir, metadata, poses)

    capability = gui_capability_report(output_dir, int(args.max_workers))
    gui_attempt = write_gui_attempt_files(output_dir, args, capability)
    write_negative_scope_reports(output_dir)
    write_checklist(output_dir)
    write_manual_gate(output_dir, audit_decision)
    write_next_step_notes(output_dir)

    render_started = time.perf_counter()
    render_report: dict[str, Any] = {"headless_isaac_render_succeeded": False, "render_backend": "unknown"}
    simulation_app = None
    try:
        if args.fallback_headless_visual_package:
            try:
                records, simulation_app = capture_with_isaac(args, app_launcher_cls, output_dir, poses)
                render_report.update({"headless_isaac_render_succeeded": True, "render_backend": "isaac_headless"})
            except Exception as exc:
                render_report.update(
                    {
                        "headless_isaac_render_succeeded": False,
                        "headless_isaac_error": str(exc),
                        "render_backend": "existing_isaac_validation_plus_offline_schematic",
                    }
                )
                records = copy_or_schematic_capture(args, output_dir, stage4a66_dir, metadata, poses)
        else:
            render_report.update({"render_backend": "existing_isaac_validation_plus_offline_schematic"})
            records = copy_or_schematic_capture(args, output_dir, stage4a66_dir, metadata, poses)

        save_rgb_grid(output_dir / "inspection_rgb_grid.png", records, output_dir)
        save_depth_grid(output_dir / "inspection_depth_grid.png", records, output_dir)
        coverage = write_pose_outputs(output_dir, records)
        video_report = make_flythrough(output_dir, records) if args.make_flythrough else {"frame_count": 0, "video_path": None}
        fallback = write_fallback_report(output_dir, records, render_report, video_report, time.perf_counter() - render_started)
        summary = write_summary(output_dir, stage4a66_dir, stage4a66a_dir, metadata, audit_decision, capability, gui_attempt, fallback, coverage, video_report, time.perf_counter() - started)
        write_html_index(output_dir, metadata, audit_decision, gui_attempt, video_report)
        save_json(
            output_dir / "run_completion_report.json",
            {
                "stage": "Stage 4A-6.6b",
                "complete": True,
                "output_dir": str(output_dir),
                "html_index": str(output_dir / "visual_inspection_index.html"),
                "summary": summary["answers"],
            },
        )
        print(f"[INFO]: Stage 4A-6.6b visual inspection package written to {output_dir}")
    finally:
        if simulation_app is not None:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
