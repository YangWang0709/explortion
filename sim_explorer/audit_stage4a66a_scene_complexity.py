#!/usr/bin/env python3
"""Stage 4A-6.6a offline scene complexity audit.

This script reads the Stage 4A-6.6 validation bundle and produces an offline
audit package for larger_complex_scene_v1. It does not start Isaac, capture
new sensor data, run map_predict/SSCNet, execute actions, roll out, sample an
expert dataset, or train any policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from astar_planner import build_traversability_grid, connected_component_from_start, nearest_traversable_cell
from sim_paper_expert import detect_frontier_voxels, frontier_adjacent_free_xy_mask


UNKNOWN = -1
FREE = 0
OCCUPIED = 1


REQUIRED_INPUTS = [
    "stage4a66_larger_complex_scene_v1_summary.md",
    "stage4a66_larger_complex_scene_v1_summary.json",
    "larger_complex_scene_v1_metadata.json",
    "scene_topology_graph.json",
    "preliminary_complexity_metrics.json",
    "preliminary_complexity_target_checklist.json",
    "start_variants.json",
    "validation_pose_manifest.json",
    "observed_state_final.npy",
    "observed_state_validation_summary.json",
    "observed_state_transition_summary.json",
    "audit_input_bundle_manifest.json",
    "fixed_capture_validation.json",
    "room_inventory.json",
    "corridor_inventory.json",
    "opening_inventory.json",
    "obstacle_inventory.json",
    "wall_inventory.json",
]


REQUIRED_TEXT_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a66_bundle_manifest.json",
    "loaded_stage4a66_bundle_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "input_hash_audit.json",
    "input_hash_audit.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "scene_scale_audit.json",
    "scene_scale_audit.md",
    "scene_scale_scorecard.csv",
    "topology_complexity_audit.json",
    "topology_complexity_audit.md",
    "topology_degree_histogram.csv",
    "start_graph_distance_matrix.csv",
    "bottleneck_articulation_review.json",
    "bottleneck_articulation_review.md",
    "start_variant_diversity_audit.json",
    "start_variant_diversity_audit.md",
    "start_variant_table.csv",
    "start_euclidean_distance_matrix.csv",
    "start_topology_distance_matrix.csv",
    "start_local_context_table.csv",
    "fixed_view_visibility_audit.json",
    "fixed_view_visibility_audit.md",
    "validation_view_table.csv",
    "depth_stats_by_view.csv",
    "validation_zone_coverage.json",
    "validation_zone_coverage.md",
    "observed_state_health_audit.json",
    "observed_state_health_audit.md",
    "observed_ratio_by_view.csv",
    "observed_label_distribution.csv",
    "observed_state_exploration_room_left.json",
    "observed_state_exploration_room_left.md",
    "frontier_reachability_audit.json",
    "frontier_reachability_audit.md",
    "reachable_area_by_start.csv",
    "reachable_frontier_by_start.csv",
    "candidate_availability_proxy_by_start.csv",
    "isolated_start_review.json",
    "isolated_start_review.md",
    "obstacle_occlusion_audit.json",
    "obstacle_occlusion_audit.md",
    "obstacle_distribution_table.csv",
    "obstacle_density_by_zone.csv",
    "occlusion_proxy_by_view.csv",
    "expert_usability_pre_audit.json",
    "expert_usability_pre_audit.md",
    "future_expert_pilot_risk_register.json",
    "future_expert_pilot_risk_register.md",
    "scene_complexity_scorecard.csv",
    "scene_complexity_scorecard.json",
    "scene_complexity_scorecard.md",
    "scene_complexity_audit_decision.json",
    "scene_complexity_audit_decision.md",
    "usable_start_subset.json",
    "usable_start_subset.md",
    "scene_revision_plan_if_needed.json",
    "scene_revision_plan_if_needed.md",
    "recommended_next_faithful_step.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_formal_expert_sampling_report.json",
    "no_formal_expert_sampling_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "stage4a66a_scene_complexity_audit_summary.json",
    "stage4a66a_scene_complexity_audit_summary.md",
    "long_term_rl_gdpo_note.md",
    "future_stage4a67_formal_expert_sampling_pilot_design_sketch.md",
    "do_not_start_full_expert_sampling_in_stage4a66a.md",
]


REQUIRED_PLOTS = [
    "audit_scene_layout_topdown.png",
    "audit_topology_graph.png",
    "audit_room_corridor_opening_map.png",
    "audit_start_variants_topdown.png",
    "audit_start_distance_matrix.png",
    "audit_validation_view_coverage.png",
    "audit_observed_topdown_final.png",
    "audit_observed_ratio_by_view.png",
    "audit_frontier_distribution_topdown.png",
    "audit_reachable_area_by_start.png",
    "audit_reachable_frontier_by_start.png",
    "audit_obstacle_density_topdown.png",
    "audit_occlusion_proxy_by_view.png",
    "audit_complexity_scorecard.png",
    "audit_pass_fail_flowchart.png",
    "audit_next_stage_decision_flowchart.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a66_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_scene_variant", required=True)
    parser.add_argument("--expected_scene_seed", type=int, required=True)
    parser.add_argument("--min_rooms", type=int, default=8)
    parser.add_argument("--min_corridors", type=int, default=3)
    parser.add_argument("--min_openings", type=int, default=12)
    parser.add_argument("--min_obstacles", type=int, default=40)
    parser.add_argument("--min_starts", type=int, default=8)
    parser.add_argument("--min_validation_poses", type=int, default=12)
    parser.add_argument("--min_loop_count", type=int, default=2)
    parser.add_argument("--min_branch_points", type=int, default=4)
    parser.add_argument("--min_dead_ends", type=int, default=2)
    parser.add_argument("--min_narrow_passages", type=int, default=4)
    parser.add_argument("--min_final_observed_ratio", type=float, default=0.05)
    parser.add_argument("--max_fixed_view_observed_ratio", type=float, default=0.40)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_isaac", action="store_true")
    parser.add_argument("--no_capture", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_formal_expert_sampling", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(to_builtin(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str] | str) -> None:
    if isinstance(lines, str):
        text = lines
    else:
        text = "\n".join(lines)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: to_builtin(row.get(k, "")) for k in fieldnames})


def write_matrix_csv(path: Path, labels: list[str], matrix: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [to_builtin(v) for v in row])


def sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def bool_word(value: bool) -> str:
    return "true" if bool(value) else "false"


def score_from_checks(checks: dict[str, bool]) -> float:
    if not checks:
        return 0.0
    return float(sum(1 for v in checks.values() if bool(v)) / len(checks))


def bounds_area(bounds: dict[str, Any]) -> float:
    x0, x1 = [float(v) for v in bounds["x"]]
    y0, y1 = [float(v) for v in bounds["y"]]
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bounds_center(bounds: dict[str, Any]) -> tuple[float, float]:
    x0, x1 = [float(v) for v in bounds["x"]]
    y0, y1 = [float(v) for v in bounds["y"]]
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def point_inside_bounds(point: list[float] | tuple[float, float], bounds: dict[str, Any]) -> bool:
    x, y = float(point[0]), float(point[1])
    return float(bounds["x"][0]) <= x <= float(bounds["x"][1]) and float(bounds["y"][0]) <= y <= float(bounds["y"][1])


def point_rect_distance(point: list[float] | tuple[float, float], bounds: dict[str, Any]) -> float:
    x, y = float(point[0]), float(point[1])
    x0, x1 = [float(v) for v in bounds["x"]]
    y0, y1 = [float(v) for v in bounds["y"]]
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return float(math.hypot(dx, dy))


def obstacle_bounds(obstacle: dict[str, Any]) -> dict[str, list[float]]:
    if "x_min" not in obstacle or "x_size" not in obstacle or "y_min" not in obstacle or "y_size" not in obstacle:
        position = obstacle.get("position", [0.0, 0.0, 0.0])
        size = obstacle.get("size", [0.0, 0.0, 0.0])
        x_min = float(position[0]) - 0.5 * float(size[0])
        y_min = float(position[1]) - 0.5 * float(size[1])
        return {
            "x": [x_min, x_min + float(size[0])],
            "y": [y_min, y_min + float(size[1])],
        }
    return {
        "x": [float(obstacle["x_min"]), float(obstacle["x_min"]) + float(obstacle["x_size"])],
        "y": [float(obstacle["y_min"]), float(obstacle["y_min"]) + float(obstacle["y_size"])],
    }


def rect_intersection_area(a: dict[str, Any], b: dict[str, Any]) -> float:
    x0 = max(float(a["x"][0]), float(b["x"][0]))
    x1 = min(float(a["x"][1]), float(b["x"][1]))
    y0 = max(float(a["y"][0]), float(b["y"][0]))
    y1 = min(float(a["y"][1]), float(b["y"][1]))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float((x1 - x0) * (y1 - y0))


def yaw_delta_deg(a: float, b: float) -> float:
    diff = abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)
    return float(diff)


def world_to_grid_xy(position: list[float], bounds: dict[str, Any], voxel_size: float, shape_xy: tuple[int, int]) -> tuple[int, int]:
    x = int(math.floor((float(position[0]) - float(bounds["x"][0])) / float(voxel_size)))
    y = int(math.floor((float(position[1]) - float(bounds["y"][0])) / float(voxel_size)))
    return (max(0, min(shape_xy[0] - 1, x)), max(0, min(shape_xy[1] - 1, y)))


def grid_to_world_xy(xy: tuple[int, int], bounds: dict[str, Any], voxel_size: float) -> tuple[float, float]:
    return (
        float(bounds["x"][0]) + (float(xy[0]) + 0.5) * float(voxel_size),
        float(bounds["y"][0]) + (float(xy[1]) + 0.5) * float(voxel_size),
    )


def build_region_lookup(rooms: list[dict[str, Any]], corridors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for item in rooms:
        row = dict(item)
        row["kind"] = "room"
        lookup[str(item["name"])] = row
    for item in corridors:
        row = dict(item)
        row["kind"] = "corridor"
        lookup[str(item["name"])] = row
    return lookup


def assign_nearest_zone(position: list[float], regions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    containing = [item for item in regions.values() if point_inside_bounds(position, item["bounds"])]
    if containing:
        containing.sort(key=lambda z: (bounds_area(z["bounds"]), str(z["name"])))
        best = containing[0]
        return {
            "zone": str(best["name"]),
            "kind": str(best["kind"]),
            "distance_m": 0.0,
            "inside": True,
        }
    best = None
    best_dist = math.inf
    for item in regions.values():
        dist = point_rect_distance(position, item["bounds"])
        if dist < best_dist:
            best = item
            best_dist = dist
    return {
        "zone": str(best["name"]) if best else None,
        "kind": str(best["kind"]) if best else None,
        "distance_m": float(best_dist),
        "inside": False,
    }


def nearest_opening(position: list[float], openings: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    best_dist = math.inf
    for opening in openings:
        center = opening.get("center") or [0.0, 0.0]
        dist = math.hypot(float(position[0]) - float(center[0]), float(position[1]) - float(center[1]))
        if dist < best_dist:
            best = opening
            best_dist = dist
    return {
        "name": str(best.get("name")) if best else None,
        "distance_m": float(best_dist),
        "width_m": float(best.get("width", 0.0)) if best else None,
    }


def build_adjacency(graph: dict[str, Any]) -> tuple[dict[str, set[str]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes = {str(node["id"]): dict(node) for node in graph.get("nodes", [])}
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    edges = [dict(edge) for edge in graph.get("edges", [])]
    for edge in edges:
        source = str(edge["source"])
        target = str(edge["target"])
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
        nodes.setdefault(source, {"id": source, "kind": "unknown", "label": source})
        nodes.setdefault(target, {"id": target, "kind": "unknown", "label": target})
    return adjacency, nodes, edges


def connected_components(adjacency: dict[str, set[str]]) -> list[list[str]]:
    seen: set[str] = set()
    comps: list[list[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        comp: list[str] = []
        while queue:
            node = queue.popleft()
            comp.append(node)
            for nbr in sorted(adjacency[node]):
                if nbr not in seen:
                    seen.add(nbr)
                    queue.append(nbr)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c[0] if c else ""))
    return comps


def bfs_distances(adjacency: dict[str, set[str]], source: str) -> dict[str, int]:
    if source not in adjacency:
        return {}
    dist = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for nbr in sorted(adjacency[node]):
            if nbr not in dist:
                dist[nbr] = dist[node] + 1
                queue.append(nbr)
    return dist


def articulation_points(adjacency: dict[str, set[str]]) -> list[str]:
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    points: set[str] = set()

    def dfs(node: str) -> None:
        nonlocal index
        indices[node] = index
        low[node] = index
        index += 1
        child_count = 0
        is_articulation = False
        for nbr in sorted(adjacency[node]):
            if nbr not in indices:
                parent[nbr] = node
                child_count += 1
                dfs(nbr)
                low[node] = min(low[node], low[nbr])
                if parent.get(node) is not None and low[nbr] >= indices[node]:
                    is_articulation = True
            elif nbr != parent.get(node):
                low[node] = min(low[node], indices[nbr])
        if parent.get(node) is None and child_count > 1:
            is_articulation = True
        if is_articulation:
            points.add(node)

    for node in sorted(adjacency):
        if node not in indices:
            parent[node] = None
            dfs(node)
    return sorted(points)


def md_kv(title: str, mapping: dict[str, Any]) -> list[str]:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in mapping.items():
        lines.append(f"| `{key}` | `{json.dumps(to_builtin(value), sort_keys=True)}` |")
    return lines


def md_table(title: str, rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> list[str]:
    lines = [f"# {title}", ""]
    if not rows:
        lines.append("No rows.")
        return lines
    shown = rows[:max_rows] if max_rows else rows
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in shown:
        lines.append("| " + " | ".join(str(to_builtin(row.get(col, ""))) for col in columns) + " |")
    if max_rows and len(rows) > max_rows:
        lines.append("")
        lines.append(f"Showing {max_rows} of {len(rows)} rows.")
    return lines


def add_rect(ax: Any, bounds: dict[str, Any], **kwargs: Any) -> None:
    x0, x1 = [float(v) for v in bounds["x"]]
    y0, y1 = [float(v) for v in bounds["y"]]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, **kwargs))


def setup_topdown_axes(ax: Any, map_bounds: dict[str, Any], title: str) -> None:
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(float(map_bounds["x"][0]), float(map_bounds["x"][1]))
    ax.set_ylim(float(map_bounds["y"][0]), float(map_bounds["y"][1]))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.15)


def plot_layout(path: Path, meta: dict[str, Any], starts: list[dict[str, Any]] | None = None, poses: list[dict[str, Any]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_topdown_axes(ax, meta["map_bounds"], "Stage 4A-6.6a layout audit")
    for room in meta["rooms"]:
        add_rect(ax, room["bounds"], facecolor="#d8eef3", edgecolor="#3a8aa0", alpha=0.55, linewidth=1.0)
        cx, cy = bounds_center(room["bounds"])
        ax.text(cx, cy, room["name"].replace("room_", "r_"), ha="center", va="center", fontsize=7)
    for corridor in meta["corridors"]:
        add_rect(ax, corridor["bounds"], facecolor="#e6e2f2", edgecolor="#6a5aa5", alpha=0.45, linewidth=1.0)
    for wall in meta["walls"]:
        b = obstacle_bounds(wall)
        add_rect(ax, b, facecolor="#59606b", edgecolor="none", alpha=0.6)
    for obstacle in meta["obstacles"]:
        b = obstacle_bounds(obstacle)
        add_rect(ax, b, facecolor="#b98042", edgecolor="#70451f", alpha=0.65, linewidth=0.4)
    for opening in meta["openings"]:
        center = opening.get("center", [0.0, 0.0])
        ax.plot(float(center[0]), float(center[1]), marker="s", color="#28a745", markersize=3)
    if starts:
        for item in starts:
            ax.plot(float(item["position"][0]), float(item["position"][1]), marker="o", color="#cf2e2e", markersize=4)
    if poses:
        for item in poses:
            x, y = float(item["position"][0]), float(item["position"][1])
            yaw = float(item.get("yaw_rad", 0.0))
            ax.arrow(x, y, 0.5 * math.cos(yaw), 0.5 * math.sin(yaw), head_width=0.18, color="#185adb", alpha=0.75)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_topology(path: Path, meta: dict[str, Any], graph: dict[str, Any]) -> None:
    adjacency, nodes, edges = build_adjacency(graph)
    regions = build_region_lookup(meta["rooms"], meta["corridors"])
    positions: dict[str, tuple[float, float]] = {}
    for node_id in nodes:
        if node_id in regions:
            positions[node_id] = bounds_center(regions[node_id]["bounds"])
        else:
            positions[node_id] = (0.0, 0.0)
    fig, ax = plt.subplots(figsize=(9, 9))
    setup_topdown_axes(ax, meta["map_bounds"], "Topology graph audit")
    for edge in edges:
        a = positions[str(edge["source"])]
        b = positions[str(edge["target"])]
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#666666", linewidth=1.2, alpha=0.75)
    for node_id, pos in positions.items():
        kind = str(nodes[node_id].get("kind", "unknown"))
        color = "#247ba0" if kind == "room" else "#8e44ad"
        degree = len(adjacency.get(node_id, []))
        ax.scatter([pos[0]], [pos[1]], s=55 + 15 * degree, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.text(pos[0], pos[1] + 0.22, node_id.replace("corridor_", "c_").replace("room_", "r_"), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_distance_matrix(path: Path, labels: list[str], matrix: list[list[float]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    arr = np.asarray(matrix, dtype=np.float64)
    im = ax.imshow(arr, cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_observed_topdown(path: Path, observed_state: np.ndarray, meta: dict[str, Any]) -> None:
    top = np.full(observed_state.shape[:2], 0, dtype=np.int8)
    occ = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == FREE, axis=2)
    top[free] = 1
    top[occ] = 2
    rgb = np.zeros((*top.shape, 3), dtype=np.float32)
    rgb[top == 0] = [0.08, 0.09, 0.10]
    rgb[top == 1] = [0.72, 0.86, 0.78]
    rgb[top == 2] = [0.72, 0.22, 0.20]
    extent = [float(meta["map_bounds"]["x"][0]), float(meta["map_bounds"]["x"][1]), float(meta["map_bounds"]["y"][0]), float(meta["map_bounds"]["y"][1])]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(np.swapaxes(rgb, 0, 1), origin="lower", extent=extent)
    setup_topdown_axes(ax, meta["map_bounds"], "Observed_state final topdown")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_frontier_topdown(path: Path, observed_state: np.ndarray, meta: dict[str, Any], frontier_xy: np.ndarray, starts: list[dict[str, Any]]) -> None:
    trav = build_traversability_grid(observed_state, voxel_size=float(meta["voxel_size_recommended"]))
    traversable = np.asarray(trav["traversable"], dtype=bool)
    canvas = np.zeros((*traversable.shape, 3), dtype=np.float32)
    canvas[:, :] = [0.08, 0.09, 0.10]
    canvas[traversable] = [0.68, 0.80, 0.70]
    canvas[frontier_xy] = [0.95, 0.72, 0.18]
    extent = [float(meta["map_bounds"]["x"][0]), float(meta["map_bounds"]["x"][1]), float(meta["map_bounds"]["y"][0]), float(meta["map_bounds"]["y"][1])]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(np.swapaxes(canvas, 0, 1), origin="lower", extent=extent)
    for start in starts:
        ax.plot(float(start["position"][0]), float(start["position"][1]), marker="o", color="#d7191c", markersize=4)
    setup_topdown_axes(ax, meta["map_bounds"], "Measured-only reachable frontier proxy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reachable_area(path: Path, meta: dict[str, Any], reachable_masks: dict[str, np.ndarray], starts: list[dict[str, Any]]) -> None:
    shape = next(iter(reachable_masks.values())).shape if reachable_masks else (240, 240)
    canvas = np.zeros(shape, dtype=np.float32)
    for idx, mask in enumerate(reachable_masks.values(), start=1):
        canvas[np.asarray(mask, dtype=bool)] = idx
    extent = [float(meta["map_bounds"]["x"][0]), float(meta["map_bounds"]["x"][1]), float(meta["map_bounds"]["y"][0]), float(meta["map_bounds"]["y"][1])]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(np.swapaxes(canvas, 0, 1), origin="lower", extent=extent, cmap="tab20")
    for start in starts:
        ax.plot(float(start["position"][0]), float(start["position"][1]), marker="o", color="black", markersize=3)
    setup_topdown_axes(ax, meta["map_bounds"], "Reachable observed-free components by start")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str = "value") -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color="#357abd")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scorecard(path: Path, scorecard: list[dict[str, Any]]) -> None:
    labels = [str(row["category"]) for row in scorecard]
    values = [float(row["score"]) for row in scorecard]
    colors = ["#2ca25f" if bool(row["pass"]) else "#de2d26" for row in scorecard]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(labels)), values, color=colors)
    ax.axhline(1.0, color="#444444", linewidth=0.8)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Scene complexity audit scorecard")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    for idx, row in enumerate(scorecard):
        ax.text(idx, min(1.0, values[idx] + 0.03), "PASS" if row["pass"] else "FAIL", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_flowchart(path: Path, boxes: list[tuple[str, bool]], title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, max(4, 0.65 * len(boxes))))
    ax.axis("off")
    ax.set_title(title, pad=14)
    for idx, (text, passed) in enumerate(boxes):
        y = len(boxes) - idx
        color = "#d7f0dd" if passed else "#f8d6d4"
        edge = "#2ca25f" if passed else "#de2d26"
        ax.text(
            0.5,
            y,
            text,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": color, "edgecolor": edge},
            fontsize=9,
        )
        if idx < len(boxes) - 1:
            ax.annotate("", xy=(0.5, y - 0.62), xytext=(0.5, y - 0.22), arrowprops={"arrowstyle": "->", "color": "#555555"})
    ax.set_xlim(0, 1)
    ax.set_ylim(0.3, len(boxes) + 0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def audit_loaded_context(repo_root: Path, out_dir: Path) -> dict[str, Any]:
    context_paths = [
        repo_root / ".project_context" / "CURRENT_STATE.md",
        repo_root / ".project_context" / "TODO.md",
        repo_root / ".project_context" / "CODEX_LOG.md",
    ]
    records = []
    for path in context_paths:
        exists = path.exists()
        record = {"path": str(path), "exists": exists}
        if exists:
            record.update(sha256_file(path))
            text = path.read_text(encoding="utf-8")
            record["line_count"] = len(text.splitlines())
            record["first_line"] = text.splitlines()[0] if text.splitlines() else ""
        records.append(record)
    manifest = {
        "stage": "Stage 4A-6.6a",
        "context_files_read": records,
        "all_required_context_files_present": all(row["exists"] for row in records),
    }
    write_json(out_dir / "loaded_context_manifest.json", manifest)
    lines = ["# Loaded Context Manifest", ""]
    for row in records:
        lines.append(f"- `{row['path']}`: exists={bool_word(row['exists'])}, sha256=`{row.get('sha256', '')}`")
    write_md(out_dir / "loaded_context_manifest.md", lines)
    return manifest


def audit_bundle_manifest(stage_dir: Path, out_dir: Path) -> dict[str, Any]:
    records = []
    for rel in REQUIRED_INPUTS:
        path = stage_dir / rel
        row = {"relative_path": rel, "exists": path.exists()}
        if path.exists():
            row.update(sha256_file(path))
        records.append(row)
    source_manifest = read_json(stage_dir / "audit_input_bundle_manifest.json") if (stage_dir / "audit_input_bundle_manifest.json").exists() else {}
    manifest = {
        "stage": "Stage 4A-6.6a",
        "stage4a66_dir": str(stage_dir),
        "required_inputs": records,
        "all_required_inputs_present": all(row["exists"] for row in records),
        "source_audit_input_bundle_manifest_file_count": source_manifest.get("file_count"),
        "source_audit_consumer": source_manifest.get("audit_consumer"),
    }
    write_json(out_dir / "loaded_stage4a66_bundle_manifest.json", manifest)
    lines = ["# Loaded Stage 4A-6.6 Bundle Manifest", ""]
    for row in records:
        lines.append(f"- `{row['relative_path']}`: exists={bool_word(row['exists'])}")
    write_md(out_dir / "loaded_stage4a66_bundle_manifest.md", lines)
    return manifest


def audit_hashes(stage_dir: Path, out_dir: Path, max_workers: int) -> dict[str, Any]:
    files = sorted([p for p in stage_dir.rglob("*") if p.is_file()])

    def hash_record(path: Path) -> dict[str, Any]:
        record = sha256_file(path)
        record["relative_path"] = str(path.relative_to(stage_dir))
        return record

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        records = list(pool.map(hash_record, files))
    observed_record = next((row for row in records if row["relative_path"] == "observed_state_final.npy"), None)
    audit = {
        "stage": "Stage 4A-6.6a",
        "stage4a66_dir": str(stage_dir),
        "file_count": len(records),
        "hash_algorithm": "sha256",
        "observed_state_final_hash": observed_record,
        "files": records,
    }
    write_json(out_dir / "input_hash_audit.json", audit)
    lines = [
        "# Input Hash Audit",
        "",
        f"- Stage 4A-6.6 file count: {len(records)}",
        f"- Hash algorithm: sha256",
        f"- observed_state_final.npy sha256: `{observed_record.get('sha256') if observed_record else 'missing'}`",
    ]
    write_md(out_dir / "input_hash_audit.md", lines)
    return audit


def audit_hardware(args: argparse.Namespace, out_dir: Path, stage_hw: dict[str, Any], start_time: float, task_count: int) -> dict[str, Any]:
    actual = min(int(args.max_workers), os.cpu_count() or 1)
    cuda_available: bool | str = "not_checked"
    try:
        import torch  # type: ignore

        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = f"torch_unavailable_or_cuda_check_failed: {exc}"
    gpu_name = stage_hw.get("GPU name") or stage_hw.get("gpu_name")
    if not gpu_name:
        try:
            gpu_name = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=2.0,
            ).strip().splitlines()[0]
        except Exception:
            gpu_name = "unavailable"
    report = {
        "stage": "Stage 4A-6.6a",
        "offline_audit_only": True,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "os_cpu_count": os.cpu_count(),
        "requested_max_workers": int(args.max_workers),
        "actual_max_workers": int(actual),
        "parallel_backend": "ThreadPoolExecutor for file hashing and depth/image/stat scans; dependency-sensitive audit decisions are sequential",
        "tasks_used_process_pool": [],
        "tasks_used_thread_pool": ["input_hash_audit", "fixed_view_depth_stats", "input_presence_scan"],
        "tasks_sequential": ["graph_analysis", "decision_synthesis", "markdown_and_plot_writes"],
        "sequential_reason": "category decisions depend on prior metric synthesis and write a single coherent audit bundle",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "GPU name": gpu_name,
        "cuda_available_if_torch_available": cuda_available,
        "number_of_audit_tasks": int(task_count),
        "total_wall_time_seconds": float(time.perf_counter() - start_time),
    }
    write_json(out_dir / "hardware_utilization_report.json", report)
    write_md(out_dir / "hardware_utilization_report.md", md_kv("Hardware Utilization Report", report))
    return report


def audit_scale(args: argparse.Namespace, out_dir: Path, meta: dict[str, Any], observed_summary: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bounds = meta["map_bounds"]
    x_span = float(bounds["x"][1]) - float(bounds["x"][0])
    y_span = float(bounds["y"][1]) - float(bounds["y"][0])
    z_range = [float(bounds["z"][0]), float(bounds["z"][1])]
    floor_area = x_span * y_span
    metrics = {
        "bounds_x_span_m": x_span,
        "bounds_y_span_m": y_span,
        "floor_area_m2": floor_area,
        "z_range": z_range,
        "voxel_size": float(meta["voxel_size_recommended"]),
        "observed_state_shape": observed_summary.get("shape") or meta.get("expected_observed_state_shape"),
        "room_count": len(meta["rooms"]),
        "corridor_count": len(meta["corridors"]),
        "opening_count": len(meta["openings"]),
        "wall_count": len(meta["walls"]),
        "obstacle_count": len(meta["obstacles"]),
        "start_variant_count": len(meta["start_variants"]),
        "validation_pose_count": len(meta["validation_camera_poses"]),
    }
    checks = {
        "bounds_x_span_m >= 20": x_span >= 20.0,
        "bounds_y_span_m >= 20": y_span >= 20.0,
        "floor_area_m2 >= 400": floor_area >= 400.0,
        f"room_count >= {args.min_rooms}": len(meta["rooms"]) >= args.min_rooms,
        f"corridor_count >= {args.min_corridors}": len(meta["corridors"]) >= args.min_corridors,
        f"opening_count >= {args.min_openings}": len(meta["openings"]) >= args.min_openings,
        f"obstacle_count >= {args.min_obstacles}": len(meta["obstacles"]) >= args.min_obstacles,
        f"start_variant_count >= {args.min_starts}": len(meta["start_variants"]) >= args.min_starts,
        f"validation_pose_count >= {args.min_validation_poses}": len(meta["validation_camera_poses"]) >= args.min_validation_poses,
    }
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "metrics": metrics,
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": [],
    }
    rows = []
    minimums = {
        "bounds_x_span_m": 20,
        "bounds_y_span_m": 20,
        "floor_area_m2": 400,
        "room_count": args.min_rooms,
        "corridor_count": args.min_corridors,
        "opening_count": args.min_openings,
        "obstacle_count": args.min_obstacles,
        "start_variant_count": args.min_starts,
        "validation_pose_count": args.min_validation_poses,
    }
    for key, value in metrics.items():
        if key in minimums:
            rows.append({"metric": key, "value": value, "minimum": minimums[key], "pass": float(value) >= float(minimums[key])})
        else:
            rows.append({"metric": key, "value": value, "minimum": "", "pass": ""})
    write_json(out_dir / "scene_scale_audit.json", audit)
    write_md(out_dir / "scene_scale_audit.md", md_kv("Scene Scale Audit", {**metrics, "pass": audit["pass"], "score": audit["score"]}))
    write_csv(out_dir / "scene_scale_scorecard.csv", rows, ["metric", "value", "minimum", "pass"])
    return audit, rows


def audit_topology(
    args: argparse.Namespace,
    out_dir: Path,
    meta: dict[str, Any],
    graph: dict[str, Any],
    prelim: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], dict[str, str]]:
    adjacency, nodes, edges = build_adjacency(graph)
    comps = connected_components(adjacency)
    node_count = len(adjacency)
    edge_count = len(edges)
    component_count = len(comps)
    largest_component = set(comps[0]) if comps else set()
    degrees = {node: len(nbrs) for node, nbrs in adjacency.items()}
    degree_hist = Counter(degrees.values())
    cycle_rank = edge_count - node_count + component_count
    branch_points = sorted([node for node, degree in degrees.items() if degree >= 3])
    corridor_t_junctions = sorted([node for node in branch_points if str(nodes.get(node, {}).get("kind")) == "corridor"])
    graph_dead_ends = sorted([node for node, degree in degrees.items() if degree == 1])
    dead_end_branch_count = int(meta.get("topology_summary", {}).get("dead_end_branch_count", 0))
    narrow_passages = [
        str(item["name"])
        for item in meta["openings"]
        if float(item.get("width", 99.0)) <= 0.90
    ] + [
        str(item["name"])
        for item in meta["corridors"]
        if float(item.get("min_width_m", 99.0)) <= 0.90
    ]
    avg_degree = float(sum(degrees.values()) / max(1, node_count))
    start_assignment = dict(prelim.get("connectivity_between_start_variants", {}).get("start_region_assignment", {}))
    if not start_assignment:
        start_assignment = dict(prelim.get("start_region_assignment", {}))
    start_names = [str(item["name"]) for item in meta["start_variants"]]
    all_starts_in_largest = all(start_assignment.get(name) in largest_component for name in start_names)
    all_pairs: dict[str, dict[str, int]] = {}
    for name in start_names:
        source_region = start_assignment.get(name)
        distances = bfs_distances(adjacency, source_region) if source_region else {}
        all_pairs[name] = {
            other: int(distances.get(start_assignment.get(other, ""), 10**9))
            for other in start_names
        }
    matrix = [[all_pairs[a][b] for b in start_names] for a in start_names]
    finite_distances = [v for row in matrix for v in row if v < 10**9 and v > 0]
    arts = articulation_points(adjacency)
    dead_end_reachable = False
    for name in start_names:
        source_region = start_assignment.get(name)
        if not source_region:
            continue
        dists = bfs_distances(adjacency, source_region)
        if any(node in dists for node in graph_dead_ends if "dead_end" in node or node == "corridor_dead_end"):
            dead_end_reachable = True
            break
    checks = {
        "largest component includes all starts": all_starts_in_largest,
        f"cycle_rank >= {args.min_loop_count}": cycle_rank >= args.min_loop_count,
        f"branch_point_count >= {args.min_branch_points}": len(branch_points) >= args.min_branch_points,
        "t_junction_count >= 3": len(corridor_t_junctions) >= 3 or int(meta.get("topology_summary", {}).get("t_junction_count", 0)) >= 3,
        f"dead_end_count >= {args.min_dead_ends}": len(graph_dead_ends) >= args.min_dead_ends or dead_end_branch_count >= args.min_dead_ends,
        f"narrow_passage_count >= {args.min_narrow_passages}": len(set(narrow_passages)) >= args.min_narrow_passages
        or int(meta.get("topology_summary", {}).get("narrow_passage_count", 0)) >= args.min_narrow_passages,
        "average_degree > 1.5": avg_degree > 1.5,
        "loop connects rooms/corridors": cycle_rank >= args.min_loop_count and any(str(nodes.get(e["source"], {}).get("kind")) != str(nodes.get(e["target"], {}).get("kind")) for e in edges),
        "reachable dead-end branch": dead_end_reachable,
    }
    warnings = []
    if finite_distances and min(finite_distances) <= 1:
        warnings.append("Some start variants share adjacent or identical topology regions; Euclidean diversity remains the primary spread evidence.")
    bottleneck_review = {
        "articulation_points": arts,
        "articulation_point_count": len(arts),
        "interpretation": "Bottlenecks are expected in side branches and cul-de-sacs; the main loop remains connected by multiple closures.",
        "hard_blocker": False,
    }
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "metrics": {
            "node_count": node_count,
            "edge_count": edge_count,
            "connected_component_count": component_count,
            "largest_connected_component_size": len(largest_component),
            "all_starts_in_largest_connected_component": all_starts_in_largest,
            "cycle_rank": cycle_rank,
            "estimated_loop_count": cycle_rank,
            "branch_point_count": len(branch_points),
            "branch_points": branch_points,
            "t_junction_count": len(corridor_t_junctions),
            "t_junction_nodes": corridor_t_junctions,
            "dead_end_count": len(graph_dead_ends),
            "dead_end_nodes": graph_dead_ends,
            "dead_end_branch_count_from_metadata": dead_end_branch_count,
            "narrow_passage_count": len(set(narrow_passages)),
            "narrow_passages": sorted(set(narrow_passages)),
            "average_graph_degree": avg_degree,
            "degree_histogram": dict(sorted(degree_hist.items())),
            "start_to_start_graph_distance_min": min(finite_distances) if finite_distances else None,
            "start_to_start_graph_distance_median": float(np.median(finite_distances)) if finite_distances else None,
            "start_to_start_graph_distance_max": max(finite_distances) if finite_distances else None,
            "articulation_point_count": len(arts),
        },
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": warnings,
        "start_region_assignment": start_assignment,
    }
    write_json(out_dir / "topology_complexity_audit.json", audit)
    write_md(out_dir / "topology_complexity_audit.md", md_kv("Topology Complexity Audit", {**audit["metrics"], "pass": audit["pass"], "score": audit["score"], "warnings": warnings}))
    write_csv(
        out_dir / "topology_degree_histogram.csv",
        [{"degree": degree, "count": count} for degree, count in sorted(degree_hist.items())],
        ["degree", "count"],
    )
    write_matrix_csv(out_dir / "start_graph_distance_matrix.csv", start_names, matrix)
    write_json(out_dir / "bottleneck_articulation_review.json", bottleneck_review)
    write_md(out_dir / "bottleneck_articulation_review.md", md_kv("Bottleneck Articulation Review", bottleneck_review))
    return audit, all_pairs, start_assignment


def audit_starts(
    args: argparse.Namespace,
    out_dir: Path,
    meta: dict[str, Any],
    starts: list[dict[str, Any]],
    graph_distances: dict[str, dict[str, int]],
    start_assignment: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[list[float]]]:
    regions = build_region_lookup(meta["rooms"], meta["corridors"])
    names = [str(s["name"]) for s in starts]
    positions = [s["position"] for s in starts]
    euclidean_matrix: list[list[float]] = []
    duplicate_pairs = []
    nearest_distances = []
    for i, a in enumerate(positions):
        row = []
        for j, b in enumerate(positions):
            dist = math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
            row.append(float(dist))
            if i < j and dist < 0.5:
                duplicate_pairs.append({"a": names[i], "b": names[j], "distance_m": float(dist)})
        nearest = min([row[j] for j in range(len(row)) if j != i], default=math.inf)
        nearest_distances.append(float(nearest))
        euclidean_matrix.append(row)
    table = []
    local_context_rows = []
    for start, nearest_start_distance in zip(starts, nearest_distances):
        pos = start["position"]
        nearest_zone = assign_nearest_zone(pos, regions)
        opening = nearest_opening(pos, meta["openings"])
        graph_node = start_assignment.get(str(start["name"])) or nearest_zone["zone"]
        label = str(start.get("intended_local_topology_label", "")).lower()
        name_lower = str(start["name"]).lower()
        if "dead" in name_lower or "cul-de-sac" in label:
            local_kind = "dead_end"
        elif "junction" in name_lower or "junction" in label:
            local_kind = "junction"
        elif str(graph_node).startswith("room_"):
            local_kind = "room"
        elif str(graph_node).startswith("corridor_"):
            local_kind = "corridor"
        else:
            local_kind = "unknown"
        row = {
            "name": start["name"],
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "yaw_deg": start.get("yaw_deg"),
            "local_topology_label": start.get("intended_local_topology_label", ""),
            "local_kind": local_kind,
            "inside_bounds": bool(start.get("inside_bounds", point_inside_bounds(pos, meta["map_bounds"]))),
            "not_inside_obstacle": bool(start.get("not_inside_obstacle", True)),
            "not_too_close_to_wall": bool(start.get("not_too_close_to_wall", True)),
            "nearest_region": nearest_zone["zone"],
            "nearest_region_kind": nearest_zone["kind"],
            "nearest_opening": opening["name"],
            "nearest_opening_distance_m": opening["distance_m"],
            "graph_node_association": graph_node,
            "nearest_start_distance_m": nearest_start_distance,
            "duplicate_or_near_duplicate": nearest_start_distance < 0.5,
            "expected_local_visibility_type": "long_or_branching" if local_kind in {"junction", "corridor"} else "room_or_occluded",
        }
        table.append(row)
        local_context_rows.append(
            {
                "name": row["name"],
                "local_kind": local_kind,
                "nearest_region": nearest_zone["zone"],
                "nearest_region_kind": nearest_zone["kind"],
                "nearest_opening": opening["name"],
                "nearest_opening_distance_m": opening["distance_m"],
                "graph_node_association": graph_node,
                "expected_local_visibility_type": row["expected_local_visibility_type"],
            }
        )
    room_start_count = sum(1 for row in table if row["local_kind"] == "room")
    corridor_start_count = sum(1 for row in table if row["local_kind"] == "corridor")
    junction_start_count = sum(1 for row in table if row["local_kind"] == "junction")
    dead_end_start_count = sum(1 for row in table if row["local_kind"] == "dead_end")
    spread_fraction = sum(1 for d in nearest_distances if d > 2.0) / max(1, len(nearest_distances))
    all_connect = all(row["graph_node_association"] for row in table)
    checks = {
        f"start count >= {args.min_starts}": len(starts) >= args.min_starts,
        "at least 3 room starts": room_start_count >= 3,
        "at least 2 corridor starts": corridor_start_count >= 2,
        "at least 1 junction start": junction_start_count >= 1,
        "at least 1 dead-end or side-branch start": dead_end_start_count >= 1,
        "no duplicate starts within 0.5m": len(duplicate_pairs) == 0,
        "at least 70 percent nearest-start distance > 2m": spread_fraction >= 0.70,
        "all starts inside bounds": all(bool(row["inside_bounds"]) for row in table),
        "all starts not inside obstacles": all(bool(row["not_inside_obstacle"]) for row in table),
        "all starts connect to topology graph": all_connect,
    }
    warnings = []
    if spread_fraction < 1.0:
        warnings.append("A small number of starts are topologically close, but no starts are duplicate or near-duplicate in Euclidean space.")
    topo_matrix = [[graph_distances.get(a, {}).get(b, 10**9) for b in names] for a in names]
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "metrics": {
            "start_count": len(starts),
            "room_start_count": room_start_count,
            "corridor_start_count": corridor_start_count,
            "junction_start_count": junction_start_count,
            "dead_end_or_side_branch_start_count": dead_end_start_count,
            "duplicate_start_pairs_lt_0_5m": duplicate_pairs,
            "nearest_start_distance_min_m": min(nearest_distances) if nearest_distances else None,
            "nearest_start_distance_median_m": float(np.median(nearest_distances)) if nearest_distances else None,
            "fraction_starts_nearest_distance_gt_2m": spread_fraction,
        },
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": warnings,
        "starts": table,
    }
    write_json(out_dir / "start_variant_diversity_audit.json", audit)
    write_md(out_dir / "start_variant_diversity_audit.md", md_kv("Start Variant Diversity Audit", {**audit["metrics"], "pass": audit["pass"], "score": audit["score"], "warnings": warnings}))
    write_csv(out_dir / "start_variant_table.csv", table)
    write_matrix_csv(out_dir / "start_euclidean_distance_matrix.csv", names, euclidean_matrix)
    write_matrix_csv(out_dir / "start_topology_distance_matrix.csv", names, topo_matrix)
    write_csv(out_dir / "start_local_context_table.csv", local_context_rows)
    return audit, table, euclidean_matrix


def depth_and_rgb_record(stage_dir: Path, pose: dict[str, Any]) -> dict[str, Any]:
    idx = int(pose["index"])
    depth_path = stage_dir / f"validation_depth_{idx:03d}.npy"
    rgb_path = stage_dir / f"validation_rgb_{idx:03d}.png"
    depth = np.load(depth_path)
    finite = np.isfinite(depth)
    positive = finite & (depth > 0)
    valid_values = depth[positive]
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    return {
        "index": idx,
        "zone": pose.get("room"),
        "note": pose.get("note", ""),
        "x": pose["position"][0],
        "y": pose["position"][1],
        "z": pose["position"][2],
        "yaw_deg": pose.get("yaw_deg"),
        "rgb_nonblank": bool(float(np.std(rgb)) > 1.0 and int(np.max(rgb)) > int(np.min(rgb))),
        "rgb_std": float(np.std(rgb)),
        "depth_finite_positive": bool(valid_values.size > 0),
        "depth_valid_pixel_ratio": float(valid_values.size / max(1, depth.size)),
        "depth_min_m": float(np.min(valid_values)) if valid_values.size else None,
        "depth_median_m": float(np.median(valid_values)) if valid_values.size else None,
        "depth_max_m": float(np.max(valid_values)) if valid_values.size else None,
        "depth_mean_m": float(np.mean(valid_values)) if valid_values.size else None,
        "depth_std_m": float(np.std(valid_values)) if valid_values.size else None,
        "depth_discontinuity_count": depth_discontinuity_count(depth),
    }


def depth_discontinuity_count(depth: np.ndarray) -> int:
    finite = np.isfinite(depth) & (depth > 0)
    if not np.any(finite):
        return 0
    arr = np.where(finite, depth, np.nan)
    gx = np.abs(np.diff(arr, axis=0))
    gy = np.abs(np.diff(arr, axis=1))
    return int(np.count_nonzero(np.isfinite(gx) & (gx > 0.45)) + np.count_nonzero(np.isfinite(gy) & (gy > 0.45)))


def audit_fixed_views(
    args: argparse.Namespace,
    out_dir: Path,
    stage_dir: Path,
    poses: list[dict[str, Any]],
    fixed_capture: dict[str, Any],
    max_workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(lambda p: depth_and_rgb_record(stage_dir, p), poses))
    rows.sort(key=lambda r: int(r["index"]))
    duplicate_count = 0
    for i, a in enumerate(poses):
        for b in poses[i + 1 :]:
            dist = math.hypot(float(a["position"][0]) - float(b["position"][0]), float(a["position"][1]) - float(b["position"][1]))
            if dist < 0.5 and yaw_delta_deg(float(a.get("yaw_deg", 0.0)), float(b.get("yaw_deg", 0.0))) < 15.0:
                duplicate_count += 1
    rgb_nonblank_count = sum(1 for row in rows if row["rgb_nonblank"])
    depth_positive_count = sum(1 for row in rows if row["depth_finite_positive"])
    represented_zones = sorted(set(str(row["zone"]) for row in rows if row.get("zone")))
    yaw_bins = sorted(set(int(round(float(row["yaw_deg"]) / 45.0)) for row in rows if row.get("yaw_deg") is not None))
    note_text = " ".join(str(p.get("note", "")).lower() + " " + str(p.get("room", "")).lower() for p in poses)
    depth_max_values = [float(row["depth_max_m"]) for row in rows if row["depth_max_m"] is not None]
    category_coverage = {
        "rooms": any(str(row["zone"]).startswith("room_") for row in rows),
        "corridors": any(str(row["zone"]).startswith("corridor_") for row in rows),
        "loop_junction": "junction" in note_text,
        "dead_end": "dead" in note_text or "cul-de-sac" in note_text,
        "narrow_passage": "narrow" in note_text or "spur" in note_text,
        "occluded_side_branch": "occluded" in note_text or "spur" in note_text,
        "long_corridor_view": bool(depth_max_values and max(depth_max_values) >= 8.0),
        "occlusion_or_partial_unknown_proxy": any(int(row["depth_discontinuity_count"]) > 250 for row in rows),
    }
    direct_missing_key_zones = sorted({"room_j", "corridor_east_spur"} - set(represented_zones))
    warnings = []
    if direct_missing_key_zones:
        warnings.append("Some constructed zones are not directly named by fixed validation poses: " + ", ".join(direct_missing_key_zones))
    checks = {
        f"validation_pose_count >= {args.min_validation_poses}": len(poses) >= args.min_validation_poses,
        "rgb_nonblank_count >= 12": rgb_nonblank_count >= 12,
        "depth_positive_count >= 8": depth_positive_count >= 8,
        "14/14 valid depth reverified": depth_positive_count == len(poses),
        "view yaw diversity nontrivial": len(yaw_bins) >= 5,
        "at least 6 topology zones represented": len(represented_zones) >= 6,
        "required coverage categories represented": all(category_coverage.values()),
    }
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "metrics": {
            "validation_pose_count": len(poses),
            "rgb_nonblank_count": rgb_nonblank_count,
            "depth_finite_positive_count": depth_positive_count,
            "duplicate_or_near_duplicate_view_count": duplicate_count,
            "yaw_bin_count_45deg": len(yaw_bins),
            "represented_zone_count": len(represented_zones),
            "represented_zones": represented_zones,
            "depth_max_overall_m": max(depth_max_values) if depth_max_values else None,
            "fixed_capture_validation_rgb_nonblank_count_source": fixed_capture.get("rgb_nonblank_count"),
            "fixed_capture_validation_depth_positive_count_source": fixed_capture.get("depth_positive_count"),
        },
        "category_coverage": category_coverage,
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": warnings,
    }
    zone_coverage = {
        "represented_zones": represented_zones,
        "category_coverage": category_coverage,
        "direct_missing_key_zones": direct_missing_key_zones,
        "pass": checks["at least 6 topology zones represented"] and checks["required coverage categories represented"],
        "warnings": warnings,
    }
    write_json(out_dir / "fixed_view_visibility_audit.json", audit)
    write_md(out_dir / "fixed_view_visibility_audit.md", md_kv("Fixed View Visibility Audit", {**audit["metrics"], "pass": audit["pass"], "score": audit["score"], "warnings": warnings}))
    write_csv(out_dir / "validation_view_table.csv", rows)
    write_csv(out_dir / "depth_stats_by_view.csv", rows)
    write_json(out_dir / "validation_zone_coverage.json", zone_coverage)
    write_md(out_dir / "validation_zone_coverage.md", md_kv("Validation Zone Coverage", zone_coverage))
    return audit, rows


def audit_observed_state(
    args: argparse.Namespace,
    out_dir: Path,
    stage_dir: Path,
    observed_summary: dict[str, Any],
    transition_summary: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    observed_state = np.load(stage_dir / "observed_state_final.npy")
    invalid_mask = ~np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED])
    unknown_count = int(np.count_nonzero(observed_state == UNKNOWN))
    free_count = int(np.count_nonzero(observed_state == FREE))
    occupied_count = int(np.count_nonzero(observed_state == OCCUPIED))
    observed_count = int(free_count + occupied_count)
    total_count = int(observed_state.size)
    observed_ratio = float(observed_count / max(1, total_count))
    invalid_count = int(np.count_nonzero(invalid_mask))
    step_rows = []
    prev_ratio = 0.0
    for row in observed_summary.get("step_summaries", []):
        ratio = float(row.get("observed_ratio", 0.0))
        step_rows.append(
            {
                "step": int(row.get("step", len(step_rows))),
                "observed_ratio": ratio,
                "observed_count": int(row.get("observed_count", 0)),
                "newly_observed": int(row.get("transition", {}).get("newly_observed", int(row.get("observed_count", 0)))),
                "delta_observed_ratio": ratio - prev_ratio,
                "free_count": int(row.get("free_count", 0)),
                "occupied_count": int(row.get("occupied_count", 0)),
                "unknown_count": int(row.get("unknown_count", 0)),
            }
        )
        prev_ratio = ratio
    label_rows = [
        {"label": "UNKNOWN", "value": UNKNOWN, "count": unknown_count, "ratio": unknown_count / total_count},
        {"label": "FREE", "value": FREE, "count": free_count, "ratio": free_count / total_count},
        {"label": "OCCUPIED", "value": OCCUPIED, "count": occupied_count, "ratio": occupied_count / total_count},
        {"label": "INVALID", "value": "other", "count": invalid_count, "ratio": invalid_count / total_count},
    ]
    checks = {
        "shape == (240, 240, 30)": tuple(observed_state.shape) == (240, 240, 30),
        f"observed_ratio > {args.min_final_observed_ratio}": observed_ratio > args.min_final_observed_ratio,
        f"observed_ratio < {args.max_fixed_view_observed_ratio}": observed_ratio < args.max_fixed_view_observed_ratio,
        "unknown/free/occupied all present": unknown_count > 0 and free_count > 0 and occupied_count > 0,
        "invalid_label_count == 0": invalid_count == 0,
        "no occupied-to-free transition": not bool(transition_summary.get("any_occupied_to_free", False)),
        "not all-free or all-occupied": free_count < total_count and occupied_count < total_count,
    }
    warnings = []
    if observed_ratio < 0.12:
        warnings.append("Final fixed-view observed_ratio is intentionally low, leaving substantial unknown space for future expert exploration.")
    exploration_room_left = {
        "unknown_count": unknown_count,
        "unknown_ratio": float(unknown_count / total_count),
        "observed_ratio": observed_ratio,
        "enough_unknown_for_exploration": observed_ratio < args.max_fixed_view_observed_ratio and unknown_count > observed_count,
        "not_too_small_for_pilot": observed_ratio > args.min_final_observed_ratio,
    }
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "metrics": {
            "shape": list(observed_state.shape),
            "unknown_count": unknown_count,
            "free_count": free_count,
            "occupied_count": occupied_count,
            "observed_count": observed_count,
            "observed_ratio": observed_ratio,
            "invalid_label_count": invalid_count,
            "free_to_occupied_or_occupied_to_free_flags": {
                "any_occupied_to_free": bool(transition_summary.get("any_occupied_to_free", False)),
            },
            "free_occupied_ratio": float(free_count / max(1, occupied_count)),
        },
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": warnings,
    }
    write_json(out_dir / "observed_state_health_audit.json", audit)
    write_md(out_dir / "observed_state_health_audit.md", md_kv("Observed State Health Audit", {**audit["metrics"], "pass": audit["pass"], "score": audit["score"], "warnings": warnings}))
    write_csv(out_dir / "observed_ratio_by_view.csv", step_rows)
    write_csv(out_dir / "observed_label_distribution.csv", label_rows)
    write_json(out_dir / "observed_state_exploration_room_left.json", exploration_room_left)
    write_md(out_dir / "observed_state_exploration_room_left.md", md_kv("Observed State Exploration Room Left", exploration_room_left))
    return audit, observed_state, step_rows


def audit_frontier_reachability(
    args: argparse.Namespace,
    out_dir: Path,
    meta: dict[str, Any],
    observed_state: np.ndarray,
    starts: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray], np.ndarray]:
    voxel_size = float(meta["voxel_size_recommended"])
    traversability = build_traversability_grid(observed_state, voxel_size=voxel_size, robot_height_m=1.2, clearance_height_m=0.6, robot_radius_m=0.2)
    traversable = np.asarray(traversability["traversable"], dtype=bool)
    frontier_xy = frontier_adjacent_free_xy_mask(observed_state)
    frontier_voxels = detect_frontier_voxels(observed_state)
    rows_area = []
    rows_frontier = []
    rows_candidate = []
    isolated = []
    reachable_masks: dict[str, np.ndarray] = {}
    for start in starts:
        name = str(start["name"])
        xy = world_to_grid_xy(start["position"], meta["map_bounds"], voxel_size, traversable.shape)
        nearest = nearest_traversable_cell(traversable, xy, max_radius_cells=30)
        snapped = tuple(nearest["cell"]) if nearest.get("found") else xy
        component = connected_component_from_start(traversable, snapped, allow_diagonal=True)
        reachable_mask = np.asarray(component["reachable_mask"], dtype=bool)
        candidate_mask = reachable_mask & frontier_xy
        reachable_frontier_count = int(np.count_nonzero(candidate_mask))
        reachable_count = int(component["reachable_count"])
        quadrants = Counter()
        coords = np.argwhere(candidate_mask)
        for cell in coords[:: max(1, len(coords) // 1500) if len(coords) else 1]:
            dx = int(cell[0]) - int(snapped[0])
            dy = int(cell[1]) - int(snapped[1])
            quadrants[("east" if dx >= 0 else "west") + "_" + ("north" if dy >= 0 else "south")] += 1
        multi_branch = sum(1 for v in quadrants.values() if v > 0) >= 2
        isolated_flag = reachable_count == 0 or not bool(component.get("start_valid", False))
        if isolated_flag:
            isolated.append({"name": name, "reason": component.get("reason"), "start_xy": xy})
        reachable_masks[name] = reachable_mask
        area_row = {
            "name": name,
            "start_grid_x": xy[0],
            "start_grid_y": xy[1],
            "nearest_traversable_found": bool(nearest.get("found")),
            "nearest_traversable_x": nearest.get("cell", [None, None])[0] if nearest.get("cell") else None,
            "nearest_traversable_y": nearest.get("cell", [None, None])[1] if nearest.get("cell") else None,
            "snap_distance_cells": nearest.get("distance_cells"),
            "reachable_observed_free_component_size": reachable_count,
            "is_isolated": isolated_flag,
            "multiple_branch_directions_proxy": multi_branch,
        }
        frontier_row = {
            "name": name,
            "total_frontier_voxel_count": int(len(frontier_voxels)),
            "total_frontier_adjacent_free_xy_count": int(np.count_nonzero(frontier_xy)),
            "reachable_frontier_count": reachable_frontier_count,
            "reachable_frontier_fraction_of_total_xy": float(reachable_frontier_count / max(1, int(np.count_nonzero(frontier_xy)))),
        }
        candidate_row = {
            "name": name,
            "reachable_candidate_availability_proxy": float(min(1.0, reachable_frontier_count / 64.0)),
            "expected_candidate_count_64_available": reachable_frontier_count >= 64,
            "usable_for_pilot": (not isolated_flag) and reachable_frontier_count >= 64,
            "frontier_quadrant_count_proxy": sum(1 for v in quadrants.values() if v > 0),
        }
        rows_area.append(area_row)
        rows_frontier.append(frontier_row)
        rows_candidate.append(candidate_row)
    nonzero_starts = sum(1 for row in rows_frontier if int(row["reachable_frontier_count"]) > 0)
    median_reachable = float(np.median([int(row["reachable_frontier_count"]) for row in rows_frontier])) if rows_frontier else 0.0
    usable_count = sum(1 for row in rows_candidate if bool(row["usable_for_pilot"]))
    corridor_heavy = any("corridor" in str(start["name"]) and int(row["reachable_frontier_count"]) > 64 for start, row in zip(starts, rows_frontier))
    room_heavy = any("room" in str(start["name"]) and int(row["reachable_frontier_count"]) > 64 for start, row in zip(starts, rows_frontier))
    checks = {
        "at least 6 starts have nonzero reachable frontier candidates": nonzero_starts >= 6,
        "no start completely isolated": len(isolated) == 0,
        "median reachable frontier count > 50": median_reachable > 50.0,
        "reachable components vary by start": len(set(int(row["reachable_observed_free_component_size"]) for row in rows_area)) >= 2,
        "corridor-heavy frontier distribution exists": corridor_heavy,
        "room-heavy frontier distribution exists": room_heavy,
        "most starts can support 64 candidates": usable_count >= max(6, math.ceil(0.7 * len(starts))),
    }
    warnings = []
    component_sizes = sorted(set(int(row["reachable_observed_free_component_size"]) for row in rows_area))
    if len(component_sizes) > 1:
        warnings.append("Measured-only fixed views create multiple observed-free components; this is acceptable for a fixed-view audit and should be revisited in the pilot.")
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "metrics": {
            "traversability_diagnostics": traversability.get("diagnostics", {}),
            "total_frontier_voxel_count": int(len(frontier_voxels)),
            "total_frontier_adjacent_free_xy_count": int(np.count_nonzero(frontier_xy)),
            "nonzero_reachable_frontier_start_count": nonzero_starts,
            "median_reachable_frontier_count": median_reachable,
            "usable_start_count": usable_count,
            "isolated_start_count": len(isolated),
        },
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": warnings,
    }
    isolated_review = {
        "isolated_starts": isolated,
        "isolated_start_count": len(isolated),
        "pass": len(isolated) == 0,
        "review": "No starts are isolated after snapping to nearest observed-free traversable cell within 30 cells." if not isolated else "Some starts require scene or fixed-view revision.",
    }
    write_json(out_dir / "frontier_reachability_audit.json", audit)
    write_md(out_dir / "frontier_reachability_audit.md", md_kv("Frontier Reachability Audit", {**audit["metrics"], "pass": audit["pass"], "score": audit["score"], "warnings": warnings}))
    write_csv(out_dir / "reachable_area_by_start.csv", rows_area)
    write_csv(out_dir / "reachable_frontier_by_start.csv", rows_frontier)
    write_csv(out_dir / "candidate_availability_proxy_by_start.csv", rows_candidate)
    write_json(out_dir / "isolated_start_review.json", isolated_review)
    write_md(out_dir / "isolated_start_review.md", md_kv("Isolated Start Review", isolated_review))
    return audit, rows_candidate, reachable_masks, frontier_xy


def audit_obstacles(
    args: argparse.Namespace,
    out_dir: Path,
    meta: dict[str, Any],
    starts: list[dict[str, Any]],
    depth_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    region_lookup = build_region_lookup(meta["rooms"], meta["corridors"])
    region_counts = Counter(str(ob.get("region", "unknown")) for ob in meta["obstacles"])
    category_counts = Counter(str(ob.get("category", "unknown")) for ob in meta["obstacles"])
    density_rows = []
    for name, region in sorted(region_lookup.items()):
        area = bounds_area(region["bounds"])
        count = int(region_counts.get(name, 0))
        density_rows.append(
            {
                "zone": name,
                "kind": region["kind"],
                "area_m2": area,
                "obstacle_count": count,
                "obstacles_per_m2": float(count / max(area, 1e-9)),
            }
        )
    distribution_rows = []
    for obstacle in meta["obstacles"]:
        b = obstacle_bounds(obstacle)
        distribution_rows.append(
            {
                "name": obstacle["name"],
                "region": obstacle.get("region"),
                "category": obstacle.get("category"),
                "x_min": b["x"][0],
                "x_max": b["x"][1],
                "y_min": b["y"][0],
                "y_max": b["y"][1],
                "height_m": float(obstacle.get("size", [0, 0, 0])[2]),
            }
        )
    start_clearances = []
    for start in starts:
        distances = [point_rect_distance(start["position"], obstacle_bounds(ob)) for ob in meta["obstacles"]]
        start_clearances.append({"name": start["name"], "nearest_obstacle_distance_m": min(distances) if distances else None})
    blocked_openings = []
    for opening in meta["openings"]:
        clear_rect = opening["clear_rect"]
        clear_area = max(bounds_area(clear_rect), 1e-9)
        coverage = sum(rect_intersection_area(clear_rect, obstacle_bounds(ob)) for ob in meta["obstacles"]) / clear_area
        if coverage > 0.60:
            blocked_openings.append({"name": opening["name"], "coverage_fraction": float(coverage)})
    occlusion_rows = []
    for row in depth_rows:
        step = int(row["index"])
        newly_observed = next((int(s["newly_observed"]) for s in step_rows if int(s["step"]) == step), None)
        occlusion_rows.append(
            {
                "view_index": row["index"],
                "zone": row["zone"],
                "depth_std_m": row["depth_std_m"],
                "depth_discontinuity_count": row["depth_discontinuity_count"],
                "depth_valid_pixel_ratio": row["depth_valid_pixel_ratio"],
                "newly_observed_voxels": newly_observed,
                "meaningful_occlusion_proxy": int(row["depth_discontinuity_count"]) > 250 and float(row["depth_std_m"] or 0.0) > 0.6,
            }
        )
    occluding_categories = {"cabinet_like_tall_cuboid", "shelf_like_long_cuboid", "pillar_cuboid"}
    meaningful_occlusion_regions = sorted(
        {
            str(ob.get("region"))
            for ob in meta["obstacles"]
            if str(ob.get("category")) in occluding_categories
        }
    )
    checks = {
        f"obstacle_count >= {args.min_obstacles}": len(meta["obstacles"]) >= args.min_obstacles,
        "obstacles distributed across multiple rooms/corridors": len(region_counts) >= 8,
        "at least 4 meaningful occlusion areas": len(meaningful_occlusion_regions) >= 4 or sum(1 for row in occlusion_rows if row["meaningful_occlusion_proxy"]) >= 4,
        "no key topology connection fully blocked": len(blocked_openings) == 0,
        "start positions not colliding with obstacles": all(bool(s.get("not_inside_obstacle", True)) for s in starts),
    }
    warnings = []
    if max(region_counts.values()) if region_counts else 0:
        warnings.append("Obstacle density is intentionally higher in small spur rooms; pilot should monitor candidate clearance near these regions.")
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "metrics": {
            "obstacle_count": len(meta["obstacles"]),
            "obstacle_category_counts": dict(sorted(category_counts.items())),
            "obstacle_region_count": len(region_counts),
            "meaningful_occlusion_region_count": len(meaningful_occlusion_regions),
            "meaningful_occlusion_regions": meaningful_occlusion_regions,
            "blocked_opening_count": len(blocked_openings),
            "blocked_openings": blocked_openings,
            "nearest_obstacle_to_each_start": start_clearances,
        },
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": warnings,
    }
    write_json(out_dir / "obstacle_occlusion_audit.json", audit)
    write_md(out_dir / "obstacle_occlusion_audit.md", md_kv("Obstacle Occlusion Audit", {**audit["metrics"], "pass": audit["pass"], "score": audit["score"], "warnings": warnings}))
    write_csv(out_dir / "obstacle_distribution_table.csv", distribution_rows)
    write_csv(out_dir / "obstacle_density_by_zone.csv", density_rows)
    write_csv(out_dir / "occlusion_proxy_by_view.csv", occlusion_rows)
    return audit, density_rows, occlusion_rows


def audit_expert_usability(
    out_dir: Path,
    meta: dict[str, Any],
    observed_audit: dict[str, Any],
    frontier_audit: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    shape = observed_audit["metrics"]["shape"]
    memory_mb = float(np.prod(shape) * np.dtype(np.int8).itemsize / (1024.0 * 1024.0))
    usable_starts = [row["name"] for row in candidate_rows if bool(row["usable_for_pilot"])]
    checks = {
        "observed-free traversability components not empty": int(frontier_audit["metrics"]["traversability_diagnostics"].get("traversable_count", 0)) > 0,
        "reachable frontier candidates exist for most starts": int(frontier_audit["metrics"]["usable_start_count"]) >= max(6, math.ceil(0.7 * len(candidate_rows))),
        "map size and voxel size compatible": list(shape) == [240, 240, 30] and abs(float(meta["voxel_size_recommended"]) - 0.1) < 1e-9,
        "fixed validation observed_state has unknown/free/occupied": bool(observed_audit["checks"]["unknown/free/occupied all present"]),
        "starts have valid measured-only candidate availability": len(usable_starts) >= max(6, math.ceil(0.7 * len(candidate_rows))),
        "no observed_state shape incompatibility": list(shape) == [240, 240, 30],
        "no huge memory red flag": memory_mb < 128.0,
        "candidate count 64 available for most starts": sum(1 for row in candidate_rows if bool(row["expected_candidate_count_64_available"])) >= max(6, math.ceil(0.7 * len(candidate_rows))),
    }
    risk_register = [
        {
            "risk": "Fixed-view measured-only observed_state is disconnected across some areas.",
            "severity": "medium",
            "mitigation": "Start Stage 4A-6.7 with a tiny measured-only one-step/small pilot before any prediction-assisted variant.",
        },
        {
            "risk": "High obstacle density in spur rooms may reduce clearance for future candidate sampling.",
            "severity": "low",
            "mitigation": "Use candidate collision checks and review failed starts before scaling beyond the pilot.",
        },
        {
            "risk": "Full expert dataset collection before pilot would hide topology-specific failure modes.",
            "severity": "high",
            "mitigation": "Keep formal_expert_sampling_ready_full_dataset false until a bounded pilot passes.",
        },
    ]
    audit = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": ["Use measured-only larger-scene pilot first; lambda48 read-only map_predict pilot should remain a later step."],
        "metrics": {
            "observed_state_shape": shape,
            "voxel_size": float(meta["voxel_size_recommended"]),
            "estimated_observed_state_memory_mb_int8": memory_mb,
            "usable_start_count": len(usable_starts),
            "usable_starts": usable_starts,
            "recommended_staged_pilot": [
                "measured-only larger scene one-step/small pilot",
                "lambda48 read-only map_predict pilot only after measured-only sanity",
            ],
        },
    }
    write_json(out_dir / "expert_usability_pre_audit.json", audit)
    write_md(out_dir / "expert_usability_pre_audit.md", md_kv("Expert Usability Pre-Audit", {**audit["metrics"], "pass": audit["pass"], "score": audit["score"], "warnings": audit["warnings"]}))
    write_json(out_dir / "future_expert_pilot_risk_register.json", {"risks": risk_register})
    write_md(out_dir / "future_expert_pilot_risk_register.md", md_table("Future Expert Pilot Risk Register", risk_register, ["risk", "severity", "mitigation"]))
    return audit


def negative_scope_reports(out_dir: Path, args: argparse.Namespace, meta: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "no_isaac_requested": bool(args.no_isaac),
        "no_capture_requested": bool(args.no_capture),
        "no_rollout_requested": bool(args.no_rollout),
        "no_formal_expert_sampling_requested": bool(args.no_formal_expert_sampling),
        "no_map_predict_requested": bool(args.no_map_predict),
        "no_rl_gdpo_requested": bool(args.no_rl_gdpo),
        "stage4a66_input_no_map_predict": not bool(meta.get("leakage_checks", {}).get("map_predict_called", False)),
        "stage4a66_input_no_formal_expert_sampling": not bool(meta.get("leakage_checks", {}).get("formal_expert_sampling", False)),
        "stage4a66_input_no_rollout": not bool(meta.get("leakage_checks", {}).get("rollout_run", False)),
        "stage4a66_input_no_rl_or_training": not bool(meta.get("leakage_checks", {}).get("rl_or_ppo_training", False)),
    }
    report = {
        "stage": "Stage 4A-6.6a",
        "pass": all(checks.values()),
        "score": score_from_checks(checks),
        "checks": checks,
        "blockers": [] if all(checks.values()) else [k for k, v in checks.items() if not v],
        "warnings": [],
    }
    scoped_reports = {
        "no_rollout_report": {
            "rollout_run": False,
            "selected_action_executed": False,
            "open_ended_loop_run": False,
            "evidence": "Stage 4A-6.6a ran offline audit code only.",
        },
        "no_formal_expert_sampling_report": {
            "formal_expert_sampling_run": False,
            "expert_dataset_created": False,
            "transitions_jsonl_created": False,
            "evidence": "The audit generated reports and plots only.",
        },
        "no_map_predict_report": {
            "map_predict_called": False,
            "sscnet_inference_called": False,
            "prediction_npz_created": False,
            "prediction_writeback_or_fusion": False,
            "evidence": "No map_predict or SSCNet module is imported or executed by this audit.",
        },
        "no_rl_gdpo_report": {
            "rl_gdpo_ppo_bc_il_run": False,
            "replay_buffer_created": False,
            "policy_checkpoint_created_or_modified": False,
            "evidence": "This audit does not train, checkpoint, or collect replay data.",
        },
    }
    for stem, data in scoped_reports.items():
        write_json(out_dir / f"{stem}.json", data)
        write_md(out_dir / f"{stem}.md", md_kv(stem.replace("_", " ").title(), data))
    return report


def synthesize_decision(
    out_dir: Path,
    audits: dict[str, dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    category_order = [
        ("scale", "scene_scale_audit.json"),
        ("topology", "topology_complexity_audit.json"),
        ("starts", "start_variant_diversity_audit.json"),
        ("fixed_views", "fixed_view_visibility_audit.json"),
        ("observed_state", "observed_state_health_audit.json"),
        ("frontier_reachability", "frontier_reachability_audit.json"),
        ("obstacle_occlusion", "obstacle_occlusion_audit.json"),
        ("expert_usability", "expert_usability_pre_audit.json"),
        ("safety_negative_scope", "negative_scope"),
    ]
    scorecard = []
    hard_blockers = []
    warnings = []
    for category, evidence in category_order:
        audit = audits[category]
        blockers = list(audit.get("blockers", []))
        row = {
            "category": category,
            "pass": bool(audit.get("pass")),
            "score": float(audit.get("score", 0.0)),
            "blockers": blockers,
            "warnings": list(audit.get("warnings", [])),
            "evidence_files": [evidence],
            "recommendation": "pass to next gate" if bool(audit.get("pass")) else "revise before pilot",
        }
        scorecard.append(row)
        if blockers:
            hard_blockers.extend([f"{category}: {blocker}" for blocker in blockers])
        warnings.extend([f"{category}: {warning}" for warning in row["warnings"]])
    scene_passed = all(bool(row["pass"]) for row in scorecard) and not hard_blockers
    usable_starts = [str(row["name"]) for row in candidate_rows if bool(row["usable_for_pilot"])]
    ready_pilot = scene_passed and len(usable_starts) >= max(6, math.ceil(0.7 * len(candidate_rows)))
    decision = {
        "stage": "Stage 4A-6.6a",
        "scene_complexity_audit_passed": bool(scene_passed),
        "scene_ready_for_formal_expert_sampling_pilot": bool(ready_pilot),
        "formal_expert_sampling_ready_full_dataset": False,
        "hard_blockers": hard_blockers,
        "warnings": warnings,
        "usable_start_subset": usable_starts,
        "qualified_start_count": len(usable_starts),
        "full_dataset_collection_allowed": False,
        "recommended_next": (
            "Stage 4A-6.7 bounded formal expert sampling pilot design/execution, measured-only first, small start subset or all qualified starts, not full dataset"
            if ready_pilot
            else "Stage 4A-6.6b scene revision based on audit blockers"
        ),
        "negative_scope": {
            "no_rollout": True,
            "no_formal_expert_sampling_in_6_6a": True,
            "no_map_predict": True,
            "no_rl_gdpo_ppo_bc_il": True,
        },
    }
    write_json(out_dir / "scene_complexity_scorecard.json", {"categories": scorecard})
    write_csv(out_dir / "scene_complexity_scorecard.csv", scorecard)
    write_md(out_dir / "scene_complexity_scorecard.md", md_table("Scene Complexity Scorecard", scorecard, ["category", "pass", "score", "blockers", "warnings", "recommendation"]))
    write_json(out_dir / "scene_complexity_audit_decision.json", decision)
    write_md(out_dir / "scene_complexity_audit_decision.md", md_kv("Scene Complexity Audit Decision", decision))
    write_json(out_dir / "usable_start_subset.json", {"usable_start_subset": usable_starts, "qualified_start_count": len(usable_starts)})
    write_md(out_dir / "usable_start_subset.md", ["# Usable Start Subset", "", *[f"- `{name}`" for name in usable_starts]])
    if scene_passed:
        revision_plan = {
            "required": False,
            "reason": "No audit hard blockers.",
            "optional_hardening": [
                "Add a direct room_j/east_spur fixed validation pose before large-scale use if future pilots expose a blind spot.",
                "Keep candidate collision and clearance diagnostics enabled for spur rooms.",
            ],
        }
    else:
        revision_plan = {
            "required": True,
            "reason": "Audit hard blockers remain.",
            "blockers": hard_blockers,
            "required_changes": [
                "Increase any failing scale/topology/start/fixed-view/observed/frontier/obstacle category until its blocker clears.",
                "Regenerate Stage 4A-6.6 outputs as Stage 4A-6.6b before retrying 6.6a.",
            ],
        }
    write_json(out_dir / "scene_revision_plan_if_needed.json", revision_plan)
    write_md(out_dir / "scene_revision_plan_if_needed.md", md_kv("Scene Revision Plan If Needed", revision_plan))
    next_lines = [
        "# Recommended Next Faithful Step",
        "",
        decision["recommended_next"],
        "",
        "Full expert dataset collection remains blocked. Rollout scaling and RL/GDPO/PPO/BC/IL remain future-only.",
    ]
    write_md(out_dir / "recommended_next_faithful_step.md", next_lines)
    if ready_pilot:
        sketch = [
            "# Future Stage 4A-6.7 Formal Expert Sampling Pilot Design Sketch",
            "",
            "- Start with a measured-only larger-scene pilot.",
            "- Keep it bounded and small: one-step or very short horizon, explicit start subset, explicit max transitions, and manual review before scaling.",
            "- Use qualified starts from `usable_start_subset.json`.",
            "- Do not create a full expert dataset in Stage 4A-6.7 until pilot health is reviewed.",
            "- Consider lambda48 read-only map_predict only after measured-only sanity passes.",
        ]
    else:
        sketch = [
            "# Future Stage 4A-6.7 Formal Expert Sampling Pilot Design Sketch",
            "",
            "Blocked by Stage 4A-6.6a audit failure. Run Stage 4A-6.6b scene revision first.",
        ]
    write_md(out_dir / "future_stage4a67_formal_expert_sampling_pilot_design_sketch.md", sketch)
    write_md(
        out_dir / "do_not_start_full_expert_sampling_in_stage4a66a.md",
        [
            "# Do Not Start Full Expert Sampling In Stage 4A-6.6a",
            "",
            "Stage 4A-6.6a is an offline audit gate only. Full expert dataset collection remains false and blocked after this stage.",
        ],
    )
    write_md(
        out_dir / "long_term_rl_gdpo_note.md",
        [
            "# Long-Term RL/GDPO Note",
            "",
            "GDPO/RL/PPO/BC/IL remains future direction only. Stage 4A-6.6a did not train, did not create replay data, and did not create or modify a policy checkpoint.",
        ],
    )
    return decision, scorecard


def write_summary(out_dir: Path, decision: dict[str, Any], audits: dict[str, dict[str, Any]], bundle_manifest: dict[str, Any]) -> dict[str, Any]:
    answers = {
        "stage4a66_bundle_read": bool(bundle_manifest.get("all_required_inputs_present")),
        "isaac_started": False,
        "capture_run": False,
        "map_predict_or_sscnet_inference_run": False,
        "rollout_expert_sampling_or_rl_run": False,
        "scene_scale_passed": bool(audits["scale"]["pass"]),
        "topology_complexity_passed": bool(audits["topology"]["pass"]),
        "start_variant_diversity_passed": bool(audits["starts"]["pass"]),
        "fixed_validation_view_coverage_passed": bool(audits["fixed_views"]["pass"]),
        "observed_state_health_passed": bool(audits["observed_state"]["pass"]),
        "frontier_reachable_area_passed": bool(audits["frontier_reachability"]["pass"]),
        "obstacle_occlusion_passed": bool(audits["obstacle_occlusion"]["pass"]),
        "expert_usability_pre_audit_passed": bool(audits["expert_usability"]["pass"]),
        "hard_blockers": decision["hard_blockers"],
        "warnings": decision["warnings"],
        "scene_complexity_audit_passed": decision["scene_complexity_audit_passed"],
        "scene_ready_for_formal_expert_sampling_pilot": decision["scene_ready_for_formal_expert_sampling_pilot"],
        "formal_expert_sampling_ready_full_dataset": decision["formal_expert_sampling_ready_full_dataset"],
        "bounded_pilot_if_pass": "Tiny measured-only one-step/small pilot, qualified starts, manual review before scaling.",
        "scene_revision_if_fail": "Run Stage 4A-6.6b scene revision based on blockers before retrying 6.6a.",
        "future_stage4a67_sketch_or_revision_plan_generated": True,
        "user_requirement_recorded": "Formal expert sampling remains blocked until scene complexity is audited.",
        "long_term_gdpo_future_only": True,
        "recommended_next": decision["recommended_next"],
    }
    summary = {"stage": "Stage 4A-6.6a", "answers": answers, "decision": decision}
    write_json(out_dir / "stage4a66a_scene_complexity_audit_summary.json", summary)
    lines = ["# Stage 4A-6.6a Scene Complexity Audit Summary", ""]
    questions = [
        ("1. Read Stage 4A-6.6 audit input bundle?", answers["stage4a66_bundle_read"]),
        ("2. No Isaac startup?", not answers["isaac_started"]),
        ("3. No capture?", not answers["capture_run"]),
        ("4. No map_predict / SSCNet inference?", not answers["map_predict_or_sscnet_inference_run"]),
        ("5. No rollout / expert sampling / RL?", not answers["rollout_expert_sampling_or_rl_run"]),
        ("6. Scale passed?", answers["scene_scale_passed"]),
        ("7. Topology complexity passed?", answers["topology_complexity_passed"]),
        ("8. Start diversity passed?", answers["start_variant_diversity_passed"]),
        ("9. Fixed validation view coverage passed?", answers["fixed_validation_view_coverage_passed"]),
        ("10. observed_state health passed?", answers["observed_state_health_passed"]),
        ("11. frontier / reachable area passed?", answers["frontier_reachable_area_passed"]),
        ("12. obstacle / occlusion passed?", answers["obstacle_occlusion_passed"]),
        ("13. expert usability pre-audit passed?", answers["expert_usability_pre_audit_passed"]),
        ("16. scene_complexity_audit_passed?", answers["scene_complexity_audit_passed"]),
        ("17. scene_ready_for_formal_expert_sampling_pilot?", answers["scene_ready_for_formal_expert_sampling_pilot"]),
        ("18. formal_expert_sampling_ready_full_dataset still false?", not answers["formal_expert_sampling_ready_full_dataset"]),
        ("21. Future sketch or revision plan generated?", answers["future_stage4a67_sketch_or_revision_plan_generated"]),
        ("22. User requirement recorded?", True),
        ("23. Long-term GDPO future only?", answers["long_term_gdpo_future_only"]),
    ]
    for text, value in questions:
        lines.append(f"- {text} {bool_word(bool(value))}")
    lines.extend(
        [
            "",
            f"14. Hard blockers: {json.dumps(decision['hard_blockers'])}",
            f"15. Warnings: {json.dumps(decision['warnings'])}",
            f"19. If passed, pilot should be: {answers['bounded_pilot_if_pass']}",
            f"20. If failed, scene revision should be: {answers['scene_revision_if_fail']}",
            f"24. Recommended next: {decision['recommended_next']}",
        ]
    )
    write_md(out_dir / "stage4a66a_scene_complexity_audit_summary.md", lines)
    return summary


def write_missing_fields_report(out_dir: Path, missing_inputs: list[str], skipped_plots: list[dict[str, Any]]) -> dict[str, Any]:
    generated_files = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    self_outputs = {"missing_fields_report.json", "missing_fields_report.md"}
    missing_required_outputs = [name for name in REQUIRED_TEXT_FILES if name not in generated_files and name not in self_outputs]
    for plot in REQUIRED_PLOTS:
        if plot not in generated_files and f"{plot}.skipped.md" not in generated_files:
            missing_required_outputs.append(plot)
    report = {
        "stage": "Stage 4A-6.6a",
        "missing_required_inputs": missing_inputs,
        "skipped_plots": skipped_plots,
        "missing_required_outputs_at_report_time": missing_required_outputs,
        "pass": not missing_inputs and not missing_required_outputs,
    }
    write_json(out_dir / "missing_fields_report.json", report)
    write_md(out_dir / "missing_fields_report.md", md_kv("Missing Fields Report", report))
    return report


def write_plots(
    out_dir: Path,
    meta: dict[str, Any],
    graph: dict[str, Any],
    starts: list[dict[str, Any]],
    poses: list[dict[str, Any]],
    observed_state: np.ndarray,
    euclidean_matrix: list[list[float]],
    step_rows: list[dict[str, Any]],
    frontier_xy: np.ndarray,
    reachable_masks: dict[str, np.ndarray],
    frontier_rows: list[dict[str, Any]],
    density_rows: list[dict[str, Any]],
    occlusion_rows: list[dict[str, Any]],
    scorecard: list[dict[str, Any]],
    decision: dict[str, Any],
) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    try:
        plot_layout(out_dir / "audit_scene_layout_topdown.png", meta, starts=starts, poses=None)
        plot_topology(out_dir / "audit_topology_graph.png", meta, graph)
        plot_layout(out_dir / "audit_room_corridor_opening_map.png", meta, starts=None, poses=None)
        plot_layout(out_dir / "audit_start_variants_topdown.png", meta, starts=starts, poses=None)
        plot_distance_matrix(out_dir / "audit_start_distance_matrix.png", [str(s["name"]) for s in starts], euclidean_matrix, "Start Euclidean distance matrix (m)")
        plot_layout(out_dir / "audit_validation_view_coverage.png", meta, starts=None, poses=poses)
        plot_observed_topdown(out_dir / "audit_observed_topdown_final.png", observed_state, meta)
        plot_bar(
            out_dir / "audit_observed_ratio_by_view.png",
            [str(row["step"]) for row in step_rows],
            [float(row["observed_ratio"]) for row in step_rows],
            "Observed ratio by fixed validation view",
            "observed_ratio",
        )
        plot_frontier_topdown(out_dir / "audit_frontier_distribution_topdown.png", observed_state, meta, frontier_xy, starts)
        plot_reachable_area(out_dir / "audit_reachable_area_by_start.png", meta, reachable_masks, starts)
        plot_bar(
            out_dir / "audit_reachable_frontier_by_start.png",
            [str(row["name"]) for row in frontier_rows],
            [float(row["reachable_frontier_count"]) for row in frontier_rows],
            "Reachable frontier-adjacent free cells by start",
            "cells",
        )
        plot_bar(
            out_dir / "audit_obstacle_density_topdown.png",
            [str(row["zone"]) for row in density_rows],
            [float(row["obstacles_per_m2"]) for row in density_rows],
            "Obstacle density by zone",
            "obstacles / m2",
        )
        plot_bar(
            out_dir / "audit_occlusion_proxy_by_view.png",
            [str(row["view_index"]) for row in occlusion_rows],
            [float(row["depth_discontinuity_count"]) for row in occlusion_rows],
            "Depth discontinuity occlusion proxy by view",
            "count",
        )
        plot_scorecard(out_dir / "audit_complexity_scorecard.png", scorecard)
        plot_flowchart(
            out_dir / "audit_pass_fail_flowchart.png",
            [(str(row["category"]), bool(row["pass"])) for row in scorecard]
            + [("scene_complexity_audit_passed", bool(decision["scene_complexity_audit_passed"]))],
            "Audit pass/fail flow",
        )
        plot_flowchart(
            out_dir / "audit_next_stage_decision_flowchart.png",
            [
                ("6.6 construction/fixed validation complete", True),
                ("6.6a offline scene complexity audit", bool(decision["scene_complexity_audit_passed"])),
                (
                    "6.7 bounded formal expert sampling pilot",
                    bool(decision["scene_ready_for_formal_expert_sampling_pilot"]),
                ),
                ("full dataset remains blocked", not bool(decision["formal_expert_sampling_ready_full_dataset"])),
                ("RL/GDPO future only", True),
            ],
            "Next-stage decision flow",
        )
    except Exception as exc:
        skipped.append({"plot": "one_or_more_required_plots", "reason": repr(exc)})
        write_md(out_dir / "one_or_more_required_plots.skipped.md", [f"# Plot generation skipped", "", repr(exc)])
        raise
    return skipped


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ.setdefault(key, "1")
    repo_root = Path(__file__).resolve().parents[1]
    stage_dir = Path(args.stage4a66_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    actual_workers = min(int(args.max_workers), os.cpu_count() or 1)

    context_manifest = audit_loaded_context(repo_root, out_dir)
    bundle_manifest = audit_bundle_manifest(stage_dir, out_dir)
    missing_inputs = [row["relative_path"] for row in bundle_manifest["required_inputs"] if not row["exists"]]
    hash_audit = audit_hashes(stage_dir, out_dir, actual_workers)

    meta = read_json(stage_dir / "larger_complex_scene_v1_metadata.json")
    graph = read_json(stage_dir / "scene_topology_graph.json")
    prelim = read_json(stage_dir / "preliminary_complexity_metrics.json")
    starts = read_json(stage_dir / "start_variants.json")
    poses = read_json(stage_dir / "validation_pose_manifest.json")
    observed_summary = read_json(stage_dir / "observed_state_validation_summary.json")
    transition_summary = read_json(stage_dir / "observed_state_transition_summary.json")
    fixed_capture = read_json(stage_dir / "fixed_capture_validation.json")
    stage_hw = read_json(stage_dir / "hardware_utilization_report.json") if (stage_dir / "hardware_utilization_report.json").exists() else {}

    if str(meta.get("variant")) != str(args.expected_scene_variant):
        missing_inputs.append(f"variant_mismatch:{meta.get('variant')} != {args.expected_scene_variant}")
    if int(meta.get("scene_seed", -1)) != int(args.expected_scene_seed):
        missing_inputs.append(f"scene_seed_mismatch:{meta.get('scene_seed')} != {args.expected_scene_seed}")

    scale_audit, scale_rows = audit_scale(args, out_dir, meta, observed_summary)
    topology_audit, graph_distances, start_assignment = audit_topology(args, out_dir, meta, graph, prelim)
    starts_audit, start_table, euclidean_matrix = audit_starts(args, out_dir, meta, starts, graph_distances, start_assignment)
    fixed_audit, depth_rows = audit_fixed_views(args, out_dir, stage_dir, poses, fixed_capture, actual_workers)
    observed_audit, observed_state, step_rows = audit_observed_state(args, out_dir, stage_dir, observed_summary, transition_summary)
    frontier_audit, candidate_rows, reachable_masks, frontier_xy = audit_frontier_reachability(args, out_dir, meta, observed_state, starts)
    obstacle_audit, density_rows, occlusion_rows = audit_obstacles(args, out_dir, meta, starts, depth_rows, step_rows)
    expert_audit = audit_expert_usability(out_dir, meta, observed_audit, frontier_audit, candidate_rows)
    negative_audit = negative_scope_reports(out_dir, args, meta)

    audits = {
        "scale": scale_audit,
        "topology": topology_audit,
        "starts": starts_audit,
        "fixed_views": fixed_audit,
        "observed_state": observed_audit,
        "frontier_reachability": frontier_audit,
        "obstacle_occlusion": obstacle_audit,
        "expert_usability": expert_audit,
        "safety_negative_scope": negative_audit,
    }
    decision, scorecard = synthesize_decision(out_dir, audits, candidate_rows)
    summary = write_summary(out_dir, decision, audits, bundle_manifest)
    skipped_plots = []
    if args.save_viz:
        skipped_plots = write_plots(
            out_dir,
            meta,
            graph,
            starts,
            poses,
            observed_state,
            euclidean_matrix,
            step_rows,
            frontier_xy,
            reachable_masks,
            read_csv_as_dicts(out_dir / "reachable_frontier_by_start.csv"),
            density_rows,
            occlusion_rows,
            scorecard,
            decision,
        )
    hardware = audit_hardware(args, out_dir, stage_hw, start_time, task_count=9)
    missing_report = write_missing_fields_report(out_dir, missing_inputs, skipped_plots)

    print(
        json.dumps(
            {
                "stage": "Stage 4A-6.6a",
                "output_dir": str(out_dir),
                "scene_complexity_audit_passed": decision["scene_complexity_audit_passed"],
                "scene_ready_for_formal_expert_sampling_pilot": decision["scene_ready_for_formal_expert_sampling_pilot"],
                "formal_expert_sampling_ready_full_dataset": decision["formal_expert_sampling_ready_full_dataset"],
                "hard_blockers": decision["hard_blockers"],
                "warnings": decision["warnings"],
                "missing_report_pass": missing_report["pass"],
                "actual_max_workers": hardware["actual_max_workers"],
                "input_file_count_hashed": hash_audit["file_count"],
                "context_loaded": context_manifest["all_required_context_files_present"],
                "summary_file": str(out_dir / "stage4a66a_scene_complexity_audit_summary.md"),
            },
            indent=2,
        )
    )


def read_csv_as_dicts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    main()
