#!/usr/bin/env python3
"""Stage 4A-6.6 larger_complex_scene_v1 construction and validation.

This is scene construction and fixed-view validation only.  It does not run a
rollout, execute actions, call map_predict, call SSCNet inference, generate an
expert dataset, create a replay buffer, write a policy checkpoint, or perform
RL/GDPO/PPO/BC/IL.
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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

from depth_to_voxel import create_observed_grid, integrate_depth_frame, summarize_observed_grid
from scene_factory import LARGE_BOUNDS, build_larger_complex_scene_v1


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
SIM_DIR = WORKSPACE / "sim_explorer"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a66_larger_complex_scene_v1_validation"
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
SCENE_FACTORY_HASH_BEFORE_IMPLEMENTATION = "deedd45e12373e992884d8840dca80fa294b9056bd4af7f5e4381872104b2798"
DEPTH_KEY = "distance_to_image_plane"
RGB_KEY_CANDIDATES = ("rgb", "rgba")
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
STATE_CMAP = ListedColormap(["#2f343b", "#80b9c4", "#c95c5c"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)


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


def sha256_file(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_kv(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            value_text = "`" + json.dumps(value, sort_keys=True)[:700] + "`"
        else:
            value_text = f"`{value}`"
        lines.append(f"| {key} | {value_text} |")
    return "\n".join(lines)


def list_md(title: str, rows: list[str]) -> str:
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in rows]])


def write_table(path: Path, rows: list[dict[str, Any]], preferred_keys: list[str]) -> None:
    keys = [key for key in preferred_keys if any(key in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key], sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def inventory_md(title: str, rows: list[dict[str, Any]], keys: list[str]) -> str:
    lines = [f"# {title}", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        values = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def center_of_bounds(bounds: dict[str, list[float]]) -> tuple[float, float]:
    return (
        0.5 * (float(bounds["x"][0]) + float(bounds["x"][1])),
        0.5 * (float(bounds["y"][0]) + float(bounds["y"][1])),
    )


def area_of_bounds(bounds: dict[str, list[float]]) -> float:
    return (float(bounds["x"][1]) - float(bounds["x"][0])) * (float(bounds["y"][1]) - float(bounds["y"][0]))


def box_xy(spec: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, _ = (float(v) for v in spec["position"])
    sx, sy, _ = (float(v) for v in spec["size"])
    return x - 0.5 * sx, y - 0.5 * sy, sx, sy


def rect_distance_xy(point: tuple[float, float], rect: tuple[float, float, float, float]) -> float:
    x, y = point
    rx, ry, w, h = rect
    dx = max(rx - x, 0.0, x - (rx + w))
    dy = max(ry - y, 0.0, y - (ry + h))
    return float(math.hypot(dx, dy))


def point_inside_rect(point: tuple[float, float], rect: tuple[float, float, float, float], margin: float = 0.0) -> bool:
    x, y = point
    rx, ry, w, h = rect
    return rx - margin <= x <= rx + w + margin and ry - margin <= y <= ry + h + margin


def graph_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    graph = metadata["topology_graph"]
    nodes = [node["id"] for node in graph["nodes"]]
    adjacency = {node: set() for node in nodes}
    for edge in graph["edges"]:
        source = edge["source"]
        target = edge["target"]
        adjacency[source].add(target)
        adjacency[target].add(source)

    seen: set[str] = set()
    components: list[list[str]] = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        component = []
        seen.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    degrees = {node: len(adjacency[node]) for node in nodes}
    branch_points = sorted(node for node, degree in degrees.items() if degree >= 3)
    dead_ends = sorted(node for node, degree in degrees.items() if degree <= 1)
    cycle_rank = len(graph["edges"]) - len(nodes) + len(components)
    return {
        "node_count": len(nodes),
        "edge_count": len(graph["edges"]),
        "degrees": degrees,
        "average_graph_degree": float(sum(degrees.values()) / max(len(degrees), 1)),
        "branch_point_count": len(branch_points),
        "branch_points": branch_points,
        "dead_end_count": len(dead_ends),
        "dead_ends": dead_ends,
        "cycle_rank": int(cycle_rank),
        "estimated_loop_count": int(max(cycle_rank, 0)),
        "component_count": len(components),
        "largest_connected_component": max((len(component) for component in components), default=0),
        "connected_components": components,
        "all_nodes_connected": len(components) == 1,
    }


def shortest_path_lengths(adjacency: dict[str, set[str]], start: str) -> dict[str, int]:
    distances = {start: 0}
    queue = [start]
    while queue:
        current = queue.pop(0)
        for neighbor in sorted(adjacency[current]):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def topology_connectivity(metadata: dict[str, Any]) -> dict[str, Any]:
    nodes = [node["id"] for node in metadata["topology_graph"]["nodes"]]
    adjacency = {node: set() for node in nodes}
    for edge in metadata["topology_graph"]["edges"]:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])
    start_regions = {}
    for start in metadata["start_variants"]:
        name = start["name"]
        x, y, _ = (float(v) for v in start["position"])
        region = None
        for room in metadata["rooms"]:
            bounds = room["bounds"]
            if bounds["x"][0] <= x <= bounds["x"][1] and bounds["y"][0] <= y <= bounds["y"][1]:
                region = room["name"]
        for corridor in metadata["corridors"]:
            bounds = corridor["bounds"]
            if bounds["x"][0] <= x <= bounds["x"][1] and bounds["y"][0] <= y <= bounds["y"][1]:
                region = corridor["name"]
        start_regions[name] = region
    graph_distances = {}
    for source, source_region in start_regions.items():
        graph_distances[source] = {}
        if source_region is None:
            continue
        distances = shortest_path_lengths(adjacency, source_region)
        for target, target_region in start_regions.items():
            graph_distances[source][target] = distances.get(target_region) if target_region else None
    return {
        "start_region_assignment": start_regions,
        "start_to_start_graph_distance_edges": graph_distances,
        "all_start_regions_found": all(region is not None for region in start_regions.values()),
        "all_start_variants_mutually_connected": all(
            value is not None for row in graph_distances.values() for value in row.values()
        ),
    }


def geometry_checks(metadata: dict[str, Any]) -> dict[str, Any]:
    bounds = metadata["map_bounds"]
    obstacle_rects = [(obstacle["name"], box_xy(obstacle)) for obstacle in metadata["obstacles"]]
    wall_rects = [(wall["name"], box_xy(wall)) for wall in metadata["walls"]]
    starts = []
    for start in metadata["start_variants"]:
        point = (float(start["position"][0]), float(start["position"][1]))
        inside_bounds = bounds["x"][0] <= point[0] <= bounds["x"][1] and bounds["y"][0] <= point[1] <= bounds["y"][1]
        obstacle_hits = [name for name, rect in obstacle_rects if point_inside_rect(point, rect, margin=0.05)]
        nearest_wall = min((rect_distance_xy(point, rect) for _, rect in wall_rects), default=float("inf"))
        starts.append(
            {
                "name": start["name"],
                "inside_bounds": bool(inside_bounds),
                "obstacle_hits": obstacle_hits,
                "not_inside_obstacle": not obstacle_hits,
                "nearest_wall_clearance_m": float(nearest_wall),
                "not_too_close_to_wall": bool(nearest_wall >= 0.25),
                "expected_valid_depth_view": bool(start.get("expected_valid_depth_view")),
            }
        )
    positions = {start["name"]: np.asarray(start["position"][:2], dtype=np.float64) for start in metadata["start_variants"]}
    names = list(positions)
    matrix = [[float(np.linalg.norm(positions[a] - positions[b])) for b in names] for a in names]
    duplicates = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if float(np.linalg.norm(positions[a] - positions[b])) < 0.10:
                duplicates.append([a, b])
    return {
        "start_checks": starts,
        "starts_inside_bounds": all(item["inside_bounds"] for item in starts),
        "starts_collision_free_by_metadata": all(item["not_inside_obstacle"] for item in starts),
        "starts_not_too_close_to_wall": all(item["not_too_close_to_wall"] for item in starts),
        "starts_not_duplicated": not duplicates,
        "duplicate_start_pairs": duplicates,
        "start_distance_names": names,
        "start_distance_matrix_m": matrix,
    }


def preliminary_metrics(metadata: dict[str, Any]) -> dict[str, Any]:
    graph = graph_summary(metadata)
    geometry = geometry_checks(metadata)
    connectivity = topology_connectivity(metadata)
    room_areas = {room["name"]: area_of_bounds(room["bounds"]) for room in metadata["rooms"]}
    obstacle_density = {}
    for room in metadata["rooms"]:
        name = room["name"]
        count = sum(1 for obstacle in metadata["obstacles"] if obstacle.get("region") == name)
        obstacle_density[name] = {
            "obstacle_count": count,
            "area_m2": float(room_areas[name]),
            "obstacles_per_m2": float(count / room_areas[name]),
        }
    metrics = {
        "room_count": len(metadata["rooms"]),
        "corridor_count": len(metadata["corridors"]),
        "wall_count": len(metadata["walls"]),
        "opening_count": len(metadata["openings"]),
        "obstacle_count": len(metadata["obstacles"]),
        "start_variant_count": len(metadata["start_variants"]),
        "validation_pose_count": len(metadata["validation_camera_poses"]),
        "branch_point_count": graph["branch_point_count"],
        "dead_end_count": graph["dead_end_count"],
        "estimated_loop_count": graph["estimated_loop_count"],
        "cycle_rank": graph["cycle_rank"],
        "average_graph_degree": graph["average_graph_degree"],
        "largest_connected_component": graph["largest_connected_component"],
        "connectivity_between_start_variants": connectivity,
        "narrow_passage_count": metadata["topology_summary"]["narrow_passage_count"],
        "minimum_doorway_width_m": metadata["topology_summary"]["minimum_passage_width_m"],
        "maximum_corridor_chain_length_estimate": 4,
        "occlusion_object_count": sum(1 for obstacle in metadata["obstacles"] if obstacle["category"] in {"half_wall_occluder", "shelf_like_long_cuboid", "cabinet_like_tall_cuboid", "pillar_cuboid"}),
        "room_size_diversity": {
            "min_area_m2": float(min(room_areas.values())),
            "max_area_m2": float(max(room_areas.values())),
            "ratio": float(max(room_areas.values()) / min(room_areas.values())),
        },
        "obstacle_density_per_room": obstacle_density,
        **{key: value for key, value in geometry.items() if key != "start_distance_matrix_m"},
        "start_distance_names": geometry["start_distance_names"],
        "start_distance_matrix_m": geometry["start_distance_matrix_m"],
        "formal_expert_sampling_ready": False,
        "scene_complexity_audit_passed": False,
        "audit_required_next": True,
    }
    return metrics


def hardware_report(requested_max_workers: int) -> dict[str, Any]:
    actual_workers = min(int(requested_max_workers), os.cpu_count() or 1)
    gpu_name = "unavailable"
    cuda_available = False
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
    except Exception as exc:  # pragma: no cover - hardware dependent
        gpu_name = f"nvidia-smi unavailable: {exc}"
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        cuda_available = False
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "os_cpu_count": os.cpu_count(),
        "requested_max_workers": int(requested_max_workers),
        "actual_max_workers": int(actual_workers),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "GPU name": gpu_name,
        "cuda_available_if_torch_available": bool(cuda_available),
        "parallel_backend": "offline metadata only uses local python loops; Isaac capture uses single app startup",
    }


def context_manifest() -> dict[str, Any]:
    entries = []
    combined = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_stage4a65av": "Stage 4A-6.5av" in text,
                "contains_larger_complex_scene_v1": "larger_complex_scene_v1" in text,
                "contains_no_rollout": "no rollout" in text or "not rollout" in text,
                "contains_no_formal_expert_sampling": "formal expert sampling" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.6",
        "loaded_context_files": entries,
        "confirmed_stage4a65av_completed": "Stage 4A-6.5av" in combined and "spatially_consistent_healthy_start_room_b" in combined,
        "confirmed_medium_three_starts_completed": all(token in combined for token in ("start_room_a", "start_corridor", "start_room_b")),
        "confirmed_next_gate_larger_scene_then_audit": "Stage 4A-6.6" in combined and "Stage 4A-6.6a" in combined,
        "confirmed_not_formal_expert_sampling": "not formal expert sampling" in combined or "not formal" in combined,
        "chat_history_not_used_as_source": True,
    }


def write_context_outputs(output_dir: Path, manifest: dict[str, Any]) -> None:
    save_json(output_dir / "loaded_context_manifest.json", manifest)
    rows = [
        f"Stage 4A-6.5av completed: `{manifest['confirmed_stage4a65av_completed']}`",
        f"Next gate larger scene then audit: `{manifest['confirmed_next_gate_larger_scene_then_audit']}`",
        f"Not formal expert sampling: `{manifest['confirmed_not_formal_expert_sampling']}`",
        *[f"`{item['path']}` sha256 `{item['sha256']}`" for item in manifest["loaded_context_files"]],
    ]
    write_text(output_dir / "loaded_context_manifest.md", list_md("Loaded Context Manifest", rows))


def write_inventory_outputs(output_dir: Path, metadata: dict[str, Any]) -> None:
    rooms = metadata["rooms"]
    corridors = metadata["corridors"]
    openings = metadata["openings"]
    walls = metadata["walls"]
    obstacles = metadata["obstacles"]

    room_rows = [room | {"area_m2": area_of_bounds(room["bounds"])} for room in rooms]
    corridor_rows = [corridor | {"area_m2": area_of_bounds(corridor["bounds"])} for corridor in corridors]
    opening_rows = [opening | {"source": opening["connects"][0], "target": opening["connects"][1]} for opening in openings]
    wall_rows = [wall | {"x_min": box_xy(wall)[0], "y_min": box_xy(wall)[1], "x_size": box_xy(wall)[2], "y_size": box_xy(wall)[3]} for wall in walls]
    obstacle_rows = [obstacle | {"x_min": box_xy(obstacle)[0], "y_min": box_xy(obstacle)[1], "x_size": box_xy(obstacle)[2], "y_size": box_xy(obstacle)[3]} for obstacle in obstacles]

    save_json(output_dir / "room_inventory.json", room_rows)
    write_table(output_dir / "room_inventory.csv", room_rows, ["name", "label", "role", "area_m2", "bounds"])
    write_text(output_dir / "room_inventory.md", inventory_md("Room Inventory", room_rows, ["name", "label", "role", "area_m2", "bounds"]))

    save_json(output_dir / "corridor_inventory.json", corridor_rows)
    write_table(output_dir / "corridor_inventory.csv", corridor_rows, ["name", "label", "role", "min_width_m", "area_m2", "bounds"])
    write_text(output_dir / "corridor_inventory.md", inventory_md("Corridor Inventory", corridor_rows, ["name", "label", "role", "min_width_m", "area_m2", "bounds"]))

    save_json(output_dir / "opening_inventory.json", opening_rows)
    write_table(output_dir / "opening_inventory.csv", opening_rows, ["name", "source", "target", "width", "orientation", "center"])
    write_text(output_dir / "opening_inventory.md", inventory_md("Opening Inventory", opening_rows, ["name", "source", "target", "width", "orientation", "center"]))

    save_json(output_dir / "wall_inventory.json", wall_rows)
    write_table(output_dir / "wall_inventory.csv", wall_rows, ["name", "axis", "position", "size", "x_min", "y_min", "x_size", "y_size"])

    save_json(output_dir / "obstacle_inventory.json", obstacle_rows)
    write_table(output_dir / "obstacle_inventory.csv", obstacle_rows, ["name", "region", "category", "position", "size", "x_min", "y_min", "x_size", "y_size"])
    write_text(output_dir / "obstacle_inventory.md", inventory_md("Obstacle Inventory", obstacle_rows, ["name", "region", "category", "position", "size"]))

    save_json(output_dir / "start_variants.json", metadata["start_variants"])
    write_text(output_dir / "start_variants.md", inventory_md("Start Variants", metadata["start_variants"], ["name", "position", "yaw_deg", "intended_local_topology_label", "inside_bounds", "not_inside_obstacle", "not_too_close_to_wall", "expected_valid_depth_view"]))

    save_json(output_dir / "validation_pose_manifest.json", metadata["validation_camera_poses"])
    write_text(output_dir / "validation_pose_manifest.md", inventory_md("Validation Pose Manifest", metadata["validation_camera_poses"], ["index", "position", "yaw_deg", "room", "note"]))


def write_scene_config_outputs(output_dir: Path, args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    config = {
        "stage": "Stage 4A-6.6",
        "scene_id": "larger_complex_scene_v1",
        "scene_seed": int(args.scene_seed),
        "bounds": metadata["map_bounds"],
        "floor_size_m": [24.0, 24.0],
        "voxel_size": float(args.voxel_size),
        "expected_observed_state_shape": metadata["expected_observed_state_shape"],
        "camera_width": int(args.camera_width),
        "camera_height": int(args.camera_height),
        "max_depth": float(args.max_depth),
        "pixel_stride": int(args.pixel_stride),
        "fixed_validation_pose_count": len(metadata["validation_camera_poses"]),
        "isaac_startup_policy": "exactly one clean AppLauncher startup for validation capture",
        "formal_expert_sampling_ready": False,
        "scene_complexity_audit_passed": False,
    }
    save_json(output_dir / "scene_construction_config.json", config)
    write_text(output_dir / "scene_construction_config.md", markdown_kv("Scene Construction Config", config))

    save_json(output_dir / "larger_complex_scene_v1_metadata.json", metadata)
    write_text(output_dir / "larger_complex_scene_v1_metadata.md", markdown_kv("Larger Complex Scene V1 Metadata", {
        "rooms": len(metadata["rooms"]),
        "corridors": len(metadata["corridors"]),
        "openings": len(metadata["openings"]),
        "walls": len(metadata["walls"]),
        "obstacles": len(metadata["obstacles"]),
        "starts": len(metadata["start_variants"]),
        "validation_poses": len(metadata["validation_camera_poses"]),
        "formal_expert_sampling_ready": False,
    }))

    save_json(output_dir / "scene_topology_graph.json", metadata["topology_graph"])
    write_text(output_dir / "scene_topology_graph.md", list_md("Scene Topology Graph", [
        f"Nodes: `{len(metadata['topology_graph']['nodes'])}`",
        f"Edges: `{len(metadata['topology_graph']['edges'])}`",
        "Graph is metadata-only and is the input to Stage 4A-6.6a audit.",
    ]))


def write_preliminary_outputs(output_dir: Path, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    connectivity = metrics["connectivity_between_start_variants"]
    save_json(output_dir / "topology_connectivity_summary.json", connectivity | graph_summary(metadata))
    write_text(output_dir / "topology_connectivity_summary.md", markdown_kv("Topology Connectivity Summary", connectivity | {
        "node_count": graph_summary(metadata)["node_count"],
        "edge_count": graph_summary(metadata)["edge_count"],
        "cycle_rank": graph_summary(metadata)["cycle_rank"],
        "all_nodes_connected": graph_summary(metadata)["all_nodes_connected"],
    }))
    save_json(output_dir / "preliminary_complexity_metrics.json", metrics)
    write_text(output_dir / "preliminary_complexity_metrics.md", markdown_kv("Preliminary Complexity Metrics", {
        "room_count": metrics["room_count"],
        "corridor_count": metrics["corridor_count"],
        "wall_count": metrics["wall_count"],
        "opening_count": metrics["opening_count"],
        "obstacle_count": metrics["obstacle_count"],
        "branch_point_count": metrics["branch_point_count"],
        "cycle_rank": metrics["cycle_rank"],
        "narrow_passage_count": metrics["narrow_passage_count"],
        "formal_expert_sampling_ready": False,
    }))
    checklist = {
        key: {
            "passed": bool(value),
            "note": "construction target check; final audit is Stage 4A-6.6a",
        }
        for key, value in metadata["complexity_targets"].items()
    }
    checklist["formal_expert_sampling_ready"] = {"passed": False, "note": "blocked until Stage 4A-6.6a audit passes"}
    save_json(output_dir / "preliminary_complexity_target_checklist.json", checklist)
    write_text(output_dir / "preliminary_complexity_target_checklist.md", inventory_md("Preliminary Complexity Target Checklist", [{"target": key, **value} for key, value in checklist.items()], ["target", "passed", "note"]))


def _pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def save_depth_png(path: Path, depth: np.ndarray, title: str) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError(f"No finite positive depth values for {path}")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def extract_depth(camera: Any, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    tensor = camera.data.output.get(DEPTH_KEY)
    if tensor is None:
        raise KeyError(f"{label}: camera output missing {DEPTH_KEY}; keys={list(camera.data.output.keys())}")
    depth = tensor[0].detach().cpu().numpy().astype(np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    finite = depth[np.isfinite(depth)]
    positive = finite[finite > 0.0]
    return depth, {
        "shape": [int(v) for v in depth.shape],
        "dtype": str(depth.dtype),
        "finite_count": int(finite.size),
        "positive_count": int(positive.size),
        "min": float(positive.min()) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
        "has_positive_finite_depth": bool(positive.size > 0),
    }


def extract_rgb(camera: Any, label: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEY_CANDIDATES:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        source = tensor[0].detach().cpu().numpy()
        if source.ndim != 3 or source.shape[-1] not in (3, 4):
            raise ValueError(f"{label}: expected RGB/RGBA image, got {source.shape}")
        rgb = normalize_rgb(source)
        stats = {
            "key": key,
            "shape": [int(v) for v in rgb.shape],
            "min": int(rgb.min()) if rgb.size else None,
            "max": int(rgb.max()) if rgb.size else None,
            "mean": float(rgb.mean()) if rgb.size else None,
            "std": float(rgb.std()) if rgb.size else None,
            "nonblank": bool(rgb.size and int(rgb.max()) > 2 and float(rgb.std()) >= 1.0),
        }
        return rgb, key, stats
    raise KeyError(f"{label}: camera output missing rgb/rgba; keys={list(camera.data.output.keys())}")


def state_transition(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    transitions = {
        "unknown_to_free": int(np.count_nonzero((before == UNKNOWN) & (after == FREE))),
        "unknown_to_occupied": int(np.count_nonzero((before == UNKNOWN) & (after == OCCUPIED))),
        "free_to_occupied": int(np.count_nonzero((before == FREE) & (after == OCCUPIED))),
        "occupied_to_free": int(np.count_nonzero((before == OCCUPIED) & (after == FREE))),
        "unchanged_unknown": int(np.count_nonzero((before == UNKNOWN) & (after == UNKNOWN))),
        "unchanged_free": int(np.count_nonzero((before == FREE) & (after == FREE))),
        "unchanged_occupied": int(np.count_nonzero((before == OCCUPIED) & (after == OCCUPIED))),
    }
    transitions["newly_observed"] = transitions["unknown_to_free"] + transitions["unknown_to_occupied"]
    transitions["invalid_label_count_after"] = int(np.count_nonzero(~np.isin(after, [UNKNOWN, FREE, OCCUPIED])))
    return transitions


def launch_isaac_and_capture(args: argparse.Namespace, app_launcher_cls: Any, output_dir: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    if hasattr(args, "enable_cameras"):
        setattr(args, "enable_cameras", True)

    app_launcher = app_launcher_cls(args)
    simulation_app = app_launcher.app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        def add_lighting() -> None:
            dome_cfg = sim_utils.DomeLightCfg(intensity=3200.0, color=(0.82, 0.84, 0.80))
            dome_cfg.func("/World/Light", dome_cfg)

        def make_camera() -> Camera:
            sim_utils.create_prim("/World/CameraRig", "Xform")
            cfg = CameraCfg(
                prim_path="/World/CameraRig/CameraSensor",
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
            return Camera(cfg=cfg)

        def set_camera_pose(camera: Camera, sim: sim_utils.SimulationContext, pose: dict[str, Any]) -> None:
            position = [float(v) for v in pose["position"]]
            yaw_rad = float(pose.get("yaw_rad", math.radians(float(pose["yaw_deg"]))))
            target = _pose_target(position, yaw_rad)
            camera.set_world_poses_from_view(
                eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
                targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
            )

        def settle(camera: Camera, sim: sim_utils.SimulationContext) -> None:
            for _ in range(max(int(args.settle_steps), 1)):
                sim.step()
                camera.update(dt=sim.get_physics_dt())

        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args.device)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([0.0, -16.5, 13.5], [0.0, 0.0, 0.6])
        add_lighting()
        runtime_metadata = build_larger_complex_scene_v1(seed=int(args.scene_seed), spawn=True, sim_utils_module=sim_utils)
        camera = make_camera()
        sim.reset()

        intrinsic_matrix = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
        camera_info = {
            "sensor_api_depth_key": DEPTH_KEY,
            "depth_units": "meters",
            "width": int(args.camera_width),
            "height": int(args.camera_height),
            "max_depth": float(args.max_depth),
            "near_depth": 0.05,
            "horizontal_fov_deg": 90.0,
            "intrinsic_matrix": intrinsic_matrix.tolist(),
            "fx": float(intrinsic_matrix[0, 0]),
            "fy": float(intrinsic_matrix[1, 1]),
            "cx": float(intrinsic_matrix[0, 2]),
            "cy": float(intrinsic_matrix[1, 2]),
            "data_types_requested": ["rgb", DEPTH_KEY],
        }
        save_json(output_dir / "camera_info.json", camera_info)

        observed_state = create_observed_grid(metadata["map_bounds"], float(args.voxel_size))
        capture_records: list[dict[str, Any]] = []
        pose_records: list[dict[str, Any]] = []
        transition_records: list[dict[str, Any]] = []
        step_summaries: list[dict[str, Any]] = []

        for pose in metadata["validation_camera_poses"]:
            idx = int(pose["index"])
            set_camera_pose(camera, sim, pose)
            settle(camera, sim)
            depth, depth_stats = extract_depth(camera, f"validation pose {idx}")
            rgb, rgb_key, rgb_stats = extract_rgb(camera, f"validation pose {idx}")

            depth_npy_path = output_dir / f"validation_depth_{idx:03d}.npy"
            depth_png_path = output_dir / f"validation_depth_{idx:03d}.png"
            rgb_path = output_dir / f"validation_rgb_{idx:03d}.png"
            pose_path = output_dir / f"validation_pose_{idx:03d}.json"
            np.save(depth_npy_path, depth)
            save_depth_png(depth_png_path, depth, f"validation depth {idx:03d}")
            Image.fromarray(rgb).save(rgb_path)

            pose_record = {
                "index": idx,
                "position": [float(v) for v in pose["position"]],
                "yaw_rad": float(pose["yaw_rad"]),
                "yaw_deg": float(pose["yaw_deg"]),
                "target": _pose_target([float(v) for v in pose["position"]], float(pose["yaw_rad"])),
                "note": str(pose.get("note", "")),
                "room": str(pose.get("room", "")),
                "convention_for_voxel": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
            }
            if hasattr(camera.data, "pos_w"):
                pose_record["isaac_camera_pos_w"] = camera.data.pos_w[0].detach().cpu().numpy().astype(float).tolist()
            if hasattr(camera.data, "quat_w_ros"):
                pose_record["isaac_camera_quat_w_ros"] = camera.data.quat_w_ros[0].detach().cpu().numpy().astype(float).tolist()
            save_json(pose_path, pose_record)
            pose_records.append(pose_record)

            before = observed_state.copy()
            observed_state = integrate_depth_frame(
                observed_state=observed_state,
                depth_image=depth,
                camera_pose=pose_record,
                intrinsics=camera_info,
                map_bounds=metadata["map_bounds"],
                voxel_size=float(args.voxel_size),
                pixel_stride=int(args.pixel_stride),
                max_depth=float(args.max_depth),
            )
            transition = {"step": idx, **state_transition(before, observed_state)}
            transition_records.append(transition)
            step_path = output_dir / f"observed_state_step{idx:03d}.npy"
            np.save(step_path, observed_state)
            step_summary = summarize_observed_grid(observed_state)
            step_summary.update({"step": idx, "observed_state_file": str(step_path), "transition": transition})
            step_summaries.append(step_summary)

            capture_records.append(
                {
                    "index": idx,
                    "depth_file": str(depth_npy_path),
                    "depth_png_file": str(depth_png_path),
                    "rgb_file": str(rgb_path),
                    "pose_file": str(pose_path),
                    "rgb_key_used": rgb_key,
                    "camera_output_keys": sorted(camera.data.output.keys()),
                    "depth_stats": depth_stats,
                    "rgb_stats": rgb_stats,
                }
            )
            print(
                f"[INFO]: validation pose {idx:03d} rgb={rgb_path.name} depth={depth_npy_path.name} "
                f"positive_depth={depth_stats['positive_count']} nonblank_rgb={rgb_stats['nonblank']}"
            )

        final_observed_path = output_dir / "observed_state_final.npy"
        np.save(final_observed_path, observed_state)
        observed_summary = summarize_observed_grid(observed_state)
        observed_summary.update(
            {
                "voxel_size": float(args.voxel_size),
                "map_bounds": metadata["map_bounds"],
                "pixel_stride": int(args.pixel_stride),
                "step_summaries": step_summaries,
                "observed_state_final": str(final_observed_path),
                "measured_only": True,
                "prediction_used": False,
                "map_predict_called": False,
                "sscnet_inference_called": False,
            }
        )
        save_json(output_dir / "observed_summary.json", observed_summary)

        scene_metadata = runtime_metadata | {
            "output_dir": str(output_dir),
            "camera": camera_info | {"poses": pose_records},
            "poses": pose_records,
            "capture_records": capture_records,
            "depth_summaries": [record["depth_stats"] | {"index": record["index"]} for record in capture_records],
            "observed_summary": observed_summary,
            "prediction_used": False,
            "expert_used": False,
            "observed_map_modified_by_prediction": False,
            "rl_or_il_training_used": False,
            "sscnet_training_used": False,
        }
        save_json(output_dir / "scene_metadata.json", scene_metadata)

        return {
            "scene_load_validation": {
                "stage": "Stage 4A-6.6",
                "isaac_startup_count": 1,
                "scene_loaded": True,
                "scene_id": "larger_complex_scene_v1",
                "scene_seed": int(args.scene_seed),
                "spawned_wall_count": len(runtime_metadata["walls"]),
                "spawned_obstacle_count": len(runtime_metadata["obstacles"]),
                "spawned_floor": True,
                "headless_requested": bool(getattr(args, "headless", False)),
            },
            "fixed_capture_validation": {
                "fixed_validation_pose_count": len(capture_records),
                "rgb_nonblank_count": sum(1 for record in capture_records if record["rgb_stats"]["nonblank"]),
                "depth_positive_count": sum(1 for record in capture_records if record["depth_stats"]["has_positive_finite_depth"]),
                "minimum_required_views": 12,
                "minimum_required_rgb_nonblank": 8,
                "minimum_required_depth_positive": 8,
                "all_capture_records": capture_records,
                "no_actions_executed": True,
                "no_planner_run": True,
                "no_map_predict": True,
                "no_sscnet_inference": True,
            },
            "observed_state_validation": {
                **observed_summary,
                "expected_shape": [240, 240, 30],
                "shape_matches_expected": list(observed_state.shape) == [240, 240, 30],
                "invalid_label_count": int(np.count_nonzero(~np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED]))),
                "measured_only": True,
                "prediction_used": False,
            },
            "observed_state_transition_summary": {
                "transition_records": transition_records,
                "total_newly_observed": int(sum(record["newly_observed"] for record in transition_records)),
                "total_unknown_to_free": int(sum(record["unknown_to_free"] for record in transition_records)),
                "total_unknown_to_occupied": int(sum(record["unknown_to_occupied"] for record in transition_records)),
                "any_occupied_to_free": any(record["occupied_to_free"] > 0 for record in transition_records),
                "invalid_label_count_final": int(np.count_nonzero(~np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED]))),
            },
        }, simulation_app
    except Exception:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        raise


def draw_base_layout(ax: Any, metadata: dict[str, Any], include_labels: bool = True) -> None:
    bounds = metadata["map_bounds"]
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.grid(True, color="#e1e5e8", linewidth=0.45)
    for corridor in metadata["corridors"]:
        x0, x1 = corridor["bounds"]["x"]
        y0, y1 = corridor["bounds"]["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#d8eef1", edgecolor="#6aa2a8", linewidth=0.9, alpha=0.55))
        if include_labels:
            cx, cy = center_of_bounds(corridor["bounds"])
            ax.text(cx, cy, corridor["name"], ha="center", va="center", fontsize=6.5, color="#2f6066")
    for room in metadata["rooms"]:
        x0, x1 = room["bounds"]["x"]
        y0, y1 = room["bounds"]["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f7efe2", edgecolor="#a78352", linewidth=0.9, alpha=0.5))
        if include_labels:
            cx, cy = center_of_bounds(room["bounds"])
            ax.text(cx, cy, room["name"], ha="center", va="center", fontsize=7, color="#674d2a")
    for wall in metadata["walls"]:
        x, y, w, h = box_xy(wall)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#4e5358", edgecolor="#2d3034", linewidth=0.25, alpha=0.85))


def draw_camera_arrow(ax: Any, pose: dict[str, Any], color: str = "#1954a6") -> None:
    x, y, _ = (float(v) for v in pose["position"])
    yaw = float(pose["yaw_rad"])
    dx = math.cos(yaw) * 0.75
    dy = math.sin(yaw) * 0.75
    ax.arrow(x, y, dx, dy, width=0.025, head_width=0.18, length_includes_head=True, color=color, zorder=5)
    ax.scatter([x], [y], c=[color], s=20, zorder=6)
    ax.text(x + 0.10, y + 0.10, str(pose["index"]), fontsize=7, color=color, weight="bold")


def save_layout_plots(output_dir: Path, metadata: dict[str, Any], metrics: dict[str, Any], missing: list[dict[str, str]]) -> None:
    try:
        fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
        draw_base_layout(ax, metadata)
        for obstacle in metadata["obstacles"]:
            x, y, w, h = box_xy(obstacle)
            ax.add_patch(Rectangle((x, y), w, h, facecolor="#b25f36", edgecolor="#63311d", linewidth=0.3, alpha=0.78))
        for opening in metadata["openings"]:
            ax.scatter([opening["center"][0]], [opening["center"][1]], marker="s", s=15, c="#33a653", zorder=7)
        ax.set_title("larger_complex_scene_v1 topdown layout")
        fig.savefig(output_dir / "scene_layout_topdown.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
        draw_base_layout(ax, metadata)
        for opening in metadata["openings"]:
            ax.scatter([opening["center"][0]], [opening["center"][1]], marker="s", s=24, c="#2ea44f", zorder=7)
            ax.text(opening["center"][0] + 0.08, opening["center"][1] + 0.08, opening["name"].replace("door_", ""), fontsize=5.5)
        ax.set_title("rooms, corridors, and openings")
        fig.savefig(output_dir / "rooms_corridors_openings_topdown.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
        draw_base_layout(ax, metadata, include_labels=False)
        category_colors = {
            "table_like_cuboid": "#c67d35",
            "cabinet_like_tall_cuboid": "#4f8464",
            "shelf_like_long_cuboid": "#8d7851",
            "pillar_cuboid": "#6a7380",
            "half_wall_occluder": "#5f6268",
            "small_clutter_cuboid": "#b96d3c",
        }
        for obstacle in metadata["obstacles"]:
            x, y, w, h = box_xy(obstacle)
            ax.add_patch(Rectangle((x, y), w, h, facecolor=category_colors.get(obstacle["category"], "#9a6b45"), edgecolor="#303030", linewidth=0.35))
        ax.set_title("obstacles by category")
        fig.savefig(output_dir / "obstacles_topdown.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
        draw_base_layout(ax, metadata)
        for start in metadata["start_variants"]:
            pose = {"position": start["position"], "yaw_rad": start["yaw_rad"], "index": start["name"].replace("start_", "")}
            draw_camera_arrow(ax, pose, color="#7d3c98")
        ax.set_title("start variants")
        fig.savefig(output_dir / "start_variants_topdown.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
        draw_base_layout(ax, metadata)
        for pose in metadata["validation_camera_poses"]:
            draw_camera_arrow(ax, pose, color="#1954a6")
        ax.set_title("fixed validation camera poses")
        fig.savefig(output_dir / "validation_camera_poses_topdown.png", dpi=160)
        plt.close(fig)

        save_topology_plot(output_dir / "scene_topology_graph.png", metadata, title="scene topology graph")
        save_topology_plot(output_dir / "connectivity_graph.png", metadata, title="connectivity graph")
        save_complexity_checklist_plot(output_dir / "complexity_target_checklist.png", metadata["complexity_targets"])
        save_start_distance_matrix(output_dir / "start_distance_matrix.png", metrics)
        save_obstacle_density_plot(output_dir / "obstacle_density_topdown.png", metadata, metrics)
        save_audit_gate_flowchart(output_dir / "audit_gate_flowchart.png")
    except Exception as exc:
        missing.append({"plot_group": "layout_and_topology", "reason": str(exc)})


def save_topology_plot(path: Path, metadata: dict[str, Any], title: str) -> None:
    centers: dict[str, tuple[float, float]] = {}
    for room in metadata["rooms"]:
        centers[room["name"]] = center_of_bounds(room["bounds"])
    for corridor in metadata["corridors"]:
        centers[corridor["name"]] = center_of_bounds(corridor["bounds"])
    fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
    ax.set_aspect("equal", adjustable="box")
    for edge in metadata["topology_graph"]["edges"]:
        x0, y0 = centers[edge["source"]]
        x1, y1 = centers[edge["target"]]
        ax.plot([x0, x1], [y0, y1], color="#667085", linewidth=1.0, alpha=0.8)
        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), f"{edge['width_m']:.2f}", fontsize=6, color="#475467")
    for node in metadata["topology_graph"]["nodes"]:
        x, y = centers[node["id"]]
        color = "#d99b43" if node["kind"] == "room" else "#5aa6b8"
        ax.scatter([x], [y], c=[color], s=110, edgecolors="#2f3437", zorder=3)
        ax.text(x, y + 0.25, node["id"], ha="center", va="bottom", fontsize=7, weight="bold")
    ax.set_xlim(metadata["map_bounds"]["x"])
    ax.set_ylim(metadata["map_bounds"]["y"])
    ax.grid(True, color="#e5e7eb", linewidth=0.45)
    ax.set_title(title)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_complexity_checklist_plot(path: Path, targets: dict[str, bool]) -> None:
    rows = list(targets.items())
    labels = [key.replace("_", "\n") for key, _ in rows]
    values = [1 if bool(value) else 0 for _, value in rows]
    colors = ["#3a9b63" if value else "#c95c5c" for value in values]
    fig, ax = plt.subplots(figsize=(11.0, 5.5), constrained_layout=True)
    ax.bar(range(len(rows)), values, color=colors)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["false", "true"])
    ax.set_title("preliminary complexity target checklist")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_start_distance_matrix(path: Path, metrics: dict[str, Any]) -> None:
    names = metrics["start_distance_names"]
    matrix = np.asarray(metrics["start_distance_matrix_m"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
    image = ax.imshow(matrix, cmap="magma")
    fig.colorbar(image, ax=ax, label="meters")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_title("start-to-start distance matrix")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_obstacle_density_plot(path: Path, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    densities = metrics["obstacle_density_per_room"]
    max_density = max((item["obstacles_per_m2"] for item in densities.values()), default=1.0)
    fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
    ax.set_xlim(metadata["map_bounds"]["x"])
    ax.set_ylim(metadata["map_bounds"]["y"])
    ax.set_aspect("equal", adjustable="box")
    cmap = plt.get_cmap("YlGnBu")
    for room in metadata["rooms"]:
        density = densities[room["name"]]["obstacles_per_m2"]
        x0, x1 = room["bounds"]["x"]
        y0, y1 = room["bounds"]["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=cmap(density / max_density), edgecolor="#3a3a3a", linewidth=0.7))
        ax.text(*center_of_bounds(room["bounds"]), f"{room['name']}\n{density:.2f}/m2", ha="center", va="center", fontsize=7)
    ax.set_title("obstacle density by room")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_audit_gate_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.8), constrained_layout=True)
    ax.axis("off")
    boxes = [
        (0.08, 0.55, "6.6 construct\nlarger scene"),
        (0.35, 0.55, "fixed capture\nmeasured only"),
        (0.62, 0.55, "6.6a scene\ncomplexity audit"),
        (0.62, 0.18, "expert sampling\nblocked now"),
    ]
    for x, y, text in boxes:
        color = "#dff2e4" if "blocked" not in text else "#f8d7da"
        ax.add_patch(Rectangle((x, y), 0.22, 0.22, facecolor=color, edgecolor="#444", linewidth=1.0, transform=ax.transAxes))
        ax.text(x + 0.11, y + 0.11, text, ha="center", va="center", fontsize=10, transform=ax.transAxes)
    for start, end in [((0.30, 0.66), (0.35, 0.66)), ((0.57, 0.66), (0.62, 0.66)), ((0.73, 0.55), (0.73, 0.40))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, transform=ax.transAxes))
    ax.set_title("Stage 4A-6.6 audit gate flow")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def topdown_state(observed: np.ndarray) -> np.ndarray:
    occupied = np.any(observed == OCCUPIED, axis=2)
    free = np.any(observed == FREE, axis=2)
    top = np.full(observed.shape[:2], UNKNOWN, dtype=np.int8)
    top[free] = FREE
    top[occupied] = OCCUPIED
    return top


def save_capture_plots(output_dir: Path, capture_validation: dict[str, Any], observed_validation: dict[str, Any], missing: list[dict[str, str]]) -> None:
    try:
        records = capture_validation["all_capture_records"]
        rgbs = [np.asarray(Image.open(record["rgb_file"]).convert("RGB")) for record in records]
        depths = [np.load(record["depth_file"]) for record in records]
        labels = [f"{record['index']:03d}" for record in records]

        cols = 4
        rows = int(math.ceil(len(rgbs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        for ax, rgb, label in zip(axes, rgbs, labels):
            ax.imshow(rgb)
            ax.set_title(label)
            ax.axis("off")
        for ax in axes[len(rgbs) :]:
            ax.axis("off")
        fig.savefig(output_dir / "camera_rgb_grid.png", dpi=150)
        plt.close(fig)

        finite_values = [depth[np.isfinite(depth) & (depth > 0.0)] for depth in depths]
        vmin = min(float(values.min()) for values in finite_values if values.size)
        vmax = max(float(values.max()) for values in finite_values if values.size)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), constrained_layout=True)
        axes = np.asarray(axes).reshape(-1)
        last_image = None
        for ax, depth, label in zip(axes, depths, labels):
            masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
            last_image = ax.imshow(masked, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(label)
            ax.set_xlabel("u")
            ax.set_ylabel("v")
        for ax in axes[len(depths) :]:
            ax.axis("off")
        fig.colorbar(last_image, ax=axes[: len(depths)].tolist(), shrink=0.70, label="depth (m)")
        fig.savefig(output_dir / "camera_depth_grid.png", dpi=150)
        plt.close(fig)

        observed = np.load(output_dir / "observed_state_final.npy")
        top = topdown_state(observed)
        fig, ax = plt.subplots(figsize=(8.5, 8.0), constrained_layout=True)
        image = ax.imshow(top.T, origin="lower", cmap=STATE_CMAP, norm=STATE_NORM)
        fig.colorbar(image, ax=ax, ticks=[UNKNOWN, FREE, OCCUPIED], label="state")
        ax.set_title("observed topdown final, fixed views only")
        ax.set_xlabel("grid x")
        ax.set_ylabel("grid y")
        fig.savefig(output_dir / "observed_topdown_final.png", dpi=160)
        plt.close(fig)

        ratios = [step["observed_ratio"] for step in observed_validation["step_summaries"]]
        fig, ax = plt.subplots(figsize=(8.0, 4.0), constrained_layout=True)
        ax.plot(range(len(ratios)), ratios, marker="o", color="#2878a8")
        ax.set_xlabel("fixed validation view index")
        ax.set_ylabel("observed ratio")
        ax.set_title("observed ratio from fixed validation views")
        ax.grid(True, color="#e4e7ea")
        fig.savefig(output_dir / "observed_ratio_fixed_views.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        missing.append({"plot_group": "capture_and_observed", "reason": str(exc)})


def write_validation_outputs(output_dir: Path, validation: dict[str, Any]) -> None:
    scene_load = validation["scene_load_validation"]
    capture = validation["fixed_capture_validation"]
    observed = validation["observed_state_validation"]
    transition = validation["observed_state_transition_summary"]
    save_json(output_dir / "scene_load_validation.json", scene_load)
    write_text(output_dir / "scene_load_validation.md", markdown_kv("Scene Load Validation", scene_load))
    save_json(output_dir / "fixed_capture_validation.json", capture)
    write_text(output_dir / "fixed_capture_validation.md", markdown_kv("Fixed Capture Validation", {
        "fixed_validation_pose_count": capture["fixed_validation_pose_count"],
        "rgb_nonblank_count": capture["rgb_nonblank_count"],
        "depth_positive_count": capture["depth_positive_count"],
        "no_actions_executed": capture["no_actions_executed"],
        "no_map_predict": capture["no_map_predict"],
        "no_sscnet_inference": capture["no_sscnet_inference"],
    }))
    save_json(output_dir / "observed_state_validation_summary.json", observed)
    write_text(output_dir / "observed_state_validation_summary.md", markdown_kv("Observed State Validation Summary", {
        "shape": observed["shape"],
        "observed_ratio": observed["observed_ratio"],
        "unknown_count": observed["unknown_count"],
        "free_count": observed["free_count"],
        "occupied_count": observed["occupied_count"],
        "invalid_label_count": observed["invalid_label_count"],
        "measured_only": observed["measured_only"],
    }))
    save_json(output_dir / "observed_state_transition_summary.json", transition)
    write_text(output_dir / "observed_state_transition_summary.md", markdown_kv("Observed State Transition Summary", {
        "total_newly_observed": transition["total_newly_observed"],
        "total_unknown_to_free": transition["total_unknown_to_free"],
        "total_unknown_to_occupied": transition["total_unknown_to_occupied"],
        "any_occupied_to_free": transition["any_occupied_to_free"],
        "invalid_label_count_final": transition["invalid_label_count_final"],
    }))


def write_no_reports(output_dir: Path) -> None:
    reports = {
        "no_rollout_report": {
            "rollout_run": False,
            "open_ended_loop_run": False,
            "selected_expert_action_executed": False,
            "runtime_policy": "fixed validation capture only",
        },
        "no_formal_expert_sampling_report": {
            "formal_expert_sampling_run": False,
            "expert_dataset_generated": False,
            "formal_expert_sampling_ready": False,
            "blocked_until": "Stage 4A-6.6a scene complexity audit passes",
        },
        "no_map_predict_report": {
            "map_predict_called": False,
            "prediction_npz_created": False,
            "prediction_written_to_observed_state": False,
            "sscnet_inference_called": False,
        },
        "no_rl_gdpo_report": {
            "rl_run": False,
            "gdpo_run": False,
            "ppo_run": False,
            "behavior_cloning_run": False,
            "imitation_learning_run": False,
            "policy_checkpoint_created": False,
            "replay_buffer_created": False,
        },
    }
    for stem, report in reports.items():
        save_json(output_dir / f"{stem}.json", report)
        write_text(output_dir / f"{stem}.md", markdown_kv(stem.replace("_", " ").title(), report))


def write_code_and_hardware_outputs(output_dir: Path, args: argparse.Namespace) -> None:
    hardware = hardware_report(int(args.max_workers))
    save_json(output_dir / "hardware_utilization_report.json", hardware)
    write_text(output_dir / "hardware_utilization_report.md", markdown_kv("Hardware Utilization Report", hardware))

    code_hash = {
        "scene_factory_py": {
            "path": str(SIM_DIR / "scene_factory.py"),
            "sha256_before_stage4a66_implementation": SCENE_FACTORY_HASH_BEFORE_IMPLEMENTATION,
            "sha256_after_stage4a66_implementation": sha256_file(SIM_DIR / "scene_factory.py"),
        },
        "build_stage4a66_script": {
            "path": str(SIM_DIR / "build_stage4a66_larger_complex_scene_v1.py"),
            "sha256": sha256_file(SIM_DIR / "build_stage4a66_larger_complex_scene_v1.py"),
        },
        "test_stage4a66_script": {
            "path": str(SIM_DIR / "test_stage4a66_larger_complex_scene_v1.py"),
            "sha256": sha256_file(SIM_DIR / "test_stage4a66_larger_complex_scene_v1.py"),
        },
        "checkpoint_audit_only": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256_file(Path(args.checkpoint).resolve()),
            "loaded": False,
        },
        "context_hashes": {str(path): sha256_file(path) for path in CONTEXT_FILES},
        "map_predict_code_loaded": False,
        "sscnet_checkpoint_loaded": False,
    }
    save_json(output_dir / "code_hash_audit.json", code_hash)
    write_text(output_dir / "code_hash_audit.md", markdown_kv("Code Hash Audit", {
        "scene_factory_before": code_hash["scene_factory_py"]["sha256_before_stage4a66_implementation"],
        "scene_factory_after": code_hash["scene_factory_py"]["sha256_after_stage4a66_implementation"],
        "build_script": code_hash["build_stage4a66_script"]["sha256"],
        "test_script": code_hash["test_stage4a66_script"]["sha256"],
        "checkpoint_loaded": False,
    }))


def write_final_text_outputs(output_dir: Path) -> None:
    write_text(
        output_dir / "future_stage4a66a_scene_complexity_audit_command_sketch.md",
        "\n".join(
            [
                "# Future Stage 4A-6.6a Scene Complexity Audit Command Sketch",
                "",
                "DO NOT RUN AS PART OF STAGE 4A-6.6.",
                "",
                "```bash",
                "cd /home/ubuntu22/sc_explorer_ws",
                "python sim_explorer/audit_stage4a66a_scene_complexity.py \\",
                "  --input_dir outputs/isaac_stage4a66_larger_complex_scene_v1_validation \\",
                "  --scene_id larger_complex_scene_v1 \\",
                "  --max_workers 32",
                "```",
            ]
        ),
    )
    write_text(
        output_dir / "do_not_start_expert_sampling_before_audit.md",
        "# Do Not Start Expert Sampling Before Audit\n\nformal expert sampling remains blocked until Stage 4A-6.6a scene_complexity_audit passes.",
    )
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\nRun Stage 4A-6.6a scene complexity audit using this output bundle. Do not start rollout or formal expert sampling yet.",
    )
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "# Long-Term RL/GDPO Note\n\nRL/GDPO/PPO/BC/IL remain future directions only and were not run in Stage 4A-6.6.",
    )


def write_summary_and_manifest(output_dir: Path, metadata: dict[str, Any], metrics: dict[str, Any], validation: dict[str, Any], missing: list[dict[str, str]], elapsed_s: float) -> None:
    target_values = [bool(value) for key, value in metadata["complexity_targets"].items() if key != "formal_expert_sampling_ready"]
    capture = validation["fixed_capture_validation"]
    observed = validation["observed_state_validation"]
    summary = {
        "stage": "Stage 4A-6.6",
        "scene_id": "larger_complex_scene_v1",
        "scene_seed": metadata["scene_seed"],
        "larger_scene_constructed": True,
        "isaac_headless_load_validated": bool(validation["scene_load_validation"]["scene_loaded"]),
        "isaac_startup_count": int(validation["scene_load_validation"]["isaac_startup_count"]),
        "fixed_capture_validated": capture["fixed_validation_pose_count"] >= 12 and capture["rgb_nonblank_count"] >= 8 and capture["depth_positive_count"] >= 8,
        "observed_state_shape": observed["shape"],
        "observed_state_shape_matches_expected": observed["shape_matches_expected"],
        "measured_only_observed_state": True,
        "preliminary_complexity_targets_met": all(target_values),
        "scene_complexity_audit_passed": False,
        "formal_expert_sampling_ready": False,
        "formal_expert_sampling_blocked": True,
        "rollout_run": False,
        "map_predict_called": False,
        "sscnet_inference_called": False,
        "rl_gdpo_ppo_bc_il_run": False,
        "metrics": {
            "rooms": metrics["room_count"],
            "corridors": metrics["corridor_count"],
            "openings": metrics["opening_count"],
            "walls": metrics["wall_count"],
            "obstacles": metrics["obstacle_count"],
            "starts": metrics["start_variant_count"],
            "validation_poses": metrics["validation_pose_count"],
            "cycle_rank": metrics["cycle_rank"],
        },
        "missing_fields_or_plots": missing,
        "elapsed_seconds": float(elapsed_s),
    }
    save_json(output_dir / "stage4a66_larger_complex_scene_v1_summary.json", summary)
    write_text(output_dir / "stage4a66_larger_complex_scene_v1_summary.md", markdown_kv("Stage 4A-6.6 Summary", summary))

    files = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    bundle = {
        "stage": "Stage 4A-6.6",
        "output_dir": str(output_dir),
        "file_count": len(files),
        "files": files,
        "audit_consumer": "Stage 4A-6.6a scene complexity audit",
        "formal_expert_sampling_ready": False,
        "missing_fields_or_plots": missing,
    }
    save_json(output_dir / "audit_input_bundle_manifest.json", bundle)
    write_text(output_dir / "audit_input_bundle_manifest.md", list_md("Audit Input Bundle Manifest", [
        f"Output dir: `{output_dir}`",
        f"File count: `{len(files)}`",
        "Consumer: `Stage 4A-6.6a scene complexity audit`",
        "formal expert sampling ready: `False`",
        *[f"`{name}`" for name in files[:120]],
    ]))

    save_json(output_dir / "missing_fields_report.json", {"missing": missing, "all_required_plots_generated": not missing})
    write_text(output_dir / "missing_fields_report.md", list_md("Missing Fields Report", ["No missing plot groups recorded." if not missing else json.dumps(missing, sort_keys=True)]))


def check_forbidden_outputs(output_dir: Path) -> None:
    forbidden = [
        "transitions.jsonl",
        "rollout_topdown_path.png",
        "observed_ratio_curve.png",
        "rollout_index.html",
        "expert_dataset_manifest.jsonl",
        "replay_buffer",
        "policy_checkpoint",
        "global_prediction_layer.npz",
    ]
    hits = []
    for path in output_dir.rglob("*"):
        name = path.name
        if name.startswith("frame003") or name.startswith("action002"):
            hits.append(str(path))
        for token in forbidden:
            if token in name:
                hits.append(str(path))
    if hits:
        raise RuntimeError(f"Forbidden Stage 4A-6.6 outputs created: {hits}")


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Stage 4A-6.6 larger_complex_scene_v1 construction and validation.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=160)
    parser.add_argument("--camera_height", type=int, default=120)
    parser.add_argument("--max_depth", type=float, default=18.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--pixel_stride", type=int, default=4)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    return args, AppLauncher


def main() -> None:
    started = time.perf_counter()
    args, app_launcher_cls = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    simulation_app = None

    try:
        context = context_manifest()
        write_context_outputs(output_dir, context)
        if not context["confirmed_stage4a65av_completed"]:
            raise RuntimeError("Stage 4A-6.5av completion was not confirmed in project context")
        if int(args.scene_seed) != 0:
            raise ValueError("Stage 4A-6.6 clean validation requires deterministic scene_seed=0")

        metadata = build_larger_complex_scene_v1(seed=int(args.scene_seed), spawn=False)
        metrics = preliminary_metrics(metadata)
        write_code_and_hardware_outputs(output_dir, args)
        write_scene_config_outputs(output_dir, args, metadata)
        write_inventory_outputs(output_dir, metadata)
        write_preliminary_outputs(output_dir, metadata, metrics)
        write_no_reports(output_dir)
        write_final_text_outputs(output_dir)

        validation, simulation_app = launch_isaac_and_capture(args, app_launcher_cls, output_dir, metadata)
        write_validation_outputs(output_dir, validation)

        missing: list[dict[str, str]] = []
        save_layout_plots(output_dir, metadata, metrics, missing)
        save_capture_plots(output_dir, validation["fixed_capture_validation"], validation["observed_state_validation"], missing)

        check_forbidden_outputs(output_dir)
        write_summary_and_manifest(output_dir, metadata, metrics, validation, missing, time.perf_counter() - started)
        print(f"[INFO]: Stage 4A-6.6 larger_complex_scene_v1 validation outputs written to {output_dir}")
    finally:
        if simulation_app is not None:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
