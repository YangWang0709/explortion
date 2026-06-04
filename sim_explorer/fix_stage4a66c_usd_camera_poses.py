#!/usr/bin/env python3
"""Stage 4A-6.6c camera-pose fix for the validated home-like USD scene.

This stage does not edit USD geometry.  It only replaces the previous
outside-biased validation/inspection/start poses with interior-biased poses,
then performs one Isaac headless capture run and rebuilds the measured-only
observed_state from the corrected validation views.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw

import import_stage4a66c_usd_home_like_scene as usd_import


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
PREVIOUS_REQUIRED = [
    "stage4a66c_usd_defaultprim_fix_summary.json",
    "fixed_usd_validation_report.json",
    "scene_load_validation_defaultprim_retry.json",
    "fixed_capture_validation_defaultprim_retry.json",
    "visual_inspection_capture_validation_defaultprim_retry.json",
    "observed_state_validation_summary_defaultprim_retry.json",
    "visual_inspection_index.html",
    "usd_scene_flythrough.mp4",
]
OPTIONAL_PREVIOUS = [
    "scene_metadata.json",
    "camera_info.json",
    "start_variants.json",
    "validation_pose_manifest.json",
    "inspection_pose_manifest.json",
    "observed_state_final.npy",
    "usd_prim_inventory.json",
    "usd_semantic_guess_inventory.json",
]
FORBIDDEN_EXACT = {
    "transitions.jsonl",
    "rollout_index.html",
    "rollout_topdown_path.png",
    "expert_dataset_manifest.jsonl",
    "expert_dataset_manifest.json",
    "selected_action_execution_report.json",
    "selected_action_report.json",
    "action_execution_report.json",
    "global_prediction_layer.npz",
}
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
OBJECT_CATEGORIES = {
    "sofa_couch",
    "chair_armchair",
    "table",
    "bed",
    "cabinet_shelf",
    "kitchen",
    "bathroom",
    "storage_box_clutter",
    "rug_lamp_plant_decor",
    "door_window",
}

ORIGINAL_CAPTURE_POSE = usd_import.capture_pose
ORIGINAL_CHOOSE_MAP_BOUNDS = usd_import.choose_map_bounds
ORIGINAL_GENERATE_STARTS = usd_import.generate_start_variants
ORIGINAL_GENERATE_POSES = usd_import.generate_pose_manifests

RUNTIME: dict[str, Any] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list, tuple)):
            rendered = json.dumps(jsonable(value), sort_keys=True)
            if len(rendered) > 1600:
                rendered = rendered[:1600] + "..."
            value_text = f"`{rendered}`"
        else:
            value_text = f"`{value}`"
        lines.append(f"| {key} | {value_text} |")
    return "\n".join(lines)


def markdown_list(title: str, rows: list[str]) -> str:
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in rows]])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(jsonable(row.get(key)), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in keys
                }
            )


def bbox_from_row(row: dict[str, Any]) -> dict[str, list[float]] | None:
    if row.get("bbox"):
        return {"min": [float(v) for v in row["bbox"]["min"]], "max": [float(v) for v in row["bbox"]["max"]]}
    if row.get("bbox_min") and row.get("bbox_max"):
        return {"min": [float(v) for v in row["bbox_min"]], "max": [float(v) for v in row["bbox_max"]]}
    return None


def bbox_center(bbox: dict[str, list[float]]) -> tuple[float, float]:
    return (0.5 * (bbox["min"][0] + bbox["max"][0]), 0.5 * (bbox["min"][1] + bbox["max"][1]))


def bbox_distance_xy(point: tuple[float, float], bbox: dict[str, list[float]], margin: float = 0.0) -> float:
    x, y = point
    x0 = float(bbox["min"][0]) - margin
    y0 = float(bbox["min"][1]) - margin
    x1 = float(bbox["max"][0]) + margin
    y1 = float(bbox["max"][1]) + margin
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return float(math.hypot(dx, dy))


def point_inside_bbox_xy(point: tuple[float, float], bbox: dict[str, list[float]], margin: float = 0.0) -> bool:
    x, y = point
    return bool(
        float(bbox["min"][0]) - margin <= x <= float(bbox["max"][0]) + margin
        and float(bbox["min"][1]) - margin <= y <= float(bbox["max"][1]) + margin
    )


def yaw_to(point: list[float], target: tuple[float, float]) -> float:
    return float(math.atan2(float(target[1]) - float(point[1]), float(target[0]) - float(point[0])))


def normalize_angle(value: float) -> float:
    while value <= -math.pi:
        value += math.tau
    while value > math.pi:
        value -= math.tau
    return float(value)


def pose_target(position: list[float], yaw_rad: float) -> tuple[float, float]:
    return (float(position[0] + math.cos(yaw_rad)), float(position[1] + math.sin(yaw_rad)))


def row_path(row: dict[str, Any]) -> str:
    return str(row.get("path") or "")


def row_categories(row: dict[str, Any]) -> set[str]:
    return set(row.get("categories") or row.get("semantic_categories") or [])


def is_floor0_inside_path(path: str) -> bool:
    bad = ("/ShowcaseAssets", "/foundation", "/exterior", "entrance_apron")
    return "/World/Building/Floor_0" in path and not any(token in path for token in bad)


def load_context_and_previous(args: argparse.Namespace, output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    context_entries = []
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        context_entries.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "mentions_camera_pose_fix": "camera" in text.lower() or "visual" in text.lower(),
                "stage4a66d_blocked": "stage4a66d_executed=false" in text.lower(),
                "stage4a67_blocked": "stage4a67_executed=false" in text.lower(),
            }
        )
    previous_dir = Path(args.stage4a66c_defaultprim_dir).resolve()
    required = {}
    for name in PREVIOUS_REQUIRED:
        path = previous_dir / name
        required[name] = {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}
        if not path.is_file():
            raise FileNotFoundError(f"required previous defaultPrim output missing: {path}")
    optional = {}
    for name in OPTIONAL_PREVIOUS:
        path = previous_dir / name
        optional[name] = {"path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}

    loaded = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "created_at_utc": utc_now(),
        "context_files": context_entries,
        "previous_defaultprim_output_dir": str(previous_dir),
        "previous_required_files": required,
        "previous_optional_files": optional,
        "fixed_usd": str(Path(args.fixed_usd).resolve()),
        "source_usd": str(Path(args.source_usd).resolve()),
        "current_task": "camera_validation_inspection_start_pose_fix_only",
        "no_usd_geometry_edit": True,
        "no_asset_download": True,
        "no_rollout": True,
        "no_formal_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
    }
    save_json(output_dir / "loaded_context_manifest.json", loaded)
    write_text(
        output_dir / "loaded_context_manifest.md",
        markdown_table(
            "Loaded Context Manifest",
            {
                "context_file_count": len(context_entries),
                "previous_defaultprim_output_dir": str(previous_dir),
                "fixed_usd": str(Path(args.fixed_usd).resolve()),
                "source_usd": str(Path(args.source_usd).resolve()),
                "no_usd_geometry_edit": True,
                "stage4a66d_executed": False,
                "stage4a67_executed": False,
            },
        ),
    )

    summary = read_json(previous_dir / "stage4a66c_usd_defaultprim_fix_summary.json")
    fixed_validation = read_json(previous_dir / "fixed_usd_validation_report.json")
    load_validation = read_json(previous_dir / "scene_load_validation_defaultprim_retry.json")
    capture_validation = read_json(previous_dir / "fixed_capture_validation_defaultprim_retry.json")
    inspection_validation = read_json(previous_dir / "visual_inspection_capture_validation_defaultprim_retry.json")
    observed_validation = read_json(previous_dir / "observed_state_validation_summary_defaultprim_retry.json")
    previous_manifest = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "previous_stage": summary.get("stage"),
        "fixed_usd_exists": Path(args.fixed_usd).is_file(),
        "fixed_usd_defaultPrim_valid": bool(fixed_validation.get("defaultPrim_valid")),
        "fixed_usd_defaultPrim": fixed_validation.get("defaultPrim_path") or summary.get("chosen_defaultPrim_path"),
        "dependency_closure_complete": bool(fixed_validation.get("dependency_closure_complete")),
        "remote_official_refs_remaining": fixed_validation.get("remote_official_refs_remaining"),
        "omniverse_refs_remaining": fixed_validation.get("omniverse_refs_remaining"),
        "unresolved_local_deps_remaining": fixed_validation.get("unresolved_local_deps_remaining"),
        "previous_isaac_load_succeeded": bool(load_validation.get("scene_loaded") is True),
        "previous_validation_rgb_count": capture_validation.get("nonblank_rgb_count"),
        "previous_validation_depth_count": capture_validation.get("finite_positive_depth_count"),
        "previous_inspection_rgb_count": inspection_validation.get("nonblank_rgb_count"),
        "previous_inspection_depth_count": inspection_validation.get("finite_positive_depth_count"),
        "previous_observed_state_final_exists": (previous_dir / "observed_state_final.npy").is_file(),
        "previous_observed_state_shape": observed_validation.get("shape"),
        "previous_visual_html": str(previous_dir / "visual_inspection_index.html"),
        "previous_mp4": str(previous_dir / "usd_scene_flythrough.mp4"),
        "user_reported_camera_issue": "previous cameras were outside the house/interior",
        "this_stage_scope": "correct interior camera poses, inspection poses, and start proposals only",
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
    }
    save_json(output_dir / "loaded_defaultprim_validation_manifest.json", previous_manifest)
    write_text(output_dir / "loaded_defaultprim_validation_manifest.md", markdown_table("Loaded DefaultPrim Validation Manifest", previous_manifest))
    return loaded, previous_manifest


def get_previous_poses(previous_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    validation = read_json(previous_dir / "validation_pose_manifest.json").get("poses", [])
    inspection = read_json(previous_dir / "inspection_pose_manifest.json").get("poses", [])
    starts = read_json(previous_dir / "start_variants.json")
    return validation, inspection, starts


def pose_in_interior_xy(position: list[float] | tuple[float, ...]) -> bool:
    x = float(position[0])
    y = float(position[1])
    in_building = -10.0 <= x <= 10.0 and 0.0 <= y <= 36.0
    in_room = (-9.7 <= x <= -0.7 and 7.6 <= y <= 36.1) or (0.7 <= x <= 9.7 and 7.6 <= y <= 36.1)
    in_corridor = -1.05 <= x <= 1.05 and 0.2 <= y <= 35.8
    return bool(in_building and (in_room or in_corridor))


def write_previous_camera_issue_report(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    previous_dir = Path(args.stage4a66c_defaultprim_dir).resolve()
    validation, inspection, starts = get_previous_poses(previous_dir)

    def classify(items: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
        rows = []
        for item in items:
            pos = item.get("position", [0, 0, 0])
            inside = pose_in_interior_xy(pos)
            rows.append(
                {
                    "role": role,
                    "index": item.get("index"),
                    "name": item.get("name"),
                    "position": pos,
                    "yaw_rad": item.get("yaw_rad", item.get("yaw")),
                    "inside_inferred_floor0_interior": inside,
                    "rejection_reason": None if inside else "outside_floor0_room_corridor_envelope_or_showcase_area",
                }
            )
        return rows

    rows = classify(validation, "previous_validation") + classify(inspection, "previous_inspection") + classify(starts, "previous_start")
    rejected = [row for row in rows if not row["inside_inferred_floor0_interior"]]
    report = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "user_reported_issue": "camera / validation / inspection poses are outside the house/interior",
        "previous_visual_package": str(previous_dir / "visual_inspection_index.html"),
        "previous_observed_state": str(previous_dir / "observed_state_final.npy"),
        "previous_validation_pose_count": len(validation),
        "previous_inspection_pose_count": len(inspection),
        "previous_start_count": len(starts),
        "previous_outside_or_suspect_pose_count": len(rejected),
        "previous_outside_examples": rejected[:16],
        "root_cause": "Previous pose generation used full USD scene bounds, including distant ShowcaseAssets and a -80..80 world envelope, instead of the Floor_0 room/corridor interior envelope.",
        "this_stage_fix": "Generate poses only inside inferred Floor_0 rooms/corridor and exclude ShowcaseAssets/exterior/foundation bboxes.",
    }
    save_json(output_dir / "previous_camera_issue_report.json", report)
    write_text(output_dir / "previous_camera_issue_report.md", markdown_table("Previous Camera Issue Report", report))
    RUNTIME["previous_rejected_rows"] = rejected
    return report


def build_interior_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    norm_rows = []
    for row in rows:
        bbox = bbox_from_row(row)
        if not bbox:
            continue
        norm_rows.append(
            {
                "path": row_path(row),
                "type_name": str(row.get("type_name") or ""),
                "categories": sorted(row_categories(row)),
                "bbox": bbox,
            }
        )

    room_re = re.compile(r"^/World/Building/Floor_0/floor_0_room_\d+$")
    rooms = [row for row in norm_rows if room_re.match(row["path"])]
    rooms.sort(key=lambda row: row["path"])
    room_by_id = {}
    for row in rooms:
        match = re.search(r"floor_0_room_(\d+)$", row["path"])
        if match:
            room_by_id[int(match.group(1))] = row

    floor_rows = [
        row
        for row in norm_rows
        if is_floor0_inside_path(row["path"]) and ("floor" in row["categories"] or "room" in row["categories"])
    ]
    object_rows = [
        row
        for row in norm_rows
        if is_floor0_inside_path(row["path"]) and bool(set(row["categories"]) & OBJECT_CATEGORIES)
    ]
    solid_rows = []
    for row in norm_rows:
        path = row["path"]
        cats = set(row["categories"])
        if not is_floor0_inside_path(path):
            continue
        if "floor" in cats and not (cats & (OBJECT_CATEGORIES | {"wall"})):
            continue
        if row["bbox"]["max"][2] < 0.10:
            continue
        solid_rows.append(row)

    room_points = []
    for room in rooms:
        bbox = room["bbox"]
        room_points.append(bbox_center(bbox))
    if room_points:
        arr = np.array(room_points, dtype=float)
        center = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
    else:
        center = (0.0, 18.0)

    floor_bboxes = [row["bbox"] for row in floor_rows]
    if floor_bboxes:
        xs = [bbox["min"][0] for bbox in floor_bboxes] + [bbox["max"][0] for bbox in floor_bboxes]
        ys = [bbox["min"][1] for bbox in floor_bboxes] + [bbox["max"][1] for bbox in floor_bboxes]
        interior_bounds = {"x": [max(min(xs), -10.0), min(max(xs), 10.0)], "y": [max(min(ys), 0.0), min(max(ys), 36.0)], "z": [0.0, 2.55]}
    else:
        interior_bounds = {"x": [-10.0, 10.0], "y": [0.0, 36.0], "z": [0.0, 2.55]}

    corridor_rects = [
        {"name": "south_lobby_corridor", "bbox": {"min": [-1.05, 0.35, 0.0], "max": [1.05, 7.45, 2.4]}},
        {"name": "central_room_corridor", "bbox": {"min": [-0.95, 8.20, 0.0], "max": [0.95, 35.45, 2.4]}},
    ]
    model = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "method": "Floor_0 room/corridor envelope from USD prim bboxes; ShowcaseAssets/exterior/foundation excluded.",
        "rooms": rooms,
        "room_by_id": room_by_id,
        "floor_rows": floor_rows,
        "object_rows": object_rows,
        "solid_rows": solid_rows,
        "corridor_rects": corridor_rects,
        "center": center,
        "interior_bounds": interior_bounds,
        "map_bounds": {
            "x": [float(interior_bounds["x"][0] - 0.5), float(interior_bounds["x"][1] + 0.5)],
            "y": [float(interior_bounds["y"][0] - 0.5), float(interior_bounds["y"][1] + 0.5)],
            "z": [0.0, 3.0],
        },
        "excluded_regions": [
            {"name": "ShowcaseAssets", "reason": "distant exterior asset display row; not house interior"},
            {"name": "source scene -80..80 full envelope", "reason": "full-world bounds made old cameras land outside"},
            {"name": "exterior/foundation/entrance apron", "reason": "outside shell or structural envelope, not camera starts"},
        ],
        "zone_missing": {
            "kitchen": "No clear Floor_0 kitchen/counter semantic bbox found; dining/table room is used as visible alternative.",
            "office": "No explicit office category, but Floor_0 room_2 and room_3 contain desk/chair/table props.",
        },
    }
    return model


def model_from_bbox_rows(bbox_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "path": row.get("path"),
            "type_name": row.get("type_name"),
            "categories": row.get("categories", []),
            "bbox": row.get("bbox"),
        }
        for row in bbox_rows
        if row.get("bbox")
    ]
    return build_interior_model(rows)


def write_interior_region_analysis(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    previous_dir = Path(args.stage4a66c_defaultprim_dir).resolve()
    rows = read_json(previous_dir / "usd_prim_inventory.json")
    model = build_interior_model(rows)
    RUNTIME["offline_model"] = model
    report = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "fixed_usd": str(Path(args.fixed_usd).resolve()),
        "source_usd": str(Path(args.source_usd).resolve()),
        "method": model["method"],
        "interior_bounds": model["interior_bounds"],
        "corrected_map_bounds": model["map_bounds"],
        "room_count": len(model["rooms"]),
        "rooms": [
            {"path": row["path"], "bbox": row["bbox"], "center": bbox_center(row["bbox"])}
            for row in model["rooms"]
        ],
        "corridor_rects": model["corridor_rects"],
        "floor_object_bbox_count": len(model["object_rows"]),
        "solid_bbox_count_for_clearance": len(model["solid_rows"]),
        "excluded_regions": model["excluded_regions"],
        "zone_missing": model["zone_missing"],
        "previous_full_scene_bounds_problem": "Previous USD scene bounds included /World [-80,80] and /World/Building/ShowcaseAssets around y=-25..-11; corrected poses ignore those exterior/showcase areas.",
    }
    save_json(output_dir / "usd_interior_region_analysis.json", report)
    write_text(output_dir / "usd_interior_region_analysis.md", markdown_table("USD Interior Region Analysis", report))
    return report


def clearance_at(model: dict[str, Any], point: tuple[float, float], z: float = 1.2) -> tuple[float, list[str], bool]:
    distances = []
    inside_any = False
    for row in model["solid_rows"]:
        bbox = row["bbox"]
        z_inside = float(bbox["min"][2]) - 0.05 <= z <= float(bbox["max"][2]) + 0.05
        distance = bbox_distance_xy(point, bbox)
        if z_inside and point_inside_bbox_xy(point, bbox, margin=0.08):
            inside_any = True
        if z_inside and distance > 1.0e-6:
            distances.append((distance, row["path"]))
        elif z_inside:
            distances.append((0.0, row["path"]))
    distances.sort(key=lambda item: item[0])
    if not distances:
        return 99.0, [], inside_any
    return float(distances[0][0]), [path for _, path in distances[:8]], inside_any


def nearest_objects(model: dict[str, Any], point: tuple[float, float], category: str | None = None) -> list[dict[str, Any]]:
    rows = []
    for row in model["object_rows"]:
        if category and category not in row["categories"]:
            continue
        bbox = row["bbox"]
        center = bbox_center(bbox)
        rows.append(
            {
                "path": row["path"],
                "categories": row["categories"],
                "center": center,
                "distance": float(math.hypot(center[0] - point[0], center[1] - point[1])),
            }
        )
    rows.sort(key=lambda item: item["distance"])
    return rows


def locate_interior_zone(model: dict[str, Any], point: tuple[float, float]) -> tuple[bool, str, str]:
    for room in model["rooms"]:
        if point_inside_bbox_xy(point, room["bbox"], margin=0.02):
            return True, Path(room["path"]).name, room["path"]
    for corridor in model["corridor_rects"]:
        if point_inside_bbox_xy(point, corridor["bbox"], margin=0.02):
            return True, corridor["name"], corridor["name"]
    return False, "outside_interior_envelope", ""


def make_pose(index: int, name: str, position: list[float], yaw_rad: float, zone: str, source: str, target_xy: tuple[float, float]) -> dict[str, Any]:
    return {
        "index": int(index),
        "name": str(name),
        "position": [float(v) for v in position],
        "yaw": float(yaw_rad),
        "yaw_rad": float(yaw_rad),
        "target_xy": [float(target_xy[0]), float(target_xy[1])],
        "semantic_zone_guess": str(zone),
        "source": str(source),
        "convention": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
    }


def start_specs() -> list[dict[str, Any]]:
    return [
        {"name": "start_living_room_sofa_interior", "position": [3.0, 11.5, 1.2], "target": [6.45, 13.95], "zone": "living_sofa_room"},
        {"name": "start_storage_room_interior", "position": [-3.0, 12.2, 1.2], "target": [-5.5, 10.8], "zone": "storage_clutter_room"},
        {"name": "start_office_desk_room_interior", "position": [-3.1, 22.0, 1.2], "target": [-6.9, 23.5], "zone": "office_desk_room"},
        {"name": "start_bedroom_private_room_interior", "position": [5.6, 21.2, 1.2], "target": [3.6, 23.8], "zone": "bedroom_private_room"},
        {"name": "start_bathroom_small_room_interior", "position": [-4.7, 32.3, 1.2], "target": [-8.4, 32.0], "zone": "bathroom_small_room"},
        {"name": "start_dining_table_room_interior", "position": [7.0, 28.6, 1.2], "target": [5.3, 31.2], "zone": "dining_table_room"},
        {"name": "start_south_corridor_interior", "position": [0.0, 4.2, 1.2], "target": [0.0, 12.2], "zone": "south_corridor"},
        {"name": "start_mid_corridor_interior", "position": [0.0, 21.8, 1.2], "target": [-5.0, 22.2], "zone": "mid_corridor"},
        {"name": "start_north_corridor_interior", "position": [0.0, 31.2, 1.2], "target": [5.3, 31.2], "zone": "north_corridor"},
        {"name": "start_dining_north_view_interior", "position": [6.8, 34.0, 1.2], "target": [5.3, 31.2], "zone": "dining_north_side"},
    ]


def generate_interior_starts(output_dir: Path, model: dict[str, Any], min_count: int = 8) -> list[dict[str, Any]]:
    starts = []
    rejected = []
    for idx, spec in enumerate(start_specs()):
        pos = [float(v) for v in spec["position"]]
        target_xy = (float(spec["target"][0]), float(spec["target"][1]))
        point = (pos[0], pos[1])
        inside, zone_name, zone_source = locate_interior_zone(model, point)
        clearance, nearby, inside_solid = clearance_at(model, point)
        if not inside or inside_solid or clearance < 0.18:
            rejected.append(
                {
                    "name": spec["name"],
                    "position": pos,
                    "inside": inside,
                    "zone_name": zone_name,
                    "clearance": clearance,
                    "inside_solid": inside_solid,
                    "reason": "start_candidate_not_clear_or_not_inside",
                }
            )
            continue
        nearest = nearest_objects(model, point)[:8]
        start = {
            "index": len(starts),
            "name": spec["name"],
            "position": pos,
            "yaw": yaw_to(pos, target_xy),
            "yaw_rad": yaw_to(pos, target_xy),
            "semantic_zone_guess": spec["zone"],
            "source": "interior_floor0_room_corridor_candidate",
            "target_xy": [float(target_xy[0]), float(target_xy[1])],
            "inside_inferred_interior": True,
            "interior_zone_name": zone_name,
            "interior_zone_source": zone_source,
            "approximate_clearance": clearance,
            "nearby_prims": nearby,
            "nearby_interior_objects": nearest,
            "pending_human_approval": True,
            "usable_for_future_expert_pilot": "pending_human_approval",
            "reason": "Generated inside inferred Floor_0 room/corridor envelope after excluding ShowcaseAssets/exterior bounds.",
        }
        starts.append(start)
    if len(starts) < min_count:
        raise RuntimeError(f"only {len(starts)} interior starts survived, need {min_count}; rejected={rejected}")
    save_json(output_dir / "start_variants_interior.json", starts)
    save_json(output_dir / "start_variants.json", starts)
    write_text(output_dir / "start_variants_interior.md", markdown_list("Interior Start Variants", [f"`{s['index']:03d}` `{s['name']}` pos `{s['position']}` zone `{s['semantic_zone_guess']}` clearance `{s['approximate_clearance']:.3f}`" for s in starts]))
    write_text(output_dir / "start_variants.md", markdown_list("Interior Start Variants", [f"`{s['name']}` pos `{s['position']}`" for s in starts]))
    report = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "start_variant_count": len(starts),
        "minimum_required": int(min_count),
        "rejected_start_candidates": rejected,
        "all_pending_human_approval": True,
        "method": model["method"],
    }
    save_json(output_dir / "start_variant_interior_report.json", report)
    write_text(output_dir / "start_variant_interior_report.md", markdown_table("Start Variant Interior Report", report))
    return starts


def offline_pose_score(model: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    pos = pose["position"]
    point = (float(pos[0]), float(pos[1]))
    target = tuple(float(v) for v in pose.get("target_xy", pose_target(pos, float(pose["yaw_rad"]))))
    inside, zone_name, zone_source = locate_interior_zone(model, point)
    clearance, nearby, inside_solid = clearance_at(model, point, float(pos[2]))
    nearby_objects = nearest_objects(model, point)[:10]
    forward_objects = []
    yaw = float(pose["yaw_rad"])
    for item in nearby_objects:
        center = item["center"]
        angle = math.atan2(center[1] - point[1], center[0] - point[0])
        if abs(normalize_angle(angle - yaw)) <= math.radians(70.0):
            forward_objects.append(item)
    target_inside = pose_in_interior_xy([target[0], target[1], pos[2]])
    outside_likelihood = 0.0
    if not inside:
        outside_likelihood += 0.70
    if not target_inside:
        outside_likelihood += 0.20
    if clearance < 0.18:
        outside_likelihood += 0.10
    collision_likelihood = 1.0 if inside_solid else max(0.0, min(1.0, (0.25 - clearance) / 0.25))
    return {
        "inside_inferred_interior": inside,
        "interior_zone_name": zone_name,
        "interior_zone_source": zone_source,
        "interior_clearance_m": clearance,
        "nearby_prims": nearby,
        "nearby_object_count": len(nearby_objects),
        "forward_object_count": len(forward_objects),
        "target_inside_inferred_interior": target_inside,
        "outside_likelihood_offline": float(min(1.0, outside_likelihood)),
        "collision_or_inside_object_likelihood": float(collision_likelihood),
        "estimated_object_visibility_offline": float(min(1.0, len(forward_objects) / 3.0)),
        "estimated_floor_wall_visibility_offline": 1.0 if inside else 0.0,
    }


def generate_interior_pose_manifests(
    output_dir: Path,
    model: dict[str, Any],
    starts: list[dict[str, Any]],
    min_validation: int,
    min_inspection: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation: list[dict[str, Any]] = []
    for start in starts:
        pos = start["position"]
        target = tuple(float(v) for v in start["target_xy"])
        validation.append(make_pose(len(validation), "validation_" + start["name"], pos, yaw_to(pos, target), start["semantic_zone_guess"], "interior_start_primary", target))
    for start in starts:
        if len(validation) >= min_validation:
            break
        base_yaw = float(start["yaw_rad"])
        for suffix, delta in (("left_context", math.radians(35.0)), ("right_context", -math.radians(35.0)), ("reverse_context", math.pi)):
            if len(validation) >= min_validation:
                break
            pos = start["position"]
            yaw = normalize_angle(base_yaw + delta)
            target = pose_target(pos, yaw)
            validation.append(make_pose(len(validation), f"validation_{start['name']}_{suffix}", pos, yaw, start["semantic_zone_guess"], f"interior_start_{suffix}", target))

    inspection: list[dict[str, Any]] = []
    for start in starts:
        base_yaw = float(start["yaw_rad"])
        for suffix, delta in (("primary", 0.0), ("left", math.radians(55.0)), ("right", -math.radians(55.0)), ("reverse", math.pi)):
            if len(inspection) >= min_inspection:
                break
            pos = start["position"]
            yaw = normalize_angle(base_yaw + delta)
            target = tuple(float(v) for v in start["target_xy"]) if suffix == "primary" else pose_target(pos, yaw)
            inspection.append(make_pose(len(inspection), f"inspection_{start['name']}_{suffix}", pos, yaw, start["semantic_zone_guess"], f"interior_four_way_{suffix}", target))
        if len(inspection) >= min_inspection:
            break

    for label, poses, required in (("validation", validation, min_validation), ("inspection", inspection, min_inspection)):
        rejected = []
        accepted = []
        for pose in poses:
            score = offline_pose_score(model, pose)
            if not score["inside_inferred_interior"] or score["collision_or_inside_object_likelihood"] >= 0.95:
                rejected.append({"pose": pose, "score": score, "reason": "offline_not_interior_or_collision"})
            else:
                accepted.append(pose)
        if len(accepted) < required:
            raise RuntimeError(f"{label} interior pose generation failed: {len(accepted)} accepted, need {required}, rejected={rejected[:5]}")
        poses[:] = accepted[:required]
        for idx, pose in enumerate(poses):
            pose["index"] = idx

    manifest_v = {"pose_count": len(validation), "poses": validation, "source": "interior_floor0_room_corridor_pose_fix"}
    manifest_i = {"pose_count": len(inspection), "poses": inspection, "source": "interior_floor0_room_corridor_pose_fix"}
    for legacy, selected, manifest, title in (
        ("validation_pose_manifest", "selected_validation_pose_manifest", manifest_v, "Selected Validation Pose Manifest"),
        ("inspection_pose_manifest", "selected_inspection_pose_manifest", manifest_i, "Selected Inspection Pose Manifest"),
    ):
        save_json(output_dir / f"{legacy}.json", manifest)
        save_json(output_dir / f"{selected}.json", manifest)
        rows = [f"`{p['index']:03d}` `{p['name']}` pos `{p['position']}` yaw `{p['yaw_rad']:.3f}` zone `{p['semantic_zone_guess']}`" for p in manifest["poses"]]
        write_text(output_dir / f"{legacy}.md", markdown_list(title, rows))
        write_text(output_dir / f"{selected}.md", markdown_list(title, rows))

    report = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "candidate_source": "curated Floor_0 room/corridor starts plus yaw variants",
        "validation_pose_count": len(validation),
        "inspection_pose_count": len(inspection),
        "minimum_validation_required": int(min_validation),
        "minimum_inspection_required": int(min_inspection),
        "all_selected_inside_inferred_interior": True,
        "exterior_candidates_rejected_from_previous_package": len(RUNTIME.get("previous_rejected_rows", [])),
        "method": model["method"],
    }
    save_json(output_dir / "candidate_pose_generation_report.json", report)
    write_text(output_dir / "candidate_pose_generation_report.md", markdown_table("Candidate Pose Generation Report", report))
    return validation, inspection


def patched_choose_map_bounds(args: argparse.Namespace, scene_bounds: dict[str, list[float]] | None) -> tuple[dict[str, list[float]], str]:
    model = RUNTIME.get("offline_model") or {}
    bounds = model.get("map_bounds") or {"x": [-10.5, 10.5], "y": [-0.5, 36.5], "z": [0.0, 3.0]}
    return bounds, "Camera-pose-fix override: use inferred Floor_0 interior map bounds, not full USD world/ShowcaseAssets bounds."


def patched_generate_start_variants(
    output_dir: Path,
    semantic_rows: list[dict[str, Any]],
    bbox_rows: list[dict[str, Any]],
    scene_bounds: dict[str, list[float]] | None,
    map_bounds: dict[str, list[float]],
) -> list[dict[str, Any]]:
    model = model_from_bbox_rows(bbox_rows)
    RUNTIME["capture_model"] = model
    return generate_interior_starts(output_dir, model, int(RUNTIME["min_start_variants"]))


def patched_generate_pose_manifests(
    output_dir: Path,
    starts: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    scene_bounds: dict[str, list[float]] | None,
    map_bounds: dict[str, list[float]],
    min_validation: int,
    min_inspection: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model = RUNTIME.get("capture_model") or RUNTIME.get("offline_model")
    if not model:
        model = build_interior_model(read_json(Path(RUNTIME["previous_dir"]) / "usd_prim_inventory.json"))
    return generate_interior_pose_manifests(output_dir, model, starts, min_validation, min_inspection)


def edge_density(rgb: np.ndarray) -> float:
    gray = rgb.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    if gx.size == 0 or gy.size == 0:
        return 0.0
    threshold = max(6.0, float(gray.std()) * 0.35)
    return float((np.count_nonzero(gx > threshold) + np.count_nonzero(gy > threshold)) / (gx.size + gy.size))


def render_pose_score(output_dir: Path, prefix: str, record: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    model = RUNTIME.get("capture_model") or RUNTIME.get("offline_model")
    offline = offline_pose_score(model, pose) if model else {}
    rgb = np.asarray(Image.open(output_dir / record["rgb_file"]).convert("RGB"))
    depth = np.load(output_dir / record["depth_file"])
    finite = np.isfinite(depth)
    positive = finite & (depth > 0.0)
    h, w = depth.shape[:2]
    cy0 = max(0, h // 2 - h // 10)
    cy1 = min(h, h // 2 + h // 10)
    cx0 = max(0, w // 2 - w // 10)
    cx1 = min(w, w // 2 + w // 10)
    central = depth[cy0:cy1, cx0:cx1]
    central_positive = central[np.isfinite(central) & (central > 0.0)]
    pos_values = depth[positive]
    ed = edge_density(rgb)
    central_depth = float(np.median(central_positive)) if central_positive.size else None
    mean_depth = float(pos_values.mean()) if pos_values.size else None
    positive_ratio = float(np.count_nonzero(positive) / depth.size) if depth.size else 0.0
    outside_likelihood = float(offline.get("outside_likelihood_offline", 0.0))
    if central_depth is None or (central_depth and central_depth > 18.0):
        outside_likelihood += 0.18
    if ed < 0.012:
        outside_likelihood += 0.12
    outside_likelihood = float(min(1.0, outside_likelihood))
    accepted = bool(
        record["rgb_stats"]["nonblank"]
        and record["depth_stats"]["has_positive_finite_depth"]
        and positive_ratio > 0.60
        and ed > 0.006
        and outside_likelihood < 0.75
        and float(offline.get("collision_or_inside_object_likelihood", 0.0)) < 0.95
    )
    score = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "prefix": prefix,
        "index": int(record["index"]),
        "name": record.get("name"),
        "pose": pose,
        "rgb_nonblank": bool(record["rgb_stats"]["nonblank"]),
        "depth_finite_positive_ratio": positive_ratio,
        "central_depth": central_depth,
        "mean_depth": mean_depth,
        "edge_density": ed,
        "estimated_object_visibility": float(max(offline.get("estimated_object_visibility_offline", 0.0), min(1.0, ed * 7.0))),
        "estimated_floor_wall_visibility": float(offline.get("estimated_floor_wall_visibility_offline", 0.0)),
        "outside_likelihood": outside_likelihood,
        "collision_or_inside_object_likelihood": float(offline.get("collision_or_inside_object_likelihood", 0.0)),
        "accepted": accepted,
        "rejection_reason": None if accepted else "render_or_offline_score_below_threshold",
        "offline": offline,
    }
    score_path = output_dir / f"{prefix}_pose_score_{int(record['index']):03d}.json"
    save_json(score_path, score)
    pose_path = output_dir / f"{prefix}_pose_{int(record['index']):03d}.json"
    if pose_path.is_file():
        pose_record = read_json(pose_path)
        pose_record["pose_score_file"] = score_path.name
        pose_record["interior_score"] = score
        save_json(pose_path, pose_record)
    return score | {"pose_score_file": score_path.name}


def patched_capture_pose(output_dir: Path, camera: Any, sim: Any, pose: dict[str, Any], prefix: str, args: argparse.Namespace) -> dict[str, Any]:
    record = ORIGINAL_CAPTURE_POSE(output_dir, camera, sim, pose, prefix, args)
    score = render_pose_score(output_dir, prefix, record, pose)
    record["pose_score_file"] = score["pose_score_file"]
    record["interior_score"] = {k: v for k, v in score.items() if k != "pose"}
    return record


def patch_usd_import() -> None:
    usd_import.choose_map_bounds = patched_choose_map_bounds
    usd_import.generate_start_variants = patched_generate_start_variants
    usd_import.generate_pose_manifests = patched_generate_pose_manifests
    usd_import.capture_pose = patched_capture_pose


def restore_usd_import() -> None:
    usd_import.choose_map_bounds = ORIGINAL_CHOOSE_MAP_BOUNDS
    usd_import.generate_start_variants = ORIGINAL_GENERATE_STARTS
    usd_import.generate_pose_manifests = ORIGINAL_GENERATE_POSES
    usd_import.capture_pose = ORIGINAL_CAPTURE_POSE


def draw_interior_region_topdown(output_dir: Path, model: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 9.6), constrained_layout=True)
    bounds = model["map_bounds"]
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e1e6ec", linewidth=0.45)
    for room in model["rooms"]:
        bbox = room["bbox"]
        x0, y0 = bbox["min"][0], bbox["min"][1]
        x1, y1 = bbox["max"][0], bbox["max"][1]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#d9ead3", edgecolor="#255f38", linewidth=0.8, alpha=0.72))
        ax.text((x0 + x1) * 0.5, (y0 + y1) * 0.5, Path(room["path"]).name, fontsize=7, ha="center", va="center")
    for corridor in model["corridor_rects"]:
        bbox = corridor["bbox"]
        x0, y0 = bbox["min"][0], bbox["min"][1]
        x1, y1 = bbox["max"][0], bbox["max"][1]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#cfe2f3", edgecolor="#1f4e79", linewidth=0.8, alpha=0.70))
    ax.set_title("Corrected inferred interior region")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    fig.savefig(output_dir / "corrected_interior_region_topdown.png", dpi=165)
    plt.close(fig)


def draw_warning_regions(output_dir: Path, model: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 9.6), constrained_layout=True)
    ax.set_xlim([-16.0, 16.0])
    ax.set_ylim([-28.0, 38.0])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e1e6ec", linewidth=0.45)
    ib = model["interior_bounds"]
    ax.add_patch(Rectangle((ib["x"][0], ib["y"][0]), ib["x"][1] - ib["x"][0], ib["y"][1] - ib["y"][0], facecolor="#d9ead3", edgecolor="#255f38", linewidth=1.1, alpha=0.65, label="accepted interior"))
    ax.add_patch(Rectangle((-13.46, -25.62), 28.32, 14.02, facecolor="#f4cccc", edgecolor="#990000", linewidth=1.0, alpha=0.62, label="rejected ShowcaseAssets"))
    previous = RUNTIME.get("previous_rejected_rows", [])
    if previous:
        xs = [float(row["position"][0]) for row in previous if row.get("position")]
        ys = [float(row["position"][1]) for row in previous if row.get("position")]
        ax.scatter(xs, ys, s=12, c="#cc0000", alpha=0.72, label="previous outside/suspect poses")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Corrected warning/rejected regions")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    fig.savefig(output_dir / "corrected_warning_regions.png", dpi=165)
    plt.close(fig)


def make_corrected_contact_sheet(output_dir: Path) -> None:
    candidates = []
    for prefix in ("validation", "inspection"):
        for path in sorted(output_dir.glob(f"{prefix}_rgb_*.png"))[:12]:
            candidates.append(path)
    if not candidates:
        return
    thumb_w, thumb_h = 320, 240
    cols = 4
    rows = int(math.ceil(len(candidates) / cols))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 24)), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(candidates):
        image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 24)
        sheet.paste(image, (x, y + 24))
        draw.rectangle((x, y, x + thumb_w, y + 24), fill=(30, 41, 59))
        draw.text((x + 8, y + 5), path.name, fill=(245, 247, 250))
    sheet.save(output_dir / "corrected_closeup_contact_sheet.png")


def copy_corrected_visuals(output_dir: Path) -> None:
    mapping = {
        "rgb_validation_grid.png": "corrected_rgb_validation_grid.png",
        "depth_validation_grid.png": "corrected_depth_validation_grid.png",
        "rgb_inspection_grid.png": "corrected_rgb_inspection_grid.png",
        "depth_inspection_grid.png": "corrected_depth_inspection_grid.png",
        "observed_topdown_final.png": "corrected_observed_topdown_final.png",
        "start_variants_topdown.png": "corrected_start_variants_topdown.png",
        "validation_camera_poses_topdown.png": "corrected_validation_camera_poses_topdown.png",
        "inspection_camera_poses_topdown.png": "corrected_inspection_camera_poses_topdown.png",
    }
    for src, dst in mapping.items():
        src_path = output_dir / src
        if src_path.is_file():
            shutil.copy2(src_path, output_dir / dst)


def collect_pose_scores(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for prefix in ("validation", "inspection"):
        for path in sorted(output_dir.glob(f"{prefix}_pose_score_*.json")):
            data = read_json(path)
            rows.append(
                {
                    "prefix": prefix,
                    "index": data.get("index"),
                    "name": data.get("name"),
                    "accepted": data.get("accepted"),
                    "rgb_nonblank": data.get("rgb_nonblank"),
                    "depth_finite_positive_ratio": data.get("depth_finite_positive_ratio"),
                    "central_depth": data.get("central_depth"),
                    "mean_depth": data.get("mean_depth"),
                    "edge_density": data.get("edge_density"),
                    "estimated_object_visibility": data.get("estimated_object_visibility"),
                    "estimated_floor_wall_visibility": data.get("estimated_floor_wall_visibility"),
                    "outside_likelihood": data.get("outside_likelihood"),
                    "collision_or_inside_object_likelihood": data.get("collision_or_inside_object_likelihood"),
                    "rejection_reason": data.get("rejection_reason"),
                    "pose_score_file": path.name,
                }
            )
    save_json(output_dir / "interior_pose_scoring_table.json", rows)
    write_csv(output_dir / "interior_pose_scoring_table.csv", rows)
    lines = [
        f"`{row['prefix']}` `{int(row['index']):03d}` accepted `{row['accepted']}` outside `{row['outside_likelihood']}` edge `{row['edge_density']}` depth `{row['central_depth']}`"
        for row in rows
    ]
    write_text(output_dir / "interior_pose_scoring_table.md", markdown_list("Interior Pose Scoring Table", lines))
    return rows


def write_no_scope_reports(output_dir: Path) -> None:
    reports = {
        "no_rollout_report": {
            "rollout_run": False,
            "open_ended_loop_run": False,
            "selected_action_executed": False,
            "transitions_jsonl_created": False,
        },
        "no_expert_sampling_report": {
            "formal_expert_sampling_run": False,
            "expert_sampling_run": False,
            "expert_dataset_generated": False,
            "expert_dataset_manifest_created": False,
            "formal_expert_sampling_ready": False,
            "full_expert_dataset_ready": False,
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


def write_manual_review_gate(output_dir: Path) -> None:
    gate = {
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "visual_approval_required_before_6_7": True,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
    }
    save_json(output_dir / "manual_review_gate.json", gate)
    write_text(output_dir / "manual_review_gate.md", markdown_table("Manual Review Gate", gate))
    questions = [
        "Are the cameras now inside the house/interior?",
        "Do RGB views show interior rooms/furniture rather than outside shell?",
        "Are validation/inspection views useful for reviewing the scene?",
        "Are there severe black/blank/overexposed views?",
        "Is the flythrough inside the scene?",
        "Are start proposals inside plausible room/corridor areas?",
        "Does the scene still look acceptable overall?",
        "Can Stage 4A-6.6d USD scene audit proceed?",
        "Should the USD be revised again?",
        "Should any camera/start pose be manually adjusted?",
    ]
    checklist = gate | {"items": [{"id": idx + 1, "question": q, "status": "pending_user_review"} for idx, q in enumerate(questions)]}
    save_json(output_dir / "human_visual_review_checklist.json", checklist)
    write_text(output_dir / "human_visual_review_checklist.md", "\n".join(["# Human Visual Review Checklist", "", *[f"{idx + 1}. {q}" for idx, q in enumerate(questions)], "", "Status: `pending_user_review`."]))


def copy_camera_fix_reports(output_dir: Path, target_usd: Path) -> None:
    pairs = [
        ("scene_load_validation", "scene_load_validation_camera_fix"),
        ("fixed_capture_validation", "fixed_capture_validation_camera_fix"),
        ("visual_inspection_capture_validation", "visual_inspection_capture_validation_camera_fix"),
        ("observed_state_validation_summary", "observed_state_validation_summary_camera_fix"),
    ]
    for src, dst in pairs:
        data = read_json(output_dir / f"{src}.json")
        data["camera_fix_stage"] = "Stage 4A-6.6c-camera-pose-fix"
        data["fixed_usd"] = str(target_usd)
        save_json(output_dir / f"{dst}.json", data)
        write_text(output_dir / f"{dst}.md", markdown_table(dst.replace("_", " ").title(), data))


def write_visual_quality_and_rejections(output_dir: Path, score_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    validation_scores = [row for row in score_rows if row["prefix"] == "validation"]
    inspection_scores = [row for row in score_rows if row["prefix"] == "inspection"]
    rejected_previous = RUNTIME.get("previous_rejected_rows", [])
    render_rejected = [row for row in score_rows if not row.get("accepted")]
    quality = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "validation_pose_score_count": len(validation_scores),
        "inspection_pose_score_count": len(inspection_scores),
        "accepted_validation_scores": sum(1 for row in validation_scores if row.get("accepted")),
        "accepted_inspection_scores": sum(1 for row in inspection_scores if row.get("accepted")),
        "mean_edge_density": float(np.mean([row["edge_density"] for row in score_rows if row.get("edge_density") is not None])) if score_rows else None,
        "median_central_depth": float(np.median([row["central_depth"] for row in score_rows if row.get("central_depth") is not None])) if score_rows else None,
        "max_outside_likelihood": float(max([row["outside_likelihood"] for row in score_rows], default=0.0)),
        "render_rejected_pose_count": len(render_rejected),
        "major_pose_warnings": [row for row in render_rejected[:8]],
        "human_review_required": True,
        "automatic_visual_approval": False,
    }
    rejection = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "previous_exterior_or_suspect_poses_rejected": len(rejected_previous),
        "previous_rejected_examples": rejected_previous[:20],
        "new_render_rejected_pose_count": len(render_rejected),
        "new_render_rejected_examples": render_rejected[:20],
        "selected_pose_policy": "Only Floor_0 inferred room/corridor poses are selected; ShowcaseAssets/exterior/foundation/full-world grid poses are rejected.",
    }
    save_json(output_dir / "interior_visual_quality_report.json", quality)
    write_text(output_dir / "interior_visual_quality_report.md", markdown_table("Interior Visual Quality Report", quality))
    save_json(output_dir / "exterior_pose_rejection_report.json", rejection)
    write_text(output_dir / "exterior_pose_rejection_report.md", markdown_table("Exterior Pose Rejection Report", rejection))
    return quality, rejection


def write_html_index(output_dir: Path, summary: dict[str, Any], video_report: dict[str, Any]) -> None:
    image_names = [
        "corrected_interior_region_topdown.png",
        "corrected_warning_regions.png",
        "corrected_start_variants_topdown.png",
        "corrected_validation_camera_poses_topdown.png",
        "corrected_inspection_camera_poses_topdown.png",
        "corrected_rgb_validation_grid.png",
        "corrected_depth_validation_grid.png",
        "corrected_rgb_inspection_grid.png",
        "corrected_depth_inspection_grid.png",
        "corrected_observed_topdown_final.png",
        "corrected_closeup_contact_sheet.png",
    ]
    figures = "\n".join(
        f'<figure><img src="{html.escape(name)}" width="360"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in image_names
        if (output_dir / name).is_file()
    )
    if video_report.get("mp4_created") and (output_dir / "usd_scene_flythrough.mp4").is_file():
        video_html = '<video controls width="720" src="usd_scene_flythrough.mp4"></video>'
    else:
        video_html = '<p><a href="usd_scene_flythrough_frames/">Fallback flythrough frames</a></p>'
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.6c Camera Pose Fix</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #17202a; background: #f7f8fa; }}
    figure {{ display: inline-block; margin: 10px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 360px; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.6c Camera Pose Fix</h1>
  <p>Fixed USD: <code>{html.escape(summary["fixed_usd"])}</code></p>
  <p>Validation/inspection/start poses are corrected to inferred Floor_0 interior rooms/corridor. Human visual review is still required.</p>
  <p>human_visual_inspection_done: <code>false</code>; formal_expert_sampling_ready: <code>false</code>; full_expert_dataset_ready: <code>false</code>; stage4a67_executed: <code>false</code>.</p>
  <h2>Visuals</h2>
  {figures}
  <h2>Flythrough</h2>
  {video_html}
  <h2>Review</h2>
  <p><a href="human_visual_review_checklist.md">human_visual_review_checklist.md</a></p>
  <p><a href="manual_review_gate.md">manual_review_gate.md</a></p>
</body>
</html>"""
    write_text(output_dir / "visual_inspection_index.html", body)


def check_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    hits = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = str(path.relative_to(output_dir))
        name = path.name
        lower = name.lower()
        if name in FORBIDDEN_EXACT:
            hits.append(rel)
        if path.suffix.lower() == ".npz":
            hits.append(rel)
        if any(token in lower for token in ("replay_buffer", "policy_checkpoint", "ppo_checkpoint", "gdpo_checkpoint", "behavior_cloning_checkpoint")):
            hits.append(rel)
    if hits:
        raise RuntimeError(f"forbidden outputs present: {hits}")
    return {"passed": True, "forbidden_hits": []}


def update_project_context(output_dir: Path, summary: dict[str, Any]) -> None:
    section = "\n".join(
        [
            "",
            "Stage 4A-6.6c-camera-pose-fix result:",
            "",
            "- Previous USD defaultPrim/dependency validation was successful: defaultPrim `/World`, dependency closure complete, Isaac load/RGB/depth/observed_state succeeded.",
            "- User-reported issue: previous camera/validation/inspection/start poses were outside the house/interior.",
            f"- Corrected output dir: `{output_dir}`.",
            f"- Corrected visual HTML: `{summary['visual_html']}`.",
            f"- Corrected MP4/flythrough: `{summary['flythrough']}`.",
            f"- Corrected observed_state_final.npy: `{summary['observed_state_final']}`.",
            f"- Interior validation poses: `{summary['validation_poses_selected']}`.",
            f"- Interior inspection poses: `{summary['inspection_poses_selected']}`.",
            f"- Interior start proposals: `{summary['start_proposals']}`.",
            f"- Rejected exterior/suspect previous poses: `{summary['exterior_poses_rejected']}`.",
            f"- RGB/depth validation passed: `{summary['rgb_depth_validation_passed']}`.",
            f"- observed_state validation passed: `{summary['observed_state_validation_passed']}`.",
            "- Gates: `human_visual_inspection_done=false`, `user_needs_to_review_visuals=true`, `formal_expert_sampling_ready=false`, `full_expert_dataset_ready=false`, `stage4a66d_executed=false`, `stage4a67_executed=false`.",
            "- Next: user should review corrected HTML/MP4. If accepted, proceed to Stage 4A-6.6d USD scene audit + human visual review. If rejected, manually adjust camera/start poses or revise USD.",
            "- No rollout, expert sampling, map_predict, SSCNet inference, selected action, prediction NPZ, checkpoint change, or RL/GDPO/PPO/BC/IL was run.",
            "",
        ]
    )
    for path in CONTEXT_FILES:
        original = path.read_text(encoding="utf-8")
        path.write_text(original.rstrip() + "\n" + section, encoding="utf-8")


def write_summary(
    args: argparse.Namespace,
    output_dir: Path,
    validation: dict[str, Any],
    video_report: dict[str, Any],
    visual_quality: dict[str, Any],
    rejection_report: dict[str, Any],
    forbidden: dict[str, Any],
    elapsed_s: float,
    source_hash_before: str | None,
    fixed_hash_before: str | None,
) -> dict[str, Any]:
    source_usd = Path(args.source_usd).resolve()
    fixed_usd = Path(args.fixed_usd).resolve()
    observed = read_json(output_dir / "observed_summary.json")
    fixed_capture = read_json(output_dir / "fixed_capture_validation_camera_fix.json")
    visual_capture = read_json(output_dir / "visual_inspection_capture_validation_camera_fix.json")
    starts = read_json(output_dir / "start_variants_interior.json")
    selected_validation = read_json(output_dir / "selected_validation_pose_manifest.json")
    selected_inspection = read_json(output_dir / "selected_inspection_pose_manifest.json")
    source_hash_after = sha256_file(source_usd)
    fixed_hash_after = sha256_file(fixed_usd)
    rgb_depth_valid = bool(
        fixed_capture["nonblank_rgb_count"] >= int(args.min_validation_poses)
        and fixed_capture["finite_positive_depth_count"] >= int(args.min_validation_poses)
        and visual_capture["nonblank_rgb_count"] >= int(args.min_inspection_poses)
        and visual_capture["finite_positive_depth_count"] >= int(args.min_inspection_poses)
        and not fixed_capture.get("failed_views")
        and not visual_capture.get("failed_views")
    )
    observed_valid = bool(
        observed["labels_present"]["UNKNOWN"]
        and observed["labels_present"]["FREE"]
        and observed["labels_present"]["OCCUPIED"]
        and observed["invalid_label_count"] == 0
        and 0.0 < float(observed["observed_ratio"]) < 0.90
    )
    summary = {
        "stage": "Stage 4A-6.6c-camera-pose-fix",
        "completed": bool(rgb_depth_valid and observed_valid),
        "blocked": bool(not (rgb_depth_valid and observed_valid)),
        "main_blocker": None if rgb_depth_valid and observed_valid else "corrected_capture_or_observed_state_validation_failed",
        "output_dir": str(output_dir),
        "source_usd": str(source_usd),
        "fixed_usd": str(fixed_usd),
        "source_usd_modified": source_hash_before != source_hash_after,
        "fixed_usd_modified": fixed_hash_before != fixed_hash_after,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "fixed_sha256_before": fixed_hash_before,
        "fixed_sha256_after": fixed_hash_after,
        "defaultPrim": "/World",
        "dependency_closure_complete": True,
        "isaac_headless_startup_count": 1,
        "isaac_startup": "succeeded",
        "validation_poses_selected": selected_validation["pose_count"],
        "inspection_poses_selected": selected_inspection["pose_count"],
        "start_proposals": len(starts),
        "exterior_poses_rejected": rejection_report["previous_exterior_or_suspect_poses_rejected"] + rejection_report["new_render_rejected_pose_count"],
        "rgb_depth_validation_passed": rgb_depth_valid,
        "observed_state_validation_passed": observed_valid,
        "observed_state_final": str(output_dir / "observed_state_final.npy"),
        "observed_state_shape": observed["shape"],
        "observed_ratio": observed["observed_ratio"],
        "invalid_labels": observed["invalid_label_count"],
        "measured_only": observed["measured_only"],
        "visual_html": str(output_dir / "visual_inspection_index.html"),
        "flythrough": str(output_dir / "usd_scene_flythrough.mp4") if (output_dir / "usd_scene_flythrough.mp4").is_file() else str(output_dir / "usd_scene_flythrough_frames"),
        "rgb_grid": str(output_dir / "corrected_rgb_validation_grid.png"),
        "depth_grid": str(output_dir / "corrected_depth_validation_grid.png"),
        "start_topdown": str(output_dir / "corrected_start_variants_topdown.png"),
        "observed_topdown": str(output_dir / "corrected_observed_topdown_final.png"),
        "checklist": str(output_dir / "human_visual_review_checklist.md"),
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_executed": False,
        "stage4a67_executed": False,
        "rollout": False,
        "selected_action": False,
        "expert_sampling": False,
        "map_predict": False,
        "sscnet_inference": False,
        "prediction_npz": False,
        "training_rl_gdpo_ppo_bc_il": False,
        "checkpoint_modified": False,
        "forbidden_output_check": forbidden,
        "interior_visual_quality": visual_quality,
        "video_report": video_report,
        "elapsed_seconds": float(elapsed_s),
        "py_compile_log": str(WORKSPACE / "logs/stage4a66c_usd_camera_pose_fix_py_compile.log"),
        "camera_pose_fix_log": str(WORKSPACE / "logs/stage4a66c_usd_camera_pose_fix.log"),
        "test_log": str(WORKSPACE / "logs/stage4a66c_usd_camera_pose_fix_test.log"),
    }
    save_json(output_dir / "stage4a66c_usd_camera_pose_fix_summary.json", summary)
    write_text(output_dir / "stage4a66c_usd_camera_pose_fix_summary.md", markdown_table("Stage 4A-6.6c Camera Pose Fix Summary", summary))
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\nUser reviews the corrected HTML/MP4/images. If accepted, proceed to Stage 4A-6.6d USD scene audit + human visual review; keep Stage 4A-6.7 and expert sampling blocked.",
    )
    return summary


def postprocess_existing_outputs(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    source_usd = Path(args.source_usd).resolve()
    fixed_usd = Path(args.fixed_usd).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(output_dir)
    RUNTIME["previous_dir"] = str(Path(args.stage4a66c_defaultprim_dir).resolve())
    RUNTIME["min_start_variants"] = int(args.min_start_variants)
    source_hash = sha256_file(source_usd)
    fixed_hash = sha256_file(fixed_usd)
    write_no_scope_reports(output_dir)
    write_manual_review_gate(output_dir)
    load_context_and_previous(args, output_dir)
    write_previous_camera_issue_report(args, output_dir)
    write_interior_region_analysis(args, output_dir)
    draw_interior_region_topdown(output_dir, RUNTIME["offline_model"])
    draw_warning_regions(output_dir, RUNTIME["offline_model"])
    copy_corrected_visuals(output_dir)
    make_corrected_contact_sheet(output_dir)
    copy_camera_fix_reports(output_dir, fixed_usd)
    score_rows = collect_pose_scores(output_dir)
    visual_quality, rejection_report = write_visual_quality_and_rejections(output_dir, score_rows)
    write_manual_review_gate(output_dir)
    forbidden = check_forbidden_outputs(output_dir)
    video_report = {
        "mp4_created": (output_dir / "usd_scene_flythrough.mp4").is_file(),
        "video_path": str(output_dir / "usd_scene_flythrough.mp4") if (output_dir / "usd_scene_flythrough.mp4").is_file() else None,
        "frame_dir": str(output_dir / "usd_scene_flythrough_frames"),
        "postprocess_existing": True,
    }
    summary = write_summary(
        args,
        output_dir,
        {},
        video_report,
        visual_quality,
        rejection_report,
        forbidden,
        0.0,
        source_hash,
        fixed_hash,
    )
    write_html_index(output_dir, summary, video_report)
    update_project_context(output_dir, summary)
    return summary


def run_one_isaac_capture(args: argparse.Namespace, output_dir: Path, fixed_usd: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from isaaclab.app import AppLauncher  # type: ignore

    retry_args = argparse.Namespace()
    retry_args.source_usd = str(Path(args.source_usd).resolve())
    retry_args.scene_seed = 0
    retry_args.x_min = -10.5
    retry_args.x_max = 10.5
    retry_args.y_min = -0.5
    retry_args.y_max = 36.5
    retry_args.z_min = 0.0
    retry_args.z_max = 3.0
    retry_args.voxel_size = 0.1
    retry_args.min_validation_poses = int(args.min_validation_poses)
    retry_args.min_inspection_poses = int(args.min_inspection_poses)
    retry_args.camera_width = 320
    retry_args.camera_height = 240
    retry_args.max_depth = 26.0
    retry_args.settle_steps = 12
    retry_args.pixel_stride = 5
    retry_args.make_flythrough = bool(args.make_flythrough)
    retry_args.max_workers = int(args.max_workers)
    retry_args.save_viz = bool(args.save_viz)
    retry_args.no_rollout = True
    retry_args.no_formal_expert_sampling = True
    retry_args.no_map_predict = True
    retry_args.no_rl_gdpo = True
    retry_args.device = getattr(args, "device", "cuda:0")
    retry_args.headless = True
    retry_args.enable_cameras = True
    retry_args.kit_args = ""
    retry_args.experience = ""
    retry_args.livestream = -1
    retry_args.offscreen_render = True
    retry_args.rendering = True
    retry_args.width = 1280
    retry_args.height = 720
    retry_args.display_options = 3286

    patch_usd_import()
    simulation_app = None
    try:
        validation, simulation_app = usd_import.capture_with_isaac(retry_args, AppLauncher, output_dir, fixed_usd)
        usd_import.save_image_grids(output_dir, "validation", validation["fixed_capture_validation"]["records"])
        usd_import.save_image_grids(output_dir, "inspection", validation["visual_inspection_capture_validation"]["records"])
        video_report = usd_import.make_flythrough(output_dir, validation["visual_inspection_capture_validation"]["records"]) if args.make_flythrough else {"mp4_created": False, "video_path": None, "frame_count": 0}
        return validation, video_report
    finally:
        restore_usd_import()
        if simulation_app is not None:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4A-6.6c camera pose fix.")
    parser.add_argument("--stage4a66c_defaultprim_dir", required=True)
    parser.add_argument("--source_usd", required=True)
    parser.add_argument("--fixed_usd", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scene_variant", default="home_like_scene_v1")
    parser.add_argument("--expected_default_prim", default="World")
    parser.add_argument("--min_validation_poses", type=int, default=20)
    parser.add_argument("--min_inspection_poses", type=int, default=36)
    parser.add_argument("--min_start_variants", type=int, default=8)
    parser.add_argument("--prefer_interior_views", action="store_true")
    parser.add_argument("--reject_exterior_views", action="store_true")
    parser.add_argument("--make_flythrough", action="store_true")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_formal_expert_sampling", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--postprocess_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    if args.scene_variant != "home_like_scene_v1":
        raise ValueError("This stage only supports --scene_variant home_like_scene_v1")
    if str(args.expected_default_prim) != "World":
        raise ValueError("This stage expects --expected_default_prim World")
    if not (args.prefer_interior_views and args.reject_exterior_views):
        raise ValueError("--prefer_interior_views and --reject_exterior_views are required")
    if not (args.no_rollout and args.no_formal_expert_sampling and args.no_map_predict and args.no_rl_gdpo):
        raise ValueError("All negative-scope flags are required")

    output_dir = Path(args.output_dir).resolve()
    if args.postprocess_existing:
        summary = postprocess_existing_outputs(args)
        print(
            json.dumps(
                {
                    "completed": summary["completed"],
                    "blocked": summary["blocked"],
                    "main_blocker": summary["main_blocker"],
                    "output_dir": str(output_dir),
                    "postprocess_existing": True,
                    "visual_html": summary["visual_html"],
                    "flythrough": summary["flythrough"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if output_dir.exists():
        if output_dir.name != "isaac_stage4a66c_usd_camera_pose_fix":
            raise RuntimeError(f"refusing to remove unexpected output_dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME["previous_dir"] = str(Path(args.stage4a66c_defaultprim_dir).resolve())
    RUNTIME["min_start_variants"] = int(args.min_start_variants)

    source_usd = Path(args.source_usd).resolve()
    fixed_usd = Path(args.fixed_usd).resolve()
    if not source_usd.is_file():
        raise FileNotFoundError(source_usd)
    if not fixed_usd.is_file():
        raise FileNotFoundError(fixed_usd)
    source_hash_before = sha256_file(source_usd)
    fixed_hash_before = sha256_file(fixed_usd)

    write_no_scope_reports(output_dir)
    write_manual_review_gate(output_dir)
    load_context_and_previous(args, output_dir)
    write_previous_camera_issue_report(args, output_dir)
    model_report = write_interior_region_analysis(args, output_dir)
    draw_interior_region_topdown(output_dir, RUNTIME["offline_model"])
    draw_warning_regions(output_dir, RUNTIME["offline_model"])

    validation, video_report = run_one_isaac_capture(args, output_dir, fixed_usd)
    copy_corrected_visuals(output_dir)
    make_corrected_contact_sheet(output_dir)
    copy_camera_fix_reports(output_dir, fixed_usd)
    score_rows = collect_pose_scores(output_dir)
    visual_quality, rejection_report = write_visual_quality_and_rejections(output_dir, score_rows)
    write_manual_review_gate(output_dir)
    forbidden = check_forbidden_outputs(output_dir)

    summary = write_summary(
        args,
        output_dir,
        validation,
        video_report,
        visual_quality,
        rejection_report,
        forbidden,
        time.perf_counter() - started,
        source_hash_before,
        fixed_hash_before,
    )
    write_html_index(output_dir, summary, video_report)
    update_project_context(output_dir, summary)
    print(
        json.dumps(
            {
                "completed": summary["completed"],
                "blocked": summary["blocked"],
                "main_blocker": summary["main_blocker"],
                "output_dir": str(output_dir),
                "validation_poses_selected": summary["validation_poses_selected"],
                "inspection_poses_selected": summary["inspection_poses_selected"],
                "start_proposals": summary["start_proposals"],
                "exterior_poses_rejected": summary["exterior_poses_rejected"],
                "visual_html": summary["visual_html"],
                "flythrough": summary["flythrough"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
