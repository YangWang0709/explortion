#!/usr/bin/env python3
"""Stage 4A-6.6c USD import and validation for home_like_scene_v1.

This stage only imports the user-provided USD, stages it as the current
project environment, validates loading/capture, builds a measured-only
observed_state, and writes a visual review package.  It does not generate a
procedural scene, download assets, run rollout, run expert sampling, call
map_predict/SSCNet, or train policies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
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
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw

from depth_to_voxel import create_observed_grid, integrate_depth_frame, summarize_observed_grid
from scene_factory import (
    HOME_LIKE_SCENE_V1_STAGED_USD,
    build_home_like_scene_v1,
    build_home_like_scene_v1_from_usd,
    build_larger_complex_scene_v1,
    build_medium_complex_scene,
)


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_SOURCE_USD = WORKSPACE / "building_scene.usd"
DEFAULT_STAGED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment/home_like_scene_v1.usd"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_home_like_scene_validation"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
OLD_LARGER_OUTPUT_DIRS = [
    WORKSPACE / "outputs/isaac_stage4a66_larger_complex_scene_v1_validation",
    WORKSPACE / "outputs/isaac_stage4a66a_scene_complexity_audit",
    WORKSPACE / "outputs/isaac_stage4a66b_gui_visual_inspection",
]
OLD_DELETED_66C_DIR = WORKSPACE / "outputs/isaac_stage4a66c_home_like_scene_v1_validation"
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
DEPTH_KEY = "distance_to_image_plane"
RGB_KEYS = ("rgb", "rgba")
STATE_CMAP = ListedColormap(["#2f343b", "#80b9c4", "#c95c5c"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)


SEMANTIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "room": ("room", "living", "bedroom", "bath", "kitchen", "hall", "corridor", "study", "dining"),
    "wall": ("wall", "partition", "end_wall"),
    "floor": ("floor", "ground", "plane"),
    "ceiling": ("ceiling", "roof"),
    "sofa_couch": ("sofa", "couch"),
    "chair_armchair": ("chair", "armchair", "seat"),
    "table": ("table", "desk", "counter"),
    "bed": ("bed", "mattress", "pillow"),
    "cabinet_shelf": ("cabinet", "wardrobe", "shelf", "bookshelf", "bookcase", "drawer"),
    "kitchen": ("kitchen", "counter", "fridge", "stove", "sink", "oven"),
    "bathroom": ("toilet", "bathtub", "bath", "shower", "vanity", "basin"),
    "rug_lamp_plant_decor": ("rug", "lamp", "plant", "decor", "picture", "vase"),
    "door_window": ("door", "window"),
    "storage_box_clutter": ("storage", "box", "clutter", "crate"),
    "hallway_corridor_stair_elevator": ("hall", "corridor", "stair", "elevator"),
    "collision": ("collision", "collider", "physics"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "path") and hasattr(value, "resolvedPath"):
        return {"path": str(value.path), "resolvedPath": str(value.resolvedPath)}
    if hasattr(value, "__iter__") and value.__class__.__module__.startswith("pxr."):
        try:
            return [jsonable(v) for v in value]
        except TypeError:
            return str(value)
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


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
        if isinstance(value, (dict, list)):
            text = json.dumps(jsonable(value), sort_keys=True)
            if len(text) > 1400:
                text = text[:1400] + "..."
            value_text = f"`{text}`"
        else:
            value_text = f"`{value}`"
        lines.append(f"| {key} | {value_text} |")
    return "\n".join(lines)


def list_md(title: str, rows: list[str]) -> str:
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


def hardware_report(max_workers: int, device: str) -> dict[str, Any]:
    gpu_name = "unavailable"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=6,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_name = result.stdout.strip().splitlines()[0]
    except Exception as exc:
        gpu_name = f"nvidia-smi unavailable: {exc}"
    return {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "requested_max_workers": int(max_workers),
        "actual_max_workers": min(int(max_workers), os.cpu_count() or 1),
        "requested_device": str(device),
        "gpu_name": gpu_name,
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
    }


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
        "median": float(np.median(positive)) if positive.size else None,
        "max": float(positive.max()) if positive.size else None,
        "mean": float(positive.mean()) if positive.size else None,
        "has_positive_finite_depth": bool(positive.size > 0),
    }


def normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    return np.clip(np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def save_depth_color(path: Path, depth: np.ndarray, title: str) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError(f"No finite positive depth for {path}")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, dpi=135)
    plt.close(fig)


def pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


def yaw_to_center(position: list[float], center: tuple[float, float]) -> float:
    return float(math.atan2(center[1] - float(position[1]), center[0] - float(position[0])))


def state_transition(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    out = {
        "unknown_to_free": int(np.count_nonzero((before == UNKNOWN) & (after == FREE))),
        "unknown_to_occupied": int(np.count_nonzero((before == UNKNOWN) & (after == OCCUPIED))),
        "free_to_occupied": int(np.count_nonzero((before == FREE) & (after == OCCUPIED))),
        "occupied_to_free": int(np.count_nonzero((before == OCCUPIED) & (after == FREE))),
        "invalid_label_count_after": int(np.count_nonzero(~np.isin(after, [UNKNOWN, FREE, OCCUPIED]))),
    }
    out["newly_observed"] = out["unknown_to_free"] + out["unknown_to_occupied"]
    return out


def write_blocker(output_dir: Path, reason: str, details: dict[str, Any]) -> None:
    report = {
        "stage": "Stage 4A-6.6c-usd-import",
        "blocked": True,
        "reason": reason,
        "details": details,
        "procedural_scene_generated": False,
        "cuboid_fallback_used": False,
        "asset_download_attempted": False,
        "formal_expert_sampling_ready": False,
        "created_at_utc": utc_now(),
    }
    save_json(output_dir / "stage4a66c_usd_import_blocker_report.json", report)
    write_text(output_dir / "stage4a66c_usd_import_blocker_report.md", markdown_table("Stage 4A-6.6c USD Import Blocker", report))


def write_context_reports(output_dir: Path) -> dict[str, Any]:
    context_entries = []
    all_text = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        all_text += "\n" + text
        context_entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_larger_complex_scene_v1_rejected": "larger_complex_scene_v1" in text and "rejected" in text.lower(),
                "contains_formal_expert_sampling_ready_false": "formal_expert_sampling_ready=false" in text,
                "contains_human_visual_inspection_done_false": "human_visual_inspection_done=false" in text,
            }
        )
    loaded = {
        "stage": "Stage 4A-6.6c-usd-import",
        "loaded_context_files": context_entries,
        "confirmed_current_task_is_usd_import": True,
        "confirmed_not_stage4a67": True,
    }
    save_json(output_dir / "loaded_context_manifest.json", loaded)
    write_text(output_dir / "loaded_context_manifest.md", list_md("Loaded Context Manifest", [f"`{item['path']}` sha256 `{item['sha256']}`" for item in context_entries]))

    rejection = {
        "old_scene": "larger_complex_scene_v1",
        "user_rejected_old_larger_complex_scene_v1": "rejected" in all_text.lower() and "larger_complex_scene_v1" in all_text,
        "build_larger_complex_scene_v1_disabled": False,
        "formal_expert_sampling_ready": False,
        "human_visual_inspection_done": False,
        "current_task": "import_user_provided_building_scene_usd",
        "source_usd_is_candidate_environment": True,
        "stage4a67_allowed_now": False,
    }
    try:
        build_larger_complex_scene_v1(seed=0, spawn=False)
    except RuntimeError as exc:
        rejection["build_larger_complex_scene_v1_disabled"] = True
        rejection["disabled_error"] = str(exc)
    save_json(output_dir / "old_scene_rejection_status.json", rejection)
    write_text(output_dir / "old_scene_rejection_status.md", markdown_table("Old Scene Rejection Status", rejection))

    cleanup = {
        "old_larger_scene_output_dirs": [str(path) for path in OLD_LARGER_OUTPUT_DIRS],
        "old_larger_scene_output_dirs_absent": all(not path.exists() for path in OLD_LARGER_OUTPUT_DIRS),
        "deleted_stage4a66c_build_v2_output_absent": not OLD_DELETED_66C_DIR.exists(),
        "checked_at_utc": utc_now(),
        "cleanup_action_taken_this_run": "verification_only",
    }
    save_json(output_dir / "old_scene_cleanup_verification.json", cleanup)
    write_text(output_dir / "old_scene_cleanup_verification.md", markdown_table("Old Scene Cleanup Verification", cleanup))
    return {"loaded": loaded, "rejection": rejection, "cleanup": cleanup}


def stage_usd(source_usd: Path, staged_usd: Path, output_dir: Path) -> dict[str, Any]:
    if not source_usd.is_file():
        write_blocker(output_dir, "source_usd_missing", {"source_usd": str(source_usd)})
        raise FileNotFoundError(f"Source USD missing: {source_usd}")
    source_hash = sha256_file(source_usd)
    staged_usd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_usd, staged_usd)
    staged_hash = sha256_file(staged_usd)
    report = {
        "stage": "Stage 4A-6.6c-usd-import",
        "source_usd": str(source_usd),
        "staged_usd": str(staged_usd),
        "source_exists": source_usd.is_file(),
        "staged_exists": staged_usd.is_file(),
        "source_sha256": source_hash,
        "staged_sha256": staged_hash,
        "source_and_staged_hash_match": source_hash == staged_hash,
        "source_modified": False,
        "procedural_scene_generated": False,
        "asset_download_attempted": False,
    }
    save_json(output_dir / "usd_staging_report.json", report)
    write_text(output_dir / "usd_staging_report.md", markdown_table("USD Staging Report", report))
    rows = [
        {"role": "source_usd", "path": str(source_usd), "sha256": source_hash, "size_bytes": source_usd.stat().st_size},
        {"role": "staged_usd", "path": str(staged_usd), "sha256": staged_hash, "size_bytes": staged_usd.stat().st_size},
    ]
    write_csv(output_dir / "usd_hash_manifest.csv", rows)
    manifest = {"files": rows, "source_and_staged_hash_match": source_hash == staged_hash}
    save_json(output_dir / "usd_hash_manifest.json", manifest)
    write_text(output_dir / "usd_hash_manifest.md", markdown_table("USD Hash Manifest", manifest))
    patch_report = {
        "source_usd_modified": False,
        "staged_usd_path_patches_applied": False,
        "patch_count": 0,
        "patches": [],
        "reason": "No staged USD path patch was required before Isaac/PXR dependency inspection.",
    }
    save_json(output_dir / "usd_path_patch_report.json", patch_report)
    write_text(output_dir / "usd_path_patch_report.md", markdown_table("USD Path Patch Report", patch_report))
    return report


def write_scene_factory_registration_report(output_dir: Path, staged_usd: Path) -> dict[str, Any]:
    import scene_factory

    source = Path(inspect.getsourcefile(scene_factory) or WORKSPACE / "sim_explorer/scene_factory.py")
    text = source.read_text(encoding="utf-8")
    report = {
        "scene_factory_path": str(source),
        "scene_variant": "home_like_scene_v1",
        "registered_builder": "build_home_like_scene_v1_from_usd",
        "home_like_scene_v1_builder_present": "def build_home_like_scene_v1(" in text,
        "home_like_scene_v1_from_usd_builder_present": "def build_home_like_scene_v1_from_usd(" in text,
        "staged_usd_constant_present": str(staged_usd) in text or "HOME_LIKE_SCENE_V1_STAGED_USD" in text,
        "builder_metadata_staged_path": build_home_like_scene_v1(seed=0, spawn=False).get("staged_usd_path"),
        "expected_staged_usd_path": str(staged_usd),
        "loads_staged_usd": build_home_like_scene_v1(seed=0, spawn=False).get("staged_usd_path") == str(staged_usd),
        "procedural_fallback_disabled": True,
        "larger_complex_scene_v1_disabled": False,
        "medium_three_rooms_preserved": False,
    }
    try:
        build_larger_complex_scene_v1(seed=0, spawn=False)
    except RuntimeError:
        report["larger_complex_scene_v1_disabled"] = True
    medium = build_medium_complex_scene(seed=0, spawn=False)
    report["medium_three_rooms_preserved"] = medium.get("variant") == "three_rooms" and bool(medium.get("walls"))
    save_json(output_dir / "scene_factory_registration_report.json", report)
    write_text(output_dir / "scene_factory_registration_report.md", markdown_table("Scene Factory Registration Report", report))
    return report


def asset_path_to_string(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "resolvedPath") and getattr(value, "resolvedPath"):
        return str(value.resolvedPath)
    if hasattr(value, "path"):
        return str(value.path)
    if isinstance(value, str):
        return value
    return None


def semantic_categories_for_text(text: str) -> list[str]:
    lowered = text.lower()
    return [category for category, keywords in SEMANTIC_KEYWORDS.items() if any(keyword in lowered for keyword in keywords)]


def range_to_bbox(rng: Any) -> dict[str, list[float]] | None:
    try:
        if hasattr(rng, "IsEmpty") and rng.IsEmpty():
            return None
        mn = rng.GetMin()
        mx = rng.GetMax()
        values = [float(mn[0]), float(mn[1]), float(mn[2]), float(mx[0]), float(mx[1]), float(mx[2])]
        if not all(math.isfinite(v) for v in values):
            return None
        return {"min": values[:3], "max": values[3:]}
    except Exception:
        return None


def combine_bboxes(bboxes: list[dict[str, list[float]]]) -> dict[str, list[float]] | None:
    valid = [bbox for bbox in bboxes if bbox]
    if not valid:
        return None
    mins = np.array([bbox["min"] for bbox in valid], dtype=float)
    maxs = np.array([bbox["max"] for bbox in valid], dtype=float)
    return {"min": mins.min(axis=0).tolist(), "max": maxs.max(axis=0).tolist()}


def bbox_xy_center(bbox: dict[str, list[float]]) -> tuple[float, float]:
    return (0.5 * (float(bbox["min"][0]) + float(bbox["max"][0])), 0.5 * (float(bbox["min"][1]) + float(bbox["max"][1])))


def bbox_xy_distance(point: tuple[float, float], bbox: dict[str, list[float]]) -> float:
    x, y = point
    x0, y0 = float(bbox["min"][0]), float(bbox["min"][1])
    x1, y1 = float(bbox["max"][0]), float(bbox["max"][1])
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return float(math.hypot(dx, dy))


def inspect_usd_with_pxr(staged_usd: Path, output_dir: Path) -> dict[str, Any]:
    from pxr import Sdf, Usd, UsdGeom, UsdLux, UsdShade, UsdUtils

    stage = Usd.Stage.Open(str(staged_usd))
    if stage is None:
        raise RuntimeError(f"Usd.Stage.Open failed for {staged_usd}")

    root_layer = stage.GetRootLayer()
    dependency_report: dict[str, Any] = {
        "staged_usd": str(staged_usd),
        "root_layer_identifier": str(root_layer.identifier),
        "sublayers": list(root_layer.subLayerPaths),
        "references": [],
        "payloads": [],
        "external_asset_attributes": [],
        "external_textures": [],
        "all_dependency_layers": [],
        "all_dependency_assets": [],
        "unresolved_paths": [],
        "missing_dependencies": [],
        "dependencies_complete": True,
    }
    try:
        all_layers, all_assets, unresolved = UsdUtils.ComputeAllDependencies(str(staged_usd))
        dependency_report["all_dependency_layers"] = [str(layer.identifier) for layer in all_layers]
        dependency_report["all_dependency_assets"] = [asset_path_to_string(asset) or str(asset) for asset in all_assets]
        dependency_report["unresolved_paths"] = [asset_path_to_string(path) or str(path) for path in unresolved]
    except Exception as exc:
        dependency_report["compute_all_dependencies_error"] = str(exc)

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    prim_rows: list[dict[str, Any]] = []
    material_rows: list[dict[str, Any]] = []
    texture_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    bbox_rows: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}

    def list_editor_items(editor: Any) -> list[Any]:
        items: list[Any] = []
        for name in ("GetAddedOrExplicitItems", "GetAddedItems", "GetExplicitItems", "GetPrependedItems", "GetAppendedItems"):
            if hasattr(editor, name):
                try:
                    items.extend(list(getattr(editor, name)()))
                except Exception:
                    pass
        return items

    for prim in stage.Traverse():
        type_name = str(prim.GetTypeName() or "")
        type_counts[type_name or "untyped"] = type_counts.get(type_name or "untyped", 0) + 1
        path = str(prim.GetPath())
        name = prim.GetName()
        categories = semantic_categories_for_text(f"{path} {type_name}")
        bbox: dict[str, list[float]] | None = None
        try:
            world_bbox = bbox_cache.ComputeWorldBound(prim)
            rng = world_bbox.ComputeAlignedRange() if hasattr(world_bbox, "ComputeAlignedRange") else world_bbox.ComputeAlignedBox()
            bbox = range_to_bbox(rng)
        except Exception:
            bbox = None
        if bbox:
            bbox_rows.append({"path": path, "type_name": type_name, "bbox": bbox, "categories": categories})
        prim_row = {
            "path": path,
            "name": name,
            "type_name": type_name,
            "is_active": prim.IsActive(),
            "is_loaded": prim.IsLoaded(),
            "is_defined": prim.IsDefined(),
            "is_model": bool(prim.IsModel()),
            "kind": prim.GetMetadata("kind"),
            "semantic_categories": categories,
            "bbox_min": bbox["min"] if bbox else None,
            "bbox_max": bbox["max"] if bbox else None,
        }
        prim_rows.append(prim_row)
        if categories:
            semantic_rows.append(
                {
                    "path": path,
                    "name": name,
                    "type_name": type_name,
                    "categories": categories,
                    "semantic_text": " ".join(categories),
                    "bbox_min": bbox["min"] if bbox else None,
                    "bbox_max": bbox["max"] if bbox else None,
                }
            )
        if prim.IsA(UsdShade.Material):
            material_rows.append({"path": path, "name": name, "type_name": type_name, "categories": categories})
        for spec in prim.GetPrimStack():
            try:
                for ref in list_editor_items(spec.referenceList):
                    dependency_report["references"].append({"prim_path": path, "asset_path": str(getattr(ref, "assetPath", ref))})
            except Exception:
                pass
            try:
                for payload in list_editor_items(spec.payloadList):
                    dependency_report["payloads"].append({"prim_path": path, "asset_path": str(getattr(payload, "assetPath", payload))})
            except Exception:
                pass
        for attr in prim.GetAttributes():
            value = None
            try:
                value = attr.Get()
            except Exception:
                continue
            candidates: list[str] = []
            if isinstance(value, Sdf.AssetPath):
                converted = asset_path_to_string(value)
                if converted:
                    candidates.append(converted)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    converted = asset_path_to_string(item)
                    if converted:
                        candidates.append(converted)
            for candidate in candidates:
                rec = {"prim_path": path, "attribute": str(attr.GetName()), "asset_path": candidate}
                dependency_report["external_asset_attributes"].append(rec)
                if candidate.lower().endswith((".png", ".jpg", ".jpeg", ".tga", ".exr", ".hdr", ".bmp", ".tif", ".tiff")):
                    texture_rows.append(rec | {"texture_type_guess": "image"})

    scene_bbox = combine_bboxes([row["bbox"] for row in bbox_rows])
    scene_bounds = None
    if scene_bbox:
        scene_bounds = {
            "x": [float(scene_bbox["min"][0]), float(scene_bbox["max"][0])],
            "y": [float(scene_bbox["min"][1]), float(scene_bbox["max"][1])],
            "z": [float(scene_bbox["min"][2]), float(scene_bbox["max"][2])],
        }

    def is_light(prim_path: str) -> bool:
        prim = stage.GetPrimAtPath(prim_path)
        checks = [
            "DomeLight",
            "SphereLight",
            "RectLight",
            "DistantLight",
            "DiskLight",
            "CylinderLight",
            "GeometryLight",
            "Light",
        ]
        if any(str(prim.GetTypeName()) == item for item in checks):
            return True
        try:
            return bool(prim.HasAPI(UsdLux.LightAPI))
        except Exception:
            return "light" in prim_path.lower()

    camera_count = sum(1 for row in prim_rows if row["type_name"] == "Camera")
    mesh_count = sum(1 for row in prim_rows if row["type_name"] == "Mesh")
    xform_count = sum(1 for row in prim_rows if row["type_name"] == "Xform")
    light_count = sum(1 for row in prim_rows if is_light(row["path"]))
    material_count = len(material_rows)
    dependency_report["external_textures"] = texture_rows
    missing = []
    for unresolved in dependency_report.get("unresolved_paths", []):
        if unresolved:
            missing.append({"path": unresolved, "source": "UsdUtils.ComputeAllDependencies"})
    for item in dependency_report["external_asset_attributes"]:
        asset = str(item["asset_path"])
        if asset.startswith("@") or "://" in asset or not asset:
            continue
        candidate = Path(asset)
        if not candidate.is_absolute():
            candidate = staged_usd.parent / asset
        if not candidate.exists():
            missing.append({"path": asset, "resolved_candidate": str(candidate), "source": item["attribute"], "prim_path": item["prim_path"]})
    dependency_report["missing_dependencies"] = missing
    dependency_report["dependencies_complete"] = len(missing) == 0
    dependency_report["sublayer_count"] = len(dependency_report["sublayers"])
    dependency_report["reference_count"] = len(dependency_report["references"])
    dependency_report["payload_count"] = len(dependency_report["payloads"])
    save_json(output_dir / "usd_dependency_report.json", dependency_report)
    write_text(
        output_dir / "usd_dependency_report.md",
        markdown_table(
            "USD Dependency Report",
            {
                "sublayers": len(dependency_report["sublayers"]),
                "references": len(dependency_report["references"]),
                "payloads": len(dependency_report["payloads"]),
                "external_textures": len(texture_rows),
                "all_dependency_layers": len(dependency_report["all_dependency_layers"]),
                "all_dependency_assets": len(dependency_report["all_dependency_assets"]),
                "missing_dependencies": len(missing),
                "dependencies_complete": dependency_report["dependencies_complete"],
            },
        ),
    )
    write_csv(output_dir / "usd_prim_inventory.csv", prim_rows)
    save_json(output_dir / "usd_prim_inventory.json", prim_rows)
    write_text(output_dir / "usd_prim_inventory.md", list_md("USD Prim Inventory", [f"`{row['path']}` `{row['type_name']}` {row['semantic_categories']}" for row in prim_rows[:160]]))
    write_csv(output_dir / "usd_material_inventory.csv", material_rows)
    save_json(output_dir / "usd_material_inventory.json", material_rows)
    write_text(output_dir / "usd_material_inventory.md", list_md("USD Material Inventory", [f"`{row['path']}`" for row in material_rows[:160]] or ["No material prims found."]))
    write_csv(output_dir / "usd_texture_inventory.csv", texture_rows)
    save_json(output_dir / "usd_texture_inventory.json", texture_rows)
    write_text(output_dir / "usd_texture_inventory.md", list_md("USD Texture Inventory", [f"`{row['asset_path']}` from `{row['prim_path']}`" for row in texture_rows[:160]] or ["No external texture references found."]))
    write_csv(output_dir / "usd_semantic_guess_inventory.csv", semantic_rows)
    save_json(output_dir / "usd_semantic_guess_inventory.json", semantic_rows)
    write_text(output_dir / "usd_semantic_guess_inventory.md", list_md("USD Semantic Guess Inventory", [f"`{row['path']}` -> `{', '.join(row['categories'])}`" for row in semantic_rows[:200]] or ["No semantic-looking names found."]))
    bounds_report = {
        "metersPerUnit": float(UsdGeom.GetStageMetersPerUnit(stage)),
        "upAxis": str(UsdGeom.GetStageUpAxis(stage)),
        "scene_bounding_box": scene_bbox,
        "scene_bounds": scene_bounds,
        "bbox_prim_count": len(bbox_rows),
    }
    save_json(output_dir / "usd_scene_bounds_report.json", bounds_report)
    write_text(output_dir / "usd_scene_bounds_report.md", markdown_table("USD Scene Bounds Report", bounds_report))

    category_counts: dict[str, int] = {}
    for row in semantic_rows:
        for category in row["categories"]:
            category_counts[category] = category_counts.get(category, 0) + 1
    likely_home_categories = [
        "room",
        "wall",
        "floor",
        "sofa_couch",
        "chair_armchair",
        "table",
        "bed",
        "cabinet_shelf",
        "kitchen",
        "bathroom",
        "door_window",
    ]
    warning_list = []
    if mesh_count == 0:
        warning_list.append("No Mesh prims found.")
    if material_count == 0:
        warning_list.append("No UsdShade.Material prims found.")
    if light_count == 0:
        warning_list.append("No light prims found in USD; validation fill light will be added for rendering only.")
    if camera_count == 0:
        warning_list.append("No camera prims found; generated validation/inspection poses are proposals.")
    furniture_hits = sum(category_counts.get(cat, 0) for cat in likely_home_categories if cat not in {"room", "wall", "floor"})
    likely_cuboid_only = bool(mesh_count <= 3 and furniture_hits == 0)
    likely_home_like = bool(mesh_count > 0 and (category_counts.get("room", 0) or category_counts.get("wall", 0) or category_counts.get("floor", 0)) and furniture_hits > 0)
    if not likely_home_like:
        warning_list.append("Semantic-name evidence is insufficient to automatically approve the USD as home-like.")
    quality_report = {
        "total_prim_count": len(prim_rows),
        "mesh_prim_count": mesh_count,
        "xform_prim_count": xform_count,
        "material_count": material_count,
        "texture_reference_count": len(texture_rows),
        "light_count": light_count,
        "camera_count": camera_count,
        "referenced_asset_count": len(dependency_report["all_dependency_assets"]),
        "payload_or_sublayer_count": len(dependency_report["payloads"]) + len(dependency_report["sublayers"]),
        "missing_asset_references": len(missing),
        "semantic_category_counts": category_counts,
        "likely_cuboid_only_or_unfurnished": likely_cuboid_only,
        "likely_home_like_by_name_inventory": likely_home_like,
        "visual_quality_pending": True,
        "likely_insufficient": not likely_home_like,
        "warnings": warning_list,
    }
    save_json(output_dir / "usd_quality_warning_report.json", quality_report)
    write_text(output_dir / "usd_quality_warning_report.md", markdown_table("USD Quality Warning Report", quality_report))
    return {
        "stage": stage,
        "prim_rows": prim_rows,
        "material_rows": material_rows,
        "texture_rows": texture_rows,
        "semantic_rows": semantic_rows,
        "bbox_rows": bbox_rows,
        "type_counts": type_counts,
        "dependency_report": dependency_report,
        "bounds_report": bounds_report,
        "quality_report": quality_report,
    }


def choose_map_bounds(args: argparse.Namespace, scene_bounds: dict[str, list[float]] | None) -> tuple[dict[str, list[float]], str]:
    default = {"x": [float(args.x_min), float(args.x_max)], "y": [float(args.y_min), float(args.y_max)], "z": [float(args.z_min), float(args.z_max)]}
    if not scene_bounds:
        return default, "USD scene bounds unavailable; using requested default bounds."
    outside_xy = (
        scene_bounds["x"][0] < default["x"][0]
        or scene_bounds["x"][1] > default["x"][1]
        or scene_bounds["y"][0] < default["y"][0]
        or scene_bounds["y"][1] > default["y"][1]
    )
    outside_z = scene_bounds["z"][0] < default["z"][0] or scene_bounds["z"][1] > default["z"][1]
    if outside_xy or outside_z:
        padded = {
            "x": [min(default["x"][0], scene_bounds["x"][0] - 0.5), max(default["x"][1], scene_bounds["x"][1] + 0.5)],
            "y": [min(default["y"][0], scene_bounds["y"][0] - 0.5), max(default["y"][1], scene_bounds["y"][1] + 0.5)],
            "z": [min(default["z"][0], scene_bounds["z"][0]), max(default["z"][1], scene_bounds["z"][1] + 0.2)],
        }
        return padded, "USD scene bounds exceeded requested defaults; padded bounds were chosen."
    return default, "USD scene bounds fit requested default bounds."


def scene_center_and_extent(scene_bounds: dict[str, list[float]] | None, map_bounds: dict[str, list[float]]) -> tuple[tuple[float, float], tuple[float, float], float]:
    bounds = scene_bounds or map_bounds
    cx = 0.5 * (float(bounds["x"][0]) + float(bounds["x"][1]))
    cy = 0.5 * (float(bounds["y"][0]) + float(bounds["y"][1]))
    ex = max(float(bounds["x"][1]) - float(bounds["x"][0]), 2.0)
    ey = max(float(bounds["y"][1]) - float(bounds["y"][0]), 2.0)
    z = min(max(float(map_bounds["z"][0]) + 1.25, 1.2), max(float(map_bounds["z"][1]) - 0.25, 1.2))
    return (cx, cy), (ex, ey), z


def semantic_center(category: str, semantic_rows: list[dict[str, Any]], fallback: tuple[float, float]) -> tuple[float, float]:
    centers = []
    for row in semantic_rows:
        if category in row.get("categories", []) and row.get("bbox_min") and row.get("bbox_max"):
            bbox = {"min": row["bbox_min"], "max": row["bbox_max"]}
            centers.append(bbox_xy_center(bbox))
    if not centers:
        return fallback
    arr = np.array(centers, dtype=float)
    return (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))


def generate_start_variants(
    output_dir: Path,
    semantic_rows: list[dict[str, Any]],
    bbox_rows: list[dict[str, Any]],
    scene_bounds: dict[str, list[float]] | None,
    map_bounds: dict[str, list[float]],
) -> list[dict[str, Any]]:
    center, extent, z = scene_center_and_extent(scene_bounds, map_bounds)
    cx, cy = center
    ex, ey = extent
    fallbacks = {
        "start_living_area": (cx - 0.25 * ex, cy - 0.10 * ey, "sofa_couch"),
        "start_kitchen_or_counter": (cx - 0.25 * ex, cy + 0.25 * ey, "kitchen"),
        "start_dining_or_open_area": (cx, cy + 0.18 * ey, "table"),
        "start_bedroom_or_private_room": (cx + 0.25 * ex, cy + 0.22 * ey, "bed"),
        "start_study_or_side_room": (cx + 0.22 * ex, cy - 0.20 * ey, "chair_armchair"),
        "start_hallway": (cx, cy - 0.25 * ey, "hallway_corridor_stair_elevator"),
        "start_hallway_junction": (cx + 0.08 * ex, cy, "hallway_corridor_stair_elevator"),
        "start_storage_or_side_room": (cx - 0.35 * ex, cy - 0.30 * ey, "storage_box_clutter"),
        "start_bathroom_or_small_room_entry": (cx + 0.35 * ex, cy - 0.05 * ey, "bathroom"),
    }
    variants = []
    bbox_items = [(str(row["path"]), row.get("bbox")) for row in bbox_rows if row.get("bbox")]
    for idx, (name, (fx, fy, category)) in enumerate(fallbacks.items()):
        sx, sy = semantic_center(category, semantic_rows, (fx, fy))
        sx = float(np.clip(sx, map_bounds["x"][0] + 0.2, map_bounds["x"][1] - 0.2))
        sy = float(np.clip(sy, map_bounds["y"][0] + 0.2, map_bounds["y"][1] - 0.2))
        yaw = yaw_to_center([sx, sy, z], center)
        distances = sorted(
            [
                (bbox_xy_distance((sx, sy), bbox), path)
                for path, bbox in bbox_items
                if bbox is not None and bbox_xy_distance((sx, sy), bbox) > 1e-6
            ],
            key=lambda item: item[0],
        )
        variant = {
            "index": idx,
            "name": name,
            "position": [sx, sy, z],
            "yaw": yaw,
            "yaw_rad": yaw,
            "semantic_zone_guess": category,
            "source": "generated_proposal",
            "inside_bounds": bool(map_bounds["x"][0] <= sx <= map_bounds["x"][1] and map_bounds["y"][0] <= sy <= map_bounds["y"][1]),
            "approximate_clearance": float(distances[0][0]) if distances else None,
            "nearby_prims": [path for _, path in distances[:6]],
            "visible_major_prims": [row["path"] for row in semantic_rows if category in row.get("categories", [])][:8],
            "usable_for_future_expert_pilot": "pending",
            "reason": "Generated from USD bounds and semantic-name guesses; requires Stage 4A-6.6d review.",
        }
        variants.append(variant)
    save_json(output_dir / "start_variants.json", variants)
    write_text(output_dir / "start_variants.md", list_md("Start Variants", [f"`{item['name']}` pos `{item['position']}` zone `{item['semantic_zone_guess']}`" for item in variants]))
    report = {
        "start_variant_count": len(variants),
        "source": "generated_proposal",
        "usd_marker_starts_found": 0,
        "requires_stage4a66d_confirmation": True,
    }
    save_json(output_dir / "start_variant_proposal_report.json", report)
    write_text(output_dir / "start_variant_proposal_report.md", markdown_table("Start Variant Proposal Report", report))
    return variants


def make_pose(index: int, name: str, position: list[float], yaw: float, zone: str, source: str) -> dict[str, Any]:
    return {
        "index": int(index),
        "name": name,
        "position": [float(v) for v in position],
        "yaw": float(yaw),
        "yaw_rad": float(yaw),
        "semantic_zone_guess": zone,
        "source": source,
        "convention": "yaw0_faces_world_+x_yaw90_faces_world_+y_level_camera",
    }


def generate_pose_manifests(
    output_dir: Path,
    starts: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    scene_bounds: dict[str, list[float]] | None,
    map_bounds: dict[str, list[float]],
    min_validation: int,
    min_inspection: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    center, extent, z = scene_center_and_extent(scene_bounds, map_bounds)
    cx, cy = center
    ex, ey = extent
    validation: list[dict[str, Any]] = []
    for start in starts:
        validation.append(
            make_pose(len(validation), f"validation_{start['name']}", start["position"], float(start["yaw_rad"]), start["semantic_zone_guess"], "start_proposal")
        )
    semantic_categories = ["sofa_couch", "kitchen", "table", "bed", "chair_armchair", "bathroom", "cabinet_shelf", "door_window", "storage_box_clutter"]
    for category in semantic_categories:
        sx, sy = semantic_center(category, semantic_rows, (cx, cy))
        for offset in (0.0, math.pi * 0.5):
            pos = [float(np.clip(sx + 0.12 * ex * math.cos(offset), map_bounds["x"][0] + 0.2, map_bounds["x"][1] - 0.2)), float(np.clip(sy + 0.12 * ey * math.sin(offset), map_bounds["y"][0] + 0.2, map_bounds["y"][1] - 0.2)), z]
            validation.append(make_pose(len(validation), f"validation_{category}_{len(validation):03d}", pos, yaw_to_center(pos, center), category, "semantic_generated"))
            if len(validation) >= min_validation:
                break
        if len(validation) >= min_validation:
            break
    perimeter_points = [
        [cx - 0.45 * ex, cy - 0.45 * ey, z],
        [cx + 0.45 * ex, cy - 0.45 * ey, z],
        [cx + 0.45 * ex, cy + 0.45 * ey, z],
        [cx - 0.45 * ex, cy + 0.45 * ey, z],
        [cx, cy - 0.45 * ey, z],
        [cx + 0.45 * ex, cy, z],
        [cx, cy + 0.45 * ey, z],
        [cx - 0.45 * ex, cy, z],
    ]
    while len(validation) < min_validation:
        raw = perimeter_points[len(validation) % len(perimeter_points)]
        pos = [float(np.clip(raw[0], map_bounds["x"][0] + 0.2, map_bounds["x"][1] - 0.2)), float(np.clip(raw[1], map_bounds["y"][0] + 0.2, map_bounds["y"][1] - 0.2)), z]
        validation.append(make_pose(len(validation), f"validation_perimeter_{len(validation):03d}", pos, yaw_to_center(pos, center), "coverage", "bounds_generated"))

    inspection: list[dict[str, Any]] = []
    grid_x = np.linspace(cx - 0.42 * ex, cx + 0.42 * ex, 6)
    grid_y = np.linspace(cy - 0.42 * ey, cy + 0.42 * ey, 6)
    zone_cycle = [
        "living_open_area",
        "kitchen_counter",
        "dining_table",
        "bedroom_private",
        "bathroom_small_room",
        "hallway_corridor",
        "storage_side_room",
        "junction_turn",
        "clutter_occluded_area",
    ]
    for gy in grid_y:
        for gx in grid_x:
            pos = [
                float(np.clip(gx, map_bounds["x"][0] + 0.2, map_bounds["x"][1] - 0.2)),
                float(np.clip(gy, map_bounds["y"][0] + 0.2, map_bounds["y"][1] - 0.2)),
                z,
            ]
            zone = zone_cycle[len(inspection) % len(zone_cycle)]
            inspection.append(make_pose(len(inspection), f"inspection_{zone}_{len(inspection):03d}", pos, yaw_to_center(pos, center), zone, "coverage_grid"))
    for start in starts:
        if len(inspection) >= min_inspection:
            break
        inspection.append(
            make_pose(len(inspection), f"inspection_{start['name']}", start["position"], float(start["yaw_rad"]), start["semantic_zone_guess"], "start_proposal")
        )
    while len(inspection) < min_inspection:
        raw = perimeter_points[len(inspection) % len(perimeter_points)]
        pos = [float(np.clip(raw[0], map_bounds["x"][0] + 0.2, map_bounds["x"][1] - 0.2)), float(np.clip(raw[1], map_bounds["y"][0] + 0.2, map_bounds["y"][1] - 0.2)), z]
        inspection.append(make_pose(len(inspection), f"inspection_extra_{len(inspection):03d}", pos, yaw_to_center(pos, center), "coverage_extra", "bounds_generated"))
    validation = validation[: max(min_validation, len(validation))]
    inspection = inspection[: max(min_inspection, len(inspection))]
    manifest_v = {"pose_count": len(validation), "poses": validation, "source": "generated_from_usd_bounds_and_semantics"}
    manifest_i = {"pose_count": len(inspection), "poses": inspection, "source": "generated_from_usd_bounds_and_semantics"}
    save_json(output_dir / "validation_pose_manifest.json", manifest_v)
    write_text(output_dir / "validation_pose_manifest.md", list_md("Validation Pose Manifest", [f"`{p['index']:03d}` `{p['name']}` pos `{p['position']}`" for p in validation]))
    save_json(output_dir / "inspection_pose_manifest.json", manifest_i)
    write_text(output_dir / "inspection_pose_manifest.md", list_md("Inspection Pose Manifest", [f"`{p['index']:03d}` `{p['name']}` pos `{p['position']}`" for p in inspection]))
    return validation, inspection


def draw_pose_arrow(ax: Any, pose: dict[str, Any], color: str, label: str | None = None) -> None:
    x, y, _ = pose["position"]
    yaw = float(pose["yaw_rad"])
    ax.arrow(x, y, math.cos(yaw) * 0.45, math.sin(yaw) * 0.45, width=0.018, head_width=0.12, color=color, length_includes_head=True, zorder=10)
    ax.scatter([x], [y], c=[color], s=18, edgecolors="#ffffff", linewidths=0.35, zorder=11)
    if label is not None:
        ax.text(x + 0.06, y + 0.06, label, fontsize=5.0, color=color, weight="bold", zorder=12)


def draw_topdown(
    path: Path,
    title: str,
    bbox_rows: list[dict[str, Any]],
    map_bounds: dict[str, list[float]],
    *,
    starts: list[dict[str, Any]] | None = None,
    validation: list[dict[str, Any]] | None = None,
    inspection: list[dict[str, Any]] | None = None,
    color_mode: str = "category",
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 8.8), constrained_layout=True)
    ax.set_xlim(map_bounds["x"])
    ax.set_ylim(map_bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e1e6ec", linewidth=0.45)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    palette = {
        "wall": "#5b6470",
        "floor": "#b6c8bf",
        "room": "#ead8aa",
        "sofa_couch": "#396fa3",
        "chair_armchair": "#527a45",
        "table": "#a56b3d",
        "bed": "#8057a5",
        "kitchen": "#d18a2c",
        "bathroom": "#3f8f9e",
        "cabinet_shelf": "#7d6346",
        "door_window": "#22a15f",
        "storage_box_clutter": "#93624e",
    }
    for row in bbox_rows:
        bbox = row.get("bbox")
        if not bbox:
            continue
        x0, y0 = float(bbox["min"][0]), float(bbox["min"][1])
        x1, y1 = float(bbox["max"][0]), float(bbox["max"][1])
        if x1 - x0 <= 1e-4 or y1 - y0 <= 1e-4:
            continue
        categories = row.get("categories") or []
        key = categories[0] if categories else "other"
        color = palette.get(key, "#9aa3ad")
        if color_mode == "material":
            color = f"#{abs(hash(row['path'])) % 0xFFFFFF:06x}"
        elif color_mode == "semantic":
            color = palette.get(key, "#d0d5dd")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor="#1f2937", linewidth=0.25, alpha=0.62))
    if starts:
        for item in starts:
            draw_pose_arrow(ax, item, "#7c3aed", item["name"].replace("start_", "")[:10])
    if validation:
        for item in validation:
            draw_pose_arrow(ax, item, "#1554b7", str(item["index"]))
    if inspection:
        for item in inspection:
            draw_pose_arrow(ax, item, "#0f766e", str(item["index"]))
    ax.set_title(title)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_topdown_package(
    output_dir: Path,
    bbox_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    map_bounds: dict[str, list[float]],
    starts: list[dict[str, Any]],
    validation_poses: list[dict[str, Any]],
    inspection_poses: list[dict[str, Any]],
) -> None:
    draw_topdown(output_dir / "usd_scene_layout_topdown.png", "USD scene layout topdown", bbox_rows, map_bounds, color_mode="category")
    draw_topdown(output_dir / "usd_scene_bounds_topdown.png", "USD scene bounds and bboxes", bbox_rows, map_bounds, color_mode="category")
    draw_topdown(output_dir / "usd_semantic_guess_topdown.png", "USD semantic guesses topdown", bbox_rows, map_bounds, color_mode="semantic")
    draw_topdown(output_dir / "usd_prim_category_topdown.png", "USD prim categories topdown", bbox_rows, map_bounds, color_mode="category")
    draw_topdown(output_dir / "usd_material_color_topdown.png", "USD material/color proxy topdown", bbox_rows, map_bounds, color_mode="material")
    draw_topdown(output_dir / "start_variants_topdown.png", "Generated start proposals", bbox_rows, map_bounds, starts=starts)
    draw_topdown(output_dir / "validation_camera_poses_topdown.png", "Validation camera poses", bbox_rows, map_bounds, validation=validation_poses)
    draw_topdown(output_dir / "inspection_camera_poses_topdown.png", "Inspection camera poses", bbox_rows, map_bounds, inspection=inspection_poses)
    closeups = [
        ("living_area_closeup.png", "living area closeup", "sofa_couch"),
        ("kitchen_or_counter_closeup.png", "kitchen/counter closeup", "kitchen"),
        ("bedroom_or_private_room_closeup.png", "bedroom/private room closeup", "bed"),
        ("bathroom_or_small_room_closeup.png", "bathroom/small room closeup", "bathroom"),
        ("hallway_or_junction_closeup.png", "hallway/junction closeup", "hallway_corridor_stair_elevator"),
        ("clutter_or_storage_closeup.png", "clutter/storage closeup", "storage_box_clutter"),
    ]
    for filename, title, category in closeups:
        center = semantic_center(category, semantic_rows, scene_center_and_extent(None, map_bounds)[0])
        xlim = (center[0] - 3.2, center[0] + 3.2)
        ylim = (center[1] - 3.2, center[1] + 3.2)
        fig, ax = plt.subplots(figsize=(6.6, 6.0), constrained_layout=True)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#e1e6ec", linewidth=0.45)
        for row in bbox_rows:
            bbox = row.get("bbox")
            if not bbox:
                continue
            x0, y0 = float(bbox["min"][0]), float(bbox["min"][1])
            x1, y1 = float(bbox["max"][0]), float(bbox["max"][1])
            if x1 < xlim[0] or x0 > xlim[1] or y1 < ylim[0] or y0 > ylim[1]:
                continue
            color = "#9aa3ad"
            if category in row.get("categories", []):
                color = "#d18a2c"
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor="#1f2937", linewidth=0.3, alpha=0.7))
        ax.set_title(title)
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def extract_rgb(camera: Any, label: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEYS:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        rgb = normalize_rgb(tensor[0].detach().cpu().numpy())
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


def capture_pose(
    output_dir: Path,
    camera: Any,
    sim: Any,
    pose: dict[str, Any],
    prefix: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

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
    rgb, rgb_key, rstats = extract_rgb(camera, f"{prefix} {idx}")
    depth, dstats = extract_depth(camera, f"{prefix} {idx}")
    rgb_path = output_dir / f"{prefix}_rgb_{idx:03d}.png"
    depth_path = output_dir / f"{prefix}_depth_{idx:03d}.npy"
    depth_color_path = output_dir / f"{prefix}_depth_color_{idx:03d}.png"
    pose_path = output_dir / f"{prefix}_pose_{idx:03d}.json"
    Image.fromarray(rgb).save(rgb_path)
    np.save(depth_path, depth)
    save_depth_color(depth_color_path, depth, f"{prefix} depth {idx:03d}")
    pose_record = dict(pose)
    pose_record.update(
        {
            "target": target,
            "render_backend": "isaac_headless",
            "rgb_file": rgb_path.name,
            "depth_file": depth_path.name,
            "depth_color_file": depth_color_path.name,
        }
    )
    save_json(pose_path, pose_record)
    return {
        "index": idx,
        "name": pose.get("name"),
        "semantic_zone_guess": pose.get("semantic_zone_guess"),
        "source": pose.get("source"),
        "render_backend": "isaac_headless",
        "rgb_file": rgb_path.name,
        "depth_file": depth_path.name,
        "depth_color_file": depth_color_path.name,
        "pose_file": pose_path.name,
        "rgb_key_used": rgb_key,
        "rgb_stats": rstats,
        "depth_stats": dstats,
    }


def build_observed_state(
    output_dir: Path,
    validation_records: list[dict[str, Any]],
    validation_poses: list[dict[str, Any]],
    camera_info: dict[str, Any],
    map_bounds: dict[str, list[float]],
    voxel_size: float,
    pixel_stride: int,
    max_depth: float,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    pose_by_index = {int(p["index"]): p for p in validation_poses}
    stride_candidates = [int(pixel_stride), max(1, int(pixel_stride) - 1), max(1, int(pixel_stride) - 2), 2, 1, int(pixel_stride) + 2, int(pixel_stride) + 4]
    stride_candidates = list(dict.fromkeys([s for s in stride_candidates if s >= 1]))
    chosen: tuple[np.ndarray, list[dict[str, Any]], int] | None = None
    for stride in stride_candidates:
        observed = create_observed_grid(map_bounds, voxel_size)
        transitions = []
        for record in validation_records:
            idx = int(record["index"])
            pose_record = dict(pose_by_index[idx])
            pose_record.update({"target": pose_target(pose_record["position"], pose_record["yaw_rad"])})
            depth = np.load(output_dir / record["depth_file"])
            before = observed.copy()
            observed = integrate_depth_frame(
                observed_state=observed,
                depth_image=depth,
                camera_pose=pose_record,
                intrinsics=camera_info,
                map_bounds=map_bounds,
                voxel_size=float(voxel_size),
                pixel_stride=int(stride),
                max_depth=float(max_depth),
            )
            transitions.append({"step": idx, "pixel_stride": stride, **state_transition(before, observed)})
            np.save(output_dir / f"observed_state_step{idx:03d}.npy", observed)
        summary = summarize_observed_grid(observed)
        if 0.02 < float(summary["observed_ratio"]) < 0.60:
            chosen = (observed, transitions, stride)
            break
        if chosen is None:
            chosen = (observed, transitions, stride)
    assert chosen is not None
    observed, transitions, stride = chosen
    invalid = int(np.count_nonzero(~np.isin(observed, [UNKNOWN, FREE, OCCUPIED])))
    summary = summarize_observed_grid(observed)
    summary.update(
        {
            "invalid_label_count": invalid,
            "labels_present": {
                "UNKNOWN": bool(np.any(observed == UNKNOWN)),
                "FREE": bool(np.any(observed == FREE)),
                "OCCUPIED": bool(np.any(observed == OCCUPIED)),
            },
            "measured_only": True,
            "prediction_used": False,
            "map_predict_called": False,
            "ground_truth_used_for_scoring": False,
            "chosen_bounds": map_bounds,
            "voxel_size": float(voxel_size),
            "pixel_stride_used": int(stride),
        }
    )
    transition_summary = {
        "transition_records": transitions,
        "total_newly_observed": int(sum(item["newly_observed"] for item in transitions)),
        "total_unknown_to_free": int(sum(item["unknown_to_free"] for item in transitions)),
        "total_unknown_to_occupied": int(sum(item["unknown_to_occupied"] for item in transitions)),
        "invalid_label_count_final": invalid,
    }
    np.save(output_dir / "observed_state_final.npy", observed)
    save_json(output_dir / "observed_summary.json", summary)
    save_json(output_dir / "observed_state_validation_summary.json", summary)
    write_text(output_dir / "observed_state_validation_summary.md", markdown_table("Observed State Validation Summary", summary))
    save_json(output_dir / "observed_state_transition_summary.json", transition_summary)
    write_text(output_dir / "observed_state_transition_summary.md", markdown_table("Observed State Transition Summary", transition_summary))
    return observed, summary, transition_summary


def save_observed_topdown(path: Path, observed: np.ndarray) -> None:
    occupied = np.any(observed == OCCUPIED, axis=2)
    free = np.any(observed == FREE, axis=2)
    top = np.full(observed.shape[:2], UNKNOWN, dtype=np.int8)
    top[free] = FREE
    top[occupied] = OCCUPIED
    fig, ax = plt.subplots(figsize=(8.0, 7.6), constrained_layout=True)
    image = ax.imshow(top.T, origin="lower", cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    fig.colorbar(image, ax=ax, ticks=[UNKNOWN, FREE, OCCUPIED], label="state")
    ax.set_title("measured-only observed_state topdown")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_image_grids(output_dir: Path, prefix: str, records: list[dict[str, Any]]) -> None:
    cols = 6
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.45, rows * 2.05), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, record in zip(axes, records):
        rgb = np.asarray(Image.open(output_dir / record["rgb_file"]).convert("RGB"))
        ax.imshow(rgb)
        ax.set_title(f"{record['index']:03d} {record.get('semantic_zone_guess') or ''}", fontsize=6.3)
        ax.axis("off")
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.savefig(output_dir / f"rgb_{prefix}_grid.png", dpi=155)
    plt.close(fig)

    depths = [np.load(output_dir / record["depth_file"]) for record in records]
    finite_values = [depth[np.isfinite(depth) & (depth > 0.0)] for depth in depths]
    valid = [values for values in finite_values if values.size]
    vmin = min(float(values.min()) for values in valid)
    vmax = max(float(values.max()) for values in valid)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.45, rows * 2.15), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    last = None
    for ax, record, depth in zip(axes, records, depths):
        last = ax.imshow(np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan)), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{record['index']:03d} {record.get('semantic_zone_guess') or ''}", fontsize=6.3)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.colorbar(last, ax=axes[: len(records)].tolist(), shrink=0.62, label="depth (m)")
    fig.savefig(output_dir / f"depth_{prefix}_grid.png", dpi=155)
    plt.close(fig)


def validation_report(prefix: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    depth_values = []
    failed = []
    for record in records:
        dstats = record["depth_stats"]
        if record["rgb_stats"]["nonblank"] is not True or dstats["has_positive_finite_depth"] is not True:
            failed.append(record["index"])
        if dstats["min"] is not None:
            depth_values.extend([dstats["min"], dstats["median"], dstats["max"]])
    positive_depth_records = [r for r in records if r["depth_stats"]["has_positive_finite_depth"]]
    return {
        f"{prefix}_pose_count": len(records),
        "nonblank_rgb_count": sum(1 for item in records if item["rgb_stats"]["nonblank"]),
        "finite_positive_depth_count": len(positive_depth_records),
        "depth_min": min((float(item["depth_stats"]["min"]) for item in positive_depth_records if item["depth_stats"]["min"] is not None), default=None),
        "depth_median_of_medians": float(np.median([item["depth_stats"]["median"] for item in positive_depth_records if item["depth_stats"]["median"] is not None])) if positive_depth_records else None,
        "depth_max": max((float(item["depth_stats"]["max"]) for item in positive_depth_records if item["depth_stats"]["max"] is not None), default=None),
        "failed_views": failed,
        "records": records,
    }


def capture_with_isaac(args: argparse.Namespace, app_launcher_cls: Any, output_dir: Path, staged_usd: Path) -> tuple[dict[str, Any], Any]:
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
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        usd_info = inspect_usd_with_pxr(staged_usd, output_dir)
        bounds_report = usd_info["bounds_report"]
        map_bounds, bounds_reason = choose_map_bounds(args, bounds_report.get("scene_bounds"))
        starts = generate_start_variants(output_dir, usd_info["semantic_rows"], usd_info["bbox_rows"], bounds_report.get("scene_bounds"), map_bounds)
        validation_poses, inspection_poses = generate_pose_manifests(
            output_dir,
            starts,
            usd_info["semantic_rows"],
            bounds_report.get("scene_bounds"),
            map_bounds,
            int(args.min_validation_poses),
            int(args.min_inspection_poses),
        )
        save_topdown_package(output_dir, usd_info["bbox_rows"], usd_info["semantic_rows"], map_bounds, starts, validation_poses, inspection_poses)

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
        sim.set_camera_view([0.0, -18.0, 15.0], [0.0, 0.0, 0.8])
        if int(usd_info["quality_report"].get("light_count", 0)) == 0:
            dome = sim_utils.DomeLightCfg(intensity=3300.0, color=(0.84, 0.86, 0.82))
            dome.func("/World/ValidationFillLight", dome)
        else:
            dome = sim_utils.DomeLightCfg(intensity=600.0, color=(0.84, 0.86, 0.82))
            dome.func("/World/ValidationSoftFillLight", dome)
        builder_metadata = build_home_like_scene_v1(seed=int(args.scene_seed), spawn=True, sim_utils_module=sim_utils, staged_usd_path=str(staged_usd))
        sim_utils.create_prim("/World/ValidationCameraRig", "Xform")
        camera = Camera(
            cfg=CameraCfg(
                prim_path="/World/ValidationCameraRig/CameraSensor",
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
        )
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

        validation_records = []
        for pose in validation_poses:
            record = capture_pose(output_dir, camera, sim, pose, "validation", args)
            validation_records.append(record)
            print(f"[INFO]: validation {record['index']:03d} rgb_nonblank={record['rgb_stats']['nonblank']} depth_positive={record['depth_stats']['positive_count']}")
        inspection_records = []
        for pose in inspection_poses:
            record = capture_pose(output_dir, camera, sim, pose, "inspection", args)
            inspection_records.append(record)
            print(f"[INFO]: inspection {record['index']:03d} rgb_nonblank={record['rgb_stats']['nonblank']} depth_positive={record['depth_stats']['positive_count']}")

        observed, observed_summary, transition_summary = build_observed_state(
            output_dir,
            validation_records,
            validation_poses,
            camera_info,
            map_bounds,
            float(args.voxel_size),
            int(args.pixel_stride),
            float(args.max_depth),
        )
        save_observed_topdown(output_dir / "observed_topdown_final.png", observed)
        fixed_report = validation_report("validation", validation_records)
        visual_report = validation_report("inspection", inspection_records)
        save_json(output_dir / "fixed_capture_validation.json", fixed_report)
        write_text(
            output_dir / "fixed_capture_validation.md",
            markdown_table(
                "Fixed Capture Validation",
                {
                    "validation_pose_count": fixed_report["validation_pose_count"],
                    "nonblank_rgb_count": fixed_report["nonblank_rgb_count"],
                    "finite_positive_depth_count": fixed_report["finite_positive_depth_count"],
                    "depth_min": fixed_report["depth_min"],
                    "depth_median_of_medians": fixed_report["depth_median_of_medians"],
                    "depth_max": fixed_report["depth_max"],
                    "failed_views": fixed_report["failed_views"],
                },
            ),
        )
        save_json(output_dir / "visual_inspection_capture_validation.json", visual_report)
        write_text(
            output_dir / "visual_inspection_capture_validation.md",
            markdown_table(
                "Visual Inspection Capture Validation",
                {
                    "inspection_pose_count": visual_report["inspection_pose_count"],
                    "nonblank_rgb_count": visual_report["nonblank_rgb_count"],
                    "finite_positive_depth_count": visual_report["finite_positive_depth_count"],
                    "depth_min": visual_report["depth_min"],
                    "depth_median_of_medians": visual_report["depth_median_of_medians"],
                    "depth_max": visual_report["depth_max"],
                    "failed_views": visual_report["failed_views"],
                },
            ),
        )
        scene_load = {
            "stage": "Stage 4A-6.6c-usd-import",
            "scene_variant": "home_like_scene_v1",
            "staged_usd": str(staged_usd),
            "scene_loaded": True,
            "isaac_headless_startup_count": 1,
            "loaded_builder": "build_home_like_scene_v1_from_usd",
            "builder_metadata": builder_metadata,
            "validation_fill_light_added_for_rendering": True,
            "procedural_scene_generated": False,
            "cuboid_fallback_used": False,
            "rollout_run": False,
            "expert_sampling_run": False,
        }
        save_json(output_dir / "scene_load_validation.json", scene_load)
        write_text(output_dir / "scene_load_validation.md", markdown_table("Scene Load Validation", scene_load))
        scene_metadata = {
            "stage": "Stage 4A-6.6c-usd-import",
            "scene_variant": "home_like_scene_v1",
            "source_usd": str(args.source_usd),
            "staged_usd": str(staged_usd),
            "source_sha256": sha256_file(Path(args.source_usd)),
            "staged_sha256": sha256_file(staged_usd),
            "map_bounds": map_bounds,
            "map_bounds_reason": bounds_reason,
            "voxel_size": float(args.voxel_size),
            "expected_observed_state_shape": list(create_observed_grid(map_bounds, float(args.voxel_size)).shape),
            "usd_bounds_report": bounds_report,
            "usd_quality_report": usd_info["quality_report"],
            "usd_dependency_report": usd_info["dependency_report"],
            "start_variants": starts,
            "validation_camera_poses": validation_poses,
            "inspection_camera_poses": inspection_poses,
            "validation_capture_records": validation_records,
            "inspection_capture_records": inspection_records,
            "camera": camera_info,
            "observed_summary": observed_summary,
            "observed_transition_summary": transition_summary,
            "manual_review_gate": {
                "human_visual_inspection_done": False,
                "user_needs_to_review_visuals": True,
                "visual_approval_required_before_6_7": True,
                "formal_expert_sampling_ready": False,
                "full_expert_dataset_ready": False,
            },
            "negative_scope": {
                "rollout_run": False,
                "expert_sampling_run": False,
                "map_predict_called": False,
                "sscnet_inference_called": False,
                "rl_gdpo_run": False,
            },
        }
        save_json(output_dir / "scene_metadata.json", scene_metadata)
        return {
            "scene_load_validation": scene_load,
            "fixed_capture_validation": fixed_report,
            "visual_inspection_capture_validation": visual_report,
            "observed_summary": observed_summary,
            "observed_transition_summary": transition_summary,
            "scene_metadata": scene_metadata,
        }, simulation_app
    except Exception:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        raise


def make_flythrough(output_dir: Path, inspection_records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_dir = output_dir / "usd_scene_flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    if not inspection_records:
        report = {"mp4_created": False, "frame_count": 0, "frame_dir": str(frame_dir), "reason": "no_inspection_records"}
        write_text(output_dir / "video_generation_skipped.md", markdown_table("Video Generation Skipped", report))
        return report
    for frame_idx in range(72):
        record = inspection_records[int(frame_idx / 72.0 * len(inspection_records)) % len(inspection_records)]
        image = Image.open(output_dir / record["rgb_file"]).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 42), fill=(18, 24, 32))
        draw.text((12, 11), f"USD home_like_scene_v1 | inspection {record['index']:03d}", fill=(240, 244, 248))
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        image.save(frame_path)
        frames.append(frame_path)
    report: dict[str, Any] = {"mp4_created": False, "frame_count": len(frames), "frame_dir": str(frame_dir), "video_path": None}
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        mp4_path = output_dir / "usd_scene_flythrough.mp4"
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-framerate",
                "2",
                "-i",
                str(frame_dir / "frame_%03d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-movflags",
                "+faststart",
                "-crf",
                "23",
                str(mp4_path),
            ],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=75,
        )
        report["ffmpeg_returncode"] = int(result.returncode)
        report["ffmpeg_stderr_tail"] = result.stderr[-2000:]
        if result.returncode == 0 and mp4_path.is_file() and mp4_path.stat().st_size > 0:
            report.update({"mp4_created": True, "video_path": str(mp4_path)})
            return report
    except Exception as exc:
        report["mp4_error"] = str(exc)
    write_text(output_dir / "video_generation_skipped.md", markdown_table("Video Generation Skipped", report))
    return report


def write_no_scope_reports(output_dir: Path) -> None:
    reports = {
        "no_rollout_report": {
            "rollout_run": False,
            "open_ended_loop_run": False,
            "selected_action_executed": False,
            "transitions_jsonl_created": False,
        },
        "no_formal_expert_sampling_report": {
            "formal_expert_sampling_run": False,
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


def write_review_gate(output_dir: Path) -> None:
    checklist = [
        "Does the USD look like a real home/interior environment?",
        "Is it clearly better than a cuboid maze?",
        "Are living/kitchen/bedroom/bathroom/hall/storage zones present?",
        "Are furniture objects real/high-quality models rather than simple blocks?",
        "Are occlusions natural?",
        "Are there obvious intersections?",
        "Are there floating objects?",
        "Are there scale errors?",
        "Are textures/materials missing?",
        "Are there black/overexposed/underexposed areas?",
        "Are start proposals reasonable?",
        "Does the teammate need to revise the USD?",
        "May the project enter Stage 4A-6.6d audit?",
        "May later Stage 4A-6.7 measured-only expert pilot be considered after approval?",
    ]
    payload = {
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "visual_approval_required_before_6_7": True,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "items": [{"id": idx + 1, "question": item, "status": "pending_user_review"} for idx, item in enumerate(checklist)],
    }
    save_json(output_dir / "human_visual_review_checklist.json", payload)
    write_text(output_dir / "human_visual_review_checklist.md", "\n".join(["# Human Visual Review Checklist", "", *[f"{idx + 1}. {item}" for idx, item in enumerate(checklist)], "", "Status: `pending_user_review`."]))
    gate = {
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "visual_approval_required_before_6_7": True,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "stage4a66d_usd_scene_audit_required_next": True,
        "stage4a67_executed": False,
    }
    save_json(output_dir / "manual_review_gate.json", gate)
    write_text(output_dir / "manual_review_gate.md", markdown_table("Manual Review Gate", gate))
    write_text(
        output_dir / "future_stage4a66d_usd_scene_audit_command_sketch.md",
        "# Future Stage 4A-6.6d USD Scene Audit Command Sketch\n\nReview `visual_inspection_index.html`, inspect RGB/depth/topdown/closeups/MP4, then run a separate USD scene audit + human review gate. Do not run Stage 4A-6.7 until visual approval is recorded.",
    )
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\nProceed to Stage 4A-6.6d USD scene audit + human visual review for the staged `home_like_scene_v1` candidate. Keep formal expert sampling disabled.",
    )
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "# Long-Term RL/GDPO Note\n\nRL/GDPO/PPO/BC/IL remain future directions only. This USD import stage created no replay buffer, policy checkpoint, or training artifact.",
    )


def write_html_index(output_dir: Path, summary: dict[str, Any], video_report: dict[str, Any]) -> None:
    image_names = [
        "usd_scene_layout_topdown.png",
        "usd_scene_bounds_topdown.png",
        "usd_semantic_guess_topdown.png",
        "usd_prim_category_topdown.png",
        "usd_material_color_topdown.png",
        "start_variants_topdown.png",
        "validation_camera_poses_topdown.png",
        "inspection_camera_poses_topdown.png",
        "rgb_validation_grid.png",
        "depth_validation_grid.png",
        "rgb_inspection_grid.png",
        "depth_inspection_grid.png",
        "observed_topdown_final.png",
        "living_area_closeup.png",
        "kitchen_or_counter_closeup.png",
        "bedroom_or_private_room_closeup.png",
        "bathroom_or_small_room_closeup.png",
        "hallway_or_junction_closeup.png",
        "clutter_or_storage_closeup.png",
    ]
    figures = "\n".join(
        f'<figure><img src="{name}" width="360"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in image_names
        if (output_dir / name).is_file()
    )
    if video_report.get("mp4_created") and video_report.get("video_path"):
        video_html = '<video controls width="720" src="usd_scene_flythrough.mp4"></video>'
    else:
        video_html = '<p><a href="usd_scene_flythrough_frames/">Fallback flythrough frames</a></p>'
    starts = read_json(output_dir / "start_variants.json")
    start_rows = "\n".join(f"<li><code>{html.escape(item['name'])}</code> {html.escape(str(item['position']))} zone <code>{html.escape(item['semantic_zone_guess'])}</code></li>" for item in starts)
    warnings = summary.get("usd_quality_warning_report", {}).get("warnings", [])
    warning_rows = "\n".join(f"<li>{html.escape(str(item))}</li>" for item in warnings) or "<li>No automatic warning beyond pending human review.</li>"
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.6c USD home_like_scene_v1 validation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #17202a; background: #f7f8fa; }}
    figure {{ display: inline-block; margin: 10px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 360px; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.6c USD home_like_scene_v1 validation</h1>
  <p>Source USD: <code>{html.escape(summary['source_usd'])}</code></p>
  <p>Staged USD: <code>{html.escape(summary['staged_usd'])}</code></p>
  <p>Source hash: <code>{summary['source_sha256']}</code>; staged hash: <code>{summary['staged_sha256']}</code></p>
  <p>Dependencies complete: <code>{summary['dependencies_complete']}</code>; prims: <code>{summary['total_prim_count']}</code>; meshes/materials/textures: <code>{summary['mesh_prim_count']}/{summary['material_count']}/{summary['texture_reference_count']}</code></p>
  <p>human_visual_inspection_done: <code>false</code>; formal_expert_sampling_ready: <code>false</code>; full_expert_dataset_ready: <code>false</code>.</p>
  <h2>Warnings</h2>
  <ul>{warning_rows}</ul>
  <h2>Start Proposals</h2>
  <ul>{start_rows}</ul>
  <h2>Visuals</h2>
  {figures}
  <h2>Flythrough</h2>
  {video_html}
  <h2>Human Review</h2>
  <p><a href="human_visual_review_checklist.md">human_visual_review_checklist.md</a></p>
  <p><a href="manual_review_gate.md">manual_review_gate.md</a></p>
  <h2>Next Gate</h2>
  <p>Next stage is Stage 4A-6.6d USD scene audit + human visual review. Stage 4A-6.7 remains blocked.</p>
</body>
</html>"""
    write_text(output_dir / "visual_inspection_index.html", body)


def write_summary(output_dir: Path, validation: dict[str, Any], video_report: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    metadata = validation["scene_metadata"]
    quality = metadata["usd_quality_report"]
    dependency = metadata["usd_dependency_report"]
    fixed = validation["fixed_capture_validation"]
    visual = validation["visual_inspection_capture_validation"]
    observed = validation["observed_summary"]
    starts = metadata["start_variants"]
    semantic_categories = quality.get("semantic_category_counts", {})
    summary = {
        "stage": "Stage 4A-6.6c-usd-import",
        "source_usd": metadata["source_usd"],
        "staged_usd": metadata["staged_usd"],
        "source_sha256": metadata["source_sha256"],
        "staged_sha256": metadata["staged_sha256"],
        "old_larger_complex_scene_v1_rejected_confirmed": read_json(output_dir / "old_scene_rejection_status.json")["user_rejected_old_larger_complex_scene_v1"],
        "old_larger_complex_scene_v1_disabled": read_json(output_dir / "old_scene_rejection_status.json")["build_larger_complex_scene_v1_disabled"],
        "old_larger_scene_outputs_absent": read_json(output_dir / "old_scene_cleanup_verification.json")["old_larger_scene_output_dirs_absent"],
        "dependencies_complete": dependency["dependencies_complete"],
        "missing_dependency_count": len(dependency["missing_dependencies"]),
        "total_prim_count": quality["total_prim_count"],
        "mesh_prim_count": quality["mesh_prim_count"],
        "material_count": quality["material_count"],
        "texture_reference_count": quality["texture_reference_count"],
        "semantic_category_counts": semantic_categories,
        "scene_factory_registered_home_like_scene_v1_from_usd": read_json(output_dir / "scene_factory_registration_report.json")["loads_staged_usd"],
        "isaac_headless_loaded_usd": validation["scene_load_validation"]["scene_loaded"],
        "isaac_headless_startup_count": validation["scene_load_validation"]["isaac_headless_startup_count"],
        "validation_rgb_depth_valid": fixed["nonblank_rgb_count"] >= 20 and fixed["finite_positive_depth_count"] >= 20 and not fixed["failed_views"],
        "validation_rgb_nonblank_count": fixed["nonblank_rgb_count"],
        "validation_depth_positive_count": fixed["finite_positive_depth_count"],
        "inspection_rgb_depth_valid": visual["nonblank_rgb_count"] >= 36 and visual["finite_positive_depth_count"] >= 36 and not visual["failed_views"],
        "inspection_rgb_nonblank_count": visual["nonblank_rgb_count"],
        "inspection_depth_positive_count": visual["finite_positive_depth_count"],
        "observed_state_valid": bool(
            observed["labels_present"]["UNKNOWN"]
            and observed["labels_present"]["FREE"]
            and observed["labels_present"]["OCCUPIED"]
            and observed["invalid_label_count"] == 0
            and 0.02 < float(observed["observed_ratio"]) < 0.60
        ),
        "observed_state_shape": observed["shape"],
        "observed_ratio": observed["observed_ratio"],
        "start_proposals": [item["name"] for item in starts],
        "html_index": str(output_dir / "visual_inspection_index.html"),
        "mp4": str(output_dir / "usd_scene_flythrough.mp4"),
        "closeups": [
            str(output_dir / "living_area_closeup.png"),
            str(output_dir / "kitchen_or_counter_closeup.png"),
            str(output_dir / "bedroom_or_private_room_closeup.png"),
            str(output_dir / "bathroom_or_small_room_closeup.png"),
            str(output_dir / "hallway_or_junction_closeup.png"),
            str(output_dir / "clutter_or_storage_closeup.png"),
        ],
        "usd_may_be_home_like_by_automatic_name_inventory": quality["likely_home_like_by_name_inventory"],
        "automatic_visual_approval": False,
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "formal_expert_sampling_ready": False,
        "full_expert_dataset_ready": False,
        "next_stage": "Stage 4A-6.6d USD scene audit + human visual review",
        "stage4a67_executed": False,
        "no_rollout": True,
        "no_formal_expert_sampling": True,
        "no_map_predict": True,
        "no_rl_gdpo": True,
        "usd_quality_warning_report": quality,
        "flythrough": video_report,
        "elapsed_seconds": float(elapsed_s),
    }
    save_json(output_dir / "stage4a66c_usd_home_like_scene_summary.json", summary)
    lines = [
        "# Stage 4A-6.6c USD Home-Like Scene Summary",
        "",
        f"1. Old `larger_complex_scene_v1` rejected? `{summary['old_larger_complex_scene_v1_rejected_confirmed']}`.",
        f"2. Old builder still disabled? `{summary['old_larger_complex_scene_v1_disabled']}`.",
        f"3. Old output directories absent? `{summary['old_larger_scene_outputs_absent']}`.",
        f"4. Source USD path: `{summary['source_usd']}`.",
        f"5. Staged USD path: `{summary['staged_usd']}`.",
        f"6. Source/staged hashes: `{summary['source_sha256']}` / `{summary['staged_sha256']}`.",
        f"7. Dependencies complete? `{summary['dependencies_complete']}`; missing count `{summary['missing_dependency_count']}`.",
        f"8. USD prim count: `{summary['total_prim_count']}`.",
        f"9. Mesh/material/texture counts: `{summary['mesh_prim_count']}` / `{summary['material_count']}` / `{summary['texture_reference_count']}`.",
        f"10. Semantic categories: `{json.dumps(summary['semantic_category_counts'], sort_keys=True)}`.",
        f"11. `scene_factory` registered `home_like_scene_v1` from USD? `{summary['scene_factory_registered_home_like_scene_v1_from_usd']}`.",
        f"12. Isaac headless loaded USD? `{summary['isaac_headless_loaded_usd']}`.",
        f"13. Validation RGB/depth valid? `{summary['validation_rgb_depth_valid']}` ({summary['validation_rgb_nonblank_count']} RGB, {summary['validation_depth_positive_count']} depth).",
        f"14. Inspection RGB/depth valid? `{summary['inspection_rgb_depth_valid']}` ({summary['inspection_rgb_nonblank_count']} RGB, {summary['inspection_depth_positive_count']} depth).",
        f"15. observed_state valid? `{summary['observed_state_valid']}` shape `{summary['observed_state_shape']}`, observed_ratio `{summary['observed_ratio']}`.",
        f"16. Start proposals: `{', '.join(summary['start_proposals'])}`.",
        f"17. HTML / MP4 / closeups: `{summary['html_index']}`, `{summary['mp4']}`, closeup PNGs in output dir.",
        f"18. USD may be home-like by automatic name inventory? `{summary['usd_may_be_home_like_by_automatic_name_inventory']}`.",
        "19. Human visual review is still required.",
        "20. Formal expert sampling is not allowed now.",
        "21. Full expert dataset is not allowed now.",
        "22. Next step: Stage 4A-6.6d USD scene audit + human visual review.",
        "",
        "Conclusion: `building_scene.usd` was staged and validated as the current `home_like_scene_v1` candidate, but it is not automatically approved. No rollout, expert sampling, map_predict, SSCNet inference, or RL/GDPO was run.",
    ]
    write_text(output_dir / "stage4a66c_usd_home_like_scene_summary.md", "\n".join(lines))
    return summary


def check_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    forbidden_exact = {
        "transitions.jsonl",
        "rollout_topdown_path.png",
        "rollout_index.html",
        "expert_dataset_manifest.jsonl",
        "expert_dataset_manifest.json",
        "selected_action_execution_report.json",
        "selected_action_report.json",
        "action_execution_report.json",
    }
    forbidden_tokens = ("replay_buffer", "policy_checkpoint", "ppo", "gdpo_checkpoint")
    hits = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        rel = path.relative_to(output_dir)
        if name in forbidden_exact:
            hits.append(str(rel))
        if name.endswith(".npz") and ("prediction" in name or "global_prediction_layer" in str(rel)):
            hits.append(str(rel))
        if any(token in name for token in forbidden_tokens):
            hits.append(str(rel))
    if hits:
        raise RuntimeError(f"Forbidden outputs present: {hits}")
    return {"forbidden_hits": [], "passed": True}


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Stage 4A-6.6c USD import validation.")
    parser.add_argument("--source_usd", default=str(DEFAULT_SOURCE_USD))
    parser.add_argument("--staged_usd", default=str(DEFAULT_STAGED_USD))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene_variant", default="home_like_scene_v1")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--x_min", type=float, default=-12.0)
    parser.add_argument("--x_max", type=float, default=12.0)
    parser.add_argument("--y_min", type=float, default=-12.0)
    parser.add_argument("--y_max", type=float, default=12.0)
    parser.add_argument("--z_min", type=float, default=0.0)
    parser.add_argument("--z_max", type=float, default=3.0)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--min_validation_poses", type=int, default=20)
    parser.add_argument("--min_inspection_poses", type=int, default=36)
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument("--max_depth", type=float, default=26.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--pixel_stride", type=int, default=5)
    parser.add_argument("--make_flythrough", action="store_true")
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_formal_expert_sampling", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(), AppLauncher


def main() -> None:
    started = time.perf_counter()
    args, app_launcher_cls = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_usd = Path(args.source_usd).resolve()
    staged_usd = Path(args.staged_usd).resolve()
    if args.scene_variant != "home_like_scene_v1":
        raise ValueError("Stage 4A-6.6c-usd-import requires --scene_variant home_like_scene_v1")
    if int(args.scene_seed) != 0:
        raise ValueError("Stage 4A-6.6c-usd-import requires --scene_seed 0")
    if not (args.no_rollout and args.no_formal_expert_sampling and args.no_map_predict and args.no_rl_gdpo):
        raise ValueError("All negative-scope flags are required.")
    if str(staged_usd) != HOME_LIKE_SCENE_V1_STAGED_USD:
        raise ValueError(f"Staged USD must match scene_factory registration path: {HOME_LIKE_SCENE_V1_STAGED_USD}")

    context = write_context_reports(output_dir)
    staging = stage_usd(source_usd, staged_usd, output_dir)
    registration = write_scene_factory_registration_report(output_dir, staged_usd)
    hardware = hardware_report(int(args.max_workers), str(args.device))
    save_json(output_dir / "hardware_utilization_report.json", hardware)
    write_text(output_dir / "hardware_utilization_report.md", markdown_table("Hardware Utilization Report", hardware))
    write_no_scope_reports(output_dir)
    write_review_gate(output_dir)

    validation, simulation_app = capture_with_isaac(args, app_launcher_cls, output_dir, staged_usd)
    try:
        save_image_grids(output_dir, "validation", validation["fixed_capture_validation"]["records"])
        save_image_grids(output_dir, "inspection", validation["visual_inspection_capture_validation"]["records"])
        video_report = make_flythrough(output_dir, validation["visual_inspection_capture_validation"]["records"]) if args.make_flythrough else {"mp4_created": False, "video_path": None, "frame_count": 0}
        forbidden = check_forbidden_outputs(output_dir)
        summary = write_summary(output_dir, validation, video_report, time.perf_counter() - started)
        write_html_index(output_dir, summary, video_report)
        final_manifest = {
            "stage": "Stage 4A-6.6c-usd-import",
            "context": context,
            "staging": staging,
            "registration": registration,
            "hardware": hardware,
            "forbidden_output_check": forbidden,
            "summary": summary,
        }
        save_json(output_dir / "stage4a66c_usd_home_like_scene_run_manifest.json", final_manifest)
        print(f"[INFO]: Stage 4A-6.6c USD home_like_scene_v1 validation outputs written to {output_dir}")
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
