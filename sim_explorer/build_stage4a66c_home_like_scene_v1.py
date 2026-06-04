#!/usr/bin/env python3
"""Stage 4A-6.6c home_like_scene_v1 construction and visual validation.

This stage replaces the old larger_complex_scene_v1 outputs with a more
home-like scene.  It is scene construction, fixed-view capture, measured-only
observed_state construction, and human visual inspection packaging only.

It does not run rollout, expert sampling, selected action execution,
map_predict, SSCNet inference, prediction NPZ generation, replay buffer or
checkpoint creation, or RL/GDPO/PPO/BC/IL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
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
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image, ImageDraw

from depth_to_voxel import create_observed_grid, integrate_depth_frame, summarize_observed_grid
from scene_factory import LARGE_BOUNDS, build_home_like_scene_v1


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a66c_home_like_scene_v1_validation"
KENNEY_ASSET_PAGE_URL = "https://kenney.nl/assets/furniture-kit"
KENNEY_ASSET_DOWNLOAD_URL = "https://kenney.nl/media/pages/assets/furniture-kit/440e0608a4-1677580847/kenney_furniture-kit.zip"
KENNEY_ASSET_ZIP_SHA256 = "e67652d0932cee41683f74711c03d3e192a2af9979ef8e6b237711f5482d46b0"
KENNEY_ASSET_LICENSE = "Creative Commons Zero, CC0 1.0"
KENNEY_ASSET_BUNDLE_DIRNAME = "kenney_furniture_kit"
KENNEY_GLTF_RELATIVE_DIR = Path("extracted") / "Models" / "GLTF format"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
]
OLD_OUTPUT_DIRS = [
    WORKSPACE / "outputs/isaac_stage4a66_larger_complex_scene_v1_validation",
    WORKSPACE / "outputs/isaac_stage4a66a_scene_complexity_audit",
    WORKSPACE / "outputs/isaac_stage4a66b_gui_visual_inspection",
]
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
DEPTH_KEY = "distance_to_image_plane"
RGB_KEYS = ("rgb", "rgba")
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


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_from_mtime(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def write_blocker_report(output_dir: Path, reason: str, details: dict[str, Any]) -> None:
    report = {
        "stage": "Stage 4A-6.6c",
        "scene_id": "home_like_scene_v1",
        "blocked": True,
        "reason": reason,
        "procedural_composite_furniture_fallback_used": False,
        "cuboid_furniture_fallback_used": False,
        "created_at_utc": utc_now(),
        "details": details,
    }
    save_json(output_dir / "asset_blocker_report.json", report)
    write_text(output_dir / "asset_blocker_report.md", markdown_table("Asset Blocker Report", report))


def ensure_kenney_assets(output_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = output_dir / "downloaded_assets" / KENNEY_ASSET_BUNDLE_DIRNAME
    bundle_dir.mkdir(parents=True, exist_ok=True)
    zip_path = bundle_dir / "kenney_furniture-kit.zip"
    extract_dir = bundle_dir / "extracted"
    asset_root = bundle_dir / KENNEY_GLTF_RELATIVE_DIR
    downloaded_this_run = False

    if not zip_path.is_file():
        try:
            with urllib.request.urlopen(KENNEY_ASSET_DOWNLOAD_URL, timeout=180) as response:
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                with zip_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            downloaded_this_run = True
        except Exception as exc:
            details = {
                "source_url": KENNEY_ASSET_PAGE_URL,
                "download_url": KENNEY_ASSET_DOWNLOAD_URL,
                "error": str(exc),
            }
            write_blocker_report(output_dir, "Kenney CC0 furniture asset download failed", details)
            raise RuntimeError("Stage 4A-6.6c asset download failed; cuboid/procedural fallback is disabled") from exc

    zip_sha = sha256_file(zip_path)
    if zip_sha != KENNEY_ASSET_ZIP_SHA256:
        details = {
            "zip_path": str(zip_path),
            "expected_sha256": KENNEY_ASSET_ZIP_SHA256,
            "actual_sha256": zip_sha,
            "download_url": KENNEY_ASSET_DOWNLOAD_URL,
        }
        write_blocker_report(output_dir, "Downloaded Kenney asset hash mismatch", details)
        raise RuntimeError("Stage 4A-6.6c asset hash mismatch; cuboid/procedural fallback is disabled")

    if not asset_root.is_dir() or len(list(asset_root.glob("*.glb"))) < 80:
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
        except Exception as exc:
            details = {"zip_path": str(zip_path), "extract_dir": str(extract_dir), "error": str(exc)}
            write_blocker_report(output_dir, "Kenney furniture asset extraction failed", details)
            raise RuntimeError("Stage 4A-6.6c asset extraction failed; cuboid/procedural fallback is disabled") from exc

    license_path = extract_dir / "License.txt"
    license_text = license_path.read_text(encoding="utf-8", errors="replace") if license_path.is_file() else ""
    if "Creative Commons Zero" not in license_text or "CC0" not in license_text:
        details = {
            "license_path": str(license_path),
            "license_text_present": bool(license_text),
            "required_license": KENNEY_ASSET_LICENSE,
        }
        write_blocker_report(output_dir, "Kenney furniture asset license missing or insufficient", details)
        raise RuntimeError("Stage 4A-6.6c asset license invalid; cuboid/procedural fallback is disabled")

    used_asset_files = sorted({str(item["asset_file"]) for item in metadata.get("furniture_assets", [])})
    used_asset_records = []
    missing_files = []
    for asset_file in used_asset_files:
        path = asset_root / asset_file
        if not path.is_file():
            missing_files.append(asset_file)
            continue
        used_asset_records.append(
            {
                "asset_file": asset_file,
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "source_url": KENNEY_ASSET_PAGE_URL,
                "download_url": KENNEY_ASSET_DOWNLOAD_URL,
                "license": KENNEY_ASSET_LICENSE,
            }
        )
    if missing_files:
        details = {"asset_root": str(asset_root), "missing_asset_files": missing_files}
        write_blocker_report(output_dir, "Required Kenney furniture GLB assets missing after extraction", details)
        raise RuntimeError("Stage 4A-6.6c required assets missing; cuboid/procedural fallback is disabled")

    local_manifest = {
        "stage": "Stage 4A-6.6c",
        "scene_id": "home_like_scene_v1",
        "manifest_created_at_utc": utc_now(),
        "policy": "Local furniture USD/mesh assets were checked first; none sufficient were used as the primary furniture set.",
        "preexisting_local_furniture_assets_used": 0,
        "downloaded_asset_fallback_used": True,
        "searched_workspace_root": str(WORKSPACE),
        "downloaded_assets_are_not_counted_as_preexisting_local": True,
    }
    downloaded_manifest = {
        "stage": "Stage 4A-6.6c",
        "scene_id": "home_like_scene_v1",
        "manifest_created_at_utc": utc_now(),
        "asset_bundle": "Kenney Furniture Kit",
        "source_url": KENNEY_ASSET_PAGE_URL,
        "download_url": KENNEY_ASSET_DOWNLOAD_URL,
        "license": KENNEY_ASSET_LICENSE,
        "license_file": str(license_path),
        "license_file_sha256": sha256_file(license_path),
        "zip_path": str(zip_path),
        "zip_sha256": zip_sha,
        "expected_zip_sha256": KENNEY_ASSET_ZIP_SHA256,
        "zip_hash_verified": bool(zip_sha == KENNEY_ASSET_ZIP_SHA256),
        "download_time_utc": utc_from_mtime(zip_path),
        "downloaded_this_run": bool(downloaded_this_run),
        "asset_root": str(asset_root),
        "used_asset_file_count": len(used_asset_records),
        "used_assets": used_asset_records,
    }
    save_json(output_dir / "local_asset_manifest.json", local_manifest)
    write_text(output_dir / "local_asset_manifest.md", markdown_table("Local Asset Manifest", local_manifest))
    save_json(output_dir / "downloaded_asset_manifest.json", downloaded_manifest)
    write_text(
        output_dir / "downloaded_asset_manifest.md",
        markdown_table(
            "Downloaded Asset Manifest",
            {
                "asset_bundle": downloaded_manifest["asset_bundle"],
                "source_url": downloaded_manifest["source_url"],
                "download_url": downloaded_manifest["download_url"],
                "license": downloaded_manifest["license"],
                "zip_sha256": downloaded_manifest["zip_sha256"],
                "zip_hash_verified": downloaded_manifest["zip_hash_verified"],
                "download_time_utc": downloaded_manifest["download_time_utc"],
                "used_asset_file_count": downloaded_manifest["used_asset_file_count"],
            },
        ),
    )
    return {
        "asset_root": str(asset_root),
        "local_asset_manifest": local_manifest,
        "downloaded_asset_manifest": downloaded_manifest,
    }


def hardware_report(requested_max_workers: int, requested_device: str) -> dict[str, Any]:
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
        "requested_device": str(requested_device),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "GPU name": gpu_name,
        "cuda_available_if_torch_available": bool(cuda_available),
        "parallel_backend": "fixed-view Isaac rendering plus local measured-only map integration",
    }


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
            writer.writerow({key: json.dumps(row.get(key), sort_keys=True) if isinstance(row.get(key), (dict, list)) else row.get(key) for key in keys})


def center_of_bounds(bounds: dict[str, list[float]]) -> tuple[float, float]:
    return (
        0.5 * (float(bounds["x"][0]) + float(bounds["x"][1])),
        0.5 * (float(bounds["y"][0]) + float(bounds["y"][1])),
    )


def box_xy(spec: dict[str, Any]) -> tuple[float, float, float, float]:
    x, y, _ = (float(v) for v in spec["position"])
    sx, sy, _ = (float(v) for v in spec.get("size", [0.0, 0.0, 0.0]))
    return x - 0.5 * sx, y - 0.5 * sy, sx, sy


def pose_target(position: list[float], yaw_rad: float) -> list[float]:
    return [
        float(position[0] + math.cos(float(yaw_rad))),
        float(position[1] + math.sin(float(yaw_rad))),
        float(position[2]),
    ]


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


def normalize_rgb(source: np.ndarray) -> np.ndarray:
    rgb = source[..., :3]
    if np.issubdtype(rgb.dtype, np.floating):
        finite = rgb[np.isfinite(rgb)]
        if finite.size and float(finite.max()) <= 1.0:
            rgb = rgb * 255.0
    return np.clip(np.nan_to_num(rgb, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


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


def save_depth_png(path: Path, depth: np.ndarray, title: str) -> None:
    finite = depth[np.isfinite(depth) & (depth > 0.0)]
    if finite.size == 0:
        raise ValueError(f"No finite positive depth values for {path}")
    masked = np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan))
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    image = ax.imshow(masked, cmap="viridis", vmin=float(finite.min()), vmax=float(finite.max()))
    fig.colorbar(image, ax=ax, label="depth (m)")
    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def extract_rgb(camera: Any, label: str) -> tuple[np.ndarray, str, dict[str, Any]]:
    for key in RGB_KEYS:
        tensor = camera.data.output.get(key)
        if tensor is None:
            continue
        rgb = normalize_rgb(tensor[0].detach().cpu().numpy())
        stats = rgb_stats(rgb)
        if not stats["nonblank"]:
            raise ValueError(f"{label}: blank RGB {stats}")
        return rgb, key, stats
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


def write_context_and_deletion_reports(output_dir: Path) -> dict[str, Any]:
    context_entries = []
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        context_entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_stage4a66c": "Stage 4A-6.6c" in text,
                "contains_home_like_scene": "home_like_scene_v1" in text,
            }
        )
    save_json(output_dir / "loaded_context_manifest.json", {"stage": "Stage 4A-6.6c", "loaded_context_files": context_entries})
    write_text(output_dir / "loaded_context_manifest.md", list_md("Loaded Context Manifest", [f"`{item['path']}` sha256 `{item['sha256']}`" for item in context_entries]))
    removed_dirs = []
    already_absent_dirs = []
    for path in OLD_OUTPUT_DIRS:
        if path.exists():
            shutil.rmtree(path)
            removed_dirs.append(str(path))
        else:
            already_absent_dirs.append(str(path))
    deletion = {
        "stage": "Stage 4A-6.6c",
        "old_scene": "larger_complex_scene_v1",
        "old_output_dirs_expected_deleted": [str(path) for path in OLD_OUTPUT_DIRS],
        "old_output_dirs_removed": removed_dirs,
        "old_output_dirs_already_absent": already_absent_dirs,
        "old_output_dirs_absent": all(not path.exists() for path in OLD_OUTPUT_DIRS),
        "medium_three_rooms_preserved": True,
        "minimal_scene_preserved": True,
    }
    save_json(output_dir / "old_larger_scene_deletion_report.json", deletion)
    write_text(output_dir / "old_larger_scene_deletion_report.md", markdown_table("Old Larger Scene Deletion Report", deletion))
    save_json(output_dir / "old_scene_cleanup_manifest.json", deletion)
    write_text(output_dir / "old_scene_cleanup_manifest.md", markdown_table("Old Scene Cleanup Manifest", deletion))
    return deletion


def write_inventory_outputs(output_dir: Path, metadata: dict[str, Any]) -> None:
    semantic_room_inventory = [
        {
            "name": room["name"],
            "label": room.get("label"),
            "bounds": room.get("bounds"),
            "role": room.get("role"),
            "semantic_type": "room",
        }
        for room in metadata["rooms"]
    ]
    groups = {
        "rooms": metadata["rooms"],
        "corridors": metadata["corridors"],
        "openings": metadata["openings"],
        "walls": metadata["walls"],
        "semantic_room_inventory": semantic_room_inventory,
        "furniture_assets": metadata["furniture_assets"],
        "furniture_inventory": metadata["furniture_inventory"],
        "structural_primitive_obstacles": metadata["structural_primitive_obstacles"],
        "primitive_inventory": metadata["primitive_inventory"],
        "material_color_inventory": metadata["material_color_inventory"],
        "cuboid_obstacles": metadata["cuboid_obstacles"],
        "non_cuboid_primitives": metadata["non_cuboid_primitives"],
        "obstacles": metadata["obstacles"],
        "start_variants": metadata["start_variants"],
        "validation_camera_poses": metadata["validation_camera_poses"],
        "inspection_camera_poses": metadata["inspection_camera_poses"],
    }
    for name, rows in groups.items():
        save_json(output_dir / f"{name}.json", rows)
        if rows and isinstance(rows, list) and isinstance(rows[0], dict):
            write_csv(output_dir / f"{name}.csv", rows)
            write_text(output_dir / f"{name}.md", list_md(name.replace("_", " ").title(), [f"`{row.get('name', row.get('index'))}`" for row in rows[:120]]))
    save_json(output_dir / "materials.json", metadata["materials"])
    write_text(output_dir / "materials.md", markdown_table("Materials", metadata["materials"]))
    save_json(output_dir / "topology_graph.json", metadata["topology_graph"])
    write_text(
        output_dir / "topology_graph.md",
        markdown_table(
            "Topology Graph",
            {
                "node_count": len(metadata["topology_graph"]["nodes"]),
                "edge_count": len(metadata["topology_graph"]["edges"]),
                "narrow_passage_count": metadata["topology_summary"]["narrow_passage_count"],
                "loop_closure_count": metadata["topology_summary"]["loop_closure_count"],
            },
        ),
    )


def draw_base_layout(ax: Any, metadata: dict[str, Any], *, labels: bool = True, obstacles: bool = True, non_cuboids: bool = True) -> None:
    ax.set_xlim(metadata["map_bounds"]["x"])
    ax.set_ylim(metadata["map_bounds"]["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e2e8f0", linewidth=0.45)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    for corridor in metadata["corridors"]:
        x0, x1 = corridor["bounds"]["x"]
        y0, y1 = corridor["bounds"]["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#d8eef1", edgecolor="#47818c", linewidth=0.8, alpha=0.66))
        if labels:
            ax.text(*center_of_bounds(corridor["bounds"]), corridor["name"], ha="center", va="center", fontsize=6.0, color="#155e75")
    for room in metadata["rooms"]:
        x0, x1 = room["bounds"]["x"]
        y0, y1 = room["bounds"]["y"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f3ead7", edgecolor="#9b7444", linewidth=0.8, alpha=0.64))
        if labels:
            ax.text(*center_of_bounds(room["bounds"]), room["name"], ha="center", va="center", fontsize=6.2, color="#5b3b15")
    for wall in metadata["walls"]:
        x, y, w, h = box_xy(wall)
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#545b64", edgecolor="#2b3036", linewidth=0.20, alpha=0.88))
    if obstacles:
        for obstacle in metadata.get("structural_primitive_obstacles", metadata.get("cuboid_obstacles", [])):
            x, y, w, h = box_xy(obstacle)
            ax.add_patch(Rectangle((x, y), w, h, facecolor=obstacle["color"], edgecolor="#475569", linewidth=0.24, alpha=0.45, hatch="//"))
        for item in metadata.get("furniture_assets", []):
            x, y, w, h = box_xy(item)
            ax.add_patch(Rectangle((x, y), w, h, facecolor=item["color"], edgecolor="#111827", linewidth=0.28, alpha=0.84))
    if non_cuboids:
        for primitive in metadata["non_cuboid_primitives"]:
            x, y, _ = primitive["position"]
            r = float(primitive.get("radius") or 0.25)
            ax.scatter([x], [y], s=max(30, r * 360), c=[primitive["color"]], marker="o", edgecolors="#111827", linewidths=0.35, zorder=8)
    for opening in metadata["openings"]:
        ax.scatter([opening["center"][0]], [opening["center"][1]], s=22, marker="s", c="#22c55e", edgecolors="#14532d", linewidths=0.3, zorder=9)


def draw_pose_arrow(ax: Any, pose: dict[str, Any], color: str, label: str | None = None) -> None:
    x, y, _ = (float(v) for v in pose["position"])
    yaw = float(pose["yaw_rad"])
    ax.arrow(x, y, math.cos(yaw) * 0.70, math.sin(yaw) * 0.70, width=0.025, head_width=0.18, color=color, length_includes_head=True, zorder=11)
    ax.scatter([x], [y], c=[color], s=20, edgecolors="#ffffff", linewidths=0.35, zorder=12)
    ax.text(x + 0.10, y + 0.10, label or str(pose.get("index", pose.get("name", "pose"))), fontsize=5.6, color=color, weight="bold", zorder=13)


def save_layout_plots(output_dir: Path, metadata: dict[str, Any]) -> None:
    plots = [
        ("scene_layout_topdown_human.png", "home_like_scene_v1 topdown layout", True, True, True),
        ("room_corridor_opening_labels_topdown.png", "rooms, corridors, and openings", True, False, False),
        ("obstacle_labels_topdown.png", "mesh furniture assets and structural primitives", False, True, True),
    ]
    for filename, title, labels, obstacles, non_cuboids in plots:
        fig, ax = plt.subplots(figsize=(10.2, 9.8), constrained_layout=True)
        draw_base_layout(ax, metadata, labels=labels, obstacles=obstacles, non_cuboids=non_cuboids)
        if "opening" in filename:
            for opening in metadata["openings"]:
                ax.text(opening["center"][0] + 0.08, opening["center"][1] + 0.08, opening["name"].replace("door_", ""), fontsize=4.8, color="#166534")
        if "obstacle" in filename:
            for obstacle in [*metadata.get("furniture_assets", []), *metadata.get("structural_primitive_obstacles", [])]:
                x, y, w, h = box_xy(obstacle)
                ax.text(x + 0.5 * w, y + 0.5 * h, obstacle["category"][:7], fontsize=3.8, ha="center", va="center")
        ax.set_title(title)
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 9.8), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, non_cuboids=True)
    for start in metadata["start_variants"]:
        draw_pose_arrow(ax, start, "#7c3aed", start["name"].replace("start_", ""))
    ax.set_title("start variants")
    fig.savefig(output_dir / "start_variants_labeled_topdown.png", dpi=170)
    plt.close(fig)

    for filename, poses, title, color in [
        ("validation_poses_labeled_topdown.png", metadata["validation_camera_poses"], "validation poses", "#1554b7"),
        ("inspection_poses_labeled_topdown.png", metadata["inspection_camera_poses"], "inspection poses", "#0f766e"),
    ]:
        fig, ax = plt.subplots(figsize=(10.2, 9.8), constrained_layout=True)
        draw_base_layout(ax, metadata, labels=True, obstacles=True, non_cuboids=True)
        for pose in poses:
            draw_pose_arrow(ax, pose, color, str(pose["index"]))
        ax.set_title(title)
        fig.savefig(output_dir / filename, dpi=170)
        plt.close(fig)

    save_topology_graph(output_dir / "topology_graph_labeled.png", metadata)
    save_warning_regions(output_dir / "audit_warning_regions_topdown.png", metadata)
    save_closeups(output_dir, metadata)


def save_topology_graph(path: Path, metadata: dict[str, Any]) -> None:
    centers = {item["name"]: center_of_bounds(item["bounds"]) for item in [*metadata["rooms"], *metadata["corridors"]]}
    fig, ax = plt.subplots(figsize=(10.0, 9.6), constrained_layout=True)
    ax.set_xlim(metadata["map_bounds"]["x"])
    ax.set_ylim(metadata["map_bounds"]["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.45)
    for edge in metadata["topology_graph"]["edges"]:
        x0, y0 = centers[edge["source"]]
        x1, y1 = centers[edge["target"]]
        ax.plot([x0, x1], [y0, y1], color="#64748b", linewidth=1.0)
        ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), f"{edge['width_m']:.2f}", fontsize=5.0)
    for node in metadata["topology_graph"]["nodes"]:
        x, y = centers[node["id"]]
        color = "#d19038" if node["kind"] == "room" else "#5597a9"
        ax.scatter([x], [y], s=110, c=[color], edgecolors="#1f2937", zorder=4)
        ax.text(x, y + 0.22, node["id"], ha="center", va="bottom", fontsize=6.1, weight="bold")
    ax.set_title("home_like_scene_v1 topology graph")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def outline_region(ax: Any, metadata: dict[str, Any], name: str, color: str) -> None:
    for item in [*metadata["rooms"], *metadata["corridors"]]:
        if item["name"] == name:
            x0, x1 = item["bounds"]["x"]
            y0, y1 = item["bounds"]["y"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor=color, linewidth=2.0, linestyle="--", zorder=20))
            ax.text(x0, y1 + 0.15, name, fontsize=6.8, color=color, weight="bold", zorder=21)


def save_warning_regions(path: Path, metadata: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 9.8), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, non_cuboids=True)
    for name, color in [
        ("bathroom", "#dc2626"),
        ("entry_passage", "#dc2626"),
        ("laundry_storage", "#b45309"),
        ("main_corridor", "#7c3aed"),
        ("east_hall", "#0f766e"),
    ]:
        outline_region(ax, metadata, name, color)
    ax.set_title("review warning regions: narrow passages, service branch, loops")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_closeup(path: Path, metadata: dict[str, Any], title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=True)
    draw_base_layout(ax, metadata, labels=True, obstacles=True, non_cuboids=True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_closeups(output_dir: Path, metadata: dict[str, Any]) -> None:
    closeups = [
        ("closeup_living_room.png", "closeup: living room", (-11.8, -2.6), (-3.6, 5.6)),
        ("closeup_kitchen_dining_loop.png", "closeup: kitchen/dining loop", (-11.8, 3.8), (4.4, 11.8)),
        ("closeup_bedroom_wing.png", "closeup: bedroom wing", (2.6, 11.8), (-1.6, 11.8)),
        ("closeup_study_bathroom_hall.png", "closeup: study/bathroom/east hall", (-3.8, 11.8), (-7.8, -0.2)),
        ("closeup_entry_laundry.png", "closeup: entry/laundry", (-11.8, -2.4), (-9.8, -2.4)),
        ("closeup_narrow_passages.png", "closeup: narrow passages", (-4.0, 4.2), (-7.8, 1.0)),
        ("closeup_mesh_asset_furniture.png", "closeup: mesh asset furniture", (-11.8, 4.2), (-3.6, 11.8)),
    ]
    for filename, title, xlim, ylim in closeups:
        save_closeup(output_dir / filename, metadata, title, xlim, ylim)


def topdown_state(observed: np.ndarray) -> np.ndarray:
    occupied = np.any(observed == OCCUPIED, axis=2)
    free = np.any(observed == FREE, axis=2)
    top = np.full(observed.shape[:2], UNKNOWN, dtype=np.int8)
    top[free] = FREE
    top[occupied] = OCCUPIED
    return top


def save_observed_topdown(path: Path, observed: np.ndarray) -> None:
    top = topdown_state(observed)
    fig, ax = plt.subplots(figsize=(8.0, 7.6), constrained_layout=True)
    image = ax.imshow(top.T, origin="lower", cmap=STATE_CMAP, norm=STATE_NORM)
    fig.colorbar(image, ax=ax, ticks=[UNKNOWN, FREE, OCCUPIED], label="state")
    ax.set_title("measured-only observed_state topdown")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_image_grids(output_dir: Path, prefix: str, records: list[dict[str, Any]]) -> None:
    cols = 6
    rows = int(math.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.25), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, record in zip(axes, records):
        rgb = np.asarray(Image.open(output_dir / record["rgb_file"]).convert("RGB"))
        ax.imshow(rgb)
        ax.set_title(f"{record['index']:03d} {record.get('room', '')}", fontsize=7)
        ax.axis("off")
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.savefig(output_dir / f"{prefix}_rgb_grid.png", dpi=155)
    plt.close(fig)

    depths = [np.load(output_dir / record["depth_file"]) for record in records]
    finite_values = [depth[np.isfinite(depth) & (depth > 0.0)] for depth in depths]
    vmin = min(float(values.min()) for values in finite_values if values.size)
    vmax = max(float(values.max()) for values in finite_values if values.size)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.7, rows * 2.35), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    last = None
    for ax, record, depth in zip(axes, records, depths):
        last = ax.imshow(np.ma.masked_invalid(np.where(depth > 0.0, depth, np.nan)), cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{record['index']:03d} {record.get('room', '')}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(records) :]:
        ax.axis("off")
    fig.colorbar(last, ax=axes[: len(records)].tolist(), shrink=0.65, label="depth (m)")
    fig.savefig(output_dir / f"{prefix}_depth_grid.png", dpi=155)
    plt.close(fig)


async def convert_glb_to_usd_async(in_file: Path, out_file: Path, *, load_materials: bool = True, timeout_s: float = 120.0) -> bool:
    import asyncio
    import omni.kit.asset_converter

    def progress_callback(progress: float, total_steps: float) -> None:
        return None

    context = omni.kit.asset_converter.AssetConverterContext()
    settings = {
        "ignore_materials": not load_materials,
        "ignore_animation": True,
        "ignore_animations": True,
        "ignore_cameras": True,
        "ignore_lights": True,
        "single_mesh": False,
        "smooth_normals": True,
        "preview_surface": True,
        "export_preview_surface": True,
        "support_point_instancer": False,
        "embed_mdl_in_usd": False,
        "embed_textures": True,
        "use_meter_as_world_unit": True,
        "create_world_as_default_root_prim": True,
        "convert_stage_up_z": True,
    }
    for key, value in settings.items():
        if hasattr(context, key):
            setattr(context, key, value)
    task = omni.kit.asset_converter.get_instance().create_converter_task(str(in_file), str(out_file), progress_callback, context)
    deadline = time.perf_counter() + float(timeout_s)
    while True:
        success = await task.wait_until_finished()
        if success:
            return True
        if time.perf_counter() > deadline:
            return False
        await asyncio.sleep(0.1)


def prepare_converted_assets(output_dir: Path, metadata: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    import asyncio

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.kit.asset_converter")
    import omni.kit.asset_converter  # noqa: F401

    asset_preparation = metadata.get("asset_preparation") or {}
    asset_root = Path(asset_preparation.get("asset_root", ""))
    if not asset_root.is_dir():
        details = {"asset_root": str(asset_root), "asset_preparation_present": bool(asset_preparation)}
        write_blocker_report(output_dir, "Converted asset preparation failed because asset_root is missing", details)
        raise RuntimeError("Stage 4A-6.6c asset root missing; cuboid/procedural fallback is disabled")

    converted_root = output_dir / "converted_assets" / KENNEY_ASSET_BUNDLE_DIRNAME
    converted_root.mkdir(parents=True, exist_ok=True)
    used_asset_files = sorted({str(item["asset_file"]) for item in metadata.get("furniture_assets", [])})
    asset_usd_map: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for asset_file in used_asset_files:
        in_file = asset_root / asset_file
        out_file = converted_root / f"{Path(asset_file).stem}.usd"
        record: dict[str, Any] = {
            "asset_file": asset_file,
            "input_glb_path": str(in_file),
            "input_glb_sha256": sha256_file(in_file),
            "output_usd_path": str(out_file),
            "output_existed_before_conversion": out_file.is_file(),
            "source_url": KENNEY_ASSET_PAGE_URL,
            "license": KENNEY_ASSET_LICENSE,
        }
        if not in_file.is_file():
            record["conversion_success"] = False
            record["failure_reason"] = "input_glb_missing"
            failures.append(record)
            records.append(record)
            continue
        if not out_file.is_file() or out_file.stat().st_size <= 0:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            success = loop.run_until_complete(convert_glb_to_usd_async(in_file, out_file))
        else:
            success = True
        record["conversion_success"] = bool(success and out_file.is_file() and out_file.stat().st_size > 0)
        record["output_usd_sha256"] = sha256_file(out_file)
        record["output_usd_size_bytes"] = out_file.stat().st_size if out_file.is_file() else 0
        if record["conversion_success"]:
            asset_usd_map[asset_file] = str(out_file)
        else:
            record["failure_reason"] = "asset_converter_failed_or_empty_output"
            failures.append(record)
        records.append(record)

    report = {
        "stage": "Stage 4A-6.6c",
        "scene_id": "home_like_scene_v1",
        "asset_bundle": "Kenney Furniture Kit",
        "asset_root": str(asset_root),
        "converted_root": str(converted_root),
        "source_url": KENNEY_ASSET_PAGE_URL,
        "download_url": KENNEY_ASSET_DOWNLOAD_URL,
        "license": KENNEY_ASSET_LICENSE,
        "requested_distinct_asset_count": len(used_asset_files),
        "converted_asset_count": len(asset_usd_map),
        "conversion_success": len(failures) == 0 and len(asset_usd_map) == len(used_asset_files),
        "import_spawn_validation_pending": True,
        "spawned_mesh_furniture_count": 0,
        "fallback_to_procedural_or_cuboid_furniture": False,
        "records": records,
        "failures": failures,
    }
    save_json(output_dir / "asset_usd_map.json", asset_usd_map)
    save_json(output_dir / "conversion_import_validation_report.json", report)
    write_text(output_dir / "conversion_import_validation_report.md", markdown_table("Conversion Import Validation Report", {
        "requested_distinct_asset_count": report["requested_distinct_asset_count"],
        "converted_asset_count": report["converted_asset_count"],
        "conversion_success": report["conversion_success"],
        "import_spawn_validation_pending": report["import_spawn_validation_pending"],
        "fallback_to_procedural_or_cuboid_furniture": report["fallback_to_procedural_or_cuboid_furniture"],
    }))
    if failures or len(asset_usd_map) != len(used_asset_files):
        details = {"conversion_report": str(output_dir / "conversion_import_validation_report.json"), "failures": failures}
        write_blocker_report(output_dir, "One or more GLB furniture assets failed Isaac USD conversion", details)
        raise RuntimeError("Stage 4A-6.6c GLB to USD conversion failed; cuboid/procedural fallback is disabled")
    return asset_usd_map, report


def capture_scene(args: argparse.Namespace, app_launcher_cls: Any, output_dir: Path, metadata: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    if hasattr(args, "enable_cameras"):
        setattr(args, "enable_cameras", True)
    app_launcher = app_launcher_cls(args)
    simulation_app = app_launcher.app
    try:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
        sim.set_camera_view([0.0, -17.5, 15.0], [0.0, 0.0, 0.6])
        dome = sim_utils.DomeLightCfg(intensity=3600.0, color=(0.84, 0.85, 0.80))
        dome.func("/World/HomeLikeLight", dome)
        asset_usd_map, conversion_report = prepare_converted_assets(output_dir, metadata)
        runtime_metadata = build_home_like_scene_v1(
            seed=int(args.scene_seed),
            spawn=True,
            sim_utils_module=sim_utils,
            asset_usd_map=asset_usd_map,
        )
        runtime_metadata["asset_preparation"] = metadata.get("asset_preparation")
        runtime_metadata["conversion_import_validation"] = conversion_report
        sim_utils.create_prim("/World/CameraRig", "Xform")
        camera = Camera(
            cfg=CameraCfg(
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
        )
        sim.reset()
        intrinsic = camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(float)
        camera_info = {
            "sensor_api_depth_key": DEPTH_KEY,
            "width": int(args.camera_width),
            "height": int(args.camera_height),
            "max_depth": float(args.max_depth),
            "near_depth": 0.05,
            "intrinsic_matrix": intrinsic.tolist(),
            "fx": float(intrinsic[0, 0]),
            "fy": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]),
            "cy": float(intrinsic[1, 2]),
        }
        save_json(output_dir / "camera_info.json", camera_info)

        def set_pose(pose: dict[str, Any]) -> None:
            position = [float(v) for v in pose["position"]]
            target = pose_target(position, float(pose["yaw_rad"]))
            camera.set_world_poses_from_view(
                eyes=torch.tensor([position], dtype=torch.float32, device=sim.device),
                targets=torch.tensor([target], dtype=torch.float32, device=sim.device),
            )
            for _ in range(max(int(args.settle_steps), 1)):
                sim.step()
                camera.update(dt=sim.get_physics_dt())

        observed_state = create_observed_grid(metadata["map_bounds"], float(args.voxel_size))
        validation_records: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        step_summaries: list[dict[str, Any]] = []
        for pose in metadata["validation_camera_poses"]:
            idx = int(pose["index"])
            set_pose(pose)
            rgb, rgb_key, rstats = extract_rgb(camera, f"validation {idx}")
            depth, dstats = extract_depth(camera, f"validation {idx}")
            rgb_path = output_dir / f"validation_rgb_{idx:03d}.png"
            depth_path = output_dir / f"validation_depth_{idx:03d}.npy"
            depth_png_path = output_dir / f"validation_depth_color_{idx:03d}.png"
            pose_path = output_dir / f"validation_pose_{idx:03d}.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth)
            save_depth_png(depth_png_path, depth, f"validation depth {idx:03d}")
            pose_record = dict(pose)
            pose_record.update({"target": pose_target(pose["position"], pose["yaw_rad"]), "rgb_file": rgb_path.name, "depth_file": depth_path.name})
            save_json(pose_path, pose_record)
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
            transitions.append(transition)
            step_path = output_dir / f"observed_state_step{idx:03d}.npy"
            np.save(step_path, observed_state)
            step_summary = summarize_observed_grid(observed_state)
            step_summary.update({"step": idx, "transition": transition, "observed_state_file": str(step_path)})
            step_summaries.append(step_summary)
            validation_records.append(
                {
                    "index": idx,
                    "room": pose.get("room"),
                    "note": pose.get("note"),
                    "rgb_file": rgb_path.name,
                    "depth_file": depth_path.name,
                    "depth_color_file": depth_png_path.name,
                    "pose_file": pose_path.name,
                    "rgb_key_used": rgb_key,
                    "rgb_stats": rstats,
                    "depth_stats": dstats,
                }
            )
            print(f"[INFO]: validation {idx:03d} rgb_nonblank={rstats['nonblank']} depth_positive={dstats['positive_count']}")

        observed_summary = summarize_observed_grid(observed_state)
        observed_summary.update(
            {
                "shape_matches_expected": list(observed_state.shape) == metadata["expected_observed_state_shape"],
                "invalid_label_count": int(np.count_nonzero(~np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED]))),
                "measured_only": True,
                "prediction_used": False,
                "map_predict_called": False,
                "sscnet_inference_called": False,
                "step_summaries": step_summaries,
            }
        )
        np.save(output_dir / "observed_state_final.npy", observed_state)
        save_json(output_dir / "observed_summary.json", observed_summary)
        save_observed_topdown(output_dir / "observed_topdown_final.png", observed_state)

        inspection_records: list[dict[str, Any]] = []
        for pose in metadata["inspection_camera_poses"]:
            idx = int(pose["index"])
            set_pose(pose)
            rgb, rgb_key, rstats = extract_rgb(camera, f"inspection {idx}")
            depth, dstats = extract_depth(camera, f"inspection {idx}")
            rgb_path = output_dir / f"inspection_rgb_{idx:03d}.png"
            depth_path = output_dir / f"inspection_depth_{idx:03d}.npy"
            depth_png_path = output_dir / f"inspection_depth_color_{idx:03d}.png"
            pose_path = output_dir / f"inspection_pose_{idx:03d}.json"
            Image.fromarray(rgb).save(rgb_path)
            np.save(depth_path, depth)
            save_depth_png(depth_png_path, depth, f"inspection depth {idx:03d}")
            pose_record = dict(pose)
            pose_record.update({"target": pose_target(pose["position"], pose["yaw_rad"]), "rgb_file": rgb_path.name, "depth_file": depth_path.name})
            save_json(pose_path, pose_record)
            inspection_records.append(
                {
                    "index": idx,
                    "room": pose.get("room"),
                    "note": pose.get("note"),
                    "rgb_file": rgb_path.name,
                    "depth_file": depth_path.name,
                    "depth_color_file": depth_png_path.name,
                    "pose_file": pose_path.name,
                    "rgb_key_used": rgb_key,
                    "rgb_stats": rstats,
                    "depth_stats": dstats,
                }
            )
            print(f"[INFO]: inspection {idx:03d} rgb_nonblank={rstats['nonblank']} depth_positive={dstats['positive_count']}")

        conversion_report.update(
            {
                "import_spawn_validation_pending": False,
                "spawned_mesh_furniture_count": len(runtime_metadata.get("furniture_assets", [])),
                "spawned_distinct_usd_count": len(asset_usd_map),
                "import_spawn_success": len(runtime_metadata.get("furniture_assets", [])) >= 80 and len(asset_usd_map) == conversion_report["converted_asset_count"],
                "fallback_to_procedural_or_cuboid_furniture": False,
            }
        )
        save_json(output_dir / "conversion_import_validation_report.json", conversion_report)
        write_text(
            output_dir / "conversion_import_validation_report.md",
            markdown_table(
                "Conversion Import Validation Report",
                {
                    "requested_distinct_asset_count": conversion_report["requested_distinct_asset_count"],
                    "converted_asset_count": conversion_report["converted_asset_count"],
                    "conversion_success": conversion_report["conversion_success"],
                    "spawned_mesh_furniture_count": conversion_report["spawned_mesh_furniture_count"],
                    "import_spawn_success": conversion_report["import_spawn_success"],
                    "fallback_to_procedural_or_cuboid_furniture": conversion_report["fallback_to_procedural_or_cuboid_furniture"],
                },
            ),
        )

        scene_metadata = runtime_metadata | {
            "output_dir": str(output_dir),
            "camera": camera_info,
            "validation_capture_records": validation_records,
            "inspection_capture_records": inspection_records,
            "observed_summary": observed_summary,
            "prediction_used": False,
            "expert_used": False,
            "rollout_run": False,
            "rl_or_il_training_used": False,
        }
        save_json(output_dir / "scene_metadata.json", scene_metadata)
        return {
            "scene_load_validation": {
                "stage": "Stage 4A-6.6c",
                "scene_loaded": True,
                "scene_id": "home_like_scene_v1",
                "scene_seed": int(args.scene_seed),
                "isaac_startup_count": 1,
                "headless_requested": bool(getattr(args, "headless", False)),
                "spawned_wall_count": len(runtime_metadata["walls"]),
                "spawned_mesh_furniture_count": len(runtime_metadata["furniture_assets"]),
                "spawned_distinct_usd_asset_count": len(asset_usd_map),
                "spawned_cuboid_obstacle_count": len(runtime_metadata["cuboid_obstacles"]),
                "spawned_non_cuboid_primitive_count": len(runtime_metadata["non_cuboid_primitives"]),
                "procedural_composite_furniture_fallback_used": False,
                "cuboid_furniture_fallback_used": False,
                "conversion_import_validation_report": str(output_dir / "conversion_import_validation_report.json"),
            },
            "validation_capture": {
                "validation_pose_count": len(validation_records),
                "rgb_nonblank_count": sum(1 for item in validation_records if item["rgb_stats"]["nonblank"]),
                "depth_positive_count": sum(1 for item in validation_records if item["depth_stats"]["has_positive_finite_depth"]),
                "records": validation_records,
            },
            "inspection_capture": {
                "inspection_pose_count": len(inspection_records),
                "rgb_nonblank_count": sum(1 for item in inspection_records if item["rgb_stats"]["nonblank"]),
                "depth_positive_count": sum(1 for item in inspection_records if item["depth_stats"]["has_positive_finite_depth"]),
                "records": inspection_records,
            },
            "observed_state_validation": observed_summary,
            "observed_state_transition_summary": {
                "transition_records": transitions,
                "total_newly_observed": int(sum(item["newly_observed"] for item in transitions)),
                "total_unknown_to_free": int(sum(item["unknown_to_free"] for item in transitions)),
                "total_unknown_to_occupied": int(sum(item["unknown_to_occupied"] for item in transitions)),
                "invalid_label_count_final": int(observed_summary["invalid_label_count"]),
            },
        }, simulation_app
    except Exception:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        raise


def make_flythrough(output_dir: Path, inspection_records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_dir = output_dir / "flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame_idx in range(72):
        record = inspection_records[int(frame_idx / 72.0 * len(inspection_records)) % len(inspection_records)]
        image = Image.open(output_dir / record["rgb_file"]).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 44), fill=(18, 24, 32))
        draw.text((14, 12), f"home_like_scene_v1 | view {record['index']:03d} | {record.get('room', '')}", fill=(240, 244, 248))
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        image.save(frame_path)
        frames.append(frame_path)
    report: dict[str, Any] = {"frame_count": len(frames), "frame_dir": str(frame_dir), "mp4_created": False, "gif_created": False, "video_path": None}
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        mp4_path = output_dir / "home_like_scene_v1_flythrough.mp4"
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
            timeout=60,
        )
        report["ffmpeg_returncode"] = int(result.returncode)
        report["ffmpeg_stderr_tail"] = result.stderr[-2000:]
        if result.returncode == 0 and mp4_path.is_file() and mp4_path.stat().st_size > 0:
            report.update({"mp4_created": True, "video_path": str(mp4_path)})
            return report
    except Exception as exc:
        report["mp4_error"] = str(exc)
    gif_path = output_dir / "home_like_scene_v1_flythrough.gif"
    gif_frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for path in frames]
    gif_frames[0].save(gif_path, save_all=True, append_images=gif_frames[1:], duration=500, loop=0, optimize=True)
    report.update({"gif_created": True, "video_path": str(gif_path)})
    return report


def write_no_reports(output_dir: Path) -> None:
    reports = {
        "no_expert_sampling_report": {
            "formal_expert_sampling_run": False,
            "expert_dataset_generated": False,
            "selected_expert_action_executed": False,
            "formal_expert_sampling_ready": False,
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


def write_review_outputs(output_dir: Path) -> None:
    checklist = [
        "home_like_scene_v1 是否比旧 larger_complex_scene_v1 更像真实住宅？",
        "客厅、厨房、卧室、书房、卫生间、走廊是否都能辨认？",
        "家具是否像有效遮挡，而不是随机装饰？",
        "Kenney CC0 mesh/USD 家具资产是否明显改善视觉真实感？",
        "门洞、回环、窄通道是否自然？",
        "主走廊、east hall、north hall 的连接是否清楚？",
        "bathroom / laundry / entry 的窄通道是否可见且未完全堵死？",
        "start variants 是否分散且合理？",
        "validation/inspection views 是否覆盖主要房间？",
        "是否存在明显穿模、漂浮、墙体错位？",
        "是否适合作为下一步 Stage 4A-6.6d 审查对象？",
        "是否需要再次回到 Stage 4A-6.6c-revision 修改场景？",
    ]
    payload = {
        "stage": "Stage 4A-6.6c",
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "formal_expert_sampling_ready": False,
        "items": [{"id": idx + 1, "question": item, "status": "pending_user_review"} for idx, item in enumerate(checklist)],
    }
    save_json(output_dir / "human_visual_review_checklist.json", payload)
    write_text(output_dir / "human_visual_review_checklist.md", "\n".join(["# Human Visual Review Checklist", "", *[f"{idx + 1}. {item}" for idx, item in enumerate(checklist)], "", "Status: `pending_user_review`."]))
    gate = {
        "stage": "Stage 4A-6.6c",
        "human_visual_inspection_done": False,
        "user_needs_to_review_visuals": True,
        "formal_expert_sampling_ready": False,
        "stage4a66d_required_next": True,
        "stage4a67_executed": False,
    }
    save_json(output_dir / "manual_review_gate.json", gate)
    write_text(output_dir / "manual_review_gate.md", markdown_table("Manual Review Gate", gate))
    write_text(output_dir / "recommended_next_faithful_step.md", "# Recommended Next Faithful Step\n\nRun Stage 4A-6.6d review + human visual confirmation for `home_like_scene_v1`. Do not start formal expert sampling, rollout, map_predict, or RL/GDPO.")
    write_text(output_dir / "long_term_rl_gdpo_note.md", "# Long-Term RL/GDPO Note\n\nGDPO/RL/PPO/BC/IL remain future directions only. Stage 4A-6.6c did not train or create policy artifacts.")


def write_html(output_dir: Path, metadata: dict[str, Any], video_report: dict[str, Any]) -> None:
    image_names = [
        "scene_layout_topdown_human.png",
        "room_corridor_opening_labels_topdown.png",
        "obstacle_labels_topdown.png",
        "start_variants_labeled_topdown.png",
        "validation_poses_labeled_topdown.png",
        "inspection_poses_labeled_topdown.png",
        "topology_graph_labeled.png",
        "audit_warning_regions_topdown.png",
        "validation_rgb_grid.png",
        "validation_depth_grid.png",
        "inspection_rgb_grid.png",
        "inspection_depth_grid.png",
        "observed_topdown_final.png",
        "closeup_living_room.png",
        "closeup_kitchen_dining_loop.png",
        "closeup_bedroom_wing.png",
        "closeup_study_bathroom_hall.png",
        "closeup_entry_laundry.png",
        "closeup_narrow_passages.png",
        "closeup_mesh_asset_furniture.png",
    ]
    figures = "\n".join(f'<figure><img src="{name}" width="360"><figcaption>{name}</figcaption></figure>' for name in image_names)
    video_path = Path(video_report["video_path"]).name if video_report.get("video_path") else ""
    video_html = f'<video controls width="720" src="{video_path}"></video>' if video_path.endswith(".mp4") else f'<img src="{video_path}" width="720">'
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.6c home_like_scene_v1</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #18202a; background: #f7f8fa; }}
    figure {{ display: inline-block; margin: 10px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 360px; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.6c home_like_scene_v1</h1>
  <p>Rooms: <code>{len(metadata['rooms'])}</code>, corridors: <code>{len(metadata['corridors'])}</code>, openings: <code>{len(metadata['openings'])}</code>, furniture mesh assets: <code>{len(metadata['furniture_assets'])}</code>, structural primitives: <code>{len(metadata['structural_primitive_obstacles'])}</code>, materials: <code>{len(metadata['materials'])}</code>.</p>
  <p>human_visual_inspection_done: <code>false</code>. formal_expert_sampling_ready: <code>false</code>. Next: Stage 4A-6.6d review + human visual confirmation.</p>
  <h2>Visuals</h2>
  {figures}
  <h2>Flythrough</h2>
  {video_html}
  <h2>Checklist</h2>
  <p><a href="human_visual_review_checklist.md">human_visual_review_checklist.md</a></p>
</body>
</html>"""
    write_text(output_dir / "visual_inspection_index.html", body)


def write_summary(
    output_dir: Path,
    metadata: dict[str, Any],
    validation: dict[str, Any],
    video_report: dict[str, Any],
    hardware: dict[str, Any],
    deletion: dict[str, Any],
    elapsed_s: float,
) -> None:
    validation_capture = validation["validation_capture"]
    inspection_capture = validation["inspection_capture"]
    observed = validation["observed_state_validation"]
    summary = {
        "stage": "Stage 4A-6.6c",
        "scene_id": "home_like_scene_v1",
        "old_larger_scene_outputs_deleted": all(not path.exists() for path in OLD_OUTPUT_DIRS),
        "old_larger_builder_disabled": True,
        "old_larger_output_deletion_report": deletion,
        "home_like_scene_constructed": True,
        "isaac_headless_load_validated": bool(validation["scene_load_validation"]["scene_loaded"]),
        "rooms": len(metadata["rooms"]),
        "corridors": len(metadata["corridors"]),
        "openings": len(metadata["openings"]),
        "walls": len(metadata["walls"]),
        "furniture_objects": len(metadata["furniture_assets"]),
        "distinct_mesh_assets": metadata["topology_summary"]["distinct_mesh_asset_count"],
        "non_cuboid_or_composite_assets": metadata["topology_summary"]["non_cuboid_or_composite_asset_count"],
        "structural_primitives": len(metadata["structural_primitive_obstacles"]),
        "obstacles": len(metadata["obstacles"]),
        "non_cuboid_primitives": len(metadata["non_cuboid_primitives"]),
        "materials": len(metadata["materials"]),
        "start_variants": len(metadata["start_variants"]),
        "validation_poses": len(metadata["validation_camera_poses"]),
        "inspection_poses": len(metadata["inspection_camera_poses"]),
        "validation_rgb_nonblank": validation_capture["rgb_nonblank_count"],
        "validation_depth_positive": validation_capture["depth_positive_count"],
        "inspection_rgb_nonblank": inspection_capture["rgb_nonblank_count"],
        "inspection_depth_positive": inspection_capture["depth_positive_count"],
        "observed_state_shape": observed["shape"],
        "observed_state_shape_matches_expected": observed["shape_matches_expected"],
        "observed_state_invalid_label_count": observed["invalid_label_count"],
        "observed_ratio": observed["observed_ratio"],
        "measured_only": True,
        "downloaded_asset_manifest": str(output_dir / "downloaded_asset_manifest.json"),
        "local_asset_manifest": str(output_dir / "local_asset_manifest.json"),
        "conversion_import_validation_report": str(output_dir / "conversion_import_validation_report.json"),
        "human_visual_inspection_done": False,
        "formal_expert_sampling_ready": False,
        "next_stage": "Stage 4A-6.6d review + human visual confirmation",
        "no_expert_sampling": True,
        "no_rollout": True,
        "no_map_predict": True,
        "no_sscnet_inference": True,
        "no_rl_gdpo_bc_il": True,
        "hardware": hardware,
        "flythrough": video_report,
        "elapsed_seconds": float(elapsed_s),
    }
    save_json(output_dir / "stage4a66c_home_like_scene_v1_summary.json", summary)
    write_text(output_dir / "stage4a66c_home_like_scene_v1_summary.md", markdown_table("Stage 4A-6.6c Summary", summary))
    save_json(output_dir / "visual_inspection_summary.json", summary)
    write_text(output_dir / "visual_inspection_summary.md", markdown_table("Visual Inspection Summary", summary))


def check_forbidden_outputs(output_dir: Path) -> None:
    forbidden_exact = {
        "transitions.jsonl",
        "expert_dataset_manifest.jsonl",
        "expert_dataset_manifest.json",
        "rollout_index.html",
        "rollout_topdown_path.png",
        "observed_ratio_curve.png",
        "global_prediction_layer.npz",
        "selected_action_report.json",
        "action_execution_report.json",
    }
    hits = []
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in forbidden_exact or (name.startswith("prediction") and name.endswith(".npz")):
            hits.append(str(path))
        if any(token in name for token in ("replay_buffer", "policy_checkpoint", "frame003", "action002")):
            hits.append(str(path))
    if hits:
        raise RuntimeError(f"Forbidden Stage 4A-6.6c outputs created: {hits}")


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Stage 4A-6.6c home_like_scene_v1 validation.")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument("--max_depth", type=float, default=22.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--pixel_stride", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_expert_sampling", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args(), AppLauncher


def main() -> None:
    started = time.perf_counter()
    args, app_launcher_cls = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.scene_seed) != 0:
        raise ValueError("Stage 4A-6.6c requires scene_seed=0")
    if not (args.no_expert_sampling and args.no_rollout and args.no_map_predict and args.no_rl_gdpo):
        raise ValueError("Stage 4A-6.6c requires all negative-scope flags")

    metadata = build_home_like_scene_v1(seed=int(args.scene_seed), spawn=False)
    deletion = write_context_and_deletion_reports(output_dir)
    asset_preparation = ensure_kenney_assets(output_dir, metadata)
    metadata["asset_preparation"] = asset_preparation
    metadata["asset_sources"][0]["zip_sha256"] = asset_preparation["downloaded_asset_manifest"]["zip_sha256"]
    metadata["asset_sources"][0]["download_time_utc"] = asset_preparation["downloaded_asset_manifest"]["download_time_utc"]
    metadata["asset_sources"][0]["used_asset_file_count"] = asset_preparation["downloaded_asset_manifest"]["used_asset_file_count"]
    hardware = hardware_report(int(args.max_workers), str(args.device))
    save_json(output_dir / "hardware_utilization_report.json", hardware)
    write_text(output_dir / "hardware_utilization_report.md", markdown_table("Hardware Utilization Report", hardware))
    save_json(output_dir / "home_like_scene_v1_metadata.json", metadata)
    write_text(output_dir / "home_like_scene_v1_metadata.md", markdown_table("Home Like Scene V1 Metadata", {
        "rooms": len(metadata["rooms"]),
        "corridors": len(metadata["corridors"]),
        "openings": len(metadata["openings"]),
        "furniture_assets": len(metadata["furniture_assets"]),
        "distinct_mesh_assets": metadata["topology_summary"]["distinct_mesh_asset_count"],
        "non_cuboid_or_composite_assets": metadata["topology_summary"]["non_cuboid_or_composite_asset_count"],
        "structural_primitives": len(metadata["structural_primitive_obstacles"]),
        "materials": len(metadata["materials"]),
        "starts": len(metadata["start_variants"]),
        "validation_poses": len(metadata["validation_camera_poses"]),
        "inspection_poses": len(metadata["inspection_camera_poses"]),
        "formal_expert_sampling_ready": False,
    }))
    save_json(output_dir / "scene_complexity_targets.json", metadata["complexity_targets"])
    write_text(output_dir / "scene_complexity_targets.md", markdown_table("Scene Complexity Targets", metadata["complexity_targets"]))
    write_inventory_outputs(output_dir, metadata)
    save_layout_plots(output_dir, metadata)
    write_no_reports(output_dir)
    write_review_outputs(output_dir)

    validation, simulation_app = capture_scene(args, app_launcher_cls, output_dir, metadata)
    try:
        save_json(output_dir / "scene_load_validation.json", validation["scene_load_validation"])
        write_text(output_dir / "scene_load_validation.md", markdown_table("Scene Load Validation", validation["scene_load_validation"]))
        save_json(output_dir / "fixed_capture_validation.json", validation["validation_capture"])
        write_text(output_dir / "fixed_capture_validation.md", markdown_table("Fixed Capture Validation", {
            "validation_pose_count": validation["validation_capture"]["validation_pose_count"],
            "rgb_nonblank_count": validation["validation_capture"]["rgb_nonblank_count"],
            "depth_positive_count": validation["validation_capture"]["depth_positive_count"],
        }))
        save_json(output_dir / "inspection_capture_validation.json", validation["inspection_capture"])
        write_text(output_dir / "inspection_capture_validation.md", markdown_table("Inspection Capture Validation", {
            "inspection_pose_count": validation["inspection_capture"]["inspection_pose_count"],
            "rgb_nonblank_count": validation["inspection_capture"]["rgb_nonblank_count"],
            "depth_positive_count": validation["inspection_capture"]["depth_positive_count"],
        }))
        save_json(output_dir / "observed_state_validation_summary.json", validation["observed_state_validation"])
        write_text(output_dir / "observed_state_validation_summary.md", markdown_table("Observed State Validation Summary", {
            "shape": validation["observed_state_validation"]["shape"],
            "observed_ratio": validation["observed_state_validation"]["observed_ratio"],
            "invalid_label_count": validation["observed_state_validation"]["invalid_label_count"],
            "measured_only": True,
        }))
        save_json(output_dir / "observed_state_transition_summary.json", validation["observed_state_transition_summary"])
        write_text(output_dir / "observed_state_transition_summary.md", markdown_table("Observed State Transition Summary", validation["observed_state_transition_summary"]))
        save_image_grids(output_dir, "validation", validation["validation_capture"]["records"])
        save_image_grids(output_dir, "inspection", validation["inspection_capture"]["records"])
        video_report = make_flythrough(output_dir, validation["inspection_capture"]["records"])
        write_html(output_dir, metadata, video_report)
        check_forbidden_outputs(output_dir)
        write_summary(output_dir, metadata, validation, video_report, hardware, deletion, time.perf_counter() - started)
        print(f"[INFO]: Stage 4A-6.6c home_like_scene_v1 outputs written to {output_dir}")
    finally:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()
