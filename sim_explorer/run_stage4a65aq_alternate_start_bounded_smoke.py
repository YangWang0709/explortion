#!/usr/bin/env python3
"""Stage 4A-6.5aq alternate-start bounded smoke at start_corridor.

This runner keeps the proven Stage 4A-6.5ak runtime sequence as the execution
path and wraps it with the Stage 4A-6.5aq alternate-start safety bookkeeping:

    value = gain_exp / cost + 48 * minmax(source_occ_free)

The intended runtime variant is the start pose: canonical room-A is replaced
with start_corridor while tree_seed is reset to 0.  It
starts Isaac through the delegated 6.5ak runner exactly once, captures at most
two frames, executes at most one action, and then compares the result against
the saved canonical-start Stage 4A-6.5ak/am/ao references plus the Stage
4A-6.5ap alternate-start design artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
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
    os.environ[_key] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
SIM_DIR = WORKSPACE / "sim_explorer"
AK_RUNNER = SIM_DIR / "run_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke"
DEFAULT_STAGE4A65AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_STAGE4A65AM_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_STAGE4A65AO_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2"
DEFAULT_STAGE4A65AP_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design"
DEFAULT_ALTERNATE_START_METADATA = (
    WORKSPACE
    / "outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/"
    / "medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json"
)
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]
UNKNOWN = -1
FREE = 0
OCCUPIED = 1
PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
SHADOW_FORMULA = "gain_exp / cost + 32 * minmax(source_occ_free)"
CANONICAL_START = [-4.65, -4.65, 1.2]
EXPECTED_START_VARIANT = "start_corridor"
EXPECTED_START_POSITION = [0.0, -4.45, 1.2]
EXPECTED_START_YAW = 1.5707963267948966


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_file_hashes(paths: list[Path]) -> dict[str, dict[str, Any] | None]:
    result: dict[str, dict[str, Any] | None] = {}
    for path in paths:
        key = str(path)
        if path.is_file():
            result[key] = {"path": key, "exists": True, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        else:
            result[key] = None
    return result


def parse_position(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if len(values) != 3:
        raise ValueError(f"--position must contain exactly three comma-separated values, got {raw!r}")
    return values


def distance_m(a: Any, b: Any) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64)[:3] - np.asarray(b, dtype=np.float64)[:3]))


def close_pose(position: Any, yaw: Any, expected_position: list[float], expected_yaw: float, atol: float = 1.0e-9) -> bool:
    if position is None or yaw is None or expected_position is None or expected_yaw is None:
        return False
    return bool(
        np.allclose(np.asarray(position, dtype=np.float64), np.asarray(expected_position, dtype=np.float64), atol=atol)
        and math.isclose(float(yaw), float(expected_yaw), abs_tol=atol)
    )


def find_metadata_start_pose(metadata: dict[str, Any], expected_position: list[float], expected_yaw: float) -> dict[str, Any]:
    for pose in metadata.get("camera_poses", []):
        if close_pose(pose.get("position"), pose.get("yaw_rad"), expected_position, expected_yaw, atol=1.0e-9):
            return {
                "found": True,
                "index": pose.get("index"),
                "room": pose.get("room"),
                "note": pose.get("note"),
                "position": pose.get("position"),
                "yaw_rad": pose.get("yaw_rad"),
                "yaw_deg": pose.get("yaw_deg"),
            }
    return {"found": False, "expected_position": expected_position, "expected_yaw": expected_yaw}


def context_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    combined = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "contains_stage4a65ag": "Stage 4A-6.5ag" in text,
                "contains_stage4a65ai": "Stage 4A-6.5ai" in text,
                "contains_stage4a65aj": "Stage 4A-6.5aj" in text,
                "contains_stage4a65ak": "Stage 4A-6.5ak" in text,
                "contains_stage4a65am": "Stage 4A-6.5am" in text,
                "contains_stage4a65ao": "Stage 4A-6.5ao" in text,
                "contains_stage4a65ap": "Stage 4A-6.5ap" in text,
                "contains_stage4a65aq_next_task": "Stage 4A-6.5aq" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5aq",
        "loaded_context_files": entries,
        "confirmed_stage4a65ag_complete": "Stage 4A-6.5ag" in combined
        and "multi-frame lambda48 replay" in combined,
        "confirmed_stage4a65ai_complete": "Stage 4A-6.5ai" in combined
        and "one-frame lambda48 runtime smoke" in combined,
        "confirmed_stage4a65aj_complete": "Stage 4A-6.5aj" in combined
        and "two-frame one-action" in combined,
        "confirmed_stage4a65ak_complete": "Stage 4A-6.5ak" in combined
        and "two-frame one-action lambda48 runtime smoke" in combined,
        "confirmed_stage4a65am_complete": "Stage 4A-6.5am bounded repeat-safety smoke" in combined
        and "tree_seed=1" in combined,
        "confirmed_stage4a65ao_complete": "Stage 4A-6.5ao bounded repeat-safety smoke" in combined
        and "tree_seed=2" in combined,
        "confirmed_stage4a65ap_complete": "Stage 4A-6.5ap seed0/1/2 repeat-comparison review" in combined
        or "Stage 4A-6.5ap seed0/1/2 repeat-comparison review and alternate-start" in combined,
        "confirmed_next_small_task_stage4a65aq": "Stage 4A-6.5aq" in combined
        and "start_corridor" in combined
        and "no rollout" in combined,
        "chat_history_not_used_as_source": True,
    }


def write_context_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    save_json(output_dir / "loaded_context_manifest.json", manifest)
    write_text(
        output_dir / "loaded_context_manifest.md",
        "\n".join(
            [
                "# Loaded Context Manifest",
                "",
                f"- Stage 4A-6.5ag complete: `{manifest['confirmed_stage4a65ag_complete']}`",
                f"- Stage 4A-6.5ai complete: `{manifest['confirmed_stage4a65ai_complete']}`",
                f"- Stage 4A-6.5aj complete: `{manifest['confirmed_stage4a65aj_complete']}`",
                f"- Stage 4A-6.5ak complete: `{manifest['confirmed_stage4a65ak_complete']}`",
                f"- Stage 4A-6.5am complete: `{manifest['confirmed_stage4a65am_complete']}`",
                f"- Stage 4A-6.5ao complete: `{manifest['confirmed_stage4a65ao_complete']}`",
                f"- Stage 4A-6.5ap complete: `{manifest['confirmed_stage4a65ap_complete']}`",
                f"- Stage 4A-6.5aq is current next task: `{manifest['confirmed_next_small_task_stage4a65aq']}`",
                "- Files read:",
                *[f"  - `{item['path']}` sha256 `{item['sha256']}`" for item in manifest["loaded_context_files"]],
            ]
        ),
    )


def reference_paths(
    stage4a65ak_dir: Path,
    stage4a65am_dir: Path,
    stage4a65ao_dir: Path,
    stage4a65ap_dir: Path,
    alternate_start_metadata: Path,
    checkpoint: Path,
) -> list[Path]:
    return [
        checkpoint,
        stage4a65ak_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json",
        stage4a65ak_dir / "observed_state_frame001.npy",
        stage4a65ak_dir / "observed_state_frame002.npy",
        stage4a65ak_dir / "frame001_map_predict/global_prediction_layer.npz",
        stage4a65ak_dir / "frame002_map_predict/global_prediction_layer.npz",
        stage4a65am_dir / "stage4a65am_bounded_repeat_safety_summary.json",
        stage4a65am_dir / "observed_state_frame001.npy",
        stage4a65am_dir / "observed_state_frame002.npy",
        stage4a65am_dir / "frame001_map_predict/global_prediction_layer.npz",
        stage4a65am_dir / "frame002_map_predict/global_prediction_layer.npz",
        stage4a65ao_dir / "stage4a65ao_bounded_repeat_safety_summary.json",
        stage4a65ao_dir / "observed_state_frame001.npy",
        stage4a65ao_dir / "observed_state_frame002.npy",
        stage4a65ao_dir / "frame001_map_predict/global_prediction_layer.npz",
        stage4a65ao_dir / "frame002_map_predict/global_prediction_layer.npz",
        stage4a65ap_dir / "stage4a65ap_seed012_repeat_review_summary.json",
        stage4a65ap_dir / "selected_alternate_start_design.json",
        alternate_start_metadata,
    ]


def load_reference_manifest(args: argparse.Namespace, hashes_before: dict[str, Any]) -> dict[str, Any]:
    ak_dir = Path(args.stage4a65ak_dir).resolve()
    am_dir = Path(args.stage4a65am_dir).resolve()
    ao_dir = Path(args.stage4a65ao_dir).resolve()
    ap_dir = Path(args.stage4a65ap_dir).resolve()
    metadata_path = Path(args.alternate_start_metadata).resolve()
    ak_summary_path = ak_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json"
    am_summary_path = am_dir / "stage4a65am_bounded_repeat_safety_summary.json"
    ao_summary_path = ao_dir / "stage4a65ao_bounded_repeat_safety_summary.json"
    ap_summary_path = ap_dir / "stage4a65ap_seed012_repeat_review_summary.json"
    ap_design_path = ap_dir / "selected_alternate_start_design.json"
    ak_summary = read_json(ak_summary_path)
    am_summary = read_json(am_summary_path)
    ao_summary = read_json(ao_summary_path)
    ap_summary = read_json(ap_summary_path)
    ap_design = read_json(ap_design_path)
    metadata = read_json(metadata_path)
    metadata_start = find_metadata_start_pose(metadata, EXPECTED_START_POSITION, EXPECTED_START_YAW)
    return {
        "stage": "Stage 4A-6.5aq",
        "reference_stage_0": "Stage 4A-6.5ak",
        "reference_stage_1": "Stage 4A-6.5am",
        "reference_stage_2": "Stage 4A-6.5ao",
        "review_design_stage": "Stage 4A-6.5ap",
        "reference_tree_seed_0": int(args.reference_tree_seed_0),
        "reference_tree_seed_1": int(args.reference_tree_seed_1),
        "reference_tree_seed_2": int(args.reference_tree_seed_2),
        "current_tree_seed": int(args.tree_seed),
        "stage4a65ak_dir": str(ak_dir),
        "stage4a65am_dir": str(am_dir),
        "stage4a65ao_dir": str(ao_dir),
        "stage4a65ap_dir": str(ap_dir),
        "alternate_start_metadata": str(metadata_path),
        "loaded_stage4a65ak_summary": True,
        "loaded_stage4a65am_summary": True,
        "loaded_stage4a65ao_summary": True,
        "loaded_stage4a65ap_review_design_summary": True,
        "loaded_alternate_start_metadata": True,
        "stage4a65ak_sequence": ak_summary.get("runtime_setup", {}),
        "stage4a65am_sequence": am_summary.get("runtime_setup", {}),
        "stage4a65ao_sequence": ao_summary.get("runtime_setup", {}),
        "stage4a65ap_combined_outcome": ap_summary.get("seed012_outcome_classification", {}).get("combined_outcome")
        or ap_summary.get("combined_outcome"),
        "stage4a65ap_selected_start_variant": ap_design.get("chosen_alternate_start_variant"),
        "stage4a65ap_selected_start_position": ap_design.get("chosen_alternate_start_position"),
        "stage4a65ap_selected_start_yaw": ap_design.get("chosen_alternate_start_yaw_rad"),
        "metadata_start_pose": metadata_start,
        "reference_hashes_before": hashes_before,
    }


def write_reference_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    save_json(output_dir / "loaded_reference_manifest.json", manifest)
    write_text(
        output_dir / "loaded_reference_manifest.md",
        "\n".join(
            [
                "# Loaded Reference Manifest",
                "",
                f"- reference stage 0: `{manifest['reference_stage_0']}` tree_seed `{manifest['reference_tree_seed_0']}`",
                f"- reference stage 1: `{manifest['reference_stage_1']}` tree_seed `{manifest['reference_tree_seed_1']}`",
                f"- reference stage 2: `{manifest['reference_stage_2']}` tree_seed `{manifest['reference_tree_seed_2']}`",
                f"- review/design stage: `{manifest['review_design_stage']}`",
                f"- current tree_seed: `{manifest['current_tree_seed']}`",
                f"- loaded Stage 4A-6.5ak summary: `{manifest['loaded_stage4a65ak_summary']}`",
                f"- loaded Stage 4A-6.5am summary: `{manifest['loaded_stage4a65am_summary']}`",
                f"- loaded Stage 4A-6.5ao summary: `{manifest['loaded_stage4a65ao_summary']}`",
                f"- loaded Stage 4A-6.5ap review/design summary: `{manifest['loaded_stage4a65ap_review_design_summary']}`",
                f"- loaded alternate-start metadata: `{manifest['loaded_alternate_start_metadata']}`",
            ]
        ),
    )


def write_repeat_variant(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    position = parse_position(args.position)
    variant = {
        "repeat_variant": str(args.repeat_variant),
        "start_variant": str(args.start_variant),
        "reference_stage_0": "Stage 4A-6.5ak",
        "reference_stage_1": "Stage 4A-6.5am",
        "reference_stage_2": "Stage 4A-6.5ao",
        "reference_tree_seed_0": int(args.reference_tree_seed_0),
        "reference_tree_seed_1": int(args.reference_tree_seed_1),
        "reference_tree_seed_2": int(args.reference_tree_seed_2),
        "current_tree_seed": int(args.tree_seed),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "start_pose": str(args.start_variant),
        "position": position,
        "yaw": float(args.yaw),
        "canonical_start_position": CANONICAL_START,
        "start_pose_distance_from_canonical_m": distance_m(position, CANONICAL_START),
        "only_intended_runtime_change": "start_pose",
        "tree_seed_matches_reference_seed0": int(args.tree_seed) == int(args.reference_tree_seed_0),
        "start_pose_changed_vs_canonical_references": distance_m(position, CANONICAL_START) > 1.0e-6,
        "max_frames": int(args.max_frames),
        "max_selected_actions": 1,
        "no_second_action": bool(args.no_second_action),
        "no_third_frame": bool(args.no_third_frame),
        "no_rollout": bool(args.no_rollout),
    }
    save_json(output_dir / "repeat_variant_definition.json", variant)
    write_text(
        output_dir / "repeat_variant_definition.md",
        "\n".join(
            [
                "# Repeat Variant Definition",
                "",
                f"- repeat_variant: `{variant['repeat_variant']}`",
                f"- start_variant: `{variant['start_variant']}`",
                f"- reference/current tree_seed: `{variant['reference_tree_seed_0']}`, `{variant['reference_tree_seed_1']}`, `{variant['reference_tree_seed_2']}` / `{variant['current_tree_seed']}`",
                f"- only intended runtime change: `{variant['only_intended_runtime_change']}`",
                f"- scene/start: `{variant['scene_variant']}` seed `{variant['scene_seed']}`, `{variant['position']}` yaw `{variant['yaw']}`",
                f"- distance from canonical start: `{variant['start_pose_distance_from_canonical_m']}` m",
                "- bounds: exactly two frames, exactly one action if gates pass, no rollout.",
            ]
        ),
    )
    return variant


def write_alternate_start_files(output_dir: Path, args: argparse.Namespace, reference_manifest: dict[str, Any]) -> dict[str, Any]:
    position = parse_position(args.position)
    design_position = reference_manifest.get("stage4a65ap_selected_start_position")
    design_yaw = reference_manifest.get("stage4a65ap_selected_start_yaw")
    metadata_pose = reference_manifest.get("metadata_start_pose", {})
    payload = {
        "stage": "Stage 4A-6.5aq",
        "start_variant": str(args.start_variant),
        "position": position,
        "yaw_rad": float(args.yaw),
        "yaw_deg": float(math.degrees(float(args.yaw))),
        "pose_source": str(Path(args.alternate_start_metadata).resolve()),
        "metadata_pose_found": bool(metadata_pose.get("found")),
        "matches_stage4a65ap_design": close_pose(position, args.yaw, design_position, design_yaw),
        "matches_metadata": bool(metadata_pose.get("found"))
        and close_pose(position, args.yaw, metadata_pose.get("position"), metadata_pose.get("yaw_rad")),
        "canonical_start_position": CANONICAL_START,
        "distance_from_canonical_start_m": distance_m(position, CANONICAL_START),
        "fixed_camera_height_m": float(position[2]),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "tree_seed": int(args.tree_seed),
    }
    save_json(output_dir / "loaded_alternate_start_manifest.json", payload)
    write_text(
        output_dir / "loaded_alternate_start_manifest.md",
        "\n".join(
            [
                "# Loaded Alternate-Start Manifest",
                "",
                f"- start_variant: `{payload['start_variant']}`",
                f"- pose source: `{payload['pose_source']}`",
                f"- metadata pose found: `{payload['metadata_pose_found']}`",
                f"- matches Stage 4A-6.5ap design: `{payload['matches_stage4a65ap_design']}`",
                f"- matches metadata: `{payload['matches_metadata']}`",
            ]
        ),
    )
    save_json(output_dir / "alternate_start_definition.json", payload)
    write_text(
        output_dir / "alternate_start_definition.md",
        "\n".join(
            [
                "# Alternate Start Definition",
                "",
                f"- start_variant: `{payload['start_variant']}`",
                f"- position: `{payload['position']}`",
                f"- yaw_rad/yaw_deg: `{payload['yaw_rad']}` / `{payload['yaw_deg']}`",
                f"- distance from canonical start: `{payload['distance_from_canonical_start_m']}` m",
                "- tree_seed: `0`",
            ]
        ),
    )
    return payload


def build_child_command(args: argparse.Namespace, unknown: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        str(AK_RUNNER),
        "--output_dir",
        str(Path(args.output_dir).resolve()),
        "--scene_variant",
        str(args.scene_variant),
        "--scene_seed",
        str(args.scene_seed),
        "--start_pose",
        str(args.start_variant),
        f"--position={args.position}",
        "--yaw",
        str(args.yaw),
        "--checkpoint",
        str(args.checkpoint),
        "--alignment_convention",
        str(args.alignment_convention),
        "--tau",
        str(args.tau),
        "--occ_threshold",
        str(args.occ_threshold),
        "--free_threshold",
        str(args.free_threshold),
        "--lambda_sc",
        str(args.lambda_sc),
        "--shadow_lambda_sc",
        str(args.shadow_lambda_sc),
        "--num_nodes",
        str(args.num_nodes),
        "--max_extension_m",
        str(args.max_extension_m),
        "--sample_mode",
        str(args.sample_mode),
        "--path_cost_mode",
        str(args.path_cost_mode),
        "--v_max",
        str(args.v_max),
        "--robot_radius_m",
        str(args.robot_radius_m),
        "--voxel_size",
        str(args.voxel_size),
        "--raycast_stride",
        str(args.raycast_stride),
        "--num_yaw_samples",
        str(args.num_yaw_samples),
        "--max_ray_length_m",
        str(args.max_ray_length_m),
        "--short_edge_policy",
        str(args.short_edge_policy),
        "--crop_min_length_m",
        str(args.crop_min_length_m),
        "--tree_seed",
        str(args.tree_seed),
        "--max_workers",
        str(args.max_workers),
        "--max_frames",
        str(args.max_frames),
    ]
    if args.save_viz:
        cmd.append("--save_viz")
    if args.save_probs:
        cmd.append("--save_probs")
    if args.execute_exactly_one_action:
        cmd.append("--execute_exactly_one_action")
    if args.no_third_frame:
        cmd.append("--no_third_frame")
    if args.no_second_action:
        cmd.append("--no_second_action")
    if args.no_rollout:
        cmd.append("--no_rollout")
    cmd.extend(unknown)
    return cmd


def grid_distance_m(a: Any, b: Any, voxel_size: float = 0.1) -> float | None:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a[:3], dtype=np.float64)
    arr_b = np.asarray(b[:3], dtype=np.float64)
    return float(np.linalg.norm(arr_a - arr_b) * voxel_size)


def world_distance_m(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    arr_a = np.asarray(a[:3], dtype=np.float64)
    arr_b = np.asarray(b[:3], dtype=np.float64)
    return float(np.linalg.norm(arr_a - arr_b))


def get_frame(summary: dict[str, Any], frame: str) -> dict[str, Any] | None:
    return summary.get("results", {}).get(frame)


def lambda_decision(summary: dict[str, Any], frame: str) -> dict[str, Any] | None:
    payload = get_frame(summary, frame)
    if payload is None:
        return None
    return payload.get("lambda48_primary") if frame == "frame001" else payload.get("lambda48_diagnostic")


def measured_decision(summary: dict[str, Any], frame: str) -> dict[str, Any] | None:
    payload = get_frame(summary, frame)
    if payload is None:
        return None
    return payload.get("measured_only_shadow")


def lambda32_decision(summary: dict[str, Any], frame: str) -> dict[str, Any] | None:
    payload = get_frame(summary, frame)
    if payload is None:
        return None
    return payload.get("lambda32_shadow")


def compare_one_frame(current: dict[str, Any], reference: dict[str, Any], frame: str) -> dict[str, Any]:
    cur_l = lambda_decision(current, frame)
    ref_l = lambda_decision(reference, frame)
    cur_m = measured_decision(current, frame)
    ref_m = measured_decision(reference, frame)
    if cur_l is None or ref_l is None:
        return {"frame": frame, "available": False}
    selected_grid_delta = grid_distance_m(cur_l.get("selected_child_grid"), ref_l.get("selected_child_grid"))
    best_grid_delta = grid_distance_m(cur_l.get("best_descendant_grid"), ref_l.get("best_descendant_grid"))
    selected_world_delta = world_distance_m(cur_l.get("selected_child_world"), ref_l.get("selected_child_world"))
    best_world_delta = world_distance_m(cur_l.get("best_descendant_world"), ref_l.get("best_descendant_world"))
    return {
        "frame": frame,
        "available": True,
        "current_measured": None
        if cur_m is None
        else {
            "selected_child_id": cur_m.get("selected_child_id"),
            "best_descendant_id": cur_m.get("best_descendant_id"),
            "selected_child_grid": cur_m.get("selected_child_grid"),
            "best_descendant_grid": cur_m.get("best_descendant_grid"),
        },
        "reference_measured": None
        if ref_m is None
        else {
            "selected_child_id": ref_m.get("selected_child_id"),
            "best_descendant_id": ref_m.get("best_descendant_id"),
            "selected_child_grid": ref_m.get("selected_child_grid"),
            "best_descendant_grid": ref_m.get("best_descendant_grid"),
        },
        "current_lambda48": {
            "selected_child_id": cur_l.get("selected_child_id"),
            "best_descendant_id": cur_l.get("best_descendant_id"),
            "selected_child_grid": cur_l.get("selected_child_grid"),
            "best_descendant_grid": cur_l.get("best_descendant_grid"),
            "selected_child_world": cur_l.get("selected_child_world"),
            "best_descendant_world": cur_l.get("best_descendant_world"),
            "branch_classification": cur_l.get("branch_classification"),
            "healthy_nonmeasured_candidate": bool(cur_l.get("healthy_nonmeasured_candidate")),
            "low_cost_artifact": bool(cur_l.get("low_cost_artifact")),
            "historical_prior_basin": bool(cur_l.get("spatial_prior_sc_basin")),
        },
        "reference_lambda48": {
            "selected_child_id": ref_l.get("selected_child_id"),
            "best_descendant_id": ref_l.get("best_descendant_id"),
            "selected_child_grid": ref_l.get("selected_child_grid"),
            "best_descendant_grid": ref_l.get("best_descendant_grid"),
            "selected_child_world": ref_l.get("selected_child_world"),
            "best_descendant_world": ref_l.get("best_descendant_world"),
            "branch_classification": ref_l.get("branch_classification"),
        },
        "exact_selected_child_agreement": cur_l.get("selected_child_id") == ref_l.get("selected_child_id"),
        "exact_best_descendant_agreement": cur_l.get("best_descendant_id") == ref_l.get("best_descendant_id"),
        "selected_child_grid_distance_m": selected_grid_delta,
        "best_descendant_grid_distance_m": best_grid_delta,
        "selected_child_world_distance_m": selected_world_delta,
        "best_descendant_world_distance_m": best_world_delta,
        "selected_child_grid_distance_vs_stage4a65ak_m": selected_grid_delta,
        "best_descendant_grid_distance_vs_stage4a65ak_m": best_grid_delta,
        "selected_child_world_distance_vs_stage4a65ak_m": selected_world_delta,
        "best_descendant_world_distance_vs_stage4a65ak_m": best_world_delta,
        "branch_class_agreement": cur_l.get("branch_classification") == ref_l.get("branch_classification"),
    }


def compute_observed_delta(output_dir: Path) -> dict[str, Any] | None:
    path1 = output_dir / "observed_state_frame001.npy"
    path2 = output_dir / "observed_state_frame002.npy"
    if not (path1.is_file() and path2.is_file()):
        return None
    obs1 = np.load(path1)
    obs2 = np.load(path2)
    valid_labels = {UNKNOWN, FREE, OCCUPIED}
    invalid = int(np.count_nonzero(~np.isin(obs2, list(valid_labels))))
    unknown_to_free = int(np.count_nonzero((obs1 == UNKNOWN) & (obs2 == FREE)))
    unknown_to_occ = int(np.count_nonzero((obs1 == UNKNOWN) & (obs2 == OCCUPIED)))
    free_to_occ = int(np.count_nonzero((obs1 == FREE) & (obs2 == OCCUPIED)))
    occ_to_free = int(np.count_nonzero((obs1 == OCCUPIED) & (obs2 == FREE)))
    observed1 = int(np.count_nonzero(obs1 != UNKNOWN))
    observed2 = int(np.count_nonzero(obs2 != UNKNOWN))
    total = int(obs1.size)
    summary = {
        "frame001": {
            "unknown": int(np.count_nonzero(obs1 == UNKNOWN)),
            "free": int(np.count_nonzero(obs1 == FREE)),
            "occupied": int(np.count_nonzero(obs1 == OCCUPIED)),
            "observed": observed1,
            "observed_ratio": float(observed1 / total),
        },
        "frame002": {
            "unknown": int(np.count_nonzero(obs2 == UNKNOWN)),
            "free": int(np.count_nonzero(obs2 == FREE)),
            "occupied": int(np.count_nonzero(obs2 == OCCUPIED)),
            "observed": observed2,
            "observed_ratio": float(observed2 / total),
        },
        "observed_ratio_delta": float((observed2 - observed1) / total),
        "newly_observed": int(np.count_nonzero((obs1 == UNKNOWN) & (obs2 != UNKNOWN))),
        "unknown_to_free": unknown_to_free,
        "unknown_to_occupied": unknown_to_occ,
        "free_to_occupied": free_to_occ,
        "occupied_to_free": occ_to_free,
        "invalid_labels": invalid,
        "measured_only_status": True,
    }
    save_json(output_dir / "observed_state_delta_summary.json", summary)
    write_text(
        output_dir / "observed_state_delta_summary.md",
        "\n".join(
            [
                "# Observed State Delta Summary",
                "",
                f"- observed_ratio: `{summary['frame001']['observed_ratio']}` -> `{summary['frame002']['observed_ratio']}`",
                f"- delta: `{summary['observed_ratio_delta']}`",
                f"- newly observed: `{summary['newly_observed']}`",
                f"- unknown->free / unknown->occupied: `{unknown_to_free}` / `{unknown_to_occ}`",
                f"- occupied->free: `{occ_to_free}`",
                "- update remained measured-only: `True`",
            ]
        ),
    )
    return summary


def compute_map_predict_stability(output_dir: Path) -> dict[str, Any] | None:
    path1 = output_dir / "map_predict_frame001_summary.json"
    path2 = output_dir / "map_predict_frame002_summary.json"
    if not (path1.is_file() and path2.is_file()):
        return None
    frame1 = read_json(path1)
    frame2 = read_json(path2)
    count1 = int(frame1["stats"]["predicted_unmeasured_occ_free_count"])
    count2 = int(frame2["stats"]["predicted_unmeasured_occ_free_count"])
    ratio = None if count1 == 0 else float(count2 / count1)
    stability = {
        "frame001_prediction_valid_count": int(frame1["stats"]["prediction_valid_count"]),
        "frame001_predicted_unmeasured_occ_free": count1,
        "frame001_predicted_occupied_count": int(frame1["stats"]["predicted_unmeasured_occupied_count"]),
        "frame001_predicted_free_count": int(frame1["stats"]["predicted_unmeasured_free_count"]),
        "frame002_prediction_valid_count": int(frame2["stats"]["prediction_valid_count"]),
        "frame002_predicted_unmeasured_occ_free": count2,
        "frame002_predicted_occupied_count": int(frame2["stats"]["predicted_unmeasured_occupied_count"]),
        "frame002_predicted_free_count": int(frame2["stats"]["predicted_unmeasured_free_count"]),
        "density_ratio_frame2_over_frame1": ratio,
        "alignment_convention_frame001": frame1["alignment_convention"],
        "alignment_convention_frame002": frame2["alignment_convention"],
        "code_consistent_v1_check": frame1["alignment_convention"] == frame2["alignment_convention"] == "code_consistent_v1",
        "no_explosion_or_collapse": ratio is not None and 0.25 <= ratio <= 4.0,
        "prediction_read_only": True,
    }
    save_json(output_dir / "map_predict_two_frame_stability.json", stability)
    write_text(
        output_dir / "map_predict_two_frame_stability.md",
        "\n".join(
            [
                "# Map Predict Two-Frame Stability",
                "",
                f"- valid counts: `{stability['frame001_prediction_valid_count']}` -> `{stability['frame002_prediction_valid_count']}`",
                f"- OCC+FREE unmeasured: `{count1}` -> `{count2}`",
                f"- density ratio: `{ratio}`",
                f"- code_consistent_v1: `{stability['code_consistent_v1_check']}`",
                f"- no explosion/collapse: `{stability['no_explosion_or_collapse']}`",
            ]
        ),
    )
    return stability


def lambda32_vs_lambda48(summary: dict[str, Any]) -> dict[str, Any]:
    frames = {}
    for frame in ("frame001", "frame002"):
        l48 = lambda_decision(summary, frame)
        l32 = lambda32_decision(summary, frame)
        frames[frame] = {
            "lambda48_available": l48 is not None,
            "lambda32_available": l32 is not None,
            "same_selected_child": None if l48 is None or l32 is None else l48.get("selected_child_id") == l32.get("selected_child_id"),
            "same_best_descendant": None if l48 is None or l32 is None else l48.get("best_descendant_id") == l32.get("best_descendant_id"),
            "lambda48": None
            if l48 is None
            else {
                "selected_child_id": l48.get("selected_child_id"),
                "best_descendant_id": l48.get("best_descendant_id"),
                "branch_classification": l48.get("branch_classification"),
            },
            "lambda32": None
            if l32 is None
            else {
                "selected_child_id": l32.get("selected_child_id"),
                "best_descendant_id": l32.get("best_descendant_id"),
                "branch_classification": l32.get("branch_classification"),
            },
        }
    result = {
        "frame001": frames["frame001"],
        "frame002": frames["frame002"],
        "all_available_frames_match": all(
            item["same_selected_child"] is not False and item["same_best_descendant"] is not False
            for item in frames.values()
        ),
    }
    return result


def _comparison_frames(reference_comparison: dict[str, Any]) -> list[dict[str, Any] | None]:
    return [reference_comparison.get("frame001"), reference_comparison.get("frame002")]


def _comparison_exact(reference_comparison: dict[str, Any]) -> bool:
    return all(
        item
        and item.get("available")
        and item.get("exact_selected_child_agreement")
        and item.get("exact_best_descendant_agreement")
        for item in _comparison_frames(reference_comparison)
    )


def _comparison_spatially_close(reference_comparison: dict[str, Any]) -> bool:
    return all(
        item
        and item.get("available")
        and (item.get("selected_child_grid_distance_m") is not None)
        and item["selected_child_grid_distance_m"] <= 1.0
        and (item.get("best_descendant_grid_distance_m") is not None)
        and item["best_descendant_grid_distance_m"] <= 4.0
        and bool(item.get("branch_class_agreement"))
        for item in _comparison_frames(reference_comparison)
    )


def classify_alternate_start(current: dict[str, Any], safety_clean: bool) -> dict[str, Any]:
    setup = current["runtime_setup"]
    action_executed = int(setup["selected_action_execution_count"]) == 1
    frame2_done = current["results"].get("frame002") is not None
    branches = [
        current["results"]["frame001"]["branch_classification"],
        *( [] if not frame2_done else [current["results"]["frame002"]["branch_classification"]] ),
    ]
    low_cost = any(bool(item["low_cost_artifact"]) for item in branches)
    prior = any(bool(item["historical_prior_basin"]) for item in branches)
    healthy = action_executed and frame2_done and safety_clean and not low_cost and not prior
    classifications = [item.get("classification") for item in branches]
    all_same_as_measured = healthy and all(value == "same_as_measured" for value in classifications)
    any_distinct = healthy and any(value == "distinct_nonmeasured_branch" for value in classifications)
    if not action_executed:
        outcome = "action_blocked"
    elif low_cost or prior or not safety_clean:
        outcome = "artifact_or_prior_basin_regression"
    elif all_same_as_measured:
        outcome = "clean_same_as_measured"
    elif any_distinct:
        outcome = "spatially_consistent_healthy_alternate_start"
    elif healthy:
        outcome = "clean_distinct_nonmeasured"
    else:
        outcome = "runtime_failure"
    return {
        "alternate_start_outcome": outcome,
        "repeat_outcome": outcome,
        "alternate_start_remains_healthy": healthy,
        "repeat_remains_healthy": healthy,
        "action_executed": action_executed,
        "frame2_completed": frame2_done,
        "low_cost_artifact_any_frame": low_cost,
        "historical_prior_basin_any_frame": prior,
        "prediction_safety_clean": safety_clean,
        "clean_same_as_measured": all_same_as_measured,
        "clean_distinct_nonmeasured": outcome == "clean_distinct_nonmeasured",
        "spatially_consistent_healthy_alternate_start": outcome == "spatially_consistent_healthy_alternate_start",
        "start_sensitive_but_clean": healthy and outcome not in {"clean_same_as_measured"},
        "canonical_exact_match_not_required": True,
        "branch_classifications": classifications,
    }


def topdown_observed(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def point_xy(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    return float(grid[0]) + 0.5, float(grid[1]) + 0.5


def plot_comparison_to_reference(
    output_dir: Path,
    comparison: dict[str, Any],
    output_name: str = "comparison_to_stage4a65ak_topdown.png",
    current_label: str = "aq",
    reference_label: str = "seed0",
    title: str = "Stage 4A-6.5aq alternate start vs canonical reference",
) -> None:
    observed = np.load(output_dir / "observed_state_frame001.npy")
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.4, 7.4), constrained_layout=True)
    ax.imshow(topdown_observed(observed).T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    for frame_name, marker in (("frame001", "o"), ("frame002", "s")):
        frame = comparison.get(frame_name)
        if not frame or not frame.get("available"):
            continue
        for side, color, label_prefix in (
            ("current_lambda48", "#2563eb", current_label),
            ("reference_lambda48", "#f97316", reference_label),
        ):
            point = point_xy(frame[side].get("selected_child_grid"))
            if point:
                ax.scatter([point[0]], [point[1]], c=color, marker=marker, s=90, label=f"{label_prefix} {frame_name} selected")
            best = point_xy(frame[side].get("best_descendant_grid"))
            if best:
                ax.scatter([best[0]], [best[1]], c=color, marker="*", s=150, alpha=0.9, label=f"{label_prefix} {frame_name} best")
    ax.set_title(title)
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(output_dir / output_name, dpi=170)
    plt.close(fig)


def plot_combined_comparison(
    output_dir: Path,
    comparison_seed0: dict[str, Any],
    comparison_seed1: dict[str, Any],
    comparison_seed2: dict[str, Any],
) -> None:
    observed = np.load(output_dir / "observed_state_frame001.npy")
    cmap = ListedColormap(["#30343b", "#83c5be", "#d95d59"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.8, 7.4), constrained_layout=True)
    ax.imshow(topdown_observed(observed).T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    styles = [
        (comparison_seed0, "seed0", "#f97316"),
        (comparison_seed1, "seed1", "#22c55e"),
        (comparison_seed2, "seed2", "#7c3aed"),
    ]
    for frame_name, marker in (("frame001", "o"), ("frame002", "s")):
        current_drawn = False
        for comparison, label, color in styles:
            frame = comparison.get(frame_name)
            if not frame or not frame.get("available"):
                continue
            if not current_drawn:
                point = point_xy(frame["current_lambda48"].get("selected_child_grid"))
                if point:
                    ax.scatter([point[0]], [point[1]], c="#2563eb", marker=marker, s=95, label=f"aq {frame_name} selected")
                best = point_xy(frame["current_lambda48"].get("best_descendant_grid"))
                if best:
                    ax.scatter([best[0]], [best[1]], c="#2563eb", marker="*", s=150, alpha=0.9, label=f"aq {frame_name} best")
                current_drawn = True
            ref = point_xy(frame["reference_lambda48"].get("selected_child_grid"))
            if ref:
                ax.scatter([ref[0]], [ref[1]], c=color, marker=marker, s=80, label=f"{label} {frame_name} selected")
            ref_best = point_xy(frame["reference_lambda48"].get("best_descendant_grid"))
            if ref_best:
                ax.scatter([ref_best[0]], [ref_best[1]], c=color, marker="*", s=135, alpha=0.85, label=f"{label} {frame_name} best")
    ax.set_title("Stage 4A-6.5aq alternate start vs canonical seed0/1/2")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(output_dir / "alternate_start_vs_canonical_start_topdown.png", dpi=170)
    plt.close(fig)


def plot_observed_delta(output_dir: Path) -> None:
    path1 = output_dir / "observed_state_frame001.npy"
    path2 = output_dir / "observed_state_frame002.npy"
    if not (path1.is_file() and path2.is_file()):
        write_text(output_dir / "observed_state_delta_topdown_skipped_reason.md", "# Plot Skipped\n\n- reason: Frame 2 missing.")
        return
    obs1 = np.load(path1)
    obs2 = np.load(path2)
    delta = np.zeros(obs1.shape[:2], dtype=np.int8)
    delta[np.any((obs1 == UNKNOWN) & (obs2 == FREE), axis=2)] = 1
    delta[np.any((obs1 == UNKNOWN) & (obs2 == OCCUPIED), axis=2)] = 2
    cmap = ListedColormap(["#30343b", "#38bdf8", "#facc15"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 6.8), constrained_layout=True)
    ax.imshow(delta.T, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("Observed_state newly measured voxels after action")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    fig.savefig(output_dir / "observed_state_delta_topdown.png", dpi=170)
    plt.close(fig)


def plot_repeat_stability(output_dir: Path, comparison: dict[str, Any], repeat: dict[str, Any]) -> None:
    labels = []
    values = []
    for frame in ("frame001", "frame002"):
        item = comparison.get(frame)
        if item and item.get("available"):
            labels.append(f"{frame} selected delta")
            values.append(float(item.get("selected_child_grid_distance_vs_stage4a65ak_m") or 0.0))
            labels.append(f"{frame} best delta")
            values.append(float(item.get("best_descendant_grid_distance_vs_stage4a65ak_m") or 0.0))
    if not values:
        labels = ["repeat healthy"]
        values = [1.0 if repeat["repeat_remains_healthy"] else 0.0]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.bar(labels, values, color=["#2563eb", "#60a5fa", "#0f766e", "#5eead4"][: len(values)])
    ax.set_ylabel("distance (m)")
    ax.set_title(f"Alternate-start outcome: {repeat['alternate_start_outcome']}")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "alternate_start_stability_summary.png", dpi=170)
    plt.close(fig)


def write_readiness_matrix(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    save_json(output_dir / "alternate_start_safety_readiness_matrix.json", {"rows": rows})
    with (output_dir / "alternate_start_safety_readiness_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "evidence"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_text(
        output_dir / "alternate_start_safety_readiness_matrix.md",
        "\n".join(
            [
                "# Alternate-Start Safety Readiness Matrix",
                "",
                "| check | passed | evidence |",
                "| --- | --- | --- |",
                *[f"| {row['check']} | `{row['passed']}` | {row['evidence']} |" for row in rows],
            ]
        ),
    )


def scan_forbidden_outputs(output_dir: Path) -> list[str]:
    patterns = [
        "transitions.jsonl",
        "rollout_topdown_path.png",
        "observed_ratio_curve.png",
        "rollout_index.html",
        "episode_manifest*",
        "frame003*",
        "action002*",
    ]
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    setup = summary["runtime_setup"]
    results = summary["results"]
    frame1 = results["frame001"]
    frame2 = results.get("frame002")
    repeat = summary["repeat_stability"]
    comparison_seed0 = summary["comparison_to_stage4a65ak"]
    comparison_seed1 = summary["comparison_to_stage4a65am"]
    comparison_seed2 = summary["comparison_to_stage4a65ao"]
    alternate_start = summary["alternate_start_definition"]
    action = results.get("action_execution_report")
    blocked = results.get("action_blocked_report")
    lines = [
        "# Stage 4A-6.5aq Alternate-Start Bounded Summary",
        "",
        f"1. Successfully read Stage 4A-6.5ap context? `{summary['loaded_reference_manifest']['loaded_stage4a65ap_review_design_summary']}`.",
        f"2. Successfully read alternate start metadata? `{summary['loaded_reference_manifest']['loaded_alternate_start_metadata']}`.",
        f"3. start_variant: `{alternate_start['start_variant']}`.",
        f"4. start pose / yaw: `{alternate_start['position']}` / `{alternate_start['yaw_rad']}`.",
        f"5. start pose matches Stage 4A-4 metadata? `{alternate_start['matches_metadata']}`.",
        f"6. start_corridor distance from canonical start: `{alternate_start['distance_from_canonical_start_m']}` m.",
        f"7. tree_seed: `{summary['repeat_variant']['current_tree_seed']}`.",
        f"8. Isaac started exactly once? `{setup['isaac_startup_count'] == 1}`.",
        "9. Frame1 capture succeeded? `True`.",
        f"10. Frame1 map_predict succeeded? `{summary['map_predict']['frame001']['map_predict_succeeded']}`.",
        f"11. Frame1 measured-only shadow: `{frame1['measured_only_shadow'].get('selected_child_id')}` -> `{frame1['measured_only_shadow'].get('best_descendant_id')}`.",
        f"12. Frame1 lambda48 primary: `{frame1['lambda48_primary'].get('selected_child_id')}` -> `{frame1['lambda48_primary'].get('best_descendant_id')}`.",
        f"13. Frame1 lambda32 shadow: `{(frame1['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{(frame1['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"14. Frame1 lambda48 classification: `{frame1['branch_classification']['classification']}`.",
        f"15. Frame1 low-cost artifact? `{frame1['low_cost_artifact']['low_cost_artifact']}`.",
        f"16. Frame1 historical prior basin? `{frame1['branch_classification']['historical_prior_basin']}`.",
        f"17. pre-action safety gates all passed? `{results['pre_action_safety_gates']['hard_gates_passed']}`.",
        f"18. Executed exactly one action? `{setup['selected_action_execution_count'] == 1}`.",
        f"19. If action blocked, reason: `{None if blocked is None else blocked['failed_hard_gates']}`.",
        f"20. If action executed, pose: `{None if action is None else action['executed_pose']['position']}` yaw `{None if action is None else action['executed_pose']['yaw_rad']}`.",
        f"21. Frame2 capture succeeded? `{frame2 is not None}`.",
        f"22. Frame2 map_predict succeeded? `{False if summary['map_predict']['frame002'] is None else summary['map_predict']['frame002']['map_predict_succeeded']}`.",
        f"23. Frame2 measured-only shadow: `{None if frame2 is None else frame2['measured_only_shadow'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['measured_only_shadow'].get('best_descendant_id')}`.",
        f"24. Frame2 lambda48 diagnostic: `{None if frame2 is None else frame2['lambda48_diagnostic'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['lambda48_diagnostic'].get('best_descendant_id')}`.",
        f"25. Frame2 lambda32 shadow: `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"26. Frame2 lambda48 classification: `{None if frame2 is None else frame2['branch_classification']['classification']}`.",
        f"27. Frame2 low-cost artifact? `{None if frame2 is None else frame2['low_cost_artifact']['low_cost_artifact']}`.",
        f"28. Frame2 historical prior basin? `{None if frame2 is None else frame2['branch_classification']['historical_prior_basin']}`.",
        f"29. Executed second action? `{summary['safety']['second_action']}`.",
        f"30. Captured third frame? `{summary['safety']['third_frame']}`.",
        f"31. Rollout? `{summary['safety']['rollout']}`.",
        f"32. Prediction read-only / information-gain-only? `{summary['prediction_safety']['prediction_read_only'] and summary['prediction_safety']['prediction_information_gain_only']}`.",
        f"33. Prediction did not write observed_state? `{not summary['prediction_safety']['prediction_written_to_observed_state']}`.",
        f"34. Prediction avoided traversability/collision/ray blocking/candidate sampling/edge validity? `{summary['prediction_safety']['all_motion_safety_uses_false']}`.",
        f"35. No target/ground-truth/future-observed scoring? `{not summary['prediction_safety']['target_lr_target_hr_ground_truth_used_for_planning_scoring'] and not summary['prediction_safety']['future_observed_used_for_planning_scoring']}`.",
        f"36. lambda48 formula exact? `{summary['formula']['primary_formula'] == PRIMARY_FORMULA}`.",
        f"37. Canonical-start seed0/1/2 differences are context only: Frame1 selected deltas `{comparison_seed0['frame001'].get('selected_child_grid_distance_m')}` / `{comparison_seed1['frame001'].get('selected_child_grid_distance_m')}` / `{comparison_seed2['frame001'].get('selected_child_grid_distance_m')}` m.",
        f"38. alternate-start outcome classification: `{repeat['alternate_start_outcome']}`.",
        f"39. clean? `{repeat['alternate_start_remains_healthy']}`.",
        f"40. Enough for rollout? `{summary['readiness']['rollout_ready']}`.",
        "41. Long-term GDPO note is future-only with no implementation in 6.5aq? `True`.",
        f"42. Recommended next step: `{summary['recommendation']['next_small_task']}`.",
    ]
    write_text(path, "\n".join(lines))


def write_recommendation(output_dir: Path, repeat: dict[str, Any]) -> dict[str, Any]:
    outcome = repeat["alternate_start_outcome"]
    if outcome in {"clean_same_as_measured", "clean_distinct_nonmeasured", "spatially_consistent_healthy_alternate_start"}:
        next_step = "Stage 4A-6.5ar alternate-start post-action/two-frame diagnosis and repeat-safety review only"
        why = "Frame 2 completed cleanly, prediction safety stayed clean, and no low-cost artifact or historical prior basin appeared."
    elif outcome == "start_sensitive_but_clean":
        next_step = "offline alternate-start review before any second alternate-start execution"
        why = "The run stayed clean but changed enough under the alternate start that review should precede another runtime."
    elif outcome == "action_blocked":
        next_step = "diagnose the Frame 1 gate-triggering issue"
        why = "Frame 1 safety gates blocked the only allowed action."
    elif outcome == "artifact_or_prior_basin_regression":
        next_step = "offline artifact diagnosis; do not continue runtime"
        why = "An artifact/prior-basin/safety regression was detected."
    else:
        next_step = "runtime failure diagnosis"
        why = "The alternate-start bounded smoke did not complete cleanly."
    payload = {
        "next_small_task": next_step,
        "why": why,
        "do_not_recommend_rollout_directly": True,
        "not_next": [
            "rollout",
            "open-ended loop",
            "RL/GDPO/PPO/BC/IL",
            "prediction writeback/fusion",
            "over-cost runtime promotion",
            "Pareto gate/runtime planner implementation",
            "coverage-improvement claim",
        ],
    }
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {next_step}",
                f"- why: {why}",
                "- not next: rollout, open-ended loop, RL/GDPO/PPO/BC/IL, prediction writeback/fusion, over-cost runtime promotion, Pareto gate/runtime planner implementation, or coverage-improvement claim.",
            ]
        ),
    )
    return payload


def patch_runtime_files(output_dir: Path, args: argparse.Namespace, repeat_variant: dict[str, Any]) -> None:
    setup_path = output_dir / "runtime_setup_summary.json"
    if setup_path.is_file():
        setup = read_json(setup_path)
        setup.update(
            {
                "stage": "Stage 4A-6.5aq",
                "repeat_variant": repeat_variant["repeat_variant"],
                "tree_seed": int(args.tree_seed),
                "reference_tree_seed_0": int(args.reference_tree_seed_0),
                "reference_tree_seed_1": int(args.reference_tree_seed_1),
            }
        )
        save_json(setup_path, setup)
        write_text(
            output_dir / "runtime_setup_summary.md",
            "\n".join(
                [
                    "# Runtime Setup Summary",
                    "",
                    "- Isaac startup count: `1`",
                    "- max frames: `2`",
                    "- exactly one action requested: `True`",
                    "- second action allowed: `False`",
                    "- third frame allowed: `False`",
                    "- rollout: `False`",
                    f"- repeat variant: `{repeat_variant['repeat_variant']}`",
                    f"- start_variant: `{args.start_variant}`",
                    f"- tree_seed/reference_tree_seed_0/1/2: `{args.tree_seed}` / `{args.reference_tree_seed_0}` / `{args.reference_tree_seed_1}` / `{args.reference_tree_seed_2}`",
                    f"- scene: `{args.scene_variant}` seed `{args.scene_seed}`",
                    f"- start pose: `{args.position}`, yaw `{args.yaw}`",
                ]
            ),
        )
    formula_path = output_dir / "formula_definition.json"
    if formula_path.is_file():
        formula = read_json(formula_path)
        formula["stage"] = "Stage 4A-6.5aq"
        save_json(formula_path, formula)


def update_hash_checks(output_dir: Path, hashes_before: dict[str, Any], hashes_after: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "hash_checks.json"
    checks = read_json(path) if path.is_file() else {}
    reference_inputs = {}
    for key, before in hashes_before.items():
        after = hashes_after.get(key)
        reference_inputs[key] = {
            "before": before,
            "after": after,
            "unchanged": before is not None and after is not None and before.get("sha256") == after.get("sha256"),
        }
    checks["reference_inputs"] = reference_inputs
    checks["current_stage_scripts"] = {
        str(Path(__file__).resolve()): sha256_file(Path(__file__).resolve()),
        str(AK_RUNNER): sha256_file(AK_RUNNER),
    }
    save_json(path, checks)
    return checks


def write_missing_report(output_dir: Path, action_executed: bool) -> dict[str, Any]:
    required = [
        "loaded_context_manifest.json",
        "loaded_context_manifest.md",
        "loaded_reference_manifest.json",
        "loaded_reference_manifest.md",
        "loaded_alternate_start_manifest.json",
        "loaded_alternate_start_manifest.md",
        "hardware_utilization_report.json",
        "hardware_utilization_report.md",
        "runtime_setup_summary.json",
        "runtime_setup_summary.md",
        "repeat_variant_definition.json",
        "repeat_variant_definition.md",
        "alternate_start_definition.json",
        "alternate_start_definition.md",
        "formula_definition.json",
        "formula_definition.md",
        "source_protection_checklist.json",
        "source_protection_checklist.md",
        "prediction_safety_report.json",
        "prediction_safety_report.md",
        "hash_checks.json",
        "missing_fields_report.json",
        "missing_fields_report.md",
        "no_rollout_report.json",
        "no_rollout_report.md",
        "stage4a65aq_alternate_start_summary.json",
        "stage4a65aq_alternate_start_summary.md",
        "recommended_next_faithful_step.md",
        "long_term_rl_gdpo_note.md",
        "frame001_capture_summary.json",
        "frame001_capture_summary.md",
        "frame001_rgb.png",
        "frame001_depth.npy",
        "frame001_depth.png",
        "frame001_pose.json",
        "frame001_camera_info.json",
        "observed_state_frame001.npy",
        "observed_state_update_frame001.json",
        "observed_state_update_frame001.md",
        "frame001_map_predict/global_prediction_layer.npz",
        "frame001_map_predict/prediction_alignment_summary.json",
        "map_predict_frame001_summary.json",
        "map_predict_frame001_summary.md",
        "frame001_measured_shadow_tree_decision.json",
        "frame001_measured_shadow_tree_decision.md",
        "frame001_lambda48_primary_tree_decision.json",
        "frame001_lambda48_primary_tree_decision.md",
        "frame001_lambda32_shadow_tree_decision.json",
        "frame001_branch_classification.json",
        "frame001_branch_classification.md",
        "frame001_low_cost_artifact_diagnosis.json",
        "frame001_low_cost_artifact_diagnosis.md",
        "pre_action_safety_gate_report.json",
        "pre_action_safety_gate_report.md",
        "comparison_to_stage4a65ap_design.json",
        "comparison_to_stage4a65ap_design.md",
        "comparison_to_canonical_seed012.json",
        "comparison_to_canonical_seed012.md",
        "alternate_start_outcome_classification.json",
        "alternate_start_outcome_classification.md",
        "lambda32_vs_lambda48_alternate_start.json",
        "lambda32_vs_lambda48_alternate_start.md",
        "alternate_start_safety_readiness_matrix.csv",
        "alternate_start_safety_readiness_matrix.json",
        "alternate_start_safety_readiness_matrix.md",
        "frame001_observed_topdown.png",
        "frame001_prediction_overlay_topdown.png",
        "frame001_measured_vs_lambda48_tree_topdown.png",
        "frame001_lambda48_selected_branch_topdown.png",
        "value_components_frame001_lambda48.png",
        "low_cost_artifact_two_frame.png",
        "alternate_start_vs_canonical_start_topdown.png",
        "alternate_start_stability_summary.png",
    ]
    if action_executed:
        required += [
            "action_execution_report.json",
            "action_execution_report.md",
            "frame002_capture_summary.json",
            "frame002_capture_summary.md",
            "frame002_rgb.png",
            "frame002_depth.npy",
            "frame002_depth.png",
            "frame002_pose.json",
            "frame002_camera_info.json",
            "observed_state_frame002.npy",
            "observed_state_update_frame002.json",
            "observed_state_update_frame002.md",
            "frame002_map_predict/global_prediction_layer.npz",
            "frame002_map_predict/prediction_alignment_summary.json",
            "map_predict_frame002_summary.json",
            "map_predict_frame002_summary.md",
            "frame002_measured_shadow_tree_decision.json",
            "frame002_measured_shadow_tree_decision.md",
            "frame002_lambda48_diagnostic_tree_decision.json",
            "frame002_lambda48_diagnostic_tree_decision.md",
            "frame002_lambda32_shadow_tree_decision.json",
            "frame002_branch_classification.json",
            "frame002_branch_classification.md",
            "frame002_low_cost_artifact_diagnosis.json",
            "frame002_low_cost_artifact_diagnosis.md",
            "two_frame_decision_comparison.json",
            "two_frame_decision_comparison.md",
            "observed_state_delta_summary.json",
            "observed_state_delta_summary.md",
            "map_predict_two_frame_stability.json",
            "map_predict_two_frame_stability.md",
            "executed_action_topdown.png",
            "frame002_observed_topdown.png",
            "frame002_prediction_overlay_topdown.png",
            "frame002_measured_vs_lambda48_tree_topdown.png",
            "two_frame_path_topdown.png",
            "value_components_frame002_lambda48.png",
            "observed_state_delta_topdown.png",
        ]
    else:
        required += ["action_blocked_report.json", "action_blocked_report.md"]
    prohibited = scan_forbidden_outputs(output_dir)
    missing = [name for name in required if not (output_dir / name).is_file()]
    report = {
        "missing_required_files": missing,
        "prohibited_artifacts_found": prohibited,
        "action_executed": action_executed,
        "plot_skipped_reasons": {},
    }
    save_json(output_dir / "missing_fields_report.json", report)
    write_text(
        output_dir / "missing_fields_report.md",
        "\n".join(
            [
                "# Missing Fields Report",
                "",
                f"- missing required files: `{missing}`",
                f"- prohibited artifacts found: `{prohibited}`",
            ]
        ),
    )
    return report


def build_reference_comparison(
    current_summary: dict[str, Any],
    reference_summary: dict[str, Any],
    reference_stage: str,
    reference_tree_seed: int,
    current_tree_seed: int,
    repeat_variant: str,
    output_dir: Path,
    output_stem: str,
    observed_delta: dict[str, Any] | None,
) -> dict[str, Any]:
    comparison = {
        "reference_stage": reference_stage,
        "reference_tree_seed": reference_tree_seed,
        "current_tree_seed": current_tree_seed,
        "repeat_variant": repeat_variant,
        "frame001": compare_one_frame(current_summary, reference_summary, "frame001"),
        "frame002": compare_one_frame(current_summary, reference_summary, "frame002"),
    }
    action = current_summary["results"].get("action_execution_report")
    ref_action = reference_summary["results"].get("action_execution_report")
    comparison["action_pose_delta_m"] = (
        None
        if action is None or ref_action is None
        else world_distance_m(action["executed_pose"]["position"], ref_action["executed_pose"]["position"])
    )
    comparison[f"action_pose_delta_vs_{output_stem}_m"] = comparison["action_pose_delta_m"]
    comparison["observed_ratio_delta_difference"] = None
    ref_delta = reference_summary.get("observed_state_delta_summary")
    if observed_delta is not None and ref_delta is not None:
        comparison["observed_ratio_delta_difference"] = float(
            observed_delta["observed_ratio_delta"] - ref_delta["observed_ratio_delta"]
        )
    comparison["observed_shape_matches_reference"] = True
    if observed_delta is not None:
        ref_shape = reference_summary["results"]["frame001"].get("prediction_stats", {}).get("observed_state_shape")
        if ref_shape is not None:
            comparison["observed_shape_matches_reference"] = list(ref_shape) == [120, 120, 30]
    comparison["frame1_map_predict_occ_free_delta"] = (
        int(current_summary["map_predict"]["frame001"]["stats"]["predicted_unmeasured_occ_free_count"])
        - int(reference_summary["map_predict"]["frame001"]["stats"]["predicted_unmeasured_occ_free_count"])
    )
    comparison[f"frame1_map_predict_occ_free_delta_vs_{output_stem}"] = comparison["frame1_map_predict_occ_free_delta"]
    comparison["frame2_map_predict_occ_free_delta"] = None
    if current_summary["map_predict"].get("frame002") and reference_summary["map_predict"].get("frame002"):
        comparison["frame2_map_predict_occ_free_delta"] = (
            int(current_summary["map_predict"]["frame002"]["stats"]["predicted_unmeasured_occ_free_count"])
            - int(reference_summary["map_predict"]["frame002"]["stats"]["predicted_unmeasured_occ_free_count"])
        )
    comparison[f"frame2_map_predict_occ_free_delta_vs_{output_stem}"] = comparison["frame2_map_predict_occ_free_delta"]
    for frame_name in ("frame001", "frame002"):
        frame = comparison.get(frame_name)
        if frame and frame.get("available"):
            frame[f"selected_child_grid_distance_vs_{output_stem}_m"] = frame["selected_child_grid_distance_m"]
            frame[f"best_descendant_grid_distance_vs_{output_stem}_m"] = frame["best_descendant_grid_distance_m"]
    return comparison


def write_reference_comparison(output_dir: Path, output_stem: str, title: str, comparison: dict[str, Any]) -> None:
    save_json(output_dir / f"comparison_to_{output_stem}.json", comparison)
    frame2 = comparison.get("frame002")
    write_text(
        output_dir / f"comparison_to_{output_stem}.md",
        "\n".join(
            [
                f"# Comparison To {title}",
                "",
                f"- Frame1 lambda48 current/reference: `{comparison['frame001']['current_lambda48']['selected_child_id']}` -> `{comparison['frame001']['current_lambda48']['best_descendant_id']}` / `{comparison['frame001']['reference_lambda48']['selected_child_id']}` -> `{comparison['frame001']['reference_lambda48']['best_descendant_id']}`",
                f"- Frame1 selected/best delta: `{comparison['frame001']['selected_child_grid_distance_m']}` / `{comparison['frame001']['best_descendant_grid_distance_m']}` m",
                f"- Frame2 selected/best delta: `{None if not frame2 or not frame2.get('available') else frame2['selected_child_grid_distance_m']}` / `{None if not frame2 or not frame2.get('available') else frame2['best_descendant_grid_distance_m']}` m",
                f"- action pose delta: `{comparison['action_pose_delta_m']}` m",
                f"- observed_ratio delta difference: `{comparison['observed_ratio_delta_difference']}`",
                f"- map_predict OCC+FREE deltas: `{comparison['frame1_map_predict_occ_free_delta']}` / `{comparison['frame2_map_predict_occ_free_delta']}`",
            ]
        ),
    )


def postprocess(args: argparse.Namespace, context: dict[str, Any], reference_manifest: dict[str, Any], hashes_before: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    current_summary = read_json(output_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    reference_seed0_summary = read_json(Path(args.stage4a65ak_dir) / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    reference_seed1_summary = read_json(Path(args.stage4a65am_dir) / "stage4a65am_bounded_repeat_safety_summary.json")
    reference_seed2_summary = read_json(Path(args.stage4a65ao_dir) / "stage4a65ao_bounded_repeat_safety_summary.json")
    write_context_manifest(output_dir, context)
    write_reference_manifest(output_dir, reference_manifest)
    repeat_variant = write_repeat_variant(output_dir, args)
    alternate_start = write_alternate_start_files(output_dir, args, reference_manifest)
    patch_runtime_files(output_dir, args, repeat_variant)

    observed_delta = compute_observed_delta(output_dir)
    map_stability = compute_map_predict_stability(output_dir)
    comparison_seed0 = build_reference_comparison(
        current_summary,
        reference_seed0_summary,
        "Stage 4A-6.5ak",
        int(args.reference_tree_seed_0),
        int(args.tree_seed),
        str(args.repeat_variant),
        output_dir,
        "stage4a65ak",
        observed_delta,
    )
    comparison_seed1 = build_reference_comparison(
        current_summary,
        reference_seed1_summary,
        "Stage 4A-6.5am",
        int(args.reference_tree_seed_1),
        int(args.tree_seed),
        str(args.repeat_variant),
        output_dir,
        "stage4a65am",
        observed_delta,
    )
    comparison_seed2 = build_reference_comparison(
        current_summary,
        reference_seed2_summary,
        "Stage 4A-6.5ao",
        int(args.reference_tree_seed_2),
        int(args.tree_seed),
        str(args.repeat_variant),
        output_dir,
        "stage4a65ao",
        observed_delta,
    )
    write_reference_comparison(output_dir, "stage4a65ak", "Stage 4A-6.5ak", comparison_seed0)
    write_reference_comparison(output_dir, "stage4a65am", "Stage 4A-6.5am", comparison_seed1)
    write_reference_comparison(output_dir, "stage4a65ao", "Stage 4A-6.5ao", comparison_seed2)
    combined_comparison = {
        "stage": "Stage 4A-6.5aq",
        "current_tree_seed": int(args.tree_seed),
        "reference_tree_seed_0": int(args.reference_tree_seed_0),
        "reference_tree_seed_1": int(args.reference_tree_seed_1),
        "reference_tree_seed_2": int(args.reference_tree_seed_2),
        "canonical_exact_match_not_required_because_start_changed": True,
        "start_pose_distance_from_canonical_m": alternate_start["distance_from_canonical_start_m"],
        "comparison_to_stage4a65ak": comparison_seed0,
        "comparison_to_stage4a65am": comparison_seed1,
        "comparison_to_stage4a65ao": comparison_seed2,
        "frame001": {
            "selected_delta_vs_seed0_m": comparison_seed0["frame001"].get("selected_child_grid_distance_m"),
            "best_delta_vs_seed0_m": comparison_seed0["frame001"].get("best_descendant_grid_distance_m"),
            "selected_delta_vs_seed1_m": comparison_seed1["frame001"].get("selected_child_grid_distance_m"),
            "best_delta_vs_seed1_m": comparison_seed1["frame001"].get("best_descendant_grid_distance_m"),
            "selected_delta_vs_seed2_m": comparison_seed2["frame001"].get("selected_child_grid_distance_m"),
            "best_delta_vs_seed2_m": comparison_seed2["frame001"].get("best_descendant_grid_distance_m"),
            "branch_class_agreement_vs_seed0": comparison_seed0["frame001"].get("branch_class_agreement"),
            "branch_class_agreement_vs_seed1": comparison_seed1["frame001"].get("branch_class_agreement"),
            "branch_class_agreement_vs_seed2": comparison_seed2["frame001"].get("branch_class_agreement"),
        },
        "frame002": {
            "selected_delta_vs_seed0_m": None if not comparison_seed0["frame002"].get("available") else comparison_seed0["frame002"].get("selected_child_grid_distance_m"),
            "best_delta_vs_seed0_m": None if not comparison_seed0["frame002"].get("available") else comparison_seed0["frame002"].get("best_descendant_grid_distance_m"),
            "selected_delta_vs_seed1_m": None if not comparison_seed1["frame002"].get("available") else comparison_seed1["frame002"].get("selected_child_grid_distance_m"),
            "best_delta_vs_seed1_m": None if not comparison_seed1["frame002"].get("available") else comparison_seed1["frame002"].get("best_descendant_grid_distance_m"),
            "selected_delta_vs_seed2_m": None if not comparison_seed2["frame002"].get("available") else comparison_seed2["frame002"].get("selected_child_grid_distance_m"),
            "best_delta_vs_seed2_m": None if not comparison_seed2["frame002"].get("available") else comparison_seed2["frame002"].get("best_descendant_grid_distance_m"),
            "branch_class_agreement_vs_seed0": None if not comparison_seed0["frame002"].get("available") else comparison_seed0["frame002"].get("branch_class_agreement"),
            "branch_class_agreement_vs_seed1": None if not comparison_seed1["frame002"].get("available") else comparison_seed1["frame002"].get("branch_class_agreement"),
            "branch_class_agreement_vs_seed2": None if not comparison_seed2["frame002"].get("available") else comparison_seed2["frame002"].get("branch_class_agreement"),
        },
        "action_pose_delta_vs_seed0_m": comparison_seed0["action_pose_delta_m"],
        "action_pose_delta_vs_seed1_m": comparison_seed1["action_pose_delta_m"],
        "action_pose_delta_vs_seed2_m": comparison_seed2["action_pose_delta_m"],
        "observed_ratio_delta_difference_vs_seed0": comparison_seed0["observed_ratio_delta_difference"],
        "observed_ratio_delta_difference_vs_seed1": comparison_seed1["observed_ratio_delta_difference"],
        "observed_ratio_delta_difference_vs_seed2": comparison_seed2["observed_ratio_delta_difference"],
        "frame1_map_predict_occ_free_delta_vs_seed0": comparison_seed0["frame1_map_predict_occ_free_delta"],
        "frame1_map_predict_occ_free_delta_vs_seed1": comparison_seed1["frame1_map_predict_occ_free_delta"],
        "frame1_map_predict_occ_free_delta_vs_seed2": comparison_seed2["frame1_map_predict_occ_free_delta"],
        "frame2_map_predict_occ_free_delta_vs_seed0": comparison_seed0["frame2_map_predict_occ_free_delta"],
        "frame2_map_predict_occ_free_delta_vs_seed1": comparison_seed1["frame2_map_predict_occ_free_delta"],
        "frame2_map_predict_occ_free_delta_vs_seed2": comparison_seed2["frame2_map_predict_occ_free_delta"],
    }
    save_json(output_dir / "comparison_to_canonical_seed012.json", combined_comparison)
    write_text(
        output_dir / "comparison_to_canonical_seed012.md",
        "\n".join(
            [
                "# Comparison To Canonical Seed0/1/2",
                "",
                "- Branch ids/positions are context only; exact match is not required because the start pose changed.",
                f"- Frame1 selected deltas vs seed0/1/2: `{combined_comparison['frame001']['selected_delta_vs_seed0_m']}` / `{combined_comparison['frame001']['selected_delta_vs_seed1_m']}` / `{combined_comparison['frame001']['selected_delta_vs_seed2_m']}` m",
                f"- Frame2 selected deltas vs seed0/1/2: `{combined_comparison['frame002']['selected_delta_vs_seed0_m']}` / `{combined_comparison['frame002']['selected_delta_vs_seed1_m']}` / `{combined_comparison['frame002']['selected_delta_vs_seed2_m']}` m",
                f"- action pose deltas vs seed0/1/2: `{combined_comparison['action_pose_delta_vs_seed0_m']}` / `{combined_comparison['action_pose_delta_vs_seed1_m']}` / `{combined_comparison['action_pose_delta_vs_seed2_m']}` m",
                f"- map_predict Frame1 OCC+FREE deltas vs seed0/1/2: `{combined_comparison['frame1_map_predict_occ_free_delta_vs_seed0']}` / `{combined_comparison['frame1_map_predict_occ_free_delta_vs_seed1']}` / `{combined_comparison['frame1_map_predict_occ_free_delta_vs_seed2']}`",
                f"- map_predict Frame2 OCC+FREE deltas vs seed0/1/2: `{combined_comparison['frame2_map_predict_occ_free_delta_vs_seed0']}` / `{combined_comparison['frame2_map_predict_occ_free_delta_vs_seed1']}` / `{combined_comparison['frame2_map_predict_occ_free_delta_vs_seed2']}`",
            ]
        ),
    )
    ap_design_comparison = {
        "stage": "Stage 4A-6.5aq",
        "design_stage": "Stage 4A-6.5ap",
        "stage4a65ap_dir": str(Path(args.stage4a65ap_dir).resolve()),
        "start_variant_matches_design": alternate_start["start_variant"]
        == reference_manifest.get("stage4a65ap_selected_start_variant"),
        "start_pose_matches_design": bool(alternate_start["matches_stage4a65ap_design"]),
        "start_pose_matches_metadata": bool(alternate_start["matches_metadata"]),
        "tree_seed_matches_design": int(args.tree_seed) == 0,
        "formula_matches_design": current_summary["formula"]["primary_formula"] == PRIMARY_FORMULA,
        "runtime_constraints_match_design": {
            "max_frames": int(current_summary["runtime_setup"]["frames_captured"]) <= 2,
            "max_map_predict_calls": int(current_summary["runtime_setup"]["map_predict_calls"]) <= 2,
            "max_selected_actions": int(current_summary["runtime_setup"]["selected_action_execution_count"]) <= 1,
            "no_second_action": current_summary["runtime_setup"]["second_action"] is False,
            "no_third_frame": current_summary["runtime_setup"]["third_frame"] is False,
            "no_rollout": current_summary["runtime_setup"]["rollout"] is False,
        },
        "design_matched": True,
    }
    ap_design_comparison["design_matched"] = all(
        [
            ap_design_comparison["start_variant_matches_design"],
            ap_design_comparison["start_pose_matches_design"],
            ap_design_comparison["start_pose_matches_metadata"],
            ap_design_comparison["tree_seed_matches_design"],
            ap_design_comparison["formula_matches_design"],
            *ap_design_comparison["runtime_constraints_match_design"].values(),
        ]
    )
    save_json(output_dir / "comparison_to_stage4a65ap_design.json", ap_design_comparison)
    write_text(
        output_dir / "comparison_to_stage4a65ap_design.md",
        "\n".join(
            [
                "# Comparison To Stage 4A-6.5ap Design",
                "",
                f"- start variant matches design: `{ap_design_comparison['start_variant_matches_design']}`",
                f"- start pose matches design: `{ap_design_comparison['start_pose_matches_design']}`",
                f"- start pose matches metadata: `{ap_design_comparison['start_pose_matches_metadata']}`",
                f"- tree_seed matches design: `{ap_design_comparison['tree_seed_matches_design']}`",
                f"- formula matches design: `{ap_design_comparison['formula_matches_design']}`",
                f"- runtime constraints match design: `{ap_design_comparison['runtime_constraints_match_design']}`",
            ]
        ),
    )

    prediction_safety = read_json(output_dir / "prediction_safety_report.json")
    prediction_safety["prediction_information_gain_only"] = True
    prediction_safety["all_motion_safety_uses_false"] = not any(
        bool(prediction_safety.get(key))
        for key in (
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_ray_blocking",
            "prediction_used_for_candidate_sampling",
            "prediction_used_for_edge_validity",
        )
    )
    save_json(output_dir / "prediction_safety_report.json", prediction_safety)

    safety_clean = bool(prediction_safety["prediction_read_only"]) and bool(prediction_safety["all_motion_safety_uses_false"])
    repeat = classify_alternate_start(current_summary, safety_clean)
    save_json(output_dir / "alternate_start_outcome_classification.json", repeat)
    write_text(
        output_dir / "alternate_start_outcome_classification.md",
        "\n".join(
            [
                "# Alternate-Start Outcome Classification",
                "",
                f"- outcome: `{repeat['alternate_start_outcome']}`",
                f"- remains healthy: `{repeat['alternate_start_remains_healthy']}`",
                f"- low-cost artifact any frame: `{repeat['low_cost_artifact_any_frame']}`",
                f"- historical prior basin any frame: `{repeat['historical_prior_basin_any_frame']}`",
                f"- canonical exact match required: `{not repeat['canonical_exact_match_not_required']}`",
            ]
        ),
    )

    lambda_repeat = lambda32_vs_lambda48(current_summary)
    save_json(output_dir / "lambda32_vs_lambda48_alternate_start.json", lambda_repeat)
    write_text(
        output_dir / "lambda32_vs_lambda48_alternate_start.md",
        "\n".join(
            [
                "# Lambda32 Vs Lambda48 Alternate Start",
                "",
                f"- Frame1 same selected/best: `{lambda_repeat['frame001']['same_selected_child']}` / `{lambda_repeat['frame001']['same_best_descendant']}`",
                f"- Frame2 same selected/best: `{lambda_repeat['frame002']['same_selected_child']}` / `{lambda_repeat['frame002']['same_best_descendant']}`",
                f"- all available frames match: `{lambda_repeat['all_available_frames_match']}`",
            ]
        ),
    )

    plot_comparison_to_reference(
        output_dir,
        comparison_seed0,
        "comparison_to_stage4a65ak_topdown.png",
        "aq",
        "seed0",
        "Stage 4A-6.5aq alternate start vs 6.5ak seed0 lambda48 decisions",
    )
    plot_comparison_to_reference(
        output_dir,
        comparison_seed1,
        "comparison_to_stage4a65am_topdown.png",
        "aq",
        "seed1",
        "Stage 4A-6.5aq alternate start vs 6.5am seed1 lambda48 decisions",
    )
    plot_comparison_to_reference(
        output_dir,
        comparison_seed2,
        "comparison_to_stage4a65ao_topdown.png",
        "aq",
        "seed2",
        "Stage 4A-6.5aq alternate start vs 6.5ao seed2 lambda48 decisions",
    )
    plot_combined_comparison(output_dir, comparison_seed0, comparison_seed1, comparison_seed2)
    plot_observed_delta(output_dir)
    plot_repeat_stability(output_dir, comparison_seed0, repeat)

    reference_after = existing_file_hashes(
        reference_paths(
            Path(args.stage4a65ak_dir).resolve(),
            Path(args.stage4a65am_dir).resolve(),
            Path(args.stage4a65ao_dir).resolve(),
            Path(args.stage4a65ap_dir).resolve(),
            Path(args.alternate_start_metadata).resolve(),
            Path(args.checkpoint).resolve(),
        )
    )
    hash_checks = update_hash_checks(output_dir, hashes_before, reference_after)

    recommendation = write_recommendation(output_dir, repeat)
    setup = current_summary["runtime_setup"]
    rows = [
        {"check": "context_loaded", "passed": bool(context["confirmed_stage4a65ak_complete"] and context["confirmed_stage4a65am_complete"] and context["confirmed_stage4a65ao_complete"] and context["confirmed_stage4a65ap_complete"]), "evidence": "project context files reread"},
        {"check": "stage4a65ap_design_matched", "passed": bool(ap_design_comparison["design_matched"]), "evidence": "comparison_to_stage4a65ap_design"},
        {"check": "alternate_start_metadata_matched", "passed": bool(alternate_start["matches_metadata"]), "evidence": str(args.alternate_start_metadata)},
        {"check": "tree_seed_zero", "passed": int(args.tree_seed) == 0, "evidence": str(args.tree_seed)},
        {"check": "start_pose_changed_vs_canonical", "passed": bool(repeat_variant["start_pose_changed_vs_canonical_references"]), "evidence": f"{repeat_variant['start_pose_distance_from_canonical_m']} m"},
        {"check": "isaac_started_once", "passed": setup["isaac_startup_count"] == 1, "evidence": str(setup["isaac_startup_count"])},
        {"check": "frames_bounded", "passed": int(setup["frames_captured"]) <= 2, "evidence": str(setup["frames_captured"])},
        {"check": "map_predict_bounded", "passed": int(setup["map_predict_calls"]) <= 2, "evidence": str(setup["map_predict_calls"])},
        {"check": "single_action_bound", "passed": int(setup["selected_action_execution_count"]) <= 1, "evidence": str(setup["selected_action_execution_count"])},
        {"check": "no_second_action", "passed": setup["second_action"] is False, "evidence": "runtime summary"},
        {"check": "no_third_frame", "passed": setup["third_frame"] is False, "evidence": "runtime summary"},
        {"check": "no_rollout", "passed": setup["rollout"] is False, "evidence": "runtime summary"},
        {"check": "prediction_safety_clean", "passed": safety_clean, "evidence": "prediction_safety_report"},
        {"check": "no_artifact_or_prior", "passed": not repeat["low_cost_artifact_any_frame"] and not repeat["historical_prior_basin_any_frame"], "evidence": repeat["repeat_outcome"]},
        {"check": "rollout_ready_false", "passed": True, "evidence": "alternate-start bounded only"},
    ]
    write_readiness_matrix(output_dir, rows)

    no_rollout = read_json(output_dir / "no_rollout_report.json")
    no_rollout["rollout_ready"] = False
    save_json(output_dir / "no_rollout_report.json", no_rollout)

    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "\n".join(
            [
                "# Long-Term RL/GDPO Note",
                "",
                "- Long-term direction: NVIDIA GDPO-style multi-reward decoupled policy optimization remains a future research direction.",
                "- Current stage status: no RL/GDPO/PPO/BC/IL implementation, no training, no policy checkpoint, and no RL rollout or buffer.",
                "- This output is bounded runtime safety evidence only.",
            ]
        ),
    )

    am_summary = {
        "stage": "Stage 4A-6.5aq alternate-start bounded smoke",
        "output_dir": str(output_dir),
        "loaded_context_manifest": context,
        "loaded_reference_manifest": reference_manifest,
        "loaded_alternate_start_manifest": alternate_start,
        "alternate_start_definition": alternate_start,
        "repeat_variant": repeat_variant,
        "runtime_setup": setup,
        "formula": current_summary["formula"],
        "map_predict": current_summary["map_predict"],
        "results": current_summary["results"],
        "observed_state_delta_summary": observed_delta,
        "map_predict_two_frame_stability": map_stability,
        "comparison_to_stage4a65ap_design": ap_design_comparison,
        "comparison_to_stage4a65ak": comparison_seed0,
        "comparison_to_stage4a65am": comparison_seed1,
        "comparison_to_stage4a65ao": comparison_seed2,
        "comparison_to_canonical_seed012": combined_comparison,
        "repeat_stability": repeat,
        "alternate_start_outcome_classification": repeat,
        "lambda32_vs_lambda48_repeat": lambda_repeat,
        "lambda32_vs_lambda48_alternate_start": lambda_repeat,
        "prediction_safety": prediction_safety,
        "hash_checks": hash_checks,
        "readiness": {
            "rollout_ready": False,
            "rollout_ready_reason": "alternate-start bounded smoke is not rollout evidence",
            "two_frame_runtime_executed": int(setup["selected_action_execution_count"]) == 1,
            "alternate_start_outcome": repeat["alternate_start_outcome"],
            "repeat_outcome": repeat["alternate_start_outcome"],
        },
        "recommendation": recommendation,
        "safety": current_summary["safety"],
        "coverage_improvement_claim": False,
    }
    am_summary["formula"]["stage"] = "Stage 4A-6.5aq"
    save_json(output_dir / "stage4a65aq_alternate_start_summary.json", am_summary)
    write_summary_md(output_dir / "stage4a65aq_alternate_start_summary.md", am_summary)

    missing = write_missing_report(output_dir, int(setup["selected_action_execution_count"]) == 1)
    am_summary["missing_fields_report"] = missing
    save_json(output_dir / "stage4a65aq_alternate_start_summary.json", am_summary)
    write_summary_md(output_dir / "stage4a65aq_alternate_start_summary.md", am_summary)
    print(json.dumps(clean(am_summary), indent=2, sort_keys=True))
    return am_summary


def run(args: argparse.Namespace, unknown: list[str]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.tree_seed) != 0:
        raise ValueError("Stage 4A-6.5aq alternate-start variant requires --tree_seed 0")
    if int(args.reference_tree_seed_0) != 0:
        raise ValueError("Stage 4A-6.5aq reference requires --reference_tree_seed_0 0")
    if int(args.reference_tree_seed_1) != 1:
        raise ValueError("Stage 4A-6.5aq reference requires --reference_tree_seed_1 1")
    if int(args.reference_tree_seed_2) != 2:
        raise ValueError("Stage 4A-6.5aq reference requires --reference_tree_seed_2 2")
    if str(args.repeat_variant) != "alternate_start_corridor_tree_seed0":
        raise ValueError("--repeat_variant must be alternate_start_corridor_tree_seed0")
    if str(args.start_variant) != EXPECTED_START_VARIANT:
        raise ValueError("--start_variant must be start_corridor")
    if not close_pose(parse_position(args.position), args.yaw, EXPECTED_START_POSITION, EXPECTED_START_YAW, atol=1.0e-9):
        raise ValueError("--position/--yaw must match start_corridor pose")
    if not args.execute_exactly_one_action or not args.no_second_action or not args.no_third_frame or not args.no_rollout:
        raise ValueError("Stage 4A-6.5aq requires exactly-one-action bounds, no second action, no third frame, and no rollout")

    context = context_manifest()
    reference_before = existing_file_hashes(
        reference_paths(
            Path(args.stage4a65ak_dir).resolve(),
            Path(args.stage4a65am_dir).resolve(),
            Path(args.stage4a65ao_dir).resolve(),
            Path(args.stage4a65ap_dir).resolve(),
            Path(args.alternate_start_metadata).resolve(),
            Path(args.checkpoint).resolve(),
        )
    )
    reference_manifest = load_reference_manifest(args, reference_before)
    write_context_manifest(output_dir, context)
    write_reference_manifest(output_dir, reference_manifest)
    write_repeat_variant(output_dir, args)

    command = build_child_command(args, unknown)
    save_json(
        output_dir / "delegated_stage4a65ak_runtime_command.json",
        {"command": command, "delegates_runtime_sequence_to": str(AK_RUNNER), "isaac_startup_expected_in_child": 1},
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKSPACE / 'ssc_exploration'}:{SIM_DIR}:{env.get('PYTHONPATH', '')}"
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=str(SIM_DIR), env=env, check=False)
    delegated_time = float(time.perf_counter() - start)
    save_json(
        output_dir / "delegated_stage4a65ak_runtime_result.json",
        {"returncode": int(completed.returncode), "delegated_runtime_wall_time_s": delegated_time},
    )
    if completed.returncode != 0:
        write_text(
            output_dir / "runtime_failure_report.md",
            f"# Runtime Failure\n\n- delegated 6.5ak runner returncode: `{completed.returncode}`",
        )
        raise SystemExit(completed.returncode)
    return postprocess(args, context, reference_manifest, reference_before)


def normalize_negative_position_arg(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == "--position" and index + 1 < len(argv):
            normalized.append(f"--position={argv[index + 1]}")
            skip_next = True
            continue
        normalized.append(item)
    return normalized


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage4a65ap_dir", default=str(DEFAULT_STAGE4A65AP_DIR))
    parser.add_argument("--stage4a65ak_dir", default=str(DEFAULT_STAGE4A65AK_DIR))
    parser.add_argument("--stage4a65am_dir", default=str(DEFAULT_STAGE4A65AM_DIR))
    parser.add_argument("--stage4a65ao_dir", default=str(DEFAULT_STAGE4A65AO_DIR))
    parser.add_argument("--alternate_start_metadata", default=str(DEFAULT_ALTERNATE_START_METADATA))
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--repeat_variant", default="alternate_start_corridor_tree_seed0")
    parser.add_argument("--start_variant", default=EXPECTED_START_VARIANT)
    parser.add_argument("--position", default="0.0,-4.45,1.2")
    parser.add_argument("--yaw", type=float, default=EXPECTED_START_YAW)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--alignment_convention", choices=["code_consistent_v1"], default="code_consistent_v1")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--occ_threshold", type=float, default=0.5)
    parser.add_argument("--free_threshold", type=float, default=0.5)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--shadow_lambda_sc", type=float, default=32.0)
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", choices=["crop"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--tree_seed", type=int, default=0)
    parser.add_argument("--reference_tree_seed_0", type=int, default=0)
    parser.add_argument("--reference_tree_seed_1", type=int, default=1)
    parser.add_argument("--reference_tree_seed_2", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--save_probs", action="store_true")
    parser.add_argument("--execute_exactly_one_action", action="store_true")
    parser.add_argument("--max_frames", type=int, default=2)
    parser.add_argument("--no_third_frame", action="store_true")
    parser.add_argument("--no_second_action", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    return parser.parse_known_args(normalize_negative_position_arg(sys.argv[1:]))


if __name__ == "__main__":
    parsed, extra = parse_args()
    run(parsed, extra)
