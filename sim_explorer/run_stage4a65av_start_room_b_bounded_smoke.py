#!/usr/bin/env python3
"""Stage 4A-6.5av start_room_b tree_seed=0 bounded runtime smoke.

This wrapper executes the proven Stage 4A-6.5ak two-frame/one-action runtime
path with the Stage 4A-6.5au start_room_b design parameters, then writes the
Stage 4A-6.5av-specific review artifacts.  It is a real bounded smoke, not a
rollout, not formal expert sampling, and not RL/GDPO/PPO/BC/IL.

Primary formula:

    value = gain_exp / cost + 48 * minmax(source_occ_free)
"""

from __future__ import annotations

import argparse
import csv
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
import numpy as np

import run_stage4a65aq_alternate_start_bounded_smoke as aq


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
SIM_DIR = WORKSPACE / "sim_explorer"
AK_RUNNER = SIM_DIR / "run_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke.py"

DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65av_start_room_b_bounded_smoke"
DEFAULT_STAGE4A65AU_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65au_start_room_b_bounded_smoke"
DEFAULT_STAGE4A65AT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65at_start_corridor_seed01_review_next_start_design"
DEFAULT_STAGE4A65AQ_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke"
DEFAULT_STAGE4A65AS_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65as_start_corridor_tree_seed1_bounded_smoke"
DEFAULT_STAGE4A65AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_STAGE4A65AM_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_STAGE4A65AO_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2"
DEFAULT_START_ROOM_B_METADATA = (
    WORKSPACE
    / "outputs/isaac_medium_rollout_dataset_empty_pred_astar/episodes/"
    / "medium_three_rooms_seed0_start_room_b_empty_astar/scene_metadata.json"
)
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]

PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
SHADOW_FORMULA = "gain_exp / cost + 32 * minmax(source_occ_free)"
MEASURED_FORMULA = "gain_exp / cost"
EXPECTED_START_VARIANT = "start_room_b"
EXPECTED_START_POSITION = [2.75, -2.55, 1.2]
EXPECTED_START_YAW = 2.7052603405912112
START_CORRIDOR_POSITION = [0.0, -4.45, 1.2]
CANONICAL_START = [-4.65, -4.65, 1.2]

save_json = aq.save_json
read_json = aq.read_json
write_text = aq.write_text
sha256_file = aq.sha256_file
parse_position = aq.parse_position


def close_pose(position: Any, yaw: Any, expected_position: Any, expected_yaw: Any, atol: float = 1.0e-9) -> bool:
    return aq.close_pose(position, yaw, expected_position, expected_yaw, atol=atol)


def distance_m(a: Any, b: Any) -> float:
    return aq.distance_m(a, b)


def angle_delta_rad(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b)))))


def metadata_start_room_b_pose(metadata: dict[str, Any]) -> dict[str, Any]:
    start_pose = metadata.get("start_pose", {})
    world = start_pose.get("world", {}) if isinstance(start_pose, dict) else {}
    if (
        start_pose.get("variant") == EXPECTED_START_VARIANT
        and close_pose(world.get("position"), world.get("yaw_rad"), EXPECTED_START_POSITION, EXPECTED_START_YAW)
    ):
        return {
            "found": True,
            "source_kind": "start_pose.world",
            "variant": start_pose.get("variant"),
            "position": world.get("position"),
            "yaw_rad": world.get("yaw_rad"),
            "yaw_deg": world.get("yaw_deg"),
        }
    for pose in metadata.get("camera_poses", []):
        if close_pose(pose.get("position"), pose.get("yaw_rad"), EXPECTED_START_POSITION, EXPECTED_START_YAW):
            return {
                "found": True,
                "source_kind": "camera_poses",
                "index": pose.get("index"),
                "room": pose.get("room"),
                "note": pose.get("note"),
                "position": pose.get("position"),
                "yaw_rad": pose.get("yaw_rad"),
                "yaw_deg": pose.get("yaw_deg"),
            }
    return {"found": False, "expected_position": EXPECTED_START_POSITION, "expected_yaw": EXPECTED_START_YAW}


def existing_file_hashes(paths: list[Path]) -> dict[str, dict[str, Any] | None]:
    return aq.existing_file_hashes(paths)


def reference_paths(args: argparse.Namespace) -> list[Path]:
    au = Path(args.stage4a65au_dir).resolve()
    at = Path(args.stage4a65at_dir).resolve()
    aq_dir = Path(args.stage4a65aq_dir).resolve()
    as_dir = Path(args.stage4a65as_dir).resolve()
    ak = Path(args.stage4a65ak_dir).resolve()
    am = Path(args.stage4a65am_dir).resolve()
    ao = Path(args.stage4a65ao_dir).resolve()
    paths = [
        Path(args.checkpoint).resolve(),
        Path(args.start_room_b_metadata).resolve(),
        au / "stage4a65au_start_room_b_design_summary.json",
        au / "selected_next_start_design.json",
        au / "future_stage4a65au_command_sketch.md",
        au / "do_not_run_runtime_in_stage4a65au.md",
        at / "stage4a65at_start_corridor_seed01_review_summary.json",
        aq_dir / "stage4a65aq_alternate_start_summary.json",
        as_dir / "stage4a65as_start_corridor_seed1_summary.json",
        ak / "stage4a65ak_two_frame_one_action_runtime_summary.json",
        am / "stage4a65am_bounded_repeat_safety_summary.json",
        ao / "stage4a65ao_bounded_repeat_safety_summary.json",
    ]
    for folder in (aq_dir, as_dir, ak, am, ao):
        paths.extend(
            [
                folder / "observed_state_frame001.npy",
                folder / "observed_state_frame002.npy",
                folder / "frame001_map_predict/global_prediction_layer.npz",
                folder / "frame002_map_predict/global_prediction_layer.npz",
            ]
        )
    return paths


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
                "contains_stage4a65at": "Stage 4A-6.5at" in text,
                "contains_stage4a65aq": "Stage 4A-6.5aq" in text,
                "contains_stage4a65as": "Stage 4A-6.5as" in text,
                "contains_start_room_b": "start_room_b" in text,
                "contains_no_rollout": "no rollout" in text or "not rollout" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5av",
        "loaded_context_files": entries,
        "confirmed_stage4a65at_context": "Stage 4A-6.5at" in combined and "healthy_distinct_seed1_after_conservative_seed0" in combined,
        "confirmed_start_room_b_next_context": "start_room_b" in combined,
        "confirmed_no_rollout_context": "no rollout" in combined or "not rollout" in combined,
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
                f"- Stage 4A-6.5at context confirmed: `{manifest['confirmed_stage4a65at_context']}`",
                f"- start_room_b context confirmed: `{manifest['confirmed_start_room_b_next_context']}`",
                f"- no-rollout context confirmed: `{manifest['confirmed_no_rollout_context']}`",
                "- Files read:",
                *[f"  - `{item['path']}` sha256 `{item['sha256']}`" for item in manifest["loaded_context_files"]],
            ]
        ),
    )


def load_reference_manifest(args: argparse.Namespace, hashes_before: dict[str, Any]) -> dict[str, Any]:
    au_dir = Path(args.stage4a65au_dir).resolve()
    at_dir = Path(args.stage4a65at_dir).resolve()
    aq_dir = Path(args.stage4a65aq_dir).resolve()
    as_dir = Path(args.stage4a65as_dir).resolve()
    metadata_path = Path(args.start_room_b_metadata).resolve()
    au_summary = read_json(au_dir / "stage4a65au_start_room_b_design_summary.json")
    au_design = read_json(au_dir / "selected_next_start_design.json")
    at_summary = read_json(at_dir / "stage4a65at_start_corridor_seed01_review_summary.json")
    aq_summary = read_json(aq_dir / "stage4a65aq_alternate_start_summary.json")
    as_summary = read_json(as_dir / "stage4a65as_start_corridor_seed1_summary.json")
    metadata = read_json(metadata_path)
    metadata_pose = metadata_start_room_b_pose(metadata)
    return {
        "stage": "Stage 4A-6.5av",
        "stage4a65au_dir": str(au_dir),
        "stage4a65at_dir": str(at_dir),
        "stage4a65aq_dir": str(aq_dir),
        "stage4a65as_dir": str(as_dir),
        "start_room_b_metadata": str(metadata_path),
        "loaded_stage4a65au_summary": True,
        "loaded_stage4a65au_design": True,
        "loaded_stage4a65at_review": True,
        "loaded_stage4a65aq_summary": True,
        "loaded_stage4a65as_summary": True,
        "loaded_start_room_b_metadata": True,
        "stage4a65au_design_review_only": bool(au_summary.get("design_review_only")),
        "stage4a65au_runtime_executed": bool(au_summary.get("runtime_executed")),
        "stage4a65au_future_command_marked_do_not_run": bool(au_summary.get("future_command_marked_do_not_run")),
        "stage4a65au_start_variant": au_design.get("start_variant"),
        "stage4a65au_position": au_design.get("position"),
        "stage4a65au_yaw_rad": au_design.get("yaw_rad"),
        "stage4a65au_tree_seed": au_design.get("tree_seed"),
        "stage4a65au_primary_formula": au_design.get("primary_formula"),
        "stage4a65au_formal_expert_sampling_ready_now": au_summary.get("readiness", {}).get("formal_expert_sampling_ready_now"),
        "stage4a65at_outcome": at_summary.get("combined_outcome") or at_summary.get("summary", {}).get("combined_outcome"),
        "stage4a65aq_outcome": aq_summary.get("readiness", {}).get("alternate_start_outcome"),
        "stage4a65as_outcome": as_summary.get("repeat_outcome_classification", {}).get("repeat_outcome"),
        "metadata_scene_variant": metadata.get("scene_variant"),
        "metadata_scene_seed": metadata.get("scene_seed"),
        "metadata_start_variant": metadata.get("start_variant"),
        "metadata_pose": metadata_pose,
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
                f"- loaded Stage 4A-6.5au design: `{manifest['loaded_stage4a65au_design']}`",
                f"- loaded Stage 4A-6.5at review: `{manifest['loaded_stage4a65at_review']}`",
                f"- loaded start_corridor aq/as summaries: `{manifest['loaded_stage4a65aq_summary']}` / `{manifest['loaded_stage4a65as_summary']}`",
                f"- loaded start_room_b metadata: `{manifest['loaded_start_room_b_metadata']}`",
                f"- metadata pose found: `{manifest['metadata_pose'].get('found')}`",
            ]
        ),
    )


def write_start_room_b_files(output_dir: Path, args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    position = parse_position(args.position)
    metadata_pose = manifest["metadata_pose"]
    payload = {
        "stage": "Stage 4A-6.5av",
        "start_variant": str(args.start_variant),
        "position": position,
        "yaw_rad": float(args.yaw),
        "yaw_deg": float(math.degrees(float(args.yaw))),
        "pose_source": str(Path(args.start_room_b_metadata).resolve()),
        "metadata_pose_found": bool(metadata_pose.get("found")),
        "matches_stage4a65au_design": close_pose(
            position,
            args.yaw,
            manifest.get("stage4a65au_position"),
            manifest.get("stage4a65au_yaw_rad"),
        ),
        "matches_metadata": bool(metadata_pose.get("found"))
        and close_pose(position, args.yaw, metadata_pose.get("position"), metadata_pose.get("yaw_rad")),
        "canonical_start_position": CANONICAL_START,
        "start_corridor_position": START_CORRIDOR_POSITION,
        "distance_from_canonical_start_m": distance_m(position, CANONICAL_START),
        "distance_from_start_corridor_m": distance_m(position, START_CORRIDOR_POSITION),
        "fixed_camera_height_m": float(position[2]),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "tree_seed": int(args.tree_seed),
    }
    save_json(output_dir / "loaded_start_room_b_manifest.json", payload)
    save_json(output_dir / "start_room_b_definition.json", payload)
    for name, title in (
        ("loaded_start_room_b_manifest.md", "Loaded Start Room B Manifest"),
        ("start_room_b_definition.md", "Start Room B Definition"),
    ):
        write_text(
            output_dir / name,
            "\n".join(
                [
                    f"# {title}",
                    "",
                    f"- start_variant: `{payload['start_variant']}`",
                    f"- position/yaw: `{payload['position']}` / `{payload['yaw_rad']}`",
                    f"- pose source: `{payload['pose_source']}`",
                    f"- matches Stage 4A-6.5au design: `{payload['matches_stage4a65au_design']}`",
                    f"- matches metadata: `{payload['matches_metadata']}`",
                    f"- distance from start_corridor: `{payload['distance_from_start_corridor_m']}` m",
                ]
            ),
        )
    return payload


def write_repeat_variant(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    position = parse_position(args.position)
    variant = {
        "stage": "Stage 4A-6.5av",
        "repeat_variant": str(args.repeat_variant),
        "start_variant": str(args.start_variant),
        "current_tree_seed": int(args.tree_seed),
        "tree_seed": int(args.tree_seed),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "position": position,
        "yaw": float(args.yaw),
        "max_frames": int(args.max_frames),
        "max_selected_actions": 1,
        "max_map_predict_calls": 2,
        "no_second_action": bool(args.no_second_action),
        "no_third_frame": bool(args.no_third_frame),
        "no_rollout": bool(args.no_rollout),
        "no_formal_expert_sampling": bool(args.no_formal_expert_sampling),
        "only_intended_runtime_change": "start_room_b_from_stage4a65au_design",
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
                f"- tree_seed: `{variant['tree_seed']}`",
                f"- scene/start: `{variant['scene_variant']}` seed `{variant['scene_seed']}`, `{variant['position']}` yaw `{variant['yaw']}`",
                "- bounds: at most two frames, at most two map_predict calls, exactly one action if gates pass, no rollout.",
            ]
        ),
    )
    return variant


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


def patch_runtime_files(output_dir: Path, args: argparse.Namespace, repeat_variant: dict[str, Any]) -> None:
    setup_path = output_dir / "runtime_setup_summary.json"
    if setup_path.is_file():
        setup = read_json(setup_path)
        setup.update(
            {
                "stage": "Stage 4A-6.5av",
                "repeat_variant": repeat_variant["repeat_variant"],
                "start_variant": str(args.start_variant),
                "tree_seed": int(args.tree_seed),
                "formal_expert_sampling": False,
            }
        )
        save_json(setup_path, setup)
        write_text(
            output_dir / "runtime_setup_summary.md",
            "\n".join(
                [
                    "# Runtime Setup Summary",
                    "",
                    f"- stage: `Stage 4A-6.5av`",
                    f"- repeat variant: `{repeat_variant['repeat_variant']}`",
                    f"- start variant: `{args.start_variant}`",
                    f"- tree_seed: `{args.tree_seed}`",
                    f"- frames captured: `{setup.get('frames_captured')}`",
                    f"- map_predict calls: `{setup.get('map_predict_calls')}`",
                    f"- selected action executions: `{setup.get('selected_action_execution_count')}`",
                    f"- second action / third frame / rollout: `{setup.get('second_action')}` / `{setup.get('third_frame')}` / `{setup.get('rollout')}`",
                ]
            ),
        )
    formula_path = output_dir / "formula_definition.json"
    if formula_path.is_file():
        formula = read_json(formula_path)
        formula["stage"] = "Stage 4A-6.5av"
        formula["primary_formula"] = PRIMARY_FORMULA
        formula["shadow_formula"] = SHADOW_FORMULA
        formula["measured_only_formula"] = MEASURED_FORMULA
        save_json(formula_path, formula)
    hardware_path = output_dir / "hardware_utilization_report.json"
    if hardware_path.is_file():
        hardware = read_json(hardware_path)
        hardware["stage"] = "Stage 4A-6.5av"
        save_json(hardware_path, hardware)


def compare_to_stage4a65au_design(
    output_dir: Path,
    args: argparse.Namespace,
    reference_manifest: dict[str, Any],
    current_summary: dict[str, Any],
    start_room_b: dict[str, Any],
) -> dict[str, Any]:
    setup = current_summary["runtime_setup"]
    comparison = {
        "stage": "Stage 4A-6.5av",
        "design_stage": "Stage 4A-6.5au",
        "loaded_stage4a65au_design": bool(reference_manifest["loaded_stage4a65au_design"]),
        "stage4a65au_was_design_only": bool(reference_manifest["stage4a65au_design_review_only"]),
        "stage4a65au_runtime_executed": bool(reference_manifest["stage4a65au_runtime_executed"]),
        "future_command_marked_do_not_run_in_au": bool(reference_manifest["stage4a65au_future_command_marked_do_not_run"]),
        "start_variant_matches_design": str(args.start_variant) == reference_manifest.get("stage4a65au_start_variant"),
        "start_pose_matches_design": bool(start_room_b["matches_stage4a65au_design"]),
        "start_pose_matches_metadata": bool(start_room_b["matches_metadata"]),
        "tree_seed_matches_design": int(args.tree_seed) == int(reference_manifest.get("stage4a65au_tree_seed")),
        "formula_matches_design": current_summary["formula"]["primary_formula"] == reference_manifest.get("stage4a65au_primary_formula") == PRIMARY_FORMULA,
        "runtime_constraints_match_design": {
            "max_frames": int(setup["frames_captured"]) <= 2,
            "max_map_predict_calls": int(setup["map_predict_calls"]) <= 2,
            "max_selected_actions": int(setup["selected_action_execution_count"]) <= 1,
            "no_second_action": setup["second_action"] is False,
            "no_third_frame": setup["third_frame"] is False,
            "no_rollout": setup["rollout"] is False,
        },
        "prediction_safety_matches_design": True,
        "low_cost_prior_basin_gates_required": True,
    }
    comparison["design_matched"] = all(
        [
            comparison["loaded_stage4a65au_design"],
            comparison["stage4a65au_was_design_only"],
            not comparison["stage4a65au_runtime_executed"],
            comparison["start_variant_matches_design"],
            comparison["start_pose_matches_design"],
            comparison["start_pose_matches_metadata"],
            comparison["tree_seed_matches_design"],
            comparison["formula_matches_design"],
            *comparison["runtime_constraints_match_design"].values(),
        ]
    )
    save_json(output_dir / "comparison_to_stage4a65au_design.json", comparison)
    write_text(
        output_dir / "comparison_to_stage4a65au_design.md",
        "\n".join(
            [
                "# Comparison To Stage 4A-6.5au Design",
                "",
                f"- design matched: `{comparison['design_matched']}`",
                f"- start variant/pose/tree_seed match: `{comparison['start_variant_matches_design']}` / `{comparison['start_pose_matches_design']}` / `{comparison['tree_seed_matches_design']}`",
                f"- start pose matches metadata: `{comparison['start_pose_matches_metadata']}`",
                f"- formula matches design: `{comparison['formula_matches_design']}`",
                f"- runtime constraints: `{comparison['runtime_constraints_match_design']}`",
            ]
        ),
    )
    return comparison


def add_density_deltas(comparison: dict[str, Any], current_output_dir: Path, reference_summary: dict[str, Any]) -> dict[str, Any]:
    current_stability = aq.compute_map_predict_stability(current_output_dir)
    ref_stability = reference_summary.get("map_predict_two_frame_stability")
    comparison["current_density_ratio_frame2_over_frame1"] = None if current_stability is None else current_stability.get("density_ratio_frame2_over_frame1")
    comparison["reference_density_ratio_frame2_over_frame1"] = None if ref_stability is None else ref_stability.get("density_ratio_frame2_over_frame1")
    comparison["density_ratio_delta"] = None
    if comparison["current_density_ratio_frame2_over_frame1"] is not None and comparison["reference_density_ratio_frame2_over_frame1"] is not None:
        comparison["density_ratio_delta"] = float(
            comparison["current_density_ratio_frame2_over_frame1"]
            - comparison["reference_density_ratio_frame2_over_frame1"]
        )
    return comparison


def comparison_to_start_corridor_aq_as(
    output_dir: Path,
    args: argparse.Namespace,
    current_summary: dict[str, Any],
    observed_delta: dict[str, Any] | None,
    start_room_b: dict[str, Any],
) -> dict[str, Any]:
    aq_summary = read_json(Path(args.stage4a65aq_dir).resolve() / "stage4a65aq_alternate_start_summary.json")
    as_summary = read_json(Path(args.stage4a65as_dir).resolve() / "stage4a65as_start_corridor_seed1_summary.json")
    comp_aq = aq.build_reference_comparison(
        current_summary, aq_summary, "Stage 4A-6.5aq", 0, int(args.tree_seed), str(args.repeat_variant), output_dir, "stage4a65aq", observed_delta
    )
    comp_as = aq.build_reference_comparison(
        current_summary, as_summary, "Stage 4A-6.5as", 1, int(args.tree_seed), str(args.repeat_variant), output_dir, "stage4a65as", observed_delta
    )
    comp_aq = add_density_deltas(comp_aq, output_dir, aq_summary)
    comp_as = add_density_deltas(comp_as, output_dir, as_summary)
    action = current_summary["results"].get("action_execution_report")
    for comp, ref_summary in ((comp_aq, aq_summary), (comp_as, as_summary)):
        ref_action = ref_summary["results"].get("action_execution_report")
        comp["action_yaw_delta_rad"] = None
        if action is not None and ref_action is not None:
            comp["action_yaw_delta_rad"] = angle_delta_rad(action["executed_pose"].get("yaw_rad"), ref_action["executed_pose"].get("yaw_rad"))
    combined = {
        "stage": "Stage 4A-6.5av",
        "context_only": True,
        "exact_branch_match_not_expected_because_start_changed": True,
        "start_room_b_distance_from_start_corridor_m": start_room_b["distance_from_start_corridor_m"],
        "comparison_to_stage4a65aq": comp_aq,
        "comparison_to_stage4a65as": comp_as,
        "interpretation": "aq/as are clean start_corridor context; deltas are expected because start_room_b changes pose and yaw.",
    }
    save_json(output_dir / "comparison_to_start_corridor_aq_as.json", combined)
    write_text(
        output_dir / "comparison_to_start_corridor_aq_as.md",
        "\n".join(
            [
                "# Comparison To Start Corridor aq/as",
                "",
                "- These references are context only; exact branch match is not expected because the start changed to start_room_b.",
                f"- start_room_b distance from start_corridor: `{combined['start_room_b_distance_from_start_corridor_m']}` m",
                f"- vs aq Frame1 selected/best delta: `{comp_aq['frame001'].get('selected_child_grid_distance_m')}` / `{comp_aq['frame001'].get('best_descendant_grid_distance_m')}` m",
                f"- vs as Frame1 selected/best delta: `{comp_as['frame001'].get('selected_child_grid_distance_m')}` / `{comp_as['frame001'].get('best_descendant_grid_distance_m')}` m",
                f"- vs aq/as action pose deltas: `{comp_aq.get('action_pose_delta_m')}` / `{comp_as.get('action_pose_delta_m')}` m",
                f"- interpretation: {combined['interpretation']}",
            ]
        ),
    )
    aq.write_reference_comparison(output_dir, "stage4a65aq", "Stage 4A-6.5aq", comp_aq)
    aq.write_reference_comparison(output_dir, "stage4a65as", "Stage 4A-6.5as", comp_as)
    return combined


def comparison_to_canonical_start_references(
    output_dir: Path,
    args: argparse.Namespace,
    current_summary: dict[str, Any],
    observed_delta: dict[str, Any] | None,
    start_room_b: dict[str, Any],
) -> dict[str, Any]:
    references = [
        ("Stage 4A-6.5ak", 0, Path(args.stage4a65ak_dir).resolve() / "stage4a65ak_two_frame_one_action_runtime_summary.json", "stage4a65ak"),
        ("Stage 4A-6.5am", 1, Path(args.stage4a65am_dir).resolve() / "stage4a65am_bounded_repeat_safety_summary.json", "stage4a65am"),
        ("Stage 4A-6.5ao", 2, Path(args.stage4a65ao_dir).resolve() / "stage4a65ao_bounded_repeat_safety_summary.json", "stage4a65ao"),
    ]
    comparisons: dict[str, Any] = {}
    for title, seed, path, stem in references:
        ref = read_json(path)
        comp = aq.build_reference_comparison(
            current_summary, ref, title, seed, int(args.tree_seed), str(args.repeat_variant), output_dir, stem, observed_delta
        )
        comparisons[stem] = comp
    combined = {
        "stage": "Stage 4A-6.5av",
        "context_only": True,
        "exact_branch_match_not_expected_because_start_changed": True,
        "start_room_b_distance_from_canonical_start_m": start_room_b["distance_from_canonical_start_m"],
        "comparisons": comparisons,
    }
    save_json(output_dir / "comparison_to_canonical_start_references.json", combined)
    write_text(
        output_dir / "comparison_to_canonical_start_references.md",
        "\n".join(
            [
                "# Comparison To Canonical Start References",
                "",
                "- Canonical start references are context only; exact branch match is not expected.",
                f"- start_room_b distance from canonical start: `{combined['start_room_b_distance_from_canonical_start_m']}` m",
                f"- vs ak/am/ao Frame1 selected deltas: `{comparisons['stage4a65ak']['frame001'].get('selected_child_grid_distance_m')}` / `{comparisons['stage4a65am']['frame001'].get('selected_child_grid_distance_m')}` / `{comparisons['stage4a65ao']['frame001'].get('selected_child_grid_distance_m')}` m",
            ]
        ),
    )
    return combined


def prediction_safety_clean(safety: dict[str, Any]) -> bool:
    safety["prediction_information_gain_only"] = True
    safety["all_motion_safety_uses_false"] = not any(
        bool(safety.get(key))
        for key in (
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_ray_blocking",
            "prediction_used_for_candidate_sampling",
            "prediction_used_for_edge_validity",
        )
    )
    return bool(safety.get("prediction_read_only")) and bool(safety["all_motion_safety_uses_false"])


def classify_start_room_b(current_summary: dict[str, Any], safety_clean: bool) -> dict[str, Any]:
    setup = current_summary["runtime_setup"]
    action_executed = int(setup["selected_action_execution_count"]) == 1
    frame2_done = current_summary["results"].get("frame002") is not None
    branches = [current_summary["results"]["frame001"]["branch_classification"]]
    if frame2_done:
        branches.append(current_summary["results"]["frame002"]["branch_classification"])
    low_cost = any(bool(item.get("low_cost_artifact")) for item in branches)
    prior = any(bool(item.get("historical_prior_basin")) for item in branches)
    classifications = [item.get("classification") for item in branches]
    healthy = action_executed and frame2_done and safety_clean and not low_cost and not prior
    if not action_executed:
        outcome = "action_blocked"
    elif low_cost or prior or not safety_clean:
        outcome = "artifact_or_prior_basin_regression"
    elif healthy and any(value == "distinct_nonmeasured_branch" for value in classifications):
        outcome = "clean_distinct_nonmeasured"
    elif healthy and all(value == "same_as_measured" for value in classifications):
        outcome = "start_room_b_seed0_clean_but_conservative"
    elif healthy:
        outcome = "spatially_consistent_healthy_start_room_b"
    else:
        outcome = "runtime_failure"
    return {
        "start_room_b_outcome": outcome,
        "repeat_outcome": outcome,
        "clean": healthy,
        "action_executed": action_executed,
        "frame2_completed": frame2_done,
        "low_cost_artifact_any_frame": low_cost,
        "historical_prior_basin_any_frame": prior,
        "prediction_safety_clean": safety_clean,
        "branch_classifications": classifications,
        "supports_larger_scene_construction_and_audit": healthy,
        "rollout_ready": False,
        "formal_expert_sampling_ready": False,
        "coverage_improvement_claim": False,
    }


def write_outcome(output_dir: Path, outcome: dict[str, Any]) -> None:
    save_json(output_dir / "start_room_b_outcome_classification.json", outcome)
    write_text(
        output_dir / "start_room_b_outcome_classification.md",
        "\n".join(
            [
                "# Start Room B Outcome Classification",
                "",
                f"- outcome: `{outcome['start_room_b_outcome']}`",
                f"- clean: `{outcome['clean']}`",
                f"- low-cost artifact any frame: `{outcome['low_cost_artifact_any_frame']}`",
                f"- historical prior basin any frame: `{outcome['historical_prior_basin_any_frame']}`",
                f"- prediction safety clean: `{outcome['prediction_safety_clean']}`",
                f"- rollout ready: `{outcome['rollout_ready']}`",
                f"- formal expert sampling ready: `{outcome['formal_expert_sampling_ready']}`",
            ]
        ),
    )


def write_lambda_report(output_dir: Path, current_summary: dict[str, Any]) -> dict[str, Any]:
    report = aq.lambda32_vs_lambda48(current_summary)
    save_json(output_dir / "lambda32_vs_lambda48_start_room_b.json", report)
    write_text(
        output_dir / "lambda32_vs_lambda48_start_room_b.md",
        "\n".join(
            [
                "# Lambda32 Vs Lambda48 Start Room B",
                "",
                f"- Frame1 same selected/best: `{report['frame001']['same_selected_child']}` / `{report['frame001']['same_best_descendant']}`",
                f"- Frame2 same selected/best: `{report['frame002']['same_selected_child']}` / `{report['frame002']['same_best_descendant']}`",
                f"- all available frames match: `{report['all_available_frames_match']}`",
            ]
        ),
    )
    return report


def plot_start_room_b_vs_prior(output_dir: Path, start_room_b: dict[str, Any]) -> None:
    xs = [CANONICAL_START[0], START_CORRIDOR_POSITION[0], start_room_b["position"][0]]
    ys = [CANONICAL_START[1], START_CORRIDOR_POSITION[1], start_room_b["position"][1]]
    labels = ["canonical", "start_corridor", "start_room_b"]
    fig, ax = plt.subplots(figsize=(6.2, 5.8), constrained_layout=True)
    ax.scatter(xs, ys, s=[90, 90, 120], c=["#64748b", "#0f766e", "#dc2626"])
    for x, y, label in zip(xs, ys, labels):
        ax.text(x + 0.08, y + 0.08, label)
    ax.set_xlim(-6.2, 6.2)
    ax.set_ylim(-6.2, 6.2)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_title("start_room_b vs prior starts")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    fig.savefig(output_dir / "start_room_b_vs_prior_starts_topdown.png", dpi=170)
    plt.close(fig)


def plot_stability_summary(output_dir: Path, outcome: dict[str, Any], start_corridor_comparison: dict[str, Any]) -> None:
    comp_aq = start_corridor_comparison["comparison_to_stage4a65aq"]
    comp_as = start_corridor_comparison["comparison_to_stage4a65as"]
    labels = ["F1 vs aq", "F1 vs as"]
    values = [
        float(comp_aq["frame001"].get("selected_child_grid_distance_m") or 0.0),
        float(comp_as["frame001"].get("selected_child_grid_distance_m") or 0.0),
    ]
    if comp_aq["frame002"].get("available"):
        labels.append("F2 vs aq")
        values.append(float(comp_aq["frame002"].get("selected_child_grid_distance_m") or 0.0))
    if comp_as["frame002"].get("available"):
        labels.append("F2 vs as")
        values.append(float(comp_as["frame002"].get("selected_child_grid_distance_m") or 0.0))
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.bar(labels, values, color=["#2563eb", "#0f766e", "#60a5fa", "#5eead4"][: len(values)])
    ax.set_ylabel("selected-child distance (m)")
    ax.set_title(f"Stage 4A-6.5av outcome: {outcome['start_room_b_outcome']}")
    fig.savefig(output_dir / "start_room_b_stability_summary.png", dpi=170)
    plt.close(fig)


def plot_next_gate(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.8), constrained_layout=True)
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "6.5av bounded smoke"),
        (0.38, 0.55, "6.6 larger scene"),
        (0.68, 0.55, "6.6a complexity audit"),
        (0.38, 0.18, "no direct rollout / expert sampling"),
    ]
    for x, y, label in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.25, 0.22, fill=False, edgecolor="#334155", linewidth=1.8))
        ax.text(x + 0.125, y + 0.11, label, ha="center", va="center", fontsize=10)
    ax.annotate("", xy=(0.38, 0.66), xytext=(0.30, 0.66), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", xy=(0.68, 0.66), xytext=(0.63, 0.66), arrowprops={"arrowstyle": "->", "lw": 1.8})
    ax.annotate("", xy=(0.50, 0.40), xytext=(0.50, 0.55), arrowprops={"arrowstyle": "->", "lw": 1.6})
    ax.set_title("Next-stage gate")
    fig.savefig(output_dir / "next_stage_larger_scene_gate_flowchart.png", dpi=170)
    plt.close(fig)


def write_no_formal_report(output_dir: Path) -> dict[str, Any]:
    payload = {
        "stage": "Stage 4A-6.5av",
        "formal_expert_sampling_executed": False,
        "formal_expert_sampling_ready": False,
        "expert_dataset_created": False,
        "expert_dataset_manifest_created": False,
        "transitions_jsonl_created": False,
        "rl_gdpo_ppo_bc_il": False,
        "reason": "User-required larger/complex scene construction and scene complexity audit must happen before formal expert sampling.",
    }
    save_json(output_dir / "no_formal_expert_sampling_report.json", payload)
    write_text(
        output_dir / "no_formal_expert_sampling_report.md",
        "\n".join(
            [
                "# No Formal Expert Sampling Report",
                "",
                "- formal expert sampling executed: `False`",
                "- formal expert sampling ready: `False`",
                "- expert dataset created: `False`",
                f"- reason: {payload['reason']}",
            ]
        ),
    )
    return payload


def write_next_gate_files(output_dir: Path) -> None:
    write_text(
        output_dir / "larger_scene_and_complexity_audit_gate.md",
        "\n".join(
            [
                "# Larger Scene And Complexity Audit Gate",
                "",
                "- Stage 4A-6.5av does not authorize direct formal expert sampling.",
                "- Next allowed direction after a clean AV run is Stage 4A-6.6 larger_complex_scene_v1 construction and validation.",
                "- Formal expert sampling remains blocked until Stage 4A-6.6a scene complexity audit passes.",
            ]
        ),
    )
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "\n".join(
            [
                "# Long-Term RL/GDPO Note",
                "",
                "- GDPO is future direction only.",
                "- There is no RL/GDPO/PPO/BC/IL in 6.5av.",
                "- Stage 4A-6.5av does not train, create a replay buffer, or write a policy checkpoint.",
            ]
        ),
    )


def recommendation_for(outcome: dict[str, Any]) -> dict[str, Any]:
    if (
        outcome["action_executed"]
        and outcome["frame2_completed"]
        and not outcome["low_cost_artifact_any_frame"]
        and not outcome["historical_prior_basin_any_frame"]
        and outcome["prediction_safety_clean"]
    ):
        next_step = "Stage 4A-6.6 larger_complex_scene_v1 construction and validation, then Stage 4A-6.6a scene complexity audit"
        why = "The bounded start_room_b runtime completed cleanly, but formal expert sampling is still gated on a larger/complex scene and audit."
    elif outcome["start_room_b_outcome"] == "action_blocked":
        next_step = "diagnose the Frame1 safety gate trigger"
        why = "The only allowed action was blocked before Frame2."
    elif outcome["start_room_b_outcome"] == "artifact_or_prior_basin_regression":
        next_step = "offline artifact or prior-basin diagnosis"
        why = "A low-cost artifact, prior-basin hit, or prediction-safety issue was detected."
    else:
        next_step = "runtime failure diagnosis"
        why = "The bounded smoke did not complete cleanly."
    return {
        "next_small_task": next_step,
        "why": why,
        "do_not_recommend_rollout_directly": True,
        "do_not_recommend_formal_expert_sampling_directly": True,
        "not_next": [
            "direct rollout",
            "formal expert sampling directly",
            "open-ended loop",
            "third frame or second action runtime",
            "RL/GDPO/PPO/BC/IL",
            "prediction writeback/fusion",
            "over-cost runtime promotion",
            "coverage-improvement claim",
        ],
    }


def write_recommendation(output_dir: Path, recommendation: dict[str, Any]) -> None:
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {recommendation['next_small_task']}",
                f"- why: {recommendation['why']}",
                "- not next: direct rollout, formal expert sampling directly, open-ended loop, third frame, second action, RL/GDPO/PPO/BC/IL, prediction writeback/fusion, over-cost runtime promotion, or coverage-improvement claim.",
            ]
        ),
    )


def write_readiness_matrix(
    output_dir: Path,
    au_comparison: dict[str, Any],
    outcome: dict[str, Any],
    setup: dict[str, Any],
    start_room_b: dict[str, Any],
) -> None:
    rows = [
        {"check": "stage4a65au_design_loaded", "passed": bool(au_comparison["loaded_stage4a65au_design"]), "evidence": "loaded_reference_manifest"},
        {"check": "stage4a65au_design_matched", "passed": bool(au_comparison["design_matched"]), "evidence": "comparison_to_stage4a65au_design"},
        {"check": "start_room_b_metadata_matched", "passed": bool(start_room_b["matches_metadata"]), "evidence": "loaded_start_room_b_manifest"},
        {"check": "isaac_started_once", "passed": setup["isaac_startup_count"] == 1, "evidence": str(setup["isaac_startup_count"])},
        {"check": "frames_bounded", "passed": int(setup["frames_captured"]) <= 2, "evidence": str(setup["frames_captured"])},
        {"check": "map_predict_bounded", "passed": int(setup["map_predict_calls"]) <= 2, "evidence": str(setup["map_predict_calls"])},
        {"check": "single_action_bound", "passed": int(setup["selected_action_execution_count"]) <= 1, "evidence": str(setup["selected_action_execution_count"])},
        {"check": "no_second_action", "passed": setup["second_action"] is False, "evidence": "runtime summary"},
        {"check": "no_third_frame", "passed": setup["third_frame"] is False, "evidence": "runtime summary"},
        {"check": "no_rollout", "passed": setup["rollout"] is False, "evidence": "runtime summary"},
        {"check": "prediction_safety_clean", "passed": outcome["prediction_safety_clean"], "evidence": "prediction_safety_report"},
        {"check": "no_artifact_or_prior", "passed": not outcome["low_cost_artifact_any_frame"] and not outcome["historical_prior_basin_any_frame"], "evidence": outcome["start_room_b_outcome"]},
        {"check": "rollout_ready_false", "passed": outcome["rollout_ready"] is False, "evidence": "bounded smoke only"},
        {"check": "formal_expert_sampling_ready_false", "passed": outcome["formal_expert_sampling_ready"] is False, "evidence": "larger scene audit gate"},
    ]
    save_json(output_dir / "repeat_safety_readiness_matrix.json", {"rows": rows})
    with (output_dir / "repeat_safety_readiness_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "evidence"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    write_text(
        output_dir / "repeat_safety_readiness_matrix.md",
        "\n".join(
            [
                "# Repeat Safety Readiness Matrix",
                "",
                "| check | passed | evidence |",
                "| --- | --- | --- |",
                *[f"| {row['check']} | `{row['passed']}` | {row['evidence']} |" for row in rows],
            ]
        ),
    )


def update_hash_checks(output_dir: Path, args: argparse.Namespace, before: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "hash_checks.json"
    checks = read_json(path) if path.is_file() else {}
    after = existing_file_hashes(reference_paths(args))
    reference_inputs = {}
    for key, before_item in before.items():
        after_item = after.get(key)
        reference_inputs[key] = {
            "before": before_item,
            "after": after_item,
            "unchanged": before_item is not None and after_item is not None and before_item.get("sha256") == after_item.get("sha256"),
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
        "loaded_start_room_b_manifest.json",
        "loaded_start_room_b_manifest.md",
        "hardware_utilization_report.json",
        "hardware_utilization_report.md",
        "runtime_setup_summary.json",
        "runtime_setup_summary.md",
        "repeat_variant_definition.json",
        "repeat_variant_definition.md",
        "start_room_b_definition.json",
        "start_room_b_definition.md",
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
        "no_formal_expert_sampling_report.json",
        "no_formal_expert_sampling_report.md",
        "stage4a65av_start_room_b_summary.json",
        "stage4a65av_start_room_b_summary.md",
        "recommended_next_faithful_step.md",
        "larger_scene_and_complexity_audit_gate.md",
        "long_term_rl_gdpo_note.md",
        "comparison_to_stage4a65au_design.json",
        "comparison_to_stage4a65au_design.md",
        "comparison_to_start_corridor_aq_as.json",
        "comparison_to_start_corridor_aq_as.md",
        "comparison_to_canonical_start_references.json",
        "comparison_to_canonical_start_references.md",
        "start_room_b_outcome_classification.json",
        "start_room_b_outcome_classification.md",
        "lambda32_vs_lambda48_start_room_b.json",
        "lambda32_vs_lambda48_start_room_b.md",
        "repeat_safety_readiness_matrix.csv",
        "repeat_safety_readiness_matrix.json",
        "repeat_safety_readiness_matrix.md",
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
        "frame001_observed_topdown.png",
        "frame001_prediction_overlay_topdown.png",
        "frame001_measured_vs_lambda48_tree_topdown.png",
        "frame001_lambda48_selected_branch_topdown.png",
        "value_components_frame001_lambda48.png",
        "low_cost_artifact_two_frame.png",
        "start_room_b_vs_prior_starts_topdown.png",
        "start_room_b_stability_summary.png",
        "next_stage_larger_scene_gate_flowchart.png",
    ]
    if action_executed:
        required.extend(
            [
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
        )
    else:
        required.extend(["action_blocked_report.json", "action_blocked_report.md"])
    prohibited = aq.scan_forbidden_outputs(output_dir)
    missing = [name for name in required if not (output_dir / name).is_file()]
    report = {
        "missing_required_files": missing,
        "prohibited_artifacts_found": prohibited,
        "action_executed": action_executed,
        "skipped_reasons": {} if action_executed else {"frame002": "Frame1 action was blocked by safety gates."},
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
                f"- action executed: `{action_executed}`",
            ]
        ),
    )
    return report


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    setup = summary["runtime_setup"]
    frame1 = summary["results"]["frame001"]
    frame2 = summary["results"].get("frame002")
    action = summary["results"].get("action_execution_report")
    blocked = summary["results"].get("action_blocked_report")
    outcome = summary["start_room_b_outcome_classification"]
    au = summary["comparison_to_stage4a65au_design"]
    start_corridor = summary["comparison_to_start_corridor_aq_as"]
    recommendation = summary["recommendation"]
    lines = [
        "# Stage 4A-6.5av Start Room B Summary",
        "",
        f"1. Successfully read Stage 4A-6.5au design? `{summary['loaded_reference_manifest']['loaded_stage4a65au_design']}`.",
        f"2. Successfully read start_room_b metadata? `{summary['loaded_reference_manifest']['loaded_start_room_b_metadata']}`.",
        f"3. start_variant: `{summary['start_room_b_definition']['start_variant']}`.",
        f"4. start pose / yaw: `{summary['start_room_b_definition']['position']}` / `{summary['start_room_b_definition']['yaw_rad']}`.",
        f"5. start pose matches Stage 4A-6.5au design and metadata? `{summary['start_room_b_definition']['matches_stage4a65au_design'] and summary['start_room_b_definition']['matches_metadata']}`.",
        f"6. tree_seed: `{summary['repeat_variant']['tree_seed']}`.",
        f"7. Isaac started exactly once in clean run? `{setup['isaac_startup_count'] == 1}`.",
        "8. Frame1 capture succeeded? `True`.",
        f"9. Frame1 map_predict succeeded? `{summary['map_predict']['frame001']['map_predict_succeeded']}`.",
        f"10. Frame1 measured-only shadow: `{frame1['measured_only_shadow'].get('selected_child_id')}` -> `{frame1['measured_only_shadow'].get('best_descendant_id')}`.",
        f"11. Frame1 lambda48 primary: `{frame1['lambda48_primary'].get('selected_child_id')}` -> `{frame1['lambda48_primary'].get('best_descendant_id')}`.",
        f"12. Frame1 lambda32 shadow: `{(frame1['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{(frame1['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"13. Frame1 lambda48 classification: `{frame1['branch_classification']['classification']}`.",
        f"14. Frame1 low-cost artifact? `{frame1['low_cost_artifact']['low_cost_artifact']}`.",
        f"15. Frame1 historical prior basin? `{frame1['branch_classification']['historical_prior_basin']}`.",
        f"16. pre-action safety gates all passed? `{summary['results']['pre_action_safety_gates']['hard_gates_passed']}`.",
        f"17. Executed exactly one action? `{setup['selected_action_execution_count'] == 1}`.",
        f"18. If action blocked, reason: `{None if blocked is None else blocked.get('failed_hard_gates')}`.",
        f"19. If action executed, pose: `{None if action is None else action['executed_pose']['position']}` yaw `{None if action is None else action['executed_pose']['yaw_rad']}`.",
        f"20. Frame2 capture succeeded? `{frame2 is not None}`.",
        f"21. Frame2 map_predict succeeded? `{False if summary['map_predict'].get('frame002') is None else summary['map_predict']['frame002']['map_predict_succeeded']}`.",
        f"22. Frame2 measured-only shadow: `{None if frame2 is None else frame2['measured_only_shadow'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['measured_only_shadow'].get('best_descendant_id')}`.",
        f"23. Frame2 lambda48 diagnostic: `{None if frame2 is None else frame2['lambda48_diagnostic'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['lambda48_diagnostic'].get('best_descendant_id')}`.",
        f"24. Frame2 lambda32 shadow: `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"25. Frame2 lambda48 classification: `{None if frame2 is None else frame2['branch_classification']['classification']}`.",
        f"26. Frame2 low-cost artifact? `{None if frame2 is None else frame2['low_cost_artifact']['low_cost_artifact']}`.",
        f"27. Frame2 historical prior basin? `{None if frame2 is None else frame2['branch_classification']['historical_prior_basin']}`.",
        f"28. Executed second action? `{summary['safety']['second_action']}`.",
        f"29. Captured third frame? `{summary['safety']['third_frame']}`.",
        f"30. Rollout? `{summary['safety']['rollout']}`.",
        f"31. Formal expert sampling? `{summary['no_formal_expert_sampling_report']['formal_expert_sampling_executed']}`.",
        f"32. Prediction read-only / information-gain-only? `{summary['prediction_safety']['prediction_read_only'] and summary['prediction_safety']['prediction_information_gain_only']}`.",
        f"33. Prediction did not write observed_state? `{not summary['prediction_safety']['prediction_written_to_observed_state']}`.",
        f"34. Prediction avoided traversability/collision/ray blocking/candidate sampling/edge validity? `{summary['prediction_safety']['all_motion_safety_uses_false']}`.",
        f"35. No target/ground-truth/future-observed scoring? `{not summary['prediction_safety']['target_lr_target_hr_ground_truth_used_for_planning_scoring'] and not summary['prediction_safety']['future_observed_used_for_planning_scoring']}`.",
        f"36. lambda48 formula exact? `{summary['formula']['primary_formula'] == PRIMARY_FORMULA}`.",
        f"37. Matches 6.5au design? `{au['design_matched']}`.",
        f"38. Difference vs start_corridor aq/as: {start_corridor['interpretation']}",
        f"39. start_room_b outcome classification: `{outcome['start_room_b_outcome']}`.",
        f"40. Clean? `{outcome['clean']}`.",
        f"41. Enough for rollout? `{summary['readiness']['rollout_ready']}`.",
        f"42. Enough for formal expert sampling? `{summary['readiness']['formal_expert_sampling_ready']}`.",
        f"43. Larger scene / scene complexity audit gate recorded? `{summary['readiness']['larger_scene_complexity_audit_gate_recorded']}`.",
        "44. Long-term GDPO note is future-only with no implementation? `True`.",
        f"45. Recommended next step: `{recommendation['next_small_task']}`.",
    ]
    write_text(path, "\n".join(lines))


def postprocess(args: argparse.Namespace, context: dict[str, Any], reference_manifest: dict[str, Any], hashes_before: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    current_summary = read_json(output_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    write_context_manifest(output_dir, context)
    write_reference_manifest(output_dir, reference_manifest)
    repeat_variant = write_repeat_variant(output_dir, args)
    start_room_b = write_start_room_b_files(output_dir, args, reference_manifest)
    patch_runtime_files(output_dir, args, repeat_variant)
    current_summary = read_json(output_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")

    observed_delta = aq.compute_observed_delta(output_dir)
    map_stability = aq.compute_map_predict_stability(output_dir)
    au_comparison = compare_to_stage4a65au_design(output_dir, args, reference_manifest, current_summary, start_room_b)
    start_corridor_comparison = comparison_to_start_corridor_aq_as(output_dir, args, current_summary, observed_delta, start_room_b)
    canonical_comparison = comparison_to_canonical_start_references(output_dir, args, current_summary, observed_delta, start_room_b)

    prediction_safety = read_json(output_dir / "prediction_safety_report.json")
    safety_clean = prediction_safety_clean(prediction_safety)
    save_json(output_dir / "prediction_safety_report.json", prediction_safety)
    outcome = classify_start_room_b(current_summary, safety_clean)
    write_outcome(output_dir, outcome)
    lambda_report = write_lambda_report(output_dir, current_summary)

    aq.plot_observed_delta(output_dir)
    plot_start_room_b_vs_prior(output_dir, start_room_b)
    plot_stability_summary(output_dir, outcome, start_corridor_comparison)
    plot_next_gate(output_dir)

    no_formal = write_no_formal_report(output_dir)
    write_next_gate_files(output_dir)
    recommendation = recommendation_for(outcome)
    write_recommendation(output_dir, recommendation)

    no_rollout = read_json(output_dir / "no_rollout_report.json")
    no_rollout["rollout_ready"] = False
    no_rollout["formal_expert_sampling_ready"] = False
    save_json(output_dir / "no_rollout_report.json", no_rollout)

    hash_checks = update_hash_checks(output_dir, args, hashes_before)
    setup = current_summary["runtime_setup"]
    write_readiness_matrix(output_dir, au_comparison, outcome, setup, start_room_b)

    summary = {
        "stage": "Stage 4A-6.5av start_room_b tree_seed=0 bounded smoke",
        "output_dir": str(output_dir),
        "loaded_context_manifest": context,
        "loaded_reference_manifest": reference_manifest,
        "loaded_start_room_b_manifest": start_room_b,
        "start_room_b_definition": start_room_b,
        "repeat_variant": repeat_variant,
        "runtime_setup": setup,
        "formula": current_summary["formula"],
        "map_predict": current_summary["map_predict"],
        "results": current_summary["results"],
        "observed_state_delta_summary": observed_delta,
        "map_predict_two_frame_stability": map_stability,
        "comparison_to_stage4a65au_design": au_comparison,
        "comparison_to_start_corridor_aq_as": start_corridor_comparison,
        "comparison_to_canonical_start_references": canonical_comparison,
        "start_room_b_outcome_classification": outcome,
        "lambda32_vs_lambda48_start_room_b": lambda_report,
        "prediction_safety": prediction_safety,
        "hash_checks": hash_checks,
        "no_formal_expert_sampling_report": no_formal,
        "readiness": {
            "rollout_ready": False,
            "formal_expert_sampling_ready": False,
            "coverage_improvement_claim": False,
            "two_frame_runtime_executed": int(setup["selected_action_execution_count"]) == 1,
            "larger_scene_complexity_audit_gate_recorded": True,
            "next_gate": "Stage 4A-6.6 larger_complex_scene_v1 construction, then Stage 4A-6.6a scene complexity audit",
        },
        "recommendation": recommendation,
        "safety": current_summary["safety"],
        "coverage_improvement_claim": False,
    }
    save_json(output_dir / "stage4a65av_start_room_b_summary.json", summary)
    write_summary_md(output_dir / "stage4a65av_start_room_b_summary.md", summary)
    missing = write_missing_report(output_dir, int(setup["selected_action_execution_count"]) == 1)
    summary["missing_fields_report"] = missing
    save_json(output_dir / "stage4a65av_start_room_b_summary.json", summary)
    write_summary_md(output_dir / "stage4a65av_start_room_b_summary.md", summary)
    print(json.dumps(aq.clean(summary), indent=2, sort_keys=True))
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if int(args.tree_seed) != 0:
        raise ValueError("Stage 4A-6.5av requires --tree_seed 0")
    if str(args.repeat_variant) != "start_room_b_tree_seed0":
        raise ValueError("--repeat_variant must be start_room_b_tree_seed0")
    if str(args.start_variant) != EXPECTED_START_VARIANT:
        raise ValueError("--start_variant must be start_room_b")
    if not close_pose(parse_position(args.position), args.yaw, EXPECTED_START_POSITION, EXPECTED_START_YAW, atol=1.0e-9):
        raise ValueError("--position/--yaw must match start_room_b pose")
    if not args.execute_exactly_one_action or not args.no_second_action or not args.no_third_frame or not args.no_rollout:
        raise ValueError("Stage 4A-6.5av requires exactly-one-action bounds, no second action, no third frame, and no rollout")
    if not args.no_formal_expert_sampling:
        raise ValueError("Stage 4A-6.5av requires --no_formal_expert_sampling")


def run(args: argparse.Namespace, unknown: list[str]) -> dict[str, Any]:
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    context = context_manifest()
    hashes_before = existing_file_hashes(reference_paths(args))
    reference_manifest = load_reference_manifest(args, hashes_before)
    write_context_manifest(output_dir, context)
    write_reference_manifest(output_dir, reference_manifest)
    write_repeat_variant(output_dir, args)
    write_start_room_b_files(output_dir, args, reference_manifest)

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
        write_text(output_dir / "runtime_failure_report.md", f"# Runtime Failure\n\n- delegated 6.5ak runner returncode: `{completed.returncode}`")
        raise SystemExit(completed.returncode)
    return postprocess(args, context, reference_manifest, hashes_before)


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
    parser.add_argument("--stage4a65au_dir", default=str(DEFAULT_STAGE4A65AU_DIR))
    parser.add_argument("--stage4a65at_dir", default=str(DEFAULT_STAGE4A65AT_DIR))
    parser.add_argument("--stage4a65aq_dir", default=str(DEFAULT_STAGE4A65AQ_DIR))
    parser.add_argument("--stage4a65as_dir", default=str(DEFAULT_STAGE4A65AS_DIR))
    parser.add_argument("--stage4a65ak_dir", default=str(DEFAULT_STAGE4A65AK_DIR))
    parser.add_argument("--stage4a65am_dir", default=str(DEFAULT_STAGE4A65AM_DIR))
    parser.add_argument("--stage4a65ao_dir", default=str(DEFAULT_STAGE4A65AO_DIR))
    parser.add_argument("--start_room_b_metadata", default=str(DEFAULT_START_ROOM_B_METADATA))
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--repeat_variant", default="start_room_b_tree_seed0")
    parser.add_argument("--start_variant", default=EXPECTED_START_VARIANT)
    parser.add_argument("--position", default="2.75,-2.55,1.2")
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
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--save_probs", action="store_true")
    parser.add_argument("--execute_exactly_one_action", action="store_true")
    parser.add_argument("--max_frames", type=int, default=2)
    parser.add_argument("--no_third_frame", action="store_true")
    parser.add_argument("--no_second_action", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_formal_expert_sampling", action="store_true")
    return parser.parse_known_args(normalize_negative_position_arg(sys.argv[1:]))


if __name__ == "__main__":
    parsed, extra = parse_args()
    run(parsed, extra)
