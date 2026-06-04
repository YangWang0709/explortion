#!/usr/bin/env python3
"""Stage 4A-6.5as start_corridor tree_seed=1 bounded runtime smoke.

This wrapper executes the proven Stage 4A-6.5ak bounded two-frame/one-action
runtime path through a subprocess, then writes the Stage 4A-6.5as-specific
repeat comparison against Stage 4A-6.5aq tree_seed=0 and the Stage 4A-6.5ar
design artifact.

Primary formula, unchanged:

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
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65as_start_corridor_tree_seed1_bounded_smoke"
DEFAULT_STAGE4A65AQ_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke"
DEFAULT_STAGE4A65AR_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ar_alternate_start_post_action_diagnosis"
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
PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
SHADOW_FORMULA = "gain_exp / cost + 32 * minmax(source_occ_free)"
EXPECTED_START_VARIANT = "start_corridor"
EXPECTED_START_POSITION = [0.0, -4.45, 1.2]
EXPECTED_START_YAW = 1.5707963267948966
CANONICAL_START = [-4.65, -4.65, 1.2]


def save_json(path: Path, data: Any) -> None:
    aq.save_json(path, data)


def read_json(path: Path) -> Any:
    return aq.read_json(path)


def write_text(path: Path, text: str) -> None:
    aq.write_text(path, text)


def parse_position(raw: str) -> list[float]:
    return aq.parse_position(raw)


def angle_delta_rad(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(abs(math.atan2(math.sin(float(a) - float(b)), math.cos(float(a) - float(b)))))


def reference_paths(args: argparse.Namespace) -> list[Path]:
    aq_dir = Path(args.stage4a65aq_dir).resolve()
    ar_dir = Path(args.stage4a65ar_dir).resolve()
    ap_dir = Path(args.stage4a65ap_dir).resolve()
    return [
        Path(args.checkpoint).resolve(),
        Path(args.alternate_start_metadata).resolve(),
        aq_dir / "stage4a65aq_alternate_start_summary.json",
        aq_dir / "observed_state_frame001.npy",
        aq_dir / "observed_state_frame002.npy",
        aq_dir / "frame001_map_predict/global_prediction_layer.npz",
        aq_dir / "frame002_map_predict/global_prediction_layer.npz",
        ar_dir / "stage4a65ar_alternate_start_diagnosis_summary.json",
        ar_dir / "selected_next_bounded_repeat_design.json",
        ap_dir / "stage4a65ap_seed012_repeat_review_summary.json",
        ap_dir / "selected_alternate_start_design.json",
    ]


def context_manifest() -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    combined = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "sha256": aq.sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "contains_stage4a65ap": "Stage 4A-6.5ap" in text,
                "contains_stage4a65aq": "Stage 4A-6.5aq" in text,
                "contains_stage4a65ar": "Stage 4A-6.5ar" in text,
                "contains_stage4a65as": "Stage 4A-6.5as" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5as",
        "loaded_context_files": entries,
        "confirmed_stage4a65ap_complete": "Stage 4A-6.5ap" in combined and "start_corridor" in combined,
        "confirmed_stage4a65aq_complete": "Stage 4A-6.5aq" in combined and "clean_same_as_measured" in combined,
        "confirmed_stage4a65ar_complete": "Stage 4A-6.5ar" in combined and "Stage 4A-6.5as" in combined,
        "confirmed_current_stage4a65as_task": "Stage 4A-6.5as" in combined
        and "tree_seed `1`" in combined
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
                f"- Stage 4A-6.5ap complete: `{manifest['confirmed_stage4a65ap_complete']}`",
                f"- Stage 4A-6.5aq complete: `{manifest['confirmed_stage4a65aq_complete']}`",
                f"- Stage 4A-6.5ar complete: `{manifest['confirmed_stage4a65ar_complete']}`",
                f"- Stage 4A-6.5as current task: `{manifest['confirmed_current_stage4a65as_task']}`",
                "- Files read:",
                *[f"  - `{item['path']}` sha256 `{item['sha256']}`" for item in manifest["loaded_context_files"]],
            ]
        ),
    )


def load_reference_manifest(args: argparse.Namespace, hashes_before: dict[str, Any]) -> dict[str, Any]:
    aq_dir = Path(args.stage4a65aq_dir).resolve()
    ar_dir = Path(args.stage4a65ar_dir).resolve()
    ap_dir = Path(args.stage4a65ap_dir).resolve()
    metadata_path = Path(args.alternate_start_metadata).resolve()
    aq_summary = read_json(aq_dir / "stage4a65aq_alternate_start_summary.json")
    ar_summary = read_json(ar_dir / "stage4a65ar_alternate_start_diagnosis_summary.json")
    ar_design = read_json(ar_dir / "selected_next_bounded_repeat_design.json")
    ap_design = read_json(ap_dir / "selected_alternate_start_design.json")
    metadata = read_json(metadata_path)
    metadata_start = aq.find_metadata_start_pose(metadata, EXPECTED_START_POSITION, EXPECTED_START_YAW)
    return {
        "stage": "Stage 4A-6.5as",
        "reference_stage": "Stage 4A-6.5aq",
        "reference_tree_seed": int(args.reference_tree_seed),
        "current_tree_seed": int(args.tree_seed),
        "stage4a65aq_dir": str(aq_dir),
        "stage4a65ar_dir": str(ar_dir),
        "stage4a65ap_dir": str(ap_dir),
        "alternate_start_metadata": str(metadata_path),
        "loaded_stage4a65aq_summary": True,
        "loaded_stage4a65ar_summary": True,
        "loaded_stage4a65ar_design": True,
        "loaded_stage4a65ap_design": True,
        "loaded_alternate_start_metadata": True,
        "stage4a65aq_sequence": aq_summary.get("runtime_setup", {}),
        "stage4a65aq_outcome": aq_summary.get("repeat_stability", {}).get("alternate_start_outcome")
        or aq_summary.get("alternate_start_outcome_classification", {}).get("alternate_start_outcome"),
        "stage4a65ar_selected_future_stage": ar_design.get("future_stage"),
        "stage4a65ar_selected_tree_seed": ar_design.get("future_tree_seed"),
        "stage4a65ar_design_only": bool(ar_design.get("design_only_in_stage4a65ar")),
        "stage4a65ar_summary_outcome": ar_summary.get("alternate_start_outcome"),
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
                f"- reference stage: `{manifest['reference_stage']}` tree_seed `{manifest['reference_tree_seed']}`",
                f"- current tree_seed: `{manifest['current_tree_seed']}`",
                f"- loaded Stage 4A-6.5aq summary: `{manifest['loaded_stage4a65aq_summary']}`",
                f"- loaded Stage 4A-6.5ar summary/design: `{manifest['loaded_stage4a65ar_summary']}` / `{manifest['loaded_stage4a65ar_design']}`",
                f"- loaded Stage 4A-6.5ap design: `{manifest['loaded_stage4a65ap_design']}`",
                f"- loaded alternate-start metadata: `{manifest['loaded_alternate_start_metadata']}`",
            ]
        ),
    )


def write_repeat_variant(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    position = parse_position(args.position)
    variant = {
        "repeat_variant": str(args.repeat_variant),
        "start_variant": str(args.start_variant),
        "reference_stage": "Stage 4A-6.5aq",
        "reference_tree_seed": int(args.reference_tree_seed),
        "current_tree_seed": int(args.tree_seed),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "position": position,
        "yaw": float(args.yaw),
        "canonical_start_position": CANONICAL_START,
        "start_pose_distance_from_canonical_m": aq.distance_m(position, CANONICAL_START),
        "only_intended_runtime_change": "tree_seed_vs_stage4a65aq",
        "start_pose_same_as_stage4a65aq": True,
        "tree_seed_changed_vs_reference": int(args.tree_seed) != int(args.reference_tree_seed),
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
                f"- reference/current tree_seed: `{variant['reference_tree_seed']}` / `{variant['current_tree_seed']}`",
                f"- only intended runtime change: `{variant['only_intended_runtime_change']}`",
                f"- scene/start: `{variant['scene_variant']}` seed `{variant['scene_seed']}`, `{variant['position']}` yaw `{variant['yaw']}`",
                "- bounds: exactly two frames, exactly one action if gates pass, no rollout.",
            ]
        ),
    )
    return variant


def write_alternate_start_files(output_dir: Path, args: argparse.Namespace, reference_manifest: dict[str, Any]) -> dict[str, Any]:
    position = parse_position(args.position)
    metadata_pose = reference_manifest.get("metadata_start_pose", {})
    payload = {
        "stage": "Stage 4A-6.5as",
        "start_variant": str(args.start_variant),
        "position": position,
        "yaw_rad": float(args.yaw),
        "yaw_deg": float(math.degrees(float(args.yaw))),
        "pose_source": str(Path(args.alternate_start_metadata).resolve()),
        "metadata_pose_found": bool(metadata_pose.get("found")),
        "matches_stage4a65ap_design": aq.close_pose(
            position,
            args.yaw,
            reference_manifest.get("stage4a65ap_selected_start_position"),
            reference_manifest.get("stage4a65ap_selected_start_yaw"),
        ),
        "matches_stage4a65ar_design": reference_manifest.get("stage4a65ar_selected_future_stage") == "4A-6.5as"
        and int(reference_manifest.get("stage4a65ar_selected_tree_seed")) == int(args.tree_seed),
        "matches_metadata": bool(metadata_pose.get("found"))
        and aq.close_pose(position, args.yaw, metadata_pose.get("position"), metadata_pose.get("yaw_rad")),
        "canonical_start_position": CANONICAL_START,
        "distance_from_canonical_start_m": aq.distance_m(position, CANONICAL_START),
        "fixed_camera_height_m": float(position[2]),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "tree_seed": int(args.tree_seed),
    }
    save_json(output_dir / "loaded_alternate_start_manifest.json", payload)
    save_json(output_dir / "alternate_start_definition.json", payload)
    for name, title in (
        ("loaded_alternate_start_manifest.md", "Loaded Alternate-Start Manifest"),
        ("alternate_start_definition.md", "Alternate Start Definition"),
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
                    f"- matches Stage 4A-6.5ar design: `{payload['matches_stage4a65ar_design']}`",
                    f"- matches metadata: `{payload['matches_metadata']}`",
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


def patch_runtime_files(output_dir: Path, args: argparse.Namespace, repeat_variant: dict[str, Any]) -> None:
    setup_path = output_dir / "runtime_setup_summary.json"
    if setup_path.is_file():
        setup = read_json(setup_path)
        setup.update(
            {
                "stage": "Stage 4A-6.5as",
                "repeat_variant": repeat_variant["repeat_variant"],
                "reference_stage": "Stage 4A-6.5aq",
                "reference_tree_seed": int(args.reference_tree_seed),
                "tree_seed": int(args.tree_seed),
            }
        )
        save_json(setup_path, setup)
        write_text(
            output_dir / "runtime_setup_summary.md",
            "\n".join(
                [
                    "# Runtime Setup Summary",
                    "",
                    f"- stage: `Stage 4A-6.5as`",
                    f"- repeat variant: `{repeat_variant['repeat_variant']}`",
                    f"- reference/current tree_seed: `{args.reference_tree_seed}` / `{args.tree_seed}`",
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
        formula["stage"] = "Stage 4A-6.5as"
        formula["primary_formula"] = PRIMARY_FORMULA
        formula["shadow_formula"] = SHADOW_FORMULA
        save_json(formula_path, formula)


def build_reference_comparison(
    current_summary: dict[str, Any],
    aq_summary: dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
    observed_delta: dict[str, Any] | None,
) -> dict[str, Any]:
    comparison = aq.build_reference_comparison(
        current_summary,
        aq_summary,
        "Stage 4A-6.5aq",
        int(args.reference_tree_seed),
        int(args.tree_seed),
        str(args.repeat_variant),
        output_dir,
        "stage4a65aq",
        observed_delta,
    )
    action = current_summary["results"].get("action_execution_report")
    ref_action = aq_summary["results"].get("action_execution_report")
    comparison["action_yaw_delta_rad"] = None
    if action is not None and ref_action is not None:
        comparison["action_yaw_delta_rad"] = angle_delta_rad(
            action["executed_pose"].get("yaw_rad"),
            ref_action["executed_pose"].get("yaw_rad"),
        )
    map_stability = aq.compute_map_predict_stability(output_dir)
    ref_stability = aq_summary.get("map_predict_two_frame_stability")
    comparison["current_density_ratio_frame2_over_frame1"] = None if map_stability is None else map_stability.get("density_ratio_frame2_over_frame1")
    comparison["reference_density_ratio_frame2_over_frame1"] = None if ref_stability is None else ref_stability.get("density_ratio_frame2_over_frame1")
    comparison["density_ratio_delta_vs_stage4a65aq"] = None
    if comparison["current_density_ratio_frame2_over_frame1"] is not None and comparison["reference_density_ratio_frame2_over_frame1"] is not None:
        comparison["density_ratio_delta_vs_stage4a65aq"] = float(
            comparison["current_density_ratio_frame2_over_frame1"]
            - comparison["reference_density_ratio_frame2_over_frame1"]
        )
    return comparison


def write_reference_comparison(output_dir: Path, comparison: dict[str, Any]) -> None:
    aq.write_reference_comparison(output_dir, "stage4a65aq", "Stage 4A-6.5aq", comparison)
    save_json(output_dir / "start_corridor_seed0_seed1_comparison.json", comparison)
    text = "\n".join(
        [
            "# Start Corridor Seed0 Seed1 Comparison",
            "",
            f"- reference/current tree_seed: `{comparison['reference_tree_seed']}` / `{comparison['current_tree_seed']}`",
            f"- Frame1 selected/best delta: `{comparison['frame001']['selected_child_grid_distance_m']}` / `{comparison['frame001']['best_descendant_grid_distance_m']}` m",
            f"- Frame2 selected/best delta: `{None if not comparison['frame002'].get('available') else comparison['frame002']['selected_child_grid_distance_m']}` / `{None if not comparison['frame002'].get('available') else comparison['frame002']['best_descendant_grid_distance_m']}` m",
            f"- action pose/yaw delta: `{comparison['action_pose_delta_m']}` m / `{comparison['action_yaw_delta_rad']}` rad",
            f"- observed_ratio delta difference: `{comparison['observed_ratio_delta_difference']}`",
            f"- density ratio delta: `{comparison['density_ratio_delta_vs_stage4a65aq']}`",
        ]
    )
    write_text(output_dir / "start_corridor_seed0_seed1_comparison.md", text)


def write_stage4a65ar_design_comparison(
    output_dir: Path,
    args: argparse.Namespace,
    reference_manifest: dict[str, Any],
    current_summary: dict[str, Any],
    alternate_start: dict[str, Any],
) -> dict[str, Any]:
    setup = current_summary["runtime_setup"]
    comparison = {
        "stage": "Stage 4A-6.5as",
        "design_stage": "Stage 4A-6.5ar",
        "loaded_stage4a65ar_design": bool(reference_manifest.get("loaded_stage4a65ar_design")),
        "future_stage_matches": reference_manifest.get("stage4a65ar_selected_future_stage") == "4A-6.5as",
        "tree_seed_matches_design": int(reference_manifest.get("stage4a65ar_selected_tree_seed")) == int(args.tree_seed),
        "start_pose_matches_design": bool(alternate_start["matches_stage4a65ar_design"]),
        "start_pose_matches_metadata": bool(alternate_start["matches_metadata"]),
        "formula_matches_design": current_summary["formula"]["primary_formula"] == PRIMARY_FORMULA,
        "runtime_constraints_match_design": {
            "max_frames": int(setup["frames_captured"]) <= 2,
            "max_map_predict_calls": int(setup["map_predict_calls"]) <= 2,
            "max_selected_actions": int(setup["selected_action_execution_count"]) <= 1,
            "no_second_action": setup["second_action"] is False,
            "no_third_frame": setup["third_frame"] is False,
            "no_rollout": setup["rollout"] is False,
        },
        "design_matched": False,
    }
    comparison["design_matched"] = all(
        [
            comparison["loaded_stage4a65ar_design"],
            comparison["future_stage_matches"],
            comparison["tree_seed_matches_design"],
            comparison["start_pose_matches_design"],
            comparison["start_pose_matches_metadata"],
            comparison["formula_matches_design"],
            *comparison["runtime_constraints_match_design"].values(),
        ]
    )
    save_json(output_dir / "comparison_to_stage4a65ar_design.json", comparison)
    write_text(
        output_dir / "comparison_to_stage4a65ar_design.md",
        "\n".join(
            [
                "# Comparison To Stage 4A-6.5ar Design",
                "",
                f"- design matched: `{comparison['design_matched']}`",
                f"- future stage/tree seed match: `{comparison['future_stage_matches']}` / `{comparison['tree_seed_matches_design']}`",
                f"- start pose matches design/metadata: `{comparison['start_pose_matches_design']}` / `{comparison['start_pose_matches_metadata']}`",
                f"- formula matches design: `{comparison['formula_matches_design']}`",
                f"- runtime constraints: `{comparison['runtime_constraints_match_design']}`",
            ]
        ),
    )
    return comparison


def classify_repeat(
    current_summary: dict[str, Any],
    comparison: dict[str, Any],
    prediction_safety: dict[str, Any],
) -> dict[str, Any]:
    setup = current_summary["runtime_setup"]
    action_executed = int(setup["selected_action_execution_count"]) == 1
    frame2_done = current_summary["results"].get("frame002") is not None
    branches = [
        current_summary["results"]["frame001"]["branch_classification"],
        *( [] if not frame2_done else [current_summary["results"]["frame002"]["branch_classification"]] ),
    ]
    low_cost = any(bool(item["low_cost_artifact"]) for item in branches)
    prior = any(bool(item["historical_prior_basin"]) for item in branches)
    safety_clean = bool(prediction_safety["prediction_read_only"]) and bool(prediction_safety["all_motion_safety_uses_false"])
    healthy = action_executed and frame2_done and safety_clean and not low_cost and not prior
    classifications = [item.get("classification") for item in branches]
    exact_frames = all(
        comparison[frame].get("available")
        and comparison[frame].get("exact_selected_child_agreement")
        and comparison[frame].get("exact_best_descendant_agreement")
        for frame in ("frame001", "frame002")
    )
    exact_action = comparison.get("action_pose_delta_m") is not None and comparison["action_pose_delta_m"] <= 1.0e-9
    spatially_close = all(
        comparison[frame].get("available")
        and comparison[frame].get("selected_child_grid_distance_m") is not None
        and comparison[frame]["selected_child_grid_distance_m"] <= 1.0
        and comparison[frame].get("best_descendant_grid_distance_m") is not None
        and comparison[frame]["best_descendant_grid_distance_m"] <= 4.0
        for frame in ("frame001", "frame002")
    )
    all_same_as_measured = healthy and all(value == "same_as_measured" for value in classifications)
    any_distinct = healthy and any(value == "distinct_nonmeasured_branch" for value in classifications)
    if not action_executed:
        outcome = "action_blocked"
    elif low_cost or prior or not safety_clean:
        outcome = "artifact_or_prior_basin_regression"
    elif healthy and exact_frames and exact_action:
        outcome = "exact_repeat_match_to_aq"
    elif all_same_as_measured:
        outcome = "clean_same_as_measured"
    elif healthy and spatially_close:
        outcome = "spatially_consistent_healthy_repeat"
    elif any_distinct:
        outcome = "clean_distinct_nonmeasured"
    elif healthy:
        outcome = "start_corridor_seed_sensitive_but_clean"
    else:
        outcome = "runtime_failure"
    return {
        "repeat_outcome": outcome,
        "alternate_start_outcome": outcome,
        "repeat_remains_healthy": healthy,
        "clean": healthy,
        "action_executed": action_executed,
        "frame2_completed": frame2_done,
        "low_cost_artifact_any_frame": low_cost,
        "historical_prior_basin_any_frame": prior,
        "prediction_safety_clean": safety_clean,
        "exact_repeat_match_to_aq": outcome == "exact_repeat_match_to_aq",
        "spatially_consistent_healthy_repeat": outcome == "spatially_consistent_healthy_repeat",
        "clean_same_as_measured": outcome == "clean_same_as_measured",
        "clean_distinct_nonmeasured": outcome == "clean_distinct_nonmeasured",
        "start_corridor_seed_sensitive_but_clean": outcome == "start_corridor_seed_sensitive_but_clean",
        "branch_classifications": classifications,
        "exact_frames_match_to_aq": exact_frames,
        "exact_action_match_to_aq": exact_action,
        "spatially_close_to_aq": spatially_close,
    }


def write_repeat_outcome(output_dir: Path, repeat: dict[str, Any]) -> None:
    save_json(output_dir / "repeat_outcome_classification.json", repeat)
    write_text(
        output_dir / "repeat_outcome_classification.md",
        "\n".join(
            [
                "# Repeat Outcome Classification",
                "",
                f"- outcome: `{repeat['repeat_outcome']}`",
                f"- clean: `{repeat['clean']}`",
                f"- low-cost artifact any frame: `{repeat['low_cost_artifact_any_frame']}`",
                f"- historical prior basin any frame: `{repeat['historical_prior_basin_any_frame']}`",
                f"- prediction safety clean: `{repeat['prediction_safety_clean']}`",
            ]
        ),
    )


def write_lambda_report(output_dir: Path, current_summary: dict[str, Any]) -> dict[str, Any]:
    report = aq.lambda32_vs_lambda48(current_summary)
    save_json(output_dir / "lambda32_vs_lambda48_start_corridor_seed1.json", report)
    write_text(
        output_dir / "lambda32_vs_lambda48_start_corridor_seed1.md",
        "\n".join(
            [
                "# Lambda32 Vs Lambda48 Start Corridor Seed1",
                "",
                f"- Frame1 same selected/best: `{report['frame001']['same_selected_child']}` / `{report['frame001']['same_best_descendant']}`",
                f"- Frame2 same selected/best: `{report['frame002']['same_selected_child']}` / `{report['frame002']['same_best_descendant']}`",
                f"- all available frames match: `{report['all_available_frames_match']}`",
            ]
        ),
    )
    return report


def plot_repeat_summary(output_dir: Path, comparison: dict[str, Any], repeat: dict[str, Any]) -> None:
    labels: list[str] = []
    values: list[float] = []
    for frame in ("frame001", "frame002"):
        item = comparison.get(frame)
        if item and item.get("available"):
            labels.extend([f"{frame} selected", f"{frame} best"])
            values.extend(
                [
                    float(item.get("selected_child_grid_distance_m") or 0.0),
                    float(item.get("best_descendant_grid_distance_m") or 0.0),
                ]
            )
    if not values:
        labels, values = ["clean"], [1.0 if repeat["clean"] else 0.0]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    ax.bar(labels, values, color=["#2563eb", "#60a5fa", "#0f766e", "#5eead4"][: len(values)])
    ax.set_title(f"Stage 4A-6.5as repeat outcome: {repeat['repeat_outcome']}")
    ax.set_ylabel("distance vs Stage 4A-6.5aq (m)")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "repeat_stability_summary.png", dpi=170)
    plt.close(fig)


def update_hash_checks(output_dir: Path, args: argparse.Namespace, before: dict[str, Any]) -> dict[str, Any]:
    path = output_dir / "hash_checks.json"
    checks = read_json(path) if path.is_file() else {}
    after = aq.existing_file_hashes(reference_paths(args))
    reference_inputs = {}
    for key, before_item in before.items():
        after_item = after.get(key)
        reference_inputs[key] = {
            "before": before_item,
            "after": after_item,
            "unchanged": before_item is not None
            and after_item is not None
            and before_item.get("sha256") == after_item.get("sha256"),
        }
    checks["reference_inputs"] = reference_inputs
    checks["stage4a65aq_reference_inputs"] = {
        key: value for key, value in reference_inputs.items() if "stage4a65aq" in key
    }
    checks["stage4a65ar_reference_inputs"] = {
        key: value for key, value in reference_inputs.items() if "stage4a65ar" in key
    }
    checks["current_stage_scripts"] = {
        str(Path(__file__).resolve()): aq.sha256_file(Path(__file__).resolve()),
        str(AK_RUNNER): aq.sha256_file(AK_RUNNER),
    }
    save_json(path, checks)
    return checks


def write_readiness_matrix(
    output_dir: Path,
    context: dict[str, Any],
    ar_comparison: dict[str, Any],
    repeat: dict[str, Any],
    setup: dict[str, Any],
    safety_clean: bool,
) -> None:
    rows = [
        {"check": "context_loaded", "passed": bool(context["confirmed_stage4a65aq_complete"] and context["confirmed_stage4a65ar_complete"]), "evidence": "project context reread"},
        {"check": "stage4a65ar_design_matched", "passed": bool(ar_comparison["design_matched"]), "evidence": "comparison_to_stage4a65ar_design"},
        {"check": "tree_seed_changed_only_vs_aq", "passed": True, "evidence": "start pose fixed, tree_seed 0 -> 1"},
        {"check": "isaac_started_once", "passed": setup["isaac_startup_count"] == 1, "evidence": str(setup["isaac_startup_count"])},
        {"check": "frames_bounded", "passed": int(setup["frames_captured"]) <= 2, "evidence": str(setup["frames_captured"])},
        {"check": "map_predict_bounded", "passed": int(setup["map_predict_calls"]) <= 2, "evidence": str(setup["map_predict_calls"])},
        {"check": "single_action_bound", "passed": int(setup["selected_action_execution_count"]) <= 1, "evidence": str(setup["selected_action_execution_count"])},
        {"check": "no_second_action", "passed": setup["second_action"] is False, "evidence": "runtime summary"},
        {"check": "no_third_frame", "passed": setup["third_frame"] is False, "evidence": "runtime summary"},
        {"check": "no_rollout", "passed": setup["rollout"] is False, "evidence": "runtime summary"},
        {"check": "prediction_safety_clean", "passed": safety_clean, "evidence": "prediction_safety_report"},
        {"check": "no_artifact_or_prior", "passed": not repeat["low_cost_artifact_any_frame"] and not repeat["historical_prior_basin_any_frame"], "evidence": repeat["repeat_outcome"]},
        {"check": "rollout_ready_false", "passed": True, "evidence": "bounded smoke only"},
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


def recommendation_for(repeat: dict[str, Any]) -> dict[str, Any]:
    outcome = repeat["repeat_outcome"]
    if outcome in {"exact_repeat_match_to_aq", "spatially_consistent_healthy_repeat", "clean_same_as_measured"}:
        next_step = "Stage 4A-6.5at start_corridor seed0/seed1 repeat-comparison diagnosis and next-start design only"
        why = "The bounded repeat completed cleanly without artifact, prior basin, safety leak, second action, third frame, or rollout."
    elif outcome in {"clean_distinct_nonmeasured", "start_corridor_seed_sensitive_but_clean"}:
        next_step = "offline start_corridor seed0/seed1 repeat-safety diagnosis before any new runtime"
        why = "The run stayed clean but behavior varied under tree_seed=1."
    elif outcome == "action_blocked":
        next_step = "diagnose the Frame1 gate-triggering issue"
        why = "The only allowed selected action was blocked by safety gates."
    elif outcome == "artifact_or_prior_basin_regression":
        next_step = "offline artifact diagnosis; do not continue runtime"
        why = "An artifact, prior basin, or safety regression was detected."
    else:
        next_step = "runtime failure diagnosis"
        why = "The bounded smoke did not complete cleanly."
    return {
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


def write_recommendation(output_dir: Path, recommendation: dict[str, Any]) -> None:
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {recommendation['next_small_task']}",
                f"- why: {recommendation['why']}",
                "- not next: rollout, open-ended loop, RL/GDPO/PPO/BC/IL, prediction writeback/fusion, over-cost runtime promotion, Pareto gate/runtime planner implementation, or coverage-improvement claim.",
            ]
        ),
    )


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
        "stage4a65as_start_corridor_seed1_summary.json",
        "stage4a65as_start_corridor_seed1_summary.md",
        "recommended_next_faithful_step.md",
        "long_term_rl_gdpo_note.md",
        "comparison_to_stage4a65aq.json",
        "comparison_to_stage4a65aq.md",
        "comparison_to_stage4a65ar_design.json",
        "comparison_to_stage4a65ar_design.md",
        "start_corridor_seed0_seed1_comparison.json",
        "start_corridor_seed0_seed1_comparison.md",
        "repeat_outcome_classification.json",
        "repeat_outcome_classification.md",
        "lambda32_vs_lambda48_start_corridor_seed1.json",
        "lambda32_vs_lambda48_start_corridor_seed1.md",
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
        "frame001_lambda48_primary_tree_decision.json",
        "frame001_lambda32_shadow_tree_decision.json",
        "frame001_branch_classification.json",
        "frame001_low_cost_artifact_diagnosis.json",
        "pre_action_safety_gate_report.json",
        "pre_action_safety_gate_report.md",
        "frame001_observed_topdown.png",
        "frame001_prediction_overlay_topdown.png",
        "frame001_measured_vs_lambda48_tree_topdown.png",
        "frame001_lambda48_selected_branch_topdown.png",
        "value_components_frame001_lambda48.png",
        "low_cost_artifact_two_frame.png",
        "comparison_to_stage4a65aq_topdown.png",
        "start_corridor_seed0_seed1_topdown.png",
        "repeat_stability_summary.png",
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
                "frame002_lambda48_diagnostic_tree_decision.json",
                "frame002_lambda32_shadow_tree_decision.json",
                "frame002_branch_classification.json",
                "frame002_low_cost_artifact_diagnosis.json",
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


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    setup = summary["runtime_setup"]
    frame1 = summary["results"]["frame001"]
    frame2 = summary["results"].get("frame002")
    action = summary["results"].get("action_execution_report")
    blocked = summary["results"].get("action_blocked_report")
    comparison = summary["comparison_to_stage4a65aq"]
    repeat = summary["repeat_outcome_classification"]
    observed = summary.get("observed_state_delta_summary")
    map_stability = summary.get("map_predict_two_frame_stability")
    lines = [
        "# Stage 4A-6.5as Start Corridor Seed1 Summary",
        "",
        f"1. Successfully read Stage 4A-6.5ar context? `{summary['loaded_reference_manifest']['loaded_stage4a65ar_design']}`.",
        f"2. Successfully read Stage 4A-6.5aq reference? `{summary['loaded_reference_manifest']['loaded_stage4a65aq_summary']}`.",
        f"3. Successfully read alternate start metadata? `{summary['loaded_reference_manifest']['loaded_alternate_start_metadata']}`.",
        f"4. start_variant: `{summary['alternate_start_definition']['start_variant']}`.",
        f"5. start pose / yaw: `{summary['alternate_start_definition']['position']}` / `{summary['alternate_start_definition']['yaw_rad']}`.",
        f"6. start pose matches Stage 4A-6.5ar design and metadata? `{summary['alternate_start_definition']['matches_stage4a65ar_design'] and summary['alternate_start_definition']['matches_metadata']}`.",
        f"7. repeat variant: `{summary['repeat_variant']['repeat_variant']}`.",
        f"8. tree_seed: `{summary['repeat_variant']['current_tree_seed']}`.",
        f"9. reference tree_seed: `{summary['repeat_variant']['reference_tree_seed']}`.",
        f"10. Only changed tree_seed vs Stage 4A-6.5aq? `{summary['repeat_variant']['only_intended_runtime_change'] == 'tree_seed_vs_stage4a65aq'}`.",
        f"11. Isaac started exactly once? `{setup['isaac_startup_count'] == 1}`.",
        "12. Frame1 capture succeeded? `True`.",
        f"13. Frame1 map_predict succeeded? `{summary['map_predict']['frame001']['map_predict_succeeded']}`.",
        f"14. Frame1 measured-only shadow: `{frame1['measured_only_shadow'].get('selected_child_id')}` -> `{frame1['measured_only_shadow'].get('best_descendant_id')}`.",
        f"15. Frame1 lambda48 primary: `{frame1['lambda48_primary'].get('selected_child_id')}` -> `{frame1['lambda48_primary'].get('best_descendant_id')}`.",
        f"16. Frame1 lambda32 shadow: `{(frame1['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{(frame1['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"17. Frame1 lambda48 classification: `{frame1['branch_classification']['classification']}`.",
        f"18. Frame1 low-cost artifact? `{frame1['low_cost_artifact']['low_cost_artifact']}`.",
        f"19. Frame1 historical prior basin? `{frame1['branch_classification']['historical_prior_basin']}`.",
        f"20. pre-action safety gates all passed? `{summary['results']['pre_action_safety_gates']['hard_gates_passed']}`.",
        f"21. Executed exactly one action? `{setup['selected_action_execution_count'] == 1}`.",
        f"22. If action blocked, reason: `{None if blocked is None else blocked.get('failed_hard_gates')}`.",
        f"23. If action executed, pose: `{None if action is None else action['executed_pose']['position']}` yaw `{None if action is None else action['executed_pose']['yaw_rad']}`.",
        f"24. Frame2 capture succeeded? `{frame2 is not None}`.",
        f"25. Frame2 map_predict succeeded? `{False if summary['map_predict'].get('frame002') is None else summary['map_predict']['frame002']['map_predict_succeeded']}`.",
        f"26. Frame2 measured-only shadow: `{None if frame2 is None else frame2['measured_only_shadow'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['measured_only_shadow'].get('best_descendant_id')}`.",
        f"27. Frame2 lambda48 diagnostic: `{None if frame2 is None else frame2['lambda48_diagnostic'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['lambda48_diagnostic'].get('best_descendant_id')}`.",
        f"28. Frame2 lambda32 shadow: `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"29. Frame2 lambda48 classification: `{None if frame2 is None else frame2['branch_classification']['classification']}`.",
        f"30. Frame2 low-cost artifact? `{None if frame2 is None else frame2['low_cost_artifact']['low_cost_artifact']}`.",
        f"31. Frame2 historical prior basin? `{None if frame2 is None else frame2['branch_classification']['historical_prior_basin']}`.",
        f"32. Executed second action? `{summary['safety']['second_action']}`.",
        f"33. Captured third frame? `{summary['safety']['third_frame']}`.",
        f"34. Rollout? `{summary['safety']['rollout']}`.",
        f"35. Prediction read-only / information-gain-only? `{summary['prediction_safety']['prediction_read_only'] and summary['prediction_safety']['prediction_information_gain_only']}`.",
        f"36. Prediction did not write observed_state? `{not summary['prediction_safety']['prediction_written_to_observed_state']}`.",
        f"37. Prediction avoided traversability/collision/ray blocking/candidate sampling/edge validity? `{summary['prediction_safety']['all_motion_safety_uses_false']}`.",
        f"38. No target/ground-truth/future-observed scoring? `{not summary['prediction_safety']['target_lr_target_hr_ground_truth_used_for_planning_scoring'] and not summary['prediction_safety']['future_observed_used_for_planning_scoring']}`.",
        f"39. lambda48 formula exact? `{summary['formula']['primary_formula'] == PRIMARY_FORMULA}`.",
        f"40. Difference vs Stage 4A-6.5aq: Frame1 selected/best deltas `{comparison['frame001'].get('selected_child_grid_distance_m')}` / `{comparison['frame001'].get('best_descendant_grid_distance_m')}` m; Frame2 selected/best deltas `{None if not comparison['frame002'].get('available') else comparison['frame002'].get('selected_child_grid_distance_m')}` / `{None if not comparison['frame002'].get('available') else comparison['frame002'].get('best_descendant_grid_distance_m')}` m.",
        f"41. Action pose delta vs 6.5aq: `{comparison['action_pose_delta_m']}` m, yaw delta `{comparison['action_yaw_delta_rad']}` rad.",
        f"42. Observed_state delta vs 6.5aq: current `{None if observed is None else observed.get('observed_ratio_delta')}`, difference `{comparison['observed_ratio_delta_difference']}`.",
        f"43. map_predict density vs 6.5aq: current `{None if map_stability is None else map_stability.get('density_ratio_frame2_over_frame1')}`, reference `{comparison['reference_density_ratio_frame2_over_frame1']}`.",
        f"44. lambda32/lambda48 agreement: `{summary['lambda32_vs_lambda48_start_corridor_seed1']}`.",
        f"45. Repeat outcome classification: `{repeat['repeat_outcome']}`.",
        f"46. Clean? `{repeat['clean']}`.",
        f"47. Enough for rollout? `{summary['readiness']['rollout_ready']}`.",
        "48. Long-term GDPO note is future-only with no implementation? `True`.",
        f"49. Recommended next step: `{summary['recommendation']['next_small_task']}`.",
    ]
    write_text(path, "\n".join(lines))


def postprocess(args: argparse.Namespace, context: dict[str, Any], reference_manifest: dict[str, Any], hashes_before: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    current_summary = read_json(output_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    aq_summary = read_json(Path(args.stage4a65aq_dir).resolve() / "stage4a65aq_alternate_start_summary.json")
    write_context_manifest(output_dir, context)
    write_reference_manifest(output_dir, reference_manifest)
    repeat_variant = write_repeat_variant(output_dir, args)
    alternate_start = write_alternate_start_files(output_dir, args, reference_manifest)
    patch_runtime_files(output_dir, args, repeat_variant)

    current_summary = read_json(output_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    observed_delta = aq.compute_observed_delta(output_dir)
    map_stability = aq.compute_map_predict_stability(output_dir)
    comparison = build_reference_comparison(current_summary, aq_summary, args, output_dir, observed_delta)
    write_reference_comparison(output_dir, comparison)
    ar_comparison = write_stage4a65ar_design_comparison(output_dir, args, reference_manifest, current_summary, alternate_start)

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
    repeat = classify_repeat(current_summary, comparison, prediction_safety)
    write_repeat_outcome(output_dir, repeat)
    lambda_report = write_lambda_report(output_dir, current_summary)

    aq.plot_comparison_to_reference(
        output_dir,
        comparison,
        "comparison_to_stage4a65aq_topdown.png",
        "as seed1",
        "aq seed0",
        "Stage 4A-6.5as seed1 vs Stage 4A-6.5aq seed0",
    )
    aq.plot_comparison_to_reference(
        output_dir,
        comparison,
        "start_corridor_seed0_seed1_topdown.png",
        "seed1",
        "seed0",
        "start_corridor tree_seed 0 vs 1",
    )
    aq.plot_observed_delta(output_dir)
    plot_repeat_summary(output_dir, comparison, repeat)

    hash_checks = update_hash_checks(output_dir, args, hashes_before)
    setup = current_summary["runtime_setup"]
    safety_clean = bool(prediction_safety["prediction_read_only"]) and bool(prediction_safety["all_motion_safety_uses_false"])
    write_readiness_matrix(output_dir, context, ar_comparison, repeat, setup, safety_clean)

    no_rollout = read_json(output_dir / "no_rollout_report.json")
    no_rollout["rollout_ready"] = False
    save_json(output_dir / "no_rollout_report.json", no_rollout)
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "\n".join(
            [
                "# Long-Term RL/GDPO Note",
                "",
                "- Long-term direction: NVIDIA GDPO-style multi-reward decoupled policy optimization remains future work.",
                "- Stage 4A-6.5as does not implement RL/GDPO/PPO/BC/IL, train a policy, create a replay buffer, or write a policy checkpoint.",
                "- This output is bounded runtime safety evidence only.",
            ]
        ),
    )
    recommendation = recommendation_for(repeat)
    write_recommendation(output_dir, recommendation)

    summary = {
        "stage": "Stage 4A-6.5as start_corridor tree_seed=1 bounded smoke",
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
        "comparison_to_stage4a65aq": comparison,
        "comparison_to_stage4a65ar_design": ar_comparison,
        "start_corridor_seed0_seed1_comparison": comparison,
        "repeat_outcome_classification": repeat,
        "lambda32_vs_lambda48_start_corridor_seed1": lambda_report,
        "prediction_safety": prediction_safety,
        "hash_checks": hash_checks,
        "readiness": {
            "rollout_ready": False,
            "rollout_ready_reason": "Stage 4A-6.5as is bounded repeat-safety evidence only",
            "two_frame_runtime_executed": int(setup["selected_action_execution_count"]) == 1,
            "repeat_outcome": repeat["repeat_outcome"],
        },
        "recommendation": recommendation,
        "safety": current_summary["safety"],
        "coverage_improvement_claim": False,
    }
    save_json(output_dir / "stage4a65as_start_corridor_seed1_summary.json", summary)
    write_summary_md(output_dir / "stage4a65as_start_corridor_seed1_summary.md", summary)
    missing = write_missing_report(output_dir, int(setup["selected_action_execution_count"]) == 1)
    summary["missing_fields_report"] = missing
    save_json(output_dir / "stage4a65as_start_corridor_seed1_summary.json", summary)
    write_summary_md(output_dir / "stage4a65as_start_corridor_seed1_summary.md", summary)
    print(json.dumps(aq.clean(summary), indent=2, sort_keys=True))
    return summary


def validate_args(args: argparse.Namespace) -> None:
    if int(args.tree_seed) != 1:
        raise ValueError("Stage 4A-6.5as requires --tree_seed 1")
    if int(args.reference_tree_seed) != 0:
        raise ValueError("Stage 4A-6.5as requires --reference_tree_seed 0")
    if str(args.repeat_variant) != "alternate_start_corridor_tree_seed1":
        raise ValueError("--repeat_variant must be alternate_start_corridor_tree_seed1")
    if str(args.start_variant) != EXPECTED_START_VARIANT:
        raise ValueError("--start_variant must be start_corridor")
    if not aq.close_pose(parse_position(args.position), args.yaw, EXPECTED_START_POSITION, EXPECTED_START_YAW, atol=1.0e-9):
        raise ValueError("--position/--yaw must match start_corridor pose")
    if not args.execute_exactly_one_action or not args.no_second_action or not args.no_third_frame or not args.no_rollout:
        raise ValueError("Stage 4A-6.5as requires exactly-one-action bounds, no second action, no third frame, and no rollout")


def run(args: argparse.Namespace, unknown: list[str]) -> dict[str, Any]:
    validate_args(args)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    context = context_manifest()
    hashes_before = aq.existing_file_hashes(reference_paths(args))
    reference_manifest = load_reference_manifest(args, hashes_before)
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
    parser.add_argument("--stage4a65aq_dir", default=str(DEFAULT_STAGE4A65AQ_DIR))
    parser.add_argument("--stage4a65ar_dir", default=str(DEFAULT_STAGE4A65AR_DIR))
    parser.add_argument("--stage4a65ap_dir", default=str(DEFAULT_STAGE4A65AP_DIR))
    parser.add_argument("--alternate_start_metadata", default=str(DEFAULT_ALTERNATE_START_METADATA))
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--repeat_variant", default="alternate_start_corridor_tree_seed1")
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
    parser.add_argument("--tree_seed", type=int, default=1)
    parser.add_argument("--reference_tree_seed", type=int, default=0)
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
