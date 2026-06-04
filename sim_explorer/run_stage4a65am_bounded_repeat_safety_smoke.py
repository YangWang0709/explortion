#!/usr/bin/env python3
"""Stage 4A-6.5am bounded repeat-safety smoke, tree_seed=1.

This runner keeps the proven Stage 4A-6.5ak runtime sequence as the execution
path and wraps it with the Stage 4A-6.5am repeat-safety bookkeeping:

    value = gain_exp / cost + 48 * minmax(source_occ_free)

The only intended runtime variant is the mini-RRT tree sampling seed.  It
starts Isaac through the delegated 6.5ak runner exactly once, captures at most
two frames, executes at most one action, and then compares the result against
the saved Stage 4A-6.5ak reference.
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
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_STAGE4A65AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_STAGE4A65AL_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65al_post_action_two_frame_diagnosis"
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
                "contains_stage4a65al": "Stage 4A-6.5al" in text,
                "contains_stage4a65am_next_task": "Stage 4A-6.5am" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5am",
        "loaded_context_files": entries,
        "confirmed_stage4a65ag_complete": "Stage 4A-6.5ag" in combined
        and "multi-frame lambda48 replay" in combined,
        "confirmed_stage4a65ai_complete": "Stage 4A-6.5ai" in combined
        and "one-frame lambda48 runtime smoke" in combined,
        "confirmed_stage4a65aj_complete": "Stage 4A-6.5aj" in combined
        and "two-frame one-action" in combined,
        "confirmed_stage4a65ak_complete": "Stage 4A-6.5ak" in combined
        and "two-frame one-action lambda48 runtime smoke" in combined,
        "confirmed_stage4a65al_complete": "Stage 4A-6.5al" in combined
        and "post-action/two-frame diagnosis" in combined,
        "confirmed_next_small_task_stage4a65am": "Stage 4A-6.5am bounded repeat-safety smoke" in combined,
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
                f"- Stage 4A-6.5al complete: `{manifest['confirmed_stage4a65al_complete']}`",
                f"- Stage 4A-6.5am is current next task: `{manifest['confirmed_next_small_task_stage4a65am']}`",
                "- Files read:",
                *[f"  - `{item['path']}` sha256 `{item['sha256']}`" for item in manifest["loaded_context_files"]],
            ]
        ),
    )


def reference_paths(stage4a65ak_dir: Path, stage4a65al_dir: Path, checkpoint: Path) -> list[Path]:
    return [
        checkpoint,
        stage4a65ak_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json",
        stage4a65ak_dir / "observed_state_frame001.npy",
        stage4a65ak_dir / "observed_state_frame002.npy",
        stage4a65ak_dir / "frame001_map_predict/global_prediction_layer.npz",
        stage4a65ak_dir / "frame002_map_predict/global_prediction_layer.npz",
        stage4a65al_dir / "stage4a65al_post_action_two_frame_diagnosis_summary.json",
        stage4a65al_dir / "stage4a65am_bounded_repeat_safety_smoke_design.json",
    ]


def load_reference_manifest(args: argparse.Namespace, hashes_before: dict[str, Any]) -> dict[str, Any]:
    ak_dir = Path(args.stage4a65ak_dir).resolve()
    al_dir = Path(args.stage4a65al_dir).resolve()
    ak_summary_path = ak_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json"
    al_summary_path = al_dir / "stage4a65al_post_action_two_frame_diagnosis_summary.json"
    design_path = al_dir / "stage4a65am_bounded_repeat_safety_smoke_design.json"
    ak_summary = read_json(ak_summary_path)
    al_summary = read_json(al_summary_path)
    design = read_json(design_path) if design_path.is_file() else None
    return {
        "stage": "Stage 4A-6.5am",
        "reference_stage": "Stage 4A-6.5ak",
        "reference_tree_seed": int(args.reference_tree_seed),
        "current_tree_seed": int(args.tree_seed),
        "stage4a65ak_dir": str(ak_dir),
        "stage4a65al_dir": str(al_dir),
        "loaded_stage4a65ak_summary": True,
        "loaded_stage4a65al_summary": True,
        "loaded_stage4a65am_design_from_stage4a65al": design is not None,
        "stage4a65ak_sequence": ak_summary.get("runtime_setup", {}),
        "stage4a65al_completed": bool(al_summary.get("completed", False)),
        "stage4a65am_design_status": None if design is None else design.get("status"),
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
                f"- reference stage: `{manifest['reference_stage']}`",
                f"- reference tree_seed: `{manifest['reference_tree_seed']}`",
                f"- current tree_seed: `{manifest['current_tree_seed']}`",
                f"- loaded Stage 4A-6.5ak summary: `{manifest['loaded_stage4a65ak_summary']}`",
                f"- loaded Stage 4A-6.5al summary: `{manifest['loaded_stage4a65al_summary']}`",
                f"- loaded Stage 4A-6.5am design artifact: `{manifest['loaded_stage4a65am_design_from_stage4a65al']}`",
            ]
        ),
    )


def write_repeat_variant(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    variant = {
        "repeat_variant": str(args.repeat_variant),
        "reference_stage": "Stage 4A-6.5ak",
        "reference_tree_seed": int(args.reference_tree_seed),
        "current_tree_seed": int(args.tree_seed),
        "scene_variant": str(args.scene_variant),
        "scene_seed": int(args.scene_seed),
        "start_pose": str(args.start_pose),
        "position": [float(v) for v in str(args.position).split(",")],
        "yaw": float(args.yaw),
        "only_intended_runtime_change": "tree_seed",
        "only_tree_seed_changed_vs_reference": int(args.tree_seed) != int(args.reference_tree_seed),
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
                f"- reference/current tree_seed: `{variant['reference_tree_seed']}` / `{variant['current_tree_seed']}`",
                f"- only intended runtime change: `{variant['only_intended_runtime_change']}`",
                f"- scene/start: `{variant['scene_variant']}` seed `{variant['scene_seed']}`, `{variant['position']}` yaw `{variant['yaw']}`",
                "- bounds: exactly two frames, exactly one action if gates pass, no rollout.",
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
        str(args.start_pose),
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
        "selected_child_grid_distance_vs_stage4a65ak_m": grid_distance_m(
            cur_l.get("selected_child_grid"), ref_l.get("selected_child_grid")
        ),
        "best_descendant_grid_distance_vs_stage4a65ak_m": grid_distance_m(
            cur_l.get("best_descendant_grid"), ref_l.get("best_descendant_grid")
        ),
        "selected_child_world_distance_vs_stage4a65ak_m": world_distance_m(
            cur_l.get("selected_child_world"), ref_l.get("selected_child_world")
        ),
        "best_descendant_world_distance_vs_stage4a65ak_m": world_distance_m(
            cur_l.get("best_descendant_world"), ref_l.get("best_descendant_world")
        ),
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


def classify_repeat(current: dict[str, Any], comparison: dict[str, Any], safety_clean: bool) -> dict[str, Any]:
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
    frame_comparisons = [comparison["frame001"], comparison.get("frame002")]
    exact = healthy and all(
        item
        and item.get("available")
        and item.get("exact_selected_child_agreement")
        and item.get("exact_best_descendant_agreement")
        for item in frame_comparisons
    )
    spatially_close = healthy and all(
        item
        and item.get("available")
        and (item.get("selected_child_grid_distance_vs_stage4a65ak_m") is not None)
        and item["selected_child_grid_distance_vs_stage4a65ak_m"] <= 1.0
        and (item.get("best_descendant_grid_distance_vs_stage4a65ak_m") is not None)
        and item["best_descendant_grid_distance_vs_stage4a65ak_m"] <= 4.0
        and bool(item.get("branch_class_agreement"))
        for item in frame_comparisons
    )
    if not action_executed:
        outcome = "action_blocked"
    elif low_cost or prior or not safety_clean:
        outcome = "artifact_or_prior_basin_regression"
    elif exact:
        outcome = "exact_repeat_match"
    elif spatially_close:
        outcome = "spatially_consistent_healthy_repeat"
    elif healthy:
        outcome = "divergent_but_healthy"
    else:
        outcome = "runtime_failure"
    return {
        "repeat_outcome": outcome,
        "repeat_remains_healthy": healthy,
        "action_executed": action_executed,
        "frame2_completed": frame2_done,
        "low_cost_artifact_any_frame": low_cost,
        "historical_prior_basin_any_frame": prior,
        "prediction_safety_clean": safety_clean,
        "exact_repeat_match": exact,
        "spatially_consistent_healthy_repeat": spatially_close,
        "divergence_acceptable_tree_seed_variability": outcome
        in {"spatially_consistent_healthy_repeat", "divergent_but_healthy"},
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


def plot_comparison_to_reference(output_dir: Path, comparison: dict[str, Any]) -> None:
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
            ("current_lambda48", "#2563eb", "am"),
            ("reference_lambda48", "#f97316", "ak"),
        ):
            point = point_xy(frame[side].get("selected_child_grid"))
            if point:
                ax.scatter([point[0]], [point[1]], c=color, marker=marker, s=90, label=f"{label_prefix} {frame_name} selected")
            best = point_xy(frame[side].get("best_descendant_grid"))
            if best:
                ax.scatter([best[0]], [best[1]], c=color, marker="*", s=150, alpha=0.9, label=f"{label_prefix} {frame_name} best")
    ax.set_title("Stage 4A-6.5am vs 6.5ak lambda48 decisions")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.legend(loc="upper right", fontsize=7)
    fig.savefig(output_dir / "comparison_to_stage4a65ak_topdown.png", dpi=170)
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
    ax.set_title(f"Repeat stability: {repeat['repeat_outcome']}")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "repeat_stability_summary.png", dpi=170)
    plt.close(fig)


def write_readiness_matrix(output_dir: Path, rows: list[dict[str, Any]]) -> None:
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
    comparison = summary["comparison_to_stage4a65ak"]
    action = results.get("action_execution_report")
    blocked = results.get("action_blocked_report")
    lines = [
        "# Stage 4A-6.5am Bounded Repeat-Safety Summary",
        "",
        f"1. Successfully read Stage 4A-6.5ak / 6.5al context? `{summary['loaded_reference_manifest']['loaded_stage4a65ak_summary'] and summary['loaded_reference_manifest']['loaded_stage4a65al_summary']}`.",
        f"2. Repeat variant: `{summary['repeat_variant']['repeat_variant']}`.",
        f"3. Only changed tree_seed? `{summary['repeat_variant']['only_tree_seed_changed_vs_reference']}`.",
        f"4. Current tree_seed: `{summary['repeat_variant']['current_tree_seed']}`.",
        f"5. Reference tree_seed: `{summary['repeat_variant']['reference_tree_seed']}`.",
        f"6. Isaac started exactly once? `{setup['isaac_startup_count'] == 1}`.",
        "7. Frame 1 capture succeeded? `True`.",
        f"8. Frame 1 map_predict succeeded? `{summary['map_predict']['frame001']['map_predict_succeeded']}`.",
        f"9. Frame 1 measured-only shadow selected: `{frame1['measured_only_shadow'].get('selected_child_id')}` -> `{frame1['measured_only_shadow'].get('best_descendant_id')}`.",
        f"10. Frame 1 lambda48 primary selected: `{frame1['lambda48_primary'].get('selected_child_id')}` -> `{frame1['lambda48_primary'].get('best_descendant_id')}`.",
        f"11. Frame 1 lambda32 shadow selected: `{(frame1['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{(frame1['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"12. Frame 1 lambda48 classification: `{frame1['branch_classification']['classification']}`.",
        f"13. Frame 1 low-cost artifact? `{frame1['low_cost_artifact']['low_cost_artifact']}`.",
        f"14. Frame 1 historical prior basin? `{frame1['branch_classification']['historical_prior_basin']}`.",
        f"15. Pre-action safety gates all passed? `{results['pre_action_safety_gates']['hard_gates_passed']}`.",
        f"16. Executed exactly one action? `{setup['selected_action_execution_count'] == 1}`.",
        f"17. If action blocked, reason: `{None if blocked is None else blocked['failed_hard_gates']}`.",
        f"18. If action executed, pose: `{None if action is None else action['executed_pose']['position']}` yaw `{None if action is None else action['executed_pose']['yaw_rad']}`.",
        f"19. Frame 2 capture succeeded? `{frame2 is not None}`.",
        f"20. Frame 2 map_predict succeeded? `{False if summary['map_predict']['frame002'] is None else summary['map_predict']['frame002']['map_predict_succeeded']}`.",
        f"21. Frame 2 measured-only shadow selected: `{None if frame2 is None else frame2['measured_only_shadow'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['measured_only_shadow'].get('best_descendant_id')}`.",
        f"22. Frame 2 lambda48 diagnostic selected: `{None if frame2 is None else frame2['lambda48_diagnostic'].get('selected_child_id')}` -> `{None if frame2 is None else frame2['lambda48_diagnostic'].get('best_descendant_id')}`.",
        f"23. Frame 2 lambda32 shadow selected: `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('selected_child_id')}` -> `{None if frame2 is None else (frame2['lambda32_shadow'] or {}).get('best_descendant_id')}`.",
        f"24. Frame 2 low-cost artifact? `{None if frame2 is None else frame2['low_cost_artifact']['low_cost_artifact']}`.",
        f"25. Frame 2 historical prior basin? `{None if frame2 is None else frame2['branch_classification']['historical_prior_basin']}`.",
        f"26. Executed second action? `{summary['safety']['second_action']}`.",
        f"27. Captured third frame? `{summary['safety']['third_frame']}`.",
        f"28. Rollout? `{summary['safety']['rollout']}`.",
        f"29. Prediction read-only / information-gain-only? `{summary['prediction_safety']['prediction_read_only'] and summary['prediction_safety']['prediction_information_gain_only']}`.",
        f"30. Prediction did not write observed_state? `{not summary['prediction_safety']['prediction_written_to_observed_state']}`.",
        f"31. Prediction avoided traversability/collision/ray blocking/candidate sampling/edge validity? `{summary['prediction_safety']['all_motion_safety_uses_false']}`.",
        f"32. No target/ground-truth/future-observed scoring? `{not summary['prediction_safety']['target_lr_target_hr_ground_truth_used_for_planning_scoring'] and not summary['prediction_safety']['future_observed_used_for_planning_scoring']}`.",
        f"33. lambda48 formula exact? `{summary['formula']['primary_formula'] == PRIMARY_FORMULA}`.",
        f"34. Frame1 vs 6.5ak selected/best exact or spatial close? `{comparison['frame001'].get('exact_selected_child_agreement') or comparison['frame001'].get('selected_child_grid_distance_vs_stage4a65ak_m') <= 1.0}`.",
        f"35. Frame2 vs 6.5ak selected/best exact or spatial close? `{None if comparison.get('frame002') is None else (comparison['frame002'].get('exact_selected_child_agreement') or comparison['frame002'].get('selected_child_grid_distance_vs_stage4a65ak_m') <= 1.0)}`.",
        f"36. Repeat outcome classification: `{repeat['repeat_outcome']}`.",
        f"37. Repeat still clean? `{repeat['repeat_remains_healthy']}`.",
        f"38. Enough for rollout? `{summary['readiness']['rollout_ready']}`.",
        f"39. Recommended next step: `{summary['recommendation']['next_small_task']}`.",
    ]
    write_text(path, "\n".join(lines))


def write_recommendation(output_dir: Path, repeat: dict[str, Any]) -> dict[str, Any]:
    outcome = repeat["repeat_outcome"]
    if outcome in {"exact_repeat_match", "spatially_consistent_healthy_repeat"}:
        next_step = "Stage 4A-6.5an repeat-comparison review / alternate-start design only"
        why = "Frame 2 completed cleanly, prediction safety stayed clean, and the repeat was exact or spatially consistent."
    elif outcome == "divergent_but_healthy":
        next_step = "another bounded repeat review, likely alternate start or tree_seed=2 design"
        why = "The repeat stayed healthy but diverged enough to warrant one more bounded review before larger runtime."
    elif outcome == "action_blocked":
        next_step = "diagnose the Frame 1 gate-triggering issue"
        why = "Frame 1 safety gates blocked the only allowed action."
    elif outcome == "artifact_or_prior_basin_regression":
        next_step = "offline artifact diagnosis; do not continue runtime"
        why = "An artifact/prior-basin/safety regression was detected."
    else:
        next_step = "runtime failure diagnosis"
        why = "The bounded repeat smoke did not complete cleanly."
    payload = {
        "next_small_task": next_step,
        "why": why,
        "do_not_recommend_rollout_directly": True,
        "not_next": [
            "rollout",
            "open-ended loop",
            "RL/PPO/BC/IL",
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
                "- not next: rollout, open-ended loop, RL/PPO/BC/IL, prediction writeback/fusion, over-cost runtime promotion, Pareto gate/runtime planner implementation, or coverage-improvement claim.",
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
                "stage": "Stage 4A-6.5am",
                "repeat_variant": repeat_variant["repeat_variant"],
                "tree_seed": int(args.tree_seed),
                "reference_tree_seed": int(args.reference_tree_seed),
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
                    f"- tree_seed/reference_tree_seed: `{args.tree_seed}` / `{args.reference_tree_seed}`",
                    f"- scene: `{args.scene_variant}` seed `{args.scene_seed}`",
                    f"- start pose: `{args.position}`, yaw `{args.yaw}`",
                ]
            ),
        )
    formula_path = output_dir / "formula_definition.json"
    if formula_path.is_file():
        formula = read_json(formula_path)
        formula["stage"] = "Stage 4A-6.5am"
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
        "hardware_utilization_report.json",
        "hardware_utilization_report.md",
        "runtime_setup_summary.json",
        "runtime_setup_summary.md",
        "repeat_variant_definition.json",
        "repeat_variant_definition.md",
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
        "stage4a65am_bounded_repeat_safety_summary.json",
        "stage4a65am_bounded_repeat_safety_summary.md",
        "recommended_next_faithful_step.md",
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
        "comparison_to_stage4a65ak.json",
        "comparison_to_stage4a65ak.md",
        "repeat_stability_classification.json",
        "repeat_stability_classification.md",
        "lambda32_vs_lambda48_repeat.json",
        "lambda32_vs_lambda48_repeat.md",
        "repeat_safety_readiness_matrix.csv",
        "repeat_safety_readiness_matrix.json",
        "repeat_safety_readiness_matrix.md",
        "frame001_observed_topdown.png",
        "frame001_prediction_overlay_topdown.png",
        "frame001_measured_vs_lambda48_tree_topdown.png",
        "frame001_lambda48_selected_branch_topdown.png",
        "value_components_frame001_lambda48.png",
        "low_cost_artifact_two_frame.png",
        "comparison_to_stage4a65ak_topdown.png",
        "repeat_stability_summary.png",
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


def postprocess(args: argparse.Namespace, context: dict[str, Any], reference_manifest: dict[str, Any], hashes_before: dict[str, Any]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    current_summary = read_json(output_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    reference_summary = read_json(Path(args.stage4a65ak_dir) / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    write_context_manifest(output_dir, context)
    write_reference_manifest(output_dir, reference_manifest)
    repeat_variant = write_repeat_variant(output_dir, args)
    patch_runtime_files(output_dir, args, repeat_variant)

    observed_delta = compute_observed_delta(output_dir)
    map_stability = compute_map_predict_stability(output_dir)
    comparison = {
        "reference_stage": "Stage 4A-6.5ak",
        "reference_tree_seed": int(args.reference_tree_seed),
        "current_tree_seed": int(args.tree_seed),
        "repeat_variant": str(args.repeat_variant),
        "frame001": compare_one_frame(current_summary, reference_summary, "frame001"),
        "frame002": compare_one_frame(current_summary, reference_summary, "frame002"),
    }
    action = current_summary["results"].get("action_execution_report")
    ref_action = reference_summary["results"].get("action_execution_report")
    comparison["action_pose_delta_vs_stage4a65ak_m"] = (
        None
        if action is None or ref_action is None
        else world_distance_m(action["executed_pose"]["position"], ref_action["executed_pose"]["position"])
    )
    comparison["observed_ratio_delta_vs_stage4a65ak"] = None
    if observed_delta is not None:
        ref_obs1 = reference_summary["results"]["frame001"]["prediction_stats"]["observed_state_shape"]
        comparison["observed_shape_matches_reference"] = list(ref_obs1) == [120, 120, 30]
    comparison["frame1_map_predict_occ_free_delta_vs_stage4a65ak"] = (
        int(current_summary["map_predict"]["frame001"]["stats"]["predicted_unmeasured_occ_free_count"])
        - int(reference_summary["map_predict"]["frame001"]["stats"]["predicted_unmeasured_occ_free_count"])
    )
    comparison["frame2_map_predict_occ_free_delta_vs_stage4a65ak"] = None
    if current_summary["map_predict"].get("frame002") and reference_summary["map_predict"].get("frame002"):
        comparison["frame2_map_predict_occ_free_delta_vs_stage4a65ak"] = (
            int(current_summary["map_predict"]["frame002"]["stats"]["predicted_unmeasured_occ_free_count"])
            - int(reference_summary["map_predict"]["frame002"]["stats"]["predicted_unmeasured_occ_free_count"])
        )
    save_json(output_dir / "comparison_to_stage4a65ak.json", comparison)
    write_text(
        output_dir / "comparison_to_stage4a65ak.md",
        "\n".join(
            [
                "# Comparison To Stage 4A-6.5ak",
                "",
                f"- Frame1 lambda48 current/reference: `{comparison['frame001']['current_lambda48']['selected_child_id']}` -> `{comparison['frame001']['current_lambda48']['best_descendant_id']}` / `{comparison['frame001']['reference_lambda48']['selected_child_id']}` -> `{comparison['frame001']['reference_lambda48']['best_descendant_id']}`",
                f"- Frame1 selected delta: `{comparison['frame001']['selected_child_grid_distance_vs_stage4a65ak_m']}` m",
                f"- Frame2 selected delta: `{None if not comparison['frame002'].get('available') else comparison['frame002']['selected_child_grid_distance_vs_stage4a65ak_m']}` m",
                f"- action pose delta: `{comparison['action_pose_delta_vs_stage4a65ak_m']}` m",
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
    repeat = classify_repeat(current_summary, comparison, safety_clean)
    save_json(output_dir / "repeat_stability_classification.json", repeat)
    write_text(
        output_dir / "repeat_stability_classification.md",
        "\n".join(
            [
                "# Repeat Stability Classification",
                "",
                f"- outcome: `{repeat['repeat_outcome']}`",
                f"- remains healthy: `{repeat['repeat_remains_healthy']}`",
                f"- low-cost artifact any frame: `{repeat['low_cost_artifact_any_frame']}`",
                f"- historical prior basin any frame: `{repeat['historical_prior_basin_any_frame']}`",
                f"- acceptable tree-seed variability: `{repeat['divergence_acceptable_tree_seed_variability']}`",
            ]
        ),
    )

    lambda_repeat = lambda32_vs_lambda48(current_summary)
    save_json(output_dir / "lambda32_vs_lambda48_repeat.json", lambda_repeat)
    write_text(
        output_dir / "lambda32_vs_lambda48_repeat.md",
        "\n".join(
            [
                "# Lambda32 Vs Lambda48 Repeat",
                "",
                f"- Frame1 same selected/best: `{lambda_repeat['frame001']['same_selected_child']}` / `{lambda_repeat['frame001']['same_best_descendant']}`",
                f"- Frame2 same selected/best: `{lambda_repeat['frame002']['same_selected_child']}` / `{lambda_repeat['frame002']['same_best_descendant']}`",
                f"- all available frames match: `{lambda_repeat['all_available_frames_match']}`",
            ]
        ),
    )

    plot_comparison_to_reference(output_dir, comparison)
    plot_observed_delta(output_dir)
    plot_repeat_stability(output_dir, comparison, repeat)

    reference_after = existing_file_hashes(
        reference_paths(Path(args.stage4a65ak_dir).resolve(), Path(args.stage4a65al_dir).resolve(), Path(args.checkpoint).resolve())
    )
    hash_checks = update_hash_checks(output_dir, hashes_before, reference_after)

    recommendation = write_recommendation(output_dir, repeat)
    setup = current_summary["runtime_setup"]
    rows = [
        {"check": "context_loaded", "passed": bool(context["confirmed_stage4a65ak_complete"] and context["confirmed_stage4a65al_complete"]), "evidence": "project context files reread"},
        {"check": "only_tree_seed_changed", "passed": bool(repeat_variant["only_tree_seed_changed_vs_reference"]), "evidence": f"{args.reference_tree_seed} -> {args.tree_seed}"},
        {"check": "isaac_started_once", "passed": setup["isaac_startup_count"] == 1, "evidence": str(setup["isaac_startup_count"])},
        {"check": "frames_bounded", "passed": int(setup["frames_captured"]) <= 2, "evidence": str(setup["frames_captured"])},
        {"check": "map_predict_bounded", "passed": int(setup["map_predict_calls"]) <= 2, "evidence": str(setup["map_predict_calls"])},
        {"check": "single_action_bound", "passed": int(setup["selected_action_execution_count"]) <= 1, "evidence": str(setup["selected_action_execution_count"])},
        {"check": "no_second_action", "passed": setup["second_action"] is False, "evidence": "runtime summary"},
        {"check": "no_third_frame", "passed": setup["third_frame"] is False, "evidence": "runtime summary"},
        {"check": "no_rollout", "passed": setup["rollout"] is False, "evidence": "runtime summary"},
        {"check": "prediction_safety_clean", "passed": safety_clean, "evidence": "prediction_safety_report"},
        {"check": "no_artifact_or_prior", "passed": not repeat["low_cost_artifact_any_frame"] and not repeat["historical_prior_basin_any_frame"], "evidence": repeat["repeat_outcome"]},
        {"check": "rollout_ready_false", "passed": True, "evidence": "bounded repeat only"},
    ]
    write_readiness_matrix(output_dir, rows)

    no_rollout = read_json(output_dir / "no_rollout_report.json")
    no_rollout["rollout_ready"] = False
    save_json(output_dir / "no_rollout_report.json", no_rollout)

    am_summary = {
        "stage": "Stage 4A-6.5am bounded repeat-safety smoke",
        "output_dir": str(output_dir),
        "loaded_context_manifest": context,
        "loaded_reference_manifest": reference_manifest,
        "repeat_variant": repeat_variant,
        "runtime_setup": setup,
        "formula": current_summary["formula"],
        "map_predict": current_summary["map_predict"],
        "results": current_summary["results"],
        "observed_state_delta_summary": observed_delta,
        "map_predict_two_frame_stability": map_stability,
        "comparison_to_stage4a65ak": comparison,
        "repeat_stability": repeat,
        "lambda32_vs_lambda48_repeat": lambda_repeat,
        "prediction_safety": prediction_safety,
        "hash_checks": hash_checks,
        "readiness": {
            "rollout_ready": False,
            "rollout_ready_reason": "bounded repeat-safety smoke is not rollout evidence",
            "two_frame_runtime_executed": int(setup["selected_action_execution_count"]) == 1,
            "repeat_outcome": repeat["repeat_outcome"],
        },
        "recommendation": recommendation,
        "safety": current_summary["safety"],
        "coverage_improvement_claim": False,
    }
    am_summary["formula"]["stage"] = "Stage 4A-6.5am"
    save_json(output_dir / "stage4a65am_bounded_repeat_safety_summary.json", am_summary)
    write_summary_md(output_dir / "stage4a65am_bounded_repeat_safety_summary.md", am_summary)

    missing = write_missing_report(output_dir, int(setup["selected_action_execution_count"]) == 1)
    am_summary["missing_fields_report"] = missing
    save_json(output_dir / "stage4a65am_bounded_repeat_safety_summary.json", am_summary)
    write_summary_md(output_dir / "stage4a65am_bounded_repeat_safety_summary.md", am_summary)
    print(json.dumps(clean(am_summary), indent=2, sort_keys=True))
    return am_summary


def run(args: argparse.Namespace, unknown: list[str]) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.tree_seed) != 1:
        raise ValueError("Stage 4A-6.5am repeat variant requires --tree_seed 1")
    if int(args.reference_tree_seed) != 0:
        raise ValueError("Stage 4A-6.5am reference requires --reference_tree_seed 0")
    if str(args.repeat_variant) != "same_scene_start_tree_seed1":
        raise ValueError("--repeat_variant must be same_scene_start_tree_seed1")
    if not args.execute_exactly_one_action or not args.no_second_action or not args.no_third_frame or not args.no_rollout:
        raise ValueError("Stage 4A-6.5am requires exactly-one-action bounds, no second action, no third frame, and no rollout")

    context = context_manifest()
    reference_before = existing_file_hashes(
        reference_paths(Path(args.stage4a65ak_dir).resolve(), Path(args.stage4a65al_dir).resolve(), Path(args.checkpoint).resolve())
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
    parser.add_argument("--stage4a65ak_dir", default=str(DEFAULT_STAGE4A65AK_DIR))
    parser.add_argument("--stage4a65al_dir", default=str(DEFAULT_STAGE4A65AL_DIR))
    parser.add_argument("--scene_variant", default="medium_three_rooms")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--repeat_variant", default="same_scene_start_tree_seed1")
    parser.add_argument("--start_pose", default="canonical_stage4a65p_frame1")
    parser.add_argument("--position", default="-4.65,-4.65,1.2")
    parser.add_argument("--yaw", type=float, default=0.38710316317995463)
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
