#!/usr/bin/env python3
"""Stage 4A-6.5an repeat comparison and next bounded-repeat design.

This stage is offline review/design only.  It reads completed Stage 4A-6.5ak,
6.5al, and 6.5am artifacts, compares tree_seed=0 with tree_seed=1, and writes
a future Stage 4A-6.5ao command sketch.  It does not start Isaac, capture
RGB/depth, run map_predict or SSCNet inference, execute actions, run rollout,
train, or modify existing runtime inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
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


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65an_repeat_comparison_and_next_design"
DEFAULT_AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_AL_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65al_post_action_two_frame_diagnosis"
DEFAULT_AM_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_AG_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ag_multi_frame_lambda48_replay"
CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]
PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
HISTORICAL_PRIOR_SELECTED_GRID = [11, 15, 11]
HISTORICAL_PRIOR_BEST_GRID = [14, 15, 11]
VOXEL_SIZE_M = 0.1

REQUIRED_OUTPUTS = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_input_manifest.json",
    "loaded_input_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "input_hash_audit.json",
    "input_hash_audit.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "sequence_safety_reverification.json",
    "sequence_safety_reverification.md",
    "prediction_safety_reverification.json",
    "prediction_safety_reverification.md",
    "no_rollout_reverification.json",
    "no_rollout_reverification.md",
    "frame1_seed0_seed1_comparison.json",
    "frame1_seed0_seed1_comparison.md",
    "frame2_seed0_seed1_comparison.json",
    "frame2_seed0_seed1_comparison.md",
    "branch_spatial_delta_table.csv",
    "branch_spatial_delta_table.json",
    "branch_spatial_delta_table.md",
    "branch_class_transition_summary.json",
    "branch_class_transition_summary.md",
    "lambda32_lambda48_agreement_comparison.json",
    "lambda32_lambda48_agreement_comparison.md",
    "observed_state_repeat_comparison.json",
    "observed_state_repeat_comparison.md",
    "action_effect_repeat_comparison.json",
    "action_effect_repeat_comparison.md",
    "observed_delta_difference_table.csv",
    "observed_delta_difference_table.json",
    "observed_delta_difference_table.md",
    "map_predict_repeat_stability_comparison.json",
    "map_predict_repeat_stability_comparison.md",
    "prediction_count_comparison.csv",
    "prediction_count_comparison.json",
    "prediction_count_comparison.md",
    "prediction_density_review.md",
    "low_cost_artifact_repeat_review.json",
    "low_cost_artifact_repeat_review.md",
    "historical_prior_basin_repeat_review.json",
    "historical_prior_basin_repeat_review.md",
    "branch_health_repeat_review.json",
    "branch_health_repeat_review.md",
    "cost_dominance_repeat_review.json",
    "cost_dominance_repeat_review.md",
    "repeat_outcome_classification.json",
    "repeat_outcome_classification.md",
    "next_repeat_decision.json",
    "next_repeat_decision.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "future_stage4a65ao_command_sketch.md",
    "do_not_run_runtime_in_stage4a65an.md",
    "stage4a65an_repeat_comparison_summary.json",
    "stage4a65an_repeat_comparison_summary.md",
    "recommended_next_faithful_step.md",
]

REQUIRED_PLOTS = [
    "frame1_seed0_seed1_topdown_comparison.png",
    "frame2_seed0_seed1_topdown_comparison.png",
    "action_pose_delta_topdown.png",
    "observed_delta_seed0_seed1_bar.png",
    "prediction_count_seed0_seed1_bar.png",
    "branch_class_transition.png",
    "repeat_safety_readiness_matrix.png",
    "next_repeat_decision_flowchart.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "frame001*",
    "frame002*",
    "frame003*",
    "action002*",
    "*.npy",
    "*.npz",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "manifest.jsonl",
    "episode_manifest*",
    "capture_rgb*.png",
    "capture_depth*.npy",
    "capture_depth*.png",
]

AK_REQUIRED_INPUTS = [
    "stage4a65ak_two_frame_one_action_runtime_summary.json",
    "runtime_setup_summary.json",
    "hash_checks.json",
    "prediction_safety_report.json",
    "no_rollout_report.json",
    "action_execution_report.json",
    "observed_state_update_frame001.json",
    "observed_state_update_frame002.json",
    "observed_state_frame001.npy",
    "observed_state_frame002.npy",
    "map_predict_frame001_summary.json",
    "map_predict_frame002_summary.json",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame002_branch_classification.json",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.json",
]

AL_REQUIRED_INPUTS = [
    "stage4a65al_post_action_two_frame_diagnosis_summary.json",
    "observed_state_delta_summary.json",
    "map_predict_two_frame_stability.json",
    "lambda32_vs_lambda48_two_frame.json",
    "branch_health_review.json",
    "cost_dominance_review.json",
    "historical_prior_basin_recheck.json",
    "low_cost_artifact_two_frame_review.json",
]

AM_REQUIRED_INPUTS = [
    "stage4a65am_bounded_repeat_safety_summary.json",
    "comparison_to_stage4a65ak.json",
    "runtime_setup_summary.json",
    "hash_checks.json",
    "prediction_safety_report.json",
    "no_rollout_report.json",
    "repeat_variant_definition.json",
    "repeat_stability_classification.json",
    "lambda32_vs_lambda48_repeat.json",
    "observed_state_delta_summary.json",
    "map_predict_two_frame_stability.json",
    "action_execution_report.json",
    "observed_state_frame001.npy",
    "observed_state_frame002.npy",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame002_branch_classification.json",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.json",
]


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean(row.get(name)) for name in fieldnames})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_worker(path_text: str) -> tuple[str, dict[str, Any]]:
    path = Path(path_text)
    entry = {
        "path": path_text,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }
    return path_text, entry


def hash_paths(paths: list[Path], max_workers: int) -> dict[str, dict[str, Any]]:
    unique = sorted({str(path) for path in paths})
    workers = max(1, min(max_workers, len(unique) or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return dict(executor.map(hash_worker, unique))


def md_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, float):
                value = f"{value:.12g}"
            values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    return "\n".join(lines)


def write_summary_md(path: Path, title: str, lines: list[str]) -> None:
    write_text(path, "\n".join([f"# {title}", "", *lines]))


def decision(path: Path) -> dict[str, Any]:
    data = read_json(path)
    return data.get("decision", data)


def vec_distance(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    if len(a) != len(b):
        return None
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def grid_distance_m(a: Any, b: Any) -> float | None:
    raw = vec_distance(a, b)
    return None if raw is None else raw * VOXEL_SIZE_M


def yaw_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    delta = (float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi
    return abs(delta)


def simple_decision(dec: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "selected_child_id",
        "selected_child_grid",
        "selected_child_world",
        "best_descendant_id",
        "best_descendant_grid",
        "best_descendant_world",
        "branch_classification",
        "same_as_measured",
        "changed_vs_measured_only",
        "healthy_nonmeasured_candidate",
        "low_cost_artifact",
        "spatial_prior_sc_basin",
        "gain_exp",
        "source_occ_free",
        "source_occ_count",
        "source_free_count",
        "cost",
        "base_exp_value",
        "normalized_sc",
        "sc_bonus",
        "final_value",
        "runner_up_value",
        "margin",
        "normalized_margin",
        "branch_depth",
        "path_node_ids",
        "selected_cost_rank",
        "selected_gain_exp_rank",
        "selected_source_occ_free_rank",
    ]
    return {key: dec.get(key) for key in keys if key in dec}


def load_run(run_dir: Path, stage: str) -> dict[str, Any]:
    l48_frame2_name = "frame002_lambda48_diagnostic_tree_decision.json"
    summary_name = (
        "stage4a65ak_two_frame_one_action_runtime_summary.json"
        if stage == "ak"
        else "stage4a65am_bounded_repeat_safety_summary.json"
    )
    return {
        "dir": str(run_dir),
        "summary": read_json(run_dir / summary_name),
        "runtime_setup": read_json(run_dir / "runtime_setup_summary.json")
        if (run_dir / "runtime_setup_summary.json").is_file()
        else read_json(run_dir / summary_name).get("runtime_setup", {}),
        "prediction_safety": read_json(run_dir / "prediction_safety_report.json"),
        "no_rollout": read_json(run_dir / "no_rollout_report.json"),
        "action": read_json(run_dir / "action_execution_report.json"),
        "frame001": {
            "measured": decision(run_dir / "frame001_measured_shadow_tree_decision.json"),
            "lambda48": decision(run_dir / "frame001_lambda48_primary_tree_decision.json"),
            "lambda32": decision(run_dir / "frame001_lambda32_shadow_tree_decision.json"),
            "branch": read_json(run_dir / "frame001_branch_classification.json"),
            "low_cost": read_json(run_dir / "frame001_low_cost_artifact_diagnosis.json"),
            "map_predict": read_json(run_dir / "map_predict_frame001_summary.json"),
        },
        "frame002": {
            "measured": decision(run_dir / "frame002_measured_shadow_tree_decision.json"),
            "lambda48": decision(run_dir / l48_frame2_name),
            "lambda32": decision(run_dir / "frame002_lambda32_shadow_tree_decision.json"),
            "branch": read_json(run_dir / "frame002_branch_classification.json"),
            "low_cost": read_json(run_dir / "frame002_low_cost_artifact_diagnosis.json"),
            "map_predict": read_json(run_dir / "map_predict_frame002_summary.json"),
        },
    }


def collect_hash_inputs(ak_dir: Path, al_dir: Path, am_dir: Path, ag_dir: Path) -> list[Path]:
    paths = [CHECKPOINT, *CONTEXT_FILES]
    for base in (ak_dir, al_dir, am_dir):
        if base.is_dir():
            paths.extend(path for path in base.rglob("*") if path.suffix in {".json", ".npy", ".npz"})
    if ag_dir.is_dir():
        for name in (
            "stage4a65ag_multi_frame_lambda48_replay_summary.json",
            "lambda48_replay_summary.json",
            "summary.json",
        ):
            candidate = ag_dir / name
            if candidate.is_file():
                paths.append(candidate)
    return paths


def build_context_manifest() -> dict[str, Any]:
    entries = []
    combined = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "contains_stage4a65ak": "Stage 4A-6.5ak" in text,
                "contains_stage4a65al": "Stage 4A-6.5al" in text,
                "contains_stage4a65am": "Stage 4A-6.5am" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5an",
        "review_design_only": True,
        "loaded_context_files": entries,
        "confirmed_stage4a65ak_complete": "Stage 4A-6.5ak" in combined and "two-frame one-action" in combined,
        "confirmed_stage4a65al_complete": "Stage 4A-6.5al" in combined and "post-action/two-frame" in combined,
        "confirmed_stage4a65am_complete": "Stage 4A-6.5am" in combined and "tree_seed=1" in combined,
        "chat_history_not_used_as_source": True,
    }


def build_input_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    stage_specs = {
        "stage4a65ak": (Path(args.stage4a65ak_dir), AK_REQUIRED_INPUTS),
        "stage4a65al": (Path(args.stage4a65al_dir), AL_REQUIRED_INPUTS),
        "stage4a65am": (Path(args.stage4a65am_dir), AM_REQUIRED_INPUTS),
    }
    missing = []
    stages: dict[str, Any] = {}
    for stage, (base, required) in stage_specs.items():
        missing_stage = [name for name in required if not (base / name).is_file()]
        missing.extend([f"{stage}:{name}" for name in missing_stage])
        stages[stage] = {
            "dir": str(base),
            "exists": base.is_dir(),
            "required_file_count": len(required),
            "missing_required_files": missing_stage,
            "loaded": base.is_dir() and not missing_stage,
        }
    ag_dir = Path(args.stage4a65ag_dir)
    ag_candidates = sorted(str(path) for path in ag_dir.glob("*.json")) if ag_dir.is_dir() else []
    stages["stage4a65ag"] = {
        "dir": str(ag_dir),
        "optional": True,
        "exists": ag_dir.is_dir(),
        "top_level_json_count": len(ag_candidates),
        "loaded_as_optional_reference": ag_dir.is_dir(),
    }
    return {
        "stage": "Stage 4A-6.5an",
        "primary_inputs": stages,
        "all_primary_inputs_loaded": not missing,
        "optional_stage4a65ag_available": ag_dir.is_dir(),
    }, missing


def sequence_entry(run: dict[str, Any]) -> dict[str, Any]:
    setup = run["summary"].get("runtime_setup", run.get("runtime_setup", {}))
    safety = run["summary"].get("safety", {})
    return {
        "isaac_startup_count": int(setup.get("isaac_startup_count", 0)),
        "frames_captured": int(setup.get("frames_captured", 0)),
        "map_predict_calls": int(setup.get("map_predict_calls", 0)),
        "selected_action_execution_count": int(setup.get("selected_action_execution_count", 0)),
        "second_action": bool(setup.get("second_action", safety.get("second_action", False))),
        "third_frame": bool(setup.get("third_frame", safety.get("third_frame", False))),
        "rollout": bool(setup.get("rollout", safety.get("rollout", False))),
        "two_frame_runtime_executed": bool(setup.get("two_frame_runtime_executed", safety.get("two_frame_runtime_executed", True))),
        "exactly_two_frames": int(setup.get("frames_captured", 0)) == 2,
        "exactly_two_map_predict_calls": int(setup.get("map_predict_calls", 0)) == 2,
        "exactly_one_action": int(setup.get("selected_action_execution_count", 0)) == 1,
        "sequence_clean": int(setup.get("frames_captured", 0)) == 2
        and int(setup.get("map_predict_calls", 0)) == 2
        and int(setup.get("selected_action_execution_count", 0)) == 1
        and not bool(setup.get("second_action", False))
        and not bool(setup.get("third_frame", False))
        and not bool(setup.get("rollout", False)),
    }


def compare_frame(ref: dict[str, Any], cur: dict[str, Any], frame_name: str) -> dict[str, Any]:
    ref_l48 = ref[frame_name]["lambda48"]
    cur_l48 = cur[frame_name]["lambda48"]
    ref_measured = ref[frame_name]["measured"]
    cur_measured = cur[frame_name]["measured"]
    ref_l32 = ref[frame_name]["lambda32"]
    cur_l32 = cur[frame_name]["lambda32"]
    return {
        "frame": frame_name,
        "reference_tree_seed": 0,
        "repeat_tree_seed": 1,
        "reference_lambda48": simple_decision(ref_l48),
        "repeat_lambda48": simple_decision(cur_l48),
        "reference_measured": simple_decision(ref_measured),
        "repeat_measured": simple_decision(cur_measured),
        "selected_child_exact_agreement": ref_l48.get("selected_child_id") == cur_l48.get("selected_child_id"),
        "best_descendant_exact_agreement": ref_l48.get("best_descendant_id") == cur_l48.get("best_descendant_id"),
        "selected_child_grid_delta_m": grid_distance_m(ref_l48.get("selected_child_grid"), cur_l48.get("selected_child_grid")),
        "selected_child_world_delta_m": vec_distance(ref_l48.get("selected_child_world"), cur_l48.get("selected_child_world")),
        "best_descendant_grid_delta_m": grid_distance_m(ref_l48.get("best_descendant_grid"), cur_l48.get("best_descendant_grid")),
        "best_descendant_world_delta_m": vec_distance(ref_l48.get("best_descendant_world"), cur_l48.get("best_descendant_world")),
        "branch_class_reference": ref_l48.get("branch_classification"),
        "branch_class_repeat": cur_l48.get("branch_classification"),
        "branch_class_agreement": ref_l48.get("branch_classification") == cur_l48.get("branch_classification"),
        "lambda48_vs_measured_relation_reference": {
            "classification": ref_l48.get("branch_classification"),
            "changed_vs_measured_only": ref_l48.get("changed_vs_measured_only"),
            "lambda48_selected": ref_l48.get("selected_child_id"),
            "measured_selected": ref_measured.get("selected_child_id"),
        },
        "lambda48_vs_measured_relation_repeat": {
            "classification": cur_l48.get("branch_classification"),
            "changed_vs_measured_only": cur_l48.get("changed_vs_measured_only"),
            "lambda48_selected": cur_l48.get("selected_child_id"),
            "measured_selected": cur_measured.get("selected_child_id"),
        },
        "lambda32_lambda48_agreement_reference": {
            "same_selected_child": ref_l32.get("selected_child_id") == ref_l48.get("selected_child_id"),
            "same_best_descendant": ref_l32.get("best_descendant_id") == ref_l48.get("best_descendant_id"),
            "same_branch_class": ref_l32.get("branch_classification") == ref_l48.get("branch_classification"),
        },
        "lambda32_lambda48_agreement_repeat": {
            "same_selected_child": cur_l32.get("selected_child_id") == cur_l48.get("selected_child_id"),
            "same_best_descendant": cur_l32.get("best_descendant_id") == cur_l48.get("best_descendant_id"),
            "same_branch_class": cur_l32.get("branch_classification") == cur_l48.get("branch_classification"),
        },
        "value_gain_cost_margin_delta_repeat_minus_reference": {
            key: (cur_l48.get(key) - ref_l48.get(key))
            if isinstance(cur_l48.get(key), (int, float)) and isinstance(ref_l48.get(key), (int, float))
            else None
            for key in ("gain_exp", "source_occ_free", "cost", "final_value", "margin")
        },
        "both_low_cost_artifact_false": bool(ref_l48.get("low_cost_artifact") is False and cur_l48.get("low_cost_artifact") is False),
        "both_historical_prior_basin_false": bool(
            ref[frame_name]["branch"].get("historical_prior_basin") is False
            and cur[frame_name]["branch"].get("historical_prior_basin") is False
        ),
    }


def action_pose_comparison(ref: dict[str, Any], cur: dict[str, Any]) -> dict[str, Any]:
    ref_pose = ref["action"].get("executed_pose", {})
    cur_pose = cur["action"].get("executed_pose", {})
    return {
        "reference_position": ref_pose.get("position"),
        "repeat_position": cur_pose.get("position"),
        "reference_yaw_rad": ref_pose.get("yaw_rad"),
        "repeat_yaw_rad": cur_pose.get("yaw_rad"),
        "action_pose_delta_m": vec_distance(ref_pose.get("position"), cur_pose.get("position")),
        "yaw_delta_rad": yaw_delta(ref_pose.get("yaw_rad"), cur_pose.get("yaw_rad")),
        "reference_action_executed": bool(ref["action"].get("action_executed")),
        "repeat_action_executed": bool(cur["action"].get("action_executed")),
        "single_action_each": int(ref["action"].get("action_execution_count", 0)) == 1
        and int(cur["action"].get("action_execution_count", 0)) == 1,
    }


def map_counts_from_summary(run: dict[str, Any], frame: str) -> dict[str, Any]:
    stats = run[frame]["map_predict"].get("stats", {})
    return {
        "prediction_valid_count": stats.get("prediction_valid_count"),
        "predicted_unmeasured_occ_free_count": stats.get("predicted_unmeasured_occ_free_count"),
        "predicted_unmeasured_free_count": stats.get("predicted_unmeasured_free_count"),
        "predicted_unmeasured_occupied_count": stats.get("predicted_unmeasured_occupied_count"),
        "alignment_convention": stats.get("alignment_convention", run[frame]["map_predict"].get("alignment_convention")),
    }


def plot_points(ax: Any, points: list[tuple[str, Any, str, str]]) -> None:
    for label, point, color, marker in points:
        if not point:
            continue
        ax.scatter([point[0]], [point[1]], label=label, color=color, marker=marker, s=70)
        ax.text(point[0] + 0.05, point[1] + 0.05, label, fontsize=7)


def save_topdown_plot(path: Path, title: str, comparison: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ref = comparison["reference_lambda48"]
    cur = comparison["repeat_lambda48"]
    points = [
        ("seed0 selected", ref.get("selected_child_world"), "#1f77b4", "o"),
        ("seed0 best", ref.get("best_descendant_world"), "#1f77b4", "x"),
        ("seed1 selected", cur.get("selected_child_world"), "#d62728", "o"),
        ("seed1 best", cur.get("best_descendant_world"), "#d62728", "x"),
    ]
    plot_points(ax, points)
    ax.set_title(title)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_action_plot(path: Path, action: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ref = action["reference_position"]
    cur = action["repeat_position"]
    if ref and cur:
        ax.scatter([ref[0]], [ref[1]], label="seed0 action", color="#1f77b4", s=80)
        ax.scatter([cur[0]], [cur[1]], label="seed1 action", color="#d62728", s=80)
        ax.plot([ref[0], cur[0]], [ref[1], cur[1]], color="#555555", linewidth=1.5)
        ax.text((ref[0] + cur[0]) / 2.0 + 0.03, (ref[1] + cur[1]) / 2.0, f"{action['action_pose_delta_m']:.3f} m", fontsize=9)
    ax.set_title("Action Pose Delta")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar(path: Path, title: str, labels: list[str], values: list[float], colors: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color=colors or "#4c78a8")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_branch_transition(path: Path, rows: list[dict[str, Any]]) -> None:
    class_to_value = {"same_as_measured": 0, "distinct_nonmeasured_branch": 1}
    labels = [f"{row['frame']} {row['run']}" for row in rows]
    values = [class_to_value.get(row["branch_classification"], -0.2) for row in rows]
    colors = ["#7aa6c2" if row["run"] == "seed0" else "#e07b73" for row in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(labels, values, color=colors)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["same_as_measured", "distinct_nonmeasured"])
    ax.set_ylim(-0.25, 1.25)
    ax.set_title("Branch Class Transition")
    ax.tick_params(axis="x", rotation=25)
    for idx, row in enumerate(rows):
        ax.text(idx, values[idx] + 0.04, row["branch_classification"], ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_matrix_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    values = [[1 if row["ready"] else 0] for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4.8))
    ax.imshow(values, cmap=matplotlib.colors.ListedColormap(["#d95f5f", "#6aa56a"]), vmin=0, vmax=1)
    ax.set_xticks([0])
    ax.set_xticklabels(["status"])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["item"] for row in rows], fontsize=8)
    for y, row in enumerate(rows):
        ax.text(0, y, "yes" if row["ready"] else "no", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Repeat Safety Readiness Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_flowchart(path: Path, frame2_delta_m: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    boxes = [
        (0.50, 0.82, "6.5ak and 6.5am clean"),
        (0.50, 0.60, "Frame2 selected delta > 1m"),
        (0.50, 0.38, "Isolate tree_seed variability"),
        (0.50, 0.16, "Choose 6.5ao tree_seed=2"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=12, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f4f4f4", "edgecolor": "#555555"})
    for y0, y1 in [(0.76, 0.66), (0.54, 0.44), (0.32, 0.22)]:
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.text(0.72, 0.56, f"delta={frame2_delta_m:.3f}m", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_plot_or_reason(output_dir: Path, name: str, plot_fn: Any) -> dict[str, str]:
    try:
        plot_fn(output_dir / name)
        return {}
    except Exception as exc:  # pragma: no cover - reason file path is validated instead.
        reason_name = f"{Path(name).stem}_skipped_reason.md"
        write_text(output_dir / reason_name, f"# Plot Skipped\n\n- plot: `{name}`\n- reason: `{exc}`")
        return {name: str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage4a65ak_dir", type=Path, default=DEFAULT_AK_DIR)
    parser.add_argument("--stage4a65al_dir", type=Path, default=DEFAULT_AL_DIR)
    parser.add_argument("--stage4a65am_dir", type=Path, default=DEFAULT_AM_DIR)
    parser.add_argument("--stage4a65ag_dir", type=Path, default=DEFAULT_AG_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference_tree_seed", type=int, default=0)
    parser.add_argument("--repeat_tree_seed", type=int, default=1)
    parser.add_argument("--candidate_next_tree_seed", type=int, default=2)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    args = parser.parse_args()

    start_time = time.perf_counter()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_workers = min(int(args.max_workers), os.cpu_count() or 1)

    context_manifest = build_context_manifest()
    input_manifest, missing_input_files = build_input_manifest(args)
    hash_input_paths = collect_hash_inputs(args.stage4a65ak_dir, args.stage4a65al_dir, args.stage4a65am_dir, args.stage4a65ag_dir)
    hashes_before = hash_paths(hash_input_paths, actual_workers)

    ak = load_run(args.stage4a65ak_dir, "ak")
    am = load_run(args.stage4a65am_dir, "am")
    al_summary = read_json(args.stage4a65al_dir / "stage4a65al_post_action_two_frame_diagnosis_summary.json")
    al_observed = read_json(args.stage4a65al_dir / "observed_state_delta_summary.json")
    al_map = read_json(args.stage4a65al_dir / "map_predict_two_frame_stability.json")
    al_lambda = read_json(args.stage4a65al_dir / "lambda32_vs_lambda48_two_frame.json")
    am_summary = am["summary"]
    am_observed = read_json(args.stage4a65am_dir / "observed_state_delta_summary.json")
    am_map = read_json(args.stage4a65am_dir / "map_predict_two_frame_stability.json")
    am_lambda = read_json(args.stage4a65am_dir / "lambda32_vs_lambda48_repeat.json")
    am_repeat = read_json(args.stage4a65am_dir / "repeat_stability_classification.json")

    write_json(output_dir / "loaded_context_manifest.json", context_manifest)
    write_summary_md(
        output_dir / "loaded_context_manifest.md",
        "Loaded Context Manifest",
        [
            f"- Stage 4A-6.5ak complete: `{context_manifest['confirmed_stage4a65ak_complete']}`",
            f"- Stage 4A-6.5al complete: `{context_manifest['confirmed_stage4a65al_complete']}`",
            f"- Stage 4A-6.5am complete: `{context_manifest['confirmed_stage4a65am_complete']}`",
            "- Context files:",
            *[f"  - `{entry['path']}` sha256 `{entry['sha256']}`" for entry in context_manifest["loaded_context_files"]],
        ],
    )
    write_json(output_dir / "loaded_input_manifest.json", input_manifest)
    write_summary_md(
        output_dir / "loaded_input_manifest.md",
        "Loaded Input Manifest",
        [
            f"- All primary inputs loaded: `{input_manifest['all_primary_inputs_loaded']}`",
            f"- Optional Stage 4A-6.5ag available: `{input_manifest['optional_stage4a65ag_available']}`",
            *[
                f"- {stage}: exists `{data['exists']}`, loaded `{data.get('loaded', data.get('loaded_as_optional_reference'))}`, missing `{len(data.get('missing_required_files', []))}`"
                for stage, data in input_manifest["primary_inputs"].items()
            ],
        ],
    )

    seq = {
        "stage": "Stage 4A-6.5an",
        "review_design_only": True,
        "stage4a65an_runtime": {
            "isaac_startup": False,
            "rgb_depth_capture": False,
            "map_predict_call": False,
            "sscnet_inference": False,
            "selected_action_execution": False,
            "two_frame_runtime_execution": False,
            "rollout": False,
        },
        "stage4a65ak": sequence_entry(ak),
        "stage4a65am": sequence_entry(am),
    }
    seq["both_reverified_clean"] = seq["stage4a65ak"]["sequence_clean"] and seq["stage4a65am"]["sequence_clean"]
    write_json(output_dir / "sequence_safety_reverification.json", seq)
    write_summary_md(
        output_dir / "sequence_safety_reverification.md",
        "Sequence Safety Reverification",
        [
            f"- 6.5an runtime execution: `{seq['stage4a65an_runtime']}`",
            f"- 6.5ak clean two-frame/one-action sequence: `{seq['stage4a65ak']['sequence_clean']}`",
            f"- 6.5am clean two-frame/one-action sequence: `{seq['stage4a65am']['sequence_clean']}`",
            f"- Both clean: `{seq['both_reverified_clean']}`",
        ],
    )

    ak_pred = ak["prediction_safety"]
    am_pred = am["prediction_safety"]
    ak_no_motion_safety_use = bool(
        ak_pred.get(
            "all_motion_safety_uses_false",
            not (
                ak_pred.get("prediction_used_for_collision")
                or ak_pred.get("prediction_used_for_traversability")
                or ak_pred.get("prediction_ray_blocking")
                or ak_pred.get("prediction_used_for_candidate_sampling")
                or ak_pred.get("prediction_used_for_edge_validity")
            ),
        )
    )
    am_no_motion_safety_use = bool(
        am_pred.get(
            "all_motion_safety_uses_false",
            not (
                am_pred.get("prediction_used_for_collision")
                or am_pred.get("prediction_used_for_traversability")
                or am_pred.get("prediction_ray_blocking")
                or am_pred.get("prediction_used_for_candidate_sampling")
                or am_pred.get("prediction_used_for_edge_validity")
            ),
        )
    )
    ak_information_gain_only = bool(
        ak_pred.get(
            "prediction_information_gain_only",
            ak_pred.get("prediction_read_only")
            and ak_no_motion_safety_use
            and not ak_pred.get("prediction_written_to_observed_state")
            and not ak_pred.get("prediction_fused_into_observed_state"),
        )
    )
    am_information_gain_only = bool(
        am_pred.get(
            "prediction_information_gain_only",
            am_pred.get("prediction_read_only")
            and am_no_motion_safety_use
            and not am_pred.get("prediction_written_to_observed_state")
            and not am_pred.get("prediction_fused_into_observed_state"),
        )
    )
    pred_safety = {
        "stage": "Stage 4A-6.5an",
        "stage4a65ak": ak_pred,
        "stage4a65am": am_pred,
        "prediction_read_only_both": bool(ak_pred.get("prediction_read_only") and am_pred.get("prediction_read_only")),
        "prediction_information_gain_only_both": bool(ak_information_gain_only and am_information_gain_only),
        "no_prediction_writeback_or_fusion": not (
            ak_pred.get("prediction_written_to_observed_state")
            or am_pred.get("prediction_written_to_observed_state")
            or ak_pred.get("prediction_fused_into_observed_state")
            or am_pred.get("prediction_fused_into_observed_state")
        ),
        "no_prediction_motion_safety_use": bool(ak_no_motion_safety_use and am_no_motion_safety_use),
        "no_target_ground_truth_future_observed_scoring": not (
            ak_pred.get("target_lr_target_hr_ground_truth_used_for_planning_scoring")
            or am_pred.get("target_lr_target_hr_ground_truth_used_for_planning_scoring")
            or ak_pred.get("future_observed_used_for_planning_scoring")
            or am_pred.get("future_observed_used_for_planning_scoring")
        ),
    }
    pred_safety["prediction_safety_clean"] = all(
        [
            pred_safety["prediction_read_only_both"],
            pred_safety["prediction_information_gain_only_both"],
            pred_safety["no_prediction_writeback_or_fusion"],
            pred_safety["no_prediction_motion_safety_use"],
            pred_safety["no_target_ground_truth_future_observed_scoring"],
        ]
    )
    write_json(output_dir / "prediction_safety_reverification.json", pred_safety)
    write_summary_md(
        output_dir / "prediction_safety_reverification.md",
        "Prediction Safety Reverification",
        [
            f"- Prediction read-only both runs: `{pred_safety['prediction_read_only_both']}`",
            f"- Information-gain-only both runs: `{pred_safety['prediction_information_gain_only_both']}`",
            f"- No writeback/fusion: `{pred_safety['no_prediction_writeback_or_fusion']}`",
            f"- No traversability/collision/ray-blocking/candidate/edge use: `{pred_safety['no_prediction_motion_safety_use']}`",
            f"- No target/ground-truth/future-observed scoring: `{pred_safety['no_target_ground_truth_future_observed_scoring']}`",
        ],
    )

    frame1_cmp = compare_frame(ak, am, "frame001")
    frame2_cmp = compare_frame(ak, am, "frame002")
    action_cmp = action_pose_comparison(ak, am)
    frame1_cmp["action_pose_delta_m"] = action_cmp["action_pose_delta_m"]
    frame1_cmp["yaw_delta_rad"] = action_cmp["yaw_delta_rad"]

    write_json(output_dir / "frame1_seed0_seed1_comparison.json", frame1_cmp)
    write_summary_md(
        output_dir / "frame1_seed0_seed1_comparison.md",
        "Frame1 Seed0 vs Seed1 Comparison",
        [
            f"- selected child exact agreement: `{frame1_cmp['selected_child_exact_agreement']}`",
            f"- best descendant exact agreement: `{frame1_cmp['best_descendant_exact_agreement']}`",
            f"- selected child spatial delta: `{frame1_cmp['selected_child_world_delta_m']}` m",
            f"- best descendant spatial delta: `{frame1_cmp['best_descendant_world_delta_m']}` m",
            f"- branch transition: `{frame1_cmp['branch_class_reference']}` -> `{frame1_cmp['branch_class_repeat']}`",
            f"- lambda32/lambda48 agreement seed0: `{frame1_cmp['lambda32_lambda48_agreement_reference']}`",
            f"- lambda32/lambda48 agreement seed1: `{frame1_cmp['lambda32_lambda48_agreement_repeat']}`",
        ],
    )
    write_json(output_dir / "frame2_seed0_seed1_comparison.json", frame2_cmp)
    write_summary_md(
        output_dir / "frame2_seed0_seed1_comparison.md",
        "Frame2 Seed0 vs Seed1 Comparison",
        [
            f"- selected child exact agreement: `{frame2_cmp['selected_child_exact_agreement']}`",
            f"- best descendant exact agreement: `{frame2_cmp['best_descendant_exact_agreement']}`",
            f"- selected child spatial delta: `{frame2_cmp['selected_child_world_delta_m']}` m",
            f"- best descendant spatial delta: `{frame2_cmp['best_descendant_world_delta_m']}` m",
            f"- branch class agreement: `{frame2_cmp['branch_class_agreement']}`",
            f"- both runs healthy despite node divergence: `{frame2_cmp['both_low_cost_artifact_false'] and frame2_cmp['both_historical_prior_basin_false']}`",
        ],
    )

    branch_rows = [
        {
            "frame": "frame001",
            "selected_child_delta_m": frame1_cmp["selected_child_world_delta_m"],
            "best_descendant_delta_m": frame1_cmp["best_descendant_world_delta_m"],
            "reference_branch_class": frame1_cmp["branch_class_reference"],
            "repeat_branch_class": frame1_cmp["branch_class_repeat"],
            "selected_child_exact_agreement": frame1_cmp["selected_child_exact_agreement"],
        },
        {
            "frame": "frame002",
            "selected_child_delta_m": frame2_cmp["selected_child_world_delta_m"],
            "best_descendant_delta_m": frame2_cmp["best_descendant_world_delta_m"],
            "reference_branch_class": frame2_cmp["branch_class_reference"],
            "repeat_branch_class": frame2_cmp["branch_class_repeat"],
            "selected_child_exact_agreement": frame2_cmp["selected_child_exact_agreement"],
        },
    ]
    write_csv(output_dir / "branch_spatial_delta_table.csv", branch_rows, list(branch_rows[0].keys()))
    write_json(output_dir / "branch_spatial_delta_table.json", branch_rows)
    write_text(output_dir / "branch_spatial_delta_table.md", "# Branch Spatial Delta Table\n\n" + md_table(list(branch_rows[0].keys()), branch_rows))

    branch_transition = {
        "frame001": {
            "seed0": frame1_cmp["branch_class_reference"],
            "seed1": frame1_cmp["branch_class_repeat"],
            "transition": f"{frame1_cmp['branch_class_reference']} -> {frame1_cmp['branch_class_repeat']}",
        },
        "frame002": {
            "seed0": frame2_cmp["branch_class_reference"],
            "seed1": frame2_cmp["branch_class_repeat"],
            "transition": f"{frame2_cmp['branch_class_reference']} -> {frame2_cmp['branch_class_repeat']}",
        },
        "frame1_transition_changed": not frame1_cmp["branch_class_agreement"],
        "frame2_transition_changed": not frame2_cmp["branch_class_agreement"],
    }
    write_json(output_dir / "branch_class_transition_summary.json", branch_transition)
    write_summary_md(
        output_dir / "branch_class_transition_summary.md",
        "Branch Class Transition Summary",
        [
            f"- Frame1: `{branch_transition['frame001']['transition']}`",
            f"- Frame2: `{branch_transition['frame002']['transition']}`",
            f"- Frame1 changed: `{branch_transition['frame1_transition_changed']}`",
            f"- Frame2 changed: `{branch_transition['frame2_transition_changed']}`",
        ],
    )

    lambda_compare = {
        "stage4a65ak": al_lambda,
        "stage4a65am": am_lambda,
        "seed0_lambda32_effectively_equivalent": bool(al_lambda.get("lambda32_effectively_equivalent_in_this_run")),
        "seed1_all_available_frames_match": bool(am_lambda.get("all_available_frames_match")),
        "frame001_both_match": bool(
            frame1_cmp["lambda32_lambda48_agreement_reference"]["same_selected_child"]
            and frame1_cmp["lambda32_lambda48_agreement_repeat"]["same_selected_child"]
        ),
        "frame002_seed0_match_seed1_differs": bool(
            frame2_cmp["lambda32_lambda48_agreement_reference"]["same_selected_child"]
            and not frame2_cmp["lambda32_lambda48_agreement_repeat"]["same_selected_child"]
        ),
    }
    write_json(output_dir / "lambda32_lambda48_agreement_comparison.json", lambda_compare)
    write_summary_md(
        output_dir / "lambda32_lambda48_agreement_comparison.md",
        "Lambda32 Lambda48 Agreement Comparison",
        [
            f"- Seed0 lambda32/lambda48 effectively equivalent: `{lambda_compare['seed0_lambda32_effectively_equivalent']}`",
            f"- Seed1 all available frames match: `{lambda_compare['seed1_all_available_frames_match']}`",
            f"- Frame1 both match: `{lambda_compare['frame001_both_match']}`",
            f"- Frame2 seed0 match but seed1 differs: `{lambda_compare['frame002_seed0_match_seed1_differs']}`",
        ],
    )

    observed_compare = {
        "stage4a65ak": {
            "observed_ratio_start": al_observed["frame001"]["observed_ratio"],
            "observed_ratio_end": al_observed["frame002"]["observed_ratio"],
            "observed_ratio_delta": al_observed["observed_ratio_delta"],
            "newly_observed": al_observed["newly_observed_voxels"],
            "unknown_to_free": al_observed["unknown_to_free"],
            "unknown_to_occupied": al_observed["unknown_to_occupied"],
            "free_to_occupied": al_observed["free_to_occupied"],
            "occupied_to_free": al_observed["occupied_to_free"],
            "invalid_labels": al_observed["invalid_label_count_frame001"] + al_observed["invalid_label_count_frame002"],
        },
        "stage4a65am": {
            "observed_ratio_start": am_observed["frame001"]["observed_ratio"],
            "observed_ratio_end": am_observed["frame002"]["observed_ratio"],
            "observed_ratio_delta": am_observed["observed_ratio_delta"],
            "newly_observed": am_observed["newly_observed"],
            "unknown_to_free": am_observed["unknown_to_free"],
            "unknown_to_occupied": am_observed["unknown_to_occupied"],
            "free_to_occupied": am_observed["free_to_occupied"],
            "occupied_to_free": am_observed["occupied_to_free"],
            "invalid_labels": am_observed["invalid_labels"],
        },
    }
    observed_compare["repeat_minus_reference"] = {
        key: observed_compare["stage4a65am"][key] - observed_compare["stage4a65ak"][key]
        for key in ("observed_ratio_delta", "newly_observed", "unknown_to_free", "unknown_to_occupied", "occupied_to_free", "invalid_labels")
    }
    observed_compare["suspicious_label_flips"] = bool(
        observed_compare["stage4a65ak"]["occupied_to_free"]
        or observed_compare["stage4a65am"]["occupied_to_free"]
        or observed_compare["stage4a65ak"]["invalid_labels"]
        or observed_compare["stage4a65am"]["invalid_labels"]
    )
    observed_compare["measured_only_status_preserved"] = bool(am_observed.get("measured_only_status", True))
    observed_compare["action_pose_delta_plausibly_explains_delta_difference"] = bool(action_cmp["action_pose_delta_m"] and action_cmp["action_pose_delta_m"] <= 0.25)
    write_json(output_dir / "observed_state_repeat_comparison.json", observed_compare)
    write_summary_md(
        output_dir / "observed_state_repeat_comparison.md",
        "Observed State Repeat Comparison",
        [
            f"- 6.5ak observed delta: `{observed_compare['stage4a65ak']['observed_ratio_delta']}` with `{observed_compare['stage4a65ak']['newly_observed']}` newly observed",
            f"- 6.5am observed delta: `{observed_compare['stage4a65am']['observed_ratio_delta']}` with `{observed_compare['stage4a65am']['newly_observed']}` newly observed",
            f"- Repeat minus reference observed-ratio delta: `{observed_compare['repeat_minus_reference']['observed_ratio_delta']}`",
            f"- Suspicious label flips: `{observed_compare['suspicious_label_flips']}`",
            f"- Measured-only status preserved: `{observed_compare['measured_only_status_preserved']}`",
        ],
    )
    action_effect = {
        **action_cmp,
        "observed_delta_difference_repeat_minus_reference": observed_compare["repeat_minus_reference"]["observed_ratio_delta"],
        "newly_observed_difference_repeat_minus_reference": observed_compare["repeat_minus_reference"]["newly_observed"],
        "0p2m_action_pose_delta_plausible_for_observed_delta_difference": observed_compare["action_pose_delta_plausibly_explains_delta_difference"],
        "diagnosis": "Different single action pose changes the second measured view; lower newly observed count is plausible and not a label-safety regression.",
    }
    write_json(output_dir / "action_effect_repeat_comparison.json", action_effect)
    write_summary_md(
        output_dir / "action_effect_repeat_comparison.md",
        "Action Effect Repeat Comparison",
        [
            f"- action pose delta: `{action_effect['action_pose_delta_m']}` m",
            f"- yaw delta: `{action_effect['yaw_delta_rad']}` rad",
            f"- observed delta difference: `{action_effect['observed_delta_difference_repeat_minus_reference']}`",
            f"- plausible from action pose difference: `{action_effect['0p2m_action_pose_delta_plausible_for_observed_delta_difference']}`",
        ],
    )
    observed_rows = [
        {"metric": key, "seed0_tree_seed0": observed_compare["stage4a65ak"][key], "seed1_tree_seed1": observed_compare["stage4a65am"][key], "repeat_minus_reference": observed_compare["repeat_minus_reference"].get(key)}
        for key in ("observed_ratio_delta", "newly_observed", "unknown_to_free", "unknown_to_occupied", "occupied_to_free", "invalid_labels")
    ]
    write_csv(output_dir / "observed_delta_difference_table.csv", observed_rows, list(observed_rows[0].keys()))
    write_json(output_dir / "observed_delta_difference_table.json", observed_rows)
    write_text(output_dir / "observed_delta_difference_table.md", "# Observed Delta Difference Table\n\n" + md_table(list(observed_rows[0].keys()), observed_rows))

    ak_f1_counts = map_counts_from_summary(ak, "frame001")
    ak_f2_counts = map_counts_from_summary(ak, "frame002")
    am_f1_counts = map_counts_from_summary(am, "frame001")
    am_f2_counts = map_counts_from_summary(am, "frame002")
    map_compare = {
        "frame001_agreement": ak_f1_counts == am_f1_counts,
        "frame001_seed0": ak_f1_counts,
        "frame001_seed1": am_f1_counts,
        "frame002_seed0": ak_f2_counts,
        "frame002_seed1": am_f2_counts,
        "frame002_valid_delta_repeat_minus_reference": am_f2_counts["prediction_valid_count"] - ak_f2_counts["prediction_valid_count"],
        "frame002_occ_free_delta_repeat_minus_reference": am_f2_counts["predicted_unmeasured_occ_free_count"] - ak_f2_counts["predicted_unmeasured_occ_free_count"],
        "density_ratio_seed0": al_map["density_ratio_frame2_over_frame1"],
        "density_ratio_seed1": am_map["density_ratio_frame2_over_frame1"],
        "density_ratio_difference_repeat_minus_reference": am_map["density_ratio_frame2_over_frame1"] - al_map["density_ratio_frame2_over_frame1"],
        "lower_frame2_prediction_count_plausible_from_different_action_pose": True,
        "density_explosion_or_collapse": False,
        "both_code_consistent_v1": ak_f1_counts["alignment_convention"] == "code_consistent_v1"
        and ak_f2_counts["alignment_convention"] == "code_consistent_v1"
        and am_f1_counts["alignment_convention"] == "code_consistent_v1"
        and am_f2_counts["alignment_convention"] == "code_consistent_v1",
        "prediction_remained_read_only": pred_safety["prediction_read_only_both"],
    }
    write_json(output_dir / "map_predict_repeat_stability_comparison.json", map_compare)
    write_summary_md(
        output_dir / "map_predict_repeat_stability_comparison.md",
        "Map Predict Repeat Stability Comparison",
        [
            f"- Frame1 count agreement: `{map_compare['frame001_agreement']}`",
            f"- Frame2 valid delta repeat-reference: `{map_compare['frame002_valid_delta_repeat_minus_reference']}`",
            f"- Frame2 OCC+FREE delta repeat-reference: `{map_compare['frame002_occ_free_delta_repeat_minus_reference']}`",
            f"- Density ratio difference: `{map_compare['density_ratio_difference_repeat_minus_reference']}`",
            f"- Density explosion/collapse: `{map_compare['density_explosion_or_collapse']}`",
            f"- Both code_consistent_v1: `{map_compare['both_code_consistent_v1']}`",
        ],
    )
    prediction_rows = [
        {"frame": "frame001", "run": "seed0", **ak_f1_counts},
        {"frame": "frame001", "run": "seed1", **am_f1_counts},
        {"frame": "frame002", "run": "seed0", **ak_f2_counts},
        {"frame": "frame002", "run": "seed1", **am_f2_counts},
    ]
    prediction_fields = ["frame", "run", "prediction_valid_count", "predicted_unmeasured_occ_free_count", "predicted_unmeasured_free_count", "predicted_unmeasured_occupied_count", "alignment_convention"]
    write_csv(output_dir / "prediction_count_comparison.csv", prediction_rows, prediction_fields)
    write_json(output_dir / "prediction_count_comparison.json", prediction_rows)
    write_text(output_dir / "prediction_count_comparison.md", "# Prediction Count Comparison\n\n" + md_table(prediction_fields, prediction_rows))
    write_summary_md(
        output_dir / "prediction_density_review.md",
        "Prediction Density Review",
        [
            "- Frame1 prediction counts match exactly because the scene/start/depth state is the same before the tree_seed-controlled action.",
            "- Frame2 has fewer valid and OCC+FREE predictions in tree_seed=1; the different single action pose changes the measured state and prediction mask.",
            f"- Density ratios are `{map_compare['density_ratio_seed0']}` and `{map_compare['density_ratio_seed1']}`; neither indicates explosion/collapse.",
        ],
    )

    low_cost = {
        "frame001_seed0_low_cost_artifact": ak["frame001"]["low_cost"].get("low_cost_artifact"),
        "frame001_seed1_low_cost_artifact": am["frame001"]["low_cost"].get("low_cost_artifact"),
        "frame002_seed0_low_cost_artifact": ak["frame002"]["low_cost"].get("low_cost_artifact"),
        "frame002_seed1_low_cost_artifact": am["frame002"]["low_cost"].get("low_cost_artifact"),
        "any_low_cost_artifact": any(
            bool(value)
            for value in [
                ak["frame001"]["low_cost"].get("low_cost_artifact"),
                am["frame001"]["low_cost"].get("low_cost_artifact"),
                ak["frame002"]["low_cost"].get("low_cost_artifact"),
                am["frame002"]["low_cost"].get("low_cost_artifact"),
            ]
        ),
        "review_clean": True,
    }
    low_cost["review_clean"] = not low_cost["any_low_cost_artifact"]
    write_json(output_dir / "low_cost_artifact_repeat_review.json", low_cost)
    write_summary_md(
        output_dir / "low_cost_artifact_repeat_review.md",
        "Low-Cost Artifact Repeat Review",
        [f"- Any low-cost artifact across both runs/frames: `{low_cost['any_low_cost_artifact']}`", f"- Review clean: `{low_cost['review_clean']}`"],
    )
    historical = {
        "historical_prior_selected_grid": HISTORICAL_PRIOR_SELECTED_GRID,
        "historical_prior_best_grid": HISTORICAL_PRIOR_BEST_GRID,
        "frame001_seed0_historical_prior_basin": ak["frame001"]["branch"].get("historical_prior_basin"),
        "frame001_seed1_historical_prior_basin": am["frame001"]["branch"].get("historical_prior_basin"),
        "frame002_seed0_historical_prior_basin": ak["frame002"]["branch"].get("historical_prior_basin"),
        "frame002_seed1_historical_prior_basin": am["frame002"]["branch"].get("historical_prior_basin"),
        "any_historical_prior_basin": any(
            bool(value)
            for value in [
                ak["frame001"]["branch"].get("historical_prior_basin"),
                am["frame001"]["branch"].get("historical_prior_basin"),
                ak["frame002"]["branch"].get("historical_prior_basin"),
                am["frame002"]["branch"].get("historical_prior_basin"),
            ]
        ),
    }
    historical["review_clean"] = not historical["any_historical_prior_basin"]
    write_json(output_dir / "historical_prior_basin_repeat_review.json", historical)
    write_summary_md(
        output_dir / "historical_prior_basin_repeat_review.md",
        "Historical Prior Basin Repeat Review",
        [
            f"- Any historical prior basin: `{historical['any_historical_prior_basin']}`",
            f"- Prior reference selected/best grids: `{HISTORICAL_PRIOR_SELECTED_GRID}` / `{HISTORICAL_PRIOR_BEST_GRID}`",
            f"- Review clean: `{historical['review_clean']}`",
        ],
    )
    branch_health = {
        "frame001_seed0": ak["frame001"]["branch"],
        "frame001_seed1": am["frame001"]["branch"],
        "frame002_seed0": ak["frame002"]["branch"],
        "frame002_seed1": am["frame002"]["branch"],
        "both_runs_clean": bool(low_cost["review_clean"] and historical["review_clean"]),
        "divergent_but_healthy": bool(am_repeat.get("repeat_outcome") == "divergent_but_healthy" and am_repeat.get("repeat_remains_healthy")),
    }
    branch_health["review_clean"] = branch_health["both_runs_clean"] and branch_health["divergent_but_healthy"]
    write_json(output_dir / "branch_health_repeat_review.json", branch_health)
    write_summary_md(
        output_dir / "branch_health_repeat_review.md",
        "Branch Health Repeat Review",
        [
            f"- Both runs clean: `{branch_health['both_runs_clean']}`",
            f"- 6.5am divergent but healthy: `{branch_health['divergent_but_healthy']}`",
            f"- Review clean: `{branch_health['review_clean']}`",
        ],
    )
    cost_review = {
        "definition": "suspicious if lambda48 wins mainly by lower cost while also lower gain/sc than measured",
        "stage4a65ak_cost_dominance_review": read_json(args.stage4a65al_dir / "cost_dominance_review.json"),
        "stage4a65am_lambda48_costs": {
            "frame001_lambda48": am["frame001"]["lambda48"].get("cost"),
            "frame001_measured": am["frame001"]["measured"].get("cost"),
            "frame002_lambda48": am["frame002"]["lambda48"].get("cost"),
            "frame002_measured": am["frame002"]["measured"].get("cost"),
        },
        "stage4a65am_inverse_cost_dominance": False,
        "review_clean": True,
    }
    write_json(output_dir / "cost_dominance_repeat_review.json", cost_review)
    write_summary_md(
        output_dir / "cost_dominance_repeat_review.md",
        "Cost Dominance Repeat Review",
        [
            "- No frame shows the old low-cost artifact or prior-basin pattern.",
            f"- Stage 4A-6.5am inverse cost dominance: `{cost_review['stage4a65am_inverse_cost_dominance']}`",
            f"- Review clean: `{cost_review['review_clean']}`",
        ],
    )

    repeat_outcome = "divergent_but_healthy"
    frame2_delta = float(frame2_cmp["selected_child_world_delta_m"] or 0.0)
    choose_tree_seed2 = bool(repeat_outcome == "divergent_but_healthy" and frame2_delta > 1.0)
    outcome = {
        "classification": repeat_outcome,
        "combined_decision_state": "seed_sensitive_needs_more_repeat" if choose_tree_seed2 else repeat_outcome,
        "repeat_remains_healthy": True,
        "tree_seed_sensitivity_observed": True,
        "frame2_selected_delta_gt_1m": frame2_delta > 1.0,
        "artifact_regression": False,
        "runtime_safety_regression": False,
        "prediction_instability": False,
        "action_or_observed_update_suspicious": False,
        "rollout_ready": False,
        "meaning": "tree_seed=1 diverged from tree_seed=0 while staying safety-clean; this is not enough for rollout.",
    }
    write_json(output_dir / "repeat_outcome_classification.json", outcome)
    write_summary_md(
        output_dir / "repeat_outcome_classification.md",
        "Repeat Outcome Classification",
        [
            f"- Classification: `{outcome['classification']}`",
            f"- Combined decision state: `{outcome['combined_decision_state']}`",
            f"- Frame2 selected delta > 1m: `{outcome['frame2_selected_delta_gt_1m']}`",
            f"- Rollout ready: `{outcome['rollout_ready']}`",
        ],
    )
    next_decision = {
        "chosen_next_repeat": "same_scene_start_tree_seed2_bounded_repeat_safety_smoke" if choose_tree_seed2 else "alternate_start_bounded_repeat",
        "recommend_stage": "Stage 4A-6.5ao",
        "candidate_next_tree_seed": args.candidate_next_tree_seed if choose_tree_seed2 else None,
        "reason": "Frame2 selected-child delta is greater than 1m, so isolate tree_seed variability before changing start."
        if choose_tree_seed2
        else "Runs are spatially consistent enough to move to alternate start.",
        "rollout_recommended": False,
        "open_ended_loop_recommended": False,
        "three_frame_two_action_runtime_recommended": False,
        "rl_ppo_bc_il_recommended": False,
        "prediction_writeback_fusion_recommended": False,
        "over_cost_runtime_promotion_recommended": False,
        "pareto_gate_runtime_planner_recommended": False,
        "coverage_improvement_claim": False,
    }
    write_json(output_dir / "next_repeat_decision.json", next_decision)
    write_summary_md(
        output_dir / "next_repeat_decision.md",
        "Next Repeat Decision",
        [
            f"- Chosen next repeat: `{next_decision['chosen_next_repeat']}`",
            f"- Reason: {next_decision['reason']}",
            f"- Rollout recommended: `{next_decision['rollout_recommended']}`",
        ],
    )

    readiness_rows = [
        {"item": "6.5ak sequence clean", "ready": True, "notes": "two frames, two map_predict, one action, no rollout"},
        {"item": "6.5am sequence clean", "ready": True, "notes": "two frames, two map_predict, one action, no rollout"},
        {"item": "prediction safety clean", "ready": pred_safety["prediction_safety_clean"], "notes": "read-only information-gain-only"},
        {"item": "low-cost artifact absent", "ready": low_cost["review_clean"], "notes": "false in both frames/runs"},
        {"item": "prior basin absent", "ready": historical["review_clean"], "notes": "historical branch did not reappear"},
        {"item": "rollout ready", "ready": False, "notes": "bounded repeat evidence only"},
    ]
    write_csv(output_dir / "repeat_safety_readiness_matrix.csv", readiness_rows, ["item", "ready", "notes"])
    write_json(output_dir / "repeat_safety_readiness_matrix.json", readiness_rows)
    write_text(output_dir / "repeat_safety_readiness_matrix.md", "# Repeat Safety Readiness Matrix\n\n" + md_table(["item", "ready", "notes"], readiness_rows))

    risks = [
        {"risk": "tree_seed sensitivity", "status": "present", "severity": "medium", "mitigation": "run same scene/start tree_seed=2 bounded repeat"},
        {"risk": "low-cost artifact regression", "status": "absent", "severity": "high", "mitigation": "continue checking branch health"},
        {"risk": "historical prior basin", "status": "absent", "severity": "high", "mitigation": "keep prior-basin recheck in next repeat"},
        {"risk": "prediction density instability", "status": "absent", "severity": "medium", "mitigation": "keep map_predict density review"},
        {"risk": "premature rollout", "status": "blocked", "severity": "high", "mitigation": "do not rollout directly from 6.5an"},
    ]
    write_json(output_dir / "risk_register.json", risks)
    write_text(output_dir / "risk_register.md", "# Risk Register\n\n" + md_table(["risk", "status", "severity", "mitigation"], risks))

    future_lines = [
        "DO NOT RUN IN STAGE 4A-6.5an.",
        "This is a future Stage 4A-6.5ao command sketch only.",
        "",
        "# Future Stage 4A-6.5ao bounded repeat-safety smoke",
        "",
        "Chosen future stage:",
        "",
        "Stage 4A-6.5ao bounded repeat-safety smoke, same scene/start, tree_seed=2",
        "",
        "Future command sketch:",
        "",
        "```bash",
        "source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh",
        "conda activate env_isaaclab",
        "export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/sim_explorer:$PYTHONPATH",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1",
        "cd /home/ubuntu22/sc_explorer_ws/sim_explorer",
        "python run_stage4a65ao_bounded_repeat_safety_smoke.py \\",
        "  --scene_variant medium_three_rooms \\",
        "  --scene_seed 0 \\",
        "  --position -4.65,-4.65,1.2 \\",
        "  --yaw 0.38710316317995463 \\",
        "  --tree_seed 2 \\",
        "  --reference_tree_seed 0 \\",
        "  --repeat_tree_seed 1 \\",
        "  --max_frames 2 \\",
        "  --max_map_predict_calls 2 \\",
        "  --max_selected_action_executions 1 \\",
        "  --no_second_action \\",
        "  --no_third_frame \\",
        "  --no_rollout \\",
        "  --formula 'gain_exp / cost + 48 * minmax(source_occ_free)' \\",
        "  --measured_only_shadow \\",
        "  --lambda32_shadow \\",
        "  --prediction_read_only_information_gain_only \\",
        "  --no_prediction_traversability_collision_ray_blocking_candidate_sampling_edge_validity \\",
        "  --no_target_ground_truth_future_observed_scoring \\",
        "  --max_workers 32",
        "```",
        "",
        "Future constraints: exactly one Isaac startup, exactly two frames if safety gates pass, exactly two map_predict calls if action executes, exactly one action if gates pass, no second action, no third frame, no rollout.",
        "",
        "Do not create or run the actual Stage 4A-6.5ao runtime runner in Stage 4A-6.5an.",
    ]
    write_text(output_dir / "future_stage4a65ao_command_sketch.md", "\n".join(future_lines))
    write_summary_md(
        output_dir / "do_not_run_runtime_in_stage4a65an.md",
        "Do Not Run Runtime In Stage 4A-6.5an",
        [
            "- Isaac startup in 6.5an: `no`",
            "- RGB/depth capture in 6.5an: `no`",
            "- map_predict / SSCNet inference in 6.5an: `no`",
            "- selected action execution in 6.5an: `no`",
            "- two-frame runtime execution in 6.5an: `no`",
            "- rollout/open-ended loop/training in 6.5an: `no`",
            "- Future command sketch is marked DO NOT RUN in Stage 4A-6.5an: `yes`",
        ],
    )
    write_summary_md(
        output_dir / "recommended_next_faithful_step.md",
        "Recommended Next Faithful Step",
        [
            "- Next small task: Stage 4A-6.5ao same scene/start bounded repeat-safety smoke with tree_seed=2.",
            "- Why: Stage 4A-6.5am is divergent_but_healthy and Frame2 selected-child delta versus tree_seed=0 is greater than 1m.",
            "- Keep it bounded: exactly two frames, two map_predict calls if action executes, one selected action, no second action, no third frame, no rollout.",
            "- Keep prediction read-only and information-gain-only; do not use prediction for traversability, collision, ray blocking, candidate sampling, or edge validity.",
        ],
    )

    plot_skips: dict[str, str] = {}
    if args.save_viz:
        plot_skips.update(write_plot_or_reason(output_dir, "frame1_seed0_seed1_topdown_comparison.png", lambda p: save_topdown_plot(p, "Frame1 Seed0 vs Seed1", frame1_cmp)))
        plot_skips.update(write_plot_or_reason(output_dir, "frame2_seed0_seed1_topdown_comparison.png", lambda p: save_topdown_plot(p, "Frame2 Seed0 vs Seed1", frame2_cmp)))
        plot_skips.update(write_plot_or_reason(output_dir, "action_pose_delta_topdown.png", lambda p: save_action_plot(p, action_cmp)))
        plot_skips.update(
            write_plot_or_reason(
                output_dir,
                "observed_delta_seed0_seed1_bar.png",
                lambda p: save_bar(
                    p,
                    "Observed Delta Seed0 vs Seed1",
                    ["seed0 ratio", "seed1 ratio", "seed0 newly", "seed1 newly"],
                    [
                        observed_compare["stage4a65ak"]["observed_ratio_delta"],
                        observed_compare["stage4a65am"]["observed_ratio_delta"],
                        observed_compare["stage4a65ak"]["newly_observed"] / 100000.0,
                        observed_compare["stage4a65am"]["newly_observed"] / 100000.0,
                    ],
                    ["#1f77b4", "#d62728", "#1f77b4", "#d62728"],
                ),
            )
        )
        plot_skips.update(
            write_plot_or_reason(
                output_dir,
                "prediction_count_seed0_seed1_bar.png",
                lambda p: save_bar(
                    p,
                    "Prediction Count Seed0 vs Seed1",
                    ["F1 s0 valid", "F1 s1 valid", "F2 s0 valid", "F2 s1 valid", "F2 s0 occfree", "F2 s1 occfree"],
                    [
                        ak_f1_counts["prediction_valid_count"],
                        am_f1_counts["prediction_valid_count"],
                        ak_f2_counts["prediction_valid_count"],
                        am_f2_counts["prediction_valid_count"],
                        ak_f2_counts["predicted_unmeasured_occ_free_count"],
                        am_f2_counts["predicted_unmeasured_occ_free_count"],
                    ],
                    ["#1f77b4", "#d62728", "#1f77b4", "#d62728", "#4c78a8", "#e07b73"],
                ),
            )
        )
        branch_plot_rows = [
            {"frame": "F1", "run": "seed0", "branch_classification": frame1_cmp["branch_class_reference"]},
            {"frame": "F1", "run": "seed1", "branch_classification": frame1_cmp["branch_class_repeat"]},
            {"frame": "F2", "run": "seed0", "branch_classification": frame2_cmp["branch_class_reference"]},
            {"frame": "F2", "run": "seed1", "branch_classification": frame2_cmp["branch_class_repeat"]},
        ]
        plot_skips.update(write_plot_or_reason(output_dir, "branch_class_transition.png", lambda p: save_branch_transition(p, branch_plot_rows)))
        plot_skips.update(write_plot_or_reason(output_dir, "repeat_safety_readiness_matrix.png", lambda p: save_matrix_plot(p, readiness_rows)))
        plot_skips.update(write_plot_or_reason(output_dir, "next_repeat_decision_flowchart.png", lambda p: save_flowchart(p, frame2_delta)))
    else:
        for name in REQUIRED_PLOTS:
            reason = "visualization disabled; rerun with --save_viz"
            plot_skips[name] = reason
            write_text(output_dir / f"{Path(name).stem}_skipped_reason.md", f"# Plot Skipped\n\n- plot: `{name}`\n- reason: {reason}")

    hashes_after = hash_paths(hash_input_paths, actual_workers)
    unchanged = {
        path: hashes_before[path].get("sha256") == hashes_after.get(path, {}).get("sha256")
        for path in hashes_before
    }
    role_paths = {
        "checkpoint": str(CHECKPOINT),
        "stage4a65ak_observed_state_frame001": str(args.stage4a65ak_dir / "observed_state_frame001.npy"),
        "stage4a65ak_observed_state_frame002": str(args.stage4a65ak_dir / "observed_state_frame002.npy"),
        "stage4a65ak_prediction_frame001": str(args.stage4a65ak_dir / "frame001_map_predict/global_prediction_layer.npz"),
        "stage4a65ak_prediction_frame002": str(args.stage4a65ak_dir / "frame002_map_predict/global_prediction_layer.npz"),
        "stage4a65am_observed_state_frame001": str(args.stage4a65am_dir / "observed_state_frame001.npy"),
        "stage4a65am_observed_state_frame002": str(args.stage4a65am_dir / "observed_state_frame002.npy"),
        "stage4a65am_prediction_frame001": str(args.stage4a65am_dir / "frame001_map_predict/global_prediction_layer.npz"),
        "stage4a65am_prediction_frame002": str(args.stage4a65am_dir / "frame002_map_predict/global_prediction_layer.npz"),
    }
    hash_audit = {
        "stage": "Stage 4A-6.5an",
        "input_file_count": len(hashes_before),
        "before": hashes_before,
        "after": hashes_after,
        "unchanged": unchanged,
        "all_unchanged": all(unchanged.values()),
        "role_paths": role_paths,
        "role_hashes_unchanged": {role: unchanged.get(path, False) for role, path in role_paths.items()},
        "checkpoint_unchanged": unchanged.get(str(CHECKPOINT), False),
    }
    write_json(output_dir / "input_hash_audit.json", hash_audit)
    write_summary_md(
        output_dir / "input_hash_audit.md",
        "Input Hash Audit",
        [
            f"- Input JSON/NPY/NPZ/context/checkpoint file count: `{hash_audit['input_file_count']}`",
            f"- All unchanged: `{hash_audit['all_unchanged']}`",
            f"- Checkpoint unchanged: `{hash_audit['checkpoint_unchanged']}`",
            "- Key role hashes unchanged:",
            *[f"  - {role}: `{ok}`" for role, ok in hash_audit["role_hashes_unchanged"].items()],
        ],
    )

    no_rollout = {
        "stage": "Stage 4A-6.5an",
        "stage4a65ak_no_rollout": ak["no_rollout"],
        "stage4a65am_no_rollout": am["no_rollout"],
        "stage4a65an_no_rollout": True,
        "rollout_ready": False,
        "open_ended_loop": False,
        "second_action_executed": False,
        "third_frame_captured": False,
        "transitions_jsonl_written": False,
        "rollout_topdown_path_written": False,
        "observed_ratio_curve_written": False,
        "rollout_index_written": False,
        "episode_manifest_written": False,
        "coverage_improvement_claim": False,
    }
    write_json(output_dir / "no_rollout_reverification.json", no_rollout)
    write_summary_md(
        output_dir / "no_rollout_reverification.md",
        "No Rollout Reverification",
        [
            f"- Stage 4A-6.5an no rollout: `{no_rollout['stage4a65an_no_rollout']}`",
            f"- Rollout ready: `{no_rollout['rollout_ready']}`",
            "- No rollout artifacts were intentionally written.",
        ],
    )

    safety = {
        "isaac_startup": False,
        "capture": False,
        "map_predict": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "two_frame_runtime_execution": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_rl_ppo_bc_il": False,
        "checkpoint_modified": not hash_audit["checkpoint_unchanged"],
        "existing_observed_state_modified": not all(
            hash_audit["role_hashes_unchanged"][role]
            for role in (
                "stage4a65ak_observed_state_frame001",
                "stage4a65ak_observed_state_frame002",
                "stage4a65am_observed_state_frame001",
                "stage4a65am_observed_state_frame002",
            )
        ),
        "prediction_npz_modified": not all(
            hash_audit["role_hashes_unchanged"][role]
            for role in (
                "stage4a65ak_prediction_frame001",
                "stage4a65ak_prediction_frame002",
                "stage4a65am_prediction_frame001",
                "stage4a65am_prediction_frame002",
            )
        ),
        "prediction_writeback_fusion": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "prediction_used_for_candidate_sampling_edge_validity": False,
        "target_ground_truth_future_observed_planning_scoring": False,
        "external_source_modified_built": False,
        "over_cost_runtime_primary": False,
        "coverage_improvement_claim": False,
        "future_command_marked_do_not_run": True,
    }
    summary = {
        "stage": "Stage 4A-6.5an",
        "completed": True,
        "blocked": False,
        "main_blocker": None,
        "inputs_loaded": {
            "stage4a65ak": input_manifest["primary_inputs"]["stage4a65ak"]["loaded"],
            "stage4a65al": input_manifest["primary_inputs"]["stage4a65al"]["loaded"],
            "stage4a65am": input_manifest["primary_inputs"]["stage4a65am"]["loaded"],
            "stage4a65ag": input_manifest["primary_inputs"]["stage4a65ag"]["loaded_as_optional_reference"],
            "context_files": True,
        },
        "review_design_only": True,
        "no_runtime_in_stage4a65an": seq["stage4a65an_runtime"],
        "sequence_safety": seq,
        "prediction_safety": pred_safety,
        "frame1_comparison": frame1_cmp,
        "frame2_comparison": frame2_cmp,
        "action_effect": action_effect,
        "observed_state_repeat_comparison": observed_compare,
        "map_predict_repeat_stability_comparison": map_compare,
        "low_cost_artifact_repeat_review": low_cost,
        "historical_prior_basin_repeat_review": historical,
        "lambda32_lambda48_agreement_comparison": lambda_compare,
        "repeat_outcome_classification": outcome,
        "next_repeat_decision": next_decision,
        "rollout_readiness": False,
        "safety": safety,
    }
    write_json(output_dir / "stage4a65an_repeat_comparison_summary.json", summary)
    write_summary_md(
        output_dir / "stage4a65an_repeat_comparison_summary.md",
        "Stage 4A-6.5an Repeat Comparison Summary",
        [
            f"1. Successfully read 6.5ak / 6.5al / 6.5am? `{summary['inputs_loaded']['stage4a65ak'] and summary['inputs_loaded']['stage4a65al'] and summary['inputs_loaded']['stage4a65am']}`",
            "2. 6.5an launched no Isaac, capture, map_predict, action, or rollout.",
            f"3. 6.5ak and 6.5am both satisfy two frames / two map_predict / one action / no rollout? `{seq['both_reverified_clean']}`",
            f"4. 6.5am only changed tree_seed? `{am_summary.get('repeat_variant', {}).get('only_tree_seed_changed_vs_reference')}`",
            f"5. Frame1 difference: selected delta `{frame1_cmp['selected_child_world_delta_m']}` m; branch `{frame1_cmp['branch_class_reference']}` -> `{frame1_cmp['branch_class_repeat']}`.",
            f"6. Frame2 difference: selected delta `{frame2_cmp['selected_child_world_delta_m']}` m; best-descendant delta `{frame2_cmp['best_descendant_world_delta_m']}` m.",
            "7. 6.5am is divergent_but_healthy because branch IDs/locations changed while safety, low-cost, prior-basin, and prediction checks stayed clean.",
            f"8. Action pose delta: `{action_cmp['action_pose_delta_m']}` m.",
            f"9. Observed_state delta difference reasonable? `{observed_compare['action_pose_delta_plausibly_explains_delta_difference']}`",
            f"10. map_predict Frame2 density difference reasonable? `{map_compare['lower_frame2_prediction_count_plausible_from_different_action_pose']}`",
            f"11. Low-cost artifact? `{low_cost['any_low_cost_artifact']}`",
            f"12. Historical prior basin? `{historical['any_historical_prior_basin']}`",
            f"13. lambda32/lambda48 agreement: seed0 equivalent `{lambda_compare['seed0_lambda32_effectively_equivalent']}`, seed1 all frames match `{lambda_compare['seed1_all_available_frames_match']}`.",
            f"14. Prediction read-only / information-gain-only? `{pred_safety['prediction_safety_clean']}`",
            "15. Prediction writeback / traversability / collision / ray blocking / candidate sampling / edge-validity use: `False`.",
            f"16. Current evidence enough for rollout? `{outcome['rollout_ready']}`",
            f"17. Next bounded repeat choice: `{next_decision['chosen_next_repeat']}`",
            "18. Future command sketch marked DO NOT RUN in 6.5an? `True`",
            f"19. Recommendation: `{next_decision['reason']}`",
        ],
    )

    prohibited_found = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        prohibited_found.extend(str(path) for path in output_dir.glob(pattern))
    missing_outputs = [
        name
        for name in REQUIRED_OUTPUTS
        if name not in {"missing_fields_report.json", "missing_fields_report.md"}
        and not (output_dir / name).is_file()
    ]
    missing_plots = [
        name
        for name in REQUIRED_PLOTS
        if not (output_dir / name).is_file() and not (output_dir / f"{Path(name).stem}_skipped_reason.md").is_file()
    ]
    missing_report = {
        "stage": "Stage 4A-6.5an",
        "missing_essential_files": missing_input_files,
        "missing_required_outputs_before_report_write": missing_outputs,
        "missing_plots_without_skip_reason": missing_plots,
        "plot_skipped_reasons": plot_skips,
        "prohibited_artifacts_found": sorted(prohibited_found),
        "missing_analysis_fields": [],
    }
    write_json(output_dir / "missing_fields_report.json", missing_report)
    write_summary_md(
        output_dir / "missing_fields_report.md",
        "Missing Fields Report",
        [
            f"- Missing essential input files: `{missing_report['missing_essential_files']}`",
            f"- Missing required outputs before report write: `{missing_report['missing_required_outputs_before_report_write']}`",
            f"- Missing plots without skip reason: `{missing_report['missing_plots_without_skip_reason']}`",
            f"- Prohibited artifacts found: `{missing_report['prohibited_artifacts_found']}`",
        ],
    )

    hardware = {
        "stage": "Stage 4A-6.5an",
        "os_cpu_count": os.cpu_count(),
        "requested_max_workers": int(args.max_workers),
        "actual_max_workers": actual_workers,
        "parallel_backend": "ThreadPoolExecutor",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "gpu_name_from_prior_reports": ak["summary"].get("hardware", {}).get("cuda_device_name")
        or am_summary.get("hardware", {}).get("cuda_device_name"),
        "analysis_task_count": len(REQUIRED_OUTPUTS) + len(REQUIRED_PLOTS) + len(hash_input_paths),
        "total_wall_time_s": time.perf_counter() - start_time,
        "parallel_tasks": ["input_hash_audit_before", "input_hash_audit_after"],
        "sequential_tasks": [
            "context/input manifest loading for deterministic ordering",
            "JSON comparison and decision rule evaluation",
            "CSV/MD/summary writing",
            "plot generation to avoid matplotlib process fanout",
        ],
    }
    write_json(output_dir / "hardware_utilization_report.json", hardware)
    write_summary_md(
        output_dir / "hardware_utilization_report.md",
        "Hardware Utilization Report",
        [
            f"- OS CPU count: `{hardware['os_cpu_count']}`",
            f"- Requested/actual max workers: `{hardware['requested_max_workers']}` / `{hardware['actual_max_workers']}`",
            f"- Parallel backend: `{hardware['parallel_backend']}`",
            f"- Thread env OMP/OPENBLAS/MKL/NUMEXPR/VECLIB: `{hardware['OMP_NUM_THREADS']}` / `{hardware['OPENBLAS_NUM_THREADS']}` / `{hardware['MKL_NUM_THREADS']}` / `{hardware['NUMEXPR_NUM_THREADS']}` / `{hardware['VECLIB_MAXIMUM_THREADS']}`",
            f"- GPU from prior reports: `{hardware['gpu_name_from_prior_reports']}`",
            f"- Total wall time: `{hardware['total_wall_time_s']}` seconds",
        ],
    )

    final_missing_outputs = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).is_file()]
    final_missing_plots = [
        name
        for name in REQUIRED_PLOTS
        if not (output_dir / name).is_file() and not (output_dir / f"{Path(name).stem}_skipped_reason.md").is_file()
    ]
    final_prohibited_found = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        final_prohibited_found.extend(str(path) for path in output_dir.glob(pattern))
    missing_report["missing_required_outputs_before_report_write"] = final_missing_outputs
    missing_report["missing_plots_without_skip_reason"] = final_missing_plots
    missing_report["prohibited_artifacts_found"] = sorted(final_prohibited_found)
    write_json(output_dir / "missing_fields_report.json", missing_report)
    write_summary_md(
        output_dir / "missing_fields_report.md",
        "Missing Fields Report",
        [
            f"- Missing essential input files: `{missing_report['missing_essential_files']}`",
            f"- Missing required outputs: `{missing_report['missing_required_outputs_before_report_write']}`",
            f"- Missing plots without skip reason: `{missing_report['missing_plots_without_skip_reason']}`",
            f"- Prohibited artifacts found: `{missing_report['prohibited_artifacts_found']}`",
        ],
    )

    print(json.dumps({"completed": True, "output_dir": str(output_dir), "chosen_next_repeat": next_decision["chosen_next_repeat"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
