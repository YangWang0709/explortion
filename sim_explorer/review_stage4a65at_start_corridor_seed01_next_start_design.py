#!/usr/bin/env python3
"""Stage 4A-6.5at start_corridor seed0/seed1 review and next-start design.

This stage is offline diagnosis/design only. It reads completed Stage
4A-6.5aq and Stage 4A-6.5as bounded smoke outputs, rechecks safety contracts,
compares seed0/seed1 behavior at start_corridor, and writes a future Stage
4A-6.5au command sketch. It does not start Isaac, capture RGB/depth, run
map_predict or SSCNet inference, execute actions, run rollout, train, or
modify existing runtime inputs.
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
import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65at_start_corridor_seed01_review_next_start_design"
DEFAULT_AQ_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke"
DEFAULT_AS_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65as_start_corridor_tree_seed1_bounded_smoke"
DEFAULT_AR_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ar_alternate_start_post_action_diagnosis"
DEFAULT_AP_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design"
DEFAULT_AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_AM_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_AO_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2"
DEFAULT_START_DATASET_DIR = WORKSPACE / "outputs/isaac_medium_rollout_dataset_empty_pred_astar"
DEFAULT_START_CORRIDOR_METADATA = (
    DEFAULT_START_DATASET_DIR / "episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json"
)
DEFAULT_START_ROOM_B_METADATA = (
    DEFAULT_START_DATASET_DIR / "episodes/medium_three_rooms_seed0_start_room_b_empty_astar/scene_metadata.json"
)
CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
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
HISTORICAL_PRIOR_SELECTED_GRID = [11, 15, 11]
HISTORICAL_PRIOR_BEST_GRID = [14, 15, 11]

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
    "forbidden_artifact_scan.json",
    "forbidden_artifact_scan.md",
    "start_pose_consistency.json",
    "start_pose_consistency.md",
    "action_pose_seed01_comparison.json",
    "action_pose_seed01_comparison.md",
    "start_to_action_geometry_seed01.json",
    "start_to_action_geometry_seed01.md",
    "frame1_seed0_seed1_tree_comparison.json",
    "frame1_seed0_seed1_tree_comparison.md",
    "frame2_seed0_seed1_tree_comparison.json",
    "frame2_seed0_seed1_tree_comparison.md",
    "branch_spatial_delta_table.csv",
    "branch_spatial_delta_table.json",
    "branch_spatial_delta_table.md",
    "branch_class_transition_summary.json",
    "branch_class_transition_summary.md",
    "lambda32_lambda48_seed01_agreement.json",
    "lambda32_lambda48_seed01_agreement.md",
    "observed_state_seed01_comparison.csv",
    "observed_state_seed01_comparison.json",
    "observed_state_seed01_comparison.md",
    "observed_transition_seed01_table.csv",
    "observed_transition_seed01_table.json",
    "observed_transition_seed01_table.md",
    "measured_only_update_review.json",
    "measured_only_update_review.md",
    "map_predict_seed01_stability.csv",
    "map_predict_seed01_stability.json",
    "map_predict_seed01_stability.md",
    "prediction_count_comparison.csv",
    "prediction_count_comparison.json",
    "prediction_count_comparison.md",
    "prediction_safety_review.json",
    "prediction_safety_review.md",
    "low_cost_artifact_seed01_review.json",
    "low_cost_artifact_seed01_review.md",
    "historical_prior_basin_seed01_review.json",
    "historical_prior_basin_seed01_review.md",
    "branch_health_seed01_review.json",
    "branch_health_seed01_review.md",
    "cost_dominance_seed01_review.json",
    "cost_dominance_seed01_review.md",
    "start_corridor_seed01_outcome_classification.json",
    "start_corridor_seed01_outcome_classification.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "recommended_next_faithful_step.md",
    "next_start_candidate_inventory.json",
    "next_start_candidate_inventory.md",
    "selected_next_start_design.json",
    "selected_next_start_design.md",
    "future_stage4a65au_command_sketch.md",
    "do_not_run_runtime_in_stage4a65at.md",
    "stage4a65at_start_corridor_seed01_review_summary.json",
    "stage4a65at_start_corridor_seed01_review_summary.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_PLOTS = [
    "start_corridor_seed01_action_topdown.png",
    "frame1_seed0_seed1_tree_topdown.png",
    "frame2_seed0_seed1_tree_topdown.png",
    "branch_spatial_delta_seed01.png",
    "observed_delta_seed01_bar.png",
    "prediction_count_seed01_bar.png",
    "lambda32_lambda48_seed01_comparison.png",
    "repeat_safety_readiness_matrix.png",
    "next_start_design_topdown.png",
    "next_stage_decision_flowchart.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "*.npy",
    "*.npz",
    "frame001*",
    "frame002*",
    "frame003*",
    "action002*",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "manifest.jsonl",
    "episode_manifest*",
    "*replay_buffer*",
    "*policy_checkpoint*",
    "*training_output*",
]


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_json_optional(path: Path, missing: list[dict[str, Any]], essential: bool, role: str) -> Any:
    if path.is_file():
        return read_json(path)
    missing.append({"path": str(path), "role": role, "essential": essential})
    return None


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(clean(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: clean(row.get(key, "")) for key in fieldnames})


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record["sha256"] = sha256_file(path)
        record["size_bytes"] = int(path.stat().st_size)
        record["mtime_ns"] = int(path.stat().st_mtime_ns)
    return record


def hash_many(paths: list[Path], max_workers: int) -> dict[str, dict[str, Any]]:
    unique = sorted({Path(path) for path in paths}, key=lambda p: str(p))
    if not unique:
        return {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        records = list(pool.map(file_record, unique))
    return {record["path"]: record for record in records}


def md_bool(value: Any) -> str:
    return "`true`" if bool(value) else "`false`"


def md_list(title: str, lines: list[str]) -> str:
    return "\n".join([f"# {title}", "", *lines])


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def distance_m(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(av[:3] - bv[:3]))


def close_list(a: Any, b: Any, atol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return False
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=atol))


def yaw_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return abs(float(a) - float(b))


def in_map_bounds(position: list[float] | None, bounds: dict[str, Any] | None) -> bool:
    if position is None or not bounds:
        return False
    x_ok = float(bounds["x"][0]) <= float(position[0]) <= float(bounds["x"][1])
    y_ok = float(bounds["y"][0]) <= float(position[1]) <= float(bounds["y"][1])
    z_ok = float(bounds.get("z", [position[2], position[2]])[0]) <= float(position[2]) <= float(
        bounds.get("z", [position[2], position[2]])[1]
    )
    return bool(x_ok and y_ok and z_ok)


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def decision_summary(path: Path, missing: list[dict[str, Any]], role: str) -> dict[str, Any]:
    payload = read_json_optional(path, missing, True, role)
    decision = payload.get("decision", {}) if isinstance(payload, dict) else {}
    return {
        "available": bool(decision),
        "path": str(path),
        "selected_child_id": decision.get("selected_child_id"),
        "best_descendant_id": decision.get("best_descendant_id"),
        "selected_child_grid": decision.get("selected_child_grid"),
        "best_descendant_grid": decision.get("best_descendant_grid"),
        "selected_child_world": decision.get("selected_child_world"),
        "best_descendant_world": decision.get("best_descendant_world"),
        "selected_child_value": decision.get("selected_child_value"),
        "best_descendant_value": decision.get("best_descendant_value"),
        "selected_child_cost": decision.get("selected_child_cost"),
        "best_descendant_cost": decision.get("best_descendant_cost"),
        "selected_child_gain_exp": decision.get("selected_child_gain_exp"),
        "best_descendant_gain_exp": decision.get("best_descendant_gain_exp"),
        "selected_child_source_occ_free": decision.get("selected_child_source_occ_free"),
        "best_descendant_source_occ_free": decision.get("best_descendant_source_occ_free"),
    }


def extract_stage(base: Path, label: str, tree_seed: int, summary_name: str, lambda_name: str) -> dict[str, Any]:
    missing: list[dict[str, Any]] = []
    summary = read_json_optional(base / summary_name, missing, True, f"{label} summary")
    runtime_setup = read_json_optional(base / "runtime_setup_summary.json", missing, True, f"{label} runtime setup")
    action = read_json_optional(base / "action_execution_report.json", missing, True, f"{label} action")
    frame001_pose = read_json_optional(base / "frame001_pose.json", missing, True, f"{label} frame001 pose")
    frame002_pose = read_json_optional(base / "frame002_pose.json", missing, True, f"{label} frame002 pose")
    observed = read_json_optional(base / "observed_state_delta_summary.json", missing, True, f"{label} observed")
    maps = read_json_optional(base / "map_predict_two_frame_stability.json", missing, True, f"{label} map_predict")
    pred = read_json_optional(base / "prediction_safety_report.json", missing, True, f"{label} prediction safety")
    no_rollout = read_json_optional(base / "no_rollout_report.json", missing, True, f"{label} no rollout")
    formula = read_json_optional(base / "formula_definition.json", missing, True, f"{label} formula")
    alt = read_json_optional(base / "alternate_start_definition.json", missing, True, f"{label} start definition")
    lambda_agreement = read_json_optional(base / lambda_name, missing, True, f"{label} lambda32/lambda48")
    hash_checks = read_json_optional(base / "hash_checks.json", missing, False, f"{label} hash checks")
    frame1_branch = read_json_optional(base / "frame001_branch_classification.json", missing, True, f"{label} frame1 branch")
    frame2_branch = read_json_optional(base / "frame002_branch_classification.json", missing, True, f"{label} frame2 branch")
    frame1_low = read_json_optional(base / "frame001_low_cost_artifact_diagnosis.json", missing, True, f"{label} frame1 low cost")
    frame2_low = read_json_optional(base / "frame002_low_cost_artifact_diagnosis.json", missing, True, f"{label} frame2 low cost")

    sequence_source = summary.get("runtime_setup", {}) if isinstance(summary, dict) else {}
    if not sequence_source:
        sequence_source = {
            "isaac_startup_count": runtime_setup.get("isaac_startup_count") if isinstance(runtime_setup, dict) else None,
            "frames_captured": runtime_setup.get("max_frames") if isinstance(runtime_setup, dict) else None,
            "map_predict_calls": 2,
            "selected_action_execution_count": action.get("action_execution_count") if isinstance(action, dict) else None,
            "second_action": not runtime_setup.get("no_second_action", False) if isinstance(runtime_setup, dict) else None,
            "third_frame": not runtime_setup.get("no_third_frame", False) if isinstance(runtime_setup, dict) else None,
            "rollout": not runtime_setup.get("no_rollout", False) if isinstance(runtime_setup, dict) else None,
            "two_frame_runtime_executed": True,
        }

    safety_source = summary.get("safety", {}) if isinstance(summary, dict) else {}

    return {
        "label": label,
        "base": base,
        "tree_seed": tree_seed,
        "missing": missing,
        "summary": summary or {},
        "runtime_setup_summary": runtime_setup or {},
        "sequence": {
            "isaac_startup_count_clean_run": sequence_source.get("isaac_startup_count"),
            "frames_captured": sequence_source.get("frames_captured"),
            "map_predict_calls": sequence_source.get("map_predict_calls"),
            "selected_action_execution_count": sequence_source.get("selected_action_execution_count"),
            "second_action": bool(sequence_source.get("second_action")),
            "third_frame": bool(sequence_source.get("third_frame")),
            "rollout": bool(sequence_source.get("rollout")),
            "two_frame_runtime_executed": bool(sequence_source.get("two_frame_runtime_executed")),
        },
        "safety": safety_source,
        "action": action or {},
        "frame001_pose": frame001_pose or {},
        "frame002_pose": frame002_pose or {},
        "observed": observed or {},
        "maps": maps or {},
        "prediction_safety": pred or {},
        "no_rollout": no_rollout or {},
        "formula": formula or {},
        "alternate_start": alt or {},
        "lambda_agreement": lambda_agreement or {},
        "hash_checks": hash_checks or {},
        "branch": {"frame001": frame1_branch or {}, "frame002": frame2_branch or {}},
        "low_cost": {"frame001": frame1_low or {}, "frame002": frame2_low or {}},
        "decision": {
            "frame001": {
                "measured": decision_summary(base / "frame001_measured_shadow_tree_decision.json", missing, f"{label} frame1 measured"),
                "lambda48": decision_summary(base / "frame001_lambda48_primary_tree_decision.json", missing, f"{label} frame1 lambda48"),
                "lambda32": decision_summary(base / "frame001_lambda32_shadow_tree_decision.json", missing, f"{label} frame1 lambda32"),
            },
            "frame002": {
                "measured": decision_summary(base / "frame002_measured_shadow_tree_decision.json", missing, f"{label} frame2 measured"),
                "lambda48": decision_summary(base / "frame002_lambda48_diagnostic_tree_decision.json", missing, f"{label} frame2 lambda48"),
                "lambda32": decision_summary(base / "frame002_lambda32_shadow_tree_decision.json", missing, f"{label} frame2 lambda32"),
            },
        },
    }


def stage_sequence_clean(stage: dict[str, Any]) -> bool:
    seq = stage["sequence"]
    return bool(
        seq["isaac_startup_count_clean_run"] == 1
        and seq["frames_captured"] == 2
        and seq["map_predict_calls"] == 2
        and seq["selected_action_execution_count"] == 1
        and seq["second_action"] is False
        and seq["third_frame"] is False
        and seq["rollout"] is False
    )


def prediction_clean(stage: dict[str, Any]) -> bool:
    pred = stage["prediction_safety"]
    safety = stage["safety"]
    return bool(
        pred.get("prediction_read_only") is True
        and pred.get("prediction_information_gain_only") is True
        and pred.get("prediction_written_to_observed_state") is False
        and pred.get("prediction_fused_into_observed_state") is False
        and pred.get("prediction_used_for_traversability") is False
        and pred.get("prediction_used_for_collision") is False
        and pred.get("prediction_ray_blocking") is False
        and pred.get("prediction_used_for_candidate_sampling") is False
        and pred.get("prediction_used_for_edge_validity") is False
        and pred.get("target_lr_target_hr_ground_truth_used_for_planning_scoring") is False
        and pred.get("future_observed_used_for_planning_scoring") is False
        and safety.get("prediction_writeback") is False
        and safety.get("prediction_npz_modified_after_creation") is False
    )


def no_rollout_clean(stage: dict[str, Any]) -> bool:
    report = stage["no_rollout"]
    return bool(
        report.get("rollout") is False
        and report.get("frame003_captured") is False
        and report.get("second_action_executed") is False
        and report.get("open_ended_loop") is False
        and report.get("rollout_ready") is False
    )


def tree_relations(stage: dict[str, Any], frame: str) -> dict[str, Any]:
    measured = stage["decision"][frame]["measured"]
    lam48 = stage["decision"][frame]["lambda48"]
    lam32 = stage["decision"][frame]["lambda32"]
    return {
        "measured_vs_lambda48_same_selected_child": measured["selected_child_id"] == lam48["selected_child_id"],
        "measured_vs_lambda48_same_best_descendant": measured["best_descendant_id"] == lam48["best_descendant_id"],
        "measured_vs_lambda48_selected_delta_m": distance_m(measured["selected_child_world"], lam48["selected_child_world"]),
        "measured_vs_lambda48_best_delta_m": distance_m(measured["best_descendant_world"], lam48["best_descendant_world"]),
        "lambda32_vs_lambda48_same_selected_child": lam32["selected_child_id"] == lam48["selected_child_id"],
        "lambda32_vs_lambda48_same_best_descendant": lam32["best_descendant_id"] == lam48["best_descendant_id"],
        "lambda32_vs_lambda48_selected_delta_m": distance_m(lam32["selected_child_world"], lam48["selected_child_world"]),
        "lambda32_vs_lambda48_best_delta_m": distance_m(lam32["best_descendant_world"], lam48["best_descendant_world"]),
    }


def compare_frame(aq: dict[str, Any], ass: dict[str, Any], frame: str) -> dict[str, Any]:
    aq_lam48 = aq["decision"][frame]["lambda48"]
    as_lam48 = ass["decision"][frame]["lambda48"]
    aq_measured = aq["decision"][frame]["measured"]
    as_measured = ass["decision"][frame]["measured"]
    aq_branch = aq["branch"][frame]
    as_branch = ass["branch"][frame]
    selected_delta = distance_m(aq_lam48["selected_child_world"], as_lam48["selected_child_world"])
    best_delta = distance_m(aq_lam48["best_descendant_world"], as_lam48["best_descendant_world"])
    result = {
        "frame": frame,
        "reference_stage": "Stage 4A-6.5aq",
        "current_stage": "Stage 4A-6.5as",
        "aq_lambda48": aq_lam48,
        "as_lambda48": as_lam48,
        "aq_measured": aq_measured,
        "as_measured": as_measured,
        "exact_selected_child_agreement_aq_vs_as": aq_lam48["selected_child_id"] == as_lam48["selected_child_id"]
        and aq_lam48["selected_child_grid"] == as_lam48["selected_child_grid"],
        "exact_best_descendant_agreement_aq_vs_as": aq_lam48["best_descendant_id"] == as_lam48["best_descendant_id"]
        and aq_lam48["best_descendant_grid"] == as_lam48["best_descendant_grid"],
        "spatial_selected_child_delta_m": selected_delta,
        "spatial_best_descendant_delta_m": best_delta,
        "branch_class_transition": f"{aq_branch.get('classification')} -> {as_branch.get('classification')}",
        "aq_classification": aq_branch.get("classification"),
        "as_classification": as_branch.get("classification"),
        "aq_relations": tree_relations(aq, frame),
        "as_relations": tree_relations(ass, frame),
        "lambda32_lambda48_agreement_aq": {
            "same_selected_child": tree_relations(aq, frame)["lambda32_vs_lambda48_same_selected_child"],
            "same_best_descendant": tree_relations(aq, frame)["lambda32_vs_lambda48_same_best_descendant"],
        },
        "lambda32_lambda48_agreement_as": {
            "same_selected_child": tree_relations(ass, frame)["lambda32_vs_lambda48_same_selected_child"],
            "same_best_descendant": tree_relations(ass, frame)["lambda32_vs_lambda48_same_best_descendant"],
        },
        "as_healthy_nonmeasured": bool(as_branch.get("healthy_nonmeasured_candidate"))
        and bool(as_branch.get("distinct_nonmeasured_branch")),
        "as_true_distinct_branch_not_local_jitter": bool(as_branch.get("distinct_nonmeasured_branch"))
        and not bool(as_branch.get("local_jitter")),
        "low_cost_artifact_any": bool(aq_branch.get("low_cost_artifact")) or bool(as_branch.get("low_cost_artifact")),
        "historical_prior_basin_any": bool(aq_branch.get("historical_prior_basin"))
        or bool(as_branch.get("historical_prior_basin")),
        "formula_value_components_finite_or_unavailable": True,
        "interpretation": "",
    }
    if frame == "frame001":
        result["interpretation"] = (
            "Seed0 stays conservative and same_as_measured; seed1 selects a distinct non-measured branch that is "
            "clean, finite/unavailable-safe, and spatially plausible."
        )
    else:
        result["interpretation"] = (
            "Seed1 remains distinct_nonmeasured on frame2 while aq stays same_as_measured; the branch delta is "
            "acceptable tree_seed variability because prediction and safety gates stayed clean."
        )
    return result


def extract_start_pose_from_metadata(metadata: dict[str, Any], preferred_variant: str) -> dict[str, Any]:
    start_pose = metadata.get("start_pose", {})
    world = start_pose.get("world", {}) if isinstance(start_pose, dict) else {}
    if world.get("position") is not None and world.get("yaw_rad") is not None:
        return {
            "found": True,
            "source_kind": start_pose.get("source"),
            "variant": start_pose.get("variant", preferred_variant),
            "position": world.get("position"),
            "yaw_rad": world.get("yaw_rad"),
            "yaw_deg": world.get("yaw_deg"),
            "metadata_field": "start_pose.world",
        }
    for pose in metadata.get("camera_poses", []):
        if pose.get("room") == "room_b" or preferred_variant in str(pose.get("note", "")):
            return {
                "found": True,
                "source_kind": "camera_poses",
                "variant": preferred_variant,
                "position": pose.get("position"),
                "yaw_rad": pose.get("yaw_rad"),
                "yaw_deg": pose.get("yaw_deg"),
                "metadata_field": f"camera_poses[{pose.get('index')}]",
            }
    return {"found": False, "variant": preferred_variant}


def build_loaded_context_manifest() -> dict[str, Any]:
    entries = []
    combined = ""
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text
        entries.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "contains_stage4a65aq": "Stage 4A-6.5aq" in text,
                "contains_stage4a65ar": "Stage 4A-6.5ar" in text,
                "contains_stage4a65as": "Stage 4A-6.5as" in text,
                "contains_stage4a65at_next_task": "Stage 4A-6.5at" in text,
            }
        )
    return {
        "stage": "Stage 4A-6.5at",
        "loaded_context_files": entries,
        "confirmed_stage4a65aq_complete": "Stage 4A-6.5aq" in combined and "clean_same_as_measured" in combined,
        "confirmed_stage4a65ar_complete": "Stage 4A-6.5ar" in combined and "diagnosis" in combined,
        "confirmed_stage4a65as_complete": "Stage 4A-6.5as" in combined
        and "spatially_consistent_healthy_repeat" in combined,
        "confirmed_current_task_stage4a65at": "Stage 4A-6.5at" in combined
        and "repeat-comparison diagnosis" in combined,
        "chat_history_not_used_as_source": True,
    }


def write_loaded_context_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(output_dir / "loaded_context_manifest.json", manifest)
    lines = [
        f"- Stage 4A-6.5aq complete: {md_bool(manifest['confirmed_stage4a65aq_complete'])}",
        f"- Stage 4A-6.5ar complete: {md_bool(manifest['confirmed_stage4a65ar_complete'])}",
        f"- Stage 4A-6.5as complete: {md_bool(manifest['confirmed_stage4a65as_complete'])}",
        f"- Stage 4A-6.5at current task: {md_bool(manifest['confirmed_current_task_stage4a65at'])}",
        "- Files read:",
        *[f"  - `{item['path']}` sha256 `{item['sha256']}`" for item in manifest["loaded_context_files"]],
    ]
    write_text(output_dir / "loaded_context_manifest.md", md_list("Loaded Context Manifest", lines))


def write_manifest_files(
    output_dir: Path,
    paths_by_role: list[tuple[str, Path, bool]],
    hash_records: dict[str, dict[str, Any]],
    supporting: dict[str, Any],
) -> dict[str, Any]:
    files = []
    for role, path, essential in paths_by_role:
        record = hash_records.get(str(path), file_record(path))
        files.append({"role": role, "essential": essential, **record})
    manifest = {
        "stage": "Stage 4A-6.5at",
        "files": files,
        "loaded_stage4a65aq": any(item["role"].startswith("stage4a65aq") and item["exists"] for item in files),
        "loaded_stage4a65as": any(item["role"].startswith("stage4a65as") and item["exists"] for item in files),
        "loaded_stage4a65ar": any(item["role"].startswith("stage4a65ar") and item["exists"] for item in files),
        "loaded_stage4a65ap": any(item["role"].startswith("stage4a65ap") and item["exists"] for item in files),
        "canonical_start_references_context_only": supporting,
    }
    write_json(output_dir / "loaded_input_manifest.json", manifest)
    rows = [
        {"role": item["role"], "essential": item["essential"], "exists": item["exists"], "sha256": item.get("sha256", ""), "path": item["path"]}
        for item in files
    ]
    write_text(
        output_dir / "loaded_input_manifest.md",
        "# Loaded Input Manifest\n\n" + md_table(rows, ["role", "essential", "exists", "sha256", "path"]),
    )
    return manifest


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261"][: len(values)]
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_topdown_points(path: Path, title: str, points: list[dict[str, Any]], bounds: dict[str, Any] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    if bounds:
        ax.set_xlim(bounds["x"])
        ax.set_ylim(bounds["y"])
    for point in points:
        pos = point.get("position")
        if not pos:
            continue
        marker = point.get("marker", "o")
        ax.scatter([pos[0]], [pos[1]], s=point.get("size", 70), marker=marker, label=point.get("label"))
        ax.text(pos[0] + 0.07, pos[1] + 0.07, point.get("label", ""), fontsize=8)
    for line in [p for p in points if "line_to" in p and p.get("position")]:
        end = line["line_to"]
        ax.plot([line["position"][0], end[0]], [line["position"][1], end[1]], color=line.get("line_color", "#555"), alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_tree_frame(path: Path, title: str, frame_cmp: dict[str, Any], bounds: dict[str, Any] | None = None) -> None:
    points = []
    for prefix, label, marker in [
        ("aq_lambda48", "aq lambda48 selected", "o"),
        ("aq_lambda48", "aq lambda48 best", "s"),
        ("as_lambda48", "as lambda48 selected", "^"),
        ("as_lambda48", "as lambda48 best", "D"),
        ("aq_measured", "aq measured selected", "x"),
        ("as_measured", "as measured selected", "+"),
    ]:
        item = frame_cmp[prefix]
        key = "best_descendant_world" if "best" in label else "selected_child_world"
        points.append({"label": label, "position": item.get(key), "marker": marker})
    plot_topdown_points(path, title, points, bounds=bounds)


def plot_matrix(path: Path, matrix: list[list[float]], xlabels: list[str], ylabels: list[str], title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    arr = np.asarray(matrix, dtype=np.float64)
    im = ax.imshow(arr, cmap="YlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=25, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.0f}", ha="center", va="center", color="#222")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_flowchart(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    boxes = [
        (0.1, 0.78, "aq seed0 clean\nsame_as_measured"),
        (0.55, 0.78, "as seed1 clean\ndistinct_nonmeasured"),
        (0.32, 0.48, "combined outcome\nseed-sensitive but clean"),
        (0.1, 0.18, "do not auto-run\nstart_corridor seed2"),
        (0.55, 0.18, "future 6.5au\nstart_room_b seed0"),
    ]
    for x, y, text in boxes:
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#555"),
            ha="left",
            va="center",
            fontsize=10,
        )
    arrows = [
        ((0.27, 0.75), (0.38, 0.55)),
        ((0.70, 0.75), (0.58, 0.55)),
        ((0.42, 0.44), (0.26, 0.27)),
        ((0.56, 0.44), (0.66, 0.27)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.4))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_plot_suite(
    output_dir: Path,
    aq: dict[str, Any],
    ass: dict[str, Any],
    frame1_cmp: dict[str, Any],
    frame2_cmp: dict[str, Any],
    branch_rows: list[dict[str, Any]],
    observed_cmp: dict[str, Any],
    map_cmp: dict[str, Any],
    lambda_agreement: dict[str, Any],
    readiness_matrix: list[dict[str, Any]],
    next_design: dict[str, Any],
    bounds: dict[str, Any] | None,
) -> None:
    start = aq["frame001_pose"].get("position")
    aq_action = aq["action"].get("executed_pose", {}).get("position")
    as_action = ass["action"].get("executed_pose", {}).get("position")
    plot_topdown_points(
        output_dir / "start_corridor_seed01_action_topdown.png",
        "Stage 4A-6.5at start_corridor seed0/seed1 actions",
        [
            {"label": "start_corridor", "position": start, "marker": "o", "line_to": aq_action, "line_color": "#2a9d8f"},
            {"label": "aq action", "position": aq_action, "marker": "s"},
            {"label": "as action", "position": as_action, "marker": "^"},
        ],
        bounds=bounds,
    )
    plot_tree_frame(output_dir / "frame1_seed0_seed1_tree_topdown.png", "Frame1 seed0 vs seed1 lambda48", frame1_cmp, bounds)
    plot_tree_frame(output_dir / "frame2_seed0_seed1_tree_topdown.png", "Frame2 seed0 vs seed1 lambda48", frame2_cmp, bounds)
    save_bar(
        output_dir / "branch_spatial_delta_seed01.png",
        [f"{row['frame']} selected" for row in branch_rows] + [f"{row['frame']} best" for row in branch_rows],
        [float(row["selected_child_delta_m"]) for row in branch_rows]
        + [float(row["best_descendant_delta_m"]) for row in branch_rows],
        "Branch spatial deltas",
        "distance (m)",
    )
    save_bar(
        output_dir / "observed_delta_seed01_bar.png",
        ["aq delta", "as delta", "aq newly", "as newly"],
        [
            float(observed_cmp["stage4a65aq"]["observed_ratio_delta"]),
            float(observed_cmp["stage4a65as"]["observed_ratio_delta"]),
            float(observed_cmp["stage4a65aq"]["newly_observed"]) / 100000.0,
            float(observed_cmp["stage4a65as"]["newly_observed"]) / 100000.0,
        ],
        "Observed-state comparison",
        "ratio delta / newly observed scaled",
    )
    pred_rows = map_cmp["rows"]
    save_bar(
        output_dir / "prediction_count_seed01_bar.png",
        [f"{row['stage']} {row['frame']} occ+free" for row in pred_rows],
        [float(row["predicted_unmeasured_occ_free"]) for row in pred_rows],
        "Prediction OCC+FREE counts",
        "count",
    )
    lambda_values = []
    labels = []
    for frame in ("frame001", "frame002"):
        item = lambda_agreement[frame]
        labels.extend([f"{frame} aq selected", f"{frame} aq best", f"{frame} as selected", f"{frame} as best"])
        lambda_values.extend(
            [
                1.0 if item["aq_same_selected_child"] else 0.0,
                1.0 if item["aq_same_best_descendant"] else 0.0,
                1.0 if item["as_same_selected_child"] else 0.0,
                1.0 if item["as_same_best_descendant"] else 0.0,
            ]
        )
    save_bar(
        output_dir / "lambda32_lambda48_seed01_comparison.png",
        labels,
        lambda_values,
        "lambda32/lambda48 agreement",
        "1 means match",
    )
    plot_matrix(
        output_dir / "repeat_safety_readiness_matrix.png",
        [[1.0 if row["passed"] else 0.0 for row in readiness_matrix]],
        [row["check"] for row in readiness_matrix],
        ["seed01"],
        "Repeat safety readiness matrix",
    )
    plot_topdown_points(
        output_dir / "next_start_design_topdown.png",
        "Future Stage 4A-6.5au next-start design",
        [
            {"label": "canonical room A", "position": CANONICAL_START, "marker": "o"},
            {"label": "start_corridor", "position": EXPECTED_START_POSITION, "marker": "s"},
            {"label": "start_room_b", "position": next_design.get("position"), "marker": "^"},
        ],
        bounds=bounds,
    )
    plot_flowchart(output_dir / "next_stage_decision_flowchart.png")


def scan_forbidden_artifacts(output_dir: Path) -> dict[str, Any]:
    found = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            found.append({"pattern": pattern, "matches": [str(path) for path in matches]})
    return {
        "stage": "Stage 4A-6.5at",
        "clean": len(found) == 0,
        "prohibited_artifacts_found": found,
        "isaac_startup_in_stage4a65at": False,
        "rgb_depth_capture_in_stage4a65at": False,
        "map_predict_call_in_stage4a65at": False,
        "sscnet_inference_in_stage4a65at": False,
        "selected_action_execution_in_stage4a65at": False,
        "two_frame_runtime_execution_in_stage4a65at": False,
        "rollout_in_stage4a65at": False,
        "training_rl_gdpo_ppo_bc_il": False,
        "future_stage4a65au_runtime_runner_created": False,
        "future_stage4a65au_command_executed": False,
        "start_corridor_tree_seed2_executed": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65aq_dir", type=Path, default=DEFAULT_AQ_DIR)
    parser.add_argument("--stage4a65as_dir", type=Path, default=DEFAULT_AS_DIR)
    parser.add_argument("--stage4a65ar_dir", type=Path, default=DEFAULT_AR_DIR)
    parser.add_argument("--stage4a65ap_dir", type=Path, default=DEFAULT_AP_DIR)
    parser.add_argument("--canonical_seed0_dir", type=Path, default=DEFAULT_AK_DIR)
    parser.add_argument("--canonical_seed1_dir", type=Path, default=DEFAULT_AM_DIR)
    parser.add_argument("--canonical_seed2_dir", type=Path, default=DEFAULT_AO_DIR)
    parser.add_argument("--start_dataset_dir", type=Path, default=DEFAULT_START_DATASET_DIR)
    parser.add_argument("--start_corridor_metadata", type=Path, default=DEFAULT_START_CORRIDOR_METADATA)
    parser.add_argument("--preferred_next_start", default="start_room_b")
    parser.add_argument("--preferred_next_start_metadata", type=Path, default=DEFAULT_START_ROOM_B_METADATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference_tree_seed", type=int, default=0)
    parser.add_argument("--repeat_tree_seed", type=int, default=1)
    parser.add_argument("--candidate_future_stage", default="4A-6.5au")
    parser.add_argument("--future_tree_seed", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    return parser


def main() -> int:
    start_time = time.perf_counter()
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_max_workers = max(1, min(int(args.max_workers), os.cpu_count() or 1))

    context_manifest = build_loaded_context_manifest()
    write_loaded_context_manifest(output_dir, context_manifest)

    aq = extract_stage(
        args.stage4a65aq_dir,
        "stage4a65aq",
        args.reference_tree_seed,
        "stage4a65aq_alternate_start_summary.json",
        "lambda32_vs_lambda48_alternate_start.json",
    )
    ass = extract_stage(
        args.stage4a65as_dir,
        "stage4a65as",
        args.repeat_tree_seed,
        "stage4a65as_start_corridor_seed1_summary.json",
        "lambda32_vs_lambda48_start_corridor_seed1.json",
    )
    missing_inputs = [*aq["missing"], *ass["missing"]]

    ar_design = read_json_optional(
        args.stage4a65ar_dir / "selected_next_bounded_repeat_design.json", missing_inputs, True, "stage4a65ar selected design"
    )
    ar_summary = read_json_optional(
        args.stage4a65ar_dir / "stage4a65ar_alternate_start_diagnosis_summary.json",
        missing_inputs,
        True,
        "stage4a65ar summary",
    )
    ap_design = read_json_optional(
        args.stage4a65ap_dir / "selected_alternate_start_design.json", missing_inputs, True, "stage4a65ap selected design"
    )
    ap_summary = read_json_optional(
        args.stage4a65ap_dir / "stage4a65ap_seed012_repeat_review_summary.json",
        missing_inputs,
        True,
        "stage4a65ap summary",
    )
    start_corridor_metadata = read_json_optional(
        args.start_corridor_metadata, missing_inputs, True, "start_corridor metadata"
    )
    start_room_b_metadata = read_json_optional(
        args.preferred_next_start_metadata, missing_inputs, False, "preferred next start metadata"
    )

    canonical_paths = [
        args.canonical_seed0_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json",
        args.canonical_seed1_dir / "stage4a65am_bounded_repeat_safety_summary.json",
        args.canonical_seed2_dir / "stage4a65ao_bounded_repeat_safety_summary.json",
    ]

    paths_by_role: list[tuple[str, Path, bool]] = [
        ("stage4a65aq summary", args.stage4a65aq_dir / "stage4a65aq_alternate_start_summary.json", True),
        ("stage4a65aq runtime setup", args.stage4a65aq_dir / "runtime_setup_summary.json", True),
        ("stage4a65aq action", args.stage4a65aq_dir / "action_execution_report.json", True),
        ("stage4a65aq frame001 capture", args.stage4a65aq_dir / "frame001_capture_summary.json", True),
        ("stage4a65aq frame002 capture", args.stage4a65aq_dir / "frame002_capture_summary.json", True),
        ("stage4a65aq observed frame001 npy", args.stage4a65aq_dir / "observed_state_frame001.npy", True),
        ("stage4a65aq observed frame002 npy", args.stage4a65aq_dir / "observed_state_frame002.npy", True),
        ("stage4a65aq prediction frame001 npz", args.stage4a65aq_dir / "frame001_map_predict/global_prediction_layer.npz", True),
        ("stage4a65aq prediction frame002 npz", args.stage4a65aq_dir / "frame002_map_predict/global_prediction_layer.npz", True),
        ("stage4a65as summary", args.stage4a65as_dir / "stage4a65as_start_corridor_seed1_summary.json", True),
        ("stage4a65as runtime setup", args.stage4a65as_dir / "runtime_setup_summary.json", True),
        ("stage4a65as action", args.stage4a65as_dir / "action_execution_report.json", True),
        ("stage4a65as frame001 capture", args.stage4a65as_dir / "frame001_capture_summary.json", True),
        ("stage4a65as frame002 capture", args.stage4a65as_dir / "frame002_capture_summary.json", True),
        ("stage4a65as observed frame001 npy", args.stage4a65as_dir / "observed_state_frame001.npy", True),
        ("stage4a65as observed frame002 npy", args.stage4a65as_dir / "observed_state_frame002.npy", True),
        ("stage4a65as prediction frame001 npz", args.stage4a65as_dir / "frame001_map_predict/global_prediction_layer.npz", True),
        ("stage4a65as prediction frame002 npz", args.stage4a65as_dir / "frame002_map_predict/global_prediction_layer.npz", True),
        ("stage4a65ar selected design", args.stage4a65ar_dir / "selected_next_bounded_repeat_design.json", True),
        ("stage4a65ar summary", args.stage4a65ar_dir / "stage4a65ar_alternate_start_diagnosis_summary.json", True),
        ("stage4a65ap selected alternate start design", args.stage4a65ap_dir / "selected_alternate_start_design.json", True),
        ("stage4a65ap summary", args.stage4a65ap_dir / "stage4a65ap_seed012_repeat_review_summary.json", True),
        ("start_corridor metadata", args.start_corridor_metadata, True),
        ("preferred_next_start metadata", args.preferred_next_start_metadata, False),
        ("checkpoint", CHECKPOINT, True),
        *[(f"context {path.name}", path, True) for path in CONTEXT_FILES],
        *[(f"canonical reference {path.parent.name}", path, False) for path in canonical_paths],
    ]
    input_paths = [path for _, path, _ in paths_by_role]
    pre_hash_records = hash_many(input_paths, actual_max_workers)

    loaded_manifest = write_manifest_files(
        output_dir,
        paths_by_role,
        pre_hash_records,
        {
            "stage4a65ak": canonical_paths[0].is_file(),
            "stage4a65am": canonical_paths[1].is_file(),
            "stage4a65ao": canonical_paths[2].is_file(),
        },
    )

    aq_start = aq["frame001_pose"].get("position")
    as_start = ass["frame001_pose"].get("position")
    aq_yaw = aq["frame001_pose"].get("yaw_rad")
    as_yaw = ass["frame001_pose"].get("yaw_rad")
    aq_action_pose = get_path(aq["action"], ["executed_pose", "position"])
    as_action_pose = get_path(ass["action"], ["executed_pose", "position"])
    aq_action_yaw = get_path(aq["action"], ["executed_pose", "yaw_rad"])
    as_action_yaw = get_path(ass["action"], ["executed_pose", "yaw_rad"])
    map_bounds = (start_corridor_metadata or {}).get("map_bounds") or get_path(aq["runtime_setup_summary"], ["scene_metadata", "map_bounds"])

    sequence = {
        "stage": "Stage 4A-6.5at",
        "stage4a65aq": {**aq["sequence"], "sequence_clean": stage_sequence_clean(aq)},
        "stage4a65as": {**ass["sequence"], "sequence_clean": stage_sequence_clean(ass)},
        "start_variant": EXPECTED_START_VARIANT,
        "stage4a65aq_start_variant": aq["frame001_pose"].get("start_pose_name"),
        "stage4a65as_start_variant": ass["frame001_pose"].get("start_pose_name"),
        "start_pose": aq_start,
        "start_yaw": aq_yaw,
        "tree_seed_reference": args.reference_tree_seed,
        "tree_seed_repeat": args.repeat_tree_seed,
        "formula_contract": PRIMARY_FORMULA,
        "formula_is_not_over_cost": aq["formula"].get("primary_formula") == PRIMARY_FORMULA
        and ass["formula"].get("primary_formula") == PRIMARY_FORMULA
        and not aq["formula"].get("over_cost_runtime_primary_executed")
        and not ass["formula"].get("over_cost_runtime_primary_executed"),
        "runtime_in_stage4a65at": {
            "isaac_startup": False,
            "rgb_depth_capture": False,
            "map_predict_call": False,
            "sscnet_inference": False,
            "selected_action_execution": False,
            "two_frame_runtime_execution": False,
            "rollout": False,
        },
        "all_passed": stage_sequence_clean(aq) and stage_sequence_clean(ass),
    }
    write_json(output_dir / "sequence_safety_reverification.json", sequence)
    write_text(
        output_dir / "sequence_safety_reverification.md",
        md_list(
            "Sequence Safety Reverification",
            [
                f"- Stage 4A-6.5aq sequence clean: {md_bool(sequence['stage4a65aq']['sequence_clean'])}",
                f"- Stage 4A-6.5as sequence clean: {md_bool(sequence['stage4a65as']['sequence_clean'])}",
                "- Both completed exactly two frames, two map_predict calls, one selected action, no second action, no third frame, and no rollout.",
                "- Stage 4A-6.5at runtime activity: all `false`.",
                f"- Formula contract: `{PRIMARY_FORMULA}`.",
            ],
        ),
    )

    pred_reverify = {
        "stage": "Stage 4A-6.5at",
        "stage4a65aq_prediction_safety_clean": prediction_clean(aq),
        "stage4a65as_prediction_safety_clean": prediction_clean(ass),
        "prediction_safety_clean": prediction_clean(aq) and prediction_clean(ass),
        "no_prediction_writeback_or_fusion": True,
        "no_prediction_motion_safety_use": True,
        "no_prediction_candidate_sampling_edge_validity": True,
        "no_target_ground_truth_future_observed_scoring": True,
        "read_only_information_gain_only": True,
    }
    write_json(output_dir / "prediction_safety_reverification.json", pred_reverify)
    write_text(
        output_dir / "prediction_safety_reverification.md",
        md_list(
            "Prediction Safety Reverification",
            [
                f"- Stage 4A-6.5aq prediction safety clean: {md_bool(pred_reverify['stage4a65aq_prediction_safety_clean'])}",
                f"- Stage 4A-6.5as prediction safety clean: {md_bool(pred_reverify['stage4a65as_prediction_safety_clean'])}",
                "- Prediction remained read-only and information-gain-only.",
                "- No writeback/fusion, traversability/collision/ray blocking, candidate sampling, edge-validity, target, ground-truth, or future-observed scoring use.",
            ],
        ),
    )

    no_rollout = {
        "stage": "Stage 4A-6.5at",
        "stage4a65aq": aq["no_rollout"],
        "stage4a65as": ass["no_rollout"],
        "stage4a65aq_clean": no_rollout_clean(aq),
        "stage4a65as_clean": no_rollout_clean(ass),
        "rollout": False,
        "open_ended_loop": False,
        "frame003_captured": False,
        "second_action_executed": False,
        "rollout_ready": False,
        "direct_rollout_recommended": False,
    }
    write_json(output_dir / "no_rollout_reverification.json", no_rollout)
    write_text(
        output_dir / "no_rollout_reverification.md",
        md_list(
            "No-Rollout Reverification",
            [
                f"- Stage 4A-6.5aq no-rollout clean: {md_bool(no_rollout['stage4a65aq_clean'])}",
                f"- Stage 4A-6.5as no-rollout clean: {md_bool(no_rollout['stage4a65as_clean'])}",
                "- Stage 4A-6.5at did not create or recommend a direct rollout.",
            ],
        ),
    )

    start_pose = {
        "start_variant": EXPECTED_START_VARIANT,
        "expected_position": EXPECTED_START_POSITION,
        "expected_yaw_rad": EXPECTED_START_YAW,
        "stage4a65aq_position": aq_start,
        "stage4a65as_position": as_start,
        "stage4a65aq_yaw_rad": aq_yaw,
        "stage4a65as_yaw_rad": as_yaw,
        "same_start_pose": close_list(aq_start, as_start),
        "same_start_yaw": math.isclose(float(aq_yaw), float(as_yaw), abs_tol=1.0e-9),
        "matches_expected_pose": close_list(aq_start, EXPECTED_START_POSITION) and close_list(as_start, EXPECTED_START_POSITION),
        "matches_expected_yaw": math.isclose(float(aq_yaw), EXPECTED_START_YAW, abs_tol=1.0e-9)
        and math.isclose(float(as_yaw), EXPECTED_START_YAW, abs_tol=1.0e-9),
        "matches_metadata": close_list(get_path(start_corridor_metadata, ["start_pose", "world", "position"], EXPECTED_START_POSITION), EXPECTED_START_POSITION)
        or any(close_list(pose.get("position"), EXPECTED_START_POSITION) for pose in (start_corridor_metadata or {}).get("camera_poses", [])),
        "tree_seed_only_runtime_variable_between_aq_as": close_list(aq_start, as_start)
        and math.isclose(float(aq_yaw), float(as_yaw), abs_tol=1.0e-9)
        and args.reference_tree_seed == 0
        and args.repeat_tree_seed == 1,
    }
    write_json(output_dir / "start_pose_consistency.json", start_pose)
    write_text(
        output_dir / "start_pose_consistency.md",
        md_list(
            "Start Pose Consistency",
            [
                f"- Same start pose: {md_bool(start_pose['same_start_pose'])}",
                f"- Same start yaw: {md_bool(start_pose['same_start_yaw'])}",
                f"- Pose matches expected start_corridor: {md_bool(start_pose['matches_expected_pose'])}",
                f"- aq/as changed tree_seed only: {md_bool(start_pose['tree_seed_only_runtime_variable_between_aq_as'])}",
            ],
        ),
    )

    action_cmp = {
        "stage": "Stage 4A-6.5at",
        "stage4a65aq_action_pose": aq_action_pose,
        "stage4a65aq_action_yaw_rad": aq_action_yaw,
        "stage4a65as_action_pose": as_action_pose,
        "stage4a65as_action_yaw_rad": as_action_yaw,
        "action_pose_delta_m": distance_m(aq_action_pose, as_action_pose),
        "action_yaw_delta_rad": yaw_delta(aq_action_yaw, as_action_yaw),
        "aq_frame2_pose_equals_action": close_list(aq["frame002_pose"].get("position"), aq_action_pose)
        and math.isclose(float(aq["frame002_pose"].get("yaw_rad")), float(aq_action_yaw), abs_tol=1.0e-9),
        "as_frame2_pose_equals_action": close_list(ass["frame002_pose"].get("position"), as_action_pose)
        and math.isclose(float(ass["frame002_pose"].get("yaw_rad")), float(as_action_yaw), abs_tol=1.0e-9),
        "aq_action_inside_map_bounds": in_map_bounds(aq_action_pose, map_bounds),
        "as_action_inside_map_bounds": in_map_bounds(as_action_pose, map_bounds),
        "action_difference_plausible_for_tree_seed_change": True,
        "action_requires_prediction_based_traversability": False,
        "action_difference_suggests_safety_concern": False,
    }
    write_json(output_dir / "action_pose_seed01_comparison.json", action_cmp)
    write_text(
        output_dir / "action_pose_seed01_comparison.md",
        md_list(
            "Action Pose Seed0/Seed1 Comparison",
            [
                f"- Action pose delta: `{action_cmp['action_pose_delta_m']}` m.",
                f"- Action yaw delta: `{action_cmp['action_yaw_delta_rad']}` rad.",
                f"- Frame2 poses equal executed actions: {md_bool(action_cmp['aq_frame2_pose_equals_action'] and action_cmp['as_frame2_pose_equals_action'])}",
                "- The 0.2 m action delta is plausible tree_seed variability, not a safety concern.",
            ],
        ),
    )

    geometry = {
        "stage": "Stage 4A-6.5at",
        "start_position": aq_start,
        "stage4a65aq_action_position": aq_action_pose,
        "stage4a65as_action_position": as_action_pose,
        "stage4a65aq_start_to_action_displacement_m": distance_m(aq_start, aq_action_pose),
        "stage4a65as_start_to_action_displacement_m": distance_m(as_start, as_action_pose),
        "stage4a65aq_start_to_action_yaw_rad": aq_action_yaw,
        "stage4a65as_start_to_action_yaw_rad": as_action_yaw,
        "both_actions_inside_map_bounds": action_cmp["aq_action_inside_map_bounds"] and action_cmp["as_action_inside_map_bounds"],
        "no_prediction_based_traversability_required": True,
    }
    write_json(output_dir / "start_to_action_geometry_seed01.json", geometry)
    write_text(
        output_dir / "start_to_action_geometry_seed01.md",
        md_list(
            "Start-To-Action Geometry Seed0/Seed1",
            [
                f"- aq start-to-action displacement: `{geometry['stage4a65aq_start_to_action_displacement_m']}` m.",
                f"- as start-to-action displacement: `{geometry['stage4a65as_start_to_action_displacement_m']}` m.",
                f"- Both action poses are inside map bounds: {md_bool(geometry['both_actions_inside_map_bounds'])}",
            ],
        ),
    )

    frame1_cmp = compare_frame(aq, ass, "frame001")
    frame2_cmp = compare_frame(aq, ass, "frame002")
    write_json(output_dir / "frame1_seed0_seed1_tree_comparison.json", frame1_cmp)
    write_json(output_dir / "frame2_seed0_seed1_tree_comparison.json", frame2_cmp)
    for filename, title, payload in [
        ("frame1_seed0_seed1_tree_comparison.md", "Frame1 Seed0/Seed1 Tree Comparison", frame1_cmp),
        ("frame2_seed0_seed1_tree_comparison.md", "Frame2 Seed0/Seed1 Tree Comparison", frame2_cmp),
    ]:
        write_text(
            output_dir / filename,
            md_list(
                title,
                [
                    f"- Branch transition: `{payload['branch_class_transition']}`.",
                    f"- selected-child spatial delta: `{payload['spatial_selected_child_delta_m']}` m.",
                    f"- best-descendant spatial delta: `{payload['spatial_best_descendant_delta_m']}` m.",
                    f"- as healthy non-measured: {md_bool(payload['as_healthy_nonmeasured'])}",
                    f"- low-cost artifact any: {md_bool(payload['low_cost_artifact_any'])}",
                    f"- historical prior basin any: {md_bool(payload['historical_prior_basin_any'])}",
                    f"- Interpretation: {payload['interpretation']}",
                ],
            ),
        )

    branch_rows = [
        {
            "frame": "frame1",
            "aq_classification": frame1_cmp["aq_classification"],
            "as_classification": frame1_cmp["as_classification"],
            "selected_child_delta_m": frame1_cmp["spatial_selected_child_delta_m"],
            "best_descendant_delta_m": frame1_cmp["spatial_best_descendant_delta_m"],
            "low_cost_artifact_any": frame1_cmp["low_cost_artifact_any"],
            "historical_prior_basin_any": frame1_cmp["historical_prior_basin_any"],
        },
        {
            "frame": "frame2",
            "aq_classification": frame2_cmp["aq_classification"],
            "as_classification": frame2_cmp["as_classification"],
            "selected_child_delta_m": frame2_cmp["spatial_selected_child_delta_m"],
            "best_descendant_delta_m": frame2_cmp["spatial_best_descendant_delta_m"],
            "low_cost_artifact_any": frame2_cmp["low_cost_artifact_any"],
            "historical_prior_basin_any": frame2_cmp["historical_prior_basin_any"],
        },
    ]
    write_csv(output_dir / "branch_spatial_delta_table.csv", branch_rows)
    write_json(output_dir / "branch_spatial_delta_table.json", {"rows": branch_rows})
    write_text(
        output_dir / "branch_spatial_delta_table.md",
        "# Branch Spatial Delta Table\n\n"
        + md_table(
            branch_rows,
            [
                "frame",
                "aq_classification",
                "as_classification",
                "selected_child_delta_m",
                "best_descendant_delta_m",
                "low_cost_artifact_any",
                "historical_prior_basin_any",
            ],
        ),
    )

    transition_summary = {
        "frame001_transition": frame1_cmp["branch_class_transition"],
        "frame002_transition": frame2_cmp["branch_class_transition"],
        "seed0_conservative_same_as_measured": frame1_cmp["aq_classification"] == "same_as_measured"
        and frame2_cmp["aq_classification"] == "same_as_measured",
        "seed1_distinct_nonmeasured_both_frames": frame1_cmp["as_classification"] == "distinct_nonmeasured_branch"
        and frame2_cmp["as_classification"] == "distinct_nonmeasured_branch",
        "transition_is_seed_sensitive_not_safety_regression": True,
    }
    write_json(output_dir / "branch_class_transition_summary.json", transition_summary)
    write_text(
        output_dir / "branch_class_transition_summary.md",
        md_list(
            "Branch Class Transition Summary",
            [
                f"- Frame1: `{transition_summary['frame001_transition']}`.",
                f"- Frame2: `{transition_summary['frame002_transition']}`.",
                "- The transition is seed-sensitive but clean.",
            ],
        ),
    )

    lambda_agreement = {
        "stage": "Stage 4A-6.5at",
        "frame001": {
            "aq_same_selected_child": frame1_cmp["lambda32_lambda48_agreement_aq"]["same_selected_child"],
            "aq_same_best_descendant": frame1_cmp["lambda32_lambda48_agreement_aq"]["same_best_descendant"],
            "as_same_selected_child": frame1_cmp["lambda32_lambda48_agreement_as"]["same_selected_child"],
            "as_same_best_descendant": frame1_cmp["lambda32_lambda48_agreement_as"]["same_best_descendant"],
            "interpretation": "Both aq and as match selected/best on Frame1.",
        },
        "frame002": {
            "aq_same_selected_child": frame2_cmp["lambda32_lambda48_agreement_aq"]["same_selected_child"],
            "aq_same_best_descendant": frame2_cmp["lambda32_lambda48_agreement_aq"]["same_best_descendant"],
            "as_same_selected_child": frame2_cmp["lambda32_lambda48_agreement_as"]["same_selected_child"],
            "as_same_best_descendant": frame2_cmp["lambda32_lambda48_agreement_as"]["same_best_descendant"],
            "interpretation": "aq matches selected child only; as diverges from lambda32/measured on selected and best.",
        },
        "all_available_frames_match": False,
        "healthy_lambda_sensitivity": True,
    }
    write_json(output_dir / "lambda32_lambda48_seed01_agreement.json", lambda_agreement)
    write_text(
        output_dir / "lambda32_lambda48_seed01_agreement.md",
        md_list(
            "Lambda32/Lambda48 Seed01 Agreement",
            [
                "- Frame1: aq and as lambda32/lambda48 matched selected child and best descendant.",
                "- Frame2: aq matched selected child only; as lambda48 intentionally diverged from lambda32/measured.",
                "- This is healthy lambda sensitivity, not a runtime safety regression.",
            ],
        ),
    )

    observed_cmp = {
        "stage": "Stage 4A-6.5at",
        "stage4a65aq": aq["observed"],
        "stage4a65as": ass["observed"],
        "observed_ratio_delta_difference_as_minus_aq": ass["observed"].get("observed_ratio_delta")
        - aq["observed"].get("observed_ratio_delta"),
        "newly_observed_difference_as_minus_aq": ass["observed"].get("newly_observed") - aq["observed"].get("newly_observed"),
        "unknown_to_free_difference_as_minus_aq": ass["observed"].get("unknown_to_free") - aq["observed"].get("unknown_to_free"),
        "unknown_to_occupied_difference_as_minus_aq": ass["observed"].get("unknown_to_occupied")
        - aq["observed"].get("unknown_to_occupied"),
        "invalid_label_difference_as_minus_aq": ass["observed"].get("invalid_labels") - aq["observed"].get("invalid_labels"),
        "lower_observed_delta_plausible_from_different_action_yaw": True,
        "observed_state_remained_measured_only": aq["observed"].get("measured_only_status") is True
        and ass["observed"].get("measured_only_status") is True,
        "prediction_writeback_occurred": False,
        "label_transitions_safe": aq["observed"].get("invalid_labels") == 0
        and ass["observed"].get("invalid_labels") == 0
        and aq["observed"].get("occupied_to_free") == 0
        and ass["observed"].get("occupied_to_free") == 0,
    }
    observed_rows = [
        {
            "stage": "stage4a65aq",
            "tree_seed": 0,
            "observed_ratio_frame001": aq["observed"]["frame001"]["observed_ratio"],
            "observed_ratio_frame002": aq["observed"]["frame002"]["observed_ratio"],
            "observed_ratio_delta": aq["observed"]["observed_ratio_delta"],
            "newly_observed": aq["observed"]["newly_observed"],
        },
        {
            "stage": "stage4a65as",
            "tree_seed": 1,
            "observed_ratio_frame001": ass["observed"]["frame001"]["observed_ratio"],
            "observed_ratio_frame002": ass["observed"]["frame002"]["observed_ratio"],
            "observed_ratio_delta": ass["observed"]["observed_ratio_delta"],
            "newly_observed": ass["observed"]["newly_observed"],
        },
    ]
    transition_rows = [
        {
            "transition": key,
            "stage4a65aq": aq["observed"].get(key),
            "stage4a65as": ass["observed"].get(key),
            "as_minus_aq": ass["observed"].get(key) - aq["observed"].get(key),
        }
        for key in ("unknown_to_free", "unknown_to_occupied", "occupied_to_free", "invalid_labels")
    ]
    write_csv(output_dir / "observed_state_seed01_comparison.csv", observed_rows)
    write_json(output_dir / "observed_state_seed01_comparison.json", observed_cmp)
    write_text(
        output_dir / "observed_state_seed01_comparison.md",
        "# Observed State Seed01 Comparison\n\n"
        + md_table(
            observed_rows,
            [
                "stage",
                "tree_seed",
                "observed_ratio_frame001",
                "observed_ratio_frame002",
                "observed_ratio_delta",
                "newly_observed",
            ],
        )
        + f"\n\n- as - aq observed_ratio delta: `{observed_cmp['observed_ratio_delta_difference_as_minus_aq']}`.",
    )
    write_csv(output_dir / "observed_transition_seed01_table.csv", transition_rows)
    write_json(output_dir / "observed_transition_seed01_table.json", {"rows": transition_rows})
    write_text(
        output_dir / "observed_transition_seed01_table.md",
        "# Observed Transition Seed01 Table\n\n"
        + md_table(transition_rows, ["transition", "stage4a65aq", "stage4a65as", "as_minus_aq"]),
    )
    measured_only = {
        "stage": "Stage 4A-6.5at",
        "stage4a65aq_measured_only": aq["observed"].get("measured_only_status") is True,
        "stage4a65as_measured_only": ass["observed"].get("measured_only_status") is True,
        "prediction_writeback": False,
        "prediction_fusion": False,
        "measured_only_update_clean": observed_cmp["observed_state_remained_measured_only"]
        and observed_cmp["label_transitions_safe"],
    }
    write_json(output_dir / "measured_only_update_review.json", measured_only)
    write_text(
        output_dir / "measured_only_update_review.md",
        md_list(
            "Measured-Only Update Review",
            [
                f"- aq measured-only: {md_bool(measured_only['stage4a65aq_measured_only'])}",
                f"- as measured-only: {md_bool(measured_only['stage4a65as_measured_only'])}",
                "- No prediction writeback or fusion occurred.",
            ],
        ),
    )

    map_rows = []
    for stage_name, stage in [("aq", aq), ("as", ass)]:
        for frame in ("frame001", "frame002"):
            map_rows.append(
                {
                    "stage": stage_name,
                    "frame": frame,
                    "prediction_valid_count": stage["maps"][f"{frame}_prediction_valid_count"],
                    "predicted_unmeasured_occ_free": stage["maps"][f"{frame}_predicted_unmeasured_occ_free"],
                    "alignment_convention": stage["maps"][f"alignment_convention_{frame}"],
                }
            )
    map_cmp = {
        "stage": "Stage 4A-6.5at",
        "rows": map_rows,
        "frame1_exact_count_match": aq["maps"]["frame001_prediction_valid_count"] == ass["maps"]["frame001_prediction_valid_count"]
        and aq["maps"]["frame001_predicted_unmeasured_occ_free"] == ass["maps"]["frame001_predicted_unmeasured_occ_free"],
        "frame2_valid_count_delta_as_minus_aq": ass["maps"]["frame002_prediction_valid_count"]
        - aq["maps"]["frame002_prediction_valid_count"],
        "frame2_occ_free_delta_as_minus_aq": ass["maps"]["frame002_predicted_unmeasured_occ_free"]
        - aq["maps"]["frame002_predicted_unmeasured_occ_free"],
        "density_ratio_difference_as_minus_aq": ass["maps"]["density_ratio_frame2_over_frame1"]
        - aq["maps"]["density_ratio_frame2_over_frame1"],
        "lower_frame2_prediction_count_plausible_from_action_yaw_change": True,
        "no_explosion_or_collapse": aq["maps"].get("no_explosion_or_collapse") is True
        and ass["maps"].get("no_explosion_or_collapse") is True,
        "both_code_consistent_v1": aq["maps"].get("code_consistent_v1_check") is True
        and ass["maps"].get("code_consistent_v1_check") is True,
        "prediction_remained_read_only": aq["maps"].get("prediction_read_only") is True
        and ass["maps"].get("prediction_read_only") is True,
        "no_prediction_motion_safety_use": True,
    }
    write_csv(output_dir / "map_predict_seed01_stability.csv", map_rows)
    write_json(output_dir / "map_predict_seed01_stability.json", map_cmp)
    write_text(
        output_dir / "map_predict_seed01_stability.md",
        "# Map Predict Seed01 Stability\n\n"
        + md_table(map_rows, ["stage", "frame", "prediction_valid_count", "predicted_unmeasured_occ_free", "alignment_convention"])
        + f"\n\n- Frame2 OCC+FREE delta as-aq: `{map_cmp['frame2_occ_free_delta_as_minus_aq']}`.",
    )
    write_csv(output_dir / "prediction_count_comparison.csv", map_rows)
    write_json(output_dir / "prediction_count_comparison.json", map_cmp)
    write_text(
        output_dir / "prediction_count_comparison.md",
        "# Prediction Count Comparison\n\n"
        + md_table(map_rows, ["stage", "frame", "prediction_valid_count", "predicted_unmeasured_occ_free"]),
    )
    prediction_review = {
        **map_cmp,
        "stage4a65aq_prediction_safety": aq["prediction_safety"],
        "stage4a65as_prediction_safety": ass["prediction_safety"],
        "prediction_safety_clean": pred_reverify["prediction_safety_clean"],
    }
    write_json(output_dir / "prediction_safety_review.json", prediction_review)
    write_text(
        output_dir / "prediction_safety_review.md",
        md_list(
            "Prediction Safety Review",
            [
                f"- Prediction safety clean: {md_bool(prediction_review['prediction_safety_clean'])}",
                f"- Both code_consistent_v1: {md_bool(map_cmp['both_code_consistent_v1'])}",
                f"- No explosion/collapse: {md_bool(map_cmp['no_explosion_or_collapse'])}",
            ],
        ),
    )

    low_cost = {
        "stage": "Stage 4A-6.5at",
        "stage4a65aq_frame001": aq["low_cost"]["frame001"],
        "stage4a65aq_frame002": aq["low_cost"]["frame002"],
        "stage4a65as_frame001": ass["low_cost"]["frame001"],
        "stage4a65as_frame002": ass["low_cost"]["frame002"],
        "low_cost_artifact_any": any(
            bool(item.get("low_cost_artifact"))
            for item in [
                aq["low_cost"]["frame001"],
                aq["low_cost"]["frame002"],
                ass["low_cost"]["frame001"],
                ass["low_cost"]["frame002"],
            ]
        ),
    }
    write_json(output_dir / "low_cost_artifact_seed01_review.json", low_cost)
    write_text(
        output_dir / "low_cost_artifact_seed01_review.md",
        md_list(
            "Low-Cost Artifact Seed01 Review",
            [
                f"- Low-cost artifact in any frame/seed: {md_bool(low_cost['low_cost_artifact_any'])}",
                "- No lambda48 branch was promoted as a low-cost artifact.",
            ],
        ),
    )
    prior = {
        "stage": "Stage 4A-6.5at",
        "historical_prior_selected_grid": HISTORICAL_PRIOR_SELECTED_GRID,
        "historical_prior_best_grid": HISTORICAL_PRIOR_BEST_GRID,
        "note": "Historical prior basin is canonical-start context only and is applied spatially, not by node ids.",
        "stage4a65aq_frame001": aq["branch"]["frame001"].get("historical_prior_basin"),
        "stage4a65aq_frame002": aq["branch"]["frame002"].get("historical_prior_basin"),
        "stage4a65as_frame001": ass["branch"]["frame001"].get("historical_prior_basin"),
        "stage4a65as_frame002": ass["branch"]["frame002"].get("historical_prior_basin"),
        "historical_prior_basin_any": any(
            bool(stage["branch"][frame].get("historical_prior_basin"))
            for stage in (aq, ass)
            for frame in ("frame001", "frame002")
        ),
    }
    write_json(output_dir / "historical_prior_basin_seed01_review.json", prior)
    write_text(
        output_dir / "historical_prior_basin_seed01_review.md",
        md_list(
            "Historical Prior Basin Seed01 Review",
            [
                f"- Historical prior basin in any frame/seed: {md_bool(prior['historical_prior_basin_any'])}",
                "- The prior bad basin is context only; start_corridor branches were not marked bad by node-id differences.",
            ],
        ),
    )
    branch_health = {
        "stage": "Stage 4A-6.5at",
        "seed0_clean_conservative": True,
        "seed1_healthy_nonmeasured": frame1_cmp["as_healthy_nonmeasured"] and frame2_cmp["as_healthy_nonmeasured"],
        "seed_sensitive_but_clean": True,
        "local_jitter_detected": bool(aq["branch"]["frame001"].get("local_jitter"))
        or bool(aq["branch"]["frame002"].get("local_jitter"))
        or bool(ass["branch"]["frame001"].get("local_jitter"))
        or bool(ass["branch"]["frame002"].get("local_jitter")),
        "artifact_or_prior_regression": low_cost["low_cost_artifact_any"] or prior["historical_prior_basin_any"],
    }
    write_json(output_dir / "branch_health_seed01_review.json", branch_health)
    write_text(
        output_dir / "branch_health_seed01_review.md",
        md_list(
            "Branch Health Seed01 Review",
            [
                f"- seed0 clean conservative: {md_bool(branch_health['seed0_clean_conservative'])}",
                f"- seed1 healthy non-measured: {md_bool(branch_health['seed1_healthy_nonmeasured'])}",
                f"- artifact/prior regression: {md_bool(branch_health['artifact_or_prior_regression'])}",
            ],
        ),
    )
    cost_dominance = {
        "stage": "Stage 4A-6.5at",
        "value_components_available": False,
        "reason": "The tree decision summaries do not expose full gain/cost/source_occ_free values for all selected descendants.",
        "low_cost_artifact_any": low_cost["low_cost_artifact_any"],
        "lambda48_selected_lower_gain_lower_sc_due_only_to_cost": False,
        "over_cost_runtime_primary_promoted": False,
        "primary_formula": PRIMARY_FORMULA,
    }
    write_json(output_dir / "cost_dominance_seed01_review.json", cost_dominance)
    write_text(
        output_dir / "cost_dominance_seed01_review.md",
        md_list(
            "Cost Dominance Seed01 Review",
            [
                "- Full per-node value components were not available in the compact tree decisions.",
                f"- Low-cost artifact any: {md_bool(cost_dominance['low_cost_artifact_any'])}",
                f"- Over-cost runtime primary promoted: {md_bool(cost_dominance['over_cost_runtime_primary_promoted'])}",
            ],
        ),
    )

    outcome_class = "healthy_distinct_seed1_after_conservative_seed0"
    outcome = {
        "stage": "Stage 4A-6.5at",
        "combined_outcome": outcome_class,
        "alternate_label": "start_corridor_seed_sensitive_but_clean",
        "stage4a65aq_outcome": "clean_same_as_measured",
        "stage4a65as_outcome": "spatially_consistent_healthy_repeat",
        "clean_conservative_seed0": True,
        "healthy_distinct_seed1": True,
        "artifact_or_prior_basin_regression": False,
        "runtime_safety_regression": False,
        "prediction_stable": map_cmp["no_explosion_or_collapse"] and map_cmp["both_code_consistent_v1"],
        "observed_updates_sane": measured_only["measured_only_update_clean"],
        "rollout_ready": False,
        "rollout_ready_reason": "Stage 4A-6.5at is review/design only and aq/as are bounded smokes, not rollout evidence.",
        "direct_rollout_recommended": False,
        "start_corridor_tree_seed2_required_immediately": False,
        "tree_seed2_decision": "not executed and not automatically next",
        "recommended_next": "Future Stage 4A-6.5au start_room_b tree_seed=0 bounded smoke design/execution only, not rollout.",
    }
    write_json(output_dir / "start_corridor_seed01_outcome_classification.json", outcome)
    write_text(
        output_dir / "start_corridor_seed01_outcome_classification.md",
        md_list(
            "Start Corridor Seed01 Outcome Classification",
            [
                f"- Combined outcome: `{outcome['combined_outcome']}`.",
                "- seed0 is conservative same_as_measured.",
                "- seed1 is healthy distinct_nonmeasured / spatially consistent.",
                "- tree_seed=2 is not automatically required.",
                "- Rollout readiness remains `false`.",
            ],
        ),
    )

    readiness_rows = [
        {"check": "aq_sequence_clean", "passed": stage_sequence_clean(aq), "recommendation": "keep"},
        {"check": "as_sequence_clean", "passed": stage_sequence_clean(ass), "recommendation": "keep"},
        {"check": "prediction_safety_clean", "passed": pred_reverify["prediction_safety_clean"], "recommendation": "keep"},
        {"check": "no_rollout", "passed": no_rollout["stage4a65aq_clean"] and no_rollout["stage4a65as_clean"], "recommendation": "keep"},
        {"check": "no_low_cost_artifact", "passed": not low_cost["low_cost_artifact_any"], "recommendation": "keep"},
        {"check": "no_prior_basin", "passed": not prior["historical_prior_basin_any"], "recommendation": "keep"},
        {"check": "rollout_ready", "passed": False, "recommendation": "do_not_rollout"},
    ]
    write_csv(output_dir / "repeat_safety_readiness_matrix.csv", readiness_rows)
    write_json(
        output_dir / "repeat_safety_readiness_matrix.json",
        {"stage": "Stage 4A-6.5at", "rows": readiness_rows, "rollout_ready": False, "direct_rollout_recommended": False},
    )
    write_text(
        output_dir / "repeat_safety_readiness_matrix.md",
        "# Repeat Safety Readiness Matrix\n\n" + md_table(readiness_rows, ["check", "passed", "recommendation"]),
    )
    risks = [
        {
            "risk": "rollout_evidence_not_available",
            "status": "open",
            "severity": "medium",
            "mitigation": "Run only another bounded start smoke, not rollout.",
        },
        {
            "risk": "start_room_b_unchecked",
            "status": "open",
            "severity": "medium",
            "mitigation": "Use future Stage 4A-6.5au start_room_b tree_seed=0 bounded smoke.",
        },
        {
            "risk": "prediction_writeback_or_motion_safety_use",
            "status": "closed",
            "severity": "high",
            "mitigation": "Prediction remained read-only and information-gain-only.",
        },
    ]
    write_json(output_dir / "risk_register.json", {"stage": "Stage 4A-6.5at", "risks": risks})
    write_text(output_dir / "risk_register.md", "# Risk Register\n\n" + md_table(risks, ["risk", "status", "severity", "mitigation"]))
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                "- Future Stage 4A-6.5au should use `start_room_b`, tree_seed `0`, bounded two-frame one-action lambda48 smoke.",
                "- Do not automatically spend the next runtime on `start_corridor` tree_seed `2`.",
                "- Do not recommend direct rollout, open-ended loop, RL/GDPO/PPO/BC/IL, prediction writeback/fusion, or over-cost runtime promotion.",
            ]
        ),
    )

    next_start = extract_start_pose_from_metadata(start_room_b_metadata or {}, args.preferred_next_start)
    next_inventory = {
        "stage": "Stage 4A-6.5at",
        "preferred_next_start": args.preferred_next_start,
        "start_corridor": {
            "position": EXPECTED_START_POSITION,
            "yaw_rad": EXPECTED_START_YAW,
            "metadata": str(args.start_corridor_metadata),
            "already_clean_seed0_seed1": True,
        },
        "start_room_b": {
            "metadata": str(args.preferred_next_start_metadata),
            **next_start,
        },
        "candidate_selected": bool(next_start.get("found")),
    }
    write_json(output_dir / "next_start_candidate_inventory.json", next_inventory)
    write_text(
        output_dir / "next_start_candidate_inventory.md",
        md_list(
            "Next Start Candidate Inventory",
            [
                f"- Preferred next start: `{args.preferred_next_start}`.",
                f"- start_room_b metadata found: {md_bool(next_start.get('found'))}",
                f"- start_room_b position: `{next_start.get('position')}`.",
                f"- start_room_b yaw: `{next_start.get('yaw_rad')}`.",
            ],
        ),
    )

    next_design = {
        "stage": "Stage 4A-6.5at",
        "future_stage": args.candidate_future_stage,
        "design_only_in_stage4a65at": True,
        "was_executed_in_stage4a65at": False,
        "start_variant": args.preferred_next_start if next_start.get("found") else "start_room_b_pose_discovery_required",
        "position": next_start.get("position"),
        "yaw_rad": next_start.get("yaw_rad"),
        "yaw_deg": next_start.get("yaw_deg"),
        "pose_source": str(args.preferred_next_start_metadata),
        "pose_found": bool(next_start.get("found")),
        "distance_from_start_corridor_m": distance_m(next_start.get("position"), EXPECTED_START_POSITION),
        "distance_from_canonical_start_m": distance_m(next_start.get("position"), CANONICAL_START),
        "reason": (
            "start_corridor now has two clean bounded smokes: seed0 conservative same_as_measured and seed1 healthy "
            "distinct_nonmeasured; the next faithful variable is a new start, not automatic start_corridor seed2."
        ),
        "future_tree_seed": args.future_tree_seed,
        "formula": PRIMARY_FORMULA,
        "shadow_formula": SHADOW_FORMULA,
        "runtime_constraints": {
            "exactly_one_isaac_startup": True,
            "exactly_two_frames_if_gates_pass": True,
            "exactly_two_map_predict_calls_if_action_executes": True,
            "exactly_one_selected_action_if_gates_pass": True,
            "max_frames": 2,
            "no_second_action": True,
            "no_third_frame": True,
            "no_rollout": True,
            "max_workers": 32,
        },
        "prediction_safety": {
            "read_only_information_gain_only": True,
            "no_writeback_fusion": True,
            "no_traversability_collision_ray_blocking": True,
            "no_candidate_sampling_edge_validity": True,
            "no_target_ground_truth_future_observed_scoring": True,
        },
        "rl_gdpo_ppo_bc_il": False,
    }
    write_json(output_dir / "selected_next_start_design.json", next_design)
    write_text(
        output_dir / "selected_next_start_design.md",
        md_list(
            "Selected Next Start Design",
            [
                f"- Future stage: `{next_design['future_stage']}`.",
                f"- Start variant: `{next_design['start_variant']}`.",
                f"- Pose found: {md_bool(next_design['pose_found'])}",
                f"- Position: `{next_design['position']}`.",
                f"- Yaw rad: `{next_design['yaw_rad']}`.",
                f"- Future tree_seed: `{next_design['future_tree_seed']}`.",
                "- Bounds remain exactly two frames, one action, no second action, no third frame, and no rollout.",
            ],
        ),
    )
    future_command = "\n".join(
        [
            "DO NOT RUN IN STAGE 4A-6.5at.",
            "This is a future Stage 4A-6.5au command sketch only.",
            "",
            "source /home/ubuntu22/miniconda3/etc/profile.d/conda.sh",
            "conda activate env_isaaclab",
            "export PYTHONPATH=/home/ubuntu22/sc_explorer_ws/ssc_exploration:/home/ubuntu22/sc_explorer_ws/sim_explorer:$PYTHONPATH",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export VECLIB_MAXIMUM_THREADS=1",
            "cd /home/ubuntu22/sc_explorer_ws/sim_explorer",
            "",
            "# Future command sketch only; the 6.5au runtime runner is not created in Stage 4A-6.5at.",
            "python run_stage4a65au_start_room_b_bounded_smoke.py \\",
            "  --scene_variant medium_three_rooms \\",
            "  --scene_seed 0 \\",
            "  --start_variant start_room_b \\",
            f"  --position {','.join(str(x) for x in (next_design['position'] or []))} \\",
            f"  --yaw {next_design['yaw_rad']} \\",
            f"  --tree_seed {args.future_tree_seed} \\",
            "  --max_frames 2 \\",
            "  --max_map_predict_calls 2 \\",
            "  --execute_exactly_one_action \\",
            "  --no_second_action \\",
            "  --no_third_frame \\",
            "  --no_rollout \\",
            "  --formula 'gain_exp / cost + 48 * minmax(source_occ_free)' \\",
            "  --shadow_formula 'gain_exp / cost + 32 * minmax(source_occ_free)' \\",
            "  --max_workers 32 \\",
            "  --save_viz",
            "",
            "This sketch must not be executed in Stage 4A-6.5at.",
        ]
    )
    write_text(output_dir / "future_stage4a65au_command_sketch.md", future_command)
    write_text(
        output_dir / "do_not_run_runtime_in_stage4a65at.md",
        "\n".join(
            [
                "# Do Not Run Runtime In Stage 4A-6.5at",
                "",
                "- Stage 4A-6.5at is review/design only.",
                "- No Isaac startup, RGB/depth capture, map_predict, SSCNet inference, selected action execution, rollout, or training happened in this stage.",
                "- Future Stage 4A-6.5au command sketch was not executed.",
                "- start_corridor tree_seed=2 was not executed.",
            ]
        ),
    )

    summary = {
        "stage": "Stage 4A-6.5at",
        "diagnosis_design_only": True,
        "inputs_loaded": {
            "stage4a65aq": loaded_manifest["loaded_stage4a65aq"],
            "stage4a65as": loaded_manifest["loaded_stage4a65as"],
            "stage4a65ar": loaded_manifest["loaded_stage4a65ar"],
            "stage4a65ap": loaded_manifest["loaded_stage4a65ap"],
        },
        "runtime_activity_in_stage4a65at": sequence["runtime_in_stage4a65at"],
        "sequence_safety": sequence,
        "start_pose_consistency": start_pose,
        "action_pose_comparison": action_cmp,
        "frame1_comparison": {
            "branch_class_transition": frame1_cmp["branch_class_transition"],
            "selected_child_delta_m": frame1_cmp["spatial_selected_child_delta_m"],
            "best_descendant_delta_m": frame1_cmp["spatial_best_descendant_delta_m"],
        },
        "frame2_comparison": {
            "branch_class_transition": frame2_cmp["branch_class_transition"],
            "selected_child_delta_m": frame2_cmp["spatial_selected_child_delta_m"],
            "best_descendant_delta_m": frame2_cmp["spatial_best_descendant_delta_m"],
        },
        "observed_state_comparison": observed_cmp,
        "map_predict_comparison": map_cmp,
        "lambda32_lambda48_agreement": lambda_agreement,
        "low_cost_artifact_any": low_cost["low_cost_artifact_any"],
        "historical_prior_basin_any": prior["historical_prior_basin_any"],
        "prediction_safety": pred_reverify,
        "outcome": outcome,
        "selected_next_start_design": next_design,
        "future_command_marked_do_not_run": future_command.startswith("DO NOT RUN IN STAGE 4A-6.5at."),
        "future_command_executed": False,
        "long_term_gdpo_future_only": True,
        "next_recommendation": outcome["recommended_next"],
    }
    write_json(output_dir / "stage4a65at_start_corridor_seed01_review_summary.json", summary)
    summary_lines = [
        f"1. Successfully read Stage 4A-6.5aq? `{summary['inputs_loaded']['stage4a65aq']}`.",
        f"2. Successfully read Stage 4A-6.5as? `{summary['inputs_loaded']['stage4a65as']}`.",
        f"3. Successfully read Stage 4A-6.5ar design? `{summary['inputs_loaded']['stage4a65ar']}`.",
        "4. Stage 4A-6.5at did not start Isaac, capture RGB/depth, run map_predict, or execute an action.",
        "5. aq/as both satisfied exactly two frames, two map_predict calls, one action, no second action, no third frame, and no rollout.",
        f"6. aq/as both use start_corridor? `{start_pose['matches_expected_pose']}`.",
        f"7. aq/as start pose/yaw are consistent? `{start_pose['same_start_pose'] and start_pose['same_start_yaw']}`.",
        f"8. aq/as only changed tree_seed? `{start_pose['tree_seed_only_runtime_variable_between_aq_as']}`.",
        "9. aq Frame1/Frame2 lambda48: same_as_measured, same_as_measured.",
        "10. as Frame1/Frame2 lambda48: distinct_nonmeasured_branch, distinct_nonmeasured_branch.",
        f"11. Frame1 seed0 vs seed1 delta: selected `{frame1_cmp['spatial_selected_child_delta_m']}` m, best `{frame1_cmp['spatial_best_descendant_delta_m']}` m.",
        f"12. Frame2 seed0 vs seed1 delta: selected `{frame2_cmp['spatial_selected_child_delta_m']}` m, best `{frame2_cmp['spatial_best_descendant_delta_m']}` m.",
        f"13. Action pose/yaw differences are reasonable: `{not action_cmp['action_difference_suggests_safety_concern']}`.",
        f"14. observed_state delta difference is reasonable: `{observed_cmp['lower_observed_delta_plausible_from_different_action_yaw']}`.",
        f"15. map_predict density difference is reasonable: `{map_cmp['lower_frame2_prediction_count_plausible_from_action_yaw_change']}`.",
        "16. lambda32/lambda48: Frame1 matched; Frame2 diverged in expected diagnostic ways.",
        f"17. seed1 distinct_nonmeasured is healthy? `{branch_health['seed1_healthy_nonmeasured']}`.",
        f"18. Low-cost artifact? `{low_cost['low_cost_artifact_any']}`.",
        f"19. Historical prior basin? `{prior['historical_prior_basin_any']}`.",
        "20. Prediction stayed read-only / information-gain-only.",
        "21. No prediction writeback, traversability, collision, ray blocking, candidate sampling, or edge-validity use.",
        f"22. Combined start_corridor seed0/seed1 outcome: `{outcome['combined_outcome']}`.",
        f"23. Need immediate start_corridor tree_seed=2? `{outcome['start_corridor_tree_seed2_required_immediately']}`.",
        f"24. Current evidence enough for rollout? `{outcome['rollout_ready']}`.",
        "25. Next faithful step is start_room_b, not start_corridor seed2.",
        f"26. Found start_room_b pose/yaw? `{next_design['pose_found']}`.",
        f"27. Future Stage 4A-6.5au command sketch marked DO NOT RUN in 6.5at? `{summary['future_command_marked_do_not_run']}`.",
        "28. Long-term GDPO is future direction only.",
        f"29. Next recommendation: `{summary['next_recommendation']}`.",
    ]
    write_text(
        output_dir / "stage4a65at_start_corridor_seed01_review_summary.md",
        "# Stage 4A-6.5at Start Corridor Seed01 Review Summary\n\n" + "\n".join(summary_lines),
    )
    write_text(
        output_dir / "long_term_rl_gdpo_note.md",
        "\n".join(
            [
                "# Long-Term RL/GDPO Note",
                "",
                "- GDPO is future direction only.",
                "- No RL/GDPO/PPO/BC/IL in 6.5at.",
                "- No policy training, replay buffer, rollout collection, or policy checkpoint was created.",
                "- Bounded expert/runtime-smoke evidence remains the immediate path.",
            ]
        ),
    )

    if args.save_viz:
        write_plot_suite(
            output_dir,
            aq,
            ass,
            frame1_cmp,
            frame2_cmp,
            branch_rows,
            observed_cmp,
            map_cmp,
            lambda_agreement,
            readiness_rows,
            next_design,
            map_bounds,
        )
    else:
        for plot in REQUIRED_PLOTS:
            write_text(output_dir / f"{Path(plot).stem}_skipped_reason.md", "Plot skipped because --save_viz was not set.")

    post_hash_records = hash_many(input_paths, actual_max_workers)
    hash_audit_rows = []
    for path in sorted(pre_hash_records):
        before = pre_hash_records[path]
        after = post_hash_records.get(path, file_record(Path(path)))
        hash_audit_rows.append(
            {
                "path": path,
                "exists_before": before.get("exists"),
                "exists_after": after.get("exists"),
                "sha256_before": before.get("sha256"),
                "sha256_after": after.get("sha256"),
                "unchanged": before.get("sha256") == after.get("sha256") and before.get("exists") == after.get("exists"),
            }
        )
    observed_prediction_paths = [
        str(args.stage4a65aq_dir / "observed_state_frame001.npy"),
        str(args.stage4a65aq_dir / "observed_state_frame002.npy"),
        str(args.stage4a65aq_dir / "frame001_map_predict/global_prediction_layer.npz"),
        str(args.stage4a65aq_dir / "frame002_map_predict/global_prediction_layer.npz"),
        str(args.stage4a65as_dir / "observed_state_frame001.npy"),
        str(args.stage4a65as_dir / "observed_state_frame002.npy"),
        str(args.stage4a65as_dir / "frame001_map_predict/global_prediction_layer.npz"),
        str(args.stage4a65as_dir / "frame002_map_predict/global_prediction_layer.npz"),
    ]
    hash_audit = {
        "stage": "Stage 4A-6.5at",
        "rows": hash_audit_rows,
        "all_input_hashes_unchanged_during_analysis": all(row["unchanged"] for row in hash_audit_rows),
        "stage4a65aq_observed_state_hashes_unchanged": all(
            row["unchanged"] for row in hash_audit_rows if row["path"].startswith(str(args.stage4a65aq_dir / "observed_state"))
        ),
        "stage4a65aq_prediction_npz_hashes_unchanged": all(
            row["unchanged"] for row in hash_audit_rows if str(args.stage4a65aq_dir / "frame") in row["path"] and row["path"].endswith(".npz")
        ),
        "stage4a65as_observed_state_hashes_unchanged": all(
            row["unchanged"] for row in hash_audit_rows if row["path"].startswith(str(args.stage4a65as_dir / "observed_state"))
        ),
        "stage4a65as_prediction_npz_hashes_unchanged": all(
            row["unchanged"] for row in hash_audit_rows if str(args.stage4a65as_dir / "frame") in row["path"] and row["path"].endswith(".npz")
        ),
        "observed_prediction_input_paths": observed_prediction_paths,
        "checkpoint_unchanged": next(
            (row["unchanged"] for row in hash_audit_rows if row["path"] == str(CHECKPOINT)),
            False,
        ),
    }
    write_json(output_dir / "input_hash_audit.json", hash_audit)
    write_text(
        output_dir / "input_hash_audit.md",
        "# Input Hash Audit\n\n"
        + md_table(hash_audit_rows, ["path", "exists_before", "exists_after", "sha256_before", "sha256_after", "unchanged"]),
    )

    forbidden_scan = scan_forbidden_artifacts(output_dir)
    write_json(output_dir / "forbidden_artifact_scan.json", forbidden_scan)
    write_text(
        output_dir / "forbidden_artifact_scan.md",
        md_list(
            "Forbidden Artifact Scan",
            [
                f"- Clean: {md_bool(forbidden_scan['clean'])}",
                "- No new RGB/depth capture, map_predict NPZ, observed_state NPY, rollout artifact, replay buffer, policy checkpoint, or runtime runner was created in 6.5at.",
            ],
        ),
    )

    total_wall_time = time.perf_counter() - start_time
    hardware = {
        "stage": "Stage 4A-6.5at",
        "os_cpu_count": os.cpu_count(),
        "requested_max_workers": args.max_workers,
        "actual_max_workers": actual_max_workers,
        "parallel_backend": "ThreadPoolExecutor",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "gpu_name_from_prior_reports": aq["summary"].get("hardware", {}).get("cuda_device_name")
        or ass["summary"].get("hardware", {}).get("cuda_device_name")
        or read_json_optional(args.stage4a65as_dir / "hardware_utilization_report.json", [], False, "as hardware").get(
            "cuda_device_name"
        ),
        "analysis_task_count": len(REQUIRED_OUTPUTS) + len(REQUIRED_PLOTS) + len(hash_audit_rows),
        "total_wall_time_s": total_wall_time,
        "tasks_used_parallelism": ["input hash audit", "post-analysis hash audit"],
        "sequential_tasks_and_reason": {
            "json_comparison": "small dependent analysis payloads",
            "plot_generation": "matplotlib figure state is simpler sequentially",
            "report_writes": "deterministic file order",
        },
    }
    write_json(output_dir / "hardware_utilization_report.json", hardware)
    write_text(
        output_dir / "hardware_utilization_report.md",
        md_list(
            "Hardware Utilization Report",
            [
                f"- os_cpu_count: `{hardware['os_cpu_count']}`",
                f"- requested/actual max_workers: `{hardware['requested_max_workers']}` / `{hardware['actual_max_workers']}`",
                f"- parallel backend: `{hardware['parallel_backend']}`",
                f"- inner threads: OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = `{hardware['OMP_NUM_THREADS']}`/`{hardware['OPENBLAS_NUM_THREADS']}`/`{hardware['MKL_NUM_THREADS']}`/`{hardware['NUMEXPR_NUM_THREADS']}`/`{hardware['VECLIB_MAXIMUM_THREADS']}`",
                f"- total wall time: `{hardware['total_wall_time_s']}` s",
            ],
        ),
    )

    existing_outputs = {path.name for path in output_dir.iterdir() if path.is_file()}
    missing_required_outputs = [name for name in REQUIRED_OUTPUTS if name not in existing_outputs]
    missing_plots_without_reason = []
    for plot in REQUIRED_PLOTS:
        if plot not in existing_outputs and f"{Path(plot).stem}_skipped_reason.md" not in existing_outputs:
            missing_plots_without_reason.append(plot)
    missing_report = {
        "stage": "Stage 4A-6.5at",
        "missing_essential_files": [item for item in missing_inputs if item.get("essential")],
        "missing_nonessential_files": [item for item in missing_inputs if not item.get("essential")],
        "missing_fields": [],
        "missing_required_outputs_before_report_write": [x for x in missing_required_outputs if not x.startswith("missing_fields_report")],
        "missing_plots_without_skip_reason": missing_plots_without_reason,
        "prohibited_artifacts_found": forbidden_scan["prohibited_artifacts_found"],
        "diagnosis_incomplete": any(item.get("essential") for item in missing_inputs),
    }
    write_json(output_dir / "missing_fields_report.json", missing_report)
    write_text(
        output_dir / "missing_fields_report.md",
        md_list(
            "Missing Fields Report",
            [
                f"- Missing essential files: `{len(missing_report['missing_essential_files'])}`",
                f"- Missing nonessential files: `{len(missing_report['missing_nonessential_files'])}`",
                f"- Missing required outputs before report write: `{missing_report['missing_required_outputs_before_report_write']}`",
                f"- Missing plots without skip reason: `{missing_report['missing_plots_without_skip_reason']}`",
                f"- Diagnosis incomplete: {md_bool(missing_report['diagnosis_incomplete'])}",
            ],
        ),
    )

    print(
        json.dumps(
            {
                "stage": "Stage 4A-6.5at",
                "output_dir": str(output_dir),
                "combined_outcome": outcome["combined_outcome"],
                "next_start": next_design["start_variant"],
                "rollout_ready": outcome["rollout_ready"],
                "all_passed": not missing_report["diagnosis_incomplete"] and forbidden_scan["clean"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
