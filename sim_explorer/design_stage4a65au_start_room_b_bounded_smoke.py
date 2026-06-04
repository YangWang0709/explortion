#!/usr/bin/env python3
"""Stage 4A-6.5au start_room_b bounded-smoke design review.

This stage is offline design/review only. It reads completed Stage 4A-6.5aq,
4A-6.5as, 4A-6.5ar, 4A-6.5ap, canonical 4A-6.5ak/am/ao references, and
start_room_b metadata. It writes a start_room_b two-frame one-action lambda48
runtime smoke design package and a future command sketch. It does not start
Isaac, capture RGB/depth, run map_predict or SSCNet inference, execute an
action, run rollout, train, or create a runtime runner.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
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
from matplotlib.patches import Rectangle


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65au_start_room_b_bounded_smoke"
DEFAULT_AQ_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65aq_alternate_start_corridor_bounded_smoke"
DEFAULT_AS_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65as_start_corridor_tree_seed1_bounded_smoke"
DEFAULT_AR_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ar_alternate_start_post_action_diagnosis"
DEFAULT_AP_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ap_seed012_repeat_review_alternate_start_design"
DEFAULT_AT_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65at_start_corridor_seed01_review_next_start_design"
DEFAULT_AK_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ak_two_frame_one_action_lambda48_runtime_smoke"
DEFAULT_AM_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65am_bounded_repeat_safety_smoke_tree_seed1"
DEFAULT_AO_DIR = WORKSPACE / "outputs/isaac_sc_pred_stage4a65ao_bounded_repeat_safety_smoke_tree_seed2"
DEFAULT_START_DATASET_DIR = WORKSPACE / "outputs/isaac_medium_rollout_dataset_empty_pred_astar"
DEFAULT_START_ROOM_B_METADATA = (
    DEFAULT_START_DATASET_DIR / "episodes/medium_three_rooms_seed0_start_room_b_empty_astar/scene_metadata.json"
)
DEFAULT_START_CORRIDOR_METADATA = (
    DEFAULT_START_DATASET_DIR / "episodes/medium_three_rooms_seed0_start_corridor_empty_astar/scene_metadata.json"
)
CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]

PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
SHADOW_FORMULA = "gain_exp / cost + 32 * minmax(source_occ_free)"
MEASURED_FORMULA = "gain_exp / cost"
PROHIBITED_FORMULAS = [
    "(gain_exp + 48 * source_occ_free) / cost",
    "(gain_exp + source_occ_free) / cost",
]
EXPECTED_SCENE_VARIANT = "medium_three_rooms"
EXPECTED_SCENE_SEED = 0
EXPECTED_START_VARIANT = "start_room_b"
EXPECTED_START_POSITION = [2.75, -2.55, 1.2]
EXPECTED_START_YAW = 2.7052603405912112
START_CORRIDOR_POSITION = [0.0, -4.45, 1.2]
START_CORRIDOR_YAW = 1.5707963267948966
CANONICAL_START = [-4.65, -4.65, 1.2]
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
    "prediction_safety_review.json",
    "prediction_safety_review.md",
    "no_rollout_reverification.json",
    "no_rollout_reverification.md",
    "start_room_b_pose_consistency.json",
    "start_room_b_pose_consistency.md",
    "observed_state_delta_summary.json",
    "observed_state_delta_summary.md",
    "map_predict_two_frame_stability.json",
    "map_predict_two_frame_stability.md",
    "frame1_tree_decision_diagnosis.json",
    "frame1_tree_decision_diagnosis.md",
    "frame2_tree_decision_diagnosis.json",
    "frame2_tree_decision_diagnosis.md",
    "tree_sc_gain_logging_contract.csv",
    "tree_sc_gain_logging_contract.json",
    "tree_sc_gain_logging_contract.md",
    "low_cost_artifact_review.json",
    "low_cost_artifact_review.md",
    "historical_prior_basin_review.json",
    "historical_prior_basin_review.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "next_start_candidate_inventory.json",
    "next_start_candidate_inventory.md",
    "selected_next_start_design.json",
    "selected_next_start_design.md",
    "future_stage4a65au_command_sketch.md",
    "do_not_run_runtime_in_stage4a65au.md",
    "stage4a65au_start_room_b_design_summary.json",
    "stage4a65au_start_room_b_design_summary.md",
]

REQUIRED_PLOTS = [
    "start_room_b_design_topdown.png",
    "reference_seed01_frame1_tree_topdown.png",
    "reference_seed01_frame2_tree_topdown.png",
    "canonical_seed012_reference_tree_topdown.png",
    "observed_state_reference_delta_bar.png",
    "map_predict_reference_stability_bar.png",
    "tree_sc_gain_reference_frame1.png",
    "tree_sc_gain_reference_frame2.png",
    "repeat_safety_readiness_matrix.png",
    "future_two_frame_sequence_timeline.png",
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
    "run_stage4a65au_start_room_b_bounded_smoke.py",
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
    missing.append({"path": str(path), "essential": essential, "role": role})
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
    with path.open("w", newline="", encoding="utf-8") as handle:
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
    worker_count = max(1, min(max_workers, len(unique), os.cpu_count() or 1))
    if worker_count == 1:
        records = [file_record(path) for path in unique]
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            records = list(pool.map(file_record, unique))
    return {record["path"]: record for record in records}


def md_bool(value: Any) -> str:
    return "`true`" if bool(value) else "`false`"


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def md_doc(title: str, lines: list[str]) -> str:
    return "\n".join([f"# {title}", "", *lines])


def get_path(data: Any, path: list[str], default: Any = None) -> Any:
    current = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def close_list(a: Any, b: Any, atol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return False
    return bool(np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), atol=atol))


def distance_m(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(av[:3] - bv[:3]))


def yaw_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(abs(float(a) - float(b)))


def format_float(value: Any, digits: int = 6) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def decision_summary(path: Path, missing: list[dict[str, Any]], role: str, essential: bool = True) -> dict[str, Any]:
    payload = read_json_optional(path, missing, essential, role)
    decision = payload.get("decision", {}) if isinstance(payload, dict) else {}
    return {
        "available": bool(decision),
        "path": str(path),
        "timing_s": payload.get("timing_s") if isinstance(payload, dict) else None,
        "tree_profile": payload.get("tree_profile") if isinstance(payload, dict) else None,
        "tree_dir": payload.get("tree_dir") if isinstance(payload, dict) else None,
        "selected_child_id": decision.get("selected_child_id"),
        "best_descendant_id": decision.get("best_descendant_id"),
        "selected_child_grid": decision.get("selected_child_grid"),
        "best_descendant_grid": decision.get("best_descendant_grid"),
        "selected_child_world": decision.get("selected_child_world"),
        "best_descendant_world": decision.get("best_descendant_world"),
        "branch_classification": decision.get("branch_classification"),
        "changed_vs_measured_only": decision.get("changed_vs_measured_only"),
        "same_as_measured": decision.get("same_as_measured"),
        "healthy_nonmeasured_candidate": decision.get("healthy_nonmeasured_candidate"),
        "low_cost_artifact": decision.get("low_cost_artifact"),
        "same_as_prior_low_cost_sc": decision.get("same_as_prior_low_cost_sc"),
        "spatial_prior_sc_basin": decision.get("spatial_prior_sc_basin"),
        "final_value": decision.get("final_value"),
        "runner_up_value": decision.get("runner_up_value"),
        "margin": decision.get("margin"),
        "normalized_margin": decision.get("normalized_margin"),
        "cost": decision.get("cost"),
        "gain_exp": decision.get("gain_exp"),
        "source_occ_free": decision.get("source_occ_free"),
        "source_occ_free_count": decision.get("source_occ_free_count"),
        "source_occ_count": decision.get("source_occ_count"),
        "source_free_count": decision.get("source_free_count"),
        "normalized_sc": decision.get("normalized_sc"),
        "sc_bonus": decision.get("sc_bonus"),
        "sc_gain": decision.get("sc_gain"),
        "max_sc": decision.get("max_sc"),
        "min_sc": decision.get("min_sc"),
        "formula": decision.get("formula"),
        "formula_name": decision.get("formula_name"),
        "utility_mode": decision.get("utility_mode"),
        "selected_cost_rank": decision.get("selected_cost_rank"),
        "selected_gain_exp_rank": decision.get("selected_gain_exp_rank"),
        "selected_source_occ_free_rank": decision.get("selected_source_occ_free_rank"),
        "candidate_count": decision.get("candidate_count"),
        "tree_total_nodes": decision.get("tree_total_nodes"),
        "path_node_ids": decision.get("path_node_ids"),
    }


def load_stage(
    base: Path,
    label: str,
    tree_seed: int,
    summary_name: str,
    start_variant: str,
    missing: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = read_json_optional(base / summary_name, missing, False, f"{label} summary") or {}
    runtime_setup = read_json_optional(base / "runtime_setup_summary.json", missing, False, f"{label} runtime setup") or {}
    observed = read_json_optional(base / "observed_state_delta_summary.json", missing, False, f"{label} observed delta") or {}
    maps = read_json_optional(base / "map_predict_two_frame_stability.json", missing, False, f"{label} map stability") or {}
    hardware = read_json_optional(base / "hardware_utilization_report.json", missing, False, f"{label} hardware") or {}
    formula = read_json_optional(base / "formula_definition.json", missing, False, f"{label} formula") or {}
    prediction = read_json_optional(base / "prediction_safety_report.json", missing, False, f"{label} prediction safety") or {}
    no_rollout = read_json_optional(base / "no_rollout_report.json", missing, False, f"{label} no rollout") or {}
    action = read_json_optional(base / "action_execution_report.json", missing, False, f"{label} action") or {}
    frame1_pose = read_json_optional(base / "frame001_pose.json", missing, False, f"{label} frame001 pose") or {}
    frame2_pose = read_json_optional(base / "frame002_pose.json", missing, False, f"{label} frame002 pose") or {}
    frame1_branch = read_json_optional(base / "frame001_branch_classification.json", missing, False, f"{label} frame001 branch") or {}
    frame2_branch = read_json_optional(base / "frame002_branch_classification.json", missing, False, f"{label} frame002 branch") or {}
    frame1_low = read_json_optional(base / "frame001_low_cost_artifact_diagnosis.json", missing, False, f"{label} frame001 low cost") or {}
    frame2_low = read_json_optional(base / "frame002_low_cost_artifact_diagnosis.json", missing, False, f"{label} frame002 low cost") or {}
    sequence = summary.get("runtime_setup", {}) if isinstance(summary, dict) else {}
    return {
        "label": label,
        "base": str(base),
        "tree_seed": tree_seed,
        "start_variant": start_variant,
        "summary": summary,
        "runtime_setup": runtime_setup,
        "sequence": sequence,
        "observed": observed,
        "maps": maps,
        "hardware": hardware,
        "formula": formula,
        "prediction_safety": prediction,
        "no_rollout": no_rollout,
        "action": action,
        "pose": {"frame001": frame1_pose, "frame002": frame2_pose},
        "branch": {"frame001": frame1_branch, "frame002": frame2_branch},
        "low_cost": {"frame001": frame1_low, "frame002": frame2_low},
        "decision": {
            "frame001": {
                "measured": decision_summary(base / "frame001_measured_shadow_tree_decision.json", missing, f"{label} frame001 measured", False),
                "lambda32": decision_summary(base / "frame001_lambda32_shadow_tree_decision.json", missing, f"{label} frame001 lambda32", False),
                "lambda48": decision_summary(base / "frame001_lambda48_primary_tree_decision.json", missing, f"{label} frame001 lambda48", False),
            },
            "frame002": {
                "measured": decision_summary(base / "frame002_measured_shadow_tree_decision.json", missing, f"{label} frame002 measured", False),
                "lambda32": decision_summary(base / "frame002_lambda32_shadow_tree_decision.json", missing, f"{label} frame002 lambda32", False),
                "lambda48": decision_summary(base / "frame002_lambda48_diagnostic_tree_decision.json", missing, f"{label} frame002 lambda48", False),
            },
        },
    }


def extract_start_pose_from_metadata(metadata: dict[str, Any], variant: str) -> dict[str, Any]:
    start_pose = metadata.get("start_pose", {}) if isinstance(metadata, dict) else {}
    world = start_pose.get("world", {}) if isinstance(start_pose, dict) else {}
    if world.get("position") is not None and world.get("yaw_rad") is not None:
        return {
            "found": True,
            "variant": start_pose.get("variant", variant),
            "source_kind": start_pose.get("source"),
            "position": world.get("position"),
            "yaw_rad": world.get("yaw_rad"),
            "yaw_deg": world.get("yaw_deg"),
            "metadata_field": "start_pose.world",
        }
    return {"found": False, "variant": variant}


def query_gpu() -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"nvidia_smi_available": False, "cuda_available": None, "devices": []}
    devices = []
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            parts = [item.strip() for item in line.split(",")]
            if len(parts) >= 3:
                devices.append({"name": parts[0], "memory_total_mb": parts[1], "driver_version": parts[2]})
    return {
        "nvidia_smi_available": result.returncode == 0,
        "cuda_available": bool(devices),
        "devices": devices,
        "stderr": result.stderr.strip() if result.returncode != 0 else "",
    }


def stage_sequence_clean(stage: dict[str, Any]) -> bool:
    seq = stage.get("sequence", {})
    return bool(
        seq.get("isaac_startup_count") == 1
        and seq.get("frames_captured") == 2
        and seq.get("map_predict_calls") == 2
        and seq.get("selected_action_execution_count") == 1
        and seq.get("second_action") is False
        and seq.get("third_frame") is False
        and seq.get("rollout") is False
    )


def prediction_clean(stage: dict[str, Any]) -> bool:
    pred = stage.get("prediction_safety", {})
    safety = stage.get("summary", {}).get("safety", {})
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
    report = stage.get("no_rollout", {})
    return bool(
        report.get("rollout") is False
        and report.get("frame003_captured") is False
        and report.get("second_action_executed") is False
        and report.get("open_ended_loop") is False
    )


def decision_relation(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {
        "same_selected_child_id": a.get("selected_child_id") == b.get("selected_child_id"),
        "same_best_descendant_id": a.get("best_descendant_id") == b.get("best_descendant_id"),
        "selected_child_spatial_delta_m": distance_m(a.get("selected_child_world"), b.get("selected_child_world")),
        "best_descendant_spatial_delta_m": distance_m(a.get("best_descendant_world"), b.get("best_descendant_world")),
    }


def value_rows_for_frame(stages: list[dict[str, Any]], frame: str) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        for mode in ("measured", "lambda32", "lambda48"):
            dec = stage["decision"][frame][mode]
            if not dec.get("available"):
                continue
            rows.append(
                {
                    "stage": stage["label"],
                    "tree_seed": stage["tree_seed"],
                    "frame": frame,
                    "mode": mode,
                    "formula": dec.get("formula"),
                    "branch_classification": dec.get("branch_classification"),
                    "selected_child_id": dec.get("selected_child_id"),
                    "best_descendant_id": dec.get("best_descendant_id"),
                    "final_value": dec.get("final_value"),
                    "cost": dec.get("cost"),
                    "gain_exp": dec.get("gain_exp"),
                    "source_occ_free": dec.get("source_occ_free"),
                    "normalized_sc": dec.get("normalized_sc"),
                    "sc_bonus": dec.get("sc_bonus"),
                    "margin": dec.get("margin"),
                    "low_cost_artifact": dec.get("low_cost_artifact"),
                    "same_as_prior_low_cost_sc": dec.get("same_as_prior_low_cost_sc"),
                    "timing_s": dec.get("timing_s"),
                }
            )
    return rows


def build_frame_diagnosis(frame: str, frame_index: int, aq: dict[str, Any], ass: dict[str, Any]) -> dict[str, Any]:
    seed0_l48 = aq["decision"][frame]["lambda48"]
    seed1_l48 = ass["decision"][frame]["lambda48"]
    seed0_measured = aq["decision"][frame]["measured"]
    seed1_measured = ass["decision"][frame]["measured"]
    seed0_l32 = aq["decision"][frame]["lambda32"]
    seed1_l32 = ass["decision"][frame]["lambda32"]
    return {
        "stage": "Stage 4A-6.5au",
        "frame": f"frame{frame_index}",
        "design_only": True,
        "start_room_b_runtime_executed": False,
        "actual_start_room_b_tree_decision_available": False,
        "tree_seed": 0,
        "scene_variant": EXPECTED_SCENE_VARIANT,
        "scene_seed": EXPECTED_SCENE_SEED,
        "primary_formula": PRIMARY_FORMULA,
        "measured_only_shadow_required": True,
        "lambda32_shadow_required": True,
        "required_decision_logs": [
            "selected_child_id",
            "best_descendant_id",
            "selected_child_grid",
            "best_descendant_grid",
            "selected_child_world",
            "best_descendant_world",
            "gain_exp",
            "cost",
            "source_occ_free",
            "normalized_sc",
            "sc_bonus",
            "final_value",
            "margin",
            "candidate_count",
            "tree_total_nodes",
            "timing_s",
        ],
        "formal_expert_acceptance_gate": {
            "lambda48_must_be_healthy": True,
            "prefer_distinct_nonmeasured_branch": True,
            "distinct_nonmeasured_required_for_promotion_to_expert_template": True,
            "same_as_measured_allowed_only_as_bounded_smoke_diagnostic": True,
            "low_cost_artifact_must_be_false": True,
            "historical_prior_basin_must_be_false": True,
            "prediction_safety_flags_must_all_be_false": True,
        },
        "reference_context": {
            "stage4a65aq_start_corridor_tree_seed0_lambda48": seed0_l48,
            "stage4a65aq_measured_vs_lambda48": decision_relation(seed0_measured, seed0_l48),
            "stage4a65aq_lambda32_vs_lambda48": decision_relation(seed0_l32, seed0_l48),
            "stage4a65as_start_corridor_tree_seed1_lambda48": seed1_l48,
            "stage4a65as_measured_vs_lambda48": decision_relation(seed1_measured, seed1_l48),
            "stage4a65as_lambda32_vs_lambda48": decision_relation(seed1_l32, seed1_l48),
        },
        "reference_health": {
            "seed0_low_cost_artifact": bool(seed0_l48.get("low_cost_artifact")),
            "seed1_low_cost_artifact": bool(seed1_l48.get("low_cost_artifact")),
            "seed0_historical_prior_basin": bool(aq["branch"][frame].get("historical_prior_basin")),
            "seed1_historical_prior_basin": bool(ass["branch"][frame].get("historical_prior_basin")),
            "seed0_classification": aq["branch"][frame].get("classification"),
            "seed1_classification": ass["branch"][frame].get("classification"),
            "seed1_distinct_nonmeasured_reference_clean": bool(ass["branch"][frame].get("healthy_nonmeasured_candidate"))
            and bool(ass["branch"][frame].get("distinct_nonmeasured_branch"))
            and not bool(seed1_l48.get("low_cost_artifact"))
            and not bool(ass["branch"][frame].get("historical_prior_basin")),
        },
        "interpretation": (
            "No start_room_b tree was run in this design stage. The future bounded smoke must record "
            "lambda48/measured/lambda32 tree decisions and SC gain components for this frame before any expert data sampling."
        ),
    }


def runtime_timing_reference_rows(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stage in stages:
        timing = stage.get("hardware", {}).get("timing", {})
        for key in (
            "frame001_capture_time_s",
            "frame001_observed_state_update_time_s",
            "map_predict_frame001_total_time_s",
            "frame001_measured_shadow_tree_time_s",
            "frame001_lambda48_primary_tree_time_s",
            "frame001_lambda32_shadow_tree_time_s",
            "frame002_capture_time_s",
            "frame002_observed_state_update_time_s",
            "map_predict_frame002_total_time_s",
            "frame002_measured_shadow_tree_time_s",
            "frame002_lambda48_primary_tree_time_s",
            "frame002_lambda32_shadow_tree_time_s",
            "total_wall_time_s",
        ):
            if key in timing:
                rows.append({"stage": stage["label"], "tree_seed": stage["tree_seed"], "timer": key, "time_s": timing[key]})
    return rows


def collect_input_paths(args: argparse.Namespace) -> list[tuple[str, Path, bool]]:
    paths: list[tuple[str, Path, bool]] = [
        ("stage4a65aq summary", args.stage4a65aq_dir / "stage4a65aq_alternate_start_summary.json", True),
        ("stage4a65aq observed delta", args.stage4a65aq_dir / "observed_state_delta_summary.json", True),
        ("stage4a65aq map stability", args.stage4a65aq_dir / "map_predict_two_frame_stability.json", True),
        ("stage4a65aq frame1 lambda48", args.stage4a65aq_dir / "frame001_lambda48_primary_tree_decision.json", True),
        ("stage4a65aq frame2 lambda48", args.stage4a65aq_dir / "frame002_lambda48_diagnostic_tree_decision.json", True),
        ("stage4a65as summary", args.stage4a65as_dir / "stage4a65as_start_corridor_seed1_summary.json", True),
        ("stage4a65as observed delta", args.stage4a65as_dir / "observed_state_delta_summary.json", True),
        ("stage4a65as map stability", args.stage4a65as_dir / "map_predict_two_frame_stability.json", True),
        ("stage4a65as frame1 lambda48", args.stage4a65as_dir / "frame001_lambda48_primary_tree_decision.json", True),
        ("stage4a65as frame2 lambda48", args.stage4a65as_dir / "frame002_lambda48_diagnostic_tree_decision.json", True),
        ("stage4a65ar summary", args.stage4a65ar_dir / "stage4a65ar_alternate_start_diagnosis_summary.json", True),
        ("stage4a65ar selected design", args.stage4a65ar_dir / "selected_next_bounded_repeat_design.json", True),
        ("stage4a65ap summary", args.stage4a65ap_dir / "stage4a65ap_seed012_repeat_review_summary.json", True),
        ("stage4a65ap selected alternate start", args.stage4a65ap_dir / "selected_alternate_start_design.json", True),
        ("stage4a65at summary", args.stage4a65at_dir / "stage4a65at_start_corridor_seed01_review_summary.json", True),
        ("stage4a65at selected next start", args.stage4a65at_dir / "selected_next_start_design.json", True),
        ("start_room_b metadata", args.start_room_b_metadata, True),
        ("start_corridor metadata", args.start_corridor_metadata, True),
        ("checkpoint", CHECKPOINT, True),
        ("canonical seed0 summary", args.canonical_seed0_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json", False),
        ("canonical seed1 summary", args.canonical_seed1_dir / "stage4a65am_bounded_repeat_safety_summary.json", False),
        ("canonical seed2 summary", args.canonical_seed2_dir / "stage4a65ao_bounded_repeat_safety_summary.json", False),
    ]
    for path in CONTEXT_FILES:
        paths.append((f"context {path.name}", path, True))
    for base, label in (
        (args.canonical_seed0_dir, "canonical seed0"),
        (args.canonical_seed1_dir, "canonical seed1"),
        (args.canonical_seed2_dir, "canonical seed2"),
    ):
        paths.extend(
            [
                (f"{label} frame1 lambda48", base / "frame001_lambda48_primary_tree_decision.json", False),
                (f"{label} frame2 lambda48", base / "frame002_lambda48_diagnostic_tree_decision.json", False),
                (f"{label} frame1 branch", base / "frame001_branch_classification.json", False),
                (f"{label} frame2 branch", base / "frame002_branch_classification.json", False),
                (f"{label} hardware", base / "hardware_utilization_report.json", False),
            ]
        )
    return paths


def write_loaded_context_manifest(output_dir: Path) -> dict[str, Any]:
    entries = []
    combined = ""
    for path in CONTEXT_FILES:
        exists = path.is_file()
        text = path.read_text(encoding="utf-8") if exists else ""
        combined += "\n" + text
        record = file_record(path)
        entries.append(
            {
                **record,
                "contains_stage4a65aq": "Stage 4A-6.5aq" in text,
                "contains_stage4a65as": "Stage 4A-6.5as" in text,
                "contains_stage4a65at": "Stage 4A-6.5at" in text,
                "contains_stage4a65au": "Stage 4A-6.5au" in text,
            }
        )
    manifest = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "loaded_context_files": entries,
        "confirmed_prior_chain_mentions": {
            "stage4a65aq": "Stage 4A-6.5aq" in combined,
            "stage4a65as": "Stage 4A-6.5as" in combined,
            "stage4a65at": "Stage 4A-6.5at" in combined,
        },
        "chat_history_not_used_as_source": True,
    }
    write_json(output_dir / "loaded_context_manifest.json", manifest)
    lines = [
        "- Stage 4A-6.5au is design/review only.",
        "- Files read:",
        *[f"  - `{item['path']}` exists {md_bool(item['exists'])} sha256 `{item.get('sha256', '')}`" for item in entries],
    ]
    write_text(output_dir / "loaded_context_manifest.md", md_doc("Loaded Context Manifest", lines))
    return manifest


def write_loaded_input_manifest(
    output_dir: Path,
    paths_by_role: list[tuple[str, Path, bool]],
    hash_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    files = []
    for role, path, essential in paths_by_role:
        files.append({"role": role, "essential": essential, **hash_records.get(str(path), file_record(path))})
    manifest = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "files": files,
        "loaded_stage4a65aq": any(item["role"].startswith("stage4a65aq") and item["exists"] for item in files),
        "loaded_stage4a65as": any(item["role"].startswith("stage4a65as") and item["exists"] for item in files),
        "loaded_stage4a65ar": any(item["role"].startswith("stage4a65ar") and item["exists"] for item in files),
        "loaded_stage4a65ap": any(item["role"].startswith("stage4a65ap") and item["exists"] for item in files),
        "loaded_stage4a65at": any(item["role"].startswith("stage4a65at") and item["exists"] for item in files),
        "loaded_start_room_b_metadata": any(item["role"] == "start_room_b metadata" and item["exists"] for item in files),
        "canonical_start_references_context_only": {
            "stage4a65ak": any(item["role"].startswith("canonical seed0") and item["exists"] for item in files),
            "stage4a65am": any(item["role"].startswith("canonical seed1") and item["exists"] for item in files),
            "stage4a65ao": any(item["role"].startswith("canonical seed2") and item["exists"] for item in files),
        },
    }
    write_json(output_dir / "loaded_input_manifest.json", manifest)
    rows = [
        {
            "role": item["role"],
            "essential": item["essential"],
            "exists": item["exists"],
            "size_bytes": item.get("size_bytes", ""),
            "sha256": item.get("sha256", ""),
            "path": item["path"],
        }
        for item in files
    ]
    write_text(
        output_dir / "loaded_input_manifest.md",
        "# Loaded Input Manifest\n\n" + md_table(rows, ["role", "essential", "exists", "size_bytes", "sha256", "path"]),
    )
    return manifest


def build_hardware_report(
    args: argparse.Namespace,
    hash_time_s: float,
    wall_time_so_far_s: float,
    timing_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    gpu = query_gpu()
    return {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "runtime_executed": False,
        "os_cpu_count": os.cpu_count(),
        "requested_max_workers": args.max_workers,
        "actual_max_workers": max(1, min(args.max_workers, os.cpu_count() or 1)),
        "parallel_backend_used_in_design_stage": "process_pool_for_file_hash_audit_only",
        "process_pool_hash_time_s": hash_time_s,
        "design_wall_time_so_far_s": wall_time_so_far_s,
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        },
        "future_runtime_hardware_policy": {
            "cpu_threads_available": 32,
            "gpu_target": "RTX 5080",
            "blas_omp_threads": 1,
            "max_workers": 32,
            "parallel_backend": "bounded process pool where safe; single Isaac/runtime process for simulator and action side effects",
            "avoid_overload": "do not run multiple Isaac apps; keep OMP/BLAS/MKL/NUMEXPR/VECLIB at 1 inside worker processes",
        },
        "gpu": gpu,
        "reference_timing_rows": timing_rows,
    }


def plot_layout(
    path: Path,
    metadata: dict[str, Any],
    start_room_b: dict[str, Any],
    selected_design: dict[str, Any],
) -> None:
    bounds = metadata.get("map_bounds", {"x": [-6, 6], "y": [-6, 6]})
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    for room in metadata.get("rooms", []):
        rb = room.get("bounds", {})
        if "x" in rb and "y" in rb:
            x0, x1 = rb["x"]
            y0, y1 = rb["y"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#f2f2f2", edgecolor="#777", lw=1.0))
            ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0, room.get("label", room.get("name", "")), ha="center", va="center", fontsize=9)
    for obs in metadata.get("obstacles", []):
        pos = obs.get("position")
        size = obs.get("size")
        if pos and size:
            ax.add_patch(
                Rectangle(
                    (pos[0] - size[0] / 2.0, pos[1] - size[1] / 2.0),
                    size[0],
                    size[1],
                    facecolor="#b8c0c8",
                    edgecolor="#59636e",
                    alpha=0.55,
                )
            )
    for door in metadata.get("doors", []):
        rect = door.get("clear_rect", {})
        if "x" in rect and "y" in rect:
            x0, x1 = rect["x"]
            y0, y1 = rect["y"]
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="#ffffff", edgecolor="#2a9d8f", lw=1.5))
    points = [
        ("canonical", CANONICAL_START, "#444", "o"),
        ("start_corridor", START_CORRIDOR_POSITION, "#457b9d", "s"),
        ("start_room_b", start_room_b.get("position"), "#e76f51", "^"),
    ]
    for label, pos, color, marker in points:
        if pos:
            ax.scatter([pos[0]], [pos[1]], s=90, c=color, marker=marker, label=label, zorder=5)
            ax.text(pos[0] + 0.08, pos[1] + 0.08, label, fontsize=9)
    if start_room_b.get("position") is not None and start_room_b.get("yaw_rad") is not None:
        pos = start_room_b["position"]
        yaw = float(start_room_b["yaw_rad"])
        ax.arrow(pos[0], pos[1], 0.6 * math.cos(yaw), 0.6 * math.sin(yaw), width=0.02, color="#e76f51", length_includes_head=True)
    ax.set_title("Stage 4A-6.5au start_room_b design")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reference_tree(path: Path, title: str, frame: str, stages: list[dict[str, Any]], bounds: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(bounds.get("x", [-6, 6]))
    ax.set_ylim(bounds.get("y", [-6, 6]))
    colors = {"stage4a65aq": "#2a9d8f", "stage4a65as": "#e76f51", "stage4a65ak": "#457b9d", "stage4a65am": "#f4a261", "stage4a65ao": "#6d597a"}
    for stage in stages:
        dec = stage["decision"][frame]["lambda48"]
        if not dec.get("available"):
            continue
        color = colors.get(stage["label"], "#333")
        selected = dec.get("selected_child_world")
        best = dec.get("best_descendant_world")
        if selected:
            ax.scatter([selected[0]], [selected[1]], s=80, c=color, marker="o", label=f"{stage['label']} selected")
            ax.text(selected[0] + 0.06, selected[1] + 0.06, f"{stage['label']} sel", fontsize=7)
        if best:
            ax.scatter([best[0]], [best[1]], s=80, c=color, marker="s", label=f"{stage['label']} best")
            ax.text(best[0] + 0.06, best[1] + 0.06, f"{stage['label']} best", fontsize=7)
        if selected and best:
            ax.plot([selected[0], best[0]], [selected[1], best[1]], color=color, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str, rotate: int = 25) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261", "#6d597a", "#8ab17d"] * 4
    ax.bar(labels, values, color=colors[: len(values)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_gain_rows(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    labels = [f"{row['stage']} {row['mode']}" for row in rows]
    x = np.arange(len(labels))
    gain = np.asarray([float(row.get("gain_exp") or 0.0) for row in rows], dtype=np.float64)
    sc = np.asarray([float(row.get("source_occ_free") or 0.0) for row in rows], dtype=np.float64)
    final_value = np.asarray([float(row.get("final_value") or 0.0) for row in rows], dtype=np.float64)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].bar(x - 0.2, gain, width=0.4, color="#457b9d", label="gain_exp")
    axes[0].bar(x + 0.2, sc / 20.0, width=0.4, color="#e76f51", label="source_occ_free / 20")
    axes[0].set_ylabel("gain components")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, final_value, color="#2a9d8f", label="final_value")
    axes[1].set_ylabel("final value")
    axes[1].set_xticks(x, labels, rotation=35, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_readiness(path: Path, readiness_rows: list[dict[str, Any]]) -> None:
    values = np.asarray([[1.0 if row["passed"] else 0.0 for row in readiness_rows]], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11, 2.8))
    im = ax.imshow(values, cmap="YlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(readiness_rows)), [row["check"] for row in readiness_rows], rotation=45, ha="right")
    ax.set_yticks([0], ["6.5au design"])
    for j, row in enumerate(readiness_rows):
        ax.text(j, 0, "Y" if row["passed"] else "N", ha="center", va="center", color="#222")
    ax.set_title("Stage 4A-6.5au readiness matrix")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_timeline(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    items = [
        (0.07, "Frame1 capture\nobserved delta\nmap_predict\nlambda48 action"),
        (0.38, "Execute exactly\none selected action\nif gates pass"),
        (0.65, "Frame2 capture\nobserved delta\nmap_predict\ndiagnostic tree"),
        (0.88, "Frame3\nnot captured\nno rollout"),
    ]
    for x, text in items:
        ax.text(
            x,
            0.55,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#f7f7f7", edgecolor="#555"),
        )
    for x0, x1 in ((0.19, 0.30), (0.49, 0.56), (0.76, 0.81)):
        ax.annotate("", xy=(x1, 0.55), xytext=(x0, 0.55), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title("Future bounded two-frame one-action sequence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scan_forbidden_artifacts(output_dir: Path) -> dict[str, Any]:
    found = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        matches = sorted(output_dir.glob(pattern))
        if matches:
            found.append({"pattern": pattern, "matches": [str(path) for path in matches]})
    return {
        "stage": "Stage 4A-6.5au",
        "clean": len(found) == 0,
        "prohibited_artifacts_found": found,
        "isaac_startup_in_stage4a65au": False,
        "rgb_depth_capture_in_stage4a65au": False,
        "map_predict_call_in_stage4a65au": False,
        "sscnet_inference_in_stage4a65au": False,
        "selected_action_execution_in_stage4a65au": False,
        "two_frame_runtime_execution_in_stage4a65au": False,
        "rollout_in_stage4a65au": False,
        "rl_gdpo_ppo_bc_il_training_in_stage4a65au": False,
        "future_command_executed": False,
        "runtime_runner_created": False,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65aq_dir", type=Path, default=DEFAULT_AQ_DIR)
    parser.add_argument("--stage4a65as_dir", type=Path, default=DEFAULT_AS_DIR)
    parser.add_argument("--stage4a65ar_dir", type=Path, default=DEFAULT_AR_DIR)
    parser.add_argument("--stage4a65ap_dir", type=Path, default=DEFAULT_AP_DIR)
    parser.add_argument("--stage4a65at_dir", type=Path, default=DEFAULT_AT_DIR)
    parser.add_argument("--canonical_seed0_dir", type=Path, default=DEFAULT_AK_DIR)
    parser.add_argument("--canonical_seed1_dir", type=Path, default=DEFAULT_AM_DIR)
    parser.add_argument("--canonical_seed2_dir", type=Path, default=DEFAULT_AO_DIR)
    parser.add_argument("--start_room_b_metadata", type=Path, default=DEFAULT_START_ROOM_B_METADATA)
    parser.add_argument("--start_corridor_metadata", type=Path, default=DEFAULT_START_CORRIDOR_METADATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tree_seed", type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--save_viz", action="store_true")
    return parser


def main() -> int:
    start_time = time.perf_counter()
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_workers = max(1, min(int(args.max_workers), os.cpu_count() or 1))

    context_manifest = write_loaded_context_manifest(output_dir)
    paths_by_role = collect_input_paths(args)
    hash_start = time.perf_counter()
    hash_records = hash_many([path for _, path, _ in paths_by_role], actual_workers)
    hash_time_s = time.perf_counter() - hash_start
    input_manifest = write_loaded_input_manifest(output_dir, paths_by_role, hash_records)
    write_json(output_dir / "input_hash_audit.json", {"stage": "Stage 4A-6.5au", "hash_records": hash_records})
    write_text(
        output_dir / "input_hash_audit.md",
        md_doc(
            "Input Hash Audit",
            [
                f"- Files hashed: `{len(hash_records)}`.",
                f"- Process-pool workers: `{actual_workers}`.",
                f"- Hash audit time: `{hash_time_s}` s.",
            ],
        ),
    )

    missing: list[dict[str, Any]] = []
    aq = load_stage(
        args.stage4a65aq_dir,
        "stage4a65aq",
        0,
        "stage4a65aq_alternate_start_summary.json",
        "start_corridor",
        missing,
    )
    ass = load_stage(
        args.stage4a65as_dir,
        "stage4a65as",
        1,
        "stage4a65as_start_corridor_seed1_summary.json",
        "start_corridor",
        missing,
    )
    ak = load_stage(
        args.canonical_seed0_dir,
        "stage4a65ak",
        0,
        "stage4a65ak_two_frame_one_action_runtime_summary.json",
        "canonical_stage4a65p_frame1",
        missing,
    )
    am = load_stage(
        args.canonical_seed1_dir,
        "stage4a65am",
        1,
        "stage4a65am_bounded_repeat_safety_summary.json",
        "canonical_stage4a65p_frame1",
        missing,
    )
    ao = load_stage(
        args.canonical_seed2_dir,
        "stage4a65ao",
        2,
        "stage4a65ao_bounded_repeat_safety_summary.json",
        "canonical_stage4a65p_frame1",
        missing,
    )
    ar_design = read_json_optional(args.stage4a65ar_dir / "selected_next_bounded_repeat_design.json", missing, True, "stage4a65ar selected design") or {}
    ap_design = read_json_optional(args.stage4a65ap_dir / "selected_alternate_start_design.json", missing, True, "stage4a65ap selected alternate start") or {}
    at_design = read_json_optional(args.stage4a65at_dir / "selected_next_start_design.json", missing, True, "stage4a65at selected next start") or {}
    start_room_b_metadata = read_json_optional(args.start_room_b_metadata, missing, True, "start_room_b metadata") or {}
    start_corridor_metadata = read_json_optional(args.start_corridor_metadata, missing, False, "start_corridor metadata") or {}
    start_room_b = extract_start_pose_from_metadata(start_room_b_metadata, EXPECTED_START_VARIANT)
    map_bounds = start_room_b_metadata.get("map_bounds") or {"x": [-6, 6], "y": [-6, 6], "z": [0, 3]}

    pose_consistency = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "metadata_path": str(args.start_room_b_metadata),
        "pose_found": bool(start_room_b.get("found")),
        "metadata_scene_variant": start_room_b_metadata.get("scene_variant"),
        "metadata_scene_seed": start_room_b_metadata.get("scene_seed"),
        "metadata_start_variant": start_room_b_metadata.get("start_variant"),
        "extracted_start_pose": start_room_b,
        "expected_position": EXPECTED_START_POSITION,
        "expected_yaw_rad": EXPECTED_START_YAW,
        "position_matches_expected": close_list(start_room_b.get("position"), EXPECTED_START_POSITION),
        "yaw_matches_expected": math.isclose(float(start_room_b.get("yaw_rad", float("nan"))), EXPECTED_START_YAW, abs_tol=1.0e-12),
        "scene_variant_matches": start_room_b_metadata.get("scene_variant") == EXPECTED_SCENE_VARIANT,
        "scene_seed_matches": start_room_b_metadata.get("scene_seed") == EXPECTED_SCENE_SEED,
        "start_variant_matches": start_room_b_metadata.get("start_variant") == EXPECTED_START_VARIANT,
        "distance_from_start_corridor_m": distance_m(start_room_b.get("position"), START_CORRIDOR_POSITION),
        "distance_from_canonical_start_m": distance_m(start_room_b.get("position"), CANONICAL_START),
    }
    pose_consistency["all_pose_checks_passed"] = bool(
        pose_consistency["pose_found"]
        and pose_consistency["position_matches_expected"]
        and pose_consistency["yaw_matches_expected"]
        and pose_consistency["scene_variant_matches"]
        and pose_consistency["scene_seed_matches"]
        and pose_consistency["start_variant_matches"]
    )
    write_json(output_dir / "start_room_b_pose_consistency.json", pose_consistency)
    write_text(
        output_dir / "start_room_b_pose_consistency.md",
        md_doc(
            "Start Room B Pose Consistency",
            [
                f"- Pose found: {md_bool(pose_consistency['pose_found'])}",
                f"- Position matches `[2.75, -2.55, 1.2]`: {md_bool(pose_consistency['position_matches_expected'])}",
                f"- Yaw matches `2.7052603405912112`: {md_bool(pose_consistency['yaw_matches_expected'])}",
                f"- Scene/seed/start variant match: {md_bool(pose_consistency['scene_variant_matches'] and pose_consistency['scene_seed_matches'] and pose_consistency['start_variant_matches'])}",
                f"- Distance from start_corridor: `{pose_consistency['distance_from_start_corridor_m']}` m.",
            ],
        ),
    )

    sequence_safety = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "runtime_activity_in_stage4a65au": {
            "isaac_startup": False,
            "rgb_depth_capture": False,
            "map_predict_call": False,
            "sscnet_inference": False,
            "selected_action_execution": False,
            "two_frame_runtime_execution": False,
            "rollout": False,
            "rl_gdpo_ppo_bc_il": False,
        },
        "future_runtime_sequence_contract": {
            "scene_variant": EXPECTED_SCENE_VARIANT,
            "scene_seed": EXPECTED_SCENE_SEED,
            "start_variant": EXPECTED_START_VARIANT,
            "position": EXPECTED_START_POSITION,
            "yaw_rad": EXPECTED_START_YAW,
            "tree_seed": args.tree_seed,
            "frame1": "capture, observed_state delta, map_predict read-only, lambda48 primary tree, measured/lambda32 shadows, execute exactly one selected action if gates pass",
            "frame2": "capture, observed_state delta, map_predict read-only, lambda48 diagnostic tree, measured/lambda32 shadows, no action",
            "frame3": "not captured",
            "max_frames": 2,
            "max_map_predict_calls": 2,
            "max_selected_actions": 1,
            "no_second_action": True,
            "no_third_frame": True,
            "no_rollout": True,
        },
        "reference_sequences_clean": {
            "stage4a65aq": stage_sequence_clean(aq),
            "stage4a65as": stage_sequence_clean(ass),
            "stage4a65ak": stage_sequence_clean(ak),
            "stage4a65am": stage_sequence_clean(am),
            "stage4a65ao": stage_sequence_clean(ao),
        },
    }
    write_json(output_dir / "sequence_safety_reverification.json", sequence_safety)
    write_text(
        output_dir / "sequence_safety_reverification.md",
        md_doc(
            "Sequence Safety Reverification",
            [
                "- Stage 4A-6.5au performed no runtime actions.",
                "- Future sequence is bounded to Frame1 capture/action and Frame2 capture/diagnostic.",
                "- Frame3 is explicitly not captured.",
                "- Rollout, open-ended loops, RL, GDPO, PPO, BC, and IL remain forbidden.",
            ],
        ),
    )

    prediction_safety = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "runtime_executed": False,
        "future_map_predict_policy": {
            "read_only": True,
            "information_gain_only": True,
            "written_to_observed_state": False,
            "fused_into_observed_state": False,
            "used_for_traversability": False,
            "used_for_collision": False,
            "used_for_ray_blocking": False,
            "used_for_edge_validity": False,
            "used_for_candidate_sampling": False,
            "used_target_ground_truth_or_future_observed_for_planning_scoring": False,
        },
        "reference_prediction_safety_clean": {
            "stage4a65aq": prediction_clean(aq),
            "stage4a65as": prediction_clean(ass),
            "stage4a65ak": prediction_clean(ak),
            "stage4a65am": prediction_clean(am),
            "stage4a65ao": prediction_clean(ao),
        },
        "prohibited_primary_formulas": PROHIBITED_FORMULAS,
        "primary_formula": PRIMARY_FORMULA,
        "shadow_formula": SHADOW_FORMULA,
        "over_cost_primary_allowed": False,
    }
    write_json(output_dir / "prediction_safety_review.json", prediction_safety)
    write_text(
        output_dir / "prediction_safety_review.md",
        md_doc(
            "Prediction Safety Review",
            [
                "- map_predict is read-only and information-gain-only in the future design.",
                "- It must not write/fuse into observed_state or affect motion safety, edge validity, or candidate sampling.",
                f"- Primary formula: `{PRIMARY_FORMULA}`.",
                f"- Shadow formula: `{SHADOW_FORMULA}`.",
            ],
        ),
    )

    no_rollout = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "runtime_executed": False,
        "rollout": False,
        "open_ended_loop": False,
        "frame003_captured": False,
        "second_action_executed": False,
        "rl_gdpo_ppo_bc_il": False,
        "reference_no_rollout_clean": {
            "stage4a65aq": no_rollout_clean(aq),
            "stage4a65as": no_rollout_clean(ass),
            "stage4a65ak": no_rollout_clean(ak),
            "stage4a65am": no_rollout_clean(am),
            "stage4a65ao": no_rollout_clean(ao),
        },
    }
    write_json(output_dir / "no_rollout_reverification.json", no_rollout)
    write_text(
        output_dir / "no_rollout_reverification.md",
        md_doc(
            "No Rollout Reverification",
            [
                "- This stage did not run rollout or an open-ended loop.",
                "- Future command remains bounded smoke only.",
                "- Formal expert data sampling is blocked until the bounded smoke review passes.",
            ],
        ),
    )

    reference_stages = [aq, ass]
    canonical_stages = [ak, am, ao]
    all_stages = [aq, ass, ak, am, ao]
    observed_rows = []
    for stage in all_stages:
        obs = stage.get("observed", {})
        if not obs:
            continue
        observed_rows.append(
            {
                "stage": stage["label"],
                "tree_seed": stage["tree_seed"],
                "start_variant": stage["start_variant"],
                "frame001_observed_ratio": get_path(obs, ["frame001", "observed_ratio"]),
                "frame002_observed_ratio": get_path(obs, ["frame002", "observed_ratio"]),
                "observed_ratio_delta": obs.get("observed_ratio_delta"),
                "newly_observed": obs.get("newly_observed"),
                "unknown_to_free": obs.get("unknown_to_free"),
                "unknown_to_occupied": obs.get("unknown_to_occupied"),
                "occupied_to_free": obs.get("occupied_to_free"),
                "invalid_labels": obs.get("invalid_labels"),
                "measured_only_status": obs.get("measured_only_status"),
            }
        )
    observed_summary = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "start_room_b_runtime_observed_state_available": False,
        "future_logging_contract": {
            "frame1": {
                "capture": True,
                "observed_state_delta_required": True,
                "required_fields": [
                    "free",
                    "occupied",
                    "unknown",
                    "observed",
                    "observed_ratio",
                    "unknown_to_free",
                    "unknown_to_occupied",
                    "occupied_to_free",
                    "free_to_occupied",
                    "invalid_labels",
                    "update_time_s",
                ],
            },
            "frame2": {
                "capture": True,
                "observed_state_delta_required": True,
                "required_fields": [
                    "free",
                    "occupied",
                    "unknown",
                    "observed",
                    "observed_ratio",
                    "newly_observed",
                    "observed_ratio_delta",
                    "unknown_to_free",
                    "unknown_to_occupied",
                    "occupied_to_free",
                    "free_to_occupied",
                    "invalid_labels",
                    "update_time_s",
                ],
            },
            "frame3": {"capture": False, "observed_state_delta_required": False},
        },
        "acceptance_gates": {
            "measured_only_status_must_be_true": True,
            "prediction_writeback_must_be_false": True,
            "invalid_labels_must_be_zero": True,
            "occupied_to_free_must_be_zero": True,
        },
        "reference_rows": observed_rows,
    }
    write_json(output_dir / "observed_state_delta_summary.json", observed_summary)
    write_text(
        output_dir / "observed_state_delta_summary.md",
        "# Observed State Delta Summary\n\n"
        + md_table(
            observed_rows,
            [
                "stage",
                "tree_seed",
                "start_variant",
                "frame001_observed_ratio",
                "frame002_observed_ratio",
                "observed_ratio_delta",
                "newly_observed",
                "invalid_labels",
                "measured_only_status",
            ],
        )
        + "\n\n- Stage 4A-6.5au did not capture start_room_b frames; this file defines the required future frame logs.",
    )

    map_rows = []
    for stage in all_stages:
        maps = stage.get("maps", {})
        if not maps:
            continue
        for frame in ("frame001", "frame002"):
            if f"{frame}_prediction_valid_count" not in maps:
                continue
            map_rows.append(
                {
                    "stage": stage["label"],
                    "tree_seed": stage["tree_seed"],
                    "start_variant": stage["start_variant"],
                    "frame": frame,
                    "prediction_valid_count": maps.get(f"{frame}_prediction_valid_count"),
                    "predicted_unmeasured_occ_free": maps.get(f"{frame}_predicted_unmeasured_occ_free"),
                    "predicted_free_count": maps.get(f"{frame}_predicted_free_count"),
                    "predicted_occupied_count": maps.get(f"{frame}_predicted_occupied_count"),
                    "alignment_convention": maps.get(f"alignment_convention_{frame}"),
                }
            )
    map_summary = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "start_room_b_runtime_map_predict_available": False,
        "future_logging_contract": {
            "frame1": {
                "map_predict_call": True,
                "must_log_timing_s": True,
                "must_log_global_prediction_layer_hash": True,
                "must_log_counts": [
                    "prediction_valid_count",
                    "predicted_free_count",
                    "predicted_occupied_count",
                    "predicted_unmeasured_occ_free",
                ],
            },
            "frame2": {
                "map_predict_call": True,
                "must_log_timing_s": True,
                "must_log_global_prediction_layer_hash": True,
                "must_log_two_frame_density_ratio": True,
            },
            "frame3": {"map_predict_call": False},
        },
        "acceptance_gates": {
            "code_consistent_v1_check": True,
            "prediction_read_only": True,
            "no_explosion_or_collapse": True,
            "density_ratio_frame2_over_frame1_reasonable": True,
        },
        "reference_rows": map_rows,
    }
    write_json(output_dir / "map_predict_two_frame_stability.json", map_summary)
    write_text(
        output_dir / "map_predict_two_frame_stability.md",
        "# Map Predict Two-Frame Stability\n\n"
        + md_table(
            map_rows,
            [
                "stage",
                "tree_seed",
                "start_variant",
                "frame",
                "prediction_valid_count",
                "predicted_unmeasured_occ_free",
                "alignment_convention",
            ],
        )
        + "\n\n- Stage 4A-6.5au did not call map_predict; this is the future logging and stability contract.",
    )

    frame1_diag = build_frame_diagnosis("frame001", 1, aq, ass)
    frame2_diag = build_frame_diagnosis("frame002", 2, aq, ass)
    write_json(output_dir / "frame1_tree_decision_diagnosis.json", frame1_diag)
    write_json(output_dir / "frame2_tree_decision_diagnosis.json", frame2_diag)
    for name, title, payload in (
        ("frame1_tree_decision_diagnosis.md", "Frame1 Tree Decision Diagnosis", frame1_diag),
        ("frame2_tree_decision_diagnosis.md", "Frame2 Tree Decision Diagnosis", frame2_diag),
    ):
        ref = payload["reference_health"]
        write_text(
            output_dir / name,
            md_doc(
                title,
                [
                    "- No start_room_b tree was run in this design stage.",
                    f"- Future tree_seed: `{payload['tree_seed']}`.",
                    f"- Formula: `{payload['primary_formula']}`.",
                    f"- Reference seed0 classification: `{ref['seed0_classification']}`.",
                    f"- Reference seed1 classification: `{ref['seed1_classification']}`.",
                    f"- Reference seed1 distinct non-measured clean: {md_bool(ref['seed1_distinct_nonmeasured_reference_clean'])}",
                    "- Future expert-template promotion requires healthy distinct_nonmeasured, low_cost_artifact false, and historical_prior_basin false.",
                ],
            ),
        )

    gain_rows = value_rows_for_frame(reference_stages, "frame001") + value_rows_for_frame(reference_stages, "frame002")
    write_csv(output_dir / "tree_sc_gain_logging_contract.csv", gain_rows)
    write_json(
        output_dir / "tree_sc_gain_logging_contract.json",
        {
            "stage": "Stage 4A-6.5au",
            "design_review_only": True,
            "start_room_b_actual_gain_rows_available": False,
            "required_future_fields": [
                "gain_exp",
                "cost",
                "source_occ_free",
                "normalized_sc",
                "sc_bonus",
                "final_value",
                "runner_up_value",
                "margin",
                "candidate_count",
                "tree_total_nodes",
                "timing_s",
            ],
            "reference_rows": gain_rows,
        },
    )
    write_text(
        output_dir / "tree_sc_gain_logging_contract.md",
        "# Tree SC Gain Logging Contract\n\n"
        + md_table(
            gain_rows,
            [
                "stage",
                "tree_seed",
                "frame",
                "mode",
                "branch_classification",
                "gain_exp",
                "source_occ_free",
                "normalized_sc",
                "sc_bonus",
                "final_value",
                "timing_s",
            ],
        ),
    )

    low_cost_flags = {
        "stage4a65aq_frame001": bool(aq["low_cost"]["frame001"].get("low_cost_artifact")),
        "stage4a65aq_frame002": bool(aq["low_cost"]["frame002"].get("low_cost_artifact")),
        "stage4a65as_frame001": bool(ass["low_cost"]["frame001"].get("low_cost_artifact")),
        "stage4a65as_frame002": bool(ass["low_cost"]["frame002"].get("low_cost_artifact")),
        "stage4a65ak_frame001": bool(ak["low_cost"]["frame001"].get("low_cost_artifact")),
        "stage4a65ak_frame002": bool(ak["low_cost"]["frame002"].get("low_cost_artifact")),
        "stage4a65am_frame001": bool(am["low_cost"]["frame001"].get("low_cost_artifact")),
        "stage4a65am_frame002": bool(am["low_cost"]["frame002"].get("low_cost_artifact")),
        "stage4a65ao_frame001": bool(ao["low_cost"]["frame001"].get("low_cost_artifact")),
        "stage4a65ao_frame002": bool(ao["low_cost"]["frame002"].get("low_cost_artifact")),
    }
    low_cost = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "start_room_b_runtime_branch_available": False,
        "low_cost_artifact": False,
        "meaning": "No Stage 4A-6.5au runtime branch was selected; the offline design itself has no low-cost artifact.",
        "future_runtime_low_cost_artifact_must_be_false": True,
        "reference_low_cost_artifact_any": any(low_cost_flags.values()),
        "reference_flags": low_cost_flags,
        "acceptance": "If the future start_room_b lambda48 winner is a low-cost artifact, do not promote to expert sampling.",
    }
    write_json(output_dir / "low_cost_artifact_review.json", low_cost)
    write_text(
        output_dir / "low_cost_artifact_review.md",
        md_doc(
            "Low-Cost Artifact Review",
            [
                f"- Offline design low-cost artifact: {md_bool(low_cost['low_cost_artifact'])}",
                f"- Reference low-cost artifact any: {md_bool(low_cost['reference_low_cost_artifact_any'])}",
                "- Future start_room_b runtime acceptance requires `low_cost_artifact=false` on both frames.",
            ],
        ),
    )

    prior_flags = {
        "stage4a65aq_frame001": bool(aq["branch"]["frame001"].get("historical_prior_basin")),
        "stage4a65aq_frame002": bool(aq["branch"]["frame002"].get("historical_prior_basin")),
        "stage4a65as_frame001": bool(ass["branch"]["frame001"].get("historical_prior_basin")),
        "stage4a65as_frame002": bool(ass["branch"]["frame002"].get("historical_prior_basin")),
        "stage4a65ak_frame001": bool(ak["branch"]["frame001"].get("historical_prior_basin")),
        "stage4a65ak_frame002": bool(ak["branch"]["frame002"].get("historical_prior_basin")),
        "stage4a65am_frame001": bool(am["branch"]["frame001"].get("historical_prior_basin")),
        "stage4a65am_frame002": bool(am["branch"]["frame002"].get("historical_prior_basin")),
        "stage4a65ao_frame001": bool(ao["branch"]["frame001"].get("historical_prior_basin")),
        "stage4a65ao_frame002": bool(ao["branch"]["frame002"].get("historical_prior_basin")),
    }
    prior = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "start_room_b_runtime_branch_available": False,
        "historical_prior_basin": False,
        "meaning": "No Stage 4A-6.5au runtime branch was selected; the offline design itself is not in the historical prior basin.",
        "historical_prior_selected_grid": HISTORICAL_PRIOR_SELECTED_GRID,
        "historical_prior_best_grid": HISTORICAL_PRIOR_BEST_GRID,
        "reference_historical_prior_basin_any": any(prior_flags.values()),
        "reference_flags": prior_flags,
        "future_runtime_historical_prior_basin_must_be_false": True,
    }
    write_json(output_dir / "historical_prior_basin_review.json", prior)
    write_text(
        output_dir / "historical_prior_basin_review.md",
        md_doc(
            "Historical Prior Basin Review",
            [
                f"- Offline design historical prior basin: {md_bool(prior['historical_prior_basin'])}",
                f"- Reference historical prior basin any: {md_bool(prior['reference_historical_prior_basin_any'])}",
                "- Future start_room_b runtime acceptance requires `historical_prior_basin=false` on both frames.",
            ],
        ),
    )

    selected_design = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "runtime_executed_in_stage4a65au": False,
        "future_runtime_runner_created": False,
        "future_stage_label": "Stage 4A-6.5au bounded runtime smoke command sketch",
        "output_dir": str(output_dir),
        "scene_variant": EXPECTED_SCENE_VARIANT,
        "scene_seed": EXPECTED_SCENE_SEED,
        "start_variant": EXPECTED_START_VARIANT,
        "position": EXPECTED_START_POSITION,
        "yaw_rad": EXPECTED_START_YAW,
        "yaw_deg": 155.00000000000003,
        "tree_seed": args.tree_seed,
        "primary_formula": PRIMARY_FORMULA,
        "shadow_formula": SHADOW_FORMULA,
        "measured_shadow_formula": MEASURED_FORMULA,
        "bounded_sequence": sequence_safety["future_runtime_sequence_contract"],
        "prediction_safety": prediction_safety["future_map_predict_policy"],
        "logging_requirements": {
            "per_frame_observed_state_delta": True,
            "per_frame_map_predict_timing_and_counts": True,
            "per_frame_tree_decision": True,
            "per_frame_sc_gain_components": True,
            "hash_prediction_layers_after_creation": True,
            "record_tree_profiles": True,
        },
        "acceptance_gates_before_formal_expert_sampling": {
            "start_pose_matches_metadata": pose_consistency["all_pose_checks_passed"],
            "exactly_two_frames_one_action": True,
            "frame3_not_captured": True,
            "map_predict_read_only": True,
            "lambda48_distinct_nonmeasured_clean_preferred": True,
            "low_cost_artifact_false": True,
            "historical_prior_basin_false": True,
            "no_rollout": True,
            "no_rl_gdpo_ppo_bc_il": True,
        },
        "not_expert_sampling_yet": True,
        "reason": (
            "start_corridor seed0/seed1 references are safety-clean; start_room_b is the selected next start to validate "
            "before any formal expert data sampling."
        ),
        "source_designs": {
            "stage4a65ar_selected_next_bounded_repeat_design": ar_design,
            "stage4a65ap_selected_alternate_start_design": ap_design,
            "stage4a65at_selected_next_start_design": at_design,
        },
    }
    write_json(output_dir / "selected_next_start_design.json", selected_design)
    write_text(
        output_dir / "selected_next_start_design.md",
        md_doc(
            "Selected Next Start Design",
            [
                "- Future bounded runtime target: `start_room_b`, tree_seed `0`.",
                f"- Position: `{EXPECTED_START_POSITION}`.",
                f"- Yaw: `{EXPECTED_START_YAW}` rad.",
                f"- Formula: `{PRIMARY_FORMULA}`.",
                "- The design is a template for validation before formal expert data sampling, not expert data collection itself.",
            ],
        ),
    )

    inventory = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "candidates": [
            {
                "name": "canonical_start",
                "position": CANONICAL_START,
                "status": "reference_context_only",
                "references": ["Stage 4A-6.5ak", "Stage 4A-6.5am", "Stage 4A-6.5ao"],
            },
            {
                "name": "start_corridor",
                "position": START_CORRIDOR_POSITION,
                "yaw_rad": START_CORRIDOR_YAW,
                "status": "already_checked_seed0_seed1_context",
                "references": ["Stage 4A-6.5aq", "Stage 4A-6.5as", "Stage 4A-6.5at"],
            },
            {
                "name": "start_room_b",
                "position": EXPECTED_START_POSITION,
                "yaw_rad": EXPECTED_START_YAW,
                "status": "selected_for_future_bounded_smoke",
                "metadata_path": str(args.start_room_b_metadata),
                "pose_checks_passed": pose_consistency["all_pose_checks_passed"],
            },
        ],
        "selected": "start_room_b",
        "selection_reason": selected_design["reason"],
    }
    write_json(output_dir / "next_start_candidate_inventory.json", inventory)
    write_text(
        output_dir / "next_start_candidate_inventory.md",
        md_doc(
            "Next Start Candidate Inventory",
            [
                "- `canonical_start`: reference context only.",
                "- `start_corridor`: already checked with seed0/seed1 bounded smokes.",
                "- `start_room_b`: selected next bounded smoke design.",
                f"- Pose checks passed: {md_bool(pose_consistency['all_pose_checks_passed'])}",
            ],
        ),
    )

    readiness_rows = [
        {"check": "offline_design_only", "passed": True, "recommendation": "keep"},
        {"check": "start_pose_metadata_match", "passed": pose_consistency["all_pose_checks_passed"], "recommendation": "required"},
        {"check": "scene_seed0_medium_three_rooms", "passed": pose_consistency["scene_variant_matches"] and pose_consistency["scene_seed_matches"], "recommendation": "required"},
        {"check": "tree_seed0_declared", "passed": args.tree_seed == 0, "recommendation": "required"},
        {"check": "bounded_two_frame_one_action", "passed": True, "recommendation": "required"},
        {"check": "frame3_not_captured", "passed": True, "recommendation": "required"},
        {"check": "lambda48_formula", "passed": PRIMARY_FORMULA == "gain_exp / cost + 48 * minmax(source_occ_free)", "recommendation": "required"},
        {"check": "lambda32_shadow", "passed": True, "recommendation": "required"},
        {"check": "measured_only_shadow", "passed": True, "recommendation": "required"},
        {"check": "map_predict_read_only", "passed": True, "recommendation": "required"},
        {"check": "no_motion_safety_prediction_use", "passed": True, "recommendation": "required"},
        {"check": "observed_delta_logging_contract", "passed": True, "recommendation": "required"},
        {"check": "map_predict_stability_logging_contract", "passed": True, "recommendation": "required"},
        {"check": "tree_sc_gain_logging_contract", "passed": True, "recommendation": "required"},
        {"check": "low_cost_artifact_false_design", "passed": low_cost["low_cost_artifact"] is False, "recommendation": "must_remain_false_runtime"},
        {"check": "historical_prior_basin_false_design", "passed": prior["historical_prior_basin"] is False, "recommendation": "must_remain_false_runtime"},
        {"check": "no_rollout_or_training", "passed": True, "recommendation": "required"},
        {"check": "hardware_threads_recorded", "passed": True, "recommendation": "required"},
        {"check": "future_command_marked_do_not_run", "passed": True, "recommendation": "required"},
        {"check": "formal_expert_sampling_ready_now", "passed": False, "recommendation": "blocked_until_runtime_smoke_passes"},
    ]
    write_csv(output_dir / "repeat_safety_readiness_matrix.csv", readiness_rows)
    write_json(
        output_dir / "repeat_safety_readiness_matrix.json",
        {
            "stage": "Stage 4A-6.5au",
            "design_template_ready": all(row["passed"] for row in readiness_rows if row["check"] != "formal_expert_sampling_ready_now"),
            "formal_expert_sampling_ready_now": False,
            "rows": readiness_rows,
        },
    )
    write_text(output_dir / "repeat_safety_readiness_matrix.md", "# Repeat Safety Readiness Matrix\n\n" + md_table(readiness_rows, ["check", "passed", "recommendation"]))

    future_command = "\n".join(
        [
            "DO NOT RUN IN STAGE 4A-6.5au.",
            "This is a future bounded-runtime command sketch only. It was not executed while generating this design package.",
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
            "# Future command sketch only; the runtime runner is intentionally not created by Stage 4A-6.5au design review.",
            "python run_stage4a65au_start_room_b_bounded_smoke.py \\",
            "  --scene_variant medium_three_rooms \\",
            "  --scene_seed 0 \\",
            "  --start_variant start_room_b \\",
            "  --position 2.75,-2.55,1.2 \\",
            "  --yaw 2.7052603405912112 \\",
            "  --tree_seed 0 \\",
            "  --max_frames 2 \\",
            "  --max_map_predict_calls 2 \\",
            "  --execute_exactly_one_action \\",
            "  --no_second_action \\",
            "  --no_third_frame \\",
            "  --no_rollout \\",
            "  --primary_formula 'gain_exp / cost + 48 * minmax(source_occ_free)' \\",
            "  --measured_only_shadow \\",
            "  --lambda32_shadow_formula 'gain_exp / cost + 32 * minmax(source_occ_free)' \\",
            "  --map_predict_read_only \\",
            "  --no_prediction_writeback \\",
            "  --no_prediction_traversability \\",
            "  --no_prediction_collision \\",
            "  --no_prediction_ray_blocking \\",
            "  --no_prediction_edge_validity \\",
            "  --no_prediction_candidate_sampling \\",
            "  --log_observed_state_delta_per_frame \\",
            "  --log_map_predict_stability_per_frame \\",
            "  --log_tree_decision_per_frame \\",
            "  --log_sc_gain_components \\",
            "  --max_workers 32 \\",
            "  --parallel_backend process_pool \\",
            f"  --output_dir {output_dir} \\",
            "  --save_viz",
            "",
            "This sketch must not be executed in Stage 4A-6.5au.",
        ]
    )
    write_text(output_dir / "future_stage4a65au_command_sketch.md", future_command)
    write_text(
        output_dir / "do_not_run_runtime_in_stage4a65au.md",
        md_doc(
            "Do Not Run Runtime In Stage 4A-6.5au",
            [
                "- Stage 4A-6.5au is offline design/review only.",
                "- No Isaac startup, capture, map_predict, SSCNet inference, action, rollout, RL, GDPO, PPO, BC, or IL was run.",
                "- The future command sketch is a template and is explicitly marked DO NOT RUN.",
                "- Formal expert data sampling remains blocked until this bounded smoke is actually executed and reviewed in a later runtime stage.",
            ],
        ),
    )

    timing_rows = runtime_timing_reference_rows([aq, ass])
    hardware_report = build_hardware_report(args, hash_time_s, time.perf_counter() - start_time, timing_rows)
    write_json(output_dir / "hardware_utilization_report.json", hardware_report)
    write_text(
        output_dir / "hardware_utilization_report.md",
        md_doc(
            "Hardware Utilization Report",
            [
                f"- CPU count: `{hardware_report['os_cpu_count']}`.",
                f"- Design-stage process pool workers for hash audit: `{hardware_report['actual_max_workers']}`.",
                "- OMP/BLAS/MKL/NUMEXPR/VECLIB threads are set to `1`.",
                f"- GPU query available: {md_bool(hardware_report['gpu']['nvidia_smi_available'])}",
                f"- Future runtime GPU target: `{hardware_report['future_runtime_hardware_policy']['gpu_target']}`.",
                "- No Stage 4A-6.5au GPU runtime work was executed.",
            ],
        ),
    )

    write_json(output_dir / "missing_fields_report.json", {"stage": "Stage 4A-6.5au", "missing": missing})
    write_text(
        output_dir / "missing_fields_report.md",
        "# Missing Fields Report\n\n"
        + md_table(missing, ["role", "essential", "path"])
        if missing
        else "# Missing Fields Report\n\n- No required missing fields blocked the design package.",
    )

    if args.save_viz:
        plot_layout(output_dir / "start_room_b_design_topdown.png", start_room_b_metadata, start_room_b, selected_design)
        plot_reference_tree(
            output_dir / "reference_seed01_frame1_tree_topdown.png",
            "start_corridor seed0/seed1 Frame1 lambda48 reference",
            "frame001",
            reference_stages,
            map_bounds,
        )
        plot_reference_tree(
            output_dir / "reference_seed01_frame2_tree_topdown.png",
            "start_corridor seed0/seed1 Frame2 lambda48 reference",
            "frame002",
            reference_stages,
            map_bounds,
        )
        plot_reference_tree(
            output_dir / "canonical_seed012_reference_tree_topdown.png",
            "canonical start seed0/1/2 Frame1 lambda48 reference",
            "frame001",
            canonical_stages,
            map_bounds,
        )
        save_bar(
            output_dir / "observed_state_reference_delta_bar.png",
            [row["stage"] for row in observed_rows],
            [float(row.get("observed_ratio_delta") or 0.0) for row in observed_rows],
            "Reference observed-state ratio delta",
            "observed ratio delta",
            rotate=35,
        )
        save_bar(
            output_dir / "map_predict_reference_stability_bar.png",
            [f"{row['stage']} {row['frame'][-3:]}" for row in map_rows],
            [float(row.get("predicted_unmeasured_occ_free") or 0.0) for row in map_rows],
            "Reference predicted unmeasured OCC+FREE",
            "count",
            rotate=45,
        )
        plot_gain_rows(
            output_dir / "tree_sc_gain_reference_frame1.png",
            value_rows_for_frame(reference_stages, "frame001"),
            "Frame1 reference gain and SC components",
        )
        plot_gain_rows(
            output_dir / "tree_sc_gain_reference_frame2.png",
            value_rows_for_frame(reference_stages, "frame002"),
            "Frame2 reference gain and SC components",
        )
        plot_readiness(output_dir / "repeat_safety_readiness_matrix.png", readiness_rows)
        plot_timeline(output_dir / "future_two_frame_sequence_timeline.png")

    forbidden = scan_forbidden_artifacts(output_dir)
    write_json(output_dir / "forbidden_artifact_scan.json", forbidden)
    write_text(
        output_dir / "forbidden_artifact_scan.md",
        md_doc(
            "Forbidden Artifact Scan",
            [
                f"- Clean: {md_bool(forbidden['clean'])}",
                "- Runtime artifacts such as NPY/NPZ frames, rollout files, replay buffers, and policy checkpoints are prohibited in this design output.",
            ],
        ),
    )

    required_status = {
        "required_outputs": {name: (output_dir / name).is_file() for name in REQUIRED_OUTPUTS},
        "required_plots": {name: (output_dir / name).is_file() for name in REQUIRED_PLOTS},
    }
    summary = {
        "stage": "Stage 4A-6.5au",
        "design_review_only": True,
        "runtime_executed": False,
        "inputs_loaded": {
            "stage4a65aq": input_manifest["loaded_stage4a65aq"],
            "stage4a65as": input_manifest["loaded_stage4a65as"],
            "stage4a65ar": input_manifest["loaded_stage4a65ar"],
            "stage4a65ap": input_manifest["loaded_stage4a65ap"],
            "stage4a65at": input_manifest["loaded_stage4a65at"],
            "start_room_b_metadata": input_manifest["loaded_start_room_b_metadata"],
        },
        "context_manifest": context_manifest,
        "pose_consistency": pose_consistency,
        "sequence_safety": sequence_safety,
        "prediction_safety": prediction_safety,
        "observed_state_delta_summary": observed_summary,
        "map_predict_two_frame_stability": map_summary,
        "frame1_tree_decision_diagnosis": frame1_diag,
        "frame2_tree_decision_diagnosis": frame2_diag,
        "low_cost_artifact_review": low_cost,
        "historical_prior_basin_review": prior,
        "readiness": {
            "design_template_ready": all(row["passed"] for row in readiness_rows if row["check"] != "formal_expert_sampling_ready_now"),
            "formal_expert_sampling_ready_now": False,
            "rows": readiness_rows,
        },
        "selected_next_start_design": selected_design,
        "hardware_report": hardware_report,
        "forbidden_artifact_scan": forbidden,
        "required_status": required_status,
        "future_command_marked_do_not_run": future_command.startswith("DO NOT RUN IN STAGE 4A-6.5au."),
        "future_command_executed": False,
        "final_wall_time_s": time.perf_counter() - start_time,
    }
    write_json(output_dir / "stage4a65au_start_room_b_design_summary.json", summary)
    summary_lines = [
        f"1. Design/review only? `{summary['design_review_only']}`.",
        "2. Stage 4A-6.5au did not start Isaac, capture RGB/depth, call map_predict, execute action, or rollout.",
        f"3. Loaded Stage 4A-6.5aq/as/ar/ap/at context? `{all(summary['inputs_loaded'][k] for k in ['stage4a65aq', 'stage4a65as', 'stage4a65ar', 'stage4a65ap', 'stage4a65at'])}`.",
        f"4. Loaded start_room_b metadata? `{summary['inputs_loaded']['start_room_b_metadata']}`.",
        f"5. start_room_b pose/yaw matches metadata and requested values? `{pose_consistency['all_pose_checks_passed']}`.",
        f"6. Future scene/tree seed: `{EXPECTED_SCENE_VARIANT}`, scene_seed `{EXPECTED_SCENE_SEED}`, tree_seed `{args.tree_seed}`.",
        f"7. Future formula: `{PRIMARY_FORMULA}`.",
        "8. Measured-only shadow and lambda32 shadow are required.",
        "9. map_predict is read-only and does not affect observed_state, traversability, collision, ray blocking, edge validity, or candidate sampling.",
        "10. Future sequence is bounded Frame1 capture/action, Frame2 capture/diagnostic, Frame3 not captured.",
        f"11. Offline low-cost artifact flag: `{low_cost['low_cost_artifact']}`; future runtime must keep it false.",
        f"12. Offline historical prior basin flag: `{prior['historical_prior_basin']}`; future runtime must keep it false.",
        "13. Per-frame observed_state delta, map_predict stability, tree decision, and SC gain logs are specified.",
        "14. CPU/GPU/thread policy is recorded; OMP/BLAS/MKL/NUMEXPR/VECLIB threads are 1.",
        f"15. Future command sketch marked DO NOT RUN in Stage 4A-6.5au? `{summary['future_command_marked_do_not_run']}`.",
        f"16. Formal expert sampling ready now? `{summary['readiness']['formal_expert_sampling_ready_now']}`.",
    ]
    write_text(output_dir / "stage4a65au_start_room_b_design_summary.md", "# Stage 4A-6.5au Start Room B Design Summary\n\n" + "\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
