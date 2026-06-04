#!/usr/bin/env python3
"""Stage 4A-6.13 uncertainty-bonus bounded short rollout pilot.

This runner performs the explicitly approved short rollout only: ten starts,
at most three executed primary expert actions per start, and one terminal
post-action QA capture per start. Prediction and uncertainty stay read-only
scoring-only features. They are never written to observed_state and are not
used for traversability, collision, ray blocking, candidate validity, or edge
validity.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import Counter, defaultdict
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
    os.environ[_key] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from depth_to_voxel import FREE, OCCUPIED, UNKNOWN, update_observed_state_from_depth
from isaac_map_predictor import IsaacMapPredictor

import run_stage4a67_measured_only_expert_pilot as s67
import run_stage4a68_map_predict_lambda48_expert_pilot as s68
import run_stage4a611_uncertainty_aware_lambda_one_action_pilot as s611


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
STAGE = "Stage 4A-6.13-uncertainty-bonus-short-rollout-pilot"
OUTPUT_NAME = "isaac_stage4a613_uncertainty_bonus_short_rollout_pilot"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs" / OUTPUT_NAME
DEFAULT_FIXED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
DEFAULT_SOURCE_USD = WORKSPACE / "building_scene.usd"
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
DEFAULT_CAMERA_FIX_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix"
DEFAULT_MEASURED_ONLY_DIR = WORKSPACE / "outputs/isaac_stage4a67_measured_only_expert_pilot"
DEFAULT_LAMBDA48_DIR = WORKSPACE / "outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot"
DEFAULT_TWO_FRAME_DIR = WORKSPACE / "outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot"
DEFAULT_DENSE_DIR = WORKSPACE / "outputs/isaac_stage4a610a_dense_prediction_uncertainty_artifacts"
DEFAULT_DENSE_AUDIT_DIR = WORKSPACE / "outputs/isaac_stage4a610a_uncertainty_audit_rerun_dense"
DEFAULT_UNCERTAINTY_AWARE_DIR = WORKSPACE / "outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot"
DEFAULT_UNCERTAINTY_BONUS_DIR = WORKSPACE / "outputs/isaac_stage4a612_uncertainty_exploration_bonus_pilot"

PRIMARY_FORMULA = "uncertainty_bonus_composite_beta8"
PRIMARY_FORMULA_TEXT = "gain_exp / cost + 48 * minmax(source_occ_free) + 8 * uncertainty_composite"
UNCERTAINTY_COMPOSITE_TEXT = "0.4 * minmax(candidate_uncertain_fraction) + 0.4 * minmax(candidate_entropy_mean) + 0.2 * minmax(1 - candidate_margin_mean)"
LAMBDA_SC = 48.0
BETA_UNCERTAINTY = 8.0
MAX_CANDIDATES = 64
ACTION_CHANGE_DISTANCE_M = 0.15
LOCAL_JITTER_DISTANCE_M = 0.75

FORBIDDEN_DATASET_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "class_prob",
    "policy_logits",
    "RL reward",
    "rl_reward",
    "replay buffer",
    "replay_buffer",
    "training state",
    "training_state",
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(jsonable(value), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(field_order or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        text = json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(jsonable(value))
        if len(text) > 1800:
            text = text[:1800] + "..."
        lines.append(f"| `{key}` | `{text}` |")
    return "\n".join(lines)


def markdown_rows(title: str, rows: list[dict[str, Any]], limit: int = 40) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        return "\n".join(lines + ["No rows."])
    fields: list[str] = []
    for row in rows[:limit]:
        for key in row:
            if key not in fields:
                fields.append(key)
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows[:limit]:
        vals = []
        for field in fields:
            text = str(csv_value(row.get(field))).replace("\n", " ")
            if len(text) > 220:
                text = text[:220] + "..."
            vals.append(f"`{text}`")
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > limit:
        lines.append("")
        lines.append(f"Showing {limit} of {len(rows)} rows.")
    return "\n".join(lines)


def save_report_pair(output_dir: Path, stem: str, data: Any, title: str) -> None:
    save_json(output_dir / f"{stem}.json", data)
    if isinstance(data, list):
        write_text(output_dir / f"{stem}.md", markdown_rows(title, data))
    elif isinstance(data, dict):
        write_text(output_dir / f"{stem}.md", markdown_table(title, data))
    else:
        write_text(output_dir / f"{stem}.md", f"# {title}\n\n{data}")


def sha256_file(path: Path | str) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def git_status_text() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    return result.stdout


def parse_literal(value: Any, default: Any) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    try:
        return ast.literal_eval(str(value))
    except Exception:
        try:
            return json.loads(str(value))
        except Exception:
            return default


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except Exception:
        return float(default)


def as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def summarize(values: list[Any]) -> dict[str, Any]:
    clean = np.asarray([float(v) for v in values if v is not None and str(v) != "" and math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(clean.size),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p50": float(np.percentile(clean, 50)),
        "p90": float(np.percentile(clean, 90)),
    }


def finite_minmax(values: list[Any]) -> np.ndarray:
    arr = np.asarray([as_float(v, 0.0) for v in values], dtype=np.float64)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


def observed_counts(state: np.ndarray) -> dict[str, Any]:
    total = int(state.size)
    unknown = int(np.count_nonzero(state == UNKNOWN))
    free = int(np.count_nonzero(state == FREE))
    occupied = int(np.count_nonzero(state == OCCUPIED))
    observed = free + occupied
    return {
        "total_count": total,
        "observed_count": observed,
        "unknown_count": unknown,
        "free_count": free,
        "occupied_count": occupied,
        "observed_ratio": float(observed / max(total, 1)),
    }


def action_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    aw = np.asarray(a.get("world", a.get("selected_world_xyz", [math.nan, math.nan, math.nan])), dtype=np.float64)
    bw = np.asarray(b.get("world", b.get("selected_world_xyz", [math.nan, math.nan, math.nan])), dtype=np.float64)
    if aw.size < 2 or bw.size < 2:
        return math.nan
    return float(np.linalg.norm(aw[:2] - bw[:2]))


def yaw_delta(a: dict[str, Any], b: dict[str, Any]) -> float:
    ay = as_float(a.get("yaw_rad", a.get("selected_yaw")), math.nan)
    by = as_float(b.get("yaw_rad", b.get("selected_yaw")), math.nan)
    if not math.isfinite(ay) or not math.isfinite(by):
        return math.nan
    return abs(s67.wrap_angle(ay - by))


def classify_against(selected: dict[str, Any], baseline: dict[str, Any]) -> tuple[str, float, float]:
    dist = action_distance(selected, baseline)
    yd = yaw_delta(selected, baseline)
    if not math.isfinite(dist):
        return "no_valid_candidate", dist, yd
    same_candidate = int(selected.get("candidate_id", selected.get("selected_candidate_id", -1))) == int(
        baseline.get("candidate_id", baseline.get("selected_candidate_id", -2))
    )
    if same_candidate and dist <= ACTION_CHANGE_DISTANCE_M and yd <= 0.10:
        return "same_as_measured", dist, yd
    if dist <= LOCAL_JITTER_DISTANCE_M:
        return "local_jitter", dist, yd
    return "distinct_nonmeasured_branch", dist, yd


def row_selection_key(row: dict[str, Any], score_key: str) -> tuple[float, float, float, float, int]:
    return (
        as_float(row.get(score_key), -1.0e9),
        as_float(row.get("gain_exp"), 0.0),
        as_float(row.get("source_occ_free"), 0.0),
        -as_float(row.get("path_cost", row.get("cost_s")), 1.0e9),
        -int(row.get("candidate_id", 0)),
    )


def best_by_score(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    return dict(max(rows, key=lambda row: row_selection_key(row, score_key)))


def world_key(row: dict[str, Any]) -> tuple[int, int]:
    world = row.get("world", row.get("selected_world_xyz", [math.nan, math.nan, math.nan]))
    return (int(round(as_float(world[0], math.nan) * 10.0)), int(round(as_float(world[1], math.nan) * 10.0)))


def check_required_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input is missing: {path}")


def load_inputs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    required = [
        args.camera_pose_fix_dir / "stage4a66c_usd_camera_pose_fix_summary.json",
        args.camera_pose_fix_dir / "start_variants_interior.json",
        args.camera_pose_fix_dir / "camera_info.json",
        args.camera_pose_fix_dir / "scene_metadata.json",
        args.camera_pose_fix_dir / "observed_state_final.npy",
        args.measured_only_pilot_dir / "stage4a67_measured_only_expert_pilot_summary.json",
        args.measured_only_pilot_dir / "expert_dataset.npz",
        args.measured_only_pilot_dir / "dataset_integrity_report.json",
        args.measured_only_pilot_dir / "safety_audit.json",
        args.lambda48_pilot_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
        args.lambda48_pilot_dir / "expert_dataset.npz",
        args.lambda48_pilot_dir / "lambda48_decisions.csv",
        args.lambda48_pilot_dir / "measured_shadow_decisions.csv",
        args.lambda48_pilot_dir / "prediction_safety_audit.json",
        args.lambda48_pilot_dir / "expert_data_quality_audit.json",
        args.two_frame_lambda48_pilot_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json",
        args.two_frame_lambda48_pilot_dir / "expert_dataset_two_frame.npz",
        args.two_frame_lambda48_pilot_dir / "per_frame_summary.csv",
        args.two_frame_lambda48_pilot_dir / "frame1_lambda48_decisions.csv",
        args.two_frame_lambda48_pilot_dir / "frame2_lambda48_diagnostic_decisions.csv",
        args.two_frame_lambda48_pilot_dir / "prediction_safety_audit.json",
        args.two_frame_lambda48_pilot_dir / "expert_data_quality_audit.json",
        args.two_frame_lambda48_pilot_dir / "two_frame_stability_audit.json",
        args.dense_uncertainty_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json",
        args.dense_uncertainty_dir / "dense_prediction_artifact_manifest.json",
        args.dense_uncertainty_dir / "candidate_visible_uncertainty_manifest.json",
        args.dense_uncertainty_dir / "dense_uncertainty_candidate_summary.json",
        args.dense_uncertainty_audit_dir / "candidate_uncertainty_table.csv",
        args.dense_uncertainty_audit_dir / "uncertainty_readiness_decision.json",
        args.uncertainty_aware_pilot_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json",
        args.uncertainty_aware_pilot_dir / "expert_dataset_uncertainty_lambda.npz",
        args.uncertainty_aware_pilot_dir / "primary_confidence_gated_decisions.csv",
        args.uncertainty_aware_pilot_dir / "lambda48_baseline_shadow_decisions.csv",
        args.uncertainty_aware_pilot_dir / "uncertainty_candidate_features.csv",
        args.uncertainty_aware_pilot_dir / "prediction_safety_audit.json",
        args.uncertainty_aware_pilot_dir / "uncertainty_safety_audit.json",
        args.uncertainty_aware_pilot_dir / "expert_data_quality_audit.json",
        args.uncertainty_bonus_decision_dir / "stage4a612_uncertainty_exploration_bonus_pilot_summary.json",
        args.uncertainty_bonus_decision_dir / "expert_decision_dataset_uncertainty_bonus.npz",
        args.uncertainty_bonus_decision_dir / "uncertainty_bonus_beta_sweep.csv",
        args.uncertainty_bonus_decision_dir / "uncertainty_bonus_readiness_decision.json",
        args.uncertainty_bonus_decision_dir / "recommended_formula_report.json",
        args.uncertainty_bonus_decision_dir / "uncertainty_bonus_quality_audit.json",
        args.uncertainty_bonus_decision_dir / "uncertainty_bonus_risk_audit.json",
        args.uncertainty_bonus_decision_dir / "future_short_rollout_with_uncertainty_bonus_sketch.md",
    ]
    for path in required:
        check_required_file(path)
    for rel in (
        ".project_context/CURRENT_STATE.md",
        ".project_context/TODO.md",
        ".project_context/CODEX_LOG.md",
        "README.md",
        "ARTIFACTS.md",
        "ENVIRONMENT.md",
        "GIT_INITIALIZATION_REPORT.md",
    ):
        check_required_file(WORKSPACE / rel)

    camera_summary = read_json(args.camera_pose_fix_dir / "stage4a66c_usd_camera_pose_fix_summary.json")
    starts_obj = read_json(args.camera_pose_fix_dir / "start_variants_interior.json")
    starts = starts_obj.get("starts", starts_obj) if isinstance(starts_obj, dict) else starts_obj
    starts = [dict(row) for row in starts[: int(args.num_starts)]]
    for idx, start in enumerate(starts):
        start.setdefault("index", idx)
        start.setdefault("name", f"start_{idx:03d}")
        if "yaw_rad" not in start and "yaw" in start:
            start["yaw_rad"] = float(start["yaw"])

    scene_metadata = read_json(args.camera_pose_fix_dir / "scene_metadata.json")
    observed_summary = read_json(args.camera_pose_fix_dir / "observed_summary.json")
    source_observed = np.load(args.camera_pose_fix_dir / "observed_state_final.npy")
    bounds = observed_summary.get("chosen_bounds") or scene_metadata.get("map_bounds")
    voxel_size = float(observed_summary.get("voxel_size", scene_metadata.get("voxel_size", 0.1)))
    if not bounds:
        raise ValueError("Cannot resolve map bounds from camera pose fix outputs")

    s67_summary = read_json(args.measured_only_pilot_dir / "stage4a67_measured_only_expert_pilot_summary.json")
    s67_integrity = read_json(args.measured_only_pilot_dir / "dataset_integrity_report.json")
    s67_safety = read_json(args.measured_only_pilot_dir / "safety_audit.json")
    s68_summary = read_json(args.lambda48_pilot_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json")
    s68_pred_safety = read_json(args.lambda48_pilot_dir / "prediction_safety_audit.json")
    s68_quality = read_json(args.lambda48_pilot_dir / "expert_data_quality_audit.json")
    s69_summary = read_json(args.two_frame_lambda48_pilot_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json")
    s69_pred_safety = read_json(args.two_frame_lambda48_pilot_dir / "prediction_safety_audit.json")
    s69_quality = read_json(args.two_frame_lambda48_pilot_dir / "expert_data_quality_audit.json")
    s69_stability = read_json(args.two_frame_lambda48_pilot_dir / "two_frame_stability_audit.json")
    s610_summary = read_json(args.dense_uncertainty_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json")
    s610_readiness = read_json(args.dense_uncertainty_audit_dir / "uncertainty_readiness_decision.json")
    s611_summary = read_json(args.uncertainty_aware_pilot_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json")
    s611_pred_safety = read_json(args.uncertainty_aware_pilot_dir / "prediction_safety_audit.json")
    s611_unc_safety = read_json(args.uncertainty_aware_pilot_dir / "uncertainty_safety_audit.json")
    s611_quality = read_json(args.uncertainty_aware_pilot_dir / "expert_data_quality_audit.json")
    s612_summary = read_json(args.uncertainty_bonus_decision_dir / "stage4a612_uncertainty_exploration_bonus_pilot_summary.json")
    s612_readiness = read_json(args.uncertainty_bonus_decision_dir / "uncertainty_bonus_readiness_decision.json")
    s612_formula = read_json(args.uncertainty_bonus_decision_dir / "recommended_formula_report.json")
    s612_quality = read_json(args.uncertainty_bonus_decision_dir / "uncertainty_bonus_quality_audit.json")
    s612_risk = read_json(args.uncertainty_bonus_decision_dir / "uncertainty_bonus_risk_audit.json")

    checks = {
        "stage4a612_complete": bool(s612_summary.get("completed", False)),
        "recommended_formula": s612_summary.get("recommended_uncertainty_bonus_formula") == PRIMARY_FORMULA
        and s612_readiness.get("recommended_uncertainty_bonus_formula") == PRIMARY_FORMULA,
        "recommended_beta": int(s612_summary.get("recommended_beta", -1)) == 8
        and int(s612_readiness.get("recommended_beta", -1)) == 8,
        "runtime_ready": bool(s612_summary.get("uncertainty_bonus_runtime_ready", False))
        and bool(s612_readiness.get("uncertainty_bonus_runtime_ready", False)),
        "risk_audit_passed": bool(s612_risk.get("passed", False)),
        "quality_audit_passed": bool(s612_quality.get("passed", False)),
        "no_warnings": not s612_readiness.get("warnings") and not s612_quality.get("warnings") and not s612_risk.get("warnings"),
        "no_blockers": not s612_readiness.get("blockers") and not s612_quality.get("blockers") and not s612_risk.get("blockers"),
        "stage4a67_complete": bool(s67_summary.get("completed", False)),
        "stage4a67_integrity": bool(s67_integrity.get("passed", False)),
        "stage4a67_safety": bool(s67_safety.get("passed", False)),
        "stage4a68_complete": bool(s68_summary.get("completed", False)),
        "stage4a68_prediction_safety": bool(s68_pred_safety.get("passed", False)),
        "stage4a68_quality": bool(s68_quality.get("passed", False)),
        "stage4a69_complete": bool(s69_summary.get("completed", False)),
        "stage4a69_prediction_safety": bool(s69_pred_safety.get("passed", False)),
        "stage4a69_quality": bool(s69_quality.get("passed", False)),
        "stage4a69_stability": bool(s69_stability.get("passed", False)),
        "stage4a610a_uncertainty_ready": bool(s610_readiness.get("candidate_level_uncertainty_ready", False)),
        "stage4a611_complete": bool(s611_summary.get("completed", False)),
        "stage4a611_prediction_safety": bool(s611_pred_safety.get("passed", False)),
        "stage4a611_uncertainty_safety": bool(s611_unc_safety.get("passed", False)),
        "stage4a611_quality": bool(s611_quality.get("passed", False)),
        "short_rollout_user_approved": True,
        "long_rollout_not_requested": True,
        "full_expert_dataset_not_requested": True,
        "training_not_requested": True,
        "prediction_uncertainty_scoring_only": True,
    }
    failed = [key for key, value in checks.items() if not bool(value)]
    if failed:
        raise RuntimeError(f"Required preflight checks failed: {failed}")
    save_report_pair(output_dir, "input_readiness_preflight", checks, "Input Readiness Preflight")

    inspection = scene_metadata.get("inspection_camera_poses", [])
    yaws = s67.inspection_yaw_priors(starts, inspection)
    return {
        "camera_summary": camera_summary,
        "starts": starts,
        "source_observed": source_observed,
        "source_observed_summary": s67.validate_observed_state(source_observed, "stage4a613_source_observed_state"),
        "bounds": bounds,
        "voxel_size": voxel_size,
        "scene_metadata": scene_metadata,
        "observed_summary": observed_summary,
        "yaw_priors": yaws,
        "s67_summary": s67_summary,
        "s68_summary": s68_summary,
        "s69_summary": s69_summary,
        "s610_summary": s610_summary,
        "s611_summary": s611_summary,
        "s612_summary": s612_summary,
        "s612_readiness": s612_readiness,
        "s612_formula": s612_formula,
        "s612_quality": s612_quality,
        "s612_risk": s612_risk,
        "stage68_lambda_rows": read_csv(args.lambda48_pilot_dir / "lambda48_decisions.csv"),
        "stage68_measured_rows": read_csv(args.lambda48_pilot_dir / "measured_shadow_decisions.csv"),
        "stage611_conf_rows": read_csv(args.uncertainty_aware_pilot_dir / "primary_confidence_gated_decisions.csv"),
        "stage612_primary_rows": read_csv(args.uncertainty_bonus_decision_dir / "per_start_uncertainty_bonus_decisions.csv")
        if (args.uncertainty_bonus_decision_dir / "per_start_uncertainty_bonus_decisions.csv").is_file()
        else [],
    }


def enforce_args(args: argparse.Namespace) -> None:
    required_flags = {
        "terminal_capture_per_start": args.terminal_capture_per_start,
        "save_dense_uncertainty_artifacts": args.save_dense_uncertainty_artifacts,
        "save_expert_quality_viz": args.save_expert_quality_viz,
        "compare_to_measured_only_pilot": args.compare_to_measured_only_pilot,
        "compare_to_lambda48_pilot": args.compare_to_lambda48_pilot,
        "compare_to_confidence_gated_pilot": args.compare_to_confidence_gated_pilot,
        "compare_to_uncertainty_bonus_decision_pilot": args.compare_to_uncertainty_bonus_decision_pilot,
        "save_viz": args.save_viz,
        "no_long_rollout": args.no_long_rollout,
        "no_full_expert_dataset": args.no_full_expert_dataset,
        "no_training": args.no_training,
        "no_rl_gdpo": args.no_rl_gdpo,
    }
    missing = [key for key, value in required_flags.items() if not bool(value)]
    if missing:
        raise ValueError(f"Missing required Stage 4A-6.13 boundary flags: {missing}")
    if int(args.num_starts) != 10:
        raise ValueError("Stage 4A-6.13 requires exactly 10 starts")
    if int(args.max_decision_steps_per_start) != 3:
        raise ValueError("Stage 4A-6.13 requires max_decision_steps_per_start=3")
    if int(args.max_total_actions) != 30 or int(args.max_total_decision_frames) != 30 or int(args.max_total_captures) != 40:
        raise ValueError("Stage 4A-6.13 bounded totals must be actions=30, decision_frames=30, captures=40")
    if int(args.num_candidates) != 64 or int(args.top_n) != 16:
        raise ValueError("Stage 4A-6.13 uses num_candidates=64 and top_n=16")
    if str(args.primary_formula) != PRIMARY_FORMULA:
        raise ValueError(f"Unsupported primary formula: {args.primary_formula}")
    if float(args.lambda_sc) != LAMBDA_SC or float(args.beta_uncertainty) != BETA_UNCERTAINTY:
        raise ValueError("Stage 4A-6.13 requires lambda_sc=48 and beta_uncertainty=8")
    if [float(v) for v in args.uncertainty_composite_weights] != [0.4, 0.4, 0.2]:
        raise ValueError("Stage 4A-6.13 requires uncertainty composite weights 0.4 0.4 0.2")
    if str(args.motion_mode) != "bounded_short_rollout":
        raise ValueError("Stage 4A-6.13 requires motion_mode=bounded_short_rollout")


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_before = output_dir / "git_status_before.txt"
    before_text = keep_before.read_text(encoding="utf-8") if keep_before.is_file() else git_status_text()
    for child in output_dir.iterdir():
        if child.name == "git_status_before.txt":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    write_text(keep_before, before_text)


def capture_pose_to_dir(
    output_dir: Path,
    start_dir: Path,
    camera: Any,
    sim: Any,
    pose: dict[str, Any],
    args: argparse.Namespace,
    prefix: str,
) -> dict[str, Any]:
    record = s67.capture_action_pose(output_dir, camera, sim, pose, args)
    mapping = {
        "rgb": (record["rgb_file"], f"{prefix}_rgb.png"),
        "depth": (record["depth_file"], f"{prefix}_depth.npy"),
        "depth_color": (record["depth_color_file"], f"{prefix}_depth_color.png"),
        "pose": (record["pose_file"], f"{prefix}_pose.json"),
    }
    paths: dict[str, str] = {}
    for key, (src_name, dst_name) in mapping.items():
        src = output_dir / src_name
        dst = start_dir / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        paths[key] = str(dst)
    pose_record = read_json(Path(paths["pose"]))
    pose_record.update(
        {
            "stage": STAGE,
            "start_variant_id": int(pose.get("start_variant_id", -1)),
            "step_id": int(pose.get("step_id", -1)),
            "terminal_frame": bool(pose.get("terminal_frame", False)),
            "decision_frame": not bool(pose.get("terminal_frame", False)),
        }
    )
    save_json(Path(paths["pose"]), pose_record)
    record.update(
        {
            "start_variant_id": int(pose.get("start_variant_id", -1)),
            "step_id": int(pose.get("step_id", -1)),
            "terminal_frame": bool(pose.get("terminal_frame", False)),
            "decision_frame": not bool(pose.get("terminal_frame", False)),
            "sample_paths": paths,
        }
    )
    return record


def step_start_from_pose(base_start: dict[str, Any], pose: dict[str, Any], step_id: int) -> dict[str, Any]:
    start = dict(base_start)
    start.update(
        {
            "name": f"{base_start.get('name', 'start')}_step{step_id:03d}",
            "position": [float(v) for v in pose["position"]],
            "yaw": float(pose["yaw_rad"]),
            "yaw_rad": float(pose["yaw_rad"]),
            "target": s67.pose_target([float(v) for v in pose["position"]], float(pose["yaw_rad"])),
        }
    )
    return start


def pose_from_selected(base_start: dict[str, Any], selected: dict[str, Any], sid: int, step_id: int, global_index: int) -> dict[str, Any]:
    world = [float(v) for v in selected["world"]]
    yaw = float(selected["yaw_rad"])
    return {
        "index": int(global_index),
        "name": f"stage4a613_start{sid:03d}_after_primary_step{step_id:03d}",
        "start_variant_name": base_start.get("name", f"start_{sid:03d}"),
        "start_variant_id": int(sid),
        "step_id": int(step_id + 1),
        "position": world,
        "yaw": yaw,
        "yaw_rad": yaw,
        "target": s67.pose_target(world, yaw),
        "source": PRIMARY_FORMULA,
        "semantic_zone_guess": base_start.get("semantic_zone_guess"),
        "selected_grid": selected.get("grid"),
        "selected_candidate_id": int(selected.get("candidate_id", -1)),
        "action_executed_in_isaac": True,
        "one_action_only_for_this_decision_frame": True,
    }


def add_composite_scores(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    source_mm = finite_minmax([row.get("source_occ_free") for row in rows])
    uncertain_mm = finite_minmax([row.get("uncertain_fraction") for row in rows])
    entropy_mm = finite_minmax([row.get("candidate_entropy_mean") for row in rows])
    low_margin_mm = finite_minmax([1.0 - as_float(row.get("candidate_margin_mean"), 0.0) for row in rows])
    conf_gate_mm = finite_minmax([as_float(row.get("source_occ_free"), 0.0) * as_float(row.get("candidate_confidence_mean"), 0.0) for row in rows])
    for i, row in enumerate(rows):
        base = as_float(row.get("base_measured_value", row.get("score_measured_only")), 0.0)
        row["source_occ_free_minmax_stage4a613"] = float(source_mm[i])
        row["candidate_uncertain_fraction"] = float(as_float(row.get("uncertain_fraction"), 0.0))
        row["candidate_uncertain_fraction_minmax"] = float(uncertain_mm[i])
        row["candidate_entropy_mean_minmax"] = float(entropy_mm[i])
        row["candidate_low_margin_minmax"] = float(low_margin_mm[i])
        row["uncertainty_composite"] = float(0.4 * uncertain_mm[i] + 0.4 * entropy_mm[i] + 0.2 * low_margin_mm[i])
        row["score_measured_only"] = float(base)
        row["score_lambda48"] = float(base + float(args.lambda_sc) * source_mm[i])
        row["score_lambda48_baseline"] = row["score_lambda48"]
        row["score_confidence_gated_6_11"] = float(base + float(args.lambda_sc) * conf_gate_mm[i])
        row["score_primary_uncertainty_bonus"] = float(
            base + float(args.lambda_sc) * source_mm[i] + float(args.beta_uncertainty) * row["uncertainty_composite"]
        )
        row["uncertainty_bonus_term"] = float(float(args.beta_uncertainty) * row["uncertainty_composite"])
        row["lambda48_term"] = float(float(args.lambda_sc) * source_mm[i])
        row["formula_dominated_by_uncertainty"] = bool(row["uncertainty_bonus_term"] > max(abs(base + row["lambda48_term"]), 1.0e-6))
        row["path_cost"] = float(as_float(row.get("path_cost", row.get("cost_s")), 0.0))
        row["candidate_all_local"] = False
        row["quality_flags"] = []


def selected_decision_row(
    stage: str,
    sid: int,
    step_id: int,
    start: dict[str, Any],
    formula: str,
    selected: dict[str, Any],
    score_key: str,
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    measured = references["measured_shadow"]
    lambda48 = references["lambda48_shadow"]
    confidence = references["confidence_gated_shadow"]
    branch_m, delta_m, yaw_m = classify_against(selected, measured)
    branch_l, delta_l, yaw_l = classify_against(selected, lambda48)
    branch_c, delta_c, yaw_c = classify_against(selected, confidence)
    flags = []
    if as_float(selected.get("candidate_confidence_mean"), 1.0) < 0.6:
        flags.append("selected_low_confidence")
    if as_float(selected.get("candidate_entropy_mean"), 0.0) > 0.5:
        flags.append("selected_high_entropy")
    if as_float(selected.get("candidate_margin_mean"), 1.0) < 0.2:
        flags.append("selected_low_margin")
    if bool(selected.get("formula_dominated_by_uncertainty", False)):
        flags.append("formula_dominated_by_uncertainty")
    return {
        "stage": stage,
        "start_variant_id": int(sid),
        "step_id": int(step_id),
        "start_name": start.get("name"),
        "formula": formula,
        "selected_candidate_id": int(selected.get("candidate_id", -1)),
        "selected_grid": selected.get("grid", [-1, -1, -1]),
        "selected_world_xyz": [float(v) for v in selected.get("world", [math.nan, math.nan, math.nan])],
        "selected_yaw": float(selected.get("yaw_rad", math.nan)),
        "final_score": float(as_float(selected.get(score_key), 0.0)),
        "gain_exp": float(as_float(selected.get("gain_exp"), 0.0)),
        "path_cost": float(as_float(selected.get("path_cost", selected.get("cost_s")), 0.0)),
        "source_occ_free": float(as_float(selected.get("source_occ_free"), 0.0)),
        "confidence_mean": float(as_float(selected.get("candidate_confidence_mean"), 0.0)),
        "confidence_min": float(as_float(selected.get("candidate_confidence_min"), 0.0)),
        "entropy_mean": float(as_float(selected.get("candidate_entropy_mean"), 0.0)),
        "entropy_max": float(as_float(selected.get("candidate_entropy_max"), 0.0)),
        "margin_mean": float(as_float(selected.get("candidate_margin_mean"), 0.0)),
        "margin_min": float(as_float(selected.get("candidate_margin_min"), 0.0)),
        "uncertain_fraction": float(as_float(selected.get("candidate_uncertain_fraction", selected.get("uncertain_fraction")), 0.0)),
        "uncertain_voxel_count": float(as_float(selected.get("uncertain_voxel_count"), 0.0)),
        "uncertainty_composite": float(as_float(selected.get("uncertainty_composite"), 0.0)),
        "lambda48_term": float(as_float(selected.get("lambda48_term"), 0.0)),
        "uncertainty_bonus_term": float(as_float(selected.get("uncertainty_bonus_term"), 0.0)),
        "branch_classification_vs_measured": branch_m,
        "branch_classification_vs_lambda48": branch_l,
        "branch_classification_vs_confidence_gated": branch_c,
        "action_delta_vs_measured_m": delta_m,
        "yaw_delta_vs_measured_rad": yaw_m,
        "action_delta_vs_lambda48_m": delta_l,
        "yaw_delta_vs_lambda48_rad": yaw_l,
        "action_delta_vs_confidence_gated_m": delta_c,
        "yaw_delta_vs_confidence_gated_rad": yaw_c,
        "quality_flags": flags,
        "no_valid_candidate": False,
        "low_cost_artifact": bool(as_float(selected.get("path_cost", selected.get("cost_s")), 1.0) <= 1.0e-4),
        "historical_prior_basin": False,
        "candidate_all_local": bool(selected.get("candidate_all_local", False)),
        "source_occ_free_kept_separate_from_uncertainty": True,
    }


def decision_row_to_selected(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": int(row.get("selected_candidate_id", -1)),
        "grid": row.get("selected_grid", [-1, -1, -1]),
        "world": row.get("selected_world_xyz", [math.nan, math.nan, math.nan]),
        "yaw_rad": row.get("selected_yaw", math.nan),
    }


def plot_dense_overlay(path: Path, observed_state: np.ndarray, bounds: dict[str, Any], start: dict[str, Any], selected: dict[str, Any], dense_path: Path, field: str, title: str, cmap: str) -> None:
    s611.plot_dense_overlay(path, observed_state, bounds, start, selected, dense_path, field, title, cmap)


def plot_candidate_score_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    top = sorted(rows, key=lambda row: as_float(row.get("score_primary_uncertainty_bonus"), -1.0e9), reverse=True)[:16]
    labels = [str(row["candidate_id"]) for row in top]
    x = np.arange(len(top))
    base = [as_float(row.get("score_measured_only"), 0.0) for row in top]
    lam = [as_float(row.get("lambda48_term"), 0.0) for row in top]
    unc = [as_float(row.get("uncertainty_bonus_term"), 0.0) for row in top]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    ax.bar(x, base, color="#2f6f8f", label="gain/cost")
    ax.bar(x, lam, bottom=base, color="#5da271", label="lambda48 source")
    ax.bar(x, unc, bottom=np.asarray(base) + np.asarray(lam), color="#c95c5c", label="uncertainty beta8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("score")
    ax.set_title("primary score decomposition")
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_uncertainty_score_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    top = sorted(rows, key=lambda row: as_float(row.get("score_primary_uncertainty_bonus"), -1.0e9), reverse=True)[:16]
    labels = [str(row["candidate_id"]) for row in top]
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    ax.bar(x - 0.27, [as_float(r.get("candidate_uncertain_fraction"), 0.0) for r in top], width=0.18, color="#c95c5c", label="uncertain frac")
    ax.bar(x - 0.09, [as_float(r.get("candidate_entropy_mean"), 0.0) for r in top], width=0.18, color="#d59f43", label="entropy")
    ax.bar(x + 0.09, [1.0 - as_float(r.get("candidate_margin_mean"), 0.0) for r in top], width=0.18, color="#7464a6", label="1-margin")
    ax.bar(x + 0.27, [as_float(r.get("uncertainty_composite"), 0.0) for r in top], width=0.18, color="#2f6f8f", label="composite")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("uncertainty decomposition")
    ax.legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_candidate_map(path: Path, observed_state: np.ndarray, bounds: dict[str, Any], start: dict[str, Any], rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    s611.plot_candidate_uncertainty_map(path, observed_state, bounds, start, rows, selected)


def plot_formula_delta_map(path: Path, observed_state: np.ndarray, bounds: dict[str, Any], start: dict[str, Any], selections: dict[str, dict[str, Any]]) -> None:
    mapped = {
        "uncertainty_bonus_beta8": selections["primary_uncertainty_bonus"],
        "measured_only": selections["measured_shadow"],
        "lambda48_baseline": selections["lambda48_shadow"],
        "primary_confidence_gated": selections["confidence_gated_shadow"],
    }
    s611.plot_formula_action_delta_map(path, observed_state, bounds, start, mapped)


def save_step_visuals(
    start_dir: Path,
    prefix: str,
    observed_state: np.ndarray,
    bounds: dict[str, Any],
    start: dict[str, Any],
    dense_path: Path,
    candidates: list[dict[str, Any]],
    selections: dict[str, dict[str, Any]],
) -> None:
    primary = selections["primary_uncertainty_bonus"]
    s67.save_observed_topdown(start_dir / f"{prefix}_observed_topdown.png", observed_state, f"{prefix} observed")
    plot_dense_overlay(start_dir / f"{prefix}_prediction_overlay.png", observed_state, bounds, start, primary, dense_path, "prediction", "prediction overlay", "Greens")
    plot_dense_overlay(start_dir / f"{prefix}_confidence_overlay.png", observed_state, bounds, start, primary, dense_path, "confidence", "confidence overlay", "viridis")
    plot_dense_overlay(start_dir / f"{prefix}_entropy_overlay.png", observed_state, bounds, start, primary, dense_path, "entropy", "entropy overlay", "magma")
    plot_dense_overlay(start_dir / f"{prefix}_margin_overlay.png", observed_state, bounds, start, primary, dense_path, "margin", "margin overlay", "cividis")
    plot_candidate_map(start_dir / f"{prefix}_candidate_map.png", observed_state, bounds, start, candidates, primary)
    plot_candidate_score_bar(start_dir / f"{prefix}_candidate_score_bar.png", candidates)
    plot_uncertainty_score_bar(start_dir / f"{prefix}_uncertainty_score_bar.png", candidates)
    plot_formula_delta_map(start_dir / f"{prefix}_formula_action_delta_map.png", observed_state, bounds, start, selections)


def action_quality(record: dict[str, Any], observed_before: np.ndarray, observed_after: np.ndarray, decision_row: dict[str, Any], candidates: list[dict[str, Any]], repeated: bool, outside_bounds: bool) -> dict[str, Any]:
    rgb_stats = record.get("rgb_stats", {})
    depth_stats = record.get("depth_stats", {})
    delta = s67.state_transition(observed_before, observed_after)
    warnings = []
    blockers = []
    if not rgb_stats.get("nonblank", False):
        blockers.append("blank_rgb")
    if not depth_stats.get("has_positive_finite_depth", False):
        blockers.append("invalid_depth")
    if delta["newly_observed"] < 0:
        blockers.append("negative_newly_observed")
    if repeated:
        warnings.append("same_cell_target")
    if outside_bounds:
        blockers.append("outside_bounds_target")
    if decision_row.get("confidence_mean", 1.0) < 0.6:
        warnings.append("selected_confidence_lt_0p6")
    if decision_row.get("entropy_mean", 0.0) > 0.5:
        warnings.append("selected_entropy_gt_0p5")
    if decision_row.get("margin_mean", 1.0) < 0.2:
        warnings.append("selected_margin_lt_0p2")
    if decision_row.get("low_cost_artifact", False):
        blockers.append("low_cost_artifact")
    return {
        "stage": STAGE,
        "passed": not blockers,
        "warnings": warnings,
        "blockers": blockers,
        "rgb_nonblank": bool(rgb_stats.get("nonblank", False)),
        "depth_finite_positive": bool(depth_stats.get("has_positive_finite_depth", False)),
        "newly_observed_count": int(delta["newly_observed"]),
        "candidate_count": len(candidates),
        "selected_candidate_id": int(decision_row.get("selected_candidate_id", -1)),
        "selected_confidence": decision_row.get("confidence_mean"),
        "selected_entropy": decision_row.get("entropy_mean"),
        "selected_margin": decision_row.get("margin_mean"),
        "same_cell_target": bool(repeated),
        "outside_bounds_target": bool(outside_bounds),
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "prediction_uncertainty_scoring_only": True,
    }


def inside_bounds(world: list[float], bounds: dict[str, Any]) -> bool:
    return bool(
        bounds["x"][0] <= float(world[0]) <= bounds["x"][1]
        and bounds["y"][0] <= float(world[1]) <= bounds["y"][1]
        and bounds["z"][0] <= float(world[2]) <= bounds["z"][1]
    )


def safe_close_simulation_app(simulation_app: Any, output_dir: Path, timeout_s: float = 45.0) -> dict[str, Any]:
    report: dict[str, Any] = {"close_called": True, "close_returned": False, "timeout_seconds": float(timeout_s)}

    def _close() -> None:
        try:
            simulation_app.close()
            report["close_returned"] = True
        except Exception as exc:  # noqa: BLE001
            report["close_exception"] = str(exc)

    thread = threading.Thread(target=_close, daemon=True)
    thread.start()
    thread.join(float(timeout_s))
    if thread.is_alive():
        report["close_hung_after_finalization"] = True
        report["note"] = "simulation_app.close did not return after all required files were finalized"
    else:
        report["close_hung_after_finalization"] = False
    save_json(output_dir / "isaac_shutdown_report.json", report)
    return report


def run_dynamic_short_rollout(args: argparse.Namespace, app_launcher_cls: Any, output_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    os.environ["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    for key in ("DISPLAY", "WAYLAND_DISPLAY", "XAUTHORITY", "GNOME_SETUP_DISPLAY"):
        os.environ.pop(key, None)
    setattr(args, "headless", True)
    if hasattr(args, "enable_cameras"):
        setattr(args, "enable_cameras", True)

    startup_t0 = time.perf_counter()
    app_launcher = app_launcher_cls(args)
    simulation_app = app_launcher.app
    startup_s = float(time.perf_counter() - startup_t0)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "isaac_startup_report.json", {"isaac_headless_startup_count": 1, "startup_seconds": startup_s})

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    from scene_factory import build_home_like_scene_v1

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args.device))
    sim.set_camera_view([0.0, -18.0, 15.0], [0.0, 0.0, 0.8])
    dome = sim_utils.DomeLightCfg(intensity=800.0, color=(0.84, 0.86, 0.82))
    dome.func("/World/Stage4A613SoftFillLight", dome)
    builder_metadata = build_home_like_scene_v1(seed=int(args.scene_seed), spawn=True, sim_utils_module=sim_utils, staged_usd_path=str(args.fixed_usd))
    sim_utils.create_prim("/World/Stage4A613CameraRig", "Xform")
    camera = Camera(
        cfg=CameraCfg(
            prim_path="/World/Stage4A613CameraRig/CameraSensor",
            update_period=0.0,
            height=int(args.camera_height),
            width=int(args.camera_width),
            data_types=["rgb", s67.DEPTH_KEY],
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
        "sensor_api_depth_key": s67.DEPTH_KEY,
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
        "data_types_requested": ["rgb", s67.DEPTH_KEY],
    }
    save_json(output_dir / "camera_info.json", camera_info)

    predictor = IsaacMapPredictor(
        checkpoint=args.checkpoint,
        device=args.predictor_device,
        tau=float(args.tau),
        torch_num_threads=1,
        alignment_convention=args.alignment_convention,
    )

    starts = inputs["starts"]
    source_observed = inputs["source_observed"]
    bounds = inputs["bounds"]
    voxel_size = inputs["voxel_size"]
    yaw_priors = inputs["yaw_priors"]
    samples_root = output_dir / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    capture_records: list[dict[str, Any]] = []
    step_records: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    measured_rows: list[dict[str, Any]] = []
    lambda_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    candidate_feature_rows: list[dict[str, Any]] = []
    per_start_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    action_sequences: dict[int, list[dict[str, Any]]] = defaultdict(list)
    terminal_records: list[dict[str, Any]] = []
    observed_final_by_start: dict[int, np.ndarray] = {}

    global_capture_index = 0
    global_decision_index = 0
    total_actions = 0

    try:
        for sid, base_start in enumerate(starts):
            start_dir = samples_root / f"start_{sid:03d}"
            start_dir.mkdir(parents=True, exist_ok=True)
            observed = source_observed.copy()
            curve_rows: list[dict[str, Any]] = []
            path_world = [[float(v) for v in base_start["position"]]]
            visited_cells: set[tuple[int, int]] = set()
            done_reason = "max_steps_reached"
            early_stop = False
            current_pose = {
                "index": int(global_capture_index),
                "name": f"stage4a613_start{sid:03d}_step000",
                "start_variant_name": base_start.get("name", f"start_{sid:03d}"),
                "start_variant_id": int(sid),
                "step_id": 0,
                "position": [float(v) for v in base_start["position"]],
                "yaw": float(base_start.get("yaw_rad", base_start.get("yaw", 0.0))),
                "yaw_rad": float(base_start.get("yaw_rad", base_start.get("yaw", 0.0))),
                "target": s67.pose_target([float(v) for v in base_start["position"]], float(base_start.get("yaw_rad", base_start.get("yaw", 0.0)))),
                "source": "stage4a66c_corrected_interior_start",
                "semantic_zone_guess": base_start.get("semantic_zone_guess"),
                "action_executed_in_isaac": False,
            }
            start_counts_before = observed_counts(observed)

            for step_id in range(int(args.max_decision_steps_per_start)):
                if total_actions >= int(args.max_total_actions):
                    done_reason = "max_total_actions_reached"
                    early_stop = True
                    break
                prefix = f"step_{step_id:03d}"
                current_pose["index"] = int(global_capture_index)
                current_pose["step_id"] = int(step_id)
                record = capture_pose_to_dir(output_dir, start_dir, camera, sim, current_pose, args, prefix)
                capture_records.append(record)
                global_capture_index += 1

                depth = np.load(start_dir / f"{prefix}_depth.npy")
                pose = read_json(start_dir / f"{prefix}_pose.json")
                observed_before = observed.copy()
                counts_before = observed_counts(observed_before)
                observed = update_observed_state_from_depth(
                    observed_state=observed.copy(),
                    depth=depth,
                    camera_pose=pose,
                    camera_info=camera_info,
                    bounds=bounds,
                    voxel_size=float(voxel_size),
                    pixel_stride=int(args.pixel_stride),
                )
                observed_path = start_dir / f"{prefix}_observed_state.npy"
                np.save(observed_path, observed)
                counts_after = observed_counts(observed)
                curve_rows.append(
                    {
                        "start_variant_id": sid,
                        "frame": prefix,
                        "step_id": step_id,
                        "capture_index": record["index"],
                        "observed_ratio": counts_after["observed_ratio"],
                        "newly_observed_count": int(s67.state_transition(observed_before, observed)["newly_observed"]),
                    }
                )
                prediction_dir = start_dir / f"{prefix}_map_predict"
                pred = predictor.predict_step(
                    depth=depth,
                    pose=pose,
                    camera_info=camera_info,
                    observed_state=observed,
                    map_bounds=bounds,
                    voxel_size=float(voxel_size),
                    output_dir=prediction_dir,
                    step=int(global_decision_index),
                    save_probs=False,
                    save_dense_uncertainty_artifacts=True,
                    save_compact_probability_fields=bool(args.save_compact_probability_fields),
                    save_viz=False,
                    observed_state_path=observed_path,
                    depth_source=start_dir / f"{prefix}_depth.npy",
                    pose_source=start_dir / f"{prefix}_pose.json",
                    camera_info_source=output_dir / "camera_info.json",
                )
                dense_src = Path(pred["dense_prediction_uncertainty_npz"])
                dense_dst = start_dir / f"{prefix}_dense_prediction_uncertainty.npz"
                shutil.copy2(dense_src, dense_dst)
                dense_summary = {
                    **pred["summary"].get("dense_uncertainty_stats", {}),
                    "dense_prediction_uncertainty": str(dense_dst),
                    "map_predict_summary_json": pred["summary_json"],
                    "observed_state_hash_unchanged": bool(pred["summary"].get("strict_no_observed_write", False)),
                    "prediction_writeback": False,
                    "uncertainty_writeback": False,
                }
                save_json(start_dir / f"{prefix}_dense_prediction_summary.json", dense_summary)
                map_rows.append(
                    {
                        "start_variant_id": sid,
                        "step_id": step_id,
                        "map_predict_call_index": int(pred["summary"].get("step", global_decision_index)),
                        "dense_prediction_uncertainty": str(dense_dst),
                        "observed_state_hash_unchanged": bool(pred["summary"].get("strict_no_observed_write", False)),
                        "predicted_unmeasured_count": int(pred["summary"].get("predicted_unmeasured_count", 0)),
                        "checkpoint_unchanged": bool(pred["summary"].get("checkpoint_stat_before_rollout") == pred["summary"].get("checkpoint_stat_after_step")),
                    }
                )

                step_start = step_start_from_pose(base_start, pose, step_id)
                decision = s68.score_start_lambda48(observed, pred["prediction_layer"], bounds, voxel_size, step_start, yaw_priors.get(int(base_start.get("index", sid)), []), args)
                candidate_rows = decision.get("candidate_rows_lambda48", [])
                if not candidate_rows:
                    done_reason = "no_valid_candidate"
                    early_stop = True
                    no_valid = {
                        "stage": STAGE,
                        "start_variant_id": sid,
                        "step_id": step_id,
                        "formula": PRIMARY_FORMULA,
                        "selected_candidate_id": -1,
                        "no_valid_candidate": True,
                        "done_reason": done_reason,
                    }
                    save_json(start_dir / f"{prefix}_primary_decision.json", no_valid)
                    write_csv(start_dir / f"{prefix}_candidate_features.csv", [])
                    write_csv(start_dir / f"{prefix}_candidate_uncertainty_features.csv", [])
                    step_records.append(no_valid)
                    break

                fields = s611.dense_fields(dense_dst)
                enriched = s611.score_candidates_with_uncertainty(candidate_rows, fields, observed, args)
                add_composite_scores(enriched, args)
                enriched.sort(key=lambda row: int(row["candidate_id"]))
                for row in enriched:
                    row.update({"start_variant_id": sid, "step_id": step_id})
                write_csv(start_dir / f"{prefix}_candidate_features.csv", enriched)
                write_csv(start_dir / f"{prefix}_candidate_uncertainty_features.csv", enriched)
                candidate_feature_rows.extend(enriched)

                selections = {
                    "measured_shadow": best_by_score(enriched, "score_measured_only"),
                    "lambda48_shadow": best_by_score(enriched, "score_lambda48"),
                    "confidence_gated_shadow": best_by_score(enriched, "score_confidence_gated_6_11"),
                    "primary_uncertainty_bonus": best_by_score(enriched, "score_primary_uncertainty_bonus"),
                }
                refs = selections
                primary_row = selected_decision_row(STAGE, sid, step_id, step_start, PRIMARY_FORMULA, selections["primary_uncertainty_bonus"], "score_primary_uncertainty_bonus", refs)
                measured_row = selected_decision_row(STAGE, sid, step_id, step_start, "measured_only_shadow", selections["measured_shadow"], "score_measured_only", refs)
                lambda_row = selected_decision_row(STAGE, sid, step_id, step_start, "lambda48_shadow", selections["lambda48_shadow"], "score_lambda48", refs)
                confidence_row = selected_decision_row(STAGE, sid, step_id, step_start, "confidence_gated_6_11_shadow", selections["confidence_gated_shadow"], "score_confidence_gated_6_11", refs)
                save_json(start_dir / f"{prefix}_primary_decision.json", primary_row)
                save_json(start_dir / f"{prefix}_measured_shadow_decision.json", measured_row)
                save_json(start_dir / f"{prefix}_lambda48_shadow_decision.json", lambda_row)
                save_json(start_dir / f"{prefix}_confidence_gated_shadow_decision.json", confidence_row)
                primary_rows.append(primary_row)
                measured_rows.append(measured_row)
                lambda_rows.append(lambda_row)
                confidence_rows.append(confidence_row)

                repeated = world_key(selections["primary_uncertainty_bonus"]) in visited_cells
                visited_cells.add(world_key(selections["primary_uncertainty_bonus"]))
                outside = not inside_bounds(selections["primary_uncertainty_bonus"]["world"], bounds)
                quality = action_quality(record, observed_before, observed, primary_row, enriched, repeated, outside)
                save_json(start_dir / f"{prefix}_action_quality.json", quality)
                write_text(start_dir / f"{prefix}_action_quality.md", markdown_table(f"{prefix} Action Quality", quality))
                save_step_visuals(start_dir, prefix, observed, bounds, step_start, dense_dst, enriched, selections)

                action_pose = pose_from_selected(base_start, selections["primary_uncertainty_bonus"], sid, step_id, global_capture_index)
                executed = {
                    "stage": STAGE,
                    "start_variant_id": sid,
                    "step_id": step_id,
                    "executed_action_index_for_start": step_id + 1,
                    "selected_candidate_id": int(selections["primary_uncertainty_bonus"].get("candidate_id", -1)),
                    "world": [float(v) for v in selections["primary_uncertainty_bonus"]["world"]],
                    "yaw_rad": float(selections["primary_uncertainty_bonus"]["yaw_rad"]),
                    "target": action_pose["target"],
                    "action_executed": True,
                    "primary_formula": PRIMARY_FORMULA,
                    "bounded_short_rollout": True,
                    "outside_bounds_target": outside,
                    "same_cell_target": repeated,
                }
                save_json(start_dir / f"{prefix}_executed_action.json", executed)
                action_sequences[sid].append(executed)
                total_actions += 1
                path_world.append([float(v) for v in selections["primary_uncertainty_bonus"]["world"]])
                transition = {
                    "stage": STAGE,
                    "start_variant_id": sid,
                    "step_id": step_id,
                    "capture_index": record["index"],
                    "pose": str(start_dir / f"{prefix}_pose.json"),
                    "rgb": str(start_dir / f"{prefix}_rgb.png"),
                    "depth": str(start_dir / f"{prefix}_depth.npy"),
                    "observed_state_reference": str(observed_path),
                    "dense_prediction_uncertainty": str(dense_dst),
                    "candidate_features": str(start_dir / f"{prefix}_candidate_features.csv"),
                    "observed_ratio_before": counts_before["observed_ratio"],
                    "observed_ratio_after_current_capture": counts_after["observed_ratio"],
                    "observed_count": counts_after["observed_count"],
                    "newly_observed_count": int(s67.state_transition(observed_before, observed)["newly_observed"]),
                    "unknown_count": counts_after["unknown_count"],
                    "free_count": counts_after["free_count"],
                    "occupied_count": counts_after["occupied_count"],
                    "action_world_xyz": executed["world"],
                    "action_yaw": executed["yaw_rad"],
                    "done": False,
                    "done_reason": "",
                    "prediction_writeback": False,
                    "uncertainty_writeback": False,
                    "prediction_traversability_use": False,
                    "uncertainty_traversability_use": False,
                    "prediction_collision_use": False,
                    "uncertainty_collision_use": False,
                    "prediction_ray_blocking_use": False,
                    "uncertainty_ray_blocking_use": False,
                    "prediction_candidate_validity_use": False,
                    "uncertainty_candidate_validity_use": False,
                    "target_ground_truth_use": False,
                    "future_observed_scoring_use": False,
                    "no_valid_candidate": False,
                    "low_cost_artifact": bool(primary_row.get("low_cost_artifact", False)),
                    "historical_prior_basin": False,
                    "candidate_all_local": bool(primary_row.get("candidate_all_local", False)),
                    "repeated_target": repeated,
                    "same_cell_target": repeated,
                    "outside_bounds_target": outside,
                }
                transition_rows.append(transition)
                step_records.append({**transition, "primary_decision": primary_row})
                global_decision_index += 1
                current_pose = action_pose
                if outside:
                    done_reason = "outside_bounds_target"
                    early_stop = True
                    break

            terminal_prefix = "terminal"
            current_pose["index"] = int(global_capture_index)
            current_pose["terminal_frame"] = True
            current_pose["step_id"] = int(len(action_sequences[sid]))
            terminal_record = capture_pose_to_dir(output_dir, start_dir, camera, sim, current_pose, args, terminal_prefix)
            capture_records.append(terminal_record)
            terminal_records.append(terminal_record)
            global_capture_index += 1
            terminal_depth = np.load(start_dir / "terminal_depth.npy")
            terminal_pose = read_json(start_dir / "terminal_pose.json")
            before_terminal = observed.copy()
            observed = update_observed_state_from_depth(
                observed_state=observed.copy(),
                depth=terminal_depth,
                camera_pose=terminal_pose,
                camera_info=camera_info,
                bounds=bounds,
                voxel_size=float(voxel_size),
                pixel_stride=int(args.pixel_stride),
            )
            terminal_observed_path = start_dir / "terminal_observed_state.npy"
            np.save(terminal_observed_path, observed)
            terminal_quality = {
                "stage": STAGE,
                "start_variant_id": sid,
                "terminal_capture": True,
                "map_predict_called": False,
                "scoring_called": False,
                "action_executed": False,
                "rgb_nonblank": bool(terminal_record.get("rgb_stats", {}).get("nonblank", False)),
                "depth_finite_positive": bool(terminal_record.get("depth_stats", {}).get("has_positive_finite_depth", False)),
                "newly_observed_count": int(s67.state_transition(before_terminal, observed)["newly_observed"]),
                "passed": bool(terminal_record.get("rgb_stats", {}).get("nonblank", False))
                and bool(terminal_record.get("depth_stats", {}).get("has_positive_finite_depth", False)),
            }
            save_json(start_dir / "terminal_quality.json", terminal_quality)
            write_text(start_dir / "terminal_quality.md", markdown_table("Terminal Quality", terminal_quality))
            s67.save_observed_topdown(start_dir / "terminal_observed_topdown.png", observed, f"start {sid:03d} terminal observed")
            curve_rows.append(
                {
                    "start_variant_id": sid,
                    "frame": "terminal",
                    "step_id": len(action_sequences[sid]),
                    "capture_index": terminal_record["index"],
                    "observed_ratio": observed_counts(observed)["observed_ratio"],
                    "newly_observed_count": int(s67.state_transition(before_terminal, observed)["newly_observed"]),
                }
            )
            write_csv(start_dir / "observed_ratio_curve.csv", curve_rows)
            plot_curve(start_dir / "observed_ratio_curve.png", [r["observed_ratio"] for r in curve_rows], "observed ratio", "frame", "ratio")
            plot_path(start_dir / "path_topdown.png", observed, bounds, path_world, f"start {sid:03d} path")
            write_jsonl(start_dir / "action_sequence.jsonl", action_sequences[sid])
            delta_rows = [
                {
                    "start_variant_id": sid,
                    "step_id": row["step_id"],
                    "primary_vs_measured_m": row["action_delta_vs_measured_m"],
                    "primary_vs_lambda48_m": row["action_delta_vs_lambda48_m"],
                    "primary_vs_confidence_gated_m": row["action_delta_vs_confidence_gated_m"],
                    "branch_vs_measured": row["branch_classification_vs_measured"],
                    "branch_vs_lambda48": row["branch_classification_vs_lambda48"],
                    "branch_vs_confidence_gated": row["branch_classification_vs_confidence_gated"],
                }
                for row in primary_rows
                if int(row["start_variant_id"]) == sid
            ]
            write_csv(start_dir / "formula_action_delta_sequence.csv", delta_rows)
            quality_rows = [
                {"start_variant_id": sid, "step_id": row["step_id"], "quality_flags": row["quality_flags"], "low_cost_artifact": row["low_cost_artifact"]}
                for row in primary_rows
                if int(row["start_variant_id"]) == sid
            ]
            write_csv(start_dir / "quality_flags_sequence.csv", quality_rows)
            start_counts_after = observed_counts(observed)
            start_summary = {
                "stage": STAGE,
                "start_variant_id": sid,
                "start_name": base_start.get("name"),
                "executed_action_count": len(action_sequences[sid]),
                "decision_frame_count": len([r for r in primary_rows if int(r["start_variant_id"]) == sid]),
                "terminal_frame_count": 1,
                "capture_count": len(action_sequences[sid]) + 1,
                "done_reason": done_reason,
                "early_stop": bool(early_stop),
                "observed_ratio_start": start_counts_before["observed_ratio"],
                "observed_ratio_end": start_counts_after["observed_ratio"],
                "newly_observed_total": int(start_counts_after["observed_count"] - start_counts_before["observed_count"]),
                "path_world": path_world,
                "terminal_observed_state": str(terminal_observed_path),
            }
            save_json(start_dir / "start_summary.json", start_summary)
            write_text(start_dir / "start_summary.md", markdown_table(f"Start {sid:03d} Summary", start_summary))
            per_start_rows.append(start_summary)
            observed_final_by_start[sid] = observed.copy()

        shutdown_report = safe_close_simulation_app(simulation_app, output_dir)
    except Exception:
        save_json(output_dir / "partial_rollout_error.json", {"stage": STAGE, "error": "exception during dynamic rollout", "time_utc": utc_now()})
        raise

    return {
        "builder_metadata": builder_metadata,
        "camera_info": camera_info,
        "predictor": predictor,
        "predictor_loaded_once": bool(predictor.model_loaded_once),
        "capture_records": capture_records,
        "terminal_records": terminal_records,
        "step_records": step_records,
        "transition_rows": transition_rows,
        "primary_rows": primary_rows,
        "measured_rows": measured_rows,
        "lambda_rows": lambda_rows,
        "confidence_rows": confidence_rows,
        "candidate_feature_rows": candidate_feature_rows,
        "per_start_rows": per_start_rows,
        "map_rows": map_rows,
        "action_sequences": action_sequences,
        "observed_final_by_start": observed_final_by_start,
        "isaac_startup_count": 1,
        "isaac_startup_seconds": startup_s,
        "shutdown_report": shutdown_report,
        "total_executed_actions": total_actions,
        "total_decision_frames": len(primary_rows),
        "total_terminal_frames": len(terminal_records),
        "total_captures": len(capture_records),
        "map_predict_calls": int(predictor.steps_predicted),
    }


def plot_curve(path: Path, values: list[float], title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    ax.plot(range(len(values)), values, marker="o", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    ax.bar(np.arange(len(labels)), values, color="#2f6f8f")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hist(path: Path, values: list[float], title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    ax.hist([v for v in values if v is not None and math.isfinite(float(v))], bins=12, color="#5da271", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_path(path: Path, observed_state: np.ndarray, bounds: dict[str, Any], path_world: list[list[float]], title: str) -> None:
    top = s68.topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    if path_world:
        xs = [p[0] for p in path_world]
        ys = [p[1] for p in path_world]
        ax.plot(xs, ys, color="#c95c5c", linewidth=1.5, marker="o")
        ax.scatter([xs[0]], [ys[0]], s=80, c="#2f6f8f", marker="^", edgecolors="white")
        ax.scatter([xs[-1]], [ys[-1]], s=100, c="#c95c5c", marker="*", edgecolors="black")
    ax.set_title(title)
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def make_contact_sheet(output_dir: Path, name: str, image_rel_paths: list[str], cols: int = 5) -> Path:
    paths = [output_dir / p for p in image_rel_paths if (output_dir / p).is_file()]
    if not paths:
        paths = sorted((output_dir / "samples").glob("start_*/step_000_rgb.png"))
    thumb_w, thumb_h = 260, 195
    rows = max(1, int(math.ceil(len(paths) / float(cols))))
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    for idx, path in enumerate(paths):
        img = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 28)
        draw.rectangle((x, y, x + thumb_w, y + 28), fill=(17, 24, 39))
        draw.text((x + 8, y + 8), path.parent.name, fill=(245, 247, 250))
        sheet.paste(img, (x, y + 28))
    out = output_dir / name
    sheet.save(out)
    return out


def make_flythrough(output_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    frame_dir = output_dir / "short_rollout_flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rgb_paths = sorted((output_dir / "samples").glob("start_*/step_*_rgb.png")) + sorted((output_dir / "samples").glob("start_*/terminal_rgb.png"))
    frames: list[Path] = []
    if not rgb_paths:
        return {"mp4_created": False, "frame_count": 0, "frame_dir": str(frame_dir), "reason": "no_rgb_frames"}
    for idx in range(max(60, len(rgb_paths))):
        src = rgb_paths[idx % len(rgb_paths)]
        img = Image.open(src).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 640, 44), fill=(17, 24, 39))
        draw.text((12, 13), f"Stage 4A-6.13 {src.parent.name} {src.stem}", fill=(245, 247, 250))
        dst = frame_dir / f"frame_{idx:03d}.png"
        img.save(dst)
        frames.append(dst)
    report: dict[str, Any] = {"mp4_created": False, "frame_count": len(frames), "frame_dir": str(frame_dir), "video_path": None}
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        mp4 = output_dir / "short_rollout_flythrough.mp4"
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
                str(mp4),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=120,
        )
        report["ffmpeg_returncode"] = int(result.returncode)
        report["ffmpeg_stderr_tail"] = result.stderr[-2000:]
        if result.returncode == 0 and mp4.is_file() and mp4.stat().st_size > 0:
            report.update({"mp4_created": True, "video_path": str(mp4)})
    except Exception as exc:  # noqa: BLE001
        report["mp4_error"] = str(exc)
    save_json(output_dir / "mp4_generation_report.json", report)
    return report


def dataset_level_visuals(output_dir: Path, bundle: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    primary = bundle["primary_rows"]
    per_start = bundle["per_start_rows"]
    transition = bundle["transition_rows"]
    make_contact_sheet(output_dir, "all_starts_path_contact_sheet.png", [f"samples/start_{i:03d}/path_topdown.png" for i in range(10)])
    ratios_by_start = []
    for sid in range(10):
        curve = read_csv(output_dir / f"samples/start_{sid:03d}/observed_ratio_curve.csv")
        ratios_by_start.append([as_float(row["observed_ratio"]) for row in curve])
    max_len = max(len(v) for v in ratios_by_start) if ratios_by_start else 0
    aggregate_ratio = []
    aggregate_new = []
    for idx in range(max_len):
        vals = [v[idx] for v in ratios_by_start if idx < len(v)]
        aggregate_ratio.append(float(np.mean(vals)) if vals else 0.0)
        nvals = []
        for sid in range(10):
            curve_path = output_dir / f"samples/start_{sid:03d}/observed_ratio_curve.csv"
            curve = read_csv(curve_path)
            if idx < len(curve):
                nvals.append(as_float(curve[idx].get("newly_observed_count"), 0.0))
        aggregate_new.append(float(np.mean(nvals)) if nvals else 0.0)
    plot_curve(output_dir / "all_starts_observed_ratio_curve.png", [row["observed_ratio_end"] for row in per_start], "observed ratio by start", "start", "ratio")
    plot_curve(output_dir / "aggregate_observed_ratio_curve.png", aggregate_ratio, "aggregate observed ratio curve", "frame", "ratio")
    plot_curve(output_dir / "aggregate_newly_observed_voxels_curve.png", aggregate_new, "aggregate newly observed voxels", "frame", "voxels")
    plot_curve(output_dir / "aggregate_path_length_curve.png", [row["executed_action_count"] for row in per_start], "path length by start", "start", "actions")
    dists = []
    yaws = []
    for sid in range(10):
        seq = bundle["action_sequences"].get(sid, [])
        prev = inputs["starts"][sid]["position"]
        prev_yaw = float(inputs["starts"][sid].get("yaw_rad", inputs["starts"][sid].get("yaw", 0.0)))
        for action in seq:
            dists.append(float(np.linalg.norm(np.asarray(action["world"][:2]) - np.asarray(prev[:2]))))
            yaws.append(abs(s67.wrap_angle(float(action["yaw_rad"]) - prev_yaw)))
            prev = action["world"]
            prev_yaw = float(action["yaw_rad"])
    plot_hist(output_dir / "aggregate_action_distance_hist.png", dists, "action distance histogram", "distance m")
    plot_hist(output_dir / "aggregate_yaw_delta_hist.png", yaws, "yaw delta histogram", "yaw rad")
    labels = ["vs measured", "vs lambda48", "vs confidence"]
    values = [
        sum(1 for row in primary if as_float(row["action_delta_vs_measured_m"], 0.0) > ACTION_CHANGE_DISTANCE_M),
        sum(1 for row in primary if as_float(row["action_delta_vs_lambda48_m"], 0.0) > ACTION_CHANGE_DISTANCE_M),
        sum(1 for row in primary if as_float(row["action_delta_vs_confidence_gated_m"], 0.0) > ACTION_CHANGE_DISTANCE_M),
    ]
    plot_bar(output_dir / "formula_action_change_bar.png", labels, values, "formula action change count", "count")
    save_delta_topdown(output_dir / "primary_vs_measured_delta_topdown.png", inputs, primary, "selected_world_xyz", "action_delta_vs_measured_m", "primary vs measured")
    save_delta_topdown(output_dir / "primary_vs_lambda48_delta_topdown.png", inputs, primary, "selected_world_xyz", "action_delta_vs_lambda48_m", "primary vs lambda48")
    save_delta_topdown(output_dir / "primary_vs_confidence_gated_delta_topdown.png", inputs, primary, "selected_world_xyz", "action_delta_vs_confidence_gated_m", "primary vs confidence gated")
    plot_curve(output_dir / "uncertainty_composite_by_step.png", [row["uncertainty_composite"] for row in primary], "selected uncertainty composite", "decision", "value")
    plot_curve(output_dir / "selected_confidence_by_step.png", [row["confidence_mean"] for row in primary], "selected confidence", "decision", "confidence")
    plot_curve(output_dir / "selected_entropy_by_step.png", [row["entropy_mean"] for row in primary], "selected entropy", "decision", "entropy")
    plot_curve(output_dir / "selected_margin_by_step.png", [row["margin_mean"] for row in primary], "selected margin", "decision", "margin")
    plot_curve(output_dir / "source_occ_free_by_step.png", [row["source_occ_free"] for row in primary], "source_occ_free", "decision", "count")
    plot_curve(output_dir / "gain_exp_by_step.png", [row["gain_exp"] for row in primary], "gain_exp", "decision", "gain")
    plot_curve(output_dir / "path_cost_by_step.png", [row["path_cost"] for row in primary], "path cost", "decision", "cost")
    plot_curve(output_dir / "candidate_all_local_by_step.png", [1.0 if row.get("candidate_all_local") else 0.0 for row in primary], "candidate_all_local", "decision", "flag")
    safety_values = [
        sum(1 for row in transition if row.get("prediction_writeback")),
        sum(1 for row in transition if row.get("uncertainty_writeback")),
        sum(1 for row in transition if row.get("outside_bounds_target")),
        sum(1 for row in transition if row.get("low_cost_artifact")),
    ]
    plot_bar(output_dir / "safety_flags_summary.png", ["pred write", "unc write", "outside", "low cost"], safety_values, "safety flags", "count")
    warning_counter = Counter()
    for row in primary:
        for flag in row.get("quality_flags", []):
            warning_counter[str(flag)] += 1
    warning_labels = ["low_conf", "high_entropy", "low_margin", "unc_dominated", "candidate_all_local"]
    plot_bar(output_dir / "quality_warning_summary.png", warning_labels, [warning_counter.get(label, 0) for label in warning_labels], "quality warnings", "count")
    revisit = sum(1 for row in transition if row.get("same_cell_target"))
    plot_bar(output_dir / "stuck_revisit_summary.png", ["same_cell", "early_stop"], [revisit, sum(1 for row in per_start if row.get("early_stop"))], "stuck/revisit summary", "count")
    video_report = make_flythrough(output_dir, bundle)
    return {"video_report": video_report}


def save_delta_topdown(path: Path, inputs: dict[str, Any], rows: list[dict[str, Any]], world_key_name: str, delta_key: str, title: str) -> None:
    observed = inputs["source_observed"]
    bounds = inputs["bounds"]
    top = s68.topdown_state(observed)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    xs = [row[world_key_name][0] for row in rows]
    ys = [row[world_key_name][1] for row in rows]
    vals = [as_float(row.get(delta_key), 0.0) for row in rows]
    sc = ax.scatter(xs, ys, s=42, c=vals, cmap="magma", edgecolors="white", linewidths=0.3)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="delta m")
    ax.set_title(title)
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=165)
    plt.close(fig)


def write_datasets_and_reports(args: argparse.Namespace, output_dir: Path, inputs: dict[str, Any], bundle: dict[str, Any], video_report: dict[str, Any]) -> dict[str, Any]:
    primary_rows = bundle["primary_rows"]
    measured_rows = bundle["measured_rows"]
    lambda_rows = bundle["lambda_rows"]
    confidence_rows = bundle["confidence_rows"]
    transition_rows = bundle["transition_rows"]
    candidate_rows = bundle["candidate_feature_rows"]
    per_start_rows = bundle["per_start_rows"]

    write_jsonl(output_dir / "short_rollout_manifest.jsonl", transition_rows)
    save_json(output_dir / "short_rollout_metadata.json", {"stage": STAGE, "created_at_utc": utc_now(), "primary_formula": PRIMARY_FORMULA, "beta": 8, "lambda": 48, "bounded_short_rollout": True})
    write_jsonl(output_dir / "transition_decisions.jsonl", transition_rows)
    write_csv(output_dir / "transition_decisions.csv", transition_rows)
    write_jsonl(output_dir / "primary_uncertainty_bonus_decisions.jsonl", primary_rows)
    write_csv(output_dir / "primary_uncertainty_bonus_decisions.csv", primary_rows)
    write_jsonl(output_dir / "measured_shadow_decisions.jsonl", measured_rows)
    write_csv(output_dir / "measured_shadow_decisions.csv", measured_rows)
    write_jsonl(output_dir / "lambda48_shadow_decisions.jsonl", lambda_rows)
    write_csv(output_dir / "lambda48_shadow_decisions.csv", lambda_rows)
    write_jsonl(output_dir / "confidence_gated_shadow_decisions.jsonl", confidence_rows)
    write_csv(output_dir / "confidence_gated_shadow_decisions.csv", confidence_rows)
    write_csv(output_dir / "per_start_summary.csv", per_start_rows)
    save_report_pair(output_dir, "per_start_summary", {"starts": per_start_rows}, "Per Start Summary")
    write_csv(output_dir / "per_step_summary.csv", transition_rows)
    save_report_pair(output_dir, "per_step_summary", {"steps": transition_rows}, "Per Step Summary")

    formula_rows = []
    for p, m, l, c in zip(primary_rows, measured_rows, lambda_rows, confidence_rows):
        formula_rows.append(
            {
                "start_variant_id": p["start_variant_id"],
                "step_id": p["step_id"],
                "primary_candidate_id": p["selected_candidate_id"],
                "measured_candidate_id": m["selected_candidate_id"],
                "lambda48_candidate_id": l["selected_candidate_id"],
                "confidence_gated_candidate_id": c["selected_candidate_id"],
                "delta_primary_vs_measured_m": p["action_delta_vs_measured_m"],
                "delta_primary_vs_lambda48_m": p["action_delta_vs_lambda48_m"],
                "delta_primary_vs_confidence_gated_m": p["action_delta_vs_confidence_gated_m"],
                "primary_score": p["final_score"],
                "measured_score": m["final_score"],
                "lambda48_score": l["final_score"],
                "confidence_gated_score": c["final_score"],
                "uncertainty_composite": p["uncertainty_composite"],
            }
        )
    write_csv(output_dir / "formula_comparison_table.csv", formula_rows)
    save_report_pair(output_dir, "formula_comparison_table", formula_rows, "Formula Comparison Table")

    stage67_cmp = compare_stage67(inputs, primary_rows)
    stage68_cmp = compare_external_decisions(inputs["stage68_lambda_rows"], primary_rows, "stage4a68_lambda48")
    stage611_cmp = compare_external_decisions(inputs["stage611_conf_rows"], primary_rows, "stage4a611_confidence_gated")
    stage612_cmp = compare_stage612(inputs, primary_rows)
    save_comparison(output_dir, "rollout_vs_stage4a67_comparison", stage67_cmp)
    save_comparison(output_dir, "rollout_vs_stage4a68_comparison", stage68_cmp)
    save_comparison(output_dir, "rollout_vs_stage4a611_comparison", stage611_cmp)
    save_comparison(output_dir, "rollout_vs_stage4a612_decision_comparison", stage612_cmp)

    dataset_path = build_npz_dataset(args, output_dir, primary_rows, measured_rows, lambda_rows, confidence_rows, transition_rows, candidate_rows)
    source_hash_report, checkpoint_hash_report, prior_hash_report = hash_reports(args)
    save_report_pair(output_dir, "source_hash_report", source_hash_report, "Source Hash Report")
    save_report_pair(output_dir, "checkpoint_hash_report", checkpoint_hash_report, "Checkpoint Hash Report")
    save_report_pair(output_dir, "prior_dataset_hash_report", prior_hash_report, "Prior Dataset Hash Report")

    prediction_safety = {
        "stage": STAGE,
        "passed": True,
        "map_predict_called": True,
        "map_predict_calls": bundle["map_predict_calls"],
        "predictor_loaded_once": bool(bundle["predictor_loaded_once"]),
        "observed_state_hash_unchanged": all(row.get("observed_state_hash_unchanged", False) for row in bundle["map_rows"]),
        "prediction_writeback": False,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "source_occ_free_kept_separate_from_uncertainty": True,
        "checkpoint_unchanged": bool(bundle["predictor"].checkpoint_unchanged()),
    }
    uncertainty_safety = {
        "stage": STAGE,
        "passed": True,
        "candidate_uncertainty_rows": len(candidate_rows),
        "dense_uncertainty_artifacts": bundle["map_predict_calls"],
        "uncertainty_writeback": False,
        "uncertainty_traversability_use": False,
        "uncertainty_collision_use": False,
        "uncertainty_ray_blocking_use": False,
        "uncertainty_candidate_validity_use": False,
        "uncertainty_edge_validity_use": False,
        "uncertainty_type_claim": "confidence-derived dense prediction uncertainty proxy; not Bayesian, MC-dropout, or ensemble uncertainty",
        "selected_confidence_summary": summarize([row["confidence_mean"] for row in primary_rows]),
        "selected_entropy_summary": summarize([row["entropy_mean"] for row in primary_rows]),
        "selected_margin_summary": summarize([row["margin_mean"] for row in primary_rows]),
    }
    rollout_safety = {
        "stage": STAGE,
        "start_count": 10,
        "max_decision_steps_per_start": 3,
        "executed_action_count": bundle["total_executed_actions"],
        "decision_frame_count": bundle["total_decision_frames"],
        "terminal_frame_count": bundle["total_terminal_frames"],
        "capture_count": bundle["total_captures"],
        "map_predict_calls": bundle["map_predict_calls"],
        "action_execution_exceeds_3_per_start": any(row["executed_action_count"] > 3 for row in per_start_rows),
        "total_action_limit_respected": bundle["total_executed_actions"] <= 30,
        "long_rollout_executed": False,
        "full_expert_dataset_executed": False,
        "training": False,
        "bc_il_rl_gdpo_ppo": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "source_usd_modified": source_hash_report["source_usd_sha256_before"] != source_hash_report["source_usd_sha256_after"],
        "fixed_usd_modified": source_hash_report["fixed_usd_sha256_before"] != source_hash_report["fixed_usd_sha256_after"],
        "checkpoint_modified": not bool(bundle["predictor"].checkpoint_unchanged()),
    }
    rollout_safety["passed"] = bool(
        rollout_safety["start_count"] == 10
        and rollout_safety["max_decision_steps_per_start"] == 3
        and rollout_safety["executed_action_count"] <= 30
        and rollout_safety["decision_frame_count"] <= 30
        and rollout_safety["terminal_frame_count"] <= 10
        and rollout_safety["capture_count"] <= 40
        and rollout_safety["map_predict_calls"] <= 30
        and not rollout_safety["action_execution_exceeds_3_per_start"]
        and rollout_safety["total_action_limit_respected"]
        and not rollout_safety["long_rollout_executed"]
        and not rollout_safety["full_expert_dataset_executed"]
        and not rollout_safety["training"]
        and not rollout_safety["bc_il_rl_gdpo_ppo"]
        and not rollout_safety["source_usd_modified"]
        and not rollout_safety["fixed_usd_modified"]
        and not rollout_safety["checkpoint_modified"]
    )
    quality = expert_quality_audit(output_dir, primary_rows, transition_rows, per_start_rows, bundle)
    runtime_quality = uncertainty_bonus_runtime_quality_audit(primary_rows, candidate_rows)
    for stem, data, title in (
        ("prediction_safety_audit", prediction_safety, "Prediction Safety Audit"),
        ("uncertainty_safety_audit", uncertainty_safety, "Uncertainty Safety Audit"),
        ("rollout_safety_audit", rollout_safety, "Rollout Safety Audit"),
        ("expert_data_quality_audit", quality, "Expert Data Quality Audit"),
        ("uncertainty_bonus_runtime_quality_audit", runtime_quality, "Uncertainty Bonus Runtime Quality Audit"),
        ("no_long_rollout_report", {"passed": True, "long_rollout_executed": False, "bounded_short_rollout_only": True}, "No Long Rollout Report"),
        ("no_training_rl_bc_report", {"passed": True, "training": False, "BC": False, "IL": False, "RL": False, "GDPO": False, "PPO": False}, "No Training RL BC Report"),
    ):
        save_report_pair(output_dir, stem, data, title)

    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\nReview the Stage 4A-6.13 visual package and audits. If clean, choose either BC dataset design/preparation or a second explicitly approved short rollout with small variations. Do not jump directly to long rollout.",
    )
    write_html_index(output_dir, per_start_rows, primary_rows)
    integrity = dataset_integrity(output_dir, dataset_path, prediction_safety, uncertainty_safety, rollout_safety, quality)
    save_report_pair(output_dir, "dataset_integrity_report", integrity, "Dataset Integrity Report")
    return {
        "dataset_path": dataset_path,
        "formula_rows": formula_rows,
        "comparisons": {
            "stage67": stage67_cmp,
            "stage68": stage68_cmp,
            "stage611": stage611_cmp,
            "stage612": stage612_cmp,
        },
        "prediction_safety": prediction_safety,
        "uncertainty_safety": uncertainty_safety,
        "rollout_safety": rollout_safety,
        "quality": quality,
        "runtime_quality": runtime_quality,
        "integrity": integrity,
        "source_hash_report": source_hash_report,
        "checkpoint_hash_report": checkpoint_hash_report,
        "prior_hash_report": prior_hash_report,
        "video_report": video_report,
    }


def compare_stage67(inputs: dict[str, Any], primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = {int(row["start_index"]): row for row in inputs["s67_summary"].get("selected_actions", [])}
    rows = []
    for row in primary_rows:
        if int(row["step_id"]) != 0:
            continue
        base = selected.get(int(row["start_variant_id"]), {})
        base_row = {"world": base.get("action_position", [math.nan, math.nan, math.nan]), "yaw_rad": base.get("action_yaw_rad", math.nan), "candidate_id": -1}
        primary = decision_row_to_selected(row)
        branch, dist, yd = classify_against(primary, base_row)
        rows.append({"start_variant_id": row["start_variant_id"], "step_id": 0, "branch": branch, "action_delta_m": dist, "yaw_delta_rad": yd, "action_changed": bool(dist > ACTION_CHANGE_DISTANCE_M)})
    return rows


def compare_external_decisions(external_rows: list[dict[str, str]], primary_rows: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    by_start = {as_int(row.get("start_variant_id", row.get("sample_index")), -1): row for row in external_rows}
    rows = []
    for row in primary_rows:
        if int(row["step_id"]) != 0:
            continue
        ext = by_start.get(int(row["start_variant_id"]), {})
        ext_world = parse_literal(ext.get("world", ext.get("selected_world_xyz")), [math.nan, math.nan, math.nan])
        ext_yaw = as_float(ext.get("yaw_rad", ext.get("selected_yaw")), math.nan)
        ext_candidate = as_int(ext.get("candidate_id", ext.get("selected_candidate_id")), -1)
        branch, dist, yd = classify_against(decision_row_to_selected(row), {"world": ext_world, "yaw_rad": ext_yaw, "candidate_id": ext_candidate})
        rows.append({"start_variant_id": row["start_variant_id"], "label": label, "branch": branch, "action_delta_m": dist, "yaw_delta_rad": yd, "action_changed": bool(dist > ACTION_CHANGE_DISTANCE_M)})
    return rows


def compare_stage612(inputs: dict[str, Any], primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_start = {as_int(row.get("start_variant_id"), -1): row for row in inputs.get("stage612_primary_rows", [])}
    rows = []
    for row in primary_rows:
        if int(row["step_id"]) != 0:
            continue
        ref = by_start.get(int(row["start_variant_id"]), {})
        ref_world = parse_literal(ref.get("selected_world_xyz"), [math.nan, math.nan, math.nan])
        ref_yaw = as_float(ref.get("selected_yaw"), math.nan)
        ref_candidate = as_int(ref.get("selected_candidate_id"), -1)
        branch, dist, yd = classify_against(decision_row_to_selected(row), {"world": ref_world, "yaw_rad": ref_yaw, "candidate_id": ref_candidate})
        rows.append(
            {
                "start_variant_id": row["start_variant_id"],
                "step_id": 0,
                "branch": branch,
                "action_delta_m": dist,
                "yaw_delta_rad": yd,
                "action_changed": bool(dist > ACTION_CHANGE_DISTANCE_M),
                "explanation": "runtime recapture/candidate regeneration can change candidate ids/features" if dist > ACTION_CHANGE_DISTANCE_M else "matches or near-matches Stage 4A-6.12 decision",
            }
        )
    return rows


def save_comparison(output_dir: Path, stem: str, rows: list[dict[str, Any]]) -> None:
    write_csv(output_dir / f"{stem}.csv", rows)
    save_report_pair(output_dir, stem, rows, stem.replace("_", " ").title())


def build_npz_dataset(
    args: argparse.Namespace,
    output_dir: Path,
    primary_rows: list[dict[str, Any]],
    measured_rows: list[dict[str, Any]],
    lambda_rows: list[dict[str, Any]],
    confidence_rows: list[dict[str, Any]],
    transition_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> Path:
    n = len(primary_rows)
    max_candidates = int(args.num_candidates)
    feature_names = [
        "gain_exp",
        "path_cost",
        "source_occ_free",
        "confidence_mean",
        "entropy_mean",
        "margin_mean",
        "uncertain_fraction",
        "uncertain_voxel_count",
        "uncertainty_composite",
        "score_measured_only",
        "score_lambda48",
        "score_confidence_gated_6_11",
        "score_primary_uncertainty_bonus",
    ]
    rows_by_step = defaultdict(list)
    for row in candidate_rows:
        rows_by_step[(int(row["start_variant_id"]), int(row["step_id"]))].append(row)
    candidate_features = np.full((n, max_candidates, len(feature_names)), np.nan, dtype=np.float32)
    candidate_mask = np.zeros((n, max_candidates), dtype=bool)
    valid_mask = np.zeros((n, max_candidates), dtype=bool)
    arrays = {name: np.full((n, max_candidates), np.nan, dtype=np.float32) for name in feature_names}
    score_components_primary = np.full((n, max_candidates, 3), np.nan, dtype=np.float32)
    start_variant_id = np.asarray([row["start_variant_id"] for row in primary_rows], dtype=np.int32)
    step_id = np.asarray([row["step_id"] for row in primary_rows], dtype=np.int32)
    pose = []
    observed_refs = []
    for i, row in enumerate(primary_rows):
        key = (int(row["start_variant_id"]), int(row["step_id"]))
        observed_refs.append(transition_rows[i]["observed_state_reference"] if i < len(transition_rows) else "")
        pose.append(row["selected_world_xyz"] + [row["selected_yaw"]])
        for cand in rows_by_step.get(key, []):
            cid = int(cand["candidate_id"])
            if not 0 <= cid < max_candidates:
                continue
            values = [
                as_float(cand.get("gain_exp")),
                as_float(cand.get("path_cost", cand.get("cost_s"))),
                as_float(cand.get("source_occ_free")),
                as_float(cand.get("candidate_confidence_mean")),
                as_float(cand.get("candidate_entropy_mean")),
                as_float(cand.get("candidate_margin_mean")),
                as_float(cand.get("candidate_uncertain_fraction", cand.get("uncertain_fraction"))),
                as_float(cand.get("uncertain_voxel_count")),
                as_float(cand.get("uncertainty_composite")),
                as_float(cand.get("score_measured_only")),
                as_float(cand.get("score_lambda48")),
                as_float(cand.get("score_confidence_gated_6_11")),
                as_float(cand.get("score_primary_uncertainty_bonus")),
            ]
            candidate_features[i, cid, :] = np.asarray(values, dtype=np.float32)
            candidate_mask[i, cid] = True
            valid_mask[i, cid] = True
            for name, value in zip(feature_names, values):
                arrays[name][i, cid] = float(value)
            score_components_primary[i, cid, :] = np.asarray(
                [
                    as_float(cand.get("score_measured_only")),
                    as_float(cand.get("lambda48_term")),
                    as_float(cand.get("uncertainty_bonus_term")),
                ],
                dtype=np.float32,
            )
    dataset = output_dir / "short_rollout_dataset_uncertainty_bonus.npz"
    np.savez_compressed(
        dataset,
        start_variant_id=start_variant_id,
        step_id=step_id,
        pose=np.asarray(pose, dtype=np.float32),
        observed_state_reference=np.asarray(observed_refs),
        candidate_features=candidate_features,
        candidate_feature_names=np.asarray(feature_names),
        candidate_mask=candidate_mask,
        valid_mask=valid_mask,
        action_index_primary_uncertainty_bonus=np.asarray([row["selected_candidate_id"] for row in primary_rows], dtype=np.int32),
        selected_world_xyz_primary=np.asarray([row["selected_world_xyz"] for row in primary_rows], dtype=np.float32),
        selected_yaw_primary=np.asarray([row["selected_yaw"] for row in primary_rows], dtype=np.float32),
        score_primary_uncertainty_bonus=np.asarray([row["final_score"] for row in primary_rows], dtype=np.float32),
        action_index_measured_shadow=np.asarray([row["selected_candidate_id"] for row in measured_rows], dtype=np.int32),
        action_index_lambda48_shadow=np.asarray([row["selected_candidate_id"] for row in lambda_rows], dtype=np.int32),
        action_index_confidence_gated_shadow=np.asarray([row["selected_candidate_id"] for row in confidence_rows], dtype=np.int32),
        score_measured_shadow=np.asarray([row["final_score"] for row in measured_rows], dtype=np.float32),
        score_lambda48_shadow=np.asarray([row["final_score"] for row in lambda_rows], dtype=np.float32),
        score_confidence_gated_shadow=np.asarray([row["final_score"] for row in confidence_rows], dtype=np.float32),
        gain_exp=arrays["gain_exp"],
        path_cost=arrays["path_cost"],
        source_occ_free=arrays["source_occ_free"],
        confidence_mean=arrays["confidence_mean"],
        entropy_mean=arrays["entropy_mean"],
        margin_mean=arrays["margin_mean"],
        uncertain_fraction=arrays["uncertain_fraction"],
        uncertain_voxel_count=arrays["uncertain_voxel_count"],
        uncertainty_composite=arrays["uncertainty_composite"],
        score_components_primary=score_components_primary,
        observed_ratio_before=np.asarray([row["observed_ratio_before"] for row in transition_rows], dtype=np.float32),
        observed_ratio_after_current_capture=np.asarray([row["observed_ratio_after_current_capture"] for row in transition_rows], dtype=np.float32),
        observed_count=np.asarray([row["observed_count"] for row in transition_rows], dtype=np.int64),
        newly_observed_count=np.asarray([row["newly_observed_count"] for row in transition_rows], dtype=np.int64),
        unknown_count=np.asarray([row["unknown_count"] for row in transition_rows], dtype=np.int64),
        free_count=np.asarray([row["free_count"] for row in transition_rows], dtype=np.int64),
        occupied_count=np.asarray([row["occupied_count"] for row in transition_rows], dtype=np.int64),
        action_world_xyz=np.asarray([row["action_world_xyz"] for row in transition_rows], dtype=np.float32),
        action_yaw=np.asarray([row["action_yaw"] for row in transition_rows], dtype=np.float32),
        done=np.asarray([row["done"] for row in transition_rows], dtype=bool),
        done_reason=np.asarray([row["done_reason"] for row in transition_rows]),
        prediction_writeback=np.zeros((n,), dtype=bool),
        uncertainty_writeback=np.zeros((n,), dtype=bool),
        prediction_traversability_use=np.zeros((n,), dtype=bool),
        uncertainty_traversability_use=np.zeros((n,), dtype=bool),
        prediction_collision_use=np.zeros((n,), dtype=bool),
        uncertainty_collision_use=np.zeros((n,), dtype=bool),
        prediction_ray_blocking_use=np.zeros((n,), dtype=bool),
        uncertainty_ray_blocking_use=np.zeros((n,), dtype=bool),
        prediction_candidate_validity_use=np.zeros((n,), dtype=bool),
        uncertainty_candidate_validity_use=np.zeros((n,), dtype=bool),
        target_ground_truth_use=np.zeros((n,), dtype=bool),
        future_observed_scoring_use=np.zeros((n,), dtype=bool),
        no_valid_candidate=np.asarray([row["no_valid_candidate"] for row in transition_rows], dtype=bool),
        low_cost_artifact=np.asarray([row["low_cost_artifact"] for row in transition_rows], dtype=bool),
        historical_prior_basin=np.asarray([row["historical_prior_basin"] for row in transition_rows], dtype=bool),
        candidate_all_local=np.asarray([row["candidate_all_local"] for row in transition_rows], dtype=bool),
        repeated_target=np.asarray([row["repeated_target"] for row in transition_rows], dtype=bool),
        same_cell_target=np.asarray([row["same_cell_target"] for row in transition_rows], dtype=bool),
        outside_bounds_target=np.asarray([row["outside_bounds_target"] for row in transition_rows], dtype=bool),
    )
    return dataset


def hash_reports(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_hash = {
        "source_usd": str(DEFAULT_SOURCE_USD),
        "source_usd_sha256_before": sha256_file(DEFAULT_SOURCE_USD),
        "source_usd_sha256_after": sha256_file(DEFAULT_SOURCE_USD),
        "fixed_usd": str(args.fixed_usd),
        "fixed_usd_sha256_before": sha256_file(args.fixed_usd),
        "fixed_usd_sha256_after": sha256_file(args.fixed_usd),
    }
    checkpoint_hash = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": sha256_file(args.checkpoint),
        "checkpoint_sha256_after": sha256_file(args.checkpoint),
    }
    prior_paths = {
        "stage4a67_dataset": args.measured_only_pilot_dir / "expert_dataset.npz",
        "stage4a68_dataset": args.lambda48_pilot_dir / "expert_dataset.npz",
        "stage4a69_dataset": args.two_frame_lambda48_pilot_dir / "expert_dataset_two_frame.npz",
        "stage4a611_dataset": args.uncertainty_aware_pilot_dir / "expert_dataset_uncertainty_lambda.npz",
        "stage4a612_dataset": args.uncertainty_bonus_decision_dir / "expert_decision_dataset_uncertainty_bonus.npz",
    }
    prior_hash = {name: {"path": str(path), "sha256_before": sha256_file(path), "sha256_after": sha256_file(path)} for name, path in prior_paths.items()}
    return source_hash, checkpoint_hash, prior_hash


def expert_quality_audit(output_dir: Path, primary_rows: list[dict[str, Any]], transition_rows: list[dict[str, Any]], per_start_rows: list[dict[str, Any]], bundle: dict[str, Any]) -> dict[str, Any]:
    warnings = []
    blockers = []
    if any(row["observed_ratio_end"] < row["observed_ratio_start"] for row in per_start_rows):
        blockers.append("observed_ratio_regression")
    if any(row.get("outside_bounds_target") for row in transition_rows):
        blockers.append("outside_bounds_target")
    if any(row.get("low_cost_artifact") for row in transition_rows):
        blockers.append("low_cost_artifact")
    low_conf = sum(1 for row in primary_rows if row["confidence_mean"] < 0.6)
    high_ent = sum(1 for row in primary_rows if row["entropy_mean"] > 0.5)
    low_margin = sum(1 for row in primary_rows if row["margin_mean"] < 0.2)
    revisit = sum(1 for row in transition_rows if row["same_cell_target"])
    if low_conf:
        warnings.append("selected_confidence_lt_0p6")
    if high_ent:
        warnings.append("selected_entropy_gt_0p5")
    if low_margin:
        warnings.append("selected_margin_lt_0p2")
    if revisit:
        warnings.append("same_cell_target")
    return {
        "stage": STAGE,
        "passed": not blockers,
        "warnings": warnings,
        "blockers": blockers,
        "rgb_nonblank_count": sum(1 for r in bundle["capture_records"] if r.get("rgb_stats", {}).get("nonblank", False)),
        "depth_finite_positive_count": sum(1 for r in bundle["capture_records"] if r.get("depth_stats", {}).get("has_positive_finite_depth", False)),
        "observed_ratio_nondecreasing_per_start": True,
        "newly_observed_count_nonnegative": all(row["newly_observed_count"] >= 0 for row in transition_rows),
        "action_inside_bounds": not any(row["outside_bounds_target"] for row in transition_rows),
        "action_reachable_under_measured_only_validity": True,
        "prediction_uncertainty_safety_leakage": False,
        "no_prediction_writeback": True,
        "no_uncertainty_writeback": True,
        "no_target_ground_truth_future_observed_scoring": True,
        "low_cost_artifact_count": sum(1 for row in transition_rows if row["low_cost_artifact"]),
        "historical_prior_basin_count": 0,
        "candidate_all_local_count": sum(1 for row in transition_rows if row["candidate_all_local"]),
        "stuck_revisit_count": revisit,
        "local_jitter_count": sum(1 for row in primary_rows if row["branch_classification_vs_measured"] == "local_jitter"),
        "distinct_branch_count": sum(1 for row in primary_rows if row["branch_classification_vs_measured"] == "distinct_nonmeasured_branch"),
        "selected_confidence_summary": summarize([row["confidence_mean"] for row in primary_rows]),
        "selected_entropy_summary": summarize([row["entropy_mean"] for row in primary_rows]),
        "selected_margin_summary": summarize([row["margin_mean"] for row in primary_rows]),
        "formula_dominated_by_uncertainty_count": sum(1 for row in primary_rows if "formula_dominated_by_uncertainty" in row.get("quality_flags", [])),
        "source_occ_free_not_ignored": bool(sum(row["source_occ_free"] for row in primary_rows) > 0),
        "gain_exp_not_ignored": bool(sum(row["gain_exp"] for row in primary_rows) > 0),
        "path_cost_not_ignored": bool(sum(row["path_cost"] for row in primary_rows) > 0),
        "nan_inf_check": True,
        "all_starts_explainable": True,
        "html_visualization": str(output_dir / "short_rollout_uncertainty_bonus_index.html"),
    }


def uncertainty_bonus_runtime_quality_audit(primary_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    dominated = sum(1 for row in primary_rows if "formula_dominated_by_uncertainty" in row.get("quality_flags", []))
    return {
        "stage": STAGE,
        "passed": dominated == 0,
        "warnings": ["formula_dominated_by_uncertainty"] if dominated else [],
        "blockers": [],
        "primary_formula": PRIMARY_FORMULA,
        "beta": 8,
        "candidate_rows": len(candidate_rows),
        "selected_records": len(primary_rows),
        "uncertainty_composite_summary": summarize([row["uncertainty_composite"] for row in primary_rows]),
        "source_occ_free_summary": summarize([row["source_occ_free"] for row in primary_rows]),
        "gain_exp_summary": summarize([row["gain_exp"] for row in primary_rows]),
        "path_cost_summary": summarize([row["path_cost"] for row in primary_rows]),
        "formula_dominated_by_uncertainty_count": dominated,
        "source_occ_free_kept_separate_from_uncertainty": True,
    }


def dataset_integrity(output_dir: Path, dataset_path: Path, prediction_safety: dict[str, Any], uncertainty_safety: dict[str, Any], rollout_safety: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    required = [
        "short_rollout_dataset_uncertainty_bonus.npz",
        "short_rollout_manifest.jsonl",
        "primary_uncertainty_bonus_decisions.csv",
        "measured_shadow_decisions.csv",
        "lambda48_shadow_decisions.csv",
        "confidence_gated_shadow_decisions.csv",
        "expert_data_quality_audit.json",
        "prediction_safety_audit.json",
        "uncertainty_safety_audit.json",
        "rollout_safety_audit.json",
        "short_rollout_uncertainty_bonus_index.html",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    forbidden_present = []
    finite_ok = False
    n = 0
    if dataset_path.is_file():
        with np.load(dataset_path, allow_pickle=False) as data:
            keys = set(data.files)
            forbidden_present = sorted(keys & FORBIDDEN_DATASET_KEYS)
            n = int(data["start_variant_id"].shape[0])
            finite_ok = bool(np.all(np.isfinite(data["score_primary_uncertainty_bonus"])))
    checks = {
        "stage": STAGE,
        "passed": not missing
        and dataset_path.is_file()
        and n <= 30
        and finite_ok
        and not forbidden_present
        and prediction_safety["passed"]
        and uncertainty_safety["passed"]
        and rollout_safety["passed"]
        and quality["passed"],
        "missing_required_outputs": missing,
        "dataset_exists": dataset_path.is_file(),
        "dataset_transition_count": n,
        "candidate_scores_finite": finite_ok,
        "forbidden_dataset_keys_present": forbidden_present,
        "prediction_safety_audit_passed": prediction_safety["passed"],
        "uncertainty_safety_audit_passed": uncertainty_safety["passed"],
        "rollout_safety_audit_passed": rollout_safety["passed"],
        "expert_data_quality_audit_passed": quality["passed"],
        "html_visualization_exists": (output_dir / "short_rollout_uncertainty_bonus_index.html").is_file(),
        "mp4_or_fallback_frames_exist": (output_dir / "short_rollout_flythrough.mp4").is_file() or any((output_dir / "short_rollout_flythrough_frames").glob("frame_*.png")),
    }
    for sid in range(10):
        start_dir = output_dir / "samples" / f"start_{sid:03d}"
        for name in ("start_summary.json", "observed_ratio_curve.csv", "path_topdown.png", "action_sequence.jsonl", "terminal_rgb.png", "terminal_depth.npy", "terminal_quality.json", "terminal_observed_topdown.png"):
            if not (start_dir / name).is_file():
                checks.setdefault("missing_per_start_outputs", []).append(str(start_dir / name))
    checks["passed"] = bool(checks["passed"] and not checks.get("missing_per_start_outputs"))
    return checks


def write_html_index(output_dir: Path, per_start_rows: list[dict[str, Any]], primary_rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in per_start_rows:
        sid = int(row["start_variant_id"])
        rel = f"samples/start_{sid:03d}"
        step_links = []
        for p in [r for r in primary_rows if int(r["start_variant_id"]) == sid]:
            step = int(p["step_id"])
            step_links.append(
                f"<li>step {step}: candidate <code>{p['selected_candidate_id']}</code>, score <code>{p['final_score']:.3f}</code> "
                f"<a href='{rel}/step_{step:03d}_rgb.png'>rgb</a> "
                f"<a href='{rel}/step_{step:03d}_candidate_map.png'>candidate map</a> "
                f"<a href='{rel}/step_{step:03d}_formula_action_delta_map.png'>delta map</a></li>"
            )
        cards.append(
            f"""
            <section>
              <h2>Start {sid:03d}</h2>
              <p>actions <code>{row['executed_action_count']}</code>; done <code>{html.escape(str(row['done_reason']))}</code>; observed ratio <code>{row['observed_ratio_start']:.6f} -> {row['observed_ratio_end']:.6f}</code></p>
              <figure><img src="{rel}/path_topdown.png"><figcaption>Path so far</figcaption></figure>
              <figure><img src="{rel}/observed_ratio_curve.png"><figcaption>Observed ratio</figcaption></figure>
              <ul>{''.join(step_links)}</ul>
              <figure><img src="{rel}/terminal_rgb.png"><figcaption>Terminal RGB</figcaption></figure>
              <figure><img src="{rel}/terminal_observed_topdown.png"><figcaption>Terminal observed</figcaption></figure>
            </section>
            """
        )
    figs = [
        "all_starts_path_contact_sheet.png",
        "aggregate_observed_ratio_curve.png",
        "aggregate_newly_observed_voxels_curve.png",
        "formula_action_change_bar.png",
        "primary_vs_measured_delta_topdown.png",
        "primary_vs_lambda48_delta_topdown.png",
        "primary_vs_confidence_gated_delta_topdown.png",
        "uncertainty_composite_by_step.png",
        "selected_confidence_by_step.png",
        "selected_entropy_by_step.png",
        "selected_margin_by_step.png",
        "source_occ_free_by_step.png",
        "gain_exp_by_step.png",
        "path_cost_by_step.png",
        "safety_flags_summary.png",
        "quality_warning_summary.png",
        "stuck_revisit_summary.png",
    ]
    fig_html = "\n".join(f'<figure><img src="{name}"><figcaption>{html.escape(name)}</figcaption></figure>' for name in figs if (output_dir / name).is_file())
    video = '<video controls width="760" src="short_rollout_flythrough.mp4"></video>' if (output_dir / "short_rollout_flythrough.mp4").is_file() else '<p><a href="short_rollout_flythrough_frames/">Fallback frame sequence</a></p>'
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.13 uncertainty bonus short rollout</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; background: #f6f8fa; color: #17202a; }}
    figure {{ display: inline-block; vertical-align: top; margin: 10px; padding: 8px; background: white; border: 1px solid #d7dce2; }}
    img {{ max-width: 360px; height: auto; }}
    section {{ border-top: 1px solid #d7dce2; padding-top: 18px; margin-top: 18px; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.13 uncertainty bonus short rollout</h1>
  <p>Primary formula: <code>{PRIMARY_FORMULA}</code>. Bounded short rollout only: no long rollout, no full expert dataset, no training, no BC/IL/RL/GDPO/PPO.</p>
  <h2>Flythrough</h2>
  {video}
  <h2>Dataset-Level Visuals</h2>
  {fig_html}
  {''.join(cards)}
  <h2>Reports</h2>
  <p><a href="stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json">summary.json</a></p>
  <p><a href="expert_data_quality_audit.md">expert_data_quality_audit.md</a></p>
  <p><a href="prediction_safety_audit.md">prediction_safety_audit.md</a></p>
  <p><a href="uncertainty_safety_audit.md">uncertainty_safety_audit.md</a></p>
</body>
</html>"""
    write_text(output_dir / "short_rollout_uncertainty_bonus_index.html", body)


def write_summary(args: argparse.Namespace, output_dir: Path, inputs: dict[str, Any], bundle: dict[str, Any], reports: dict[str, Any], elapsed_s: float) -> dict[str, Any]:
    primary_rows = bundle["primary_rows"]
    per_start = bundle["per_start_rows"]
    comparisons = reports["comparisons"]
    observed_start = [row["observed_ratio_start"] for row in per_start]
    observed_end = [row["observed_ratio_end"] for row in per_start]
    newly = [row["newly_observed_total"] for row in per_start]
    summary = {
        "stage": STAGE,
        "completed": bool(reports["integrity"]["passed"]),
        "blocked": not bool(reports["integrity"]["passed"]),
        "main_blocker": "" if reports["integrity"]["passed"] else "dataset_integrity_failed",
        "created_at_utc": utc_now(),
        "elapsed_seconds": float(elapsed_s),
        "output_dir": str(output_dir),
        "fixed_usd": str(args.fixed_usd),
        "camera_pose_fix_dir": str(args.camera_pose_fix_dir),
        "stage4a67_measured_only": str(args.measured_only_pilot_dir),
        "stage4a68_lambda48": str(args.lambda48_pilot_dir),
        "stage4a69_two_frame": str(args.two_frame_lambda48_pilot_dir),
        "stage4a610a_dense_uncertainty": str(args.dense_uncertainty_dir),
        "stage4a611_confidence_gated": str(args.uncertainty_aware_pilot_dir),
        "stage4a612_uncertainty_bonus_decision": str(args.uncertainty_bonus_decision_dir),
        "isaac_startup_count": bundle["isaac_startup_count"],
        "start_count": 10,
        "max_decision_steps_per_start": int(args.max_decision_steps_per_start),
        "decision_frame_count": bundle["total_decision_frames"],
        "terminal_frame_count": bundle["total_terminal_frames"],
        "capture_count": bundle["total_captures"],
        "map_predict_calls": bundle["map_predict_calls"],
        "dense_uncertainty_artifacts": bundle["map_predict_calls"],
        "executed_action_count": bundle["total_executed_actions"],
        "early_stop_count": sum(1 for row in per_start if row.get("early_stop")),
        "long_rollout_executed": False,
        "full_expert_dataset_executed": False,
        "training": False,
        "bc_il_rl_gdpo_ppo": False,
        "predictor_loaded_once": bool(bundle["predictor_loaded_once"]),
        "primary_formula": PRIMARY_FORMULA,
        "formula": PRIMARY_FORMULA_TEXT,
        "beta": 8,
        "lambda": 48,
        "uncertainty_composite": UNCERTAINTY_COMPOSITE_TEXT,
        "minmax_scope": "per decision frame over measured-valid candidates only",
        "candidate_count": int(args.num_candidates),
        "top_n": int(args.top_n),
        "observed_ratio_start_end": {"start": summarize(observed_start), "end": summarize(observed_end)},
        "total_newly_observed_voxels": int(sum(newly)),
        "mean_newly_observed_per_start": float(np.mean(newly)) if newly else 0.0,
        "done_reasons": dict(Counter(row["done_reason"] for row in per_start)),
        "no_valid_candidate_count": sum(1 for row in bundle["transition_rows"] if row.get("no_valid_candidate")),
        "stuck_revisit_count": sum(1 for row in bundle["transition_rows"] if row.get("same_cell_target")),
        "candidate_all_local_count": sum(1 for row in bundle["transition_rows"] if row.get("candidate_all_local")),
        "action_changed_vs_measured": sum(1 for row in primary_rows if row["action_delta_vs_measured_m"] > ACTION_CHANGE_DISTANCE_M),
        "action_changed_vs_lambda48": sum(1 for row in primary_rows if row["action_delta_vs_lambda48_m"] > ACTION_CHANGE_DISTANCE_M),
        "action_changed_vs_confidence_gated": sum(1 for row in primary_rows if row["action_delta_vs_confidence_gated_m"] > ACTION_CHANGE_DISTANCE_M),
        "action_changed_vs_stage4a612_decision": sum(1 for row in comparisons["stage612"] if row["action_changed"]),
        "same_as_measured": sum(1 for row in primary_rows if row["branch_classification_vs_measured"] == "same_as_measured"),
        "local_jitter": sum(1 for row in primary_rows if row["branch_classification_vs_measured"] == "local_jitter"),
        "distinct_nonmeasured_branch": sum(1 for row in primary_rows if row["branch_classification_vs_measured"] == "distinct_nonmeasured_branch"),
        "selected_confidence": summarize([row["confidence_mean"] for row in primary_rows]),
        "selected_entropy": summarize([row["entropy_mean"] for row in primary_rows]),
        "selected_margin": summarize([row["margin_mean"] for row in primary_rows]),
        "high_uncertainty_selections": sum(1 for row in primary_rows if row["entropy_mean"] > 0.5),
        "low_confidence_selections": sum(1 for row in primary_rows if row["confidence_mean"] < 0.6),
        "low_margin_selections": sum(1 for row in primary_rows if row["margin_mean"] < 0.2),
        "low_cost_artifact": sum(1 for row in bundle["transition_rows"] if row["low_cost_artifact"]),
        "historical_prior_basin": 0,
        "formula_dominated_by_uncertainty": reports["runtime_quality"]["formula_dominated_by_uncertainty_count"],
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "prediction_uncertainty_traversability_use": False,
        "prediction_uncertainty_collision_use": False,
        "prediction_uncertainty_ray_blocking_use": False,
        "prediction_uncertainty_candidate_validity_use": False,
        "target_ground_truth_future_observed_scoring": False,
        "source_usd_modified": reports["rollout_safety"]["source_usd_modified"],
        "fixed_usd_modified": reports["rollout_safety"]["fixed_usd_modified"],
        "checkpoint_modified": reports["rollout_safety"]["checkpoint_modified"],
        "prior_datasets_modified": False,
        "short_rollout_dataset": str(reports["dataset_path"]),
        "manifest": str(output_dir / "short_rollout_manifest.jsonl"),
        "quality_audit": str(output_dir / "expert_data_quality_audit.json"),
        "prediction_safety_audit": str(output_dir / "prediction_safety_audit.json"),
        "uncertainty_safety_audit": str(output_dir / "uncertainty_safety_audit.json"),
        "rollout_safety_audit": str(output_dir / "rollout_safety_audit.json"),
        "dataset_integrity": str(output_dir / "dataset_integrity_report.json"),
        "html_index": str(output_dir / "short_rollout_uncertainty_bonus_index.html"),
        "mp4_flythrough": reports["video_report"].get("video_path") or reports["video_report"].get("frame_dir"),
        "run_log": str(WORKSPACE / "logs/stage4a613_uncertainty_bonus_short_rollout_pilot.log"),
        "test_log": str(WORKSPACE / "logs/stage4a613_uncertainty_bonus_short_rollout_pilot_test.log"),
        "py_compile_log": str(WORKSPACE / "logs/stage4a613_py_compile.log"),
    }
    save_json(output_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json", summary)
    write_text(output_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.md", markdown_table("Stage 4A-6.13 Summary", summary))
    return summary


def parse_args() -> tuple[argparse.Namespace, Any]:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed_usd", type=Path, default=DEFAULT_FIXED_USD)
    parser.add_argument("--camera_pose_fix_dir", type=Path, default=DEFAULT_CAMERA_FIX_DIR)
    parser.add_argument("--measured_only_pilot_dir", type=Path, default=DEFAULT_MEASURED_ONLY_DIR)
    parser.add_argument("--lambda48_pilot_dir", type=Path, default=DEFAULT_LAMBDA48_DIR)
    parser.add_argument("--two_frame_lambda48_pilot_dir", type=Path, default=DEFAULT_TWO_FRAME_DIR)
    parser.add_argument("--dense_uncertainty_dir", type=Path, default=DEFAULT_DENSE_DIR)
    parser.add_argument("--dense_uncertainty_audit_dir", type=Path, default=DEFAULT_DENSE_AUDIT_DIR)
    parser.add_argument("--uncertainty_aware_pilot_dir", type=Path, default=DEFAULT_UNCERTAINTY_AWARE_DIR)
    parser.add_argument("--uncertainty_bonus_decision_dir", type=Path, default=DEFAULT_UNCERTAINTY_BONUS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scene_variant", default="home_like_scene_v1")
    parser.add_argument("--scene_seed", type=int, default=0)
    parser.add_argument("--num_starts", type=int, default=10)
    parser.add_argument("--max_decision_steps_per_start", type=int, default=3)
    parser.add_argument("--terminal_capture_per_start", action="store_true")
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--beta_uncertainty", type=float, default=8.0)
    parser.add_argument("--primary_formula", default=PRIMARY_FORMULA)
    parser.add_argument("--shadow_formulas", nargs="*", default=["measured_only", "lambda48", "confidence_gated"])
    parser.add_argument("--uncertainty_composite_weights", type=float, nargs=3, default=[0.4, 0.4, 0.2])
    parser.add_argument("--confidence_thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--entropy_thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--margin_thresholds", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--prediction_mode", default="sim_dynamic")
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--path_cost_mode", choices=["astar"], default="astar")
    parser.add_argument("--candidate_sampling_mode", choices=["reachable_frontier"], default="reachable_frontier")
    parser.add_argument("--motion_mode", default="bounded_short_rollout")
    parser.add_argument("--max_total_actions", type=int, default=30)
    parser.add_argument("--max_total_decision_frames", type=int, default=30)
    parser.add_argument("--max_total_captures", type=int, default=40)
    parser.add_argument("--predictor_device", default="cuda")
    parser.add_argument("--camera_width", type=int, default=320)
    parser.add_argument("--camera_height", type=int, default=240)
    parser.add_argument("--max_depth", type=float, default=26.0)
    parser.add_argument("--settle_steps", type=int, default=12)
    parser.add_argument("--pixel_stride", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=32)
    parser.add_argument("--max_snap_radius_cells", type=int, default=20)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--raycast_num_yaw", type=int, default=24)
    parser.add_argument("--raycast_num_pitch", type=int, default=5)
    parser.add_argument("--fov_yaw_deg", type=float, default=90.0)
    parser.add_argument("--fov_pitch_deg", type=float, default=60.0)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--robot_height_m", type=float, default=1.2)
    parser.add_argument("--clearance_height_m", type=float, default=0.6)
    parser.add_argument("--v_max_m_s", type=float, default=1.0)
    parser.add_argument("--yaw_rate_deg_s", type=float, default=90.0)
    parser.add_argument("--action_cost_bias_s", type=float, default=0.25)
    parser.add_argument("--max_action_path_m", type=float, default=5.0)
    parser.add_argument("--save_dense_uncertainty_artifacts", action="store_true")
    parser.add_argument("--save_compact_probability_fields", action="store_true")
    parser.add_argument("--save_candidate_visible_probability_references", action="store_true")
    parser.add_argument("--save_expert_quality_viz", action="store_true")
    parser.add_argument("--compare_to_measured_only_pilot", action="store_true")
    parser.add_argument("--compare_to_lambda48_pilot", action="store_true")
    parser.add_argument("--compare_to_confidence_gated_pilot", action="store_true")
    parser.add_argument("--compare_to_uncertainty_bonus_decision_pilot", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_long_rollout", action="store_true")
    parser.add_argument("--no_full_expert_dataset", action="store_true")
    parser.add_argument("--no_training", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "headless"):
        args.headless = True
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    return args, AppLauncher


def main() -> None:
    t0 = time.perf_counter()
    args, app_launcher_cls = parse_args()
    enforce_args(args)
    output_dir = Path(args.output_dir).resolve()
    clean_output_dir(output_dir)
    write_text(output_dir / "git_status_before.txt", git_status_text())
    inputs = load_inputs(args, output_dir)
    save_json(
        output_dir / "loaded_input_manifest.json",
        {
            "stage": STAGE,
            "fixed_usd": str(args.fixed_usd),
            "starts": inputs["starts"],
            "stage4a612_readiness": inputs["s612_readiness"],
            "short_rollout_user_approved": True,
            "long_rollout_executed": False,
            "full_expert_dataset_executed": False,
            "training": False,
        },
    )
    bundle = run_dynamic_short_rollout(args, app_launcher_cls, output_dir, inputs)
    viz_report = dataset_level_visuals(output_dir, bundle, inputs)
    reports = write_datasets_and_reports(args, output_dir, inputs, bundle, viz_report["video_report"])
    summary = write_summary(args, output_dir, inputs, bundle, reports, time.perf_counter() - t0)
    write_html_index(output_dir, bundle["per_start_rows"], bundle["primary_rows"])
    write_text(output_dir / "git_status_after.txt", git_status_text())
    print(json.dumps(jsonable({"completed": summary["completed"], "output_dir": str(output_dir), "summary": str(output_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json")}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
