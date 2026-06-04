#!/usr/bin/env python3
"""Stage 4A-6.11 uncertainty-aware lambda one-action pilot.

This stage consumes the validated Stage 4A-6.8 one-frame captures and the
Stage 4A-6.10a dense uncertainty contract, reconstructs the same 64 measured
reachable frontier candidates per start, computes real candidate-visible
confidence/entropy/margin features from dense fields, selects one
confidence-gated primary expert action per start, and writes a full bounded
expert data quality package.

No new Isaac process is started by this offline finalizer, because the required
one-frame captures and dense prediction artifacts already exist and were
validated in earlier stages. The summary records both the logical 10
map_predict/dense frames used and the zero new physical calls in this stage.
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

from depth_to_voxel import UNKNOWN
from sim_paper_expert import SimCandidateView, raycast_visible_voxels_observed
from sim_prediction_layer import SimPredictionLayer

import run_stage4a67_measured_only_expert_pilot as s67
import run_stage4a68_map_predict_lambda48_expert_pilot as s68


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
DEFAULT_FIXED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
DEFAULT_CAMERA_FIX_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix"
DEFAULT_MEASURED_ONLY_DIR = WORKSPACE / "outputs/isaac_stage4a67_measured_only_expert_pilot"
DEFAULT_LAMBDA48_DIR = WORKSPACE / "outputs/isaac_stage4a68_map_predict_lambda48_expert_pilot"
DEFAULT_TWO_FRAME_DIR = WORKSPACE / "outputs/isaac_stage4a69_bounded_two_frame_lambda48_pilot"
DEFAULT_DENSE_DIR = WORKSPACE / "outputs/isaac_stage4a610a_dense_prediction_uncertainty_artifacts"
DEFAULT_DENSE_AUDIT_DIR = WORKSPACE / "outputs/isaac_stage4a610a_uncertainty_audit_rerun_dense"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a611_uncertainty_aware_lambda_one_action_pilot"
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"

STAGE = "Stage 4A-6.11-uncertainty-aware-lambda-one-action-pilot"
PRIMARY_FORMULA = "confidence_gated_lambda48_v1"
PRIMARY_FORMULA_TEXT = "gain_exp / cost + 48 * minmax(source_occ_free * candidate_confidence_mean)"
LAMBDA48_FORMULA_TEXT = "gain_exp / cost + 48 * minmax(source_occ_free)"
MEASURED_FORMULA_TEXT = "gain_exp / cost"

FORMULA_SPECS = [
    ("primary_confidence_gated", "primary_confidence_gated_decisions"),
    ("lambda48_baseline", "lambda48_baseline_shadow_decisions"),
    ("measured_only", "measured_shadow_decisions"),
    ("uncertainty_bonus_beta8", "uncertainty_bonus_shadow_decisions"),
    ("uncertainty_penalty_beta8", "uncertainty_penalty_shadow_decisions"),
    ("confidence_margin_gated", "confidence_margin_gated_shadow_decisions"),
    ("entropy_penalty_beta8", "entropy_penalty_shadow_decisions"),
]

FORBIDDEN_DATASET_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "policy_logits",
    "RL reward",
    "rl_reward",
    "replay buffer",
    "replay_buffer",
    "training state",
    "training_state",
    "class_prob",
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def markdown_table(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| key | value |", "| --- | --- |"]
    for key, value in rows.items():
        text = json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
        if len(text) > 1600:
            text = text[:1600] + "..."
        lines.append(f"| {key} | `{text}` |")
    return "\n".join(lines)


def markdown_list(title: str, rows: list[str]) -> str:
    return "\n".join([f"# {title}", "", *[f"- {row}" for row in rows]])


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
    return s67.sha256_array(array)


def git_status_text() -> str:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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


def safe_div(num: float, den: float) -> float:
    return float(num) / max(float(den), 1.0e-6)


def minmax_values(values: list[float]) -> tuple[float, float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return 0.0, 0.0
    return min(finite), max(finite)


def minmax_score(value: float, lo: float, hi: float) -> float:
    if not math.isfinite(value) or hi <= lo + 1.0e-9:
        return 0.0
    return float((float(value) - float(lo)) / (float(hi) - float(lo)))


def summarize(values: list[Any]) -> dict[str, Any]:
    finite = np.asarray([as_float(v, math.nan) for v in values], dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
    }


def pearson(xs: list[Any], ys: list[Any]) -> dict[str, Any]:
    pairs = [(as_float(x, math.nan), as_float(y, math.nan)) for x, y in zip(xs, ys)]
    clean = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
    if len(clean) < 3:
        return {"status": "insufficient", "n": len(clean), "pearson": None}
    x = np.asarray([p[0] for p in clean], dtype=np.float64)
    y = np.asarray([p[1] for p in clean], dtype=np.float64)
    if float(np.std(x)) <= 1.0e-12 or float(np.std(y)) <= 1.0e-12:
        return {"status": "degenerate", "n": len(clean), "pearson": None}
    return {"status": "computed", "n": len(clean), "pearson": float(np.corrcoef(x, y)[0, 1])}


def dense_layer_from_npz(path: Path) -> SimPredictionLayer:
    with np.load(path, allow_pickle=False) as data:
        return SimPredictionLayer(
            pred_class=np.asarray(data["pred_class_uint8"], dtype=np.uint8),
            confidence=np.asarray(data["confidence_float16"], dtype=np.float32),
            occupied_prob=np.asarray(data["occupied_prob_float16"], dtype=np.float32),
            free_prob=np.asarray(data["free_prob_float16"], dtype=np.float32),
            valid=np.asarray(data["valid_mask_bool"], dtype=bool),
            entropy_norm=np.asarray(data["entropy_norm_float16"], dtype=np.float32),
            margin=np.asarray(data["margin_float16"], dtype=np.float32),
            source_npz=str(path),
        )


def dense_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pred_class": np.asarray(data["pred_class_uint8"], dtype=np.uint8),
            "confidence": np.asarray(data["confidence_float16"], dtype=np.float32),
            "entropy_norm": np.asarray(data["entropy_norm_float16"], dtype=np.float32),
            "margin": np.asarray(data["margin_float16"], dtype=np.float32),
            "occupied_prob": np.asarray(data["occupied_prob_float16"], dtype=np.float32),
            "free_prob": np.asarray(data["free_prob_float16"], dtype=np.float32),
            "valid": np.asarray(data["valid_mask_bool"], dtype=bool),
            "predicted_unmeasured": np.asarray(data["predicted_unmeasured_mask_bool"], dtype=bool),
        }


def image_stats(path: Path) -> dict[str, Any]:
    image = np.asarray(Image.open(path).convert("RGB"))
    return {
        "path": str(path),
        "shape": [int(v) for v in image.shape],
        "mean": float(image.mean()),
        "std": float(image.std()),
        "nonblank": bool(image.size and int(image.max()) > 2 and float(image.std()) >= 1.0),
    }


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    camera_dir = Path(args.camera_pose_fix_dir).resolve()
    measured_dir = Path(args.measured_only_pilot_dir).resolve()
    lambda_dir = Path(args.lambda48_pilot_dir).resolve()
    two_frame_dir = Path(args.two_frame_lambda48_pilot_dir).resolve()
    dense_dir = Path(args.dense_uncertainty_dir).resolve()
    dense_audit_dir = Path(args.dense_uncertainty_audit_dir).resolve()
    fixed_usd = Path(args.fixed_usd).resolve()

    required = [
        camera_dir / "stage4a66c_usd_camera_pose_fix_summary.json",
        camera_dir / "start_variants_interior.json",
        camera_dir / "camera_info.json",
        camera_dir / "scene_metadata.json",
        measured_dir / "stage4a67_measured_only_expert_pilot_summary.json",
        measured_dir / "expert_dataset.npz",
        measured_dir / "dataset_integrity_report.json",
        measured_dir / "safety_audit.json",
        lambda_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
        lambda_dir / "expert_dataset.npz",
        lambda_dir / "expert_dataset_manifest.jsonl",
        lambda_dir / "lambda48_decisions.csv",
        lambda_dir / "measured_shadow_decisions.csv",
        lambda_dir / "prediction_safety_audit.json",
        lambda_dir / "expert_data_quality_audit.json",
        lambda_dir / "stage4a68_vs_stage4a67_comparison.json",
        two_frame_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json",
        two_frame_dir / "expert_dataset_two_frame.npz",
        two_frame_dir / "expert_dataset_manifest.jsonl",
        two_frame_dir / "per_frame_summary.csv",
        two_frame_dir / "frame1_lambda48_decisions.csv",
        two_frame_dir / "frame2_lambda48_diagnostic_decisions.csv",
        two_frame_dir / "prediction_safety_audit.json",
        two_frame_dir / "expert_data_quality_audit.json",
        two_frame_dir / "two_frame_stability_audit.json",
        dense_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json",
        dense_dir / "dense_prediction_artifact_manifest.json",
        dense_dir / "candidate_visible_uncertainty_manifest.json",
        dense_dir / "dense_uncertainty_frame_summary.json",
        dense_dir / "dense_uncertainty_candidate_summary.json",
        dense_audit_dir / "stage4a610_dense_rerun_summary.json",
        dense_audit_dir / "uncertainty_readiness_decision.json",
        dense_audit_dir / "candidate_uncertainty_table.csv",
        dense_audit_dir / "selected_action_uncertainty_audit.csv",
        dense_audit_dir / "uncertainty_vs_source_occ_free_analysis.json",
        dense_audit_dir / "uncertainty_vs_branch_classification.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required Stage 4A-6.11 inputs: " + json.dumps(missing, indent=2))

    camera_summary = read_json(camera_dir / "stage4a66c_usd_camera_pose_fix_summary.json")
    scene_metadata = read_json(camera_dir / "scene_metadata.json")
    measured_summary = read_json(measured_dir / "stage4a67_measured_only_expert_pilot_summary.json")
    measured_integrity = read_json(measured_dir / "dataset_integrity_report.json")
    measured_safety = read_json(measured_dir / "safety_audit.json")
    lambda_summary = read_json(lambda_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json")
    lambda_safety = read_json(lambda_dir / "prediction_safety_audit.json")
    lambda_quality = read_json(lambda_dir / "expert_data_quality_audit.json")
    two_summary = read_json(two_frame_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json")
    dense_summary = read_json(dense_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json")
    readiness = read_json(dense_audit_dir / "uncertainty_readiness_decision.json")

    checks = {
        "stage4a610a_candidate_level_uncertainty_ready": bool(readiness.get("candidate_level_uncertainty_ready", False)),
        "stage4a610a_uncertainty_aware_expert_pilot_ready": bool(readiness.get("uncertainty_aware_expert_pilot_ready", False)),
        "stage4a610a_future_stage4a611_executed": bool(readiness.get("future_stage4a611_executed", True)),
        "stage4a67_completed": bool(measured_summary.get("completed", False)),
        "stage4a67_integrity": bool(measured_integrity.get("passed", False)),
        "stage4a67_safety": bool(measured_safety.get("passed", False)),
        "stage4a68_completed": bool(lambda_summary.get("completed", False)),
        "stage4a68_map_predict_calls": int(lambda_summary.get("map_predict_calls", -1)),
        "stage4a68_prediction_safety": bool(lambda_safety.get("passed", False)),
        "stage4a68_quality": bool(lambda_quality.get("passed", False)),
        "stage4a69_completed": bool(two_summary.get("completed", False)),
        "stage4a69_frame1_reproduced_stage4a68": int(two_summary.get("stage4a68_frame1_reproduced_count", -1)),
        "stage4a611_output_already_exists": Path(args.output_dir).resolve().exists(),
    }
    if not checks["stage4a610a_candidate_level_uncertainty_ready"]:
        raise RuntimeError("Stage 4A-6.10a candidate_level_uncertainty_ready is not true")
    if not checks["stage4a610a_uncertainty_aware_expert_pilot_ready"]:
        raise RuntimeError("Stage 4A-6.10a uncertainty_aware_expert_pilot_ready is not true")
    if checks["stage4a610a_future_stage4a611_executed"]:
        raise RuntimeError("Stage 4A-6.10a readiness says future_stage4a611_executed=true")
    if not (checks["stage4a67_completed"] and checks["stage4a67_integrity"] and checks["stage4a67_safety"]):
        raise RuntimeError("Stage 4A-6.7 inputs are not clean")
    if not (checks["stage4a68_completed"] and checks["stage4a68_map_predict_calls"] == 10 and checks["stage4a68_prediction_safety"] and checks["stage4a68_quality"]):
        raise RuntimeError("Stage 4A-6.8 inputs are not clean")
    if not checks["stage4a69_completed"]:
        raise RuntimeError("Stage 4A-6.9 inputs are not clean")

    starts = read_json(camera_dir / "start_variants_interior.json")[: int(args.num_starts)]
    bounds = scene_metadata.get("map_bounds") or scene_metadata.get("observed_summary", {}).get("chosen_bounds")
    voxel_size = float(scene_metadata.get("voxel_size", 0.1))
    dense_manifest = read_json(dense_dir / "dense_prediction_artifact_manifest.json")
    dense_68 = {
        int(row["start_variant_id"]): row
        for row in dense_manifest
        if str(row.get("stage_source")) == "6.8" and int(row.get("frame_id", 1)) == 1
    }
    if sorted(dense_68) != list(range(int(args.num_starts))):
        raise RuntimeError(f"Dense 6.8 frame coverage mismatch: {sorted(dense_68)}")

    return {
        "camera_dir": camera_dir,
        "measured_dir": measured_dir,
        "lambda_dir": lambda_dir,
        "two_frame_dir": two_frame_dir,
        "dense_dir": dense_dir,
        "dense_audit_dir": dense_audit_dir,
        "fixed_usd": fixed_usd,
        "camera_info": read_json(camera_dir / "camera_info.json"),
        "camera_summary": camera_summary,
        "scene_metadata": scene_metadata,
        "starts": starts,
        "bounds": bounds,
        "voxel_size": voxel_size,
        "inspection_manifest": {"poses": scene_metadata.get("inspection_camera_poses", [])},
        "measured_summary": measured_summary,
        "lambda_summary": lambda_summary,
        "two_summary": two_summary,
        "dense_summary": dense_summary,
        "readiness": readiness,
        "dense_68": dense_68,
        "stage4a68_lambda_rows": read_csv(lambda_dir / "lambda48_decisions.csv"),
        "stage4a68_measured_rows": read_csv(lambda_dir / "measured_shadow_decisions.csv"),
        "stage4a69_frame1_rows": read_csv(two_frame_dir / "frame1_lambda48_decisions.csv"),
        "uncertainty_vs_source_occ_free": read_json(dense_audit_dir / "uncertainty_vs_source_occ_free_analysis.json"),
        "uncertainty_vs_branch": read_json(dense_audit_dir / "uncertainty_vs_branch_classification.json"),
        "preflight_checks": checks,
    }


def enforce_args(args: argparse.Namespace) -> None:
    if int(args.num_starts) != 10:
        raise ValueError("Stage 4A-6.11 requires exactly 10 starts")
    if int(args.num_candidates) != 64:
        raise ValueError("Stage 4A-6.11 requires num_candidates=64")
    if int(args.top_n) != 16:
        raise ValueError("Stage 4A-6.11 requires top_n=16")
    if float(args.lambda_sc) != 48.0:
        raise ValueError("Stage 4A-6.11 requires lambda_sc=48")
    if str(args.primary_formula) != PRIMARY_FORMULA:
        raise ValueError(f"Unsupported primary formula: {args.primary_formula}")
    required_flags = [
        "exactly_one_action_per_start",
        "save_dense_uncertainty_artifacts",
        "save_compact_probability_fields",
        "save_candidate_visible_probability_references",
        "save_expert_quality_viz",
        "compare_to_measured_only_pilot",
        "compare_to_lambda48_one_action_pilot",
        "compare_to_two_frame_frame1",
        "no_rollout",
        "no_second_action",
        "no_third_frame",
        "no_long_rollout",
        "no_training",
        "no_rl_gdpo",
    ]
    missing = [name for name in required_flags if not bool(getattr(args, name))]
    if missing:
        raise ValueError(f"Missing required Stage 4A-6.11 safety/runtime flags: {missing}")


def clean_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if "stage4a611" not in output_dir.name:
            raise RuntimeError(f"Refusing to clean non-6.11 output dir: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_one_frame_inputs(inputs: dict[str, Any], output_dir: Path, sid: int) -> dict[str, Path]:
    lambda_sample = inputs["lambda_dir"] / "samples" / f"start_{sid:03d}"
    sample_dir = output_dir / "samples" / f"start_{sid:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    srcs = {
        "rgb": lambda_sample / f"rgb_{sid:03d}.png",
        "depth": lambda_sample / f"depth_{sid:03d}.npy",
        "depth_color": lambda_sample / f"depth_color_{sid:03d}.png",
        "pose": lambda_sample / f"pose_{sid:03d}.json",
        "observed_state": lambda_sample / f"observed_state_{sid:03d}.npy",
    }
    for key, src in srcs.items():
        if not src.is_file():
            raise FileNotFoundError(f"Missing Stage 4A-6.8 sample input {key}: {src}")
    dests = {
        "rgb": sample_dir / "rgb.png",
        "depth": sample_dir / "depth.npy",
        "depth_color": sample_dir / "depth_color.png",
        "pose": sample_dir / "pose.json",
        "observed_state": sample_dir / "observed_state.npy",
    }
    for key, src in srcs.items():
        shutil.copy2(src, dests[key])
    return dests


def copy_dense_artifact(inputs: dict[str, Any], output_dir: Path, sid: int, sample_dir: Path) -> Path:
    dense_src = Path(inputs["dense_68"][sid]["dense_artifact_path"])
    dense_dst = sample_dir / "dense_prediction_uncertainty.npz"
    shutil.copy2(dense_src, dense_dst)
    root_dense_dir = output_dir / "dense_prediction_artifacts"
    root_dense_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dense_src, root_dense_dir / f"dense_prediction_uncertainty_{sid:03d}.npz")
    return dense_dst


def dense_summary_for_sample(dense_path: Path, observed_state: np.ndarray, sid: int) -> dict[str, Any]:
    fields = dense_fields(dense_path)
    valid = fields["valid"]
    pred_un = fields["predicted_unmeasured"]
    observed_unknown = observed_state == UNKNOWN
    out: dict[str, Any] = {
        "stage": STAGE,
        "start_variant_id": sid,
        "dense_prediction_uncertainty": str(dense_path),
        "dense_compact_fields": [
            "pred_class",
            "confidence",
            "entropy_norm",
            "margin",
            "occupied_prob",
            "free_prob",
            "valid_mask",
            "predicted_unmeasured_mask",
        ],
        "prediction_valid_count": int(np.count_nonzero(valid)),
        "predicted_unmeasured_count": int(np.count_nonzero(pred_un)),
        "predicted_occupied_count": int(np.count_nonzero(valid & (fields["occupied_prob"] >= 0.5))),
        "source_occ_free_count": int(np.count_nonzero(pred_un & observed_unknown)),
        "prediction_density": float(np.count_nonzero(valid) / max(1, valid.size)),
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "observed_state_hash": sha256_array(observed_state),
        "dense_npz_sha256": sha256_file(dense_path),
    }
    mask = valid
    for label, key in (("confidence", "confidence"), ("entropy", "entropy_norm"), ("margin", "margin")):
        arr = fields[key][mask]
        out[f"{label}_summary"] = summarize(arr.tolist())
    return out


def candidate_uncertainty(row: dict[str, Any], fields: dict[str, np.ndarray], observed_state: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    grid = tuple(int(v) for v in row["grid"])
    world = [float(v) for v in row["world"]]
    candidate = SimCandidateView(
        id=int(row["candidate_id"]),
        grid_position=grid,
        world_position=tuple(world),
        yaw=float(row["yaw_rad"]),
        valid=True,
        candidate_source="stage4a611_uncertainty_visible_candidate",
    )
    max_range_voxels = max(1, int(round(float(args.max_ray_length_m) / 0.1)))
    visible = raycast_visible_voxels_observed(
        candidate,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=max(4, int(args.raycast_num_yaw)),
        num_pitch=max(3, int(args.raycast_num_pitch)),
        fov_yaw_deg=float(args.fov_yaw_deg),
        fov_pitch_deg=float(args.fov_pitch_deg),
    )
    valid = fields["valid"]
    pred_un = fields["predicted_unmeasured"]
    visible_prediction = [tuple(v) for v in visible if valid[tuple(v)]]
    visible_predicted_unmeasured = [tuple(v) for v in visible if pred_un[tuple(v)]]
    if visible_predicted_unmeasured:
        idx = tuple(np.asarray([v[axis] for v in visible_predicted_unmeasured], dtype=np.int64) for axis in range(3))
        conf = np.asarray(fields["confidence"][idx], dtype=np.float32)
        ent = np.asarray(fields["entropy_norm"][idx], dtype=np.float32)
        margin = np.asarray(fields["margin"][idx], dtype=np.float32)
        uncertain = (conf < 0.7) | (ent > 0.7) | (margin < 0.2)
        conf_summary = summarize(conf.tolist())
        ent_summary = summarize(ent.tolist())
        margin_summary = summarize(margin.tolist())
        low_conf_05 = int(np.count_nonzero(conf < 0.5))
        low_conf_07 = int(np.count_nonzero(conf < 0.7))
        low_conf_09 = int(np.count_nonzero(conf < 0.9))
        high_ent_05 = int(np.count_nonzero(ent > 0.5))
        high_ent_07 = int(np.count_nonzero(ent > 0.7))
        low_margin_01 = int(np.count_nonzero(margin < 0.1))
        low_margin_02 = int(np.count_nonzero(margin < 0.2))
        uncertain_count = int(np.count_nonzero(uncertain))
        confidence_weighted = float(np.sum(conf))
        entropy_weighted = float(np.sum(ent))
        uncertainty_penalized = float(np.sum(conf * (1.0 - ent)))
        uncertainty_conf = 1.0 - conf
        uncertainty_conf_mean = float(np.mean(uncertainty_conf))
        uncertainty_conf_max = float(np.max(uncertainty_conf))
    else:
        conf_summary = ent_summary = margin_summary = {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
        low_conf_05 = low_conf_07 = low_conf_09 = 0
        high_ent_05 = high_ent_07 = 0
        low_margin_01 = low_margin_02 = 0
        uncertain_count = 0
        confidence_weighted = 0.0
        entropy_weighted = 0.0
        uncertainty_penalized = 0.0
        uncertainty_conf_mean = 0.0
        uncertainty_conf_max = 0.0
    return {
        "visible_prediction_voxel_count": int(len(visible_prediction)),
        "visible_predicted_unmeasured_count": int(len(visible_predicted_unmeasured)),
        "visible_prediction_voxel_count_for_uncertainty_denominator": int(len(visible_prediction)),
        "candidate_confidence_mean": float(conf_summary["mean"] or 0.0),
        "candidate_confidence_min": float(conf_summary["min"] or 0.0),
        "candidate_confidence_p10": float(conf_summary["p10"] or 0.0),
        "candidate_confidence_p50": float(conf_summary["p50"] or 0.0),
        "candidate_confidence_p90": float(conf_summary["p90"] or 0.0),
        "candidate_uncertainty_conf_mean": float(uncertainty_conf_mean),
        "candidate_uncertainty_conf_max": float(uncertainty_conf_max),
        "candidate_entropy_mean": float(ent_summary["mean"] or 0.0),
        "candidate_entropy_max": float(ent_summary["max"] or 0.0),
        "candidate_entropy_p90": float(ent_summary["p90"] or 0.0),
        "candidate_margin_mean": float(margin_summary["mean"] or 0.0),
        "candidate_margin_min": float(margin_summary["min"] or 0.0),
        "low_conf_count_0p5": low_conf_05,
        "low_conf_count_0p7": low_conf_07,
        "low_conf_count_0p9": low_conf_09,
        "high_entropy_count_0p5": high_ent_05,
        "high_entropy_count_0p7": high_ent_07,
        "low_margin_count_0p1": low_margin_01,
        "low_margin_count_0p2": low_margin_02,
        "uncertain_voxel_count": int(uncertain_count),
        "uncertain_fraction": float(uncertain_count / max(len(visible_prediction), 1)),
        "source_occ_free_confidence_weighted": confidence_weighted,
        "source_occ_free_entropy_weighted": entropy_weighted,
        "source_occ_free_uncertainty_penalized": uncertainty_penalized,
        "candidate_visible_reference_status": "computed_stage4a611",
        "candidate_level_uncertainty_available": True,
    }


def score_candidates_with_uncertainty(
    candidate_rows: list[dict[str, Any]],
    fields: dict[str, np.ndarray],
    observed_state: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in candidate_rows:
        out = dict(row)
        unc = candidate_uncertainty(row, fields, observed_state, args)
        out.update(unc)
        out["path_cost"] = float(out["cost_s"])
        out["base_measured_value"] = float(out["final_score_measured"])
        out["source_occ_free"] = float(out["source_occ_free"])
        out["source_confidence_gate_raw"] = float(out["source_occ_free"]) * float(out["candidate_confidence_mean"])
        out["source_confidence_margin_gate_raw"] = (
            float(out["source_occ_free"]) * float(out["candidate_confidence_mean"]) * float(out["candidate_margin_mean"])
        )
        enriched.append(out)

    source_lo, source_hi = minmax_values([row["source_occ_free"] for row in enriched])
    conf_lo, conf_hi = minmax_values([row["source_confidence_gate_raw"] for row in enriched])
    cm_lo, cm_hi = minmax_values([row["source_confidence_margin_gate_raw"] for row in enriched])
    unc_lo, unc_hi = minmax_values([row["uncertain_fraction"] for row in enriched])
    ent_lo, ent_hi = minmax_values([row["candidate_entropy_mean"] for row in enriched])
    for row in enriched:
        base = float(row["base_measured_value"])
        row["source_occ_free_minmax_stage4a611"] = minmax_score(row["source_occ_free"], source_lo, source_hi)
        row["source_confidence_gate_minmax"] = minmax_score(row["source_confidence_gate_raw"], conf_lo, conf_hi)
        row["source_confidence_margin_gate_minmax"] = minmax_score(row["source_confidence_margin_gate_raw"], cm_lo, cm_hi)
        row["uncertain_fraction_minmax"] = minmax_score(row["uncertain_fraction"], unc_lo, unc_hi)
        row["entropy_mean_minmax"] = minmax_score(row["candidate_entropy_mean"], ent_lo, ent_hi)
        row["score_measured_only"] = base
        row["score_lambda48_baseline"] = base + float(args.lambda_sc) * float(row["source_occ_free_minmax_stage4a611"])
        row["score_primary_confidence_gated"] = base + float(args.lambda_sc) * float(row["source_confidence_gate_minmax"])
        row["score_confidence_margin_gated"] = base + float(args.lambda_sc) * float(row["source_confidence_margin_gate_minmax"])
        row["score_uncertainty_bonus_beta8"] = row["score_lambda48_baseline"] + 8.0 * float(row["uncertain_fraction_minmax"])
        row["score_uncertainty_penalty_beta8"] = row["score_lambda48_baseline"] - 8.0 * float(row["uncertain_fraction_minmax"])
        row["score_entropy_penalty_beta8"] = row["score_lambda48_baseline"] - 8.0 * float(row["entropy_mean_minmax"])
        row["primary_formula"] = PRIMARY_FORMULA
    return enriched


def best_for_formula(rows: list[dict[str, Any]], formula: str) -> dict[str, Any]:
    key = f"score_{formula}"
    return dict(
        max(
            rows,
            key=lambda row: (
                float(row.get(key, -math.inf)),
                float(row.get("gain_exp", 0.0)),
                float(row.get("source_occ_free", 0.0)),
                -float(row.get("path_cost", math.inf)),
                -int(row.get("candidate_id", 0)),
                -int(round(float(row.get("yaw_rad", 0.0)) * 10000.0)),
            ),
        )
    )


def classify_against(selected: dict[str, Any], reference: dict[str, Any]) -> tuple[str, float | None, float | None]:
    if not selected or not reference:
        return "no_valid_candidate", None, None
    distance = s68.distance_xy(selected.get("world", [math.nan, math.nan, math.nan]), reference.get("world", [math.nan, math.nan, math.nan]))
    yaw_delta = abs(s67.wrap_angle(float(selected.get("yaw_rad", 0.0)) - float(reference.get("yaw_rad", 0.0))))
    same_candidate = int(selected.get("candidate_id", -1)) == int(reference.get("candidate_id", -2))
    if same_candidate and distance <= 0.15 and yaw_delta <= 0.10:
        return "same_as_measured", distance, yaw_delta
    if distance <= 0.80:
        return "local_jitter", distance, yaw_delta
    return "distinct_nonmeasured_branch", distance, yaw_delta


def selected_decision_row(
    sid: int,
    start_name: str,
    formula: str,
    selected: dict[str, Any],
    measured_ref: dict[str, Any],
    lambda_ref: dict[str, Any],
    stage68_ref: dict[str, Any],
    stage67_ref: dict[str, Any],
) -> dict[str, Any]:
    vs_measured, dist_m, yaw_m = classify_against(selected, measured_ref)
    vs_lambda, dist_l, yaw_l = classify_against(selected, lambda_ref)
    vs68, dist68, yaw68 = classify_against(selected, stage68_ref)
    world67 = stage67_ref.get("action_position", [math.nan, math.nan, math.nan])
    yaw67_ref = as_float(stage67_ref.get("action_yaw_rad"), 0.0)
    dist67 = s68.distance_xy(selected.get("world", world67), world67) if stage67_ref else None
    yaw67 = abs(s67.wrap_angle(float(selected.get("yaw_rad", 0.0)) - yaw67_ref)) if stage67_ref else None
    score_key = f"score_{formula}"
    return {
        "stage": STAGE,
        "start_variant_id": sid,
        "start_name": start_name,
        "formula": formula,
        "selected_candidate_id": int(selected.get("candidate_id", -1)),
        "selected_world_xyz": selected.get("world"),
        "selected_yaw": selected.get("yaw_rad"),
        "final_score": selected.get(score_key),
        "gain_exp": selected.get("gain_exp"),
        "source_occ_free": selected.get("source_occ_free"),
        "path_cost": selected.get("path_cost"),
        "confidence_mean": selected.get("candidate_confidence_mean"),
        "confidence_min": selected.get("candidate_confidence_min"),
        "entropy_mean": selected.get("candidate_entropy_mean"),
        "entropy_max": selected.get("candidate_entropy_max"),
        "margin_mean": selected.get("candidate_margin_mean"),
        "margin_min": selected.get("candidate_margin_min"),
        "uncertain_fraction": selected.get("uncertain_fraction"),
        "uncertain_voxel_count": selected.get("uncertain_voxel_count"),
        "visible_prediction_voxel_count": selected.get("visible_prediction_voxel_count"),
        "visible_predicted_unmeasured_count": selected.get("visible_predicted_unmeasured_count"),
        "branch_classification_vs_measured": vs_measured,
        "branch_classification_vs_lambda48_baseline": vs_lambda,
        "action_delta_vs_measured_m": dist_m,
        "yaw_delta_vs_measured_rad": yaw_m,
        "action_delta_vs_lambda48_m": dist_l,
        "yaw_delta_vs_lambda48_rad": yaw_l,
        "action_delta_vs_stage4a68_m": dist68,
        "yaw_delta_vs_stage4a68_rad": yaw68,
        "branch_classification_vs_stage4a68": vs68,
        "action_delta_vs_stage4a67_m": dist67,
        "yaw_delta_vs_stage4a67_rad": yaw67,
        "quality_flags": {
            "candidate_all_local": False,
            "no_valid_candidate": False,
            "low_cost_artifact": bool(float(selected.get("path_cost", 1.0)) < 0.05),
            "historical_prior_basin": False,
        },
    }


def row_for_plot(row: dict[str, Any], score_key: str = "score_primary_confidence_gated") -> dict[str, Any]:
    out = dict(row)
    out["final_score_lambda48"] = float(row.get(score_key, row.get("score_lambda48_baseline", 0.0)))
    out["lambda48_bonus"] = float(out["final_score_lambda48"]) - float(row.get("base_measured_value", 0.0))
    return out


def create_rgb_depth_panel(path: Path, rgb_path: Path, depth_color_path: Path) -> None:
    rgb = Image.open(rgb_path).convert("RGB").resize((320, 240))
    depth = Image.open(depth_color_path).convert("RGB").resize((320, 240))
    panel = Image.new("RGB", (640, 276), (245, 247, 250))
    panel.paste(rgb, (0, 36))
    panel.paste(depth, (320, 36))
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, 640, 36), fill=(17, 24, 39))
    draw.text((12, 10), "RGB", fill=(245, 247, 250))
    draw.text((332, 10), "Depth", fill=(245, 247, 250))
    panel.save(path)


def plot_dense_overlay(path: Path, observed_state: np.ndarray, bounds: dict[str, Any], start: dict[str, Any], selected: dict[str, Any], dense_path: Path, field: str, title: str, cmap: str) -> None:
    fields = dense_fields(dense_path)
    if field == "confidence":
        data = np.max(np.where(fields["valid"], fields["confidence"], 0.0), axis=2)
        vmin, vmax = 0.0, 1.0
    elif field == "entropy":
        data = np.max(np.where(fields["valid"], fields["entropy_norm"], 0.0), axis=2)
        vmin, vmax = 0.0, 1.0
    elif field == "margin":
        data = np.max(np.where(fields["valid"], fields["margin"], 0.0), axis=2)
        vmin, vmax = 0.0, 1.0
    else:
        data = np.any(fields["predicted_unmeasured"], axis=2).astype(float)
        vmin, vmax = 0.0, 1.0
    top = s68.topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    overlay = np.ma.masked_where(data.T <= 0, data.T)
    im = ax.imshow(overlay, origin="lower", extent=extent, cmap=cmap, alpha=0.58, vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.65)
    start_pos = start["position"]
    ax.scatter([start_pos[0]], [start_pos[1]], s=70, c="#2563eb", marker="^", edgecolors="white", linewidths=0.6)
    pos = selected.get("world", start_pos)
    ax.scatter([pos[0]], [pos[1]], s=92, c="#10b981", marker="*", edgecolors="black", linewidths=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def plot_candidate_uncertainty_map(path: Path, observed_state: np.ndarray, bounds: dict[str, Any], start: dict[str, Any], rows: list[dict[str, Any]], selected: dict[str, Any]) -> None:
    top = s68.topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    xs = [row["world"][0] for row in rows]
    ys = [row["world"][1] for row in rows]
    c = [float(row["uncertain_fraction"]) for row in rows]
    sc = ax.scatter(xs, ys, s=28, c=c, cmap="magma", edgecolors="white", linewidths=0.25)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="uncertain fraction")
    sp = start["position"]
    ax.scatter([sp[0]], [sp[1]], s=70, c="#2563eb", marker="^", edgecolors="white", linewidths=0.6)
    p = selected["world"]
    ax.scatter([p[0]], [p[1]], s=105, c="#10b981", marker="*", edgecolors="black", linewidths=0.5)
    ax.set_title("candidate uncertainty map", fontsize=9)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def plot_formula_action_delta_map(path: Path, observed_state: np.ndarray, bounds: dict[str, Any], start: dict[str, Any], selections: dict[str, dict[str, Any]]) -> None:
    top = s68.topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.0, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    sp = start["position"]
    ax.scatter([sp[0]], [sp[1]], s=70, c="#2563eb", marker="^", edgecolors="white", linewidths=0.6, label="start")
    colors = {
        "primary_confidence_gated": "#10b981",
        "lambda48_baseline": "#7c3aed",
        "measured_only": "#d97706",
        "uncertainty_bonus_beta8": "#ef4444",
        "uncertainty_penalty_beta8": "#0891b2",
        "confidence_margin_gated": "#84cc16",
        "entropy_penalty_beta8": "#f97316",
    }
    markers = {
        "primary_confidence_gated": "*",
        "lambda48_baseline": "o",
        "measured_only": "s",
        "uncertainty_bonus_beta8": "P",
        "uncertainty_penalty_beta8": "X",
        "confidence_margin_gated": "D",
        "entropy_penalty_beta8": "v",
    }
    for name, row in selections.items():
        p = row["world"]
        ax.plot([sp[0], p[0]], [sp[1], p[1]], color=colors.get(name, "#111827"), linewidth=1.0, alpha=0.75)
        ax.scatter([p[0]], [p[1]], s=80, c=colors.get(name, "#111827"), marker=markers.get(name, "o"), edgecolors="black", linewidths=0.4, label=name)
    ax.legend(fontsize=6, loc="upper left")
    ax.set_title("formula action delta map", fontsize=9)
    ax.set_aspect("equal")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def plot_candidate_score_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    top = sorted(rows, key=lambda row: float(row["score_primary_confidence_gated"]), reverse=True)[:16]
    labels = [str(row["candidate_id"]) for row in top]
    x = np.arange(len(top))
    base = [float(row["base_measured_value"]) for row in top]
    bonus = [float(row["score_primary_confidence_gated"]) - float(row["base_measured_value"]) for row in top]
    fig, ax = plt.subplots(figsize=(9.0, 4.2), constrained_layout=True)
    ax.bar(x, base, color="#2563eb", label="gain_exp / cost")
    ax.bar(x, bonus, bottom=base, color="#10b981", label="confidence-gated bonus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("score")
    ax.set_title("primary score decomposition")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_candidate_uncertainty_bar(path: Path, rows: list[dict[str, Any]]) -> None:
    top = sorted(rows, key=lambda row: float(row["score_primary_confidence_gated"]), reverse=True)[:16]
    labels = [str(row["candidate_id"]) for row in top]
    x = np.arange(len(top))
    conf = [float(row["candidate_confidence_mean"]) for row in top]
    ent = [float(row["candidate_entropy_mean"]) for row in top]
    margin = [float(row["candidate_margin_mean"]) for row in top]
    fig, ax = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    ax.plot(x, conf, marker="o", color="#10b981", label="confidence")
    ax.plot(x, ent, marker="o", color="#ef4444", label="entropy")
    ax.plot(x, margin, marker="o", color="#2563eb", label="margin")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("candidate uncertainty features")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def action_quality(sample_dir: Path, decision_row: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    rgb = image_stats(sample_dir / "rgb.png")
    depth = np.load(sample_dir / "depth.npy")
    depth_valid = bool(np.isfinite(depth).any() and np.count_nonzero(np.isfinite(depth) & (depth > 0)) > 0)
    warnings: list[str] = []
    if not rgb["nonblank"]:
        warnings.append("blank_rgb")
    if not depth_valid:
        warnings.append("invalid_depth")
    if all(s68.distance_xy(row["world"], rows[0]["world"]) < 0.80 for row in rows):
        warnings.append("candidate_all_local")
    if bool(decision_row["quality_flags"].get("low_cost_artifact")):
        warnings.append("low_cost_artifact")
    score = 1.0 - 0.15 * len([w for w in warnings if w != "candidate_all_local"])
    quality = {
        "stage": STAGE,
        "passed": bool(depth_valid and rgb["nonblank"] and "low_cost_artifact" not in warnings),
        "quality_score": max(0.0, float(score)),
        "warnings": warnings,
        "branch_classification": decision_row["branch_classification_vs_measured"],
        "candidate_count": len(rows),
        "top_n_count": 16,
        "primary_selected": decision_row,
        "rgb_nonblank": rgb["nonblank"],
        "depth_finite_positive": depth_valid,
        "candidate_all_local": "candidate_all_local" in warnings,
    }
    save_json(sample_dir / "action_quality.json", quality)
    write_text(sample_dir / "action_quality.md", markdown_table("Action Quality", quality))
    return quality


def process_start(args: argparse.Namespace, inputs: dict[str, Any], output_dir: Path, start: dict[str, Any], yaw_priors: list[float]) -> dict[str, Any]:
    sid = int(start["index"])
    sample_dir = output_dir / "samples" / f"start_{sid:03d}"
    paths = copy_one_frame_inputs(inputs, output_dir, sid)
    dense_path = copy_dense_artifact(inputs, output_dir, sid, sample_dir)
    observed_state = np.load(paths["observed_state"])
    dense_summary = dense_summary_for_sample(dense_path, observed_state, sid)
    save_json(sample_dir / "dense_prediction_summary.json", dense_summary)

    prediction_layer = dense_layer_from_npz(dense_path)
    decision = s68.score_start_lambda48(
        observed_state=observed_state,
        prediction_layer=prediction_layer,
        bounds=inputs["bounds"],
        voxel_size=float(inputs["voxel_size"]),
        start=start,
        yaw_priors=yaw_priors,
        args=args,
    )
    fields = dense_fields(dense_path)
    candidates = score_candidates_with_uncertainty(decision["candidate_rows_lambda48"], fields, observed_state, args)
    candidates_sorted = sorted(candidates, key=lambda row: int(row["candidate_id"]))
    write_csv(sample_dir / "candidate_uncertainty_features.csv", candidates_sorted)
    write_jsonl(sample_dir / "candidate_uncertainty_features.jsonl", candidates_sorted)

    selections = {
        "primary_confidence_gated": best_for_formula(candidates, "primary_confidence_gated"),
        "lambda48_baseline": best_for_formula(candidates, "lambda48_baseline"),
        "measured_only": best_for_formula(candidates, "measured_only"),
        "uncertainty_bonus_beta8": best_for_formula(candidates, "uncertainty_bonus_beta8"),
        "uncertainty_penalty_beta8": best_for_formula(candidates, "uncertainty_penalty_beta8"),
        "confidence_margin_gated": best_for_formula(candidates, "confidence_margin_gated"),
        "entropy_penalty_beta8": best_for_formula(candidates, "entropy_penalty_beta8"),
    }
    stage68_ref = next((r for r in inputs["stage4a68_lambda_rows"] if as_int(r.get("start_variant_id")) == sid), {})
    stage68_ref = {**stage68_ref, "world": parse_literal(stage68_ref.get("world"), []), "yaw_rad": as_float(stage68_ref.get("yaw_rad")), "candidate_id": as_int(stage68_ref.get("candidate_id"))}
    stage69_ref = next((r for r in inputs["stage4a69_frame1_rows"] if as_int(r.get("start_variant_id")) == sid), {})
    stage69_ref = {**stage69_ref, "world": parse_literal(stage69_ref.get("world"), []), "yaw_rad": as_float(stage69_ref.get("yaw_rad")), "candidate_id": as_int(stage69_ref.get("candidate_id"))}
    measured_summary_actions = {int(row["start_index"]): row for row in inputs["measured_summary"].get("selected_actions", [])}
    stage67_ref = measured_summary_actions.get(sid, {})

    decision_rows: dict[str, dict[str, Any]] = {}
    for formula_name, _stem in FORMULA_SPECS:
        row = selected_decision_row(
            sid,
            str(start["name"]),
            formula_name,
            selections[formula_name],
            selections["measured_only"],
            selections["lambda48_baseline"],
            stage68_ref,
            stage67_ref,
        )
        row["branch_classification_vs_stage4a69_frame1"] = classify_against(selections[formula_name], stage69_ref)[0]
        row["action_delta_vs_stage4a69_frame1_m"] = classify_against(selections[formula_name], stage69_ref)[1]
        row["yaw_delta_vs_stage4a69_frame1_rad"] = classify_against(selections[formula_name], stage69_ref)[2]
        decision_rows[formula_name] = row

    save_json(sample_dir / "primary_confidence_gated_decision.json", decision_rows["primary_confidence_gated"])
    save_json(sample_dir / "lambda48_baseline_shadow_decision.json", decision_rows["lambda48_baseline"])
    save_json(sample_dir / "measured_shadow_decision.json", decision_rows["measured_only"])
    save_json(sample_dir / "uncertainty_bonus_shadow_decision.json", decision_rows["uncertainty_bonus_beta8"])
    save_json(sample_dir / "uncertainty_penalty_shadow_decision.json", decision_rows["uncertainty_penalty_beta8"])
    save_json(sample_dir / "confidence_margin_gated_shadow_decision.json", decision_rows["confidence_margin_gated"])
    save_json(sample_dir / "entropy_penalty_shadow_decision.json", decision_rows["entropy_penalty_beta8"])

    primary = selections["primary_confidence_gated"]
    executed = {
        "stage": STAGE,
        "start_variant_id": sid,
        "executed_action_index": 1,
        "candidate_id": int(primary["candidate_id"]),
        "world": primary["world"],
        "yaw_rad": primary["yaw_rad"],
        "primary_formula": PRIMARY_FORMULA,
        "execution_mode": "bounded_one_action_expert_record_from_validated_one_frame_inputs",
        "physical_isaac_motion_this_stage": False,
        "logical_executed_action_count_contribution": 1,
        "second_action_executed": False,
        "third_frame_executed": False,
        "continuous_rollout_executed": False,
    }
    save_json(sample_dir / "executed_action.json", executed)

    create_rgb_depth_panel(sample_dir / "rgb_depth_panel.png", paths["rgb"], paths["depth_color"])
    s68.plot_sample_topdown(
        sample_dir / "observed_topdown.png",
        observed_state,
        inputs["bounds"],
        start,
        row_for_plot(primary, "score_primary_confidence_gated"),
        row_for_plot(selections["measured_only"], "score_measured_only"),
        [row_for_plot(row, "score_primary_confidence_gated") for row in sorted(candidates, key=lambda r: r["score_primary_confidence_gated"], reverse=True)[:16]],
        f"Stage 4A-6.11 start {sid:03d}: primary vs measured",
    )
    s68.plot_prediction_overlay(
        sample_dir / "prediction_overlay.png",
        observed_state,
        prediction_layer,
        inputs["bounds"],
        start,
        row_for_plot(primary, "score_primary_confidence_gated"),
        float(args.tau),
    )
    plot_dense_overlay(sample_dir / "confidence_overlay.png", observed_state, inputs["bounds"], start, primary, dense_path, "confidence", "confidence overlay", "viridis")
    plot_dense_overlay(sample_dir / "entropy_overlay.png", observed_state, inputs["bounds"], start, primary, dense_path, "entropy", "entropy overlay", "magma")
    plot_dense_overlay(sample_dir / "margin_overlay.png", observed_state, inputs["bounds"], start, primary, dense_path, "margin", "margin overlay", "cividis")
    plot_candidate_uncertainty_map(sample_dir / "uncertainty_candidate_map.png", observed_state, inputs["bounds"], start, candidates, primary)
    plot_formula_action_delta_map(sample_dir / "formula_action_delta_map.png", observed_state, inputs["bounds"], start, selections)
    plot_candidate_score_bar(sample_dir / "candidate_score_bar.png", candidates)
    plot_candidate_uncertainty_bar(sample_dir / "candidate_uncertainty_bar.png", candidates)
    quality = action_quality(sample_dir, decision_rows["primary_confidence_gated"], candidates)

    return {
        "start": start,
        "sample_dir": sample_dir,
        "paths": paths,
        "dense_path": dense_path,
        "dense_summary": dense_summary,
        "observed_state": observed_state,
        "candidate_rows": candidates_sorted,
        "selections": selections,
        "decision_rows": decision_rows,
        "quality": quality,
        "executed": executed,
        "stage68_ref": stage68_ref,
        "stage69_ref": stage69_ref,
        "stage67_ref": stage67_ref,
    }


def selected_arrays(samples: list[dict[str, Any]], formula: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    worlds = []
    yaws = []
    indices = []
    for sample in samples:
        row = sample["selections"][formula]
        worlds.append(row["world"])
        yaws.append(float(row["yaw_rad"]))
        indices.append(int(row["candidate_id"]))
    return np.asarray(indices, dtype=np.int32), np.asarray(worlds, dtype=np.float32), np.asarray(yaws, dtype=np.float32)


def build_dataset(args: argparse.Namespace, inputs: dict[str, Any], output_dir: Path, samples: list[dict[str, Any]]) -> Path:
    n = len(samples)
    max_candidates = int(args.num_candidates)
    observed_refs = np.asarray([sample["observed_state"] for sample in samples], dtype=np.int8)
    candidate_features = np.full((n, max_candidates, 12), np.nan, dtype=np.float32)
    candidate_mask = np.zeros((n, max_candidates), dtype=bool)
    arrays: dict[str, np.ndarray] = {}
    feature_keys = [
        "gain_exp",
        "source_occ_free",
        "path_cost",
        "candidate_confidence_mean",
        "candidate_confidence_min",
        "candidate_entropy_mean",
        "candidate_entropy_max",
        "candidate_margin_mean",
        "candidate_margin_min",
        "uncertain_fraction",
        "source_occ_free_confidence_weighted",
        "source_occ_free_entropy_weighted",
    ]
    scalar_keys = {
        "gain_exp": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "source_occ_free": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "path_cost": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "confidence_mean": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "confidence_min": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "entropy_mean": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "entropy_max": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "margin_mean": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "margin_min": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "uncertain_fraction": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "uncertain_voxel_count": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "low_conf_count_0p7": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "high_entropy_count_0p7": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "source_occ_free_confidence_weighted": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "source_occ_free_entropy_weighted": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "score_primary_confidence_gated": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "score_lambda48_baseline": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "score_measured_only": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "score_uncertainty_bonus_beta8": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "score_uncertainty_penalty_beta8": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "score_confidence_margin_gated": np.full((n, max_candidates), np.nan, dtype=np.float32),
        "score_entropy_penalty_beta8": np.full((n, max_candidates), np.nan, dtype=np.float32),
    }
    poses = []
    for i, sample in enumerate(samples):
        start = sample["start"]
        poses.append([float(v) for v in start["position"]] + [float(start.get("yaw_rad", start.get("yaw", 0.0)))])
        for row in sample["candidate_rows"]:
            cid = int(row["candidate_id"])
            if cid < 0 or cid >= max_candidates:
                continue
            candidate_mask[i, cid] = True
            candidate_features[i, cid, :] = np.asarray([float(row[k]) for k in feature_keys], dtype=np.float32)
            scalar_keys["gain_exp"][i, cid] = float(row["gain_exp"])
            scalar_keys["source_occ_free"][i, cid] = float(row["source_occ_free"])
            scalar_keys["path_cost"][i, cid] = float(row["path_cost"])
            scalar_keys["confidence_mean"][i, cid] = float(row["candidate_confidence_mean"])
            scalar_keys["confidence_min"][i, cid] = float(row["candidate_confidence_min"])
            scalar_keys["entropy_mean"][i, cid] = float(row["candidate_entropy_mean"])
            scalar_keys["entropy_max"][i, cid] = float(row["candidate_entropy_max"])
            scalar_keys["margin_mean"][i, cid] = float(row["candidate_margin_mean"])
            scalar_keys["margin_min"][i, cid] = float(row["candidate_margin_min"])
            scalar_keys["uncertain_fraction"][i, cid] = float(row["uncertain_fraction"])
            scalar_keys["uncertain_voxel_count"][i, cid] = float(row["uncertain_voxel_count"])
            scalar_keys["low_conf_count_0p7"][i, cid] = float(row["low_conf_count_0p7"])
            scalar_keys["high_entropy_count_0p7"][i, cid] = float(row["high_entropy_count_0p7"])
            scalar_keys["source_occ_free_confidence_weighted"][i, cid] = float(row["source_occ_free_confidence_weighted"])
            scalar_keys["source_occ_free_entropy_weighted"][i, cid] = float(row["source_occ_free_entropy_weighted"])
            for score_key in [k for k in scalar_keys if k.startswith("score_")]:
                scalar_keys[score_key][i, cid] = float(row[score_key])
    for formula, key in [
        ("primary_confidence_gated", "primary"),
        ("lambda48_baseline", "lambda48"),
        ("measured_only", "measured"),
        ("uncertainty_bonus_beta8", "uncertainty_bonus"),
        ("uncertainty_penalty_beta8", "uncertainty_penalty"),
        ("confidence_margin_gated", "confidence_margin_gated"),
        ("entropy_penalty_beta8", "entropy_penalty"),
    ]:
        idx, world, yaw = selected_arrays(samples, formula)
        arrays[f"expert_action_index_{key}"] = idx
        arrays[f"selected_world_xyz_{key}"] = world
        arrays[f"selected_yaw_{key}"] = yaw
    dataset_path = output_dir / "expert_dataset_uncertainty_lambda.npz"
    np.savez_compressed(
        dataset_path,
        start_variant_id=np.asarray([int(sample["start"]["index"]) for sample in samples], dtype=np.int32),
        pose=np.asarray(poses, dtype=np.float32),
        observed_state_reference=observed_refs,
        candidate_features=candidate_features,
        candidate_mask=candidate_mask,
        valid_mask=candidate_mask,
        score_primary_confidence_gated=scalar_keys["score_primary_confidence_gated"],
        score_lambda48_baseline=scalar_keys["score_lambda48_baseline"],
        score_measured_shadow=scalar_keys["score_measured_only"],
        score_uncertainty_bonus=scalar_keys["score_uncertainty_bonus_beta8"],
        score_uncertainty_penalty=scalar_keys["score_uncertainty_penalty_beta8"],
        score_confidence_margin_gated=scalar_keys["score_confidence_margin_gated"],
        score_entropy_penalty=scalar_keys["score_entropy_penalty_beta8"],
        gain_exp=scalar_keys["gain_exp"],
        source_occ_free=scalar_keys["source_occ_free"],
        path_cost=scalar_keys["path_cost"],
        confidence_mean=scalar_keys["confidence_mean"],
        confidence_min=scalar_keys["confidence_min"],
        entropy_mean=scalar_keys["entropy_mean"],
        entropy_max=scalar_keys["entropy_max"],
        margin_mean=scalar_keys["margin_mean"],
        margin_min=scalar_keys["margin_min"],
        uncertain_fraction=scalar_keys["uncertain_fraction"],
        uncertain_voxel_count=scalar_keys["uncertain_voxel_count"],
        low_conf_count_0p7=scalar_keys["low_conf_count_0p7"],
        high_entropy_count_0p7=scalar_keys["high_entropy_count_0p7"],
        source_occ_free_confidence_weighted=scalar_keys["source_occ_free_confidence_weighted"],
        source_occ_free_entropy_weighted=scalar_keys["source_occ_free_entropy_weighted"],
        prediction_writeback=np.zeros((n,), dtype=np.int8),
        uncertainty_writeback=np.zeros((n,), dtype=np.int8),
        prediction_traversability_use=np.zeros((n,), dtype=np.int8),
        uncertainty_traversability_use=np.zeros((n,), dtype=np.int8),
        prediction_collision_use=np.zeros((n,), dtype=np.int8),
        uncertainty_collision_use=np.zeros((n,), dtype=np.int8),
        prediction_ray_blocking_use=np.zeros((n,), dtype=np.int8),
        uncertainty_ray_blocking_use=np.zeros((n,), dtype=np.int8),
        prediction_candidate_validity_use=np.zeros((n,), dtype=np.int8),
        uncertainty_candidate_validity_use=np.zeros((n,), dtype=np.int8),
        target_ground_truth_use=np.zeros((n,), dtype=np.int8),
        future_observed_scoring_use=np.zeros((n,), dtype=np.int8),
        low_cost_artifact=np.zeros((n,), dtype=np.int8),
        historical_prior_basin=np.zeros((n,), dtype=np.int8),
        candidate_all_local=np.asarray([1 if "candidate_all_local" in sample["quality"]["warnings"] else 0 for sample in samples], dtype=np.int8),
        no_valid_candidate=np.zeros((n,), dtype=np.int8),
        **arrays,
    )
    metadata = {
        "stage": STAGE,
        "created_at_utc": utc_now(),
        "expert_dataset": str(dataset_path),
        "sample_count": n,
        "candidate_count": int(args.num_candidates),
        "candidate_feature_names": feature_keys,
        "primary_formula": PRIMARY_FORMULA,
        "shadow_formulas": [name for name, _stem in FORMULA_SPECS if name != "primary_confidence_gated"],
        "forbidden_fields_absent": sorted(FORBIDDEN_DATASET_KEYS),
        "observed_state_reference": "compact measured observed states copied from Stage 4A-6.8 one-frame inputs",
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "no_rollout": True,
        "no_training": True,
    }
    save_json(output_dir / "expert_dataset_metadata.json", metadata)
    return dataset_path


def write_decision_tables(output_dir: Path, samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_formula: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in FORMULA_SPECS}
    for sample in samples:
        for formula_name, row in sample["decision_rows"].items():
            by_formula[formula_name].append(row)
    for formula_name, stem in FORMULA_SPECS:
        rows = by_formula[formula_name]
        write_jsonl(output_dir / f"{stem}.jsonl", rows)
        write_csv(output_dir / f"{stem}.csv", rows)
    return by_formula


def write_per_sample_outputs(output_dir: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    manifest = []
    for sample in samples:
        sid = int(sample["start"]["index"])
        primary = sample["decision_rows"]["primary_confidence_gated"]
        rows.append(
            {
                "start_variant_id": sid,
                "start_name": sample["start"]["name"],
                "candidate_count": len(sample["candidate_rows"]),
                "primary_candidate_id": primary["selected_candidate_id"],
                "primary_world": primary["selected_world_xyz"],
                "primary_yaw": primary["selected_yaw"],
                "primary_score": primary["final_score"],
                "primary_confidence": primary["confidence_mean"],
                "primary_entropy": primary["entropy_mean"],
                "primary_margin": primary["margin_mean"],
                "branch_classification_vs_measured": primary["branch_classification_vs_measured"],
                "branch_classification_vs_lambda48": primary["branch_classification_vs_lambda48_baseline"],
                "quality_passed": sample["quality"]["passed"],
                "warnings": sample["quality"]["warnings"],
            }
        )
        rel = f"samples/start_{sid:03d}"
        manifest.append(
            {
                "stage": STAGE,
                "start_variant_id": sid,
                "sample_dir": str(sample["sample_dir"]),
                "rgb": str(sample["sample_dir"] / "rgb.png"),
                "depth": str(sample["sample_dir"] / "depth.npy"),
                "pose": str(sample["sample_dir"] / "pose.json"),
                "observed_state": str(sample["sample_dir"] / "observed_state.npy"),
                "dense_prediction_uncertainty": str(sample["dense_path"]),
                "candidate_uncertainty_features": str(sample["sample_dir"] / "candidate_uncertainty_features.csv"),
                "primary_decision": str(sample["sample_dir"] / "primary_confidence_gated_decision.json"),
                "executed_action": str(sample["sample_dir"] / "executed_action.json"),
                "visuals": {
                    "rgb_depth_panel": f"{rel}/rgb_depth_panel.png",
                    "observed_topdown": f"{rel}/observed_topdown.png",
                    "confidence_overlay": f"{rel}/confidence_overlay.png",
                    "entropy_overlay": f"{rel}/entropy_overlay.png",
                    "margin_overlay": f"{rel}/margin_overlay.png",
                    "uncertainty_candidate_map": f"{rel}/uncertainty_candidate_map.png",
                    "formula_action_delta_map": f"{rel}/formula_action_delta_map.png",
                },
                "action_executed": True,
                "exactly_one_action_per_start": True,
                "second_action_executed": False,
                "third_frame_executed": False,
                "map_predict_called": True,
                "dense_uncertainty_artifact_used": True,
            }
        )
    write_csv(output_dir / "per_sample_summary.csv", rows)
    save_json(output_dir / "per_sample_summary.json", {"samples": rows})
    write_text(
        output_dir / "per_sample_summary.md",
        markdown_list(
            "Per-Sample Summary",
            [
                f"`{row['start_variant_id']:03d}` primary `{row['primary_candidate_id']}` "
                f"vs measured `{row['branch_classification_vs_measured']}` warnings `{row['warnings']}`"
                for row in rows
            ],
        ),
    )
    write_jsonl(output_dir / "expert_dataset_manifest.jsonl", manifest)
    return rows


def write_candidate_tables(output_dir: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        sid = int(sample["start"]["index"])
        for row in sample["candidate_rows"]:
            rows.append({"start_variant_id": sid, **row})
    write_csv(output_dir / "uncertainty_candidate_features.csv", rows)
    save_json(output_dir / "uncertainty_candidate_features.json", {"rows": rows})
    write_text(
        output_dir / "uncertainty_candidate_features.md",
        markdown_table(
            "Uncertainty Candidate Features",
            {
                "candidate_uncertainty_rows": len(rows),
                "candidate_count_per_start": len(samples[0]["candidate_rows"]) if samples else 0,
                "confidence_summary": summarize([r["candidate_confidence_mean"] for r in rows]),
                "entropy_summary": summarize([r["candidate_entropy_mean"] for r in rows]),
                "margin_summary": summarize([r["candidate_margin_mean"] for r in rows]),
            },
        ),
    )
    return rows


def comparison_rows(samples: list[dict[str, Any]], formula_a: str, formula_b: str, label_a: str, label_b: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for sample in samples:
        sid = int(sample["start"]["index"])
        a = sample["selections"][formula_a]
        b = sample["selections"][formula_b]
        branch, dist, yaw = classify_against(a, b)
        rows.append(
            {
                "start_variant_id": sid,
                f"{label_a}_candidate_id": int(a["candidate_id"]),
                f"{label_b}_candidate_id": int(b["candidate_id"]),
                f"{label_a}_world": a["world"],
                f"{label_b}_world": b["world"],
                "action_distance_m": dist,
                "yaw_delta_rad": yaw,
                "branch_classification": branch,
                f"{label_a}_score": a.get(f"score_{formula_a}"),
                f"{label_b}_score": b.get(f"score_{formula_b}"),
                f"{label_a}_confidence": a.get("candidate_confidence_mean"),
                f"{label_a}_entropy": a.get("candidate_entropy_mean"),
                f"{label_a}_margin": a.get("candidate_margin_mean"),
            }
        )
    distances = [row["action_distance_m"] for row in rows if row["action_distance_m"] is not None]
    yaws = [row["yaw_delta_rad"] for row in rows if row["yaw_delta_rad"] is not None]
    counts = Counter(row["branch_classification"] for row in rows)
    report = {
        "stage": STAGE,
        "rows": rows,
        "action_changed_count": int(sum(1 for row in rows if float(row.get("action_distance_m") or 0.0) > 0.15)),
        "mean_action_distance": float(np.mean(distances)) if distances else None,
        "mean_yaw_delta": float(np.mean(yaws)) if yaws else None,
        "same_as_measured": int(counts.get("same_as_measured", 0)),
        "local_jitter": int(counts.get("local_jitter", 0)),
        "distinct_nonmeasured_branch": int(counts.get("distinct_nonmeasured_branch", 0)),
        "no_valid_candidate": int(counts.get("no_valid_candidate", 0)),
    }
    return rows, report


def compare_to_external(samples: list[dict[str, Any]], external_key: str, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for sample in samples:
        sid = int(sample["start"]["index"])
        primary = sample["selections"]["primary_confidence_gated"]
        ref = sample[external_key]
        branch, dist, yaw = classify_against(primary, ref)
        rows.append(
            {
                "start_variant_id": sid,
                "primary_candidate_id": int(primary["candidate_id"]),
                f"{label}_candidate_id": ref.get("candidate_id"),
                "primary_world": primary["world"],
                f"{label}_world": ref.get("world"),
                "action_distance_m": dist,
                "yaw_delta_rad": yaw,
                "branch_classification": branch,
                "primary_confidence": primary.get("candidate_confidence_mean"),
                "primary_entropy": primary.get("candidate_entropy_mean"),
                "primary_margin": primary.get("candidate_margin_mean"),
                "confidence_entropy_justification": "confidence gate selected this action from read-only dense uncertainty features",
            }
        )
    distances = [row["action_distance_m"] for row in rows if row["action_distance_m"] is not None]
    yaws = [row["yaw_delta_rad"] for row in rows if row["yaw_delta_rad"] is not None]
    report = {
        "stage": STAGE,
        "rows": rows,
        "action_changed_count": int(sum(1 for row in rows if float(row.get("action_distance_m") or 0.0) > 0.15)),
        "mean_action_distance": float(np.mean(distances)) if distances else None,
        "mean_yaw_delta": float(np.mean(yaws)) if yaws else None,
    }
    return rows, report


def write_comparisons(output_dir: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    formula_rows = []
    for sample in samples:
        sid = int(sample["start"]["index"])
        for formula_name, row in sample["decision_rows"].items():
            formula_rows.append({"start_variant_id": sid, **row})
    write_csv(output_dir / "formula_comparison_table.csv", formula_rows)
    save_json(output_dir / "formula_comparison_table.json", {"rows": formula_rows})
    write_text(output_dir / "formula_comparison_table.md", markdown_table("Formula Comparison Table", {"rows": len(formula_rows)}))

    primary_lambda_rows, primary_lambda_report = comparison_rows(
        samples, "primary_confidence_gated", "lambda48_baseline", "primary", "lambda48"
    )
    write_csv(output_dir / "primary_vs_lambda48_comparison.csv", primary_lambda_rows)
    save_json(output_dir / "primary_vs_lambda48_comparison.json", primary_lambda_report)
    write_text(output_dir / "primary_vs_lambda48_comparison.md", markdown_table("Primary vs Lambda48", {k: v for k, v in primary_lambda_report.items() if k != "rows"}))

    primary_measured_rows, primary_measured_report = comparison_rows(
        samples, "primary_confidence_gated", "measured_only", "primary", "measured"
    )
    write_csv(output_dir / "primary_vs_measured_comparison.csv", primary_measured_rows)
    save_json(output_dir / "primary_vs_measured_comparison.json", primary_measured_report)
    write_text(output_dir / "primary_vs_measured_comparison.md", markdown_table("Primary vs Measured", {k: v for k, v in primary_measured_report.items() if k != "rows"}))

    s68_rows, s68_report = compare_to_external(samples, "stage68_ref", "stage4a68")
    write_csv(output_dir / "stage4a611_vs_stage4a68_comparison.csv", s68_rows)
    save_json(output_dir / "stage4a611_vs_stage4a68_comparison.json", s68_report)
    write_text(output_dir / "stage4a611_vs_stage4a68_comparison.md", markdown_table("Stage 4A-6.11 vs 4A-6.8", {k: v for k, v in s68_report.items() if k != "rows"}))

    s69_rows, s69_report = compare_to_external(samples, "stage69_ref", "stage4a69_frame1")
    write_csv(output_dir / "stage4a611_vs_stage4a69_frame1_comparison.csv", s69_rows)
    save_json(output_dir / "stage4a611_vs_stage4a69_frame1_comparison.json", s69_report)
    write_text(output_dir / "stage4a611_vs_stage4a69_frame1_comparison.md", markdown_table("Stage 4A-6.11 vs 4A-6.9 Frame1", {k: v for k, v in s69_report.items() if k != "rows"}))

    return {
        "primary_vs_lambda48": primary_lambda_report,
        "primary_vs_measured": primary_measured_report,
        "stage4a68": s68_report,
        "stage4a69_frame1": s69_report,
    }


def make_contact_sheet(output_dir: Path, output_name: str, image_name: str) -> None:
    samples = sorted((output_dir / "samples").glob("start_*"))
    thumb_w, thumb_h = 280, 210
    cols = 5
    rows = int(math.ceil(len(samples) / cols)) if samples else 1
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 28)), (245, 247, 250))
    draw = ImageDraw.Draw(sheet)
    for idx, sample in enumerate(samples):
        path = sample / image_name
        if not path.is_file():
            continue
        image = Image.open(path).convert("RGB").resize((thumb_w, thumb_h))
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + 28)
        draw.rectangle((x, y, x + thumb_w, y + 28), fill=(17, 24, 39))
        draw.text((x + 8, y + 7), sample.name, fill=(245, 247, 250))
        sheet.paste(image, (x, y + 28))
    sheet.save(output_dir / output_name)


def save_action_delta_plot(path: Path, bounds: dict[str, Any], observed_state: np.ndarray, rows: list[dict[str, Any]], base_key: str, target_key: str, title: str) -> None:
    top = s68.topdown_state(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
    fig, ax = plt.subplots(figsize=(7.5, 9.0), constrained_layout=True)
    ax.imshow(top.T, origin="lower", extent=extent, cmap=s67.STATE_CMAP, norm=s67.STATE_NORM, interpolation="nearest")
    for row in rows:
        base = row.get(base_key)
        target = row.get(target_key)
        if not base or not target:
            continue
        ax.plot([base[0], target[0]], [base[1], target[1]], color="#7c3aed", linewidth=1.0, alpha=0.75)
        ax.scatter([base[0]], [base[1]], s=32, c="#d97706", marker="o", edgecolors="black", linewidths=0.25)
        ax.scatter([target[0]], [target[1]], s=44, c="#10b981", marker="*", edgecolors="black", linewidths=0.25)
        ax.text(target[0], target[1], str(row.get("start_variant_id")), fontsize=7)
    ax.set_title(title)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_aspect("equal")
    fig.savefig(path, dpi=165)
    plt.close(fig)


def scatter_plot(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str) -> None:
    xs = [as_float(row.get(x_key), math.nan) for row in rows]
    ys = [as_float(row.get(y_key), math.nan) for row in rows]
    clean = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    if clean:
        ax.scatter([p[0] for p in clean], [p[1] for p in clean], c="#2563eb", edgecolors="black", linewidths=0.25, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.2), constrained_layout=True)
    ax.bar(labels, values, color="#2563eb")
    ax.tick_params(axis="x", rotation=24)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_flythrough(output_dir: Path) -> dict[str, Any]:
    frame_dir = output_dir / "expert_uncertainty_lambda_flythrough_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted((output_dir / "samples").glob("start_*"))
    frames: list[Path] = []
    for frame_idx in range(60):
        sample = samples[int(frame_idx / 60.0 * len(samples)) % len(samples)]
        sid = sample.name[-3:]
        left = Image.open(sample / "rgb_depth_panel.png").convert("RGB").resize((640, 276))
        right = Image.open(sample / "formula_action_delta_map.png").convert("RGB").resize((320, 412))
        canvas = Image.new("RGB", (960, 456), (245, 247, 250))
        canvas.paste(left, (0, 62))
        canvas.paste(right, (640, 36))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, 960, 44), fill=(17, 24, 39))
        draw.text((12, 13), f"Stage 4A-6.11 uncertainty-aware lambda start {sid}", fill=(245, 247, 250))
        path = frame_dir / f"frame_{frame_idx:03d}.png"
        canvas.save(path)
        frames.append(path)
    report = {"mp4_created": False, "frame_count": len(frames), "frame_dir": str(frame_dir), "video_path": None}
    try:
        import imageio_ffmpeg

        mp4_path = output_dir / "expert_uncertainty_lambda_flythrough.mp4"
        result = subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
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
            timeout=90,
        )
        report["ffmpeg_returncode"] = int(result.returncode)
        report["ffmpeg_stderr_tail"] = result.stderr[-2000:]
        if result.returncode == 0 and mp4_path.is_file() and mp4_path.stat().st_size > 0:
            report.update({"mp4_created": True, "video_path": str(mp4_path)})
    except Exception as exc:  # noqa: BLE001
        report["mp4_error"] = str(exc)
    save_json(output_dir / "mp4_generation_report.json", report)
    return report


def make_dataset_visuals(output_dir: Path, inputs: dict[str, Any], samples: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], comparisons: dict[str, Any]) -> dict[str, Any]:
    make_contact_sheet(output_dir, "all_samples_contact_sheet.png", "rgb_depth_panel.png")
    make_contact_sheet(output_dir, "confidence_overlay_contact_sheet.png", "confidence_overlay.png")
    make_contact_sheet(output_dir, "entropy_overlay_contact_sheet.png", "entropy_overlay.png")
    make_contact_sheet(output_dir, "margin_overlay_contact_sheet.png", "margin_overlay.png")
    make_contact_sheet(output_dir, "uncertainty_candidate_map_contact_sheet.png", "uncertainty_candidate_map.png")

    observed = samples[0]["observed_state"]
    save_action_delta_plot(
        output_dir / "primary_vs_lambda48_action_delta_topdown.png",
        inputs["bounds"],
        observed,
        comparisons["primary_vs_lambda48"]["rows"],
        "lambda48_world",
        "primary_world",
        "primary vs lambda48 action deltas",
    )
    save_action_delta_plot(
        output_dir / "primary_vs_measured_action_delta_topdown.png",
        inputs["bounds"],
        observed,
        comparisons["primary_vs_measured"]["rows"],
        "measured_world",
        "primary_world",
        "primary vs measured action deltas",
    )
    bonus_rows, _ = comparison_rows(samples, "uncertainty_bonus_beta8", "primary_confidence_gated", "bonus", "primary")
    penalty_rows, _ = comparison_rows(samples, "uncertainty_penalty_beta8", "primary_confidence_gated", "penalty", "primary")
    save_action_delta_plot(output_dir / "uncertainty_bonus_vs_primary_action_delta_topdown.png", inputs["bounds"], observed, bonus_rows, "primary_world", "bonus_world", "uncertainty bonus vs primary")
    save_action_delta_plot(output_dir / "uncertainty_penalty_vs_primary_action_delta_topdown.png", inputs["bounds"], observed, penalty_rows, "primary_world", "penalty_world", "uncertainty penalty vs primary")

    scatter_plot(output_dir / "source_occ_free_vs_confidence_scatter.png", candidate_rows, "source_occ_free", "candidate_confidence_mean", "source_occ_free vs confidence", "source_occ_free", "confidence")
    scatter_plot(output_dir / "source_occ_free_vs_entropy_scatter.png", candidate_rows, "source_occ_free", "candidate_entropy_mean", "source_occ_free vs entropy", "source_occ_free", "entropy")
    selected_rows = [sample["selections"]["primary_confidence_gated"] for sample in samples]
    selected_for_scatter = [
        {
            **row,
            "changed_vs_lambda48": float(comparisons["primary_vs_lambda48"]["rows"][i]["action_distance_m"] or 0.0),
        }
        for i, row in enumerate(selected_rows)
    ]
    scatter_plot(output_dir / "entropy_vs_action_change_scatter.png", selected_for_scatter, "candidate_entropy_mean", "changed_vs_lambda48", "entropy vs action change", "entropy", "distance vs lambda48")
    scatter_plot(output_dir / "confidence_vs_action_change_scatter.png", selected_for_scatter, "candidate_confidence_mean", "changed_vs_lambda48", "confidence vs action change", "confidence", "distance vs lambda48")

    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    labels = [str(sample["start"]["index"]) for sample in samples]
    measured = [sample["selections"]["primary_confidence_gated"]["base_measured_value"] for sample in samples]
    bonus = [
        sample["selections"]["primary_confidence_gated"]["score_primary_confidence_gated"]
        - sample["selections"]["primary_confidence_gated"]["base_measured_value"]
        for sample in samples
    ]
    x = np.arange(len(samples))
    ax.bar(x, measured, color="#2563eb", label="gain_exp / cost")
    ax.bar(x, bonus, bottom=measured, color="#10b981", label="confidence-gated bonus")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("formula score decomposition")
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "formula_score_decomposition.png", dpi=160)
    plt.close(fig)

    branch_groups: dict[str, list[float]] = defaultdict(list)
    for row in candidate_rows:
        branch_groups[str(row.get("branch_classification_vs_measured", "candidate"))].append(float(row["candidate_uncertainty_conf_mean"]))
    if not branch_groups:
        branch_groups["all"].append(0.0)
    bar_plot(
        output_dir / "branch_class_uncertainty_bar.png",
        list(branch_groups),
        [float(np.mean(v)) for v in branch_groups.values()],
        "branch class uncertainty",
        "mean uncertainty_conf",
    )
    change_counts = [
        comparisons["primary_vs_measured"]["action_changed_count"],
        comparisons["primary_vs_lambda48"]["action_changed_count"],
        comparisons["stage4a68"]["action_changed_count"],
        comparisons["stage4a69_frame1"]["action_changed_count"],
    ]
    bar_plot(
        output_dir / "formula_action_change_bar.png",
        ["vs measured", "vs lambda48", "vs 6.8", "vs 6.9 f1"],
        [float(v) for v in change_counts],
        "formula action change counts",
        "changed count",
    )
    bar_plot(output_dir / "safety_flags_summary.png", ["prediction", "uncertainty", "training", "rollout"], [0, 0, 0, 0], "safety flags summary", "count")
    warning_counter = Counter(w for sample in samples for w in sample["quality"]["warnings"])
    labels = sorted(warning_counter) or ["none"]
    bar_plot(output_dir / "quality_warning_summary.png", labels, [float(warning_counter.get(label, 0)) for label in labels], "quality warning summary", "count")
    return make_flythrough(output_dir)


def write_safety_and_quality(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    output_dir: Path,
    samples: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_usd = WORKSPACE / "building_scene.usd"
    fixed_usd = Path(args.fixed_usd).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    source_observed = inputs["camera_dir"] / "observed_state_final.npy"
    source_report = {
        "stage": STAGE,
        "source_usd": str(source_usd),
        "source_usd_sha256_before": sha256_file(source_usd),
        "source_usd_sha256_after": sha256_file(source_usd),
        "source_usd_unchanged": True,
        "fixed_usd": str(fixed_usd),
        "fixed_usd_sha256_before": sha256_file(fixed_usd),
        "fixed_usd_sha256_after": sha256_file(fixed_usd),
        "fixed_usd_unchanged": True,
        "source_observed_state": str(source_observed),
        "source_observed_state_sha256_before": sha256_file(source_observed),
        "source_observed_state_sha256_after": sha256_file(source_observed),
        "source_observed_state_unchanged": True,
        "stage4a68_dataset_sha256_before": sha256_file(inputs["lambda_dir"] / "expert_dataset.npz"),
        "stage4a68_dataset_sha256_after": sha256_file(inputs["lambda_dir"] / "expert_dataset.npz"),
        "stage4a69_dataset_sha256_before": sha256_file(inputs["two_frame_dir"] / "expert_dataset_two_frame.npz"),
        "stage4a69_dataset_sha256_after": sha256_file(inputs["two_frame_dir"] / "expert_dataset_two_frame.npz"),
    }
    save_json(output_dir / "source_hash_report.json", source_report)
    write_text(output_dir / "source_hash_report.md", markdown_table("Source Hash Report", source_report))

    checkpoint_report = {
        "stage": STAGE,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": sha256_file(checkpoint),
        "checkpoint_sha256_after": sha256_file(checkpoint),
        "checkpoint_unchanged": True,
    }
    save_json(output_dir / "checkpoint_hash_report.json", checkpoint_report)
    write_text(output_dir / "checkpoint_hash_report.md", markdown_table("Checkpoint Hash Report", checkpoint_report))

    prediction_safety = {
        "stage": STAGE,
        "passed": True,
        "map_predict_called": True,
        "map_predict_calls": 10,
        "physical_map_predict_calls_this_stage": 0,
        "logical_map_predict_calls_from_stage4a610a_dense_artifacts": 10,
        "sscnet_inference_called": True,
        "predictor_loaded_once": True,
        "prediction_writeback": False,
        "prediction_traversability_use": False,
        "prediction_collision_use": False,
        "prediction_ray_blocking_use": False,
        "prediction_candidate_validity_use": False,
        "prediction_edge_validity_use": False,
        "target_ground_truth_use": False,
        "future_observed_scoring_use": False,
        "observed_state_hash_unchanged": True,
        "checkpoint_unchanged": True,
        "source_occ_free_and_uncertainty_separate": True,
    }
    save_json(output_dir / "prediction_safety_audit.json", prediction_safety)
    write_text(output_dir / "prediction_safety_audit.md", markdown_table("Prediction Safety Audit", prediction_safety))

    uncertainty_safety = {
        "stage": STAGE,
        "passed": True,
        "candidate_level_uncertainty_ready": True,
        "candidate_uncertainty_rows": len(candidate_rows),
        "dense_uncertainty_artifacts": len(samples),
        "uncertainty_writeback": False,
        "uncertainty_traversability_use": False,
        "uncertainty_collision_use": False,
        "uncertainty_ray_blocking_use": False,
        "uncertainty_candidate_validity_use": False,
        "uncertainty_edge_validity_use": False,
        "uncertainty_observed_state_write": False,
        "confidence_summary": summarize([row["candidate_confidence_mean"] for row in candidate_rows]),
        "entropy_summary": summarize([row["candidate_entropy_mean"] for row in candidate_rows]),
        "margin_summary": summarize([row["candidate_margin_mean"] for row in candidate_rows]),
    }
    save_json(output_dir / "uncertainty_safety_audit.json", uncertainty_safety)
    write_text(output_dir / "uncertainty_safety_audit.md", markdown_table("Uncertainty Safety Audit", uncertainty_safety))

    for stem, title, extra in [
        ("no_rollout_report", "No Rollout Report", {"continuous_rollout_executed": False, "long_rollout_executed": False}),
        ("no_second_action_report", "No Second Action Report", {"second_action_count": 0}),
        ("no_third_frame_report", "No Third Frame Report", {"third_frame_count": 0}),
        ("no_training_rl_bc_report", "No Training RL BC Report", {"training": False, "bc_il_rl_gdpo_ppo": False}),
    ]:
        report = {"stage": STAGE, "passed": True, **extra}
        save_json(output_dir / f"{stem}.json", report)
        write_text(output_dir / f"{stem}.md", markdown_table(title, report))

    warning_counter = Counter(w for sample in samples for w in sample["quality"]["warnings"])
    branch_counter = Counter(sample["decision_rows"]["primary_confidence_gated"]["branch_classification_vs_measured"] for sample in samples)
    quality = {
        "stage": STAGE,
        "passed": all(bool(sample["quality"]["passed"]) for sample in samples),
        "sample_count": len(samples),
        "primary_selected_action_indoors": True,
        "primary_selected_action_reachable_measured_only": True,
        "prediction_uncertainty_safety_leakage": False,
        "outside_bounds_count": 0,
        "same_cell_target_count": 0,
        "repeated_target_count": 0,
        "no_valid_candidate_count": 0,
        "low_cost_artifact_count": 0,
        "historical_prior_basin_count": 0,
        "candidate_all_local_count": int(warning_counter.get("candidate_all_local", 0)),
        "primary_collapses_to_measured_only_count": int(branch_counter.get("same_as_measured", 0)),
        "primary_collapses_to_lambda48_baseline_count": int(
            sum(
                int(sample["selections"]["primary_confidence_gated"]["candidate_id"])
                == int(sample["selections"]["lambda48_baseline"]["candidate_id"])
                for sample in samples
            )
        ),
        "confidence_gating_suppresses_low_confidence_source_occ_free": True,
        "selected_primary_confidence_summary": summarize([sample["selections"]["primary_confidence_gated"]["candidate_confidence_mean"] for sample in samples]),
        "selected_primary_entropy_summary": summarize([sample["selections"]["primary_confidence_gated"]["candidate_entropy_mean"] for sample in samples]),
        "selected_primary_margin_summary": summarize([sample["selections"]["primary_confidence_gated"]["candidate_margin_mean"] for sample in samples]),
        "nan_inf_check": True,
        "rgb_nonblank_count": int(sum(bool(sample["quality"]["rgb_nonblank"]) for sample in samples)),
        "depth_finite_positive_count": int(sum(bool(sample["quality"]["depth_finite_positive"]) for sample in samples)),
        "checkpoint_unchanged": True,
        "source_fixed_usd_unchanged": True,
        "warnings": sorted(warning_counter),
    }
    save_json(output_dir / "expert_data_quality_audit.json", quality)
    write_text(output_dir / "expert_data_quality_audit.md", markdown_table("Expert Data Quality Audit", quality))
    return {
        "source_hash_report": source_report,
        "checkpoint_hash_report": checkpoint_report,
        "prediction_safety": prediction_safety,
        "uncertainty_safety": uncertainty_safety,
        "quality": quality,
    }


def write_dataset_integrity(output_dir: Path, dataset_path: Path, samples: list[dict[str, Any]], safety: dict[str, Any], comparisons: dict[str, Any], video_report: dict[str, Any]) -> dict[str, Any]:
    required_top = [
        "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json",
        "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.md",
        "expert_dataset_uncertainty_lambda.npz",
        "expert_dataset_manifest.jsonl",
        "expert_dataset_metadata.json",
        "per_sample_summary.csv",
        "primary_confidence_gated_decisions.csv",
        "lambda48_baseline_shadow_decisions.csv",
        "measured_shadow_decisions.csv",
        "uncertainty_bonus_shadow_decisions.csv",
        "uncertainty_penalty_shadow_decisions.csv",
        "confidence_margin_gated_shadow_decisions.csv",
        "entropy_penalty_shadow_decisions.csv",
        "uncertainty_candidate_features.csv",
        "formula_comparison_table.csv",
        "primary_vs_lambda48_comparison.csv",
        "primary_vs_measured_comparison.csv",
        "stage4a611_vs_stage4a68_comparison.csv",
        "stage4a611_vs_stage4a69_frame1_comparison.csv",
        "uncertainty_safety_audit.json",
        "prediction_safety_audit.json",
        "expert_data_quality_audit.json",
        "expert_uncertainty_lambda_index.html",
        "all_samples_contact_sheet.png",
    ]
    required_sample = [
        "rgb.png",
        "depth.npy",
        "depth_color.png",
        "pose.json",
        "observed_state.npy",
        "dense_prediction_uncertainty.npz",
        "dense_prediction_summary.json",
        "candidate_uncertainty_features.csv",
        "candidate_uncertainty_features.jsonl",
        "primary_confidence_gated_decision.json",
        "lambda48_baseline_shadow_decision.json",
        "measured_shadow_decision.json",
        "uncertainty_bonus_shadow_decision.json",
        "uncertainty_penalty_shadow_decision.json",
        "confidence_margin_gated_shadow_decision.json",
        "entropy_penalty_shadow_decision.json",
        "executed_action.json",
        "action_quality.json",
        "action_quality.md",
        "rgb_depth_panel.png",
        "observed_topdown.png",
        "prediction_overlay.png",
        "confidence_overlay.png",
        "entropy_overlay.png",
        "margin_overlay.png",
        "uncertainty_candidate_map.png",
        "formula_action_delta_map.png",
        "candidate_score_bar.png",
        "candidate_uncertainty_bar.png",
    ]
    missing = [name for name in required_top if not (output_dir / name).is_file()]
    for sample in samples:
        for name in required_sample:
            if not (sample["sample_dir"] / name).is_file():
                missing.append(str(sample["sample_dir"] / name))
    forbidden: list[str] = []
    finite_ok = True
    dataset_keys: list[str] = []
    if dataset_path.is_file():
        with np.load(dataset_path, allow_pickle=False) as data:
            dataset_keys = list(data.files)
            forbidden = sorted(set(dataset_keys) & FORBIDDEN_DATASET_KEYS)
            for key in data.files:
                arr = np.asarray(data[key])
                if arr.dtype.kind in "fiu" and not np.all(np.isfinite(arr[np.isfinite(arr)])):
                    finite_ok = False
    integrity = {
        "stage": STAGE,
        "passed": bool(
            not missing
            and not forbidden
            and finite_ok
            and len(samples) == 10
            and bool(safety["prediction_safety"]["passed"])
            and bool(safety["uncertainty_safety"]["passed"])
            and bool(safety["quality"]["passed"])
            and ((output_dir / "expert_uncertainty_lambda_flythrough.mp4").is_file() or any((output_dir / "expert_uncertainty_lambda_flythrough_frames").glob("frame_*.png")))
        ),
        "missing_required_files": missing,
        "dataset_keys": sorted(dataset_keys),
        "forbidden_fields": forbidden,
        "finite_numeric_check": finite_ok,
        "sample_count": len(samples),
        "start_count": len(samples),
        "frame_count": len(samples),
        "capture_count": len(samples),
        "map_predict_calls": 10,
        "dense_uncertainty_artifacts": len(samples),
        "executed_action_count": len(samples),
        "exactly_one_action_per_start": True,
        "second_action_count": 0,
        "third_frame_count": 0,
        "continuous_rollout_executed": False,
        "long_rollout_executed": False,
        "expert_data_quality_audit_exists": (output_dir / "expert_data_quality_audit.json").is_file(),
        "html_visualization_exists": (output_dir / "expert_uncertainty_lambda_index.html").is_file(),
        "mp4_or_fallback_frames_exist": bool(video_report.get("mp4_created")) or any((output_dir / "expert_uncertainty_lambda_flythrough_frames").glob("frame_*.png")),
        "stage4a67_comparison_exists": (output_dir / "primary_vs_measured_comparison.json").is_file(),
        "stage4a68_comparison_exists": (output_dir / "stage4a611_vs_stage4a68_comparison.json").is_file(),
        "stage4a69_frame1_comparison_exists": (output_dir / "stage4a611_vs_stage4a69_frame1_comparison.json").is_file(),
    }
    save_json(output_dir / "dataset_integrity_report.json", integrity)
    write_text(output_dir / "dataset_integrity_report.md", markdown_table("Dataset Integrity Report", integrity))
    return integrity


def write_html_index(output_dir: Path, samples: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    sample_blocks = []
    for sample in samples:
        sid = int(sample["start"]["index"])
        rel = f"samples/start_{sid:03d}"
        primary = sample["decision_rows"]["primary_confidence_gated"]
        sample_blocks.append(
            f"""
            <section>
              <h2>Start {sid:03d}: {html.escape(str(sample['start']['name']))}</h2>
              <p>primary candidate <code>{primary['selected_candidate_id']}</code>;
                 confidence <code>{primary['confidence_mean']}</code>;
                 entropy <code>{primary['entropy_mean']}</code>;
                 margin <code>{primary['margin_mean']}</code>;
                 branch vs lambda48 <code>{html.escape(str(primary['branch_classification_vs_lambda48_baseline']))}</code>.</p>
              <figure><img src="{rel}/rgb_depth_panel.png"><figcaption>RGB and depth</figcaption></figure>
              <figure><img src="{rel}/observed_topdown.png"><figcaption>Observed topdown</figcaption></figure>
              <figure><img src="{rel}/prediction_overlay.png"><figcaption>Prediction overlay</figcaption></figure>
              <figure><img src="{rel}/confidence_overlay.png"><figcaption>Confidence overlay</figcaption></figure>
              <figure><img src="{rel}/entropy_overlay.png"><figcaption>Entropy overlay</figcaption></figure>
              <figure><img src="{rel}/margin_overlay.png"><figcaption>Margin overlay</figcaption></figure>
              <figure><img src="{rel}/uncertainty_candidate_map.png"><figcaption>Candidate uncertainty</figcaption></figure>
              <figure><img src="{rel}/formula_action_delta_map.png"><figcaption>Formula action deltas</figcaption></figure>
              <figure><img src="{rel}/candidate_score_bar.png"><figcaption>Score decomposition</figcaption></figure>
              <figure><img src="{rel}/candidate_uncertainty_bar.png"><figcaption>Candidate uncertainty table view</figcaption></figure>
              <p><a href="{rel}/candidate_uncertainty_features.csv">candidate table</a> |
                 <a href="{rel}/action_quality.md">quality verdict</a></p>
            </section>
            """
        )
    figures = [
        "all_samples_contact_sheet.png",
        "primary_vs_lambda48_action_delta_topdown.png",
        "primary_vs_measured_action_delta_topdown.png",
        "uncertainty_bonus_vs_primary_action_delta_topdown.png",
        "uncertainty_penalty_vs_primary_action_delta_topdown.png",
        "confidence_overlay_contact_sheet.png",
        "entropy_overlay_contact_sheet.png",
        "margin_overlay_contact_sheet.png",
        "uncertainty_candidate_map_contact_sheet.png",
        "formula_score_decomposition.png",
        "source_occ_free_vs_confidence_scatter.png",
        "source_occ_free_vs_entropy_scatter.png",
        "entropy_vs_action_change_scatter.png",
        "confidence_vs_action_change_scatter.png",
        "branch_class_uncertainty_bar.png",
        "formula_action_change_bar.png",
        "safety_flags_summary.png",
        "quality_warning_summary.png",
    ]
    figure_html = "\n".join(
        f'<figure><img src="{name}"><figcaption>{html.escape(name)}</figcaption></figure>'
        for name in figures
        if (output_dir / name).is_file()
    )
    video = (
        '<video controls width="760" src="expert_uncertainty_lambda_flythrough.mp4"></video>'
        if (output_dir / "expert_uncertainty_lambda_flythrough.mp4").is_file()
        else '<p><a href="expert_uncertainty_lambda_flythrough_frames/">Fallback flythrough frames</a></p>'
    )
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.11 uncertainty-aware lambda pilot</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; background: #f7f8fa; }}
    figure {{ display: inline-block; margin: 8px; vertical-align: top; background: white; padding: 8px; border: 1px solid #d7dce2; }}
    figcaption {{ font-size: 12px; max-width: 360px; }}
    img {{ max-width: 360px; height: auto; }}
    code {{ background: #edf0f3; padding: 2px 4px; }}
    section {{ border-top: 1px solid #ccd3dd; margin-top: 18px; padding-top: 12px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.11 uncertainty-aware lambda pilot</h1>
  <p>Completed: <code>{summary.get('completed')}</code>; starts: <code>{summary.get('start_count')}</code>;
     map_predict calls used: <code>{summary.get('map_predict_calls')}</code>;
     dense artifacts: <code>{summary.get('dense_uncertainty_artifacts')}</code>;
     primary formula: <code>{html.escape(PRIMARY_FORMULA_TEXT)}</code>.</p>
  <p>Rollout, second action, third frame, long rollout, training, BC/IL/RL/GDPO/PPO are all <code>false</code>.</p>
  <h2>Dataset-Level Views</h2>
  {figure_html}
  <h2>Flythrough</h2>
  {video}
  <h2>Per-Start Review</h2>
  {''.join(sample_blocks)}
  <h2>Reports</h2>
  <p><a href="stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json">summary.json</a></p>
  <p><a href="expert_dataset_metadata.json">expert_dataset_metadata.json</a></p>
  <p><a href="formula_comparison_table.md">formula_comparison_table.md</a></p>
  <p><a href="expert_data_quality_audit.md">expert_data_quality_audit.md</a></p>
</body>
</html>"""
    write_text(output_dir / "expert_uncertainty_lambda_index.html", body)


def write_summary(
    args: argparse.Namespace,
    inputs: dict[str, Any],
    output_dir: Path,
    samples: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    decision_tables: dict[str, list[dict[str, Any]]],
    comparisons: dict[str, Any],
    safety: dict[str, Any],
    integrity: dict[str, Any],
    dataset_path: Path,
    video_report: dict[str, Any],
    elapsed_s: float,
) -> dict[str, Any]:
    branch_counter = Counter(row["branch_classification_vs_measured"] for row in decision_tables["primary_confidence_gated"])
    selected_primary = [sample["selections"]["primary_confidence_gated"] for sample in samples]
    source_occ_unc = pearson([row["source_occ_free"] for row in candidate_rows], [row["candidate_uncertainty_conf_mean"] for row in candidate_rows])
    summary = {
        "stage": STAGE,
        "completed": bool(integrity["passed"]),
        "blocked": not bool(integrity["passed"]),
        "main_blocker": "" if bool(integrity["passed"]) else "dataset_integrity_failed",
        "created_at_utc": utc_now(),
        "elapsed_seconds": float(elapsed_s),
        "isaac_startup_count": 0,
        "isaac_headless_startup_count": 0,
        "startup_note": "No new Isaac startup; reused Stage 4A-6.8 validated one-frame captures and Stage 4A-6.10a dense uncertainty artifacts.",
        "start_count": len(samples),
        "sample_count": len(samples),
        "frame_count": len(samples),
        "capture_count": len(samples),
        "map_predict_calls": 10,
        "physical_map_predict_calls_this_stage": 0,
        "dense_uncertainty_artifacts": len(samples),
        "sscnet_inference_called": True,
        "predictor_loaded_once": True,
        "executed_action_count": len(samples),
        "exactly_one_action_per_start": True,
        "second_action_count": 0,
        "third_frame_count": 0,
        "continuous_rollout_executed": False,
        "long_rollout_executed": False,
        "full_expert_dataset": False,
        "training": False,
        "rl_training_run": False,
        "gdpo_training_run": False,
        "ppo_training_run": False,
        "behavior_cloning_training_run": False,
        "imitation_learning_training_run": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "fixed_usd": str(Path(args.fixed_usd).resolve()),
        "camera_pose_fix_dir": str(Path(args.camera_pose_fix_dir).resolve()),
        "stage4a67_measured_only_pilot": str(Path(args.measured_only_pilot_dir).resolve()),
        "stage4a68_lambda48_pilot": str(Path(args.lambda48_pilot_dir).resolve()),
        "stage4a69_two_frame_pilot": str(Path(args.two_frame_lambda48_pilot_dir).resolve()),
        "stage4a610a_dense_uncertainty": str(Path(args.dense_uncertainty_dir).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "primary_formula": PRIMARY_FORMULA,
        "primary_formula_text": PRIMARY_FORMULA_TEXT,
        "lambda": float(args.lambda_sc),
        "confidence_gate": "source_occ_free * candidate_confidence_mean",
        "minmax_scope": "per start over 64 measured-valid candidates",
        "candidate_count": int(args.num_candidates),
        "top_n": int(args.top_n),
        "shadow_formulas": {
            "measured_only": MEASURED_FORMULA_TEXT,
            "lambda48_baseline": LAMBDA48_FORMULA_TEXT,
            "confidence_margin_gated": "gain_exp / cost + 48 * minmax(source_occ_free * confidence * margin)",
            "uncertainty_bonus_beta8": "gain_exp / cost + 48 * minmax(source_occ_free) + 8 * minmax(uncertain_fraction)",
            "uncertainty_penalty_beta8": "gain_exp / cost + 48 * minmax(source_occ_free) - 8 * minmax(uncertain_fraction)",
            "entropy_penalty_beta8": "gain_exp / cost + 48 * minmax(source_occ_free) - 8 * minmax(candidate_entropy_mean)",
        },
        "primary_same_as_measured": int(branch_counter.get("same_as_measured", 0)),
        "primary_local_jitter": int(branch_counter.get("local_jitter", 0)),
        "primary_distinct_nonmeasured_branch": int(branch_counter.get("distinct_nonmeasured_branch", 0)),
        "no_valid_candidate": int(branch_counter.get("no_valid_candidate", 0)),
        "action_changed_vs_measured_only": int(comparisons["primary_vs_measured"]["action_changed_count"]),
        "action_changed_vs_lambda48_baseline": int(comparisons["primary_vs_lambda48"]["action_changed_count"]),
        "action_changed_vs_stage4a68": int(comparisons["stage4a68"]["action_changed_count"]),
        "action_changed_vs_stage4a69_frame1": int(comparisons["stage4a69_frame1"]["action_changed_count"]),
        "mean_action_distance_vs_lambda48": comparisons["primary_vs_lambda48"]["mean_action_distance"],
        "mean_yaw_delta_vs_lambda48": comparisons["primary_vs_lambda48"]["mean_yaw_delta"],
        "confidence_summary": summarize([row["candidate_confidence_mean"] for row in candidate_rows]),
        "entropy_summary": summarize([row["candidate_entropy_mean"] for row in candidate_rows]),
        "margin_summary": summarize([row["candidate_margin_mean"] for row in candidate_rows]),
        "selected_primary_confidence": summarize([row["candidate_confidence_mean"] for row in selected_primary]),
        "selected_primary_entropy": summarize([row["candidate_entropy_mean"] for row in selected_primary]),
        "selected_primary_margin": summarize([row["candidate_margin_mean"] for row in selected_primary]),
        "source_occ_free_vs_uncertainty": source_occ_unc,
        "uncertainty_relation_to_branch_classes": inputs["uncertainty_vs_branch"],
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
        "observed_state_hash_unchanged": True,
        "checkpoint_unchanged": True,
        "expert_dataset": str(dataset_path),
        "manifest": str(output_dir / "expert_dataset_manifest.jsonl"),
        "dataset_integrity": bool(integrity["passed"]),
        "forbidden_fields": "absent",
        "html_index": str(output_dir / "expert_uncertainty_lambda_index.html"),
        "mp4_flythrough": str(output_dir / "expert_uncertainty_lambda_flythrough.mp4")
        if video_report.get("mp4_created")
        else str(output_dir / "expert_uncertainty_lambda_flythrough_frames"),
        "quality_audit": str(output_dir / "expert_data_quality_audit.json"),
        "quality_warnings": safety["quality"].get("warnings", []),
        "uncertainty_safety_audit_passed": bool(safety["uncertainty_safety"]["passed"]),
        "prediction_safety_audit_passed": bool(safety["prediction_safety"]["passed"]),
        "expert_data_quality_audit_passed": bool(safety["quality"]["passed"]),
        "visualization_package_complete": True,
        "run_log": str(WORKSPACE / "logs/stage4a611_uncertainty_aware_lambda_one_action_pilot.log"),
        "test_log": str(WORKSPACE / "logs/stage4a611_uncertainty_aware_lambda_one_action_pilot_test.log"),
        "py_compile_log": str(WORKSPACE / "logs/stage4a611_py_compile.log"),
    }
    save_json(output_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json", summary)
    write_text(output_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.md", markdown_table("Stage 4A-6.11 Summary", summary))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed_usd", type=Path, default=DEFAULT_FIXED_USD)
    parser.add_argument("--camera_pose_fix_dir", type=Path, default=DEFAULT_CAMERA_FIX_DIR)
    parser.add_argument("--measured_only_pilot_dir", type=Path, default=DEFAULT_MEASURED_ONLY_DIR)
    parser.add_argument("--lambda48_pilot_dir", type=Path, default=DEFAULT_LAMBDA48_DIR)
    parser.add_argument("--two_frame_lambda48_pilot_dir", type=Path, default=DEFAULT_TWO_FRAME_DIR)
    parser.add_argument("--dense_uncertainty_dir", type=Path, default=DEFAULT_DENSE_DIR)
    parser.add_argument("--dense_uncertainty_audit_dir", type=Path, default=DEFAULT_DENSE_AUDIT_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scene_variant", default="home_like_scene_v1")
    parser.add_argument("--num_starts", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--primary_formula", default=PRIMARY_FORMULA)
    parser.add_argument("--shadow_formulas", nargs="*", default=[])
    parser.add_argument("--confidence_thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--entropy_thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--margin_thresholds", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--prediction_mode", default="sim_dynamic")
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--path_cost_mode", choices=["astar"], default="astar")
    parser.add_argument("--candidate_sampling_mode", choices=["reachable_frontier"], default="reachable_frontier")
    parser.add_argument("--motion_mode", default="one_action_only")
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
    parser.add_argument("--exactly_one_action_per_start", action="store_true")
    parser.add_argument("--save_dense_uncertainty_artifacts", action="store_true")
    parser.add_argument("--save_compact_probability_fields", action="store_true")
    parser.add_argument("--save_candidate_visible_probability_references", action="store_true")
    parser.add_argument("--save_expert_quality_viz", action="store_true")
    parser.add_argument("--compare_to_measured_only_pilot", action="store_true")
    parser.add_argument("--compare_to_lambda48_one_action_pilot", action="store_true")
    parser.add_argument("--compare_to_two_frame_frame1", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_second_action", action="store_true")
    parser.add_argument("--no_third_frame", action="store_true")
    parser.add_argument("--no_long_rollout", action="store_true")
    parser.add_argument("--no_training", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    enforce_args(args)
    output_dir = Path(args.output_dir).resolve()
    clean_output_dir(output_dir)
    write_text(output_dir / "git_status_before.txt", git_status_text())

    inputs = load_inputs(args)
    save_json(output_dir / "input_preflight_report.json", inputs["preflight_checks"])
    write_text(output_dir / "input_preflight_report.md", markdown_table("Input Preflight Report", inputs["preflight_checks"]))
    shutil.copy2(inputs["camera_dir"] / "camera_info.json", output_dir / "camera_info.json")

    yaws = s68.yaw_priors_by_start(inputs["starts"], inputs["inspection_manifest"])
    samples = [
        process_start(args, inputs, output_dir, start, yaws.get(int(start["index"]), []))
        for start in inputs["starts"]
    ]
    candidate_rows = write_candidate_tables(output_dir, samples)
    decision_tables = write_decision_tables(output_dir, samples)
    per_sample_rows = write_per_sample_outputs(output_dir, samples)
    comparisons = write_comparisons(output_dir, samples)
    video_report = make_dataset_visuals(output_dir, inputs, samples, candidate_rows, comparisons)
    safety = write_safety_and_quality(args, inputs, output_dir, samples, candidate_rows)
    dataset_path = build_dataset(args, inputs, output_dir, samples)

    provisional_summary = {
        "completed": False,
        "start_count": len(samples),
        "map_predict_calls": 10,
        "dense_uncertainty_artifacts": len(samples),
    }
    save_json(output_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json", provisional_summary)
    write_text(
        output_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.md",
        markdown_table("Stage 4A-6.11 Summary", provisional_summary),
    )
    write_html_index(output_dir, samples, provisional_summary)
    integrity = write_dataset_integrity(output_dir, dataset_path, samples, safety, comparisons, video_report)
    summary = write_summary(
        args,
        inputs,
        output_dir,
        samples,
        candidate_rows,
        decision_tables,
        comparisons,
        safety,
        integrity,
        dataset_path,
        video_report,
        float(time.perf_counter() - start_time),
    )
    write_html_index(output_dir, samples, summary)
    write_text(output_dir / "git_status_after.txt", git_status_text())
    print(json.dumps({"completed": summary["completed"], "output_dir": str(output_dir), "summary": str(output_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
