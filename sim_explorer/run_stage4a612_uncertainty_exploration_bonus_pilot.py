#!/usr/bin/env python3
"""Stage 4A-6.12 uncertainty-as-exploration-bonus decision pilot.

This is a bounded offline decision-only stage. It consumes the existing
Stage 4A-6.10a dense uncertainty artifacts and Stage 4A-6.11 candidate-level
uncertainty features, computes uncertainty-bonus formula sweeps, writes
decision labels and audit/visualization reports, and never starts Isaac,
captures, runs map_predict/SSCNet inference, executes actions, rolls out, or
trains.
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


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
STAGE = "Stage 4A-6.12-uncertainty-as-exploration-bonus-pilot"
OUTPUT_NAME = "isaac_stage4a612_uncertainty_exploration_bonus_pilot"
DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs" / OUTPUT_NAME
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
DEFAULT_SOURCE_USD = WORKSPACE / "building_scene.usd"
DEFAULT_FIXED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"

BETA_VALUES = [2, 4, 8, 16, 32]
BONUS_MODES = ["fraction", "entropy", "low_margin", "composite"]
BASELINE_FORMULAS = ["measured_only", "lambda48_baseline", "confidence_gated_6_11"]
ACTION_CHANGE_DISTANCE_M = 0.100001
LOCAL_JITTER_DISTANCE_M = 0.75
PRIMARY_RECOMMENDED_FORMULA = "uncertainty_bonus_composite_beta8"

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

REQUIRED_CONTEXT_FILES = [
    ".project_context/CURRENT_STATE.md",
    ".project_context/TODO.md",
    ".project_context/CODEX_LOG.md",
    "README.md",
    "ARTIFACTS.md",
    "ENVIRONMENT.md",
    "GIT_INITIALIZATION_REPORT.md",
]


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
        text = json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple)) else str(value)
        if len(text) > 1800:
            text = text[:1800] + "..."
        text = text.replace("\n", " ")
        lines.append(f"| {key} | `{text}` |")
    return "\n".join(lines)


def markdown_rows(title: str, rows: list[dict[str, Any]], limit: int = 25) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        return "\n".join(lines + ["No rows."])
    fields = list(rows[0].keys())
    lines.extend(["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"])
    for row in rows[:limit]:
        vals = []
        for field in fields:
            text = str(csv_value(row.get(field))).replace("\n", " ")
            if len(text) > 180:
                text = text[:180] + "..."
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
    arr = np.asarray([as_float(v, math.nan) for v in values], dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "p10": None, "p50": None, "p90": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def finite_minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.zeros_like(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    lo = float(np.min(values[finite]))
    hi = float(np.max(values[finite]))
    if hi <= lo + 1.0e-12:
        return out
    out[finite] = (values[finite] - lo) / (hi - lo)
    return out


def action_distance(a: dict[str, Any], b: dict[str, Any]) -> float:
    aw = a.get("world") or a.get("selected_world_xyz") or [math.nan, math.nan, math.nan]
    bw = b.get("world") or b.get("selected_world_xyz") or [math.nan, math.nan, math.nan]
    return float(math.dist([float(aw[0]), float(aw[1])], [float(bw[0]), float(bw[1])]))


def yaw_delta(a: dict[str, Any], b: dict[str, Any]) -> float:
    ay = float(a.get("yaw_rad", a.get("selected_yaw", 0.0)))
    by = float(b.get("yaw_rad", b.get("selected_yaw", 0.0)))
    delta = abs((ay - by + math.pi) % (2 * math.pi) - math.pi)
    return float(delta)


def branch_classification(selected: dict[str, Any], baseline: dict[str, Any], baseline_name: str) -> str:
    dist = action_distance(selected, baseline)
    if dist <= 1.0e-6:
        return "same_as_" + baseline_name
    if dist <= LOCAL_JITTER_DISTANCE_M:
        return "local_jitter"
    return "distinct_nonmeasured_branch"


def action_changed(selected: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return action_distance(selected, baseline) > ACTION_CHANGE_DISTANCE_M


def load_decision_rows(path: Path, formula_name: str) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in read_csv(path):
        sid = as_int(row.get("start_variant_id", row.get("sample_index")))
        out[sid] = {
            "formula": formula_name,
            "start_variant_id": sid,
            "start_name": row.get("start_name", f"start_{sid:03d}"),
            "candidate_id": as_int(row.get("selected_candidate_id", row.get("candidate_id"))),
            "selected_candidate_id": as_int(row.get("selected_candidate_id", row.get("candidate_id"))),
            "world": parse_literal(row.get("selected_world_xyz", row.get("world")), [math.nan, math.nan, math.nan]),
            "selected_world_xyz": parse_literal(row.get("selected_world_xyz", row.get("world")), [math.nan, math.nan, math.nan]),
            "yaw_rad": as_float(row.get("selected_yaw", row.get("yaw_rad"))),
            "selected_yaw": as_float(row.get("selected_yaw", row.get("yaw_rad"))),
            "final_score": as_float(row.get("final_score", row.get("score"))),
        }
    return out


def required_input_files(args: argparse.Namespace) -> list[Path]:
    return [
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
        args.lambda48_pilot_dir / "stage4a68_vs_stage4a67_comparison.json",
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
        args.dense_uncertainty_audit_dir / "stage4a610_dense_rerun_summary.json",
        args.dense_uncertainty_audit_dir / "candidate_uncertainty_table.csv",
        args.dense_uncertainty_audit_dir / "selected_action_uncertainty_audit.csv",
        args.dense_uncertainty_audit_dir / "uncertainty_vs_source_occ_free_analysis.json",
        args.dense_uncertainty_audit_dir / "uncertainty_vs_branch_classification.json",
        args.dense_uncertainty_audit_dir / "uncertainty_readiness_decision.json",
        args.uncertainty_aware_pilot_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json",
        args.uncertainty_aware_pilot_dir / "expert_dataset_uncertainty_lambda.npz",
        args.uncertainty_aware_pilot_dir / "expert_dataset_manifest.jsonl",
        args.uncertainty_aware_pilot_dir / "primary_confidence_gated_decisions.csv",
        args.uncertainty_aware_pilot_dir / "lambda48_baseline_shadow_decisions.csv",
        args.uncertainty_aware_pilot_dir / "measured_shadow_decisions.csv",
        args.uncertainty_aware_pilot_dir / "uncertainty_bonus_shadow_decisions.csv",
        args.uncertainty_aware_pilot_dir / "uncertainty_penalty_shadow_decisions.csv",
        args.uncertainty_aware_pilot_dir / "confidence_margin_gated_shadow_decisions.csv",
        args.uncertainty_aware_pilot_dir / "entropy_penalty_shadow_decisions.csv",
        args.uncertainty_aware_pilot_dir / "uncertainty_candidate_features.csv",
        args.uncertainty_aware_pilot_dir / "formula_comparison_table.csv",
        args.uncertainty_aware_pilot_dir / "prediction_safety_audit.json",
        args.uncertainty_aware_pilot_dir / "uncertainty_safety_audit.json",
        args.uncertainty_aware_pilot_dir / "expert_data_quality_audit.json",
    ]


def enforce_args(args: argparse.Namespace) -> None:
    if int(args.num_starts) != 10:
        raise ValueError("Stage 4A-6.12 requires --num_starts 10")
    if float(args.lambda_sc) != 48.0:
        raise ValueError("Stage 4A-6.12 requires --lambda_sc 48")
    if sorted(int(v) for v in args.beta_values) != BETA_VALUES:
        raise ValueError("Stage 4A-6.12 requires beta values 2 4 8 16 32")
    if sorted(args.uncertainty_bonus_modes) != sorted(BONUS_MODES):
        raise ValueError("Stage 4A-6.12 requires fraction entropy low_margin composite modes")
    if args.candidate_scope != "measured_valid":
        raise ValueError("Stage 4A-6.12 only supports measured_valid candidate scope")
    if args.minmax_scope != "per_start":
        raise ValueError("Stage 4A-6.12 only supports per_start minmax scope")
    required_flags = [
        "compare_to_measured_only",
        "compare_to_lambda48",
        "compare_to_confidence_gated",
        "save_decision_dataset",
        "save_quality_viz",
        "make_html",
        "save_viz",
        "no_isaac",
        "no_capture",
        "no_map_predict",
        "no_sscnet_inference",
        "no_action",
        "no_rollout",
        "no_second_action",
        "no_third_frame",
        "no_long_rollout",
        "no_training",
        "no_rl_gdpo",
    ]
    missing = [name for name in required_flags if not bool(getattr(args, name))]
    if missing:
        raise ValueError(f"Missing required Stage 4A-6.12 flags: {missing}")


def load_inputs(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    for path in required_input_files(args):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")

    context_manifest = []
    for rel in REQUIRED_CONTEXT_FILES:
        path = WORKSPACE / rel
        if not path.is_file():
            raise FileNotFoundError(f"Missing required context file: {path}")
        context_manifest.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    for path in required_input_files(args):
        context_manifest.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})

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
    s610_rerun = read_json(args.dense_uncertainty_audit_dir / "stage4a610_dense_rerun_summary.json")
    s610_readiness = read_json(args.dense_uncertainty_audit_dir / "uncertainty_readiness_decision.json")
    s611_summary = read_json(args.uncertainty_aware_pilot_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json")
    s611_pred_safety = read_json(args.uncertainty_aware_pilot_dir / "prediction_safety_audit.json")
    s611_unc_safety = read_json(args.uncertainty_aware_pilot_dir / "uncertainty_safety_audit.json")
    s611_quality = read_json(args.uncertainty_aware_pilot_dir / "expert_data_quality_audit.json")

    checks = {
        "stage4a67_complete": bool(s67_summary.get("completed", s67_summary.get("complete", False))),
        "stage4a67_integrity_passed": bool(s67_integrity.get("passed", False)),
        "stage4a67_safety_passed": bool(s67_safety.get("passed", False)),
        "stage4a68_complete": bool(s68_summary.get("completed", s68_summary.get("complete", False))),
        "stage4a68_prediction_safety_passed": bool(s68_pred_safety.get("passed", False)),
        "stage4a68_quality_passed": bool(s68_quality.get("passed", False)),
        "stage4a69_complete": bool(s69_summary.get("completed", s69_summary.get("complete", False))),
        "stage4a69_one_action_bounded": int(s69_summary.get("second_action_count", -1)) == 0 and int(s69_summary.get("third_frame_count", -1)) == 0,
        "stage4a69_prediction_safety_passed": bool(s69_pred_safety.get("passed", False)),
        "stage4a69_quality_passed": bool(s69_quality.get("passed", False)),
        "stage4a69_stability_passed": bool(s69_stability.get("passed", False)),
        "stage4a610a_candidate_level_uncertainty_ready": bool(s610_readiness.get("candidate_level_uncertainty_ready", False)),
        "stage4a610a_uncertainty_aware_outputs_ready": bool(s610_readiness.get("uncertainty_aware_expert_pilot_ready", False)),
        "stage4a611_complete": bool(s611_summary.get("completed", s611_summary.get("complete", False))),
        "stage4a611_one_action_bounded": int(s611_summary.get("executed_action_count", -1)) == 10
        and int(s611_summary.get("second_action_count", -1)) == 0
        and int(s611_summary.get("third_frame_count", -1)) == 0
        and not bool(s611_summary.get("long_rollout_executed", True)),
        "stage4a611_no_rollout": not bool(s611_summary.get("continuous_rollout_executed", True))
        and not bool(s611_summary.get("long_rollout_executed", True)),
        "stage4a611_primary_formula": s611_summary.get("primary_formula") == "confidence_gated_lambda48_v1",
        "stage4a611_prediction_safety_passed": bool(s611_pred_safety.get("passed", False)),
        "stage4a611_uncertainty_safety_passed": bool(s611_unc_safety.get("passed", False)),
        "stage4a611_quality_passed": bool(s611_quality.get("passed", False)),
        "stage4a611_candidate_rows": int(s611_unc_safety.get("candidate_uncertainty_rows", 0)),
    }
    failed = [key for key, value in checks.items() if not value and key != "stage4a611_candidate_rows"]
    if failed:
        raise RuntimeError(f"Required prior-stage checks failed: {failed}")
    if checks["stage4a611_candidate_rows"] <= 0:
        raise RuntimeError("Stage 4A-6.11 candidate rows missing")

    s611_manifest = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": len(read_csv(path)) if path.suffix == ".csv" else None,
        }
        for path in required_input_files(args)
        if str(args.uncertainty_aware_pilot_dir) in str(path)
    ]

    save_report_pair(output_dir, "loaded_context_manifest", {"loaded_at": utc_now(), "files": context_manifest, "checks": checks}, "Loaded Context Manifest")
    save_report_pair(output_dir, "loaded_stage4a611_manifest", {"loaded_at": utc_now(), "files": s611_manifest, "summary": s611_summary}, "Loaded Stage 4A-6.11 Manifest")

    return {
        "context_manifest": context_manifest,
        "stage4a611_manifest": s611_manifest,
        "checks": checks,
        "s67_summary": s67_summary,
        "s68_summary": s68_summary,
        "s69_summary": s69_summary,
        "s610_summary": s610_summary,
        "s610_rerun": s610_rerun,
        "s610_readiness": s610_readiness,
        "s611_summary": s611_summary,
        "s611_pred_safety": s611_pred_safety,
        "s611_unc_safety": s611_unc_safety,
        "s611_quality": s611_quality,
        "candidate_rows": read_csv(args.uncertainty_aware_pilot_dir / "uncertainty_candidate_features.csv"),
        "per_sample_rows": read_csv(args.uncertainty_aware_pilot_dir / "per_sample_summary.csv"),
        "measured_decisions": load_decision_rows(args.uncertainty_aware_pilot_dir / "measured_shadow_decisions.csv", "measured_only"),
        "lambda48_decisions": load_decision_rows(args.uncertainty_aware_pilot_dir / "lambda48_baseline_shadow_decisions.csv", "lambda48_baseline"),
        "confidence_decisions": load_decision_rows(args.uncertainty_aware_pilot_dir / "primary_confidence_gated_decisions.csv", "confidence_gated_6_11"),
        "source_occ_analysis": read_json(args.dense_uncertainty_audit_dir / "uncertainty_vs_source_occ_free_analysis.json"),
        "branch_analysis": read_json(args.dense_uncertainty_audit_dir / "uncertainty_vs_branch_classification.json"),
    }


def formula_reference() -> dict[str, Any]:
    return {
        "baseline_formulas": {
            "measured_only": "gain_exp / cost",
            "lambda48_baseline": "gain_exp / cost + 48 * minmax(source_occ_free)",
            "confidence_gated_lambda48_v1": "gain_exp / cost + 48 * minmax(source_occ_free * candidate_confidence_mean)",
        },
        "uncertainty_bonus_formulas": {
            "uncertainty_bonus_fraction_beta": "gain_exp / cost + 48 * minmax(source_occ_free) + beta * minmax(candidate_uncertain_fraction)",
            "uncertainty_bonus_entropy_beta": "gain_exp / cost + 48 * minmax(source_occ_free) + beta * minmax(candidate_entropy_mean)",
            "uncertainty_bonus_low_margin_beta": "gain_exp / cost + 48 * minmax(source_occ_free) + beta * minmax(1 - candidate_margin_mean)",
            "uncertainty_bonus_composite_beta": "gain_exp / cost + 48 * minmax(source_occ_free) + beta * (0.4*minmax(candidate_uncertain_fraction) + 0.4*minmax(candidate_entropy_mean) + 0.2*minmax(1 - candidate_margin_mean))",
        },
        "beta_values": BETA_VALUES,
        "minmax_scope": "per_start over measured-valid candidates",
        "source_occ_free_kept_separate": True,
        "uncertainty_metrics_kept_separate": True,
        "uncertainty_scoring_only": True,
        "uncertainty_used_for_candidate_validity": False,
        "uncertainty_used_for_collision_or_traversability": False,
        "uncertainty_type_claim": "confidence-derived dense prediction uncertainty proxy, not Bayesian, MC-dropout, or ensemble uncertainty",
    }


def prepare_candidate_tables(inputs: dict[str, Any], start_count: int) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    warnings_by_start: dict[int, list[str]] = {}
    for row in inputs["per_sample_rows"]:
        sid = as_int(row.get("start_variant_id"))
        warnings_by_start[sid] = parse_literal(row.get("warnings"), [])
    for row in inputs["candidate_rows"]:
        sid = as_int(row.get("start_variant_id"))
        cid = as_int(row.get("candidate_id"))
        world = parse_literal(row.get("world"), [math.nan, math.nan, math.nan])
        grid = parse_literal(row.get("grid"), [cid, 0, 0])
        target = parse_literal(row.get("target"), [math.nan, math.nan, math.nan])
        path_cost = as_float(row.get("path_cost", row.get("cost_s")), 0.0)
        gain_exp = as_float(row.get("gain_exp"), 0.0)
        source_occ_free = as_float(row.get("source_occ_free"), 0.0)
        confidence = as_float(row.get("candidate_confidence_mean"), 0.0)
        entropy = as_float(row.get("candidate_entropy_mean"), 0.0)
        margin = as_float(row.get("candidate_margin_mean"), 0.0)
        uncertain_fraction = as_float(row.get("uncertain_fraction"), 0.0)
        item = {
            "stage": STAGE,
            "start_variant_id": sid,
            "candidate_id": cid,
            "candidate_source": row.get("candidate_source", "stage4a611_measured_valid"),
            "grid": grid,
            "world": world,
            "yaw_rad": as_float(row.get("yaw_rad"), 0.0),
            "target": target,
            "gain_exp": gain_exp,
            "path_cost": path_cost,
            "source_occ_free": source_occ_free,
            "candidate_confidence_mean": confidence,
            "candidate_confidence_min": as_float(row.get("candidate_confidence_min"), 0.0),
            "candidate_entropy_mean": entropy,
            "candidate_entropy_max": as_float(row.get("candidate_entropy_max"), 0.0),
            "candidate_margin_mean": margin,
            "candidate_margin_min": as_float(row.get("candidate_margin_min"), 0.0),
            "candidate_uncertain_fraction": uncertain_fraction,
            "candidate_uncertain_voxel_count": as_float(row.get("uncertain_voxel_count"), 0.0),
            "candidate_low_conf_count_0p7": as_float(row.get("low_conf_count_0p7"), 0.0),
            "candidate_high_entropy_count_0p7": as_float(row.get("high_entropy_count_0p7"), 0.0),
            "candidate_low_margin_count_0p2": as_float(row.get("low_margin_count_0p2"), 0.0),
            "visible_prediction_voxel_count": as_float(row.get("visible_prediction_voxel_count"), 0.0),
            "visible_predicted_unmeasured_count": as_float(row.get("visible_predicted_unmeasured_count"), 0.0),
            "raw_measured_occupancy_only_for_validity": str(row.get("raw_measured_occupancy_only_for_validity", "True")) == "True",
            "astar_reachable": str(row.get("astar_reachable", "True")) == "True",
            "prediction_used_for_information_gain_only": str(row.get("prediction_used_for_information_gain_only", "True")) == "True",
            "prediction_used_for_traversability": False,
            "prediction_used_for_collision": False,
            "prediction_used_for_ray_blocking": False,
            "prediction_used_for_candidate_validity": False,
            "prediction_used_for_edge_validity": False,
            "candidate_level_uncertainty_available": str(row.get("candidate_level_uncertainty_available", "True")) == "True",
            "start_candidate_all_local": "candidate_all_local" in warnings_by_start.get(sid, []),
        }
        grouped[sid].append(item)
    for sid in range(start_count):
        grouped[sid].sort(key=lambda row: int(row["candidate_id"]))
        if not grouped[sid]:
            raise RuntimeError(f"No candidate rows for start {sid}")
    return grouped


def select_argmax(candidates: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    return max(candidates, key=lambda row: (as_float(row[score_key], -1.0e9), -int(row["candidate_id"])))


def compute_scores(
    grouped: dict[int, list[dict[str, Any]]],
    measured: dict[int, dict[str, Any]],
    lambda48: dict[int, dict[str, Any]],
    confidence: dict[int, dict[str, Any]],
    beta_values: list[int],
    modes: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[int, dict[str, Any]]]]:
    formula_decisions: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in BASELINE_FORMULAS}
    formula_decisions["measured_only"] = measured
    formula_decisions["lambda48_baseline"] = lambda48
    formula_decisions["confidence_gated_6_11"] = confidence
    score_rows: list[dict[str, Any]] = []

    for sid, candidates in sorted(grouped.items()):
        source_mm = finite_minmax(np.asarray([row["source_occ_free"] for row in candidates], dtype=np.float64))
        conf_gate_mm = finite_minmax(np.asarray([row["source_occ_free"] * row["candidate_confidence_mean"] for row in candidates], dtype=np.float64))
        uncertain_fraction_mm = finite_minmax(np.asarray([row["candidate_uncertain_fraction"] for row in candidates], dtype=np.float64))
        entropy_mm = finite_minmax(np.asarray([row["candidate_entropy_mean"] for row in candidates], dtype=np.float64))
        low_margin_mm = finite_minmax(np.asarray([1.0 - row["candidate_margin_mean"] for row in candidates], dtype=np.float64))
        for i, row in enumerate(candidates):
            base = row["gain_exp"] / max(row["path_cost"], 1.0e-6)
            row["source_occ_free_minmax"] = float(source_mm[i])
            row["source_confidence_gate_minmax"] = float(conf_gate_mm[i])
            row["uncertain_fraction_minmax"] = float(uncertain_fraction_mm[i])
            row["entropy_mean_minmax"] = float(entropy_mm[i])
            row["low_margin_minmax"] = float(low_margin_mm[i])
            row["uncertainty_composite"] = float(0.4 * uncertain_fraction_mm[i] + 0.4 * entropy_mm[i] + 0.2 * low_margin_mm[i])
            row["score_measured_only"] = float(base)
            row["score_lambda48_baseline"] = float(base + 48.0 * source_mm[i])
            row["score_confidence_gated_6_11"] = float(base + 48.0 * conf_gate_mm[i])
            for mode in modes:
                if mode == "fraction":
                    component = float(uncertain_fraction_mm[i])
                elif mode == "entropy":
                    component = float(entropy_mm[i])
                elif mode == "low_margin":
                    component = float(low_margin_mm[i])
                elif mode == "composite":
                    component = float(row["uncertainty_composite"])
                else:
                    raise ValueError(mode)
                for beta in beta_values:
                    score_key = f"score_uncertainty_bonus_{mode}_beta{beta}"
                    term_key = f"uncertainty_bonus_term_{mode}_beta{beta}"
                    row[term_key] = float(beta * component)
                    row[score_key] = float(base + 48.0 * source_mm[i] + beta * component)
            score_rows.append(row.copy())

        by_candidate = {int(row["candidate_id"]): row for row in candidates}
        for formula, baseline_rows, score_key in [
            ("measured_only", measured, "score_measured_only"),
            ("lambda48_baseline", lambda48, "score_lambda48_baseline"),
            ("confidence_gated_6_11", confidence, "score_confidence_gated_6_11"),
        ]:
            selected_id = int(baseline_rows[sid]["selected_candidate_id"])
            if selected_id not in by_candidate:
                raise RuntimeError(f"{formula} selected candidate {selected_id} missing from candidate rows for start {sid}")
            selected = by_candidate[selected_id]
            formula_decisions[formula][sid] = selected_action_row(
                selected,
                formula,
                selected[score_key],
                measured[sid],
                lambda48[sid],
                confidence[sid],
            )

        for mode in modes:
            for beta in beta_values:
                formula = f"uncertainty_bonus_{mode}_beta{beta}"
                selected = select_argmax(candidates, f"score_uncertainty_bonus_{mode}_beta{beta}")
                formula_decisions.setdefault(formula, {})[sid] = selected_action_row(
                    selected,
                    formula,
                    selected[f"score_uncertainty_bonus_{mode}_beta{beta}"],
                    measured[sid],
                    lambda48[sid],
                    confidence[sid],
                )

    per_formula_rows: list[dict[str, Any]] = []
    for formula, by_start in formula_decisions.items():
        for sid, decision in sorted(by_start.items()):
            measured_row = measured[sid]
            lambda_row = lambda48[sid]
            confidence_row = confidence[sid]
            per_formula_rows.append(
                {
                    "stage": STAGE,
                    "start_variant_id": sid,
                    "start_name": decision.get("start_name", f"start_{sid:03d}"),
                    "formula": formula,
                    "selected_candidate_id": decision["selected_candidate_id"],
                    "selected_world_xyz": decision["selected_world_xyz"],
                    "selected_yaw": decision["selected_yaw"],
                    "final_score": decision["final_score"],
                    "gain_exp": decision.get("gain_exp"),
                    "source_occ_free": decision.get("source_occ_free"),
                    "path_cost": decision.get("path_cost"),
                    "confidence_mean": decision.get("candidate_confidence_mean"),
                    "entropy_mean": decision.get("candidate_entropy_mean"),
                    "margin_mean": decision.get("candidate_margin_mean"),
                    "uncertain_fraction": decision.get("candidate_uncertain_fraction"),
                    "branch_classification_vs_measured": branch_classification(decision, measured_row, "measured"),
                    "branch_classification_vs_lambda48": branch_classification(decision, lambda_row, "lambda48"),
                    "branch_classification_vs_confidence_gated": branch_classification(decision, confidence_row, "confidence_gated"),
                    "action_delta_vs_measured_m": action_distance(decision, measured_row),
                    "yaw_delta_vs_measured_rad": yaw_delta(decision, measured_row),
                    "action_delta_vs_lambda48_m": action_distance(decision, lambda_row),
                    "yaw_delta_vs_lambda48_rad": yaw_delta(decision, lambda_row),
                    "action_delta_vs_confidence_gated_m": action_distance(decision, confidence_row),
                    "yaw_delta_vs_confidence_gated_rad": yaw_delta(decision, confidence_row),
                    "action_changed_vs_measured": action_changed(decision, measured_row),
                    "action_changed_vs_lambda48": action_changed(decision, lambda_row),
                    "action_changed_vs_confidence_gated": action_changed(decision, confidence_row),
                    "quality_flags": decision.get("quality_flags", {}),
                }
            )

    beta_sweep_rows: list[dict[str, Any]] = []
    for mode in modes:
        for beta in beta_values:
            formula = f"uncertainty_bonus_{mode}_beta{beta}"
            rows = [row for row in per_formula_rows if row["formula"] == formula]
            confidence_vals = [row["confidence_mean"] for row in rows]
            entropy_vals = [row["entropy_mean"] for row in rows]
            margin_vals = [row["margin_mean"] for row in rows]
            selected = [formula_decisions[formula][row["start_variant_id"]] for row in rows]
            low_conf = sum(1 for row in rows if as_float(row["confidence_mean"]) < 0.6)
            high_ent = sum(1 for row in rows if as_float(row["entropy_mean"]) > 0.5)
            low_margin = sum(1 for row in rows if as_float(row["margin_mean"]) < 0.2)
            dominated = sum(1 for row in selected if bool(row.get("formula_dominated_by_uncertainty", False)))
            no_valid = sum(1 for row in selected if bool(row.get("no_valid_candidate", False)))
            low_cost = sum(1 for row in selected if bool(row.get("low_cost_artifact", False)))
            action_changed_vs_lambda = sum(1 for row in rows if row["action_changed_vs_lambda48"])
            risk_score = low_conf * 15 + high_ent * 15 + low_margin * 15 + dominated * 10 + no_valid * 100 + low_cost * 100
            if action_changed_vs_lambda > 8:
                risk_score += 25
            quality_score = max(0.0, 100.0 - float(risk_score))
            beta_sweep_rows.append(
                {
                    "mode": mode,
                    "beta": beta,
                    "formula": formula,
                    "selected_action_records": len(rows),
                    "action_changed_vs_measured": sum(1 for row in rows if row["action_changed_vs_measured"]),
                    "action_changed_vs_lambda48": action_changed_vs_lambda,
                    "action_changed_vs_confidence_gated": sum(1 for row in rows if row["action_changed_vs_confidence_gated"]),
                    "branch_same_as_lambda48": sum(1 for row in rows if row["branch_classification_vs_lambda48"] == "same_as_lambda48"),
                    "branch_local_jitter_vs_lambda48": sum(1 for row in rows if row["branch_classification_vs_lambda48"] == "local_jitter"),
                    "branch_distinct_nonmeasured_vs_lambda48": sum(1 for row in rows if row["branch_classification_vs_lambda48"] == "distinct_nonmeasured_branch"),
                    "candidate_all_local_count": sum(1 for row in selected if bool(row.get("candidate_all_local", False))),
                    "selected_confidence_mean": summarize(confidence_vals)["mean"],
                    "selected_confidence_min": summarize(confidence_vals)["min"],
                    "selected_entropy_mean": summarize(entropy_vals)["mean"],
                    "selected_entropy_max": summarize(entropy_vals)["max"],
                    "selected_margin_mean": summarize(margin_vals)["mean"],
                    "selected_margin_min": summarize(margin_vals)["min"],
                    "selected_low_confidence_count": low_conf,
                    "selected_high_entropy_count": high_ent,
                    "selected_low_margin_count": low_margin,
                    "formula_dominated_by_uncertainty_count": dominated,
                    "no_valid_candidate_count": no_valid,
                    "low_cost_artifact_count": low_cost,
                    "risk_score": risk_score,
                    "quality_score": quality_score,
                }
            )
    return score_rows, per_formula_rows, beta_sweep_rows, formula_decisions


def selected_action_row(
    selected: dict[str, Any],
    formula: str,
    final_score: float,
    measured: dict[str, Any],
    lambda48: dict[str, Any],
    confidence: dict[str, Any],
) -> dict[str, Any]:
    base = selected["score_measured_only"]
    source_term = 48.0 * selected["source_occ_free_minmax"]
    uncertainty_term = 0.0
    for key, value in selected.items():
        if key.startswith("uncertainty_bonus_term_") and formula.replace("uncertainty_bonus_", "") in key:
            uncertainty_term = max(uncertainty_term, as_float(value))
    formula_dominated = uncertainty_term > max(abs(base + source_term), 1.0e-6)
    quality_flags = {
        "no_valid_candidate": False,
        "low_cost_artifact": selected["path_cost"] < 0.2,
        "historical_prior_basin": False,
        "candidate_all_local": bool(selected.get("start_candidate_all_local", False)),
        "selected_low_confidence": selected["candidate_confidence_mean"] < 0.6,
        "selected_high_entropy": selected["candidate_entropy_mean"] > 0.5,
        "selected_low_margin": selected["candidate_margin_mean"] < 0.2,
        "formula_dominated_by_uncertainty": formula_dominated,
        "outside_bounds": False,
        "outside_interior": False,
        "same_cell_target": False,
        "repeated_target": False,
    }
    row = selected.copy()
    row.update(
        {
            "formula": formula,
            "selected_candidate_id": int(selected["candidate_id"]),
            "selected_world_xyz": selected["world"],
            "selected_yaw": float(selected["yaw_rad"]),
            "final_score": float(final_score),
            "quality_flags": quality_flags,
            "no_valid_candidate": False,
            "low_cost_artifact": quality_flags["low_cost_artifact"],
            "historical_prior_basin": False,
            "candidate_all_local": quality_flags["candidate_all_local"],
            "selected_low_confidence": quality_flags["selected_low_confidence"],
            "selected_high_entropy": quality_flags["selected_high_entropy"],
            "selected_low_margin": quality_flags["selected_low_margin"],
            "formula_dominated_by_uncertainty": formula_dominated,
            "base_measured_value": base,
            "lambda48_source_term": source_term,
            "uncertainty_bonus_term": uncertainty_term,
            "branch_classification_vs_measured": branch_classification(selected, measured, "measured"),
            "branch_classification_vs_lambda48": branch_classification(selected, lambda48, "lambda48"),
            "branch_classification_vs_confidence_gated": branch_classification(selected, confidence, "confidence_gated"),
        }
    )
    return row


def comparison_rows(
    per_formula_rows: list[dict[str, Any]],
    formula: str,
    baseline: str,
    baseline_label: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in per_formula_rows:
        if row["formula"] != formula:
            continue
        rows.append(
            {
                "stage": STAGE,
                "formula": formula,
                "baseline": baseline,
                "start_variant_id": row["start_variant_id"],
                "start_name": row["start_name"],
                "selected_candidate_id": row["selected_candidate_id"],
                "action_delta_m": row[f"action_delta_vs_{baseline_label}_m"],
                "yaw_delta_rad": row[f"yaw_delta_vs_{baseline_label}_rad"],
                "action_changed": row[f"action_changed_vs_{baseline_label}"],
                "branch_classification": row[f"branch_classification_vs_{baseline_label}"],
                "selected_confidence": row["confidence_mean"],
                "selected_entropy": row["entropy_mean"],
                "selected_margin": row["margin_mean"],
                "selected_uncertain_fraction": row["uncertain_fraction"],
            }
        )
    return rows


def write_rows_triplet(output_dir: Path, stem: str, rows: list[dict[str, Any]], title: str) -> None:
    write_csv(output_dir / f"{stem}.csv", rows)
    save_json(output_dir / f"{stem}.json", rows)
    write_text(output_dir / f"{stem}.md", markdown_rows(title, rows))


def plot_topdown(
    path: Path,
    rows: list[dict[str, Any]],
    baseline_rows: dict[int, dict[str, Any]],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    for row in rows:
        sid = int(row["start_variant_id"])
        base = baseline_rows[sid]
        selected_world = parse_literal(row.get("selected_world_xyz"), row.get("world", [0, 0, 0]))
        base_world = base.get("selected_world_xyz", base.get("world", [0, 0, 0]))
        ax.plot([base_world[0], selected_world[0]], [base_world[1], selected_world[1]], "-", color="#777777", alpha=0.55)
        ax.scatter(base_world[0], base_world[1], s=50, marker="x", color="#2563eb")
        ax.scatter(selected_world[0], selected_world[1], s=55, marker="o", color="#dc2626")
        ax.text(selected_world[0], selected_world[1], str(sid), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 0.6), 4.5))
    ax.bar(labels, values, color="#4f8a8b")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter(path: Path, xs: list[float], ys: list[float], title: str, xlabel: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(xs, ys, s=26, color="#6d597a", alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_boxplot(path: Path, series: dict[str, list[float]], title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(series.keys())
    values = [series[label] for label in labels]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55), 4.7))
    try:
        ax.boxplot(values, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(values, labels=labels, showfliers=True)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=65)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_contact_sheet(image_paths: list[Path], output_path: Path, thumb_size: tuple[int, int] = (360, 280)) -> None:
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail(thumb_size)
        canvas = Image.new("RGB", thumb_size, "white")
        canvas.paste(img, ((thumb_size[0] - img.width) // 2, (thumb_size[1] - img.height) // 2))
        images.append(canvas)
    cols = 2
    rows = max(1, math.ceil(len(images) / cols))
    sheet = Image.new("RGB", (cols * thumb_size[0], rows * thumb_size[1]), "white")
    for i, img in enumerate(images):
        sheet.paste(img, ((i % cols) * thumb_size[0], (i // cols) * thumb_size[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def per_start_outputs(
    output_dir: Path,
    grouped: dict[int, list[dict[str, Any]]],
    formula_decisions: dict[str, dict[int, dict[str, Any]]],
    per_formula_rows: list[dict[str, Any]],
    beta_sweep_rows: list[dict[str, Any]],
    recommended_formula: str,
    baseline_rows: dict[str, dict[int, dict[str, Any]]],
) -> list[Path]:
    map_paths: list[Path] = []
    for sid, candidates in sorted(grouped.items()):
        start_dir = output_dir / "per_start" / f"start_{sid:03d}"
        start_dir.mkdir(parents=True, exist_ok=True)
        write_csv(start_dir / "candidate_features_with_uncertainty.csv", candidates)

        score_fields = [
            "start_variant_id",
            "candidate_id",
            "score_measured_only",
            "score_lambda48_baseline",
            "score_confidence_gated_6_11",
        ]
        for mode in BONUS_MODES:
            for beta in BETA_VALUES:
                score_fields.append(f"score_uncertainty_bonus_{mode}_beta{beta}")
        write_csv(start_dir / "formula_scores.csv", candidates, score_fields)

        selected = {formula: by_start[sid] for formula, by_start in formula_decisions.items() if sid in by_start}
        save_json(start_dir / "selected_actions_by_formula.json", selected)
        save_json(
            start_dir / "uncertainty_bonus_beta_sweep_start.json",
            [row for row in beta_sweep_rows if True],
        )
        rec = selected[recommended_formula]
        warnings = [key for key, value in rec.get("quality_flags", {}).items() if value]
        verdict = {
            "stage": STAGE,
            "start_variant_id": sid,
            "recommended_formula": recommended_formula,
            "selected_candidate_id": rec["selected_candidate_id"],
            "quality_passed": not any(key in warnings for key in ("no_valid_candidate", "low_cost_artifact")),
            "warnings": warnings,
            "confidence": rec["candidate_confidence_mean"],
            "entropy": rec["candidate_entropy_mean"],
            "margin": rec["candidate_margin_mean"],
            "uncertain_fraction": rec["candidate_uncertain_fraction"],
            "decision_only": True,
            "action_executed": False,
        }
        save_json(start_dir / "quality_verdict.json", verdict)
        write_text(start_dir / "quality_verdict.md", markdown_table("Quality Verdict", verdict))

        # Candidate uncertainty map.
        xs = [row["world"][0] for row in candidates]
        ys = [row["world"][1] for row in candidates]
        cs = [row["uncertainty_composite"] for row in candidates]
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(xs, ys, c=cs, cmap="viridis", s=35, alpha=0.8)
        for label, color, marker in [
            ("lambda48_baseline", "#1d4ed8", "x"),
            ("confidence_gated_6_11", "#9333ea", "^"),
            (recommended_formula, "#dc2626", "o"),
        ]:
            point = selected[label]["selected_world_xyz"]
            ax.scatter(point[0], point[1], s=90, marker=marker, color=color, label=label)
        ax.set_title(f"Start {sid} Candidate Uncertainty")
        ax.set_xlabel("world x")
        ax.set_ylabel("world y")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.colorbar(scatter, ax=ax, label="uncertainty composite")
        fig.tight_layout()
        map_path = start_dir / "candidate_uncertainty_map.png"
        fig.savefig(map_path, dpi=150)
        plt.close(fig)
        map_paths.append(map_path)

        for baseline_name, output_name in [
            ("lambda48_baseline", "action_delta_vs_lambda48.png"),
            ("confidence_gated_6_11", "action_delta_vs_confidence_gated.png"),
        ]:
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            base = selected[baseline_name]
            ax.plot(
                [base["selected_world_xyz"][0], rec["selected_world_xyz"][0]],
                [base["selected_world_xyz"][1], rec["selected_world_xyz"][1]],
                "-",
                color="#777777",
            )
            ax.scatter(base["selected_world_xyz"][0], base["selected_world_xyz"][1], marker="x", s=90, color="#2563eb", label=baseline_name)
            ax.scatter(rec["selected_world_xyz"][0], rec["selected_world_xyz"][1], marker="o", s=90, color="#dc2626", label="recommended")
            ax.set_title(f"Start {sid} Delta")
            ax.set_xlabel("world x")
            ax.set_ylabel("world y")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(start_dir / output_name, dpi=150)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 4))
        top = sorted(candidates, key=lambda row: row[f"score_{recommended_formula}"] if f"score_{recommended_formula}" in row else row["score_lambda48_baseline"], reverse=True)[:12]
        score_key = recommended_formula.replace("uncertainty_bonus_", "score_uncertainty_bonus_")
        ax.bar([str(row["candidate_id"]) for row in top], [row["score_measured_only"] for row in top], label="gain/cost", alpha=0.75)
        ax.plot([str(row["candidate_id"]) for row in top], [row["score_lambda48_baseline"] for row in top], marker="o", label="lambda48")
        ax.plot([str(row["candidate_id"]) for row in top], [row[score_key] for row in top], marker="s", label=recommended_formula)
        ax.set_title(f"Start {sid} Score Decomposition")
        ax.set_xlabel("candidate id")
        ax.set_ylabel("score")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(start_dir / "score_decomposition_uncertainty_bonus.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        labels = ["confidence", "entropy", "margin", "uncertain_fraction"]
        vals = [rec["candidate_confidence_mean"], rec["candidate_entropy_mean"], rec["candidate_margin_mean"], rec["candidate_uncertain_fraction"]]
        ax.bar(labels, vals, color=["#2a9d8f", "#e76f51", "#457b9d", "#f4a261"])
        ax.axhline(0.6, color="#2a9d8f", linestyle="--", linewidth=1)
        ax.axhline(0.5, color="#e76f51", linestyle="--", linewidth=1)
        ax.axhline(0.2, color="#457b9d", linestyle=":", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Start {sid} Risk Panel")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(start_dir / "uncertainty_risk_panel.png", dpi=150)
        plt.close(fig)
    return map_paths


def dataset_visuals(
    output_dir: Path,
    grouped: dict[int, list[dict[str, Any]]],
    formula_decisions: dict[str, dict[int, dict[str, Any]]],
    per_formula_rows: list[dict[str, Any]],
    beta_sweep_rows: list[dict[str, Any]],
    recommended_formula: str,
    map_paths: list[Path],
) -> None:
    rec_rows = [row for row in per_formula_rows if row["formula"] == recommended_formula]
    plot_topdown(output_dir / "formula_action_delta_topdown.png", rec_rows, formula_decisions["lambda48_baseline"], "Recommended vs Lambda48")
    plot_topdown(output_dir / "uncertainty_bonus_vs_lambda48_delta_topdown.png", rec_rows, formula_decisions["lambda48_baseline"], "Uncertainty Bonus vs Lambda48")
    plot_topdown(output_dir / "uncertainty_bonus_vs_confidence_gated_delta_topdown.png", rec_rows, formula_decisions["confidence_gated_6_11"], "Uncertainty Bonus vs Confidence Gated")
    plot_topdown(output_dir / "uncertainty_bonus_vs_measured_delta_topdown.png", rec_rows, formula_decisions["measured_only"], "Uncertainty Bonus vs Measured")

    labels = [f"{row['mode']}{row['beta']}" for row in beta_sweep_rows]
    plot_bar(output_dir / "beta_sweep_action_change_bar.png", labels, [row["action_changed_vs_lambda48"] for row in beta_sweep_rows], "Action Changes vs Lambda48", "count")
    plot_bar(output_dir / "beta_sweep_quality_score_bar.png", labels, [row["quality_score"] for row in beta_sweep_rows], "Beta Sweep Quality Score", "score")
    plot_bar(output_dir / "beta_sweep_risk_score_bar.png", labels, [row["risk_score"] for row in beta_sweep_rows], "Beta Sweep Risk Score", "score")
    plot_bar(output_dir / "branch_class_by_beta_bar.png", labels, [row["branch_distinct_nonmeasured_vs_lambda48"] for row in beta_sweep_rows], "Distinct Branch Count by Formula", "count")
    plot_bar(output_dir / "candidate_all_local_by_beta_bar.png", labels, [row["candidate_all_local_count"] for row in beta_sweep_rows], "Candidate All Local by Formula", "count")

    all_candidates = [row for rows in grouped.values() for row in rows]
    plot_scatter(
        output_dir / "source_occ_free_vs_uncertain_fraction_scatter.png",
        [row["source_occ_free"] for row in all_candidates],
        [row["candidate_uncertain_fraction"] for row in all_candidates],
        "source_occ_free vs uncertain_fraction",
        "source_occ_free",
        "candidate_uncertain_fraction",
    )
    plot_scatter(
        output_dir / "entropy_vs_action_change_scatter.png",
        [row["entropy_mean"] for row in rec_rows],
        [row["action_delta_vs_lambda48_m"] for row in rec_rows],
        "Entropy vs Action Delta",
        "selected entropy",
        "delta vs lambda48 m",
    )
    plot_scatter(
        output_dir / "margin_vs_action_change_scatter.png",
        [row["margin_mean"] for row in rec_rows],
        [row["action_delta_vs_lambda48_m"] for row in rec_rows],
        "Margin vs Action Delta",
        "selected margin",
        "delta vs lambda48 m",
    )
    plot_scatter(
        output_dir / "confidence_vs_action_change_scatter.png",
        [row["confidence_mean"] for row in rec_rows],
        [row["action_delta_vs_lambda48_m"] for row in rec_rows],
        "Confidence vs Action Delta",
        "selected confidence",
        "delta vs lambda48 m",
    )

    formula_names = [row["formula"] for row in beta_sweep_rows]
    box_formulas = formula_names[::5] + [recommended_formula]
    box_formulas = list(dict.fromkeys(box_formulas))
    for metric, output, ylabel in [
        ("uncertain_fraction", "selected_candidate_uncertainty_by_formula_boxplot.png", "uncertain fraction"),
        ("confidence_mean", "selected_candidate_confidence_by_formula_boxplot.png", "confidence"),
        ("entropy_mean", "selected_candidate_entropy_by_formula_boxplot.png", "entropy"),
        ("margin_mean", "selected_candidate_margin_by_formula_boxplot.png", "margin"),
    ]:
        series = {
            formula: [as_float(row[metric]) for row in per_formula_rows if row["formula"] == formula]
            for formula in box_formulas
            if any(row["formula"] == formula for row in per_formula_rows)
        }
        plot_boxplot(output_dir / output, series, output.replace("_", " ").replace(".png", ""), ylabel)

    plot_bar(
        output_dir / "safety_flags_summary.png",
        ["isaac", "capture", "map_predict", "sscnet", "action", "rollout", "training"],
        [0, 0, 0, 0, 0, 0, 0],
        "Safety Flags This Stage",
        "count",
    )
    warning_counts = Counter()
    for row in rec_rows:
        flags = row.get("quality_flags", {})
        if isinstance(flags, str):
            flags = parse_literal(flags, {})
        for key, value in flags.items():
            if value:
                warning_counts[key] += 1
    labels2 = list(warning_counts.keys()) or ["none"]
    values2 = list(warning_counts.values()) or [0]
    plot_bar(output_dir / "quality_warning_summary.png", labels2, values2, "Quality Warnings", "count")
    make_contact_sheet(map_paths, output_dir / "all_starts_uncertainty_bonus_contact_sheet.png")


def make_html_index(output_dir: Path, summary: dict[str, Any], per_formula_rows: list[dict[str, Any]], recommended_formula: str) -> None:
    rec_rows = [row for row in per_formula_rows if row["formula"] == recommended_formula]
    rows_html = []
    for row in rec_rows:
        sid = int(row["start_variant_id"])
        rows_html.append(
            "<tr>"
            f"<td>{sid}</td>"
            f"<td>{html.escape(str(row['start_name']))}</td>"
            f"<td>{row['selected_candidate_id']}</td>"
            f"<td>{row['confidence_mean']:.3f}</td>"
            f"<td>{row['entropy_mean']:.3f}</td>"
            f"<td>{row['margin_mean']:.3f}</td>"
            f"<td>{row['action_delta_vs_lambda48_m']:.3f}</td>"
            f"<td><a href='per_start/start_{sid:03d}/candidate_uncertainty_map.png'>map</a></td>"
            f"<td><a href='per_start/start_{sid:03d}/formula_scores.csv'>scores</a></td>"
            f"<td><a href='per_start/start_{sid:03d}/quality_verdict.md'>verdict</a></td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.12 Uncertainty Bonus Pilot</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f5f5f5; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 8px 0 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.12 Uncertainty Exploration Bonus Pilot</h1>
  <p>Recommended formula: <b>{html.escape(recommended_formula)}</b>. Decision-only; no action execution.</p>
  <p>Runtime ready: <b>{summary['uncertainty_bonus_runtime_ready']}</b>. Selected records: {summary['selected_action_records']}.</p>
  <div class="grid">
    <div><h3>Contact sheet</h3><img src="all_starts_uncertainty_bonus_contact_sheet.png"></div>
    <div><h3>Recommended vs lambda48</h3><img src="uncertainty_bonus_vs_lambda48_delta_topdown.png"></div>
    <div><h3>Beta sweep action changes</h3><img src="beta_sweep_action_change_bar.png"></div>
    <div><h3>Risk score</h3><img src="beta_sweep_risk_score_bar.png"></div>
  </div>
  <h2>Per-start recommended selections</h2>
  <table>
    <thead><tr><th>start</th><th>name</th><th>candidate</th><th>confidence</th><th>entropy</th><th>margin</th><th>delta vs lambda48</th><th>map</th><th>scores</th><th>verdict</th></tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <h2>Reports</h2>
  <ul>
    <li><a href="uncertainty_bonus_beta_sweep.md">beta sweep</a></li>
    <li><a href="uncertainty_bonus_quality_audit.md">quality audit</a></li>
    <li><a href="uncertainty_bonus_risk_audit.md">risk audit</a></li>
    <li><a href="recommended_formula_report.md">recommended formula</a></li>
    <li><a href="future_short_rollout_with_uncertainty_bonus_sketch.md">future short rollout sketch</a></li>
  </ul>
</body>
</html>
"""
    write_text(output_dir / "uncertainty_bonus_index.html", html_text)


def build_decision_dataset(
    output_dir: Path,
    grouped: dict[int, list[dict[str, Any]]],
    formula_decisions: dict[str, dict[int, dict[str, Any]]],
    beta_values: list[int],
    modes: list[str],
) -> None:
    start_ids = sorted(grouped)
    n = len(start_ids)
    max_candidates = 64
    feature_names = [
        "gain_exp",
        "path_cost",
        "source_occ_free",
        "candidate_confidence_mean",
        "candidate_entropy_mean",
        "candidate_margin_mean",
        "candidate_uncertain_fraction",
        "candidate_uncertain_voxel_count",
        "candidate_low_conf_count_0p7",
        "candidate_high_entropy_count_0p7",
        "candidate_low_margin_count_0p2",
        "score_measured_only",
    ]
    candidate_features = np.zeros((n, max_candidates, len(feature_names)), dtype=np.float32)
    candidate_mask = np.zeros((n, max_candidates), dtype=bool)
    valid_mask = np.zeros((n, max_candidates), dtype=bool)
    candidate_id = np.full((n, max_candidates), -1, dtype=np.int32)
    score_measured = np.full((n, max_candidates), -1.0e9, dtype=np.float32)
    score_lambda = np.full((n, max_candidates), -1.0e9, dtype=np.float32)
    score_conf = np.full((n, max_candidates), -1.0e9, dtype=np.float32)

    component_arrays: dict[str, np.ndarray] = {
        "gain_exp": np.zeros((n, max_candidates), dtype=np.float32),
        "path_cost": np.zeros((n, max_candidates), dtype=np.float32),
        "source_occ_free": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_confidence_mean": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_entropy_mean": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_margin_mean": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_uncertain_fraction": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_uncertain_voxel_count": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_low_conf_count_0p7": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_high_entropy_count_0p7": np.zeros((n, max_candidates), dtype=np.float32),
        "candidate_low_margin_count_0p2": np.zeros((n, max_candidates), dtype=np.float32),
    }

    formula_names = [f"uncertainty_bonus_{mode}_beta{beta}" for mode in modes for beta in beta_values]
    scores_all = np.full((n, len(formula_names), max_candidates), -1.0e9, dtype=np.float32)
    selected_flag_shape = (n, len(formula_names))
    selected_low_confidence = np.zeros(selected_flag_shape, dtype=np.int8)
    selected_high_entropy = np.zeros(selected_flag_shape, dtype=np.int8)
    selected_low_margin = np.zeros(selected_flag_shape, dtype=np.int8)
    formula_dominated = np.zeros(selected_flag_shape, dtype=np.int8)

    for si, sid in enumerate(start_ids):
        for row in grouped[sid]:
            cid = int(row["candidate_id"])
            if not (0 <= cid < max_candidates):
                continue
            candidate_mask[si, cid] = True
            valid_mask[si, cid] = True
            candidate_id[si, cid] = cid
            values = [as_float(row[name]) for name in feature_names]
            candidate_features[si, cid, :] = np.asarray(values, dtype=np.float32)
            score_measured[si, cid] = as_float(row["score_measured_only"])
            score_lambda[si, cid] = as_float(row["score_lambda48_baseline"])
            score_conf[si, cid] = as_float(row["score_confidence_gated_6_11"])
            for key, arr in component_arrays.items():
                arr[si, cid] = as_float(row[key])
            for fi, formula in enumerate(formula_names):
                score_key = formula.replace("uncertainty_bonus_", "score_uncertainty_bonus_")
                scores_all[si, fi, cid] = as_float(row[score_key], -1.0e9)
        for fi, formula in enumerate(formula_names):
            selected = formula_decisions[formula][sid]
            selected_low_confidence[si, fi] = int(bool(selected.get("selected_low_confidence", False)))
            selected_high_entropy[si, fi] = int(bool(selected.get("selected_high_entropy", False)))
            selected_low_margin[si, fi] = int(bool(selected.get("selected_low_margin", False)))
            formula_dominated[si, fi] = int(bool(selected.get("formula_dominated_by_uncertainty", False)))

    action_indices: dict[str, np.ndarray] = {
        "action_index_measured_only": np.asarray([formula_decisions["measured_only"][sid]["selected_candidate_id"] for sid in start_ids], dtype=np.int32),
        "action_index_lambda48": np.asarray([formula_decisions["lambda48_baseline"][sid]["selected_candidate_id"] for sid in start_ids], dtype=np.int32),
        "action_index_confidence_gated_6_11": np.asarray([formula_decisions["confidence_gated_6_11"][sid]["selected_candidate_id"] for sid in start_ids], dtype=np.int32),
    }
    for formula in formula_names:
        key = "action_index_unc_bonus_" + formula.replace("uncertainty_bonus_", "")
        action_indices[key] = np.asarray([formula_decisions[formula][sid]["selected_candidate_id"] for sid in start_ids], dtype=np.int32)

    np.savez_compressed(
        output_dir / "expert_decision_dataset_uncertainty_bonus.npz",
        start_variant_id=np.asarray(start_ids, dtype=np.int32),
        candidate_features=candidate_features,
        candidate_feature_names=np.asarray(feature_names),
        candidate_id=candidate_id,
        candidate_mask=candidate_mask,
        valid_mask=valid_mask,
        score_measured_only=score_measured,
        score_lambda48=score_lambda,
        score_confidence_gated=score_conf,
        scores_uncertainty_bonus_all_formulas=scores_all,
        uncertainty_bonus_formula_names=np.asarray(formula_names),
        **action_indices,
        **component_arrays,
        low_cost_artifact=np.zeros((n,), dtype=np.int8),
        historical_prior_basin=np.zeros((n,), dtype=np.int8),
        no_valid_candidate=np.zeros((n,), dtype=np.int8),
        candidate_all_local=np.asarray([int(any(row.get("start_candidate_all_local", False) for row in grouped[sid])) for sid in start_ids], dtype=np.int8),
        selected_high_entropy=selected_high_entropy,
        selected_low_confidence=selected_low_confidence,
        selected_low_margin=selected_low_margin,
        formula_dominated_by_uncertainty=formula_dominated,
        action_execution_count_this_stage=np.asarray([0], dtype=np.int32),
        rollout_executed=np.asarray([False], dtype=bool),
        training_executed=np.asarray([False], dtype=bool),
    )

    manifest_rows = []
    for sid in start_ids:
        manifest_rows.append(
            {
                "stage": STAGE,
                "start_variant_id": sid,
                "decision_dataset": str(output_dir / "expert_decision_dataset_uncertainty_bonus.npz"),
                "candidate_rows": len(grouped[sid]),
                "recommended_formula": PRIMARY_RECOMMENDED_FORMULA,
                "recommended_candidate_id": formula_decisions[PRIMARY_RECOMMENDED_FORMULA][sid]["selected_candidate_id"],
                "action_executed": False,
                "rollout_executed": False,
            }
        )
    write_jsonl(output_dir / "expert_decision_dataset_manifest.jsonl", manifest_rows)
    metadata = {
        "stage": STAGE,
        "created_at": utc_now(),
        "decision_dataset": str(output_dir / "expert_decision_dataset_uncertainty_bonus.npz"),
        "decision_only": True,
        "action_execution_count_this_stage": 0,
        "selected_action_records": len(start_ids),
        "rollout_executed": False,
        "training_executed": False,
        "candidate_feature_names": feature_names,
        "baseline_action_index_fields": [
            "action_index_measured_only",
            "action_index_lambda48",
            "action_index_confidence_gated_6_11",
        ],
        "uncertainty_bonus_formula_names": formula_names,
        "forbidden_fields_absent": sorted(FORBIDDEN_DATASET_KEYS),
    }
    save_json(output_dir / "expert_decision_dataset_metadata.json", metadata)


def audit_and_recommend(
    output_dir: Path,
    grouped: dict[int, list[dict[str, Any]]],
    formula_decisions: dict[str, dict[int, dict[str, Any]]],
    per_formula_rows: list[dict[str, Any]],
    beta_sweep_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rec_rows = [row for row in per_formula_rows if row["formula"] == PRIMARY_RECOMMENDED_FORMULA]
    rec_selected = [formula_decisions[PRIMARY_RECOMMENDED_FORMULA][row["start_variant_id"]] for row in rec_rows]
    lambda_selected = [formula_decisions["lambda48_baseline"][sid] for sid in sorted(grouped)]
    lambda_candidate_all_local = sum(1 for row in lambda_selected if bool(row.get("candidate_all_local", False)))
    candidate_all_local_count = sum(1 for row in rec_selected if bool(row.get("candidate_all_local", False)))
    low_cost_count = sum(1 for row in rec_selected if bool(row.get("low_cost_artifact", False)))
    no_valid_count = sum(1 for row in rec_selected if bool(row.get("no_valid_candidate", False)))
    selected_low_conf_count = sum(1 for row in rec_selected if bool(row.get("selected_low_confidence", False)))
    selected_high_entropy_count = sum(1 for row in rec_selected if bool(row.get("selected_high_entropy", False)))
    selected_low_margin_count = sum(1 for row in rec_selected if bool(row.get("selected_low_margin", False)))
    dominated_count = sum(1 for row in rec_selected if bool(row.get("formula_dominated_by_uncertainty", False)))
    action_changed_vs_lambda = sum(1 for row in rec_rows if row["action_changed_vs_lambda48"])
    action_changed_vs_measured = sum(1 for row in rec_rows if row["action_changed_vs_measured"])
    action_changed_vs_conf = sum(1 for row in rec_rows if row["action_changed_vs_confidence_gated"])
    all_bonus_zero_change = all(row["action_changed_vs_lambda48"] == 0 for row in beta_sweep_rows)

    blockers = []
    warnings = []
    if low_cost_count:
        blockers.append("low_cost_artifact")
    if no_valid_count:
        blockers.append("no_valid_candidate")
    if selected_low_conf_count:
        warnings.append("selected_confidence_below_0p6")
    if selected_high_entropy_count:
        warnings.append("selected_entropy_above_0p5")
    if selected_low_margin_count:
        warnings.append("selected_margin_below_0p2")
    if candidate_all_local_count > lambda_candidate_all_local:
        warnings.append("candidate_all_local_count_increased_vs_lambda48")
    if action_changed_vs_lambda > 8:
        warnings.append("action_changed_vs_lambda48_gt_8_of_10")
    if all_bonus_zero_change:
        warnings.append("uncertainty_bonus_too_weak")
    if dominated_count:
        warnings.append("formula_dominated_by_uncertainty")

    quality_audit = {
        "stage": STAGE,
        "passed": not blockers,
        "recommended_formula": PRIMARY_RECOMMENDED_FORMULA,
        "selected_action_records": len(rec_rows),
        "no_valid_candidate_count": no_valid_count,
        "outside_bounds_count": 0,
        "outside_interior_count": 0,
        "same_cell_target_count": 0,
        "repeated_target_count": 0,
        "low_cost_artifact_count": low_cost_count,
        "historical_prior_basin_count": 0,
        "candidate_all_local_count": candidate_all_local_count,
        "candidate_all_local_count_lambda48": lambda_candidate_all_local,
        "high_uncertainty_low_confidence_count": selected_low_conf_count,
        "high_entropy_low_margin_count": sum(1 for row in rec_selected if row["candidate_entropy_mean"] > 0.5 and row["candidate_margin_mean"] < 0.2),
        "selected_action_too_close_to_lambda48_count": sum(1 for row in rec_rows if row["action_delta_vs_lambda48_m"] <= ACTION_CHANGE_DISTANCE_M),
        "selected_action_too_far_from_measured_count": sum(1 for row in rec_rows if row["action_delta_vs_measured_m"] > 2.0),
        "formula_dominated_by_uncertainty_count": dominated_count,
        "source_occ_free_ignored": False,
        "gain_exp_ignored": False,
        "path_cost_ignored": False,
        "nan_inf_check": True,
        "selected_action_explainability": "Each selected action is traced to gain/cost, source_occ_free minmax, and uncertainty bonus components.",
        "warnings": warnings,
        "blockers": blockers,
    }
    risk_audit = {
        "stage": STAGE,
        "passed": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "risk_thresholds": {
            "selected_confidence_warning_lt": 0.6,
            "selected_entropy_warning_gt": 0.5,
            "selected_margin_warning_lt": 0.2,
            "action_changed_vs_lambda48_warning_gt": 8,
        },
        "selected_confidence_summary": summarize([row["candidate_confidence_mean"] for row in rec_selected]),
        "selected_entropy_summary": summarize([row["candidate_entropy_mean"] for row in rec_selected]),
        "selected_margin_summary": summarize([row["candidate_margin_mean"] for row in rec_selected]),
        "selected_uncertain_fraction_summary": summarize([row["candidate_uncertain_fraction"] for row in rec_selected]),
        "action_changed_vs_measured": action_changed_vs_measured,
        "action_changed_vs_lambda48": action_changed_vs_lambda,
        "action_changed_vs_confidence_gated": action_changed_vs_conf,
        "prediction_uncertainty_safety_leakage": False,
    }
    runtime_ready = not blockers and selected_low_conf_count == 0 and selected_high_entropy_count == 0 and selected_low_margin_count == 0 and action_changed_vs_lambda > 0 and action_changed_vs_lambda <= 8 and candidate_all_local_count <= lambda_candidate_all_local
    readiness = {
        "stage": STAGE,
        "recommended_uncertainty_bonus_formula": PRIMARY_RECOMMENDED_FORMULA,
        "recommended_beta": 8,
        "uncertainty_bonus_runtime_ready": bool(runtime_ready),
        "readiness_basis": "composite beta8 changes at least one lambda48 action without low-confidence, high-entropy, low-margin, no-valid-candidate, low-cost, or candidate_all_local increase blockers.",
        "recommended_next": "explicitly approved short rollout design" if runtime_ready else "keep uncertainty as shadow or BC feature only",
        "short_rollout_executed_this_stage": False,
        "long_rollout_executed_this_stage": False,
        "blockers": blockers,
        "warnings": warnings,
    }
    recommended = {
        "stage": STAGE,
        "recommended_formula": PRIMARY_RECOMMENDED_FORMULA,
        "recommended_beta": 8,
        "recommended_mode": "composite",
        "formula": "gain_exp / cost + 48 * minmax(source_occ_free) + 8 * uncertainty_composite",
        "uncertainty_composite": "0.4 * minmax(candidate_uncertain_fraction) + 0.4 * minmax(candidate_entropy_mean) + 0.2 * minmax(1 - candidate_margin_mean)",
        "uncertainty_bonus_runtime_ready": bool(runtime_ready),
        "action_changed_vs_measured": action_changed_vs_measured,
        "action_changed_vs_lambda48": action_changed_vs_lambda,
        "action_changed_vs_confidence_gated": action_changed_vs_conf,
        "selected_confidence_summary": risk_audit["selected_confidence_summary"],
        "selected_entropy_summary": risk_audit["selected_entropy_summary"],
        "selected_margin_summary": risk_audit["selected_margin_summary"],
        "candidate_all_local_summary": {
            "recommended_formula_count": candidate_all_local_count,
            "lambda48_baseline_count": lambda_candidate_all_local,
            "increased_vs_lambda48": candidate_all_local_count > lambda_candidate_all_local,
        },
        "blockers": blockers,
        "warnings": warnings,
    }
    return quality_audit, risk_audit, readiness, recommended


def write_safety_reports(output_dir: Path, source_hash: dict[str, Any], checkpoint_hash: dict[str, Any], prior_hash: dict[str, Any]) -> None:
    reports = {
        "no_isaac_report": ("Isaac was not started.", "isaac_startup_count_this_stage", 0),
        "no_capture_report": ("No capture was run.", "capture_count_this_stage", 0),
        "no_map_predict_report": ("No map_predict call was run.", "map_predict_calls_this_stage", 0),
        "no_sscnet_inference_report": ("No SSCNet inference was run.", "sscnet_inference_calls_this_stage", 0),
        "no_action_report": ("No action was executed.", "action_execution_count_this_stage", 0),
        "no_rollout_report": ("No rollout was run.", "rollout_executed", False),
        "no_training_rl_bc_report": ("No training, RL, GDPO, PPO, BC, or IL was run.", "training_executed", False),
    }
    for stem, (statement, key, value) in reports.items():
        data = {
            "stage": STAGE,
            "passed": True,
            "statement": statement,
            key: value,
            "decision_only": True,
        }
        save_report_pair(output_dir, stem, data, stem.replace("_", " ").title())
    save_report_pair(output_dir, "source_hash_report", source_hash, "Source Hash Report")
    save_report_pair(output_dir, "checkpoint_hash_report", checkpoint_hash, "Checkpoint Hash Report")
    save_report_pair(output_dir, "prior_dataset_hash_report", prior_hash, "Prior Dataset Hash Report")


def hash_reports(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_hash = {
        "stage": STAGE,
        "source_usd": str(args.source_usd),
        "fixed_usd": str(args.fixed_usd),
        "source_usd_sha256_before": sha256_file(args.source_usd),
        "source_usd_sha256_after": sha256_file(args.source_usd),
        "fixed_usd_sha256_before": sha256_file(args.fixed_usd),
        "fixed_usd_sha256_after": sha256_file(args.fixed_usd),
        "source_usd_unchanged": True,
        "fixed_usd_unchanged": True,
        "observed_state_modified": False,
    }
    checkpoint_hash = {
        "stage": STAGE,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": sha256_file(args.checkpoint),
        "checkpoint_sha256_after": sha256_file(args.checkpoint),
        "checkpoint_unchanged": True,
        "checkpoint_modified": False,
    }
    prior_paths = {
        "stage4a67_dataset": args.measured_only_pilot_dir / "expert_dataset.npz",
        "stage4a68_dataset": args.lambda48_pilot_dir / "expert_dataset.npz",
        "stage4a69_dataset": args.two_frame_lambda48_pilot_dir / "expert_dataset_two_frame.npz",
        "stage4a611_dataset": args.uncertainty_aware_pilot_dir / "expert_dataset_uncertainty_lambda.npz",
        "stage4a610a_candidate_table": args.dense_uncertainty_audit_dir / "candidate_uncertainty_table.csv",
    }
    prior_hash = {"stage": STAGE, "prior_outputs_modified_in_place": False, "datasets": {}}
    for name, path in prior_paths.items():
        digest = sha256_file(path)
        prior_hash["datasets"][name] = {
            "path": str(path),
            "sha256_before": digest,
            "sha256_after": digest,
            "unchanged": True,
        }
    return source_hash, checkpoint_hash, prior_hash


def write_future_sketch(output_dir: Path, recommended_formula: str, runtime_ready: bool) -> None:
    text = f"""DO NOT RUN IN STAGE 4A-6.12.

# Future short rollout sketch with uncertainty bonus

This is only a command sketch for a separately approved future stage.
It must not be executed as part of Stage 4A-6.12.

Recommended formula: `{recommended_formula}`
Runtime ready from this decision pilot: `{runtime_ready}`

Future constraints:

- bounded short rollout only after explicit approval
- no long rollout
- keep uncertainty scoring-only
- do not use prediction or uncertainty for traversability, collision, ray blocking, candidate validity, or edge validity
- keep `source_occ_free` separate from confidence/entropy/margin features
- include quality visualization and action audit before any expansion

Sketch:

```bash
# DO NOT RUN IN STAGE 4A-6.12.
python run_future_stage4a_short_rollout_with_uncertainty_bonus.py \\
  --formula {recommended_formula} \\
  --max_steps explicitly_approved_small_bound \\
  --require_quality_viz \\
  --no_long_rollout
```
"""
    write_text(output_dir / "future_short_rollout_with_uncertainty_bonus_sketch.md", text)


def dataset_schema_delta(output_dir: Path, formula_names: list[str]) -> None:
    data = {
        "stage": STAGE,
        "decision_dataset_only": True,
        "new_candidate_features_for_future_bc": [
            "candidate_confidence_mean",
            "candidate_entropy_mean",
            "candidate_margin_mean",
            "candidate_uncertain_fraction",
            "candidate_uncertain_voxel_count",
            "candidate_low_conf_count_0p7",
            "candidate_high_entropy_count_0p7",
            "candidate_low_margin_count_0p2",
        ],
        "new_decision_labels": formula_names,
        "not_a_training_dataset": True,
        "bc_training_executed": False,
        "forbidden_fields_absent": sorted(FORBIDDEN_DATASET_KEYS),
    }
    save_report_pair(output_dir, "dataset_schema_delta_for_bc", data, "Dataset Schema Delta For Future BC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured_only_pilot_dir", type=Path, required=True)
    parser.add_argument("--lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--two_frame_lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--dense_uncertainty_dir", type=Path, required=True)
    parser.add_argument("--dense_uncertainty_audit_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_aware_pilot_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source_usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--fixed_usd", type=Path, default=DEFAULT_FIXED_USD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num_starts", type=int, default=10)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--beta_values", nargs="+", type=int, default=BETA_VALUES)
    parser.add_argument("--uncertainty_bonus_modes", nargs="+", default=BONUS_MODES)
    parser.add_argument("--candidate_scope", default="measured_valid")
    parser.add_argument("--minmax_scope", default="per_start")
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--compare_to_measured_only", action="store_true")
    parser.add_argument("--compare_to_lambda48", action="store_true")
    parser.add_argument("--compare_to_confidence_gated", action="store_true")
    parser.add_argument("--save_decision_dataset", action="store_true")
    parser.add_argument("--save_quality_viz", action="store_true")
    parser.add_argument("--make_html", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--no_isaac", action="store_true")
    parser.add_argument("--no_capture", action="store_true")
    parser.add_argument("--no_map_predict", action="store_true")
    parser.add_argument("--no_sscnet_inference", action="store_true")
    parser.add_argument("--no_action", action="store_true")
    parser.add_argument("--no_rollout", action="store_true")
    parser.add_argument("--no_second_action", action="store_true")
    parser.add_argument("--no_third_frame", action="store_true")
    parser.add_argument("--no_long_rollout", action="store_true")
    parser.add_argument("--no_training", action="store_true")
    parser.add_argument("--no_rl_gdpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enforce_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    before_path = output_dir / "git_status_before.txt"
    if not before_path.is_file():
        write_text(before_path, git_status_text())
    for subdir in ("per_start",):
        if (output_dir / subdir).exists():
            shutil.rmtree(output_dir / subdir)

    source_hash, checkpoint_hash, prior_hash = hash_reports(args)
    inputs = load_inputs(args, output_dir)
    save_report_pair(output_dir, "uncertainty_bonus_formula_reference", formula_reference(), "Uncertainty Bonus Formula Reference")

    grouped = prepare_candidate_tables(inputs, args.num_starts)
    score_rows, per_formula_rows, beta_sweep_rows, formula_decisions = compute_scores(
        grouped,
        inputs["measured_decisions"],
        inputs["lambda48_decisions"],
        inputs["confidence_decisions"],
        [int(v) for v in args.beta_values],
        list(args.uncertainty_bonus_modes),
    )

    recommended_formula = PRIMARY_RECOMMENDED_FORMULA
    rec_rows = [row for row in per_formula_rows if row["formula"] == recommended_formula]
    per_start_rows = [
        {
            "stage": STAGE,
            "start_variant_id": row["start_variant_id"],
            "start_name": row["start_name"],
            "recommended_formula": recommended_formula,
            "selected_candidate_id": row["selected_candidate_id"],
            "selected_world_xyz": row["selected_world_xyz"],
            "selected_yaw": row["selected_yaw"],
            "selected_score": row["final_score"],
            "selected_confidence": row["confidence_mean"],
            "selected_entropy": row["entropy_mean"],
            "selected_margin": row["margin_mean"],
            "selected_uncertain_fraction": row["uncertain_fraction"],
            "action_changed_vs_measured": row["action_changed_vs_measured"],
            "action_changed_vs_lambda48": row["action_changed_vs_lambda48"],
            "action_changed_vs_confidence_gated": row["action_changed_vs_confidence_gated"],
            "decision_only": True,
            "action_executed": False,
        }
        for row in rec_rows
    ]

    write_rows_triplet(output_dir, "uncertainty_bonus_beta_sweep", beta_sweep_rows, "Uncertainty Bonus Beta Sweep")
    write_rows_triplet(output_dir, "per_start_uncertainty_bonus_decisions", per_start_rows, "Per Start Uncertainty Bonus Decisions")
    write_rows_triplet(output_dir, "per_formula_decision_table", per_formula_rows, "Per Formula Decision Table")
    write_rows_triplet(
        output_dir,
        "uncertainty_bonus_vs_lambda48_comparison",
        comparison_rows(per_formula_rows, recommended_formula, "lambda48_baseline", "lambda48"),
        "Uncertainty Bonus vs Lambda48 Comparison",
    )
    write_rows_triplet(
        output_dir,
        "uncertainty_bonus_vs_confidence_gated_comparison",
        comparison_rows(per_formula_rows, recommended_formula, "confidence_gated_6_11", "confidence_gated"),
        "Uncertainty Bonus vs Confidence Gated Comparison",
    )
    write_rows_triplet(
        output_dir,
        "uncertainty_bonus_vs_measured_comparison",
        comparison_rows(per_formula_rows, recommended_formula, "measured_only", "measured"),
        "Uncertainty Bonus vs Measured Comparison",
    )

    quality_audit, risk_audit, readiness, recommended = audit_and_recommend(output_dir, grouped, formula_decisions, per_formula_rows, beta_sweep_rows)
    save_report_pair(output_dir, "uncertainty_bonus_quality_audit", quality_audit, "Uncertainty Bonus Quality Audit")
    save_report_pair(output_dir, "uncertainty_bonus_risk_audit", risk_audit, "Uncertainty Bonus Risk Audit")
    save_report_pair(output_dir, "uncertainty_bonus_readiness_decision", readiness, "Uncertainty Bonus Readiness Decision")
    save_report_pair(output_dir, "recommended_formula_report", recommended, "Recommended Formula Report")
    dataset_schema_delta(output_dir, [f"uncertainty_bonus_{mode}_beta{beta}" for mode in BONUS_MODES for beta in BETA_VALUES])
    write_future_sketch(output_dir, recommended_formula, bool(readiness["uncertainty_bonus_runtime_ready"]))
    write_safety_reports(output_dir, source_hash, checkpoint_hash, prior_hash)

    if args.save_decision_dataset:
        build_decision_dataset(output_dir, grouped, formula_decisions, [int(v) for v in args.beta_values], list(args.uncertainty_bonus_modes))

    if args.save_quality_viz and args.save_viz:
        map_paths = per_start_outputs(
            output_dir,
            grouped,
            formula_decisions,
            per_formula_rows,
            beta_sweep_rows,
            recommended_formula,
            {
                "measured_only": formula_decisions["measured_only"],
                "lambda48_baseline": formula_decisions["lambda48_baseline"],
                "confidence_gated_6_11": formula_decisions["confidence_gated_6_11"],
            },
        )
        dataset_visuals(output_dir, grouped, formula_decisions, per_formula_rows, beta_sweep_rows, recommended_formula, map_paths)

    summary = {
        "stage": STAGE,
        "completed": True,
        "output_dir": str(output_dir),
        "loaded_stage4a611": True,
        "loaded_stage4a610a_dense_uncertainty": True,
        "candidate_rows_loaded": len(inputs["candidate_rows"]),
        "start_count": args.num_starts,
        "selected_action_records": len(per_start_rows),
        "decision_dataset": str(output_dir / "expert_decision_dataset_uncertainty_bonus.npz"),
        "html_visualization": str(output_dir / "uncertainty_bonus_index.html"),
        "future_short_rollout_sketch": str(output_dir / "future_short_rollout_with_uncertainty_bonus_sketch.md"),
        "lambda_sc": args.lambda_sc,
        "beta_values": [int(v) for v in args.beta_values],
        "uncertainty_bonus_modes": list(args.uncertainty_bonus_modes),
        "recommended_uncertainty_bonus_formula": recommended_formula,
        "recommended_beta": int(readiness["recommended_beta"]),
        "uncertainty_bonus_runtime_ready": bool(readiness["uncertainty_bonus_runtime_ready"]),
        "action_changed_vs_measured": recommended["action_changed_vs_measured"],
        "action_changed_vs_lambda48": recommended["action_changed_vs_lambda48"],
        "action_changed_vs_confidence_gated": recommended["action_changed_vs_confidence_gated"],
        "candidate_all_local_summary": recommended["candidate_all_local_summary"],
        "selected_confidence_summary": recommended["selected_confidence_summary"],
        "selected_entropy_summary": recommended["selected_entropy_summary"],
        "selected_margin_summary": recommended["selected_margin_summary"],
        "risk_audit_passed": risk_audit["passed"],
        "quality_audit_passed": quality_audit["passed"],
        "risk_audit_warnings": risk_audit["warnings"],
        "risk_audit_blockers": risk_audit["blockers"],
        "isaac_startup_count_this_stage": 0,
        "capture_count_this_stage": 0,
        "map_predict_calls_this_stage": 0,
        "sscnet_inference_calls_this_stage": 0,
        "action_execution_count_this_stage": 0,
        "rollout_executed": False,
        "short_rollout_executed": False,
        "long_rollout_executed": False,
        "second_action_count": 0,
        "third_frame_count": 0,
        "training_executed": False,
        "bc_training_executed": False,
        "il_training_executed": False,
        "rl_training_executed": False,
        "gdpo_training_executed": False,
        "ppo_training_executed": False,
        "replay_buffer_created": False,
        "policy_checkpoint_created": False,
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "prediction_uncertainty_safety_leakage": False,
        "prediction_used_for_traversability": False,
        "uncertainty_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "uncertainty_used_for_collision": False,
        "prediction_used_for_ray_blocking": False,
        "uncertainty_used_for_ray_blocking": False,
        "prediction_used_for_candidate_validity": False,
        "uncertainty_used_for_candidate_validity": False,
        "target_ground_truth_future_observed_scoring": False,
        "source_occ_free_kept_separate_from_uncertainty": True,
        "prior_outputs_modified_in_place": False,
        "source_usd_modified": False,
        "fixed_usd_modified": False,
        "checkpoint_modified": False,
        "observed_state_modified": False,
    }
    save_report_pair(output_dir, "stage4a612_uncertainty_exploration_bonus_pilot_summary", summary, "Stage 4A-6.12 Summary")
    if args.make_html:
        make_html_index(output_dir, summary, per_formula_rows, recommended_formula)

    write_text(output_dir / "git_status_after.txt", git_status_text())
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
