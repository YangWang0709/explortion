#!/usr/bin/env python3
"""Stage 4A-6.10 prediction-derived uncertainty offline audit.

This script only reads existing Stage 4A-6.8 / 6.9 artifacts. It never starts
Isaac, never reruns map_predict or SSCNet inference, and never writes
prediction data back into observed_state. If dense probability/confidence
artifacts are unavailable, it produces a summary-only limited audit instead of
fabricating candidate-level uncertainty.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
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
    os.environ.setdefault(_key, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
SOURCE_USD = WORKSPACE / "building_scene.usd"
SOURCE_OBSERVED_STATE = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix/observed_state_final.npy"
STAGE = "Stage 4A-6.10-prediction-uncertainty-offline-audit"
MODE = "summary_only_limited"
NA = "not_available_summary_only"
BLOCKED = "blocked_missing_dense_prediction_probability_fields"

CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / "README.md",
    WORKSPACE / "ARTIFACTS.md",
    WORKSPACE / "ENVIRONMENT.md",
    WORKSPACE / "GIT_INITIALIZATION_REPORT.md",
]

REQUIRED_STAGE68 = [
    "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
    "expert_dataset.npz",
    "expert_dataset_manifest.jsonl",
    "per_sample_summary.csv",
    "lambda48_decisions.csv",
    "measured_shadow_decisions.csv",
    "map_predict_summary.json",
    "prediction_safety_audit.json",
    "expert_data_quality_audit.json",
    "stage4a68_vs_stage4a67_comparison.json",
]

REQUIRED_STAGE69 = [
    "stage4a69_bounded_two_frame_lambda48_pilot_summary.json",
    "expert_dataset_two_frame.npz",
    "expert_dataset_manifest.jsonl",
    "per_start_summary.csv",
    "per_frame_summary.csv",
    "frame1_lambda48_decisions.csv",
    "frame2_lambda48_diagnostic_decisions.csv",
    "map_predict_summary.json",
    "prediction_safety_audit.json",
    "expert_data_quality_audit.json",
    "two_frame_stability_audit.json",
]

DENSE_FIELD_HINTS = (
    "class_prob",
    "prob",
    "logit",
    "confidence",
    "entropy",
    "margin",
    "free_prob",
    "occupied_prob",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(data), handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fieldnames = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, include_hash: bool = True) -> dict[str, Any]:
    exists = path.exists()
    stat = path.stat() if exists else None
    return {
        "path": str(path),
        "exists": exists,
        "is_file": path.is_file(),
        "size_bytes": int(stat.st_size) if stat else None,
        "mtime_ns": int(stat.st_mtime_ns) if stat else None,
        "sha256": sha256_file(path) if include_hash and path.is_file() else None,
    }


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


def markdown_kv(title: str, rows: dict[str, Any]) -> str:
    lines = [f"# {title}", "", "| Field | Value |", "| --- | --- |"]
    for key, value in rows.items():
        if isinstance(value, (dict, list)):
            rendered = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        lines.append(f"| `{key}` | {html.escape(rendered)} |")
    lines.append("")
    return "\n".join(lines)


def markdown_rows(title: str, rows: list[dict[str, Any]], max_rows: int = 30) -> str:
    lines = [f"# {title}", ""]
    if not rows:
        lines.extend(["No rows.", ""])
        return "\n".join(lines)
    fields: list[str] = []
    for row in rows[:max_rows]:
        for key in row:
            if key not in fields:
                fields.append(key)
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows[:max_rows]:
        vals = []
        for key in fields:
            value = row.get(key, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True)
            vals.append(html.escape(str(value)))
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"\nShowing {max_rows} of {len(rows)} rows.")
    lines.append("")
    return "\n".join(lines)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def as_int(value: Any) -> int | None:
    v = as_float(value)
    return int(v) if v is not None else None


def summarize(vals: list[float]) -> dict[str, Any]:
    clean = np.asarray([v for v in vals if v is not None and math.isfinite(v)], dtype=float)
    if clean.size == 0:
        return {
            "count": 0,
            "min": NA,
            "max": NA,
            "mean": NA,
            "p10": NA,
            "p50": NA,
            "p90": NA,
        }
    return {
        "count": int(clean.size),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p50": float(np.percentile(clean, 50)),
        "p90": float(np.percentile(clean, 90)),
    }


def pearson(x_vals: list[float], y_vals: list[float]) -> dict[str, Any]:
    pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return {"status": "insufficient_pairs", "n": len(pairs), "pearson": NA}
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return {"status": "degenerate_zero_variance", "n": int(x.size), "pearson": NA}
    return {"status": "computed", "n": int(x.size), "pearson": float(np.corrcoef(x, y)[0, 1])}


def rank_value(values: np.ndarray, valid_mask: np.ndarray, selected_index: int, descending: bool) -> int | str:
    if selected_index < 0 or selected_index >= values.shape[0] or not bool(valid_mask[selected_index]):
        return "selected_index_invalid_or_unmasked"
    valid_values = values[valid_mask.astype(bool)]
    selected = float(values[selected_index])
    if descending:
        return int(1 + np.sum(valid_values > selected))
    return int(1 + np.sum(valid_values < selected))


def row_key(stage_alias: str, start_id: int | None, frame_id: int | None, sample_index: int | None = None) -> str:
    if stage_alias == "stage4a68":
        return f"{stage_alias}:sample:{sample_index if sample_index is not None else start_id}"
    return f"{stage_alias}:start:{start_id}:frame:{frame_id}"


def load_context_manifest() -> tuple[dict[str, Any], dict[str, str]]:
    text_by_path: dict[str, str] = {}
    rows = []
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        text_by_path[str(path)] = text
        rows.append(
            {
                **file_record(path, include_hash=True),
                "line_count": text.count("\n") + (0 if text.endswith("\n") else 1),
                "loaded": True,
            }
        )
    manifest = {
        "stage": STAGE,
        "loaded_at_utc": utc_now(),
        "context_files_loaded": rows,
        "context_loaded": True,
    }
    return manifest, text_by_path


def dataset_npz_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        keys = sorted(data.files)
        shapes = {key: [int(v) for v in data[key].shape] for key in data.files}
        dtypes = {key: str(data[key].dtype) for key in data.files}
        dense_keys = [key for key in keys if any(hint in key.lower() for hint in DENSE_FIELD_HINTS)]
    return {"keys": keys, "shapes": shapes, "dtypes": dtypes, "dense_probability_like_keys": dense_keys}


def load_stage_manifest(stage_dir: Path, required_files: list[str], stage_alias: str) -> dict[str, Any]:
    files = []
    missing = []
    for name in required_files:
        path = stage_dir / name
        if not path.is_file():
            missing.append(name)
        files.append(file_record(path, include_hash=True))
    if missing:
        raise FileNotFoundError(f"Missing required {stage_alias} files: {missing}")

    summary_name = required_files[0]
    safety = load_json(stage_dir / "prediction_safety_audit.json")
    quality = load_json(stage_dir / "expert_data_quality_audit.json")
    dataset_name = "expert_dataset.npz" if stage_alias == "stage4a68" else "expert_dataset_two_frame.npz"
    map_predict = load_json(stage_dir / "map_predict_summary.json")
    manifest_lines = load_jsonl(stage_dir / "expert_dataset_manifest.jsonl")
    npz_info = dataset_npz_summary(stage_dir / dataset_name)
    summary = load_json(stage_dir / summary_name)

    complete_checks = {
        "completed": bool(summary.get("completed")),
        "prediction_safety_passed": bool(safety.get("passed")),
        "expert_data_quality_passed": bool(quality.get("passed")),
        "prediction_writeback": bool(safety.get("prediction_writeback")),
        "prediction_traversability_use": bool(safety.get("prediction_traversability_use")),
        "prediction_collision_use": bool(safety.get("prediction_collision_use")),
        "prediction_ray_blocking_use": bool(safety.get("prediction_ray_blocking_use")),
        "prediction_candidate_validity_use": bool(safety.get("prediction_candidate_validity_use")),
        "prediction_edge_validity_use": bool(safety.get("prediction_edge_validity_use")),
        "target_ground_truth_use": bool(safety.get("target_ground_truth_use")),
        "future_observed_scoring_use": bool(safety.get("future_observed_scoring_use")),
    }
    if stage_alias == "stage4a68":
        complete_checks.update(
            {
                "sample_count": int(summary.get("sample_count", -1)),
                "capture_count": int(summary.get("capture_count", -1)),
                "map_predict_calls": int(summary.get("map_predict_calls", -1)),
                "exactly_one_action_per_start": bool(summary.get("exactly_one_action_per_start")),
                "continuous_rollout_executed": bool(summary.get("continuous_rollout_executed")),
            }
        )
        if not complete_checks["completed"] or complete_checks["sample_count"] != 10 or complete_checks["capture_count"] != 10:
            raise ValueError("Stage 4A-6.8 completion/count checks failed")
        if complete_checks["map_predict_calls"] != 10 or not complete_checks["exactly_one_action_per_start"]:
            raise ValueError("Stage 4A-6.8 map_predict/action checks failed")
        if complete_checks["continuous_rollout_executed"]:
            raise ValueError("Stage 4A-6.8 unexpectedly records rollout")
    else:
        complete_checks.update(
            {
                "start_count": int(summary.get("start_count", -1)),
                "frame_count": int(summary.get("frame_count", -1)),
                "capture_count": int(summary.get("capture_count", -1)),
                "map_predict_calls": int(summary.get("map_predict_calls", -1)),
                "executed_action_count": int(summary.get("executed_action_count", -1)),
                "exactly_one_action_per_start": bool(summary.get("exactly_one_action_per_start")),
                "second_action_count": int(summary.get("second_action_count", -1)),
                "third_frame_count": int(summary.get("third_frame_count", -1)),
                "continuous_rollout_executed": bool(summary.get("continuous_rollout_executed")),
                "long_rollout_executed": bool(summary.get("long_rollout_executed")),
            }
        )
        if not complete_checks["completed"] or complete_checks["start_count"] != 10 or complete_checks["frame_count"] != 20:
            raise ValueError("Stage 4A-6.9 completion/count checks failed")
        if complete_checks["capture_count"] != 20 or complete_checks["map_predict_calls"] != 20:
            raise ValueError("Stage 4A-6.9 capture/map_predict count checks failed")
        if (
            complete_checks["second_action_count"] != 0
            or complete_checks["third_frame_count"] != 0
            or complete_checks["continuous_rollout_executed"]
            or complete_checks["long_rollout_executed"]
        ):
            raise ValueError("Stage 4A-6.9 bounded/no-rollout checks failed")

    if not complete_checks["prediction_safety_passed"] or not complete_checks["expert_data_quality_passed"]:
        raise ValueError(f"{stage_alias} safety/quality checks failed")
    forbidden_true = [
        key
        for key in (
            "prediction_writeback",
            "prediction_traversability_use",
            "prediction_collision_use",
            "prediction_ray_blocking_use",
            "prediction_candidate_validity_use",
            "prediction_edge_validity_use",
            "target_ground_truth_use",
            "future_observed_scoring_use",
        )
        if complete_checks[key]
    ]
    if forbidden_true:
        raise ValueError(f"{stage_alias} forbidden prediction safety flags are true: {forbidden_true}")

    map_predict_rows = map_predict.get("samples") or map_predict.get("frames") or []
    summary_only_count = sum(1 for row in map_predict_rows if bool(row.get("prediction_summary_only")))
    removed_npz = []
    for row in map_predict_rows:
        removed_npz.extend(row.get("prediction_array_npz_removed_after_summary") or [])

    return {
        "stage": stage_alias,
        "stage_dir": str(stage_dir),
        "loaded_at_utc": utc_now(),
        "required_files": files,
        "manifest_row_count": len(manifest_lines),
        "npz": npz_info,
        "summary_keys": sorted(summary.keys()),
        "complete_checks": complete_checks,
        "map_predict_row_count": len(map_predict_rows),
        "prediction_summary_only_count": summary_only_count,
        "removed_prediction_npz_paths_recorded": removed_npz,
        "loaded": True,
    }


def inventory_prediction_artifacts(stage68_dir: Path, stage69_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_alias, root in (("stage4a68", stage68_dir), ("stage4a69", stage69_dir)):
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            rel = path.relative_to(root)
            category = None
            if "prediction_summary" in name or name == "map_predict_summary.json" or name == "map_predict_summary.csv":
                category = "prediction_summary"
            elif "prediction_safety" in name:
                category = "prediction_safety"
            elif "top_candidates" in name or "candidate" in name:
                category = "candidate_table_or_visual"
            elif "prediction_overlay" in name or "prediction_density" in name:
                category = "prediction_visual"
            elif path.suffix.lower() == ".npz" and any(hint in name for hint in ("prediction", "prob", "logit", "confidence")):
                category = "dense_prediction_npz"
            if category is None:
                continue
            dense_like = category == "dense_prediction_npz" or any(hint in name for hint in DENSE_FIELD_HINTS)
            rows.append(
                {
                    "stage": stage_alias,
                    "relative_path": str(rel),
                    "path": str(path),
                    "category": category,
                    "extension": path.suffix.lower(),
                    "size_bytes": int(path.stat().st_size),
                    "dense_probability_like": bool(dense_like and path.suffix.lower() in (".npz", ".npy")),
                    "exists": True,
                }
            )
    return rows


def frame_rows_from_map_predict(stage_dir: Path, stage_alias: str) -> list[dict[str, Any]]:
    data = load_json(stage_dir / "map_predict_summary.json")
    rows = data.get("samples") or data.get("frames") or []
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        start_id = as_int(row.get("start_variant_id"))
        sample_index = as_int(row.get("sample_index"))
        frame_id = as_int(row.get("frame_id")) if row.get("frame_id") is not None else 1
        frame_key = row_key(stage_alias, start_id, frame_id, sample_index)
        out.append(
            {
                "stage": stage_alias,
                "frame_key": frame_key,
                "sample_index": sample_index if sample_index is not None else "",
                "start_variant_id": start_id if start_id is not None else sample_index,
                "frame_id": frame_id,
                "frame_prediction_valid_count": as_int(row.get("prediction_valid_count")),
                "frame_predicted_unmeasured_count": as_int(row.get("predicted_unmeasured_count")),
                "frame_predicted_occupied_count": as_int(row.get("predicted_occupied_count")),
                "frame_predicted_free_count": as_int(row.get("predicted_free_count")),
                "prediction_density": as_float(row.get("prediction_density")),
                "predicted_unmeasured_density": as_float(row.get("predicted_unmeasured_density")),
                "frame_confidence_mean": NA,
                "frame_confidence_p10": NA,
                "frame_confidence_p50": NA,
                "frame_confidence_p90": NA,
                "frame_entropy_mean": NA,
                "frame_entropy_p90": NA,
                "frame_margin_mean": NA,
                "frame_low_conf_fraction_0p7": NA,
                "frame_high_entropy_fraction_0p7": NA,
                "frame_uncertainty_mode": MODE,
                "dense_prediction_available": False,
                "candidate_uncertainty_available": False,
                "missing_uncertainty_fields": "confidence, entropy, margin, probability/logit tensors",
                "prediction_summary_only": bool(row.get("prediction_summary_only")),
                "prediction_array_npz_removed_after_summary": ";".join(row.get("prediction_array_npz_removed_after_summary") or []),
            }
        )
    return out


def load_branch_maps(stage68_dir: Path, stage69_dir: Path) -> dict[str, str]:
    branches: dict[str, str] = {}
    for row in read_csv(stage68_dir / "lambda48_decisions.csv"):
        sample = as_int(row.get("sample_index"))
        start = as_int(row.get("start_variant_id"))
        branches[row_key("stage4a68", start, 1, sample)] = row.get("branch_classification", "unknown")
    for name in ("frame1_lambda48_decisions.csv", "frame2_lambda48_diagnostic_decisions.csv"):
        for row in read_csv(stage69_dir / name):
            start = as_int(row.get("start_variant_id"))
            frame_id = as_int(row.get("frame_id"))
            branches[row_key("stage4a69", start, frame_id)] = row.get("branch_classification", "unknown")
    return branches


def load_top_candidate_rows(stage68_dir: Path, stage69_dir: Path, branches: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for csv_path in sorted((stage68_dir / "samples").glob("start_*/top_candidates_*.csv")):
        start = as_int(csv_path.parent.name.split("_")[-1])
        sample = start
        key = row_key("stage4a68", start, 1, sample)
        for raw in read_csv(csv_path):
            rows.append(candidate_base_row(raw, "stage4a68", key, start, 1, sample, branches.get(key, "unknown"), csv_path))
    for csv_path in sorted((stage69_dir / "samples").glob("start_*/frame*_top_candidates.csv")):
        start = as_int(csv_path.parent.name.split("_")[-1])
        frame_id = 1 if "frame1_" in csv_path.name else 2
        key = row_key("stage4a69", start, frame_id)
        for raw in read_csv(csv_path):
            rows.append(candidate_base_row(raw, "stage4a69", key, start, frame_id, None, branches.get(key, "unknown"), csv_path))
    return rows


def candidate_base_row(
    raw: dict[str, str],
    stage_alias: str,
    key: str,
    start: int | None,
    frame_id: int,
    sample: int | None,
    branch: str,
    source_path: Path,
) -> dict[str, Any]:
    source_occ = as_float(raw.get("source_occ_free"))
    final_score = as_float(raw.get("final_score_lambda48") or raw.get("value_lambda48"))
    measured_score = as_float(raw.get("final_score_measured") or raw.get("measured_score"))
    cost = as_float(raw.get("cost_s") or raw.get("path_cost_m") or raw.get("path_cost"))
    return {
        "stage": stage_alias,
        "frame_key": key,
        "sample_index": sample if sample is not None else "",
        "start_variant_id": start,
        "frame_id": frame_id,
        "candidate_id": as_int(raw.get("candidate_id")),
        "rank": as_int(raw.get("rank")),
        "branch_classification": branch,
        "candidate_source": raw.get("candidate_source", ""),
        "valid_mask": bool(str(raw.get("astar_reachable", "")).lower() == "true"),
        "world": raw.get("world", ""),
        "yaw_rad": as_float(raw.get("yaw_rad")),
        "gain_exp": as_float(raw.get("gain_exp")),
        "cost": cost,
        "path_cost": cost,
        "path_cost_m": as_float(raw.get("path_cost_m")),
        "source_occ_free": source_occ,
        "lambda48_score": final_score,
        "measured_score": measured_score,
        "visible_count": as_int(raw.get("visible_count")),
        "measured_visible_count": as_int(raw.get("measured_visible_count")),
        "predicted_unmeasured_visible_count": as_int(raw.get("predicted_unmeasured_visible_count")),
        "source_occ_free_minmax": as_float(raw.get("source_occ_free_minmax")),
        "lambda48_bonus": as_float(raw.get("lambda48_bonus")),
        "selected_action_id": "",
        "candidate_confidence_mean": NA,
        "candidate_confidence_min": NA,
        "candidate_confidence_p10": NA,
        "candidate_confidence_p50": NA,
        "candidate_confidence_p90": NA,
        "candidate_uncertainty_conf_mean": NA,
        "candidate_uncertainty_conf_max": NA,
        "candidate_entropy_mean": NA,
        "candidate_entropy_max": NA,
        "candidate_entropy_p90": NA,
        "candidate_margin_mean": NA,
        "candidate_margin_min": NA,
        "candidate_low_conf_count_0p5": NA,
        "candidate_low_conf_count_0p7": NA,
        "candidate_low_conf_count_0p9": NA,
        "candidate_high_entropy_count_0p5": NA,
        "candidate_high_entropy_count_0p7": NA,
        "candidate_low_margin_count_0p1": NA,
        "candidate_low_margin_count_0p2": NA,
        "candidate_uncertain_voxel_count": NA,
        "candidate_visible_prediction_voxel_count": NA,
        "candidate_uncertain_fraction": NA,
        "source_occ_free_confidence_weighted": NA,
        "source_occ_free_entropy_weighted": NA,
        "source_occ_free_uncertainty_penalized": NA,
        "uncertainty_bonus_shadow_score_beta1": NA,
        "uncertainty_penalty_shadow_score_beta1": NA,
        "confidence_gated_lambda48_shadow_score": NA,
        "candidate_uncertainty_available": False,
        "uncertainty_mode": MODE,
        "missing_uncertainty_fields": "candidate visibility voxel probabilities/confidence/entropy/margin",
        "source_table": str(source_path),
    }


def selection_rows(stage68_dir: Path, stage69_dir: Path, branches: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(stage68_dir / "expert_dataset.npz", allow_pickle=False) as data:
        rows.extend(
            selection_rows_from_arrays(
                stage_alias="stage4a68",
                frame_id=1,
                start_ids=data["start_variant_id"],
                valid_masks=data["valid_mask"],
                lambda_indices=data["expert_action_index_lambda48"],
                measured_indices=data["expert_action_index_measured_shadow"],
                gain_exp=data["gain_exp"],
                source_occ_free=data["source_occ_free"],
                path_cost=data["path_cost"],
                lambda_scores=data["final_score_lambda48"],
                measured_scores=data["final_score_measured"],
                branches=branches,
                sample_indices=list(range(int(data["start_variant_id"].shape[0]))),
            )
        )
    with np.load(stage69_dir / "expert_dataset_two_frame.npz", allow_pickle=False) as data:
        start_ids = data["start_variant_id"]
        rows.extend(
            selection_rows_from_arrays(
                stage_alias="stage4a69",
                frame_id=1,
                start_ids=start_ids,
                valid_masks=data["frame1_candidate_mask"],
                lambda_indices=data["frame1_lambda48_action_index"],
                measured_indices=data["frame1_measured_action_index"],
                gain_exp=data["frame1_gain_exp"],
                source_occ_free=data["frame1_source_occ_free"],
                path_cost=data["frame1_path_cost"],
                lambda_scores=data["frame1_final_score_lambda48"],
                measured_scores=data["frame1_measured_scores"],
                branches=branches,
                sample_indices=None,
            )
        )
        rows.extend(
            selection_rows_from_arrays(
                stage_alias="stage4a69",
                frame_id=2,
                start_ids=start_ids,
                valid_masks=data["frame2_candidate_mask"],
                lambda_indices=data["frame2_lambda48_diagnostic_action_index"],
                measured_indices=data["frame2_measured_diagnostic_action_index"],
                gain_exp=data["frame2_gain_exp"],
                source_occ_free=data["frame2_source_occ_free"],
                path_cost=data["frame2_path_cost"],
                lambda_scores=data["frame2_final_score_lambda48"],
                measured_scores=data["frame2_measured_scores"],
                branches=branches,
                sample_indices=None,
            )
        )
    return rows


def selection_rows_from_arrays(
    *,
    stage_alias: str,
    frame_id: int,
    start_ids: np.ndarray,
    valid_masks: np.ndarray,
    lambda_indices: np.ndarray,
    measured_indices: np.ndarray,
    gain_exp: np.ndarray,
    source_occ_free: np.ndarray,
    path_cost: np.ndarray,
    lambda_scores: np.ndarray,
    measured_scores: np.ndarray,
    branches: dict[str, str],
    sample_indices: list[int] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i in range(int(start_ids.shape[0])):
        start = int(start_ids[i])
        sample = sample_indices[i] if sample_indices is not None else None
        key = row_key(stage_alias, start, frame_id, sample)
        branch = branches.get(key, "unknown")
        valid = valid_masks[i].astype(bool)
        for action_label, selected_index in (
            ("lambda48", int(lambda_indices[i])),
            ("measured_shadow", int(measured_indices[i])),
        ):
            if selected_index < 0 or selected_index >= gain_exp.shape[1]:
                selected_valid = False
                selected_gain = selected_source = selected_cost = selected_lambda_score = selected_measured_score = None
            else:
                selected_valid = bool(valid[selected_index])
                selected_gain = float(gain_exp[i, selected_index])
                selected_source = float(source_occ_free[i, selected_index])
                selected_cost = float(path_cost[i, selected_index])
                selected_lambda_score = float(lambda_scores[i, selected_index])
                selected_measured_score = float(measured_scores[i, selected_index])
            rows.append(
                {
                    "stage": stage_alias,
                    "frame_key": key,
                    "sample_index": sample if sample is not None else "",
                    "start_variant_id": start,
                    "frame_id": frame_id,
                    "action_type": action_label,
                    "selected_action_id": selected_index,
                    "selected_valid": selected_valid,
                    "branch_classification": branch,
                    "selected_gain_exp": selected_gain,
                    "selected_source_occ_free": selected_source,
                    "selected_path_cost": selected_cost,
                    "selected_lambda48_score": selected_lambda_score,
                    "selected_measured_score": selected_measured_score,
                    "selected_lambda48_confidence_mean": NA,
                    "selected_lambda48_entropy_mean": NA,
                    "selected_lambda48_margin_mean": NA,
                    "selected_lambda48_uncertain_fraction": NA,
                    "selected_measured_shadow_confidence_mean": NA,
                    "selected_measured_shadow_entropy_mean": NA,
                    "selected_uncertainty_rank": NA,
                    "selected_source_occ_free_rank": rank_value(source_occ_free[i], valid, selected_index, descending=True),
                    "selected_gain_exp_rank": rank_value(gain_exp[i], valid, selected_index, descending=True),
                    "selected_cost_rank": rank_value(path_cost[i], valid, selected_index, descending=False),
                    "did_lambda48_select_high_uncertainty_candidate": NA,
                    "did_lambda48_avoid_high_uncertainty_candidate": NA,
                    "uncertainty_correlates_with_source_occ_free": NA,
                    "uncertainty_correlates_with_action_change": NA,
                    "uncertainty_mode": MODE,
                    "candidate_uncertainty_available": False,
                }
            )
    return rows


def stage68_stage69_comparison(frame_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames = {row["frame_key"]: row for row in frame_rows}
    selected_lambda = {
        (row["stage"], row["start_variant_id"], row["frame_id"]): row
        for row in selected_rows
        if row["action_type"] == "lambda48"
    }
    out: list[dict[str, Any]] = []
    starts = sorted({row["start_variant_id"] for row in selected_rows if row["start_variant_id"] != ""})
    for start in starts:
        k68 = row_key("stage4a68", int(start), 1, int(start))
        k691 = row_key("stage4a69", int(start), 1)
        k692 = row_key("stage4a69", int(start), 2)
        for label, other_key, other_tuple in (
            ("stage4a68_vs_stage4a69_frame1", k691, ("stage4a69", int(start), 1)),
            ("stage4a68_vs_stage4a69_frame2", k692, ("stage4a69", int(start), 2)),
        ):
            f68 = frames.get(k68, {})
            fother = frames.get(other_key, {})
            s68 = selected_lambda.get(("stage4a68", int(start), 1), {})
            sother = selected_lambda.get(other_tuple, {})
            out.append(
                {
                    "comparison": label,
                    "start_variant_id": start,
                    "dense_uncertainty_available": False,
                    "uncertainty_mode": MODE,
                    "confidence_delta": NA,
                    "entropy_delta": NA,
                    "margin_delta": NA,
                    "prediction_density_delta": delta(fother.get("prediction_density"), f68.get("prediction_density")),
                    "predicted_unmeasured_count_delta": delta(
                        fother.get("frame_predicted_unmeasured_count"), f68.get("frame_predicted_unmeasured_count")
                    ),
                    "lambda48_source_occ_free_delta": delta(
                        sother.get("selected_source_occ_free"), s68.get("selected_source_occ_free")
                    ),
                    "lambda48_action_id_changed": (
                        bool(sother.get("selected_action_id") != s68.get("selected_action_id"))
                        if sother and s68
                        else NA
                    ),
                }
            )
    return out


def delta(a: Any, b: Any) -> float | str:
    fa = as_float(a)
    fb = as_float(b)
    if fa is None or fb is None:
        return NA
    return float(fa - fb)


def source_occ_analysis(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["stage"], int(row["frame_id"]))].append(row)
    out: list[dict[str, Any]] = []
    for (stage, frame_id), rows in sorted(groups.items()):
        source = [as_float(r.get("source_occ_free")) for r in rows]
        gain = [as_float(r.get("gain_exp")) for r in rows]
        cost = [as_float(r.get("path_cost")) for r in rows]
        score = [as_float(r.get("lambda48_score")) for r in rows]
        source_summary = summarize([v for v in source if v is not None])
        out.append(
            {
                "stage": stage,
                "frame_id": frame_id,
                "uncertainty_mode": MODE,
                "candidate_count": len(rows),
                "source_occ_free_mean": source_summary["mean"],
                "source_occ_free_min": source_summary["min"],
                "source_occ_free_max": source_summary["max"],
                "source_occ_free_vs_uncertainty_correlation": NA,
                "source_occ_free_vs_uncertainty_status": BLOCKED,
                "source_occ_free_vs_gain_exp_pearson": pearson(
                    [v for v in source if v is not None], [v for v in gain if v is not None]
                ).get("pearson"),
                "source_occ_free_vs_path_cost_pearson": pearson(
                    [v for v in source if v is not None], [v for v in cost if v is not None]
                ).get("pearson"),
                "source_occ_free_vs_lambda48_score_pearson": pearson(
                    [v for v in source if v is not None], [v for v in score if v is not None]
                ).get("pearson"),
                "high_source_occ_free_low_confidence_candidates": NA,
                "high_uncertainty_low_source_occ_free_candidates": NA,
            }
        )
    return out


def branch_analysis(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(row["stage"], int(row["frame_id"]), str(row["branch_classification"]))].append(row)
    out: list[dict[str, Any]] = []
    for (stage, frame_id, branch), rows in sorted(groups.items()):
        source = summarize([as_float(r.get("source_occ_free")) for r in rows if as_float(r.get("source_occ_free")) is not None])
        gain = summarize([as_float(r.get("gain_exp")) for r in rows if as_float(r.get("gain_exp")) is not None])
        out.append(
            {
                "stage": stage,
                "frame_id": frame_id,
                "branch_classification": branch,
                "candidate_rows": len(rows),
                "uncertainty_mode": MODE,
                "candidate_uncertainty_available": False,
                "confidence_mean": NA,
                "entropy_mean": NA,
                "margin_mean": NA,
                "low_conf_fraction_0p7": NA,
                "high_entropy_fraction_0p7": NA,
                "source_occ_free_mean": source["mean"],
                "source_occ_free_range": f"{source['min']}..{source['max']}",
                "gain_exp_mean": gain["mean"],
                "candidate_all_local_relation": "blocked_for_uncertainty; warning class exists but confidence/entropy fields are absent",
            }
        )
    return out


def frame1_frame2_analysis(frame_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_start: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in frame_rows:
        if row["stage"] != "stage4a69":
            continue
        by_start[int(row["start_variant_id"])][int(row["frame_id"])] = row
    out = []
    for start, frames in sorted(by_start.items()):
        f1 = frames.get(1, {})
        f2 = frames.get(2, {})
        out.append(
            {
                "start_variant_id": start,
                "uncertainty_mode": MODE,
                "confidence_mean_delta_frame2_minus_frame1": NA,
                "entropy_mean_delta_frame2_minus_frame1": NA,
                "margin_mean_delta_frame2_minus_frame1": NA,
                "prediction_density_delta_frame2_minus_frame1": delta(f2.get("prediction_density"), f1.get("prediction_density")),
                "predicted_unmeasured_count_delta_frame2_minus_frame1": delta(
                    f2.get("frame_predicted_unmeasured_count"), f1.get("frame_predicted_unmeasured_count")
                ),
                "predicted_occupied_count_delta_frame2_minus_frame1": delta(
                    f2.get("frame_predicted_occupied_count"), f1.get("frame_predicted_occupied_count")
                ),
                "frame2_uncertainty_increased": NA,
                "frame2_candidate_health_impacted_by_uncertainty": NA,
                "limited_mode_note": "Frame2 uncertainty cannot be measured without dense confidence/probability fields.",
            }
        )
    return out


def shadow_score_audit(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in candidate_rows if as_int(row.get("rank")) == 1]
    rows: list[dict[str, Any]] = []
    for row in selected:
        rows.append(
            {
                "stage": row["stage"],
                "frame_key": row["frame_key"],
                "start_variant_id": row["start_variant_id"],
                "frame_id": row["frame_id"],
                "candidate_id": row["candidate_id"],
                "baseline_lambda48_score": row["lambda48_score"],
                "uncertainty_bonus_shadow_score_beta1": NA,
                "uncertainty_penalty_shadow_score_beta1": NA,
                "confidence_gated_lambda48_shadow_score": NA,
                "baseline_selected_candidate_id": row["candidate_id"],
                "uncertainty_bonus_selected_candidate_id": NA,
                "uncertainty_penalty_selected_candidate_id": NA,
                "confidence_gated_selected_candidate_id": NA,
                "action_change_under_uncertainty_bonus": NA,
                "action_change_under_uncertainty_penalty": NA,
                "action_change_under_confidence_gating": NA,
                "shadow_score_status": BLOCKED,
                "interpretation": "Shadow uncertainty scores were not computed because confidence/entropy/margin fields are missing.",
            }
        )
    return rows


def write_report_triplet(output_dir: Path, stem: str, rows_or_data: Any, md_title: str) -> None:
    if isinstance(rows_or_data, list):
        save_json(output_dir / f"{stem}.json", rows_or_data)
        write_csv(output_dir / f"{stem}.csv", rows_or_data)
        write_text(output_dir / f"{stem}.md", markdown_rows(md_title, rows_or_data))
    elif isinstance(rows_or_data, dict):
        save_json(output_dir / f"{stem}.json", rows_or_data)
        write_text(output_dir / f"{stem}.md", markdown_kv(md_title, rows_or_data))
    else:
        raise TypeError(type(rows_or_data))


def make_scatter(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str, xlabel: str, ylabel: str) -> None:
    xs = [as_float(r.get(x_key)) for r in rows]
    ys = [as_float(r.get(y_key)) for r in rows]
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    plt.figure(figsize=(7, 5))
    if pairs:
        plt.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=24, alpha=0.75, color="#1f77b4")
    else:
        plt.text(0.5, 0.5, "No numeric pairs available", ha="center", va="center")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.figtext(0.02, 0.01, "Limited mode: dense uncertainty fields unavailable; plot uses available score/count fields.", fontsize=8)
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def make_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.8))
    if values:
        plt.bar(labels, values, color="#2a9d8f")
        plt.xticks(rotation=35, ha="right")
    else:
        plt.text(0.5, 0.5, "No numeric values available", ha="center", va="center")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.figtext(0.02, 0.01, "Limited mode: no confidence/entropy/margin tensors were available.", fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def make_blocker_png(path: Path, title: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.axis("off")
    text = (
        f"{title}\n\n"
        "Blocked in Stage 4A-6.10 limited mode.\n"
        "Existing 6.8/6.9 artifacts only contain prediction summary counts and candidate scores.\n"
        "Dense probability/confidence/entropy/margin maps were removed after summary.\n"
        "No dense topdown uncertainty map is fabricated."
    )
    plt.text(0.5, 0.55, text, ha="center", va="center", wrap=True, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def make_visuals(output_dir: Path, frame_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
    make_bar(
        output_dir / "frame_uncertainty_contact_sheet.png",
        [r["frame_key"].replace("stage4a", "s4a") for r in frame_rows],
        [as_float(r.get("frame_predicted_unmeasured_count")) or 0.0 for r in frame_rows],
        "Frame Prediction Summary Counts (Uncertainty Limited)",
        "predicted_unmeasured_count",
    )
    selected_lambda = [r for r in selected_rows if r.get("action_type") == "lambda48"]
    make_bar(
        output_dir / "selected_action_uncertainty_contact_sheet.png",
        [r["frame_key"].replace("stage4a", "s4a") for r in selected_lambda],
        [as_float(r.get("selected_source_occ_free")) or 0.0 for r in selected_lambda],
        "Selected Lambda48 Source Occ Free (Uncertainty Limited)",
        "selected_source_occ_free",
    )
    make_scatter(
        output_dir / "uncertainty_vs_source_occ_free_scatter.png",
        candidate_rows,
        "source_occ_free",
        "predicted_unmeasured_visible_count",
        "Source Occ Free vs Predicted-Unmeasured Visible Count",
        "source_occ_free",
        "predicted_unmeasured_visible_count",
    )
    make_scatter(
        output_dir / "uncertainty_vs_gain_exp_scatter.png",
        candidate_rows,
        "gain_exp",
        "source_occ_free",
        "Gain Exp vs Source Occ Free (Uncertainty Blocked)",
        "gain_exp",
        "source_occ_free",
    )
    make_scatter(
        output_dir / "uncertainty_vs_path_cost_scatter.png",
        candidate_rows,
        "path_cost",
        "source_occ_free",
        "Path Cost vs Source Occ Free (Uncertainty Blocked)",
        "path_cost",
        "source_occ_free",
    )
    make_scatter(
        output_dir / "uncertainty_vs_lambda48_score_scatter.png",
        candidate_rows,
        "lambda48_score",
        "source_occ_free",
        "Lambda48 Score vs Source Occ Free (Uncertainty Blocked)",
        "lambda48_score",
        "source_occ_free",
    )
    branch_means: dict[str, list[float]] = defaultdict(list)
    for row in candidate_rows:
        v = as_float(row.get("source_occ_free"))
        if v is not None:
            branch_means[str(row.get("branch_classification"))].append(v)
    make_bar(
        output_dir / "branch_class_uncertainty_bar.png",
        sorted(branch_means),
        [float(np.mean(branch_means[k])) for k in sorted(branch_means)],
        "Branch Class Mean Source Occ Free (Uncertainty Blocked)",
        "mean source_occ_free",
    )
    f12 = frame1_frame2_analysis(frame_rows)
    make_bar(
        output_dir / "frame1_frame2_uncertainty_delta_bar.png",
        [str(r["start_variant_id"]) for r in f12],
        [as_float(r.get("predicted_unmeasured_count_delta_frame2_minus_frame1")) or 0.0 for r in f12],
        "Frame2 - Frame1 Predicted-Unmeasured Count Delta",
        "delta count",
    )
    plt.figure(figsize=(6, 4.8))
    sel_ids = {(r["frame_key"], r["selected_action_id"]) for r in selected_lambda}
    selected_vals = []
    nonselected_vals = []
    for row in candidate_rows:
        v = as_float(row.get("source_occ_free"))
        if v is None:
            continue
        if (row["frame_key"], row["candidate_id"]) in sel_ids:
            selected_vals.append(v)
        else:
            nonselected_vals.append(v)
    plt.boxplot([selected_vals or [0.0], nonselected_vals or [0.0]], labels=["selected", "nonselected"])
    plt.title("Selected vs Nonselected Source Occ Free (Uncertainty Blocked)")
    plt.ylabel("source_occ_free")
    plt.figtext(0.02, 0.01, "Limited mode: source_occ_free is not uncertainty.", fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(output_dir / "selected_vs_nonselected_uncertainty_boxplot.png", dpi=150)
    plt.close()

    for filename, title in (
        ("high_uncertainty_candidate_examples.png", "High-Uncertainty Candidate Examples"),
        ("low_confidence_warning_map.png", "Low-Confidence Warning Map"),
        ("entropy_topdown_examples.png", "Entropy Topdown Examples"),
        ("confidence_topdown_examples.png", "Confidence Topdown Examples"),
        ("margin_topdown_examples.png", "Margin Topdown Examples"),
    ):
        make_blocker_png(output_dir / filename, title)


def make_html_index(
    output_dir: Path,
    stage69_dir: Path,
    frame_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    readiness: dict[str, Any],
) -> None:
    selected_by_key = {r["frame_key"]: r for r in selected_rows if r["action_type"] == "lambda48"}
    rows = []
    for row in frame_rows:
        if row["stage"] != "stage4a69":
            continue
        start = int(row["start_variant_id"])
        frame_id = int(row["frame_id"])
        sample_dir = stage69_dir / f"samples/start_{start:03d}"
        rgb = sample_dir / f"frame{frame_id}_rgb.png"
        depth = sample_dir / f"frame{frame_id}_depth.npy"
        sel = selected_by_key.get(row["frame_key"], {})
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['frame_key'])}</td>"
            f"<td><a href='{html.escape(os.path.relpath(rgb, output_dir))}'>RGB</a></td>"
            f"<td><a href='{html.escape(os.path.relpath(depth, output_dir))}'>depth npy</a></td>"
            f"<td>{row.get('frame_predicted_unmeasured_count')}</td>"
            f"<td>{row.get('prediction_density')}</td>"
            f"<td>{MODE}</td>"
            f"<td>{sel.get('selected_action_id', '')}</td>"
            f"<td>{sel.get('selected_source_occ_free', '')}</td>"
            f"<td>{NA}</td>"
            f"<td>Dense confidence/entropy/margin unavailable; no map_predict rerun.</td>"
            "</tr>"
        )
    body = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Stage 4A-6.10 Prediction Uncertainty Offline Audit</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border: 1px solid #d7dde5; padding: 6px 8px; text-align: left; }}
    th {{ background: #eef2f7; }}
    .note {{ padding: 12px; background: #fff7e6; border: 1px solid #f4c26b; margin-bottom: 16px; }}
    img {{ max-width: 32%; min-width: 280px; margin: 6px; border: 1px solid #d7dde5; }}
  </style>
</head>
<body>
  <h1>Stage 4A-6.10 Prediction-Derived Uncertainty Offline Audit</h1>
  <div class="note">
    This audit is confidence/probability-derived only, not Bayesian uncertainty,
    not ensemble uncertainty, and not MC-dropout uncertainty. Existing 6.8/6.9
    artifacts are summary-only, so candidate-level confidence, entropy, and
    margin are blocked. No expert action was executed, no map_predict was
    rerun, no SSCNet inference occurred, and no rollout occurred in Stage 4A-6.10.
  </div>
  <h2>Readiness</h2>
  <pre>{html.escape(json.dumps(jsonable(readiness), indent=2, sort_keys=True))}</pre>
  <h2>Stage 4A-6.9 Frame Review</h2>
  <table>
    <tr>
      <th>frame</th><th>RGB</th><th>depth</th><th>predicted unmeasured</th>
      <th>prediction density</th><th>mode</th><th>lambda48 action</th>
      <th>selected source_occ_free</th><th>uncertainty</th><th>interpretation</th>
    </tr>
    {''.join(rows)}
  </table>
  <h2>Plots</h2>
  <p>
    <img src="frame_uncertainty_contact_sheet.png">
    <img src="selected_action_uncertainty_contact_sheet.png">
    <img src="uncertainty_vs_source_occ_free_scatter.png">
    <img src="branch_class_uncertainty_bar.png">
    <img src="frame1_frame2_uncertainty_delta_bar.png">
    <img src="selected_vs_nonselected_uncertainty_boxplot.png">
  </p>
  <h2>Reports</h2>
  <ul>
    <li><a href="uncertainty_readiness_decision.md">readiness decision</a></li>
    <li><a href="dense_uncertainty_blocker.md">dense uncertainty blocker</a></li>
    <li><a href="selected_action_uncertainty_audit.md">selected action audit</a></li>
    <li><a href="future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md">future 6.11 sketch, do not run</a></li>
  </ul>
</body>
</html>
"""
    write_text(output_dir / "uncertainty_overview_index.html", body)


def make_readiness(
    dense_available: bool,
    candidate_uncertainty_rows: int,
    frame_rows: list[dict[str, Any]],
    safety_recheck: dict[str, Any],
) -> dict[str, Any]:
    feature_complete = bool(frame_rows) and bool(safety_recheck.get("passed")) and not dense_available
    return {
        "stage": STAGE,
        "uncertainty_feature_extraction_complete": feature_complete,
        "candidate_level_uncertainty_ready": False,
        "uncertainty_aware_expert_pilot_ready": False,
        "uncertainty_mode": MODE,
        "dense_prediction_available": dense_available,
        "candidate_level_uncertainty_rows": candidate_uncertainty_rows,
        "meaningful_uncertainty_variation_available": False,
        "correlation_analysis_degenerate_or_blocked": True,
        "blockers": [
            "dense probability/confidence/logit fields are absent from Stage 4A-6.8/6.9 artifacts",
            "map_predict_summary records prediction_summary_only=true",
            "map_predict_summary records prediction_array_npz_removed_after_summary paths",
            "candidate visibility voxel probability lists are absent",
            "confidence/entropy/margin cannot be computed without rerunning map_predict, which is prohibited in Stage 4A-6.10",
        ],
        "recommended_next": "Update future map_predict artifact saving to persist dense probability/confidence/entropy/margin fields, then rerun this offline audit. Do not jump to long rollout.",
        "future_stage4a611_sketch_generated": True,
    }


def safety_recheck(stage68_dir: Path, stage69_dir: Path, fixed_usd: Path, checkpoint: Path) -> dict[str, Any]:
    pred68 = load_json(stage68_dir / "prediction_safety_audit.json")
    pred69 = load_json(stage69_dir / "prediction_safety_audit.json")
    src68 = load_json(stage68_dir / "source_hash_report.json")
    src69 = load_json(stage69_dir / "source_hash_report.json")
    ck68 = load_json(stage68_dir / "checkpoint_hash_report.json")
    ck69 = load_json(stage69_dir / "checkpoint_hash_report.json")
    current_source_hash = sha256_file(SOURCE_USD)
    current_fixed_hash = sha256_file(fixed_usd)
    current_checkpoint_hash = sha256_file(checkpoint)
    current_source_observed_hash = sha256_file(SOURCE_OBSERVED_STATE)

    forbidden_flags = (
        "prediction_writeback",
        "prediction_traversability_use",
        "prediction_collision_use",
        "prediction_ray_blocking_use",
        "prediction_candidate_validity_use",
        "prediction_edge_validity_use",
        "target_ground_truth_use",
        "future_observed_scoring_use",
    )
    stage_safety_ok = bool(pred68.get("passed")) and bool(pred69.get("passed"))
    flags_ok = all(not bool(pred68.get(flag)) and not bool(pred69.get(flag)) for flag in forbidden_flags)
    hash_ok = (
        current_source_hash == src68.get("source_usd_sha256_after") == src69.get("source_usd_sha256_after")
        and current_fixed_hash == src68.get("fixed_usd_sha256_after") == src69.get("fixed_usd_sha256_after")
        and current_checkpoint_hash == ck68.get("after", {}).get("sha256") == ck69.get("after", {}).get("sha256")
        and current_source_observed_hash
        == src68.get("source_observed_state_sha256_after")
        == src69.get("source_observed_state_sha256_after")
    )
    this_stage = {
        "isaac_startup_count_this_stage": 0,
        "capture_count_this_stage": 0,
        "map_predict_calls_this_stage": 0,
        "sscnet_inference_calls_this_stage": 0,
        "action_execution_count_this_stage": 0,
        "rollout_executed_this_stage": False,
        "long_rollout_executed_this_stage": False,
        "training_run_this_stage": False,
        "bc_il_rl_run_this_stage": False,
        "second_action_count_this_stage": 0,
        "third_frame_count_this_stage": 0,
    }
    return {
        "stage": STAGE,
        "stage4a68_prediction_safety_passed": bool(pred68.get("passed")),
        "stage4a69_prediction_safety_passed": bool(pred69.get("passed")),
        "forbidden_prediction_flags_false": flags_ok,
        "forbidden_prediction_flags": {flag: {"stage4a68": bool(pred68.get(flag)), "stage4a69": bool(pred69.get(flag))} for flag in forbidden_flags},
        "checkpoint_unchanged": current_checkpoint_hash == ck69.get("after", {}).get("sha256"),
        "source_usd_unchanged": current_source_hash == src69.get("source_usd_sha256_after"),
        "fixed_usd_unchanged": current_fixed_hash == src69.get("fixed_usd_sha256_after"),
        "source_observed_state_unchanged": current_source_observed_hash == src69.get("source_observed_state_sha256_after"),
        "source_hashes": {
            "source_usd_sha256": current_source_hash,
            "fixed_usd_sha256": current_fixed_hash,
            "source_observed_state_sha256": current_source_observed_hash,
            "checkpoint_sha256": current_checkpoint_hash,
        },
        **this_stage,
        "passed": stage_safety_ok and flags_ok and hash_ok and all(v in (0, False) for v in this_stage.values()),
    }


def no_report(name: str, passed: bool, facts: dict[str, Any]) -> dict[str, Any]:
    return {"stage": STAGE, "report": name, "passed": passed, **facts}


def write_future_sketch(path: Path) -> None:
    write_text(
        path,
        """DO NOT RUN IN STAGE 4A-6.10.

# Future Stage 4A-6.11 Uncertainty-Aware Lambda Pilot Sketch

This is a design sketch only. Stage 4A-6.11 should remain bounded and should not
be a rollout unless explicitly authorized.

Required precondition:

- Future map_predict artifacts must persist dense probability or confidence
  fields, plus enough candidate-visible voxel references to compute per-candidate
  confidence, entropy, and margin offline.
- Prediction must remain read-only and must not affect traversability,
  collision, ray blocking, candidate validity, or edge validity.

Possible score sketches after dense fields exist:

- uncertainty bonus: `gain_exp / cost + 48 * minmax(source_occ_free) + beta * minmax(candidate_uncertain_fraction)`
- confidence gate: `gain_exp / cost + 48 * minmax(source_occ_free_confidence_weighted)`
- uncertainty penalty: `gain_exp / cost + 48 * minmax(source_occ_free) - beta * minmax(candidate_uncertain_fraction)`

Recommended first step:

Persist dense probability/confidence/entropy/margin artifacts in a future
map_predict run, rerun Stage 4A-6.10 offline audit, and only then choose one
bounded Stage 4A-6.11 pilot score. Do not jump to long rollout.
""",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a68_dir", type=Path, required=True)
    parser.add_argument("--stage4a69_dir", type=Path, required=True)
    parser.add_argument("--stage4a64_calibration_dir", type=Path, default=None)
    parser.add_argument("--stage4a62_diagnostics_dir", type=Path, default=None)
    parser.add_argument("--fixed_usd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--confidence_thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--entropy_thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--margin_thresholds", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--compute_candidate_uncertainty_if_possible", action="store_true")
    parser.add_argument("--compute_shadow_uncertainty_scores", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--make_html", action="store_true")
    parser.add_argument("--no_isaac", action="store_true", required=True)
    parser.add_argument("--no_capture", action="store_true", required=True)
    parser.add_argument("--no_map_predict", action="store_true", required=True)
    parser.add_argument("--no_sscnet_inference", action="store_true", required=True)
    parser.add_argument("--no_rollout", action="store_true", required=True)
    parser.add_argument("--no_training", action="store_true", required=True)
    parser.add_argument("--no_rl_gdpo", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "git_status_before.txt", git_status_text())

    context_manifest, _context_text = load_context_manifest()
    save_json(output_dir / "loaded_context_manifest.json", context_manifest)
    write_text(output_dir / "loaded_context_manifest.md", markdown_rows("Loaded Context Manifest", context_manifest["context_files_loaded"]))

    stage68_manifest = load_stage_manifest(args.stage4a68_dir.resolve(), REQUIRED_STAGE68, "stage4a68")
    stage69_manifest = load_stage_manifest(args.stage4a69_dir.resolve(), REQUIRED_STAGE69, "stage4a69")
    save_json(output_dir / "loaded_stage4a68_manifest.json", stage68_manifest)
    write_text(output_dir / "loaded_stage4a68_manifest.md", markdown_kv("Loaded Stage 4A-6.8 Manifest", stage68_manifest))
    save_json(output_dir / "loaded_stage4a69_manifest.json", stage69_manifest)
    write_text(output_dir / "loaded_stage4a69_manifest.md", markdown_kv("Loaded Stage 4A-6.9 Manifest", stage69_manifest))

    inventory = inventory_prediction_artifacts(args.stage4a68_dir.resolve(), args.stage4a69_dir.resolve())
    write_csv(output_dir / "prediction_artifact_inventory.csv", inventory)
    save_json(output_dir / "prediction_artifact_inventory.json", inventory)
    write_text(output_dir / "prediction_artifact_inventory.md", markdown_rows("Prediction Artifact Inventory", inventory, max_rows=80))

    dense_artifacts = [row for row in inventory if row["dense_probability_like"]]
    dense_available = bool(dense_artifacts)
    available_fields = {
        "stage": STAGE,
        "uncertainty_mode": MODE,
        "dense_prediction_available": dense_available,
        "confidence_available": False,
        "entropy_available": False,
        "margin_available": False,
        "candidate_uncertainty_available": False,
        "stage4a68_dense_probability_like_npz_keys": stage68_manifest["npz"]["dense_probability_like_keys"],
        "stage4a69_dense_probability_like_npz_keys": stage69_manifest["npz"]["dense_probability_like_keys"],
        "prediction_summary_only_counts": {
            "stage4a68": stage68_manifest["prediction_summary_only_count"],
            "stage4a69": stage69_manifest["prediction_summary_only_count"],
        },
        "removed_prediction_npz_paths_recorded": {
            "stage4a68": stage68_manifest["removed_prediction_npz_paths_recorded"],
            "stage4a69": stage69_manifest["removed_prediction_npz_paths_recorded"],
        },
        "missing_critical_fields": [
            "dense class probability or logits",
            "confidence maps",
            "entropy maps",
            "margin maps",
            "candidate-visible voxel probability lists",
        ],
        "limited_mode_reason": BLOCKED,
    }
    save_json(output_dir / "uncertainty_available_fields_report.json", available_fields)
    write_text(output_dir / "uncertainty_available_fields_report.md", markdown_kv("Uncertainty Available Fields Report", available_fields))

    formula_ref = {
        "stage": STAGE,
        "uncertainty_is": "prediction-derived/confidence-derived only",
        "uncertainty_is_not": ["Bayesian uncertainty", "ensemble uncertainty", "MC-dropout uncertainty"],
        "existing_primary_prediction_feature": {
            "name": "source_occ_free",
            "definition": "raw visible predicted-unmeasured voxel count from read-only prediction layer",
        },
        "dense_probability_formula_if_available": {
            "confidence": "max softmax probability",
            "entropy": "-sum_i p_i * log(p_i + eps)",
            "entropy_norm": "entropy / log(num_classes)",
            "margin": "top1_prob - top2_prob",
            "uncertainty_conf": "1 - confidence",
        },
        "thresholds": {
            "confidence_thresholds": args.confidence_thresholds,
            "entropy_thresholds": args.entropy_thresholds,
            "margin_thresholds": args.margin_thresholds,
            "eps": 1e-8,
        },
        "applied_mode": MODE,
        "applied_mode_note": "Dense probability/confidence/logit fields are not available in 6.8/6.9 outputs, so formulas are documented but not computed.",
    }
    save_json(output_dir / "uncertainty_formula_reference.json", formula_ref)
    write_text(output_dir / "uncertainty_formula_reference.md", markdown_kv("Uncertainty Formula Reference", formula_ref))

    branches = load_branch_maps(args.stage4a68_dir.resolve(), args.stage4a69_dir.resolve())
    frame_rows = frame_rows_from_map_predict(args.stage4a68_dir.resolve(), "stage4a68")
    frame_rows.extend(frame_rows_from_map_predict(args.stage4a69_dir.resolve(), "stage4a69"))
    candidate_rows = load_top_candidate_rows(args.stage4a68_dir.resolve(), args.stage4a69_dir.resolve(), branches)
    selected_rows = selection_rows(args.stage4a68_dir.resolve(), args.stage4a69_dir.resolve(), branches)

    write_csv(output_dir / "frame_uncertainty_summary.csv", frame_rows)
    save_json(output_dir / "frame_uncertainty_summary.json", frame_rows)
    write_text(output_dir / "frame_uncertainty_summary.md", markdown_rows("Frame Uncertainty Summary", frame_rows))
    write_csv(output_dir / "candidate_uncertainty_table.csv", candidate_rows)
    save_json(output_dir / "candidate_uncertainty_table.json", candidate_rows)
    write_text(output_dir / "candidate_uncertainty_table.md", markdown_rows("Candidate Uncertainty Table", candidate_rows))
    write_csv(output_dir / "selected_action_uncertainty_audit.csv", selected_rows)
    save_json(output_dir / "selected_action_uncertainty_audit.json", selected_rows)
    write_text(output_dir / "selected_action_uncertainty_audit.md", markdown_rows("Selected Action Uncertainty Audit", selected_rows))

    source_analysis = source_occ_analysis(candidate_rows)
    branch_rows = branch_analysis(candidate_rows)
    f12_rows = frame1_frame2_analysis(frame_rows)
    stage_cmp = stage68_stage69_comparison(frame_rows, selected_rows)
    shadow_rows = shadow_score_audit(candidate_rows)
    write_report_triplet(output_dir, "uncertainty_vs_source_occ_free_analysis", source_analysis, "Uncertainty vs Source Occ Free Analysis")
    write_report_triplet(output_dir, "uncertainty_vs_branch_classification", branch_rows, "Uncertainty vs Branch Classification")
    write_report_triplet(output_dir, "frame1_vs_frame2_uncertainty_analysis", f12_rows, "Frame1 vs Frame2 Uncertainty Analysis")
    write_report_triplet(output_dir, "stage4a68_vs_stage4a69_uncertainty_comparison", stage_cmp, "Stage 4A-6.8 vs 6.9 Uncertainty Comparison")
    write_report_triplet(output_dir, "uncertainty_shadow_score_audit", shadow_rows, "Uncertainty Shadow Score Audit")

    safety = safety_recheck(args.stage4a68_dir.resolve(), args.stage4a69_dir.resolve(), args.fixed_usd.resolve(), args.checkpoint.resolve())
    save_json(output_dir / "prediction_safety_recheck.json", safety)
    write_text(output_dir / "prediction_safety_recheck.md", markdown_kv("Prediction Safety Recheck", safety))

    readiness = make_readiness(dense_available, 0, frame_rows, safety)
    save_json(output_dir / "uncertainty_readiness_decision.json", readiness)
    write_text(output_dir / "uncertainty_readiness_decision.md", markdown_kv("Uncertainty Readiness Decision", readiness))

    dense_blocker = {
        "stage": STAGE,
        "blocked": True,
        "uncertainty_mode": MODE,
        "main_blocker": BLOCKED,
        "candidate_level_uncertainty_ready": False,
        "uncertainty_aware_expert_pilot_ready": False,
        "evidence": {
            "stage4a68_prediction_summary_only_count": stage68_manifest["prediction_summary_only_count"],
            "stage4a69_prediction_summary_only_count": stage69_manifest["prediction_summary_only_count"],
            "stage4a68_removed_npz_paths_recorded": stage68_manifest["removed_prediction_npz_paths_recorded"],
            "stage4a69_removed_npz_paths_recorded": stage69_manifest["removed_prediction_npz_paths_recorded"],
        },
        "recommendation": "Persist dense probability/confidence/entropy/margin outputs in a future map_predict run, then rerun the offline audit.",
    }
    save_json(output_dir / "dense_uncertainty_blocker.json", dense_blocker)
    write_text(output_dir / "dense_uncertainty_blocker.md", markdown_kv("Dense Uncertainty Blocker", dense_blocker))

    source_report = {
        "stage": STAGE,
        "source_usd": str(SOURCE_USD),
        "fixed_usd": str(args.fixed_usd.resolve()),
        "source_observed_state": str(SOURCE_OBSERVED_STATE),
        "source_usd_sha256": sha256_file(SOURCE_USD),
        "fixed_usd_sha256": sha256_file(args.fixed_usd.resolve()),
        "source_observed_state_sha256": sha256_file(SOURCE_OBSERVED_STATE),
        "source_usd_modified_this_stage": False,
        "fixed_usd_modified_this_stage": False,
        "source_observed_state_modified_this_stage": False,
    }
    save_json(output_dir / "source_hash_report.json", source_report)
    write_text(output_dir / "source_hash_report.md", markdown_kv("Source Hash Report", source_report))
    checkpoint_report = {
        "stage": STAGE,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "checkpoint_modified_this_stage": False,
        "checkpoint_loaded_for_inference_this_stage": False,
    }
    save_json(output_dir / "checkpoint_hash_report.json", checkpoint_report)
    write_text(output_dir / "checkpoint_hash_report.md", markdown_kv("Checkpoint Hash Report", checkpoint_report))

    no_reports = [
        ("no_isaac_report", {"isaac_startup_count_this_stage": 0, "isaac_started": False}),
        ("no_capture_report", {"capture_count_this_stage": 0, "capture_executed": False}),
        ("no_map_predict_report", {"map_predict_calls_this_stage": 0, "map_predict_rerun": False}),
        ("no_sscnet_inference_report", {"sscnet_inference_calls_this_stage": 0, "checkpoint_loaded_for_inference": False}),
        ("no_rollout_report", {"rollout_executed_this_stage": False, "long_rollout_executed_this_stage": False, "second_action_count_this_stage": 0, "third_frame_count_this_stage": 0}),
        ("no_training_rl_bc_report", {"training_run_this_stage": False, "bc_il_rl_gdpo_ppo_run_this_stage": False, "replay_buffer_created": False, "policy_checkpoint_created": False}),
    ]
    for stem, facts in no_reports:
        report = no_report(stem, True, facts)
        save_json(output_dir / f"{stem}.json", report)
        write_text(output_dir / f"{stem}.md", markdown_kv(stem, report))

    if args.save_viz:
        make_visuals(output_dir, frame_rows, candidate_rows, selected_rows)
    if args.make_html:
        make_html_index(output_dir, args.stage4a69_dir.resolve(), frame_rows, selected_rows, readiness)

    write_text(
        output_dir / "recommended_next_faithful_step.md",
        """# Recommended Next Faithful Step

Update the future map_predict artifact contract to persist dense probability or
confidence fields, plus candidate-visible voxel probability references. Then
rerun this offline audit before choosing any uncertainty-aware lambda score.

Do not jump to long rollout.
""",
    )
    write_future_sketch(output_dir / "future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md")

    summary = {
        "stage": STAGE,
        "completed": True,
        "blocked": True,
        "main_blocker": BLOCKED,
        "output_dir": str(output_dir),
        "created_at_utc": utc_now(),
        "stage4a68_loaded": True,
        "stage4a69_loaded": True,
        "stage4a64_calibration_dir": str(args.stage4a64_calibration_dir) if args.stage4a64_calibration_dir else None,
        "stage4a64_calibration_loaded": bool(args.stage4a64_calibration_dir and args.stage4a64_calibration_dir.exists()),
        "stage4a62_diagnostics_dir": str(args.stage4a62_diagnostics_dir) if args.stage4a62_diagnostics_dir else None,
        "stage4a62_diagnostics_loaded": bool(args.stage4a62_diagnostics_dir and args.stage4a62_diagnostics_dir.exists()),
        "fixed_usd": str(args.fixed_usd.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "prediction_artifacts_found": len(inventory),
        "dense_prediction_probability_artifacts": len(dense_artifacts),
        "summary_only_artifacts": sum(1 for row in inventory if row["category"] == "prediction_summary"),
        "candidate_tables": sum(1 for row in inventory if row["category"] == "candidate_table_or_visual" and row["extension"] in (".csv", ".jsonl")),
        "frames_analyzed": len(frame_rows),
        "candidates_analyzed": len(candidate_rows),
        "candidate_level_uncertainty_rows": 0,
        "uncertainty_mode": MODE,
        "confidence_available": False,
        "entropy_available": False,
        "margin_available": False,
        "candidate_level_uncertainty_available": False,
        "confidence_summary": summarize([]),
        "entropy_summary": summarize([]),
        "margin_summary": summarize([]),
        "source_occ_free_summary": summarize([as_float(r.get("source_occ_free")) for r in candidate_rows if as_float(r.get("source_occ_free")) is not None]),
        "selected_source_occ_free_summary": summarize([as_float(r.get("selected_source_occ_free")) for r in selected_rows if r.get("action_type") == "lambda48" and as_float(r.get("selected_source_occ_free")) is not None]),
        "prediction_density_summary": summarize([as_float(r.get("prediction_density")) for r in frame_rows if as_float(r.get("prediction_density")) is not None]),
        "frame1_vs_frame2_prediction_unmeasured_delta_summary": summarize(
            [
                as_float(r.get("predicted_unmeasured_count_delta_frame2_minus_frame1"))
                for r in f12_rows
                if as_float(r.get("predicted_unmeasured_count_delta_frame2_minus_frame1")) is not None
            ]
        ),
        "source_occ_free_vs_uncertainty": BLOCKED,
        "branch_class_uncertainty": BLOCKED,
        "candidate_all_local_relation": "blocked_for_uncertainty; warning class exists but dense fields are absent",
        "uncertainty_feature_extraction_complete": readiness["uncertainty_feature_extraction_complete"],
        "candidate_level_uncertainty_ready": readiness["candidate_level_uncertainty_ready"],
        "uncertainty_aware_expert_pilot_ready": readiness["uncertainty_aware_expert_pilot_ready"],
        "safety_recheck_passed": safety["passed"],
        "isaac_startup_count_this_stage": 0,
        "capture_count_this_stage": 0,
        "map_predict_calls_this_stage": 0,
        "sscnet_inference_calls_this_stage": 0,
        "action_execution_count_this_stage": 0,
        "rollout_executed_this_stage": False,
        "long_rollout_executed_this_stage": False,
        "training_run_this_stage": False,
        "bc_il_rl_run_this_stage": False,
        "prediction_writeback_this_stage": False,
        "source_usd_modified": False,
        "fixed_usd_modified": False,
        "checkpoint_modified": False,
        "observed_state_modified": False,
        "future_stage4a611_executed": False,
        "run_log": str(WORKSPACE / "logs/stage4a610_prediction_uncertainty_offline_audit.log"),
        "test_log": str(WORKSPACE / "logs/stage4a610_prediction_uncertainty_offline_audit_test.log"),
        "py_compile_log": str(WORKSPACE / "logs/stage4a610_py_compile.log"),
    }
    save_json(output_dir / "stage4a610_prediction_uncertainty_offline_audit_summary.json", summary)
    write_text(output_dir / "stage4a610_prediction_uncertainty_offline_audit_summary.md", markdown_kv("Stage 4A-6.10 Prediction Uncertainty Offline Audit Summary", summary))

    write_text(output_dir / "git_status_after.txt", git_status_text())
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
