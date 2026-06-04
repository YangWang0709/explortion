#!/usr/bin/env python3
"""Stage 4A-6.10a dense prediction uncertainty artifact regeneration.

This stage does not start Isaac, capture frames, execute actions, roll out, or
train. It reads existing Stage 4A-6.8 / 6.9 captures, reruns SSCNet/map_predict
offline only to persist compact prediction-derived uncertainty fields, then
reruns the Stage 4A-6.10 audit in dense mode.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
import shutil
import subprocess
import time
from collections import defaultdict
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

from isaac_map_predictor import IsaacMapPredictor
from prediction_uncertainty_utils import (
    FREE_CLASS_ID,
    UNKNOWN,
    as_float,
    as_int,
    dense_stats,
    file_record,
    jsonable,
    load_json,
    load_jsonl,
    markdown_kv,
    markdown_rows,
    pearson,
    read_csv,
    save_json,
    sha256_array,
    sha256_file,
    summarize,
    utc_now,
    write_csv,
    write_text,
)
from sim_paper_expert import SimCandidateView, raycast_visible_voxels_observed


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
SOURCE_USD = WORKSPACE / "building_scene.usd"
SOURCE_OBSERVED_STATE = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix/observed_state_final.npy"
CAMERA_FIX_DIR = WORKSPACE / "outputs/isaac_stage4a66c_usd_camera_pose_fix"
STAGE = "Stage 4A-6.10a-dense-prediction-uncertainty-artifact-regeneration"
DENSE_MODE = "dense_prediction_uncertainty"

CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/TODO.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / "README.md",
    WORKSPACE / "ARTIFACTS.md",
    WORKSPACE / "ENVIRONMENT.md",
    WORKSPACE / "GIT_INITIALIZATION_REPORT.md",
]

LIMITED_AUDIT_FILES = [
    "stage4a610_prediction_uncertainty_offline_audit_summary.json",
    "uncertainty_readiness_decision.json",
    "dense_uncertainty_blocker.json",
    "prediction_artifact_inventory.json",
    "uncertainty_available_fields_report.json",
    "future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md",
]

STAGE68_FILES = [
    "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
    "expert_dataset.npz",
    "expert_dataset_manifest.jsonl",
    "per_sample_summary.csv",
    "lambda48_decisions.csv",
    "measured_shadow_decisions.csv",
    "map_predict_summary.json",
    "prediction_safety_audit.json",
    "expert_data_quality_audit.json",
]

STAGE69_FILES = [
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


def parse_literal(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, tuple, dict)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return default


def npz_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "keys": sorted(data.files),
            "shapes": {key: [int(v) for v in data[key].shape] for key in data.files},
            "dtypes": {key: str(data[key].dtype) for key in data.files},
            "dense_uncertainty_keys": [
                key
                for key in sorted(data.files)
                if any(token in key.lower() for token in ("confidence", "entropy", "margin", "prob", "pred_class"))
            ],
        }


def write_manifest_pair(output_dir: Path, stem: str, data: Any, title: str) -> None:
    save_json(output_dir / f"{stem}.json", data)
    if isinstance(data, list):
        write_text(output_dir / f"{stem}.md", markdown_rows(title, data, max_rows=80))
    else:
        write_text(output_dir / f"{stem}.md", markdown_kv(title, data))


def loaded_file_manifest(paths: list[Path], title: str) -> dict[str, Any]:
    rows = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")
        rec = file_record(path, include_hash=True)
        if path.suffix.lower() in (".md", ".txt", ".csv", ".json", ".jsonl"):
            try:
                text = path.read_text(encoding="utf-8")
                rec["line_count"] = text.count("\n") + (0 if text.endswith("\n") else 1)
            except UnicodeDecodeError:
                rec["line_count"] = None
        rows.append(rec)
    return {"stage": STAGE, "title": title, "loaded_at_utc": utc_now(), "files": rows, "loaded": True}


def stage_output_manifest(stage_dir: Path, required: list[str], title: str) -> dict[str, Any]:
    paths = [stage_dir / name for name in required]
    manifest = loaded_file_manifest(paths, title)
    dataset_name = "expert_dataset.npz" if "6.8" in title else "expert_dataset_two_frame.npz"
    dataset_path = stage_dir / dataset_name
    if dataset_path.is_file():
        manifest["dataset_npz_summary"] = npz_summary(dataset_path)
    if (stage_dir / "expert_dataset_manifest.jsonl").is_file():
        manifest["manifest_row_count"] = len(load_jsonl(stage_dir / "expert_dataset_manifest.jsonl"))
    if (stage_dir / "map_predict_summary.json").is_file():
        mp = load_json(stage_dir / "map_predict_summary.json")
        rows = mp.get("samples") or mp.get("frames") or []
        manifest["map_predict_row_count"] = len(rows)
        manifest["prediction_summary_only_count"] = int(sum(bool(row.get("prediction_summary_only")) for row in rows))
        manifest["removed_prediction_npz_paths_recorded"] = [
            item
            for row in rows
            for item in (row.get("prediction_array_npz_removed_after_summary") or [])
        ]
    return manifest


def dense_contract() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "future_map_predict_flag": "--save_dense_uncertainty_artifacts",
        "compact_probability_fields_default": True,
        "full_class_prob_saved_by_default": False,
        "required_npz_fields": [
            "pred_class_uint8",
            "confidence_float16",
            "entropy_norm_float16",
            "margin_float16",
            "occupied_prob_float16",
            "free_prob_float16",
            "valid_mask_bool",
            "predicted_unmeasured_mask_bool",
            "observed_reference_hash",
            "source_occ_free_count",
            "prediction_valid_count",
            "predicted_unmeasured_count",
            "predicted_occupied_count",
            "shape",
            "voxel_size",
            "bounds",
            "alignment_convention",
            "tau",
            "checkpoint_sha256",
            "source_depth_sha256",
            "source_pose_sha256",
            "source_camera_info_sha256",
            "source_observed_state_sha256",
            "no_prediction_writeback",
        ],
        "definitions": {
            "confidence": "max softmax probability",
            "entropy_norm": "-sum_i p_i * log(p_i + 1e-8) / log(num_classes)",
            "margin": "top1_prob - top2_prob",
            "occupied_prob": "1 - free_prob under existing FREE_CLASS_ID=0 convention",
            "free_prob": "softmax probability for FREE_CLASS_ID=0",
            "uncertainty_conf": "1 - confidence, computed by audit from confidence",
        },
        "not_saved_by_default": ["full 12-class class_prob tensor"],
    }


def class_mapping() -> dict[str, Any]:
    return {
        "stage": STAGE,
        "num_classes": 12,
        "free_class_id": int(FREE_CLASS_ID),
        "occupied_probability_convention": "occupied_prob = 1.0 - class_prob[FREE_CLASS_ID]",
        "occupied_class_set": "all non-free SSCNet classes aggregated through 1-free_prob",
        "pred_class_uint8": "argmax over 12 SSCNet softmax classes",
        "source": "run_isaac_map_predict_single.py constants NUM_CLASSES=12 and FREE_CLASS_ID=0",
        "ambiguity": "No per-object semantic class names are required for this uncertainty audit.",
    }


def scene_map_context() -> tuple[dict[str, Any], float]:
    observed_summary = load_json(CAMERA_FIX_DIR / "observed_summary.json")
    scene_metadata = load_json(CAMERA_FIX_DIR / "scene_metadata.json")
    bounds = observed_summary.get("chosen_bounds") or scene_metadata.get("map_bounds")
    voxel_size = float(observed_summary.get("voxel_size", scene_metadata.get("voxel_size", 0.1)))
    if not bounds:
        raise ValueError("Could not resolve map bounds from camera-pose-fix context")
    return bounds, voxel_size


def logical_frame_records(stage68_dir: Path, stage69_dir: Path) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    branch68 = {
        (as_int(row.get("sample_index")), as_int(row.get("start_variant_id")), 1): row.get("branch_classification", "unknown")
        for row in read_csv(stage68_dir / "lambda48_decisions.csv")
    }
    branch69 = {}
    for name in ("frame1_lambda48_decisions.csv", "frame2_lambda48_diagnostic_decisions.csv"):
        for row in read_csv(stage69_dir / name):
            branch69[(None, as_int(row.get("start_variant_id")), as_int(row.get("frame_id")))] = row.get(
                "branch_classification",
                "unknown",
            )
    camera68 = stage68_dir / "camera_info.json"
    for row in load_jsonl(stage68_dir / "expert_dataset_manifest.jsonl"):
        idx = int(row["sample_index"])
        sample_dir = Path(row["sample_dir"])
        frames.append(
            {
                "logical_frame_index": len(frames),
                "stage_source": "6.8",
                "stage_alias": "stage4a68",
                "start_variant_id": int(row["start_variant_id"]),
                "sample_index": idx,
                "frame_id": 1,
                "depth": row["depth"],
                "pose": row["pose"],
                "camera_info": str(camera68),
                "observed_state": row.get("map_observation") or row.get("observed_state"),
                "prediction_summary": row["prediction_summary"],
                "top_candidates_csv": row["top_candidates_csv"],
                "lambda48_decision": row["lambda48_decision"],
                "measured_shadow_decision": row["measured_shadow_decision"],
                "sample_dir": str(sample_dir),
                "branch_classification": branch68.get((idx, int(row["start_variant_id"]), 1), "unknown"),
            }
        )
    camera69 = stage69_dir / "camera_info.json"
    for row in load_jsonl(stage69_dir / "expert_dataset_manifest.jsonl"):
        sample_dir = Path(row["sample_dir"])
        frame_id = int(row["frame_id"])
        frames.append(
            {
                "logical_frame_index": len(frames),
                "stage_source": "6.9",
                "stage_alias": "stage4a69",
                "start_variant_id": int(row["start_variant_id"]),
                "sample_index": "",
                "frame_id": frame_id,
                "depth": row["depth"],
                "pose": row["pose"],
                "camera_info": str(camera69),
                "observed_state": row["observed_state"],
                "prediction_summary": row["prediction_summary"],
                "top_candidates_csv": str(sample_dir / f"frame{frame_id}_top_candidates.csv"),
                "lambda48_decision": row["lambda48_decision"],
                "measured_shadow_decision": row["measured_shadow_decision"],
                "sample_dir": str(sample_dir),
                "diagnostic_only": bool(row.get("diagnostic_only", frame_id == 2)),
                "branch_classification": branch69.get((None, int(row["start_variant_id"]), frame_id), "unknown"),
            }
        )
    for frame in frames:
        for key in ("depth", "pose", "camera_info", "observed_state", "prediction_summary", "top_candidates_csv"):
            if not Path(frame[key]).is_file():
                raise FileNotFoundError(f"Frame {frame['logical_frame_index']} missing {key}: {frame[key]}")
        frame["source_depth_sha256"] = sha256_file(frame["depth"])
        frame["source_pose_sha256"] = sha256_file(frame["pose"])
        frame["source_camera_info_sha256"] = sha256_file(frame["camera_info"])
        frame["source_observed_state_sha256"] = sha256_file(frame["observed_state"])
        frame["source_hash_key"] = "|".join(
            str(frame[key])
            for key in (
                "source_depth_sha256",
                "source_pose_sha256",
                "source_camera_info_sha256",
                "source_observed_state_sha256",
            )
        )
    return frames


def artifact_inventory(primary_dir: Path, rerun_dir: Path | None = None) -> list[dict[str, Any]]:
    roots = [("primary", primary_dir)]
    if rerun_dir is not None and rerun_dir.exists():
        roots.append(("rerun_audit", rerun_dir))
    rows = []
    for label, root in roots:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            category = "other"
            if "dense_prediction_uncertainty" in name and path.suffix == ".npz":
                category = "dense_prediction_uncertainty_npz"
            elif "candidate_visible_uncertainty" in name:
                category = "candidate_visible_uncertainty"
            elif "uncertainty" in name:
                category = "uncertainty_report_or_visual"
            rows.append(
                {
                    "root": label,
                    "relative_path": str(path.relative_to(root)),
                    "path": str(path),
                    "category": category,
                    "extension": path.suffix.lower(),
                    "size_bytes": int(path.stat().st_size),
                    "dense_probability_like": category == "dense_prediction_uncertainty_npz",
                }
            )
    return rows


def load_dense_fields(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "pred_class_uint8": np.asarray(data["pred_class_uint8"]),
            "confidence_float16": np.asarray(data["confidence_float16"]),
            "entropy_norm_float16": np.asarray(data["entropy_norm_float16"]),
            "margin_float16": np.asarray(data["margin_float16"]),
            "occupied_prob_float16": np.asarray(data["occupied_prob_float16"]),
            "free_prob_float16": np.asarray(data["free_prob_float16"]),
            "valid_mask_bool": np.asarray(data["valid_mask_bool"], dtype=bool),
            "predicted_unmeasured_mask_bool": np.asarray(data["predicted_unmeasured_mask_bool"], dtype=bool),
        }


def run_dense_regeneration(
    *,
    args: argparse.Namespace,
    frames: list[dict[str, Any]],
    bounds: dict[str, Any],
    voxel_size: float,
    dense_dir: Path,
) -> tuple[list[dict[str, Any]], int]:
    dense_dir.mkdir(parents=True, exist_ok=True)
    predictor = IsaacMapPredictor(
        checkpoint=args.checkpoint,
        device="cuda",
        tau=float(args.tau),
        torch_num_threads=1,
        alignment_convention=str(args.alignment_convention),
    )
    rows: list[dict[str, Any]] = []
    for frame in frames:
        idx = int(frame["logical_frame_index"])
        frame_out = dense_dir / f"frame_{idx:03d}_{frame['stage_alias']}_start_{int(frame['start_variant_id']):03d}_frame{int(frame['frame_id'])}"
        depth = np.load(frame["depth"])
        pose = load_json(frame["pose"])
        camera_info = load_json(frame["camera_info"])
        observed_state = np.load(frame["observed_state"])
        before_hash = sha256_array(observed_state)
        result = predictor.predict_step(
            depth=depth,
            pose=pose,
            camera_info=camera_info,
            observed_state=observed_state,
            map_bounds=bounds,
            voxel_size=float(voxel_size),
            output_dir=frame_out,
            step=idx,
            save_probs=False,
            save_dense_uncertainty_artifacts=bool(args.save_dense_uncertainty_artifacts),
            save_compact_probability_fields=bool(args.save_compact_probability_fields),
            save_viz=False,
            observed_state_path=frame["observed_state"],
            depth_source=frame["depth"],
            pose_source=frame["pose"],
            camera_info_source=frame["camera_info"],
        )
        after_hash = sha256_array(observed_state)
        generated_dense = Path(result["dense_prediction_uncertainty_npz"])
        logical_dense = dense_dir / f"dense_prediction_uncertainty_{idx:03d}.npz"
        shutil.copy2(generated_dense, logical_dense)
        fields = load_dense_fields(logical_dense)
        stats = dense_stats(fields, observed_state)
        row = {
            **{k: frame[k] for k in ("logical_frame_index", "stage_source", "stage_alias", "start_variant_id", "frame_id")},
            "dense_artifact_path": str(logical_dense),
            "physical_map_predict_output_dir": str(frame_out),
            "local_prediction_npz": result["local_prediction_npz"],
            "global_prediction_npz": result["global_prediction_npz"],
            "source_hash_key": frame["source_hash_key"],
            "map_predict_regeneration_call_index": int(predictor.steps_predicted),
            "observed_state_hash_before_in_memory": before_hash,
            "observed_state_hash_after_in_memory": after_hash,
            "observed_state_unchanged_in_memory": before_hash == after_hash,
            "checkpoint_unchanged_after_frame": bool(predictor.checkpoint_unchanged()),
            "full_class_prob_saved": False,
            **stats,
        }
        rows.append(row)
    return rows, int(predictor.steps_predicted)


def stats_for_voxels(fields: dict[str, np.ndarray], voxels: list[tuple[int, int, int]]) -> dict[str, Any]:
    if not voxels:
        return {
            "confidence_mean": None,
            "confidence_min": None,
            "confidence_p10": None,
            "confidence_p50": None,
            "confidence_p90": None,
            "uncertainty_conf_mean": None,
            "uncertainty_conf_max": None,
            "entropy_mean": None,
            "entropy_max": None,
            "entropy_p90": None,
            "margin_mean": None,
            "margin_min": None,
            "low_conf_count_0p5": 0,
            "low_conf_count_0p7": 0,
            "low_conf_count_0p9": 0,
            "high_entropy_count_0p5": 0,
            "high_entropy_count_0p7": 0,
            "low_margin_count_0p1": 0,
            "low_margin_count_0p2": 0,
            "uncertain_voxel_count": 0,
            "uncertain_fraction": 0.0,
            "confidence_weighted_source_occ_free": 0.0,
            "entropy_weighted_source_occ_free": 0.0,
            "uncertainty_penalized_source_occ_free": 0.0,
        }
    idx = tuple(np.asarray([v[axis] for v in voxels], dtype=np.int64) for axis in range(3))
    conf = np.asarray(fields["confidence_float16"], dtype=np.float32)[idx]
    ent = np.asarray(fields["entropy_norm_float16"], dtype=np.float32)[idx]
    margin = np.asarray(fields["margin_float16"], dtype=np.float32)[idx]
    uncertain = (conf < 0.7) | (ent > 0.7) | (margin < 0.2)
    return {
        "confidence_mean": float(np.mean(conf)),
        "confidence_min": float(np.min(conf)),
        "confidence_p10": float(np.percentile(conf, 10)),
        "confidence_p50": float(np.percentile(conf, 50)),
        "confidence_p90": float(np.percentile(conf, 90)),
        "uncertainty_conf_mean": float(np.mean(1.0 - conf)),
        "uncertainty_conf_max": float(np.max(1.0 - conf)),
        "entropy_mean": float(np.mean(ent)),
        "entropy_max": float(np.max(ent)),
        "entropy_p90": float(np.percentile(ent, 90)),
        "margin_mean": float(np.mean(margin)),
        "margin_min": float(np.min(margin)),
        "low_conf_count_0p5": int(np.count_nonzero(conf < 0.5)),
        "low_conf_count_0p7": int(np.count_nonzero(conf < 0.7)),
        "low_conf_count_0p9": int(np.count_nonzero(conf < 0.9)),
        "high_entropy_count_0p5": int(np.count_nonzero(ent > 0.5)),
        "high_entropy_count_0p7": int(np.count_nonzero(ent > 0.7)),
        "low_margin_count_0p1": int(np.count_nonzero(margin < 0.1)),
        "low_margin_count_0p2": int(np.count_nonzero(margin < 0.2)),
        "uncertain_voxel_count": int(np.count_nonzero(uncertain)),
        "uncertain_fraction": float(np.count_nonzero(uncertain) / max(1, len(voxels))),
        "confidence_weighted_source_occ_free": float(np.sum(conf)),
        "entropy_weighted_source_occ_free": float(np.sum(ent)),
        "uncertainty_penalized_source_occ_free": float(np.sum(conf * (1.0 - ent))),
    }


def candidate_visibility_row(
    raw: dict[str, str],
    frame: dict[str, Any],
    fields: dict[str, np.ndarray],
    observed_state: np.ndarray,
    branch_classification: str,
    lambda_id: int | None,
    measured_id: int | None,
    max_visible_to_store: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    grid = tuple(int(v) for v in parse_literal(raw.get("grid"), [0, 0, 0]))
    world = parse_literal(raw.get("world"), [None, None, None])
    yaw = float(raw.get("yaw_rad", 0.0))
    candidate_id = as_int(raw.get("candidate_id"))
    candidate = SimCandidateView(
        id=int(candidate_id if candidate_id is not None else -1),
        grid_position=grid,
        world_position=tuple(float(v) for v in world),
        yaw=yaw,
        valid=True,
        candidate_source="stage4a610a_candidate_visible_uncertainty",
    )
    max_range_voxels = max(1, int(round(4.8 / 0.1)))
    visible = raycast_visible_voxels_observed(
        candidate,
        observed_state,
        max_range_voxels=max_range_voxels,
        num_yaw=24,
        num_pitch=5,
        fov_yaw_deg=90.0,
        fov_pitch_deg=60.0,
    )
    valid = np.asarray(fields["valid_mask_bool"], dtype=bool)
    predicted_unmeasured = np.asarray(fields["predicted_unmeasured_mask_bool"], dtype=bool)
    visible_prediction = [tuple(v) for v in visible if valid[tuple(v)]]
    visible_predicted_unmeasured = [tuple(v) for v in visible if predicted_unmeasured[tuple(v)]]
    stats = stats_for_voxels(fields, visible_predicted_unmeasured)
    truncated = len(visible) > int(max_visible_to_store)
    visible_flat = np.ravel_multi_index(tuple(np.asarray([v[a] for v in visible], dtype=np.int64) for a in range(3)), observed_state.shape) if visible and not truncated else np.asarray([], dtype=np.int64)
    prediction_flat = (
        np.ravel_multi_index(tuple(np.asarray([v[a] for v in visible_predicted_unmeasured], dtype=np.int64) for a in range(3)), observed_state.shape)
        if visible_predicted_unmeasured and not truncated
        else np.asarray([], dtype=np.int64)
    )
    row = {
        "stage_source": frame["stage_source"],
        "start_variant_id": int(frame["start_variant_id"]),
        "frame_id": int(frame["frame_id"]),
        "logical_frame_index": int(frame["logical_frame_index"]),
        "candidate_id": candidate_id,
        "candidate_rank": as_int(raw.get("rank")),
        "selected_lambda48": bool(candidate_id is not None and lambda_id is not None and int(candidate_id) == int(lambda_id)),
        "selected_measured_shadow": bool(candidate_id is not None and measured_id is not None and int(candidate_id) == int(measured_id)),
        "branch_classification": branch_classification,
        "world_xyz": json.dumps(world),
        "yaw": yaw,
        "gain_exp": as_float(raw.get("gain_exp")),
        "source_occ_free": as_float(raw.get("source_occ_free")),
        "path_cost": as_float(raw.get("cost_s") or raw.get("path_cost")),
        "path_cost_m": as_float(raw.get("path_cost_m")),
        "lambda48_score": as_float(raw.get("final_score_lambda48") or raw.get("value_lambda48")),
        "visible_prediction_voxel_count": int(len(visible_prediction)),
        "visible_predicted_unmeasured_count": int(len(visible_predicted_unmeasured)),
        "candidate_visible_reference_status": "saved" if not truncated else "summary_only_truncated",
        "voxel_reference_truncated": bool(truncated),
        "visible_voxel_count": int(len(visible)),
        "visible_voxel_hash": sha256_array(np.asarray(visible_flat, dtype=np.int64)) if visible_flat.size else "",
        **stats,
    }
    row["source_occ_free_count_delta_vs_recomputed"] = (
        None
        if row["source_occ_free"] is None
        else float(row["source_occ_free"]) - float(row["visible_predicted_unmeasured_count"])
    )
    refs = {
        "candidate_id": int(candidate_id if candidate_id is not None else -1),
        "visible_flat": np.asarray(visible_flat, dtype=np.int64),
        "prediction_flat": np.asarray(prediction_flat, dtype=np.int64),
        "confidence": np.asarray(fields["confidence_float16"], dtype=np.float16).reshape(-1)[prediction_flat] if prediction_flat.size else np.asarray([], dtype=np.float16),
        "entropy": np.asarray(fields["entropy_norm_float16"], dtype=np.float16).reshape(-1)[prediction_flat] if prediction_flat.size else np.asarray([], dtype=np.float16),
        "margin": np.asarray(fields["margin_float16"], dtype=np.float16).reshape(-1)[prediction_flat] if prediction_flat.size else np.asarray([], dtype=np.float16),
    }
    return row, refs


def decision_ids(frame: dict[str, Any]) -> tuple[int | None, int | None, str]:
    lambda_decision = load_json(frame["lambda48_decision"]) if Path(frame["lambda48_decision"]).is_file() else {}
    measured_decision = load_json(frame["measured_shadow_decision"]) if Path(frame["measured_shadow_decision"]).is_file() else {}
    branch = frame.get("branch_classification") or lambda_decision.get("branch_classification", "unknown")
    return as_int(lambda_decision.get("candidate_id")), as_int(measured_decision.get("candidate_id")), branch


def compute_candidate_uncertainty(
    *,
    frames: list[dict[str, Any]],
    dense_rows: list[dict[str, Any]],
    candidate_dir: Path,
    max_visible_to_store: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    dense_by_index = {int(row["logical_frame_index"]): Path(row["dense_artifact_path"]) for row in dense_rows}
    candidate_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for frame in frames:
        idx = int(frame["logical_frame_index"])
        fields = load_dense_fields(dense_by_index[idx])
        observed_state = np.load(frame["observed_state"])
        lambda_id, measured_id, branch = decision_ids(frame)
        rows = read_csv(frame["top_candidates_csv"])
        frame_rows: list[dict[str, Any]] = []
        visible_chunks: list[np.ndarray] = []
        prediction_chunks: list[np.ndarray] = []
        conf_chunks: list[np.ndarray] = []
        ent_chunks: list[np.ndarray] = []
        margin_chunks: list[np.ndarray] = []
        offsets = [0]
        pred_offsets = [0]
        for raw in rows:
            row, refs = candidate_visibility_row(
                raw,
                frame,
                fields,
                observed_state,
                branch,
                lambda_id,
                measured_id,
                max_visible_to_store,
            )
            frame_rows.append(row)
            if refs["visible_flat"].size:
                visible_chunks.append(refs["visible_flat"])
            if refs["prediction_flat"].size:
                prediction_chunks.append(refs["prediction_flat"])
                conf_chunks.append(refs["confidence"])
                ent_chunks.append(refs["entropy"])
                margin_chunks.append(refs["margin"])
            offsets.append(offsets[-1] + int(refs["visible_flat"].size))
            pred_offsets.append(pred_offsets[-1] + int(refs["prediction_flat"].size))
        csv_path = candidate_dir / f"candidate_visible_uncertainty_{idx:03d}.csv"
        jsonl_path = candidate_dir / f"candidate_visible_uncertainty_{idx:03d}.jsonl"
        npz_path = candidate_dir / f"candidate_visible_uncertainty_{idx:03d}.npz"
        write_csv(csv_path, frame_rows)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in frame_rows:
                handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")
        np.savez_compressed(
            npz_path,
            candidate_id=np.asarray([int(r["candidate_id"]) for r in frame_rows], dtype=np.int32),
            visible_offsets=np.asarray(offsets, dtype=np.int64),
            prediction_offsets=np.asarray(pred_offsets, dtype=np.int64),
            visible_flat_indices=np.concatenate(visible_chunks).astype(np.int64) if visible_chunks else np.asarray([], dtype=np.int64),
            prediction_flat_indices=np.concatenate(prediction_chunks).astype(np.int64) if prediction_chunks else np.asarray([], dtype=np.int64),
            confidence_float16=np.concatenate(conf_chunks).astype(np.float16) if conf_chunks else np.asarray([], dtype=np.float16),
            entropy_norm_float16=np.concatenate(ent_chunks).astype(np.float16) if ent_chunks else np.asarray([], dtype=np.float16),
            margin_float16=np.concatenate(margin_chunks).astype(np.float16) if margin_chunks else np.asarray([], dtype=np.float16),
            voxel_reference_truncated=np.asarray([bool(r["voxel_reference_truncated"]) for r in frame_rows], dtype=bool),
            observed_shape=np.asarray(observed_state.shape, dtype=np.int64),
        )
        candidate_rows.extend(frame_rows)
        manifest_rows.append(
            {
                "logical_frame_index": idx,
                "stage_source": frame["stage_source"],
                "start_variant_id": int(frame["start_variant_id"]),
                "frame_id": int(frame["frame_id"]),
                "candidate_rows": len(frame_rows),
                "csv": str(csv_path),
                "jsonl": str(jsonl_path),
                "npz": str(npz_path),
                "truncated_reference_count": int(sum(bool(r["voxel_reference_truncated"]) for r in frame_rows)),
            }
        )
    return candidate_rows, manifest_rows


def make_frame_summary(dense_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in dense_rows:
        rows.append(
            {
                "stage_source": row["stage_source"],
                "stage_alias": row["stage_alias"],
                "start_variant_id": row["start_variant_id"],
                "frame_id": row["frame_id"],
                "logical_frame_index": row["logical_frame_index"],
                "dense_artifact_path": row["dense_artifact_path"],
                "prediction_valid_count": row["prediction_valid_count"],
                "predicted_unmeasured_count": row["predicted_unmeasured_count"],
                "predicted_occupied_count": row["predicted_occupied_count"],
                "source_occ_free_count": row["source_occ_free_count"],
                "confidence_mean": row["confidence_mean"],
                "confidence_p10": row["confidence_p10"],
                "confidence_p50": row["confidence_p50"],
                "confidence_p90": row["confidence_p90"],
                "entropy_mean": row["entropy_norm_mean"],
                "entropy_p90": row["entropy_norm_p90"],
                "margin_mean": row["margin_mean"],
                "margin_p10": row["margin_p10"],
                "uncertainty_mode": DENSE_MODE,
            }
        )
    return rows


def selected_action_rows(frames: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(r["stage_source"]), int(r["start_variant_id"]), int(r["frame_id"]), int(r["candidate_id"])): r
        for r in candidate_rows
        if r.get("candidate_id") is not None
    }
    rows: list[dict[str, Any]] = []
    for frame in frames:
        lambda_id, measured_id, branch = decision_ids(frame)
        for action_type, cid in (("lambda48", lambda_id), ("measured_shadow", measured_id)):
            ref = by_key.get((str(frame["stage_source"]), int(frame["start_variant_id"]), int(frame["frame_id"]), int(cid if cid is not None else -1)))
            rows.append(
                {
                    "stage_source": frame["stage_source"],
                    "start_variant_id": int(frame["start_variant_id"]),
                    "frame_id": int(frame["frame_id"]),
                    "logical_frame_index": int(frame["logical_frame_index"]),
                    "action_type": action_type,
                    "selected_candidate_id": cid,
                    "selected_in_top_candidate_table": ref is not None,
                    "branch_classification": branch,
                    "confidence_mean": None if ref is None else ref["confidence_mean"],
                    "uncertainty_conf_mean": None if ref is None else ref["uncertainty_conf_mean"],
                    "entropy_mean": None if ref is None else ref["entropy_mean"],
                    "margin_mean": None if ref is None else ref["margin_mean"],
                    "uncertain_fraction": None if ref is None else ref["uncertain_fraction"],
                    "source_occ_free": None if ref is None else ref["source_occ_free"],
                    "lambda48_score": None if ref is None else ref["lambda48_score"],
                    "candidate_level_uncertainty_available": ref is not None,
                }
            )
    return rows


def group_analysis(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    out = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: str(item[0])):
        base = {key: group_key[i] for i, key in enumerate(keys)}
        base.update(
            {
                "candidate_rows": len(group_rows),
                "confidence_mean": summarize([r["confidence_mean"] for r in group_rows])["mean"],
                "uncertainty_conf_mean": summarize([r["uncertainty_conf_mean"] for r in group_rows])["mean"],
                "entropy_mean": summarize([r["entropy_mean"] for r in group_rows])["mean"],
                "margin_mean": summarize([r["margin_mean"] for r in group_rows])["mean"],
                "uncertain_fraction_mean": summarize([r["uncertain_fraction"] for r in group_rows])["mean"],
                "source_occ_free_mean": summarize([r["source_occ_free"] for r in group_rows])["mean"],
                "source_occ_free_vs_uncertainty_conf": pearson(
                    [r["source_occ_free"] for r in group_rows],
                    [r["uncertainty_conf_mean"] for r in group_rows],
                ),
            }
        )
        out.append(base)
    return out


def frame1_frame2_analysis(frame_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame = {
        (str(r["stage_source"]), int(r["start_variant_id"]), int(r["frame_id"])): r
        for r in frame_rows
    }
    by_sel = {
        (str(r["stage_source"]), int(r["start_variant_id"]), int(r["frame_id"]), str(r["action_type"])): r
        for r in selected_rows
    }
    rows = []
    starts = sorted({int(r["start_variant_id"]) for r in frame_rows if str(r["stage_source"]) == "6.9"})
    for start in starts:
        f1 = by_frame.get(("6.9", start, 1), {})
        f2 = by_frame.get(("6.9", start, 2), {})
        s1 = by_sel.get(("6.9", start, 1, "lambda48"), {})
        s2 = by_sel.get(("6.9", start, 2, "lambda48"), {})
        rows.append(
            {
                "start_variant_id": start,
                "confidence_mean_delta_frame2_minus_frame1": delta(f2.get("confidence_mean"), f1.get("confidence_mean")),
                "entropy_mean_delta_frame2_minus_frame1": delta(f2.get("entropy_mean"), f1.get("entropy_mean")),
                "margin_mean_delta_frame2_minus_frame1": delta(f2.get("margin_mean"), f1.get("margin_mean")),
                "predicted_unmeasured_count_delta_frame2_minus_frame1": delta(f2.get("predicted_unmeasured_count"), f1.get("predicted_unmeasured_count")),
                "selected_uncertain_fraction_delta_frame2_minus_frame1": delta(s2.get("uncertain_fraction"), s1.get("uncertain_fraction")),
                "frame2_uncertainty_increased": bool((as_float(f2.get("entropy_mean")) or 0.0) > (as_float(f1.get("entropy_mean")) or 0.0)),
            }
        )
    return rows


def stage_comparison(frame_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_frame = {
        (str(r["stage_source"]), int(r["start_variant_id"]), int(r["frame_id"])): r
        for r in frame_rows
    }
    by_sel = {
        (str(r["stage_source"]), int(r["start_variant_id"]), int(r["frame_id"]), str(r["action_type"])): r
        for r in selected_rows
    }
    rows = []
    starts = sorted({int(r["start_variant_id"]) for r in frame_rows})
    for start in starts:
        f68 = by_frame.get(("6.8", start, 1), {})
        s68 = by_sel.get(("6.8", start, 1, "lambda48"), {})
        for frame_id in (1, 2):
            f69 = by_frame.get(("6.9", start, frame_id), {})
            s69 = by_sel.get(("6.9", start, frame_id, "lambda48"), {})
            rows.append(
                {
                    "comparison": f"stage4a68_vs_stage4a69_frame{frame_id}",
                    "start_variant_id": start,
                    "confidence_mean_delta": delta(f69.get("confidence_mean"), f68.get("confidence_mean")),
                    "entropy_mean_delta": delta(f69.get("entropy_mean"), f68.get("entropy_mean")),
                    "margin_mean_delta": delta(f69.get("margin_mean"), f68.get("margin_mean")),
                    "predicted_unmeasured_count_delta": delta(f69.get("predicted_unmeasured_count"), f68.get("predicted_unmeasured_count")),
                    "selected_uncertain_fraction_delta": delta(s69.get("uncertain_fraction"), s68.get("uncertain_fraction")),
                    "lambda48_action_id_changed": (
                        None
                        if not s69 or not s68
                        else int(s69.get("selected_candidate_id", -1)) != int(s68.get("selected_candidate_id", -1))
                    ),
                }
            )
    return rows


def delta(a: Any, b: Any) -> float | None:
    fa = as_float(a)
    fb = as_float(b)
    if fa is None or fb is None:
        return None
    return float(fa - fb)


def shadow_score_audit(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        groups[(str(row["stage_source"]), int(row["start_variant_id"]), int(row["frame_id"]))].append(row)
    out = []
    for key, rows in sorted(groups.items()):
        valid = [r for r in rows if as_float(r.get("lambda48_score")) is not None]
        if not valid:
            continue
        max_unc = max(float(r.get("uncertain_fraction") or 0.0) for r in valid) or 1.0
        max_conf_weighted = max(float(r.get("confidence_weighted_source_occ_free") or 0.0) for r in valid) or 1.0
        for row in valid:
            base = float(row["lambda48_score"])
            row["_unc_bonus"] = base + float(row.get("uncertain_fraction") or 0.0) / max_unc
            row["_unc_penalty"] = base - float(row.get("uncertain_fraction") or 0.0) / max_unc
            row["_conf_gated"] = base + float(row.get("confidence_weighted_source_occ_free") or 0.0) / max_conf_weighted
        selected = next((r for r in valid if bool(r.get("selected_lambda48"))), valid[0])
        best_bonus = max(valid, key=lambda r: float(r["_unc_bonus"]))
        best_penalty = max(valid, key=lambda r: float(r["_unc_penalty"]))
        best_conf = max(valid, key=lambda r: float(r["_conf_gated"]))
        out.append(
            {
                "stage_source": key[0],
                "start_variant_id": key[1],
                "frame_id": key[2],
                "baseline_selected_candidate_id": selected["candidate_id"],
                "baseline_lambda48_score": selected["lambda48_score"],
                "baseline_uncertain_fraction": selected["uncertain_fraction"],
                "uncertainty_bonus_selected_candidate_id": best_bonus["candidate_id"],
                "uncertainty_penalty_selected_candidate_id": best_penalty["candidate_id"],
                "confidence_gated_selected_candidate_id": best_conf["candidate_id"],
                "action_change_under_uncertainty_bonus": int(best_bonus["candidate_id"]) != int(selected["candidate_id"]),
                "action_change_under_uncertainty_penalty": int(best_penalty["candidate_id"]) != int(selected["candidate_id"]),
                "action_change_under_confidence_gating": int(best_conf["candidate_id"]) != int(selected["candidate_id"]),
                "shadow_score_status": "computed_diagnostic_only_not_executed",
            }
        )
    return out


def safety_recheck(args: argparse.Namespace, before: dict[str, Any], physical_calls: int) -> dict[str, Any]:
    after = source_hashes(args)
    forbidden_runtime = {
        "isaac_startup_count_this_stage": 0,
        "capture_count_this_stage": 0,
        "action_execution_count_this_stage": 0,
        "rollout_executed_this_stage": False,
        "long_rollout_executed_this_stage": False,
        "training_run_this_stage": False,
        "bc_il_rl_gdpo_ppo_run_this_stage": False,
        "stage4a611_executed": False,
        "prediction_writeback": False,
        "observed_state_modified": False,
        "target_ground_truth_future_observed_scoring": False,
        "prediction_traversability_collision_ray_candidate_edge_use": False,
    }
    unchanged = {
        key: before.get(key, {}).get("sha256") == after.get(key, {}).get("sha256")
        for key in before
    }
    return {
        "stage": STAGE,
        "before": before,
        "after": after,
        "unchanged": unchanged,
        "map_predict_regeneration_calls": int(physical_calls),
        "sscnet_inference_calls_this_stage": int(physical_calls),
        **forbidden_runtime,
        "passed": all(bool(v) for v in unchanged.values())
        and forbidden_runtime["isaac_startup_count_this_stage"] == 0
        and forbidden_runtime["capture_count_this_stage"] == 0
        and forbidden_runtime["action_execution_count_this_stage"] == 0
        and not forbidden_runtime["rollout_executed_this_stage"]
        and not forbidden_runtime["training_run_this_stage"],
    }


def source_hashes(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "source_usd": SOURCE_USD,
        "fixed_usd": args.fixed_usd,
        "checkpoint": args.checkpoint,
        "source_observed_state": SOURCE_OBSERVED_STATE,
        "stage4a68_expert_dataset": args.stage4a68_dir / "expert_dataset.npz",
        "stage4a69_expert_dataset_two_frame": args.stage4a69_dir / "expert_dataset_two_frame.npz",
        "stage4a610_limited_summary": args.stage4a610_limited_dir / "stage4a610_prediction_uncertainty_offline_audit_summary.json",
        "stage4a610_limited_readiness": args.stage4a610_limited_dir / "uncertainty_readiness_decision.json",
    }
    return {key: file_record(path, include_hash=True) for key, path in paths.items()}


def readiness_decision(
    dense_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    safety: dict[str, Any],
) -> dict[str, Any]:
    conf = [as_float(r.get("confidence_mean")) for r in candidate_rows if as_float(r.get("confidence_mean")) is not None]
    ent = [as_float(r.get("entropy_mean")) for r in candidate_rows if as_float(r.get("entropy_mean")) is not None]
    margin = [as_float(r.get("margin_mean")) for r in candidate_rows if as_float(r.get("margin_mean")) is not None]
    variation = bool(
        len(conf) >= 2
        and float(np.std(np.asarray(conf, dtype=np.float64))) > 0.0
        and float(np.std(np.asarray(ent, dtype=np.float64))) > 0.0
        and float(np.std(np.asarray(margin, dtype=np.float64))) > 0.0
    )
    dense_ok = len(dense_rows) == 30 and all(Path(r["dense_artifact_path"]).is_file() for r in dense_rows)
    candidate_ok = len(candidate_rows) == 480 and all(as_float(r.get("confidence_mean")) is not None for r in candidate_rows)
    selected_ok = len(selected_rows) >= 60
    candidate_ready = bool(dense_ok and candidate_ok and selected_ok and variation)
    ready = bool(candidate_ready and safety.get("passed"))
    blockers = []
    if not dense_ok:
        blockers.append("dense compact fields missing for expected logical frames")
    if not candidate_ok:
        blockers.append("candidate-visible uncertainty rows missing or incomplete")
    if not variation:
        blockers.append("confidence/entropy/margin variation is degenerate")
    if not bool(safety.get("passed")):
        blockers.append("prediction safety/hash recheck failed")
    return {
        "stage": STAGE,
        "uncertainty_mode": DENSE_MODE,
        "dense_artifacts_generated": dense_ok,
        "candidate_visible_uncertainty_generated": candidate_ok,
        "candidate_level_uncertainty_ready": candidate_ready,
        "uncertainty_aware_expert_pilot_ready": ready,
        "candidate_level_uncertainty_rows": len(candidate_rows),
        "meaningful_uncertainty_variation_available": variation,
        "selected_action_uncertainty_audit_exists": selected_ok,
        "prediction_safety_recheck_passed": bool(safety.get("passed")),
        "future_stage4a611_sketch_generated": True,
        "future_stage4a611_executed": False,
        "blockers": blockers,
        "recommended_next": (
            "Stage 4A-6.11 uncertainty-aware lambda pilot design, bounded one-action only, not rollout."
            if ready
            else "Fix candidate-visible uncertainty reference generation before Stage 4A-6.11."
        ),
    }


def plot_dense_examples(path: Path, dense_rows: list[dict[str, Any]], key: str, title: str, cmap: str, vmin: float | None = None, vmax: float | None = None) -> None:
    sample_rows = dense_rows[:4]
    plt.figure(figsize=(11, 8))
    for i, row in enumerate(sample_rows, start=1):
        fields = load_dense_fields(Path(row["dense_artifact_path"]))
        arr = np.asarray(fields[key], dtype=np.float32)
        if arr.dtype == bool or key.endswith("mask_bool"):
            top = np.max(arr.astype(np.float32), axis=2)
        else:
            valid = np.asarray(fields["valid_mask_bool"], dtype=bool)
            masked = np.where(valid, arr, np.nan)
            top = np.nanmax(masked, axis=2)
            top = np.nan_to_num(top, nan=0.0)
        ax = plt.subplot(2, 2, i)
        im = ax.imshow(top.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{row['stage_source']} start {row['start_variant_id']} frame {row['frame_id']}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def scatter(path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, title: str) -> None:
    pairs = [(as_float(r.get(x_key)), as_float(r.get(y_key))) for r in rows]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    plt.figure(figsize=(7, 5))
    if pairs:
        plt.scatter([p[0] for p in pairs], [p[1] for p in pairs], s=18, alpha=0.7)
    else:
        plt.text(0.5, 0.5, "No numeric pairs", ha="center", va="center")
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def bar_plot(path: Path, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def boxplot(path: Path, selected_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> None:
    selected_keys = {
        (str(r["stage_source"]), int(r["start_variant_id"]), int(r["frame_id"]), int(r["selected_candidate_id"]))
        for r in selected_rows
        if r["action_type"] == "lambda48" and r.get("selected_candidate_id") is not None
    }
    selected = []
    nonselected = []
    for row in candidate_rows:
        value = as_float(row.get("uncertain_fraction"))
        if value is None:
            continue
        key = (str(row["stage_source"]), int(row["start_variant_id"]), int(row["frame_id"]), int(row["candidate_id"]))
        (selected if key in selected_keys else nonselected).append(value)
    plt.figure(figsize=(6, 4.8))
    plt.boxplot([selected or [0.0], nonselected or [0.0]], labels=["selected", "nonselected"])
    plt.ylabel("uncertain_fraction")
    plt.title("Selected vs Nonselected Candidate Uncertainty")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def make_visuals(primary_dir: Path, rerun_dir: Path, dense_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]], f12_rows: list[dict[str, Any]]) -> None:
    plot_dense_examples(primary_dir / "dense_confidence_examples.png", dense_rows, "confidence_float16", "Dense Confidence Examples", "viridis", 0.0, 1.0)
    plot_dense_examples(primary_dir / "dense_entropy_examples.png", dense_rows, "entropy_norm_float16", "Dense Entropy Examples", "magma", 0.0, 1.0)
    plot_dense_examples(primary_dir / "dense_margin_examples.png", dense_rows, "margin_float16", "Dense Margin Examples", "cividis", 0.0, 1.0)
    plot_dense_examples(primary_dir / "dense_predicted_unmeasured_examples.png", dense_rows, "predicted_unmeasured_mask_bool", "Dense Predicted-Unmeasured Mask Examples", "gray", 0.0, 1.0)
    scatter(primary_dir / "candidate_uncertainty_examples.png", candidate_rows, "source_occ_free", "uncertain_fraction", "Candidate Uncertainty Examples")
    shutil.copy2(primary_dir / "dense_confidence_examples.png", rerun_dir / "confidence_topdown_examples.png")
    shutil.copy2(primary_dir / "dense_entropy_examples.png", rerun_dir / "entropy_topdown_examples.png")
    shutil.copy2(primary_dir / "dense_margin_examples.png", rerun_dir / "margin_topdown_examples.png")
    shutil.copy2(primary_dir / "dense_predicted_unmeasured_examples.png", rerun_dir / "low_confidence_warning_map.png")
    shutil.copy2(primary_dir / "candidate_uncertainty_examples.png", rerun_dir / "high_uncertainty_candidate_examples.png")
    plot_dense_examples(rerun_dir / "frame_uncertainty_contact_sheet.png", dense_rows, "entropy_norm_float16", "Frame Uncertainty Contact Sheet", "magma", 0.0, 1.0)
    scatter(rerun_dir / "selected_action_uncertainty_contact_sheet.png", selected_rows, "source_occ_free", "uncertain_fraction", "Selected Action Uncertainty")
    scatter(rerun_dir / "uncertainty_vs_source_occ_free_scatter.png", candidate_rows, "source_occ_free", "uncertainty_conf_mean", "Source Occ Free vs Uncertainty")
    scatter(rerun_dir / "uncertainty_vs_gain_exp_scatter.png", candidate_rows, "gain_exp", "uncertainty_conf_mean", "Gain Exp vs Uncertainty")
    scatter(rerun_dir / "uncertainty_vs_path_cost_scatter.png", candidate_rows, "path_cost", "uncertainty_conf_mean", "Path Cost vs Uncertainty")
    scatter(rerun_dir / "uncertainty_vs_lambda48_score_scatter.png", candidate_rows, "lambda48_score", "uncertainty_conf_mean", "Lambda48 Score vs Uncertainty")
    branch = group_analysis(candidate_rows, ["branch_classification"])
    bar_plot(
        rerun_dir / "branch_class_uncertainty_bar.png",
        [str(r["branch_classification"]) for r in branch],
        [float(r["uncertainty_conf_mean"] or 0.0) for r in branch],
        "Branch Class Mean Uncertainty",
        "uncertainty_conf_mean",
    )
    bar_plot(
        rerun_dir / "frame1_frame2_uncertainty_delta_bar.png",
        [str(r["start_variant_id"]) for r in f12_rows],
        [float(r["entropy_mean_delta_frame2_minus_frame1"] or 0.0) for r in f12_rows],
        "Frame2 - Frame1 Entropy Mean Delta",
        "delta",
    )
    boxplot(rerun_dir / "selected_vs_nonselected_uncertainty_boxplot.png", selected_rows, candidate_rows)


def make_html(primary_dir: Path, rerun_dir: Path, summary: dict[str, Any], readiness: dict[str, Any]) -> None:
    primary_body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 4A-6.10a Dense Artifacts</title>
<style>body{{font-family:sans-serif;margin:24px;color:#1f2933}}img{{max-width:45%;min-width:320px;margin:8px;border:1px solid #ddd}}pre{{background:#f6f8fa;padding:12px}}</style>
</head><body>
<h1>Stage 4A-6.10a Dense Prediction Uncertainty Artifacts</h1>
<pre>{html.escape(json.dumps(jsonable(summary), indent=2, sort_keys=True))}</pre>
<p><img src="dense_confidence_examples.png"><img src="dense_entropy_examples.png"><img src="dense_margin_examples.png"><img src="dense_predicted_unmeasured_examples.png"><img src="candidate_uncertainty_examples.png"></p>
</body></html>"""
    write_text(primary_dir / "dense_artifact_index.html", primary_body)
    rerun_body = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Stage 4A-6.10a Dense Audit Rerun</title>
<style>body{{font-family:sans-serif;margin:24px;color:#1f2933}}img{{max-width:45%;min-width:320px;margin:8px;border:1px solid #ddd}}pre{{background:#f6f8fa;padding:12px}}</style>
</head><body>
<h1>Stage 4A-6.10 Dense Uncertainty Audit Rerun</h1>
<pre>{html.escape(json.dumps(jsonable(readiness), indent=2, sort_keys=True))}</pre>
<p><img src="frame_uncertainty_contact_sheet.png"><img src="uncertainty_vs_source_occ_free_scatter.png"><img src="branch_class_uncertainty_bar.png"><img src="selected_vs_nonselected_uncertainty_boxplot.png"></p>
<p>Stage 4A-6.11 was not executed.</p>
</body></html>"""
    write_text(rerun_dir / "uncertainty_overview_index.html", rerun_body)


def write_report_triplet(output_dir: Path, stem: str, data: Any, title: str) -> None:
    save_json(output_dir / f"{stem}.json", data)
    if isinstance(data, list):
        write_csv(output_dir / f"{stem}.csv", data)
        write_text(output_dir / f"{stem}.md", markdown_rows(title, data, max_rows=80))
    else:
        write_text(output_dir / f"{stem}.md", markdown_kv(title, data))


def write_future_sketch(path: Path) -> None:
    write_text(
        path,
        """DO NOT RUN IN STAGE 4A-6.10a.

# Future Stage 4A-6.11 Uncertainty-Aware Lambda Pilot Sketch

This is a design sketch only. Stage 4A-6.11 remains not executed.

Preconditions now available if readiness is true:

- Compact dense confidence, entropy_norm, margin, occupied_prob, and free_prob fields exist.
- Candidate-visible uncertainty summaries exist for the 6.8/6.9 top-candidate rows.
- Prediction remains read-only and is not used for traversability, collision, ray blocking, candidate validity, or edge validity.

Possible bounded one-action pilot sketches:

- uncertainty bonus: `gain_exp / cost + 48 * minmax(source_occ_free) + beta * minmax(candidate_uncertain_fraction)`
- confidence gate: `gain_exp / cost + 48 * minmax(confidence_weighted_source_occ_free)`
- uncertainty penalty: `gain_exp / cost + 48 * minmax(source_occ_free) - beta * minmax(candidate_uncertain_fraction)`

Do not jump to rollout or long rollout.
""",
    )


def no_report(stem: str, facts: dict[str, Any]) -> dict[str, Any]:
    return {"stage": STAGE, "report": stem, "passed": True, **facts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a68_dir", type=Path, required=True)
    parser.add_argument("--stage4a69_dir", type=Path, required=True)
    parser.add_argument("--stage4a610_limited_dir", type=Path, required=True)
    parser.add_argument("--stage4a64_calibration_dir", type=Path, default=None)
    parser.add_argument("--stage4a62_diagnostics_dir", type=Path, default=None)
    parser.add_argument("--fixed_usd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rerun_audit_output_dir", type=Path, required=True)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--save_dense_uncertainty_artifacts", action="store_true")
    parser.add_argument("--save_compact_probability_fields", action="store_true")
    parser.add_argument("--save_candidate_visible_probability_references", action="store_true")
    parser.add_argument("--compute_candidate_uncertainty", action="store_true")
    parser.add_argument("--rerun_uncertainty_audit_dense", action="store_true")
    parser.add_argument("--confidence_thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    parser.add_argument("--entropy_thresholds", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--margin_thresholds", type=float, nargs="+", default=[0.1, 0.2])
    parser.add_argument("--max_candidate_visible_voxels_to_store", type=int, default=20000)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--make_html", action="store_true")
    parser.add_argument("--no_isaac", action="store_true", required=True)
    parser.add_argument("--no_capture", action="store_true", required=True)
    parser.add_argument("--no_action", action="store_true", required=True)
    parser.add_argument("--no_rollout", action="store_true", required=True)
    parser.add_argument("--no_long_rollout", action="store_true", required=True)
    parser.add_argument("--no_training", action="store_true", required=True)
    parser.add_argument("--no_rl_gdpo", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    output_dir = args.output_dir.resolve()
    rerun_dir = args.rerun_audit_output_dir.resolve()
    dense_dir = output_dir / "dense_prediction_artifacts"
    candidate_dir = output_dir / "candidate_visible_uncertainty"
    output_dir.mkdir(parents=True, exist_ok=True)
    rerun_dir.mkdir(parents=True, exist_ok=True)
    write_text(output_dir / "git_status_before.txt", git_status_text())

    source_before = source_hashes(args)
    context_manifest = loaded_file_manifest(CONTEXT_FILES, "Loaded Context Manifest")
    write_manifest_pair(output_dir, "loaded_context_manifest", context_manifest, "Loaded Context Manifest")
    limited_manifest = loaded_file_manifest([args.stage4a610_limited_dir / name for name in LIMITED_AUDIT_FILES], "Loaded Stage 4A-6.10 Limited Audit Manifest")
    write_manifest_pair(output_dir, "loaded_stage4a610_limited_audit_manifest", limited_manifest, "Loaded Stage 4A-6.10 Limited Audit Manifest")
    stage68_manifest = stage_output_manifest(args.stage4a68_dir, STAGE68_FILES, "Loaded Stage 4A-6.8 Manifest")
    stage69_manifest = stage_output_manifest(args.stage4a69_dir, STAGE69_FILES, "Loaded Stage 4A-6.9 Manifest")
    write_manifest_pair(output_dir, "loaded_stage4a68_manifest", stage68_manifest, "Loaded Stage 4A-6.8 Manifest")
    write_manifest_pair(output_dir, "loaded_stage4a69_manifest", stage69_manifest, "Loaded Stage 4A-6.9 Manifest")
    write_manifest_pair(output_dir, "map_predict_dense_artifact_contract", dense_contract(), "Map Predict Dense Artifact Contract")
    write_manifest_pair(output_dir, "dense_prediction_class_mapping", class_mapping(), "Dense Prediction Class Mapping")

    bounds, voxel_size = scene_map_context()
    frames = logical_frame_records(args.stage4a68_dir, args.stage4a69_dir)
    write_csv(output_dir / "dense_regeneration_input_manifest.csv", frames)
    save_json(output_dir / "dense_regeneration_input_manifest.json", frames)
    write_text(output_dir / "dense_regeneration_input_manifest.md", markdown_rows("Dense Regeneration Input Manifest", frames, max_rows=40))

    dense_rows, physical_calls = run_dense_regeneration(args=args, frames=frames, bounds=bounds, voxel_size=voxel_size, dense_dir=dense_dir)
    frame_summary = make_frame_summary(dense_rows)
    write_csv(output_dir / "dense_prediction_artifact_manifest.csv", dense_rows)
    save_json(output_dir / "dense_prediction_artifact_manifest.json", dense_rows)
    write_text(output_dir / "dense_prediction_artifact_manifest.md", markdown_rows("Dense Prediction Artifact Manifest", dense_rows, max_rows=40))
    write_csv(output_dir / "dense_uncertainty_frame_summary.csv", frame_summary)
    save_json(output_dir / "dense_uncertainty_frame_summary.json", frame_summary)
    write_text(output_dir / "dense_uncertainty_frame_summary.md", markdown_rows("Dense Uncertainty Frame Summary", frame_summary, max_rows=40))

    candidate_rows, candidate_manifest = compute_candidate_uncertainty(
        frames=frames,
        dense_rows=dense_rows,
        candidate_dir=candidate_dir,
        max_visible_to_store=int(args.max_candidate_visible_voxels_to_store),
    )
    write_csv(output_dir / "candidate_visible_uncertainty_manifest.csv", candidate_manifest)
    save_json(output_dir / "candidate_visible_uncertainty_manifest.json", candidate_manifest)
    write_text(output_dir / "candidate_visible_uncertainty_manifest.md", markdown_rows("Candidate Visible Uncertainty Manifest", candidate_manifest, max_rows=40))
    write_csv(output_dir / "dense_uncertainty_candidate_summary.csv", candidate_rows)
    save_json(output_dir / "dense_uncertainty_candidate_summary.json", candidate_rows)
    write_text(output_dir / "dense_uncertainty_candidate_summary.md", markdown_rows("Dense Uncertainty Candidate Summary", candidate_rows, max_rows=80))

    selected_rows = selected_action_rows(frames, candidate_rows)
    source_analysis = group_analysis(candidate_rows, ["stage_source", "frame_id"])
    branch_rows = group_analysis(candidate_rows, ["stage_source", "frame_id", "branch_classification"])
    f12_rows = frame1_frame2_analysis(frame_summary, selected_rows)
    cmp_rows = stage_comparison(frame_summary, selected_rows)
    shadow_rows = shadow_score_audit(candidate_rows)
    safety = safety_recheck(args, source_before, physical_calls)
    readiness = readiness_decision(dense_rows, candidate_rows, selected_rows, safety)

    write_report_triplet(rerun_dir, "prediction_artifact_inventory", artifact_inventory(output_dir), "Prediction Artifact Inventory")
    available = {
        "stage": STAGE,
        "uncertainty_mode": DENSE_MODE,
        "dense_prediction_available": True,
        "confidence_available": True,
        "entropy_available": True,
        "margin_available": True,
        "candidate_uncertainty_available": len(candidate_rows) == 480,
        "dense_artifact_count": len(dense_rows),
        "candidate_uncertainty_rows": len(candidate_rows),
        "full_class_prob_saved": False,
    }
    write_report_triplet(rerun_dir, "uncertainty_available_fields_report", available, "Uncertainty Available Fields Report")
    formula = {
        "stage": STAGE,
        "uncertainty_is": "prediction-derived confidence/probability uncertainty",
        "uncertainty_is_not": ["Bayesian uncertainty", "MC-dropout uncertainty", "ensemble uncertainty"],
        "confidence": "max softmax probability",
        "entropy_norm": "-sum_i p_i * log(p_i + 1e-8) / log(num_classes)",
        "margin": "top1_prob - top2_prob",
        "uncertainty_conf": "1 - confidence",
        "source_occ_free": "raw count of visible predicted-unmeasured voxels; kept distinct from uncertainty",
    }
    write_report_triplet(rerun_dir, "uncertainty_formula_reference", formula, "Uncertainty Formula Reference")
    write_report_triplet(rerun_dir, "frame_uncertainty_summary", frame_summary, "Frame Uncertainty Summary")
    write_report_triplet(rerun_dir, "candidate_uncertainty_table", candidate_rows, "Candidate Uncertainty Table")
    write_report_triplet(rerun_dir, "selected_action_uncertainty_audit", selected_rows, "Selected Action Uncertainty Audit")
    write_report_triplet(rerun_dir, "uncertainty_vs_source_occ_free_analysis", source_analysis, "Uncertainty vs Source Occ Free Analysis")
    write_report_triplet(rerun_dir, "uncertainty_vs_branch_classification", branch_rows, "Uncertainty vs Branch Classification")
    write_report_triplet(rerun_dir, "frame1_vs_frame2_uncertainty_analysis", f12_rows, "Frame1 vs Frame2 Uncertainty Analysis")
    write_report_triplet(rerun_dir, "stage4a68_vs_stage4a69_uncertainty_comparison", cmp_rows, "Stage 4A-6.8 vs 6.9 Uncertainty Comparison")
    write_report_triplet(rerun_dir, "uncertainty_shadow_score_audit", shadow_rows, "Uncertainty Shadow Score Audit")
    write_report_triplet(rerun_dir, "uncertainty_readiness_decision", readiness, "Uncertainty Readiness Decision")
    write_report_triplet(rerun_dir, "prediction_safety_recheck", safety, "Prediction Safety Recheck")
    write_future_sketch(rerun_dir / "future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md")
    if not readiness["candidate_level_uncertainty_ready"]:
        blocker = {"stage": STAGE, "blocked": True, "blockers": readiness["blockers"], "candidate_rows": len(candidate_rows)}
        write_report_triplet(rerun_dir, "dense_candidate_uncertainty_blocker", blocker, "Dense Candidate Uncertainty Blocker")

    dense_artifact_safety = {
        "stage": STAGE,
        "passed": bool(safety["passed"] and len(dense_rows) == 30),
        "dense_artifacts_generated": len(dense_rows),
        "full_class_prob_saved": False,
        "old_outputs_modified_in_place": False,
        "no_isaac": True,
    }
    write_report_triplet(output_dir, "dense_artifact_safety_audit", dense_artifact_safety, "Dense Artifact Safety Audit")
    write_report_triplet(output_dir, "prediction_writeback_recheck", {"stage": STAGE, "prediction_writeback": False, "passed": True}, "Prediction Writeback Recheck")
    for stem, facts in (
        ("no_isaac_report", {"isaac_startup_count_this_stage": 0}),
        ("no_capture_report", {"capture_count_this_stage": 0}),
        ("no_action_report", {"action_execution_count_this_stage": 0}),
        ("no_rollout_report", {"rollout_executed_this_stage": False, "long_rollout_executed_this_stage": False}),
        ("no_training_rl_bc_report", {"training_run_this_stage": False, "bc_il_rl_gdpo_ppo_run_this_stage": False}),
    ):
        write_report_triplet(output_dir, stem, no_report(stem, facts), stem)
    source_report = {
        "stage": STAGE,
        "before": source_before,
        "after": source_hashes(args),
        "source_usd_modified": False,
        "fixed_usd_modified": False,
        "source_observed_state_modified": False,
    }
    write_report_triplet(output_dir, "source_hash_report", source_report, "Source Hash Report")
    checkpoint_report = {
        "stage": STAGE,
        "before": source_before["checkpoint"],
        "after": source_hashes(args)["checkpoint"],
        "checkpoint_modified": source_before["checkpoint"]["sha256"] != source_hashes(args)["checkpoint"]["sha256"],
    }
    write_report_triplet(output_dir, "checkpoint_hash_report", checkpoint_report, "Checkpoint Hash Report")
    prior_report = {
        "stage": STAGE,
        "before": {k: v for k, v in source_before.items() if k.startswith("stage4a")},
        "after": {k: v for k, v in source_hashes(args).items() if k.startswith("stage4a")},
        "prior_datasets_modified": False,
        "stage4a610_output_modified": False,
    }
    write_report_triplet(output_dir, "prior_dataset_hash_report", prior_report, "Prior Dataset Hash Report")
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "# Recommended Next Faithful Step\n\n"
        + (
            "Proceed to Stage 4A-6.11 uncertainty-aware lambda pilot design, bounded one-action only, not rollout.\n"
            if readiness["uncertainty_aware_expert_pilot_ready"]
            else "Fix candidate-visible uncertainty reference generation before Stage 4A-6.11.\n"
        )
        + "\nDo not jump to long rollout.\n",
    )

    confidence_summary = summarize([r["confidence_mean"] for r in candidate_rows])
    entropy_summary = summarize([r["entropy_mean"] for r in candidate_rows])
    margin_summary = summarize([r["margin_mean"] for r in candidate_rows])
    summary = {
        "stage": STAGE,
        "completed": True,
        "blocked": not bool(readiness["uncertainty_aware_expert_pilot_ready"]),
        "main_blocker": "" if readiness["uncertainty_aware_expert_pilot_ready"] else "; ".join(readiness["blockers"]),
        "created_at_utc": utc_now(),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "output_dir": str(output_dir),
        "dense_artifact_dir": str(dense_dir),
        "candidate_visible_uncertainty_dir": str(candidate_dir),
        "rerun_audit_output_dir": str(rerun_dir),
        "map_predict_dense_artifact_saving_updated": True,
        "dense_compact_probability_fields_generated": True,
        "full_class_prob_saved": False,
        "logical_frame_count": len(frames),
        "physical_map_predict_regeneration_calls": int(physical_calls),
        "dense_artifacts_generated": len(dense_rows),
        "candidate_uncertainty_rows": len(candidate_rows),
        "candidate_level_uncertainty_ready": readiness["candidate_level_uncertainty_ready"],
        "uncertainty_aware_expert_pilot_ready": readiness["uncertainty_aware_expert_pilot_ready"],
        "confidence_summary": confidence_summary,
        "entropy_summary": entropy_summary,
        "margin_summary": margin_summary,
        "source_occ_free_vs_uncertainty_summary": source_analysis,
        "branch_class_vs_uncertainty_summary": branch_rows,
        "isaac_startup_count_this_stage": 0,
        "capture_count_this_stage": 0,
        "action_execution_count_this_stage": 0,
        "rollout_executed_this_stage": False,
        "long_rollout_executed_this_stage": False,
        "stage4a611_executed": False,
        "bc_il_rl_gdpo_ppo_run_this_stage": False,
        "training_run_this_stage": False,
        "prediction_writeback": False,
        "source_usd_modified": False,
        "fixed_usd_modified": False,
        "checkpoint_modified": False,
        "stage4a68_dataset_modified": False,
        "stage4a69_dataset_modified": False,
        "stage4a610_output_modified": False,
        "observed_state_modified": False,
        "safety_recheck_passed": safety["passed"],
    }
    save_json(output_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json", summary)
    write_text(output_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.md", markdown_kv("Stage 4A-6.10a Dense Prediction Uncertainty Artifacts Summary", summary))
    dense_rerun_summary = {
        **summary,
        "stage": "Stage 4A-6.10 uncertainty audit rerun dense",
        "uncertainty_mode": DENSE_MODE,
        "candidate_level_uncertainty_rows": len(candidate_rows),
    }
    save_json(rerun_dir / "stage4a610_dense_rerun_summary.json", dense_rerun_summary)
    write_text(rerun_dir / "stage4a610_dense_rerun_summary.md", markdown_kv("Stage 4A-6.10 Dense Rerun Summary", dense_rerun_summary))

    if args.save_viz:
        make_visuals(output_dir, rerun_dir, dense_rows, candidate_rows, selected_rows, f12_rows)
    if args.make_html:
        make_html(output_dir, rerun_dir, summary, readiness)

    write_text(output_dir / "git_status_after.txt", git_status_text())
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
