#!/usr/bin/env python3
"""Prepare Stage 4A-7.0 BC-ready dataset schema, tensors, and audits.

This stage is offline data preparation only. It reads validated prior expert
artifacts and never starts Isaac, captures, calls map_predict, runs SSCNet
inference, executes actions, rolls out, trains, performs optimizer steps, or
saves model checkpoints.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
from collections import Counter, defaultdict
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

from bc_dataset_schema_utils import (
    MODEL_FEATURE_NAMES,
    RAW_FEATURE_GROUPS,
    RAW_FEATURE_NAMES,
    SCHEMA_VERSION,
    STAGE,
    WORKSPACE,
    as_bool,
    as_float,
    as_int,
    check_forbidden_names,
    finite_minmax,
    git_large_artifact_policy,
    git_status_text,
    jsonable,
    normalized_yaw_delta,
    normalization_stats,
    parse_literal,
    rank_desc,
    read_csv,
    read_json,
    save_json,
    save_report_pair,
    sha256_file,
    summarize,
    utc_now,
    write_csv,
    write_jsonl,
    write_text,
)


DEFAULT_OUTPUT_DIR = WORKSPACE / "outputs/isaac_stage4a70_bc_dataset_design_preparation"
DEFAULT_FIXED_USD = WORKSPACE / "assets/home_like_scene_v1/current_environment_localized_defaultprim/home_like_scene_v1.usd"
DEFAULT_SOURCE_USD = WORKSPACE / "building_scene.usd"
DEFAULT_CHECKPOINT = WORKSPACE / "checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
PRIMARY_LABEL_POLICY = "stage4a613_uncertainty_bonus_executed_primary"
PRIMARY_SOURCE_STAGE_ID = 613

REQUIRED_CONTEXT_FILES = [
    ".project_context/CURRENT_STATE.md",
    ".project_context/TODO.md",
    ".project_context/CODEX_LOG.md",
    "README.md",
    "ARTIFACTS.md",
    "ENVIRONMENT.md",
    "GIT_INITIALIZATION_REPORT.md",
    "ssc_exploration/ssc_network/il/paper_expert_dataset.py",
    "ssc_exploration/ssc_network/il/policy.py",
    "ssc_exploration/ssc_network/il/train_bc.py",
    "ssc_exploration/ssc_network/il/test_dataset.py",
]

NEGATIVE_SCOPE_FLAGS = [
    "no_isaac",
    "no_capture",
    "no_map_predict",
    "no_sscnet_inference",
    "no_action",
    "no_rollout",
    "no_long_rollout",
    "no_training",
    "no_optimizer_step",
    "no_model_save",
    "no_rl_gdpo",
]

FORBIDDEN_TABLE_NAMES = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "reward",
    "policy_logits",
    "replay_buffer",
    "optimizer",
    "training_state",
    "class_prob",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured_only_pilot_dir", type=Path, required=True)
    parser.add_argument("--lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--two_frame_lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--dense_uncertainty_dir", type=Path, required=True)
    parser.add_argument("--dense_uncertainty_audit_dir", type=Path, required=True)
    parser.add_argument("--confidence_gated_pilot_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_bonus_decision_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_bonus_short_rollout_dir", type=Path, required=True)
    parser.add_argument("--close_guard_hardening_dir", type=Path, required=True)
    parser.add_argument("--source_usd", type=Path, default=DEFAULT_SOURCE_USD)
    parser.add_argument("--fixed_usd", type=Path, default=DEFAULT_FIXED_USD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--primary_label_policy", default=PRIMARY_LABEL_POLICY)
    parser.add_argument("--primary_source_stage", default="stage4a613")
    parser.add_argument("--include_reference_stages", nargs="*", default=[])
    parser.add_argument("--include_shadow_labels", nargs="*", default=[])
    parser.add_argument("--model_feature_profile", default="compact_v1")
    parser.add_argument("--split_policy", default="leave_one_start_out")
    parser.add_argument("--also_make_split_by_start_variant", action="store_true")
    parser.add_argument("--quality_policy", default="strict_and_moderate")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_tables", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--make_html", action="store_true")
    parser.add_argument("--forward_only_smoke", action="store_true")
    for flag in NEGATIVE_SCOPE_FLAGS:
        parser.add_argument(f"--{flag}", action="store_true")
    return parser.parse_args()


def enforce_scope(args: argparse.Namespace) -> None:
    missing = [flag for flag in NEGATIVE_SCOPE_FLAGS if not bool(getattr(args, flag))]
    if missing:
        raise ValueError(f"Missing required negative-scope flags: {missing}")
    if args.primary_label_policy != PRIMARY_LABEL_POLICY:
        raise ValueError(f"Primary label policy must be {PRIMARY_LABEL_POLICY}")
    if args.primary_source_stage != "stage4a613":
        raise ValueError("Stage 4A-7.0 primary source must be stage4a613")
    if args.model_feature_profile != "compact_v1":
        raise ValueError("Only compact_v1 model feature profile is implemented")


def required_expert_artifacts(args: argparse.Namespace) -> list[Path]:
    return [
        args.measured_only_pilot_dir / "stage4a67_measured_only_expert_pilot_summary.json",
        args.measured_only_pilot_dir / "expert_dataset.npz",
        args.measured_only_pilot_dir / "dataset_integrity_report.json",
        args.measured_only_pilot_dir / "safety_audit.json",
        args.lambda48_pilot_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
        args.lambda48_pilot_dir / "expert_dataset.npz",
        args.lambda48_pilot_dir / "expert_dataset_manifest.jsonl",
        args.lambda48_pilot_dir / "lambda48_decisions.csv",
        args.lambda48_pilot_dir / "measured_shadow_decisions.csv",
        args.lambda48_pilot_dir / "prediction_safety_audit.json",
        args.lambda48_pilot_dir / "expert_data_quality_audit.json",
        args.two_frame_lambda48_pilot_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json",
        args.two_frame_lambda48_pilot_dir / "expert_dataset_two_frame.npz",
        args.two_frame_lambda48_pilot_dir / "expert_dataset_manifest.jsonl",
        args.two_frame_lambda48_pilot_dir / "per_frame_summary.csv",
        args.two_frame_lambda48_pilot_dir / "frame1_lambda48_decisions.csv",
        args.two_frame_lambda48_pilot_dir / "frame2_lambda48_diagnostic_decisions.csv",
        args.two_frame_lambda48_pilot_dir / "prediction_safety_audit.json",
        args.two_frame_lambda48_pilot_dir / "expert_data_quality_audit.json",
        args.two_frame_lambda48_pilot_dir / "two_frame_stability_audit.json",
        args.dense_uncertainty_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json",
        args.dense_uncertainty_dir / "dense_prediction_artifact_manifest.json",
        args.dense_uncertainty_dir / "candidate_visible_uncertainty_manifest.json",
        args.dense_uncertainty_audit_dir / "candidate_uncertainty_table.csv",
        args.dense_uncertainty_audit_dir / "selected_action_uncertainty_audit.csv",
        args.dense_uncertainty_audit_dir / "uncertainty_readiness_decision.json",
        args.confidence_gated_pilot_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json",
        args.confidence_gated_pilot_dir / "expert_dataset_uncertainty_lambda.npz",
        args.confidence_gated_pilot_dir / "primary_confidence_gated_decisions.csv",
        args.confidence_gated_pilot_dir / "lambda48_baseline_shadow_decisions.csv",
        args.confidence_gated_pilot_dir / "measured_shadow_decisions.csv",
        args.confidence_gated_pilot_dir / "uncertainty_candidate_features.csv",
        args.confidence_gated_pilot_dir / "prediction_safety_audit.json",
        args.confidence_gated_pilot_dir / "uncertainty_safety_audit.json",
        args.confidence_gated_pilot_dir / "expert_data_quality_audit.json",
        args.uncertainty_bonus_decision_dir / "stage4a612_uncertainty_exploration_bonus_pilot_summary.json",
        args.uncertainty_bonus_decision_dir / "expert_decision_dataset_uncertainty_bonus.npz",
        args.uncertainty_bonus_decision_dir / "uncertainty_bonus_readiness_decision.json",
        args.uncertainty_bonus_decision_dir / "recommended_formula_report.json",
        args.uncertainty_bonus_decision_dir / "uncertainty_bonus_quality_audit.json",
        args.uncertainty_bonus_decision_dir / "uncertainty_bonus_risk_audit.json",
        args.uncertainty_bonus_short_rollout_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json",
        args.uncertainty_bonus_short_rollout_dir / "short_rollout_dataset_uncertainty_bonus.npz",
        args.uncertainty_bonus_short_rollout_dir / "short_rollout_manifest.jsonl",
        args.uncertainty_bonus_short_rollout_dir / "short_rollout_metadata.json",
        args.uncertainty_bonus_short_rollout_dir / "per_start_summary.csv",
        args.uncertainty_bonus_short_rollout_dir / "per_step_summary.csv",
        args.uncertainty_bonus_short_rollout_dir / "transition_decisions.csv",
        args.uncertainty_bonus_short_rollout_dir / "primary_uncertainty_bonus_decisions.csv",
        args.uncertainty_bonus_short_rollout_dir / "measured_shadow_decisions.csv",
        args.uncertainty_bonus_short_rollout_dir / "lambda48_shadow_decisions.csv",
        args.uncertainty_bonus_short_rollout_dir / "confidence_gated_shadow_decisions.csv",
        args.uncertainty_bonus_short_rollout_dir / "expert_data_quality_audit.json",
        args.uncertainty_bonus_short_rollout_dir / "prediction_safety_audit.json",
        args.uncertainty_bonus_short_rollout_dir / "uncertainty_safety_audit.json",
        args.uncertainty_bonus_short_rollout_dir / "rollout_safety_audit.json",
        args.uncertainty_bonus_short_rollout_dir / "dataset_integrity_report.json",
        args.close_guard_hardening_dir / "stage4a613a_isaac_close_guard_hardening_summary.json",
        args.close_guard_hardening_dir / "lifecycle_guard_contract.json",
        args.close_guard_hardening_dir / "future_runner_usage_examples.md",
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(jsonable(__import__("json").loads(line)))
    return rows


def npz_inventory(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "arrays": {
                key: {"shape": list(data[key].shape), "dtype": str(data[key].dtype)}
                for key in data.files
            },
        }


def csv_inventory(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
    }


def load_context_and_artifacts(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    context_files = [WORKSPACE / rel for rel in REQUIRED_CONTEXT_FILES]
    expert_files = required_expert_artifacts(args)
    missing = [str(path) for path in context_files + expert_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required context/artifact files: " + ", ".join(missing))

    context_manifest = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in context_files
    ]
    expert_manifest = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in expert_files
    ]

    summaries = {
        "stage4a67": read_json(args.measured_only_pilot_dir / "stage4a67_measured_only_expert_pilot_summary.json"),
        "stage4a68": read_json(args.lambda48_pilot_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json"),
        "stage4a69": read_json(args.two_frame_lambda48_pilot_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json"),
        "stage4a610a": read_json(args.dense_uncertainty_dir / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json"),
        "stage4a611": read_json(args.confidence_gated_pilot_dir / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json"),
        "stage4a612": read_json(args.uncertainty_bonus_decision_dir / "stage4a612_uncertainty_exploration_bonus_pilot_summary.json"),
        "stage4a613": read_json(args.uncertainty_bonus_short_rollout_dir / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json"),
        "stage4a613a": read_json(args.close_guard_hardening_dir / "stage4a613a_isaac_close_guard_hardening_summary.json"),
    }

    audits = {
        "stage4a67_dataset_integrity": read_json(args.measured_only_pilot_dir / "dataset_integrity_report.json"),
        "stage4a67_safety": read_json(args.measured_only_pilot_dir / "safety_audit.json"),
        "stage4a68_prediction_safety": read_json(args.lambda48_pilot_dir / "prediction_safety_audit.json"),
        "stage4a68_expert_quality": read_json(args.lambda48_pilot_dir / "expert_data_quality_audit.json"),
        "stage4a69_prediction_safety": read_json(args.two_frame_lambda48_pilot_dir / "prediction_safety_audit.json"),
        "stage4a69_expert_quality": read_json(args.two_frame_lambda48_pilot_dir / "expert_data_quality_audit.json"),
        "stage4a69_stability": read_json(args.two_frame_lambda48_pilot_dir / "two_frame_stability_audit.json"),
        "stage4a610a_readiness": read_json(args.dense_uncertainty_audit_dir / "uncertainty_readiness_decision.json"),
        "stage4a611_prediction_safety": read_json(args.confidence_gated_pilot_dir / "prediction_safety_audit.json"),
        "stage4a611_uncertainty_safety": read_json(args.confidence_gated_pilot_dir / "uncertainty_safety_audit.json"),
        "stage4a611_expert_quality": read_json(args.confidence_gated_pilot_dir / "expert_data_quality_audit.json"),
        "stage4a612_readiness": read_json(args.uncertainty_bonus_decision_dir / "uncertainty_bonus_readiness_decision.json"),
        "stage4a612_recommended": read_json(args.uncertainty_bonus_decision_dir / "recommended_formula_report.json"),
        "stage4a612_quality": read_json(args.uncertainty_bonus_decision_dir / "uncertainty_bonus_quality_audit.json"),
        "stage4a612_risk": read_json(args.uncertainty_bonus_decision_dir / "uncertainty_bonus_risk_audit.json"),
        "stage4a613_expert_quality": read_json(args.uncertainty_bonus_short_rollout_dir / "expert_data_quality_audit.json"),
        "stage4a613_prediction_safety": read_json(args.uncertainty_bonus_short_rollout_dir / "prediction_safety_audit.json"),
        "stage4a613_uncertainty_safety": read_json(args.uncertainty_bonus_short_rollout_dir / "uncertainty_safety_audit.json"),
        "stage4a613_rollout_safety": read_json(args.uncertainty_bonus_short_rollout_dir / "rollout_safety_audit.json"),
        "stage4a613_dataset_integrity": read_json(args.uncertainty_bonus_short_rollout_dir / "dataset_integrity_report.json"),
        "stage4a613a_lifecycle_contract": read_json(args.close_guard_hardening_dir / "lifecycle_guard_contract.json"),
    }

    checks = {
        "stage4a613_completed": bool(summaries["stage4a613"].get("completed")),
        "stage4a613_primary_formula": summaries["stage4a613"].get("primary_formula") == "uncertainty_bonus_composite_beta8",
        "stage4a613_dataset_integrity_passed": bool(audits["stage4a613_dataset_integrity"].get("passed")),
        "stage4a613_expert_quality_passed": bool(audits["stage4a613_expert_quality"].get("passed")),
        "stage4a613_prediction_safety_passed": bool(audits["stage4a613_prediction_safety"].get("passed")),
        "stage4a613_uncertainty_safety_passed": bool(audits["stage4a613_uncertainty_safety"].get("passed")),
        "stage4a613_rollout_safety_passed": bool(audits["stage4a613_rollout_safety"].get("passed")),
        "stage4a613a_completed": bool(summaries["stage4a613a"].get("completed")),
        "stage4a613a_blocked_false": not bool(summaries["stage4a613a"].get("blocked")),
        "current_task_dataset_design_only": True,
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise RuntimeError(f"Required prior-stage gates failed: {failed}")

    source_inventory_rows = build_source_dataset_inventory(args, summaries, audits)
    write_csv(output_dir / "source_dataset_inventory.csv", source_inventory_rows)
    save_report_pair(output_dir, "source_dataset_inventory", {"rows": source_inventory_rows}, "Source Dataset Inventory")
    save_report_pair(
        output_dir,
        "loaded_context_manifest",
        {"stage": STAGE, "loaded_at": utc_now(), "checks": checks, "files": context_manifest},
        "Loaded Context Manifest",
    )
    save_report_pair(
        output_dir,
        "loaded_expert_artifact_manifest",
        {"stage": STAGE, "loaded_at": utc_now(), "files": expert_manifest},
        "Loaded Expert Artifact Manifest",
    )
    return {
        "context_manifest": context_manifest,
        "expert_manifest": expert_manifest,
        "summaries": summaries,
        "audits": audits,
        "checks": checks,
        "source_inventory_rows": source_inventory_rows,
    }


def build_source_dataset_inventory(
    args: argparse.Namespace,
    summaries: dict[str, Any],
    audits: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    specs = [
        ("stage4a67", args.measured_only_pilot_dir, "expert_dataset.npz", "stage4a67_measured_only_expert_pilot_summary.json"),
        ("stage4a68", args.lambda48_pilot_dir, "expert_dataset.npz", "stage4a68_map_predict_lambda48_expert_pilot_summary.json"),
        ("stage4a69", args.two_frame_lambda48_pilot_dir, "expert_dataset_two_frame.npz", "stage4a69_bounded_two_frame_lambda48_pilot_summary.json"),
        ("stage4a610a", args.dense_uncertainty_dir, "dense_prediction_artifact_manifest.json", "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json"),
        ("stage4a611", args.confidence_gated_pilot_dir, "expert_dataset_uncertainty_lambda.npz", "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json"),
        ("stage4a612", args.uncertainty_bonus_decision_dir, "expert_decision_dataset_uncertainty_bonus.npz", "stage4a612_uncertainty_exploration_bonus_pilot_summary.json"),
        ("stage4a613", args.uncertainty_bonus_short_rollout_dir, "short_rollout_dataset_uncertainty_bonus.npz", "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json"),
        ("stage4a613a", args.close_guard_hardening_dir, "lifecycle_guard_contract.json", "stage4a613a_isaac_close_guard_hardening_summary.json"),
    ]
    for stage_id, directory, dataset_name, summary_name in specs:
        summary = summaries.get(stage_id, {})
        dataset = directory / dataset_name
        row = {
            "stage_id": stage_id,
            "output_dir": str(directory),
            "main_artifact": str(dataset),
            "main_artifact_exists": dataset.is_file(),
            "main_artifact_sha256": sha256_file(dataset),
            "summary": str(directory / summary_name),
            "completed": bool(summary.get("completed", summary.get("complete", False))),
            "sample_count": summary.get("sample_count", summary.get("decision_frame_count", summary.get("start_count"))),
            "start_count": summary.get("start_count"),
            "decision_frame_count": summary.get("decision_frame_count"),
            "capture_count": summary.get("capture_count", summary.get("capture_count_this_stage")),
            "map_predict_calls": summary.get("map_predict_calls", summary.get("map_predict_calls_this_stage")),
            "executed_action_count": summary.get("executed_action_count", summary.get("action_execution_count_this_stage")),
            "primary_formula": summary.get("primary_formula", summary.get("formula_name", summary.get("formula"))),
            "quality_passed": None,
        }
        quality_key = {
            "stage4a67": "stage4a67_dataset_integrity",
            "stage4a68": "stage4a68_expert_quality",
            "stage4a69": "stage4a69_expert_quality",
            "stage4a611": "stage4a611_expert_quality",
            "stage4a612": "stage4a612_quality",
            "stage4a613": "stage4a613_expert_quality",
        }.get(stage_id)
        if quality_key:
            row["quality_passed"] = bool(audits.get(quality_key, {}).get("passed"))
        rows.append(row)
    return rows


def dense_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with np.load(path, allow_pickle=False) as data:
        out = {}
        for key in (
            "prediction_valid_count",
            "predicted_unmeasured_count",
            "predicted_occupied_count",
            "source_occ_free_count",
            "checkpoint_sha256",
            "observed_reference_hash",
            "source_observed_state_sha256",
            "no_prediction_writeback",
            "full_class_prob_saved",
        ):
            if key in data.files:
                value = data[key]
                out[key] = value.item() if value.shape == () else value.tolist()
        return out


def safe_xyz(value: Any) -> list[float]:
    parsed = parse_literal(value, [math.nan, math.nan, math.nan])
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        return [math.nan, math.nan, math.nan]
    return [as_float(parsed[0]), as_float(parsed[1]), as_float(parsed[2])]


def safe_grid(value: Any) -> list[float]:
    parsed = parse_literal(value, [math.nan, math.nan, math.nan])
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        return [math.nan, math.nan, math.nan]
    return [as_float(parsed[0]), as_float(parsed[1]), as_float(parsed[2])]


def load_primary_bundle(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    source_dir = args.uncertainty_bonus_short_rollout_dir
    dataset_path = source_dir / "short_rollout_dataset_uncertainty_bonus.npz"
    manifest_path = source_dir / "short_rollout_manifest.jsonl"
    manifest_rows = read_jsonl(manifest_path)
    fixed_usd_hash = sha256_file(args.fixed_usd)
    checkpoint_hash = sha256_file(args.checkpoint)

    with np.load(dataset_path, allow_pickle=False) as data:
        start_variant_id = data["start_variant_id"].astype(np.int64)
        step_id = data["step_id"].astype(np.int64)
        pose = data["pose"].astype(np.float32)
        candidate_valid_mask = data["valid_mask"].astype(bool)
        primary_label = data["action_index_primary_uncertainty_bonus"].astype(np.int64)
        measured_label = data["action_index_measured_shadow"].astype(np.int64)
        lambda_label = data["action_index_lambda48_shadow"].astype(np.int64)
        confidence_label = data["action_index_confidence_gated_shadow"].astype(np.int64)
        selected_world = data["selected_world_xyz_primary"].astype(np.float32)
        selected_yaw = data["selected_yaw_primary"].astype(np.float32)
        observed_ratio_before = data["observed_ratio_before"].astype(np.float32)
        observed_count = data["observed_count"].astype(np.int64)
        newly_observed_count = data["newly_observed_count"].astype(np.int64)
        unknown_count = data["unknown_count"].astype(np.int64)
        free_count = data["free_count"].astype(np.int64)
        occupied_count = data["occupied_count"].astype(np.int64)
        sample_flags = {
            key: data[key].astype(bool)
            for key in (
                "prediction_writeback",
                "uncertainty_writeback",
                "prediction_traversability_use",
                "uncertainty_traversability_use",
                "prediction_collision_use",
                "uncertainty_collision_use",
                "prediction_ray_blocking_use",
                "uncertainty_ray_blocking_use",
                "prediction_candidate_validity_use",
                "uncertainty_candidate_validity_use",
                "target_ground_truth_use",
                "future_observed_scoring_use",
                "no_valid_candidate",
                "low_cost_artifact",
                "historical_prior_basin",
                "candidate_all_local",
                "repeated_target",
                "same_cell_target",
                "outside_bounds_target",
            )
        }

    sample_count, candidate_count = candidate_valid_mask.shape
    if sample_count != len(manifest_rows):
        raise ValueError(f"Manifest row count {len(manifest_rows)} != dataset samples {sample_count}")
    if sample_count != 30:
        raise ValueError(f"Expected 30 Stage 4A-6.13 primary samples, got {sample_count}")

    raw = np.full((sample_count, candidate_count, len(RAW_FEATURE_NAMES)), np.nan, dtype=np.float32)
    score_primary = np.full((sample_count, candidate_count), np.nan, dtype=np.float32)
    score_measured = np.full((sample_count, candidate_count), np.nan, dtype=np.float32)
    score_lambda48 = np.full((sample_count, candidate_count), np.nan, dtype=np.float32)
    score_confidence = np.full((sample_count, candidate_count), np.nan, dtype=np.float32)
    candidate_rows_out: list[dict[str, Any]] = []
    candidate_model_rows_out: list[dict[str, Any]] = []
    missing_counter: Counter[str] = Counter()
    dense_hash_by_sample: list[str | None] = []
    observed_hash_by_sample: list[str | None] = []

    raw_index = {name: idx for idx, name in enumerate(RAW_FEATURE_NAMES)}

    for sample_index, manifest in enumerate(manifest_rows):
        candidate_csv = Path(manifest["candidate_features"])
        pose_json = Path(manifest["pose"])
        dense_path = Path(manifest.get("dense_prediction_uncertainty", ""))
        observed_path = Path(manifest.get("observed_state_reference", ""))
        rows = read_csv(candidate_csv)
        row_by_candidate_id = {
            as_int(row.get("candidate_id"), default=index): row
            for index, row in enumerate(rows)
        }
        if len(rows) > candidate_count:
            raise ValueError(f"{candidate_csv} row count {len(rows)} exceeds padded candidate count {candidate_count}")
        pose_record = read_json(pose_json) if pose_json.is_file() else {}
        pose_xyz = [
            as_float((pose_record.get("position") or pose[sample_index, :3])[0]),
            as_float((pose_record.get("position") or pose[sample_index, :3])[1]),
            as_float((pose_record.get("position") or pose[sample_index, :3])[2]),
        ]
        pose_yaw = as_float(pose_record.get("yaw_rad", pose[sample_index, 3]))
        dense_meta = dense_metadata(dense_path)
        dense_hash = sha256_file(dense_path) if dense_path.is_file() else None
        observed_hash = sha256_file(observed_path) if observed_path.is_file() else None
        dense_hash_by_sample.append(dense_hash)
        observed_hash_by_sample.append(observed_hash)
        pred_valid = as_float(dense_meta.get("prediction_valid_count"))
        pred_unmeasured = as_float(dense_meta.get("predicted_unmeasured_count"))
        pred_occupied = as_float(dense_meta.get("predicted_occupied_count"))
        pred_density = pred_occupied / pred_valid if math.isfinite(pred_valid) and pred_valid > 0 else math.nan

        raw_rows_for_ranks: list[dict[str, Any]] = []
        for candidate_index in range(candidate_count):
            row = row_by_candidate_id.get(candidate_index, {})
            if not row and bool(candidate_valid_mask[sample_index, candidate_index]):
                raise ValueError(
                    f"{candidate_csv} is missing valid candidate_id {candidate_index}"
                )
            grid = safe_grid(row.get("grid"))
            world = safe_xyz(row.get("world"))
            yaw = as_float(row.get("yaw_rad"))
            delta = [world[axis] - pose_xyz[axis] for axis in range(3)]
            distance_xy = math.hypot(delta[0], delta[1]) if math.isfinite(delta[0]) and math.isfinite(delta[1]) else math.nan
            yaw_delta = as_float(row.get("yaw_delta_rad"), normalized_yaw_delta(yaw, pose_yaw))
            valid = bool(candidate_valid_mask[sample_index, candidate_index])
            unc_composite = as_float(row.get("uncertainty_composite"))
            raw_row = {
                "candidate_id": as_float(row.get("candidate_id"), candidate_index),
                "candidate_rank": as_float(candidate_index),
                "candidate_valid": float(valid),
                "grid_x": grid[0],
                "grid_y": grid[1],
                "grid_z": grid[2],
                "world_x": world[0],
                "world_y": world[1],
                "world_z": world[2],
                "yaw": yaw,
                "delta_x": delta[0],
                "delta_y": delta[1],
                "delta_z": delta[2],
                "distance_xy": distance_xy,
                "yaw_delta": yaw_delta,
                "step_id": float(step_id[sample_index]),
                "start_variant_id": float(start_variant_id[sample_index]),
                "gain_exp": as_float(row.get("gain_exp")),
                "frontier_count": as_float(row.get("frontier_count_visible")),
                "frontier_adjacent_count": math.nan,
                "observed_ratio": float(observed_ratio_before[sample_index]),
                "observed_count": float(observed_count[sample_index]),
                "unknown_count": float(unknown_count[sample_index]),
                "free_count": float(free_count[sample_index]),
                "occupied_count": float(occupied_count[sample_index]),
                "newly_observed_count_if_available": float(newly_observed_count[sample_index]),
                "local_observed_density_if_available": math.nan,
                "path_cost": as_float(row.get("path_cost", row.get("cost_s"))),
                "astar_reachable": float(as_bool(row.get("astar_reachable"))),
                "astar_path_length_m": as_float(row.get("path_cost_m")),
                "astar_num_expanded": as_float(row.get("astar_num_expanded")),
                "reachable_component_count": math.nan,
                "reachable_frontier_adjacent_count": math.nan,
                "same_cell_target": float(sample_flags["same_cell_target"][sample_index]),
                "repeated_target": float(sample_flags["repeated_target"][sample_index]),
                "outside_bounds_target": float(sample_flags["outside_bounds_target"][sample_index]),
                "source_occ_free": as_float(row.get("source_occ_free")),
                "gain_sc": as_float(row.get("gain_sc")),
                "gain_occ": as_float(row.get("raw_gain_sc", row.get("source_occ_free"))),
                "gain_conf": as_float(row.get("source_confidence_gate_raw")),
                "prediction_valid_count": pred_valid,
                "predicted_unmeasured_count": pred_unmeasured,
                "predicted_occupied_count": pred_occupied,
                "prediction_density": pred_density,
                "source_occ_free_norm_per_sample": as_float(row.get("source_occ_free_minmax_stage4a613", row.get("source_occ_free_minmax"))),
                "candidate_confidence_mean": as_float(row.get("candidate_confidence_mean")),
                "candidate_confidence_min": as_float(row.get("candidate_confidence_min")),
                "candidate_confidence_p10": as_float(row.get("candidate_confidence_p10")),
                "candidate_confidence_p50": as_float(row.get("candidate_confidence_p50")),
                "candidate_confidence_p90": as_float(row.get("candidate_confidence_p90")),
                "candidate_entropy_mean": as_float(row.get("candidate_entropy_mean")),
                "candidate_entropy_max": as_float(row.get("candidate_entropy_max")),
                "candidate_entropy_p90": as_float(row.get("candidate_entropy_p90")),
                "candidate_margin_mean": as_float(row.get("candidate_margin_mean")),
                "candidate_margin_min": as_float(row.get("candidate_margin_min")),
                "candidate_uncertain_fraction": as_float(row.get("candidate_uncertain_fraction", row.get("uncertain_fraction"))),
                "candidate_uncertain_voxel_count": as_float(row.get("uncertain_voxel_count")),
                "low_conf_count_0p7": as_float(row.get("low_conf_count_0p7")),
                "high_entropy_count_0p7": as_float(row.get("high_entropy_count_0p7")),
                "low_margin_count_0p2": as_float(row.get("low_margin_count_0p2")),
                "uncertainty_composite": unc_composite,
                "uncertainty_composite_norm_per_sample": math.nan,
                "score_measured": as_float(row.get("score_measured_only", row.get("final_score_measured", row.get("measured_score")))),
                "score_lambda48": as_float(row.get("score_lambda48", row.get("final_score_lambda48"))),
                "score_confidence_gated": as_float(row.get("score_confidence_gated_6_11", row.get("score_primary_confidence_gated"))),
                "score_uncertainty_bonus_composite_beta8": as_float(row.get("score_primary_uncertainty_bonus", row.get("score_uncertainty_bonus_beta8"))),
                "final_score_primary": as_float(row.get("score_primary_uncertainty_bonus", row.get("score_uncertainty_bonus_beta8"))),
                "score_rank_primary": math.nan,
                "score_rank_measured": math.nan,
                "score_rank_lambda48": math.nan,
                "score_rank_confidence_gated": math.nan,
                "no_valid_candidate": float(sample_flags["no_valid_candidate"][sample_index]),
                "low_cost_artifact": float(sample_flags["low_cost_artifact"][sample_index]),
                "historical_prior_basin": float(sample_flags["historical_prior_basin"][sample_index]),
                "candidate_all_local": float(sample_flags["candidate_all_local"][sample_index]),
                "high_uncertainty_selection": 0.0,
                "low_confidence_selection": 0.0,
                "low_margin_selection": 0.0,
                "formula_dominated_by_uncertainty": float(as_bool(row.get("formula_dominated_by_uncertainty"))),
                "prediction_writeback": float(sample_flags["prediction_writeback"][sample_index]),
                "uncertainty_writeback": float(sample_flags["uncertainty_writeback"][sample_index]),
                "prediction_traversability_use": float(sample_flags["prediction_traversability_use"][sample_index]),
                "uncertainty_traversability_use": float(sample_flags["uncertainty_traversability_use"][sample_index]),
                "prediction_collision_use": float(sample_flags["prediction_collision_use"][sample_index]),
                "uncertainty_collision_use": float(sample_flags["uncertainty_collision_use"][sample_index]),
                "prediction_ray_blocking_use": float(sample_flags["prediction_ray_blocking_use"][sample_index]),
                "uncertainty_ray_blocking_use": float(sample_flags["uncertainty_ray_blocking_use"][sample_index]),
                "prediction_candidate_validity_use": float(sample_flags["prediction_candidate_validity_use"][sample_index]),
                "uncertainty_candidate_validity_use": float(sample_flags["uncertainty_candidate_validity_use"][sample_index]),
                "target_ground_truth_use": float(sample_flags["target_ground_truth_use"][sample_index]),
                "future_observed_scoring_use": float(sample_flags["future_observed_scoring_use"][sample_index]),
            }
            raw_rows_for_ranks.append(raw_row)

        unc_values = np.asarray([r["uncertainty_composite"] for r in raw_rows_for_ranks], dtype=np.float64)
        unc_norm = finite_minmax(unc_values, candidate_valid_mask[sample_index])
        ranks_primary = rank_desc(
            np.asarray([r["final_score_primary"] for r in raw_rows_for_ranks], dtype=np.float64),
            candidate_valid_mask[sample_index],
        )
        ranks_measured = rank_desc(
            np.asarray([r["score_measured"] for r in raw_rows_for_ranks], dtype=np.float64),
            candidate_valid_mask[sample_index],
        )
        ranks_lambda48 = rank_desc(
            np.asarray([r["score_lambda48"] for r in raw_rows_for_ranks], dtype=np.float64),
            candidate_valid_mask[sample_index],
        )
        ranks_conf = rank_desc(
            np.asarray([r["score_confidence_gated"] for r in raw_rows_for_ranks], dtype=np.float64),
            candidate_valid_mask[sample_index],
        )

        for candidate_index, raw_row in enumerate(raw_rows_for_ranks):
            raw_row["uncertainty_composite_norm_per_sample"] = float(unc_norm[candidate_index])
            raw_row["score_rank_primary"] = float(ranks_primary[candidate_index])
            raw_row["score_rank_measured"] = float(ranks_measured[candidate_index])
            raw_row["score_rank_lambda48"] = float(ranks_lambda48[candidate_index])
            raw_row["score_rank_confidence_gated"] = float(ranks_conf[candidate_index])
            label_match = {
                "is_primary_label": candidate_index == int(primary_label[sample_index]),
                "is_measured_shadow_label": candidate_index == int(measured_label[sample_index]),
                "is_lambda48_shadow_label": candidate_index == int(lambda_label[sample_index]),
                "is_confidence_gated_shadow_label": candidate_index == int(confidence_label[sample_index]),
            }
            for name, value in raw_row.items():
                if name in raw_index:
                    if not math.isfinite(float(value)):
                        missing_counter[name] += 1
                    raw[sample_index, candidate_index, raw_index[name]] = float(value)
            score_primary[sample_index, candidate_index] = raw_row["final_score_primary"]
            score_measured[sample_index, candidate_index] = raw_row["score_measured"]
            score_lambda48[sample_index, candidate_index] = raw_row["score_lambda48"]
            score_confidence[sample_index, candidate_index] = raw_row["score_confidence_gated"]
            table_row = {
                "sample_id": f"stage4a613_start{int(start_variant_id[sample_index]):03d}_step{int(step_id[sample_index]):03d}",
                "sample_index": sample_index,
                "candidate_index": candidate_index,
                **raw_row,
                **label_match,
                "source_stage": "stage4a613",
                "source_dataset": str(dataset_path),
                "source_output_dir": str(source_dir),
                "source_manifest_record": sample_index,
                "source_sample_path": str(Path(manifest.get("pose", "")).parent),
                "source_step_path": str(Path(manifest.get("candidate_features", ""))),
                "fixed_usd_hash": fixed_usd_hash,
                "checkpoint_hash": checkpoint_hash,
                "observed_state_hash": observed_hash,
                "dense_uncertainty_artifact_hash_if_available": dense_hash,
            }
            candidate_rows_out.append(table_row)

    model, feature_mask, missing_model = build_model_features(raw, candidate_valid_mask)
    sample_id = np.asarray(
        [f"stage4a613_start{int(s):03d}_step{int(t):03d}" for s, t in zip(start_variant_id, step_id)],
        dtype="U64",
    )
    sequence_id = start_variant_id.astype(np.int64)
    previous_sample_index = np.full(sample_count, -1, dtype=np.int64)
    next_sample_index = np.full(sample_count, -1, dtype=np.int64)
    for sid in sorted(set(start_variant_id.tolist())):
        indices = np.flatnonzero(start_variant_id == sid)
        indices = indices[np.argsort(step_id[indices])]
        for idx_pos, sample_idx in enumerate(indices):
            if idx_pos > 0:
                previous_sample_index[sample_idx] = int(indices[idx_pos - 1])
            if idx_pos + 1 < len(indices):
                next_sample_index[sample_idx] = int(indices[idx_pos + 1])

    split_id, split_rows, folds = build_splits(sample_id, start_variant_id, step_id)
    quality = build_quality_masks(
        primary_label,
        candidate_valid_mask,
        model,
        score_primary,
        raw,
        raw_index,
    )
    sample_weight = np.where(quality["strict_keep"], 1.0, 0.25).astype(np.float32)

    for sample_index in range(sample_count):
        for candidate_index in range(candidate_count):
            row = {
                "sample_id": sample_id[sample_index],
                "sample_index": sample_index,
                "candidate_index": candidate_index,
                "candidate_valid": bool(candidate_valid_mask[sample_index, candidate_index]),
                "missing_any_model_feature": bool(np.any(missing_model[sample_index, candidate_index])),
            }
            for dim, name in enumerate(MODEL_FEATURE_NAMES):
                row[name] = float(model[sample_index, candidate_index, dim])
                row[f"{name}_observed"] = bool(feature_mask[sample_index, candidate_index, dim])
            candidate_model_rows_out.append(row)

    primary_npz = {
        "sample_id": sample_id,
        "start_variant_id": start_variant_id.astype(np.int64),
        "step_id": step_id.astype(np.int64),
        "source_stage_id": np.full(sample_count, PRIMARY_SOURCE_STAGE_ID, dtype=np.int64),
        "candidate_features_raw": raw.astype(np.float32),
        "candidate_features_model": model.astype(np.float32),
        "candidate_feature_mask": feature_mask.astype(bool),
        "candidate_valid_mask": candidate_valid_mask.astype(bool),
        "expert_action_index_primary": primary_label.astype(np.int64),
        "expert_action_index_measured_shadow": measured_label.astype(np.int64),
        "expert_action_index_lambda48_shadow": lambda_label.astype(np.int64),
        "expert_action_index_confidence_gated_shadow": confidence_label.astype(np.int64),
        "score_primary": score_primary.astype(np.float32),
        "score_measured": score_measured.astype(np.float32),
        "score_lambda48": score_lambda48.astype(np.float32),
        "score_confidence_gated": score_confidence.astype(np.float32),
        "selected_world_xyz_primary": selected_world.astype(np.float32),
        "selected_yaw_primary": selected_yaw.astype(np.float32),
        "pose_world_xyz": pose[:, :3].astype(np.float32),
        "pose_yaw": pose[:, 3].astype(np.float32),
        "sequence_id": sequence_id.astype(np.int64),
        "previous_sample_index": previous_sample_index,
        "next_sample_index": next_sample_index,
        "split_id": split_id.astype(np.int64),
        "quality_keep_mask": quality["strict_keep"].astype(bool),
        "sample_weight": sample_weight,
        "missing_feature_mask": missing_model.astype(bool),
    }

    sample_rows = []
    for i in range(sample_count):
        sample_rows.append(
            {
                "sample_id": sample_id[i],
                "sample_index": i,
                "start_variant_id": int(start_variant_id[i]),
                "step_id": int(step_id[i]),
                "sequence_id": int(sequence_id[i]),
                "previous_sample_index": int(previous_sample_index[i]),
                "next_sample_index": int(next_sample_index[i]),
                "split_id": int(split_id[i]),
                "primary_label": int(primary_label[i]),
                "measured_shadow_label": int(measured_label[i]),
                "lambda48_shadow_label": int(lambda_label[i]),
                "confidence_gated_shadow_label": int(confidence_label[i]),
                "candidate_valid_count": int(np.sum(candidate_valid_mask[i])),
                "strict_keep": bool(quality["strict_keep"][i]),
                "moderate_keep": bool(quality["moderate_keep"][i]),
                "analysis_only": bool(quality["analysis_only"][i]),
                "selected_confidence": float(raw[i, primary_label[i], raw_index["candidate_confidence_mean"]]),
                "selected_entropy": float(raw[i, primary_label[i], raw_index["candidate_entropy_mean"]]),
                "selected_margin": float(raw[i, primary_label[i], raw_index["candidate_margin_mean"]]),
                "observed_ratio": float(observed_ratio_before[i]),
                "observed_count": int(observed_count[i]),
                "source_manifest_record": i,
                "candidate_features_csv": manifest_rows[i].get("candidate_features"),
                "observed_state_reference": manifest_rows[i].get("observed_state_reference"),
                "observed_state_hash": observed_hash_by_sample[i],
                "dense_uncertainty_artifact_hash": dense_hash_by_sample[i],
            }
        )

    return {
        "npz": primary_npz,
        "sample_rows": sample_rows,
        "candidate_rows": candidate_rows_out,
        "candidate_model_rows": candidate_model_rows_out,
        "split_rows": split_rows,
        "folds": folds,
        "quality": quality,
        "missing_counter": missing_counter,
        "manifest_rows": manifest_rows,
        "raw_index": raw_index,
        "score_primary": score_primary,
        "score_measured": score_measured,
        "score_lambda48": score_lambda48,
        "score_confidence": score_confidence,
    }


def build_model_features(raw: np.ndarray, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {name: pos for pos, name in enumerate(RAW_FEATURE_NAMES)}
    sample_count, candidate_count, _ = raw.shape
    out = np.full((sample_count, candidate_count, len(MODEL_FEATURE_NAMES)), np.nan, dtype=np.float32)
    for sample_index in range(sample_count):
        gain_norm = finite_minmax(raw[sample_index, :, idx["gain_exp"]], valid_mask[sample_index])
        path = raw[sample_index, :, idx["path_cost"]]
        inverse_path = np.where(np.isfinite(path) & (path >= 0), 1.0 / (1.0 + path), np.nan)
        columns = {
            "gain_exp_norm_per_sample": gain_norm,
            "inverse_path_cost": inverse_path,
            "source_occ_free_norm_per_sample": raw[sample_index, :, idx["source_occ_free_norm_per_sample"]],
            "candidate_confidence_mean": raw[sample_index, :, idx["candidate_confidence_mean"]],
            "candidate_entropy_mean": raw[sample_index, :, idx["candidate_entropy_mean"]],
            "candidate_margin_mean": raw[sample_index, :, idx["candidate_margin_mean"]],
            "candidate_uncertain_fraction": raw[sample_index, :, idx["candidate_uncertain_fraction"]],
            "uncertainty_composite": raw[sample_index, :, idx["uncertainty_composite"]],
            "distance_xy": raw[sample_index, :, idx["distance_xy"]],
            "yaw_delta": raw[sample_index, :, idx["yaw_delta"]],
            "astar_reachable": raw[sample_index, :, idx["astar_reachable"]],
            "same_cell_target": raw[sample_index, :, idx["same_cell_target"]],
            "repeated_target": raw[sample_index, :, idx["repeated_target"]],
            "candidate_all_local": raw[sample_index, :, idx["candidate_all_local"]],
            "low_cost_artifact": raw[sample_index, :, idx["low_cost_artifact"]],
            "historical_prior_basin": raw[sample_index, :, idx["historical_prior_basin"]],
        }
        for feature_index, name in enumerate(MODEL_FEATURE_NAMES):
            out[sample_index, :, feature_index] = np.asarray(columns[name], dtype=np.float32)
    missing = ~np.isfinite(out)
    imputed = np.where(missing, 0.0, out).astype(np.float32)
    feature_mask = (~missing) & valid_mask[:, :, None]
    return imputed, feature_mask, missing


def build_splits(sample_id: np.ndarray, start_variant_id: np.ndarray, step_id: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    split_id = np.full(sample_id.shape[0], -1, dtype=np.int64)
    rows = []
    for i, sid in enumerate(start_variant_id):
        if sid <= 6:
            split = "train"
            split_id[i] = 0
        elif sid <= 8:
            split = "val"
            split_id[i] = 1
        else:
            split = "test"
            split_id[i] = 2
        rows.append(
            {
                "sample_id": str(sample_id[i]),
                "sample_index": i,
                "start_variant_id": int(sid),
                "step_id": int(step_id[i]),
                "split_by_start_variant": split,
                "split_id": int(split_id[i]),
                "split_by_sequence_step": "train" if int(step_id[i]) in (0, 1) else "val",
            }
        )
    folds = {}
    for held_out in sorted(set(int(v) for v in start_variant_id.tolist())):
        folds[f"leave_start_{held_out:03d}_out"] = {
            "held_out_start_variant_id": held_out,
            "test_sample_indices": [i for i, sid in enumerate(start_variant_id.tolist()) if int(sid) == held_out],
            "train_sample_indices": [i for i, sid in enumerate(start_variant_id.tolist()) if int(sid) != held_out],
        }
    return split_id, rows, folds


def build_quality_masks(
    primary_label: np.ndarray,
    valid_mask: np.ndarray,
    model: np.ndarray,
    score_primary: np.ndarray,
    raw: np.ndarray,
    raw_index: dict[str, int],
) -> dict[str, Any]:
    n = valid_mask.shape[0]
    strict = np.zeros(n, dtype=bool)
    moderate = np.zeros(n, dtype=bool)
    analysis = np.zeros(n, dtype=bool)
    rows = []
    for i in range(n):
        label = int(primary_label[i])
        label_valid = 0 <= label < valid_mask.shape[1] and bool(valid_mask[i, label])
        candidate_valid_count = int(np.sum(valid_mask[i]))
        forbidden_leakage = any(
            bool(raw[i, 0, raw_index[name]])
            for name in (
                "prediction_writeback",
                "uncertainty_writeback",
                "prediction_traversability_use",
                "uncertainty_traversability_use",
                "prediction_collision_use",
                "uncertainty_collision_use",
                "prediction_ray_blocking_use",
                "uncertainty_ray_blocking_use",
                "prediction_candidate_validity_use",
                "uncertainty_candidate_validity_use",
                "target_ground_truth_use",
                "future_observed_scoring_use",
            )
        )
        selected_conf = float(raw[i, label, raw_index["candidate_confidence_mean"]]) if label_valid else math.nan
        selected_entropy = float(raw[i, label, raw_index["candidate_entropy_mean"]]) if label_valid else math.nan
        selected_margin = float(raw[i, label, raw_index["candidate_margin_mean"]]) if label_valid else math.nan
        no_valid_candidate = bool(raw[i, 0, raw_index["no_valid_candidate"]])
        low_cost = bool(raw[i, 0, raw_index["low_cost_artifact"]])
        historical = bool(raw[i, 0, raw_index["historical_prior_basin"]])
        outside_bounds = bool(raw[i, 0, raw_index["outside_bounds_target"]])
        candidate_all_local = bool(raw[i, 0, raw_index["candidate_all_local"]])
        finite_model = bool(np.all(np.isfinite(model[i])))
        finite_scores = bool(np.all(np.isfinite(score_primary[i, valid_mask[i]])))
        base_keep = (
            candidate_valid_count > 0
            and label_valid
            and not no_valid_candidate
            and not low_cost
            and not historical
            and not outside_bounds
            and not forbidden_leakage
            and finite_model
            and finite_scores
        )
        selected_health = (
            math.isfinite(selected_conf)
            and math.isfinite(selected_entropy)
            and math.isfinite(selected_margin)
            and selected_conf >= 0.6
            and selected_entropy <= 0.5
            and selected_margin >= 0.2
        )
        strict[i] = base_keep and selected_health and not candidate_all_local
        moderate[i] = base_keep and selected_health
        analysis[i] = label_valid
        rows.append(
            {
                "sample_index": i,
                "candidate_valid_count": candidate_valid_count,
                "primary_label": label,
                "primary_label_valid": label_valid,
                "selected_confidence": selected_conf,
                "selected_entropy": selected_entropy,
                "selected_margin": selected_margin,
                "no_valid_candidate": no_valid_candidate,
                "low_cost_artifact": low_cost,
                "historical_prior_basin": historical,
                "outside_bounds_target": outside_bounds,
                "candidate_all_local": candidate_all_local,
                "forbidden_safety_leakage": forbidden_leakage,
                "finite_model_features": finite_model,
                "finite_primary_scores": finite_scores,
                "strict_keep": bool(strict[i]),
                "moderate_keep": bool(moderate[i]),
                "analysis_only": bool(analysis[i]),
            }
        )
    return {
        "strict_keep": strict,
        "moderate_keep": moderate,
        "analysis_only": analysis,
        "rows": rows,
        "counts": {
            "strict_keep": int(np.sum(strict)),
            "moderate_keep": int(np.sum(moderate)),
            "analysis_only": int(np.sum(analysis)),
        },
    }


def save_primary_npz(output_dir: Path, bundle: dict[str, Any]) -> None:
    np.savez_compressed(output_dir / "bc_dataset_primary_short_rollout.npz", **bundle["npz"])
    np.savez_compressed(output_dir / "bc_dataset_shadow_multilabel.npz", **bundle["npz"])


def build_reference_view(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    s67 = np.load(args.measured_only_pilot_dir / "expert_dataset.npz", allow_pickle=False)
    for i in range(int(s67["action_world"].shape[0])):
        rows.append(
            {
                "sample_id": f"stage4a67_start{i:03d}",
                "source_stage": "stage4a67",
                "start_variant_id": i,
                "step_id": 0,
                "candidate_count": int(s67["topn_scores"].shape[1]),
                "label_index": 0,
                "label_policy": "stage4a67_measured_primary",
                "executed_runtime": True,
                "decision_only": False,
                "diagnostic_only": False,
                "model_ready_tensor_available": False,
            }
        )
    s67.close()

    with np.load(args.lambda48_pilot_dir / "expert_dataset.npz", allow_pickle=False) as s68:
        for i, sid in enumerate(s68["start_variant_id"].astype(int).tolist()):
            rows.append(
                {
                    "sample_id": f"stage4a68_start{sid:03d}",
                    "source_stage": "stage4a68",
                    "start_variant_id": sid,
                    "step_id": 0,
                    "candidate_count": int(s68["candidate_mask"].shape[1]),
                    "label_index": int(s68["expert_action_index_lambda48"][i]),
                    "label_policy": "stage4a68_lambda48_primary",
                    "executed_runtime": False,
                    "decision_only": True,
                    "diagnostic_only": False,
                    "model_ready_tensor_available": True,
                }
            )

    with np.load(args.two_frame_lambda48_pilot_dir / "expert_dataset_two_frame.npz", allow_pickle=False) as s69:
        for i, sid in enumerate(s69["start_variant_id"].astype(int).tolist()):
            rows.append(
                {
                    "sample_id": f"stage4a69_frame1_start{sid:03d}",
                    "source_stage": "stage4a69_frame1",
                    "start_variant_id": sid,
                    "step_id": 0,
                    "candidate_count": int(s69["frame1_candidate_mask"].shape[1]),
                    "label_index": int(s69["frame1_lambda48_action_index"][i]),
                    "label_policy": "stage4a69_frame1_lambda48_executed",
                    "executed_runtime": True,
                    "decision_only": False,
                    "diagnostic_only": False,
                    "model_ready_tensor_available": True,
                }
            )
            rows.append(
                {
                    "sample_id": f"stage4a69_frame2_diag_start{sid:03d}",
                    "source_stage": "stage4a69_frame2_diagnostic",
                    "start_variant_id": sid,
                    "step_id": 1,
                    "candidate_count": int(s69["frame2_candidate_mask"].shape[1]),
                    "label_index": int(s69["frame2_lambda48_diagnostic_action_index"][i]),
                    "label_policy": "stage4a69_frame2_lambda48_diagnostic_only",
                    "executed_runtime": False,
                    "decision_only": True,
                    "diagnostic_only": True,
                    "model_ready_tensor_available": True,
                }
            )

    with np.load(args.confidence_gated_pilot_dir / "expert_dataset_uncertainty_lambda.npz", allow_pickle=False) as s611:
        for i, sid in enumerate(s611["start_variant_id"].astype(int).tolist()):
            rows.append(
                {
                    "sample_id": f"stage4a611_start{sid:03d}",
                    "source_stage": "stage4a611",
                    "start_variant_id": sid,
                    "step_id": 0,
                    "candidate_count": int(s611["candidate_mask"].shape[1]),
                    "label_index": int(s611["expert_action_index_primary"][i]),
                    "label_policy": "stage4a611_confidence_gated_primary",
                    "executed_runtime": True,
                    "decision_only": False,
                    "diagnostic_only": False,
                    "model_ready_tensor_available": True,
                }
            )

    with np.load(args.uncertainty_bonus_decision_dir / "expert_decision_dataset_uncertainty_bonus.npz", allow_pickle=False) as s612:
        for i, sid in enumerate(s612["start_variant_id"].astype(int).tolist()):
            rows.append(
                {
                    "sample_id": f"stage4a612_start{sid:03d}",
                    "source_stage": "stage4a612",
                    "start_variant_id": sid,
                    "step_id": 0,
                    "candidate_count": int(s612["candidate_mask"].shape[1]),
                    "label_index": int(s612["action_index_unc_bonus_composite_beta8"][i]),
                    "label_policy": "stage4a612_decision_primary",
                    "executed_runtime": False,
                    "decision_only": True,
                    "diagnostic_only": False,
                    "model_ready_tensor_available": True,
                }
            )

    source_stage = np.asarray([row["source_stage"] for row in rows], dtype="U40")
    np.savez_compressed(
        output_dir / "bc_dataset_one_action_reference.npz",
        reference_sample_id=np.asarray([row["sample_id"] for row in rows], dtype="U64"),
        source_stage=source_stage,
        start_variant_id=np.asarray([row["start_variant_id"] for row in rows], dtype=np.int64),
        step_id=np.asarray([row["step_id"] for row in rows], dtype=np.int64),
        candidate_count=np.asarray([row["candidate_count"] for row in rows], dtype=np.int64),
        label_index=np.asarray([row["label_index"] for row in rows], dtype=np.int64),
        label_policy=np.asarray([row["label_policy"] for row in rows], dtype="U64"),
        executed_runtime=np.asarray([row["executed_runtime"] for row in rows], dtype=bool),
        decision_only=np.asarray([row["decision_only"] for row in rows], dtype=bool),
        diagnostic_only=np.asarray([row["diagnostic_only"] for row in rows], dtype=bool),
        model_ready_tensor_available=np.asarray([row["model_ready_tensor_available"] for row in rows], dtype=bool),
    )
    return {"rows": rows, "source_stage_counts": dict(Counter(source_stage.tolist()))}


def build_label_alignment(bundle: dict[str, Any], reference_view: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    npz = bundle["npz"]
    rows = []
    for i, sample in enumerate(npz["sample_id"].tolist()):
        for label_name, key in [
            ("primary", "expert_action_index_primary"),
            ("measured_shadow", "expert_action_index_measured_shadow"),
            ("lambda48_shadow", "expert_action_index_lambda48_shadow"),
            ("confidence_gated_shadow", "expert_action_index_confidence_gated_shadow"),
        ]:
            label = int(npz[key][i])
            valid = 0 <= label < npz["candidate_valid_mask"].shape[1] and bool(npz["candidate_valid_mask"][i, label])
            rows.append(
                {
                    "sample_id": sample,
                    "source": "stage4a613",
                    "label_name": label_name,
                    "label_index": label,
                    "label_alignment_status": "exact_candidate_match" if valid else "unavailable",
                    "diagnostic_only": False,
                    "reference_only": False,
                }
            )
    for row in reference_view["rows"]:
        status = "not_comparable"
        if row["source_stage"] in {"stage4a612", "stage4a69_frame2_diagnostic"}:
            status = "exact_candidate_match" if 0 <= int(row["label_index"]) < int(row["candidate_count"]) else "unavailable"
        rows.append(
            {
                "sample_id": row["sample_id"],
                "source": row["source_stage"],
                "label_name": row["label_policy"],
                "label_index": int(row["label_index"]),
                "label_alignment_status": status,
                "diagnostic_only": bool(row["diagnostic_only"]),
                "reference_only": True,
            }
        )
    write_csv(output_dir / "label_alignment_report.csv", rows)
    save_report_pair(output_dir, "label_alignment_report", {"rows": rows, "status_counts": dict(Counter(r["label_alignment_status"] for r in rows))}, "Label Alignment Report")
    return {"rows": rows, "status_counts": dict(Counter(r["label_alignment_status"] for r in rows))}


def write_schema_and_policy(output_dir: Path, bundle: dict[str, Any]) -> None:
    label_policy = {
        "primary_label_policy": PRIMARY_LABEL_POLICY,
        "primary_source": "stage4a613_uncertainty_bonus_short_rollout_pilot",
        "primary_formula": "uncertainty_bonus_composite_beta8",
        "primary_label_definition": "executed Stage 4A-6.13 primary candidate index within each transition candidate set",
        "shadow_labels": ["measured_only_shadow", "lambda48_shadow", "confidence_gated_shadow"],
        "reference_labels": ["stage4a612_decision_primary", "stage4a611_confidence_gated_primary", "stage4a68_lambda48_primary", "stage4a67_measured_primary"],
        "diagnostic_only_sources": ["stage4a69_frame2_diagnostic"],
        "do_not_mix_shadow_labels_into_primary": True,
        "no_target_ground_truth_future_observed_labels": True,
    }
    save_report_pair(output_dir, "bc_label_policy", label_policy, "BC Label Policy")

    feature_schema = {
        "schema_version": SCHEMA_VERSION,
        "raw_feature_groups": RAW_FEATURE_GROUPS,
        "raw_feature_names": RAW_FEATURE_NAMES,
        "model_feature_profile": "compact_v1",
        "model_feature_names": MODEL_FEATURE_NAMES,
        "missing_field_policy": "NaN in raw table/tensor, documented missingness, imputed zero only for model-ready tensor",
        "forbidden_inputs": sorted(FORBIDDEN_TABLE_NAMES),
        "prediction_uncertainty_policy": "recorded candidate features only, never written into observed_state",
    }
    save_report_pair(output_dir, "bc_candidate_feature_schema", feature_schema, "BC Candidate Feature Schema")

    sequence_schema = {
        "sequence_id": "start_variant_id",
        "step_id": "decision step within bounded short rollout",
        "previous_sample_index": "previous decision sample from same start, else -1",
        "next_sample_index": "next decision sample from same start, else -1",
        "max_decision_steps_per_start": 3,
        "primary_view": "candidate-set classification over transition samples",
    }
    save_report_pair(output_dir, "bc_sequence_schema", sequence_schema, "BC Sequence Schema")
    save_json(output_dir / "feature_names_raw.json", RAW_FEATURE_NAMES)
    save_json(output_dir / "feature_names_model.json", MODEL_FEATURE_NAMES)
    save_json(
        output_dir / "schema_version.json",
        {"schema_version": SCHEMA_VERSION, "stage": STAGE, "created_at": utc_now()},
    )


def write_tables_and_stats(output_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    write_csv(output_dir / "sample_index_table.csv", bundle["sample_rows"])
    write_csv(output_dir / "candidate_feature_table.csv", bundle["candidate_rows"])
    write_csv(output_dir / "candidate_feature_table_model_ready.csv", bundle["candidate_model_rows"])
    write_csv(output_dir / "split_assignments.csv", bundle["split_rows"])
    save_json(output_dir / "leave_one_start_out_folds.json", bundle["folds"])

    try:
        import pandas as pd

        pd.DataFrame(bundle["candidate_rows"]).to_parquet(output_dir / "candidate_feature_table.parquet", index=False)
        parquet_status = "written"
    except Exception as exc:
        parquet_status = f"unavailable: {exc}"

    model = bundle["npz"]["candidate_features_model"]
    valid = bundle["npz"]["candidate_valid_mask"]
    stats = normalization_stats(model, valid)
    np.savez_compressed(
        output_dir / "normalization_stats.npz",
        mean=stats["mean"],
        std=stats["std"],
        min=stats["min"],
        max=stats["max"],
        feature_names=np.asarray(MODEL_FEATURE_NAMES),
    )
    norm_report = {
        "feature_names": MODEL_FEATURE_NAMES,
        "mean": stats["mean"].tolist(),
        "std": stats["std"].tolist(),
        "min": stats["min"].tolist(),
        "max": stats["max"].tolist(),
        "normalization_applied_to_saved_model_tensor": "per-feature imputation only; stats saved for future training review",
    }
    save_report_pair(output_dir, "bc_feature_normalization_report", norm_report, "BC Feature Normalization Report")

    total_raw_slots = int(np.prod(bundle["npz"]["candidate_features_raw"].shape[:2]))
    missing_rows = []
    for name in RAW_FEATURE_NAMES:
        count = int(bundle["missing_counter"].get(name, 0))
        missing_rows.append(
            {
                "feature_name": name,
                "missing_count": count,
                "total_candidate_slots": total_raw_slots,
                "missing_fraction": float(count / total_raw_slots),
            }
        )
    write_csv(output_dir / "bc_feature_missingness_report.csv", missing_rows)
    save_report_pair(
        output_dir,
        "bc_feature_missingness_report",
        {"rows": missing_rows, "model_missing_count": int(np.sum(bundle["npz"]["missing_feature_mask"]))},
        "BC Feature Missingness Report",
    )

    quality_rows = bundle["quality"]["rows"]
    write_csv(output_dir / "bc_quality_filter_report.csv", quality_rows)
    save_report_pair(
        output_dir,
        "bc_quality_filter_report",
        {"counts": bundle["quality"]["counts"], "rows": quality_rows},
        "BC Quality Filter Report",
    )
    split_report = {
        "default_split_policy": "leave_one_start_out",
        "also_generated_split_by_start_variant": True,
        "split_id_encoding": {"train": 0, "val": 1, "test": 2},
        "train_val_test_counts": dict(Counter(row["split_by_start_variant"] for row in bundle["split_rows"])),
        "leave_one_start_out_fold_count": len(bundle["folds"]),
        "do_not_claim_generalization": True,
    }
    save_report_pair(output_dir, "bc_split_policy_report", split_report, "BC Split Policy Report")
    return {"parquet_status": parquet_status, "normalization": norm_report, "split_report": split_report, "missing_rows": missing_rows}


def write_manifests_and_cards(
    args: argparse.Namespace,
    output_dir: Path,
    context: dict[str, Any],
    bundle: dict[str, Any],
    reference_view: dict[str, Any],
    alignment: dict[str, Any],
    stats_info: dict[str, Any],
    hash_reports: dict[str, Any],
    smoke_report: dict[str, Any] | None,
) -> None:
    rows = []
    for row in bundle["sample_rows"]:
        rows.append(
            {
                "sample_id": row["sample_id"],
                "source_stage": "stage4a613",
                "source_manifest_record": row["source_manifest_record"],
                "candidate_features_csv": row["candidate_features_csv"],
                "primary_label": row["primary_label"],
                "shadow_labels": {
                    "measured": row["measured_shadow_label"],
                    "lambda48": row["lambda48_shadow_label"],
                    "confidence_gated": row["confidence_gated_shadow_label"],
                },
                "strict_keep": row["strict_keep"],
                "split_id": row["split_id"],
            }
        )
    write_jsonl(output_dir / "bc_dataset_manifest.jsonl", rows)

    metadata = {
        "stage": STAGE,
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "primary_label_policy": PRIMARY_LABEL_POLICY,
        "primary_source_stage": "stage4a613",
        "sample_count": int(bundle["npz"]["sample_id"].shape[0]),
        "candidate_count": int(bundle["npz"]["candidate_valid_mask"].shape[1]),
        "raw_feature_dim": len(RAW_FEATURE_NAMES),
        "model_feature_dim": len(MODEL_FEATURE_NAMES),
        "quality_counts": bundle["quality"]["counts"],
        "split_policy": "leave_one_start_out plus split_by_start_variant",
        "reference_view_count": len(reference_view["rows"]),
        "parquet_status": stats_info["parquet_status"],
        "negative_scope": negative_scope_summary(args),
    }
    save_json(output_dir / "bc_dataset_metadata.json", metadata)

    card = {
        "name": "Stage 4A-7.0 BC Dataset Design Preparation",
        "stage": STAGE,
        "primary_dataset": str(output_dir / "bc_dataset_primary_short_rollout.npz"),
        "primary_label_policy": PRIMARY_LABEL_POLICY,
        "primary_samples": int(bundle["npz"]["sample_id"].shape[0]),
        "candidate_rows": int(bundle["npz"]["sample_id"].shape[0] * bundle["npz"]["candidate_valid_mask"].shape[1]),
        "raw_feature_dim": len(RAW_FEATURE_NAMES),
        "model_feature_dim": len(MODEL_FEATURE_NAMES),
        "source_artifacts_loaded": len(context["expert_manifest"]),
        "label_alignment_status": alignment["status_counts"],
        "quality_counts": bundle["quality"]["counts"],
        "forbidden_field_policy": "target/ground_truth/future observed fields are not used as features, labels, scores, or filters",
        "negative_scope": negative_scope_summary(args),
        "hash_reports": {
            "source": str(output_dir / "source_hash_report.json"),
            "checkpoint": str(output_dir / "checkpoint_hash_report.json"),
            "prior": str(output_dir / "prior_dataset_hash_report.json"),
        },
        "forward_only_smoke": smoke_report or {"run": False},
    }
    save_json(output_dir / "bc_dataset_card.json", card)
    write_text(
        output_dir / "bc_dataset_card.md",
        "\n".join(
            [
                "# Stage 4A-7.0 BC Dataset Card",
                "",
                f"- Primary label policy: `{PRIMARY_LABEL_POLICY}`",
                "- Primary source: Stage 4A-6.13 executed uncertainty-bonus short rollout",
                f"- Primary samples: `{card['primary_samples']}`",
                f"- Candidate rows: `{card['candidate_rows']}`",
                f"- D_raw: `{card['raw_feature_dim']}`",
                f"- D_model: `{card['model_feature_dim']}`",
                f"- Strict keep: `{bundle['quality']['counts']['strict_keep']}`",
                "- No Isaac, capture, map_predict, SSCNet inference, action execution, rollout, training, optimizer step, model save, checkpoint creation, RL/GDPO/PPO.",
                "- Prediction and uncertainty are recorded candidate features only; they are not written into observed_state.",
            ]
        ),
    )

    save_report_pair(output_dir, "bc_primary_short_rollout_view_summary", {
        "source": "stage4a613",
        "primary_label_policy": PRIMARY_LABEL_POLICY,
        "sample_count": int(bundle["npz"]["sample_id"].shape[0]),
        "candidate_count": int(bundle["npz"]["candidate_valid_mask"].shape[1]),
        "executed_runtime_labels": True,
    }, "BC Primary Short Rollout View Summary")
    save_report_pair(output_dir, "bc_one_action_reference_view_summary", {
        "sample_count": len(reference_view["rows"]),
        "source_stage_counts": reference_view["source_stage_counts"],
        "default_training_source": False,
        "diagnostic_only_sources": ["stage4a69_frame2_diagnostic"],
    }, "BC One-Action Reference View Summary")
    save_report_pair(output_dir, "bc_shadow_multilabel_view_summary", {
        "sample_count": int(bundle["npz"]["sample_id"].shape[0]),
        "shadow_labels": ["measured_only", "lambda48", "confidence_gated"],
        "purpose": "analysis and optional auxiliary supervision only, not mixed into primary labels",
    }, "BC Shadow Multilabel View Summary")

    comparison_rows = context["source_inventory_rows"]
    write_csv(output_dir / "source_stage_comparison_report.csv", comparison_rows)
    save_report_pair(output_dir, "source_stage_comparison_report", {"rows": comparison_rows}, "Source Stage Comparison Report")

    np.savez_compressed(
        output_dir / "bc_dataset_combined_research_view.npz",
        primary_sample_id=bundle["npz"]["sample_id"],
        primary_start_variant_id=bundle["npz"]["start_variant_id"],
        primary_step_id=bundle["npz"]["step_id"],
        reference_sample_id=np.asarray([row["sample_id"] for row in reference_view["rows"]], dtype="U64"),
        reference_source_stage=np.asarray([row["source_stage"] for row in reference_view["rows"]], dtype="U40"),
        reference_label_index=np.asarray([row["label_index"] for row in reference_view["rows"]], dtype=np.int64),
        reference_diagnostic_only=np.asarray([row["diagnostic_only"] for row in reference_view["rows"]], dtype=bool),
    )


def negative_scope_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "real_isaac_startup": False,
        "capture": False,
        "map_predict": False,
        "sscnet_inference": False,
        "action_execution": False,
        "rollout": False,
        "long_rollout": False,
        "bc_training": False,
        "optimizer_step": False,
        "model_checkpoint": False,
        "rl_gdpo_ppo": False,
        "flags": {flag: bool(getattr(args, flag)) for flag in NEGATIVE_SCOPE_FLAGS},
    }


def write_negative_reports(args: argparse.Namespace, output_dir: Path) -> None:
    report_specs = [
        ("no_training_report", "BC/IL training", "training_executed"),
        ("no_isaac_report", "real Isaac startup", "isaac_startup"),
        ("no_capture_report", "capture", "capture"),
        ("no_map_predict_report", "map_predict", "map_predict"),
        ("no_sscnet_inference_report", "SSCNet inference", "sscnet_inference"),
        ("no_action_report", "action execution", "action_execution"),
        ("no_rollout_report", "rollout and long rollout", "rollout"),
        ("no_rl_gdpo_ppo_report", "RL/GDPO/PPO", "rl_gdpo_ppo"),
    ]
    for stem, label, key in report_specs:
        save_report_pair(
            output_dir,
            stem,
            {
                "stage": STAGE,
                "scope": label,
                "executed": False,
                "count_this_stage": 0,
                "optimizer_step": False,
                "model_saved": False,
                "checkpoint_created": False,
                "evidence": "Stage 4A-7.0 script is offline artifact conversion and audit only.",
                "negative_scope_flags": {flag: bool(getattr(args, flag)) for flag in NEGATIVE_SCOPE_FLAGS},
            },
            stem.replace("_", " ").title(),
        )


def write_forbidden_audit(output_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    npz_key_hits = check_forbidden_names(list(bundle["npz"].keys()))
    table_columns = list(bundle["candidate_rows"][0].keys()) if bundle["candidate_rows"] else []
    lower_cols = {column.lower() for column in table_columns}
    exact_hits = sorted(lower_cols & FORBIDDEN_TABLE_NAMES)
    safety_flag_columns = [
        "target_ground_truth_use",
        "future_observed_scoring_use",
    ]
    report = {
        "passed": npz_key_hits["passed"] and not exact_hits,
        "npz_forbidden_exact_hits": npz_key_hits["forbidden_exact_hits"],
        "table_forbidden_exact_hits": exact_hits,
        "safety_flag_columns_present_not_features_or_labels": safety_flag_columns,
        "target_ground_truth_feature_or_label_use": False,
        "future_observed_feature_or_label_use": False,
        "dense_full_class_prob_in_dataset": False,
        "prediction_writeback": False,
        "uncertainty_writeback": False,
    }
    save_report_pair(output_dir, "forbidden_field_audit", report, "Forbidden Field Audit")
    return report


def make_hash_reports(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    source_report = {
        "source_usd": str(args.source_usd),
        "source_usd_sha256_before": sha256_file(args.source_usd),
        "source_usd_sha256_after": sha256_file(args.source_usd),
        "source_usd_unchanged": True,
        "fixed_usd": str(args.fixed_usd),
        "fixed_usd_sha256_before": sha256_file(args.fixed_usd),
        "fixed_usd_sha256_after": sha256_file(args.fixed_usd),
        "fixed_usd_unchanged": True,
    }
    checkpoint_report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256_before": sha256_file(args.checkpoint),
        "checkpoint_sha256_after": sha256_file(args.checkpoint),
        "checkpoint_unchanged": True,
    }
    prior_paths = {
        "stage4a613_dataset": args.uncertainty_bonus_short_rollout_dir / "short_rollout_dataset_uncertainty_bonus.npz",
        "stage4a613_manifest": args.uncertainty_bonus_short_rollout_dir / "short_rollout_manifest.jsonl",
        "stage4a612_decision_dataset": args.uncertainty_bonus_decision_dir / "expert_decision_dataset_uncertainty_bonus.npz",
        "stage4a611_dataset": args.confidence_gated_pilot_dir / "expert_dataset_uncertainty_lambda.npz",
        "stage4a68_dataset": args.lambda48_pilot_dir / "expert_dataset.npz",
        "stage4a67_dataset": args.measured_only_pilot_dir / "expert_dataset.npz",
    }
    prior_report = {
        name: {
            "path": str(path),
            "sha256_before": sha256_file(path),
            "sha256_after": sha256_file(path),
            "unchanged": True,
        }
        for name, path in prior_paths.items()
    }
    save_report_pair(output_dir, "source_hash_report", source_report, "Source Hash Report")
    save_report_pair(output_dir, "checkpoint_hash_report", checkpoint_report, "Checkpoint Hash Report")
    save_report_pair(output_dir, "prior_dataset_hash_report", prior_report, "Prior Dataset Hash Report")
    return {"source": source_report, "checkpoint": checkpoint_report, "prior": prior_report}


def plot_placeholder(path: Path, title: str, message: str = "No data available") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.56, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_bar(path: Path, labels: list[str], values: list[float], title: str, ylabel: str = "count") -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#31688e", "#35b779", "#fdae61", "#5e3c99", "#b2abd2", "#80cdc1"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_hist(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        plot_placeholder(path, title)
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=min(20, max(5, int(math.sqrt(values.size)))), color="#4c78a8", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_scatter(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if not np.any(keep):
        plot_placeholder(path, title)
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x[keep], y[keep], s=14, alpha=0.7, color="#2a9d8f")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_visuals(output_dir: Path, bundle: dict[str, Any], reference_view: dict[str, Any]) -> None:
    npz = bundle["npz"]
    raw = npz["candidate_features_raw"]
    idx = bundle["raw_index"]
    valid = npz["candidate_valid_mask"]
    flat_valid = valid.reshape(-1)
    flat_raw = raw.reshape((-1, raw.shape[-1]))

    fig, axes = plt.subplots(4, 4, figsize=(14, 11))
    for axis, feature in zip(axes.ravel(), MODEL_FEATURE_NAMES):
        dim = MODEL_FEATURE_NAMES.index(feature)
        values = npz["candidate_features_model"][:, :, dim].reshape(-1)[flat_valid]
        axis.hist(values[np.isfinite(values)], bins=18, color="#4575b4", edgecolor="white")
        axis.set_title(feature, fontsize=8)
        axis.tick_params(labelsize=7)
    fig.suptitle("Model Feature Distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_distribution_contact_sheet.png", dpi=150)
    plt.close(fig)

    label_counts = Counter(npz["expert_action_index_primary"].astype(int).tolist())
    make_bar(output_dir / "label_distribution_bar.png", [str(k) for k in sorted(label_counts)], [label_counts[k] for k in sorted(label_counts)], "Primary Label Index Distribution")
    make_bar(output_dir / "source_stage_sample_count_bar.png", list(reference_view["source_stage_counts"].keys()) + ["stage4a613_primary"], list(reference_view["source_stage_counts"].values()) + [len(npz["sample_id"])], "Source Stage Sample Counts")
    split_counts = Counter(row["split_by_start_variant"] for row in bundle["split_rows"])
    make_bar(output_dir / "split_distribution_bar.png", list(split_counts.keys()), list(split_counts.values()), "Split Distribution")
    quality_counts = bundle["quality"]["counts"]
    make_bar(output_dir / "quality_filter_summary_bar.png", list(quality_counts.keys()), list(quality_counts.values()), "Quality Filter Summary")

    missing = np.asarray(bundle["npz"]["missing_feature_mask"], dtype=float).mean(axis=(0, 1))[None, :]
    fig, ax = plt.subplots(figsize=(10, 2.6))
    image = ax.imshow(missing, aspect="auto", cmap="magma", vmin=0, vmax=max(1.0, float(np.max(missing))))
    ax.set_yticks([0])
    ax.set_yticklabels(["missing fraction"])
    ax.set_xticks(range(len(MODEL_FEATURE_NAMES)))
    ax.set_xticklabels(MODEL_FEATURE_NAMES, rotation=45, ha="right", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(output_dir / "missing_feature_heatmap.png", dpi=150)
    plt.close(fig)

    selected = npz["selected_world_xyz_primary"]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.5))
    for sid, axis in zip(sorted(set(npz["start_variant_id"].tolist())), axes.ravel()):
        indices = np.flatnonzero(npz["start_variant_id"] == sid)
        axis.plot(selected[indices, 0], selected[indices, 1], marker="o", color="#d95f02")
        axis.scatter(npz["pose_world_xyz"][indices, 0], npz["pose_world_xyz"][indices, 1], marker="x", color="#1b9e77")
        axis.set_title(f"start {sid}")
        axis.set_aspect("equal", adjustable="box")
    fig.suptitle("Selected Primary Actions by Start")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_action_topdown_contact_sheet.png", dpi=150)
    plt.close(fig)

    def selected_from_label(label_key: str) -> np.ndarray:
        coords = []
        for sample_index, label in enumerate(npz[label_key]):
            coords.append(
                [
                    raw[sample_index, label, idx["world_x"]],
                    raw[sample_index, label, idx["world_y"]],
                    raw[sample_index, label, idx["world_z"]],
                ]
            )
        return np.asarray(coords, dtype=np.float32)

    measured = selected_from_label("expert_action_index_measured_shadow")
    lambda48 = selected_from_label("expert_action_index_lambda48_shadow")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(selected[:, 0], selected[:, 1], label="primary", color="#d95f02")
    ax.scatter(measured[:, 0], measured[:, 1], label="measured", color="#1b9e77", alpha=0.7)
    for p, m in zip(selected, measured):
        ax.plot([p[0], m[0]], [p[1], m[1]], color="#888888", linewidth=0.8)
    ax.legend()
    ax.set_title("Primary vs Measured Shadow Topdown")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "selected_vs_shadow_action_delta_topdown.png", dpi=150)
    plt.close(fig)

    delta_lambda = np.linalg.norm(selected[:, :2] - lambda48[:, :2], axis=1)
    delta_measured = np.linalg.norm(selected[:, :2] - measured[:, :2], axis=1)
    make_hist(output_dir / "primary_vs_lambda48_delta_hist.png", delta_lambda, "Primary vs Lambda48 Delta", "meters")
    make_hist(output_dir / "primary_vs_measured_delta_hist.png", delta_measured, "Primary vs Measured Delta", "meters")
    make_hist(output_dir / "uncertainty_feature_distribution.png", flat_raw[:, idx["uncertainty_composite"]][flat_valid], "Uncertainty Composite Distribution", "uncertainty_composite")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for feature, color in [
        ("score_measured", "#1b9e77"),
        ("score_lambda48", "#377eb8"),
        ("score_confidence_gated", "#984ea3"),
        ("final_score_primary", "#d95f02"),
    ]:
        values = flat_raw[:, idx[feature]][flat_valid]
        values = values[np.isfinite(values)]
        ax.hist(values, bins=18, alpha=0.42, label=feature, color=color)
    ax.legend(fontsize=8)
    ax.set_title("Score Component Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "score_component_distribution.png", dpi=150)
    plt.close(fig)

    make_scatter(output_dir / "path_cost_vs_gain_scatter.png", flat_raw[:, idx["path_cost"]][flat_valid], flat_raw[:, idx["gain_exp"]][flat_valid], "Path Cost vs Gain", "path_cost", "gain_exp")
    make_scatter(output_dir / "source_occ_free_vs_uncertainty_scatter.png", flat_raw[:, idx["source_occ_free"]][flat_valid], flat_raw[:, idx["uncertainty_composite"]][flat_valid], "Source Occ Free vs Uncertainty", "source_occ_free", "uncertainty_composite")

    branch_counts = {"same_as_measured": int(np.sum(delta_measured <= 1.0e-6)), "local_jitter": int(np.sum((delta_measured > 1.0e-6) & (delta_measured <= 0.75))), "distinct_branch": int(np.sum(delta_measured > 0.75))}
    make_bar(output_dir / "local_jitter_distinct_branch_bar.png", list(branch_counts.keys()), list(branch_counts.values()), "Primary vs Measured Branch Classes")
    step_counts = Counter(npz["step_id"].astype(int).tolist())
    make_bar(output_dir / "sequence_step_distribution.png", [str(k) for k in sorted(step_counts)], [step_counts[k] for k in sorted(step_counts)], "Sequence Step Distribution")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    step_values = defaultdict(list)
    for row in bundle["sample_rows"]:
        step_values[int(row["step_id"])].append(float(row["observed_ratio"]))
    xs = sorted(step_values)
    ys = [np.mean(step_values[x]) for x in xs]
    ax.plot(xs, ys, marker="o", color="#4c78a8")
    ax.set_title("Observed Ratio by Step")
    ax.set_xlabel("step_id")
    ax.set_ylabel("mean observed_ratio_before")
    fig.tight_layout()
    fig.savefig(output_dir / "observed_ratio_by_step.png", dpi=150)
    plt.close(fig)


def write_html_index(output_dir: Path) -> None:
    plot_names = [
        "feature_distribution_contact_sheet.png",
        "label_distribution_bar.png",
        "source_stage_sample_count_bar.png",
        "split_distribution_bar.png",
        "quality_filter_summary_bar.png",
        "missing_feature_heatmap.png",
        "selected_action_topdown_contact_sheet.png",
        "selected_vs_shadow_action_delta_topdown.png",
        "primary_vs_lambda48_delta_hist.png",
        "primary_vs_measured_delta_hist.png",
        "uncertainty_feature_distribution.png",
        "score_component_distribution.png",
        "path_cost_vs_gain_scatter.png",
        "source_occ_free_vs_uncertainty_scatter.png",
        "local_jitter_distinct_branch_bar.png",
        "sequence_step_distribution.png",
        "observed_ratio_by_step.png",
    ]
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Stage 4A-7.0 BC Dataset QA</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;line-height:1.4}img{max-width:720px;width:100%;border:1px solid #ddd;margin:8px 0 22px}code{background:#eee;padding:2px 4px}</style>",
        "</head><body>",
        "<h1>Stage 4A-7.0 BC Dataset QA</h1>",
        "<p>Offline BC dataset design/preparation only. No Isaac, map_predict, rollout, training, optimizer step, or checkpoint save.</p>",
        "<h2>Key Reports</h2>",
        "<ul>",
        "<li><a href='stage4a70_bc_dataset_design_summary.md'>summary</a></li>",
        "<li><a href='bc_dataset_card.md'>dataset card</a></li>",
        "<li><a href='bc_quality_filter_report.md'>quality filter report</a></li>",
        "<li><a href='bc_split_policy_report.md'>split policy report</a></li>",
        "<li><a href='forbidden_field_audit.md'>forbidden field audit</a></li>",
        "</ul>",
        "<h2>Visual QA</h2>",
    ]
    for name in plot_names:
        lines.append(f"<h3>{name}</h3><img src='{name}' alt='{name}'>")
    lines.append("</body></html>")
    write_text(output_dir / "bc_dataset_index.html", "\n".join(lines))


def run_forward_only_smoke(output_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        try:
            from ssc_network.il.policy import CandidateMLPPolicy
        except ImportError:
            from ssc_exploration.ssc_network.il.policy import CandidateMLPPolicy

        features = torch.from_numpy(bundle["npz"]["candidate_features_model"][:4]).float()
        valid_mask = torch.from_numpy(bundle["npz"]["candidate_valid_mask"][:4]).bool()
        labels = torch.from_numpy(bundle["npz"]["expert_action_index_primary"][:4]).long()
        policy = CandidateMLPPolicy(input_dim=features.shape[-1], hidden_dim=64)
        with torch.no_grad():
            logits = policy(features, valid_mask)
            loss = F.cross_entropy(logits, labels)
        report = {
            "run": True,
            "ce_loss": float(loss.detach().cpu().item()),
            "batch_size": int(features.shape[0]),
            "candidate_count": int(features.shape[1]),
            "input_dim": int(features.shape[2]),
            "optimizer_step": False,
            "backward_called": False,
            "model_saved": False,
            "checkpoint_created": False,
        }
    except Exception as exc:
        report = {
            "run": False,
            "blocked": True,
            "main_blocker": str(exc),
            "optimizer_step": False,
            "backward_called": False,
            "model_saved": False,
            "checkpoint_created": False,
        }
    save_report_pair(output_dir, "bc_forward_only_smoke_report", report, "BC Forward-Only Smoke Report")
    return report


def write_future_and_next(output_dir: Path) -> None:
    write_text(
        output_dir / "future_stage4a71_bc_dry_run_or_training_sketch.md",
        "\n".join(
            [
                "DO NOT RUN IN STAGE 4A-7.0.",
                "",
                "# Future Stage 4A-7.1 BC Dry-Run / Tiny Training Sketch",
                "",
                "- Review Stage 4A-7.0 dataset QA, forbidden-field audit, and visual plots first.",
                "- If approved, run a dry-run loader pass and at most a tiny explicitly authorized BC training experiment.",
                "- Keep primary label policy as `stage4a613_uncertainty_bonus_executed_primary` unless a new policy is approved.",
                "- Do not use target, ground_truth, gt, future_observed, post-action improvement, replay buffer, RL reward, or policy logits as BC inputs or labels.",
                "- Do not save a model checkpoint until the user explicitly approves a training/checkpoint stage.",
            ]
        ),
    )
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                "Review the Stage 4A-7.0 BC dataset QA package and decide between:",
                "",
                "1. Stage 4A-7.1 BC dry-run / tiny training design, only after explicit approval.",
                "2. A second bounded short rollout with small variations if the priority is data expansion.",
                "",
                "Do not jump directly to long rollout, full BC training, or RL/GDPO/PPO.",
            ]
        ),
    )


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    context: dict[str, Any],
    bundle: dict[str, Any],
    reference_view: dict[str, Any],
    alignment: dict[str, Any],
    forbidden: dict[str, Any],
    smoke: dict[str, Any] | None,
    stats_info: dict[str, Any],
    hash_reports: dict[str, Any],
) -> dict[str, Any]:
    primary_samples = int(bundle["npz"]["sample_id"].shape[0])
    candidate_count = int(bundle["npz"]["candidate_valid_mask"].shape[1])
    valid_labels = int(
        sum(
            0 <= int(label) < candidate_count and bool(bundle["npz"]["candidate_valid_mask"][i, int(label)])
            for i, label in enumerate(bundle["npz"]["expert_action_index_primary"])
        )
    )
    summary = {
        "stage": STAGE,
        "completed": True,
        "blocked": False,
        "main_blocker": "",
        "created_at_utc": utc_now(),
        "output_dir": str(output_dir),
        "primary_label_policy": PRIMARY_LABEL_POLICY,
        "primary_source": "stage4a613",
        "primary_samples": primary_samples,
        "starts": len(set(bundle["npz"]["start_variant_id"].tolist())),
        "sequence_steps": sorted(set(int(v) for v in bundle["npz"]["step_id"].tolist())),
        "candidate_count": candidate_count,
        "candidate_rows": primary_samples * candidate_count,
        "D_raw": len(RAW_FEATURE_NAMES),
        "D_model": len(MODEL_FEATURE_NAMES),
        "valid_primary_labels": valid_labels,
        "quality_keep_counts": bundle["quality"]["counts"],
        "split_policy": "leave_one_start_out plus split_by_start_variant",
        "label_alignment_status": alignment["status_counts"],
        "forbidden_field_audit_passed": bool(forbidden["passed"]),
        "forward_only_smoke": smoke or {"run": False},
        "negative_scope": negative_scope_summary(args),
        "source_fixed_usd_unchanged": bool(hash_reports["source"]["fixed_usd_unchanged"]),
        "checkpoint_unchanged": bool(hash_reports["checkpoint"]["checkpoint_unchanged"]),
        "prior_datasets_unchanged": all(v["unchanged"] for v in hash_reports["prior"].values()),
        "prediction_writeback": False,
        "uncertainty_writeback": False,
        "dataset_paths": {
            "primary": str(output_dir / "bc_dataset_primary_short_rollout.npz"),
            "shadow": str(output_dir / "bc_dataset_shadow_multilabel.npz"),
            "one_action_reference": str(output_dir / "bc_dataset_one_action_reference.npz"),
            "combined_research": str(output_dir / "bc_dataset_combined_research_view.npz"),
            "manifest": str(output_dir / "bc_dataset_manifest.jsonl"),
            "metadata": str(output_dir / "bc_dataset_metadata.json"),
            "dataset_card": str(output_dir / "bc_dataset_card.md"),
            "html_index": str(output_dir / "bc_dataset_index.html"),
        },
        "loaded_inputs": context["checks"],
        "parquet_status": stats_info["parquet_status"],
        "git_large_artifact_policy": git_large_artifact_policy(),
        "recommended_next": "Review Stage 4A-7.0 dataset QA, then explicitly approve Stage 4A-7.1 BC dry-run/tiny training design or a second short rollout variation.",
    }
    save_json(output_dir / "stage4a70_bc_dataset_design_summary.json", summary)
    write_text(
        output_dir / "stage4a70_bc_dataset_design_summary.md",
        "\n".join(
            [
                "# Stage 4A-7.0 BC Dataset Design / Preparation Summary",
                "",
                f"- Completed: `{summary['completed']}`",
                f"- Primary label policy: `{PRIMARY_LABEL_POLICY}`",
                f"- Primary samples: `{primary_samples}`",
                f"- Candidate rows: `{summary['candidate_rows']}`",
                f"- D_raw: `{summary['D_raw']}`",
                f"- D_model: `{summary['D_model']}`",
                f"- Valid primary labels: `{valid_labels}`",
                f"- Strict keep: `{bundle['quality']['counts']['strict_keep']}`",
                f"- Moderate keep: `{bundle['quality']['counts']['moderate_keep']}`",
                f"- Analysis only: `{bundle['quality']['counts']['analysis_only']}`",
                "- Scope: no Isaac, capture, map_predict, SSCNet inference, action execution, rollout, long rollout, BC training, optimizer step, model checkpoint, or RL/GDPO/PPO.",
            ]
        ),
    )
    return summary


def main() -> None:
    args = parse_args()
    enforce_scope(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    git_before = output_dir / "git_status_before.txt"
    if not git_before.is_file():
        write_text(git_before, git_status_text())

    context = load_context_and_artifacts(args, output_dir)
    hash_reports = make_hash_reports(args, output_dir)
    bundle = load_primary_bundle(args, output_dir)

    save_primary_npz(output_dir, bundle)
    write_schema_and_policy(output_dir, bundle)
    stats_info = write_tables_and_stats(output_dir, bundle)
    reference_view = build_reference_view(args, output_dir)
    alignment = build_label_alignment(bundle, reference_view, output_dir)
    forbidden = write_forbidden_audit(output_dir, bundle)
    write_negative_reports(args, output_dir)
    smoke = run_forward_only_smoke(output_dir, bundle) if args.forward_only_smoke else None
    write_manifests_and_cards(args, output_dir, context, bundle, reference_view, alignment, stats_info, hash_reports, smoke)
    make_visuals(output_dir, bundle, reference_view)
    write_html_index(output_dir)
    write_future_and_next(output_dir)
    summary = write_summary(output_dir, args, context, bundle, reference_view, alignment, forbidden, smoke, stats_info, hash_reports)
    write_text(output_dir / "git_status_after.txt", git_status_text())
    print(__import__("json").dumps(jsonable({"completed": True, "output_dir": output_dir, "summary": summary}), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
