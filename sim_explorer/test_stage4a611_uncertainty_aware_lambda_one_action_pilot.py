#!/usr/bin/env python3
"""Validate Stage 4A-6.11 uncertainty-aware lambda one-action pilot outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")

FORBIDDEN_KEYS = {
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

REQUIRED_TOP_LEVEL = [
    "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json",
    "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.md",
    "expert_dataset_uncertainty_lambda.npz",
    "expert_dataset_manifest.jsonl",
    "expert_dataset_metadata.json",
    "per_sample_summary.csv",
    "per_sample_summary.json",
    "per_sample_summary.md",
    "primary_confidence_gated_decisions.jsonl",
    "primary_confidence_gated_decisions.csv",
    "lambda48_baseline_shadow_decisions.jsonl",
    "lambda48_baseline_shadow_decisions.csv",
    "measured_shadow_decisions.jsonl",
    "measured_shadow_decisions.csv",
    "uncertainty_bonus_shadow_decisions.jsonl",
    "uncertainty_bonus_shadow_decisions.csv",
    "uncertainty_penalty_shadow_decisions.jsonl",
    "uncertainty_penalty_shadow_decisions.csv",
    "confidence_margin_gated_shadow_decisions.jsonl",
    "confidence_margin_gated_shadow_decisions.csv",
    "entropy_penalty_shadow_decisions.jsonl",
    "entropy_penalty_shadow_decisions.csv",
    "uncertainty_candidate_features.csv",
    "uncertainty_candidate_features.json",
    "uncertainty_candidate_features.md",
    "formula_comparison_table.csv",
    "formula_comparison_table.json",
    "formula_comparison_table.md",
    "primary_vs_lambda48_comparison.csv",
    "primary_vs_lambda48_comparison.json",
    "primary_vs_lambda48_comparison.md",
    "primary_vs_measured_comparison.csv",
    "primary_vs_measured_comparison.json",
    "primary_vs_measured_comparison.md",
    "stage4a611_vs_stage4a68_comparison.csv",
    "stage4a611_vs_stage4a68_comparison.json",
    "stage4a611_vs_stage4a68_comparison.md",
    "stage4a611_vs_stage4a69_frame1_comparison.csv",
    "stage4a611_vs_stage4a69_frame1_comparison.json",
    "stage4a611_vs_stage4a69_frame1_comparison.md",
    "uncertainty_safety_audit.json",
    "uncertainty_safety_audit.md",
    "prediction_safety_audit.json",
    "prediction_safety_audit.md",
    "dataset_integrity_report.json",
    "dataset_integrity_report.md",
    "expert_data_quality_audit.json",
    "expert_data_quality_audit.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_second_action_report.json",
    "no_second_action_report.md",
    "no_third_frame_report.json",
    "no_third_frame_report.md",
    "no_training_rl_bc_report.json",
    "no_training_rl_bc_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "checkpoint_hash_report.json",
    "checkpoint_hash_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "expert_uncertainty_lambda_index.html",
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

REQUIRED_SAMPLE_FILES = [
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def jsonl_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def assert_dense_npz(path: Path) -> None:
    require(path.is_file(), f"dense artifact missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in (
            "pred_class_uint8",
            "confidence_float16",
            "entropy_norm_float16",
            "margin_float16",
            "occupied_prob_float16",
            "free_prob_float16",
            "valid_mask_bool",
            "predicted_unmeasured_mask_bool",
        ):
            require(key in data.files, f"{path} missing {key}")
        conf = np.asarray(data["confidence_float16"], dtype=np.float32)
        ent = np.asarray(data["entropy_norm_float16"], dtype=np.float32)
        margin = np.asarray(data["margin_float16"], dtype=np.float32)
        valid = np.asarray(data["valid_mask_bool"], dtype=bool)
        require(conf.shape == ent.shape == margin.shape == valid.shape, "dense shape mismatch")
        require(np.all(np.isfinite(conf)), "confidence has NaN/Inf")
        require(np.all(np.isfinite(ent)), "entropy has NaN/Inf")
        require(np.all(np.isfinite(margin)), "margin has NaN/Inf")
        require(0.0 <= float(np.min(conf)) <= float(np.max(conf)) <= 1.0, "confidence outside [0,1]")
        require(0.0 <= float(np.min(ent)) <= float(np.max(ent)) <= 1.0005, "entropy outside [0,1]")
        require(0.0 <= float(np.min(margin)) <= float(np.max(margin)) <= 1.0, "margin outside [0,1]")
        require(int(np.count_nonzero(valid)) > 0, "dense valid mask empty")


def git_policy() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    staged_forbidden = []
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/")
    forbidden_suffixes = (".usd", ".npy", ".npz", ".png", ".mp4")
    for line in result.stdout.splitlines():
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if status.strip() and status[0] != "?" and (path.startswith(forbidden_prefixes) or path.endswith(forbidden_suffixes)):
            staged_forbidden.append(line)
    return {"git_status_short": result.stdout, "staged_forbidden_large_artifacts": staged_forbidden}


def validate_dataset(path: Path, expected: int) -> dict[str, Any]:
    require(path.is_file(), "expert_dataset_uncertainty_lambda.npz missing")
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        required = {
            "start_variant_id",
            "pose",
            "observed_state_reference",
            "candidate_features",
            "candidate_mask",
            "valid_mask",
            "expert_action_index_primary",
            "expert_action_index_lambda48",
            "expert_action_index_measured",
            "expert_action_index_uncertainty_bonus",
            "expert_action_index_uncertainty_penalty",
            "expert_action_index_confidence_margin_gated",
            "expert_action_index_entropy_penalty",
            "score_primary_confidence_gated",
            "score_lambda48_baseline",
            "score_measured_shadow",
            "score_uncertainty_bonus",
            "score_uncertainty_penalty",
            "score_confidence_margin_gated",
            "score_entropy_penalty",
            "gain_exp",
            "source_occ_free",
            "path_cost",
            "confidence_mean",
            "confidence_min",
            "entropy_mean",
            "entropy_max",
            "margin_mean",
            "margin_min",
            "uncertain_fraction",
            "uncertain_voxel_count",
            "low_conf_count_0p7",
            "high_entropy_count_0p7",
            "source_occ_free_confidence_weighted",
            "source_occ_free_entropy_weighted",
            "selected_world_xyz_primary",
            "selected_yaw_primary",
            "selected_world_xyz_lambda48",
            "selected_yaw_lambda48",
            "selected_world_xyz_measured",
            "selected_yaw_measured",
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
        }
        missing = sorted(required - keys)
        forbidden = sorted(keys & FORBIDDEN_KEYS)
        require(not missing, f"dataset missing required keys: {missing}")
        require(not forbidden, f"dataset contains forbidden keys: {forbidden}")
        require(int(data["start_variant_id"].shape[0]) == expected, "dataset start count mismatch")
        require(int(data["candidate_features"].shape[1]) == 64, "candidate feature count mismatch")
        require(np.all(data["expert_action_index_primary"] >= 0), "primary action index missing")
        for key in [
            "confidence_mean",
            "entropy_mean",
            "margin_mean",
            "score_primary_confidence_gated",
            "score_lambda48_baseline",
            "score_measured_shadow",
        ]:
            arr = np.asarray(data[key], dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            require(np.all(np.isfinite(finite)), f"{key} finite check failed")
        require(float(np.nanmin(data["confidence_mean"])) >= 0.0 and float(np.nanmax(data["confidence_mean"])) <= 1.0, "dataset confidence outside [0,1]")
        require(float(np.nanmin(data["entropy_mean"])) >= 0.0 and float(np.nanmax(data["entropy_mean"])) <= 1.0005, "dataset entropy outside [0,1]")
        require(float(np.nanmin(data["margin_mean"])) >= 0.0 and float(np.nanmax(data["margin_mean"])) <= 1.0, "dataset margin outside [0,1]")
        return {"dataset_keys": sorted(keys), "forbidden": forbidden}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--camera_pose_fix_dir", type=Path, required=True)
    parser.add_argument("--measured_only_pilot_dir", type=Path, required=True)
    parser.add_argument("--lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--two_frame_lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--dense_uncertainty_dir", type=Path, required=True)
    parser.add_argument("--expected_start_count", type=int, default=10)
    parser.add_argument("--expected_frame_count", type=int, default=10)
    parser.add_argument("--expected_map_predict_calls", type=int, default=10)
    parser.add_argument("--expect_exactly_one_action_per_start", action="store_true")
    parser.add_argument("--expect_no_second_action", action="store_true")
    parser.add_argument("--expect_no_third_frame", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_long_rollout", action="store_true")
    parser.add_argument("--expect_map_predict", action="store_true")
    parser.add_argument("--expect_uncertainty_features", action="store_true")
    parser.add_argument("--expect_confidence_gated_primary", action="store_true")
    parser.add_argument("--expect_shadow_formulas", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_quality_viz", action="store_true")
    parser.add_argument("--expect_stage4a67_comparison", action="store_true")
    parser.add_argument("--expect_stage4a68_comparison", action="store_true")
    parser.add_argument("--expect_stage4a69_frame1_comparison", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    require(out.is_dir(), f"output dir missing: {out}")

    for name in REQUIRED_TOP_LEVEL:
        require((out / name).is_file(), f"missing top-level output {name}")

    summary = load_json(out / "stage4a611_uncertainty_aware_lambda_one_action_pilot_summary.json")
    integrity = load_json(out / "dataset_integrity_report.json")
    pred_safety = load_json(out / "prediction_safety_audit.json")
    unc_safety = load_json(out / "uncertainty_safety_audit.json")
    quality = load_json(out / "expert_data_quality_audit.json")
    source = load_json(out / "source_hash_report.json")
    checkpoint = load_json(out / "checkpoint_hash_report.json")

    require(summary["completed"] is True, "summary completed false")
    require(int(summary["start_count"]) == args.expected_start_count, "start_count mismatch")
    require(int(summary["frame_count"]) == args.expected_frame_count, "frame_count mismatch")
    require(int(summary["capture_count"]) == args.expected_start_count, "capture_count mismatch")
    require(int(summary["map_predict_calls"]) == args.expected_map_predict_calls, "map_predict_calls mismatch")
    require(int(summary["dense_uncertainty_artifacts"]) == args.expected_start_count, "dense uncertainty count mismatch")
    require(int(summary["executed_action_count"]) == args.expected_start_count, "executed action count mismatch")
    if args.expect_exactly_one_action_per_start:
        require(summary["exactly_one_action_per_start"] is True, "exactly_one_action_per_start false")
    if args.expect_no_second_action:
        require(int(summary["second_action_count"]) == 0, "second action count nonzero")
    if args.expect_no_third_frame:
        require(int(summary["third_frame_count"]) == 0, "third frame count nonzero")
    if args.expect_no_rollout:
        require(summary["continuous_rollout_executed"] is False, "rollout executed")
    if args.expect_no_long_rollout:
        require(summary["long_rollout_executed"] is False, "long rollout executed")
    if args.expect_map_predict:
        require(summary["sscnet_inference_called"] is True, "SSCNet flag false")
        require(summary["predictor_loaded_once"] is True, "predictor_loaded_once false")
    if args.expect_confidence_gated_primary:
        require(summary["primary_formula"] == "confidence_gated_lambda48_v1", "primary formula mismatch")
    if args.expect_no_training:
        require(summary["training"] is False, "training true")
        require(summary["behavior_cloning_training_run"] is False, "BC true")
        require(summary["imitation_learning_training_run"] is False, "IL true")
    if args.expect_no_rl_gdpo:
        require(summary["rl_training_run"] is False, "RL true")
        require(summary["gdpo_training_run"] is False, "GDPO true")
        require(summary["ppo_training_run"] is False, "PPO true")

    require(integrity["passed"] is True, "dataset integrity failed")
    require(pred_safety["passed"] is True, "prediction safety failed")
    require(unc_safety["passed"] is True, "uncertainty safety failed")
    require(quality["passed"] is True, "expert quality failed")
    for key in [
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
    ]:
        require(summary[key] is False, f"{key} should be false")
    require(summary["observed_state_hash_unchanged"] is True, "observed_state hash changed")
    require(summary["checkpoint_unchanged"] is True, "checkpoint changed")
    require(checkpoint["checkpoint_unchanged"] is True, "checkpoint report changed")
    require(source["source_usd_unchanged"] is True, "source USD changed")
    require(source["fixed_usd_unchanged"] is True, "fixed USD changed")
    require(source["source_observed_state_unchanged"] is True, "source observed state changed")
    require(source["stage4a68_dataset_sha256_before"] == source["stage4a68_dataset_sha256_after"], "6.8 dataset modified")
    require(source["stage4a69_dataset_sha256_before"] == source["stage4a69_dataset_sha256_after"], "6.9 dataset modified")

    dataset_info = validate_dataset(out / "expert_dataset_uncertainty_lambda.npz", args.expected_start_count)

    samples_dir = out / "samples"
    sample_dirs = sorted(samples_dir.glob("start_*"))
    require(len(sample_dirs) == args.expected_start_count, "sample dir count mismatch")
    for sample_dir in sample_dirs:
        for name in REQUIRED_SAMPLE_FILES:
            require((sample_dir / name).is_file(), f"missing sample file {sample_dir / name}")
        assert_dense_npz(sample_dir / "dense_prediction_uncertainty.npz")
        sample_candidate_rows = csv_row_count(sample_dir / "candidate_uncertainty_features.csv")
        require(16 <= sample_candidate_rows <= 64, f"candidate rows mismatch in {sample_dir}: {sample_candidate_rows}")
        executed = load_json(sample_dir / "executed_action.json")
        require(executed["logical_executed_action_count_contribution"] == 1, f"executed action record bad in {sample_dir}")
        require(executed["second_action_executed"] is False, f"second action true in {sample_dir}")
        require(executed["third_frame_executed"] is False, f"third frame true in {sample_dir}")

    if args.expect_uncertainty_features:
        root_candidate_rows = csv_row_count(out / "uncertainty_candidate_features.csv")
        require(root_candidate_rows == int(unc_safety["candidate_uncertainty_rows"]), "candidate uncertainty row count mismatch")
        require(root_candidate_rows >= args.expected_start_count * 16, "candidate uncertainty row count below top_n coverage")
    if args.expect_shadow_formulas:
        for name in [
            "lambda48_baseline_shadow_decisions.jsonl",
            "measured_shadow_decisions.jsonl",
            "uncertainty_bonus_shadow_decisions.jsonl",
            "uncertainty_penalty_shadow_decisions.jsonl",
            "confidence_margin_gated_shadow_decisions.jsonl",
            "entropy_penalty_shadow_decisions.jsonl",
        ]:
            require(jsonl_row_count(out / name) == args.expected_start_count, f"{name} row count mismatch")
    if args.expect_quality_viz:
        require((out / "expert_uncertainty_lambda_index.html").is_file(), "HTML missing")
        require((out / "expert_uncertainty_lambda_flythrough.mp4").is_file() or any((out / "expert_uncertainty_lambda_flythrough_frames").glob("frame_*.png")), "MP4/fallback missing")
    if args.expect_stage4a67_comparison:
        require((out / "primary_vs_measured_comparison.json").is_file(), "measured comparison missing")
    if args.expect_stage4a68_comparison:
        require((out / "stage4a611_vs_stage4a68_comparison.json").is_file(), "6.8 comparison missing")
    if args.expect_stage4a69_frame1_comparison:
        require((out / "stage4a611_vs_stage4a69_frame1_comparison.json").is_file(), "6.9 frame1 comparison missing")

    policy = git_policy()
    require(not policy["staged_forbidden_large_artifacts"], f"large artifact policy violation: {policy['staged_forbidden_large_artifacts']}")

    result = {
        "all_passed": True,
        "output_dir": str(out),
        "start_count": summary["start_count"],
        "map_predict_calls": summary["map_predict_calls"],
        "dense_uncertainty_artifacts": summary["dense_uncertainty_artifacts"],
        "candidate_uncertainty_rows": unc_safety["candidate_uncertainty_rows"],
        "dataset_key_count": len(dataset_info["dataset_keys"]),
        "git_policy_forbidden_status_rows": policy["staged_forbidden_large_artifacts"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
