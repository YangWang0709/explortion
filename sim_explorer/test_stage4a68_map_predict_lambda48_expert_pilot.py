#!/usr/bin/env python3
"""Validate Stage 4A-6.8 map_predict lambda48 expert pilot outputs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
FORBIDDEN_DATASET_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "future_observed",
    "class_prob",
    "policy_logits",
    "rl_reward",
    "replay_buffer",
    "training_state",
}
REQUIRED_TOP_LEVEL = [
    "stage4a68_map_predict_lambda48_expert_pilot_summary.json",
    "stage4a68_map_predict_lambda48_expert_pilot_summary.md",
    "expert_dataset.npz",
    "expert_dataset_manifest.jsonl",
    "expert_dataset_metadata.json",
    "per_sample_summary.csv",
    "per_sample_summary.json",
    "per_sample_summary.md",
    "lambda48_decisions.jsonl",
    "lambda48_decisions.csv",
    "measured_shadow_decisions.jsonl",
    "measured_shadow_decisions.csv",
    "lambda48_vs_measured_comparison.csv",
    "lambda48_vs_measured_comparison.json",
    "lambda48_vs_measured_comparison.md",
    "map_predict_summary.csv",
    "map_predict_summary.json",
    "map_predict_summary.md",
    "prediction_safety_audit.json",
    "prediction_safety_audit.md",
    "dataset_integrity_report.json",
    "dataset_integrity_report.md",
    "safety_audit.json",
    "safety_audit.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "checkpoint_hash_report.json",
    "checkpoint_hash_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "expert_data_quality_audit.json",
    "expert_data_quality_audit.md",
    "per_sample_quality_table.csv",
    "per_sample_quality_table.json",
    "dataset_quality_summary.json",
    "dataset_quality_summary.md",
    "expert_pilot_index.html",
    "all_samples_contact_sheet.png",
    "lambda48_vs_measured_action_delta_topdown.png",
    "selected_action_distance_hist.png",
    "gain_exp_vs_gain_sc_scatter.png",
    "path_cost_vs_final_score_scatter.png",
    "branch_classification_bar.png",
    "prediction_density_bar.png",
    "safety_flags_summary.png",
    "action_quality_score_bar.png",
    "stage4a68_vs_stage4a67_comparison.csv",
    "stage4a68_vs_stage4a67_comparison.json",
    "stage4a68_vs_stage4a67_comparison.md",
    "stage4a68_vs_stage4a67_action_delta_topdown.png",
    "stage4a68_vs_stage4a67_summary.md",
]
REQUIRED_SAMPLE_GLOBS = [
    "rgb_{idx:03d}.png",
    "depth_{idx:03d}.npy",
    "depth_color_{idx:03d}.png",
    "pose_{idx:03d}.json",
    "observed_state_{idx:03d}.npy",
    "prediction_summary_{idx:03d}.json",
    "lambda48_decision_{idx:03d}.json",
    "measured_shadow_decision_{idx:03d}.json",
    "top_candidates_{idx:03d}.csv",
    "top_candidates_{idx:03d}.jsonl",
    "action_quality_{idx:03d}.json",
    "action_quality_{idx:03d}.md",
    "expert_topdown_{idx:03d}.png",
    "prediction_overlay_{idx:03d}.png",
    "candidate_score_bar_{idx:03d}.png",
    "candidate_map_{idx:03d}.png",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def git_large_artifact_policy_preserved() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    staged_forbidden = []
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/", "assets/home_like_scene_v1/current_environment")
    forbidden_suffixes = (".usd", ".npy", ".npz", ".png", ".mp4")
    for line in result.stdout.splitlines():
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if status.strip() and status[0] != "?" and (
            path.startswith(forbidden_prefixes) or path.endswith(forbidden_suffixes)
        ):
            staged_forbidden.append(line)
    return {
        "git_status_short": result.stdout,
        "staged_forbidden_large_artifacts": staged_forbidden,
        "passed": not staged_forbidden,
    }


def validate_dataset(output_dir: Path, expected_sample_count: int) -> dict[str, Any]:
    dataset_path = output_dir / "expert_dataset.npz"
    require(dataset_path.is_file(), "expert_dataset.npz missing")
    with np.load(dataset_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [
            key
            for key in [
                "map_observation",
                "candidate_features",
                "candidate_mask",
                "expert_action_index_lambda48",
                "expert_action_index_measured_shadow",
                "expert_scores_lambda48",
                "expert_scores_measured",
                "selected_world_xyz_lambda48",
                "selected_yaw_lambda48",
                "selected_world_xyz_measured",
                "selected_yaw_measured",
                "gain_exp",
                "gain_sc",
                "source_occ_free",
                "path_cost",
                "final_score_lambda48",
                "final_score_measured",
                "start_variant_id",
                "pose",
                "valid_mask",
                "safety_flags",
                "prediction_valid_count",
                "predicted_unmeasured_count",
                "predicted_occupied_count",
                "prediction_density",
            ]
            if key not in keys
        ]
        require(not missing, f"dataset missing required keys: {missing}")
        forbidden = sorted(keys & FORBIDDEN_DATASET_KEYS)
        require(not forbidden, f"dataset contains forbidden keys: {forbidden}")
        require(int(data["pose"].shape[0]) == expected_sample_count, "dataset pose sample_count mismatch")
        require(int(data["map_observation"].shape[0]) == expected_sample_count, "map_observation sample_count mismatch")
        require(int(data["candidate_features"].shape[0]) == expected_sample_count, "candidate_features sample_count mismatch")
        for key in ["candidate_features", "expert_scores_lambda48", "expert_scores_measured", "gain_exp", "gain_sc", "path_cost"]:
            arr = np.asarray(data[key])
            finite_values = arr[np.isfinite(arr)]
            require(np.all(np.isfinite(finite_values)), f"{key} has non-finite finite subset failure")
        require(np.all(data["expert_action_index_lambda48"] >= 0), "lambda48 selected action index missing")
        require(np.all(data["expert_action_index_measured_shadow"] >= 0), "measured selected action index missing")
        return {"dataset_keys": sorted(keys), "forbidden": forbidden}


def validate_samples(output_dir: Path, expected_sample_count: int) -> None:
    samples_dir = output_dir / "samples"
    require(samples_dir.is_dir(), "samples dir missing")
    sample_dirs = sorted(samples_dir.glob("start_*"))
    require(len(sample_dirs) == expected_sample_count, f"sample dir count mismatch: {len(sample_dirs)}")
    for sample_dir in sample_dirs:
        idx = int(sample_dir.name.rsplit("_", 1)[-1])
        for pattern in REQUIRED_SAMPLE_GLOBS:
            path = sample_dir / pattern.format(idx=idx)
            require(path.is_file(), f"missing per-sample artifact: {path}")
        pred = load_json(sample_dir / f"prediction_summary_{idx:03d}.json")
        require(pred["map_predict_called"] is True, f"map_predict flag false for sample {idx}")
        require(pred["sscnet_inference_called"] is True, f"sscnet flag false for sample {idx}")
        require(pred["observed_state_hash_unchanged"] is True, f"observed_state changed during prediction for sample {idx}")
        require(pred["prediction_writeback"] is False, f"prediction writeback true for sample {idx}")
        require(pred["prediction_traversability_use"] is False, f"prediction traversability true for sample {idx}")
        require(pred["prediction_collision_use"] is False, f"prediction collision true for sample {idx}")
        require(pred["prediction_ray_blocking_use"] is False, f"prediction ray blocking true for sample {idx}")
        require(pred["prediction_candidate_validity_use"] is False, f"prediction candidate validity true for sample {idx}")
        require(pred["target_ground_truth_use"] is False, f"target/GT use true for sample {idx}")
        quality = load_json(sample_dir / f"action_quality_{idx:03d}.json")
        require("passed" in quality, f"quality missing passed for sample {idx}")


def validate_reports(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a68_map_predict_lambda48_expert_pilot_summary.json")
    integrity = load_json(output_dir / "dataset_integrity_report.json")
    safety = load_json(output_dir / "safety_audit.json")
    pred_safety = load_json(output_dir / "prediction_safety_audit.json")
    quality = load_json(output_dir / "expert_data_quality_audit.json")
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
    source_hash = load_json(output_dir / "source_hash_report.json")
    checkpoint_hash = load_json(output_dir / "checkpoint_hash_report.json")
    comparison = load_json(output_dir / "stage4a68_vs_stage4a67_comparison.json")

    require(summary["completed"] is True, "summary completed false")
    require(summary["dataset_integrity"] is True, "summary dataset_integrity false")
    require(summary["sample_count"] == args.expected_sample_count, "summary sample_count mismatch")
    require(summary["capture_count"] == args.expected_sample_count, "summary capture_count mismatch")
    require(summary["map_predict_calls"] == args.expected_map_predict_calls, "summary map_predict_calls mismatch")
    require(summary["sscnet_inference_called"] is True, "summary SSCNet false")
    require(summary["map_predict_called"] is True, "summary map_predict false")
    require(summary["predictor_loaded_once"] is True, "predictor_loaded_once false")
    require(summary["exactly_one_headless_capture_per_start"] is True, "one capture per start false")
    require(summary["exactly_one_action_per_start"] is True, "one action per start false")
    require(summary["continuous_rollout_executed"] is False, "continuous rollout true")
    require(summary["second_action_executed"] is False, "second action true")
    require(summary["third_action_executed"] is False, "third frame/action true")
    require(summary["rl_training_run"] is False, "rl true")
    require(summary["gdpo_training_run"] is False, "gdpo true")
    require(summary["ppo_training_run"] is False, "ppo true")
    require(summary["behavior_cloning_training_run"] is False, "bc true")
    require(summary["imitation_learning_training_run"] is False, "il true")
    require(summary["training"] is False, "training true")
    require(summary["formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)", "formula mismatch")
    require(float(summary["lambda"]) == 48.0, "lambda mismatch")

    require(integrity["passed"] is True, "dataset integrity failed")
    require(safety["passed"] is True, "safety audit failed")
    require(pred_safety["passed"] is True, "prediction safety failed")
    require(pred_safety["observed_state_hash_unchanged"] is True, "observed hash not unchanged")
    require(pred_safety["prediction_writeback"] is False, "prediction writeback true")
    require(pred_safety["prediction_traversability_use"] is False, "prediction traversability true")
    require(pred_safety["prediction_collision_use"] is False, "prediction collision true")
    require(pred_safety["prediction_ray_blocking_use"] is False, "prediction ray blocking true")
    require(pred_safety["prediction_candidate_validity_use"] is False, "prediction candidate validity true")
    require(pred_safety["target_ground_truth_use"] is False, "target GT true")
    require(pred_safety["future_observed_scoring_use"] is False, "future observed scoring true")
    require(no_rollout["passed"] is True, "no rollout report failed")
    require(no_rollout["continuous_rollout_executed"] is False, "no rollout continuous true")
    require(no_rollout["second_action_executed"] is False, "no rollout second action true")
    require(no_rollout["third_action_executed"] is False, "no rollout third frame true")
    require(no_rl["passed"] is True, "no RL report failed")
    require(no_rl["replay_buffer_created"] is False, "replay buffer created")
    require(no_rl["policy_checkpoint_created"] is False, "policy checkpoint created")
    require(no_rl["checkpoint_modified"] is False, "checkpoint modified")
    require(source_hash["source_usd_unchanged"] is True, "source USD changed")
    require(source_hash["fixed_usd_unchanged"] is True, "fixed USD changed")
    require(source_hash["source_observed_state_unchanged"] is True, "source observed_state changed")
    require(checkpoint_hash["checkpoint_unchanged"] is True, "checkpoint hash changed")
    require(comparison["same_start_variant_count"] is True, "stage67 start count mismatch")
    require(comparison["same_start_variant_ids"] is True, "stage67 start IDs mismatch")
    require((output_dir / "expert_pilot_index.html").is_file(), "HTML missing")
    require(
        (output_dir / "expert_action_flythrough.mp4").is_file()
        or any((output_dir / "expert_action_flythrough_frames").glob("frame_*.png")),
        "MP4 or fallback frames missing",
    )
    require(quality["sample_count"] == args.expected_sample_count, "quality sample_count mismatch")
    require("warnings" in quality, "quality warnings field missing")
    return {
        "summary": summary,
        "integrity": integrity,
        "safety": safety,
        "prediction_safety": pred_safety,
        "quality": quality,
        "comparison": comparison,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_pose_fix_dir", required=True)
    parser.add_argument("--measured_only_pilot_dir", required=True)
    parser.add_argument("--expected_sample_count", type=int, default=10)
    parser.add_argument("--expected_map_predict_calls", type=int, default=10)
    parser.add_argument("--expect_exactly_one_action_per_start", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_second_action", action="store_true")
    parser.add_argument("--expect_no_third_frame", action="store_true")
    parser.add_argument("--expect_map_predict", action="store_true")
    parser.add_argument("--expect_lambda48", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_quality_viz", action="store_true")
    parser.add_argument("--expect_stage4a67_comparison", action="store_true")
    args = parser.parse_args()

    required_flags = [
        args.expect_exactly_one_action_per_start,
        args.expect_no_rollout,
        args.expect_no_second_action,
        args.expect_no_third_frame,
        args.expect_map_predict,
        args.expect_lambda48,
        args.expect_no_rl_gdpo,
        args.expect_no_training,
        args.expect_quality_viz,
        args.expect_stage4a67_comparison,
    ]
    require(all(required_flags), "required expectation flags were not all provided")

    output_dir = Path(args.output_dir)
    require(output_dir.is_dir(), f"output dir missing: {output_dir}")
    require(Path(args.camera_pose_fix_dir).is_dir(), "camera pose fix dir missing")
    require(Path(args.measured_only_pilot_dir).is_dir(), "measured-only pilot dir missing")
    missing_top = [name for name in REQUIRED_TOP_LEVEL if not (output_dir / name).is_file()]
    require(not missing_top, f"missing top-level outputs: {missing_top}")
    require(count_jsonl(output_dir / "expert_dataset_manifest.jsonl") == args.expected_sample_count, "manifest JSONL count mismatch")
    require(count_jsonl(output_dir / "lambda48_decisions.jsonl") == args.expected_sample_count, "lambda48 JSONL count mismatch")
    require(count_jsonl(output_dir / "measured_shadow_decisions.jsonl") == args.expected_sample_count, "measured JSONL count mismatch")

    dataset_info = validate_dataset(output_dir, args.expected_sample_count)
    validate_samples(output_dir, args.expected_sample_count)
    report_info = validate_reports(output_dir, args)
    git_policy = git_large_artifact_policy_preserved()
    require(git_policy["passed"], f"forbidden large artifacts staged: {git_policy['staged_forbidden_large_artifacts']}")

    print(
        json.dumps(
            {
                "all_passed": True,
                "sample_count": report_info["summary"]["sample_count"],
                "map_predict_calls": report_info["summary"]["map_predict_calls"],
                "dataset_keys": dataset_info["dataset_keys"],
                "quality_passed": report_info["quality"]["passed"],
                "git_large_artifact_policy_preserved": git_policy["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
