#!/usr/bin/env python3
"""Validate Stage 4A-6.9 bounded two-frame lambda48 pilot outputs."""

from __future__ import annotations

import argparse
import json
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
    "stage4a69_bounded_two_frame_lambda48_pilot_summary.json",
    "stage4a69_bounded_two_frame_lambda48_pilot_summary.md",
    "expert_dataset_two_frame.npz",
    "expert_dataset_manifest.jsonl",
    "expert_dataset_metadata.json",
    "per_start_summary.csv",
    "per_start_summary.json",
    "per_start_summary.md",
    "per_frame_summary.csv",
    "per_frame_summary.json",
    "per_frame_summary.md",
    "frame1_lambda48_decisions.jsonl",
    "frame1_lambda48_decisions.csv",
    "frame1_measured_shadow_decisions.jsonl",
    "frame1_measured_shadow_decisions.csv",
    "frame2_lambda48_diagnostic_decisions.jsonl",
    "frame2_lambda48_diagnostic_decisions.csv",
    "frame2_measured_shadow_diagnostic_decisions.jsonl",
    "frame2_measured_shadow_diagnostic_decisions.csv",
    "frame1_lambda48_vs_measured_comparison.csv",
    "frame1_lambda48_vs_measured_comparison.json",
    "frame1_lambda48_vs_measured_comparison.md",
    "frame2_lambda48_vs_measured_comparison.csv",
    "frame2_lambda48_vs_measured_comparison.json",
    "frame2_lambda48_vs_measured_comparison.md",
    "stage4a69_vs_stage4a68_comparison.csv",
    "stage4a69_vs_stage4a68_comparison.json",
    "stage4a69_vs_stage4a68_comparison.md",
    "stage4a69_vs_stage4a67_comparison.csv",
    "stage4a69_vs_stage4a67_comparison.json",
    "stage4a69_vs_stage4a67_comparison.md",
    "map_predict_summary.csv",
    "map_predict_summary.json",
    "map_predict_summary.md",
    "prediction_safety_audit.json",
    "prediction_safety_audit.md",
    "dataset_integrity_report.json",
    "dataset_integrity_report.md",
    "safety_audit.json",
    "safety_audit.md",
    "expert_data_quality_audit.json",
    "expert_data_quality_audit.md",
    "two_frame_stability_audit.json",
    "two_frame_stability_audit.md",
    "no_long_rollout_report.json",
    "no_long_rollout_report.md",
    "no_second_action_report.json",
    "no_second_action_report.md",
    "no_third_frame_report.json",
    "no_third_frame_report.md",
    "no_rl_gdpo_report.json",
    "no_rl_gdpo_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "checkpoint_hash_report.json",
    "checkpoint_hash_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "expert_two_frame_index.html",
    "all_samples_frame1_contact_sheet.png",
    "all_samples_frame2_contact_sheet.png",
    "two_frame_path_contact_sheet.png",
    "frame1_lambda48_vs_measured_action_delta_topdown.png",
    "frame2_lambda48_vs_measured_action_delta_topdown.png",
    "observed_delta_by_start.png",
    "map_predict_density_by_frame.png",
    "action_distance_hist_frame1.png",
    "action_distance_hist_frame2.png",
    "gain_exp_vs_source_occ_free_scatter_frame1.png",
    "gain_exp_vs_source_occ_free_scatter_frame2.png",
    "path_cost_vs_final_score_scatter_frame1.png",
    "path_cost_vs_final_score_scatter_frame2.png",
    "branch_classification_bar_frame1.png",
    "branch_classification_bar_frame2.png",
    "safety_flags_summary.png",
    "quality_warning_summary.png",
]
REQUIRED_SAMPLE_FILES = [
    "frame1_rgb.png",
    "frame1_depth.npy",
    "frame1_depth_color.png",
    "frame1_pose.json",
    "frame1_observed_state.npy",
    "frame1_prediction_summary.json",
    "frame1_lambda48_decision.json",
    "frame1_measured_shadow_decision.json",
    "frame1_top_candidates.csv",
    "frame1_top_candidates.jsonl",
    "frame1_action_quality.json",
    "frame1_action_quality.md",
    "frame1_expert_topdown.png",
    "frame1_prediction_overlay.png",
    "frame1_candidate_score_bar.png",
    "frame1_candidate_map.png",
    "executed_action.json",
    "executed_action.md",
    "frame2_rgb.png",
    "frame2_depth.npy",
    "frame2_depth_color.png",
    "frame2_pose.json",
    "frame2_observed_state.npy",
    "frame2_prediction_summary.json",
    "frame2_lambda48_diagnostic_decision.json",
    "frame2_measured_shadow_diagnostic_decision.json",
    "frame2_top_candidates.csv",
    "frame2_top_candidates.jsonl",
    "frame2_action_quality.json",
    "frame2_action_quality.md",
    "frame2_expert_topdown.png",
    "frame2_prediction_overlay.png",
    "frame2_candidate_score_bar.png",
    "frame2_candidate_map.png",
    "two_frame_path_topdown.png",
    "observed_delta_topdown.png",
    "action_quality_two_frame.md",
    "sample_quality_summary.json",
    "sample_quality_summary.md",
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
        if status.strip() and status[0] != "?" and (path.startswith(forbidden_prefixes) or path.endswith(forbidden_suffixes)):
            staged_forbidden.append(line)
    return {"git_status_short": result.stdout, "staged_forbidden_large_artifacts": staged_forbidden, "passed": not staged_forbidden}


def validate_dataset(output_dir: Path, expected_start_count: int) -> dict[str, Any]:
    dataset_path = output_dir / "expert_dataset_two_frame.npz"
    require(dataset_path.is_file(), "expert_dataset_two_frame.npz missing")
    required_keys = [
        "start_variant_id",
        "frame1_observed_state_reference",
        "frame1_candidate_features",
        "frame1_candidate_mask",
        "frame1_lambda48_action_index",
        "frame1_measured_action_index",
        "frame1_lambda48_scores",
        "frame1_measured_scores",
        "frame1_selected_world_xyz_lambda48",
        "frame1_selected_yaw_lambda48",
        "frame1_selected_world_xyz_measured",
        "frame1_selected_yaw_measured",
        "frame1_gain_exp",
        "frame1_source_occ_free",
        "frame1_path_cost",
        "frame1_final_score_lambda48",
        "frame1_prediction_summary",
        "executed_action_world_xyz",
        "executed_action_yaw",
        "frame2_observed_state_reference",
        "frame2_candidate_features",
        "frame2_candidate_mask",
        "frame2_lambda48_diagnostic_action_index",
        "frame2_measured_diagnostic_action_index",
        "frame2_lambda48_scores",
        "frame2_measured_scores",
        "frame2_selected_world_xyz_lambda48",
        "frame2_selected_yaw_lambda48",
        "frame2_selected_world_xyz_measured",
        "frame2_selected_yaw_measured",
        "frame2_gain_exp",
        "frame2_source_occ_free",
        "frame2_path_cost",
        "frame2_final_score_lambda48",
        "frame2_prediction_summary",
        "observed_delta",
        "safety_flags",
        "quality_flags",
    ]
    with np.load(dataset_path, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [key for key in required_keys if key not in keys]
        require(not missing, f"dataset missing keys: {missing}")
        forbidden = sorted(keys & FORBIDDEN_DATASET_KEYS)
        require(not forbidden, f"dataset contains forbidden keys: {forbidden}")
        require(int(data["start_variant_id"].shape[0]) == expected_start_count, "start count mismatch")
        require(int(data["frame1_candidate_features"].shape[0]) == expected_start_count, "frame1 feature count mismatch")
        require(int(data["frame2_candidate_features"].shape[0]) == expected_start_count, "frame2 feature count mismatch")
        for key in [
            "frame1_candidate_features",
            "frame1_lambda48_scores",
            "frame1_measured_scores",
            "frame2_candidate_features",
            "frame2_lambda48_scores",
            "frame2_measured_scores",
        ]:
            arr = np.asarray(data[key])
            require(np.all(np.isfinite(arr[np.isfinite(arr)])), f"{key} finite subset failed")
        require(np.all(data["frame1_lambda48_action_index"] >= 0), "frame1 lambda action missing")
        require(np.all(data["frame2_lambda48_diagnostic_action_index"] >= 0), "frame2 lambda diagnostic action missing")
        return {"dataset_keys": sorted(keys), "forbidden": forbidden}


def validate_samples(output_dir: Path, expected_start_count: int) -> None:
    samples_dir = output_dir / "samples"
    require(samples_dir.is_dir(), "samples dir missing")
    sample_dirs = sorted(samples_dir.glob("start_*"))
    require(len(sample_dirs) == expected_start_count, f"sample dir count mismatch: {len(sample_dirs)}")
    for sample_dir in sample_dirs:
        for name in REQUIRED_SAMPLE_FILES:
            require((sample_dir / name).is_file(), f"missing per-start artifact: {sample_dir / name}")
        for frame_id in (1, 2):
            pred = load_json(sample_dir / f"frame{frame_id}_prediction_summary.json")
            require(pred["map_predict_called"] is True, f"map_predict false in {sample_dir} frame {frame_id}")
            require(pred["sscnet_inference_called"] is True, f"sscnet false in {sample_dir} frame {frame_id}")
            require(pred["observed_state_hash_unchanged"] is True, f"observed hash changed in {sample_dir} frame {frame_id}")
            require(pred["prediction_writeback"] is False, f"prediction writeback true in {sample_dir} frame {frame_id}")
            require(pred["prediction_traversability_use"] is False, f"prediction traversability true in {sample_dir} frame {frame_id}")
            require(pred["prediction_collision_use"] is False, f"prediction collision true in {sample_dir} frame {frame_id}")
            require(pred["prediction_ray_blocking_use"] is False, f"prediction ray blocking true in {sample_dir} frame {frame_id}")
            require(pred["prediction_candidate_validity_use"] is False, f"prediction candidate validity true in {sample_dir} frame {frame_id}")
            require(pred["prediction_edge_validity_use"] is False, f"prediction edge validity true in {sample_dir} frame {frame_id}")
            require(pred["target_ground_truth_use"] is False, f"target GT use true in {sample_dir} frame {frame_id}")
            require(pred["future_observed_scoring_use"] is False, f"future observed true in {sample_dir} frame {frame_id}")
        executed = load_json(sample_dir / "executed_action.json")
        require(executed["second_action_executed"] is False, f"second action true in {sample_dir}")
        require(executed["third_frame_executed"] is False, f"third frame true in {sample_dir}")


def validate_reports(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a69_bounded_two_frame_lambda48_pilot_summary.json")
    integrity = load_json(output_dir / "dataset_integrity_report.json")
    safety = load_json(output_dir / "safety_audit.json")
    pred_safety = load_json(output_dir / "prediction_safety_audit.json")
    quality = load_json(output_dir / "expert_data_quality_audit.json")
    stability = load_json(output_dir / "two_frame_stability_audit.json")
    no_long = load_json(output_dir / "no_long_rollout_report.json")
    no_second = load_json(output_dir / "no_second_action_report.json")
    no_third = load_json(output_dir / "no_third_frame_report.json")
    no_rl = load_json(output_dir / "no_rl_gdpo_report.json")
    source_hash = load_json(output_dir / "source_hash_report.json")
    checkpoint_hash = load_json(output_dir / "checkpoint_hash_report.json")
    stage68_cmp = load_json(output_dir / "stage4a69_vs_stage4a68_comparison.json")
    stage67_cmp = load_json(output_dir / "stage4a69_vs_stage4a67_comparison.json")

    require(summary["completed"] is True, "summary completed false")
    require(summary["start_count"] == args.expected_start_count, "start_count mismatch")
    require(summary["frame_count"] == args.expected_frame_count, "frame_count mismatch")
    require(summary["capture_count"] == args.expected_frame_count, "capture_count mismatch")
    require(summary["map_predict_calls"] == args.expected_map_predict_calls, "map_predict_calls mismatch")
    require(summary["executed_action_count"] == args.expected_start_count, "executed_action_count mismatch")
    require(summary["exactly_one_action_per_start"] is True, "one action per start false")
    require(summary["second_action_count"] == 0, "second action count nonzero")
    require(summary["third_frame_count"] == 0, "third frame count nonzero")
    require(summary["continuous_rollout_executed"] is False, "continuous rollout true")
    require(summary["long_rollout_executed"] is False, "long rollout true")
    require(summary["sscnet_inference_called"] is True, "sscnet false")
    require(summary["map_predict_called"] is True, "map_predict false")
    require(summary["predictor_loaded_once"] is True, "predictor_loaded_once false")
    require(summary["formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)", "formula mismatch")
    require(float(summary["lambda_sc"]) == 48.0, "lambda_sc mismatch")

    require(integrity["passed"] is True, "dataset integrity failed")
    require(safety["passed"] is True, "safety audit failed")
    require(pred_safety["passed"] is True, "prediction safety failed")
    require(quality["passed"] is True, "quality audit failed")
    require(stability["passed"] is True, "two-frame stability failed")
    for report, name in ((no_long, "no_long"), (no_second, "no_second"), (no_third, "no_third"), (no_rl, "no_rl")):
        require(report["passed"] is True, f"{name} report failed")
    require(no_long["long_rollout_executed"] is False, "long rollout report true")
    require(no_second["second_action_count"] == 0, "second action report nonzero")
    require(no_third["third_frame_count"] == 0, "third frame report nonzero")
    require(no_rl["replay_buffer_created"] is False, "replay buffer created")
    require(no_rl["policy_checkpoint_created"] is False, "policy checkpoint created")
    require(no_rl["checkpoint_modified"] is False, "checkpoint modified")
    require(source_hash["source_usd_unchanged"] is True, "source USD changed")
    require(source_hash["fixed_usd_unchanged"] is True, "fixed USD changed")
    require(source_hash["source_observed_state_unchanged"] is True, "source observed_state changed")
    require(checkpoint_hash["checkpoint_unchanged"] is True, "checkpoint changed")
    require(stage68_cmp["same_10_starts"] is True, "Stage 4A-6.8 comparison start mismatch")
    require(stage67_cmp["same_start_variant_ids"] is True, "Stage 4A-6.7 comparison start mismatch")
    require((output_dir / "expert_two_frame_index.html").is_file(), "HTML missing")
    require(
        (output_dir / "expert_two_frame_flythrough.mp4").is_file()
        or any((output_dir / "expert_two_frame_flythrough_frames").glob("frame_*.png")),
        "MP4 or fallback frames missing",
    )
    return {
        "summary": summary,
        "integrity": integrity,
        "safety": safety,
        "prediction_safety": pred_safety,
        "quality": quality,
        "stability": stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--camera_pose_fix_dir", required=True)
    parser.add_argument("--measured_only_pilot_dir", required=True)
    parser.add_argument("--lambda48_pilot_dir", required=True)
    parser.add_argument("--expected_start_count", type=int, default=10)
    parser.add_argument("--expected_frame_count", type=int, default=20)
    parser.add_argument("--expected_map_predict_calls", type=int, default=20)
    parser.add_argument("--expect_exactly_one_action_per_start", action="store_true")
    parser.add_argument("--expect_no_second_action", action="store_true")
    parser.add_argument("--expect_no_third_frame", action="store_true")
    parser.add_argument("--expect_no_long_rollout", action="store_true")
    parser.add_argument("--expect_map_predict", action="store_true")
    parser.add_argument("--expect_lambda48", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_quality_viz", action="store_true")
    parser.add_argument("--expect_stage4a68_comparison", action="store_true")
    parser.add_argument("--expect_stage4a67_comparison", action="store_true")
    args = parser.parse_args()

    require(
        all(
            [
                args.expect_exactly_one_action_per_start,
                args.expect_no_second_action,
                args.expect_no_third_frame,
                args.expect_no_long_rollout,
                args.expect_map_predict,
                args.expect_lambda48,
                args.expect_no_rl_gdpo,
                args.expect_no_training,
                args.expect_quality_viz,
                args.expect_stage4a68_comparison,
                args.expect_stage4a67_comparison,
            ]
        ),
        "required expectation flags were not all provided",
    )
    output_dir = Path(args.output_dir)
    require(output_dir.is_dir(), f"output dir missing: {output_dir}")
    require(Path(args.camera_pose_fix_dir).is_dir(), "camera pose fix dir missing")
    require(Path(args.measured_only_pilot_dir).is_dir(), "measured-only pilot dir missing")
    require(Path(args.lambda48_pilot_dir).is_dir(), "lambda48 pilot dir missing")
    missing_top = [name for name in REQUIRED_TOP_LEVEL if not (output_dir / name).is_file()]
    require(not missing_top, f"missing top-level outputs: {missing_top}")
    require(count_jsonl(output_dir / "expert_dataset_manifest.jsonl") == args.expected_frame_count, "manifest JSONL count mismatch")
    require(count_jsonl(output_dir / "frame1_lambda48_decisions.jsonl") == args.expected_start_count, "frame1 lambda JSONL count mismatch")
    require(count_jsonl(output_dir / "frame1_measured_shadow_decisions.jsonl") == args.expected_start_count, "frame1 measured JSONL count mismatch")
    require(count_jsonl(output_dir / "frame2_lambda48_diagnostic_decisions.jsonl") == args.expected_start_count, "frame2 lambda JSONL count mismatch")
    require(count_jsonl(output_dir / "frame2_measured_shadow_diagnostic_decisions.jsonl") == args.expected_start_count, "frame2 measured JSONL count mismatch")

    dataset_info = validate_dataset(output_dir, args.expected_start_count)
    validate_samples(output_dir, args.expected_start_count)
    report_info = validate_reports(output_dir, args)
    git_policy = git_large_artifact_policy_preserved()
    require(git_policy["passed"], f"forbidden large artifacts staged: {git_policy['staged_forbidden_large_artifacts']}")

    print(
        json.dumps(
            {
                "all_passed": True,
                "start_count": report_info["summary"]["start_count"],
                "frame_count": report_info["summary"]["frame_count"],
                "map_predict_calls": report_info["summary"]["map_predict_calls"],
                "dataset_keys": dataset_info["dataset_keys"],
                "quality_passed": report_info["quality"]["passed"],
                "two_frame_stability_passed": report_info["stability"]["passed"],
                "git_large_artifact_policy_preserved": git_policy["passed"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
