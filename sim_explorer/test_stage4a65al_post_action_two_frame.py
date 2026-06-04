#!/usr/bin/env python3
"""Validate Stage 4A-6.5al post-action/two-frame diagnosis outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_inputs_manifest.json",
    "loaded_inputs_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "input_hash_audit.json",
    "input_hash_audit.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "runtime_sequence_verification.json",
    "runtime_sequence_verification.md",
    "action_pose_consistency.json",
    "action_pose_consistency.md",
    "no_rollout_reverification.json",
    "no_rollout_reverification.md",
    "observed_state_delta_summary.json",
    "observed_state_delta_summary.md",
    "observed_state_label_transition_matrix.csv",
    "observed_state_label_transition_matrix.json",
    "observed_state_label_transition_matrix.md",
    "observed_state_safety_review.json",
    "observed_state_safety_review.md",
    "map_predict_two_frame_stability.json",
    "map_predict_two_frame_stability.md",
    "prediction_count_delta.csv",
    "prediction_count_delta.json",
    "prediction_count_delta.md",
    "prediction_overlap_summary.json",
    "prediction_overlap_summary.md",
    "prediction_alignment_recheck.json",
    "prediction_alignment_recheck.md",
    "tree_decision_value_components.csv",
    "tree_decision_value_components.json",
    "tree_decision_value_components.md",
    "frame1_decision_diagnosis.json",
    "frame1_decision_diagnosis.md",
    "frame2_decision_diagnosis.json",
    "frame2_decision_diagnosis.md",
    "lambda48_frame1_frame2_comparison.json",
    "lambda48_frame1_frame2_comparison.md",
    "lambda32_vs_lambda48_two_frame.json",
    "lambda32_vs_lambda48_two_frame.md",
    "branch_health_review.json",
    "branch_health_review.md",
    "low_cost_artifact_two_frame_review.json",
    "low_cost_artifact_two_frame_review.md",
    "historical_prior_basin_recheck.json",
    "historical_prior_basin_recheck.md",
    "cost_dominance_review.json",
    "cost_dominance_review.md",
    "consistency_with_stage4a65ag_ai_aj.json",
    "consistency_with_stage4a65ag_ai_aj.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "stage4a65al_post_action_two_frame_diagnosis_summary.json",
    "stage4a65al_post_action_two_frame_diagnosis_summary.md",
    "recommended_next_faithful_step.md",
]

REQUIRED_PLOTS = [
    "observed_state_frame1_frame2_delta_topdown.png",
    "observed_state_label_transition_bar.png",
    "prediction_count_delta_bar.png",
    "prediction_overlay_frame1_vs_frame2.png",
    "tree_branches_frame1_vs_frame2_topdown.png",
    "lambda48_vs_measured_frame1_topdown.png",
    "lambda48_vs_measured_frame2_topdown.png",
    "action_pose_consistency_topdown.png",
    "value_components_frame1_frame2.png",
    "low_cost_artifact_two_frame.png",
    "repeat_safety_readiness_matrix.png",
]

PROHIBITED_PATTERNS = [
    "frame001_rgb.png",
    "frame001_depth.npy",
    "frame001_depth.png",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_depth.png",
    "capture_rgb*.png",
    "capture_depth*.npy",
    "capture_depth*.png",
    "frame003*",
    "action002*",
    "observed_state*.npy",
    "*.npz",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "manifest.jsonl",
    "episode_manifest*",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing required file: {path}"
    assert path.stat().st_size > 0, f"empty required file: {path}"


def assert_png_or_reason(output_dir: Path, name: str) -> None:
    png_path = output_dir / name
    if png_path.is_file():
        assert png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {png_path}"
        return
    reason_path = output_dir / f"{Path(name).stem}_skipped_reason.md"
    assert_file(reason_path)


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_FILES:
        assert_file(output_dir / name)
    for name in REQUIRED_PLOTS:
        assert_png_or_reason(output_dir, name)
    missing = load_json(output_dir / "missing_fields_report.json")
    assert missing["missing_essential_files"] == [], missing["missing_essential_files"]
    assert missing["prohibited_artifacts_found"] == [], missing["prohibited_artifacts_found"]
    return {"passed": True, "required_file_count": len(REQUIRED_FILES)}


def test_hardware(output_dir: Path) -> dict[str, Any]:
    hardware = load_json(output_dir / "hardware_utilization_report.json")
    for key in (
        "os_cpu_count",
        "requested_max_workers",
        "actual_max_workers",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "parallel_backend",
    ):
        assert key in hardware, f"hardware report missing {key}"
    assert int(hardware["requested_max_workers"]) == 32
    assert int(hardware["actual_max_workers"]) == min(32, os.cpu_count() or 1)
    assert str(hardware["OMP_NUM_THREADS"]) == "1"
    assert str(hardware["OPENBLAS_NUM_THREADS"]) == "1"
    assert str(hardware["MKL_NUM_THREADS"]) == "1"
    assert str(hardware["NUMEXPR_NUM_THREADS"]) == "1"
    return {"passed": True}


def test_sequence(output_dir: Path) -> dict[str, Any]:
    seq = load_json(output_dir / "runtime_sequence_verification.json")
    assert seq["frames_captured"] == 2
    assert seq["map_predict_calls"] == 2
    assert seq["selected_action_execution_count"] == 1
    assert seq["second_action"] is False
    assert seq["third_frame"] is False
    assert seq["rollout"] is False
    assert seq["verification_clean"] is True

    action = load_json(output_dir / "action_pose_consistency.json")
    assert action["post_action_pose_consistent"] is True
    assert action["position_matches"] is True
    assert action["yaw_matches"] is True

    no_rollout = load_json(output_dir / "no_rollout_reverification.json")
    assert no_rollout["no_rollout_reverified"] is True
    return {"passed": True}


def test_diagnosis_outputs(output_dir: Path) -> dict[str, Any]:
    observed = load_json(output_dir / "observed_state_delta_summary.json")
    assert observed["observed_ratio_non_decreasing"] is True
    assert observed["meaningful_measured_information_added"] is True
    assert observed["suspicious_label_flips"] is False
    assert observed["newly_observed_voxels"] > 0

    observed_safety = load_json(output_dir / "observed_state_safety_review.json")
    assert observed_safety["review_clean"] is True
    assert observed_safety["prediction_written_into_observed_state"] is False

    map_pred = load_json(output_dir / "map_predict_two_frame_stability.json")
    assert map_pred["map_predict_succeeded_both_frames"] is True
    assert map_pred["prediction_density_exploded"] is False
    assert map_pred["prediction_density_collapsed"] is False
    assert map_pred["frame2_prediction_remains_reasonable_post_action"] is True

    alignment = load_json(output_dir / "prediction_alignment_recheck.json")
    assert alignment["alignment_recheck_clean"] is True
    assert alignment["frame001_code_consistent_v1"] is True
    assert alignment["frame002_code_consistent_v1"] is True

    for name in (
        "tree_decision_value_components.json",
        "frame1_decision_diagnosis.json",
        "frame2_decision_diagnosis.json",
        "lambda48_frame1_frame2_comparison.json",
        "lambda32_vs_lambda48_two_frame.json",
        "low_cost_artifact_two_frame_review.json",
        "historical_prior_basin_recheck.json",
        "consistency_with_stage4a65ag_ai_aj.json",
        "repeat_safety_readiness_matrix.json",
        "risk_register.json",
    ):
        assert_file(output_dir / name)

    branch = load_json(output_dir / "branch_health_review.json")
    assert branch["review_clean"] is True
    low_cost = load_json(output_dir / "low_cost_artifact_two_frame_review.json")
    assert low_cost["review_clean"] is True
    prior = load_json(output_dir / "historical_prior_basin_recheck.json")
    assert prior["review_clean"] is True
    cost = load_json(output_dir / "cost_dominance_review.json")
    assert cost["review_clean"] is True
    return {"passed": True}


def test_hashes_and_forbidden_outputs(output_dir: Path, stage4a65ak_dir: Path) -> dict[str, Any]:
    audit = load_json(output_dir / "input_hash_audit.json")
    assert audit["all_unchanged"] is True
    unchanged = audit["unchanged"]
    for suffix in (
        "observed_state_frame001.npy",
        "observed_state_frame002.npy",
        "frame001_map_predict/global_prediction_layer.npz",
        "frame002_map_predict/global_prediction_layer.npz",
    ):
        path = str(stage4a65ak_dir / suffix)
        assert unchanged[path] is True, f"input modified: {path}"

    for pattern in PROHIBITED_PATTERNS:
        found = sorted(output_dir.glob(pattern))
        assert found == [], f"forbidden 6.5al output pattern {pattern}: {found}"
    return {"passed": True}


def test_readiness_and_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65al_post_action_two_frame_diagnosis_summary.json")
    readiness = summary["readiness"]
    safety = summary["safety"]
    assert readiness["rollout_ready"] is False
    assert readiness["open_ended_loop_ready"] is False
    assert readiness["rl_ppo_bc_il_ready"] is False
    assert readiness["prediction_writeback_fusion_ready"] is False
    assert readiness["over_cost_runtime_promotion_ready"] is False
    assert "rollout" not in readiness["next_small_task"].lower()

    recommendation = (output_dir / "recommended_next_faithful_step.md").read_text(encoding="utf-8").lower()
    forbidden_recommendations = [
        "recommend rollout directly",
        "rl/ppo/bc/il",
        "prediction writeback/fusion",
        "over-cost runtime promotion",
    ]
    for phrase in forbidden_recommendations:
        assert phrase in recommendation, f"recommendation should explicitly reject {phrase}"

    false_keys = [
        "isaac_startup_in_stage4a65al",
        "rgb_depth_capture_in_stage4a65al",
        "map_predict_call_in_stage4a65al",
        "sscnet_inference_in_stage4a65al",
        "selected_action_execution_in_stage4a65al",
        "two_frame_runtime_execution_in_stage4a65al",
        "rollout_in_stage4a65al",
        "open_ended_loop",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "existing_prediction_npz_modified",
        "prediction_writeback",
        "prediction_fusion",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "prediction_used_for_candidate_sampling_edge_validity",
        "target_ground_truth_planning_scoring",
        "future_observed_planning_scoring",
        "external_source_modified_built",
        "over_cost_runtime_primary",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        assert safety[key] is False, f"safety key should be false: {key}"

    readiness_rows = load_json(output_dir / "repeat_safety_readiness_matrix.json")
    rollout_rows = [row for row in readiness_rows if row["stage_option"] == "rollout"]
    assert len(rollout_rows) == 1
    assert rollout_rows[0]["recommended_now"] is False
    assert "not rollout evidence" in rollout_rows[0]["blocked_reason"]
    return {"passed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage4a65ak_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    stage4a65ak_dir = Path(args.stage4a65ak_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "hardware": test_hardware(output_dir),
        "sequence": test_sequence(output_dir),
        "diagnosis_outputs": test_diagnosis_outputs(output_dir),
        "hashes_and_forbidden_outputs": test_hashes_and_forbidden_outputs(output_dir, stage4a65ak_dir),
        "readiness_and_safety": test_readiness_and_safety(output_dir),
    }
    print(json.dumps({"passed": True, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
