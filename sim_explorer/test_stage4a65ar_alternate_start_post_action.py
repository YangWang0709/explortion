#!/usr/bin/env python3
"""Validate Stage 4A-6.5ar alternate-start diagnosis outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_input_manifest.json",
    "loaded_input_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "input_hash_audit.json",
    "input_hash_audit.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "sequence_safety_reverification.json",
    "sequence_safety_reverification.md",
    "prediction_safety_reverification.json",
    "prediction_safety_reverification.md",
    "no_rollout_reverification.json",
    "no_rollout_reverification.md",
    "forbidden_artifact_scan.json",
    "forbidden_artifact_scan.md",
    "alternate_start_pose_consistency.json",
    "alternate_start_pose_consistency.md",
    "action_pose_consistency.json",
    "action_pose_consistency.md",
    "start_to_action_geometry.json",
    "start_to_action_geometry.md",
    "observed_state_delta_summary.json",
    "observed_state_delta_summary.md",
    "observed_state_transition_table.csv",
    "observed_state_transition_table.json",
    "observed_state_transition_table.md",
    "measured_only_update_review.json",
    "measured_only_update_review.md",
    "map_predict_two_frame_stability.json",
    "map_predict_two_frame_stability.md",
    "prediction_count_comparison.csv",
    "prediction_count_comparison.json",
    "prediction_count_comparison.md",
    "prediction_safety_review.json",
    "prediction_safety_review.md",
    "frame1_tree_decision_diagnosis.json",
    "frame1_tree_decision_diagnosis.md",
    "frame2_tree_decision_diagnosis.json",
    "frame2_tree_decision_diagnosis.md",
    "lambda32_lambda48_agreement.json",
    "lambda32_lambda48_agreement.md",
    "value_component_review.csv",
    "value_component_review.json",
    "value_component_review.md",
    "low_cost_artifact_review.json",
    "low_cost_artifact_review.md",
    "historical_prior_basin_review.json",
    "historical_prior_basin_review.md",
    "branch_health_review.json",
    "branch_health_review.md",
    "alternate_start_outcome_classification.json",
    "alternate_start_outcome_classification.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "recommended_next_faithful_step.md",
    "selected_next_bounded_repeat_design.json",
    "selected_next_bounded_repeat_design.md",
    "future_stage4a65as_command_sketch.md",
    "do_not_run_runtime_in_stage4a65ar.md",
    "stage4a65ar_alternate_start_diagnosis_summary.json",
    "stage4a65ar_alternate_start_diagnosis_summary.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_PLOTS = [
    "alternate_start_pose_and_action_topdown.png",
    "observed_state_delta_topdown.png",
    "observed_transition_bar.png",
    "prediction_count_two_frame_bar.png",
    "frame1_measured_vs_lambda48_topdown.png",
    "frame2_measured_vs_lambda48_topdown.png",
    "lambda32_lambda48_comparison_topdown.png",
    "value_component_comparison.png",
    "repeat_safety_readiness_matrix.png",
    "next_stage_decision_flowchart.png",
]

PROHIBITED_PATTERNS = [
    "*.npy",
    "*.npz",
    "frame001*",
    "frame002*",
    "frame003*",
    "action002*",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "manifest.jsonl",
    "episode_manifest*",
    "*replay_buffer*",
    "*policy_checkpoint*",
    "*training_output*",
]

PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
EXPECTED_POSITION = [0.0, -4.45, 1.2]
EXPECTED_YAW = 1.5707963267948966


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"


def assert_png_or_reason(output_dir: Path, name: str) -> None:
    path = output_dir / name
    if path.is_file():
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
        return
    assert_file(output_dir / f"{Path(name).stem}_skipped_reason.md")


def assert_close_list(actual: list[float], expected: list[float], atol: float = 1.0e-9) -> None:
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        assert math.isclose(float(a), float(b), abs_tol=atol), (actual, expected)


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_FILES:
        assert_file(output_dir / name)
    for name in REQUIRED_PLOTS:
        assert_png_or_reason(output_dir, name)
    missing = load_json(output_dir / "missing_fields_report.json")
    assert missing["missing_essential_files"] == [], missing["missing_essential_files"]
    assert missing["missing_required_outputs_before_report_write"] == [], missing["missing_required_outputs_before_report_write"]
    assert missing["missing_plots_without_skip_reason"] == [], missing["missing_plots_without_skip_reason"]
    assert missing["prohibited_artifacts_found"] == [], missing["prohibited_artifacts_found"]
    forbidden = load_json(output_dir / "forbidden_artifact_scan.json")
    assert forbidden["clean"] is True
    for pattern in PROHIBITED_PATTERNS:
        found = sorted(output_dir.glob(pattern))
        assert found == [], f"forbidden output pattern {pattern}: {found}"
    return {"passed": True, "required_files": len(REQUIRED_FILES), "required_plots": len(REQUIRED_PLOTS)}


def test_hardware(output_dir: Path) -> dict[str, Any]:
    hardware = load_json(output_dir / "hardware_utilization_report.json")
    assert "os_cpu_count" in hardware
    assert int(hardware["requested_max_workers"]) == 32
    assert int(hardware["actual_max_workers"]) == min(32, os.cpu_count() or 1)
    assert hardware["parallel_backend"] == "ThreadPoolExecutor"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        assert str(hardware[key]) == "1", f"{key} was {hardware[key]}"
    assert int(hardware["analysis_task_count"]) > 0
    return {"passed": True}


def test_sequence_and_safety(output_dir: Path) -> dict[str, Any]:
    seq = load_json(output_dir / "sequence_safety_reverification.json")
    aq = seq["stage4a65aq"]
    assert aq["isaac_startup_count_clean_run"] == 1
    assert aq["frames_captured"] == 2
    assert aq["map_predict_calls"] == 2
    assert aq["selected_action_execution_count"] == 1
    assert aq["second_action"] is False
    assert aq["third_frame"] is False
    assert aq["rollout"] is False
    assert aq["sequence_clean"] is True
    assert seq["start_variant"] == "start_corridor"
    assert_close_list(seq["start_pose"], EXPECTED_POSITION)
    assert math.isclose(float(seq["start_yaw"]), EXPECTED_YAW, abs_tol=1.0e-9)
    assert seq["tree_seed"] == 0
    assert seq["formula_contract"] == PRIMARY_FORMULA
    assert seq["formula_is_not_over_cost"] is True
    runtime = seq["runtime_in_stage4a65ar"]
    for key in ("isaac_startup", "rgb_depth_capture", "map_predict_call", "sscnet_inference", "selected_action_execution", "two_frame_runtime_execution", "rollout"):
        assert runtime[key] is False, key

    pred = load_json(output_dir / "prediction_safety_reverification.json")
    assert pred["prediction_safety_clean"] is True
    assert pred["no_prediction_writeback_or_fusion"] is True
    assert pred["no_prediction_motion_safety_use"] is True
    assert pred["no_target_ground_truth_future_observed_scoring"] is True

    no_rollout = load_json(output_dir / "no_rollout_reverification.json")
    assert no_rollout["rollout"] is False
    assert no_rollout["frame003_captured"] is False
    assert no_rollout["second_action_executed"] is False
    assert no_rollout["rollout_ready"] is False
    return {"passed": True}


def test_pose_observed_prediction(output_dir: Path) -> dict[str, Any]:
    pose = load_json(output_dir / "alternate_start_pose_consistency.json")
    assert pose["start_variant"] == "start_corridor"
    assert pose["matches_expected_pose"] is True
    assert pose["matches_stage4a65ap_design"] is True
    assert pose["matches_metadata"] is True
    assert_close_list(pose["stage4a65aq_position"], EXPECTED_POSITION)

    action = load_json(output_dir / "action_pose_consistency.json")
    assert action["action_executed"] is True
    assert action["action_execution_count"] == 1
    assert action["frame2_pose_matches_executed_action"] is True
    assert action["selected_child_xy_matches_executed_position_xy"] is True
    assert action["action_inside_map_bounds"] is True
    assert action["no_prediction_based_traversability_required"] is True

    observed = load_json(output_dir / "observed_state_delta_summary.json")
    assert observed["observed_delta_positive"] is True
    assert observed["newly_observed"] == 5222
    assert observed["unknown_to_free"] == 4876
    assert observed["unknown_to_occupied"] == 346
    assert observed["occupied_to_free"] == 0
    assert observed["invalid_labels"] == 0
    assert observed["measured_only_update_status"] is True

    maps = load_json(output_dir / "map_predict_two_frame_stability.json")
    assert maps["frame001_prediction_valid_count"] == 61152
    assert maps["frame001_predicted_unmeasured_occ_free"] == 49164
    assert maps["frame002_prediction_valid_count"] == 52988
    assert maps["frame002_predicted_unmeasured_occ_free"] == 43828
    assert math.isclose(float(maps["density_ratio_frame2_over_frame1"]), 0.8914652998128713, rel_tol=0.0, abs_tol=1.0e-12)
    assert maps["code_consistent_v1_check"] is True
    assert maps["no_explosion_or_collapse"] is True
    return {"passed": True}


def test_tree_branch_outcome(output_dir: Path) -> dict[str, Any]:
    frame1 = load_json(output_dir / "frame1_tree_decision_diagnosis.json")
    frame2 = load_json(output_dir / "frame2_tree_decision_diagnosis.json")
    assert frame1["lambda48"]["selected_child_id"] == "n0001"
    assert frame1["lambda48"]["best_descendant_id"] == "n0104"
    assert frame1["measured_only"]["best_descendant_id"] == "n0104"
    assert frame1["measured_vs_lambda48"]["same_selected_child_id"] is True
    assert frame1["measured_vs_lambda48"]["same_best_descendant_id"] is True
    assert frame1["classification"] == "same_as_measured"

    assert frame2["lambda48"]["selected_child_id"] == "n0001"
    assert frame2["lambda48"]["best_descendant_id"] == "n0127"
    assert frame2["measured_only"]["best_descendant_id"] == "n0126"
    assert frame2["measured_vs_lambda48"]["same_selected_child_id"] is True
    assert frame2["measured_vs_lambda48"]["same_best_descendant_id"] is False
    assert frame2["classification"] == "same_as_measured"

    lam = load_json(output_dir / "lambda32_lambda48_agreement.json")
    assert lam["frame001_selected_child_match"] is True
    assert lam["frame001_best_descendant_match"] is True
    assert lam["frame002_selected_child_match"] is True
    assert lam["frame002_best_descendant_match"] is False

    low = load_json(output_dir / "low_cost_artifact_review.json")
    prior = load_json(output_dir / "historical_prior_basin_review.json")
    branch = load_json(output_dir / "branch_health_review.json")
    assert low["any_low_cost_artifact"] is False
    assert prior["any_historical_prior_basin"] is False
    assert branch["branch_health_clean"] is True

    outcome = load_json(output_dir / "alternate_start_outcome_classification.json")
    assert outcome["classification"] == "clean_same_as_measured"
    assert outcome["clean_same_as_measured_is_bad"] is False
    assert outcome["coverage_improvement_proven"] is False
    assert outcome["rollout_ready"] is False
    assert outcome["rollout_recommended"] is False
    assert outcome["rl_gdpo_ppo_bc_il_recommended"] is False
    return {"passed": True}


def test_future_command_and_notes(output_dir: Path, expected_future_stage: str) -> dict[str, Any]:
    design = load_json(output_dir / "selected_next_bounded_repeat_design.json")
    assert design["future_stage"] == expected_future_stage
    assert design["design_only_in_stage4a65ar"] is True
    assert design["was_executed_in_stage4a65ar"] is False
    assert design["start_variant"] == "start_corridor"
    assert_close_list(design["pose"], EXPECTED_POSITION)
    assert math.isclose(float(design["yaw_rad"]), EXPECTED_YAW, abs_tol=1.0e-9)
    assert design["future_tree_seed"] == 1
    assert design["formula"] == PRIMARY_FORMULA
    assert design["runtime_constraints"]["max_frames"] == 2
    assert design["runtime_constraints"]["no_second_action"] is True
    assert design["runtime_constraints"]["no_third_frame"] is True
    assert design["runtime_constraints"]["no_rollout"] is True

    sketch = (output_dir / "future_stage4a65as_command_sketch.md").read_text(encoding="utf-8")
    lines = sketch.splitlines()
    assert lines[0] == "DO NOT RUN IN STAGE 4A-6.5ar."
    assert lines[1] == f"This is a future Stage {expected_future_stage} command sketch only."
    assert "--max_workers 32" in sketch
    assert "--max_frames 2" in sketch
    assert "--no_second_action" in sketch
    assert "--no_third_frame" in sketch
    assert "--no_rollout" in sketch
    assert PRIMARY_FORMULA in sketch

    do_not_run = (output_dir / "do_not_run_runtime_in_stage4a65ar.md").read_text(encoding="utf-8")
    assert "Isaac startup in 6.5ar: `False`" in do_not_run
    assert "map_predict call in 6.5ar: `False`" in do_not_run
    assert "rollout in 6.5ar: `False`" in do_not_run

    note = (output_dir / "long_term_rl_gdpo_note.md").read_text(encoding="utf-8")
    assert "GDPO is future direction only" in note
    assert "no RL/GDPO/PPO/BC/IL in 6.5ar" in note

    readiness = load_json(output_dir / "repeat_safety_readiness_matrix.json")
    assert readiness["rollout_ready"] is False
    assert readiness["rollout_recommended"] is False
    assert readiness["alternate_start_bounded_repeat_recommended"] is True
    assert readiness["rl_gdpo_ppo_bc_il_recommended"] is False
    assert readiness["prediction_writeback_fusion_recommended"] is False
    assert readiness["over_cost_runtime_promotion_recommended"] is False
    return {"passed": True}


def test_hashes(output_dir: Path, stage4a65aq_dir: Path) -> dict[str, Any]:
    audit = load_json(output_dir / "input_hash_audit.json")
    assert audit["aq_observed_state_hashes_unchanged"] is True
    assert audit["aq_prediction_npz_hashes_unchanged"] is True
    assert audit["checkpoint_hash_unchanged"] is True
    for rel in (
        "observed_state_frame001.npy",
        "observed_state_frame002.npy",
        "frame001_map_predict/global_prediction_layer.npz",
        "frame002_map_predict/global_prediction_layer.npz",
    ):
        key = str(stage4a65aq_dir / rel)
        assert key in audit["entries"], key
        assert audit["entries"][key]["unchanged"] is True, key
    summary = load_json(output_dir / "stage4a65ar_alternate_start_diagnosis_summary.json")
    runtime = summary["runtime_in_stage4a65ar"]
    for key, value in runtime.items():
        assert value is False, key
    assert summary["rollout_ready"] is False
    assert summary["future_command_marked_do_not_run"] is True
    assert summary["future_command_executed"] is False
    assert summary["long_term_gdpo_future_only"] is True
    return {"passed": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage4a65aq_dir", type=Path, required=True)
    parser.add_argument("--expected_future_stage", default="4A-6.5as")
    parser.add_argument("--expect_no_runtime", action="store_true")
    parser.add_argument("--expect_future_command_do_not_run", action="store_true")
    args = parser.parse_args()

    results = {
        "required_outputs": test_required_outputs(args.output_dir),
        "hardware": test_hardware(args.output_dir),
        "sequence_and_safety": test_sequence_and_safety(args.output_dir),
        "pose_observed_prediction": test_pose_observed_prediction(args.output_dir),
        "tree_branch_outcome": test_tree_branch_outcome(args.output_dir),
        "future_command_and_notes": test_future_command_and_notes(args.output_dir, args.expected_future_stage),
        "hashes": test_hashes(args.output_dir, args.stage4a65aq_dir),
    }
    print(json.dumps({"passed": True, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
