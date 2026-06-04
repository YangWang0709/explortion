#!/usr/bin/env python3
"""Validate Stage 4A-6.5at start_corridor seed01 review/design outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
EXPECTED_START_POSITION = [0.0, -4.45, 1.2]
EXPECTED_START_YAW = 1.5707963267948966

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
    "start_pose_consistency.json",
    "start_pose_consistency.md",
    "action_pose_seed01_comparison.json",
    "action_pose_seed01_comparison.md",
    "start_to_action_geometry_seed01.json",
    "start_to_action_geometry_seed01.md",
    "frame1_seed0_seed1_tree_comparison.json",
    "frame1_seed0_seed1_tree_comparison.md",
    "frame2_seed0_seed1_tree_comparison.json",
    "frame2_seed0_seed1_tree_comparison.md",
    "branch_spatial_delta_table.csv",
    "branch_spatial_delta_table.json",
    "branch_spatial_delta_table.md",
    "branch_class_transition_summary.json",
    "branch_class_transition_summary.md",
    "lambda32_lambda48_seed01_agreement.json",
    "lambda32_lambda48_seed01_agreement.md",
    "observed_state_seed01_comparison.csv",
    "observed_state_seed01_comparison.json",
    "observed_state_seed01_comparison.md",
    "observed_transition_seed01_table.csv",
    "observed_transition_seed01_table.json",
    "observed_transition_seed01_table.md",
    "measured_only_update_review.json",
    "measured_only_update_review.md",
    "map_predict_seed01_stability.csv",
    "map_predict_seed01_stability.json",
    "map_predict_seed01_stability.md",
    "prediction_count_comparison.csv",
    "prediction_count_comparison.json",
    "prediction_count_comparison.md",
    "prediction_safety_review.json",
    "prediction_safety_review.md",
    "low_cost_artifact_seed01_review.json",
    "low_cost_artifact_seed01_review.md",
    "historical_prior_basin_seed01_review.json",
    "historical_prior_basin_seed01_review.md",
    "branch_health_seed01_review.json",
    "branch_health_seed01_review.md",
    "cost_dominance_seed01_review.json",
    "cost_dominance_seed01_review.md",
    "start_corridor_seed01_outcome_classification.json",
    "start_corridor_seed01_outcome_classification.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "recommended_next_faithful_step.md",
    "next_start_candidate_inventory.json",
    "next_start_candidate_inventory.md",
    "selected_next_start_design.json",
    "selected_next_start_design.md",
    "future_stage4a65au_command_sketch.md",
    "do_not_run_runtime_in_stage4a65at.md",
    "stage4a65at_start_corridor_seed01_review_summary.json",
    "stage4a65at_start_corridor_seed01_review_summary.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_PLOTS = [
    "start_corridor_seed01_action_topdown.png",
    "frame1_seed0_seed1_tree_topdown.png",
    "frame2_seed0_seed1_tree_topdown.png",
    "branch_spatial_delta_seed01.png",
    "observed_delta_seed01_bar.png",
    "prediction_count_seed01_bar.png",
    "lambda32_lambda48_seed01_comparison.png",
    "repeat_safety_readiness_matrix.png",
    "next_start_design_topdown.png",
    "next_stage_decision_flowchart.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
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
    assert missing["missing_required_outputs_before_report_write"] == [], missing[
        "missing_required_outputs_before_report_write"
    ]
    assert missing["missing_plots_without_skip_reason"] == [], missing["missing_plots_without_skip_reason"]
    assert missing["prohibited_artifacts_found"] == [], missing["prohibited_artifacts_found"]
    forbidden = load_json(output_dir / "forbidden_artifact_scan.json")
    assert forbidden["clean"] is True
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        matches = sorted(output_dir.glob(pattern))
        assert matches == [], f"forbidden output pattern {pattern}: {matches}"
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


def test_sequence_and_pose(output_dir: Path) -> dict[str, Any]:
    seq = load_json(output_dir / "sequence_safety_reverification.json")
    for key in ("stage4a65aq", "stage4a65as"):
        item = seq[key]
        assert item["isaac_startup_count_clean_run"] == 1
        assert item["frames_captured"] == 2
        assert item["map_predict_calls"] == 2
        assert item["selected_action_execution_count"] == 1
        assert item["second_action"] is False
        assert item["third_frame"] is False
        assert item["rollout"] is False
        assert item["sequence_clean"] is True
    assert seq["start_variant"] == "start_corridor"
    assert seq["stage4a65aq_start_variant"] == "start_corridor"
    assert seq["stage4a65as_start_variant"] == "start_corridor"
    assert_close_list(seq["start_pose"], EXPECTED_START_POSITION)
    assert math.isclose(float(seq["start_yaw"]), EXPECTED_START_YAW, abs_tol=1.0e-9)
    assert seq["tree_seed_reference"] == 0
    assert seq["tree_seed_repeat"] == 1
    assert seq["formula_contract"] == PRIMARY_FORMULA
    assert seq["formula_is_not_over_cost"] is True
    for key, value in seq["runtime_in_stage4a65at"].items():
        assert value is False, key

    pose = load_json(output_dir / "start_pose_consistency.json")
    assert pose["same_start_pose"] is True
    assert pose["same_start_yaw"] is True
    assert pose["matches_expected_pose"] is True
    assert pose["matches_expected_yaw"] is True
    assert pose["tree_seed_only_runtime_variable_between_aq_as"] is True
    return {"passed": True}


def test_prediction_no_rollout_and_hashes(output_dir: Path) -> dict[str, Any]:
    pred = load_json(output_dir / "prediction_safety_reverification.json")
    assert pred["prediction_safety_clean"] is True
    assert pred["no_prediction_writeback_or_fusion"] is True
    assert pred["no_prediction_motion_safety_use"] is True
    assert pred["no_target_ground_truth_future_observed_scoring"] is True

    no_rollout = load_json(output_dir / "no_rollout_reverification.json")
    assert no_rollout["rollout"] is False
    assert no_rollout["open_ended_loop"] is False
    assert no_rollout["frame003_captured"] is False
    assert no_rollout["second_action_executed"] is False
    assert no_rollout["rollout_ready"] is False

    hashes = load_json(output_dir / "input_hash_audit.json")
    assert hashes["stage4a65aq_observed_state_hashes_unchanged"] is True
    assert hashes["stage4a65aq_prediction_npz_hashes_unchanged"] is True
    assert hashes["stage4a65as_observed_state_hashes_unchanged"] is True
    assert hashes["stage4a65as_prediction_npz_hashes_unchanged"] is True
    assert hashes["checkpoint_unchanged"] is True
    return {"passed": True}


def test_seed01_comparison(output_dir: Path) -> dict[str, Any]:
    frame1 = load_json(output_dir / "frame1_seed0_seed1_tree_comparison.json")
    frame2 = load_json(output_dir / "frame2_seed0_seed1_tree_comparison.json")
    assert frame1["aq_lambda48"]["selected_child_id"] == "n0001"
    assert frame1["aq_lambda48"]["best_descendant_id"] == "n0104"
    assert frame1["as_lambda48"]["selected_child_id"] == "n0001"
    assert frame1["as_lambda48"]["best_descendant_id"] == "n0135"
    assert frame1["branch_class_transition"] == "same_as_measured -> distinct_nonmeasured_branch"
    assert math.isclose(float(frame1["spatial_selected_child_delta_m"]), 0.2, abs_tol=1.0e-12)
    assert math.isclose(float(frame1["spatial_best_descendant_delta_m"]), 1.6881943016134136, abs_tol=1.0e-12)

    assert frame2["aq_lambda48"]["selected_child_id"] == "n0001"
    assert frame2["aq_lambda48"]["best_descendant_id"] == "n0127"
    assert frame2["as_lambda48"]["selected_child_id"] == "n0008"
    assert frame2["as_lambda48"]["best_descendant_id"] == "n0137"
    assert frame2["branch_class_transition"] == "same_as_measured -> distinct_nonmeasured_branch"
    assert math.isclose(float(frame2["spatial_selected_child_delta_m"]), 0.458257569495584, abs_tol=1.0e-12)
    assert math.isclose(float(frame2["spatial_best_descendant_delta_m"]), 2.4103941586387903, abs_tol=1.0e-12)

    action = load_json(output_dir / "action_pose_seed01_comparison.json")
    assert math.isclose(float(action["action_pose_delta_m"]), 0.20000000000000018, abs_tol=1.0e-12)
    assert math.isclose(float(action["action_yaw_delta_rad"]), 2.7504672066207645, abs_tol=1.0e-12)
    assert action["action_difference_suggests_safety_concern"] is False

    observed = load_json(output_dir / "observed_state_seed01_comparison.json")
    assert math.isclose(
        float(observed["observed_ratio_delta_difference_as_minus_aq"]),
        -0.005733796296296297,
        abs_tol=1.0e-15,
    )
    assert observed["observed_state_remained_measured_only"] is True
    assert observed["prediction_writeback_occurred"] is False

    maps = load_json(output_dir / "map_predict_seed01_stability.json")
    assert maps["frame1_exact_count_match"] is True
    assert maps["frame2_occ_free_delta_as_minus_aq"] == -1891
    assert maps["no_explosion_or_collapse"] is True
    assert maps["both_code_consistent_v1"] is True

    low = load_json(output_dir / "low_cost_artifact_seed01_review.json")
    prior = load_json(output_dir / "historical_prior_basin_seed01_review.json")
    assert low["low_cost_artifact_any"] is False
    assert prior["historical_prior_basin_any"] is False
    return {"passed": True}


def test_outcome_and_future_design(output_dir: Path, expected_future_stage: str) -> dict[str, Any]:
    outcome = load_json(output_dir / "start_corridor_seed01_outcome_classification.json")
    assert outcome["combined_outcome"] in (
        "start_corridor_seed_sensitive_but_clean",
        "healthy_distinct_seed1_after_conservative_seed0",
    )
    assert outcome["runtime_safety_regression"] is False
    assert outcome["artifact_or_prior_basin_regression"] is False
    assert outcome["start_corridor_tree_seed2_required_immediately"] is False
    assert outcome["rollout_ready"] is False
    assert outcome["direct_rollout_recommended"] is False

    design = load_json(output_dir / "selected_next_start_design.json")
    assert design["future_stage"] == expected_future_stage
    assert design["design_only_in_stage4a65at"] is True
    assert design["was_executed_in_stage4a65at"] is False
    assert design["start_variant"] == "start_room_b"
    assert design["pose_found"] is True
    assert_close_list(design["position"], [2.75, -2.55, 1.2])
    assert math.isclose(float(design["yaw_rad"]), 2.7052603405912112, abs_tol=1.0e-12)
    assert design["future_tree_seed"] == 0
    assert design["formula"] == PRIMARY_FORMULA
    assert design["runtime_constraints"]["max_frames"] == 2
    assert design["runtime_constraints"]["exactly_two_map_predict_calls_if_action_executes"] is True
    assert design["runtime_constraints"]["exactly_one_selected_action_if_gates_pass"] is True
    assert design["runtime_constraints"]["no_second_action"] is True
    assert design["runtime_constraints"]["no_third_frame"] is True
    assert design["runtime_constraints"]["no_rollout"] is True
    assert design["runtime_constraints"]["max_workers"] == 32
    assert design["rl_gdpo_ppo_bc_il"] is False

    command = (output_dir / "future_stage4a65au_command_sketch.md").read_text(encoding="utf-8")
    assert command.startswith("DO NOT RUN IN STAGE 4A-6.5at.\nThis is a future Stage 4A-6.5au command sketch only.")
    for text in (
        "--max_workers 32",
        "--max_frames 2",
        "--no_second_action",
        "--no_third_frame",
        "--no_rollout",
        "gain_exp / cost + 48 * minmax(source_occ_free)",
    ):
        assert text in command, text
    note = (output_dir / "long_term_rl_gdpo_note.md").read_text(encoding="utf-8")
    assert "GDPO is future direction only" in note
    assert "No RL/GDPO/PPO/BC/IL in 6.5at" in note
    return {"passed": True}


def test_no_runtime_runner_created(expect_no_runtime: bool) -> dict[str, Any]:
    if expect_no_runtime:
        matches = sorted((WORKSPACE / "sim_explorer").glob("run_stage4a65au*.py"))
        assert matches == [], f"6.5at must not create 6.5au runtime runner: {matches}"
    return {"passed": True}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage4a65aq_dir", type=Path, required=True)
    parser.add_argument("--stage4a65as_dir", type=Path, required=True)
    parser.add_argument("--expected_future_stage", default="4A-6.5au")
    parser.add_argument("--expect_no_runtime", action="store_true")
    parser.add_argument("--expect_future_command_do_not_run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    results = {
        "required_outputs": test_required_outputs(args.output_dir),
        "hardware": test_hardware(args.output_dir),
        "sequence_and_pose": test_sequence_and_pose(args.output_dir),
        "prediction_no_rollout_and_hashes": test_prediction_no_rollout_and_hashes(args.output_dir),
        "seed01_comparison": test_seed01_comparison(args.output_dir),
        "outcome_and_future_design": test_outcome_and_future_design(args.output_dir, args.expected_future_stage),
        "no_runtime_runner_created": test_no_runtime_runner_created(args.expect_no_runtime),
    }
    print(json.dumps({"all_passed": True, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
