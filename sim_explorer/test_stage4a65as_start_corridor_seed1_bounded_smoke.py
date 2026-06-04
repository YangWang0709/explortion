#!/usr/bin/env python3
"""Validate Stage 4A-6.5as start_corridor tree_seed=1 bounded smoke outputs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"

COMMON_REQUIRED = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_reference_manifest.json",
    "loaded_reference_manifest.md",
    "loaded_alternate_start_manifest.json",
    "loaded_alternate_start_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "runtime_setup_summary.json",
    "runtime_setup_summary.md",
    "repeat_variant_definition.json",
    "repeat_variant_definition.md",
    "alternate_start_definition.json",
    "alternate_start_definition.md",
    "formula_definition.json",
    "formula_definition.md",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "prediction_safety_report.json",
    "prediction_safety_report.md",
    "hash_checks.json",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "stage4a65as_start_corridor_seed1_summary.json",
    "stage4a65as_start_corridor_seed1_summary.md",
    "recommended_next_faithful_step.md",
    "long_term_rl_gdpo_note.md",
    "comparison_to_stage4a65aq.json",
    "comparison_to_stage4a65aq.md",
    "comparison_to_stage4a65ar_design.json",
    "comparison_to_stage4a65ar_design.md",
    "start_corridor_seed0_seed1_comparison.json",
    "start_corridor_seed0_seed1_comparison.md",
    "repeat_outcome_classification.json",
    "repeat_outcome_classification.md",
    "lambda32_vs_lambda48_start_corridor_seed1.json",
    "lambda32_vs_lambda48_start_corridor_seed1.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "delegated_stage4a65ak_runtime_command.json",
    "delegated_stage4a65ak_runtime_result.json",
]

FRAME1_REQUIRED = [
    "frame001_capture_summary.json",
    "frame001_capture_summary.md",
    "frame001_rgb.png",
    "frame001_depth.npy",
    "frame001_depth.png",
    "frame001_pose.json",
    "frame001_camera_info.json",
    "observed_state_frame001.npy",
    "observed_state_update_frame001.json",
    "observed_state_update_frame001.md",
    "frame001_map_predict/global_prediction_layer.npz",
    "frame001_map_predict/prediction_alignment_summary.json",
    "map_predict_frame001_summary.json",
    "map_predict_frame001_summary.md",
    "frame001_measured_shadow_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.md",
    "frame001_lambda32_shadow_tree_decision.json",
    "frame001_branch_classification.json",
    "frame001_branch_classification.md",
    "frame001_low_cost_artifact_diagnosis.json",
    "frame001_low_cost_artifact_diagnosis.md",
    "pre_action_safety_gate_report.json",
    "pre_action_safety_gate_report.md",
]

FRAME1_PLOTS = [
    "frame001_observed_topdown.png",
    "frame001_prediction_overlay_topdown.png",
    "frame001_measured_vs_lambda48_tree_topdown.png",
    "frame001_lambda48_selected_branch_topdown.png",
    "value_components_frame001_lambda48.png",
    "low_cost_artifact_two_frame.png",
    "comparison_to_stage4a65aq_topdown.png",
    "start_corridor_seed0_seed1_topdown.png",
    "repeat_stability_summary.png",
]

FRAME2_REQUIRED = [
    "action_execution_report.json",
    "action_execution_report.md",
    "frame002_capture_summary.json",
    "frame002_capture_summary.md",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_depth.png",
    "frame002_pose.json",
    "frame002_camera_info.json",
    "observed_state_frame002.npy",
    "observed_state_update_frame002.json",
    "observed_state_update_frame002.md",
    "frame002_map_predict/global_prediction_layer.npz",
    "frame002_map_predict/prediction_alignment_summary.json",
    "map_predict_frame002_summary.json",
    "map_predict_frame002_summary.md",
    "frame002_measured_shadow_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.md",
    "frame002_lambda32_shadow_tree_decision.json",
    "frame002_branch_classification.json",
    "frame002_branch_classification.md",
    "frame002_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.md",
    "two_frame_decision_comparison.json",
    "two_frame_decision_comparison.md",
    "observed_state_delta_summary.json",
    "observed_state_delta_summary.md",
    "map_predict_two_frame_stability.json",
    "map_predict_two_frame_stability.md",
]

FRAME2_PLOTS = [
    "executed_action_topdown.png",
    "frame002_observed_topdown.png",
    "frame002_prediction_overlay_topdown.png",
    "frame002_measured_vs_lambda48_tree_topdown.png",
    "two_frame_path_topdown.png",
    "value_components_frame002_lambda48.png",
    "observed_state_delta_topdown.png",
]

PROHIBITED_PATTERNS = [
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "episode_manifest*",
    "frame003*",
    "action002*",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_expected_position(raw: str) -> np.ndarray:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    assert len(values) == 3, raw
    return np.asarray(values, dtype=np.float64)


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"


def assert_png(path: Path) -> None:
    assert_file(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not PNG: {path}"
    image = Image.open(path)
    assert image.size[0] > 0 and image.size[1] > 0, f"empty PNG dimensions: {path}"


def assert_required(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    for name in COMMON_REQUIRED + FRAME1_REQUIRED:
        assert_file(output_dir / name)
    for name in FRAME1_PLOTS:
        assert_png(output_dir / name)

    action_executed = int(summary["runtime_setup"]["selected_action_execution_count"]) == 1
    if action_executed:
        for name in FRAME2_REQUIRED:
            assert_file(output_dir / name)
        for name in FRAME2_PLOTS:
            assert_png(output_dir / name)
    else:
        assert_file(output_dir / "action_blocked_report.json")
        assert not (output_dir / "frame002_depth.npy").exists(), "Frame 2 should not exist when action is blocked"

    prefixes = ("frame001", "frame002") if action_executed else ("frame001",)
    for prefix in prefixes:
        depth = np.load(output_dir / f"{prefix}_depth.npy")
        assert depth.ndim == 2, f"{prefix} depth not HxW: {depth.shape}"
        assert np.count_nonzero(np.isfinite(depth) & (depth > 0.0)) > 0, f"{prefix} depth has no positive finite values"
        observed = np.load(output_dir / f"observed_state_{prefix}.npy")
        assert tuple(observed.shape) == (120, 120, 30), f"{prefix} observed shape {observed.shape}"
    return {"passed": True, "action_executed": action_executed}


def test_variant_and_hardware(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    variant = load_json(output_dir / "repeat_variant_definition.json")
    alternate = load_json(output_dir / "alternate_start_definition.json")
    hardware = load_json(output_dir / "hardware_utilization_report.json")
    reference = load_json(output_dir / "loaded_reference_manifest.json")
    ar_design = load_json(output_dir / "comparison_to_stage4a65ar_design.json")

    assert variant["repeat_variant"] == args.expected_repeat_variant
    assert variant["start_variant"] == args.expected_start_variant
    assert int(variant["current_tree_seed"]) == int(args.expected_tree_seed)
    assert int(variant["reference_tree_seed"]) == int(args.expected_reference_tree_seed)
    assert variant["only_intended_runtime_change"] == "tree_seed_vs_stage4a65aq"
    assert variant["tree_seed_changed_vs_reference"] is True
    np.testing.assert_allclose(np.asarray(variant["position"], dtype=np.float64), parse_expected_position(args.expected_position), atol=1.0e-9)
    assert abs(float(variant["yaw"]) - float(args.expected_yaw)) <= 1.0e-9
    assert alternate["start_variant"] == args.expected_start_variant
    assert alternate["matches_metadata"] is True
    assert alternate["matches_stage4a65ar_design"] is True
    assert reference["loaded_stage4a65aq_summary"] is True
    assert reference["loaded_stage4a65ar_design"] is True
    assert ar_design["design_matched"] is True
    assert summary["repeat_variant"]["repeat_variant"] == args.expected_repeat_variant
    assert int(summary["repeat_variant"]["current_tree_seed"]) == int(args.expected_tree_seed)

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
    return {"passed": True}


def test_contract_and_formula(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    setup = summary["runtime_setup"]
    formula = load_json(output_dir / "formula_definition.json")
    gate = load_json(output_dir / "pre_action_safety_gate_report.json")
    frame1_pred = load_json(output_dir / "map_predict_frame001_summary.json")
    frame1_l48 = load_json(output_dir / "frame001_lambda48_primary_tree_decision.json")["decision"]

    assert int(setup["isaac_startup_count"]) == 1
    assert int(setup["frames_captured"]) in (1, int(args.expected_max_frames))
    assert int(setup["map_predict_calls"]) in (1, int(args.expected_max_map_predict_calls))
    assert int(setup["selected_action_execution_count"]) in (0, int(args.expected_max_selected_action_executions))
    assert setup["second_action"] is False
    assert setup["third_frame"] is False
    assert setup["rollout"] is False
    assert formula["primary_formula"] == PRIMARY_FORMULA
    assert frame1_l48["formula"] == PRIMARY_FORMULA
    assert "(gain_exp + 48 * source_occ_free) / cost" in formula["prohibited_runtime_primary_formulas"]
    assert formula["over_cost_runtime_primary_executed"] is False
    assert frame1_pred["map_predict_call_count_so_far"] == 1
    assert frame1_pred["prediction_layer_shape_aligned_to_observed_state"] is True
    assert isinstance(gate["hard_gates_passed"], bool)

    if int(setup["selected_action_execution_count"]) == 1:
        frame2_pred = load_json(output_dir / "map_predict_frame002_summary.json")
        frame2_l48 = load_json(output_dir / "frame002_lambda48_diagnostic_tree_decision.json")["decision"]
        action = load_json(output_dir / "action_execution_report.json")
        pose2 = load_json(output_dir / "frame002_pose.json")
        assert frame2_pred["map_predict_call_count_so_far"] == 2
        assert frame2_pred["prediction_layer_shape_aligned_to_observed_state"] is True
        assert frame2_l48["formula"] == PRIMARY_FORMULA
        assert int(action["action_execution_count"]) == 1
        np.testing.assert_allclose(
            np.asarray(action["executed_pose"]["position"], dtype=np.float64),
            np.asarray(pose2["position"], dtype=np.float64),
            atol=1.0e-6,
        )
        assert summary["readiness"]["two_frame_runtime_executed"] is True
    else:
        assert summary["readiness"]["two_frame_runtime_executed"] is False
    return {"passed": True}


def test_safety_hashes_and_no_rollout(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    safety = load_json(output_dir / "prediction_safety_report.json")
    hashes = load_json(output_dir / "hash_checks.json")
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    missing = load_json(output_dir / "missing_fields_report.json")
    note = (output_dir / "long_term_rl_gdpo_note.md").read_text(encoding="utf-8")

    assert safety["prediction_read_only"] is True
    assert safety["prediction_information_gain_only"] is True
    assert safety["all_motion_safety_uses_false"] is True
    for key in (
        "prediction_written_to_observed_state",
        "prediction_fused_into_observed_state",
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_ray_blocking",
        "prediction_used_for_candidate_sampling",
        "prediction_used_for_edge_validity",
        "target_lr_target_hr_ground_truth_used_for_planning_scoring",
        "future_observed_used_for_planning_scoring",
    ):
        assert safety[key] is False, f"safety leak: {key}"

    assert hashes["checkpoint"]["unchanged"] is True
    assert sha256_file(hashes["checkpoint"]["path"]) == hashes["checkpoint"]["sha256_after"]
    for item in hashes["observed_states"].values():
        if item is not None:
            assert item["unchanged_after_update"] is True
            assert sha256_file(item["path"]) == item["sha256_after_prediction_and_tree"]
    for item in hashes["prediction_npzs"].values():
        if item is not None:
            assert item["unchanged_after_creation"] is True
            assert sha256_file(item["path"]) == item["sha256_after_tree"]
    for item in hashes["reference_inputs"].values():
        assert item["unchanged"] is True, f"reference input changed: {item}"
    assert hashes["stage4a65aq_reference_inputs"], "missing aq reference hash inputs"
    assert hashes["stage4a65ar_reference_inputs"], "missing ar reference hash inputs"

    for key, expected in (
        ("rollout", False),
        ("open_ended_loop", False),
        ("frame003_captured", False),
        ("second_action_executed", False),
        ("coverage_improvement_claim", False),
    ):
        assert no_rollout[key] is expected
    assert summary["readiness"]["rollout_ready"] is False
    assert summary["coverage_improvement_claim"] is False
    assert missing["missing_required_files"] == [], missing["missing_required_files"]
    assert missing["prohibited_artifacts_found"] == [], missing["prohibited_artifacts_found"]
    assert "does not implement RL/GDPO/PPO/BC/IL" in note
    return {"passed": True}


def test_repeat_outputs(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    comparison = load_json(output_dir / "comparison_to_stage4a65aq.json")
    ar_design = load_json(output_dir / "comparison_to_stage4a65ar_design.json")
    repeat = load_json(output_dir / "repeat_outcome_classification.json")
    lambda_repeat = load_json(output_dir / "lambda32_vs_lambda48_start_corridor_seed1.json")
    readiness = load_json(output_dir / "repeat_safety_readiness_matrix.json")

    assert ar_design["design_matched"] is True
    assert comparison["reference_stage"] == "Stage 4A-6.5aq"
    assert int(comparison["reference_tree_seed"]) == 0
    assert int(comparison["current_tree_seed"]) == 1
    assert comparison["frame001"]["available"] is True
    assert "action_pose_delta_m" in comparison
    assert "density_ratio_delta_vs_stage4a65aq" in comparison
    assert repeat["repeat_outcome"] in {
        "exact_repeat_match_to_aq",
        "spatially_consistent_healthy_repeat",
        "clean_same_as_measured",
        "clean_distinct_nonmeasured",
        "start_corridor_seed_sensitive_but_clean",
        "artifact_or_prior_basin_regression",
        "action_blocked",
        "runtime_failure",
    }
    if int(summary["runtime_setup"]["selected_action_execution_count"]) == 1:
        assert comparison["frame002"]["available"] is True
        assert repeat["low_cost_artifact_any_frame"] is False
        assert repeat["historical_prior_basin_any_frame"] is False
        assert repeat["prediction_safety_clean"] is True
        assert lambda_repeat["frame001"]["lambda32_available"] is True
        assert lambda_repeat["frame002"]["lambda32_available"] is True
        assert summary["observed_state_delta_summary"]["newly_observed"] > 0
        assert summary["map_predict_two_frame_stability"]["no_explosion_or_collapse"] is True
    assert all(bool(row["passed"]) for row in readiness["rows"]), readiness
    return {"passed": True, "repeat_outcome": repeat["repeat_outcome"]}


def test_absence_of_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    for pattern in PROHIBITED_PATTERNS:
        found = sorted(glob.glob(str(output_dir / "**" / pattern), recursive=True))
        assert not found, f"forbidden outputs for {pattern}: {found[:5]}"
    for prediction_dir in (output_dir / "frame001_map_predict", output_dir / "frame002_map_predict"):
        local = prediction_dir / "local_prediction.npz"
        if local.is_file():
            with np.load(local, allow_pickle=False) as data:
                assert "class_prob" not in data.files, f"dense class_prob unexpectedly saved: {local}"
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    summary = load_json(output_dir / "stage4a65as_start_corridor_seed1_summary.json")
    results = {
        "required_outputs": assert_required(output_dir, summary),
        "variant_and_hardware": test_variant_and_hardware(output_dir, args, summary),
        "contract_and_formula": test_contract_and_formula(output_dir, args, summary),
        "safety_hashes_and_no_rollout": test_safety_hashes_and_no_rollout(output_dir, summary),
        "repeat_outputs": test_repeat_outputs(output_dir, summary),
        "absence_of_forbidden_outputs": test_absence_of_forbidden_outputs(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stage4a65aq_dir", required=True)
    parser.add_argument("--stage4a65ar_dir", required=True)
    parser.add_argument("--expected_repeat_variant", default="alternate_start_corridor_tree_seed1")
    parser.add_argument("--expected_start_variant", default="start_corridor")
    parser.add_argument("--expected_position", default="0.0,-4.45,1.2")
    parser.add_argument("--expected_yaw", type=float, default=1.5707963267948966)
    parser.add_argument("--expected_tree_seed", type=int, default=1)
    parser.add_argument("--expected_reference_tree_seed", type=int, default=0)
    parser.add_argument("--expected_max_frames", type=int, default=2)
    parser.add_argument("--expected_max_map_predict_calls", type=int, default=2)
    parser.add_argument("--expected_max_selected_action_executions", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
