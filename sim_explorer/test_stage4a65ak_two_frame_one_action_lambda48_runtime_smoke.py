#!/usr/bin/env python3
"""Validate Stage 4A-6.5ak two-frame one-action lambda48 runtime smoke outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from offline_mini_rrt_tree import sha256_file


COMMON_REQUIRED = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "runtime_setup_summary.json",
    "runtime_setup_summary.md",
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
    "stage4a65ak_two_frame_one_action_runtime_summary.json",
    "stage4a65ak_two_frame_one_action_runtime_summary.md",
    "recommended_next_faithful_step.md",
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
    "frame001_measured_shadow_tree_decision.md",
    "frame001_lambda48_primary_tree_decision.json",
    "frame001_lambda48_primary_tree_decision.md",
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
    "frame002_measured_shadow_tree_decision.md",
    "frame002_lambda48_diagnostic_tree_decision.json",
    "frame002_lambda48_diagnostic_tree_decision.md",
    "frame002_branch_classification.json",
    "frame002_branch_classification.md",
    "frame002_low_cost_artifact_diagnosis.json",
    "frame002_low_cost_artifact_diagnosis.md",
    "two_frame_decision_comparison.json",
    "two_frame_decision_comparison.md",
]

FRAME2_PLOTS = [
    "executed_action_topdown.png",
    "frame002_observed_topdown.png",
    "frame002_prediction_overlay_topdown.png",
    "frame002_measured_vs_lambda48_tree_topdown.png",
    "two_frame_path_topdown.png",
    "value_components_frame002_lambda48.png",
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


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"


def assert_png(path: Path) -> None:
    assert_file(path)
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"not PNG: {path}"


def test_required_outputs(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    for name in COMMON_REQUIRED:
        assert_file(output_dir / name)
    for name in FRAME1_PLOTS:
        assert_png(output_dir / name)
    assert_file(output_dir / "frame001_lambda32_shadow_tree_decision.json")

    action_executed = int(summary["runtime_setup"]["selected_action_execution_count"]) == 1
    if action_executed:
        for name in FRAME2_REQUIRED:
            assert_file(output_dir / name)
        for name in FRAME2_PLOTS:
            assert_png(output_dir / name)
        assert_file(output_dir / "frame002_lambda32_shadow_tree_decision.json")
    else:
        assert_file(output_dir / "action_blocked_report.json")
        assert not (output_dir / "frame002_depth.npy").exists(), "Frame 2 should not exist when action blocked"

    for prefix in ("frame001", "frame002") if action_executed else ("frame001",):
        depth = np.load(output_dir / f"{prefix}_depth.npy")
        assert depth.ndim == 2, f"{prefix} depth is not HxW: {depth.shape}"
        assert np.count_nonzero(np.isfinite(depth) & (depth > 0.0)) > 0, f"{prefix} depth has no positive values"
        rgb = Image.open(output_dir / f"{prefix}_rgb.png")
        assert rgb.size[0] > 0 and rgb.size[1] > 0, f"{prefix} RGB has invalid dimensions"
        observed = np.load(output_dir / f"observed_state_{prefix}.npy")
        assert tuple(observed.shape) == (120, 120, 30), f"{prefix} observed shape {observed.shape}"
    return {"passed": True, "action_executed": action_executed}


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
    return {"passed": True, "gpu": hardware.get("cuda_device_name")}


def test_contract_and_formula(output_dir: Path, args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    setup = summary["runtime_setup"]
    formula = load_json(output_dir / "formula_definition.json")
    gates = load_json(output_dir / "pre_action_safety_gate_report.json")
    frame1_pred = load_json(output_dir / "map_predict_frame001_summary.json")

    assert int(setup["isaac_startup_count"]) == 1
    assert int(setup["frames_captured"]) in (1, int(args.expected_max_frames))
    assert int(setup["map_predict_calls"]) in (1, int(args.expected_max_map_predict_calls))
    assert int(setup["selected_action_execution_count"]) in (0, int(args.expected_max_selected_action_executions))
    assert setup["second_action"] is False
    assert setup["third_frame"] is False
    assert setup["rollout"] is False

    assert formula["primary_formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)"
    assert "(gain_exp + 48 * source_occ_free) / cost" in formula["prohibited_runtime_primary_formulas"]
    assert formula["over_cost_runtime_primary_executed"] is False

    frame1 = load_json(output_dir / "frame001_lambda48_primary_tree_decision.json")["decision"]
    assert frame1["formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)"
    for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value", "min_sc", "max_sc"):
        assert key in frame1, f"frame1 lambda48 decision missing {key}"
    assert frame1_pred["prediction_layer_shape_aligned_to_observed_state"] is True
    assert isinstance(gates["hard_gates_passed"], bool)

    if int(setup["selected_action_execution_count"]) == 1:
        frame2 = load_json(output_dir / "frame002_lambda48_diagnostic_tree_decision.json")["decision"]
        assert frame2["formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)"
        assert load_json(output_dir / "map_predict_frame002_summary.json")[
            "prediction_layer_shape_aligned_to_observed_state"
        ] is True
        action = load_json(output_dir / "action_execution_report.json")
        pose2 = load_json(output_dir / "frame002_pose.json")
        assert int(action["action_execution_count"]) == 1
        np.testing.assert_allclose(
            np.asarray(action["executed_pose"]["position"], dtype=np.float64),
            np.asarray(pose2["position"], dtype=np.float64),
            atol=1.0e-6,
        )
    return {"passed": True}


def test_safety_and_hashes(output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    safety = load_json(output_dir / "prediction_safety_report.json")
    hashes = load_json(output_dir / "hash_checks.json")
    no_rollout = load_json(output_dir / "no_rollout_report.json")
    missing = load_json(output_dir / "missing_fields_report.json")

    assert safety["prediction_read_only"] is True
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
        if item is None:
            continue
        assert item["unchanged_after_update"] is True
        assert sha256_file(item["path"]) == item["sha256_after_prediction_and_tree"]
    for item in hashes["prediction_npzs"].values():
        if item is None:
            continue
        assert item["unchanged_after_creation"] is True
        assert sha256_file(item["path"]) == item["sha256_after_tree"]

    for key, expected in (
        ("rollout", False),
        ("open_ended_loop", False),
        ("frame003_captured", False),
        ("second_action_executed", False),
        ("coverage_improvement_claim", False),
    ):
        assert no_rollout[key] is expected
    assert summary["readiness"]["rollout_ready"] is False
    assert summary["safety"]["coverage_improvement_claim"] is False
    assert missing["missing_required_files"] == [], missing["missing_required_files"]
    assert missing["prohibited_artifacts_found"] == [], missing["prohibited_artifacts_found"]
    return {"passed": True}


def test_absence_of_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    for pattern in PROHIBITED_PATTERNS:
        found = sorted(output_dir.rglob(pattern))
        assert not found, f"forbidden outputs for {pattern}: {[str(path) for path in found[:5]]}"
    for prediction_dir in (output_dir / "frame001_map_predict", output_dir / "frame002_map_predict"):
        local = prediction_dir / "local_prediction.npz"
        if local.is_file():
            with np.load(local, allow_pickle=False) as data:
                assert "class_prob" not in data.files, f"dense class_prob unexpectedly saved: {local}"
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    summary = load_json(output_dir / "stage4a65ak_two_frame_one_action_runtime_summary.json")
    results = {
        "required_outputs": test_required_outputs(output_dir, summary),
        "hardware": test_hardware(output_dir),
        "contract_and_formula": test_contract_and_formula(output_dir, args, summary),
        "safety_and_hashes": test_safety_and_hashes(output_dir, summary),
        "absence_of_forbidden_outputs": test_absence_of_forbidden_outputs(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_max_frames", type=int, default=2)
    parser.add_argument("--expected_max_map_predict_calls", type=int, default=2)
    parser.add_argument("--expected_max_selected_action_executions", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
