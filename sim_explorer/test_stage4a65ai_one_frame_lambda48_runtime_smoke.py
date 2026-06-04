#!/usr/bin/env python3
"""Validate Stage 4A-6.5ai one-frame lambda48 runtime smoke outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from offline_mini_rrt_tree import sha256_file


REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "runtime_capture_summary.json",
    "runtime_capture_summary.md",
    "observed_state_update_summary.json",
    "observed_state_update_summary.md",
    "map_predict_runtime_summary.json",
    "map_predict_runtime_summary.md",
    "formula_definition.json",
    "formula_definition.md",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "measured_shadow_tree_decision.json",
    "measured_shadow_tree_decision.md",
    "lambda48_primary_tree_decision.json",
    "lambda48_primary_tree_decision.md",
    "tree_decision_comparison.json",
    "tree_decision_comparison.md",
    "tree_decision_comparison.csv",
    "branch_classification.json",
    "branch_classification.md",
    "low_cost_artifact_diagnosis.json",
    "low_cost_artifact_diagnosis.md",
    "prediction_safety_report.json",
    "prediction_safety_report.md",
    "hash_checks.json",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "no_action_execution_report.json",
    "no_action_execution_report.md",
    "stage4a65ai_one_frame_lambda48_runtime_summary.json",
    "stage4a65ai_one_frame_lambda48_runtime_summary.md",
    "recommended_next_faithful_step.md",
    "capture_rgb_000.png",
    "capture_depth_000.npy",
    "capture_depth_000.png",
    "capture_pose_000.json",
    "capture_camera_info_000.json",
    "observed_state_frame000.npy",
    "map_predict/global_prediction_layer.npz",
    "map_predict/prediction_alignment_summary.json",
]

PLOT_FILES = [
    "observed_runtime_topdown.png",
    "prediction_overlay_topdown.png",
    "measured_vs_lambda48_tree_topdown.png",
    "lambda48_selected_branch_topdown.png",
    "tree_value_components_lambda48.png",
    "low_cost_artifact_runtime.png",
]

PROHIBITED_PATTERNS = [
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "episode_manifest*",
    "frame002*",
    "capture_*_001.*",
    "step_*.npz",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_png_or_reason(output_dir: Path, name: str) -> None:
    path = output_dir / name
    if path.is_file():
        data = path.read_bytes()
        assert len(data) > 8, f"empty PNG: {path}"
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
        return
    reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
    assert reason.is_file(), f"missing plot or skipped reason: {name}"


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_FILES:
        path = output_dir / name
        assert path.is_file(), f"missing required file: {path}"
        assert path.stat().st_size > 0, f"empty required file: {path}"
    for name in PLOT_FILES:
        assert_png_or_reason(output_dir, name)

    rgb = Image.open(output_dir / "capture_rgb_000.png")
    assert rgb.size[0] > 0 and rgb.size[1] > 0, "invalid RGB dimensions"
    depth = np.load(output_dir / "capture_depth_000.npy")
    assert depth.ndim == 2, f"depth should be HxW, got {depth.shape}"
    assert np.count_nonzero(np.isfinite(depth) & (depth > 0.0)) > 0, "depth has no finite positive values"
    observed = np.load(output_dir / "observed_state_frame000.npy")
    assert tuple(observed.shape) == (120, 120, 30), f"unexpected observed shape: {observed.shape}"
    return {"passed": True, "required_files": len(REQUIRED_FILES), "plots": len(PLOT_FILES)}


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
    expected_actual = min(32, os.cpu_count() or 1)
    assert int(hardware["actual_max_workers"]) == expected_actual
    assert hardware["parallel_backend"]
    return {
        "passed": True,
        "os_cpu_count": hardware["os_cpu_count"],
        "actual_max_workers": hardware["actual_max_workers"],
        "gpu": hardware.get("cuda_device_name"),
    }


def test_runtime_contract(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    capture = load_json(output_dir / "runtime_capture_summary.json")
    map_predict = load_json(output_dir / "map_predict_runtime_summary.json")
    summary = load_json(output_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.json")
    no_action = load_json(output_dir / "no_action_execution_report.json")

    assert int(capture["isaac_startup_count"]) == 1
    assert int(capture["frames_captured"]) == int(args.expected_frames)
    assert bool(capture["second_frame_captured"]) is False
    assert int(map_predict["map_predict_call_count"]) == int(args.expected_map_predict_calls)
    assert map_predict["alignment_convention"] == "code_consistent_v1"
    assert bool(map_predict["prediction_layer_shape_aligned_to_observed_state"]) is True
    assert int(no_action["selected_action_execution_count"]) == int(args.expected_selected_action_executions)
    assert bool(no_action["camera_pose_after_decision_equals_captured_frame_pose"]) is True
    assert summary["runtime_setup"]["rollout"] is False
    assert summary["runtime_setup"]["two_frame_runtime"] is False
    return {"passed": True}


def test_formula_and_decisions(output_dir: Path) -> dict[str, Any]:
    formula = load_json(output_dir / "formula_definition.json")
    measured = load_json(output_dir / "measured_shadow_tree_decision.json")
    lambda48 = load_json(output_dir / "lambda48_primary_tree_decision.json")
    comparison = load_json(output_dir / "tree_decision_comparison.json")
    branch = load_json(output_dir / "branch_classification.json")
    low_cost = load_json(output_dir / "low_cost_artifact_diagnosis.json")

    assert formula["primary_formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)"
    for bad in formula["prohibited_runtime_primary_formulas"]:
        assert bad != formula["primary_formula"], f"primary formula used prohibited expression: {bad}"
    assert bool(formula["over_cost_runtime_primary_executed"]) is False

    measured_decision = measured["decision"]
    lambda48_decision = lambda48["decision"]
    assert measured_decision["formula"] == "gain_exp / cost"
    assert "minmax(source_occ_free)" in lambda48_decision["formula"]
    assert lambda48_decision["formula"] != "(gain_exp + source_occ_free) / cost"
    for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value", "min_sc", "max_sc"):
        assert key in lambda48_decision, f"lambda48 decision missing {key}"
    assert comparison["branch_classification"] == branch["classification"]
    assert "low_cost_artifact" in low_cost
    assert (output_dir / "lambda32_shadow_tree_decision.json").is_file() or (
        output_dir / "lambda32_shadow_skipped.md"
    ).is_file()
    return {
        "passed": True,
        "lambda48_branch": branch["classification"],
        "low_cost_artifact": low_cost["low_cost_artifact"],
    }


def test_safety_and_hashes(output_dir: Path) -> dict[str, Any]:
    safety = load_json(output_dir / "prediction_safety_report.json")
    hashes = load_json(output_dir / "hash_checks.json")
    final_summary = load_json(output_dir / "stage4a65ai_one_frame_lambda48_runtime_summary.json")
    missing = load_json(output_dir / "missing_fields_report.json")

    assert safety["prediction_read_only"] is True
    false_safety_keys = [
        "prediction_written_to_observed_state",
        "prediction_fused_into_observed_state",
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_ray_blocking",
        "prediction_used_for_candidate_sampling",
        "prediction_used_for_edge_validity",
        "target_lr_target_hr_ground_truth_used_for_planning_scoring",
        "future_observed_used_for_planning_scoring",
    ]
    for key in false_safety_keys:
        assert bool(safety[key]) is False, f"safety flag should be false: {key}"

    assert hashes["checkpoint"]["unchanged"] is True
    assert hashes["new_observed_state"]["unchanged_after_update"] is True
    assert hashes["prediction_npz"]["unchanged_after_creation"] is True
    assert sha256_file(hashes["checkpoint"]["path"]) == hashes["checkpoint"]["sha256_after"]
    assert sha256_file(hashes["new_observed_state"]["path"]) == hashes["new_observed_state"][
        "sha256_after_prediction_and_tree"
    ]
    assert sha256_file(hashes["prediction_npz"]["path"]) == hashes["prediction_npz"]["sha256_after_tree"]

    summary_safety = final_summary["safety"]
    for key in (
        "selected_action_execution",
        "second_frame",
        "two_frame_runtime",
        "rollout",
        "open_ended_loop",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_npz_modified_after_creation",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "prediction_used_for_candidate_sampling_edge_validity",
        "target_ground_truth_future_observed_planning_scoring",
        "external_source_modified_built",
        "over_cost_runtime_primary",
        "coverage_improvement_claim",
    ):
        assert bool(summary_safety[key]) is False, f"summary safety flag should be false: {key}"

    assert final_summary["readiness"]["runtime_executed"] is True
    assert final_summary["readiness"]["rollout_ready"] is False
    assert final_summary["readiness"]["two_frame_runtime_ready"] is False
    assert missing["missing_required_files"] == [], f"missing required files: {missing['missing_required_files']}"
    assert missing["prohibited_artifacts_found"] == [], f"prohibited artifacts: {missing['prohibited_artifacts_found']}"
    return {"passed": True}


def test_absence_of_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    for pattern in PROHIBITED_PATTERNS:
        found = sorted(output_dir.rglob(pattern))
        assert not found, f"forbidden outputs for {pattern}: {[str(path) for path in found[:5]]}"
    local_prediction = output_dir / "map_predict/local_prediction.npz"
    if local_prediction.is_file():
        with np.load(local_prediction, allow_pickle=False) as data:
            assert "class_prob" not in data.files, "dense class_prob should not be saved by default"
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "hardware": test_hardware(output_dir),
        "runtime_contract": test_runtime_contract(output_dir, args),
        "formula_and_decisions": test_formula_and_decisions(output_dir),
        "safety_and_hashes": test_safety_and_hashes(output_dir),
        "absence_of_forbidden_outputs": test_absence_of_forbidden_outputs(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_frames", type=int, default=1)
    parser.add_argument("--expected_map_predict_calls", type=int, default=1)
    parser.add_argument("--expected_selected_action_executions", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
