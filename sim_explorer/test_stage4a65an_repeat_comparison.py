#!/usr/bin/env python3
"""Validate Stage 4A-6.5an repeat comparison/design outputs."""

from __future__ import annotations

import argparse
import json
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
    "frame1_seed0_seed1_comparison.json",
    "frame1_seed0_seed1_comparison.md",
    "frame2_seed0_seed1_comparison.json",
    "frame2_seed0_seed1_comparison.md",
    "branch_spatial_delta_table.csv",
    "branch_spatial_delta_table.json",
    "branch_spatial_delta_table.md",
    "branch_class_transition_summary.json",
    "branch_class_transition_summary.md",
    "lambda32_lambda48_agreement_comparison.json",
    "lambda32_lambda48_agreement_comparison.md",
    "observed_state_repeat_comparison.json",
    "observed_state_repeat_comparison.md",
    "action_effect_repeat_comparison.json",
    "action_effect_repeat_comparison.md",
    "observed_delta_difference_table.csv",
    "observed_delta_difference_table.json",
    "observed_delta_difference_table.md",
    "map_predict_repeat_stability_comparison.json",
    "map_predict_repeat_stability_comparison.md",
    "prediction_count_comparison.csv",
    "prediction_count_comparison.json",
    "prediction_count_comparison.md",
    "prediction_density_review.md",
    "low_cost_artifact_repeat_review.json",
    "low_cost_artifact_repeat_review.md",
    "historical_prior_basin_repeat_review.json",
    "historical_prior_basin_repeat_review.md",
    "branch_health_repeat_review.json",
    "branch_health_repeat_review.md",
    "cost_dominance_repeat_review.json",
    "cost_dominance_repeat_review.md",
    "repeat_outcome_classification.json",
    "repeat_outcome_classification.md",
    "next_repeat_decision.json",
    "next_repeat_decision.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "future_stage4a65ao_command_sketch.md",
    "do_not_run_runtime_in_stage4a65an.md",
    "stage4a65an_repeat_comparison_summary.json",
    "stage4a65an_repeat_comparison_summary.md",
    "recommended_next_faithful_step.md",
]

REQUIRED_PLOTS = [
    "frame1_seed0_seed1_topdown_comparison.png",
    "frame2_seed0_seed1_topdown_comparison.png",
    "action_pose_delta_topdown.png",
    "observed_delta_seed0_seed1_bar.png",
    "prediction_count_seed0_seed1_bar.png",
    "branch_class_transition.png",
    "repeat_safety_readiness_matrix.png",
    "next_repeat_decision_flowchart.png",
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
    "capture_rgb*.png",
    "capture_depth*.npy",
    "capture_depth*.png",
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
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not PNG: {path}"
        return
    assert_file(output_dir / f"{Path(name).stem}_skipped_reason.md")


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
    for pattern in PROHIBITED_PATTERNS:
        found = sorted(output_dir.glob(pattern))
        assert found == [], f"forbidden output pattern {pattern}: {found}"
    return {"passed": True, "required_files": len(REQUIRED_FILES)}


def test_hardware(output_dir: Path) -> dict[str, Any]:
    hardware = load_json(output_dir / "hardware_utilization_report.json")
    assert int(hardware["requested_max_workers"]) == 32
    assert int(hardware["actual_max_workers"]) == min(32, os.cpu_count() or 1)
    assert hardware["parallel_backend"] == "ThreadPoolExecutor"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert str(hardware[key]) == "1", f"{key} was {hardware[key]}"
    return {"passed": True}


def test_sequence(output_dir: Path) -> dict[str, Any]:
    seq = load_json(output_dir / "sequence_safety_reverification.json")
    for stage in ("stage4a65ak", "stage4a65am"):
        entry = seq[stage]
        assert entry["frames_captured"] == 2
        assert entry["map_predict_calls"] == 2
        assert entry["selected_action_execution_count"] == 1
        assert entry["second_action"] is False
        assert entry["third_frame"] is False
        assert entry["rollout"] is False
        assert entry["sequence_clean"] is True
    runtime = seq["stage4a65an_runtime"]
    assert runtime["isaac_startup"] is False
    assert runtime["rgb_depth_capture"] is False
    assert runtime["map_predict_call"] is False
    assert runtime["sscnet_inference"] is False
    assert runtime["selected_action_execution"] is False
    assert runtime["two_frame_runtime_execution"] is False
    assert runtime["rollout"] is False
    return {"passed": True}


def test_core_outputs(output_dir: Path) -> dict[str, Any]:
    frame1 = load_json(output_dir / "frame1_seed0_seed1_comparison.json")
    frame2 = load_json(output_dir / "frame2_seed0_seed1_comparison.json")
    observed = load_json(output_dir / "observed_state_repeat_comparison.json")
    map_pred = load_json(output_dir / "map_predict_repeat_stability_comparison.json")
    low_cost = load_json(output_dir / "low_cost_artifact_repeat_review.json")
    prior = load_json(output_dir / "historical_prior_basin_repeat_review.json")
    outcome = load_json(output_dir / "repeat_outcome_classification.json")
    next_decision = load_json(output_dir / "next_repeat_decision.json")

    assert frame1["selected_child_world_delta_m"] == 0.20000000000000018 or abs(frame1["selected_child_world_delta_m"] - 0.2) < 1e-9
    assert frame2["selected_child_world_delta_m"] > 1.0
    assert observed["suspicious_label_flips"] is False
    assert observed["measured_only_status_preserved"] is True
    assert map_pred["frame001_agreement"] is True
    assert map_pred["density_explosion_or_collapse"] is False
    assert map_pred["both_code_consistent_v1"] is True
    assert low_cost["any_low_cost_artifact"] is False
    assert prior["any_historical_prior_basin"] is False
    assert outcome["classification"] == "divergent_but_healthy"
    assert outcome["rollout_ready"] is False
    assert next_decision["chosen_next_repeat"] == "same_scene_start_tree_seed2_bounded_repeat_safety_smoke"
    assert next_decision["rollout_recommended"] is False
    assert next_decision["rl_ppo_bc_il_recommended"] is False
    assert next_decision["prediction_writeback_fusion_recommended"] is False
    assert next_decision["over_cost_runtime_promotion_recommended"] is False
    return {"passed": True}


def test_future_command(output_dir: Path) -> dict[str, Any]:
    sketch = (output_dir / "future_stage4a65ao_command_sketch.md").read_text(encoding="utf-8")
    lines = sketch.splitlines()
    assert lines[0] == "DO NOT RUN IN STAGE 4A-6.5an."
    assert lines[1] == "This is a future Stage 4A-6.5ao command sketch only."
    assert "--max_workers 32" in sketch
    assert "--no_rollout" in sketch
    assert "--no_third_frame" in sketch
    assert "--no_second_action" in sketch
    assert "tree_seed 2" in sketch or "--tree_seed 2" in sketch
    return {"passed": True}


def test_hashes_and_safety(output_dir: Path, stage4a65ak_dir: Path, stage4a65am_dir: Path) -> dict[str, Any]:
    audit = load_json(output_dir / "input_hash_audit.json")
    assert audit["all_unchanged"] is True
    assert audit["checkpoint_unchanged"] is True
    for path in (
        stage4a65ak_dir / "observed_state_frame001.npy",
        stage4a65ak_dir / "observed_state_frame002.npy",
        stage4a65ak_dir / "frame001_map_predict/global_prediction_layer.npz",
        stage4a65ak_dir / "frame002_map_predict/global_prediction_layer.npz",
        stage4a65am_dir / "observed_state_frame001.npy",
        stage4a65am_dir / "observed_state_frame002.npy",
        stage4a65am_dir / "frame001_map_predict/global_prediction_layer.npz",
        stage4a65am_dir / "frame002_map_predict/global_prediction_layer.npz",
    ):
        assert audit["unchanged"][str(path)] is True, f"hash changed: {path}"

    pred = load_json(output_dir / "prediction_safety_reverification.json")
    assert pred["prediction_safety_clean"] is True
    assert pred["no_prediction_writeback_or_fusion"] is True
    assert pred["no_prediction_motion_safety_use"] is True
    assert pred["no_target_ground_truth_future_observed_scoring"] is True

    summary = load_json(output_dir / "stage4a65an_repeat_comparison_summary.json")
    safety = summary["safety"]
    assert safety["isaac_startup"] is False
    assert safety["capture"] is False
    assert safety["map_predict"] is False
    assert safety["selected_action_execution"] is False
    assert safety["two_frame_runtime_execution"] is False
    assert safety["rollout"] is False
    assert safety["training_rl_ppo_bc_il"] is False
    assert safety["checkpoint_modified"] is False
    assert safety["existing_observed_state_modified"] is False
    assert safety["prediction_npz_modified"] is False
    assert safety["prediction_writeback_fusion"] is False
    assert safety["prediction_used_for_collision_traversability"] is False
    assert safety["prediction_ray_blocking"] is False
    assert safety["target_ground_truth_future_observed_planning_scoring"] is False
    assert safety["external_source_modified_built"] is False
    assert safety["coverage_improvement_claim"] is False
    return {"passed": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage4a65ak_dir", type=Path, required=True)
    parser.add_argument("--stage4a65am_dir", type=Path, required=True)
    args = parser.parse_args()

    results = {
        "required_outputs": test_required_outputs(args.output_dir),
        "hardware": test_hardware(args.output_dir),
        "sequence": test_sequence(args.output_dir),
        "core_outputs": test_core_outputs(args.output_dir),
        "future_command": test_future_command(args.output_dir),
        "hashes_and_safety": test_hashes_and_safety(args.output_dir, args.stage4a65ak_dir, args.stage4a65am_dir),
    }
    print(json.dumps({"passed": True, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
