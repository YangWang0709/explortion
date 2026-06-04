#!/usr/bin/env python3
"""Validate Stage 4A-6.5ap seed0/1/2 review/design outputs."""

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
    "forbidden_artifact_scan.json",
    "forbidden_artifact_scan.md",
    "seed012_decision_table.csv",
    "seed012_decision_table.json",
    "seed012_decision_table.md",
    "frame1_seed012_comparison.json",
    "frame1_seed012_comparison.md",
    "frame2_seed012_comparison.json",
    "frame2_seed012_comparison.md",
    "branch_spatial_delta_matrix.csv",
    "branch_spatial_delta_matrix.json",
    "branch_spatial_delta_matrix.md",
    "branch_class_transition_summary.json",
    "branch_class_transition_summary.md",
    "lambda32_lambda48_seed012_agreement.json",
    "lambda32_lambda48_seed012_agreement.md",
    "action_pose_seed012_comparison.json",
    "action_pose_seed012_comparison.md",
    "observed_state_seed012_comparison.csv",
    "observed_state_seed012_comparison.json",
    "observed_state_seed012_comparison.md",
    "map_predict_seed012_stability.csv",
    "map_predict_seed012_stability.json",
    "map_predict_seed012_stability.md",
    "low_cost_artifact_seed012_review.json",
    "low_cost_artifact_seed012_review.md",
    "historical_prior_basin_seed012_review.json",
    "historical_prior_basin_seed012_review.md",
    "branch_health_seed012_review.json",
    "branch_health_seed012_review.md",
    "cost_dominance_seed012_review.json",
    "cost_dominance_seed012_review.md",
    "seed012_outcome_classification.json",
    "seed012_outcome_classification.md",
    "repeat_safety_readiness_matrix.csv",
    "repeat_safety_readiness_matrix.json",
    "repeat_safety_readiness_matrix.md",
    "risk_register.json",
    "risk_register.md",
    "recommended_next_faithful_step.md",
    "alternate_start_candidate_inventory.json",
    "alternate_start_candidate_inventory.md",
    "selected_alternate_start_design.json",
    "selected_alternate_start_design.md",
    "future_stage4a65aq_command_sketch.md",
    "do_not_run_runtime_in_stage4a65ap.md",
    "stage4a65ap_seed012_repeat_review_summary.json",
    "stage4a65ap_seed012_repeat_review_summary.md",
    "long_term_rl_gdpo_note.md",
]

REQUIRED_PLOTS = [
    "frame1_seed012_topdown_comparison.png",
    "frame2_seed012_topdown_comparison.png",
    "action_pose_seed012_topdown.png",
    "observed_delta_seed012_bar.png",
    "prediction_count_seed012_bar.png",
    "branch_class_transition_seed012.png",
    "seed012_spatial_delta_heatmap.png",
    "repeat_safety_readiness_matrix.png",
    "alternate_start_design_topdown.png",
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
    "capture_rgb*.png",
    "capture_depth*.npy",
    "capture_depth*.png",
    "*replay_buffer*",
    "*policy_checkpoint*",
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
    forbidden = load_json(output_dir / "forbidden_artifact_scan.json")
    assert forbidden["clean"] is True
    for pattern in PROHIBITED_PATTERNS:
        found = sorted(output_dir.glob(pattern))
        assert found == [], f"forbidden output pattern {pattern}: {found}"
    return {"passed": True, "required_files": len(REQUIRED_FILES), "required_plots": len(REQUIRED_PLOTS)}


def test_hardware(output_dir: Path) -> dict[str, Any]:
    hardware = load_json(output_dir / "hardware_utilization_report.json")
    assert int(hardware["requested_max_workers"]) == 32
    assert int(hardware["actual_max_workers"]) == min(32, os.cpu_count() or 1)
    assert hardware["parallel_backend"] == "ThreadPoolExecutor"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        assert str(hardware[key]) == "1", f"{key} was {hardware[key]}"
    assert int(hardware["analysis_task_count"]) > 0
    return {"passed": True}


def test_sequence(output_dir: Path) -> dict[str, Any]:
    seq = load_json(output_dir / "sequence_safety_reverification.json")
    for stage in ("stage4a65ak", "stage4a65am", "stage4a65ao"):
        entry = seq[stage]
        assert entry["frames_captured"] == 2
        assert entry["map_predict_calls"] == 2
        assert entry["selected_action_execution_count"] == 1
        assert entry["second_action"] is False
        assert entry["third_frame"] is False
        assert entry["rollout"] is False
        assert entry["sequence_clean"] is True
    runtime = seq["stage4a65ap_runtime"]
    assert runtime["isaac_startup"] is False
    assert runtime["rgb_depth_capture"] is False
    assert runtime["map_predict_call"] is False
    assert runtime["sscnet_inference"] is False
    assert runtime["selected_action_execution"] is False
    assert runtime["two_frame_runtime_execution"] is False
    assert runtime["rollout"] is False
    return {"passed": True}


def test_core_outputs(output_dir: Path) -> dict[str, Any]:
    decision_table = load_json(output_dir / "seed012_decision_table.json")
    frame1 = load_json(output_dir / "frame1_seed012_comparison.json")
    frame2 = load_json(output_dir / "frame2_seed012_comparison.json")
    low_cost = load_json(output_dir / "low_cost_artifact_seed012_review.json")
    prior = load_json(output_dir / "historical_prior_basin_seed012_review.json")
    branch = load_json(output_dir / "branch_health_seed012_review.json")
    observed = load_json(output_dir / "observed_state_seed012_comparison.json")
    maps = load_json(output_dir / "map_predict_seed012_stability.json")
    outcome = load_json(output_dir / "seed012_outcome_classification.json")
    readiness = load_json(output_dir / "repeat_safety_readiness_matrix.json")
    design = load_json(output_dir / "selected_alternate_start_design.json")

    assert len(decision_table) == 6
    assert len(frame1["pairs"]) == 3
    assert len(frame2["pairs"]) == 3
    assert low_cost["any_low_cost_artifact"] is False
    assert prior["any_historical_prior_basin"] is False
    assert branch["all_branch_health_clean"] is True
    assert observed["all_positive_delta"] is True
    assert observed["all_measured_only"] is True
    assert maps["all_code_consistent_v1"] is True
    assert maps["density_explosion_or_collapse"] is False
    assert outcome["classification"] in {"spatially_consistent_healthy", "seed_sensitive_but_clean"}
    assert outcome["rollout_ready"] is False
    assert outcome["rollout_recommended"] is False
    assert outcome["alternate_start_bounded_repeat_recommended"] is True
    assert outcome["rl_gdpo_ppo_bc_il_recommended"] is False
    assert outcome["prediction_writeback_fusion_recommended"] is False
    assert outcome["over_cost_runtime_promotion_recommended"] is False
    assert readiness["rollout_ready"] is False
    assert readiness["rollout_recommended"] is False
    assert readiness["alternate_start_bounded_repeat_recommended"] is True
    assert design["chosen_alternate_start_variant"] == "start_corridor"
    assert design["future_tree_seed"] == 0
    assert design["runtime_constraints"]["no_rollout"] is True
    return {"passed": True}


def test_future_command_and_notes(output_dir: Path, expected_future_stage: str) -> dict[str, Any]:
    sketch = (output_dir / "future_stage4a65aq_command_sketch.md").read_text(encoding="utf-8")
    lines = sketch.splitlines()
    assert lines[0] == "DO NOT RUN IN STAGE 4A-6.5ap."
    assert lines[1] == f"This is a future Stage {expected_future_stage} command sketch only."
    assert "--max_workers 32" in sketch
    assert "--max_frames 2" in sketch
    assert "--no_second_action" in sketch
    assert "--no_third_frame" in sketch
    assert "--no_rollout" in sketch
    assert "gain_exp / cost + 48 * minmax(source_occ_free)" in sketch

    do_not_run = (output_dir / "do_not_run_runtime_in_stage4a65ap.md").read_text(encoding="utf-8")
    assert "Isaac startup in 6.5ap: `False`" in do_not_run
    assert "rollout in 6.5ap: `False`" in do_not_run

    note = (output_dir / "long_term_rl_gdpo_note.md").read_text(encoding="utf-8")
    assert "GDPO is future direction only" in note
    assert "no RL/GDPO/PPO/BC/IL in 6.5ap" in note
    return {"passed": True}


def test_hashes_and_safety(output_dir: Path, stage4a65ak_dir: Path, stage4a65am_dir: Path, stage4a65ao_dir: Path) -> dict[str, Any]:
    audit = load_json(output_dir / "input_hash_audit.json")
    assert audit["all_unchanged"] is True
    assert audit["checkpoint_unchanged"] is True
    for base in (stage4a65ak_dir, stage4a65am_dir, stage4a65ao_dir):
        for rel in (
            "observed_state_frame001.npy",
            "observed_state_frame002.npy",
            "frame001_map_predict/global_prediction_layer.npz",
            "frame002_map_predict/global_prediction_layer.npz",
        ):
            path = str(base / rel)
            assert audit["unchanged"][path] is True, f"hash changed: {path}"

    pred = load_json(output_dir / "prediction_safety_reverification.json")
    assert pred["prediction_safety_clean"] is True
    assert pred["no_prediction_writeback_or_fusion"] is True
    assert pred["no_prediction_motion_safety_use"] is True
    assert pred["no_target_ground_truth_future_observed_scoring"] is True

    summary = load_json(output_dir / "stage4a65ap_seed012_repeat_review_summary.json")
    runtime = summary["runtime_in_stage4a65ap"]
    assert runtime["isaac_startup"] is False
    assert runtime["rgb_depth_capture"] is False
    assert runtime["map_predict_call"] is False
    assert runtime["sscnet_inference"] is False
    assert runtime["selected_action_execution"] is False
    assert runtime["two_frame_runtime_execution"] is False
    assert runtime["rollout"] is False
    assert summary["future_command_marked_do_not_run"] is True
    assert summary["long_term_gdpo_future_only"] is True
    return {"passed": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage4a65ak_dir", type=Path, required=True)
    parser.add_argument("--stage4a65am_dir", type=Path, required=True)
    parser.add_argument("--stage4a65ao_dir", type=Path, required=True)
    parser.add_argument("--expected_future_stage", default="4A-6.5aq")
    parser.add_argument("--expect_no_runtime", action="store_true")
    parser.add_argument("--expect_future_command_do_not_run", action="store_true")
    args = parser.parse_args()

    results = {
        "required_outputs": test_required_outputs(args.output_dir),
        "hardware": test_hardware(args.output_dir),
        "sequence": test_sequence(args.output_dir),
        "core_outputs": test_core_outputs(args.output_dir),
        "future_command_and_notes": test_future_command_and_notes(args.output_dir, args.expected_future_stage),
        "hashes_and_safety": test_hashes_and_safety(args.output_dir, args.stage4a65ak_dir, args.stage4a65am_dir, args.stage4a65ao_dir),
    }
    print(json.dumps({"passed": True, "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
