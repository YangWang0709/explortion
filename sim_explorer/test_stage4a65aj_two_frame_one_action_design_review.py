#!/usr/bin/env python3
"""Validate Stage 4A-6.5aj two-frame one-action design review outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
CONTEXT_FILES = [
    WORKSPACE / ".project_context/CURRENT_STATE.md",
    WORKSPACE / ".project_context/CODEX_LOG.md",
    WORKSPACE / ".project_context/TODO.md",
]
PRIMARY_FORMULA = "gain_exp / cost + 48 * minmax(source_occ_free)"
BAD_FORMULAS = [
    "(gain_exp + 48 * source_occ_free) / cost",
    "(gain_exp + source_occ_free) / cost",
]

REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "stage4a65ai_result_review.json",
    "stage4a65ai_result_review.md",
    "future_runtime_smoke_design.json",
    "future_runtime_smoke_design.md",
    "future_two_frame_sequence_spec.json",
    "future_two_frame_sequence_spec.md",
    "future_lambda48_formula_spec.json",
    "future_lambda48_formula_spec.md",
    "future_source_protection_profile.json",
    "future_source_protection_profile.md",
    "future_pre_action_safety_gates.json",
    "future_pre_action_safety_gates.md",
    "future_frame2_stop_conditions.json",
    "future_frame2_stop_conditions.md",
    "future_required_outputs.json",
    "future_required_outputs.md",
    "future_test_requirements.json",
    "future_test_requirements.md",
    "future_stage4a65ak_command_sketch.md",
    "do_not_run_runtime_in_stage4a65aj.md",
    "hardware_policy_for_future_runtime.json",
    "hardware_policy_for_future_runtime.md",
    "prediction_safety_design.json",
    "prediction_safety_design.md",
    "risk_register.json",
    "risk_register.md",
    "rollout_blocker_statement.md",
    "stage4a65aj_design_review_summary.json",
    "stage4a65aj_design_review_summary.md",
    "recommended_next_faithful_step.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
]

OPTIONAL_DIAGRAMS = [
    "flowchart_two_frame_one_action.png",
    "safety_gate_flowchart.png",
    "runtime_timeline.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "episode_manifest*",
    "frame003*",
    "action002*",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "prediction_alignment_summary.json",
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "capture_rgb*.png",
    "capture_depth*.npy",
    "capture_depth*.png",
    "map_predict",
    "frame001_map_predict",
    "frame002_map_predict",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing required file: {path}"
    assert path.stat().st_size > 0, f"empty required file: {path}"


def assert_png_or_reason(output_dir: Path, name: str) -> None:
    path = output_dir / name
    if path.is_file():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), f"not a PNG: {path}"
        return
    reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
    assert reason.is_file(), f"missing optional diagram or skipped reason: {name}"


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_FILES:
        assert_file(output_dir / name)
    for name in OPTIONAL_DIAGRAMS:
        assert_png_or_reason(output_dir, name)
    missing = load_json(output_dir / "missing_fields_report.json")
    assert missing["missing_required_files"] == [], missing["missing_required_files"]
    assert missing["missing_optional_diagrams_without_reason"] == [], missing[
        "missing_optional_diagrams_without_reason"
    ]
    assert missing["prohibited_artifacts_found"] == [], missing["prohibited_artifacts_found"]
    return {"passed": True, "required_files": len(REQUIRED_FILES)}


def test_loaded_inputs(output_dir: Path) -> dict[str, Any]:
    manifest = load_json(output_dir / "loaded_context_manifest.json")
    assert manifest["confirmed"]["stage4a65ag_complete"] is True
    assert manifest["confirmed"]["stage4a65ah_complete"] is True
    assert manifest["confirmed"]["stage4a65ai_complete"] is True
    assert manifest["confirmed"]["stage4a65aj_runtime_executed"] is False
    for item in manifest["context_files"]:
        assert item["exists"] is True, item
        assert item["sha256"], item
    return {"passed": True}


def test_formula_and_design(output_dir: Path) -> dict[str, Any]:
    formula = load_json(output_dir / "future_lambda48_formula_spec.json")
    design = load_json(output_dir / "future_runtime_smoke_design.json")
    sequence = load_json(output_dir / "future_two_frame_sequence_spec.json")
    requirements = load_json(output_dir / "future_test_requirements.json")
    command = (output_dir / "future_stage4a65ak_command_sketch.md").read_text(encoding="utf-8")

    assert formula["primary_formula"] == PRIMARY_FORMULA
    assert formula["sc_bonus_location"] == "outside cost denominator"
    assert formula["over_cost_runtime_primary_allowed"] is False
    assert formula["primary_formula"] not in BAD_FORMULAS
    for bad in BAD_FORMULAS:
        assert bad in formula["prohibited_runtime_primary_formulas"]

    counts = sequence["exact_counts"]
    assert counts["captured_frames"] == 2
    assert counts["measured_only_observed_state_updates"] == 2
    assert counts["map_predict_calls"] == 2
    assert counts["selected_action_executions"] == 1
    assert counts["second_action"] == 0
    assert counts["third_frame"] == 0
    assert counts["rollout"] == 0
    assert design["runtime_executed"] is False
    assert design["rollout_ready"] is False

    expected = requirements["expected_counts"]
    assert expected["frames"] == 2
    assert expected["map_predict_calls"] == 2
    assert expected["selected_action_executions"] == 1
    assert expected["rollout_artifacts"] == 0

    assert command.startswith("DO NOT RUN IN STAGE 4A-6.5aj."), "command sketch not clearly marked"
    for flag in (
        "--max_workers 32",
        "--execute_exactly_one_action",
        "--max_frames 2",
        "--no_third_frame",
        "--no_second_action",
        "--no_rollout",
    ):
        assert flag in command, f"command sketch missing {flag}"
    return {"passed": True}


def test_safety_gates(output_dir: Path) -> dict[str, Any]:
    gates = load_json(output_dir / "future_pre_action_safety_gates.json")
    frame2 = load_json(output_dir / "future_frame2_stop_conditions.json")
    prediction = load_json(output_dir / "prediction_safety_design.json")
    summary = load_json(output_dir / "stage4a65aj_design_review_summary.json")

    gate_text = "\n".join(gates["pre_action_gates"] + gates["action_block_conditions"])
    assert "low_cost_artifact" in gate_text
    assert "historical prior basin" in gate_text
    assert "prediction is read-only" in gate_text
    for phrase in (
        "traversability",
        "collision",
        "ray blocking",
        "candidate sampling",
        "edge validity",
        "target/ground-truth/future-observed",
    ):
        assert phrase in gate_text, f"missing safety phrase: {phrase}"

    assert gates["if_any_hard_gate_fails"]["execute_action"] is False
    assert gates["if_any_hard_gate_fails"]["capture_frame2"] is False
    assert gates["if_any_hard_gate_fails"]["rollout"] is False
    assert frame2["second_action_allowed"] is False
    assert frame2["third_frame_allowed"] is False
    assert frame2["rollout_allowed"] is False

    assert prediction["prediction_read_only"] is True
    false_prediction_keys = [
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
    for key in false_prediction_keys:
        assert prediction[key] is False, f"prediction safety key should be false: {key}"

    summary_safety = summary["safety"]
    for key in (
        "isaac_startup",
        "rgb_depth_capture",
        "map_predict_call",
        "sscnet_inference",
        "selected_action_execution",
        "two_frame_runtime_execution",
        "rollout",
        "open_ended_loop",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_npz_modified",
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
    ):
        assert summary_safety[key] is False, f"summary safety key should be false: {key}"
    assert summary_safety["future_command_marked_do_not_run_in_stage4a65aj"] is True
    return {"passed": True}


def test_hardware(output_dir: Path) -> dict[str, Any]:
    hardware = load_json(output_dir / "hardware_policy_for_future_runtime.json")
    assert hardware["os_cpu_count_expected"] == 32
    assert hardware["requested_max_workers"] == 32
    assert hardware["actual_max_workers"] == min(32, hardware["os_cpu_count"] or 1)
    assert hardware["future_command_includes_max_workers_32"] is True
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert key in hardware["thread_env"], key
    return {"passed": True, "actual_max_workers": hardware["actual_max_workers"]}


def test_absence_of_forbidden_outputs(output_dir: Path) -> dict[str, Any]:
    found: list[str] = []
    for pattern in PROHIBITED_OUTPUT_PATTERNS:
        found.extend(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
    assert not found, f"forbidden runtime artifacts found: {found[:20]}"
    return {"passed": True}


def test_context_updated() -> dict[str, Any]:
    required_phrases = [
        "Stage 4A-6.5aj",
        "design review only",
        "Stage 4A-6.5ak",
        "exactly two frames",
        "exactly two map_predict calls",
        "exactly one action",
        PRIMARY_FORMULA,
        "no rollout",
    ]
    for path in CONTEXT_FILES:
        text = path.read_text(encoding="utf-8")
        missing = [phrase for phrase in required_phrases if phrase not in text]
        assert not missing, f"context file {path} missing phrases: {missing}"
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "loaded_inputs": test_loaded_inputs(output_dir),
        "formula_and_design": test_formula_and_design(output_dir),
        "safety_gates": test_safety_gates(output_dir),
        "hardware": test_hardware(output_dir),
        "absence_of_forbidden_outputs": test_absence_of_forbidden_outputs(output_dir),
        "context_updated": test_context_updated(),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
