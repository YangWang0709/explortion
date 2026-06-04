#!/usr/bin/env python3
"""Validate Stage 4A-6.5ah multiscene/design-review outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_COMMON_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "hardware_utilization_report.json",
    "hardware_utilization_report.md",
    "additional_frame_discovery_inventory.csv",
    "additional_frame_discovery_inventory.json",
    "additional_frame_discovery_inventory.md",
    "additional_frame_duplicates.csv",
    "additional_frame_duplicates.json",
    "new_complete_frame_manifest.csv",
    "new_complete_frame_manifest.json",
    "new_complete_frame_manifest.md",
    "skipped_frame_candidates.csv",
    "skipped_frame_candidates.json",
    "skipped_frame_candidates.md",
    "stage4a65ah_multiscene_or_runtime_design_review_summary.json",
    "stage4a65ah_multiscene_or_runtime_design_review_summary.md",
    "recommended_next_faithful_step.md",
]

REQUIRED_DESIGN_FILES = [
    "runtime_smoke_design_review.md",
    "runtime_smoke_safety_checklist.json",
    "runtime_smoke_safety_checklist.md",
    "future_stage4a65ai_command_sketch.md",
]

FORBIDDEN_PATTERNS = [
    "depth_*.npy",
    "depth_*.png",
    "rgb_*.png",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "sscnet_depth_input.npy",
    "sscnet_position.npy",
    "sscnet_input_debug.npz",
    "valid_position_mask.npy",
    "transitions.jsonl",
    "step_*.npz",
    "observed_state*.npy",
    "episode_summary.json",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_file(path: Path) -> None:
    assert path.is_file(), f"missing required file: {path}"
    assert path.stat().st_size > 0, f"empty required file: {path}"


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_COMMON_FILES:
        assert_file(output_dir / name)
    summary = load_json(output_dir / "stage4a65ah_multiscene_or_runtime_design_review_summary.json")
    if summary["branch"] == "runtime_design_review_only":
        for name in REQUIRED_DESIGN_FILES:
            assert_file(output_dir / name)
    return {"passed": True, "required_common_files": len(REQUIRED_COMMON_FILES)}


def test_content(output_dir: Path, expected_new_complete_frames: int | None) -> dict[str, Any]:
    context = load_json(output_dir / "loaded_context_manifest.json")
    hardware = load_json(output_dir / "hardware_utilization_report.json")
    inventory = load_json(output_dir / "additional_frame_discovery_inventory.json")
    duplicates = load_json(output_dir / "additional_frame_duplicates.json")
    new_manifest = load_json(output_dir / "new_complete_frame_manifest.json")
    skipped = load_json(output_dir / "skipped_frame_candidates.json")
    summary = load_json(output_dir / "stage4a65ah_multiscene_or_runtime_design_review_summary.json")

    assert context["stage"] == "Stage 4A-6.5ah"
    assert context["confirmed_prior_context"]["stage4a65af_complete"] is True
    assert context["confirmed_prior_context"]["stage4a65ag_complete"] is True
    assert context["safety_scope"]["isaac_startup"] is False
    assert context["safety_scope"]["map_predict_rerun"] is False
    assert context["safety_scope"]["rollout"] is False

    assert hardware["requested_max_workers"] == 32
    assert hardware["actual_max_workers"] == min(32, hardware["os_cpu_count"])
    assert hardware["parallel_backend"] == "ProcessPoolExecutor"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert str(hardware[key]) == "1", f"expected {key}=1, got {hardware[key]}"

    assert inventory, "inventory should contain discovered candidate rows"
    classes = {row["classification"] for row in inventory}
    assert "already_in_stage4a65ag" in classes or "duplicate_of_existing" in classes
    assert isinstance(duplicates, list)
    assert isinstance(skipped, list)
    assert len(read_csv(output_dir / "additional_frame_discovery_inventory.csv")) == len(inventory)

    if expected_new_complete_frames is not None:
        assert len(new_manifest) == expected_new_complete_frames, (
            f"expected {expected_new_complete_frames} new frames, got {len(new_manifest)}"
        )
    assert summary["discovery"]["new_complete_frame_count"] == len(new_manifest)
    assert summary["runtime_executed"] is False
    assert summary["runtime_smoke_readiness"] is False
    assert summary["rollout_readiness"] is False
    assert summary["coverage_improvement_claimed"] is False

    if not new_manifest:
        checklist = load_json(output_dir / "runtime_smoke_safety_checklist.json")
        assert checklist["runtime_executed"] is False
        future = checklist["future_runtime_smoke_design"]
        assert future["one_isaac_startup"] is True
        assert future["one_frame_only"] is True
        assert future["one_map_predict_call_only"] is True
        assert future["execute_selected_action"] is False
        assert future["rollout"] is False
        command_sketch = (output_dir / "future_stage4a65ai_command_sketch.md").read_text(encoding="utf-8")
        assert "DO NOT RUN IN THIS STAGE" in command_sketch
        assert "--max_workers 32" in command_sketch

    return {
        "passed": True,
        "candidate_rows": len(inventory),
        "new_complete_frame_count": len(new_manifest),
        "duplicate_rows": len(duplicates),
        "skipped_rows": len(skipped),
    }


def test_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65ah_multiscene_or_runtime_design_review_summary.json")
    safety = summary["safety"]
    false_keys = [
        "isaac_startup",
        "new_capture",
        "map_predict_rerun",
        "sscnet_inference",
        "selected_action_execution",
        "two_frame_runtime",
        "rollout",
        "open_ended_loop",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_ray_blocking",
        "target_ground_truth_planning_scoring",
        "future_observed_planning_scoring",
        "external_source_modified_built",
        "pareto_dominance_gate_implemented",
        "runtime_planner_implemented",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        assert not bool(safety.get(key)), f"safety flag should be false: {key}"
    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.rglob(pattern))
        assert not matches, f"forbidden generated artifact pattern {pattern}: {matches[:5]}"
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    expected = None if args.expected_new_complete_frames < 0 else int(args.expected_new_complete_frames)
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "content": test_content(output_dir, expected),
        "safety": test_safety(output_dir),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_new_complete_frames", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
