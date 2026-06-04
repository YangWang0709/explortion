#!/usr/bin/env python3
"""Validate Stage 4A-6.5x SC gain design review outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = [
    "source_sc_gain_evidence.json",
    "source_sc_gain_evidence.md",
    "source_gain_formula_comparison.md",
    "current_gain_formula_audit.csv",
    "current_gain_formula_audit.json",
    "current_gain_formula_audit.md",
    "branch_visible_voxel_decomposition.csv",
    "branch_visible_voxel_decomposition.json",
    "branch_visible_voxel_decomposition.md",
    "visible_voxel_missing_fields_report.json",
    "candidate_sc_gain_variants.csv",
    "candidate_sc_gain_variants.json",
    "candidate_sc_gain_variants.md",
    "stage4a65x_sc_gain_design_review_summary.json",
    "stage4a65x_sc_gain_design_review_summary.md",
    "recommended_next_faithful_step.md",
    "safety_summary.json",
    "plot_status.json",
]

PLOT_FILES = [
    "branch_visible_voxel_counts.png",
    "branch_prediction_confidence_distributions.png",
    "branch_gain_component_stack.png",
    "measured_vs_sc_branch_overlap_topdown.png",
    "predicted_voxel_spatial_distribution_topdown.png",
    "candidate_formula_effects_bar.png",
    "seed_replay_if_done.png",
]

PROHIBITED_PATTERNS = [
    "depth_*.npy",
    "rgb_*.png",
    "frame*_depth.npy",
    "frame*_rgb.png",
    "local_prediction.npz",
    "global_prediction_layer.npz",
    "sscnet_depth_input.npy",
    "sscnet_position.npy",
    "valid_position_mask.npy",
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "step_*.npz",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_file(path: Path) -> None:
    if not path.is_file():
        raise AssertionError(f"missing required file: {path}")
    if path.stat().st_size <= 0:
        raise AssertionError(f"empty required file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise AssertionError(f"output dir does not exist: {output_dir}")

    for name in REQUIRED_FILES:
        assert_file(output_dir / name)

    replay_csv = output_dir / "candidate_formula_seed_replay.csv"
    replay_skip = output_dir / "candidate_formula_seed_replay_skipped_reason.md"
    if not replay_csv.is_file() and not replay_skip.is_file():
        raise AssertionError("missing candidate formula replay output or skipped reason")

    plot_status = read_json(output_dir / "plot_status.json")
    for name in PLOT_FILES:
        if bool(plot_status.get("plots", {}).get(name, False)):
            assert_file(output_dir / name)
        elif name not in set(plot_status.get("skipped", [])):
            raise AssertionError(f"plot neither present nor explicitly skipped: {name}")

    missing = read_json(output_dir / "visible_voxel_missing_fields_report.json")
    if "visible_voxel_ids_saved_in_tree_artifacts" not in missing:
        raise AssertionError("missing visible voxel saved-field marker")

    evidence = read_json(output_dir / "source_sc_gain_evidence.json")
    if "source_ambiguous_parts" not in evidence:
        raise AssertionError("source ambiguity field missing")
    if not evidence["source_ambiguous_parts"]:
        raise AssertionError("source ambiguity list should record at least the simulator-NPZ mapping limitation")

    summary = read_json(output_dir / "stage4a65x_sc_gain_design_review_summary.json")
    safety = summary.get("safety", {})
    expected_false = [
        "isaac_startup",
        "new_capture",
        "map_predict_rerun",
        "sscnet_inference",
        "selected_action_execution",
        "rollout",
        "open_ended_loop",
        "training_rl",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_scoring",
        "external_source_modified_or_built",
        "coverage_improvement_claim",
    ]
    for key in expected_false:
        if bool(safety.get(key, True)):
            raise AssertionError(f"safety flag should be false: {key}={safety.get(key)}")
    hashes_before = safety.get("input_hashes_before", {})
    hashes_after = safety.get("input_hashes_after", {})
    for key in ("observed_state", "prediction_npz", "checkpoint"):
        if hashes_before.get(key) != hashes_after.get(key):
            raise AssertionError(f"input hash changed for {key}")

    for pattern in PROHIBITED_PATTERNS:
        matches = list(output_dir.rglob(pattern))
        if matches:
            raise AssertionError(f"prohibited rollout/capture/inference artifacts found for {pattern}: {matches[:3]}")

    if summary.get("answers", {}).get("runtime_smoke_readiness") is not False:
        raise AssertionError("runtime smoke should not be marked ready")
    if summary.get("answers", {}).get("rollout_readiness") is not False:
        raise AssertionError("rollout should not be marked ready")

    print(json.dumps({"output_dir": str(output_dir), "validation": "passed"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
