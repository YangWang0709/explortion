#!/usr/bin/env python3
"""Validate Stage 4A-6.5u gated-tree seed robustness diagnosis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "decision_comparison_seed0_seed1.csv",
    "decision_comparison_seed0_seed1.json",
    "decision_comparison_seed0_seed1.md",
    "frame002_topk_branch_seed0.csv",
    "frame002_topk_branch_seed1.csv",
    "frame002_topk_spatial_match_seed0_seed1.csv",
    "frame002_topk_spatial_match_summary.json",
    "frame002_topk_spatial_match_summary.md",
    "frame002_rank_margin_seed0.json",
    "frame002_rank_margin_seed0.md",
    "frame002_rank_margin_seed1.json",
    "frame002_rank_margin_seed1.md",
    "rank_margin_comparison.csv",
    "rank_margin_comparison.json",
    "rank_margin_comparison.md",
    "branch_classification.json",
    "branch_classification.md",
    "missing_fields_report.json",
    "stage4a65u_seed_robustness_summary.json",
    "stage4a65u_seed_robustness_summary.md",
    "recommended_next_faithful_step.md",
    "frame002_seed0_seed1_selected_branches_topdown.png",
    "frame002_topk_branch_cloud_seed0_seed1.png",
    "frame002_value_margin_seed0_seed1.png",
    "frame002_gain_exp_vs_effective_sc_seed0_seed1.png",
    "frame002_cost_vs_value_seed0_seed1.png",
]

FORBIDDEN_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "frame*_pose.json",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "map_predict*",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_topdown_path.png",
    "rollout_*.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_png_nonempty(path: Path) -> None:
    data = path.read_bytes()
    _assert(len(data) > 8, f"empty PNG: {path}")
    _assert(data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}")


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    for name in REQUIRED_FILES:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
        if path.suffix == ".png":
            assert_png_nonempty(path)
    return {"passed": True, "required_files": len(REQUIRED_FILES)}


def test_content(output_dir: Path) -> dict[str, Any]:
    decision_rows = read_csv_rows(output_dir / "decision_comparison_seed0_seed1.csv")
    _assert(len(decision_rows) == 12, f"expected 12 decision rows, got {len(decision_rows)}")
    row_index = {(row["seed"], row["frame"], row["mode"]): row for row in decision_rows}
    _assert(row_index[("seed0", "2", "confidence_weighted")]["selected_child_id"] == "n0127", "seed0 confidence winner changed")
    _assert(row_index[("seed1", "2", "confidence_weighted")]["selected_child_id"] == "n0057", "seed1 confidence winner changed")
    _assert(row_index[("seed1", "2", "measured")]["selected_child_id"] == "n0057", "seed1 measured winner changed")

    branch = load_json(output_dir / "branch_classification.json")
    entries = branch.get("entries", {})
    for key in ("seed0_confidence_weighted", "seed1_confidence_weighted", "seed1_cap25_shadow"):
        _assert(key in entries, f"missing branch classification entry: {key}")
    _assert(entries["seed1_confidence_weighted"]["same_as_measured"], "seed1 confidence should be same_as_measured")
    _assert(
        entries["seed1_confidence_weighted"]["spatially_same_as_seed0_sc"],
        "seed1 confidence should be spatially close to seed0 SC",
    )
    _assert(entries["seed1_confidence_weighted"]["selected_to_seed0_sc_m"] is not None, "missing seed1 selected delta")
    _assert(entries["seed1_confidence_weighted"]["best_to_seed0_sc_m"] is not None, "missing seed1 best delta")

    rank = load_json(output_dir / "rank_margin_comparison.json")
    _assert(len(rank.get("seeds", [])) == 2, "rank comparison should include two seeds")
    csv_rows = read_csv_rows(output_dir / "rank_margin_comparison.csv")
    _assert(len(csv_rows) == 2, "rank comparison CSV should include two rows")

    topk = load_json(output_dir / "frame002_topk_spatial_match_summary.json")
    _assert("seed0_confidence_reference_nearest_seed1_confidence" in topk, "missing nearest top-K match")
    nearest = topk["seed0_confidence_reference_nearest_seed1_confidence"]
    _assert(nearest["root_child_distance_m"] is not None, "missing top-K root distance")
    _assert(nearest["best_descendant_distance_m"] is not None, "missing top-K best distance")

    summary = load_json(output_dir / "stage4a65u_seed_robustness_summary.json")
    _assert(summary["seed1_to_seed0_sc_selected_child_delta_m"] is not None, "missing summary selected delta")
    _assert(summary["seed1_to_seed0_sc_best_descendant_delta_m"] is not None, "missing summary best delta")
    _assert(summary["still_cannot_rollout"], "summary must keep rollout disallowed")
    _assert(summary["recommended_next_faithful_step"] != "rollout", "must not recommend rollout")
    return {"passed": True, "decision_rows": len(decision_rows)}


def test_safety_and_absence(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65u_seed_robustness_summary.json")
    safety = summary["safety"]
    false_keys = [
        "isaac_startup",
        "new_capture",
        "map_predict_rerun",
        "sscnet_inference",
        "selected_action_execution",
        "rollout",
        "open_ended_loop",
        "training_or_rl",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_scoring",
        "source_modified_built",
        "new_map_predict_outputs_created",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        _assert(not bool(safety.get(key)), f"safety flag should be false: {key}")
    _assert(safety["input_hashes_before"] == safety["input_hashes_after"], "input artifact hashes changed")
    _assert(safety["checkpoint_sha256_before"] == safety["checkpoint_sha256_after"], "checkpoint hash changed")
    _assert(not safety["prohibited_artifacts_in_output"], "summary recorded prohibited output artifacts")
    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.glob(pattern))
        _assert(not matches, f"forbidden artifacts for pattern {pattern}: {matches}")
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "content": test_content(output_dir),
        "safety_and_absence": test_safety_and_absence(output_dir),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_tests(args)


if __name__ == "__main__":
    main()
