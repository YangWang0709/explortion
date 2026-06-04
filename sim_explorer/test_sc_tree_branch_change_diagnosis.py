#!/usr/bin/env python3
"""Validate Stage 4A-6.5q branch-change diagnosis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "frame002_branch_change_nodes.csv",
    "frame002_branch_change_paths.json",
    "frame002_branch_change_summary.md",
    "frame001_tree_rank_table.csv",
    "frame002_tree_rank_table.csv",
    "frame001_rank_summary.json",
    "frame001_rank_summary.md",
    "frame002_rank_summary.json",
    "frame002_rank_summary.md",
    "frame002_gated_replay_results.csv",
    "frame002_gated_replay_summary.json",
    "frame002_gated_replay_summary.md",
    "missing_fields_report.json",
    "stage4a65q_sc_tree_branch_change_summary.json",
    "stage4a65q_sc_tree_branch_change_summary.md",
    "recommended_next_faithful_step.md",
    "frame002_measured_vs_sc_selected_topdown.png",
    "frame002_branch_paths_topdown.png",
    "frame002_gated_replay_selected_children_topdown.png",
    "frame002_gain_exp_vs_gain_sc_scatter.png",
    "frame002_value_rank_scatter.png",
]

FORBIDDEN_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "frame*_pose.json",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
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


def test_branch_rank_and_replay_content(output_dir: Path) -> dict[str, Any]:
    branch_rows = read_csv_rows(output_dir / "frame002_branch_change_nodes.csv")
    roles = {row.get("role") for row in branch_rows}
    for role in (
        "measured_selected_child",
        "measured_best_descendant",
        "sc_selected_child",
        "sc_best_descendant",
    ):
        _assert(role in roles, f"missing branch role: {role}")

    summary = load_json(output_dir / "stage4a65q_sc_tree_branch_change_summary.json")
    branch = summary["branch_change"]
    _assert(branch["measured_selected_child_id"] == "n0001", "wrong measured selected child")
    _assert(branch["sc_selected_child_id"] == "n0127", "wrong SC selected child")
    _assert(branch["sc_best_descendant_id"] == "n0162", "wrong SC best descendant")
    _assert(float(branch["selected_child_world_delta_m"]) >= 0.5, "selected child change should be spatially meaningful")

    frame1_rows = read_csv_rows(output_dir / "frame001_tree_rank_table.csv")
    frame2_rows = read_csv_rows(output_dir / "frame002_tree_rank_table.csv")
    _assert(len(frame1_rows) >= 250, "frame001 rank table too small")
    _assert(len(frame2_rows) >= 250, "frame002 rank table too small")
    for row in frame2_rows[:5]:
        for key in ("measured_value_rank", "raw_hybrid_value_rank", "sc_only_value_rank"):
            _assert(key in row, f"rank table missing {key}")

    replay_rows = read_csv_rows(output_dir / "frame002_gated_replay_results.csv")
    formulas = {row.get("formula") for row in replay_rows}
    for formula in (
        "raw_count",
        "weight_0p0",
        "weight_0p25",
        "weight_0p5",
        "weight_1p0",
        "cap_10",
        "cap_25",
        "cap_50",
        "occupied_only_gain_occ",
        "confidence_weighted_gain_conf",
        "calibrated_occupied",
    ):
        _assert(formula in formulas, f"missing replay formula: {formula}")

    gated = load_json(output_dir / "frame002_gated_replay_summary.json")
    _assert("raw_count" in gated["formulas_preserving_sc_branch"], "raw_count should preserve SC branch")
    _assert("weight_0p0" in gated["formulas_returning_to_measured_selected_child"], "weight_0 should return to measured child")
    _assert(gated["minimum_sc_weight_changes_selected_child"] is not None, "missing minimum SC weight")
    missing = load_json(output_dir / "missing_fields_report.json")
    _assert(
        missing["calibrated_occupied_replay_status"] == "skipped_missing_per_node_probability_samples",
        "calibrated replay should be skipped with explicit reason",
    )
    return {"passed": True, "replay_rows": len(replay_rows)}


def test_safety_and_absence(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65q_sc_tree_branch_change_summary.json")
    safety = summary["safety"]
    false_keys = [
        "isaac_startup",
        "map_predict_rerun",
        "sscnet_inference",
        "rollout",
        "selected_action_execution",
        "training_or_rl",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_scoring",
        "source_modified_built",
        "new_map_predict_outputs_created",
    ]
    for key in false_keys:
        _assert(not bool(safety.get(key)), f"safety flag should be false: {key}")

    _assert(safety["input_hashes_before"] == safety["input_hashes_after"], "input artifact hashes changed")
    _assert(safety["checkpoint_sha256_before"] == safety["checkpoint_sha256_after"], "checkpoint hash changed")
    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.glob(pattern))
        _assert(not matches, f"forbidden artifacts for pattern {pattern}: {matches}")
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "branch_rank_and_replay_content": test_branch_rank_and_replay_content(output_dir),
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
