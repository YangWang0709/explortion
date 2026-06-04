#!/usr/bin/env python3
"""Smoke and unit tests for Stage 4A-6.5i offline mini-RRT tree builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from astar_planner import build_traversability_grid
from offline_tree_utility_prototype import compute_global_normalized_gain, select_subsequent_best
from offline_mini_rrt_tree import (
    FREE,
    ROOT_ID,
    UNKNOWN,
    MiniRRTSegment,
    add_tree_child,
    build_mini_rrt_tree,
    line_is_traversable,
    sha256_file,
)


REQUIRED_OUTPUTS = [
    "mini_rrt_tree_segments.jsonl",
    "mini_rrt_tree_summary.json",
    "mini_rrt_tree_summary.md",
    "subsequent_best_decision.json",
    "subsequent_best_decision.md",
    "tree_vs_one_step_comparison.json",
    "tree_vs_one_step_comparison.md",
    "sampled_nodes.csv",
    "rejected_samples.csv",
    "gain_cost_value_table.csv",
    "tree_formula_reference.md",
    "missing_or_limited_features.md",
    "recommended_next_faithful_step.md",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_edge_collision() -> dict[str, Any]:
    traversable = np.ones((5, 5), dtype=bool)
    traversable[2, 2] = False
    ok, reason, cells = line_is_traversable(traversable, (0, 0), (4, 4))
    _assert(not ok, "line through blocked cell should be rejected")
    _assert(reason == "edge_non_traversable_or_unknown", f"unexpected reason: {reason}")
    ok2, reason2, _ = line_is_traversable(traversable, (0, 0), (0, 4))
    _assert(ok2 and reason2 == "ok", "clear line should be accepted")
    return {"passed": True, "blocked_line_cells": cells}


def test_reachable_unknown_not_traversable() -> dict[str, Any]:
    observed = np.full((5, 5, 3), FREE, dtype=np.int8)
    observed[2, 2, :] = UNKNOWN
    traversability = build_traversability_grid(observed, voxel_size=0.1, robot_radius_m=0.0)
    traversable = np.asarray(traversability["traversable"], dtype=bool)
    _assert(not bool(traversable[2, 2]), "UNKNOWN xy should not be traversable")
    _assert(bool(traversable[1, 1]), "FREE xy should remain traversable")
    return {"passed": True, "traversable_count": int(np.count_nonzero(traversable))}


def test_parent_children() -> dict[str, Any]:
    tree = {
        ROOT_ID: MiniRRTSegment(
            segment_id=ROOT_ID,
            parent_id=None,
            end_grid=[0, 0, 0],
            end_world=[0.0, 0.0, 0.0],
        )
    }
    child = MiniRRTSegment(segment_id="n0001", parent_id=ROOT_ID, gain=1.0, cost=1.0)
    add_tree_child(tree, ROOT_ID, child)
    _assert("n0001" in tree, "child missing from tree")
    _assert(tree["n0001"].parent_id == ROOT_ID, "child parent mismatch")
    _assert(tree[ROOT_ID].children == ["n0001"], "parent children mismatch")
    return {"passed": True, "children": tree[ROOT_ID].children}


def test_global_normalized_gain_low_cost_trap() -> dict[str, Any]:
    tree = {
        ROOT_ID: MiniRRTSegment(segment_id=ROOT_ID, parent_id=None, gain=0.0, cost=0.0),
    }
    add_tree_child(tree, ROOT_ID, MiniRRTSegment(segment_id="A", parent_id=ROOT_ID, gain=10.0, cost=1.0))
    add_tree_child(tree, ROOT_ID, MiniRRTSegment(segment_id="B", parent_id=ROOT_ID, gain=2.0, cost=0.1))
    add_tree_child(tree, "A", MiniRRTSegment(segment_id="A1", parent_id="A", gain=100.0, cost=4.0))
    warnings = compute_global_normalized_gain(tree, ROOT_ID)
    decision = select_subsequent_best(tree, ROOT_ID)
    _assert(decision["selected_child_id"] == "A", "low-cost trap should be overcome by A1 descendant")
    _assert(decision["selected_child_best_descendant_id"] == "A1", "A1 should make A win")
    return {"passed": True, "decision": decision, "warnings": warnings}


def make_synthetic_observed() -> np.ndarray:
    observed = np.full((30, 30, 8), UNKNOWN, dtype=np.int8)
    observed[6:24, 6:24, :] = FREE
    observed[6:24, 6, :] = UNKNOWN
    observed[6:24, 23, :] = UNKNOWN
    observed[6, 6:24, :] = UNKNOWN
    observed[23, 6:24, :] = UNKNOWN
    observed[14:17, 14:17, :] = FREE
    return observed


def test_deterministic_seed() -> dict[str, Any]:
    observed = make_synthetic_observed()
    bounds = {"x": (0.0, 3.0), "y": (0.0, 3.0), "z": (0.0, 0.8)}
    kwargs = dict(
        observed_state=observed,
        root_grid=[15, 15, 3],
        root_world=[1.55, 1.55, 0.35],
        root_yaw=0.0,
        bounds=bounds,
        seed=11,
        num_nodes=40,
        max_extension_m=0.35,
        sample_mode="mixed",
        gain_mode="exp",
        v_max=1.0,
        robot_radius_m=0.0,
        voxel_size=0.1,
        raycast_stride=4,
        num_yaw_samples=4,
        max_ray_length_m=1.5,
        profile=False,
    )
    first = build_mini_rrt_tree(**kwargs)
    second = build_mini_rrt_tree(**kwargs)
    first_grids = [row["end_grid"] for row in first["accepted_rows"]]
    second_grids = [row["end_grid"] for row in second["accepted_rows"]]
    _assert(first_grids == second_grids, "same seed should produce same accepted nodes")
    _assert(first["decision"]["selected_child_id"] == second["decision"]["selected_child_id"], "same seed should select same child")
    _assert(first["decision"]["selected_child_id"] is not None, "synthetic tree should select a child")
    return {
        "passed": True,
        "accepted_nodes": len(first["accepted_rows"]),
        "selected_child": first["decision"]["selected_child_id"],
    }


def test_required_outputs_and_real_hash(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.exists(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")

    summary = load_json(output_dir / "mini_rrt_tree_summary.json")
    observed_path = Path(summary["inputs"]["observed_state"])
    before = summary["map"]["observed_state_sha256_before"]
    after = summary["map"]["observed_state_sha256_after"]
    current = sha256_file(observed_path)
    _assert(before == after == current, "observed_state hash changed or summary mismatch")
    _assert(summary["safety"]["observed_state_modified"] is False, "summary claims observed_state modified")
    for key in (
        "isaac_startup",
        "rollout",
        "online_expert_loop",
        "map_predict_rerun",
        "sscnet_inference_or_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "prediction_writeback",
        "prediction_used_for_traversability_collision",
        "prediction_blocks_rays",
        "target_lr_target_hr_ground_truth_scoring",
        "external_source_modified_or_built",
    ):
        _assert(not bool(summary["safety"].get(key, False)), f"safety flag is true: {key}")
    _assert(summary["tree"]["built_successfully"] is True, "real mini-RRT tree did not build successfully")
    return {
        "passed": True,
        "observed_state": str(observed_path),
        "observed_state_sha256": current,
        "accepted_nodes": summary["tree"]["accepted_nodes_excluding_root"],
    }


def run_tests(output_dir: Path) -> dict[str, Any]:
    results = {
        "edge_collision": test_edge_collision(),
        "reachable_unknown": test_reachable_unknown_not_traversable(),
        "tree_parent_children": test_parent_children(),
        "global_normalized_gain_subsequent_best": test_global_normalized_gain_low_cost_trap(),
        "deterministic_seed": test_deterministic_seed(),
        "real_output_hash": test_required_outputs_and_real_hash(output_dir),
    }
    summary = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tests(Path(args.output_dir))
