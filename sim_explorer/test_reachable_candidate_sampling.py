#!/usr/bin/env python3
"""Tests for Stage 4A-3.6 reachability-aware candidate sampling."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from astar_planner import connected_component_from_start, nearest_traversable_cell
from sim_paper_expert import (
    FREE,
    UNKNOWN,
    build_traversability_grid,
    compute_reachable_frontier_candidate_cells,
    frontier_adjacent_free_xy_mask,
    sample_candidate_views_from_frontiers,
)

ONE_STEP_REACHABLE_OUTPUT = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_reachable_smoke/expert_step_decision.json"
)


def synthetic_two_component_observed() -> np.ndarray:
    observed = np.full((12, 12, 8), UNKNOWN, dtype=np.int8)
    observed[1:5, 1:5, 4] = FREE
    observed[8:11, 8:11, 4] = FREE
    return observed


def test_connected_component_simple_grid() -> None:
    traversable = np.zeros((9, 9), dtype=bool)
    traversable[1:4, 1:4] = True
    traversable[6:8, 6:8] = True
    component = connected_component_from_start(traversable, (2, 2), allow_diagonal=True)
    assert component["start_valid"] is True, component
    assert component["reachable_count"] == 9, component
    assert bool(component["reachable_mask"][3, 3])
    assert not bool(component["reachable_mask"][6, 6])


def test_nearest_traversable_cell_blocked_start() -> None:
    traversable = np.zeros((8, 8), dtype=bool)
    traversable[3, 4] = True
    snap = nearest_traversable_cell(traversable, (2, 2), max_radius_cells=4)
    assert snap["found"] is True, snap
    assert snap["cell"] == [3, 4], snap
    assert abs(float(snap["distance_cells"]) - np.sqrt(5.0)) < 1e-6

    miss = nearest_traversable_cell(traversable, (0, 0), max_radius_cells=1)
    assert miss["found"] is False, miss


def test_reachable_frontier_mask_excludes_disconnected_free() -> None:
    observed = synthetic_two_component_observed()
    traversability = build_traversability_grid(
        observed,
        voxel_size=0.1,
        robot_height_m=0.45,
        clearance_height_m=0.2,
        robot_radius_m=0.0,
    )
    frontier_xy = frontier_adjacent_free_xy_mask(observed)
    result = compute_reachable_frontier_candidate_cells(
        observed_state=observed,
        traversability=traversability,
        current_xy=(2, 2),
        frontier_adjacent_free_mask=frontier_xy,
        max_snap_radius_cells=3,
        snap_start_to_traversable=True,
    )
    assert result["reason"] == "ok", result
    assert result["reachable_count"] == 16, result
    assert result["reachable_frontier_adjacent_count"] > 0, result
    assert result["candidate_source"] == "reachable_frontier", result
    candidate_mask = np.asarray(result["candidate_mask"], dtype=bool)
    assert np.count_nonzero(candidate_mask[8:11, 8:11]) == 0
    assert np.count_nonzero(candidate_mask[1:5, 1:5]) > 0


def test_reachable_sampling_only_samples_reachable_component() -> None:
    observed = synthetic_two_component_observed()
    traversability = build_traversability_grid(
        observed,
        voxel_size=0.1,
        robot_height_m=0.45,
        clearance_height_m=0.2,
        robot_radius_m=0.0,
    )
    sampled = sample_candidate_views_from_frontiers(
        observed_state=observed,
        current_grid=(2, 2, 4),
        bounds={"x": (0.0, 1.2), "y": (0.0, 1.2), "z": (0.0, 0.8)},
        voxel_size=0.1,
        num_candidates=32,
        seed=7,
        candidate_sampling_mode="reachable_frontier",
        traversability=traversability,
        snap_start_to_traversable=True,
        max_snap_radius_cells=3,
        return_diagnostics=True,
    )
    candidates, diagnostics = sampled
    reachable_mask = np.asarray(diagnostics["reachable_mask"], dtype=bool)
    assert len(candidates) > 0
    for candidate in candidates:
        i, j, k = candidate.grid_position
        assert reachable_mask[i, j], candidate
        assert i < 6 and j < 6, candidate
        assert observed[i, j, k] == FREE, candidate
        assert candidate.candidate_source == "reachable_frontier", candidate


def test_reachable_sampling_snaps_blocked_start() -> None:
    observed = synthetic_two_component_observed()
    traversability = build_traversability_grid(
        observed,
        voxel_size=0.1,
        robot_height_m=0.45,
        clearance_height_m=0.2,
        robot_radius_m=0.0,
    )
    frontier_xy = frontier_adjacent_free_xy_mask(observed)
    result = compute_reachable_frontier_candidate_cells(
        observed_state=observed,
        traversability=traversability,
        current_xy=(0, 0),
        frontier_adjacent_free_mask=frontier_xy,
        max_snap_radius_cells=3,
        snap_start_to_traversable=True,
    )
    assert result["reason"] == "ok", result
    assert result["snapped"] is True, result
    assert result["snapped_current_xy"] == [1, 1], result
    assert result["reachable_count"] == 16, result


def test_one_step_reachable_output_if_present() -> None:
    if not ONE_STEP_REACHABLE_OUTPUT.exists():
        print(f"one_step_reachable_output: skipped missing {ONE_STEP_REACHABLE_OUTPUT}")
        return
    with ONE_STEP_REACHABLE_OUTPUT.open("r", encoding="utf-8") as handle:
        decision = json.load(handle)
    diagnostics = decision["diagnostics"]
    assert diagnostics["path_cost_mode"] == "astar", diagnostics
    assert diagnostics["candidate_sampling_mode"] == "reachable_frontier", diagnostics
    assert int(diagnostics["reachable_candidates"]) > 0, diagnostics
    assert int(diagnostics["unreachable_candidates"]) < 52, diagnostics
    assert int(diagnostics["reachable_component_count"]) > 0, diagnostics
    assert diagnostics["candidate_source"] in {"reachable_frontier", "reachable_free_fallback"}, diagnostics
    assert diagnostics["target_lr_used"] is False
    assert diagnostics["target_hr_used"] is False
    assert diagnostics["ground_truth_used"] is False
    assert diagnostics["rl_or_training_used"] is False


def main() -> None:
    test_connected_component_simple_grid()
    test_nearest_traversable_cell_blocked_start()
    test_reachable_frontier_mask_excludes_disconnected_free()
    test_reachable_sampling_only_samples_reachable_component()
    test_reachable_sampling_snaps_blocked_start()
    test_one_step_reachable_output_if_present()
    print("Stage 4A-3.6 reachable candidate sampling tests passed.")
    print("connected_component: ok")
    print("nearest_traversable_cell: ok")
    print("reachable_frontier_mask: ok")
    print("reachable_sampling_component_filter: ok")
    print("snap_start_to_traversable: ok")
    print("unknown_traversable: no")
    print("target_or_ground_truth_fields: none")
    print("rl_optimizer_bc_il_training_run: no")


if __name__ == "__main__":
    main()
