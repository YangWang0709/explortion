#!/usr/bin/env python3
"""Smoke tests for the Stage 4A-3.5 observed-free A* planner."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astar_planner import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    astar_2d,
    build_traversability_grid,
    connected_component_from_start,
    nearest_traversable_cell,
    path_length_m,
    summarize_traversability,
)

MEDIUM_OBSERVED = Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke/observed_state_final.npy")


def test_astar_empty_grid() -> None:
    traversable = np.ones((12, 12), dtype=bool)
    result = astar_2d(traversable, (0, 0), (6, 8), allow_diagonal=True)
    assert result["reachable"] is True, result
    assert result["path"][0] == (0, 0)
    assert result["path"][-1] == (6, 8)
    assert result["cost_cells"] > 0.0
    assert path_length_m(result["path"], 0.1) > 0.0


def test_astar_avoids_obstacle() -> None:
    traversable = np.ones((10, 10), dtype=bool)
    traversable[4, :] = False
    traversable[4, 8] = True
    result = astar_2d(traversable, (1, 1), (8, 8), allow_diagonal=True)
    assert result["reachable"] is True, result
    assert (4, 8) in result["path"], result["path"]
    assert all(traversable[p] for p in result["path"])


def test_astar_reports_unreachable() -> None:
    traversable = np.ones((10, 10), dtype=bool)
    traversable[4, :] = False
    result = astar_2d(traversable, (1, 1), (8, 8), allow_diagonal=True)
    assert result["reachable"] is False, result
    assert result["reason"] == "no_path"
    assert result["path"] == []


def test_connected_component_and_snap() -> None:
    traversable = np.zeros((8, 8), dtype=bool)
    traversable[1:4, 1:4] = True
    traversable[5:7, 5:7] = True
    component = connected_component_from_start(traversable, (2, 2), allow_diagonal=True)
    assert component["start_valid"] is True, component
    assert component["reachable_count"] == 9, component
    assert bool(component["reachable_mask"][1, 1])
    assert not bool(component["reachable_mask"][5, 5])

    invalid = connected_component_from_start(traversable, (0, 0), allow_diagonal=True)
    assert invalid["start_valid"] is False, invalid
    assert invalid["reason"] == "start_not_traversable"

    snap = nearest_traversable_cell(traversable, (0, 0), max_radius_cells=3)
    assert snap["found"] is True, snap
    assert snap["cell"] == [1, 1], snap
    assert snap["distance_cells"] > 0.0


def test_traversability_synthetic() -> None:
    observed = np.full((20, 20, 20), UNKNOWN, dtype=np.int8)
    observed[:, :, 9:15] = FREE
    observed[9:11, :, 10:13] = OCCUPIED
    trav = build_traversability_grid(
        observed,
        voxel_size=0.1,
        robot_height_m=1.2,
        clearance_height_m=0.6,
        robot_radius_m=0.0,
    )
    summary = summarize_traversability(trav)
    assert summary["traversable_count"] > 0, summary
    assert summary["blocked_count"] == 40, summary
    assert trav["traversable"].shape == (20, 20)
    assert trav["traversable"][0, 0]
    assert not trav["traversable"][9, 0]


def test_medium_observed_map_has_traversable_cells() -> dict[str, int | float]:
    assert MEDIUM_OBSERVED.exists(), f"missing medium observed map: {MEDIUM_OBSERVED}"
    observed = np.load(MEDIUM_OBSERVED)
    trav = build_traversability_grid(
        observed,
        voxel_size=0.1,
        robot_height_m=1.2,
        clearance_height_m=0.6,
        robot_radius_m=0.2,
    )
    summary = summarize_traversability(trav)
    assert observed.shape == (120, 120, 30), observed.shape
    assert summary["traversable_count"] > 0, summary
    assert summary["blocked_count"] > 0, summary
    assert summary["unknown_count"] > 0, summary
    return summary


def main() -> None:
    test_astar_empty_grid()
    test_astar_avoids_obstacle()
    test_astar_reports_unreachable()
    test_connected_component_and_snap()
    test_traversability_synthetic()
    medium_summary = test_medium_observed_map_has_traversable_cells()
    print("Stage 4A-3.5 A* planner tests passed.")
    print("astar_empty_grid: ok")
    print("astar_avoids_obstacle: ok")
    print("astar_unreachable: ok")
    print("connected_component_and_snap: ok")
    print("traversability_synthetic: ok")
    print(
        "medium_traversability: "
        f"traversable={medium_summary['traversable_count']} "
        f"blocked={medium_summary['blocked_count']} "
        f"unknown={medium_summary['unknown_count']} "
        f"ratio={medium_summary['traversable_ratio']:.6f}"
    )


if __name__ == "__main__":
    main()
