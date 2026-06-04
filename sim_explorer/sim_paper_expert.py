#!/usr/bin/env python3
"""One-step simulator observed-map expert scorer.

Axis convention:
- observed_state[i, j, k] uses x, y, z axis order.
- i maps to world x, j maps to world y, k maps to world z.
- grid_to_world returns voxel centers:
  x = x_min + (i + 0.5) * voxel_size, and likewise for y/z.

This module keeps the simulator map measured-only. It does not call SSCNet,
does not use NYU targets or ground truth, and never writes predictions into
observed_state. Stage 4A-5.1 may pass a read-only simulator prediction layer,
but that layer is used only for information gain.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from astar_planner import (
    astar_2d,
    build_traversability_grid,
    connected_component_from_start,
    frontier_reachable_candidate_mask,
    nearest_traversable_cell,
    path_length_m,
    summarize_traversability,
)

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "x": (-4.0, 4.0),
    "y": (-4.0, 4.0),
    "z": (0.0, 3.0),
}

FEATURE_NAMES = [
    "gain_exp",
    "gain_sc",
    "gain_hybrid",
    "gain_occ",
    "gain_conf",
    "path_cost",
    "utility_exp",
    "utility_sc",
    "utility_hybrid",
    "utility_occ",
    "utility_conf",
    "final_score",
    "visible_count",
    "measured_visible_count",
    "predicted_unmeasured_visible_count",
    "frontier_distance",
    "frontier_count_visible",
    "astar_reachable",
    "astar_path_length_m",
    "astar_num_expanded",
    "weighted_gain_sc",
    "gain_hybrid_weighted",
    "utility_hybrid_weighted",
    "sc_gain_weight",
    "sc_gain_cap_value",
    "raw_gain_sc",
    "effective_gain_sc",
    "gain_hybrid_effective",
    "utility_effective_sc",
    "utility_hybrid_effective",
    "sc_selected_voxel_count",
    "base_exp_utility",
    "final_score_decoupled_sc",
]

SC_GAIN_FORMULAS = (
    "raw_count",
    "occupied_only",
    "occupied_margin",
    "confidence_weighted",
    "entropy_weighted",
    "calibrated_occupied",
    "novelty_discounted",
)
SC_COUNT_MODES = ("raw_count", "selective")

RAYCAST_MODE_OBSERVED_CONSERVATIVE = "observed_conservative_unknown_blocking"
PREDICTION_MODE_EMPTY = "empty"
PREDICTION_MODE_SIM_NPZ = "sim_npz"
PREDICTION_MODES = (PREDICTION_MODE_EMPTY, PREDICTION_MODE_SIM_NPZ)
STRICT_NO_PREDICTION_WRITE_NOTE = (
    "The simulator expert uses measured-only observed_state; prediction is "
    "read-only, may affect information gain only, and is never written into "
    "observed_state."
)
NO_TARGET_OR_GT_NOTE = (
    "No NYU target_lr/target_hr, scene ground truth, or simulator ground truth "
    "is used for expert scoring."
)
TREE_LIMITATION_NOTE = (
    "Per-candidate gain/cost expert; Stage 4A-3.5 can use observed-free A* "
    "for path-cost scoring, but still has no full RRT tree utility and no "
    "physical path execution."
)


class PredictionLayerLike(Protocol):
    def shape(self) -> tuple[int, int, int]:
        ...

    def is_predicted(self, index: tuple[int, int, int], tau: float = 0.1) -> bool:
        ...

    def is_predicted_occupied(self, index: tuple[int, int, int], tau: float = 0.1) -> bool:
        ...

    def get_occupied_prob(self, index: tuple[int, int, int]) -> float:
        ...

    def get_confidence(self, index: tuple[int, int, int]) -> float:
        ...

    def get_free_prob(self, index: tuple[int, int, int]) -> float:
        ...


def normalize_bounds(bounds: dict[str, Any] | None = None) -> dict[str, tuple[float, float]]:
    raw = DEFAULT_BOUNDS if bounds is None else bounds
    return {axis: (float(raw[axis][0]), float(raw[axis][1])) for axis in ("x", "y", "z")}


def grid_shape_from_bounds(bounds: dict[str, Any], voxel_size: float) -> tuple[int, int, int]:
    normalized = normalize_bounds(bounds)
    return tuple(
        int(round((normalized[axis][1] - normalized[axis][0]) / float(voxel_size)))
        for axis in ("x", "y", "z")
    )


def world_to_grid(
    world_xyz: tuple[float, float, float] | list[float] | np.ndarray,
    bounds: dict[str, Any] | None,
    voxel_size: float,
    shape: tuple[int, int, int] | None = None,
    clip: bool = False,
) -> tuple[int, int, int]:
    """Convert world xyz to observed_state[i, j, k] index.

    By default the function returns the mathematical floor index and does not
    hide out-of-bounds positions. Set clip=True when a caller explicitly wants
    to clamp a pose to an existing grid.
    """

    normalized = normalize_bounds(bounds)
    mins = np.array([normalized["x"][0], normalized["y"][0], normalized["z"][0]], dtype=np.float64)
    idx = np.floor((np.asarray(world_xyz, dtype=np.float64) - mins) / float(voxel_size)).astype(np.int64)
    if clip:
        if shape is None:
            shape = grid_shape_from_bounds(normalized, voxel_size)
        idx = np.clip(idx, 0, np.asarray(shape, dtype=np.int64) - 1)
    return int(idx[0]), int(idx[1]), int(idx[2])


def grid_to_world(
    grid_ijk: tuple[int, int, int] | list[int] | np.ndarray,
    bounds: dict[str, Any] | None,
    voxel_size: float,
) -> tuple[float, float, float]:
    """Convert observed_state[i, j, k] index to voxel-center world xyz."""

    normalized = normalize_bounds(bounds)
    mins = np.array([normalized["x"][0], normalized["y"][0], normalized["z"][0]], dtype=np.float64)
    xyz = mins + (np.asarray(grid_ijk, dtype=np.float64) + 0.5) * float(voxel_size)
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


class EmptyPredictionLayer:
    """Stage 4A-2 placeholder prediction layer.

    It exposes the same small query surface the scorer needs, while always
    reporting no prediction. This keeps the simulator observed-map integration
    separate from SSCNet, which is deferred to a later stage.
    """

    def __init__(self, shape: tuple[int, int, int]):
        self._shape = tuple(int(v) for v in shape)

    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def is_predicted(self, index: tuple[int, int, int], tau: float = 0.1) -> bool:
        return False

    def is_predicted_occupied(self, index: tuple[int, int, int], tau: float = 0.1) -> bool:
        return False

    def is_predicted_free(self, index: tuple[int, int, int], tau: float = 0.1) -> bool:
        return False

    def get_confidence(self, index: tuple[int, int, int]) -> float:
        return 0.0

    def get_occupied_prob(self, index: tuple[int, int, int]) -> float:
        return 0.5

    def get_free_prob(self, index: tuple[int, int, int]) -> float:
        return 0.5

    def get_prediction_gain(self, index: tuple[int, int, int], tau: float = 0.1) -> float:
        return 0.0


@dataclass
class SimCandidateView:
    id: int
    grid_position: tuple[int, int, int]
    world_position: tuple[float, float, float]
    yaw: float
    valid: bool = True
    invalid_reason: str = ""
    candidate_source: str = "frontier"

    gain_exp: float = 0.0
    gain_sc: float = 0.0
    gain_hybrid: float = 0.0
    gain_occ: float = 0.0
    gain_conf: float = 0.0

    path_cost: float = 0.0
    utility_exp: float = 0.0
    utility_sc: float = 0.0
    utility_hybrid: float = 0.0
    utility_occ: float = 0.0
    utility_conf: float = 0.0
    utility_hybrid_weighted: float = 0.0
    final_score: float = 0.0
    weighted_gain_sc: float = 0.0
    gain_hybrid_weighted: float = 0.0
    raw_gain_sc: float = 0.0
    effective_gain_sc: float = 0.0
    gain_hybrid_effective: float = 0.0
    utility_effective_sc: float = 0.0
    utility_hybrid_effective: float = 0.0
    sc_gain_weight: float = 1.0
    sc_gain_cap_value: float = -1.0
    base_exp_utility: float = 0.0
    final_score_decoupled_sc: float = 0.0
    sc_gain_formula: str = "raw_count"
    sc_occ_threshold: float = 0.7
    sc_conf_threshold: float = 0.3
    sc_count_mode: str = "raw_count"

    visible_count: int = 0
    measured_visible_count: int = 0
    predicted_unmeasured_visible_count: int = 0
    sc_selected_voxel_count: int = 0
    frontier_distance: float = 0.0
    frontier_count_visible: int = 0

    astar_reachable: bool = False
    astar_path_length_m: float = 0.0
    astar_num_expanded: int = 0
    astar_path_xy: list[tuple[int, int]] = field(default_factory=list)


def _in_bounds(index: tuple[int, int, int], shape: tuple[int, int, int]) -> bool:
    return (
        0 <= index[0] < shape[0]
        and 0 <= index[1] < shape[1]
        and 0 <= index[2] < shape[2]
    )


def _validate_observed_state(observed_state: np.ndarray) -> np.ndarray:
    observed_state = np.asarray(observed_state)
    if observed_state.ndim != 3:
        raise ValueError(f"observed_state must be 3D, got shape {observed_state.shape}")
    allowed = np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED])
    if not bool(np.all(allowed)):
        bad = np.unique(observed_state[~allowed])
        raise ValueError(f"observed_state contains unsupported state values: {bad.tolist()}")
    return observed_state


def _neighbor_mask(source_mask: np.ndarray) -> np.ndarray:
    neighbors = np.zeros_like(source_mask, dtype=bool)
    neighbors[1:, :, :] |= source_mask[:-1, :, :]
    neighbors[:-1, :, :] |= source_mask[1:, :, :]
    neighbors[:, 1:, :] |= source_mask[:, :-1, :]
    neighbors[:, :-1, :] |= source_mask[:, 1:, :]
    neighbors[:, :, 1:] |= source_mask[:, :, :-1]
    neighbors[:, :, :-1] |= source_mask[:, :, 1:]
    return neighbors


def detect_frontier_voxels(observed_state: np.ndarray) -> np.ndarray:
    """Return UNKNOWN voxels adjacent to at least one observed FREE voxel."""

    observed_state = _validate_observed_state(observed_state)
    unknown_mask = observed_state == UNKNOWN
    free_neighbor_mask = _neighbor_mask(observed_state == FREE)
    return np.argwhere(unknown_mask & free_neighbor_mask).astype(np.int32, copy=False)


def detect_frontier_adjacent_free_voxels(observed_state: np.ndarray) -> np.ndarray:
    """Return FREE voxels adjacent to at least one UNKNOWN voxel."""

    observed_state = _validate_observed_state(observed_state)
    free_mask = observed_state == FREE
    unknown_neighbor_mask = _neighbor_mask(observed_state == UNKNOWN)
    return np.argwhere(free_mask & unknown_neighbor_mask).astype(np.int32, copy=False)


def frontier_adjacent_free_xy_mask(observed_state: np.ndarray) -> np.ndarray:
    """Project frontier-adjacent FREE voxels into a 2D x/y candidate mask."""

    observed_state = _validate_observed_state(observed_state)
    adjacent_free = detect_frontier_adjacent_free_voxels(observed_state)
    mask = np.zeros(observed_state.shape[:2], dtype=bool)
    if len(adjacent_free):
        mask[adjacent_free[:, 0], adjacent_free[:, 1]] = True
    return mask


def _traversable_grid_from_result(traversability: dict[str, Any] | np.ndarray) -> np.ndarray:
    if isinstance(traversability, dict):
        return np.asarray(traversability["traversable"], dtype=bool)
    return np.asarray(traversability, dtype=bool)


def compute_reachable_frontier_candidate_cells(
    observed_state: np.ndarray,
    traversability: dict[str, Any] | np.ndarray,
    current_xy: tuple[int, int] | list[int] | np.ndarray,
    frontier_adjacent_free_mask: np.ndarray,
    max_snap_radius_cells: int = 5,
    snap_start_to_traversable: bool = True,
    allow_diagonal: bool = True,
) -> dict[str, Any]:
    """Return a 2D candidate mask inside the current reachable FREE component."""

    observed_state = _validate_observed_state(observed_state)
    traversable = _traversable_grid_from_result(traversability)
    if traversable.shape != observed_state.shape[:2]:
        raise ValueError(f"traversability shape {traversable.shape} differs from observed_state {observed_state.shape[:2]}")
    frontier_mask = np.asarray(frontier_adjacent_free_mask, dtype=bool)
    if frontier_mask.shape != traversable.shape:
        raise ValueError(
            f"frontier_adjacent_free_mask shape {frontier_mask.shape} differs from traversability {traversable.shape}"
        )

    current = np.asarray(current_xy, dtype=np.int64)
    if current.shape[0] < 2:
        raise ValueError(f"current_xy must contain at least 2 values, got {current_xy}")
    current_tuple = (int(current[0]), int(current[1]))
    shape = tuple(int(v) for v in traversable.shape)
    start_in_bounds = 0 <= current_tuple[0] < shape[0] and 0 <= current_tuple[1] < shape[1]
    start_traversable = bool(start_in_bounds and traversable[current_tuple])
    snapped = False
    snapped_xy = current_tuple
    snap_distance = 0.0 if start_traversable else math.inf
    snap_result: dict[str, Any] = {
        "found": start_traversable,
        "cell": [int(current_tuple[0]), int(current_tuple[1])] if start_traversable else None,
        "distance_cells": 0.0 if start_traversable else math.inf,
        "reason": "start_is_traversable" if start_traversable else "snap_not_requested",
    }

    if not start_traversable:
        if bool(snap_start_to_traversable):
            snap_result = nearest_traversable_cell(
                traversable,
                current_tuple,
                max_radius_cells=int(max_snap_radius_cells),
            )
            if bool(snap_result.get("found", False)):
                cell = snap_result["cell"]
                assert cell is not None
                snapped_xy = (int(cell[0]), int(cell[1]))
                snap_distance = float(snap_result.get("distance_cells", math.inf))
                snapped = True
            else:
                empty = np.zeros_like(traversable, dtype=bool)
                return {
                    "candidate_mask": empty,
                    "reachable_mask": empty,
                    "current_xy": [int(current_tuple[0]), int(current_tuple[1])],
                    "snapped_current_xy": None,
                    "snapped": False,
                    "start_traversable": False,
                    "start_in_bounds": bool(start_in_bounds),
                    "snap_distance_cells": None,
                    "snap_result": snap_result,
                    "reachable_count": 0,
                    "reachable_frontier_adjacent_count": 0,
                    "reachable_free_fallback_count": 0,
                    "candidate_source": "none",
                    "candidate_count_available": 0,
                    "component_reason": "start_not_traversable",
                    "reason": "no_reachable_free_component",
                }
        else:
            empty = np.zeros_like(traversable, dtype=bool)
            return {
                "candidate_mask": empty,
                "reachable_mask": empty,
                "current_xy": [int(current_tuple[0]), int(current_tuple[1])],
                "snapped_current_xy": None,
                "snapped": False,
                "start_traversable": False,
                "start_in_bounds": bool(start_in_bounds),
                "snap_distance_cells": None,
                "snap_result": snap_result,
                "reachable_count": 0,
                "reachable_frontier_adjacent_count": 0,
                "reachable_free_fallback_count": 0,
                "candidate_source": "none",
                "candidate_count_available": 0,
                "component_reason": "start_not_traversable",
                "reason": "no_reachable_free_component",
            }

    mask_result = frontier_reachable_candidate_mask(
        traversable=traversable,
        start_xy=snapped_xy,
        frontier_adjacent_free_mask=frontier_mask,
        allow_diagonal=bool(allow_diagonal),
    )
    reachable_mask = np.asarray(mask_result["reachable_mask"], dtype=bool)
    reachable_count = int(mask_result["reachable_count"])
    reachable_frontier_count = int(mask_result["candidate_count"])
    if reachable_count <= 0:
        empty = np.zeros_like(traversable, dtype=bool)
        return {
            "candidate_mask": empty,
            "reachable_mask": reachable_mask,
            "current_xy": [int(current_tuple[0]), int(current_tuple[1])],
            "snapped_current_xy": [int(snapped_xy[0]), int(snapped_xy[1])],
            "snapped": bool(snapped),
            "start_traversable": bool(start_traversable),
            "start_in_bounds": bool(start_in_bounds),
            "snap_distance_cells": None if not np.isfinite(snap_distance) else float(snap_distance),
            "snap_result": snap_result,
            "reachable_count": 0,
            "reachable_frontier_adjacent_count": 0,
            "reachable_free_fallback_count": 0,
            "candidate_source": "none",
            "candidate_count_available": 0,
            "component_reason": str(mask_result["reason"]),
            "reason": "no_reachable_free_component",
        }

    if reachable_frontier_count > 0:
        candidate_mask = np.asarray(mask_result["candidate_mask"], dtype=bool)
        candidate_source = "reachable_frontier"
        reachable_free_fallback_count = 0
    else:
        candidate_mask = reachable_mask.astype(bool, copy=True)
        candidate_source = "reachable_free_fallback"
        reachable_free_fallback_count = int(np.count_nonzero(candidate_mask))

    excluded_start_cell = False
    if 0 <= snapped_xy[0] < candidate_mask.shape[0] and 0 <= snapped_xy[1] < candidate_mask.shape[1]:
        if bool(candidate_mask[snapped_xy]) and int(np.count_nonzero(candidate_mask)) > 1:
            candidate_mask[snapped_xy] = False
            excluded_start_cell = True

    return {
        "candidate_mask": candidate_mask,
        "reachable_mask": reachable_mask,
        "current_xy": [int(current_tuple[0]), int(current_tuple[1])],
        "snapped_current_xy": [int(snapped_xy[0]), int(snapped_xy[1])],
        "snapped": bool(snapped),
        "start_traversable": bool(start_traversable),
        "start_in_bounds": bool(start_in_bounds),
        "snap_distance_cells": None if not np.isfinite(snap_distance) else float(snap_distance),
        "snap_result": snap_result,
        "reachable_count": int(reachable_count),
        "reachable_frontier_adjacent_count": int(reachable_frontier_count),
        "reachable_free_fallback_count": int(reachable_free_fallback_count),
        "excluded_start_cell_from_candidates": bool(excluded_start_cell),
        "candidate_source": candidate_source,
        "candidate_count_available": int(np.count_nonzero(candidate_mask)),
        "component_reason": str(mask_result["reason"]),
        "reason": "ok",
    }


def _has_free_neighbor(observed_state: np.ndarray, index: tuple[int, int, int]) -> bool:
    i, j, k = index
    shape = observed_state.shape
    for di, dj, dk in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        n = (i + di, j + dj, k + dk)
        if _in_bounds(n, shape) and observed_state[n] == FREE:
            return True
    return False


def _wrap_angle(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _yaw_from_delta(delta_xy: np.ndarray, fallback: float = 0.0) -> float:
    if float(np.linalg.norm(delta_xy)) <= 1e-6:
        return float(fallback)
    return float(math.atan2(float(delta_xy[1]), float(delta_xy[0])))


def _frontier_target_for_candidate(
    candidate_grid: np.ndarray,
    frontier_voxels: np.ndarray,
    radius_voxels: float = 12.0,
) -> tuple[np.ndarray | None, float]:
    if len(frontier_voxels) == 0:
        return None, float("inf")

    deltas = frontier_voxels.astype(np.float32) - candidate_grid.astype(np.float32)
    distances = np.linalg.norm(deltas, axis=1)
    nearest_distance = float(np.min(distances))
    near = frontier_voxels[distances <= float(radius_voxels)]
    if len(near) > 0:
        return near.astype(np.float32).mean(axis=0), nearest_distance

    nearest_count = min(16, len(frontier_voxels))
    nearest_ids = np.argpartition(distances, nearest_count - 1)[:nearest_count]
    return frontier_voxels[nearest_ids].astype(np.float32).mean(axis=0), nearest_distance


def _nearest_free_z_for_xy(observed_state: np.ndarray, xy: tuple[int, int], preferred_k: int) -> int | None:
    i, j = int(xy[0]), int(xy[1])
    if not (0 <= i < observed_state.shape[0] and 0 <= j < observed_state.shape[1]):
        return None
    free_z = np.flatnonzero(observed_state[i, j, :] == FREE)
    if free_z.size == 0:
        return None
    distances = np.abs(free_z.astype(np.int64) - int(preferred_k))
    best_idx = int(np.argmin(distances))
    return int(free_z[best_idx])


def _make_candidate_views_from_grid_positions(
    observed_state: np.ndarray,
    current_grid: tuple[int, int, int],
    bounds: dict[str, Any],
    voxel_size: float,
    selected: list[tuple[int, int, int]],
    candidate_sources: list[str],
) -> list[SimCandidateView]:
    frontier_voxels = detect_frontier_voxels(observed_state)
    current_arr = np.asarray(current_grid, dtype=np.float32)
    candidates: list[SimCandidateView] = []
    for candidate_id, grid_position in enumerate(selected):
        source = candidate_sources[candidate_id] if candidate_id < len(candidate_sources) else "frontier"
        if observed_state[grid_position] != FREE:
            candidates.append(
                SimCandidateView(
                    id=candidate_id,
                    grid_position=grid_position,
                    world_position=grid_to_world(grid_position, bounds, voxel_size),
                    yaw=0.0,
                    valid=False,
                    invalid_reason="candidate voxel is not FREE",
                    candidate_source=source,
                )
            )
            continue

        grid_arr = np.asarray(grid_position, dtype=np.float32)
        target, nearest_frontier_distance_vox = _frontier_target_for_candidate(grid_arr, frontier_voxels)
        if target is not None:
            yaw = _yaw_from_delta(target[:2] - grid_arr[:2])
        else:
            yaw = _yaw_from_delta(grid_arr[:2] - current_arr[:2], fallback=0.0)
            nearest_frontier_distance_vox = 0.0

        candidate = SimCandidateView(
            id=candidate_id,
            grid_position=grid_position,
            world_position=grid_to_world(grid_position, bounds, voxel_size),
            yaw=float(yaw),
            candidate_source=source,
            frontier_distance=float(nearest_frontier_distance_vox * float(voxel_size)),
        )
        candidates.append(candidate)

    return candidates


def sample_candidate_views_from_frontiers(
    observed_state: np.ndarray,
    current_grid: tuple[int, int, int],
    bounds: dict[str, Any],
    voxel_size: float,
    num_candidates: int = 64,
    seed: int = 0,
    candidate_sampling_mode: str = "frontier",
    traversability: dict[str, Any] | np.ndarray | None = None,
    max_snap_radius_cells: int = 5,
    snap_start_to_traversable: bool = True,
    astar_allow_diagonal: bool = True,
    return_diagnostics: bool = False,
) -> list[SimCandidateView] | tuple[list[SimCandidateView], dict[str, Any]]:
    """Sample candidate viewpoints from measured free frontier-adjacent cells.

    The sampler uses only observed_state. It does not inspect simulator scene
    geometry, prediction, target labels, or ground truth.
    """

    observed_state = _validate_observed_state(observed_state)
    if int(num_candidates) <= 0:
        raise ValueError("num_candidates must be positive")

    rng = np.random.default_rng(seed)
    sampling_mode = str(candidate_sampling_mode)
    if sampling_mode not in ("frontier", "reachable_frontier"):
        raise ValueError("candidate_sampling_mode must be one of: frontier, reachable_frontier")

    sampling_info: dict[str, Any] = {
        "candidate_sampling_mode": sampling_mode,
        "candidate_source": "frontier",
        "current_xy": [int(current_grid[0]), int(current_grid[1])],
        "snapped_current_xy": [int(current_grid[0]), int(current_grid[1])],
        "snapped": False,
        "start_traversable": None,
        "snap_distance_cells": 0.0,
        "reachable_count": None,
        "reachable_frontier_adjacent_count": None,
        "reachable_free_fallback_count": 0,
        "candidate_count_available": None,
        "reachable_mask": None,
        "candidate_mask": None,
        "reason": "frontier_sampling",
    }

    adjacent_free = detect_frontier_adjacent_free_voxels(observed_state)
    all_free = np.argwhere(observed_state == FREE).astype(np.int32, copy=False)
    if len(all_free) == 0:
        raise ValueError("No FREE voxels available for candidate sampling")

    selected: list[tuple[int, int, int]] = []
    candidate_sources: list[str] = []
    selected_set: set[tuple[int, int, int]] = set()

    if sampling_mode == "reachable_frontier":
        if traversability is None:
            raise ValueError("traversability is required when candidate_sampling_mode='reachable_frontier'")
        frontier_xy = frontier_adjacent_free_xy_mask(observed_state)
        reachable_info = compute_reachable_frontier_candidate_cells(
            observed_state=observed_state,
            traversability=traversability,
            current_xy=current_grid[:2],
            frontier_adjacent_free_mask=frontier_xy,
            max_snap_radius_cells=int(max_snap_radius_cells),
            snap_start_to_traversable=bool(snap_start_to_traversable),
            allow_diagonal=bool(astar_allow_diagonal),
        )
        sampling_info.update(reachable_info)
        sampling_info["candidate_sampling_mode"] = sampling_mode
        candidate_mask = np.asarray(reachable_info["candidate_mask"], dtype=bool)
        xy_pool = np.argwhere(candidate_mask).astype(np.int32, copy=False)
        if len(xy_pool) > 0:
            take = min(int(num_candidates), len(xy_pool))
            choice = rng.choice(len(xy_pool), size=take, replace=False)
            preferred_k = int(np.clip(int(current_grid[2]), 0, observed_state.shape[2] - 1))
            source = str(reachable_info["candidate_source"])
            for row in xy_pool[choice]:
                xy = (int(row[0]), int(row[1]))
                k = _nearest_free_z_for_xy(observed_state, xy, preferred_k)
                if k is None:
                    continue
                item = (int(xy[0]), int(xy[1]), int(k))
                if item in selected_set:
                    continue
                selected.append(item)
                selected_set.add(item)
                candidate_sources.append(source)
        sampling_info["sampled_candidate_count"] = int(len(selected))
    else:
        if len(adjacent_free) > 0:
            take = min(int(num_candidates), len(adjacent_free))
            choice = rng.choice(len(adjacent_free), size=take, replace=False)
            for row in adjacent_free[choice]:
                item = tuple(int(v) for v in row)
                selected.append(item)
                selected_set.add(item)
                candidate_sources.append("frontier_adjacent")

        remaining = int(num_candidates) - len(selected)
        if remaining > 0:
            supplement_pool = [
                tuple(int(v) for v in row) for row in all_free if tuple(int(v) for v in row) not in selected_set
            ]
            if supplement_pool:
                take = min(remaining, len(supplement_pool))
                choice = rng.choice(len(supplement_pool), size=take, replace=False)
                for idx in np.atleast_1d(choice):
                    item = supplement_pool[int(idx)]
                    selected.append(item)
                    selected_set.add(item)
                    candidate_sources.append("free_supplement")
        sampling_info["sampled_candidate_count"] = int(len(selected))
        sampling_info["candidate_count_available"] = int(len(adjacent_free) + max(0, len(all_free) - len(adjacent_free)))

    candidates = _make_candidate_views_from_grid_positions(
        observed_state=observed_state,
        current_grid=current_grid,
        bounds=bounds,
        voxel_size=voxel_size,
        selected=selected,
        candidate_sources=candidate_sources,
    )

    if return_diagnostics:
        return candidates, sampling_info
    return candidates


def _sample_fov_directions(
    yaw_center: float,
    num_yaw: int,
    num_pitch: int,
    fov_yaw_deg: float,
    fov_pitch_deg: float,
) -> list[np.ndarray]:
    if int(num_yaw) <= 0 or int(num_pitch) <= 0:
        raise ValueError("num_yaw and num_pitch must be positive")

    if int(num_yaw) == 1:
        yaw_offsets = np.array([0.0], dtype=np.float64)
    else:
        yaw_offsets = np.deg2rad(np.linspace(-0.5 * float(fov_yaw_deg), 0.5 * float(fov_yaw_deg), int(num_yaw)))
    if int(num_pitch) == 1:
        pitch_offsets = np.array([0.0], dtype=np.float64)
    else:
        pitch_offsets = np.deg2rad(
            np.linspace(-0.5 * float(fov_pitch_deg), 0.5 * float(fov_pitch_deg), int(num_pitch))
        )

    directions: list[np.ndarray] = []
    for pitch in pitch_offsets:
        cos_pitch = math.cos(float(pitch))
        sin_pitch = math.sin(float(pitch))
        for yaw_offset in yaw_offsets:
            yaw = float(yaw_center) + float(yaw_offset)
            direction = np.array(
                [
                    cos_pitch * math.cos(yaw),
                    cos_pitch * math.sin(yaw),
                    sin_pitch,
                ],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(direction))
            if norm > 0.0:
                directions.append(direction / norm)
    return directions


def raycast_visible_voxels_observed(
    candidate: SimCandidateView,
    observed_state: np.ndarray,
    max_range_voxels: int = 50,
    num_yaw: int = 32,
    num_pitch: int = 7,
    fov_yaw_deg: float = 90.0,
    fov_pitch_deg: float = 60.0,
) -> list[tuple[int, int, int]]:
    """Raycast visibility through observed_state only.

    OCCUPIED blocks. FREE passes through. UNKNOWN is counted as visible and
    then blocks conservatively. Prediction never participates in blocking.
    """

    observed_state = _validate_observed_state(observed_state)
    if not candidate.valid:
        return []
    if not _in_bounds(candidate.grid_position, observed_state.shape):
        raise ValueError(f"Candidate out of bounds: {candidate.grid_position}")
    if observed_state[candidate.grid_position] != FREE:
        return []

    origin = np.asarray(candidate.grid_position, dtype=np.float64) + 0.5
    start_voxel = tuple(int(v) for v in candidate.grid_position)
    visible: set[tuple[int, int, int]] = set()
    directions = _sample_fov_directions(
        yaw_center=candidate.yaw,
        num_yaw=num_yaw,
        num_pitch=num_pitch,
        fov_yaw_deg=fov_yaw_deg,
        fov_pitch_deg=fov_pitch_deg,
    )
    step_size = 0.5

    for direction in directions:
        distance = step_size
        last_voxel: tuple[int, int, int] | None = None
        while distance <= float(max_range_voxels):
            point = origin + direction * distance
            voxel = tuple(int(math.floor(float(v))) for v in point)
            if not _in_bounds(voxel, observed_state.shape):
                break
            if voxel == last_voxel:
                distance += step_size
                continue
            last_voxel = voxel
            if voxel == start_voxel:
                distance += step_size
                continue

            visible.add(voxel)
            state = observed_state[voxel]
            if state == OCCUPIED:
                break
            if state == UNKNOWN:
                break
            distance += step_size

    return sorted(visible)


def load_calibration_table(path: str | Path | None) -> dict[str, Any] | None:
    """Load a Stage 4A-6.4 calibration table for read-only scoring."""

    if path is None or str(path) == "":
        return None
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(f"calibration table not found: {table_path}")
    with table_path.open("r", encoding="utf-8") as handle:
        table = json.load(handle)
    if "occupied_prob_bins" not in table and "calibration_bins" not in table:
        raise ValueError(f"calibration table {table_path} missing occupied_prob_bins/calibration_bins")
    return table


def _calibration_bins(table: dict[str, Any] | None, key: str = "occupied_prob_bins") -> list[dict[str, Any]]:
    if table is None:
        return []
    rows = table.get(key)
    if rows is None and key == "occupied_prob_bins":
        rows = table.get("calibration_bins")
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise ValueError(f"calibration table field {key} must be a list")
    return [dict(row) for row in rows]


def calibrated_value_from_bins(
    value: float,
    rows: list[dict[str, Any]],
    value_min_key: str = "prob_min",
    value_max_key: str = "prob_max",
    output_key: str = "empirical_occupied_rate",
    default: float = 0.0,
) -> float:
    """Lookup a scalar in reliability bins.

    The table is generated from delayed sensor observations and is treated as a
    fixed diagnostic artifact. Runtime scoring never reads future maps.
    """

    if not rows:
        return float(default)
    x = float(np.clip(value, 0.0, 1.0))
    fallback: float | None = None
    for row in rows:
        if row.get(output_key) is None:
            continue
        try:
            lo = float(row[value_min_key])
            hi = float(row[value_max_key])
            y = float(row[output_key])
        except (KeyError, TypeError, ValueError):
            continue
        if fallback is None:
            fallback = y
        if (x >= lo and x < hi) or (x == 1.0 and x <= hi and hi >= 1.0):
            return float(y)
    return float(default if fallback is None else fallback)


def _binary_entropy(prob: float) -> float:
    p = float(np.clip(prob, 1.0e-6, 1.0 - 1.0e-6))
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def sc_gain_contribution(
    *,
    formula: str,
    occupied_prob: float,
    confidence: float,
    free_neighbor: bool,
    occ_threshold: float,
    conf_threshold: float,
    calibration_table: dict[str, Any] | None = None,
) -> float:
    """Return the selective prediction gain for one predicted-unmeasured voxel."""

    formula = str(formula)
    if formula not in SC_GAIN_FORMULAS:
        raise ValueError(f"sc_gain_formula must be one of: {', '.join(SC_GAIN_FORMULAS)}")
    occ = float(np.clip(occupied_prob, 0.0, 1.0))
    conf = float(np.clip(confidence, 0.0, 1.0))
    occ_threshold = float(occ_threshold)
    conf_threshold = float(conf_threshold)

    if formula == "raw_count":
        return 1.0
    if conf < conf_threshold:
        return 0.0
    if formula == "occupied_only":
        return 1.0 if occ >= occ_threshold else 0.0
    if formula == "occupied_margin":
        if occ < occ_threshold:
            return 0.0
        return float(max(0.0, occ - 0.5))
    if formula == "confidence_weighted":
        return conf
    if formula == "entropy_weighted":
        return _binary_entropy(occ)
    if formula == "calibrated_occupied":
        if calibration_table is None:
            raise ValueError("sc_gain_formula='calibrated_occupied' requires a calibration_table")
        if occ < occ_threshold:
            return 0.0
        rows = _calibration_bins(calibration_table, "occupied_prob_bins")
        return calibrated_value_from_bins(
            occ,
            rows,
            value_min_key="prob_min",
            value_max_key="prob_max",
            output_key="empirical_occupied_rate",
            default=0.0,
        )
    if formula == "novelty_discounted":
        if occ < occ_threshold:
            return 0.0
        base = max(0.0, occ - 0.5)
        return float(base * (0.25 if bool(free_neighbor) else 1.0))
    raise AssertionError(f"unhandled sc_gain_formula: {formula}")


def compute_paper_gains_for_candidate(
    candidate: SimCandidateView,
    observed_state: np.ndarray,
    prediction_layer: PredictionLayerLike,
    visible_voxels: list[tuple[int, int, int]],
    tau: float = 0.1,
    sc_gain_formula: str = "raw_count",
    sc_occ_threshold: float = 0.7,
    sc_conf_threshold: float = 0.3,
    sc_count_mode: str = "raw_count",
    calibration_table: dict[str, Any] | None = None,
) -> SimCandidateView:
    """Apply paper-style gains to one candidate using measured S and P."""

    observed_state = _validate_observed_state(observed_state)
    sc_gain_formula = str(sc_gain_formula)
    if sc_gain_formula not in SC_GAIN_FORMULAS:
        raise ValueError(f"sc_gain_formula must be one of: {', '.join(SC_GAIN_FORMULAS)}")
    sc_count_mode = str(sc_count_mode)
    if sc_count_mode not in SC_COUNT_MODES:
        raise ValueError(f"sc_count_mode must be one of: {', '.join(SC_COUNT_MODES)}")
    if sc_gain_formula == "calibrated_occupied" and calibration_table is None:
        raise ValueError("calibrated_occupied SC gain requires --calibration_table")
    if tuple(prediction_layer.shape()) != tuple(observed_state.shape):
        raise ValueError(
            f"prediction_layer shape {prediction_layer.shape()} differs from observed_state {observed_state.shape}"
        )

    gain_exp = 0.0
    gain_sc = 0.0
    effective_gain_sc = 0.0
    gain_hybrid = 0.0
    gain_occ = 0.0
    gain_conf = 0.0
    visible_count = 0
    measured_visible_count = 0
    predicted_unmeasured_visible_count = 0
    sc_selected_voxel_count = 0
    frontier_count_visible = 0

    for raw_voxel in visible_voxels:
        voxel = tuple(int(v) for v in raw_voxel)
        if not _in_bounds(voxel, observed_state.shape):
            continue
        visible_count += 1

        in_s = bool(observed_state[voxel] != UNKNOWN)
        in_p = bool(prediction_layer.is_predicted(voxel, tau=tau) and not in_s)
        i_exp = 0.0 if in_s else 1.0
        i_sc = 1.0 if in_p else 0.0
        free_neighbor = False

        if in_s:
            measured_visible_count += 1
        else:
            gain_exp += 1.0
            free_neighbor = _has_free_neighbor(observed_state, voxel)
            if free_neighbor:
                frontier_count_visible += 1

        if in_p:
            gain_sc += 1.0
            predicted_unmeasured_visible_count += 1
            occupied_prob = float(prediction_layer.get_occupied_prob(voxel))
            confidence = float(prediction_layer.get_confidence(voxel))
            contribution = sc_gain_contribution(
                formula=sc_gain_formula,
                occupied_prob=occupied_prob,
                confidence=confidence,
                free_neighbor=free_neighbor,
                occ_threshold=float(sc_occ_threshold),
                conf_threshold=float(sc_conf_threshold),
                calibration_table=calibration_table,
            )
            if contribution > 0.0:
                sc_selected_voxel_count += 1
            effective_gain_sc += float(contribution)
            if prediction_layer.is_predicted_occupied(voxel, tau=tau):
                gain_occ += 1.0
            gain_conf += abs(0.5 - occupied_prob)

        gain_hybrid += i_exp + i_sc

    candidate.gain_exp = float(gain_exp)
    candidate.gain_sc = float(gain_sc)
    candidate.raw_gain_sc = float(gain_sc)
    candidate.effective_gain_sc = float(effective_gain_sc)
    candidate.gain_hybrid = float(gain_hybrid)
    candidate.gain_hybrid_effective = float(gain_exp + effective_gain_sc)
    candidate.gain_occ = float(gain_occ)
    candidate.gain_conf = float(gain_conf)
    candidate.visible_count = int(visible_count)
    candidate.measured_visible_count = int(measured_visible_count)
    candidate.predicted_unmeasured_visible_count = int(predicted_unmeasured_visible_count)
    candidate.sc_selected_voxel_count = int(sc_selected_voxel_count)
    candidate.frontier_count_visible = int(frontier_count_visible)
    candidate.sc_gain_formula = str(sc_gain_formula)
    candidate.sc_occ_threshold = float(sc_occ_threshold)
    candidate.sc_conf_threshold = float(sc_conf_threshold)
    candidate.sc_count_mode = str(sc_count_mode)
    return candidate


def compute_cost_and_score(
    candidate: SimCandidateView,
    current_grid: tuple[int, int, int],
    current_yaw: float,
    gain_mode: str = "hybrid",
    score_gain_mode: str = "hybrid_raw",
    sc_gain_weight: float = 1.0,
    sc_gain_cap: float | None = None,
    voxel_size: float = 0.1,
    v_max: float = 1.0,
    yaw_rate_deg: float = 90.0,
    path_cost_mode: str = "euclidean",
    astar_result: dict[str, Any] | None = None,
) -> SimCandidateView:
    eps = 1e-6
    score_gain_mode = str(score_gain_mode)
    if score_gain_mode not in ("hybrid_raw", "hybrid_weighted", "decoupled_sc"):
        raise ValueError("score_gain_mode must be one of: hybrid_raw, hybrid_weighted, decoupled_sc")
    sc_gain_weight = float(sc_gain_weight)
    if sc_gain_cap is not None:
        sc_gain_cap = float(sc_gain_cap)
        if sc_gain_cap < 0.0:
            raise ValueError("sc_gain_cap must be non-negative when provided")
    selected_gain_sc = float(getattr(candidate, "effective_gain_sc", candidate.gain_sc))
    if str(getattr(candidate, "sc_gain_formula", "raw_count")) == "raw_count" and selected_gain_sc == 0.0:
        selected_gain_sc = float(candidate.gain_sc)
        candidate.effective_gain_sc = selected_gain_sc
    if float(getattr(candidate, "raw_gain_sc", 0.0)) == 0.0 and float(candidate.gain_sc) != 0.0:
        candidate.raw_gain_sc = float(candidate.gain_sc)
    effective_gain_sc = min(selected_gain_sc, sc_gain_cap) if sc_gain_cap is not None else selected_gain_sc
    candidate.sc_gain_weight = sc_gain_weight
    candidate.sc_gain_cap_value = -1.0 if sc_gain_cap is None else float(sc_gain_cap)
    candidate.weighted_gain_sc = float(sc_gain_weight * effective_gain_sc)
    candidate.gain_hybrid_weighted = float(candidate.gain_exp + candidate.weighted_gain_sc)
    candidate.gain_hybrid_effective = float(candidate.gain_exp + selected_gain_sc)

    yaw_rate = math.radians(float(yaw_rate_deg))
    time_yaw = abs(_wrap_angle(float(candidate.yaw) - float(current_yaw))) / max(yaw_rate, eps)

    path_cost_mode = str(path_cost_mode)
    if path_cost_mode == "euclidean":
        distance_vox = float(
            np.linalg.norm(
                np.asarray(candidate.grid_position, dtype=np.float64) - np.asarray(current_grid, dtype=np.float64)
            )
        )
        distance_m = distance_vox * float(voxel_size)
        time_pos = distance_m / max(float(v_max), eps)
        candidate.path_cost = float(time_pos + time_yaw)
        candidate.astar_reachable = False
        candidate.astar_path_length_m = 0.0
        candidate.astar_num_expanded = 0
        candidate.astar_path_xy = []
    elif path_cost_mode == "astar":
        if astar_result is None:
            raise ValueError("astar_result is required when path_cost_mode='astar'")
        candidate.astar_num_expanded = int(astar_result.get("num_expanded", 0))
        if not bool(astar_result.get("reachable", False)):
            reason = str(astar_result.get("reason", "unknown"))
            candidate.valid = False
            candidate.invalid_reason = f"unreachable_astar:{reason}"
            candidate.path_cost = float("inf")
            candidate.astar_reachable = False
            candidate.astar_path_length_m = float("inf")
            candidate.astar_path_xy = []
            candidate.utility_exp = 0.0
            candidate.utility_sc = 0.0
            candidate.utility_hybrid = 0.0
            candidate.utility_occ = 0.0
            candidate.utility_conf = 0.0
            candidate.utility_effective_sc = 0.0
            candidate.utility_hybrid_effective = 0.0
            candidate.utility_hybrid_weighted = 0.0
            candidate.base_exp_utility = 0.0
            candidate.final_score_decoupled_sc = float("-inf")
            candidate.final_score = float("-inf")
            return candidate

        path_xy = [tuple(int(v) for v in xy) for xy in astar_result.get("path", [])]
        distance_m = path_length_m(path_xy, voxel_size=float(voxel_size))
        time_pos = distance_m / max(float(v_max), eps)
        candidate.path_cost = float(time_pos + time_yaw)
        candidate.astar_reachable = True
        candidate.astar_path_length_m = float(distance_m)
        candidate.astar_path_xy = path_xy
    else:
        raise ValueError("path_cost_mode must be one of: euclidean, astar")

    denom = max(candidate.path_cost, eps)
    candidate.utility_exp = float(candidate.gain_exp / denom)
    candidate.base_exp_utility = float(candidate.utility_exp)
    candidate.utility_sc = float(candidate.gain_sc / denom)
    candidate.utility_hybrid = float(candidate.gain_hybrid / denom)
    candidate.utility_occ = float(candidate.gain_occ / denom)
    candidate.utility_conf = float(candidate.gain_conf / denom)
    candidate.utility_effective_sc = float(selected_gain_sc / denom)
    candidate.utility_hybrid_effective = float(candidate.gain_hybrid_effective / denom)
    candidate.utility_hybrid_weighted = float(candidate.gain_hybrid_weighted / denom)
    candidate.final_score_decoupled_sc = float(candidate.base_exp_utility + sc_gain_weight * selected_gain_sc)

    if score_gain_mode == "decoupled_sc":
        candidate.final_score = float(candidate.final_score_decoupled_sc)
        return candidate

    if score_gain_mode == "hybrid_weighted":
        candidate.final_score = float(candidate.utility_hybrid_weighted)
        return candidate

    score_attr = {
        "exp": "utility_exp",
        "sc": "utility_sc",
        "hybrid": "utility_hybrid",
        "occ": "utility_occ",
        "conf": "utility_conf",
    }.get(str(gain_mode))
    if score_attr is None:
        raise ValueError("gain_mode must be one of: exp, sc, hybrid, occ, conf")
    candidate.final_score = float(getattr(candidate, score_attr))
    return candidate


def _pose_position_yaw(current_pose_world: dict[str, Any] | tuple[float, float, float] | list[float]) -> tuple[np.ndarray, float, str]:
    if isinstance(current_pose_world, dict):
        if "position" not in current_pose_world:
            raise KeyError("current_pose_world dict must contain 'position'")
        position = np.asarray(current_pose_world["position"], dtype=np.float64)
        if "yaw_rad" in current_pose_world:
            yaw = float(current_pose_world["yaw_rad"])
            note = "yaw_rad read from pose metadata"
        elif "yaw_deg" in current_pose_world:
            yaw = math.radians(float(current_pose_world["yaw_deg"]))
            note = "yaw_deg read from pose metadata"
        else:
            yaw = 0.0
            note = "yaw missing in pose metadata; yaw=0 fallback"
    else:
        position = np.asarray(current_pose_world, dtype=np.float64)
        yaw = 0.0
        note = "current_pose_world provided as xyz only; yaw=0 fallback"
    if position.shape != (3,):
        raise ValueError(f"current pose position must have shape (3,), got {position.shape}")
    return position, yaw, note


def summarize_observed_state(observed_state: np.ndarray) -> dict[str, Any]:
    observed_state = _validate_observed_state(observed_state)
    unknown_count = int(np.count_nonzero(observed_state == UNKNOWN))
    free_count = int(np.count_nonzero(observed_state == FREE))
    occupied_count = int(np.count_nonzero(observed_state == OCCUPIED))
    observed_count = free_count + occupied_count
    total_count = int(observed_state.size)
    return {
        "shape": [int(v) for v in observed_state.shape],
        "unknown_count": unknown_count,
        "free_count": free_count,
        "occupied_count": occupied_count,
        "observed_count": observed_count,
        "total_count": total_count,
        "observed_ratio": float(observed_count / total_count) if total_count else 0.0,
    }


def summarize_prediction_layer(
    prediction_layer: PredictionLayerLike,
    observed_state: np.ndarray,
    tau: float = 0.1,
    sc_occ_threshold: float = 0.7,
    sc_conf_threshold: float = 0.3,
) -> dict[str, Any]:
    """Summarize a read-only prediction layer without using it for planning."""

    observed_state = _validate_observed_state(observed_state)
    shape = tuple(int(v) for v in prediction_layer.shape())
    if shape != tuple(observed_state.shape):
        raise ValueError(f"prediction_layer shape {shape} differs from observed_state {observed_state.shape}")

    summary: dict[str, Any] = {
        "prediction_layer_shape": [int(v) for v in shape],
        "prediction_source_npz": getattr(prediction_layer, "source_npz", None),
        "prediction_tau": float(tau),
        "prediction_valid_voxels": 0,
        "prediction_predicted_voxels": 0,
        "prediction_predicted_occupied_voxels": 0,
        "prediction_predicted_free_voxels": 0,
        "prediction_predicted_unmeasured_voxels": 0,
        "prediction_predicted_measured_voxels": 0,
        "prediction_selective_occupied_voxels": 0,
        "prediction_selective_unmeasured_occupied_voxels": 0,
        "prediction_confident_voxels": 0,
        "prediction_confidence_min": 0.0,
        "prediction_confidence_mean": 0.0,
        "prediction_confidence_max": 0.0,
        "prediction_occupied_prob_min": 0.0,
        "prediction_occupied_prob_mean": 0.0,
        "prediction_occupied_prob_max": 0.0,
    }

    valid = getattr(prediction_layer, "valid", None)
    confidence = getattr(prediction_layer, "confidence", None)
    occupied_prob = getattr(prediction_layer, "occupied_prob", None)
    free_prob = getattr(prediction_layer, "free_prob", None)
    if valid is None or confidence is None or occupied_prob is None:
        return summary

    valid_mask = np.asarray(valid, dtype=bool)
    confidence_arr = np.asarray(confidence, dtype=np.float32)
    occupied_arr = np.asarray(occupied_prob, dtype=np.float32)
    if valid_mask.shape != shape or confidence_arr.shape != shape or occupied_arr.shape != shape:
        raise ValueError("prediction layer array shapes do not match prediction_layer.shape()")
    predicted = valid_mask & (confidence_arr >= float(tau))
    confident = valid_mask & (confidence_arr >= float(sc_conf_threshold))
    measured = observed_state != UNKNOWN
    predicted_occupied = predicted & (occupied_arr >= 0.5)
    selective_occupied = predicted & confident & (occupied_arr >= float(sc_occ_threshold))
    if free_prob is not None:
        free_arr = np.asarray(free_prob, dtype=np.float32)
        if free_arr.shape != shape:
            raise ValueError("prediction free_prob shape does not match prediction_layer.shape()")
        predicted_free = predicted & (free_arr >= 0.5)
    else:
        predicted_free = predicted & ~predicted_occupied

    valid_conf = confidence_arr[valid_mask]
    valid_occ = occupied_arr[valid_mask]
    summary.update(
        {
            "prediction_valid_voxels": int(np.count_nonzero(valid_mask)),
            "prediction_predicted_voxels": int(np.count_nonzero(predicted)),
            "prediction_predicted_occupied_voxels": int(np.count_nonzero(predicted_occupied)),
            "prediction_predicted_free_voxels": int(np.count_nonzero(predicted_free)),
            "prediction_predicted_unmeasured_voxels": int(np.count_nonzero(predicted & ~measured)),
            "prediction_predicted_measured_voxels": int(np.count_nonzero(predicted & measured)),
            "prediction_confident_voxels": int(np.count_nonzero(confident)),
            "prediction_selective_occupied_voxels": int(np.count_nonzero(selective_occupied)),
            "prediction_selective_unmeasured_occupied_voxels": int(np.count_nonzero(selective_occupied & ~measured)),
        }
    )
    if valid_conf.size:
        summary.update(
            {
                "prediction_confidence_min": float(np.min(valid_conf)),
                "prediction_confidence_mean": float(np.mean(valid_conf)),
                "prediction_confidence_max": float(np.max(valid_conf)),
            }
        )
    if valid_occ.size:
        summary.update(
            {
                "prediction_occupied_prob_min": float(np.min(valid_occ)),
                "prediction_occupied_prob_mean": float(np.mean(valid_occ)),
                "prediction_occupied_prob_max": float(np.max(valid_occ)),
            }
        )
    return summary


def candidate_feature_vector(candidate: SimCandidateView) -> np.ndarray:
    return np.array(
        [
            candidate.gain_exp,
            candidate.gain_sc,
            candidate.gain_hybrid,
            candidate.gain_occ,
            candidate.gain_conf,
            candidate.path_cost,
            candidate.utility_exp,
            candidate.utility_sc,
            candidate.utility_hybrid,
            candidate.utility_occ,
            candidate.utility_conf,
            candidate.final_score,
            candidate.visible_count,
            candidate.measured_visible_count,
            candidate.predicted_unmeasured_visible_count,
            candidate.frontier_distance,
            candidate.frontier_count_visible,
            1.0 if candidate.astar_reachable else 0.0,
            candidate.astar_path_length_m if np.isfinite(candidate.astar_path_length_m) else 0.0,
            float(candidate.astar_num_expanded),
            candidate.weighted_gain_sc,
            candidate.gain_hybrid_weighted,
            candidate.utility_hybrid_weighted,
            candidate.sc_gain_weight,
            candidate.sc_gain_cap_value,
            candidate.raw_gain_sc,
            candidate.effective_gain_sc,
            candidate.gain_hybrid_effective,
            candidate.utility_effective_sc,
            candidate.utility_hybrid_effective,
            float(candidate.sc_selected_voxel_count),
            candidate.base_exp_utility,
            candidate.final_score_decoupled_sc,
        ],
        dtype=np.float32,
    )


def _candidate_to_json(candidate: SimCandidateView, rank: int | None = None) -> dict[str, Any]:
    payload = asdict(candidate)
    payload["grid_position"] = [int(v) for v in candidate.grid_position]
    payload["world_position"] = [float(v) for v in candidate.world_position]
    payload["yaw"] = float(candidate.yaw)
    payload["yaw_deg"] = float(math.degrees(candidate.yaw))
    payload["rank"] = None if rank is None else int(rank)
    payload["gains"] = {
        "gain_exp": float(candidate.gain_exp),
        "gain_sc": float(candidate.gain_sc),
        "gain_hybrid": float(candidate.gain_hybrid),
        "weighted_gain_sc": float(candidate.weighted_gain_sc),
        "gain_hybrid_weighted": float(candidate.gain_hybrid_weighted),
        "raw_gain_sc": float(candidate.raw_gain_sc),
        "effective_gain_sc": float(candidate.effective_gain_sc),
        "gain_hybrid_effective": float(candidate.gain_hybrid_effective),
        "gain_occ": float(candidate.gain_occ),
        "gain_conf": float(candidate.gain_conf),
    }
    payload["utilities"] = {
        "utility_exp": float(candidate.utility_exp),
        "utility_sc": float(candidate.utility_sc),
        "utility_hybrid": float(candidate.utility_hybrid),
        "utility_hybrid_weighted": float(candidate.utility_hybrid_weighted),
        "utility_effective_sc": float(candidate.utility_effective_sc),
        "utility_hybrid_effective": float(candidate.utility_hybrid_effective),
        "base_exp_utility": float(candidate.base_exp_utility),
        "utility_occ": float(candidate.utility_occ),
        "utility_conf": float(candidate.utility_conf),
    }
    payload["final_score_decoupled_sc"] = float(candidate.final_score_decoupled_sc)
    payload["sc_gain_weight"] = float(candidate.sc_gain_weight)
    payload["sc_gain_cap_value"] = float(candidate.sc_gain_cap_value)
    payload["sc_gain_formula"] = str(candidate.sc_gain_formula)
    payload["sc_occ_threshold"] = float(candidate.sc_occ_threshold)
    payload["sc_conf_threshold"] = float(candidate.sc_conf_threshold)
    payload["sc_count_mode"] = str(candidate.sc_count_mode)
    return payload


def _feature_arrays(top_candidates: list[SimCandidateView]) -> dict[str, np.ndarray]:
    if not top_candidates:
        raise ValueError("top_candidates is empty")
    return {
        "candidate_features": np.stack([candidate_feature_vector(c) for c in top_candidates], axis=0).astype(np.float32),
        "feature_names": np.array(FEATURE_NAMES),
        "candidate_positions_grid": np.array([c.grid_position for c in top_candidates], dtype=np.int32),
        "candidate_positions_world": np.array([c.world_position for c in top_candidates], dtype=np.float32),
        "candidate_yaws": np.array([c.yaw for c in top_candidates], dtype=np.float32),
        "valid_mask": np.array([c.valid for c in top_candidates], dtype=bool),
        "expert_scores": np.array([c.final_score for c in top_candidates], dtype=np.float32),
        "astar_reachable": np.array([c.astar_reachable for c in top_candidates], dtype=bool),
        "astar_path_length_m": np.array(
            [c.astar_path_length_m if np.isfinite(c.astar_path_length_m) else 0.0 for c in top_candidates],
            dtype=np.float32,
        ),
        "astar_num_expanded": np.array([c.astar_num_expanded for c in top_candidates], dtype=np.int32),
    }


def select_sim_expert_action(
    observed_state: np.ndarray,
    current_pose_world: dict[str, Any] | tuple[float, float, float] | list[float],
    bounds: dict[str, Any],
    voxel_size: float,
    prediction_layer: PredictionLayerLike | None = None,
    prediction_mode: str = "empty",
    num_candidates: int = 64,
    top_n: int = 16,
    gain_mode: str = "hybrid",
    sc_gain_formula: str = "raw_count",
    sc_occ_threshold: float = 0.7,
    sc_conf_threshold: float = 0.3,
    sc_count_mode: str = "raw_count",
    score_gain_mode: str = "hybrid_raw",
    sc_gain_weight: float = 1.0,
    sc_gain_cap: float | None = None,
    calibration_table: dict[str, Any] | str | Path | None = None,
    seed: int = 0,
    max_range_voxels: int = 50,
    num_yaw: int = 32,
    num_pitch: int = 7,
    fov_yaw_deg: float = 90.0,
    fov_pitch_deg: float = 60.0,
    tau: float = 0.1,
    v_max: float = 1.0,
    yaw_rate_deg: float = 90.0,
    path_cost_mode: str = "euclidean",
    robot_height_m: float = 1.2,
    clearance_height_m: float = 0.6,
    robot_radius_m: float = 0.2,
    astar_allow_diagonal: bool = True,
    candidate_sampling_mode: str = "auto",
    snap_start_to_traversable: bool = True,
    max_snap_radius_cells: int = 5,
) -> dict[str, Any]:
    """Score one simulator observed map and choose top-N views.

    Prediction, when provided, contributes only to paper-style information gain.
    Candidate sampling, traversability, A*, collision bookkeeping, and ray
    blocking remain measured-only.
    """

    observed_state = _validate_observed_state(observed_state)
    score_gain_mode = str(score_gain_mode)
    if score_gain_mode not in ("hybrid_raw", "hybrid_weighted", "decoupled_sc"):
        raise ValueError("score_gain_mode must be one of: hybrid_raw, hybrid_weighted, decoupled_sc")
    sc_gain_formula = str(sc_gain_formula)
    if sc_gain_formula not in SC_GAIN_FORMULAS:
        raise ValueError(f"sc_gain_formula must be one of: {', '.join(SC_GAIN_FORMULAS)}")
    sc_count_mode = str(sc_count_mode)
    if sc_count_mode not in SC_COUNT_MODES:
        raise ValueError(f"sc_count_mode must be one of: {', '.join(SC_COUNT_MODES)}")
    sc_occ_threshold = float(sc_occ_threshold)
    sc_conf_threshold = float(sc_conf_threshold)
    if not (0.0 <= sc_occ_threshold <= 1.0):
        raise ValueError("sc_occ_threshold must be in [0, 1]")
    if not (0.0 <= sc_conf_threshold <= 1.0):
        raise ValueError("sc_conf_threshold must be in [0, 1]")
    if isinstance(calibration_table, (str, Path)):
        calibration_table = load_calibration_table(calibration_table)
    if sc_gain_formula == "calibrated_occupied" and calibration_table is None:
        raise ValueError("sc_gain_formula='calibrated_occupied' requires calibration_table")
    sc_gain_weight = float(sc_gain_weight)
    if sc_gain_cap is not None:
        sc_gain_cap = float(sc_gain_cap)
    prediction_mode = str(prediction_mode)
    if prediction_mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of: {', '.join(PREDICTION_MODES)}")
    if prediction_mode == PREDICTION_MODE_EMPTY:
        prediction_layer = EmptyPredictionLayer(tuple(observed_state.shape))
    elif prediction_layer is None:
        raise ValueError("prediction_layer is required when prediction_mode='sim_npz'")
    if tuple(prediction_layer.shape()) != tuple(observed_state.shape):
        raise ValueError(
            f"prediction_layer shape {prediction_layer.shape()} differs from observed_state {observed_state.shape}"
        )
    prediction_summary = summarize_prediction_layer(
        prediction_layer=prediction_layer,
        observed_state=observed_state,
        tau=tau,
        sc_occ_threshold=sc_occ_threshold,
        sc_conf_threshold=sc_conf_threshold,
    )
    path_cost_mode = str(path_cost_mode)
    if path_cost_mode not in ("euclidean", "astar"):
        raise ValueError("path_cost_mode must be one of: euclidean, astar")
    requested_candidate_sampling_mode = str(candidate_sampling_mode)
    if requested_candidate_sampling_mode == "auto":
        resolved_candidate_sampling_mode = "reachable_frontier" if path_cost_mode == "astar" else "frontier"
    else:
        resolved_candidate_sampling_mode = requested_candidate_sampling_mode
    if resolved_candidate_sampling_mode not in ("frontier", "reachable_frontier"):
        raise ValueError("candidate_sampling_mode must be one of: frontier, reachable_frontier, auto")

    normalized_bounds = normalize_bounds(bounds)
    current_position, current_yaw, pose_note = _pose_position_yaw(current_pose_world)
    current_grid = world_to_grid(
        current_position,
        normalized_bounds,
        voxel_size,
        shape=tuple(observed_state.shape),
        clip=True,
    )

    frontier_voxels = detect_frontier_voxels(observed_state)
    frontier_adjacent_free_voxels = detect_frontier_adjacent_free_voxels(observed_state)

    traversability_result: dict[str, Any] | None = None
    traversability_summary: dict[str, Any] | None = None
    if path_cost_mode == "astar" or resolved_candidate_sampling_mode == "reachable_frontier":
        traversability_result = build_traversability_grid(
            observed_state=observed_state,
            voxel_size=float(voxel_size),
            robot_height_m=float(robot_height_m),
            clearance_height_m=float(clearance_height_m),
            robot_radius_m=float(robot_radius_m),
        )
        traversability_summary = summarize_traversability(traversability_result)

    sampled = sample_candidate_views_from_frontiers(
        observed_state=observed_state,
        current_grid=current_grid,
        bounds=normalized_bounds,
        voxel_size=voxel_size,
        num_candidates=num_candidates,
        seed=seed,
        candidate_sampling_mode=resolved_candidate_sampling_mode,
        traversability=traversability_result,
        max_snap_radius_cells=int(max_snap_radius_cells),
        snap_start_to_traversable=bool(snap_start_to_traversable),
        astar_allow_diagonal=bool(astar_allow_diagonal),
        return_diagnostics=True,
    )
    candidates, sampling_info = sampled
    assert isinstance(candidates, list)
    assert isinstance(sampling_info, dict)

    traversable_2d = (
        np.asarray(traversability_result["traversable"], dtype=bool)
        if isinstance(traversability_result, dict)
        else None
    )
    planning_start_xy = (int(current_grid[0]), int(current_grid[1]))
    start_traversable = None
    start_in_bounds = None
    snapped_current = False
    snapped_current_xy: list[int] | None = [int(planning_start_xy[0]), int(planning_start_xy[1])]
    snap_distance_cells: float | None = 0.0
    reachable_mask_for_viz: np.ndarray | None = None
    reachable_component_count: int | None = None
    reachable_frontier_adjacent_count: int | None = None
    reachable_free_fallback_count: int | None = None
    candidate_source = str(sampling_info.get("candidate_source", "frontier"))
    no_valid_candidate_reason = ""

    if traversable_2d is not None:
        shape_xy = tuple(int(v) for v in traversable_2d.shape)
        start_in_bounds = bool(0 <= planning_start_xy[0] < shape_xy[0] and 0 <= planning_start_xy[1] < shape_xy[1])
        start_traversable = bool(start_in_bounds and traversable_2d[planning_start_xy])

        if resolved_candidate_sampling_mode == "reachable_frontier":
            snapped_value = sampling_info.get("snapped_current_xy")
            if snapped_value is not None:
                planning_start_xy = (int(snapped_value[0]), int(snapped_value[1]))
                snapped_current_xy = [int(planning_start_xy[0]), int(planning_start_xy[1])]
            else:
                snapped_current_xy = None
            snapped_current = bool(sampling_info.get("snapped", False))
            snap_distance_cells = sampling_info.get("snap_distance_cells")
            reachable_mask_value = sampling_info.get("reachable_mask")
            if reachable_mask_value is not None:
                reachable_mask_for_viz = np.asarray(reachable_mask_value, dtype=bool)
            reachable_component_count = (
                None if sampling_info.get("reachable_count") is None else int(sampling_info.get("reachable_count"))
            )
            reachable_frontier_adjacent_count = (
                None
                if sampling_info.get("reachable_frontier_adjacent_count") is None
                else int(sampling_info.get("reachable_frontier_adjacent_count"))
            )
            reachable_free_fallback_count = int(sampling_info.get("reachable_free_fallback_count") or 0)
            if str(sampling_info.get("reason")) == "no_reachable_free_component":
                no_valid_candidate_reason = "no_reachable_free_component"
        else:
            if not start_traversable and bool(snap_start_to_traversable):
                snap_result = nearest_traversable_cell(
                    traversable_2d,
                    planning_start_xy,
                    max_radius_cells=int(max_snap_radius_cells),
                )
                if bool(snap_result.get("found", False)):
                    cell = snap_result["cell"]
                    assert cell is not None
                    planning_start_xy = (int(cell[0]), int(cell[1]))
                    snapped_current = True
                    snapped_current_xy = [int(planning_start_xy[0]), int(planning_start_xy[1])]
                    snap_distance_cells = float(snap_result.get("distance_cells", 0.0))
                else:
                    snap_distance_cells = None
            component = connected_component_from_start(
                traversable_2d,
                planning_start_xy,
                allow_diagonal=bool(astar_allow_diagonal),
            )
            reachable_mask_for_viz = np.asarray(component["reachable_mask"], dtype=bool)
            reachable_component_count = int(component["reachable_count"])
            frontier_xy = frontier_adjacent_free_xy_mask(observed_state)
            reachable_frontier_adjacent_count = int(np.count_nonzero(reachable_mask_for_viz & frontier_xy))
            reachable_free_fallback_count = 0
            if reachable_component_count <= 0:
                no_valid_candidate_reason = "no_reachable_free_component"

        if isinstance(traversability_result, dict) and reachable_mask_for_viz is not None:
            traversability_result["reachable_mask"] = reachable_mask_for_viz
            traversability_result["astar_start_xy"] = [int(planning_start_xy[0]), int(planning_start_xy[1])]

    if not candidates and no_valid_candidate_reason == "no_reachable_free_component":
        raise ValueError(
            "no_reachable_free_component: current pose has no traversable observed-free component "
            f"(current_xy={[int(current_grid[0]), int(current_grid[1])]}, "
            f"snap_start_to_traversable={bool(snap_start_to_traversable)}, "
            f"max_snap_radius_cells={int(max_snap_radius_cells)})"
        )
    if not candidates:
        raise ValueError(
            "No candidates sampled "
            f"(candidate_sampling_mode={resolved_candidate_sampling_mode}, candidate_source={candidate_source})"
        )

    for candidate in candidates:
        astar_result: dict[str, Any] | None = None
        if path_cost_mode == "astar":
            assert traversability_result is not None
            astar_result = astar_2d(
                traversability_result["traversable"],
                start_xy=planning_start_xy,
                goal_xy=candidate.grid_position[:2],
                allow_diagonal=bool(astar_allow_diagonal),
            )
            if not bool(astar_result.get("reachable", False)):
                compute_cost_and_score(
                    candidate=candidate,
                    current_grid=current_grid,
                    current_yaw=current_yaw,
                    gain_mode=gain_mode,
                    score_gain_mode=score_gain_mode,
                    sc_gain_weight=sc_gain_weight,
                    sc_gain_cap=sc_gain_cap,
                    voxel_size=voxel_size,
                    v_max=v_max,
                    yaw_rate_deg=yaw_rate_deg,
                    path_cost_mode=path_cost_mode,
                    astar_result=astar_result,
                )
                continue

        visible = raycast_visible_voxels_observed(
            candidate=candidate,
            observed_state=observed_state,
            max_range_voxels=max_range_voxels,
            num_yaw=num_yaw,
            num_pitch=num_pitch,
            fov_yaw_deg=fov_yaw_deg,
            fov_pitch_deg=fov_pitch_deg,
        )
        compute_paper_gains_for_candidate(
            candidate=candidate,
            observed_state=observed_state,
            prediction_layer=prediction_layer,
            visible_voxels=visible,
            tau=tau,
            sc_gain_formula=sc_gain_formula,
            sc_occ_threshold=sc_occ_threshold,
            sc_conf_threshold=sc_conf_threshold,
            sc_count_mode=sc_count_mode,
            calibration_table=calibration_table,
        )
        compute_cost_and_score(
            candidate=candidate,
            current_grid=current_grid,
            current_yaw=current_yaw,
            gain_mode=gain_mode,
            score_gain_mode=score_gain_mode,
            sc_gain_weight=sc_gain_weight,
            sc_gain_cap=sc_gain_cap,
            voxel_size=voxel_size,
            v_max=v_max,
            yaw_rate_deg=yaw_rate_deg,
            path_cost_mode=path_cost_mode,
            astar_result=astar_result,
        )

    valid_candidates = [c for c in candidates if c.valid and np.isfinite(c.final_score)]
    if not valid_candidates:
        reachable_count = int(sum(1 for c in candidates if c.valid and c.astar_reachable))
        unreachable_count = int(sum(1 for c in candidates if str(c.invalid_reason).startswith("unreachable_astar:")))
        if no_valid_candidate_reason:
            detail = f"{no_valid_candidate_reason}: "
        else:
            detail = ""
        trav_note = ""
        if traversability_summary is not None:
            trav_note = (
                " "
                f"traversable={traversability_summary['traversable_count']} "
                f"blocked={traversability_summary['blocked_count']} "
                f"unknown={traversability_summary['unknown_count']}"
            )
        raise ValueError(
            f"{detail}No valid finite-scored candidates available "
            f"(path_cost_mode={path_cost_mode}, reachable={reachable_count}, "
            f"unreachable={unreachable_count}, candidates={len(candidates)}).{trav_note}"
        )

    sorted_candidates = sorted(valid_candidates, key=lambda c: (-c.final_score, c.id))
    top_candidates = sorted_candidates[: min(int(top_n), len(sorted_candidates))]
    best_candidate = top_candidates[0]
    expert_action = 0
    arrays = _feature_arrays(top_candidates)

    observed_summary = summarize_observed_state(observed_state)
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        source_counts[candidate.candidate_source] = source_counts.get(candidate.candidate_source, 0) + 1
    sampling_diagnostics = {
        key: value
        for key, value in sampling_info.items()
        if key not in {"reachable_mask", "candidate_mask"}
    }
    diagnostics = {
        **observed_summary,
        **prediction_summary,
        "frontier_count": int(len(frontier_voxels)),
        "frontier_adjacent_free_count": int(len(frontier_adjacent_free_voxels)),
        "frontier_adjacent_free_xy_count": int(np.count_nonzero(frontier_adjacent_free_xy_mask(observed_state))),
        "num_candidates_requested": int(num_candidates),
        "num_candidates": int(len(candidates)),
        "top_n_requested": int(top_n),
        "top_n": int(len(top_candidates)),
        "prediction_mode": str(prediction_mode),
        "prediction_layer_class": type(prediction_layer).__name__,
        "prediction_used_for_information_gain": bool(prediction_mode == PREDICTION_MODE_SIM_NPZ),
        "prediction_affects_information_gain_only": True,
        "prediction_used_for_candidate_sampling": False,
        "prediction_used_for_traversability": False,
        "prediction_used_for_collision": False,
        "prediction_used_for_astar": False,
        "prediction_blocks_rays": False,
        "raycast_uses_prediction_for_blocking": False,
        "traversability_source": "observed_state_only",
        "gain_mode": str(gain_mode),
        "sc_gain_formula": str(sc_gain_formula),
        "sc_occ_threshold": float(sc_occ_threshold),
        "sc_conf_threshold": float(sc_conf_threshold),
        "sc_count_mode": str(sc_count_mode),
        "calibration_table_loaded": calibration_table is not None,
        "calibration_table_source": str(calibration_table.get("source", "")) if isinstance(calibration_table, dict) else None,
        "future_observations_used_for_planning": False,
        "future_observations_used_for_scoring": False,
        "future_observations_usage": "calibration/evaluation-only; runtime expert scoring reads only fixed calibration table when provided",
        "score_gain_mode": str(score_gain_mode),
        "sc_gain_weight": float(sc_gain_weight),
        "sc_gain_cap": None if sc_gain_cap is None else float(sc_gain_cap),
        "sc_gain_cap_value": -1.0 if sc_gain_cap is None else float(sc_gain_cap),
        "path_cost_mode": str(path_cost_mode),
        "candidate_sampling_mode_requested": str(requested_candidate_sampling_mode),
        "candidate_sampling_mode": str(resolved_candidate_sampling_mode),
        "candidate_source": str(candidate_source),
        "candidate_source_counts": source_counts,
        "candidate_sampling_diagnostics": sampling_diagnostics,
        "raycast_mode": RAYCAST_MODE_OBSERVED_CONSERVATIVE,
        "max_range_voxels": int(max_range_voxels),
        "num_yaw": int(num_yaw),
        "num_pitch": int(num_pitch),
        "fov_yaw_deg": float(fov_yaw_deg),
        "fov_pitch_deg": float(fov_pitch_deg),
        "tau": float(tau),
        "voxel_size": float(voxel_size),
        "v_max": float(v_max),
        "yaw_rate_deg": float(yaw_rate_deg),
        "bounds": {axis: [normalized_bounds[axis][0], normalized_bounds[axis][1]] for axis in ("x", "y", "z")},
        "current_pose_grid_state": int(observed_state[current_grid]),
        "current_xy": [int(current_grid[0]), int(current_grid[1])],
        "astar_start_xy": [int(planning_start_xy[0]), int(planning_start_xy[1])],
        "snapped_current": bool(snapped_current),
        "snapped_current_xy": None if snapped_current_xy is None else [int(v) for v in snapped_current_xy],
        "snap_distance_cells": None if snap_distance_cells is None else float(snap_distance_cells),
        "start_traversable": None if start_traversable is None else bool(start_traversable),
        "start_in_bounds": None if start_in_bounds is None else bool(start_in_bounds),
        "reachable_component_count": reachable_component_count,
        "reachable_frontier_adjacent_count": reachable_frontier_adjacent_count,
        "reachable_free_fallback_count": reachable_free_fallback_count,
        "no_valid_candidate_reason": str(no_valid_candidate_reason),
        "reachable_candidates": int(sum(1 for c in candidates if c.valid and c.astar_reachable)),
        "unreachable_candidates": int(sum(1 for c in candidates if str(c.invalid_reason).startswith("unreachable_astar:"))),
        "astar_allow_diagonal": bool(astar_allow_diagonal),
        "traversability": traversability_summary,
        "traversable_count": int(traversability_summary["traversable_count"]) if traversability_summary else None,
        "blocked_count": int(traversability_summary["blocked_count"]) if traversability_summary else None,
        "traversability_unknown_count": int(traversability_summary["unknown_count"]) if traversability_summary else None,
        "unknown_2d_count": int(traversability_summary["unknown_count"]) if traversability_summary else None,
        "robot_height_m": float(robot_height_m),
        "clearance_height_m": float(clearance_height_m),
        "robot_radius_m": float(robot_radius_m),
        "snap_start_to_traversable": bool(snap_start_to_traversable),
        "max_snap_radius_cells": int(max_snap_radius_cells),
        "pose_note": pose_note,
        "strict_no_prediction_write_note": STRICT_NO_PREDICTION_WRITE_NOTE,
        "no_target_or_ground_truth_note": NO_TARGET_OR_GT_NOTE,
        "tree_limitation_note": TREE_LIMITATION_NOTE,
        "prediction_written_to_observed_state": False,
        "observed_state_writeback": False,
        "collision_checking_used": False,
        "rollout_run": False,
        "target_lr_used": False,
        "target_hr_used": False,
        "ground_truth_used": False,
        "rl_or_training_used": False,
        "rl_optimizer_bc_il_training_run": False,
        "behavior_cloning_training_used": False,
        "imitation_learning_training_used": False,
        "optimizer_used": False,
        "policy_training_used": False,
    }

    return {
        "best_candidate": best_candidate,
        "top_candidates": top_candidates,
        "all_candidates": candidates,
        "feature_arrays": arrays,
        "diagnostics": diagnostics,
        "expert_action": int(expert_action),
        "current_pose_world": current_position.astype(np.float64),
        "current_yaw": float(current_yaw),
        "current_grid": np.array(current_grid, dtype=np.int32),
        "frontier_voxels": frontier_voxels,
        "frontier_adjacent_free_voxels": frontier_adjacent_free_voxels,
        "prediction_mode": str(prediction_mode),
        "gain_mode": str(gain_mode),
        "score_gain_mode": str(score_gain_mode),
        "path_cost_mode": str(path_cost_mode),
        "raycast_mode": RAYCAST_MODE_OBSERVED_CONSERVATIVE,
        "traversability": traversability_result,
    }


def save_expert_step_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
    observed_state_path: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = result["feature_arrays"]
    current_pose_world = np.asarray(result["current_pose_world"], dtype=np.float32)
    current_pose_grid = np.asarray(result["current_grid"], dtype=np.int32)
    diagnostics = result["diagnostics"]
    best_astar_path = np.asarray(result["best_candidate"].astar_path_xy, dtype=np.int32)
    if best_astar_path.size == 0:
        best_astar_path = np.zeros((0, 2), dtype=np.int32)
    snapped_xy = diagnostics.get("snapped_current_xy")
    snapped_xy_array = (
        np.asarray(snapped_xy, dtype=np.int32)
        if snapped_xy is not None
        else np.asarray([-1, -1], dtype=np.int32)
    )
    npz_path = output_dir / "expert_step_decision.npz"
    np.savez_compressed(
        npz_path,
        candidate_features=arrays["candidate_features"],
        feature_names=arrays["feature_names"],
        candidate_positions_grid=arrays["candidate_positions_grid"],
        candidate_positions_world=arrays["candidate_positions_world"],
        candidate_yaws=arrays["candidate_yaws"],
        valid_mask=arrays["valid_mask"],
        expert_action=np.array(int(result["expert_action"]), dtype=np.int64),
        expert_scores=arrays["expert_scores"],
        astar_reachable=arrays["astar_reachable"],
        astar_path_length_m=arrays["astar_path_length_m"],
        astar_num_expanded=arrays["astar_num_expanded"],
        best_astar_path_xy=best_astar_path,
        current_pose_world=current_pose_world,
        current_pose_grid=current_pose_grid,
        observed_state_path=np.array(str(observed_state_path)),
        prediction_mode=np.array(str(result["prediction_mode"])),
        prediction_source_npz=np.array(str(diagnostics.get("prediction_source_npz") or "")),
        tau=np.array(float(diagnostics.get("tau", 0.1)), dtype=np.float32),
        gain_mode=np.array(str(result["gain_mode"])),
        score_gain_mode=np.array(str(diagnostics.get("score_gain_mode", "hybrid_raw"))),
        sc_gain_formula=np.array(str(diagnostics.get("sc_gain_formula", "raw_count"))),
        sc_occ_threshold=np.array(float(diagnostics.get("sc_occ_threshold", 0.7)), dtype=np.float32),
        sc_conf_threshold=np.array(float(diagnostics.get("sc_conf_threshold", 0.3)), dtype=np.float32),
        sc_count_mode=np.array(str(diagnostics.get("sc_count_mode", "raw_count"))),
        sc_gain_weight=np.array(float(diagnostics.get("sc_gain_weight", 1.0)), dtype=np.float32),
        sc_gain_cap_value=np.array(float(diagnostics.get("sc_gain_cap_value", -1.0)), dtype=np.float32),
        path_cost_mode=np.array(str(result["path_cost_mode"])),
        candidate_sampling_mode=np.array(str(diagnostics.get("candidate_sampling_mode", "frontier"))),
        candidate_source=np.array(str(diagnostics.get("candidate_source", "frontier"))),
        reachable_component_count=np.array(int(diagnostics.get("reachable_component_count") or 0), dtype=np.int64),
        reachable_frontier_adjacent_count=np.array(
            int(diagnostics.get("reachable_frontier_adjacent_count") or 0),
            dtype=np.int64,
        ),
        snapped_current=np.array(bool(diagnostics.get("snapped_current", False))),
        snapped_current_xy=snapped_xy_array,
        snap_distance_cells=np.array(
            -1.0 if diagnostics.get("snap_distance_cells") is None else float(diagnostics.get("snap_distance_cells")),
            dtype=np.float32,
        ),
        raycast_mode=np.array(str(result["raycast_mode"])),
        prediction_used_for_information_gain=np.array(bool(diagnostics.get("prediction_used_for_information_gain", False))),
        prediction_used_for_traversability=np.array(bool(diagnostics.get("prediction_used_for_traversability", False))),
        prediction_used_for_collision=np.array(bool(diagnostics.get("prediction_used_for_collision", False))),
        prediction_used_for_astar=np.array(bool(diagnostics.get("prediction_used_for_astar", False))),
        prediction_blocks_rays=np.array(bool(diagnostics.get("prediction_blocks_rays", False))),
        prediction_written_to_observed_state=np.array(bool(diagnostics.get("prediction_written_to_observed_state", False))),
        future_observations_used_for_planning=np.array(bool(diagnostics.get("future_observations_used_for_planning", False))),
        future_observations_used_for_scoring=np.array(bool(diagnostics.get("future_observations_used_for_scoring", False))),
        strict_no_prediction_write_note=np.array(STRICT_NO_PREDICTION_WRITE_NOTE),
    )

    top_by_id = {candidate.id: rank for rank, candidate in enumerate(result["top_candidates"])}
    best = result["best_candidate"]
    decision = {
        "stage": (
            "Stage 4A-5.1"
            if str(result["prediction_mode"]) == PREDICTION_MODE_SIM_NPZ
            else ("Stage 4A-3.6" if str(result["path_cost_mode"]) == "astar" else "Stage 4A-2")
        ),
        "observed_state_path": str(observed_state_path),
        "prediction_mode": str(result["prediction_mode"]),
        "gain_mode": str(result["gain_mode"]),
        "sc_gain_formula": str(diagnostics.get("sc_gain_formula", "raw_count")),
        "sc_occ_threshold": float(diagnostics.get("sc_occ_threshold", 0.7)),
        "sc_conf_threshold": float(diagnostics.get("sc_conf_threshold", 0.3)),
        "sc_count_mode": str(diagnostics.get("sc_count_mode", "raw_count")),
        "score_gain_mode": str(diagnostics.get("score_gain_mode", "hybrid_raw")),
        "sc_gain_weight": float(diagnostics.get("sc_gain_weight", 1.0)),
        "sc_gain_cap": diagnostics.get("sc_gain_cap"),
        "sc_gain_cap_value": float(diagnostics.get("sc_gain_cap_value", -1.0)),
        "path_cost_mode": str(result["path_cost_mode"]),
        "raycast_mode": str(result["raycast_mode"]),
        "expert_action": int(result["expert_action"]),
        "current_pose": {
            "world": [float(v) for v in np.asarray(result["current_pose_world"]).tolist()],
            "grid": [int(v) for v in np.asarray(result["current_grid"]).tolist()],
            "yaw": float(result["current_yaw"]),
            "yaw_deg": float(math.degrees(float(result["current_yaw"]))),
        },
        "best_candidate": _candidate_to_json(best, rank=0),
        "top_candidates": [
            _candidate_to_json(candidate, rank=rank) for rank, candidate in enumerate(result["top_candidates"])
        ],
        "diagnostics": result["diagnostics"],
        "feature_names": FEATURE_NAMES,
        "notes": [
            STRICT_NO_PREDICTION_WRITE_NOTE,
            NO_TARGET_OR_GT_NOTE,
            TREE_LIMITATION_NOTE,
        ],
    }
    json_path = output_dir / "expert_step_decision.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(decision, handle, indent=2, sort_keys=True)
        handle.write("\n")

    jsonl_path = output_dir / "expert_step_candidates.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for candidate in result["all_candidates"]:
            record = _candidate_to_json(candidate, rank=top_by_id.get(candidate.id))
            record["is_top_candidate"] = candidate.id in top_by_id
            record["is_best_candidate"] = candidate.id == best.id
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    return {
        "npz": str(npz_path),
        "json": str(json_path),
        "jsonl": str(jsonl_path),
    }


def format_top_candidates(top_candidates: list[SimCandidateView]) -> str:
    lines = [
        "rank id score gain_exp raw_gain_sc effective_gain_sc weighted_gain_sc gain_hybrid gain_hybrid_effective gain_hybrid_weighted gain_occ gain_conf path_cost astar_len astar_exp visible pred_unmeasured_visible selected_sc frontier_visible source grid world yaw_deg"
    ]
    for rank, candidate in enumerate(top_candidates):
        lines.append(
            f"{rank:>4d} {candidate.id:>2d} {candidate.final_score:>10.6f} "
            f"{candidate.gain_exp:>8.1f} {candidate.gain_sc:>11.1f} "
            f"{candidate.effective_gain_sc:>17.3f} {candidate.weighted_gain_sc:>16.3f} "
            f"{candidate.gain_hybrid:>11.1f} {candidate.gain_hybrid_effective:>21.3f} "
            f"{candidate.gain_hybrid_weighted:>20.1f} {candidate.gain_occ:>8.1f} "
            f"{candidate.gain_conf:>9.3f} {candidate.path_cost:>9.6f} "
            f"{candidate.astar_path_length_m:>9.3f} {candidate.astar_num_expanded:>9d} "
            f"{candidate.visible_count:>7d} {candidate.predicted_unmeasured_visible_count:>26d} "
            f"{candidate.sc_selected_voxel_count:>11d} "
            f"{candidate.frontier_count_visible:>16d} "
            f"{candidate.candidate_source:>24s} "
            f"{candidate.grid_position} "
            f"({candidate.world_position[0]:.2f},{candidate.world_position[1]:.2f},{candidate.world_position[2]:.2f}) "
            f"{math.degrees(candidate.yaw):.2f}"
        )
    return "\n".join(lines)
