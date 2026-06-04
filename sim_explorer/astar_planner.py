#!/usr/bin/env python3
"""Observed-free 2D A* planner for simulator expert path-cost scoring.

The planner is intentionally measured-only. It derives traversability from the
current observed voxel state and never reads scene metadata, target labels,
ground truth, or prediction output.
"""

from __future__ import annotations

import heapq
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

UNKNOWN = np.int8(-1)
FREE = np.int8(0)
OCCUPIED = np.int8(1)


def _validate_observed_state(observed_state: np.ndarray) -> np.ndarray:
    observed_state = np.asarray(observed_state)
    if observed_state.ndim != 3:
        raise ValueError(f"observed_state must be 3D [X,Y,Z], got shape {observed_state.shape}")
    allowed = np.isin(observed_state, [UNKNOWN, FREE, OCCUPIED])
    if not bool(np.all(allowed)):
        bad = np.unique(observed_state[~allowed])
        raise ValueError(f"observed_state contains unsupported values: {bad.tolist()}")
    return observed_state


def _dilate_square(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if int(radius_cells) <= 0:
        return mask.astype(bool, copy=True)

    radius = int(radius_cells)
    try:
        from scipy import ndimage  # type: ignore

        structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        return ndimage.binary_dilation(mask.astype(bool), structure=structure)
    except Exception:
        padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
        out = np.zeros_like(mask, dtype=bool)
        for di in range(2 * radius + 1):
            for dj in range(2 * radius + 1):
                out |= padded[di : di + mask.shape[0], dj : dj + mask.shape[1]]
        return out


def build_traversability_grid(
    observed_state: np.ndarray,
    voxel_size: float = 0.1,
    robot_height_m: float = 1.2,
    clearance_height_m: float = 0.6,
    robot_radius_m: float = 0.2,
) -> dict[str, Any]:
    """Build a 2D traversability grid from observed FREE/OCCUPIED voxels.

    FREE support in the robot-height band is required. UNKNOWN is not
    traversable. OCCUPIED cells in the band are inflated in x/y by
    ``robot_radius_m`` before traversability is computed.
    """

    observed_state = _validate_observed_state(observed_state)
    if float(voxel_size) <= 0.0:
        raise ValueError("voxel_size must be positive")
    if float(clearance_height_m) <= 0.0:
        raise ValueError("clearance_height_m must be positive")
    if float(robot_radius_m) < 0.0:
        raise ValueError("robot_radius_m must be non-negative")

    shape = tuple(int(v) for v in observed_state.shape)
    z_centers = (np.arange(shape[2], dtype=np.float64) + 0.5) * float(voxel_size)
    z_min_m = float(robot_height_m) - 0.5 * float(clearance_height_m)
    z_max_m = float(robot_height_m) + 0.5 * float(clearance_height_m)
    band_ids = np.where((z_centers >= z_min_m) & (z_centers <= z_max_m))[0]
    if band_ids.size == 0:
        nearest = int(np.argmin(np.abs(z_centers - float(robot_height_m))))
        band_ids = np.array([nearest], dtype=np.int64)

    z_min_idx = int(band_ids.min())
    z_max_idx = int(band_ids.max())
    band = observed_state[:, :, z_min_idx : z_max_idx + 1]

    blocked_raw = np.any(band == OCCUPIED, axis=2)
    free_support = np.any(band == FREE, axis=2)
    unknown_raw = ~free_support

    robot_radius_cells = int(math.ceil(float(robot_radius_m) / float(voxel_size)))
    blocked = _dilate_square(blocked_raw, robot_radius_cells)
    traversable = free_support & ~blocked
    unknown = ~traversable & ~blocked

    diagnostics = {
        "shape": [int(shape[0]), int(shape[1])],
        "voxel_size": float(voxel_size),
        "robot_height_m": float(robot_height_m),
        "clearance_height_m": float(clearance_height_m),
        "robot_radius_m": float(robot_radius_m),
        "z_min_m": float(z_min_m),
        "z_max_m": float(z_max_m),
        "z_min_idx": int(z_min_idx),
        "z_max_idx": int(z_max_idx),
        "z_band_count": int(z_max_idx - z_min_idx + 1),
        "robot_radius_cells": int(robot_radius_cells),
        "blocked_raw_count": int(np.count_nonzero(blocked_raw)),
        "free_support_count": int(np.count_nonzero(free_support)),
        "unknown_raw_count": int(np.count_nonzero(unknown_raw)),
        "blocked_count": int(np.count_nonzero(blocked)),
        "traversable_count": int(np.count_nonzero(traversable)),
        "unknown_count": int(np.count_nonzero(unknown)),
        "used_scipy_dilation": bool(_scipy_available()),
    }

    return {
        "traversable": traversable.astype(bool, copy=False),
        "blocked": blocked.astype(bool, copy=False),
        "unknown": unknown.astype(bool, copy=False),
        "z_min_idx": int(z_min_idx),
        "z_max_idx": int(z_max_idx),
        "robot_radius_cells": int(robot_radius_cells),
        "diagnostics": diagnostics,
    }


def _scipy_available() -> bool:
    try:
        import scipy.ndimage  # noqa: F401

        return True
    except Exception:
        return False


def _xy_tuple(value: tuple[int, int] | list[int] | np.ndarray) -> tuple[int, int]:
    arr = np.asarray(value, dtype=np.int64)
    if arr.shape[0] < 2:
        raise ValueError(f"xy index must contain at least 2 values, got {value}")
    return int(arr[0]), int(arr[1])


def _in_bounds_xy(xy: tuple[int, int], shape: tuple[int, int]) -> bool:
    return 0 <= xy[0] < shape[0] and 0 <= xy[1] < shape[1]


def _validate_traversable_2d(traversable: np.ndarray) -> np.ndarray:
    grid = np.asarray(traversable, dtype=bool)
    if grid.ndim != 2:
        raise ValueError(f"traversable must be 2D, got shape {grid.shape}")
    return grid


def _neighbor_offsets_2d(allow_diagonal: bool) -> list[tuple[int, int]]:
    cardinal = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if not bool(allow_diagonal):
        return cardinal
    diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return cardinal + diagonal


def connected_component_from_start(
    traversable: np.ndarray,
    start_xy: tuple[int, int] | list[int] | np.ndarray,
    allow_diagonal: bool = True,
) -> dict[str, Any]:
    """Return the traversable connected component containing ``start_xy``.

    The input is a measured-only 2D traversability grid. UNKNOWN handling is
    decided by the caller when building ``traversable``; this routine only
    follows cells already marked True.
    """

    grid = _validate_traversable_2d(traversable)
    shape = (int(grid.shape[0]), int(grid.shape[1]))
    start = _xy_tuple(start_xy)
    reachable_mask = np.zeros(shape, dtype=bool)

    if not _in_bounds_xy(start, shape):
        return {
            "reachable_mask": reachable_mask,
            "reachable_count": 0,
            "start_valid": False,
            "reason": "start_out_of_bounds",
            "start_xy": [int(start[0]), int(start[1])],
        }
    if not bool(grid[start]):
        return {
            "reachable_mask": reachable_mask,
            "reachable_count": 0,
            "start_valid": False,
            "reason": "start_not_traversable",
            "start_xy": [int(start[0]), int(start[1])],
        }

    queue: deque[tuple[int, int]] = deque([start])
    reachable_mask[start] = True
    offsets = _neighbor_offsets_2d(allow_diagonal)
    reachable_count = 0

    while queue:
        ci, cj = queue.popleft()
        reachable_count += 1
        for di, dj in offsets:
            nxt = (ci + di, cj + dj)
            if not _in_bounds_xy(nxt, shape):
                continue
            if bool(reachable_mask[nxt]) or not bool(grid[nxt]):
                continue
            reachable_mask[nxt] = True
            queue.append(nxt)

    return {
        "reachable_mask": reachable_mask,
        "reachable_count": int(reachable_count),
        "start_valid": True,
        "reason": "ok",
        "start_xy": [int(start[0]), int(start[1])],
    }


def nearest_traversable_cell(
    traversable: np.ndarray,
    start_xy: tuple[int, int] | list[int] | np.ndarray,
    max_radius_cells: int = 5,
) -> dict[str, Any]:
    """Find the nearest traversable 2D cell within a local search radius."""

    grid = _validate_traversable_2d(traversable)
    shape = (int(grid.shape[0]), int(grid.shape[1]))
    start = _xy_tuple(start_xy)
    radius = int(max_radius_cells)
    if radius < 0:
        raise ValueError("max_radius_cells must be non-negative")

    if _in_bounds_xy(start, shape) and bool(grid[start]):
        return {
            "found": True,
            "cell": [int(start[0]), int(start[1])],
            "distance_cells": 0.0,
            "reason": "start_is_traversable",
        }

    i0 = max(0, start[0] - radius)
    i1 = min(shape[0], start[0] + radius + 1)
    j0 = max(0, start[1] - radius)
    j1 = min(shape[1], start[1] + radius + 1)
    if i0 >= i1 or j0 >= j1:
        return {
            "found": False,
            "cell": None,
            "distance_cells": math.inf,
            "reason": "not_found_within_radius",
        }

    local = grid[i0:i1, j0:j1]
    hits = np.argwhere(local)
    if hits.size == 0:
        return {
            "found": False,
            "cell": None,
            "distance_cells": math.inf,
            "reason": "not_found_within_radius",
        }

    best_cell: tuple[int, int] | None = None
    best_distance = math.inf
    for local_i, local_j in hits:
        cell = (int(i0 + local_i), int(j0 + local_j))
        distance = math.hypot(float(cell[0] - start[0]), float(cell[1] - start[1]))
        if distance > float(radius) + 1e-12:
            continue
        if (
            distance < best_distance - 1e-12
            or (abs(distance - best_distance) <= 1e-12 and best_cell is not None and cell < best_cell)
            or best_cell is None
        ):
            best_distance = float(distance)
            best_cell = cell

    if best_cell is None:
        return {
            "found": False,
            "cell": None,
            "distance_cells": math.inf,
            "reason": "not_found_within_radius",
        }

    return {
        "found": True,
        "cell": [int(best_cell[0]), int(best_cell[1])],
        "distance_cells": float(best_distance),
        "reason": "found_nearest_traversable",
    }


def frontier_reachable_candidate_mask(
    traversable: np.ndarray,
    start_xy: tuple[int, int] | list[int] | np.ndarray,
    frontier_adjacent_free_mask: np.ndarray,
    allow_diagonal: bool = True,
) -> dict[str, Any]:
    """Intersect the start component with a 2D frontier-adjacent FREE mask."""

    grid = _validate_traversable_2d(traversable)
    frontier_mask = np.asarray(frontier_adjacent_free_mask, dtype=bool)
    if frontier_mask.shape != grid.shape:
        raise ValueError(
            f"frontier_adjacent_free_mask shape {frontier_mask.shape} differs from traversable {grid.shape}"
        )

    component = connected_component_from_start(grid, start_xy, allow_diagonal=allow_diagonal)
    reachable_mask = np.asarray(component["reachable_mask"], dtype=bool)
    candidate_mask = reachable_mask & frontier_mask
    return {
        "candidate_mask": candidate_mask,
        "reachable_mask": reachable_mask,
        "candidate_count": int(np.count_nonzero(candidate_mask)),
        "reachable_count": int(component["reachable_count"]),
        "start_valid": bool(component["start_valid"]),
        "reason": str(component["reason"]),
        "start_xy": component["start_xy"],
    }


def astar_2d(
    traversable: np.ndarray,
    start_xy: tuple[int, int] | list[int] | np.ndarray,
    goal_xy: tuple[int, int] | list[int] | np.ndarray,
    allow_diagonal: bool = True,
) -> dict[str, Any]:
    """Run A* on a 2D boolean traversability grid."""

    grid = _validate_traversable_2d(traversable)

    shape = (int(grid.shape[0]), int(grid.shape[1]))
    start = _xy_tuple(start_xy)
    goal = _xy_tuple(goal_xy)

    if not _in_bounds_xy(start, shape):
        return {"reachable": False, "path": [], "cost_cells": math.inf, "num_expanded": 0, "reason": "start_out_of_bounds"}
    if not _in_bounds_xy(goal, shape):
        return {"reachable": False, "path": [], "cost_cells": math.inf, "num_expanded": 0, "reason": "goal_out_of_bounds"}
    if not bool(grid[start]):
        return {"reachable": False, "path": [], "cost_cells": math.inf, "num_expanded": 0, "reason": "start_not_traversable"}
    if not bool(grid[goal]):
        return {"reachable": False, "path": [], "cost_cells": math.inf, "num_expanded": 0, "reason": "goal_not_traversable"}
    if start == goal:
        return {"reachable": True, "path": [start], "cost_cells": 0.0, "num_expanded": 1, "reason": "start_is_goal"}

    cardinal = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
    diagonal = [
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    ]
    neighbors = cardinal + (diagonal if allow_diagonal else [])

    def heuristic(node: tuple[int, int]) -> float:
        return math.hypot(float(goal[0] - node[0]), float(goal[1] - node[1]))

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (heuristic(start), 0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score = np.full(shape, np.inf, dtype=np.float64)
    g_score[start] = 0.0
    closed = np.zeros(shape, dtype=bool)
    num_expanded = 0

    while open_heap:
        _, current_g, current = heapq.heappop(open_heap)
        if closed[current]:
            continue
        if current_g > float(g_score[current]) + 1e-12:
            continue

        closed[current] = True
        num_expanded += 1

        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return {
                "reachable": True,
                "path": path,
                "cost_cells": float(g_score[goal]),
                "num_expanded": int(num_expanded),
                "reason": "reached",
            }

        ci, cj = current
        for di, dj, step_cost in neighbors:
            nxt = (ci + di, cj + dj)
            if not _in_bounds_xy(nxt, shape) or not bool(grid[nxt]) or bool(closed[nxt]):
                continue
            tentative_g = float(g_score[current]) + float(step_cost)
            if tentative_g + 1e-12 < float(g_score[nxt]):
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                heapq.heappush(open_heap, (tentative_g + heuristic(nxt), tentative_g, nxt))

    return {
        "reachable": False,
        "path": [],
        "cost_cells": math.inf,
        "num_expanded": int(num_expanded),
        "reason": "no_path",
    }


def path_length_m(path: list[tuple[int, int]], voxel_size: float) -> float:
    """Return metric length of a 2D grid path."""

    if len(path) <= 1:
        return 0.0
    total_cells = 0.0
    for a, b in zip(path, path[1:]):
        total_cells += math.hypot(float(b[0] - a[0]), float(b[1] - a[1]))
    return float(total_cells * float(voxel_size))


def summarize_traversability(traversability: dict[str, Any] | np.ndarray) -> dict[str, Any]:
    """Return basic counts for a traversability-grid result."""

    if isinstance(traversability, dict):
        traversable = np.asarray(traversability["traversable"], dtype=bool)
        blocked = np.asarray(traversability["blocked"], dtype=bool)
        unknown = np.asarray(traversability["unknown"], dtype=bool)
        diagnostics = dict(traversability.get("diagnostics", {}))
    else:
        traversable = np.asarray(traversability, dtype=bool)
        blocked = np.zeros_like(traversable, dtype=bool)
        unknown = ~traversable
        diagnostics = {}
    total = int(traversable.size)
    summary = {
        "shape": [int(v) for v in traversable.shape],
        "total_count": total,
        "traversable_count": int(np.count_nonzero(traversable)),
        "blocked_count": int(np.count_nonzero(blocked)),
        "unknown_count": int(np.count_nonzero(unknown)),
        "traversable_ratio": float(np.count_nonzero(traversable) / total) if total else 0.0,
    }
    for key, value in diagnostics.items():
        summary.setdefault(key, value)
    return summary


def visualize_traversability(
    traversability: dict[str, Any],
    output_path: str | Path,
    bounds: dict[str, tuple[float, float]] | None = None,
    path_xy: list[tuple[int, int]] | None = None,
    current_xy: tuple[int, int] | None = None,
    goal_xy: tuple[int, int] | None = None,
) -> str:
    """Save a simple traversability topdown image."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    traversable = np.asarray(traversability["traversable"], dtype=bool)
    blocked = np.asarray(traversability["blocked"], dtype=bool)
    unknown = np.asarray(traversability["unknown"], dtype=bool)
    image = np.zeros(traversable.shape, dtype=np.int8)
    image[unknown] = 0
    image[traversable] = 1
    image[blocked] = 2

    if bounds is None:
        extent = [0, traversable.shape[0], 0, traversable.shape[1]]
        to_plot = lambda xy: (float(xy[0]) + 0.5, float(xy[1]) + 0.5)
    else:
        extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]
        voxel_size_x = (bounds["x"][1] - bounds["x"][0]) / float(traversable.shape[0])
        voxel_size_y = (bounds["y"][1] - bounds["y"][0]) / float(traversable.shape[1])

        def to_plot(xy: tuple[int, int]) -> tuple[float, float]:
            return (
                float(bounds["x"][0] + (xy[0] + 0.5) * voxel_size_x),
                float(bounds["y"][0] + (xy[1] + 0.5) * voxel_size_y),
            )

    cmap = ListedColormap(["#2f343f", "#9fd0d8", "#c95f5f"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(image.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")

    if path_xy:
        path_world = np.asarray([to_plot(tuple(p)) for p in path_xy], dtype=np.float64)
        ax.plot(path_world[:, 0], path_world[:, 1], color="#22a06b", linewidth=2.0, label="A* path")
    if current_xy is not None:
        xy = to_plot(current_xy)
        ax.scatter(xy[0], xy[1], s=120, c="#2563eb", marker="^", edgecolors="white", linewidths=0.8, label="current")
    if goal_xy is not None:
        xy = to_plot(goal_xy)
        ax.scatter(xy[0], xy[1], s=150, c="#22a06b", marker="*", edgecolors="black", linewidths=0.6, label="goal")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Observed-free traversability")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)
