#!/usr/bin/env python3
"""Visualization for Stage 4A-2 one-step simulator expert decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from sim_paper_expert import (
    OCCUPIED,
    FREE,
    UNKNOWN,
    SimCandidateView,
    detect_frontier_voxels,
    grid_to_world,
    normalize_bounds,
    raycast_visible_voxels_observed,
)
from sim_prediction_layer import SimPredictionLayer
from astar_planner import build_traversability_grid, connected_component_from_start

STATE_CMAP = ListedColormap(["#2f343f", "#9fd0d8", "#c95f5f"])
STATE_NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], STATE_CMAP.N)
STATE_TICKS = [int(UNKNOWN), int(FREE), int(OCCUPIED)]
STATE_TICKLABELS = ["unknown", "free", "occupied"]


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def project_topdown(observed_state: np.ndarray) -> np.ndarray:
    occupied = np.any(observed_state == OCCUPIED, axis=2)
    free = np.any(observed_state == FREE, axis=2)
    topdown = np.full(observed_state.shape[:2], UNKNOWN, dtype=np.int8)
    topdown[free] = FREE
    topdown[occupied] = OCCUPIED
    return topdown


def project_mask_topdown(mask: np.ndarray) -> np.ndarray:
    return np.any(np.asarray(mask, dtype=bool), axis=2)


def prediction_masks(
    prediction_layer: Any,
    observed_state: np.ndarray,
    tau: float,
) -> dict[str, np.ndarray]:
    if prediction_layer is None:
        shape = tuple(int(v) for v in observed_state.shape)
        empty = np.zeros(shape, dtype=bool)
        return {"valid": empty, "predicted": empty, "predicted_unmeasured": empty, "predicted_occupied": empty}
    valid = np.asarray(prediction_layer.valid, dtype=bool)
    confidence = np.asarray(prediction_layer.confidence, dtype=np.float32)
    occupied_prob = np.asarray(prediction_layer.occupied_prob, dtype=np.float32)
    if valid.shape != observed_state.shape or confidence.shape != observed_state.shape:
        raise ValueError("prediction layer shape differs from observed_state")
    predicted = valid & (confidence >= float(tau))
    measured = observed_state != UNKNOWN
    predicted_unmeasured = predicted & ~measured
    predicted_occupied = predicted_unmeasured & (occupied_prob >= 0.5)
    return {
        "valid": valid,
        "predicted": predicted,
        "predicted_unmeasured": predicted_unmeasured,
        "predicted_occupied": predicted_occupied,
    }


def sample_rows(rows: np.ndarray, max_count: int, seed: int) -> np.ndarray:
    if len(rows) <= int(max_count):
        return rows
    rng = np.random.default_rng(seed)
    choice = rng.choice(len(rows), size=int(max_count), replace=False)
    return rows[np.sort(choice)]


def _candidate_records_from_result(result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    def as_record(candidate: SimCandidateView, rank: int | None = None) -> dict[str, Any]:
        return {
            "id": int(candidate.id),
            "grid_position": [int(v) for v in candidate.grid_position],
            "world_position": [float(v) for v in candidate.world_position],
            "yaw": float(candidate.yaw),
            "valid": bool(candidate.valid),
            "invalid_reason": str(candidate.invalid_reason),
            "candidate_source": str(candidate.candidate_source),
            "final_score": float(candidate.final_score),
            "gain_exp": float(candidate.gain_exp),
            "gain_sc": float(candidate.gain_sc),
            "gain_hybrid": float(candidate.gain_hybrid),
            "path_cost": float(candidate.path_cost),
            "visible_count": int(candidate.visible_count),
            "astar_path_xy": [[int(v) for v in xy] for xy in candidate.astar_path_xy],
            "astar_path_length_m": float(candidate.astar_path_length_m),
            "astar_num_expanded": int(candidate.astar_num_expanded),
            "astar_reachable": bool(candidate.astar_reachable),
            "rank": rank,
        }

    all_records = [as_record(candidate) for candidate in result["all_candidates"]]
    top_records = [as_record(candidate, rank=rank) for rank, candidate in enumerate(result["top_candidates"])]
    best_record = as_record(result["best_candidate"], rank=0)
    return all_records, top_records, best_record


def _candidate_records_from_files(
    decision_json: str | Path,
    candidates_jsonl: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    decision = load_json(decision_json)
    all_records = load_jsonl(candidates_jsonl)
    top_records = decision["top_candidates"]
    best_record = decision["best_candidate"]
    return all_records, top_records, best_record, decision


def _draw_state_colorbar(fig: plt.Figure, ax: plt.Axes) -> None:
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=STATE_NORM, cmap=STATE_CMAP),
        ax=ax,
        ticks=STATE_TICKS,
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(STATE_TICKLABELS)


def _records_to_xy(records: list[dict[str, Any]]) -> np.ndarray:
    if not records:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray([record["world_position"][:2] for record in records], dtype=np.float32)


def _path_xy_to_world(path_xy: list[Any], bounds: dict[str, tuple[float, float]], voxel_size: float) -> np.ndarray:
    if not path_xy:
        return np.zeros((0, 2), dtype=np.float32)
    points = []
    for xy in path_xy:
        i, j = int(xy[0]), int(xy[1])
        points.append(grid_to_world((i, j, 0), bounds, voxel_size)[:2])
    return np.asarray(points, dtype=np.float32)


def _candidate_from_record(record: dict[str, Any]) -> SimCandidateView:
    return SimCandidateView(
        id=int(record["id"]),
        grid_position=tuple(int(v) for v in record["grid_position"]),
        world_position=tuple(float(v) for v in record["world_position"]),
        yaw=float(record.get("yaw", 0.0)),
        valid=bool(record.get("valid", True)),
        invalid_reason=str(record.get("invalid_reason", "")),
        candidate_source=str(record.get("candidate_source", "frontier")),
    )


def _draw_candidate_points(
    ax: plt.Axes,
    all_candidates: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
    best_candidate: dict[str, Any],
) -> None:
    all_xy = _records_to_xy(all_candidates)
    if len(all_xy):
        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=22,
            c="#111827",
            alpha=0.58,
            edgecolors="white",
            linewidths=0.3,
            label="candidates",
        )
    top_xy = _records_to_xy(top_candidates)
    if len(top_xy):
        ax.scatter(
            top_xy[:, 0],
            top_xy[:, 1],
            s=82,
            facecolors="none",
            edgecolors="#f28e2b",
            linewidths=1.6,
            label=f"top-{len(top_candidates)}",
        )
    best_xy = np.asarray(best_candidate["world_position"][:2], dtype=np.float32)
    ax.scatter(best_xy[0], best_xy[1], s=210, c="#22a06b", marker="*", edgecolors="black", linewidths=0.7, label="best")


def save_topdown_visualization(
    observed_state: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    current_pose_world: list[float] | np.ndarray,
    frontier_voxels: np.ndarray,
    all_candidates: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
    best_candidate: dict[str, Any],
    output_dir: str | Path,
    max_frontier_points: int = 3500,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    topdown = project_topdown(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")

    sampled_frontiers = sample_rows(frontier_voxels, max_count=max_frontier_points, seed=23)
    if len(sampled_frontiers):
        frontier_xy = np.asarray([grid_to_world(row, bounds, voxel_size)[:2] for row in sampled_frontiers], dtype=np.float32)
        ax.scatter(
            frontier_xy[:, 0],
            frontier_xy[:, 1],
            s=4,
            c="#f2c14e",
            alpha=0.45,
            linewidths=0,
            label=f"frontiers ({len(sampled_frontiers)} shown)",
        )

    all_xy = _records_to_xy(all_candidates)
    if len(all_xy):
        ax.scatter(
            all_xy[:, 0],
            all_xy[:, 1],
            s=24,
            c="#1f2933",
            alpha=0.72,
            edgecolors="white",
            linewidths=0.4,
            label="sampled candidates",
        )

    top_xy = _records_to_xy(top_candidates)
    if len(top_xy):
        ax.scatter(
            top_xy[:, 0],
            top_xy[:, 1],
            s=88,
            facecolors="none",
            edgecolors="#f28e2b",
            linewidths=1.8,
            label=f"top-{len(top_candidates)}",
        )

    current_xy = np.asarray(current_pose_world[:2], dtype=np.float32)
    best_xy = np.asarray(best_candidate["world_position"][:2], dtype=np.float32)
    ax.scatter(current_xy[0], current_xy[1], s=170, c="#2563eb", marker="^", edgecolors="white", linewidths=1.0, label="current pose")
    ax.scatter(best_xy[0], best_xy[1], s=230, c="#2ca25f", marker="*", edgecolors="black", linewidths=0.8, label="best candidate")
    best_path = _path_xy_to_world(best_candidate.get("astar_path_xy", []), bounds, voxel_size)
    if len(best_path):
        ax.plot(best_path[:, 0], best_path[:, 1], color="#22a06b", linewidth=2.4, label="best A* path")
    else:
        ax.annotate(
            "",
            xy=(best_xy[0], best_xy[1]),
            xytext=(current_xy[0], current_xy[1]),
            arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#2ca25f"},
        )

    yaw = float(best_candidate.get("yaw", 0.0))
    yaw_len = 0.45
    ax.arrow(
        best_xy[0],
        best_xy[1],
        yaw_len * np.cos(yaw),
        yaw_len * np.sin(yaw),
        width=0.018,
        head_width=0.11,
        head_length=0.14,
        color="#166534",
        length_includes_head=True,
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Stage 4A-2 expert topdown\n"
        f"best id {best_candidate['id']} score {float(best_candidate['final_score']):.3f}"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    _draw_state_colorbar(fig, ax)

    out_path = output_dir / "expert_topdown.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_traversability_topdown(
    traversability: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    current_grid: list[int] | np.ndarray,
    all_candidates: list[dict[str, Any]],
    best_candidate: dict[str, Any],
    output_dir: str | Path,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traversable = np.asarray(traversability["traversable"], dtype=bool)
    blocked = np.asarray(traversability["blocked"], dtype=bool)
    unknown = np.asarray(traversability["unknown"], dtype=bool)
    image = np.zeros(traversable.shape, dtype=np.int8)
    image[unknown] = 0
    image[traversable] = 1
    reachable = np.asarray(traversability.get("reachable_mask", np.zeros_like(traversable)), dtype=bool)
    image[reachable] = 3
    image[blocked] = 2
    cmap = ListedColormap(["#2f343f", "#9fd0d8", "#c95f5f", "#4fb286"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(image.T, origin="lower", extent=extent, cmap=cmap, norm=norm, interpolation="nearest")

    reachable_records = [r for r in all_candidates if bool(r.get("astar_reachable", False)) or bool(r.get("valid", False))]
    unreachable_records = [
        r
        for r in all_candidates
        if str(r.get("invalid_reason", "")).startswith("unreachable_astar:")
        or (not bool(r.get("astar_reachable", False)) and not bool(r.get("valid", True)))
    ]
    reachable_xy = _records_to_xy(reachable_records)
    if len(reachable_xy):
        ax.scatter(
            reachable_xy[:, 0],
            reachable_xy[:, 1],
            s=24,
            c="#111827",
            alpha=0.65,
            edgecolors="white",
            linewidths=0.35,
            label="reachable/scored candidates",
        )
    unreachable_xy = _records_to_xy(unreachable_records)
    if len(unreachable_xy):
        ax.scatter(
            unreachable_xy[:, 0],
            unreachable_xy[:, 1],
            s=28,
            c="#b91c1c",
            alpha=0.82,
            marker="x",
            linewidths=0.9,
            label="unreachable candidates",
        )

    current_xy = grid_to_world((int(current_grid[0]), int(current_grid[1]), 0), bounds, voxel_size)[:2]
    best_xy = np.asarray(best_candidate["world_position"][:2], dtype=np.float32)
    best_path = _path_xy_to_world(best_candidate.get("astar_path_xy", []), bounds, voxel_size)
    if len(best_path):
        ax.plot(best_path[:, 0], best_path[:, 1], color="#22a06b", linewidth=2.4, label="best A* path")
    ax.scatter(current_xy[0], current_xy[1], s=150, c="#2563eb", marker="^", edgecolors="white", linewidths=0.9, label="current")
    if diagnostics is not None and diagnostics.get("snapped_current_xy") is not None:
        snapped = diagnostics["snapped_current_xy"]
        snapped_xy = grid_to_world((int(snapped[0]), int(snapped[1]), 0), bounds, voxel_size)[:2]
        if bool(diagnostics.get("snapped_current", False)):
            ax.scatter(
                snapped_xy[0],
                snapped_xy[1],
                s=105,
                c="#0f766e",
                marker="o",
                edgecolors="white",
                linewidths=0.8,
                label="snapped start",
            )
    ax.scatter(best_xy[0], best_xy[1], s=210, c="#22a06b", marker="*", edgecolors="black", linewidths=0.7, label="best")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    source = "" if diagnostics is None else f", source={diagnostics.get('candidate_source', 'unknown')}"
    ax.set_title(f"Stage 4A-3.6 observed-free reachability{source}")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        ticks=[0, 1, 2, 3],
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.set_yticklabels(["unknown", "traversable", "blocked", "reachable"])

    out_path = output_dir / "traversability_topdown.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_prediction_overlay_topdown(
    observed_state: np.ndarray,
    prediction_layer: Any,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    all_candidates: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
    best_candidate: dict[str, Any],
    output_dir: str | Path,
    tau: float = 0.1,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = prediction_masks(prediction_layer, observed_state, tau=tau)
    topdown = project_topdown(observed_state)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]

    valid_xy = project_mask_topdown(masks["valid"])
    predicted_unmeasured_xy = project_mask_topdown(masks["predicted_unmeasured"])
    occupied_xy = project_mask_topdown(masks["predicted_occupied"])

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    ax.imshow(
        np.ma.masked_where(~valid_xy.T, valid_xy.T),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#3b82f6"]),
        alpha=0.20,
        interpolation="nearest",
    )
    ax.imshow(
        np.ma.masked_where(~predicted_unmeasured_xy.T, predicted_unmeasured_xy.T),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#d946ef"]),
        alpha=0.36,
        interpolation="nearest",
    )
    ax.imshow(
        np.ma.masked_where(~occupied_xy.T, occupied_xy.T),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#ef4444"]),
        alpha=0.46,
        interpolation="nearest",
    )
    _draw_candidate_points(ax, all_candidates, top_candidates, best_candidate)

    best_path = _path_xy_to_world(best_candidate.get("astar_path_xy", []), bounds, voxel_size)
    if len(best_path):
        ax.plot(best_path[:, 0], best_path[:, 1], color="#22a06b", linewidth=2.3, label="best A* path")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Stage 4A-5.1 prediction overlay\n"
        f"valid={int(np.count_nonzero(masks['valid']))}, P={int(np.count_nonzero(masks['predicted_unmeasured']))}, "
        f"occupied P={int(np.count_nonzero(masks['predicted_occupied']))}"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    _draw_state_colorbar(fig, ax)

    out_path = output_dir / "prediction_overlay_topdown.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_predicted_unmeasured_visible_topdown(
    observed_state: np.ndarray,
    prediction_layer: Any,
    bounds: dict[str, tuple[float, float]],
    voxel_size: float,
    all_candidates: list[dict[str, Any]],
    top_candidates: list[dict[str, Any]],
    best_candidate: dict[str, Any],
    diagnostics: dict[str, Any],
    output_dir: str | Path,
    tau: float = 0.1,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks = prediction_masks(prediction_layer, observed_state, tau=tau)
    predicted_unmeasured = masks["predicted_unmeasured"]
    visible_mask = np.zeros(observed_state.shape, dtype=bool)
    best_visible_mask = np.zeros(observed_state.shape, dtype=bool)

    raycast_kwargs = {
        "max_range_voxels": int(diagnostics.get("max_range_voxels", 50)),
        "num_yaw": int(diagnostics.get("num_yaw", 32)),
        "num_pitch": int(diagnostics.get("num_pitch", 7)),
        "fov_yaw_deg": float(diagnostics.get("fov_yaw_deg", 90.0)),
        "fov_pitch_deg": float(diagnostics.get("fov_pitch_deg", 60.0)),
    }
    for record in top_candidates:
        candidate = _candidate_from_record(record)
        if not candidate.valid:
            continue
        for voxel in raycast_visible_voxels_observed(candidate, observed_state, **raycast_kwargs):
            visible_mask[voxel] = True
            if int(record["id"]) == int(best_candidate["id"]):
                best_visible_mask[voxel] = True

    visible_p = visible_mask & predicted_unmeasured
    best_visible_p = best_visible_mask & predicted_unmeasured
    topdown = project_topdown(observed_state)
    p_xy = project_mask_topdown(predicted_unmeasured)
    visible_xy = project_mask_topdown(visible_p)
    best_visible_xy = project_mask_topdown(best_visible_p)
    extent = [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.imshow(topdown.T, origin="lower", extent=extent, cmap=STATE_CMAP, norm=STATE_NORM, interpolation="nearest")
    ax.imshow(
        np.ma.masked_where(~p_xy.T, p_xy.T),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#d946ef"]),
        alpha=0.22,
        interpolation="nearest",
    )
    ax.imshow(
        np.ma.masked_where(~visible_xy.T, visible_xy.T),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#facc15"]),
        alpha=0.58,
        interpolation="nearest",
    )
    ax.imshow(
        np.ma.masked_where(~best_visible_xy.T, best_visible_xy.T),
        origin="lower",
        extent=extent,
        cmap=ListedColormap(["#22c55e"]),
        alpha=0.70,
        interpolation="nearest",
    )
    _draw_candidate_points(ax, all_candidates, top_candidates, best_candidate)

    best_path = _path_xy_to_world(best_candidate.get("astar_path_xy", []), bounds, voxel_size)
    if len(best_path):
        ax.plot(best_path[:, 0], best_path[:, 1], color="#16a34a", linewidth=2.3, label="best A* path")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        "Predicted-unmeasured visible topdown\n"
        f"top visible P={int(np.count_nonzero(visible_p))}, best visible P={int(np.count_nonzero(best_visible_p))}"
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    _draw_state_colorbar(fig, ax)

    out_path = output_dir / "predicted_unmeasured_visible_topdown.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_score_bar(
    top_candidates: list[dict[str, Any]],
    gain_mode: str,
    prediction_mode: str,
    output_dir: str | Path,
) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = [int(record["id"]) for record in top_candidates]
    scores = np.asarray([float(record["final_score"]) for record in top_candidates], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(ids))
    colors = ["#2ca25f" if idx == 0 else "#f28e2b" for idx in range(len(ids))]
    ax.bar(x, scores, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([str(candidate_id) for candidate_id in ids], rotation=45, ha="right")
    ax.set_xlabel("candidate id")
    ax.set_ylabel("final_score")
    ax.set_title(f"Top candidate scores: gain_mode={gain_mode}, prediction_mode={prediction_mode}")
    ax.grid(axis="y", alpha=0.25)

    out_path = output_dir / "expert_score_bar.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return str(out_path)


def save_expert_visualizations(
    observed_state: np.ndarray,
    result: dict[str, Any],
    output_dir: str | Path,
    prediction_layer: Any | None = None,
) -> dict[str, str]:
    diagnostics = result["diagnostics"]
    bounds = normalize_bounds(diagnostics["bounds"])
    voxel_size = float(diagnostics["voxel_size"])
    all_records, top_records, best_record = _candidate_records_from_result(result)
    frontier_voxels = result.get("frontier_voxels")
    if frontier_voxels is None:
        frontier_voxels = detect_frontier_voxels(observed_state)
    current_pose_world = np.asarray(result["current_pose_world"], dtype=np.float32)

    topdown_path = save_topdown_visualization(
        observed_state=observed_state,
        bounds=bounds,
        voxel_size=voxel_size,
        current_pose_world=current_pose_world,
        frontier_voxels=np.asarray(frontier_voxels, dtype=np.int32),
        all_candidates=all_records,
        top_candidates=top_records,
        best_candidate=best_record,
        output_dir=output_dir,
    )
    score_bar_path = save_score_bar(
        top_candidates=top_records,
        gain_mode=str(result["gain_mode"]),
        prediction_mode=str(result["prediction_mode"]),
        output_dir=output_dir,
    )
    paths = {"topdown": topdown_path, "score_bar": score_bar_path}
    traversability = result.get("traversability")
    if isinstance(traversability, dict):
        paths["traversability_topdown"] = save_traversability_topdown(
            traversability=traversability,
            bounds=bounds,
            voxel_size=voxel_size,
            current_grid=np.asarray(result["current_grid"], dtype=np.int32),
            all_candidates=all_records,
            best_candidate=best_record,
            output_dir=output_dir,
            diagnostics=diagnostics,
        )
    if prediction_layer is not None:
        tau = float(diagnostics.get("tau", 0.1))
        paths["prediction_overlay_topdown"] = save_prediction_overlay_topdown(
            observed_state=observed_state,
            prediction_layer=prediction_layer,
            bounds=bounds,
            voxel_size=voxel_size,
            all_candidates=all_records,
            top_candidates=top_records,
            best_candidate=best_record,
            output_dir=output_dir,
            tau=tau,
        )
        paths["predicted_unmeasured_visible_topdown"] = save_predicted_unmeasured_visible_topdown(
            observed_state=observed_state,
            prediction_layer=prediction_layer,
            bounds=bounds,
            voxel_size=voxel_size,
            all_candidates=all_records,
            top_candidates=top_records,
            best_candidate=best_record,
            diagnostics=diagnostics,
            output_dir=output_dir,
            tau=tau,
        )
    return paths


def save_visualizations_from_files(
    observed_state_path: str | Path,
    decision_json: str | Path,
    candidates_jsonl: str | Path,
    output_dir: str | Path,
    prediction_npz: str | Path | None = None,
    tau: float | None = None,
) -> dict[str, str]:
    observed_state = np.load(observed_state_path)
    all_records, top_records, best_record, decision = _candidate_records_from_files(decision_json, candidates_jsonl)
    diagnostics = decision["diagnostics"]
    bounds = normalize_bounds(diagnostics["bounds"])
    voxel_size = float(diagnostics["voxel_size"])
    frontier_voxels = detect_frontier_voxels(observed_state)
    current_pose_world = np.asarray(decision["current_pose"]["world"], dtype=np.float32)

    topdown_path = save_topdown_visualization(
        observed_state=observed_state,
        bounds=bounds,
        voxel_size=voxel_size,
        current_pose_world=current_pose_world,
        frontier_voxels=frontier_voxels,
        all_candidates=all_records,
        top_candidates=top_records,
        best_candidate=best_record,
        output_dir=output_dir,
    )
    score_bar_path = save_score_bar(
        top_candidates=top_records,
        gain_mode=str(decision["gain_mode"]),
        prediction_mode=str(decision["prediction_mode"]),
        output_dir=output_dir,
    )
    paths = {"topdown": topdown_path, "score_bar": score_bar_path}
    if str(decision.get("path_cost_mode", "euclidean")) == "astar":
        traversability = build_traversability_grid(
            observed_state,
            voxel_size=voxel_size,
            robot_height_m=float(diagnostics.get("robot_height_m", 1.2)),
            clearance_height_m=float(diagnostics.get("clearance_height_m", 0.6)),
            robot_radius_m=float(diagnostics.get("robot_radius_m", 0.2)),
        )
        start_xy = diagnostics.get("astar_start_xy") or diagnostics.get("snapped_current_xy") or decision["current_pose"]["grid"][:2]
        component = connected_component_from_start(
            traversability["traversable"],
            start_xy=start_xy,
            allow_diagonal=bool(diagnostics.get("astar_allow_diagonal", True)),
        )
        traversability["reachable_mask"] = component["reachable_mask"]
        paths["traversability_topdown"] = save_traversability_topdown(
            traversability=traversability,
            bounds=bounds,
            voxel_size=voxel_size,
            current_grid=np.asarray(decision["current_pose"]["grid"], dtype=np.int32),
            all_candidates=all_records,
            best_candidate=best_record,
            output_dir=output_dir,
            diagnostics=diagnostics,
        )
    if prediction_npz is not None:
        prediction_layer = SimPredictionLayer.from_npz(prediction_npz)
        overlay_tau = float(tau if tau is not None else diagnostics.get("tau", 0.1))
        paths["prediction_overlay_topdown"] = save_prediction_overlay_topdown(
            observed_state=observed_state,
            prediction_layer=prediction_layer,
            bounds=bounds,
            voxel_size=voxel_size,
            all_candidates=all_records,
            top_candidates=top_records,
            best_candidate=best_record,
            output_dir=output_dir,
            tau=overlay_tau,
        )
        paths["predicted_unmeasured_visible_topdown"] = save_predicted_unmeasured_visible_topdown(
            observed_state=observed_state,
            prediction_layer=prediction_layer,
            bounds=bounds,
            voxel_size=voxel_size,
            all_candidates=all_records,
            top_candidates=top_records,
            best_candidate=best_record,
            diagnostics=diagnostics,
            output_dir=output_dir,
            tau=overlay_tau,
        )
    return paths


def parse_args() -> argparse.Namespace:
    default_output = Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke")
    parser = argparse.ArgumentParser(description="Visualize Stage 4A-2 simulator expert decision.")
    parser.add_argument(
        "--observed_state",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke/observed_state_step2.npy",
    )
    parser.add_argument("--decision_json", default=str(default_output / "expert_step_decision.json"))
    parser.add_argument("--candidates_jsonl", default=str(default_output / "expert_step_candidates.jsonl"))
    parser.add_argument("--output_dir", default=str(default_output))
    parser.add_argument("--prediction_npz", default=None)
    parser.add_argument("--tau", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = save_visualizations_from_files(
        observed_state_path=args.observed_state,
        decision_json=args.decision_json,
        candidates_jsonl=args.candidates_jsonl,
        output_dir=args.output_dir,
        prediction_npz=args.prediction_npz,
        tau=args.tau,
    )
    print("Stage 4A-2 expert visualization complete.")
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
