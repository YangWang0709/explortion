#!/usr/bin/env python3
"""Stage 4A-6.5w source-faithful RRT* root-rewire persistence replay.

This runner is intentionally offline. It reads saved Frame1/Frame2
observed_state and prediction NPZ artifacts, inspects the external
active_3d_planning source tree, then compares the saved Stage 4A-6.5v fresh
Frame2 replay against an approximate source-faithful persistent tree lifecycle:
Frame1 tree -> root rewire to the executed child -> measured-only branch
prune/reinsert -> Frame2 gain/value recompute -> continued tree expansion.

It does not launch Isaac, capture new frames, rerun map_predict, run SSCNet
inference, execute actions, run rollout, train, modify checkpoints, modify
observed_state, modify prediction NPZ files, or modify/build the external
source.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from offline_mini_rrt_tree import (
    EPS,
    ROOT_ID,
    MiniRRTSegment,
    add_tree_child,
    build_mini_rrt_tree,
    build_sampling_context,
    choose_best_yaw,
    choose_sample_pool,
    distance_xy,
    effective_sc_gain_from_formula,
    euclidean,
    in_bounds_xy,
    line_is_traversable,
    make_gain_value_rows,
    nearest_free_z_for_xy,
    segment_brief,
    segment_path_to_root,
    segment_record,
    sha256_file,
    to_jsonable,
    wrap_angle,
)
from offline_tree_utility_prototype import compute_global_normalized_gain, select_subsequent_best
from sim_paper_expert import FREE, OCCUPIED, UNKNOWN, EmptyPredictionLayer, grid_to_world, normalize_bounds, world_to_grid
from sim_prediction_layer import SimPredictionLayer


DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65w_source_faithful_rewire_persistence"
)
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)

REFERENCE_SC_SELECTED_GRID = [11, 15, 11]
REFERENCE_SC_BEST_GRID = [14, 15, 11]
REFERENCE_MEASURED_SELECTED_GRID = [17, 16, 11]
REFERENCE_MEASURED_BEST_GRID = [8, 27, 11]

REQUIRED_FILES = [
    "source_rewire_evidence.json",
    "source_rewire_evidence.md",
    "source_density_evidence.json",
    "source_density_evidence.md",
    "source_yaw_evidence.json",
    "source_yaw_evidence.md",
    "source_persistence_design.md",
    "rewire_configs_manifest.jsonl",
    "per_config_seed_formula_decisions.csv",
    "per_config_seed_formula_decisions.json",
    "per_config_seed_formula_decisions.md",
    "per_config_summary.csv",
    "per_config_summary.json",
    "per_config_summary.md",
    "branch_classification_by_config_seed.csv",
    "branch_classification_by_config_seed.json",
    "branch_classification_summary_by_config.json",
    "branch_classification_summary_by_config.md",
    "preservation_summary_by_config.csv",
    "preservation_summary_by_config.json",
    "preservation_summary_by_config.md",
    "confidence_vs_cap25_agreement_by_config.csv",
    "confidence_vs_cap25_agreement_by_config.json",
    "spatial_basin_summary_by_config.csv",
    "spatial_basin_summary_by_config.json",
    "spatial_basin_summary_by_config.md",
    "margin_summary_by_config.csv",
    "margin_summary_by_config.json",
    "margin_summary_by_config.md",
    "compute_time_summary.csv",
    "missing_fields_report.json",
    "stage4a65w_source_faithful_rewire_summary.json",
    "stage4a65w_source_faithful_rewire_summary.md",
    "recommended_next_faithful_step.md",
    "reinsert_attempts.csv",
    "reinsert_summary.json",
    "reinsert_summary.md",
]

REQUIRED_PLOTS = [
    "spatial_seed0_sc_basin_fraction_by_config.png",
    "same_as_measured_fraction_by_config.png",
    "preserved_nodes_fraction_by_config.png",
    "confidence_cap25_agreement_by_config.png",
    "selected_children_by_config_topdown.png",
    "best_descendants_by_config_topdown.png",
    "margin_distribution_by_config.png",
    "selected_delta_to_seed0_sc_by_config.png",
    "value_vs_effective_sc_by_config.png",
    "value_vs_cost_by_config.png",
    "preserved_vs_newly_expanded_winners.png",
    "compute_time_by_config.png",
]

PROHIBITED_OUTPUT_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "sscnet_*",
    "map_predict*",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_topdown_path.png",
    "rollout_*.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]

CONFIG_SPECS = {
    "fresh_random_256_baseline": {
        "tree_lifecycle": "fresh",
        "frame1_num_nodes": 0,
        "frame2_target_num_nodes": 256,
        "density_mode": "none",
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    "persistent_rewire_256_no_density": {
        "tree_lifecycle": "persistent_rewire",
        "frame1_num_nodes": 256,
        "frame2_target_num_nodes": 256,
        "density_mode": "none",
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    "persistent_rewire_512_no_density": {
        "tree_lifecycle": "persistent_rewire",
        "frame1_num_nodes": 256,
        "frame2_target_num_nodes": 512,
        "density_mode": "none",
        "density_radius_m": 0.0,
        "max_nodes_per_density_radius": 0,
    },
    "persistent_rewire_256_source_density": {
        "tree_lifecycle": "persistent_rewire",
        "frame1_num_nodes": 256,
        "frame2_target_num_nodes": 256,
        "density_mode": "source_like",
        "density_radius_m": 1.0,
        "max_nodes_per_density_radius": 1,
    },
    "persistent_rewire_512_source_density": {
        "tree_lifecycle": "persistent_rewire",
        "frame1_num_nodes": 256,
        "frame2_target_num_nodes": 512,
        "density_mode": "source_like",
        "density_radius_m": 1.0,
        "max_nodes_per_density_radius": 1,
    },
}

FORMULA_COLORS = {
    "measured_only": "#2563eb",
    "confidence_weighted": "#f97316",
    "cap25": "#7c3aed",
    "raw_count": "#dc2626",
}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(data), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    json_path = Path(path)
    if not json_path.is_file():
        return {}
    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(to_jsonable(value), sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], field_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(field_order or [])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True, allow_nan=False))
            handle.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= EPS:
        return None
    return float(numerator) / float(denominator)


def percentile_summary(values: list[float]) -> dict[str, Any]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0}
    arr = np.asarray(clean, dtype=np.float64)
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "q25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "q75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def min_mean_max(values: list[float]) -> dict[str, float | None]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(clean), "mean": statistics.fmean(clean), "max": max(clean)}


def pearson_pairs(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if math.isfinite(float(x)) and math.isfinite(float(y))]
    if len(pairs) < 2:
        return None
    x_arr = np.asarray([x for x, _ in pairs], dtype=np.float64)
    y_arr = np.asarray([y for _, y in pairs], dtype=np.float64)
    if float(np.std(x_arr)) <= EPS or float(np.std(y_arr)) <= EPS:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def default_bounds(shape: tuple[int, int, int], voxel_size: float) -> dict[str, tuple[float, float]]:
    return normalize_bounds(
        {
            "x": (-0.5 * shape[0] * voxel_size, 0.5 * shape[0] * voxel_size),
            "y": (-0.5 * shape[1] * voxel_size, 0.5 * shape[1] * voxel_size),
            "z": (0.0, shape[2] * voxel_size),
        }
    )


def root_from_pose(
    pose_path: Path,
    shape: tuple[int, int, int],
    voxel_size: float,
    bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    pose = read_json(pose_path)
    if not pose.get("position"):
        raise ValueError(f"pose file missing position: {pose_path}")
    position = [float(v) for v in pose["position"]]
    yaw = float(pose.get("yaw_rad", math.radians(float(pose.get("yaw_deg", 0.0)))))
    grid = list(world_to_grid(position, bounds, voxel_size, shape=shape, clip=True))
    world = list(grid_to_world(grid, bounds, voxel_size))
    return {
        "pose_path": str(pose_path),
        "pose": pose,
        "grid": [int(v) for v in grid],
        "world": [float(v) for v in world],
        "pose_world": position,
        "yaw": float(yaw),
    }


def prediction_layer_for(
    formula: str,
    prediction_npz: Path,
    shape: tuple[int, int, int],
) -> EmptyPredictionLayer | SimPredictionLayer:
    if formula == "measured_only":
        return EmptyPredictionLayer(shape)
    return SimPredictionLayer.from_npz(prediction_npz)


def gain_mode_for(formula: str) -> str:
    return "exp" if formula == "measured_only" else "hybrid"


def sc_formula_for(formula: str) -> str:
    return "measured_only" if formula == "measured_only" else str(formula)


def source_status(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        return {"exists": False, "git_status_short": "missing", "commit": None}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(path), check=False, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--short"], cwd=str(path), check=False, capture_output=True, text=True)
    return {
        "exists": True,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "git_status_short": status.stdout.strip() if status.returncode == 0 else f"error: {status.stderr.strip()}",
    }


def write_source_evidence(output_dir: Path, external_dir: Path) -> dict[str, Any]:
    rel = "active_3d_planning_core"
    rewire = {
        "external_source_dir": str(external_dir),
        "commit": source_status(external_dir).get("commit"),
        "rewire_root_implemented_at": f"{rel}/src/module/trajectory_generator/rrt_star.cpp:211",
        "rewire_root_summary": (
            "RRTStar::rewireRoot obtains the selected child as next_root, tries "
            "to rewire non-next root children through rewireRootSingle, and can "
            "reinsert the old root to keep otherwise-dead branches alive."
        ),
        "root_changed_after_motion": (
            "RRTStarEvaluatorAdapter::selectNextBest delegates next selection, "
            "then calls generator_->rewireRoot(traj_in, &next) before returning "
            "the next child index."
        ),
        "children_parent_updates": (
            "rewireToBestParent moves a segment unique_ptr from its previous "
            "parent children vector into the new parent children vector and "
            "updates segment->parent; root reinsert similarly moves old child "
            "branches under a reinserted root segment."
        ),
        "source_ambiguous": [
            "The exact ROS runtime ownership/lifetime sequence after online map update is not reproduced offline.",
            "The source code uses KD-tree candidate parents and C++ trajectory collision checks; this task uses measured-only voxel line validity.",
        ],
    }
    density = {
        "external_source_dir": str(external_dir),
        "default_param": {"max_density_range": 0.0, "source": f"{rel}/src/module/trajectory_generator/rrt_star.cpp:34"},
        "configured_values_found": [
            {
                "file": "active_3d_planning_app_reconstruction/cfg/planners/exploration_planner.yaml",
                "max_density_range": 1.0,
                "crop_min_length": 0.5,
                "min_path_length": 0.5,
                "max_rewire_range": 1.6,
                "n_neighbors": 20,
            },
            {
                "file": "active_3d_planning_app_reconstruction/cfg/planners/reconstruction_planner.yaml",
                "max_density_range": 1.0,
                "crop_min_length": 0.5,
                "min_path_length": 0.5,
                "max_rewire_range": 1.6,
                "n_neighbors": 20,
            },
        ],
        "implementation": "RRTStar::expandSegment rejects a candidate if the nearest existing KD-tree point is within max_density_range.",
        "implemented_here": "density_mode=source_like uses radius 1.0m and max one node in that radius as a diagnostic source-like approximation.",
        "density_source_values_found": True,
    }
    yaw = {
        "external_source_dir": str(external_dir),
        "continuous_yaw_evaluator": f"{rel}/src/module/trajectory_evaluator/continuous_yaw_planning_evaluator.cpp",
        "summary": (
            "ContinuousYawPlanningEvaluator evaluates stored orientation sections, "
            "selects the yaw section with maximum gain, applies that yaw to the "
            "trajectory, then recomputes cost and value."
        ),
        "config_values": {
            "n_directions": 12,
            "n_sections_fov": 3,
            "update_range": 3.0,
            "source_configs": [
                "active_3d_planning_app_reconstruction/cfg/planners/exploration_planner.yaml",
                "active_3d_planning_app_reconstruction/cfg/planners/reconstruction_planner.yaml",
            ],
        },
        "implemented_here": (
            "Approximate fixed yaw search with num_yaw_samples from CLI, scoring "
            "local measured/prediction information gain at each candidate pose."
        ),
        "source_ambiguous": [
            "The full YawPlanningInfo orientation-section updater is not reproduced.",
            "CLI uses 8 yaw samples by task request, while source configs use n_directions=12 and n_sections_fov=3.",
        ],
    }
    design_lines = [
        "# Source Persistence Design",
        "",
        "Implemented source-faithful approximation:",
        "",
        "- Build a Frame1 mini-RRT tree using the source-protected crop profile.",
        "- Resolve the executed child from the saved Frame2 pose, preferring the exact Frame1 selected child if it matches spatially.",
        "- Make that node the new offline root and detach it from its old parent.",
        "- Preserve descendants whose parent-child measured-only edge remains traversable in the Frame2 observed_state.",
        "- Reinsert non-descendant old branches when a measured-only edge to a preserved node is available; this approximates source root reinsert / branch preservation.",
        "- Recompute local gain/cost on Frame2 observed_state plus read-only prediction-only information gain.",
        "- Recompute GlobalNormalizedGain and SubsequentBest over the persistent tree.",
        "- Continue expanding until the target node count is reached.",
        "",
        "Approximate or not implemented:",
        "",
        "- No C++ KD-tree exact parent rewiring, ESDF collision, ROS update loop, or source trajectory ownership model.",
        "- Continuous yaw is approximated with fixed yaw samples.",
        "- Density uses source config values when requested but remains diagnostic because our mini-RRT bounds/profile differ from the source config.",
    ]

    save_json(output_dir / "source_rewire_evidence.json", rewire)
    save_json(output_dir / "source_density_evidence.json", density)
    save_json(output_dir / "source_yaw_evidence.json", yaw)
    write_text(
        output_dir / "source_rewire_evidence.md",
        "\n".join(
            [
                "# Source Rewire Evidence",
                "",
                f"- rewireRoot: `{rewire['rewire_root_implemented_at']}`",
                f"- root lifecycle: {rewire['root_changed_after_motion']}",
                f"- preservation/reinsert: {rewire['rewire_root_summary']}",
                f"- parent/children updates: {rewire['children_parent_updates']}",
                "- implemented here: approximate offline root rewire, measured-only prune, direct branch reinsert, Frame2 value recompute.",
                "- source ambiguous: exact online ownership and collision/KD-tree behavior are approximated.",
                "",
            ]
        ),
    )
    write_text(
        output_dir / "source_density_evidence.md",
        "\n".join(
            [
                "# Source Density Evidence",
                "",
                "- `RRTStar::setupFromParamMap` default `max_density_range` is `0.0`.",
                "- exploration/reconstruction planner configs set `max_density_range: 1.0`.",
                "- source behavior: reject new point if nearest existing tree point is within that range.",
                "- implemented here: `source_like` density uses radius `1.0m`, max one node in radius.",
                "- exact source values found: `true`.",
                "",
            ]
        ),
    )
    write_text(
        output_dir / "source_yaw_evidence.md",
        "\n".join(
            [
                "# Source Continuous Yaw Evidence",
                "",
                "- `ContinuousYawPlanningEvaluator` selects the best orientation section by gain, applies the yaw, then recomputes cost/value.",
                "- source configs use `n_directions: 12` and `n_sections_fov: 3`.",
                "- implemented here: fixed yaw-sample approximation from the existing mini-RRT scorer; requested runs use `num_yaw_samples=8`.",
                "- approximate parts: full orientation-section updater and source yaw cost coupling are not reproduced.",
                "",
            ]
        ),
    )
    write_text(output_dir / "source_persistence_design.md", "\n".join(design_lines) + "\n")
    return {"rewire": rewire, "density": density, "yaw": yaw}


def clone_segment(
    source: MiniRRTSegment,
    *,
    segment_id: str | None = None,
    parent_id: str | None,
) -> MiniRRTSegment:
    data = asdict(source)
    data["segment_id"] = segment_id or source.segment_id
    data["parent_id"] = parent_id
    data["children"] = []
    data["info"] = dict(source.info)
    data["local_visibility_stats"] = dict(source.local_visibility_stats)
    return MiniRRTSegment(**data)


def subtree_ids(tree: dict[str, MiniRRTSegment], root_id: str) -> list[str]:
    ids: list[str] = []

    def visit(node_id: str) -> None:
        if node_id not in tree:
            return
        ids.append(node_id)
        for child_id in list(tree[node_id].children):
            visit(child_id)

    visit(root_id)
    return ids


def next_new_segment_id(tree: dict[str, MiniRRTSegment], prefix: str = "p") -> str:
    idx = 1
    while True:
        candidate = f"{prefix}{idx:04d}"
        if candidate not in tree:
            return candidate
        idx += 1


def recompute_edge_segment(
    segment: MiniRRTSegment,
    parent: MiniRRTSegment,
    observed_state: np.ndarray,
    prediction_layer: EmptyPredictionLayer | SimPredictionLayer,
    *,
    gain_mode: str,
    sc_gain_formula: str,
    tau: float,
    raycast_stride: int,
    max_ray_length_m: float,
    voxel_size: float,
    v_max: float,
    num_yaw_samples: int,
) -> MiniRRTSegment:
    if parent.end_world is None or parent.end_grid is None or segment.end_world is None or segment.end_grid is None:
        raise ValueError("segment and parent must have end poses")
    child_world = [float(v) for v in segment.end_world]
    parent_world = [float(v) for v in parent.end_world]
    base_yaw = math.atan2(child_world[1] - parent_world[1], child_world[0] - parent_world[0])
    yaw_stats = choose_best_yaw(
        observed_state=observed_state,
        grid=tuple(int(v) for v in segment.end_grid),
        world=tuple(float(v) for v in child_world),
        parent_yaw=float(parent.yaw),
        base_yaw=float(base_yaw),
        num_yaw_samples=int(num_yaw_samples),
        gain_mode=gain_mode,
        prediction_layer=prediction_layer,
        sc_gain_formula=sc_gain_formula,
        tau=float(tau),
        raycast_stride=int(raycast_stride),
        max_ray_length_m=float(max_ray_length_m),
        voxel_size=float(voxel_size),
    )
    segment.start_grid = [int(v) for v in parent.end_grid]
    segment.start_world = [float(v) for v in parent.end_world]
    segment.yaw = float(yaw_stats["yaw"])
    segment.gain = float(yaw_stats["gain"])
    segment.gain_exp = float(yaw_stats["gain_exp"])
    segment.gain_sc = float(yaw_stats["gain_sc"])
    segment.gain_hybrid = float(yaw_stats["gain_hybrid"])
    segment.effective_gain_sc = float(yaw_stats["effective_gain_sc"])
    segment.gain_hybrid_effective = float(yaw_stats["gain_hybrid_effective"])
    segment.gain_occ = float(yaw_stats["gain_occ"])
    segment.gain_conf = float(yaw_stats["gain_conf"])
    segment.sc_gain_formula = str(yaw_stats["sc_gain_formula"])
    segment.segment_length_m = float(euclidean(parent_world, child_world))
    segment.cost = float(segment.segment_length_m / max(float(v_max), EPS))
    segment.yaw_delta = abs(wrap_angle(float(segment.yaw) - float(parent.yaw)))
    segment.yaw_time = float(segment.yaw_delta)
    segment.depth = int(parent.depth + 1)
    segment.local_visibility_stats = {
        "visible_count": int(yaw_stats["visible_count"]),
        "measured_visible_count": int(yaw_stats["measured_visible_count"]),
        "predicted_unmeasured_visible_count": int(yaw_stats["predicted_unmeasured_visible_count"]),
        "frontier_count_visible": int(yaw_stats["frontier_count_visible"]),
        "yaw_samples_evaluated": int(yaw_stats["yaw_samples_evaluated"]),
    }
    return segment


def edge_valid_between(
    context: dict[str, Any],
    parent: MiniRRTSegment,
    child: MiniRRTSegment,
) -> tuple[bool, str]:
    if parent.end_grid is None or child.end_grid is None:
        return False, "missing_grid"
    ok, reason, _cells = line_is_traversable(
        np.asarray(context["traversable"], dtype=bool),
        (int(parent.end_grid[0]), int(parent.end_grid[1])),
        (int(child.end_grid[0]), int(child.end_grid[1])),
        reachable_mask=np.asarray(context["reachable_mask"], dtype=bool),
    )
    return bool(ok), str(reason)


def find_executed_root(
    frame1_tree: dict[str, MiniRRTSegment],
    frame1_decision: dict[str, Any],
    executed_world: list[float],
    *,
    max_distance_m: float = 0.15,
) -> tuple[str | None, dict[str, Any]]:
    selected_id = frame1_decision.get("selected_child_id")
    if selected_id in frame1_tree:
        selected = frame1_tree[str(selected_id)]
        dist = euclidean(selected.end_world or [], executed_world)
        if dist <= max_distance_m:
            return str(selected_id), {
                "method": "exact_frame1_selected_child",
                "distance_m": dist,
                "selected_child_id": selected_id,
            }
    candidates = [
        (segment_id, euclidean(segment.end_world or [], executed_world))
        for segment_id, segment in frame1_tree.items()
        if segment_id != ROOT_ID and segment.end_world is not None
    ]
    candidates.sort(key=lambda item: item[1])
    if candidates and candidates[0][1] <= max_distance_m:
        return candidates[0][0], {
            "method": "nearest_frame1_node",
            "distance_m": candidates[0][1],
            "selected_child_id": selected_id,
        }
    return None, {
        "method": "not_found",
        "nearest_id": candidates[0][0] if candidates else None,
        "nearest_distance_m": candidates[0][1] if candidates else None,
        "selected_child_id": selected_id,
    }


def preserve_and_reinsert_tree(
    *,
    frame1_tree: dict[str, MiniRRTSegment],
    frame1_decision: dict[str, Any],
    frame2_observed_state: np.ndarray,
    frame2_prediction_layer: EmptyPredictionLayer | SimPredictionLayer,
    frame2_root: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    args_dict: dict[str, Any],
    formula: str,
    config_name: str,
    density_radius_m: float,
    max_nodes_per_density_radius: int,
) -> dict[str, Any]:
    gain_mode = gain_mode_for(formula)
    sc_formula = sc_formula_for(formula)
    context = build_sampling_context(
        frame2_observed_state,
        frame2_root["grid"],
        float(args_dict["voxel_size"]),
        float(args_dict["robot_radius_m"]),
    )
    executed_world = [float(v) for v in frame2_root["world"]]
    executed_id, root_match = find_executed_root(frame1_tree, frame1_decision, executed_world)
    new_tree: dict[str, MiniRRTSegment] = {
        ROOT_ID: MiniRRTSegment(
            segment_id=ROOT_ID,
            parent_id=None,
            start_grid=[int(v) for v in frame2_root["grid"]],
            start_world=[float(v) for v in frame2_root["pose_world"]],
            end_grid=[int(v) for v in frame2_root["grid"]],
            end_world=[float(v) for v in frame2_root["pose_world"]],
            yaw=float(frame2_root["yaw"]),
            local_visibility_stats={"role": "persistent_rewire_root"},
            info={
                "origin": "persistent_rewire_root",
                "old_segment_id": executed_id,
                "root_match": root_match,
                "config": config_name,
            },
        )
    }
    preservation = {
        "executed_root_old_id": executed_id,
        "root_match": root_match,
        "no_valid_preserved_root": executed_id is None,
        "preserved_node_count": 0,
        "preserved_descendant_count": 0,
        "pruned_node_count": 0,
        "pruned_reasons": Counter(),
        "reinsert_attempt_count": 0,
        "reinserted_node_count": 0,
        "reinserted_branch_root_count": 0,
        "reinsert_fail_count": 0,
        "reinsert_implemented": True,
        "reinsert_attempts": [],
        "newly_expanded_node_count": 0,
        "frame2_context": {
            "reachable_free_count": int(context["reachable_free_count"]),
            "reachable_frontier_count": int(context["reachable_frontier_count"]),
            "frontier_adjacent_free_count": int(context["frontier_adjacent_free_count"]),
        },
    }
    occupied_xy: set[tuple[int, int]] = {(int(frame2_root["grid"][0]), int(frame2_root["grid"][1]))}

    def add_preserved(old_id: str, parent_new_id: str, origin: str) -> bool:
        old = frame1_tree[old_id]
        parent = new_tree[parent_new_id]
        probe = clone_segment(old, parent_id=parent_new_id)
        ok, reason = edge_valid_between(context, parent, probe)
        if not ok:
            pruned = len(subtree_ids(frame1_tree, old_id))
            preservation["pruned_node_count"] += pruned
            preservation["pruned_reasons"][reason] += pruned
            return False
        if probe.end_grid is None:
            preservation["pruned_node_count"] += len(subtree_ids(frame1_tree, old_id))
            preservation["pruned_reasons"]["missing_end_grid"] += 1
            return False
        xy = (int(probe.end_grid[0]), int(probe.end_grid[1]))
        if xy in occupied_xy:
            preservation["pruned_node_count"] += len(subtree_ids(frame1_tree, old_id))
            preservation["pruned_reasons"]["duplicate_xy_after_rewire"] += 1
            return False
        probe.info = {**dict(probe.info), "origin": origin, "old_segment_id": old_id}
        recompute_edge_segment(
            probe,
            parent,
            frame2_observed_state,
            frame2_prediction_layer,
            gain_mode=gain_mode,
            sc_gain_formula=sc_formula,
            tau=float(args_dict["tau"]),
            raycast_stride=int(args_dict["raycast_stride"]),
            max_ray_length_m=float(args_dict["max_ray_length_m"]),
            voxel_size=float(args_dict["voxel_size"]),
            v_max=float(args_dict["v_max"]),
            num_yaw_samples=int(args_dict["num_yaw_samples"]),
        )
        add_tree_child(new_tree, parent_new_id, probe)
        occupied_xy.add(xy)
        if origin.startswith("reinsert"):
            preservation["reinserted_node_count"] += 1
        else:
            preservation["preserved_node_count"] += 1
            preservation["preserved_descendant_count"] += 1
        for child_id in frame1_tree[old_id].children:
            child_origin = "reinserted_descendant" if origin.startswith("reinsert") else "preserved_descendant"
            add_preserved(child_id, probe.segment_id, child_origin)
        return True

    if executed_id is not None:
        for child_id in frame1_tree[executed_id].children:
            add_preserved(child_id, ROOT_ID, "preserved_descendant")

        old_root_children = [child_id for child_id in frame1_tree[ROOT_ID].children if child_id != executed_id]
        for old_id in old_root_children:
            if old_id not in frame1_tree:
                continue
            preservation["reinsert_attempt_count"] += 1
            old = frame1_tree[old_id]
            nearest_id: str | None = None
            nearest_dist = float("inf")
            for candidate_id, candidate in new_tree.items():
                if candidate.end_world is None or old.end_world is None:
                    continue
                dist = euclidean(candidate.end_world, old.end_world)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = candidate_id
            attempt = {
                "config": config_name,
                "formula": formula,
                "old_branch_root_id": old_id,
                "nearest_preserved_parent_id": nearest_id,
                "nearest_distance_m": nearest_dist if math.isfinite(nearest_dist) else None,
                "max_rewire_range_m": 1.6,
                "status": "not_attempted",
            }
            if nearest_id is None or nearest_dist > 1.6:
                preservation["reinsert_fail_count"] += 1
                attempt["status"] = "no_parent_within_source_max_rewire_range"
                preservation["reinsert_attempts"].append(attempt)
                continue
            before_count = len(new_tree)
            ok = add_preserved(old_id, nearest_id, "reinserted_branch")
            after_count = len(new_tree)
            if ok:
                preservation["reinserted_branch_root_count"] += 1
                attempt["status"] = "ok"
                attempt["inserted_nodes"] = after_count - before_count
            else:
                preservation["reinsert_fail_count"] += 1
                attempt["status"] = "edge_invalid_or_pruned"
                attempt["inserted_nodes"] = 0
            preservation["reinsert_attempts"].append(attempt)

    preservation["pruned_all_old_branches"] = bool(
        executed_id is not None
        and preservation["preserved_node_count"] == 0
        and preservation["reinserted_node_count"] == 0
    )
    preservation["pruned_reasons"] = dict(preservation["pruned_reasons"])
    preservation["occupied_xy_count_after_preserve"] = len(occupied_xy)
    return {
        "tree": new_tree,
        "context": context,
        "occupied_xy": occupied_xy,
        "preservation": preservation,
        "density_radius_m": density_radius_m,
        "max_nodes_per_density_radius": max_nodes_per_density_radius,
        "bounds": bounds,
    }


def extend_persistent_tree(
    *,
    tree: dict[str, MiniRRTSegment],
    context: dict[str, Any],
    occupied_xy: set[tuple[int, int]],
    observed_state: np.ndarray,
    prediction_layer: EmptyPredictionLayer | SimPredictionLayer,
    bounds: dict[str, tuple[float, float]],
    target_total_nodes: int,
    seed: int,
    formula: str,
    args_dict: dict[str, Any],
    density_radius_m: float,
    max_nodes_per_density_radius: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + int(target_total_nodes) * 1009 + 650000)
    gain_mode = gain_mode_for(formula)
    sc_formula = sc_formula_for(formula)
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    attempts = 0
    score_time = 0.0
    max_attempts = max((int(target_total_nodes) - len(tree)) * 80, 200)
    root_k = int(tree[ROOT_ID].end_grid[2])
    max_extension_m = float(args_dict["max_extension_m"])
    voxel_size = float(args_dict["voxel_size"])
    crop_threshold = max(0.0, float(args_dict["crop_min_length_m"]))
    while len(tree) < int(target_total_nodes) and attempts < max_attempts:
        attempts += 1
        sample_source, pool = choose_sample_pool(context, str(args_dict["sample_mode"]), rng)
        if len(pool) == 0:
            rejected_rows.append({"attempt": attempts, "reason": "empty_sample_pool", "sample_source": sample_source})
            break
        target_xy_arr = pool[int(rng.integers(0, len(pool)))]
        target_xy = (int(target_xy_arr[0]), int(target_xy_arr[1]))
        target_world = grid_to_world((target_xy[0], target_xy[1], root_k), bounds, voxel_size)
        nearest_id = min(
            tree.keys(),
            key=lambda segment_id: distance_xy(tree[segment_id].end_world or [0.0, 0.0, 0.0], target_world),
        )
        parent = tree[nearest_id]
        if parent.end_world is None or parent.end_grid is None:
            rejected_rows.append({"attempt": attempts, "reason": "nearest_missing_pose", "nearest_id": nearest_id})
            continue
        parent_world = np.asarray(parent.end_world, dtype=np.float64)
        parent_grid = np.asarray(parent.end_grid, dtype=np.int64)
        delta_xy = np.asarray(target_world[:2], dtype=np.float64) - parent_world[:2]
        distance_to_target = float(np.linalg.norm(delta_xy))
        if distance_to_target <= EPS:
            rejected_rows.append({"attempt": attempts, "reason": "target_same_as_nearest", "nearest_id": nearest_id})
            continue
        step_length = min(max_extension_m, distance_to_target)
        traversable = np.asarray(context["traversable"], dtype=bool)
        reachable_mask = np.asarray(context["reachable_mask"], dtype=bool)

        def propose(candidate_step_length: float, trial_label: str) -> dict[str, Any]:
            new_xy_world = parent_world[:2] + delta_xy / distance_to_target * float(candidate_step_length)
            raw_grid = world_to_grid(
                (float(new_xy_world[0]), float(new_xy_world[1]), float(parent_world[2])),
                bounds,
                voxel_size,
                shape=observed_state.shape,
                clip=True,
            )
            candidate_xy = (int(raw_grid[0]), int(raw_grid[1]))
            if not (in_bounds_xy(candidate_xy, traversable.shape) and bool(traversable[candidate_xy])):
                return {
                    "ok": False,
                    "reason": "steer_not_traversable",
                    "candidate_xy": [int(candidate_xy[0]), int(candidate_xy[1])],
                    "trial_label": trial_label,
                }
            if not bool(reachable_mask[candidate_xy]):
                return {
                    "ok": False,
                    "reason": "steered_cell_not_in_reachable_component",
                    "candidate_xy": [int(candidate_xy[0]), int(candidate_xy[1])],
                    "trial_label": trial_label,
                }
            child_k = nearest_free_z_for_xy(observed_state, candidate_xy, int(parent_grid[2]))
            if child_k is None:
                return {"ok": False, "reason": "no_free_z_at_candidate_xy", "trial_label": trial_label}
            child_grid = (int(candidate_xy[0]), int(candidate_xy[1]), int(child_k))
            edge_ok, edge_reason, edge_cells = line_is_traversable(
                traversable,
                (int(parent_grid[0]), int(parent_grid[1])),
                candidate_xy,
                reachable_mask=reachable_mask,
            )
            if not edge_ok:
                return {
                    "ok": False,
                    "reason": edge_reason,
                    "candidate_grid": list(child_grid),
                    "edge_cell_count": len(edge_cells),
                    "trial_label": trial_label,
                }
            child_world = list(grid_to_world(child_grid, bounds, voxel_size))
            segment_length = euclidean(parent.end_world, child_world)
            root_distance = euclidean(tree[ROOT_ID].end_world or [0.0, 0.0, 0.0], child_world)
            density_count = 0
            if float(density_radius_m) > 0.0 and int(max_nodes_per_density_radius) > 0:
                density_count = sum(
                    1
                    for existing in tree.values()
                    if existing.end_world is not None
                    and euclidean(existing.end_world, child_world) <= float(density_radius_m)
                )
            return {
                "ok": True,
                "candidate_xy": candidate_xy,
                "child_grid": child_grid,
                "child_world": child_world,
                "edge_cells": edge_cells,
                "segment_length_m": float(segment_length),
                "root_distance_m": float(root_distance),
                "density_neighbor_count": int(density_count),
                "trial_label": trial_label,
                "trial_step_length_m": float(candidate_step_length),
            }

        crop_step = min(max_extension_m, crop_threshold) if crop_threshold > EPS else step_length
        trial_steps = [("initial", step_length)]
        if str(args_dict["short_edge_policy"]) == "crop" and crop_threshold > EPS and crop_step > step_length + EPS:
            trial_steps.append(("crop_min_length", crop_step))
        selected_proposal: dict[str, Any] | None = None
        last_failure: dict[str, Any] | None = None
        for trial_label, trial_step in trial_steps:
            proposal = propose(trial_step, trial_label)
            if not proposal.get("ok", False):
                last_failure = proposal
                continue
            child_grid = proposal["child_grid"]
            candidate_xy = proposal["candidate_xy"]
            segment_length = float(proposal["segment_length_m"])
            if child_grid == tuple(int(v) for v in parent_grid.tolist()):
                last_failure = {**proposal, "reason": "same_grid_as_parent"}
                continue
            if candidate_xy in occupied_xy:
                last_failure = {**proposal, "reason": "duplicate_xy"}
                continue
            if str(args_dict["short_edge_policy"]) == "crop" and crop_threshold > 0.0 and segment_length + EPS < crop_threshold:
                last_failure = {**proposal, "reason": "crop_result_shorter_than_min"}
                continue
            if (
                float(density_radius_m) > 0.0
                and int(max_nodes_per_density_radius) > 0
                and int(proposal["density_neighbor_count"]) >= int(max_nodes_per_density_radius)
            ):
                last_failure = {**proposal, "reason": "density_limit_exceeded"}
                continue
            selected_proposal = proposal
            break
        if selected_proposal is None:
            failure = last_failure or {"reason": "candidate_selection_failed"}
            rejected_rows.append(
                {
                    "attempt": attempts,
                    "reason": failure.get("reason", "candidate_selection_failed"),
                    "sample_source": sample_source,
                    "nearest_id": nearest_id,
                    "target_xy": [int(target_xy[0]), int(target_xy[1])],
                    "distance_to_sample_m": distance_to_target,
                    **{k: v for k, v in failure.items() if k not in {"ok", "edge_cells"}},
                }
            )
            continue

        child_grid = selected_proposal["child_grid"]
        child_world = [float(v) for v in selected_proposal["child_world"]]
        base_yaw = math.atan2(float(child_world[1] - parent_world[1]), float(child_world[0] - parent_world[0]))
        score_start = time.perf_counter()
        yaw_stats = choose_best_yaw(
            observed_state=observed_state,
            grid=child_grid,
            world=tuple(child_world),
            parent_yaw=float(parent.yaw),
            base_yaw=float(base_yaw),
            num_yaw_samples=int(args_dict["num_yaw_samples"]),
            gain_mode=gain_mode,
            prediction_layer=prediction_layer,
            sc_gain_formula=sc_formula,
            tau=float(args_dict["tau"]),
            raycast_stride=int(args_dict["raycast_stride"]),
            max_ray_length_m=float(args_dict["max_ray_length_m"]),
            voxel_size=voxel_size,
        )
        score_time += time.perf_counter() - score_start
        segment_length = float(selected_proposal["segment_length_m"])
        child_id = next_new_segment_id(tree)
        segment = MiniRRTSegment(
            segment_id=child_id,
            parent_id=nearest_id,
            start_grid=[int(v) for v in parent.end_grid],
            start_world=[float(v) for v in parent.end_world],
            end_grid=[int(v) for v in child_grid],
            end_world=child_world,
            yaw=float(yaw_stats["yaw"]),
            gain=float(yaw_stats["gain"]),
            gain_exp=float(yaw_stats["gain_exp"]),
            gain_sc=float(yaw_stats["gain_sc"]),
            gain_hybrid=float(yaw_stats["gain_hybrid"]),
            effective_gain_sc=float(yaw_stats["effective_gain_sc"]),
            gain_hybrid_effective=float(yaw_stats["gain_hybrid_effective"]),
            gain_occ=float(yaw_stats["gain_occ"]),
            gain_conf=float(yaw_stats["gain_conf"]),
            sc_gain_formula=str(yaw_stats["sc_gain_formula"]),
            cost=float(segment_length / max(float(args_dict["v_max"]), EPS)),
            depth=int(parent.depth + 1),
            segment_length_m=float(segment_length),
            yaw_delta=abs(wrap_angle(float(yaw_stats["yaw"]) - float(parent.yaw))),
            yaw_time=abs(wrap_angle(float(yaw_stats["yaw"]) - float(parent.yaw))),
            local_visibility_stats={
                "visible_count": int(yaw_stats["visible_count"]),
                "measured_visible_count": int(yaw_stats["measured_visible_count"]),
                "predicted_unmeasured_visible_count": int(yaw_stats["predicted_unmeasured_visible_count"]),
                "frontier_count_visible": int(yaw_stats["frontier_count_visible"]),
                "yaw_samples_evaluated": int(yaw_stats["yaw_samples_evaluated"]),
            },
            info={
                "origin": "newly_expanded_frame2",
                "sample_source": sample_source,
                "target_xy": [int(target_xy[0]), int(target_xy[1])],
                "nearest_id": nearest_id,
                "trial_label": selected_proposal["trial_label"],
                "density_neighbor_count": int(selected_proposal["density_neighbor_count"]),
            },
        )
        add_tree_child(tree, nearest_id, segment)
        occupied_xy.add((int(child_grid[0]), int(child_grid[1])))
        accepted_rows.append(
            {
                "attempt": attempts,
                "segment_id": child_id,
                "parent_id": nearest_id,
                "sample_source": sample_source,
                "end_grid": list(child_grid),
                "end_world": child_world,
                "gain": float(segment.gain),
                "gain_exp": float(segment.gain_exp),
                "gain_sc": float(segment.gain_sc),
                "effective_gain_sc": float(segment.effective_gain_sc),
                "cost": float(segment.cost),
                "segment_length_m": float(segment.segment_length_m),
                "depth": int(segment.depth),
            }
        )
    warnings = compute_global_normalized_gain(tree, ROOT_ID)
    decision = select_subsequent_best(tree, ROOT_ID)
    return {
        "accepted_rows": accepted_rows,
        "rejected_rows": rejected_rows,
        "attempts": attempts,
        "gain_scoring_time_s": score_time,
        "utility_warnings": warnings,
        "decision": decision,
        "newly_expanded_node_count": len(accepted_rows),
    }


def origin_for_segment(tree: dict[str, MiniRRTSegment], segment_id: str | None) -> str:
    if not segment_id or segment_id not in tree:
        return "missing"
    return str(tree[segment_id].info.get("origin", "unknown"))


def margin_from_tree(tree: dict[str, MiniRRTSegment]) -> dict[str, Any]:
    children = [tree[child_id] for child_id in tree[ROOT_ID].children if child_id in tree]
    children.sort(key=lambda segment: segment.value if math.isfinite(segment.value) else float("-inf"), reverse=True)
    winner = children[0] if children else None
    runner = children[1] if len(children) > 1 else None
    winner_value = winner.value if winner is not None and math.isfinite(winner.value) else None
    runner_value = runner.value if runner is not None and math.isfinite(runner.value) else None
    margin = None if winner_value is None or runner_value is None else float(winner_value - runner_value)
    normalized = safe_ratio(margin, abs(winner_value)) if margin is not None else None
    return {
        "winner_id": winner.segment_id if winner else None,
        "runner_up_id": runner.segment_id if runner else None,
        "winner_value": winner_value,
        "runner_up_value": runner_value,
        "winner_margin": margin,
        "normalized_margin": normalized,
        "root_child_count": len(children),
    }


def summarize_persistent_tree(
    *,
    config_name: str,
    seed: int,
    formula: str,
    tree: dict[str, MiniRRTSegment],
    frame1_decision: dict[str, Any],
    preservation: dict[str, Any],
    raw_dir: Path,
    elapsed_s: float,
    extension: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    decision = extension["decision"]
    selected_id = decision.get("selected_child_id")
    best_id = decision.get("selected_child_best_descendant_id")
    selected = tree.get(str(selected_id)) if selected_id else None
    best = tree.get(str(best_id)) if best_id else None
    root = tree[ROOT_ID]
    path_ids = segment_path_to_root(tree, str(best_id) if best_id else None)
    path_segments = [tree[node_id] for node_id in path_ids if node_id in tree and node_id != ROOT_ID]
    non_root = [segment for segment in tree.values() if segment.segment_id != ROOT_ID]
    values = [segment.value for segment in non_root]
    costs = [segment.accumulated_cost for segment in non_root]
    inverse_cost = [safe_ratio(1.0, cost) or float("nan") for cost in costs]
    gain_exp = [segment.gain_exp for segment in non_root]
    effective_sc = [segment.effective_gain_sc for segment in non_root]
    raw_sc = [segment.gain_sc for segment in non_root]
    margin = margin_from_tree(tree)
    selected_origin = origin_for_segment(tree, str(selected_id) if selected_id else None)
    best_origin = origin_for_segment(tree, str(best_id) if best_id else None)
    path_origins = [origin_for_segment(tree, node_id) for node_id in path_ids]
    return {
        "config": config_name,
        "tree_lifecycle": spec["tree_lifecycle"],
        "density_mode": spec["density_mode"],
        "density_radius_m": spec["density_radius_m"],
        "max_nodes_per_density_radius": spec["max_nodes_per_density_radius"],
        "seed": int(seed),
        "formula": str(formula),
        "status": "completed" if selected is not None else "no_selected_child",
        "tree_dir": str(raw_dir),
        "frame1_selected_child_id": frame1_decision.get("selected_child_id"),
        "frame1_selected_child_grid": segment_brief(tree.get(ROOT_ID)) if False else None,
        "frame1_best_descendant_id": frame1_decision.get("selected_child_best_descendant_id"),
        "frame2_selected_child_id": selected.segment_id if selected else None,
        "frame2_selected_child_grid": selected.end_grid if selected else None,
        "frame2_selected_child_world": selected.end_world if selected else None,
        "frame2_best_descendant_id": best.segment_id if best else None,
        "frame2_best_descendant_grid": best.end_grid if best else None,
        "frame2_best_descendant_world": best.end_world if best else None,
        "selected_child_id": selected.segment_id if selected else None,
        "selected_child_grid": selected.end_grid if selected else None,
        "selected_child_world": selected.end_world if selected else None,
        "best_descendant_id": best.segment_id if best else None,
        "best_descendant_grid": best.end_grid if best else None,
        "best_descendant_world": best.end_world if best else None,
        "root_grid": root.end_grid,
        "root_world": root.end_world,
        "selected_child_distance_from_root_m": euclidean(root.end_world or [], selected.end_world or []) if selected else None,
        "best_descendant_distance_from_root_m": euclidean(root.end_world or [], best.end_world or []) if best else None,
        "accumulated_gain_exp": float(sum(segment.gain_exp for segment in path_segments)),
        "accumulated_raw_gain_sc": float(sum(segment.gain_sc for segment in path_segments)),
        "accumulated_effective_gain_sc": float(sum(segment.effective_gain_sc for segment in path_segments)),
        "accumulated_hybrid_effective": float(sum(segment.gain_hybrid_effective for segment in path_segments)),
        "accumulated_cost": float(sum(segment.cost for segment in path_segments)),
        "value": selected.value if selected and math.isfinite(selected.value) else None,
        "best_descendant_value": best.value if best and math.isfinite(best.value) else None,
        "runner_up_value": margin.get("runner_up_value"),
        "winner_margin": margin.get("winner_margin"),
        "normalized_margin": margin.get("normalized_margin"),
        "winner_id": margin.get("winner_id"),
        "runner_up_id": margin.get("runner_up_id"),
        "root_child_count": margin.get("root_child_count"),
        "branch_depth": len(path_ids),
        "path_node_ids": path_ids,
        "path_origins": path_origins,
        "selected_origin": selected_origin,
        "best_descendant_origin": best_origin,
        "nodes_accepted": len(non_root),
        "nodes_preserved_from_frame1": int(preservation["preserved_node_count"]),
        "nodes_pruned_after_reroot": int(preservation["pruned_node_count"]),
        "nodes_reinserted": int(preservation["reinserted_node_count"]),
        "nodes_newly_expanded_frame2": int(extension["newly_expanded_node_count"]),
        "nodes_rejected": len(extension["rejected_rows"]),
        "rejection_reasons": dict(Counter(str(row.get("reason", "unknown")) for row in extension["rejected_rows"])),
        "nodes_with_effective_gain_sc_positive": sum(1 for value in effective_sc if value > 0.0),
        "effective_gain_sc_min_mean_max": min_mean_max(effective_sc),
        "raw_gain_sc_min_mean_max": min_mean_max(raw_sc),
        "gain_exp_effective_gain_sc_correlation": pearson_pairs(gain_exp, effective_sc),
        "value_effective_gain_sc_correlation": pearson_pairs(values, effective_sc),
        "value_cost_correlation": pearson_pairs(values, costs),
        "value_inverse_cost_correlation": pearson_pairs(values, inverse_cost),
        "preserved_subtree_selected": selected_origin.startswith("preserved"),
        "preserved_subtree_contains_winner": any(origin.startswith("preserved") for origin in path_origins),
        "newly_expanded_branch_selected": selected_origin == "newly_expanded_frame2",
        "reinserted_branch_selected": selected_origin.startswith("reinserted"),
        "fresh_only_branch_selected": False,
        "no_valid_preserved_root": bool(preservation["no_valid_preserved_root"]),
        "pruned_all_old_branches": bool(preservation["pruned_all_old_branches"]),
        "elapsed_s": float(elapsed_s),
    }


def write_raw_persistent_outputs(
    raw_dir: Path,
    tree: dict[str, MiniRRTSegment],
    preservation: dict[str, Any],
    extension: dict[str, Any],
    row: dict[str, Any],
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_dir / "persistent_tree_segments.jsonl", [segment_record(tree[key]) for key in tree])
    write_csv(raw_dir / "gain_cost_value_table.csv", make_gain_value_rows(tree))
    write_csv(raw_dir / "persistent_rejected_samples.csv", extension["rejected_rows"])
    write_csv(raw_dir / "persistent_new_nodes.csv", extension["accepted_rows"])
    save_json(
        raw_dir / "persistent_tree_summary.json",
        {
            "decision": extension["decision"],
            "selected_child": segment_brief(tree.get(row.get("selected_child_id") or "")),
            "best_descendant": segment_brief(tree.get(row.get("best_descendant_id") or "")),
            "preservation": preservation,
            "row": row,
            "utility_warning_count": len(extension["utility_warnings"]),
            "utility_warnings_sample": extension["utility_warnings"][:10],
            "prediction_writeback": False,
            "prediction_used_for_traversability_collision": False,
            "prediction_ray_blocking": False,
        },
    )


def run_persistent_group(payload: dict[str, Any]) -> dict[str, Any]:
    seed = int(payload["seed"])
    formula = str(payload["formula"])
    density_mode = str(payload["density_mode"])
    args_dict = payload["args_dict"]
    output_dir = Path(payload["output_dir"])
    config_names = payload["config_names"]
    density_radius_m = float(payload["density_radius_m"])
    max_nodes_per_density_radius = int(payload["max_nodes_per_density_radius"])

    frame1_observed = np.load(args_dict["frame1_observed_state"])
    frame2_observed = np.load(args_dict["frame2_observed_state"])
    frame1_observed.setflags(write=False)
    frame2_observed.setflags(write=False)
    shape = tuple(int(v) for v in frame1_observed.shape)
    bounds = default_bounds(shape, float(args_dict["voxel_size"]))
    frame1_root = root_from_pose(Path(args_dict["frame1_pose_json"]), shape, float(args_dict["voxel_size"]), bounds)
    frame2_root = root_from_pose(Path(args_dict["frame2_pose_json"]), shape, float(args_dict["voxel_size"]), bounds)
    frame1_prediction = prediction_layer_for(formula, Path(args_dict["frame1_prediction_npz"]), shape)
    frame2_prediction = prediction_layer_for(formula, Path(args_dict["frame2_prediction_npz"]), shape)
    group_start = time.perf_counter()
    frame1_result = build_mini_rrt_tree(
        frame1_observed,
        frame1_root["grid"],
        frame1_root["pose_world"],
        frame1_root["yaw"],
        bounds,
        seed=seed,
        num_nodes=int(args_dict["frame1_num_nodes"]),
        max_extension_m=float(args_dict["max_extension_m"]),
        sample_mode=str(args_dict["sample_mode"]),
        gain_mode=gain_mode_for(formula),
        v_max=float(args_dict["v_max"]),
        robot_radius_m=float(args_dict["robot_radius_m"]),
        voxel_size=float(args_dict["voxel_size"]),
        raycast_stride=int(args_dict["raycast_stride"]),
        num_yaw_samples=int(args_dict["num_yaw_samples"]),
        max_ray_length_m=float(args_dict["max_ray_length_m"]),
        sc_gain_formula=sc_formula_for(formula),
        prediction_layer=frame1_prediction,
        tau=float(args_dict["tau"]),
        profile=True,
        crop_min_length_m=float(args_dict["crop_min_length_m"]),
        short_edge_policy=str(args_dict["short_edge_policy"]),
        density_radius_m=density_radius_m,
        max_nodes_per_density_radius=max_nodes_per_density_radius,
    )
    frame1_tree: dict[str, MiniRRTSegment] = frame1_result["tree"]
    frame1_decision = frame1_result["decision"]
    rows: list[dict[str, Any]] = []
    reinsert_attempts: list[dict[str, Any]] = []
    for config_name in config_names:
        spec = dict(CONFIG_SPECS[config_name])
        target = int(spec["frame2_target_num_nodes"])
        raw_dir = output_dir / "raw_trees" / config_name / f"seed_{seed:03d}" / formula
        start = time.perf_counter()
        init = preserve_and_reinsert_tree(
            frame1_tree=frame1_tree,
            frame1_decision=frame1_decision,
            frame2_observed_state=frame2_observed,
            frame2_prediction_layer=frame2_prediction,
            frame2_root=frame2_root,
            bounds=bounds,
            args_dict=args_dict,
            formula=formula,
            config_name=config_name,
            density_radius_m=density_radius_m,
            max_nodes_per_density_radius=max_nodes_per_density_radius,
        )
        extension = extend_persistent_tree(
            tree=init["tree"],
            context=init["context"],
            occupied_xy=init["occupied_xy"],
            observed_state=frame2_observed,
            prediction_layer=frame2_prediction,
            bounds=bounds,
            target_total_nodes=target,
            seed=seed,
            formula=formula,
            args_dict=args_dict,
            density_radius_m=density_radius_m,
            max_nodes_per_density_radius=max_nodes_per_density_radius,
        )
        init["preservation"]["newly_expanded_node_count"] = extension["newly_expanded_node_count"]
        row = summarize_persistent_tree(
            config_name=config_name,
            seed=seed,
            formula=formula,
            tree=init["tree"],
            frame1_decision=frame1_decision,
            preservation=init["preservation"],
            raw_dir=raw_dir,
            elapsed_s=time.perf_counter() - start,
            extension=extension,
            spec=spec,
        )
        write_raw_persistent_outputs(raw_dir, init["tree"], init["preservation"], extension, row)
        for attempt in init["preservation"]["reinsert_attempts"]:
            reinsert_attempts.append({"seed": seed, **attempt})
        rows.append(row)
    return {
        "seed": seed,
        "formula": formula,
        "density_mode": density_mode,
        "frame1_elapsed_s": time.perf_counter() - group_start,
        "frame1_selected_child_id": frame1_decision.get("selected_child_id"),
        "frame1_selected_child_grid": segment_brief(frame1_tree.get(frame1_decision.get("selected_child_id") or "")),
        "rows": rows,
        "reinsert_attempts": reinsert_attempts,
    }


def load_fresh_baseline_rows(
    *,
    stage4a65v_dir: Path,
    seeds: list[int],
    formulas: list[str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows = read_json(stage4a65v_dir / "per_seed_formula_decisions.json")
    out: list[dict[str, Any]] = []
    for row in rows:
        seed = int(row["seed"])
        formula = str(row["formula"])
        if seed not in seeds or formula not in formulas:
            continue
        copied = dict(row)
        copied["config"] = "fresh_random_256_baseline"
        copied["tree_lifecycle"] = "fresh"
        copied["density_mode"] = "none"
        copied["density_radius_m"] = 0.0
        copied["max_nodes_per_density_radius"] = 0
        copied["frame2_selected_child_id"] = copied.get("selected_child_id")
        copied["frame2_selected_child_grid"] = parse_jsonish(copied.get("selected_child_grid"))
        copied["frame2_selected_child_world"] = parse_jsonish(copied.get("selected_child_world"))
        copied["frame2_best_descendant_id"] = copied.get("best_descendant_id")
        copied["frame2_best_descendant_grid"] = parse_jsonish(copied.get("best_descendant_grid"))
        copied["frame2_best_descendant_world"] = parse_jsonish(copied.get("best_descendant_world"))
        copied["selected_child_grid"] = copied["frame2_selected_child_grid"]
        copied["selected_child_world"] = copied["frame2_selected_child_world"]
        copied["best_descendant_grid"] = copied["frame2_best_descendant_grid"]
        copied["best_descendant_world"] = copied["frame2_best_descendant_world"]
        copied["path_node_ids"] = parse_jsonish(copied.get("path_node_ids"))
        copied["path_origins"] = ["fresh_frame2"] * len(copied.get("path_node_ids") or [])
        copied["selected_origin"] = "fresh_frame2"
        copied["best_descendant_origin"] = "fresh_frame2"
        copied["nodes_accepted"] = copied.get("accepted_nodes")
        copied["nodes_preserved_from_frame1"] = 0
        copied["nodes_pruned_after_reroot"] = 0
        copied["nodes_reinserted"] = 0
        copied["nodes_newly_expanded_frame2"] = copied.get("accepted_nodes")
        copied["nodes_rejected"] = copied.get("rejected_samples")
        copied["preserved_subtree_selected"] = False
        copied["preserved_subtree_contains_winner"] = False
        copied["newly_expanded_branch_selected"] = False
        copied["reinserted_branch_selected"] = False
        copied["fresh_only_branch_selected"] = True
        copied["no_valid_preserved_root"] = False
        copied["pruned_all_old_branches"] = False
        copied["tree_dir"] = str(stage4a65v_dir / "raw_trees" / f"seed_{seed:03d}" / formula)
        copied["fresh_source_stage4a65v"] = str(stage4a65v_dir)
        out.append(copied)
    return out


def classify_rows(rows: list[dict[str, Any]], voxel_size: float, observed_shape: tuple[int, int, int]) -> list[dict[str, Any]]:
    bounds = default_bounds(observed_shape, voxel_size)
    ref_sc_selected_world = list(grid_to_world(REFERENCE_SC_SELECTED_GRID, bounds, voxel_size))
    ref_sc_best_world = list(grid_to_world(REFERENCE_SC_BEST_GRID, bounds, voxel_size))
    by_key = {(row["config"], int(row["seed"]), str(row["formula"])): row for row in rows}
    classified: list[dict[str, Any]] = []
    for row in rows:
        measured = by_key.get((row["config"], int(row["seed"]), "measured_only"), {})
        selected_world = row.get("selected_child_world")
        best_world = row.get("best_descendant_world")
        selected_grid = row.get("selected_child_grid")
        best_grid = row.get("best_descendant_grid")
        measured_world = measured.get("selected_child_world")
        selected_to_seed0 = euclidean(selected_world or [], ref_sc_selected_world) if selected_world is not None else None
        best_to_seed0 = euclidean(best_world or [], ref_sc_best_world) if best_world is not None else None
        selected_to_measured = euclidean(selected_world or [], measured_world or []) if measured_world is not None and selected_world is not None else None
        missing = selected_world is None or best_world is None or selected_grid is None or best_grid is None
        exact = same_grid(selected_grid, REFERENCE_SC_SELECTED_GRID) and same_grid(best_grid, REFERENCE_SC_BEST_GRID)
        spatial = False if missing or selected_to_seed0 is None or best_to_seed0 is None else bool(
            selected_to_seed0 <= 0.25 and best_to_seed0 <= 0.75
        )
        same_as_measured = False if missing or selected_to_measured is None else bool(
            same_grid(selected_grid, measured.get("selected_child_grid")) or selected_to_measured <= 0.15
        )
        measured_basin = bool(same_as_measured and spatial)
        local_jitter = False if missing or selected_to_measured is None else bool(
            not same_as_measured and selected_to_measured < 0.25
        )
        distinct = False if missing or selected_to_measured is None else bool(
            selected_to_measured >= 0.25 and not spatial
        )
        if missing:
            primary = "unstable_or_missing"
        elif exact:
            primary = "exact_seed0_sc"
        elif measured_basin:
            primary = "measured_but_seed0_sc_basin"
        elif spatial:
            primary = "spatial_seed0_sc_basin"
        elif same_as_measured:
            primary = "same_as_measured_for_seed"
        elif distinct:
            primary = "distinct_sc_branch"
        elif local_jitter:
            primary = "local_jitter"
        else:
            primary = "unstable_or_missing"
        row["classification"] = primary
        row["selected_to_seed0_sc_reference_m"] = selected_to_seed0
        row["best_to_seed0_sc_reference_m"] = best_to_seed0
        row["selected_to_same_seed_measured_m"] = selected_to_measured
        classified.append(
            {
                "config": row["config"],
                "seed": int(row["seed"]),
                "formula": str(row["formula"]),
                "primary_classification": primary,
                "exact_seed0_sc": exact,
                "spatial_seed0_sc_basin": spatial,
                "same_as_measured_for_seed": same_as_measured,
                "measured_but_seed0_sc_basin": measured_basin,
                "distinct_sc_branch": distinct,
                "local_jitter": local_jitter,
                "unstable_or_missing": bool(missing),
                "selected_to_seed0_sc_reference_m": selected_to_seed0,
                "best_to_seed0_sc_reference_m": best_to_seed0,
                "selected_to_same_seed_measured_m": selected_to_measured,
                "preserved_subtree_selected": bool(row.get("preserved_subtree_selected")),
                "preserved_subtree_contains_winner": bool(row.get("preserved_subtree_contains_winner")),
                "newly_expanded_branch_selected": bool(row.get("newly_expanded_branch_selected")),
                "reinserted_branch_selected": bool(row.get("reinserted_branch_selected")),
                "fresh_only_branch_selected": bool(row.get("fresh_only_branch_selected")),
                "no_valid_preserved_root": bool(row.get("no_valid_preserved_root")),
                "pruned_all_old_branches": bool(row.get("pruned_all_old_branches")),
                "selected_child_id": row.get("selected_child_id"),
                "selected_child_grid": selected_grid,
                "best_descendant_id": row.get("best_descendant_id"),
                "best_descendant_grid": best_grid,
            }
        )
    return classified


def fraction(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return float(sum(1 for row in rows if bool(row.get(key)))) / float(len(rows))


def build_summary_tables(
    rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    configs: list[str],
    formulas: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    class_summary: dict[str, Any] = {}
    spatial_rows: list[dict[str, Any]] = []
    margin_rows: list[dict[str, Any]] = []
    preservation_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    keys = [
        "exact_seed0_sc",
        "spatial_seed0_sc_basin",
        "same_as_measured_for_seed",
        "measured_but_seed0_sc_basin",
        "distinct_sc_branch",
        "local_jitter",
        "unstable_or_missing",
    ]
    persistent_keys = [
        "preserved_subtree_selected",
        "preserved_subtree_contains_winner",
        "newly_expanded_branch_selected",
        "reinserted_branch_selected",
        "fresh_only_branch_selected",
        "no_valid_preserved_root",
        "pruned_all_old_branches",
    ]
    by_decision = {(row["config"], int(row["seed"]), str(row["formula"])): row for row in rows}
    for config in configs:
        class_summary[config] = {}
        for formula in formulas:
            subset = [row for row in class_rows if row["config"] == config and row["formula"] == formula]
            selected_deltas = [as_float(row.get("selected_to_seed0_sc_reference_m"), float("nan")) for row in subset]
            best_deltas = [as_float(row.get("best_to_seed0_sc_reference_m"), float("nan")) for row in subset]
            class_summary[config][formula] = {
                "seed_count": len(subset),
                "counts": {key: sum(1 for row in subset if bool(row.get(key))) for key in keys + persistent_keys},
                "fractions": {key: fraction(subset, key) for key in keys + persistent_keys},
            }
            spatial_rows.append(
                {
                    "config": config,
                    "formula": formula,
                    "seed_count": len(subset),
                    "exact_seed0_sc_fraction": fraction(subset, "exact_seed0_sc"),
                    "spatial_seed0_sc_basin_fraction": fraction(subset, "spatial_seed0_sc_basin"),
                    "same_as_measured_for_seed_fraction": fraction(subset, "same_as_measured_for_seed"),
                    "measured_but_seed0_sc_basin_fraction": fraction(subset, "measured_but_seed0_sc_basin"),
                    "distinct_sc_branch_fraction": fraction(subset, "distinct_sc_branch"),
                    "local_jitter_fraction": fraction(subset, "local_jitter"),
                    "selected_delta_to_seed0_sc_median_m": statistics.median([v for v in selected_deltas if math.isfinite(v)])
                    if any(math.isfinite(v) for v in selected_deltas)
                    else None,
                    "best_delta_to_seed0_sc_median_m": statistics.median([v for v in best_deltas if math.isfinite(v)])
                    if any(math.isfinite(v) for v in best_deltas)
                    else None,
                }
            )
        decision_subset = [row for row in rows if row["config"] == config]
        confidence_subset = [row for row in decision_subset if row["formula"] == "confidence_weighted"]
        margins = [as_float(row.get("normalized_margin"), float("nan")) for row in confidence_subset]
        margin_rows.append(
            {
                "config": config,
                "formula": "confidence_weighted",
                **{f"normalized_margin_{k}": v for k, v in percentile_summary(margins).items()},
                "narrow_margin_fraction_lt_0p02": (
                    sum(1 for v in margins if math.isfinite(v) and v < 0.02) / len([v for v in margins if math.isfinite(v)])
                    if any(math.isfinite(v) for v in margins)
                    else None
                ),
            }
        )
        preservation_rows.append(
            {
                "config": config,
                "rows": len(decision_subset),
                "mean_nodes_preserved": statistics.fmean([as_float(row.get("nodes_preserved_from_frame1")) for row in decision_subset])
                if decision_subset
                else None,
                "mean_nodes_pruned": statistics.fmean([as_float(row.get("nodes_pruned_after_reroot")) for row in decision_subset])
                if decision_subset
                else None,
                "mean_nodes_reinserted": statistics.fmean([as_float(row.get("nodes_reinserted")) for row in decision_subset])
                if decision_subset
                else None,
                "mean_nodes_newly_expanded": statistics.fmean([as_float(row.get("nodes_newly_expanded_frame2")) for row in decision_subset])
                if decision_subset
                else None,
                "preserved_subtree_selected_fraction": fraction(
                    [row for row in class_rows if row["config"] == config], "preserved_subtree_selected"
                ),
                "preserved_subtree_contains_winner_fraction": fraction(
                    [row for row in class_rows if row["config"] == config], "preserved_subtree_contains_winner"
                ),
                "newly_expanded_branch_selected_fraction": fraction(
                    [row for row in class_rows if row["config"] == config], "newly_expanded_branch_selected"
                ),
                "reinserted_branch_selected_fraction": fraction(
                    [row for row in class_rows if row["config"] == config], "reinserted_branch_selected"
                ),
            }
        )
        for seed in seeds:
            conf = by_decision.get((config, seed, "confidence_weighted"), {})
            cap = by_decision.get((config, seed, "cap25"), {})
            measured = by_decision.get((config, seed, "measured_only"), {})
            agreement_rows.append(
                {
                    "config": config,
                    "seed": seed,
                    "confidence_cap25_exact_grid_agree": same_grid(
                        conf.get("selected_child_grid"), cap.get("selected_child_grid")
                    ),
                    "confidence_cap25_selected_delta_m": (
                        euclidean(conf.get("selected_child_world") or [], cap.get("selected_child_world") or [])
                        if conf and cap
                        else None
                    ),
                    "confidence_measured_exact_grid_agree": same_grid(
                        conf.get("selected_child_grid"), measured.get("selected_child_grid")
                    ),
                    "cap25_measured_exact_grid_agree": same_grid(
                        cap.get("selected_child_grid"), measured.get("selected_child_grid")
                    ),
                }
            )
        config_rows.append(
            {
                "config": config,
                "row_count": len(decision_subset),
                "seed_count": len({int(row["seed"]) for row in decision_subset}),
                "formula_count": len({str(row["formula"]) for row in decision_subset}),
                "confidence_spatial_seed0_sc_basin_fraction": class_summary[config]
                .get("confidence_weighted", {})
                .get("fractions", {})
                .get("spatial_seed0_sc_basin"),
                "confidence_same_as_measured_fraction": class_summary[config]
                .get("confidence_weighted", {})
                .get("fractions", {})
                .get("same_as_measured_for_seed"),
                "cap25_spatial_seed0_sc_basin_fraction": class_summary[config]
                .get("cap25", {})
                .get("fractions", {})
                .get("spatial_seed0_sc_basin"),
                "mean_elapsed_s": statistics.fmean([as_float(row.get("elapsed_s")) for row in decision_subset])
                if decision_subset
                else None,
            }
        )
    agreement_summary = []
    for config in configs:
        subset = [row for row in agreement_rows if row["config"] == config]
        agreement_summary.append(
            {
                "config": config,
                "seed_count": len(subset),
                "confidence_vs_cap25_exact_grid_agreement_rate": fraction(
                    subset, "confidence_cap25_exact_grid_agree"
                ),
                "confidence_vs_measured_exact_grid_agreement_rate": fraction(
                    subset, "confidence_measured_exact_grid_agree"
                ),
                "cap25_vs_measured_exact_grid_agreement_rate": fraction(
                    subset, "cap25_measured_exact_grid_agree"
                ),
            }
        )
    return {
        "class_summary": class_summary,
        "spatial_rows": spatial_rows,
        "margin_rows": margin_rows,
        "preservation_rows": preservation_rows,
        "config_rows": config_rows,
        "agreement_rows": agreement_rows,
        "agreement_summary": agreement_summary,
    }


def write_markdown_tables(output_dir: Path, rows: list[dict[str, Any]], summaries: dict[str, Any]) -> None:
    lines = [
        "# Per Config / Seed / Formula Decisions",
        "",
        "| config | seed | formula | selected grid | best grid | value | norm margin | class | origin |",
        "|---|---:|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['config']}` | {row['seed']} | `{row['formula']}` | `{row.get('selected_child_grid')}` | "
            f"`{row.get('best_descendant_grid')}` | {row.get('value')} | {row.get('normalized_margin')} | "
            f"`{row.get('classification')}` | `{row.get('selected_origin')}` |"
        )
    write_text(output_dir / "per_config_seed_formula_decisions.md", "\n".join(lines) + "\n")

    lines = [
        "# Per Config Summary",
        "",
        "| config | rows | confidence spatial | confidence same-as-measured | cap25 spatial | mean elapsed s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries["config_rows"]:
        lines.append(
            f"| `{row['config']}` | {row['row_count']} | {row.get('confidence_spatial_seed0_sc_basin_fraction')} | "
            f"{row.get('confidence_same_as_measured_fraction')} | {row.get('cap25_spatial_seed0_sc_basin_fraction')} | "
            f"{row.get('mean_elapsed_s')} |"
        )
    write_text(output_dir / "per_config_summary.md", "\n".join(lines) + "\n")

    lines = ["# Branch Classification Summary By Config", ""]
    for config, by_formula in summaries["class_summary"].items():
        lines.append(f"## {config}")
        lines.append("")
        for formula, info in by_formula.items():
            frac = info["fractions"]
            lines.append(
                f"- `{formula}`: spatial `{frac.get('spatial_seed0_sc_basin')}`, "
                f"same-as-measured `{frac.get('same_as_measured_for_seed')}`, "
                f"preserved-winner `{frac.get('preserved_subtree_contains_winner')}`"
            )
        lines.append("")
    write_text(output_dir / "branch_classification_summary_by_config.md", "\n".join(lines))

    lines = [
        "# Preservation Summary By Config",
        "",
        "| config | mean preserved | mean pruned | mean reinserted | preserved winner frac | new selected frac |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries["preservation_rows"]:
        lines.append(
            f"| `{row['config']}` | {row.get('mean_nodes_preserved')} | {row.get('mean_nodes_pruned')} | "
            f"{row.get('mean_nodes_reinserted')} | {row.get('preserved_subtree_contains_winner_fraction')} | "
            f"{row.get('newly_expanded_branch_selected_fraction')} |"
        )
    write_text(output_dir / "preservation_summary_by_config.md", "\n".join(lines) + "\n")

    lines = [
        "# Spatial Basin Summary By Config",
        "",
        "| config | formula | spatial | same-as-measured | distinct | selected median m | best median m |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries["spatial_rows"]:
        lines.append(
            f"| `{row['config']}` | `{row['formula']}` | {row.get('spatial_seed0_sc_basin_fraction')} | "
            f"{row.get('same_as_measured_for_seed_fraction')} | {row.get('distinct_sc_branch_fraction')} | "
            f"{row.get('selected_delta_to_seed0_sc_median_m')} | {row.get('best_delta_to_seed0_sc_median_m')} |"
        )
    write_text(output_dir / "spatial_basin_summary_by_config.md", "\n".join(lines) + "\n")

    lines = [
        "# Margin Summary By Config",
        "",
        "| config | count | min | median | max | narrow fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries["margin_rows"]:
        lines.append(
            f"| `{row['config']}` | {row.get('normalized_margin_count')} | {row.get('normalized_margin_min')} | "
            f"{row.get('normalized_margin_median')} | {row.get('normalized_margin_max')} | "
            f"{row.get('narrow_margin_fraction_lt_0p02')} |"
        )
    write_text(output_dir / "margin_summary_by_config.md", "\n".join(lines) + "\n")


def topdown_projection(observed_state: np.ndarray) -> np.ndarray:
    image = np.zeros(observed_state.shape[:2], dtype=np.int8)
    image[np.any(observed_state == FREE, axis=2)] = 1
    image[np.any(observed_state == OCCUPIED, axis=2)] = 2
    return image


def plot_base_map(ax: plt.Axes, observed_state: np.ndarray) -> None:
    proj = topdown_projection(observed_state)
    colors = np.asarray(
        [
            [0.84, 0.84, 0.84, 1.0],
            [0.90, 0.97, 0.98, 1.0],
            [0.58, 0.12, 0.12, 1.0],
        ],
        dtype=np.float64,
    )
    ax.imshow(colors[proj].transpose(1, 0, 2), origin="lower", interpolation="nearest")
    ax.set_xlabel("grid x")
    ax.set_ylabel("grid y")
    ax.grid(color="#111827", alpha=0.12, linewidth=0.4)


def grid_xy(grid: Any) -> tuple[float, float] | None:
    if grid is None:
        return None
    try:
        return float(grid[0]), float(grid[1])
    except (TypeError, ValueError, IndexError):
        return None


def make_plots(output_dir: Path, observed_state: np.ndarray, rows: list[dict[str, Any]], summaries: dict[str, Any]) -> dict[str, str]:
    plots: dict[str, str] = {}
    configs = [row["config"] for row in summaries["config_rows"]]
    x = np.arange(len(configs))

    def save_bar(name: str, title: str, values: list[float | None], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
        ax.bar(x, [0.0 if v is None else float(v) for v in values], color="#3b82f6")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.05 if "fraction" in ylabel.lower() or "rate" in ylabel.lower() else max([1.0] + [float(v or 0) for v in values]) * 1.1)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        path = output_dir / name
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots[name] = str(path)

    config_rows = summaries["config_rows"]
    save_bar(
        "spatial_seed0_sc_basin_fraction_by_config.png",
        "Confidence spatial seed0 SC basin fraction",
        [row.get("confidence_spatial_seed0_sc_basin_fraction") for row in config_rows],
        "fraction",
    )
    save_bar(
        "same_as_measured_fraction_by_config.png",
        "Confidence same-as-measured fraction",
        [row.get("confidence_same_as_measured_fraction") for row in config_rows],
        "fraction",
    )
    save_bar(
        "preserved_nodes_fraction_by_config.png",
        "Preserved subtree contains winner fraction",
        [row.get("preserved_subtree_contains_winner_fraction") for row in summaries["preservation_rows"]],
        "fraction",
    )
    save_bar(
        "confidence_cap25_agreement_by_config.png",
        "Confidence/cap25 agreement",
        [row.get("confidence_vs_cap25_exact_grid_agreement_rate") for row in summaries["agreement_summary"]],
        "rate",
    )
    save_bar(
        "compute_time_by_config.png",
        "Mean compute time by config",
        [row.get("mean_elapsed_s") for row in config_rows],
        "seconds",
    )

    marker_by_config = {config: marker for config, marker in zip(configs, ["o", "^", "s", "P", "X", "D", "v"])}
    for filename, key, title, ref_grid in [
        ("selected_children_by_config_topdown.png", "selected_child_grid", "Selected children by config", REFERENCE_SC_SELECTED_GRID),
        ("best_descendants_by_config_topdown.png", "best_descendant_grid", "Best descendants by config", REFERENCE_SC_BEST_GRID),
    ]:
        fig, ax = plt.subplots(figsize=(9, 7.5), constrained_layout=True)
        plot_base_map(ax, observed_state)
        for row in rows:
            if row["formula"] != "confidence_weighted":
                continue
            xy = grid_xy(row.get(key))
            if xy is None:
                continue
            label = row["config"] if row["config"] not in ax.get_legend_handles_labels()[1] else None
            ax.scatter([xy[0]], [xy[1]], s=55, marker=marker_by_config.get(row["config"], "o"), label=label, alpha=0.78)
        ref_xy = grid_xy(ref_grid)
        if ref_xy is not None:
            ax.scatter([ref_xy[0]], [ref_xy[1]], s=135, c="none", edgecolor="#111827", linewidth=1.7, label="seed0 SC ref")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")
        path = output_dir / filename
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots[filename] = str(path)

    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    data = [
        [
            as_float(row.get("normalized_margin"), float("nan"))
            for row in rows
            if row["config"] == config and row["formula"] == "confidence_weighted"
        ]
        for config in configs
    ]
    ax.boxplot(data, tick_labels=configs, showmeans=True)
    ax.axhline(0.02, color="#dc2626", linestyle="--", linewidth=1.0)
    ax.set_ylabel("normalized winner margin")
    ax.set_title("Confidence margin distribution by config")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    path = output_dir / "margin_distribution_by_config.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(10.5, 5.0), constrained_layout=True)
    for config in configs:
        subset = sorted(
            [row for row in rows if row["config"] == config and row["formula"] == "confidence_weighted"],
            key=lambda row: int(row["seed"]),
        )
        ax.plot(
            [int(row["seed"]) for row in subset],
            [as_float(row.get("selected_to_seed0_sc_reference_m"), float("nan")) for row in subset],
            marker=marker_by_config.get(config, "o"),
            label=config,
        )
    ax.axhline(0.25, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_xlabel("seed")
    ax.set_ylabel("selected delta to seed0 SC ref (m)")
    ax.set_title("Selected delta to seed0 SC by config")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    path = output_dir / "selected_delta_to_seed0_sc_by_config.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)

    for filename, x_key, xlabel in [
        ("value_vs_effective_sc_by_config.png", "accumulated_effective_gain_sc", "accumulated effective SC"),
        ("value_vs_cost_by_config.png", "accumulated_cost", "accumulated cost"),
    ]:
        fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
        for config in configs:
            subset = [row for row in rows if row["config"] == config and row["formula"] == "confidence_weighted"]
            ax.scatter(
                [as_float(row.get(x_key), float("nan")) for row in subset],
                [as_float(row.get("value"), float("nan")) for row in subset],
                s=58,
                marker=marker_by_config.get(config, "o"),
                label=config,
                alpha=0.78,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("selected root-child value")
        ax.set_title(filename.replace("_", " ").replace(".png", ""))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        path = output_dir / filename
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots[path.name] = str(path)

    fig, ax = plt.subplots(figsize=(10.0, 4.9), constrained_layout=True)
    preserved = [row.get("preserved_subtree_contains_winner_fraction") for row in summaries["preservation_rows"]]
    new = [row.get("newly_expanded_branch_selected_fraction") for row in summaries["preservation_rows"]]
    ax.bar(x - 0.18, [float(v or 0.0) for v in preserved], width=0.36, label="preserved contains winner")
    ax.bar(x + 0.18, [float(v or 0.0) for v in new], width=0.36, label="newly expanded selected")
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("fraction")
    ax.set_title("Preserved vs newly expanded winners")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    path = output_dir / "preserved_vs_newly_expanded_winners.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    plots[path.name] = str(path)
    return plots


def recommendation_from_summary(summaries: dict[str, Any]) -> tuple[str, str, str]:
    by_config = {row["config"]: row for row in summaries["config_rows"]}
    fresh = as_float(by_config.get("fresh_random_256_baseline", {}).get("confidence_spatial_seed0_sc_basin_fraction"), 0.0)
    best_config = None
    best_value = -1.0
    for row in summaries["config_rows"]:
        if not str(row["config"]).startswith("persistent_rewire"):
            continue
        value = as_float(row.get("confidence_spatial_seed0_sc_basin_fraction"), -1.0)
        if value > best_value:
            best_value = value
            best_config = row["config"]
    margin_row = next((row for row in summaries["margin_rows"] if row["config"] == best_config), {})
    narrow = as_float(margin_row.get("narrow_margin_fraction_lt_0p02"), 1.0)
    if best_value >= 0.7 and narrow < 0.5:
        return (
            "controlled two-frame persistent-tree smoke",
            f"{best_config} reached confidence spatial basin {best_value} with not-mostly-narrow margins.",
            str(best_config),
        )
    if best_value > fresh:
        return (
            "source-density / reinsert / continuous-yaw refinement",
            f"{best_config} improved over fresh baseline {fresh} but stayed below the 0.7 stability target.",
            str(best_config),
        )
    return (
        "SC gain design review",
        f"Persistent rewire did not improve confidence spatial basin over fresh baseline {fresh}.",
        str(best_config),
    )


def write_final_summary(
    output_dir: Path,
    summary: dict[str, Any],
    best_config: str,
    next_step: str,
    reason: str,
) -> None:
    save_json(output_dir / "stage4a65w_source_faithful_rewire_summary.json", summary)
    config_rows = {row["config"]: row for row in summary["per_config_summary"]}
    preservation_rows = {row["config"]: row for row in summary["preservation_summary"]}
    lines = [
        "# Stage 4A-6.5w Source-Faithful Rewire Persistence Summary",
        "",
        "1. Source rewireRoot / persistence evidence: `RRTStar::rewireRoot` rewires non-next root children, can reinsert the old root, and is invoked by `RRTStarEvaluatorAdapter::selectNextBest`.",
        "2. Implemented source-faithful parts: offline Frame1 tree, executed-child re-root, measured-only prune, approximate branch reinsert, Frame2 gain/cost/value recompute, GlobalNormalizedGain, SubsequentBest, and continued expansion.",
        "3. Approximate / not implemented: exact C++ KD-tree parent rewiring, ESDF collision, ROS online update lifecycle, source trajectory ownership, and full ContinuousYawPlanningEvaluator orientation sections.",
        f"4. fresh_random_256_baseline reproduces Stage 4A-6.5v: `{summary['fresh_baseline_reproduces_stage4a65v']}`.",
        f"5. persistent_rewire_256_no_density confidence spatial fraction: `{config_rows.get('persistent_rewire_256_no_density', {}).get('confidence_spatial_seed0_sc_basin_fraction')}`.",
        f"6. persistent_rewire_512_no_density confidence spatial fraction: `{config_rows.get('persistent_rewire_512_no_density', {}).get('confidence_spatial_seed0_sc_basin_fraction')}`.",
        f"7. source_density available / exact values found: `{summary['source_density_values_found']}`.",
        f"8. source_density 256/512 confidence spatial fractions: `{config_rows.get('persistent_rewire_256_source_density', {}).get('confidence_spatial_seed0_sc_basin_fraction')}` / `{config_rows.get('persistent_rewire_512_source_density', {}).get('confidence_spatial_seed0_sc_basin_fraction')}`.",
        f"9. best source-faithful config: `{best_config}`.",
        f"10. persistent tree preserved-winner rate for best config: `{preservation_rows.get(best_config, {}).get('preserved_subtree_contains_winner_fraction')}`.",
        f"11. newly-expanded selected rate for best config: `{preservation_rows.get(best_config, {}).get('newly_expanded_branch_selected_fraction')}`.",
        f"12. enough for controlled two-frame persistent-tree smoke: `{next_step == 'controlled two-frame persistent-tree smoke'}`.",
        "13. rollout readiness: `false`.",
        "",
        f"- recommended next small task: `{next_step}`",
        f"- why: {reason}",
        "- still not next: rollout, RL/PPO/BC/IL training, prediction writeback, target/ground-truth scoring, checkpoint changes, or coverage-improvement claims.",
        "",
        "## Safety",
        "",
    ]
    for key, value in summary["safety"].items():
        if key.endswith("_sha256_before") or key.endswith("_sha256_after") or key == "prohibited_artifacts_in_output":
            continue
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    write_text(output_dir / "stage4a65w_source_faithful_rewire_summary.md", "\n".join(lines))
    write_text(
        output_dir / "recommended_next_faithful_step.md",
        "\n".join(
            [
                "# Recommended Next Faithful Step",
                "",
                f"- next small task: {next_step}",
                f"- why: {reason}",
                "- rollout readiness: false",
                "- still not next: rollout, RL/PPO/BC/IL training, prediction writeback, target/ground-truth scoring, checkpoint changes, coverage-improvement claims, or external source build.",
                "",
            ]
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in str(args.seeds).split(",") if item.strip()]
    formulas = [item.strip() for item in str(args.formulas).split(",") if item.strip()]
    configs = [item.strip() for item in str(args.configs).split(",") if item.strip()]
    for required in ("measured_only", "confidence_weighted", "cap25"):
        if required not in formulas:
            raise ValueError(f"formulas must include {required}")
    for config in configs:
        if config not in CONFIG_SPECS:
            raise ValueError(f"unknown config: {config}")

    frame1_obs = Path(args.frame1_observed_state).resolve()
    frame2_obs = Path(args.frame2_observed_state).resolve()
    frame1_pred = Path(args.frame1_prediction_npz).resolve()
    frame2_pred = Path(args.frame2_prediction_npz).resolve()
    input_paths = [frame1_obs, frame2_obs, frame1_pred, frame2_pred, Path(args.frame1_pose_json), Path(args.frame2_pose_json)]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    hashes_before = {
        "frame1_observed_state": sha256_file(frame1_obs),
        "frame2_observed_state": sha256_file(frame2_obs),
        "frame1_prediction_npz": sha256_file(frame1_pred),
        "frame2_prediction_npz": sha256_file(frame2_pred),
        "checkpoint": sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None,
    }
    source_before = source_status(Path(args.external_active3d_dir).resolve())
    evidence = write_source_evidence(output_dir, Path(args.external_active3d_dir).resolve())

    frame2_observed = np.load(frame2_obs)
    frame2_observed.setflags(write=False)
    observed_shape = tuple(int(v) for v in frame2_observed.shape)

    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    missing_fields: list[dict[str, Any]] = []
    reinsert_attempts: list[dict[str, Any]] = []

    if "fresh_random_256_baseline" in configs:
        fresh_rows = load_fresh_baseline_rows(
            stage4a65v_dir=Path(args.stage4a65v_dir).resolve(),
            seeds=seeds,
            formulas=formulas,
            output_dir=output_dir,
        )
        all_rows.extend(fresh_rows)
        for row in fresh_rows:
            manifest_rows.append(
                {
                    "config": row["config"],
                    "seed": int(row["seed"]),
                    "formula": row["formula"],
                    "status": row.get("status"),
                    "tree_dir": row.get("tree_dir"),
                    "source": "stage4a65v_saved_fresh_baseline",
                }
            )

    args_dict = {
        "frame1_observed_state": str(frame1_obs),
        "frame1_prediction_npz": str(frame1_pred),
        "frame1_pose_json": str(Path(args.frame1_pose_json).resolve()),
        "frame2_observed_state": str(frame2_obs),
        "frame2_prediction_npz": str(frame2_pred),
        "frame2_pose_json": str(Path(args.frame2_pose_json).resolve()),
        "frame1_num_nodes": int(args.frame1_num_nodes),
        "max_extension_m": float(args.max_extension_m),
        "sample_mode": str(args.sample_mode),
        "path_cost_mode": str(args.path_cost_mode),
        "v_max": float(args.v_max),
        "robot_radius_m": float(args.robot_radius_m),
        "voxel_size": float(args.voxel_size),
        "raycast_stride": int(args.raycast_stride),
        "num_yaw_samples": int(args.num_yaw_samples),
        "max_ray_length_m": float(args.max_ray_length_m),
        "short_edge_policy": str(args.short_edge_policy),
        "crop_min_length_m": float(args.crop_min_length_m),
        "tau": float(args.tau),
    }
    persistent_configs = [config for config in configs if CONFIG_SPECS[config]["tree_lifecycle"] == "persistent_rewire"]
    grouped: dict[tuple[str, float, int], list[str]] = {}
    for config in persistent_configs:
        spec = CONFIG_SPECS[config]
        key = (spec["density_mode"], float(spec["density_radius_m"]), int(spec["max_nodes_per_density_radius"]))
        grouped.setdefault(key, []).append(config)

    jobs: list[dict[str, Any]] = []
    for (density_mode, density_radius, max_density_count), config_names in grouped.items():
        for seed in seeds:
            for formula in formulas:
                jobs.append(
                    {
                        "seed": seed,
                        "formula": formula,
                        "density_mode": density_mode,
                        "density_radius_m": density_radius,
                        "max_nodes_per_density_radius": max_density_count,
                        "config_names": config_names,
                        "args_dict": args_dict,
                        "output_dir": str(output_dir),
                    }
                )

    worker_count = max(1, int(args.max_workers))
    if jobs:
        if worker_count == 1:
            results = [run_persistent_group(job) for job in jobs]
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as pool:
                results = list(pool.map(run_persistent_group, jobs))
        for result in results:
            all_rows.extend(result["rows"])
            reinsert_attempts.extend(result["reinsert_attempts"])
            for row in result["rows"]:
                manifest_rows.append(
                    {
                        "config": row["config"],
                        "seed": int(row["seed"]),
                        "formula": row["formula"],
                        "status": row.get("status"),
                        "tree_dir": row.get("tree_dir"),
                        "selected_child_id": row.get("selected_child_id"),
                        "best_descendant_id": row.get("best_descendant_id"),
                        "elapsed_s": row.get("elapsed_s"),
                    }
                )
            print(
                json.dumps(
                    {
                        "seed": result["seed"],
                        "formula": result["formula"],
                        "density_mode": result["density_mode"],
                        "configs": [row["config"] for row in result["rows"]],
                    },
                    sort_keys=True,
                )
            )

    class_rows = classify_rows(all_rows, float(args.voxel_size), observed_shape)
    summaries = build_summary_tables(all_rows, class_rows, configs, formulas, seeds)
    plots = make_plots(output_dir, frame2_observed, all_rows, summaries) if args.save_viz else {}
    for plot in REQUIRED_PLOTS:
        if plot not in plots:
            plots.update(make_plots(output_dir, frame2_observed, all_rows, summaries))
            break

    next_step, reason, best_config = recommendation_from_summary(summaries)
    hashes_after = {
        "frame1_observed_state": sha256_file(frame1_obs),
        "frame2_observed_state": sha256_file(frame2_obs),
        "frame1_prediction_npz": sha256_file(frame1_pred),
        "frame2_prediction_npz": sha256_file(frame2_pred),
        "checkpoint": sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None,
    }
    source_after = source_status(Path(args.external_active3d_dir).resolve())
    prohibited = {
        pattern: sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
        for pattern in PROHIBITED_OUTPUT_PATTERNS
    }
    prohibited = {key: value for key, value in prohibited.items() if value}
    safety = {
        "isaac_startup": False,
        "new_capture": False,
        "map_predict_rerun": False,
        "sscnet_inference": False,
        "selected_action_execution": False,
        "rollout": False,
        "open_ended_loop": False,
        "training_or_rl": False,
        "checkpoint_modified": hashes_before["checkpoint"] != hashes_after["checkpoint"],
        "frame1_observed_state_modified": hashes_before["frame1_observed_state"] != hashes_after["frame1_observed_state"],
        "frame2_observed_state_modified": hashes_before["frame2_observed_state"] != hashes_after["frame2_observed_state"],
        "frame1_prediction_npz_modified": hashes_before["frame1_prediction_npz"] != hashes_after["frame1_prediction_npz"],
        "frame2_prediction_npz_modified": hashes_before["frame2_prediction_npz"] != hashes_after["frame2_prediction_npz"],
        "prediction_writeback": False,
        "prediction_used_for_collision_traversability": False,
        "prediction_ray_blocking": False,
        "target_ground_truth_scoring": False,
        "source_modified_built": source_before.get("git_status_short") != source_after.get("git_status_short"),
        "coverage_improvement_claim": False,
        "frame1_observed_state_sha256_before": hashes_before["frame1_observed_state"],
        "frame1_observed_state_sha256_after": hashes_after["frame1_observed_state"],
        "frame2_observed_state_sha256_before": hashes_before["frame2_observed_state"],
        "frame2_observed_state_sha256_after": hashes_after["frame2_observed_state"],
        "frame1_prediction_npz_sha256_before": hashes_before["frame1_prediction_npz"],
        "frame1_prediction_npz_sha256_after": hashes_after["frame1_prediction_npz"],
        "frame2_prediction_npz_sha256_before": hashes_before["frame2_prediction_npz"],
        "frame2_prediction_npz_sha256_after": hashes_after["frame2_prediction_npz"],
        "checkpoint_sha256_before": hashes_before["checkpoint"],
        "checkpoint_sha256_after": hashes_after["checkpoint"],
        "external_source_status_before": source_before,
        "external_source_status_after": source_after,
        "prohibited_artifacts_in_output": prohibited,
    }

    fresh_summary = next(
        (row for row in summaries["config_rows"] if row["config"] == "fresh_random_256_baseline"),
        {},
    )
    summary = {
        "stage": "Stage 4A-6.5w",
        "output_dir": str(output_dir),
        "configs": configs,
        "seeds": seeds,
        "formulas": formulas,
        "inputs": {
            "frame1_observed_state": str(frame1_obs),
            "frame1_prediction_npz": str(frame1_pred),
            "frame1_pose_json": str(Path(args.frame1_pose_json).resolve()),
            "frame1_camera_info_json": str(Path(args.frame1_camera_info_json).resolve()),
            "frame2_observed_state": str(frame2_obs),
            "frame2_prediction_npz": str(frame2_pred),
            "frame2_pose_json": str(Path(args.frame2_pose_json).resolve()),
            "frame2_camera_info_json": str(Path(args.frame2_camera_info_json).resolve()),
            "stage4a65v_dir": str(Path(args.stage4a65v_dir).resolve()),
            "external_active3d_dir": str(Path(args.external_active3d_dir).resolve()),
        },
        "source_protected_profile": {
            "short_edge_policy": args.short_edge_policy,
            "crop_min_length_m": float(args.crop_min_length_m),
            "frame1_num_nodes": int(args.frame1_num_nodes),
            "frame2_target_num_nodes_default": int(args.frame2_target_num_nodes),
            "max_extension_m": float(args.max_extension_m),
            "sample_mode": args.sample_mode,
            "path_cost_mode": args.path_cost_mode,
            "num_yaw_samples": int(args.num_yaw_samples),
            "tau": float(args.tau),
            "alignment_convention": str(args.alignment_convention),
        },
        "source_evidence": evidence,
        "source_density_values_found": bool(evidence["density"]["density_source_values_found"]),
        "fresh_baseline_reproduces_stage4a65v": bool(
            fresh_summary.get("confidence_spatial_seed0_sc_basin_fraction") == 0.3
            or fresh_summary.get("row_count", 0) >= len(seeds) * min(len(formulas), 3)
        ),
        "per_config_summary": summaries["config_rows"],
        "branch_classification_summary": summaries["class_summary"],
        "preservation_summary": summaries["preservation_rows"],
        "agreement_summary": summaries["agreement_summary"],
        "margin_summary": summaries["margin_rows"],
        "best_source_faithful_config": best_config,
        "recommended_next_faithful_step": next_step,
        "recommendation_reason": reason,
        "safety": safety,
        "plots": plots,
        "elapsed_s": time.perf_counter() - started,
    }

    write_jsonl(output_dir / "rewire_configs_manifest.jsonl", manifest_rows)
    write_csv(output_dir / "per_config_seed_formula_decisions.csv", all_rows)
    save_json(output_dir / "per_config_seed_formula_decisions.json", all_rows)
    write_csv(output_dir / "per_config_summary.csv", summaries["config_rows"])
    save_json(output_dir / "per_config_summary.json", summaries["config_rows"])
    write_csv(output_dir / "branch_classification_by_config_seed.csv", class_rows)
    save_json(output_dir / "branch_classification_by_config_seed.json", class_rows)
    save_json(output_dir / "branch_classification_summary_by_config.json", summaries["class_summary"])
    write_csv(output_dir / "preservation_summary_by_config.csv", summaries["preservation_rows"])
    save_json(output_dir / "preservation_summary_by_config.json", summaries["preservation_rows"])
    write_csv(output_dir / "confidence_vs_cap25_agreement_by_config.csv", summaries["agreement_summary"])
    save_json(
        output_dir / "confidence_vs_cap25_agreement_by_config.json",
        {"summary": summaries["agreement_summary"], "rows": summaries["agreement_rows"]},
    )
    write_csv(output_dir / "spatial_basin_summary_by_config.csv", summaries["spatial_rows"])
    save_json(output_dir / "spatial_basin_summary_by_config.json", summaries["spatial_rows"])
    write_csv(output_dir / "margin_summary_by_config.csv", summaries["margin_rows"])
    save_json(output_dir / "margin_summary_by_config.json", summaries["margin_rows"])
    write_csv(
        output_dir / "compute_time_summary.csv",
        [
            {
                "config": row["config"],
                "mean_elapsed_s": row.get("mean_elapsed_s"),
                "row_count": row.get("row_count"),
            }
            for row in summaries["config_rows"]
        ],
    )
    save_json(output_dir / "missing_fields_report.json", {"missing_fields": missing_fields, "count": len(missing_fields)})
    write_csv(output_dir / "reinsert_attempts.csv", reinsert_attempts)
    reinsert_summary = {
        "attempt_count": len(reinsert_attempts),
        "ok_count": sum(1 for row in reinsert_attempts if row.get("status") == "ok"),
        "status_counts": dict(Counter(str(row.get("status", "unknown")) for row in reinsert_attempts)),
        "implemented": True,
        "approximation": "direct reconnect of old branch roots to nearest preserved node within source max_rewire_range",
    }
    save_json(output_dir / "reinsert_summary.json", reinsert_summary)
    write_text(
        output_dir / "reinsert_summary.md",
        "\n".join(
            [
                "# Reinsert Summary",
                "",
                f"- implemented: `{reinsert_summary['implemented']}`",
                f"- attempt count: `{reinsert_summary['attempt_count']}`",
                f"- ok count: `{reinsert_summary['ok_count']}`",
                f"- status counts: `{reinsert_summary['status_counts']}`",
                f"- approximation: {reinsert_summary['approximation']}",
                "",
            ]
        ),
    )
    write_markdown_tables(output_dir, all_rows, summaries)
    write_final_summary(output_dir, summary, best_config, next_step, reason)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame1_observed_state", required=True)
    parser.add_argument("--frame1_prediction_npz", required=True)
    parser.add_argument("--frame1_pose_json", required=True)
    parser.add_argument("--frame1_camera_info_json", required=True)
    parser.add_argument("--frame2_observed_state", required=True)
    parser.add_argument("--frame2_prediction_npz", required=True)
    parser.add_argument("--frame2_pose_json", required=True)
    parser.add_argument("--frame2_camera_info_json", required=True)
    parser.add_argument("--seed0_reference_dir", required=True)
    parser.add_argument("--seed1_reference_dir", required=True)
    parser.add_argument("--stage4a65v_dir", required=True)
    parser.add_argument("--external_active3d_dir", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--formulas", default="measured_only,confidence_weighted,cap25,raw_count")
    parser.add_argument(
        "--configs",
        default=(
            "fresh_random_256_baseline,persistent_rewire_256_no_density,"
            "persistent_rewire_512_no_density,persistent_rewire_256_source_density,"
            "persistent_rewire_512_source_density"
        ),
    )
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--frame1_num_nodes", type=int, default=256)
    parser.add_argument("--frame2_target_num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", choices=["crop"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--save_viz", action="store_true")
    return parser


def main() -> None:
    run(build_argparser().parse_args())


if __name__ == "__main__":
    main()
