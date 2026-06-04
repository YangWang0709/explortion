#!/usr/bin/env python3
"""CLI for Stage 4A-2 Isaac observed-map expert decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim_paper_expert import (
    DEFAULT_BOUNDS,
    EmptyPredictionLayer,
    SC_GAIN_FORMULAS,
    format_top_candidates,
    normalize_bounds,
    save_expert_step_outputs,
    select_sim_expert_action,
)
from sim_prediction_layer import SimPredictionLayer

DEFAULT_INPUT_DIR = Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_depth_smoke")
DEFAULT_OUTPUT_DIR = Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_expert_step_smoke")


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_current_pose(path: str | Path) -> dict[str, Any]:
    pose = load_json(path)
    if "position" not in pose:
        raise KeyError(f"{path} does not contain a position field")
    current_pose = {"position": [float(v) for v in pose["position"]]}
    if "yaw_rad" in pose:
        current_pose["yaw_rad"] = float(pose["yaw_rad"])
        current_pose["yaw_source"] = "yaw_rad"
    elif "yaw_deg" in pose:
        current_pose["yaw_rad"] = float(np.deg2rad(float(pose["yaw_deg"])))
        current_pose["yaw_source"] = "yaw_deg"
    else:
        current_pose["yaw_rad"] = 0.0
        current_pose["yaw_source"] = "fallback_zero"
        current_pose["yaw_limitation"] = "pose metadata did not contain yaw_rad or yaw_deg"
    return current_pose


def resolve_bounds_and_voxel_size(
    observed_summary: dict[str, Any],
    scene_metadata: dict[str, Any],
) -> tuple[dict[str, tuple[float, float]], float]:
    raw_bounds = observed_summary.get("map_bounds") or scene_metadata.get("map_bounds") or DEFAULT_BOUNDS
    bounds = normalize_bounds(raw_bounds)
    voxel_size = float(
        observed_summary.get(
            "voxel_size",
            scene_metadata.get("voxel_size_recommended", 0.1),
        )
    )
    return bounds, voxel_size


def run_expert_step(args: argparse.Namespace) -> dict[str, Any]:
    observed_state_path = Path(args.observed_state).resolve()
    observed_summary_path = Path(args.observed_summary).resolve() if args.observed_summary else None
    episode_summary_arg = getattr(args, "episode_summary", None)
    episode_summary_path = Path(episode_summary_arg).resolve() if episode_summary_arg else None
    camera_info_path = Path(args.camera_info).resolve()
    pose_json_path = Path(args.pose_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    sc_gain_formula = str(getattr(args, "sc_gain_formula", "raw_count"))
    sc_occ_threshold = float(getattr(args, "sc_occ_threshold", 0.7))
    sc_conf_threshold = float(getattr(args, "sc_conf_threshold", 0.3))
    sc_count_mode = str(getattr(args, "sc_count_mode", "raw_count"))
    calibration_table = getattr(args, "calibration_table", None)
    alignment_convention = str(getattr(args, "alignment_convention", "code_consistent_v1"))

    observed_state_sha256_before = sha256_file(observed_state_path)
    observed_state = np.load(observed_state_path)
    observed_state.setflags(write=False)
    observed_summary = load_json(observed_summary_path) if observed_summary_path and observed_summary_path.exists() else {}
    episode_summary = load_json(episode_summary_path) if episode_summary_path else {}
    camera_info = load_json(camera_info_path)
    scene_metadata_path = observed_state_path.parent / "scene_metadata.json"
    scene_metadata = load_json(scene_metadata_path) if scene_metadata_path.exists() else {}
    current_pose = load_current_pose(pose_json_path)
    summary_for_bounds = episode_summary if episode_summary else observed_summary
    bounds, voxel_size = resolve_bounds_and_voxel_size(summary_for_bounds, scene_metadata)

    prediction_npz_path: Path | None = None
    if args.prediction_mode == "empty":
        prediction_layer = EmptyPredictionLayer(tuple(observed_state.shape))
    elif args.prediction_mode == "sim_npz":
        if not args.prediction_npz:
            raise ValueError("--prediction_npz is required when --prediction_mode sim_npz")
        prediction_npz_path = Path(args.prediction_npz).resolve()
        prediction_layer = SimPredictionLayer.from_npz(prediction_npz_path)
        if tuple(prediction_layer.shape()) != tuple(observed_state.shape):
            raise ValueError(
                f"prediction layer shape {prediction_layer.shape()} differs from observed_state {observed_state.shape}"
            )
        with np.load(prediction_npz_path, allow_pickle=False) as data:
            if "alignment_convention" in data.files:
                actual_alignment = str(np.asarray(data["alignment_convention"]).item())
                if actual_alignment and actual_alignment != alignment_convention:
                    raise ValueError(
                        f"prediction alignment_convention {actual_alignment} != requested {alignment_convention}"
                    )
    else:
        raise ValueError("prediction_mode must be one of: empty, sim_npz")

    result = select_sim_expert_action(
        observed_state=observed_state,
        current_pose_world=current_pose,
        bounds=bounds,
        voxel_size=voxel_size,
        prediction_layer=prediction_layer,
        prediction_mode=args.prediction_mode,
        num_candidates=args.num_candidates,
        top_n=args.top_n,
        gain_mode=args.gain_mode,
        sc_gain_formula=sc_gain_formula,
        sc_occ_threshold=sc_occ_threshold,
        sc_conf_threshold=sc_conf_threshold,
        sc_count_mode=sc_count_mode,
        score_gain_mode=str(getattr(args, "score_gain_mode", "hybrid_raw")),
        sc_gain_weight=float(getattr(args, "sc_gain_weight", 1.0)),
        sc_gain_cap=getattr(args, "sc_gain_cap", None),
        calibration_table=calibration_table,
        seed=args.seed,
        tau=float(getattr(args, "tau", 0.1)),
        max_range_voxels=args.max_range_voxels,
        num_yaw=args.num_yaw,
        num_pitch=args.num_pitch,
        fov_yaw_deg=args.fov_yaw_deg,
        fov_pitch_deg=args.fov_pitch_deg,
        path_cost_mode=args.path_cost_mode,
        candidate_sampling_mode=args.candidate_sampling_mode,
        snap_start_to_traversable=bool(args.snap_start_to_traversable),
        max_snap_radius_cells=int(args.max_snap_radius_cells),
    )
    observed_state_sha256_after = sha256_file(observed_state_path)
    result["diagnostics"].update(
        {
            "observed_state_path": str(observed_state_path),
            "observed_summary_path": str(observed_summary_path) if observed_summary_path else None,
            "episode_summary_path": str(episode_summary_path) if episode_summary_path else None,
            "camera_info_path": str(camera_info_path),
            "pose_json_path": str(pose_json_path),
            "scene_metadata_path": str(scene_metadata_path) if scene_metadata_path.exists() else None,
            "prediction_npz_path": str(prediction_npz_path) if prediction_npz_path else None,
            "alignment_convention": alignment_convention,
            "sc_gain_formula": sc_gain_formula,
            "sc_occ_threshold": sc_occ_threshold,
            "sc_conf_threshold": sc_conf_threshold,
            "sc_count_mode": sc_count_mode,
            "calibration_table": str(Path(calibration_table).resolve()) if calibration_table else None,
            "observed_state_sha256_before": observed_state_sha256_before,
            "observed_state_sha256_after": observed_state_sha256_after,
            "observed_state_hash_unchanged": observed_state_sha256_before == observed_state_sha256_after,
            "camera_info": {
                "width": camera_info.get("width"),
                "height": camera_info.get("height"),
                "horizontal_fov_deg": camera_info.get("horizontal_fov_deg"),
                "max_depth": camera_info.get("max_depth"),
            },
            "pose_source": current_pose.get("yaw_source", "unknown"),
        }
    )

    output_paths = save_expert_step_outputs(
        result=result,
        output_dir=output_dir,
        observed_state_path=observed_state_path,
    )

    viz_paths: dict[str, str] = {}
    if args.save_viz:
        from visualize_sim_expert_step import save_expert_visualizations

        viz_paths = save_expert_visualizations(
            observed_state=observed_state,
            result=result,
            output_dir=output_dir,
            prediction_layer=prediction_layer if args.prediction_mode == "sim_npz" else None,
        )
        output_paths.update(viz_paths)

    result["output_paths"] = output_paths
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4A-2 one-step simulator expert scorer.")
    parser.add_argument("--observed_state", default=str(DEFAULT_INPUT_DIR / "observed_state_step2.npy"))
    parser.add_argument("--observed_summary", default=str(DEFAULT_INPUT_DIR / "observed_summary.json"))
    parser.add_argument("--episode_summary", default=None)
    parser.add_argument("--camera_info", default=str(DEFAULT_INPUT_DIR / "camera_info.json"))
    parser.add_argument("--pose_json", default=str(DEFAULT_INPUT_DIR / "pose_002.json"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num_candidates", type=int, default=64)
    parser.add_argument("--top_n", type=int, default=16)
    parser.add_argument(
        "--gain_mode",
        choices=("exp", "sc", "hybrid", "occ", "conf"),
        default="hybrid",
    )
    parser.add_argument("--prediction_mode", choices=("empty", "sim_npz"), default="empty")
    parser.add_argument("--prediction_npz", default=None)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--sc_gain_formula", choices=SC_GAIN_FORMULAS, default="raw_count")
    parser.add_argument("--sc_occ_threshold", type=float, default=0.7)
    parser.add_argument("--sc_conf_threshold", type=float, default=0.3)
    parser.add_argument("--sc_count_mode", choices=("raw_count", "selective"), default="raw_count")
    parser.add_argument("--calibration_table", default=None)
    parser.add_argument(
        "--alignment_convention",
        choices=("current_default_v0", "code_consistent_v1", "xz_swap_diagnostic"),
        default="code_consistent_v1",
    )
    parser.add_argument("--sc_gain_weight", type=float, default=1.0)
    parser.add_argument("--sc_gain_cap", type=float, default=None)
    parser.add_argument(
        "--score_gain_mode",
        choices=("hybrid_raw", "hybrid_weighted", "decoupled_sc"),
        default="hybrid_raw",
    )
    parser.add_argument("--path_cost_mode", choices=("euclidean", "astar"), default="euclidean")
    parser.add_argument("--candidate_sampling_mode", choices=("frontier", "reachable_frontier", "auto"), default="auto")
    parser.add_argument("--snap_start_to_traversable", action="store_true")
    parser.add_argument("--max_snap_radius_cells", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_range_voxels", type=int, default=50)
    parser.add_argument("--num_yaw", type=int, default=32)
    parser.add_argument("--num_pitch", type=int, default=7)
    parser.add_argument("--fov_yaw_deg", type=float, default=90.0)
    parser.add_argument("--fov_pitch_deg", type=float, default=60.0)
    parser.add_argument("--print_topn", action="store_true")
    parser.add_argument("--save_viz", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_expert_step(args)
    diagnostics = result["diagnostics"]
    best = result["best_candidate"]

    if args.prediction_mode == "sim_npz":
        stage_message = "Stage 4A-5.1 simulator expert step complete."
    elif args.path_cost_mode == "astar":
        stage_message = "Stage 4A-3.6 simulator expert step complete."
    else:
        stage_message = "Stage 4A-2 simulator expert step complete."
    print(stage_message)
    print(f"observed_state: {Path(args.observed_state).resolve()}")
    print(f"prediction_mode: {args.prediction_mode}")
    if args.prediction_mode == "sim_npz":
        print(f"prediction_npz: {Path(args.prediction_npz).resolve()}")
    print(f"tau: {getattr(args, 'tau', 0.1)}")
    print(f"gain_mode: {args.gain_mode}")
    print(f"sc_gain_formula: {getattr(args, 'sc_gain_formula', 'raw_count')}")
    print(f"sc_occ_threshold: {getattr(args, 'sc_occ_threshold', 0.7)}")
    print(f"sc_conf_threshold: {getattr(args, 'sc_conf_threshold', 0.3)}")
    print(f"alignment_convention: {getattr(args, 'alignment_convention', 'code_consistent_v1')}")
    print(f"score_gain_mode: {getattr(args, 'score_gain_mode', 'hybrid_raw')}")
    print(f"sc_gain_weight: {getattr(args, 'sc_gain_weight', 1.0)}")
    print(f"sc_gain_cap: {getattr(args, 'sc_gain_cap', None)}")
    print(f"path_cost_mode: {args.path_cost_mode}")
    print(f"candidate_sampling_mode: {diagnostics.get('candidate_sampling_mode')}")
    print(f"shape: {tuple(diagnostics['shape'])}")
    print(
        "counts: "
        f"unknown={diagnostics['unknown_count']} free={diagnostics['free_count']} "
        f"occupied={diagnostics['occupied_count']} observed_ratio={diagnostics['observed_ratio']:.6f}"
    )
    print(
        "frontiers/candidates: "
        f"frontier_count={diagnostics['frontier_count']} "
        f"frontier_adjacent_free_count={diagnostics['frontier_adjacent_free_count']} "
        f"candidates={diagnostics['num_candidates']} top_n={diagnostics['top_n']}"
    )
    if diagnostics.get("path_cost_mode") == "astar":
        print(
            "traversability: "
            f"traversable={diagnostics.get('traversable_count')} "
            f"blocked={diagnostics.get('blocked_count')} "
            f"unknown={diagnostics.get('traversability_unknown_count')}"
        )
        print(
            "astar_candidates: "
            f"reachable={diagnostics.get('reachable_candidates')} "
            f"unreachable={diagnostics.get('unreachable_candidates')}"
        )
        print(
            "reachable_component: "
            f"count={diagnostics.get('reachable_component_count')} "
            f"frontier_adjacent={diagnostics.get('reachable_frontier_adjacent_count')} "
            f"source={diagnostics.get('candidate_source')} "
            f"snapped={diagnostics.get('snapped_current')} "
            f"snap_xy={diagnostics.get('snapped_current_xy')} "
            f"snap_distance={diagnostics.get('snap_distance_cells')}"
        )
    print(f"expert_action: {result['expert_action']}")
    print(
        "best_candidate: "
        f"id={best.id} score={best.final_score:.6f} gain_exp={best.gain_exp:.1f} "
        f"gain_sc={best.gain_sc:.1f} gain_hybrid={best.gain_hybrid:.1f} "
        f"weighted_gain_sc={best.weighted_gain_sc:.1f} gain_hybrid_weighted={best.gain_hybrid_weighted:.1f} "
        f"gain_occ={best.gain_occ:.1f} gain_conf={best.gain_conf:.6f} "
        f"path_cost={best.path_cost:.6f} grid={best.grid_position} "
        f"world=({best.world_position[0]:.2f},{best.world_position[1]:.2f},{best.world_position[2]:.2f}) "
        f"yaw={best.yaw:.6f} astar_path_length_m={best.astar_path_length_m:.6f}"
    )
    if args.prediction_mode == "sim_npz":
        print(
            "prediction_counts: "
            f"valid={diagnostics.get('prediction_valid_voxels')} "
            f"predicted={diagnostics.get('prediction_predicted_voxels')} "
            f"predicted_unmeasured={diagnostics.get('prediction_predicted_unmeasured_voxels')} "
            f"predicted_occupied={diagnostics.get('prediction_predicted_occupied_voxels')}"
        )
    for label, path in result["output_paths"].items():
        print(f"output_{label}: {path}")

    if args.print_topn:
        print("top_candidates:")
        print(format_top_candidates(result["top_candidates"]))


if __name__ == "__main__":
    main()
