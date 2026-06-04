#!/usr/bin/env python3
"""Stage 4A-6.5ae real Frame1 saved-frame lambda48 formula smoke.

This runner is offline-only. It reads the saved Stage 4A-6.5p real
medium_three_rooms Frame1 observed map and prediction NPZ, rebuilds one-frame
source-protected mini-RRT decisions for seeds 0..9, and evaluates:

    value = gain_exp / cost + 48 * minmax(source_occ_free)

It does not start Isaac, capture frames, rerun map_predict, run SSCNet
inference, execute selected actions, write predictions into observed_state, or
use prediction for traversability, collision, edge validity, or ray blocking.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import run_real_frame_lambda48_formula_smoke as base


DEFAULT_STAGE4A65P_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65p_map_predict_tree_two_frame_smoke"
)
DEFAULT_STAGE4A65AC_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ac_saved_frame_lambda48_formula_smoke"
)
DEFAULT_STAGE4A65AD_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ad_real_frame_lambda48_formula_smoke"
)
DEFAULT_STAGE4A65Y_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65y_source_gain_seed_replay"
)
DEFAULT_STAGE4A65Z_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65z_decoupled_sc_utility_sweep"
)
DEFAULT_STAGE4A65Z1_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65z1_decoupled_signal_strength_diagnosis"
)
DEFAULT_OUTPUT_DIR = (
    "/home/ubuntu22/sc_explorer_ws/outputs/"
    "isaac_sc_pred_stage4a65ae_real_frame1_lambda48_formula_smoke"
)
DEFAULT_CHECKPOINT = (
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/"
    "cpBest_SSCNet_NYU_full_train.pth.tar"
)

FRAME1_FILES = {
    "observed_state": "observed_state_frame001.npy",
    "prediction_npz": "frame001_prediction/global_prediction_layer.npz",
    "pose_json": "frame001_pose.json",
    "camera_info_json": "frame001_camera_info.json",
    "observed_summary": "frame001_observed_summary.json",
}

HISTORICAL_FRAME2_PRIOR = {
    "label": "stage4a65ad_frame2_prior_low_cost_sc_reference",
    "selected_child_id": "n0127",
    "selected_child_grid": [11, 15, 11],
    "best_descendant_id": "n0162",
    "best_descendant_grid": [14, 15, 11],
}
HISTORICAL_FRAME2_SEED1 = {
    "label": "stage4a65ad_frame2_seed1_near_sc_basin_reference",
    "selected_child_id": "n0057",
    "selected_child_grid": [12, 16, 11],
    "best_descendant_id": "n0118",
    "best_descendant_grid": [12, 19, 11],
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def branch_from_decision(path: Path, label: str) -> dict[str, Any]:
    decision = load_json(path)
    selected = decision.get("selected_child") or {}
    best = decision.get("best_descendant") or {}
    return {
        "label": label,
        "selected_child_id": decision.get("selected_child_id") or selected.get("segment_id"),
        "selected_child_grid": selected.get("end_grid"),
        "best_descendant_id": decision.get("selected_child_best_descendant_id")
        or best.get("best_descendant_id")
        or best.get("segment_id"),
        "best_descendant_grid": best.get("end_grid"),
    }


def build_reference_branches(stage4a65p_dir: Path) -> dict[str, dict[str, Any]]:
    measured = branch_from_decision(
        stage4a65p_dir / "frame001_measured_tree_raw" / "subsequent_best_decision.json",
        "stage4a65p_frame1_measured_reference",
    )
    sc_seed0 = branch_from_decision(
        stage4a65p_dir / "frame001_sc_tree_raw" / "subsequent_best_decision.json",
        "stage4a65p_frame1_sc_seed0_reference",
    )
    return {
        "measured_reference": measured,
        "prior_low_cost_sc_reference": dict(HISTORICAL_FRAME2_PRIOR),
        "seed1_near_sc_basin": dict(HISTORICAL_FRAME2_SEED1),
        "stage4a65p_frame1_sc_seed0_reference": sc_seed0,
    }


def patch_base_module(stage4a65p_dir: Path, stage4a65ad_dir: Path, refs: dict[str, dict[str, Any]]) -> None:
    original_read_json = base.read_json
    original_compare = base.compare_to_stage4a65z_z1

    def read_json_frame1(path: Path | str | None) -> Any:
        if path is not None:
            candidate = Path(path)
            if candidate.name == "frame002_observed_summary.json" and candidate.parent == stage4a65p_dir:
                return original_read_json(stage4a65p_dir / FRAME1_FILES["observed_summary"])
        return original_read_json(path)

    def write_formula_definition(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
        definition = {
            "stage": "Stage 4A-6.5ae",
            "selected_frame": "Stage 4A-6.5p Frame1",
            "recommended_formula_name": "map_predict_source_occ_free_decoupled_minmax_lambda48",
            "recommended_formula": "gain_exp / cost + 48 * minmax(source_occ_free)",
            "lambda": float(args.lambda_sc),
            "lambda32_diagnostic_formula": "gain_exp / cost + 32 * minmax(source_occ_free)",
            "tau": float(args.tau),
            "occ_threshold": float(args.occ_threshold),
            "free_threshold": float(args.free_threshold),
            "source_occ_free": {
                "counts": "predicted OCC plus predicted FREE",
                "confidence_threshold": float(args.tau),
                "occ_threshold": float(args.occ_threshold),
                "free_threshold": float(args.free_threshold),
                "validity": "prediction-valid and unmeasured voxels only",
                "prediction_unknown_counted": False,
            },
            "minmax": {
                "scope": "per seed/tree over valid root-to-descendant path accumulated source_occ_free",
                "formula": "(sc - min_sc) / (max_sc - min_sc)",
                "flat_case": "0 for all candidates when max_sc == min_sc",
            },
            "diagnostic_only": {
                "source_occ_free_over_cost": "(gain_exp + source_occ_free) / cost",
                "raw_hybrid_over_cost": "(gain_exp + source_occ_free) / cost",
                "source_occ_free_no_cost": "source_occ_free",
            },
            "prediction_information_gain_only": True,
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
        }
        base.save_json(output_dir / "formula_definition.json", definition)
        lines = [
            "# Formula Definition",
            "",
            f"- stage: `{definition['stage']}`",
            f"- selected frame: `{definition['selected_frame']}`",
            f"- formula: `{definition['recommended_formula']}`",
            f"- lambda: `{definition['lambda']}`",
            f"- tau: `{definition['tau']}`",
            f"- occ/free thresholds: `{definition['occ_threshold']}` / `{definition['free_threshold']}`",
            "- SC is outside the cost denominator for lambda48 and lambda32.",
            "- over-cost formulas are diagnostic only.",
            "- prediction is information-gain-only; it is not used for traversability, collision, ray blocking, or observed-map writeback.",
        ]
        base.write_text(output_dir / "formula_definition.md", "\n".join(lines))
        return definition

    def write_reference_branches(output_dir: Path, voxel_size: float) -> None:
        payload = dict(refs)
        payload["spatial_rules"] = {
            "voxel_size": float(voxel_size),
            "same_as_measured": "selected child <=0.15m from same-seed measured selected child",
            "historical_prior_sc_basin": "Stage 4A-6.5ad Frame2 prior-risk reference retained only as an old-risk diagnostic",
            "distinct_nonmeasured_branch": "selected child differs from measured by >=0.25m and is not the historical prior basin",
        }
        base.save_json(output_dir / "real_frame_reference_branches.json", payload)
        measured = refs["measured_reference"]
        sc_seed0 = refs["stage4a65p_frame1_sc_seed0_reference"]
        lines = [
            "# Real Frame Reference Branches",
            "",
            f"- selected saved frame: Stage 4A-6.5p Frame1.",
            f"- measured Frame1 seed0 reference: `{measured['selected_child_id']} -> {measured['best_descendant_id']}`, grids `{measured['selected_child_grid']} -> {measured['best_descendant_grid']}`.",
            f"- Stage 4A-6.5p Frame1 SC seed0 reference: `{sc_seed0['selected_child_id']} -> {sc_seed0['best_descendant_id']}`, grids `{sc_seed0['selected_child_grid']} -> {sc_seed0['best_descendant_grid']}`.",
            f"- historical Frame2 prior-risk reference retained for diagnostic distance only: `{HISTORICAL_FRAME2_PRIOR['selected_child_id']} -> {HISTORICAL_FRAME2_PRIOR['best_descendant_id']}`.",
            "- this real-frame smoke does not use hidden-room labels or target/ground-truth.",
        ]
        base.write_text(output_dir / "real_frame_reference_branches.md", "\n".join(lines))

    def summarize_lambda48(decisions: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
        mode_summary = base.summarize_modes(decisions)
        by_mode = {row["mode"]: row for row in mode_summary}
        map48_rows = base.rows_for_mode(decisions, "map_predict_lambda48")
        measured_rows = base.rows_for_mode(decisions, "measured_only")
        seed0_measured = next((row for row in measured_rows if int(row["seed"]) == 0), {})
        seed0_map48 = next((row for row in map48_rows if int(row["seed"]) == 0), {})
        counts = Counter(str(row.get("branch_classification")) for row in map48_rows)
        if not map48_rows:
            behavior = "missing"
        elif counts.get("spatial_prior_sc_basin", 0) or counts.get("same_as_prior_low_cost_sc", 0):
            behavior = "historical_prior_sc_basin"
        elif counts.get("distinct_nonmeasured_branch", 0) or counts.get("same_as_seed1_near_sc_basin", 0):
            behavior = "distinct_or_near_nonmeasured"
        elif counts.get("same_as_measured", 0) == len(map48_rows):
            behavior = "collapse_to_measured"
        else:
            behavior = "mixed_seed_sensitive"
        measured_ref = refs["measured_reference"]
        return {
            "stage": "Stage 4A-6.5ae",
            "status": "completed",
            "selected_frame": "Stage 4A-6.5p Frame1",
            "seed_count": len(seeds),
            "mode_count": len({row["mode"] for row in decisions}),
            "decision_row_count": len(decisions),
            "map_predict_lambda48_behavior": behavior,
            "measured_only": by_mode.get("measured_only"),
            "map_predict_lambda32": by_mode.get("map_predict_lambda32"),
            "map_predict_lambda48": by_mode.get("map_predict_lambda48"),
            "source_occ_free_over_cost": by_mode.get("source_occ_free_over_cost"),
            "raw_hybrid_over_cost": by_mode.get("raw_hybrid_over_cost"),
            "source_occ_free_no_cost": by_mode.get("source_occ_free_no_cost"),
            "map_predict_lambda48_branch_classification_counts": dict(counts),
            "map_predict_lambda48_same_as_measured_fraction": base.fraction(map48_rows, "same_as_measured"),
            "map_predict_lambda48_spatial_prior_sc_basin_fraction": base.fraction(map48_rows, "spatial_prior_sc_basin"),
            "map_predict_lambda48_healthy_nonmeasured_fraction": base.fraction(
                map48_rows, "healthy_nonmeasured_candidate"
            ),
            "map_predict_lambda48_low_cost_artifact_fraction": base.fraction(map48_rows, "low_cost_artifact"),
            "map_predict_lambda48_avoids_prior_low_cost_sc_fraction": base.fraction(
                map48_rows, "avoids_prior_low_cost_sc"
            ),
            "seed0_measured_reference_reproduced": bool(
                seed0_measured.get("selected_child_id") == measured_ref["selected_child_id"]
                and seed0_measured.get("best_descendant_id") == measured_ref["best_descendant_id"]
            ),
            "seed0_map_predict_lambda48": {
                "selected_child_id": seed0_map48.get("selected_child_id"),
                "best_descendant_id": seed0_map48.get("best_descendant_id"),
                "selected_child_grid": seed0_map48.get("selected_child_grid"),
                "best_descendant_grid": seed0_map48.get("best_descendant_grid"),
                "branch_classification": seed0_map48.get("branch_classification"),
                "low_cost_artifact": seed0_map48.get("low_cost_artifact"),
                "healthy_nonmeasured_candidate": seed0_map48.get("healthy_nonmeasured_candidate"),
            },
            "formula_components_logged": all(
                key in row for row in map48_rows for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value")
            ),
            "saved_frame_only_readiness": True,
            "runtime_smoke_readiness": False,
            "rollout_readiness": False,
        }

    def compare_to_stage4a65z_z1(
        stage4a65z_dir: Path,
        stage4a65z1_dir: Path,
        lambda_summary: dict[str, Any],
    ) -> dict[str, Any]:
        comparison = original_compare(stage4a65z_dir, stage4a65z1_dir, lambda_summary)
        comparison.update(
            {
                "stage": "Stage 4A-6.5ae",
                "selected_frame": "Stage 4A-6.5p Frame1",
                "stage4a65ad_reference_dir": str(stage4a65ad_dir),
                "interpretation": (
                    "Frame1 lambda48 behavior is compared only to historical saved-frame "
                    "Stage 4A-6.5z/z.1/6.5ad diagnostics. This stage does not use Frame2 "
                    "as the evaluated frame and does not claim coverage improvement."
                ),
            }
        )
        return comparison

    def write_final_summary(
        output_dir: Path,
        *,
        answers: dict[str, Any],
        lambda_summary: dict[str, Any],
        comparison: dict[str, Any],
        next_step: str,
        why: str,
    ) -> dict[str, Any]:
        answers["selected_frame"] = "Stage 4A-6.5p Frame1"
        answers["selected_frame_index"] = 1
        answers["loaded_stage4a65p_frame1_inputs"] = bool(answers.pop("loaded_stage4a65p_frame2_inputs", False))
        answers["measured_only_reproduced_frame1_reference"] = bool(
            answers.pop("measured_only_reproduced_frame2_reference", False)
        )
        summary = {
            "stage": "Stage 4A-6.5ae",
            "status": "completed",
            "selected_frame": "Stage 4A-6.5p Frame1",
            "answers": answers,
            "lambda48_behavior_summary": lambda_summary,
            "comparison_to_stage4a65z_z1": comparison,
            "recommended_next_faithful_step": next_step,
            "recommendation_reason": why,
            "readiness": {
                "saved_frame_only": True,
                "runtime_smoke": False,
                "rollout": False,
            },
            "coverage_improvement_claimed": False,
        }
        base.save_json(output_dir / "stage4a65ae_real_frame1_lambda48_formula_summary.json", summary)
        lines = [
            "# Stage 4A-6.5ae Real Frame1 Lambda48 Formula Summary",
            "",
            f"1. Loaded Stage 4A-6.5p Frame1 inputs: `{answers['loaded_stage4a65p_frame1_inputs']}`.",
            f"2. No Isaac / new capture / map_predict rerun: `{answers['no_isaac_no_capture_no_map_predict_rerun']}`.",
            f"3. Seeds / modes / decision rows: `{answers['seed_count']}` / `{answers['mode_count']}` / `{answers['decision_row_count']}`.",
            f"4. measured_only reproduced Frame1 measured reference: `{answers['measured_only_reproduced_frame1_reference']}`.",
            f"5. map_predict lambda48 seed0 branch: `{answers['map_predict_lambda48_seed0_branch']}`.",
            f"6. lambda48 branch counts: `{answers['map_predict_lambda48_branch_counts']}`.",
            f"7. lambda48 classification: `{answers['map_predict_lambda48_behavior']}`.",
            f"8. lambda48 historical prior basin flag: `{answers['lambda48_prior_low_cost_sc_or_spatial_basin']}`.",
            f"9. lambda48 low-cost artifact fraction: `{answers['lambda48_low_cost_artifact_fraction']}`.",
            f"10. lambda48 healthy non-measured fraction: `{answers['lambda48_healthy_nonmeasured_fraction']}`.",
            f"11. Supported next step: `{next_step}`.",
            "12. Runtime two-frame / rollout readiness: `false / false`.",
            "",
            f"Why: {why}",
            "",
            "This is a saved-frame-only classification smoke and does not claim coverage improvement.",
        ]
        base.write_text(output_dir / "stage4a65ae_real_frame1_lambda48_formula_summary.md", "\n".join(lines))
        return summary

    base.read_json = read_json_frame1
    base.REFERENCE_BRANCHES = refs
    base.write_formula_definition = write_formula_definition
    base.write_reference_branches = write_reference_branches
    base.summarize_lambda48 = summarize_lambda48
    base.compare_to_stage4a65z_z1 = compare_to_stage4a65z_z1
    base.write_final_summary = write_final_summary


def exact_frame1_paths(stage4a65p_dir: Path) -> dict[str, Path]:
    return {key: stage4a65p_dir / rel for key, rel in FRAME1_FILES.items()}


def write_selected_frame_report(output_dir: Path, args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, Any]:
    exact_exists = {key: path.is_file() for key, path in paths.items()}
    report = {
        "stage": "Stage 4A-6.5ae",
        "selected_frame": "Stage 4A-6.5p Frame1",
        "selected_frame_index": 1,
        "selection_reason": "preferred primary Frame1 saved artifacts were present",
        "fallback_used": False,
        "selected_stage4a65ad_frame2": False,
        "no_silent_fallback_to_frame2": True,
        "exact_frame1_files_present": exact_exists,
        "inputs": {
            "observed_state": str(Path(args.observed_state).resolve()),
            "prediction_npz": str(Path(args.prediction_npz).resolve()),
            "pose_json": str(Path(args.pose_json).resolve()),
            "camera_info_json": str(Path(args.camera_info_json).resolve()),
            "observed_summary": str(paths["observed_summary"].resolve()),
        },
    }
    base.save_json(output_dir / "selected_frame_report.json", report)
    lines = [
        "# Selected Frame Report",
        "",
        "- selected frame: `Stage 4A-6.5p Frame1`",
        "- fallback used: `false`",
        "- selected Stage 4A-6.5ad Frame2: `false`",
        "- reason: preferred Frame1 observed_state, prediction NPZ, pose JSON, and camera info JSON were present.",
        f"- observed_state: `{report['inputs']['observed_state']}`",
        f"- prediction NPZ: `{report['inputs']['prediction_npz']}`",
        f"- pose/camera: `{report['inputs']['pose_json']}` / `{report['inputs']['camera_info_json']}`",
    ]
    base.write_text(output_dir / "selected_frame_report.md", "\n".join(lines))
    return report


def rewrite_loaded_manifest(output_dir: Path, frame_report: dict[str, Any]) -> None:
    path = output_dir / "loaded_inputs_manifest.json"
    if not path.is_file():
        return
    manifest = load_json(path)
    manifest.update(
        {
            "stage": "Stage 4A-6.5ae",
            "selected_frame": "Stage 4A-6.5p Frame1",
            "selected_frame_index": 1,
            "selected_frame_report": str((output_dir / "selected_frame_report.json").resolve()),
            "no_silent_fallback_to_frame2": True,
            "selected_stage4a65ad_frame2": False,
            "tree_root_source": "rebuilt_from_frame001_pose_json",
        }
    )
    manifest["observed_summary"] = frame_report["inputs"]["observed_summary"]
    base.save_json(path, manifest)
    lines = [
        "# Loaded Inputs Manifest",
        "",
        f"- stage: `{manifest['stage']}`",
        f"- selected frame: `{manifest['selected_frame']}`",
        f"- Stage 4A-6.5p dir: `{manifest['stage4a65p_dir']}`",
        f"- observed_state: `{manifest['observed_state']['path']}`",
        f"- prediction NPZ: `{manifest['prediction_npz']['path']}`",
        f"- pose/camera: `{manifest['pose_json']['path']}` / `{manifest['camera_info_json']['path']}`",
        f"- observed shape: `{manifest['observed_state']['shape']}`",
        f"- bounds: `{manifest['bounds']}`",
        f"- tree root/source: `{manifest['tree_root_source']}`",
        "- no Isaac startup, no new capture, no map_predict rerun.",
        "- no silent fallback to Stage 4A-6.5ad Frame2.",
    ]
    base.write_text(output_dir / "loaded_inputs_manifest.md", "\n".join(lines))


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage4a65p_dir = Path(args.stage4a65p_dir).resolve()
    stage4a65ad_dir = Path(args.stage4a65ad_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    paths = exact_frame1_paths(stage4a65p_dir)

    missing = [str(path) for key, path in paths.items() if key != "observed_summary" and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing preferred Frame1 saved inputs: {missing}")
    if not paths["observed_summary"].is_file():
        raise FileNotFoundError(f"missing preferred Frame1 observed summary: {paths['observed_summary']}")

    args.observed_state = str(paths["observed_state"] if not args.observed_state else Path(args.observed_state))
    args.prediction_npz = str(paths["prediction_npz"] if not args.prediction_npz else Path(args.prediction_npz))
    args.pose_json = str(paths["pose_json"] if not args.pose_json else Path(args.pose_json))
    args.camera_info_json = str(paths["camera_info_json"] if not args.camera_info_json else Path(args.camera_info_json))
    args.prefer_saved_stage4a65y_trees = False

    refs = build_reference_branches(stage4a65p_dir)
    patch_base_module(stage4a65p_dir, stage4a65ad_dir, refs)
    summary = base.run(args)
    frame_report = write_selected_frame_report(output_dir, args, paths)
    rewrite_loaded_manifest(output_dir, frame_report)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage4a65p_dir", default=DEFAULT_STAGE4A65P_DIR)
    parser.add_argument("--stage4a65ac_dir", default=DEFAULT_STAGE4A65AC_DIR)
    parser.add_argument("--stage4a65ad_dir", default=DEFAULT_STAGE4A65AD_DIR)
    parser.add_argument("--stage4a65y_dir", default=DEFAULT_STAGE4A65Y_DIR)
    parser.add_argument("--stage4a65z_dir", default=DEFAULT_STAGE4A65Z_DIR)
    parser.add_argument("--stage4a65z1_dir", default=DEFAULT_STAGE4A65Z1_DIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--observed_state", default="")
    parser.add_argument("--prediction_npz", default="")
    parser.add_argument("--pose_json", default="")
    parser.add_argument("--camera_info_json", default="")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--occ_threshold", type=float, default=0.5)
    parser.add_argument("--free_threshold", type=float, default=0.5)
    parser.add_argument("--lambda_sc", type=float, default=48.0)
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", choices=["reject", "crop", "allow"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--alignment_convention", default="code_consistent_v1")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--save_viz", action="store_true")
    parser.add_argument("--save_raw_tree_summaries", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
