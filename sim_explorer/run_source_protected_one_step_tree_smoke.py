#!/usr/bin/env python3
"""Stage 4A-6.5l source-protected no-prediction one-step tree smoke.

This runner wraps the saved-map mini-RRT builder with the Stage 4A-6.5k
source-like crop/min-length profile and writes a one-step tree-expert smoke
report. It does not start Isaac, run rollout, rerun map_predict, train, modify
observed_state, or modify/build external active_3d_planning source.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

from offline_mini_rrt_tree import (
    read_json,
    run as run_mini_rrt,
    save_json,
    scan_rollout_like_outputs,
    sha256_file,
    to_jsonable,
)


EPS = 1.0e-6
PROFILE_NAME = "source_like_crop_min_length_0p25"
ONE_STEP_BASELINE_GRID = [15, 16, 11]
ONE_STEP_BASELINE_WORLD = [-4.45, -4.35, 1.15]
DECOUPLED_GRID = [14, 18, 11]
DECOUPLED_WORLD = [-4.55, -4.15, 1.15]
OLD_BASELINE_CHILD_ID = "n0140"
OLD_BASELINE_GRID = [14, 13, 11]
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def euclidean(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        av = [float(v) for v in a]
        bv = [float(v) for v in b]
    except (TypeError, ValueError):
        return None
    if len(av) != len(bv):
        return None
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(av, bv))))


def close_float(a: Any, b: Any, tol: float = 1.0e-9) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def git_status_short(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=str(path),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return f"error: {exc}"
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip()}"
    return completed.stdout.strip()


def copy_if_exists(output_dir: Path, src_name: str, dst_name: str) -> str | None:
    src = output_dir / src_name
    dst = output_dir / dst_name
    if not src.is_file():
        return None
    if src.resolve() != dst.resolve():
        shutil.copyfile(src, dst)
    return str(dst)


def first_float(values: list[Any]) -> float | None:
    for value in values:
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def decision_parts(summary: dict[str, Any]) -> dict[str, Any]:
    decision = summary.get("decision", {})
    comparison = summary.get("comparison", {})
    mini = comparison.get("mini_rrt", {})
    selected = decision.get("selected_child") or mini.get("selected_child") or {}
    best = decision.get("best_descendant") or mini.get("best_descendant") or {}
    selected_distance = first_float(
        [
            mini.get("selected_child_distance_from_root_m"),
            selected.get("accumulated_cost") if selected.get("parent_id") == "root" else None,
            selected.get("segment_length_m") if selected.get("parent_id") == "root" else None,
        ]
    )
    best_distance = first_float([mini.get("best_descendant_distance_from_root_m")])
    return {
        "decision": decision,
        "selected": selected,
        "best": best,
        "selected_distance_m": selected_distance,
        "best_distance_m": best_distance,
    }


def build_run_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        case_json=args.case_json,
        episode_dir=args.episode_dir,
        observed_state="",
        pose_json="",
        camera_info="",
        episode_summary="",
        prediction_npz="",
        output_dir=args.output_dir,
        seed=args.seed,
        num_nodes=args.num_nodes,
        max_extension_m=args.max_extension_m,
        sample_mode=args.sample_mode,
        gain_mode=args.gain_mode,
        path_cost_mode=args.path_cost_mode,
        v_max=args.v_max,
        yaw_rate=1.0,
        robot_radius_m=args.robot_radius_m,
        voxel_size=args.voxel_size,
        raycast_stride=args.raycast_stride,
        num_yaw_samples=args.num_yaw_samples,
        max_ray_length_m=args.max_ray_length_m,
        tau=0.1,
        save_viz=bool(args.save_viz),
        profile=True,
        min_edge_length_m=args.min_edge_length_m,
        min_root_child_length_m=args.min_root_child_length_m,
        min_root_distance_m=args.min_root_distance_m,
        crop_min_length_m=args.crop_min_length_m,
        short_edge_policy=args.short_edge_policy,
        density_radius_m=args.density_radius_m,
        max_nodes_per_density_radius=args.max_nodes_per_density_radius,
        variant_name=args.variant_name or PROFILE_NAME,
    )


def load_reference_bundle(args: argparse.Namespace) -> dict[str, Any]:
    reference_variant_dir = Path(args.reference_variant_dir).resolve()
    baseline_case_dir = Path(args.baseline_case_dir).resolve()
    reference_summary = load_json(reference_variant_dir / "mini_rrt_tree_summary.json")
    reference_decision = load_json(reference_variant_dir / "subsequent_best_decision.json")
    parent = reference_variant_dir.parent
    baseline_allow = load_json(parent / "baseline_allow" / "subsequent_best_decision.json")
    baseline_allow_summary = load_json(parent / "baseline_allow" / "mini_rrt_tree_summary.json")
    one_step = load_json(baseline_case_dir / "one_step_comparison.json")
    return {
        "reference_variant_dir": str(reference_variant_dir),
        "reference_summary": reference_summary,
        "reference_decision": reference_decision,
        "baseline_allow_dir": str(parent / "baseline_allow"),
        "baseline_allow_decision": baseline_allow,
        "baseline_allow_summary": baseline_allow_summary,
        "baseline_case_dir": str(baseline_case_dir),
        "one_step_comparison": one_step,
    }


def make_protection_checklist(summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    params = summary.get("parameters", {})
    crop_value = float(params.get("crop_min_length_m", args.crop_min_length_m))
    density_radius = float(params.get("density_radius_m", args.density_radius_m))
    density_max = int(params.get("max_nodes_per_density_radius", args.max_nodes_per_density_radius))
    yaw_samples = int(params.get("num_yaw_samples", args.num_yaw_samples))
    short_edge_policy = str(params.get("short_edge_policy", args.short_edge_policy))
    return {
        "profile_name": str(args.variant_name or PROFILE_NAME),
        "profile_parameters": {
            "short_edge_policy": short_edge_policy,
            "crop_min_length_m": crop_value,
            "min_edge_length_m": float(params.get("min_edge_length_m", args.min_edge_length_m)),
            "min_root_child_length_m": float(
                params.get("min_root_child_length_m", args.min_root_child_length_m)
            ),
            "min_root_distance_m": float(params.get("min_root_distance_m", args.min_root_distance_m)),
            "density_radius_m": density_radius,
            "max_nodes_per_density_radius": density_max,
            "num_nodes": int(params.get("num_nodes", args.num_nodes)),
            "max_extension_m": float(params.get("max_extension_m", args.max_extension_m)),
            "sample_mode": str(params.get("sample_mode", args.sample_mode)),
            "gain_mode": str(params.get("gain_mode", args.gain_mode)),
            "path_cost_mode": str(params.get("path_cost_mode", args.path_cost_mode)),
            "v_max": float(params.get("v_max", args.v_max)),
            "robot_radius_m": float(params.get("robot_radius_m", args.robot_radius_m)),
            "voxel_size": float(params.get("voxel_size", args.voxel_size)),
            "raycast_stride": int(params.get("raycast_stride", args.raycast_stride)),
            "num_yaw_samples": yaw_samples,
            "max_ray_length_m": float(params.get("max_ray_length_m", args.max_ray_length_m)),
            "seed": int(params.get("seed", args.seed)),
        },
        "mechanisms": {
            "crop_min_length_min_path_length": {
                "implemented": True,
                "active": short_edge_policy == "crop" and crop_value > 0.0,
                "value_m": crop_value,
                "source_relation": "source-like crop/min-length protection",
            },
            "density_limiting_max_density_range": {
                "implemented_in_offline_mini_rrt_tree": True,
                "active": density_radius > 0.0 and density_max > 0,
                "density_radius_m": density_radius,
                "max_nodes_per_density_radius": density_max,
                "reason": (
                    "inactive in this profile because Stage 4A-6.5k density_limited "
                    "was too restrictive at radius 0.25 / max nodes 1"
                ),
            },
            "continuous_yaw": {
                "implemented_approximation": True,
                "active": yaw_samples > 0,
                "num_yaw_samples": yaw_samples,
                "source_relation": "fixed-sample approximation of ContinuousYawPlanningEvaluator",
            },
            "root_rewiring_reinsert": {
                "full_implementation": False,
                "hook_checklist_present": True,
                "active": False,
                "reason": "root changes require a multi-step planner; not testable in one saved one-step smoke",
            },
            "optional_parent_visible_clearing": {
                "source_evidence": "optional, not proven mandatory",
                "active": False,
                "reason": "no mandatory config evidence; do not invent dedup behavior",
            },
            "root_visible_filtering_near_root_discount": {
                "source_evidence": "not found as mandatory",
                "active": False,
                "reason": "do not add non-source mechanism",
            },
        },
    }


def write_protection_checklist_md(path: Path, checklist: dict[str, Any]) -> None:
    mech = checklist["mechanisms"]
    lines = [
        "# Source Protection Checklist",
        "",
        f"- profile: `{checklist['profile_name']}`",
        f"- crop_min_length / min_path_length: implemented `{mech['crop_min_length_min_path_length']['implemented']}`, active `{mech['crop_min_length_min_path_length']['active']}`, value `{mech['crop_min_length_min_path_length']['value_m']}` m.",
        f"- density limiting / max_density_range: implemented `{mech['density_limiting_max_density_range']['implemented_in_offline_mini_rrt_tree']}`, active `{mech['density_limiting_max_density_range']['active']}`; {mech['density_limiting_max_density_range']['reason']}.",
        f"- continuous yaw: approximation implemented `{mech['continuous_yaw']['implemented_approximation']}`, active `{mech['continuous_yaw']['active']}`, samples `{mech['continuous_yaw']['num_yaw_samples']}`.",
        f"- root rewiring / reinsert: full implementation `{mech['root_rewiring_reinsert']['full_implementation']}`, hook/checklist `{mech['root_rewiring_reinsert']['hook_checklist_present']}`, active `{mech['root_rewiring_reinsert']['active']}`; {mech['root_rewiring_reinsert']['reason']}.",
        f"- optional parent visible clearing: active `{mech['optional_parent_visible_clearing']['active']}`; {mech['optional_parent_visible_clearing']['reason']}.",
        f"- root-visible filtering / near-root discount: active `{mech['root_visible_filtering_near_root_discount']['active']}`; {mech['root_visible_filtering_near_root_discount']['reason']}.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def make_comparison(summary: dict[str, Any], references: dict[str, Any]) -> dict[str, Any]:
    parts = decision_parts(summary)
    selected = parts["selected"]
    best = parts["best"]
    ref_parts = decision_parts(references.get("reference_summary", {}))
    ref_selected = references.get("reference_decision", {}).get("selected_child") or ref_parts["selected"]
    ref_best = references.get("reference_decision", {}).get("best_descendant") or ref_parts["best"]
    one_step = references.get("one_step_comparison", {})
    one_baseline = (one_step.get("baseline") or {}).get("best_candidate") or {}
    decoupled = (one_step.get("decoupled_sc") or {}).get("best_candidate") or {}
    baseline_allow = references.get("baseline_allow_decision", {})

    selected_grid = selected.get("end_grid")
    selected_world = selected.get("end_world")
    best_grid = best.get("end_grid")
    best_world = best.get("end_world")
    ref_selected_grid = ref_selected.get("end_grid")
    ref_selected_world = ref_selected.get("end_world")
    ref_best_grid = ref_best.get("end_grid")
    ref_best_world = ref_best.get("end_world")
    selected_distance = parts["selected_distance_m"]
    best_distance = parts["best_distance_m"]

    exact = bool(
        selected.get("segment_id") == ref_selected.get("segment_id")
        and best.get("segment_id") == ref_best.get("segment_id")
        and same_grid(selected_grid, ref_selected_grid)
        and same_grid(best_grid, ref_best_grid)
        and close_float(selected_distance, ref_parts["selected_distance_m"])
        and close_float(best_distance, ref_parts["best_distance_m"])
        and close_float(best.get("accumulated_gain"), ref_best.get("accumulated_gain"))
        and close_float(best.get("accumulated_cost"), ref_best.get("accumulated_cost"))
    )
    selected_world_distance = euclidean(selected_world, ref_selected_world)
    best_world_distance = euclidean(best_world, ref_best_world)
    spatially_close = bool(
        (selected_world_distance is not None and selected_world_distance <= 0.15)
        and (best_world_distance is not None and best_world_distance <= 0.15)
    )
    selected_avoids_n0140 = bool(
        selected.get("segment_id") != OLD_BASELINE_CHILD_ID and not same_grid(selected_grid, OLD_BASELINE_GRID)
    )
    differs_one_step = not same_grid(selected_grid, one_baseline.get("grid_position") or ONE_STEP_BASELINE_GRID)
    differs_decoupled = not same_grid(selected_grid, decoupled.get("grid_position") or DECOUPLED_GRID)
    nonlocal_branch = bool(
        (selected_distance is not None and selected_distance >= 0.5)
        or (best_distance is not None and best_distance >= 1.0)
    )
    prediction_used = bool(
        summary.get("inputs", {}).get("prediction_npz")
        or summary.get("parameters", {}).get("gain_mode") in {"hybrid", "sc"}
    )
    return {
        "references": {
            "one_step_baseline": {
                "grid": one_baseline.get("grid_position") or ONE_STEP_BASELINE_GRID,
                "world": one_baseline.get("world_position") or ONE_STEP_BASELINE_WORLD,
            },
            "decoupled": {
                "grid": decoupled.get("grid_position") or DECOUPLED_GRID,
                "world": decoupled.get("world_position") or DECOUPLED_WORLD,
            },
            "old_mini_rrt_baseline_allow": {
                "selected_child_id": baseline_allow.get("selected_child_id") or OLD_BASELINE_CHILD_ID,
                "selected_child": baseline_allow.get("selected_child"),
                "expected_grid": OLD_BASELINE_GRID,
            },
            "stage4a65k_crop_min_length_0p25": {
                "selected_child": ref_selected,
                "selected_child_distance_from_root_m": ref_parts["selected_distance_m"],
                "best_descendant": ref_best,
                "best_descendant_distance_from_root_m": ref_parts["best_distance_m"],
                "reference_variant_dir": references.get("reference_variant_dir"),
            },
        },
        "source_protected_one_step_tree_smoke": {
            "selected_child": selected,
            "selected_child_distance_from_root_m": selected_distance,
            "best_descendant": best,
            "best_descendant_distance_from_root_m": best_distance,
            "value": selected.get("value"),
            "accumulated_gain": best.get("accumulated_gain"),
            "accumulated_cost": best.get("accumulated_cost"),
            "accepted_nodes": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
            "rejected_samples": summary.get("tree", {}).get("rejected_samples"),
            "profile_name": summary.get("variant_name"),
        },
        "judgement": {
            "matches_stage4a65k_crop_variant_exactly": exact,
            "spatially_close_to_stage4a65k_crop_variant": spatially_close,
            "selected_child_world_distance_to_crop_reference_m": selected_world_distance,
            "best_descendant_world_distance_to_crop_reference_m": best_world_distance,
            "avoids_old_short_edge_winner_n0140": selected_avoids_n0140,
            "differs_from_one_step_baseline_grid": differs_one_step,
            "differs_from_decoupled_grid": differs_decoupled,
            "nonlocal_branch_found": nonlocal_branch,
            "observed_state_hash_unchanged": bool(summary.get("map", {}).get("observed_state_hash_unchanged")),
            "prediction_used": prediction_used,
            "map_predict_used": False,
            "measured_only_gain_exp": bool(summary.get("parameters", {}).get("gain_mode") == "exp" and not prediction_used),
        },
    }


def write_comparison_md(path: Path, comparison: dict[str, Any]) -> None:
    smoke = comparison["source_protected_one_step_tree_smoke"]
    judge = comparison["judgement"]
    refs = comparison["references"]
    lines = [
        "# Tree Vs Baseline Comparison",
        "",
        f"- one-step baseline grid/world: `{refs['one_step_baseline']['grid']}` / `{refs['one_step_baseline']['world']}`",
        f"- decoupled grid/world: `{refs['decoupled']['grid']}` / `{refs['decoupled']['world']}`",
        f"- old mini-RRT baseline_allow child: `{refs['old_mini_rrt_baseline_allow']['selected_child_id']}`",
        f"- Stage 4A-6.5k crop reference child/best: `{refs['stage4a65k_crop_min_length_0p25']['selected_child'].get('segment_id')}` / `{refs['stage4a65k_crop_min_length_0p25']['best_descendant'].get('segment_id')}`",
        "",
        "## New Smoke",
        f"- selected child: `{smoke['selected_child'].get('segment_id')}` grid `{smoke['selected_child'].get('end_grid')}` world `{smoke['selected_child'].get('end_world')}`",
        f"- selected child distance: `{smoke['selected_child_distance_from_root_m']}` m",
        f"- best descendant: `{smoke['best_descendant'].get('segment_id')}` grid `{smoke['best_descendant'].get('end_grid')}` world `{smoke['best_descendant'].get('end_world')}`",
        f"- best descendant distance: `{smoke['best_descendant_distance_from_root_m']}` m",
        f"- accumulated gain/cost: `{smoke['accumulated_gain']}` / `{smoke['accumulated_cost']}`",
        "",
        "## Judgement",
        f"- exact match to Stage 4A-6.5k crop variant: `{judge['matches_stage4a65k_crop_variant_exactly']}`",
        f"- spatially close to Stage 4A-6.5k crop variant: `{judge['spatially_close_to_stage4a65k_crop_variant']}`",
        f"- avoids `n0140`: `{judge['avoids_old_short_edge_winner_n0140']}`",
        f"- differs from one-step baseline: `{judge['differs_from_one_step_baseline_grid']}`",
        f"- differs from decoupled: `{judge['differs_from_decoupled_grid']}`",
        f"- nonlocal branch found: `{judge['nonlocal_branch_found']}`",
        f"- measured-only gain_exp: `{judge['measured_only_gain_exp']}`",
        f"- prediction/map_predict used: `{judge['prediction_used']}` / `{judge['map_predict_used']}`",
    ]
    write_text(path, "\n".join(lines) + "\n")


def make_decision_payload(
    summary: dict[str, Any],
    checklist: dict[str, Any],
    comparison: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    parts = decision_parts(summary)
    return {
        "stage": "Stage 4A-6.5l source-protected no-prediction one-step tree smoke",
        "profile_name": args.variant_name or PROFILE_NAME,
        "profile": checklist["profile_parameters"],
        "selected_child": parts["selected"],
        "selected_child_distance_from_root_m": parts["selected_distance_m"],
        "best_descendant": parts["best"],
        "best_descendant_distance_from_root_m": parts["best_distance_m"],
        "value": parts["selected"].get("value"),
        "accumulated_gain": parts["best"].get("accumulated_gain"),
        "accumulated_cost": parts["best"].get("accumulated_cost"),
        "accepted_nodes": summary.get("tree", {}).get("accepted_nodes_excluding_root"),
        "rejected_samples": summary.get("tree", {}).get("rejected_samples"),
        "raw_decision": parts["decision"],
        "comparison_judgement": comparison["judgement"],
        "observed_state": {
            "path": summary.get("inputs", {}).get("observed_state"),
            "sha256_before": summary.get("map", {}).get("observed_state_sha256_before"),
            "sha256_after": summary.get("map", {}).get("observed_state_sha256_after"),
            "unchanged": summary.get("map", {}).get("observed_state_hash_unchanged"),
        },
        "safety": summary.get("safety", {}),
    }


def write_decision_md(path: Path, payload: dict[str, Any]) -> None:
    selected = payload["selected_child"]
    best = payload["best_descendant"]
    lines = [
        "# Source-Protected Tree Decision",
        "",
        f"- profile: `{payload['profile_name']}`",
        f"- selected child: `{selected.get('segment_id')}`",
        f"- selected child grid/world: `{selected.get('end_grid')}` / `{selected.get('end_world')}`",
        f"- selected child distance: `{payload['selected_child_distance_from_root_m']}` m",
        f"- best descendant: `{best.get('segment_id')}`",
        f"- best descendant grid/world: `{best.get('end_grid')}` / `{best.get('end_world')}`",
        f"- best descendant distance: `{payload['best_descendant_distance_from_root_m']}` m",
        f"- value: `{payload['value']}`",
        f"- accumulated gain/cost: `{payload['accumulated_gain']}` / `{payload['accumulated_cost']}`",
        f"- accepted/rejected: `{payload['accepted_nodes']}` / `{payload['rejected_samples']}`",
        f"- observed_state unchanged: `{payload['observed_state']['unchanged']}`",
        f"- prediction used: `{payload['comparison_judgement']['prediction_used']}`",
    ]
    write_text(path, "\n".join(lines) + "\n")


def make_smoke_summary(
    summary: dict[str, Any],
    checklist: dict[str, Any],
    comparison: dict[str, Any],
    generated: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    judge = comparison["judgement"]
    mechanisms = checklist["mechanisms"]
    safety = summary.get("safety", {})
    ready_one_step_capture = bool(
        summary.get("tree", {}).get("built_successfully")
        and (judge["matches_stage4a65k_crop_variant_exactly"] or judge["spatially_close_to_stage4a65k_crop_variant"])
        and judge["avoids_old_short_edge_winner_n0140"]
        and judge["nonlocal_branch_found"]
        and judge["measured_only_gain_exp"]
        and not judge["prediction_used"]
        and bool(summary.get("map", {}).get("observed_state_hash_unchanged"))
    )
    return {
        "stage": "Stage 4A-6.5l source-protected no-prediction one-step tree smoke",
        "output_dir": str(Path(args.output_dir).resolve()),
        "source_protected_one_step_tree_smoke_ran": bool(summary.get("tree", {}).get("built_successfully")),
        "active_source_like_protections": [
            name for name, item in mechanisms.items() if bool(item.get("active", False))
        ],
        "present_but_inactive_protections": [
            "density_limiting_max_density_range",
            "root_rewiring_reinsert_hook",
        ],
        "intentionally_not_implemented_or_inactive": [
            "optional_parent_visible_clearing",
            "root_visible_filtering_near_root_discount",
            "full_root_rewiring_reinsert",
        ],
        "answers": {
            "one_step_tree_smoke_ran": bool(summary.get("tree", {}).get("built_successfully")),
            "source_like_protection_profile_enabled": True,
            "reproduces_stage4a65k_crop_min_length_0p25": judge["matches_stage4a65k_crop_variant_exactly"],
            "spatially_close_if_not_exact": judge["spatially_close_to_stage4a65k_crop_variant"],
            "leaves_old_short_edge_winner_n0140": judge["avoids_old_short_edge_winner_n0140"],
            "differs_from_one_step_baseline": judge["differs_from_one_step_baseline_grid"],
            "differs_from_decoupled": judge["differs_from_decoupled_grid"],
            "finds_nonlocal_branch": judge["nonlocal_branch_found"],
            "measured_only_traversability_and_gain_exp": judge["measured_only_gain_exp"],
            "prediction_or_map_predict_used": judge["prediction_used"] or judge["map_predict_used"],
            "ready_for_no_prediction_isaac_one_step_capture_smoke": ready_one_step_capture,
            "ready_for_rollout": False,
        },
        "tree_decision": comparison["source_protected_one_step_tree_smoke"],
        "comparison_judgement": judge,
        "safety": {
            **safety,
            "rollout_like_outputs_created": bool(generated.get("rollout_like_outputs_created")),
            "map_predict_artifacts_created": bool(generated.get("map_predict_artifacts_created")),
            "checkpoint_sha256_before": generated.get("checkpoint_sha256_before"),
            "checkpoint_sha256_after": generated.get("checkpoint_sha256_after"),
            "checkpoint_hash_unchanged": generated.get("checkpoint_hash_unchanged"),
            "external_source_git_status_before": generated.get("external_source_git_status_before"),
            "external_source_git_status_after": generated.get("external_source_git_status_after"),
        },
        "limitation": (
            "one saved observed_state, no Isaac startup, no online root rewiring/reinsert, "
            "no dynamic map update, and no rollout or coverage claim"
        ),
        "recommended_next_faithful_step": (
            "no-prediction Isaac one-step capture + tree decision smoke"
            if ready_one_step_capture
            else "reproducibility/debug seed/sampling/API mismatch before Isaac one-step capture"
        ),
    }


def write_smoke_summary_md(path: Path, payload: dict[str, Any]) -> None:
    answers = payload["answers"]
    decision = payload["tree_decision"]
    judge = payload["comparison_judgement"]
    lines = [
        "# Stage 4A-6.5l Source-Protected One-Step Tree Smoke Summary",
        "",
        f"1. source-protected one-step tree smoke ran? `{answers['one_step_tree_smoke_ran']}`.",
        f"2. active source-like protections: `{payload['active_source_like_protections']}`.",
        f"3. present but inactive protections: `{payload['present_but_inactive_protections']}`.",
        f"4. intentionally not implemented or inactive: `{payload['intentionally_not_implemented_or_inactive']}`.",
        f"5. reproduced Stage 4A-6.5k `crop_min_length_0p25`? `{answers['reproduces_stage4a65k_crop_min_length_0p25']}`.",
        f"6. if not exact, spatially close? `{answers['spatially_close_if_not_exact']}`.",
        f"7. left `n0140`? `{answers['leaves_old_short_edge_winner_n0140']}`.",
        f"8. differs from one-step baseline / decoupled? `{answers['differs_from_one_step_baseline']}` / `{answers['differs_from_decoupled']}`.",
        f"9. nonlocal branch found? `{answers['finds_nonlocal_branch']}`.",
        f"10. still measured-only `gain_exp`? `{answers['measured_only_traversability_and_gain_exp']}`.",
        f"11. prediction / map_predict used? `{judge['prediction_used']}` / `{judge['map_predict_used']}`.",
        f"12. ready for no-prediction Isaac one-step capture smoke? `{answers['ready_for_no_prediction_isaac_one_step_capture_smoke']}`.",
        f"13. ready for rollout? `{answers['ready_for_rollout']}`.",
        "",
        "## Decision",
        f"- selected child: `{decision['selected_child'].get('segment_id')}` grid `{decision['selected_child'].get('end_grid')}` world `{decision['selected_child'].get('end_world')}` distance `{decision['selected_child_distance_from_root_m']}` m.",
        f"- best descendant: `{decision['best_descendant'].get('segment_id')}` grid `{decision['best_descendant'].get('end_grid')}` world `{decision['best_descendant'].get('end_world')}` distance `{decision['best_descendant_distance_from_root_m']}` m.",
        f"- value: `{decision['value']}`; accumulated gain/cost: `{decision['accumulated_gain']}` / `{decision['accumulated_cost']}`.",
        f"- accepted/rejected: `{decision['accepted_nodes']}` / `{decision['rejected_samples']}`.",
        "",
        f"Limitation: {payload['limitation']}.",
        f"Recommended next faithful step: {payload['recommended_next_faithful_step']}. Still no rollout.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def write_recommendation(path: Path, summary_payload: dict[str, Any]) -> None:
    if summary_payload["answers"]["ready_for_no_prediction_isaac_one_step_capture_smoke"]:
        next_step = "no-prediction Isaac one-step capture + tree decision smoke"
        reason = "the source-protected tree smoke reproduced the crop-min-length behavior and found a nonlocal measured-only branch"
    elif not summary_payload["comparison_judgement"]["matches_stage4a65k_crop_variant_exactly"]:
        next_step = "reproducibility/debug seed/sampling/API mismatch"
        reason = "the new one-step smoke did not exactly reproduce the Stage 4A-6.5k crop variant"
    else:
        next_step = "sampling/raycast/min-length refinement"
        reason = "the one-step smoke did not satisfy the nonlocal/safety gates"
    lines = [
        "# Recommended Next Faithful Step",
        "",
        f"- next small task: {next_step}",
        f"- reason: {reason}",
        "- still not next: rollout, RL/PPO/BC/IL, map_predict tree integration, prediction writeback, observed_map prediction fusion, target/ground-truth scoring, checkpoint changes, or external source build.",
    ]
    write_text(path, "\n".join(lines) + "\n")


def create_alias_outputs(output_dir: Path) -> dict[str, Any]:
    aliases = {
        "source_protected_tree_segments.jsonl": copy_if_exists(
            output_dir, "mini_rrt_tree_segments.jsonl", "source_protected_tree_segments.jsonl"
        ),
        "source_protected_gain_cost_value_table.csv": copy_if_exists(
            output_dir, "gain_cost_value_table.csv", "source_protected_gain_cost_value_table.csv"
        ),
        "source_protected_sampled_nodes.csv": copy_if_exists(
            output_dir, "sampled_nodes.csv", "source_protected_sampled_nodes.csv"
        ),
        "source_protected_rejected_samples.csv": copy_if_exists(
            output_dir, "rejected_samples.csv", "source_protected_rejected_samples.csv"
        ),
        "source_protected_tree_topdown.png": copy_if_exists(
            output_dir, "mini_rrt_tree_topdown.png", "source_protected_tree_topdown.png"
        ),
        "tree_vs_baseline_topdown.png": copy_if_exists(
            output_dir, "mini_rrt_tree_topdown.png", "tree_vs_baseline_topdown.png"
        ),
        "gain_cost_value_scatter.png": copy_if_exists(
            output_dir, "gain_cost_scatter.png", "gain_cost_value_scatter.png"
        ),
    }
    if (output_dir / "selected_branch_topdown.png").is_file():
        aliases["selected_branch_topdown.png"] = str(output_dir / "selected_branch_topdown.png")
    return aliases


def scan_map_predict_artifacts(output_dir: Path) -> list[str]:
    patterns = ["*map_predict*", "*prediction*.npz", "*class_prob*", "*logits*.npy"]
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash_before = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_status_before = git_status_short(EXTERNAL_SOURCE_DIR)

    mini_summary = run_mini_rrt(build_run_args(args))
    references = load_reference_bundle(args)
    checklist = make_protection_checklist(mini_summary, args)
    comparison = make_comparison(mini_summary, references)
    aliases = create_alias_outputs(output_dir)

    checkpoint_hash_after = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    external_status_after = git_status_short(EXTERNAL_SOURCE_DIR)
    rollout_like = scan_rollout_like_outputs(output_dir)
    map_predict_artifacts = scan_map_predict_artifacts(output_dir)
    generated = {
        "aliases": aliases,
        "rollout_like_outputs_created": rollout_like,
        "map_predict_artifacts_created": map_predict_artifacts,
        "checkpoint_sha256_before": checkpoint_hash_before,
        "checkpoint_sha256_after": checkpoint_hash_after,
        "checkpoint_hash_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "external_source_git_status_before": external_status_before,
        "external_source_git_status_after": external_status_after,
    }

    decision_payload = make_decision_payload(mini_summary, checklist, comparison, args)
    smoke_summary = make_smoke_summary(mini_summary, checklist, comparison, generated, args)

    save_json(output_dir / "source_protected_tree_decision.json", decision_payload)
    write_decision_md(output_dir / "source_protected_tree_decision.md", decision_payload)
    save_json(output_dir / "source_protection_checklist.json", checklist)
    write_protection_checklist_md(output_dir / "source_protection_checklist.md", checklist)
    save_json(output_dir / "tree_vs_baseline_comparison.json", comparison)
    write_comparison_md(output_dir / "tree_vs_baseline_comparison.md", comparison)
    save_json(output_dir / "one_step_tree_smoke_summary.json", smoke_summary)
    write_smoke_summary_md(output_dir / "one_step_tree_smoke_summary.md", smoke_summary)
    write_recommendation(output_dir / "recommended_next_faithful_step.md", smoke_summary)

    print(json.dumps(to_jsonable(smoke_summary), indent=2, sort_keys=True))
    return smoke_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_json", required=True)
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--reference_variant_dir", required=True)
    parser.add_argument("--baseline_case_dir", required=True)
    parser.add_argument("--external_inspection_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=256)
    parser.add_argument("--max_extension_m", type=float, default=0.5)
    parser.add_argument("--sample_mode", choices=["reachable_frontier", "reachable_free", "mixed"], default="mixed")
    parser.add_argument("--gain_mode", choices=["exp"], default="exp")
    parser.add_argument("--path_cost_mode", choices=["segment_time"], default="segment_time")
    parser.add_argument("--v_max", type=float, default=1.0)
    parser.add_argument("--robot_radius_m", type=float, default=0.2)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--raycast_stride", type=int, default=2)
    parser.add_argument("--num_yaw_samples", type=int, default=8)
    parser.add_argument("--max_ray_length_m", type=float, default=4.8)
    parser.add_argument("--short_edge_policy", choices=["crop"], default="crop")
    parser.add_argument("--crop_min_length_m", type=float, default=0.25)
    parser.add_argument("--min_edge_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_child_length_m", type=float, default=0.0)
    parser.add_argument("--min_root_distance_m", type=float, default=0.0)
    parser.add_argument("--density_radius_m", type=float, default=0.0)
    parser.add_argument("--max_nodes_per_density_radius", type=int, default=0)
    parser.add_argument("--variant_name", default=PROFILE_NAME)
    parser.add_argument("--save_viz", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
