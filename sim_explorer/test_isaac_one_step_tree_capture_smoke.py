#!/usr/bin/env python3
"""Validate Stage 4A-6.5m Isaac one-frame capture tree smoke outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from offline_mini_rrt_tree import sha256_file


CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)
ONE_STEP_BASELINE_GRID = [15, 16, 11]
REQUIRED_OUTPUTS = [
    "capture_rgb_001.png",
    "capture_depth_001.npy",
    "capture_depth_001.png",
    "capture_pose_001.json",
    "capture_camera_info.json",
    "observed_state_prior_hash.json",
    "observed_state_isaac_capture_step001.npy",
    "observed_state_capture_summary.json",
    "source_protected_tree_decision.json",
    "source_protected_tree_decision.md",
    "source_protected_tree_segments.jsonl",
    "source_protected_gain_cost_value_table.csv",
    "source_protected_sampled_nodes.csv",
    "source_protected_rejected_samples.csv",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "comparison_to_saved_tree_smoke.json",
    "comparison_to_saved_tree_smoke.md",
    "isaac_one_step_tree_capture_summary.json",
    "isaac_one_step_tree_capture_summary.md",
    "recommended_next_faithful_step.md",
    "observed_capture_topdown.png",
    "tree_capture_topdown.png",
    "selected_branch_capture_topdown.png",
    "saved_vs_capture_tree_decision_topdown.png",
    "gain_cost_value_capture_scatter.png",
]
PROHIBITED_ROLLOUT_OUTPUTS = [
    "transitions.jsonl",
    "step_001.npz",
    "observed_ratio_curve.png",
    "rollout_topdown_path.png",
    "rollout_index.html",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _git_status_short(path: Path) -> str:
    if not path.exists():
        return "missing"
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(path),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip()}"
    return completed.stdout.strip()


def _same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def _scan_map_predict_artifacts(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ("*map_predict*", "*prediction*.npz", "*class_prob*", "*logits*.npy"):
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    depth = np.load(output_dir / "capture_depth_001.npy")
    positive = depth[np.isfinite(depth) & (depth > 0.0)]
    _assert(depth.ndim == 2, f"depth should be HxW, got {depth.shape}")
    _assert(positive.size > 0, "depth has no finite positive values")
    rgb = Image.open(output_dir / "capture_rgb_001.png")
    _assert(rgb.size[0] > 0 and rgb.size[1] > 0, "RGB image has invalid dimensions")
    return {"passed": True, "required_outputs": len(REQUIRED_OUTPUTS), "depth_shape": list(depth.shape)}


def test_profile_decision_and_comparison(output_dir: Path) -> dict[str, Any]:
    checklist = _load_json(output_dir / "source_protection_checklist.json")
    decision = _load_json(output_dir / "source_protected_tree_decision.json")
    comparison = _load_json(output_dir / "comparison_to_saved_tree_smoke.json")
    summary = _load_json(output_dir / "isaac_one_step_tree_capture_summary.json")

    crop = checklist["mechanisms"]["crop_min_length_min_path_length"]
    _assert(crop["active"] is True, "crop_min_length should be active")
    _assert(abs(float(crop["value_m"]) - 0.25) <= 1.0e-9, f"wrong crop value: {crop}")
    _assert(
        checklist["mechanisms"]["density_limiting_max_density_range"]["active"] is False,
        "density limiting should be inactive",
    )
    _assert(checklist["mechanisms"]["continuous_yaw"]["active"] is True, "continuous yaw should be active")

    selected = decision["selected_child"]
    best = decision["best_descendant"]
    _assert(selected.get("segment_id") != "n0140", f"selected child is still n0140: {selected}")
    _assert(not _same_grid(selected.get("end_grid"), ONE_STEP_BASELINE_GRID), "selected child equals one-step baseline")
    nonlocal_branch = bool(
        float(decision.get("selected_child_distance_from_root_m") or 0.0) >= 0.5
        or float(decision.get("best_descendant_distance_from_root_m") or 0.0) >= 1.0
    )
    _assert(nonlocal_branch, "selected child / best descendant is not nonlocal")

    judge = comparison["judgement"]
    _assert(judge["prediction_used"] is False, "prediction should be absent")
    _assert(judge["map_predict_used"] is False, "map_predict should be absent")
    _assert(judge["moved_off_old_short_edge_n0140"] is True, "decision did not move off n0140")
    _assert(judge["nonlocal_branch_found"] is True, "comparison did not mark a nonlocal branch")
    _assert(summary["answers"]["ready_for_rollout"] is False, "summary should not mark rollout ready")
    return {
        "passed": True,
        "selected_child": selected.get("segment_id"),
        "selected_grid": selected.get("end_grid"),
        "best_descendant": best.get("segment_id"),
        "best_grid": best.get("end_grid"),
        "exact_match": judge["exact_match_with_stage4a65l_saved_map"],
    }


def test_hashes_and_safety(output_dir: Path) -> dict[str, Any]:
    prior_hash = _load_json(output_dir / "observed_state_prior_hash.json")
    summary = _load_json(output_dir / "isaac_one_step_tree_capture_summary.json")
    _assert(prior_hash["prior_hash_unchanged"] is True, "prior observed_state hash changed")
    prior_path = Path(prior_hash["prior_observed_state"])
    _assert(prior_path.is_file(), f"prior observed_state missing: {prior_path}")
    _assert(sha256_file(prior_path) == prior_hash["prior_sha256_after"], "current prior hash mismatch")

    new_observed = Path(prior_hash["new_observed_state"])
    _assert(new_observed.is_file(), f"new observed_state missing: {new_observed}")
    _assert(new_observed.parent == output_dir.resolve(), "new observed_state is not under output dir")

    safety = summary["safety"]
    _assert(safety["isaac_startup"] is True, "Isaac startup flag should be true")
    for key in (
        "rollout",
        "selected_action_execution",
        "online_multi_step_loop",
        "map_predict_rerun",
        "sscnet_inference",
        "sscnet_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "target_lr_target_hr_ground_truth_scoring",
        "external_source_modified_or_built_by_stage",
    ):
        _assert(not bool(safety.get(key, False)), f"safety flag should be false: {key}")

    checkpoint_hash = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    if checkpoint_hash is not None:
        _assert(safety["checkpoint_sha256_before"] == safety["checkpoint_sha256_after"], "checkpoint summary hash changed")
        _assert(safety["checkpoint_sha256_after"] == checkpoint_hash, "checkpoint current hash mismatch")

    external_status = _git_status_short(EXTERNAL_SOURCE_DIR)
    _assert(external_status == safety["external_source_git_status_after"], "external source status changed after summary")
    _assert(external_status == "", f"external source git status not clean: {external_status}")

    for name in PROHIBITED_ROLLOUT_OUTPUTS:
        _assert(not (output_dir / name).exists(), f"prohibited rollout-like output exists: {name}")
    _assert(not list(output_dir.glob("step_topdown_*.png")), "prohibited step_topdown outputs exist")
    map_predict_artifacts = _scan_map_predict_artifacts(output_dir)
    _assert(not map_predict_artifacts, f"map_predict artifacts found: {map_predict_artifacts[:5]}")
    return {
        "passed": True,
        "prior_hash": prior_hash["prior_sha256_after"],
        "new_observed_hash": prior_hash["new_observed_state_sha256"],
        "external_source_status": external_status,
    }


def run_tests(output_dir: Path, reference_tree_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    _assert(reference_tree_dir.resolve().exists(), f"missing reference tree dir: {reference_tree_dir}")
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "profile_decision_and_comparison": test_profile_decision_and_comparison(output_dir),
        "hashes_and_safety": test_hashes_and_safety(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_tree_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    run_tests(Path(parsed.output_dir), Path(parsed.reference_tree_dir))
