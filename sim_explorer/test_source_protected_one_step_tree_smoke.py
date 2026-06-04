#!/usr/bin/env python3
"""Validate Stage 4A-6.5l source-protected one-step tree smoke outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from offline_mini_rrt_tree import sha256_file


ONE_STEP_BASELINE_GRID = [15, 16, 11]
DECOUPLED_GRID = [14, 18, 11]
EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)
CHECKPOINT_PATH = Path(
    "/home/ubuntu22/sc_explorer_ws/checkpoints/full_train/cpBest_SSCNet_NYU_full_train.pth.tar"
)
REQUIRED_OUTPUTS = [
    "source_protected_tree_decision.json",
    "source_protected_tree_decision.md",
    "source_protected_tree_segments.jsonl",
    "source_protected_gain_cost_value_table.csv",
    "source_protected_sampled_nodes.csv",
    "source_protected_rejected_samples.csv",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "tree_vs_baseline_comparison.json",
    "tree_vs_baseline_comparison.md",
    "one_step_tree_smoke_summary.json",
    "one_step_tree_smoke_summary.md",
    "recommended_next_faithful_step.md",
]
ROLLOUT_LIKE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]
MAP_PREDICT_PATTERNS = ["*map_predict*", "*prediction*.npz", "*class_prob*", "*logits*.npy"]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    try:
        return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]
    except (TypeError, ValueError):
        return False


def git_status_short(path: Path) -> str:
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


def scan_patterns(output_dir: Path, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    return {"passed": True, "required_outputs": len(REQUIRED_OUTPUTS)}


def test_profile_and_decision(output_dir: Path, reference_variant_dir: Path) -> dict[str, Any]:
    checklist = load_json(output_dir / "source_protection_checklist.json")
    decision = load_json(output_dir / "source_protected_tree_decision.json")
    comparison = load_json(output_dir / "tree_vs_baseline_comparison.json")
    summary = load_json(output_dir / "one_step_tree_smoke_summary.json")
    reference = load_json(reference_variant_dir / "subsequent_best_decision.json")

    crop = checklist["mechanisms"]["crop_min_length_min_path_length"]
    _assert(crop["active"] is True, "crop_min_length is not active")
    _assert(abs(float(crop["value_m"]) - 0.25) <= 1.0e-9, f"wrong crop_min_length_m: {crop}")
    _assert(
        checklist["mechanisms"]["density_limiting_max_density_range"]["active"] is False,
        "density limiting should be inactive in this profile",
    )
    _assert(
        checklist["mechanisms"]["continuous_yaw"]["active"] is True,
        "continuous yaw approximation should be active",
    )

    selected = decision["selected_child"]
    best = decision["best_descendant"]
    _assert(selected.get("segment_id") != "n0140", f"selected child is still n0140: {selected}")
    _assert(not same_grid(selected.get("end_grid"), ONE_STEP_BASELINE_GRID), "selected child equals one-step baseline")
    _assert(not same_grid(selected.get("end_grid"), DECOUPLED_GRID), "selected child equals decoupled")
    nonlocal_branch = bool(
        float(decision.get("selected_child_distance_from_root_m") or 0.0) >= 0.5
        or float(decision.get("best_descendant_distance_from_root_m") or 0.0) >= 1.0
    )
    _assert(nonlocal_branch, "selected child / best descendant is not nonlocal by Stage 4A-6.5k definition")
    _assert(
        same_grid(selected.get("end_grid"), reference.get("selected_child", {}).get("end_grid")),
        "selected child does not match reference crop variant grid",
    )
    _assert(
        same_grid(best.get("end_grid"), reference.get("best_descendant", {}).get("end_grid")),
        "best descendant does not match reference crop variant grid",
    )
    _assert(comparison["judgement"]["prediction_used"] is False, "prediction mode should be absent/empty")
    _assert(comparison["judgement"]["map_predict_used"] is False, "map_predict should not be used")
    _assert(summary["answers"]["ready_for_rollout"] is False, "summary should not mark rollout ready")
    return {
        "passed": True,
        "selected_child": selected.get("segment_id"),
        "selected_grid": selected.get("end_grid"),
        "best_descendant": best.get("segment_id"),
        "best_grid": best.get("end_grid"),
    }


def test_hashes_and_safety(output_dir: Path) -> dict[str, Any]:
    decision = load_json(output_dir / "source_protected_tree_decision.json")
    summary = load_json(output_dir / "one_step_tree_smoke_summary.json")
    observed = decision["observed_state"]
    observed_path = Path(observed["path"])
    _assert(observed_path.is_file(), f"observed_state path missing: {observed_path}")
    observed_hash = sha256_file(observed_path)
    _assert(observed["sha256_before"] == observed["sha256_after"] == observed_hash, "observed_state hash changed")

    checkpoint_hash = sha256_file(CHECKPOINT_PATH) if CHECKPOINT_PATH.is_file() else None
    _assert(
        summary["safety"].get("checkpoint_sha256_before") == summary["safety"].get("checkpoint_sha256_after"),
        "checkpoint hash changed in summary",
    )
    if checkpoint_hash is not None:
        _assert(summary["safety"].get("checkpoint_sha256_after") == checkpoint_hash, "checkpoint current hash mismatch")

    for key in (
        "isaac_startup",
        "rollout",
        "online_expert_loop",
        "map_predict_rerun",
        "sscnet_inference_or_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_writeback",
        "prediction_used_for_traversability_collision",
        "target_lr_target_hr_ground_truth_scoring",
        "external_source_modified_or_built",
    ):
        _assert(not bool(summary["safety"].get(key, False)), f"safety flag true: {key}")

    rollout_like = scan_patterns(output_dir, ROLLOUT_LIKE_PATTERNS)
    map_predict_artifacts = scan_patterns(output_dir, MAP_PREDICT_PATTERNS)
    _assert(not rollout_like, f"rollout-like outputs found: {rollout_like[:5]}")
    _assert(not map_predict_artifacts, f"map_predict artifacts found: {map_predict_artifacts[:5]}")

    external_status = git_status_short(EXTERNAL_SOURCE_DIR)
    _assert(external_status == "", f"external source git status not clean: {external_status}")
    _assert(
        summary["safety"].get("external_source_git_status_before") == "",
        f"external source was dirty before run: {summary['safety'].get('external_source_git_status_before')}",
    )
    _assert(
        summary["safety"].get("external_source_git_status_after") == "",
        f"external source was dirty after run: {summary['safety'].get('external_source_git_status_after')}",
    )
    return {
        "passed": True,
        "observed_state_hash": observed_hash,
        "checkpoint_hash_checked": checkpoint_hash is not None,
        "external_source_status": external_status,
    }


def run_tests(output_dir: Path, reference_variant_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    reference_variant_dir = reference_variant_dir.resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "profile_and_decision": test_profile_and_decision(output_dir, reference_variant_dir),
        "hashes_and_safety": test_hashes_and_safety(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_variant_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    run_tests(Path(parsed.output_dir), Path(parsed.reference_variant_dir))
