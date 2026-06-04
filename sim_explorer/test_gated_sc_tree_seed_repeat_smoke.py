#!/usr/bin/env python3
"""Validate Stage 4A-6.5t alternate-tree-seed gated two-frame smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from offline_mini_rrt_tree import sha256_file


REQUIRED_OUTPUTS = [
    "frame001_rgb.png",
    "frame001_depth.npy",
    "frame001_pose.json",
    "observed_state_frame001.npy",
    "frame001_prediction/global_prediction_layer.npz",
    "frame001_measured_tree_decision.json",
    "frame001_confidence_weighted_tree_decision.json",
    "frame001_cap25_shadow_tree_decision.json",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_pose.json",
    "observed_state_frame002.npy",
    "frame002_prediction/global_prediction_layer.npz",
    "frame002_measured_tree_decision.json",
    "frame002_confidence_weighted_tree_decision.json",
    "frame002_cap25_shadow_tree_decision.json",
    "gated_sc_tree_two_frame_summary.json",
    "gated_sc_tree_two_frame_summary.md",
    "stage4a65t_alternate_seed_summary.json",
    "stage4a65t_alternate_seed_summary.md",
    "prediction_safety_checklist.json",
    "source_protection_checklist.json",
    "observed_state_hashes.json",
]
PROHIBITED_PATTERNS = [
    "frame003*",
    "transitions.jsonl",
    "step_*.npz",
    "step_topdown_*.png",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def close(a: Any, b: Any, tol: float = 1.0e-5) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def expected_effective(formula: str, row: dict[str, Any]) -> float:
    raw = float(row.get("gain_sc") or 0.0)
    conf = float(row.get("gain_conf") or 0.0)
    if formula == "confidence_weighted":
        return conf
    if formula == "cap25":
        return min(raw, 25.0)
    raise AssertionError(f"unexpected formula: {formula}")


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    for frame in (1, 2):
        depth = np.load(output_dir / f"frame{frame:03d}_depth.npy")
        observed = np.load(output_dir / f"observed_state_frame{frame:03d}.npy")
        _assert(depth.ndim == 2, f"frame{frame:03d} depth should be HxW")
        _assert(int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0))) > 0, f"frame{frame:03d} depth empty")
        _assert(observed.ndim == 3, f"frame{frame:03d} observed should be 3D")
    return {"passed": True, "required_outputs": len(REQUIRED_OUTPUTS)}


def test_t_summary(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65t_alternate_seed_summary.json")
    answers = summary["answers"]
    safety = summary["safety"]
    _assert(summary["tree_seed"] == 1, "Stage 4A-6.5t tree seed must be 1")
    _assert(summary["scene_seed"] == 0, "scene seed should remain 0")
    _assert(summary["profile_name"] == "source_like_crop_min_length_0p25_seed1", "wrong seed profile")
    _assert(summary["primary_sc_gain_formula"] == "confidence_weighted", "wrong primary formula")
    _assert(summary["shadow_sc_gain_formula"] == "cap25", "wrong shadow formula")
    _assert(answers["seed1_repeat_completed_exactly_2_frames"] is True, "seed repeat did not complete two frames")
    _assert(answers["executed_action_from_confidence_weighted"] is True, "executed action source mismatch")
    _assert(answers["prediction_completely_read_only"] is True, "prediction should be read-only")
    _assert(
        answers["prediction_not_used_for_traversability_collision_ray_blocking"] is True,
        "prediction leaked into planning safety",
    )
    _assert(answers["still_not_ready_for_rollout"] is True, "repeat smoke must not be rollout-ready")

    _assert(safety["isaac_startup_count"] == 1, "expected exactly one Isaac startup")
    _assert(int(safety["frames_captured"]) == 2, "expected exactly two frames")
    _assert(int(safety["map_predict_predictions"]) == 2, "expected exactly two map_predict calls")
    _assert(int(safety["selected_action_execution_count"]) == 1, "expected exactly one action")
    _assert(int(safety["shadow_action_execution_count"]) == 0, "shadow must not execute")
    for key in (
        "rollout",
        "online_open_ended_loop",
        "sscnet_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_blocks_rays",
        "target_lr_target_hr_ground_truth_scoring",
        "external_source_modified_or_built_by_stage",
        "coverage_improvement_claimed",
    ):
        _assert(not bool(safety.get(key, False)), f"safety flag should be false: {key}")
    _assert(not safety["prohibited_output_matches"], "prohibited outputs found in t summary")
    return {"passed": True}


def test_frame_reports_and_prediction_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65t_alternate_seed_summary.json")
    for frame in (1, 2):
        report = summary["frame_reports"][f"frame{frame:03d}"]
        for label in ("measured_only", "confidence_weighted", "cap25_shadow"):
            _assert(report[label]["built_successfully"] is True, f"frame{frame:03d} {label} tree failed")
            _assert(report["required_gain_fields_present"][label] is True, f"missing gain fields for {label}")
        stats = report["prediction_stats"]
        _assert(stats["shape_aligned_to_observed_state"] is True, f"frame{frame:03d} prediction shape mismatch")
        _assert(stats["large_dense_class_prob_saved"] is False, f"frame{frame:03d} saved dense class_prob")
        with np.load(stats["prediction_npz"], allow_pickle=False) as data:
            _assert("class_prob" not in data.files, f"frame{frame:03d} prediction NPZ contains class_prob")
        observed_path = output_dir / f"observed_state_frame{frame:03d}.npy"
        _assert(observed_path.is_file(), f"missing observed state {observed_path}")
    hashes = load_json(output_dir / "observed_state_hashes.json")
    for frame in (1, 2):
        observed_path = Path(hashes[f"frame{frame:03d}_observed_state"])
        expected_hash = hashes.get(
            f"frame{frame:03d}_sha256_after_all_tree_evals",
            hashes.get(f"frame{frame:03d}_sha256_after_prediction_and_tree"),
        )
        _assert(expected_hash, f"missing frame{frame:03d} expected observed hash")
        _assert(sha256_file(observed_path) == expected_hash, f"frame{frame:03d} observed hash changed")
    return {"passed": True}


def test_formula_identities(output_dir: Path) -> dict[str, Any]:
    checked = 0
    for frame, formula, path in [
        (1, "confidence_weighted", output_dir / "frame001_confidence_weighted_gain_cost_value_table.csv"),
        (1, "cap25", output_dir / "frame001_cap25_shadow_gain_cost_value_table.csv"),
        (2, "confidence_weighted", output_dir / "frame002_confidence_weighted_gain_cost_value_table.csv"),
        (2, "cap25", output_dir / "frame002_cap25_shadow_gain_cost_value_table.csv"),
    ]:
        _assert(path.is_file(), f"missing gain table: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("segment_id") == "root":
                    continue
                gain_exp = float(row.get("gain_exp") or 0.0)
                gain = float(row.get("gain") or 0.0)
                effective = float(row.get("effective_gain_sc") or 0.0)
                hybrid_effective = float(row.get("gain_hybrid_effective") or 0.0)
                _assert(close(effective, expected_effective(formula, row)), f"frame{frame:03d} {formula} effective")
                _assert(close(hybrid_effective, gain_exp + effective), f"frame{frame:03d} {formula} hybrid")
                _assert(close(gain, hybrid_effective), f"frame{frame:03d} {formula} gain")
                checked += 1
    _assert(checked > 0, "no formula rows checked")
    return {"passed": True, "checked_rows": checked}


def test_no_prohibited_outputs(output_dir: Path) -> dict[str, Any]:
    found: dict[str, list[str]] = {}
    for pattern in PROHIBITED_PATTERNS:
        matches = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob(pattern))
        if matches:
            found[pattern] = matches
    _assert(not found, f"prohibited outputs found: {found}")
    return {"passed": True}


def run_tests(output_dir: Path) -> dict[str, Any]:
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "t_summary": test_t_summary(output_dir),
        "frame_reports_and_prediction_safety": test_frame_reports_and_prediction_safety(output_dir),
        "formula_identities": test_formula_identities(output_dir),
        "no_prohibited_outputs": test_no_prohibited_outputs(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65t_alternate_tree_seed_gated_sc_tree_two_frame_smoke",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tests(Path(args.output_dir))
