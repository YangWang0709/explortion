#!/usr/bin/env python3
"""Validate Stage 4A-6.5s gated SC tree two-frame smoke outputs."""

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
    "frame001_confidence_weighted_tree_segments.jsonl",
    "frame001_cap25_shadow_tree_segments.jsonl",
    "frame002_rgb.png",
    "frame002_depth.npy",
    "frame002_pose.json",
    "observed_state_frame002.npy",
    "frame002_prediction/global_prediction_layer.npz",
    "frame002_measured_tree_decision.json",
    "frame002_confidence_weighted_tree_decision.json",
    "frame002_cap25_shadow_tree_decision.json",
    "frame002_confidence_weighted_tree_segments.jsonl",
    "frame002_cap25_shadow_tree_segments.jsonl",
    "gated_sc_tree_two_frame_summary.json",
    "gated_sc_tree_two_frame_summary.md",
    "gated_sc_tree_two_frame_decisions.csv",
    "gated_sc_tree_two_frame_decisions.json",
    "prediction_safety_checklist.json",
    "source_protection_checklist.json",
    "observed_state_hashes.json",
    "recommended_next_faithful_step.md",
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def close(a: Any, b: Any, tol: float = 1.0e-5) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def expected_effective(formula: str, row: dict[str, Any]) -> float:
    raw = float(row.get("gain_sc") or 0.0)
    occ = float(row.get("gain_occ") or 0.0)
    conf = float(row.get("gain_conf") or 0.0)
    if formula == "measured_only":
        return 0.0
    if formula in {"raw_count", "weight_1p0"}:
        return raw
    if formula == "weight_0p5":
        return 0.5 * raw
    if formula == "cap25":
        return min(raw, 25.0)
    if formula == "cap50":
        return min(raw, 50.0)
    if formula == "confidence_weighted":
        return conf
    if formula == "occupied_only":
        return occ
    if formula == "confidence_weighted_cap25":
        return min(conf, 25.0)
    raise AssertionError(f"unknown formula: {formula}")


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    for name in REQUIRED_OUTPUTS:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    for frame in (1, 2):
        depth = np.load(output_dir / f"frame{frame:03d}_depth.npy")
        _assert(depth.ndim == 2, f"frame{frame:03d} depth should be HxW")
        _assert(int(np.count_nonzero(np.isfinite(depth) & (depth > 0.0))) > 0, f"frame{frame:03d} depth empty")
        observed = np.load(output_dir / f"observed_state_frame{frame:03d}.npy")
        _assert(observed.ndim == 3, f"frame{frame:03d} observed_state should be 3D")
    return {"passed": True, "required_outputs": len(REQUIRED_OUTPUTS)}


def test_summary_answers(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "gated_sc_tree_two_frame_summary.json")
    answers = summary["answers"]
    safety = summary["safety"]
    _assert(summary["primary_sc_gain_formula"] == "confidence_weighted", "wrong primary formula")
    _assert(summary["shadow_sc_gain_formula"] == "cap25", "wrong shadow formula")
    _assert(answers["frame1_capture_observed_update_map_predict_success"] is True, "frame1 capture/update/predict failed")
    _assert(answers["frame1_measured_confidence_weighted_cap25_trees_success"] is True, "frame1 tree failed")
    _assert(answers["executed_exactly_one_confidence_weighted_selected_child_move"] is True, "move count/selection failed")
    _assert(answers["frame2_capture_observed_update_map_predict_success"] is True, "frame2 capture/update/predict failed")
    _assert(answers["frame2_confidence_weighted_changed_measured_selected_child"] is True, "frame2 primary should change measured")
    _assert(answers["frame2_cap25_shadow_matches_confidence_weighted"] is True, "frame2 cap25 should match primary")
    _assert(answers["frame2_confidence_weighted_kept_stage4a65p_r_sc_branch"] is True, "frame2 primary should keep SC branch")
    _assert(answers["prediction_read_only"] is True, "prediction read-only answer false")
    _assert(answers["prediction_used_for_traversability_collision_ray_blocking"] is False, "prediction planning leakage")
    _assert(answers["ready_for_rollout"] is False, "must not be rollout-ready")

    _assert(safety["isaac_startup"] is True, "Isaac should start once through primary runner")
    _assert(int(safety["frames_captured"]) == 2, "expected exactly two frames")
    _assert(int(safety["selected_action_execution_count"]) == 1, "expected exactly one action")
    _assert(int(safety["shadow_action_execution_count"]) == 0, "shadow must not execute action")
    _assert(int(safety["map_predict_predictions"]) == 2, "expected exactly two map_predict calls")
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
    return {"passed": True}


def test_expected_frame_decisions(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "gated_sc_tree_two_frame_summary.json")
    f1 = summary["frames"]["frame001"]
    f2 = summary["frames"]["frame002"]
    _assert(f1["measured_only"]["selected_child_id"] == "n0001", "frame1 measured selected mismatch")
    _assert(f1["confidence_weighted"]["selected_child_id"] == "n0001", "frame1 primary selected mismatch")
    _assert(f1["cap25_shadow"]["built_successfully"] is True, "frame1 shadow tree should build")
    _assert(f1["cap25_shadow"]["selected_child_id"], "frame1 shadow selected child missing")
    _assert(f2["measured_only"]["selected_child_id"] == "n0001", "frame2 measured selected mismatch")
    _assert(f2["confidence_weighted"]["selected_child_id"] == "n0127", "frame2 confidence selected mismatch")
    _assert(f2["cap25_shadow"]["selected_child_id"] == "n0127", "frame2 cap25 selected mismatch")
    _assert(f2["confidence_weighted"]["best_descendant_id"] == "n0162", "frame2 confidence descendant mismatch")
    _assert(f2["cap25_shadow"]["best_descendant_id"] == "n0162", "frame2 cap25 descendant mismatch")
    _assert(close(f2["confidence_weighted"]["accumulated_effective_gain_sc"], 31.506256222724915), "confidence effective gain mismatch")
    _assert(close(f2["cap25_shadow"]["accumulated_effective_gain_sc"], 50.0), "cap25 effective gain mismatch")
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
                expected = expected_effective(formula, row)
                _assert(close(effective, expected), f"frame{frame:03d} {formula} effective gain mismatch")
                _assert(close(hybrid_effective, gain_exp + effective), f"frame{frame:03d} {formula} effective hybrid identity")
                _assert(close(gain, hybrid_effective), f"frame{frame:03d} {formula} utility gain should use effective hybrid")
                checked += 1
    _assert(checked > 0, "no rows checked")
    return {"passed": True, "checked_rows": checked}


def test_prediction_and_hash_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "gated_sc_tree_two_frame_summary.json")
    for frame in (1, 2):
        stats = summary["frames"][f"frame{frame:03d}"]["prediction_stats"]
        _assert(stats["shape_aligned_to_observed_state"] is True, f"frame{frame:03d} prediction shape mismatch")
        _assert(stats["large_dense_class_prob_saved"] is False, f"frame{frame:03d} saved dense class_prob")
        with np.load(stats["prediction_npz"], allow_pickle=False) as data:
            _assert("class_prob" not in data.files, f"frame{frame:03d} prediction NPZ contains class_prob")
        observed_path = Path(summary["observed_state_hashes"][f"frame{frame:03d}_observed_state"])
        expected_hash = summary["observed_state_hashes"][f"frame{frame:03d}_sha256_after_all_tree_evals"]
        _assert(sha256_file(observed_path) == expected_hash, f"frame{frame:03d} observed hash changed")
    return {"passed": True}


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
        "summary_answers": test_summary_answers(output_dir),
        "expected_frame_decisions": test_expected_frame_decisions(output_dir),
        "formula_identities": test_formula_identities(output_dir),
        "prediction_and_hash_safety": test_prediction_and_hash_safety(output_dir),
        "no_prohibited_outputs": test_no_prohibited_outputs(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65s_gated_sc_tree_two_frame_smoke",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tests(Path(args.output_dir))
