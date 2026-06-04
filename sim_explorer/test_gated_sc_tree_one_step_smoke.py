#!/usr/bin/env python3
"""Validate Stage 4A-6.5r gated SC tree one-step smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from offline_mini_rrt_tree import sha256_file


FORMULAS = [
    "measured_only",
    "raw_count",
    "weight_0p5",
    "weight_1p0",
    "cap25",
    "cap50",
    "confidence_weighted",
    "occupied_only",
    "confidence_weighted_cap25",
]

REQUIRED_OUTPUTS = [
    "gated_sc_tree_one_step_summary.json",
    "gated_sc_tree_one_step_summary.md",
    "gated_formula_decisions.json",
    "gated_formula_decisions.csv",
    "source_protection_checklist.json",
    "source_protection_checklist.md",
    "prediction_safety_checklist.json",
    "prediction_safety_checklist.md",
    "observed_state_hashes.json",
    "recommended_next_faithful_step.md",
    "gated_formula_selected_children_topdown.png",
    "gated_formula_value_bar.png",
    "raw_count_gain_exp_vs_effective_sc.png",
]

PROHIBITED_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_*.png",
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


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    required = list(REQUIRED_OUTPUTS)
    for formula in FORMULAS:
        required.extend(
            [
                f"{formula}_tree_decision.json",
                f"{formula}_tree_segments.jsonl",
                f"{formula}_gain_cost_value_table.csv",
                f"{formula}_subsequent_best_decision.json",
                f"{formula}_mini_rrt_tree_summary.json",
            ]
        )
    for name in required:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    return {"passed": True, "required_outputs": len(required)}


def test_decisions_and_groups(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "gated_sc_tree_one_step_summary.json")
    rows = {row["formula"]: row for row in summary["formula_decisions"]}
    _assert(summary["formulas"] == FORMULAS, "formula order changed")
    _assert(summary["all_expected_selection_checks_passed"] is True, "expected selection checks failed")
    _assert(rows["measured_only"]["selected_child_id"] == "n0001", "measured selected child mismatch")
    _assert(rows["measured_only"]["selected_child_grid"] == [17, 16, 11], "measured selected grid mismatch")
    _assert(rows["raw_count"]["selected_child_id"] == "n0127", "raw_count selected child mismatch")
    _assert(rows["raw_count"]["best_descendant_id"] == "n0162", "raw_count best descendant mismatch")
    _assert(rows["weight_0p5"]["selected_child_id"] == "n0001", "weight_0p5 should return to measured")
    _assert(rows["occupied_only"]["selected_child_id"] == "n0001", "occupied_only should return to measured")
    for formula in ("weight_1p0", "cap25", "cap50", "confidence_weighted", "confidence_weighted_cap25"):
        _assert(rows[formula]["selected_child_id"] == "n0127", f"{formula} should preserve raw SC branch")
    groups = summary["formula_groups"]
    _assert("weight_0p5" in groups["returning_to_measured_selected_child"], "weight_0p5 missing from return group")
    _assert("occupied_only" in groups["returning_to_measured_selected_child"], "occupied_only missing from return group")
    _assert("confidence_weighted" in groups["preserving_raw_sc_selected_child"], "confidence formula missing from preserve group")
    return {"passed": True, "groups": groups}


def expected_effective(formula: str, row: dict[str, Any]) -> float:
    raw = float(row.get("gain_sc", 0.0) or 0.0)
    occ = float(row.get("gain_occ", 0.0) or 0.0)
    conf = float(row.get("gain_conf", 0.0) or 0.0)
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
    raise AssertionError(f"unhandled formula: {formula}")


def test_formula_identities(output_dir: Path) -> dict[str, Any]:
    checked = 0
    for formula in FORMULAS:
        with (output_dir / f"{formula}_gain_cost_value_table.csv").open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("segment_id") == "root":
                    continue
                gain_exp = float(row.get("gain_exp") or 0.0)
                gain = float(row.get("gain") or 0.0)
                effective = float(row.get("effective_gain_sc") or 0.0)
                hybrid_effective = float(row.get("gain_hybrid_effective") or 0.0)
                expected = expected_effective(formula, row)
                _assert(close(effective, expected), f"{formula} effective_gain_sc mismatch at {row.get('segment_id')}")
                _assert(
                    close(hybrid_effective, gain_exp + effective),
                    f"{formula} gain_hybrid_effective identity failed at {row.get('segment_id')}",
                )
                if formula == "measured_only":
                    _assert(close(gain, gain_exp), f"measured gain should equal gain_exp at {row.get('segment_id')}")
                else:
                    _assert(
                        close(gain, hybrid_effective),
                        f"{formula} tree utility gain should equal effective hybrid at {row.get('segment_id')}",
                    )
                checked += 1
    _assert(checked > 0, "no formula rows checked")
    return {"passed": True, "checked_rows": checked}


def test_prediction_and_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "gated_sc_tree_one_step_summary.json")
    safety = summary["safety"]
    hashes = load_json(output_dir / "observed_state_hashes.json")
    pred_stats = summary["prediction_stats"]

    _assert(pred_stats["shape_aligned_to_observed_state"] is True, "prediction shape not aligned")
    _assert(pred_stats["large_dense_class_prob_saved"] is False, "global prediction should not contain class_prob")
    with np.load(pred_stats["prediction_npz"], allow_pickle=False) as data:
        _assert("class_prob" not in data.files, "prediction NPZ contains class_prob")

    observed_path = Path(hashes["observed_state_frame002"])
    prediction_path = Path(hashes["prediction_npz"])
    _assert(sha256_file(observed_path) == hashes["observed_state_sha256_after"], "observed hash mismatch")
    _assert(sha256_file(prediction_path) == hashes["prediction_npz_sha256_after"], "prediction hash mismatch")
    _assert(hashes["observed_state_hash_unchanged"] is True, "observed_state changed")
    _assert(hashes["prediction_npz_hash_unchanged"] is True, "prediction NPZ changed")

    for key in (
        "isaac_startup",
        "rgb_depth_capture",
        "map_predict_rerun",
        "sscnet_inference_or_training",
        "two_frame",
        "selected_action_execution",
        "rollout",
        "online_open_ended_loop",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_traversability_collision",
        "prediction_blocks_rays",
        "target_lr_target_hr_ground_truth_scoring",
        "external_source_modified_or_built",
        "coverage_improvement_claimed",
    ):
        _assert(not bool(safety.get(key, False)), f"safety flag should be false: {key}")
    _assert(safety["offline_saved_frame_only"] is True, "offline_saved_frame_only should be true")
    _assert(safety["checkpoint_hash_unchanged"] is True, "checkpoint hash changed")
    _assert(safety["external_source_status_unchanged"] is True, "external source status changed")
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
        "decisions_and_groups": test_decisions_and_groups(output_dir),
        "formula_identities": test_formula_identities(output_dir),
        "prediction_and_safety": test_prediction_and_safety(output_dir),
        "no_prohibited_outputs": test_no_prohibited_outputs(output_dir),
    }
    summary = {"all_passed": all(result["passed"] for result in results.values()), "tests": results}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_sc_pred_stage4a65r_gated_sc_tree_one_step_smoke",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tests(Path(args.output_dir))
