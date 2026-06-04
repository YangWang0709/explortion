#!/usr/bin/env python3
"""Validate Stage 4A-6.4 SC gain gating outputs and leakage boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim_paper_expert import SimCandidateView, compute_cost_and_score, sc_gain_contribution
from sim_rollout_utils import load_json, load_jsonl

FORBIDDEN_NPZ_FIELDS = {
    "target_lr",
    "target_hr",
    "target",
    "targets",
    "gt",
    "ground_truth",
    "ground_truth_map",
    "optimizer",
    "ppo",
    "policy",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _require(path: Path, label: str) -> Path:
    _assert(path.exists(), f"missing {label}: {path}")
    if path.is_file():
        _assert(path.stat().st_size > 0, f"empty {label}: {path}")
    return path


def _assert_non_decreasing(transitions: list[dict[str, Any]], episode_dir: Path) -> None:
    prev = float(transitions[0].get("observed_ratio_before", 0.0)) if transitions else 0.0
    for transition in transitions:
        value = float(transition.get("observed_ratio_after", prev))
        _assert(value + 1.0e-12 >= prev, f"{episode_dir} observed_ratio decreased: {prev} -> {value}")
        prev = value


def _assert_false_flags(summary: dict[str, Any], episode_dir: Path) -> None:
    false_keys = [
        "checkpoint_modified",
        "prediction_writeback",
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_used_for_a_star",
        "prediction_blocks_rays",
        "prediction_used_for_candidate_reachability",
        "prediction_used_for_collision_checking",
        "prediction_used_for_a_star_traversability",
        "rl_optimizer_training_run",
        "rl_optimizer_bc_il_training_run",
        "rl_or_ppo_training",
        "optimizer_step",
        "behavior_cloning_training",
        "imitation_learning_training",
        "sscnet_training",
    ]
    for key in false_keys:
        _assert(not bool(summary.get(key, False)), f"{episode_dir} reports forbidden true flag: {key}")
    leakage = summary.get("leakage_checks", {})
    for key in (
        "target_lr_used",
        "target_hr_used",
        "scene_ground_truth_used",
        "simulator_ground_truth_used",
        "prediction_wrote_observed_map",
        "optimizer_step",
        "rl_or_ppo_training",
        "behavior_cloning_training",
        "imitation_learning_training",
    ):
        _assert(not bool(leakage.get(key, False)), f"{episode_dir} leakage flag true: {key}")


def _assert_npz_clean(episode_dir: Path) -> None:
    for npz_path in sorted(episode_dir.glob("step_*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            fields = {str(name) for name in data.files}
            bad = sorted(fields & FORBIDDEN_NPZ_FIELDS)
            _assert(not bad, f"{npz_path} contains forbidden fields: {bad}")
            _assert("best_gain_sc" in fields, f"{npz_path} missing raw best_gain_sc")
            _assert("best_effective_gain_sc" in fields, f"{npz_path} missing best_effective_gain_sc")
            _assert("effective_gain_sc" in fields, f"{npz_path} missing effective_gain_sc")


def _assert_transition_gain_formula(transitions: list[dict[str, Any]], episode_dir: Path) -> None:
    for transition in transitions:
        gain_exp = float(transition.get("best_gain_exp", 0.0))
        effective = float(transition.get("best_effective_gain_sc", transition.get("best_gain_sc", 0.0)))
        weight = float(transition.get("sc_gain_weight", 1.0))
        cap = float(transition.get("sc_gain_cap_value", -1.0))
        capped = min(effective, cap) if cap >= 0.0 else effective
        expected_weighted = weight * capped
        actual_weighted = float(transition.get("best_weighted_gain_sc", transition.get("weighted_gain_sc", 0.0)))
        _assert(
            abs(actual_weighted - expected_weighted) <= 1.0e-4,
            f"{episode_dir} step {transition.get('step')} weighted gain mismatch {actual_weighted} != {expected_weighted}",
        )
        if str(transition.get("score_gain_mode", "hybrid_raw")) == "hybrid_weighted":
            expected_hybrid = gain_exp + expected_weighted
            actual_hybrid = float(transition.get("best_gain_hybrid_weighted", 0.0))
            _assert(
                abs(actual_hybrid - expected_hybrid) <= 1.0e-4,
                f"{episode_dir} step {transition.get('step')} hybrid weighted mismatch",
            )
            path_cost = max(float(transition.get("best_path_cost", 0.0)), 1.0e-6)
            expected_score = expected_hybrid / path_cost
            actual_score = float(transition.get("best_score", 0.0))
            _assert(abs(actual_score - expected_score) <= 1.0e-3, f"{episode_dir} weighted score mismatch")


def _assert_synthetic_formulas() -> None:
    assert abs(sc_gain_contribution(formula="raw_count", occupied_prob=0.2, confidence=0.1, free_neighbor=True, occ_threshold=0.7, conf_threshold=0.9) - 1.0) <= 1e-9
    assert abs(sc_gain_contribution(formula="occupied_only", occupied_prob=0.8, confidence=0.4, free_neighbor=True, occ_threshold=0.7, conf_threshold=0.3) - 1.0) <= 1e-9
    assert sc_gain_contribution(formula="occupied_only", occupied_prob=0.65, confidence=0.4, free_neighbor=True, occ_threshold=0.7, conf_threshold=0.3) == 0.0
    assert abs(sc_gain_contribution(formula="occupied_margin", occupied_prob=0.8, confidence=0.4, free_neighbor=True, occ_threshold=0.6, conf_threshold=0.3) - 0.3) <= 1e-6
    assert abs(sc_gain_contribution(formula="confidence_weighted", occupied_prob=0.2, confidence=0.6, free_neighbor=False, occ_threshold=0.7, conf_threshold=0.5) - 0.6) <= 1e-6
    assert sc_gain_contribution(formula="confidence_weighted", occupied_prob=0.2, confidence=0.4, free_neighbor=False, occ_threshold=0.7, conf_threshold=0.5) == 0.0
    entropy_mid = sc_gain_contribution(formula="entropy_weighted", occupied_prob=0.5, confidence=0.6, free_neighbor=False, occ_threshold=0.7, conf_threshold=0.5)
    entropy_hi = sc_gain_contribution(formula="entropy_weighted", occupied_prob=0.99, confidence=0.6, free_neighbor=False, occ_threshold=0.7, conf_threshold=0.5)
    assert entropy_mid > entropy_hi
    table = {"occupied_prob_bins": [{"prob_min": 0.7, "prob_max": 0.8, "empirical_occupied_rate": 0.42, "count": 10}]}
    assert abs(sc_gain_contribution(formula="calibrated_occupied", occupied_prob=0.75, confidence=0.6, free_neighbor=False, occ_threshold=0.7, conf_threshold=0.3, calibration_table=table) - 0.42) <= 1e-9

    candidate = SimCandidateView(id=0, grid_position=(1, 1, 1), world_position=(0.15, 0.15, 0.15), yaw=0.0)
    candidate.gain_exp = 10.0
    candidate.gain_sc = 100.0
    candidate.raw_gain_sc = 100.0
    candidate.effective_gain_sc = 5.0
    candidate.gain_hybrid = 110.0
    candidate.gain_hybrid_effective = 15.0
    candidate.sc_gain_formula = "occupied_margin"
    compute_cost_and_score(
        candidate,
        current_grid=(1, 1, 1),
        current_yaw=0.0,
        score_gain_mode="hybrid_weighted",
        sc_gain_weight=2.0,
        sc_gain_cap=3.0,
        path_cost_mode="euclidean",
    )
    assert abs(candidate.weighted_gain_sc - 6.0) <= 1e-9
    assert abs(candidate.gain_hybrid_weighted - 16.0) <= 1e-9


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.gating_root).resolve()
    calibration_dir = root / "calibration"
    one_step_dir = root / "one_step"
    ablation_dir = root / "ablation"
    summary_dir = root / "summary"

    _require(calibration_dir / "calibration_table.json", "calibration table json")
    _require(calibration_dir / "calibration_table.csv", "calibration table csv")
    _require(calibration_dir / "calibration_summary.md", "calibration summary")
    _require(calibration_dir / "occupied_prob_reliability.png", "occupied probability reliability plot")
    _require(calibration_dir / "confidence_reliability.png", "confidence reliability plot")
    calibration = load_json(calibration_dir / "calibration_table.json")
    _assert("post-hoc" in str(calibration.get("future_observations_usage", "")), "calibration must mark future observations post-hoc")
    for key in ("planning_or_training_used", "prediction_writeback", "prediction_used_for_traversability", "prediction_used_for_collision", "prediction_used_for_a_star", "prediction_blocks_rays"):
        _assert(not bool(calibration.get(key, False)), f"calibration forbidden flag true: {key}")

    _require(one_step_dir, "one-step output dir")
    one_step_cases = sorted(one_step_dir.glob("*/expert_step_decision.json"))
    _assert(len(one_step_cases) >= 2, f"expected at least 2 one-step outputs, got {len(one_step_cases)}")
    for decision_path in one_step_cases:
        decision = load_json(decision_path)
        diag = decision.get("diagnostics", {})
        for key in ("prediction_used_for_traversability", "prediction_used_for_collision", "prediction_used_for_astar", "prediction_blocks_rays", "prediction_written_to_observed_state"):
            _assert(not bool(diag.get(key, False)), f"{decision_path} leakage flag true: {key}")
        if decision.get("prediction_mode") == "sim_npz":
            _assert(diag.get("alignment_convention") == "code_consistent_v1", f"{decision_path} wrong alignment")

    manifest_json = _require(ablation_dir / "ablation_manifest.json", "ablation manifest json")
    manifest_jsonl = _require(ablation_dir / "ablation_manifest.jsonl", "ablation manifest jsonl")
    manifest = load_json(manifest_json)
    records = [row for row in load_jsonl(manifest_jsonl) if str(row.get("status")) == "ok"]
    _assert(len(records) >= 2, f"expected at least 2 completed gating configs, got {len(records)}")
    _assert(manifest.get("parallel_isaac_instances") is False, "parallel Isaac instances must be false")

    for record in records:
        episode_dir = Path(record["episode_dir"]).resolve()
        summary = load_json(episode_dir / "episode_summary.json")
        transitions = load_jsonl(episode_dir / "transitions.jsonl")
        _assert(transitions, f"{episode_dir} has no transitions")
        _assert(summary.get("alignment_convention") == "code_consistent_v1", f"{episode_dir} wrong alignment convention")
        _assert_non_decreasing(transitions, episode_dir)
        _assert_false_flags(summary, episode_dir)
        _assert_npz_clean(episode_dir)
        _assert_transition_gain_formula(transitions, episode_dir)
        for transition in transitions:
            _assert("best_gain_sc" in transition, f"{episode_dir} missing raw gain_sc in transition")
            _assert("best_effective_gain_sc" in transition, f"{episode_dir} missing effective gain_sc in transition")
            _assert(not bool(transition.get("future_observations_used_for_planning", False)), "future obs used for planning")
            _assert(not bool(transition.get("future_observations_used_for_scoring", False)), "future obs used for scoring")

    _require(summary_dir / "gain_gating_summary.json", "gain gating summary json")
    _require(summary_dir / "gain_gating_summary.md", "gain gating summary md")
    _require(summary_dir / "gain_gating_table.csv", "gain gating table csv")
    summary = load_json(summary_dir / "gain_gating_summary.json")
    for key, value in summary.get("prediction_safety_flags", {}).items():
        _assert(not bool(value), f"summary safety flag true: {key}")

    _assert_synthetic_formulas()
    result = {
        "calibration_output_exists": True,
        "one_step_outputs_exist": True,
        "ablation_manifest_exists": True,
        "completed_configs": [record["config_name"] for record in records],
        "completed_count": int(len(records)),
        "observed_ratio_non_decreasing": True,
        "raw_and_effective_gain_logged": True,
        "synthetic_formulas_correct": True,
        "code_consistent_v1_alignment_used": True,
        "prediction_read_only": True,
        "prediction_writeback_false": True,
        "prediction_used_for_traversability_false": True,
        "prediction_used_for_collision_false": True,
        "prediction_used_for_astar_false": True,
        "prediction_blocked_rays_false": True,
        "future_observations_evaluation_only": True,
        "target_ground_truth_leakage_false": True,
        "rl_optimizer_training_false": True,
        "checkpoint_not_modified": True,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gating_root", required=True)
    return parser.parse_args()


def main() -> None:
    result = run_tests(parse_args())
    print("Stage 4A-6.4 SC gain gating validation passed.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
