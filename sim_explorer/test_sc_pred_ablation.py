#!/usr/bin/env python3
"""Validate Stage 4A-6.1 analysis and ablation outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from sim_paper_expert import SimCandidateView, compute_cost_and_score
from sim_rollout_utils import load_json, load_jsonl


FORBIDDEN_ARRAY_FIELDS = {
    "target_lr",
    "target_hr",
    "target",
    "targets",
    "gt",
    "ground_truth",
    "ground_truth_map",
}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _completed_records(ablation_dir: Path) -> list[dict[str, Any]]:
    manifest = ablation_dir / "ablation_manifest.jsonl"
    _assert(manifest.exists(), f"missing ablation manifest: {manifest}")
    return [record for record in load_jsonl(manifest) if str(record.get("status")) == "ok"]


def _assert_non_decreasing_observed_ratio(transitions: list[dict[str, Any]], episode_dir: Path) -> None:
    ratios = [float(t["observed_ratio_after"]) for t in transitions]
    prev = float(transitions[0].get("observed_ratio_before", 0.0)) if transitions else 0.0
    for idx, ratio in enumerate(ratios):
        _assert(ratio + 1e-12 >= prev, f"{episode_dir} observed_ratio decreased at step {idx}: {prev} -> {ratio}")
        prev = ratio


def _assert_safety(summary: dict[str, Any], episode_dir: Path) -> None:
    false_keys = [
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
        "checkpoint_modified",
    ]
    for key in false_keys:
        _assert(not bool(summary.get(key, False)), f"{episode_dir} safety flag {key} is true")
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
        _assert(not bool(leakage.get(key, False)), f"{episode_dir} leakage flag {key} is true")


def _assert_npz_has_no_forbidden_fields(episode_dir: Path) -> None:
    for npz_path in sorted(episode_dir.glob("step_*.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            fields = set(str(name) for name in data.files)
            bad = sorted(fields & FORBIDDEN_ARRAY_FIELDS)
            _assert(not bad, f"{npz_path} contains forbidden array fields: {bad}")


def _assert_weighted_formula(transitions: list[dict[str, Any]], episode_dir: Path) -> None:
    for transition in transitions:
        gain_exp = float(transition.get("best_gain_exp", transition.get("gain_exp", 0.0)))
        gain_sc = float(transition.get("best_gain_sc", transition.get("gain_sc", 0.0)))
        weight = float(transition.get("sc_gain_weight", 1.0))
        cap_value = float(transition.get("sc_gain_cap_value", -1.0))
        effective = min(gain_sc, cap_value) if cap_value >= 0.0 else gain_sc
        expected_weighted = weight * effective
        expected_hybrid = gain_exp + expected_weighted
        actual_weighted = float(transition.get("best_weighted_gain_sc", transition.get("weighted_gain_sc", expected_weighted)))
        actual_hybrid = float(
            transition.get("best_gain_hybrid_weighted", transition.get("gain_hybrid_weighted", expected_hybrid))
        )
        _assert(
            abs(actual_weighted - expected_weighted) <= 1e-4,
            f"{episode_dir} step {transition.get('step')} weighted_gain_sc {actual_weighted} != {expected_weighted}",
        )
        _assert(
            abs(actual_hybrid - expected_hybrid) <= 1e-4,
            f"{episode_dir} step {transition.get('step')} gain_hybrid_weighted {actual_hybrid} != {expected_hybrid}",
        )
        if str(transition.get("score_gain_mode", "hybrid_raw")) == "hybrid_weighted":
            path_cost = max(float(transition.get("best_path_cost", transition.get("path_cost", 0.0))), 1e-6)
            expected_score = expected_hybrid / path_cost
            actual_score = float(transition.get("best_score", transition.get("final_score", 0.0)))
            _assert(
                abs(actual_score - expected_score) <= 1e-3,
                f"{episode_dir} step {transition.get('step')} weighted final_score {actual_score} != {expected_score}",
            )


def _assert_default_identity() -> None:
    candidate = SimCandidateView(
        id=0,
        grid_position=(1, 1, 1),
        world_position=(0.15, 0.15, 0.15),
        yaw=0.0,
    )
    candidate.gain_exp = 7.0
    candidate.gain_sc = 5.0
    candidate.gain_hybrid = 12.0
    compute_cost_and_score(
        candidate,
        current_grid=(1, 1, 1),
        current_yaw=0.0,
        gain_mode="hybrid",
        score_gain_mode="hybrid_raw",
        sc_gain_weight=1.0,
        sc_gain_cap=None,
        path_cost_mode="euclidean",
    )
    _assert(abs(candidate.gain_hybrid_weighted - candidate.gain_hybrid) <= 1e-9, "default weighted hybrid differs")
    _assert(abs(candidate.utility_hybrid_weighted - candidate.utility_hybrid) <= 1e-9, "default weighted utility differs")
    _assert(abs(candidate.final_score - candidate.utility_hybrid) <= 1e-9, "hybrid_raw default final score changed")


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    analysis_dir = Path(args.analysis_dir).resolve()
    ablation_dir = Path(args.ablation_dir).resolve()
    summary_dir = Path(args.summary_dir).resolve()

    analysis_summary = analysis_dir / "analysis_summary.json"
    _assert(analysis_summary.exists(), f"missing analysis_summary: {analysis_summary}")
    summary_json = summary_dir / "ablation_summary.json"
    _assert(summary_json.exists(), f"missing ablation_summary: {summary_json}")
    ablation_manifest = ablation_dir / "ablation_manifest.json"
    _assert(ablation_manifest.exists(), f"missing ablation manifest json: {ablation_manifest}")

    completed = _completed_records(ablation_dir)
    _assert(len(completed) >= 2, f"expected at least 2 completed configs, got {len(completed)}")

    for record in completed:
        episode_dir = Path(record["episode_dir"]).resolve()
        summary = load_json(episode_dir / "episode_summary.json")
        transitions = load_jsonl(episode_dir / "transitions.jsonl")
        _assert(transitions, f"{episode_dir} has no transitions")
        _assert_non_decreasing_observed_ratio(transitions, episode_dir)
        _assert_safety(summary, episode_dir)
        _assert_npz_has_no_forbidden_fields(episode_dir)
        _assert_weighted_formula(transitions, episode_dir)

    summary = load_json(summary_json)
    flags = summary.get("prediction_safety_flags", {})
    for key, value in flags.items():
        _assert(not bool(value), f"summary safety flag {key} is true")
    _assert_default_identity()

    result = {
        "analysis_summary_exists": True,
        "ablation_manifest_exists": True,
        "completed_configs": [record["config_name"] for record in completed],
        "completed_count": int(len(completed)),
        "observed_ratio_non_decreasing": True,
        "prediction_read_only_checks": True,
        "weighted_gain_formula": True,
        "default_weight_identity": True,
        "checkpoint_not_modified": True,
        "rl_training_absent": True,
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6.1 outputs.")
    parser.add_argument("--analysis_dir", required=True)
    parser.add_argument("--ablation_dir", required=True)
    parser.add_argument("--summary_dir", required=True)
    return parser.parse_args()


def main() -> None:
    result = run_tests(parse_args())
    print("Stage 4A-6.1 ablation validation passed.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
