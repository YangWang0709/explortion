#!/usr/bin/env python3
"""Stage 4A-6 validation for short SC-aware read-only map_predict rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from sim_rollout_utils import load_json, load_jsonl

FORBIDDEN_EXACT_FIELDS = {"target_lr", "target_hr", "ground_truth", "gt"}
FORBIDDEN_NPZ_FIELDS = FORBIDDEN_EXACT_FIELDS | {"policy", "optimizer", "rl", "ppo"}


def require(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise AssertionError(f"Empty {label}: {path}")
    return path


def _assert_no_forbidden_exact_fields(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN_EXACT_FIELDS.intersection(value.keys())
        if overlap:
            raise AssertionError(f"Forbidden exact fields at {path}: {sorted(overlap)}")
        for key, child in value.items():
            _assert_no_forbidden_exact_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            _assert_no_forbidden_exact_fields(child, f"{path}[{idx}]")


def _assert_npz_clean(path: Path) -> None:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
    bad = sorted(key for key in keys if key in FORBIDDEN_NPZ_FIELDS or key.startswith("target_"))
    if bad:
        raise AssertionError(f"Forbidden fields in {path}: {bad}")


def _assert_false(summary: dict[str, Any], key: str) -> None:
    if bool(summary.get(key, False)):
        raise AssertionError(f"Expected {key}=false, got {summary.get(key)!r}")


def validate_episode(episode_dir: Path) -> dict[str, Any]:
    summary_path = require(episode_dir / "episode_summary.json", "episode_summary.json")
    transitions_path = require(episode_dir / "transitions.jsonl", "transitions.jsonl")
    final_map_path = require(episode_dir / "observed_state_final.npy", "observed_state_final.npy")
    summary = load_json(summary_path)
    transitions = load_jsonl(transitions_path)

    _assert_no_forbidden_exact_fields(summary)
    for transition in transitions:
        _assert_no_forbidden_exact_fields(transition)

    steps_completed = int(summary.get("steps_completed", len(transitions)))
    if steps_completed != len(transitions):
        raise AssertionError(f"steps_completed {steps_completed} != transition count {len(transitions)}")
    if steps_completed < 3:
        raise AssertionError(f"expected at least 3 SC-aware rollout steps, got {steps_completed}")
    if str(summary.get("prediction_mode")) != "sim_dynamic":
        raise AssertionError(f"prediction_mode is not sim_dynamic: {summary.get('prediction_mode')}")
    if str(summary.get("path_cost_mode")) != "astar":
        raise AssertionError(f"path_cost_mode is not astar: {summary.get('path_cost_mode')}")
    if str(summary.get("candidate_sampling_mode")) != "reachable_frontier":
        raise AssertionError(
            f"candidate_sampling_mode is not reachable_frontier: {summary.get('candidate_sampling_mode')}"
        )

    ratios_after = [float(t["observed_ratio_after"]) for t in transitions]
    ratios_before = [float(t["observed_ratio_before"]) for t in transitions]
    for before, after in zip(ratios_before, ratios_after):
        if after + 1e-12 < before:
            raise AssertionError(f"observed_ratio decreased within step: {before} -> {after}")
    if any(b + 1e-12 < a for a, b in zip(ratios_after, ratios_after[1:])):
        raise AssertionError(f"observed_ratio_after is not non-decreasing: {ratios_after}")

    gain_sc_values = [float(t.get("best_gain_sc", 0.0)) for t in transitions]
    if not any(value > 0.0 for value in gain_sc_values):
        raise AssertionError("No step has nonzero selected gain_sc")
    if not any(int(t.get("candidates_with_gain_sc_positive", 0)) > 0 for t in transitions):
        raise AssertionError("No step has candidates_with_gain_sc_positive > 0")

    for transition in transitions:
        step = int(transition["step"])
        step_npz = require(episode_dir / f"step_{step:03d}.npz", f"step_{step:03d}.npz")
        observed_step = require(episode_dir / f"observed_state_step{step:03d}.npy", f"observed_state_step{step:03d}.npy")
        prediction_dir = require(episode_dir / f"prediction_step{step:03d}", f"prediction_step{step:03d}")
        require(prediction_dir / "global_prediction_layer.npz", f"prediction_step{step:03d}/global_prediction_layer.npz")
        require(prediction_dir / "local_prediction.npz", f"prediction_step{step:03d}/local_prediction.npz")
        require(prediction_dir / "sscnet_input_debug.npz", f"prediction_step{step:03d}/sscnet_input_debug.npz")
        require(
            prediction_dir / "prediction_alignment_summary.json",
            f"prediction_step{step:03d}/prediction_alignment_summary.json",
        )
        _assert_npz_clean(step_npz)
        _assert_npz_clean(prediction_dir / "global_prediction_layer.npz")
        _assert_npz_clean(prediction_dir / "local_prediction.npz")
        _assert_npz_clean(prediction_dir / "sscnet_input_debug.npz")
        observed_arr = np.load(observed_step)
        if not set(np.unique(observed_arr).tolist()).issubset({-1, 0, 1}):
            raise AssertionError(f"{observed_step} contains invalid observed_state values")

        gain_exp = float(transition.get("best_gain_exp", 0.0))
        gain_sc = float(transition.get("best_gain_sc", 0.0))
        gain_hybrid = float(transition.get("best_gain_hybrid", 0.0))
        if not np.isclose(gain_hybrid, gain_exp + gain_sc, atol=1e-5):
            raise AssertionError(
                f"step {step} violates gain_hybrid=gain_exp+gain_sc: {gain_hybrid} != {gain_exp}+{gain_sc}"
            )
        if transition.get("observed_state_hash_before_prediction") != transition.get(
            "observed_state_hash_after_prediction"
        ):
            raise AssertionError(f"observed_state hash changed during prediction at step {step}")
        if bool(transition.get("observed_state_prediction_modified", False)):
            raise AssertionError(f"observed_state_prediction_modified true at step {step}")
        for key in (
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_used_for_a_star",
            "prediction_blocks_rays",
            "prediction_writeback",
        ):
            if bool(transition.get(key, False)):
                raise AssertionError(f"{key} true at step {step}")
        with np.load(step_npz, allow_pickle=False) as data:
            if str(np.asarray(data["prediction_mode"]).item()) != "sim_dynamic":
                raise AssertionError(f"step npz prediction_mode is not sim_dynamic at step {step}")
            if bool(np.asarray(data["prediction_used_for_traversability"]).item()):
                raise AssertionError(f"step npz prediction_used_for_traversability true at step {step}")
            if bool(np.asarray(data["prediction_used_for_collision"]).item()):
                raise AssertionError(f"step npz prediction_used_for_collision true at step {step}")
            if bool(np.asarray(data["prediction_used_for_a_star"]).item()):
                raise AssertionError(f"step npz prediction_used_for_a_star true at step {step}")

    final_state = np.load(final_map_path)
    if not set(np.unique(final_state).tolist()).issubset({-1, 0, 1}):
        raise AssertionError(f"observed_state_final contains invalid values: {np.unique(final_state).tolist()}")

    for key in (
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_used_for_a_star",
        "prediction_blocks_rays",
        "prediction_writeback",
        "rl_optimizer_training_run",
        "rl_or_ppo_training",
        "optimizer_step",
        "behavior_cloning_training",
        "imitation_learning_training",
        "sscnet_training",
        "checkpoint_modified",
    ):
        _assert_false(summary, key)
    if not bool(summary.get("strict_no_prediction_write", False)):
        raise AssertionError("summary strict_no_prediction_write is not true")
    if not bool(summary.get("model_loaded_once", False)):
        raise AssertionError("summary model_loaded_once is not true")

    return {
        "steps_completed": steps_completed,
        "observed_ratio_start": float(summary["observed_ratio_start"]),
        "observed_ratio_end": float(summary["observed_ratio_end"]),
        "gain_sc_nonzero": True,
        "done_reason": summary.get("done_reason"),
    }


def validate_comparison(comparison_dir: Path) -> dict[str, Any]:
    summary_path = require(comparison_dir / "comparison_summary.json", "comparison_summary.json")
    require(comparison_dir / "comparison_summary.md", "comparison_summary.md")
    require(comparison_dir / "observed_ratio_comparison.png", "observed_ratio_comparison.png")
    require(comparison_dir / "best_score_comparison.png", "best_score_comparison.png")
    require(comparison_dir / "path_topdown_comparison.png", "path_topdown_comparison.png")
    require(comparison_dir / "gain_comparison.png", "gain_comparison.png")
    summary = load_json(summary_path)
    _assert_no_forbidden_exact_fields(summary)
    if int(summary.get("compared_steps", 0)) < 3:
        raise AssertionError(f"expected comparison over at least 3 steps, got {summary.get('compared_steps')}")
    return {
        "compared_steps": int(summary["compared_steps"]),
        "changed_actions": int(summary["number_of_changed_selected_actions"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-6 SC-aware rollout outputs.")
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--comparison_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_result = validate_episode(Path(args.episode_dir).resolve())
    comparison_result = validate_comparison(Path(args.comparison_dir).resolve())
    print("Stage 4A-6 SC-aware rollout test passed.")
    print(f"steps_completed: {episode_result['steps_completed']}")
    print(f"done_reason: {episode_result['done_reason']}")
    print(
        "observed_ratio: "
        f"{episode_result['observed_ratio_start']:.6f} -> {episode_result['observed_ratio_end']:.6f}"
    )
    print("observed_ratio_non_decreasing: yes")
    print("gain_sc_nonzero: yes")
    print("prediction_read_only: yes")
    print("prediction_traversability_collision_astar_ray_leakage: no")
    print("rl_optimizer_bc_il_training_run: no")
    print("sscnet_training_run: no")
    print("checkpoint_modified: no")
    print(f"comparison_compared_steps: {comparison_result['compared_steps']}")
    print(f"comparison_changed_actions: {comparison_result['changed_actions']}")


if __name__ == "__main__":
    main()
