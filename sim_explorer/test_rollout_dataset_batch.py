#!/usr/bin/env python3
"""Validate Stage 4A-4 multi-episode EmptyPredictionLayer A* rollout datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_DATASET_DIR = "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_dataset_empty_pred_astar"
FORBIDDEN_EXACT_FIELDS = {"target_lr", "target_hr", "ground_truth", "gt"}
REQUIRED_TRANSITION_FIELDS = {
    "episode_id",
    "step",
    "current_pose_world",
    "selected_next_pose_world",
    "candidate_features",
    "feature_names",
    "expert_action",
    "expert_scores",
    "valid_mask",
    "gain_exp",
    "gain_sc",
    "gain_hybrid",
    "path_cost",
    "final_score",
    "reachable_candidates",
    "unreachable_candidates",
    "reachable_component_count",
    "reachable_frontier_adjacent_count",
    "candidate_source",
    "observed_ratio_before",
    "observed_ratio_after",
    "delta_observed_ratio",
    "done",
    "done_reason",
}
REQUIRED_EPISODE_PLOTS = {
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "frontier_count_curve.png",
    "reachable_candidates_curve.png",
    "reachable_component_count_curve.png",
    "rollout_index.html",
}
REQUIRED_DATASET_OUTPUTS = {
    "manifest.jsonl",
    "dataset_summary.json",
    "dataset_summary.md",
    "rollout_dataset_index.html",
    "aggregate_observed_ratio_curve.png",
    "aggregate_observed_ratio_end_bar.png",
    "aggregate_steps_completed_bar.png",
    "aggregate_steps_hist.png",
    "aggregate_done_reasons.png",
    "aggregate_reachable_candidates_curve.png",
    "aggregate_no_valid_candidate_stats.png",
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def require_file(path: str | Path) -> Path:
    path = Path(path)
    assert path.exists(), f"missing file: {path}"
    assert path.stat().st_size > 0, f"empty file: {path}"
    return path


def assert_no_forbidden_exact_fields(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN_EXACT_FIELDS.intersection(value.keys())
        assert not overlap, f"forbidden exact fields at {path}: {sorted(overlap)}"
        for key, child in value.items():
            assert_no_forbidden_exact_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            assert_no_forbidden_exact_fields(child, f"{path}[{idx}]")


def assert_no_unknown_traversability_shortcut() -> None:
    source = Path(__file__).resolve().parent / "astar_planner.py"
    text = source.read_text(encoding="utf-8")
    assert "UNKNOWN is not" in text, "astar planner no longer documents UNKNOWN as non-traversable"
    assert "free_support = np.any(band == FREE" in text, "traversability must require observed FREE support"
    assert "traversable = free_support & ~blocked" in text, "traversability must not mark UNKNOWN traversable"


def ok_manifest_rows(manifest_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        episode_id = str(row.get("episode_id", ""))
        if episode_id:
            deduped[episode_id] = row
    return {episode_id: row for episode_id, row in deduped.items() if str(row.get("status", "ok")) == "ok"}


def failed_manifest_rows(manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        episode_id = str(row.get("episode_id", ""))
        if episode_id:
            deduped[episode_id] = row
    return [row for row in deduped.values() if str(row.get("status", "ok")) != "ok"]


def validate_transition_npz(path: Path, transition: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        files = set(data.files)
        assert FORBIDDEN_EXACT_FIELDS.isdisjoint(files), f"forbidden fields in {path}: {FORBIDDEN_EXACT_FIELDS & files}"
        required = {
            "candidate_features",
            "feature_names",
            "valid_mask",
            "expert_action",
            "expert_scores",
            "gain_exp",
            "gain_sc",
            "gain_hybrid",
            "path_cost",
            "final_score",
            "best_gain_sc",
            "prediction_wrote_observed_map",
            "rl_or_ppo_training",
            "optimizer_step",
            "path_cost_mode",
            "candidate_sampling_mode",
            "reachable_candidates",
            "unreachable_candidates",
        }
        assert required.issubset(files), f"missing npz fields in {path}: {sorted(required - files)}"
        features = data["candidate_features"]
        valid_mask = data["valid_mask"].astype(bool)
        expert_action = int(data["expert_action"])
        feature_names = [str(v) for v in data["feature_names"]]
        assert features.ndim == 2, path
        assert features.shape[0] >= 1, path
        assert features.shape[1] == len(feature_names), path
        assert np.isfinite(features).all(), path
        assert 0 <= expert_action < features.shape[0], path
        assert bool(valid_mask[expert_action]), path
        assert "gain_sc" in feature_names, feature_names
        assert np.allclose(features[:, feature_names.index("gain_sc")], 0.0), path
        assert float(data["gain_sc"]) == 0.0, path
        assert float(data["best_gain_sc"]) == 0.0, path
        assert str(data["path_cost_mode"]) == "astar", path
        assert str(data["candidate_sampling_mode"]) == "reachable_frontier", path
        assert int(data["reachable_candidates"]) >= 1, path
        assert int(data["unreachable_candidates"]) == 0, path
        assert bool(data["prediction_wrote_observed_map"]) is False, path
        assert bool(data["rl_or_ppo_training"]) is False, path
        assert bool(data["optimizer_step"]) is False, path
        assert abs(float(data["gain_sc"]) - float(transition["gain_sc"])) <= 1e-6


def validate_episode(episode_dir: Path) -> dict[str, Any]:
    summary_path = require_file(episode_dir / "episode_summary.json")
    transitions_path = require_file(episode_dir / "transitions.jsonl")
    final_map_path = require_file(episode_dir / "observed_state_final.npy")
    for name in REQUIRED_EPISODE_PLOTS:
        require_file(episode_dir / name)

    summary = load_json(summary_path)
    transitions = load_jsonl(transitions_path)
    assert len(transitions) >= 2, f"expected at least 2 transitions in {episode_dir}, got {len(transitions)}"
    assert_no_forbidden_exact_fields(summary)
    assert "start_pose" in summary, f"episode summary missing start_pose: {summary_path}"
    assert summary["prediction_mode"] == "empty"
    assert summary["prediction_layer"] == "EmptyPredictionLayer"
    assert summary["path_cost_mode"] == "astar"
    assert summary["candidate_sampling_mode"] in {"reachable_frontier", "auto"}
    assert summary["leakage_checks"]["prediction_wrote_observed_map"] is False
    assert summary["leakage_checks"]["optimizer_step"] is False
    assert summary["leakage_checks"]["rl_or_ppo_training"] is False
    assert summary["leakage_checks"]["behavior_cloning_training"] is False
    assert summary["leakage_checks"]["imitation_learning_training"] is False
    assert summary["leakage_checks"]["target_lr_used"] is False
    assert summary["leakage_checks"]["target_hr_used"] is False
    assert summary["leakage_checks"]["scene_ground_truth_used"] is False
    assert summary["leakage_checks"]["simulator_ground_truth_used"] is False

    ratios_after = []
    for idx, transition in enumerate(transitions):
        assert_no_forbidden_exact_fields(transition)
        missing = REQUIRED_TRANSITION_FIELDS - set(transition.keys())
        assert not missing, f"missing transition fields in {episode_dir}: {sorted(missing)}"
        assert transition["episode_id"] == summary["episode_id"]
        assert int(transition["step"]) == idx, f"non-sequential step in {episode_dir}: {transition['step']} != {idx}"
        assert transition["prediction_mode"] == "empty"
        assert transition["path_cost_mode"] == "astar"
        assert transition["candidate_sampling_mode"] == "reachable_frontier"
        assert transition["candidate_source"] in {"reachable_frontier", "reachable_free_fallback"}
        assert float(transition["gain_sc"]) == 0.0, transition
        assert float(transition["best_gain_sc"]) == 0.0, transition
        assert int(transition["reachable_candidates"]) >= 1, transition
        assert int(transition["unreachable_candidates"]) == 0, transition
        assert int(transition["reachable_component_count"]) > 0, transition
        assert float(transition["observed_ratio_after"]) + 1e-12 >= float(transition["observed_ratio_before"])
        ratios_after.append(float(transition["observed_ratio_after"]))
        checks = transition["leakage_checks"]
        assert checks["prediction_mode"] == "empty"
        assert checks["prediction_layer"] == "EmptyPredictionLayer"
        assert checks["prediction_wrote_observed_map"] is False
        assert checks["optimizer_step"] is False
        assert checks["rl_or_ppo_training"] is False
        assert checks["behavior_cloning_training"] is False
        assert checks["imitation_learning_training"] is False
        assert checks["target_lr_used"] is False
        assert checks["target_hr_used"] is False
        assert checks["scene_ground_truth_used"] is False
        assert checks["simulator_ground_truth_used"] is False

        step = int(transition["step"])
        validate_transition_npz(require_file(episode_dir / f"step_{step:03d}.npz"), transition)
        require_file(episode_dir / f"observed_state_step{step:03d}.npy")
        require_file(episode_dir / f"pose_{step:03d}.json")
        require_file(episode_dir / f"depth_{step:03d}.npy")
        require_file(episode_dir / f"rgb_{step:03d}.png")

    assert all(b + 1e-12 >= a for a, b in zip(ratios_after, ratios_after[1:])), ratios_after
    final_state = np.load(final_map_path)
    unique = set(np.unique(final_state).tolist())
    assert unique.issubset({-1, 0, 1}), unique
    assert -1 in unique, "final map has no UNKNOWN cells; unexpected for Stage 4A-4 medium smoke"

    return {
        "episode_id": summary["episode_id"],
        "steps": len(transitions),
        "done_reason": summary["done_reason"],
        "observed_ratio_end": float(summary["observed_ratio_end"]),
        "gain_sc_zero": True,
    }


def validate_dataset(dataset_dir: Path, min_ok_episodes: int, preferred_ok_episodes: int) -> dict[str, Any]:
    for name in REQUIRED_DATASET_OUTPUTS:
        require_file(dataset_dir / name)
    manifest_rows = load_jsonl(dataset_dir / "manifest.jsonl")
    assert manifest_rows, "manifest.jsonl is empty"
    summary = load_json(dataset_dir / "dataset_summary.json")
    assert_no_forbidden_exact_fields(summary)
    assert summary["leakage_summary"]["prediction_mode"] == "empty"
    assert summary["leakage_summary"]["prediction_wrote_observed_map"] is False
    assert summary["leakage_summary"]["rl_ppo_bc_il_training_run"] is False
    assert summary["leakage_summary"]["unknown_traversability_shortcut"] is False
    assert summary["leakage_summary"]["euclidean_fallback"] is False
    assert int(summary["gain_sc_nonzero_count"]) == 0, summary.get("validation_errors")

    ok_rows = ok_manifest_rows(manifest_rows)
    failed_rows = failed_manifest_rows(manifest_rows)
    assert len(ok_rows) >= int(min_ok_episodes), f"ok episodes {len(ok_rows)} < required {min_ok_episodes}"
    episode_results = []
    for episode_id, row in sorted(ok_rows.items()):
        episode_dir = Path(row["episode_dir"])
        episode_results.append(validate_episode(episode_dir))

    assert int(summary["ok_episodes"]) >= len(episode_results), summary
    assert int(summary["total_transitions"]) == sum(int(item["steps"]) for item in episode_results), summary
    assert_no_unknown_traversability_shortcut()

    return {
        "ok_episodes": len(episode_results),
        "preferred_ok_episodes_met": len(episode_results) >= int(preferred_ok_episodes),
        "failed_episodes": len(failed_rows),
        "failed_rows": failed_rows,
        "total_transitions": sum(int(item["steps"]) for item in episode_results),
        "done_reasons": summary["done_reason_counts"],
        "observed_ratio_end": summary["observed_ratio_end"],
        "average_reachable_candidates": summary["transition_averages"]["reachable_candidates"],
        "average_gain_sc": summary["transition_averages"]["gain_sc"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Stage 4A-4 rollout dataset.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--min_ok_episodes", type=int, default=3)
    parser.add_argument("--preferred_ok_episodes", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_dataset(Path(args.dataset_dir).resolve(), args.min_ok_episodes, args.preferred_ok_episodes)
    print("Stage 4A-4 rollout dataset batch test passed.")
    print(f"ok_episodes: {result['ok_episodes']}")
    print(f"preferred_ok_episodes_met: {result['preferred_ok_episodes_met']}")
    print(f"failed_episodes: {result['failed_episodes']}")
    print(f"total_transitions: {result['total_transitions']}")
    print(f"done_reasons: {result['done_reasons']}")
    print(f"observed_ratio_end: {result['observed_ratio_end']}")
    print(f"average_reachable_candidates: {result['average_reachable_candidates']}")
    print(f"average_gain_sc: {result['average_gain_sc']}")
    print("observed_ratio_non_decreasing: yes")
    print("gain_sc_zero: yes")
    print("leakage_checks: passed")
    print("rl_optimizer_bc_il_training_run: no")
    print("prediction_writes_observed_map: no")
    print("unknown_traversability_shortcut: no")
    print("euclidean_fallback: no")
    if result["failed_rows"]:
        print("failed_episodes_detail:")
        for row in result["failed_rows"]:
            print(f"- {row.get('episode_id')}: {row.get('error') or row.get('done_reason')}")


if __name__ == "__main__":
    main()
