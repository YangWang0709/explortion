#!/usr/bin/env python3
"""Stage 4A-3.6 smoke validator for A* simulator expert outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from astar_planner import FREE, OCCUPIED, UNKNOWN, build_traversability_grid, summarize_traversability

FORBIDDEN_EXACT_FIELDS = {"target_lr", "target_hr", "ground_truth", "gt"}
MEDIUM_OBSERVED = Path("/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_smoke/observed_state_final.npy")


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


def validate_medium_traversability() -> dict[str, Any]:
    observed_path = require_file(MEDIUM_OBSERVED)
    observed = np.load(observed_path)
    assert observed.shape == (120, 120, 30), observed.shape
    assert set(np.unique(observed).tolist()).issubset({int(UNKNOWN), int(FREE), int(OCCUPIED)})
    trav = build_traversability_grid(observed, voxel_size=0.1, robot_height_m=1.2)
    summary = summarize_traversability(trav)
    assert summary["traversable_count"] > 0, summary
    return summary


def validate_one_step(one_step_dir: Path) -> dict[str, Any]:
    decision_path = require_file(one_step_dir / "expert_step_decision.json")
    npz_path = require_file(one_step_dir / "expert_step_decision.npz")
    jsonl_path = require_file(one_step_dir / "expert_step_candidates.jsonl")

    decision = load_json(decision_path)
    assert_no_forbidden_exact_fields(decision)
    candidates = load_jsonl(jsonl_path)
    for record in candidates:
        assert_no_forbidden_exact_fields(record)

    diagnostics = decision["diagnostics"]
    best = decision["best_candidate"]
    assert decision["prediction_mode"] == "empty"
    assert decision["path_cost_mode"] == "astar"
    assert diagnostics["path_cost_mode"] == "astar"
    if "candidate_sampling_mode" in diagnostics:
        assert diagnostics["candidate_sampling_mode"] in {"frontier", "reachable_frontier"}, diagnostics
    assert int(diagnostics["traversable_count"]) > 0, diagnostics
    assert int(diagnostics["reachable_candidates"]) >= 1, diagnostics
    assert int(diagnostics["unreachable_candidates"]) >= 0, diagnostics
    if diagnostics.get("candidate_sampling_mode") == "reachable_frontier":
        assert int(diagnostics["reachable_component_count"]) > 0, diagnostics
        assert int(diagnostics["reachable_frontier_adjacent_count"]) >= 0, diagnostics
        assert diagnostics["candidate_source"] in {"reachable_frontier", "reachable_free_fallback"}, diagnostics
        assert int(diagnostics["unreachable_candidates"]) < 52, diagnostics
    assert best["valid"] is True, best
    assert best["astar_reachable"] is True, best
    assert float(best["gain_sc"]) == 0.0, best
    assert float(best["gain_hybrid"]) == float(best["gain_exp"]), best
    assert float(best["path_cost"]) > 0.0, best
    assert float(best["astar_path_length_m"]) >= 0.0, best

    with np.load(npz_path, allow_pickle=False) as data:
        required = {
            "candidate_features",
            "feature_names",
            "valid_mask",
            "expert_action",
            "expert_scores",
            "path_cost_mode",
            "astar_reachable",
            "astar_path_length_m",
            "astar_num_expanded",
            "best_astar_path_xy",
        }
        assert required.issubset(set(data.files)), sorted(data.files)
        assert str(data["path_cost_mode"]) == "astar"
        feature_names = [str(v) for v in data["feature_names"]]
        for name in ("astar_reachable", "astar_path_length_m", "astar_num_expanded"):
            assert name in feature_names, feature_names
        features = data["candidate_features"]
        assert features.ndim == 2
        assert features.shape[0] >= 1
        assert np.isfinite(features).all()
        assert np.allclose(features[:, feature_names.index("gain_sc")], 0.0)
        assert int(data["expert_action"]) == 0
        assert bool(data["valid_mask"][0])
        assert data["best_astar_path_xy"].ndim == 2
        assert data["best_astar_path_xy"].shape[1] == 2
        if "candidate_sampling_mode" in data.files:
            assert str(data["candidate_sampling_mode"]) in {"frontier", "reachable_frontier"}
        assert FORBIDDEN_EXACT_FIELDS.isdisjoint(set(data.files))

    return {
        "reachable_candidates": int(diagnostics["reachable_candidates"]),
        "unreachable_candidates": int(diagnostics["unreachable_candidates"]),
        "candidate_sampling_mode": str(diagnostics.get("candidate_sampling_mode", "unknown")),
        "candidate_source": str(diagnostics.get("candidate_source", "unknown")),
        "reachable_component_count": int(diagnostics.get("reachable_component_count") or 0),
        "reachable_frontier_adjacent_count": int(diagnostics.get("reachable_frontier_adjacent_count") or 0),
        "best_score": float(best["final_score"]),
        "best_path_cost": float(best["path_cost"]),
        "best_path_length": float(best["astar_path_length_m"]),
    }


def validate_rollout_episode(episode_dir: Path) -> dict[str, Any] | None:
    if not episode_dir.exists():
        return None
    summary = load_json(require_file(episode_dir / "episode_summary.json"))
    transitions = load_jsonl(require_file(episode_dir / "transitions.jsonl"))
    final_map_path = require_file(episode_dir / "observed_state_final.npy")
    assert len(transitions) >= 2, f"expected at least 2 rollout steps, got {len(transitions)}"
    if any(t.get("candidate_sampling_mode") == "reachable_frontier" for t in transitions):
        assert len(transitions) >= 5, f"reachable rollout regressed below previous 5 steps: {len(transitions)}"
    assert_no_forbidden_exact_fields(summary)
    for transition in transitions:
        assert_no_forbidden_exact_fields(transition)

    assert summary["prediction_mode"] == "empty"
    assert summary["prediction_layer"] == "EmptyPredictionLayer"
    assert summary["path_cost_mode"] == "astar"
    if "candidate_sampling_mode" in summary:
        assert summary["candidate_sampling_mode"] in {"frontier", "reachable_frontier", "auto"}, summary
    assert summary["leakage_checks"]["prediction_wrote_observed_map"] is False
    assert summary["leakage_checks"]["a_star_planner"] is True
    assert summary["leakage_checks"]["optimizer_step"] is False
    assert summary["leakage_checks"]["rl_or_ppo_training"] is False
    assert summary["leakage_checks"]["behavior_cloning_training"] is False
    assert summary["leakage_checks"]["imitation_learning_training"] is False

    ratios = [float(t["observed_ratio_after"]) for t in transitions]
    assert all(b + 1e-12 >= a for a, b in zip(ratios, ratios[1:])), ratios
    reachable = []
    component_counts = []
    reachable_frontier_counts = []
    path_costs = []
    for transition in transitions:
        assert transition["prediction_mode"] == "empty"
        assert transition["path_cost_mode"] == "astar"
        assert float(transition["best_gain_sc"]) == 0.0
        assert transition["leakage_checks"]["prediction_wrote_observed_map"] is False
        assert transition["leakage_checks"]["a_star_planner"] is True
        assert transition["leakage_checks"]["optimizer_step"] is False
        assert transition["leakage_checks"]["rl_or_ppo_training"] is False
        assert int(transition.get("reachable_candidates", 0)) >= 1
        assert float(transition["best_path_cost"]) > 0.0
        if "candidate_sampling_mode" in transition:
            assert transition["candidate_sampling_mode"] in {"frontier", "reachable_frontier"}, transition
        if transition.get("candidate_sampling_mode") == "reachable_frontier":
            assert int(transition.get("reachable_component_count", 0)) > 0, transition
            assert int(transition.get("reachable_frontier_adjacent_count", 0)) >= 0, transition
        reachable.append(int(transition.get("reachable_candidates", 0)))
        component_counts.append(int(transition.get("reachable_component_count", 0)))
        reachable_frontier_counts.append(int(transition.get("reachable_frontier_adjacent_count", 0)))
        path_costs.append(float(transition["best_path_cost"]))

        step_npz = require_file(episode_dir / f"step_{int(transition['step']):03d}.npz")
        with np.load(step_npz, allow_pickle=False) as data:
            assert str(data["path_cost_mode"]) == "astar"
            assert bool(data["prediction_wrote_observed_map"]) is False
            assert bool(data["rl_or_ppo_training"]) is False
            assert bool(data["optimizer_step"]) is False
            assert float(data["best_gain_sc"]) == 0.0
            assert int(data["reachable_candidates"]) >= 1
            if "reachable_component_count" in data.files:
                assert int(data["reachable_component_count"]) >= 0
            assert FORBIDDEN_EXACT_FIELDS.isdisjoint(set(data.files))

    final_state = np.load(final_map_path)
    assert set(np.unique(final_state).tolist()).issubset({int(UNKNOWN), int(FREE), int(OCCUPIED)})

    return {
        "steps": len(transitions),
        "observed_ratio_start": float(summary["observed_ratio_start"]),
        "observed_ratio_end": float(summary["observed_ratio_end"]),
        "done_reason": summary["done_reason"],
        "average_reachable_candidates": float(np.mean(reachable)),
        "average_reachable_component_count": float(np.mean(component_counts)) if component_counts else 0.0,
        "average_reachable_frontier_adjacent_count": float(np.mean(reachable_frontier_counts))
        if reachable_frontier_counts
        else 0.0,
        "average_best_path_cost": float(np.mean(path_costs)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 4A-3.6 simulator A* expert outputs.")
    parser.add_argument(
        "--one_step_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_scene_expert_step_astar_smoke",
    )
    parser.add_argument(
        "--episode_dir",
        default="/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_astar_empty_pred/episodes/medium_three_rooms_astar_empty_pred_000",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trav_summary = validate_medium_traversability()
    one_step = validate_one_step(Path(args.one_step_dir))
    rollout = validate_rollout_episode(Path(args.episode_dir))

    print("Stage 4A-3.6 simulator A* smoke test passed.")
    print(
        "medium_traversability: "
        f"traversable={trav_summary['traversable_count']} "
        f"blocked={trav_summary['blocked_count']} "
        f"unknown={trav_summary['unknown_count']}"
    )
    print(
        "one_step_astar: "
        f"sampling={one_step['candidate_sampling_mode']} "
        f"source={one_step['candidate_source']} "
        f"reachable={one_step['reachable_candidates']} "
        f"unreachable={one_step['unreachable_candidates']} "
        f"component={one_step['reachable_component_count']} "
        f"reachable_frontier={one_step['reachable_frontier_adjacent_count']} "
        f"best_score={one_step['best_score']:.6f} "
        f"best_path_cost={one_step['best_path_cost']:.6f}"
    )
    if rollout is None:
        print(f"rollout_astar: skipped, episode_dir not found: {Path(args.episode_dir)}")
    else:
        print(
            "rollout_astar: "
            f"steps={rollout['steps']} done_reason={rollout['done_reason']} "
            f"observed_ratio={rollout['observed_ratio_start']:.6f}->{rollout['observed_ratio_end']:.6f} "
            f"avg_reachable={rollout['average_reachable_candidates']:.3f} "
            f"avg_component={rollout['average_reachable_component_count']:.3f} "
            f"avg_reachable_frontier={rollout['average_reachable_frontier_adjacent_count']:.3f} "
            f"avg_path_cost={rollout['average_best_path_cost']:.6f}"
        )
    print("observed_ratio_non_decreasing: yes")
    print("gain_sc_empty_prediction: zero")
    print("prediction_writes_observed_map: no")
    print("target_or_ground_truth_fields: none")
    print("rl_optimizer_bc_il_training_run: no")


if __name__ == "__main__":
    main()
