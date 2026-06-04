#!/usr/bin/env python3
"""Validate Stage 4A-6.5aa synthetic SC validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


BASE_REQUIRED = [
    "scene_metadata.json",
    "synthetic_scene_summary.json",
    "synthetic_scene_summary.md",
    "rgb_000.png",
    "depth_000.npy",
    "pose_000.json",
    "camera_info.json",
    "observed_state_synthetic_frame000.npy",
    "observed_state_summary.json",
    "scene_layout_topdown.png",
    "observed_topdown.png",
    "oracle_global_prediction_layer.npz",
    "oracle_prediction_summary.json",
    "oracle_prediction_summary.md",
    "oracle_prediction_topdown.png",
    "tree_decision_manifest.jsonl",
    "per_seed_mode_decisions.csv",
    "per_seed_mode_decisions.json",
    "per_seed_mode_decisions.md",
    "branch_direction_classification.csv",
    "branch_direction_classification.json",
    "branch_direction_summary.md",
    "oracle_vs_measured_comparison.json",
    "oracle_vs_measured_comparison.md",
    "map_predict_vs_oracle_comparison.json",
    "map_predict_vs_oracle_comparison.md",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.md",
    "synthetic_sc_validation_summary.json",
    "synthetic_sc_validation_summary.md",
    "recommended_next_faithful_step.md",
    "measured_vs_oracle_selected_branches_topdown.png",
    "oracle_prediction_overlay_topdown.png",
    "hidden_region_prediction_counts.png",
    "branch_value_components.png",
    "selected_branch_by_seed_topdown.png",
]


PROHIBITED = [
    "transitions.jsonl",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
    "frame002*",
    "frame003*",
    "step_*.npz",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def require(path: Path) -> None:
    assert path.is_file(), f"missing required file: {path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_min_seeds", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    assert output_dir.is_dir(), f"output dir does not exist: {output_dir}"
    for name in BASE_REQUIRED:
        require(output_dir / name)

    observed = np.load(output_dir / "observed_state_synthetic_frame000.npy")
    assert observed.ndim == 3
    assert int(np.count_nonzero(observed == -1)) > 0, "observed_state has no UNKNOWN"
    assert int(np.count_nonzero(observed == 0)) > 0, "observed_state has no FREE"
    assert int(np.count_nonzero(observed == 1)) > 0, "observed_state has no OCCUPIED"

    with np.load(output_dir / "oracle_global_prediction_layer.npz", allow_pickle=False) as data:
        for key in (
            "global_pred_class",
            "global_confidence",
            "global_occupied_prob",
            "global_free_prob",
            "global_prediction_valid",
        ):
            assert key in data.files, f"oracle prediction missing {key}"
        assert bool(np.asarray(data["oracle_prediction"]).item()) is True
        assert bool(np.asarray(data["diagnostic_only"]).item()) is True
        assert bool(np.asarray(data["not_map_predict"]).item()) is True
        assert bool(np.asarray(data["not_ground_truth_runtime_planning"]).item()) is True
        assert bool(np.asarray(data["prediction_writeback"]).item()) is False
        assert bool(np.asarray(data["prediction_used_for_traversability"]).item()) is False
        assert bool(np.asarray(data["prediction_used_for_collision"]).item()) is False
        assert bool(np.asarray(data["prediction_blocks_rays"]).item()) is False
        assert int(np.count_nonzero(data["global_prediction_valid"])) > 0

    manifest = read_jsonl(output_dir / "tree_decision_manifest.jsonl")
    seeds = {int(row["seed"]) for row in manifest if row.get("status") == "completed"}
    assert len(seeds) >= int(args.expected_min_seeds), f"expected at least {args.expected_min_seeds} seeds, got {len(seeds)}"

    decisions = read_json(output_dir / "per_seed_mode_decisions.json")
    modes = {row["mode"] for row in decisions}
    assert "measured_only" in modes
    assert "oracle_source_occ_free_over_cost" in modes
    assert "oracle_decoupled_source_minmax" in modes
    assert len([row for row in decisions if row["mode"] == "measured_only"]) >= int(args.expected_min_seeds)

    branch = read_json(output_dir / "branch_direction_classification.json")
    assert branch, "branch direction classification is empty"
    oracle_vs_measured = read_json(output_dir / "oracle_vs_measured_comparison.json")
    assert "per_seed_mode" in oracle_vs_measured

    map_summary = output_dir / "map_predict_summary.json"
    map_skip = output_dir / "map_predict_skipped_or_failed_reason.md"
    assert map_summary.is_file() or map_skip.is_file(), "map_predict summary or skipped reason is required"
    if map_summary.is_file():
        require(output_dir / "map_predict_summary.md")
        require(output_dir / "map_predict_overlay_topdown.png")
        require(output_dir / "map_predict_prediction_overlay_topdown.png")

    depth_files = sorted(output_dir.glob("depth_*.npy"))
    rgb_files = sorted(output_dir.glob("rgb_*.png"))
    pose_files = sorted(output_dir.glob("pose_*.json"))
    assert [path.name for path in depth_files] == ["depth_000.npy"], f"expected exactly one depth frame, got {depth_files}"
    assert [path.name for path in rgb_files] == ["rgb_000.png"], f"expected exactly one rgb frame, got {rgb_files}"
    assert [path.name for path in pose_files] == ["pose_000.json"], f"expected exactly one pose frame, got {pose_files}"

    for pattern in PROHIBITED:
        found = list(output_dir.rglob(pattern))
        assert not found, f"prohibited rollout/two-frame artifacts found for {pattern}: {found[:5]}"

    summary = read_json(output_dir / "synthetic_sc_validation_summary.json")
    assert summary["frames_captured"] == 1
    assert summary["selected_action_executed"] is False
    assert summary["selected_action_execution_count"] == 0
    assert summary["two_frame_runtime"] is False
    assert summary["rollout"] is False
    assert summary["online_open_ended_loop"] is False
    assert summary["prediction_writeback"] is False
    assert summary["prediction_used_for_traversability_collision_ray_blocking"] is False
    assert summary["target_or_ground_truth_used_for_planning_scoring"] is False
    assert summary["training_rl_ppo_bc_il"] is False
    assert summary["coverage_improvement_claimed"] is False
    assert summary["answers"]["runtime_smoke_supported_now"] is False
    assert summary["answers"]["rollout_supported_now"] is False

    print("Stage 4A-6.5aa synthetic SC validation outputs passed validation.")
    print(f"output_dir={output_dir}")
    print(f"seeds_completed={len(seeds)} decision_rows={len(decisions)}")
    print(f"map_predict={'available' if map_summary.is_file() else 'skipped_or_failed'}")


if __name__ == "__main__":
    main()
