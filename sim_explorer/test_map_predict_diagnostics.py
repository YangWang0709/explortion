#!/usr/bin/env python3
"""Validate Stage 4A-6.2 diagnostic outputs and safety boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


SC_EPISODE = Path(
    "/home/ubuntu22/sc_explorer_ws/outputs/isaac_medium_rollout_sc_pred_dynamic_smoke/"
    "episodes/medium_three_rooms_seed0_start_room_a_sc_pred_dynamic_000"
)
NEW_SCRIPTS = [
    "diagnose_isaac_sscnet_preprocess.py",
    "diagnose_prediction_global_alignment.py",
    "evaluate_prediction_against_future_observed.py",
    "run_map_predict_alignment_variant_sweep.py",
    "summarize_map_predict_diagnostics.py",
    "test_map_predict_diagnostics.py",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(path: Path) -> Path:
    if not path.exists():
        raise AssertionError(f"missing required diagnostic output: {path}")
    if path.is_file() and path.stat().st_size <= 0:
        raise AssertionError(f"empty diagnostic output: {path}")
    return path


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(tuple(int(v) for v in arr.shape)).encode("utf-8"))
    digest.update(arr.view(np.uint8))
    return digest.hexdigest()


def validate_required_outputs(root: Path) -> None:
    required = [
        root / "preprocess_stats" / "preprocess_stats_isaac_steps.csv",
        root / "preprocess_stats" / "preprocess_stats_nyu_samples.csv",
        root / "preprocess_stats" / "preprocess_comparison_summary.json",
        root / "preprocess_stats" / "preprocess_comparison_summary.md",
        root / "preprocess_stats" / "depth_hist_isaac_vs_nyu.png",
        root / "preprocess_stats" / "position_valid_ratio_isaac_vs_nyu.png",
        root / "preprocess_stats" / "position_index_hist_isaac_vs_nyu.png",
        root / "preprocess_stats" / "valid_position_mask_grid.png",
        root / "preprocess_stats" / "depth_input_grid.png",
        root / "global_alignment" / "alignment_summary.json",
        root / "global_alignment" / "alignment_per_step.csv",
        root / "global_alignment" / "alignment_grid_index.html",
        root / "future_observed_eval" / "future_eval_per_step.csv",
        root / "future_observed_eval" / "future_eval_tau_sweep.csv",
        root / "future_observed_eval" / "future_eval_summary.json",
        root / "future_observed_eval" / "future_eval_summary.md",
        root / "future_observed_eval" / "calibration_bins.csv",
        root / "future_observed_eval" / "reliability_diagram.png",
        root / "future_observed_eval" / "tau_vs_predicted_unmeasured.png",
        root / "future_observed_eval" / "tau_vs_later_measured_fraction.png",
        root / "future_observed_eval" / "tau_vs_brier.png",
        root / "future_observed_eval" / "confidence_vs_correctness.png",
        root / "alignment_variant_sweep" / "variant_metrics.csv",
        root / "alignment_variant_sweep" / "variant_metrics.json",
        root / "alignment_variant_sweep" / "variant_summary.md",
        root / "alignment_variant_sweep" / "variant_topdown_grid_step0.png",
        root / "alignment_variant_sweep" / "variant_reliability_comparison.png",
        root / "alignment_variant_sweep" / "variant_future_eval_comparison.png",
        root / "alignment_variant_sweep" / "recommendation_alignment.md",
        root / "candidate_score_decomposition" / "rank_correlation_summary.json",
        root / "candidate_score_decomposition" / "rank_correlation_summary.md",
        root / "candidate_score_decomposition" / "gain_exp_vs_gain_sc_scatter.png",
        root / "candidate_score_decomposition" / "path_cost_vs_final_score.png",
        root / "candidate_score_decomposition" / "rank_stability_heatmap.png",
        root / "summary" / "stage4a62_diagnostic_summary.json",
        root / "summary" / "stage4a62_diagnostic_summary.md",
        root / "summary" / "stage4a62_recommendation.md",
        root / "summary" / "issue_priority_table.csv",
        root / "summary" / "key_plots_index.html",
    ]
    for path in required:
        require(path)
    for step in range(5):
        require(root / "candidate_score_decomposition" / f"candidate_rank_decomposition_step{step:03d}.csv")


def validate_observed_state_unchanged() -> None:
    for summary_path in sorted(SC_EPISODE.glob("prediction_step*/prediction_alignment_summary.json")):
        summary = load_json(summary_path)
        observed_path = Path(summary["observed_state_source"])
        require(observed_path)
        actual = sha256_array(np.load(observed_path))
        before = str(summary.get("observed_state_sha256_before", ""))
        after = str(summary.get("observed_state_sha256_after", ""))
        if actual != before or actual != after:
            raise AssertionError(f"observed state hash mismatch for {observed_path}")
        if not bool(summary.get("strict_no_observed_write", False)):
            raise AssertionError(f"prediction summary reports observed write in {summary_path}")


def validate_safety_flags(root: Path) -> None:
    episode_summary = load_json(SC_EPISODE / "episode_summary.json")
    false_keys = [
        "prediction_writeback",
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_used_for_a_star",
        "prediction_blocks_rays",
        "prediction_used_for_candidate_reachability",
        "optimizer_step",
        "behavior_cloning_training",
        "imitation_learning_training",
        "sscnet_training",
        "rl_or_ppo_training",
    ]
    for key in false_keys:
        if bool(episode_summary.get(key, False)):
            raise AssertionError(f"safety flag is unexpectedly true: {key}")
    for step_npz in sorted(SC_EPISODE.glob("step_*.npz")):
        with np.load(step_npz, allow_pickle=False) as data:
            for key in (
                "prediction_used_for_traversability",
                "prediction_used_for_collision",
                "prediction_used_for_a_star",
                "prediction_blocks_rays",
                "prediction_written_to_observed_state",
                "optimizer_step",
            ):
                if key in data.files and bool(np.asarray(data[key]).item()):
                    raise AssertionError(f"{step_npz} has forbidden true flag {key}")
    for rel in (
        "future_observed_eval/future_eval_summary.json",
        "alignment_variant_sweep/variant_metrics.json",
        "summary/stage4a62_diagnostic_summary.json",
    ):
        summary = load_json(root / rel)
        if "post-hoc" not in str(summary.get("future_observations_usage", "")):
            raise AssertionError(f"future-observation evaluation-only note missing in {rel}")
        for key in (
            "prediction_writeback",
            "prediction_used_for_traversability",
            "prediction_used_for_collision",
            "prediction_used_for_a_star",
            "prediction_blocks_rays",
            "planning_or_training_used",
        ):
            if bool(summary.get(key, False)):
                raise AssertionError(f"diagnostic summary reports forbidden true flag {key} in {rel}")


def validate_no_sim_startup_imports() -> None:
    base = Path(__file__).resolve().parent
    forbidden_imports = ("isaaclab", "isaacsim", "omni.", "pxr")
    for name in NEW_SCRIPTS:
        text = (base / name).read_text(encoding="utf-8")
        import_lines = "\n".join(
            line.strip().lower()
            for line in text.splitlines()
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        )
        for token in forbidden_imports:
            if token in import_lines:
                raise AssertionError(f"diagnostic script imports simulator startup dependency: {name}")


def validate_checkpoint_unchanged() -> None:
    episode_summary = load_json(SC_EPISODE / "episode_summary.json")
    if bool(episode_summary.get("checkpoint_modified", False)):
        raise AssertionError("episode summary reports checkpoint modification")
    stat_after = episode_summary.get("checkpoint_stat_after", {})
    checkpoint = Path(str(episode_summary.get("checkpoint", "")))
    require(checkpoint)
    current = checkpoint.stat()
    if stat_after:
        if int(stat_after.get("size_bytes", current.st_size)) != int(current.st_size):
            raise AssertionError("checkpoint size changed")
        if int(stat_after.get("mtime_ns", current.st_mtime_ns)) != int(current.st_mtime_ns):
            raise AssertionError("checkpoint mtime changed")


def validate_recommendation(root: Path) -> None:
    summary = load_json(root / "summary" / "stage4a62_diagnostic_summary.json")
    recommendation = str(summary.get("final_recommendation", "")).strip()
    if not recommendation:
        raise AssertionError("missing final diagnostic recommendation")
    if bool(summary.get("planning_or_training_used", False)):
        raise AssertionError("summary reports planning/training use")


def run_tests(args: argparse.Namespace) -> None:
    root = Path(args.diagnostics_root).resolve()
    validate_required_outputs(root)
    validate_observed_state_unchanged()
    validate_safety_flags(root)
    validate_no_sim_startup_imports()
    validate_checkpoint_unchanged()
    validate_recommendation(root)
    print(f"Stage 4A-6.2 diagnostic validation passed: {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics_root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    run_tests(parse_args())


if __name__ == "__main__":
    main()
