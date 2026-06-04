#!/usr/bin/env python3
"""Validate Stage 4A-6.13 uncertainty-bonus short rollout outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")
FORBIDDEN_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "gt",
    "future_observed",
    "class_prob",
    "policy_logits",
    "rl_reward",
    "replay_buffer",
    "training_state",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def require_file(path: Path, checks: dict[str, Any], key: str) -> None:
    checks[key] = path.is_file()
    if not path.is_file():
        checks.setdefault("missing_files", []).append(str(path))


def finite_npz_scores(dataset: Path) -> bool:
    with np.load(dataset, allow_pickle=False) as data:
        return bool(np.all(np.isfinite(data["score_primary_uncertainty_bonus"])))


def git_large_artifact_policy_preserved() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    tracked = result.stdout.splitlines()
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/")
    forbidden_suffixes = (".npy", ".npz", ".png", ".mp4", ".usd", ".pth", ".tar")
    offenders = [p for p in tracked if p.startswith(forbidden_prefixes) or p.lower().endswith(forbidden_suffixes)]
    return {"passed": result.returncode == 0 and not offenders, "offenders": offenders[:40]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--camera_pose_fix_dir", type=Path, required=True)
    parser.add_argument("--measured_only_pilot_dir", type=Path, required=True)
    parser.add_argument("--lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--two_frame_lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--dense_uncertainty_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_aware_pilot_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_bonus_decision_dir", type=Path, required=True)
    parser.add_argument("--expected_start_count", type=int, default=10)
    parser.add_argument("--max_decision_steps_per_start", type=int, default=3)
    parser.add_argument("--max_total_actions", type=int, default=30)
    parser.add_argument("--max_total_decision_frames", type=int, default=30)
    parser.add_argument("--max_total_captures", type=int, default=40)
    parser.add_argument("--expect_uncertainty_bonus_composite_beta8", action="store_true")
    parser.add_argument("--expect_shadow_formulas", action="store_true")
    parser.add_argument("--expect_dense_uncertainty_artifacts", action="store_true")
    parser.add_argument("--expect_quality_viz", action="store_true")
    parser.add_argument("--expect_terminal_capture", action="store_true")
    parser.add_argument("--expect_no_long_rollout", action="store_true")
    parser.add_argument("--expect_no_full_expert_dataset", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    checks: dict[str, Any] = {"output_dir": str(out), "output_dir_exists": out.is_dir()}
    require_file(out / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json", checks, "summary_exists")
    require_file(out / "short_rollout_dataset_uncertainty_bonus.npz", checks, "dataset_exists")
    require_file(out / "short_rollout_manifest.jsonl", checks, "manifest_exists")
    require_file(out / "short_rollout_metadata.json", checks, "metadata_exists")
    require_file(out / "short_rollout_uncertainty_bonus_index.html", checks, "html_visualization_exists")

    summary = read_json(out / "stage4a613_uncertainty_bonus_short_rollout_pilot_summary.json")
    prediction = read_json(out / "prediction_safety_audit.json")
    uncertainty = read_json(out / "uncertainty_safety_audit.json")
    rollout = read_json(out / "rollout_safety_audit.json")
    quality = read_json(out / "expert_data_quality_audit.json")
    integrity = read_json(out / "dataset_integrity_report.json")
    runtime_quality = read_json(out / "uncertainty_bonus_runtime_quality_audit.json")
    source_hash = read_json(out / "source_hash_report.json")
    checkpoint_hash = read_json(out / "checkpoint_hash_report.json")
    prior_hash = read_json(out / "prior_dataset_hash_report.json")

    checks.update(
        {
            "completed": bool(summary.get("completed")),
            "start_count": int(summary.get("start_count", -1)) == args.expected_start_count,
            "max_decision_steps_per_start": int(summary.get("max_decision_steps_per_start", -1)) == args.max_decision_steps_per_start,
            "total_executed_actions_limit": int(summary.get("executed_action_count", 9999)) <= args.max_total_actions,
            "total_decision_frames_limit": int(summary.get("decision_frame_count", 9999)) <= args.max_total_decision_frames,
            "total_terminal_frames_limit": int(summary.get("terminal_frame_count", 9999)) <= args.expected_start_count,
            "total_captures_limit": int(summary.get("capture_count", 9999)) <= args.max_total_captures,
            "map_predict_calls_limit": int(summary.get("map_predict_calls", 9999)) <= args.max_total_decision_frames,
            "predictor_loaded_once": bool(summary.get("predictor_loaded_once")),
            "long_rollout_false": not bool(summary.get("long_rollout_executed")),
            "full_expert_dataset_false": not bool(summary.get("full_expert_dataset_executed")),
            "training_false": not bool(summary.get("training")),
            "no_bc_il_rl_gdpo_ppo": not bool(summary.get("bc_il_rl_gdpo_ppo")),
            "primary_formula": summary.get("primary_formula") == "uncertainty_bonus_composite_beta8",
            "beta": int(summary.get("beta", -1)) == 8,
            "shadow_formulas_exist": (out / "measured_shadow_decisions.csv").is_file()
            and (out / "lambda48_shadow_decisions.csv").is_file()
            and (out / "confidence_gated_shadow_decisions.csv").is_file(),
            "dense_uncertainty_artifacts_count": int(summary.get("dense_uncertainty_artifacts", -1)) == int(summary.get("decision_frame_count", -2)),
            "prediction_safety_passed": bool(prediction.get("passed")),
            "uncertainty_safety_passed": bool(uncertainty.get("passed")),
            "rollout_safety_passed": bool(rollout.get("passed")),
            "expert_quality_passed_or_no_blockers": bool(quality.get("passed")) or not quality.get("blockers"),
            "runtime_quality_no_blockers": not runtime_quality.get("blockers"),
            "dataset_integrity_passed": bool(integrity.get("passed")),
            "prediction_writeback_false": not bool(summary.get("prediction_writeback")),
            "uncertainty_writeback_false": not bool(summary.get("uncertainty_writeback")),
            "no_prediction_uncertainty_traversability": not bool(summary.get("prediction_uncertainty_traversability_use")),
            "no_prediction_uncertainty_collision": not bool(summary.get("prediction_uncertainty_collision_use")),
            "no_prediction_uncertainty_ray": not bool(summary.get("prediction_uncertainty_ray_blocking_use")),
            "no_prediction_uncertainty_candidate_validity": not bool(summary.get("prediction_uncertainty_candidate_validity_use")),
            "no_target_ground_truth_future_scoring": not bool(summary.get("target_ground_truth_future_observed_scoring")),
            "source_usd_unchanged": source_hash.get("source_usd_sha256_before") == source_hash.get("source_usd_sha256_after"),
            "fixed_usd_unchanged": source_hash.get("fixed_usd_sha256_before") == source_hash.get("fixed_usd_sha256_after"),
            "checkpoint_unchanged": checkpoint_hash.get("checkpoint_sha256_before") == checkpoint_hash.get("checkpoint_sha256_after"),
            "prior_datasets_unchanged": all(v.get("sha256_before") == v.get("sha256_after") for v in prior_hash.values()),
            "mp4_or_fallback_exists": (out / "short_rollout_flythrough.mp4").is_file() or any((out / "short_rollout_flythrough_frames").glob("frame_*.png")),
        }
    )

    dataset_path = out / "short_rollout_dataset_uncertainty_bonus.npz"
    if dataset_path.is_file():
        with np.load(dataset_path, allow_pickle=False) as data:
            keys = set(data.files)
            checks["no_forbidden_dataset_keys"] = not bool(keys & FORBIDDEN_KEYS)
            checks["forbidden_dataset_keys"] = sorted(keys & FORBIDDEN_KEYS)
            checks["dataset_scores_finite"] = finite_npz_scores(dataset_path)
            checks["dataset_transition_count_matches_summary"] = int(data["start_variant_id"].shape[0]) == int(summary.get("decision_frame_count", -1))

    per_step_rows = read_csv(out / "per_step_summary.csv") if (out / "per_step_summary.csv").is_file() else []
    checks["per_step_summary_rows"] = len(per_step_rows) == int(summary.get("decision_frame_count", -1))
    primary_rows = read_csv(out / "primary_uncertainty_bonus_decisions.csv") if (out / "primary_uncertainty_bonus_decisions.csv").is_file() else []
    checks["primary_rows_match_decision_frames"] = len(primary_rows) == int(summary.get("decision_frame_count", -1))
    checks["no_nan_inf_in_scores"] = all(math.isfinite(float(row["final_score"])) for row in primary_rows)

    per_start_missing = []
    for sid in range(args.expected_start_count):
        start_dir = out / "samples" / f"start_{sid:03d}"
        for name in (
            "start_summary.json",
            "start_summary.md",
            "observed_ratio_curve.csv",
            "observed_ratio_curve.png",
            "path_topdown.png",
            "action_sequence.jsonl",
            "formula_action_delta_sequence.csv",
            "quality_flags_sequence.csv",
            "terminal_rgb.png",
            "terminal_depth.npy",
            "terminal_depth_color.png",
            "terminal_pose.json",
            "terminal_observed_state.npy",
            "terminal_quality.json",
            "terminal_observed_topdown.png",
        ):
            if not (start_dir / name).is_file():
                per_start_missing.append(str(start_dir / name))
        start_summary = read_json(start_dir / "start_summary.json") if (start_dir / "start_summary.json").is_file() else {}
        for step in range(int(start_summary.get("executed_action_count", 0))):
            for name in (
                f"step_{step:03d}_rgb.png",
                f"step_{step:03d}_depth.npy",
                f"step_{step:03d}_depth_color.png",
                f"step_{step:03d}_pose.json",
                f"step_{step:03d}_observed_state.npy",
                f"step_{step:03d}_dense_prediction_uncertainty.npz",
                f"step_{step:03d}_dense_prediction_summary.json",
                f"step_{step:03d}_candidate_features.csv",
                f"step_{step:03d}_candidate_uncertainty_features.csv",
                f"step_{step:03d}_primary_decision.json",
                f"step_{step:03d}_measured_shadow_decision.json",
                f"step_{step:03d}_lambda48_shadow_decision.json",
                f"step_{step:03d}_confidence_gated_shadow_decision.json",
                f"step_{step:03d}_executed_action.json",
                f"step_{step:03d}_action_quality.json",
                f"step_{step:03d}_action_quality.md",
                f"step_{step:03d}_observed_topdown.png",
                f"step_{step:03d}_prediction_overlay.png",
                f"step_{step:03d}_confidence_overlay.png",
                f"step_{step:03d}_entropy_overlay.png",
                f"step_{step:03d}_margin_overlay.png",
                f"step_{step:03d}_candidate_map.png",
                f"step_{step:03d}_candidate_score_bar.png",
                f"step_{step:03d}_uncertainty_score_bar.png",
                f"step_{step:03d}_formula_action_delta_map.png",
            ):
                if not (start_dir / name).is_file():
                    per_start_missing.append(str(start_dir / name))
    checks["per_start_and_step_outputs_exist"] = not per_start_missing
    checks["missing_per_start_or_step_outputs"] = per_start_missing[:80]

    for name in (
        "all_starts_path_contact_sheet.png",
        "all_starts_observed_ratio_curve.png",
        "aggregate_observed_ratio_curve.png",
        "aggregate_newly_observed_voxels_curve.png",
        "aggregate_path_length_curve.png",
        "aggregate_action_distance_hist.png",
        "aggregate_yaw_delta_hist.png",
        "formula_action_change_bar.png",
        "primary_vs_measured_delta_topdown.png",
        "primary_vs_lambda48_delta_topdown.png",
        "primary_vs_confidence_gated_delta_topdown.png",
        "uncertainty_composite_by_step.png",
        "selected_confidence_by_step.png",
        "selected_entropy_by_step.png",
        "selected_margin_by_step.png",
        "source_occ_free_by_step.png",
        "gain_exp_by_step.png",
        "path_cost_by_step.png",
        "candidate_all_local_by_step.png",
        "safety_flags_summary.png",
        "quality_warning_summary.png",
        "stuck_revisit_summary.png",
    ):
        require_file(out / name, checks, f"viz_{name}")

    git_policy = git_large_artifact_policy_preserved()
    checks["git_large_artifact_policy_preserved"] = git_policy["passed"]
    checks["git_large_artifact_offenders"] = git_policy["offenders"]

    checks["all_passed"] = all(
        bool(value)
        for key, value in checks.items()
        if key
        not in {
            "output_dir",
            "missing_files",
            "forbidden_dataset_keys",
            "missing_per_start_or_step_outputs",
            "git_large_artifact_offenders",
        }
    )
    report_path = out / "stage4a613_test_report.json"
    report_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not checks["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
