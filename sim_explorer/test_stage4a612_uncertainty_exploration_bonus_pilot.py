#!/usr/bin/env python3
"""Validate Stage 4A-6.12 uncertainty exploration bonus pilot outputs."""

from __future__ import annotations

import argparse
import csv
import json
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

REQUIRED_TOP_LEVEL = [
    "stage4a612_uncertainty_exploration_bonus_pilot_summary.json",
    "stage4a612_uncertainty_exploration_bonus_pilot_summary.md",
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a611_manifest.json",
    "loaded_stage4a611_manifest.md",
    "uncertainty_bonus_formula_reference.json",
    "uncertainty_bonus_formula_reference.md",
    "uncertainty_bonus_beta_sweep.csv",
    "uncertainty_bonus_beta_sweep.json",
    "uncertainty_bonus_beta_sweep.md",
    "per_start_uncertainty_bonus_decisions.csv",
    "per_start_uncertainty_bonus_decisions.json",
    "per_start_uncertainty_bonus_decisions.md",
    "per_formula_decision_table.csv",
    "per_formula_decision_table.json",
    "per_formula_decision_table.md",
    "uncertainty_bonus_vs_lambda48_comparison.csv",
    "uncertainty_bonus_vs_lambda48_comparison.json",
    "uncertainty_bonus_vs_lambda48_comparison.md",
    "uncertainty_bonus_vs_confidence_gated_comparison.csv",
    "uncertainty_bonus_vs_confidence_gated_comparison.json",
    "uncertainty_bonus_vs_confidence_gated_comparison.md",
    "uncertainty_bonus_vs_measured_comparison.csv",
    "uncertainty_bonus_vs_measured_comparison.json",
    "uncertainty_bonus_vs_measured_comparison.md",
    "uncertainty_bonus_quality_audit.json",
    "uncertainty_bonus_quality_audit.md",
    "uncertainty_bonus_risk_audit.json",
    "uncertainty_bonus_risk_audit.md",
    "uncertainty_bonus_readiness_decision.json",
    "uncertainty_bonus_readiness_decision.md",
    "recommended_formula_report.json",
    "recommended_formula_report.md",
    "dataset_schema_delta_for_bc.json",
    "dataset_schema_delta_for_bc.md",
    "future_short_rollout_with_uncertainty_bonus_sketch.md",
    "no_isaac_report.json",
    "no_isaac_report.md",
    "no_capture_report.json",
    "no_capture_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_sscnet_inference_report.json",
    "no_sscnet_inference_report.md",
    "no_action_report.json",
    "no_action_report.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_training_rl_bc_report.json",
    "no_training_rl_bc_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "checkpoint_hash_report.json",
    "checkpoint_hash_report.md",
    "prior_dataset_hash_report.json",
    "prior_dataset_hash_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "expert_decision_dataset_uncertainty_bonus.npz",
    "expert_decision_dataset_manifest.jsonl",
    "expert_decision_dataset_metadata.json",
]

REQUIRED_PLOTS = [
    "uncertainty_bonus_index.html",
    "all_starts_uncertainty_bonus_contact_sheet.png",
    "formula_action_delta_topdown.png",
    "beta_sweep_action_change_bar.png",
    "beta_sweep_quality_score_bar.png",
    "beta_sweep_risk_score_bar.png",
    "uncertainty_bonus_vs_lambda48_delta_topdown.png",
    "uncertainty_bonus_vs_confidence_gated_delta_topdown.png",
    "uncertainty_bonus_vs_measured_delta_topdown.png",
    "source_occ_free_vs_uncertain_fraction_scatter.png",
    "entropy_vs_action_change_scatter.png",
    "margin_vs_action_change_scatter.png",
    "confidence_vs_action_change_scatter.png",
    "branch_class_by_beta_bar.png",
    "candidate_all_local_by_beta_bar.png",
    "selected_candidate_uncertainty_by_formula_boxplot.png",
    "selected_candidate_confidence_by_formula_boxplot.png",
    "selected_candidate_entropy_by_formula_boxplot.png",
    "selected_candidate_margin_by_formula_boxplot.png",
    "safety_flags_summary.png",
    "quality_warning_summary.png",
]

REQUIRED_PER_START = [
    "candidate_features_with_uncertainty.csv",
    "formula_scores.csv",
    "selected_actions_by_formula.json",
    "uncertainty_bonus_beta_sweep_start.json",
    "action_delta_vs_lambda48.png",
    "action_delta_vs_confidence_gated.png",
    "candidate_uncertainty_map.png",
    "score_decomposition_uncertainty_bonus.png",
    "uncertainty_risk_panel.png",
    "quality_verdict.md",
    "quality_verdict.json",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def csv_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def jsonl_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def git_policy() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    staged_forbidden = []
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/")
    forbidden_suffixes = (".usd", ".npy", ".npz", ".png", ".mp4")
    for line in result.stdout.splitlines():
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if status.strip() and status[0] != "?" and (path.startswith(forbidden_prefixes) or path.endswith(forbidden_suffixes)):
            staged_forbidden.append(line)
    return {"git_status_short": result.stdout, "staged_forbidden_large_artifacts": staged_forbidden}


def validate_dataset(path: Path, expected_starts: int) -> dict[str, Any]:
    require(path.is_file(), "decision dataset missing")
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        required = {
            "start_variant_id",
            "candidate_features",
            "candidate_mask",
            "valid_mask",
            "action_index_measured_only",
            "action_index_lambda48",
            "action_index_confidence_gated_6_11",
            "score_measured_only",
            "score_lambda48",
            "score_confidence_gated",
            "scores_uncertainty_bonus_all_formulas",
            "gain_exp",
            "path_cost",
            "source_occ_free",
            "candidate_confidence_mean",
            "candidate_entropy_mean",
            "candidate_margin_mean",
            "candidate_uncertain_fraction",
            "candidate_uncertain_voxel_count",
            "candidate_low_conf_count_0p7",
            "candidate_high_entropy_count_0p7",
            "candidate_low_margin_count_0p2",
            "low_cost_artifact",
            "historical_prior_basin",
            "no_valid_candidate",
            "candidate_all_local",
            "selected_high_entropy",
            "selected_low_confidence",
            "selected_low_margin",
            "formula_dominated_by_uncertainty",
        }
        for mode in ("fraction", "entropy", "low_margin", "composite"):
            for beta in (2, 4, 8, 16, 32):
                required.add(f"action_index_unc_bonus_{mode}_beta{beta}")
        missing = sorted(required - keys)
        forbidden = sorted(keys & FORBIDDEN_KEYS)
        require(not missing, f"dataset missing keys: {missing}")
        require(not forbidden, f"dataset contains forbidden keys: {forbidden}")
        lower_keys = [key.lower() for key in keys]
        require(not any(key in lower_keys for key in ("target_hr", "target_lr", "ground_truth", "gt", "future_observed", "class_prob")), "forbidden dataset field substring present")
        require(data["start_variant_id"].shape[0] == expected_starts, "start count mismatch in dataset")
        require(data["candidate_features"].shape[:2] == (expected_starts, 64), "candidate feature shape mismatch")
        require(data["scores_uncertainty_bonus_all_formulas"].shape == (expected_starts, 20, 64), "formula score shape mismatch")
        for key in ["score_measured_only", "score_lambda48", "score_confidence_gated", "scores_uncertainty_bonus_all_formulas"]:
            arr = np.asarray(data[key])
            require(np.all(np.isfinite(arr)), f"{key} contains NaN/Inf")
        for key in [
            "candidate_confidence_mean",
            "candidate_entropy_mean",
            "candidate_margin_mean",
            "candidate_uncertain_fraction",
        ]:
            arr = np.asarray(data[key])
            require(np.all(np.isfinite(arr)), f"{key} contains NaN/Inf")
            require(float(np.min(arr)) >= 0.0, f"{key} below zero")
            require(float(np.max(arr)) <= 1.0005, f"{key} above one")
        return {"dataset_keys": sorted(keys), "forbidden": forbidden}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--measured_only_pilot_dir", type=Path, required=True)
    parser.add_argument("--lambda48_pilot_dir", type=Path, required=True)
    parser.add_argument("--dense_uncertainty_dir", type=Path, required=True)
    parser.add_argument("--uncertainty_aware_pilot_dir", type=Path, required=True)
    parser.add_argument("--expected_start_count", type=int, default=10)
    parser.add_argument("--expect_no_isaac", action="store_true")
    parser.add_argument("--expect_no_capture", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_sscnet_inference", action="store_true")
    parser.add_argument("--expect_no_action", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_second_action", action="store_true")
    parser.add_argument("--expect_no_third_frame", action="store_true")
    parser.add_argument("--expect_no_long_rollout", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_quality_viz", action="store_true")
    parser.add_argument("--expect_future_short_rollout_not_executed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    require(out.is_dir(), f"output dir missing: {out}")
    for name in REQUIRED_TOP_LEVEL:
        require((out / name).is_file(), f"missing top-level output: {name}")

    summary = load_json(out / "stage4a612_uncertainty_exploration_bonus_pilot_summary.json")
    quality = load_json(out / "uncertainty_bonus_quality_audit.json")
    risk = load_json(out / "uncertainty_bonus_risk_audit.json")
    readiness = load_json(out / "uncertainty_bonus_readiness_decision.json")
    source = load_json(out / "source_hash_report.json")
    checkpoint = load_json(out / "checkpoint_hash_report.json")
    prior = load_json(out / "prior_dataset_hash_report.json")

    require(summary["completed"] is True, "summary completed false")
    require(summary["loaded_stage4a611"] is True, "6.11 not loaded")
    require(summary["loaded_stage4a610a_dense_uncertainty"] is True, "6.10a dense uncertainty not loaded")
    require(int(summary["candidate_rows_loaded"]) > 0, "candidate rows not loaded")
    require(int(summary["start_count"]) == args.expected_start_count, "start count mismatch")
    require(int(summary["selected_action_records"]) == args.expected_start_count, "selected action records mismatch")
    require(summary["beta_values"] == [2, 4, 8, 16, 32], "beta values mismatch")
    require(sorted(summary["uncertainty_bonus_modes"]) == ["composite", "entropy", "fraction", "low_margin"], "bonus modes mismatch")
    require(summary["source_occ_free_kept_separate_from_uncertainty"] is True, "source_occ_free separation false")
    require(quality["passed"] is True, "quality audit failed")
    require(risk["passed"] is True, "risk audit failed")
    require(readiness["recommended_uncertainty_bonus_formula"] == summary["recommended_uncertainty_bonus_formula"], "readiness formula mismatch")
    require(readiness["short_rollout_executed_this_stage"] is False, "short rollout executed")
    require(readiness["long_rollout_executed_this_stage"] is False, "long rollout executed")

    if args.expect_no_isaac:
        require(int(summary["isaac_startup_count_this_stage"]) == 0, "Isaac startup count nonzero")
        require(load_json(out / "no_isaac_report.json")["isaac_startup_count_this_stage"] == 0, "no_isaac report bad")
    if args.expect_no_capture:
        require(int(summary["capture_count_this_stage"]) == 0, "capture count nonzero")
    if args.expect_no_map_predict:
        require(int(summary["map_predict_calls_this_stage"]) == 0, "map_predict count nonzero")
    if args.expect_no_sscnet_inference:
        require(int(summary["sscnet_inference_calls_this_stage"]) == 0, "SSCNet inference count nonzero")
    if args.expect_no_action:
        require(int(summary["action_execution_count_this_stage"]) == 0, "action execution count nonzero")
    if args.expect_no_rollout:
        require(summary["rollout_executed"] is False and summary["short_rollout_executed"] is False, "rollout executed")
    if args.expect_no_second_action:
        require(int(summary["second_action_count"]) == 0, "second action count nonzero")
    if args.expect_no_third_frame:
        require(int(summary["third_frame_count"]) == 0, "third frame count nonzero")
    if args.expect_no_long_rollout:
        require(summary["long_rollout_executed"] is False, "long rollout executed")
    if args.expect_no_training:
        for key in ("training_executed", "bc_training_executed", "il_training_executed"):
            require(summary[key] is False, f"{key} true")
    if args.expect_no_rl_gdpo:
        for key in ("rl_training_executed", "gdpo_training_executed", "ppo_training_executed"):
            require(summary[key] is False, f"{key} true")

    for key in (
        "prediction_writeback",
        "uncertainty_writeback",
        "prediction_uncertainty_safety_leakage",
        "prediction_used_for_traversability",
        "uncertainty_used_for_traversability",
        "prediction_used_for_collision",
        "uncertainty_used_for_collision",
        "prediction_used_for_ray_blocking",
        "uncertainty_used_for_ray_blocking",
        "prediction_used_for_candidate_validity",
        "uncertainty_used_for_candidate_validity",
        "target_ground_truth_future_observed_scoring",
        "source_usd_modified",
        "fixed_usd_modified",
        "checkpoint_modified",
        "observed_state_modified",
    ):
        require(summary[key] is False, f"{key} true")

    beta_rows = list(csv.DictReader((out / "uncertainty_bonus_beta_sweep.csv").open("r", encoding="utf-8", newline="")))
    require(len(beta_rows) == 20, "beta sweep row count mismatch")
    formulas = {row["formula"] for row in beta_rows}
    for mode in ("fraction", "entropy", "low_margin", "composite"):
        for beta in (2, 4, 8, 16, 32):
            require(f"uncertainty_bonus_{mode}_beta{beta}" in formulas, f"missing formula {mode} beta {beta}")

    require(csv_row_count(out / "per_start_uncertainty_bonus_decisions.csv") == args.expected_start_count, "per start decisions row count mismatch")
    require(csv_row_count(out / "uncertainty_bonus_vs_lambda48_comparison.csv") == args.expected_start_count, "lambda comparison row count mismatch")
    require(csv_row_count(out / "uncertainty_bonus_vs_confidence_gated_comparison.csv") == args.expected_start_count, "confidence comparison row count mismatch")
    require(csv_row_count(out / "uncertainty_bonus_vs_measured_comparison.csv") == args.expected_start_count, "measured comparison row count mismatch")
    require(jsonl_row_count(out / "expert_decision_dataset_manifest.jsonl") == args.expected_start_count, "decision dataset manifest row count mismatch")

    dataset_info = validate_dataset(out / "expert_decision_dataset_uncertainty_bonus.npz", args.expected_start_count)

    if args.expect_quality_viz:
        for name in REQUIRED_PLOTS:
            path = out / name
            require(path.is_file(), f"missing visualization: {name}")
            require(path.stat().st_size > 0, f"empty visualization: {name}")
    per_start_dirs = sorted((out / "per_start").glob("start_*"))
    require(len(per_start_dirs) == args.expected_start_count, "per-start dir count mismatch")
    for start_dir in per_start_dirs:
        for name in REQUIRED_PER_START:
            path = start_dir / name
            require(path.is_file(), f"missing per-start file: {path}")
            require(path.stat().st_size > 0, f"empty per-start file: {path}")
        verdict = load_json(start_dir / "quality_verdict.json")
        require(verdict["action_executed"] is False, f"per-start action executed in {start_dir}")
        require(csv_row_count(start_dir / "candidate_features_with_uncertainty.csv") >= 1, f"no candidates in {start_dir}")

    if args.expect_future_short_rollout_not_executed:
        sketch = (out / "future_short_rollout_with_uncertainty_bonus_sketch.md").read_text(encoding="utf-8")
        require(sketch.startswith("DO NOT RUN IN STAGE 4A-6.12."), "future sketch missing DO NOT RUN prefix")
        require("DO NOT RUN IN STAGE 4A-6.12." in sketch, "future sketch safety text missing")

    require(source["source_usd_sha256_before"] == source["source_usd_sha256_after"], "source USD hash changed")
    require(source["fixed_usd_sha256_before"] == source["fixed_usd_sha256_after"], "fixed USD hash changed")
    require(checkpoint["checkpoint_sha256_before"] == checkpoint["checkpoint_sha256_after"], "checkpoint hash changed")
    for name, item in prior["datasets"].items():
        require(item["sha256_before"] == item["sha256_after"], f"prior dataset changed: {name}")
        require(item["unchanged"] is True, f"prior dataset unchanged false: {name}")

    policy = git_policy()
    require(not policy["staged_forbidden_large_artifacts"], f"large artifact policy violation: {policy['staged_forbidden_large_artifacts']}")

    result = {
        "all_passed": True,
        "output_dir": str(out),
        "start_count": summary["start_count"],
        "candidate_rows_loaded": summary["candidate_rows_loaded"],
        "recommended_formula": summary["recommended_uncertainty_bonus_formula"],
        "recommended_beta": summary["recommended_beta"],
        "uncertainty_bonus_runtime_ready": summary["uncertainty_bonus_runtime_ready"],
        "dataset_key_count": len(dataset_info["dataset_keys"]),
        "git_policy_forbidden_status_rows": policy["staged_forbidden_large_artifacts"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
