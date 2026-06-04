#!/usr/bin/env python3
"""Validate Stage 4A-6.10 prediction uncertainty offline audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")

REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "loaded_stage4a68_manifest.json",
    "loaded_stage4a68_manifest.md",
    "loaded_stage4a69_manifest.json",
    "loaded_stage4a69_manifest.md",
    "prediction_artifact_inventory.csv",
    "prediction_artifact_inventory.json",
    "prediction_artifact_inventory.md",
    "uncertainty_available_fields_report.json",
    "uncertainty_available_fields_report.md",
    "uncertainty_formula_reference.json",
    "uncertainty_formula_reference.md",
    "frame_uncertainty_summary.csv",
    "frame_uncertainty_summary.json",
    "frame_uncertainty_summary.md",
    "candidate_uncertainty_table.csv",
    "candidate_uncertainty_table.json",
    "candidate_uncertainty_table.md",
    "selected_action_uncertainty_audit.csv",
    "selected_action_uncertainty_audit.json",
    "selected_action_uncertainty_audit.md",
    "uncertainty_vs_source_occ_free_analysis.csv",
    "uncertainty_vs_source_occ_free_analysis.json",
    "uncertainty_vs_source_occ_free_analysis.md",
    "uncertainty_vs_branch_classification.csv",
    "uncertainty_vs_branch_classification.json",
    "uncertainty_vs_branch_classification.md",
    "frame1_vs_frame2_uncertainty_analysis.csv",
    "frame1_vs_frame2_uncertainty_analysis.json",
    "frame1_vs_frame2_uncertainty_analysis.md",
    "stage4a68_vs_stage4a69_uncertainty_comparison.csv",
    "stage4a68_vs_stage4a69_uncertainty_comparison.json",
    "stage4a68_vs_stage4a69_uncertainty_comparison.md",
    "uncertainty_shadow_score_audit.csv",
    "uncertainty_shadow_score_audit.json",
    "uncertainty_shadow_score_audit.md",
    "uncertainty_readiness_decision.json",
    "uncertainty_readiness_decision.md",
    "prediction_safety_recheck.json",
    "prediction_safety_recheck.md",
    "no_isaac_report.json",
    "no_isaac_report.md",
    "no_capture_report.json",
    "no_capture_report.md",
    "no_map_predict_report.json",
    "no_map_predict_report.md",
    "no_sscnet_inference_report.json",
    "no_sscnet_inference_report.md",
    "no_rollout_report.json",
    "no_rollout_report.md",
    "no_training_rl_bc_report.json",
    "no_training_rl_bc_report.md",
    "source_hash_report.json",
    "source_hash_report.md",
    "checkpoint_hash_report.json",
    "checkpoint_hash_report.md",
    "git_status_before.txt",
    "git_status_after.txt",
    "stage4a610_prediction_uncertainty_offline_audit_summary.json",
    "stage4a610_prediction_uncertainty_offline_audit_summary.md",
    "recommended_next_faithful_step.md",
    "future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md",
    "uncertainty_overview_index.html",
    "frame_uncertainty_contact_sheet.png",
    "selected_action_uncertainty_contact_sheet.png",
    "uncertainty_vs_source_occ_free_scatter.png",
    "uncertainty_vs_gain_exp_scatter.png",
    "uncertainty_vs_path_cost_scatter.png",
    "uncertainty_vs_lambda48_score_scatter.png",
    "branch_class_uncertainty_bar.png",
    "frame1_frame2_uncertainty_delta_bar.png",
    "selected_vs_nonselected_uncertainty_boxplot.png",
    "high_uncertainty_candidate_examples.png",
    "low_confidence_warning_map.png",
    "entropy_topdown_examples.png",
    "confidence_topdown_examples.png",
    "margin_topdown_examples.png",
]

LIMITED_MODE_FILES = [
    "dense_uncertainty_blocker.json",
    "dense_uncertainty_blocker.md",
]

FORBIDDEN_DATASET_KEYS = {
    "target_lr",
    "target_hr",
    "ground_truth",
    "future_observed",
    "policy_logits",
    "rl_reward",
    "replay_buffer",
    "training_state",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_csv_no_nan_inf(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            for key, value in row.items():
                text = str(value).strip()
                if not text:
                    continue
                lowered = text.lower()
                require(lowered not in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}, f"{path.name} row {row_index} has non-finite literal in {key}")
                try:
                    number = float(text)
                except ValueError:
                    continue
                require(math.isfinite(number), f"{path.name} row {row_index} has non-finite number in {key}")


def git_large_artifact_policy_preserved() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    forbidden_prefixes = ("outputs/", "logs/", "checkpoints/", "data/", "assets/home_like_scene_v1/current_environment")
    forbidden_suffixes = (".usd", ".npy", ".npz", ".png", ".mp4")
    staged_forbidden = []
    for line in result.stdout.splitlines():
        status = line[:2]
        path = line[3:] if len(line) > 3 else ""
        if status.strip() and status[0] != "?" and (path.startswith(forbidden_prefixes) or path.endswith(forbidden_suffixes)):
            staged_forbidden.append(line)
    return {"passed": not staged_forbidden, "staged_forbidden_large_artifacts": staged_forbidden, "git_status_short": result.stdout}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage4a68_dir", type=Path, required=True)
    parser.add_argument("--stage4a69_dir", type=Path, required=True)
    parser.add_argument("--fixed_usd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--allow_limited_if_dense_prediction_missing", action="store_true")
    parser.add_argument("--expect_no_isaac", action="store_true")
    parser.add_argument("--expect_no_capture", action="store_true")
    parser.add_argument("--expect_no_map_predict", action="store_true")
    parser.add_argument("--expect_no_sscnet_inference", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_future_stage4a611_not_executed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    require(output_dir.is_dir(), "output dir missing")

    for name in REQUIRED_FILES:
        require((output_dir / name).is_file(), f"required file missing: {name}")

    summary = load_json(output_dir / "stage4a610_prediction_uncertainty_offline_audit_summary.json")
    readiness = load_json(output_dir / "uncertainty_readiness_decision.json")
    safety = load_json(output_dir / "prediction_safety_recheck.json")
    available = load_json(output_dir / "uncertainty_available_fields_report.json")
    context = load_json(output_dir / "loaded_context_manifest.json")
    manifest68 = load_json(output_dir / "loaded_stage4a68_manifest.json")
    manifest69 = load_json(output_dir / "loaded_stage4a69_manifest.json")

    require(summary.get("completed") is True, "summary completed is not true")
    require(summary.get("stage4a68_loaded") is True, "Stage 4A-6.8 not loaded")
    require(summary.get("stage4a69_loaded") is True, "Stage 4A-6.9 not loaded")
    require(context.get("context_loaded") is True, "context not loaded")
    require(manifest68.get("loaded") is True, "Stage 4A-6.8 manifest not loaded")
    require(manifest69.get("loaded") is True, "Stage 4A-6.9 manifest not loaded")
    require(summary.get("frames_analyzed") == 30, "expected 30 frames analyzed across 6.8 and 6.9")
    require(summary.get("candidates_analyzed", 0) > 0, "candidate rows were not analyzed")
    require(summary.get("candidate_level_uncertainty_rows") == 0, "limited mode should not compute candidate uncertainty rows")
    require(readiness.get("uncertainty_feature_extraction_complete") is True, "feature extraction should be complete in limited mode")
    require(readiness.get("candidate_level_uncertainty_ready") is False, "candidate-level uncertainty should be blocked")
    require(readiness.get("uncertainty_aware_expert_pilot_ready") is False, "uncertainty-aware pilot should not be ready")
    require(available.get("uncertainty_mode") == "summary_only_limited", "expected summary_only_limited mode")

    if available.get("dense_prediction_available") is False:
        require(args.allow_limited_if_dense_prediction_missing, "dense prediction missing but limited mode not allowed")
        for name in LIMITED_MODE_FILES:
            require((output_dir / name).is_file(), f"limited-mode blocker file missing: {name}")

    require(safety.get("passed") is True, "prediction safety recheck failed")
    require(safety.get("stage4a68_prediction_safety_passed") is True, "6.8 prediction safety not passed")
    require(safety.get("stage4a69_prediction_safety_passed") is True, "6.9 prediction safety not passed")
    require(safety.get("forbidden_prediction_flags_false") is True, "forbidden prediction flags are not all false")

    if args.expect_no_isaac:
        require(summary.get("isaac_startup_count_this_stage") == 0, "Isaac startup count is not zero")
    if args.expect_no_capture:
        require(summary.get("capture_count_this_stage") == 0, "capture count is not zero")
    if args.expect_no_map_predict:
        require(summary.get("map_predict_calls_this_stage") == 0, "map_predict calls are not zero")
    if args.expect_no_sscnet_inference:
        require(summary.get("sscnet_inference_calls_this_stage") == 0, "SSCNet inference calls are not zero")
    if args.expect_no_rollout:
        require(summary.get("rollout_executed_this_stage") is False, "rollout executed")
        require(summary.get("long_rollout_executed_this_stage") is False, "long rollout executed")
        require(safety.get("second_action_count_this_stage") == 0, "second action count not zero")
        require(safety.get("third_frame_count_this_stage") == 0, "third frame count not zero")
    if args.expect_no_training or args.expect_no_rl_gdpo:
        require(summary.get("training_run_this_stage") is False, "training run executed")
        require(summary.get("bc_il_rl_run_this_stage") is False, "BC/IL/RL/GDPO/PPO run executed")
    if args.expect_future_stage4a611_not_executed:
        require(summary.get("future_stage4a611_executed") is False, "future 6.11 executed")

    sketch = (output_dir / "future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md").read_text(encoding="utf-8")
    require(sketch.startswith("DO NOT RUN IN STAGE 4A-6.10."), "future sketch missing DO NOT RUN prefix")

    for csv_name in [
        "frame_uncertainty_summary.csv",
        "candidate_uncertainty_table.csv",
        "selected_action_uncertainty_audit.csv",
        "uncertainty_vs_source_occ_free_analysis.csv",
        "uncertainty_vs_branch_classification.csv",
        "frame1_vs_frame2_uncertainty_analysis.csv",
        "stage4a68_vs_stage4a69_uncertainty_comparison.csv",
        "uncertainty_shadow_score_audit.csv",
    ]:
        check_csv_no_nan_inf(output_dir / csv_name)

    require(sha256_file(args.stage4a68_dir / "expert_dataset.npz") == manifest68["required_files"][1]["sha256"], "6.8 dataset hash changed")
    require(sha256_file(args.stage4a69_dir / "expert_dataset_two_frame.npz") == manifest69["required_files"][1]["sha256"], "6.9 dataset hash changed")
    require(sha256_file(args.fixed_usd) == safety["source_hashes"]["fixed_usd_sha256"], "fixed USD hash changed")
    require(sha256_file(args.checkpoint) == safety["source_hashes"]["checkpoint_sha256"], "checkpoint hash changed")

    for manifest in (manifest68, manifest69):
        forbidden_keys = set(manifest["npz"]["keys"]) & FORBIDDEN_DATASET_KEYS
        require(not forbidden_keys, f"forbidden target/ground_truth/future/training keys in dataset: {forbidden_keys}")

    policy = git_large_artifact_policy_preserved()
    require(policy["passed"], f"large artifact policy violated: {policy['staged_forbidden_large_artifacts']}")

    result = {
        "all_passed": True,
        "output_dir": str(output_dir),
        "limited_mode": available.get("uncertainty_mode") == "summary_only_limited",
        "dense_uncertainty_blocked": available.get("dense_prediction_available") is False,
        "frames_analyzed": summary.get("frames_analyzed"),
        "candidates_analyzed": summary.get("candidates_analyzed"),
        "git_large_artifact_policy_preserved": policy["passed"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
