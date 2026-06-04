#!/usr/bin/env python3
"""Validate Stage 4A-6.10a dense uncertainty artifacts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE = Path("/home/ubuntu22/sc_explorer_ws")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str | None:
    import hashlib

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_dense_npz(path: Path) -> None:
    require(path.is_file(), f"dense artifact missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        for key in (
            "confidence_float16",
            "entropy_norm_float16",
            "margin_float16",
            "valid_mask_bool",
            "predicted_unmeasured_mask_bool",
        ):
            require(key in data.files, f"{path} missing {key}")
        conf = np.asarray(data["confidence_float16"], dtype=np.float32)
        ent = np.asarray(data["entropy_norm_float16"], dtype=np.float32)
        margin = np.asarray(data["margin_float16"], dtype=np.float32)
        valid = np.asarray(data["valid_mask_bool"], dtype=bool)
        predicted_unmeasured = np.asarray(data["predicted_unmeasured_mask_bool"], dtype=bool)
        require(conf.shape == ent.shape == margin.shape == valid.shape == predicted_unmeasured.shape, "dense field shape mismatch")
        for name, arr in (("confidence", conf), ("entropy_norm", ent), ("margin", margin)):
            require(np.all(np.isfinite(arr)), f"{name} contains NaN/Inf")
        require(float(np.min(conf)) >= 0.0 and float(np.max(conf)) <= 1.0, "confidence outside [0,1]")
        require(float(np.min(ent)) >= 0.0 and float(np.max(ent)) <= 1.0005, "entropy_norm outside [0,1] tolerance")
        require(float(np.min(margin)) >= 0.0 and float(np.max(margin)) <= 1.0, "margin outside [0,1]")
        require(int(np.count_nonzero(valid)) > 0, "valid mask is empty")


def csv_row_count(path: Path) -> int:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def check_git_policy() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    tracked_forbidden = []
    for line in result.stdout.splitlines():
        path = line[3:] if len(line) > 3 else line
        if path.startswith(("outputs/", "logs/", "checkpoints/")):
            tracked_forbidden.append(line)
    return {"status": result.stdout, "forbidden_status_rows": tracked_forbidden}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rerun_audit_output_dir", type=Path, required=True)
    parser.add_argument("--stage4a68_dir", type=Path, required=True)
    parser.add_argument("--stage4a69_dir", type=Path, required=True)
    parser.add_argument("--stage4a610_limited_dir", type=Path, required=True)
    parser.add_argument("--fixed_usd", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected_logical_frame_count", type=int, default=30)
    parser.add_argument("--expected_candidate_rows", type=int, default=480)
    parser.add_argument("--allow_blocked_if_candidate_visibility_missing", action="store_true")
    parser.add_argument("--expect_no_isaac", action="store_true")
    parser.add_argument("--expect_no_capture", action="store_true")
    parser.add_argument("--expect_no_action", action="store_true")
    parser.add_argument("--expect_no_rollout", action="store_true")
    parser.add_argument("--expect_no_long_rollout", action="store_true")
    parser.add_argument("--expect_no_training", action="store_true")
    parser.add_argument("--expect_no_rl_gdpo", action="store_true")
    parser.add_argument("--expect_stage4a611_not_executed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir.resolve()
    rerun = args.rerun_audit_output_dir.resolve()
    require(out.is_dir(), f"primary output dir missing: {out}")
    require(rerun.is_dir(), f"rerun audit dir missing: {rerun}")

    required_primary = [
        "loaded_context_manifest.json",
        "loaded_stage4a610_limited_audit_manifest.json",
        "loaded_stage4a68_manifest.json",
        "loaded_stage4a69_manifest.json",
        "map_predict_dense_artifact_contract.json",
        "dense_prediction_class_mapping.json",
        "dense_regeneration_input_manifest.csv",
        "dense_prediction_artifact_manifest.csv",
        "dense_uncertainty_frame_summary.csv",
        "dense_artifact_safety_audit.json",
        "prediction_writeback_recheck.json",
        "no_isaac_report.json",
        "no_capture_report.json",
        "no_action_report.json",
        "no_rollout_report.json",
        "no_training_rl_bc_report.json",
        "source_hash_report.json",
        "checkpoint_hash_report.json",
        "prior_dataset_hash_report.json",
        "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json",
    ]
    for name in required_primary:
        require((out / name).is_file(), f"missing primary output {name}")

    summary = load_json(out / "stage4a610a_dense_prediction_uncertainty_artifacts_summary.json")
    require(int(summary["logical_frame_count"]) == int(args.expected_logical_frame_count), "logical frame count mismatch")
    require(int(summary["dense_artifacts_generated"]) == int(args.expected_logical_frame_count), "dense artifact count mismatch")
    require(int(summary["candidate_uncertainty_rows"]) == int(args.expected_candidate_rows), "candidate row count mismatch")
    require(summary["map_predict_dense_artifact_saving_updated"] is True, "contract update flag false")
    require(summary["dense_compact_probability_fields_generated"] is True, "compact field flag false")
    require(summary["full_class_prob_saved"] is False, "full class_prob should not be saved")

    for idx in range(int(args.expected_logical_frame_count)):
        assert_dense_npz(out / "dense_prediction_artifacts" / f"dense_prediction_uncertainty_{idx:03d}.npz")

    require(csv_row_count(out / "dense_regeneration_input_manifest.csv") == int(args.expected_logical_frame_count), "input manifest frame count mismatch")
    require(csv_row_count(out / "dense_prediction_artifact_manifest.csv") == int(args.expected_logical_frame_count), "dense manifest frame count mismatch")
    require((out / "candidate_visible_uncertainty_manifest.csv").is_file(), "candidate uncertainty manifest missing")
    require(csv_row_count(out / "dense_uncertainty_candidate_summary.csv") == int(args.expected_candidate_rows), "candidate summary row count mismatch")

    rerun_required = [
        "stage4a610_dense_rerun_summary.json",
        "prediction_artifact_inventory.csv",
        "uncertainty_available_fields_report.json",
        "uncertainty_formula_reference.json",
        "frame_uncertainty_summary.csv",
        "candidate_uncertainty_table.csv",
        "selected_action_uncertainty_audit.csv",
        "uncertainty_vs_source_occ_free_analysis.csv",
        "uncertainty_vs_branch_classification.csv",
        "frame1_vs_frame2_uncertainty_analysis.csv",
        "stage4a68_vs_stage4a69_uncertainty_comparison.csv",
        "uncertainty_shadow_score_audit.csv",
        "uncertainty_readiness_decision.json",
        "prediction_safety_recheck.json",
        "future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md",
    ]
    for name in rerun_required:
        require((rerun / name).is_file(), f"missing rerun output {name}")

    readiness = load_json(rerun / "uncertainty_readiness_decision.json")
    if readiness.get("candidate_level_uncertainty_ready"):
        require(readiness["candidate_level_uncertainty_ready"] is True, "candidate readiness inconsistent")
        require(readiness["uncertainty_aware_expert_pilot_ready"] is True, "expert pilot readiness should be true when candidate ready")
    else:
        require(args.allow_blocked_if_candidate_visibility_missing, "candidate uncertainty blocked but blocking not allowed")
        require((rerun / "dense_candidate_uncertainty_blocker.json").is_file(), "candidate blocker missing")

    safety = load_json(rerun / "prediction_safety_recheck.json")
    if args.expect_no_isaac:
        require(int(safety["isaac_startup_count_this_stage"]) == 0, "Isaac startup count not zero")
    if args.expect_no_capture:
        require(int(safety["capture_count_this_stage"]) == 0, "capture count not zero")
    if args.expect_no_action:
        require(int(safety["action_execution_count_this_stage"]) == 0, "action execution count not zero")
    if args.expect_no_rollout:
        require(safety["rollout_executed_this_stage"] is False, "rollout executed")
    if args.expect_no_long_rollout:
        require(safety["long_rollout_executed_this_stage"] is False, "long rollout executed")
    if args.expect_no_training:
        require(safety["training_run_this_stage"] is False, "training ran")
    if args.expect_no_rl_gdpo:
        require(safety["bc_il_rl_gdpo_ppo_run_this_stage"] is False, "BC/IL/RL/GDPO/PPO ran")
    if args.expect_stage4a611_not_executed:
        require(safety["stage4a611_executed"] is False, "Stage 4A-6.11 executed")
    require(safety["prediction_writeback"] is False, "prediction writeback true")
    require(safety["target_ground_truth_future_observed_scoring"] is False, "forbidden target/GT/future scoring")
    require(safety["prediction_traversability_collision_ray_candidate_edge_use"] is False, "forbidden prediction safety use")
    require(safety["passed"] is True, "prediction safety recheck failed")

    source = load_json(out / "source_hash_report.json")
    for key in ("source_usd", "fixed_usd", "source_observed_state"):
        require(source["before"][key]["sha256"] == source["after"][key]["sha256"], f"{key} hash changed")
    checkpoint = load_json(out / "checkpoint_hash_report.json")
    require(checkpoint["before"]["sha256"] == checkpoint["after"]["sha256"], "checkpoint hash changed")
    prior = load_json(out / "prior_dataset_hash_report.json")
    require(prior["prior_datasets_modified"] is False, "prior datasets modified")
    require(prior["stage4a610_output_modified"] is False, "old 6.10 output modified")

    policy = check_git_policy()
    require(not policy["forbidden_status_rows"], f"large artifact policy violation in git status: {policy['forbidden_status_rows']}")

    sketch = (rerun / "future_stage4a611_uncertainty_aware_lambda_pilot_sketch.md").read_text(encoding="utf-8")
    require(sketch.startswith("DO NOT RUN IN STAGE 4A-6.10a."), "future 6.11 sketch missing required prefix")

    result = {
        "all_passed": True,
        "logical_frame_count": summary["logical_frame_count"],
        "candidate_uncertainty_rows": summary["candidate_uncertainty_rows"],
        "candidate_level_uncertainty_ready": readiness["candidate_level_uncertainty_ready"],
        "uncertainty_aware_expert_pilot_ready": readiness["uncertainty_aware_expert_pilot_ready"],
        "git_policy_forbidden_status_rows": policy["forbidden_status_rows"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
