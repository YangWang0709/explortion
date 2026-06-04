#!/usr/bin/env python3
"""Validate Stage 4A-6.5z.1 decoupled SC signal-strength diagnosis outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "loaded_6p5z_inputs_manifest.json",
    "loaded_6p5z_inputs_manifest.md",
    "near_miss_branch_table.csv",
    "near_miss_branch_table.json",
    "near_miss_branch_summary.md",
    "required_lambda_to_flip.csv",
    "required_lambda_to_flip.json",
    "required_lambda_to_flip_summary.md",
    "normalization_diagnostics_by_seed.csv",
    "normalization_diagnostics_by_seed.json",
    "normalization_diagnostics_summary.md",
    "adaptive_lambda_gap_analysis.csv",
    "adaptive_lambda_gap_analysis.json",
    "adaptive_lambda_gap_analysis.md",
    "measured_vs_nonmeasured_sc_rank.csv",
    "measured_vs_nonmeasured_sc_rank.json",
    "measured_vs_nonmeasured_sc_rank.md",
    "impossible_under_positive_lambda.csv",
    "impossible_under_positive_lambda.json",
    "low_cost_artifact_followup.csv",
    "low_cost_artifact_followup.json",
    "low_cost_artifact_followup.md",
    "debug_tree_regeneration_report.json",
    "debug_tree_regeneration_report.md",
    "missing_fields_report.json",
    "stage4a65z1_decoupled_signal_strength_summary.json",
    "stage4a65z1_decoupled_signal_strength_summary.md",
    "recommended_next_faithful_step.md",
    "required_lambda_distribution.png",
    "required_lambda_by_seed.png",
    "adaptive_lambda_vs_required_lambda.png",
    "measured_vs_best_nonmeasured_base_exp.png",
    "measured_vs_best_nonmeasured_normalized_sc.png",
    "measured_vs_best_nonmeasured_final_value_gap.png",
    "normalized_sc_distribution_by_seed.png",
    "sc_rank_of_measured_winner.png",
    "near_miss_topdown.png",
    "value_component_near_miss_stack.png",
]

FORBIDDEN_PATTERNS = [
    "frame*_rgb.png",
    "frame*_depth.npy",
    "frame*_depth.png",
    "observed_state*.npy",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "sscnet_*",
    "map_predict*",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_topdown_path.png",
    "rollout_*.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_png(path: Path) -> None:
    data = path.read_bytes()
    _assert(len(data) > 8, f"empty PNG: {path}")
    _assert(data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}")


def as_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    for name in REQUIRED_FILES:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
        if path.suffix == ".png":
            assert_png(path)
    return {"passed": True, "required_files": len(REQUIRED_FILES)}


def test_content(output_dir: Path, expected_min_seeds: int) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65z1_decoupled_signal_strength_summary.json")
    manifest = load_json(output_dir / "loaded_6p5z_inputs_manifest.json")
    near_rows = load_json(output_dir / "near_miss_branch_table.json")
    required_rows = load_json(output_dir / "required_lambda_to_flip.json")
    norm_rows = load_json(output_dir / "normalization_diagnostics_by_seed.json")
    adaptive_rows = load_json(output_dir / "adaptive_lambda_gap_analysis.json")
    rank_rows = load_json(output_dir / "measured_vs_nonmeasured_sc_rank.json")
    impossible_rows = load_json(output_dir / "impossible_under_positive_lambda.json")
    regen = load_json(output_dir / "debug_tree_regeneration_report.json")
    missing = load_json(output_dir / "missing_fields_report.json")

    seeds = summary["seeds"]
    bases = summary["sc_bases"]
    _assert(len(seeds) >= expected_min_seeds, f"expected at least {expected_min_seeds} seeds")
    _assert(summary["fixed_lambdas"] == [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0], "fixed lambdas changed")
    _assert(summary["adaptive_lambda_scales"] == [0.25, 0.5, 1.0, 2.0], "adaptive scales changed")
    _assert(summary["diagnostic_status"] == "offline-only table/saved-tree diagnosis", "wrong diagnostic status")
    _assert(summary["answers"]["runtime_two_frame_readiness"] is False, "runtime two-frame must not be ready")
    _assert(summary["answers"]["rollout_readiness"] is False, "rollout must not be ready")
    _assert(summary["answers"]["coverage_improvement_claimed"] is False, "must not claim coverage improvement")
    _assert(regen["recomputed_debug_tree_nodes"] is True, "debug tree rows should be regenerated")
    _assert(regen["candidate_path_rows"] > 0, "no regenerated candidate rows")

    _assert(manifest["loaded_count"] > 0, "no 6.5z inputs loaded")
    _assert("missing_fields" in missing, "missing fields report malformed")
    _assert(len(near_rows) > 0, "near-miss table empty")
    _assert(len(required_rows) > 0, "required-lambda table empty")
    _assert(len(norm_rows) == len(seeds) * len(bases), "normalization row count mismatch")
    _assert(len(adaptive_rows) == len(seeds) * len(bases), "adaptive gap row count mismatch")
    _assert(len(rank_rows) == len(norm_rows), "rank row count mismatch")
    _assert(len(impossible_rows) <= len(required_rows), "impossible rows exceed required rows")

    finite_required = [
        as_float(row.get("required_lambda_to_beat_measured"))
        for row in required_rows
        if row.get("required_lambda_to_beat_measured") is not None
    ]
    _assert(finite_required or impossible_rows, "required-lambda diagnosis produced neither finite nor impossible rows")
    if finite_required:
        _assert(min(finite_required) >= 0.0, "required lambda should be non-negative")
    _assert(
        summary["stage4a65z_input_validation"]["stage4a65y_seed0_confidence_reproduces_reference"] is True,
        "Stage 4A-6.5y seed0 confidence reference not confirmed",
    )
    _assert(
        summary["stage4a65z_input_validation"]["stage4a65y_source_occ_free_seed0_reproduces_reference"] is True,
        "Stage 4A-6.5y seed0 source OCC+FREE reference not confirmed",
    )
    _assert(
        summary["stage4a65z_input_validation"]["seed0_base_gap_measured_minus_sc"] is not None,
        "seed0 base gap missing",
    )
    return {
        "passed": True,
        "seed_count": len(seeds),
        "sc_bases": bases,
        "near_miss_rows": len(near_rows),
        "required_rows": len(required_rows),
        "normalization_rows": len(norm_rows),
    }


def test_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65z1_decoupled_signal_strength_summary.json")
    safety = summary["safety"]
    false_keys = [
        "isaac_startup",
        "new_capture",
        "map_predict_rerun",
        "sscnet_inference",
        "selected_action_execution",
        "two_frame_runtime",
        "rollout",
        "open_ended_loop",
        "training_or_rl",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_npz_modified",
        "pose_json_modified",
        "camera_info_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_scoring",
        "external_source_modified_or_built",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        _assert(not bool(safety.get(key)), f"safety flag should be false: {key}")
    _assert(safety["observed_state_sha256_before"] == safety["observed_state_sha256_after"], "observed_state changed")
    _assert(safety["prediction_npz_sha256_before"] == safety["prediction_npz_sha256_after"], "prediction NPZ changed")
    _assert(safety["pose_json_sha256_before"] == safety["pose_json_sha256_after"], "pose JSON changed")
    _assert(safety["camera_info_sha256_before"] == safety["camera_info_sha256_after"], "camera info changed")
    if safety.get("checkpoint_sha256_before") is not None:
        _assert(safety["checkpoint_sha256_before"] == safety["checkpoint_sha256_after"], "checkpoint changed")
    _assert(not safety.get("prohibited_artifacts_in_output"), "summary recorded prohibited output artifacts")
    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.rglob(pattern))
        _assert(not matches, f"forbidden artifacts for pattern {pattern}: {matches[:5]}")
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "content": test_content(output_dir, int(args.expected_min_seeds)),
        "safety": test_safety(output_dir),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_min_seeds", type=int, default=10)
    return parser


def main() -> None:
    run_tests(build_argparser().parse_args())


if __name__ == "__main__":
    main()
