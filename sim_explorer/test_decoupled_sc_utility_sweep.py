#!/usr/bin/env python3
"""Validate Stage 4A-6.5z decoupled SC utility sweep outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "prediction_npz_field_inventory.json",
    "prediction_npz_field_inventory.md",
    "decoupled_sc_mapping_report.json",
    "decoupled_sc_mapping_report.md",
    "normalization_summary_by_seed_basis.csv",
    "normalization_summary_by_seed_basis.json",
    "adaptive_lambda_values.csv",
    "adaptive_lambda_values.json",
    "decoupled_candidate_topk.csv",
    "decoupled_candidate_topk.json",
    "decoupled_sc_sweep_decisions.csv",
    "decoupled_sc_sweep_decisions.json",
    "decoupled_sc_sweep_decisions.md",
    "branch_classification_by_formula_seed.csv",
    "branch_classification_by_formula_seed.json",
    "branch_classification_summary_by_formula.json",
    "branch_classification_summary_by_basis_variant.csv",
    "branch_classification_summary_by_basis_variant.json",
    "lambda_sweep_summary_by_basis_variant.csv",
    "lambda_sweep_summary_by_basis_variant.json",
    "seed0_base_gap_report.json",
    "seed0_base_gap_report.md",
    "safety_summary.json",
    "stage4a65z_decoupled_sc_utility_sweep_summary.json",
    "stage4a65z_decoupled_sc_utility_sweep_summary.md",
    "recommended_next_diagnostic_step.md",
    "decoupled_sc_sweep_manifest.jsonl",
    "fixed_lambda_seed0_sc_basin_fraction.png",
    "fixed_lambda_same_as_measured_fraction.png",
    "fixed_lambda_source_occ_free_heatmap.png",
    "adaptive_lambda_values.png",
    "seed0_source_occ_free_value_components.png",
    "selected_children_fixed_lambda_topdown.png",
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


def close(a: Any, b: Any, tol: float = 1.0e-6) -> bool:
    aa = as_float(a)
    bb = as_float(b)
    return math.isfinite(aa) and math.isfinite(bb) and abs(aa - bb) <= tol


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    for name in REQUIRED_FILES:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
        if path.suffix == ".png":
            assert_png(path)
    return {"passed": True, "required_files": len(REQUIRED_FILES)}


def test_sweep_content(output_dir: Path, expected_min_seeds: int) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65z_decoupled_sc_utility_sweep_summary.json")
    decisions = load_json(output_dir / "decoupled_sc_sweep_decisions.json")
    grouped = load_json(output_dir / "lambda_sweep_summary_by_basis_variant.json")
    adaptive = load_json(output_dir / "adaptive_lambda_values.json")
    gap = load_json(output_dir / "seed0_base_gap_report.json")

    seeds = summary["seeds"]
    bases = summary["sc_bases"]
    fixed = summary["fixed_lambdas"]
    scales = summary["adaptive_lambda_scales"]
    _assert(len(seeds) >= expected_min_seeds, f"expected at least {expected_min_seeds} seeds")
    _assert(fixed == [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0], "fixed lambdas changed")
    _assert(scales == [0.25, 0.5, 1.0, 2.0], "adaptive scales changed")
    expected_rows = len(seeds) * len(bases) * (len(fixed) + len(scales))
    _assert(len(decisions) == expected_rows, f"decision row count mismatch: {len(decisions)} != {expected_rows}")
    _assert(summary["decision_row_count"] == expected_rows, "summary decision row count mismatch")
    _assert(len(grouped) == len(bases) * (len(fixed) + len(scales)), "basis/lambda summary row count mismatch")
    _assert(len(adaptive) == len(seeds) * len(bases) * len(scales), "adaptive lambda row count mismatch")

    _assert(summary["diagnostic_status"].startswith("diagnostic/engineering"), "must be diagnostic status")
    _assert(summary["answers"]["source_faithful"] is False, "must not claim source-faithful")
    _assert(summary["answers"]["runtime_smoke_readiness"] is False, "runtime smoke must not be ready")
    _assert(summary["answers"]["rollout_readiness"] is False, "rollout must not be ready")
    _assert("gain_exp / cost + lambda * normalized_sc" in summary["utility_formula"], "wrong utility formula")

    for row in decisions:
        _assert(row["sc_inside_cost_division"] is False, "SC bonus was marked inside cost division")
        expected_value = as_float(row["base_exp_value"]) + as_float(row["lambda_value"]) * as_float(row["normalized_sc"])
        _assert(close(row["value"], expected_value, tol=1.0e-5), f"decoupled value mismatch for {row['formula']}")
        if row["lambda_family"] == "fixed" and row["lambda_label"] == "0":
            _assert(close(row["lambda_sc_bonus"], 0.0), "fixed lambda 0 should have no SC bonus")
            _assert(close(row["value"], row["base_exp_value"], tol=1.0e-5), "lambda 0 should equal base value")
        if row["lambda_family"] == "adaptive":
            _assert(close(row["lambda_value"], as_float(row["lambda_scale"]) * as_float(row["lambda_base"])), "adaptive lambda mismatch")

    seed0_source_l0 = summary["key_results"]["seed0_source_occ_free_fixed_lambda0"]
    _assert(
        seed0_source_l0["selected_child_grid"] == [17, 16, 11],
        f"seed0 source_occ_free fixed lambda 0 should select measured branch, got {seed0_source_l0}",
    )
    _assert(seed0_source_l0["classification"] != "exact_seed0_sc", "lambda 0 should not reproduce exact seed0 SC branch")
    _assert(as_float(gap["base_value_gap_measured_minus_sc"]) > 9.0, "seed0 base value gap should be order-10")
    _assert(close(gap["measured_branch"]["gain_exp"], 323.0), "measured gain_exp changed")
    _assert(close(gap["seed0_sc_branch"]["gain_exp"], 76.0), "seed0 SC gain_exp changed")
    _assert(
        close(gap["seed0_sc_branch"]["stage4a65x_context_source_occ_free_sc"], 135.0),
        "seed0 Stage 4A-6.5x source OCC+FREE SC reference changed",
    )

    return {
        "passed": True,
        "seed_count": len(seeds),
        "sc_bases": bases,
        "decision_rows": len(decisions),
    }


def test_safety(output_dir: Path) -> dict[str, Any]:
    safety = load_json(output_dir / "safety_summary.json")
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
        "sweep_content": test_sweep_content(output_dir, int(args.expected_min_seeds)),
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
