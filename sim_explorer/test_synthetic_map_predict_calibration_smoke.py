#!/usr/bin/env python3
"""Validate Stage 4A-6.5ab synthetic map_predict calibration smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "loaded_stage4a65aa_manifest.json",
    "loaded_stage4a65aa_manifest.md",
    "calibration_sweep_manifest.jsonl",
    "calibration_formula_definitions.json",
    "calibration_formula_definitions.md",
    "skipped_combinations.json",
    "per_seed_calibration_decisions.csv",
    "per_seed_calibration_decisions.json",
    "per_seed_calibration_decisions.md",
    "calibration_summary_by_config.csv",
    "calibration_summary_by_config.json",
    "calibration_summary_by_config.md",
    "threshold_sensitivity_summary.csv",
    "threshold_sensitivity_summary.json",
    "threshold_sensitivity_summary.md",
    "lambda_sensitivity_summary.csv",
    "lambda_sensitivity_summary.json",
    "lambda_sensitivity_summary.md",
    "oracle_map_predict_agreement.csv",
    "oracle_map_predict_agreement.json",
    "oracle_map_predict_agreement.md",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.json",
    "low_cost_artifact_diagnosis.md",
    "hidden_region_signal_summary.csv",
    "hidden_region_signal_summary.json",
    "hidden_region_signal_summary.md",
    "best_config_candidates.json",
    "best_config_candidates.md",
    "missing_fields_report.json",
    "stage4a65ab_synthetic_calibration_summary.json",
    "stage4a65ab_synthetic_calibration_summary.md",
    "recommended_next_faithful_step.md",
]

PLOT_FILES = [
    "hidden_room_selection_by_config.png",
    "oracle_vs_map_predict_agreement.png",
    "tau_sensitivity_hidden_room_fraction.png",
    "lambda_sensitivity_hidden_room_fraction.png",
    "low_cost_artifact_fraction.png",
    "selected_branch_topdown_by_best_config.png",
    "hidden_region_signal_by_threshold.png",
    "value_component_stack_best_configs.png",
    "map_predict_vs_oracle_prediction_overlay_topdown.png",
]

FORBIDDEN_PATTERNS = [
    "depth_*.npy",
    "rgb_*.png",
    "pose_*.json",
    "global_prediction_layer.npz",
    "local_prediction.npz",
    "sscnet_depth_input.npy",
    "sscnet_position.npy",
    "sscnet_input_debug.npz",
    "valid_position_mask.npy",
    "transitions.jsonl",
    "step_*.npz",
    "episode_summary.json",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assert_png_or_reason(output_dir: Path, name: str) -> None:
    path = output_dir / name
    if path.is_file():
        data = path.read_bytes()
        assert len(data) > 8, f"empty plot: {path}"
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
        return
    reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
    assert reason.is_file(), f"missing plot or skipped reason: {name}"


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_FILES:
        path = output_dir / name
        assert path.is_file(), f"missing required file: {path}"
        assert path.stat().st_size > 0, f"empty required file: {path}"
    for name in PLOT_FILES:
        assert_png_or_reason(output_dir, name)
    return {"passed": True, "required_files": len(REQUIRED_FILES), "plots": len(PLOT_FILES)}


def test_content(output_dir: Path, expected_min_seeds: int) -> dict[str, Any]:
    manifest = load_json(output_dir / "loaded_stage4a65aa_manifest.json")
    summary = load_json(output_dir / "stage4a65ab_synthetic_calibration_summary.json")
    decisions = load_json(output_dir / "per_seed_calibration_decisions.json")
    config_summary = load_json(output_dir / "calibration_summary_by_config.json")
    threshold = load_json(output_dir / "threshold_sensitivity_summary.json")
    lambdas = load_json(output_dir / "lambda_sensitivity_summary.json")
    agreement = load_json(output_dir / "oracle_map_predict_agreement.json")
    best = load_json(output_dir / "best_config_candidates.json")

    assert manifest["observed_state"]["exists"] if "exists" in manifest["observed_state"] else True
    assert Path(manifest["observed_state"]["path"]).is_file(), "loaded observed_state path missing"
    assert Path(manifest["oracle_prediction"]["path"]).is_file(), "loaded oracle prediction path missing"
    if manifest["map_predict_available"]:
        assert Path(manifest["map_predict_prediction"]["path"]).is_file(), "loaded map_predict prediction path missing"

    seeds = {int(row["seed"]) for row in decisions}
    assert len(seeds) >= expected_min_seeds, f"expected >= {expected_min_seeds} seeds, got {len(seeds)}"
    assert any(row["prediction_source"] == "none" and row["formula_name"] == "measured_only" for row in decisions)
    assert any(row["prediction_source"] == "oracle" for row in decisions), "oracle configs missing"
    if manifest["map_predict_available"]:
        assert any(row["prediction_source"] == "map_predict" for row in decisions), "map_predict configs missing"
    else:
        assert (output_dir / "map_predict_missing_skipped_reason.md").is_file(), "map_predict skip reason missing"

    formulas = {row["formula_name"] for row in decisions}
    assert "source_occ_free_over_cost" in formulas, "source_occ_free over_cost missing"
    assert any("decoupled_minmax_lambda32" in name for name in formulas), "decoupled lambda32 missing"
    assert any("decoupled_minmax_lambda48" in name for name in formulas), "decoupled lambda48 missing"
    assert threshold, "threshold sensitivity summary empty"
    assert lambdas, "lambda sensitivity summary empty"
    assert agreement or not manifest["map_predict_available"], "oracle/map_predict agreement empty"
    assert config_summary, "config summary empty"
    assert "candidate_count" in best, "best candidate file malformed"

    answers = summary["answers"]
    assert answers["loaded_stage4a65aa_outputs"] is True
    assert answers["no_isaac_no_capture_no_map_predict_rerun"] is True
    assert answers["seed_count"] >= expected_min_seeds
    assert answers["rollout_readiness"] is False
    assert summary["coverage_improvement_claimed"] is False
    return {
        "passed": True,
        "seed_count": len(seeds),
        "decision_rows": len(decisions),
        "config_rows": len(config_summary),
        "best_candidate_count": best["candidate_count"],
    }


def test_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65ab_synthetic_calibration_summary.json")
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
        "existing_observed_state_modified",
        "prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_planning_scoring",
        "external_source_modified_or_built",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        assert not bool(safety.get(key)), f"safety flag should be false: {key}"
    assert safety["observed_state_sha256_before"] == safety["observed_state_sha256_after"], "observed_state changed"
    assert safety["oracle_npz_sha256_before"] == safety["oracle_npz_sha256_after"], "oracle NPZ changed"
    if safety.get("map_predict_npz_sha256_before") is not None:
        assert safety["map_predict_npz_sha256_before"] == safety["map_predict_npz_sha256_after"], "map_predict NPZ changed"

    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.rglob(pattern))
        assert not matches, f"forbidden artifact pattern {pattern}: {matches[:5]}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_min_seeds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
