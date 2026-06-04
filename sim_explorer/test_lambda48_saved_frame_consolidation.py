#!/usr/bin/env python3
"""Validate Stage 4A-6.5af lambda48 consolidation outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "loaded_inputs_manifest.json",
    "loaded_inputs_manifest.md",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "unified_config_table.csv",
    "unified_config_table.json",
    "unified_config_table.md",
    "lambda48_cross_frame_summary.csv",
    "lambda48_cross_frame_summary.json",
    "lambda48_cross_frame_summary.md",
    "real_frame_aggregate_lambda48.csv",
    "real_frame_aggregate_lambda48.json",
    "real_frame_aggregate_lambda48.md",
    "lambda32_vs_lambda48_comparison.csv",
    "lambda32_vs_lambda48_comparison.json",
    "lambda32_vs_lambda48_comparison.md",
    "over_cost_diagnostic_comparison.csv",
    "over_cost_diagnostic_comparison.json",
    "over_cost_diagnostic_comparison.md",
    "low_cost_artifact_cross_frame.csv",
    "low_cost_artifact_cross_frame.json",
    "low_cost_artifact_cross_frame.md",
    "readiness_matrix.csv",
    "readiness_matrix.json",
    "readiness_matrix.md",
    "design_review_findings.json",
    "design_review_findings.md",
    "stage4a65af_lambda48_consolidation_summary.json",
    "stage4a65af_lambda48_consolidation_summary.md",
    "recommended_next_faithful_step.md",
]

PLOT_FILES = [
    "lambda48_real_frame_branch_fractions.png",
    "lambda48_synthetic_vs_real_summary.png",
    "lambda32_vs_lambda48_branch_comparison.png",
    "over_cost_vs_lambda48_comparison.png",
    "low_cost_artifact_cross_frame.png",
    "readiness_matrix_heatmap.png",
    "healthy_nonmeasured_fraction_by_frame.png",
    "same_as_measured_fraction_by_frame.png",
]

FORBIDDEN_PATTERNS = [
    "depth_*.npy",
    "depth_*.png",
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
    "observed_state*.npy",
    "episode_summary.json",
    "rollout_topdown_path.png",
    "observed_ratio_curve.png",
    "rollout_index.html",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(value: float | None, expected: float, tol: float = 1e-9) -> None:
    assert value is not None, f"missing value, expected {expected}"
    assert abs(float(value) - expected) <= tol, f"expected {expected}, got {value}"


def assert_png_or_reason(output_dir: Path, name: str) -> None:
    path = output_dir / name
    if path.is_file():
        data = path.read_bytes()
        assert len(data) > 8, f"empty plot: {path}"
        assert data[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"
        return
    reason = output_dir / f"{Path(name).stem}_skipped_reason.md"
    assert reason.is_file(), f"missing plot or skipped reason: {name}"


def validate_required_outputs(output_dir: Path) -> dict[str, Any]:
    assert output_dir.is_dir(), f"missing output dir: {output_dir}"
    for name in REQUIRED_FILES:
        path = output_dir / name
        assert path.is_file(), f"missing required file: {path}"
        assert path.stat().st_size > 0, f"empty required file: {path}"
    for name in PLOT_FILES:
        assert_png_or_reason(output_dir, name)
    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.rglob(pattern))
        assert not matches, f"forbidden runtime/capture artifact written: {pattern}: {matches[:3]}"
    return {"required_files": len(REQUIRED_FILES), "plots_checked": len(PLOT_FILES)}


def validate_tables(output_dir: Path) -> dict[str, Any]:
    unified = load_json(output_dir / "unified_config_table.json")
    cross = load_json(output_dir / "lambda48_cross_frame_summary.json")
    aggregate = load_json(output_dir / "real_frame_aggregate_lambda48.json")[0]
    lambda_cmp = load_json(output_dir / "lambda32_vs_lambda48_comparison.json")
    over_cmp = load_json(output_dir / "over_cost_diagnostic_comparison.json")
    readiness = load_json(output_dir / "readiness_matrix.json")
    low_cost = load_json(output_dir / "low_cost_artifact_cross_frame.json")
    summary = load_json(output_dir / "stage4a65af_lambda48_consolidation_summary.json")
    findings = load_json(output_dir / "design_review_findings.json")
    missing = load_json(output_dir / "missing_fields_report.json")
    manifest = load_json(output_dir / "loaded_inputs_manifest.json")

    stages = {row["stage"] for row in unified}
    for stage in ("Stage 4A-6.5ab", "Stage 4A-6.5ac", "Stage 4A-6.5ad", "Stage 4A-6.5ae"):
        assert stage in stages, f"unified table missing {stage}"
    modes = {row["mode"] for row in unified}
    for mode in ("measured_only", "map_predict_lambda32", "map_predict_lambda48", "source_occ_free_over_cost"):
        assert mode in modes, f"unified table missing mode {mode}"

    by_frame = {row["frame_id"]: row for row in cross}
    assert_close(by_frame["synthetic_000"]["hidden_room_fraction"], 1.0)
    assert_close(by_frame["synthetic_000"]["oracle_agreement"], 1.0)
    assert_close(by_frame["synthetic_000"]["low_cost_artifact_fraction"], 0.0)
    assert_close(by_frame["stage4a65p_frame002"]["same_as_measured_fraction"], 0.6)
    assert_close(by_frame["stage4a65p_frame002"]["healthy_nonmeasured_fraction"], 0.4)
    assert_close(by_frame["stage4a65p_frame002"]["historical_prior_basin_fraction"], 0.0)
    assert_close(by_frame["stage4a65p_frame001"]["same_as_measured_fraction"], 0.8)
    assert_close(by_frame["stage4a65p_frame001"]["healthy_nonmeasured_fraction"], 0.2)
    assert_close(by_frame["stage4a65p_frame001"]["historical_prior_basin_fraction"], 0.0)

    assert aggregate["total_seed_frame_rows"] == 20
    assert aggregate["same_as_measured_count"] == 14
    assert aggregate["distinct_nonmeasured_count"] == 6
    assert aggregate["healthy_nonmeasured_count"] == 6
    assert aggregate["historical_prior_basin_count"] == 0
    assert aggregate["low_cost_artifact_count"] == 0
    assert_close(aggregate["same_as_measured_fraction"], 0.7)
    assert_close(aggregate["healthy_nonmeasured_fraction"], 0.3)
    assert_close(aggregate["low_cost_artifact_fraction"], 0.0)

    real_lambda_cmp = [row for row in lambda_cmp if row["stage"] == "real_aggregate"][0]
    assert real_lambda_cmp["seed_count"] == 20
    assert_close(real_lambda_cmp["branch_class_match_fraction"], 1.0)
    assert real_lambda_cmp["lambda48_changed_branch_class_beyond_lambda32"] is False
    assert real_lambda_cmp["lambda48_changed_exact_selection_beyond_lambda32"] is True

    frame2_over = [
        row
        for row in over_cmp
        if row["frame_id"] == "stage4a65p_frame002" and row["mode"] == "source_occ_free_over_cost"
    ][0]
    assert_close(frame2_over["prior_basin_fraction"], 0.5)
    assert frame2_over["remains_risky"] is True
    assert frame2_over["diagnostic_only"] is True
    over_aggregate = [row for row in over_cmp if row["stage"] == "real_aggregate"][0]
    assert over_aggregate["remains_risky"] is True
    assert over_aggregate["recommendation"] == "keep diagnostic-only; do not use for runtime"

    lambda48_low = [row for row in low_cost if row["mode"] == "map_predict_lambda48"]
    assert len(lambda48_low) == 3
    for row in lambda48_low:
        assert_close(row["low_cost_artifact_fraction"], 0.0)

    readiness_by_case = {row["case"]: row for row in readiness}
    assert readiness_by_case["real aggregate lambda48"]["saved_frame_ready"] == "yes"
    assert readiness_by_case["real aggregate lambda48"]["runtime_smoke_ready"] == "no"
    assert readiness_by_case["real aggregate lambda48"]["rollout_ready"] == "no"
    assert readiness_by_case["over-cost diagnostic"]["diagnostic_only"] == "yes"

    assert summary["status"] == "completed"
    assert summary["coverage_improvement_claimed"] is False
    assert summary["readiness"]["saved_frame_only_ready"] is True
    assert summary["readiness"]["runtime_smoke_ready"] is False
    assert summary["readiness"]["rollout_ready"] is False
    for key, value in summary["safety"].items():
        assert value is False, f"safety flag should be false: {key}"

    recommendation = findings["recommendation"]["exactly_one_next_small_task"]
    assert "offline saved-frame-only multi-frame lambda48 replay" in recommendation
    assert findings["questions"]["can_enter_runtime_or_rollout"]["answer"] == "no"
    assert findings["questions"]["can_enter_saved_multi_frame_replay"]["answer"].startswith("yes")
    assert missing["nonessential_missing_files_allowed"] is True
    assert len(manifest["stages"]) == 4

    csv_rows = read_csv_rows(output_dir / "unified_config_table.csv")
    assert len(csv_rows) == len(unified), "unified CSV/JSON row mismatch"
    return {
        "unified_rows": len(unified),
        "cross_rows": len(cross),
        "real_seed_frame_rows": aggregate["total_seed_frame_rows"],
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result = {
        "required_outputs": validate_required_outputs(output_dir),
        "tables": validate_tables(output_dir),
        "status": "passed",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
