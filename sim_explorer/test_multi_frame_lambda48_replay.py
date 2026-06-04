#!/usr/bin/env python3
"""Validate Stage 4A-6.5ag multi-frame lambda48 replay outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "loaded_context_manifest.json",
    "loaded_context_manifest.md",
    "frame_discovery_inventory.csv",
    "frame_discovery_inventory.json",
    "frame_discovery_inventory.md",
    "frame_inventory_duplicates.csv",
    "frame_inventory_duplicates.json",
    "selected_frame_manifest.csv",
    "selected_frame_manifest.json",
    "selected_frame_manifest.md",
    "skipped_frame_candidates.csv",
    "skipped_frame_candidates.json",
    "formula_definition.json",
    "formula_definition.md",
    "root_alignment_report.csv",
    "root_alignment_report.json",
    "root_alignment_report.md",
    "per_frame_seed_mode_decisions.csv",
    "per_frame_seed_mode_decisions.json",
    "per_frame_seed_mode_decisions.md",
    "per_frame_value_components.csv",
    "per_frame_value_components.json",
    "branch_classification_by_frame_seed_mode.csv",
    "branch_classification_by_frame_seed_mode.json",
    "branch_classification_summary.md",
    "lambda48_multiframe_summary.csv",
    "lambda48_multiframe_summary.json",
    "lambda48_multiframe_summary.md",
    "lambda32_vs_lambda48_multiframe.csv",
    "lambda32_vs_lambda48_multiframe.json",
    "lambda32_vs_lambda48_multiframe.md",
    "over_cost_multiframe_diagnostic.csv",
    "over_cost_multiframe_diagnostic.json",
    "over_cost_multiframe_diagnostic.md",
    "low_cost_artifact_multiframe.csv",
    "low_cost_artifact_multiframe.json",
    "low_cost_artifact_multiframe.md",
    "prediction_safety_report.json",
    "prediction_safety_report.md",
    "hash_checks.json",
    "missing_fields_report.json",
    "missing_fields_report.md",
    "stage4a65ag_multi_frame_lambda48_replay_summary.json",
    "stage4a65ag_multi_frame_lambda48_replay_summary.md",
    "recommended_next_faithful_step.md",
]

PLOT_FILES = [
    "lambda48_branch_fraction_by_frame.png",
    "lambda48_aggregate_branch_fractions.png",
    "healthy_nonmeasured_fraction_by_frame.png",
    "same_as_measured_fraction_by_frame.png",
    "low_cost_artifact_by_frame.png",
    "prior_basin_fraction_by_frame.png",
    "lambda32_vs_lambda48_multiframe.png",
    "over_cost_vs_lambda48_multiframe.png",
    "margin_by_frame_lambda48.png",
    "source_occ_free_by_branch_class.png",
]

FORBIDDEN_PATTERNS = [
    "depth_*.npy",
    "depth_*.png",
    "rgb_*.png",
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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def test_content(output_dir: Path, expected_min_unique_frames: int, expected_min_seeds: int) -> dict[str, Any]:
    context = load_json(output_dir / "loaded_context_manifest.json")
    inventory = load_json(output_dir / "frame_discovery_inventory.json")
    selected = load_json(output_dir / "selected_frame_manifest.json")
    duplicates = load_json(output_dir / "frame_inventory_duplicates.json")
    skipped = load_json(output_dir / "skipped_frame_candidates.json")
    formula = load_json(output_dir / "formula_definition.json")
    root_alignment = load_json(output_dir / "root_alignment_report.json")
    decisions = load_json(output_dir / "per_frame_seed_mode_decisions.json")
    values = load_json(output_dir / "per_frame_value_components.json")
    branches = load_json(output_dir / "branch_classification_by_frame_seed_mode.json")
    lambda_summary = load_json(output_dir / "lambda48_multiframe_summary.json")
    lambda32 = load_json(output_dir / "lambda32_vs_lambda48_multiframe.json")
    over_cost = load_json(output_dir / "over_cost_multiframe_diagnostic.json")
    low_cost = load_json(output_dir / "low_cost_artifact_multiframe.json")
    missing = load_json(output_dir / "missing_fields_report.json")
    final_summary = load_json(output_dir / "stage4a65ag_multi_frame_lambda48_replay_summary.json")

    assert context["stage"] == "Stage 4A-6.5ag"
    assert context["safety_scope"]["offline_saved_frame_only"] is True
    assert context["safety_scope"]["isaac_startup"] is False
    assert inventory, "frame inventory missing rows"
    assert len(selected) >= expected_min_unique_frames, (
        f"expected >= {expected_min_unique_frames} unique frames, got {len(selected)}"
    )
    assert isinstance(duplicates, list), "duplicate frame handling missing"
    assert isinstance(skipped, list), "skipped frame report missing"

    for frame in selected:
        blob = json.dumps(frame).lower()
        assert "synthetic" not in blob, f"synthetic frame selected: {frame}"
        for key in ("observed_state_path", "prediction_npz_path", "pose_json_path", "camera_info_json_path"):
            assert Path(frame[key]).is_file(), f"selected frame missing {key}: {frame[key]}"

    assert formula["recommended_formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)"
    assert formula["diagnostic_only"]["source_occ_free_over_cost"] == "(gain_exp + source_occ_free) / cost"
    assert formula["prediction_information_gain_only"] is True

    seeds = {int(row["seed"]) for row in decisions}
    modes = {row["mode"] for row in decisions}
    assert len(seeds) >= expected_min_seeds, f"expected >= {expected_min_seeds} seeds, got {len(seeds)}"
    for mode in (
        "measured_only",
        "map_predict_lambda48",
        "map_predict_lambda32",
        "source_occ_free_over_cost",
    ):
        assert mode in modes, f"required mode missing: {mode}"
    for optional in ("raw_hybrid_over_cost", "source_occ_free_no_cost"):
        assert optional in modes, f"optional diagnostic mode missing: {optional}"
    assert len(decisions) >= len(selected) * expected_min_seeds * 4, "not enough decision rows"
    assert len(read_csv(output_dir / "per_frame_seed_mode_decisions.csv")) == len(decisions)

    map48 = [row for row in decisions if row["mode"] == "map_predict_lambda48"]
    assert map48, "map_predict lambda48 rows missing"
    for row in map48:
        for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value"):
            assert key in row and row[key] is not None, f"lambda48 missing formula component: {key}"
        assert "minmax(source_occ_free)" in row["formula"], "lambda48 formula missing minmax"
        assert row["formula"] != "(gain_exp + source_occ_free) / cost", "lambda48 used over-cost formula"
        flags = row["prediction_safety_flags"]
        assert flags["prediction_writeback"] is False
        assert flags["prediction_used_for_traversability"] is False
        assert flags["prediction_used_for_collision"] is False
        assert flags["prediction_ray_blocking"] is False
        assert flags["target_ground_truth_planning_scoring"] is False

    assert root_alignment, "root alignment rows missing"
    assert values, "value component rows missing"
    assert branches, "branch rows missing"
    assert lambda_summary["aggregate"]["formula_components_logged"] is True
    assert lambda_summary["aggregate"]["runtime_smoke_readiness"] is False
    assert lambda_summary["aggregate"]["rollout_readiness"] is False
    assert lambda32["summary"]["row_count"] >= len(selected) * expected_min_seeds
    assert over_cost["summary"]["diagnostic_only"] is True
    assert low_cost["summary"]["map_predict_lambda48_row_count"] >= len(selected) * expected_min_seeds
    assert "required_plots" in missing

    answers = final_summary["answers"]
    assert answers["unique_real_medium_frames_selected"] == len(selected)
    assert answers["no_isaac_no_capture_no_map_predict_rerun"] is True
    assert answers["runtime_smoke_readiness"] is False
    assert answers["rollout_readiness"] is False
    assert final_summary["coverage_improvement_claimed"] is False
    return {
        "passed": True,
        "unique_frames": len(selected),
        "candidate_rows": len(inventory),
        "duplicate_rows": len(duplicates),
        "decision_rows": len(decisions),
        "modes": sorted(modes),
    }


def test_safety(output_dir: Path) -> dict[str, Any]:
    safety = load_json(output_dir / "prediction_safety_report.json")
    hash_checks = load_json(output_dir / "hash_checks.json")
    false_keys = [
        "isaac_startup",
        "new_capture",
        "map_predict_rerun",
        "sscnet_inference",
        "selected_action_execution",
        "two_frame_runtime",
        "rollout",
        "open_ended_loop",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "existing_observed_state_modified",
        "prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_used_for_traversability",
        "prediction_used_for_collision",
        "prediction_ray_blocking",
        "target_ground_truth_planning_scoring",
        "future_observed_planning_scoring",
        "external_source_modified_built",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        assert not bool(safety.get(key)), f"safety flag should be false: {key}"
    assert safety["runtime_smoke_ready"] is False
    assert safety["rollout_ready"] is False
    assert hash_checks["checkpoint"]["unchanged"] is True
    for item in hash_checks["frames"]:
        for key in ("observed_state", "prediction_npz", "pose_json", "camera_info_json"):
            assert item[key]["unchanged"] is True, f"input hash changed: {item['frame_id']} {key}"
    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.rglob(pattern))
        assert not matches, f"forbidden artifact pattern {pattern}: {matches[:5]}"
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "content": test_content(
            output_dir,
            int(args.expected_min_unique_frames),
            int(args.expected_min_seeds),
        ),
        "safety": test_safety(output_dir),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_min_unique_frames", type=int, default=2)
    parser.add_argument("--expected_min_seeds", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
