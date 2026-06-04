#!/usr/bin/env python3
"""Validate Stage 4A-6.5ad real-frame lambda48 formula smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "loaded_inputs_manifest.json",
    "loaded_inputs_manifest.md",
    "formula_definition.json",
    "formula_definition.md",
    "real_frame_reference_branches.json",
    "real_frame_reference_branches.md",
    "per_seed_mode_decisions.csv",
    "per_seed_mode_decisions.json",
    "per_seed_mode_decisions.md",
    "per_seed_value_components.csv",
    "per_seed_value_components.json",
    "branch_classification_by_seed_mode.csv",
    "branch_classification_by_seed_mode.json",
    "branch_classification_summary.md",
    "lambda48_behavior_summary.json",
    "lambda48_behavior_summary.md",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.json",
    "low_cost_artifact_diagnosis.md",
    "comparison_to_stage4a65z_z1.json",
    "comparison_to_stage4a65z_z1.md",
    "prediction_safety_report.json",
    "prediction_safety_report.md",
    "hash_checks.json",
    "missing_fields_report.json",
    "stage4a65ad_real_frame_lambda48_formula_summary.json",
    "stage4a65ad_real_frame_lambda48_formula_summary.md",
    "recommended_next_faithful_step.md",
]

PLOT_FILES = [
    "selected_branches_topdown_real_frame.png",
    "measured_vs_lambda48_topdown.png",
    "branch_classification_bar.png",
    "value_components_lambda48_real_frame.png",
    "source_occ_free_rank_by_mode.png",
    "low_cost_artifact_by_mode.png",
    "margin_by_mode.png",
    "prior_sc_basin_distance_by_mode.png",
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


def test_content(output_dir: Path, expected_min_seeds: int) -> dict[str, Any]:
    manifest = load_json(output_dir / "loaded_inputs_manifest.json")
    formula = load_json(output_dir / "formula_definition.json")
    refs = load_json(output_dir / "real_frame_reference_branches.json")
    decisions = load_json(output_dir / "per_seed_mode_decisions.json")
    value_components = load_json(output_dir / "per_seed_value_components.json")
    branches = load_json(output_dir / "branch_classification_by_seed_mode.json")
    lambda_summary = load_json(output_dir / "lambda48_behavior_summary.json")
    low_cost = load_json(output_dir / "low_cost_artifact_diagnosis.json")
    comparison = load_json(output_dir / "comparison_to_stage4a65z_z1.json")
    final_summary = load_json(output_dir / "stage4a65ad_real_frame_lambda48_formula_summary.json")

    assert Path(manifest["observed_state"]["path"]).is_file(), "loaded observed_state path missing"
    assert Path(manifest["prediction_npz"]["path"]).is_file(), "loaded prediction path missing"
    assert Path(manifest["pose_json"]["path"]).is_file(), "loaded pose path missing"
    assert Path(manifest["camera_info_json"]["path"]).is_file(), "loaded camera path missing"
    assert manifest["prediction_npz"]["shape_aligned_to_observed_state"] is True
    assert manifest["no_isaac_startup"] is True
    assert manifest["no_new_capture"] is True
    assert manifest["no_map_predict_rerun"] is True

    assert formula["recommended_formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)"
    assert formula["diagnostic_only"]["source_occ_free_over_cost"] == "(gain_exp + source_occ_free) / cost"
    assert refs["measured_reference"]["selected_child_id"] == "n0001"
    assert refs["prior_low_cost_sc_reference"]["best_descendant_id"] == "n0162"

    seeds = {int(row["seed"]) for row in decisions}
    assert len(seeds) >= expected_min_seeds, f"expected >= {expected_min_seeds} seeds, got {len(seeds)}"
    modes = {row["mode"] for row in decisions}
    for mode in (
        "measured_only",
        "map_predict_lambda48",
        "map_predict_lambda32",
        "source_occ_free_over_cost",
    ):
        assert mode in modes, f"required mode missing: {mode}"
    for optional in ("raw_hybrid_over_cost", "source_occ_free_no_cost"):
        assert optional in modes, f"optional diagnostic mode missing: {optional}"

    assert len(decisions) >= expected_min_seeds * 4, "not enough decision rows"
    csv_rows = read_csv(output_dir / "per_seed_mode_decisions.csv")
    assert len(csv_rows) == len(decisions), "CSV/JSON decision row mismatch"

    map48 = [row for row in decisions if row["mode"] == "map_predict_lambda48"]
    assert len(map48) >= expected_min_seeds, "map_predict lambda48 rows missing"
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

    assert value_components, "value component rows missing"
    assert branches, "branch classification rows missing"
    assert low_cost, "low-cost diagnosis rows missing"
    assert lambda_summary["formula_components_logged"] is True
    assert lambda_summary["seed0_measured_reference_reproduced"] is True
    assert lambda_summary["runtime_smoke_readiness"] is False
    assert lambda_summary["rollout_readiness"] is False
    assert "did_lambda48_still_collapse_to_measured" in comparison
    assert "did_lambda48_pick_only_old_bad_branch" in comparison

    answers = final_summary["answers"]
    assert answers["loaded_stage4a65p_frame2_inputs"] is True
    assert answers["no_isaac_no_capture_no_map_predict_rerun"] is True
    assert answers["seed_count"] >= expected_min_seeds
    assert answers["measured_only_reproduced_frame2_reference"] is True
    assert answers["runtime_smoke_readiness"] is False
    assert answers["rollout_readiness"] is False
    assert final_summary["coverage_improvement_claimed"] is False
    return {
        "passed": True,
        "seed_count": len(seeds),
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
        "prediction_ray_blocking",
        "target_ground_truth_planning_scoring",
        "external_source_modified_built",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        assert not bool(safety.get(key)), f"safety flag should be false: {key}"
    for name, item in hash_checks.items():
        assert item["unchanged"] is True, f"input hash changed for {name}: {item['path']}"

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
    parser.add_argument("--expected_min_seeds", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(parse_args())
