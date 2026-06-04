#!/usr/bin/env python3
"""Validate Stage 4A-6.5ac saved-frame lambda48 formula smoke outputs."""

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
    "per_seed_mode_decisions.csv",
    "per_seed_mode_decisions.json",
    "per_seed_mode_decisions.md",
    "per_seed_value_components.csv",
    "per_seed_value_components.json",
    "branch_direction_classification.csv",
    "branch_direction_classification.json",
    "branch_direction_summary.md",
    "lambda48_reproduction_summary.json",
    "lambda48_reproduction_summary.md",
    "oracle_vs_map_predict_lambda48.json",
    "oracle_vs_map_predict_lambda48.md",
    "low_cost_artifact_diagnosis.csv",
    "low_cost_artifact_diagnosis.json",
    "low_cost_artifact_diagnosis.md",
    "comparison_to_stage4a65ab.json",
    "comparison_to_stage4a65ab.md",
    "prediction_safety_report.json",
    "prediction_safety_report.md",
    "hash_checks.json",
    "missing_fields_report.json",
    "stage4a65ac_saved_frame_lambda48_formula_summary.json",
    "stage4a65ac_saved_frame_lambda48_formula_summary.md",
    "recommended_next_faithful_step.md",
]

PLOT_FILES = [
    "selected_branches_topdown_lambda48.png",
    "measured_vs_lambda48_topdown.png",
    "oracle_vs_map_predict_lambda48_topdown.png",
    "value_components_lambda48.png",
    "source_occ_free_rank_by_mode.png",
    "low_cost_artifact_by_mode.png",
    "margin_by_mode.png",
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
    decisions = load_json(output_dir / "per_seed_mode_decisions.json")
    value_components = load_json(output_dir / "per_seed_value_components.json")
    branches = load_json(output_dir / "branch_direction_classification.json")
    reproduction = load_json(output_dir / "lambda48_reproduction_summary.json")
    oracle_map = load_json(output_dir / "oracle_vs_map_predict_lambda48.json")
    comparison = load_json(output_dir / "comparison_to_stage4a65ab.json")
    summary = load_json(output_dir / "stage4a65ac_saved_frame_lambda48_formula_summary.json")

    assert Path(manifest["observed_state"]["path"]).is_file(), "loaded observed_state path missing"
    assert Path(manifest["oracle_prediction"]["path"]).is_file(), "loaded oracle prediction path missing"
    assert Path(manifest["map_predict_prediction"]["path"]).is_file(), "loaded map_predict prediction path missing"
    assert Path(manifest["stage4a65ab_best_config_candidates"]["path"]).is_file(), "6.5ab best config path missing"

    assert formula["recommended_formula"] == "gain_exp / cost + 48 * minmax(source_occ_free)"
    assert formula["explicitly_not_recommended"]["over_cost"] == "(gain_exp + source_occ_free) / cost"

    seeds = {int(row["seed"]) for row in decisions}
    assert len(seeds) >= expected_min_seeds, f"expected >= {expected_min_seeds} seeds, got {len(seeds)}"
    modes = {row["mode"] for row in decisions}
    for mode in ("measured_only", "oracle_lambda48", "map_predict_lambda48"):
        assert mode in modes, f"required mode missing: {mode}"
    for optional in ("map_predict_lambda32", "map_predict_over_cost", "oracle_over_cost"):
        assert optional in modes, f"optional diagnostic mode missing: {optional}"

    measured = [row for row in decisions if row["mode"] == "measured_only"]
    assert measured, "measured_only rows missing"
    assert not all(bool(row["hidden_room_selected"]) for row in measured), "measured_only falsely marked hidden-room 5/5"
    assert all(bool(row["measured_frontier_selected"]) for row in measured), "measured_only did not reproduce measured-frontier"

    map48 = [row for row in decisions if row["mode"] == "map_predict_lambda48"]
    oracle48 = [row for row in decisions if row["mode"] == "oracle_lambda48"]
    assert len(map48) >= expected_min_seeds, "map_predict lambda48 rows missing"
    assert len(oracle48) >= expected_min_seeds, "oracle lambda48 rows missing"
    assert all(bool(row["hidden_room_selected"]) for row in map48) or reproduction["status"] == "completed_with_reproduction_warning"
    assert reproduction["map_predict_oracle_lambda48_agreement_fraction"] is not None
    assert reproduction["map_predict_lambda48_low_cost_artifact_fraction"] == 0.0 or reproduction[
        "status"
    ] == "completed_with_reproduction_warning"

    for row in value_components:
        for key in ("base_exp_value", "normalized_sc", "sc_bonus", "final_value"):
            assert key in row, f"missing formula component: {key}"
    assert branches, "branch classification empty"
    assert "agreement_fraction" in oracle_map, "oracle/map agreement missing"
    assert "same_hidden_room_fraction" in comparison, "6.5ab comparison malformed"

    forbidden_lambda48_formula = "(gain_exp + source_occ_free) / cost"
    for row in map48 + oracle48:
        assert row["formula"] != forbidden_lambda48_formula, "lambda48 used over-cost formula"
        assert "minmax(source_occ_free)" in row["formula"], "lambda48 formula missing minmax"

    assert summary["answers"]["loaded_inputs"] is True
    assert summary["answers"]["no_isaac_no_capture_no_map_predict_rerun"] is True
    assert summary["answers"]["over_cost_recommended"] is False
    assert summary["answers"]["readiness"]["runtime_smoke_ready"] is False
    assert summary["answers"]["readiness"]["rollout_ready"] is False
    assert summary["coverage_improvement_claimed"] is False
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
        "leakage",
    ]
    for key in false_keys:
        assert not bool(safety.get(key)), f"safety flag should be false: {key}"
    for item in hash_checks.values():
        assert item["unchanged"] is True, f"input hash changed: {item['path']}"

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
