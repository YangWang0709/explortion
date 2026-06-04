#!/usr/bin/env python3
"""Validate Stage 4A-6.5y source-gain seed replay outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "prediction_npz_field_inventory.json",
    "prediction_npz_field_inventory.md",
    "source_occ_free_mapping_report.json",
    "source_occ_free_mapping_report.md",
    "source_gain_replay_manifest.jsonl",
    "per_seed_formula_decisions.csv",
    "per_seed_formula_decisions.json",
    "per_seed_formula_decisions.md",
    "per_seed_formula_gain_components.csv",
    "per_seed_formula_gain_components.json",
    "branch_classification_by_formula_seed.csv",
    "branch_classification_by_formula_seed.json",
    "branch_classification_summary_by_formula.json",
    "branch_classification_summary_by_formula.md",
    "margin_summary_by_formula.csv",
    "margin_summary_by_formula.json",
    "margin_summary_by_formula.md",
    "overlap_novelty_summary_by_formula.csv",
    "overlap_novelty_summary_by_formula.json",
    "overlap_novelty_summary_by_formula.md",
    "frontier_local_summary_by_formula.csv",
    "frontier_local_summary_by_formula.json",
    "frontier_local_summary_by_formula.md",
    "formula_source_faithfulness_table.csv",
    "formula_source_faithfulness_table.json",
    "formula_source_faithfulness_table.md",
    "missing_fields_report.json",
    "stage4a65y_source_gain_seed_replay_summary.json",
    "stage4a65y_source_gain_seed_replay_summary.md",
    "recommended_next_faithful_step.md",
    "formula_branch_classification_bar.png",
    "formula_same_as_measured_fraction.png",
    "formula_seed0_sc_basin_fraction.png",
    "formula_avoids_short_local_sc_fraction.png",
    "winner_source_occ_free_by_formula.png",
    "winner_cost_by_formula.png",
    "winner_gain_cost_stack_by_formula.png",
    "root_overlap_fraction_by_formula.png",
    "frontier_local_fraction_by_formula.png",
    "selected_children_by_formula_topdown.png",
    "best_descendants_by_formula_topdown.png",
    "value_vs_source_occ_free_by_formula.png",
    "value_vs_cost_by_formula.png",
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


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    for name in REQUIRED_FILES:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
        if path.suffix == ".png":
            assert_png(path)
    return {"passed": True, "required_files": len(REQUIRED_FILES)}


def test_replay_content(output_dir: Path, expected_min_seeds: int) -> dict[str, Any]:
    decisions = load_json(output_dir / "per_seed_formula_decisions.json")
    class_rows = load_json(output_dir / "branch_classification_by_formula_seed.json")
    summary = load_json(output_dir / "stage4a65y_source_gain_seed_replay_summary.json")
    mapping = load_json(output_dir / "source_occ_free_mapping_report.json")
    faithfulness = load_json(output_dir / "formula_source_faithfulness_table.json")

    seeds = sorted({int(row["seed"]) for row in decisions})
    formulas = sorted({str(row["formula"]) for row in decisions})
    _assert(len(seeds) >= expected_min_seeds, f"expected at least {expected_min_seeds} seeds, got {len(seeds)}")
    for formula in (
        "measured_only",
        "source_occ_free",
        "source_occ_free_thresholded",
        "parent_visible_cleared_source_occ_free",
        "frontier_local_source_occ_free",
    ):
        _assert(formula in formulas, f"missing formula: {formula}")
    _assert(len(class_rows) == len(decisions), "classification rows must cover every decision")
    _assert(mapping["required_fields_present"], "prediction mapping required fields missing")

    by_key = {(int(row["seed"]), str(row["formula"])): row for row in decisions}
    seed0_conf = by_key.get((0, "current_confidence_weighted"))
    _assert(seed0_conf is not None, "missing seed0 current_confidence_weighted row")
    _assert(
        seed0_conf["selected_child_grid"] == [11, 15, 11]
        and seed0_conf["best_descendant_grid"] == [14, 15, 11],
        "seed0 current_confidence_weighted did not reproduce n0127 -> n0162 spatial grid",
    )
    seed0_source = by_key.get((0, "source_occ_free"))
    _assert(seed0_source is not None, "missing seed0 source_occ_free row")
    _assert(seed0_source.get("selected_child_id") is not None, "source_occ_free seed0 did not select a child")

    faith_by_formula = {row["formula"]: row for row in faithfulness}
    source_status = faith_by_formula["source_occ_free"]["source_faithfulness"]
    _assert(
        source_status in {"source-faithful", "source-faithful-approx"},
        f"source_occ_free must be source-faithful or approximate, got {source_status}",
    )
    _assert(
        faith_by_formula["current_confidence_weighted"]["source_faithfulness"] != "source-faithful",
        "confidence weighted must not be marked exact source-faithful",
    )
    _assert(
        faith_by_formula["current_cap25"]["source_faithfulness"] != "source-faithful",
        "cap25 must not be marked exact source-faithful",
    )

    class_summary = summary["branch_classification_summary"]
    for formula in ("source_occ_free", "frontier_local_source_occ_free"):
        _assert(formula in class_summary, f"missing class summary for {formula}")
        _assert(class_summary[formula]["seed_count"] >= expected_min_seeds, f"too few rows for {formula}")
    _assert(summary["recommended_next_faithful_step"] != "rollout", "must not recommend rollout")
    _assert(summary["answers"]["runtime_smoke_readiness"] is False, "runtime smoke should not be ready")
    _assert(summary["answers"]["rollout_readiness"] is False, "rollout should not be ready")
    return {"passed": True, "seed_count": len(seeds), "formulas": formulas, "decision_rows": len(decisions)}


def test_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65y_source_gain_seed_replay_summary.json")
    safety = summary["safety"]
    false_keys = [
        "isaac_startup",
        "new_capture",
        "map_predict_rerun",
        "sscnet_inference",
        "selected_action_execution",
        "rollout",
        "open_ended_loop",
        "training_or_rl",
        "checkpoint_modified",
        "observed_state_modified",
        "prediction_npz_modified",
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
        "replay_content": test_replay_content(output_dir, int(args.expected_min_seeds)),
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
