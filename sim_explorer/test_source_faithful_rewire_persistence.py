#!/usr/bin/env python3
"""Validate Stage 4A-6.5w source-faithful rewire persistence outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "source_rewire_evidence.json",
    "source_rewire_evidence.md",
    "source_density_evidence.json",
    "source_density_evidence.md",
    "source_yaw_evidence.json",
    "source_yaw_evidence.md",
    "source_persistence_design.md",
    "rewire_configs_manifest.jsonl",
    "per_config_seed_formula_decisions.csv",
    "per_config_seed_formula_decisions.json",
    "per_config_seed_formula_decisions.md",
    "per_config_summary.csv",
    "per_config_summary.json",
    "per_config_summary.md",
    "branch_classification_by_config_seed.csv",
    "branch_classification_by_config_seed.json",
    "branch_classification_summary_by_config.json",
    "branch_classification_summary_by_config.md",
    "preservation_summary_by_config.csv",
    "preservation_summary_by_config.json",
    "preservation_summary_by_config.md",
    "confidence_vs_cap25_agreement_by_config.csv",
    "confidence_vs_cap25_agreement_by_config.json",
    "spatial_basin_summary_by_config.csv",
    "spatial_basin_summary_by_config.json",
    "spatial_basin_summary_by_config.md",
    "margin_summary_by_config.csv",
    "margin_summary_by_config.json",
    "margin_summary_by_config.md",
    "compute_time_summary.csv",
    "missing_fields_report.json",
    "stage4a65w_source_faithful_rewire_summary.json",
    "stage4a65w_source_faithful_rewire_summary.md",
    "recommended_next_faithful_step.md",
    "reinsert_attempts.csv",
    "reinsert_summary.json",
    "reinsert_summary.md",
    "spatial_seed0_sc_basin_fraction_by_config.png",
    "same_as_measured_fraction_by_config.png",
    "preserved_nodes_fraction_by_config.png",
    "confidence_cap25_agreement_by_config.png",
    "selected_children_by_config_topdown.png",
    "best_descendants_by_config_topdown.png",
    "margin_distribution_by_config.png",
    "selected_delta_to_seed0_sc_by_config.png",
    "value_vs_effective_sc_by_config.png",
    "value_vs_cost_by_config.png",
    "preserved_vs_newly_expanded_winners.png",
    "compute_time_by_config.png",
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


def same_grid(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return False
    return [int(round(float(v))) for v in a] == [int(round(float(v))) for v in b]


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.is_dir(), f"missing output dir: {output_dir}")
    for name in REQUIRED_FILES:
        path = output_dir / name
        _assert(path.is_file(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
        if path.suffix == ".png":
            assert_png(path)
    return {"passed": True, "required_files": len(REQUIRED_FILES)}


def test_replay_content(output_dir: Path, expected_min_seeds: int, expected_min_configs: int) -> dict[str, Any]:
    decisions = load_json(output_dir / "per_config_seed_formula_decisions.json")
    class_rows = load_json(output_dir / "branch_classification_by_config_seed.json")
    summary = load_json(output_dir / "stage4a65w_source_faithful_rewire_summary.json")
    preservation = load_json(output_dir / "preservation_summary_by_config.json")
    agreement = load_json(output_dir / "confidence_vs_cap25_agreement_by_config.json")

    configs = sorted({str(row["config"]) for row in decisions})
    seeds = sorted({int(row["seed"]) for row in decisions if str(row["config"]) == "fresh_random_256_baseline"})
    formulas = sorted({str(row["formula"]) for row in decisions})
    _assert(len(seeds) >= expected_min_seeds, f"expected at least {expected_min_seeds} fresh seeds, got {len(seeds)}")
    _assert(len(configs) >= expected_min_configs, f"expected at least {expected_min_configs} configs, got {len(configs)}")
    for formula in ("measured_only", "confidence_weighted", "cap25"):
        _assert(formula in formulas, f"missing formula: {formula}")
    _assert(len(class_rows) == len(decisions), "classification rows must cover every decision")

    by_key = {(str(row["config"]), int(row["seed"]), str(row["formula"])): row for row in decisions}
    seed0 = by_key.get(("fresh_random_256_baseline", 0, "confidence_weighted"))
    _assert(seed0 is not None, "missing seed0 fresh confidence row")
    _assert(same_grid(seed0.get("selected_child_grid"), [11, 15, 11]), "seed0 fresh selected child mismatch")
    _assert(same_grid(seed0.get("best_descendant_grid"), [14, 15, 11]), "seed0 fresh best descendant mismatch")
    if expected_min_seeds >= 2:
        seed1 = by_key.get(("fresh_random_256_baseline", 1, "confidence_weighted"))
        _assert(seed1 is not None, "missing seed1 fresh confidence row")
        _assert(same_grid(seed1.get("selected_child_grid"), [12, 16, 11]), "seed1 fresh selected child mismatch")
        _assert(same_grid(seed1.get("best_descendant_grid"), [12, 19, 11]), "seed1 fresh best descendant mismatch")

    _assert(summary["fresh_baseline_reproduces_stage4a65v"], "fresh baseline did not reproduce Stage 4A-6.5v")
    _assert(any(str(row["config"]).startswith("persistent_rewire") for row in preservation), "missing persistent preservation summary")
    for row in preservation:
        _assert("mean_nodes_preserved" in row, "preserved node counts missing")
        _assert("mean_nodes_pruned" in row, "pruned node counts missing")
    _assert("summary" in agreement and agreement["summary"], "missing confidence/cap25 agreement summary")
    _assert(summary["recommended_next_faithful_step"] != "rollout", "must not recommend rollout")
    return {
        "passed": True,
        "config_count": len(configs),
        "fresh_seed_count": len(seeds),
        "formulas": formulas,
        "decision_rows": len(decisions),
    }


def test_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65w_source_faithful_rewire_summary.json")
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
        "frame1_observed_state_modified",
        "frame2_observed_state_modified",
        "frame1_prediction_npz_modified",
        "frame2_prediction_npz_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "prediction_ray_blocking",
        "target_ground_truth_scoring",
        "source_modified_built",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        _assert(not bool(safety.get(key)), f"safety flag should be false: {key}")
    _assert(
        safety["frame1_observed_state_sha256_before"] == safety["frame1_observed_state_sha256_after"],
        "frame1 observed_state hash changed",
    )
    _assert(
        safety["frame2_observed_state_sha256_before"] == safety["frame2_observed_state_sha256_after"],
        "frame2 observed_state hash changed",
    )
    _assert(
        safety["frame1_prediction_npz_sha256_before"] == safety["frame1_prediction_npz_sha256_after"],
        "frame1 prediction hash changed",
    )
    _assert(
        safety["frame2_prediction_npz_sha256_before"] == safety["frame2_prediction_npz_sha256_after"],
        "frame2 prediction hash changed",
    )
    if safety.get("checkpoint_sha256_before") is not None:
        _assert(safety["checkpoint_sha256_before"] == safety["checkpoint_sha256_after"], "checkpoint hash changed")
    _assert(not safety.get("prohibited_artifacts_in_output"), "summary recorded prohibited artifacts")
    for pattern in FORBIDDEN_PATTERNS:
        matches = sorted(output_dir.rglob(pattern))
        _assert(not matches, f"forbidden output artifacts for pattern {pattern}: {matches[:5]}")
    return {"passed": True}


def run_tests(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "replay_content": test_replay_content(
            output_dir,
            int(args.expected_min_seeds),
            int(args.expected_min_configs),
        ),
        "safety": test_safety(output_dir),
    }
    print(json.dumps(results, indent=2, sort_keys=True))
    return results


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--expected_min_seeds", type=int, default=10)
    parser.add_argument("--expected_min_configs", type=int, default=3)
    return parser


def main() -> None:
    run_tests(build_argparser().parse_args())


if __name__ == "__main__":
    main()
