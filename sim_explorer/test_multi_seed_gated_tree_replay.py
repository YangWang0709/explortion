#!/usr/bin/env python3
"""Validate Stage 4A-6.5v multi-seed gated-tree offline replay outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "multi_seed_replay_manifest.jsonl",
    "per_seed_formula_decisions.csv",
    "per_seed_formula_decisions.json",
    "per_seed_formula_decisions.md",
    "per_seed_rank_margin.csv",
    "per_seed_rank_margin.json",
    "branch_classification_by_seed.csv",
    "branch_classification_by_seed.json",
    "branch_classification_summary.json",
    "branch_classification_summary.md",
    "confidence_vs_cap25_agreement.csv",
    "confidence_vs_cap25_agreement.json",
    "spatial_basin_summary.csv",
    "spatial_basin_summary.json",
    "spatial_basin_summary.md",
    "missing_fields_report.json",
    "stage4a65v_multi_seed_replay_summary.json",
    "stage4a65v_multi_seed_replay_summary.md",
    "recommended_next_faithful_step.md",
    "selected_children_by_seed_topdown.png",
    "best_descendants_by_seed_topdown.png",
    "seed_classification_bar.png",
    "margin_distribution_by_formula.png",
    "selected_delta_to_seed0_sc.png",
    "confidence_vs_cap25_selected_delta.png",
    "value_vs_effective_sc_by_seed.png",
    "value_vs_cost_by_seed.png",
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
    class_rows = load_json(output_dir / "branch_classification_by_seed.json")
    summary = load_json(output_dir / "stage4a65v_multi_seed_replay_summary.json")
    agreement = load_json(output_dir / "confidence_vs_cap25_agreement.json")

    seeds = sorted({int(row["seed"]) for row in decisions})
    formulas = sorted({str(row["formula"]) for row in decisions})
    _assert(len(seeds) >= expected_min_seeds, f"expected at least {expected_min_seeds} seeds, got {len(seeds)}")
    for formula in ("measured_only", "confidence_weighted", "cap25"):
        _assert(formula in formulas, f"missing formula: {formula}")
    _assert(len(class_rows) == len(decisions), "classification rows must cover every decision")

    by_key = {(int(row["seed"]), str(row["formula"])): row for row in decisions}
    _assert((0, "confidence_weighted") in by_key, "missing seed0 confidence row")
    _assert((1, "confidence_weighted") in by_key, "missing seed1 confidence row")

    ref_checks = summary["reference_checks"]
    seed0_ref = ref_checks["seed0_confidence_vs_stage4a65s"]
    seed1_ref = ref_checks["seed1_confidence_vs_stage4a65t"]
    _assert(seed0_ref["exact_grid_match"] or seed0_ref["spatial_match"], "seed0 replay does not match Stage 4A-6.5s")
    _assert(seed1_ref["exact_grid_match"] or seed1_ref["spatial_match"], "seed1 replay does not match Stage 4A-6.5t")

    class_summary = summary["branch_classification_summary"]
    for formula in ("confidence_weighted", "cap25"):
        _assert(formula in class_summary, f"missing class summary: {formula}")
        _assert(class_summary[formula]["seed_count"] >= expected_min_seeds, f"too few class rows for {formula}")
    _assert("confidence_vs_cap25_exact_grid_agreement_rate" in agreement, "missing confidence/cap25 agreement")
    _assert(summary["recommended_next_faithful_step"] != "rollout", "must not recommend rollout")
    return {
        "passed": True,
        "seed_count": len(seeds),
        "formulas": formulas,
        "decision_rows": len(decisions),
    }


def test_safety(output_dir: Path) -> dict[str, Any]:
    summary = load_json(output_dir / "stage4a65v_multi_seed_replay_summary.json")
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
        "source_modified_built",
        "coverage_improvement_claim",
    ]
    for key in false_keys:
        _assert(not bool(safety.get(key)), f"safety flag should be false: {key}")
    _assert(safety["observed_state_sha256_before"] == safety["observed_state_sha256_after"], "observed_state hash changed")
    _assert(safety["prediction_npz_sha256_before"] == safety["prediction_npz_sha256_after"], "prediction hash changed")
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
