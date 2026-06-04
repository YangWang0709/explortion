#!/usr/bin/env python3
"""Validate Stage 4A-6.5k offline mini-RRT minimum-edge variant outputs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from offline_mini_rrt_tree import sha256_file


REQUIRED_ROOT_OUTPUTS = [
    "variants_manifest.jsonl",
    "variants_summary.csv",
    "variants_summary.json",
    "variants_comparison.md",
    "selected_child_comparison.csv",
    "rejection_reason_comparison.csv",
    "segment_length_distribution_by_variant.csv",
    "gain_cost_correlation_by_variant.csv",
    "source_min_length_reference.md",
    "recommended_next_faithful_step.md",
    "stage4a65k_min_edge_variant_summary.json",
    "stage4a65k_min_edge_variant_summary.md",
]

REQUIRED_VARIANT_OUTPUTS = [
    "mini_rrt_tree_segments.jsonl",
    "mini_rrt_tree_summary.json",
    "subsequent_best_decision.json",
    "tree_vs_one_step_comparison.json",
    "gain_cost_value_table.csv",
    "sampled_nodes.csv",
    "rejected_samples.csv",
]

ROLLOUT_LIKE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]

EXTERNAL_SOURCE_DIR = Path(
    "/home/ubuntu22/sc_explorer_ws/external_src/active_3d_planning_inspection/mav_active_3d_planning"
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def git_status_short(path: Path) -> str:
    if not path.exists():
        return "missing"
    completed = subprocess.run(
        ["git", "status", "--short"],
        cwd=str(path),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"error: {completed.stderr.strip()}"
    return completed.stdout.strip()


def scan_rollout_like_outputs(output_dir: Path) -> list[str]:
    found: list[str] = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        found.extend(str(path) for path in sorted(output_dir.rglob(pattern)))
    return found


def test_required_outputs(output_dir: Path) -> dict[str, Any]:
    _assert(output_dir.exists(), f"missing output dir: {output_dir}")
    for name in REQUIRED_ROOT_OUTPUTS:
        path = output_dir / name
        _assert(path.exists(), f"missing required output: {path}")
        _assert(path.stat().st_size > 0, f"empty required output: {path}")
    return {"passed": True, "required_outputs": len(REQUIRED_ROOT_OUTPUTS)}


def test_manifest_and_baseline(output_dir: Path) -> dict[str, Any]:
    manifest = load_jsonl(output_dir / "variants_manifest.jsonl")
    _assert(manifest, "manifest is empty")
    by_name = {row.get("variant_name"): row for row in manifest}
    _assert("baseline_allow" in by_name, "baseline_allow missing from manifest")
    baseline_dir = output_dir / "baseline_allow"
    _assert(baseline_dir.exists(), "baseline_allow output dir missing")
    for name in REQUIRED_VARIANT_OUTPUTS:
        path = baseline_dir / name
        _assert(path.exists(), f"baseline missing required file: {path}")
        _assert(path.stat().st_size > 0, f"baseline required file is empty: {path}")
    decision = load_json(baseline_dir / "subsequent_best_decision.json")
    _assert(decision.get("selected_child_id") == "n0140", f"baseline did not reproduce n0140: {decision}")
    return {
        "passed": True,
        "manifest_rows": len(manifest),
        "baseline_selected_child": decision.get("selected_child_id"),
    }


def test_variant_summaries(output_dir: Path) -> dict[str, Any]:
    rows = load_csv(output_dir / "variants_summary.csv")
    _assert(rows, "variants_summary.csv has no rows")
    completed = [row for row in rows if row.get("status") == "completed"]
    _assert(completed, "no completed variants")
    non_baseline_completed = [
        row
        for row in completed
        if row.get("variant_name") != "baseline_allow"
        and (
            row.get("short_edge_policy") in {"reject", "crop"}
            or row.get("min_root_child_length_m") not in {"", "0", "0.0"}
            or row.get("density_radius_m") not in {"", "0", "0.0"}
        )
    ]
    _assert(non_baseline_completed, "no min-edge/crop/density variant completed")
    required_child_fields = [
        "selected_child_id",
        "selected_child_grid",
        "selected_child_world",
        "selected_child_distance_from_root_m",
        "selected_child_segment_length_m",
        "selected_child_local_gain",
        "selected_child_cost",
        "selected_child_value",
        "best_descendant_id",
        "best_descendant_grid",
        "best_descendant_distance_from_root_m",
    ]
    for row in completed:
        for field in required_child_fields:
            _assert(row.get(field) not in {None, ""}, f"missing {field} for {row.get('variant_name')}")
    return {
        "passed": True,
        "completed": len(completed),
        "non_baseline_completed": len(non_baseline_completed),
    }


def test_hashes_and_safety(output_dir: Path) -> dict[str, Any]:
    payload = load_json(output_dir / "stage4a65k_min_edge_variant_summary.json")
    _assert(not payload.get("rollout_like_outputs_created"), "runner reported rollout-like outputs")
    _assert(not payload.get("safety", {}).get("prediction_writeback", False), "prediction_writeback safety failed")
    _assert(
        not payload.get("safety", {}).get("prediction_used_for_traversability_collision", False),
        "prediction used for traversability/collision",
    )
    _assert(not payload.get("safety", {}).get("observed_state_modified", False), "observed_state modified")
    _assert(
        not payload.get("safety", {}).get("external_source_modified_or_built", False),
        "external source modified or built",
    )
    for row in payload.get("variants", []):
        if row.get("status") != "completed":
            continue
        summary = load_json(Path(row["output_dir"]) / "mini_rrt_tree_summary.json")
        observed_path = Path(summary["inputs"]["observed_state"])
        current_hash = sha256_file(observed_path)
        _assert(
            summary["map"]["observed_state_sha256_before"]
            == summary["map"]["observed_state_sha256_after"]
            == current_hash,
            f"observed_state hash mismatch for {row['variant_name']}",
        )
        safety = summary.get("safety", {})
        for key in (
            "isaac_startup",
            "rollout",
            "online_expert_loop",
            "map_predict_rerun",
            "sscnet_inference_or_training",
            "training_rl_ppo_bc_il",
            "checkpoint_modified",
            "prediction_writeback",
            "prediction_used_for_traversability_collision",
            "target_lr_target_hr_ground_truth_scoring",
            "external_source_modified_or_built",
        ):
            _assert(not bool(safety.get(key, False)), f"safety flag true for {row['variant_name']}: {key}")
    rollout_like = scan_rollout_like_outputs(output_dir)
    _assert(not rollout_like, f"rollout-like outputs found: {rollout_like[:5]}")
    external_status = git_status_short(EXTERNAL_SOURCE_DIR)
    _assert(external_status == "", f"external source git status not clean: {external_status}")
    return {"passed": True, "external_source_status": external_status}


def run_tests(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    results = {
        "required_outputs": test_required_outputs(output_dir),
        "manifest_and_baseline": test_manifest_and_baseline(output_dir),
        "variant_summaries": test_variant_summaries(output_dir),
        "hashes_and_safety": test_hashes_and_safety(output_dir),
    }
    payload = {"all_passed": all(item["passed"] for item in results.values()), "tests": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(Path(parse_args().output_dir))
