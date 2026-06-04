#!/usr/bin/env python3
"""Validation for Stage 4A-6.5j mini-RRT gain/sampling diagnosis outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    "segment_length_cost_diagnosis.csv",
    "segment_length_cost_diagnosis.json",
    "selected_node_raycast_audit.json",
    "selected_node_visible_voxels.csv",
    "selected_node_visible_topdown.png",
    "node_novelty_overlap_table.csv",
    "novelty_rerank_summary.json",
    "novelty_rerank_summary.md",
    "sampling_steering_diagnosis.csv",
    "sampling_steering_diagnosis.json",
    "sampling_rejection_summary.md",
    "offline_filter_rerank_table.csv",
    "offline_filter_rerank_summary.json",
    "offline_filter_rerank_summary.md",
    "source_anti_local_mechanisms.md",
    "source_anti_local_hits.csv",
    "stage4a65j_gain_raycast_sampling_summary.json",
    "stage4a65j_gain_raycast_sampling_summary.md",
    "recommended_next_faithful_step.md",
]

ROLLOUT_LIKE_PATTERNS = [
    "step_*.npz",
    "observed_state*.npy",
    "depth_*.npy",
    "rgb_*.png",
    "transitions.jsonl",
    "episode_summary.json",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_tests(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    assert_true(output_dir.is_dir(), f"output dir missing: {output_dir}")
    for name in REQUIRED_FILES:
        path = output_dir / name
        assert_true(path.is_file(), f"required output missing: {path}")
        assert_true(path.stat().st_size > 0, f"required output is empty: {path}")

    summary = load_json(output_dir / "stage4a65j_gain_raycast_sampling_summary.json")
    audit = load_json(output_dir / "selected_node_raycast_audit.json")
    segment = load_json(output_dir / "segment_length_cost_diagnosis.json")
    novelty_rows = read_csv_rows(output_dir / "node_novelty_overlap_table.csv")
    filter_rows = read_csv_rows(output_dir / "offline_filter_rerank_table.csv")
    source_hits = read_csv_rows(output_dir / "source_anti_local_hits.csv")

    assert_true("counts" in audit, "selected audit missing counts")
    assert_true("gain_match" in audit["counts"], "selected gain recompute status missing")
    assert_true("distributions" in segment, "segment diagnostics missing distributions")
    assert_true("segment_length_m" in segment["distributions"], "segment length diagnostics missing")
    assert_true(len(novelty_rows) > 0, "novelty table empty")
    assert_true(
        any(row.get("segment_id") == audit["selected_node"]["segment_id"] for row in novelty_rows),
        "novelty table missing selected node",
    )
    assert_true(len(filter_rows) > 0, "offline filter rerank table empty")
    assert_true(len(source_hits) > 0, "source anti-local hit table empty")

    safety = summary.get("safety", {})
    assert_true(safety.get("observed_state_modified") is False, "observed_state hash changed")
    assert_true(
        safety.get("external_source_modified_or_built") is False,
        "external source git status changed during diagnosis",
    )
    for key in (
        "isaac_startup",
        "rollout",
        "online_expert_loop",
        "map_predict_rerun",
        "sscnet_inference_or_training",
        "training_rl_ppo_bc_il",
        "checkpoint_modified",
        "prediction_writeback",
        "prediction_used_for_collision_traversability",
        "target_lr_target_hr_ground_truth_scoring",
    ):
        assert_true(safety.get(key) is False, f"safety flag is not false: {key}")

    rollout_like_found: list[str] = []
    for pattern in ROLLOUT_LIKE_PATTERNS:
        rollout_like_found.extend(str(path) for path in output_dir.glob(pattern))
    assert_true(not rollout_like_found, f"rollout-like outputs found: {rollout_like_found}")

    result = {
        "all_passed": True,
        "output_dir": str(output_dir),
        "required_files": len(REQUIRED_FILES),
        "selected_node": audit["selected_node"]["segment_id"],
        "gain_match": audit["counts"]["gain_match"],
        "novelty_rows": len(novelty_rows),
        "filter_rows": len(filter_rows),
        "source_hits": len(source_hits),
        "observed_state_hash_unchanged": True,
        "external_source_status_unchanged": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_tests(Path(parse_args().output_dir))
